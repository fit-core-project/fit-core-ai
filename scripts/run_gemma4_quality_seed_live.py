from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LOCAL_SQLITE_DB_PATH = Path(__file__).resolve().parents[1] / "fit_core.sqlite"
os.environ.setdefault("DATABASE_URL", f"sqlite:///{LOCAL_SQLITE_DB_PATH}")
os.environ.setdefault("FITCORE_SQLITE_DB_PATH", str(LOCAL_SQLITE_DB_PATH))

from database import SessionLocal
from engines.db_queries import get_recent_sets, get_user_profile_context
from engines.routine_pipeline import generate_smart_routine
from engines.schemas import PainAreaEntry, RoutineDraftResponse, RoutineRequest
from scripts import run_gemma4_quality_seed as seed_runner


DEFAULT_SEED_PATH = Path("tests/fixtures/gemma4_routine_quality_eval_seed_v2.jsonl")
DEFAULT_OUTPUT_DIR = Path("tests/evaluation/.artifacts/gemma4-quality-seed-live")
DEFAULT_SCENARIO_TIMEOUT_SEC = 360


class ScenarioTimeoutError(TimeoutError):
    """Raised when one quality scenario exceeds the live runner time budget."""


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("gemma4-live-%Y%m%dT%H%M%SZ")


def sanitize_run_id(value: str | None) -> str:
    raw = value or utc_run_id()
    sanitized = re.sub(r"[^A-Za-z0-9_.:-]", "-", raw).strip("-")
    return sanitized or utc_run_id()


def parse_scenario_ids(values: list[str] | None) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        for raw_item in str(value).split(","):
            scenario_id = raw_item.strip()
            if not scenario_id or scenario_id in seen:
                continue
            ids.append(scenario_id)
            seen.add(scenario_id)
    return ids


def filter_seed_rows(
    rows: list[dict[str, Any]],
    scenario_ids: list[str] | None = None,
    max_scenarios: int | None = None,
) -> list[dict[str, Any]]:
    filtered = rows
    if scenario_ids:
        requested = set(scenario_ids)
        found = {str(row["scenarioId"]) for row in rows if str(row["scenarioId"]) in requested}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"scenarioId not found in seed: {', '.join(missing)}")
        filtered = [row for row in rows if str(row["scenarioId"]) in requested]
    if max_scenarios is not None:
        filtered = filtered[:max_scenarios]
    return filtered


def apply_local_llm_defaults() -> None:
    os.environ.setdefault("APP_ENV", "development")
    os.environ.setdefault("LLM_PROVIDER", "local")
    os.environ.setdefault("LOCAL_LLM_MODEL", "gemma4:latest")
    os.environ.setdefault("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    os.environ.setdefault("ENABLE_LOCAL_RAW_JSON_INVOKE", "true")
    os.environ.setdefault("LOCAL_LLM_NUM_PREDICT", "4096")
    os.environ.setdefault("LOCAL_LLM_NUM_CTX", "8192")


@contextlib.contextmanager
def scenario_timeout(seconds: float | None):
    if not seconds or seconds <= 0:
        yield
        return

    def handle_timeout(signum: int, frame: Any) -> None:  # noqa: ARG001 - signal callback shape
        raise ScenarioTimeoutError(f"scenario timed out after {seconds}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def pain_entries(value: Any) -> list[PainAreaEntry]:
    entries: list[PainAreaEntry] = []
    for item in value or []:
        if isinstance(item, str):
            entries.append(PainAreaEntry(body_part=item))
        elif isinstance(item, dict):
            body_part = item.get("bodyPart") or item.get("body_part") or item.get("area")
            if body_part:
                entries.append(
                    PainAreaEntry(
                        body_part=str(body_part),
                        side=item.get("side"),
                        severity=item.get("severity"),
                    )
                )
    return entries


def doms_map(value: Any) -> dict[str, int]:
    if isinstance(value, dict):
        return {str(key): int(val) for key, val in value.items() if str(key).strip()}
    result: dict[str, int] = {}
    level_map = {
        "mild": 1,
        "slight": 1,
        "moderate": 2,
        "severe": 3,
        "injury": 3,
    }
    for item in value or []:
        if not isinstance(item, dict):
            continue
        body_part = item.get("bodyPart") or item.get("body_part") or item.get("area")
        if not body_part:
            continue
        raw_level = item.get("level", 1)
        if isinstance(raw_level, str):
            level = level_map.get(raw_level.strip().lower(), 1)
        else:
            level = int(raw_level or 1)
        result[str(body_part)] = level
    return result


def condition_policies(input_payload: dict[str, Any]) -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    for item in input_payload.get("surgeryHistory") or []:
        if isinstance(item, dict):
            status = item.get("status")
            policies.append({
                "policyType": "surgeryHistory",
                "bodyPart": item.get("bodyPart"),
                "status": status,
                "professionalClearance": False if status == "needs_clearance" else item.get("professionalClearance"),
            })
    return policies


def anthropometry_signals(input_payload: dict[str, Any]) -> dict[str, Any]:
    signals = input_payload.get("anthropometrySignals") or []
    if isinstance(signals, dict):
        return signals
    if isinstance(signals, list):
        return {"signals": [str(item) for item in signals if str(item).strip()]}
    return {}


def request_from_seed(row: dict[str, Any]) -> RoutineRequest:
    payload = row.get("input", {})
    return RoutineRequest(
        user_id=f"gemma4-eval-{row['scenarioId']}",
        target_split_label=payload.get("targetSplitLabel"),
        target_muscles=list(payload.get("targetMuscles") or []),
        readiness_level=payload.get("readinessLevel") or "normal",
        time_available_min=int(payload.get("timeAvailableMin") or 60),
        pain_areas=pain_entries(payload.get("currentPainAreas") or payload.get("painAreas")),
        doms_data=doms_map(payload.get("currentDoms") or payload.get("doms")),
        equipment=list(payload.get("unavailableEquipment") or []),
        goal=payload.get("goal"),
        experience_level=payload.get("experienceLevel"),
        user_note=f"quality-seed {row['scenarioId']}",
        mobility_limits=list(payload.get("mobilityLimits") or []),
        condition_policies=condition_policies(payload),
        anthropometry_signals=anthropometry_signals(payload),
    )


def text_blob(response: RoutineDraftResponse) -> str:
    parts: list[str] = []
    for block in response.routine_blocks:
        parts.extend([
            block.exercise_id,
            block.exercise_name,
            block.exercise_rationale,
            block.equipment_type or "",
            " ".join(block.primary_muscles or []),
            " ".join(block.reasons or []),
            " ".join(item.rule_code or "" for item in (block.boosts or [])),
            " ".join(item.rule_code or "" for item in (block.penalties or [])),
            " ".join(item.constraint_code or "" for item in (block.boosts or [])),
            " ".join(item.constraint_code or "" for item in (block.penalties or [])),
            " ".join(item.profile_signal_code or "" for item in (block.boosts or [])),
            " ".join(item.profile_signal_code or "" for item in (block.penalties or [])),
            " ".join(item.reason for item in (block.boosts or []) if item.reason),
            " ".join(item.reason for item in (block.penalties or []) if item.reason),
        ])
        for sub in block.substitution_candidates:
            parts.extend([sub.exercise_id, sub.exercise_name, sub.reason])
    parts.extend(response.warnings or [])
    parts.extend(response.rationale_summary or [])
    return " ".join(str(item).lower() for item in parts if item)


def selected_ids(response: RoutineDraftResponse) -> list[str]:
    return [block.exercise_id for block in response.routine_blocks]


def derived_quality_tags(row: dict[str, Any], response: RoutineDraftResponse) -> list[str]:
    tags: list[str] = []
    text = text_blob(response)
    time_available_min = int(row.get("input", {}).get("timeAvailableMin") or 0)
    first = response.routine_blocks[0] if response.routine_blocks else None
    first_text = " ".join([
        getattr(first, "exercise_id", ""),
        getattr(first, "exercise_name", ""),
        getattr(first, "exercise_rationale", ""),
    ]).lower() if first else ""
    ids = " ".join(selected_ids(response)).lower()

    if ("deadlift" in first_text or "데드" in first_text) and "lower-back" in str(row.get("input", {})):
        tags.append("barbell_deadlift_as_main")
    if ("barbell_row" in ids or "바벨 로우" in text or "barbell row" in text) and "lower-back" in str(row.get("input", {})):
        tags.append("barbell_row_as_main")
    if "overhead_press" in first_text or "overhead press" in first_text or "오버헤드" in first_text:
        tags.append("overhead_press_as_first_main")
    if any(token in ids for token in ["barbell", "machine", "cable"]):
        unavailable = {str(item).upper() for item in row.get("input", {}).get("unavailableEquipment", [])}
        if "BARBELL" in unavailable and "barbell" in ids:
            tags.append("barbell_or_machine_selected")
        if "MACHINE" in unavailable and "machine" in ids:
            tags.append("barbell_or_machine_selected")
        if "CABLE" in unavailable and "cable" in ids:
            tags.append("cable_selected")
    if len(response.routine_blocks) >= 6:
        tags.append("six_exercise_plan")
    if time_available_min <= 30 and len(response.routine_blocks) > 3:
        tags.append("short_time_over_3_exercises")
    if "guaranteed safe" in text or "pain-free" in text or "통증 없이" in text:
        tags.append("pain_free_claim")
    if "rehab" in text or "treat" in text or "치료" in text or "재활" in text:
        tags.append("rehab_treatment_claim")
    if row["scenarioId"] == "all-major-pain-extreme-001" and response.generation_status == "success":
        if len(response.routine_blocks) >= 4 and response.total_estimated_time >= 30:
            tags.append("normal_intensity_fullbody")
    if row["scenarioId"] == "low-readiness-fullbody-001":
        total_sets = sum(len(block.prescription or []) for block in response.routine_blocks)
        if total_sets >= 15:
            tags.append("high_volume_compound_stack")
    if row["scenarioId"] == "profile-injury-plus-doms-001":
        if "biceps" not in text and "doms" not in text and "근육통" not in text:
            tags.append("ignores_biceps_doms")
    return sorted(set(tags))


def _contains_any(text: str, tokens: list[str]) -> bool:
    return any(token.lower() in text for token in tokens)


def _total_working_sets(response: RoutineDraftResponse) -> int:
    return sum(len(block.prescription or []) for block in response.routine_blocks)


def _selected_id_text(response: RoutineDraftResponse) -> str:
    return " ".join(selected_ids(response)).lower()


def _has_equipment(response: RoutineDraftResponse, equipment: str) -> bool:
    token = equipment.lower()
    return any(token in str(block.equipment_type or "").lower() for block in response.routine_blocks)


def derived_preferred_tags(row: dict[str, Any], response: RoutineDraftResponse) -> list[str]:
    tags: list[str] = []
    scenario_id = str(row.get("scenarioId") or "")
    text = text_blob(response)
    ids = _selected_id_text(response)

    if scenario_id == "lower-back-legs-001":
        if "leg_press" in ids or "레그프레스" in text:
            tags.append("leg_press")
        if "leg_curl" in ids or "레그 컬" in text:
            tags.append("leg_curl")
        if "hip_thrust" in ids or "힙 쓰러스트" in text:
            tags.append("hip_thrust")
        if _contains_any(text, ["supported", "chest-supported", "seated", "등받이", "지지"]):
            tags.append("supported")
        if _has_equipment(response, "machine") or "machine" in ids or "머신" in text:
            tags.append("machine")

    elif scenario_id == "limited-ankle-legs-001":
        if "leg_press" in ids or "레그프레스" in text:
            tags.append("leg_press")
        if _contains_any(ids + " " + text, ["leg_extension", "machine_quad", "레그 익스텐션"]):
            tags.append("machine_quad")
        if "box_squat" in ids or "박스 스쿼트" in text:
            tags.append("box_squat")
        if _contains_any(text, ["ankle", "dorsiflexion", "limited_ankle_dorsiflexion", "발목", "배굴"]):
            tags.append("ankle_mobility_reason")

    elif scenario_id == "shoulder-push-001":
        if _contains_any(ids + " " + text, ["machine_chest_press", "머신 체스트 프레스"]):
            tags.append("machine_chest_press")
        if _contains_any(ids + " " + text, ["cable_press", "케이블 프레스"]):
            tags.append("cable_press")
        if _contains_any(ids + " " + text, ["neutral_grip", "뉴트럴", "중립 그립"]):
            tags.append("neutral_grip_press")
        if _contains_any(text, ["shoulder", "front-deltoids", "어깨", "전면 삼각근"]):
            tags.append("shoulder_context_reason")

    elif scenario_id == "low-readiness-fullbody-001":
        if _total_working_sets(response) <= 12:
            tags.append("lower_volume")
        if any(60 <= (preset.target_rest_sec or 0) <= 120 for block in response.routine_blocks for preset in block.prescription):
            tags.append("moderate_rest")
        if _has_equipment(response, "machine") or _has_equipment(response, "dumbbell") or _has_equipment(response, "bodyweight"):
            tags.append("machine_or_stable_options")
        if _contains_any(text, ["readiness", "컨디션", "low", "낮"]):
            tags.append("readiness_reason")

    elif scenario_id == "home-equipment-001":
        if _has_equipment(response, "dumbbell") or "dumbbell" in ids:
            tags.append("dumbbell_when_available")
        if _has_equipment(response, "bodyweight") or "pushup" in ids or "plank" in ids:
            tags.append("bodyweight_when_sensible")
        if _contains_any(text, ["equipment", "장비", "alternative", "대체", "mapped_substitute"]):
            tags.append("equipment_alternative_reason")

    elif scenario_id == "short-time-push-001":
        if 2 <= len(response.routine_blocks) <= 3:
            tags.append("two_to_three_exercises")
        if response.routine_blocks:
            tags.append("main_compound_then_accessory")
        if _contains_any(text, ["short", "time", "20", "짧", "시간"]):
            tags.append("short_time_reason")

    elif scenario_id == "beginner-lower-skill-001":
        if _has_equipment(response, "machine") or _contains_any(text, ["stable", "안정", "bodyweight"]):
            tags.append("machine_or_stable_option")
        if _contains_any(text, ["technique", "difficulty", "난이도", "기술", "failure_penalty"]):
            tags.append("technique_risk_reason")
        if _contains_any(text, ["progression", "regression", "단계", "점진"]):
            tags.append("progression_path")

    elif scenario_id == "post-surgery-caution-001":
        if _contains_any(text, ["medical", "clearance", "의료진", "허가", "전문가"]):
            tags.append("medical_clearance_notice")
        if _contains_any(text, ["alternative", "substitution", "대체", "후보"]):
            tags.append("safe_alternative_candidate")
        if _contains_any(text, ["neutral", "supported", "중립", "지지", "등받이"]):
            tags.append("neutral_grip_or_supported_press")

    elif scenario_id == "all-major-pain-extreme-001":
        if response.generation_status == "failed" or len(response.routine_blocks) <= 2:
            tags.append("failed_or_minimal_safe")
        if response.status_reason_code == "emptyCandidate" or _contains_any(text, ["low risk", "저위험", "emptycandidate"]):
            tags.append("empty_or_low_risk_candidate_reason")
        if response.warnings or response.generation_status == "failed":
            tags.append("clear_warning")

    elif scenario_id == "high-risk-high-effect-substitution-001":
        if _contains_any(text, ["substitution", "alternative", "대체", "risk", "위험", "lumbar_load"]):
            tags.append("risk_adjusted_substitution")
        if _contains_any(text, ["effect", "risk", "효과", "위험", "부하"]):
            tags.append("effect_vs_risk_reason")
        if _contains_any(text, ["lumbar", "허리", "lower-back", "요추"]):
            tags.append("lower_lumbar_alternative")

    elif scenario_id == "long-femur-squat-setup-001":
        if _contains_any(text, ["femur", "anthropometry", "대퇴", "체형", "세팅"]):
            tags.append("setup_sensitivity_reason")
        if _contains_any(ids + " " + text, ["leg_press", "front_foot", "레그프레스", "전족"]):
            tags.append("leg_press_or_front_foot_elevated_option")
        if _contains_any(text, ["squat", "variation", "스쿼트", "변형"]):
            tags.append("squat_variation_reason")

    elif scenario_id == "profile-injury-plus-doms-001":
        if _contains_any(ids + " " + text, ["supported_row", "chest-supported", "서포티드"]):
            tags.append("supported_row")
        if _contains_any(ids + " " + text, ["lat_pulldown", "machine_pull", "랫풀다운", "머신"]):
            tags.append("lat_pulldown_or_machine_pull")
        if _contains_any(text, ["doms", "근육통", "biceps", "doms_penalty"]):
            tags.append("doms_volume_reason")
        if _contains_any(text, ["lower-back", "lumbar", "허리", "요추"]):
            tags.append("lower_back_reason")

    elif scenario_id == "knee-pain-limited-ankle-beginner-001":
        if _contains_any(text, ["knee_shear", "knees", "무릎"]):
            tags.append("low_knee_shear_option")
        if _has_equipment(response, "machine") or _contains_any(text, ["stable", "안정", "bodyweight"]):
            tags.append("machine_or_stable_option")
        if _contains_any(text, ["ankle", "dorsiflexion", "limited_ankle_dorsiflexion", "발목", "배굴"]):
            tags.append("ankle_mobility_reason")
        if _contains_any(text, ["technique", "difficulty", "난이도", "기술", "failure_penalty"]):
            tags.append("technique_risk_reason")

    elif scenario_id == "wrist-limited-push-001":
        if _contains_any(text, ["wrist", "wrist_extension", "forearm", "손목", "전완"]):
            tags.append("wrist_context_reason")
        if _contains_any(text, ["neutral_grip", "뉴트럴", "중립 그립"]):
            tags.append("neutral_grip_press")
        if _contains_any(ids + " " + text, ["machine_chest_press", "머신 체스트 프레스"]):
            tags.append("machine_chest_press")
        if _has_equipment(response, "dumbbell") or "dumbbell" in ids:
            tags.append("dumbbell_when_available")

    elif scenario_id == "low-readiness-home-short-fullbody-001":
        if 2 <= len(response.routine_blocks) <= 3:
            tags.append("two_to_three_exercises")
        if _total_working_sets(response) <= 10:
            tags.append("lower_volume")
        if _has_equipment(response, "dumbbell") or "dumbbell" in ids:
            tags.append("dumbbell_when_available")
        if _has_equipment(response, "bodyweight") or "pushup" in ids or "plank" in ids:
            tags.append("bodyweight_when_sensible")
        if _contains_any(text, ["readiness", "컨디션", "low", "낮"]):
            tags.append("readiness_reason")

    elif scenario_id == "shoulder-surgery-home-push-001":
        if _contains_any(text, ["medical", "clearance", "의료진", "허가", "전문가"]):
            tags.append("medical_clearance_notice")
        if _contains_any(text, ["alternative", "substitution", "대체", "후보"]):
            tags.append("safe_alternative_candidate")
        if _contains_any(text, ["shoulder", "front-deltoids", "어깨", "전면 삼각근"]):
            tags.append("shoulder_context_reason")
        if _has_equipment(response, "bodyweight") or "pushup" in ids or "plank" in ids:
            tags.append("bodyweight_when_sensible")

    elif scenario_id == "upper-back-doms-lower-back-pull-short-001":
        if _contains_any(ids + " " + text, ["supported_row", "chest-supported", "서포티드"]):
            tags.append("supported_row")
        if _contains_any(ids + " " + text, ["lat_pulldown", "machine_pull", "랫풀다운", "머신"]):
            tags.append("lat_pulldown_or_machine_pull")
        if _contains_any(text, ["doms", "근육통", "biceps", "doms_penalty"]):
            tags.append("doms_volume_reason")
        if _contains_any(text, ["lower-back", "lumbar", "허리", "요추"]):
            tags.append("lower_back_reason")
        if 2 <= len(response.routine_blocks) <= 3:
            tags.append("two_to_three_exercises")

    elif scenario_id == "advanced-high-readiness-effect-safe-001":
        if _contains_any(text, ["substitution", "alternative", "대체", "risk", "위험", "lumbar_load"]):
            tags.append("risk_adjusted_substitution")
        if _contains_any(text, ["effect", "risk", "효과", "위험", "부하"]):
            tags.append("effect_vs_risk_reason")
        if _contains_any(text, ["lumbar", "허리", "lower-back", "요추"]):
            tags.append("lower_lumbar_alternative")
        if _contains_any(text, ["readiness", "컨디션", "high", "높"]):
            tags.append("readiness_reason")

    return sorted(set(tags))


def summarize_rationale(response: RoutineDraftResponse) -> str:
    parts: list[str] = []
    parts.extend(response.rationale_summary or [])
    for block in response.routine_blocks[:4]:
        if block.exercise_rationale:
            parts.append(f"{block.exercise_id}: {block.exercise_rationale}")
    return " | ".join(parts)[:1600]


def summarize_score_evidence(response: RoutineDraftResponse) -> str:
    parts: list[str] = []
    for block in response.routine_blocks[:4]:
        boost_rules = [
            item.rule_code
            for item in (block.boosts or [])[:4]
            if item.rule_code
        ]
        penalty_rules = [
            item.rule_code
            for item in (block.penalties or [])[:4]
            if item.rule_code
        ]
        visible_reasons = list(block.reasons or [])[:3]
        if boost_rules or penalty_rules or visible_reasons:
            parts.append(
                " ".join([
                    f"{block.exercise_id}:",
                    f"reasons={','.join(visible_reasons) or 'none'}",
                    f"boosts={','.join(boost_rules) or 'none'}",
                    f"penalties={','.join(penalty_rules) or 'none'}",
                ])
            )
    return " | ".join(parts)[:1600]


def result_row(row: dict[str, Any], response: RoutineDraftResponse, elapsed_ms: int) -> dict[str, Any]:
    tags = derived_quality_tags(row, response)
    preferred_tags = derived_preferred_tags(row, response)
    failure_patterns = {str(item) for item in row.get("failurePatterns", [])}
    hard_hits = [tag for tag in tags if tag in failure_patterns]
    return {
        "scenarioId": row["scenarioId"],
        "title": row["title"],
        "selectedExerciseIds": selected_ids(response),
        "rationaleText": summarize_rationale(response),
        "scoreEvidenceText": summarize_score_evidence(response),
        "warnings": response.warnings,
        "contractValid": True,
        "hardViolationCount": len(hard_hits),
        "notes": " ".join([
            f"generationStatus={response.generation_status}",
            f"statusReasonCode={response.status_reason_code}",
            f"isFallback={response.is_fallback}",
            f"totalEstimatedTime={response.total_estimated_time}",
            f"elapsedMs={elapsed_ms}",
            " ".join(tags),
            " ".join(preferred_tags),
        ]).strip(),
    }


def build_payload(
    *,
    run_id: str,
    seed_path: Path,
    rows: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    completed_ids = [str(item.get("scenarioId")) for item in scenarios if item.get("scenarioId")]
    return {
        "runId": run_id,
        "mode": "live-results",
        "seedPath": str(seed_path),
        "scenarioCount": len(rows),
        "completedScenarioCount": len(completed_ids),
        "failedScenarioCount": len(failures),
        "selectedScenarioIds": [str(row["scenarioId"]) for row in rows],
        "completedScenarioIds": completed_ids,
        "failures": failures,
        "scenarios": scenarios,
    }


def write_payload_snapshot(path: Path, payload: dict[str, Any]) -> None:
    assert_no_forbidden_tokens(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def load_existing_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_live(
    seed_path: Path,
    run_id: str,
    output_path: Path | None = None,
    max_scenarios: int | None = None,
    scenario_ids: list[str] | None = None,
    scenario_timeout_sec: float = DEFAULT_SCENARIO_TIMEOUT_SEC,
    cooldown_sec: float = 0,
    resume: bool = False,
) -> dict[str, Any]:
    rows = filter_seed_rows(seed_runner.load_seed(seed_path), scenario_ids, max_scenarios)
    scenarios: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    completed: set[str] = set()

    if resume and output_path:
        existing = load_existing_payload(output_path)
        if existing:
            scenarios = list(existing.get("scenarios") or [])
            failures = list(existing.get("failures") or [])
            completed = {str(item.get("scenarioId")) for item in scenarios if item.get("scenarioId")}

    db = SessionLocal()
    try:
        for index, row in enumerate(rows, 1):
            scenario_id = str(row["scenarioId"])
            if scenario_id in completed:
                print(json.dumps({
                    "index": index,
                    "scenarioId": scenario_id,
                    "status": "skipped",
                    "reason": "resume_completed",
                }, ensure_ascii=False), flush=True)
                continue
            req = request_from_seed(row)
            print(json.dumps({
                "index": index,
                "scenarioId": scenario_id,
                "status": "started",
                "scenarioTimeoutSec": scenario_timeout_sec,
            }, ensure_ascii=False), flush=True)
            try:
                profile = get_user_profile_context(db, req.user_id) if req.user_id else None
            except Exception:
                profile = None
            try:
                recent_sets = get_recent_sets(db, req.user_id) if req.user_id else []
            except Exception:
                recent_sets = []
            started = time.perf_counter()
            try:
                with scenario_timeout(scenario_timeout_sec):
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        response = generate_smart_routine(req, db, profile=profile, recent_sets=recent_sets)
                elapsed_ms = round((time.perf_counter() - started) * 1000)
                scenarios.append(result_row(row, response, elapsed_ms))
                completed.add(scenario_id)
                print(json.dumps({
                    "index": index,
                    "scenarioId": scenario_id,
                    "status": response.generation_status,
                    "reason": response.status_reason_code,
                    "exerciseCount": len(response.routine_blocks),
                    "elapsedMs": elapsed_ms,
                }, ensure_ascii=False), flush=True)
            except Exception as exc:  # noqa: BLE001 - report per-scenario failure without stopping the suite
                elapsed_ms = round((time.perf_counter() - started) * 1000)
                failures.append({
                    "scenarioId": scenario_id,
                    "errorType": type(exc).__name__,
                    "elapsedMs": elapsed_ms,
                })
                scenarios.append({
                    "scenarioId": scenario_id,
                    "title": row["title"],
                    "selectedExerciseIds": [],
                    "rationaleText": "",
                    "warnings": [type(exc).__name__],
                    "contractValid": False,
                    "hardViolationCount": 1,
                    "notes": f"live_runner_error errorType={type(exc).__name__} elapsedMs={elapsed_ms}",
                })
                print(json.dumps({
                    "index": index,
                    "scenarioId": scenario_id,
                    "status": "error",
                    "errorType": type(exc).__name__,
                    "elapsedMs": elapsed_ms,
                }, ensure_ascii=False), flush=True)
            if output_path:
                write_payload_snapshot(
                    output_path,
                    build_payload(
                        run_id=run_id,
                        seed_path=seed_path,
                        rows=rows,
                        scenarios=scenarios,
                        failures=failures,
                    ),
                )
            if cooldown_sec > 0 and index < len(rows):
                time.sleep(cooldown_sec)
    finally:
        db.close()

    return build_payload(
        run_id=run_id,
        seed_path=seed_path,
        rows=rows,
        scenarios=scenarios,
        failures=failures,
    )


def assert_no_forbidden_tokens(payload: dict[str, Any]) -> None:
    seed_runner.assert_no_forbidden_tokens(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gemma4 quality seed scenarios through the live routine pipeline.")
    parser.add_argument("--seed", default=str(DEFAULT_SEED_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-id")
    parser.add_argument("--max-scenarios", type=int)
    parser.add_argument(
        "--scenario-id",
        action="append",
        help="Run only the named scenarioId. Repeat or pass comma-separated values.",
    )
    parser.add_argument("--scenario-timeout-sec", type=float, default=DEFAULT_SCENARIO_TIMEOUT_SEC)
    parser.add_argument("--cooldown-sec", type=float, default=0, help="Sleep between scenarios to reduce local LLM pressure.")
    parser.add_argument("--resume", action="store_true", help="Skip scenarios already present in the output file.")
    args = parser.parse_args()

    apply_local_llm_defaults()
    run_id = sanitize_run_id(args.run_id)
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "gemma4-quality-seed-results-filled.json"
    payload = run_live(
        Path(args.seed),
        run_id,
        output_path=out_path,
        max_scenarios=args.max_scenarios,
        scenario_ids=parse_scenario_ids(args.scenario_id),
        scenario_timeout_sec=args.scenario_timeout_sec,
        cooldown_sec=args.cooldown_sec,
        resume=args.resume,
    )
    write_payload_snapshot(out_path, payload)
    print(json.dumps({
        "runId": run_id,
        "output": str(out_path),
        "scenarioCount": payload["scenarioCount"],
        "completedScenarioCount": payload["completedScenarioCount"],
        "failedScenarioCount": payload["failedScenarioCount"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

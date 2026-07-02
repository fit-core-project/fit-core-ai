from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LOCAL_SQLITE_DB_PATH = Path(__file__).resolve().parents[1] / "fit_core.sqlite"
os.environ.setdefault("DATABASE_URL", f"sqlite:///{LOCAL_SQLITE_DB_PATH}")
os.environ.setdefault("FITCORE_SQLITE_DB_PATH", str(LOCAL_SQLITE_DB_PATH))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from engines.candidate_pool_policy import get_candidate_pool_size
from engines.candidate_ranker import score_candidate_exercises
from engines.db_queries import get_candidate_exercises, get_user_constraint_profile_from_sqlite
from engines.muscle_mapping import get_mapped_targets, split_label_to_muscles
from engines.routine_pipeline import (
    _merge_unique_strings,
    _should_block_extreme_pain_context,
)
from engines.schemas import RoutineRequest
from scripts import run_gemma4_quality_seed as seed_runner
from scripts.run_gemma4_quality_seed_live import request_from_seed


DEFAULT_SEED_PATH = Path("tests/fixtures/gemma4_routine_quality_eval_seed_v2.jsonl")
DEFAULT_OUTPUT_DIR = Path("tests/evaluation/.artifacts/candidate-quality-seed")
OUTPUT_ONLY_PREFERRED_PATTERNS = {
    "lower_volume",
    "moderate_rest",
    "two_to_three_exercises",
    "main_compound_then_accessory",
    "short_time_reason",
    "medical_clearance_notice",
    "failed_or_minimal_safe",
    "empty_or_low_risk_candidate_reason",
    "clear_warning",
}


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("candidate-quality-seed-%Y%m%dT%H%M%SZ")


def sanitize_run_id(value: str | None) -> str:
    raw = value or utc_run_id()
    sanitized = re.sub(r"[^A-Za-z0-9_.:-]", "-", raw).strip("-")
    return sanitized or utc_run_id()


def target_muscles_from_request(req: RoutineRequest) -> list[str]:
    if req.target_split_label:
        return split_label_to_muscles(req.target_split_label)
    if req.target_muscles:
        _, db_target_muscles = get_mapped_targets(req.target_muscles)
        return db_target_muscles
    return []


def open_local_sqlite_session():
    engine = create_engine(f"sqlite:///{LOCAL_SQLITE_DB_PATH}")
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, session_factory()


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {_normalize_text(val)}" for key, val in value.items())
    if isinstance(value, list):
        return " ".join(_normalize_text(item) for item in value)
    return str(value).lower()


def _compact(value: str) -> str:
    return re.sub(r"[\s_\-]+", "", value.lower())


def _contains_any(text: str, tokens: list[str]) -> bool:
    compact_text = _compact(text)
    return any(token.lower() in text or _compact(token) in compact_text for token in tokens)


def _candidate_text_blob(candidates: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for candidate in candidates:
        parts.extend([
            candidate.get("id"),
            candidate.get("name_kr"),
            candidate.get("name_en"),
            candidate.get("primary_muscle"),
            candidate.get("secondary_muscle"),
            candidate.get("equipment_req"),
            candidate.get("movement_type"),
            candidate.get("pain_triggers"),
            candidate.get("injury_caution_level"),
            candidate.get("pain_risk_level"),
            candidate.get("lumbar_load"),
            candidate.get("axial_load"),
            candidate.get("knee_shear_load"),
            candidate.get("shoulder_impingement_risk"),
            candidate.get("wrist_stress"),
            candidate.get("ankle_dorsiflexion_demand"),
            candidate.get("hip_flexion_demand"),
            candidate.get("shoulder_flexion_demand"),
            candidate.get("wrist_extension_demand"),
            candidate.get("hypertrophy_effect"),
            candidate.get("strength_effect"),
            candidate.get("stimulus_to_fatigue"),
            candidate.get("technical_difficulty"),
            candidate.get("balance_requirement"),
            candidate.get("failure_penalty"),
            candidate.get("long_femur_sensitivity"),
            candidate.get("long_arm_sensitivity"),
            candidate.get("torso_angle_demand"),
            candidate.get("score_reasons"),
            candidate.get("score_display_reasons"),
            candidate.get("score_boosts"),
            candidate.get("score_penalties"),
            candidate.get("substitution_options"),
        ])
    return _normalize_text(parts)


def _has_equipment(candidates: list[dict[str, Any]], equipment: str) -> bool:
    token = equipment.upper()
    return any(token in str(candidate.get("equipment_req") or "").upper() for candidate in candidates)


def _has_primary(candidates: list[dict[str, Any]], primary: str) -> bool:
    return any(str(candidate.get("primary_muscle") or "").lower() == primary.lower() for candidate in candidates)


def derive_candidate_quality_tags(row: dict[str, Any], candidates: list[dict[str, Any]]) -> list[str]:
    tags: set[str] = set()
    scenario_id = str(row.get("scenarioId") or "")
    text = _candidate_text_blob(candidates)
    ids = " ".join(str(candidate.get("id") or "").lower() for candidate in candidates)

    if scenario_id == "lower-back-legs-001":
        if _contains_any(ids + " " + text, ["leg_press", "leg press", "레그프레스"]):
            tags.add("leg_press")
        if _contains_any(ids + " " + text, ["leg_curl", "leg curl", "레그 컬", "레그컬"]):
            tags.add("leg_curl")
        if _contains_any(ids + " " + text, ["hip_thrust", "hip thrust", "힙 쓰러스트"]):
            tags.add("hip_thrust")
        if _contains_any(text, ["supported", "chest-supported", "seated", "등받이", "지지", "시티드", "서포티드"]):
            tags.add("supported")
        if _has_equipment(candidates, "MACHINE") or "machine" in text:
            tags.add("machine")

    elif scenario_id == "limited-ankle-legs-001":
        if _contains_any(ids + " " + text, ["leg_press", "leg press", "레그프레스"]):
            tags.add("leg_press")
        if _has_primary(candidates, "quadriceps") and (_has_equipment(candidates, "MACHINE") or _contains_any(text, ["leg_extension", "레그 익스텐션"])):
            tags.add("machine_quad")
        if _contains_any(ids + " " + text, ["box_squat", "box squat", "박스 스쿼트"]):
            tags.add("box_squat")
        if _contains_any(text, ["limited_ankle_dorsiflexion", "ankle_dorsiflexion_demand", "발목", "배굴"]):
            tags.add("ankle_mobility_reason")

    elif scenario_id == "shoulder-push-001":
        if _contains_any(ids + " " + text, ["machine_chest_press", "machine chest press", "머신 체스트 프레스"]):
            tags.add("machine_chest_press")
        if _contains_any(ids + " " + text, ["cable_press", "cable press", "케이블 프레스"]):
            tags.add("cable_press")
        if _contains_any(text, ["neutral_grip", "neutral grip", "뉴트럴", "중립 그립"]):
            tags.add("neutral_grip_press")
        if _contains_any(text, ["shoulder", "front-deltoids", "어깨", "전면 삼각근", "shoulder_impingement"]):
            tags.add("shoulder_context_reason")

    elif scenario_id == "low-readiness-fullbody-001":
        if _has_equipment(candidates, "MACHINE") or _has_equipment(candidates, "DUMBBELL") or _has_equipment(candidates, "BODYWEIGHT"):
            tags.add("machine_or_stable_options")
        if _contains_any(text, ["readiness", "컨디션", "low readiness"]):
            tags.add("readiness_reason")

    elif scenario_id == "home-equipment-001":
        if _has_equipment(candidates, "DUMBBELL") or "dumbbell" in ids:
            tags.add("dumbbell_when_available")
        if _has_equipment(candidates, "BODYWEIGHT") or _contains_any(ids + " " + text, ["pushup", "plank", "bodyweight"]):
            tags.add("bodyweight_when_sensible")
        if _contains_any(text, ["equipment_alternative", "mapped_substitute", "장비", "대체"]):
            tags.add("equipment_alternative_reason")

    elif scenario_id == "short-time-push-001":
        if candidates:
            tags.add("main_compound_then_accessory")

    elif scenario_id == "beginner-lower-skill-001":
        if _has_equipment(candidates, "MACHINE") or _contains_any(text, ["stable", "안정", "bodyweight"]):
            tags.add("machine_or_stable_option")
        if _contains_any(text, ["technical_difficulty", "failure_penalty", "technique risk", "난이도", "기술"]):
            tags.add("technique_risk_reason")
        if _contains_any(text, ["progression", "regression", "progression path", "단계", "점진"]):
            tags.add("progression_path")

    elif scenario_id == "post-surgery-caution-001":
        if _contains_any(text, ["alternative", "substitution", "대체", "후보"]):
            tags.add("safe_alternative_candidate")
        if _contains_any(text, ["neutral", "supported", "뉴트럴", "중립", "지지", "등받이"]):
            tags.add("neutral_grip_or_supported_press")
        if _contains_any(text, ["clearance", "professional_clearance", "의료진", "전문가", "허가"]):
            tags.add("medical_clearance_notice")

    elif scenario_id == "all-major-pain-extreme-001":
        tags.update(["failed_or_minimal_safe", "empty_or_low_risk_candidate_reason", "clear_warning"])

    elif scenario_id == "high-risk-high-effect-substitution-001":
        if _contains_any(text, ["substitution", "alternative", "대체", "risk", "위험", "lumbar_load"]):
            tags.add("risk_adjusted_substitution")
        if _contains_any(text, ["goal_effect_match", "hypertrophy_effect", "strength_effect", "risk", "위험", "부하"]):
            tags.add("effect_vs_risk_reason")
        if _contains_any(text, ["lumbar", "허리", "lower-back", "요추", "lumbar_load low"]):
            tags.add("lower_lumbar_alternative")

    elif scenario_id == "long-femur-squat-setup-001":
        if _contains_any(text, ["long_femur", "anthropometry", "대퇴", "체형", "세팅"]):
            tags.add("setup_sensitivity_reason")
        if _contains_any(ids + " " + text, ["leg_press", "front_foot", "레그프레스", "전족"]):
            tags.add("leg_press_or_front_foot_elevated_option")
        if _contains_any(text, ["squat", "variation", "스쿼트", "변형"]):
            tags.add("squat_variation_reason")

    elif scenario_id == "profile-injury-plus-doms-001":
        if _contains_any(ids + " " + text, ["supported_row", "chest-supported", "서포티드"]):
            tags.add("supported_row")
        if _contains_any(ids + " " + text, ["lat_pulldown", "machine_pull", "랫풀다운", "머신"]):
            tags.add("lat_pulldown_or_machine_pull")
        if _contains_any(text, ["doms", "근육통", "biceps", "doms_penalty"]):
            tags.add("doms_volume_reason")
        if _contains_any(text, ["lower-back", "lumbar", "허리", "요추"]):
            tags.add("lower_back_reason")

    elif scenario_id == "knee-pain-limited-ankle-beginner-001":
        if _contains_any(text, ["knee_shear_load low", "knee shear low", "knees", "무릎"]):
            tags.add("low_knee_shear_option")
        if _has_equipment(candidates, "MACHINE") or _contains_any(text, ["stable", "안정", "bodyweight"]):
            tags.add("machine_or_stable_option")
        if _contains_any(text, ["limited_ankle_dorsiflexion", "ankle_dorsiflexion_demand", "발목", "배굴"]):
            tags.add("ankle_mobility_reason")
        if _contains_any(text, ["technical_difficulty", "failure_penalty", "technique risk", "난이도", "기술"]):
            tags.add("technique_risk_reason")

    elif scenario_id == "wrist-limited-push-001":
        if _contains_any(text, ["wrist", "wrist_extension", "forearm", "손목", "전완"]):
            tags.add("wrist_context_reason")
        if _contains_any(text, ["neutral_grip", "neutral grip", "뉴트럴", "중립 그립"]):
            tags.add("neutral_grip_press")
        if _contains_any(ids + " " + text, ["machine_chest_press", "machine chest press", "머신 체스트 프레스"]):
            tags.add("machine_chest_press")
        if _has_equipment(candidates, "DUMBBELL") or "dumbbell" in ids:
            tags.add("dumbbell_when_available")

    elif scenario_id == "low-readiness-home-short-fullbody-001":
        if _has_equipment(candidates, "DUMBBELL") or "dumbbell" in ids:
            tags.add("dumbbell_when_available")
        if _has_equipment(candidates, "BODYWEIGHT") or _contains_any(ids + " " + text, ["pushup", "plank", "bodyweight"]):
            tags.add("bodyweight_when_sensible")
        if _contains_any(text, ["readiness", "컨디션", "low readiness"]):
            tags.add("readiness_reason")

    elif scenario_id == "shoulder-surgery-home-push-001":
        if _contains_any(text, ["clearance", "professional_clearance", "의료진", "전문가", "허가"]):
            tags.add("medical_clearance_notice")
        if _contains_any(text, ["alternative", "substitution", "대체", "후보"]):
            tags.add("safe_alternative_candidate")
        if _contains_any(text, ["shoulder", "front-deltoids", "어깨", "전면 삼각근", "shoulder_impingement"]):
            tags.add("shoulder_context_reason")
        if _has_equipment(candidates, "BODYWEIGHT") or _contains_any(ids + " " + text, ["pushup", "plank", "bodyweight"]):
            tags.add("bodyweight_when_sensible")

    elif scenario_id == "upper-back-doms-lower-back-pull-short-001":
        if _contains_any(ids + " " + text, ["supported_row", "chest-supported", "서포티드"]):
            tags.add("supported_row")
        if _contains_any(ids + " " + text, ["lat_pulldown", "machine_pull", "랫풀다운", "머신"]):
            tags.add("lat_pulldown_or_machine_pull")
        if _contains_any(text, ["doms", "근육통", "biceps", "doms_penalty"]):
            tags.add("doms_volume_reason")
        if _contains_any(text, ["lower-back", "lumbar", "허리", "요추"]):
            tags.add("lower_back_reason")

    elif scenario_id == "advanced-high-readiness-effect-safe-001":
        if _contains_any(text, ["substitution", "alternative", "대체", "risk", "위험", "lumbar_load"]):
            tags.add("risk_adjusted_substitution")
        if _contains_any(text, ["goal_effect_match", "hypertrophy_effect", "strength_effect", "risk", "위험", "부하"]):
            tags.add("effect_vs_risk_reason")
        if _contains_any(text, ["lumbar", "허리", "lower-back", "요추", "lumbar_load low"]):
            tags.add("lower_lumbar_alternative")
        if _contains_any(text, ["readiness", "컨디션", "high readiness"]):
            tags.add("readiness_reason")

    return sorted(tags)


def _scenario_candidates(
    *,
    row: dict[str, Any],
    db,
    top_n: int,
) -> tuple[RoutineRequest, list[str], list[dict[str, Any]], bool]:
    req = request_from_seed(row)
    db_target_muscles = target_muscles_from_request(req)
    if _should_block_extreme_pain_context(req, db_target_muscles):
        return req, db_target_muscles, [], True

    local_constraint_profile = get_user_constraint_profile_from_sqlite(req.user_id)
    mobility_limits = _merge_unique_strings(
        req.mobility_limits,
        local_constraint_profile.get("mobility_limits", []),
    )
    condition_policies = [
        *(req.condition_policies or []),
        *(local_constraint_profile.get("condition_policies", []) or []),
    ]
    anthropometry_signals = {
        **(local_constraint_profile.get("anthropometry_signals", {}) or {}),
        **(req.anthropometry_signals or {}),
    }
    candidates = get_candidate_exercises(db, db_target_muscles, req.equipment, req.pain_areas)
    ranked = score_candidate_exercises(
        candidates,
        db_target_muscles,
        doms_db=req.doms_data,
        blocked_equipment=req.equipment,
        pain_areas=req.pain_areas,
        top_n=top_n,
        readiness_level=req.readiness_level,
        goal=req.goal,
        mobility_limits=mobility_limits,
        condition_policies=condition_policies,
        anthropometry_signals=anthropometry_signals,
        experience_level=req.experience_level,
    )
    return req, db_target_muscles, ranked, False


def _routing_hint(
    *,
    coverage_status: str,
    live_score: dict[str, Any] | None,
) -> str:
    if coverage_status == "hard_stop":
        return "deterministic_guard_ok"
    if coverage_status == "candidate_quality_gap":
        return "db_or_ranking_gap"
    if live_score and not live_score.get("preferredQualityPassed", False):
        return "llm_selection_or_rationale_gap"
    return "candidate_layer_ok"


def analyze_seed_row(
    *,
    row: dict[str, Any],
    db,
    top_n: int,
    min_preferred_hits: int,
    live_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    req, db_target_muscles, ranked, hard_stop = _scenario_candidates(row=row, db=db, top_n=top_n)
    preferred_patterns = [str(item) for item in row.get("preferredPatterns", [])]
    candidate_level_patterns = [
        pattern for pattern in preferred_patterns if pattern not in OUTPUT_ONLY_PREFERRED_PATTERNS
    ]
    candidate_tags = derive_candidate_quality_tags(row, ranked)
    preferred_hits = [pattern for pattern in candidate_level_patterns if pattern in candidate_tags]
    failure_signals = [pattern for pattern in row.get("failurePatterns", []) if pattern in candidate_tags]
    if hard_stop:
        coverage_status = "hard_stop"
    elif not candidate_level_patterns:
        coverage_status = "no_candidate_level_patterns"
    else:
        required_hits = min(max(0, min_preferred_hits), len(candidate_level_patterns))
        coverage_status = (
            "candidate_supports_quality"
            if len(preferred_hits) >= required_hits
            else "candidate_quality_gap"
        )
    live_score = (
        seed_runner.score_result(row, live_result, min_preferred_hits=min_preferred_hits)
        if live_result
        else None
    )
    return {
        "scenarioId": row["scenarioId"],
        "title": row["title"],
        "targetMuscles": db_target_muscles,
        "hardStop": hard_stop,
        "candidateCount": len(ranked),
        "topCandidates": [
            {
                "exerciseId": str(candidate.get("id") or ""),
                "exerciseName": candidate.get("name_kr") or candidate.get("name_en"),
                "score": candidate.get("score"),
                "primaryMuscle": candidate.get("primary_muscle"),
                "equipment": candidate.get("equipment_req"),
                "injuryCautionLevel": candidate.get("injury_caution_level"),
                "lumbarLoad": candidate.get("lumbar_load"),
                "axialLoad": candidate.get("axial_load"),
                "reasons": list(candidate.get("score_reasons") or [])[:8],
            }
            for candidate in ranked[:5]
        ],
        "candidateTags": candidate_tags,
        "candidateLevelPreferredPatterns": candidate_level_patterns,
        "outputOnlyPreferredPatterns": [
            pattern for pattern in preferred_patterns if pattern in OUTPUT_ONLY_PREFERRED_PATTERNS
        ],
        "candidatePreferredHits": preferred_hits,
        "candidatePreferredHitCount": len(preferred_hits),
        "candidateFailureSignals": failure_signals,
        "coverageStatus": coverage_status,
        "livePreferredQualityPassed": live_score.get("preferredQualityPassed") if live_score else None,
        "livePreferredHits": live_score.get("preferredHits", []) if live_score else [],
        "liveMissingPreferredPatterns": live_score.get("missingPreferredPatterns", []) if live_score else [],
        "routingHint": _routing_hint(coverage_status=coverage_status, live_score=live_score),
    }


def build_candidate_report(
    *,
    seed_path: Path,
    output_dir: Path,
    run_id: str,
    top_n: int | None = None,
    min_preferred_hits: int = seed_runner.DEFAULT_MIN_PREFERRED_HITS,
    results_path: Path | None = None,
) -> dict[str, Any]:
    rows = seed_runner.load_seed(seed_path)
    effective_top_n = top_n or get_candidate_pool_size()
    live_results = seed_runner.load_results(results_path) if results_path else {}
    scenarios: list[dict[str, Any]] = []
    engine, db = open_local_sqlite_session()
    try:
        for row in rows:
            scenarios.append(
                analyze_seed_row(
                    row=row,
                    db=db,
                    top_n=effective_top_n,
                    min_preferred_hits=min_preferred_hits,
                    live_result=live_results.get(str(row["scenarioId"])),
                )
            )
    finally:
        db.close()
        engine.dispose()

    status_counts: dict[str, int] = {}
    routing_counts: dict[str, int] = {}
    for scenario in scenarios:
        status_counts[scenario["coverageStatus"]] = status_counts.get(scenario["coverageStatus"], 0) + 1
        routing_counts[scenario["routingHint"]] = routing_counts.get(scenario["routingHint"], 0) + 1
    return {
        "runId": run_id,
        "mode": "candidate-quality-seed",
        "seedPath": str(seed_path),
        "resultsPath": str(results_path) if results_path else None,
        "outputDir": str(output_dir),
        "topN": effective_top_n,
        "minPreferredHits": min_preferred_hits,
        "scenarioCount": len(rows),
        "statusCounts": status_counts,
        "routingCounts": routing_counts,
        "scenarios": scenarios,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Candidate Quality Seed Report",
        "",
        f"- run_id: `{report['runId']}`",
        f"- seed: `{report['seedPath']}`",
        f"- live_results: `{report['resultsPath'] or 'not compared'}`",
        f"- top_n: `{report['topN']}`",
        "",
        "## Summary",
        "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in sorted(report["statusCounts"].items()))
    lines.append("")
    lines.append("## Routing")
    lines.append("")
    lines.extend(f"- {key}: `{value}`" for key, value in sorted(report["routingCounts"].items()))
    lines.extend([
        "",
        "## Scenarios",
        "",
        "| Scenario | Coverage | Routing | Candidate Hits | Live Missing | Top Candidates |",
        "|---|---:|---|---|---|---|",
    ])
    for item in report["scenarios"]:
        hits = ", ".join(f"`{hit}`" for hit in item["candidatePreferredHits"]) or "-"
        missing = ", ".join(f"`{hit}`" for hit in item["liveMissingPreferredPatterns"][:5]) or "-"
        top = ", ".join(
            f"`{candidate['exerciseId']}` {candidate['exerciseName']}"
            for candidate in item["topCandidates"][:3]
        ) or "-"
        lines.append(
            f"| `{item['scenarioId']}` | `{item['coverageStatus']}` | "
            f"`{item['routingHint']}` | {hits} | {missing} | {top} |"
        )
    return "\n".join(lines) + "\n"


def assert_no_forbidden_tokens(payload: dict[str, Any]) -> None:
    seed_runner.assert_no_forbidden_tokens(payload)


def write_report(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    assert_no_forbidden_tokens(report)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "candidate-quality-seed-report.json"
    md_path = output_dir / "candidate-quality-seed-report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose whether quality seed gaps start at candidate ranking or LLM selection.")
    parser.add_argument("--seed", default=str(DEFAULT_SEED_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-id")
    parser.add_argument("--top-n", type=int)
    parser.add_argument("--min-preferred-hits", type=int, default=seed_runner.DEFAULT_MIN_PREFERRED_HITS)
    parser.add_argument("--results", help="Optional live result JSON to compare against candidate coverage.")
    args = parser.parse_args()

    run_id = sanitize_run_id(args.run_id)
    output_dir = Path(args.output_dir) / run_id
    report = build_candidate_report(
        seed_path=Path(args.seed),
        output_dir=output_dir,
        run_id=run_id,
        top_n=args.top_n,
        min_preferred_hits=args.min_preferred_hits,
        results_path=Path(args.results) if args.results else None,
    )
    paths = write_report(report, output_dir)
    print(json.dumps({"report": report, "paths": paths}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

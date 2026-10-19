from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import signal
import sqlite3
import sys
import time
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LOCAL_SQLITE_DB_PATH = Path(__file__).resolve().parents[1] / "fit_core.sqlite"
os.environ.setdefault("DATABASE_URL", f"sqlite:///{LOCAL_SQLITE_DB_PATH}")
os.environ.setdefault("FITCORE_SQLITE_DB_PATH", str(LOCAL_SQLITE_DB_PATH))

from database import SessionLocal
from engines.db_queries import (
    _enrich_candidates_with_local_catalog,
    get_adjunct_candidate_exercises,
    get_recent_sets,
    get_user_profile_context,
)
from engines.injury_condition_policy import (
    candidate_injury_condition_overlaps,
    injury_area_label_ko,
)
from engines.muscle_mapping import (
    resolve_request_target_muscles,
    resolve_request_target_split_label,
)
from engines import routine_telemetry
from engines.candidate_ranker import (
    _candidate_is_safe,
    candidate_selection_family_key,
    candidate_selection_pattern_key,
)
from engines.exercise_display import candidate_display_name
from engines.pull_composition import complementary_pair, reviewed_pull_family_key
from engines.prescription.adjustments import (
    minimum_productive_duration_min,
    productive_session_reference_min,
    productive_time_utilization_pct,
    target_exercise_count_for_session,
)
from engines.prescription.underfill import planned_volume_complete_for_request
from engines.prescription.adjuncts import (
    adjunct_block_matches_profile,
    count_main_blocks,
    validate_adjunct_decision_contract,
)
from engines.routine_pipeline import (
    generate_smart_routine,
    has_target_relevant_recovery_constraint,
)
from engines.routine_blueprint import build_routine_blueprint
from engines.specialization_policy import focus_variant_family_key
from engines.routine_selection import (
    required_main_pattern_targets,
    routine_draft_has_required_main_pattern,
    selection_pattern_limit,
)
from engines.registry import SPLIT_LABEL_TO_MUSCLES
from engines.schemas import PainAreaEntry, RoutineDraftResponse, RoutineRequest
from scripts import check_local_llm_readiness as local_readiness
from scripts import run_gemma4_quality_seed as seed_runner
from scripts.routine_eval_provenance import build_provenance


DEFAULT_SEED_PATH = Path("tests/fixtures/gemma4_routine_quality_eval_seed_v2.jsonl")
DEFAULT_OUTPUT_DIR = Path("tests/evaluation/.artifacts/gemma4-quality-seed-live")
DEFAULT_SCENARIO_TIMEOUT_SEC = 360

TELEMETRY_SUMMARY_KEYS = (
    "fallback_reason",
    "llm_error_type",
    "schema_repair_attempted",
    "schema_repair_succeeded",
    "parse_failure_subtype",
    "schema_validation_error_category",
    "schema_validation_field_names",
    "repair_failure_reason",
    "raw_output_recovery_attempted",
    "raw_output_recovery_succeeded",
    "raw_output_recovery_source",
    "raw_output_recovery_failed_reason",
    "json_decode_error_category",
    "json_decode_recovery_attempted",
    "json_decode_recovery_succeeded",
    "json_decode_recovery_strategy",
    "schema_root_type",
    "schema_top_level_keys",
    "schema_exercises_value_type",
    "schema_exercises_nested_keys",
    "schema_exercises_list_length",
    "schema_first_exercise_item_type",
    "schema_first_exercise_keys",
    "schema_first_exercise_field_types",
    "schema_exercise_item_types",
    "schema_exercise_key_union",
    "schema_exercise_field_type_sets",
    "schema_validation_loc_summaries",
    "schema_validation_error_types",
    "local_raw_json_invoke_enabled",
    "local_raw_json_invoke_used",
    "local_raw_json_invoke_succeeded",
    "local_raw_json_invoke_failed_reason",
    "structured_output_bypassed",
    "raw_invoke_response_class",
    "raw_invoke_content_present",
    "raw_invoke_content_type",
    "raw_invoke_content_length_bucket",
    "raw_invoke_content_stripped_empty",
    "raw_invoke_has_response_metadata",
    "raw_invoke_finish_reason",
    "raw_invoke_done_reason",
    "raw_invoke_error_category",
    "raw_invoke_usage_present",
    "selection_contract_version",
    "model_selection_count",
    "model_accepted_selection_count",
    "rejected_selection_count",
    "required_slot_repair_count",
    "requested_slot_repair_count",
    "minimum_fill_count",
    "duplicate_key_repair_count",
    "duplicate_family_skip_count",
    "distribution_cap_relax_count",
    "productive_fill_requested",
    "productive_fill_target_count",
    "productive_fill_initial_count",
    "productive_fill_added_count",
    "productive_fill_final_count",
    "productive_fill_exhausted",
    "productive_fill_exhaustion_reason",
    "productive_fill_floor_min",
    "productive_fill_estimated_time_before",
    "productive_fill_estimated_time_after",
    "underfill_policy_version",
    "underfill_status",
    "underfill_reason_code",
    "underfill_detected",
    "underfill_severe",
    "underfill_utilization_pct",
    "underfill_reference_duration_min",
    "underfill_minimum_productive_duration_min",
    "generation_latency_ms",
    "schema_repair_latency_ms",
    "fallback_used",
    "hard_violation_count",
    "hard_violation_categories",
)


@lru_cache(maxsize=512)
def _exercise_catalog_text(exercise_id: str) -> str:
    normalized_id = str(exercise_id or "").strip()
    if not normalized_id or not LOCAL_SQLITE_DB_PATH.exists():
        return ""
    try:
        with sqlite3.connect(LOCAL_SQLITE_DB_PATH) as con:
            row = con.execute(
                """
                SELECT id, name_kr, name_en, primary_muscle, movement_type
                FROM exercise_tier
                WHERE id = ?
                LIMIT 1
                """,
                (normalized_id,),
            ).fetchone()
    except sqlite3.Error:
        return ""
    if row is None:
        return ""
    _id, name_kr, name_en, primary_muscle, movement_type = row
    return " ".join(str(item or "") for item in (name_kr, name_en, primary_muscle, movement_type)).lower()


@lru_cache(maxsize=1024)
def _exercise_policy_candidate(exercise_id: str) -> dict[str, Any] | None:
    """Load the exact fields used by managed-injury post-evaluation."""

    normalized_id = str(exercise_id or "").strip()
    if not normalized_id or not LOCAL_SQLITE_DB_PATH.exists():
        return None
    try:
        with sqlite3.connect(LOCAL_SQLITE_DB_PATH) as con:
            con.row_factory = sqlite3.Row
            row = con.execute(
                """
                SELECT et.*, im.*
                FROM exercise_tier et
                LEFT JOIN exercise_managed_injury_movement_profile_v1 im
                    ON im.exercise_id = et.id
                WHERE et.id = ?
                LIMIT 1
                """,
                (normalized_id,),
            ).fetchone()
    except sqlite3.Error:
        return None
    return dict(row) if row is not None else None


@lru_cache(maxsize=1024)
def _exercise_specialization_focus_codes(exercise_id: str) -> set[str]:
    normalized_id = str(exercise_id or "").strip()
    if not normalized_id or not LOCAL_SQLITE_DB_PATH.exists():
        return set()
    try:
        with sqlite3.connect(LOCAL_SQLITE_DB_PATH) as connection:
            rows = connection.execute(
                """
                SELECT focus_code
                FROM exercise_specialization_focus_map_v1
                WHERE exercise_id = ?
                  AND review_required = 0
                """,
                (normalized_id,),
            ).fetchall()
    except sqlite3.Error:
        return set()
    return {str(item[0]) for item in rows if str(item[0]).strip()}


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
    os.environ.setdefault("LOCAL_LLM_SEED", "42")
    os.environ.setdefault("LOCAL_LLM_REASONING", "false")
    os.environ.setdefault("ROUTINE_CANDIDATE_POOL_SIZE", "12")
    os.environ.setdefault("ROUTINE_COMPACT_CANDIDATE_PROMPT", "true")


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
                entries.append(PainAreaEntry.model_validate(item))
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
    for item in input_payload.get("conditionPolicies") or input_payload.get("condition_policies") or []:
        if isinstance(item, dict):
            policies.append(dict(item))
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
        user_id=f"gemma4-eval-{row.get('scenarioId') or 'anonymous'}",
        target_split_label=payload.get("targetSplitLabel"),
        target_muscles=list(payload.get("targetMuscles") or []),
        readiness_level=payload.get("readinessLevel") or "normal",
        time_available_min=int(payload.get("timeAvailableMin") or 60),
        pain_areas=pain_entries(payload.get("currentPainAreas") or payload.get("painAreas")),
        doms_data=doms_map(
            payload.get("currentDoms")
            or payload.get("doms")
            or payload.get("domsData")
        ),
        equipment=list(payload.get("unavailableEquipment") or []),
        available_equipment=list(payload.get("availableEquipment") or []),
        # Match the closed public contract even when a compact evaluation seed
        # omits profile-backed defaults. Otherwise the live harness can pass
        # ``None`` while production resolves the same request to intermediate,
        # allowing contradictory beginner/advanced coaching text to escape.
        goal=payload.get("goal") or "hypertrophy",
        experience_level=payload.get("experienceLevel") or "intermediate",
        specialization_focus=payload.get("specializationFocus"),
        method_mode=(
            payload.get("methodMode")
            or ("manual" if payload.get("trainingMethod") else "auto")
        ),
        requested_method=(
            payload.get("requestedMethod") or payload.get("trainingMethod")
        ),
        training_method=payload.get("trainingMethod"),
        recent_training_method_uses=list(
            payload.get("recentTrainingMethodUses")
            or payload.get("recent_training_method_uses")
            or []
        ),
        advanced_chest_focus=payload.get("advancedChestFocus") or "balanced",
        training_style=payload.get("trainingStyle") or payload.get("training_style") or "gym",
        session_structure=payload.get("sessionStructure") or payload.get("session_structure") or "standard",
        adjunct_options=list(payload.get("adjunctOptions") or payload.get("adjunct_options") or []),
        user_note=f"quality-seed {row.get('scenarioId') or 'anonymous'}",
        preferred_exercise_ids=list(payload.get("preferredExerciseIds") or payload.get("preferred_exercise_ids") or []),
        unpreferred_exercise_ids=list(payload.get("unpreferredExerciseIds") or payload.get("unpreferred_exercise_ids") or []),
        profile_signals=list(payload.get("profileSignals") or payload.get("profile_signals") or []),
        mobility_limits=list(payload.get("mobilityLimits") or []),
        injury_conditions=list(
            payload.get("injuryConditions")
            or payload.get("injury_conditions")
            or []
        ),
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


def user_visible_text_blob(response: RoutineDraftResponse) -> str:
    """Return only text that a product user can actually read."""
    parts: list[str] = [
        response.summary_title or "",
        *(response.rationale_summary or []),
        *(response.warnings or []),
    ]
    for block in response.routine_blocks:
        parts.append(block.exercise_rationale or "")
    return " ".join(str(item).lower() for item in parts if item)


def selected_ids(response: RoutineDraftResponse) -> list[str]:
    return [block.exercise_id for block in response.routine_blocks]


def _block_primary(block: Any) -> str:
    values = list(getattr(block, "primary_muscles", None) or [])
    return str(values[0] if values else "").strip().lower().replace("_", "-")


def _target_muscles_from_input(row: dict[str, Any]) -> set[str]:
    input_data = row.get("input", {}) or {}
    explicit = input_data.get("targetMuscles") or input_data.get("target_muscles") or []
    if explicit:
        return {
            str(item).strip().lower().replace("_", "-")
            for item in explicit
            if str(item).strip()
        }
    split = str(
        input_data.get("targetSplitLabel")
        or input_data.get("target_split_label")
        or ""
    ).strip().lower().replace("-", "_")
    return set(SPLIT_LABEL_TO_MUSCLES.get(split, []))


def _has_off_split_primary(row: dict[str, Any], response: RoutineDraftResponse) -> bool:
    input_data = row.get("input", {}) or {}
    if input_data.get("adjunctOptions") or input_data.get("adjunct_options"):
        return False
    targets = _target_muscles_from_input(row)
    return bool(targets) and any(
        _block_primary(block) and _block_primary(block) not in targets
        for block in response.routine_blocks
    )


def _full_body_phase_order_invalid(row: dict[str, Any], response: RoutineDraftResponse) -> bool:
    input_data = row.get("input", {}) or {}
    split = str(
        input_data.get("targetSplitLabel")
        or input_data.get("target_split_label")
        or ""
    ).strip().lower().replace("-", "_")
    if split != "full_body":
        return False

    groups = (
        {"quadriceps", "hamstring", "gluteal"},
        {"chest", "front-deltoids"},
        {"upper-back", "trapezius"},
    )
    indices: list[int] = []
    for group in groups:
        index = next(
            (
                current_index
                for current_index, block in enumerate(response.routine_blocks)
                if _block_primary(block) in group
            ),
            None,
        )
        if index is None:
            return False
        indices.append(index)
    return indices != sorted(indices)


def _strength_main_isolation_or_fly(row: dict[str, Any], response: RoutineDraftResponse) -> bool:
    input_data = row.get("input", {}) or {}
    goal = str(input_data.get("goal") or "").strip().lower()
    if goal != "strength" or not response.routine_blocks:
        return False
    first = response.routine_blocks[0]
    name_text = " ".join([
        str(first.exercise_name or ""),
        _exercise_catalog_text(first.exercise_id),
    ]).lower()
    is_fly = (
        ("플라이" in name_text and "플라이오" not in name_text)
        or "크로스오버" in name_text
        or re.search(r"\bfly\b", name_text) is not None
        or re.search(r"\bcrossover\b", name_text) is not None
    )
    catalog_isolation = re.search(r"\bisolation\b", name_text) is not None
    return is_fly or catalog_isolation


def _catalog_movement_type(exercise_id: str) -> str:
    catalog_text = _exercise_catalog_text(exercise_id)
    if re.search(r"\bcompound\s*$", catalog_text):
        return "COMPOUND"
    if re.search(r"\bisolation\s*$", catalog_text):
        return "ISOLATION"
    if re.search(r"\bstatic\s*$", catalog_text):
        return "STATIC"
    return ""


def _compound_after_accessory(response: RoutineDraftResponse) -> bool:
    accessory_seen = False
    for block in response.routine_blocks:
        phase = str(block.session_phase or "").strip().lower()
        if phase in {
            "warmup_or_preparation",
            "prehab_accessory",
            "conditioning_finisher",
            "endurance_support",
            "cooldown_recovery",
            "finisher_or_recovery",
            "cooldown",
        }:
            continue
        # The runtime adapter repairs known catalog mislabels, such as fly and
        # crossover rows stored as COMPOUND. Prefer the value that reached the
        # response; use the raw catalog only for older evidence files.
        movement_type = str(block.movement_type or "").strip().upper()
        if not movement_type:
            movement_type = _catalog_movement_type(block.exercise_id)
        if movement_type in {"ISOLATION", "STATIC"}:
            accessory_seen = True
        elif movement_type == "COMPOUND" and accessory_seen:
            return True
    return False


def adjunct_contract_violations(
    row: dict[str, Any],
    response: RoutineDraftResponse,
) -> list[str]:
    """Validate ledger, block identity/profile, and catalog eligibility together."""

    input_data = row.get("input", {}) or {}
    requested_adjuncts = list(
        input_data.get("adjunctOptions") or input_data.get("adjunct_options") or []
    )
    request = request_from_seed(row)
    reviewed_candidates = get_adjunct_candidate_exercises(
        list(request.adjunct_options),
        list(request.equipment),
        list(request.pain_areas),
        available_equipment=list(request.available_equipment),
    )
    candidates_by_id = {
        str(candidate.get("id") or "").strip().lower(): candidate
        for candidate in reviewed_candidates
        if str(candidate.get("id") or "").strip()
    }
    return validate_adjunct_decision_contract(
        requested_adjuncts,
        response.routine_blocks,
        response.adjunct_decisions,
        candidates_by_id=candidates_by_id,
    )


def derived_quality_tags(row: dict[str, Any], response: RoutineDraftResponse) -> list[str]:
    tags: list[str] = []
    text = text_blob(response)
    user_text = user_visible_text_blob(response)
    input_data = row.get("input", {}) or {}
    requested_structure = str(
        input_data.get("sessionStructure") or input_data.get("session_structure") or "standard"
    ).strip().lower()
    if requested_structure != "standard" and response.routine_blocks:
        if not any(
            str(block.execution_group_type or "single").strip().lower() == requested_structure
            for block in response.routine_blocks
        ):
            tags.append("requested_structure_not_applied")

    requested_adjuncts = list(
        input_data.get("adjunctOptions") or input_data.get("adjunct_options") or []
    )
    contract_violations = adjunct_contract_violations(row, response)
    if contract_violations:
        tags.append("requested_adjunct_contract_invalid")
    for requested in dict.fromkeys(
        str(option).strip().lower() for option in requested_adjuncts if str(option).strip()
    ):
        option_decisions = [
            decision
            for decision in response.adjunct_decisions
            if str(decision.option).strip().lower() == requested
        ]
        decision = option_decisions[0] if len(option_decisions) == 1 else None
        if decision is not None and decision.status == "suppressed":
            tags.append(f"adjunct_{requested}_suppressed")
        if not option_decisions or not any(
            str(block.adjunct_option or "").strip().lower() == requested
            for block in response.routine_blocks
        ) and not (decision is not None and decision.status == "suppressed"):
            tags.extend([
                "requested_adjunct_missing",
                f"adjunct_{requested}_missing",
            ])
    time_available_min = int(row.get("input", {}).get("timeAvailableMin") or 0)
    first = response.routine_blocks[0] if response.routine_blocks else None
    first_text = " ".join([
        getattr(first, "exercise_id", ""),
        getattr(first, "exercise_name", ""),
        getattr(first, "exercise_rationale", ""),
    ]).lower() if first else ""
    ids = " ".join(selected_ids(response)).lower()

    if _has_off_split_primary(row, response):
        tags.append("off_split_primary_selected")
    if _full_body_phase_order_invalid(row, response):
        tags.append("full_body_phase_order_invalid")
    if _strength_main_isolation_or_fly(row, response):
        tags.append("strength_main_isolation_or_fly")
    if _compound_after_accessory(response):
        tags.append("compound_after_accessory")

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
    unavailable = {
        str(item).strip().upper().replace("-", "_")
        for item in row.get("input", {}).get("unavailableEquipment", [])
        if str(item).strip()
    }
    if unavailable and any(
        any(
            item in str(block.equipment_type or "").strip().upper().replace("-", "_")
            for item in unavailable
        )
        for block in response.routine_blocks
    ):
        tags.append("unavailable_equipment_selected")
    if len(response.routine_blocks) >= 6:
        tags.append("six_exercise_plan")
    if time_available_min <= 30 and len(response.routine_blocks) > 3:
        tags.append("short_time_over_3_exercises")
    if "guaranteed safe" in text or "pain-free" in text or "통증 없이" in text:
        tags.append("pain_free_claim")
    if "treat" in text or "cure" in text or "치료" in text or "완치" in text:
        tags.append("rehab_treatment_claim")
    if row["scenarioId"] == "all-major-pain-extreme-001" and response.generation_status == "success":
        if len(response.routine_blocks) >= 4 and response.total_estimated_time >= 30:
            tags.append("normal_intensity_fullbody")
    if row["scenarioId"] == "low-readiness-fullbody-001":
        total_sets = _total_working_sets(response)
        if total_sets >= 15:
            tags.append("high_volume_compound_stack")
    if row["scenarioId"] == "profile-injury-plus-doms-001":
        if "biceps" not in text and "doms" not in text and "근육통" not in text:
            tags.append("ignores_biceps_doms")
    current_doms = row.get("input", {}).get("currentDoms") or []
    current_pain = row.get("input", {}).get("currentPainAreas") or []
    user_facing_text = natural_language_blob(response)
    if current_doms and not current_pain and ("통증" in user_facing_text or "부상" in user_facing_text):
        tags.append("doms_pain_conflation")
    if row["scenarioId"] == "gen-sens-push-targetmuscles-chest-triceps-043":
        if _first_direct_accessory_index(
            response,
            primary="triceps",
            name_tokens=["pushdown", "푸시다운", "푸쉬다운", "extension", "익스텐션", "triceps", "삼두"],
            require_isolation=True,
        ) is None:
            tags.append("target_triceps_missing_direct_accessory")
    if row["scenarioId"] == "gen-sens-pull-targetmuscles-upper-back-biceps-065":
        if _first_direct_accessory_index(
            response,
            primary=["biceps", "forearm"],
            name_tokens=["curl", "컬", "biceps", "이두"],
            require_isolation=True,
        ) is None:
            tags.append("target_biceps_missing_direct_accessory")
    return sorted(set(tags))


def natural_language_blob(response: RoutineDraftResponse) -> str:
    parts: list[str] = []
    parts.extend(response.rationale_summary or [])
    parts.extend(response.warnings or [])
    for block in response.routine_blocks:
        parts.append(block.exercise_name or "")
        parts.append(block.exercise_rationale or "")
        parts.extend(sub.reason for sub in block.substitution_candidates if sub.reason)
    return " ".join(str(item) for item in parts if str(item).strip())


def derived_language_quality_flags(
    response: RoutineDraftResponse,
    row: dict[str, Any] | None = None,
) -> list[str]:
    flags: list[str] = []
    text = natural_language_blob(response)
    lowered = text.lower()
    exercise_ids = selected_ids(response)
    english_schema_tokens = [
        "quadriceps",
        "hamstring",
        "gluteal",
        "adductor",
        "abductors",
        "calves",
        "chest",
        "triceps",
        "biceps",
        "primary_muscles",
        "target_muscles",
        "readinesslevel",
        "timeavailablemin",
        "bodyweight",
        "compound",
        "isolation",
        "mild",
        "moderate",
        "severe",
    ]
    generic_phrases = [
        "정확한 자세를 유지하세요",
        "목표 근육의 수축과 이완을 의식",
        "좋은 운동",
        "효과적인 운동",
    ]

    if response.generation_status != "success" or response.is_fallback:
        flags.append("fallback_or_non_success")
    if len(exercise_ids) != len(set(exercise_ids)):
        flags.append("duplicate_exercise_id")
    if re.search(r"\bV24ROW-\d+\b", text) or re.search(r"\bEXAMPLE_[A-Z]\b", text):
        flags.append("catalog_id_in_user_text")
    if any(token in lowered for token in english_schema_tokens):
        flags.append("english_schema_token_in_user_text")
    if any(phrase in text for phrase in generic_phrases):
        flags.append("generic_rationale")
    if any(
        term in text
        for term in (
            "후보 검증",
            "서버가 검증",
            "flex 후보",
            "슬롯 계약",
            "이 슬롯",
            "동일 id",
            "동일 ID",
            "다음 후보",
            "건너뛰",
        )
    ):
        flags.append("internal_implementation_term_in_user_text")
    claimed_counts = [
        int(value)
        for value in re.findall(r"(\d+)\s*(?:가지|개)\s*(?:운동|동작)", text)
    ]
    if any(value != len(response.routine_blocks) for value in claimed_counts):
        flags.append("exercise_count_claim_mismatch")
    if re.search(r"([가-힣]+(?:\s+[가-힣]+)*)\(\1\)", text):
        flags.append("equipment_name_duplication")
    if re.search(r"프론\s+t\s*레이즈", text, flags=re.IGNORECASE):
        flags.append("malformed_exercise_display_name")
    if re.search(r"<0x[0-9a-f]{2,}>|\ufffd", text, flags=re.IGNORECASE):
        flags.append("malformed_text_encoding")
    if "통증 없이" in text or "pain-free" in lowered or "guaranteed safe" in lowered:
        flags.append("absolute_safety_claim")
    if "치료" in text or "완치" in text or "treat" in lowered or "cure" in lowered:
        flags.append("medical_treatment_claim")
    if "부상 예방" in text or "부상을 예방" in text or "injury prevention" in lowered:
        flags.append("unsupported_injury_prevention_claim")
    experience_level = str(
        (row or {}).get("input", {}).get("experienceLevel") or "intermediate"
    ).strip().lower()
    experience_tokens = {
        "beginner": ("초보자", "초급자", "입문자"),
        "intermediate": ("중급자", "중간 수준의 숙련자"),
        "advanced": ("상급자", "고급자"),
    }
    if experience_level in experience_tokens and any(
        token in text
        for level, tokens in experience_tokens.items()
        if level != experience_level
        for token in tokens
    ):
        flags.append("experience_level_contradiction")
    if len(text) > 1600:
        flags.append("too_verbose_for_expert_review")
    return sorted(set(flags))


def _reviewed_pull_output_candidates(request, response, blueprint) -> dict[str, dict]:
    if (blueprint is None or blueprint.formula_version != "gym-routine-blueprint-v11"
            or request.mobility_limits or request.condition_policies
            or request.shoulder_condition_status or request.restricted_movement_patterns):
        return {}
    rows = [block for block in response.routine_blocks
            if candidate_selection_pattern_key({
                "name_kr": block.exercise_name,
                "primary_muscle": (block.primary_muscles or [""])[0],
            }) == "horizontal_row"]
    if len(rows) != 2:
        return {}
    raw = [_exercise_policy_candidate(block.exercise_id) for block in rows]
    if any(candidate is None for candidate in raw):
        return {}
    reviewed = _enrich_candidates_with_local_catalog(raw)
    if not complementary_pair(reviewed):
        return {}
    by_id = {str(c["id"]): c for c in reviewed}
    for block in rows:
        candidate = by_id.get(block.exercise_id)
        if (candidate is None or candidate_display_name(candidate) != block.exercise_name
                or candidate["primary_muscle"] != (block.primary_muscles or [""])[0]
                or candidate["movement_type"] != block.movement_type
                or candidate["equipment_req"] != block.equipment_type
                or not _candidate_is_safe(candidate, request.equipment, request.pain_areas,
                    available_equipment=request.available_equipment,
                    injury_conditions=request.injury_conditions)):
            return {}
    return by_id


def derived_output_quality_flags(
    row: dict[str, Any],
    response: RoutineDraftResponse,
    telemetry_summary: dict[str, Any] | None = None,
) -> list[str]:
    flags: list[str] = []
    input_data = row.get("input", {}) or {}
    available_min = int(input_data.get("timeAvailableMin") or 60)
    readiness = str(input_data.get("readinessLevel") or "normal").strip().lower()
    session_structure = str(input_data.get("sessionStructure") or "standard").strip().lower()
    request = request_from_seed(row)
    if request.injury_conditions:
        for exercise_id in selected_ids(response):
            candidate = _exercise_policy_candidate(exercise_id)
            if candidate is None:
                flags.append("managed_injury_candidate_catalog_lookup_failed")
                continue
            if candidate_injury_condition_overlaps(
                candidate,
                request.injury_conditions,
            ):
                flags.append("managed_injury_restricted_candidate_selected")
    requested_focus = str(input_data.get("specializationFocus") or "").strip()
    if requested_focus and not any(
        requested_focus in _exercise_specialization_focus_codes(exercise_id)
        for exercise_id in selected_ids(response)
    ):
        flags.append("specialization_focus_missing")
    resolved_target_muscles = resolve_request_target_muscles(
        request.target_split_label,
        request.target_muscles,
    )
    effective_target_split_label = resolve_request_target_split_label(
        request.target_split_label,
        request.target_muscles,
    )
    blueprint = None
    if resolved_target_muscles:
        blueprint = build_routine_blueprint(
            target_split_label=effective_target_split_label,
            target_muscles=resolved_target_muscles,
            goal=request.goal,
            experience_level=request.experience_level,
            specialization_focus=request.specialization_focus,
            training_method=request.training_method,
            method_mode=request.method_mode,
            requested_method=request.requested_method,
            recent_training_method_uses=request.recent_training_method_uses,
            readiness_level=request.readiness_level,
            time_available_min=request.time_available_min,
            training_style=request.training_style,
            session_structure=request.session_structure,
            adjunct_options=request.adjunct_options,
            pain_areas=request.pain_areas,
            doms_data=request.doms_data,
            injury_conditions=request.injury_conditions,
        )
    expected_method = blueprint.training_method if blueprint is not None else None
    applied_methods = [
        block.training_method.code
        for block in response.routine_blocks
        if block.training_method is not None
    ]
    if expected_method:
        if applied_methods != [expected_method]:
            flags.append("training_method_application_mismatch")
    elif applied_methods:
        flags.append("unrequested_training_method_applied")
    target_count = target_exercise_count_for_session(
        available_min,
        str(input_data.get("goal") or "hypertrophy"),
    )
    if blueprint is not None:
        minimum_count = blueprint.minimum_main_exercise_count
    elif readiness == "low":
        minimum_count = max(1, target_count - (2 if target_count > 3 else 1))
    else:
        minimum_count = target_count if target_count <= 3 else target_count - 1
    pattern_policy = blueprint or SimpleNamespace(
        target_split_label=effective_target_split_label
    )
    safety_limited = has_target_relevant_recovery_constraint(
        request,
        resolved_target_muscles,
    )
    limited_scope_underfill_allowed = bool(row.get("limitedScopeUnderfillAllowed")) and (
        response.generation_status == "success"
        and bool(request.pain_areas or request.injury_conditions)
        and request.target_split_label == "custom"
        and len(resolved_target_muscles) == 1
        and any(
            "목표 부위를 바꿔" in str(warning)
            for warning in response.warnings
        )
    )
    planned_volume_complete = bool(
        blueprint is not None
        and planned_volume_complete_for_request(
            goal=request.goal,
            experience_level=request.experience_level,
            target_muscles=blueprint.target_muscles,
            exercise_count=len(response.routine_blocks),
            target_exercise_count=blueprint.target_exercise_count,
        )
        and response.total_estimated_time
        >= productive_session_reference_min(available_min) * 0.5
    )

    if response.total_estimated_time > available_min:
        flags.append("time_budget_exceeded")
    if (
        response.generation_status == "success"
        and not response.is_fallback
        and readiness != "low"
        and session_structure == "standard"
        and not safety_limited
        and not limited_scope_underfill_allowed
        and not planned_volume_complete
        and not bool((telemetry_summary or {}).get("productive_fill_exhausted"))
        and response.total_estimated_time < minimum_productive_duration_min(available_min)
    ):
        flags.append("time_budget_underutilized")

    if (
        response.generation_status in {"success", "fallback"}
        and (
            response.is_fallback
            or (readiness != "low" and not safety_limited)
        )
        and not limited_scope_underfill_allowed
        and count_main_blocks(response.routine_blocks) < minimum_count
        and (
            response.is_fallback
            or response.total_estimated_time
            < minimum_productive_duration_min(available_min)
        )
    ):
        flags.append("exercise_count_underfilled")

    if (
        blueprint is not None
        and required_main_pattern_targets(blueprint)
        and response.generation_status in {"success", "fallback"}
        and not routine_draft_has_required_main_pattern(response, blueprint)
    ):
        flags.append("required_main_pattern_missing")

    available_equipment = {
        str(item or "").strip().upper().replace("-", "_")
        for item in input_data.get("availableEquipment", [])
        if str(item or "").strip()
    }
    if available_equipment and any(
        (
            str(block.equipment_type or "").strip().upper().replace("-", "_")
            not in available_equipment
            and str(block.equipment_type or "").strip().upper().replace("-", "_")
            != "BODYWEIGHT"
        )
        for block in response.routine_blocks
        if str(block.equipment_type or "").strip()
    ):
        flags.append("available_equipment_whitelist_violation")

    family_ids: dict[str, list[str]] = {}
    pattern_ids: dict[str, list[str]] = {}
    reviewed_pull = _reviewed_pull_output_candidates(request, response, blueprint)
    for block in response.routine_blocks:
        candidate = {
            "id": block.exercise_id,
            "name_kr": block.exercise_name,
            "primary_muscle": (block.primary_muscles or [""])[0],
            "movement_type": block.movement_type,
            "equipment_req": block.equipment_type,
        }
        family = reviewed_pull_family_key(
            reviewed_pull.get(block.exercise_id, candidate),
            candidate_selection_family_key(candidate), enabled=bool(reviewed_pull),
        )
        family = focus_variant_family_key(
            candidate,
            request.specialization_focus if request.readiness_level != "low" else None,
            family,
        )
        pattern = candidate_selection_pattern_key(candidate)
        family_ids.setdefault(family, []).append(block.exercise_id)
        pattern_ids.setdefault(pattern, []).append(block.exercise_id)
    if any(len(ids) > 1 for ids in family_ids.values() if ids):
        flags.append("duplicate_movement_family")
    for pattern, ids in pattern_ids.items():
        limit = selection_pattern_limit(pattern, pattern_policy)
        if limit is not None and len(ids) > limit:
            flags.append(f"coarse_pattern_over_cap:{pattern}")

    pain_parts = {
        str(item.body_part or "").strip().lower().replace("_", "-")
        for item in request.pain_areas
        if str(item.body_part or "").strip()
    }
    target_parts = {
        str(item or "").strip().lower().replace("_", "-")
        for item in resolved_target_muscles
        if str(item or "").strip()
    }
    if "shoulder" in pain_parts:
        pain_parts.update({"front-deltoids", "back-deltoids", "side-deltoids"})
    if pain_parts & {"front-deltoids", "back-deltoids", "side-deltoids"}:
        pain_parts.add("shoulder")
    if "shoulder" in target_parts:
        target_parts.update({"front-deltoids", "back-deltoids", "side-deltoids"})
    if target_parts & {"front-deltoids", "back-deltoids", "side-deltoids"}:
        target_parts.add("shoulder")
    scope_warning_present = any(
        "직접 겹치는 목표 운동은 후보에서 제외" in str(warning)
        for warning in response.warnings
    )
    if (
        response.generation_status == "success"
        and pain_parts & target_parts
        and not scope_warning_present
    ):
        flags.append("missing_pain_target_scope_change_warning")
    return sorted(set(flags))


def _contains_any(text: str, tokens: list[str]) -> bool:
    return any(token.lower() in text for token in tokens)


def _total_working_sets(response: RoutineDraftResponse) -> int:
    return sum(
        1
        for block in response.routine_blocks
        for item in (block.prescription or [])
        if str(item.set_type or "working").lower() == "working"
    )


def _selected_id_text(response: RoutineDraftResponse) -> str:
    return " ".join(selected_ids(response)).lower()


def _has_equipment(response: RoutineDraftResponse, equipment: str) -> bool:
    token = equipment.lower()
    return any(token in str(block.equipment_type or "").lower() for block in response.routine_blocks)


def _has_primary(response: RoutineDraftResponse, primary: str) -> bool:
    expected = str(primary or "").strip().lower()
    return any(
        expected in {str(item).strip().lower() for item in (block.primary_muscles or [])}
        for block in response.routine_blocks
    )


def _has_neutral_grip_press(response: RoutineDraftResponse) -> bool:
    for block in response.routine_blocks:
        primaries = {
            str(item or "").strip().lower()
            for item in (block.primary_muscles or [])
        }
        if (
            str(block.movement_type or "").strip().upper() != "COMPOUND"
            or not (primaries & {"chest", "front-deltoids"})
        ):
            continue
        text = " ".join([
            block.exercise_id or "",
            block.exercise_name or "",
            block.exercise_rationale or "",
            block.movement_pattern or "",
        ]).lower()
        if _contains_any(text, ["neutral_grip", "뉴트럴", "중립 그립"]) and _contains_any(
            text,
            ["press", "프레스"],
        ):
            return True
    return False


def _has_direct_accessory_block(
    response: RoutineDraftResponse,
    *,
    primary: str | list[str],
    name_tokens: list[str],
) -> bool:
    accepted_primaries = {
        str(item).strip().lower()
        for item in ([primary] if isinstance(primary, str) else primary)
    }
    for block in response.routine_blocks:
        primaries = {str(item).strip().lower() for item in (block.primary_muscles or [])}
        if not (accepted_primaries & primaries):
            continue
        text = " ".join([
            block.exercise_id or "",
            block.exercise_name or "",
            block.exercise_rationale or "",
            block.movement_pattern or "",
        ]).lower()
        if _contains_any(text, name_tokens):
            return True
    return False


def _block_has_rule(block: Any, rule_code: str) -> bool:
    expected = str(rule_code or "").strip()
    if not expected:
        return False
    for item in list(getattr(block, "boosts", None) or []) + list(getattr(block, "penalties", None) or []):
        if str(getattr(item, "rule_code", "") or "") == expected:
            return True
    return False


def _has_training_style_match_evidence(response: RoutineDraftResponse, style: str) -> bool:
    normalized = str(style or "").strip().lower().replace("-", "_").replace(" ", "_")
    style_name_tokens = {
        "crossfit": [
            "crossfit",
            "크로스핏",
            "amrap",
            "emom",
            "스러스터",
            "thruster",
            "스내치",
            "snatch",
            "클린",
            "clean",
            "저크",
            "jerk",
            "월볼",
            "wall ball",
        ],
        "mobility": [
            "mobility",
            "모빌리티",
            "가동성",
            "정렬",
            "alignment",
            "range of motion",
            "매트",
            "체어",
        ],
        "yoga_pilates": [
            "yoga",
            "요가",
            "pilates",
            "필라테스",
            "reformer",
            "리포머",
            "운다체어",
            "매트",
            "블록",
            "미니볼",
        ],
        "rehab_prehab": [
            "prehab",
            "프리햅",
            "rehab",
            "재활",
            "저부하",
            "stability",
            "안정화",
        ],
    }
    tokens = style_name_tokens.get(normalized, [])
    for block in response.routine_blocks:
        if _block_has_rule(block, "training_style_match_boost"):
            return True
        block_text = " ".join([
            getattr(block, "exercise_id", "") or "",
            getattr(block, "exercise_name", "") or "",
            getattr(block, "movement_pattern", "") or "",
        ]).lower()
        if tokens and _contains_any(block_text, tokens):
            return True
    return False


def _first_primary_index(response: RoutineDraftResponse, primary: str | list[str]) -> int | None:
    accepted_primaries = {
        str(item).strip().lower()
        for item in ([primary] if isinstance(primary, str) else primary)
    }
    for index, block in enumerate(response.routine_blocks):
        primaries = {str(item).strip().lower() for item in (block.primary_muscles or [])}
        if accepted_primaries & primaries:
            return index
    return None


def _first_direct_accessory_index(
    response: RoutineDraftResponse,
    *,
    primary: str | list[str],
    name_tokens: list[str],
    require_isolation: bool = False,
) -> int | None:
    accepted_primaries = {
        str(item).strip().lower()
        for item in ([primary] if isinstance(primary, str) else primary)
    }
    for index, block in enumerate(response.routine_blocks):
        primaries = {str(item).strip().lower() for item in (block.primary_muscles or [])}
        name_text = " ".join([
            block.exercise_id or "",
            block.exercise_name or "",
            block.movement_pattern or "",
            _exercise_catalog_text(block.exercise_id or ""),
        ]).lower()
        display_text = " ".join([
            block.exercise_name or "",
            block.movement_pattern or "",
            _exercise_catalog_text(block.exercise_id or ""),
        ]).lower()
        direct_name_match = _contains_any(name_text, name_tokens)
        if not (accepted_primaries & primaries) and not (require_isolation and direct_name_match):
            continue
        if require_isolation and _contains_any(
            display_text,
            [
                "press",
                "프레스",
                "row",
                "로우",
                "pulldown",
                "랫풀다운",
                "pullup",
                "pull-up",
                "풀업",
                "squat",
                "스쿼트",
            ],
        ):
            continue
        text = " ".join([
            name_text,
            block.exercise_rationale or "",
        ]).lower()
        if direct_name_match or (not require_isolation and _contains_any(text, name_tokens)):
            return index
    return None


def _mentions_phase_role(text: str) -> bool:
    return _contains_any(
        text,
        [
            "메인 운동",
            "주요 운동",
            "복합 운동",
            "보조 운동",
            "직접 보조",
            "보조근",
            "마무리 운동",
            "대체 운동",
            "main movement",
            "accessory",
        ],
    )


def derived_preferred_tags(row: dict[str, Any], response: RoutineDraftResponse) -> list[str]:
    tags: list[str] = []
    scenario_id = str(row.get("scenarioId") or "")
    text = text_blob(response)
    user_text = user_visible_text_blob(response)
    ids = _selected_id_text(response)
    input_data = row.get("input", {}) or {}
    request = request_from_seed(row)
    if request.injury_conditions:
        selected_overlap = False
        for exercise_id in selected_ids(response):
            candidate = _exercise_policy_candidate(exercise_id)
            if candidate is not None and candidate_injury_condition_overlaps(
                candidate,
                request.injury_conditions,
            ):
                selected_overlap = True
                break
        if not selected_overlap:
            tags.append("managed_injury_selection_safe")

        guarded_conditions = [
            condition
            for condition in request.injury_conditions
            if condition.status in {"activeUncleared", "managedWithRestrictions"}
        ]
        area_tokens = {
            condition.area_code: [
                injury_area_label_ko(condition.area_code),
                condition.area_code,
            ]
            for condition in guarded_conditions
        }
        if guarded_conditions and all(
            _contains_any(user_text, tokens)
            for tokens in area_tokens.values()
        ):
            tags.append("managed_injury_context_reason")
    profile_signals = {
        str(value).strip()
        for value in (input_data.get("profileSignals") or input_data.get("profile_signals") or [])
        if str(value).strip()
    }
    profile_reason_tokens = {
        "heel_down_deep_squat_difficult": (
            "ankle_mobility_reason",
            ["발뒤꿈치", "깊은 앉기", "발목", "배굴"],
        ),
        "hip_hinge_difficult": (
            "hip_hinge_difficulty_reason",
            ["엉덩이를 접어", "힙힌지", "고관절 굴곡"],
        ),
        "overhead_reach_difficult": (
            "overhead_reach_difficulty_reason",
            ["팔을 머리 위", "오버헤드", "어깨 굴곡"],
        ),
        "loaded_wrist_extension_difficult": (
            "wrist_extension_difficulty_reason",
            ["손목을 뒤로", "손목 신전", "손목 가동"],
        ),
        "self_reported_long_arms": (
            "long_arm_profile_reason",
            ["팔이 유난히 긴", "긴 팔", "팔 길이 신호"],
        ),
        "self_reported_long_thighs": (
            "long_thigh_profile_reason",
            ["허벅지 구간이 유난히 긴", "긴 허벅지", "대퇴 길이 신호"],
        ),
    }
    for signal_code, (tag, tokens) in profile_reason_tokens.items():
        if signal_code in profile_signals and _contains_any(user_text, tokens):
            tags.append(tag)
    requested_structure = str(
        input_data.get("sessionStructure") or input_data.get("session_structure") or "standard"
    ).strip().lower()
    group_types = [
        str(block.execution_group_type or "single").strip().lower()
        for block in response.routine_blocks
    ]
    if requested_structure == "standard" and group_types and all(item == "single" for item in group_types):
        tags.append("execution_standard_all_single")
    if requested_structure in {"superset", "circuit", "density"} and requested_structure in group_types:
        tags.append(f"execution_{requested_structure}_grouped")
    main_blocks = [
        block
        for block in response.routine_blocks
        if block.session_phase in {"power_if_explicit", "main_compound"}
    ]
    if main_blocks and all(block.execution_group_type == "single" for block in main_blocks):
        tags.append("main_lifts_remain_single")

    if requested_structure == "superset":
        by_group: dict[str, set[str]] = {}
        for block in response.routine_blocks:
            if block.execution_group_type != "superset" or not block.execution_group_id:
                continue
            by_group.setdefault(block.execution_group_id, set()).update(block.primary_muscles or [])
        if any("chest" in muscles and bool(muscles & {"upper-back", "trapezius"}) for muscles in by_group.values()):
            tags.append("antagonist_main_superset")

    requested_adjuncts = list(
        input_data.get("adjunctOptions") or input_data.get("adjunct_options") or []
    )
    contract_valid = not adjunct_contract_violations(row, response)
    for requested in dict.fromkeys(
        str(option).strip().lower() for option in requested_adjuncts if str(option).strip()
    ):
        if contract_valid and any(
            str(block.adjunct_option or "").strip().lower() == requested
            and adjunct_block_matches_profile(block, requested)
            for block in response.routine_blocks
        ):
            tags.extend(["requested_adjunct_applied", f"adjunct_{requested}_applied"])

    if scenario_id.startswith(("sens-", "gen-sens-")):
        expected_tokens = [str(token) for token in row.get("expectedSignalTokens", []) if str(token).strip()]
        if any(_contains_any(text, [token]) for token in expected_tokens):
            tags.append("sensitivity_signal_observed")
        if response.generation_status == "success" and not response.is_fallback:
            tags.append("live_success")
        if not derived_language_quality_flags(response, row):
            tags.append("language_quality_ok")
        style = str(row.get("input", {}).get("trainingStyle") or row.get("input", {}).get("training_style") or "").strip().lower().replace("-", "_")
        if style and _has_training_style_match_evidence(response, style):
            tags.append(f"{style}_style_match")
        if scenario_id == "gen-sens-push-targetmuscles-chest-triceps-043":
            if _has_primary(response, "chest"):
                tags.append("chest_main")
            if _first_direct_accessory_index(
                response,
                primary="triceps",
                name_tokens=["pushdown", "푸시다운", "푸쉬다운", "extension", "익스텐션", "triceps", "삼두"],
                require_isolation=True,
            ) is not None:
                tags.append("direct_triceps_accessory")
        if scenario_id == "gen-sens-pull-targetmuscles-upper-back-biceps-065":
            if _has_primary(response, "upper-back") or _has_primary(response, "lower-back"):
                tags.append("back_main")
            if _first_direct_accessory_index(
                response,
                primary=["biceps", "forearm"],
                name_tokens=["curl", "컬", "biceps", "이두"],
                require_isolation=True,
            ) is not None:
                tags.append("direct_biceps_accessory")

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
        if _contains_any(text, ["ankle", "dorsiflexion", "limited_ankle_dorsiflexion", "발목", "배굴", "발뒤꿈치", "깊은 앉기"]):
            tags.append("ankle_mobility_reason")

    elif scenario_id == "shoulder-push-001":
        if _contains_any(ids + " " + text, ["machine_chest_press", "머신 체스트 프레스"]):
            tags.append("machine_chest_press")
        if _contains_any(ids + " " + text, ["cable_press", "케이블 프레스"]):
            tags.append("cable_press")
        if _has_neutral_grip_press(response):
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

    elif scenario_id == "push-accessory-coverage-001":
        if _has_primary(response, "chest"):
            tags.append("chest_main")
        triceps_index = _first_direct_accessory_index(
            response,
            primary="triceps",
            name_tokens=["pushdown", "푸시다운", "푸쉬다운", "extension", "익스텐션", "triceps", "삼두"],
            require_isolation=True,
        )
        if triceps_index is not None:
            tags.append("direct_triceps_accessory")
        chest_index = _first_primary_index(response, "chest")
        if chest_index is not None and triceps_index is not None and chest_index < triceps_index:
            tags.append("main_before_accessory")
        if _contains_any(text, ["보조근", "삼두", "triceps", "split_accessory_direct_coverage_boost"]):
            tags.append("accessory_coverage_reason")
        if _mentions_phase_role(text):
            tags.append("phase_role_rationale")
        if _has_equipment(response, "machine") or _has_equipment(response, "cable") or _has_equipment(response, "dumbbell"):
            tags.append("practical_gym_option")

    elif scenario_id == "pull-accessory-coverage-001":
        if _has_primary(response, "upper-back") or _has_primary(response, "lower-back"):
            tags.append("back_main")
        biceps_index = _first_direct_accessory_index(
            response,
            primary=["biceps", "forearm"],
            name_tokens=["curl", "컬", "biceps", "이두"],
            require_isolation=True,
        )
        if biceps_index is not None:
            tags.append("direct_biceps_accessory")
        back_index = _first_primary_index(response, ["upper-back", "lower-back"])
        if back_index is not None and biceps_index is not None and back_index < biceps_index:
            tags.append("main_before_accessory")
        if _contains_any(text, ["보조근", "이두", "biceps", "split_accessory_direct_coverage_boost"]):
            tags.append("accessory_coverage_reason")
        if _mentions_phase_role(text):
            tags.append("phase_role_rationale")
        if _has_equipment(response, "machine") or _has_equipment(response, "cable") or _has_equipment(response, "dumbbell"):
            tags.append("practical_gym_option")

    elif scenario_id == "beginner-lower-skill-001":
        if _has_equipment(response, "machine") or _contains_any(text, ["stable", "안정", "bodyweight"]):
            tags.append("machine_or_stable_option")
        if _contains_any(text, ["technique", "difficulty", "난이도", "기술", "failure_penalty"]):
            tags.append("technique_risk_reason")
        if _contains_any(text, ["progression", "regression", "단계", "점진"]):
            tags.append("progression_path")

    elif scenario_id.startswith(("option-policy-lower-back-acute", "option-matrix-lower-back-injury")):
        if "leg_curl" in ids or "레그 컬" in text or "레그컬" in text:
            tags.append("leg_curl")
        if "hip_thrust" in ids or "힙 쓰러스트" in text:
            tags.append("hip_thrust")
        if _contains_any(text, ["lumbar", "lower-back", "허리", "요추"]):
            tags.append("lower_lumbar_alternative")
        if _has_equipment(response, "machine") or _contains_any(text, ["machine", "머신", "stable", "안정", "supported", "지지"]):
            tags.append("machine_or_stable_option")

    elif scenario_id.startswith("option-policy-lower-back-mild"):
        if _contains_any(text, ["option_policy_warning", "mild_direct_discomfort_requires_downgrade", "보수", "줄여", "낮춰"]):
            tags.append("mild_discomfort_downgrade")
        if _contains_any(text, ["lumbar", "lower-back", "허리", "요추"]):
            tags.append("lower_back_reason")
        if "leg_curl" in ids or "레그 컬" in text or "hip_thrust" in ids or "힙 쓰러스트" in text:
            tags.append("lower_lumbar_alternative")

    elif scenario_id.startswith(("option-policy-shoulder-not-cleared", "option-matrix-shoulder-not-cleared")):
        if _contains_any(text, ["medical", "clearance", "의료진", "허가", "전문가"]):
            tags.append("medical_clearance_notice")
        if _contains_any(text, ["alternative", "substitution", "대체", "후보"]):
            tags.append("safe_alternative_candidate")
        if _contains_any(text, ["shoulder", "front-deltoids", "어깨", "전면 삼각근"]):
            tags.append("shoulder_context_reason")
        if _contains_any(text, ["neutral", "supported", "중립", "지지", "등받이", "머신"]):
            tags.append("neutral_grip_or_supported_press")

    elif scenario_id.startswith(("option-policy-knee-active", "option-matrix-knee-injury")):
        if _contains_any(text, ["knee_shear", "knees", "무릎"]):
            tags.append("low_knee_shear_option")
        if _has_equipment(response, "machine") or _contains_any(text, ["stable", "안정", "bodyweight", "머신"]):
            tags.append("machine_or_stable_option")
        if _contains_any(text, ["무릎", "knee", "주의", "부하"]):
            tags.append("knee_policy_guard")

    elif scenario_id.startswith(("option-policy-beginner-ballistic", "option-matrix-beginner-ballistic")):
        if _has_equipment(response, "machine") or _contains_any(text, ["stable", "안정", "고정식", "bodyweight", "맨몸", "머신"]):
            tags.append("machine_or_stable_option")
        if _contains_any(text, ["beginner", "초급", "초보", "입문", "낮은 난이도", "안정", "실패 부담"]):
            tags.append("beginner_reason")
        if _contains_any(text, ["progression", "regression", "단계", "점진"]):
            tags.append("progression_path")

    elif scenario_id.startswith("legs-exp-beginner"):
        if _has_equipment(response, "machine") or _contains_any(text, ["stable", "안정", "고정식", "bodyweight", "맨몸"]):
            tags.append("machine_or_stable_option")
        if _contains_any(text, ["mainstream gym", "헬스장", "보편", "익숙", "familiar", "common", "machine", "머신"]):
            tags.append("common_gym_option")
        if _contains_any(text, ["beginner", "초급", "초보", "입문", "낮은 난이도", "안정", "실패 부담", "낮은 기술"]):
            tags.append("beginner_reason")

    elif scenario_id.startswith("legs-exp-intermediate"):
        if _contains_any(text, ["mainstream gym", "헬스장", "보편", "익숙", "common", "machine", "머신", "스미스"]):
            tags.append("common_gym_option")
        if _contains_any(text, ["intermediate", "중급", "점진", "과부하", "progressive", "main movement", "메인 운동", "복합 운동"]):
            tags.append("intermediate_reason")

    elif scenario_id.startswith("legs-exp-advanced"):
        if _contains_any(text, ["advanced", "상급", "고급", "기술", "고효과", "효과가 높은", "복합 운동", "variation", "변형"]):
            tags.append("advanced_variation_reason")
        if _contains_any(text, ["effect", "risk", "효과", "위험", "부하", "컨디션", "readiness", "tradeoff", "균형"]):
            tags.append("effect_risk_tradeoff")

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
        if _contains_any(text, ["ankle", "dorsiflexion", "limited_ankle_dorsiflexion", "발목", "배굴", "발뒤꿈치", "깊은 앉기"]):
            tags.append("ankle_mobility_reason")
        if _contains_any(text, ["technique", "difficulty", "난이도", "기술", "failure_penalty"]):
            tags.append("technique_risk_reason")

    elif scenario_id == "wrist-limited-push-001":
        if _contains_any(text, ["wrist", "wrist_extension", "forearm", "손목", "전완"]):
            tags.append("wrist_context_reason")
        if _has_neutral_grip_press(response):
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

    elif scenario_id == "holdout-beginner-legs-machine-001":
        if _has_equipment(response, "machine") or _contains_any(text, ["stable", "안정", "고정식", "머신"]):
            tags.append("machine_or_stable_option")
        if _contains_any(user_text, ["beginner", "초급", "초보", "입문", "낮은 난이도", "안정", "실패 부담"]):
            tags.append("beginner_reason")
        first = response.routine_blocks[0] if response.routine_blocks else None
        if first and set(first.primary_muscles or []) & {"quadriceps", "hamstring", "gluteal"}:
            tags.append("large_lower_main_first")

    elif scenario_id == "holdout-low-readiness-full-body-001":
        if _total_working_sets(response) <= 12:
            tags.append("lower_volume")
        if _has_equipment(response, "machine") or _has_equipment(response, "dumbbell") or _has_equipment(response, "bodyweight"):
            tags.append("machine_or_stable_options")
        if _contains_any(user_text, ["readiness", "컨디션", "low", "낮", "피로"]):
            tags.append("readiness_reason")

    elif scenario_id == "holdout-ankle-limited-legs-001":
        if _contains_any(user_text, ["ankle", "dorsiflexion", "limited_ankle_dorsiflexion", "발목", "배굴", "발뒤꿈치", "깊은 앉기"]):
            tags.append("ankle_mobility_reason")
        if _has_equipment(response, "machine") or _contains_any(text, ["stable", "안정", "머신", "레그 프레스"]):
            tags.append("machine_or_stable_option")

    elif scenario_id == "holdout-lower-back-pain-legs-001":
        if _contains_any(user_text, ["lumbar", "lower-back", "허리", "요추"]):
            tags.append("lower_back_reason")
        if "leg_curl" in ids or "레그 컬" in text or "힙 쓰러스트" in text or "hip_thrust" in ids:
            tags.append("lower_lumbar_alternative")
        if _has_equipment(response, "machine") or _contains_any(text, ["supported", "지지", "등받이", "머신"]):
            tags.append("machine_or_stable_option")

    elif scenario_id.endswith("shoulder-wrist-push-001"):
        if _contains_any(user_text, ["shoulder", "front-deltoids", "어깨", "전면 삼각근"]):
            tags.append("shoulder_context_reason")
        if _contains_any(user_text, ["wrist", "wrist_extension", "forearm", "손목", "전완"]):
            tags.append("wrist_context_reason")
        if _has_primary(response, "chest"):
            tags.append("chest_main")
        triceps_index = _first_direct_accessory_index(
            response,
            primary="triceps",
            name_tokens=["pushdown", "푸시다운", "푸쉬다운", "extension", "익스텐션", "triceps", "삼두"],
            require_isolation=True,
        )
        if triceps_index is not None:
            tags.append("direct_triceps_accessory")
        chest_index = _first_primary_index(response, "chest")
        if chest_index is not None and triceps_index is not None and chest_index < triceps_index:
            tags.append("main_before_accessory")

    elif scenario_id.endswith("upper-back-doms-pull-accessory-001"):
        if _has_primary(response, "upper-back") or _has_primary(response, "lower-back") or _has_primary(response, "trapezius"):
            tags.append("back_main")
        biceps_index = _first_direct_accessory_index(
            response,
            primary=["biceps", "forearm"],
            name_tokens=["curl", "컬", "biceps", "이두"],
            require_isolation=True,
        )
        if biceps_index is not None:
            tags.append("direct_biceps_accessory")
        back_index = _first_primary_index(response, ["upper-back", "lower-back", "trapezius"])
        if back_index is not None and biceps_index is not None and back_index < biceps_index:
            tags.append("main_before_accessory")
        if _contains_any(user_text, ["doms", "근육통", "볼륨", "세트 수", "세트수", "줄였", "조절"]):
            tags.append("doms_volume_reason")
        if _total_working_sets(response) <= 10:
            tags.append("doms_volume_controlled")

    elif scenario_id.endswith("cable-dumbbell-full-body-001"):
        has_lower = any(set(block.primary_muscles or []) & {"quadriceps", "hamstring", "gluteal"} for block in response.routine_blocks)
        has_push = any(set(block.primary_muscles or []) & {"chest", "front-deltoids"} for block in response.routine_blocks)
        has_pull = any(set(block.primary_muscles or []) & {"upper-back", "lower-back", "trapezius"} for block in response.routine_blocks)
        if has_lower and has_push and has_pull:
            tags.append("full_body_lower_push_pull_coverage")
        if _contains_any(user_text, ["equipment", "장비", "덤벨", "케이블", "맨몸"]):
            tags.append("equipment_context_reason")

    elif scenario_id.endswith("advanced-high-upper-strength-001"):
        has_push = any(set(block.primary_muscles or []) & {"chest", "front-deltoids"} for block in response.routine_blocks)
        has_pull = any(set(block.primary_muscles or []) & {"upper-back", "lower-back", "trapezius"} for block in response.routine_blocks)
        if has_push and has_pull:
            tags.append("upper_push_pull_coverage")
        if _contains_any(user_text, ["advanced", "상급", "근력", "strength"]):
            tags.append("advanced_strength_reason")
        if _contains_any(user_text, ["readiness", "컨디션", "high", "높"]):
            tags.append("readiness_reason")
        if _contains_any(user_text, ["효과", "effect"]) and _contains_any(user_text, ["위험", "주의", "부하", "risk"]):
            tags.append("effect_risk_tradeoff")

    return sorted(set(tags))


def summarize_rationale(response: RoutineDraftResponse) -> str:
    parts: list[str] = []
    parts.extend(response.rationale_summary or [])
    for block in response.routine_blocks:
        if block.exercise_rationale:
            label = block.exercise_name or block.exercise_id
            parts.append(f"{label}: {block.exercise_rationale}")
    return " | ".join(parts)


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


def summarize_telemetry_event(event: dict[str, Any] | None) -> dict[str, Any] | None:
    if not event:
        return None
    summary: dict[str, Any] = {}
    for key in TELEMETRY_SUMMARY_KEYS:
        if key not in event:
            continue
        value = event.get(key)
        if value is None or value == "" or value == []:
            continue
        summary[key] = value
    return summary or None


def latest_telemetry_summary(events: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if not events:
        return None
    for event in reversed(events):
        summary = summarize_telemetry_event(event)
        if summary:
            return summary
    return None


def telemetry_note(summary: dict[str, Any] | None) -> str:
    if not summary:
        return ""
    parts: list[str] = []
    for key in (
        "fallback_reason",
        "parse_failure_subtype",
        "schema_validation_error_category",
        "repair_failure_reason",
        "raw_output_recovery_failed_reason",
        "json_decode_error_category",
        "local_raw_json_invoke_failed_reason",
    ):
        value = summary.get(key)
        if value:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def result_row(
    row: dict[str, Any],
    response: RoutineDraftResponse,
    elapsed_ms: int,
    telemetry_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    tags = derived_quality_tags(row, response)
    preferred_tags = derived_preferred_tags(row, response)
    language_flags = derived_language_quality_flags(response, row)
    telemetry_summary = latest_telemetry_summary(telemetry_events)
    output_quality_flags = derived_output_quality_flags(
        row,
        response,
        telemetry_summary=telemetry_summary,
    )
    failure_patterns = {str(item) for item in row.get("failurePatterns", [])}
    hard_hits = [tag for tag in tags if tag in failure_patterns]
    adjunct_contract_errors = adjunct_contract_violations(row, response)
    adjunct_contract_valid = not adjunct_contract_errors
    exercise_count = len(response.routine_blocks)
    return {
        "scenarioId": row["scenarioId"],
        "title": row["title"],
        "generationStatus": response.generation_status,
        "statusReasonCode": response.status_reason_code,
        "isFallback": response.is_fallback,
        "totalEstimatedTime": response.total_estimated_time,
        "elapsedMs": elapsed_ms,
        "exerciseCount": exercise_count,
        "selectedExerciseIds": selected_ids(response),
        "blockSummaries": [
            {
                "order": block.order,
                "exerciseId": block.exercise_id,
                "exerciseName": block.exercise_name,
                "adjunctOption": block.adjunct_option,
                "movementType": block.movement_type,
                "sessionPhase": block.session_phase,
                "primaryMuscles": list(block.primary_muscles or []),
                "equipmentType": block.equipment_type,
                "executionGroupType": block.execution_group_type,
                "executionGroupId": block.execution_group_id,
                "executionGroupPosition": block.execution_group_position,
                "executionGroupSize": block.execution_group_size,
                "transitionRestSec": block.transition_rest_sec,
                "groupRestSec": block.group_rest_sec,
                "defaultRestSec": block.default_rest_sec,
                "trainingMethod": (
                    block.training_method.model_dump(
                        by_alias=True,
                        exclude_none=True,
                    )
                    if block.training_method is not None
                    else None
                ),
                "prescription": [
                    {
                        "setIndex": set_item.set_index,
                        "setType": set_item.set_type,
                        "targetReps": set_item.target_reps,
                        "targetWeightKg": set_item.target_weight_kg,
                        "targetRir": set_item.target_rir,
                        "targetRestSec": set_item.target_rest_sec,
                        "targetDurationSec": set_item.target_duration_sec,
                        "intervalWorkSec": set_item.interval_work_sec,
                        "intervalRecoverySec": set_item.interval_recovery_sec,
                        "intervalRounds": set_item.interval_rounds,
                    }
                    for set_item in block.prescription
                ],
            }
            for block in response.routine_blocks
        ],
        "adjunctDecisions": [
            {
                "option": decision.option,
                "status": decision.status,
                "reasonCode": decision.reason_code,
                "slotId": decision.slot_id,
                "exerciseId": decision.exercise_id,
            }
            for decision in response.adjunct_decisions
        ],
        "personalizationSuggestions": [
            suggestion.model_dump(by_alias=True, exclude_none=True)
            for suggestion in response.personalization_suggestions
        ],
        "adjunctContractViolations": adjunct_contract_errors,
        "rationaleText": summarize_rationale(response),
        "scoreEvidenceText": summarize_score_evidence(response),
        "warnings": response.warnings,
        "languageQualityFlags": language_flags,
        "outputQualityFlags": output_quality_flags,
        "productiveTimeUtilizationPct": productive_time_utilization_pct(
            response.total_estimated_time,
            int((row.get("input", {}) or {}).get("timeAvailableMin") or 60),
        ),
        "telemetryEventCount": len(telemetry_events or []),
        "telemetrySummary": telemetry_summary,
        "underfillDiagnostics": dict(response._underfill_diagnostics or {}),
        "contractValid": adjunct_contract_valid,
        "hardViolationCount": len(hard_hits),
        "notes": " ".join([
            f"generationStatus={response.generation_status}",
            f"statusReasonCode={response.status_reason_code}",
            f"isFallback={response.is_fallback}",
            f"totalEstimatedTime={response.total_estimated_time}",
            f"elapsedMs={elapsed_ms}",
            " ".join(tags),
            " ".join(preferred_tags),
            " ".join(language_flags),
            " ".join(output_quality_flags),
            telemetry_note(telemetry_summary),
        ]).strip(),
    }


def build_payload(
    *,
    run_id: str,
    seed_path: Path,
    rows: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    provenance: dict[str, Any] | None = None,
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
        "provenance": provenance or {},
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
    provenance = build_provenance(
        runner="scripts/run_gemma4_quality_seed_live.py",
        input_paths=[seed_path],
        model=os.getenv("LOCAL_LLM_MODEL") or "gemma4:latest",
    )
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
    captured_telemetry: list[dict[str, Any]] = []
    original_emit_telemetry = routine_telemetry.emit_routine_quality_telemetry
    original_telemetry_env = os.environ.get(routine_telemetry.ENV_RULE_CRITIC_TELEMETRY)

    def capture_telemetry(payload: dict[str, Any]) -> None:
        captured_telemetry.append(dict(payload))

    os.environ[routine_telemetry.ENV_RULE_CRITIC_TELEMETRY] = "true"
    routine_telemetry.emit_routine_quality_telemetry = capture_telemetry
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
            telemetry_start_index = len(captured_telemetry)
            try:
                with scenario_timeout(scenario_timeout_sec):
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        response = generate_smart_routine(req, db, profile=profile, recent_sets=recent_sets)
                elapsed_ms = round((time.perf_counter() - started) * 1000)
                scenario_telemetry = captured_telemetry[telemetry_start_index:]
                scenarios.append(result_row(row, response, elapsed_ms, scenario_telemetry))
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
                telemetry_summary = latest_telemetry_summary(captured_telemetry[telemetry_start_index:])
                failures.append({
                    "scenarioId": scenario_id,
                    "errorType": type(exc).__name__,
                    "elapsedMs": elapsed_ms,
                    "telemetrySummary": telemetry_summary,
                })
                scenarios.append({
                    "scenarioId": scenario_id,
                    "title": row["title"],
                    "generationStatus": "error",
                    "statusReasonCode": "liveRunnerError",
                    "isFallback": False,
                    "totalEstimatedTime": None,
                    "elapsedMs": elapsed_ms,
                    "exerciseCount": 0,
                    "selectedExerciseIds": [],
                    "rationaleText": "",
                    "warnings": [type(exc).__name__],
                    "contractValid": False,
                    "hardViolationCount": 1,
                    "telemetryEventCount": len(captured_telemetry[telemetry_start_index:]),
                    "telemetrySummary": telemetry_summary,
                    "notes": " ".join([
                        f"live_runner_error errorType={type(exc).__name__}",
                        f"elapsedMs={elapsed_ms}",
                        telemetry_note(telemetry_summary),
                    ]).strip(),
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
                        provenance=provenance,
                    ),
                )
            if cooldown_sec > 0 and index < len(rows):
                time.sleep(cooldown_sec)
    finally:
        routine_telemetry.emit_routine_quality_telemetry = original_emit_telemetry
        if original_telemetry_env is None:
            os.environ.pop(routine_telemetry.ENV_RULE_CRITIC_TELEMETRY, None)
        else:
            os.environ[routine_telemetry.ENV_RULE_CRITIC_TELEMETRY] = original_telemetry_env
        db.close()

    return build_payload(
        run_id=run_id,
        seed_path=seed_path,
        rows=rows,
        scenarios=scenarios,
        failures=failures,
        provenance=provenance,
    )


def assert_no_forbidden_tokens(payload: dict[str, Any]) -> None:
    seed_runner.assert_no_forbidden_tokens(payload)


def run_preflight_warmup(
    *,
    base_url: str,
    model: str,
    timeout_sec: float,
    warmup_timeout_sec: float,
    warmup_num_predict: int,
    output_path: Path | None = None,
) -> dict[str, Any]:
    report = local_readiness.build_readiness_report(
        base_url=base_url,
        model=model,
        timeout_sec=timeout_sec,
        probe=True,
        warmup=True,
        warmup_timeout_sec=warmup_timeout_sec,
        warmup_num_predict=warmup_num_predict,
    )
    if output_path:
        local_readiness._write_json_if_requested(str(output_path), report)
    print(json.dumps({
        "status": "preflight",
        "readinessStatus": report.get("readiness_status"),
        "warmupLatencyMs": report.get("warmup_latency_ms"),
        "probeLatencyMs": report.get("probe_latency_ms"),
        "nextAction": report.get("next_action"),
    }, ensure_ascii=False), flush=True)
    if report.get("readiness_status") != "pass":
        raise SystemExit(2)
    return report


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
    parser.add_argument("--preflight-warmup", action="store_true", help="Run local Ollama warmup readiness before live scenarios.")
    parser.add_argument("--preflight-timeout-sec", type=float, default=30)
    parser.add_argument("--preflight-warmup-timeout-sec", type=float, default=180)
    parser.add_argument("--preflight-warmup-num-predict", type=int, default=1)
    args = parser.parse_args()

    apply_local_llm_defaults()
    run_id = sanitize_run_id(args.run_id)
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "gemma4-quality-seed-results-filled.json"
    if args.preflight_warmup:
        run_preflight_warmup(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            model=os.getenv("LOCAL_LLM_MODEL", "gemma4:latest"),
            timeout_sec=args.preflight_timeout_sec,
            warmup_timeout_sec=args.preflight_warmup_timeout_sec,
            warmup_num_predict=args.preflight_warmup_num_predict,
            output_path=output_dir / "local-llm-readiness-preflight.json",
        )
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

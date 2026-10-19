"""Deterministic quality critic for generated routine drafts.

V1 is observe-only: it computes score, grade, warnings, and metric scores
without changing pipeline behavior or triggering rebuild/fallback.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from engines.prescription.adjustments import (
    _calculate_max_total_sets,
    minimum_productive_duration_min,
    target_exercise_count_for_session,
    target_productive_duration_min,
)
from engines.prescription.adjuncts import (
    adjunct_block_matches_profile,
    count_main_blocks,
    validate_adjunct_decision_contract,
)
from engines.candidate_ranker import _candidate_injury_overlap, _pain_tokens_from_entries
from engines.option_resolver import (
    ADJUNCT_OPTION_PHASE,
    candidate_matches_adjunct_option,
)
from engines.routine_blueprint import build_routine_blueprint
from engines.routine_selection import (
    required_main_pattern_slot_specs,
    required_main_pattern_targets,
    routine_items_have_required_main_pattern,
)
from engines.schemas import PainAreaEntry, RecentSetRecord, RoutineDraftResponse, RoutineRequest

logger = logging.getLogger(__name__)


@dataclass
class RoutineEvalContext:
    goal: str
    target_muscles: List[str]
    duration_min: int
    readiness_level: str
    available_equipment: List[str]
    unavailable_equipment: List[str]
    current_pain_areas: List[PainAreaEntry]
    doms: Dict[str, int]
    recent_exercises: List[str]
    max_working_sets: Optional[int] = None
    target_main_exercise_count: Optional[int] = None
    minimum_main_exercise_count: Optional[int] = None
    adjunct_options: List[str] = field(default_factory=list)
    required_main_pattern_targets: List[str] = field(default_factory=list)
    required_main_pattern_slot_specs: List[dict] = field(default_factory=list)
    overrides: dict = field(default_factory=dict)

    @property
    def time_available_min(self) -> int:
        return self.duration_min

    @property
    def blocked_equipment(self) -> List[str]:
        return self.unavailable_equipment

    @property
    def pain_areas(self) -> List[PainAreaEntry]:
        return self.current_pain_areas

    @property
    def doms_db(self) -> Dict[str, int]:
        return self.doms

    @property
    def recent_exercise_ids(self) -> List[str]:
        return self.recent_exercises


@dataclass
class RuleCriticResult:
    score: int
    grade: str
    hard_violations: List[str]
    warnings: List[str]
    metric_scores: Dict[str, int]
    should_rebuild: bool
    should_fallback: bool
    notes: List[str] = field(default_factory=list)


def build_eval_context(
    req: RoutineRequest,
    db_target_muscles: List[str],
    recent_sets: Optional[List[RecentSetRecord]] = None,
    max_working_sets: Optional[int] = None,
) -> RoutineEvalContext:
    recent_ids = [
        record.exercise_id
        for record in (recent_sets or [])
        if record.exercise_id
    ]
    blueprint = build_routine_blueprint(
        target_split_label=req.target_split_label,
        target_muscles=db_target_muscles,
        goal=req.goal,
        experience_level=req.experience_level,
        advanced_chest_focus=req.advanced_chest_focus,
        specialization_focus=req.specialization_focus,
        training_method=req.training_method,
        method_mode=req.method_mode,
        requested_method=req.requested_method,
        recent_training_method_uses=req.recent_training_method_uses,
        readiness_level=req.readiness_level,
        time_available_min=req.time_available_min,
        training_style=req.training_style,
        session_structure=req.session_structure,
        adjunct_options=req.adjunct_options,
        pain_areas=req.pain_areas,
        doms_data=req.doms_data,
        injury_conditions=req.injury_conditions,
    )
    return RoutineEvalContext(
        goal=req.goal or "hypertrophy",
        target_muscles=list(db_target_muscles or req.target_muscles or []),
        duration_min=req.time_available_min,
        readiness_level=req.readiness_level or "normal",
        available_equipment=list(req.available_equipment or []),
        unavailable_equipment=list(req.equipment or []),
        current_pain_areas=list(req.pain_areas or []),
        doms=dict(req.doms_data or {}),
        recent_exercises=recent_ids,
        max_working_sets=max_working_sets,
        target_main_exercise_count=blueprint.target_main_exercise_count,
        minimum_main_exercise_count=blueprint.minimum_main_exercise_count,
        adjunct_options=list(blueprint.planned_adjunct_options),
        required_main_pattern_targets=list(required_main_pattern_targets(blueprint)),
        required_main_pattern_slot_specs=[
            {
                "target_muscles": list(spec["target_muscles"]),
                "preferred_movement_types": list(spec["preferred_movement_types"]),
            }
            for spec in required_main_pattern_slot_specs(blueprint)
        ],
    )


def _candidate_lookup(candidates: List[dict]) -> Dict[str, dict]:
    return {str(candidate.get("id")): candidate for candidate in candidates}


def _block_ids(blocks: List[dict]) -> List[str]:
    return [str(block.get("exercise_id") or "") for block in blocks]


def _primary_muscles(block: dict, candidates_by_id: Dict[str, dict]) -> List[str]:
    muscles = block.get("primary_muscles") or []
    if muscles:
        return [str(muscle) for muscle in muscles]
    candidate = candidates_by_id.get(str(block.get("exercise_id") or ""))
    if candidate and candidate.get("primary_muscle"):
        return [str(candidate["primary_muscle"])]
    return []


def _movement_type(block: dict, candidates_by_id: Dict[str, dict]) -> str:
    candidate = candidates_by_id.get(str(block.get("exercise_id") or ""))
    return str(candidate.get("movement_type") or "UNKNOWN").upper() if candidate else "UNKNOWN"


def _equipment_tokens(candidate: dict) -> set[str]:
    return {
        token.strip().upper()
        for token in str(candidate.get("equipment_req") or "").split(",")
        if token.strip()
    }


def _total_working_sets(blocks: List[dict]) -> int:
    return sum(
        1
        for block in blocks
        for item in (block.get("prescription") or [])
        if str(item.get("set_type") or item.get("setType") or "working").lower()
        in {"working", "distance"}
    )


def _prescription_contract_violations(
    block: dict,
    candidate: dict,
    context: RoutineEvalContext,
) -> List[str]:
    """Reject unit/phase drift that would make a block unsafe or unrenderable."""

    if not candidate.get("prescription_mode"):
        return []

    exercise_id = str(block.get("exercise_id") or "")
    phase = str(block.get("session_phase") or "").strip()
    prescriptions = list(block.get("prescription") or [])
    violations: List[str] = []
    requested_adjunct_match = any(
        phase == ADJUNCT_OPTION_PHASE.get(option)
        and candidate_matches_adjunct_option(candidate, option)
        for option in context.adjunct_options
    )

    if not phase:
        violations.append(f"[PRESCRIPTION_CONTRACT] session phase missing: {exercise_id}")
    elif phase in {"finisher_or_recovery", "cooldown"}:
        violations.append(
            f"[PRESCRIPTION_CONTRACT] legacy merged session phase present in {exercise_id}: {phase}"
        )
    elif not requested_adjunct_match:
        allowed_phases = {
            token.strip()
            for token in str(candidate.get("allowed_session_phases") or "").split(",")
            if token.strip()
        }
        if allowed_phases and phase not in allowed_phases:
            violations.append(
                f"[PRESCRIPTION_CONTRACT] phase not allowed for {exercise_id}: "
                f"actual={phase}, allowed={sorted(allowed_phases)}"
            )

    expected_mode = str(candidate.get("prescription_mode") or "reps").strip().lower()
    expected_types = {
        "reps": {"working", "warmup", "mobility", "prehab"},
        "duration": {"timed"},
        "interval": {"interval"},
        "distance": {"distance"},
    }.get(expected_mode, set())

    for item in prescriptions:
        set_type = str(item.get("set_type") or item.get("setType") or "working").lower()
        duration = item.get("target_duration_sec")
        work = item.get("interval_work_sec")
        recovery = item.get("interval_recovery_sec")
        rounds = item.get("interval_rounds")
        distance = item.get("target_distance_m")
        if not requested_adjunct_match and expected_types and set_type not in expected_types:
            violations.append(
                f"[PRESCRIPTION_CONTRACT] unit mismatch in {exercise_id}: "
                f"mode={expected_mode}, setType={set_type}"
            )
        if set_type == "timed":
            valid = duration is not None and all(
                value is None for value in (work, recovery, rounds, distance)
            )
        elif set_type == "interval":
            valid = all(value is not None for value in (duration, work, recovery, rounds)) and distance is None
        elif set_type == "distance":
            valid = distance is not None and all(
                value is None for value in (duration, work, recovery, rounds)
            )
        else:
            valid = all(value is None for value in (duration, work, recovery, rounds, distance))
        if not valid:
            violations.append(
                f"[PRESCRIPTION_CONTRACT] incomplete or mixed unit fields in {exercise_id}: {set_type}"
            )

    if phase == "conditioning_finisher" and any(
        str(item.get("set_type") or "working").lower() != "interval"
        for item in prescriptions
    ):
        violations.append(
            f"[PRESCRIPTION_CONTRACT] conditioning finisher must use interval units: {exercise_id}"
        )
    if phase == "endurance_support" and any(
        str(item.get("set_type") or "working").lower() != "timed"
        for item in prescriptions
    ):
        violations.append(
            f"[PRESCRIPTION_CONTRACT] endurance support must use timed units: {exercise_id}"
        )
    if phase == "cooldown_recovery" and any(
        str(item.get("set_type") or "working").lower() == "interval"
        for item in prescriptions
    ):
        violations.append(
            f"[PRESCRIPTION_CONTRACT] interval work cannot be cooldown/recovery: {exercise_id}"
        )
    return violations


def _check_hard_constraints_sanity(
    blocks: List[dict],
    candidates_by_id: Dict[str, dict],
    context: RoutineEvalContext,
    max_working_sets: int,
    adjunct_decisions: Optional[List[dict]] = None,
) -> List[str]:
    violations: List[str] = []
    blocked = {item.strip().upper() for item in context.unavailable_equipment}
    available = {item.strip().upper() for item in context.available_equipment}
    injury_tokens = _pain_tokens_from_entries(context.current_pain_areas)
    ids = set(_block_ids(blocks))

    for forbidden_id in context.overrides.get("forbidden_exercise_ids", []):
        if forbidden_id in ids:
            violations.append(f"[GUARD_BYPASS] forbidden exercise present: {forbidden_id}")

    for block in blocks:
        exercise_id = str(block.get("exercise_id") or "")
        candidate = candidates_by_id.get(exercise_id)
        if candidate is None:
            violations.append(f"[GUARD_BYPASS] exercise_id '{exercise_id}' not in candidates")
            continue

        equip_tokens = _equipment_tokens(candidate) - {"BODYWEIGHT"}
        if equip_tokens & blocked:
            violations.append(f"[GUARD_BYPASS] blocked equipment in {exercise_id}: {sorted(equip_tokens & blocked)}")
        if available and equip_tokens and not (equip_tokens & available):
            violations.append(
                f"[GUARD_BYPASS] unavailable whitelist equipment in {exercise_id}: {sorted(equip_tokens)}"
            )

        injury_overlap = _candidate_injury_overlap(candidate, injury_tokens)
        if injury_overlap:
            violations.append(
                f"[GUARD_BYPASS] registered injury overlap in {exercise_id}: "
                f"{sorted(injury_overlap)}"
            )

        for muscle in _primary_muscles(block, candidates_by_id):
            if int(context.doms.get(str(muscle), 0) or 0) >= 3:
                violations.append(f"[GUARD_BYPASS] DOMS level 3 muscle in {exercise_id}: {muscle}")

        if _equipment_tokens(candidate) <= {"BODYWEIGHT"}:
            weighted_sets = [
                prescription
                for prescription in block.get("prescription") or []
                if prescription.get("target_weight_kg") is not None
            ]
            if weighted_sets:
                violations.append(f"[GUARD_BYPASS] BODYWEIGHT target_weight_kg must be null: {exercise_id}")

        violations.extend(_prescription_contract_violations(block, candidate, context))

    total_sets = _total_working_sets(blocks)
    if total_sets > max_working_sets:
        violations.append(f"[GUARD_BYPASS] working set cap exceeded: actual={total_sets}, max={max_working_sets}")

    minimum_main = context.minimum_main_exercise_count
    if minimum_main is not None:
        actual_main = count_main_blocks(blocks)
        adjunct_block_count = len(blocks) - actual_main
        if context.adjunct_options and adjunct_block_count > 0 and actual_main < minimum_main:
            violations.append(
                "[MAIN_MINIMUM] adjunct retained before main minimum: "
                f"mainActual={actual_main}, mainMinimum={minimum_main}, "
                f"adjunctBlocks={adjunct_block_count}"
            )

    if adjunct_decisions is None:
        # Backward-compatible direct helper mode used by the offline harness.
        for adjunct_option in context.adjunct_options:
            if not any(
                adjunct_block_matches_profile(
                    block,
                    adjunct_option,
                    candidate=candidates_by_id.get(str(block.get("exercise_id") or "")),
                )
                for block in blocks
            ):
                violations.append(
                    f"[ADJUNCT_CONTRACT] requested adjunct missing or invalid: {adjunct_option}"
                )
    else:
        violations.extend(
            f"[ADJUNCT_CONTRACT] {item}"
            for item in validate_adjunct_decision_contract(
                context.adjunct_options,
                blocks,
                adjunct_decisions,
                candidates_by_id=candidates_by_id,
            )
        )

    return violations


def _coverage_score(
    blocks: List[dict],
    candidates_by_id: Dict[str, dict],
    context: RoutineEvalContext,
    fallback_used: bool,
    candidate_count: int,
) -> Tuple[int, List[str]]:
    required = list(context.overrides.get("required_muscle_coverage") or context.target_muscles)
    if not required:
        return 25, []
    covered = {
        muscle
        for block in blocks
        for muscle in _primary_muscles(block, candidates_by_id)
    } & set(required)
    ratio = len(covered) / len(required)
    score = round(ratio * 25)
    warnings: List[str] = []
    exception = (
        bool(context.overrides.get("coverage_exception", False))
        or fallback_used
        or candidate_count <= 2
        or context.readiness_level == "low"
    )
    if ratio < 0.7:
        message = f"coverage below 70%: covered={sorted(covered)}, required={required}"
        warnings.append(message if exception else "FAIL " + message)
    elif ratio < 1.0:
        warnings.append(f"partial coverage: covered={sorted(covered)}, required={required}")
    return score, warnings


def _time_score(
    response: RoutineDraftResponse,
    context: RoutineEvalContext,
    fallback_used: bool,
    candidate_count: int,
) -> Tuple[int, List[str]]:
    actual = response.total_estimated_time
    requested = context.duration_min
    target_exercise_count = int(
        context.overrides.get("target_exercise_count")
        or target_exercise_count_for_session(requested, context.goal)
    )
    minimum_exercise_count = int(
        context.overrides.get("min_exercise_count")
        or max(1, target_exercise_count - 1)
    )
    explicit_underfill_exception = (
        bool(context.overrides.get("allowed_underfill", False))
        or fallback_used
        or context.readiness_level == "low"
        or bool(context.overrides.get("coverage_exception", False))
        or bool(getattr(response, "_underfill_diagnostics", {}).get("reason_code"))
    )
    selected_count = len(response.routine_blocks)
    candidate_shortage = (
        0 < candidate_count < minimum_exercise_count
        and selected_count >= candidate_count
    )
    if actual > requested:
        return 0, [f"FAIL time budget exceeded: actual={actual}, available={requested}"]
    target = int(
        context.overrides.get("target_productive_duration_min")
        or target_productive_duration_min(requested)
    )
    floor = int(
        context.overrides.get("minimum_productive_duration_min")
        or minimum_productive_duration_min(requested)
    )
    if actual >= target:
        return 20, []
    if actual >= floor:
        return 10, [f"time utilization warning: actual={actual}, target={target}, available={requested}"]
    if explicit_underfill_exception:
        return 10, [f"allowed underfill: actual={actual}, floor={floor}, available={requested}"]
    if candidate_shortage:
        return 10, [
            f"candidate-limited underfill: actual={actual}, floor={floor}, "
            f"available={requested}, candidates={candidate_count}, minExpected={minimum_exercise_count}"
        ]
    return 0, [f"FAIL time utilization too low: actual={actual}, floor={floor}, available={requested}"]


def _exercise_count_score(
    blocks: List[dict],
    context: RoutineEvalContext,
    fallback_used: bool,
    candidate_count: int,
    underfill_reason: Optional[str] = None,
) -> Tuple[int, List[str]]:
    actual = count_main_blocks(blocks)
    has_adjunct_contract = bool(context.adjunct_options) or actual < len(blocks)
    target = int(
        context.overrides.get("target_exercise_count")
        or (context.target_main_exercise_count if has_adjunct_contract else None)
        or target_exercise_count_for_session(context.duration_min, context.goal)
    )
    min_expected = int(
        context.overrides.get("min_exercise_count")
        or (context.minimum_main_exercise_count if has_adjunct_contract else None)
        or max(1, target - 1)
    )
    candidate_shortage = (
        0 < candidate_count < min_expected
        and actual >= candidate_count
    )
    allowed_underfill = (
        bool(context.overrides.get("allowed_underfill", False))
        or fallback_used
        or context.readiness_level == "low"
        or candidate_shortage
        or bool(context.overrides.get("coverage_exception", False))
        or bool(underfill_reason)
    )

    if actual >= min_expected:
        return 10, []
    message = f"exercise count underfilled: actual={actual}, minExpected={min_expected}, target={target}"
    if allowed_underfill:
        return 5, [message]
    return 0, ["FAIL " + message]


def _diversity_score(
    blocks: List[dict],
    context: RoutineEvalContext,
    candidate_count: int,
) -> Tuple[int, List[str]]:
    routine_ids = set(_block_ids(blocks))
    if not routine_ids:
        return 0, ["FAIL no routine blocks for diversity"]
    repeat_ratio = len(routine_ids & set(context.recent_exercises)) / len(routine_ids)
    if repeat_ratio <= 0.25:
        return 15, []
    if repeat_ratio <= 0.50:
        return 10, [f"recent repetition: ratio={repeat_ratio:.2f}"]
    if repeat_ratio <= 0.75:
        return 5, [f"recent repetition warning: ratio={repeat_ratio:.2f}"]
    if bool(context.overrides.get("candidate_shortage", False)) or candidate_count <= 2:
        return 5, [f"candidate shortage allows repetition warning: ratio={repeat_ratio:.2f}"]
    return 0, [f"FAIL recent repetition too high: ratio={repeat_ratio:.2f}"]


def _balance_score(
    blocks: List[dict],
    candidates_by_id: Dict[str, dict],
    context: RoutineEvalContext,
    candidate_count: int,
    fallback_used: bool,
) -> Tuple[int, List[str]]:
    if not blocks:
        return 0, ["FAIL no routine blocks for balance"]

    compound_count = sum(1 for block in blocks if _movement_type(block, candidates_by_id) == "COMPOUND")
    ratio = compound_count / len(blocks)
    min_compound = int(context.overrides.get("min_compound_count", 0) or 0)
    shortage_exception = bool(context.overrides.get("candidate_shortage", False)) or candidate_count <= 2 or fallback_used
    goal_key = (context.goal or "hypertrophy").replace("_", "").lower()

    if compound_count < min_compound:
        message = f"compound count below expected: actual={compound_count}, expected={min_compound}"
        return (8, [message]) if shortage_exception else (0, ["FAIL " + message])

    if goal_key == "strength":
        if compound_count == 0:
            message = "strength routine has no compound exercise"
            return (8, [message]) if shortage_exception else (0, ["FAIL " + message])
        if ratio < 0.4:
            return 8, [f"strength compound ratio low: ratio={ratio:.2f}"]
        return 15, []
    if goal_key in {"hypertrophy", "generalfitness"}:
        if 0.3 <= ratio <= 0.7:
            return 15, []
        return 8, [f"{goal_key} compound/isolation mix outside preferred range: ratio={ratio:.2f}"]
    if goal_key in {"fatloss", "endurance", "recomposition"}:
        if 0.2 <= ratio <= 0.8:
            return 15, []
        return 8, [f"{goal_key} balance warning: ratio={ratio:.2f}"]
    return 8, [f"unknown goal balance warning: goal={context.goal}"]


def _rationale_score(blocks: List[dict], context: RoutineEvalContext) -> Tuple[int, List[str]]:
    if not blocks:
        return 0, ["FAIL no routine blocks for rationale"]
    ratio = sum(1 for block in blocks if str(block.get("exercise_rationale") or "").strip()) / len(blocks)
    must_be_nonempty = bool(context.overrides.get("rationale_must_be_nonempty", True))
    warnings: List[str] = []

    rationale_text = " ".join(str(block.get("exercise_rationale") or "").lower() for block in blocks)
    suspicious_terms = ("diagnosed", "doctor said", "medical history", "injury history shows")
    if any(term in rationale_text for term in suspicious_terms):
        warnings.append("FAIL unsupported medical claim or invented user history in rationale")

    if ratio == 1.0:
        return 15, warnings
    if ratio >= 0.5:
        message = f"rationale partially missing: ratio={ratio:.2f}"
        warnings.append("FAIL " + message if must_be_nonempty else message)
        return (0 if must_be_nonempty else 8), warnings
    warnings.append(f"FAIL rationale missing: ratio={ratio:.2f}")
    return 0, warnings


def _repair_hygiene_score(
    response: RoutineDraftResponse,
    repair_count: int,
    fallback_used: bool,
    context: RoutineEvalContext,
) -> Tuple[int, List[str]]:
    if response.generation_status == "failed":
        return 0, ["FAIL generation status failed"]

    expected_fallback = bool(context.overrides.get("is_fallback", False))
    expected_repair = bool(context.overrides.get("expected_repair", False))
    if fallback_used != expected_fallback:
        return 0, [f"FAIL fallback mismatch: actual={fallback_used}, expected={expected_fallback}"]
    if expected_repair and repair_count == 0:
        return 0, ["FAIL expected repair did not occur"]
    if expected_fallback and fallback_used:
        return 10, []
    if expected_repair and repair_count > 0:
        return 10, []

    if fallback_used:
        return 7, ["fallback used"]
    if repair_count > 2:
        return 3, [f"high repair count: {repair_count}"]
    if repair_count > 0:
        return 7, [f"repair occurred: count={repair_count}"]
    return 10, []


def _grade(score: int, hard_violations: List[str]) -> str:
    if hard_violations:
        return "FAIL"
    if score >= 80:
        return "PASS"
    if score >= 65:
        return "WARN"
    return "FAIL"


def evaluate_routine_quality(
    response: RoutineDraftResponse,
    candidates: List[dict],
    context: RoutineEvalContext,
    repair_count: int = 0,
    fallback_used: bool = False,
) -> RuleCriticResult:
    blocks = [block.model_dump(by_alias=False) for block in response.routine_blocks]
    candidates_by_id = _candidate_lookup(candidates)
    candidate_count = len(candidates)
    max_sets = context.max_working_sets or _calculate_max_total_sets(context.duration_min, context.goal)

    adjunct_decisions = [
        decision.model_dump(by_alias=False)
        for decision in response.adjunct_decisions
    ]
    hard_violations = _check_hard_constraints_sanity(
        blocks,
        candidates_by_id,
        context,
        max_sets,
        adjunct_decisions=adjunct_decisions,
    )
    if (
        response.generation_status in {"success", "fallback"}
        and context.required_main_pattern_targets
        and not routine_items_have_required_main_pattern(
            blocks,
            context.required_main_pattern_targets,
            candidates=candidates,
            required_slot_specs=context.required_main_pattern_slot_specs,
        )
    ):
        hard_violations.append(
            "[MAIN_PATTERN] successful routine has no eligible required main pattern"
        )
    warnings: List[str] = []
    metric_scores: Dict[str, int] = {}

    metric_scores["coverage"], metric_warnings = _coverage_score(
        blocks, candidates_by_id, context, fallback_used, candidate_count
    )
    warnings.extend(metric_warnings)
    metric_scores["time"], metric_warnings = _time_score(response, context, fallback_used, candidate_count)
    warnings.extend(metric_warnings)
    metric_scores["exercise_count"], metric_warnings = _exercise_count_score(
        blocks,
        context,
        fallback_used,
        candidate_count,
        underfill_reason=getattr(response, "_underfill_diagnostics", {}).get("reason_code"),
    )
    warnings.extend(metric_warnings)
    metric_scores["diversity"], metric_warnings = _diversity_score(blocks, context, candidate_count)
    warnings.extend(metric_warnings)
    metric_scores["balance"], metric_warnings = _balance_score(
        blocks, candidates_by_id, context, candidate_count, fallback_used
    )
    warnings.extend(metric_warnings)
    metric_scores["rationale"], metric_warnings = _rationale_score(blocks, context)
    warnings.extend(metric_warnings)
    metric_scores["hygiene"], metric_warnings = _repair_hygiene_score(
        response, repair_count, fallback_used, context
    )
    warnings.extend(metric_warnings)

    score = 0 if hard_violations else min(100, sum(metric_scores.values()))
    grade = _grade(score, hard_violations)
    notes = []
    if context.overrides.get("notes"):
        notes.append(str(context.overrides["notes"]))

    return RuleCriticResult(
        score=score,
        grade=grade,
        hard_violations=hard_violations,
        warnings=warnings,
        metric_scores=metric_scores,
        should_rebuild=False,
        should_fallback=False,
        notes=notes,
    )


def log_critic_result(result: RuleCriticResult) -> None:
    log_fn = logger.warning if result.grade == "FAIL" else logger.info
    log_fn(
        "[Critic] grade=%s score=%s hard_violations=%s should_rebuild=%s should_fallback=%s",
        result.grade,
        result.score,
        len(result.hard_violations),
        result.should_rebuild,
        result.should_fallback,
    )
    metrics = " ".join(f"{key}={value}" for key, value in result.metric_scores.items())
    logger.info("[Critic] metrics: %s", metrics)
    for violation in result.hard_violations:
        logger.warning("[Critic][GUARD_BYPASS] %s", violation)
    for warning in result.warnings:
        logger.warning("[Critic] warning: %s", warning)

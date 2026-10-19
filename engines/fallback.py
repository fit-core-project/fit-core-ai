"""규칙 기반 Fallback 루틴 생성 + 루틴 빌드·디버그 유틸리티."""
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.prompts import ChatPromptTemplate

from engines.llm_router import StatusReasonCode
from .schemas import (
    GenerationStatus,
    LLMExercisePlan,
    LLMRoutineOutput,
    LLMRoutineSelectionOutput,
    RoutineDraftResponse,
    RoutineBlock,
    RoutineRequest,
    RecentSetRecord,
    SetPrescription,
    SubstitutionCandidate,
    UserProfileContext,
)
from .exercise_display import candidate_display_name
from .registry import LARGE_MUSCLE_SLUGS, ACCESSORY_MUSCLE_SLUGS
from .prescription.estimator import estimate_routine_time_min, _movement_type_key
from .prescription.params import _prescription_params_for_exercise
from .prescription.weight import _resolve_target_weight
from .prescription.adjustments import (
    _apply_doms_volume_policy,
    _apply_large_muscle_volume_guard,
    _apply_readiness_to_exercise,
    _fill_available_time,
    target_exercise_count_for_session,
)
from .prescription.adjuncts import apply_adjunct_prescription_profile, is_working_prescription
from .prescription.experience import apply_experience_prescription
from .prescription.focus_capacity import complete_lower_chest_focus_capacity
from .prescription.upper_role_allocation import allocate_upper_role_sets, uses_joint_upper_budget
from .prescription.metadata import apply_candidate_prescription_metadata
from .candidate_ranker import (
    candidate_selection_pattern_key,
    _candidate_injury_overlap,
    _condition_policy_matches_lumbar,
    _candidate_doms_overlap,
    _candidate_doms_prescription_signal,
    _is_beginner_experience,
    _is_loadable_equipment,
    _is_tibialis_accessory_candidate,
    _level_rank,
    _normalize_body_part,
    _normalize_constraint_tokens,
    _split_primary_targets,
)
from .log_redaction import (
    sanitize_dev_log_payload,
    sanitize_exception_for_log,
    summarize_text_for_log,
)
from .muscle_mapping import resolve_request_target_muscles
from .llm_parser import trim_routine_to_time_budget
from .korean_text_guard import enforce_korean_user_text_on_output, enforce_korean_user_text_on_response
from .injury_condition_policy import candidate_injury_condition_overlaps
from .equipment_policy import select_equipment_option
from .option_resolver import candidate_matches_adjunct_option
from .routine_blueprint import RoutineBlueprint, build_blueprint_candidate_bundles
from .experience_policy import automatic_recommendation_eligibility
from .adjunct_admission import admission_purposes
from .routine_selection import (
    _candidate_public_rationale,
    is_redundant_push_front_raise,
    adapt_slot_selection_output,
    apply_blueprint_execution_policy,
    apply_blueprint_execution_policy_to_draft,
    prioritize_antagonist_superset_candidates,
    routine_output_has_required_main_pattern,
)
from .specialization_policy import (
    primary_focus_candidate_ids,
    apply_blueprint_specialization_volume,
    apply_training_method_to_plan,
)

logger = logging.getLogger(__name__)

# ==========================================
# 9. Fallback 루틴 (규칙 기반)  →  fallback.py 로 이동됨
# ==========================================

_GOAL_PARAMS = {
    "strength":    {"sets": 5, "reps": 5,  "rest_sec": 180},
    "hypertrophy": {"sets": 3, "reps": 10, "rest_sec": 90},
    "endurance":   {"sets": 3, "reps": 15, "rest_sec": 60},
    "fatloss":     {"sets": 3, "reps": 15, "rest_sec": 60},
    "recomposition": {"sets": 3, "reps": 10, "rest_sec": 90},
    "generalfitness": {"sets": 3, "reps": 12, "rest_sec": 75},
}

_MOVEMENT_PRIORITY = {"COMPOUND": 0, "ISOLATION": 1, "STATIC": 2}
_GYM_BASE_ADJUNCT_STYLES = {"crossfit", "mobility", "yoga_pilates", "rehab_prehab"}
_MUSCLE_ORDER_PRIORITY = {
    "chest": 10,
    "upper-back": 20,
    "lower-back": 25,
    "quadriceps": 30,
    "hamstring": 35,
    "gluteal": 40,
    "front-deltoids": 50,
    "back-deltoids": 55,
    "triceps": 60,
    "biceps": 65,
    "forearm": 70,
    "calves": 75,
    "adductor": 80,
    "abductors": 85,
    "abs": 90,
    "obliques": 95,
    "trapezius": 100,
    "neck": 110,
}


def _primary_muscle_priority(primary: Optional[str]) -> int:
    if primary in LARGE_MUSCLE_SLUGS:
        return 0
    if primary in ACCESSORY_MUSCLE_SLUGS:
        return 1
    return 2


def _planned_exercise_order_key(item: Tuple[int, LLMExercisePlan]) -> Tuple[int, int, float, int]:
    original_order, exercise = item
    primary = exercise.primary_muscles[0] if exercise.primary_muscles else None
    weight = exercise.target_weight_kg or 0
    return (
        _MUSCLE_ORDER_PRIORITY.get(str(primary or ""), 999),
        _MOVEMENT_PRIORITY.get(_movement_type_key(exercise.movement_type), 3),
        -weight,
        original_order,
    )


def _candidate_order_key(candidate: dict) -> Tuple[int, int, int, int, str]:
    return (
        0 if _is_loadable_equipment(
            candidate.get("selected_equipment_code") or candidate.get("equipment_req")
        ) else 1,
        _MOVEMENT_PRIORITY.get(_movement_type_key(candidate.get("movement_type")), 3),
        _primary_muscle_priority(candidate.get("primary_muscle")),
        int(candidate.get("efficiency_tier") or 99),
        str(candidate.get("id") or ""),
    )


def _candidate_secondary_set(candidate: dict) -> set[str]:
    return {
        _normalize_body_part(part)
        for part in str(candidate.get("secondary_muscle") or "").split(",")
        if part.strip()
    }


def _request_pain_tokens(req: RoutineRequest) -> set[str]:
    tokens = {
        _normalize_body_part(entry.body_part)
        for entry in (req.pain_areas or [])
        if entry.body_part
    }
    if tokens & {"left-soleus", "right-soleus", "soleus"}:
        tokens.update({"soleus", "calves", "ankle"})
    if tokens & {"front-deltoids", "back-deltoids", "side-deltoids"}:
        tokens.add("shoulder")
    if "shoulder" in tokens:
        tokens.update({"front-deltoids", "back-deltoids", "side-deltoids"})
    return tokens


def _fallback_request_warnings(req: RoutineRequest) -> List[str]:
    warnings = ["중량은 본인의 컨디션에 맞게 조절하세요."]
    pain_tokens = _request_pain_tokens(req)
    if pain_tokens & {"lower-back", "back", "lumbar"}:
        warnings.append("등록된 허리/요추 부상 부위에 관여하는 운동은 후보에서 제외했습니다.")
    if pain_tokens & {"knees", "knee"}:
        warnings.append("등록된 무릎 부상 부위에 관여하는 운동은 후보에서 제외했습니다.")
    if pain_tokens & {"shoulder", "front-deltoids", "back-deltoids", "side-deltoids"}:
        warnings.append("등록된 어깨 부상 부위에 관여하는 운동은 후보에서 제외했습니다.")
    if pain_tokens & {"forearm", "wrist"}:
        warnings.append("등록된 전완/손목 부상 부위에 관여하는 운동은 후보에서 제외했습니다.")
    if pain_tokens & {"ankle", "calves", "soleus", "left-soleus", "right-soleus"}:
        warnings.append("등록된 발목/가자미근 부상 부위에 관여하는 운동은 후보에서 제외했습니다.")
    if any(
        str(policy.get("status") or "").strip().lower() == "needs_clearance"
        for policy in (req.condition_policies or [])
        if isinstance(policy, dict)
    ):
        warnings.append("수술/질환 이력이 있는 부위는 의료진 허가 범위 안에서만 진행하세요.")
    return warnings


def _fallback_context_penalty(
    candidate: dict,
    req: RoutineRequest,
    fallback_target_muscles: List[str],
) -> int:
    """Fallback도 normal ranking과 같은 안전/분할 맥락을 따르도록 보수 정렬한다."""
    penalty = 0
    primary = _normalize_body_part(candidate.get("primary_muscle"))
    secondary = _candidate_secondary_set(candidate)
    target_set = {_normalize_body_part(part) for part in fallback_target_muscles}
    split_targets = _split_primary_targets(req.target_split_label)

    if (
        split_targets
        and primary
        and primary not in split_targets
        and primary not in target_set
        and not (secondary & split_targets)
        and not (secondary & target_set)
    ):
        penalty += 80

    split_label = str(req.target_split_label or "").strip().lower()
    if (
        split_label in {"legs", "lower"}
        and _is_tibialis_accessory_candidate(candidate)
    ):
        penalty += 90 if _is_beginner_experience(req.experience_level) else 55

    mobility_limits = _normalize_constraint_tokens(req.mobility_limits)
    if "limited_ankle_dorsiflexion" in mobility_limits:
        ankle_rank = _level_rank(candidate.get("ankle_dorsiflexion_demand"))
        if ankle_rank >= 3:
            penalty += 45
        elif ankle_rank == 2:
            penalty += 18

    pain_tokens = _request_pain_tokens(req)
    has_lumbar_constraint = (
        bool(pain_tokens & {"lower-back", "back", "lumbar"})
        or _condition_policy_matches_lumbar(req.condition_policies)
    )
    if has_lumbar_constraint:
        lumbar_rank = _level_rank(candidate.get("lumbar_load"))
        axial_rank = _level_rank(candidate.get("axial_load"))
        if lumbar_rank >= 3:
            penalty += 70
        elif lumbar_rank == 2:
            penalty += 28
        if axial_rank >= 3:
            penalty += 40
        elif axial_rank == 2:
            penalty += 14

    if pain_tokens & {"knees", "knee"}:
        knee_rank = _level_rank(candidate.get("knee_shear_load"))
        if knee_rank >= 3:
            penalty += 70
        elif knee_rank == 2:
            penalty += 28

    if pain_tokens & {"shoulder", "front-deltoids", "back-deltoids", "side-deltoids"}:
        shoulder_rank = _level_rank(candidate.get("shoulder_impingement_risk"))
        if shoulder_rank >= 3:
            penalty += 70
        elif shoulder_rank == 2:
            penalty += 28
        if int(candidate.get("overhead_position_required") or 0) == 1:
            penalty += 35

    if pain_tokens & {"forearm", "wrist"}:
        wrist_rank = _level_rank(candidate.get("wrist_stress"))
        if wrist_rank >= 3:
            penalty += 70
        elif wrist_rank == 2:
            penalty += 28

    if pain_tokens & {"ankle", "calves", "soleus", "left-soleus", "right-soleus"}:
        ankle_rank = _level_rank(candidate.get("ankle_dorsiflexion_demand"))
        if ankle_rank >= 3:
            penalty += 70
        elif ankle_rank == 2:
            penalty += 28

    if _level_rank(candidate.get("injury_caution_level")) >= 3:
        penalty += 35
    return penalty


def _contextual_candidate_order_key(
    candidate: dict,
    req: RoutineRequest,
    fallback_target_muscles: List[str],
) -> Tuple[int, int, int, int, int, int, str]:
    base = _candidate_order_key(candidate)
    return (
        _fallback_context_penalty(candidate, req, fallback_target_muscles),
        -int(candidate.get("score") or 0),
        *base,
    )


class _FallbackCandidateList(list[dict]):
    """List-compatible fallback candidates with bounded internal selection facts."""

    def __init__(
        self, candidates: Sequence[dict], diagnostics: Dict[str, int | bool],
        *, slot_by_exercise_id: Optional[Dict[str, str]] = None,
        rationale_by_exercise_id: Optional[Dict[str, str]] = None,
    ):
        super().__init__(candidates)
        self._fallback_selection_diagnostics = diagnostics
        self._fallback_slot_by_exercise_id = dict(slot_by_exercise_id or {})
        self._fallback_rationale_by_exercise_id = dict(rationale_by_exercise_id or {})


def _blueprint_fallback_candidates(
    blueprint: RoutineBlueprint,
    ranked_candidates: Sequence[dict],
) -> List[dict]:
    """Select fallback exercises through the normal deterministic slot policy."""
    bundles = build_blueprint_candidate_bundles(blueprint, ranked_candidates)
    selections: List[dict] = []
    for bundle in bundles:
        slot_id = str(bundle.get("slotId") or "").strip()
        candidates = [
            item
            for item in (bundle.get("candidates") or [])
            if isinstance(item, dict) and str(item.get("exerciseId") or "").strip()
        ]
        selected_items = candidates if slot_id == "flex" else candidates[:1]
        selections.extend(
            {
                "slotId": slot_id,
                "exerciseId": str(item.get("exerciseId") or "").strip(),
            }
            for item in selected_items
        )

    output = adapt_slot_selection_output(
        LLMRoutineSelectionOutput.model_validate({"selections": selections}),
        blueprint=blueprint,
        slot_candidate_bundles=bundles,
        ranked_candidates=ranked_candidates,
    )
    selection_diagnostics = dict(output._selection_diagnostics or {})
    selected_count = len(output.exercises)
    selected_main_count = sum(
        str(plan.exercise_id or "").strip().lower()
        not in output._adjunct_option_by_exercise_id
        for plan in output.exercises
    )
    selection_shortfall = max(
        0,
        blueprint.minimum_main_exercise_count - selected_main_count,
    )
    family_skip_count = int(selection_diagnostics.get("duplicate_family_skip_count") or 0)
    pattern_skip_count = int(selection_diagnostics.get("coarse_pattern_skip_count") or 0)
    fallback_diagnostics = {
        "specialization_focus_unavailable": bool(
            selection_diagnostics.get("specialization_focus_unavailable", False)
        ),
        "fallback_selection_measured": True,
        "fallback_candidate_pool_count": len(ranked_candidates),
        "fallback_selected_count": selected_count,
        "fallback_existing_family_skip_count": family_skip_count,
        "fallback_existing_pattern_skip_count": pattern_skip_count,
        "fallback_selection_shortfall": selection_shortfall,
        # This is intentionally narrower than ranked-pool exhaustion: it only
        # records a measured adapter suppression while the minimum was missed.
        "fallback_composition_limit_observed": bool(
            selection_shortfall > 0 and (family_skip_count > 0 or pattern_skip_count > 0)
        ),
    }
    if not routine_output_has_required_main_pattern(
        output,
        blueprint,
        ranked_candidates,
    ):
        return _FallbackCandidateList([], fallback_diagnostics)

    candidate_by_id = {
        str(candidate.get("id") or "").strip().lower(): candidate
        for candidate in ranked_candidates
        if str(candidate.get("id") or "").strip()
    }
    return _FallbackCandidateList([
        candidate_by_id[exercise_id]
        for exercise_id in (
            str(plan.exercise_id or "").strip().lower()
            for plan in output.exercises
        )
        if exercise_id in candidate_by_id
    ], fallback_diagnostics, slot_by_exercise_id=output._selection_slot_by_exercise_id,
        rationale_by_exercise_id={
            str(plan.exercise_id or "").strip().lower(): plan.exercise_rationale
            for plan in output.exercises
        })


def _normalize_training_style(value: Optional[str]) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _candidate_has_training_style_match(candidate: dict, normalized_training_style: str) -> bool:
    if not normalized_training_style:
        return False
    for boost in candidate.get("score_boosts") or []:
        if not isinstance(boost, dict):
            continue
        if (
            boost.get("ruleCode") == "training_style_match_boost"
            or boost.get("rule_code") == "training_style_match_boost"
        ) and _normalize_training_style(
            boost.get("constraintCode") or boost.get("constraint_code")
        ) == normalized_training_style:
            return True

    raw_tags = candidate.get("training_style_tags")
    if isinstance(raw_tags, str):
        tags = {
            _normalize_training_style(tag)
            for tag in raw_tags.replace(";", ",").split(",")
            if tag.strip()
        }
    elif isinstance(raw_tags, (list, tuple, set)):
        tags = {_normalize_training_style(tag) for tag in raw_tags}
    else:
        tags = set()
    return normalized_training_style in tags


def _ensure_requested_style_adjunct_for_fallback(
    sorted_candidates: List[dict],
    req: RoutineRequest,
    max_exercise_count: int,
) -> List[dict]:
    """Keep one requested non-gym style candidate visible without replacing the gym formula."""
    normalized_training_style = _normalize_training_style(getattr(req, "training_style", None))
    if (
        max_exercise_count <= 1
        or normalized_training_style not in _GYM_BASE_ADJUNCT_STYLES
        or any(
            _candidate_has_training_style_match(candidate, normalized_training_style)
            for candidate in sorted_candidates[:max_exercise_count]
        )
    ):
        return sorted_candidates

    style_candidate = next(
        (
            candidate
            for candidate in sorted_candidates[max_exercise_count:]
            if _candidate_has_training_style_match(candidate, normalized_training_style)
        ),
        None,
    )
    if style_candidate is None:
        return sorted_candidates

    style_id = str(style_candidate.get("id") or "")
    without_style = [
        candidate
        for candidate in sorted_candidates
        if str(candidate.get("id") or "") != style_id
    ]
    return [
        *without_style[: max_exercise_count - 1],
        style_candidate,
        *without_style[max_exercise_count - 1 :],
    ]


def _ensure_requested_option_adjuncts_for_fallback(
    sorted_candidates: List[dict],
    adjunct_options: Sequence[str] | None,
    max_exercise_count: int,
    minimum_main_exercise_count: int,
) -> List[dict]:
    requested = list(dict.fromkeys(
        str(option or "").strip().lower().replace("-", "_")
        for option in (adjunct_options or [])
        if str(option or "").strip()
    ))
    if max_exercise_count <= 1 or not requested:
        return sorted_candidates

    reserved: list[dict] = []
    reserved_ids: set[str] = set()
    for option in requested:
        candidate = next(
            (
                item
                for item in sorted_candidates
                if str(item.get("id") or "").strip().lower() not in reserved_ids
                and candidate_matches_adjunct_option(item, option)
                and automatic_recommendation_eligibility(item, None, adjunct_options=(option,)).eligible
            ),
            None,
        )
        if candidate is None:
            continue
        reserved.append(candidate)
        reserved_ids.add(str(candidate.get("id") or "").strip().lower())

    if not reserved:
        return sorted_candidates
    reserved = reserved[: max(0, max_exercise_count - minimum_main_exercise_count)]
    reserved_ids = {
        str(candidate.get("id") or "").strip().lower()
        for candidate in reserved
    }
    without_reserved = [
        candidate
        for candidate in sorted_candidates
        if str(candidate.get("id") or "").strip().lower() not in reserved_ids
    ]
    main_window = max(0, max_exercise_count - len(reserved))
    return [
        *without_reserved[:main_window],
        *reserved,
        *without_reserved[main_window:],
    ]


def _fallback_target_exercise_count(
    time_available_min: Optional[int],
    goal: str = "hypertrophy",
) -> int:
    """Fallback should be more conservative than the normal LLM target count."""
    minutes = int(time_available_min or 60)
    if minutes <= 25:
        return 2
    if minutes <= 30:
        return 3
    return target_exercise_count_for_session(minutes, goal)


def _order_exercises_for_training(exercises: List[LLMExercisePlan]) -> List[LLMExercisePlan]:
    return [
        exercise
        for _, exercise in sorted(
            enumerate(exercises),
            key=_planned_exercise_order_key,
        )
    ]

def _build_prescription(
    sets: int,
    reps: int,
    weight_kg: Optional[float],
    rest_sec: int,
    target_rir: Optional[int] = 2,
    set_type: str = "working",
    target_duration_sec: Optional[int] = None,
    interval_work_sec: Optional[int] = None,
    interval_recovery_sec: Optional[int] = None,
    interval_rounds: Optional[int] = None,
    target_distance_m: Optional[int] = None,
) -> List[SetPrescription]:
    return [
        SetPrescription(
            set_index=i + 1,
            set_type=set_type,
            target_reps=reps,
            target_weight_kg=weight_kg,
            target_rir=target_rir,
            target_rest_sec=rest_sec,
            target_duration_sec=target_duration_sec,
            interval_work_sec=interval_work_sec,
            interval_recovery_sec=interval_recovery_sec,
            interval_rounds=interval_rounds,
            target_distance_m=target_distance_m,
        )
        for i in range(sets)
    ]


def _candidate_by_id(ranked_candidates: Optional[List[dict]]) -> Dict[str, dict]:
    return {
        str(candidate.get("id") or "").strip().lower(): candidate
        for candidate in (ranked_candidates or [])
        if str(candidate.get("id") or "").strip()
    }


def _candidate_score_context(candidate: Optional[dict]) -> Dict[str, List[Any]]:
    if not candidate:
        return {"reasons": [], "boosts": [], "penalties": []}
    return {
        "reasons": list(candidate.get("score_display_reasons") or []),
        "boosts": list(candidate.get("score_boosts") or []),
        "penalties": list(candidate.get("score_penalties") or []),
    }


def _candidate_display_name(candidate: dict) -> str:
    return candidate_display_name(candidate)


def build_routine_draft(
    llm_output: LLMRoutineOutput,
    generation_status: GenerationStatus,
    status_reason_code: StatusReasonCode,
    is_fallback: bool,
    ranked_candidates: Optional[List[dict]] = None,
    preserve_input_order: bool = False,
) -> RoutineDraftResponse:
    blocks = []
    ranked_by_id = _candidate_by_id(ranked_candidates)
    llm_output = enforce_korean_user_text_on_output(llm_output)
    ordered_exercises = (
        list(llm_output.exercises)
        if preserve_input_order
        else _order_exercises_for_training(llm_output.exercises)
    )
    execution_metadata = dict(llm_output._execution_metadata_by_exercise_id or {})
    adjunct_option_by_id = dict(llm_output._adjunct_option_by_exercise_id or {})
    for order, ex in enumerate(ordered_exercises, start=1):
        score_context = _candidate_score_context(
            ranked_by_id.get(str(ex.exercise_id or "").strip().lower())
        )
        metadata = execution_metadata.get(str(ex.exercise_id or "").strip().lower(), {})
        blocks.append(RoutineBlock(
            order=order,
            exercise_id=ex.exercise_id,
            exercise_name=ex.exercise_name,
            movement_pattern=ex.movement_pattern,
            movement_type=ex.movement_type,
            primary_muscles=ex.primary_muscles,
            equipment_type=ex.equipment_type,
            default_rest_sec=ex.rest_time_sec,
            prescription=_build_prescription(
                sets=ex.sets,
                reps=ex.target_reps,
                weight_kg=ex.target_weight_kg,
                rest_sec=ex.rest_time_sec,
                target_rir=ex.target_rir,
                set_type=getattr(ex, "_prescription_set_type", "working"),
                target_duration_sec=getattr(ex, "_target_duration_sec", None),
                interval_work_sec=getattr(ex, "_interval_work_sec", None),
                interval_recovery_sec=getattr(ex, "_interval_recovery_sec", None),
                interval_rounds=getattr(ex, "_interval_rounds", None),
                target_distance_m=getattr(ex, "_target_distance_m", None),
            ),
            exercise_rationale=ex.exercise_rationale,
            substitution_candidates=[
                SubstitutionCandidate(
                    exercise_id=s.exercise_id,
                    exercise_name=s.exercise_name,
                    reason=s.reason,
                )
                for s in ex.substitution_candidates
            ],
            reasons=score_context["reasons"],
            boosts=score_context["boosts"],
            penalties=score_context["penalties"],
            session_phase=metadata.get("sessionPhase"),
            execution_group_id=metadata.get("executionGroupId"),
            execution_group_type=metadata.get("executionGroupType", "single"),
            execution_group_position=int(metadata.get("executionGroupPosition") or 1),
            execution_group_size=int(metadata.get("executionGroupSize") or 1),
            transition_rest_sec=metadata.get("transitionRestSec"),
            group_rest_sec=metadata.get("groupRestSec"),
            adjunct_option=adjunct_option_by_id.get(
                str(ex.exercise_id or "").strip().lower()
            ),
            training_method=getattr(ex, "_training_method_instruction", None),
            unilateral_prescription=ex._unilateral_prescription,
        ))
        if ex._unilateral_set_overrides is not None:
            if len(ex._unilateral_set_overrides) != ex.sets:
                raise ValueError('Unilateral prescription count changed without rebuilding sets')
            blocks[-1].prescription = [SetPrescription.model_validate(s) for s in ex._unilateral_set_overrides]
        blocks[-1]._weight_guidance = ex._weight_guidance

    response = RoutineDraftResponse(
        generation_status=generation_status,
        status_reason_code=status_reason_code,
        is_fallback=is_fallback,
        total_estimated_time=estimate_routine_time_min(
            ordered_exercises,
            execution_metadata=execution_metadata,
        ),
        summary_title=llm_output.summary_title,
        rationale_summary=llm_output.rationale_summary,
        routine_blocks=blocks,
        warnings=llm_output.warnings,
    )
    response._adjunct_suppression_reason_by_option = dict(
        llm_output._adjunct_suppression_reason_by_option or {}
    )
    return enforce_korean_user_text_on_response(response)


def _debug_print_prompt(prompt: ChatPromptTemplate, invoke_kwargs: Dict[str, Any]) -> None:
    try:
        rendered = prompt.format(**invoke_kwargs)
    except Exception as exc:
        error = sanitize_exception_for_log(exc)
        logger.debug(
            "[AI DEBUG][PROMPT] %s",
            sanitize_dev_log_payload(
                {
                    "event": "routine_prompt_render_failed",
                    "redacted": True,
                    "error_type": error["type"],
                    "error_category": error["category"],
                }
            ),
        )
        return

    summary = summarize_text_for_log(rendered)
    logger.debug(
        "[AI DEBUG][PROMPT] %s",
        sanitize_dev_log_payload(
            {
                "event": "routine_prompt_rendered",
                "redacted": True,
                "prompt_char_count": summary["char_count"],
                "prompt_approx_tokens": summary["approx_tokens"],
            }
        ),
    )


def _debug_print_llm_output(label: str, output: LLMRoutineOutput) -> None:
    serialized = output.model_dump_json()
    summary = summarize_text_for_log(serialized)
    logger.debug(
        "[AI DEBUG][%s] %s",
        label,
        sanitize_dev_log_payload(
            {
                "event": "routine_llm_output_parsed",
                "redacted": True,
                "llm_output_char_count": summary["char_count"],
                "llm_output_approx_tokens": summary["approx_tokens"],
                "parsed_exercise_count": len(output.exercises),
            }
        ),
    )


def _debug_print_draft(label: str, draft: RoutineDraftResponse) -> None:
    logger.debug("\n========== [AI DEBUG] %s ==========", label)
    logger.debug(
        "status=%s reason=%s is_fallback=%s total_estimated_time=%s",
        draft.generation_status,
        draft.status_reason_code,
        draft.is_fallback,
        draft.total_estimated_time,
    )
    for block in draft.routine_blocks:
        first_set = block.prescription[0] if block.prescription else None
        logger.debug(
            "%s. id=%s name=%s primary=%s sets=%s reps=%s weight=%skg rir=%s rest=%ss",
            block.order,
            block.exercise_id,
            block.exercise_name,
            block.primary_muscles,
            len(block.prescription),
            first_set.target_reps if first_set else None,
            first_set.target_weight_kg if first_set else None,
            first_set.target_rir if first_set else None,
            first_set.target_rest_sec if first_set else None,
        )
    logger.debug("========== [AI DEBUG] END %s ==========\n", label)


def generate_fallback_routine(
    req: RoutineRequest,
    candidates: List[dict],
    max_total_sets: int,
    doms_db: Optional[Dict[str, int]] = None,
    goal: str = "hypertrophy",
    status_reason_code: StatusReasonCode = "networkError",
    recent_sets: Optional[List[RecentSetRecord]] = None,
    profile: Optional[UserProfileContext] = None,
    target_exercise_count: Optional[int] = None,
    minimum_main_exercise_count: Optional[int] = None,
    applied_adjunct_options: Optional[Sequence[str]] = None,
    routine_blueprint: Optional[RoutineBlueprint] = None,
) -> RoutineDraftResponse:
    logger.info("[Fallback] 규칙 기반 루틴 생성 시작")

    injury_tokens = _request_pain_tokens(req)
    admission_experience = (
        req.experience_level
        or (profile.experience_level if profile else None)
        or (routine_blueprint.experience_level if routine_blueprint else None)
    )
    requested_adjuncts = list(admission_purposes(
        applied_adjunct_options if applied_adjunct_options is not None
        else routine_blueprint.planned_adjunct_options if routine_blueprint is not None
        else req.adjunct_options or [],
        req.training_style if applied_adjunct_options is None and routine_blueprint is None else None,
    ))
    usable_candidates = []
    for candidate in candidates:
        if not automatic_recommendation_eligibility(candidate, admission_experience,
            adjunct_options=requested_adjuncts).eligible:
            continue
        if _candidate_injury_overlap(candidate, injury_tokens) or candidate_injury_condition_overlaps(
            candidate, req.injury_conditions
        ):
            continue
        selected_equipment = select_equipment_option(
            candidate.get("equipment_req"), req.equipment, req.available_equipment
        )
        if selected_equipment is not None:
            usable_candidates.append({
                **candidate,
                "selected_equipment_code": selected_equipment,
            })
    candidates = usable_candidates
    if not candidates:
        injury_context = bool(injury_tokens or req.injury_conditions)
        return enforce_korean_user_text_on_response(RoutineDraftResponse(
            generation_status="failed",
            status_reason_code="emptyCandidate",
            is_fallback=False,
            total_estimated_time=0,
            summary_title=(
                "등록된 부상 부위를 피해 구성할 수 없어요"
                if injury_context
                else "기본 루틴"
            ),
            rationale_summary=[
                (
                    "등록된 부상 부위에 관여하는 운동을 제외한 뒤 안전한 후보가 남지 않았습니다."
                    if injury_context
                    else "선택한 조건에 맞는 운동이 없습니다."
                )
            ],
            routine_blocks=[],
            warnings=[
                (
                    "오늘 목표 부위를 바꾸거나 운동을 보류해 주세요."
                    if injury_context
                    else "타겟 근육 또는 장비 조건을 변경해 주세요."
                )
            ],
        ))

    params = _GOAL_PARAMS.get(goal.lower(), _GOAL_PARAMS["hypertrophy"])
    doms = doms_db or {}
    focus_ids = primary_focus_candidate_ids(routine_blueprint, candidates)
    fallback_target_muscles = resolve_request_target_muscles(
        req.target_split_label,
        req.target_muscles,
    )

    sorted_candidates = sorted(
        candidates,
        key=lambda candidate: _contextual_candidate_order_key(
            candidate,
            req,
            fallback_target_muscles,
        ),
    )
    max_exercise_count = target_exercise_count or _fallback_target_exercise_count(
        req.time_available_min,
        goal,
    )
    minimum_main_count = max(
        1,
        min(
            max_exercise_count,
            minimum_main_exercise_count
            if minimum_main_exercise_count is not None
            else max(1, max_exercise_count - 1),
        ),
    )
    sorted_candidates = _ensure_requested_style_adjunct_for_fallback(
        sorted_candidates,
        req,
        max_exercise_count,
    )
    fallback_selection_diagnostics: dict[str, int | bool] = {}
    if routine_blueprint is not None:
        sorted_candidates = prioritize_antagonist_superset_candidates(
            sorted_candidates,
            routine_blueprint,
        )
    sorted_candidates = _ensure_requested_option_adjuncts_for_fallback(
        sorted_candidates,
        requested_adjuncts,
        max_exercise_count,
        minimum_main_count,
    )
    if routine_blueprint is not None:
        blueprint_candidates = [
            candidate
            for candidate in sorted_candidates
            if max(_candidate_doms_overlap(candidate, doms).values(), default=0) < 3
        ]
        sorted_candidates = _blueprint_fallback_candidates(
            routine_blueprint,
            blueprint_candidates,
        )
        fallback_selection_diagnostics = dict(
            getattr(sorted_candidates, "_fallback_selection_diagnostics", {}) or {}
        )

    joint_upper_budget = uses_joint_upper_budget(
        routine_blueprint, getattr(sorted_candidates, "_fallback_slot_by_exercise_id", {})
    ) and len(sorted_candidates) == 5 and max_exercise_count >= 5
    blocks: List[RoutineBlock] = []
    fallback_plans: List[LLMExercisePlan] = []
    ranked_by_id = _candidate_by_id(candidates)
    remaining_sets = max_total_sets
    order = 1
    applied_adjuncts: set[str] = set()
    reserved_adjunct_candidates: dict[str, dict] = {}
    reserved_candidate_ids: set[str] = set()
    adjunct_suppression_reasons: dict[str, str] = {}
    method_candidate_ids: set[str] = set()
    training_method_applied = False
    if (
        routine_blueprint is not None
        and routine_blueprint.training_method
        and routine_blueprint.training_method_slot_id
    ):
        method_bundle = next(
            (
                bundle
                for bundle in build_blueprint_candidate_bundles(
                    routine_blueprint,
                    sorted_candidates,
                )
                if bundle.get("slotId") == routine_blueprint.training_method_slot_id
            ),
            None,
        )
        method_candidate_ids = {
            str(candidate.get("exerciseId") or "").strip().lower()
            for candidate in (method_bundle or {}).get("candidates", [])
            if str(candidate.get("exerciseId") or "").strip()
        }
    for option in requested_adjuncts:
        reserved_candidate = next(
            (
                candidate
                for candidate in sorted_candidates[:max_exercise_count]
                if str(candidate.get("id") or "").strip().lower() not in reserved_candidate_ids
                and candidate_matches_adjunct_option(candidate, option)
                and automatic_recommendation_eligibility(candidate, admission_experience,
                    adjunct_options=(option,)).eligible
            ),
            None,
        )
        if reserved_candidate is not None:
            reserved_adjunct_candidates[option] = reserved_candidate
            reserved_candidate_ids.add(str(reserved_candidate.get("id") or "").strip().lower())
        else:
            adjunct_suppression_reasons[option] = "noEligibleCandidate"

    for candidate_index, ex in enumerate(sorted_candidates):
        if len(fallback_plans) >= max_exercise_count:
            break

        if is_redundant_push_front_raise(
            req.target_split_label, req.specialization_focus, candidate_selection_pattern_key(ex),
            {
                candidate_selection_pattern_key(
                    ranked_by_id.get(str(plan.exercise_id).strip().lower(), {})
                )
                for plan in fallback_plans
            },
            candidates,
        ):
            continue

        candidate_id = str(ex.get("id") or "").strip().lower()
        matched_adjunct = next(
            (
                option
                for option, candidate in reserved_adjunct_candidates.items()
                if option not in applied_adjuncts
                and candidate_id == str(candidate.get("id") or "").strip().lower()
            ),
            None,
        )
        is_requested_adjunct_candidate = matched_adjunct is not None
        if not automatic_recommendation_eligibility(ex, admission_experience,
            adjunct_options=(matched_adjunct,) if matched_adjunct else ()).eligible:
            continue
        if not joint_upper_budget and remaining_sets <= 0 and not is_requested_adjunct_candidate:
            if set(reserved_adjunct_candidates) - applied_adjuncts:
                continue
            break

        doms_signal = _candidate_doms_prescription_signal(ex, doms)
        doms_level = max(_candidate_doms_overlap(ex, doms).values(), default=0)
        if doms_level >= 3:
            if matched_adjunct:
                adjunct_suppression_reasons[matched_adjunct] = "noEligibleCandidate"
            continue

        sets = params["sets"]
        if not joint_upper_budget and not is_requested_adjunct_candidate:
            sets = min(sets, remaining_sets)
        target_reps, rest_sec = _prescription_params_for_exercise(
            LLMExercisePlan(
                exercise_id=str(ex.get("id", "")),
                exercise_name=_candidate_display_name(ex),
                movement_type=ex.get("movement_type"),
                primary_muscles=[ex["primary_muscle"]] if ex.get("primary_muscle") else [],
                target_reps=params["reps"],
                sets=sets,
                rest_time_sec=params["rest_sec"],
                exercise_rationale="",
            ),
            goal,
        )

        # Preserve the accepted slot/candidate explanation across the list boundary.
        selected_rationale = getattr(
            sorted_candidates, "_fallback_rationale_by_exercise_id", {}
        ).get(candidate_id) or _candidate_public_rationale(ex)
        plan = LLMExercisePlan(
            exercise_id=str(ex.get("id", _candidate_display_name(ex).replace(" ", "_").lower())),
            exercise_name=_candidate_display_name(ex),
            movement_pattern=ex.get("movement_pattern"),
            movement_type=ex.get("movement_type"),
            primary_muscles=[ex["primary_muscle"]] if ex.get("primary_muscle") else [],
            equipment_type=ex.get("selected_equipment_code"),
            target_weight_kg=None,
            target_reps=target_reps,
            sets=sets,
            rest_time_sec=rest_sec,
            target_rir=2,
            exercise_rationale=selected_rationale,
        )
        apply_candidate_prescription_metadata(plan, ex)
        if (
            not training_method_applied
            and candidate_id in method_candidate_ids
            and routine_blueprint is not None
            and routine_blueprint.training_method
        ):
            apply_training_method_to_plan(
                plan,
                routine_blueprint.training_method,
                source=routine_blueprint.training_method_source,
                reason_code=routine_blueprint.training_method_reason_code,
            )
            training_method_applied = True
        if is_working_prescription(plan):
            apply_experience_prescription(
                plan,
                goal,
                req.experience_level,
                base_reps=target_reps,
                base_rest_sec=rest_sec,
            )
        _apply_readiness_to_exercise(plan, req.readiness_level)
        _apply_large_muscle_volume_guard(
            plan, req.target_split_label, fallback_target_muscles,
            primary_focus=candidate_id in focus_ids
            and not doms_signal["direct_level"] and not doms_signal["secondary_level"],
        )
        _apply_doms_volume_policy(
            plan,
            goal,
            doms_signal["direct_level"],
            doms_signal["secondary_level"],
            experience_level=req.experience_level,
            target_split_label=req.target_split_label,
        )
        plan.target_weight_kg = _resolve_target_weight(plan, goal, recent_sets, profile)
        is_adjunct_plan = bool(
            is_requested_adjunct_candidate
            and apply_adjunct_prescription_profile(plan, matched_adjunct)
        )
        if not joint_upper_budget and routine_blueprint is not None and not is_adjunct_plan:
            # Reserve one working set for each remaining minimum-coverage slot.
            # Earlier main lifts must not consume an accessory's entire budget.
            selected_main_count = sum(block.adjunct_option is None for block in blocks)
            pending_main_count = sum(
                str(candidate.get("id") or "").strip().lower() not in reserved_candidate_ids
                for candidate in sorted_candidates[candidate_index + 1:max_exercise_count]
            )
            reserved_sets = min(
                pending_main_count,
                max(0, minimum_main_count - selected_main_count - 1),
            )
            plan.sets = min(plan.sets, max(1, remaining_sets - reserved_sets))
        while (
            not joint_upper_budget
            and not is_adjunct_plan
            and plan.sets > 1
            and estimate_routine_time_min([*fallback_plans, plan])
            > req.time_available_min
        ):
            plan.sets -= 1

        if (
            not joint_upper_budget
            and routine_blueprint is not None
            and not is_adjunct_plan
            and sum(block.adjunct_option is None for block in blocks) < minimum_main_count
            and estimate_routine_time_min([*fallback_plans, plan]) > req.time_available_min
        ):
            # Try reducing earlier working sets before losing minimum coverage.
            # Work on copies so an impossible fit leaves accepted blocks intact.
            trial = LLMRoutineOutput(
                total_estimated_time=0,
                summary_title="기본 루틴",
                rationale_summary=[],
                exercises=[item.model_copy(deep=True) for item in [*fallback_plans, plan]],
            )
            if trim_routine_to_time_budget(
                trial,
                req.time_available_min,
                protected_exercise_ids={item.exercise_id for item in trial.exercises},
                adjunct_option_by_exercise_id={},
                minimum_main_exercise_count=len(trial.exercises),
            ):
                for accepted, adjusted, block in zip(fallback_plans, trial.exercises, blocks):
                    if block.adjunct_option is None:
                        remaining_sets += accepted.sets - adjusted.sets
                    accepted.sets = adjusted.sets
                    block.prescription = block.prescription[:adjusted.sets]
                plan = trial.exercises[-1]

        if not joint_upper_budget and estimate_routine_time_min([*fallback_plans, plan]) > req.time_available_min:
            if matched_adjunct:
                adjunct_suppression_reasons[matched_adjunct] = (
                    "insufficientTimeAfterMainMinimum"
                )
            continue

        sets = plan.sets
        if not is_adjunct_plan:
            remaining_sets -= sets
        score_context = _candidate_score_context(
            ranked_by_id.get(str(plan.exercise_id or "").strip().lower())
        )

        blocks.append(RoutineBlock(
            order=order,
            exercise_id=plan.exercise_id,
            exercise_name=plan.exercise_name,
            movement_pattern=plan.movement_pattern,
            movement_type=plan.movement_type,
            primary_muscles=plan.primary_muscles,
            equipment_type=plan.equipment_type,
            default_rest_sec=plan.rest_time_sec,
            prescription=_build_prescription(
                sets=plan.sets,
                reps=plan.target_reps,
                weight_kg=plan.target_weight_kg,
                rest_sec=plan.rest_time_sec,
                target_rir=plan.target_rir,
                set_type=getattr(plan, "_prescription_set_type", "working"),
                target_duration_sec=getattr(plan, "_target_duration_sec", None),
                interval_work_sec=getattr(plan, "_interval_work_sec", None),
                interval_recovery_sec=getattr(plan, "_interval_recovery_sec", None),
                interval_rounds=getattr(plan, "_interval_rounds", None),
                target_distance_m=getattr(plan, "_target_distance_m", None),
            ),
            exercise_rationale=plan.exercise_rationale,
            substitution_candidates=[],
            reasons=score_context["reasons"],
            boosts=score_context["boosts"],
            penalties=score_context["penalties"],
            adjunct_option=matched_adjunct if is_adjunct_plan else None,
            training_method=getattr(plan, "_training_method_instruction", None),
            unilateral_prescription=plan._unilateral_prescription,
        ))
        blocks[-1]._weight_guidance = plan._weight_guidance
        fallback_plans.append(plan)
        if is_adjunct_plan and matched_adjunct:
            applied_adjuncts.add(matched_adjunct)
        order += 1
        if not joint_upper_budget and remaining_sets <= 0 and not (set(reserved_adjunct_candidates) - applied_adjuncts):
            break

    if not blocks:
        return enforce_korean_user_text_on_response(RoutineDraftResponse(
            generation_status="failed",
            status_reason_code="emptyCandidate",
            is_fallback=False,
            total_estimated_time=0,
            summary_title="기본 루틴",
            rationale_summary=["모든 후보 운동이 DOMS 제약으로 제외되었습니다."],
            routine_blocks=[],
            warnings=["컨디션이 회복된 후 다시 시도해 주세요."],
        ))

    if routine_blueprint is not None:
        fallback_output = LLMRoutineOutput(
            total_estimated_time=estimate_routine_time_min(fallback_plans),
            summary_title="기본 루틴",
            rationale_summary=["규칙 기반 fallback 처방"],
            exercises=fallback_plans,
        )
        selected_ids = {str(plan.exercise_id or "").strip().lower() for plan in fallback_plans}
        fallback_output._selection_slot_by_exercise_id = {
            exercise_id: slot_id
            for exercise_id, slot_id in getattr(sorted_candidates, "_fallback_slot_by_exercise_id", {}).items()
            if exercise_id in selected_ids
        }
        doms_restricted_exercise_ids = {
            str(candidate.get("id") or "").strip().lower()
            for candidate in candidates
            if (
                _candidate_doms_prescription_signal(candidate, doms)["direct_level"] > 0
                or _candidate_doms_prescription_signal(candidate, doms)["secondary_level"] > 0
            )
        }
        time_fill_diagnostics = _fill_available_time(
            fallback_output,
            req.time_available_min,
            goal,
            req.readiness_level,
            req.target_split_label,
            doms,
            req.experience_level,
            doms_restricted_exercise_ids,
            max_total_sets=max_total_sets,
            primary_focus_exercise_ids=focus_ids,
        )
        adjunct_exercise_ids = {
            str(block.exercise_id or "").strip().lower()
            for block in blocks
            if block.adjunct_option is not None
        }
        completed_main_roles = sum(
            str(plan.exercise_id or "").strip().lower() not in adjunct_exercise_ids
            and getattr(plan, "_prescription_set_type", "working") == "working"
            and not getattr(plan, "_target_duration_sec", None)
            and not getattr(plan, "_target_distance_m", None)
            for plan in fallback_plans
        ) >= minimum_main_count
        sufficient_candidate_pool = len(candidates) >= minimum_main_count
        fallback_selection_diagnostics.update({
            "role_set_capacity_exhausted": bool(
                completed_main_roles
                and sufficient_candidate_pool
                and time_fill_diagnostics["role_set_capacity_exhausted"]
            ),
            "role_set_capacity_total_budget_exhausted": bool(
                time_fill_diagnostics["role_set_capacity_total_budget_exhausted"]
            ),
            "role_set_capacity_eligible_count": int(
                time_fill_diagnostics["role_set_capacity_eligible_count"]
            ),
            "role_set_capacity_restricted_working_role": bool(
                time_fill_diagnostics["role_set_capacity_restricted_working_role"]
            ),
            "role_set_capacity_unmeasured_working_role": bool(
                time_fill_diagnostics["role_set_capacity_unmeasured_working_role"]
            ),
        })
        pre_focus_sets = {
            str(plan.exercise_id or "").strip().lower(): plan.sets
            for plan in fallback_output.exercises
        }
        apply_blueprint_specialization_volume(
            fallback_output,
            routine_blueprint,
            candidates,
            restricted_exercise_ids=doms_restricted_exercise_ids,
        )
        focus_capacity = complete_lower_chest_focus_capacity(
            fallback_output, routine_blueprint, req, candidates, pre_focus_sets,
            doms_db=doms, restricted_exercise_ids=doms_restricted_exercise_ids,
            max_total_sets=max_total_sets,
        )
        if focus_capacity is not None:
            fallback_selection_diagnostics.update(focus_capacity)
            fallback_selection_diagnostics.update(fallback_output._selection_diagnostics)
        if joint_upper_budget:
            phases = {slot.slot_id: slot.phase for slot in routine_blueprint.slots}
            fallback_output._selection_phase_by_exercise_id = {
                eid: phases[slot] for eid, slot in fallback_output._selection_slot_by_exercise_id.items()
            }
            fallback_output = apply_blueprint_execution_policy(fallback_output, routine_blueprint)
            if not allocate_upper_role_sets(
                fallback_output, routine_blueprint, req.time_available_min, max_total_sets
            ):
                return RoutineDraftResponse(
                    generation_status="failed", status_reason_code="schemaError", is_fallback=False,
                    total_estimated_time=0, summary_title="루틴 시간 배분을 확인해 주세요",
                    rationale_summary=["선택된 운동 구성을 시간과 세트 제한 안에 배분하지 못했습니다."],
                    routine_blocks=[], warnings=["가용 시간이나 운동 부위를 조정해 다시 시도해 주세요."],
                )
            fallback_plans = fallback_output.exercises
            fallback_selection_diagnostics.update(fallback_output._selection_diagnostics)
            fallback_selection_diagnostics["role_set_capacity_exhausted"] = False
        plan_by_id = {
            str(plan.exercise_id or "").strip().lower(): plan
            for plan in fallback_plans
        }
        for block in blocks:
            plan = plan_by_id.get(str(block.exercise_id or "").strip().lower())
            if plan is None:
                continue
            block.prescription = _build_prescription(
                sets=plan.sets,
                reps=plan.target_reps,
                weight_kg=plan.target_weight_kg,
                rest_sec=plan.rest_time_sec,
                target_rir=plan.target_rir,
                set_type=getattr(plan, "_prescription_set_type", "working"),
                target_duration_sec=getattr(plan, "_target_duration_sec", None),
                interval_work_sec=getattr(plan, "_interval_work_sec", None),
                interval_recovery_sec=getattr(plan, "_interval_recovery_sec", None),
                interval_rounds=getattr(plan, "_interval_rounds", None),
                target_distance_m=getattr(plan, "_target_distance_m", None),
            )
            block.exercise_rationale = plan.exercise_rationale
            block._weight_guidance = plan._weight_guidance

    for option in requested_adjuncts:
        if option not in applied_adjuncts and option not in adjunct_suppression_reasons:
            adjunct_suppression_reasons[option] = "postValidationRemoved"

    response = RoutineDraftResponse(
        generation_status="fallback",
        status_reason_code=status_reason_code,
        is_fallback=True,
        total_estimated_time=estimate_routine_time_min(fallback_plans),
        summary_title=f"기본 {req.target_split_label or '맞춤형'} 루틴",
        rationale_summary=["AI 코치 연결이 원활하지 않아 기본 루틴으로 대체되었습니다."],
        routine_blocks=blocks,
        warnings=_fallback_request_warnings(req),
    )
    response._adjunct_suppression_reason_by_option = adjunct_suppression_reasons
    response._fallback_selection_diagnostics = fallback_selection_diagnostics
    if joint_upper_budget:
        response = apply_blueprint_execution_policy_to_draft(response, routine_blueprint, candidates)
    return enforce_korean_user_text_on_response(response)

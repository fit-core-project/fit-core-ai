"""apply_deterministic_targets — 위 모듈들을 조율."""
from typing import Dict, List, Optional

from ..schemas import LLMRoutineOutput, RecentSetRecord, UserProfileContext
from .adjuncts import is_working_prescription
from .experience import apply_experience_prescription, normalize_experience_level
from .params import _prescription_params_for_exercise
from .weight import _resolve_target_weight
from .adjustments import (
    _apply_doms_volume_policy,
    _apply_readiness_to_exercise,
    _apply_large_muscle_volume_guard,
    _fill_available_time,
    enforce_total_set_cap,
)


def apply_deterministic_targets(
    llm_output: LLMRoutineOutput,
    goal: str,
    recent_sets: Optional[List[RecentSetRecord]],
    profile: Optional[UserProfileContext],
    readiness_level: Optional[str] = "normal",
    target_split_label: Optional[str] = None,
    target_muscles: Optional[List[str]] = None,
    doms_db: Optional[Dict[str, int]] = None,
    time_available_min: Optional[int] = None,
    max_total_sets: Optional[int] = None,
    protected_exercise_ids: Optional[set[str]] = None,
    doms_level_by_exercise_id: Optional[Dict[str, int]] = None,
    secondary_doms_level_by_exercise_id: Optional[Dict[str, int]] = None,
    experience_level: Optional[str] = "intermediate",
    primary_focus_exercise_ids: Optional[set[str]] = None,
) -> LLMRoutineOutput:
    """
    반복 수와 여력은 서버 정책으로 정하고, kg은 비교 가능한 기록만 참조한다.
    기준 중량의 종목 간 환산이나 LLM의 kg 추정은 사용하지 않는다.
    """
    adjusted = llm_output.model_copy(deep=True)
    restricted_ids = {
        exercise_id
        for exercise_id in {
            *(doms_level_by_exercise_id or {}),
            *(secondary_doms_level_by_exercise_id or {}),
        }
        if int((doms_level_by_exercise_id or {}).get(exercise_id, 0) or 0) > 0
        or int((secondary_doms_level_by_exercise_id or {}).get(exercise_id, 0) or 0) > 0
    }
    focus_ids = {
        str(value or "").strip().lower()
        for value in (primary_focus_exercise_ids or set())
    } - restricted_ids if (
        normalize_experience_level(experience_level) == "advanced"
        and (readiness_level or "normal").strip().lower() != "low"
    ) else set()

    for exercise in adjusted.exercises:
        if not is_working_prescription(exercise):
            continue
        target_reps, rest_sec = _prescription_params_for_exercise(exercise, goal)
        apply_experience_prescription(
            exercise,
            goal,
            experience_level,
            base_reps=target_reps,
            base_rest_sec=rest_sec,
        )
        _apply_readiness_to_exercise(exercise, readiness_level)
        exercise_id = str(exercise.exercise_id or "").strip().lower()
        _apply_large_muscle_volume_guard(
            exercise, target_split_label, target_muscles,
            primary_focus=exercise_id in focus_ids
            and not any((doms_db or {}).get(muscle, 0) for muscle in exercise.primary_muscles),
        )
        _apply_doms_volume_policy(
            exercise,
            goal,
            int((doms_level_by_exercise_id or {}).get(exercise_id, 0) or 0),
            int((secondary_doms_level_by_exercise_id or {}).get(exercise_id, 0) or 0),
            experience_level=experience_level,
            target_split_label=target_split_label,
        )
        exercise.target_weight_kg = _resolve_target_weight(exercise, goal, recent_sets, profile)

    if time_available_min:
        adjusted._time_fill_diagnostics = _fill_available_time(
            adjusted,
            time_available_min,
            goal,
            readiness_level,
            target_split_label,
            doms_db=doms_db,
            experience_level=experience_level,
            doms_restricted_exercise_ids=restricted_ids,
            primary_focus_exercise_ids=focus_ids,
        )
    else:
        adjusted._time_fill_diagnostics = {
            "role_set_capacity_exhausted": False,
            "role_set_capacity_total_budget_exhausted": False,
            "role_set_capacity_eligible_count": 0,
            "role_set_capacity_restricted_working_role": False,
            "role_set_capacity_unmeasured_working_role": False,
        }

    if max_total_sets is not None:
        if enforce_total_set_cap(
            adjusted,
            max_total_sets,
            target_muscles,
            protected_exercise_ids=protected_exercise_ids,
        ):
            adjusted._time_fill_diagnostics["role_set_capacity_exhausted"] = False
            adjusted._time_fill_diagnostics[
                "role_set_capacity_total_budget_exhausted"
            ] = True

    return adjusted

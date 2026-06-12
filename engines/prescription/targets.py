"""apply_deterministic_targets — 위 모듈들을 조율."""
from typing import Dict, List, Optional

from ..schemas import LLMRoutineOutput, RecentSetRecord, UserProfileContext
from .params import _prescription_params_for_exercise
from .weight import _resolve_target_weight
from .adjustments import (
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
) -> LLMRoutineOutput:
    """
    LLM의 kg/reps는 hint로만 두고 최종 값은 서버가 확정한다.
    최근 세트 > strength_baseline > LLM hint 순으로 참조한다.
    """
    adjusted = llm_output.model_copy(deep=True)

    for exercise in adjusted.exercises:
        target_reps, rest_sec = _prescription_params_for_exercise(exercise, goal)
        exercise.target_reps = target_reps
        exercise.rest_time_sec = rest_sec
        _apply_readiness_to_exercise(exercise, readiness_level)
        _apply_large_muscle_volume_guard(exercise, target_split_label, target_muscles)
        exercise.target_weight_kg = _resolve_target_weight(exercise, goal, recent_sets, profile)

    if time_available_min:
        _fill_available_time(
            adjusted,
            time_available_min,
            goal,
            readiness_level,
            target_split_label,
            doms_db=doms_db,
        )

    if max_total_sets is not None:
        enforce_total_set_cap(adjusted, max_total_sets, target_muscles)

    return adjusted

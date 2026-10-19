"""Use comparable recorded loads as references; never invent calibrated kg."""
import math
from typing import List, Optional

from ..schemas import LLMExercisePlan, RecentSetRecord, UserProfileContext


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _set_weight_guidance(exercise: LLMExercisePlan, guidance: str) -> None:
    previous = exercise._weight_guidance
    rationale = exercise.exercise_rationale
    if previous:
        rationale = rationale.replace(previous, "").strip()
    exercise.exercise_rationale = f"{rationale} {guidance}".strip()
    exercise._weight_guidance = guidance or None


def _uncalibrated_weight_guidance(exercise: LLMExercisePlan) -> str:
    rir = exercise.target_rir
    reserve = f"{rir}회 여력을 남기도록" if _finite_number(rir) and rir > 0 else "여력을 남기도록"
    return (
        f"목표 횟수 뒤 {reserve} 첫 세트에서 중량을 맞추세요. "
        "자동 증량하지 않으며, 여력이 부족하면 낮추세요."
    )


def invalidate_weight_for_training_method(exercise: LLMExercisePlan) -> None:
    """Method selection may occur after ordinary-repetition prescription."""
    exercise.target_weight_kg = _resolve_target_weight(exercise, "", None, None)


def _find_recent_reference(
    exercise: LLMExercisePlan,
    recent_sets: Optional[List[RecentSetRecord]],
) -> Optional[RecentSetRecord]:
    # Queries supply newest-first records. Do not skip an unsuitable newest
    # record to retrieve a heavier or more convenient older performance.
    exercise_id = exercise.exercise_id.strip()
    if not exercise_id:
        return None
    return next((record for record in recent_sets or []
                 if record.exercise_id and record.exercise_id.strip() == exercise_id), None)


def _resolve_target_weight(
    exercise: LLMExercisePlan,
    goal: str,
    recent_sets: Optional[List[RecentSetRecord]],
    profile: Optional[UserProfileContext],
) -> Optional[float]:
    # Keep the shared API, without goal coefficients or other lifts' baselines.
    if (
        "BODYWEIGHT" in (exercise.equipment_type or "").upper()
        or exercise._prescription_set_type != "working"
        or exercise._target_duration_sec or exercise._target_distance_m
    ):
        _set_weight_guidance(exercise, "")
        return None
    guidance = _uncalibrated_weight_guidance(exercise)
    rir = exercise.target_rir
    recent = _find_recent_reference(exercise, recent_sets)
    comparable = (
        not exercise._training_method_code
        and _finite_number(rir) and 0 < rir < 3
        and isinstance(exercise.target_reps, int) and not isinstance(exercise.target_reps, bool)
        and exercise.target_reps > 0
        and recent is not None
        and _finite_number(recent.weight_kg) and recent.weight_kg > 0
        and isinstance(recent.reps, int) and not isinstance(recent.reps, bool)
        and recent.reps == exercise.target_reps
        and _finite_number(recent.rir) and recent.rir == rir
        and recent.is_failure is False
    )
    if comparable:
        guidance = (
            f"동일 운동에서 {recent.reps}회, 여유 {recent.rir:g}회로 기록한 "
            f"{recent.weight_kg:g}kg을 증량 없이 시작 참고값으로 제시합니다. "
            "기록 시점과 기구 설정·중량 표기 기준을 확인하세요. "
            "현재 상태나 기구가 다르면 그대로 적용하지 말고, "
            "첫 세트에서 목표 횟수와 여력에 맞춰 중량을 조절하세요."
        )
    _set_weight_guidance(exercise, guidance)
    return float(recent.weight_kg) if comparable else None

"""목표 중량 추정 — 5단계 폴백 로직."""
from typing import List, Optional, Tuple

from ..schemas import LLMExercisePlan, RecentSetRecord, UserProfileContext
from .estimator import _movement_type_key, calculate_1rm
from .params import _round_to_nearest_2_5, _goal_key

_GOAL_INTENSITY = {
    "strength": 0.85,
    "hypertrophy": 0.72,
    "endurance": 0.60,
    "fatloss": 0.60,
    "recomposition": 0.72,
    "generalfitness": 0.65,
}

_BIG_FOUR_BASELINE = {
    "overhead_press": {"ids": {"75", "barbell_overhead_press"}, "ratio": 0.70},
    "bench_press":    {"ids": {"30", "barbell_bench_press"},    "ratio": 0.75},
    "deadlift":       {"ids": {"7",  "deadlift"},               "ratio": 0.60},
    "squat":          {"ids": {"98", "back_squat"},             "ratio": 0.65},
}

_MUSCLE_TO_BIG_FOUR = {
    "chest":          "bench_press",
    "triceps":        "bench_press",
    "front-deltoids": "overhead_press",
    "back-deltoids":  "overhead_press",
    "trapezius":      "overhead_press",
    "upper-back":     "deadlift",
    "lower-back":     "deadlift",
    "hamstring":      "deadlift",
    "gluteal":        "squat",
    "quadriceps":     "squat",
    "adductor":       "squat",
    "abductors":      "squat",
    "calves":         "squat",
}


def _find_recent_reference(
    exercise: LLMExercisePlan,
    recent_sets: Optional[List[RecentSetRecord]],
) -> Optional[RecentSetRecord]:
    if not recent_sets:
        return None
    exercise_id = exercise.exercise_id.strip().lower()
    for record in recent_sets:
        if record.exercise_id and record.exercise_id.strip().lower() == exercise_id and record.weight_kg:
            return record

    normalized_names = {exercise.exercise_name.strip().lower(), exercise_id}
    matching = [
        record for record in recent_sets
        if record.exercise_name.strip().lower() in normalized_names and record.weight_kg
    ]
    return matching[0] if matching else None


def _find_same_muscle_recent_reference(
    exercise: LLMExercisePlan,
    recent_sets: Optional[List[RecentSetRecord]],
) -> Optional[RecentSetRecord]:
    if not recent_sets or not exercise.primary_muscles:
        return None
    primary = exercise.primary_muscles[0]
    same_muscle = [
        record
        for record in recent_sets
        if record.primary_muscle == primary and record.weight_kg
    ]
    if not same_muscle:
        return None
    return max(same_muscle, key=lambda record: calculate_1rm(record.weight_kg or 0, record.reps))


def _find_baseline_reference(
    exercise: LLMExercisePlan,
    profile: Optional[UserProfileContext],
) -> Optional[Tuple[float, int]]:
    if not profile or not profile.strength_baseline:
        return None
    normalized_keys = {exercise.exercise_name.strip().lower(), exercise.exercise_id.strip().lower()}
    for key, value in profile.strength_baseline.items():
        if not isinstance(value, dict):
            continue
        value_keys = {
            key.strip().lower(),
            str(value.get("exercise_id") or "").strip().lower(),
            str(value.get("exerciseId") or "").strip().lower(),
            str(value.get("exercise_name_snapshot") or "").strip().lower(),
            str(value.get("exerciseNameSnapshot") or "").strip().lower(),
        }
        if not (value_keys & normalized_keys):
            continue
        weight = value.get("weight_kg")
        if weight is None:
            weight = value.get("workingWeightKg") or value.get("working_weight_kg")
        reps = value.get("reps")
        if weight and reps:
            return float(weight), int(reps)
    return None


def _find_big_four_baseline_reference(
    exercise: LLMExercisePlan,
    profile: Optional[UserProfileContext],
) -> Optional[Tuple[float, int, float]]:
    if not profile or not profile.strength_baseline or not exercise.primary_muscles:
        return None
    primary = exercise.primary_muscles[0]
    baseline_key = _MUSCLE_TO_BIG_FOUR.get(primary)
    if not baseline_key:
        return None

    target = _BIG_FOUR_BASELINE[baseline_key]
    target_ids = target["ids"]
    base_ratio = float(target["ratio"])
    movement_ratio = 1.0 if _movement_type_key(exercise.movement_type) == "COMPOUND" else 0.55

    for key, value in profile.strength_baseline.items():
        if not isinstance(value, dict):
            continue
        value_ids = {
            key.strip().lower(),
            str(value.get("exercise_id") or "").strip().lower(),
            str(value.get("exerciseId") or "").strip().lower(),
        }
        if not (value_ids & target_ids):
            continue
        weight = value.get("weight_kg")
        if weight is None:
            weight = value.get("workingWeightKg") or value.get("working_weight_kg")
        reps = value.get("reps")
        if weight and reps:
            return float(weight), int(reps), base_ratio * movement_ratio
    return None


def _resolve_target_weight(
    exercise: LLMExercisePlan,
    goal: str,
    recent_sets: Optional[List[RecentSetRecord]],
    profile: Optional[UserProfileContext],
) -> Optional[float]:
    equipment = (exercise.equipment_type or "").upper()
    if "BODYWEIGHT" in equipment:
        return None

    intensity = _GOAL_INTENSITY.get(_goal_key(goal), _GOAL_INTENSITY["hypertrophy"])
    exact_recent = _find_recent_reference(exercise, recent_sets)
    if exact_recent and exact_recent.weight_kg:
        base = _round_to_nearest_2_5(calculate_1rm(exact_recent.weight_kg, exact_recent.reps) * intensity)
        # 점진적 과부하: 마지막 세트 RIR >= 2이고 실패하지 않았으면 +2.5kg 제안
        if exact_recent.rir is not None and exact_recent.rir >= 2.0 and not exact_recent.is_failure:
            return base + 2.5
        return base

    same_muscle_recent = _find_same_muscle_recent_reference(exercise, recent_sets)
    if same_muscle_recent and same_muscle_recent.weight_kg:
        return _round_to_nearest_2_5(calculate_1rm(same_muscle_recent.weight_kg, same_muscle_recent.reps) * intensity * 0.85)

    exact_baseline = _find_baseline_reference(exercise, profile)
    if exact_baseline:
        return _round_to_nearest_2_5(calculate_1rm(exact_baseline[0], exact_baseline[1]) * intensity)

    big_four = _find_big_four_baseline_reference(exercise, profile)
    if big_four:
        weight, reps, ratio = big_four
        return _round_to_nearest_2_5(calculate_1rm(weight, reps) * intensity * ratio)

    if exercise.target_weight_kg is not None:
        return max(0, _round_to_nearest_2_5(exercise.target_weight_kg))
    return None

"""목표 중량 추정 — 같은 운동 기록 > 본인 기준 > 같은 근육 기록 > 빅4 기준 > LLM 힌트."""
from typing import List, Optional, Tuple

from ..schemas import LLMExercisePlan, RecentSetRecord, UserProfileContext
from .estimator import _movement_type_key, calculate_1rm
from .params import _round_to_nearest_2_5

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


# Load an implement carries relative to a barbell for the same muscle (dumbbell = per hand).
_EQUIPMENT_LOAD_FACTOR = {
    "BARBELL": 1.0,
    "SMITH_MACHINE": 0.95,
    "MACHINE": 1.0,
    "CABLE": 0.65,
    "DUMBBELL": 0.4,
    "KETTLEBELL": 0.4,
    "PLATE": 0.3,
}
_MOVEMENT_LOAD_FACTOR = {"COMPOUND": 1.0, "ISOLATION": 0.55, "STATIC": 0.55}
# A borrowed estimate (other exercise, same muscle) starts a little conservative.
_BORROWED_SAFETY = 0.9
# Plain-number guard: no single implement prescription above this.
_MAX_TARGET_KG = 400.0


def _primary_equipment(raw: Optional[str]) -> str:
    tokens = [t.strip().upper() for t in str(raw or "").split(",") if t.strip()]
    return tokens[0] if tokens else ""


def _load_factor(equipment: Optional[str], movement_type: Optional[str]) -> float:
    eq = _EQUIPMENT_LOAD_FACTOR.get(_primary_equipment(equipment), 1.0)
    mv = _MOVEMENT_LOAD_FACTOR.get(_movement_type_key(movement_type), 1.0)
    return eq * mv


def _record_rir(record: RecentSetRecord) -> float:
    if record.is_failure:
        return 0.0
    if record.rir is not None:
        return max(0.0, float(record.rir))
    if record.rpe is not None:
        return max(0.0, 10.0 - float(record.rpe))
    return 0.0


def _estimated_1rm(weight: float, reps: int, rir: float = 0.0) -> float:
    """Epley with reps in reserve: a set of 8 at RIR 2 is treated as a 10-rep max."""
    return calculate_1rm(weight, int(round(reps + rir)))


def _weight_for_reps(one_rm: float, target_reps: int, target_rir: Optional[int]) -> float:
    effort = max(1, int(target_reps) + int(target_rir if target_rir is not None else 2))
    if effort <= 1:
        return one_rm
    return one_rm / (1 + effort / 30.0)


def _best_exact_record(exercise: LLMExercisePlan, recent_sets: Optional[List[RecentSetRecord]]) -> Optional[RecentSetRecord]:
    """Strongest recent set of this exact exercise (not the tired last set)."""
    if not recent_sets:
        return None
    exercise_id = exercise.exercise_id.strip().lower()
    names = {exercise.exercise_name.strip().lower(), exercise_id}
    matches = [
        r for r in recent_sets
        if r.weight_kg and (
            (r.exercise_id and r.exercise_id.strip().lower() == exercise_id)
            or r.exercise_name.strip().lower() in names
        )
    ]
    if not matches:
        return None
    return max(matches, key=lambda r: _estimated_1rm(r.weight_kg or 0, r.reps, _record_rir(r)))


def _finish(weight: float) -> Optional[float]:
    if weight <= 0:
        return None
    return min(_MAX_TARGET_KG, _round_to_nearest_2_5(weight))


def _resolve_target_weight(
    exercise: LLMExercisePlan,
    goal: str,
    recent_sets: Optional[List[RecentSetRecord]],
    profile: Optional[UserProfileContext],
) -> Optional[float]:
    """Order: same exercise history > own baseline > same-muscle history > big-four baseline > LLM hint.

    Weight is solved from an estimated 1RM for this exercise's target reps + RIR, so
    a lighter rep target never drops below what the user just lifted for those reps.
    """
    if _primary_equipment(exercise.equipment_type) in ("", "BODYWEIGHT", "BAND", "BALL", "BOX"):
        return None

    target_reps = exercise.target_reps
    target_rir = exercise.target_rir
    target_factor = _load_factor(exercise.equipment_type, exercise.movement_type)

    exact = _best_exact_record(exercise, recent_sets)
    if exact and exact.weight_kg:
        one_rm = _estimated_1rm(exact.weight_kg, exact.reps, _record_rir(exact))
        weight = _weight_for_reps(one_rm, target_reps, target_rir)
        # Progressive overload: a clean set with reps in reserve earns a small step up.
        if not exact.is_failure and _record_rir(exact) >= 2.0:
            weight += 2.5
        return _finish(weight)

    exact_baseline = _find_baseline_reference(exercise, profile)
    if exact_baseline:
        return _finish(_weight_for_reps(_estimated_1rm(*exact_baseline), target_reps, target_rir))

    same_muscle = _find_same_muscle_recent_reference(exercise, recent_sets)
    if same_muscle and same_muscle.weight_kg:
        source_factor = _load_factor(same_muscle.equipment_type, same_muscle.movement_type)
        one_rm = _estimated_1rm(same_muscle.weight_kg, same_muscle.reps, _record_rir(same_muscle))
        scaled = one_rm * (target_factor / source_factor) * _BORROWED_SAFETY
        return _finish(_weight_for_reps(scaled, target_reps, target_rir))

    big_four = _find_big_four_baseline_reference(exercise, profile)
    if big_four:
        weight, reps, ratio = big_four
        # ratio already holds the compound/isolation step; add the implement step only.
        implement = _EQUIPMENT_LOAD_FACTOR.get(_primary_equipment(exercise.equipment_type), 1.0)
        return _finish(_weight_for_reps(_estimated_1rm(weight, reps) * ratio * implement, target_reps, target_rir))

    if exercise.target_weight_kg is not None:
        return _finish(exercise.target_weight_kg)
    return None

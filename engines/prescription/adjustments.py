"""운동 세트/중량/시간 조정 로직."""
import math
from typing import Dict, List, Optional

from ..schemas import LLMExercisePlan, LLMRoutineOutput
from ..registry import ACCESSORY_MUSCLE_SLUGS, LARGE_MUSCLE_SLUGS, ACCESSORY_PRIMARY_SPLITS
from .estimator import _movement_type_key, estimate_routine_time_min, _work_seconds_for_movement
from .params import _goal_key, GOAL_PARAMS


def _apply_readiness_to_exercise(exercise: LLMExercisePlan, readiness_level: Optional[str]) -> None:
    readiness = (readiness_level or "normal").strip().lower()
    movement_type = _movement_type_key(exercise.movement_type)
    base_rir = exercise.target_rir if exercise.target_rir is not None else 2

    if readiness == "low" and movement_type == "COMPOUND":
        if exercise.sets > 2:
            exercise.sets = max(2, exercise.sets - 1)
        exercise.target_rir = min(5, base_rir + 2)
    elif readiness == "high":
        exercise.target_rir = max(0, base_rir - 1)


def _uses_large_muscle_guard(
    target_split_label: Optional[str],
    target_muscles: Optional[List[str]],
) -> bool:
    split = (target_split_label or "").strip().lower()
    if split in ACCESSORY_PRIMARY_SPLITS:
        return False
    if target_muscles and not (set(target_muscles) & LARGE_MUSCLE_SLUGS):
        return False
    return True


def _apply_large_muscle_volume_guard(
    exercise: LLMExercisePlan,
    target_split_label: Optional[str],
    target_muscles: Optional[List[str]],
) -> None:
    if not _uses_large_muscle_guard(target_split_label, target_muscles):
        return

    primary = exercise.primary_muscles[0] if exercise.primary_muscles else ""
    movement_type = _movement_type_key(exercise.movement_type)
    if primary in ACCESSORY_MUSCLE_SLUGS and movement_type != "COMPOUND":
        exercise.sets = min(exercise.sets, 2)


def _max_sets_for_exercise(exercise: LLMExercisePlan, goal: str, target_split_label: Optional[str]) -> int:
    goal_key = _goal_key(goal)
    movement_type = _movement_type_key(exercise.movement_type)
    primary = exercise.primary_muscles[0] if exercise.primary_muscles else ""
    if movement_type == "COMPOUND":
        return 5 if goal_key == "strength" else 4
    if (target_split_label or "").lower() in ACCESSORY_PRIMARY_SPLITS:
        return 4
    if primary in LARGE_MUSCLE_SLUGS:
        return 3
    return 2


def _calculate_max_total_sets(time_available_min: int, goal: str) -> int:
    rest_sec = GOAL_PARAMS.get(goal.lower(), GOAL_PARAMS["hypertrophy"])["rest_sec"]
    available_sec = max(0, time_available_min - 5) * 60
    per_set_sec = _work_seconds_for_movement("COMPOUND") + rest_sec
    return max(1, int(available_sec / per_set_sec))


def _target_exercise_count(time_available_min: int) -> int:
    if time_available_min <= 30:
        return 4
    if time_available_min <= 45:
        return 5
    if time_available_min <= 60:
        return 6
    if time_available_min <= 75:
        return 7
    return 8


def _fill_available_time(
    output: LLMRoutineOutput,
    time_available_min: int,
    goal: str,
    readiness_level: Optional[str],
    target_split_label: Optional[str],
    doms_db: Optional[Dict[str, int]] = None,
) -> None:
    if (readiness_level or "normal").lower() == "low":
        return
    target_min = max(1, math.floor(time_available_min * 0.85))
    current = estimate_routine_time_min(output.exercises)
    if current >= target_min:
        return

    candidates = sorted(
        output.exercises,
        key=lambda ex: (
            0 if _movement_type_key(ex.movement_type) == "COMPOUND" else 1,
            0 if (ex.primary_muscles and ex.primary_muscles[0] in LARGE_MUSCLE_SLUGS) else 1,
        ),
    )
    doms = doms_db or {}
    changed = True
    while changed and current < target_min:
        changed = False
        for exercise in candidates:
            primary = exercise.primary_muscles[0] if exercise.primary_muscles else ""
            if doms.get(primary, 0) > 0:
                continue
            max_sets = _max_sets_for_exercise(exercise, goal, target_split_label)
            if exercise.sets >= max_sets:
                continue
            exercise.sets += 1
            next_time = estimate_routine_time_min(output.exercises)
            if next_time > time_available_min:
                exercise.sets -= 1
                continue
            current = next_time
            changed = True
            if current >= target_min:
                break

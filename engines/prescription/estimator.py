"""루틴 시간 추정 및 1RM 계산 (순수 함수)."""
import math
from typing import Any, Dict, List, Optional

from ..schemas import LLMExercisePlan
from .unilateral import unilateral_work_seconds, unilateral_rest_seconds

WARMUP_SECONDS = 5 * 60
EXERCISE_TRANSITION_SECONDS = 2 * 60

def _movement_type_key(value: Optional[str]) -> str:
    normalized = (value or "").strip().upper()
    return normalized if normalized in {
        "COMPOUND",
        "ISOLATION",
        "STATIC",
        "MOBILITY",
        "CONDITIONING",
        "CARRY",
    } else "UNKNOWN"


def _work_seconds_for_movement(movement_type: Optional[str]) -> int:
    return {
        "COMPOUND": 60,
        "ISOLATION": 40,
        "STATIC": 45,
        "MOBILITY": 30,
        "CONDITIONING": 45,
        "CARRY": 45,
    }.get(_movement_type_key(movement_type), 45)


def _work_seconds_for_exercise(exercise: LLMExercisePlan, set_index: int = 0) -> int:
    unilateral = unilateral_work_seconds(exercise, set_index)
    if unilateral is not None:
        return unilateral
    duration_sec = getattr(exercise, "_target_duration_sec", None)
    if duration_sec is not None:
        return max(0, int(duration_sec))
    distance_m = getattr(exercise, "_target_distance_m", None)
    if distance_m is not None:
        return max(20, round(int(distance_m) * 1.5))
    base_seconds = _work_seconds_for_movement(exercise.movement_type)
    instruction = getattr(exercise, "_training_method_instruction", None) or {}
    reps = max(1, int(exercise.target_reps or 1))
    pause_sec = max(0, int(instruction.get("pauseSec") or 0))
    eccentric_sec = max(0, int(instruction.get("eccentricSec") or 0))
    # The base model already budgets roughly two seconds per repetition.
    # Pauses are fully additive; tempo eccentric adds only time beyond that
    # ordinary two-second movement assumption.
    return (
        base_seconds
        + reps * pause_sec
        + reps * max(0, eccentric_sec - 2)
    )


def estimate_routine_time_min(
    exercises: List[LLMExercisePlan],
    execution_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> int:
    """
    서버 기준 예상 시간. LLM 산수는 신뢰하지 않고 최종 루틴 블록에서 재계산한다.
    - warmup: 5분
    - 운동 간 전환: 각 2분
    - set 수행 시간: compound 60초, isolation 40초, static/unknown 45초
    - 휴식은 같은 운동의 세트 사이에만 계산한다.
    """
    active = [exercise for exercise in exercises if exercise.sets > 0]
    if not active:
        return 0

    metadata = execution_metadata or {}
    total_sec = WARMUP_SECONDS
    if not metadata:
        total_sec += max(0, len(active) - 1) * EXERCISE_TRANSITION_SECONDS
        for exercise in active:
            sets = max(0, exercise.sets)
            total_sec += sum(_work_seconds_for_exercise(exercise, i) for i in range(sets))
            total_sec += sum(unilateral_rest_seconds(exercise, i) for i in range(max(0, sets - 1)))
        return max(1, math.ceil(total_sec / 60))

    groups: list[list[LLMExercisePlan]] = []
    group_indexes: dict[str, int] = {}
    for exercise in active:
        exercise_id = str(exercise.exercise_id or "").strip().lower()
        item = metadata.get(exercise_id, {})
        group_id = str(item.get("executionGroupId") or f"single:{exercise_id}")
        if group_id not in group_indexes:
            group_indexes[group_id] = len(groups)
            groups.append([])
        groups[group_indexes[group_id]].append(exercise)

    total_sec += max(0, len(groups) - 1) * EXERCISE_TRANSITION_SECONDS
    for group in groups:
        first_id = str(group[0].exercise_id or "").strip().lower()
        group_type = str(metadata.get(first_id, {}).get("executionGroupType") or "single")
        if group_type == "single" or len(group) == 1:
            exercise = group[0]
            sets = max(0, exercise.sets)
            total_sec += sum(_work_seconds_for_exercise(exercise, i) for i in range(sets))
            total_sec += sum(unilateral_rest_seconds(exercise, i) for i in range(max(0, sets - 1)))
            continue

        rounds = max(max(0, exercise.sets) for exercise in group)
        for round_index in range(rounds):
            active_in_round = [exercise for exercise in group if exercise.sets > round_index]
            for position, exercise in enumerate(active_in_round):
                total_sec += _work_seconds_for_exercise(exercise, round_index)
                if position < len(active_in_round) - 1:
                    exercise_id = str(exercise.exercise_id or "").strip().lower()
                    total_sec += max(
                        0,
                        int(metadata.get(exercise_id, {}).get("transitionRestSec") or 0),
                    )
            if round_index < rounds - 1:
                last_id = str(active_in_round[-1].exercise_id or "").strip().lower()
                total_sec += max(
                    0,
                    int(metadata.get(last_id, {}).get("groupRestSec") or 0),
                )
    return max(1, math.ceil(total_sec / 60))


def calculate_1rm(weight: int, reps: int) -> float:
    """1RM 계산 (Epley 공식)"""
    if reps <= 1:
        return float(weight)
    return weight * (1 + reps / 30.0)

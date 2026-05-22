"""루틴 시간 추정 및 1RM 계산 (순수 함수)."""
import math
from typing import List, Optional

from ..schemas import LLMExercisePlan


def _movement_type_key(value: Optional[str]) -> str:
    normalized = (value or "").strip().upper()
    return normalized if normalized in {"COMPOUND", "ISOLATION", "STATIC"} else "UNKNOWN"


def _work_seconds_for_movement(movement_type: Optional[str]) -> int:
    return {
        "COMPOUND": 60,
        "ISOLATION": 40,
        "STATIC": 45,
    }.get(_movement_type_key(movement_type), 45)


def estimate_routine_time_min(exercises: List[LLMExercisePlan]) -> int:
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

    total_sec = 5 * 60
    total_sec += max(0, len(active) - 1) * 2 * 60
    for exercise in active:
        sets = max(0, exercise.sets)
        work_sec = _work_seconds_for_movement(exercise.movement_type)
        rest_sec = max(0, exercise.rest_time_sec)
        total_sec += sets * work_sec
        total_sec += max(0, sets - 1) * rest_sec
    return max(1, math.ceil(total_sec / 60))


def calculate_1rm(weight: int, reps: int) -> float:
    """1RM 계산 (Epley 공식)"""
    if reps <= 1:
        return float(weight)
    return weight * (1 + reps / 30.0)

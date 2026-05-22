"""목표별 rep/rest 룩업 테이블 및 Fallback 파라미터."""
from typing import Tuple

from ..schemas import LLMExercisePlan
from .estimator import _movement_type_key

# Fallback 루틴 및 최대 세트 계산에서 공유하는 목표별 기본 파라미터
GOAL_PARAMS = {
    "strength":       {"sets": 5, "reps": 5,  "rest_sec": 180},
    "hypertrophy":    {"sets": 3, "reps": 10, "rest_sec": 90},
    "endurance":      {"sets": 3, "reps": 15, "rest_sec": 60},
    "fatloss":        {"sets": 3, "reps": 15, "rest_sec": 60},
    "recomposition":  {"sets": 3, "reps": 10, "rest_sec": 90},
    "generalfitness": {"sets": 3, "reps": 12, "rest_sec": 75},
}


def _round_to_nearest_2_5(value: float) -> float:
    return round(value / 2.5) * 2.5


def _goal_key(goal: str) -> str:
    return (goal or "hypertrophy").replace("_", "").lower()


def _prescription_params_for_exercise(exercise: LLMExercisePlan, goal: str) -> Tuple[int, int]:
    goal_key = _goal_key(goal)
    movement_type = _movement_type_key(exercise.movement_type)
    table = {
        "strength": {
            "COMPOUND": (5, 180),
            "ISOLATION": (8, 90),
            "STATIC": (30, 60),
            "UNKNOWN": (6, 120),
        },
        "hypertrophy": {
            "COMPOUND": (8, 120),
            "ISOLATION": (12, 75),
            "STATIC": (30, 60),
            "UNKNOWN": (10, 90),
        },
        "recomposition": {
            "COMPOUND": (8, 90),
            "ISOLATION": (12, 60),
            "STATIC": (30, 45),
            "UNKNOWN": (10, 75),
        },
        "fatloss": {
            "COMPOUND": (12, 75),
            "ISOLATION": (15, 45),
            "STATIC": (30, 45),
            "UNKNOWN": (15, 60),
        },
        "generalfitness": {
            "COMPOUND": (10, 90),
            "ISOLATION": (12, 60),
            "STATIC": (30, 45),
            "UNKNOWN": (12, 75),
        },
        "endurance": {
            "COMPOUND": (15, 60),
            "ISOLATION": (15, 45),
            "STATIC": (30, 45),
            "UNKNOWN": (15, 60),
        },
    }
    return table.get(goal_key, table["hypertrophy"]).get(movement_type, table["hypertrophy"]["UNKNOWN"])

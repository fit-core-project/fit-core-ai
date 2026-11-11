"""Shared trimming order for set/time budgets.

Every budget trim follows the same order so accessories are not left at 1 set and the
exercise that brings a split's only triceps / biceps / front-delt work is not the first
to go:
  1. reduce sets down to 2 (lowest-priority exercise first),
  2. remove whole exercises, preferring ones whose primary muscle is already covered,
  3. only then allow 1-set exercises.
"""
from __future__ import annotations

from typing import Callable, List, Optional

from ..schemas import LLMExercisePlan

MIN_SETS_BEFORE_REMOVAL = 2


def _primary(exercise: LLMExercisePlan) -> str:
    return (exercise.primary_muscles or [""])[0]


def removal_index(exercises: List[LLMExercisePlan]) -> int:
    """Index of the exercise to drop: the last one whose muscle repeats, else the last one."""
    muscles = [_primary(e) for e in exercises]
    for i in range(len(exercises) - 1, -1, -1):
        if muscles.count(muscles[i]) > 1:
            return i
    return len(exercises) - 1


def trim_to_budget(
    exercises: List[LLMExercisePlan],
    over_budget: Callable[[], bool],
    reduce_order: Optional[Callable[[List[LLMExercisePlan]], List[LLMExercisePlan]]] = None,
    on_remove: Optional[Callable[[LLMExercisePlan], None]] = None,
) -> bool:
    """Trim in place until over_budget() is False. Returns True when it fits."""
    order = reduce_order or (lambda items: list(reversed(items)))
    for floor in (MIN_SETS_BEFORE_REMOVAL, None):
        if floor is None:
            while over_budget() and len(exercises) > 1:
                removed = exercises.pop(removal_index(exercises))
                if on_remove is not None:
                    on_remove(removed)
            floor = 1
        # Drain the lowest-priority exercise to the floor before touching the next one.
        while over_budget():
            target = next((e for e in order(exercises) if e.sets > floor), None)
            if target is None:
                break
            target.sets -= 1
    return not over_budget()

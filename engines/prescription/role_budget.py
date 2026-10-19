"""Candidate-capacity ceiling, not a prescription or a minimum volume target."""
from collections.abc import Sequence

from ..schemas import LLMExercisePlan
from .estimator import (
    EXERCISE_TRANSITION_SECONDS,
    WARMUP_SECONDS,
    _work_seconds_for_exercise,
)


def role_time_set_ceiling(
    roles: Sequence[LLMExercisePlan],
    time_available_min: int,
    max_exercises: int,
    *,
    grouped: bool = False,
) -> int:
    """Maximize count under each supplied role cap and the timing model.

    Candidate families/required slots are deliberately relaxed: this is only
    an admission ceiling, never a chosen routine. For grouped sessions, omit
    rest/transition costs to avoid assuming independent-set rests. The exact
    group/equipment validator and final estimator remain authoritative.
    """
    if max_exercises <= 0 or not roles:
        return 0
    ids = [str(role.exercise_id).strip().lower() for role in roles]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate IDs in role budget")
    available_sec = max(0, int(time_available_min) * 60 - WARMUP_SECONDS)
    # (selected exercise count, total sets) -> cheapest achievable seconds.
    states = {(0, 0): 0}
    for role in roles:
        next_states = dict(states)
        work = _work_seconds_for_exercise(role)
        rest = 0 if grouped else max(0, role.rest_time_sec)
        for (count, total_sets), seconds in states.items():
            if count >= max_exercises:
                continue
            transition = EXERCISE_TRANSITION_SECONDS if count and not grouped else 0
            for sets in range(1, max(0, role.sets) + 1):
                cost = seconds + transition + sets * work + (sets - 1) * rest
                if cost > available_sec:
                    break
                key = (count + 1, total_sets + sets)
                next_states[key] = min(next_states.get(key, cost), cost)
        states = next_states
    return max(total_sets for _, total_sets in states)

"""Versioned experience-aware resistance-training prescription matrix."""

from __future__ import annotations

from dataclasses import dataclass

from ..schemas import LLMExercisePlan
from .estimator import _movement_type_key
from .params import _goal_key


EXPERIENCE_PRESCRIPTION_VERSION = "routine-experience-prescription-v3-2026-07-22"


def normalize_experience_level(value: str | None) -> str:
    normalized = str(value or "intermediate").strip().lower()
    if normalized in {"beginner", "novice", "entry", "초보", "초보자", "입문"}:
        return "beginner"
    if normalized in {"advanced", "expert", "athlete", "상급", "상급자", "고급", "고급자"}:
        return "advanced"
    return "intermediate"


@dataclass(frozen=True)
class ExperiencePrescriptionDecision:
    target_reps: int
    sets: int
    rest_sec: int
    target_rir: int
    policy_version: str = EXPERIENCE_PRESCRIPTION_VERSION


def experience_set_cap(
    exercise: LLMExercisePlan,
    goal: str,
    experience_level: str | None,
    target_split_label: str | None,
) -> int:
    experience = normalize_experience_level(experience_level)
    movement = _movement_type_key(exercise.movement_type)
    if movement == "COMPOUND":
        if experience == "beginner":
            return 3
        if experience == "advanced":
            return 5 if _goal_key(goal) == "strength" else 4
        return 4
    if str(target_split_label or "").strip().lower() in {"arm", "core", "shoulder", "neck"}:
        return {"beginner": 3, "intermediate": 4, "advanced": 4}[experience]
    return {"beginner": 2, "intermediate": 3, "advanced": 3}[experience]


def resolve_experience_prescription(
    exercise: LLMExercisePlan,
    goal: str,
    experience_level: str | None,
    *,
    base_reps: int,
    base_rest_sec: int,
) -> ExperiencePrescriptionDecision:
    experience = normalize_experience_level(experience_level)
    goal_key = _goal_key(goal)
    movement = _movement_type_key(exercise.movement_type)
    reps = int(base_reps)
    rest = int(base_rest_sec)
    sets = max(1, int(exercise.sets or 1))

    if goal_key == "strength" and movement == "COMPOUND":
        reps = {"beginner": 8, "intermediate": 5, "advanced": 4}[experience]
        # Do not shorten heavy compound rest merely to make an advanced session denser.
        rest = max(rest, 180 if experience != "beginner" else 120)
    elif goal_key in {"hypertrophy", "recomposition"} and movement == "COMPOUND":
        reps = 10 if experience == "beginner" else 8
        rest = max(rest, 120)
    elif goal_key in {"hypertrophy", "recomposition"} and movement == "ISOLATION":
        reps = 12 if experience != "advanced" else 10
        rest = max(rest, 75)

    if movement == "COMPOUND":
        floor = {"beginner": 2, "intermediate": 3, "advanced": 3}[experience]
        cap = {"beginner": 3, "intermediate": 4, "advanced": 5 if goal_key == "strength" else 4}[experience]
    else:
        floor = {"beginner": 2, "intermediate": 2, "advanced": 3}[experience]
        cap = {"beginner": 2, "intermediate": 3, "advanced": 3}[experience]
    sets = min(cap, max(floor, sets))

    # ACSM's 2026 position stand identifies near-failure work around 2-3 RIR
    # as sufficient for healthy adults. Experience changes exercise selection,
    # reps, sets, and rest, but does not silently authorize routine failure work.
    target_rir = {"beginner": 3, "intermediate": 2, "advanced": 2}[experience]
    return ExperiencePrescriptionDecision(
        target_reps=reps,
        sets=sets,
        rest_sec=rest,
        target_rir=target_rir,
    )


def apply_experience_prescription(
    exercise: LLMExercisePlan,
    goal: str,
    experience_level: str | None,
    *,
    base_reps: int,
    base_rest_sec: int,
) -> LLMExercisePlan:
    decision = resolve_experience_prescription(
        exercise,
        goal,
        experience_level,
        base_reps=base_reps,
        base_rest_sec=base_rest_sec,
    )
    exercise.target_reps = decision.target_reps
    exercise.sets = decision.sets
    exercise.rest_time_sec = decision.rest_sec
    exercise.target_rir = decision.target_rir
    return exercise

"""Low readiness reduces role capacity; it does not disable planning entirely."""
import pytest

from engines.prescription.adjustments import _fill_available_time, _max_sets_for_exercise
from engines.prescription.estimator import estimate_routine_time_min
from engines.prescription.experience import experience_set_cap
from engines.prescription.targets import apply_deterministic_targets
from engines.schemas import LLMExercisePlan, LLMRoutineOutput


def routine(movement="COMPOUND", primary="quadriceps"):
    return LLMRoutineOutput(
        total_estimated_time=0, summary_title="Synthetic recovery check",
        rationale_summary=[], warnings=[], exercises=[LLMExercisePlan(
            exercise_id="test", exercise_name="Synthetic exercise",
            primary_muscles=[primary], equipment_type="MACHINE", movement_type=movement,
            sets=2, target_reps=8, rest_time_sec=120, target_rir=4,
            exercise_rationale="Synthetic candidate",
        )],
    )


@pytest.mark.parametrize("goal", ["strength", "hypertrophy"])
@pytest.mark.parametrize("experience", ["beginner", "intermediate", "advanced"])
def test_low_readiness_fills_only_the_reduced_compound_role_cap(goal, experience):
    output = routine()
    exercise = output.exercises[0]
    normal_cap = min(_max_sets_for_exercise(exercise, goal, "legs"),
                     experience_set_cap(exercise, goal, experience, "legs"))
    _fill_available_time(output, 60, goal, "low", "legs", experience_level=experience)
    assert exercise.sets == max(2, normal_cap - 1)
    assert exercise.target_rir == 4
    assert exercise.target_weight_kg is None


@pytest.mark.parametrize("restriction", ["primary", "direct_id", "secondary_id"])
def test_low_readiness_does_not_refill_a_doms_restricted_role(restriction):
    output = routine()
    kwargs = {"doms_db": {"quadriceps": 1}} if restriction == "primary" else {
        "doms_restricted_exercise_ids": {"test"},
    }
    _fill_available_time(output, 60, "hypertrophy", "low", "legs", **kwargs)
    assert output.exercises[0].sets == 2


@pytest.mark.parametrize("goal", ["strength", "hypertrophy"])
@pytest.mark.parametrize("available", [15, 30, 60, 90])
def test_recovery_fill_keeps_time_and_total_set_budget(goal, available):
    output = routine()
    _fill_available_time(output, available, goal, "low", "legs",
                         experience_level="advanced", max_total_sets=3)
    assert output.exercises[0].sets <= 3
    assert estimate_routine_time_min(output.exercises) <= available


@pytest.mark.parametrize("goal", ["strength", "hypertrophy"])
@pytest.mark.parametrize("experience", ["beginner", "intermediate", "advanced"])
def test_end_to_end_prescription_retains_low_effort_and_does_not_exceed_normal(goal, experience):
    common = dict(goal=goal, recent_sets=[], profile=None, time_available_min=60,
                  experience_level=experience, target_split_label="legs",
                  target_muscles=["quadriceps"])
    low = apply_deterministic_targets(routine(), readiness_level="low", **common)
    normal = apply_deterministic_targets(routine(), readiness_level="normal", **common)
    low_ex, normal_ex = low.exercises[0], normal.exercises[0]
    assert low_ex.sets == max(2, normal_ex.sets - 1)
    assert low_ex.target_rir >= 4
    assert low_ex.target_weight_kg is None
    assert low_ex.rest_time_sec == normal_ex.rest_time_sec
    assert [ex.exercise_id for ex in low.exercises] == ["test"]


def test_recovery_fill_does_not_increase_isolation_capacity_or_focus_budget():
    output = routine("ISOLATION", "calves")
    _fill_available_time(output, 60, "hypertrophy", "low", "legs",
                         experience_level="advanced", primary_focus_exercise_ids={"test"})
    assert output.exercises[0].sets == 2


def test_recovery_fill_cannot_restore_doms2_volume_removed_by_prescription():
    result = apply_deterministic_targets(
        routine(), "strength", [], None, readiness_level="low",
        target_split_label="legs", target_muscles=["quadriceps"],
        doms_level_by_exercise_id={"test": 2}, time_available_min=60,
        experience_level="advanced",
    )
    assert result.exercises[0].sets <= 2
    assert result.exercises[0].target_rir >= 4


@pytest.mark.parametrize("set_type,duration,distance", [
    ("timed", 30, None), ("mobility", None, None),
    ("distance", None, 30), ("working", 30, None),
])
def test_low_readiness_fill_does_not_expand_non_rep_work(set_type, duration, distance):
    output = routine()
    exercise = output.exercises[0]
    exercise.sets = 1
    exercise._prescription_set_type = set_type
    if duration:
        exercise._target_duration_sec = duration
    if distance:
        exercise._target_distance_m = distance
    before = exercise.model_copy(deep=True)
    _fill_available_time(output, 60, "hypertrophy", "low", "legs",
                         experience_level="advanced")
    assert exercise.model_dump() == before.model_dump()
    assert getattr(exercise, "_target_duration_sec", None) == duration
    assert getattr(exercise, "_target_distance_m", None) == distance

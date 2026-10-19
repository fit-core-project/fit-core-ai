import pytest

from engines.prescription.adjustments import _apply_doms_volume_policy
from engines.prescription.targets import apply_deterministic_targets
from engines.prescription.weight import _resolve_target_weight
from engines.schemas import LLMExercisePlan, LLMRoutineOutput, RecentSetRecord, UserProfileContext
from engines.routine_blueprint import build_routine_blueprint


def _plan(**changes):
    values = dict(exercise_id="press", exercise_name="벤치 프레스", primary_muscles=["chest"],
                  equipment_type="BARBELL", movement_type="COMPOUND", target_reps=5,
                  sets=5, rest_time_sec=180, target_rir=2, exercise_rationale="가슴 운동")
    return LLMExercisePlan(**(values | changes))


def _history():
    return [RecentSetRecord(exercise_id="press", exercise_name="벤치 프레스",
                            primary_muscle="chest", weight_kg=100, reps=5, rir=3)]


@pytest.mark.parametrize("rir", [3, 4, 5])
@pytest.mark.parametrize("reference", ["exact", "same_muscle", "baseline", "hint"])
def test_recovery_effort_does_not_invent_calibrated_kilograms(rir, reference):
    exercise = _plan(target_rir=rir, target_weight_kg=100)
    history = _history() if reference in {"exact", "same_muscle"} else []
    if reference == "same_muscle":
        history[0].exercise_id = "different_press"
        history[0].exercise_name = "다른 프레스"
    profile = UserProfileContext(goal_type="strength", split_type="push", strength_baseline={
        "press": {"weight_kg": 100, "reps": 5}
    }) if reference == "baseline" else None
    assert _resolve_target_weight(exercise, "strength", history, profile) is None
    assert "3~5" not in exercise.exercise_rationale
    assert f"{rir}회" in exercise.exercise_rationale
    assert "중량" in exercise.exercise_rationale


def test_effort_guidance_is_idempotent_and_not_invented_for_bodyweight():
    exercise = _plan(target_rir=4)
    _resolve_target_weight(exercise, "strength", _history(), None)
    first = exercise.exercise_rationale
    _resolve_target_weight(exercise, "strength", _history(), None)
    assert exercise.exercise_rationale == first
    bodyweight = _plan(equipment_type="BODYWEIGHT", target_rir=4)
    assert _resolve_target_weight(bodyweight, "strength", _history(), None) is None
    assert bodyweight.exercise_rationale == "가슴 운동"


def test_normal_effort_retains_existing_weight_policy():
    # Historical ID retained; v2 replaces the former unconditional +2.5 kg oracle.
    assert _resolve_target_weight(_plan(), "strength", _history(), None) is None


@pytest.mark.parametrize("doms", [1, 2])
@pytest.mark.parametrize("sets", [1, 2, 3])
def test_doms_never_restores_sets_removed_by_experience_or_readiness(doms, sets):
    exercise = _plan(sets=sets)
    _apply_doms_volume_policy(exercise, "strength", doms)
    assert exercise.sets <= sets
    assert exercise.target_rir >= (3 if doms == 1 else 4)


@pytest.mark.parametrize("readiness,doms", [("low", 0), ("normal", 1), ("normal", 2)])
def test_pipeline_dose_order_does_not_reintroduce_progression_after_effort_reduction(readiness, doms):
    routine = LLMRoutineOutput(total_estimated_time=30, summary_title="운동", rationale_summary=["목표"],
                               warnings=[], exercises=[_plan()])
    result = apply_deterministic_targets(
        routine, "strength", _history(), None, readiness_level=readiness,
        target_split_label="push", target_muscles=["chest"],
        doms_level_by_exercise_id={"press": doms}, experience_level="advanced",
    )
    assert result.exercises[0].target_weight_kg is None
    assert result.exercises[0].target_rir >= 3
    assert routine.exercises[0].exercise_rationale == "가슴 운동"


@pytest.mark.parametrize("experience,goal,expected", [
    ("beginner", "strength", 2), ("intermediate", "strength", 3),
    ("advanced", "strength", 3), ("advanced", "hypertrophy", 3),
])
def test_doms2_uses_experience_role_capacity_not_global_two_set_cap(experience, goal, expected):
    output = LLMRoutineOutput(total_estimated_time=30, summary_title="test",
                             rationale_summary=[], exercises=[_plan()])
    revised = apply_deterministic_targets(
        output, goal, [], None, experience_level=experience,
        doms_level_by_exercise_id={"press": 2}, time_available_min=60,
        target_split_label="chest", target_muscles=["chest"],
    )
    assert revised.exercises[0].sets == expected
    assert revised.exercises[0].target_rir >= 4
    assert revised.exercises[0].target_weight_kg is None


@pytest.mark.parametrize("split", ["push", "pull", "legs", "full_body"])
@pytest.mark.parametrize("minutes", [30, 45, 60, 90])
def test_low_readiness_does_not_double_reduce_role_coverage(split, minutes):
    kwargs = dict(target_split_label=split, target_muscles=[], goal="hypertrophy",
                  experience_level="intermediate", time_available_min=minutes)
    normal = build_routine_blueprint(readiness_level="normal", **kwargs)
    low = build_routine_blueprint(readiness_level="low", **kwargs)
    assert low.minimum_exercise_count == normal.minimum_exercise_count


@pytest.mark.parametrize("set_type", ["timed", "distance", "mobility"])
def test_doms_repetition_set_policy_leaves_nonworking_prescriptions_unchanged(set_type):
    exercise = _plan()
    exercise._prescription_set_type = set_type
    before = exercise.model_dump()
    _apply_doms_volume_policy(exercise, "strength", 2)
    assert exercise.model_dump() == before

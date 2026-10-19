"""Focused boundary evidence for the remaining role/volume short plans.

These checks intentionally describe current admitted-candidate and set-cap
policy. They are not an approval to add candidates or raise volume limits.
"""

from engines.prescription.adjustments import _fill_available_time
from engines.prescription.estimator import estimate_routine_time_min
from engines.routine_blueprint import build_routine_blueprint
from engines.schemas import LLMExercisePlan, LLMRoutineOutput


def _plan(exercise_id: str, movement_type: str, sets: int, primary: str = "chest") -> LLMExercisePlan:
    return LLMExercisePlan(
        exercise_id=exercise_id,
        exercise_name=exercise_id,
        movement_type=movement_type,
        primary_muscles=[primary],
        target_reps=8 if movement_type == "COMPOUND" else 10,
        sets=sets,
        rest_time_sec=120 if movement_type == "COMPOUND" else 75,
        target_rir=2,
        exercise_rationale="test",
    )


def test_three_selected_lower_roles_fill_available_capacity_but_not_beyond_caps():
    output = LLMRoutineOutput(
        total_estimated_time=0, summary_title="test", rationale_summary=[],
        exercises=[
            _plan("synthetic-gluteal-main", "COMPOUND", 3, "gluteal"),
            _plan("synthetic-hamstring-accessory", "ISOLATION", 2, "hamstring"),
            _plan("synthetic-quadriceps-accessory", "ISOLATION", 2, "quadriceps"),
        ],
    )
    diagnostics = _fill_available_time(
        output, 50, "hypertrophy", "normal", "legs",
        experience_level="intermediate", max_total_sets=20,
    )
    assert [plan.sets for plan in output.exercises] == [4, 3, 3]
    assert estimate_routine_time_min(output.exercises) == 28
    assert diagnostics["role_set_capacity_exhausted"] is True
    assert diagnostics["role_set_capacity_total_budget_exhausted"] is False
    _fill_available_time(
        output, 50, "hypertrophy", "normal", "legs",
        experience_level="intermediate", max_total_sets=20,
    )
    assert [plan.sets for plan in output.exercises] == [4, 3, 3]


def test_three_selected_chest_roles_reach_the_existing_caps():
    blueprint = build_routine_blueprint(
        target_split_label="chest",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="advanced",
        readiness_level="normal",
        time_available_min=60,
        training_style="gym",
    )
    output = LLMRoutineOutput(
        total_estimated_time=0,
        summary_title="test",
        rationale_summary=[],
        exercises=[
            _plan("synthetic-chest-main", "COMPOUND", 3),
            _plan("synthetic-chest-secondary", "COMPOUND", 3),
            _plan("synthetic-chest-isolation", "ISOLATION", 2),
        ],
    )

    diagnostics = _fill_available_time(
        output,
        60,
        "hypertrophy",
        "normal",
        "chest",
        experience_level="advanced",
        max_total_sets=blueprint.max_total_sets,
    )

    assert blueprint.target_exercise_count == 3
    assert [slot.slot_id for slot in blueprint.slots] == [
        "chest_main", "chest_secondary", "chest_isolation",
    ]
    assert [plan.sets for plan in output.exercises] == [4, 4, 3]
    assert estimate_routine_time_min(output.exercises) == 34
    assert diagnostics["role_set_capacity_exhausted"] is True
    assert diagnostics["role_set_capacity_total_budget_exhausted"] is False

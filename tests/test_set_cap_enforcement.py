from __future__ import annotations

import json
from pathlib import Path

from engines.feedback_aggregation import FeedbackAdjustment
from engines.prescription.adjustments import enforce_total_set_cap
from engines.prescription.targets import apply_deterministic_targets
from engines.routine_telemetry import classify_hard_violation_category
from engines.schemas import LLMExercisePlan, LLMRoutineOutput
from tests.evaluation.helpers import EvalHarness


def _plan(
    exercise_id: str,
    *,
    sets: int,
    movement_type: str = "COMPOUND",
    primary: str = "quadriceps",
) -> LLMExercisePlan:
    return LLMExercisePlan(
        exercise_id=exercise_id,
        exercise_name=exercise_id.replace("_", " ").title(),
        movement_type=movement_type,
        primary_muscles=[primary],
        equipment_type="BARBELL",
        target_reps=5,
        sets=sets,
        rest_time_sec=180,
        target_rir=2,
        exercise_rationale="test rationale",
    )


def _output(exercises: list[LLMExercisePlan]) -> LLMRoutineOutput:
    return LLMRoutineOutput(
        total_estimated_time=60,
        summary_title="Strength lower",
        rationale_summary=["test"],
        warnings=[],
        exercises=exercises,
    )


def _total_sets(output: LLMRoutineOutput) -> int:
    return sum(exercise.sets for exercise in output.exercises)


def _strength_lower_scenario() -> dict:
    scenarios = json.loads(
        (Path(__file__).parent / "evaluation" / "fixtures" / "scenarios.json").read_text(encoding="utf-8")
    )
    return next(scenario for scenario in scenarios if scenario["name"] == "strength_lower_barbell")


def test_total_sets_within_cap_is_unchanged():
    output = _output([_plan("back_squat", sets=4), _plan("romanian_deadlift", sets=4, primary="hamstring")])
    before = output.model_dump()

    changed = enforce_total_set_cap(output, 8, ["quadriceps", "hamstring"])

    assert changed is False
    assert output.model_dump() == before


def test_total_sets_over_cap_are_trimmed_to_cap():
    output = _output(
        [
            _plan("back_squat", sets=5),
            _plan("romanian_deadlift", sets=5, primary="hamstring"),
            _plan("leg_curl", sets=4, movement_type="ISOLATION", primary="hamstring"),
        ]
    )

    changed = enforce_total_set_cap(output, 10, ["quadriceps", "hamstring"])

    assert changed is True
    assert _total_sets(output) <= 10
    assert "set cap enforced" in output.warnings


def test_set_cap_trimming_keeps_at_least_one_block():
    output = _output([_plan("back_squat", sets=3), _plan("leg_curl", sets=3, movement_type="ISOLATION")])

    enforce_total_set_cap(output, 1, ["quadriceps"])

    assert len(output.exercises) == 1
    assert _total_sets(output) == 1


def test_compound_target_lower_movement_is_preserved_before_accessory():
    output = _output(
        [
            _plan("back_squat", sets=5, movement_type="COMPOUND", primary="quadriceps"),
            _plan("romanian_deadlift", sets=5, movement_type="COMPOUND", primary="hamstring"),
            _plan("leg_curl", sets=5, movement_type="ISOLATION", primary="hamstring"),
        ]
    )

    enforce_total_set_cap(output, 9, ["quadriceps", "hamstring"])

    by_id = {exercise.exercise_id: exercise for exercise in output.exercises}
    assert by_id["leg_curl"].sets == 1
    assert by_id["back_squat"].sets >= by_id["leg_curl"].sets
    assert _total_sets(output) <= 9


def test_deterministic_fill_cannot_reintroduce_set_cap_overflow():
    output = _output(
        [
            _plan("back_squat", sets=3, movement_type="COMPOUND", primary="quadriceps"),
            _plan("romanian_deadlift", sets=3, movement_type="COMPOUND", primary="hamstring"),
            _plan("leg_curl", sets=2, movement_type="ISOLATION", primary="hamstring"),
        ]
    )

    adjusted = apply_deterministic_targets(
        output,
        goal="strength",
        recent_sets=[],
        profile=None,
        readiness_level="normal",
        target_split_label="legs",
        target_muscles=["quadriceps", "hamstring"],
        time_available_min=60,
        max_total_sets=8,
    )

    assert _total_sets(adjusted) <= 8


def test_strength_lower_fixture_has_no_set_cap_violation():
    result = EvalHarness().run(_strength_lower_scenario())

    assert sum(len(block["prescription"]) for block in result.routine_blocks) <= 15
    assert not any(classify_hard_violation_category(item) == "set_cap" for item in result.hard_violation_details)


def test_strength_lower_fb_fixture_has_no_set_cap_violation():
    adjustments = {
        "back_squat": FeedbackAdjustment(exercise_id="back_squat", blended_adjustment=2.0),
        "leg_curl": FeedbackAdjustment(exercise_id="leg_curl", blended_adjustment=-1.0),
    }

    result = EvalHarness().run(_strength_lower_scenario(), feedback_adjustments=adjustments)

    assert sum(len(block["prescription"]) for block in result.routine_blocks) <= 15
    assert not any(classify_hard_violation_category(item) == "set_cap" for item in result.hard_violation_details)


def test_feedback_ranking_off_on_apply_same_set_cap_enforcement():
    scenario = _strength_lower_scenario()
    off = EvalHarness().run(scenario)
    on = EvalHarness().run(
        scenario,
        feedback_adjustments={"leg_curl": FeedbackAdjustment(exercise_id="leg_curl", blended_adjustment=3.0)},
    )

    assert sum(len(block["prescription"]) for block in off.routine_blocks) <= 15
    assert sum(len(block["prescription"]) for block in on.routine_blocks) <= 15


def test_set_cap_category_not_emitted_after_enforcement():
    result = EvalHarness().run(_strength_lower_scenario())
    categories = [classify_hard_violation_category(item) for item in result.hard_violation_details]

    assert "set_cap" not in categories


def test_set_cap_enforcement_does_not_store_raw_prompt_output_or_user_note():
    output = _output([_plan("back_squat", sets=6), _plan("leg_curl", sets=6, movement_type="ISOLATION")])

    enforce_total_set_cap(output, 4, ["quadriceps"])
    text = json.dumps(output.model_dump(), ensure_ascii=False)

    assert "raw prompt" not in text.lower()
    assert "raw output" not in text.lower()
    assert "user_note" not in text

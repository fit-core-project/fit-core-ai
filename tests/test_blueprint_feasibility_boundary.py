import pytest

from engines import routine_pipeline as pipeline
from engines.routine_blueprint import RoutineBlueprintValidationError, build_routine_blueprint
from engines.schemas import RoutineDraftResponse, RoutineRequest


def request(**updates):
    values = dict(user_id="synthetic-time-boundary", time_available_min=10,
                  target_split_label="chest", target_muscles=["chest"], goal="hypertrophy",
                  experience_level="advanced", method_mode="off")
    values.update(updates)
    return RoutineRequest(**values)


def invoke(req):
    return pipeline.generate_smart_routine(
        req, db=None, local_constraint_profile={}, local_constraint_profile_is_resolved=True)


def no_candidate_or_model(*args, **kwargs):
    raise AssertionError("infeasible request must stop before database/model use")


@pytest.mark.parametrize("experience", ["beginner", "intermediate", "advanced"])
@pytest.mark.parametrize("goal", ["strength", "hypertrophy"])
def test_valid_minimum_time_request_returns_bounded_failure(monkeypatch, experience, goal):
    monkeypatch.setattr(pipeline, "get_candidate_exercises", no_candidate_or_model)
    result = invoke(request(experience_level=experience, goal=goal))
    assert result.generation_status == "failed"
    assert result.status_reason_code == "schemaError"
    assert result.is_fallback is False
    assert result.total_estimated_time == 0
    assert result.routine_blocks == []
    assert "10분" in result.rationale_summary[0]
    assert "늘리거나" in result.warnings[0]
    RoutineDraftResponse.model_validate(result.model_dump(by_alias=True))


def test_blueprint_error_keeps_message_and_value_error_compatibility():
    with pytest.raises(ValueError) as caught:
        build_routine_blueprint(
            target_split_label="chest", target_muscles=["chest"], goal="hypertrophy",
            experience_level="advanced", readiness_level="normal", time_available_min=10)
    assert isinstance(caught.value, RoutineBlueprintValidationError)
    assert caught.value.errors == ("set_budget_below_required_slots",)
    assert str(caught.value) == "invalid routine blueprint: set_budget_below_required_slots"


@pytest.mark.parametrize("error", [
    ValueError("unrelated failure"),
    RoutineBlueprintValidationError(["duplicate_slot_id"]),
    RoutineBlueprintValidationError(["set_budget_below_required_slots", "empty_slot_target"]),
])
def test_unrelated_errors_are_not_disguised_as_time_constraints(monkeypatch, error):
    def fail(**kwargs):
        raise error
    monkeypatch.setattr(pipeline, "build_routine_blueprint", fail)
    with pytest.raises(type(error)) as caught:
        invoke(request())
    assert caught.value is error


def test_feasible_request_continues_to_candidates(monkeypatch):
    reached = RuntimeError("candidate stage reached")
    def stop(*args, **kwargs):
        raise reached
    monkeypatch.setattr(pipeline, "get_candidate_exercises", stop)
    with pytest.raises(RuntimeError) as caught:
        invoke(request(time_available_min=60))
    assert caught.value is reached

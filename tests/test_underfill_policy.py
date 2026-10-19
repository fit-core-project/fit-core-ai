import pytest

from engines.prescription.underfill import (
    planned_volume_complete_for_request,
    resolve_underfill_decision,
)
from engines.routine_blueprint import build_routine_blueprint
from engines.routine_pipeline import (
    _apply_underfill_explanation,
    has_target_relevant_recovery_constraint,
)
from engines.rule_critic import RoutineEvalContext, _exercise_count_score, _time_score
from engines.schemas import (
    InjuryConditionEntry,
    PainAreaEntry,
    RoutineBlock,
    RoutineDraftResponse,
    RoutineRequest,
    SetPrescription,
)


@pytest.mark.parametrize(
    ("area", "target_muscles", "expected"),
    [
        ("ankle", ["quadriceps"], True),
        ("hip", ["hamstring"], True),
        ("elbow", ["upper-back"], True),
        ("neck", ["chest"], True),
        ("ankle", ["chest"], False),
        ("elbow", ["quadriceps"], False),
    ],
)
def test_managed_injury_underfill_scope_is_target_relevant(
    area,
    target_muscles,
    expected,
):
    request = RoutineRequest(
        target_split_label="custom",
        target_muscles=target_muscles,
        time_available_min=60,
        injury_conditions=[
            InjuryConditionEntry(
                area_code=area,
                status="managedWithRestrictions",
                restricted_movement_patterns=["unknown"],
            )
        ],
    )

    assert has_target_relevant_recovery_constraint(
        request,
        target_muscles,
    ) is expected


@pytest.mark.parametrize("status", ["managedNoRestrictions", "recovered"])
def test_nonrestrictive_managed_injury_does_not_relax_underfill(status):
    request = RoutineRequest(
        target_split_label="legs",
        time_available_min=60,
        injury_conditions=[
            InjuryConditionEntry(area_code="ankle", status=status)
        ],
    )

    assert has_target_relevant_recovery_constraint(
        request,
        ["quadriceps", "hamstring"],
    ) is False


@pytest.mark.parametrize(
    ("available", "actual"),
    [(40, 26), (50, 33), (60, 39)],
)
def test_productive_floor_is_not_reported_as_underfill(available, actual):
    decision = resolve_underfill_decision(
        actual_duration_min=actual,
        available_duration_min=available,
        has_routine_blocks=True,
        candidate_count=10,
        minimum_exercise_count=4,
    )

    assert decision.status == "within_floor"
    assert decision.reason_code is None
    assert decision.user_summary_ko is None


@pytest.mark.parametrize("available", [40, 50, 60])
def test_below_half_time_always_has_candidate_exhaustion_explanation(available):
    decision = resolve_underfill_decision(
        actual_duration_min=(available // 2) - 1,
        available_duration_min=available,
        has_routine_blocks=True,
        candidate_count=12,
        minimum_exercise_count=4,
        selection_diagnostics={"productive_fill_exhausted": True},
    )

    assert decision.underfilled is True
    assert decision.severe_underfill is True
    assert decision.reason_code == "candidate_policy_exhausted"
    assert decision.user_summary_ko
    assert decision.user_option_ko


def test_completed_bounded_plan_is_not_padded_to_fill_available_clock():
    decision = resolve_underfill_decision(
        actual_duration_min=30,
        available_duration_min=60,
        has_routine_blocks=True,
        planned_volume_complete=True,
    )

    assert decision.status == "within_plan"
    assert decision.underfilled is False
    assert decision.planned_volume_complete is True


def test_planned_volume_complete_accepts_filled_small_muscle_only_plan():
    assert planned_volume_complete_for_request(
        goal="hypertrophy",
        experience_level="intermediate",
        target_muscles=("front-deltoids", "back-deltoids", "triceps"),
        exercise_count=6,
        target_exercise_count=6,
    ) is True


def test_planned_volume_complete_does_not_relax_incomplete_or_large_muscle_plan():
    assert planned_volume_complete_for_request(
        goal="hypertrophy",
        experience_level="intermediate",
        target_muscles=("front-deltoids", "triceps"),
        exercise_count=4,
        target_exercise_count=6,
    ) is False
    assert planned_volume_complete_for_request(
        goal="hypertrophy",
        experience_level="intermediate",
        target_muscles=("chest", "triceps"),
        exercise_count=6,
        target_exercise_count=6,
    ) is False


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"injury_limited": True}, "registered_injury_constraint"),
        ({"severe_doms_limited": True}, "severe_doms_constraint"),
        ({"recovery_limited": True}, "recovery_constraint"),
        ({"readiness_level": "low"}, "low_readiness"),
        ({"condition_policy_limited": True}, "condition_policy_constraint"),
        ({"equipment_constrained": True}, "equipment_shortage"),
        ({"fallback_used": True, "candidate_count": 10}, "fallback_baseline"),
    ],
)
def test_underfill_reason_policy_is_deterministic(kwargs, expected):
    candidate_count = kwargs.get("candidate_count", 2)
    policy_kwargs = {key: value for key, value in kwargs.items() if key != "candidate_count"}
    decision = resolve_underfill_decision(
        actual_duration_min=20,
        available_duration_min=60,
        has_routine_blocks=True,
        candidate_count=candidate_count,
        minimum_exercise_count=4,
        **policy_kwargs,
    )

    assert decision.reason_code == expected


def _routine_block() -> RoutineBlock:
    return RoutineBlock(
        order=1,
        exercise_id="leg-press",
        exercise_name="레그프레스",
        movement_type="COMPOUND",
        primary_muscles=["quadriceps"],
        equipment_type="MACHINE",
        default_rest_sec=90,
        prescription=[
            SetPrescription(
                set_index=1,
                target_reps=10,
                target_rest_sec=90,
            )
        ],
        exercise_rationale="하체 주동근 운동입니다.",
        session_phase="main_compound",
    )


def _short_response() -> RoutineDraftResponse:
    return RoutineDraftResponse(
        generation_status="success",
        status_reason_code="none",
        is_fallback=False,
        total_estimated_time=18,
        summary_title="하체 루틴",
        rationale_summary=["현재 조건을 반영했습니다."],
        routine_blocks=[_routine_block()],
        warnings=[],
    )


def test_pipeline_explains_injury_underfill_without_changing_public_contract():
    request = RoutineRequest(
        target_split_label="legs",
        target_muscles=["quadriceps"],
        time_available_min=60,
        pain_areas=[PainAreaEntry(body_part="quadriceps")],
    )
    blueprint = build_routine_blueprint(
        target_split_label="legs",
        target_muscles=["quadriceps"],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=60,
        pain_areas=request.pain_areas,
        doms_data={},
    )
    diagnostics = {}

    response = _apply_underfill_explanation(
        _short_response(),
        request,
        ranked_candidates=[],
        routine_blueprint=blueprint,
        selection_diagnostics=diagnostics,
    )

    assert response._underfill_diagnostics["reason_code"] == "registered_injury_constraint"
    assert diagnostics["underfill_reason_code"] == "registered_injury_constraint"
    assert any("부상 부위" in item for item in response.rationale_summary)
    assert any("목표 부위" in item for item in response.warnings)
    serialized = response.model_dump(by_alias=True)
    assert "underfillDiagnostics" not in serialized
    assert "statusReasonCode" in serialized


def test_rule_critic_allows_explained_underfill_but_not_unexplained_underfill():
    response = _short_response()
    context = RoutineEvalContext(
        goal="hypertrophy",
        target_muscles=["quadriceps"],
        duration_min=60,
        readiness_level="normal",
        available_equipment=[],
        unavailable_equipment=[],
        current_pain_areas=[],
        doms={},
        recent_exercises=[],
    )

    score, warnings = _time_score(response, context, False, 12)
    assert score == 0
    assert any(item.startswith("FAIL time") for item in warnings)

    response._underfill_diagnostics = {"reason_code": "candidate_policy_exhausted"}
    score, warnings = _time_score(response, context, False, 12)
    assert score == 10
    assert any("allowed underfill" in item for item in warnings)

    score, warnings = _exercise_count_score(
        [block.model_dump(by_alias=False) for block in response.routine_blocks],
        context,
        False,
        12,
        underfill_reason="candidate_policy_exhausted",
    )
    assert score == 5
    assert any("exercise count underfilled" in item for item in warnings)

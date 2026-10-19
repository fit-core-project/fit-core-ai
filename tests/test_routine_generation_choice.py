import inspect
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from engines.routine_generation_choice import (
    CHOICE_POLICY_VERSION,
    CHOICE_SCHEMA_VERSION,
    RoutineGenerationChoiceError,
    generate_routine_with_choice,
    make_routine_generation_selection,
    preflight_routine_generation,
)
from engines.schemas import (
    InjuryConditionEntry,
    RoutineBlock,
    RoutineDraftResponse,
    RoutineRequest,
    SetPrescription,
    TrainingMethodInstruction,
    UserProfileContext,
)


def _request(**updates) -> RoutineRequest:
    values = {
        "target_split_label": "push",
        "time_available_min": 45,
        "pain_areas": [{"bodyPart": "chest"}],
    }
    values.update(updates)
    return RoutineRequest.model_validate(values)


def _profile(**updates) -> UserProfileContext:
    values = {
        "goal_type": "hypertrophy",
        "split_type": "PPL",
        "experience_level": "intermediate",
    }
    values.update(updates)
    return UserProfileContext.model_validate(values)


def _set(index: int, *, set_type: str = "working", **updates) -> SetPrescription:
    values = {
        "set_index": index,
        "set_type": set_type,
        "target_reps": 8,
        "target_weight_kg": 40,
        "target_rir": 2,
        "target_rest_sec": 90,
    }
    values.update(updates)
    return SetPrescription(**values)


def _generated_routine() -> RoutineDraftResponse:
    return RoutineDraftResponse(
        routine_draft_id="choice-test-draft",
        generation_status="success",
        status_reason_code="none",
        is_fallback=False,
        total_estimated_time=30,
        summary_title="Bound routine",
        rationale_summary=["총 7세트, 예상 30분 구성"],
        warnings=[],
        routine_blocks=[
            RoutineBlock(
                order=1,
                exercise_id="press",
                exercise_name="Press",
                movement_pattern="horizontalPush",
                movement_type="COMPOUND",
                primary_muscles=["chest"],
                equipment_type="DUMBBELL",
                default_rest_sec=90,
                prescription=[_set(index) for index in range(1, 5)],
                exercise_rationale="4세트 메인 운동",
                execution_group_id="superset-1",
                execution_group_type="superset",
                execution_group_position=1,
                execution_group_size=2,
                transition_rest_sec=15,
                group_rest_sec=75,
                training_method=TrainingMethodInstruction(
                    code="pausedReps",
                    label_ko="Pause",
                    purpose_ko="Purpose",
                    instruction_ko="Pause one second",
                    pause_sec=1,
                ),
            ),
            RoutineBlock(
                order=2,
                exercise_id="pushdown",
                exercise_name="Pushdown",
                movement_pattern="elbowExtension",
                movement_type="ISOLATION",
                primary_muscles=["triceps"],
                equipment_type="CABLE",
                default_rest_sec=75,
                prescription=[
                    _set(index, target_reps=10, target_rest_sec=75)
                    for index in range(1, 4)
                ],
                exercise_rationale="3 sets accessory",
                execution_group_id="superset-1",
                execution_group_type="superset",
                execution_group_position=2,
                execution_group_size=2,
                transition_rest_sec=15,
                group_rest_sec=75,
            ),
            RoutineBlock(
                order=3,
                exercise_id="cooldown",
                exercise_name="Cooldown",
                movement_pattern="mobility",
                movement_type="MOBILITY",
                primary_muscles=[],
                equipment_type="BODYWEIGHT",
                default_rest_sec=15,
                prescription=[
                    _set(
                        index,
                        set_type="timed",
                        target_reps=1,
                        target_weight_kg=None,
                        target_rir=None,
                        target_rest_sec=15,
                        target_duration_sec=30,
                    )
                    for index in range(1, 3)
                ],
                exercise_rationale="Timed cooldown",
                session_phase="cooldown",
                adjunct_option="cooldown_stretch",
            ),
            RoutineBlock(
                order=4,
                exercise_id="conditioning",
                exercise_name="Conditioning",
                movement_pattern="conditioning",
                movement_type="CONDITIONING",
                primary_muscles=[],
                equipment_type="BODYWEIGHT",
                default_rest_sec=0,
                prescription=[
                    _set(
                        1,
                        set_type="interval",
                        target_reps=1,
                        target_weight_kg=None,
                        target_rir=None,
                        target_rest_sec=0,
                        target_duration_sec=480,
                        interval_work_sec=30,
                        interval_recovery_sec=60,
                        interval_rounds=6,
                    )
                ],
                exercise_rationale="Interval conditioning",
                session_phase="finisher",
                adjunct_option="conditioning_finisher",
            ),
        ],
    )


def test_unresolved_warning_requires_choice_without_generation():
    calls = []

    result = generate_routine_with_choice(
        _request(),
        db=object(),
        _generator=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    assert result.status == "choiceRequired"
    assert result.choice_required is True
    assert result.available_choices == [
        "reducedVolume",
        "normalVolume",
        "requestReevaluation",
        "cancel",
    ]
    assert result.metrics.generation_invocation_count == 0
    assert calls == []


def test_selection_is_bound_to_request_profile_history_and_policy():
    request = _request()
    profile = _profile()
    preflight = preflight_routine_generation(request, profile=profile)
    selection = make_routine_generation_selection(preflight, "normalVolume")
    calls = []

    changed_request = request.model_copy(update={"time_available_min": 50})
    with pytest.raises(RoutineGenerationChoiceError, match="does not match"):
        generate_routine_with_choice(
            changed_request,
            db=object(),
            profile=profile,
            selection=selection,
            _generator=lambda *args, **kwargs: calls.append((args, kwargs)),
        )

    changed_injury = request.model_copy(update={
        "pain_areas": [],
        "injury_conditions": [
            InjuryConditionEntry(area_code="shoulder", status="managedNoRestrictions")
        ],
    })
    with pytest.raises(RoutineGenerationChoiceError, match="does not match"):
        generate_routine_with_choice(
            changed_injury,
            db=object(),
            profile=profile,
            selection=selection,
            _generator=lambda *args, **kwargs: calls.append((args, kwargs)),
        )

    assert selection.policy_version == CHOICE_POLICY_VERSION
    assert selection.schema_version == CHOICE_SCHEMA_VERSION
    assert calls == []


@pytest.mark.parametrize(
    "selection_update",
    [
        {"choice": "unrestrictedVolume"},
        {"schemaVersion": "routine-generation-choice-v0"},
        {"unknownField": True},
    ],
)
def test_unknown_choice_schema_and_fields_are_rejected(selection_update):
    request = _request()
    preflight = preflight_routine_generation(request)
    selection = {
        "schemaVersion": CHOICE_SCHEMA_VERSION,
        "policyVersion": CHOICE_POLICY_VERSION,
        "requestFingerprint": preflight.request_fingerprint,
        "choice": "normalVolume",
        **selection_update,
    }

    with pytest.raises(ValidationError):
        generate_routine_with_choice(
            request,
            db=object(),
            selection=selection,
            _generator=lambda *args, **kwargs: pytest.fail("generation invoked"),
        )


@pytest.mark.parametrize(
    "invalid_update",
    [
        {"choice": "unrestrictedVolume"},
        {"policy_version": "injury-volume-choice-forged"},
    ],
)
def test_preinstantiated_selection_is_revalidated(invalid_update):
    request = _request()
    preflight = preflight_routine_generation(request)
    valid = make_routine_generation_selection(preflight, "normalVolume")
    forged = valid.model_copy(update=invalid_update)

    with pytest.raises(ValidationError):
        generate_routine_with_choice(
            request,
            db=object(),
            selection=forged,
            _generator=lambda *args, **kwargs: pytest.fail("generation invoked"),
        )


@pytest.mark.parametrize(
    ("choice", "expected_status"),
    [("cancel", "canceled"), ("requestReevaluation", "reevaluationRequired")],
)
def test_cancel_and_reevaluation_do_not_generate_or_rewrite_injury(choice, expected_status):
    request = _request(
        pain_areas=[],
        injury_conditions=[{
            "areaCode": "shoulder",
            "status": "managedWithRestrictions",
            "restrictedMovementPatterns": ["horizontalPush"],
        }],
    )
    before = request.model_dump(mode="json")
    preflight = preflight_routine_generation(request)
    selection = make_routine_generation_selection(preflight, choice)

    result = generate_routine_with_choice(
        request,
        db=object(),
        selection=selection,
        _generator=lambda *args, **kwargs: pytest.fail("generation invoked"),
    )

    assert result.status == expected_status
    assert result.metrics.generation_invocation_count == 0
    assert request.model_dump(mode="json") == before
    if choice == "requestReevaluation":
        assert "동의만으로 제외 동작을 허용하지 않습니다" in result.reevaluation_message


def test_managed_conditions_merge_monotonically_in_preflight_summary():
    request = _request(
        pain_areas=[],
        injury_conditions=[{
            "areaCode": "knee",
            "status": "managedNoRestrictions",
        }],
    )
    profile = _profile(injury_conditions=[{
        "areaCode": "knee",
        "status": "managedWithRestrictions",
        "restrictedMovementPatterns": ["deepKneeFlexion"],
    }])

    result = preflight_routine_generation(request, profile=profile)

    assert result.status == "choiceRequired"
    assert result.constraints.managed_restriction_count == 1
    assert result.constraints.explicit_prohibited_pattern_count == 1
    assert result.constraints.injury_conditions[0].status == "managedWithRestrictions"
    assert result.constraints.injury_conditions[0].restricted_movement_patterns == [
        "deepKneeFlexion"
    ]


def test_condition_policy_alone_requires_choice_without_generation():
    condition_policy = {
        "conditionCode": "synthetic_post_procedure",
        "clearanceStatus": "cleared_with_restrictions",
        "movementRestrictions": ["synthetic_no_loaded_flexion"],
    }
    request = _request(
        pain_areas=[],
        condition_policies=[condition_policy],
    )

    result = generate_routine_with_choice(
        request,
        db=object(),
        _generator=lambda *args, **kwargs: pytest.fail("generation invoked"),
    )

    assert result.status == "choiceRequired"
    assert result.metrics.generation_invocation_count == 0
    assert result.constraints.condition_policy_count == 1
    assert result.constraints.condition_policies == [condition_policy]


def test_normal_volume_adds_no_optional_reduction_and_preserves_inputs():
    request = _request(readiness_level="low", doms_data={"chest": 2})
    profile = _profile()
    request_before = request.model_dump(mode="json")
    profile_before = profile.model_dump(mode="json")
    generated = _generated_routine()
    calls = []

    def fake_generator(received_request, db, *, profile, recent_sets):
        calls.append((received_request, profile, recent_sets))
        assert received_request is not request
        assert received_request.readiness_level == "low"
        assert received_request.doms_data == {"chest": 2}
        return generated.model_copy(deep=True)

    preflight = preflight_routine_generation(request, profile=profile)
    result = generate_routine_with_choice(
        request,
        db=object(),
        profile=profile,
        selection=make_routine_generation_selection(preflight, "normalVolume"),
        _generator=fake_generator,
    )

    assert len(calls) == 1
    assert result.selected_choice == "normalVolume"
    assert result.routine == generated
    assert result.metrics.generation_invocation_count == 1
    assert result.metrics.optional_working_sets_removed == 0
    assert result.metrics.requested_time_min == 45
    assert result.metrics.estimated_time_min == 30
    assert result.metrics.optional_working_set_reduction_status == "notRequested"
    assert request.model_dump(mode="json") == request_before
    assert profile.model_dump(mode="json") == profile_before


def test_reduced_volume_preserves_adjuncts_and_recomputes_tempo_superset_time():
    request = _request()
    original = _generated_routine()
    original.summary_title = "예상 30분 루틴"
    original.warnings = [
        "요청 45분 중 예상 30분",
        "통증이 있으면 운동을 중단하세요.",
    ]
    original._underfill_diagnostics = {"actualDurationMin": 30}
    original_adjuncts = [
        [item.model_dump(mode="json") for item in block.prescription]
        for block in original.routine_blocks[2:]
    ]
    preflight = preflight_routine_generation(request)

    result = generate_routine_with_choice(
        request,
        db=object(),
        selection=make_routine_generation_selection(preflight, "reducedVolume"),
        _generator=lambda *args, **kwargs: original.model_copy(deep=True),
    )

    assert result.metrics.working_set_count_before_choice == 7
    assert result.metrics.working_set_count_after_choice == 5
    assert result.metrics.optional_working_sets_removed == 2
    assert result.metrics.generated_exercise_count_before_choice == 4
    assert result.metrics.generated_exercise_count_after_choice == 4
    assert result.metrics.requested_time_min == 45
    assert result.metrics.estimated_time_before_choice_min == 30
    assert result.metrics.reported_estimated_time_before_choice_min == 30
    assert result.metrics.estimated_time_min == 26
    assert result.metrics.optional_working_set_reduction_status == "applied"
    assert result.routine.total_estimated_time == 26
    assert [
        [item.model_dump(mode="json") for item in block.prescription]
        for block in result.routine.routine_blocks[2:]
    ] == original_adjuncts
    rationale = " ".join(result.routine.rationale_summary)
    assert "요청 시간은 45분" in rationale
    assert "조정 전 계획은 작업 세트 7세트, 예상 30분" in rationale
    assert "총 7세트" not in rationale
    assert "현재 처방은 작업 세트 5세트, 예상 26분입니다." in rationale
    assert "처방 조정 전" not in rationale
    assert result.routine.summary_title == "감소 볼륨 루틴: 작업 세트 5세트, 예상 26분"
    assert any(
        "요청 45분 기준 현재 처방은 작업 세트 5세트, 예상 26분입니다." in warning
        for warning in result.routine.warnings
    )
    assert any("요청 45분 중 예상 30분" in warning for warning in result.routine.warnings)
    assert "통증이 있으면 운동을 중단하세요." in result.routine.warnings
    assert result.routine.routine_blocks[0].exercise_rationale == (
        "현재 작업 세트는 3세트입니다. 감량 전 운동 선택 설명: 4세트 메인 운동"
    )
    assert result.routine.routine_blocks[1].exercise_rationale == (
        "현재 작업 세트는 2세트입니다. 감량 전 운동 선택 설명: 3 sets accessory"
    )
    assert result.routine._underfill_diagnostics == {}
    assert result.routine.routine_blocks[0].training_method.pause_sec == 1
    assert result.routine.routine_blocks[0].execution_group_type == "superset"


def test_reduced_volume_never_removes_last_working_set_from_selected_block():
    request = _request()
    original = _generated_routine()
    for block in original.routine_blocks[:2]:
        block.prescription = block.prescription[:1]
    original.total_estimated_time = 17
    preflight = preflight_routine_generation(request)

    result = generate_routine_with_choice(
        request,
        db=object(),
        selection=make_routine_generation_selection(preflight, "reducedVolume"),
        _generator=lambda *args, **kwargs: original.model_copy(deep=True),
    )

    assert result.status == "generated"
    assert result.metrics.optional_working_set_reduction_target == 1
    assert result.metrics.optional_working_sets_removed == 0
    assert result.metrics.optional_working_set_reduction_status == "noFurtherReduction"
    assert [len(block.prescription) for block in result.routine.routine_blocks[:2]] == [1, 1]
    assert len(result.routine.routine_blocks) == 4
    assert any("최소 1세트" in warning for warning in result.routine.warnings)


def test_floor_limited_reduction_reports_partial_change_truthfully():
    request = _request()
    original = _generated_routine()
    original.routine_blocks[0].prescription = original.routine_blocks[0].prescription[:2]
    original.routine_blocks[1].prescription = original.routine_blocks[1].prescription[:1]
    original.routine_blocks[2].prescription = [_set(1, target_reps=10)]
    original.routine_blocks[3].prescription = [_set(1, target_reps=10)]
    preflight = preflight_routine_generation(request)

    result = generate_routine_with_choice(
        request,
        db=object(),
        selection=make_routine_generation_selection(preflight, "reducedVolume"),
        _generator=lambda *args, **kwargs: original.model_copy(deep=True),
    )

    assert result.metrics.working_set_count_before_choice == 5
    assert result.metrics.working_set_count_after_choice == 4
    assert result.metrics.optional_working_set_reduction_target == 2
    assert result.metrics.optional_working_sets_removed == 1
    assert result.metrics.optional_working_set_reduction_status == "floorLimited"
    assert any("목표 2세트 중 1세트만 줄였습니다." in warning for warning in result.routine.warnings)
    assert not any("추가 세트 감소를 적용하지 않았습니다" in warning for warning in result.routine.warnings)


def test_zero_plan_remains_zero_and_does_not_gain_synthetic_time():
    request = _request()
    zero = RoutineDraftResponse(
        generation_status="failed",
        status_reason_code="emptyCandidate",
        is_fallback=False,
        total_estimated_time=0,
        summary_title="No eligible routine",
        rationale_summary=[],
        routine_blocks=[],
        warnings=[],
    )
    preflight = preflight_routine_generation(request)

    result = generate_routine_with_choice(
        request,
        db=object(),
        selection=make_routine_generation_selection(preflight, "reducedVolume"),
        _generator=lambda *args, **kwargs: zero.model_copy(deep=True),
    )

    assert result.routine.total_estimated_time == 0
    assert result.routine.routine_blocks == []
    assert result.metrics.working_set_count_before_choice == 0
    assert result.metrics.working_set_count_after_choice == 0
    assert result.metrics.generation_invocation_count == 1
    assert result.metrics.optional_working_set_reduction_status == "noWorkingSets"
    assert result.status == "generationFailed"


def test_active_uncleared_choice_still_uses_real_pipeline_safety_stop():
    request = _request(
        pain_areas=[],
        injury_conditions=[{
            "areaCode": "shoulder",
            "status": "activeUncleared",
        }],
    )
    preflight = preflight_routine_generation(request)

    with (
        patch("engines.routine_pipeline.get_user_constraint_profile_from_sqlite", return_value={}),
        patch("engines.routine_pipeline.get_candidate_exercises") as candidates,
        patch("engines.routine_pipeline.get_llm") as llm,
    ):
        result = generate_routine_with_choice(
            request,
            db=object(),
            selection=make_routine_generation_selection(preflight, "normalVolume"),
        )

    candidates.assert_not_called()
    llm.assert_not_called()
    assert result.constraints.active_uncleared_count == 1
    assert result.metrics.generation_invocation_count == 1
    assert result.status == "generationFailed"
    assert result.routine.generation_status == "failed"
    assert result.routine.status_reason_code == "emptyCandidate"
    assert result.routine.routine_blocks == []


def test_facade_has_no_consumer_candidate_input_and_ready_request_delegates_once():
    assert "candidates" not in inspect.signature(generate_routine_with_choice).parameters
    request = _request(pain_areas=[])
    generated = _generated_routine()
    calls = []

    result = generate_routine_with_choice(
        request,
        db=object(),
        _generator=lambda *args, **kwargs: calls.append((args, kwargs)) or generated,
    )

    assert result.status == "generated"
    assert result.selected_choice is None
    assert result.warning is None
    assert result.metrics.generation_invocation_count == 1
    assert len(calls) == 1

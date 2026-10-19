import pytest
from pydantic import ValidationError

from engines.fallback import build_routine_draft, generate_fallback_routine
from engines.prescription.adjuncts import (
    ADJUNCT_PRESCRIPTION_PROFILES,
    apply_adjunct_prescription_profile,
    apply_requested_adjunct_prescription,
)
from engines.prescription.estimator import estimate_routine_time_min
from engines.prescription.targets import apply_deterministic_targets
from engines.llm_parser import trim_routine_to_time_budget
from engines.rule_critic import _total_working_sets
from engines.routine_blueprint import build_routine_blueprint
from engines.routine_selection import apply_blueprint_execution_policy_to_draft
from engines.schemas import LLMExercisePlan, LLMRoutineOutput, RoutineRequest, SetPrescription


def _plan(exercise_id: str = "adjunct") -> LLMExercisePlan:
    return LLMExercisePlan(
        exercise_id=exercise_id,
        exercise_name="테스트 운동",
        movement_type="ISOLATION",
        primary_muscles=["chest"],
        target_reps=12,
        sets=3,
        rest_time_sec=75,
        target_rir=2,
        exercise_rationale="테스트",
    )


def test_each_adjunct_option_has_a_non_working_prescription_profile():
    assert set(ADJUNCT_PRESCRIPTION_PROFILES) == {
        "mobility_warmup",
        "cooldown_stretch",
        "conditioning_finisher",
        "endurance_support",
        "prehab_accessory",
    }
    assert all(profile.set_type != "working" for profile in ADJUNCT_PRESCRIPTION_PROFILES.values())


def test_timed_and_interval_response_shapes_require_modality_metadata():
    with pytest.raises(ValidationError):
        SetPrescription(
            set_index=1,
            set_type="timed",
            target_reps=1,
            target_rest_sec=0,
        )
    with pytest.raises(ValidationError):
        SetPrescription(
            set_index=1,
            set_type="interval",
            target_reps=1,
            target_rest_sec=0,
            target_duration_sec=480,
            interval_work_sec=30,
            interval_recovery_sec=60,
        )
    with pytest.raises(ValidationError):
        SetPrescription(
            set_index=1,
            set_type="distance",
            target_reps=1,
            target_rest_sec=60,
        )


def test_optional_modality_fields_are_omitted_from_working_set_serialization():
    item = SetPrescription(
        set_index=1,
        set_type="working",
        target_reps=8,
        target_rest_sec=90,
    )

    assert item.model_dump() == {
        "set_index": 1,
        "set_type": "working",
        "target_reps": 8,
        "target_weight_kg": None,
        "target_rir": 2,
        "target_rest_sec": 90,
    }


def test_cooldown_serializes_timed_duration_without_rir():
    plan = _plan()
    assert apply_adjunct_prescription_profile(plan, "cooldown_stretch")

    output = LLMRoutineOutput(
        total_estimated_time=0,
        summary_title="테스트",
        rationale_summary=[],
        exercises=[plan],
    )
    draft = build_routine_draft(output, "success", "none", False, preserve_input_order=True)

    block = draft.routine_blocks[0]
    assert block.default_rest_sec == 15
    assert len(block.prescription) == 2
    assert all(item.set_type == "timed" for item in block.prescription)
    assert all(item.target_duration_sec == 30 for item in block.prescription)
    assert all(item.target_rir is None for item in block.prescription)


def test_conditioning_interval_uses_one_explicit_interval_block():
    plan = _plan()
    assert apply_adjunct_prescription_profile(plan, "conditioning_finisher")

    output = LLMRoutineOutput(
        total_estimated_time=0,
        summary_title="테스트",
        rationale_summary=[],
        exercises=[plan],
    )
    draft = build_routine_draft(output, "success", "none", False, preserve_input_order=True)
    item = draft.routine_blocks[0].prescription[0]

    assert item.set_type == "interval"
    assert item.target_duration_sec == 480
    assert item.interval_work_sec == 30
    assert item.interval_recovery_sec == 60
    assert item.interval_rounds == 6
    assert item.target_rir is None
    assert draft.total_estimated_time == 13


def test_requested_adjunct_profile_only_changes_the_matching_candidate():
    working = _plan("working")
    adjunct = _plan("adjunct")
    output = LLMRoutineOutput(
        total_estimated_time=0,
        summary_title="테스트",
        rationale_summary=[],
        exercises=[working, adjunct],
    )
    candidates = [
        {"id": "working", "movement_type": "COMPOUND"},
        {
            "id": "adjunct",
            "name_kr": "문틀 흉근 동적 스트레치",
            "movement_type": "STATIC",
            "adjunct_option_codes": ["mobility_warmup"],
            "adjunct_catalog_review_status": "core-reviewed",
        },
    ]

    apply_requested_adjunct_prescription(output, ["mobility_warmup"], candidates)

    assert working._prescription_set_type == "working"
    assert adjunct._prescription_set_type == "mobility"
    assert adjunct.sets == 2
    assert adjunct.target_reps == 8
    assert adjunct.target_rir is None
    assert output._selection_phase_by_exercise_id == {
        "adjunct": "warmup_or_preparation",
    }


def test_two_requested_adjunct_profiles_apply_to_distinct_matching_candidates():
    working = _plan("working")
    mobility = _plan("mobility")
    cooldown = _plan("cooldown")
    output = LLMRoutineOutput(
        total_estimated_time=0,
        summary_title="테스트",
        rationale_summary=[],
        exercises=[working, mobility, cooldown],
    )
    candidates = [
        {"id": "working", "movement_type": "COMPOUND"},
        {
            "id": "mobility",
            "name_kr": "동적 가슴 모빌리티",
            "movement_type": "STATIC",
            "adjunct_option_codes": ["mobility_warmup"],
            "adjunct_catalog_review_status": "core-reviewed",
        },
        {
            "id": "cooldown",
            "name_kr": "문틀 가슴 스트레치",
            "movement_type": "STATIC",
            "adjunct_option_codes": ["cooldown_stretch"],
            "adjunct_catalog_review_status": "core-reviewed",
        },
    ]

    apply_requested_adjunct_prescription(
        output,
        ["mobility_warmup", "cooldown_stretch"],
        candidates,
    )

    assert working._prescription_set_type == "working"
    assert mobility._prescription_set_type == "mobility"
    assert mobility.target_reps == 8
    assert cooldown._prescription_set_type == "timed"
    assert cooldown._target_duration_sec == 30
    assert output._selection_phase_by_exercise_id == {
        "mobility": "warmup_or_preparation",
        "cooldown": "cooldown_recovery",
    }


def test_non_working_adjunct_sets_do_not_count_toward_working_set_cap():
    blocks = [
        {"prescription": [{"set_type": "working"}, {"set_type": "working"}]},
        {"prescription": [{"set_type": "timed"}, {"set_type": "timed"}]},
        {"prescription": [{"setType": "interval"}]},
        {"prescription": [{"set_type": "distance"}]},
    ]
    assert _total_working_sets(blocks) == 3


def test_timed_duration_drives_time_estimation_instead_of_rep_work_proxy():
    plan = _plan()
    apply_adjunct_prescription_profile(plan, "endurance_support")
    assert estimate_routine_time_min([plan]) == 15


def test_deterministic_target_fill_preserves_prehab_profile_exactly():
    working = _plan("working")
    prehab = _plan("prehab")
    apply_adjunct_prescription_profile(prehab, "prehab_accessory")
    output = LLMRoutineOutput(
        total_estimated_time=0,
        summary_title="테스트",
        rationale_summary=[],
        exercises=[working, prehab],
    )

    adjusted = apply_deterministic_targets(
        output,
        goal="strength",
        recent_sets=[],
        profile=None,
        readiness_level="high",
        target_split_label="push",
        target_muscles=["chest", "triceps"],
        time_available_min=30,
    )

    preserved = next(item for item in adjusted.exercises if item.exercise_id == "prehab")
    assert preserved._prescription_set_type == "prehab"
    assert preserved.sets == 2
    assert preserved.target_reps == 12
    assert preserved.target_rir == 3
    assert preserved.rest_time_sec == 45


def test_time_trim_removes_adjunct_before_reducing_main_work():
    working = _plan("working")
    working.movement_type = "COMPOUND"
    working.sets = 4
    working.rest_time_sec = 90
    prehab = _plan("prehab")
    apply_adjunct_prescription_profile(prehab, "prehab_accessory")
    output = LLMRoutineOutput(
        total_estimated_time=0,
        summary_title="테스트",
        rationale_summary=[],
        exercises=[working, prehab],
    )
    output._adjunct_option_by_exercise_id = {"prehab": "prehab_accessory"}

    assert trim_routine_to_time_budget(
        output,
        15,
        minimum_main_exercise_count=1,
    )
    assert prehab not in output.exercises
    assert working.sets == 4
    assert output._adjunct_suppression_reason_by_option == {
        "prehab_accessory": "insufficientTimeAfterMainMinimum"
    }


@pytest.mark.parametrize("budget_delta", [0, 1])
def test_time_boundary_keeps_adjunct_at_exact_or_larger_budget(budget_delta):
    working = _plan("working")
    working.movement_type = "COMPOUND"
    cooldown = _plan("cooldown")
    apply_adjunct_prescription_profile(cooldown, "cooldown_stretch")
    output = LLMRoutineOutput(
        total_estimated_time=0,
        summary_title="테스트",
        rationale_summary=[],
        exercises=[working, cooldown],
    )
    output._adjunct_option_by_exercise_id = {"cooldown": "cooldown_stretch"}
    exact_budget = estimate_routine_time_min(output.exercises)

    assert trim_routine_to_time_budget(output, exact_budget + budget_delta)
    assert [exercise.exercise_id for exercise in output.exercises] == [
        "working",
        "cooldown",
    ]


def test_time_boundary_suppresses_adjunct_at_one_minute_below_exact_budget():
    working = _plan("working")
    working.movement_type = "COMPOUND"
    cooldown = _plan("cooldown")
    apply_adjunct_prescription_profile(cooldown, "cooldown_stretch")
    output = LLMRoutineOutput(
        total_estimated_time=0,
        summary_title="테스트",
        rationale_summary=[],
        exercises=[working, cooldown],
    )
    output._adjunct_option_by_exercise_id = {"cooldown": "cooldown_stretch"}
    exact_budget = estimate_routine_time_min(output.exercises)
    assert estimate_routine_time_min([working]) <= exact_budget - 1

    assert trim_routine_to_time_budget(
        output,
        exact_budget - 1,
        minimum_main_exercise_count=1,
    )
    assert [exercise.exercise_id for exercise in output.exercises] == ["working"]


def test_time_boundary_removes_lower_priority_adjunct_first():
    working = _plan("working")
    working.movement_type = "COMPOUND"
    mobility = _plan("mobility")
    cooldown = _plan("cooldown")
    apply_adjunct_prescription_profile(mobility, "mobility_warmup")
    apply_adjunct_prescription_profile(cooldown, "cooldown_stretch")
    output = LLMRoutineOutput(
        total_estimated_time=0,
        summary_title="테스트",
        rationale_summary=[],
        exercises=[mobility, working, cooldown],
    )
    output._adjunct_option_by_exercise_id = {
        "mobility": "mobility_warmup",
        "cooldown": "cooldown_stretch",
    }
    budget = estimate_routine_time_min([mobility, working])
    assert estimate_routine_time_min(output.exercises) > budget

    assert trim_routine_to_time_budget(
        output,
        budget,
        minimum_main_exercise_count=1,
    )
    assert [exercise.exercise_id for exercise in output.exercises] == [
        "mobility",
        "working",
    ]
    assert output._adjunct_suppression_reason_by_option == {
        "cooldown_stretch": "insufficientTimeAfterMainMinimum"
    }


def test_fallback_keeps_requested_cooldown_after_working_set_budget_is_exhausted():
    request = RoutineRequest(
        user_id="adjunct-fallback",
        target_split_label="push",
        time_available_min=60,
        pain_areas=[],
        doms_data={},
        equipment=[],
        adjunct_options=["cooldown_stretch"],
    )
    candidates = [
        {
            "id": "machine_chest_press",
            "name_kr": "머신 체스트 프레스",
            "primary_muscle": "chest",
            "secondary_muscle": "triceps",
            "equipment_req": "MACHINE",
            "movement_type": "COMPOUND",
            "efficiency_tier": 4,
        },
        {
            "id": "doorway_chest_stretch",
            "name_kr": "문틀 가슴 스트레치",
            "primary_muscle": "chest",
            "secondary_muscle": "",
            "equipment_req": "BODYWEIGHT",
            "movement_type": "STATIC",
            "efficiency_tier": 3,
            "adjunct_option_codes": ["cooldown_stretch"],
        },
    ]

    result = generate_fallback_routine(
        request,
        candidates=candidates,
        max_total_sets=1,
        target_exercise_count=2,
    )

    assert [block.exercise_id for block in result.routine_blocks] == [
        "machine_chest_press",
        "doorway_chest_stretch",
    ]
    assert result.routine_blocks[0].prescription[0].set_type == "working"
    cooldown = result.routine_blocks[1]
    assert [item.set_type for item in cooldown.prescription] == ["timed", "timed"]
    assert [item.target_duration_sec for item in cooldown.prescription] == [30, 30]
    assert all(item.target_rir is None for item in cooldown.prescription)


def test_fallback_reserves_two_compatible_adjuncts_without_losing_main_work():
    request = RoutineRequest(
        user_id="two-adjunct-fallback",
        target_split_label="push",
        time_available_min=60,
        pain_areas=[],
        doms_data={},
        equipment=[],
        adjunct_options=["mobility_warmup", "cooldown_stretch"],
    )
    candidates = [
        {
            "id": "machine_chest_press",
            "name_kr": "머신 체스트 프레스",
            "primary_muscle": "chest",
            "secondary_muscle": "triceps",
            "equipment_req": "MACHINE",
            "movement_type": "COMPOUND",
            "efficiency_tier": 4,
        },
        {
            "id": "dynamic_chest_mobility",
            "name_kr": "동적 가슴 모빌리티",
            "primary_muscle": "chest",
            "equipment_req": "BODYWEIGHT",
            "movement_type": "STATIC",
            "efficiency_tier": 3,
            "adjunct_option_codes": ["mobility_warmup"],
        },
        {
            "id": "doorway_chest_stretch",
            "name_kr": "문틀 가슴 스트레치",
            "primary_muscle": "chest",
            "equipment_req": "BODYWEIGHT",
            "movement_type": "STATIC",
            "efficiency_tier": 3,
            "adjunct_option_codes": ["cooldown_stretch"],
        },
    ]

    result = generate_fallback_routine(
        request,
        candidates=candidates,
        max_total_sets=1,
        target_exercise_count=3,
        minimum_main_exercise_count=1,
        applied_adjunct_options=["mobility_warmup", "cooldown_stretch"],
    )

    by_id = {block.exercise_id: block for block in result.routine_blocks}
    assert set(by_id) == {
        "machine_chest_press",
        "dynamic_chest_mobility",
        "doorway_chest_stretch",
    }
    assert by_id["machine_chest_press"].prescription[0].set_type == "working"
    assert by_id["dynamic_chest_mobility"].prescription[0].set_type == "mobility"
    assert by_id["doorway_chest_stretch"].prescription[0].set_type == "timed"


@pytest.mark.parametrize(
    ("option", "exercise_id", "exercise_name", "set_type", "duration_sec"),
    [
        ("conditioning_finisher", "rower", "로잉머신 인터벌", "interval", 480),
        ("endurance_support", "treadmill", "트레드밀 걷기", "timed", 600),
    ],
)
def test_fallback_reserves_time_for_long_adjunct_before_filling_working_sets(
    option,
    exercise_id,
    exercise_name,
    set_type,
    duration_sec,
):
    request = RoutineRequest(
        user_id="adjunct-fallback-time-reservation",
        target_split_label="push",
        time_available_min=30,
        pain_areas=[],
        doms_data={},
        equipment=[],
        adjunct_options=[option],
        goal="strength",
    )
    candidates = [
        {
            "id": "machine_chest_press",
            "name_kr": "머신 체스트 프레스",
            "primary_muscle": "chest",
            "secondary_muscle": "triceps",
            "equipment_req": "MACHINE",
            "movement_type": "COMPOUND",
            "efficiency_tier": 4,
        },
        {
            "id": exercise_id,
            "name_kr": exercise_name,
            "name_en": exercise_name,
            "primary_muscle": "quadriceps",
            "secondary_muscle": "",
            "equipment_req": "MACHINE",
            "movement_type": "COMPOUND",
            "efficiency_tier": 3,
            "adjunct_option_codes": [option],
            "training_style_tags": ["conditioning" if option == "conditioning_finisher" else "endurance"],
        },
    ]

    result = generate_fallback_routine(
        request,
        candidates=candidates,
        max_total_sets=6,
        target_exercise_count=2,
        applied_adjunct_options=[option],
    )

    assert result.total_estimated_time <= 30
    assert any(
        item.set_type == "working"
        for block in result.routine_blocks
        for item in block.prescription
    )
    adjunct = next(block for block in result.routine_blocks if block.exercise_id == exercise_id)
    assert adjunct.prescription[0].set_type == set_type
    assert adjunct.prescription[0].target_duration_sec == duration_sec


def test_fallback_blueprint_reordering_preserves_timed_prescription_metadata():
    working = _plan("machine_chest_press")
    cooldown = _plan("doorway_chest_stretch")
    apply_adjunct_prescription_profile(cooldown, "cooldown_stretch")
    output = LLMRoutineOutput(
        total_estimated_time=0,
        summary_title="테스트",
        rationale_summary=[],
        exercises=[working, cooldown],
    )
    candidates = [
        {
            "id": "machine_chest_press",
            "name_kr": "머신 체스트 프레스",
            "primary_muscle": "chest",
            "movement_type": "COMPOUND",
        },
        {
            "id": "doorway_chest_stretch",
            "name_kr": "문틀 가슴 스트레치",
            "primary_muscle": "chest",
            "movement_type": "STATIC",
            "adjunct_option_codes": ["cooldown_stretch"],
        },
    ]
    draft = build_routine_draft(
        output,
        "fallback",
        "networkError",
        True,
        ranked_candidates=candidates,
        preserve_input_order=True,
    )
    blueprint = build_routine_blueprint(
        target_split_label="push",
        target_muscles=["chest", "front-deltoids", "triceps"],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=60,
        adjunct_options=["cooldown_stretch"],
    )

    reordered = apply_blueprint_execution_policy_to_draft(draft, blueprint, candidates)

    cooldown_block = next(
        block for block in reordered.routine_blocks
        if block.exercise_id == "doorway_chest_stretch"
    )
    assert cooldown_block.session_phase == "cooldown_recovery"
    assert [item.set_type for item in cooldown_block.prescription] == ["timed", "timed"]
    assert [item.target_duration_sec for item in cooldown_block.prescription] == [30, 30]
    assert all(item.target_rir is None for item in cooldown_block.prescription)

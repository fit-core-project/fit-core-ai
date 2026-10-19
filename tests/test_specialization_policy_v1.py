import pytest
from pydantic import ValidationError

from engines.candidate_ranker import score_candidate_exercises
from engines.exercise_display import candidate_display_name
from engines.personalization_history import suggest_specialization_from_recent_sets
from engines.prescription.estimator import estimate_routine_time_min
from engines.prescription.metadata import apply_candidate_prescription_metadata
from engines.routine_blueprint import build_routine_blueprint
from engines.schemas import (
    LLMExercisePlan,
    LLMRoutineOutput,
    PersonalizationSuggestion,
    RecentSetRecord,
    RoutineBlock,
    RoutineDraftResponse,
    RoutineRequest,
    SetPrescription,
)
from engines.specialization_policy import (
    apply_blueprint_specialization_volume,
    apply_blueprint_training_method,
    apply_training_method_to_plan,
    candidate_encodes_training_method,
    candidate_matches_training_method,
    training_method_instruction,
)


def _candidate(
    exercise_id: str,
    name_kr: str,
    primary_muscle: str,
    *,
    detail_subpart: str = "",
    movement_type: str = "COMPOUND",
    equipment_req: str = "DUMBBELL",
    bench_angle_band: str = "",
) -> dict:
    return {
        "id": exercise_id,
        "name_kr": name_kr,
        "name_en": "",
        "primary_muscle": primary_muscle,
        "secondary_muscle": "",
        "detail_subpart": detail_subpart,
        "movement_type": movement_type,
        "equipment_req": equipment_req,
        "bench_angle_band": bench_angle_band,
        "difficulty_tier": 3,
        "efficiency_tier": 3,
        "quality_score": 85,
        "strength_effect": "high",
        "hypertrophy_effect": "high",
        "failure_penalty": "medium",
        "mainstream_gym_score": 80,
        "exercise_familiarity_tier": "standard",
        "novelty_level": "low",
        "main_routine_default_allowed": 1,
        "training_style_tags": "gym,bodybuilding",
        "prescription_mode": "reps",
    }


def _request(**overrides) -> RoutineRequest:
    payload = {
        "targetSplitLabel": "legs",
        "targetMuscles": [],
        "readinessLevel": "normal",
        "timeAvailableMin": 60,
        "goal": "hypertrophy",
        "experienceLevel": "intermediate",
    }
    payload.update(overrides)
    return RoutineRequest.model_validate(payload)


def test_default_request_keeps_focus_and_training_method_unset():
    request = _request()

    assert request.specialization_focus is None
    assert request.training_method is None
    assert request.method_mode == "auto"


def test_advanced_auto_mode_resolves_goal_compatible_method_in_blueprint():
    strength = build_routine_blueprint(
        target_split_label="chest",
        target_muscles=[],
        goal="strength",
        experience_level="advanced",
        readiness_level="normal",
        time_available_min=60,
        method_mode="auto",
    )
    hypertrophy = build_routine_blueprint(
        target_split_label="legs",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="advanced",
        readiness_level="normal",
        time_available_min=60,
        method_mode="auto",
    )

    assert strength.training_method == "pausedReps"
    assert strength.training_method_source == "auto"
    assert strength.training_method_reason_code == "advancedStrengthMainControl"
    assert hypertrophy.training_method == "tempoEccentric"
    assert hypertrophy.training_method_source == "auto"


def test_auto_method_is_suppressed_for_non_advanced_low_readiness_and_target_doms():
    intermediate = build_routine_blueprint(
        target_split_label="chest",
        target_muscles=[],
        goal="strength",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=60,
        method_mode="auto",
    )
    low_readiness = build_routine_blueprint(
        target_split_label="chest",
        target_muscles=[],
        goal="strength",
        experience_level="advanced",
        readiness_level="low",
        time_available_min=60,
        method_mode="auto",
    )
    target_doms = build_routine_blueprint(
        target_split_label="chest",
        target_muscles=[],
        goal="strength",
        experience_level="advanced",
        readiness_level="normal",
        time_available_min=60,
        method_mode="auto",
        doms_data={"chest": 2},
    )

    assert intermediate.training_method is None
    assert intermediate.training_method_reason_code == "advancedExperienceRequired"
    assert low_readiness.training_method is None
    assert low_readiness.training_method_reason_code == "lowReadiness"
    assert target_doms.training_method is None
    assert target_doms.training_method_reason_code == "targetDoms"


def test_auto_method_uses_same_split_two_session_cooldown():
    recent_same_split = build_routine_blueprint(
        target_split_label="chest",
        target_muscles=[],
        goal="strength",
        experience_level="advanced",
        readiness_level="normal",
        time_available_min=60,
        method_mode="auto",
        recent_training_method_uses=[
            {
                "methodCode": "pausedReps",
                "targetSplitLabel": "chest",
                "sessionsAgo": 2,
            }
        ],
    )
    recent_other_split = build_routine_blueprint(
        target_split_label="chest",
        target_muscles=[],
        goal="strength",
        experience_level="advanced",
        readiness_level="normal",
        time_available_min=60,
        method_mode="auto",
        recent_training_method_uses=[
            {
                "methodCode": "pausedReps",
                "targetSplitLabel": "legs",
                "sessionsAgo": 1,
            }
        ],
    )
    old_same_split = build_routine_blueprint(
        target_split_label="chest",
        target_muscles=[],
        goal="strength",
        experience_level="advanced",
        readiness_level="normal",
        time_available_min=60,
        method_mode="auto",
        recent_training_method_uses=[
            {
                "methodCode": "pausedReps",
                "targetSplitLabel": "chest",
                "sessionsAgo": 3,
            }
        ],
    )

    assert recent_same_split.training_method is None
    assert recent_same_split.training_method_reason_code == "recentMethodCooldown"
    assert recent_other_split.training_method == "pausedReps"
    assert old_same_split.training_method == "pausedReps"


def test_request_validates_recent_training_method_history_shape():
    request = _request(
        experienceLevel="advanced",
        recentTrainingMethodUses=[
            {
                "methodCode": "tempoEccentric",
                "targetSplitLabel": "Legs",
                "sessionsAgo": 1,
            }
        ],
    )

    assert request.recent_training_method_uses[0].target_split_label == "legs"

    with pytest.raises(ValidationError):
        _request(
            experienceLevel="advanced",
            recentTrainingMethodUses=[
                {
                    "methodCode": "tempoEccentric",
                    "targetSplitLabel": "legs",
                    "sessionsAgo": 1,
                },
                {
                    "methodCode": "tempoEccentric",
                    "targetSplitLabel": "legs",
                    "sessionsAgo": 1,
                },
            ],
        )


def test_method_mode_off_and_manual_override_are_distinct():
    off = build_routine_blueprint(
        target_split_label="chest",
        target_muscles=[],
        goal="strength",
        experience_level="advanced",
        readiness_level="normal",
        time_available_min=60,
        method_mode="off",
    )
    manual = build_routine_blueprint(
        target_split_label="chest",
        target_muscles=[],
        goal="strength",
        experience_level="advanced",
        readiness_level="normal",
        time_available_min=60,
        method_mode="manual",
        requested_method="pausedReps",
    )

    assert off.training_method is None
    assert off.training_method_reason_code == "disabledByUser"
    assert manual.training_method == "pausedReps"
    assert manual.training_method_source == "manual"
    assert manual.training_method_reason_code == "manualOverride"


def test_manual_method_cannot_bypass_target_doms_or_managed_injury():
    target_doms = build_routine_blueprint(
        target_split_label="chest",
        target_muscles=["chest"],
        goal="strength",
        experience_level="advanced",
        readiness_level="normal",
        time_available_min=60,
        method_mode="manual",
        requested_method="pausedReps",
        doms_data={"chest": 2},
    )
    managed_injury = build_routine_blueprint(
        target_split_label="chest",
        target_muscles=["chest"],
        goal="strength",
        experience_level="advanced",
        readiness_level="normal",
        time_available_min=60,
        method_mode="manual",
        requested_method="pausedReps",
        injury_conditions=[
            {
                "areaCode": "shoulder",
                "status": "managedWithRestrictions",
                "restrictedMovementPatterns": ["horizontalPush"],
            }
        ],
    )

    assert target_doms.training_method is None
    assert target_doms.training_method_reason_code == "targetDoms"
    assert managed_injury.training_method is None
    assert managed_injury.training_method_reason_code == "managedInjuryRestriction"


def test_request_rejects_invalid_method_mode_combinations():
    with pytest.raises(ValidationError):
        _request(
            experienceLevel="advanced",
            methodMode="manual",
        )
    with pytest.raises(ValidationError):
        _request(
            experienceLevel="advanced",
            methodMode="off",
            requestedMethod="tempoEccentric",
        )
    with pytest.raises(ValidationError):
        _request(
            experienceLevel="advanced",
            methodMode="auto",
            requestedMethod="tempoEccentric",
        )

    manual = _request(
        experienceLevel="advanced",
        methodMode="manual",
        requestedMethod="tempoEccentric",
    )
    assert manual.training_method == "tempoEccentric"
    assert manual.requested_method == "tempoEccentric"


def test_legacy_paused_press_maps_to_training_method_without_fake_focus():
    request = _request(
        targetSplitLabel="chest",
        goal="strength",
        experienceLevel="advanced",
        advancedChestFocus="pausedPress",
    )

    assert request.specialization_focus is None
    assert request.training_method == "pausedReps"


def test_legacy_upper_chest_maps_to_generic_specialization_focus():
    request = _request(
        targetSplitLabel="chest",
        experienceLevel="advanced",
        advancedChestFocus="upperChest",
    )

    assert request.specialization_focus == "upperChest"
    assert request.training_method is None


def test_beginner_cannot_request_optional_specialization_or_advanced_method():
    with pytest.raises(ValidationError):
        _request(
            experienceLevel="beginner",
            specializationFocus="hamstringEmphasis",
        )

    with pytest.raises(ValidationError):
        _request(
            targetSplitLabel="chest",
            goal="strength",
            experienceLevel="beginner",
            trainingMethod="pausedReps",
        )


def test_intermediate_cannot_request_focus_or_advanced_training_method():
    with pytest.raises(ValidationError):
        _request(specializationFocus="hamstringEmphasis")

    with pytest.raises(ValidationError):
        _request(
            targetSplitLabel="chest",
            goal="strength",
            experienceLevel="intermediate",
            trainingMethod="pausedReps",
        )


def test_focus_must_belong_to_the_selected_session_targets():
    with pytest.raises(ValueError):
        build_routine_blueprint(
            target_split_label="chest",
            target_muscles=[],
            goal="hypertrophy",
            experience_level="advanced",
            readiness_level="normal",
            time_available_min=60,
            specialization_focus="hamstringEmphasis",
        )


def test_hamstring_focus_boosts_hamstring_candidate_without_removing_quadriceps():
    candidates = [
        _candidate("leg_press", "레그 프레스", "quadriceps", detail_subpart="머신 프레스"),
        _candidate(
            "seated_leg_curl",
            "시티드 레그 컬",
            "hamstring",
            detail_subpart="무릎 굴곡/레그컬",
            movement_type="ISOLATION",
            equipment_req="MACHINE",
        ),
    ]

    ranked = score_candidate_exercises(
        candidates,
        ["quadriceps", "hamstring", "gluteal"],
        top_n=5,
        target_split_label="legs",
        experience_level="advanced",
        specialization_focus="hamstringEmphasis",
        goal="hypertrophy",
    )

    by_id = {candidate["id"]: candidate for candidate in ranked}
    assert set(by_id) == {"leg_press", "seated_leg_curl"}
    assert any(
        item["rule_code"] == "specialization_focus_match"
        for item in by_id["seated_leg_curl"]["score_boosts"]
    )


def test_advanced_triceps_focus_keeps_kickback_as_accessory_not_top_overload_choice():
    extension = _candidate(
        "triceps_extension",
        "라잉 덤벨 트라이셉스 익스텐션",
        "triceps",
        movement_type="ISOLATION",
    )
    kickback = _candidate(
        "triceps_kickback",
        "덤벨 킥백",
        "triceps",
        movement_type="ISOLATION",
    )
    kickback["hypertrophy_effect"] = "elite"

    ranked = score_candidate_exercises(
        [kickback, extension],
        ["biceps", "triceps", "forearm"],
        top_n=2,
        target_split_label="arm",
        experience_level="advanced",
        specialization_focus="tricepsEmphasis",
        goal="hypertrophy",
    )

    assert [candidate["id"] for candidate in ranked] == [
        "triceps_extension",
        "triceps_kickback",
    ]
    assert any(
        item["rule_code"] == "advanced_kickback_accessory_penalty"
        for item in ranked[1]["score_penalties"]
    )


def test_back_focus_distinguishes_vertical_lat_and_horizontal_mid_back_candidates():
    lat = _candidate(
        "lat_pulldown",
        "랫풀다운",
        "upper-back",
        detail_subpart="수직 당기기/광배",
        equipment_req="CABLE",
    )
    row = _candidate(
        "seated_row",
        "시티드 로우",
        "upper-back",
        detail_subpart="수평 당기기/등두께",
        equipment_req="CABLE",
    )

    lat_ranked = score_candidate_exercises(
        [row, lat],
        ["upper-back"],
        top_n=2,
        target_split_label="pull",
        experience_level="advanced",
        specialization_focus="latEmphasis",
        goal="hypertrophy",
    )
    mid_ranked = score_candidate_exercises(
        [lat, row],
        ["upper-back"],
        top_n=2,
        target_split_label="pull",
        experience_level="advanced",
        specialization_focus="midBackEmphasis",
        goal="hypertrophy",
    )

    assert lat_ranked[0]["id"] == "lat_pulldown"
    assert mid_ranked[0]["id"] == "seated_row"


def test_paused_and_tempo_methods_are_detected_as_methods_not_new_focuses():
    paused = _candidate(
        "paused_bench",
        "바벨 플랫 하단 정지 체스트 프레스",
        "chest",
        equipment_req="BARBELL",
        bench_angle_band="flat",
    )
    tempo = _candidate(
        "tempo_squat",
        "바벨 백 스쿼트 3초 하강 템포",
        "quadriceps",
        equipment_req="BARBELL",
    )

    assert candidate_matches_training_method(paused, "pausedReps")
    assert candidate_matches_training_method(tempo, "tempoEccentric")
    assert candidate_encodes_training_method(paused, "pausedReps")
    assert candidate_encodes_training_method(tempo, "tempoEccentric")
    assert not candidate_encodes_training_method(paused, "tempoEccentric")

    paused_instruction = training_method_instruction("pausedReps")
    tempo_instruction = training_method_instruction("tempoEccentric")
    assert paused_instruction["pauseSec"] == 2
    assert tempo_instruction["eccentricSec"] == 3


def test_training_method_compatibility_rejects_non_curated_or_non_gym_contexts():
    curated_press = _candidate(
        "curated_press",
        "바벨 플랫 체스트 프레스",
        "chest",
        equipment_req="BARBELL",
    )
    crossfit_press = dict(
        curated_press,
        id="crossfit_press",
        name_kr="CrossFit 체스트 프레스 메트콘",
        training_method_codes="pausedReps",
    )
    encyclopedia_only = dict(
        curated_press,
        id="encyclopedia_only",
        main_routine_default_allowed=0,
        training_method_codes="pausedReps",
    )
    timed_press = dict(
        curated_press,
        id="timed_press",
        prescription_mode="duration",
        training_method_codes="pausedReps",
    )

    assert candidate_matches_training_method(curated_press, "pausedReps")
    assert not candidate_matches_training_method(crossfit_press, "pausedReps")
    assert not candidate_matches_training_method(encyclopedia_only, "pausedReps")
    assert not candidate_matches_training_method(timed_press, "pausedReps")


def test_training_method_is_exposed_as_instruction_on_a_familiar_exercise_name():
    paused = _candidate(
        "paused_bench",
        "바벨 플랫 일반그립 하단 정지 체스트 프레스",
        "chest",
        equipment_req="BARBELL",
        bench_angle_band="flat",
    )
    paused.update(
        {
            "user_facing_display_name_kr": "바벨 플랫 체스트 프레스",
            "main_routine_default_allowed": 1,
        }
    )
    ranked = score_candidate_exercises(
        [paused],
        ["chest"],
        top_n=1,
        target_split_label="chest",
        experience_level="advanced",
        advanced_chest_focus="balanced",
        training_method="pausedReps",
        goal="strength",
    )

    assert candidate_display_name(ranked[0]) == "바벨 플랫 체스트 프레스"
    assert any(
        item["rule_code"] == "training_method_match"
        for item in ranked[0]["score_boosts"]
    )

    plan = LLMExercisePlan(
        exercise_id="paused_bench",
        exercise_name="바벨 플랫 체스트 프레스",
        movement_type="COMPOUND",
        primary_muscles=["chest"],
        target_reps=5,
        sets=3,
        rest_time_sec=120,
        target_rir=2,
        exercise_rationale="정지 반복을 적용합니다.",
    )
    apply_candidate_prescription_metadata(plan, ranked[0])
    apply_training_method_to_plan(plan, "pausedReps", candidate=ranked[0])

    assert plan._training_method_code == "pausedReps"
    assert plan._training_method_instruction["pauseSec"] == 2
    assert "하단 지점" in plan._training_method_instruction["instructionKo"]
    assert estimate_routine_time_min([plan]) == 13


def test_blueprint_assigns_focus_and_method_to_one_compatible_slot():
    blueprint = build_routine_blueprint(
        target_split_label="legs",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="advanced",
        readiness_level="normal",
        time_available_min=60,
        specialization_focus="hamstringEmphasis",
        training_method="tempoEccentric",
    )

    # A dedicated focus role now preserves the ordinary posterior-chain role.
    assert blueprint.specialization_slot_id == "specialization_focus"
    assert blueprint.training_method_slot_id == "specialization_focus"


def test_focus_without_a_default_split_slot_gets_one_required_direct_slot():
    blueprint = build_routine_blueprint(
        target_split_label="legs",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="advanced",
        readiness_level="normal",
        time_available_min=60,
        specialization_focus="adductorEmphasis",
    )

    focus_slot = next(
        slot
        for slot in blueprint.slots
        if slot.slot_id == blueprint.specialization_slot_id
    )
    assert focus_slot.slot_id == "specialization_focus"
    assert focus_slot.target_muscles == ("adductor",)
    assert focus_slot.requirement == "required_when_safe"


def test_one_training_method_is_applied_only_to_the_blueprint_method_slot():
    blueprint = build_routine_blueprint(
        target_split_label="chest",
        target_muscles=["chest"],
        goal="strength",
        experience_level="advanced",
        readiness_level="normal",
        time_available_min=60,
        training_method="pausedReps",
    )
    candidates = [
        _candidate(
            "flat_press",
            "바벨 플랫 체스트 프레스",
            "chest",
            equipment_req="BARBELL",
            bench_angle_band="flat",
        ),
        _candidate(
            "incline_press",
            "바벨 인클라인 체스트 프레스",
            "chest",
            equipment_req="BARBELL",
            bench_angle_band="inclineUnspecified",
        ),
    ]
    plans = [
        LLMExercisePlan(
            exercise_id=candidate["id"],
            exercise_name=candidate["name_kr"],
            movement_type="COMPOUND",
            primary_muscles=["chest"],
            target_reps=5,
            sets=3,
            rest_time_sec=120,
            exercise_rationale="가슴 운동",
        )
        for candidate in candidates
    ]
    output = LLMRoutineOutput(
        total_estimated_time=30,
        summary_title="가슴 근력",
        rationale_summary=["가슴 운동"],
        exercises=plans,
    )
    output._selection_slot_by_exercise_id = {
        "flat_press": "chest_main",
        "incline_press": "chest_secondary",
    }

    apply_blueprint_training_method(output, blueprint, candidates)

    assert output.exercises[0]._training_method_code == "pausedReps"
    assert output.exercises[1]._training_method_code is None


def test_specialization_focus_transfers_one_set_without_increasing_total_volume():
    blueprint = build_routine_blueprint(
        target_split_label="legs",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="advanced",
        readiness_level="normal",
        time_available_min=60,
        specialization_focus="hamstringEmphasis",
    )
    candidates = [
        _candidate(
            "leg_press",
            "레그 프레스",
            "quadriceps",
            equipment_req="MACHINE",
        ),
        _candidate(
            "leg_curl",
            "머신 레그 컬",
            "hamstring",
            movement_type="ISOLATION",
            equipment_req="MACHINE",
        ),
        _candidate(
            "calf_raise",
            "머신 카프 레이즈",
            "calves",
            movement_type="ISOLATION",
            equipment_req="MACHINE",
        ),
    ]
    plans = [
        LLMExercisePlan(
            exercise_id=candidate["id"],
            exercise_name=candidate["name_kr"],
            movement_type=candidate["movement_type"],
            primary_muscles=[candidate["primary_muscle"]],
            target_reps=10,
            sets=sets,
            rest_time_sec=90,
            exercise_rationale="기본 처방",
        )
        for candidate, sets in zip(candidates, (4, 3, 2))
    ]
    output = LLMRoutineOutput(
        total_estimated_time=30,
        summary_title="하체 근비대",
        rationale_summary=["기본 처방"],
        exercises=plans,
    )
    output._selection_slot_by_exercise_id = {
        "leg_press": "lower_main",
        "leg_curl": "posterior_chain",
        "calf_raise": "calves_accessory",
    }

    apply_blueprint_specialization_volume(output, blueprint, candidates)

    by_id = {exercise.exercise_id: exercise for exercise in output.exercises}
    assert sum(exercise.sets for exercise in output.exercises) == 9
    assert by_id["leg_curl"].sets == 4
    assert by_id["calf_raise"].sets == 2
    assert by_id["leg_press"].sets == 3
    assert "햄스트링 집중" in by_id["leg_curl"].exercise_rationale


def test_specialization_focus_prefers_non_focus_donor_for_visible_distribution_change():
    blueprint = build_routine_blueprint(
        target_split_label="arm",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="advanced",
        readiness_level="normal",
        time_available_min=60,
        specialization_focus="tricepsEmphasis",
    )
    candidates = [
        _candidate(
            "triceps_extension",
            "머신 푸시다운",
            "triceps",
            movement_type="ISOLATION",
            equipment_req="MACHINE",
        ),
        _candidate(
            "triceps_kickback",
            "덤벨 킥백",
            "triceps",
            movement_type="ISOLATION",
        ),
        _candidate(
            "biceps_curl",
            "머신 프리처 컬",
            "biceps",
            movement_type="ISOLATION",
            equipment_req="MACHINE",
        ),
    ]
    plans = [
        LLMExercisePlan(
            exercise_id=candidate["id"],
            exercise_name=candidate["name_kr"],
            movement_type=candidate["movement_type"],
            primary_muscles=[candidate["primary_muscle"]],
            target_reps=10,
            sets=sets,
            rest_time_sec=90,
            exercise_rationale="기본 처방",
        )
        for candidate, sets in zip(candidates, (4, 4, 4))
    ]
    output = LLMRoutineOutput(
        total_estimated_time=30,
        summary_title="팔 근비대",
        rationale_summary=["기본 처방"],
        exercises=plans,
    )
    output._selection_slot_by_exercise_id = {
        "triceps_extension": "triceps_direct",
        "triceps_kickback": "flex",
        "biceps_curl": "biceps_direct",
    }

    apply_blueprint_specialization_volume(output, blueprint, candidates)

    by_id = {exercise.exercise_id: exercise for exercise in output.exercises}
    triceps_sets = (
        by_id["triceps_extension"].sets + by_id["triceps_kickback"].sets
    )
    assert sum(exercise.sets for exercise in output.exercises) == 12
    assert triceps_sets == 9
    assert by_id["biceps_curl"].sets == 3


def test_specialization_volume_does_not_change_default_or_low_readiness_plan():
    candidates = [
        _candidate(
            "leg_press",
            "레그 프레스",
            "quadriceps",
            equipment_req="MACHINE",
        ),
        _candidate(
            "leg_curl",
            "머신 레그 컬",
            "hamstring",
            movement_type="ISOLATION",
            equipment_req="MACHINE",
        ),
    ]

    def output() -> LLMRoutineOutput:
        result = LLMRoutineOutput(
            total_estimated_time=24,
            summary_title="하체 근비대",
            rationale_summary=["기본 처방"],
            exercises=[
                LLMExercisePlan(
                    exercise_id="leg_press",
                    exercise_name="레그 프레스",
                    movement_type="COMPOUND",
                    primary_muscles=["quadriceps"],
                    target_reps=10,
                    sets=3,
                    rest_time_sec=90,
                    exercise_rationale="기본 처방",
                ),
                LLMExercisePlan(
                    exercise_id="leg_curl",
                    exercise_name="머신 레그 컬",
                    movement_type="ISOLATION",
                    primary_muscles=["hamstring"],
                    target_reps=10,
                    sets=2,
                    rest_time_sec=90,
                    exercise_rationale="기본 처방",
                ),
            ],
        )
        result._selection_slot_by_exercise_id = {
            "leg_press": "lower_main",
            "leg_curl": "posterior_chain",
        }
        return result

    default_blueprint = build_routine_blueprint(
        target_split_label="legs",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=60,
    )
    default_output = output()
    apply_blueprint_specialization_volume(
        default_output,
        default_blueprint,
        candidates,
    )
    assert [exercise.sets for exercise in default_output.exercises] == [3, 2]

    low_blueprint = build_routine_blueprint(
        target_split_label="legs",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="advanced",
        readiness_level="low",
        time_available_min=60,
        specialization_focus="hamstringEmphasis",
    )
    low_output = output()
    apply_blueprint_specialization_volume(low_output, low_blueprint, candidates)
    assert [exercise.sets for exercise in low_output.exercises] == [3, 2]


def test_public_response_separates_method_instruction_and_confirmation_suggestion():
    response = RoutineDraftResponse(
        generation_status="success",
        status_reason_code="none",
        is_fallback=False,
        total_estimated_time=30,
        summary_title="가슴 근력",
        rationale_summary=["기본운동에 수행법을 적용했습니다."],
        routine_blocks=[
            RoutineBlock(
                order=1,
                exercise_id="flat_press",
                exercise_name="바벨 플랫 체스트 프레스",
                movement_type="COMPOUND",
                primary_muscles=["chest"],
                default_rest_sec=120,
                prescription=[
                    SetPrescription(
                        set_index=1,
                        set_type="working",
                        target_reps=5,
                        target_rir=2,
                        target_rest_sec=120,
                    )
                ],
                exercise_rationale="가슴 근력 메인 운동입니다.",
                training_method=training_method_instruction("pausedReps"),
            )
        ],
        personalization_suggestions=[
            PersonalizationSuggestion(
                suggestion_type="specializationFocus",
                code="upperChest",
                reason_ko="최근 직접 세트가 적어 집중 여부를 확인합니다.",
                evidence_set_count=2,
                comparison_max_set_count=8,
                requires_confirmation=True,
            )
        ],
    )

    public = response.model_dump(by_alias=True, exclude_none=True)
    assert public["routineBlocks"][0]["exerciseName"] == "바벨 플랫 체스트 프레스"
    assert public["routineBlocks"][0]["trainingMethod"]["code"] == "pausedReps"
    assert public["personalizationSuggestions"][0]["requiresConfirmation"] is True


def test_recent_set_imbalance_suggests_one_focus_without_auto_applying_it():
    recent_sets = [
        *[
            RecentSetRecord(
                exercise_id=f"quad-{index}",
                exercise_name="레그 프레스",
                primary_muscle="quadriceps",
                reps=10,
            )
            for index in range(8)
        ],
        *[
            RecentSetRecord(
                exercise_id=f"ham-{index}",
                exercise_name="레그 컬",
                primary_muscle="hamstring",
                reps=10,
            )
            for index in range(2)
        ],
    ]

    suggestion = suggest_specialization_from_recent_sets(
        recent_sets,
        target_muscles=["quadriceps", "hamstring"],
        experience_level="advanced",
        explicit_focus=None,
    )

    assert suggestion is not None
    assert suggestion.focus_code == "hamstringEmphasis"
    assert suggestion.requires_confirmation is True


def test_history_suggestion_is_suppressed_when_user_already_selected_a_focus():
    suggestion = suggest_specialization_from_recent_sets(
        [
            RecentSetRecord(
                exercise_name="레그 프레스",
                primary_muscle="quadriceps",
                reps=10,
            )
            for _ in range(8)
        ],
        target_muscles=["quadriceps", "hamstring"],
        experience_level="advanced",
        explicit_focus="gluteEmphasis",
    )

    assert suggestion is None

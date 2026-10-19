from engines.option_resolver import (
    RoutineOptionCase,
    annotate_candidates_with_option_policy,
    candidate_matches_adjunct_option,
    candidate_semantically_matches_adjunct_option,
    derive_exercise_risk_profiles,
    filter_candidates_with_option_policy,
    infer_medical_clearance,
    infer_pain_region,
    resolve_candidate_option_policy,
    resolve_candidate_policy_for_request,
    resolve_routine_option_policy,
)
from engines.candidate_ranker import score_candidate_exercises
from engines.schemas import PainAreaEntry


def _codes(resolution):
    return {hit.root_reason_code for hit in resolution.blocked}


def test_adjunct_semantic_guard_rejects_broad_style_tag_false_positives():
    athletic_bench = {
        "id": "bench",
        "name_kr": "바벨 벤치프레스",
        "movement_type": "COMPOUND",
        "training_style_tags": "athletic,conditioning",
    }
    generic_band_press = {
        "id": "band_press",
        "name_kr": "스탠딩 밴드 프레스",
        "movement_type": "ISOLATION",
        "training_style_tags": "rehab_prehab",
    }

    assert not candidate_matches_adjunct_option(athletic_bench, "conditioning_finisher")
    assert not candidate_matches_adjunct_option(athletic_bench, "endurance_support")
    assert not candidate_matches_adjunct_option(generic_band_press, "prehab_accessory")


def test_adjunct_semantic_guard_accepts_curated_purpose_specific_candidates():
    sprint = {
        "id": "sprint",
        "name_kr": "러닝머신 스프린트 인터벌",
        "movement_type": "COMPOUND",
        "adjunct_option_codes": ["conditioning_finisher"],
    }
    face_pull = {
        "id": "face_pull",
        "name_kr": "케이블 페이스풀 외회전",
        "movement_type": "ISOLATION",
        "adjunct_option_codes": ["prehab_accessory"],
    }
    cooldown = {
        "id": "pec_stretch",
        "name_kr": "문틀 흉근 스트레치 - 기본",
        "movement_type": "ISOLATION",
        "adjunct_option_codes": ["cooldown_stretch"],
    }

    assert candidate_matches_adjunct_option(sprint, "conditioning_finisher")
    assert candidate_matches_adjunct_option(face_pull, "prehab_accessory")
    assert candidate_matches_adjunct_option(cooldown, "cooldown_stretch")


def test_cooldown_semantic_guard_rejects_dynamic_stretch_variant():
    dynamic_stretch = {
        "name_kr": "문틀 흉근 스트레치 - 동적 반복",
        "movement_type": "ISOLATION",
    }

    assert not candidate_semantically_matches_adjunct_option(
        dynamic_stretch,
        "cooldown_stretch",
    )


def test_medical_not_cleared_blocks_loaded_or_complex_work():
    result = resolve_routine_option_policy(
        RoutineOptionCase(
            medical_clearance="not_cleared",
            exercise_risk_profile="high_lumbar_load",
            application_phase="main_lift",
        )
    )
    assert not result.allowed
    assert result.root_reason_code == "medical_not_cleared_loaded_or_complex_block"


def test_medical_unknown_with_active_pain_blocks_loaded_or_complex_work():
    result = resolve_routine_option_policy(
        RoutineOptionCase(
            medical_clearance="unknown",
            pain_status="active_pain",
            pain_region="lower_back",
            exercise_risk_profile="high_lumbar_load",
            application_phase="accessory",
        )
    )
    assert "medical_unknown_active_pain_loaded_or_complex_block" in _codes(result)


def test_acute_direct_region_pain_blocks_loaded_phase():
    result = resolve_routine_option_policy(
        RoutineOptionCase(
            pain_status="acute_pain",
            pain_region="shoulder",
            exercise_risk_profile="overhead_shoulder",
            application_phase="finisher",
        )
    )
    assert "acute_direct_region_loaded_phase_block" in _codes(result)


def test_active_direct_region_pain_blocks_main_lift():
    result = resolve_routine_option_policy(
        RoutineOptionCase(
            pain_status="active_pain",
            pain_region="knee",
            exercise_risk_profile="high_knee_shear",
            application_phase="main_lift",
        )
    )
    assert "active_direct_region_main_lift_block" in _codes(result)


def test_lower_back_pain_blocks_heavy_spinal_loading():
    result = resolve_routine_option_policy(
        RoutineOptionCase(
            pain_status="active_pain",
            pain_region="lower_back",
            intensity_intent="heavy",
            exercise_risk_profile="high_axial_load",
            application_phase="main_lift",
        )
    )
    assert "lower_back_heavy_spinal_loading_block" in _codes(result)


def test_shoulder_pain_blocks_overhead_loaded_work():
    result = resolve_routine_option_policy(
        RoutineOptionCase(
            pain_status="active_pain",
            pain_region="shoulder",
            exercise_risk_profile="overhead_shoulder",
            application_phase="accessory",
        )
    )
    assert "shoulder_overhead_loaded_block" in _codes(result)


def test_knee_pain_blocks_high_shear_main_lift():
    result = resolve_routine_option_policy(
        RoutineOptionCase(
            pain_status="acute_pain",
            pain_region="knee",
            exercise_risk_profile="high_knee_shear",
            application_phase="main_lift",
        )
    )
    assert "knee_high_shear_main_lift_block" in _codes(result)


def test_wrist_pain_blocks_loaded_wrist_stress():
    result = resolve_routine_option_policy(
        RoutineOptionCase(
            pain_status="active_pain",
            pain_region="wrist",
            exercise_risk_profile="high_wrist_stress",
            application_phase="accessory",
        )
    )
    assert "wrist_loaded_stress_block" in _codes(result)


def test_ankle_pain_blocks_high_demand_main_lift():
    result = resolve_routine_option_policy(
        RoutineOptionCase(
            pain_status="active_pain",
            pain_region="ankle",
            exercise_risk_profile="high_ankle_demand",
            application_phase="main_lift",
        )
    )
    assert "ankle_high_demand_main_lift_block" in _codes(result)


def test_beginner_blocks_ballistic_complex_main_or_density():
    result = resolve_routine_option_policy(
        RoutineOptionCase(
            experience_level="beginner",
            exercise_risk_profile="ballistic_complex",
            application_phase="main_lift",
        )
    )
    assert "beginner_ballistic_complex_block" in _codes(result)


def test_low_readiness_blocks_general_high_density_session():
    result = resolve_routine_option_policy(
        RoutineOptionCase(
            readiness_level="low",
            session_structure="circuit",
            application_phase="warmup",
            exercise_risk_profile="normal",
        )
    )
    assert "low_readiness_high_density_block" in _codes(result)
    assert result.root_reason_code == "low_readiness_high_density_block"


def test_low_readiness_blocks_conditioning_adjunct_option():
    result = resolve_routine_option_policy(
        RoutineOptionCase(
            readiness_level="low",
            session_structure="standard",
            adjunct_options=("conditioning_finisher",),
            application_phase="warmup",
            exercise_risk_profile="normal",
        )
    )

    assert "low_readiness_high_density_block" in _codes(result)
    assert result.root_reason_code == "low_readiness_high_density_block"
    assert "low_readiness_blocks_conditioning_or_endurance_adjunct" in result.evidence_codes


def test_candidate_policy_for_request_blocks_low_readiness_endurance_adjunct():
    result = resolve_candidate_policy_for_request(
        {"id": "safe_bodyweight", "name_kr": "가벼운 보조 운동"},
        readiness_level="low",
        session_structure="standard",
        adjunct_options=["endurance_support"],
        application_phase="warmup",
    )

    assert not result.allowed
    assert result.root_reason_code == "low_readiness_high_density_block"


def test_low_readiness_blocks_high_density_loaded_phase():
    result = resolve_routine_option_policy(
        RoutineOptionCase(
            readiness_level="low",
            session_structure="density",
            application_phase="finisher",
            exercise_risk_profile="normal",
        )
    )
    assert "low_readiness_high_density_loaded_phase_block" in _codes(result)


def test_strength_blocks_general_circuit_session():
    result = resolve_routine_option_policy(
        RoutineOptionCase(
            goal="strength",
            session_structure="circuit",
            application_phase="warmup",
            exercise_risk_profile="normal",
        )
    )
    assert "strength_circuit_block" in _codes(result)
    assert result.root_reason_code == "strength_circuit_block"


def test_strength_blocks_high_density_main_lift():
    result = resolve_routine_option_policy(
        RoutineOptionCase(
            goal="strength",
            intensity_intent="high_density",
            application_phase="main_lift",
            exercise_risk_profile="normal",
        )
    )
    assert "strength_high_density_main_lift_block" in _codes(result)


def test_strength_density_session_preserves_single_main_lift():
    result = resolve_routine_option_policy(
        RoutineOptionCase(
            goal="strength",
            session_structure="density",
            intensity_intent="heavy",
            application_phase="main_lift",
            exercise_risk_profile="normal",
        )
    )

    assert result.allowed
    assert "strength_high_density_main_lift_block" not in _codes(result)


def test_recorded_injury_blocks_direct_region_work_in_every_phase():
    result = resolve_routine_option_policy(
        RoutineOptionCase(
            pain_status="recorded_injury",
            pain_region="lower_back",
            exercise_risk_profile="high_lumbar_load",
            application_phase="cooldown",
        )
    )
    assert not result.allowed
    assert result.severity == "block"
    assert result.root_reason_code == "recorded_injury_direct_region_exercise_block"


def test_medical_not_cleared_wins_over_lower_priority_reasons():
    result = resolve_routine_option_policy(
        RoutineOptionCase(
            medical_clearance="not_cleared",
            pain_status="acute_pain",
            pain_region="shoulder",
            exercise_risk_profile="overhead_shoulder",
            application_phase="main_lift",
        )
    )
    assert {
        "medical_not_cleared_loaded_or_complex_block",
        "acute_direct_region_loaded_phase_block",
        "shoulder_overhead_loaded_block",
    } <= _codes(result)
    assert result.root_reason_code == "medical_not_cleared_loaded_or_complex_block"


def test_active_direct_region_wins_over_body_region_specific_reason_but_preserves_evidence():
    result = resolve_routine_option_policy(
        RoutineOptionCase(
            pain_status="active_pain",
            pain_region="knee",
            exercise_risk_profile="high_knee_shear",
            application_phase="main_lift",
        )
    )

    assert {
        "active_direct_region_main_lift_block",
        "knee_high_shear_main_lift_block",
    } <= _codes(result)
    assert result.root_reason_code == "active_direct_region_main_lift_block"
    assert "active_direct_region_pain_blocks_main_lift" in result.evidence_codes
    assert "knee_pain_blocks_high_shear_main_lift" in result.evidence_codes


def test_low_readiness_high_density_wins_over_strength_high_density_but_preserves_evidence():
    result = resolve_routine_option_policy(
        RoutineOptionCase(
            goal="strength",
            readiness_level="low",
            session_structure="circuit",
            intensity_intent="high_density",
            exercise_risk_profile="normal",
            application_phase="main_lift",
        )
    )

    assert {
        "low_readiness_high_density_block",
        "low_readiness_high_density_loaded_phase_block",
        "strength_high_density_main_lift_block",
    } <= _codes(result)
    assert result.root_reason_code == "low_readiness_high_density_block"
    assert "low_readiness_blocks_high_density_session" in result.evidence_codes
    assert "low_readiness_blocks_high_density_loaded_phase" in result.evidence_codes
    assert "strength_blocks_high_density_main_lift" in result.evidence_codes


def test_derive_exercise_risk_profiles_from_db_fields():
    profiles = derive_exercise_risk_profiles(
        {
            "name_kr": "바벨 스내치",
            "lumbar_load": "high",
            "axial_load": "low",
            "knee_shear_load": "high",
            "shoulder_impingement_risk": "medium",
            "wrist_stress": "high",
            "ankle_dorsiflexion_demand": "high",
            "overhead_position_required": 1,
        }
    )
    assert {
        "high_lumbar_load",
        "high_knee_shear",
        "high_wrist_stress",
        "high_ankle_demand",
        "overhead_shoulder",
        "ballistic_complex",
    } <= profiles


def test_resolve_candidate_option_policy_uses_derived_risk_profile():
    result = resolve_candidate_option_policy(
        {"name_kr": "바벨 스내치", "technical_difficulty": "high"},
        experience_level="beginner",
        application_phase="main_lift",
    )
    assert not result.allowed
    assert result.root_reason_code == "beginner_ballistic_complex_block"


def test_infer_medical_clearance_from_condition_policy():
    assert infer_medical_clearance([]) == "not_needed"
    assert infer_medical_clearance([{"professionalClearance": True}]) == "cleared"
    assert infer_medical_clearance([{"status": "needs_clearance"}]) == "not_cleared"
    assert infer_medical_clearance([{"conditionCode": "lumbar_herniation"}]) == "unknown"
    assert infer_medical_clearance([{"clearanceStatus": "notNeeded"}]) == "not_needed"
    assert infer_medical_clearance([{"clearanceStatus": "notCleared"}]) == "not_cleared"
    assert infer_medical_clearance([{"clearanceStatus": "unknown"}]) == "unknown"
    assert infer_medical_clearance([{"clearanceStatus": "clearedWithRestrictions"}]) == "cleared_with_restrictions"
    assert infer_medical_clearance([{"professionalClearance": "clearedWithRestrictions"}]) == "cleared_with_restrictions"


def test_restricted_clearance_warns_on_loaded_candidate():
    result = resolve_candidate_policy_for_request(
        {"name_kr": "바벨 백 스쿼트", "axial_load": "high"},
        condition_policies=[{"clearanceStatus": "clearedWithRestrictions"}],
        application_phase="main_lift",
    )

    assert result.allowed
    assert result.severity == "warn"
    assert result.root_reason_code == "medical_restricted_requires_caution_copy"


def test_infer_injury_region_from_public_values():
    assert infer_pain_region("front-deltoids") == "shoulder"
    assert infer_pain_region("knees") == "knee"
    assert infer_pain_region("lower-back") == "lower_back"


def test_resolve_candidate_policy_for_request_blocks_registered_lower_back_injury():
    result = resolve_candidate_policy_for_request(
        {"name_kr": "바벨 데드리프트", "lumbar_load": "high"},
        pain_areas=[PainAreaEntry(body_part="lower-back")],
        application_phase="main_lift",
    )
    assert not result.allowed
    assert result.severity == "block"
    assert result.root_reason_code == "recorded_injury_direct_region_exercise_block"


def test_filter_candidates_with_option_policy_splits_allowed_and_blocked():
    candidates = [
        {"id": "deadlift", "name_kr": "바벨 데드리프트", "lumbar_load": "high"},
        {"id": "leg_curl", "name_kr": "시티드 레그 컬", "lumbar_load": "low"},
    ]
    allowed, blocked = filter_candidates_with_option_policy(
        candidates,
        pain_areas=[{"bodyPart": "lower-back"}],
        application_phase="main_lift",
    )
    assert [item["id"] for item in allowed] == ["leg_curl"]
    assert [item["id"] for item in blocked] == ["deadlift"]
    assert blocked[0]["option_policy_root_reason_code"] == "recorded_injury_direct_region_exercise_block"


def test_annotate_candidates_with_option_policy_preserves_recorded_injury_block_evidence():
    annotated = annotate_candidates_with_option_policy(
        [{"id": "deadlift", "name_kr": "바벨 데드리프트", "lumbar_load": "high"}],
        pain_areas=[{"bodyPart": "lower-back"}],
        application_phase="main_lift",
    )
    assert annotated[0]["option_policy_allowed"] is False
    assert annotated[0]["option_policy_severity"] == "block"
    assert "high_lumbar_load" in annotated[0]["exercise_risk_profiles"]


def test_candidate_ranker_hard_blocks_not_cleared_high_risk_candidate():
    candidates = [
        {
            "id": "barbell_good_morning",
            "name_kr": "바벨 굿모닝",
            "primary_muscle": "hamstring",
            "secondary_muscle": "gluteal",
            "equipment_req": "BARBELL",
            "efficiency_tier": 3,
            "movement_type": "COMPOUND",
            "lumbar_load": "high",
            "failure_penalty": "high",
        },
        {
            "id": "seated_leg_curl",
            "name_kr": "시티드 레그 컬",
            "primary_muscle": "hamstring",
            "equipment_req": "MACHINE",
            "efficiency_tier": 3,
            "movement_type": "ISOLATION",
            "lumbar_load": "low",
            "failure_penalty": "low",
        },
    ]

    ranked = score_candidate_exercises(
        candidates,
        ["hamstring"],
        condition_policies=[{"status": "needs_clearance", "professionalClearance": False}],
        top_n=5,
    )

    assert [candidate["id"] for candidate in ranked] == ["seated_leg_curl"]


def test_candidate_ranker_hard_excludes_recorded_lower_back_injury_candidate():
    candidates = [
        {
            "id": "barbell_good_morning",
            "name_kr": "바벨 굿모닝",
            "primary_muscle": "hamstring",
            "secondary_muscle": "gluteal",
            "equipment_req": "BARBELL",
            "efficiency_tier": 3,
            "movement_type": "COMPOUND",
            "lumbar_load": "high",
            "failure_penalty": "medium",
        },
        {
            "id": "seated_leg_curl",
            "name_kr": "시티드 레그 컬",
            "primary_muscle": "hamstring",
            "equipment_req": "MACHINE",
            "efficiency_tier": 3,
            "movement_type": "ISOLATION",
            "lumbar_load": "low",
            "failure_penalty": "low",
        },
    ]

    ranked = score_candidate_exercises(
        candidates,
        ["hamstring"],
        pain_areas=[PainAreaEntry(body_part="lower-back")],
        top_n=5,
    )

    assert [candidate["id"] for candidate in ranked] == ["seated_leg_curl"]

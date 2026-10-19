import pytest

from engines.candidate_ranker import score_candidate_exercises
from engines.profile_constraint_contract import (
    PROFILE_INPUT_CONTRACT_VERSION,
    build_profile_input_manifest,
    resolve_profile_constraint_signals,
)
from engines.schemas import RoutineRequest


def _candidate(exercise_id: str, primary: str, **overrides):
    candidate = {
        "id": exercise_id,
        "name_kr": exercise_id,
        "name_en": exercise_id,
        "primary_muscle": primary,
        "secondary_muscle": "",
        "equipment_req": "DUMBBELL",
        "efficiency_tier": 3,
        "movement_type": "COMPOUND",
        "pain_triggers": "",
        "technical_difficulty": "low",
        "balance_requirement": "low",
        "failure_penalty": "low",
        "injury_caution_level": "low",
        "hypertrophy_effect": "high",
        "strength_effect": "high",
        "stimulus_to_fatigue": "good",
        "ankle_dorsiflexion_demand": "low",
        "hip_flexion_demand": "low",
        "shoulder_flexion_demand": "low",
        "wrist_extension_demand": "low",
        "long_arm_sensitivity": "low",
        "long_femur_sensitivity": "low",
    }
    candidate.update(overrides)
    return candidate


def test_empty_profile_input_is_stable_and_has_no_implicit_signal():
    resolved = resolve_profile_constraint_signals(None)

    assert resolved.to_dict() == {
        "selected_signals": [],
        "movement_signals": [],
        "body_proportion_signals": [],
        "mobility_limit_aliases": [],
        "anthropometry_signal_aliases": [],
    }


def test_profile_input_is_deduplicated_and_sorted_by_contract_order():
    resolved = resolve_profile_constraint_signals([
        "self_reported_long_arms",
        "overhead_reach_difficult",
        "heel_down_deep_squat_difficult",
        "overhead_reach_difficult",
    ])

    assert resolved.selected_signals == (
        "heel_down_deep_squat_difficult",
        "overhead_reach_difficult",
        "self_reported_long_arms",
    )
    assert resolved.mobility_limit_aliases == (
        "limited_ankle_dorsiflexion",
        "limited_shoulder_flexion",
    )
    assert resolved.anthropometry_signal_aliases == ("long_arm",)


@pytest.mark.parametrize("value", ["overhead_reach_difficult", {"code": "overhead_reach_difficult"}, ["unknown"]])
def test_profile_input_rejects_unregistered_or_non_array_values(value):
    with pytest.raises(ValueError):
        resolve_profile_constraint_signals(value)


def test_manifest_is_user_facing_default_off_and_routes_movement_pain():
    manifest = build_profile_input_manifest()
    options = [option for section in manifest["sections"] for option in section["options"]]
    movement = manifest["sections"][0]["options"]
    body = manifest["sections"][1]["options"]

    assert manifest["version"] == PROFILE_INPUT_CONTRACT_VERSION
    assert manifest["defaultBehavior"] == "allUnchecked"
    assert len(options) == 6
    assert all(option["defaultSelected"] is False for option in options)
    assert all(option["painDuringTaskAction"] == "routeToRegisteredInjury" for option in movement)
    assert all(option["decisionAuthority"] == "advisoryOnly" for option in body)
    assert manifest["priorityOrder"][:3] == [
        "registeredInjury",
        "functionalMovement",
        "selfReportedAnthropometry",
    ]


def test_routine_request_maps_profile_signals_to_stable_internal_aliases():
    request = RoutineRequest.model_validate({
        "timeAvailableMin": 60,
        "profileSignals": [
            "overhead_reach_difficult",
            "self_reported_long_arms",
            "overhead_reach_difficult",
        ],
    })

    assert request.profile_signals == [
        "overhead_reach_difficult",
        "self_reported_long_arms",
    ]
    assert request.mobility_limits == ["limited_shoulder_flexion"]
    assert request.anthropometry_signals == {"signals": ["long_arm"]}


def test_routine_request_merges_profile_signals_with_legacy_internal_values():
    request = RoutineRequest.model_validate({
        "timeAvailableMin": 60,
        "profileSignals": [
            "heel_down_deep_squat_difficult",
            "self_reported_long_thighs",
        ],
        "mobilityLimits": ["limited_wrist_extension"],
        "anthropometrySignals": {"signals": ["measured_long_arm"]},
    })

    assert request.mobility_limits == [
        "limited_wrist_extension",
        "limited_ankle_dorsiflexion",
    ]
    assert request.anthropometry_signals == {
        "signals": ["measured_long_arm", "long_femur"],
    }


def test_routine_request_default_profile_input_has_no_implicit_effect():
    request = RoutineRequest(time_available_min=60)

    assert request.profile_signals == []
    assert request.mobility_limits == []
    assert request.anthropometry_signals == {}


@pytest.mark.parametrize(
    "profile_signals",
    ["overhead_reach_difficult", ["not_registered"]],
)
def test_routine_request_rejects_invalid_profile_signal_payload(profile_signals):
    with pytest.raises(ValueError):
        RoutineRequest.model_validate({
            "timeAvailableMin": 60,
            "profileSignals": profile_signals,
        })


def test_routine_request_rejects_invalid_legacy_anthropometry_signal_shape():
    with pytest.raises(ValueError, match="anthropometrySignals.signals"):
        RoutineRequest.model_validate({
            "timeAvailableMin": 60,
            "anthropometrySignals": {"signals": "long_arm"},
        })


@pytest.mark.parametrize(
    ("profile_signal", "internal_field", "primary"),
    [
        ("heel_down_deep_squat_difficult", "ankle_dorsiflexion_demand", "quadriceps"),
        ("hip_hinge_difficult", "hip_flexion_demand", "hamstring"),
        ("overhead_reach_difficult", "shoulder_flexion_demand", "front-deltoids"),
        ("loaded_wrist_extension_difficult", "wrist_extension_demand", "chest"),
    ],
)
def test_each_movement_profile_signal_penalizes_only_the_high_demand_candidate(
    profile_signal,
    internal_field,
    primary,
):
    request = RoutineRequest.model_validate({
        "timeAvailableMin": 60,
        "profileSignals": [profile_signal],
    })
    candidates = [
        _candidate("high_demand", primary, **{internal_field: "high"}),
        _candidate("low_demand", primary, **{internal_field: "low"}),
    ]
    baseline = score_candidate_exercises(candidates, [primary], top_n=2)
    ranked = score_candidate_exercises(
        candidates,
        [primary],
        mobility_limits=request.mobility_limits,
        top_n=2,
    )
    baseline_scores = {item["id"]: item["score"] for item in baseline}
    ranked_scores = {item["id"]: item["score"] for item in ranked}

    assert ranked_scores["high_demand"] == baseline_scores["high_demand"] - 20
    assert ranked_scores["low_demand"] == baseline_scores["low_demand"]
    assert ranked[0]["id"] == "low_demand"


def test_overhead_reach_difficulty_excludes_only_explicit_high_overhead_v2_candidate():
    request = RoutineRequest.model_validate({
        "timeAvailableMin": 60,
        "profileSignals": ["overhead_reach_difficult"],
    })
    candidates = [
        _candidate(
            "overhead_press",
            "front-deltoids",
            shoulder_elevation_demand="high",
            overhead_position_required=1,
            bench_angle_band="overhead",
        ),
        _candidate(
            "flat_independent_press",
            "chest",
            shoulder_elevation_demand="low",
            overhead_position_required=0,
            bench_angle_band="flat",
        ),
    ]

    ranked = score_candidate_exercises(
        candidates,
        ["front-deltoids", "chest"],
        mobility_limits=request.mobility_limits,
        top_n=2,
    )

    assert [item["id"] for item in ranked] == ["flat_independent_press"]


def test_heel_down_difficulty_excludes_explicit_deep_squat_but_keeps_shallow_alternative():
    request = RoutineRequest.model_validate({
        "timeAvailableMin": 60,
        "profileSignals": ["heel_down_deep_squat_difficult"],
    })
    candidates = [
        _candidate(
            "full_depth_squat",
            "quadriceps",
            deep_squat_position_required=1,
            ankle_dorsiflexion_demand="medium",
        ),
        _candidate(
            "supported_shallow_squat",
            "quadriceps",
            deep_squat_position_required=0,
            ankle_dorsiflexion_demand="medium",
        ),
    ]

    ranked = score_candidate_exercises(
        candidates,
        ["quadriceps"],
        mobility_limits=request.mobility_limits,
        top_n=2,
    )

    assert [item["id"] for item in ranked] == ["supported_shallow_squat"]


def test_heel_down_difficulty_recognizes_full_squat_name_when_metadata_is_stale():
    request = RoutineRequest.model_validate({
        "timeAvailableMin": 60,
        "profileSignals": ["heel_down_deep_squat_difficult"],
    })
    candidates = [
        _candidate(
            "stale_full_squat",
            "quadriceps",
            name_kr="스미스 머신 어깨너비 풀 백 스쿼트",
            deep_squat_position_required=0,
            ankle_dorsiflexion_demand="medium",
        ),
        _candidate(
            "leg_press",
            "quadriceps",
            name_kr="머신 레그 프레스",
            ankle_dorsiflexion_demand="medium",
        ),
    ]

    ranked = score_candidate_exercises(
        candidates,
        ["quadriceps"],
        mobility_limits=request.mobility_limits,
        top_n=2,
    )

    assert [item["id"] for item in ranked] == ["leg_press"]


def test_overhead_reach_difficulty_excludes_incline_candidate_and_keeps_flat_press():
    request = RoutineRequest.model_validate({
        "timeAvailableMin": 60,
        "profileSignals": ["overhead_reach_difficult"],
    })
    candidates = [
        _candidate(
            "incline_press",
            "chest",
            shoulder_elevation_demand="medium",
            overhead_position_required=0,
            bench_angle_band="inclineUnspecified",
        ),
        _candidate(
            "flat_press",
            "chest",
            shoulder_elevation_demand="low",
            overhead_position_required=0,
            bench_angle_band="flat",
        ),
    ]

    ranked = score_candidate_exercises(
        candidates,
        ["chest"],
        mobility_limits=request.mobility_limits,
        top_n=2,
    )

    assert [item["id"] for item in ranked] == ["flat_press"]


def test_overhead_reach_difficulty_excludes_sagittal_front_raise_not_lateral_raise():
    request = RoutineRequest.model_validate({
        "timeAvailableMin": 60,
        "profileSignals": ["overhead_reach_difficult"],
    })
    candidates = [
        _candidate(
            "front_raise",
            "front-deltoids",
            shoulder_elevation_demand="medium",
            shoulder_elevation_plane="sagittal",
        ),
        _candidate(
            "lateral_raise",
            "front-deltoids",
            shoulder_elevation_demand="medium",
            shoulder_elevation_plane="frontal",
        ),
    ]

    ranked = score_candidate_exercises(
        candidates,
        ["front-deltoids"],
        mobility_limits=request.mobility_limits,
        top_n=2,
    )

    assert [item["id"] for item in ranked] == ["lateral_raise"]


def test_overhead_reach_difficulty_excludes_overhead_name_when_metadata_is_stale():
    request = RoutineRequest.model_validate({
        "timeAvailableMin": 60,
        "profileSignals": ["overhead_reach_difficult"],
    })
    candidates = [
        _candidate(
            "stale_overhead_extension",
            "triceps",
            name_kr="덤벨 오버헤드 익스텐션",
            shoulder_elevation_demand="low",
            overhead_position_required=0,
        ),
        _candidate(
            "cable_pushdown",
            "triceps",
            name_kr="케이블 푸시다운",
            shoulder_elevation_demand="low",
            overhead_position_required=0,
        ),
    ]

    ranked = score_candidate_exercises(
        candidates,
        ["triceps"],
        mobility_limits=request.mobility_limits,
        top_n=2,
    )

    assert [item["id"] for item in ranked] == ["cable_pushdown"]


def test_overhead_reach_setup_preference_uses_only_known_independent_high_freedom_data():
    request = RoutineRequest.model_validate({
        "timeAvailableMin": 60,
        "profileSignals": ["overhead_reach_difficult"],
    })
    known = _candidate(
        "known_independent",
        "chest",
        shoulder_elevation_demand="low",
        path_type="free",
        bilateral_coupling="independent",
        grip_freedom="high",
    )
    unknown = _candidate(
        "unknown_machine",
        "chest",
        shoulder_elevation_demand="low",
        path_type="machineDependent",
        bilateral_coupling="unknown",
        grip_freedom="unknown",
        seat_adjustability="unknown",
        start_position_adjustability="unknown",
    )

    ranked = score_candidate_exercises(
        [unknown, known],
        ["chest"],
        mobility_limits=request.mobility_limits,
        top_n=2,
    )
    scores = {item["id"]: item["score"] for item in ranked}

    assert scores["known_independent"] == scores["unknown_machine"] + 4
    assert not any(
        "adjustability" in reason
        for item in ranked
        for reason in item["score_reasons"]
    )


@pytest.mark.parametrize(
    ("profile_signal", "sensitivity_field", "primary", "expected_delta"),
    [
        ("self_reported_long_arms", "long_arm_sensitivity", "chest", -10),
        ("self_reported_long_thighs", "long_femur_sensitivity", "quadriceps", -14),
    ],
)
def test_reviewed_body_proportion_signals_apply_only_advisory_ranking_penalties(
    profile_signal,
    sensitivity_field,
    primary,
    expected_delta,
):
    request = RoutineRequest.model_validate({
        "timeAvailableMin": 60,
        "profileSignals": [profile_signal],
    })
    candidates = [
        _candidate("high_sensitivity", primary, **{sensitivity_field: "high"}),
        _candidate("low_sensitivity", primary, **{sensitivity_field: "low"}),
    ]
    baseline = score_candidate_exercises(candidates, [primary], top_n=2)
    ranked = score_candidate_exercises(
        candidates,
        [primary],
        anthropometry_signals=request.anthropometry_signals,
        top_n=2,
    )
    baseline_scores = {item["id"]: item["score"] for item in baseline}
    ranked_scores = {item["id"]: item["score"] for item in ranked}

    assert len(ranked) == 2
    assert ranked_scores["high_sensitivity"] == baseline_scores["high_sensitivity"] + expected_delta
    assert ranked_scores["low_sensitivity"] == baseline_scores["low_sensitivity"]


def test_unreviewed_long_lower_leg_signal_is_rejected_by_v3_contract():
    with pytest.raises(ValueError, match="unknown profile constraint signals"):
        RoutineRequest.model_validate({
            "timeAvailableMin": 60,
            "profileSignals": ["self_reported_long_lower_legs"],
        })


def test_registered_injury_exclusion_has_priority_over_profile_signal_ranking():
    request = RoutineRequest.model_validate({
        "timeAvailableMin": 60,
        "profileSignals": ["overhead_reach_difficult"],
        "painAreas": [{"area": "front-deltoids"}],
    })
    candidates = [
        _candidate(
            "injured_area_candidate",
            "front-deltoids",
            shoulder_flexion_demand="low",
        ),
        _candidate(
            "unrelated_safe_candidate",
            "quadriceps",
            shoulder_flexion_demand="low",
        ),
    ]

    ranked = score_candidate_exercises(
        candidates,
        ["front-deltoids", "quadriceps"],
        pain_areas=request.pain_areas,
        mobility_limits=request.mobility_limits,
        top_n=2,
    )

    assert [item["id"] for item in ranked] == ["unrelated_safe_candidate"]

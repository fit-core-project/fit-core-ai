from unittest.mock import patch

import pytest
from pydantic import ValidationError

from engines.candidate_ranker import score_candidate_exercises
from engines.routine_pipeline import _merge_profile_injury_areas, generate_smart_routine
from engines.schemas import PainAreaEntry, RoutineRequest, UserProfileContext
from engines.shoulder_condition_policy import (
    active_uncleared_requires_safe_stop,
    candidate_has_shoulder_load,
    candidate_restricted_movement_overlap,
    candidate_shoulder_movement_patterns,
    resolve_shoulder_condition_context,
)


def _candidate(
    exercise_id: str,
    name: str,
    primary: str,
    *,
    secondary: str = "",
    movement_type: str = "COMPOUND",
    plane: str = "unknown",
    overhead: int = 0,
) -> dict:
    return {
        "id": exercise_id,
        "name_kr": name,
        "name_en": name,
        "primary_muscle": primary,
        "secondary_muscle": secondary,
        "equipment_req": "DUMBBELL",
        "difficulty_tier": 3,
        "efficiency_tier": 3,
        "pain_triggers": None,
        "movement_type": movement_type,
        "shoulder_elevation_plane": plane,
        "shoulder_elevation_demand": "medium" if plane != "unknown" or overhead else "low",
        "overhead_position_required": overhead,
        "shoulder_impingement_risk": "low",
    }


def test_internal_contract_accepts_closed_managed_shoulder_restrictions():
    request = RoutineRequest.model_validate({
        "targetSplitLabel": "push",
        "timeAvailableMin": 45,
        "shoulderConditionStatus": "managedWithRestrictions",
        "restrictedMovementPatterns": ["horizontalPush", "armRaise"],
    })

    assert request.shoulder_condition_status == "managedWithRestrictions"
    assert request.restricted_movement_patterns == ["horizontalPush", "armRaise"]


@pytest.mark.parametrize(
    "payload",
    [
        {
            "shoulderConditionStatus": "managedWithRestrictions",
            "restrictedMovementPatterns": [],
        },
        {
            "shoulderConditionStatus": "managedNoRestrictions",
            "restrictedMovementPatterns": ["horizontalPush"],
        },
        {
            "shoulderConditionStatus": "managedWithRestrictions",
            "restrictedMovementPatterns": ["unknown", "horizontalPush"],
        },
        {
            "shoulderConditionStatus": "managedWithRestrictions",
            "restrictedMovementPatterns": ["armRaise", "armRaise"],
        },
    ],
)
def test_invalid_shoulder_condition_combinations_are_rejected(payload):
    with pytest.raises(ValidationError):
        RoutineRequest.model_validate({
            "targetSplitLabel": "push",
            "timeAvailableMin": 45,
            **payload,
        })


def test_profile_contract_accepts_and_validates_closed_shoulder_state():
    profile = UserProfileContext.model_validate({
        "goalType": "hypertrophy",
        "splitType": "PPL",
        "shoulderConditionStatus": "managedWithRestrictions",
        "restrictedMovementPatterns": ["overheadPosition", "armRaise"],
        "painAreas": [{"bodyPart": "front-deltoids"}],
    })

    assert profile.shoulder_condition_status == "managedWithRestrictions"
    assert profile.restricted_movement_patterns == ["overheadPosition", "armRaise"]
    assert [area.body_part for area in profile.pain_areas] == ["front-deltoids"]

    with pytest.raises(ValidationError):
        UserProfileContext(
            goal_type="hypertrophy",
            split_type="PPL",
            shoulder_condition_status="recovered",
            restricted_movement_patterns=["overheadPosition"],
        )


def test_profile_binary_injury_areas_are_unioned_with_request_areas():
    merged = _merge_profile_injury_areas(
        [
            PainAreaEntry(body_part="front-deltoids"),
            PainAreaEntry(body_part="knees", side="left"),
        ],
        [
            PainAreaEntry(body_part="front-deltoids"),
            PainAreaEntry(body_part="knees", side="right"),
        ],
    )

    assert [(entry.body_part, entry.side) for entry in merged] == [
        ("front-deltoids", None),
        ("knees", "right"),
        ("knees", "left"),
    ]


@pytest.mark.parametrize(
    (
        "request_status",
        "request_patterns",
        "profile_status",
        "profile_patterns",
        "expected_status",
        "expected_patterns",
    ),
    [
        (
            None,
            [],
            "managedWithRestrictions",
            ["horizontalPush"],
            "managedWithRestrictions",
            ("horizontalPush",),
        ),
        (
            "managedNoRestrictions",
            [],
            "activeUncleared",
            [],
            "activeUncleared",
            (),
        ),
        (
            "managedWithRestrictions",
            ["horizontalPush"],
            "managedWithRestrictions",
            ["armRaise"],
            "managedWithRestrictions",
            ("horizontalPush", "armRaise"),
        ),
        (
            "managedWithRestrictions",
            ["unknown"],
            "managedWithRestrictions",
            ["horizontalPull"],
            "managedWithRestrictions",
            ("unknown",),
        ),
        (
            "recovered",
            [],
            "managedNoRestrictions",
            [],
            "managedNoRestrictions",
            (),
        ),
    ],
)
def test_request_and_profile_shoulder_state_merge_is_safety_monotonic(
    request_status,
    request_patterns,
    profile_status,
    profile_patterns,
    expected_status,
    expected_patterns,
):
    assert resolve_shoulder_condition_context(
        request_status,
        request_patterns,
        profile_status,
        profile_patterns,
    ) == (expected_status, expected_patterns)


def test_all_valid_request_profile_shoulder_context_pairs_preserve_safety():
    valid_contexts = [
        (None, ()),
        ("activeUncleared", ()),
        ("managedNoRestrictions", ()),
        ("recovered", ()),
        ("managedWithRestrictions", ("overheadPosition",)),
        ("managedWithRestrictions", ("horizontalPush",)),
        ("managedWithRestrictions", ("horizontalPull",)),
        ("managedWithRestrictions", ("armRaise",)),
        ("managedWithRestrictions", ("behindBackRotation",)),
        ("managedWithRestrictions", ("unknown",)),
    ]

    for request_status, request_patterns in valid_contexts:
        for profile_status, profile_patterns in valid_contexts:
            status, patterns = resolve_shoulder_condition_context(
                request_status,
                request_patterns,
                profile_status,
                profile_patterns,
            )

            if "activeUncleared" in {request_status, profile_status}:
                assert (status, patterns) == ("activeUncleared", ())
                continue

            if "managedWithRestrictions" in {request_status, profile_status}:
                assert status == "managedWithRestrictions"
                combined = {*request_patterns, *profile_patterns}
                if "unknown" in combined:
                    assert patterns == ("unknown",)
                else:
                    assert set(patterns) == combined
                continue

            assert patterns == ()
            if "managedNoRestrictions" in {request_status, profile_status}:
                assert status == "managedNoRestrictions"
            elif "recovered" in {request_status, profile_status}:
                assert status == "recovered"
            else:
                assert status is None


def test_candidate_patterns_use_reviewed_metadata_and_high_confidence_names():
    overhead_press = _candidate(
        "overhead_press",
        "덤벨 오버헤드 프레스",
        "front-deltoids",
        overhead=1,
    )
    bench_press = _candidate("bench_press", "덤벨 벤치프레스", "chest")
    seated_row = _candidate("seated_row", "케이블 시티드 로우", "upper-back")
    lateral_raise = _candidate(
        "lateral_raise",
        "덤벨 레터럴 레이즈",
        "front-deltoids",
        movement_type="ISOLATION",
        plane="frontal",
    )
    behind_back_raise = _candidate(
        "behind_back_raise",
        "케이블 비하인드백 레터럴 레이즈",
        "front-deltoids",
        movement_type="ISOLATION",
        plane="frontal",
    )

    assert "overheadPosition" in candidate_shoulder_movement_patterns(overhead_press)
    assert "horizontalPush" in candidate_shoulder_movement_patterns(bench_press)
    assert "horizontalPull" in candidate_shoulder_movement_patterns(seated_row)
    assert "armRaise" in candidate_shoulder_movement_patterns(lateral_raise)
    assert {
        "armRaise",
        "behindBackRotation",
    }.issubset(candidate_shoulder_movement_patterns(behind_back_raise))


def test_low_bar_squat_is_not_misclassified_as_horizontal_pull():
    low_bar_squat = _candidate(
        "low_bar_squat",
        "바벨 로우바 스쿼트",
        "quadriceps",
        secondary="gluteal, hamstring",
    )

    assert "horizontalPull" not in candidate_shoulder_movement_patterns(low_bar_squat)


def test_korean_throw_word_is_not_misclassified_as_row():
    rotation_throw = _candidate(
        "rotation_throw",
        "메디신볼 로테이션 스로우",
        "quadriceps",
        secondary="upper-back, trapezius",
    )

    assert "horizontalPull" not in candidate_shoulder_movement_patterns(rotation_throw)


def test_deep_squat_is_not_misclassified_as_a_dip():
    deep_squat = _candidate(
        "deep_squat_hold",
        "호흡동반 딥 스쿼트 홀드",
        "upper-back",
        movement_type="STATIC",
    )
    bench_dip = _candidate(
        "bench_dip",
        "Bodyweight 벤치 딥 Dip",
        "triceps",
        movement_type="COMPOUND",
    )

    assert "behindBackRotation" not in candidate_shoulder_movement_patterns(deep_squat)
    assert "behindBackRotation" in candidate_shoulder_movement_patterns(bench_dip)


def test_korean_single_dip_is_classified_without_misclassifying_deep_squat():
    bench_dip = _candidate(
        "V24ROW-002609",
        "맨몸 벤치 벤치 딥",
        "triceps",
        movement_type="COMPOUND",
    )

    assert candidate_shoulder_movement_patterns(bench_dip) == frozenset({
        "behindBackRotation",
    })
    assert candidate_has_shoulder_load(bench_dip)
    assert candidate_restricted_movement_overlap(
        bench_dip,
        "activeUncleared",
        [],
    )
    assert candidate_restricted_movement_overlap(
        bench_dip,
        "managedWithRestrictions",
        ["behindBackRotation"],
    ) == frozenset({"behindBackRotation"})


def test_overhead_squat_is_filtered_even_when_primary_muscle_is_lower_body():
    overhead_squat = _candidate(
        "V24ROW-003081",
        "덤벨 내로우 패럴렐 오버헤드 스쿼트",
        "quadriceps",
        movement_type="COMPOUND",
    )

    assert "overheadPosition" in candidate_shoulder_movement_patterns(overhead_squat)
    assert candidate_has_shoulder_load(overhead_squat)
    assert candidate_restricted_movement_overlap(
        overhead_squat,
        "activeUncleared",
        [],
    )
    assert candidate_restricted_movement_overlap(
        overhead_squat,
        "managedWithRestrictions",
        ["overheadPosition"],
    ) == frozenset({"overheadPosition"})


@pytest.mark.parametrize(
    "candidate",
    [
        _candidate(
            "dumbbell_fly",
            "덤벨 플라이",
            "chest",
            movement_type="ISOLATION",
            plane="horizontal",
        ),
        _candidate(
            "machine_row",
            "머신 시티드 로우",
            "upper-back",
        ),
        _candidate(
            "hanging_leg_raise",
            "풀업바 행잉 레그 레이즈",
            "abs",
            overhead=1,
        ),
    ],
)
def test_every_known_shoulder_pattern_is_in_broad_shoulder_load_set(candidate):
    patterns = candidate_shoulder_movement_patterns(candidate)

    assert patterns - {"unknown"}
    assert candidate_has_shoulder_load(candidate)
    assert candidate_restricted_movement_overlap(
        candidate,
        "activeUncleared",
        [],
    )
    assert candidate_restricted_movement_overlap(
        candidate,
        "managedWithRestrictions",
        ["unknown"],
    )


def test_managed_horizontal_push_restriction_keeps_unrelated_arm_and_pull_work():
    candidates = [
        _candidate("bench_press", "덤벨 벤치프레스", "chest"),
        _candidate("seated_row", "케이블 시티드 로우", "upper-back"),
        _candidate(
            "supported_curl",
            "프리처 덤벨 컬",
            "biceps",
            movement_type="ISOLATION",
        ),
    ]

    ranked = score_candidate_exercises(
        candidates,
        ["chest", "upper-back", "biceps"],
        shoulder_condition_status="managedWithRestrictions",
        restricted_movement_patterns=["horizontalPush"],
        top_n=12,
    )

    ids = {candidate["id"] for candidate in ranked}
    assert "bench_press" not in ids
    assert {"seated_row", "supported_curl"} <= ids


def test_unclassified_shoulder_load_is_quarantined_while_any_restriction_is_active():
    unknown_shoulder_compound = _candidate(
        "unknown_chest_compound",
        "아크 로드",
        "chest",
    )

    assert candidate_shoulder_movement_patterns(unknown_shoulder_compound) == frozenset({"unknown"})
    assert candidate_restricted_movement_overlap(
        unknown_shoulder_compound,
        "managedWithRestrictions",
        ["horizontalPush"],
    ) == frozenset({"unknown"})
    assert candidate_restricted_movement_overlap(
        unknown_shoulder_compound,
        "managedWithRestrictions",
        ["unknown"],
    ) == frozenset({"unknown"})


def test_active_uncleared_blocks_shoulder_loaded_candidates_but_not_supported_curl():
    shoulder_press = _candidate(
        "shoulder_press",
        "머신 숄더프레스",
        "front-deltoids",
        overhead=1,
    )
    supported_curl = _candidate(
        "supported_curl",
        "프리처 덤벨 컬",
        "biceps",
        movement_type="ISOLATION",
    )

    ranked = score_candidate_exercises(
        [shoulder_press, supported_curl],
        ["front-deltoids", "biceps"],
        shoulder_condition_status="activeUncleared",
        top_n=12,
    )

    assert [candidate["id"] for candidate in ranked] == ["supported_curl"]


def test_active_uncleared_keeps_shoulder_independent_lower_body_candidate():
    back_squat = _candidate(
        "back_squat",
        "바벨 백스쿼트",
        "quadriceps",
        secondary="gluteal, hamstring",
    )
    leg_press = _candidate(
        "leg_press",
        "머신 레그프레스",
        "quadriceps",
        secondary="gluteal, hamstring",
    )

    ranked = score_candidate_exercises(
        [back_squat, leg_press],
        ["quadriceps", "gluteal", "hamstring"],
        shoulder_condition_status="activeUncleared",
        top_n=12,
    )

    assert [candidate["id"] for candidate in ranked] == ["leg_press"]


def test_active_uncleared_treats_plank_as_shoulder_loaded_core_work():
    plank = _candidate(
        "front_plank",
        "맨몸 프론트 플랭크",
        "abs",
        movement_type="STATIC",
    )
    reverse_crunch = _candidate(
        "reverse_crunch",
        "맨몸 리버스 크런치",
        "abs",
        movement_type="ISOLATION",
    )

    ranked = score_candidate_exercises(
        [plank, reverse_crunch],
        ["abs"],
        shoulder_condition_status="activeUncleared",
        top_n=12,
    )

    assert [candidate["id"] for candidate in ranked] == ["reverse_crunch"]


def test_managed_no_restrictions_does_not_apply_blanket_shoulder_filter():
    bench_press = _candidate("bench_press", "덤벨 벤치프레스", "chest")

    ranked = score_candidate_exercises(
        [bench_press],
        ["chest"],
        shoulder_condition_status="managedNoRestrictions",
        top_n=12,
    )

    assert [candidate["id"] for candidate in ranked] == ["bench_press"]


@pytest.mark.parametrize("status", ["managedNoRestrictions", "recovered"])
def test_nonrestrictive_status_is_ranking_identical_to_omitted_status(status):
    candidates = [
        _candidate("bench_press", "덤벨 벤치프레스", "chest"),
        _candidate("seated_row", "케이블 시티드 로우", "upper-back"),
        _candidate(
            "supported_curl",
            "프리처 덤벨 컬",
            "biceps",
            movement_type="ISOLATION",
        ),
    ]

    baseline = score_candidate_exercises(
        candidates,
        ["chest", "upper-back", "biceps"],
        top_n=12,
    )
    with_status = score_candidate_exercises(
        candidates,
        ["chest", "upper-back", "biceps"],
        shoulder_condition_status=status,
        top_n=12,
    )

    baseline_snapshot = [
        (candidate["id"], candidate["score"])
        for candidate in baseline
    ]
    status_snapshot = [
        (candidate["id"], candidate["score"])
        for candidate in with_status
    ]
    assert status_snapshot == baseline_snapshot


def test_managed_restriction_removes_only_matching_pattern_without_rescoring_survivors():
    candidates = [
        _candidate("bench_press", "덤벨 벤치프레스", "chest"),
        _candidate("seated_row", "케이블 시티드 로우", "upper-back"),
        _candidate(
            "supported_curl",
            "프리처 덤벨 컬",
            "biceps",
            movement_type="ISOLATION",
        ),
    ]
    baseline = score_candidate_exercises(
        candidates,
        ["chest", "upper-back", "biceps"],
        top_n=12,
    )
    restricted = score_candidate_exercises(
        candidates,
        ["chest", "upper-back", "biceps"],
        shoulder_condition_status="managedWithRestrictions",
        restricted_movement_patterns=["horizontalPush"],
        top_n=12,
    )

    baseline_by_id = {
        candidate["id"]: candidate["score"]
        for candidate in baseline
    }
    assert [candidate["id"] for candidate in restricted] == [
        candidate["id"]
        for candidate in baseline
        if candidate["id"] != "bench_press"
    ]
    assert all(
        candidate["score"] == baseline_by_id[candidate["id"]]
        for candidate in restricted
    )


@pytest.mark.parametrize("status", ["managedNoRestrictions", "recovered"])
def test_current_binary_deltoid_injury_still_overrides_nonrestrictive_history_status(status):
    bench_press = _candidate("bench_press", "덤벨 벤치프레스", "chest")

    ranked = score_candidate_exercises(
        [bench_press],
        ["chest"],
        pain_areas=[PainAreaEntry(body_part="front-deltoids")],
        shoulder_condition_status=status,
        top_n=12,
    )

    assert ranked == []


def test_active_uncleared_upper_target_stops_before_candidate_or_llm_calls():
    request = RoutineRequest(
        target_split_label="chest",
        time_available_min=45,
        shoulder_condition_status="activeUncleared",
    )

    with (
        patch("engines.routine_pipeline.get_candidate_exercises") as candidates_mock,
        patch("engines.routine_pipeline.get_llm") as llm_mock,
    ):
        response = generate_smart_routine(request, db=object())

    candidates_mock.assert_not_called()
    llm_mock.assert_not_called()
    assert response.generation_status == "failed"
    assert response.status_reason_code == "emptyCandidate"
    assert response.routine_blocks == []
    assert any("하체" in warning and "코어" in warning for warning in response.warnings)


def test_profile_active_uncleared_stops_upper_target_before_candidate_or_llm_calls():
    request = RoutineRequest(
        target_split_label="chest",
        time_available_min=45,
    )
    profile = UserProfileContext(
        goal_type="hypertrophy",
        split_type="PPL",
        shoulder_condition_status="activeUncleared",
    )

    with (
        patch("engines.routine_pipeline.get_candidate_exercises") as candidates_mock,
        patch("engines.routine_pipeline.get_llm") as llm_mock,
    ):
        response = generate_smart_routine(request, db=object(), profile=profile)

    candidates_mock.assert_not_called()
    llm_mock.assert_not_called()
    assert response.generation_status == "failed"
    assert response.status_reason_code == "emptyCandidate"
    assert any("의료진 허가 전" in warning for warning in response.warnings)


def test_request_cannot_weaken_profile_active_uncleared_state():
    request = RoutineRequest(
        target_split_label="chest",
        time_available_min=45,
        shoulder_condition_status="managedNoRestrictions",
    )
    profile = UserProfileContext(
        goal_type="hypertrophy",
        split_type="PPL",
        shoulder_condition_status="activeUncleared",
    )

    with (
        patch("engines.routine_pipeline.get_candidate_exercises") as candidates_mock,
        patch("engines.routine_pipeline.get_llm") as llm_mock,
    ):
        response = generate_smart_routine(request, db=object(), profile=profile)

    candidates_mock.assert_not_called()
    llm_mock.assert_not_called()
    assert response.generation_status == "failed"
    assert response.status_reason_code == "emptyCandidate"


@pytest.mark.parametrize(
    ("split", "targets", "expected"),
    [
        ("push", ["chest", "triceps"], True),
        ("chest_back", ["chest", "upper-back"], True),
        ("back", ["upper-back"], True),
        ("legs", ["quadriceps", "hamstring"], False),
        ("core", ["abs", "obliques"], False),
        ("arm", ["biceps", "triceps"], False),
        ("custom", ["chest", "triceps"], True),
        ("custom", ["chest", "quadriceps"], True),
        ("full_body", ["chest", "quadriceps", "hamstring"], True),
    ],
)
def test_active_uncleared_safe_stop_scope_is_target_aware(split, targets, expected):
    assert active_uncleared_requires_safe_stop(
        "activeUncleared",
        split,
        targets,
    ) is expected

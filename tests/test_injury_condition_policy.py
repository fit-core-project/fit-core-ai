from unittest.mock import patch

import pytest
from pydantic import ValidationError

from engines.candidate_ranker import score_candidate_exercises
from engines.injury_condition_policy import (
    active_uncleared_neck_requires_safe_stop,
    candidate_injury_movement_patterns,
    candidate_injury_condition_overlaps,
    resolve_injury_condition_contexts,
)
from engines.routine_pipeline import generate_smart_routine
from engines.schemas import InjuryConditionEntry, RoutineRequest, UserProfileContext


def _candidate(
    exercise_id: str,
    name: str,
    primary: str,
    *,
    secondary: str = "",
    equipment: str = "MACHINE",
    movement_type: str = "COMPOUND",
    **metadata,
) -> dict:
    return {
        "id": exercise_id,
        "name_kr": name,
        "name_en": name,
        "primary_muscle": primary,
        "secondary_muscle": secondary,
        "equipment_req": equipment,
        "difficulty_tier": 3,
        "efficiency_tier": 3,
        "pain_triggers": None,
        "movement_type": movement_type,
        "knee_shear_load": "low",
        "lumbar_load": "low",
        "axial_load": "low",
        "wrist_stress": "low",
        **metadata,
    }


def _condition(area: str, status: str, patterns=()) -> InjuryConditionEntry:
    return InjuryConditionEntry(
        area_code=area,
        status=status,
        restricted_movement_patterns=list(patterns),
    )


@pytest.mark.parametrize(
    ("area", "pattern"),
    [
        ("shoulder", "horizontalPush"),
        ("knee", "deepKneeFlexion"),
        ("lumbar", "hipHinge"),
        ("wrist", "loadedWristExtension"),
        ("ankle", "loadedPlantarFlexion"),
        ("hip", "deepHipFlexion"),
        ("elbow", "loadedElbowFlexion"),
        ("neck", "directNeckLoading"),
    ],
)
def test_closed_injury_condition_contract_accepts_area_specific_patterns(area, pattern):
    request = RoutineRequest.model_validate({
        "targetSplitLabel": "full_body",
        "timeAvailableMin": 45,
        "injuryConditions": [{
            "areaCode": area,
            "status": "managedWithRestrictions",
            "restrictedMovementPatterns": [pattern],
        }],
    })

    assert request.injury_conditions[0].area_code == area
    assert request.injury_conditions[0].restricted_movement_patterns == [pattern]


@pytest.mark.parametrize(
    "condition",
    [
        {
            "areaCode": "knee",
            "status": "managedWithRestrictions",
            "restrictedMovementPatterns": [],
        },
        {
            "areaCode": "lumbar",
            "status": "managedNoRestrictions",
            "restrictedMovementPatterns": ["hipHinge"],
        },
        {
            "areaCode": "wrist",
            "status": "managedWithRestrictions",
            "restrictedMovementPatterns": ["unknown", "fixedBarGrip"],
        },
        {
            "areaCode": "knee",
            "status": "managedWithRestrictions",
            "restrictedMovementPatterns": ["hipHinge"],
        },
        {
            "areaCode": "ankle",
            "status": "managedWithRestrictions",
            "restrictedMovementPatterns": ["loadedElbowFlexion"],
        },
    ],
)
def test_invalid_injury_condition_combinations_are_rejected(condition):
    with pytest.raises(ValidationError):
        RoutineRequest.model_validate({
            "targetSplitLabel": "full_body",
            "timeAvailableMin": 45,
            "injuryConditions": [condition],
        })


def test_duplicate_injury_area_entries_are_rejected_in_request_and_profile():
    conditions = [
        {"areaCode": "knee", "status": "activeUncleared"},
        {"areaCode": "knee", "status": "recovered"},
    ]
    with pytest.raises(ValidationError):
        RoutineRequest.model_validate({
            "targetSplitLabel": "legs",
            "timeAvailableMin": 45,
            "injuryConditions": conditions,
        })
    with pytest.raises(ValidationError):
        UserProfileContext.model_validate({
            "goalType": "hypertrophy",
            "splitType": "PPL",
            "injuryConditions": conditions,
        })


def test_recovered_condition_stays_in_profile_history_but_not_runtime_request():
    condition = {"areaCode": "ankle", "status": "recovered"}

    request = RoutineRequest.model_validate({
        "targetSplitLabel": "legs",
        "timeAvailableMin": 45,
        "injuryConditions": [condition],
    })
    profile = UserProfileContext.model_validate({
        "goalType": "hypertrophy",
        "splitType": "PPL",
        "injuryConditions": [condition],
    })

    assert request.injury_conditions == []
    assert profile.injury_conditions[0].status == "recovered"


@pytest.mark.parametrize(
    ("status", "targets", "expected"),
    [
        ("activeUncleared", ["chest", "triceps"], True),
        ("activeUncleared", ["upper-back", "biceps"], True),
        ("activeUncleared", ["quadriceps", "hamstring"], False),
        ("managedWithRestrictions", ["chest", "triceps"], False),
        ("recovered", ["chest", "triceps"], False),
    ],
)
def test_active_uncleared_neck_safe_stop_is_status_and_target_aware(
    status,
    targets,
    expected,
):
    conditions = [_condition("neck", status, ["unknown"])] if (
        status == "managedWithRestrictions"
    ) else [_condition("neck", status)]

    assert active_uncleared_neck_requires_safe_stop(conditions, targets) is expected


def test_active_uncleared_neck_upper_target_stops_before_candidate_or_llm_calls():
    request = RoutineRequest(
        target_split_label="push",
        time_available_min=45,
        injury_conditions=[_condition("neck", "activeUncleared")],
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
    assert any("목" in warning for warning in response.warnings)
    assert any("의료 전문가" in warning for warning in response.warnings)


def test_request_and_profile_conditions_merge_monotonically_with_legacy_shoulder():
    resolved = resolve_injury_condition_contexts(
        [
            _condition("knee", "managedWithRestrictions", ["deepKneeFlexion"]),
            _condition("wrist", "managedNoRestrictions"),
        ],
        [
            _condition("knee", "managedWithRestrictions", ["impactJumpRun"]),
            _condition("wrist", "activeUncleared"),
        ],
        legacy_shoulder_contexts=(
            ("managedWithRestrictions", ["horizontalPush"]),
            ("activeUncleared", []),
        ),
    )
    by_area = {condition.area_code: condition for condition in resolved}

    assert by_area["shoulder"].status == "activeUncleared"
    assert by_area["knee"].restricted_movement_patterns == (
        "deepKneeFlexion",
        "impactJumpRun",
    )
    assert by_area["wrist"].status == "activeUncleared"


def test_managed_knee_restriction_filters_only_selected_movement_family():
    deep_squat = _candidate(
        "deep_squat",
        "딥 스쿼트",
        "quadriceps",
        secondary="gluteal",
        knee_flexion_depth_demand="high",
        deep_squat_position_required=1,
    )
    leg_extension = _candidate(
        "leg_extension",
        "머신 레그 익스텐션",
        "quadriceps",
        movement_type="ISOLATION",
    )
    seated_leg_curl = _candidate(
        "leg_curl",
        "머신 시티드 레그 컬",
        "hamstring",
        movement_type="ISOLATION",
    )
    hip_thrust = _candidate(
        "hip_thrust",
        "머신 힙 쓰러스트",
        "gluteal",
    )

    ranked = score_candidate_exercises(
        [deep_squat, leg_extension, seated_leg_curl, hip_thrust],
        ["quadriceps", "hamstring", "gluteal"],
        injury_conditions=[
            _condition(
                "knee",
                "managedWithRestrictions",
                ["deepKneeFlexion"],
            )
        ],
        top_n=12,
    )

    ids = {candidate["id"] for candidate in ranked}
    assert "deep_squat" not in ids
    assert {"leg_extension", "leg_curl", "hip_thrust"} <= ids


def test_active_uncleared_knee_blocks_knee_load_but_keeps_hip_dominant_work():
    candidates = [
        _candidate("leg_extension", "레그 익스텐션", "quadriceps"),
        _candidate("leg_curl", "레그 컬", "hamstring"),
        _candidate("hip_thrust", "힙 쓰러스트", "gluteal"),
    ]

    ranked = score_candidate_exercises(
        candidates,
        ["quadriceps", "hamstring", "gluteal"],
        injury_conditions=[_condition("knee", "activeUncleared")],
        top_n=12,
    )

    assert [candidate["id"] for candidate in ranked] == ["hip_thrust"]


def test_lumbar_restrictions_distinguish_hinge_axial_and_supported_pull():
    deadlift = _candidate(
        "deadlift",
        "바벨 데드리프트",
        "hamstring",
        equipment="BARBELL",
        lumbar_load="high",
    )
    back_squat = _candidate(
        "back_squat",
        "바벨 백 스쿼트",
        "quadriceps",
        equipment="BARBELL",
        axial_load="high",
    )
    supported_row = _candidate(
        "supported_row",
        "머신 체스트 서포티드 로우",
        "upper-back",
    )

    assert "hipHinge" in candidate_injury_movement_patterns(deadlift, "lumbar")
    assert "axialLoading" in candidate_injury_movement_patterns(back_squat, "lumbar")
    assert candidate_injury_movement_patterns(supported_row, "lumbar") == frozenset()

    ranked = score_candidate_exercises(
        [deadlift, back_squat, supported_row],
        ["hamstring", "quadriceps", "upper-back"],
        injury_conditions=[
            _condition("lumbar", "managedWithRestrictions", ["hipHinge"])
        ],
        top_n=12,
    )
    ids = {candidate["id"] for candidate in ranked}
    assert "deadlift" not in ids
    assert {"back_squat", "supported_row"} <= ids


def test_wrist_restrictions_distinguish_fixed_bar_from_free_handle_grip():
    barbell_curl = _candidate(
        "barbell_curl",
        "바벨 컬",
        "biceps",
        equipment="BARBELL",
        movement_type="ISOLATION",
    )
    dumbbell_curl = _candidate(
        "dumbbell_curl",
        "덤벨 컬",
        "biceps",
        equipment="DUMBBELL",
        movement_type="ISOLATION",
    )
    push_up = _candidate(
        "push_up",
        "맨몸 푸시업",
        "chest",
        equipment="BODYWEIGHT",
    )

    assert "fixedBarGrip" in candidate_injury_movement_patterns(barbell_curl, "wrist")
    assert "loadedGrip" in candidate_injury_movement_patterns(dumbbell_curl, "wrist")
    assert {
        "loadedWristExtension",
        "weightBearingOnHands",
    } <= candidate_injury_movement_patterns(push_up, "wrist")

    ranked = score_candidate_exercises(
        [barbell_curl, dumbbell_curl, push_up],
        ["biceps", "chest"],
        injury_conditions=[
            _condition("wrist", "managedWithRestrictions", ["fixedBarGrip"])
        ],
        top_n=12,
    )
    ids = {candidate["id"] for candidate in ranked}
    assert "barbell_curl" not in ids
    assert {"dumbbell_curl", "push_up"} <= ids


def test_ankle_restrictions_distinguish_calf_loading_from_supported_leg_work():
    calf_raise = _candidate(
        "calf_raise",
        "머신 스탠딩 카프 레이즈",
        "calves",
        movement_type="ISOLATION",
    )
    leg_extension = _candidate(
        "leg_extension",
        "머신 레그 익스텐션",
        "quadriceps",
        movement_type="ISOLATION",
    )

    assert "loadedPlantarFlexion" in candidate_injury_movement_patterns(
        calf_raise,
        "ankle",
    )

    ranked = score_candidate_exercises(
        [calf_raise, leg_extension],
        ["calves", "quadriceps"],
        injury_conditions=[
            _condition(
                "ankle",
                "managedWithRestrictions",
                ["loadedPlantarFlexion"],
            )
        ],
        top_n=12,
    )

    assert [candidate["id"] for candidate in ranked] == ["leg_extension"]


def test_hip_restriction_blocks_hinge_but_keeps_knee_isolation():
    deadlift = _candidate(
        "deadlift",
        "바벨 데드리프트",
        "hamstring",
        equipment="BARBELL",
    )
    leg_extension = _candidate(
        "leg_extension",
        "머신 레그 익스텐션",
        "quadriceps",
        movement_type="ISOLATION",
    )

    assert "hipHinge" in candidate_injury_movement_patterns(deadlift, "hip")

    ranked = score_candidate_exercises(
        [deadlift, leg_extension],
        ["hamstring", "quadriceps"],
        injury_conditions=[
            _condition("hip", "managedWithRestrictions", ["hipHinge"])
        ],
        top_n=12,
    )

    assert [candidate["id"] for candidate in ranked] == ["leg_extension"]


def test_elbow_restriction_blocks_direct_curl_but_keeps_lower_body_work():
    curl = _candidate(
        "curl",
        "덤벨 바이셉 컬",
        "biceps",
        equipment="DUMBBELL",
        movement_type="ISOLATION",
    )
    leg_press = _candidate(
        "leg_press",
        "머신 레그 프레스",
        "quadriceps",
    )

    assert "loadedElbowFlexion" in candidate_injury_movement_patterns(
        curl,
        "elbow",
    )

    ranked = score_candidate_exercises(
        [curl, leg_press],
        ["biceps", "quadriceps"],
        injury_conditions=[
            _condition(
                "elbow",
                "managedWithRestrictions",
                ["loadedElbowFlexion"],
            )
        ],
        top_n=12,
    )

    assert [candidate["id"] for candidate in ranked] == ["leg_press"]


def test_elbow_flexion_restriction_blocks_row_pulldown_and_face_pull():
    candidates = [
        _candidate(
            "row",
            "덤벨 로우",
            "upper-back",
            equipment="DUMBBELL",
            movement_type="COMPOUND",
        ),
        _candidate(
            "pulldown",
            "머신 랫풀다운",
            "upper-back",
            equipment="MACHINE",
            movement_type="ISOLATION",
        ),
        _candidate(
            "face_pull",
            "케이블 페이스 풀",
            "back-deltoids",
            equipment="CABLE",
            movement_type="ISOLATION",
        ),
        _candidate(
            "reverse_fly",
            "머신 리버스 펙덱 플라이",
            "back-deltoids",
            equipment="MACHINE",
            movement_type="ISOLATION",
        ),
    ]

    ranked = score_candidate_exercises(
        candidates,
        ["upper-back", "back-deltoids"],
        injury_conditions=[
            _condition(
                "elbow",
                "managedWithRestrictions",
                ["loadedElbowFlexion"],
            )
        ],
        top_n=12,
    )

    assert [candidate["id"] for candidate in ranked] == ["reverse_fly"]


def test_neck_restriction_blocks_direct_loading_but_keeps_supported_row():
    neck_flexion = _candidate(
        "neck_flexion",
        "밴드 넥 플렉션",
        "neck",
        equipment="BAND",
        movement_type="ISOLATION",
    )
    supported_row = _candidate(
        "supported_row",
        "머신 체스트 서포티드 로우",
        "upper-back",
    )

    assert "directNeckLoading" in candidate_injury_movement_patterns(
        neck_flexion,
        "neck",
    )

    ranked = score_candidate_exercises(
        [neck_flexion, supported_row],
        ["neck", "upper-back"],
        injury_conditions=[
            _condition(
                "neck",
                "managedWithRestrictions",
                ["directNeckLoading"],
            )
        ],
        top_n=12,
    )

    assert [candidate["id"] for candidate in ranked] == ["supported_row"]


@pytest.mark.parametrize(
    "area",
    ["knee", "lumbar", "wrist", "ankle", "hip", "elbow", "neck"],
)
def test_unknown_managed_restriction_quarantines_all_region_loading(area):
    candidate_by_area = {
        "knee": _candidate("leg_extension", "레그 익스텐션", "quadriceps"),
        "lumbar": _candidate(
            "deadlift",
            "데드리프트",
            "hamstring",
            lumbar_load="high",
        ),
        "wrist": _candidate(
            "dumbbell_curl",
            "덤벨 컬",
            "biceps",
            equipment="DUMBBELL",
        ),
        "ankle": _candidate("calf_raise", "카프 레이즈", "calves"),
        "hip": _candidate("deadlift", "데드리프트", "hamstring"),
        "elbow": _candidate("curl", "덤벨 컬", "biceps"),
        "neck": _candidate("neck_flexion", "넥 플렉션", "neck"),
    }
    overlaps = candidate_injury_condition_overlaps(
        candidate_by_area[area],
        [_condition(area, "managedWithRestrictions", ["unknown"])],
    )

    assert overlaps == {area: frozenset({"unknown"})}


@pytest.mark.parametrize("status", ["managedNoRestrictions", "recovered"])
def test_nonrestrictive_managed_condition_is_ranking_neutral(status):
    candidates = [
        _candidate("deadlift", "데드리프트", "hamstring", lumbar_load="high"),
        _candidate("leg_extension", "레그 익스텐션", "quadriceps"),
    ]
    baseline = score_candidate_exercises(
        candidates,
        ["hamstring", "quadriceps"],
        top_n=12,
    )
    with_condition = score_candidate_exercises(
        candidates,
        ["hamstring", "quadriceps"],
        injury_conditions=[_condition("lumbar", status)],
        top_n=12,
    )

    assert [
        (candidate["id"], candidate["score"])
        for candidate in with_condition
    ] == [
        (candidate["id"], candidate["score"])
        for candidate in baseline
    ]

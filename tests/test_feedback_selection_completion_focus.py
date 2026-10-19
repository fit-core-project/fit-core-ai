import pytest

from engines.routine_blueprint import (
    blueprint_invariant_errors,
    build_blueprint_candidate_bundles,
    build_routine_blueprint,
)
from engines.routine_selection import adapt_slot_selection_output
from engines.schemas import LLMRoutineSelectionOutput


def _blueprint(split, focus):
    return build_routine_blueprint(
        target_split_label=split,
        target_muscles=[],
        goal="hypertrophy",
        experience_level="advanced",
        readiness_level="normal",
        time_available_min=60,
        specialization_focus=focus,
        method_mode="off",
    )


def test_lat_focus_uses_existing_vertical_role_without_starving_an_extra_slot():
    focused = _blueprint("pull", "latEmphasis")
    assert focused.specialization_slot_id == "pull_vertical_main"
    assert "specialization_focus" not in {slot.slot_id for slot in focused.slots}
    candidates = [
        {"id": "row", "name_kr": "머신 로우", "primary_muscle": "upper-back",
         "movement_type": "COMPOUND", "main_routine_default_allowed": 1,
         "exercise_familiarity_tier": "common", "failure_penalty": "low", "injury_caution_level": "low", "score": 110},
        {"id": "pull", "name_kr": "랫풀다운", "primary_muscle": "upper-back",
         "movement_type": "COMPOUND", "main_routine_default_allowed": 1,
         "exercise_familiarity_tier": "common", "failure_penalty": "low", "injury_caution_level": "low", "score": 105},
        {"id": "curl", "name_kr": "덤벨 컬", "primary_muscle": "biceps",
         "movement_type": "ISOLATION", "main_routine_default_allowed": 1, "score": 100},
    ]
    by_slot = {bundle["slotId"]: bundle for bundle in build_blueprint_candidate_bundles(focused, candidates)}
    assert by_slot["pull_main"]["candidates"][0]["exerciseId"] == "row"
    assert by_slot["pull_vertical_main"]["candidates"][0]["exerciseId"] == "pull"
    assert blueprint_invariant_errors(focused) == []


@pytest.mark.parametrize(
    ("split", "focus", "target", "pattern"),
    [
        ("pull", "midBackEmphasis", "upper-back", ("horizontal_row",)),
        ("legs", "quadricepsEmphasis", "quadriceps", ()),
        ("legs", "hamstringEmphasis", "hamstring", ()),
        ("legs", "calvesEmphasis", "calves", ()),
        ("shoulder", "sideDeltoidEmphasis", "side-deltoids", ("lateral_raise",)),
        ("shoulder", "rearDeltoidEmphasis", "back-deltoids", ()),
        ("arm", "bicepsEmphasis", "biceps", ()),
        ("arm", "tricepsEmphasis", "triceps", ()),
        ("arm", "forearmEmphasis", "forearm", ()),
    ],
)
def test_advanced_focus_reserves_direct_slot_from_existing_capacity(
    split, focus, target, pattern
):
    baseline = _blueprint(split, None)
    focused = _blueprint(split, focus)

    focus_slot = next(slot for slot in focused.slots if slot.slot_id == "specialization_focus")

    assert focused.target_exercise_count == baseline.target_exercise_count
    assert focused.target_main_exercise_count == baseline.target_main_exercise_count
    reserved = 2 if focus == "trapeziusEmphasis" else 1
    # A balanced pull complement and an explicit focus exchange roles within
    # the same capacity; the focus must not add both roles to the session.
    replaced_complement = sum(s.slot_id == "pull_supported_row" for s in baseline.slots)
    assert focused.flex_slot_count == baseline.flex_slot_count + replaced_complement - reserved
    assert "pull_supported_row" not in {s.slot_id for s in focused.slots}
    assert focus_slot.requirement == "required_when_safe"
    assert focus_slot.target_muscles == (target,)
    assert focus_slot.preferred_selection_patterns == pattern
    assert blueprint_invariant_errors(focused) == []


def test_hamstring_focus_requires_a_direct_hinge_instead_of_a_curl_or_hip_thrust():
    blueprint = _blueprint("legs", "hamstringEmphasis")
    candidates = [
        {
            "id": "squat",
            "name_kr": "스미스 스쿼트",
            "primary_muscle": "quadriceps",
            "movement_type": "COMPOUND",
            "score": 120,
        },
        {
            "id": "hip_thrust",
            "name_kr": "덤벨 힙 쓰러스트",
            "primary_muscle": "gluteal",
            "movement_type": "COMPOUND",
            "hip_hinge_load": "medium",
            "score": 115,
        },
        {
            "id": "leg_curl",
            "name_kr": "머신 레그 컬",
            "primary_muscle": "hamstring",
            "movement_type": "ISOLATION",
            "score": 110,
        },
        {
            "id": "rdl",
            "name_kr": "덤벨 루마니안 데드리프트",
            "primary_muscle": "hamstring",
            "movement_type": "COMPOUND",
            "hip_hinge_load": "high",
            "score": 105,
        },
        {
            "id": "calf_raise",
            "name_kr": "머신 스탠딩 카프 레이즈",
            "primary_muscle": "calves",
            "movement_type": "ISOLATION",
            "score": 100,
        },
    ]

    bundles = build_blueprint_candidate_bundles(blueprint, candidates)
    by_slot = {bundle["slotId"]: bundle for bundle in bundles}

    assert [item["exerciseId"] for item in by_slot["specialization_focus"]["candidates"]] == ["rdl"]


def test_rear_delt_focus_keeps_two_direct_variants_when_catalog_candidates_exist():
    blueprint = _blueprint("shoulder", "rearDeltoidEmphasis")
    candidates = [
        {
            "id": "shoulder_press",
            "name_kr": "덤벨 숄더 프레스",
            "primary_muscle": "front-deltoids",
            "movement_type": "COMPOUND",
            "score": 130,
        },
        {
            "id": "lateral_raise",
            "name_kr": "머신 레터럴 레이즈",
            "primary_muscle": "side-deltoids",
            "movement_type": "ISOLATION",
            "score": 125,
        },
        {
            "id": "reverse_pec_deck",
            "name_kr": "머신 리버스 펙덱 플라이",
            "primary_muscle": "back-deltoids",
            "movement_type": "ISOLATION",
            "score": 120,
        },
        {
            "id": "face_pull",
            "name_kr": "케이블 페이스 풀",
            "primary_muscle": "back-deltoids",
            "movement_type": "ISOLATION",
            "score": 115,
        },
        {
            "id": "shrug",
            "name_kr": "머신 슈러그",
            "primary_muscle": "trapezius",
            "movement_type": "ISOLATION",
            "score": 110,
        },
    ]

    bundles = build_blueprint_candidate_bundles(blueprint, candidates)
    by_slot = {bundle["slotId"]: bundle for bundle in bundles}

    rear_slot_ids = {
        item["exerciseId"]
        for slot_id in ("target_3_back_deltoids", "specialization_focus")
        for item in by_slot[slot_id]["candidates"]
    }
    assert rear_slot_ids == {"reverse_pec_deck", "face_pull"}

    output = adapt_slot_selection_output(
        LLMRoutineSelectionOutput.model_validate({
            "selections": [
                {"slotId": "target_3_back_deltoids", "exerciseId": "face_pull"},
                {"slotId": "specialization_focus", "exerciseId": "reverse_pec_deck"},
            ]
        }),
        blueprint=blueprint,
        slot_candidate_bundles=bundles,
        ranked_candidates=candidates,
    )

    assert {"reverse_pec_deck", "face_pull"} <= {
        plan.exercise_id for plan in output.exercises
    }

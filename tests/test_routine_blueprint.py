import json

import pytest

from engines.routine_blueprint import (
    blueprint_invariant_errors,
    build_blueprint_candidate_bundles,
    build_routine_blueprint,
    render_blueprint_for_prompt,
)


def _slot(blueprint, slot_id):
    return next(slot for slot in blueprint.slots if slot.slot_id == slot_id)


def _adjunct_slot(blueprint, option):
    return next(slot for slot in blueprint.slots if slot.adjunct_option == option)


def test_push_blueprint_requires_chest_main_and_direct_triceps_when_safe():
    blueprint = build_routine_blueprint(
        target_split_label="push",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=50,
    )

    assert blueprint.target_exercise_count == 6
    assert _slot(blueprint, "push_main").target_muscles == ("chest",)
    assert _slot(blueprint, "triceps_direct").requirement == "required_when_safe"
    assert blueprint.flex_slot_count == 3


def test_pull_blueprint_requires_direct_biceps_not_forearm_substitution():
    blueprint = build_routine_blueprint(
        target_split_label="pull",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="beginner",
        readiness_level="normal",
        time_available_min=45,
    )

    assert _slot(blueprint, "biceps_direct").target_muscles == ("biceps",)
    assert "low_technical_difficulty" in blueprint.candidate_preferences


def test_pull_blueprint_separates_horizontal_and_vertical_pull_patterns():
    blueprint = build_routine_blueprint(
        target_split_label="pull",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=50,
    )

    main = _slot(blueprint, "pull_main")
    vertical = _slot(blueprint, "pull_vertical_main")
    assert main.requirement == "required_when_safe"
    assert main.preferred_selection_patterns == ()
    assert vertical.requirement == "preferred_when_safe"
    assert vertical.preferred_selection_patterns == ("vertical_pull",)

    candidates = [
        {"id": "row", "name_kr": "머신 시티드 로우", "primary_muscle": "upper-back", "movement_type": "COMPOUND", "score": 120},
        {"id": "pulldown", "name_kr": "머신 랫풀다운", "primary_muscle": "upper-back", "movement_type": "ISOLATION", "score": 110},
        {"id": "reverse_fly", "name_kr": "머신 리버스 플라이", "primary_muscle": "back-deltoids", "movement_type": "ISOLATION", "score": 100},
        {"id": "curl", "name_kr": "덤벨 컬", "primary_muscle": "biceps", "movement_type": "ISOLATION", "score": 90},
    ]
    bundles = build_blueprint_candidate_bundles(blueprint, candidates)
    by_slot = {bundle["slotId"]: bundle for bundle in bundles}

    assert [item["exerciseId"] for item in by_slot["pull_main"]["candidates"]] == ["row"]
    assert [item["exerciseId"] for item in by_slot["pull_vertical_main"]["candidates"]] == ["pulldown"]


def test_vertical_only_pull_candidate_still_satisfies_required_back_main():
    blueprint = build_routine_blueprint(
        target_split_label="pull",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=50,
    )
    candidates = [
        {"id": "pulldown", "name_kr": "머신 랫풀다운", "primary_muscle": "upper-back", "movement_type": "ISOLATION", "score": 110},
        {"id": "curl", "name_kr": "덤벨 컬", "primary_muscle": "biceps", "movement_type": "ISOLATION", "score": 90},
    ]

    bundles = build_blueprint_candidate_bundles(blueprint, candidates)
    by_slot = {bundle["slotId"]: bundle for bundle in bundles}

    assert [item["exerciseId"] for item in by_slot["pull_main"]["candidates"]] == ["pulldown"]
    assert by_slot["pull_vertical_main"]["candidates"] == []


def test_pull_main_rejects_non_pull_isolations_and_ignores_coaching_keywords():
    blueprint = build_routine_blueprint(
        target_split_label="pull",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=50,
    )
    candidates = [
        {
            "id": "shrug",
            "name_kr": "머신 슈러그",
            "primary_muscle": "trapezius",
            "movement_type": "ISOLATION",
            "score": 130,
        },
        {
            "id": "pullover",
            "name_kr": "케이블 풀오버",
            "primary_muscle": "upper-back",
            "movement_type": "ISOLATION",
            "score": 125,
            "score_reasons": ["horizontal_row vertical_pull 코칭"],
        },
        {
            "id": "rear_delt",
            "name_kr": "머신 리버스 플라이",
            "primary_muscle": "back-deltoids",
            "movement_type": "ISOLATION",
            "score": 120,
        },
        {
            "id": "forearm",
            "name_kr": "머신 리스트 컬",
            "primary_muscle": "forearm",
            "movement_type": "ISOLATION",
            "score": 115,
        },
        {
            "id": "curl",
            "name_kr": "덤벨 컬",
            "primary_muscle": "biceps",
            "movement_type": "ISOLATION",
            "score": 110,
        },
    ]

    bundles = build_blueprint_candidate_bundles(blueprint, candidates)
    by_slot = {bundle["slotId"]: bundle for bundle in bundles}

    assert by_slot["pull_main"]["candidates"] == []
    assert [item["exerciseId"] for item in by_slot["biceps_direct"]["candidates"]] == ["curl"]


@pytest.mark.parametrize(
    ("target_split_label", "slot_id"),
    [
        ("pull", "pull_main"),
        ("upper", "upper_pull_main"),
        ("full_body", "full_pull_main"),
    ],
)
def test_back_main_slots_reject_upright_row_but_keep_supported_bent_over_row(
    target_split_label,
    slot_id,
):
    blueprint = build_routine_blueprint(
        target_split_label=target_split_label,
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=50,
    )
    candidates = [
        {
            "id": "upright_row",
            "name_kr": "바벨 업라이트 로우",
            "primary_muscle": "trapezius",
            "movement_type": "COMPOUND",
            "score": 130,
        },
        {
            "id": "supported_row",
            "name_kr": "머신 체스트 서포티드 벤트오버 로우",
            "primary_muscle": "upper-back",
            "movement_type": "COMPOUND",
            "score": 125,
        },
        {
            "id": "curl",
            "name_kr": "덤벨 컬",
            "primary_muscle": "biceps",
            "movement_type": "ISOLATION",
            "score": 110,
        },
    ]

    bundles = build_blueprint_candidate_bundles(blueprint, candidates)
    by_slot = {bundle["slotId"]: bundle for bundle in bundles}

    assert [item["exerciseId"] for item in by_slot[slot_id]["candidates"]] == [
        "supported_row"
    ]


def test_trapezius_specialization_keeps_shrug_as_explicit_accessory_role():
    blueprint = build_routine_blueprint(
        target_split_label="pull",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="advanced",
        readiness_level="normal",
        time_available_min=60,
        specialization_focus="trapeziusEmphasis",
        method_mode="off",
    )
    candidates = [
        {
            "id": "row",
            "name_kr": "덤벨 벤트오버 로우",
            "primary_muscle": "upper-back",
            "movement_type": "COMPOUND",
            "score": 130,
        },
        {
            "id": "pulldown",
            "name_kr": "머신 랫풀다운",
            "primary_muscle": "upper-back",
            "movement_type": "ISOLATION",
            "score": 125,
        },
        {
            "id": "curl",
            "name_kr": "덤벨 컬",
            "primary_muscle": "biceps",
            "movement_type": "ISOLATION",
            "score": 120,
        },
        {
            "id": "shrug",
            "name_kr": "머신 슈러그",
            "primary_muscle": "trapezius",
            "movement_type": "ISOLATION",
            "score": 115,
        },
    ]

    bundles = build_blueprint_candidate_bundles(blueprint, candidates)
    by_slot = {bundle["slotId"]: bundle for bundle in bundles}

    assert [item["exerciseId"] for item in by_slot["pull_main"]["candidates"]] == ["row"]
    assert [item["exerciseId"] for item in by_slot["specialization_focus"]["candidates"]] == ["shrug"]


def test_direct_arm_accessory_is_preferred_in_strength_instead_of_universally_required():
    push = build_routine_blueprint(
        target_split_label="push",
        target_muscles=[],
        goal="strength",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=50,
    )
    pull = build_routine_blueprint(
        target_split_label="pull",
        target_muscles=[],
        goal="generalfitness",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=50,
    )

    assert _slot(push, "triceps_direct").requirement == "preferred_when_safe"
    assert _slot(pull, "biceps_direct").requirement == "preferred_when_safe"


def test_legs_blueprint_starts_with_large_lower_body_and_posterior_chain_slots():
    blueprint = build_routine_blueprint(
        target_split_label="legs",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="beginner",
        readiness_level="normal",
        time_available_min=40,
    )

    assert blueprint.slots[0].slot_id == "lower_main"
    assert blueprint.slots[0].preferred_movement_types == ("COMPOUND",)
    assert _slot(blueprint, "posterior_chain").target_muscles == ("hamstring", "gluteal")
    assert all("tibialis" not in slot.target_muscles for slot in blueprint.slots)


def test_full_body_blueprint_has_lower_push_and_pull_required_slots():
    blueprint = build_routine_blueprint(
        target_split_label="full_body",
        target_muscles=[],
        goal="generalfitness",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=30,
    )

    assert blueprint.target_exercise_count == 3
    assert [slot.slot_id for slot in blueprint.slots] == [
        "full_lower_main",
        "full_push_main",
        "full_pull_main",
    ]
    assert blueprint.flex_slot_count == 0


def test_longer_full_body_blueprint_exposes_core_as_preferred_coverage():
    blueprint = build_routine_blueprint(
        target_split_label="full_body",
        target_muscles=[],
        goal="generalfitness",
        experience_level="beginner",
        readiness_level="normal",
        time_available_min=50,
    )

    core_slot = _slot(blueprint, "full_core_accessory")
    assert core_slot.requirement == "preferred_when_safe"
    assert core_slot.target_muscles == ("abs", "obliques")
    posterior = _slot(blueprint, "full_posterior_chain")
    assert posterior.requirement == "preferred_when_safe"
    assert posterior.target_muscles == ("hamstring", "gluteal")
    shoulder = _slot(blueprint, "full_vertical_push")
    assert shoulder.requirement == "preferred_when_safe"
    assert shoulder.preferred_selection_patterns == ("vertical_press",)


def test_full_body_posterior_preference_is_not_a_global_completion_gate():
    blueprint = build_routine_blueprint(
        target_split_label="full_body",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=45,
    )

    posterior = _slot(blueprint, "full_posterior_chain")
    assert posterior.requirement == "preferred_when_safe"
    assert {
        slot.slot_id
        for slot in blueprint.slots
        if slot.requirement == "required_when_safe"
    } == {"full_lower_main", "full_push_main", "full_pull_main"}


def test_strength_blueprint_uses_goal_aware_exercise_count():
    blueprint = build_routine_blueprint(
        target_split_label="legs",
        target_muscles=[],
        goal="strength",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=60,
    )

    assert blueprint.target_exercise_count == 4
    assert blueprint.minimum_exercise_count == 3
    assert blueprint.max_total_sets == 13
    assert blueprint_invariant_errors(blueprint) == []


def test_core_blueprint_uses_abs_and_obliques_as_main_instead_of_lower_back():
    blueprint = build_routine_blueprint(
        target_split_label="core",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=50,
    )

    assert _slot(blueprint, "core_main").target_muscles == ("abs", "obliques")
    assert _slot(blueprint, "core_main").requirement == "required_when_safe"
    posterior = _slot(blueprint, "core_posterior_support")
    assert posterior.target_muscles == ("lower-back",)
    assert posterior.requirement == "preferred_when_safe"


def test_direct_core_targets_infer_core_blueprint_without_split_label():
    blueprint = build_routine_blueprint(
        target_split_label=None,
        target_muscles=["abs", "obliques", "lower-back"],
        goal="generalfitness",
        experience_level="beginner",
        readiness_level="normal",
        time_available_min=40,
    )

    assert blueprint.target_split_label == "core"
    assert _slot(blueprint, "core_main").target_muscles == ("abs", "obliques")


def test_low_readiness_reduces_minimum_count_without_changing_safety_slots():
    normal = build_routine_blueprint(
        target_split_label="push",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=60,
    )
    low = build_routine_blueprint(
        target_split_label="push",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="low",
        time_available_min=60,
    )

    # Historical ID retained: v2 removes the second, exercise-count reduction.
    assert low.minimum_exercise_count == normal.minimum_exercise_count
    assert [slot.slot_id for slot in low.slots] == [slot.slot_id for slot in normal.slots]


def test_beginner_strength_blueprint_adds_capacity_for_productive_duration():
    blueprint = build_routine_blueprint(
        target_split_label="legs",
        target_muscles=[],
        goal="strength",
        experience_level="beginner",
        readiness_level="normal",
        time_available_min=50,
    )

    assert blueprint.target_exercise_count == 4
    assert blueprint.minimum_exercise_count == 3


def test_direct_pain_target_reduces_exercise_count_without_dropping_safe_targets():
    blueprint = build_routine_blueprint(
        target_split_label="push",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=60,
        pain_areas=[{"bodyPart": "chest"}],
    )

    assert blueprint.base_target_exercise_count == 6
    assert blueprint.target_exercise_count == 5
    assert blueprint.minimum_exercise_count == 4
    assert blueprint.safety_limited_target_muscles == ("chest",)
    assert blueprint_invariant_errors(blueprint) == []


def test_only_doms_level_three_reduces_exercise_count():
    for doms_level in (1, 2):
        blueprint = build_routine_blueprint(
            target_split_label="push",
            target_muscles=[],
            goal="hypertrophy",
            experience_level="intermediate",
            readiness_level="normal",
            time_available_min=60,
            doms_data={"chest": doms_level},
        )
        assert blueprint.target_exercise_count == 6
        assert blueprint.safety_limited_target_muscles == ()

    severe = build_routine_blueprint(
        target_split_label="push",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=60,
        doms_data={"chest": 3},
    )
    assert severe.target_exercise_count == 5
    assert severe.safety_limited_target_muscles == ("chest",)


def test_generic_shoulder_pain_expands_to_deltoid_targets():
    blueprint = build_routine_blueprint(
        target_split_label="push",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=60,
        pain_areas=[{"bodyPart": "shoulder"}],
    )

    assert blueprint.safety_limited_target_muscles == (
        "front-deltoids",
        "side-deltoids",
    )
    assert blueprint.target_exercise_count == 3


def test_explicit_shoulder_and_triceps_targets_are_not_blocked_by_chest_doms():
    blueprint = build_routine_blueprint(
        target_split_label="custom",
        target_muscles=["front-deltoids", "triceps"],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=50,
        doms_data={"chest": 3},
    )

    assert blueprint.target_split_label == "custom"
    assert blueprint.target_muscles == ("front-deltoids", "triceps")
    assert blueprint.safety_limited_target_muscles == ()
    assert blueprint.target_exercise_count > 0
    required_targets = [
        slot.target_muscles
        for slot in blueprint.slots
        if slot.requirement == "required_when_safe"
    ]
    assert required_targets == [("front-deltoids",), ("triceps",)]


def test_explicit_triceps_target_remains_valid_with_registered_front_shoulder_injury():
    blueprint = build_routine_blueprint(
        target_split_label="custom",
        target_muscles=["triceps"],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=35,
        pain_areas=[{"bodyPart": "front-deltoids"}],
    )

    assert blueprint.target_split_label == "custom"
    assert blueprint.target_muscles == ("triceps",)
    assert blueprint.safety_limited_target_muscles == ()
    assert blueprint.target_exercise_count > 0
    assert [
        slot.target_muscles
        for slot in blueprint.slots
        if slot.requirement == "required_when_safe"
    ] == [("triceps",)]


def test_single_muscle_chest_strength_uses_three_distinct_training_roles():
    blueprint = build_routine_blueprint(
        target_split_label="chest",
        target_muscles=["chest"],
        goal="strength",
        experience_level="advanced",
        readiness_level="normal",
        time_available_min=60,
    )

    assert blueprint.target_exercise_count == 3
    assert [slot.slot_id for slot in blueprint.slots] == [
        "chest_main",
        "chest_secondary",
        "chest_isolation",
    ]
    assert [slot.preferred_movement_types for slot in blueprint.slots] == [
        ("COMPOUND",),
        ("COMPOUND",),
        ("ISOLATION",),
    ]
    assert all(
        slot.requirement == "required_when_safe" for slot in blueprint.slots
    )


def test_short_single_muscle_strength_session_uses_two_roles():
    blueprint = build_routine_blueprint(
        target_split_label="chest",
        target_muscles=["chest"],
        goal="strength",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=30,
    )

    assert blueprint.target_exercise_count == 2
    assert [slot.slot_id for slot in blueprint.slots] == [
        "chest_main",
        "chest_secondary",
    ]


def test_two_compatible_adjunct_options_use_distinct_ordered_slots():
    blueprint = build_routine_blueprint(
        target_split_label="upper",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=60,
        training_style="mobility",
        adjunct_options=["mobility_warmup", "cooldown_stretch"],
    )

    warmup = _adjunct_slot(blueprint, "mobility_warmup")
    cooldown = _adjunct_slot(blueprint, "cooldown_stretch")
    assert warmup.max_exercises == 1
    assert warmup.phase == "warmup_or_preparation"
    assert cooldown.phase == "cooldown_recovery"
    assert blueprint.slots.index(warmup) == 0
    assert blueprint.slots.index(cooldown) == len(blueprint.slots) - 1
    assert blueprint.applied_adjunct_options == (
        "mobility_warmup",
        "cooldown_stretch",
    )
    assert blueprint.suppressed_adjunct_options == ()
    assert blueprint.target_main_exercise_count == 5
    assert blueprint.minimum_main_exercise_count == 5
    assert blueprint.target_exercise_count == 7


def test_two_adjuncts_at_seventy_five_minutes_preserve_main_minimum_and_cap_total_at_eight():
    blueprint = build_routine_blueprint(
        target_split_label="upper",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=75,
        adjunct_options=["mobility_warmup", "cooldown_stretch"],
    )

    assert blueprint.base_target_exercise_count == 7
    assert blueprint.target_main_exercise_count == 6
    assert blueprint.minimum_main_exercise_count == 6
    assert blueprint.target_exercise_count == 8
    assert blueprint.minimum_exercise_count == 8


def test_adjunct_plan_is_permutation_invariant():
    forward = build_routine_blueprint(
        target_split_label="upper",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=60,
        adjunct_options=["mobility_warmup", "cooldown_stretch"],
    )
    reversed_order = build_routine_blueprint(
        target_split_label="upper",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=60,
        adjunct_options=["cooldown_stretch", "mobility_warmup"],
    )

    assert forward.planned_adjunct_options == reversed_order.planned_adjunct_options
    assert forward.slots == reversed_order.slots
    assert forward.minimum_main_exercise_count == reversed_order.minimum_main_exercise_count


def test_adjunct_capacity_preserves_required_main_slots_before_optional_requests():
    blueprint = build_routine_blueprint(
        target_split_label="full_body",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=30,
        adjunct_options=["mobility_warmup", "cooldown_stretch"],
    )

    assert blueprint.target_main_exercise_count == 3
    assert blueprint.minimum_main_exercise_count == 3
    assert blueprint.target_exercise_count == 5
    assert blueprint.applied_adjunct_options == (
        "mobility_warmup",
        "cooldown_stretch",
    )
    assert blueprint.suppressed_adjunct_options == ()
    assert {
        slot.slot_id
        for slot in blueprint.slots
        if not slot.adjunct_option
    } == {
        "full_lower_main",
        "full_push_main",
        "full_pull_main",
    }


def test_internal_blueprint_suppresses_overlapping_conditioning_pair():
    blueprint = build_routine_blueprint(
        target_split_label="upper",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=60,
        adjunct_options=["conditioning_finisher", "endurance_support"],
    )

    assert blueprint.applied_adjunct_options == ("conditioning_finisher",)
    assert blueprint.suppressed_adjunct_options == ("endurance_support",)


def test_low_readiness_suppresses_high_density_adjunct_in_direct_blueprint_use():
    blueprint = build_routine_blueprint(
        target_split_label="legs",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="low",
        time_available_min=60,
        adjunct_options=["conditioning_finisher"],
    )

    assert all(not slot.adjunct_option for slot in blueprint.slots)
    assert blueprint.applied_adjunct_options == ()
    assert blueprint.suppressed_adjunct_options == ("conditioning_finisher",)


def test_unknown_adjunct_is_suppressed_without_breaking_internal_callers():
    blueprint = build_routine_blueprint(
        target_split_label="upper",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=60,
        adjunct_options=["future_unknown_option"],
    )

    assert blueprint.applied_adjunct_options == ()
    assert blueprint.suppressed_adjunct_options == ("future_unknown_option",)


def test_adjunct_bundle_matches_requested_purpose_not_any_specialty_candidate():
    blueprint = build_routine_blueprint(
        target_split_label="upper",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=60,
        adjunct_options=["mobility_warmup"],
    )
    candidates = [
        {
            "id": "mobility_prep",
            "name_kr": "밴드 숄더 모빌리티",
            "primary_muscle": "front-deltoids",
            "movement_type": "STATIC",
            "training_style_tags": "mobility,yoga_pilates",
            "score": 90,
        },
        {
            "id": "conditioning_finisher",
            "name_kr": "덤벨 스러스터",
            "primary_muscle": "quadriceps",
            "movement_type": "COMPOUND",
            "training_style_tags": "crossfit,conditioning",
            "score": 100,
        },
    ]

    bundles = build_blueprint_candidate_bundles(blueprint, candidates, top_k_per_slot=3)
    adjunct = next(
        bundle for bundle in bundles
        if bundle["adjunctOption"] == "mobility_warmup"
    )

    assert [item["exerciseId"] for item in adjunct["candidates"]] == ["mobility_prep"]


def test_adjunct_candidate_is_reserved_from_normal_and_flex_slots():
    blueprint = build_routine_blueprint(
        target_split_label="push",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=50,
        adjunct_options=["cooldown_stretch"],
    )
    candidates = [
        {
            "id": "pec_stretch",
            "name_kr": "문틀 흉근 스트레치 - 기본",
            "primary_muscle": "chest",
            "movement_type": "ISOLATION",
            "adjunct_option_codes": ["cooldown_stretch"],
            "score": 200,
        },
        {
            "id": "bench",
            "name_kr": "벤치프레스",
            "primary_muscle": "chest",
            "movement_type": "COMPOUND",
            "score": 150,
        },
        {
            "id": "chest_fly",
            "name_kr": "체스트 플라이",
            "primary_muscle": "chest",
            "movement_type": "ISOLATION",
            "score": 140,
        },
        {
            "id": "pushdown",
            "name_kr": "케이블 푸시다운",
            "primary_muscle": "triceps",
            "movement_type": "ISOLATION",
            "score": 130,
        },
    ]

    bundles = build_blueprint_candidate_bundles(blueprint, candidates, top_k_per_slot=2)
    by_slot = {bundle["slotId"]: bundle for bundle in bundles}

    adjunct_slot_id = _adjunct_slot(blueprint, "cooldown_stretch").slot_id
    assert [item["exerciseId"] for item in by_slot[adjunct_slot_id]["candidates"]] == [
        "pec_stretch"
    ]
    for slot_id, bundle in by_slot.items():
        if slot_id == adjunct_slot_id:
            continue
        assert "pec_stretch" not in {
            item["exerciseId"] for item in bundle["candidates"]
        }


def test_reviewed_adjunct_candidate_excludes_higher_scored_legacy_tag_match():
    blueprint = build_routine_blueprint(
        target_split_label="push",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=50,
        adjunct_options=["mobility_warmup"],
    )
    candidates = [
        {
            "id": "legacy_static_stretch",
            "name_kr": "문틀 흉근 스트레치",
            "primary_muscle": "chest",
            "movement_type": "ISOLATION",
            "training_style_tags": "mobility,yoga_pilates",
            "score": 300,
        },
        {
            "id": "reviewed_dynamic_stretch",
            "name_kr": "문틀 흉근 동적 스트레치",
            "primary_muscle": "chest",
            "movement_type": "ISOLATION",
            "adjunct_option_codes": ["mobility_warmup"],
            "score": 100,
        },
        {
            "id": "bench",
            "name_kr": "벤치프레스",
            "primary_muscle": "chest",
            "movement_type": "COMPOUND",
            "score": 150,
        },
        {
            "id": "pushdown",
            "name_kr": "케이블 푸시다운",
            "primary_muscle": "triceps",
            "movement_type": "ISOLATION",
            "score": 130,
        },
    ]

    bundles = build_blueprint_candidate_bundles(blueprint, candidates, top_k_per_slot=3)
    by_slot = {bundle["slotId"]: bundle for bundle in bundles}
    adjunct_slot_id = _adjunct_slot(blueprint, "mobility_warmup").slot_id

    assert [
        item["exerciseId"] for item in by_slot[adjunct_slot_id]["candidates"]
    ] == ["reviewed_dynamic_stretch"]
    assert "legacy_static_stretch" not in {
        item["exerciseId"] for item in by_slot["flex"]["candidates"]
    }


def test_blueprint_is_valid_and_prompt_payload_is_compact_json():
    blueprint = build_routine_blueprint(
        target_split_label="chest_back",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="advanced",
        readiness_level="high",
        time_available_min=75,
        session_structure="superset",
    )

    assert blueprint_invariant_errors(blueprint) == []
    payload = render_blueprint_for_prompt(blueprint)
    decoded = json.loads(payload)
    assert decoded["formula_version"] == "gym-routine-blueprint-v9"
    assert decoded["target_main_exercise_count"] <= decoded["target_exercise_count"]
    assert decoded["session_structure"] == "superset"
    assert decoded["execution_policy"]["accessory_group_type"] == "superset"
    assert decoded["execution_policy"]["pair_antagonist_main_lifts"] is True
    assert decoded["slots"][0]["slot_id"] == "upper_push_main"


def test_direct_target_muscles_infer_push_blueprint_without_split_label():
    blueprint = build_routine_blueprint(
        target_split_label=None,
        target_muscles=["chest", "triceps"],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=50,
    )

    assert blueprint.target_split_label == "push"
    assert _slot(blueprint, "triceps_direct").target_muscles == ("triceps",)


def test_candidate_bundles_separate_main_and_direct_accessory_slots():
    blueprint = build_routine_blueprint(
        target_split_label="push",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=45,
    )
    candidates = [
        {"id": "chest_fly", "name_kr": "체스트 플라이", "primary_muscle": "chest", "movement_type": "ISOLATION", "score": 120},
        {"id": "bench", "name_kr": "벤치프레스", "primary_muscle": "chest", "movement_type": "COMPOUND", "score": 110},
        {"id": "pushdown", "name_kr": "케이블 푸시다운", "primary_muscle": "triceps", "movement_type": "ISOLATION", "score": 100},
        {"id": "row", "name_kr": "시티드 로우", "primary_muscle": "upper-back", "movement_type": "COMPOUND", "score": 90},
    ]

    bundles = build_blueprint_candidate_bundles(blueprint, candidates, top_k_per_slot=2)
    by_slot = {bundle["slotId"]: bundle for bundle in bundles}
    assert [item["exerciseId"] for item in by_slot["push_main"]["candidates"]] == ["bench"]
    assert [item["exerciseId"] for item in by_slot["triceps_direct"]["candidates"]] == ["pushdown"]
    explicit_ids = [
        item["exerciseId"]
        for bundle in bundles
        if bundle["slotId"] != "flex"
        for item in bundle["candidates"]
    ]
    assert len(explicit_ids) == len(set(explicit_ids))
    assert {item["exerciseId"] for item in by_slot["flex"]["candidates"]} >= {
        "bench",
        "pushdown",
    }
    assert "row" not in {
        item["exerciseId"] for item in by_slot["flex"]["candidates"]
    }


def test_chest_strength_bundles_separate_flat_angle_and_isolation_roles():
    blueprint = build_routine_blueprint(
        target_split_label="chest",
        target_muscles=["chest"],
        goal="strength",
        experience_level="advanced",
        readiness_level="normal",
        time_available_min=60,
        method_mode="off",
    )
    candidates = [
        {
            "id": "flat",
            "name_kr": "바벨 플랫 체스트 프레스",
            "primary_muscle": "chest",
            "movement_type": "COMPOUND",
            "score": 120,
        },
        {
            "id": "incline",
            "name_kr": "바벨 인클라인 체스트 프레스",
            "primary_muscle": "chest",
            "movement_type": "COMPOUND",
            "score": 115,
        },
        {
            "id": "fly",
            "name_kr": "머신 펙덱 플라이",
            "primary_muscle": "chest",
            "movement_type": "ISOLATION",
            "score": 100,
        },
    ]

    bundles = build_blueprint_candidate_bundles(
        blueprint,
        candidates,
        top_k_per_slot=3,
    )
    by_slot = {bundle["slotId"]: bundle for bundle in bundles}

    assert [item["exerciseId"] for item in by_slot["chest_main"]["candidates"]] == [
        "flat"
    ]
    assert [
        item["exerciseId"] for item in by_slot["chest_secondary"]["candidates"]
    ] == ["incline"]
    assert [
        item["exerciseId"] for item in by_slot["chest_isolation"]["candidates"]
    ] == ["fly"]

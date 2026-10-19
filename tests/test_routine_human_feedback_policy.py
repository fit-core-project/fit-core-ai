from engines.candidate_ranker import score_candidate_exercises
import pytest
from engines.candidate_ranker import format_candidates_for_prompt
from engines.fallback import build_routine_draft, generate_fallback_routine
from engines.routine_blueprint import (
    build_blueprint_candidate_bundles,
    build_routine_blueprint,
)
from engines.routine_pipeline import _apply_pain_and_doms_warnings
from engines.routine_selection import selection_pattern_limit
from engines.schemas import (
    LLMExercisePlan,
    LLMRoutineOutput,
    PainAreaEntry,
    RoutineRequest,
)


def _candidate(
    exercise_id: str,
    name_kr: str,
    primary: str,
    *,
    secondary: str = "",
    movement_type: str = "COMPOUND",
    equipment: str = "MACHINE",
    efficiency: int = 3,
    technical: str = "low",
    failure: str = "low",
) -> dict:
    return {
        "id": exercise_id,
        "name_kr": name_kr,
        "name_en": name_kr,
        "primary_muscle": primary,
        "secondary_muscle": secondary,
        "equipment_req": equipment,
        "efficiency_tier": efficiency,
        "pain_triggers": secondary,
        "movement_type": movement_type,
        "injury_caution_level": "low",
        "pain_risk_level": "low",
        "lumbar_load": "low",
        "axial_load": "low",
        "knee_shear_load": "low",
        "shoulder_impingement_risk": "low",
        "wrist_stress": "low",
        "technical_difficulty": technical,
        "balance_requirement": "low",
        "failure_penalty": failure,
    }


@pytest.mark.parametrize('doms', [0, 1, 2])
@pytest.mark.parametrize('equipment,delta', [('BARBELL', 20), ('DUMBBELL', 16), ('MACHINE', -4), ('SMITH_MACHINE', -6)])
def test_advanced_equipment_reason_does_not_claim_absence_of_all_safety_constraints(doms, equipment, delta):
    candidate = _candidate('press', '체스트 프레스', 'chest', equipment=equipment)
    ranked = score_candidate_exercises(
        [candidate], ['chest'], doms_db={'chest': doms}, experience_level='advanced',
        readiness_level='normal', goal='hypertrophy', training_style='gym', target_split_label='push',
    )
    assert len(ranked) == 1
    entries = ranked[0]['score_boosts'] + ranked[0]['score_penalties']
    reason = next(item for item in entries if item['rule_code'] in {
        'advanced_free_weight_compound_preference', 'advanced_guided_equipment_alternative_penalty',
    })
    assert '안전 제약이 없' not in reason['reason']
    assert reason['score'] == delta


def test_beginner_gym_style_prefers_mainstream_exercise_over_unfamiliar_variation():
    ranked = score_candidate_exercises(
        [
            _candidate(
                "ring_dip",
                "링 딥스",
                "triceps",
                secondary="chest, front-deltoids",
                equipment="BODYWEIGHT",
                efficiency=0,
                technical="high",
                failure="high",
            ),
            _candidate(
                "machine_chest_press",
                "머신 체스트 프레스",
                "chest",
                secondary="triceps, front-deltoids",
                equipment="MACHINE",
                efficiency=4,
            ),
        ],
        ["chest", "front-deltoids", "triceps"],
        top_n=2,
        experience_level="beginner",
        training_style="gym",
        target_split_label="push",
    )

    assert ranked[0]["id"] == "machine_chest_press"
    assert "mainstream gym exercise priority" in ranked[0]["score_reasons"]
    assert "uncommon default gym variation penalty" in ranked[1]["score_reasons"]


def test_explicit_advanced_paused_method_uses_familiar_base_press_not_obscure_variant():
    paused_press = {
        **_candidate(
            "paused_flat_press",
            "바벨 플랫 일반그립 하단 정지 체스트 프레스",
            "chest",
            secondary="triceps, front-deltoids",
            equipment="BARBELL",
            technical="medium",
        ),
        "name_en": "Barbell Flat Standard Grip Paused Chest Press",
        "user_facing_display_name_kr": "바벨 플랫 체스트 프레스",
        "bench_angle_band": "flat",
        "strength_effect": "elite",
        "hypertrophy_effect": "elite",
        "stimulus_to_fatigue": "fair",
        "progression_ceiling": "high",
        "main_routine_default_allowed": 0,
        "exercise_familiarity_tier": "obscure",
    }
    standard_press = {
        **_candidate(
            "standard_flat_press",
            "바벨 플랫 체스트 프레스",
            "chest",
            secondary="triceps, front-deltoids",
            equipment="BARBELL",
            technical="medium",
        ),
        "bench_angle_band": "flat",
        "strength_effect": "elite",
        "hypertrophy_effect": "elite",
        "stimulus_to_fatigue": "fair",
        "progression_ceiling": "high",
        "main_routine_default_allowed": 1,
        "mainstream_gym_score": 90,
        "exercise_familiarity_tier": "standard",
        "novelty_level": "low",
        "training_style_tags": "bodybuilding,gym",
        "prescription_mode": "reps",
    }

    ranked = score_candidate_exercises(
        [standard_press, paused_press],
        ["chest"],
        top_n=5,
        experience_level="advanced",
        advanced_chest_focus="pausedPress",
        goal="strength",
        training_style="gym",
        target_split_label="chest",
    )
    blueprint = build_routine_blueprint(
        target_split_label="chest",
        target_muscles=["chest"],
        goal="strength",
        experience_level="advanced",
        advanced_chest_focus="pausedPress",
        readiness_level="normal",
        time_available_min=60,
    )
    bundles = build_blueprint_candidate_bundles(
        blueprint,
        ranked,
        top_k_per_slot=3,
    )
    main_bundle = next(bundle for bundle in bundles if bundle["slotId"] == "chest_main")

    assert [candidate["id"] for candidate in ranked] == ["standard_flat_press"]
    assert any(
        item["rule_code"] == "training_method_match"
        for item in ranked[0]["score_boosts"]
    )
    assert blueprint.specialization_focus is None
    assert blueprint.training_method == "pausedReps"
    assert blueprint.training_method_slot_id == "chest_main"
    assert [
        item["exerciseId"] for item in main_bundle["candidates"]
    ] == ["standard_flat_press"]


def test_balanced_advanced_chest_does_not_unlock_paused_specialty_by_default():
    ranked = score_candidate_exercises(
        [{
            **_candidate(
                "paused_flat_press",
                "바벨 플랫 일반그립 하단 정지 체스트 프레스",
                "chest",
                equipment="BARBELL",
                technical="medium",
            ),
            "bench_angle_band": "flat",
            "strength_effect": "elite",
            "main_routine_default_allowed": 0,
            "exercise_familiarity_tier": "obscure",
        }],
        ["chest"],
        top_n=5,
        experience_level="advanced",
        advanced_chest_focus="balanced",
        goal="strength",
        training_style="gym",
        target_split_label="chest",
    )

    assert ranked == []


def test_requested_mobility_warmup_preserves_one_matching_adjunct_candidate():
    # Preserve requested mobility coverage; P1 requires a reviewed exact ID, not a tag-only synthetic row.
    from engines.adjunct_admission import reviewed_purpose_index

    mobility_id = "V24ROW-023032"
    assert "mobility_warmup" in reviewed_purpose_index()[mobility_id.lower()]
    gym_candidate = {
        **_candidate("machine_press", "머신 체스트 프레스", "chest"),
        "training_style_tags": "gym,bodybuilding",
    }
    mobility_candidate = {
        **_candidate(
            mobility_id,
            "네발기기 흉추 회전 스트레칭",
            "upper-back",
            movement_type="ISOLATION",
            equipment="BODYWEIGHT",
            efficiency=5,
        ),
        "training_style_tags": "mobility,yoga_pilates",
        "main_routine_default_allowed": 0,
        "adjunct_option_codes": ["mobility_warmup"],
        "name_en": "Thread the Needle - 동적 반복",
    }

    ranked = score_candidate_exercises(
        [gym_candidate, mobility_candidate],
        ["chest", "upper-back"],
        top_n=2,
        training_style="gym",
        target_split_label="upper",
        adjunct_options=["mobility_warmup"],
    )

    assert {item["id"] for item in ranked} == {"machine_press", mobility_id}
    mobility = next(item for item in ranked if item["id"] == mobility_id)
    assert any(
        boost["rule_code"] == "adjunct_option_match_boost"
        and boost["constraint_code"] == "mobility_warmup"
        for boost in mobility["score_boosts"]
    )


def test_reviewed_adjunct_candidate_is_preserved_ahead_of_legacy_tag_match():
    # Preserve reviewed-purpose priority over legacy tags; P1 review comes from the exact catalog ID.
    from engines.adjunct_admission import reviewed_purpose_index

    reviewed_id = "V24ROW-022960"
    assert "mobility_warmup" in reviewed_purpose_index()[reviewed_id.lower()]
    candidates = [
        {
            **_candidate("machine_press", "머신 체스트 프레스", "chest"),
            "training_style_tags": "gym,bodybuilding",
        },
        {
            **_candidate(
                "legacy_static_stretch",
                "문틀 흉근 스트레치",
                "chest",
                movement_type="ISOLATION",
                equipment="BODYWEIGHT",
                efficiency=1,
            ),
            "training_style_tags": "mobility,yoga_pilates",
        },
        {
            **_candidate(
                reviewed_id,
                "문틀 흉근 동적 스트레치",
                "chest",
                movement_type="ISOLATION",
                equipment="BODYWEIGHT",
                efficiency=5,
            ),
            "adjunct_option_codes": ["mobility_warmup"],
            "training_style_tags": "mobility",
            "main_routine_default_allowed": 0,
        },
    ]

    ranked = score_candidate_exercises(
        candidates,
        ["chest"],
        top_n=2,
        training_style="gym",
        target_split_label="push",
        adjunct_options=["mobility_warmup"],
    )

    assert reviewed_id in {candidate["id"] for candidate in ranked}
    assert {candidate["id"] for candidate in ranked} == {"machine_press", reviewed_id}


def test_available_equipment_whitelist_excludes_unavailable_optional_equipment():
    ranked = score_candidate_exercises(
        [
            _candidate("dumbbell_press", "덤벨 체스트 프레스", "chest", equipment="DUMBBELL"),
            _candidate("landmine_press", "랜드마인 체스트 프레스", "chest", equipment="LANDMINE"),
            _candidate("plate_row", "플레이트로드 로우", "upper-back", equipment="PLATE"),
        ],
        ["chest", "upper-back"],
        top_n=5,
        blocked_equipment=["BARBELL", "MACHINE", "CABLE"],
        available_equipment=["BODYWEIGHT", "DUMBBELL"],
        training_style="gym",
    )

    assert [candidate["id"] for candidate in ranked] == ["dumbbell_press"]


def test_experience_level_changes_strength_candidate_priority():
    candidates = [
        {
            **_candidate(
                "barbell_back_squat",
                "바벨 백스쿼트",
                "quadriceps",
                equipment="BARBELL",
                efficiency=3,
                technical="medium",
                failure="medium",
            ),
            "strength_effect": "elite",
            "stimulus_to_fatigue": "good",
        },
        {
            **_candidate(
                "machine_leg_press",
                "머신 레그 프레스",
                "quadriceps",
                equipment="MACHINE",
                efficiency=3,
                technical="low",
                failure="low",
            ),
            "strength_effect": "medium",
            "stimulus_to_fatigue": "excellent",
        },
    ]

    beginner = score_candidate_exercises(
        candidates,
        ["quadriceps"],
        top_n=2,
        goal="strength",
        experience_level="beginner",
        training_style="gym",
        target_split_label="legs",
    )
    advanced = score_candidate_exercises(
        candidates,
        ["quadriceps"],
        top_n=2,
        goal="strength",
        experience_level="advanced",
        readiness_level="normal",
        training_style="gym",
        target_split_label="legs",
    )

    assert beginner[0]["id"] == "machine_leg_press"
    assert advanced[0]["id"] == "barbell_back_squat"
    assert any(item["rule_code"] == "advanced_high_effect_boost" for item in advanced[0]["score_boosts"])
    assert any(item["rule_code"] == "advanced_compound_variation_boost" for item in advanced[0]["score_boosts"])


def test_default_gym_experience_changes_equipment_priority_without_obscure_unlock():
    candidates = [
        {
            **_candidate(
                "barbell_press",
                "바벨 벤치 프레스",
                "chest",
                equipment="BARBELL",
                technical="medium",
                failure="medium",
            ),
            "main_routine_default_allowed": 1,
            "beginner_default_allowed": 0,
            "exercise_familiarity_tier": "common",
            "mainstream_gym_score": 90,
            "hypertrophy_effect": "elite",
            "stimulus_to_fatigue": "good",
        },
        {
            **_candidate(
                "dumbbell_press",
                "덤벨 체스트 프레스",
                "chest",
                equipment="DUMBBELL",
            ),
            "main_routine_default_allowed": 1,
            "beginner_default_allowed": 1,
            "exercise_familiarity_tier": "common",
            "mainstream_gym_score": 90,
            "hypertrophy_effect": "elite",
            "stimulus_to_fatigue": "good",
        },
        {
            **_candidate(
                "machine_press",
                "머신 체스트 프레스",
                "chest",
                equipment="MACHINE",
            ),
            "main_routine_default_allowed": 1,
            "beginner_default_allowed": 1,
            "exercise_familiarity_tier": "common",
            "mainstream_gym_score": 90,
            "hypertrophy_effect": "elite",
            "stimulus_to_fatigue": "excellent",
        },
        {
            **_candidate(
                "smith_press",
                "스미스 머신 체스트 프레스",
                "chest",
                equipment="SMITH_MACHINE",
            ),
            "main_routine_default_allowed": 1,
            "beginner_default_allowed": 1,
            "exercise_familiarity_tier": "common",
            "mainstream_gym_score": 90,
            "hypertrophy_effect": "elite",
            "stimulus_to_fatigue": "excellent",
        },
    ]

    beginner = score_candidate_exercises(
        candidates,
        ["chest"],
        top_n=4,
        goal="hypertrophy",
        experience_level="beginner",
        readiness_level="normal",
        training_style="gym",
        target_split_label="push",
    )
    intermediate = score_candidate_exercises(
        candidates,
        ["chest"],
        top_n=4,
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        training_style="gym",
        target_split_label="push",
    )
    advanced = score_candidate_exercises(
        candidates,
        ["chest"],
        top_n=4,
        goal="hypertrophy",
        experience_level="advanced",
        readiness_level="normal",
        training_style="gym",
        target_split_label="push",
    )

    assert beginner[0]["id"] == "machine_press"
    assert intermediate[0]["id"] == "dumbbell_press"
    assert advanced[0]["id"] == "barbell_press"
    assert any(
        item["rule_code"] == "intermediate_free_weight_progression_preference"
        for item in intermediate[0]["score_boosts"]
    )
    assert any(
        item["rule_code"] == "advanced_free_weight_compound_preference"
        for item in advanced[0]["score_boosts"]
    )
    assert any(
        item["rule_code"] == "advanced_guided_equipment_alternative_penalty"
        for item in next(item for item in advanced if item["id"] == "smith_press")[
            "score_penalties"
        ]
    )


def test_intermediate_strength_goal_keeps_reviewed_barbell_compound_available():
    candidates = [
        {
            **_candidate(
                "goblet_squat",
                "덤벨 고블릿 스쿼트",
                "quadriceps",
                equipment="DUMBBELL",
            ),
            "main_routine_default_allowed": 1,
            "beginner_default_allowed": 1,
            "exercise_familiarity_tier": "common",
            "mainstream_gym_score": 90,
            "strength_effect": "high",
            "stimulus_to_fatigue": "good",
            "technical_difficulty": "low",
            "failure_penalty": "medium",
            "injury_caution_level": "medium",
        },
        {
            **_candidate(
                "barbell_squat",
                "바벨 백 스쿼트",
                "quadriceps",
                equipment="BARBELL",
            ),
            "main_routine_default_allowed": 1,
            "beginner_default_allowed": 0,
            "exercise_familiarity_tier": "common",
            "mainstream_gym_score": 90,
            "strength_effect": "elite",
            "stimulus_to_fatigue": "fair",
            "technical_difficulty": "high",
            "failure_penalty": "high",
            "injury_caution_level": "high",
        },
    ]

    ranked = score_candidate_exercises(
        candidates,
        ["quadriceps"],
        top_n=2,
        goal="strength",
        experience_level="intermediate",
        readiness_level="normal",
        training_style="gym",
        target_split_label="legs",
    )

    assert ranked[0]["id"] == "barbell_squat"
    assert any(
        item["rule_code"]
        == "intermediate_strength_barbell_progression_preference"
        for item in ranked[0]["score_boosts"]
    )


def test_advanced_default_main_lowers_low_progression_ceiling_candidates():
    candidates = [
        {
            **_candidate(
                "goblet_squat",
                "덤벨 고블릿 스쿼트",
                "quadriceps",
                equipment="DUMBBELL",
            ),
            "main_routine_default_allowed": 1,
            "exercise_familiarity_tier": "common",
            "mainstream_gym_score": 90,
            "hypertrophy_effect": "elite",
            "stimulus_to_fatigue": "good",
            "technical_difficulty": "low",
            "failure_penalty": "medium",
            "injury_caution_level": "medium",
        },
        {
            **_candidate(
                "hack_squat",
                "핵 스쿼트",
                "quadriceps",
                equipment="MACHINE",
            ),
            "main_routine_default_allowed": 1,
            "exercise_familiarity_tier": "common",
            "mainstream_gym_score": 90,
            "hypertrophy_effect": "elite",
            "stimulus_to_fatigue": "excellent",
            "technical_difficulty": "low",
            "failure_penalty": "low",
            "injury_caution_level": "medium",
        },
    ]

    ranked = score_candidate_exercises(
        candidates,
        ["quadriceps"],
        top_n=2,
        goal="hypertrophy",
        experience_level="advanced",
        readiness_level="normal",
        training_style="gym",
        target_split_label="legs",
    )

    assert ranked[0]["id"] == "hack_squat"
    assert any(
        item["rule_code"] == "advanced_progression_ceiling_penalty"
        for item in next(item for item in ranked if item["id"] == "goblet_squat")[
            "score_penalties"
        ]
    )


def test_intermediate_experience_prefers_mainstream_progression_over_unfamiliar_high_skill():
    ranked = score_candidate_exercises(
        [
            {
                **_candidate(
                    "novel_jump_squat",
                    "고난도 점프 스쿼트 변형",
                    "quadriceps",
                    equipment="BARBELL",
                    efficiency=2,
                    technical="high",
                    failure="medium",
                ),
                "mainstream_gym_score": 30,
                "exercise_familiarity_tier": "obscure",
                "novelty_level": "high",
                "strength_effect": "high",
                "stimulus_to_fatigue": "good",
            },
            {
                **_candidate(
                    "smith_squat",
                    "스미스 머신 스쿼트",
                    "quadriceps",
                    equipment="SMITH_MACHINE",
                    efficiency=4,
                    technical="medium",
                    failure="medium",
                ),
                "mainstream_gym_score": 84,
                "exercise_familiarity_tier": "common",
                "novelty_level": "low",
                "strength_effect": "high",
                "stimulus_to_fatigue": "good",
            },
        ],
        ["quadriceps"],
        top_n=2,
        goal="strength",
        experience_level="intermediate",
        training_style="gym",
        target_split_label="legs",
    )

    assert ranked[0]["id"] == "smith_squat"
    mainstream = next(candidate for candidate in ranked if candidate["id"] == "smith_squat")
    novel = next(candidate for candidate in ranked if candidate["id"] == "novel_jump_squat")
    assert any(
        item["rule_code"] == "intermediate_mainstream_progression_boost"
        for item in mainstream["score_boosts"]
    )
    assert any(
        item["rule_code"] == "intermediate_unfamiliar_high_technique_penalty"
        for item in novel["score_penalties"]
    )


def test_advanced_effect_unlock_does_not_override_high_failure_or_injury():
    ranked = score_candidate_exercises(
        [
            {
                **_candidate(
                    "risky_elite_lift",
                    "고위험 엘리트 리프트",
                    "quadriceps",
                    equipment="BARBELL",
                    efficiency=2,
                    technical="high",
                    failure="high",
                ),
                "injury_caution_level": "high",
                "strength_effect": "elite",
                "stimulus_to_fatigue": "excellent",
            },
            {
                **_candidate(
                    "advanced_safe_squat",
                    "상급자 세이프티바 스쿼트",
                    "quadriceps",
                    equipment="BARBELL",
                    efficiency=4,
                    technical="medium",
                    failure="medium",
                ),
                "injury_caution_level": "medium",
                "strength_effect": "high",
                "stimulus_to_fatigue": "good",
            },
        ],
        ["quadriceps"],
        top_n=2,
        goal="strength",
        readiness_level="high",
        experience_level="advanced",
        training_style="gym",
        target_split_label="legs",
    )

    risky = next(candidate for candidate in ranked if candidate["id"] == "risky_elite_lift")
    safe = next(candidate for candidate in ranked if candidate["id"] == "advanced_safe_squat")
    assert not any(item["rule_code"] == "advanced_high_effect_boost" for item in risky["score_boosts"])
    assert any(item["rule_code"] == "general_injury_caution_penalty" for item in risky["score_penalties"])
    assert any(item["rule_code"] == "advanced_high_effect_boost" for item in safe["score_boosts"])


def test_beginner_default_recommendation_demotes_high_injury_caution_candidate():
    high_risk = _candidate("side_lunge", "덤벨 사이드 런지", "adductor", equipment="DUMBBELL")
    high_risk["injury_caution_level"] = "high"
    stable = _candidate("machine_adductor", "머신 힙 어덕션", "adductor", equipment="MACHINE")

    ranked = score_candidate_exercises(
        [high_risk, stable],
        ["adductor"],
        top_n=2,
        experience_level="beginner",
        training_style="gym",
        target_split_label="legs",
    )

    assert ranked[0]["id"] == "machine_adductor"
    assert "general high injury caution penalty" in ranked[1]["score_reasons"]


def test_db_commonness_metadata_overrides_name_heuristic_for_gym_priority():
    ranked = score_candidate_exercises(
        [
            {
                **_candidate("catalog_common", "검수된 표준 후보", "chest", equipment="MACHINE"),
                "mainstream_gym_score": 86,
                "exercise_familiarity_tier": "common",
                "novelty_level": "low",
                "training_style_tags": "gym,bodybuilding",
            },
            {
                **_candidate("catalog_obscure", "낯선 변형 후보", "chest", equipment="MACHINE"),
                "mainstream_gym_score": 22,
                "exercise_familiarity_tier": "obscure",
                "novelty_level": "high",
                "training_style_tags": "athletic",
            },
        ],
        ["chest"],
        top_n=2,
        experience_level="beginner",
        training_style="gym",
        target_split_label="push",
    )

    assert [item["id"] for item in ranked] == ["catalog_common"]
    assert "mainstream gym exercise priority" in ranked[0]["score_reasons"]


def test_explicitly_reviewed_standard_exercise_can_represent_specialty_source_row():
    ranked = score_candidate_exercises(
        [
            {
                **_candidate(
                    "reviewed_hip_thrust",
                    "덤벨 힙 쓰러스트",
                    "gluteal",
                    equipment="DUMBBELL",
                    movement_type="ISOLATION",
                ),
                "mainstream_gym_score": 95,
                "exercise_familiarity_tier": "common",
                "novelty_level": "low",
                "training_style_tags": "athletic,bodybuilding,gym,home",
                "main_routine_default_allowed": 1,
                "beginner_default_allowed": 1,
            },
            {
                **_candidate(
                    "unreviewed_cycle_drill",
                    "사이클 하체 보강 드릴",
                    "gluteal",
                    equipment="BODYWEIGHT",
                ),
                "mainstream_gym_score": 95,
                "exercise_familiarity_tier": "common",
                "novelty_level": "low",
                "training_style_tags": "athletic",
                "main_routine_default_allowed": 0,
            },
        ],
        ["gluteal"],
        top_n=2,
        experience_level="intermediate",
        training_style="gym",
        target_split_label="legs",
    )

    assert [item["id"] for item in ranked] == ["reviewed_hip_thrust"]


def test_standard_gym_metadata_alone_does_not_mark_variation_as_mainstream():
    ranked = score_candidate_exercises(
        [
            {
                **_candidate("standard_variation", "스미스 머신 상체 세움 스텝다운", "quadriceps"),
                "mainstream_gym_score": 67,
                "exercise_familiarity_tier": "standard",
                "novelty_level": "low",
                "training_style_tags": "gym,bodybuilding",
            },
        ],
        ["quadriceps"],
        top_n=1,
        experience_level="intermediate",
        training_style="gym",
        target_split_label="legs",
    )

    assert "mainstream gym exercise priority" not in ranked[0]["score_reasons"]


def test_common_default_candidate_ranks_before_more_efficient_standard_candidate():
    ranked = score_candidate_exercises(
        [
            {
                **_candidate(
                    "standard_candidate",
                    "머신 해머 컬",
                    "biceps",
                    equipment="MACHINE",
                    efficiency=2,
                ),
                "mainstream_gym_score": 77,
                "exercise_familiarity_tier": "standard",
                "novelty_level": "low",
                "main_routine_default_allowed": 1,
            },
            {
                **_candidate(
                    "common_candidate",
                    "덤벨 프리처 컬",
                    "biceps",
                    equipment="DUMBBELL",
                    efficiency=3,
                ),
                "mainstream_gym_score": 95,
                "exercise_familiarity_tier": "common",
                "novelty_level": "low",
                "main_routine_default_allowed": 1,
            },
        ],
        ["biceps"],
        top_n=2,
        experience_level="intermediate",
        training_style="gym",
        target_split_label="pull",
    )

    assert [item["id"] for item in ranked] == [
        "common_candidate",
        "standard_candidate",
    ]
    assert (
        "common default gym familiarity priority"
        in ranked[0]["score_reasons"]
    )


def test_standard_dumbbell_triceps_kickback_is_recognized_as_mainstream():
    ranked = score_candidate_exercises(
        [
            {
                **_candidate(
                    "dumbbell_kickback",
                    "덤벨 벤트오버 킥백",
                    "triceps",
                    equipment="DUMBBELL",
                ),
                "mainstream_gym_score": 67,
                "exercise_familiarity_tier": "standard",
                "novelty_level": "low",
                "training_style_tags": "gym,bodybuilding,home",
            },
        ],
        ["triceps"],
        top_n=1,
        experience_level="intermediate",
        training_style="gym",
        target_split_label="push",
    )

    assert "mainstream gym exercise priority" in ranked[0]["score_reasons"]


def test_default_gym_formula_excludes_non_default_micro_variation_candidate():
    ranked = score_candidate_exercises(
        [
            {
                **_candidate("micro_variation", "덤벨 발끝 바깥 싱글레그 힙 쓰러스트", "gluteal"),
                "user_facing_display_name_kr": "덤벨 힙 쓰러스트",
                "mainstream_gym_score": 82,
                "exercise_familiarity_tier": "common",
                "novelty_level": "low",
                "training_style_tags": "gym,bodybuilding",
                "beginner_default_allowed": 0,
                "main_routine_default_allowed": 0,
            },
            {
                **_candidate("stable_default", "머신 힙 쓰러스트", "gluteal", equipment="MACHINE"),
                "user_facing_display_name_kr": "머신 힙 쓰러스트",
                "mainstream_gym_score": 82,
                "exercise_familiarity_tier": "common",
                "novelty_level": "low",
                "training_style_tags": "gym,bodybuilding",
                "beginner_default_allowed": 1,
                "main_routine_default_allowed": 1,
            },
        ],
        ["gluteal"],
        top_n=2,
        experience_level="beginner",
        training_style="gym",
        target_split_label="legs",
    )

    assert [item["id"] for item in ranked] == ["stable_default"]


def test_default_gym_formula_excludes_non_default_candidate_for_advanced_user():
    ranked = score_candidate_exercises(
        [
            {
                **_candidate("obscure_elite", "고효과 특수 변형 운동", "quadriceps"),
                "mainstream_gym_score": 25,
                "exercise_familiarity_tier": "obscure",
                "novelty_level": "high",
                "training_style_tags": "gym,bodybuilding",
                "main_routine_default_allowed": 0,
                "strength_effect": "elite",
                "quality_score": 95,
            },
            {
                **_candidate("barbell_squat", "바벨 백 스쿼트", "quadriceps", equipment="BARBELL"),
                "mainstream_gym_score": 90,
                "exercise_familiarity_tier": "common",
                "novelty_level": "low",
                "training_style_tags": "gym,bodybuilding",
                "main_routine_default_allowed": 1,
                "strength_effect": "high",
                "quality_score": 90,
            },
        ],
        ["quadriceps"],
        top_n=2,
        goal="strength",
        experience_level="advanced",
        training_style="gym",
        target_split_label="legs",
    )

    assert [item["id"] for item in ranked] == ["barbell_squat"]


def test_front_raise_variations_share_one_final_selection_pattern():
    blueprint = build_routine_blueprint(
        target_split_label="push",
        target_muscles=["chest", "front-deltoids", "triceps"],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=50,
    )
    assert selection_pattern_limit("front_raise", blueprint) == 1


def test_lumbar_injury_blueprint_allows_supported_lower_main_replacement():
    blueprint = build_routine_blueprint(
        target_split_label="legs",
        target_muscles=["quadriceps", "hamstring", "gluteal", "calves"],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=50,
        pain_areas=[PainAreaEntry(body_part="lower-back")],
    )
    lower_main = next(slot for slot in blueprint.slots if slot.slot_id == "lower_main")

    assert lower_main.preferred_movement_types == ("COMPOUND", "ISOLATION")
    assert "지지형 하체 메인" in lower_main.purpose


def test_beginner_legs_keeps_stable_main_compound_before_safe_isolation_accessory():
    stable_isolation = {
        **_candidate(
            "machine_leg_curl",
            "머신 레그 컬",
            "hamstring",
            equipment="MACHINE",
            movement_type="ISOLATION",
            efficiency=3,
            technical="low",
            failure="low",
        ),
        "mainstream_gym_score": 95,
        "exercise_familiarity_tier": "common",
        "novelty_level": "low",
        "beginner_default_allowed": 1,
        "main_routine_default_allowed": 1,
        "hypertrophy_effect": "high",
        "stimulus_to_fatigue": "excellent",
    }
    progressive_compound = {
        **_candidate(
            "v_squat",
            "브이 스쿼트",
            "quadriceps",
            equipment="MACHINE",
            movement_type="COMPOUND",
            efficiency=3,
            technical="low",
            failure="low",
        ),
        "pain_risk_level": "medium",
        "injury_caution_level": "medium",
        "lumbar_load": "medium",
        "knee_shear_load": "medium",
        "mainstream_gym_score": 95,
        "exercise_familiarity_tier": "common",
        "novelty_level": "low",
        "beginner_default_allowed": 1,
        "main_routine_default_allowed": 1,
        "hypertrophy_effect": "elite",
        "stimulus_to_fatigue": "excellent",
    }

    beginner = score_candidate_exercises(
        [progressive_compound, stable_isolation],
        ["quadriceps", "hamstring", "gluteal"],
        top_n=2,
        goal="hypertrophy",
        experience_level="beginner",
        training_style="gym",
        target_split_label="legs",
    )
    intermediate = score_candidate_exercises(
        [progressive_compound, stable_isolation],
        ["quadriceps", "hamstring", "gluteal"],
        top_n=2,
        goal="hypertrophy",
        experience_level="intermediate",
        training_style="gym",
        target_split_label="legs",
    )

    assert beginner[0]["id"] == "v_squat"
    assert beginner[1]["id"] == "machine_leg_curl"
    assert intermediate[0]["id"] == "v_squat"
    assert any(
        item["rule_code"] == "beginner_low_load_baseline_boost"
        for item in beginner[1]["score_boosts"]
    )
    assert any(
        item["rule_code"] == "beginner_moderate_compound_load_caution"
        for item in beginner[0]["score_penalties"]
    )
    assert any(
        item["rule_code"] == "intermediate_mainstream_progression_boost"
        for item in intermediate[0]["score_boosts"]
    )


def test_candidate_prompt_uses_user_facing_display_name():
    prompt = format_candidates_for_prompt(
        [
            {
                **_candidate("hip_thrust_variant", "덤벨 발끝 바깥 싱글레그 힙 쓰러스트", "gluteal"),
                "user_facing_display_name_kr": "덤벨 힙 쓰러스트",
                "score": 90,
            }
        ],
        compact=True,
    )

    assert "덤벨 힙 쓰러스트" in prompt
    assert "덤벨 발끝 바깥 싱글레그 힙 쓰러스트" not in prompt


def test_candidate_prompt_uses_detailed_name_when_user_facing_display_is_too_generic():
    prompt = format_candidates_for_prompt(
        [
            {
                **_candidate("pelvic_curl_variant", "소도구 미니볼 펠빅컬 - 탄성저항 보조형 정렬 중심", "hamstring"),
                "display_name_ko": "소도구 미니볼 펠빅컬 - 탄성저항 보조형 정렬 중심",
                "user_facing_display_name_kr": "밴드 컬",
                "score": 90,
            }
        ],
        compact=True,
    )

    assert "소도구 미니볼 펠빅컬" in prompt
    assert "밴드 컬" not in prompt


def test_non_default_training_style_can_boost_matching_style_tag():
    ranked = score_candidate_exercises(
        [
            {
                **_candidate("crossfit_match", "덤벨 AMRAP 스러스터", "quadriceps", equipment="DUMBBELL"),
                "mainstream_gym_score": 35,
                "exercise_familiarity_tier": "specialized",
                "novelty_level": "medium",
                "training_style_tags": "crossfit,athletic",
            },
            _candidate("gym_default", "머신 레그 프레스", "quadriceps", equipment="MACHINE"),
        ],
        ["quadriceps"],
        top_n=2,
        training_style="crossfit",
        target_split_label="legs",
    )

    match = next(candidate for candidate in ranked if candidate["id"] == "crossfit_match")
    gym_default = next(candidate for candidate in ranked if candidate["id"] == "gym_default")
    assert "crossfit adjunct style match" in match["score_reasons"]
    assert any(item["rule_code"] == "training_style_match_boost" for item in match["score_boosts"])
    assert any(item["rule_code"] == "mainstream_gym_priority" for item in gym_default["score_boosts"])
    assert ranked[0]["id"] == "gym_default"


def test_training_style_match_does_not_enable_unsupported_equipment():
    candidate = {
        **_candidate("wall_ball", "크로스핏 월볼", "quadriceps", equipment="MEDICINE_BALL"),
        "training_style_tags": "crossfit,athletic",
    }
    assert score_candidate_exercises(
        [candidate], ['quadriceps'], training_style='crossfit', target_split_label='legs',
    ) == []


def test_adjunct_training_style_preserves_gym_main_but_keeps_one_style_option():
    ranked = score_candidate_exercises(
        [
            {
                **_candidate("gym_main", "브이 스쿼트", "quadriceps", equipment="MACHINE"),
                "training_style_tags": "gym,bodybuilding",
                "mainstream_gym_score": 95,
                "exercise_familiarity_tier": "common",
                "novelty_level": "low",
            },
            {
                **_candidate("gym_accessory", "머신 레그 컬", "hamstring", equipment="MACHINE"),
                "training_style_tags": "gym,bodybuilding",
                "mainstream_gym_score": 90,
                "exercise_familiarity_tier": "common",
                "novelty_level": "low",
            },
            {
                **_candidate("crossfit_adjunct", "덤벨 AMRAP 스러스터", "quadriceps", equipment="DUMBBELL"),
                "training_style_tags": "crossfit,home",
                "mainstream_gym_score": 35,
                "exercise_familiarity_tier": "specialized",
                "novelty_level": "medium",
            },
        ],
        ["quadriceps", "hamstring"],
        top_n=2,
        training_style="crossfit",
        target_split_label="legs",
    )

    assert ranked[0]["id"] != "crossfit_adjunct"
    assert "gym" in str(ranked[0].get("training_style_tags") or "")
    assert "crossfit_adjunct" in {candidate["id"] for candidate in ranked}
    adjunct = next(candidate for candidate in ranked if candidate["id"] == "crossfit_adjunct")
    assert any(item["rule_code"] == "training_style_match_boost" for item in adjunct["score_boosts"])


def test_yoga_pilates_training_style_can_boost_matching_specialty_candidate(monkeypatch):
    # Preserve the yoga-style adjunct boost and gym-main priority using a P1-reviewed mobility ID.
    from engines.adjunct_admission import reviewed_purpose_index

    specialty_id = "V24ROW-023052"
    assert "mobility_warmup" in reviewed_purpose_index()[specialty_id.lower()]
    ranked = score_candidate_exercises(
        [
            {
                **_candidate(specialty_id, "월드 그레이티스트 스트레치", "hamstring",
                             movement_type="ISOLATION", equipment="BODYWEIGHT"),
                "mainstream_gym_score": 39,
                "exercise_familiarity_tier": "specialized",
                "novelty_level": "medium",
                "training_style_tags": "yoga_pilates,mobility,home",
                "main_routine_default_allowed": 0,
            },
            {
                **_candidate("gym_default", "머신 레그 프레스", "quadriceps", equipment="MACHINE"),
                "training_style_tags": "gym,bodybuilding",
                "mainstream_gym_score": 90,
                "exercise_familiarity_tier": "common",
                "novelty_level": "low",
            },
        ],
        ["quadriceps", "hamstring"],
        top_n=2,
        training_style="yoga_pilates",
        target_split_label="legs",
    )

    match = next(candidate for candidate in ranked if candidate["id"] == specialty_id)
    gym_default = next(candidate for candidate in ranked if candidate["id"] == "gym_default")
    assert any(item["rule_code"] == "training_style_match_boost" for item in match["score_boosts"])
    assert not any(
        item["rule_code"] == "specialty_mobility_default_gym_suppression"
        for item in match["score_penalties"]
    )
    assert any(item["rule_code"] == "mainstream_gym_priority" for item in gym_default["score_boosts"])
    assert ranked[0]["id"] == "gym_default"

    # Isolate the suppression predicate so its style exemption is exercised without inventing a catalog name.
    from engines import candidate_ranker

    controls = {}
    with monkeypatch.context() as scoped:
        scoped.setattr(candidate_ranker, "_is_specialty_mobility_default_gym_candidate",
                       lambda candidate: candidate["id"] == specialty_id)
        for style in ("gym", "yoga_pilates"):
            controls[style] = score_candidate_exercises(
                [dict(match)], ["hamstring"], top_n=1,
                training_style=style, target_split_label="legs",
                adjunct_options=["mobility_warmup"],
            )
            assert [candidate["id"] for candidate in controls[style]] == [specialty_id]
    suppression = "specialty_mobility_default_gym_suppression"
    assert any(item["rule_code"] == suppression for item in controls["gym"][0]["score_penalties"])
    assert not any(item["rule_code"] == suppression for item in controls["yoga_pilates"][0]["score_penalties"])


def test_default_gym_demotes_rehab_and_excludes_cardio_style_when_not_requested():
    ranked = score_candidate_exercises(
        [
            {
                **_candidate("senior_hinge", "시니어 의자 힙힌지", "hamstring", equipment="DUMBBELL"),
                "training_style_tags": "home,rehab",
                "mainstream_gym_score": 45,
                "exercise_familiarity_tier": "specialized",
                "novelty_level": "medium",
            },
            {
                **_candidate("treadmill_walk", "러닝머신 평지 걷기", "hamstring", equipment="TREADMILL"),
                "training_style_tags": "cardio,athletic",
                "mainstream_gym_score": 25,
                "exercise_familiarity_tier": "obscure",
                "novelty_level": "high",
            },
            {
                **_candidate("leg_curl", "머신 레그 컬", "hamstring", movement_type="ISOLATION", equipment="MACHINE"),
                "training_style_tags": "gym,bodybuilding",
                "mainstream_gym_score": 82,
                "exercise_familiarity_tier": "common",
                "novelty_level": "low",
            },
        ],
        ["hamstring"],
        top_n=3,
        training_style="gym",
        goal="hypertrophy",
    )

    assert ranked[0]["id"] == "leg_curl"
    senior = next(candidate for candidate in ranked if candidate["id"] == "senior_hinge")
    assert "rehab style default gym penalty" in senior["score_reasons"]
    assert all(candidate["id"] != "treadmill_walk" for candidate in ranked)


def test_default_gym_suppresses_yoga_and_small_tool_alignment_variants_even_if_metadata_marks_common():
    ranked = score_candidate_exercises(
        [
            {
                **_candidate("pelvic_curl", "소도구 미니볼 펠빅컬 - 볼 지지 보조형 정렬 중심", "hamstring", equipment="BODYWEIGHT"),
                "training_style_tags": "gym,home",
                "mainstream_gym_score": 93,
                "exercise_familiarity_tier": "common",
                "novelty_level": "low",
                "main_routine_default_allowed": 1,
            },
            {
                **_candidate("warrior_2", "워리어 2 - 기본", "quadriceps", equipment="BODYWEIGHT"),
                "training_style_tags": "gym,home",
                "mainstream_gym_score": 93,
                "exercise_familiarity_tier": "common",
                "novelty_level": "low",
                "main_routine_default_allowed": 1,
            },
            {
                **_candidate("machine_leg_curl", "머신 레그 컬", "hamstring", movement_type="ISOLATION", equipment="MACHINE"),
                "training_style_tags": "gym,bodybuilding",
                "mainstream_gym_score": 82,
                "exercise_familiarity_tier": "common",
                "novelty_level": "low",
            },
            {
                **_candidate("machine_leg_press", "머신 레그 프레스", "quadriceps", equipment="MACHINE"),
                "training_style_tags": "gym,bodybuilding",
                "mainstream_gym_score": 90,
                "exercise_familiarity_tier": "common",
                "novelty_level": "low",
            },
        ],
        ["quadriceps", "hamstring", "gluteal"],
        top_n=4,
        training_style="gym",
        target_split_label="legs",
        goal="hypertrophy",
    )

    ranked_ids = [candidate["id"] for candidate in ranked]
    assert ranked_ids.index("machine_leg_press") < ranked_ids.index("warrior_2")
    assert ranked_ids.index("machine_leg_curl") < ranked_ids.index("pelvic_curl")
    pelvic = next(candidate for candidate in ranked if candidate["id"] == "pelvic_curl")
    warrior = next(candidate for candidate in ranked if candidate["id"] == "warrior_2")
    assert any(
        item["rule_code"] == "specialty_mobility_default_gym_suppression"
        for item in pelvic["score_penalties"]
    )
    assert any(
        item["rule_code"] == "specialty_mobility_default_gym_suppression"
        for item in warrior["score_penalties"]
    )


def test_push_split_marks_direct_triceps_accessory_as_coverage_candidate():
    ranked = score_candidate_exercises(
        [
            _candidate("barbell_bench", "바벨 벤치프레스", "chest", secondary="triceps", equipment="BARBELL"),
            _candidate(
                "triceps_pushdown",
                "케이블 트라이셉스 푸쉬다운",
                "triceps",
                movement_type="ISOLATION",
                equipment="CABLE",
            ),
        ],
        ["chest", "front-deltoids", "triceps"],
        top_n=2,
        training_style="gym",
        target_split_label="push",
    )

    pushdown = next(candidate for candidate in ranked if candidate["id"] == "triceps_pushdown")
    assert "push direct accessory coverage boost" in pushdown["score_reasons"]
    assert any(item["rule_code"] == "split_accessory_direct_coverage_boost" for item in pushdown["score_boosts"])


def test_push_split_demotes_non_target_pull_primary_candidates():
    ranked = score_candidate_exercises(
        [
            _candidate("cable_row", "케이블 로우", "upper-back", equipment="CABLE", efficiency=1),
            _candidate("machine_chest_press", "머신 체스트 프레스", "chest", secondary="triceps", equipment="MACHINE", efficiency=4),
        ],
        ["chest", "front-deltoids", "triceps"],
        top_n=2,
        training_style="gym",
        target_split_label="push",
    )

    assert ranked[0]["id"] == "machine_chest_press"
    row = next(candidate for candidate in ranked if candidate["id"] == "cable_row")
    assert "push split non target primary penalty" in row["score_reasons"]
    assert any(item["rule_code"] == "split_non_target_primary_penalty" for item in row["score_penalties"])


def test_beginner_legs_demotes_tibialis_accessory_as_main_candidate():
    ranked = score_candidate_exercises(
        [
            _candidate(
                "machine_tibialis_raise",
                "머신 티비알리스 레이즈",
                "calves",
                movement_type="ISOLATION",
                equipment="MACHINE",
                efficiency=1,
            ),
            _candidate("leg_press", "머신 레그 프레스", "quadriceps", equipment="MACHINE", efficiency=4),
        ],
        ["quadriceps", "hamstring", "gluteal", "calves"],
        top_n=2,
        experience_level="beginner",
        training_style="gym",
        target_split_label="legs",
    )

    assert ranked[0]["id"] == "leg_press"
    tibialis = next(candidate for candidate in ranked if candidate["id"] == "machine_tibialis_raise")
    assert "beginner legs tibialis accessory-only penalty" in tibialis["score_reasons"]
    assert any(
        item["rule_code"] == "beginner_legs_tibialis_accessory_penalty"
        for item in tibialis["score_penalties"]
    )


def test_generic_display_name_is_demoted_below_specific_exercise_name():
    ranked = score_candidate_exercises(
        [
            _candidate(
                "generic_curl",
                "컬",
                "hamstring",
                movement_type="ISOLATION",
                equipment="BODYWEIGHT",
                efficiency=3,
            ),
            _candidate(
                "machine_leg_curl",
                "머신 레그 컬",
                "hamstring",
                movement_type="ISOLATION",
                equipment="MACHINE",
                efficiency=3,
            ),
        ],
        ["hamstring"],
        top_n=2,
        training_style="gym",
        target_split_label="legs",
    )

    assert ranked[0]["id"] == "machine_leg_curl"
    generic = next(candidate for candidate in ranked if candidate["id"] == "generic_curl")
    assert any(item["rule_code"] == "generic_display_name_penalty" for item in generic["score_penalties"])


def test_multi_target_candidate_pool_keeps_direct_accessory_coverage_when_available():
    ranked = score_candidate_exercises(
        [
            _candidate("bench_a", "바벨 벤치프레스", "chest", secondary="triceps", equipment="BARBELL", efficiency=3),
            _candidate("bench_b", "덤벨 벤치프레스", "chest", secondary="triceps", equipment="DUMBBELL", efficiency=3),
            _candidate("bench_c", "머신 체스트 프레스", "chest", secondary="triceps", equipment="MACHINE", efficiency=3),
            _candidate(
                "triceps_pushdown",
                "케이블 트라이셉스 푸쉬다운",
                "triceps",
                movement_type="ISOLATION",
                equipment="CABLE",
                efficiency=5,
            ),
        ],
        ["chest", "front-deltoids", "triceps"],
        top_n=3,
        training_style="gym",
        target_split_label="push",
    )

    assert "triceps_pushdown" in {candidate["id"] for candidate in ranked}


def test_pull_direct_biceps_accessory_prefers_curl_over_biceps_labeled_pulldown():
    ranked = score_candidate_exercises(
        [
            _candidate(
                "undergrip_lat_pulldown",
                "랫풀다운 머신 언더그립 랫 풀다운",
                "biceps",
                secondary="upper-back",
                movement_type="COMPOUND",
                equipment="MACHINE",
                efficiency=3,
            ),
            _candidate(
                "machine_drag_curl",
                "머신 시티드 드래그 컬",
                "biceps",
                movement_type="ISOLATION",
                equipment="MACHINE",
                efficiency=4,
            ),
        ],
        ["upper-back", "biceps"],
        top_n=2,
        training_style="gym",
        target_split_label="pull",
    )

    assert ranked[0]["id"] == "machine_drag_curl"
    pulldown = next(candidate for candidate in ranked if candidate["id"] == "undergrip_lat_pulldown")
    assert any(
        item["rule_code"] == "pull_biceps_pulldown_ambiguity_penalty"
        for item in pulldown["score_penalties"]
    )


def test_fallback_order_respects_lumbar_and_limited_ankle_context():
    deadlift = {
        **_candidate("kettlebell_deadlift", "케틀벨 데드리프트", "hamstring", equipment="KETTLEBELL", efficiency=1),
        "score": 180,
        "ankle_dorsiflexion_demand": "medium",
        "lumbar_load": "medium",
    }
    leg_press = {
        **_candidate("leg_press", "머신 레그 프레스", "quadriceps", equipment="MACHINE", efficiency=4),
        "score": 80,
        "ankle_dorsiflexion_demand": "low",
        "lumbar_load": "low",
    }
    req = RoutineRequest(
        target_split_label="legs",
        target_muscles=["quadriceps", "hamstring", "gluteal"],
        time_available_min=45,
        mobility_limits=["limited_ankle_dorsiflexion"],
        pain_areas=[PainAreaEntry(body_part="lower-back")],
    )

    draft = generate_fallback_routine(
        req,
        [deadlift, leg_press],
        max_total_sets=6,
        goal="hypertrophy",
    )

    assert draft.generation_status == "fallback"
    assert draft.routine_blocks[0].exercise_id == "leg_press"


def test_build_routine_draft_groups_same_primary_muscle_before_next_region():
    output = LLMRoutineOutput(
        total_estimated_time=40,
        summary_title="전신 루틴",
        rationale_summary=["같은 부위끼리 묶는지 확인합니다."],
        warnings=[],
        exercises=[
            LLMExercisePlan(
                exercise_id="chest_press",
                exercise_name="체스트 프레스",
                movement_type="COMPOUND",
                primary_muscles=["chest"],
                target_reps=10,
                sets=3,
                rest_time_sec=90,
                exercise_rationale="가슴 메인",
            ),
            LLMExercisePlan(
                exercise_id="row",
                exercise_name="시티드 로우",
                movement_type="COMPOUND",
                primary_muscles=["upper-back"],
                target_reps=10,
                sets=3,
                rest_time_sec=90,
                exercise_rationale="등 메인",
            ),
            LLMExercisePlan(
                exercise_id="fly",
                exercise_name="펙덱 플라이",
                movement_type="ISOLATION",
                primary_muscles=["chest"],
                target_reps=12,
                sets=2,
                rest_time_sec=60,
                exercise_rationale="가슴 보조",
            ),
        ],
    )

    draft = build_routine_draft(output, "success", "none", False)

    assert [block.exercise_id for block in draft.routine_blocks] == ["chest_press", "fly", "row"]


def test_build_routine_draft_can_preserve_blueprint_input_order():
    output = LLMRoutineOutput(
        total_estimated_time=40,
        summary_title="전신 루틴",
        rationale_summary=["하체, 밀기, 당기기 슬롯 순서를 확인합니다."],
        warnings=[],
        exercises=[
            LLMExercisePlan(
                exercise_id="squat",
                exercise_name="덤벨 스쿼트",
                movement_type="COMPOUND",
                primary_muscles=["quadriceps"],
                target_reps=10,
                sets=3,
                rest_time_sec=90,
                exercise_rationale="하체 메인",
            ),
            LLMExercisePlan(
                exercise_id="chest_press",
                exercise_name="덤벨 체스트 프레스",
                movement_type="COMPOUND",
                primary_muscles=["chest"],
                target_reps=10,
                sets=3,
                rest_time_sec=90,
                exercise_rationale="밀기 메인",
            ),
            LLMExercisePlan(
                exercise_id="row",
                exercise_name="덤벨 로우",
                movement_type="COMPOUND",
                primary_muscles=["upper-back"],
                target_reps=10,
                sets=3,
                rest_time_sec=90,
                exercise_rationale="당기기 메인",
            ),
        ],
    )

    draft = build_routine_draft(
        output,
        "success",
        "none",
        False,
        preserve_input_order=True,
    )

    assert [block.exercise_id for block in draft.routine_blocks] == [
        "squat",
        "chest_press",
        "row",
    ]


def test_contextual_warning_mentions_selected_pain_and_doms_overlap():
    output = LLMRoutineOutput(
        total_estimated_time=30,
        summary_title="푸시 루틴",
        rationale_summary=["주의 문구 확인"],
        warnings=[],
        exercises=[
            LLMExercisePlan(
                exercise_id="bench",
                exercise_name="벤치프레스",
                movement_type="COMPOUND",
                primary_muscles=["chest"],
                target_reps=8,
                sets=3,
                rest_time_sec=90,
                exercise_rationale="가슴 메인",
            ),
        ],
    )
    draft = build_routine_draft(output, "success", "none", False)
    req = RoutineRequest(
        target_split_label="push",
        time_available_min=45,
        pain_areas=[PainAreaEntry(body_part="front-deltoids")],
        doms_data={"chest": 1},
    )
    ranked = [
        _candidate(
            "bench",
            "벤치프레스",
            "chest",
            secondary="triceps, front-deltoids",
            equipment="BARBELL",
        )
    ]

    warned = _apply_pain_and_doms_warnings(draft, req, ranked)

    assert warned.generation_status == "failed"
    assert warned.status_reason_code == "emptyCandidate"
    assert warned.routine_blocks == []
    assert any("전면 어깨" in warning and "안전 규칙에 따라 제외" in warning for warning in warned.warnings)

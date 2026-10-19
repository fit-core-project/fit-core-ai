from __future__ import annotations

from engines.experience_policy import main_role_eligibility
from engines.prescription.experience import resolve_experience_prescription
from engines.prescription.targets import apply_deterministic_targets
from engines.routine_blueprint import build_blueprint_candidate_bundles, build_routine_blueprint
from engines.schemas import LLMExercisePlan, LLMRoutineOutput


def _catalog_candidate(exercise_id: str, **overrides) -> dict:
    candidate = {
        "id": exercise_id,
        "name_kr": exercise_id,
        "primary_muscle": "quadriceps",
        "movement_type": "COMPOUND",
        "score": 100,
        "main_routine_default_allowed": 1,
        "beginner_default_allowed": 1,
        "exercise_familiarity_tier": "common",
        "mainstream_gym_score": 90,
        "technical_difficulty": "low",
        "failure_penalty": "low",
        "injury_caution_level": "low",
        "quality_score": 90,
        "hypertrophy_effect": "high",
        "strength_effect": "high",
        "loadability": "high",
        "progression_ceiling": "high",
    }
    candidate.update(overrides)
    return candidate


def _plan() -> LLMExercisePlan:
    return LLMExercisePlan(
        exercise_id="squat",
        exercise_name="스쿼트",
        movement_type="COMPOUND",
        primary_muscles=["quadriceps"],
        target_reps=8,
        sets=2,
        rest_time_sec=120,
        target_rir=2,
        exercise_rationale="하체 메인 운동입니다.",
    )


def test_beginner_main_role_requires_stable_mainstream_metadata():
    common = _catalog_candidate("leg_press")
    obscure = _catalog_candidate(
        "rare_variant",
        main_routine_default_allowed=0,
        beginner_default_allowed=0,
        exercise_familiarity_tier="obscure",
        mainstream_gym_score=20,
    )

    assert main_role_eligibility(common, "beginner", "hypertrophy").eligible
    assert not main_role_eligibility(obscure, "beginner", "hypertrophy").eligible


def test_advanced_does_not_unlock_obscure_exercise_by_effect_score_alone():
    low_value = _catalog_candidate(
        "rare_low_value",
        main_routine_default_allowed=0,
        exercise_familiarity_tier="obscure",
        hypertrophy_effect="low",
        loadability="low",
        progression_ceiling="low",
    )
    high_value = _catalog_candidate(
        "rare_high_value",
        main_routine_default_allowed=0,
        exercise_familiarity_tier="obscure",
        hypertrophy_effect="elite",
        loadability="high",
        progression_ceiling="high",
        quality_score=90,
    )

    assert not main_role_eligibility(low_value, "advanced", "hypertrophy").eligible
    assert not main_role_eligibility(high_value, "advanced", "hypertrophy").eligible


def test_blueprint_main_bundle_applies_experience_role_gate_before_llm_selection():
    blueprint = build_routine_blueprint(
        target_split_label="legs",
        target_muscles=["quadriceps", "hamstring", "gluteal", "calves"],
        goal="hypertrophy",
        experience_level="beginner",
        readiness_level="normal",
        time_available_min=50,
    )
    candidates = [
        _catalog_candidate(
            "rare_variant",
            main_routine_default_allowed=0,
            beginner_default_allowed=0,
            exercise_familiarity_tier="obscure",
            mainstream_gym_score=10,
            score=120,
        ),
        _catalog_candidate("leg_press", score=100),
    ]

    bundles = build_blueprint_candidate_bundles(blueprint, candidates)
    main_bundle = next(bundle for bundle in bundles if bundle["phase"] == "main_compound")

    assert [item["exerciseId"] for item in main_bundle["candidates"]] == ["leg_press"]
    assert main_bundle["experienceRoleRejectedCount"] == 1


def test_safety_specific_lower_main_slot_accepts_reviewed_isolation_replacement():
    blueprint = build_routine_blueprint(
        target_split_label="legs",
        target_muscles=["quadriceps", "hamstring", "gluteal", "calves"],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=50,
        pain_areas=[{"bodyPart": "lower-back"}],
    )
    leg_extension = _catalog_candidate(
        "machine_leg_extension",
        name_kr="머신 레그 익스텐션",
        movement_type="ISOLATION",
        main_routine_default_allowed=1,
        beginner_default_allowed=1,
        injury_caution_level="low",
    )

    bundles = build_blueprint_candidate_bundles(blueprint, [leg_extension])
    main_bundle = next(bundle for bundle in bundles if bundle["slotId"] == "lower_main")

    assert [item["exerciseId"] for item in main_bundle["candidates"]] == [
        "machine_leg_extension"
    ]
    assert main_bundle["experienceRoleRejectedCount"] == 0


def test_strength_prescription_triplet_changes_reps_sets_and_rir_without_shortening_rest():
    beginner = resolve_experience_prescription(
        _plan(), "strength", "beginner", base_reps=5, base_rest_sec=180
    )
    intermediate = resolve_experience_prescription(
        _plan(), "strength", "intermediate", base_reps=5, base_rest_sec=180
    )
    advanced = resolve_experience_prescription(
        _plan(), "strength", "advanced", base_reps=5, base_rest_sec=180
    )

    assert [beginner.target_reps, intermediate.target_reps, advanced.target_reps] == [8, 5, 4]
    assert [beginner.sets, intermediate.sets, advanced.sets] == [2, 3, 3]
    assert [beginner.target_rir, intermediate.target_rir, advanced.target_rir] == [3, 2, 2]
    assert all(item.rest_sec >= 180 for item in (beginner, intermediate, advanced))


def test_low_readiness_overrides_advanced_normal_effort():
    output = LLMRoutineOutput(
        total_estimated_time=0,
        summary_title="테스트",
        rationale_summary=[],
        exercises=[_plan()],
    )

    adjusted = apply_deterministic_targets(
        output,
        goal="hypertrophy",
        recent_sets=[],
        profile=None,
        readiness_level="low",
        target_split_label="legs",
        target_muscles=["quadriceps"],
        experience_level="advanced",
    )

    assert adjusted.exercises[0].target_rir == 4
    assert adjusted.exercises[0].sets == 2


def test_high_readiness_never_turns_advanced_work_into_failure_training():
    output = LLMRoutineOutput(
        total_estimated_time=0,
        summary_title="테스트",
        rationale_summary=[],
        exercises=[_plan()],
    )

    adjusted = apply_deterministic_targets(
        output,
        goal="hypertrophy",
        recent_sets=[],
        profile=None,
        readiness_level="high",
        target_split_label="legs",
        target_muscles=["quadriceps"],
        experience_level="advanced",
    )

    assert adjusted.exercises[0].target_rir == 2

"""Load evidence is specific to the exercise and prescription, not a muscle."""
import math

import pytest

from engines.prescription.weight import _resolve_target_weight
from engines.schemas import LLMExercisePlan, RecentSetRecord, UserProfileContext
from engines.specialization_policy import apply_training_method_to_plan


def plan(**changes):
    values = dict(exercise_id="press", exercise_name="벤치 프레스", primary_muscles=["chest"],
                  equipment_type="BARBELL", movement_type="COMPOUND", target_reps=5,
                  sets=3, rest_time_sec=180, target_rir=2, exercise_rationale="가슴 운동")
    return LLMExercisePlan(**(values | changes))


def record(**changes):
    values = dict(exercise_id="press", exercise_name="벤치 프레스", primary_muscle="chest",
                  weight_kg=100, reps=5, rir=2, is_failure=False)
    return RecentSetRecord(**(values | changes))


@pytest.mark.parametrize("goal", ["strength", "hypertrophy", "endurance", "fatLoss", "generalFitness"])
def test_exact_prescription_is_unchanged_reference_not_goal_multiplier(goal):
    exercise = plan()
    assert _resolve_target_weight(exercise, goal, [record(weight_kg=101.25)], None) == 101.25
    assert "시작 참고값" in exercise.exercise_rationale
    assert "100%" not in exercise.exercise_rationale
    assert "기구" in exercise.exercise_rationale


@pytest.mark.parametrize("changes", [dict(exercise_id="fly"), dict(exercise_id=None),
    dict(exercise_id="PRESS"), dict(reps=8), dict(rir=3), dict(rir=1), dict(rir=None),
    dict(is_failure=True), dict(weight_kg=0), dict(weight_kg=-1), dict(weight_kg=math.inf),
    dict(weight_kg=math.nan), dict(rir=math.nan)])
def test_missing_or_incompatible_evidence_does_not_seed_weight(changes):
    exercise = plan(target_weight_kg=500)
    assert _resolve_target_weight(exercise, "strength", [record(**changes)], None) is None
    assert "첫 세트" in exercise.exercise_rationale


def test_names_do_not_override_ids_or_make_alias_identity():
    assert _resolve_target_weight(plan(), "strength", [record(exercise_id="another")], None) is None
    assert _resolve_target_weight(plan(), "strength", [record(exercise_name="스냅샷 명칭 변경")], None) == 100


@pytest.mark.parametrize("latest", [dict(rir=3), dict(is_failure=True), dict(weight_kg=None), dict(reps=8)])
def test_latest_exact_record_does_not_fall_back_to_more_attractive_older_record(latest):
    history = [record(**latest), record(weight_kg=140)]
    assert _resolve_target_weight(plan(), "strength", history, None) is None


@pytest.mark.parametrize("baseline", [
    {"press": {"weight_kg": 100, "reps": 1}},
    {"30": {"exercise_id": "30", "weight_kg": 100, "reps": 1}},
    {"press": {"weight_kg": "invalid", "reps": "invalid"}},
])
def test_baseline_is_not_a_working_load_or_a_cross_exercise_converter(baseline):
    profile = UserProfileContext(goal_type="strength", split_type="push", strength_baseline=baseline)
    assert _resolve_target_weight(plan(target_weight_kg=500), "strength", [], profile) is None


def test_same_muscle_history_and_model_hint_do_not_supply_kilograms():
    exercise = plan(exercise_id="dumbbell_fly", exercise_name="덤벨 플라이", equipment_type="DUMBBELL")
    assert _resolve_target_weight(exercise, "hypertrophy", [record()], None) is None
    assert _resolve_target_weight(plan(target_weight_kg=500), "strength", [], None) is None


@pytest.mark.parametrize("rir", [3, 4, 5])
def test_reserve_prescription_keeps_previously_locked_abstention(rir):
    assert _resolve_target_weight(plan(target_rir=rir), "strength", [record(rir=rir)], None) is None


@pytest.mark.parametrize("method", ["pausedReps", "tempoEccentric"])
def test_method_change_invalidates_numeric_reference_and_its_description(method):
    exercise = plan()
    exercise.target_weight_kg = _resolve_target_weight(exercise, "strength", [record()], None)
    assert exercise.target_weight_kg == 100
    apply_training_method_to_plan(exercise, method)
    assert exercise.target_weight_kg is None
    assert "시작 참고값" not in exercise.exercise_rationale
    assert _resolve_target_weight(exercise, "strength", [record()], None) is None


def test_guidance_is_replaced_when_prescription_changes_and_never_duplicates():
    exercise = plan()
    _resolve_target_weight(exercise, "strength", [record()], None)
    first = exercise.exercise_rationale
    _resolve_target_weight(exercise, "strength", [record()], None)
    assert first == exercise.exercise_rationale
    exercise.target_rir = 4
    _resolve_target_weight(exercise, "strength", [record()], None)
    assert "시작 참고값" not in exercise.exercise_rationale
    assert exercise.exercise_rationale.count("4회") == 1


def test_private_load_guidance_does_not_expand_public_contract():
    exercise = plan()
    _resolve_target_weight(exercise, "strength", [record()], None)
    assert not any("guidance" in key for key in exercise.model_dump())


@pytest.mark.parametrize("value", [True, "100", math.nan, math.inf])
def test_unvalidated_internal_records_do_not_bypass_numeric_check(value):
    bad = record().model_copy(update={"weight_kg": value})
    assert _resolve_target_weight(plan(), "strength", [bad], None) is None


@pytest.mark.parametrize("rir", [2, 4])
def test_lateral_reason_rewrite_preserves_load_reference_or_reserve_guidance(rir):
    from engines.fallback import build_routine_draft
    from engines.routine_pipeline import _enforce_response_fact_consistency
    from engines.schemas import LLMRoutineOutput, RoutineRequest

    exercise = plan(exercise_name="덤벨 레터럴 레이즈", primary_muscles=["front-deltoids"],
                    movement_type="ISOLATION", equipment_type="DUMBBELL", target_rir=rir)
    exercise.target_weight_kg = _resolve_target_weight(exercise, "strength", [record(rir=rir)], None)
    output = LLMRoutineOutput(total_estimated_time=15, summary_title="어깨 운동", rationale_summary=[],
                              warnings=[], exercises=[exercise])
    draft = build_routine_draft(output, "success", "none", False)
    req = RoutineRequest(time_available_min=30, target_split_label="shoulder")
    fixed = _enforce_response_fact_consistency(draft, req)
    block = fixed.routine_blocks[0]
    assert exercise._weight_guidance in block.exercise_rationale
    assert "측면 어깨" in block.exercise_rationale
    assert block.prescription == draft.routine_blocks[0].prescription
    assert not any("guidance" in key for key in block.model_dump())
    assert _enforce_response_fact_consistency(fixed, req).model_dump() == fixed.model_dump()


def test_actual_fallback_retains_exact_reference_through_final_reason_guard():
    from engines.fallback import generate_fallback_routine
    from engines.routine_pipeline import _enforce_response_fact_consistency
    from engines.schemas import RoutineRequest

    req = RoutineRequest(time_available_min=30, target_split_label="shoulder", experience_level="intermediate")
    candidates = [dict(id="lateral", name_kr="덤벨 레터럴 레이즈", name_en="Dumbbell Lateral Raise",
        primary_muscle="front-deltoids", equipment_req="DUMBBELL", movement_type="ISOLATION",
        difficulty_tier=2, efficiency_tier=4)]
    draft = generate_fallback_routine(req, candidates, max_total_sets=3, goal="hypertrophy")
    dose = draft.routine_blocks[0].prescription[0]
    history = [record(exercise_id="lateral", reps=dose.target_reps, rir=dose.target_rir, weight_kg=10)]
    draft = generate_fallback_routine(req, candidates, max_total_sets=3, goal="hypertrophy", recent_sets=history)
    fixed = _enforce_response_fact_consistency(draft, req)
    assert fixed.routine_blocks[0].prescription[0].target_weight_kg == 10
    assert "시작 참고값" in fixed.routine_blocks[0].exercise_rationale


def test_invalid_or_missing_target_effort_does_not_create_reference():
    for rir in [None, 0, -1, True, math.inf, math.nan]:
        exercise = plan().model_copy(update={"target_rir": rir})
        assert _resolve_target_weight(exercise, "strength", [record()], None) is None


@pytest.mark.parametrize("field,value", [("_target_duration_sec", 30), ("_target_distance_m", 20),
                                       ("_prescription_set_type", "cooldown")])
def test_non_repetition_dose_cannot_reuse_a_repetition_load_or_its_instruction(field, value):
    exercise = plan()
    _resolve_target_weight(exercise, "strength", [record()], None)
    setattr(exercise, field, value)
    assert _resolve_target_weight(exercise, "strength", [record()], None) is None
    assert exercise._weight_guidance is None
    assert exercise.exercise_rationale == "가슴 운동"


def test_adjunct_override_clears_previous_kg_reference_description():
    from engines.prescription.adjuncts import ADJUNCT_PRESCRIPTION_PROFILES, apply_adjunct_prescription_profile

    for option in ADJUNCT_PRESCRIPTION_PROFILES:
        exercise = plan()
        exercise.target_weight_kg = _resolve_target_weight(exercise, "strength", [record()], None)
        assert "시작 참고값" in exercise.exercise_rationale
        assert apply_adjunct_prescription_profile(exercise, option)
        assert exercise.target_weight_kg is None
        assert exercise._weight_guidance is None
        assert "시작 참고값" not in exercise.exercise_rationale


def test_uncalibrated_guidance_stays_short_enough_for_multi_exercise_coaching():
    exercise = plan()
    _resolve_target_weight(exercise, "hypertrophy", [], None)
    assert len(exercise._weight_guidance) <= 80

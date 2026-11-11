"""Regression tests for the 2026-11-08 live quality review (routine side).

Each test pins one measured failure: custom split → empty routine, profile pain ignored,
duplicate picks, single-muscle push days, 1-set tails, and implausible borrowed weights.
"""
from unittest.mock import MagicMock, patch

from sqlalchemy.orm import Session

from engines.fallback import generate_fallback_routine
from engines.llm_parser import _fit_set_budget, validate_and_repair_routine_output
from engines.prescription.weight import _resolve_target_weight
from engines.routine_pipeline import generate_smart_routine
from engines.schemas import (
    LLMExercisePlan,
    LLMRoutineOutput,
    PainAreaEntry,
    RecentSetRecord,
    RoutineRequest,
    UserProfileContext,
)


def _cand(cid, muscle, equipment="BARBELL", movement="COMPOUND", triggers=None):
    return {
        "id": cid, "name_kr": cid, "name_en": cid, "primary_muscle": muscle, "secondary_muscle": None,
        "equipment_req": equipment, "difficulty_tier": 2, "efficiency_tier": 3,
        "pain_triggers": triggers, "movement_type": movement,
    }


def _plan(cid, muscle, sets=3, equipment="BARBELL", movement="COMPOUND", reps=8):
    return LLMExercisePlan(
        exercise_id=cid, exercise_name=cid, movement_type=movement, primary_muscles=[muscle],
        equipment_type=equipment, target_reps=reps, sets=sets, rest_time_sec=90, target_rir=2,
        exercise_rationale="r",
    )


def _output(*plans):
    return LLMRoutineOutput(total_estimated_time=40, summary_title="t", rationale_summary=["r"],
                            warnings=[], exercises=list(plans))


# --- 1. custom split ---------------------------------------------------------

def test_custom_split_uses_target_muscles():
    req = RoutineRequest(target_split_label="custom", target_muscles=["biceps", "triceps"], time_available_min=45)
    with patch("engines.routine_pipeline.get_candidate_exercises", return_value=[]) as fetch,             patch("engines.routine_pipeline.get_llm", side_effect=RuntimeError("no llm in unit tests")):
        result = generate_smart_routine(req, MagicMock(spec=Session), profile=None, recent_sets=[])
    assert fetch.call_args.args[1] == ["biceps", "triceps"]


def test_known_split_label_still_wins():
    req = RoutineRequest(target_split_label="push", target_muscles=["biceps"], time_available_min=45)
    with patch("engines.routine_pipeline.get_candidate_exercises", return_value=[]) as fetch,             patch("engines.routine_pipeline.get_llm", side_effect=RuntimeError("no llm in unit tests")):
        generate_smart_routine(req, MagicMock(spec=Session), profile=None, recent_sets=[])
    assert fetch.call_args.args[1] == ["chest", "front-deltoids", "triceps"]


# --- 5. profile pain is a hard filter ---------------------------------------

def test_profile_pain_areas_reach_candidate_filter():
    req = RoutineRequest(target_split_label="core", time_available_min=30)
    profile = UserProfileContext(goal_type="strength", split_type="ppl",
                                 pain_areas=[PainAreaEntry(body_part="lower-back")])
    with patch("engines.routine_pipeline.get_candidate_exercises", return_value=[]) as fetch,             patch("engines.routine_pipeline.get_llm", side_effect=RuntimeError("no llm in unit tests")):
        generate_smart_routine(req, MagicMock(spec=Session), profile=profile, recent_sets=[])
    assert [p.body_part for p in fetch.call_args.args[3]] == ["lower-back"]


def test_fallback_rationale_names_the_real_reason():
    req = RoutineRequest(target_split_label="push", time_available_min=30)
    cands = [_cand("bench", "chest"), _cand("pushdown", "triceps", "CABLE", "ISOLATION")]
    schema = generate_fallback_routine(req, cands, 10, status_reason_code="schemaError")
    network = generate_fallback_routine(req, cands, 10, status_reason_code="networkError")
    assert "연결" not in schema.rationale_summary[0]
    assert "연결" in network.rationale_summary[0]


def test_fallback_respects_time_budget():
    req = RoutineRequest(target_split_label="push", time_available_min=20)
    cands = [_cand(f"c{i}", "chest") for i in range(8)]
    result = generate_fallback_routine(req, cands, 30)
    assert result.total_estimated_time <= 20


# --- 4. duplicates, coverage, set budget -----------------------------------

def test_duplicate_pick_is_replaced_with_unused_candidate():
    cands = [_cand("a", "chest"), _cand("b", "chest"), _cand("c", "triceps", "CABLE", "ISOLATION")]
    result, reason = validate_and_repair_routine_output(
        _output(_plan("a", "chest"), _plan("a", "chest")), cands, [], [])
    assert reason == "none"
    ids = [e.exercise_id for e in result.exercises]
    assert len(ids) == len(set(ids))


def test_duplicate_pick_dropped_when_no_spare_candidate():
    cands = [_cand("a", "chest"), _cand("b", "triceps", "CABLE", "ISOLATION")]
    result, reason = validate_and_repair_routine_output(
        _output(_plan("a", "chest"), _plan("b", "triceps"), _plan("b", "triceps")), cands, [], [])
    assert reason == "none"
    assert [e.exercise_id for e in result.exercises] == ["a", "b"]


def test_push_day_covers_triceps_and_shoulders():
    cands = [_cand(f"chest{i}", "chest") for i in range(6)] + [
        _cand("ohp", "front-deltoids"), _cand("pushdown", "triceps", "CABLE", "ISOLATION")]
    plans = [_plan(f"chest{i}", "chest") for i in range(6)]
    result, reason = validate_and_repair_routine_output(
        _output(*plans), cands, [], [], target_muscles=["chest", "front-deltoids", "triceps"])
    assert reason == "none"
    muscles = {e.primary_muscles[0] for e in result.exercises}
    assert {"chest", "front-deltoids", "triceps"} <= muscles


def test_set_budget_drops_tail_before_one_set_exercises():
    routine = _output(*[_plan(f"e{i}", "chest", sets=3) for i in range(6)])
    assert _fit_set_budget(routine, 8)
    assert sum(e.sets for e in routine.exercises) <= 8
    assert min(e.sets for e in routine.exercises) >= 2


# --- 3. weights --------------------------------------------------------------

def test_borrowed_weight_is_scaled_for_dumbbells():
    recent = [RecentSetRecord(exercise_id="132", exercise_name="씰 로우", primary_muscle="upper-back",
                              weight_kg=120, reps=5, rir=2, equipment_type="BARBELL, MACHINE",
                              movement_type="COMPOUND")]
    row = _plan("58", "upper-back", equipment="DUMBBELL", reps=5)
    weight = _resolve_target_weight(row, "strength", recent, None)
    assert weight is not None and weight <= 45  # was 100kg per dumbbell


def test_big_four_baseline_scaled_for_dumbbell_press():
    profile = UserProfileContext(goal_type="strength", split_type="ppl", strength_baseline={
        "30": {"exercise_id": "30", "weight_kg": 125, "reps": 1}})
    press = _plan("34", "chest", equipment="DUMBBELL", reps=5)
    machine = _plan("37", "chest", equipment="MACHINE", reps=5)
    assert _resolve_target_weight(press, "strength", [], profile) <= 40  # was 80kg per dumbbell
    assert _resolve_target_weight(machine, "strength", [], profile) > 60


def test_exact_history_never_regresses_for_same_reps():
    recent = [RecentSetRecord(exercise_id="30", exercise_name="바벨 벤치프레스", primary_muscle="chest",
                              weight_kg=100, reps=8, rir=2)]
    bench = _plan("30", "chest", reps=8)
    assert _resolve_target_weight(bench, "hypertrophy", recent, None) >= 100


def test_loadable_bodyweight_option_gets_weight():
    recent = [RecentSetRecord(exercise_id="102", exercise_name="불가리안 스플릿 스쿼트",
                              primary_muscle="quadriceps", weight_kg=20, reps=10, rir=2)]
    split_squat = _plan("102", "quadriceps", equipment="DUMBBELL, BARBELL, BODYWEIGHT", reps=10)
    assert _resolve_target_weight(split_squat, "hypertrophy", recent, None) is not None
    pure = _plan("47", "chest", equipment="BODYWEIGHT")
    assert _resolve_target_weight(pure, "hypertrophy", [], None) is None


# --- 2026-11-09 follow-up: pool diversity and under-budget fill ---------------

def test_candidate_pool_keeps_slots_for_every_push_muscle():
    from engines.candidate_ranker import score_candidate_exercises
    cands = [_cand(f"chest{i}", "chest") for i in range(20)] + [
        _cand(f"delt{i}", "front-deltoids", "DUMBBELL", "ISOLATION") for i in range(5)] + [
        _cand(f"tri{i}", "triceps", "CABLE", "ISOLATION") for i in range(5)]
    pool = score_candidate_exercises(cands, ["chest", "front-deltoids", "triceps"], top_n=12)
    muscles = [c["primary_muscle"] for c in pool]
    assert len(pool) == 12
    assert muscles.count("front-deltoids") >= 2 and muscles.count("triceps") >= 2


def test_candidate_pool_does_not_reserve_minor_muscles_for_broad_splits():
    from engines.candidate_ranker import score_candidate_exercises
    legs = ["quadriceps", "hamstring", "gluteal", "calves", "adductor", "abductors"]
    cands = [_cand(f"{m}{i}", m) for m in legs for i in range(4)]
    pool = score_candidate_exercises(cands, legs, top_n=12)
    assert sum(c["primary_muscle"] in {"calves", "adductor", "abductors"} for c in pool) < 6


def test_under_budget_routine_is_filled_from_unused_candidates():
    cands = [_cand("crunch", "abs", "BODYWEIGHT", "ISOLATION"),
             _cand("plank", "abs", "BODYWEIGHT", "STATIC"),
             _cand("twist", "obliques", "BODYWEIGHT", "ISOLATION")]
    plan = _plan("crunch", "abs", sets=2, equipment="BODYWEIGHT", movement="ISOLATION")
    result, reason = validate_and_repair_routine_output(
        _output(plan), cands, [], [], time_available_min=20, target_muscles=["abs", "obliques"])
    assert reason == "none"
    from engines.prescription.estimator import estimate_routine_time_min
    assert len(result.exercises) > 1
    assert estimate_routine_time_min(result.exercises) <= 20
    assert len({e.exercise_id for e in result.exercises}) == len(result.exercises)


def test_empty_candidates_skip_llm_and_fail_cleanly():
    req = RoutineRequest(target_split_label="push", time_available_min=30)
    llm_factory = MagicMock()
    with patch("engines.routine_pipeline.get_candidate_exercises", return_value=[]), \
         patch("engines.routine_pipeline.get_llm", llm_factory):
        result = generate_smart_routine(req, MagicMock(spec=Session), profile=None, recent_sets=[])
    llm_factory.assert_not_called()
    assert result.generation_status == "failed"
    assert result.status_reason_code == "emptyCandidate"


def test_low_readiness_keeps_short_session():
    cands = [_cand("crunch", "abs", "BODYWEIGHT", "ISOLATION"),
             _cand("plank", "abs", "BODYWEIGHT", "STATIC")]
    plan = _plan("crunch", "abs", sets=2, equipment="BODYWEIGHT", movement="ISOLATION")
    result, reason = validate_and_repair_routine_output(
        _output(plan), cands, [], [], time_available_min=20, target_muscles=["abs"], readiness_level="low")
    assert reason == "none"
    assert [e.exercise_id for e in result.exercises] == ["crunch"]


def test_multi_implement_history_is_read_as_target_implement():
    recent = [RecentSetRecord(exercise_id="18", exercise_name="스파이더 컬", primary_muscle="biceps",
                              weight_kg=15, reps=15, rir=2, equipment_type="BARBELL, DUMBBELL, CABLE",
                              movement_type="ISOLATION")]
    curl = _plan("12", "biceps", equipment="DUMBBELL", movement="ISOLATION", reps=8)
    weight = _resolve_target_weight(curl, "hypertrophy", recent, None)
    assert weight is not None and weight >= 12.5  # was 5.0 (history read as barbell)


def test_custom_fallback_title_is_korean():
    req = RoutineRequest(target_split_label="custom", target_muscles=["biceps"], time_available_min=30)
    result = generate_fallback_routine(req, [_cand("curl", "biceps", "DUMBBELL", "ISOLATION")], 10)
    assert "custom" not in result.summary_title


def test_fill_prefers_uncovered_target_muscle():
    cands = [_cand("glute1", "gluteal"), _cand("glute2", "gluteal"), _cand("ham1", "hamstring"),
             _cand("quad1", "quadriceps")]
    plan = _plan("glute1", "gluteal", sets=2)
    result, reason = validate_and_repair_routine_output(
        _output(plan), cands, [], [], time_available_min=45,
        target_muscles=["quadriceps", "hamstring", "gluteal", "calves", "adductor", "abductors"])
    assert reason == "none"
    assert "quadriceps" in {e.primary_muscles[0] for e in result.exercises}


def test_pain_only_slug_knees_is_translated_in_user_text():
    from engines.korean_text_guard import translate_code_tokens
    assert "knees" not in translate_code_tokens("무릎(knees) 부담이 적은 운동").lower()
    assert "overhead" in translate_code_tokens("overhead press 자세")  # no partial-word replacement

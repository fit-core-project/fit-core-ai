"""Explicit focus changes role allocation, never health or session limits."""
import pytest

from engines.prescription.targets import apply_deterministic_targets
from engines.prescription.adjustments import _fill_available_time
from engines.prescription.estimator import estimate_routine_time_min
from engines.routine_blueprint import build_routine_blueprint
from engines.schemas import LLMExercisePlan, LLMRoutineOutput
from engines.specialization_policy import (
    apply_blueprint_specialization_volume, primary_focus_candidate_ids,
)


def plan(exercise_id, muscle, sets=3, movement="ISOLATION"):
    return LLMExercisePlan(
        exercise_id=exercise_id, exercise_name=exercise_id,
        primary_muscles=[muscle], movement_type=movement,
        sets=sets, target_reps=10, rest_time_sec=90, target_rir=2,
        exercise_rationale="fixture",
    )


def output(*plans):
    return LLMRoutineOutput(total_estimated_time=0, summary_title="fixture",
                            rationale_summary=[], exercises=list(plans))


def prescribe(**kwargs):
    options = dict(goal="hypertrophy", recent_sets=None, profile=None,
                   experience_level="advanced", target_split_label="legs",
                   target_muscles=["quadriceps", "calves"], time_available_min=60,
                   primary_focus_exercise_ids={"CALF"})
    options.update(kwargs)
    return apply_deterministic_targets(output(plan("calf", "calves")), **options).exercises[0]


def test_explicit_calf_focus_is_not_reduced_to_generic_accessory():
    assert prescribe().sets == 3
    assert prescribe(primary_focus_exercise_ids=set()).sets == 2


@pytest.mark.parametrize("experience", ["beginner", "intermediate"])
def test_primary_focus_override_is_advanced_only(experience):
    assert prescribe(experience_level=experience).sets == 2


@pytest.mark.parametrize("kwargs,sets,rir", [
    ({"readiness_level": "low"}, 2, 4),
    ({"doms_level_by_exercise_id": {"calf": 1}}, 2, 3),
    # Retain the historical ID while replacing the v1 global-half-volume oracle.
    pytest.param({"doms_level_by_exercise_id": {"calf": 2}}, 2, 4, id="kwargs2-1-4"),
    ({"secondary_doms_level_by_exercise_id": {"calf": 2}}, 1, 3),
])
def test_health_restrictions_override_focus(kwargs, sets, rir):
    result = prescribe(**kwargs)
    assert result.sets == sets
    assert result.target_rir >= rir


def test_focus_fill_preserves_budget_and_duration():
    result = output(plan("main", "quadriceps", 3, "COMPOUND"), plan("calf", "calves", 2))
    _fill_available_time(result, 60, "hypertrophy", "normal", "legs",
                         experience_level="advanced", max_total_sets=6,
                         primary_focus_exercise_ids={"CALF"})
    assert [p.sets for p in result.exercises] == [3, 3]
    assert sum(p.sets for p in result.exercises) == 6
    assert estimate_routine_time_min(result.exercises) <= 60


def transfer(plans, focus="hamstringEmphasis", restricted=None):
    blueprint = build_routine_blueprint(
        target_split_label="legs", target_muscles=[], goal="hypertrophy",
        experience_level="advanced", readiness_level="normal",
        time_available_min=60, specialization_focus=focus,
    )
    result = output(*plans)
    result._selection_slot_by_exercise_id = {
        "focus": "posterior_chain", "main": "lower_main", "calf": "calves_accessory",
    }
    candidates = [dict(id=p.exercise_id, primary_muscle=p.primary_muscles[0],
                       movement_type=p.movement_type) for p in plans]
    return apply_blueprint_specialization_volume(
        result, blueprint, candidates, restricted_exercise_ids=restricted,
    )


def test_focus_transfer_preserves_main_and_accessory_floors():
    result = transfer([plan("main", "quadriceps", 4, "COMPOUND"),
                       plan("focus", "hamstring", 3), plan("calf", "calves", 2)])
    assert [p.sets for p in result.exercises] == [3, 4, 2]


def test_focus_transfer_does_not_take_from_three_set_main():
    result = transfer([plan("main", "quadriceps", 3, "COMPOUND"),
                       plan("focus", "hamstring", 3), plan("calf", "calves", 2)])
    assert [p.sets for p in result.exercises] == [3, 3, 2]


def test_focus_transfer_leaves_restricted_donor_unchanged():
    result = transfer([plan("main", "quadriceps", 4, "COMPOUND"),
                       plan("focus", "hamstring", 3)], restricted={"main"})
    assert [p.sets for p in result.exercises] == [4, 3]


def test_transfer_between_same_focus_plans_is_not_reported_as_more_direct_volume():
    result = transfer([plan("focus", "hamstring", 3), plan("other", "hamstring", 4)])
    assert [p.sets for p in result.exercises] == [3, 4]


@pytest.mark.parametrize("readiness,expected", [("normal", {"calf"}), ("low", set())])
def test_focus_candidate_ids_use_existing_matcher_and_reps_only(readiness, expected):
    blueprint = build_routine_blueprint(
        target_split_label="legs", target_muscles=[], goal="hypertrophy",
        experience_level="advanced", readiness_level=readiness,
        time_available_min=60, specialization_focus="calvesEmphasis",
    )
    candidates = [
        dict(id="CALF", primary_muscle="calves", prescription_mode="reps"),
        dict(id="quad", primary_muscle="quadriceps"),
        dict(id="stretch", primary_muscle="calves", prescription_mode="duration"),
    ]
    assert primary_focus_candidate_ids(blueprint, candidates) == expected
    assert primary_focus_candidate_ids(None, candidates) == set()


def test_short_time_does_not_inflate_focus_fill():
    result = output(plan("main", "quadriceps", 3, "COMPOUND"), plan("calf", "calves", 2))
    before = estimate_routine_time_min(result.exercises)
    _fill_available_time(result, before, "hypertrophy", "normal", "legs",
                         experience_level="advanced", primary_focus_exercise_ids={"calf"})
    assert [p.sets for p in result.exercises] == [3, 2]


def test_fallback_focus_matches_normal_prescription_and_transfer():
    from engines.fallback import generate_fallback_routine
    from engines.schemas import RoutineRequest

    candidates = [
        dict(id="main", name_kr="레그 프레스", name_en="Leg Press",
             primary_muscle="quadriceps", movement_type="COMPOUND"),
        dict(id="curl", name_kr="머신 레그 컬", name_en="Machine Leg Curl",
             primary_muscle="hamstring", movement_type="ISOLATION"),
        dict(id="calf", name_kr="머신 카프 레이즈", name_en="Machine Calf Raise",
             primary_muscle="calves", movement_type="ISOLATION"),
    ]
    for candidate in candidates:
        candidate.update(secondary_muscle="", equipment_req="MACHINE", difficulty_tier=2,
                         efficiency_tier=3, quality_score=80, strength_effect="high",
                         hypertrophy_effect="high", failure_penalty="low",
                         mainstream_gym_score=80, main_routine_default_allowed=1,
                         prescription_mode="reps")
    args = dict(target_split_label="legs", target_muscles=[], goal="hypertrophy",
                experience_level="advanced", readiness_level="normal",
                time_available_min=60, specialization_focus="calvesEmphasis")
    blueprint = build_routine_blueprint(**args)
    request = RoutineRequest(user_id="synthetic-focus", equipment=[], **args)
    response = generate_fallback_routine(
        request, candidates=candidates, max_total_sets=20, goal="hypertrophy",
        target_exercise_count=3, minimum_main_exercise_count=3, routine_blueprint=blueprint,
    )
    normal = apply_deterministic_targets(
        output(*(plan(c["id"], c["primary_muscle"], movement=c["movement_type"]) for c in candidates)),
        goal="hypertrophy", recent_sets=None, profile=None, experience_level="advanced",
        target_split_label="legs", target_muscles=["quadriceps", "hamstring", "calves"],
        time_available_min=60, primary_focus_exercise_ids={"calf"},
    )
    normal._selection_slot_by_exercise_id = {
        "main": "lower_main", "curl": "posterior_chain", "calf": "calves_accessory",
    }
    apply_blueprint_specialization_volume(normal, blueprint, candidates)
    assert response.generation_status == "fallback"
    fallback_sets = {b.exercise_id: len(b.prescription) for b in response.routine_blocks}
    assert fallback_sets == {p.exercise_id: p.sets for p in normal.exercises}
    assert fallback_sets["calf"] == 4
    assert fallback_sets["main"] >= 3

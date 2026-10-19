"""Independent enumeration checks for the bounded time-admission ceiling."""
from itertools import product

import pytest

from engines.prescription.estimator import estimate_routine_time_min
from engines.prescription.role_budget import role_time_set_ceiling
from engines.schemas import LLMExercisePlan


def plan(i, kind="COMPOUND", sets=5, rest=None):
    return LLMExercisePlan(exercise_id=str(i), exercise_name=str(i),
        movement_type=kind, primary_muscles=["chest" if kind == "COMPOUND" else "triceps"],
        sets=sets, target_reps=5, rest_time_sec=(180 if kind == "COMPOUND" else 90) if rest is None else rest,
        exercise_rationale="Synthetic arithmetic, not clinical guidance")


def brute_force(plans, minutes, count):
    best = 0
    for allocation in product(*(range(p.sets + 1) for p in plans)):
        selected = [p.model_copy(update={"sets": n}, deep=True)
                    for p, n in zip(plans, allocation) if n]
        if len(selected) <= count and estimate_routine_time_min(selected) <= minutes:
            best = max(best, sum(allocation))
    return best


def test_mixed_roles_are_not_all_compound_sets():
    roles = [plan(1), plan(2), plan(3, "ISOLATION", 2), plan(4, "ISOLATION", 2)]
    assert estimate_routine_time_min(roles) == 51
    assert role_time_set_ceiling(roles, 60, 4) == 14


@pytest.mark.parametrize("minutes", [5, 10, 20, 35, 60, 75])
@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_ceiling_matches_independent_complete_allocation_enumeration(minutes, count):
    roles = [plan(1, sets=3), plan(2, sets=4), plan(3, "ISOLATION", 2), plan(4, "ISOLATION", 3)]
    assert role_time_set_ceiling(roles, minutes, count) == brute_force(roles, minutes, count)


def test_cannot_accept_twenty_one_compound_heavy_sets_in_sixty_minutes():
    roles = [plan(i) for i in range(3)] + [plan(i, "ISOLATION", 2) for i in range(3, 6)]
    assert estimate_routine_time_min(roles) == 75
    assert role_time_set_ceiling(roles, 60, 6) < 21
    assert role_time_set_ceiling(roles, 75, 6) == 21


def test_count_bound_still_applies_to_fast_isolations():
    roles = [plan(i, "ISOLATION", 3) for i in range(7)]
    assert role_time_set_ceiling(roles, 60, 4) == 12


def test_grouped_bound_does_not_claim_a_group_is_feasible():
    roles = [plan(i, sets=3) for i in range(4)]
    assert role_time_set_ceiling(roles, 30, 4, grouped=True) == 12
    assert role_time_set_ceiling(roles, 30, 4) < 12
    # Group/equipment policy and exact group timing must still validate the output.
    assert all(p.rest_time_sec == 180 for p in roles)


def test_role_caps_and_inputs_are_not_mutated():
    roles = [plan(1, sets=2), plan(2, "ISOLATION", 1)]
    before = [p.model_dump() for p in roles]
    assert role_time_set_ceiling(roles, 120, 4) == 3
    assert before == [p.model_dump() for p in roles]


def test_empty_and_zero_capacity():
    assert role_time_set_ceiling([], 60, 4) == 0
    assert role_time_set_ceiling([plan(1)], 60, 0) == 0


def test_pause_time_is_counted():
    regular = plan(1)
    paused = regular.model_copy(deep=True)
    paused._training_method_instruction = {"pauseSec": 3}
    assert role_time_set_ceiling([regular], 10, 1) == 2
    assert role_time_set_ceiling([paused], 10, 1) == 1


def test_duplicate_ids_do_not_create_extra_capacity():
    with pytest.raises(ValueError, match="duplicate"):
        role_time_set_ceiling([plan(1), plan(1)], 60, 4)


def candidates():
    return [dict(id=str(i), name_kr=str(i), movement_type=kind, primary_muscle=muscle,
                 secondary_muscle="", equipment_req="DUMBBELL", prescription_mode="reps")
            for i, kind, muscle in [(1, "COMPOUND", "chest"), (2, "COMPOUND", "upper-back"),
                                    (3, "ISOLATION", "triceps"), (4, "ISOLATION", "biceps")]]


def context(**kwargs):
    from engines.schemas import RoutineRequest
    from engines.routine_blueprint import build_routine_blueprint
    req = RoutineRequest(time_available_min=60, goal="strength", target_split_label="upper",
                         target_muscles=["chest", "upper-back", "triceps", "biceps"],
                         experience_level=kwargs.pop("experience_level", "advanced"), **kwargs)
    blueprint = build_routine_blueprint(
        target_split_label=req.target_split_label, target_muscles=req.target_muscles,
        time_available_min=req.time_available_min, goal=req.goal,
        experience_level=req.experience_level, readiness_level=req.readiness_level,
        session_structure=req.session_structure,
    )
    return req, blueprint


def test_candidate_budget_reuses_existing_roles_without_mutating_blueprint():
    from engines.routine_pipeline import _candidate_role_set_budget
    req, blueprint = context()
    assert blueprint.max_total_sets == 13
    assert _candidate_role_set_budget(req, blueprint, candidates(), {}) == 14
    assert blueprint.max_total_sets == 13
    assert blueprint.target_main_exercise_count == 5


@pytest.mark.parametrize("level,expected_main_cap", [("beginner", 3), ("intermediate", 4), ("advanced", 5)])
def test_candidate_capacity_cannot_unlock_experience_caps(monkeypatch, level, expected_main_cap):
    from engines import routine_pipeline
    observed = []
    original = routine_pipeline.role_time_set_ceiling
    def capture(roles, *args, **kwargs):
        observed.extend(roles)
        return original(roles, *args, **kwargs)
    monkeypatch.setattr(routine_pipeline, "role_time_set_ceiling", capture)
    req, blueprint = context(experience_level=level)
    routine_pipeline._candidate_role_set_budget(req, blueprint, candidates(), {})
    assert [p.sets for p in observed] == [expected_main_cap, expected_main_cap, 2, 2]
    assert all(p.rest_time_sec == 180 for p in observed[:2])


@pytest.mark.parametrize("doms", [1, 2, 3])
def test_candidate_capacity_retains_doms_and_does_not_reintroduce_excluded_roles(monkeypatch, doms):
    from engines import routine_pipeline
    observed = []
    original = routine_pipeline.role_time_set_ceiling
    def capture(roles, *args, **kwargs):
        observed.extend(roles)
        return original(roles, *args, **kwargs)
    monkeypatch.setattr(routine_pipeline, "role_time_set_ceiling", capture)
    req, blueprint = context()
    routine_pipeline._candidate_role_set_budget(req, blueprint, candidates(), {"chest": doms})
    chest = next((p for p in observed if p.exercise_id == "1"), None)
    if doms == 3:
        assert chest is None
    else:
        assert chest.sets <= (4 if doms == 1 else 3)
        assert chest.target_rir >= (3 if doms == 1 else 4)


def test_low_readiness_capacity_does_not_restore_normal_sets(monkeypatch):
    from engines import routine_pipeline
    observed = []
    original = routine_pipeline.role_time_set_ceiling
    def capture(roles, *args, **kwargs):
        observed.extend(roles)
        return original(roles, *args, **kwargs)
    monkeypatch.setattr(routine_pipeline, "role_time_set_ceiling", capture)
    req, blueprint = context(readiness_level="low")
    routine_pipeline._candidate_role_set_budget(req, blueprint, candidates(), {})
    assert [p.sets for p in observed[:2]] == [4, 4]
    assert all(p.target_rir >= 4 for p in observed)


def test_candidate_budget_deduplicates_and_ignores_timed_work():
    from engines.routine_pipeline import _candidate_role_set_budget
    req, blueprint = context()
    rows = candidates()
    assert _candidate_role_set_budget(req, blueprint, rows + rows, {}) == 14
    rows[3]["prescription_mode"] = "duration"
    assert _candidate_role_set_budget(req, blueprint, rows, {}) == 13


@pytest.mark.parametrize("actual_minutes", [1, 20])
def test_time_adjustment_warnings_are_unique_even_without_underfill(actual_minutes):
    from engines.routine_pipeline import _apply_underfill_explanation
    from engines.schemas import RoutineBlock, RoutineDraftResponse, SetPrescription
    req, blueprint = context()
    req = req.model_copy(update={"time_available_min": 20})
    caution = "Stop if pain occurs."
    time_note = "Working sets were adjusted to the time budget."
    response = RoutineDraftResponse(
        generation_status="success", status_reason_code="none", is_fallback=False,
        total_estimated_time=actual_minutes, summary_title="test", rationale_summary=[],
        warnings=[time_note, caution, caution, time_note],
        routine_blocks=[RoutineBlock(order=1, exercise_id="1", exercise_name="test",
            default_rest_sec=90, exercise_rationale="test",
            prescription=[SetPrescription(set_index=1, set_type="working", target_reps=10,
                                          target_rest_sec=90)])],
    )
    result = _apply_underfill_explanation(response, req, [], blueprint, {})
    assert result.warnings[:2] == [time_note, caution]
    assert len(result.warnings) == len(set(result.warnings))
    assert result.total_estimated_time == actual_minutes

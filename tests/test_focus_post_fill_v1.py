from dataclasses import replace

import pytest

from engines.prescription.focus_capacity import complete_lower_chest_focus_capacity
from engines.prescription.estimator import estimate_routine_time_min
from engines.routine_blueprint import build_routine_blueprint
from engines.schemas import LLMExercisePlan, LLMRoutineOutput, RoutineRequest


def setup_case(**overrides):
    values = dict(user_id="synthetic-focus-capacity", time_available_min=60,
                  target_split_label="chest", target_muscles=["chest"], goal="hypertrophy",
                  experience_level="advanced", readiness_level="normal",
                  specialization_focus="lowerChest", method_mode="off")
    values.update(overrides)
    if values["experience_level"] != "advanced":
        values["specialization_focus"] = None
    req = RoutineRequest(**values)
    bp = build_routine_blueprint(
        target_split_label=req.target_split_label, target_muscles=req.target_muscles,
        goal=req.goal, experience_level=req.experience_level, readiness_level=req.readiness_level,
        specialization_focus=req.specialization_focus, time_available_min=req.time_available_min,
        session_structure=req.session_structure, method_mode=req.method_mode,
        adjunct_options=req.adjunct_options,
    )
    plans = [LLMExercisePlan(
        exercise_id=eid, exercise_name=eid, primary_muscles=["chest"],
        movement_type="ISOLATION" if eid == "fly" else "COMPOUND",
        target_reps=10 if eid == "fly" else 8, sets=count,
        rest_time_sec=75 if eid == "fly" else 120, target_rir=2,
        exercise_rationale="synthetic",
    ) for eid, count in (("decline", 5), ("flat", 3), ("fly", 3))]
    output = LLMRoutineOutput(total_estimated_time=0, summary_title="synthetic",
                              rationale_summary=[], exercises=plans)
    output._selection_slot_by_exercise_id = {
        "decline": "chest_main", "flat": "chest_secondary", "fly": "chest_isolation"}
    candidates = [dict(id="fly", core_chest_focus_reviewed=1, core_admission_scope="lowerChest")]
    return output, bp, req, candidates, {"decline": 4, "flat": 4, "fly": 3}


def apply(case, *, doms=None, restricted=None, cap=22):
    return complete_lower_chest_focus_capacity(
        *case, doms_db=doms or {}, restricted_exercise_ids=restricted or set(), max_total_sets=cap)


def sets(output):
    return [p.sets for p in output.exercises]


def test_refill_restores_only_donor_with_existing_cap_and_focus_preserved():
    case = setup_case()
    output = case[0]
    original = output.model_dump()
    diag = apply(case)
    assert sets(output) == [5, 4, 3]
    assert diag["role_set_capacity_exhausted"] is True
    assert diag["role_set_capacity_total_budget_exhausted"] is False
    assert output._selection_diagnostics["focus_capacity_allocation"]["addedSets"] == 1
    assert estimate_routine_time_min(output.exercises) < 60
    for old, new in zip(original["exercises"], output.model_dump()["exercises"]):
        assert {k: v for k, v in old.items() if k != "sets"} == {
            k: v for k, v in new.items() if k != "sets"}
    snapshot = output.model_dump()
    assert apply(case) is None
    assert output.model_dump() == snapshot


@pytest.mark.parametrize("changes", [
    dict(experience_level="beginner"), dict(experience_level="intermediate"),
    dict(readiness_level="low"), dict(goal="strength"), dict(specialization_focus=None),
    dict(specialization_focus="upperChest"), dict(session_structure="superset"),
    dict(session_structure="density"), dict(target_split_label="push"),
    dict(adjunct_options=["cooldown_stretch"]), dict(time_available_min=45),
    dict(pain_areas=[{"body_part": "chest"}]),
    dict(mobility_limits=["limited_shoulder_flexion"]),
    dict(condition_policies=[{"condition_code": "lumbar_issue"}]),
])
def test_other_requests_and_health_restrictions_unchanged(changes):
    case = setup_case(**changes)
    old = case[0].model_dump()
    assert apply(case) is None
    assert case[0].model_dump() == old


@pytest.mark.parametrize("muscle", ["chest", "triceps", "front-deltoids", "quadriceps"])
@pytest.mark.parametrize("level", [1, 2, 3])
def test_any_doms_prevents_extra_volume(muscle, level):
    case = setup_case()
    assert apply(case, doms={muscle: level}) is None
    assert sets(case[0]) == [5, 3, 3]


def test_restricted_candidate_and_missing_review_fail_closed():
    case = setup_case()
    assert apply(case, restricted={"flat"}) is None
    case[3][0].pop("core_chest_focus_reviewed")
    assert apply(case) is None


@pytest.mark.parametrize("cap", [10, 11, 12, 22])
def test_total_budget_is_respected(cap):
    case = setup_case()
    diag = apply(case, cap=cap)
    assert sets(case[0]) == ([5, 4, 3] if cap >= 12 else [5, 3, 3])
    assert diag["role_set_capacity_total_budget_exhausted"] == (cap <= 12)
    assert diag["role_set_capacity_exhausted"] == (cap > 12)


def test_no_addition_when_a_set_would_exceed_request_time():
    case = setup_case(time_available_min=50)
    for plan in case[0].exercises:
        plan.rest_time_sec = 180
    before_minutes = estimate_routine_time_min(case[0].exercises)
    assert before_minutes <= 50
    diag = apply(case)
    assert sets(case[0]) == [5, 3, 3]
    assert diag["role_set_capacity_exhausted"] is False


@pytest.mark.parametrize("mutation", ["duration", "missing_slot", "duplicate_id", "not_chest", "method", "invalid_transfer"])
def test_only_complete_ordinary_roles_with_known_transfer_are_eligible(mutation):
    case = list(setup_case())
    if mutation == "duration":
        case[0].exercises[2]._target_duration_sec = 30
    elif mutation == "missing_slot":
        case[0]._selection_slot_by_exercise_id.pop("fly")
    elif mutation == "duplicate_id":
        case[0].exercises[2].exercise_id = "flat"
    elif mutation == "not_chest":
        case[0].exercises[2].primary_muscles = ["triceps"]
    elif mutation == "method":
        case[1] = replace(case[1], training_method="pause")
    else:
        case[0].exercises[0].sets = 6
    old = case[0].model_dump()
    assert apply(case) is None
    assert case[0].model_dump() == old


def test_high_readiness_uses_same_bounded_capacity():
    case = setup_case(readiness_level="high")
    assert apply(case)["role_set_capacity_exhausted"] is True
    assert sets(case[0]) == [5, 4, 3]

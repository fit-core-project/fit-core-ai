from itertools import product

import pytest

from engines.fallback import generate_fallback_routine
from engines.routine_blueprint import build_routine_blueprint
from engines.schemas import LLMExercisePlan, LLMRoutineOutput, RoutineRequest
from engines.specialization_policy import (
    SPECIALIZATION_FOCUS_TARGETS,
    apply_blueprint_specialization_volume,
)


def _candidate(exercise_id, muscle, *, focus_codes=""):
    return {
        "id": exercise_id,
        "name_kr": exercise_id,
        "name_en": exercise_id,
        "primary_muscle": muscle,
        "secondary_muscle": "",
        "movement_type": "ISOLATION",
        "equipment_req": "MACHINE",
        "specialization_focus_codes": focus_codes,
        "difficulty_tier": 2,
        "efficiency_tier": 3,
        "quality_score": 80,
        "strength_effect": "high",
        "hypertrophy_effect": "high",
        "failure_penalty": "low",
        "mainstream_gym_score": 80,
        "main_routine_default_allowed": 1,
        "prescription_mode": "reps",
    }


def _blueprint(*, readiness="normal"):
    return build_routine_blueprint(
        target_split_label="legs",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="advanced",
        readiness_level=readiness,
        time_available_min=60,
        specialization_focus="hamstringEmphasis",
    )


def _output(*plans):
    output = LLMRoutineOutput(
        total_estimated_time=30,
        summary_title="test",
        rationale_summary=["base"],
        exercises=list(plans),
    )
    output._selection_slot_by_exercise_id = {
        "focus": "posterior_chain",
        "donor": "lower_main",
    }
    return output


def _plan(exercise_id, sets):
    return LLMExercisePlan(
        exercise_id=exercise_id,
        exercise_name=exercise_id,
        movement_type="ISOLATION",
        primary_muscles=["hamstring" if exercise_id == "focus" else "quadriceps"],
        target_reps=10,
        sets=sets,
        rest_time_sec=90,
        exercise_rationale="base",
    )


def _by_id(output):
    return {plan.exercise_id: plan for plan in output.exercises}


def _arm_session(sets=(4, 4, 4, 4, 4), *, readiness="normal"):
    blueprint = build_routine_blueprint(
        target_split_label="arm", target_muscles=["biceps", "triceps", "forearm"],
        goal="hypertrophy", experience_level="advanced", readiness_level=readiness,
        time_available_min=60, specialization_focus="forearmEmphasis",
    )
    muscles = ("forearm", "biceps", "triceps", "forearm", "biceps")
    plans = [_plan(str(i), count) for i, count in enumerate(sets)]
    candidates = []
    for plan, muscle in zip(plans, muscles):
        plan.primary_muscles = [muscle]
        candidates.append(_candidate(plan.exercise_id, muscle,
            focus_codes="forearmEmphasis" if muscle == "forearm" else ""))
    output = _output(*plans)
    output._selection_slot_by_exercise_id = {
        "0": "specialization_focus", "1": "target_1_biceps", "2": "target_2_triceps",
        "3": "target_3_forearm", "4": "flex",
    }
    return output, blueprint, candidates


def test_forearm_session_allocates_whole_budget_without_adding_sets():
    result, blueprint, candidates = _arm_session()
    apply_blueprint_specialization_volume(result, blueprint, candidates)
    assert [item.sets for item in result.exercises] == [5, 4, 3, 5, 3]
    assert "상한" in result.exercises[0].exercise_rationale
    once = result.model_dump()
    apply_blueprint_specialization_volume(result, blueprint, candidates)
    assert result.model_dump() == once


@pytest.mark.parametrize("counts", list(product((3, 4, 5), repeat=5)))
@pytest.mark.parametrize("readiness", ["normal", "high", "low"])
def test_forearm_whole_budget_integer_boundaries(counts, readiness):
    result, blueprint, candidates = _arm_session(counts, readiness=readiness)
    apply_blueprint_specialization_volume(result, blueprint, candidates)
    after = [item.sets for item in result.exercises]
    assert sum(after) == sum(counts)
    assert all(2 <= count <= 5 for count in after)
    assert all(after[i] >= counts[i] for i in (0, 3))
    assert all(after[i] <= counts[i] for i in (1, 2, 4))
    if readiness == "low":
        assert after == list(counts)
    before_repeat = result.model_dump()
    apply_blueprint_specialization_volume(result, blueprint, candidates)
    assert result.model_dump() == before_repeat


@pytest.mark.parametrize("restricted", [{"0"}, {"3"}, {"2"}])
def test_forearm_session_never_changes_restricted_ids(restricted):
    result, blueprint, candidates = _arm_session()
    apply_blueprint_specialization_volume(result, blueprint, candidates,
        restricted_exercise_ids=restricted)
    assert all(item.sets == 4 for item in result.exercises if item.exercise_id in restricted)


@pytest.mark.parametrize("focus_code", sorted(SPECIALIZATION_FOCUS_TARGETS))
@pytest.mark.parametrize("goal", ["strength", "hypertrophy", "recomposition"])
@pytest.mark.parametrize("minutes", [30, 50, 60, 90])
def test_low_readiness_preserves_explicit_focus_selection_intent(focus_code, goal, minutes):
    target = sorted(SPECIALIZATION_FOCUS_TARGETS[focus_code])[0]
    args = dict(target_split_label="custom", target_muscles=[target, "biceps" if target != "biceps" else "triceps"],
        goal=goal, experience_level="advanced", time_available_min=minutes,
        specialization_focus=focus_code, method_mode="off")
    normal = build_routine_blueprint(**args, readiness_level="normal")
    low = build_routine_blueprint(**args, readiness_level="low")
    assert [slot.slot_id for slot in low.slots] == [slot.slot_id for slot in normal.slots
        if slot.slot_id != "specialization_complement"]
    assert low.minimum_main_exercise_count == normal.minimum_main_exercise_count
    assert low.target_exercise_count == normal.target_exercise_count
    assert low.training_method is None


def test_forearm_transfer_cannot_extend_time_when_focus_rest_is_longer():
    result, blueprint, candidates = _arm_session()
    for item in result.exercises:
        item.rest_time_sec = 180 if item.primary_muscles == ["forearm"] else 60
    apply_blueprint_specialization_volume(result, blueprint, candidates)
    assert [item.sets for item in result.exercises] == [4, 4, 4, 4, 4]
    assert result._selection_diagnostics["forearm_session_allocation"]["targetShareShortfall"] == 2


@pytest.mark.parametrize("outside", ["timed", "distance", "warmup", "unknown", "compound", "grouped", "strength", "custom"])
def test_whole_forearm_allocator_scope_is_explicit(outside):
    from dataclasses import replace
    from engines.specialization_policy import _allocate_forearm_session
    result, blueprint, candidates = _arm_session()
    if outside == "timed":
        result.exercises[0]._target_duration_sec = 30
    elif outside == "distance":
        result.exercises[0]._target_distance_m = 20
    elif outside == "warmup":
        result.exercises[0]._prescription_set_type = "warmup"
    elif outside == "unknown":
        candidates.pop(0)
    elif outside == "compound":
        result.exercises[0].movement_type = "COMPOUND"
    elif outside == "grouped":
        blueprint = replace(blueprint, session_structure="superset")
    elif outside == "strength":
        blueprint = replace(blueprint, goal="strength")
    elif outside == "custom":
        blueprint = replace(blueprint, target_split_label="custom")
    before = result.model_dump()
    assert not _allocate_forearm_session(result, blueprint, {row["id"]: row for row in candidates}, set())
    assert result.model_dump() == before


def test_single_focus_without_donor_preserves_volume_and_uses_priority_rationale():
    output = _output(_plan("focus", 3))
    apply_blueprint_specialization_volume(
        output,
        _blueprint(),
        [_candidate("focus", "hamstring", focus_codes="hamstringEmphasis")],
    )

    focus = _by_id(output)["focus"]
    assert focus.sets == 3
    assert "우선 배치" in focus.exercise_rationale
    assert "직접 세트를 1세트" not in focus.exercise_rationale


def test_all_donors_at_minimum_preserve_volume():
    output = _output(_plan("focus", 3), _plan("donor", 2))
    apply_blueprint_specialization_volume(
        output,
        _blueprint(),
        [
            _candidate("focus", "hamstring", focus_codes="hamstringEmphasis"),
            _candidate("donor", "quadriceps"),
        ],
    )

    plans = _by_id(output)
    assert (plans["focus"].sets, plans["donor"].sets) == (3, 2)
    assert "직접 세트를 1세트" not in plans["focus"].exercise_rationale


def test_available_donor_transfers_exactly_one_working_set():
    output = _output(_plan("focus", 3), _plan("donor", 4))
    apply_blueprint_specialization_volume(
        output,
        _blueprint(),
        [
            _candidate("focus", "hamstring", focus_codes="hamstringEmphasis"),
            _candidate("donor", "quadriceps"),
        ],
    )

    plans = _by_id(output)
    assert (plans["focus"].sets, plans["donor"].sets) == (4, 3)
    assert sum(plan.sets for plan in output.exercises) == 7


def test_protected_focus_is_not_modified():
    output = _output(_plan("focus", 3), _plan("donor", 3))
    apply_blueprint_specialization_volume(
        output,
        _blueprint(),
        [
            _candidate("focus", "hamstring", focus_codes="hamstringEmphasis"),
            _candidate("donor", "quadriceps"),
        ],
        restricted_exercise_ids={"focus"},
    )

    assert [plan.sets for plan in output.exercises] == [3, 3]


def test_low_readiness_is_not_modified():
    output = _output(_plan("focus", 3), _plan("donor", 3))
    apply_blueprint_specialization_volume(
        output,
        _blueprint(readiness="low"),
        [
            _candidate("focus", "hamstring", focus_codes="hamstringEmphasis"),
            _candidate("donor", "quadriceps"),
        ],
    )

    assert [plan.sets for plan in output.exercises] == [3, 3]


def test_nonworking_focus_receiver_is_not_modified():
    focus = _plan("focus", 3)
    focus._prescription_set_type = "warmup"
    output = _output(focus, _plan("donor", 3))
    apply_blueprint_specialization_volume(
        output,
        _blueprint(),
        [
            _candidate("focus", "hamstring", focus_codes="hamstringEmphasis"),
            _candidate("donor", "quadriceps"),
        ],
    )

    assert [plan.sets for plan in output.exercises] == [3, 3]


def _recipient_context(focus_code="forearmEmphasis", *, readiness="normal", second_sets=4):
    target = sorted(SPECIALIZATION_FOCUS_TARGETS[focus_code])[0]
    donor_target = "biceps" if target != "biceps" else "triceps"
    blueprint = build_routine_blueprint(
        target_split_label="custom", target_muscles=[target, donor_target],
        goal="hypertrophy", experience_level="advanced", readiness_level=readiness,
        time_available_min=60, specialization_focus=focus_code,
    )
    result = _output(_plan("first", 5), _plan("second", second_sets), _plan("donor", 4))
    for item, muscle in zip(result.exercises, (target, target, donor_target)):
        item.primary_muscles = [muscle]
    result._selection_slot_by_exercise_id = {
        "first": blueprint.specialization_slot_id, "second": "flex", "donor": "flex",
    }
    candidates = [
        _candidate("first", target, focus_codes=focus_code),
        _candidate("second", target, focus_codes=focus_code),
        _candidate("donor", donor_target),
    ]
    return result, blueprint, candidates


@pytest.mark.parametrize("focus_code", sorted(SPECIALIZATION_FOCUS_TARGETS))
@pytest.mark.parametrize("second_sets", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("readiness", ["normal", "high", "low"])
def test_capped_first_focus_does_not_hide_another_eligible_recipient(focus_code, second_sets, readiness):
    result, blueprint, candidates = _recipient_context(
        focus_code, readiness=readiness, second_sets=second_sets,
    )
    original_ids = [item.exercise_id for item in result.exercises]
    before_total = sum(item.sets for item in result.exercises)
    apply_blueprint_specialization_volume(result, blueprint, candidates)
    delta = int(second_sets < 5 and readiness != "low")
    assert [item.sets for item in result.exercises] == [5, second_sets + delta, 4 - delta]
    assert sum(item.sets for item in result.exercises) == before_total
    assert [item.exercise_id for item in result.exercises] == original_ids
    assert ("직접 세트를 1세트" in result.exercises[1].exercise_rationale) == bool(delta)
    assert "직접 세트를 1세트" not in result.exercises[0].exercise_rationale


@pytest.mark.parametrize("ineligible", ["restricted", "nonworking", "duration", "distance", "catalog_mode", "not_focus"])
def test_second_focus_recipient_must_be_eligible(ineligible):
    result, blueprint, candidates = _recipient_context()
    second = result.exercises[1]
    restricted = set()
    if ineligible == "restricted":
        restricted.add("second")
    elif ineligible == "nonworking":
        second._prescription_set_type = "warmup"
    elif ineligible == "duration":
        second._target_duration_sec = 30
    elif ineligible == "distance":
        second._target_distance_m = 20
    elif ineligible == "catalog_mode":
        candidates[1]["prescription_mode"] = "duration"
    else:
        candidates[1]["specialization_focus_codes"] = "bicepsEmphasis"
    apply_blueprint_specialization_volume(result, blueprint, candidates, restricted_exercise_ids=restricted)
    assert [item.sets for item in result.exercises] == [5, 4, 4]


@pytest.mark.parametrize("ineligible", ["restricted", "minimum", "nonworking", "duration", "distance", "catalog_mode"])
def test_alternate_recipient_does_not_relax_donor_eligibility(ineligible):
    result, blueprint, candidates = _recipient_context()
    donor = result.exercises[2]
    restricted = set()
    if ineligible == "restricted":
        restricted.add("donor")
    elif ineligible == "minimum":
        donor.sets = 2
    elif ineligible == "nonworking":
        donor._prescription_set_type = "warmup"
    elif ineligible == "duration":
        donor._target_duration_sec = 30
    elif ineligible == "distance":
        donor._target_distance_m = 20
    elif ineligible == "catalog_mode":
        candidates[2]["prescription_mode"] = "distance"
    before = [item.sets for item in result.exercises]
    apply_blueprint_specialization_volume(result, blueprint, candidates, restricted_exercise_ids=restricted)
    assert [item.sets for item in result.exercises] == before


def test_uncapped_primary_recipient_retains_existing_priority():
    result, blueprint, candidates = _recipient_context(second_sets=2)
    result.exercises[0].sets = 4
    apply_blueprint_specialization_volume(result, blueprint, candidates)
    assert [item.sets for item in result.exercises] == [5, 2, 3]


def test_fallback_focus_cannot_exceed_its_caller_set_cap():
    candidate = _candidate("incline", "chest", focus_codes="upperChest")
    candidate.update(name_kr="덤벨 인클라인 체스트 프레스",
                     name_en="Dumbbell Incline Chest Press", movement_type="COMPOUND",
                     equipment_req="DUMBBELL")
    request = RoutineRequest(
        user_id="specialization-boundary",
        target_split_label="chest",
        target_muscles=[],
        time_available_min=60,
        goal="hypertrophy",
        experience_level="advanced",
        readiness_level="normal",
        specialization_focus="upperChest",
        equipment=[],
    )
    blueprint = build_routine_blueprint(
        target_split_label="chest",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="advanced",
        readiness_level="normal",
        time_available_min=60,
        specialization_focus="upperChest",
    )

    response = generate_fallback_routine(
        request,
        candidates=[candidate],
        max_total_sets=3,
        goal="hypertrophy",
        target_exercise_count=1,
        minimum_main_exercise_count=1,
        routine_blueprint=blueprint,
    )

    assert [block.exercise_id for block in response.routine_blocks] == ["incline"]
    assert sum(len(block.prescription) for block in response.routine_blocks) == 3
    assert blueprint.max_total_sets > 3

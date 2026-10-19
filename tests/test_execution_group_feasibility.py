import pytest

from engines.routine_blueprint import build_routine_blueprint
from engines.routine_selection import (
    apply_blueprint_execution_policy,
    apply_blueprint_execution_policy_to_draft,
)
from engines.schemas import LLMRoutineOutput, RoutineDraftResponse


def _candidate(exercise_id, primary, equipment="DUMBBELL", movement_type="ISOLATION"):
    return {
        "id": exercise_id,
        "name_kr": exercise_id,
        "primary_muscle": primary,
        "movement_type": movement_type,
        "equipment_req": equipment,
        "score": 100,
    }


def _output(candidates, *, sets=4, rest=120):
    sets_by_id = sets if isinstance(sets, dict) else {}
    return LLMRoutineOutput.model_validate({
        "total_estimated_time": 0,
        "summary_title": "test",
        "rationale_summary": [],
        "warnings": [],
        "exercises": [
            {
                "exercise_id": candidate["id"],
                "exercise_name": candidate["name_kr"],
                "movement_type": candidate["movement_type"],
                "primary_muscles": [candidate["primary_muscle"]],
                "equipment_type": candidate["equipment_req"],
                "target_reps": 8,
                "sets": sets_by_id.get(candidate["id"], sets),
                "rest_time_sec": rest,
                "exercise_rationale": "test",
            }
            for candidate in candidates
        ],
    })


def _blueprint(structure, *, goal="strength", split="upper"):
    return build_routine_blueprint(
        target_split_label=split,
        target_muscles=[],
        goal=goal,
        experience_level="advanced",
        readiness_level="normal",
        time_available_min=60,
        session_structure=structure,
    )


def _set_accessory_phases(output):
    output._selection_phase_by_exercise_id = {
        exercise.exercise_id: "isolation_accessory"
        for exercise in output.exercises
    }


def _draft(candidates, *, sets=4, rest=120):
    sets_by_id = sets if isinstance(sets, dict) else {}
    return RoutineDraftResponse.model_validate({
        "generation_status": "fallback",
        "status_reason_code": "networkError",
        "is_fallback": True,
        "summary_title": "test",
        "rationale_summary": [],
        "routine_blocks": [
            {
                "order": index,
                "exercise_id": candidate["id"],
                "exercise_name": candidate["name_kr"],
                "movement_type": candidate["movement_type"],
                "primary_muscles": [candidate["primary_muscle"]],
                "equipment_type": candidate["equipment_req"],
                "default_rest_sec": rest,
                "prescription": [
                    {
                        "set_index": set_index,
                        "target_reps": 8,
                        "target_rir": 2,
                        "target_rest_sec": rest,
                    }
                    for set_index in range(1, sets_by_id.get(candidate["id"], sets) + 1)
                ],
                "exercise_rationale": "test",
            }
            for index, candidate in enumerate(candidates, start=1)
        ],
    })


def _metadata(output):
    return output._execution_metadata_by_exercise_id


def test_density_groups_equal_portable_pair():
    candidates = [
        _candidate("curl", "biceps", "DUMBBELL"),
        _candidate("pressdown", "triceps", "BAND"),
    ]
    output = _output(candidates)
    _set_accessory_phases(output)

    adjusted = apply_blueprint_execution_policy(output, _blueprint("density"))

    assert {item["executionGroupType"] for item in _metadata(adjusted).values()} == {"density"}
    assert {item["executionGroupSize"] for item in _metadata(adjusted).values()} == {2}


def test_density_groups_equal_portable_triple_only_when_all_pairs_are_compatible():
    candidates = [
        _candidate("curl", "biceps", "DUMBBELL"),
        _candidate("pressdown", "triceps", "BAND"),
        _candidate("raise", "deltoids", "KETTLEBELL"),
    ]
    output = _output(candidates)
    _set_accessory_phases(output)

    adjusted = apply_blueprint_execution_policy(output, _blueprint("density"))

    assert {item["executionGroupType"] for item in _metadata(adjusted).values()} == {"density"}
    assert {item["executionGroupSize"] for item in _metadata(adjusted).values()} == {3}
    assert [exercise.sets for exercise in adjusted.exercises] == [2, 2, 2]


def test_density_triple_keeps_an_item_single_when_it_is_incompatible_with_any_member():
    candidates = [
        _candidate("cable_curl", "biceps", "CABLE"),
        _candidate("cable_pressdown", "triceps", "CABLE"),
        _candidate("raise", "deltoids", "DUMBBELL"),
    ]
    output = _output(candidates)
    _set_accessory_phases(output)

    adjusted = apply_blueprint_execution_policy(output, _blueprint("density"))
    metadata = _metadata(adjusted)

    assert metadata["cable_curl"]["executionGroupType"] == "density"
    assert metadata["cable_pressdown"]["executionGroupId"] == metadata["cable_curl"]["executionGroupId"]
    assert metadata["raise"]["executionGroupType"] == "single"


def test_density_rejects_two_generic_machines_without_altering_sets_or_rest():
    candidates = [
        _candidate("preacher_curl", "biceps", "MACHINE"),
        _candidate("pushdown", "triceps", "MACHINE"),
    ]
    output = _output(candidates, sets={"preacher_curl": 2, "pushdown": 1}, rest=105)
    _set_accessory_phases(output)

    adjusted = apply_blueprint_execution_policy(output, _blueprint("density"))

    assert [item["executionGroupType"] for item in _metadata(adjusted).values()] == ["single", "single"]
    assert [exercise.sets for exercise in adjusted.exercises] == [2, 1]
    assert [exercise.rest_time_sec for exercise in adjusted.exercises] == [105, 105]
    assert any("정책에 맞게 묶을 수 있는 운동 조합이 없어" in line for line in adjusted.rationale_summary)


def test_density_skips_incompatible_singleton_and_groups_later_contiguous_pair():
    candidates = [
        _candidate("machine_curl", "biceps", "MACHINE"),
        _candidate("curl", "biceps", "DUMBBELL"),
        _candidate("pressdown", "triceps", "BAND"),
    ]
    output = _output(candidates)
    _set_accessory_phases(output)

    adjusted = apply_blueprint_execution_policy(output, _blueprint("density"))
    metadata = _metadata(adjusted)

    assert metadata["machine_curl"]["executionGroupType"] == "single"
    assert metadata["curl"]["executionGroupId"] == metadata["pressdown"]["executionGroupId"]
    assert metadata["curl"]["executionGroupType"] == "density"


def test_density_rejects_unequal_effective_rounds_and_preserves_single_rest():
    candidates = [
        _candidate("curl", "biceps"),
        _candidate("pressdown", "triceps", "BAND"),
    ]
    output = _output(candidates, sets={"curl": 2, "pressdown": 1}, rest=105)
    _set_accessory_phases(output)

    adjusted = apply_blueprint_execution_policy(output, _blueprint("density"))

    assert [item["executionGroupType"] for item in _metadata(adjusted).values()] == ["single", "single"]
    assert [exercise.sets for exercise in adjusted.exercises] == [2, 1]
    assert [exercise.rest_time_sec for exercise in adjusted.exercises] == [105, 105]


def test_density_accepts_equal_effective_rounds_after_existing_cap():
    candidates = [
        _candidate("curl", "biceps"),
        _candidate("pressdown", "triceps", "BAND"),
    ]
    output = _output(candidates, sets={"curl": 3, "pressdown": 4})
    _set_accessory_phases(output)

    adjusted = apply_blueprint_execution_policy(output, _blueprint("density"))

    assert {item["executionGroupType"] for item in _metadata(adjusted).values()} == {"density"}
    assert [exercise.sets for exercise in adjusted.exercises] == [2, 2]


def test_protected_required_mains_remain_single_while_valid_accessories_group():
    candidates = [
        _candidate("push_main", "chest", movement_type="COMPOUND"),
        _candidate("pull_main", "upper-back", movement_type="COMPOUND"),
        _candidate("curl", "biceps"),
        _candidate("pressdown", "triceps", "BAND"),
    ]
    output = _output(candidates)
    output._selection_phase_by_exercise_id = {
        "push_main": "main_compound",
        "pull_main": "secondary_compound",
        "curl": "isolation_accessory",
        "pressdown": "isolation_accessory",
    }
    output._selection_slot_by_exercise_id = {
        "push_main": "upper_push_main",
        "pull_main": "upper_pull_main",
    }

    adjusted = apply_blueprint_execution_policy(output, _blueprint("density"))
    metadata = _metadata(adjusted)

    assert metadata["push_main"]["executionGroupType"] == "single"
    assert metadata["pull_main"]["executionGroupType"] == "single"
    assert metadata["curl"]["executionGroupType"] == "density"
    assert metadata["pressdown"]["executionGroupType"] == "density"


def test_explicit_antagonist_superset_requires_equal_effective_rounds():
    candidates = [
        _candidate("push", "chest", "DUMBBELL", "COMPOUND"),
        _candidate("pull", "upper-back", "BAND", "COMPOUND"),
    ]
    blueprint = _blueprint("superset", goal="hypertrophy", split="chest_back")

    equal = _output(candidates, sets={"push": 4, "pull": 3})
    equal._selection_phase_by_exercise_id = {"push": "main_compound", "pull": "secondary_compound"}
    equal_adjusted = apply_blueprint_execution_policy(equal, blueprint)
    assert {item["executionGroupType"] for item in _metadata(equal_adjusted).values()} == {"superset"}
    assert [exercise.sets for exercise in equal_adjusted.exercises] == [3, 3]

    mismatch = _output(candidates, sets={"push": 3, "pull": 2}, rest=105)
    mismatch._selection_phase_by_exercise_id = {"push": "main_compound", "pull": "secondary_compound"}
    mismatch_adjusted = apply_blueprint_execution_policy(mismatch, blueprint)
    assert {item["executionGroupType"] for item in _metadata(mismatch_adjusted).values()} == {"single"}
    assert [exercise.sets for exercise in mismatch_adjusted.exercises] == [3, 2]
    assert [exercise.rest_time_sec for exercise in mismatch_adjusted.exercises] == [105, 105]


def test_fallback_uses_the_same_generic_machine_and_round_feasibility_policy():
    candidates = [
        _candidate("preacher_curl", "biceps", "MACHINE"),
        _candidate("pushdown", "triceps", "MACHINE"),
    ]

    adjusted = apply_blueprint_execution_policy_to_draft(
        _draft(candidates, sets={"preacher_curl": 2, "pushdown": 1}, rest=105),
        _blueprint("density"),
        candidates,
    )

    assert [block.execution_group_type for block in adjusted.routine_blocks] == ["single", "single"]
    assert [len(block.prescription) for block in adjusted.routine_blocks] == [2, 1]
    assert [block.default_rest_sec for block in adjusted.routine_blocks] == [105, 105]


def test_standard_execution_is_idempotent_and_preserves_output_values():
    candidates = [
        _candidate("curl", "biceps"),
        _candidate("pushdown", "triceps", "BAND"),
    ]
    output = _output(candidates, sets={"curl": 3, "pushdown": 1}, rest=105)
    _set_accessory_phases(output)

    once = apply_blueprint_execution_policy(output, _blueprint("standard"))
    twice = apply_blueprint_execution_policy(once, _blueprint("standard"))

    assert [(exercise.exercise_id, exercise.sets, exercise.rest_time_sec) for exercise in twice.exercises] == [
        ("curl", 3, 105),
        ("pushdown", 1, 105),
    ]
    assert _metadata(once) == _metadata(twice)


@pytest.mark.parametrize("structure", ["superset", "circuit", "density"])
def test_equal_rounds_do_not_make_two_generic_machines_compatible(structure):
    output = _output([
        _candidate("curl", "biceps", "MACHINE"),
        _candidate("pushdown", "triceps", "MACHINE"),
    ], sets=2, rest=105)
    _set_accessory_phases(output)
    adjusted = apply_blueprint_execution_policy(output, _blueprint(structure))
    assert all(m["executionGroupType"] == "single" for m in _metadata(adjusted).values())
    assert [(e.sets, e.rest_time_sec) for e in adjusted.exercises] == [(2, 105), (2, 105)]


@pytest.mark.parametrize("structure", ["superset", "circuit", "density"])
def test_compatible_equipment_does_not_override_unequal_rounds(structure):
    output = _output([
        _candidate("curl", "biceps", "DUMBBELL"),
        _candidate("pushdown", "triceps", "BAND"),
    ], sets={"curl": 2, "pushdown": 1}, rest=105)
    _set_accessory_phases(output)
    adjusted = apply_blueprint_execution_policy(output, _blueprint(structure))
    assert all(m["executionGroupType"] == "single" for m in _metadata(adjusted).values())
    assert [(e.sets, e.rest_time_sec) for e in adjusted.exercises] == [(2, 105), (1, 105)]

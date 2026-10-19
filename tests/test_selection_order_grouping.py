from __future__ import annotations

from copy import deepcopy
import pytest

from engines.routine_selection import order_routine_output_by_selection_phase as phase_order
from engines.schemas import LLMRoutineOutput


def order_routine_output_by_selection_phase(output):
    return phase_order(output, group_primary_muscles=True)


def _exercise(exercise_id, primary_muscles, *, movement_type="ISOLATION", sets=3):
    return {
        "exercise_id": exercise_id,
        "exercise_name": exercise_id,
        "movement_type": movement_type,
        "primary_muscles": primary_muscles,
        "equipment_type": "DUMBBELL",
        "target_reps": 12,
        "sets": sets,
        "rest_time_sec": 60,
        "target_rir": 2,
        "exercise_rationale": f"{exercise_id} rationale",
        "substitution_candidates": [],
    }


def _output(exercises, phases):
    output = LLMRoutineOutput(
        total_estimated_time=45,
        summary_title="selection order",
        rationale_summary=["keep selection facts"],
        warnings=["keep warning"],
        exercises=exercises,
    )
    output._selection_phase_by_exercise_id = phases
    return output


def _ids(output):
    return [exercise.exercise_id for exercise in output.exercises]


def test_standard_single_sets_group_same_primary_by_first_appearance_without_dose_changes():
    output = _output(
        [
            _exercise("forearm_a", ["forearms"], sets=4),
            _exercise("biceps_a", ["biceps"], sets=2),
            _exercise("triceps_a", ["triceps"], sets=3),
            _exercise("forearm_b", ["forearms"], sets=1),
            _exercise("biceps_b", ["biceps"], sets=5),
        ],
        {
            "forearm_a": "isolation_accessory",
            "biceps_a": "isolation_accessory",
            "triceps_a": "isolation_accessory",
            "forearm_b": "isolation_accessory",
            "biceps_b": "isolation_accessory",
        },
    )
    fields_by_id = {
        exercise.exercise_id: exercise.model_dump()
        for exercise in output.exercises
    }

    ordered = order_routine_output_by_selection_phase(output)

    assert _ids(ordered) == [
        "forearm_a",
        "forearm_b",
        "biceps_a",
        "biceps_b",
        "triceps_a",
    ]
    assert {
        exercise.exercise_id: exercise.model_dump()
        for exercise in ordered.exercises
    } == fields_by_id


def test_phase_priority_remains_before_same_phase_primary_grouping():
    output = _output(
        [
            _exercise("biceps_accessory", ["biceps"]),
            _exercise("secondary_row", ["upper-back"], movement_type="COMPOUND"),
            _exercise("forearm_accessory_a", ["forearms"]),
            _exercise("main_press", ["chest"], movement_type="COMPOUND"),
            _exercise("forearm_accessory_b", ["forearms"]),
            _exercise("triceps_accessory", ["triceps"]),
        ],
        {
            "biceps_accessory": "isolation_accessory",
            "secondary_row": "secondary_compound",
            "forearm_accessory_a": "isolation_accessory",
            "main_press": "main_compound",
            "forearm_accessory_b": "isolation_accessory",
            "triceps_accessory": "isolation_accessory",
        },
    )

    ordered = order_routine_output_by_selection_phase(output)

    assert _ids(ordered) == [
        "main_press",
        "secondary_row",
        "biceps_accessory",
        "forearm_accessory_a",
        "forearm_accessory_b",
        "triceps_accessory",
    ]


def test_mixed_movement_priority_stays_ahead_of_primary_grouping_within_phase():
    output = _output(
        [
            _exercise("biceps_isolation", ["biceps"]),
            _exercise("forearm_compound", ["forearms"], movement_type="COMPOUND"),
            _exercise("biceps_compound", ["biceps"], movement_type="COMPOUND"),
            _exercise("forearm_isolation", ["forearms"]),
        ],
        {
            "biceps_isolation": "secondary_compound",
            "forearm_compound": "secondary_compound",
            "biceps_compound": "secondary_compound",
            "forearm_isolation": "secondary_compound",
        },
    )

    ordered = order_routine_output_by_selection_phase(output)

    assert _ids(ordered) == [
        "forearm_compound",
        "biceps_compound",
        "biceps_isolation",
        "forearm_isolation",
    ]


def test_unknown_or_multiple_primaries_stay_individual_while_known_primaries_group():
    output = _output(
        [
            _exercise("unknown_first", []),
            _exercise("forearm_a", ["forearms"]),
            _exercise("multiple_primaries", ["biceps", "forearms"]),
            _exercise("biceps_a", ["biceps"]),
            _exercise("forearm_b", ["forearms"]),
            _exercise("unknown_last", ["unknown"]),
        ],
        {
            "unknown_first": "isolation_accessory",
            "forearm_a": "isolation_accessory",
            "multiple_primaries": "isolation_accessory",
            "biceps_a": "isolation_accessory",
            "forearm_b": "isolation_accessory",
            "unknown_last": "isolation_accessory",
        },
    )

    ordered = order_routine_output_by_selection_phase(output)

    assert _ids(ordered) == [
        "unknown_first",
        "forearm_a",
        "forearm_b",
        "multiple_primaries",
        "biceps_a",
        "unknown_last",
    ]


@pytest.mark.parametrize('group_type', ['superset', 'circuit', 'density'])
def test_ordering_is_idempotent_and_preserves_formed_non_single_groups(group_type):
    output = _output(
        [
            _exercise("biceps_a", ["biceps"]),
            _exercise("forearm_a", ["forearms"]),
            _exercise("triceps_a", ["triceps"]),
            _exercise("forearm_b", ["forearms"]),
        ],
        {
            "biceps_a": "isolation_accessory",
            "forearm_a": "isolation_accessory",
            "triceps_a": "isolation_accessory",
            "forearm_b": "isolation_accessory",
        },
    )
    output._execution_metadata_by_exercise_id = {
        "biceps_a": {"executionGroupId": "C1", "executionGroupType": "circuit", "executionGroupPosition": 1, "executionGroupSize": 2},
        "forearm_a": {"executionGroupId": "C1", "executionGroupType": "circuit", "executionGroupPosition": 2, "executionGroupSize": 2},
        "triceps_a": {"executionGroupId": "S1", "executionGroupType": "single", "executionGroupPosition": 1, "executionGroupSize": 1},
        "forearm_b": {"executionGroupId": "S2", "executionGroupType": "single", "executionGroupPosition": 1, "executionGroupSize": 1},
    }
    for metadata in output._execution_metadata_by_exercise_id.values():
        if metadata['executionGroupType'] == 'circuit':
            metadata['executionGroupType'] = group_type
    before_metadata = deepcopy(output._execution_metadata_by_exercise_id)
    before_fields = {
        exercise.exercise_id: exercise.model_dump()
        for exercise in output.exercises
    }

    once = order_routine_output_by_selection_phase(output)
    once_ids = _ids(once)
    twice = order_routine_output_by_selection_phase(once)

    assert _ids(twice) == once_ids == ["biceps_a", "forearm_a", "triceps_a", "forearm_b"]
    assert twice._execution_metadata_by_exercise_id == before_metadata
    assert {
        exercise.exercise_id: exercise.model_dump()
        for exercise in twice.exercises
    } == before_fields


def test_mixed_movement_grouping_is_idempotent_when_a_primary_only_has_isolation():
    output = _output([
        _exercise('a-iso', ['biceps']),
        _exercise('b-comp', ['forearm'], movement_type='COMPOUND'),
        _exercise('b-iso', ['forearm']),
    ], {key: 'secondary_compound' for key in ('a-iso', 'b-comp', 'b-iso')})
    first = order_routine_output_by_selection_phase(output)
    saved = first.model_dump()
    assert _ids(first) == ['b-comp', 'a-iso', 'b-iso']
    assert order_routine_output_by_selection_phase(first).model_dump() == saved


def test_intermediate_planning_keeps_prior_order_until_explicit_final_grouping():
    output = _output([
        _exercise('f1', ['forearm']), _exercise('b1', ['biceps']),
        _exercise('t1', ['triceps']), _exercise('f2', ['forearm']),
    ], {})
    assert _ids(phase_order(output)) == ['f1', 'b1', 't1', 'f2']
    assert _ids(order_routine_output_by_selection_phase(output)) == ['f1', 'f2', 'b1', 't1']

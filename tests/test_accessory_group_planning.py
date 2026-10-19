from copy import deepcopy
from dataclasses import replace

import pytest

from engines.routine_blueprint import build_blueprint_candidate_bundles, build_routine_blueprint
from engines.routine_selection import (
    adapt_slot_selection_output,
    apply_blueprint_execution_policy,
    render_accessory_group_plan,
)
from engines.schemas import LLMRoutineSelectionOutput


def candidate(id, name, primary, movement, equipment, score=100):
    return dict(id=id, name_kr=name, primary_muscle=primary,
                movement_type=movement, equipment_req=equipment, score=score)


def context(structure='density'):
    bp = build_routine_blueprint(target_split_label='upper', target_muscles=[],
        goal='strength', experience_level='advanced', readiness_level='normal',
        time_available_min=55, session_structure=structure)
    pool = [candidate('bench', 'Barbell Bench Press', 'chest', 'COMPOUND', 'BARBELL', 300),
            candidate('row', 'Dumbbell Row', 'upper-back', 'COMPOUND', 'DUMBBELL', 290),
            candidate('shrug', 'Machine Shrug', 'trapezius', 'ISOLATION', 'MACHINE', 210),
            candidate('kickback', 'Dumbbell Kickback', 'triceps', 'ISOLATION', 'DUMBBELL', 201),
            candidate('curl', 'Dumbbell Incline Curl', 'biceps', 'ISOLATION', 'DUMBBELL', 189)]
    return bp, pool, build_blueprint_candidate_bundles(bp, pool)


def select(bp, pool, bundles):
    selections = [dict(slotId=b['slotId'], exerciseId=b['candidates'][0]['exerciseId'])
                  for b in bundles if b['requirement'] == 'required_when_safe' and b['candidates']]
    return adapt_slot_selection_output(LLMRoutineSelectionOutput.model_validate({'selections': selections}),
        blueprint=bp, ranked_candidates=pool, slot_candidate_bundles=bundles)


def grouped(output, bp):
    result = apply_blueprint_execution_policy(output, bp)
    return {id for id, m in result._execution_metadata_by_exercise_id.items()
            if m['executionGroupType'] == bp.session_structure}


def test_density_repairs_one_machine_accessory_to_portable_pair_and_preserves_main():
    bp, pool, bundles = context()
    output = select(bp, pool, bundles)
    assert grouped(output, bp) == {'kickback', 'curl'}
    assert [x.exercise_id for x in output.exercises][:2] == ['bench', 'row']
    assert len(output.exercises) <= bp.target_exercise_count
    assert output._selection_diagnostics['model_accepted_count'] == 2
    assert output._selection_diagnostics['accessory_group_replacement_count'] == 1
    assert output._selection_diagnostics['accessory_group_addition_count'] == 1


def test_upstream_hint_identifies_server_owned_flex_without_exposing_it_as_model_slot():
    bp, pool, bundles = context()
    hint = render_accessory_group_plan(bp, bundles, pool)
    assert 'kickback' in hint and 'curl' in hint
    assert 'server-owned' in hint
    assert 'do not emit flex' in hint


def test_upper_group_prioritizes_direct_arms_over_extra_anterior_raise():
    bp, pool, _ = context()
    pool.append(candidate('front', 'Dumbbell Front Raise', 'front-deltoids',
                          'ISOLATION', 'DUMBBELL', 199))
    bundles = build_blueprint_candidate_bundles(bp, pool)
    assert grouped(select(bp, pool, bundles), bp) == {'kickback', 'curl'}


@pytest.mark.parametrize('structure', ['standard', 'superset'])
def test_other_structures_are_unchanged(structure):
    bp, pool, bundles = context(structure)
    output = select(bp, pool, bundles)
    assert len(output.exercises) == 3
    assert render_accessory_group_plan(bp, bundles, pool) == ''
    assert not output._selection_diagnostics.get('accessory_group_replacement_count')


def test_generic_machine_pair_is_not_treated_as_one_station():
    bp, pool, _ = context()
    for c in pool[2:]:
        c['equipment_req'] = 'MACHINE'
    bundles = build_blueprint_candidate_bundles(bp, pool)
    output = select(bp, pool, bundles)
    assert not grouped(output, bp)


def test_excluded_or_unbundled_candidate_is_never_added():
    bp, pool, bundles = context()
    for b in bundles:
        b['candidates'] = [c for c in b['candidates'] if c['exerciseId'] != 'curl']
    assert not grouped(select(bp, pool, bundles), bp)
    assert 'curl' not in render_accessory_group_plan(bp, bundles, pool)


def test_no_room_does_not_replace_required_main():
    bp, pool, bundles = context()
    bp = replace(bp, target_exercise_count=3, target_main_exercise_count=3)
    output = select(bp, pool, bundles)
    assert not grouped(output, bp)
    assert {x.exercise_id for x in output.exercises} >= {'bench', 'row'}


def test_inputs_are_immutable_and_pair_is_stable():
    bp, pool, bundles = context()
    before = deepcopy((pool, bundles))
    a = select(bp, pool, bundles)
    b = select(bp, pool, bundles)
    assert a.model_dump() == b.model_dump()
    assert (pool, bundles) == before


def test_circuit_replaces_only_incompatible_preferred_slot():
    bp = build_routine_blueprint(target_split_label='full_body', target_muscles=[],
        goal='generalFitness', experience_level='intermediate', readiness_level='normal',
        time_available_min=45, session_structure='circuit')
    pool = [candidate('squat', 'Dumbbell Goblet Squat', 'quadriceps', 'COMPOUND', 'DUMBBELL'),
            candidate('bench', 'Dumbbell Bench Press', 'chest', 'COMPOUND', 'DUMBBELL'),
            candidate('row', 'Dumbbell Row', 'upper-back', 'COMPOUND', 'DUMBBELL'),
            candidate('legcurl', 'Machine Leg Curl', 'hamstring', 'ISOLATION', 'MACHINE', 160),
            candidate('rdl', 'Dumbbell Romanian Deadlift', 'hamstring', 'COMPOUND', 'DUMBBELL'),
            candidate('press', 'Dumbbell Shoulder Press', 'front-deltoids', 'COMPOUND', 'DUMBBELL')]
    bundles = build_blueprint_candidate_bundles(bp, pool)
    selections = [dict(slotId=b['slotId'], exerciseId=b['candidates'][0]['exerciseId'])
                  for b in bundles if b['slotId'] != 'flex' and b['candidates']]
    output = adapt_slot_selection_output(LLMRoutineSelectionOutput.model_validate({'selections': selections}),
        blueprint=bp, ranked_candidates=pool, slot_candidate_bundles=bundles)
    assert grouped(output, bp) == {'rdl', 'press'}
    assert {x.exercise_id for x in output.exercises} >= {'squat', 'bench', 'row'}

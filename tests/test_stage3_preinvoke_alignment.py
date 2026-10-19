"""Candidate offers and early stops must match existing composition rules."""
from copy import deepcopy
from dataclasses import replace

import pytest

from engines import routine_selection as selection
from engines.routine_blueprint import build_routine_blueprint
from tests.test_push_front_raise_horizontal_boundary import rows


def blueprint(split='push', targets=(), focus=None):
    return build_routine_blueprint(
        target_split_label=split, target_muscles=list(targets), goal='hypertrophy',
        experience_level='intermediate', readiness_level='normal', time_available_min=60,
        specialization_focus=focus, method_mode='off',
    )


def bundles(main=('flat',), secondary=('front', 'fly')):
    return [{'slotId': sid, 'candidates': [{'exerciseId': eid} for eid in ids]}
            for sid, ids in [('push_main', main), ('push_secondary', secondary), ('flex', ('front',))]]


def test_guaranteed_prior_press_removes_rejected_offer_without_mutation():
    source = bundles()
    before = deepcopy(source)
    result = selection.align_candidate_bundles_with_composition(blueprint(), source, rows())
    assert result[1]['candidates'] == [{'exerciseId': 'fly'}]
    assert result[0] == before[0]
    assert result[2] == before[2]  # Flex is server-owned, with its existing guard.
    assert source == before


@pytest.mark.parametrize('case', ['sparse', 'focus', 'custom', 'empty', 'optional', 'mixed'])
def test_no_speculative_exclusion_without_guaranteed_redundancy(case):
    bp, pool, source = blueprint(), rows(), bundles()
    if case == 'sparse':
        pool = [r for r in pool if r['id'] not in {'lateral', 'fly'}]
    elif case == 'focus':
        bp = replace(bp, specialization_focus='frontDeltoidEmphasis')
    elif case == 'custom':
        bp = replace(bp, target_split_label='custom')
    elif case == 'empty':
        source = bundles(main=())
    elif case == 'optional':
        bp = replace(bp, slots=tuple(replace(s, requirement='preferred_when_safe')
                    if s.slot_id == 'push_main' else s for s in bp.slots))
    elif case == 'mixed':
        source = bundles(main=('flat', 'triceps'))
    assert selection.align_candidate_bundles_with_composition(bp, source, pool) == source


def test_missing_metadata_is_not_proof_of_redundancy():
    source = bundles(main=('unknown',))
    assert selection.align_candidate_bundles_with_composition(blueprint(), source, rows()) == source


def test_removed_last_offer_marks_slot_unfilled_and_keeps_original():
    source = bundles(secondary=('front',))
    source[1]['slotUnfilled'] = False
    result = selection.align_candidate_bundles_with_composition(blueprint(), source, rows())
    assert result[1]['candidates'] == []
    assert result[1]['slotUnfilled'] is True
    assert source[1]['slotUnfilled'] is False


@pytest.mark.parametrize('split,targets,ids,expected', [
    ('push', ('chest','triceps'), ('triceps',), False),
    ('push', ('chest','triceps'), ('flat','triceps'), True),
    ('custom', ('triceps',), ('triceps',), True),
    ('custom', ('front-deltoids','triceps'), ('front','triceps'), True),
])
def test_preinvoke_main_feasibility_reuses_final_rule(split, targets, ids, expected):
    pool = [r for r in rows() if r['id'] in ids]
    bp = blueprint(split, targets)
    assert selection.candidates_have_required_main_pattern(bp, pool) is expected

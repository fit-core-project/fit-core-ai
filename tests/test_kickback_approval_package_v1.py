"""Prepared conditions are executable evidence, never an admission approval."""
import copy
import json
from pathlib import Path

import pytest

from approval_packages.cable_standing_kickback_v1 import evaluate_prepared_conditions, preview_bilateral_set
from engines.experience_policy import automatic_recommendation_eligibility
from tests.test_short_routine_notice_v4 import request

BASE = Path(__file__).resolve().parents[1]


@pytest.fixture
def row():
    data = json.loads((BASE/'docs/evidence/glute-coverage-ab-v1-2026-10-10/candidate-review-inputs.json').read_text())
    return copy.deepcopy(next(r['effective'] for r in data['rows'] if r['id']=='V24ROW-003678'))


def setup():
    return dict(lowPulleyAvailable=True, ankleCuffAvailable=True, stableHandSupportAvailable=True,
                exactVariantFamiliar=True, restrictionContextResolved=True)


def test_ready_does_not_grant_admission_or_change_candidate(row):
    before = copy.deepcopy(row)
    review = evaluate_prepared_conditions(row, request(unpreferred_exercise_ids=[]), setup())
    assert review.ready_if_approved and review.unmet_conditions == ()
    assert review.runtime_enabled is False
    assert not automatic_recommendation_eligibility(row, 'intermediate').eligible
    assert row == before


@pytest.mark.parametrize('key', list(setup()))
def test_unknown_setup_does_not_become_true(row, key):
    data = setup()
    del data[key]
    result = evaluate_prepared_conditions(row, request(unpreferred_exercise_ids=[]), data)
    assert not result.ready_if_approved and key in result.unmet_conditions


@pytest.mark.parametrize('changes', [
    {'experience_level':'beginner'}, {'available_equipment':['MACHINE']}, {'equipment':['CABLE']},
    {'pain_areas':[{'bodyPart':'lower-back'}]}, {'doms_data':{'gluteal':1}},
    {'injury_conditions':[{'areaCode':'hip','status':'activeUncleared'}]},
    {'condition_policies':[{'clearanceStatus':'unknown'}]}, {'readiness_level':'low'},
    {'unpreferred_exercise_ids':['V24ROW-003678']},
])
def test_existing_restrictions_remain_a_gate(row, changes):
    req = request(**dict({'unpreferred_exercise_ids':[]}, **changes))
    assert not evaluate_prepared_conditions(row, req, setup()).ready_if_approved


@pytest.mark.parametrize('id', ['V24ROW-003677','V24ROW-003676','123'])
def test_neighboring_names_or_numeric_aliases_do_not_inherit_review(row, id):
    row['id'] = id
    assert not evaluate_prepared_conditions(row, request(unpreferred_exercise_ids=[]), setup()).ready_if_approved


def test_prescription_preview_explicitly_counts_both_sides():
    preview = preview_bilateral_set(reps_per_side=12, seconds_per_rep=3, side_switch_sec=10)
    assert preview['workAndSwitchSec'] == 82
    assert preview['sideCount'] == 2 and preview['setUnit']=='bothSides'
    assert preview['runtimeEnabled'] is False


def test_pending_package_is_not_imported_by_production_engines():
    assert all('approval_packages' not in p.read_text() for p in (BASE/'engines').rglob('*.py'))

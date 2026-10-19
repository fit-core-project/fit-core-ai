"""Contain unresolved metadata; these tests do not endorse incorrect equipment."""
import copy
import json

import pytest

from engines.adjunct_admission import reviewed_purpose_index
from engines.candidate_ranker import score_candidate_exercises
from engines.experience_policy import automatic_recommendation_eligibility
from engines.option_resolver import ADJUNCT_OPTION_PRIORITY
from tests.pending_equipment_audit_v1 import CORE, PAIRS, LEVELS, EQUIPMENT_CASES, connect
from tests.test_numeric_id_boundary_v1 import readonly_core, seed_rows


@pytest.fixture(scope='module')
def pending_native(readonly_core):
    with connect() as conn:
        raw = [dict(r) for r in conn.execute('SELECT * FROM exercise_tier') if r['id'] in PAIRS.values()]
    return {r['id']: r for r in readonly_core._enrich_candidates_with_local_catalog(raw)}


@pytest.mark.parametrize('cid', PAIRS.values())
@pytest.mark.parametrize('level', LEVELS)
@pytest.mark.parametrize('equipment_case', EQUIPMENT_CASES)
def test_pending_metadata_cannot_enter_normal_or_preferred_ranking(pending_native, cid, level, equipment_case):
    candidate = pending_native[cid]
    assert candidate['main_routine_default_allowed'] == 0
    assert not automatic_recommendation_eligibility(candidate, level).eligible
    available, blocked = EQUIPMENT_CASES[equipment_case]
    for preferred in ([], [cid]):
        assert score_candidate_exercises(
            [copy.deepcopy(candidate)], [candidate['primary_muscle']],
            experience_level=level, available_equipment=available, blocked_equipment=blocked,
            preferred_exercise_ids=preferred, goal='hypertrophy', readiness_level='normal',
            training_style='gym', target_split_label='custom', time_available_min=60,
        ) == []


@pytest.mark.parametrize('cid', PAIRS.values())
@pytest.mark.parametrize('level', LEVELS)
def test_pending_rows_have_no_reviewed_purpose_exception(pending_native, cid, level):
    assert cid.lower() not in reviewed_purpose_index()
    for option in ADJUNCT_OPTION_PRIORITY:
        assert not automatic_recommendation_eligibility(
            pending_native[cid], level, adjunct_options=(option,)
        ).eligible


@pytest.mark.parametrize('sid,cid', PAIRS.items())
def test_pending_numeric_identity_and_history_are_not_promoted(readonly_core, sid, cid):
    seed = next(r for r in seed_rows() if str(r['id']) == sid)
    original = copy.deepcopy(seed)
    result, = readonly_core._enrich_candidates_with_local_catalog([seed])
    assert seed == original
    assert result['id'] == int(sid)
    assert 'catalog_exercise_id' not in result
    for key in ('equipment_req', 'movement_type', 'primary_muscle'):
        assert result[key] == original[key]
    edges = json.loads((CORE/'engines/data/catalog_identity_bridge_v1.json').read_text())['edges']
    assert next(e for e in edges if e['sourceId'] == sid and e['catalogId'] == cid)['status'] == 'candidate'

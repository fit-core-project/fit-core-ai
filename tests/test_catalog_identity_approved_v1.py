"""Approved real seed edges; identity does not imply recommendation admission."""
from collections import Counter
import copy
import json

import pytest

from tests import test_numeric_id_boundary_v1 as boundary
from tests.test_numeric_id_boundary_v1 import readonly_core
from engines import catalog_identity as bridge
from engines.schemas import RoutineDraftResponse, RoutineRequest

EXPECTED = {
    '14': ('V24ROW-006036', 'biceps'),
    '115': ('V24ROW-006763', 'abductors'),
    '116': ('V24ROW-006771', 'abductors'),
    '144': ('V24ROW-006904', 'abs'),
    '73': ('V24ROW-004796', 'trapezius'),
    '74': ('V24ROW-004791', 'trapezius'),
    '124': ('V24ROW-004791', 'trapezius'),
    '130': ('V24ROW-006028', 'biceps'),
    '165': ('V24ROW-004712', 'gluteal'),
    '125': ('V24ROW-006742', 'gluteal'),
    '151': ('V24ROW-004710', 'abductors'),
}


def seeds():
    return {str(r['id']): r for r in boundary.seed_rows()}


def test_manifest_only_user_authorized_transitions():
    initial = json.loads((boundary.CORE/'tests/fixtures/catalog_identity_bridge_initial_v1.json').read_text())
    before = {(e['sourceId'], e['catalogId']): e for e in initial['edges']}
    edges = bridge.load_identity_edges()
    assert len(edges) == 46
    # User approved exactly two additional edges on 2026-10-08; all other identities stay locked.
    assert Counter(e['status'] for e in edges) == {'approved': 11, 'candidate': 34, 'rejected': 1}
    for edge in edges:
        sid, cid = edge['sourceId'], edge['catalogId']
        if sid in EXPECTED and cid == EXPECTED[sid][0]:
            assert edge['status'] == 'approved'
            reviewer = 'user' if sid in {'14', '115', '116', '144', '125', '151'} else 'user-delegated Claude review'
            assert edge['review']['reviewedBy'] == reviewer
            assert edge['review']['reviewedAt'] == ('2026-10-08' if sid in {'125', '151'} else '2026-10-07')
            assert edge['review']['rationale'] and edge['review']['evidencePath']
        elif (sid, cid) == ('98', 'V24ROW-006191'):
            assert edge['status'] == 'rejected'
        elif (sid, cid) == ('98', 'V24ROW-003251'):
            assert edge['status'] == 'candidate' and edge['review'] is None
        else:
            assert edge == before[(sid, cid)]
        if (sid, cid) in before:
            assert edge['sourceIdentity'] == before[(sid, cid)]['sourceIdentity']
            assert edge['catalogIdentity'] == before[(sid, cid)]['catalogIdentity']


@pytest.mark.parametrize('sid', EXPECTED)
def test_approved_numeric_enrichment_corrects_primary_and_keeps_id(readonly_core, sid):
    from engines.schemas import RoutineDraftResponse
    source = seeds()[sid]
    original = copy.deepcopy(source)
    candidate, = readonly_core._enrich_candidates_with_local_catalog([source])
    cid, primary = EXPECTED[sid]
    assert candidate['id'] == int(sid)
    assert candidate['catalog_exercise_id'] == cid
    assert candidate['primary_muscle'] == primary
    assert source == original
    req = RoutineRequest.model_validate(boundary.adapter_projection(boundary.public_request(source), int(sid)))
    payload = boundary.core_response(candidate).model_dump(mode='json', by_alias=True)
    assert payload['routineBlocks'][0]['primaryMuscles'] == [primary]
    restored = RoutineDraftResponse.model_validate(payload)
    assert restored.routine_blocks[0].exercise_id == req.preferred_exercise_ids[0] == sid


def test_74_and_124_share_catalog_without_merging_history(readonly_core):
    source = seeds()
    candidates = readonly_core._enrich_candidates_with_local_catalog([source['74'], source['124']])
    assert [r['id'] for r in candidates] == [74, 124]
    assert {r['catalog_exercise_id'] for r in candidates} == {'V24ROW-004791'}
    blocks = [boundary.core_response(r).routine_blocks[0] for r in candidates]
    response = RoutineDraftResponse(generation_status='success', status_reason_code='none', is_fallback=False,
        summary_title='identity boundary', rationale_summary=[], routine_blocks=blocks)
    assert [b['exerciseId'] for b in response.model_dump(mode='json', by_alias=True)['routineBlocks']] == ['74', '124']

    class Result:
        def __init__(self, rows): self.rows = rows
        def fetchall(self): return self.rows

    class History:
        def __init__(self): self.calls = 0
        def execute(self, query, params):
            self.calls += 1
            if self.calls == 1: return Result([('synthetic',)])
            return Result([(str(r['id']), r['name_kr'], r['primary_muscle'], 20.0, 8, 3, None, False) for r in candidates])

    records = readonly_core.get_recent_sets(History(), 'synthetic-identity-history')
    assert [r.exercise_id for r in records] == ['74', '124']


@pytest.mark.parametrize('sid', EXPECTED)
def test_approval_does_not_bypass_source_identity_drift(readonly_core, sid):
    source = seeds()[sid]
    source['name_en'] += ' different variant'
    candidate, = readonly_core._enrich_candidates_with_local_catalog([source])
    assert 'catalog_exercise_id' not in candidate
    assert candidate['id'] == int(sid)


def test_both_back_squat_edges_do_not_resolve(readonly_core):
    candidate, = readonly_core._enrich_candidates_with_local_catalog([seeds()['98']])
    assert candidate['id'] == 98 and 'catalog_exercise_id' not in candidate


@pytest.mark.parametrize('sid', ['115', '116', '125', '151'])
def test_identity_approval_does_not_unlock_catalog_denial(readonly_core, sid):
    from engines.experience_policy import automatic_recommendation_eligibility
    candidate, = readonly_core._enrich_candidates_with_local_catalog([seeds()[sid]])
    assert candidate['main_routine_default_allowed'] == 0
    assert not automatic_recommendation_eligibility(candidate, 'intermediate').eligible

"""Identity approvals do not grant recommendation or main-slot admission."""
import copy
import json
from unittest.mock import patch

import pytest

from engines.experience_policy import automatic_recommendation_eligibility, main_role_eligibility
from tests import test_numeric_id_boundary_v1 as boundary
from tests.test_numeric_id_boundary_v1 import readonly_core

BEFORE = boundary.CORE/'docs/evidence/two-identity-approvals-v1-2026-10-08/before-bridge.json'
PAIRS = {'125': ('V24ROW-006742', 'gluteal', 'BARBELL', 'COMPOUND'),
         '151': ('V24ROW-004710', 'abductors', 'BODYWEIGHT', 'ISOLATION')}


def test_only_two_user_authorized_edges_changed():
    old = json.loads(BEFORE.read_text())
    new = json.loads((boundary.CORE/'engines/data/catalog_identity_bridge_v1.json').read_text())
    assert {k:v for k,v in old.items() if k != 'edges'} == {k:v for k,v in new.items() if k != 'edges'}
    assert len(old['edges']) == len(new['edges']) == 46
    changes = []
    for a, b in zip(old['edges'], new['edges']):
        if a == b:
            continue
        sid = b['sourceId']
        changes.append((sid, b['catalogId']))
        assert a['status'] == 'candidate' and a['review'] is None
        assert b['status'] == 'approved'
        assert b['review']['reviewedBy'] == 'user' and b['review']['reviewedAt'] == '2026-10-08'
        assert b['review']['rationale'] and b['review']['evidencePath'].endswith('taxonomy-correction-and-pending-identity-v1-2026-10-08.md')
        assert {k:v for k,v in a.items() if k not in {'status', 'review'}} == {k:v for k,v in b.items() if k not in {'status', 'review'}}
    assert sorted(changes) == sorted((sid, v[0]) for sid,v in PAIRS.items())


@pytest.mark.parametrize('sid', PAIRS)
@pytest.mark.parametrize('level', ['beginner', 'intermediate', 'advanced'])
def test_approval_changes_legacy_admission_not_movement_type(readonly_core, sid, level):
    seed = next(r for r in boundary.seed_rows() if str(r['id']) == sid)
    old = json.loads(BEFORE.read_text())
    with patch.object(readonly_core, 'load_identity_edges', return_value=old['edges']):
        before, = readonly_core._enrich_candidates_with_local_catalog([copy.deepcopy(seed)])
    after, = readonly_core._enrich_candidates_with_local_catalog([copy.deepcopy(seed)])
    cid, primary, equipment, movement = PAIRS[sid]
    assert before['id'] == after['id'] == int(sid)
    assert 'catalog_exercise_id' not in before
    assert after['catalog_exercise_id'] == cid
    assert after['primary_muscle'] == primary and after['equipment_req'] == equipment
    assert before['movement_type'] == after['movement_type'] == movement
    assert after['canonical_movement_type'] == movement and after['prescription_mode'] == 'reps'
    assert after['main_routine_default_allowed'] == 0
    assert automatic_recommendation_eligibility(before, level).eligible
    assert main_role_eligibility(before, level, 'hypertrophy').reason_code == 'legacy_metadata_missing_allow'
    assert not automatic_recommendation_eligibility(after, level).eligible
    assert not main_role_eligibility(after, level, 'hypertrophy').eligible
    public = boundary.core_response(after).model_dump(mode='json', by_alias=True)
    assert public['routineBlocks'][0]['exerciseId'] == sid
    assert public['routineBlocks'][0]['primaryMuscles'] == [primary]


def test_clamshell_approval_does_not_approve_other_variants():
    edges = json.loads((boundary.CORE/'engines/data/catalog_identity_bridge_v1.json').read_text())['edges']
    other = [e for e in edges if e['sourceId'] == '151' and e['catalogId'] != 'V24ROW-004710']
    assert {e['catalogId'] for e in other} == {'V24ROW-007275', 'V24ROW-007276', 'V24ROW-007277', 'V24ROW-007278'}
    assert all(e['status'] == 'candidate' and e['review'] is None for e in other)

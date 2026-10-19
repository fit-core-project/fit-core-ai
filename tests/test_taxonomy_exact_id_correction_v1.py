"""Exact-ID primary correction is not identity approval or admission clearance."""
import copy
import json
import sqlite3

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from engines.muscle_taxonomy import EXACT_ID_PRIMARY_CORRECTIONS, exact_id_primary_taxonomy_patch
from engines.experience_policy import automatic_recommendation_eligibility
from tests import test_numeric_id_boundary_v1 as boundary
from tests.test_numeric_id_boundary_v1 import readonly_core


def catalog_rows():
    with sqlite3.connect((boundary.CORE/'fit_core.sqlite').as_uri()+'?mode=ro', uri=True) as conn:
        conn.row_factory = sqlite3.Row
        return {r['id']: dict(r) for r in conn.execute('SELECT * FROM exercise_tier')}


@pytest.mark.parametrize('cid', EXACT_ID_PRIMARY_CORRECTIONS)
def test_exact_correction_preserves_source_admission_and_non_primary_fields(readonly_core, monkeypatch, cid):
    source = catalog_rows()[cid]
    original = copy.deepcopy(source)
    with monkeypatch.context() as patch:
        patch.setattr('engines.muscle_taxonomy.exact_id_primary_taxonomy_patch', lambda *_: {})
        before, = readonly_core._enrich_candidates_with_local_catalog([source])
    after, = readonly_core._enrich_candidates_with_local_catalog([source])
    assert source == original
    assert after['id'] == after['catalog_exercise_id'] == cid
    assert after['primary_muscle'] == after['canonical_primary_muscle'] == EXACT_ID_PRIMARY_CORRECTIONS[cid][2]
    assert after['catalog_source_primary_muscle'] == EXACT_ID_PRIMARY_CORRECTIONS[cid][1]
    allowed = {'primary_muscle', 'canonical_primary_muscle', 'core_taxonomy_original_primary_muscle', 'core_taxonomy_review_version'}
    assert {k: v for k,v in after.items() if k not in allowed} == {k: v for k,v in before.items() if k not in allowed}
    assert after['main_routine_default_allowed'] == 0
    for experience in ('beginner', 'intermediate', 'advanced'):
        assert not automatic_recommendation_eligibility(after, experience).eligible


@pytest.mark.parametrize('cid', EXACT_ID_PRIMARY_CORRECTIONS)
def test_exact_patch_rejects_identity_drift_and_missing_provenance(readonly_core, cid):
    row = copy.deepcopy(readonly_core._load_local_catalog_enrichment_cache()['by_id'][cid.lower()])
    source = copy.deepcopy(readonly_core._load_source_equipment_by_id()[cid])
    assert exact_id_primary_taxonomy_patch(row, source)
    assert exact_id_primary_taxonomy_patch(row, None) == {}
    for key in ('catalog_exercise_id', 'catalog_name_en', 'catalog_name_kr', 'catalog_primary_muscle', 'canonical_primary_muscle', 'taxonomy_review_status'):
        changed = {**row, key: 'unreviewed'}
        assert exact_id_primary_taxonomy_patch(changed, source) == {}


def test_existing_forearm_review_is_not_reapplied_or_promoted(readonly_core, monkeypatch):
    cid = 'V24ROW-004724'
    assert cid not in EXACT_ID_PRIMARY_CORRECTIONS
    source = catalog_rows()[cid]
    with monkeypatch.context() as patch:
        patch.setattr('engines.muscle_taxonomy.exact_id_primary_taxonomy_patch', lambda *_: {})
        before = readonly_core._enrich_candidates_with_local_catalog([source])
    after = readonly_core._enrich_candidates_with_local_catalog([source])
    assert after == before
    assert after[0]['primary_muscle'] == 'forearm'
    assert after[0]['core_admission_scope'] == 'forearmEmphasis'
    assert after[0]['main_routine_default_allowed'] == 1


@pytest.mark.parametrize('target', ['upper-back', 'adductor', 'forearm', 'abductors'])
def test_canonical_query_retrieves_corrected_target_only(readonly_core, target):
    engine = create_engine(f'sqlite:///file:{boundary.CORE/"fit_core.sqlite"}?mode=ro&uri=true')
    try:
        with Session(engine) as session:
            rows = readonly_core.get_candidate_exercises(session, [target], [], [], [])
    finally:
        engine.dispose()
    assert rows and {r['primary_muscle'] for r in rows} == {target}
    ids = {r['id'] for r in rows}
    expected = {cid for cid, (_,_,muscle) in EXACT_ID_PRIMARY_CORRECTIONS.items() if muscle == target}
    assert ids & set(EXACT_ID_PRIMARY_CORRECTIONS) == expected


@pytest.mark.parametrize('sid', ['6', '9', '140', '151', '171', '135', '125', '98'])
def test_numeric_edges_follow_user_approval_without_changing_history_id(readonly_core, sid):
    source = next(r for r in boundary.seed_rows() if str(r['id']) == sid)
    candidate, = readonly_core._enrich_candidates_with_local_catalog([source])
    assert candidate['id'] == int(sid)
    manifest = json.loads((boundary.CORE/'engines/data/catalog_identity_bridge_v1.json').read_text())
    # Preserve pending-edge protection while checking the two explicit 2026-10-08 approvals.
    if sid in {'125', '151'}:
        cid, primary = {'125': ('V24ROW-006742', 'gluteal'), '151': ('V24ROW-004710', 'abductors')}[sid]
        assert candidate['catalog_exercise_id'] == cid
        assert candidate['primary_muscle'] == primary
        assert candidate['main_routine_default_allowed'] == 0
        approved = [e for e in manifest['edges'] if e['status'] == 'approved' and e['sourceId'] == sid]
        assert len(approved) == 1 and approved[0]['catalogId'] == cid
        assert approved[0]['review']['reviewedBy'] == 'user'
        assert approved[0]['review']['reviewedAt'] == '2026-10-08'
        return
    assert 'catalog_exercise_id' not in candidate
    assert 'core_taxonomy_review_version' not in candidate
    assert not any(e['status'] == 'approved' and e['sourceId'] == sid for e in manifest['edges'])


@pytest.mark.parametrize('cid', ['V24ROW-004706', 'V24ROW-004707', 'V24ROW-004708', 'V24ROW-004712', 'V24ROW-004713', 'V24ROW-004714'])
def test_other_hip_rows_are_unchanged(readonly_core, monkeypatch, cid):
    source = catalog_rows()[cid]
    with monkeypatch.context() as patch:
        patch.setattr('engines.muscle_taxonomy.exact_id_primary_taxonomy_patch', lambda *_: {})
        before = readonly_core._enrich_candidates_with_local_catalog([source])
    assert readonly_core._enrich_candidates_with_local_catalog([source]) == before

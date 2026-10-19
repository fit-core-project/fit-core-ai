"""Integration decision regressions; approval cases use synthetic mappings only."""
import copy
import json

import pytest

from engines import catalog_identity as bridge
from engines.registry import MUSCLE_SLUGS
from engines.schemas import RoutineBlock, RoutineDraftResponse


def pair():
    source = dict(id=30, name_kr='synthetic press', name_en='Synthetic Press',
                  primary_muscle='chest', equipment_req='BARBELL', movement_type='COMPOUND')
    row = {'catalog_' + k: v for k, v in source.items() if k != 'id'}
    row['catalog_exercise_id'] = 'V24ROW-TEST'
    edge = dict(sourceId='30', catalogId='V24ROW-TEST', status='approved',
                sourceIdentity=bridge.identity_snapshot(source), catalogIdentity=bridge.identity_snapshot(source),
                review=dict(reviewedBy='synthetic reviewer', reviewedAt='2026-10-06', rationale='test only'))
    return source, {'v24row-test': row}, edge


@pytest.mark.parametrize('status', ['candidate', 'rejected', 'ambiguous'])
def test_unapproved_identity_never_resolves(status):
    source, catalog, edge = pair()
    edge['status'] = status
    assert bridge.resolve_catalog_identity(source, catalog, [edge]) is None


def test_approved_identity_preserves_numeric_source():
    source, catalog, edge = pair()
    assert bridge.resolve_catalog_identity(source, catalog, [edge]) == catalog['v24row-test']
    assert source['id'] == 30


@pytest.mark.parametrize('side', ['source', 'catalog'])
@pytest.mark.parametrize('field', bridge.IDENTITY_FIELDS)
def test_metadata_drift_denies_mapping(side, field):
    source, catalog, edge = pair()
    if side == 'source':
        source[field] = 'changed'
    else:
        catalog['v24row-test']['catalog_' + field] = 'changed'
    assert bridge.resolve_catalog_identity(source, catalog, [edge]) is None


def test_one_to_many_does_not_pick_first_approval():
    source, catalog, edge = pair()
    other = dict(edge, catalogId='V24ROW-OTHER')
    catalog['v24row-other'] = dict(catalog['v24row-test'], catalog_exercise_id='V24ROW-OTHER')
    assert bridge.resolve_catalog_identity(source, catalog, [edge, other]) is None


def test_many_to_one_keeps_distinct_public_history_ids():
    source, catalog, edge = pair()
    other = dict(edge, sourceId='124')
    for sid in (30, 124):
        candidate = dict(source, id=sid)
        assert bridge.resolve_catalog_identity(candidate, catalog, [edge, other])
        assert candidate['id'] == sid


def test_initial_candidate_manifest_has_no_approvals():
    # Preserve the initial contract separately from later user-approved edges.
    edges = bridge.load_identity_edges(
        bridge.BRIDGE_PATH.parents[2] / 'tests/fixtures/catalog_identity_bridge_initial_v1.json')
    assert edges
    assert {e['status'] for e in edges} == {'candidate'}
    assert {'30', '75', '98', '14', '115', '116', '144'} <= {e['sourceId'] for e in edges}
    assert {e['sourceId'] for e in edges} == {
        '6', '7', '9', '14', '30', '47', '50', '73', '74', '75', '98', '99',
        '115', '116', '117', '124', '125', '130', '135', '140', '144',
        '151', '161', '165', '171',
    }
    assert len(edges) == 45


def test_public_enum_is_the_locked_v31_eighteen_not_the_internal_enum():
    assert MUSCLE_SLUGS == {
        'abductors', 'abs', 'adductor', 'back-deltoids', 'biceps', 'calves',
        'chest', 'forearm', 'front-deltoids', 'gluteal', 'hamstring', 'lower-back',
        'neck', 'obliques', 'quadriceps', 'trapezius', 'triceps', 'upper-back',
    }


@pytest.mark.parametrize('damage', ['review', 'status', 'duplicate', 'snapshot', 'version'])
def test_malformed_registry_fails_closed(tmp_path, damage):
    _, _, edge = pair()
    manifest = dict(version='catalog-identity-bridge-v1-draft',
                    sourceNamespace='fit-core-backend.exercise_tier', edges=[edge])
    if damage == 'review': edge['review'] = None
    if damage == 'status': edge['status'] = 'APPROVED'
    if damage == 'duplicate': manifest['edges'].append(copy.deepcopy(edge))
    if damage == 'snapshot': edge['sourceIdentity'].pop('equipment_req')
    if damage == 'version': manifest['version'] = 'other'
    path = tmp_path/'bridge.json'
    path.write_text(json.dumps(manifest))
    assert bridge.load_identity_edges(path) == []


def block(muscles):
    return RoutineBlock(order=1, exercise_id='30', exercise_name='synthetic',
                        primary_muscles=muscles, default_rest_sec=90,
                        prescription=[], exercise_rationale='boundary only')


@pytest.mark.parametrize('muscles', [sorted(MUSCLE_SLUGS), ['side-deltoids'],
    ['chest', 'side-deltoids'], ['unmapped-internal-value'], []])
def test_public_projection_preserves_detail_and_internal_taxonomy(muscles):
    b = block(muscles)
    public = json.loads(b.model_dump_json(by_alias=True))
    assert set(public['primaryMuscles']) <= MUSCLE_SLUGS
    assert public.get('primaryMuscleDetails', []) == [m for m in muscles if m not in MUSCLE_SLUGS]
    assert b.primary_muscles == muscles
    assert b.model_dump()['primary_muscles'] == muscles
    restored = RoutineBlock.model_validate(public)
    assert set(restored.primary_muscles) == set(muscles)
    assert restored.model_dump(by_alias=True) == public


def test_nested_transport_projection():
    response = RoutineDraftResponse(generation_status='success', status_reason_code='none',
        is_fallback=False, summary_title='synthetic', rationale_summary=[],
        routine_blocks=[block(['side-deltoids'])])
    data = response.model_dump(by_alias=True, mode='json')
    assert data['routineBlocks'][0]['primaryMuscles'] == []
    assert data['routineBlocks'][0]['primaryMuscleDetails'] == ['side-deltoids']
    assert RoutineDraftResponse.model_validate(data).routine_blocks[0].primary_muscles == ['side-deltoids']


def test_no_name_fallback_or_stale_unapproved_identity(monkeypatch):
    from engines import db_queries as dbq
    source, catalog, _ = pair()
    monkeypatch.setattr(dbq, '_load_local_catalog_enrichment_cache', lambda: dict(
        by_id=catalog, substitutions={}, approved_condition_policies={}))
    monkeypatch.setattr(dbq, '_load_source_equipment_by_id', lambda: {})
    monkeypatch.setattr(dbq, '_load_core_admission_reviews', lambda: {})
    monkeypatch.setattr(dbq, 'load_identity_edges', lambda: [])
    result = dbq._enrich_candidates_with_local_catalog([source])
    assert result == [source]
    stale = dict(source, catalog_exercise_id='V24ROW-TEST', lumbar_load='low')
    assert dbq._enrich_candidates_with_local_catalog([stale]) == []


def test_stale_external_identity_is_not_kept_when_catalog_unavailable(monkeypatch):
    from engines import db_queries as dbq
    monkeypatch.setattr(dbq, '_load_local_catalog_enrichment_cache', lambda: None)
    source, _, _ = pair()
    stale = dict(source, catalog_exercise_id='V24ROW-TEST', lumbar_load='low')
    assert dbq._enrich_candidates_with_local_catalog([stale]) == []
    assert dbq._enrich_candidates_with_local_catalog([source]) == [source]


def test_approved_enrichment_uses_bridge_not_name_and_preserves_public_id(monkeypatch):
    from engines import db_queries as dbq
    source, catalog, edge = pair()
    monkeypatch.setattr(dbq, '_load_local_catalog_enrichment_cache', lambda: dict(
        by_id=catalog, equipment_by_id={}, substitutions={}, approved_condition_policies={}))
    monkeypatch.setattr(dbq, '_load_source_equipment_by_id', lambda: {})
    monkeypatch.setattr(dbq, '_load_core_admission_reviews', lambda: {})
    monkeypatch.setattr(dbq, 'load_identity_edges', lambda: [edge])
    result = dbq._enrich_candidates_with_local_catalog([source])[0]
    assert result['id'] == 30
    assert result['catalog_exercise_id'] == 'V24ROW-TEST'
    assert not result.get('core_main_role_reviewed')


def test_malformed_details_produce_validation_error_not_type_error():
    from pydantic import ValidationError
    data = block(['chest']).model_dump(by_alias=True)
    data['primaryMuscleDetails'] = [{}]
    with pytest.raises(ValidationError):
        RoutineBlock.model_validate(data)

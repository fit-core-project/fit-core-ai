"""Portable numeric identity/transport checks, NOT recommendation generation.

The v4 routine and substitutions are synthetic serialization inputs. History is
a query-result double, not persistence or a Java/HTTP integration test.
"""
import copy
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from engines import catalog_identity as bridge
from engines.experience_policy import automatic_recommendation_eligibility, main_role_eligibility
from engines.routine_response_v4 import RoutineResponseV4, build_v4_response
from engines.schemas import (
    RoutineBlock, RoutineDraftResponse, RoutineRequest, SetPrescription,
    SubstitutionCandidate,
)


CORE = Path(__file__).resolve().parents[1]
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


@pytest.fixture(scope='module')
def frozen_seed():
    return json.loads((CORE / 'tests/fixtures/stage2_numeric_seed_v1.json').read_text(encoding='utf-8'))


@pytest.fixture
def sources(frozen_seed):
    return {str(r['id']): copy.deepcopy(r) for r in frozen_seed['rows']}


@pytest.fixture(scope='module')
def readonly_core():
    from engines import db_queries

    def connect():
        conn = sqlite3.connect((CORE / 'fit_core.sqlite').as_uri() + '?mode=ro', uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(db_queries, '_connect_local_catalog_db', connect)
        patch.setattr(db_queries, 'SQLITE_DB_PATH', CORE / 'fit_core.sqlite')
        db_queries.clear_local_catalog_enrichment_cache()
        try:
            yield db_queries
        finally:
            db_queries.clear_local_catalog_enrichment_cache()


def request_for(sid, source):
    return RoutineRequest.model_validate({
        'userId': 'synthetic-stage2-numeric', 'targetSplitLabel': 'custom',
        'targetMuscles': [source['primary_muscle']], 'readinessLevel': 'normal',
        'timeAvailableMin': 60, 'goal': 'hypertrophy',
        'preferredExerciseIds': [sid], 'unpreferredExerciseIds': ['98'],
    })


def synthetic_routine(candidates):
    blocks = []
    for index, candidate in enumerate(candidates):
        # A transport sentinel, not a computed or admission-approved alternative.
        alternative = candidates[(index + 1) % len(candidates)]
        blocks.append(RoutineBlock(
            order=index + 1, exercise_id=str(candidate['id']),
            exercise_name=candidate['name_kr'], primary_muscles=[candidate['primary_muscle']],
            default_rest_sec=90, exercise_rationale='Synthetic transport only',
            prescription=[SetPrescription(set_index=1, set_type='working', target_reps=8,
                target_weight_kg=20, target_rir=3, target_rest_sec=90)],
            substitution_candidates=[SubstitutionCandidate(
                exercise_id=str(alternative['id']), exercise_name=alternative['name_kr'],
                reason='Synthetic transport sentinel; not an eligibility decision')],
        ))
    return RoutineDraftResponse(
        generation_status='success', status_reason_code='none', is_fallback=False,
        total_estimated_time=5, summary_title='Synthetic numeric transport',
        rationale_summary=['Not a generated recommendation'], routine_blocks=blocks,
    )


class QueryResult:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


class HistoryResultDouble:
    def __init__(self, blocks):
        self.calls = []
        self.rows = [(b.exercise_id, b.exercise_name, b.primary_muscles[0],
                      20.0 + i, 8 + i, 3, None, False) for i, b in enumerate(blocks)]

    def execute(self, query, params):
        self.calls.append((str(query), params))
        return QueryResult([('synthetic-session',)] if len(self.calls) == 1 else self.rows)


def test_frozen_source_provenance_and_exact_approved_set(frozen_seed, sources):
    provenance = frozen_seed['provenance']
    assert frozen_seed['version'] == 'stage2-numeric-seed-v1'
    assert provenance['repository'] == 'fit-core-backend'
    assert provenance['commit'] == '1490c91a08b674291754ced722af8dd979cf116c'
    assert provenance['path'] == 'src/main/resources/db/migration/V6__seed_exercise_catalog.sql'
    assert provenance['sha256'] == 'd54e3e68b44c72ff9478695c11bca1d867c420bc482ce12d546d849d181d02ef'
    assert provenance['identityFields'] == list(bridge.IDENTITY_FIELDS)
    frozen_rows = json.dumps(frozen_seed['rows'], ensure_ascii=True, sort_keys=True,
                             separators=(',', ':')).encode()
    assert hashlib.sha256(frozen_rows).hexdigest() == '40b836f62a809f283b4c92fee121d958f51ddb5cf0e4e8e295705e3cc6175321'
    assert len(frozen_seed['rows']) == len(sources) == 12
    assert set(sources) == set(EXPECTED) | {'98'}
    assert all(set(row) == set(provenance['columns']) for row in sources.values())
    approved = [e for e in bridge.load_identity_edges() if e['status'] == 'approved']
    assert len(approved) == 11
    assert {(e['sourceId'], e['catalogId']) for e in approved} == {
        (sid, cid) for sid, (cid, _) in EXPECTED.items()}
    for edge in approved:
        assert bridge.identity_snapshot(sources[edge['sourceId']]) == edge['sourceIdentity']


@pytest.mark.parametrize('sid', EXPECTED)
def test_approved11_request_enrichment_v4_substitution_json_history(readonly_core, sources, sid):
    original = copy.deepcopy(sources)
    request = request_for(sid, sources[sid])
    ordered = [sources[sid]] + [sources[key] for key in EXPECTED if key != sid]
    candidates = readonly_core._enrich_candidates_with_local_catalog(ordered)
    assert sources == original
    assert len(candidates) == 11
    for source, candidate in zip(ordered, candidates):
        cid, primary = EXPECTED[str(source['id'])]
        assert type(candidate['id']) is int and candidate['id'] == source['id']
        assert candidate['catalog_exercise_id'] == cid
        assert candidate['primary_muscle'] == primary
        assert candidate['substitute_exercise_ids'] == source['substitute_exercise_ids']
    routine = synthetic_routine(candidates)
    envelope = build_v4_response(request, routine, local_constraint_profile={})
    payload = json.loads(envelope.model_dump_json(by_alias=True))
    restored = RoutineResponseV4.model_validate(payload)
    assert restored == envelope
    assert payload['contractVersion'] == 'routine-core-v4-draft-v3'
    blocks = payload['routine']['routineBlocks']
    ids = [str(c['id']) for c in candidates]
    assert [b['exerciseId'] for b in blocks] == ids
    assert blocks[0]['exerciseId'] == request.preferred_exercise_ids[0] == sid
    assert request.unpreferred_exercise_ids == ['98']
    assert [b['substitutionCandidates'][0]['exerciseId'] for b in blocks] == ids[1:] + ids[:1]
    assert all(type(b['exerciseId']) is str and b['exerciseId'].isdigit() for b in blocks)
    assert all(b['primaryMuscles'] == [EXPECTED[b['exerciseId']][1]] for b in blocks)
    history_db = HistoryResultDouble(restored.routine.routine_blocks)
    history = readonly_core.get_recent_sets(history_db, request.user_id)
    assert [r.exercise_id for r in history] == ids
    assert [r.weight_kg for r in history] == [20.0 + i for i in range(11)]
    assert [r.reps for r in history] == [8 + i for i in range(11)]
    assert history_db.calls[0][1] == {'uid': request.user_id}
    assert history_db.calls[1][1] == {'sid0': 'synthetic-session', 'limit': 30}


@pytest.mark.parametrize('sid', EXPECTED)
@pytest.mark.parametrize('field', bridge.IDENTITY_FIELDS)
@pytest.mark.parametrize('side', ['source', 'catalog'])
def test_stale_identity_cannot_enrich(readonly_core, sources, monkeypatch, sid, field, side):
    source = sources[sid]
    if side == 'source':
        source[field] += ' stale'
    else:
        cache = readonly_core._load_local_catalog_enrichment_cache()
        changed = dict(cache, by_id=dict(cache['by_id']))
        key = EXPECTED[sid][0].lower()
        changed['by_id'][key] = dict(cache['by_id'][key])
        changed['by_id'][key]['catalog_' + field] += ' stale'
        monkeypatch.setattr(readonly_core, '_load_local_catalog_enrichment_cache', lambda: changed)
    candidate, = readonly_core._enrich_candidates_with_local_catalog([source])
    assert candidate['id'] == int(sid)
    assert 'catalog_exercise_id' not in candidate


@pytest.mark.parametrize('sid', EXPECTED)
def test_candidate_status_and_revoked_enrichment_fail_closed(readonly_core, sources, monkeypatch, sid):
    enriched, = readonly_core._enrich_candidates_with_local_catalog([sources[sid]])
    edges = copy.deepcopy(bridge.load_identity_edges())
    for edge in edges:
        if edge['sourceId'] == sid:
            edge['status'] = 'candidate'
            edge['review'] = None
    monkeypatch.setattr(readonly_core, 'load_identity_edges', lambda: edges)
    candidate, = readonly_core._enrich_candidates_with_local_catalog([sources[sid]])
    assert candidate['id'] == int(sid) and 'catalog_exercise_id' not in candidate
    assert readonly_core._enrich_candidates_with_local_catalog([enriched]) == []


def test_existing_candidate_and_rejected_back_squat_do_not_enrich(readonly_core, sources):
    edges = [e for e in bridge.load_identity_edges() if e['sourceId'] == '98']
    assert {(e['catalogId'], e['status']) for e in edges} == {
        ('V24ROW-006191', 'rejected'), ('V24ROW-003251', 'candidate')}
    candidate, = readonly_core._enrich_candidates_with_local_catalog([sources['98']])
    assert candidate['id'] == 98 and 'catalog_exercise_id' not in candidate


def test_duplicate_names_and_n_to_one_never_merge_numeric_history(readonly_core, sources, monkeypatch):
    assert sources['74']['name_en'] == sources['124']['name_en']
    candidates = readonly_core._enrich_candidates_with_local_catalog([sources['74'], sources['124']])
    assert [c['id'] for c in candidates] == [74, 124]
    assert [c['catalog_exercise_id'] for c in candidates] == ['V24ROW-004791'] * 2
    request = request_for('74', sources['74'])
    restored = RoutineResponseV4.model_validate_json(build_v4_response(
        request, synthetic_routine(candidates)).model_dump_json(by_alias=True))
    blocks = restored.routine.routine_blocks
    assert [b.exercise_id for b in blocks] == ['74', '124']
    assert [b.substitution_candidates[0].exercise_id for b in blocks] == ['124', '74']
    history = readonly_core.get_recent_sets(HistoryResultDouble(blocks), request.user_id)
    assert [(r.exercise_id, r.weight_kg, r.reps) for r in history] == [('74', 20.0, 8), ('124', 21.0, 9)]
    lookalike = dict(sources['74'], id=999074)
    candidate, = readonly_core._enrich_candidates_with_local_catalog([lookalike])
    assert candidate['id'] == 999074 and 'catalog_exercise_id' not in candidate
    monkeypatch.setattr(readonly_core, 'load_identity_edges', lambda: [])
    unresolved = readonly_core._enrich_candidates_with_local_catalog([sources['74'], sources['124']])
    assert [c['id'] for c in unresolved] == [74, 124]
    assert all('catalog_exercise_id' not in c for c in unresolved)


def test_competing_one_to_many_approvals_require_selector(readonly_core, sources, monkeypatch):
    edges = copy.deepcopy(bridge.load_identity_edges())
    other = next(e for e in edges if e['sourceId'] == '151' and e['catalogId'] == 'V24ROW-007275')
    assert other['status'] == 'candidate'
    other['status'] = 'approved'  # In-memory conflict only; never an approval on disk.
    monkeypatch.setattr(readonly_core, 'load_identity_edges', lambda: edges)
    candidate, = readonly_core._enrich_candidates_with_local_catalog([sources['151']])
    assert candidate['id'] == 151 and 'catalog_exercise_id' not in candidate


@pytest.mark.parametrize('sid', ['125', '151'])
@pytest.mark.parametrize('level', ['beginner', 'intermediate', 'advanced'])
def test_125_151_identity_does_not_grant_admission(readonly_core, sources, sid, level):
    candidate, = readonly_core._enrich_candidates_with_local_catalog([sources[sid]])
    assert candidate['catalog_exercise_id'] == EXPECTED[sid][0]
    assert candidate['main_routine_default_allowed'] == 0
    assert not automatic_recommendation_eligibility(candidate, level).eligible
    assert not main_role_eligibility(candidate, level, 'hypertrophy').eligible

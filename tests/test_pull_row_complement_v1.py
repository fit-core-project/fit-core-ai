from copy import deepcopy
import sqlite3

import pytest

from engines import candidate_ranker as ranker, db_queries as dbq
from engines.pull_composition import (
    BENT_OVER_ID, SUPPORTED_ID, REVIEWED, VERSION, complement_scope,
    complementary_pair, reviewed_row_setup, row_setup_patch,
)
from engines.routine_blueprint import build_routine_blueprint, build_blueprint_candidate_bundles
from engines.routine_selection import adapt_slot_selection_output
from engines.schemas import LLMRoutineSelectionOutput


def raw(eid):
    with sqlite3.connect(f'file:{dbq.SQLITE_DB_PATH}?mode=ro', uri=True) as conn:
        conn.row_factory = sqlite3.Row
        return dict(conn.execute('SELECT * FROM exercise_tier WHERE id=?', (eid,)).fetchone())


def catalog():
    return dbq._enrich_candidates_with_local_catalog([
        raw(eid) for eid in (BENT_OVER_ID, SUPPORTED_ID, 'V24ROW-005645',
                            'V24ROW-005499', 'V24ROW-005539', 'V24ROW-006024')])


def test_real_review_preserves_catalog_and_health_metadata():
    before = raw(SUPPORTED_ID)
    candidates = catalog()
    row = next(c for c in candidates if c['id'] == SUPPORTED_ID)
    assert reviewed_row_setup(row) == 'chest_supported'
    assert row['main_routine_default_allowed'] == 1
    assert row['beginner_default_allowed'] == 0
    assert row['name_kr'] == '머신 체스트 서포티드 로우'
    assert raw(SUPPORTED_ID) == before
    assert row['equipment_req'] == 'MACHINE'
    repeated = dbq._enrich_candidates_with_local_catalog([row])[0]
    for key in ('core_pull_row_setup', 'core_pull_row_review_version', 'core_admission_scope',
                'main_routine_default_allowed', 'beginner_default_allowed', 'lumbar_load', 'axial_load'):
        assert repeated.get(key) == row.get(key)
    assert reviewed_row_setup(next(c for c in candidates if c['id'] == BENT_OVER_ID)) == 'bent_over'
    assert reviewed_row_setup(next(c for c in candidates if c['id'] == 'V24ROW-005539')) is None


@pytest.mark.parametrize('key,value', [
    ('primary_muscle', 'biceps'), ('equipment_req', 'DUMBBELL'),
    ('movement_type', 'ISOLATION'), ('prescription_mode', 'duration'),
    ('main_routine_default_allowed', 1), ('beginner_default_allowed', 1),
])
def test_effective_identity_drift_fails_closed(key, value):
    ko, en, equipment, _, allowed = REVIEWED[SUPPORTED_ID]
    row = dict(catalog_exercise_id=SUPPORTED_ID, catalog_name_kr=ko, catalog_name_en=en, catalog_primary_muscle='upper-back',
               catalog_secondary_muscle='biceps, trapezius')
    effective = dict(primary_muscle='upper-back', equipment_req=equipment, movement_type='COMPOUND',
                     prescription_mode='reps', main_routine_default_allowed=allowed, beginner_default_allowed=0)
    assert row_setup_patch(SUPPORTED_ID, row, effective, dict(name_kr=ko, name_en=en))
    effective[key] = value
    assert row_setup_patch(SUPPORTED_ID, row, effective, dict(name_kr=ko, name_en=en)) == {}


@pytest.mark.parametrize('mismatch', ['missing', 'renamed'])
def test_reenrichment_revokes_review_when_source_unavailable(monkeypatch, mismatch):
    candidate = next(c for c in catalog() if c['id'] == SUPPORTED_ID)
    source = deepcopy(dbq._load_source_equipment_by_id())
    if mismatch == 'missing':
        source = None
    else:
        source[SUPPORTED_ID]['name_en'] = 'Generic machine row'
    monkeypatch.setattr(dbq, '_load_source_equipment_by_id', lambda: source)
    result = dbq._enrich_candidates_with_local_catalog([candidate])[0]
    assert result['main_routine_default_allowed'] == 0
    assert 'core_pull_row_setup' not in result


SCOPE = dict(split='pull', experience='advanced', goal='hypertrophy', readiness='normal',
             minutes=60, structure='standard')


@pytest.mark.parametrize('delta,expected', [
    ({}, True), ({'readiness': 'high'}, True), ({'minutes': 90}, True),
    ({'minutes': 59}, False), ({'experience': 'intermediate'}, False),
    ({'experience': 'beginner'}, False), ({'goal': 'strength'}, False),
    ({'readiness': 'low'}, False), ({'split': 'upper'}, False),
    ({'structure': 'density'}, False), ({'style': 'yoga_pilates'}, False),
    ({'focus': 'midBackEmphasis'}, False), ({'method': 'pausedReps'}, False),
    ({'adjuncts': ['cooldownRecovery']}, False), ({'pain': ['shoulder']}, False),
    ({'injuries': ['active']}, False), ({'doms': {'biceps': 1}}, False),
    ({'constraints': True}, False),
])
def test_scope_boundaries(delta, expected):
    assert complement_scope(**{**SCOPE, **delta}) == expected


RANK = dict(target_split_label='pull', experience_level='advanced', goal='hypertrophy',
            readiness_level='normal', time_available_min=60, session_structure='standard', training_style='gym')


@pytest.mark.parametrize('delta', [
    {'experience_level': 'intermediate'}, {'experience_level': 'beginner'},
    {'time_available_min': 59}, {'readiness_level': 'low'}, {'goal': 'strength'},
    {'target_split_label': 'upper'}, {'session_structure': 'density'},
    {'specialization_focus': 'midBackEmphasis'}, {'doms_db': {'upper-back': 1}},
    {'mobility_limits': ['limited_shoulder_flexion']},
    {'shoulder_condition_status': 'activeUncleared'},
    {'adjunct_options': ['cooldownRecovery']},
    {'available_equipment': ['DUMBBELL']}, {'blocked_equipment': ['MACHINE']},
    {'unpreferred_exercise_ids': [SUPPORTED_ID]},
])
def test_full_ranker_scope_and_exclusion(delta):
    result = ranker.score_candidate_exercises(catalog(), ['upper-back', 'biceps'], **{**RANK, **delta})
    assert SUPPORTED_ID not in {c['id'] for c in result}


def blueprint(**delta):
    values = {k: v for k, v in RANK.items() if k != 'training_style'}
    values['method_mode'] = 'off'
    return build_routine_blueprint(target_muscles=['upper-back', 'biceps', 'trapezius', 'forearm', 'back-deltoids'], **{**values, **delta})


def test_real_candidates_have_separate_roles_without_arbitrary_row_duplicates():
    candidates = ranker.score_candidate_exercises(catalog(), ['upper-back', 'biceps'], **RANK)
    assert {SUPPORTED_ID, BENT_OVER_ID} <= {c['id'] for c in candidates}
    bp = blueprint()
    assert bp.formula_version == 'gym-routine-blueprint-v11'
    bundles = build_blueprint_candidate_bundles(bp, candidates)
    supported = next(b for b in bundles if b['slotId'] == 'pull_supported_row')
    assert [c['exerciseId'] for c in supported['candidates']] == [SUPPORTED_ID]
    # An empty model choice tests deterministic role recovery, not a live LLM.
    result = adapt_slot_selection_output(LLMRoutineSelectionOutput.model_validate({'selections': []}),
        blueprint=bp, slot_candidate_bundles=bundles, ranked_candidates=candidates)
    selected = {c.exercise_id for c in result.exercises}
    assert {BENT_OVER_ID, SUPPORTED_ID, 'V24ROW-005645', 'V24ROW-006024'} <= selected
    assert not {'V24ROW-005499', 'V24ROW-005539'} & selected
    assert len(result.exercises) <= bp.target_exercise_count


def test_generic_machine_row_is_not_a_verified_complement():
    candidates = catalog()
    supported = next(c for c in candidates if c['id'] == SUPPORTED_ID)
    bent = next(c for c in candidates if c['id'] == BENT_OVER_ID)
    generic = next(c for c in candidates if c['id'] == 'V24ROW-005539')
    assert complementary_pair([bent, supported])
    assert not complementary_pair([generic, supported])
    assert not complementary_pair([bent])
    assert not complementary_pair([bent, supported, generic])
    assert not complementary_pair([bent, supported, bent])


def test_intermediate_blueprint_retains_prior_roles():
    bp = blueprint(experience_level='intermediate', time_available_min=50)
    assert bp.formula_version == 'gym-routine-blueprint-v9'
    assert [s.slot_id for s in bp.slots] == ['pull_main', 'pull_vertical_main', 'biceps_direct']


def test_auto_method_blueprint_does_not_expand():
    bp = blueprint(method_mode='auto')
    assert bp.training_method is not None
    assert bp.formula_version == 'gym-routine-blueprint-v9'
    assert 'pull_supported_row' not in {s.slot_id for s in bp.slots}


def test_unreviewed_main_row_cannot_pair_with_supported_row():
    candidates = [c for c in catalog() if c['id'] != BENT_OVER_ID]
    bp = blueprint()
    bundles = build_blueprint_candidate_bundles(bp, candidates)
    result = adapt_slot_selection_output(LLMRoutineSelectionOutput.model_validate({'selections': []}),
        blueprint=bp, slot_candidate_bundles=bundles, ranked_candidates=candidates)
    assert SUPPORTED_ID not in {c.exercise_id for c in result.exercises}


@pytest.mark.parametrize('pool_size', [8, 12, 20])
def test_full_database_query_preserves_complement_through_pool_compression(pool_size):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    engine = create_engine('sqlite://', creator=lambda: sqlite3.connect(f'file:{dbq.SQLITE_DB_PATH}?mode=ro', uri=True))
    targets = ['upper-back', 'trapezius', 'biceps', 'forearm', 'back-deltoids']
    try:
        with Session(engine) as db:
            candidates = dbq.get_candidate_exercises(db, targets, [], [])
        ranked = ranker.score_candidate_exercises(candidates, targets, top_n=pool_size, **RANK)
        assert len(ranked) <= pool_size
        assert {SUPPORTED_ID, BENT_OVER_ID} <= {c['id'] for c in ranked}
        bp = blueprint()
        bundles = build_blueprint_candidate_bundles(bp, ranked)
        result = adapt_slot_selection_output(LLMRoutineSelectionOutput.model_validate({'selections': []}),
            blueprint=bp, slot_candidate_bundles=bundles, ranked_candidates=ranked)
        assert {SUPPORTED_ID, BENT_OVER_ID} <= {c.exercise_id for c in result.exercises}
    finally:
        engine.dispose()


@pytest.mark.parametrize('delta', [
    {'available_equipment': ['MACHINE', 'CABLE']},
    {'unpreferred_exercise_ids': [BENT_OVER_ID]},
])
def test_filtered_main_setup_removes_complement_at_ranking(delta):
    ranked = ranker.score_candidate_exercises(catalog(), ['upper-back', 'biceps'], **{**RANK, **delta})
    assert not {BENT_OVER_ID, SUPPORTED_ID} & {c['id'] for c in ranked}


def row_output(rows):
    from engines.schemas import LLMExercisePlan, LLMRoutineOutput

    return LLMRoutineOutput(total_estimated_time=20, summary_title='검증',
        rationale_summary=[], warnings=[], exercises=[LLMExercisePlan(
            exercise_id=c['id'], exercise_name=c['name_kr'], primary_muscles=[c['primary_muscle']],
            movement_type=c['movement_type'], equipment_type=c['equipment_req'],
            target_reps=10, sets=3, rest_time_sec=90, exercise_rationale='합성 검증',
        ) for c in rows])


@pytest.mark.parametrize('enabled', [False, True])
def test_final_validator_keeps_reviewed_pair_only_when_enabled(enabled):
    from engines.llm_parser import validate_and_repair_routine_output

    rows = [c for c in catalog() if c['id'] in {BENT_OVER_ID, SUPPORTED_ID}]
    original = deepcopy(rows)
    result, reason = validate_and_repair_routine_output(row_output(rows), rows, [], [],
        allow_reviewed_pull_pair=enabled)
    assert reason == 'none' and result is not None
    assert len(result.exercises) == (2 if enabled else 1)
    assert rows == original


@pytest.mark.parametrize('case', ['generic', 'third', 'duplicate', 'unreviewed'])
def test_final_validator_rejects_unreviewed_or_extra_row_pair(case):
    from engines.llm_parser import validate_and_repair_routine_output

    by_id = {c['id']: c for c in catalog()}
    rows = [by_id[BENT_OVER_ID], by_id[SUPPORTED_ID]]
    generic = by_id['V24ROW-005539']
    if case == 'generic':
        rows[0] = generic
    elif case == 'third':
        rows.append(generic)
    elif case == 'duplicate':
        rows.append(rows[0])
    else:
        rows[1].pop('core_pull_row_review_version')
    result, reason = validate_and_repair_routine_output(row_output(rows), rows, [], [],
        allow_reviewed_pull_pair=True)
    assert reason == 'none' and result is not None
    assert len(result.exercises) == 1


@pytest.mark.parametrize('condition', ['equipment', 'injury', 'doms'])
def test_final_validator_pair_does_not_bypass_health_or_equipment(condition):
    from engines.llm_parser import validate_and_repair_routine_output
    from engines.schemas import PainAreaEntry

    rows = [c for c in catalog() if c['id'] in {BENT_OVER_ID, SUPPORTED_ID}]
    result, reason = validate_and_repair_routine_output(row_output(rows), rows,
        ['MACHINE'] if condition == 'equipment' else [],
        [PainAreaEntry(body_part='upper-back')] if condition == 'injury' else [],
        doms_db={'upper-back': 3} if condition == 'doms' else {}, allow_reviewed_pull_pair=True)
    if condition == 'equipment':
        assert result is not None and {p.exercise_id for p in result.exercises} == {BENT_OVER_ID}
    else:
        assert result is None and reason == ('schemaError' if condition == 'doms' else 'emptyCandidate')


@pytest.mark.parametrize('bp_kind', ['expanded', 'intermediate', 'missing'])
def test_pipeline_enables_pair_only_for_resolved_composition(monkeypatch, bp_kind):
    from engines import llm_parser, routine_pipeline
    from engines.schemas import RoutineRequest

    rows = [c for c in catalog() if c['id'] in {BENT_OVER_ID, SUPPORTED_ID}]
    bp = None if bp_kind == 'missing' else blueprint(
        experience_level='intermediate' if bp_kind == 'intermediate' else 'advanced')
    req = RoutineRequest(time_available_min=60, target_split_label='pull', goal='hypertrophy',
                         experience_level='advanced', method_mode='off')
    seen = {}
    real_validate = llm_parser.validate_and_repair_routine_output

    def capture(*args, **kwargs):
        seen.update(kwargs)
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(llm_parser, 'validate_and_repair_routine_output', capture)
    monkeypatch.setattr(routine_pipeline, '_deterministic_or_fallback', lambda result, *a, **k: result)
    result = routine_pipeline._validate_or_fallback(row_output(rows), req, rows, 20, {},
        'hypertrophy', ['upper-back'], None, None, routine_blueprint=bp)
    assert seen['allow_reviewed_pull_pair'] is (bp_kind == 'expanded')
    assert len(result.exercises) == (2 if bp_kind == 'expanded' else 1)


def test_full_sql_selection_validation_prescription_public_response_keeps_pair():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from engines.routine_pipeline import _validate_or_fallback
    from engines.schemas import RoutineRequest

    engine = create_engine('sqlite://', creator=lambda: sqlite3.connect(f'file:{dbq.SQLITE_DB_PATH}?mode=ro', uri=True))
    targets = ['upper-back', 'trapezius', 'biceps', 'forearm', 'back-deltoids']
    try:
        with Session(engine) as db:
            candidates = dbq.get_candidate_exercises(db, targets, [], [])
        ranked = ranker.score_candidate_exercises(candidates, targets, top_n=12, **RANK)
        bp = blueprint()
        bundles = build_blueprint_candidate_bundles(bp, ranked)
        selected = adapt_slot_selection_output(LLMRoutineSelectionOutput.model_validate({'selections': []}),
            blueprint=bp, slot_candidate_bundles=bundles, ranked_candidates=ranked)
        req = RoutineRequest(time_available_min=60, target_split_label='pull', target_muscles=targets,
            goal='hypertrophy', experience_level='advanced', method_mode='off')
        response = _validate_or_fallback(selected, req, ranked, bp.max_total_sets, {},
            'hypertrophy', targets, None, None, routine_blueprint=bp).model_dump(by_alias=True)
        assert response['generationStatus'] == 'success'
        ids = {b['exerciseId'] for b in response['routineBlocks']}
        assert {BENT_OVER_ID, SUPPORTED_ID, 'V24ROW-005645'} <= ids
        by_id = {c['id']: c for c in ranked}
        assert any(by_id[eid]['primary_muscle'] == 'biceps'
                   and ranker.candidate_selection_pattern_key(by_id[eid]) == 'biceps_curl' for eid in ids)
        assert sum(ranker.candidate_selection_pattern_key(by_id[eid]) == 'vertical_pull' for eid in ids) == 1
        assert response['totalEstimatedTime'] <= req.time_available_min
        assert sum(len(b['prescription']) for b in response['routineBlocks']) <= bp.max_total_sets
    finally:
        engine.dispose()

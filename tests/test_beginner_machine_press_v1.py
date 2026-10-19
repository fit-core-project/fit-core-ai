import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from engines import db_queries
from engines.beginner_machine_press import EXERCISE_ID, NAME_EN, NAME_KO, admission_patch
from engines.candidate_ranker import score_candidate_exercises
from engines.registry import SPLIT_LABEL_TO_MUSCLES
from engines.routine_blueprint import build_blueprint_candidate_bundles, build_routine_blueprint
from engines.schemas import PainAreaEntry

ROOT = Path(__file__).resolve().parents[1]
TARGETS = SPLIT_LABEL_TO_MUSCLES['full_body']
EQUIPMENT = ['MACHINE', 'DUMBBELL', 'BARBELL', 'CABLE', 'BODYWEIGHT', 'SMITH_MACHINE']


@pytest.fixture(scope='module')
def candidates():
    engine = create_engine('sqlite://', creator=lambda: sqlite3.connect(
        f'file:{ROOT / "fit_core.sqlite"}?mode=ro', uri=True))
    try:
        with Session(engine) as session:
            return db_queries.get_candidate_exercises(session, TARGETS, [], [], available_equipment=EQUIPMENT)
    finally:
        engine.dispose()


def rank(candidates, **overrides):
    args = dict(experience_level='beginner', goal='hypertrophy', readiness_level='normal',
                training_style='gym', target_split_label='full_body', session_structure='standard',
                available_equipment=EQUIPMENT, blocked_equipment=[], pain_areas=[], doms_db={},
                time_available_min=50, top_n=12)
    args.update(overrides)
    return score_candidate_exercises(candidates, TARGETS, **args)


def test_exact_candidate_reaches_full_body_vertical_slot(candidates):
    candidate = next(c for c in candidates if c['id'] == EXERCISE_ID)
    assert candidate['beginner_default_allowed'] == 1
    assert candidate['main_routine_default_allowed'] == 1
    assert candidate['pain_risk_level'] == 'high'
    assert candidate['injury_caution_level'] == 'high'
    assert candidate['overhead_position_required'] == 1
    assert 'side-deltoids' in candidate['secondary_muscle']
    assert candidate['video_url'] is None
    ranked = rank(candidates)
    assert EXERCISE_ID in {c['id'] for c in ranked}
    blueprint = build_routine_blueprint(target_split_label='full_body', target_muscles=[],
        experience_level='beginner', goal='hypertrophy', readiness_level='normal',
        time_available_min=50, training_style='gym')
    bundles = {b['slotId']: b for b in build_blueprint_candidate_bundles(blueprint, ranked)}
    assert EXERCISE_ID in {c['exerciseId'] for c in bundles['full_vertical_push']['candidates']}


@pytest.mark.parametrize('override', [
    {'experience_level': 'intermediate'}, {'experience_level': 'advanced'},
    {'target_split_label': 'push'}, {'target_split_label': 'shoulder'},
    {'goal': 'strength'}, {'readiness_level': 'low'},
    {'session_structure': 'superset'}, {'adjunct_options': ['cooldown_stretch']},
    {'mobility_limits': ['limited_shoulder_flexion']},
    {'condition_policies': [{'policy_action': 'Caution'}]},
    {'doms_db': {'front-deltoids': 1}}, {'doms_db': {'side-deltoids': 3}},
    {'pain_areas': [PainAreaEntry(body_part='front-deltoids')]},
    {'available_equipment': ['DUMBBELL']}, {'blocked_equipment': ['MACHINE']},
    {'training_method': 'pausedReps'},
])
def test_candidate_denied_outside_reviewed_scope(candidates, override):
    candidate = next(c for c in candidates if c['id'] == EXERCISE_ID)
    assert not rank([candidate], **override)


def test_pronated_variant_not_silently_admitted(candidates):
    candidate = next(c for c in candidates if c['id'] == 'V24ROW-000367')
    assert candidate['main_routine_default_allowed'] == 0
    assert candidate['beginner_default_allowed'] == 0


def identity():
    row = dict(catalog_exercise_id=EXERCISE_ID, catalog_name_kr=NAME_KO, catalog_name_en=NAME_EN,
               catalog_primary_muscle='front-deltoids', catalog_secondary_muscle='upper-back, triceps, trapezius')
    effective = dict(primary_muscle='front-deltoids', secondary_muscle='upper-back, triceps, trapezius',
                     equipment_req='MACHINE', movement_type='COMPOUND', prescription_mode='reps',
                     main_routine_default_allowed=0, beginner_default_allowed=0)
    source = dict(name_kr=NAME_KO, name_en=NAME_EN, equipment='머신')
    return row, effective, source


@pytest.mark.parametrize('section,key,value', [
    (0, 'catalog_exercise_id', 'V24ROW-000367'), (0, 'catalog_name_kr', 'other'),
    (0, 'catalog_name_en', 'other'), (0, 'catalog_primary_muscle', 'chest'),
    (0, 'catalog_secondary_muscle', ''), (1, 'primary_muscle', 'chest'),
    (1, 'equipment_req', 'SMITH_MACHINE'), (1, 'movement_type', 'ISOLATION'),
    (1, 'prescription_mode', 'duration'), (1, 'main_routine_default_allowed', 1),
    (1, 'beginner_default_allowed', 1), (2, 'name_kr', 'other'),
    (2, 'name_en', 'other'), (2, 'equipment', '스미스머신'),
])
def test_identity_drift_fails_closed(section, key, value):
    parts = identity()
    assert admission_patch(EXERCISE_ID, *parts)
    parts[section][key] = value
    assert not admission_patch(EXERCISE_ID, *parts)


def test_unverified_source_fails_closed():
    row, effective, _ = identity()
    assert not admission_patch(EXERCISE_ID, row, effective, None)


def test_reenrichment_revokes_beginner_admission_without_source(candidates, monkeypatch):
    candidate = next(c for c in candidates if c['id'] == EXERCISE_ID)
    monkeypatch.setattr(db_queries, '_load_source_equipment_by_id', lambda: None)
    updated = db_queries._enrich_candidates_with_local_catalog([candidate])[0]
    assert updated['beginner_default_allowed'] == 0
    assert updated['main_routine_default_allowed'] == 0
    assert 'core_admission_scope' not in updated
    assert 'side-deltoids' not in updated['secondary_muscle']


def test_reenrichment_without_catalog_revokes_flags(candidates, monkeypatch):
    candidate = next(c for c in candidates if c['id'] == EXERCISE_ID)
    monkeypatch.setattr(db_queries, '_load_local_catalog_enrichment_cache', lambda: None)
    updated = db_queries._enrich_candidates_with_local_catalog([candidate])[0]
    assert updated['beginner_default_allowed'] == 0
    assert updated['main_routine_default_allowed'] == 0

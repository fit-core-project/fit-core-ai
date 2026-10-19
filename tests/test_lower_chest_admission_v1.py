import json
from pathlib import Path
import sqlite3

import pytest

from engines import db_queries, focus_catalog_admission
from engines.candidate_ranker import score_candidate_exercises, candidate_selection_pattern_key
from engines.exercise_display import candidate_display_name
from engines.routine_blueprint import build_routine_blueprint, build_blueprint_candidate_bundles
from engines.schemas import PainAreaEntry
from engines.specialization_policy import candidate_matches_specialization_focus

ROOT = Path(__file__).resolve().parents[1]
IDS = {'V24ROW-005299', 'V24ROW-005319'}


def source_rows():
    with sqlite3.connect(f'file:{ROOT / "fit_core.sqlite"}?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute("SELECT * FROM exercise_tier WHERE primary_muscle='chest'")]


@pytest.fixture(scope='module')
def candidates():
    return db_queries._enrich_candidates_with_local_catalog(source_rows())


def rank(rows, **overrides):
    options = dict(available_equipment=['DUMBBELL', 'CABLE', 'MACHINE', 'BARBELL'],
                   blocked_equipment=[], pain_areas=[], doms_db={}, readiness_level='normal',
                   goal='hypertrophy', experience_level='advanced', training_style='gym',
                   target_split_label='chest', specialization_focus='lowerChest',
                   session_structure='standard', time_available_min=60, top_n=24)
    options.update(overrides)
    return score_candidate_exercises(rows, ['chest'], **options)


def bundles(rows, *, focus='lowerChest'):
    blueprint = build_routine_blueprint(
        target_muscles=['chest'], target_split_label='chest', time_available_min=60,
        goal='hypertrophy', experience_level='advanced', readiness_level='normal',
        training_style='gym', session_structure='standard', specialization_focus=focus)
    result = build_blueprint_candidate_bundles(blueprint, rows)
    return blueprint, result


def test_identity_bound_focus_and_searchable_names(candidates):
    names = {'V24ROW-005299': '덤벨 디클라인 플라이', 'V24ROW-005319': '케이블 하이투로우 플라이'}
    reviewed = [row for row in candidates if row['id'] in IDS]
    assert len(reviewed) == 2
    for row in reviewed:
        assert row['core_chest_focus_reviewed'] == 1
        assert row['core_admission_scope'] == 'lowerChest'
        assert row['main_routine_default_allowed'] == 1
        assert row['beginner_default_allowed'] == 0
        assert row['pain_risk_level'] == 'medium'
        assert row['injury_caution_level'] == 'medium'
        assert row['prescription_mode'] == 'reps'
        assert row['movement_type'] == 'ISOLATION'
        assert candidate_display_name(row) == names[row['id']]
        assert candidate_matches_specialization_focus(row, 'lowerChest')
        assert not candidate_matches_specialization_focus(row, 'upperChest')
        assert candidate_selection_pattern_key(row) == 'chest_fly'
    assert IDS <= {row['id'] for row in rank(reviewed)}


@pytest.mark.parametrize('override', [
    {'specialization_focus': None}, {'specialization_focus': 'upperChest'},
    {'experience_level': 'beginner'}, {'experience_level': 'intermediate'},
    {'readiness_level': 'low'}, {'goal': 'strength'}, {'session_structure': 'superset'},
    {'doms_db': {'chest': 1}}, {'doms_db': {'chest': 2}}, {'doms_db': {'chest': 3}},
    {'mobility_limits': ['limited_shoulder_flexion']},
    {'pain_areas': [PainAreaEntry(body_part='front-deltoids')]},
    {'pain_areas': [PainAreaEntry(body_part='chest')]},
    {'condition_policies': [{'condition_code': 'shoulder_issue'}]},
])
def test_new_admission_is_not_global_or_recovery_clearance(candidates, override):
    assert not (IDS & {row['id'] for row in rank(candidates, **override)})


@pytest.mark.parametrize('equipment, expected', [
    (['DUMBBELL'], {'V24ROW-005299'}), (['CABLE'], {'V24ROW-005319'}),
    (['MACHINE'], set()), (['BARBELL'], set()),
])
def test_equipment_still_controls_reviewed_candidates(candidates, equipment, expected):
    assert IDS & {row['id'] for row in rank(candidates, available_equipment=equipment)} == expected


def test_blocked_equipment_and_unpreferred_id_remain_exclusions(candidates):
    assert not IDS & {row['id'] for row in rank(candidates, blocked_equipment=['DUMBBELL', 'CABLE'])}
    assert not IDS & {row['id'] for row in rank(candidates, unpreferred_exercise_ids=list(IDS))}


def test_lower_focus_isolation_does_not_add_fourth_role(candidates):
    blueprint, result = bundles(rank(candidates))
    assert blueprint.target_exercise_count == 3
    assert len(blueprint.slots) == 3
    isolation = next(b for b in result if b['slotId'] == 'chest_isolation')
    assert isolation['candidates'][0]['exerciseId'] in IDS


def test_missing_focus_isolation_keeps_existing_generic_role(candidates):
    rows = rank([row for row in candidates if row['id'] not in IDS])
    _, result = bundles(rows)
    isolation = next(b for b in result if b['slotId'] == 'chest_isolation')
    assert isolation['candidates']
    assert not IDS & {row['exerciseId'] for row in isolation['candidates']}


def test_manifest_and_catalog_drift_fail_closed(monkeypatch, tmp_path):
    manifest = tmp_path / 'changed.json'
    manifest.write_text('{}')
    monkeypatch.setattr(focus_catalog_admission, 'REVIEW_PATH', manifest)
    rows = db_queries._enrich_candidates_with_local_catalog(source_rows())
    assert all(row['main_routine_default_allowed'] == 0 for row in rows if row['id'] in IDS)
    assert focus_catalog_admission.admission_patch('V24ROW-005319', {}, {}, ROOT / 'missing.csv') == {}


def test_v1_reviews_preserved_in_v2():
    old = json.loads((ROOT / 'data/provenance/focus-basic-admission-review-v1-2026-09-15.json').read_text())
    new = json.loads(focus_catalog_admission.REVIEW_PATH.read_text())
    assert new['reviews'][:len(old['reviews'])] == old['reviews']
    assert {row['catalogExerciseId'] for row in new['reviews']} == IDS | {
        'V24ROW-006682', 'V24ROW-006024', 'V24ROW-006121'}


def test_forged_and_revoked_scope_does_not_survive_reenrichment():
    row = next(r for r in source_rows() if r['id'] == 'V24ROW-005315')
    row.update(core_chest_focus_reviewed=1, core_admission_scope='lowerChest',
               specialization_focus_codes='lowerChest', main_routine_default_allowed=1)
    result = db_queries._enrich_candidates_with_local_catalog([row])[0]
    assert 'core_chest_focus_reviewed' not in result
    assert 'core_admission_scope' not in result
    assert result.get('specialization_focus_codes') != 'lowerChest'
    assert result['main_routine_default_allowed'] == 0

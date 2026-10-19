from pathlib import Path
import sqlite3

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from engines import db_queries
from engines.candidate_ranker import score_candidate_exercises
from engines.exercise_display import candidate_display_name
from engines.schemas import PainAreaEntry

ROOT = Path(__file__).resolve().parents[1]
IDS = ('V24ROW-006682', 'V24ROW-006024', 'V24ROW-006121')


def candidates():
    with sqlite3.connect(f'file:{ROOT / "fit_core.sqlite"}?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = [dict(connection.execute('SELECT * FROM exercise_tier WHERE id=?', (key,)).fetchone()) for key in IDS]
    return db_queries._enrich_candidates_with_local_catalog(rows)


def rank(rows, **overrides):
    options = dict(available_equipment=['DUMBBELL'], blocked_equipment=[],
                   pain_areas=[], doms_db={}, readiness_level='normal',
                   goal='hypertrophy', experience_level='advanced', training_style='gym',
                   target_split_label='custom', top_n=12)
    options.update(overrides)
    return score_candidate_exercises(rows, ['hamstring', 'biceps', 'forearm'], **options)


def test_reviewed_basic_variants_enter_non_beginner_pool_without_lowering_risk():
    rows = candidates()
    assert {r['id'] for r in rank(rows)} == set(IDS)
    assert all(r['beginner_default_allowed'] == 0 for r in rows)
    assert rank(rows, experience_level='beginner') == []
    rdl = next(r for r in rows if r['id'] == IDS[0])
    assert rdl['pain_risk_level'] == 'high'
    assert rdl['hip_hinge_load'] == 'high'
    assert rdl['lumbar_load'] in ('medium', 'high')


def test_reverse_wrist_display_does_not_collapse_to_flexion():
    wrist = next(r for r in candidates() if r['id'] == IDS[2])
    assert candidate_display_name(wrist) == '덤벨 리버스 리스트 컬'
    assert wrist['primary_muscle'] == 'forearm'


def test_review_does_not_override_equipment_or_active_injury_exclusions():
    assert rank(candidates(), available_equipment=['MACHINE']) == []
    engine = create_engine(f'sqlite:///{ROOT / "fit_core.sqlite"}')
    try:
        with Session(engine) as session:
            rows = db_queries.get_candidate_exercises(
                session, ['hamstring'], [], [PainAreaEntry(body_part='hamstring')],
                available_equipment=['DUMBBELL'])
        assert IDS[0] not in {r['id'] for r in rows}
    finally:
        engine.dispose()


def test_review_manifest_integrity_and_identity_fail_closed(monkeypatch, tmp_path):
    from engines import focus_catalog_admission as policy
    manifest = tmp_path / 'tampered.json'
    manifest.write_text('{}', encoding='utf-8')
    monkeypatch.setattr(policy, 'REVIEW_PATH', manifest)
    assert all(r['main_routine_default_allowed'] == 0 for r in candidates())


def test_review_catalog_identity_mismatch_is_not_promoted():
    from engines.focus_catalog_admission import admission_patch
    source = ROOT / 'data/provenance/v25_canonical_filter_core.csv'
    assert admission_patch(IDS[0], {}, {}, source) == {}


def test_reenrichment_revokes_forged_review_fields():
    with sqlite3.connect(f'file:{ROOT / "fit_core.sqlite"}?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        row = dict(connection.execute('SELECT * FROM exercise_tier WHERE id=?', ('V24ROW-006674',)).fetchone())
    row.update(core_main_role_reviewed=1, main_routine_default_allowed=1,
               core_admission_review_version='focus-basic-admission-v1-2026-09-15')
    result = db_queries._enrich_candidates_with_local_catalog([row])[0]
    assert result['main_routine_default_allowed'] == 0
    assert 'core_main_role_reviewed' not in result

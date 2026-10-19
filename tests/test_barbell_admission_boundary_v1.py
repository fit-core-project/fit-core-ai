"""Current admission boundaries, not approval of catalog load classifications."""
import copy
import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from engines import db_queries
from engines.candidate_ranker import score_candidate_exercises
from engines.experience_policy import main_role_eligibility
from engines.registry import SPLIT_LABEL_TO_MUSCLES
from engines.schemas import PainAreaEntry

CORE = Path(__file__).resolve().parents[1]
IDS = ('V24ROW-003251', 'V24ROW-001649', 'V24ROW-000379')
LEVELS = ('beginner', 'intermediate', 'advanced')
EQUIPMENT = ['BARBELL', 'DUMBBELL', 'MACHINE', 'CABLE', 'BODYWEIGHT', 'SMITH_MACHINE']
TARGETS = SPLIT_LABEL_TO_MUSCLES['full_body']


def catalog_candidates():
    engine = create_engine('sqlite://', creator=lambda: sqlite3.connect(
        f'file:{CORE/"fit_core.sqlite"}?mode=ro', uri=True))
    try:
        with Session(engine) as session:
            rows = db_queries.get_candidate_exercises(session, TARGETS, [], [], available_equipment=EQUIPMENT)
        return {r['id']: r for r in rows if r['id'] in IDS}
    finally:
        engine.dispose()


@pytest.fixture(scope='module')
def candidates():
    rows = catalog_candidates()
    assert set(rows) == set(IDS)
    return rows


def rank(candidate, **kwargs):
    args = dict(experience_level='intermediate', goal='hypertrophy', readiness_level='normal',
        training_style='gym', target_split_label='full_body', session_structure='standard',
        available_equipment=EQUIPMENT, blocked_equipment=[], pain_areas=[], doms_db={},
        time_available_min=50, top_n=12)
    args.update(kwargs)
    return score_candidate_exercises([copy.deepcopy(candidate)], TARGETS, **args)


@pytest.mark.parametrize('eid', IDS)
@pytest.mark.parametrize('level', LEVELS)
@pytest.mark.parametrize('goal', ['hypertrophy', 'strength'])
def test_unreviewed_barbells_not_main_lifts(candidates, eid, level, goal):
    candidate = candidates[eid]
    assert not candidate.get('core_main_role_reviewed')
    assert not main_role_eligibility(candidate, level, goal).eligible


@pytest.mark.parametrize('eid', IDS)
def test_commonness_alone_cannot_unlock_main_lift(candidates, eid):
    candidate = dict(candidates[eid], mainstream_gym_score=100, exercise_familiarity_tier='common')
    assert not main_role_eligibility(candidate, 'intermediate', 'hypertrophy').eligible
    assert not main_role_eligibility(candidate, 'advanced', 'strength').eligible


@pytest.mark.parametrize('eid', IDS)
def test_other_exercise_review_cannot_be_reused(candidates, eid):
    from engines import experience_policy as policy
    candidate = dict(candidates[eid], core_main_role_reviewed=1,
        core_admission_review_version=policy._IDENTITY_REVIEWED_MAIN_VERSION,
        core_admission_source_sha256=policy._IDENTITY_REVIEWED_MAIN_SOURCE_SHA256,
        catalog_original_main_routine_default_allowed=0, main_routine_default_allowed=1)
    assert not main_role_eligibility(candidate, 'advanced', 'hypertrophy').eligible


@pytest.mark.parametrize('eid', IDS)
@pytest.mark.parametrize('restriction', ['unavailable', 'blocked', 'injury', 'doms3'])
def test_popularity_and_preference_never_override_existing_restrictions(candidates, eid, restriction):
    candidate = candidates[eid]
    args = {'preferred_exercise_ids': [eid]}
    if restriction == 'unavailable': args['available_equipment'] = ['DUMBBELL', 'MACHINE']
    elif restriction == 'blocked': args['blocked_equipment'] = ['BARBELL']
    elif restriction == 'injury': args['pain_areas'] = [PainAreaEntry(body_part=candidate['primary_muscle'])]
    else: args['doms_db'] = {candidate['primary_muscle']: 3}
    assert not rank(candidate, **args)


@pytest.mark.parametrize('eid', ['V24ROW-001649', 'V24ROW-000379'])
@pytest.mark.parametrize('level', LEVELS)
def test_hold_rows_not_auto_admitted_at_any_experience(candidates, eid, level):
    assert candidates[eid]['main_routine_default_allowed'] == 0
    assert not rank(candidates[eid], experience_level=level)


def test_squat_catalog_allow_is_distinct_from_main_role(candidates):
    squat = candidates['V24ROW-003251']
    assert squat['main_routine_default_allowed'] == 1
    assert squat['beginner_default_allowed'] == 0
    assert rank(squat)
    assert not main_role_eligibility(squat, 'intermediate', 'hypertrophy').eligible

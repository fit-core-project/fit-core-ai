import copy
import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from engines import db_queries
from engines.candidate_ranker import (
    _preserve_full_body_flat_press_equipment_choice,
    candidate_selection_family_key,
    score_candidate_exercises,
)
from engines.experience_policy import main_role_eligibility
from engines.registry import SPLIT_LABEL_TO_MUSCLES
from engines.routine_blueprint import build_blueprint_candidate_bundles, build_routine_blueprint
from engines.routine_selection import adapt_slot_selection_output
from engines.schemas import LLMRoutineSelectionOutput, PainAreaEntry

CORE = Path(__file__).resolve().parents[1]
TARGETS = SPLIT_LABEL_TO_MUSCLES['full_body']
EQUIPMENT = ['MACHINE', 'DUMBBELL', 'BARBELL', 'CABLE', 'BODYWEIGHT', 'SMITH_MACHINE']
BARBELL, DUMBBELL, INCLINE = 'V24ROW-005067', 'V24ROW-005115', 'V24ROW-005131'


@pytest.fixture(scope='module')
def candidates():
    engine = create_engine('sqlite://', creator=lambda: sqlite3.connect(
        f'file:{CORE/"fit_core.sqlite"}?mode=ro', uri=True))
    try:
        with Session(engine) as session:
            return db_queries.get_candidate_exercises(session, TARGETS, [], [], available_equipment=EQUIPMENT)
    finally:
        engine.dispose()


def rank(candidates, **overrides):
    args = dict(experience_level='intermediate', goal='hypertrophy', readiness_level='normal',
                training_style='gym', target_split_label='full_body', session_structure='standard',
                available_equipment=EQUIPMENT, blocked_equipment=[], pain_areas=[], doms_db={},
                time_available_min=50, top_n=12)
    args.update(overrides)
    return score_candidate_exercises(candidates, TARGETS, **args)


def blueprint(**overrides):
    args = dict(target_split_label='full_body', target_muscles=TARGETS, goal='hypertrophy',
                experience_level='intermediate', readiness_level='normal', time_available_min=50)
    args.update(overrides)
    return build_routine_blueprint(**args)


def press_ids(plan, pool):
    bundles = build_blueprint_candidate_bundles(plan, pool)
    return {c['exerciseId'] for b in bundles if b['slotId'] == 'full_push_main' for c in b['candidates']}


def test_catalog_alternative_reaches_pool_and_explicit_slot_without_score_inflation(candidates):
    before = copy.deepcopy(candidates)
    pool = rank(candidates)
    assert len(pool) == 12
    assert {BARBELL, DUMBBELL} <= press_ids(blueprint(), pool)
    assert INCLINE not in {c['id'] for c in pool}
    assert {c['id']: c['score'] for c in pool if c['id'] in {BARBELL, DUMBBELL}} == {BARBELL:200, DUMBBELL:208}
    assert len({c['id'] for c in pool}) == 12
    assert candidates == before


@pytest.mark.parametrize('overrides', [
    {'experience_level':'beginner'}, {'experience_level':'advanced'},
    {'goal':'strength'}, {'target_split_label':'push'}, {'readiness_level':'low'},
    {'session_structure':'superset'}, {'adjunct_options':['cooldown_stretch']},
    {'training_method':'pausedReps'}, {'preferred_exercise_ids':[INCLINE]},
])
def test_preservation_is_not_invoked_outside_bounded_scope(candidates, overrides, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('equipment reservation must not run outside its scope')
    monkeypatch.setattr('engines.candidate_ranker._preserve_full_body_flat_press_equipment_choice', unexpected)
    rank(candidates, **overrides)


@pytest.mark.parametrize('overrides', [
    {'blocked_equipment':['BARBELL']},
    {'available_equipment':['MACHINE','DUMBBELL','CABLE']},
    {'unpreferred_exercise_ids':[BARBELL]},
    {'pain_areas':[PainAreaEntry(body_part='chest')]},
])
def test_filtered_barbell_cannot_be_reintroduced(candidates, overrides):
    pool = rank(candidates, **overrides)
    assert BARBELL not in {c['id'] for c in pool}
    assert BARBELL not in press_ids(blueprint(), pool)


def fake(eid, name, equipment, **extra):
    return dict(id=eid, name_kr=name, primary_muscle='chest', movement_type='COMPOUND',
                equipment_req=equipment, score=100, main_routine_default_allowed=1,
                exercise_familiarity_tier='common', mainstream_gym_score=95,
                failure_penalty='low', injury_caution_level='medium', **extra)


def choices():
    return [fake('db', '덤벨 플랫 체스트 프레스', 'DUMBBELL'),
            fake('incline', '덤벨 인클라인 체스트 프레스', 'DUMBBELL'),
            fake('bb', '바벨 플랫 체스트 프레스', 'BARBELL')]


@pytest.mark.parametrize('field,value', [
    ('main_routine_default_allowed',0), ('main_routine_default_allowed',None),
    ('failure_penalty','high'), ('injury_caution_level','high'),
    ('primary_muscle','triceps'), ('exercise_familiarity_tier','uncommon'),
])
def test_unreviewed_or_ineligible_candidate_not_reserved(field, value):
    db, incline, bb = choices()
    bb[field] = value
    assert _preserve_full_body_flat_press_equipment_choice([db,incline], [db,incline,bb]) == [db,incline]


def test_no_other_role_is_displaced_to_make_equipment_room():
    db, _, bb = choices()
    row = dict(id='row', name_kr='덤벨 로우', primary_muscle='upper-back', score=80)
    assert _preserve_full_body_flat_press_equipment_choice([db,row], [db,bb,row]) == [db,row]


def test_reverse_equipment_choice_and_existing_pair_are_stable():
    db, incline, bb = choices()
    assert {c['id'] for c in _preserve_full_body_flat_press_equipment_choice([bb,incline], [bb,db,incline])} == {'bb','db'}
    assert _preserve_full_body_flat_press_equipment_choice([db,bb], [db,bb,incline]) == [db,bb]


@pytest.mark.parametrize('overrides', [
    {'experience_level':'beginner'}, {'experience_level':'advanced'},
    {'readiness_level':'low'}, {'session_structure':'superset'}, {'goal':'strength'},
    {'adjunct_options':['cooldown_stretch']},
])
def test_other_blueprint_modes_keep_family_compression(overrides):
    db, _, bb = choices()
    db['beginner_default_allowed'] = bb['beginner_default_allowed'] = 1
    assert len(press_ids(blueprint(**overrides), [db,bb])) <= 1


def test_same_equipment_variants_still_compress():
    db, _, bb = choices()
    other = dict(db, id='db2', name_kr='덤벨 플랫 뉴트럴 그립 체스트 프레스')
    assert press_ids(blueprint(), [db,other,bb]) == {'db','bb'}


def test_final_selection_still_rejects_flat_press_duplicate(candidates):
    plan = blueprint(time_available_min=90)
    assert plan.flex_slot_count > 0
    pool = rank(candidates)
    bundles = build_blueprint_candidate_bundles(plan, pool)
    assert {BARBELL,DUMBBELL} <= press_ids(plan, pool)
    selection = LLMRoutineSelectionOutput.model_validate({'selections':[
        {'slotId':'full_push_main','exerciseId':BARBELL},
        {'slotId':'flex','exerciseId':DUMBBELL},
    ]})
    result = adapt_slot_selection_output(selection, blueprint=plan, slot_candidate_bundles=bundles,
                                         ranked_candidates=pool)
    assert len({e.exercise_id for e in result.exercises} & {BARBELL,DUMBBELL}) == 1
    assert result._selection_diagnostics['duplicate_family_skip_count'] >= 1


def test_squat_and_unreviewed_shoulder_gate_not_relaxed(candidates):
    for eid in ('V24ROW-003251', 'V24ROW-000379'):
        candidate = next(c for c in candidates if c['id'] == eid)
        assert not main_role_eligibility(candidate, 'intermediate', 'hypertrophy').eligible
    assert candidate_selection_family_key(next(c for c in candidates if c['id'] == BARBELL)) == 'horizontal_press:flat'

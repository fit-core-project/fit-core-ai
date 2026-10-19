"""Restore an eligible alternative without admitting held rows or duplicate work."""
import copy
from dataclasses import replace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from engines import db_queries
from engines.candidate_ranker import score_candidate_exercises
from engines.routine_blueprint import build_blueprint_candidate_bundles, build_routine_blueprint
from engines.routine_selection import adapt_slot_selection_output
from engines.schemas import LLMRoutineSelectionOutput, PainAreaEntry
from tests.pending_equipment_audit_v1 import connect

BW, DB, BB = 'V24ROW-023880', 'V24ROW-023882', 'V24ROW-006742'
EQUIPMENT = ['BARBELL', 'DUMBBELL', 'MACHINE', 'CABLE', 'SMITH_MACHINE', 'BODYWEIGHT']


@pytest.fixture(scope='module')
def candidates():
    engine = create_engine('sqlite://', creator=connect)
    try:
        with Session(engine) as session:
            return db_queries.get_candidate_exercises(session, ['gluteal'], [], [], EQUIPMENT)
    finally:
        engine.dispose()


def rank(rows, level='intermediate', minutes=50, **overrides):
    args = dict(experience_level=level, goal='hypertrophy', readiness_level='normal', training_style='gym',
                target_split_label='custom', session_structure='standard', available_equipment=EQUIPMENT,
                blocked_equipment=[], pain_areas=[], doms_db={}, time_available_min=minutes, top_n=12)
    args.update(overrides)
    return score_candidate_exercises(copy.deepcopy(rows), ['gluteal'], **args)


def plan(level='intermediate', minutes=50):
    return build_routine_blueprint(target_split_label='custom', target_muscles=['gluteal'],
        goal='hypertrophy', experience_level=level, readiness_level='normal', time_available_min=minutes)


def main_ids(blueprint, pool):
    return {c['exerciseId'] for b in build_blueprint_candidate_bundles(blueprint, pool)
            if b['slotId']=='target_main' for c in b['candidates']}


@pytest.mark.parametrize('level', ['beginner', 'intermediate', 'advanced'])
@pytest.mark.parametrize('minutes', [50,60])
def test_eligible_equipment_choices_reach_one_main_slot(candidates, level, minutes):
    original = copy.deepcopy(candidates)
    pool = rank(candidates, level, minutes)
    assert {r['id'] for r in pool} == {BW, DB}
    blueprint = plan(level, minutes)
    # Advanced auto-tempo requires loadable equipment; restoration cannot bypass it.
    if level == 'advanced':
        assert blueprint.training_method == 'tempoEccentric'
        assert main_ids(blueprint, pool) == {DB}
        without_method = replace(blueprint, training_method_mode='off', training_method=None,
                                 training_method_slot_id=None)
        assert main_ids(without_method, pool) == {BW, DB}
    else:
        assert blueprint.training_method is None
        assert main_ids(blueprint, pool) == {BW, DB}
    assert candidates == original


@pytest.mark.parametrize('overrides', [
    {'available_equipment':['BODYWEIGHT']}, {'blocked_equipment':['DUMBBELL']},
    {'unpreferred_exercise_ids':[DB]},
])
def test_filtered_dumbbell_is_not_restored(candidates, overrides):
    assert main_ids(plan(), rank(candidates, **overrides)) == {BW}


@pytest.mark.parametrize('overrides', [
    {'pain_areas':[PainAreaEntry(body_part='gluteal')]}, {'doms_db':{'gluteal':3}},
])
def test_injury_and_severe_doms_filters_remain_authoritative(candidates, overrides):
    assert main_ids(plan(), rank(candidates, **overrides)) == set()


@pytest.mark.parametrize('field,value', [
    ('main_routine_default_allowed',0), ('failure_penalty','high'), ('injury_caution_level','high'),
    ('exercise_familiarity_tier','uncommon'),
])
def test_main_role_rejected_rows_are_not_choices(candidates, field, value):
    pool=rank(candidates)
    next(r for r in pool if r['id']==BW)[field]=value
    assert main_ids(plan(), pool) == {DB}


def test_held_barbell_cannot_enter_choices(candidates):
    assert BB in {r['id'] for r in candidates}
    pool=rank(candidates, preferred_exercise_ids=[BB])
    assert BB not in {r['id'] for r in pool}
    assert BB not in main_ids(plan(), pool)


def test_other_contexts_keep_their_compression(candidates):
    pool=rank(candidates)
    for blueprint in (replace(plan(), goal='strength'), replace(plan(), readiness_level='low'),
                      replace(plan(), session_structure='superset'),
                      replace(plan(), requested_adjunct_options=('cooldown_stretch',)),
                      replace(plan(), target_muscles=('gluteal','hamstring'))):
        assert len(main_ids(blueprint,pool)) <= 1


def test_duplicate_equipment_variant_still_compresses(candidates):
    pool=rank(candidates)
    duplicate=dict(next(r for r in pool if r['id']==DB),id='not-reviewed-duplicate')
    assert main_ids(plan(),[pool[0],duplicate,*pool[1:]]) == {BW,DB}


def test_isolation_choices_do_not_gain_compound_equipment_exception(candidates):
    pool=rank(candidates)
    # Canonical metadata, not the raw DB movement label, controls the slot match.
    for row in pool:
        row['movement_type']=row['canonical_movement_type']='ISOLATION'
    assert len(main_ids(plan(),pool)) == 1


@pytest.mark.parametrize('main,extra', [(BW,DB),(DB,BW)])
def test_final_routine_does_not_use_both_variants(candidates, main, extra):
    blueprint=plan(minutes=60)
    pool=rank(candidates,minutes=60)
    bundles=build_blueprint_candidate_bundles(blueprint,pool)
    selection=LLMRoutineSelectionOutput.model_validate({'selections':[
        {'slotId':'target_main','exerciseId':main},{'slotId':'flex','exerciseId':extra}]})
    result=adapt_slot_selection_output(selection,blueprint=blueprint,slot_candidate_bundles=bundles,ranked_candidates=pool)
    assert len({r.exercise_id for r in result.exercises}&{BW,DB}) == 1

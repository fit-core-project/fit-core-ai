"""A target relation is not a taxonomy rewrite or an admission approval."""
import copy
from dataclasses import replace
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from engines import db_queries
from engines.routine_blueprint import build_blueprint_candidate_bundles, build_routine_blueprint
from engines.routine_selection import adapt_slot_selection_output
from engines.schemas import LLMRoutineSelectionOutput, PainAreaEntry
from tests.test_glute_main_choices_v1 import EQUIPMENT, rank
from tests.pending_equipment_audit_v1 import connect

SUPPORT = 'V24ROW-003682'
SLOT = 'glute_abduction_support'


def blueprint(**changes):
    args = dict(target_split_label='custom', target_muscles=['gluteal'], goal='hypertrophy',
                experience_level='intermediate', readiness_level='normal', time_available_min=50)
    args.update(changes)
    return build_routine_blueprint(**args)


@pytest.fixture(scope='module')
def rows():
    engine = create_engine('sqlite://', creator=connect)
    try:
        with Session(engine) as db:
            return db_queries.get_candidate_exercises(db, ['gluteal', 'abductors'], [], [], EQUIPMENT)
    finally:
        engine.dispose()


@pytest.mark.parametrize('level', ['beginner', 'intermediate', 'advanced'])
@pytest.mark.parametrize('minutes', [50, 60])
def test_support_consumes_existing_flex_capacity_not_main_or_canonical_targets(level, minutes):
    plan = blueprint(experience_level=level, time_available_min=minutes)
    assert plan.target_muscles == ('gluteal',)
    assert plan.target_exercise_count == 6
    assert plan.minimum_main_exercise_count == 5
    assert plan.flex_slot_count == 4
    main, support = plan.slots
    assert main.slot_id == 'target_main' and main.requirement == 'required_when_safe'
    assert support.slot_id == SLOT and support.phase == 'isolation_accessory'
    assert support.target_muscles == ('abductors',) and support.max_exercises == 1


@pytest.mark.parametrize('changes', [
    {'target_muscles':['quadriceps']}, {'target_muscles':['abductors']},
    {'target_muscles':['gluteal','hamstring']}, {'target_split_label':'legs'},
    {'goal':'strength'}, {'goal':'recomposition'}, {'readiness_level':'low'},
    {'session_structure':'superset'}, {'session_structure':'circuit'},
    {'time_available_min':30}, {'time_available_min':44},
    {'adjunct_options':['cooldown_stretch']}, {'training_style':'yoga_pilates'},
    {'doms_data':{'gluteal':1}}, {'doms_data':{'abductors':3}},
    {'pain_areas':[PainAreaEntry(body_part='lower-back')]},
])
def test_out_of_scope_requests_do_not_get_support_slot_or_query(changes):
    from engines.glute_support import get_glute_support_candidates
    plan = blueprint(**changes)
    assert SLOT not in {s.slot_id for s in plan.slots}
    with patch.object(db_queries, 'get_candidate_exercises') as query:
        assert get_glute_support_candidates(None, plan, [], [], EQUIPMENT) == []
        query.assert_not_called()


def test_exact_reviewed_support_only_and_no_taxonomy_or_admission_mutation(rows):
    from engines.glute_support import get_glute_support_candidates
    original = copy.deepcopy(rows)
    with patch.object(db_queries, 'get_candidate_exercises', return_value=rows):
        support = get_glute_support_candidates(None, blueprint(), [], [], EQUIPMENT)
    assert [r['id'] for r in support] == [SUPPORT]
    assert rows == original
    assert support[0]['primary_muscle'] == support[0]['canonical_primary_muscle'] == 'abductors'
    assert support[0]['secondary_muscle'] is None
    assert support[0]['main_routine_default_allowed'] == 1


@pytest.mark.parametrize('field,value', [
    ('id','V24ROW-004706'), ('catalog_exercise_id','V24ROW-004706'),
    ('catalog_original_name_kr','different variant'), ('primary_muscle','gluteal'),
    ('canonical_primary_muscle','gluteal'), ('movement_type','COMPOUND'),
    ('equipment_req','BODYWEIGHT'), ('prescription_mode','duration'),
    ('main_routine_default_allowed',0), ('injury_caution_level','high'),
])
def test_identity_or_admission_drift_fails_closed(rows, field, value):
    from engines.glute_support import is_reviewed_glute_support
    row = copy.deepcopy(next(r for r in rows if r['id'] == SUPPORT))
    row[field] = value
    assert not is_reviewed_glute_support(row)


@pytest.mark.parametrize('level', ['beginner', 'intermediate', 'advanced'])
def test_support_is_explicit_and_not_a_main_replacement(rows, level):
    from engines.glute_support import is_reviewed_glute_support
    main_rows = [r for r in rows if r['primary_muscle']=='gluteal']
    pool = rank([*main_rows, *[r for r in rows if is_reviewed_glute_support(r)]], level)
    plan = blueprint(experience_level=level)
    bundles = build_blueprint_candidate_bundles(plan, pool)
    main_ids = {r['exerciseId'] for b in bundles if b['slotId']=='target_main' for r in b['candidates']}
    support_ids = {r['exerciseId'] for b in bundles if b['slotId']==SLOT for r in b['candidates']}
    assert SUPPORT not in main_ids and support_ids == {SUPPORT}
    selection = LLMRoutineSelectionOutput.model_validate({'selections':[
        {'slotId':'target_main','exerciseId':'V24ROW-023882'}]})
    # Omitted safe support is repaired through the existing requested-slot path.
    out = adapt_slot_selection_output(selection, blueprint=plan, slot_candidate_bundles=bundles, ranked_candidates=pool)
    assert [e.exercise_id for e in out.exercises] == ['V24ROW-023882', SUPPORT]
    assert out.exercises[1].primary_muscles == ['abductors']


@pytest.mark.parametrize('filters', [
    {'available_equipment':['DUMBBELL']}, {'blocked_equipment':['MACHINE']},
    {'pain_areas':[PainAreaEntry(body_part='abductors')]}, {'doms_db':{'abductors':3}},
    {'unpreferred_exercise_ids':[SUPPORT]},
])
def test_existing_ranker_filters_cannot_be_bypassed(rows, filters):
    support = next(r for r in rows if r['id']==SUPPORT)
    assert SUPPORT not in {r['id'] for r in rank([support], **filters)}

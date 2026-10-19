"""Independent-review boundary checks, without changing prior assertions."""
import copy

import pytest

from engines.glute_support import is_reviewed_glute_support
from engines.routine_blueprint import build_blueprint_candidate_bundles
from engines.routine_pipeline import _deterministic_or_fallback, _fallback
from engines.routine_selection import adapt_slot_selection_output
from engines.schemas import InjuryConditionEntry, LLMRoutineSelectionOutput, RoutineRequest
from tests.test_glute_main_choices_v1 import EQUIPMENT, rank
from tests.test_glute_support_relation_v1 import SLOT, SUPPORT, blueprint, rows


@pytest.mark.parametrize('mode', ['normal', 'fallback'])
def test_support_without_extension_main_cannot_become_a_complete_routine(rows, mode):
    pool = rank([r for r in rows if r['id'] == SUPPORT])
    assert [r['id'] for r in pool] == [SUPPORT]
    plan = blueprint()
    bundles = build_blueprint_candidate_bundles(plan, pool)
    for bundle in bundles:
        if bundle['slotId'] != SLOT:
            assert SUPPORT not in {r['exerciseId'] for r in bundle['candidates']}
    req = RoutineRequest(user_id='synthetic-review', target_split_label='custom',
        target_muscles=['gluteal'], goal='hypertrophy', experience_level='intermediate',
        readiness_level='normal', time_available_min=50, available_equipment=EQUIPMENT)
    args = dict(req=req, ranked_candidates=pool, max_total_sets=plan.max_total_sets,
                doms_db={}, goal='hypertrophy', recent_sets=None, profile=None)
    if mode == 'fallback':
        result = _fallback(**args, status_reason_code='networkError')
    else:
        selection = LLMRoutineSelectionOutput.model_validate({'selections':[
            {'slotId': SLOT, 'exerciseId': SUPPORT}]})
        out = adapt_slot_selection_output(selection, blueprint=plan,
            slot_candidate_bundles=bundles, ranked_candidates=pool)
        result = _deterministic_or_fallback(out, **args, db_target_muscles=['gluteal'],
            output_label='review', draft_label='review', routine_blueprint=plan)
    assert result.generation_status == 'failed'
    assert result.status_reason_code == 'emptyCandidate'
    assert result.routine_blocks == []


@pytest.mark.parametrize('field,value', [
    ('id', '123'), ('canonical_movement_type', 'COMPOUND'),
])
def test_numeric_alias_or_canonical_movement_drift_does_not_gain_role(rows, field, value):
    row = copy.deepcopy(next(r for r in rows if r['id'] == SUPPORT))
    row[field] = value
    assert not is_reviewed_glute_support(row)


@pytest.mark.parametrize('changes', [
    {'injury_conditions': [InjuryConditionEntry(area_code='hip', status='activeUncleared')]},
    {'experience_level': 'advanced', 'specialization_focus': 'gluteEmphasis'},
])
def test_injury_or_explicit_focus_does_not_gain_implicit_support(changes):
    assert SLOT not in {s.slot_id for s in blueprint(**changes).slots}


@pytest.mark.parametrize('readiness', ['normal', 'high'])
def test_exact_45_minute_boundary_is_available_without_volume_inflation(readiness):
    plan = blueprint(time_available_min=45, readiness_level=readiness)
    assert sum(s.slot_id == SLOT for s in plan.slots) == 1
    assert plan.target_muscles == ('gluteal',)
    # The existing 45-minute formula has five places; 50 minutes has six.
    assert plan.target_exercise_count == 5
    assert plan.flex_slot_count == 3

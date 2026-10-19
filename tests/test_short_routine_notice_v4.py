"""Draft v4 advice must not mutate a legacy response or widen a request."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from engines.routine_response_v4 import (
    RoutineResponseV4, build_confirmed_expansion_request, build_v4_response,
    generate_smart_routine_v4,
)
from engines.schemas import RoutineDraftResponse, RoutineRequest, UserProfileContext

BASE = Path(__file__).resolve().parents[1] / 'docs/evidence/glute-coverage-ab-v1-2026-10-10'


def request(**changes):
    data = dict(user_id='synthetic-short-notice', target_split_label='custom',
        target_muscles=['gluteal'], time_available_min=50, goal='hypertrophy',
        experience_level='intermediate', available_equipment=['DUMBBELL', 'MACHINE', 'CABLE'],
        equipment=['BARBELL'], unpreferred_exercise_ids=['V24ROW-003678'])
    data.update(changes)
    return RoutineRequest.model_validate(data)


@pytest.fixture
def routine():
    data = json.loads((BASE/'after/responses.json').read_text())['glute-intermediate-50']['response']
    return RoutineDraftResponse.model_validate(data)


@pytest.mark.parametrize('requested,actual,expected', [(50,20,True), (50,24,True),
    (50,25,False), (50,26,False), (60,29,True), (60,30,False), (180,80,True), (10,5,False)])
def test_half_requested_time_is_an_exact_not_rounded_threshold(routine, requested, actual, expected):
    routine.total_estimated_time = actual
    result = build_v4_response(request(time_available_min=requested), routine, local_constraint_profile={})
    notice = result.short_routine_notice
    assert notice.is_short_routine is expected
    assert notice.requested_time_min == requested and notice.planned_time_min == actual
    assert bool(notice.message_ko) is expected


def test_serialization_is_opt_in_and_preserves_legacy_payload(routine):
    req = request()
    before_req, before_res = req.model_dump(), routine.model_dump(mode='json', by_alias=True)
    response = build_v4_response(req, routine, local_constraint_profile={})
    payload = response.model_dump(mode='json', by_alias=True)
    assert set(payload) == {'contractVersion', 'routine', 'shortRoutineNotice'}
    assert payload['routine'] == before_res
    assert req.model_dump() == before_req and routine.model_dump(mode='json', by_alias=True) == before_res
    assert 'shortRoutineNotice' not in before_res
    assert RoutineResponseV4.model_validate_json(response.model_dump_json(by_alias=True)).model_dump() == response.model_dump()
    options = payload['shortRoutineNotice']['targetExpansionOptions']
    assert [o['code'] for o in options] == ['addQuadriceps','addHamstring','addCalves']
    assert all(o['requiresConfirmation'] and o['requiresRegeneration'] and o['availability']=='notEvaluated' for o in options)


@pytest.mark.parametrize('changes', [
    {'target_muscles':['chest']}, {'target_muscles':['gluteal','hamstring']}, {'target_split_label':'legs'},
    {'goal':'strength'}, {'readiness_level':'low'}, {'doms_data':{'hamstring':1}},
    {'pain_areas':[{'bodyPart':'knees'}]},
    {'injury_conditions':[{'areaCode':'hip','status':'activeUncleared'}]},
    {'condition_policies':[{'clearanceStatus':'unknown'}]},
    {'anthropometry_signals':{'long_femur':True}}, {'user_note':'검토 필요'},
    {'adjunct_options':['cooldown_stretch']}, {'session_structure':'superset'},
    {'experience_level':'advanced','specialization_focus':'gluteEmphasis'},
])
def test_short_notice_does_not_offer_expansion_outside_clear_scope(routine, changes):
    result = build_v4_response(request(**changes), routine, local_constraint_profile={})
    assert result.short_routine_notice.is_short_routine
    assert result.short_routine_notice.target_expansion_options == []


@pytest.mark.parametrize('local', [None, {'condition_policies':[{'clearanceStatus':'unknown'}]},
    {'mobility_limits':['limited_ankle_dorsiflexion']}, {'unexpected_context':True}])
def test_missing_or_nonempty_local_context_does_not_offer_targets(routine, local):
    assert build_v4_response(request(), routine, local_constraint_profile=local).short_routine_notice.target_expansion_options == []


def test_profile_only_injury_cannot_be_dropped(routine):
    profile = UserProfileContext(goal_type='hypertrophy', split_type='custom', pain_areas=[{'bodyPart':'knees'}])
    assert build_v4_response(request(), routine, profile=profile,
                            local_constraint_profile={}).short_routine_notice.target_expansion_options == []


@pytest.mark.parametrize('status,empty', [('failed', True), ('failed', False), ('success', True)])
def test_failed_or_empty_is_not_a_short_success(routine, status, empty):
    routine.generation_status = status
    if empty:
        routine.routine_blocks = []
        routine.total_estimated_time = 0
    notice = build_v4_response(request(), routine, local_constraint_profile={}).short_routine_notice
    assert not notice.is_short_routine and not notice.target_expansion_options


def test_fallback_gets_the_same_notice_without_status_laundering(routine):
    routine.generation_status, routine.is_fallback, routine.status_reason_code = 'fallback', True, 'networkError'
    result = build_v4_response(request(), routine, local_constraint_profile={})
    assert result.short_routine_notice.is_short_routine
    assert result.routine.generation_status == 'fallback' and result.routine.status_reason_code == 'networkError'


def test_nonempty_zero_minute_routine_is_rejected(routine):
    routine.total_estimated_time = 0
    with pytest.raises(ValueError, match='positive'):
        build_v4_response(request(), routine)


@pytest.mark.parametrize('option,target', [('addQuadriceps','quadriceps'),('addHamstring','hamstring'),('addCalves','calves')])
def test_confirmed_change_preserves_every_other_input_and_does_not_generate(routine, option, target):
    req = request()
    before = req.model_dump()
    with patch('engines.routine_pipeline.generate_smart_routine') as generate:
        changed = build_confirmed_expansion_request(req, routine, option_code=option,
                    confirmed=True, local_constraint_profile={})
        generate.assert_not_called()
    expected = dict(before, target_muscles=['gluteal', target])
    assert changed.model_dump() == expected
    assert req.model_dump() == before


@pytest.mark.parametrize('option,confirmed', [('addHamstring',False),('addHamstring',1),('anything',True)])
def test_unconfirmed_or_forged_option_is_rejected(routine, option, confirmed):
    with pytest.raises(ValueError):
        build_confirmed_expansion_request(request(), routine, option_code=option,
            confirmed=confirmed, local_constraint_profile={})


def test_rechecks_changed_restrictions_before_confirmation(routine):
    req = request(doms_data={'hamstring':3})
    with pytest.raises(ValueError, match='not available'):
        build_confirmed_expansion_request(req, routine, option_code='addHamstring',
            confirmed=True, local_constraint_profile={})


def test_v4_entrypoint_resolves_local_context_once_without_second_model_call(routine):
    with patch('engines.routine_pipeline.get_user_constraint_profile_from_sqlite', return_value={}) as resolve:
        with patch('engines.routine_pipeline.generate_smart_routine', return_value=routine) as generate:
            result = generate_smart_routine_v4(request(), None)
    resolve.assert_called_once()
    generate.assert_called_once()
    assert result.short_routine_notice.target_expansion_options == []


def test_v4_entrypoint_accepts_explicitly_resolved_empty_context(routine):
    with patch('engines.routine_pipeline.generate_smart_routine', return_value=routine) as generate:
        result = generate_smart_routine_v4(request(), None, local_constraint_profile={})
    generate.assert_called_once()
    assert len(result.short_routine_notice.target_expansion_options) == 3

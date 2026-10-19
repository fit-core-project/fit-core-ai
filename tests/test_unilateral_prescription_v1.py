"""Side arithmetic, full round trips, and unchanged bilateral prescriptions."""
import math
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from engines.unilateral_contract import UnilateralPrescription
from engines.schemas import LLMExercisePlan, LLMRoutineOutput, RoutineDraftResponse, RoutineBlock
from engines.prescription import unilateral
from engines.prescription.estimator import estimate_routine_time_min, _work_seconds_for_exercise
from engines.fallback import build_routine_draft


def units(**changes):
    return UnilateralPrescription.model_validate(dict(quantityMode='repsPerSide', secondsPerRep=3,
        sideSwitchSec=10, **changes))


def plan(**changes):
    data = dict(exercise_id='synthetic-unilateral', exercise_name='편측 검증 운동',
        movement_type='ISOLATION', primary_muscles=['gluteal'], target_reps=12, sets=3,
        rest_time_sec=60, exercise_rationale='단위 검증')
    data.update(changes)
    return LLMExercisePlan.model_validate(data)


def draft(p):
    output = LLMRoutineOutput(total_estimated_time=0, summary_title='검증',
        rationale_summary=[], exercises=[p])
    return build_routine_draft(output, 'success', 'none', False, preserve_input_order=True)


@pytest.mark.parametrize('sets,reps,sec,switch', [(1,1,1,0),(2,10,2,5),(3,12,3,10),(5,15,4,20)])
def test_both_sides_are_one_set_with_one_switch_and_between_set_rest(sets,reps,sec,switch):
    p = plan(sets=sets,target_reps=reps)
    p._unilateral_prescription = UnilateralPrescription(quantity_mode='repsPerSide',
        seconds_per_rep=sec, side_switch_sec=switch)
    work = 2*reps*sec+switch
    assert _work_seconds_for_exercise(p) == work
    assert estimate_routine_time_min([p]) == math.ceil((300+sets*work+(sets-1)*60)/60)
    response = draft(p)
    assert len(response.routine_blocks[0].prescription) == sets
    assert all(s.target_reps == reps for s in response.routine_blocks[0].prescription)
    assert response.total_estimated_time == estimate_routine_time_min([p])


def test_no_active_registry_or_name_inference_or_llm_injection():
    assert not unilateral.REVIEWED_UNILATERAL_PRESCRIPTIONS
    p = plan(exercise_name='싱글레그 케이블 킥백', unilateralPrescription=units().model_dump())
    assert p._unilateral_prescription is None
    row = dict(id=p.exercise_id,catalog_exercise_id=p.exercise_id,unilateral_prescription=units())
    unilateral.apply_reviewed_unilateral_metadata(p,row)
    assert p._unilateral_prescription is None
    assert _work_seconds_for_exercise(p) == 40
    assert 'unilateralPrescription' not in draft(p).model_dump(by_alias=True)['routineBlocks'][0]


def test_candidate_plan_uses_only_server_exact_review_and_numeric_alias_is_not_inherited():
    from engines.routine_selection import _candidate_plan
    row = dict(id='synthetic-unilateral', catalog_exercise_id='synthetic-unilateral',
               name_kr='편측 검증', primary_muscle='gluteal', movement_type='ISOLATION', equipment_req='CABLE')
    with patch.object(unilateral,'REVIEWED_UNILATERAL_PRESCRIPTIONS', {row['id']:units()}):
        p = _candidate_plan(row,'검증',goal='hypertrophy')
        assert p._unilateral_prescription == units()
        alias = _candidate_plan(dict(row,id='125'),'검증',goal='hypertrophy')
        assert alias._unilateral_prescription is None


@pytest.mark.parametrize('method,extra', [({'pauseSec':2},2),({'eccentricSec':4},2),({},0)])
def test_repetition_method_increment_applies_to_each_side_once(method,extra):
    p = plan()
    p._unilateral_prescription = units()
    p._training_method_instruction = method
    assert _work_seconds_for_exercise(p) == 2*12*(3+extra)+10


def test_timed_per_side_is_not_interpreted_as_reps():
    p = plan(target_reps=1,sets=2)
    p._prescription_set_type = 'timed'
    p._target_duration_sec = 30
    p._unilateral_prescription = UnilateralPrescription(quantity_mode='durationPerSide',side_switch_sec=5)
    assert _work_seconds_for_exercise(p) == 65
    response = draft(p)
    assert len(response.routine_blocks[0].prescription) == 2
    assert response.routine_blocks[0].prescription[0].target_duration_sec == 30
    assert response.total_estimated_time == 9


@pytest.mark.parametrize('bad', [
    {'quantityMode':'repsPerSide','sideSwitchSec':10},
    {'quantityMode':'durationPerSide','sideSwitchSec':10,'secondsPerRep':3},
    {'quantityMode':'repsPerSide','sideSwitchSec':True,'secondsPerRep':3},
    {'quantityMode':'repsPerSide','sideSwitchSec':-1,'secondsPerRep':3},
    {'quantityMode':'repsPerSide','sideSwitchSec':10,'secondsPerRep':3,'sideCount':1},
    {'quantityMode':'repsPerSide','sideSwitchSec':10,'secondsPerRep':3,'restPlacement':'betweenSides'},
])
def test_unsupported_units_fail_closed(bad):
    with pytest.raises(ValidationError):
        UnilateralPrescription.model_validate(bad)


@pytest.mark.parametrize('mode',['timed','distance','interval'])
def test_repetition_units_cannot_silently_double_other_modalities(mode):
    p=plan()
    p._unilateral_prescription=units()
    p._prescription_set_type=mode
    with pytest.raises(ValueError,match='working reps'):
        estimate_routine_time_min([p])


def test_json_roundtrip_and_all_reestimation_paths_preserve_nonuniform_sets():
    from engines.routine_pipeline import _estimate_routine_blocks_time
    from engines.routine_generation_choice import _estimate_choice_routine_blocks_time
    p=plan()
    p._unilateral_prescription=units()
    response=draft(p)
    block=response.routine_blocks[0]
    block.prescription[1].target_reps=10
    block.prescription[1].target_rest_sec=90
    restored=RoutineDraftResponse.model_validate_json(response.model_dump_json(by_alias=True))
    assert restored.routine_blocks[0].unilateral_prescription == units()
    # 12+10+12 each side, three switches, only the first two rests.
    expected=math.ceil((300+2*(12+10+12)*3+30+60+90)/60)
    assert _estimate_routine_blocks_time(restored.routine_blocks)==expected
    assert _estimate_choice_routine_blocks_time(restored.routine_blocks)==expected
    rebuilt=unilateral.restore_unilateral_from_block(plan(),restored.routine_blocks[0])
    emitted=draft(rebuilt)
    assert [s.target_reps for s in emitted.routine_blocks[0].prescription]==[12,10,12]
    assert emitted.total_estimated_time==expected


def test_superset_round_finishes_both_sides_before_group_transition_and_rest():
    first,second=plan(sets=2),plan(exercise_id='bilateral',sets=2)
    first._unilateral_prescription=units()
    metadata={p.exercise_id:dict(executionGroupId='pair',executionGroupType='superset',
        transitionRestSec=15,groupRestSec=90) for p in (first,second)}
    assert estimate_routine_time_min([first,second],metadata)==math.ceil((300+2*(82+15+40)+90)/60)


def test_replacement_does_not_inherit_previous_exercises_side_units():
    from engines.llm_parser import _candidate_to_plan
    p=plan()
    p._unilateral_prescription=units()
    replaced=_candidate_to_plan(dict(id='other',catalog_exercise_id='other',name_kr='양측 운동'),p)
    assert replaced._unilateral_prescription is None
    assert p._unilateral_prescription==units()


def test_empty_sets_are_zero_not_one_bilateral_pair():
    from engines.routine_pipeline import _estimate_routine_blocks_time
    p=plan(sets=0)
    p._unilateral_prescription=units()
    assert estimate_routine_time_min([p])==0
    assert _estimate_routine_blocks_time(draft(p).routine_blocks)==0


def test_serialized_mode_mismatch_is_rejected():
    data=draft(plan()).routine_blocks[0].model_dump(by_alias=True)
    data['unilateralPrescription']={'quantityMode':'durationPerSide','sideSwitchSec':5}
    with pytest.raises(ValidationError):
        RoutineBlock.model_validate(data)

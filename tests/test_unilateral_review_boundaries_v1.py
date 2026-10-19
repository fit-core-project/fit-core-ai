"""Independent review witnesses: keep set rests and rebind modality first."""
import math
from unittest.mock import patch

from engines.llm_parser import _candidate_to_plan
from engines.prescription import unilateral
from engines.prescription.estimator import _work_seconds_for_exercise
from engines.routine_blueprint import build_routine_blueprint
from engines.routine_selection import apply_blueprint_execution_policy_to_draft
from engines.unilateral_contract import UnilateralPrescription
from tests.test_unilateral_prescription_v1 import plan, units, draft


def test_standard_execution_policy_preserves_each_unilateral_set_rest():
    p=plan()
    p._unilateral_prescription=units()
    response=draft(p)
    response.routine_blocks[0].prescription[1].target_rest_sec=90
    blueprint=build_routine_blueprint(target_split_label='custom',target_muscles=['gluteal'],
        goal='hypertrophy',experience_level='intermediate',readiness_level='normal',time_available_min=50)
    result=apply_blueprint_execution_policy_to_draft(response,blueprint=blueprint)
    assert [s.target_rest_sec for s in result.routine_blocks[0].prescription]==[60,90,60]
    assert result.total_estimated_time==math.ceil((300+3*82+60+90)/60)


def test_working_template_to_reviewed_timed_candidate_applies_modality_before_units():
    row=dict(id='synthetic-timed',catalog_exercise_id='synthetic-timed',name_kr='편측 시간 검증',
             prescription_mode='duration',default_duration_sec=25,canonical_movement_type='STATIC')
    reviewed=UnilateralPrescription(quantity_mode='durationPerSide',side_switch_sec=5)
    with patch.object(unilateral,'REVIEWED_UNILATERAL_PRESCRIPTIONS',{row['id']:reviewed}):
        p=_candidate_to_plan(row,plan())
    assert p._prescription_set_type=='timed' and p._target_duration_sec==25
    assert _work_seconds_for_exercise(p)==55


def test_unilateral_timed_template_to_working_candidate_drops_all_old_units():
    old=plan(target_reps=1,target_rir=None)
    old._prescription_set_type='timed'
    old._target_duration_sec=30
    old._unilateral_prescription=UnilateralPrescription(quantity_mode='durationPerSide',side_switch_sec=5)
    p=_candidate_to_plan(dict(id='bilateral',catalog_exercise_id='bilateral',name_kr='반복 검증',
        prescription_mode='reps',canonical_movement_type='ISOLATION'),old)
    assert p._unilateral_prescription is None and p._target_duration_sec is None
    assert p._prescription_set_type=='working' and p.target_reps==10 and p.target_rir==2
    assert _work_seconds_for_exercise(p)==40
    assert old._prescription_set_type=='timed' and old._target_duration_sec==30

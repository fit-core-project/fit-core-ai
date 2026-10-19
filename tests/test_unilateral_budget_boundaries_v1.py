"""Budget and failed-group boundaries for server-owned unilateral units."""
import pytest

from engines.schemas import LLMRoutineOutput, RoutineRequest
from engines.llm_parser import trim_routine_to_time_budget
from engines.prescription.estimator import estimate_routine_time_min
from engines.routine_response_v4 import build_v4_response
from tests.test_unilateral_prescription_v1 import plan, units, draft


def test_time_budget_checks_both_sides_before_accepting_set_count():
    p=plan()
    p._unilateral_prescription=units()
    output=LLMRoutineOutput(total_estimated_time=0,summary_title='검증',rationale_summary=[],exercises=[p])
    assert estimate_routine_time_min(output.exercises)==12
    assert trim_routine_to_time_budget(output,10)
    assert p.sets==2
    assert estimate_routine_time_min(output.exercises)==9
    result=draft(p)
    assert result.total_estimated_time==9
    assert len(result.routine_blocks[0].prescription)==2


def test_stale_per_set_data_cannot_silently_fill_new_sets():
    from engines.prescription.unilateral import restore_unilateral_from_block
    p=plan()
    p._unilateral_prescription=units()
    restored=restore_unilateral_from_block(plan(),draft(p).routine_blocks[0])
    restored.sets+=1
    with pytest.raises(ValueError,match='set count'):
        estimate_routine_time_min([restored])


def test_v4_envelope_keeps_explicit_units_without_doubling_prescription_rows():
    p=plan()
    p._unilateral_prescription=units()
    req=RoutineRequest(user_id='synthetic',target_split_label='custom',target_muscles=['gluteal'],
        time_available_min=50,goal='hypertrophy')
    response=build_v4_response(req,draft(p),local_constraint_profile={})
    raw=response.model_dump(mode='json',by_alias=True)
    assert raw['contractVersion']=='routine-core-v4-draft-v3'
    assert raw['routine']['totalEstimatedTime']==12
    block=raw['routine']['routineBlocks'][0]
    assert len(block['prescription'])==3
    assert block['unilateralPrescription']['setUnit']=='bothSides'
    assert block['prescription'][0]['targetReps']==12

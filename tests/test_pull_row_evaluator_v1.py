from copy import deepcopy

import pytest

from engines import db_queries
from engines.pull_composition import BENT_OVER_ID, SUPPORTED_ID
from engines.routine_blueprint import build_routine_blueprint
from engines.schemas import RoutineDraftResponse
from scripts import run_gemma4_quality_seed_live as live


def seed(**delta):
    return {'scenarioId': 'synthetic-pull-evaluator', 'input': {
        'targetSplitLabel': 'pull', 'timeAvailableMin': 60, 'goal': 'hypertrophy',
        'experienceLevel': 'advanced', 'readinessLevel': 'normal', 'methodMode': 'off', **delta}}


def response():
    return RoutineDraftResponse.model_validate({
        'generationStatus': 'success', 'statusReasonCode': 'none', 'isFallback': False,
        'totalEstimatedTime': 49, 'summaryTitle': '합성 검증', 'rationaleSummary': [],
        'routineBlocks': [dict(order=i, exerciseId=eid, exerciseName=name, equipmentType=equipment,
            primaryMuscles=['upper-back'], movementType='COMPOUND', defaultRestSec=120,
            prescription=[dict(setIndex=1, setType='working', targetReps=8, targetRestSec=120)],
            exerciseRationale='합성 검증', reasons=[], boosts=[], penalties=[])
            for i, (eid, name, equipment) in enumerate([
                (BENT_OVER_ID, '덤벨 로우', 'DUMBBELL'),
                (SUPPORTED_ID, '머신 체스트 서포티드 로우', 'MACHINE')], 1)],
        'warnings': [],
    })


def reviewed(row, result):
    req = live.request_from_seed(row)
    bp = build_routine_blueprint(target_split_label=req.target_split_label,
        target_muscles=['upper-back', 'biceps'], goal=req.goal, experience_level=req.experience_level,
        readiness_level=req.readiness_level, time_available_min=req.time_available_min,
        method_mode=req.method_mode, training_style=req.training_style,
        session_structure=req.session_structure, adjunct_options=req.adjunct_options,
        pain_areas=req.pain_areas, doms_data=req.doms_data, injury_conditions=req.injury_conditions)
    return live._reviewed_pull_output_candidates(req, result, bp)


def test_evaluator_uses_real_source_review_without_changing_public_response():
    result = response()
    before = result.model_dump()
    assert set(reviewed(seed(), result)) == {BENT_OVER_ID, SUPPORTED_ID}
    assert 'duplicate_movement_family' not in live.derived_output_quality_flags(seed(), result)
    assert result.model_dump() == before


@pytest.mark.parametrize('delta', [
    {'experienceLevel': 'intermediate'}, {'timeAvailableMin': 50}, {'readinessLevel': 'low'},
    {'goal': 'strength'}, {'methodMode': 'auto'}, {'adjunctOptions': ['cooldown_stretch']},
    {'mobilityLimits': ['limited_shoulder_flexion']},
    {'currentPainAreas': [{'bodyPart': 'upper-back'}]},
    {'currentDoms': [{'bodyPart': 'upper-back', 'level': 1}]},
    {'unavailableEquipment': ['MACHINE']}, {'availableEquipment': ['MACHINE']},
])
def test_out_of_scope_pair_is_not_exempted(delta):
    result = response()
    assert reviewed(seed(**delta), result) == {}
    assert 'duplicate_movement_family' in live.derived_output_quality_flags(seed(**delta), result)


@pytest.mark.parametrize('field,value', [
    ('exercise_id', 'unreviewed-row'), ('exercise_name', '일반 머신 로우'),
    ('equipment_type', 'BARBELL'), ('movement_type', 'ISOLATION'), ('primary_muscles', ['trapezius']),
])
def test_output_identity_drift_revokes_evaluator_exception(field, value):
    result = response()
    setattr(result.routine_blocks[1], field, value)
    assert reviewed(seed(), result) == {}
    assert 'duplicate_movement_family' in live.derived_output_quality_flags(seed(), result)


def test_third_row_still_fails_duplicate_evaluation():
    result = response()
    result.routine_blocks.append(deepcopy(result.routine_blocks[0]))
    assert reviewed(seed(), result) == {}
    assert 'duplicate_movement_family' in live.derived_output_quality_flags(seed(), result)


@pytest.mark.parametrize('missing', ['catalog', 'provenance'])
def test_missing_catalog_or_verified_source_keeps_duplicate_flag(monkeypatch, missing):
    if missing == 'catalog':
        monkeypatch.setattr(live, '_exercise_policy_candidate', lambda _: None)
    else:
        monkeypatch.setattr(db_queries, '_load_source_equipment_by_id', lambda: None)
    assert reviewed(seed(), response()) == {}
    assert 'duplicate_movement_family' in live.derived_output_quality_flags(seed(), response())

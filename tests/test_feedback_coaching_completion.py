import pytest

from engines.routine_pipeline import (
    _apply_active_context_rationale,
    _apply_selected_mobility_demand_warnings,
    _apply_shoulder_condition_warnings,
)
from engines.schemas import RoutineDraftResponse, RoutineRequest
from engines.routine_generation_choice import preflight_routine_generation


def request(status="managedWithRestrictions", *, legacy=False):
    condition = {"status": status}
    if status == "managedWithRestrictions":
        condition["restrictedMovementPatterns"] = ["overheadPosition"]
    payload = {"targetSplitLabel": "push", "timeAvailableMin": 50}
    if legacy:
        payload.update(shoulderConditionStatus=status,
                       restrictedMovementPatterns=condition.get("restrictedMovementPatterns", []))
    else:
        payload["injuryConditions"] = [{"areaCode": "shoulder", **condition}]
    return RoutineRequest.model_validate(payload)


def response(*, fallback=False, lower_only=False):
    names = [("leg", "머신 레그 컬", "hamstring")] if lower_only else [
        ("press", "덤벨 인클라인 체스트 프레스", "chest"),
        ("fly", "머신 펙덱 플라이", "chest"),
        ("lateral", "머신 레터럴 레이즈", "side-deltoids"),
    ]
    return RoutineDraftResponse.model_validate({
        "generationStatus": "fallback" if fallback else "success",
        "statusReasonCode": "networkError" if fallback else "none",
        "isFallback": fallback, "totalEstimatedTime": 43,
        "summaryTitle": "루틴", "rationaleSummary": [],
        "routineBlocks": [{
            "order": i, "exerciseId": eid, "exerciseName": name,
            "primaryMuscles": [primary], "defaultRestSec": 90,
            "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
            "exerciseRationale": "선택한 목표에 맞춘 운동입니다.",
        } for i, (eid, name, primary) in enumerate(names, 1)],
    })


CANDIDATES = [
    {"id": "press", "name_kr": "덤벨 인클라인 체스트 프레스", "primary_muscle": "chest", "movement_type": "COMPOUND"},
    {"id": "fly", "name_kr": "머신 펙덱 플라이", "primary_muscle": "chest", "movement_type": "ISOLATION"},
    {"id": "lateral", "name_kr": "머신 레터럴 레이즈", "primary_muscle": "side-deltoids", "movement_type": "ISOLATION"},
    {"id": "leg", "name_kr": "머신 레그 컬", "primary_muscle": "hamstring", "movement_type": "ISOLATION"},
    {"id": "unselected", "name_kr": "선택되지 않은 프레스", "primary_muscle": "chest", "movement_type": "COMPOUND"},
]


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_managed_shoulder_coaching_attaches_to_selected_exercises_without_changing_dose(fallback, legacy):
    req = request(legacy=legacy)
    draft = response(fallback=fallback)
    before = [(b.exercise_id, b.prescription, b.default_rest_sec) for b in draft.routine_blocks]
    _apply_shoulder_condition_warnings(draft, req)
    _apply_selected_mobility_demand_warnings(draft, req, ranked_candidates=CANDIDATES)
    for block in draft.routine_blocks:
        assert "허용된 가동범위" in block.exercise_rationale
        assert "통증" in block.exercise_rationale and "중단" in block.exercise_rationale
        assert "의료 전문가" in block.exercise_rationale
    text = " ".join(draft.warnings)
    assert "운동 진행 선택은 의료진 허가를 대신하지 않습니다" in text
    assert "선택되지 않은 프레스" not in text
    assert "통증이 없는 운동입니다" not in text
    assert [(b.exercise_id, b.prescription, b.default_rest_sec) for b in draft.routine_blocks] == before
    assert draft.total_estimated_time == 43
    saved = draft.model_dump()
    _apply_shoulder_condition_warnings(draft, req)
    _apply_selected_mobility_demand_warnings(draft, req, ranked_candidates=CANDIDATES)
    assert draft.model_dump() == saved


@pytest.mark.parametrize("status", ["recovered", "managedNoRestrictions"])
def test_other_shoulder_states_are_not_relabelled_as_restricted(status):
    draft = response()
    original = draft.model_dump()
    _apply_shoulder_condition_warnings(draft, request(status))
    _apply_selected_mobility_demand_warnings(draft, request(status), ranked_candidates=CANDIDATES)
    assert draft.model_dump() == original


def test_managed_shoulder_does_not_attach_shoulder_exercise_instruction_to_leg_curl():
    draft = response(lower_only=True)
    _apply_selected_mobility_demand_warnings(draft, request(), ranked_candidates=CANDIDATES)
    assert "어깨" not in draft.routine_blocks[0].exercise_rationale
    assert not draft.warnings


def test_doms_or_mobility_alone_is_not_managed_injury():
    draft = response()
    req = RoutineRequest(target_split_label="push", time_available_min=50,
                         doms_data={"chest": 1}, profile_signals=["overhead_reach_difficult"])
    _apply_selected_mobility_demand_warnings(draft, req, ranked_candidates=CANDIDATES)
    assert all("관리 중" not in b.exercise_rationale for b in draft.routine_blocks)


def test_empty_failed_response_gets_no_exercise_continuation_instruction():
    draft = response()
    draft.routine_blocks = []
    draft.generation_status = "failed"
    draft.status_reason_code = "emptyCandidate"
    draft.total_estimated_time = 0
    _apply_selected_mobility_demand_warnings(draft, request(), ranked_candidates=CANDIDATES)
    assert not draft.warnings and not draft.routine_blocks


def test_managed_preflight_explains_pause_before_generation_choice():
    result = preflight_routine_generation(request())
    assert result.choice_required
    assert "보류" in result.warning
    assert "의료진 허가" in result.warning
    assert "requestReevaluation" in result.available_choices
    assert "cancel" in result.available_choices
    assert "normalVolume" in result.available_choices


@pytest.mark.parametrize('fallback', [False, True])
def test_low_readiness_explanation_does_not_claim_exercise_count_was_reduced(fallback):
    draft = response(fallback=fallback)
    req = RoutineRequest(target_split_label='push', time_available_min=60, readiness_level='low')
    blocks = [b.model_dump() for b in draft.routine_blocks]
    _apply_active_context_rationale(draft, req)
    assert '오늘 컨디션이 낮아' in ' '.join(draft.rationale_summary)
    assert '운동 수' not in ' '.join(draft.rationale_summary)
    assert [b.model_dump() for b in draft.routine_blocks] == blocks
    assert draft.total_estimated_time == 43
    saved = draft.model_dump()
    _apply_active_context_rationale(draft, req)
    assert draft.model_dump() == saved

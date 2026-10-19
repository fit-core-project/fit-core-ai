"""
generate_smart_routine 통합 테스트
LLM과 DB를 mock해서 AI 파이프라인 경로를 검증한다.

[Mock 전략]
LangChain의 `prompt | structured_llm` 체인은 `|` 연산자로 RunnableSequence를 만든다.
MagicMock은 Runnable이 아니므로 RunnableLambda로 감싸져 .invoke() 대신 직접 호출된다.
→ with_structured_output 반환값을 RunnableLambda로 만들어야 체인 invoke가 올바로 동작한다.
"""
import asyncio
from unittest.mock import MagicMock, patch

import httpx
import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda
from sqlalchemy.orm import Session

from engines.db_queries import get_user_profile_context
from engines.routine_pipeline import (
    _add_underfilled_safe_candidates,
    _apply_adjunct_application_warnings,
    _apply_active_context_rationale,
    _apply_selected_mobility_demand_warnings,
    _apply_pain_and_doms_warnings,
    _canonicalize_llm_exercise_names,
    _deterministic_or_fallback,
    _enforce_same_area_injury_precedence_wording,
    _enforce_response_fact_consistency,
    _enforce_experience_rationale_consistency,
    _ensure_requested_adjunct_style_candidate,
    _fallback,
    has_target_relevant_recovery_constraint,
    _merge_candidate_sources,
    generate_smart_routine,
)
from engines.routine_blueprint import build_routine_blueprint
from engines.prescription.adjustments import _apply_readiness_to_exercise
from engines.schemas import (
    LLMExercisePlan,
    LLMRoutineOutput,
    PainAreaEntry,
    RecentSetRecord,
    RoutineDraftResponse,
    RoutineBlock,
    RoutineRequest,
    SetPrescription,
)


def test_adjunct_decisions_record_applied_and_suppressed_without_silent_drop():
    blueprint = build_routine_blueprint(
        target_split_label="upper",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=60,
        adjunct_options=["cooldown_stretch", "mobility_warmup"],
    )
    candidates = [
        {
            "id": "mobility",
            "name_kr": "동적 어깨 모빌리티",
            "primary_muscle": "front-deltoids",
            "movement_type": "STATIC",
            "adjunct_option_codes": ["mobility_warmup"],
            "adjunct_catalog_review_status": "core-reviewed",
        },
        {
            "id": "cooldown",
            "name_kr": "상체 쿨다운 스트레치",
            "primary_muscle": "chest",
            "movement_type": "STATIC",
            "adjunct_option_codes": ["cooldown_stretch"],
            "adjunct_catalog_review_status": "core-reviewed",
        },
    ]
    response = RoutineDraftResponse(
        generation_status="success",
        status_reason_code="none",
        is_fallback=False,
        summary_title="테스트",
        rationale_summary=[],
        routine_blocks=[
            RoutineBlock(
                order=1,
                exercise_id="mobility",
                exercise_name="동적 어깨 모빌리티",
                movement_type="STATIC",
                primary_muscles=["front-deltoids"],
                default_rest_sec=15,
                prescription=[
                    SetPrescription(
                        set_index=index,
                        set_type="mobility",
                        target_reps=8,
                        target_rir=None,
                        target_rest_sec=15,
                    )
                    for index in (1, 2)
                ],
                exercise_rationale="준비 운동",
                session_phase="warmup_or_preparation",
                adjunct_option="mobility_warmup",
            )
        ],
    )
    response._adjunct_suppression_reason_by_option = {
        "cooldown_stretch": "insufficientTimeAfterMainMinimum"
    }

    guarded = _apply_adjunct_application_warnings(response, blueprint, candidates)

    assert [decision.option for decision in guarded.adjunct_decisions] == [
        "mobility_warmup",
        "cooldown_stretch",
    ]
    assert guarded.adjunct_decisions[0].status == "applied"
    assert guarded.adjunct_decisions[0].exercise_id == "mobility"
    assert guarded.adjunct_decisions[1].status == "suppressed"
    assert guarded.adjunct_decisions[1].reason_code == "insufficientTimeAfterMainMinimum"
    assert any("쿨다운" in warning and "반영하지 못했습니다" in warning for warning in guarded.warnings)


def test_adjunct_ledger_is_rebuilt_after_injury_guard_removes_applied_block():
    blueprint = build_routine_blueprint(
        target_split_label="upper",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=60,
        adjunct_options=["mobility_warmup"],
    )
    candidates = [{
        "id": "mobility",
        "name_kr": "동적 어깨 모빌리티",
        "primary_muscle": "front-deltoids",
        "movement_type": "STATIC",
        "pain_triggers": "front-deltoids",
        "adjunct_option_codes": ["mobility_warmup"],
        "adjunct_catalog_review_status": "core-reviewed",
    }]
    response = RoutineDraftResponse(
        generation_status="success",
        status_reason_code="none",
        is_fallback=False,
        summary_title="테스트",
        rationale_summary=[],
        routine_blocks=[
            RoutineBlock(
                order=1,
                exercise_id="mobility",
                exercise_name="동적 어깨 모빌리티",
                movement_type="STATIC",
                primary_muscles=["front-deltoids"],
                default_rest_sec=15,
                prescription=[
                    SetPrescription(
                        set_index=index,
                        set_type="mobility",
                        target_reps=8,
                        target_rir=None,
                        target_rest_sec=15,
                    )
                    for index in (1, 2)
                ],
                exercise_rationale="준비 운동",
                session_phase="warmup_or_preparation",
                adjunct_option="mobility_warmup",
            )
        ],
    )
    request = RoutineRequest(
        target_split_label="upper",
        time_available_min=60,
        pain_areas=[PainAreaEntry(body_part="front-deltoids")],
        adjunct_options=["mobility_warmup"],
    )

    response = _apply_adjunct_application_warnings(response, blueprint, candidates)
    assert response.adjunct_decisions[0].status == "applied"

    response = _apply_pain_and_doms_warnings(response, request, candidates)
    response = _apply_adjunct_application_warnings(response, blueprint, candidates)

    assert response.routine_blocks == []
    assert response.adjunct_decisions[0].status == "suppressed"
    assert response.adjunct_decisions[0].reason_code == "postValidationRemoved"
    assert response.adjunct_decisions[0].exercise_id is None


def test_reviewed_adjunct_source_overrides_duplicate_catalog_display_name():
    merged = _merge_candidate_sources(
        [{
            "id": "dynamic_pec_stretch",
            "name_kr": "문틀 흉근 스트레치 - 동적 반복",
            "user_facing_display_name_kr": "문틀 흉근 스트레치 - 기본",
        }],
        [{
            "id": "dynamic_pec_stretch",
            "name_kr": "문틀 흉근 동적 스트레치",
            "user_facing_display_name_kr": "문틀 흉근 동적 스트레치",
            "adjunct_display_name_ko": "문틀 흉근 동적 스트레치",
            "adjunct_option_codes": ["mobility_warmup"],
        }],
    )

    assert merged[0]["name_kr"] == "문틀 흉근 동적 스트레치"
    assert merged[0]["display_name_ko"] == "문틀 흉근 동적 스트레치"
    assert merged[0]["user_facing_display_name_kr"] == "문틀 흉근 동적 스트레치"


def test_recorded_injury_target_scope_change_is_explained_to_user():
    request = RoutineRequest(
        target_split_label="push",
        time_available_min=45,
        pain_areas=[PainAreaEntry(body_part="chest")],
    )
    response = RoutineDraftResponse(
        generation_status="success",
        status_reason_code="none",
        is_fallback=False,
        summary_title="조정된 루틴",
        rationale_summary=[],
        routine_blocks=[],
        warnings=[],
    )

    guarded = _apply_pain_and_doms_warnings(
        response,
        request,
        [],
        ["chest", "front-deltoids", "triceps"],
    )

    assert any("가슴" in warning and "후보에서 제외" in warning for warning in guarded.warnings)


def test_recorded_injury_warning_uses_reported_area_not_expanded_aliases():
    request = RoutineRequest(
        target_split_label="push",
        time_available_min=45,
        pain_areas=[PainAreaEntry(body_part="front-deltoids")],
    )
    response = RoutineDraftResponse(
        generation_status="success",
        status_reason_code="none",
        is_fallback=False,
        summary_title="조정된 루틴",
        rationale_summary=[],
        routine_blocks=[],
        warnings=[],
    )

    guarded = _apply_pain_and_doms_warnings(
        response,
        request,
        [],
        ["chest", "front-deltoids", "triceps"],
    )
    warning = next(item for item in guarded.warnings if "후보에서 제외" in item)

    assert "전면 어깨" in warning
    assert "후면 어깨" not in warning
    assert "측면 어깨" not in warning


def test_registered_injury_overlap_is_removed_from_final_response_and_time_recomputed():
    request = RoutineRequest(
        target_split_label="push",
        time_available_min=45,
        pain_areas=[PainAreaEntry(body_part="front-deltoids")],
    )
    response = RoutineDraftResponse(
        generation_status="success",
        status_reason_code="none",
        is_fallback=False,
        total_estimated_time=30,
        summary_title="푸시 루틴",
        rationale_summary=[],
        routine_blocks=[
            {
                "order": 1,
                "exerciseId": "bench",
                "exerciseName": "벤치프레스",
                "movementType": "COMPOUND",
                "primaryMuscles": ["chest"],
                "defaultRestSec": 90,
                "prescription": [
                    {"setIndex": 1, "targetReps": 8, "targetRestSec": 90},
                    {"setIndex": 2, "targetReps": 8, "targetRestSec": 90},
                ],
                "exerciseRationale": "가슴 메인 운동",
                "executionGroupId": "G1",
                "executionGroupType": "superset",
                "executionGroupPosition": 1,
                "executionGroupSize": 2,
                "transitionRestSec": 15,
                "groupRestSec": 90,
            },
            {
                "order": 2,
                "exerciseId": "pushdown",
                "exerciseName": "케이블 푸시다운",
                "movementType": "ISOLATION",
                "primaryMuscles": ["triceps"],
                "defaultRestSec": 60,
                "prescription": [
                    {"setIndex": 1, "targetReps": 12, "targetRestSec": 60},
                    {"setIndex": 2, "targetReps": 12, "targetRestSec": 60},
                ],
                "exerciseRationale": "삼두 보조 운동",
                "executionGroupId": "G1",
                "executionGroupType": "superset",
                "executionGroupPosition": 2,
                "executionGroupSize": 2,
                "transitionRestSec": 15,
                "groupRestSec": 90,
            },
        ],
        warnings=[],
    )
    candidates = [
        {
            "id": "bench",
            "primary_muscle": "chest",
            "secondary_muscle": "triceps, front-deltoids",
            "movement_type": "COMPOUND",
        },
        {
            "id": "pushdown",
            "primary_muscle": "triceps",
            "secondary_muscle": "",
            "movement_type": "ISOLATION",
        },
    ]

    guarded = _apply_pain_and_doms_warnings(response, request, candidates)

    assert [block.exercise_id for block in guarded.routine_blocks] == ["pushdown"]
    assert guarded.routine_blocks[0].order == 1
    assert guarded.routine_blocks[0].execution_group_id is None
    assert guarded.routine_blocks[0].execution_group_type == "single"
    assert guarded.routine_blocks[0].execution_group_size == 1
    assert 0 < guarded.total_estimated_time < response.total_estimated_time
    assert any("벤치프레스" in warning and "안전 규칙에 따라 제외" in warning for warning in guarded.warnings)


def test_soleus_doms_alias_warns_for_calves_exercise():
    request = RoutineRequest(
        target_split_label="legs",
        time_available_min=45,
        doms_data={"left-soleus": 2},
    )
    response = RoutineDraftResponse(
        generation_status="success",
        status_reason_code="none",
        is_fallback=False,
        summary_title="하체 루틴",
        rationale_summary=[],
        routine_blocks=[{
            "order": 1,
            "exerciseId": "calf_raise",
            "exerciseName": "머신 카프레이즈",
            "primaryMuscles": ["calves"],
            "defaultRestSec": 60,
            "prescription": [{
                "setIndex": 1,
                "targetReps": 12,
                "targetRestSec": 60,
            }],
            "exerciseRationale": "종아리 운동입니다.",
        }],
        warnings=[],
    )
    candidates = [{
        "id": "calf_raise",
        "name_kr": "머신 카프레이즈",
        "primary_muscle": "calves",
        "movement_type": "ISOLATION",
    }]

    guarded = _apply_pain_and_doms_warnings(
        response,
        request,
        candidates,
        ["quadriceps", "hamstring", "gluteal", "calves"],
    )

    assert any("강한 피로 부위" in warning and "종아리" in warning for warning in guarded.warnings)


def test_recovery_constraint_is_target_aware_for_mild_doms_and_injury():
    push_targets = ["chest", "front-deltoids", "triceps"]

    assert not has_target_relevant_recovery_constraint(
        RoutineRequest(target_split_label="push", time_available_min=45, doms_data={"calves": 1}),
        push_targets,
    )
    assert has_target_relevant_recovery_constraint(
        RoutineRequest(target_split_label="push", time_available_min=45, doms_data={"chest": 1}),
        push_targets,
    )
    assert not has_target_relevant_recovery_constraint(
        RoutineRequest(
            target_split_label="push",
            time_available_min=45,
            pain_areas=[PainAreaEntry(body_part="knees")],
        ),
        push_targets,
    )
    assert has_target_relevant_recovery_constraint(
        RoutineRequest(
            target_split_label="push",
            time_available_min=45,
            pain_areas=[PainAreaEntry(body_part="front-deltoids")],
        ),
        push_targets,
    )


@pytest.mark.parametrize("body_part", ["lower-back", "knees"])
def test_lower_body_underfill_recognizes_indirect_injury_constraints(body_part):
    lower_targets = ["quadriceps", "hamstring", "gluteal", "calves"]

    assert has_target_relevant_recovery_constraint(
        RoutineRequest(
            target_split_label="legs",
            time_available_min=50,
            pain_areas=[PainAreaEntry(body_part=body_part)],
        ),
        lower_targets,
    )


def test_productive_fill_records_candidate_policy_exhaustion_without_forcing_duplicate():
    request = RoutineRequest(
        target_split_label="pull",
        target_muscles=["upper-back", "back-deltoids", "biceps"],
        time_available_min=60,
        readiness_level="normal",
        session_structure="standard",
        pain_areas=[],
        doms_data={},
    )
    blueprint = build_routine_blueprint(
        target_split_label="pull",
        target_muscles=request.target_muscles,
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=60,
        session_structure="standard",
    )
    candidate_rows = [
        ("lat_pulldown", "랫풀다운", "upper-back", "COMPOUND"),
        ("dumbbell_row", "덤벨 로우", "upper-back", "COMPOUND"),
        ("reverse_pec_deck", "리버스 펙덱", "back-deltoids", "ISOLATION"),
        ("hammer_curl", "해머 컬", "biceps", "ISOLATION"),
        ("face_pull", "페이스 풀", "back-deltoids", "ISOLATION"),
        ("cable_curl", "케이블 컬", "biceps", "ISOLATION"),
    ]
    ranked_candidates = [
        {
            "id": exercise_id,
            "name_kr": name,
            "primary_muscle": primary,
            "secondary_muscle": None,
            "equipment_req": "MACHINE",
            "movement_type": movement_type,
        }
        for exercise_id, name, primary, movement_type in candidate_rows
    ]
    output = LLMRoutineOutput(
        total_estimated_time=35,
        summary_title="풀 루틴",
        rationale_summary=["목표 부위를 구성했습니다."],
        warnings=[],
        exercises=[
            LLMExercisePlan(
                exercise_id=exercise_id,
                exercise_name=name,
                movement_type=movement_type,
                primary_muscles=[primary],
                equipment_type="MACHINE",
                target_reps=10,
                sets=2,
                rest_time_sec=60,
                exercise_rationale="목표 부위 운동입니다.",
            )
            for exercise_id, name, primary, movement_type in candidate_rows[:5]
        ],
    )

    adjusted = _add_underfilled_safe_candidates(
        output,
        request,
        ranked_candidates,
        max_total_sets=22,
        db_target_muscles=request.target_muscles,
        routine_blueprint=blueprint,
    )

    assert len(adjusted.exercises) == 5
    assert adjusted._selection_diagnostics["productive_fill_requested"] is True
    assert adjusted._selection_diagnostics["productive_fill_added_count"] == 0
    assert adjusted._selection_diagnostics["productive_fill_exhausted"] is True
    assert (
        adjusted._selection_diagnostics["productive_fill_exhaustion_reason"]
        == "candidate_policy_exhausted"
    )
    assert any("추가할 적합한 운동 후보가 없어" in warning for warning in adjusted.warnings)


def test_beginner_strength_productive_fill_runs_after_experience_prescription():
    request = RoutineRequest(
        target_split_label="legs",
        target_muscles=["quadriceps", "hamstring", "gluteal", "adductor", "abductors"],
        time_available_min=50,
        readiness_level="normal",
        experience_level="beginner",
        session_structure="standard",
        goal="strength",
        pain_areas=[],
        doms_data={},
    )
    blueprint = build_routine_blueprint(
        target_split_label=request.target_split_label,
        target_muscles=request.target_muscles,
        goal=request.goal,
        experience_level=request.experience_level,
        readiness_level=request.readiness_level,
        time_available_min=request.time_available_min,
        session_structure=request.session_structure,
    )
    ranked_candidates = [
        {
            "id": "v_squat",
            "name_kr": "브이 스쿼트",
            "primary_muscle": "quadriceps",
            "secondary_muscle": "gluteal",
            "equipment_req": "MACHINE",
            "movement_type": "COMPOUND",
            "selection_phase": "main_compound",
        },
        {
            "id": "side_lunge",
            "name_kr": "덤벨 사이드 런지",
            "primary_muscle": "adductor",
            "secondary_muscle": "gluteal,quadriceps",
            "equipment_req": "DUMBBELL",
            "movement_type": "COMPOUND",
            "selection_phase": "secondary_compound",
        },
        {
            "id": "leg_curl",
            "name_kr": "머신 레그 컬",
            "primary_muscle": "hamstring",
            "secondary_muscle": "gluteal",
            "equipment_req": "MACHINE",
            "movement_type": "ISOLATION",
            "selection_phase": "isolation_accessory",
        },
        {
            "id": "hip_abduction",
            "name_kr": "머신 힙 어브덕션",
            "primary_muscle": "abductors",
            "secondary_muscle": "gluteal",
            "equipment_req": "MACHINE",
            "movement_type": "ISOLATION",
            "selection_phase": "isolation_accessory",
        },
    ]
    output = LLMRoutineOutput(
        total_estimated_time=40,
        summary_title="초보자 하체 근력 루틴",
        rationale_summary=["안정적인 하체 운동을 선택했습니다."],
        warnings=[],
        exercises=[
            LLMExercisePlan(
                exercise_id=candidate["id"],
                exercise_name=candidate["name_kr"],
                movement_type=candidate["movement_type"],
                primary_muscles=[candidate["primary_muscle"]],
                equipment_type=candidate["equipment_req"],
                target_reps=5,
                sets=4,
                rest_time_sec=180,
                exercise_rationale="목표 부위 운동입니다.",
            )
            for candidate in ranked_candidates[:3]
        ],
    )
    diagnostics = {}

    response = _deterministic_or_fallback(
        output,
        request,
        ranked_candidates,
        max_total_sets=11,
        doms_db={},
        goal="strength",
        db_target_muscles=request.target_muscles,
        recent_sets=[],
        profile=None,
        output_label="TEST",
        draft_label="TEST",
        routine_blueprint=blueprint,
        selection_diagnostics=diagnostics,
    )

    assert [block.exercise_id for block in response.routine_blocks] == [
        "v_squat",
        "side_lunge",
        "leg_curl",
        "hip_abduction",
    ]
    assert response.total_estimated_time >= 33
    assert diagnostics["productive_fill_requested"] is True
    assert diagnostics["productive_fill_added_count"] == 1


def test_active_context_rationale_adds_missing_user_visible_constraints():
    response = RoutineDraftResponse(
        generation_status="success",
        status_reason_code="none",
        is_fallback=False,
        summary_title="하체 루틴",
        rationale_summary=["주요 하체 근육을 균형 있게 구성했습니다."],
        routine_blocks=[],
        warnings=[],
    )
    request = RoutineRequest(
        target_split_label="legs",
        time_available_min=40,
        readiness_level="low",
        experience_level="beginner",
        pain_areas=[PainAreaEntry(body_part="lower-back")],
        mobility_limits=["limited_ankle_dorsiflexion"],
    )

    guarded = _apply_active_context_rationale(response, request)
    text = " ".join(guarded.rationale_summary)

    assert "허리/요추" in text
    assert "발뒤꿈치를 붙인 깊은 앉기 동작의 어려움" in text
    assert "컨디션이 낮아" in text
    assert "초보자 숙련도" in text


def test_active_context_rationale_does_not_duplicate_existing_context():
    response = RoutineDraftResponse(
        generation_status="success",
        status_reason_code="none",
        is_fallback=False,
        summary_title="하체 루틴",
        rationale_summary=["발목 가동범위 제한을 고려해 머신 운동을 우선했습니다."],
        routine_blocks=[],
        warnings=[],
    )
    request = RoutineRequest(
        target_split_label="legs",
        time_available_min=40,
        mobility_limits=["limited_ankle_dorsiflexion"],
    )

    guarded = _apply_active_context_rationale(response, request)

    assert guarded.rationale_summary == ["발목 가동범위 제한을 고려해 머신 운동을 우선했습니다."]


def test_selected_high_mobility_demand_adds_non_injury_guard_warning():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 20,
        "summaryTitle": "하체 루틴",
        "rationaleSummary": [],
        "routineBlocks": [{
            "order": 1,
            "exerciseId": "machine_calf_raise",
            "exerciseName": "머신 카프 레이즈",
            "primaryMuscles": ["calves"],
            "defaultRestSec": 75,
            "prescription": [{"setIndex": 1, "targetReps": 12, "targetRestSec": 75}],
            "exerciseRationale": "종아리 보조 운동입니다.",
        }],
        "warnings": [],
    })
    request = RoutineRequest(
        target_split_label="legs",
        time_available_min=40,
        profile_signals=["heel_down_deep_squat_difficult"],
    )

    guarded = _apply_selected_mobility_demand_warnings(
        response,
        request,
        ranked_candidates=[{
            "id": "machine_calf_raise",
            "ankle_dorsiflexion_demand": "high",
        }],
    )

    warning_text = " ".join(guarded.warnings)
    assert "머신 카프 레이즈" in warning_text
    assert "가동범위 요구도가 높은" in warning_text
    assert "대체 운동" in warning_text
    assert "부상" not in warning_text


def test_selected_low_mobility_demand_does_not_add_guard_warning():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 20,
        "summaryTitle": "하체 루틴",
        "rationaleSummary": [],
        "routineBlocks": [{
            "order": 1,
            "exerciseId": "leg_curl",
            "exerciseName": "머신 레그 컬",
            "primaryMuscles": ["hamstring"],
            "defaultRestSec": 75,
            "prescription": [{"setIndex": 1, "targetReps": 12, "targetRestSec": 75}],
            "exerciseRationale": "햄스트링 보조 운동입니다.",
        }],
        "warnings": [],
    })
    request = RoutineRequest(
        target_split_label="legs",
        time_available_min=40,
        profile_signals=["heel_down_deep_squat_difficult"],
    )

    guarded = _apply_selected_mobility_demand_warnings(
        response,
        request,
        ranked_candidates=[{
            "id": "leg_curl",
            "ankle_dorsiflexion_demand": "low",
        }],
    )

    assert guarded.warnings == []


def test_selected_press_with_wrist_extension_difficulty_adds_setup_guidance():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 20,
        "summaryTitle": "밀기 루틴",
        "rationaleSummary": [],
        "routineBlocks": [{
            "order": 1,
            "exerciseId": "dumbbell_press",
            "exerciseName": "덤벨 체스트 프레스",
            "primaryMuscles": ["chest"],
            "defaultRestSec": 90,
            "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
            "exerciseRationale": "가슴 메인 운동입니다.",
        }],
        "warnings": [],
    })
    request = RoutineRequest(
        target_split_label="push",
        time_available_min=40,
        profile_signals=["loaded_wrist_extension_difficult"],
    )

    guarded = _apply_selected_mobility_demand_warnings(
        response,
        request,
        mobility_limits=["limited_wrist_extension"],
        ranked_candidates=[{
            "id": "dumbbell_press",
            "name_kr": "덤벨 체스트 프레스",
            "primary_muscle": "chest",
            "movement_type": "COMPOUND",
            "wrist_extension_demand": "medium",
        }],
    )

    warning_text = " ".join(guarded.warnings)
    assert "손목을 중립에 가깝게" in warning_text
    assert "뒤로 과도하게 젖혀" in warning_text
    assert "통증이 없다" not in warning_text


def test_wrist_setup_guidance_does_not_treat_leg_press_as_upper_body_press():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 20,
        "summaryTitle": "전신 루틴",
        "rationaleSummary": [],
        "routineBlocks": [{
            "order": 1,
            "exerciseId": "leg_press",
            "exerciseName": "머신 레그 프레스",
            "primaryMuscles": ["quadriceps"],
            "defaultRestSec": 90,
            "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
            "exerciseRationale": "하체 메인 운동입니다.",
        }],
        "warnings": [],
    })
    request = RoutineRequest(
        target_split_label="full_body",
        time_available_min=40,
        profile_signals=["loaded_wrist_extension_difficult"],
    )

    guarded = _apply_selected_mobility_demand_warnings(
        response,
        request,
        mobility_limits=["limited_wrist_extension"],
        ranked_candidates=[{
            "id": "leg_press",
            "name_kr": "머신 레그 프레스",
            "primary_muscle": "quadriceps",
            "movement_type": "COMPOUND",
            "wrist_extension_demand": "low",
        }],
    )

    assert "손목을 중립에 가깝게" not in " ".join(guarded.warnings)


def test_ankle_depth_guidance_does_not_tell_calf_raise_to_keep_heel_down():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 10,
        "summaryTitle": "하체 루틴",
        "rationaleSummary": [],
        "routineBlocks": [{
            "order": 1,
            "exerciseId": "calf_raise",
            "exerciseName": "머신 카프 레이즈",
            "primaryMuscles": ["calves"],
            "defaultRestSec": 60,
            "prescription": [{"setIndex": 1, "targetReps": 12, "targetRestSec": 60}],
            "exerciseRationale": "종아리 보조 운동입니다.",
        }],
        "warnings": [],
    })
    request = RoutineRequest(
        target_split_label="legs",
        time_available_min=40,
        mobility_limits=["limited_ankle_dorsiflexion"],
    )

    guarded = _apply_selected_mobility_demand_warnings(
        response,
        request,
        ranked_candidates=[{
            "id": "calf_raise",
            "name_kr": "머신 카프 레이즈",
            "primary_muscle": "calves",
            "movement_type": "ISOLATION",
            "ankle_dorsiflexion_demand": "low",
        }],
    )

    assert guarded.warnings == []


@pytest.mark.parametrize(
    ("area_code", "pattern", "exercise_id", "exercise_name", "expected"),
    [
        ("knee", "deepKneeFlexion", "leg_press", "머신 레그 프레스", "깊은 무릎 굽힘"),
        ("wrist", "loadedWristExtension", "dumbbell_press", "덤벨 체스트 프레스", "손목을 중립에 가깝게"),
    ],
)
def test_managed_restriction_adds_selected_movement_setup_guidance(
    area_code,
    pattern,
    exercise_id,
    exercise_name,
    expected,
):
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 20,
        "summaryTitle": "관리 제한 루틴",
        "rationaleSummary": [],
        "routineBlocks": [{
            "order": 1,
            "exerciseId": exercise_id,
            "exerciseName": exercise_name,
            "primaryMuscles": ["quadriceps" if area_code == "knee" else "chest"],
            "defaultRestSec": 90,
            "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
            "exerciseRationale": "제한 범위 안에서 수행합니다.",
        }],
        "warnings": [],
    })
    request = RoutineRequest.model_validate({
        "targetSplitLabel": "legs" if area_code == "knee" else "push",
        "timeAvailableMin": 40,
        "injuryConditions": [{
            "areaCode": area_code,
            "status": "managedWithRestrictions",
            "restrictedMovementPatterns": [pattern],
        }],
    })

    guarded = _apply_selected_mobility_demand_warnings(
        response,
        request,
        ranked_candidates=[{
            "id": exercise_id,
            "name_kr": exercise_name,
            "primary_muscle": "quadriceps" if area_code == "knee" else "chest",
            "movement_type": "COMPOUND",
        }],
    )

    assert expected in " ".join(guarded.warnings)


def test_managed_setup_guidance_uses_only_selected_candidates():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 15,
        "summaryTitle": "제한 범위 하체 루틴",
        "rationaleSummary": [],
        "routineBlocks": [{
            "order": 1,
            "exerciseId": "leg_curl",
            "exerciseName": "머신 레그 컬",
            "primaryMuscles": ["hamstring"],
            "defaultRestSec": 75,
            "prescription": [{"setIndex": 1, "targetReps": 12, "targetRestSec": 75}],
            "exerciseRationale": "선택된 운동입니다.",
        }],
        "warnings": [],
    })
    request = RoutineRequest.model_validate({
        "targetSplitLabel": "legs",
        "timeAvailableMin": 40,
        "injuryConditions": [{
            "areaCode": "knee",
            "status": "managedWithRestrictions",
            "restrictedMovementPatterns": ["deepKneeFlexion"],
        }],
    })

    guarded = _apply_selected_mobility_demand_warnings(
        response,
        request,
        ranked_candidates=[
            {"id": "leg_curl", "name_kr": "머신 레그 컬", "primary_muscle": "hamstring", "movement_type": "ISOLATION"},
            {"id": "excluded_leg_press", "name_kr": "머신 레그 프레스", "primary_muscle": "quadriceps", "movement_type": "COMPOUND"},
        ],
    )

    assert "깊은 무릎 굽힘" not in " ".join(guarded.warnings)


@pytest.mark.parametrize(
    ("split_label", "profile_signal", "expected_text"),
    [
        ("push", "self_reported_long_arms", "팔이 유난히 긴 편이라는 입력"),
        ("legs", "self_reported_long_thighs", "허벅지 구간이 유난히 긴 편이라는 입력"),
    ],
)
def test_active_context_rationale_explains_reviewed_body_proportion_signal(
    split_label,
    profile_signal,
    expected_text,
):
    response = RoutineDraftResponse(
        generation_status="success",
        status_reason_code="none",
        is_fallback=False,
        summary_title="루틴",
        rationale_summary=[],
        routine_blocks=[],
        warnings=[],
    )
    request = RoutineRequest(
        target_split_label=split_label,
        time_available_min=60,
        profile_signals=[profile_signal],
    )

    guarded = _apply_active_context_rationale(response, request)

    assert expected_text in " ".join(guarded.rationale_summary)


def test_active_context_rationale_adds_short_time_reason_once():
    response = RoutineDraftResponse(
        generation_status="success",
        status_reason_code="none",
        is_fallback=False,
        summary_title="푸시 루틴",
        rationale_summary=["가슴과 삼두를 중심으로 구성했습니다."],
        routine_blocks=[],
        warnings=[],
    )
    request = RoutineRequest(
        target_split_label="push",
        time_available_min=20,
        experience_level="intermediate",
    )

    guarded = _apply_active_context_rationale(response, request)
    guarded = _apply_active_context_rationale(guarded, request)

    assert sum("20분 안에" in item for item in guarded.rationale_summary) == 1
    assert any("핵심 메인 운동" in item for item in guarded.rationale_summary)


def test_same_area_injury_and_doms_wording_keeps_injury_precedence():
    response = RoutineDraftResponse(
        generation_status="success",
        status_reason_code="none",
        is_fallback=False,
        summary_title="조정된 푸시 루틴",
        rationale_summary=["전면 어깨에 근육통이 있어 해당 부위 자극을 줄였습니다."],
        routine_blocks=[{
            "order": 1,
            "exerciseId": "pushdown",
            "exerciseName": "머신 푸시다운",
            "primaryMuscles": ["triceps"],
            "defaultRestSec": 75,
            "prescription": [{"setIndex": 1, "targetReps": 12, "targetRestSec": 75}],
            "exerciseRationale": "근육통 관리가 필요한 상황에서 삼두근을 선택했습니다.",
        }],
        warnings=["전면 어깨 피로를 고려했습니다."],
    )
    request = RoutineRequest(
        target_split_label="push",
        time_available_min=45,
        pain_areas=[PainAreaEntry(body_part="front-deltoids")],
        doms_data={"front-deltoids": 2},
    )

    guarded = _enforce_same_area_injury_precedence_wording(response, request)
    visible_text = " ".join([*guarded.rationale_summary, *guarded.warnings])

    assert "등록된 부상 부위(전면 어깨)" in visible_text
    assert "전면 어깨에 근육통" not in visible_text
    assert "전면 어깨 피로" not in visible_text
    assert "근육통 관리" not in guarded.routine_blocks[0].exercise_rationale


def test_active_context_rationale_omits_mobility_limit_irrelevant_to_selected_split():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 40,
        "summaryTitle": "푸시 루틴",
        "rationaleSummary": ["가슴과 삼두를 중심으로 구성했습니다."],
        "routineBlocks": [{
            "order": 1,
            "exerciseId": "dumbbell_press",
            "exerciseName": "덤벨 체스트 프레스",
            "equipmentType": "DUMBBELL",
            "primaryMuscles": ["chest"],
            "defaultRestSec": 90,
            "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
            "exerciseRationale": "가슴 메인 운동입니다.",
            "reasons": [],
            "boosts": [],
            "penalties": [],
        }],
        "warnings": [],
    })
    request = RoutineRequest(
        target_split_label="push",
        time_available_min=40,
        mobility_limits=["limited_hip_flexion"],
    )

    guarded = _apply_active_context_rationale(response, request)

    assert not any("고관절 굴곡" in item for item in guarded.rationale_summary)


def test_response_fact_consistency_repairs_general_fitness_goal_wording():
    response = RoutineDraftResponse(
        generation_status="success",
        status_reason_code="none",
        is_fallback=False,
        summary_title="전반적인 근력 향상 루틴",
        rationale_summary=["전반적인 근력 향상을 목표로 구성했습니다."],
        routine_blocks=[],
        warnings=[],
    )
    request = RoutineRequest(
        target_split_label="push",
        time_available_min=45,
        goal="generalFitness",
    )

    guarded = _enforce_response_fact_consistency(response, request)

    text = " ".join([guarded.summary_title, *guarded.rationale_summary])
    assert "전반적인 체력 향상" in text
    assert "전반적인 근력 향상" not in text


@pytest.mark.parametrize("exercise_name", [
    "덤벨 사이드 레터럴 레이즈",
    "머신 레터럴 레이즈",
])
def test_response_fact_consistency_repairs_lateral_raise_anatomy_without_contract_change(exercise_name):
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 20,
        "summaryTitle": "어깨 보조 루틴",
        "rationaleSummary": ["전면 삼각근과 삼두근에 집중했습니다."],
        "routineBlocks": [{
            "order": 1,
            "exerciseId": "lateral_raise",
            "exerciseName": exercise_name,
            "movementType": "ISOLATION",
            "equipmentType": "DUMBBELL",
            "primaryMuscles": ["front-deltoids"],
            "defaultRestSec": 60,
            "prescription": [{"setIndex": 1, "targetReps": 12, "targetRestSec": 60}],
            "exerciseRationale": "전면 삼각근과 가슴 보조 볼륨을 채우는 운동입니다.",
            "reasons": [],
            "boosts": [],
            "penalties": [],
        }],
        "warnings": ["전면 삼각근과 삼두근 위주로 구성했습니다."],
    })
    request = RoutineRequest(target_split_label="push", time_available_min=20)

    guarded = _enforce_response_fact_consistency(response, request)

    assert guarded.routine_blocks[0].primary_muscles == ["front-deltoids"]
    assert guarded.routine_blocks[0].exercise_rationale == (
        "측면 어깨 보조 볼륨을 채우는 고립 운동으로 배치했습니다."
    )
    assert "측면 어깨" in guarded.rationale_summary[0]
    assert "전면 삼각근" not in guarded.rationale_summary[0]
    assert "측면 어깨와 삼두근" in guarded.warnings[0]
    assert "전면 삼각근" not in guarded.warnings[0]


def test_response_fact_consistency_preserves_lateral_focus_application_evidence():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 20,
        "summaryTitle": "측면 어깨 집중 루틴",
        "rationaleSummary": ["측면 어깨 집중 루틴입니다."],
        "routineBlocks": [{
            "order": 1,
            "exerciseId": "lateral_raise",
            "exerciseName": "머신 레터럴 레이즈",
            "movementType": "ISOLATION",
            "equipmentType": "MACHINE",
            "primaryMuscles": ["side-deltoids"],
            "defaultRestSec": 60,
            "prescription": [{"setIndex": 1, "targetReps": 12, "targetRestSec": 60}],
            "exerciseRationale": (
                "측면 어깨 운동입니다. 측면 어깨 집중 목표를 반영해 "
                "해당 운동과 수행 각도를 우선 배치했습니다."
            ),
            "reasons": [],
            "boosts": [],
            "penalties": [],
        }],
        "warnings": [],
    })
    request = RoutineRequest(target_split_label="shoulder", time_available_min=20)

    guarded = _enforce_response_fact_consistency(response, request)

    rationale = guarded.routine_blocks[0].exercise_rationale
    assert "측면 어깨 보조 볼륨" in rationale
    assert "집중 목표를 반영" in rationale


def test_response_fact_consistency_repairs_lateral_isolation_summary_with_press_present():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 30,
        "summaryTitle": "푸시 루틴",
        "rationaleSummary": [
            "가슴 프레스로 시작했습니다.",
            "보조적으로 인클라인 각도를 이용해 전면 삼각근에 집중하는 고립 운동을 배치했습니다.",
            "이후 측면 어깨와 가슴의 보조적인 자극을 위해 덤벨 인클라인 벤치 뉴트럴 사이드 레터럴 레이즈를 수행합니다.",
        ],
        "routineBlocks": [{
            "order": 1,
            "exerciseId": "press",
            "exerciseName": "덤벨 인클라인 체스트 프레스",
            "movementType": "COMPOUND",
            "equipmentType": "DUMBBELL",
            "primaryMuscles": ["chest"],
            "defaultRestSec": 90,
            "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
            "exerciseRationale": "가슴 메인 운동입니다.",
        }, {
            "order": 2,
            "exerciseId": "lateral_raise",
            "exerciseName": "덤벨 인클라인 벤치 뉴트럴 사이드 레터럴 레이즈",
            "movementType": "ISOLATION",
            "equipmentType": "DUMBBELL",
            "primaryMuscles": ["front-deltoids"],
            "defaultRestSec": 60,
            "prescription": [{"setIndex": 1, "targetReps": 12, "targetRestSec": 60}],
            "exerciseRationale": "전면 삼각근 고립 운동입니다.",
        }],
        "warnings": [],
    })
    request = RoutineRequest(target_split_label="push", time_available_min=30)

    guarded = _enforce_response_fact_consistency(response, request)

    assert guarded.rationale_summary[0] == "가슴 프레스로 시작했습니다."
    assert "측면 어깨에 집중하는 고립 운동" in guarded.rationale_summary[1]
    assert "전면 삼각근에 집중하는 고립 운동" not in guarded.rationale_summary[1]
    assert "측면 어깨의 보조적인 자극" in guarded.rationale_summary[2]
    assert "측면 어깨와 가슴" not in guarded.rationale_summary[2]


def test_response_fact_consistency_removes_suppressed_target_composition_claim():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 17,
        "summaryTitle": "조정된 푸시 루틴",
        "rationaleSummary": [
            "가슴과 삼두근을 중심으로 구성된 푸시 운동 루틴입니다.",
            "전면 어깨 부위의 부담을 고려하여 고립 운동 위주로 배치했습니다.",
        ],
        "routineBlocks": [{
            "order": 1,
            "exerciseId": "machine_pushdown",
            "exerciseName": "머신 푸시다운",
            "movementType": "ISOLATION",
            "equipmentType": "MACHINE",
            "primaryMuscles": ["triceps"],
            "defaultRestSec": 75,
            "prescription": [{"setIndex": 1, "targetReps": 12, "targetRestSec": 75}],
            "exerciseRationale": "삼두 직접 운동입니다.",
        }],
        "warnings": [],
    })
    request = RoutineRequest(
        target_split_label="push",
        time_available_min=50,
        pain_areas=[PainAreaEntry(body_part="front-deltoids")],
    )

    guarded = _enforce_response_fact_consistency(response, request)

    assert guarded.rationale_summary[0] == (
        "등록 부상과 겹치는 목표 운동을 제외하고 삼두 중심으로 구성했습니다."
    )
    assert not any("가슴과 삼두" in item for item in guarded.rationale_summary)
    assert any("전면 어깨 부위의 부담" in item for item in guarded.rationale_summary)


def test_response_fact_consistency_removes_machine_cable_only_claim_when_dumbbell_is_selected():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 25,
        "summaryTitle": "푸시 보조 루틴",
        "rationaleSummary": ["중량 조절이 쉬운 머신 및 케이블 위주로 구성했습니다."],
        "routineBlocks": [{
            "order": 1,
            "exerciseId": "dumbbell_raise",
            "exerciseName": "덤벨 사이드 레터럴 레이즈",
            "movementType": "ISOLATION",
            "equipmentType": "DUMBBELL",
            "primaryMuscles": ["front-deltoids"],
            "defaultRestSec": 60,
            "prescription": [{"setIndex": 1, "targetReps": 12, "targetRestSec": 60}],
            "exerciseRationale": "측면 어깨 보조 운동입니다.",
            "reasons": [],
            "boosts": [],
            "penalties": [],
        }, {
            "order": 2,
            "exerciseId": "cable_pushdown",
            "exerciseName": "케이블 푸시다운",
            "movementType": "ISOLATION",
            "equipmentType": "CABLE",
            "primaryMuscles": ["triceps"],
            "defaultRestSec": 60,
            "prescription": [{"setIndex": 1, "targetReps": 12, "targetRestSec": 60}],
            "exerciseRationale": "삼두 보조 운동입니다.",
            "reasons": [],
            "boosts": [],
            "penalties": [],
        }],
        "warnings": [],
    })
    request = RoutineRequest(target_split_label="push", time_available_min=25)

    guarded = _enforce_response_fact_consistency(response, request)

    text = " ".join(guarded.rationale_summary)
    assert "머신 및 케이블 위주" not in text
    assert "선택된 장비를 사용하도록" in text


def test_experience_rationale_consistency_repairs_conflicting_level_claims():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 35,
        "summaryTitle": "초급자 푸시 루틴",
        "rationaleSummary": ["초보자도 안정적으로 수행할 수 있도록 구성했습니다."],
        "routineBlocks": [{
            "order": 1,
            "exerciseId": "machine_press",
            "exerciseName": "머신 체스트 프레스",
            "equipmentType": "MACHINE",
            "primaryMuscles": ["chest"],
            "defaultRestSec": 90,
            "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
            "exerciseRationale": "초급자도 점진적으로 중량을 높일 수 있습니다.",
            "reasons": [],
            "boosts": [],
            "penalties": [],
        }],
        "warnings": [],
    })

    guarded = _enforce_experience_rationale_consistency(response, "intermediate")
    text = " ".join([
        guarded.summary_title,
        *guarded.rationale_summary,
        guarded.routine_blocks[0].exercise_rationale,
    ])

    assert "중급자" in text
    assert "초급자" not in text
    assert "초보자" not in text


def test_active_context_rationale_adds_missing_intermediate_level():
    response = RoutineDraftResponse(
        generation_status="success",
        status_reason_code="none",
        is_fallback=False,
        summary_title="푸시 루틴",
        rationale_summary=["가슴과 삼두를 순서대로 구성했습니다."],
        routine_blocks=[],
        warnings=[],
    )
    request = RoutineRequest(
        target_split_label="push",
        time_available_min=45,
        experience_level="intermediate",
    )

    guarded = _apply_active_context_rationale(response, request)

    assert any("중급자 숙련도" in item for item in guarded.rationale_summary)


def test_active_context_rationale_adds_selected_effect_risk_tradeoff_for_advanced_high_strength():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 55,
        "summaryTitle": "상급자 상체 근력",
        "rationaleSummary": ["높은 컨디션에서 상체 근력을 중심으로 구성했습니다."],
        "routineBlocks": [{
            "order": 1,
            "exerciseId": "dumbbell_bench_press",
            "exerciseName": "덤벨 벤치프레스",
            "equipmentType": "DUMBBELL",
            "primaryMuscles": ["chest"],
            "defaultRestSec": 120,
            "prescription": [{"setIndex": 1, "targetReps": 6, "targetRestSec": 120}],
            "exerciseRationale": "상체 밀기 메인 운동입니다.",
            "reasons": [],
            "boosts": [],
            "penalties": [],
        }],
        "warnings": [],
    })
    request = RoutineRequest(
        target_split_label="upper",
        time_available_min=65,
        readiness_level="high",
        experience_level="advanced",
        goal="strength",
    )
    ranked_candidates = [{
        "id": "dumbbell_bench_press",
        "strength_effect": "high",
        "injury_caution_level": "medium",
        "lumbar_load": "low",
        "axial_load": "low",
    }]

    guarded = _apply_active_context_rationale(
        response,
        request,
        ranked_candidates=ranked_candidates,
    )
    text = " ".join(guarded.rationale_summary)

    assert "근력 효과" in text
    assert "부상 주의도" in text
    assert "부하" in text


def _make_db_mock():
    return MagicMock(spec=Session)


def _make_llm_mock(return_value=None, side_effect=None):
    """get_llm() 반환값 mock을 만든다. with_structured_output은 RunnableLambda를 반환한다."""
    mock_llm = MagicMock()

    if side_effect is not None:
        def _raise(_):
            raise side_effect
        structured = RunnableLambda(_raise)
    else:
        structured = RunnableLambda(lambda _: return_value)

    mock_llm.with_structured_output.return_value = structured
    return mock_llm


def test_canonicalize_llm_exercise_names_uses_ranked_candidate_display_name():
    output = LLMRoutineOutput(
        total_estimated_time=30,
        summary_title="하체 루틴",
        rationale_summary=[],
        warnings=[],
        exercises=[
            LLMExercisePlan(
                exercise_id="V24ROW-011308",
                exercise_name="컬",
                primary_muscles=["hamstring"],
                target_reps=10,
                sets=2,
                rest_time_sec=90,
                exercise_rationale="햄스트링 보조 운동입니다.",
            )
        ],
    )
    ranked_candidates = [
        {
            "id": "V24ROW-011308",
            "name_kr": "소도구 미니볼 펠빅컬 - 볼 지지 보조형 정렬 중심",
            "display_name_ko": "소도구 미니볼 펠빅컬 - 볼 지지 보조형 정렬 중심",
            "user_facing_display_name_kr": "컬",
            "primary_muscle": "hamstring",
        }
    ]

    canonicalized = _canonicalize_llm_exercise_names(output, ranked_candidates)

    assert canonicalized.exercises[0].exercise_name == "소도구 미니볼 펠빅컬 - 볼 지지 보조형 정렬 중심"


def test_requested_crossfit_adjunct_is_preserved_after_llm_selection():
    output = LLMRoutineOutput(
        total_estimated_time=42,
        summary_title="하체 루틴",
        rationale_summary=[],
        warnings=[],
        exercises=[
            LLMExercisePlan(
                exercise_id="gym_main",
                exercise_name="브이 스쿼트",
                primary_muscles=["quadriceps"],
                movement_type="COMPOUND",
                target_reps=10,
                sets=3,
                rest_time_sec=120,
                exercise_rationale="메인 운동입니다.",
            ),
            LLMExercisePlan(
                exercise_id="gym_accessory",
                exercise_name="케이블 힙 어브덕션",
                primary_muscles=["abductors"],
                movement_type="ISOLATION",
                target_reps=12,
                sets=2,
                rest_time_sec=90,
                exercise_rationale="보조 운동입니다.",
            ),
        ],
    )
    request = RoutineRequest(
        target_split_label="legs",
        time_available_min=50,
        readiness_level="normal",
        training_style="crossfit",
    )
    ranked_candidates = [
        {
            "id": "gym_main",
            "name_kr": "브이 스쿼트",
            "primary_muscle": "quadriceps",
            "movement_type": "COMPOUND",
            "equipment_req": "MACHINE",
        },
        {
            "id": "gym_accessory",
            "name_kr": "케이블 힙 어브덕션",
            "primary_muscle": "abductors",
            "movement_type": "ISOLATION",
            "equipment_req": "CABLE",
        },
        {
            "id": "crossfit_adjunct",
            "name_kr": "덤벨 AMRAP 스러스터",
            "primary_muscle": "quadriceps",
            "movement_type": "COMPOUND",
            "equipment_req": "DUMBBELL",
            "score_boosts": [{
                "ruleCode": "training_style_match_boost",
                "constraintCode": "crossfit",
                "score": 8,
                "reason": "요청한 보조 운동 스타일과 맞는 후보입니다.",
            }],
        },
    ]

    adjusted = _ensure_requested_adjunct_style_candidate(
        output,
        request,
        ranked_candidates,
        max_total_sets=5,
    )

    ids = [exercise.exercise_id for exercise in adjusted.exercises]
    assert ids[0] == "gym_main"
    assert "crossfit_adjunct" in ids
    assert "gym_accessory" not in ids
    assert any("보조 운동 스타일" in warning for warning in adjusted.warnings)


class _Row:
    def __init__(self, mapping):
        self._mapping = mapping


class _FetchOne:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class TestProfileContext:
    def test_strength_baseline_list_from_backend_is_indexed_by_exercise_id_and_name(self):
        db = _make_db_mock()
        db.execute.return_value = _FetchOne(_Row({
            "goal_type": "strength",
            "split_type": "pushPullLegs",
            "split_label": "push",
            "experience_level": "intermediate",
            "strength_baseline": [
                {
                    "exerciseId": "30",
                    "exerciseNameSnapshot": "Barbell Bench Press",
                    "workingWeightKg": 100,
                    "reps": 5,
                }
            ],
            "equipment_access": [],
            "pain_areas": [],
        }))

        profile = get_user_profile_context(db, "user-1")

        assert profile is not None
        assert profile.strength_baseline["30"]["weight_kg"] == 100.0
        assert profile.strength_baseline["Barbell Bench Press"]["reps"] == 5


class TestHappyPath:
    def test_accessory_only_push_is_failed_instead_of_reported_as_success(self):
        request = RoutineRequest(
            user_id="accessory-only-push",
            target_split_label="push",
            target_muscles=["chest", "triceps"],
            readiness_level="normal",
            time_available_min=45,
            pain_areas=[PainAreaEntry(body_part="front-deltoids")],
            doms_data={},
        )
        candidates = [{
            "id": "pushdown",
            "name_kr": "케이블 푸시다운",
            "primary_muscle": "triceps",
            "movement_type": "ISOLATION",
            "equipment_req": "CABLE",
            "score": 100,
        }]
        output = LLMRoutineOutput(
            total_estimated_time=15,
            summary_title="삼두 보조만 있는 루틴",
            rationale_summary=[],
            warnings=[],
            exercises=[LLMExercisePlan(
                exercise_id="pushdown",
                exercise_name="케이블 푸시다운",
                movement_type="ISOLATION",
                primary_muscles=["triceps"],
                sets=2,
                target_reps=12,
                rest_time_sec=75,
                exercise_rationale="삼두 보조 운동",
            )],
        )
        diagnostics = {}

        result = _deterministic_or_fallback(
            output,
            request,
            candidates,
            18,
            {},
            "hypertrophy",
            request.target_muscles,
            [],
            None,
            "TEST OUTPUT",
            "TEST DRAFT",
            selection_diagnostics=diagnostics,
        )

        assert result.generation_status == "failed"
        assert result.status_reason_code == "emptyCandidate"
        assert result.is_fallback is False
        assert result.routine_blocks == []
        assert result.total_estimated_time == 0
        assert diagnostics["required_main_pattern_missing"] is True

    def test_accessory_only_fallback_is_failed_instead_of_exposed_as_routine(self):
        request = RoutineRequest(
            user_id="accessory-only-fallback",
            target_split_label="push",
            readiness_level="normal",
            time_available_min=45,
            pain_areas=[PainAreaEntry(body_part="front-deltoids")],
            doms_data={},
        )
        candidates = [{
            "id": "pushdown",
            "name_kr": "케이블 푸시다운",
            "primary_muscle": "triceps",
            "movement_type": "ISOLATION",
            "equipment_req": "CABLE",
            "score": 100,
        }]

        result = _fallback(
            request,
            candidates,
            18,
            {},
            "hypertrophy",
            "networkError",
            [],
            None,
        )

        assert result.generation_status == "failed"
        assert result.status_reason_code == "emptyCandidate"
        assert result.routine_blocks == []

    def test_time_budget_fallback_clears_discarded_selection_diagnostics(self):
        request = RoutineRequest(
            user_id="fallback-diagnostics",
            target_split_label="push",
            target_muscles=["chest", "triceps"],
            readiness_level="normal",
            time_available_min=60,
            pain_areas=[],
            doms_data={},
        )
        candidates = [{
            "id": "bench",
            "name_kr": "벤치프레스",
            "primary_muscle": "chest",
            "movement_type": "COMPOUND",
            "equipment_req": "BARBELL",
            "score": 100,
        }]
        output = LLMRoutineOutput(
            total_estimated_time=30,
            summary_title="푸시 루틴",
            rationale_summary=[],
            warnings=[],
            exercises=[LLMExercisePlan(
                exercise_id="bench",
                exercise_name="벤치프레스",
                movement_type="COMPOUND",
                primary_muscles=["chest"],
                target_reps=10,
                sets=3,
                rest_time_sec=120,
                exercise_rationale="가슴 메인 운동입니다.",
            )],
        )
        diagnostics = {"productive_fill_requested": True, "productive_fill_added_count": 1}

        with (
            patch("engines.routine_pipeline.estimate_routine_time_min", return_value=999),
            patch("engines.routine_pipeline.trim_routine_to_time_budget", return_value=False),
        ):
            result = _deterministic_or_fallback(
                output,
                request,
                candidates,
                18,
                {},
                "hypertrophy",
                request.target_muscles,
                [],
                None,
                "TEST OUTPUT",
                "TEST DRAFT",
                selection_diagnostics=diagnostics,
            )

        assert result.is_fallback is True
        assert diagnostics == {}

    def test_superset_execution_metadata_survives_success_response(self):
        request = RoutineRequest(
            user_id="execution-success",
            target_split_label="chest_back",
            readiness_level="normal",
            experience_level="intermediate",
            time_available_min=60,
            session_structure="superset",
            pain_areas=[],
            doms_data={},
            equipment=[],
        )
        candidates = [
            {"id": "bench", "name_kr": "벤치프레스", "primary_muscle": "chest", "movement_type": "COMPOUND", "equipment_req": "BARBELL"},
            {"id": "row", "name_kr": "바벨 로우", "primary_muscle": "upper-back", "movement_type": "COMPOUND", "equipment_req": "BARBELL"},
            {"id": "pushdown", "name_kr": "케이블 푸시다운", "primary_muscle": "triceps", "movement_type": "ISOLATION", "equipment_req": "CABLE"},
            {"id": "curl", "name_kr": "덤벨 컬", "primary_muscle": "biceps", "movement_type": "ISOLATION", "equipment_req": "DUMBBELL"},
        ]
        output = LLMRoutineOutput(
            total_estimated_time=50,
            summary_title="가슴과 등 슈퍼세트",
            rationale_summary=[],
            warnings=[],
            exercises=[
                LLMExercisePlan(exercise_id=item["id"], exercise_name=item["name_kr"], movement_type=item["movement_type"], primary_muscles=[item["primary_muscle"]], sets=3, target_reps=10, rest_time_sec=90, exercise_rationale="검증 운동")
                for item in candidates
            ],
        )

        result = _deterministic_or_fallback(
            output,
            request,
            candidates,
            18,
            {},
            "hypertrophy",
            ["chest", "upper-back"],
            [],
            None,
            "TEST OUTPUT",
            "TEST DRAFT",
        )

        bench = next(block for block in result.routine_blocks if block.exercise_id == "bench")
        row = next(block for block in result.routine_blocks if block.exercise_id == "row")
        assert bench.execution_group_type == "superset"
        assert bench.execution_group_id == row.execution_group_id
        assert {bench.execution_group_position, row.execution_group_position} == {1, 2}
        assert any(block.execution_group_type == "superset" for block in result.routine_blocks)

    def test_superset_execution_repairs_incompatible_model_selection_before_prescription(self):
        request = RoutineRequest(
            user_id="execution-repair",
            target_split_label="chest_back",
            readiness_level="normal",
            experience_level="intermediate",
            time_available_min=60,
            session_structure="superset",
            pain_areas=[],
            doms_data={},
            equipment=[],
        )
        candidates = [
            {"id": "machine_press", "name_kr": "머신 체스트 프레스", "primary_muscle": "chest", "movement_type": "COMPOUND", "equipment_req": "MACHINE", "score": 130},
            {"id": "machine_row", "name_kr": "머신 시티드 로우", "primary_muscle": "upper-back", "movement_type": "COMPOUND", "equipment_req": "MACHINE", "score": 125},
            {"id": "dumbbell_press", "name_kr": "덤벨 체스트 프레스", "primary_muscle": "chest", "movement_type": "COMPOUND", "equipment_req": "DUMBBELL", "score": 120},
            {"id": "dumbbell_row", "name_kr": "덤벨 로우", "primary_muscle": "upper-back", "movement_type": "COMPOUND", "equipment_req": "DUMBBELL", "score": 115},
        ]
        output = LLMRoutineOutput(
            total_estimated_time=45,
            summary_title="가슴과 등 슈퍼세트",
            rationale_summary=[],
            warnings=[],
            exercises=[
                LLMExercisePlan(exercise_id="machine_press", exercise_name="머신 체스트 프레스", movement_type="COMPOUND", primary_muscles=["chest"], equipment_type="MACHINE", sets=3, target_reps=10, rest_time_sec=90, exercise_rationale="가슴 메인"),
                LLMExercisePlan(exercise_id="dumbbell_row", exercise_name="덤벨 로우", movement_type="COMPOUND", primary_muscles=["upper-back"], equipment_type="DUMBBELL", sets=3, target_reps=10, rest_time_sec=90, exercise_rationale="등 메인"),
            ],
        )
        output._selection_phase_by_exercise_id = {
            "machine_press": "main_compound",
            "dumbbell_row": "secondary_compound",
        }

        result = _deterministic_or_fallback(
            output,
            request,
            candidates,
            18,
            {},
            "hypertrophy",
            ["chest", "upper-back"],
            [],
            None,
            "TEST OUTPUT",
            "TEST DRAFT",
        )

        press = next(block for block in result.routine_blocks if block.exercise_id == "dumbbell_press")
        row = next(block for block in result.routine_blocks if block.exercise_id == "dumbbell_row")
        assert press.execution_group_type == "superset"
        assert press.execution_group_id == row.execution_group_id

    def test_superset_execution_fallback_promotes_compatible_pair(self):
        request = RoutineRequest(
            user_id="execution-fallback-repair",
            target_split_label="chest_back",
            readiness_level="normal",
            experience_level="intermediate",
            time_available_min=60,
            session_structure="superset",
            pain_areas=[],
            doms_data={},
            equipment=[],
        )
        candidates = [
            {"id": "machine_press", "name_kr": "머신 체스트 프레스", "primary_muscle": "chest", "movement_type": "COMPOUND", "equipment_req": "MACHINE", "score": 130},
            {"id": "machine_row", "name_kr": "머신 시티드 로우", "primary_muscle": "upper-back", "movement_type": "COMPOUND", "equipment_req": "MACHINE", "score": 125},
            {"id": "dumbbell_press", "name_kr": "덤벨 체스트 프레스", "primary_muscle": "chest", "movement_type": "COMPOUND", "equipment_req": "DUMBBELL", "score": 120},
            {"id": "dumbbell_row", "name_kr": "덤벨 로우", "primary_muscle": "upper-back", "movement_type": "COMPOUND", "equipment_req": "DUMBBELL", "score": 115},
        ]

        result = _fallback(
            request,
            candidates,
            18,
            {},
            "hypertrophy",
            "networkError",
            [],
            None,
        )

        press = next(block for block in result.routine_blocks if block.exercise_id == "dumbbell_press")
        row = next(block for block in result.routine_blocks if block.exercise_id == "dumbbell_row")
        assert press.execution_group_type == "superset"
        assert press.execution_group_id == row.execution_group_id

    def test_density_execution_metadata_survives_fallback_response(self):
        request = RoutineRequest(
            user_id="execution-fallback",
            target_split_label="push",
            readiness_level="normal",
            experience_level="intermediate",
            time_available_min=60,
            session_structure="density",
            pain_areas=[],
            doms_data={},
            equipment=[],
        )
        candidates = [
            {"id": "bench", "name_kr": "벤치프레스", "primary_muscle": "chest", "movement_type": "COMPOUND", "equipment_req": "BARBELL", "score": 100},
            {"id": "machine_press", "name_kr": "머신 체스트 프레스", "primary_muscle": "chest", "movement_type": "COMPOUND", "equipment_req": "MACHINE", "score": 90},
            {"id": "fly", "name_kr": "덤벨 플라이", "primary_muscle": "chest", "movement_type": "ISOLATION", "equipment_req": "DUMBBELL", "score": 80},
            {"id": "pushdown", "name_kr": "밴드 푸시다운", "primary_muscle": "triceps", "movement_type": "ISOLATION", "equipment_req": "BAND", "score": 70},
        ]

        result = _fallback(
            request,
            candidates,
            18,
            {},
            "hypertrophy",
            "networkError",
            [],
            None,
        )

        assert result.generation_status == "fallback"
        assert result.routine_blocks[0].execution_group_type == "single"
        grouped = [block for block in result.routine_blocks if block.execution_group_type == "density"]
        assert len(grouped) >= 2
        assert all(len(block.prescription) <= 2 for block in grouped)

    def test_success_returns_correct_status(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        db = _make_db_mock()
        mock_llm = _make_llm_mock(return_value=sample_llm_output)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                sample_request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.generation_status == "success"
        assert result.is_fallback is False
        assert result.status_reason_code == "none"
        assert result.routine_blocks
        assert "barbell_bench_press" in {
            block.exercise_id for block in result.routine_blocks
        }

    def test_professional_clearance_warning_is_deterministic_on_success(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        db = _make_db_mock()
        request = sample_request.model_copy(update={
            "condition_policies": [
                {
                    "policyType": "surgeryHistory",
                    "bodyPart": "shoulder",
                    "status": "needs_clearance",
                    "professionalClearance": False,
                }
            ]
        })
        llm_output = sample_llm_output.model_copy(update={"warnings": []})
        mock_llm = _make_llm_mock(return_value=llm_output)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.generation_status == "success"
        assert any("의료진 허가" in warning for warning in result.warnings)

    def test_doms_only_wording_is_not_returned_as_pain_or_injury(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        db = _make_db_mock()
        request = sample_request.model_copy(update={"doms_data": {"chest": 2}, "pain_areas": []})
        llm_output = sample_llm_output.model_copy(deep=True)
        llm_output.rationale_summary = ["가슴 부상 우려가 있어 부담을 낮췄습니다."]
        llm_output.exercises[0].exercise_rationale = "가슴 통증과 부상 우려를 고려해 선택했습니다."
        llm_output.warnings = ["가슴 통증이 있으면 중단하세요."]
        mock_llm = _make_llm_mock(return_value=llm_output)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(request, db, profile=sample_profile, recent_sets=[])

        user_text = " ".join([
            result.summary_title,
            *result.rationale_summary,
            *result.warnings,
            *(block.exercise_rationale for block in result.routine_blocks),
        ])
        assert "부상" not in user_text
        assert "통증" not in user_text
        assert "근육통" in user_text or "피로도" in user_text

    def test_professional_clearance_warning_is_deterministic_on_fallback(
        self, sample_request, sample_profile, mock_candidates
    ):
        db = _make_db_mock()
        request = sample_request.model_copy(update={
            "condition_policies": [
                {
                    "policyType": "surgeryHistory",
                    "bodyPart": "shoulder",
                    "status": "needs_clearance",
                    "professionalClearance": False,
                }
            ]
        })
        mock_llm = _make_llm_mock(side_effect=asyncio.TimeoutError())

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.generation_status == "fallback"
        assert any("의료진 허가" in warning for warning in result.warnings)

    def test_low_readiness_extreme_pain_context_does_not_force_normal_routine(self, sample_profile, mock_candidates):
        db = _make_db_mock()
        request = RoutineRequest(
            user_id="extreme-pain-user",
            target_split_label="full_body",
            readiness_level="low",
            time_available_min=45,
            pain_areas=[
                PainAreaEntry(body_part="lower-back"),
                PainAreaEntry(body_part="knees"),
                PainAreaEntry(body_part="front-deltoids"),
                PainAreaEntry(body_part="chest"),
                PainAreaEntry(body_part="upper-back"),
                PainAreaEntry(body_part="quadriceps"),
                PainAreaEntry(body_part="hamstring"),
            ],
            doms_data={},
            equipment=[],
            goal="hypertrophy",
            adjunct_options=["mobility_warmup"],
        )

        with (
            patch("engines.routine_pipeline.get_llm") as get_llm_mock,
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                request, db, profile=sample_profile, recent_sets=[]
            )

        get_llm_mock.assert_not_called()
        assert result.generation_status == "failed"
        assert result.status_reason_code == "emptyCandidate"
        assert result.routine_blocks == []
        assert len(result.adjunct_decisions) == 1
        assert result.adjunct_decisions[0].option == "mobility_warmup"
        assert result.adjunct_decisions[0].status == "suppressed"
        assert result.adjunct_decisions[0].reason_code == "noEligibleCandidate"
        assert result.adjunct_decisions[0].slot_id == "adjunct_mobility_warmup"

    def test_success_routine_blocks_have_prescriptions(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        db = _make_db_mock()
        mock_llm = _make_llm_mock(return_value=sample_llm_output)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                sample_request, db, profile=sample_profile, recent_sets=[]
            )

        block = result.routine_blocks[0]
        # Fixture has mild chest DOMS: intermediate compound capacity 4 -> 3.
        assert len(block.prescription) == 3
        assert block.prescription[0].set_index == 1

    def test_success_underfilled_long_session_adds_safe_ranked_candidates(
        self, sample_request, sample_profile, mock_candidates
    ):
        db = _make_db_mock()
        request = sample_request.model_copy(update={
            "time_available_min": 50,
            "readiness_level": "normal",
            "doms_data": {},
        })
        ranked_candidates = mock_candidates + [
            {
                "id": "cable_row",
                "name_kr": "케이블 로우",
                "name_en": "Cable Row",
                "primary_muscle": "upper-back",
                "secondary_muscle": "biceps",
                "equipment_req": "CABLE",
                "efficiency_tier": 1,
                "pain_triggers": None,
                "movement_type": "COMPOUND",
            },
            {
                "id": "cable_triceps_pushdown",
                "name_kr": "케이블 트라이셉스 푸쉬다운",
                "name_en": "Cable Triceps Pushdown",
                "primary_muscle": "triceps",
                "secondary_muscle": None,
                "equipment_req": "CABLE",
                "efficiency_tier": 3,
                "pain_triggers": None,
                "movement_type": "ISOLATION",
            },
            {
                "id": "machine_chest_press",
                "name_kr": "머신 체스트 프레스",
                "name_en": "Machine Chest Press",
                "primary_muscle": "chest",
                "secondary_muscle": "triceps",
                "equipment_req": "MACHINE",
                "efficiency_tier": 3,
                "pain_triggers": None,
                "movement_type": "COMPOUND",
            },
            {
                "id": "dumbbell_lateral_raise",
                "name_kr": "덤벨 레터럴 레이즈",
                "name_en": "Dumbbell Lateral Raise",
                "primary_muscle": "front-deltoids",
                "secondary_muscle": None,
                "equipment_req": "DUMBBELL",
                "efficiency_tier": 3,
                "pain_triggers": None,
                "movement_type": "ISOLATION",
            },
            {
                "id": "machine_shoulder_press",
                "name_kr": "머신 숄더 프레스",
                "name_en": "Machine Shoulder Press",
                "primary_muscle": "front-deltoids",
                "secondary_muscle": "triceps",
                "equipment_req": "MACHINE",
                "efficiency_tier": 3,
                "pain_triggers": None,
                "movement_type": "COMPOUND",
            },
        ]
        underfilled_output = LLMRoutineOutput(
            total_estimated_time=30,
            summary_title="짧게 나온 푸시 루틴",
            rationale_summary=["LLM이 핵심 운동만 골랐습니다."],
            warnings=[],
            exercises=[
                LLMExercisePlan(
                    exercise_id="barbell_bench_press",
                    exercise_name="바벨 벤치프레스",
                    primary_muscles=["chest"],
                    equipment_type="BARBELL",
                    target_reps=10,
                    sets=3,
                    rest_time_sec=90,
                    exercise_rationale="가슴 메인 운동입니다.",
                ),
                LLMExercisePlan(
                    exercise_id="dumbbell_fly",
                    exercise_name="덤벨 플라이",
                    primary_muscles=["chest"],
                    equipment_type="DUMBBELL",
                    target_reps=12,
                    sets=2,
                    rest_time_sec=75,
                    exercise_rationale="가슴 보조 운동입니다.",
                ),
                LLMExercisePlan(
                    exercise_id="pushup",
                    exercise_name="푸시업",
                    primary_muscles=["chest"],
                    equipment_type="BODYWEIGHT",
                    target_reps=12,
                    sets=2,
                    rest_time_sec=75,
                    exercise_rationale="맨몸 보조 운동입니다.",
                ),
            ],
        )
        mock_llm = _make_llm_mock(return_value=underfilled_output)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=ranked_candidates),
        ):
            result = generate_smart_routine(request, db, profile=sample_profile, recent_sets=[])

        assert result.generation_status == "success"
        assert len(result.routine_blocks) == 5
        assert "cable_triceps_pushdown" in {block.exercise_id for block in result.routine_blocks}
        assert "cable_row" not in {block.exercise_id for block in result.routine_blocks}
        assert any("루틴 구성을 균형 있게" in warning for warning in result.warnings)

    def test_success_routine_reorders_compound_large_muscle_before_isolation(
        self, sample_request, sample_profile, mock_candidates
    ):
        db = _make_db_mock()
        output = LLMRoutineOutput(
            total_estimated_time=50,
            summary_title="Reordered push",
            rationale_summary=["case"],
            warnings=[],
            exercises=[
                LLMExercisePlan(
                    exercise_id="dumbbell_fly",
                    exercise_name="Dumbbell Fly",
                    movement_type="ISOLATION",
                    primary_muscles=["chest"],
                    equipment_type="DUMBBELL",
                    target_reps=12,
                    sets=3,
                    rest_time_sec=75,
                    exercise_rationale="accessory",
                ),
                LLMExercisePlan(
                    exercise_id="barbell_bench_press",
                    exercise_name="Barbell Bench Press",
                    movement_type="COMPOUND",
                    primary_muscles=["chest"],
                    equipment_type="BARBELL",
                    target_reps=8,
                    sets=3,
                    rest_time_sec=120,
                    exercise_rationale="main",
                ),
            ],
        )
        mock_llm = _make_llm_mock(return_value=output)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                sample_request.model_copy(update={"doms_data": {}, "time_available_min": 40}),
                db,
                profile=sample_profile,
                recent_sets=[],
            )

        assert result.routine_blocks[0].exercise_id == "barbell_bench_press"
        assert result.routine_blocks[0].order == 1
        assert result.routine_blocks[1].exercise_id == "dumbbell_fly"
        assert result.routine_blocks[1].order == 2

    def test_invalid_candidate_is_repaired_to_safe_candidate(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        db = _make_db_mock()
        hallucinated = sample_llm_output.model_copy(deep=True)
        hallucinated.exercises[0].exercise_id = "ghost_press"
        mock_llm = _make_llm_mock(return_value=hallucinated)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                sample_request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.generation_status == "success"
        assert result.routine_blocks[0].exercise_id == "barbell_bench_press"
        assert any("대신" in warning for warning in result.warnings)

    def test_too_many_invalid_candidates_fall_back(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        db = _make_db_mock()
        bad = sample_llm_output.model_copy(deep=True)
        bad.exercises = [
            bad.exercises[0].model_copy(update={"exercise_id": f"ghost_{i}"})
            for i in range(5)
        ]
        mock_llm = _make_llm_mock(return_value=bad)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                sample_request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.generation_status == "fallback"
        assert result.is_fallback is True

    def test_recent_sets_override_llm_weight_and_reps(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        sample_request = sample_request.model_copy(update={"doms_data": {}})
        db = _make_db_mock()
        mock_llm = _make_llm_mock(return_value=sample_llm_output)
        recent_sets = [
            RecentSetRecord(
                exercise_name=sample_llm_output.exercises[0].exercise_name,
                weight_kg=100,
                reps=5,
            )
        ]

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                sample_request, db, profile=sample_profile, recent_sets=recent_sets
            )

        first_set = result.routine_blocks[0].prescription[0]
        assert first_set.target_reps == 8
        assert first_set.target_weight_kg is None
        assert first_set.target_rest_sec == 120

    def test_recent_sets_match_by_exercise_id_before_name(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        sample_request = sample_request.model_copy(update={"doms_data": {}})
        db = _make_db_mock()
        mock_llm = _make_llm_mock(return_value=sample_llm_output)
        recent_sets = [
            RecentSetRecord(
                exercise_id=sample_llm_output.exercises[0].exercise_id,
                exercise_name="Unrelated Snapshot",
                weight_kg=100,
                reps=5,
            )
        ]

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                sample_request, db, profile=sample_profile, recent_sets=recent_sets
            )

        assert result.routine_blocks[0].prescription[0].target_weight_kg is None

    def test_fat_loss_goal_uses_compound_conditioning_reps(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        db = _make_db_mock()
        mock_llm = _make_llm_mock(return_value=sample_llm_output)
        request = sample_request.model_copy(update={"goal": "fatLoss", "doms_data": {}})

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.routine_blocks[0].prescription[0].target_reps == 12

    def test_low_readiness_reduces_compound_sets_and_raises_rir(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        db = _make_db_mock()
        request = sample_request.model_copy(update={"readiness_level": "low", "doms_data": {}})
        output = sample_llm_output.model_copy(deep=True)
        output.exercises[0].sets = 4
        output.exercises[0].target_rir = 2
        mock_llm = _make_llm_mock(return_value=output)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                request, db, profile=sample_profile, recent_sets=[]
            )

        assert len(result.routine_blocks[0].prescription) == 3
        assert result.routine_blocks[0].prescription[0].target_rir == 4

    @pytest.mark.parametrize("base_rir", [0, 1, 2])
    def test_low_readiness_raises_isolation_rir_without_forcing_set_reduction(
        self,
        base_rir,
    ):
        exercise = LLMExercisePlan(
            exercise_id="lateral_raise",
            exercise_name="덤벨 레터럴 레이즈",
            movement_type="ISOLATION",
            primary_muscles=["side-deltoids"],
            target_reps=12,
            sets=3,
            rest_time_sec=60,
            target_rir=base_rir,
            exercise_rationale="어깨 보조 운동입니다.",
        )

        _apply_readiness_to_exercise(exercise, "low")

        assert exercise.sets == 3
        assert exercise.target_rir == 4

    def test_high_readiness_does_not_lower_rir_below_evidence_floor(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        db = _make_db_mock()
        request = sample_request.model_copy(update={"readiness_level": "high", "doms_data": {}})
        output = sample_llm_output.model_copy(deep=True)
        output.exercises[0].sets = 4
        output.exercises[0].target_rir = 2
        mock_llm = _make_llm_mock(return_value=output)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                request, db, profile=sample_profile, recent_sets=[]
            )

        assert len(result.routine_blocks[0].prescription) == 4
        assert result.routine_blocks[0].prescription[0].target_rir == 2

    def test_shoulder_injury_does_not_repair_press_to_pushup(
        self, sample_request, sample_profile
    ):
        db = _make_db_mock()
        request = sample_request.model_copy(update={
            "pain_areas": [PainAreaEntry(body_part="shoulder", side="left")],
            "doms_data": {},
        })
        candidates = [
            {
                "id": "painful_press",
                "name_kr": "Painful Press",
                "name_en": "Painful Press",
                "primary_muscle": "chest",
                "secondary_muscle": "front-deltoids",
                "equipment_req": "DUMBBELL",
                "difficulty_tier": 3,
                "efficiency_tier": 5,
                "pain_triggers": "shoulder",
                "movement_type": "COMPOUND",
            },
            {
                "id": "safe_pushup",
                "name_kr": "Safe Push-up",
                "name_en": "Safe Push-up",
                "primary_muscle": "chest",
                "secondary_muscle": "triceps",
                "equipment_req": "BODYWEIGHT",
                "difficulty_tier": 1,
                "efficiency_tier": 3,
                "pain_triggers": None,
                "movement_type": "COMPOUND",
            },
        ]
        output = LLMRoutineOutput(
            total_estimated_time=30,
            summary_title="Pain repair",
            rationale_summary=["case"],
            warnings=[],
            exercises=[
                LLMExercisePlan(
                    exercise_id="painful_press",
                    exercise_name="Painful Press",
                    movement_type="COMPOUND",
                    primary_muscles=["chest"],
                    equipment_type="DUMBBELL",
                    target_reps=8,
                    sets=3,
                    rest_time_sec=120,
                    target_rir=2,
                    exercise_rationale="case",
                ),
            ],
        )
        mock_llm = _make_llm_mock(return_value=output)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=candidates),
        ):
            result = generate_smart_routine(
                request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.generation_status == "failed"
        assert result.status_reason_code == "emptyCandidate"
        assert result.routine_blocks == []

    def test_big_four_strength_baseline_seeds_related_target_weight(
        self, sample_request, sample_profile
    ):
        db = _make_db_mock()
        request = sample_request.model_copy(update={
            "target_split_label": "legs",
            "target_muscles": ["quadriceps"],
            "doms_data": {},
            "goal": "hypertrophy",
        })
        profile = sample_profile.model_copy(update={
            "strength_baseline": {
                "98": {
                    "exercise_id": "98",
                    "exercise_name_snapshot": "Back Squat",
                    "weight_kg": 170,
                    "reps": 1,
                }
            }
        })
        candidates = [
            {
                "id": "leg_press",
                "name_kr": "Leg Press",
                "name_en": "Leg Press",
                "primary_muscle": "quadriceps",
                "secondary_muscle": "gluteal",
                "equipment_req": "MACHINE",
                "difficulty_tier": 2,
                "efficiency_tier": 4,
                "pain_triggers": None,
                "movement_type": "COMPOUND",
            }
        ]
        output = LLMRoutineOutput(
            total_estimated_time=40,
            summary_title="Legs",
            rationale_summary=["case"],
            warnings=[],
            exercises=[
                LLMExercisePlan(
                    exercise_id="leg_press",
                    exercise_name="Leg Press",
                    movement_type="COMPOUND",
                    primary_muscles=["quadriceps"],
                    equipment_type="MACHINE",
                    target_weight_kg=None,
                    target_reps=8,
                    sets=3,
                    rest_time_sec=120,
                    target_rir=2,
                    exercise_rationale="quadriceps target",
                ),
            ],
        )
        mock_llm = _make_llm_mock(return_value=output)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=candidates),
        ):
            result = generate_smart_routine(
                request, db, profile=profile, recent_sets=[]
            )

        assert result.generation_status == "success"
        assert result.routine_blocks[0].prescription[0].target_weight_kg is None

    def test_push_split_caps_accessory_isolation_sets(
        self, sample_request, sample_profile, mock_candidates
    ):
        db = _make_db_mock()
        candidates = mock_candidates + [
            {
                "id": "cable_pushdown",
                "name_kr": "Cable Pushdown",
                "name_en": "Cable Pushdown",
                "primary_muscle": "triceps",
                "secondary_muscle": None,
                "equipment_req": "CABLE",
                "difficulty_tier": 2,
                "efficiency_tier": 4,
                "pain_triggers": "triceps",
                "movement_type": "ISOLATION",
            }
        ]
        output = LLMRoutineOutput(
            total_estimated_time=50,
            summary_title="Push accessory cap",
            rationale_summary=["case"],
            warnings=[],
            exercises=[
                LLMExercisePlan(
                    exercise_id="barbell_bench_press",
                    exercise_name="Barbell Bench Press",
                    movement_type="COMPOUND",
                    primary_muscles=["chest"],
                    equipment_type="BARBELL",
                    target_reps=8,
                    sets=3,
                    rest_time_sec=90,
                    target_rir=2,
                    exercise_rationale="main",
                ),
                LLMExercisePlan(
                    exercise_id="cable_pushdown",
                    exercise_name="Cable Pushdown",
                    movement_type="ISOLATION",
                    primary_muscles=["triceps"],
                    equipment_type="CABLE",
                    target_reps=10,
                    sets=4,
                    rest_time_sec=90,
                    target_rir=2,
                    exercise_rationale="accessory",
                ),
            ],
        )
        mock_llm = _make_llm_mock(return_value=output)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=candidates),
        ):
            result = generate_smart_routine(
                sample_request.model_copy(update={"doms_data": {}}),
                db,
                profile=sample_profile,
                recent_sets=[],
            )

        pushdown = next(block for block in result.routine_blocks if block.exercise_id == "cable_pushdown")
        assert len(pushdown.prescription) == 2

    def test_arm_split_allows_accessory_muscle_as_main_volume(
        self, sample_request, sample_profile
    ):
        db = _make_db_mock()
        request = sample_request.model_copy(update={"target_split_label": "arm", "doms_data": {}})
        candidates = [
            {
                "id": "cable_pushdown",
                "name_kr": "Cable Pushdown",
                "name_en": "Cable Pushdown",
                "primary_muscle": "triceps",
                "secondary_muscle": None,
                "equipment_req": "CABLE",
                "difficulty_tier": 2,
                "efficiency_tier": 4,
                "pain_triggers": "triceps",
                "movement_type": "ISOLATION",
            }
        ]
        output = LLMRoutineOutput(
            total_estimated_time=30,
            summary_title="Arm main",
            rationale_summary=["case"],
            warnings=[],
            exercises=[
                LLMExercisePlan(
                    exercise_id="cable_pushdown",
                    exercise_name="Cable Pushdown",
                    movement_type="ISOLATION",
                    primary_muscles=["triceps"],
                    equipment_type="CABLE",
                    target_reps=10,
                    sets=4,
                    rest_time_sec=90,
                    target_rir=2,
                    exercise_rationale="main arm work",
                ),
            ],
        )
        mock_llm = _make_llm_mock(return_value=output)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=candidates),
        ):
            result = generate_smart_routine(
                request, db, profile=sample_profile, recent_sets=[]
            )

        assert len(result.routine_blocks[0].prescription) == 4

    def test_moderate_doms_clamps_sets(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        db = _make_db_mock()
        request = sample_request.model_copy(update={"doms_data": {"chest": 2}})
        mock_llm = _make_llm_mock(return_value=sample_llm_output)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.generation_status == "success"
        assert len(result.routine_blocks[0].prescription) == 3

    def test_severe_doms_forces_fallback(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        db = _make_db_mock()
        request = sample_request.model_copy(update={"doms_data": {"chest": 3}})
        mock_llm = _make_llm_mock(return_value=sample_llm_output)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.generation_status == "failed"
        assert result.status_reason_code == "emptyCandidate"

    def test_over_budget_output_is_trimmed_before_fallback(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        db = _make_db_mock()
        bad = sample_llm_output.model_copy(deep=True)
        bad.exercises[0].sets = 20
        bad.exercises[0].rest_time_sec = 300
        mock_llm = _make_llm_mock(return_value=bad)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                sample_request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.generation_status == "success"
        assert result.total_estimated_time <= sample_request.time_available_min
        assert len(result.routine_blocks[0].prescription) < 20

    def test_short_time_realistic_time_model_trims_before_fallback(
        self, sample_request, sample_profile, mock_candidates
    ):
        db = _make_db_mock()
        request = sample_request.model_copy(update={"time_available_min": 30, "doms_data": {}})
        output = LLMRoutineOutput(
            total_estimated_time=30,
            summary_title="Short push",
            rationale_summary=["case"],
            warnings=[],
            exercises=[
                LLMExercisePlan(
                    exercise_id="barbell_bench_press",
                    exercise_name="Barbell Bench Press",
                    movement_type="COMPOUND",
                    primary_muscles=["chest"],
                    equipment_type="BARBELL",
                    target_reps=8,
                    sets=4,
                    rest_time_sec=120,
                    target_rir=2,
                    exercise_rationale="case",
                ),
                LLMExercisePlan(
                    exercise_id="pushup",
                    exercise_name="Push-up",
                    movement_type="COMPOUND",
                    primary_muscles=["chest"],
                    equipment_type="BODYWEIGHT",
                    target_reps=8,
                    sets=3,
                    rest_time_sec=120,
                    target_rir=2,
                    exercise_rationale="case",
                ),
                LLMExercisePlan(
                    exercise_id="dumbbell_fly",
                    exercise_name="Dumbbell Fly",
                    movement_type="ISOLATION",
                    primary_muscles=["chest"],
                    equipment_type="DUMBBELL",
                    target_reps=8,
                    sets=3,
                    rest_time_sec=120,
                    target_rir=2,
                    exercise_rationale="case",
                ),
            ],
        )
        mock_llm = _make_llm_mock(return_value=output)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.generation_status == "success"
        assert result.total_estimated_time <= request.time_available_min


class TestTimeoutFallback:
    def test_timeout_triggers_fallback(
        self, sample_request, sample_profile, mock_candidates
    ):
        db = _make_db_mock()
        mock_llm = _make_llm_mock(side_effect=httpx.TimeoutException("timeout"))

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                sample_request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.is_fallback is True
        assert result.generation_status == "fallback"
        assert result.status_reason_code == "llmTimeout"

    def test_asyncio_timeout_triggers_fallback(
        self, sample_request, sample_profile, mock_candidates
    ):
        db = _make_db_mock()
        mock_llm = _make_llm_mock(side_effect=asyncio.TimeoutError())

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                sample_request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.status_reason_code == "llmTimeout"


class TestSchemaErrorRecovery:
    def test_schema_error_with_valid_raw_text_recovers(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        """OutputParserException + llm_output 속성에 유효한 JSON → normalize 복구 성공"""
        db = _make_db_mock()
        valid_raw = sample_llm_output.model_dump_json()

        parser_exc = OutputParserException("schema fail")
        parser_exc.llm_output = valid_raw

        mock_llm = _make_llm_mock(side_effect=parser_exc)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                sample_request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.generation_status == "success"
        assert result.is_fallback is False

    def test_schema_error_unrecoverable_falls_back(
        self, sample_request, sample_profile, mock_candidates
    ):
        """OutputParserException + 복구 불가 raw 텍스트 + LLM 재호출도 실패 → fallback"""
        db = _make_db_mock()

        parser_exc = OutputParserException("schema fail")
        parser_exc.llm_output = "완전히 깨진 JSON !!!"

        mock_llm = _make_llm_mock(side_effect=parser_exc)
        # 비구조화 재호출도 실패
        mock_llm.invoke.side_effect = RuntimeError("LLM도 죽음")

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                sample_request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.is_fallback is True
        assert result.generation_status == "fallback"


class TestNoProfile:
    def test_works_without_profile(self, sample_request, sample_llm_output, mock_candidates):
        """profile=None이어도 파이프라인이 동작해야 한다"""
        db = _make_db_mock()
        mock_llm = _make_llm_mock(return_value=sample_llm_output)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                sample_request, db, profile=None, recent_sets=[]
            )

        assert result.generation_status == "success"

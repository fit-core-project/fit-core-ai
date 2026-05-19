"""
계약 회귀(Contract Regression) 테스트
- RoutineRequest: 프론트엔드가 보내는 camelCase JSON이 올바르게 파싱되는지
- RoutineDraftResponse: 응답이 camelCase로 직렬화되는지 (FE와의 약속)
"""
import pytest
from pydantic import ValidationError

from engines.routine_engine import (
    RoutineDraftResponse,
    RoutineRequest,
)


class TestRoutineRequestParsing:
    """프론트엔드 → 백엔드 입력 계약"""

    def test_camel_case_json_parsed(self):
        payload = {
            "userId": "u-123",
            "targetSplitLabel": "push",
            "timeAvailableMin": 60,
            "readinessLevel": "normal",
            "painAreas": [],
            "domsData": {},
            "equipment": [],
        }
        req = RoutineRequest.model_validate(payload)
        assert req.user_id == "u-123"
        assert req.target_split_label == "push"
        assert req.time_available_min == 60

    def test_snake_case_also_accepted(self):
        payload = {
            "user_id": "u-123",
            "target_split_label": "legs",
            "time_available_min": 45,
        }
        req = RoutineRequest.model_validate(payload)
        assert req.target_split_label == "legs"

    def test_doms_data_camel_case_parsed(self):
        payload = {
            "timeAvailableMin": 60,
            "domsData": {"chest": 1, "triceps": 2},
        }
        req = RoutineRequest.model_validate(payload)
        assert req.doms_data["chest"] == 1
        assert req.doms_data["triceps"] == 2

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            RoutineRequest.model_validate({})  # time_available_min 필수

    def test_defaults_applied(self):
        req = RoutineRequest.model_validate({"timeAvailableMin": 30})
        assert req.readiness_level == "normal"
        assert req.target_muscles == []
        assert req.pain_areas == []
        assert req.doms_data == {}
        assert req.equipment == []


class TestRoutineDraftResponseSerialization:
    """백엔드 → 프론트엔드 출력 계약"""

    def test_top_level_keys_are_camel_case(self, sample_llm_output):
        from engines.routine_engine import _build_routine_draft

        response = _build_routine_draft(sample_llm_output, "success", "none", False)
        dumped = response.model_dump(by_alias=True)

        assert "routineDraftId" in dumped
        assert "generationStatus" in dumped
        assert "statusReasonCode" in dumped
        assert "isFallback" in dumped
        assert "summaryTitle" in dumped
        assert "rationaleSum" not in dumped  # 오타 방지
        assert "rationale_summary" not in dumped  # snake_case 유출 방지
        assert "rationaleSum mary" not in dumped

    def test_routine_blocks_keys_are_camel_case(self, sample_llm_output):
        from engines.routine_engine import _build_routine_draft

        response = _build_routine_draft(sample_llm_output, "success", "none", False)
        dumped = response.model_dump(by_alias=True)

        block = dumped["routineBlocks"][0]
        assert "exerciseId" in block
        assert "exerciseName" in block
        assert "movementPattern" in block
        assert "primaryMuscles" in block
        assert "equipmentType" in block
        assert "defaultRestSec" in block
        assert "exerciseRationale" in block
        assert "substitutionCandidates" in block

    def test_prescription_keys_are_camel_case(self, sample_llm_output):
        from engines.routine_engine import _build_routine_draft

        response = _build_routine_draft(sample_llm_output, "success", "none", False)
        dumped = response.model_dump(by_alias=True)

        prescription = dumped["routineBlocks"][0]["prescription"][0]
        assert "setIndex" in prescription
        assert "setType" in prescription
        assert "targetReps" in prescription
        assert "targetWeightKg" in prescription
        assert "targetRir" in prescription
        assert "targetRestSec" in prescription

    def test_is_fallback_false_on_success(self, sample_llm_output):
        from engines.routine_engine import _build_routine_draft

        response = _build_routine_draft(sample_llm_output, "success", "none", False)
        assert response.is_fallback is False

    def test_is_fallback_true_on_fallback(self, sample_llm_output):
        from engines.routine_engine import _build_routine_draft

        response = _build_routine_draft(sample_llm_output, "fallback", "llmTimeout", True)
        assert response.is_fallback is True
        assert response.status_reason_code == "llmTimeout"

    def test_routine_draft_id_is_uuid_string(self, sample_llm_output):
        import re
        from engines.routine_engine import _build_routine_draft

        response = _build_routine_draft(sample_llm_output, "success", "none", False)
        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )
        assert uuid_pattern.match(response.routine_draft_id)

    def test_total_estimated_time_in_response(self, sample_llm_output):
        from engines.routine_engine import _build_routine_draft, estimate_routine_time_min

        response = _build_routine_draft(sample_llm_output, "success", "none", False)
        dumped = response.model_dump(by_alias=True)

        assert "totalEstimatedTime" in dumped
        assert dumped["totalEstimatedTime"] == estimate_routine_time_min(sample_llm_output.exercises)

    def test_is_fallback_serialized_as_camel_case(self, sample_llm_output):
        from engines.routine_engine import _build_routine_draft

        response = _build_routine_draft(sample_llm_output, "success", "none", False)
        dumped = response.model_dump(by_alias=True)

        assert "isFallback" in dumped
        assert "is_fallback" not in dumped


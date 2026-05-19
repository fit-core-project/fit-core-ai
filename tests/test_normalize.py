"""normalize_llm_response 어댑터 — JSON 파싱·복구·기본값 주입 테스트"""
import json

import pytest

from engines.routine_engine import LLMRoutineOutput, normalize_llm_response


def _minimal_valid_json(extra: dict | None = None) -> str:
    base = {
        "total_estimated_time": 45,
        "summary_title": "테스트 루틴",
        "rationale_summary": ["근거1"],
        "warnings": [],
        "exercises": [
            {
                "exercise_id": "bench_press",
                "exercise_name": "벤치프레스",
                "target_reps": 8,
                "sets": 3,
                "rest_time_sec": 90,
                "exercise_rationale": "주요 복합 운동",
            }
        ],
    }
    if extra:
        base.update(extra)
    return json.dumps(base, ensure_ascii=False)


class TestMarkdownFenceStripping:
    def test_strips_json_code_fence(self):
        raw = f"```json\n{_minimal_valid_json()}\n```"
        result = normalize_llm_response(raw)
        assert isinstance(result, LLMRoutineOutput)

    def test_strips_plain_code_fence(self):
        raw = f"```\n{_minimal_valid_json()}\n```"
        result = normalize_llm_response(raw)
        assert isinstance(result, LLMRoutineOutput)

    def test_no_fence_works(self):
        result = normalize_llm_response(_minimal_valid_json())
        assert isinstance(result, LLMRoutineOutput)


class TestJsonRepair:
    def test_truncated_json_repaired(self):
        # 외부 } 하나가 잘린 경우 → _try_repair_json이 "}" 하나 붙여서 복구 가능
        truncated = '{"total_estimated_time": 45, "summary_title": "테스트", "rationale_summary": ["근거"], "warnings": [], "exercises": [{"exercise_id": "bench", "exercise_name": "벤치", "target_reps": 8, "sets": 3, "rest_time_sec": 90, "exercise_rationale": "ok"}]'
        result = normalize_llm_response(truncated)
        assert isinstance(result, LLMRoutineOutput)

    def test_completely_invalid_json_raises(self):
        with pytest.raises(Exception):
            normalize_llm_response("이것은 전혀 JSON이 아닙니다")


class TestInjectDefaults:
    def test_missing_summary_title_gets_default(self):
        data = json.loads(_minimal_valid_json())
        del data["summary_title"]
        result = normalize_llm_response(json.dumps(data))
        assert result.summary_title == "맞춤형 AI 루틴"

    def test_missing_warnings_gets_empty_list(self):
        data = json.loads(_minimal_valid_json())
        del data["warnings"]
        result = normalize_llm_response(json.dumps(data))
        assert result.warnings == []

    def test_missing_exercise_rationale_gets_default(self):
        data = json.loads(_minimal_valid_json())
        del data["exercises"][0]["exercise_rationale"]
        result = normalize_llm_response(json.dumps(data))
        assert result.exercises[0].exercise_rationale == "AI 추천 운동"

    def test_missing_exercise_id_derived_from_name(self):
        data = json.loads(_minimal_valid_json())
        del data["exercises"][0]["exercise_id"]
        data["exercises"][0]["exercise_name"] = "바벨 스쿼트"
        result = normalize_llm_response(json.dumps(data))
        assert result.exercises[0].exercise_id == "바벨_스쿼트"

    def test_missing_substitution_candidates_gets_empty_list(self):
        data = json.loads(_minimal_valid_json())
        result = normalize_llm_response(json.dumps(data))
        assert result.exercises[0].substitution_candidates == []


class TestValidOutput:
    def test_full_valid_json_parses_correctly(self, sample_llm_output):
        json_str = sample_llm_output.model_dump_json()
        result = normalize_llm_response(json_str)
        assert result.summary_title == sample_llm_output.summary_title
        assert len(result.exercises) == len(sample_llm_output.exercises)
        assert result.exercises[0].exercise_id == sample_llm_output.exercises[0].exercise_id


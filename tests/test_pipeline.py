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

from engines.routine_engine import (
    LLMExercisePlan,
    LLMRoutineOutput,
    RecentSetRecord,
    generate_smart_routine,
)


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


class TestHappyPath:
    def test_success_returns_correct_status(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        db = _make_db_mock()
        mock_llm = _make_llm_mock(return_value=sample_llm_output)

        with (
            patch("engines.routine_engine.get_llm", return_value=mock_llm),
            patch("engines.routine_engine.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                sample_request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.generation_status == "success"
        assert result.is_fallback is False
        assert result.status_reason_code == "none"
        assert len(result.routine_blocks) == 1
        assert result.routine_blocks[0].exercise_id == "barbell_bench_press"

    def test_success_routine_blocks_have_prescriptions(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        db = _make_db_mock()
        mock_llm = _make_llm_mock(return_value=sample_llm_output)

        with (
            patch("engines.routine_engine.get_llm", return_value=mock_llm),
            patch("engines.routine_engine.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                sample_request, db, profile=sample_profile, recent_sets=[]
            )

        block = result.routine_blocks[0]
        assert len(block.prescription) == sample_llm_output.exercises[0].sets
        assert block.prescription[0].set_index == 1

    def test_invalid_candidate_is_repaired_to_safe_candidate(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        db = _make_db_mock()
        hallucinated = sample_llm_output.model_copy(deep=True)
        hallucinated.exercises[0].exercise_id = "ghost_press"
        mock_llm = _make_llm_mock(return_value=hallucinated)

        with (
            patch("engines.routine_engine.get_llm", return_value=mock_llm),
            patch("engines.routine_engine.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                sample_request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.generation_status == "success"
        assert result.routine_blocks[0].exercise_id == "barbell_bench_press"
        assert any("교체" in warning for warning in result.warnings)

    def test_too_many_invalid_candidates_fall_back(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        db = _make_db_mock()
        bad = sample_llm_output.model_copy(deep=True)
        bad.exercises = [
            bad.exercises[0].model_copy(update={"exercise_id": f"ghost_{i}"})
            for i in range(3)
        ]
        mock_llm = _make_llm_mock(return_value=bad)

        with (
            patch("engines.routine_engine.get_llm", return_value=mock_llm),
            patch("engines.routine_engine.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                sample_request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.generation_status == "fallback"
        assert result.is_fallback is True

    def test_recent_sets_override_llm_weight_and_reps(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
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
            patch("engines.routine_engine.get_llm", return_value=mock_llm),
            patch("engines.routine_engine.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                sample_request, db, profile=sample_profile, recent_sets=recent_sets
            )

        first_set = result.routine_blocks[0].prescription[0]
        assert first_set.target_reps == 10
        assert first_set.target_weight_kg == 85.0


class TestTimeoutFallback:
    def test_timeout_triggers_fallback(
        self, sample_request, sample_profile, mock_candidates
    ):
        db = _make_db_mock()
        mock_llm = _make_llm_mock(side_effect=httpx.TimeoutException("timeout"))

        with (
            patch("engines.routine_engine.get_llm", return_value=mock_llm),
            patch("engines.routine_engine.get_candidate_exercises", return_value=mock_candidates),
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
            patch("engines.routine_engine.get_llm", return_value=mock_llm),
            patch("engines.routine_engine.get_candidate_exercises", return_value=mock_candidates),
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
            patch("engines.routine_engine.get_llm", return_value=mock_llm),
            patch("engines.routine_engine.get_candidate_exercises", return_value=mock_candidates),
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
            patch("engines.routine_engine.get_llm", return_value=mock_llm),
            patch("engines.routine_engine.get_candidate_exercises", return_value=mock_candidates),
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
            patch("engines.routine_engine.get_llm", return_value=mock_llm),
            patch("engines.routine_engine.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                sample_request, db, profile=None, recent_sets=[]
            )

        assert result.generation_status == "success"

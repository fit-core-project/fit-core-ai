"""map_llm_error — 순수 함수 단위 테스트"""
import asyncio

import httpx
import pytest
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from engines.llm_router import map_llm_error
from engines.routine_engine import LLMRoutineOutput


class TestMapLlmError:
    def test_asyncio_timeout(self):
        assert map_llm_error(asyncio.TimeoutError()) == "llmTimeout"

    def test_httpx_timeout(self):
        assert map_llm_error(httpx.TimeoutException("timeout")) == "llmTimeout"

    def test_httpx_read_timeout(self):
        assert map_llm_error(httpx.ReadTimeout("read timeout")) == "llmTimeout"

    def test_connect_error(self):
        assert map_llm_error(httpx.ConnectError("conn refused")) == "networkError"

    def test_connection_refused(self):
        assert map_llm_error(ConnectionRefusedError()) == "networkError"

    def test_output_parser_exception(self):
        assert map_llm_error(OutputParserException("bad parse")) == "schemaError"

    def test_pydantic_validation_error(self):
        try:
            LLMRoutineOutput(
                total_estimated_time="not_an_int",  # 잘못된 타입
                summary_title="test",
                rationale_summary=[],
                exercises=[],
            )
        except ValidationError as e:
            assert map_llm_error(e) == "schemaError"

    def test_generic_exception_falls_back_to_network_error(self):
        assert map_llm_error(RuntimeError("unexpected")) == "networkError"

    def test_timeout_keyword_in_message(self):
        assert map_llm_error(RuntimeError("deadline exceeded")) == "llmTimeout"

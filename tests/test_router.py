"""map_llm_error — 순수 함수 단위 테스트"""
import asyncio

import httpx
import pytest
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from engines.llm_router import _parse_json_content, get_llm, map_llm_error
from engines.schemas import LLMRoutineOutput


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


class TestGetLlm:
    def test_local_provider_uses_ollama_json_mode_defaults(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "local")
        monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

        llm = get_llm("routine", temperature=0.2)

        assert llm.model == "gemma4"
        assert llm.format == "json"
        assert llm.temperature == 0.2
        assert llm.base_url == "http://localhost:11434"
        assert llm.with_structured_output.__func__.__name__ == "with_structured_output"

    def test_local_provider_reads_model_and_base_url_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "local")
        monkeypatch.setenv("LOCAL_LLM_MODEL", "gemma4:latest")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

        llm = get_llm("routine")

        assert llm.model == "gemma4:latest"
        assert llm.format == "json"
        assert llm.base_url == "http://127.0.0.1:11434"

    def test_local_json_parser_accepts_markdown_fenced_json(self):
        assert _parse_json_content("```json\n{\"ok\": true}\n```") == {"ok": True}


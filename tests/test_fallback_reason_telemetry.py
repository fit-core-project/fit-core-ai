import asyncio

import httpx
import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from engines.llm_router import classify_llm_error
from engines.routine_pipeline import generate_smart_routine
from engines.routine_telemetry import (
    build_routine_quality_telemetry_payload,
    classify_fallback_reason,
)
from engines.schemas import RoutineRequest
from tests.test_routine_telemetry import _Critic, _candidate, _response


def test_classify_llm_error_timeout():
    assert classify_llm_error(asyncio.TimeoutError()) == "timeout"
    assert classify_llm_error(httpx.TimeoutException("timeout")) == "timeout"


def test_classify_llm_error_quota_http_429():
    request = httpx.Request("GET", "https://example.test")
    response = httpx.Response(429, request=request)
    exc = httpx.HTTPStatusError("redacted", request=request, response=response)
    assert classify_llm_error(exc) == "quota_exhausted"


def test_classify_llm_error_quota_string():
    assert classify_llm_error(RuntimeError("RESOURCE_EXHAUSTED quota 429")) == "quota_exhausted"


def test_classify_llm_error_schema_parse():
    assert classify_llm_error(OutputParserException("bad json")) == "schema_parse_error"


def test_classify_llm_error_validation():
    with pytest.raises(ValidationError) as exc_info:
        RoutineRequest()
    assert classify_llm_error(exc_info.value) == "validation_error"


def test_classify_llm_error_network():
    assert classify_llm_error(httpx.ConnectError("connect failed")) == "network_error"
    assert classify_llm_error(httpx.RemoteProtocolError("bad response")) == "network_error"


def test_classify_llm_error_unknown():
    assert classify_llm_error(RuntimeError("plain failure")) == "unknown"


def test_classify_llm_error_never_raises():
    class BadStr(Exception):
        def __str__(self):
            raise RuntimeError("str failed")

    assert classify_llm_error(BadStr()) == "unknown"


def test_fallback_reason_none_when_not_fallback():
    assert classify_fallback_reason(
        fallback_used=False,
        llm_error_type="quota_exhausted",
        schema_repair_attempted=True,
        schema_repair_succeeded=False,
        validation_failed=True,
        time_budget_exceeded=True,
    ) == "none"


def test_fallback_reason_timeout():
    assert classify_fallback_reason(
        fallback_used=True,
        llm_error_type="timeout",
        schema_repair_attempted=False,
        schema_repair_succeeded=False,
        validation_failed=False,
        time_budget_exceeded=False,
    ) == "timeout"


def test_fallback_reason_quota_exhausted():
    assert classify_fallback_reason(
        fallback_used=True,
        llm_error_type="quota_exhausted",
        schema_repair_attempted=False,
        schema_repair_succeeded=False,
        validation_failed=False,
        time_budget_exceeded=False,
    ) == "quota_exhausted"


def test_fallback_reason_schema_unrecoverable():
    assert classify_fallback_reason(
        fallback_used=True,
        llm_error_type="schema_parse_error",
        schema_repair_attempted=True,
        schema_repair_succeeded=False,
        validation_failed=False,
        time_budget_exceeded=False,
    ) == "schema_error_unrecoverable"


def test_fallback_reason_schema_recovered():
    assert classify_fallback_reason(
        fallback_used=True,
        llm_error_type="validation_error",
        schema_repair_attempted=True,
        schema_repair_succeeded=True,
        validation_failed=False,
        time_budget_exceeded=False,
    ) == "schema_error_recovered"


def test_fallback_reason_validation_failed():
    assert classify_fallback_reason(
        fallback_used=True,
        llm_error_type=None,
        schema_repair_attempted=False,
        schema_repair_succeeded=False,
        validation_failed=True,
        time_budget_exceeded=False,
    ) == "validation_failed"


def test_fallback_reason_time_budget_exceeded():
    assert classify_fallback_reason(
        fallback_used=True,
        llm_error_type=None,
        schema_repair_attempted=False,
        schema_repair_succeeded=False,
        validation_failed=False,
        time_budget_exceeded=True,
    ) == "time_budget_exceeded"


def test_fallback_reason_network_error():
    assert classify_fallback_reason(
        fallback_used=True,
        llm_error_type="network_error",
        schema_repair_attempted=False,
        schema_repair_succeeded=False,
        validation_failed=False,
        time_budget_exceeded=False,
    ) == "network_error"


def test_fallback_reason_unknown_llm_error():
    assert classify_fallback_reason(
        fallback_used=True,
        llm_error_type="unknown",
        schema_repair_attempted=False,
        schema_repair_succeeded=False,
        validation_failed=False,
        time_budget_exceeded=False,
    ) == "unknown_llm_error"


def test_payload_contains_new_fallback_fields():
    payload = build_routine_quality_telemetry_payload(
        response=_response(is_fallback=True, generation_status="fallback", status_reason_code="networkError"),
        candidate_pool_size=12,
        ranked_candidates=[_candidate()],
        candidate_payload="x",
        generation_temperature=0.0,
        critic_result=_Critic(),
        repair_count=0,
        fallback_used=True,
        target_duration_min=45,
        fallback_reason="quota_exhausted",
        llm_error_type="quota_exhausted",
        schema_repair_attempted=True,
        schema_repair_succeeded=False,
    )
    assert payload["fallback_reason"] == "quota_exhausted"
    assert payload["llm_error_type"] == "quota_exhausted"
    assert payload["schema_repair_attempted"] is True
    assert payload["schema_repair_succeeded"] is False


def test_payload_default_fallback_reason_none():
    payload = build_routine_quality_telemetry_payload(
        response=_response(),
        candidate_pool_size=12,
        ranked_candidates=[_candidate()],
        candidate_payload="x",
        generation_temperature=0.0,
        critic_result=_Critic(),
        repair_count=0,
        fallback_used=False,
        target_duration_min=45,
    )
    assert payload["fallback_reason"] == "none"
    assert payload["llm_error_type"] is None


def test_payload_excludes_raw_exception_and_free_text():
    payload = build_routine_quality_telemetry_payload(
        response=_response(warnings=["secret warning raw text"]),
        candidate_pool_size=12,
        ranked_candidates=[_candidate()],
        candidate_payload="raw prompt secret",
        generation_temperature=0.0,
        critic_result=_Critic(),
        repair_count=0,
        fallback_used=True,
        target_duration_min=45,
        fallback_reason="quota_exhausted",
        llm_error_type="quota_exhausted",
    )
    payload_text = str(payload)
    assert "raw prompt secret" not in payload_text
    assert "raw exception" not in payload_text
    assert "raw LLM" not in payload_text
    assert "secret warning raw text" not in payload_text
    assert "secret rationale" not in payload_text
    assert "user_note" not in payload


class _FailingLLM:
    def __init__(self, exc):
        self.exc = exc

    def with_structured_output(self, _schema):
        def _raise(_):
            raise self.exc

        return RunnableLambda(_raise)


def test_pipeline_timeout_telemetry_reason(
    monkeypatch,
    sample_request,
    sample_profile,
    mock_candidates,
):
    from engines import routine_pipeline, routine_telemetry

    emitted = []
    monkeypatch.setenv("ENABLE_RULE_CRITIC_TELEMETRY", "true")
    monkeypatch.setattr(routine_telemetry, "emit_routine_quality_telemetry", emitted.append)
    monkeypatch.setattr(routine_pipeline, "get_candidate_exercises", lambda *_args, **_kwargs: mock_candidates)
    monkeypatch.setattr(routine_pipeline, "get_llm", lambda *_args, **_kwargs: _FailingLLM(asyncio.TimeoutError()))

    result = generate_smart_routine(sample_request, db=None, profile=sample_profile, recent_sets=[])

    assert result.is_fallback is True
    assert emitted[0]["fallback_reason"] == "timeout"
    assert emitted[0]["llm_error_type"] == "timeout"


def test_pipeline_quota_telemetry_reason(
    monkeypatch,
    sample_request,
    sample_profile,
    mock_candidates,
):
    from engines import routine_pipeline, routine_telemetry

    emitted = []
    quota_exc = RuntimeError("429 RESOURCE_EXHAUSTED quota exceeded with private details")
    monkeypatch.setenv("ENABLE_RULE_CRITIC_TELEMETRY", "true")
    monkeypatch.setattr(routine_telemetry, "emit_routine_quality_telemetry", emitted.append)
    monkeypatch.setattr(routine_pipeline, "get_candidate_exercises", lambda *_args, **_kwargs: mock_candidates)
    monkeypatch.setattr(routine_pipeline, "get_llm", lambda *_args, **_kwargs: _FailingLLM(quota_exc))

    result = generate_smart_routine(sample_request, db=None, profile=sample_profile, recent_sets=[])

    assert result.is_fallback is True
    assert emitted[0]["fallback_reason"] == "quota_exhausted"
    assert emitted[0]["llm_error_type"] == "quota_exhausted"
    assert "private details" not in str(emitted[0])

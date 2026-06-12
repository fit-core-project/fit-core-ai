from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from engines.llm_parser import SchemaRepairError
from engines.llm_router import LlmProviderConfig
from engines import routine_pipeline


def _valid_payload() -> dict:
    return {
        "total_estimated_time": 45,
        "summary_title": "Raw invoke routine",
        "rationale_summary": ["reason"],
        "warnings": [],
        "exercises": [
            {
                "exercise_id": "bench",
                "exercise_name": "Bench Press",
                "target_reps": 8,
                "sets": 3,
                "rest_time_sec": 90,
                "exercise_rationale": "matches target muscle",
            }
        ],
    }


class _FakeChain:
    def __init__(self, response):
        self.response = response
        self.invoke_count = 0

    def invoke(self, _kwargs):
        self.invoke_count += 1
        return self.response


class _FakePrompt:
    def __init__(self):
        self.chain = None

    def __or__(self, llm):
        self.chain = _FakeChain(llm.response)
        return self.chain


class _FakeLLM:
    def __init__(self, content: str | None = None, *, response=None):
        self.response = response if response is not None else SimpleNamespace(content=content)


def _provider(name: str) -> LlmProviderConfig:
    return LlmProviderConfig(
        requested_provider=name,
        effective_provider=name,
        local_allowed_in_production=False,
        model_name="gemma4:latest",
        is_production=False,
    )


def test_flag_off_uses_existing_structured_output_path(monkeypatch):
    monkeypatch.delenv("ENABLE_LOCAL_RAW_JSON_INVOKE", raising=False)
    monkeypatch.setattr(routine_pipeline, "resolve_llm_provider", lambda: _provider("local"))

    assert routine_pipeline._should_use_local_raw_json_invoke() is False


def test_non_local_provider_does_not_use_raw_path_even_when_flag_true(monkeypatch):
    monkeypatch.setenv("ENABLE_LOCAL_RAW_JSON_INVOKE", "true")
    monkeypatch.setattr(routine_pipeline, "resolve_llm_provider", lambda: _provider("gemini"))

    assert routine_pipeline._should_use_local_raw_json_invoke() is False


def test_local_provider_with_flag_uses_raw_path(monkeypatch):
    monkeypatch.setenv("ENABLE_LOCAL_RAW_JSON_INVOKE", "true")
    monkeypatch.setattr(routine_pipeline, "resolve_llm_provider", lambda: _provider("local"))

    assert routine_pipeline._should_use_local_raw_json_invoke() is True


def test_raw_invoke_valid_json_succeeds():
    prompt = _FakePrompt()
    llm = _FakeLLM(json.dumps(_valid_payload()))

    output = routine_pipeline._invoke_local_raw_json_output(prompt, llm, {})

    assert output.summary_title == "Raw invoke routine"
    assert prompt.chain.invoke_count == 1


def test_response_content_str_extracts_shape():
    content, shape = routine_pipeline._extract_llm_message_content_with_shape(
        SimpleNamespace(
            content=json.dumps(_valid_payload()),
            response_metadata={"finish_reason": "stop"},
            usage_metadata={"input_tokens": 1},
        )
    )

    assert json.loads(content)["summary_title"] == "Raw invoke routine"
    assert shape["raw_invoke_response_class"] == "SimpleNamespace"
    assert shape["raw_invoke_content_present"] is True
    assert shape["raw_invoke_content_type"] == "str"
    assert shape["raw_invoke_content_length_bucket"] == "medium_lt_1000"
    assert shape["raw_invoke_content_stripped_empty"] is False
    assert shape["raw_invoke_has_response_metadata"] is True
    assert shape["raw_invoke_finish_reason"] == "stop"
    assert shape["raw_invoke_usage_present"] is True


def test_response_content_whitespace_records_stripped_empty():
    content, shape = routine_pipeline._extract_llm_message_content_with_shape(SimpleNamespace(content="   \n"))

    assert content == "   \n"
    assert shape["raw_invoke_content_present"] is True
    assert shape["raw_invoke_content_length_bucket"] == "short_lt_100"
    assert shape["raw_invoke_content_stripped_empty"] is True


def test_response_content_list_text_blocks_extracts_text():
    payload = json.dumps(_valid_payload())
    content, shape = routine_pipeline._extract_llm_message_content_with_shape(
        SimpleNamespace(content=[{"type": "text", "text": payload}])
    )

    assert json.loads(content)["exercises"][0]["exercise_id"] == "bench"
    assert shape["raw_invoke_content_type"] == "list"


def test_response_message_content_fallback_extracts_text():
    payload = json.dumps(_valid_payload())
    content, shape = routine_pipeline._extract_llm_message_content_with_shape(
        SimpleNamespace(message=SimpleNamespace(content=payload))
    )

    assert json.loads(content)["summary_title"] == "Raw invoke routine"
    assert shape["raw_invoke_content_present"] is True
    assert shape["raw_invoke_content_type"] == "message.content:str"


def test_response_metadata_done_reason_and_error_are_categorized():
    _content, shape = routine_pipeline._extract_llm_message_content_with_shape(
        SimpleNamespace(
            content="{}",
            response_metadata={"done_reason": "stop", "error": "private provider error text"},
        )
    )

    assert shape["raw_invoke_done_reason"] == "stop"
    assert shape["raw_invoke_error_category"] == "present"
    assert "private provider error text" not in json.dumps(shape)


def test_raw_invoke_fenced_and_prose_json_uses_existing_normalizer():
    raw = "Result:\n```json\n" + json.dumps(_valid_payload()) + "\n```\nDone."
    output = routine_pipeline._invoke_local_raw_json_output(_FakePrompt(), _FakeLLM(raw), {})

    assert output.exercises[0].exercise_id == "bench"


def test_raw_invoke_malformed_json_preserves_fallback_failure_path():
    with pytest.raises(SchemaRepairError) as exc_info:
        routine_pipeline._invoke_local_raw_json_output(_FakePrompt(), _FakeLLM("not json"), {})

    assert exc_info.value.parse_failure_subtype == "json_decode_error"


def test_raw_content_not_stored_in_safe_local_invoke_metadata():
    secret_raw = json.dumps(_valid_payload()).replace("Raw invoke routine", "PRIVATE RAW MODEL TEXT")
    output = routine_pipeline._invoke_local_raw_json_output(_FakePrompt(), _FakeLLM(secret_raw), {})
    metadata = {
        "local_raw_json_invoke_enabled": True,
        "local_raw_json_invoke_used": True,
        "local_raw_json_invoke_succeeded": True,
        "local_raw_json_invoke_failed_reason": None,
        "structured_output_bypassed": True,
    }

    assert output.summary_title == "PRIVATE RAW MODEL TEXT"
    assert "PRIVATE RAW MODEL TEXT" not in json.dumps(metadata)


def test_invoke_with_shape_returns_output_and_shape():
    payload = json.dumps(_valid_payload())
    msg = SimpleNamespace(
        content=payload,
        response_metadata={"finish_reason": "stop"},
        usage_metadata={"input_tokens": 1},
    )
    output, shape = routine_pipeline._invoke_local_raw_json_output_with_shape(
        _FakePrompt(), _FakeLLM(response=msg), {}
    )

    assert output.summary_title == "Raw invoke routine"
    assert shape["raw_invoke_response_class"] == "SimpleNamespace"
    assert shape["raw_invoke_content_present"] is True
    assert shape["raw_invoke_usage_present"] is True


def test_content_length_bucket_boundaries():
    assert routine_pipeline._content_length_bucket("") == "empty"
    assert routine_pipeline._content_length_bucket("x" * 99) == "short_lt_100"
    assert routine_pipeline._content_length_bucket("x" * 100) == "medium_lt_1000"
    assert routine_pipeline._content_length_bucket("x" * 999) == "medium_lt_1000"
    assert routine_pipeline._content_length_bucket("x" * 1000) == "long_gte_1000"


def test_response_missing_metadata_and_usage_records_false():
    _content, shape = routine_pipeline._extract_llm_message_content_with_shape(
        SimpleNamespace(content="{}")
    )

    assert shape["raw_invoke_has_response_metadata"] is False
    assert shape["raw_invoke_finish_reason"] is None
    assert shape["raw_invoke_done_reason"] is None
    assert shape["raw_invoke_usage_present"] is False


def test_response_content_none_records_missing_and_empty():
    content, shape = routine_pipeline._extract_llm_message_content_with_shape(
        SimpleNamespace(content=None)
    )

    assert content == ""
    assert shape["raw_invoke_content_present"] is False
    assert shape["raw_invoke_content_type"] == "missing"
    assert shape["raw_invoke_content_length_bucket"] == "empty"
    assert shape["raw_invoke_content_stripped_empty"] is True


def test_response_content_list_str_blocks_concatenated():
    content, shape = routine_pipeline._extract_llm_message_content_with_shape(
        SimpleNamespace(content=["hello", " ", "world"])
    )

    assert content == "hello world"
    assert shape["raw_invoke_content_type"] == "list"
    assert shape["raw_invoke_content_stripped_empty"] is False

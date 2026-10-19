"""The decoder narrows model output; the server still owns final acceptance."""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from jsonschema import Draft202012Validator
from langchain_core.runnables import RunnableLambda
from langchain_core.exceptions import OutputParserException

from engines import routine_pipeline
from engines.llm_router import LocalJsonChatModel
from engines.routine_selection_schema import build_selection_decoding_schema


def bundles():
    return [
        {"slotId": "press", "candidates": [{"exerciseId": "bench"}, {"exerciseId": "db"}]},
        {"slotId": "pull", "candidates": [{"exerciseId": "row"}]},
        {"slotId": "empty", "candidates": []},
        {"slotId": "flex", "candidates": [{"exerciseId": "server-only"}]},
    ]


def valid(selections):
    return Draft202012Validator(build_selection_decoding_schema(bundles())).is_valid(
        {"selections": selections}
    )


@pytest.mark.parametrize("selections", [[], [{"slotId": "press", "exerciseId": "bench"}],
    [{"slotId": "pull", "exerciseId": "row"}, {"slotId": "press", "exerciseId": "db"}]])
def test_partial_empty_and_valid_pairs_are_admitted(selections):
    assert valid(selections)


@pytest.mark.parametrize("item", [
    {"slotId": "press", "exerciseId": "row"},
    {"slotId": "pull", "exerciseId": "bench"},
    {"slotId": "empty", "exerciseId": "bench"},
    {"slotId": "flex", "exerciseId": "server-only"},
    {"slotId": "press", "exerciseId": "unknown"},
    {"slotId": "unknown", "exerciseId": "bench"},
    {"slotId": "press", "exerciseId": 1},
    {"slotId": "press"},
    {"exerciseId": "bench"},
    {"slotId": "press", "exerciseId": "bench", "sets": 999},
])
def test_invalid_pairs_or_prescription_fields_are_rejected(item):
    assert not valid([item])


def test_no_candidate_schema_only_admits_empty_selection():
    schema = build_selection_decoding_schema([])
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    assert validator.is_valid({"selections": []})
    assert not validator.is_valid({"selections": [{"slotId": "x", "exerciseId": "y"}]})
    assert not validator.is_valid({})
    assert not validator.is_valid({"selections": [], "exercises": []})


def test_slot_count_bounds_repetition_but_server_must_deduplicate():
    pair = {"slotId": "press", "exerciseId": "bench"}
    assert valid([pair, pair])  # Cross-item uniqueness is checked by the existing adapter.
    assert not valid([pair, pair, pair])


def test_schema_deterministic_without_mutating_candidates_or_leaking_metadata():
    original = bundles()
    original[0]["candidates"] += [{"exerciseId": "db", "reason": "PRIVATE FACT"}]
    before = deepcopy(original)
    schema = build_selection_decoding_schema(original)
    original[0]["candidates"].reverse()
    assert schema == build_selection_decoding_schema(original)
    assert "PRIVATE FACT" not in json.dumps(schema)
    original[0]["candidates"].reverse()
    assert original == before


class RecordingOllama:
    model = "synthetic"
    format = "json"
    temperature = 0
    base_url = "http://no-network.invalid"

    def __init__(self, *, error=None, payload=None):
        self.calls = []
        self.error = error
        self.payload = payload

    def invoke(self, _input, config=None, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        schema = kwargs["format"]
        branches = schema["properties"]["selections"]["items"].get("anyOf", [])
        pairs = [{"slotId": b["properties"]["slotId"]["const"],
                  "exerciseId": b["properties"]["exerciseId"]["enum"][0]} for b in branches]
        return SimpleNamespace(content=json.dumps(self.payload if self.payload is not None
                                                 else {"selections": pairs}))


def test_binding_is_request_local_and_keeps_underlying_default():
    inner = RecordingOllama()
    model = LocalJsonChatModel(inner)
    bound = routine_pipeline._bind_local_selection_decoder(model, bundles())
    (RunnableLambda(lambda x: x) | bound).invoke("synthetic")
    assert inner.calls[0]["format"] == build_selection_decoding_schema(bundles())
    assert inner.format == model.format == "json"
    other = routine_pipeline._bind_local_selection_decoder(model, [])
    other.invoke("synthetic")
    assert inner.calls[-1]["format"]["properties"]["selections"]["maxItems"] == 0


def test_unrecognized_provider_is_not_bound():
    model = MagicMock()
    assert routine_pipeline._bind_local_selection_decoder(model, bundles()) is model
    model.bind.assert_not_called()


def test_pipeline_repair_call_keeps_the_same_decoding_schema(
    monkeypatch, sample_request, sample_profile, mock_candidates
):
    inner = RecordingOllama()
    original = inner.invoke

    def fail_once(*args, **kwargs):
        result = original(*args, **kwargs)
        if len(inner.calls) == 1:
            raise OutputParserException("synthetic parser failure without raw content")
        return result

    inner.invoke = fail_once
    monkeypatch.setattr(routine_pipeline, "extract_raw_llm_output_from_exception", lambda _exc: ("", "none"))
    monkeypatch.setattr(routine_pipeline, "get_llm", lambda *_a, **_kw: LocalJsonChatModel(inner))
    monkeypatch.setattr(routine_pipeline, "_should_use_local_raw_json_invoke", lambda: True)
    monkeypatch.setattr(routine_pipeline, "get_candidate_exercises", lambda *_a, **_kw: mock_candidates)
    response = routine_pipeline.generate_smart_routine(
        sample_request.model_copy(update={"doms_data": {}, "pain_areas": []}),
        MagicMock(), profile=sample_profile, recent_sets=[],
    )
    assert len(inner.calls) == 2  # Existing repair path, not a new decoder retry.
    assert inner.calls[0]["format"] == inner.calls[1]["format"]
    assert response.routine_blocks


def test_no_candidates_does_not_create_or_invoke_model(
    monkeypatch, sample_request, sample_profile
):
    factory = MagicMock(side_effect=AssertionError("model must not be called"))
    monkeypatch.setattr(routine_pipeline, "get_llm", factory)
    monkeypatch.setattr(routine_pipeline, "get_candidate_exercises", lambda *_a, **_kw: [])
    response = routine_pipeline.generate_smart_routine(
        sample_request, MagicMock(), profile=sample_profile, recent_sets=[],
    )
    factory.assert_not_called()
    assert response.routine_blocks == []
    assert response.generation_status == "failed"
    assert response.status_reason_code == "emptyCandidate"


@pytest.mark.parametrize("error,expected", [
    (httpx.TimeoutException("timeout"), "llmTimeout"),
    (RuntimeError("schema format unsupported"), "networkError"),
])
def test_pipeline_schema_or_timeout_error_falls_back_without_unconstrained_retry(
    monkeypatch, sample_request, sample_profile, mock_candidates, error, expected
):
    inner = RecordingOllama(error=error)
    monkeypatch.setattr(routine_pipeline, "get_llm", lambda *_a, **_kw: LocalJsonChatModel(inner))
    monkeypatch.setattr(routine_pipeline, "_should_use_local_raw_json_invoke", lambda: True)
    monkeypatch.setattr(routine_pipeline, "get_candidate_exercises", lambda *_a, **_kw: mock_candidates)
    response = routine_pipeline.generate_smart_routine(
        sample_request.model_copy(update={"doms_data": {}, "pain_areas": []}),
        MagicMock(), profile=sample_profile, recent_sets=[],
    )
    assert len(inner.calls) == 1
    assert isinstance(inner.calls[0]["format"], dict)
    assert response.is_fallback and response.status_reason_code == expected
    assert response.routine_blocks


@pytest.mark.parametrize("payload", [None, {"selections": []},
    {"selections": [{"slotId": "bad", "exerciseId": "invented"}]},
    {"selections": [{"slotId": "bad", "exerciseId": "invented"}]*2},
])
def test_pipeline_keeps_server_guards_even_if_provider_ignores_schema(
    monkeypatch, sample_request, sample_profile, mock_candidates, payload
):
    inner = RecordingOllama(payload=payload)
    monkeypatch.setattr(routine_pipeline, "get_llm", lambda *_a, **_kw: LocalJsonChatModel(inner))
    monkeypatch.setattr(routine_pipeline, "_should_use_local_raw_json_invoke", lambda: True)
    monkeypatch.setattr(routine_pipeline, "get_candidate_exercises", lambda *_a, **_kw: mock_candidates)
    response = routine_pipeline.generate_smart_routine(
        sample_request.model_copy(update={"doms_data": {}, "pain_areas": []}),
        MagicMock(), profile=sample_profile, recent_sets=[],
    )
    assert len(inner.calls) == 1
    ids = [block.exercise_id for block in response.routine_blocks]
    assert ids and len(ids) == len(set(ids))
    assert set(ids) <= {str(c["id"]) for c in mock_candidates}
    assert "invented" not in ids

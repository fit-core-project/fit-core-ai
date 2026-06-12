from __future__ import annotations

import json

import pytest

from engines.llm_parser import (
    SchemaRepairError,
    classify_validation_error,
    normalize_llm_response,
    parse_json_candidate,
)
from engines.schemas import LLMRoutineOutput


def _valid_payload() -> dict:
    return {
        "total_estimated_time": 45,
        "summary_title": "Test routine",
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


def _valid_json() -> str:
    return json.dumps(_valid_payload())


def test_valid_json_object_parses():
    result = parse_json_candidate(_valid_json())

    assert result.success is True
    assert result.data["summary_title"] == "Test routine"


def test_markdown_fenced_json_parses_with_subtype():
    result = parse_json_candidate(f"```json\n{_valid_json()}\n```")

    assert result.success is True
    assert result.subtype == "markdown_fence_wrapped"


def test_leading_trailing_prose_balanced_json_extracts():
    result = parse_json_candidate(f"Here is the routine:\n{_valid_json()}\nDone.")

    assert result.success is True
    assert result.subtype == "leading_or_trailing_prose"
    assert result.data["exercises"][0]["exercise_id"] == "bench"


def test_empty_response_returns_empty_response_subtype():
    result = parse_json_candidate("   ")

    assert result.success is False
    assert result.subtype == "empty_response"


def test_non_object_json_returns_non_object_json_subtype():
    result = parse_json_candidate("[1, 2, 3]")

    assert result.success is False
    assert result.subtype == "non_object_json"


def test_malformed_json_returns_json_decode_error_subtype():
    result = parse_json_candidate('{"summary_title": }')

    assert result.success is False
    assert result.subtype == "json_decode_error"


def test_normalize_parses_leading_trailing_prose():
    output = normalize_llm_response(f"Result:\n{_valid_json()}\nThanks")

    assert isinstance(output, LLMRoutineOutput)
    assert output.exercises[0].exercise_id == "bench"


def test_missing_required_field_category_stores_field_names_only():
    payload = _valid_payload()
    del payload["exercises"][0]["exercise_name"]

    with pytest.raises(SchemaRepairError) as exc_info:
        normalize_llm_response(json.dumps(payload))

    error = exc_info.value
    assert error.parse_failure_subtype == "pydantic_validation_error"
    assert error.schema_validation_error_category == "missing_required_field"
    assert error.schema_validation_field_names == ["exercise_name"]
    assert "Bench Press" not in json.dumps(error.schema_validation_field_names)


def test_wrong_type_category_stores_field_names_only():
    payload = _valid_payload()
    payload["exercises"][0]["sets"] = {"bad": "shape"}

    with pytest.raises(SchemaRepairError) as exc_info:
        normalize_llm_response(json.dumps(payload))

    error = exc_info.value
    assert error.schema_validation_error_category == "wrong_field_type"
    assert error.schema_validation_field_names == ["sets"]


def test_classify_validation_error_does_not_expose_raw_message():
    payload = _valid_payload()
    payload["exercises"][0]["rest_time_sec"] = {"private": "do not store"}

    try:
        LLMRoutineOutput(**payload)
    except Exception as exc:
        category, fields = classify_validation_error(exc)

    assert category == "wrong_field_type"
    assert fields == ["rest_time_sec"]
    assert "private" not in json.dumps(fields)

from __future__ import annotations

import json

from engines.llm_parser import classify_json_decode_error, normalize_llm_response, parse_json_candidate


def _payload() -> dict:
    return {
        "total_estimated_time": 45,
        "summary_title": "JSON recovery",
        "rationale_summary": ["reason"],
        "warnings": [],
        "exercises": [
            {
                "exercise_id": "bench",
                "exercise_name": "Bench",
                "target_reps": 8,
                "sets": 3,
                "rest_time_sec": 90,
                "exercise_rationale": "safe candidate",
            }
        ],
    }


def test_unterminated_string_category():
    text = '{"summary_title": "broken}'
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        assert classify_json_decode_error(exc, text) == "json_decode_unterminated_string"


def test_unbalanced_braces_category():
    result = parse_json_candidate('{"summary_title": "broken"')

    assert result.success is True
    assert result.json_decode_error_category == "json_decode_unbalanced_braces"


def test_trailing_comma_object_recovery_succeeds():
    payload = _payload()
    payload["extra"] = {"ok": True}
    text = json.dumps(payload).replace('"ok": true}', '"ok": true,}')
    result = parse_json_candidate(text)

    assert result.success is True
    assert result.json_decode_recovery_attempted is True
    assert result.json_decode_recovery_succeeded is True
    assert result.json_decode_recovery_strategy == "trailing_comma_removed"
    assert normalize_llm_response(text).summary_title == "JSON recovery"


def test_trailing_comma_array_recovery_succeeds():
    payload = _payload()
    text = json.dumps(payload).replace('"rationale_summary": ["reason"]', '"rationale_summary": ["reason",]')
    result = parse_json_candidate(text)

    assert result.success is True
    assert result.json_decode_error_category == "json_decode_trailing_comma"
    assert result.json_decode_recovery_strategy == "trailing_comma_removed"


def test_extra_data_category_or_extraction_recovery():
    text = json.dumps(_payload()) + "\nextra"
    result = parse_json_candidate(text)

    assert result.success is True
    assert result.json_decode_error_category == "json_decode_extra_data"


def test_python_literal_outside_strings_classified():
    result = parse_json_candidate('{"warnings": None}')

    assert result.success is False
    assert result.json_decode_error_category == "json_decode_python_literal"


def test_single_quote_suspected_but_not_converted():
    result = parse_json_candidate("{'warnings': []}")

    assert result.success is False
    assert result.json_decode_error_category == "json_decode_single_quotes_suspected"


def test_invalid_control_character_classified():
    text = '{"summary_title": "bad\ntext"}'
    result = parse_json_candidate(text)

    assert result.success is False
    assert result.json_decode_error_category == "json_decode_invalid_control_character"


def test_markdown_and_prose_existing_recovery_still_parse():
    text = "```json\n" + json.dumps(_payload()) + "\n```"
    assert parse_json_candidate(text).subtype == "markdown_fence_wrapped"

    prose = "Here:\n" + json.dumps(_payload()) + "\nDone"
    assert parse_json_candidate(prose).subtype == "leading_or_trailing_prose"

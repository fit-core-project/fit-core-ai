import logging

import pytest
from fastapi import HTTPException
from langchain_core.prompts import ChatPromptTemplate

from engines.fallback import _debug_print_llm_output, _debug_print_prompt
from engines.log_redaction import (
    redact_free_text,
    sanitize_dev_log_payload,
    sanitize_exception_for_log,
    summarize_text_for_log,
)
from engines.schemas import LLMExercisePlan, LLMRoutineOutput


def test_redact_free_text_redacts_string():
    assert redact_free_text("secret") == "[REDACTED]"


def test_redact_free_text_redacts_none():
    assert redact_free_text(None) == "[REDACTED]"


def test_summarize_text_for_log_returns_counts_only():
    summary = summarize_text_for_log("abcd")

    assert summary == {"redacted": True, "char_count": 4, "approx_tokens": 1}
    assert "abcd" not in str(summary)


def test_summarize_text_for_log_none_has_zero_counts():
    summary = summarize_text_for_log(None)

    assert summary["redacted"] is True
    assert summary["char_count"] == 0
    assert summary["approx_tokens"] == 0


def test_sanitize_exception_for_log_does_not_include_message():
    sanitized = sanitize_exception_for_log(Exception("secret message"))

    assert sanitized["type"] == "Exception"
    assert "secret message" not in str(sanitized)


def test_sanitize_dev_log_payload_removes_raw_prompt():
    payload = sanitize_dev_log_payload({"raw_prompt": "secret prompt", "event": "x"})

    assert payload == {"event": "x"}
    assert "secret prompt" not in str(payload)


def test_sanitize_dev_log_payload_removes_raw_llm_output():
    payload = sanitize_dev_log_payload({"raw_llm_output": "secret output", "event": "x"})

    assert payload == {"event": "x"}
    assert "secret output" not in str(payload)


def test_sanitize_dev_log_payload_removes_user_note():
    payload = sanitize_dev_log_payload({"user_note": "secret note", "event": "x"})

    assert payload == {"event": "x"}
    assert "secret note" not in str(payload)


def test_sanitize_dev_log_payload_removes_rationale():
    payload = sanitize_dev_log_payload({"rationale": "secret rationale", "event": "x"})

    assert payload == {"event": "x"}
    assert "secret rationale" not in str(payload)


def test_sanitize_dev_log_payload_keeps_allowlisted_fields():
    payload = sanitize_dev_log_payload(
        {
            "event": "routine_generation_quality",
            "candidate_pool_size": 18,
            "fallback_reason": "quota_exhausted",
            "llm_error_type": "quota_exhausted",
            "critic_score": 87,
        }
    )

    assert payload == {
        "event": "routine_generation_quality",
        "candidate_pool_size": 18,
        "fallback_reason": "quota_exhausted",
        "llm_error_type": "quota_exhausted",
        "critic_score": 87,
    }


def test_sanitize_dev_log_payload_removes_request_body_and_traceback():
    payload = sanitize_dev_log_payload(
        {
            "request_body": {"text": "secret request"},
            "exception_message": "secret exception",
            "traceback": "secret stack",
            "error_type": "RuntimeError",
        }
    )

    assert payload == {"error_type": "RuntimeError"}
    assert "secret" not in str(payload)


def _logged_messages(caplog):
    return "\n".join(record.getMessage() for record in caplog.records)


def test_fallback_debug_prompt_does_not_print_raw_prompt_or_user_note(caplog):
    caplog.set_level(logging.DEBUG, logger="engines.fallback")
    prompt = ChatPromptTemplate.from_messages(
        [("human", "Routine prompt with note={user_note}")]
    )

    _debug_print_prompt(prompt, {"user_note": "secret-user-note"})

    output = _logged_messages(caplog)
    assert "secret-user-note" not in output
    assert "Routine prompt with note" not in output
    assert "prompt_char_count" in output


def test_fallback_debug_prompt_render_failure_does_not_print_invoke_values(caplog):
    caplog.set_level(logging.DEBUG, logger="engines.fallback")
    prompt = ChatPromptTemplate.from_messages(
        [("human", "Routine prompt with missing={missing}")]
    )

    _debug_print_prompt(prompt, {"user_note": "secret-user-note"})

    output = _logged_messages(caplog)
    assert "secret-user-note" not in output
    assert "Routine prompt with missing" not in output
    assert "error_type" in output


def test_fallback_debug_llm_output_does_not_print_raw_rationale(caplog):
    caplog.set_level(logging.DEBUG, logger="engines.fallback")
    output = LLMRoutineOutput(
        total_estimated_time=30,
        summary_title="secret summary title",
        rationale_summary=["secret rationale summary"],
        exercises=[
            LLMExercisePlan(
                exercise_id="pushup",
                exercise_name="Secret Exercise Name",
                movement_type="COMPOUND",
                primary_muscles=["chest"],
                target_reps=10,
                sets=3,
                rest_time_sec=90,
                exercise_rationale="secret exercise rationale",
            )
        ],
        warnings=["secret warning"],
    )

    _debug_print_llm_output("TEST_OUTPUT", output)

    captured = _logged_messages(caplog)
    assert "secret exercise rationale" not in captured
    assert "secret rationale summary" not in captured
    assert "secret summary title" not in captured
    assert "Secret Exercise Name" not in captured
    assert "llm_output_char_count" in captured
    assert "parsed_exercise_count" in captured


def test_main_parse_log_logging_redacts_text(monkeypatch, caplog):
    import main

    caplog.set_level(logging.INFO, logger="main")
    monkeypatch.setattr(main, "parse_natural_language_log", lambda text: "{}")

    assert main.api_parse_log(main.LogRequest(text="secret nlp text")) == {}

    captured = _logged_messages(caplog)
    assert "secret nlp text" not in captured
    assert "char_count" in captured


def test_main_supplement_logging_redacts_question(monkeypatch, caplog):
    import main

    caplog.set_level(logging.INFO, logger="main")

    class FakeSupplementRag:
        def answer_question(self, question):
            return {"answer": "ok", "sources": []}

    monkeypatch.setattr(main, "supplement_rag", FakeSupplementRag())

    assert main.api_supplement_chat(
        main.SupplementChatRequest(question="secret supplement question")
    ) == {"answer": "ok", "sources": []}

    captured = _logged_messages(caplog)
    assert "secret supplement question" not in captured
    assert "char_count" in captured


def test_main_supplement_exception_logging_redacts_message(monkeypatch, caplog):
    import main

    caplog.set_level(logging.ERROR, logger="main")

    class FailingSupplementRag:
        def answer_question(self, question):
            raise RuntimeError("secret exception message")

    monkeypatch.setattr(main, "supplement_rag", FailingSupplementRag())

    result = main.api_supplement_chat(
        main.SupplementChatRequest(question="secret supplement question")
    )

    assert result["mode"] == "degraded"

    captured = _logged_messages(caplog)
    assert "secret supplement question" not in captured
    assert "secret exception message" not in captured
    assert "RuntimeError" in captured

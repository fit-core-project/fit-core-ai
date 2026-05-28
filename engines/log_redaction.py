"""Helpers for keeping dev/debug logs free of raw user or model text."""

from __future__ import annotations

import math
from typing import Any


_REDACTED = "[REDACTED]"

SAFE_DEV_LOG_FIELDS = {
    "event",
    "routine_draft_id",
    "candidate_pool_size",
    "candidate_count",
    "prompt_char_count",
    "prompt_approx_tokens",
    "llm_output_char_count",
    "llm_output_approx_tokens",
    "parsed_exercise_count",
    "generation_status",
    "fallback_used",
    "fallback_reason",
    "llm_error_type",
    "schema_repair_attempted",
    "schema_repair_succeeded",
    "critic_score",
    "critic_grade",
    "error_type",
    "error_category",
    "char_count",
    "approx_tokens",
    "redacted",
}


def redact_free_text(value: str | None, replacement: str = _REDACTED) -> str:
    """Return a fixed marker for any free text value."""
    return replacement


def summarize_text_for_log(value: str | None) -> dict[str, Any]:
    """Return a count-only summary without preserving any original text."""
    char_count = len(value) if value is not None else 0
    return {
        "redacted": True,
        "char_count": char_count,
        "approx_tokens": math.ceil(char_count / 4) if char_count else 0,
    }


def sanitize_exception_for_log(exc: Exception | None) -> dict[str, str | None]:
    """Return exception metadata without the exception message or traceback."""
    if exc is None:
        return {"type": None, "category": "unknown"}

    category = "unknown"
    try:
        from engines.llm_router import classify_llm_error

        category = classify_llm_error(exc)
    except Exception:
        category = "unknown"

    return {
        "type": type(exc).__name__,
        "category": category,
    }


def sanitize_dev_log_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Allowlist structured dev-log fields and drop raw/free-text fields."""
    return {
        key: value
        for key, value in payload.items()
        if key in SAFE_DEV_LOG_FIELDS
    }

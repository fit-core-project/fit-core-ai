from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engines.llm_router import resolve_llm_provider
from scripts import check_local_llm_readiness as local_readiness
from scripts import run_feedback_ranking_staging_test as staging

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_OUTPUT_DIR = "tests/evaluation/.artifacts/gemma4-suite"
DEFAULT_REPEATS = 5
DEFAULT_REQUEST_TIMEOUT = 240
DEFAULT_MIN_REQUESTS_PER_GROUP = 50
DEFAULT_CANDIDATE_POOL_SIZE = 12

FORBIDDEN_REPORT_TOKENS = {
    *staging.FORBIDDEN_PRIVACY_TOKENS,
    "raw_prompt",
    "raw prompt",
    "raw_llm_output",
    "raw model response",
    "api_key",
    "GOOGLE_API_KEY",
    "user_note",
    "userNote",
}


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("gemma4-suite-%Y%m%dT%H%M%SZ")


def _sanitize_run_id(value: str | None) -> str:
    raw = value or _utc_run_id()
    sanitized = re.sub(r"[^A-Za-z0-9_.:-]", "-", raw).strip("-")
    return sanitized or _utc_run_id()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _suite_paths(output_dir: str | Path, run_id: str) -> dict[str, Path]:
    root = Path(output_dir)
    return {
        "output_dir": root,
        "preflight": root / "preflight.json",
        "group_a": root / "group_a.json",
        "group_b": root / "group_b.json",
        "group_a_summary": root / "group_a_summary.json",
        "group_b_summary": root / "group_b_summary.json",
        "compare": root / "feedback-ranking-compare.json",
        "compare_md": root / "feedback-ranking-compare.md",
        "suite_json": root / "gemma4-suite-report.json",
        "suite_md": root / "gemma4-suite-report.md",
        "provider_check": root / "provider-check.json",
    }


def _http_get_status(url: str, timeout_sec: float = 5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as response:
            return 200 <= int(response.status) < 300
    except Exception:
        return False


@contextmanager
def _temporary_gemma4_metadata_env(model: str, ollama_base_url: str):
    old = {key: os.environ.get(key) for key in ("LLM_PROVIDER", "LOCAL_LLM_MODEL", "OLLAMA_BASE_URL")}
    os.environ["LLM_PROVIDER"] = "local"
    os.environ["LOCAL_LLM_MODEL"] = model
    os.environ["OLLAMA_BASE_URL"] = ollama_base_url
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def build_preflight_report(
    *,
    run_id: str,
    base_url: str,
    ollama_base_url: str,
    model: str,
    output_dir: str,
    timeout_sec: float,
    no_seed_reset: bool = False,
    skip_readiness_check: bool = False,
) -> dict[str, Any]:
    if skip_readiness_check:
        ollama_report = {
            "readiness_status": "skipped",
            "ollama_reachable": None,
            "model_installed": None,
            "model": model,
            "probe_success": None,
            "probe_latency_ms": None,
            "error_category": None,
        }
    else:
        ollama_report = local_readiness.build_readiness_report(
            base_url=ollama_base_url,
            model=model,
            timeout_sec=timeout_sec,
            probe=True,
        )

    db_report = staging.ensure_routine_feedback_table()
    seed_report: dict[str, Any] = {}
    if db_report.get("db_readiness_status") in staging.DB_READY_STATUSES:
        if no_seed_reset:
            seed_report = staging.verify_seed_readiness()
        else:
            seed_report = staging.perform_seed(base_url=base_url, user_id=staging.DEFAULT_USER_ID, reset_seed=True)
    else:
        seed_report = {
            "seed_readiness_status": "failed_db_not_ready",
            "seed_verification_passed": False,
            "missing_seed_count": None,
        }

    verify_report = staging.verify_seed_readiness() if db_report.get("db_readiness_status") in staging.DB_READY_STATUSES else {}
    docs_reachable = _http_get_status(f"{base_url.rstrip('/')}/docs")
    dev_logs_reachable = _http_get_status(f"{base_url.rstrip('/')}/api/dev/logs?limit=1")
    readiness_passed = (
        (skip_readiness_check or ollama_report.get("readiness_status") == "pass")
        and db_report.get("db_readiness_status") in staging.DB_READY_STATUSES
        and bool(verify_report.get("seed_verification_passed", seed_report.get("seed_verification_passed")))
        and docs_reachable
        and dev_logs_reachable
    )
    error_category = None
    if not readiness_passed:
        error_category = (
            ollama_report.get("error_category")
            or db_report.get("error_category")
            or seed_report.get("seed_readiness_status")
            or "backend_unreachable"
        )

    return {
        "run_id": run_id,
        "ollama_readiness_status": ollama_report.get("readiness_status"),
        "ollama_reachable": ollama_report.get("ollama_reachable"),
        "model_installed": ollama_report.get("model_installed"),
        "model": model,
        "probe_success": ollama_report.get("probe_success"),
        "probe_latency_ms": ollama_report.get("probe_latency_ms"),
        "db_readiness_status": db_report.get("db_readiness_status"),
        "seed_readiness_status": verify_report.get("seed_readiness_status", seed_report.get("seed_readiness_status")),
        "seed_verification_passed": verify_report.get(
            "seed_verification_passed",
            seed_report.get("seed_verification_passed"),
        ),
        "docs_reachable": docs_reachable,
        "dev_logs_reachable": dev_logs_reachable,
        "readiness_passed": readiness_passed,
        "error_category": error_category,
        "output_dir": str(output_dir),
    }


def _events(raw: dict[str, Any]) -> list[dict[str, Any]]:
    return staging._events(raw)


def _avg(events: list[dict[str, Any]], key: str) -> float:
    return staging._avg(events, key)


def _p95(events: list[dict[str, Any]], key: str) -> float:
    return staging._p95(events, key)


def _first_non_null(events: list[dict[str, Any]], key: str) -> Any:
    for event in events:
        value = event.get(key)
        if value is not None:
            return value
    return None


def _rate(events: list[dict[str, Any]], key: str) -> float:
    return staging._rate(events, key)


def _candidate_pool_size(events: list[dict[str, Any]]) -> int | None:
    values = [event.get("candidate_pool_size") for event in events if isinstance(event.get("candidate_pool_size"), int)]
    if not values:
        return None
    unique = sorted(set(values))
    return unique[0] if len(unique) == 1 else None


def _schema_repair_rate(events: list[dict[str, Any]], key: str) -> float:
    return _rate(events, key)


def _counter_for_events(events: list[dict[str, Any]], key: str, *, default: str | None = None) -> dict[str, int]:
    return dict(sorted(Counter(str(event.get(key, default) if event.get(key) is not None else default) for event in events).items()))


def _scenario_ids(raw: dict[str, Any]) -> list[str]:
    return [
        str(result.get("scenario_id"))
        for result in raw.get("results", [])
        if result.get("scenario_id") is not None and isinstance(result.get("telemetry"), dict)
    ]


def _fallback_count_by_scenario(raw: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for result in raw.get("results", []):
        telemetry = result.get("telemetry")
        scenario_id = result.get("scenario_id")
        if not scenario_id or not isinstance(telemetry, dict):
            continue
        if telemetry.get("fallback_used") is True:
            counts[str(scenario_id)] += 1
    return dict(sorted(counts.items()))


def _counts_by_scenario(
    raw: dict[str, Any],
    key: str,
    *,
    fallback_only: bool = False,
    default: str | None = None,
) -> dict[str, dict[str, int]]:
    counts_by_scenario: dict[str, Counter[str]] = {}
    for result in raw.get("results", []):
        telemetry = result.get("telemetry")
        scenario_id = result.get("scenario_id")
        if not scenario_id or not isinstance(telemetry, dict):
            continue
        if fallback_only and telemetry.get("fallback_used") is not True:
            continue
        value = telemetry.get(key, default) if telemetry.get(key) is not None else default
        counts_by_scenario.setdefault(str(scenario_id), Counter())[str(value)] += 1
    return {
        scenario_id: dict(sorted(counts.items()))
        for scenario_id, counts in sorted(counts_by_scenario.items())
    }


def _fallback_rate_by_scenario(raw: dict[str, Any]) -> dict[str, float]:
    scenario_totals = Counter(_scenario_ids(raw))
    fallback_counts = _fallback_count_by_scenario(raw)
    return {
        scenario_id: fallback_counts.get(scenario_id, 0) / total
        for scenario_id, total in sorted(scenario_totals.items())
        if total > 0
    }


def _top_fallback_scenarios(counts: dict[str, int], *, limit: int = 5) -> dict[str, int]:
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit])


def _scenario_delta(a_counts: dict[str, int], b_counts: dict[str, int]) -> dict[str, int]:
    scenario_ids = sorted(set(a_counts) | set(b_counts))
    return {scenario_id: b_counts.get(scenario_id, 0) - a_counts.get(scenario_id, 0) for scenario_id in scenario_ids}


def _schema_repair_count(events: list[dict[str, Any]], key: str) -> int:
    return sum(1 for event in events if event.get(key) is True)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def build_fallback_breakdown(
    raw_a: dict[str, Any],
    raw_b: dict[str, Any],
) -> dict[str, Any]:
    events_a = _events(raw_a)
    events_b = _events(raw_b)
    fallback_events_a = [event for event in events_a if event.get("fallback_used") is True]
    fallback_events_b = [event for event in events_b if event.get("fallback_used") is True]
    fallback_counts_a = _fallback_count_by_scenario(raw_a)
    fallback_counts_b = _fallback_count_by_scenario(raw_b)
    attempted_a = _schema_repair_count(events_a, "schema_repair_attempted")
    attempted_b = _schema_repair_count(events_b, "schema_repair_attempted")
    succeeded_a = _schema_repair_count(events_a, "schema_repair_succeeded")
    succeeded_b = _schema_repair_count(events_b, "schema_repair_succeeded")
    return {
        "fallback_reason_counts_a": _counter_for_events(events_a, "fallback_reason"),
        "fallback_reason_counts_b": _counter_for_events(events_b, "fallback_reason"),
        "llm_error_type_counts_a": _counter_for_events(events_a, "llm_error_type"),
        "llm_error_type_counts_b": _counter_for_events(events_b, "llm_error_type"),
        "fallback_count_by_scenario_a": fallback_counts_a,
        "fallback_count_by_scenario_b": fallback_counts_b,
        "fallback_reason_counts_by_scenario_a": _counts_by_scenario(raw_a, "fallback_reason", fallback_only=True),
        "fallback_reason_counts_by_scenario_b": _counts_by_scenario(raw_b, "fallback_reason", fallback_only=True),
        "llm_error_type_counts_by_scenario_a": _counts_by_scenario(raw_a, "llm_error_type", fallback_only=True),
        "llm_error_type_counts_by_scenario_b": _counts_by_scenario(raw_b, "llm_error_type", fallback_only=True),
        "parse_failure_subtype_counts_a": _counter_for_events(
            fallback_events_a, "parse_failure_subtype", default="unknown"
        ),
        "parse_failure_subtype_counts_b": _counter_for_events(
            fallback_events_b, "parse_failure_subtype", default="unknown"
        ),
        "parse_failure_subtype_counts_by_scenario_a": _counts_by_scenario(
            raw_a, "parse_failure_subtype", fallback_only=True, default="unknown"
        ),
        "parse_failure_subtype_counts_by_scenario_b": _counts_by_scenario(
            raw_b, "parse_failure_subtype", fallback_only=True, default="unknown"
        ),
        "schema_validation_error_category_counts_a": _counter_for_events(
            fallback_events_a, "schema_validation_error_category", default="unknown"
        ),
        "schema_validation_error_category_counts_b": _counter_for_events(
            fallback_events_b, "schema_validation_error_category", default="unknown"
        ),
        "repair_failure_reason_counts_a": _counter_for_events(
            fallback_events_a, "repair_failure_reason", default="unknown"
        ),
        "repair_failure_reason_counts_b": _counter_for_events(
            fallback_events_b, "repair_failure_reason", default="unknown"
        ),
        "raw_output_recovery_attempted_count_a": _schema_repair_count(
            events_a, "raw_output_recovery_attempted"
        ),
        "raw_output_recovery_attempted_count_b": _schema_repair_count(
            events_b, "raw_output_recovery_attempted"
        ),
        "raw_output_recovery_succeeded_count_a": _schema_repair_count(
            events_a, "raw_output_recovery_succeeded"
        ),
        "raw_output_recovery_succeeded_count_b": _schema_repair_count(
            events_b, "raw_output_recovery_succeeded"
        ),
        "raw_output_recovery_source_counts_a": _counter_for_events(
            fallback_events_a, "raw_output_recovery_source", default="none"
        ),
        "raw_output_recovery_source_counts_b": _counter_for_events(
            fallback_events_b, "raw_output_recovery_source", default="none"
        ),
        "raw_output_recovery_failed_reason_counts_a": _counter_for_events(
            fallback_events_a, "raw_output_recovery_failed_reason", default="none"
        ),
        "raw_output_recovery_failed_reason_counts_b": _counter_for_events(
            fallback_events_b, "raw_output_recovery_failed_reason", default="none"
        ),
        "json_decode_error_category_counts_a": _counter_for_events(
            fallback_events_a, "json_decode_error_category", default="unknown"
        ),
        "json_decode_error_category_counts_b": _counter_for_events(
            fallback_events_b, "json_decode_error_category", default="unknown"
        ),
        "json_decode_error_category_counts_by_scenario_a": _counts_by_scenario(
            raw_a, "json_decode_error_category", fallback_only=True, default="unknown"
        ),
        "json_decode_error_category_counts_by_scenario_b": _counts_by_scenario(
            raw_b, "json_decode_error_category", fallback_only=True, default="unknown"
        ),
        "json_decode_recovery_attempted_count_a": _schema_repair_count(
            events_a, "json_decode_recovery_attempted"
        ),
        "json_decode_recovery_attempted_count_b": _schema_repair_count(
            events_b, "json_decode_recovery_attempted"
        ),
        "json_decode_recovery_succeeded_count_a": _schema_repair_count(
            events_a, "json_decode_recovery_succeeded"
        ),
        "json_decode_recovery_succeeded_count_b": _schema_repair_count(
            events_b, "json_decode_recovery_succeeded"
        ),
        "json_decode_recovery_strategy_counts_a": _counter_for_events(
            fallback_events_a, "json_decode_recovery_strategy", default="none"
        ),
        "json_decode_recovery_strategy_counts_b": _counter_for_events(
            fallback_events_b, "json_decode_recovery_strategy", default="none"
        ),
        "local_raw_json_invoke_used_count_a": _schema_repair_count(
            events_a, "local_raw_json_invoke_used"
        ),
        "local_raw_json_invoke_used_count_b": _schema_repair_count(
            events_b, "local_raw_json_invoke_used"
        ),
        "local_raw_json_invoke_succeeded_count_a": _schema_repair_count(
            events_a, "local_raw_json_invoke_succeeded"
        ),
        "local_raw_json_invoke_succeeded_count_b": _schema_repair_count(
            events_b, "local_raw_json_invoke_succeeded"
        ),
        "local_raw_json_invoke_failed_reason_counts_a": _counter_for_events(
            events_a, "local_raw_json_invoke_failed_reason", default="none"
        ),
        "local_raw_json_invoke_failed_reason_counts_b": _counter_for_events(
            events_b, "local_raw_json_invoke_failed_reason", default="none"
        ),
        "structured_output_bypassed_count_a": _schema_repair_count(
            events_a, "structured_output_bypassed"
        ),
        "structured_output_bypassed_count_b": _schema_repair_count(
            events_b, "structured_output_bypassed"
        ),
        "prompt_char_count_avg_a": _avg(events_a, "prompt_char_count"),
        "prompt_char_count_avg_b": _avg(events_b, "prompt_char_count"),
        "prompt_char_count_p95_a": _p95(events_a, "prompt_char_count"),
        "prompt_char_count_p95_b": _p95(events_b, "prompt_char_count"),
        "prompt_approx_tokens_avg_a": _avg(events_a, "prompt_approx_tokens"),
        "prompt_approx_tokens_avg_b": _avg(events_b, "prompt_approx_tokens"),
        "prompt_approx_tokens_p95_a": _p95(events_a, "prompt_approx_tokens"),
        "prompt_approx_tokens_p95_b": _p95(events_b, "prompt_approx_tokens"),
        "local_llm_num_predict_a": _first_non_null(events_a, "local_llm_num_predict"),
        "local_llm_num_predict_b": _first_non_null(events_b, "local_llm_num_predict"),
        "local_llm_num_ctx_a": _first_non_null(events_a, "local_llm_num_ctx"),
        "local_llm_num_ctx_b": _first_non_null(events_b, "local_llm_num_ctx"),
        "raw_invoke_response_class_counts_a": _counter_for_events(
            events_a, "raw_invoke_response_class", default="none"
        ),
        "raw_invoke_response_class_counts_b": _counter_for_events(
            events_b, "raw_invoke_response_class", default="none"
        ),
        "raw_invoke_content_present_count_a": _schema_repair_count(
            events_a, "raw_invoke_content_present"
        ),
        "raw_invoke_content_present_count_b": _schema_repair_count(
            events_b, "raw_invoke_content_present"
        ),
        "raw_invoke_stripped_empty_count_a": _schema_repair_count(
            events_a, "raw_invoke_content_stripped_empty"
        ),
        "raw_invoke_stripped_empty_count_b": _schema_repair_count(
            events_b, "raw_invoke_content_stripped_empty"
        ),
        "raw_invoke_finish_reason_counts_a": _counter_for_events(
            events_a, "raw_invoke_finish_reason", default="none"
        ),
        "raw_invoke_finish_reason_counts_b": _counter_for_events(
            events_b, "raw_invoke_finish_reason", default="none"
        ),
        "raw_invoke_done_reason_counts_a": _counter_for_events(
            events_a, "raw_invoke_done_reason", default="none"
        ),
        "raw_invoke_done_reason_counts_b": _counter_for_events(
            events_b, "raw_invoke_done_reason", default="none"
        ),
        "raw_invoke_error_category_counts_a": _counter_for_events(
            events_a, "raw_invoke_error_category", default="none"
        ),
        "raw_invoke_error_category_counts_b": _counter_for_events(
            events_b, "raw_invoke_error_category", default="none"
        ),
        "schema_repair_attempted_count_a": attempted_a,
        "schema_repair_attempted_count_b": attempted_b,
        "schema_repair_succeeded_count_a": succeeded_a,
        "schema_repair_succeeded_count_b": succeeded_b,
        "schema_repair_attempted_rate_a": _schema_repair_rate(events_a, "schema_repair_attempted"),
        "schema_repair_attempted_rate_b": _schema_repair_rate(events_b, "schema_repair_attempted"),
        "schema_repair_succeeded_rate_a": _schema_repair_rate(events_a, "schema_repair_succeeded"),
        "schema_repair_succeeded_rate_b": _schema_repair_rate(events_b, "schema_repair_succeeded"),
        "schema_repair_success_given_attempt_rate_a": _safe_ratio(succeeded_a, attempted_a),
        "schema_repair_success_given_attempt_rate_b": _safe_ratio(succeeded_b, attempted_b),
        "fallback_rate_by_scenario_a": _fallback_rate_by_scenario(raw_a),
        "fallback_rate_by_scenario_b": _fallback_rate_by_scenario(raw_b),
        "top_fallback_scenarios_a": _top_fallback_scenarios(fallback_counts_a),
        "top_fallback_scenarios_b": _top_fallback_scenarios(fallback_counts_b),
        "b_minus_a_fallback_delta_by_scenario": _scenario_delta(fallback_counts_a, fallback_counts_b),
    }


def build_group_summary(
    raw: dict[str, Any],
    *,
    group: str,
    model: str,
    require_local_provider: bool = True,
) -> dict[str, Any]:
    events = _events(raw)
    feedback_enabled_rate = _rate(events, "feedback_enabled")
    candidate_pool_size = _candidate_pool_size(events)
    provider = raw.get("llm_provider")
    local_model = raw.get("local_llm_model")
    provider_mismatch = False
    if require_local_provider and provider is not None and provider != "local":
        provider_mismatch = True
    if require_local_provider and local_model is not None and str(local_model) != model:
        provider_mismatch = True
    if candidate_pool_size is not None and candidate_pool_size != DEFAULT_CANDIDATE_POOL_SIZE:
        provider_mismatch = True
    if group == "a" and feedback_enabled_rate != 0:
        provider_mismatch = True
    if group == "b" and feedback_enabled_rate != 1:
        provider_mismatch = True

    return {
        "group": group,
        "request_count": int(raw.get("request_count") or len(raw.get("results", []))),
        "failed_requests": int(raw.get("failed_request_count") or len(raw.get("failed_requests") or [])),
        "timeout_count": int(raw.get("timeout_count") or 0),
        "telemetry_missing_count": int(raw.get("telemetry_missing_count") or 0),
        "feedback_enabled_rate": feedback_enabled_rate,
        "candidate_pool_size": candidate_pool_size,
        "llm_provider": provider,
        "local_llm_model": local_model,
        "provider_mismatch_detected": provider_mismatch,
        "avg_generation_latency_ms": _avg(events, "generation_latency_ms"),
        "p95_generation_latency_ms": _p95(events, "generation_latency_ms"),
        "max_generation_latency_ms": staging._max(events, "generation_latency_ms"),
        "privacy_leak_detected": bool(raw.get("privacy_leak_detected")),
    }


def _forbidden_tokens_detected_count(payload: dict[str, Any]) -> int:
    text = json.dumps(payload, ensure_ascii=False).lower()
    return sum(1 for token in FORBIDDEN_REPORT_TOKENS if token.lower() in text)


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def recommend_gemma4_suite(report: dict[str, Any]) -> str:
    blockers = report.setdefault("blockers", [])
    warnings = report.setdefault("warnings", [])
    fallback_threshold = 0.05

    reason_counts = Counter()
    reason_counts.update(report.get("fallback_reason_counts_a") or {})
    reason_counts.update(report.get("fallback_reason_counts_b") or {})
    reason_counts.pop("none", None)
    if reason_counts:
        dominant_reason, _count = reason_counts.most_common(1)[0]
        if dominant_reason == "schema_error_unrecoverable":
            _append_unique(warnings, "dominant_fallback_reason_schema_error_unrecoverable")
    attempted_total = int(report.get("schema_repair_attempted_count_a") or 0) + int(report.get("schema_repair_attempted_count_b") or 0)
    succeeded_total = int(report.get("schema_repair_succeeded_count_a") or 0) + int(report.get("schema_repair_succeeded_count_b") or 0)
    if attempted_total > 0 and succeeded_total == 0:
        _append_unique(warnings, "schema_repair_zero_success")

    length_a = int((report.get("raw_invoke_finish_reason_counts_a") or {}).get("length") or 0)
    length_b = int((report.get("raw_invoke_finish_reason_counts_b") or {}).get("length") or 0)
    if length_a + length_b > 0:
        _append_unique(warnings, "dominant_finish_reason_length")
    if length_b > length_a:
        _append_unique(warnings, "treatment_length_finish_spike")

    if float(report.get("fallback_rate_a") or 0) > fallback_threshold:
        _append_unique(blockers, "control_fallback_rate")
    if float(report.get("fallback_rate_b") or 0) > float(report.get("fallback_rate_a") or 0) + fallback_threshold:
        _append_unique(blockers, "treatment_fallback_rate_spike")

    if not report.get("readiness_passed"):
        _append_unique(blockers, "readiness_failed")
        return "inconclusive_gemma4_readiness_failed"
    if report.get("provider_mismatch_detected"):
        _append_unique(blockers, "provider_mismatch")
        return "inconclusive_gemma4_readiness_failed"
    if report.get("privacy_leak_detected"):
        _append_unique(blockers, "privacy_leak")
        return "keep_gemma4_only_off"
    if (
        int(report.get("request_count_a") or 0) < int(report.get("min_requests_per_group") or 0)
        or int(report.get("request_count_b") or 0) < int(report.get("min_requests_per_group") or 0)
    ):
        _append_unique(blockers, "insufficient_sample")
        return "inconclusive_insufficient_sample"
    failed_a = int(report.get("failed_requests_a") or 0)
    failed_b = int(report.get("failed_requests_b") or 0)
    timeout_a = int(report.get("timeout_count_a") or 0)
    timeout_b = int(report.get("timeout_count_b") or 0)
    if failed_a > 0 or failed_b > 0:
        _append_unique(blockers, "failed_requests")
        if timeout_a + timeout_b > 0:
            return "inconclusive_gemma4_timeout"
        return "inconclusive"
    if timeout_a > 0 or timeout_b > 0:
        _append_unique(blockers, "timeouts")
        return "inconclusive_gemma4_timeout"
    if int(report.get("telemetry_missing_count_a") or 0) > 0 or int(report.get("telemetry_missing_count_b") or 0) > 0:
        _append_unique(blockers, "telemetry_missing")
        return "inconclusive"
    if int(report.get("hard_violation_count_a") or 0) > 0 or int(report.get("hard_violation_count_b") or 0) > 0:
        _append_unique(blockers, "hard_violations")
        return "keep_gemma4_only_off"
    if float(report.get("fallback_rate_a") or 0) > fallback_threshold:
        return "inconclusive"
    if float(report.get("fallback_rate_b") or 0) > float(report.get("fallback_rate_a") or 0) + fallback_threshold:
        return "keep_gemma4_only_off"
    if float(report.get("quota_fallback_rate_a") or 0) > 0 or float(report.get("quota_fallback_rate_b") or 0) > 0:
        _append_unique(blockers, "quota_fallback")
        return "inconclusive"
    if float(report.get("feedback_query_failed_rate_b") or 0) > 0:
        _append_unique(blockers, "feedback_query_failure")
        return "keep_gemma4_only_off"
    if float(report.get("p95_feedback_query_latency_ms_b") or 0) > 50:
        _append_unique(blockers, "feedback_query_latency")
        return "keep_gemma4_only_off"
    if float(report.get("avg_feedback_adjusted_count_b") or 0) < 1:
        _append_unique(blockers, "no_feedback_adjustment")
        return "inconclusive_no_feedback_adjustment"
    if float(report.get("avg_critic_score_b") or 0) < float(report.get("avg_critic_score_a") or 0) - 5:
        _append_unique(blockers, "critic_score_drop")
        return "keep_gemma4_only_off"
    if float(report.get("p95_generation_latency_ms_a") or 0) > 60000 or float(report.get("p95_generation_latency_ms_b") or 0) > 60000:
        _append_unique(warnings, "p95_generation_latency_over_60000_ms")
        return "inconclusive"
    return "promote_gemma4_only_candidate"


def _next_action(recommendation: str) -> str:
    if recommendation == "promote_gemma4_only_candidate":
        return "Proceed to controlled Gemma4-only rollout planning."
    if recommendation.startswith("keep_"):
        return "Keep Gemma4-only production cutover disabled and investigate blockers."
    return "Collect another suite run after resolving inconclusive conditions."


def _markdown_count_table(title: str, a_counts: dict[str, int] | None, b_counts: dict[str, int] | None) -> list[str]:
    a_counts = a_counts or {}
    b_counts = b_counts or {}
    keys = sorted(set(a_counts) | set(b_counts))
    lines = [f"### {title}", "", "| value | A | B |", "|---|---:|---:|"]
    if keys:
        lines.extend(f"| `{key}` | {a_counts.get(key, 0)} | {b_counts.get(key, 0)} |" for key in keys)
    else:
        lines.append("| none | 0 | 0 |")
    lines.append("")
    return lines


def _markdown_scenario_table(title: str, counts: dict[str, int] | None) -> list[str]:
    counts = counts or {}
    lines = [f"### {title}", "", "| scenario | fallback count |", "|---|---:|"]
    if counts:
        for scenario_id, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"| `{scenario_id}` | {count} |")
    else:
        lines.append("| none | 0 |")
    lines.append("")
    return lines


def _markdown_delta_table(delta: dict[str, int] | None) -> list[str]:
    delta = delta or {}
    lines = ["### B minus A fallback delta", "", "| scenario | delta |", "|---|---:|"]
    if delta:
        for scenario_id, value in sorted(delta.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"| `{scenario_id}` | {value} |")
    else:
        lines.append("| none | 0 |")
    lines.append("")
    return lines


def _scenario_names_where(delta: dict[str, int] | None, predicate) -> list[str]:
    return [scenario_id for scenario_id, value in sorted((delta or {}).items()) if predicate(value)]


def build_suite_report(
    *,
    run_id: str,
    output_dir: str,
    model: str,
    ollama_base_url: str,
    repeats: int,
    request_timeout_sec: int,
    min_requests_per_group: int,
    production_like: bool,
    compare_report: dict[str, Any],
    raw_a: dict[str, Any],
    raw_b: dict[str, Any],
    preflight: dict[str, Any] | None = None,
    require_local_provider: bool = True,
) -> dict[str, Any]:
    events_a = _events(raw_a)
    events_b = _events(raw_b)
    summary_a = build_group_summary(raw_a, group="a", model=model, require_local_provider=require_local_provider)
    summary_b = build_group_summary(raw_b, group="b", model=model, require_local_provider=require_local_provider)
    fallback_breakdown = build_fallback_breakdown(raw_a, raw_b)
    sanitized_base = local_readiness.sanitize_base_url_for_report(ollama_base_url)
    report: dict[str, Any] = {
        "run_id": run_id,
        "provider": "local",
        "model": model,
        "base_url_host": sanitized_base["base_url_host"],
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "repeats": repeats,
        "focused_run": bool(raw_a.get("focused_run") or raw_b.get("focused_run")),
        "selected_scenario_ids": raw_a.get("selected_scenario_ids") or raw_b.get("selected_scenario_ids") or [],
        "selected_scenario_count": raw_a.get("selected_scenario_count") or raw_b.get("selected_scenario_count") or 0,
        "purpose": raw_a.get("purpose") or raw_b.get("purpose"),
        "request_timeout_sec": request_timeout_sec,
        "min_requests_per_group": min_requests_per_group,
        "production_like": production_like,
        "ollama_readiness_status": (preflight or {}).get("ollama_readiness_status"),
        "ollama_reachable": (preflight or {}).get("ollama_reachable"),
        "model_installed": (preflight or {}).get("model_installed"),
        "probe_success": (preflight or {}).get("probe_success"),
        "probe_latency_ms": (preflight or {}).get("probe_latency_ms"),
        "db_readiness_status": compare_report.get("db_readiness_status_a") or (preflight or {}).get("db_readiness_status"),
        "seed_readiness_status": compare_report.get("seed_readiness_status_a") or (preflight or {}).get("seed_readiness_status"),
        "seed_verification_passed": compare_report.get("seed_verification_passed_a", (preflight or {}).get("seed_verification_passed")),
        "readiness_passed": bool((preflight or {}).get("readiness_passed", True)),
        "request_count_a": summary_a["request_count"],
        "request_count_b": summary_b["request_count"],
        "failed_requests_a": summary_a["failed_requests"],
        "failed_requests_b": summary_b["failed_requests"],
        "timeout_count_a": int(compare_report.get("timeout_count_a") or 0),
        "timeout_count_b": int(compare_report.get("timeout_count_b") or 0),
        "telemetry_missing_count_a": int(compare_report.get("telemetry_missing_count_a") or 0),
        "telemetry_missing_count_b": int(compare_report.get("telemetry_missing_count_b") or 0),
        "feedback_enabled_rate_a": compare_report.get("feedback_enabled_rate_a"),
        "feedback_enabled_rate_b": compare_report.get("feedback_enabled_rate_b"),
        "candidate_pool_size_a": summary_a["candidate_pool_size"],
        "candidate_pool_size_b": summary_b["candidate_pool_size"],
        "provider_mismatch_detected": bool(summary_a["provider_mismatch_detected"] or summary_b["provider_mismatch_detected"]),
        "fallback_rate_a": compare_report.get("fallback_rate_a"),
        "fallback_rate_b": compare_report.get("fallback_rate_b"),
        "quota_fallback_rate_a": compare_report.get("quota_fallback_rate_a"),
        "quota_fallback_rate_b": compare_report.get("quota_fallback_rate_b"),
        **fallback_breakdown,
        "hard_violation_count_a": compare_report.get("hard_violation_count_a"),
        "hard_violation_count_b": compare_report.get("hard_violation_count_b"),
        "hard_violation_category_counts_a": compare_report.get("hard_violation_category_counts_a"),
        "hard_violation_category_counts_b": compare_report.get("hard_violation_category_counts_b"),
        "avg_critic_score_a": compare_report.get("avg_critic_score_a"),
        "avg_critic_score_b": compare_report.get("avg_critic_score_b"),
        "critic_score_delta": round(float(compare_report.get("avg_critic_score_b") or 0) - float(compare_report.get("avg_critic_score_a") or 0), 4),
        "critic_grade_distribution_a": compare_report.get("critic_grade_distribution_a"),
        "critic_grade_distribution_b": compare_report.get("critic_grade_distribution_b"),
        "avg_generation_latency_ms_a": _avg(events_a, "generation_latency_ms"),
        "avg_generation_latency_ms_b": _avg(events_b, "generation_latency_ms"),
        "p95_generation_latency_ms_a": compare_report.get("p95_latency_a", _p95(events_a, "generation_latency_ms")),
        "p95_generation_latency_ms_b": compare_report.get("p95_latency_b", _p95(events_b, "generation_latency_ms")),
        "max_generation_latency_ms_a": compare_report.get("max_latency_a", staging._max(events_a, "generation_latency_ms")),
        "max_generation_latency_ms_b": compare_report.get("max_latency_b", staging._max(events_b, "generation_latency_ms")),
        "avg_feedback_adjusted_count_b": compare_report.get("avg_feedback_adjusted_count_b"),
        "feedback_query_failed_rate_b": compare_report.get("feedback_query_failed_rate_b"),
        "p95_feedback_query_latency_ms_b": compare_report.get("p95_feedback_query_latency_ms_b"),
        "privacy_leak_detected": bool(compare_report.get("privacy_leak_detected") or raw_a.get("privacy_leak_detected") or raw_b.get("privacy_leak_detected")),
        "forbidden_tokens_detected_count": 0,
        "lower_level_recommendation": compare_report.get("recommendation"),
        "blockers": [],
        "warnings": [],
    }
    report["forbidden_tokens_detected_count"] = _forbidden_tokens_detected_count(report)
    if report["forbidden_tokens_detected_count"] > 0:
        report["privacy_leak_detected"] = True
    report["recommendation"] = recommend_gemma4_suite(report)
    report["next_action"] = _next_action(report["recommendation"])
    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    dominant_schema_error = (
        (report.get("fallback_reason_counts_a") or {}).get("schema_error_unrecoverable", 0)
        + (report.get("fallback_reason_counts_b") or {}).get("schema_error_unrecoverable", 0)
    ) > 0
    repair_attempts = int(report.get("schema_repair_attempted_count_a") or 0) + int(report.get("schema_repair_attempted_count_b") or 0)
    repair_successes = int(report.get("schema_repair_succeeded_count_a") or 0) + int(report.get("schema_repair_succeeded_count_b") or 0)
    b_minus_a_delta = report.get("b_minus_a_fallback_delta_by_scenario") or {}
    b_only_increased = _scenario_names_where(b_minus_a_delta, lambda value: value > 0)
    common_unstable = [
        scenario_id
        for scenario_id, count_a in sorted((report.get("fallback_count_by_scenario_a") or {}).items())
        if count_a > 0 and (report.get("fallback_count_by_scenario_b") or {}).get(scenario_id, 0) > 0
    ]
    lines = [
            "# Gemma4 Evaluation Suite Report",
            "",
            "## Summary",
            f"- recommendation: `{report['recommendation']}`",
            f"- next action: {report['next_action']}",
            f"- lower-level recommendation: `{report.get('lower_level_recommendation')}`",
            "",
            "## Environment",
            f"- provider: `{report['provider']}`",
            f"- model: `{report['model']}`",
            f"- Ollama host: `{report['base_url_host']}`",
            f"- repeats: {report['repeats']}",
            f"- request timeout sec: {report['request_timeout_sec']}",
            f"- production-like: `{report['production_like']}`",
            f"- focused run: `{report.get('focused_run')}`",
            f"- selected scenario count: {report.get('selected_scenario_count')}",
            f"- selected scenario ids: `{report.get('selected_scenario_ids')}`",
            f"- purpose: `{report.get('purpose')}`",
            "",
            "## Readiness",
            f"- Ollama readiness: `{report.get('ollama_readiness_status')}`",
            f"- DB readiness: `{report.get('db_readiness_status')}`",
            f"- seed readiness: `{report.get('seed_readiness_status')}`",
            f"- readiness passed: `{report.get('readiness_passed')}`",
            "",
            "## A/B Metrics",
            "| metric | A | B |",
            "|---|---:|---:|",
            f"| requests | {report['request_count_a']} | {report['request_count_b']} |",
            f"| failed requests | {report['failed_requests_a']} | {report['failed_requests_b']} |",
            f"| timeouts | {report['timeout_count_a']} | {report['timeout_count_b']} |",
            f"| telemetry missing | {report['telemetry_missing_count_a']} | {report['telemetry_missing_count_b']} |",
            f"| fallback rate | {report['fallback_rate_a']} | {report['fallback_rate_b']} |",
            f"| avg critic score | {report['avg_critic_score_a']} | {report['avg_critic_score_b']} |",
            f"| p95 generation latency ms | {report['p95_generation_latency_ms_a']} | {report['p95_generation_latency_ms_b']} |",
            "",
            "## Fallback Breakdown",
            "",
            f"- fallback rate A/B: {report.get('fallback_rate_a')} / {report.get('fallback_rate_b')}",
            "",
        ]
    lines.extend(_markdown_count_table("Fallback reasons", report.get("fallback_reason_counts_a"), report.get("fallback_reason_counts_b")))
    lines.extend(_markdown_count_table("LLM error types", report.get("llm_error_type_counts_a"), report.get("llm_error_type_counts_b")))
    lines.extend(
        _markdown_count_table(
            "Parse failure subtypes",
            report.get("parse_failure_subtype_counts_a"),
            report.get("parse_failure_subtype_counts_b"),
        )
    )
    lines.extend(
        _markdown_count_table(
            "Schema validation error categories",
            report.get("schema_validation_error_category_counts_a"),
            report.get("schema_validation_error_category_counts_b"),
        )
    )
    lines.extend(
        _markdown_count_table(
            "Repair failure reasons",
            report.get("repair_failure_reason_counts_a"),
            report.get("repair_failure_reason_counts_b"),
        )
    )
    lines.extend(
        [
            "## JSON Decode Failure Breakdown",
            "",
        ]
    )
    lines.extend(
        _markdown_count_table(
            "JSON decode error categories",
            report.get("json_decode_error_category_counts_a"),
            report.get("json_decode_error_category_counts_b"),
        )
    )
    lines.extend(
        [
            "## JSON Decode Recovery",
            "",
            "| metric | A | B |",
            "|---|---:|---:|",
            f"| attempted count | {report.get('json_decode_recovery_attempted_count_a')} | {report.get('json_decode_recovery_attempted_count_b')} |",
            f"| succeeded count | {report.get('json_decode_recovery_succeeded_count_a')} | {report.get('json_decode_recovery_succeeded_count_b')} |",
            f"| strategy counts | `{report.get('json_decode_recovery_strategy_counts_a')}` | `{report.get('json_decode_recovery_strategy_counts_b')}` |",
            "",
        ]
    )
    lines.extend(
        [
            "## Local Raw JSON Invoke",
            "",
            "| metric | A | B |",
            "|---|---:|---:|",
            f"| used count | {report.get('local_raw_json_invoke_used_count_a')} | {report.get('local_raw_json_invoke_used_count_b')} |",
            f"| succeeded count | {report.get('local_raw_json_invoke_succeeded_count_a')} | {report.get('local_raw_json_invoke_succeeded_count_b')} |",
            f"| structured output bypassed | {report.get('structured_output_bypassed_count_a')} | {report.get('structured_output_bypassed_count_b')} |",
            f"| failed reason counts | `{report.get('local_raw_json_invoke_failed_reason_counts_a')}` | `{report.get('local_raw_json_invoke_failed_reason_counts_b')}` |",
            "",
            "### Raw invoke response shape",
            "",
            "| metric | A | B |",
            "|---|---:|---:|",
            f"| response class counts | `{report.get('raw_invoke_response_class_counts_a')}` | `{report.get('raw_invoke_response_class_counts_b')}` |",
            f"| content present count | {report.get('raw_invoke_content_present_count_a')} | {report.get('raw_invoke_content_present_count_b')} |",
            f"| stripped empty count | {report.get('raw_invoke_stripped_empty_count_a')} | {report.get('raw_invoke_stripped_empty_count_b')} |",
            f"| finish reason counts | `{report.get('raw_invoke_finish_reason_counts_a')}` | `{report.get('raw_invoke_finish_reason_counts_b')}` |",
            f"| done reason counts | `{report.get('raw_invoke_done_reason_counts_a')}` | `{report.get('raw_invoke_done_reason_counts_b')}` |",
            f"| error category counts | `{report.get('raw_invoke_error_category_counts_a')}` | `{report.get('raw_invoke_error_category_counts_b')}` |",
            "",
        ]
    )
    lines.extend(
        [
            "## Prompt Size and Local Budget",
            "",
            "| metric | A | B |",
            "|---|---:|---:|",
            f"| prompt char avg | {report.get('prompt_char_count_avg_a')} | {report.get('prompt_char_count_avg_b')} |",
            f"| prompt char p95 | {report.get('prompt_char_count_p95_a')} | {report.get('prompt_char_count_p95_b')} |",
            f"| prompt approx token avg | {report.get('prompt_approx_tokens_avg_a')} | {report.get('prompt_approx_tokens_avg_b')} |",
            f"| prompt approx token p95 | {report.get('prompt_approx_tokens_p95_a')} | {report.get('prompt_approx_tokens_p95_b')} |",
            f"| local num_predict | {report.get('local_llm_num_predict_a')} | {report.get('local_llm_num_predict_b')} |",
            f"| local num_ctx | {report.get('local_llm_num_ctx_a')} | {report.get('local_llm_num_ctx_b')} |",
            "",
        ]
    )
    lines.extend(
        [
            "## Schema Parse Failure Breakdown",
            "",
            f"- parse failure subtype counts A: `{report.get('parse_failure_subtype_counts_a')}`",
            f"- parse failure subtype counts B: `{report.get('parse_failure_subtype_counts_b')}`",
            f"- schema validation error category counts A: `{report.get('schema_validation_error_category_counts_a')}`",
            f"- schema validation error category counts B: `{report.get('schema_validation_error_category_counts_b')}`",
            "",
            "## Repair Effectiveness",
            "",
            "| metric | A | B |",
            "|---|---:|---:|",
            f"| attempted count | {report.get('schema_repair_attempted_count_a')} | {report.get('schema_repair_attempted_count_b')} |",
            f"| succeeded count | {report.get('schema_repair_succeeded_count_a')} | {report.get('schema_repair_succeeded_count_b')} |",
            f"| attempted rate | {report.get('schema_repair_attempted_rate_a')} | {report.get('schema_repair_attempted_rate_b')} |",
            f"| succeeded rate | {report.get('schema_repair_succeeded_rate_a')} | {report.get('schema_repair_succeeded_rate_b')} |",
            f"| success given attempt rate | {report.get('schema_repair_success_given_attempt_rate_a')} | {report.get('schema_repair_success_given_attempt_rate_b')} |",
            f"| raw output recovery attempted | {report.get('raw_output_recovery_attempted_count_a')} | {report.get('raw_output_recovery_attempted_count_b')} |",
            f"| raw output recovery succeeded | {report.get('raw_output_recovery_succeeded_count_a')} | {report.get('raw_output_recovery_succeeded_count_b')} |",
            "",
            "### Raw output recovery",
            "",
            f"- source counts A: `{report.get('raw_output_recovery_source_counts_a')}`",
            f"- source counts B: `{report.get('raw_output_recovery_source_counts_b')}`",
            f"- failed reason counts A: `{report.get('raw_output_recovery_failed_reason_counts_a')}`",
            f"- failed reason counts B: `{report.get('raw_output_recovery_failed_reason_counts_b')}`",
            "",
            "## Scenario-level Fallbacks",
            "",
        ]
    )
    lines.extend(_markdown_scenario_table("Top fallback scenarios A", report.get("top_fallback_scenarios_a")))
    lines.extend(_markdown_scenario_table("Top fallback scenarios B", report.get("top_fallback_scenarios_b")))
    lines.extend(_markdown_delta_table(b_minus_a_delta))
    lines.extend(
        [
            "### Common unstable scenarios",
            "",
            ", ".join(f"`{scenario_id}`" for scenario_id in common_unstable) if common_unstable else "none",
            "",
            "### B-only increased scenarios",
            "",
            ", ".join(f"`{scenario_id}`" for scenario_id in b_only_increased) if b_only_increased else "none",
            "",
            "## Interpretation",
            "",
            "- Primary failure mode is Gemma4 structured output/schema parse failure."
            if dominant_schema_error
            else "- No dominant schema parse fallback reason was detected.",
            "- Schema repair is currently not recovering these outputs."
            if repair_attempts > 0 and repair_successes == 0
            else "- Schema repair had at least one success or was not attempted.",
            "",
            "## Hard Violations",
            f"- A count: {report['hard_violation_count_a']}",
            f"- B count: {report['hard_violation_count_b']}",
            f"- A categories: `{report['hard_violation_category_counts_a']}`",
            f"- B categories: `{report['hard_violation_category_counts_b']}`",
            "",
            "## Feedback Ranking",
            f"- feedback enabled rate A/B: {report['feedback_enabled_rate_a']} / {report['feedback_enabled_rate_b']}",
            f"- avg adjusted candidates B: {report['avg_feedback_adjusted_count_b']}",
            f"- feedback query failed rate B: {report['feedback_query_failed_rate_b']}",
            f"- p95 feedback query latency B: {report['p95_feedback_query_latency_ms_b']} ms",
            "",
            "## Privacy",
            f"- privacy leak detected: `{report['privacy_leak_detected']}`",
            f"- forbidden tokens detected count: {report['forbidden_tokens_detected_count']}",
            "",
            "## Blockers",
            f"`{report['blockers']}`",
            "",
            "## Warnings",
            f"`{report['warnings']}`",
            "",
        ]
    )
    return "\n".join(lines)


def provider_check_report(*, production_like: bool = False) -> dict[str, Any]:
    config = resolve_llm_provider()
    status = "pass"
    if production_like:
        if not (
            config.is_production
            and config.requested_provider == "local"
            and config.local_allowed_in_production
            and config.effective_provider == "local"
        ):
            status = "failed_production_local_opt_in_missing"
    elif config.is_production and config.requested_provider == "local" and config.effective_provider == "local":
        status = "pass"

    return {
        "app_env": os.getenv("APP_ENV", "").strip().lower() or None,
        "requested_provider": config.requested_provider,
        "effective_provider": config.effective_provider,
        "local_allowed_in_production": config.local_allowed_in_production,
        "provider_check_status": status,
        "blocked_reason": config.blocked_reason,
    }


def cmd_preflight(args: argparse.Namespace) -> None:
    run_id = _sanitize_run_id(args.run_id)
    paths = _suite_paths(args.output_dir, run_id)
    report = build_preflight_report(
        run_id=run_id,
        base_url=args.base_url,
        ollama_base_url=args.ollama_base_url,
        model=args.model,
        output_dir=args.output_dir,
        timeout_sec=args.request_timeout,
        no_seed_reset=args.no_seed_reset,
        skip_readiness_check=args.skip_readiness_check,
    )
    _write_json(paths["preflight"], report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.strict and not report["readiness_passed"]:
        raise SystemExit(1)


def _run_group(args: argparse.Namespace, group: str) -> None:
    run_id = _sanitize_run_id(args.run_id)
    paths = _suite_paths(args.output_dir, run_id)
    output_path = paths["group_a"] if group == "a" else paths["group_b"]
    with _temporary_gemma4_metadata_env(args.model, args.ollama_base_url):
        staging.cmd_run(
            argparse.Namespace(
                group=group,
                base_url=args.base_url,
                output=str(output_path),
                repeats=args.repeats,
                user_id=staging.DEFAULT_USER_ID,
                request_timeout=args.request_timeout,
                scenario_ids=getattr(args, "scenario_ids", None),
            )
        )
    raw = _read_json(output_path)
    summary = build_group_summary(raw, group=group, model=args.model, require_local_provider=args.require_local_provider)
    summary_path = paths["group_a_summary"] if group == "a" else paths["group_b_summary"]
    _write_json(summary_path, summary)
    print(json.dumps({"output": str(output_path), "summary": str(summary_path), **summary}, indent=2, ensure_ascii=False))
    if args.strict and summary["provider_mismatch_detected"]:
        raise SystemExit(1)


def cmd_run_a(args: argparse.Namespace) -> None:
    _run_group(args, "a")


def cmd_run_b(args: argparse.Namespace) -> None:
    _run_group(args, "b")


def cmd_compare(args: argparse.Namespace) -> None:
    run_id = _sanitize_run_id(args.run_id)
    paths = _suite_paths(args.output_dir, run_id)
    suite_json = Path(args.json) if args.json else paths["suite_json"]
    suite_md = Path(args.markdown) if args.markdown else paths["suite_md"]
    staging.cmd_compare(
        argparse.Namespace(
            a=str(paths["group_a"]),
            b=str(paths["group_b"]),
            output=str(paths["compare"]),
            markdown=str(paths["compare_md"]),
        )
    )
    compare_report = _read_json(paths["compare"])
    raw_a = _read_json(paths["group_a"])
    raw_b = _read_json(paths["group_b"])
    preflight = _read_json(paths["preflight"]) if paths["preflight"].exists() else None
    report = build_suite_report(
        run_id=run_id,
        output_dir=args.output_dir,
        model=args.model,
        ollama_base_url=args.ollama_base_url,
        repeats=args.repeats,
        request_timeout_sec=args.request_timeout,
        min_requests_per_group=args.min_requests_per_group,
        production_like=args.production_like,
        compare_report=compare_report,
        raw_a=raw_a,
        raw_b=raw_b,
        preflight=preflight,
        require_local_provider=args.require_local_provider,
    )
    _write_json(suite_json, report)
    _write_markdown(suite_md, render_markdown_report(report))
    print(json.dumps({"report": str(suite_json), "markdown": str(suite_md), "recommendation": report["recommendation"]}, indent=2))
    if args.strict and report["recommendation"] != "promote_gemma4_only_candidate":
        raise SystemExit(1)


def cmd_provider_check(args: argparse.Namespace) -> None:
    run_id = _sanitize_run_id(args.run_id)
    paths = _suite_paths(args.output_dir, run_id)
    report = provider_check_report(production_like=args.production_like)
    _write_json(paths["provider_check"], report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.strict and report["provider_check_status"] != "pass":
        raise SystemExit(1)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--ollama-base-url", default=os.getenv("OLLAMA_BASE_URL", local_readiness.DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.getenv("LOCAL_LLM_MODEL", local_readiness.DEFAULT_MODEL))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id")
    parser.add_argument("--strict", action="store_true")


def _add_run_common(parser: argparse.ArgumentParser) -> None:
    _add_common(parser)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--require-local-provider", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scenario-ids", help="Comma-separated scenario ids for a focused smoke run.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gemma4-only evaluation suite orchestration.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    _add_common(preflight)
    preflight.add_argument("--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT)
    preflight.add_argument("--skip-readiness-check", action="store_true")
    preflight.add_argument("--no-seed-reset", action="store_true")
    preflight.set_defaults(func=cmd_preflight)

    run_a = subparsers.add_parser("run-a")
    _add_run_common(run_a)
    run_a.set_defaults(func=cmd_run_a)

    run_b = subparsers.add_parser("run-b")
    _add_run_common(run_b)
    run_b.set_defaults(func=cmd_run_b)

    compare = subparsers.add_parser("compare")
    _add_run_common(compare)
    compare.add_argument("--min-requests-per-group", type=int, default=DEFAULT_MIN_REQUESTS_PER_GROUP)
    compare.add_argument("--production-like", action="store_true")
    compare.add_argument("--markdown")
    compare.add_argument("--json")
    compare.set_defaults(func=cmd_compare)

    provider_check = subparsers.add_parser("provider-check")
    _add_common(provider_check)
    provider_check.add_argument("--production-like", action="store_true")
    provider_check.set_defaults(func=cmd_provider_check)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import run_gemma4_evaluation_suite as suite


def _event(**overrides):
    payload = {
        "feedback_enabled": False,
        "candidate_pool_size": 12,
        "generation_latency_ms": 100,
        "prompt_char_count": 1000,
        "prompt_approx_tokens": 250,
        "local_llm_num_predict": None,
        "local_llm_num_ctx": None,
        "fallback_used": False,
        "fallback_reason": "none",
        "llm_error_type": None,
        "schema_repair_attempted": False,
        "schema_repair_succeeded": False,
        "parse_failure_subtype": None,
        "schema_validation_error_category": None,
        "schema_validation_field_names": [],
        "repair_failure_reason": None,
        "raw_output_recovery_attempted": False,
        "raw_output_recovery_succeeded": False,
        "raw_output_recovery_source": None,
        "raw_output_recovery_failed_reason": None,
        "json_decode_error_category": None,
        "json_decode_recovery_attempted": False,
        "json_decode_recovery_succeeded": False,
        "json_decode_recovery_strategy": None,
        "local_raw_json_invoke_enabled": False,
        "local_raw_json_invoke_used": False,
        "local_raw_json_invoke_succeeded": False,
        "local_raw_json_invoke_failed_reason": None,
        "structured_output_bypassed": False,
        "raw_invoke_response_class": None,
        "raw_invoke_content_present": False,
        "raw_invoke_content_type": None,
        "raw_invoke_content_length_bucket": None,
        "raw_invoke_content_stripped_empty": False,
        "raw_invoke_has_response_metadata": False,
        "raw_invoke_finish_reason": None,
        "raw_invoke_done_reason": None,
        "raw_invoke_error_category": None,
        "raw_invoke_usage_present": False,
        "critic_score": 80,
        "critic_grade": "PASS",
        "hard_violation_count": 0,
        "feedback_adjusted_candidate_count": 0,
        "feedback_query_failed": False,
        "feedback_query_latency_ms": 1,
    }
    payload.update(overrides)
    return payload


def _fallback_event(**overrides):
    payload = _event(
        fallback_used=True,
        fallback_reason="schema_error_unrecoverable",
        llm_error_type="schema_parse_error",
        schema_repair_attempted=True,
        schema_repair_succeeded=False,
        parse_failure_subtype="json_decode_error",
        repair_failure_reason="json_decode_error",
        raw_output_recovery_attempted=True,
        raw_output_recovery_succeeded=False,
        raw_output_recovery_source="none",
        raw_output_recovery_failed_reason="no_raw_content",
        json_decode_error_category="json_decode_unknown",
        json_decode_recovery_attempted=False,
        json_decode_recovery_succeeded=False,
        json_decode_recovery_strategy="none",
        critic_grade="FAIL",
        critic_score=40,
    )
    payload.update(overrides)
    return payload


def _raw(group: str, events: list[dict] | None = None, **overrides):
    events = events if events is not None else [_event(feedback_enabled=(group == "b")) for _ in range(50)]
    payload = {
        "mode": "feedback_ranking_staging",
        "group": group,
        "llm_provider": "local",
        "local_llm_model": "gemma4:latest",
        "request_count": len(events),
        "failed_request_count": 0,
        "timeout_count": 0,
        "telemetry_missing_count": 0,
        "failed_requests": [],
        "results": [{"scenario_id": f"{group}-{index}", "telemetry": event} for index, event in enumerate(events)],
        "privacy_leak_detected": False,
    }
    payload.update(overrides)
    return payload


def _compare(**overrides):
    payload = {
        "request_count_a": 50,
        "request_count_b": 50,
        "submitted_request_count_a": 50,
        "submitted_request_count_b": 50,
        "failed_request_count_a": 0,
        "failed_request_count_b": 0,
        "timeout_count_a": 0,
        "timeout_count_b": 0,
        "telemetry_missing_count_a": 0,
        "telemetry_missing_count_b": 0,
        "feedback_enabled_rate_a": 0.0,
        "feedback_enabled_rate_b": 1.0,
        "fallback_rate_a": 0.0,
        "fallback_rate_b": 0.0,
        "quota_fallback_rate_a": 0.0,
        "quota_fallback_rate_b": 0.0,
        "hard_violation_count_a": 0,
        "hard_violation_count_b": 0,
        "hard_violation_category_counts_a": {},
        "hard_violation_category_counts_b": {},
        "avg_critic_score_a": 80.0,
        "avg_critic_score_b": 80.0,
        "critic_grade_distribution_a": {"PASS": 50},
        "critic_grade_distribution_b": {"PASS": 50},
        "p95_latency_a": 100,
        "p95_latency_b": 100,
        "max_latency_a": 120,
        "max_latency_b": 120,
        "avg_feedback_adjusted_count_b": 2.0,
        "feedback_query_failed_rate_b": 0.0,
        "p95_feedback_query_latency_ms_b": 3,
        "privacy_leak_detected": False,
        "db_readiness_status_a": "pass",
        "seed_readiness_status_a": "pass",
        "seed_verification_passed_a": True,
        "recommendation": "promote_feedback_ranking_candidate",
    }
    payload.update(overrides)
    return payload


def _preflight(**overrides):
    payload = {
        "readiness_passed": True,
        "ollama_readiness_status": "pass",
        "ollama_reachable": True,
        "model_installed": True,
        "probe_success": True,
        "probe_latency_ms": 10,
        "db_readiness_status": "pass",
        "seed_readiness_status": "pass",
        "seed_verification_passed": True,
    }
    payload.update(overrides)
    return payload


def _suite_report(compare_report=None, raw_a=None, raw_b=None, preflight=None, **kwargs):
    return suite.build_suite_report(
        run_id="test-run",
        output_dir="out",
        model="gemma4:latest",
        ollama_base_url="http://user:pass@127.0.0.1:11434/path?token=secret",
        repeats=5,
        request_timeout_sec=240,
        min_requests_per_group=50,
        production_like=False,
        compare_report=compare_report or _compare(),
        raw_a=raw_a or _raw("a"),
        raw_b=raw_b or _raw("b"),
        preflight=preflight or _preflight(),
        **kwargs,
    )


def test_preflight_report_schema_includes_readiness_fields(monkeypatch):
    monkeypatch.setattr(suite.local_readiness, "build_readiness_report", lambda **_kwargs: {
        "readiness_status": "pass",
        "ollama_reachable": True,
        "model_installed": True,
        "probe_success": True,
        "probe_latency_ms": 11,
        "error_category": None,
    })
    monkeypatch.setattr(suite.staging, "ensure_routine_feedback_table", lambda: {"db_readiness_status": "pass"})
    monkeypatch.setattr(suite.staging, "perform_seed", lambda **_kwargs: {"seed_readiness_status": "pass", "seed_verification_passed": True})
    monkeypatch.setattr(suite.staging, "verify_seed_readiness", lambda: {"seed_readiness_status": "pass", "seed_verification_passed": True})
    monkeypatch.setattr(suite, "_http_get_status", lambda _url: True)

    report = suite.build_preflight_report(
        run_id="run",
        base_url="http://test",
        ollama_base_url="http://ollama",
        model="gemma4:latest",
        output_dir="out",
        timeout_sec=1,
    )

    assert report["ollama_readiness_status"] == "pass"
    assert report["db_readiness_status"] == "pass"
    assert report["seed_verification_passed"] is True
    assert report["docs_reachable"] is True
    assert report["dev_logs_reachable"] is True
    assert report["readiness_passed"] is True


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"readiness_passed": False}, "inconclusive_gemma4_readiness_failed"),
        ({"provider_mismatch_detected": True}, "inconclusive_gemma4_readiness_failed"),
        ({"privacy_leak_detected": True}, "keep_gemma4_only_off"),
        ({"request_count_a": 49}, "inconclusive_insufficient_sample"),
        ({"timeout_count_b": 1}, "inconclusive_gemma4_timeout"),
        ({"telemetry_missing_count_a": 1}, "inconclusive"),
        ({"hard_violation_count_a": 1}, "keep_gemma4_only_off"),
        ({"fallback_rate_b": 0.08}, "keep_gemma4_only_off"),
        ({"feedback_query_failed_rate_b": 0.1}, "keep_gemma4_only_off"),
        ({"p95_feedback_query_latency_ms_b": 51}, "keep_gemma4_only_off"),
        ({"avg_feedback_adjusted_count_b": 0}, "inconclusive_no_feedback_adjustment"),
        ({"avg_critic_score_b": 74.0}, "keep_gemma4_only_off"),
    ],
)
def test_recommendation_blocking_cases(updates, expected):
    report = _suite_report()
    report.update(updates)

    assert suite.recommend_gemma4_suite(report) == expected


def test_all_pass_recommends_promote():
    report = _suite_report()

    assert report["recommendation"] == "promote_gemma4_only_candidate"


def test_p95_generation_latency_over_threshold_warns_and_is_inconclusive():
    report = _suite_report(compare_report=_compare(p95_latency_b=70000))

    assert report["recommendation"] == "inconclusive"
    assert "p95_generation_latency_over_60000_ms" in report["warnings"]


def test_markdown_report_generated():
    report = _suite_report()

    markdown = suite.render_markdown_report(report)

    assert "# Gemma4 Evaluation Suite Report" in markdown
    assert "A/B Metrics" in markdown
    assert "promote_gemma4_only_candidate" in markdown


def test_suite_report_includes_fallback_reason_and_error_counts():
    raw_a = _raw("a", events=[_event(), _fallback_event()])
    raw_b = _raw("b", events=[_fallback_event(feedback_enabled=True), _fallback_event(feedback_enabled=True)])
    report = _suite_report(
        compare_report=_compare(request_count_a=2, request_count_b=2, fallback_rate_a=0.5, fallback_rate_b=1.0),
        raw_a=raw_a,
        raw_b=raw_b,
    )

    assert report["fallback_reason_counts_a"] == {"none": 1, "schema_error_unrecoverable": 1}
    assert report["fallback_reason_counts_b"] == {"schema_error_unrecoverable": 2}
    assert report["llm_error_type_counts_a"] == {"None": 1, "schema_parse_error": 1}
    assert report["llm_error_type_counts_b"] == {"schema_parse_error": 2}


def test_suite_report_includes_fallback_count_by_scenario_and_delta():
    raw_a = _raw(
        "a",
        events=[
            _fallback_event(),
            _event(),
            _fallback_event(),
        ],
    )
    raw_a["results"][0]["scenario_id"] = "stable_in_b"
    raw_a["results"][1]["scenario_id"] = "b_spike"
    raw_a["results"][2]["scenario_id"] = "common_unstable"
    raw_b = _raw(
        "b",
        events=[
            _event(feedback_enabled=True),
            _fallback_event(feedback_enabled=True),
            _fallback_event(feedback_enabled=True),
            _fallback_event(feedback_enabled=True),
        ],
    )
    raw_b["results"][0]["scenario_id"] = "stable_in_b"
    raw_b["results"][1]["scenario_id"] = "b_spike"
    raw_b["results"][2]["scenario_id"] = "b_spike"
    raw_b["results"][3]["scenario_id"] = "common_unstable"

    report = _suite_report(
        compare_report=_compare(request_count_a=3, request_count_b=4, fallback_rate_a=2 / 3, fallback_rate_b=0.75),
        raw_a=raw_a,
        raw_b=raw_b,
    )

    assert report["fallback_count_by_scenario_a"] == {"common_unstable": 1, "stable_in_b": 1}
    assert report["fallback_count_by_scenario_b"] == {"b_spike": 2, "common_unstable": 1}
    assert report["b_minus_a_fallback_delta_by_scenario"] == {
        "b_spike": 2,
        "common_unstable": 0,
        "stable_in_b": -1,
    }


def test_suite_report_includes_reason_and_error_counts_by_scenario():
    raw_a = _raw("a", events=[_fallback_event(), _event()])
    raw_a["results"][0]["scenario_id"] = "parse_case"
    raw_a["results"][1]["scenario_id"] = "parse_case"
    raw_b = _raw("b", events=[_fallback_event(feedback_enabled=True, llm_error_type="validation_error")])
    raw_b["results"][0]["scenario_id"] = "validation_case"

    report = _suite_report(
        compare_report=_compare(request_count_a=2, request_count_b=1, fallback_rate_a=0.5, fallback_rate_b=1.0),
        raw_a=raw_a,
        raw_b=raw_b,
    )

    assert report["fallback_reason_counts_by_scenario_a"] == {
        "parse_case": {"schema_error_unrecoverable": 1}
    }
    assert report["llm_error_type_counts_by_scenario_b"] == {
        "validation_case": {"validation_error": 1}
    }


def test_suite_report_includes_schema_repair_counts_and_zero_attempt_rate():
    report = _suite_report(
        compare_report=_compare(request_count_a=1, request_count_b=1),
        raw_a=_raw("a", events=[_event()]),
        raw_b=_raw("b", events=[_event(feedback_enabled=True)]),
    )

    assert report["schema_repair_attempted_count_a"] == 0
    assert report["schema_repair_succeeded_count_b"] == 0
    assert report["schema_repair_success_given_attempt_rate_a"] == 0.0
    assert report["schema_repair_success_given_attempt_rate_b"] == 0.0


def test_suite_report_includes_schema_repair_success_given_attempt_rate():
    report = _suite_report(
        compare_report=_compare(request_count_a=2, request_count_b=2, fallback_rate_a=0.5, fallback_rate_b=0.5),
        raw_a=_raw("a", events=[_fallback_event(schema_repair_succeeded=True), _fallback_event()]),
        raw_b=_raw("b", events=[_fallback_event(feedback_enabled=True), _event(feedback_enabled=True)]),
    )

    assert report["schema_repair_attempted_count_a"] == 2
    assert report["schema_repair_succeeded_count_a"] == 1
    assert report["schema_repair_success_given_attempt_rate_a"] == 0.5
    assert report["schema_repair_success_given_attempt_rate_b"] == 0.0


def test_suite_report_includes_parse_and_repair_failure_breakdown():
    report = _suite_report(
        compare_report=_compare(request_count_a=2, request_count_b=2, fallback_rate_a=0.5, fallback_rate_b=0.5),
        raw_a=_raw(
            "a",
            events=[
                _fallback_event(parse_failure_subtype="json_decode_error", repair_failure_reason="json_decode_error"),
                _event(),
            ],
        ),
        raw_b=_raw(
            "b",
            events=[
                _fallback_event(
                    feedback_enabled=True,
                    parse_failure_subtype="pydantic_validation_error",
                    schema_validation_error_category="missing_required_field",
                    schema_validation_field_names=["target_reps"],
                    repair_failure_reason="missing_required_field",
                    raw_output_recovery_succeeded=True,
                    raw_output_recovery_source="raw_retry",
                    raw_output_recovery_failed_reason=None,
                    json_decode_error_category="json_decode_trailing_comma",
                    json_decode_recovery_attempted=True,
                    json_decode_recovery_succeeded=True,
                    json_decode_recovery_strategy="trailing_comma_removed",
                    local_raw_json_invoke_enabled=True,
                    local_raw_json_invoke_used=True,
                    local_raw_json_invoke_succeeded=False,
                    local_raw_json_invoke_failed_reason="json_decode_error",
                    structured_output_bypassed=True,
                    raw_invoke_response_class="AIMessage",
                    raw_invoke_content_present=True,
                    raw_invoke_content_type="str",
                    raw_invoke_content_length_bucket="short_lt_100",
                    raw_invoke_content_stripped_empty=False,
                    raw_invoke_finish_reason="stop",
                    raw_invoke_done_reason="stop",
                    raw_invoke_error_category=None,
                    prompt_char_count=1200,
                    prompt_approx_tokens=300,
                    local_llm_num_predict=4096,
                    local_llm_num_ctx=8192,
                ),
                _fallback_event(
                    feedback_enabled=True,
                    parse_failure_subtype=None,
                    repair_failure_reason=None,
                    local_raw_json_invoke_enabled=True,
                    local_raw_json_invoke_used=True,
                    local_raw_json_invoke_succeeded=True,
                    structured_output_bypassed=True,
                    raw_invoke_response_class="AIMessage",
                    raw_invoke_content_present=True,
                    raw_invoke_content_type="list",
                    raw_invoke_content_length_bucket="medium_lt_1000",
                    raw_invoke_content_stripped_empty=False,
                    raw_invoke_finish_reason="stop",
                    raw_invoke_done_reason="stop",
                    raw_invoke_error_category=None,
                    prompt_char_count=1800,
                    prompt_approx_tokens=450,
                    local_llm_num_predict=4096,
                    local_llm_num_ctx=8192,
                ),
            ],
        ),
    )

    assert report["parse_failure_subtype_counts_a"] == {"json_decode_error": 1}
    assert report["parse_failure_subtype_counts_b"] == {
        "pydantic_validation_error": 1,
        "unknown": 1,
    }
    assert report["parse_failure_subtype_counts_by_scenario_b"] == {
        "b-0": {"pydantic_validation_error": 1},
        "b-1": {"unknown": 1},
    }
    assert report["schema_validation_error_category_counts_b"] == {
        "missing_required_field": 1,
        "unknown": 1,
    }
    assert report["repair_failure_reason_counts_b"] == {
        "missing_required_field": 1,
        "unknown": 1,
    }
    assert report["raw_output_recovery_attempted_count_a"] == 1
    assert report["raw_output_recovery_attempted_count_b"] == 2
    assert report["raw_output_recovery_succeeded_count_a"] == 0
    assert report["raw_output_recovery_source_counts_b"] == {"none": 1, "raw_retry": 1}
    assert report["raw_output_recovery_failed_reason_counts_b"] == {
        "no_raw_content": 1,
        "none": 1,
    }
    assert report["json_decode_error_category_counts_b"] == {
        "json_decode_trailing_comma": 1,
        "json_decode_unknown": 1,
    }
    assert report["json_decode_error_category_counts_by_scenario_b"] == {
        "b-0": {"json_decode_trailing_comma": 1},
        "b-1": {"json_decode_unknown": 1},
    }
    assert report["json_decode_recovery_attempted_count_b"] == 1
    assert report["json_decode_recovery_succeeded_count_b"] == 1
    assert report["json_decode_recovery_strategy_counts_b"] == {
        "none": 1,
        "trailing_comma_removed": 1,
    }
    assert report["local_raw_json_invoke_used_count_b"] == 2
    assert report["local_raw_json_invoke_succeeded_count_b"] == 1
    assert report["local_raw_json_invoke_failed_reason_counts_b"] == {
        "json_decode_error": 1,
        "none": 1,
    }
    assert report["structured_output_bypassed_count_b"] == 2
    assert report["raw_invoke_response_class_counts_b"] == {"AIMessage": 2}
    assert report["raw_invoke_content_present_count_b"] == 2
    assert report["raw_invoke_stripped_empty_count_b"] == 0
    assert report["raw_invoke_finish_reason_counts_b"] == {"stop": 2}
    assert report["raw_invoke_done_reason_counts_b"] == {"stop": 2}
    assert report["raw_invoke_error_category_counts_b"] == {"none": 2}
    assert report["prompt_char_count_avg_b"] == 1500
    assert report["prompt_char_count_p95_b"] == 1800
    assert report["prompt_approx_tokens_avg_b"] == 375
    assert report["prompt_approx_tokens_p95_b"] == 450
    assert report["local_llm_num_predict_b"] == 4096
    assert report["local_llm_num_ctx_b"] == 8192


def test_focused_run_metadata_appears_in_report():
    raw_a = _raw(
        "a",
        events=[_event()],
        focused_run=True,
        selected_scenario_ids=["normal_hypertrophy_upper"],
        selected_scenario_count=1,
        purpose="schema_failure_subtype_smoke",
    )
    raw_b = _raw(
        "b",
        events=[_event(feedback_enabled=True)],
        focused_run=True,
        selected_scenario_ids=["normal_hypertrophy_upper"],
        selected_scenario_count=1,
        purpose="schema_failure_subtype_smoke",
    )

    report = _suite_report(
        compare_report=_compare(request_count_a=1, request_count_b=1),
        raw_a=raw_a,
        raw_b=raw_b,
    )

    assert report["focused_run"] is True
    assert report["selected_scenario_ids"] == ["normal_hypertrophy_upper"]
    assert report["selected_scenario_count"] == 1
    assert report["purpose"] == "schema_failure_subtype_smoke"
    assert report["recommendation"] == "inconclusive_insufficient_sample"
    assert "parse_failure_subtype_counts_a" in report


def test_markdown_report_includes_fallback_breakdown_and_scenario_sections():
    report = _suite_report(
        compare_report=_compare(request_count_a=1, request_count_b=1, fallback_rate_a=1.0, fallback_rate_b=1.0),
        raw_a=_raw("a", events=[_fallback_event()]),
        raw_b=_raw("b", events=[_fallback_event(feedback_enabled=True)]),
    )

    markdown = suite.render_markdown_report(report)

    assert "## Fallback Breakdown" in markdown
    assert "## Scenario-level Fallbacks" in markdown
    assert "## Schema Parse Failure Breakdown" in markdown
    assert "## Repair Effectiveness" in markdown
    assert "## JSON Decode Failure Breakdown" in markdown
    assert "## JSON Decode Recovery" in markdown
    assert "## Local Raw JSON Invoke" in markdown
    assert "## Prompt Size and Local Budget" in markdown
    assert "Primary failure mode is Gemma4 structured output/schema parse failure." in markdown
    assert "Schema repair is currently not recovering these outputs." in markdown


def test_json_report_contains_no_forbidden_privacy_keys():
    report = _suite_report()
    text = json.dumps(report, ensure_ascii=False)

    assert "user_note" not in text
    assert "raw_prompt" not in text
    assert "raw_llm_output" not in text
    assert "user:pass" not in text
    assert "token=secret" not in text


def test_fallback_breakdown_report_contains_no_raw_prompt_output_or_user_note():
    report = _suite_report(
        compare_report=_compare(request_count_a=1, request_count_b=1, fallback_rate_a=1.0, fallback_rate_b=1.0),
        raw_a=_raw("a", events=[_fallback_event()]),
        raw_b=_raw("b", events=[_fallback_event(feedback_enabled=True)]),
    )
    text = json.dumps(report, ensure_ascii=False).lower()

    assert "raw_prompt" not in text
    assert "raw output" not in text
    assert "raw model response" not in text
    assert "user_note" not in text


def test_dominant_schema_error_and_zero_repair_success_warnings_added():
    report = _suite_report(
        compare_report=_compare(request_count_a=1, request_count_b=1, fallback_rate_a=1.0, fallback_rate_b=1.0),
        raw_a=_raw("a", events=[_fallback_event()]),
        raw_b=_raw("b", events=[_fallback_event(feedback_enabled=True)]),
    )

    assert "dominant_fallback_reason_schema_error_unrecoverable" in report["warnings"]
    assert "schema_repair_zero_success" in report["warnings"]
    assert "control_fallback_rate" in report["blockers"]


def test_length_finish_warnings_added():
    report = _suite_report(
        compare_report=_compare(request_count_a=2, request_count_b=2, fallback_rate_a=0.5, fallback_rate_b=1.0),
        raw_a=_raw(
            "a",
            events=[
                _fallback_event(raw_invoke_finish_reason="length"),
                _event(raw_invoke_finish_reason="stop"),
            ],
        ),
        raw_b=_raw(
            "b",
            events=[
                _fallback_event(feedback_enabled=True, raw_invoke_finish_reason="length"),
                _fallback_event(feedback_enabled=True, raw_invoke_finish_reason="length"),
            ],
        ),
    )

    assert "dominant_finish_reason_length" in report["warnings"]
    assert "treatment_length_finish_spike" in report["warnings"]


def test_provider_check_production_opt_in_true_allows_local(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("ALLOW_LOCAL_LLM_IN_PRODUCTION", "true")

    report = suite.provider_check_report(production_like=True)

    assert report["effective_provider"] == "local"
    assert report["provider_check_status"] == "pass"


def test_provider_check_production_without_opt_in_blocks_local(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.delenv("ALLOW_LOCAL_LLM_IN_PRODUCTION", raising=False)

    report = suite.provider_check_report(production_like=True)

    assert report["effective_provider"] == "gemini"
    assert report["provider_check_status"] == "failed_production_local_opt_in_missing"
    assert report["blocked_reason"] == "production_local_not_allowed"


@pytest.mark.slow
@pytest.mark.integration
def test_cli_help_works():
    result = subprocess.run(
        [sys.executable, "scripts/run_gemma4_evaluation_suite.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "preflight" in result.stdout
    assert "provider-check" in result.stdout


def test_output_dir_paths_are_created_safely(tmp_path):
    path = tmp_path / "nested" / "report.json"

    suite._write_json(path, {"ok": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True}


def test_run_a_command_composition_includes_timeout_and_repeats(monkeypatch, tmp_path):
    seen = {}

    def fake_cmd_run(namespace):
        seen["group"] = namespace.group
        seen["repeats"] = namespace.repeats
        seen["request_timeout"] = namespace.request_timeout
        seen["scenario_ids"] = namespace.scenario_ids
        Path(namespace.output).write_text(json.dumps(_raw("a")), encoding="utf-8")

    monkeypatch.setattr(suite.staging, "cmd_run", fake_cmd_run)
    args = argparse.Namespace(
        run_id="run",
        output_dir=str(tmp_path),
        model="gemma4:latest",
        ollama_base_url="http://127.0.0.1:11434",
        base_url="http://backend",
        repeats=7,
        request_timeout=123,
        scenario_ids="normal_hypertrophy_upper,strength_lower",
        require_local_provider=True,
        strict=False,
    )

    suite.cmd_run_a(args)

    assert seen == {
        "group": "a",
        "repeats": 7,
        "request_timeout": 123,
        "scenario_ids": "normal_hypertrophy_upper,strength_lower",
    }
    assert (tmp_path / "group_a_summary.json").exists()


def test_run_b_command_composition_includes_timeout_and_repeats(monkeypatch, tmp_path):
    seen = {}

    def fake_cmd_run(namespace):
        seen["group"] = namespace.group
        seen["repeats"] = namespace.repeats
        seen["request_timeout"] = namespace.request_timeout
        seen["scenario_ids"] = namespace.scenario_ids
        Path(namespace.output).write_text(json.dumps(_raw("b")), encoding="utf-8")

    monkeypatch.setattr(suite.staging, "cmd_run", fake_cmd_run)
    args = argparse.Namespace(
        run_id="run",
        output_dir=str(tmp_path),
        model="gemma4:latest",
        ollama_base_url="http://127.0.0.1:11434",
        base_url="http://backend",
        repeats=9,
        request_timeout=240,
        scenario_ids=None,
        require_local_provider=True,
        strict=False,
    )

    suite.cmd_run_b(args)

    assert seen == {"group": "b", "repeats": 9, "request_timeout": 240, "scenario_ids": None}
    assert (tmp_path / "group_b_summary.json").exists()


def test_compare_writes_json_and_markdown(monkeypatch, tmp_path):
    paths = suite._suite_paths(tmp_path, "run")
    suite._write_json(paths["group_a"], _raw("a"))
    suite._write_json(paths["group_b"], _raw("b"))
    suite._write_json(paths["preflight"], _preflight())

    def fake_compare(namespace):
        Path(namespace.output).write_text(json.dumps(_compare()), encoding="utf-8")
        Path(namespace.markdown).write_text("# lower", encoding="utf-8")

    monkeypatch.setattr(suite.staging, "cmd_compare", fake_compare)
    args = argparse.Namespace(
        run_id="run",
        output_dir=str(tmp_path),
        json=None,
        markdown=None,
        model="gemma4:latest",
        ollama_base_url="http://127.0.0.1:11434",
        repeats=5,
        request_timeout=240,
        min_requests_per_group=50,
        production_like=False,
        require_local_provider=True,
        strict=False,
    )

    suite.cmd_compare(args)

    report = json.loads(paths["suite_json"].read_text(encoding="utf-8"))
    assert report["recommendation"] == "promote_gemma4_only_candidate"
    assert paths["suite_md"].exists()


def test_group_summary_detects_feedback_or_pool_mismatch():
    raw = _raw("a", events=[_event(feedback_enabled=True, candidate_pool_size=18)])

    summary = suite.build_group_summary(raw, group="a", model="gemma4:latest")

    assert summary["provider_mismatch_detected"] is True

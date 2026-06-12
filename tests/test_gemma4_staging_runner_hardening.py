from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from engines.routine_telemetry import classify_hard_violation_category
from scripts import run_feedback_ranking_staging_test as staging


def _event(
    *,
    feedback_enabled: bool,
    critic_score: int = 70,
    hard_violation_count: int = 0,
    feedback_adjusted: int = 0,
    generation_latency_ms: int = 100,
    hard_violation_categories: list[str] | None = None,
    hard_violation_category_counts: dict[str, int] | None = None,
):
    event = {
        "feedback_enabled": feedback_enabled,
        "feedback_adjusted_candidate_count": feedback_adjusted,
        "feedback_positive_count": feedback_adjusted,
        "feedback_negative_count": 0,
        "feedback_abs_adjustment_avg": 0.1 if feedback_adjusted else 0.0,
        "feedback_query_latency_ms": 2 if feedback_enabled else None,
        "feedback_query_failed": False,
        "critic_score": critic_score,
        "critic_grade": "FAIL" if hard_violation_count else "WARN",
        "hard_violation_count": hard_violation_count,
        "fallback_used": False,
        "fallback_reason": "none",
        "llm_error_type": None,
        "generation_latency_ms": generation_latency_ms,
        "candidate_pool_size": 12,
    }
    if hard_violation_categories is not None:
        event["hard_violation_categories"] = hard_violation_categories
    if hard_violation_category_counts is not None:
        event["hard_violation_category_counts"] = hard_violation_category_counts
    return event


def _result(scenario_id: str, event: dict):
    return {"scenario_id": scenario_id, "telemetry": event}


def _raw(*, group: str, results: list[dict], **overrides):
    payload = {
        "mode": "feedback_ranking_staging",
        "group": group,
        "request_count": len(results),
        "failed_requests": [],
        "results": results,
        "privacy_leak_detected": False,
        "db_readiness_status": "pass",
        "db_table_exists": True,
        "db_table_created": False,
        "seed_readiness_status": "pass",
        "seed_verification_passed": True,
        "missing_seed_count": 0,
    }
    payload.update(overrides)
    return payload


def _write(path: Path, payload: dict):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _compare(tmp_path: Path, raw_a: dict, raw_b: dict) -> dict:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    out = tmp_path / "out.json"
    _write(a, raw_a)
    _write(b, raw_b)

    staging.cmd_compare(argparse.Namespace(a=str(a), b=str(b), output=str(out), markdown=None))

    return json.loads(out.read_text(encoding="utf-8"))


def test_run_help_includes_request_timeout():
    result = subprocess.run(
        [sys.executable, "scripts/run_feedback_ranking_staging_test.py", "run", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--request-timeout" in result.stdout


def test_request_timeout_default_remains_120():
    assert staging.DEFAULT_TIMEOUT == 120


def test_parse_scenario_ids_parses_comma_separated_ids():
    assert staging._parse_scenario_ids("normal_hypertrophy_upper, strength_lower,,low_readiness") == [
        "normal_hypertrophy_upper",
        "strength_lower",
        "low_readiness",
    ]


def test_unknown_scenario_id_fails_fast():
    with pytest.raises(SystemExit) as exc_info:
        staging._select_scenarios(staging.BASE_SCENARIOS, ["normal_hypertrophy_upper", "missing"])

    assert "unknown scenario id" in str(exc_info.value)


def test_select_scenarios_preserves_requested_order():
    selected, ids = staging._select_scenarios(
        staging.BASE_SCENARIOS + staging.SEEDED_SCENARIOS,
        ["strength_lower_fb", "normal_hypertrophy_upper"],
    )

    assert ids == ["strength_lower_fb", "normal_hypertrophy_upper"]
    assert [scenario["id"] for scenario in selected] == ids


def test_custom_request_timeout_is_applied(monkeypatch):
    seen = {}

    def fake_request(method, url, payload=None, timeout=120):
        seen["timeout"] = timeout
        return {"routineDraftId": "draft-1", "routineBlocks": []}

    monkeypatch.setattr(staging, "_json_request", fake_request)

    response, error = staging._post_generation(
        "http://test",
        {"id": "scenario-1", "payload": {"safe": True}},
        timeout=240,
        group="b",
    )

    assert error is None
    assert response["routineDraftId"] == "draft-1"
    assert seen["timeout"] == 240


def test_timeout_failures_are_counted_without_raw_body(monkeypatch):
    def fake_request(method, url, payload=None, timeout=120):
        raise TimeoutError("timed out with private body")

    monkeypatch.setattr(staging, "_json_request", fake_request)

    result = staging._run_one(
        "http://test",
        {"id": "slow_scenario", "payload": {"raw": "must-not-be-saved"}},
        timeout=180,
        group="a",
    )
    summary = staging._run_summary([result], 180)
    text = json.dumps(result)

    assert summary["failed_request_count"] == 1
    assert summary["timeout_count"] == 1
    assert summary["timeout_scenarios"] == ["slow_scenario"]
    assert summary["request_timeout_sec"] == 180
    assert result["error"]["scenario_id"] == "slow_scenario"
    assert result["error"]["group"] == "a"
    assert result["error"]["error_type"] == "TimeoutError"
    assert "raw" not in text
    assert "must-not-be-saved" not in text
    assert "private body" not in text


def test_compare_includes_timeout_and_telemetry_fields(tmp_path):
    raw_a = _raw(
        group="a",
        results=[_result("a1", _event(feedback_enabled=False, generation_latency_ms=100))],
        request_count=2,
        timeout_count=1,
        telemetry_missing_count=1,
        successful_latency_p95_ms=100,
        successful_latency_max_ms=100,
    )
    raw_b = _raw(
        group="b",
        results=[_result("b1", _event(feedback_enabled=True, feedback_adjusted=1, generation_latency_ms=200))],
        request_count=1,
        timeout_count=0,
        telemetry_missing_count=0,
        successful_latency_p95_ms=200,
        successful_latency_max_ms=200,
    )

    report = _compare(tmp_path, raw_a, raw_b)

    assert report["timeout_count_a"] == 1
    assert report["timeout_count_b"] == 0
    assert report["timeout_delta"] == -1
    assert report["telemetry_missing_count_a"] == 1
    assert report["telemetry_missing_count_b"] == 0
    assert report["p95_latency_a"] == 100
    assert report["p95_latency_b"] == 200
    assert report["max_latency_a"] == 100
    assert report["max_latency_b"] == 200


def test_compare_includes_hard_violation_breakdown(tmp_path):
    raw_a = _raw(
        group="a",
        results=[
            _result("strength_lower", _event(feedback_enabled=False, hard_violation_count=2)),
            _result("strength_lower_fb", _event(feedback_enabled=False, hard_violation_count=1)),
        ],
    )
    raw_b = _raw(group="b", results=[_result("strength_lower", _event(feedback_enabled=True, feedback_adjusted=1))])

    report = _compare(tmp_path, raw_a, raw_b)

    assert report["hard_violation_count_by_scenario_a"] == {
        "strength_lower": 2,
        "strength_lower_fb": 1,
    }
    assert report["hard_violation_count_by_scenario"]["a"]["strength_lower"] == 2
    assert report["hard_violation_category_counts_a"] == {"unknown": 3}
    assert report["control_hard_violation_count"] == 3
    assert report["treatment_hard_violation_count"] == 0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[GUARD_BYPASS] blocked equipment in bench: ['BARBELL']", "equipment"),
        ("[GUARD_BYPASS] pain trigger in press: ['shoulder']", "pain_trigger"),
        ("[GUARD_BYPASS] DOMS level 3 muscle in row: back", "doms_readiness"),
        ("[GUARD_BYPASS] working set cap exceeded: actual=14, max=10", "set_cap"),
        ("[GUARD_BYPASS] exercise_id 'ghost' not in candidates", "hallucinated_exercise"),
        ("FAIL time budget exceeded estimated time duration", "time_budget"),
        ("FAIL rationale missing: ratio=0.00", "rationale_missing"),
        ("schema parse validation malformed", "schema_malformed"),
        ("unexpected constraint", "unknown"),
    ],
)
def test_hard_violation_category_classifier(text, expected):
    assert classify_hard_violation_category(text) == expected


def test_run_summary_stores_hard_violation_category_counts_only():
    result = _result(
        "privacy_case",
        {
            **_event(
                feedback_enabled=False,
                hard_violation_count=2,
                hard_violation_categories=["equipment", "pain_trigger"],
            ),
            "hard_violation_details": [
                "blocked equipment raw warning text must not be copied",
                "pain trigger raw warning text must not be copied",
            ],
        },
    )

    summary = staging._run_summary([result], 120)
    text = json.dumps(summary)

    assert summary["hard_violation_category_counts"] == {"equipment": 1, "pain_trigger": 1}
    assert summary["hard_violation_category_counts_by_scenario"] == {
        "privacy_case": {"equipment": 1, "pain_trigger": 1}
    }
    assert "raw warning text" not in text
    assert "blocked equipment raw" not in text


def test_compare_aggregates_hard_violation_category_counts_by_group(tmp_path):
    raw_a = _raw(
        group="a",
        results=[
            _result(
                "strength_lower",
                _event(
                    feedback_enabled=False,
                    hard_violation_count=2,
                    hard_violation_category_counts={"set_cap": 1, "hallucinated_exercise": 1},
                ),
            )
        ],
    )
    raw_b = _raw(
        group="b",
        results=[
            _result(
                "normal",
                _event(
                    feedback_enabled=True,
                    hard_violation_count=1,
                    hard_violation_category_counts={"equipment": 1},
                ),
            )
        ],
    )

    report = _compare(tmp_path, raw_a, raw_b)

    assert report["hard_violation_category_counts_a"] == {"hallucinated_exercise": 1, "set_cap": 1}
    assert report["hard_violation_category_counts_b"] == {"equipment": 1}
    assert report["hard_violation_category_counts"]["a"] == {"hallucinated_exercise": 1, "set_cap": 1}
    assert report["hard_violation_category_counts_by_scenario_a"] == {
        "strength_lower": {"hallucinated_exercise": 1, "set_cap": 1}
    }
    assert report["hard_violation_category_counts_by_scenario_b"] == {"normal": {"equipment": 1}}


def test_missing_hard_violation_detail_preserves_unknown_category(tmp_path):
    raw_a = _raw(
        group="a",
        results=[_result("strength_lower", _event(feedback_enabled=False, hard_violation_count=2))],
    )
    raw_b = _raw(group="b", results=[_result("normal", _event(feedback_enabled=True, feedback_adjusted=1))])

    report = _compare(tmp_path, raw_a, raw_b)

    assert report["hard_violation_category_counts_a"] == {"unknown": 2}
    assert report["hard_violation_category_counts_by_scenario_a"] == {"strength_lower": {"unknown": 2}}


def test_local_gemma4_control_hard_violations_recommend_inconclusive(tmp_path):
    raw_a = _raw(
        group="a",
        results=[_result("strength_lower", _event(feedback_enabled=False, hard_violation_count=1))],
        llm_provider="local",
        local_llm_model="gemma4:latest",
    )
    raw_b = _raw(
        group="b",
        results=[_result("strength_lower", _event(feedback_enabled=True, feedback_adjusted=1, critic_score=75))],
        llm_provider="local",
        local_llm_model="gemma4:latest",
    )

    report = _compare(tmp_path, raw_a, raw_b)

    assert report["recommendation"] == "inconclusive_gemma4_baseline_unstable"


def test_unknown_provider_control_hard_violations_do_not_promote(tmp_path):
    raw_a = _raw(
        group="a",
        results=[_result("strength_lower", _event(feedback_enabled=False, hard_violation_count=1, critic_score=70))],
    )
    raw_b = _raw(
        group="b",
        results=[_result("strength_lower", _event(feedback_enabled=True, feedback_adjusted=2, critic_score=80))],
    )

    report = _compare(tmp_path, raw_a, raw_b)

    assert report["recommendation"] == "inconclusive_control_hard_violations"
    assert report["recommendation"] != "promote_feedback_ranking_candidate"


def test_treatment_hard_violations_keep_feedback_off(tmp_path):
    raw_a = _raw(group="a", results=[_result("normal", _event(feedback_enabled=False, critic_score=70))])
    raw_b = _raw(
        group="b",
        results=[
            _result(
                "normal",
                _event(feedback_enabled=True, feedback_adjusted=2, hard_violation_count=1, critic_score=80),
            )
        ],
    )

    report = _compare(tmp_path, raw_a, raw_b)

    assert report["recommendation"] == "keep_feedback_off"


def test_local_gemma4_timeout_recommends_inconclusive_local_timeout(tmp_path):
    raw_a = _raw(
        group="a",
        results=[_result("normal", _event(feedback_enabled=False))],
        request_count=2,
        timeout_count=1,
        llm_provider="local",
        local_llm_model="gemma4:latest",
    )
    raw_b = _raw(
        group="b",
        results=[_result("normal", _event(feedback_enabled=True, feedback_adjusted=1, critic_score=75))],
        llm_provider="local",
        local_llm_model="gemma4:latest",
    )

    report = _compare(tmp_path, raw_a, raw_b)

    assert report["recommendation"] == "inconclusive_local_timeout"


def test_control_hard_violations_take_priority_over_promote_criteria(tmp_path):
    report = {
        "readiness_blocked": False,
        "seed_verification_passed_b": True,
        "privacy_leak_detected": False,
        "control_hard_violation_count": 6,
        "treatment_hard_violation_count": 0,
        "hard_violation_count_a": 6,
        "hard_violation_count_b": 0,
        "feedback_query_failed_rate_b": 0.0,
        "timeout_count_a": 0,
        "timeout_count_b": 0,
        "llm_provider_a": None,
        "llm_provider_b": None,
        "local_llm_model_a": None,
        "local_llm_model_b": None,
        "fallback_rate_a": 0.0,
        "fallback_rate_b": 0.0,
        "p95_feedback_query_latency_ms_b": 3,
        "avg_critic_score_a": 58.4167,
        "avg_critic_score_b": 68.5,
        "quota_fallback_rate_b": 0.0,
        "avg_feedback_adjusted_count_b": 2.8333,
    }

    assert staging._recommend(report) == "inconclusive_control_hard_violations"


def test_forbidden_privacy_keys_absent_from_run_summary():
    result = {
        "scenario_id": "safe",
        "error": {
            "scenario_id": "safe",
            "group": "a",
            "elapsed_sec": 1.0,
            "error_type": "TimeoutError",
        },
    }
    summary = staging._run_summary([result], 120)
    text = json.dumps(summary)

    for forbidden in staging.FORBIDDEN_PRIVACY_TOKENS:
        assert forbidden.lower() not in text.lower()


def test_existing_compare_behavior_unchanged_for_non_local_no_instability(tmp_path):
    raw_a = _raw(group="a", results=[_result("normal", _event(feedback_enabled=False, critic_score=70))])
    raw_b = _raw(
        group="b",
        results=[_result("normal", _event(feedback_enabled=True, critic_score=75, feedback_adjusted=2))],
    )

    report = _compare(tmp_path, raw_a, raw_b)

    assert report["recommendation"] == "promote_feedback_ranking_candidate"

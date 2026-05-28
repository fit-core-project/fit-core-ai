"""Run candidate pool 12 vs 18 telemetry comparison and write a JSON report.

This script reuses the staging telemetry harness. It is framework validation:
DB, candidate lookup, and LLM calls are mocked by the harness; no external API
or Gemini call is made.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.test_staging_telemetry import _privacy_leak, _run_group  # noqa: E402

DEFAULT_REPORT_PATH = ROOT / "tests" / "evaluation" / ".artifacts" / "staging_candidate_pool_ab_report.json"


def _avg(items: list[dict[str, Any]], key: str) -> float:
    values = [item[key] for item in items if item.get(key) is not None]
    return round(mean(values), 2) if values else 0.0


def _rate(items: list[dict[str, Any]], key: str, value: Any) -> float:
    if not items:
        return 0.0
    return round(sum(1 for item in items if item.get(key) == value) / len(items), 4)


def _count_grades(items: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "PASS": sum(1 for item in items if item.get("critic_grade") == "PASS"),
        "WARN": sum(1 for item in items if item.get("critic_grade") == "WARN"),
        "FAIL": sum(1 for item in items if item.get("critic_grade") == "FAIL"),
    }


def _percentile(items: list[dict[str, Any]], key: str, percentile: float) -> float:
    values = sorted(item[key] for item in items if item.get(key) is not None)
    if not values:
        return 0.0
    if percentile == 0.5:
        return round(median(values), 2)
    index = max(0, math.ceil(len(values) * percentile) - 1)
    return round(values[index], 2)


def _recommendation(report: dict[str, Any]) -> str:
    hard_18 = report["hard_violation_count_18"]
    hard_12 = report["hard_violation_count_12"]
    fallback_18 = report["fallback_rate_18"]
    fallback_12 = report["fallback_rate_12"]
    repair_18 = report["avg_repair_count_18"]
    repair_12 = report["avg_repair_count_12"]
    score_18 = report["avg_critic_score_18"]
    score_12 = report["avg_critic_score_12"]
    payload_growth = report["payload_growth_ratio"]
    latency_growth = report["latency_growth_ratio"]

    if report["privacy_leak_detected"]:
        return "keep_pool_12"
    if hard_18 != 0 or hard_18 > hard_12:
        return "keep_pool_12"
    if fallback_18 > fallback_12 or repair_18 > repair_12:
        return "keep_pool_12"
    if score_18 < score_12:
        return "continue_staging_pool_18"
    if payload_growth > 0.60 or latency_growth > 0.30:
        return "continue_staging_pool_18"
    return "promote_pool_18_candidate"


def build_report(requests_per_group: int) -> dict[str, Any]:
    from tests import test_staging_telemetry as harness

    scenario_count = len(harness._SCENARIOS)
    repeats = max(1, math.ceil(requests_per_group / scenario_count))

    # The pytest harness tees verbose prompt/debug output to stdout. Suppress it
    # here so this script only writes the final report path and JSON summary.
    with contextlib.redirect_stdout(io.StringIO()):
        events12 = _run_group(pool_size=12, repeats=repeats)
        events18 = _run_group(pool_size=18, repeats=repeats)

    events12 = events12[:requests_per_group]
    events18 = events18[:requests_per_group]

    avg_payload_12 = _avg(events12, "candidate_payload_char_count")
    avg_payload_18 = _avg(events18, "candidate_payload_char_count")
    p95_12 = _percentile(events12, "generation_latency_ms", 0.95)
    p95_18 = _percentile(events18, "generation_latency_ms", 0.95)

    report = {
        "mode": "mock_framework_validation",
        "request_count_12": len(events12),
        "request_count_18": len(events18),
        "avg_critic_score_12": _avg(events12, "critic_score"),
        "avg_critic_score_18": _avg(events18, "critic_score"),
        "pass_warn_fail_12": _count_grades(events12),
        "pass_warn_fail_18": _count_grades(events18),
        "hard_violation_count_12": sum(item.get("hard_violation_count", 0) for item in events12),
        "hard_violation_count_18": sum(item.get("hard_violation_count", 0) for item in events18),
        "avg_repair_count_12": _avg(events12, "repair_count"),
        "avg_repair_count_18": _avg(events18, "repair_count"),
        "fallback_rate_12": _rate(events12, "fallback_used", True),
        "fallback_rate_18": _rate(events18, "fallback_used", True),
        "avg_payload_chars_12": avg_payload_12,
        "avg_payload_chars_18": avg_payload_18,
        "payload_growth_ratio": round((avg_payload_18 - avg_payload_12) / avg_payload_12, 4)
        if avg_payload_12
        else 0.0,
        "p50_latency_12": _percentile(events12, "generation_latency_ms", 0.5),
        "p50_latency_18": _percentile(events18, "generation_latency_ms", 0.5),
        "p95_latency_12": p95_12,
        "p95_latency_18": p95_18,
        "latency_growth_ratio": round((p95_18 - p95_12) / p95_12, 4) if p95_12 else 0.0,
        "privacy_leak_detected": bool(_privacy_leak(events12) or _privacy_leak(events18)),
    }
    report["recommendation"] = _recommendation(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests-per-group", type=int, default=50)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    if args.requests_per_group < 10:
        raise SystemExit("--requests-per-group must be at least 10")

    report = build_report(args.requests_per_group)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"report_path": str(args.output), **report}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

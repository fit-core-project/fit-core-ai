"""Live HTTP candidate pool telemetry A/B runner.

This script calls a running FastAPI server and collects telemetry from
`/api/dev/logs`. It never changes server env or restarts the server.
Run each pool against a server already started with the matching
ROUTINE_CANDIDATE_POOL_SIZE, then merge the two raw result files.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_REPORT = Path("tests/evaluation/.artifacts/live_candidate_pool_ab_report.json")
DEFAULT_REPORT_MD = Path("tests/evaluation/.artifacts/live_candidate_pool_ab_report.md")
QUOTA_FALLBACK_INCONCLUSIVE_THRESHOLD = 0.5

FORBIDDEN_TELEMETRY_KEYS = {
    "user_note",
    "userNote",
    "feedback",
    "raw_prompt",
    "rawPrompt",
    "raw_llm_output",
    "rawLlmOutput",
    "prompt_text",
    "promptText",
    "rationale",
    "rationale_summary",
    "rationaleSummary",
    "exercise_rationale",
    "exerciseRationale",
    "warnings_text",
    "warning_text",
    "medical_note",
}

SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "normal_hypertrophy_upper",
        "payload": {
            "userId": "live-ab-normal-upper",
            "targetSplitLabel": "push",
            "readinessLevel": "normal",
            "timeAvailableMin": 60,
            "painAreas": [],
            "domsData": {},
            "equipment": [],
            "goal": "hypertrophy",
        },
    },
    {
        "id": "strength_lower",
        "payload": {
            "userId": "live-ab-strength-lower",
            "targetSplitLabel": "legs",
            "readinessLevel": "normal",
            "timeAvailableMin": 60,
            "painAreas": [],
            "domsData": {},
            "equipment": [],
            "goal": "strength",
        },
    },
    {
        "id": "fatloss_limited_equipment",
        "payload": {
            "userId": "live-ab-fatloss-equipment",
            "targetSplitLabel": "full_body",
            "readinessLevel": "normal",
            "timeAvailableMin": 45,
            "painAreas": [],
            "domsData": {},
            "equipment": ["BARBELL", "MACHINE"],
            "goal": "fatLoss",
        },
    },
    {
        "id": "shoulder_pain_present",
        "payload": {
            "userId": "live-ab-shoulder-pain",
            "targetSplitLabel": "push",
            "readinessLevel": "normal",
            "timeAvailableMin": 50,
            "painAreas": [{"area": "shoulder", "severity": "moderate"}],
            "domsData": {},
            "equipment": [],
            "goal": "hypertrophy",
        },
    },
    {
        "id": "knee_pain_present",
        "payload": {
            "userId": "live-ab-knee-pain",
            "targetSplitLabel": "legs",
            "readinessLevel": "normal",
            "timeAvailableMin": 50,
            "painAreas": [{"area": "knee", "severity": "moderate"}],
            "domsData": {},
            "equipment": [],
            "goal": "hypertrophy",
        },
    },
    {
        "id": "low_readiness",
        "payload": {
            "userId": "live-ab-low-readiness",
            "targetSplitLabel": "push",
            "readinessLevel": "low",
            "timeAvailableMin": 40,
            "painAreas": [],
            "domsData": {"chest": 2},
            "equipment": [],
            "goal": "hypertrophy",
        },
    },
    {
        "id": "short_duration_20m",
        "payload": {
            "userId": "live-ab-short-duration",
            "targetSplitLabel": "push",
            "readinessLevel": "normal",
            "timeAvailableMin": 20,
            "painAreas": [],
            "domsData": {},
            "equipment": [],
            "goal": "hypertrophy",
        },
    },
    {
        "id": "long_duration_60m",
        "payload": {
            "userId": "live-ab-long-duration",
            "targetSplitLabel": "pull",
            "readinessLevel": "high",
            "timeAvailableMin": 60,
            "painAreas": [],
            "domsData": {},
            "equipment": [],
            "goal": "hypertrophy",
        },
    },
    {
        "id": "bodyweight_only",
        "payload": {
            "userId": "live-ab-bodyweight-only",
            "targetSplitLabel": "push",
            "readinessLevel": "normal",
            "timeAvailableMin": 40,
            "painAreas": [],
            "domsData": {},
            "equipment": ["BARBELL", "DUMBBELL", "CABLE", "MACHINE"],
            "goal": "endurance",
        },
    },
    {
        "id": "mixed_target_muscles",
        "payload": {
            "userId": "live-ab-mixed-targets",
            "targetMuscles": ["chest", "triceps", "front-deltoids"],
            "readinessLevel": "normal",
            "timeAvailableMin": 55,
            "painAreas": [],
            "domsData": {},
            "equipment": [],
            "goal": "hypertrophy",
        },
    },
]


def _json_request(method: str, url: str, payload: dict[str, Any] | None = None, timeout: int = 120) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else None


def _check_server(base_url: str) -> dict[str, Any]:
    try:
        openapi = _json_request("GET", f"{base_url.rstrip('/')}/openapi.json", timeout=10)
        paths = set((openapi.get("paths") or {}).keys())
        return {
            "reachable": True,
            "generation_endpoint_available": "/api/ai/generate-routine" in paths,
            "dev_logs_available": "/api/dev/logs" in paths,
        }
    except Exception as exc:
        return {"reachable": False, "error": f"{type(exc).__name__}: {exc}"}


def _get_logs(base_url: str, limit: int = 300) -> list[str]:
    payload = _json_request("GET", f"{base_url.rstrip('/')}/api/dev/logs?limit={limit}", timeout=10)
    if isinstance(payload, list):
        return [str(item) for item in payload]
    if isinstance(payload, dict) and isinstance(payload.get("value"), list):
        return [str(item) for item in payload["value"]]
    return []


def _parse_telemetry_from_logs(lines: list[str]) -> list[dict[str, Any]]:
    events = []
    for line in lines:
        marker = "[Telemetry] "
        if marker not in line:
            continue
        raw = line.split(marker, 1)[1].strip()
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "routine_generation_quality":
            events.append(event)
    return events


def _find_telemetry_for_draft(base_url: str, routine_draft_id: str, timeout_sec: float = 20.0) -> dict[str, Any] | None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        events = _parse_telemetry_from_logs(_get_logs(base_url, limit=300))
        for event in reversed(events):
            if event.get("routine_draft_id") == routine_draft_id:
                return event
        time.sleep(0.5)
    return None


def _post_generation(base_url: str, scenario: dict[str, Any], timeout: int) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    url = f"{base_url.rstrip('/')}/api/ai/generate-routine"
    try:
        started = time.perf_counter()
        response = _json_request("POST", url, scenario["payload"], timeout=timeout)
        http_latency_ms = round((time.perf_counter() - started) * 1000)
        response["_http_latency_ms"] = http_latency_ms
        return response, None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:800]
        return None, {"scenario_id": scenario["id"], "status_code": exc.code, "body": body}
    except Exception as exc:
        return None, {"scenario_id": scenario["id"], "error": f"{type(exc).__name__}: {exc}"}


def _validate_response(response: dict[str, Any]) -> list[str]:
    required = [
        "routineDraftId",
        "generationStatus",
        "statusReasonCode",
        "isFallback",
        "totalEstimatedTime",
        "summaryTitle",
        "rationaleSummary",
        "warnings",
        "routineBlocks",
    ]
    return [key for key in required if key not in response]


def run_pool(base_url: str, expected_pool_size: int, requests_per_scenario: int, sleep_sec: float) -> dict[str, Any]:
    server = _check_server(base_url)
    if not server.get("reachable") or not server.get("generation_endpoint_available"):
        raise SystemExit(f"Server is not ready for live generation: {server}")

    results: list[dict[str, Any]] = []
    failed_requests: list[dict[str, Any]] = []

    probe_response, probe_error = _post_generation(base_url, SCENARIOS[0], timeout=180)
    if probe_error:
        raise SystemExit(f"Probe request failed: {probe_error}")
    missing = _validate_response(probe_response or {})
    if missing:
        raise SystemExit(f"Probe response missing fields: {missing}")
    probe_event = _find_telemetry_for_draft(base_url, probe_response["routineDraftId"])
    if probe_event is None:
        raise SystemExit("Telemetry event was not found for probe request")
    actual_pool = probe_event.get("candidate_pool_size")
    if actual_pool != expected_pool_size:
        raise SystemExit(
            f"Server candidate_pool_size mismatch: expected {expected_pool_size}, got {actual_pool}. "
            "Restart/apply server env before running this pool."
        )
    results.append({
        "scenario_id": SCENARIOS[0]["id"],
        "response": _redact_response(probe_response),
        "telemetry": probe_event,
    })

    for repeat in range(requests_per_scenario):
        for scenario in SCENARIOS:
            if repeat == 0 and scenario["id"] == SCENARIOS[0]["id"]:
                continue
            time.sleep(sleep_sec)
            response, error = _post_generation(base_url, scenario, timeout=180)
            if error:
                failed_requests.append({**error, "repeat": repeat})
                continue
            missing = _validate_response(response or {})
            if missing:
                failed_requests.append({"scenario_id": scenario["id"], "repeat": repeat, "missing_fields": missing})
                continue
            event = _find_telemetry_for_draft(base_url, response["routineDraftId"])
            if event is None:
                failed_requests.append({
                    "scenario_id": scenario["id"],
                    "repeat": repeat,
                    "routine_draft_id": response.get("routineDraftId"),
                    "error": "telemetry_not_found",
                })
                continue
            if event.get("candidate_pool_size") != expected_pool_size:
                failed_requests.append({
                    "scenario_id": scenario["id"],
                    "repeat": repeat,
                    "routine_draft_id": response.get("routineDraftId"),
                    "error": "candidate_pool_size_mismatch",
                    "expected": expected_pool_size,
                    "actual": event.get("candidate_pool_size"),
                })
                continue
            results.append({
                "scenario_id": scenario["id"],
                "response": _redact_response(response),
                "telemetry": event,
            })

    return {
        "mode": "live_pool",
        "base_url": base_url,
        "expected_pool_size": expected_pool_size,
        "request_count": len(results),
        "failed_requests": failed_requests,
        "results": results,
    }


def _redact_response(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "routineDraftId": response.get("routineDraftId"),
        "generationStatus": response.get("generationStatus"),
        "statusReasonCode": response.get("statusReasonCode"),
        "isFallback": response.get("isFallback"),
        "totalEstimatedTime": response.get("totalEstimatedTime"),
        "routineBlockCount": len(response.get("routineBlocks") or []),
        "warningCount": len(response.get("warnings") or []),
    }


def _avg(events: list[dict[str, Any]], key: str) -> float:
    values = [event[key] for event in events if event.get(key) is not None]
    return round(mean(values), 2) if values else 0.0


def _rate(events: list[dict[str, Any]], key: str, value: Any) -> float:
    return round(sum(1 for event in events if event.get(key) == value) / len(events), 4) if events else 0.0


def _grades(events: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "PASS": sum(1 for event in events if event.get("critic_grade") == "PASS"),
        "WARN": sum(1 for event in events if event.get("critic_grade") == "WARN"),
        "FAIL": sum(1 for event in events if event.get("critic_grade") == "FAIL"),
    }


def _counts(events: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(event.get(key) or "none") for event in events))


def _schema_repair_success_rate(events: list[dict[str, Any]]) -> float:
    attempted = sum(1 for event in events if event.get("schema_repair_attempted") is True)
    if attempted == 0:
        return 0.0
    succeeded = sum(1 for event in events if event.get("schema_repair_succeeded") is True)
    return round(succeeded / attempted, 4)


def _pct(events: list[dict[str, Any]], key: str, percentile: float) -> float:
    values = sorted(event[key] for event in events if event.get(key) is not None)
    if not values:
        return 0.0
    if percentile == 0.5:
        return round(median(values), 2)
    index = max(0, math.ceil(len(values) * percentile) - 1)
    return round(values[index], 2)


def _privacy_leak(events: list[dict[str, Any]]) -> bool:
    for event in events:
        if any(key in event for key in FORBIDDEN_TELEMETRY_KEYS):
            return True
    return False


def _failed_rate(raw: dict[str, Any]) -> float:
    total = raw.get("request_count", 0) + len(raw.get("failed_requests") or [])
    return round(len(raw.get("failed_requests") or []) / total, 4) if total else 0.0


def merge_reports(pool12_path: Path, pool18_path: Path, output: Path, markdown_output: Path | None) -> dict[str, Any]:
    pool12 = json.loads(pool12_path.read_text(encoding="utf-8"))
    pool18 = json.loads(pool18_path.read_text(encoding="utf-8"))
    events12 = [item["telemetry"] for item in pool12.get("results", [])]
    events18 = [item["telemetry"] for item in pool18.get("results", [])]
    avg_payload12 = _avg(events12, "candidate_payload_char_count")
    avg_payload18 = _avg(events18, "candidate_payload_char_count")
    p95_12 = _pct(events12, "generation_latency_ms", 0.95)
    p95_18 = _pct(events18, "generation_latency_ms", 0.95)
    quota_12 = sum(1 for event in events12 if event.get("llm_error_type") == "quota_exhausted")
    quota_18 = sum(1 for event in events18 if event.get("llm_error_type") == "quota_exhausted")

    report = {
        "mode": "live",
        "base_url": pool12.get("base_url") or pool18.get("base_url"),
        "request_count_12": len(events12),
        "request_count_18": len(events18),
        "avg_critic_score_12": _avg(events12, "critic_score"),
        "avg_critic_score_18": _avg(events18, "critic_score"),
        "pass_warn_fail_12": _grades(events12),
        "pass_warn_fail_18": _grades(events18),
        "fallback_reason_counts_12": _counts(events12, "fallback_reason"),
        "fallback_reason_counts_18": _counts(events18, "fallback_reason"),
        "llm_error_type_counts_12": _counts(events12, "llm_error_type"),
        "llm_error_type_counts_18": _counts(events18, "llm_error_type"),
        "quota_exceeded_count_12": quota_12,
        "quota_exceeded_count_18": quota_18,
        "quota_fallback_rate_12": round(quota_12 / len(events12), 4) if events12 else 0.0,
        "quota_fallback_rate_18": round(quota_18 / len(events18), 4) if events18 else 0.0,
        "schema_repair_attempted_count_12": sum(1 for event in events12 if event.get("schema_repair_attempted") is True),
        "schema_repair_attempted_count_18": sum(1 for event in events18 if event.get("schema_repair_attempted") is True),
        "schema_repair_succeeded_count_12": sum(1 for event in events12 if event.get("schema_repair_succeeded") is True),
        "schema_repair_succeeded_count_18": sum(1 for event in events18 if event.get("schema_repair_succeeded") is True),
        "schema_repair_success_rate_12": _schema_repair_success_rate(events12),
        "schema_repair_success_rate_18": _schema_repair_success_rate(events18),
        "hard_violation_count_12": sum(event.get("hard_violation_count", 0) for event in events12),
        "hard_violation_count_18": sum(event.get("hard_violation_count", 0) for event in events18),
        "avg_repair_count_12": _avg(events12, "repair_count"),
        "avg_repair_count_18": _avg(events18, "repair_count"),
        "fallback_rate_12": _rate(events12, "fallback_used", True),
        "fallback_rate_18": _rate(events18, "fallback_used", True),
        "avg_payload_chars_12": avg_payload12,
        "avg_payload_chars_18": avg_payload18,
        "payload_growth_ratio": round((avg_payload18 - avg_payload12) / avg_payload12, 4) if avg_payload12 else 0.0,
        "p50_latency_12": _pct(events12, "generation_latency_ms", 0.5),
        "p50_latency_18": _pct(events18, "generation_latency_ms", 0.5),
        "p95_latency_12": p95_12,
        "p95_latency_18": p95_18,
        "latency_growth_ratio": round((p95_18 - p95_12) / p95_12, 4) if p95_12 else 0.0,
        "privacy_leak_detected": _privacy_leak(events12) or _privacy_leak(events18),
        "failed_requests": {
            "pool_12": pool12.get("failed_requests", []),
            "pool_18": pool18.get("failed_requests", []),
        },
        "failed_request_rate_12": _failed_rate(pool12),
        "failed_request_rate_18": _failed_rate(pool18),
    }
    report["recommendation"] = _recommendation(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if markdown_output:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(_markdown(report), encoding="utf-8")
    return report


def _recommendation(report: dict[str, Any]) -> str:
    if (
        report.get("quota_fallback_rate_12", 0.0) > QUOTA_FALLBACK_INCONCLUSIVE_THRESHOLD
        or report.get("quota_fallback_rate_18", 0.0) > QUOTA_FALLBACK_INCONCLUSIVE_THRESHOLD
    ):
        return "inconclusive_quota_exhausted"
    if report["privacy_leak_detected"]:
        return "keep_pool_12"
    if report["hard_violation_count_18"] > report["hard_violation_count_12"]:
        return "keep_pool_12"
    if report["fallback_rate_18"] > report["fallback_rate_12"]:
        return "keep_pool_12"
    if report["avg_repair_count_18"] > report["avg_repair_count_12"]:
        return "keep_pool_12"
    if report["avg_critic_score_18"] < report["avg_critic_score_12"]:
        return "keep_pool_12"
    if report["latency_growth_ratio"] > 0.30:
        return "keep_pool_12"
    if report["failed_request_rate_18"] > report["failed_request_rate_12"]:
        return "keep_pool_12"
    if report["hard_violation_count_18"] == 0 and report["payload_growth_ratio"] <= 0.60:
        return "promote_pool_18_candidate"
    return "inconclusive"


def _markdown(report: dict[str, Any]) -> str:
    return "\n".join([
        "# Live Candidate Pool A/B Report",
        "",
        f"- base_url: `{report['base_url']}`",
        f"- recommendation: `{report['recommendation']}`",
        f"- requests: 12={report['request_count_12']}, 18={report['request_count_18']}",
        f"- avg critic score: 12={report['avg_critic_score_12']}, 18={report['avg_critic_score_18']}",
        f"- PASS/WARN/FAIL 12: `{report['pass_warn_fail_12']}`",
        f"- PASS/WARN/FAIL 18: `{report['pass_warn_fail_18']}`",
        "",
        "## Fallback Reasons",
        "",
        f"- fallback reason counts 12: `{report['fallback_reason_counts_12']}`",
        f"- fallback reason counts 18: `{report['fallback_reason_counts_18']}`",
        f"- LLM error type counts 12: `{report['llm_error_type_counts_12']}`",
        f"- LLM error type counts 18: `{report['llm_error_type_counts_18']}`",
        f"- quota fallback rate: 12={report['quota_fallback_rate_12']}, 18={report['quota_fallback_rate_18']}",
        f"- schema repair attempted: 12={report['schema_repair_attempted_count_12']}, 18={report['schema_repair_attempted_count_18']}",
        f"- schema repair succeeded: 12={report['schema_repair_succeeded_count_12']}, 18={report['schema_repair_succeeded_count_18']}",
        "",
        "## Quality And Performance",
        "",
        f"- hard violations: 12={report['hard_violation_count_12']}, 18={report['hard_violation_count_18']}",
        f"- fallback rate: 12={report['fallback_rate_12']}, 18={report['fallback_rate_18']}",
        f"- avg repair count: 12={report['avg_repair_count_12']}, 18={report['avg_repair_count_18']}",
        f"- avg payload chars: 12={report['avg_payload_chars_12']}, 18={report['avg_payload_chars_18']}",
        f"- payload growth ratio: {report['payload_growth_ratio']}",
        f"- p95 latency: 12={report['p95_latency_12']} ms, 18={report['p95_latency_18']} ms",
        f"- latency growth ratio: {report['latency_growth_ratio']}",
        f"- privacy leak detected: `{report['privacy_leak_detected']}`",
        f"- failed request rate: 12={report['failed_request_rate_12']}, 18={report['failed_request_rate_18']}",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--pool-size", type=int, choices=[12, 18])
    parser.add_argument("--requests-per-scenario", type=int, default=5)
    parser.add_argument("--sleep-sec", type=float, default=0.75)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--pool12", type=Path)
    parser.add_argument("--pool18", type=Path)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        print(json.dumps(_check_server(args.base_url), indent=2, ensure_ascii=False))
        return 0

    if args.merge:
        if not args.pool12 or not args.pool18:
            raise SystemExit("--merge requires --pool12 and --pool18")
        output = args.output or DEFAULT_REPORT
        report = merge_reports(args.pool12, args.pool18, output, args.markdown_output)
        print(json.dumps({"report_path": str(output), "markdown_report": str(args.markdown_output), **report}, indent=2))
        return 0

    if args.pool_size is None:
        raise SystemExit("Provide --pool-size 12 or 18, or use --merge")
    if args.requests_per_scenario < 1:
        raise SystemExit("--requests-per-scenario must be >= 1")

    output = args.output or Path(f"tests/evaluation/.artifacts/live_pool_{args.pool_size}.json")
    raw = run_pool(args.base_url, args.pool_size, args.requests_per_scenario, args.sleep_sec)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"raw_report_path": str(output), "request_count": raw["request_count"], "failed_requests": raw["failed_requests"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

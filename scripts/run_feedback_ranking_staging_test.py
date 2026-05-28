"""Staging runner for feedback-aware ranking flag comparison.

The script assumes the server is already running with:
- ROUTINE_CANDIDATE_POOL_SIZE=12
- ENABLE_RULE_CRITIC_TELEMETRY=true
- ENABLE_FEEDBACK_AWARE_RANKING=false for group a, true for group b
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
from statistics import mean
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_USER_ID = "staging-fb-user-001"
DEFAULT_TIMEOUT = 120

FORBIDDEN_PRIVACY_TOKENS = {
    "user_id",
    "user_note",
    "userNote",
    "skipped_exercise_ids",
    "skippedExercises",
    "edited_exercises",
    "editedExercises",
    "per_exercise_adjustment_map",
    "feedback_adjustment_map",
    "per_user_feedback",
    "replacement_exercise_id",
    "replacementExerciseId",
    "staging-fb-user-001",
}

BASE_SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "normal_hypertrophy_upper",
        "payload": {
            "userId": "staging-fb-live-normal",
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
            "userId": "staging-fb-live-strength",
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
            "userId": "staging-fb-live-fatloss",
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
            "userId": "staging-fb-live-shoulder",
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
            "userId": "staging-fb-live-knee",
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
            "userId": "staging-fb-live-low",
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
            "userId": "staging-fb-live-short",
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
            "userId": "staging-fb-live-long",
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
            "userId": "staging-fb-live-bodyweight",
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
            "userId": "staging-fb-live-mixed",
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

SEEDED_SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "normal_hypertrophy_upper_fb",
        "payload": {
            "userId": DEFAULT_USER_ID,
            "targetSplitLabel": "push",
            "readinessLevel": "normal",
            "timeAvailableMin": 60,
            "painAreas": [],
            "domsData": {},
            "equipment": [],
            "goal": "hypertrophy",
        },
        "expect_adjustment": True,
    },
    {
        "id": "strength_lower_fb",
        "payload": {
            "userId": DEFAULT_USER_ID,
            "targetSplitLabel": "legs",
            "readinessLevel": "normal",
            "timeAvailableMin": 60,
            "painAreas": [],
            "domsData": {},
            "equipment": [],
            "goal": "strength",
        },
        "expect_adjustment": False,
    },
]

SEED_ROWS: list[dict[str, Any]] = [
    {
        "routine_draft_id": "staging-seed-001",
        "rating": 5,
        "completed": True,
        "accepted_without_edits": True,
        "skipped_exercises": [],
        "edited_exercises": [{"original_exercise_id": "34", "replacement_exercise_id": "40", "reason": "other"}],
    },
    {
        "routine_draft_id": "staging-seed-002",
        "rating": 5,
        "completed": True,
        "accepted_without_edits": True,
        "skipped_exercises": [],
        "edited_exercises": [{"original_exercise_id": "37", "replacement_exercise_id": "40", "reason": "other"}],
    },
    {
        "routine_draft_id": "staging-seed-003",
        "rating": 2,
        "completed": False,
        "accepted_without_edits": False,
        "skipped_exercises": ["34"],
        "edited_exercises": [],
    },
    {
        "routine_draft_id": "staging-seed-004",
        "rating": 1,
        "completed": False,
        "accepted_without_edits": False,
        "skipped_exercises": [],
        "edited_exercises": [{"original_exercise_id": "76", "replacement_exercise_id": "40", "reason": "pain"}],
    },
    {
        "routine_draft_id": "staging-seed-005",
        "rating": 4,
        "completed": True,
        "accepted_without_edits": False,
        "skipped_exercises": [],
        "edited_exercises": [{"original_exercise_id": "99", "replacement_exercise_id": "101", "reason": "other"}],
    },
    {
        "routine_draft_id": "staging-seed-006",
        "rating": 2,
        "completed": False,
        "accepted_without_edits": False,
        "skipped_exercises": ["114"],
        "edited_exercises": [],
    },
    {
        "routine_draft_id": "staging-seed-007",
        "rating": 4,
        "completed": True,
        "accepted_without_edits": True,
        "skipped_exercises": [],
        "edited_exercises": [{"original_exercise_id": "126", "replacement_exercise_id": "107", "reason": "other"}],
    },
]


def _json_request(method: str, url: str, payload: dict[str, Any] | None = None, timeout: int = DEFAULT_TIMEOUT) -> Any:
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


def _get_logs(base_url: str, limit: int = 300) -> list[str]:
    payload = _json_request("GET", f"{base_url.rstrip('/')}/api/dev/logs?limit={limit}", timeout=10)
    if isinstance(payload, list):
        return [str(item) for item in payload]
    if isinstance(payload, dict) and isinstance(payload.get("value"), list):
        return [str(item) for item in payload["value"]]
    return []


def _parse_telemetry(lines: list[str]) -> list[dict[str, Any]]:
    events = []
    for line in lines:
        marker = "[Telemetry] "
        if marker not in line:
            continue
        try:
            event = json.loads(line.split(marker, 1)[1].strip())
        except json.JSONDecodeError:
            continue
        if event.get("event") == "routine_generation_quality":
            events.append(event)
    return events


def _find_telemetry_for_draft(base_url: str, routine_draft_id: str, timeout_sec: float = 20.0) -> dict[str, Any] | None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        for event in reversed(_parse_telemetry(_get_logs(base_url))):
            if event.get("routine_draft_id") == routine_draft_id:
                return event
        time.sleep(0.5)
    return None


def _post_generation(base_url: str, scenario: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        response = _json_request("POST", f"{base_url.rstrip('/')}/api/ai/generate-routine", scenario["payload"])
        return response, None
    except urllib.error.HTTPError as exc:
        return None, {"scenario_id": scenario["id"], "status_code": exc.code, "body": exc.read().decode("utf-8", errors="replace")[:500]}
    except Exception as exc:
        return None, {"scenario_id": scenario["id"], "error": f"{type(exc).__name__}: {exc}"}


def _run_one(base_url: str, scenario: dict[str, Any]) -> dict[str, Any]:
    response, error = _post_generation(base_url, scenario)
    if error:
        return {"scenario_id": scenario["id"], "error": error}
    draft_id = response.get("routineDraftId")
    event = _find_telemetry_for_draft(base_url, draft_id) if draft_id else None
    if event is None:
        return {"scenario_id": scenario["id"], "response": _response_summary(response), "error": "telemetry_not_found"}
    return {"scenario_id": scenario["id"], "response": _response_summary(response), "telemetry": _telemetry_summary(event)}


def _response_summary(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "routineDraftId": response.get("routineDraftId"),
        "generationStatus": response.get("generationStatus"),
        "isFallback": response.get("isFallback"),
        "routineBlockCount": len(response.get("routineBlocks") or []),
    }


def _telemetry_summary(event: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "feedback_enabled",
        "feedback_adjusted_candidate_count",
        "feedback_positive_count",
        "feedback_negative_count",
        "feedback_abs_adjustment_avg",
        "feedback_query_latency_ms",
        "feedback_query_failed",
        "feedback_query_error_category",
        "critic_score",
        "critic_grade",
        "hard_violation_count",
        "fallback_used",
        "fallback_reason",
        "llm_error_type",
        "generation_latency_ms",
        "candidate_pool_size",
    ]
    return {key: event.get(key) for key in keys}


def _privacy_leak_detected(value: Any) -> bool:
    text = json.dumps(value, ensure_ascii=False).lower()
    return any(token.lower() in text for token in FORBIDDEN_PRIVACY_TOKENS)


def cmd_seed(args: argparse.Namespace) -> None:
    base_url = args.base_url.rstrip("/")
    for row in SEED_ROWS:
        payload = dict(row)
        payload["user_id"] = args.user_id
        try:
            _json_request("POST", f"{base_url}/api/ai/routine-feedback", payload, timeout=20)
        except Exception as exc:
            raise SystemExit(f"seed failed for {payload['routine_draft_id']}: {type(exc).__name__}: {exc}") from exc
    print(json.dumps({"seeded": len(SEED_ROWS)}, indent=2))


def cmd_run(args: argparse.Namespace) -> None:
    expected_enabled = args.group == "b"
    scenarios = [dict(item) for item in BASE_SCENARIOS + SEEDED_SCENARIOS]
    for scenario in scenarios:
        if scenario["id"].endswith("_fb"):
            scenario["payload"] = dict(scenario["payload"], userId=args.user_id)

    probe = _run_one(args.base_url, scenarios[-2])
    telemetry = probe.get("telemetry") or {}
    if telemetry.get("feedback_enabled") is not expected_enabled:
        raise SystemExit(
            f"probe feedback_enabled={telemetry.get('feedback_enabled')} expected={expected_enabled}"
        )
    if telemetry.get("candidate_pool_size") != 12:
        raise SystemExit(f"probe candidate_pool_size={telemetry.get('candidate_pool_size')} expected=12")

    results = []
    failed = []
    for _ in range(args.repeats):
        for scenario in scenarios:
            result = _run_one(args.base_url, scenario)
            if result.get("error"):
                failed.append(result)
            results.append(result)

    output = {
        "mode": "feedback_ranking_staging",
        "group": args.group,
        "request_count": len(results),
        "failed_requests": failed,
        "results": results,
        "privacy_leak_detected": _privacy_leak_detected(results),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(path), "request_count": len(results), "failed_requests": len(failed)}, indent=2))


def _events(raw: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        result["telemetry"]
        for result in raw.get("results", [])
        if isinstance(result.get("telemetry"), dict)
    ]


def _rate(events: list[dict[str, Any]], key: str) -> float:
    return sum(1 for event in events if event.get(key)) / max(len(events), 1)


def _avg(events: list[dict[str, Any]], key: str) -> float:
    values = [event.get(key) for event in events if isinstance(event.get(key), (int, float))]
    return round(mean(values), 4) if values else 0.0


def _p95(events: list[dict[str, Any]], key: str) -> float:
    values = sorted(event.get(key) for event in events if isinstance(event.get(key), (int, float)))
    if not values:
        return 0.0
    index = min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)
    return values[index]


def _fallback_rate(events: list[dict[str, Any]]) -> float:
    return _rate(events, "fallback_used")


def _quota_rate(events: list[dict[str, Any]]) -> float:
    return sum(1 for event in events if event.get("fallback_reason") == "quota_exhausted") / max(len(events), 1)


def _recommend(report: dict[str, Any]) -> str:
    if report["privacy_leak_detected"]:
        return "keep_feedback_off"
    if report["hard_violation_count_b"] > 0:
        return "keep_feedback_off"
    if report["feedback_query_failed_rate_b"] > 0:
        return "keep_feedback_off"
    if report["fallback_rate_b"] > report["fallback_rate_a"] + 0.05:
        return "keep_feedback_off"
    if report["p95_feedback_query_latency_ms_b"] > 100:
        return "keep_feedback_off"
    if report["avg_critic_score_b"] < report["avg_critic_score_a"] - 10:
        return "keep_feedback_off"
    if report["quota_fallback_rate_b"] > 0.5:
        return "inconclusive_quota_exhausted"
    if report["avg_feedback_adjusted_count_b"] == 0:
        return "inconclusive_no_adjustment"
    if report["avg_critic_score_b"] >= report["avg_critic_score_a"] - 5 and report["avg_feedback_adjusted_count_b"] >= 1:
        return "promote_feedback_ranking_candidate"
    return "inconclusive"


def cmd_compare(args: argparse.Namespace) -> None:
    raw_a = json.loads(Path(args.a).read_text(encoding="utf-8"))
    raw_b = json.loads(Path(args.b).read_text(encoding="utf-8"))
    events_a = _events(raw_a)
    events_b = _events(raw_b)
    report = {
        "request_count_a": len(events_a),
        "request_count_b": len(events_b),
        "feedback_enabled_rate_a": _rate(events_a, "feedback_enabled"),
        "feedback_enabled_rate_b": _rate(events_b, "feedback_enabled"),
        "avg_feedback_adjusted_count_a": _avg(events_a, "feedback_adjusted_candidate_count"),
        "avg_feedback_adjusted_count_b": _avg(events_b, "feedback_adjusted_candidate_count"),
        "avg_feedback_positive_count_b": _avg(events_b, "feedback_positive_count"),
        "avg_feedback_negative_count_b": _avg(events_b, "feedback_negative_count"),
        "avg_feedback_abs_adjustment_b": _avg(events_b, "feedback_abs_adjustment_avg"),
        "p95_feedback_query_latency_ms_b": _p95(events_b, "feedback_query_latency_ms"),
        "feedback_query_failed_rate_b": _rate(events_b, "feedback_query_failed"),
        "avg_critic_score_a": _avg(events_a, "critic_score"),
        "avg_critic_score_b": _avg(events_b, "critic_score"),
        "fallback_rate_a": _fallback_rate(events_a),
        "fallback_rate_b": _fallback_rate(events_b),
        "hard_violation_count_a": sum(int(event.get("hard_violation_count") or 0) for event in events_a),
        "hard_violation_count_b": sum(int(event.get("hard_violation_count") or 0) for event in events_b),
        "quota_fallback_rate_a": _quota_rate(events_a),
        "quota_fallback_rate_b": _quota_rate(events_b),
        "privacy_leak_detected": bool(raw_a.get("privacy_leak_detected")) or bool(raw_b.get("privacy_leak_detected")),
    }
    report["recommendation"] = _recommend(report)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.markdown:
        Path(args.markdown).write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"report": str(output), "recommendation": report["recommendation"]}, indent=2))


def _markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Feedback Ranking Staging Report",
            "",
            f"- recommendation: `{report['recommendation']}`",
            f"- requests: A={report['request_count_a']}, B={report['request_count_b']}",
            f"- feedback enabled rate: A={report['feedback_enabled_rate_a']}, B={report['feedback_enabled_rate_b']}",
            f"- avg adjusted candidates: A={report['avg_feedback_adjusted_count_a']}, B={report['avg_feedback_adjusted_count_b']}",
            f"- p95 feedback query latency B: {report['p95_feedback_query_latency_ms_b']} ms",
            f"- feedback query failed rate B: {report['feedback_query_failed_rate_b']}",
            f"- avg critic score: A={report['avg_critic_score_a']}, B={report['avg_critic_score_b']}",
            f"- fallback rate: A={report['fallback_rate_a']}, B={report['fallback_rate_b']}",
            f"- hard violations: A={report['hard_violation_count_a']}, B={report['hard_violation_count_b']}",
            f"- quota fallback rate: A={report['quota_fallback_rate_a']}, B={report['quota_fallback_rate_b']}",
            f"- privacy leak detected: `{report['privacy_leak_detected']}`",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed = subparsers.add_parser("seed")
    seed.add_argument("--base-url", default=DEFAULT_BASE_URL)
    seed.add_argument("--user-id", default=DEFAULT_USER_ID)
    seed.set_defaults(func=cmd_seed)

    run = subparsers.add_parser("run")
    run.add_argument("--group", choices=["a", "b"], required=True)
    run.add_argument("--base-url", default=DEFAULT_BASE_URL)
    run.add_argument("--output", required=True)
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--user-id", default=DEFAULT_USER_ID)
    run.set_defaults(func=cmd_run)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--a", required=True)
    compare.add_argument("--b", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--markdown")
    compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

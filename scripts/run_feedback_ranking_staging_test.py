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
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import func, inspect
from sqlalchemy.orm import sessionmaker

import database
from models.routine_feedback import RoutineFeedback

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_USER_ID = "staging-fb-user-001"
DEFAULT_TIMEOUT = 120
SEED_PREFIX = "staging-seed-"

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

EXPECTED_SEED_COUNT = len(SEED_ROWS)
EXPECTED_SEED_IDS = [str(row["routine_draft_id"]) for row in SEED_ROWS]
EXPECTED_SKIPPED_IDS = {"34", "114"}
EXPECTED_EDITED_ORIGINAL_IDS = {"34", "37", "76", "99", "126"}
EXPECTED_EDITED_REPLACEMENT_IDS = {"40", "101", "107"}
DB_READY_STATUSES = {"pass", "created"}


def _engine(db_engine: Any | None = None) -> Any:
    return db_engine if db_engine is not None else database.engine


def _session(db_engine: Any | None = None):
    return sessionmaker(autocommit=False, autoflush=False, bind=_engine(db_engine))()


def _write_json_if_requested(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _production_like_env() -> bool:
    return any(
        os.getenv(name, "").strip().lower() == "production"
        for name in ("APP_ENV", "ENV", "FIT_CORE_ENV")
    )


def _db_error_category(exc: Exception) -> str:
    return type(exc).__name__


def check_routine_feedback_table_exists(db_engine: Any | None = None) -> bool:
    return bool(inspect(_engine(db_engine)).has_table(RoutineFeedback.__tablename__))


def ensure_routine_feedback_table(
    *,
    allow_create: bool = False,
    dry_run: bool = False,
    db_engine: Any | None = None,
) -> dict[str, Any]:
    result = {
        "db_readiness_status": "pass",
        "db_table_exists": False,
        "db_table_created": False,
        "allow_create_tables": allow_create,
        "production_like_blocked": False,
        "error_category": None,
    }
    try:
        exists = check_routine_feedback_table_exists(db_engine)
    except Exception as exc:
        result.update(
            {
                "db_readiness_status": "failed_connection",
                "error_category": _db_error_category(exc),
            }
        )
        return result

    result["db_table_exists"] = exists
    if exists:
        return result
    if not allow_create:
        result["db_readiness_status"] = "failed_missing_table"
        return result
    if _production_like_env():
        result.update(
            {
                "db_readiness_status": "failed_production_like_create_blocked",
                "production_like_blocked": True,
            }
        )
        return result
    if dry_run:
        result["db_readiness_status"] = "failed_missing_table"
        return result

    try:
        RoutineFeedback.__table__.create(bind=_engine(db_engine), checkfirst=True)
        result.update(
            {
                "db_readiness_status": "created",
                "db_table_exists": True,
                "db_table_created": True,
            }
        )
    except Exception as exc:
        result.update(
            {
                "db_readiness_status": "failed_connection",
                "error_category": _db_error_category(exc),
            }
        )
    return result


def _require_db_ready(readiness: dict[str, Any]) -> None:
    if readiness.get("db_readiness_status") not in DB_READY_STATUSES:
        raise SystemExit(f"DB readiness failed: {readiness['db_readiness_status']}")


def _seed_row_counts(db_engine: Any | None = None) -> dict[str, int]:
    db = _session(db_engine)
    try:
        rows = (
            db.query(RoutineFeedback.routine_draft_id, func.count(RoutineFeedback.id))
            .filter(RoutineFeedback.routine_draft_id.in_(EXPECTED_SEED_IDS))
            .group_by(RoutineFeedback.routine_draft_id)
            .all()
        )
        counts = {seed_id: 0 for seed_id in EXPECTED_SEED_IDS}
        counts.update({str(seed_id): int(count) for seed_id, count in rows})
        return counts
    finally:
        db.close()


def _count_prefixed_seed_rows(db_engine: Any | None = None) -> int:
    db = _session(db_engine)
    try:
        return int(
            db.query(RoutineFeedback)
            .filter(RoutineFeedback.routine_draft_id.like(f"{SEED_PREFIX}%"))
            .count()
        )
    finally:
        db.close()


def _read_seed_rows(db_engine: Any | None = None) -> list[RoutineFeedback]:
    db = _session(db_engine)
    try:
        rows = (
            db.query(RoutineFeedback)
            .filter(RoutineFeedback.routine_draft_id.in_(EXPECTED_SEED_IDS))
            .all()
        )
        for row in rows:
            db.expunge(row)
        return rows
    finally:
        db.close()


def verify_seed_readiness(db_engine: Any | None = None) -> dict[str, Any]:
    counts = _seed_row_counts(db_engine)
    missing = [seed_id for seed_id, count in counts.items() if count == 0]
    duplicates = [seed_id for seed_id, count in counts.items() if count > 1]
    rows = _read_seed_rows(db_engine)

    skipped: set[str] = set()
    edited_originals: set[str] = set()
    edited_replacements: set[str] = set()
    user_note_present = False
    for row in rows:
        if row.user_note:
            user_note_present = True
        if isinstance(row.skipped_exercise_ids, list):
            skipped.update(str(item) for item in row.skipped_exercise_ids)
        if isinstance(row.edited_exercises, list):
            for item in row.edited_exercises:
                if not isinstance(item, dict):
                    continue
                original = item.get("original_exercise_id")
                replacement = item.get("replacement_exercise_id")
                if original is not None:
                    edited_originals.add(str(original))
                if replacement is not None:
                    edited_replacements.add(str(replacement))

    payload_ok = (
        EXPECTED_SKIPPED_IDS.issubset(skipped)
        and EXPECTED_EDITED_ORIGINAL_IDS.issubset(edited_originals)
        and EXPECTED_EDITED_REPLACEMENT_IDS.issubset(edited_replacements)
    )
    status = "pass"
    if user_note_present:
        status = "failed_user_note_present"
    elif duplicates:
        status = "failed_duplicate_rows"
    elif missing:
        status = "failed_missing_rows"
    elif not payload_ok:
        status = "failed_unexpected_payload"

    return {
        "seed_readiness_status": status,
        "seed_verification_passed": status == "pass",
        "seed_rows_expected": EXPECTED_SEED_COUNT,
        "seed_rows_found": sum(counts.values()),
        "seed_rows_existing": sum(1 for count in counts.values() if count >= 1),
        "missing_seed_count": len(missing),
    }


def _delete_seed_rows(db_engine: Any | None = None) -> int:
    db = _session(db_engine)
    try:
        deleted = (
            db.query(RoutineFeedback)
            .filter(RoutineFeedback.routine_draft_id.like(f"{SEED_PREFIX}%"))
            .delete(synchronize_session=False)
        )
        db.commit()
        return int(deleted)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _seed_summary_defaults(*, dry_run: bool, reset: bool) -> dict[str, Any]:
    return {
        "seed_readiness_status": "pass",
        "seed_rows_expected": EXPECTED_SEED_COUNT,
        "seed_rows_existing": 0,
        "seed_rows_inserted": 0,
        "seed_rows_deleted": 0,
        "seed_reset_performed": reset and not dry_run,
        "seed_verification_passed": False,
        "dry_run": dry_run,
    }


def perform_seed(
    *,
    base_url: str,
    user_id: str,
    reset_seed: bool = False,
    allow_create_tables: bool = False,
    dry_run: bool = False,
    db_engine: Any | None = None,
) -> dict[str, Any]:
    db_readiness = ensure_routine_feedback_table(
        allow_create=allow_create_tables,
        dry_run=dry_run,
        db_engine=db_engine,
    )
    _require_db_ready(db_readiness)

    counts = _seed_row_counts(db_engine)
    duplicate_ids = [seed_id for seed_id, count in counts.items() if count > 1]
    if duplicate_ids and not reset_seed:
        result = _seed_summary_defaults(dry_run=dry_run, reset=reset_seed)
        result.update(db_readiness)
        result.update(
            {
                "seed_readiness_status": "failed_duplicate_seed_rows",
                "seed_rows_existing": sum(counts.values()),
            }
        )
        raise SystemExit(json.dumps(result, indent=2))

    result = _seed_summary_defaults(dry_run=dry_run, reset=reset_seed)
    result.update(db_readiness)

    planned_deletes = _count_prefixed_seed_rows(db_engine) if reset_seed else 0
    if reset_seed:
        result["seed_rows_deleted"] = planned_deletes
        if not dry_run:
            result["seed_rows_deleted"] = _delete_seed_rows(db_engine)
            counts = {seed_id: 0 for seed_id in EXPECTED_SEED_IDS}

    missing_rows = [row for row in SEED_ROWS if counts.get(str(row["routine_draft_id"]), 0) == 0]
    result["seed_rows_existing"] = sum(1 for count in counts.values() if count >= 1)
    result["seed_rows_inserted"] = len(missing_rows)
    result["planned_seed_rows"] = len(missing_rows)
    result["planned_deletes"] = planned_deletes

    if dry_run:
        result["seed_readiness_status"] = "dry_run"
        return result

    for row in missing_rows:
        payload = dict(row)
        payload["user_id"] = user_id
        try:
            _json_request("POST", f"{base_url.rstrip('/')}/api/ai/routine-feedback", payload, timeout=20)
        except Exception as exc:
            raise SystemExit(f"seed failed for {payload['routine_draft_id']}: {type(exc).__name__}: {exc}") from exc

    verification = verify_seed_readiness(db_engine)
    result.update(verification)
    return result


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


def _post_generation(
    base_url: str,
    scenario: dict[str, Any],
    *,
    timeout: int = DEFAULT_TIMEOUT,
    group: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    started = time.perf_counter()
    try:
        response = _json_request(
            "POST",
            f"{base_url.rstrip('/')}/api/ai/generate-routine",
            scenario["payload"],
            timeout=timeout,
        )
        return response, None
    except urllib.error.HTTPError as exc:
        return None, {
            "scenario_id": scenario["id"],
            "group": group,
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "error_type": type(exc).__name__,
            "status_code": exc.code,
        }
    except Exception as exc:
        return None, {
            "scenario_id": scenario["id"],
            "group": group,
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "error_type": type(exc).__name__,
        }


def _run_one(base_url: str, scenario: dict[str, Any], timeout: int = DEFAULT_TIMEOUT, group: str | None = None) -> dict[str, Any]:
    response, error = _post_generation(base_url, scenario, timeout=timeout, group=group)
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
        "candidate_count",
        "candidate_payload_char_count",
        "approx_candidate_payload_tokens",
        "dynamic_temperature_enabled",
        "generation_temperature",
        "repair_count",
        "schema_repair_attempted",
        "schema_repair_succeeded",
        "schema_repair_latency_ms",
        "exercise_count",
        "total_estimated_time",
        "target_duration_min",
        "hard_violation_categories",
        "hard_violation_category_counts",
    ]
    return {key: event.get(key) for key in keys}


def _privacy_leak_detected(value: Any) -> bool:
    text = json.dumps(value, ensure_ascii=False).lower()
    return any(token.lower() in text for token in FORBIDDEN_PRIVACY_TOKENS)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(value for value in values if isinstance(value, (int, float)))
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _successful_generation_latencies(results: list[dict[str, Any]]) -> list[float]:
    return [
        result["telemetry"]["generation_latency_ms"]
        for result in results
        if isinstance(result.get("telemetry"), dict)
        and isinstance(result["telemetry"].get("generation_latency_ms"), (int, float))
    ]


def _is_timeout_error(error: Any) -> bool:
    if not isinstance(error, dict):
        return False
    error_type = str(error.get("error_type") or error.get("error") or "").lower()
    return "timeout" in error_type


def _timeout_scenarios(results: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            str(result.get("scenario_id"))
            for result in results
            if _is_timeout_error(result.get("error")) and result.get("scenario_id")
        }
    )


def _timeout_count(results: list[dict[str, Any]]) -> int:
    return sum(1 for result in results if _is_timeout_error(result.get("error")))


def _telemetry_missing_count(results: list[dict[str, Any]]) -> int:
    return sum(1 for result in results if not isinstance(result.get("telemetry"), dict))


def _hard_violation_count_by_scenario(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for result in results:
        telemetry = result.get("telemetry")
        if not isinstance(telemetry, dict):
            continue
        count = int(telemetry.get("hard_violation_count") or 0)
        if count > 0:
            counts[str(result.get("scenario_id"))] += count
    return dict(sorted(counts.items()))


_ALLOWED_HARD_VIOLATION_CATEGORIES = {
    "equipment",
    "pain_trigger",
    "doms_readiness",
    "set_cap",
    "hallucinated_exercise",
    "time_budget",
    "rationale_missing",
    "schema_malformed",
    "unknown",
}


def _hard_violation_category_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for result in results:
        telemetry = result.get("telemetry")
        if not isinstance(telemetry, dict):
            continue
        hard_count = int(telemetry.get("hard_violation_count") or 0)
        if hard_count <= 0:
            continue

        supplied_counts = telemetry.get("hard_violation_category_counts")
        if isinstance(supplied_counts, dict):
            for category, count in supplied_counts.items():
                key = str(category) if str(category) in _ALLOWED_HARD_VIOLATION_CATEGORIES else "unknown"
                counts[key] += int(count or 0)
            continue

        supplied_categories = telemetry.get("hard_violation_categories")
        if isinstance(supplied_categories, list) and supplied_categories:
            added = 0
            for category in supplied_categories:
                key = str(category) if str(category) in _ALLOWED_HARD_VIOLATION_CATEGORIES else "unknown"
                counts[key] += 1
                added += 1
            if added >= hard_count:
                continue
            counts["unknown"] += hard_count - added
            continue

        counts["unknown"] += hard_count
    return dict(sorted(counts.items()))


def _run_summary(results: list[dict[str, Any]], request_timeout: int) -> dict[str, Any]:
    latencies = _successful_generation_latencies(results)
    return {
        "failed_request_count": sum(1 for result in results if result.get("error")),
        "timeout_count": _timeout_count(results),
        "timeout_scenarios": _timeout_scenarios(results),
        "request_timeout_sec": request_timeout,
        "successful_latency_p50_ms": _percentile(latencies, 0.50),
        "successful_latency_p95_ms": _percentile(latencies, 0.95),
        "successful_latency_max_ms": max(latencies) if latencies else 0.0,
        "telemetry_missing_count": _telemetry_missing_count(results),
        "hard_violation_count_by_scenario": _hard_violation_count_by_scenario(results),
        "hard_violation_category_counts": _hard_violation_category_counts(results),
        "hard_violation_scenarios": sorted(_hard_violation_count_by_scenario(results)),
    }


def cmd_seed(args: argparse.Namespace) -> None:
    result = perform_seed(
        base_url=args.base_url,
        user_id=args.user_id,
        reset_seed=args.reset_seed,
        allow_create_tables=args.allow_create_tables,
        dry_run=args.dry_run,
    )
    _write_json_if_requested(args.output, result)
    print(json.dumps(result, indent=2))


def cmd_check_db(args: argparse.Namespace) -> None:
    result = ensure_routine_feedback_table(
        allow_create=args.allow_create_tables,
        dry_run=args.dry_run,
    )
    _write_json_if_requested(args.output, result)
    print(json.dumps(result, indent=2))
    _require_db_ready(result)


def cmd_verify_seed(args: argparse.Namespace) -> None:
    db_readiness = ensure_routine_feedback_table()
    _require_db_ready(db_readiness)
    result = dict(db_readiness)
    result.update(verify_seed_readiness())
    _write_json_if_requested(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["seed_verification_passed"]:
        raise SystemExit(f"seed verification failed: {result['seed_readiness_status']}")


def cmd_run(args: argparse.Namespace) -> None:
    expected_enabled = args.group == "b"
    scenarios = [dict(item) for item in BASE_SCENARIOS + SEEDED_SCENARIOS]
    for scenario in scenarios:
        if scenario["id"].endswith("_fb"):
            scenario["payload"] = dict(scenario["payload"], userId=args.user_id)

    db_readiness = ensure_routine_feedback_table()
    _require_db_ready(db_readiness)
    seed_readiness = verify_seed_readiness()
    if not seed_readiness["seed_verification_passed"]:
        raise SystemExit(f"seed verification failed: {seed_readiness['seed_readiness_status']}")

    probe = _run_one(args.base_url, scenarios[-2], args.request_timeout, args.group)
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
            result = _run_one(args.base_url, scenario, args.request_timeout, args.group)
            if result.get("error"):
                failed.append(result)
            results.append(result)

    output = {
        "mode": "feedback_ranking_staging",
        "group": args.group,
        "llm_provider": os.getenv("LLM_PROVIDER"),
        "local_llm_model": os.getenv("LOCAL_LLM_MODEL"),
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL"),
        "request_count": len(results),
        "failed_requests": failed,
        "results": results,
        "privacy_leak_detected": _privacy_leak_detected(results),
        **_run_summary(results, args.request_timeout),
        **db_readiness,
        **{
            "seed_readiness_status": seed_readiness["seed_readiness_status"],
            "seed_verification_passed": seed_readiness["seed_verification_passed"],
            "seed_rows_expected": seed_readiness["seed_rows_expected"],
            "seed_rows_existing": seed_readiness["seed_rows_existing"],
            "seed_rows_inserted": 0,
            "seed_rows_deleted": 0,
            "seed_reset_performed": False,
            "missing_seed_count": seed_readiness["missing_seed_count"],
        },
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


def _max(events: list[dict[str, Any]], key: str) -> float:
    values = [event.get(key) for event in events if isinstance(event.get(key), (int, float))]
    return max(values) if values else 0.0


def _fallback_rate(events: list[dict[str, Any]]) -> float:
    return _rate(events, "fallback_used")


def _quota_rate(events: list[dict[str, Any]]) -> float:
    return sum(1 for event in events if event.get("fallback_reason") == "quota_exhausted") / max(len(events), 1)


def _result_count(raw: dict[str, Any]) -> int:
    if isinstance(raw.get("request_count"), int):
        return int(raw["request_count"])
    return len(raw.get("results", []))


def _failed_request_count(raw: dict[str, Any]) -> int:
    if isinstance(raw.get("failed_request_count"), int):
        return int(raw["failed_request_count"])
    return len(raw.get("failed_requests") or [])


def _raw_timeout_count(raw: dict[str, Any]) -> int:
    if isinstance(raw.get("timeout_count"), int):
        return int(raw["timeout_count"])
    return _timeout_count(raw.get("results", []))


def _raw_telemetry_missing_count(raw: dict[str, Any], events: list[dict[str, Any]]) -> int:
    if isinstance(raw.get("telemetry_missing_count"), int):
        return int(raw["telemetry_missing_count"])
    return max(0, _result_count(raw) - len(events))


def _counter_for_events(events: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = Counter(str(event.get(key)) for event in events)
    return dict(sorted(counts.items()))


def _is_local_gemma4_run(report: dict[str, Any]) -> bool:
    provider_values = {
        str(report.get("llm_provider_a") or "").lower(),
        str(report.get("llm_provider_b") or "").lower(),
    }
    model_values = {
        str(report.get("local_llm_model_a") or "").lower(),
        str(report.get("local_llm_model_b") or "").lower(),
    }
    return "local" in provider_values and any("gemma4" in model for model in model_values)


def _recommend(report: dict[str, Any]) -> str:
    if report.get("readiness_blocked"):
        return "inconclusive_readiness_failed"
    if report.get("seed_verification_passed_b") is False:
        return "inconclusive_seed_not_ready"
    if report["privacy_leak_detected"]:
        return "keep_feedback_off"
    if report["hard_violation_count_b"] > 0:
        return "keep_feedback_off"
    if report["feedback_query_failed_rate_b"] > 0:
        return "keep_feedback_off"
    if _is_local_gemma4_run(report):
        total_timeouts = int(report.get("timeout_count_a") or 0) + int(report.get("timeout_count_b") or 0)
        total_requests = int(report.get("submitted_request_count_a") or 0) + int(report.get("submitted_request_count_b") or 0)
        timeout_rate = total_timeouts / max(total_requests, 1)
        if total_timeouts > 0 and timeout_rate > 0:
            return "inconclusive_local_timeout"
        if int(report.get("hard_violation_count_a") or 0) > 0:
            return "inconclusive_gemma4_baseline_unstable"
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
        "submitted_request_count_a": _result_count(raw_a),
        "submitted_request_count_b": _result_count(raw_b),
        "failed_request_count_a": _failed_request_count(raw_a),
        "failed_request_count_b": _failed_request_count(raw_b),
        "timeout_count_a": _raw_timeout_count(raw_a),
        "timeout_count_b": _raw_timeout_count(raw_b),
        "timeout_delta": _raw_timeout_count(raw_b) - _raw_timeout_count(raw_a),
        "telemetry_missing_count_a": _raw_telemetry_missing_count(raw_a, events_a),
        "telemetry_missing_count_b": _raw_telemetry_missing_count(raw_b, events_b),
        "p95_latency_a": raw_a.get("successful_latency_p95_ms", _p95(events_a, "generation_latency_ms")),
        "p95_latency_b": raw_b.get("successful_latency_p95_ms", _p95(events_b, "generation_latency_ms")),
        "max_latency_a": raw_a.get("successful_latency_max_ms", _max(events_a, "generation_latency_ms")),
        "max_latency_b": raw_b.get("successful_latency_max_ms", _max(events_b, "generation_latency_ms")),
        "request_timeout_sec_a": raw_a.get("request_timeout_sec"),
        "request_timeout_sec_b": raw_b.get("request_timeout_sec"),
        "llm_provider_a": raw_a.get("llm_provider"),
        "llm_provider_b": raw_b.get("llm_provider"),
        "local_llm_model_a": raw_a.get("local_llm_model"),
        "local_llm_model_b": raw_b.get("local_llm_model"),
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
        "hard_violation_count_by_scenario_a": raw_a.get(
            "hard_violation_count_by_scenario",
            _hard_violation_count_by_scenario(raw_a.get("results", [])),
        ),
        "hard_violation_count_by_scenario_b": raw_b.get(
            "hard_violation_count_by_scenario",
            _hard_violation_count_by_scenario(raw_b.get("results", [])),
        ),
        "hard_violation_category_counts_a": raw_a.get(
            "hard_violation_category_counts",
            _hard_violation_category_counts(raw_a.get("results", [])),
        ),
        "hard_violation_category_counts_b": raw_b.get(
            "hard_violation_category_counts",
            _hard_violation_category_counts(raw_b.get("results", [])),
        ),
        "hard_violation_scenarios_a": raw_a.get(
            "hard_violation_scenarios",
            sorted(_hard_violation_count_by_scenario(raw_a.get("results", []))),
        ),
        "hard_violation_scenarios_b": raw_b.get(
            "hard_violation_scenarios",
            sorted(_hard_violation_count_by_scenario(raw_b.get("results", []))),
        ),
        "control_hard_violation_count": sum(int(event.get("hard_violation_count") or 0) for event in events_a),
        "treatment_hard_violation_count": sum(int(event.get("hard_violation_count") or 0) for event in events_b),
        "critic_grade_distribution_a": _counter_for_events(events_a, "critic_grade"),
        "critic_grade_distribution_b": _counter_for_events(events_b, "critic_grade"),
        "fallback_reason_distribution_a": _counter_for_events(events_a, "fallback_reason"),
        "fallback_reason_distribution_b": _counter_for_events(events_b, "fallback_reason"),
        "llm_error_type_distribution_a": _counter_for_events(events_a, "llm_error_type"),
        "llm_error_type_distribution_b": _counter_for_events(events_b, "llm_error_type"),
        "schema_repair_attempted_count_a": sum(1 for event in events_a if event.get("schema_repair_attempted") is True),
        "schema_repair_attempted_count_b": sum(1 for event in events_b if event.get("schema_repair_attempted") is True),
        "schema_repair_succeeded_count_a": sum(1 for event in events_a if event.get("schema_repair_succeeded") is True),
        "schema_repair_succeeded_count_b": sum(1 for event in events_b if event.get("schema_repair_succeeded") is True),
        "quota_fallback_rate_a": _quota_rate(events_a),
        "quota_fallback_rate_b": _quota_rate(events_b),
        "privacy_leak_detected": bool(raw_a.get("privacy_leak_detected")) or bool(raw_b.get("privacy_leak_detected")),
        "db_readiness_status_a": raw_a.get("db_readiness_status"),
        "db_readiness_status_b": raw_b.get("db_readiness_status"),
        "db_table_exists_a": raw_a.get("db_table_exists"),
        "db_table_exists_b": raw_b.get("db_table_exists"),
        "db_table_created_a": raw_a.get("db_table_created"),
        "db_table_created_b": raw_b.get("db_table_created"),
        "seed_readiness_status_a": raw_a.get("seed_readiness_status"),
        "seed_readiness_status_b": raw_b.get("seed_readiness_status"),
        "seed_verification_passed_a": raw_a.get("seed_verification_passed"),
        "seed_verification_passed_b": raw_b.get("seed_verification_passed"),
        "missing_seed_count": int(raw_a.get("missing_seed_count") or 0) + int(raw_b.get("missing_seed_count") or 0),
    }
    report["readiness_blocked"] = any(
        status not in DB_READY_STATUSES
        for status in (report["db_readiness_status_a"], report["db_readiness_status_b"])
    )
    report["hard_violation_count_by_scenario"] = {
        "a": report["hard_violation_count_by_scenario_a"],
        "b": report["hard_violation_count_by_scenario_b"],
    }
    report["hard_violation_category_counts"] = {
        "a": report["hard_violation_category_counts_a"],
        "b": report["hard_violation_category_counts_b"],
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
            f"- submitted requests: A={report['submitted_request_count_a']}, B={report['submitted_request_count_b']}",
            f"- failed requests: A={report['failed_request_count_a']}, B={report['failed_request_count_b']}",
            f"- timeouts: A={report['timeout_count_a']}, B={report['timeout_count_b']}, delta={report['timeout_delta']}",
            f"- telemetry missing: A={report['telemetry_missing_count_a']}, B={report['telemetry_missing_count_b']}",
            f"- generation p95 latency: A={report['p95_latency_a']} ms, B={report['p95_latency_b']} ms",
            f"- generation max latency: A={report['max_latency_a']} ms, B={report['max_latency_b']} ms",
            f"- feedback enabled rate: A={report['feedback_enabled_rate_a']}, B={report['feedback_enabled_rate_b']}",
            f"- avg adjusted candidates: A={report['avg_feedback_adjusted_count_a']}, B={report['avg_feedback_adjusted_count_b']}",
            f"- p95 feedback query latency B: {report['p95_feedback_query_latency_ms_b']} ms",
            f"- feedback query failed rate B: {report['feedback_query_failed_rate_b']}",
            f"- avg critic score: A={report['avg_critic_score_a']}, B={report['avg_critic_score_b']}",
            f"- fallback rate: A={report['fallback_rate_a']}, B={report['fallback_rate_b']}",
            f"- hard violations: A={report['hard_violation_count_a']}, B={report['hard_violation_count_b']}",
            f"- hard violation scenarios A: `{report['hard_violation_scenarios_a']}`",
            f"- hard violation scenarios B: `{report['hard_violation_scenarios_b']}`",
            f"- quota fallback rate: A={report['quota_fallback_rate_a']}, B={report['quota_fallback_rate_b']}",
            f"- DB readiness: A={report['db_readiness_status_a']}, B={report['db_readiness_status_b']}",
            f"- seed readiness: A={report['seed_readiness_status_a']}, B={report['seed_readiness_status_b']}",
            f"- seed verification passed: A={report['seed_verification_passed_a']}, B={report['seed_verification_passed_b']}",
            f"- privacy leak detected: `{report['privacy_leak_detected']}`",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_db = subparsers.add_parser("check-db")
    check_db.add_argument("--allow-create-tables", action="store_true")
    check_db.add_argument("--dry-run", action="store_true")
    check_db.add_argument("--output")
    check_db.set_defaults(func=cmd_check_db)

    seed = subparsers.add_parser("seed")
    seed.add_argument("--base-url", default=DEFAULT_BASE_URL)
    seed.add_argument("--user-id", default=DEFAULT_USER_ID)
    seed.add_argument("--reset-seed", action="store_true")
    seed.add_argument("--allow-create-tables", action="store_true")
    seed.add_argument("--dry-run", action="store_true")
    seed.add_argument("--output")
    seed.set_defaults(func=cmd_seed)

    verify_seed = subparsers.add_parser("verify-seed")
    verify_seed.add_argument("--output")
    verify_seed.set_defaults(func=cmd_verify_seed)

    run = subparsers.add_parser("run")
    run.add_argument("--group", choices=["a", "b"], required=True)
    run.add_argument("--base-url", default=DEFAULT_BASE_URL)
    run.add_argument("--output", required=True)
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--user-id", default=DEFAULT_USER_ID)
    run.add_argument("--request-timeout", type=int, default=DEFAULT_TIMEOUT)
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

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database import Base
from models.routine_feedback import RoutineFeedback
from scripts import run_feedback_ranking_staging_test as staging


@pytest.fixture
def sqlite_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    try:
        yield engine
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture
def ready_engine(sqlite_engine):
    RoutineFeedback.__table__.create(bind=sqlite_engine, checkfirst=True)
    return sqlite_engine


def session_for(engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)()


def add_feedback(engine, **kwargs):
    db = session_for(engine)
    row = RoutineFeedback(
        routine_draft_id=kwargs.pop("routine_draft_id", "draft-1"),
        user_id=kwargs.pop("user_id", "user-1"),
        rating=kwargs.pop("rating", 4),
        completed=kwargs.pop("completed", True),
        accepted_without_edits=kwargs.pop("accepted_without_edits", True),
        skipped_exercise_ids=kwargs.pop("skipped_exercise_ids", []),
        edited_exercises=kwargs.pop("edited_exercises", []),
        user_note=kwargs.pop("user_note", None),
        source=kwargs.pop("source", "api"),
    )
    db.add(row)
    db.commit()
    db.close()


def count_rows(engine, prefix: str | None = None) -> int:
    db = session_for(engine)
    query = db.query(RoutineFeedback)
    if prefix:
        query = query.filter(RoutineFeedback.routine_draft_id.like(f"{prefix}%"))
    count = query.count()
    db.close()
    return count


def install_fake_seed_api(monkeypatch, engine):
    def fake_request(method, url, payload=None, timeout=120):
        assert method == "POST"
        assert "user_note" not in payload
        assert "userNote" not in payload
        db = session_for(engine)
        db.add(
            RoutineFeedback(
                routine_draft_id=payload["routine_draft_id"],
                user_id=payload.get("user_id"),
                rating=payload.get("rating"),
                completed=payload.get("completed"),
                accepted_without_edits=payload.get("accepted_without_edits"),
                skipped_exercise_ids=payload.get("skipped_exercises"),
                edited_exercises=payload.get("edited_exercises"),
                source="api",
            )
        )
        db.commit()
        db.close()
        return {"feedback_id": "fake", "stored_at": "2026-01-01T00:00:00"}

    monkeypatch.setattr(staging, "_json_request", fake_request)


def seed_all(monkeypatch, engine):
    install_fake_seed_api(monkeypatch, engine)
    return staging.perform_seed(
        base_url="http://test",
        user_id="seed-user",
        db_engine=engine,
    )


def test_db_readiness_table_exists_passes(ready_engine):
    result = staging.ensure_routine_feedback_table(db_engine=ready_engine)

    assert result["db_readiness_status"] == "pass"
    assert result["db_table_exists"] is True


def test_db_readiness_missing_without_allow_create_fails(sqlite_engine):
    result = staging.ensure_routine_feedback_table(db_engine=sqlite_engine)

    assert result["db_readiness_status"] == "failed_missing_table"
    assert result["db_table_exists"] is False


def test_db_readiness_missing_with_allow_create_creates(sqlite_engine, monkeypatch):
    monkeypatch.delenv("APP_ENV", raising=False)
    result = staging.ensure_routine_feedback_table(allow_create=True, db_engine=sqlite_engine)

    assert result["db_readiness_status"] == "created"
    assert result["db_table_created"] is True
    assert staging.check_routine_feedback_table_exists(sqlite_engine) is True


def test_db_readiness_allow_create_blocked_in_production(sqlite_engine, monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")

    result = staging.ensure_routine_feedback_table(allow_create=True, db_engine=sqlite_engine)

    assert result["db_readiness_status"] == "failed_production_like_create_blocked"
    assert result["production_like_blocked"] is True


def test_db_readiness_inspection_error_is_sanitized(ready_engine, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "mysql://secret:password@example/prod")

    def boom(engine):
        raise RuntimeError("mysql://secret:password@example/prod")

    monkeypatch.setattr(staging, "inspect", boom)

    result = staging.ensure_routine_feedback_table(db_engine=ready_engine)

    assert result["db_readiness_status"] == "failed_connection"
    assert result["error_category"] == "RuntimeError"
    assert "secret" not in json.dumps(result)
    assert "password" not in json.dumps(result)


def test_first_seed_inserts_7_rows(ready_engine, monkeypatch):
    result = seed_all(monkeypatch, ready_engine)

    assert result["seed_rows_inserted"] == 7
    assert count_rows(ready_engine, staging.SEED_PREFIX) == 7


def test_repeated_seed_without_reset_inserts_zero(ready_engine, monkeypatch):
    seed_all(monkeypatch, ready_engine)
    result = seed_all(monkeypatch, ready_engine)

    assert result["seed_rows_inserted"] == 0


def test_repeated_seed_without_reset_does_not_duplicate(ready_engine, monkeypatch):
    seed_all(monkeypatch, ready_engine)
    seed_all(monkeypatch, ready_engine)

    assert count_rows(ready_engine, staging.SEED_PREFIX) == 7


def test_reset_seed_deletes_seed_rows_and_reinserts(ready_engine, monkeypatch):
    seed_all(monkeypatch, ready_engine)
    install_fake_seed_api(monkeypatch, ready_engine)

    result = staging.perform_seed(
        base_url="http://test",
        user_id="seed-user",
        reset_seed=True,
        db_engine=ready_engine,
    )

    assert result["seed_rows_deleted"] == 7
    assert result["seed_rows_inserted"] == 7
    assert count_rows(ready_engine, staging.SEED_PREFIX) == 7


def test_reset_seed_preserves_non_seed_feedback(ready_engine, monkeypatch):
    add_feedback(ready_engine, routine_draft_id="real-user-draft")
    seed_all(monkeypatch, ready_engine)
    install_fake_seed_api(monkeypatch, ready_engine)

    staging.perform_seed(base_url="http://test", user_id="seed-user", reset_seed=True, db_engine=ready_engine)

    assert count_rows(ready_engine) == 8


def test_duplicate_seed_without_reset_fails(ready_engine, monkeypatch):
    add_feedback(ready_engine, routine_draft_id="staging-seed-001")
    add_feedback(ready_engine, routine_draft_id="staging-seed-001")
    install_fake_seed_api(monkeypatch, ready_engine)

    with pytest.raises(SystemExit):
        staging.perform_seed(base_url="http://test", user_id="seed-user", db_engine=ready_engine)


def test_dry_run_seed_reports_changes_without_mutation(ready_engine, monkeypatch):
    install_fake_seed_api(monkeypatch, ready_engine)

    result = staging.perform_seed(
        base_url="http://test",
        user_id="seed-user",
        dry_run=True,
        db_engine=ready_engine,
    )

    assert result["seed_readiness_status"] == "dry_run"
    assert result["planned_seed_rows"] == 7
    assert count_rows(ready_engine) == 0


def test_verify_seed_passes_after_seed(ready_engine, monkeypatch):
    seed_all(monkeypatch, ready_engine)

    result = staging.verify_seed_readiness(ready_engine)

    assert result["seed_verification_passed"] is True
    assert result["seed_readiness_status"] == "pass"


def test_verify_seed_fails_when_rows_missing(ready_engine):
    result = staging.verify_seed_readiness(ready_engine)

    assert result["seed_readiness_status"] == "failed_missing_rows"
    assert result["missing_seed_count"] == 7


def test_verify_seed_fails_when_duplicate_rows_exist(ready_engine):
    add_feedback(ready_engine, routine_draft_id="staging-seed-001")
    add_feedback(ready_engine, routine_draft_id="staging-seed-001")

    result = staging.verify_seed_readiness(ready_engine)

    assert result["seed_readiness_status"] == "failed_duplicate_rows"


def test_verify_seed_fails_when_expected_exercise_ids_missing(ready_engine, monkeypatch):
    seed_all(monkeypatch, ready_engine)
    db = session_for(ready_engine)
    row = db.query(RoutineFeedback).filter_by(routine_draft_id="staging-seed-003").one()
    row.skipped_exercise_ids = ["not-34"]
    db.commit()
    db.close()

    result = staging.verify_seed_readiness(ready_engine)

    assert result["seed_readiness_status"] == "failed_unexpected_payload"


def test_verify_seed_fails_if_seed_row_has_user_note(ready_engine, monkeypatch):
    seed_all(monkeypatch, ready_engine)
    db = session_for(ready_engine)
    row = db.query(RoutineFeedback).filter_by(routine_draft_id="staging-seed-001").one()
    row.user_note = "private"
    db.commit()
    db.close()

    result = staging.verify_seed_readiness(ready_engine)

    assert result["seed_readiness_status"] == "failed_user_note_present"


@pytest.mark.parametrize("command", ["check-db", "verify-seed"])
def test_new_help_commands_work(command):
    result = subprocess.run(
        [sys.executable, "scripts/run_feedback_ranking_staging_test.py", command, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout


def test_seed_help_includes_readiness_options():
    result = subprocess.run(
        [sys.executable, "scripts/run_feedback_ranking_staging_test.py", "seed", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    for option in ["--reset-seed", "--allow-create-tables", "--dry-run", "--output"]:
        assert option in result.stdout


def test_check_db_help_includes_create_dry_run_output_options():
    result = subprocess.run(
        [sys.executable, "scripts/run_feedback_ranking_staging_test.py", "check-db", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    for option in ["--allow-create-tables", "--dry-run", "--output"]:
        assert option in result.stdout


def test_run_output_includes_readiness_fields(tmp_path, monkeypatch):
    output = tmp_path / "run.json"
    monkeypatch.setattr(staging, "ensure_routine_feedback_table", lambda: {
        "db_readiness_status": "pass",
        "db_table_exists": True,
        "db_table_created": False,
        "allow_create_tables": False,
        "production_like_blocked": False,
        "error_category": None,
    })
    monkeypatch.setattr(staging, "verify_seed_readiness", lambda: {
        "seed_readiness_status": "pass",
        "seed_verification_passed": True,
        "seed_rows_expected": 7,
        "seed_rows_found": 7,
        "seed_rows_existing": 7,
        "missing_seed_count": 0,
    })
    monkeypatch.setattr(staging, "_run_one", lambda base_url, scenario, timeout=120, group=None: {
        "scenario_id": scenario["id"],
        "telemetry": {
            "feedback_enabled": False,
            "candidate_pool_size": 12,
            "fallback_used": False,
            "hard_violation_count": 0,
        },
    })

    staging.cmd_run(
        argparse.Namespace(
            group="a",
            base_url="http://test",
            output=str(output),
            repeats=1,
            user_id="seed-user",
            request_timeout=120,
        )
    )

    raw = json.loads(output.read_text(encoding="utf-8"))
    for field in [
        "db_readiness_status",
        "db_table_exists",
        "db_table_created",
        "seed_readiness_status",
        "seed_verification_passed",
        "seed_rows_expected",
        "seed_rows_existing",
        "seed_rows_inserted",
        "seed_rows_deleted",
        "seed_reset_performed",
    ]:
        assert field in raw


def write_raw_report(path: Path, **overrides):
    payload = {
        "results": [],
        "privacy_leak_detected": False,
        "db_readiness_status": "pass",
        "db_table_exists": True,
        "db_table_created": False,
        "seed_readiness_status": "pass",
        "seed_verification_passed": True,
        "missing_seed_count": 0,
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compare_output_includes_readiness_fields(tmp_path):
    raw_a = tmp_path / "a.json"
    raw_b = tmp_path / "b.json"
    out = tmp_path / "compare.json"
    write_raw_report(raw_a)
    write_raw_report(raw_b)

    staging.cmd_compare(argparse.Namespace(a=str(raw_a), b=str(raw_b), output=str(out), markdown=None))

    report = json.loads(out.read_text(encoding="utf-8"))
    for field in [
        "db_readiness_status_a",
        "db_readiness_status_b",
        "db_table_exists_a",
        "db_table_exists_b",
        "db_table_created_a",
        "db_table_created_b",
        "seed_readiness_status_a",
        "seed_readiness_status_b",
        "seed_verification_passed_a",
        "seed_verification_passed_b",
        "readiness_blocked",
        "missing_seed_count",
    ]:
        assert field in report


def test_compare_readiness_failure_recommends_inconclusive(tmp_path):
    raw_a = tmp_path / "a.json"
    raw_b = tmp_path / "b.json"
    out = tmp_path / "compare.json"
    write_raw_report(raw_a, db_readiness_status="failed_missing_table")
    write_raw_report(raw_b)

    staging.cmd_compare(argparse.Namespace(a=str(raw_a), b=str(raw_b), output=str(out), markdown=None))

    assert json.loads(out.read_text(encoding="utf-8"))["recommendation"] == "inconclusive_readiness_failed"


def test_compare_b_seed_failure_recommends_inconclusive_seed_not_ready(tmp_path):
    raw_a = tmp_path / "a.json"
    raw_b = tmp_path / "b.json"
    out = tmp_path / "compare.json"
    write_raw_report(raw_a)
    write_raw_report(raw_b, seed_verification_passed=False)

    staging.cmd_compare(argparse.Namespace(a=str(raw_a), b=str(raw_b), output=str(out), markdown=None))

    assert json.loads(out.read_text(encoding="utf-8"))["recommendation"] == "inconclusive_seed_not_ready"


def test_seed_report_omits_forbidden_private_keys(ready_engine):
    result = staging.perform_seed(
        base_url="http://test",
        user_id="seed-user",
        dry_run=True,
        db_engine=ready_engine,
    )
    text = json.dumps(result)

    for forbidden in ["user_id", "user_note", "userNote", "skipped_exercise_ids", "edited_exercises"]:
        assert forbidden not in text

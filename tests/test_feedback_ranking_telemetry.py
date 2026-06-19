from __future__ import annotations

import subprocess
import sys

import pytest

from engines.feedback_aggregation import FeedbackAdjustment
from engines.routine_pipeline import _compute_feedback_stats
from engines.routine_telemetry import build_routine_quality_telemetry_payload
from engines.schemas import RoutineBlock, RoutineDraftResponse, SetPrescription


class _Critic:
    score = 90
    grade = "PASS"
    warnings = []
    hard_violations = []
    should_rebuild = False
    should_fallback = False


def _response() -> RoutineDraftResponse:
    return RoutineDraftResponse(
        generation_status="success",
        status_reason_code="none",
        is_fallback=False,
        total_estimated_time=40,
        summary_title="Test",
        rationale_summary=["summary"],
        routine_blocks=[
            RoutineBlock(
                order=1,
                exercise_id="bench",
                exercise_name="Bench",
                primary_muscles=["chest"],
                equipment_type="DUMBBELL",
                default_rest_sec=90,
                prescription=[SetPrescription(set_index=1, target_reps=10, target_rest_sec=90)],
                exercise_rationale="summary",
            )
        ],
        warnings=[],
    )


def _payload(**overrides):
    values = {
        "response": _response(),
        "candidate_pool_size": 12,
        "ranked_candidates": [{"id": "bench"}],
        "candidate_payload": "candidate",
        "generation_temperature": 0.0,
        "critic_result": _Critic(),
        "repair_count": 0,
        "fallback_used": False,
        "target_duration_min": 40,
    }
    values.update(overrides)
    return build_routine_quality_telemetry_payload(**values)


def test_telemetry_payload_includes_feedback_fields():
    payload = _payload()

    assert payload["feedback_enabled"] is False
    assert payload["feedback_adjusted_candidate_count"] == 0
    assert payload["feedback_positive_count"] == 0
    assert payload["feedback_negative_count"] == 0
    assert payload["feedback_abs_adjustment_avg"] == 0.0
    assert payload["feedback_query_latency_ms"] is None
    assert payload["feedback_query_failed"] is False
    assert payload["feedback_query_error_category"] is None


def test_telemetry_feedback_enabled_true():
    payload = _payload(feedback_enabled=True)

    assert payload["feedback_enabled"] is True


def test_telemetry_feedback_counts():
    payload = _payload(
        feedback_adjusted_candidate_count=3,
        feedback_positive_count=2,
        feedback_negative_count=1,
        feedback_abs_adjustment_avg=1.23456,
    )

    assert payload["feedback_adjusted_candidate_count"] == 3
    assert payload["feedback_positive_count"] == 2
    assert payload["feedback_negative_count"] == 1
    assert payload["feedback_abs_adjustment_avg"] == 1.235


def test_telemetry_feedback_query_latency():
    payload = _payload(feedback_query_latency_ms=42)

    assert payload["feedback_query_latency_ms"] == 42


def test_telemetry_feedback_query_failed():
    payload = _payload(feedback_query_failed=True, feedback_query_error_category="db_error")

    assert payload["feedback_query_failed"] is True
    assert payload["feedback_query_error_category"] == "db_error"


def test_telemetry_no_forbidden_feedback_keys():
    payload = _payload(feedback_enabled=True, feedback_adjusted_candidate_count=1)
    forbidden = {
        "user_id",
        "user_note",
        "skipped_exercise_ids",
        "edited_exercises",
        "per_exercise_adjustment_map",
        "feedback_adjustment_map",
        "per_user_feedback",
    }

    assert forbidden.isdisjoint(payload)


def test_compute_feedback_stats_counts_nonzero_ranked_adjustments():
    ranked = [{"id": "A"}, {"id": "b"}, {"id": "c"}, {"id": "d"}]
    adjustments = {
        "a": FeedbackAdjustment(exercise_id="a", blended_adjustment=2.0),
        "b": FeedbackAdjustment(exercise_id="b", blended_adjustment=-4.0),
        "c": FeedbackAdjustment(exercise_id="c", blended_adjustment=0.0),
        "outside": FeedbackAdjustment(exercise_id="outside", blended_adjustment=8.0),
    }

    stats = _compute_feedback_stats(ranked, adjustments)

    assert stats == {"adjusted": 2, "positive": 1, "negative": 1, "abs_avg": 3.0}


def test_compute_feedback_stats_empty():
    assert _compute_feedback_stats([{"id": "a"}], None) == {
        "adjusted": 0,
        "positive": 0,
        "negative": 0,
        "abs_avg": 0.0,
    }


@pytest.mark.slow
@pytest.mark.integration
def test_staging_script_help_commands():
    script = "scripts/run_feedback_ranking_staging_test.py"
    for command in ["check-db", "seed", "verify-seed", "run", "compare"]:
        result = subprocess.run(
            [sys.executable, script, command, "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout

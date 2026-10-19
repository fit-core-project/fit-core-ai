"""Regression boundary for the feedback-duration completion review."""

import pytest

from engines.prescription.adjustments import (
    minimum_productive_duration_min,
    target_productive_duration_min,
)
from engines.prescription.underfill import resolve_underfill_decision


@pytest.mark.parametrize(
    ("case_id", "kwargs", "expected_reason"),
    [
        (
            "025",
            {
                "actual_duration_min": 12,
                "available_duration_min": 35,
                "candidate_count": 1,
                "minimum_exercise_count": 4,
            },
            "candidate_shortage",
        ),
        (
            "031",
            {
                "actual_duration_min": 28,
                "available_duration_min": 50,
                "condition_policy_limited": True,
            },
            "condition_policy_constraint",
        ),
        (
            "032",
            {
                "actual_duration_min": 28,
                "available_duration_min": 50,
                "condition_policy_limited": True,
            },
            "condition_policy_constraint",
        ),
        (
            "051-054",
            {
                "actual_duration_min": 35,
                "available_duration_min": 60,
                "candidate_count": 12,
                "minimum_exercise_count": 3,
                "selection_diagnostics": {"role_set_capacity_exhausted": True},
            },
            "role_set_capacity_exhausted",
        ),
        (
            "078",
            {
                "actual_duration_min": 24,
                "available_duration_min": 60,
                "readiness_level": "low",
            },
            "low_readiness",
        ),
        (
            "079",
            {
                "actual_duration_min": 22,
                "available_duration_min": 60,
                "readiness_level": "normal",
                "recovery_limited": True,
            },
            "recovery_constraint",
        ),
    ],
)
def test_duration_feedback_reason_categories_remain_separate(
    case_id,
    kwargs,
    expected_reason,
):
    decision = resolve_underfill_decision(
        has_routine_blocks=True,
        **kwargs,
    )

    assert decision.underfilled, case_id
    assert decision.reason_code == expected_reason


def test_long_availability_uses_the_existing_productive_time_reference():
    # Case 007's 90-minute availability is not a request to add unbounded work.
    assert minimum_productive_duration_min(90) == 49
    assert target_productive_duration_min(90) == 64

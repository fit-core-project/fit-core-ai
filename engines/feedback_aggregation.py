"""Exercise-level aggregation helpers for stored routine feedback."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

from models.routine_feedback import RoutineFeedback

FEEDBACK_AWARE_RANKING_ENV = "ENABLE_FEEDBACK_AWARE_RANKING"
DEFAULT_GLOBAL_FEEDBACK_WEIGHT = 0.3
MIN_FEEDBACK_ADJUSTMENT = -10.0
MAX_FEEDBACK_ADJUSTMENT = 10.0

_TRUE_VALUES = {"true", "1", "yes", "on"}
_FALSE_VALUES = {"false", "0", "no", "off", ""}
_KNOWN_REASONS = {
    "pain",
    "no_equipment",
    "dislike",
    "duplicate",
    "too_heavy",
    "too_easy",
    "other",
}


@dataclass
class ExerciseFeedbackStats:
    exercise_id: str
    user_id: Optional[str] = None
    feedback_count: int = 0
    skip_count: int = 0
    original_replaced_count: int = 0
    replacement_chosen_count: int = 0
    pain_count: int = 0
    no_equipment_count: int = 0
    dislike_count: int = 0
    duplicate_count: int = 0
    too_heavy_count: int = 0
    too_easy_count: int = 0
    other_reason_count: int = 0
    completed_count: int = 0
    accepted_without_edits_count: int = 0
    high_rating_count: int = 0
    low_rating_count: int = 0
    rating_count: int = 0
    rating_sum: int = 0
    avg_rating: Optional[float] = None
    last_feedback_at: Optional[datetime] = None


@dataclass
class FeedbackAdjustment:
    exercise_id: str
    user_adjustment: float = 0.0
    global_adjustment: float = 0.0
    blended_adjustment: float = 0.0
    confidence: float = 0.0
    feedback_count: int = 0


def resolve_feedback_aware_ranking_enabled(value: Optional[str] = None) -> bool:
    if value is None:
        return False
    normalized = str(value).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return False


def get_feedback_aware_ranking_enabled() -> bool:
    return resolve_feedback_aware_ranking_enabled(os.getenv(FEEDBACK_AWARE_RANKING_ENV))


def compute_confidence(feedback_count: int) -> float:
    return min(1.0, max(0, int(feedback_count)) / 10.0)


def clamp_feedback_adjustment(value: float) -> float:
    return max(MIN_FEEDBACK_ADJUSTMENT, min(MAX_FEEDBACK_ADJUSTMENT, float(value)))


def compute_feedback_adjustment(stats: Optional[ExerciseFeedbackStats]) -> float:
    if stats is None or stats.feedback_count <= 0:
        return 0.0

    positive_score = (
        0.4 * stats.replacement_chosen_count
        + 0.2 * stats.completed_count
        + 0.2 * stats.accepted_without_edits_count
        + 0.2 * stats.high_rating_count
    )
    negative_score = (
        0.5 * stats.skip_count
        + 0.5 * stats.original_replaced_count
        + 0.8 * stats.pain_count
        + 0.5 * stats.dislike_count
        + 0.3 * stats.no_equipment_count
        + 0.3 * stats.duplicate_count
        + 0.2 * stats.too_heavy_count
        + 0.2 * stats.too_easy_count
        + 0.2 * stats.low_rating_count
    )
    raw_adjustment = (positive_score - negative_score) * compute_confidence(stats.feedback_count)
    return clamp_feedback_adjustment(raw_adjustment)


def blend_user_and_global_feedback(
    user_adj: float,
    global_adj: float,
    global_weight: float = DEFAULT_GLOBAL_FEEDBACK_WEIGHT,
) -> float:
    return clamp_feedback_adjustment(float(user_adj) + float(global_weight) * float(global_adj))


def collect_feedback_stats(
    session,
    *,
    user_id: Optional[str] = None,
    exercise_ids: Optional[list[str]] = None,
    scope: str = "user",
) -> dict[str, ExerciseFeedbackStats]:
    if scope not in {"user", "global"}:
        raise ValueError("scope must be 'user' or 'global'")
    if scope == "user" and not user_id:
        return {}

    allowed_ids = _normalize_id_set(exercise_ids)
    query = session.query(RoutineFeedback)
    if scope == "user":
        query = query.filter(RoutineFeedback.user_id == user_id)

    stats_by_id: dict[str, ExerciseFeedbackStats] = {}
    for row in query.all():
        _aggregate_row(
            stats_by_id,
            row,
            allowed_ids=allowed_ids,
            user_id=user_id if scope == "user" else None,
        )
    return stats_by_id


def get_feedback_adjustments(
    session,
    *,
    user_id: Optional[str],
    exercise_ids: list[str],
    global_weight: float = DEFAULT_GLOBAL_FEEDBACK_WEIGHT,
) -> dict[str, FeedbackAdjustment]:
    normalized_ids = sorted(_normalize_id_set(exercise_ids))
    if not normalized_ids:
        return {}

    user_stats = collect_feedback_stats(
        session,
        user_id=user_id,
        exercise_ids=normalized_ids,
        scope="user",
    )
    global_stats = collect_feedback_stats(
        session,
        exercise_ids=normalized_ids,
        scope="global",
    )

    adjustments: dict[str, FeedbackAdjustment] = {}
    for exercise_id in normalized_ids:
        user_stat = user_stats.get(exercise_id)
        global_stat = global_stats.get(exercise_id)
        user_adj = compute_feedback_adjustment(user_stat)
        global_adj = compute_feedback_adjustment(global_stat)
        blended = blend_user_and_global_feedback(user_adj, global_adj, global_weight)
        # Global stats already include user rows, so avoid double-counting in metadata.
        feedback_count = max(
            user_stat.feedback_count if user_stat else 0,
            global_stat.feedback_count if global_stat else 0,
        )
        confidence = max(
            compute_confidence(user_stat.feedback_count if user_stat else 0),
            compute_confidence(global_stat.feedback_count if global_stat else 0),
        )
        adjustments[exercise_id] = FeedbackAdjustment(
            exercise_id=exercise_id,
            user_adjustment=user_adj,
            global_adjustment=global_adj,
            blended_adjustment=blended,
            confidence=confidence,
            feedback_count=feedback_count,
        )
    return adjustments


def _normalize_exercise_id(value) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_id_set(values: Optional[Iterable[str]]) -> set[str]:
    if values is None:
        return set()
    return {
        normalized
        for value in values
        if (normalized := _normalize_exercise_id(value)) is not None
    }


def _get_stats(
    stats_by_id: dict[str, ExerciseFeedbackStats],
    exercise_id: str,
    user_id: Optional[str],
) -> ExerciseFeedbackStats:
    if exercise_id not in stats_by_id:
        stats_by_id[exercise_id] = ExerciseFeedbackStats(exercise_id=exercise_id, user_id=user_id)
    return stats_by_id[exercise_id]


def _aggregate_row(
    stats_by_id: dict[str, ExerciseFeedbackStats],
    row: RoutineFeedback,
    *,
    allowed_ids: set[str],
    user_id: Optional[str],
) -> None:
    skipped_ids = _safe_list(row.skipped_exercise_ids)
    edited_items = _safe_list(row.edited_exercises)

    mentioned_ids = _normalize_id_set(skipped_ids)
    original_ids: set[str] = set()
    replacement_ids: set[str] = set()
    reason_by_original: dict[str, set[str]] = {}

    for item in edited_items:
        if not isinstance(item, dict):
            continue
        original_id = _normalize_exercise_id(item.get("original_exercise_id"))
        replacement_id = _normalize_exercise_id(item.get("replacement_exercise_id"))
        reason = _normalize_reason(item.get("reason"))
        if original_id:
            original_ids.add(original_id)
            mentioned_ids.add(original_id)
            reason_by_original.setdefault(original_id, set()).add(reason)
        if replacement_id:
            replacement_ids.add(replacement_id)
            mentioned_ids.add(replacement_id)

    if allowed_ids:
        mentioned_ids &= allowed_ids
        skipped_ids = [exercise_id for exercise_id in skipped_ids if _normalize_exercise_id(exercise_id) in allowed_ids]
        original_ids &= allowed_ids
        replacement_ids &= allowed_ids
        reason_by_original = {
            exercise_id: reasons
            for exercise_id, reasons in reason_by_original.items()
            if exercise_id in allowed_ids
        }

    if not mentioned_ids:
        return

    for exercise_id in sorted(mentioned_ids):
        stats = _get_stats(stats_by_id, exercise_id, user_id)
        stats.feedback_count += 1
        if row.created_at and (stats.last_feedback_at is None or row.created_at > stats.last_feedback_at):
            stats.last_feedback_at = row.created_at
        _apply_routine_level_signals(stats, row)

    for exercise_id in _normalize_id_set(skipped_ids):
        if exercise_id in stats_by_id:
            stats_by_id[exercise_id].skip_count += 1
    for exercise_id in original_ids:
        stats_by_id[exercise_id].original_replaced_count += 1
        for reason in reason_by_original.get(exercise_id, {"other"}):
            _increment_reason_count(stats_by_id[exercise_id], reason)
    for exercise_id in replacement_ids:
        if exercise_id in stats_by_id:
            stats_by_id[exercise_id].replacement_chosen_count += 1


def _safe_list(value) -> list:
    return value if isinstance(value, list) else []


def _normalize_reason(value) -> str:
    reason = str(value or "other").strip()
    return reason if reason in _KNOWN_REASONS else "other"


def _increment_reason_count(stats: ExerciseFeedbackStats, reason: str) -> None:
    if reason == "pain":
        stats.pain_count += 1
    elif reason == "no_equipment":
        stats.no_equipment_count += 1
    elif reason == "dislike":
        stats.dislike_count += 1
    elif reason == "duplicate":
        stats.duplicate_count += 1
    elif reason == "too_heavy":
        stats.too_heavy_count += 1
    elif reason == "too_easy":
        stats.too_easy_count += 1
    else:
        stats.other_reason_count += 1


def _apply_routine_level_signals(stats: ExerciseFeedbackStats, row: RoutineFeedback) -> None:
    if row.completed is True:
        stats.completed_count += 1
    if row.accepted_without_edits is True:
        stats.accepted_without_edits_count += 1
    if row.rating is not None:
        rating = int(row.rating)
        stats.rating_count += 1
        stats.rating_sum += rating
        stats.avg_rating = stats.rating_sum / stats.rating_count
        if rating >= 4:
            stats.high_rating_count += 1
        elif rating <= 2:
            stats.low_rating_count += 1

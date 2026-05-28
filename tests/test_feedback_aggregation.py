from __future__ import annotations

from datetime import datetime

from models.routine_feedback import RoutineFeedback
from engines.feedback_aggregation import (
    ExerciseFeedbackStats,
    blend_user_and_global_feedback,
    clamp_feedback_adjustment,
    collect_feedback_stats,
    compute_confidence,
    compute_feedback_adjustment,
    get_feedback_adjustments,
    get_feedback_aware_ranking_enabled,
    resolve_feedback_aware_ranking_enabled,
)


def add_feedback(test_db, **kwargs) -> RoutineFeedback:
    row = RoutineFeedback(
        routine_draft_id=kwargs.pop("routine_draft_id", "draft-1"),
        user_id=kwargs.pop("user_id", "user-1"),
        rating=kwargs.pop("rating", None),
        completed=kwargs.pop("completed", None),
        accepted_without_edits=kwargs.pop("accepted_without_edits", None),
        skipped_exercise_ids=kwargs.pop("skipped_exercise_ids", None),
        edited_exercises=kwargs.pop("edited_exercises", None),
        user_note=kwargs.pop("user_note", None),
        created_at=kwargs.pop("created_at", datetime(2026, 1, 1, 12, 0, 0)),
        source=kwargs.pop("source", "api"),
    )
    test_db.add(row)
    test_db.commit()
    return row


def edit(original: str | None = None, replacement: str | None = None, reason: str = "other") -> dict:
    return {
        "original_exercise_id": original,
        "replacement_exercise_id": replacement,
        "reason": reason,
    }


def test_no_feedback_adjustment_zero():
    assert compute_feedback_adjustment(None) == 0.0
    assert compute_feedback_adjustment(ExerciseFeedbackStats(exercise_id="pushup")) == 0.0


def test_skipped_exercise_negative_adjustment(test_db):
    add_feedback(test_db, skipped_exercise_ids=["pushup"])

    stats = collect_feedback_stats(test_db, user_id="user-1", exercise_ids=["pushup"])["pushup"]

    assert stats.skip_count == 1
    assert compute_feedback_adjustment(stats) < 0


def test_original_replaced_exercise_negative_adjustment(test_db):
    add_feedback(test_db, edited_exercises=[edit(original="pushup", replacement="dip")])

    stats = collect_feedback_stats(test_db, user_id="user-1", exercise_ids=["pushup"])["pushup"]

    assert stats.original_replaced_count == 1
    assert compute_feedback_adjustment(stats) < 0


def test_replacement_chosen_exercise_positive_adjustment(test_db):
    add_feedback(test_db, edited_exercises=[edit(original="pushup", replacement="dip")])

    stats = collect_feedback_stats(test_db, user_id="user-1", exercise_ids=["dip"])["dip"]

    assert stats.replacement_chosen_count == 1
    assert compute_feedback_adjustment(stats) > 0


def test_reason_pain_stronger_negative_than_generic_original():
    generic = ExerciseFeedbackStats(exercise_id="a", feedback_count=10, original_replaced_count=10)
    pain = ExerciseFeedbackStats(
        exercise_id="a",
        feedback_count=10,
        original_replaced_count=10,
        pain_count=10,
    )

    assert compute_feedback_adjustment(pain) < compute_feedback_adjustment(generic)


def test_reason_no_equipment_moderate_negative(test_db):
    add_feedback(test_db, edited_exercises=[edit(original="barbell_press", replacement="pushup", reason="no_equipment")])

    stats = collect_feedback_stats(test_db, user_id="user-1", exercise_ids=["barbell_press"])["barbell_press"]

    assert stats.no_equipment_count == 1
    assert compute_feedback_adjustment(stats) < 0


def test_reason_dislike_negative(test_db):
    add_feedback(test_db, edited_exercises=[edit(original="pushup", replacement="dip", reason="dislike")])

    stats = collect_feedback_stats(test_db, user_id="user-1", exercise_ids=["pushup"])["pushup"]

    assert stats.dislike_count == 1
    assert compute_feedback_adjustment(stats) < 0


def test_reason_duplicate_negative(test_db):
    add_feedback(test_db, edited_exercises=[edit(original="pushup", replacement="dip", reason="duplicate")])

    stats = collect_feedback_stats(test_db, user_id="user-1", exercise_ids=["pushup"])["pushup"]

    assert stats.duplicate_count == 1
    assert compute_feedback_adjustment(stats) < 0


def test_too_heavy_and_too_easy_are_weak_signals():
    heavy = ExerciseFeedbackStats(
        exercise_id="a",
        feedback_count=10,
        original_replaced_count=10,
        too_heavy_count=10,
    )
    pain = ExerciseFeedbackStats(
        exercise_id="a",
        feedback_count=10,
        original_replaced_count=10,
        pain_count=10,
    )

    assert compute_feedback_adjustment(heavy) < 0
    assert compute_feedback_adjustment(heavy) > compute_feedback_adjustment(pain)


def test_high_rating_positive_count_on_directly_mentioned_exercise(test_db):
    add_feedback(test_db, rating=5, edited_exercises=[edit(original="pushup", replacement="dip")])

    stats = collect_feedback_stats(test_db, user_id="user-1", exercise_ids=["dip"])["dip"]

    assert stats.high_rating_count == 1
    assert stats.avg_rating == 5.0
    assert compute_feedback_adjustment(stats) > 0


def test_low_rating_negative_count_on_directly_mentioned_exercise(test_db):
    add_feedback(test_db, rating=1, skipped_exercise_ids=["pushup"])

    stats = collect_feedback_stats(test_db, user_id="user-1", exercise_ids=["pushup"])["pushup"]

    assert stats.low_rating_count == 1
    assert stats.avg_rating == 1.0
    assert compute_feedback_adjustment(stats) < 0


def test_routine_level_only_feedback_is_not_assigned_to_exercise(test_db):
    add_feedback(test_db, rating=5, completed=True, accepted_without_edits=True)

    assert collect_feedback_stats(test_db, user_id="user-1", exercise_ids=["pushup"]) == {}


def test_completed_true_weak_positive_when_exercise_directly_mentioned(test_db):
    add_feedback(test_db, completed=True, edited_exercises=[edit(original="pushup", replacement="dip")])

    stats = collect_feedback_stats(test_db, user_id="user-1", exercise_ids=["dip"])["dip"]

    assert stats.completed_count == 1
    assert compute_feedback_adjustment(stats) > 0


def test_accepted_without_edits_weak_positive_when_exercise_directly_mentioned(test_db):
    add_feedback(test_db, accepted_without_edits=True, edited_exercises=[edit(original="pushup", replacement="dip")])

    stats = collect_feedback_stats(test_db, user_id="user-1", exercise_ids=["dip"])["dip"]

    assert stats.accepted_without_edits_count == 1
    assert compute_feedback_adjustment(stats) > 0


def test_confidence_with_one_feedback_less_than_ten_feedback():
    assert compute_confidence(1) < compute_confidence(10)
    assert compute_confidence(10) == 1.0


def test_clamp_upper_bound():
    stats = ExerciseFeedbackStats(exercise_id="a", feedback_count=100, replacement_chosen_count=100)

    assert compute_feedback_adjustment(stats) == 10.0
    assert clamp_feedback_adjustment(99) == 10.0


def test_clamp_lower_bound():
    stats = ExerciseFeedbackStats(exercise_id="a", feedback_count=100, skip_count=100, pain_count=100)

    assert compute_feedback_adjustment(stats) == -10.0
    assert clamp_feedback_adjustment(-99) == -10.0


def test_user_level_stats_only_include_that_user(test_db):
    add_feedback(test_db, user_id="user-1", skipped_exercise_ids=["pushup"])
    add_feedback(test_db, user_id="user-2", skipped_exercise_ids=["pushup"])

    stats = collect_feedback_stats(test_db, user_id="user-1", exercise_ids=["pushup"])["pushup"]

    assert stats.user_id == "user-1"
    assert stats.feedback_count == 1
    assert stats.skip_count == 1


def test_global_stats_include_all_users(test_db):
    add_feedback(test_db, user_id="user-1", skipped_exercise_ids=["pushup"])
    add_feedback(test_db, user_id="user-2", skipped_exercise_ids=["pushup"])

    stats = collect_feedback_stats(test_db, scope="global", exercise_ids=["pushup"])["pushup"]

    assert stats.user_id is None
    assert stats.feedback_count == 2
    assert stats.skip_count == 2


def test_user_id_none_scope_user_returns_empty(test_db):
    add_feedback(test_db, user_id=None, skipped_exercise_ids=["pushup"])

    assert collect_feedback_stats(test_db, user_id=None, scope="user", exercise_ids=["pushup"]) == {}


def test_exercise_id_str_type_preserved(test_db):
    add_feedback(test_db, skipped_exercise_ids=[123])

    stats = collect_feedback_stats(test_db, user_id="user-1")

    assert "123" in stats
    assert stats["123"].exercise_id == "123"


def test_user_note_not_used_in_stats_or_adjustment(test_db):
    add_feedback(test_db, skipped_exercise_ids=["pushup"], user_note="private free text")

    stats = collect_feedback_stats(test_db, user_id="user-1", exercise_ids=["pushup"])["pushup"]
    adjustment = get_feedback_adjustments(test_db, user_id="user-1", exercise_ids=["pushup"])["pushup"]

    assert "private free text" not in repr(stats)
    assert "private free text" not in repr(adjustment)


def test_blend_user_and_global_feedback_applies_weight():
    assert blend_user_and_global_feedback(2.0, 3.0, global_weight=0.3) == 2.9


def test_get_feedback_adjustments_includes_zero_for_no_feedback(test_db):
    adjustments = get_feedback_adjustments(test_db, user_id="user-1", exercise_ids=["pushup"])

    assert adjustments["pushup"].blended_adjustment == 0.0
    assert adjustments["pushup"].feedback_count == 0


def test_get_feedback_adjustments_user_dominates_global(test_db):
    add_feedback(test_db, user_id="user-1", edited_exercises=[edit(original="pushup", replacement="dip")])
    add_feedback(test_db, user_id="user-2", skipped_exercise_ids=["dip"])

    adjustment = get_feedback_adjustments(test_db, user_id="user-1", exercise_ids=["dip"])["dip"]

    assert adjustment.user_adjustment > 0
    assert adjustment.global_adjustment < 0
    assert adjustment.blended_adjustment > 0


def test_collect_feedback_stats_filters_exercise_ids(test_db):
    add_feedback(test_db, skipped_exercise_ids=["pushup", "dip"])

    stats = collect_feedback_stats(test_db, user_id="user-1", exercise_ids=["pushup"])

    assert list(stats) == ["pushup"]


def test_duplicate_exercise_ids_in_one_row_count_once(test_db):
    add_feedback(test_db, skipped_exercise_ids=["pushup", "pushup"])

    stats = collect_feedback_stats(test_db, user_id="user-1", exercise_ids=["pushup"])["pushup"]

    assert stats.feedback_count == 1
    assert stats.skip_count == 1


def test_malformed_json_fields_are_skipped_safely(test_db):
    add_feedback(test_db, skipped_exercise_ids="pushup", edited_exercises="bad")

    assert collect_feedback_stats(test_db, user_id="user-1", exercise_ids=["pushup"]) == {}


def test_unknown_reason_counts_as_other(test_db):
    add_feedback(test_db, edited_exercises=[edit(original="pushup", replacement="dip", reason="unknown")])

    stats = collect_feedback_stats(test_db, user_id="user-1", exercise_ids=["pushup"])["pushup"]

    assert stats.other_reason_count == 1


def test_feature_flag_default_false(monkeypatch):
    monkeypatch.delenv("ENABLE_FEEDBACK_AWARE_RANKING", raising=False)

    assert get_feedback_aware_ranking_enabled() is False
    assert resolve_feedback_aware_ranking_enabled(None) is False


def test_feature_flag_true_values():
    for value in ["true", "1", "yes", "on", " TRUE "]:
        assert resolve_feedback_aware_ranking_enabled(value) is True


def test_feature_flag_false_and_invalid_values():
    for value in ["false", "0", "no", "off", "", "maybe"]:
        assert resolve_feedback_aware_ranking_enabled(value) is False

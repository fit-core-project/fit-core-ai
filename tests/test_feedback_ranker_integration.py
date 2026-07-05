from __future__ import annotations

import logging

import pytest

from engines.candidate_ranker import score_candidate_exercises
from engines.feedback_aggregation import FeedbackAdjustment
from engines.routine_pipeline import _get_feedback_adjustments_if_enabled
from engines.schemas import PainAreaEntry, RecentSetRecord


def candidate(
    exercise_id: str,
    *,
    name_en: str | None = None,
    equipment: str = "DUMBBELL",
    pain_triggers: str | None = None,
    primary: str = "chest",
    movement_type: str = "COMPOUND",
    efficiency: int = 5,
) -> dict:
    return {
        "id": exercise_id,
        "name_kr": exercise_id,
        "name_en": name_en or exercise_id,
        "primary_muscle": primary,
        "secondary_muscle": "",
        "equipment_req": equipment,
        "pain_triggers": pain_triggers,
        "movement_type": movement_type,
        "efficiency_tier": efficiency,
    }


def adj(exercise_id: str, value: float) -> FeedbackAdjustment:
    return FeedbackAdjustment(exercise_id=exercise_id, blended_adjustment=value)


def ids(ranked: list[dict]) -> list[str]:
    return [str(item["id"]) for item in ranked]


def test_feedback_ranking_flag_off_no_db_query(monkeypatch):
    import engines.feedback_aggregation as feedback_aggregation

    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("feedback query should not run")

    monkeypatch.setattr(feedback_aggregation, "get_feedback_aware_ranking_enabled", lambda: False)
    monkeypatch.setattr(feedback_aggregation, "get_feedback_adjustments", fail_if_called)

    result = _get_feedback_adjustments_if_enabled(object(), "user-1", [candidate("PushUp")])

    assert result is None
    assert called is False


def test_feedback_ranking_flag_on_calls_db_with_normalized_ids(monkeypatch):
    import engines.feedback_aggregation as feedback_aggregation

    captured = {}

    def fake_get_feedback_adjustments(db, *, user_id, exercise_ids):
        captured["db"] = db
        captured["user_id"] = user_id
        captured["exercise_ids"] = exercise_ids
        return {"pushup": adj("pushup", 1)}

    db = object()
    monkeypatch.setattr(feedback_aggregation, "get_feedback_aware_ranking_enabled", lambda: True)
    monkeypatch.setattr(feedback_aggregation, "get_feedback_adjustments", fake_get_feedback_adjustments)

    result = _get_feedback_adjustments_if_enabled(db, "user-1", [candidate(" PushUp "), candidate("")])

    assert result["pushup"].blended_adjustment == 1
    assert captured == {
        "db": db,
        "user_id": "user-1",
        "exercise_ids": ["pushup"],
    }


def test_feedback_ranking_db_error_returns_none(monkeypatch, caplog):
    import engines.feedback_aggregation as feedback_aggregation

    caplog.set_level(logging.WARNING, logger="engines.routine_pipeline")
    monkeypatch.setattr(feedback_aggregation, "get_feedback_aware_ranking_enabled", lambda: True)
    monkeypatch.setattr(
        feedback_aggregation,
        "get_feedback_adjustments",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("secret db error")),
    )

    result = _get_feedback_adjustments_if_enabled(object(), "user-1", [candidate("pushup")])

    captured = "\n".join(record.getMessage() for record in caplog.records)
    assert result is None
    assert "RuntimeError" in captured
    assert "secret db error" not in captured


def test_score_merge_no_feedback_matches_empty_feedback():
    candidates = [candidate("a"), candidate("b")]

    baseline = score_candidate_exercises(candidates, ["chest"], top_n=2)
    none_feedback = score_candidate_exercises(candidates, ["chest"], top_n=2, feedback_adjustments=None)
    empty_feedback = score_candidate_exercises(candidates, ["chest"], top_n=2, feedback_adjustments={})

    assert none_feedback == baseline
    assert empty_feedback == baseline
    assert all("feedback adj" not in " ".join(item["score_reasons"]) for item in none_feedback)


def test_score_merge_positive_adjustment():
    baseline = score_candidate_exercises([candidate("a")], ["chest"], top_n=1)[0]

    ranked = score_candidate_exercises(
        [candidate("a")],
        ["chest"],
        top_n=1,
        feedback_adjustments={"a": adj("a", 5)},
    )[0]

    assert ranked["score"] == baseline["score"] + 5
    assert "feedback adj +5.0" in ranked["score_reasons"]


def test_score_merge_negative_adjustment():
    baseline = score_candidate_exercises([candidate("a")], ["chest"], top_n=1)[0]

    ranked = score_candidate_exercises(
        [candidate("a")],
        ["chest"],
        top_n=1,
        feedback_adjustments={"a": adj("a", -8)},
    )[0]

    assert ranked["score"] == baseline["score"] - 8
    assert "feedback adj -8.0" in ranked["score_reasons"]


def test_score_merge_clamp_upper():
    baseline = score_candidate_exercises([candidate("a")], ["chest"], top_n=1)[0]

    ranked = score_candidate_exercises(
        [candidate("a")],
        ["chest"],
        top_n=1,
        feedback_adjustments={"a": adj("a", 15)},
    )[0]

    assert ranked["score"] == baseline["score"] + 10
    assert "feedback adj +10.0" in ranked["score_reasons"]


def test_score_merge_clamp_lower():
    baseline = score_candidate_exercises([candidate("a")], ["chest"], top_n=1)[0]

    ranked = score_candidate_exercises(
        [candidate("a")],
        ["chest"],
        top_n=1,
        feedback_adjustments={"a": adj("a", -15)},
    )[0]

    assert ranked["score"] == baseline["score"] - 10
    assert "feedback adj -10.0" in ranked["score_reasons"]


def test_feedback_cannot_restore_blocked_equipment():
    ranked = score_candidate_exercises(
        [candidate("barbell_press", equipment="BARBELL"), candidate("pushup", equipment="BODYWEIGHT")],
        ["chest"],
        blocked_equipment=["BARBELL"],
        feedback_adjustments={"barbell_press": adj("barbell_press", 10)},
        top_n=5,
    )

    assert "barbell_press" not in ids(ranked)


def test_feedback_cannot_restore_pain_trigger():
    ranked = score_candidate_exercises(
        [candidate("shoulder_press", pain_triggers="shoulder"), candidate("pushup")],
        ["chest"],
        pain_areas=[PainAreaEntry(body_part="shoulder")],
        feedback_adjustments={"shoulder_press": adj("shoulder_press", 10)},
        top_n=5,
    )

    assert "shoulder_press" not in ids(ranked)


def test_feedback_cannot_restore_doms3():
    ranked = score_candidate_exercises(
        [candidate("chest_press", primary="chest"), candidate("row", primary="upper-back")],
        ["chest", "upper-back"],
        doms_db={"chest": 3},
        feedback_adjustments={"chest_press": adj("chest_press", 10)},
        top_n=5,
    )

    assert "chest_press" not in ids(ranked)


def test_feedback_promotes_candidate():
    candidates = [candidate("a"), candidate("b", name_en="b")]
    baseline = score_candidate_exercises(
        candidates,
        ["chest"],
        recent_sets=[RecentSetRecord(exercise_name="b", reps=8)],
        top_n=2,
    )

    ranked = score_candidate_exercises(
        candidates,
        ["chest"],
        recent_sets=[RecentSetRecord(exercise_name="b", reps=8)],
        feedback_adjustments={"b": adj("b", 10)},
        top_n=2,
    )

    assert ids(baseline) == ["a", "b"]
    assert ids(ranked) == ["b", "a"]


def test_feedback_demotes_candidate():
    candidates = [candidate("a"), candidate("b", name_en="b")]
    baseline = score_candidate_exercises(
        candidates,
        ["chest"],
        recent_sets=[RecentSetRecord(exercise_name="b", reps=8)],
        top_n=2,
    )

    ranked = score_candidate_exercises(
        candidates,
        ["chest"],
        recent_sets=[RecentSetRecord(exercise_name="b", reps=8)],
        feedback_adjustments={"a": adj("a", -10)},
        top_n=2,
    )

    assert ids(baseline) == ["a", "b"]
    assert ids(ranked) == ["b", "a"]


def test_feedback_zero_adjustment_no_reorder():
    candidates = [candidate("a"), candidate("b")]
    baseline = score_candidate_exercises(candidates, ["chest"], top_n=2)

    ranked = score_candidate_exercises(
        candidates,
        ["chest"],
        feedback_adjustments={"a": adj("a", 0.0), "b": 0.0},
        top_n=2,
    )

    assert ranked == baseline
    assert all("feedback adj" not in " ".join(item["score_reasons"]) for item in ranked)


def test_flag_off_candidate_order_same_as_baseline(monkeypatch):
    monkeypatch.delenv("ENABLE_FEEDBACK_AWARE_RANKING", raising=False)
    candidates = [candidate("a"), candidate("b", name_en="b")]
    baseline = score_candidate_exercises(candidates, ["chest"], top_n=2)

    feedback_adjustments = _get_feedback_adjustments_if_enabled(object(), "user-1", candidates)
    ranked = score_candidate_exercises(
        candidates,
        ["chest"],
        feedback_adjustments=feedback_adjustments,
        top_n=2,
    )

    assert ranked == baseline


def test_pipeline_flag_off_does_not_query_feedback(monkeypatch):
    import engines.feedback_aggregation as feedback_aggregation

    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("feedback query should not run")

    monkeypatch.setattr(feedback_aggregation, "get_feedback_aware_ranking_enabled", lambda: False)
    monkeypatch.setattr(feedback_aggregation, "get_feedback_adjustments", fail_if_called)

    assert _get_feedback_adjustments_if_enabled(object(), None, [candidate("a")]) is None
    assert called is False


def test_plain_numeric_feedback_adjustment_supported():
    baseline = score_candidate_exercises([candidate("a")], ["chest"], top_n=1)[0]

    ranked = score_candidate_exercises(
        [candidate("a")],
        ["chest"],
        top_n=1,
        feedback_adjustments={"a": 4.4},
    )[0]

    assert ranked["score"] == baseline["score"] + 4
    assert "feedback adj +4.4" in ranked["score_reasons"]

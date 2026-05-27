from __future__ import annotations

from datetime import datetime
from uuid import UUID

from models.routine_feedback import RoutineFeedback


def post_feedback(client, payload: dict):
    return client.post("/api/ai/routine-feedback", json=payload)


def latest_feedback(test_db) -> RoutineFeedback:
    return test_db.query(RoutineFeedback).order_by(RoutineFeedback.created_at.desc()).first()


def test_valid_full_feedback_stored(feedback_client, test_db):
    payload = {
        "routine_draft_id": "draft-001",
        "user_id": "user-123",
        "rating": 4,
        "completed": True,
        "accepted_without_edits": False,
        "skipped_exercises": ["barbell_bench_press"],
        "edited_exercises": [{
            "original_exercise_id": "dumbbell_fly",
            "replacement_exercise_id": "cable_fly",
            "reason": "no_equipment",
        }],
        "user_note": "  bench was too heavy  ",
    }

    response = post_feedback(feedback_client, payload)

    assert response.status_code == 201
    row = latest_feedback(test_db)
    assert row.routine_draft_id == "draft-001"
    assert row.user_id == "user-123"
    assert row.rating == 4
    assert row.completed is True
    assert row.accepted_without_edits is False
    assert row.skipped_exercise_ids == ["barbell_bench_press"]
    assert row.edited_exercises[0]["original_exercise_id"] == "dumbbell_fly"
    assert row.user_note == "bench was too heavy"


def test_rating_only_stored(feedback_client, test_db):
    response = post_feedback(feedback_client, {"routine_draft_id": "draft-rating", "rating": 5})
    assert response.status_code == 201
    assert latest_feedback(test_db).rating == 5


def test_completed_only_stored(feedback_client, test_db):
    response = post_feedback(feedback_client, {"routine_draft_id": "draft-completed", "completed": False})
    assert response.status_code == 201
    assert latest_feedback(test_db).completed is False


def test_accepted_without_edits_only_stored(feedback_client, test_db):
    response = post_feedback(
        feedback_client,
        {"routine_draft_id": "draft-accepted", "accepted_without_edits": True},
    )
    assert response.status_code == 201
    assert latest_feedback(test_db).accepted_without_edits is True


def test_skipped_exercises_stored_as_string_ids(feedback_client, test_db):
    response = post_feedback(
        feedback_client,
        {"routine_draft_id": "draft-skipped", "skipped_exercises": ["pushup", "plank"]},
    )
    assert response.status_code == 201
    assert latest_feedback(test_db).skipped_exercise_ids == ["pushup", "plank"]


def test_edited_exercises_stored_as_string_ids(feedback_client, test_db):
    response = post_feedback(
        feedback_client,
        {
            "routine_draft_id": "draft-edited",
            "edited_exercises": [{
                "original_exercise_id": "pushup",
                "replacement_exercise_id": "incline_pushup",
                "reason": "too_heavy",
            }],
        },
    )
    assert response.status_code == 201
    stored = latest_feedback(test_db).edited_exercises
    assert stored == [{
        "original_exercise_id": "pushup",
        "replacement_exercise_id": "incline_pushup",
        "reason": "too_heavy",
    }]


def test_user_note_strip_stored(feedback_client, test_db):
    response = post_feedback(
        feedback_client,
        {"routine_draft_id": "draft-note", "user_note": "  useful note  "},
    )
    assert response.status_code == 201
    assert latest_feedback(test_db).user_note == "useful note"


def test_user_note_over_500_chars_after_strip_rejected(feedback_client):
    response = post_feedback(
        feedback_client,
        {"routine_draft_id": "draft-note-long", "user_note": " " + ("x" * 501) + " "},
    )
    assert response.status_code == 422


def test_user_note_500_chars_allowed(feedback_client, test_db):
    note = "x" * 500
    response = post_feedback(feedback_client, {"routine_draft_id": "draft-note-500", "user_note": note})
    assert response.status_code == 201
    assert latest_feedback(test_db).user_note == note


def test_invalid_rating_zero_rejected(feedback_client):
    response = post_feedback(feedback_client, {"routine_draft_id": "draft-rating-zero", "rating": 0})
    assert response.status_code == 422


def test_invalid_rating_six_rejected(feedback_client):
    response = post_feedback(feedback_client, {"routine_draft_id": "draft-rating-six", "rating": 6})
    assert response.status_code == 422


def test_missing_routine_draft_id_rejected(feedback_client):
    response = post_feedback(feedback_client, {"rating": 4})
    assert response.status_code == 422


def test_empty_routine_draft_id_rejected(feedback_client):
    response = post_feedback(feedback_client, {"routine_draft_id": "   ", "rating": 4})
    assert response.status_code == 422


def test_empty_feedback_rejected(feedback_client):
    response = post_feedback(feedback_client, {"routine_draft_id": "draft-empty"})
    assert response.status_code == 422


def test_skipped_exercises_type_validation(feedback_client):
    response = post_feedback(
        feedback_client,
        {"routine_draft_id": "draft-bad-skipped", "skipped_exercises": "pushup"},
    )
    assert response.status_code == 422


def test_skipped_exercises_item_type_validation(feedback_client):
    response = post_feedback(
        feedback_client,
        {"routine_draft_id": "draft-bad-skipped-item", "skipped_exercises": [123]},
    )
    assert response.status_code == 422


def test_edited_exercises_type_validation(feedback_client):
    response = post_feedback(
        feedback_client,
        {"routine_draft_id": "draft-bad-edited", "edited_exercises": "pushup"},
    )
    assert response.status_code == 422


def test_edited_exercises_item_type_validation(feedback_client):
    response = post_feedback(
        feedback_client,
        {"routine_draft_id": "draft-bad-edited-item", "edited_exercises": ["pushup"]},
    )
    assert response.status_code == 422


def test_reason_enum_validation(feedback_client):
    response = post_feedback(
        feedback_client,
        {
            "routine_draft_id": "draft-bad-reason",
            "edited_exercises": [{
                "original_exercise_id": "pushup",
                "replacement_exercise_id": None,
                "reason": "unknown_value",
            }],
        },
    )
    assert response.status_code == 422


def test_duplicate_routine_draft_id_submission_allowed(feedback_client):
    payload = {"routine_draft_id": "draft-duplicate", "rating": 4}
    assert post_feedback(feedback_client, payload).status_code == 201
    assert post_feedback(feedback_client, payload).status_code == 201


def test_duplicate_submission_stores_two_rows(feedback_client, test_db):
    payload = {"routine_draft_id": "draft-duplicate-rows", "rating": 3}
    post_feedback(feedback_client, payload)
    post_feedback(feedback_client, payload)
    count = test_db.query(RoutineFeedback).filter_by(routine_draft_id="draft-duplicate-rows").count()
    assert count == 2


def test_anonymous_feedback_allowed(feedback_client, test_db):
    response = post_feedback(feedback_client, {"routine_draft_id": "draft-anon", "rating": 4})
    assert response.status_code == 201
    assert latest_feedback(test_db).user_id is None


def test_user_note_not_printed_to_stdout_on_success(feedback_client, capsys):
    secret_note = "private free text should not appear"
    response = post_feedback(
        feedback_client,
        {"routine_draft_id": "draft-private", "user_note": secret_note},
    )
    captured = capsys.readouterr()
    assert response.status_code == 201
    assert secret_note not in captured.out
    assert secret_note not in captured.err


def test_user_note_not_printed_to_stdout_on_validation_error(feedback_client, capsys):
    secret_note = "private invalid free text " + ("x" * 501)
    response = post_feedback(
        feedback_client,
        {"routine_draft_id": "draft-private-invalid", "user_note": secret_note},
    )
    captured = capsys.readouterr()
    assert response.status_code == 422
    assert secret_note not in captured.out
    assert secret_note not in captured.err


def test_feedback_id_is_uuid(feedback_client):
    response = post_feedback(feedback_client, {"routine_draft_id": "draft-uuid", "rating": 4})
    assert response.status_code == 201
    UUID(response.json()["feedback_id"])


def test_stored_at_is_iso8601(feedback_client):
    response = post_feedback(feedback_client, {"routine_draft_id": "draft-time", "rating": 4})
    assert response.status_code == 201
    datetime.fromisoformat(response.json()["stored_at"])


def test_source_field_defaults_to_api(feedback_client, test_db):
    response = post_feedback(feedback_client, {"routine_draft_id": "draft-source", "rating": 4})
    assert response.status_code == 201
    assert latest_feedback(test_db).source == "api"


def test_unknown_routine_draft_id_is_stored_without_lookup(feedback_client, test_db):
    response = post_feedback(feedback_client, {"routine_draft_id": "unknown-draft-id", "rating": 4})
    assert response.status_code == 201
    assert latest_feedback(test_db).routine_draft_id == "unknown-draft-id"

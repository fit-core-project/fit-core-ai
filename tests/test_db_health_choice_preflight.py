import sqlite3

import pytest

from engines import db_queries, routine_generation_choice
from engines.routine_generation_choice import (
    RoutineGenerationChoiceError,
    generate_routine_with_choice,
    make_routine_generation_selection,
    preflight_routine_generation,
)
from engines.schemas import (
    RoutineBlock,
    RoutineDraftResponse,
    RoutineRequest,
    SetPrescription,
)


def _request() -> RoutineRequest:
    return RoutineRequest.model_validate({
        "userId": "db-health-user",
        "targetSplitLabel": "push",
        "timeAvailableMin": 45,
    })


def _health_db(path):
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE user_condition_profile (
                condition_code TEXT,
                body_part TEXT,
                side TEXT,
                phase TEXT,
                professional_clearance INTEGER,
                user_id TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE user_mobility_profile (
                user_id TEXT,
                ankle_dorsiflexion_level TEXT,
                hip_flexion_level TEXT,
                shoulder_flexion_level TEXT,
                wrist_extension_level TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO user_mobility_profile VALUES
            ('db-health-user', 'normal', 'normal', 'limited', 'normal')
            """
        )
        connection.execute(
            """
            CREATE TABLE user_anthropometry_profile (
                user_id TEXT,
                femur_length_level TEXT,
                arm_length_level TEXT,
                torso_length_level TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO user_condition_profile VALUES
            ('shoulder_health', 'shoulder', 'left', 'managed', 0, 'db-health-user')
            """
        )


def _routine() -> RoutineDraftResponse:
    return RoutineDraftResponse(
        routine_draft_id="db-health-choice",
        generation_status="success",
        status_reason_code="none",
        is_fallback=False,
        total_estimated_time=30,
        summary_title="Routine",
        rationale_summary=[],
        warnings=[],
        routine_blocks=[
            RoutineBlock(
                order=1,
                exercise_id="real-catalog-boundary-id",
                exercise_name="Press",
                movement_pattern="horizontalPush",
                movement_type="COMPOUND",
                primary_muscles=["chest"],
                equipment_type="DUMBBELL",
                default_rest_sec=90,
                prescription=[
                    SetPrescription(
                        set_index=index,
                        set_type="working",
                        target_reps=8,
                        target_weight_kg=40,
                        target_rir=2,
                        target_rest_sec=90,
                    )
                    for index in range(1, 5)
                ],
                exercise_rationale="Boundary routine",
            )
        ],
    )


def test_db_only_health_policy_blocks_existing_generation_wrapper(tmp_path, monkeypatch):
    database_path = tmp_path / "health.sqlite"
    _health_db(database_path)
    monkeypatch.setattr(db_queries, "SQLITE_DB_PATH", database_path)
    request = _request()
    calls = []

    result = generate_routine_with_choice(
        request,
        object(),
        _generator=lambda *args, **kwargs: calls.append((args, kwargs)) or _routine(),
    )

    assert result.status == "choiceRequired"
    assert calls == []


def test_db_health_policy_is_bound_to_preflight_and_generation_choice(tmp_path, monkeypatch):
    database_path = tmp_path / "health.sqlite"
    _health_db(database_path)
    monkeypatch.setattr(db_queries, "SQLITE_DB_PATH", database_path)
    request = _request()
    explicit_empty_request = request.model_copy(update={"condition_policies": []})

    request_before = request.model_dump(mode="json")
    preflight = preflight_routine_generation(
        request, resolve_persisted_context=True
    )
    empty_preflight = preflight_routine_generation(
        explicit_empty_request, resolve_persisted_context=True
    )

    assert preflight.status == "choiceRequired"
    assert preflight.constraints.condition_policy_count == 1
    assert empty_preflight.constraints.condition_policies == preflight.constraints.condition_policies
    assert request.model_dump(mode="json") == request_before
    normal = make_routine_generation_selection(preflight, "normalVolume")
    reduced = make_routine_generation_selection(preflight, "reducedVolume")
    received_policies = []

    def generator(resolved_request, _db, *, profile, recent_sets):
        received_policies.append(resolved_request.condition_policies)
        return _routine()

    normal_result = generate_routine_with_choice(
        request, object(), selection=normal, _generator=generator
    )
    reduced_result = generate_routine_with_choice(
        request, object(), selection=reduced, _generator=generator
    )

    assert normal_result.selected_choice == "normalVolume"
    assert reduced_result.selected_choice == "reducedVolume"
    assert normal_result.metrics.optional_working_sets_removed == 0
    assert reduced_result.metrics.optional_working_sets_removed == 1
    assert received_policies == [
        preflight.constraints.condition_policies,
        preflight.constraints.condition_policies,
    ]

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE user_condition_profile SET phase = 'active'"
        )

    with pytest.raises(RoutineGenerationChoiceError, match="does not match"):
        generate_routine_with_choice(
            request, object(), selection=normal, _generator=generator
        )
    assert len(received_policies) == 2


def test_default_generator_receives_one_resolved_policy_snapshot(tmp_path, monkeypatch):
    database_path = tmp_path / "health.sqlite"
    _health_db(database_path)
    monkeypatch.setattr(db_queries, "SQLITE_DB_PATH", database_path)
    request = _request()
    preflight = preflight_routine_generation(
        request, resolve_persisted_context=True
    )
    request_before = request.model_dump(mode="json")
    received = []
    loads = []
    original_loader = routine_generation_choice.get_user_constraint_profile_from_sqlite

    def tracked_loader(user_id):
        loads.append(user_id)
        return original_loader(user_id)

    def default_generator(resolved_request, _db, **kwargs):
        received.append((resolved_request.condition_policies, kwargs))
        return _routine()

    monkeypatch.setattr(
        routine_generation_choice, "generate_smart_routine", default_generator
    )
    monkeypatch.setattr(
        routine_generation_choice,
        "get_user_constraint_profile_from_sqlite",
        tracked_loader,
    )

    result = generate_routine_with_choice(
        request,
        object(),
        selection=make_routine_generation_selection(preflight, "normalVolume"),
    )

    assert result.status == "generated"
    assert loads == ["db-health-user"]
    assert request.model_dump(mode="json") == request_before
    assert received[0][0] == preflight.constraints.condition_policies
    assert received[0][1]["local_constraint_profile"]["condition_policies"] == received[0][0]
    assert received[0][1]["local_constraint_profile_is_resolved"] is True


def test_persisted_mobility_change_rejects_stale_choice(tmp_path, monkeypatch):
    database_path = tmp_path / "health.sqlite"
    _health_db(database_path)
    monkeypatch.setattr(db_queries, "SQLITE_DB_PATH", database_path)
    request = _request()
    preflight = preflight_routine_generation(
        request, resolve_persisted_context=True
    )
    selection = make_routine_generation_selection(preflight, "normalVolume")

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE user_mobility_profile SET shoulder_flexion_level = 'normal'"
        )

    with pytest.raises(RoutineGenerationChoiceError, match="does not match"):
        generate_routine_with_choice(
            request, object(), selection=selection, _generator=lambda *args, **kwargs: _routine()
        )

"""Internal v4/legacy checks; no HTTP contract, DB migration, or live inference."""
import copy
import sqlite3
from unittest.mock import MagicMock

import pytest

from engines import db_queries, routine_pipeline
from engines.routine_connection_boundary import (
    check_legacy_request, check_legacy_response, generate_checked_v4,
)
from engines.routine_response_v4 import RoutineResponseV4, build_v4_response
from engines.schemas import RoutineBlock, RoutineDraftResponse, RoutineRequest, SetPrescription


def request(**kwargs):
    return RoutineRequest.model_validate({"user_id": "synthetic-stage2", "time_available_min": 45,
                                         "target_split_label": "legs", **kwargs})


def routine(**prescription):
    return RoutineDraftResponse(generation_status="success", status_reason_code="none", is_fallback=False,
        total_estimated_time=30, summary_title="Synthetic", rationale_summary=[], warnings=[],
        routine_blocks=[RoutineBlock(order=1, exercise_id="14", exercise_name="Synthetic exercise",
            default_rest_sec=90, primary_muscles=["biceps"], exercise_rationale="Synthetic fixture",
            prescription=[SetPrescription(set_index=1, target_reps=8, target_rest_sec=90, **prescription)])])


@pytest.mark.parametrize("mode,status", [("absent", "not_found"), ("empty", "not_found"),
    ("found", "found"), ("connection", "error"), ("query", "error")])
def test_typed_lookup_distinguishes_absence_and_failure_without_changing_legacy(monkeypatch, mode, status):
    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = None
    connection.execute.return_value.fetchall.return_value = []
    if mode == "found":
        connection.execute.return_value.fetchone.return_value = {"shoulder_flexion_level": "limited"}
    if mode == "query":
        connection.execute.side_effect = sqlite3.OperationalError("synthetic failure")
    factory = MagicMock(return_value=None if mode == "connection" else connection)
    monkeypatch.setattr(db_queries, "_safe_local_catalog_connection", factory)
    lookup = db_queries.lookup_user_constraint_profile_from_sqlite(None if mode == "absent" else "synthetic")
    assert lookup.status == status
    if mode == "absent":
        factory.assert_not_called()
    elif mode != "connection":
        connection.close.assert_called_once()
    if mode == "found":
        assert lookup.profile["mobility_limits"] == ["limited_shoulder_flexion"]
    else:
        assert lookup.profile == {"mobility_limits": [], "anthropometry_signals": {}, "condition_policies": []}
    monkeypatch.setattr(db_queries, "lookup_user_constraint_profile_from_sqlite", lambda _: lookup)
    assert db_queries.get_user_constraint_profile_from_sqlite("synthetic") == lookup.profile


def test_lookup_error_stops_v4_before_model_or_fallback_without_new_public_fields(monkeypatch):
    monkeypatch.setattr(db_queries, "lookup_user_constraint_profile_from_sqlite",
                        lambda _: db_queries.ConstraintProfileLookup("error", {}))
    generate = MagicMock(side_effect=AssertionError("generation must not run"))
    monkeypatch.setattr(routine_pipeline, "generate_smart_routine", generate)
    result = generate_checked_v4(request(), None)
    generate.assert_not_called()
    assert result.routine.generation_status == "failed"
    assert result.routine.status_reason_code == "schemaError"
    assert result.routine.routine_blocks == [] and result.routine.total_estimated_time == 0
    assert result.routine.is_fallback is False and result.routine.warnings
    assert result.short_routine_notice.is_short_routine is False
    assert result.short_routine_notice.target_expansion_options == []
    assert set(result.model_dump(by_alias=True)) == {"contractVersion", "routine", "shortRoutineNotice"}


@pytest.mark.parametrize("status", ["found", "not_found"])
def test_checked_v4_loads_once_and_keeps_request_and_profile_constraints(monkeypatch, status):
    context = {"mobility_limits": ["limited_shoulder_flexion"] if status == "found" else [],
               "anthropometry_signals": {}, "condition_policies": []}
    loader = MagicMock(return_value=db_queries.ConstraintProfileLookup(status, context))
    generate = MagicMock(return_value=routine())
    monkeypatch.setattr(db_queries, "lookup_user_constraint_profile_from_sqlite", loader)
    monkeypatch.setattr(routine_pipeline, "generate_smart_routine", generate)
    req = request(doms_data={"left-soleus": 2})
    before = req.model_dump()
    result = generate_checked_v4(req, None)
    loader.assert_called_once_with(req.user_id)
    generate.assert_called_once()
    assert generate.call_args.kwargs["local_constraint_profile"] == context
    assert generate.call_args.kwargs["local_constraint_profile_is_resolved"] is False
    assert generate.call_args.args[0].doms_data == {"left-soleus": 2}
    assert req.model_dump() == before
    assert RoutineResponseV4.model_validate_json(result.model_dump_json(by_alias=True)).model_dump() == result.model_dump()


def test_found_profile_reaches_real_pipeline_without_second_legacy_lookup(monkeypatch):
    from engines import routine_generation_choice

    context = {"mobility_limits": ["limited_shoulder_flexion"],
               "anthropometry_signals": {"long_femur": True}, "condition_policies": []}
    typed = MagicMock(return_value=db_queries.ConstraintProfileLookup("found", context))
    forbidden = MagicMock(side_effect=AssertionError("second legacy lookup must not run"))
    monkeypatch.setattr(db_queries, "lookup_user_constraint_profile_from_sqlite", typed)
    for module in (db_queries, routine_pipeline, routine_generation_choice):
        monkeypatch.setattr(module, "get_user_constraint_profile_from_sqlite", forbidden)
    monkeypatch.setattr(routine_pipeline, "get_candidate_exercises", lambda *a, **kw: [])
    monkeypatch.setattr(routine_pipeline, "get_adjunct_candidate_exercises", lambda *a, **kw: [])
    score = MagicMock(wraps=routine_pipeline.score_candidate_exercises)
    monkeypatch.setattr(routine_pipeline, "score_candidate_exercises", score)
    model = MagicMock(side_effect=AssertionError("empty candidate pool must not invoke model"))
    monkeypatch.setattr(routine_pipeline, "get_llm", model)
    req = request(target_split_label="push", mobility_limits=["limited_ankle_dorsiflexion"])
    result = generate_checked_v4(req, None, recent_sets=[])
    typed.assert_called_once_with(req.user_id)
    forbidden.assert_not_called()
    model.assert_not_called()
    score.assert_called_once()
    assert set(score.call_args.kwargs["mobility_limits"]) == {
        "limited_shoulder_flexion", "limited_ankle_dorsiflexion"}
    assert score.call_args.kwargs["anthropometry_signals"] == context["anthropometry_signals"]
    assert result.routine.generation_status == "failed"
    assert result.routine.status_reason_code == "emptyCandidate"
    assert result.routine.routine_blocks == [] and result.routine.total_estimated_time == 0
    assert result.short_routine_notice.target_expansion_options == []


@pytest.mark.parametrize("kwargs,field", [
    ({"session_structure": "superset"}, "sessionStructure"),
    ({"adjunct_options": ["cooldown_stretch"]}, "adjunctOptions"),
    ({"mobility_limits": ["limited_ankle_dorsiflexion"]}, "mobilityLimits"),
    ({"doms_data": {"right-soleus": 2}}, "domsData"),
    ({"injury_conditions": [{"areaCode": "shoulder", "status": "activeUncleared"}]}, "injuryConditions"),
])
def test_legacy_precheck_reports_unsupported_without_altering_conditions(kwargs, field):
    req = request(**kwargs)
    before = copy.deepcopy(req.model_dump())
    result = check_legacy_request(req)
    assert result.supported is False and field in result.unsupported_fields
    assert req.model_dump() == before


def test_ordinary_legacy_checks_and_v4_no_implicit_downgrade():
    req, response = request(), routine()
    assert check_legacy_request(req).supported
    assert check_legacy_response(response).supported
    assert not check_legacy_response(build_v4_response(req, response)).supported


@pytest.mark.parametrize("extra", [
    {"set_type": "timed", "target_duration_sec": 30},
    {"set_type": "distance", "target_distance_m": 20},
    {"set_type": "interval", "target_duration_sec": 50,
     "interval_work_sec": 20, "interval_recovery_sec": 10, "interval_rounds": 2},
])
def test_prescription_units_round_trip_v4_but_reject_legacy(extra):
    response = routine(**extra)
    before = response.model_dump(mode="json", by_alias=True)
    assert not check_legacy_response(response).supported
    envelope = build_v4_response(request(), response)
    restored = RoutineResponseV4.model_validate_json(envelope.model_dump_json(by_alias=True))
    assert restored.routine.model_dump(mode="json", by_alias=True) == before
    assert response.model_dump(mode="json", by_alias=True) == before


def test_detail_and_group_meaning_cannot_be_silently_lost():
    response = routine()
    block = response.routine_blocks[0]
    block.primary_muscles = ["side-deltoids"]
    block.execution_group_type = "superset"
    block.execution_group_id = "pair-1"
    block.execution_group_size = 2
    result = check_legacy_response(response)
    assert "routineBlocks[0].primaryMuscleDetails" in result.unsupported_fields
    assert "routineBlocks[0].executionGroup" in result.unsupported_fields

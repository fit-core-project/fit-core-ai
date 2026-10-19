"""Core failure injection; synthetic candidates are not catalog approvals."""
from unittest.mock import MagicMock

import httpx
import pytest
from langchain_core.runnables import RunnableLambda

from engines import db_queries, routine_pipeline
from engines.routine_connection_boundary import generate_checked_v4
from engines.routine_response_v4 import RoutineResponseV4


@pytest.mark.parametrize("error,reason", [(httpx.TimeoutException, "llmTimeout"),
                                        (httpx.ConnectError, "networkError")])
def test_numeric_transport_survives_timeout_fallback(monkeypatch, sample_request, sample_profile, mock_candidates, error, reason):
    # Keep the legacy unit fixture geometry, only substitute numeric transport IDs.
    candidates = [dict(row, id=str(90001 + i)) for i, row in enumerate(mock_candidates)]
    loader = MagicMock(return_value=db_queries.ConstraintProfileLookup("not_found", {
        "mobility_limits": [], "anthropometry_signals": {}, "condition_policies": []}))
    calls = []

    def timeout(_):
        calls.append(reason)
        raise error("synthetic unavailable model; no network")

    model = MagicMock()
    model.with_structured_output.return_value = RunnableLambda(timeout)
    monkeypatch.setattr(db_queries, "lookup_user_constraint_profile_from_sqlite", loader)
    monkeypatch.setattr(routine_pipeline, "get_candidate_exercises", lambda *a, **kw: candidates)
    monkeypatch.setattr(routine_pipeline, "get_adjunct_candidate_exercises", lambda *a, **kw: [])
    monkeypatch.setattr(routine_pipeline, "get_llm", lambda *a, **kw: model)
    result = generate_checked_v4(sample_request, None, profile=sample_profile, recent_sets=[])
    assert calls == [reason]
    loader.assert_called_once()
    assert result.routine.is_fallback is True
    assert result.routine.generation_status == "fallback"
    assert result.routine.status_reason_code == reason
    assert result.routine.routine_blocks
    assert {b.exercise_id for b in result.routine.routine_blocks} <= {c["id"] for c in candidates}
    assert all(b.exercise_id.isdigit() for b in result.routine.routine_blocks)
    # Public JSON must round-trip; Pydantic private diagnostics are not wire fields.
    payload = result.model_dump(mode="json", by_alias=True)
    restored = RoutineResponseV4.model_validate_json(result.model_dump_json(by_alias=True))
    assert restored.model_dump(mode="json", by_alias=True) == payload
    assert "_fallback_selection_diagnostics" not in payload["routine"]
    assert result.routine._fallback_selection_diagnostics


@pytest.mark.parametrize("reason", ["llmTimeout", "schemaError"])
def test_invalid_fallback_after_model_failure_stays_failed(reason, sample_request):
    candidates = [{"id": "90004", "name_kr": "케이블 푸시다운", "primary_muscle": "triceps",
                   "movement_type": "ISOLATION", "equipment_req": "CABLE", "score": 100}]
    result = routine_pipeline._fallback(sample_request, candidates, 18, {}, "hypertrophy", reason, [], None)
    assert result.generation_status == "failed"
    assert result.status_reason_code == "emptyCandidate"
    assert result.routine_blocks == [] and result.total_estimated_time == 0

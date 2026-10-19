from engines.schemas import RoutineDraftResponse
from scripts import run_gemma4_quality_seed_live as live


def test_result_preserves_measured_underfill_without_inferring_a_cause():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "failed", "statusReasonCode": "emptyCandidate", "isFallback": False,
        "totalEstimatedTime": 0, "summaryTitle": "No plan",
        "rationaleSummary": [], "routineBlocks": [],
    })
    row = {"scenarioId": "measured-duration", "title": "Measured duration",
           "input": {"timeAvailableMin": 60}, "failurePatterns": []}
    without = live.result_row(row, response, 1)
    assert without["underfillDiagnostics"] == {}
    response._underfill_diagnostics = {
        "reason_code": "recovery_constraint", "actual_duration_min": 22,
        "available_duration_min": 60, "underfilled": True,
        "role_set_capacity_exhausted": False,
    }
    result = live.result_row(row, response, 1)
    assert result["underfillDiagnostics"] == response._underfill_diagnostics
    result["underfillDiagnostics"]["reason_code"] = "changed_by_reader"
    assert response._underfill_diagnostics["reason_code"] == "recovery_constraint"
    assert "underfillDiagnostics" not in response.model_dump(by_alias=True)


def test_safe_telemetry_keeps_underfill_measurements_without_raw_prompt():
    event = {"underfill_reason_code": "role_set_capacity_exhausted",
             "underfill_detected": True, "underfill_utilization_pct": 58.3,
             "underfill_reference_duration_min": 60,
             "underfill_minimum_productive_duration_min": 40,
             "prompt": "private", "raw_model_output": "private"}
    result = live.summarize_telemetry_event(event)
    assert result == {key: value for key, value in event.items()
                      if key not in {"prompt", "raw_model_output"}}

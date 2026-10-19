import json
from pathlib import Path

import pytest

from tests.evaluation.helpers import EvalHarness, ScenarioResult, check_hard_constraints, save_report

# The collaboration baseline is preserved; use the approved core policy version.
SCENARIOS_PATH = Path(__file__).parent / "fixtures" / "scenarios_core_v3.json"
SCENARIOS = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
REPORT_PATH = Path(__file__).parent / ".artifacts" / "offline_eval_report.json"
RESULTS: list[ScenarioResult] = []


@pytest.fixture(scope="session")
def harness() -> EvalHarness:
    return EvalHarness()


@pytest.mark.eval
@pytest.mark.parametrize("scenario", SCENARIOS, ids=[scenario["id"] for scenario in SCENARIOS])
def test_offline_eval_scenario(harness: EvalHarness, scenario: dict):
    result = harness.run(scenario)
    RESULTS.append(result)
    save_report(RESULTS, REPORT_PATH)

    scenario_id = scenario["id"]
    expected = scenario.get("expected", {})

    assert result.hard_violations == 0, (
        f"[{scenario_id}] metric=hard_constraints actual={result.hard_violation_details} "
        f"expected=[] warnings={result.warnings}"
    )

    assert result.fallback_used == bool(expected.get("is_fallback", False)), (
        f"[{scenario_id}] metric=fallback_used actual={result.fallback_used} "
        f"expected={expected.get('is_fallback', False)} warnings={result.warnings}"
    )

    assert result.repair_used == bool(expected.get("expected_repair", False)), (
        f"[{scenario_id}] metric=repair_used actual={result.repair_used} "
        f"expected={expected.get('expected_repair', False)} warnings={result.warnings}"
    )

    for forbidden_id in expected.get("forbidden_exercise_ids", []):
        actual_ids = [block["exercise_id"] for block in result.routine_blocks]
        assert forbidden_id not in actual_ids, (
            f"[{scenario_id}] metric=forbidden_exercise actual={actual_ids} "
            f"expected_absent={forbidden_id} warnings={result.warnings}"
        )

    expected_generation_status = expected.get("generation_status")
    expected_status_reason_code = expected.get("status_reason_code")
    if expected_generation_status:
        assert result.generation_status == expected_generation_status
        if expected_status_reason_code:
            assert result.status_reason_code == expected_status_reason_code
    else:
        fail_warnings = [warning for warning in result.warnings if warning.startswith("FAIL ")]
        assert not fail_warnings, (
            f"[{scenario_id}] metric=metric_failures actual={fail_warnings} "
            f"expected=[] scores={result.metric_scores} warnings={result.warnings}"
        )

    if expected.get("expected_hard_stop", False):
        assert result.generation_status == "failed"
        assert result.status_reason_code == "emptyCandidate"
        assert result.fallback_used is False
        assert result.repair_used is False
        assert result.total_estimated_time == 0
        assert result.routine_blocks == []

    if expected.get("must_pass", True):
        if not expected_generation_status:
            min_total_score = int(expected.get("min_total_score", 80))
            assert result.total_score >= min_total_score, (
                f"[{scenario_id}] metric=total_score actual={result.total_score} "
                f"expected>={min_total_score} scores={result.metric_scores} warnings={result.warnings}"
            )
        assert result.passed, (
            f"[{scenario_id}] metric=passed actual={result.passed} expected=True "
            f"scores={result.metric_scores} warnings={result.warnings}"
        )


def test_scenario_fixture_contains_required_15_cases():
    ids = [scenario["id"] for scenario in SCENARIOS]
    assert len(SCENARIOS) == 15
    assert ids == [f"S{i:02d}" for i in range(1, 16)]


def test_expected_hard_stop_fixtures_define_the_locked_contract():
    hard_stop_scenarios = [
        scenario for scenario in SCENARIOS
        if scenario.get("expected", {}).get("expected_hard_stop", False)
    ]

    assert {scenario["id"] for scenario in hard_stop_scenarios} == {"S04", "S05", "S06"}
    for scenario in hard_stop_scenarios:
        expected = scenario["expected"]
        assert expected["generation_status"] == "failed"
        assert expected["status_reason_code"] == "emptyCandidate"
        assert expected["is_fallback"] is False
        assert expected["expected_repair"] is False


def test_hard_constraint_gate_detects_final_output_violation():
    blocks = [{
        "exercise_id": "barbell_bench_press",
        "primary_muscles": ["chest"],
        "prescription": [{"set_index": 1}],
    }]
    candidates = [{
        "id": "barbell_bench_press",
        "primary_muscle": "chest",
        "equipment_req": "BARBELL",
        "pain_triggers": "shoulder",
    }]
    scenario = {
        "unavailable_equipment": ["BARBELL"],
        "current_pain_areas": [{"body_part": "shoulder"}],
        "doms": {"chest": 3},
        "expected": {"forbidden_exercise_ids": []},
    }

    violations = check_hard_constraints(blocks, candidates, scenario, max_working_sets=0)

    assert any("blocked equipment" in violation for violation in violations)
    assert any("registered injury overlap" in violation for violation in violations)
    assert any("DOMS level 3" in violation for violation in violations)
    assert any("working set cap exceeded" in violation for violation in violations)

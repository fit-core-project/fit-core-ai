import json
from pathlib import Path

import pytest

from tests.evaluation.helpers import EvalHarness, ScenarioResult, check_hard_constraints, save_report

SCENARIOS_PATH = Path(__file__).parent / "fixtures" / "scenarios.json"
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

    fail_warnings = [warning for warning in result.warnings if warning.startswith("FAIL ")]
    assert not fail_warnings, (
        f"[{scenario_id}] metric=metric_failures actual={fail_warnings} "
        f"expected=[] scores={result.metric_scores} warnings={result.warnings}"
    )

    if expected.get("must_pass", True):
        assert result.total_score >= 80, (
            f"[{scenario_id}] metric=total_score actual={result.total_score} expected>=80 "
            f"scores={result.metric_scores} warnings={result.warnings}"
        )
        assert result.passed, (
            f"[{scenario_id}] metric=passed actual={result.passed} expected=True "
            f"scores={result.metric_scores} warnings={result.warnings}"
        )


def test_scenario_fixture_contains_required_15_cases():
    ids = [scenario["id"] for scenario in SCENARIOS]
    assert len(SCENARIOS) == 15
    assert ids == [f"S{i:02d}" for i in range(1, 16)]


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
    assert any("pain trigger" in violation for violation in violations)
    assert any("DOMS level 3" in violation for violation in violations)
    assert any("working set cap exceeded" in violation for violation in violations)

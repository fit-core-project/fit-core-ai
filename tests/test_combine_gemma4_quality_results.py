import json
from pathlib import Path

import pytest

from scripts import combine_gemma4_quality_results as combiner


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _result_payload(run_id: str, scenario_ids: list[str]) -> dict:
    return {
        "runId": run_id,
        "mode": "live-results",
        "seedPath": "seed.jsonl",
        "scenarioCount": len(scenario_ids),
        "completedScenarioCount": len(scenario_ids),
        "failedScenarioCount": 0,
        "selectedScenarioIds": scenario_ids,
        "completedScenarioIds": scenario_ids,
        "failures": [],
        "scenarios": [
            {
                "scenarioId": scenario_id,
                "title": f"title {scenario_id}",
                "selectedExerciseIds": [],
                "rationaleText": "",
                "warnings": [],
                "contractValid": True,
                "hardViolationCount": 0,
                "notes": "ok",
            }
            for scenario_id in scenario_ids
        ],
    }


def _write_seed(path: Path, scenario_ids: list[str]) -> Path:
    rows = [
        {
            "scenarioId": scenario_id,
            "title": f"title {scenario_id}",
            "input": {},
            "expectedHardConstraints": [],
            "preferredPatterns": [],
            "failurePatterns": [],
            "qualityRubric": [],
        }
        for scenario_id in scenario_ids
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    return path


def test_combine_payloads_preserves_seed_order(tmp_path):
    seed_path = _write_seed(tmp_path / "seed.jsonl", ["s1", "s2", "s3"])
    second = _write_json(tmp_path / "second.json", _result_payload("second", ["s3"]))
    first = _write_json(tmp_path / "first.json", _result_payload("first", ["s1", "s2"]))

    payload = combiner.combine_payloads(
        result_paths=[second, first],
        run_id="combined",
        seed_path=seed_path,
    )

    assert payload["mode"] == "live-results-combined"
    assert payload["selectedScenarioIds"] == ["s1", "s2", "s3"]
    assert payload["completedScenarioCount"] == 3
    assert payload["missingSeedScenarioIds"] == []
    assert [item["scenarioId"] for item in payload["scenarios"]] == ["s1", "s2", "s3"]
    assert payload["sourceRuns"][0]["runId"] == "second"


def test_combine_payloads_reports_missing_seed_ids(tmp_path):
    seed_path = _write_seed(tmp_path / "seed.jsonl", ["s1", "s2"])
    first = _write_json(tmp_path / "first.json", _result_payload("first", ["s1"]))

    payload = combiner.combine_payloads(
        result_paths=[first],
        run_id="combined",
        seed_path=seed_path,
    )

    assert payload["selectedScenarioIds"] == ["s1"]
    assert payload["missingSeedScenarioIds"] == ["s2"]


def test_combine_payloads_rejects_duplicate_scenarios(tmp_path):
    first = _write_json(tmp_path / "first.json", _result_payload("first", ["s1"]))
    second = _write_json(tmp_path / "second.json", _result_payload("second", ["s1"]))

    with pytest.raises(ValueError, match="duplicate scenarioId"):
        combiner.combine_payloads(
            result_paths=[first, second],
            run_id="combined",
        )

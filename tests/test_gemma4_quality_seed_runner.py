from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_gemma4_quality_seed as runner


def _seed_file(tmp_path: Path) -> Path:
    path = tmp_path / "seed.jsonl"
    rows = [
        {
            "scenarioId": "s1",
            "title": "Shoulder push",
            "input": {"currentPainAreas": ["front-deltoids"]},
            "expectedHardConstraints": ["avoid_overhead_press"],
            "preferredPatterns": ["machine_press"],
            "failurePatterns": ["overhead_press"],
            "qualityRubric": ["valid_contract_json"],
        },
        {
            "scenarioId": "s2",
            "title": "Low readiness",
            "input": {"readinessLevel": "low"},
            "expectedHardConstraints": ["low_volume"],
            "preferredPatterns": ["lower_volume"],
            "failurePatterns": ["high_volume_compound_stack"],
            "qualityRubric": ["readiness_reflected"],
        },
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    return path


def test_load_seed_validates_required_fields(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"scenarioId": "bad"}, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="missing fields"):
        runner.load_seed(path)


def test_build_plan_report_uses_readiness_next_action(monkeypatch, tmp_path):
    seed = _seed_file(tmp_path)
    monkeypatch.setattr(
        runner.readiness,
        "build_readiness_report",
        lambda **_kwargs: {
            "readiness_status": "failed_ollama_unreachable",
            "error_category": "ollama_unreachable",
            "next_action": "windows_ollama_detected_but_http_unreachable",
            "ollama_reachable": False,
            "model_installed": False,
            "windows_install_detected": True,
            "windows_executable_detected": True,
            "base_url_host": "127.0.0.1",
        },
    )

    report = runner.build_plan_report(
        seed_path=seed,
        output_dir=tmp_path / "out",
        run_id="run",
        model="gemma4:latest",
        ollama_base_url="http://127.0.0.1:11434",
        timeout_sec=1,
        probe=False,
    )

    assert report["can_run_live_eval"] is False
    assert report["seed_summary"]["scenario_count"] == 2
    assert "open Windows Ollama host" in report["next_actions"][0]


def test_score_file_report_flags_failure_patterns(tmp_path):
    seed = _seed_file(tmp_path)
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "scenarioId": "s1",
                        "selectedExerciseIds": ["overhead_press"],
                        "rationaleText": "overhead_press selected",
                        "contractValid": True,
                        "hardViolationCount": 0,
                    },
                    {
                        "scenarioId": "s2",
                        "selectedExerciseIds": ["machine_row"],
                        "rationaleText": "lower_volume due to low readiness",
                        "contractValid": True,
                        "hardViolationCount": 0,
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = runner.build_score_report(
        seed_path=seed,
        results_path=results,
        output_dir=tmp_path / "out",
        run_id="run",
    )

    assert report["passed"] == 1
    assert report["failed"] == 1
    assert report["recommendation"] == "needs_review"
    assert report["scenario_reports"][0]["failureHits"] == ["overhead_press"]


def test_build_results_template_contains_all_seed_scenarios(tmp_path):
    seed = _seed_file(tmp_path)

    template = runner.build_results_template(
        seed_path=seed,
        output_dir=tmp_path / "out",
        run_id="run",
    )

    assert template["mode"] == "results-template"
    assert [item["scenarioId"] for item in template["scenarios"]] == ["s1", "s2"]
    assert template["scenarios"][0]["selectedExerciseIds"] == []
    assert template["scenarios"][0]["contractValid"] is False
    assert "inputSummary" in template["scenarios"][0]
    runner.assert_no_forbidden_tokens(template)


def test_write_results_template_creates_json(tmp_path):
    seed = _seed_file(tmp_path)
    template = runner.build_results_template(
        seed_path=seed,
        output_dir=tmp_path / "out",
        run_id="run",
    )

    paths = runner.write_results_template(template, tmp_path / "out")
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))

    assert payload["runId"] == "run"
    assert len(payload["scenarios"]) == 2


def test_write_report_rejects_sensitive_tokens(tmp_path):
    report = {
        "run_id": "run",
        "mode": "plan",
        "seed_path": "seed",
        "ollama_readiness": {"readiness_status": "pass", "api_key": "secret"},
        "seed_summary": {"scenario_count": 0, "hard_constraint_count": 0, "failure_pattern_count": 0},
        "next_actions": [],
    }

    with pytest.raises(ValueError, match="forbidden report tokens"):
        runner.write_report(report, tmp_path)

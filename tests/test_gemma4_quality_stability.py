from pathlib import Path

from scripts import run_gemma4_quality_seed as seed_runner
from scripts import summarize_gemma4_quality_stability as stability


def test_stability_report_summarizes_repeated_result_files(tmp_path):
    seed_path = tmp_path / "seed.jsonl"
    run1_path = tmp_path / "run1.json"
    run2_path = tmp_path / "run2.json"
    seed_path.write_text(
        seed_runner.json.dumps({
            "scenarioId": "stable-scenario",
            "title": "stable scenario",
            "input": {"targetSplitLabel": "legs"},
            "expectedHardConstraints": [],
            "preferredPatterns": ["leg_press"],
            "failurePatterns": [],
            "qualityRubric": ["repeat_stability"],
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for path in (run1_path, run2_path):
        path.write_text(
            seed_runner.json.dumps({
                "scenarios": [{
                    "scenarioId": "stable-scenario",
                    "selectedExerciseIds": ["leg_press"],
                    "rationaleText": "leg_press 후보를 선택했습니다.",
                    "scoreEvidenceText": "leg_press: boosts=primary_target_match",
                    "warnings": [],
                    "contractValid": True,
                    "hardViolationCount": 0,
                    "notes": "generationStatus=success isFallback=False elapsedMs=1000",
                }]
            }, ensure_ascii=False),
            encoding="utf-8",
        )

    report = stability.build_stability_report(
        seed_path=seed_path,
        results_paths=[run1_path, run2_path],
        output_dir=tmp_path / "out",
        run_id="stability",
        min_preferred_hits=1,
        max_elapsed_ms=2000,
        disallow_fallback=True,
    )

    scenario = report["scenarios"][0]
    assert report["recommendation"] == "pass"
    assert report["stablePassed"] == 1
    assert scenario["passCount"] == 2
    assert scenario["qualityPassCount"] == 2
    assert scenario["latencyPassCount"] == 2
    assert scenario["fallbackPassCount"] == 2
    assert scenario["uniqueSelectionCount"] == 1
    assert scenario["selectionStable"] is True


def test_stability_report_marks_selection_drift_for_review(tmp_path):
    seed_path = tmp_path / "seed.jsonl"
    run1_path = tmp_path / "run1.json"
    run2_path = tmp_path / "run2.json"
    seed_path.write_text(
        seed_runner.json.dumps({
            "scenarioId": "drift-scenario",
            "title": "drift scenario",
            "input": {"targetSplitLabel": "legs"},
            "expectedHardConstraints": [],
            "preferredPatterns": ["machine"],
            "failurePatterns": [],
            "qualityRubric": ["repeat_stability"],
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    run1_path.write_text(
        seed_runner.json.dumps({
            "scenarios": [{
                "scenarioId": "drift-scenario",
                "selectedExerciseIds": ["leg_press"],
                "rationaleText": "machine 후보",
                "scoreEvidenceText": "machine",
                "warnings": [],
                "contractValid": True,
                "hardViolationCount": 0,
                "notes": "generationStatus=success isFallback=False elapsedMs=1000",
            }]
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    run2_path.write_text(
        seed_runner.json.dumps({
            "scenarios": [{
                "scenarioId": "drift-scenario",
                "selectedExerciseIds": ["leg_curl"],
                "rationaleText": "machine 후보",
                "scoreEvidenceText": "machine",
                "warnings": [],
                "contractValid": True,
                "hardViolationCount": 0,
                "notes": "generationStatus=success isFallback=False elapsedMs=1000",
            }]
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    report = stability.build_stability_report(
        seed_path=seed_path,
        results_paths=[run1_path, run2_path],
        output_dir=tmp_path / "out",
        run_id="stability",
    )

    scenario = report["scenarios"][0]
    assert report["recommendation"] == "needs_review"
    assert scenario["passCount"] == 2
    assert scenario["uniqueSelectionCount"] == 2
    assert scenario["selectionStable"] is False


def test_stability_report_can_focus_on_selected_scenarios(tmp_path):
    seed_path = tmp_path / "seed.jsonl"
    run_path = tmp_path / "run.json"
    seed_path.write_text(
        "\n".join([
            seed_runner.json.dumps({
                "scenarioId": "included",
                "title": "included scenario",
                "input": {"targetSplitLabel": "legs"},
                "expectedHardConstraints": [],
                "preferredPatterns": ["leg_press"],
                "failurePatterns": [],
                "qualityRubric": [],
            }, ensure_ascii=False),
            seed_runner.json.dumps({
                "scenarioId": "excluded",
                "title": "excluded scenario",
                "input": {"targetSplitLabel": "legs"},
                "expectedHardConstraints": [],
                "preferredPatterns": ["leg_press"],
                "failurePatterns": [],
                "qualityRubric": [],
            }, ensure_ascii=False),
        ]) + "\n",
        encoding="utf-8",
    )
    run_path.write_text(
        seed_runner.json.dumps({
            "scenarios": [{
                "scenarioId": "included",
                "selectedExerciseIds": ["leg_press"],
                "rationaleText": "leg_press 후보를 선택했습니다.",
                "scoreEvidenceText": "leg_press",
                "warnings": [],
                "contractValid": True,
                "hardViolationCount": 0,
                "notes": "generationStatus=success isFallback=False elapsedMs=1000",
            }]
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    report = stability.build_stability_report(
        seed_path=seed_path,
        results_paths=[run_path],
        output_dir=tmp_path / "out",
        run_id="stability",
        min_preferred_hits=1,
        max_elapsed_ms=2000,
        disallow_fallback=True,
        scenario_ids=["included"],
    )

    assert report["recommendation"] == "pass"
    assert report["selectedScenarioIds"] == ["included"]
    assert [item["scenarioId"] for item in report["scenarios"]] == ["included"]

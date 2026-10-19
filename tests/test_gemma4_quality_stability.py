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
    assert scenario["outputQualityPassCount"] == 2
    assert scenario["languageQualityPassCount"] == 2
    assert scenario["latencyPassCount"] == 2
    assert scenario["fallbackPassCount"] == 2
    assert scenario["uniqueSelectionCount"] == 1
    assert scenario["selectionStable"] is True


def test_stability_report_surfaces_output_quality_regression(tmp_path):
    seed_path = tmp_path / "seed.jsonl"
    run_path = tmp_path / "run.json"
    seed_path.write_text(
        seed_runner.json.dumps({
            "scenarioId": "output-quality-regression",
            "title": "output quality regression",
            "input": {"targetSplitLabel": "push"},
            "expectedHardConstraints": [],
            "preferredPatterns": [],
            "failurePatterns": [],
            "qualityRubric": [],
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    run_path.write_text(
        seed_runner.json.dumps({
            "scenarios": [{
                "scenarioId": "output-quality-regression",
                "selectedExerciseIds": ["machine_pushdown", "cable_pushdown"],
                "rationaleText": "삼두 운동 두 개를 선택했습니다.",
                "scoreEvidenceText": "",
                "warnings": [],
                "contractValid": True,
                "hardViolationCount": 0,
                "outputQualityFlags": ["duplicate_movement_family"],
                "notes": "generationStatus=success isFallback=False elapsedMs=1000",
            }]
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    report = stability.build_stability_report(
        seed_path=seed_path,
        results_paths=[run_path],
        output_dir=tmp_path / "out",
        run_id="output-quality-regression",
    )

    scenario = report["scenarios"][0]
    assert report["recommendation"] == "needs_review"
    assert scenario["passCount"] == 0
    assert scenario["outputQualityPassCount"] == 0
    assert "inspect time utilization, exercise count, and movement-family diversity" in scenario["actionItems"]


def test_stability_report_surfaces_language_quality_regression(tmp_path):
    seed_path = tmp_path / "seed.jsonl"
    run_path = tmp_path / "run.json"
    seed_path.write_text(
        seed_runner.json.dumps({
            "scenarioId": "language-quality-regression",
            "title": "language quality regression",
            "input": {"targetSplitLabel": "legs"},
            "expectedHardConstraints": [],
            "preferredPatterns": [],
            "failurePatterns": [],
            "qualityRubric": [],
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    run_path.write_text(
        seed_runner.json.dumps({
            "scenarios": [{
                "scenarioId": "language-quality-regression",
                "selectedExerciseIds": ["leg_press"],
                "rationaleText": "일반적인 운동입니다.",
                "scoreEvidenceText": "",
                "warnings": [],
                "contractValid": True,
                "hardViolationCount": 0,
                "languageQualityFlags": ["generic_rationale"],
                "notes": "generationStatus=success isFallback=False elapsedMs=1000",
            }]
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    report = stability.build_stability_report(
        seed_path=seed_path,
        results_paths=[run_path],
        output_dir=tmp_path / "out",
        run_id="language-quality-regression",
    )

    scenario = report["scenarios"][0]
    assert report["recommendation"] == "needs_review"
    assert scenario["passCount"] == 0
    assert scenario["languageQualityPassCount"] == 0
    assert "inspect user-facing Korean wording and unsupported coaching claims" in scenario["actionItems"]


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

    variation_allowed = stability.build_stability_report(
        seed_path=seed_path,
        results_paths=[run1_path, run2_path],
        output_dir=tmp_path / "variation-out",
        run_id="variation-allowed",
        require_exact_selection_stability=False,
    )

    variation_scenario = variation_allowed["scenarios"][0]
    assert variation_allowed["recommendation"] == "pass"
    assert variation_allowed["stablePassed"] == 1
    assert variation_scenario["exactSelectionStable"] is False
    assert variation_scenario["stabilityGatePassed"] is True
    assert variation_scenario["actionItems"] == []
    assert variation_scenario["informationalNotes"] == [
        "valid exercise selection varied while every quality gate passed"
    ]


def test_stability_report_summarizes_structured_runtime_fields(tmp_path):
    seed_path = tmp_path / "seed.jsonl"
    run_path = tmp_path / "run.json"
    seed_path.write_text(
        seed_runner.json.dumps({
            "scenarioId": "timeout-scenario",
            "title": "timeout scenario",
            "input": {"targetSplitLabel": "legs"},
            "expectedHardConstraints": [],
            "preferredPatterns": [],
            "failurePatterns": [],
            "qualityRubric": [],
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    run_path.write_text(
        seed_runner.json.dumps({
            "scenarios": [{
                "scenarioId": "timeout-scenario",
                "generationStatus": "fallback",
                "statusReasonCode": "llmTimeout",
                "isFallback": True,
                "elapsedMs": 240123,
                "selectedExerciseIds": ["leg_press"],
                "rationaleText": "fallback leg_press",
                "scoreEvidenceText": "",
                "warnings": [],
                "contractValid": True,
                "hardViolationCount": 0,
                "notes": "",
            }]
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    report = stability.build_stability_report(
        seed_path=seed_path,
        results_paths=[run_path],
        output_dir=tmp_path / "out",
        run_id="stability",
    )

    scenario = report["scenarios"][0]
    assert report["generationStatusCounts"] == {"fallback": 1}
    assert report["statusReasonCodeCounts"] == {"llmTimeout": 1}
    assert report["recommendation"] == "pass_with_runtime_observation"
    assert report["totalFallbackRuns"] == 1
    assert report["totalNonSuccessRuns"] == 1
    assert report["totalTimeoutRuns"] == 1
    assert report["scenarioActionCount"] == 2
    assert scenario["fallbackRunCount"] == 1
    assert scenario["nonSuccessRunCount"] == 1
    assert scenario["timeoutRunCount"] == 1
    assert scenario["generationStatusCounts"] == {"fallback": 1}
    assert scenario["statusReasonCodeCounts"] == {"llmTimeout": 1}
    assert scenario["actionItems"] == [
        "inspect fallback output quality and fallback trigger reason",
        "review local LLM latency and compact prompt settings",
    ]


def test_stability_report_canonicalizes_legacy_notes_runtime_fields(tmp_path):
    seed_path = tmp_path / "seed.jsonl"
    run_path = tmp_path / "run.json"
    seed_path.write_text(
        seed_runner.json.dumps({
            "scenarioId": "legacy-timeout-scenario",
            "title": "legacy timeout scenario",
            "input": {"targetSplitLabel": "legs"},
            "expectedHardConstraints": [],
            "preferredPatterns": [],
            "failurePatterns": [],
            "qualityRubric": [],
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    run_path.write_text(
        seed_runner.json.dumps({
            "scenarios": [{
                "scenarioId": "legacy-timeout-scenario",
                "selectedExerciseIds": ["leg_press"],
                "rationaleText": "fallback leg_press",
                "scoreEvidenceText": "",
                "warnings": [],
                "contractValid": True,
                "hardViolationCount": 0,
                "notes": "generationStatus=fallback statusReasonCode=llmTimeout isFallback=True elapsedMs=240123",
            }]
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    report = stability.build_stability_report(
        seed_path=seed_path,
        results_paths=[run_path],
        output_dir=tmp_path / "out",
        run_id="stability",
    )

    assert report["generationStatusCounts"] == {"fallback": 1}
    assert report["statusReasonCodeCounts"] == {"llmTimeout": 1}
    assert report["recommendation"] == "pass_with_runtime_observation"
    assert report["totalFallbackRuns"] == 1
    assert report["totalTimeoutRuns"] == 1


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

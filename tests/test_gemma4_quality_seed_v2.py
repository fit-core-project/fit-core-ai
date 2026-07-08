from pathlib import Path

from scripts import run_gemma4_quality_seed as runner


V2_SEED = Path("tests/fixtures/gemma4_routine_quality_eval_seed_v2.jsonl")


def test_gemma4_quality_seed_v2_loads_and_summarizes():
    rows = runner.load_seed(V2_SEED)
    summary = runner.summarize_seed(rows)

    assert summary["scenario_count"] == 12
    assert summary["scenario_ids"] == [
        "lower-back-legs-001",
        "limited-ankle-legs-001",
        "shoulder-push-001",
        "low-readiness-fullbody-001",
        "home-equipment-001",
        "short-time-push-001",
        "beginner-lower-skill-001",
        "post-surgery-caution-001",
        "all-major-pain-extreme-001",
        "high-risk-high-effect-substitution-001",
        "long-femur-squat-setup-001",
        "profile-injury-plus-doms-001",
    ]
    assert summary["hard_constraint_count"] >= 20
    assert summary["failure_pattern_count"] >= 25
    assert summary["rubric_item_count"] >= 30


def test_gemma4_quality_seed_v2_template_is_report_safe(tmp_path):
    template = runner.build_results_template(
        seed_path=V2_SEED,
        output_dir=tmp_path / "out",
        run_id="v2",
    )

    assert len(template["scenarios"]) == 12
    assert template["scenarios"][0]["scenarioId"] == "lower-back-legs-001"
    runner.assert_no_forbidden_tokens(template)


def test_score_report_separates_hard_safety_from_preferred_quality(tmp_path):
    seed_path = tmp_path / "seed.jsonl"
    results_path = tmp_path / "results.json"
    seed_path.write_text(
        "\n".join([
            runner.json.dumps({
                "scenarioId": "safety-pass-quality-gap",
                "title": "safety pass but quality gap",
                "input": {"targetSplitLabel": "legs"},
                "expectedHardConstraints": ["no_unsafe_pick"],
                "preferredPatterns": ["leg_press"],
                "failurePatterns": ["deadlift_selected_without_lumbar_reason"],
                "qualityRubric": ["preferred_candidate_visible"],
            }, ensure_ascii=False)
        ]),
        encoding="utf-8",
    )
    results_path.write_text(
        runner.json.dumps({
            "scenarios": [{
                "scenarioId": "safety-pass-quality-gap",
                "selectedExerciseIds": ["leg_curl"],
                "rationaleText": "안전한 후보를 선택했습니다.",
                "scoreEvidenceText": "leg_curl: boosts=primary_target_match penalties=none",
                "warnings": [],
                "contractValid": True,
                "hardViolationCount": 0,
                "notes": "generationStatus=success",
            }]
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    report = runner.build_score_report(
        seed_path=seed_path,
        results_path=results_path,
        output_dir=tmp_path / "out",
        run_id="quality-gap",
        min_preferred_hits=1,
    )

    scenario = report["scenario_reports"][0]
    assert report["passed"] == 1
    assert report["failed"] == 0
    assert report["quality_passed"] == 0
    assert report["quality_failed"] == 1
    assert report["recommendation"] == "pass_with_quality_gaps"
    assert scenario["hardSafetyPassed"] is True
    assert scenario["preferredQualityPassed"] is False
    assert scenario["missingPreferredPatterns"] == ["leg_press"]


def test_score_report_marks_preferred_quality_pass_when_pattern_is_visible(tmp_path):
    seed_path = tmp_path / "seed.jsonl"
    results_path = tmp_path / "results.json"
    seed_path.write_text(
        runner.json.dumps({
            "scenarioId": "quality-pass",
            "title": "quality pass",
            "input": {"targetSplitLabel": "legs"},
            "expectedHardConstraints": [],
            "preferredPatterns": ["leg_press"],
            "failurePatterns": [],
            "qualityRubric": ["preferred_candidate_visible"],
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    results_path.write_text(
        runner.json.dumps({
            "scenarios": [{
                "scenarioId": "quality-pass",
                "selectedExerciseIds": ["leg_press"],
                "rationaleText": "leg_press 후보를 선택했습니다.",
                "scoreEvidenceText": "leg_press: boosts=primary_target_match",
                "warnings": [],
                "contractValid": True,
                "hardViolationCount": 0,
                "notes": "generationStatus=success",
            }]
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    report = runner.build_score_report(
        seed_path=seed_path,
        results_path=results_path,
        output_dir=tmp_path / "out",
        run_id="quality-pass",
        min_preferred_hits=1,
    )

    scenario = report["scenario_reports"][0]
    assert report["quality_passed"] == 1
    assert report["quality_failed"] == 0
    assert report["recommendation"] == "pass"
    assert scenario["preferredQualityPassed"] is True
    assert scenario["preferredHits"] == ["leg_press"]


def test_score_report_can_fail_on_latency_threshold(tmp_path):
    seed_path = tmp_path / "seed.jsonl"
    results_path = tmp_path / "results.json"
    seed_path.write_text(
        runner.json.dumps({
            "scenarioId": "slow-quality-pass",
            "title": "slow quality pass",
            "input": {"targetSplitLabel": "legs"},
            "expectedHardConstraints": [],
            "preferredPatterns": ["leg_press"],
            "failurePatterns": [],
            "qualityRubric": ["latency_bound"],
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    results_path.write_text(
        runner.json.dumps({
            "scenarios": [{
                "scenarioId": "slow-quality-pass",
                "selectedExerciseIds": ["leg_press"],
                "rationaleText": "leg_press 후보를 선택했습니다.",
                "scoreEvidenceText": "leg_press: boosts=primary_target_match",
                "warnings": [],
                "contractValid": True,
                "hardViolationCount": 0,
                "notes": "generationStatus=success isFallback=False elapsedMs=250000",
            }]
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    report = runner.build_score_report(
        seed_path=seed_path,
        results_path=results_path,
        output_dir=tmp_path / "out",
        run_id="latency-fail",
        min_preferred_hits=1,
        max_elapsed_ms=180000,
    )

    scenario = report["scenario_reports"][0]
    assert scenario["hardSafetyPassed"] is True
    assert scenario["preferredQualityPassed"] is True
    assert scenario["latencyPassed"] is False
    assert scenario["passed"] is False
    assert report["latency_failed"] == 1
    assert report["recommendation"] == "needs_review"


def test_score_report_can_fail_on_disallowed_fallback(tmp_path):
    seed_path = tmp_path / "seed.jsonl"
    results_path = tmp_path / "results.json"
    seed_path.write_text(
        runner.json.dumps({
            "scenarioId": "fallback-quality-pass",
            "title": "fallback quality pass",
            "input": {"targetSplitLabel": "legs"},
            "expectedHardConstraints": [],
            "preferredPatterns": ["leg_press"],
            "failurePatterns": [],
            "qualityRubric": ["no_unexpected_fallback"],
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    results_path.write_text(
        runner.json.dumps({
            "scenarios": [{
                "scenarioId": "fallback-quality-pass",
                "selectedExerciseIds": ["leg_press"],
                "rationaleText": "leg_press 후보를 선택했습니다.",
                "scoreEvidenceText": "leg_press: boosts=primary_target_match",
                "warnings": [],
                "contractValid": True,
                "hardViolationCount": 0,
                "notes": "generationStatus=success isFallback=True elapsedMs=1000",
            }]
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    report = runner.build_score_report(
        seed_path=seed_path,
        results_path=results_path,
        output_dir=tmp_path / "out",
        run_id="fallback-fail",
        disallow_fallback=True,
    )

    scenario = report["scenario_reports"][0]
    assert scenario["fallbackPassed"] is False
    assert scenario["passed"] is False
    assert report["fallback_failed"] == 1
    assert report["recommendation"] == "needs_review"


def test_parse_scenario_ids_supports_repeated_and_comma_values():
    assert runner.parse_scenario_ids(["a,b", "b", " c "]) == ["a", "b", "c"]


def test_score_report_can_filter_selected_scenarios(tmp_path):
    seed_path = tmp_path / "seed.jsonl"
    results_path = tmp_path / "results.json"
    seed_path.write_text(
        "\n".join([
            runner.json.dumps({
                "scenarioId": "selected",
                "title": "selected",
                "input": {"targetSplitLabel": "legs"},
                "expectedHardConstraints": [],
                "preferredPatterns": ["leg_press"],
                "failurePatterns": [],
                "qualityRubric": [],
            }, ensure_ascii=False),
            runner.json.dumps({
                "scenarioId": "not-selected",
                "title": "not selected",
                "input": {"targetSplitLabel": "push"},
                "expectedHardConstraints": [],
                "preferredPatterns": ["machine_chest_press"],
                "failurePatterns": [],
                "qualityRubric": [],
            }, ensure_ascii=False),
        ]),
        encoding="utf-8",
    )
    results_path.write_text(
        runner.json.dumps({
            "scenarios": [{
                "scenarioId": "selected",
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

    report = runner.build_score_report(
        seed_path=seed_path,
        results_path=results_path,
        output_dir=tmp_path / "out",
        run_id="filtered",
        scenario_ids=["selected"],
    )

    assert report["selected_scenario_ids"] == ["selected"]
    assert report["total_scenarios"] == 1
    assert report["scored_scenarios"] == 1
    assert report["missing_results"] == []
    assert report["recommendation"] == "pass"

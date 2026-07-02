from pathlib import Path

from scripts import run_candidate_quality_seed as candidate_seed


V2_SEED = Path("tests/fixtures/gemma4_routine_quality_eval_seed_v2.jsonl")


def test_candidate_quality_tags_find_lumbar_friendly_leg_options():
    row = {
        "scenarioId": "lower-back-legs-001",
        "preferredPatterns": ["leg_press", "machine", "supported"],
        "failurePatterns": [],
    }
    candidates = [
        {
            "id": "leg_press",
            "name_kr": "레그프레스",
            "name_en": "Leg Press",
            "primary_muscle": "quadriceps",
            "equipment_req": "MACHINE",
            "lumbar_load": "low",
            "score_reasons": ["primary target match"],
        },
        {
            "id": "seated_leg_curl",
            "name_kr": "시티드 레그 컬",
            "name_en": "Seated Leg Curl",
            "primary_muscle": "hamstring",
            "equipment_req": "MACHINE",
            "lumbar_load": "low",
        },
    ]

    tags = candidate_seed.derive_candidate_quality_tags(row, candidates)

    assert "leg_press" in tags
    assert "machine" in tags
    assert "supported" in tags


def test_candidate_quality_tags_detect_korean_neutral_grip_for_surgery_context():
    row = {
        "scenarioId": "post-surgery-caution-001",
        "preferredPatterns": ["neutral_grip_or_supported_press"],
        "failurePatterns": [],
    }
    candidates = [
        {
            "id": "lat_pulldown_neutral_vbar",
            "name_kr": "랫풀다운 머신 무릎 고정 뉴트럴 V바 하이 로우 풀다운",
            "name_en": "Lat Pulldown Neutral V Bar",
            "primary_muscle": "upper-back",
            "equipment_req": "MACHINE",
        }
    ]

    tags = candidate_seed.derive_candidate_quality_tags(row, candidates)

    assert "neutral_grip_or_supported_press" in tags


def test_candidate_quality_tags_detect_beginner_risk_and_progression_reasons():
    row = {
        "scenarioId": "beginner-lower-skill-001",
        "preferredPatterns": ["technique_risk_reason", "progression_path"],
        "failurePatterns": [],
    }
    candidates = [
        {
            "id": "v_squat_machine",
            "name_kr": "브이 스쿼트 머신",
            "primary_muscle": "quadriceps",
            "equipment_req": "MACHINE",
            "score_reasons": [
                "beginner low technique risk candidate",
                "beginner progression path candidate",
            ],
        }
    ]

    tags = candidate_seed.derive_candidate_quality_tags(row, candidates)

    assert "technique_risk_reason" in tags
    assert "progression_path" in tags


def test_candidate_quality_tags_detect_safe_alternative_candidate_reason():
    row = {
        "scenarioId": "post-surgery-caution-001",
        "preferredPatterns": ["safe_alternative_candidate"],
        "failurePatterns": [],
    }
    candidates = [
        {
            "id": "neutral_machine_row",
            "name_kr": "뉴트럴 머신 로우",
            "primary_muscle": "upper-back",
            "equipment_req": "MACHINE",
            "score_reasons": ["professional clearance safe alternative candidate"],
        }
    ]

    tags = candidate_seed.derive_candidate_quality_tags(row, candidates)

    assert "safe_alternative_candidate" in tags


def test_candidate_report_can_compare_live_quality_gap(tmp_path, monkeypatch):
    def fake_scenario_candidates(*, row, db, top_n):  # noqa: ARG001 - test seam
        req = candidate_seed.request_from_seed(row)
        return req, ["quadriceps"], [
            {
                "id": "leg_press",
                "name_kr": "레그프레스",
                "name_en": "Leg Press",
                "primary_muscle": "quadriceps",
                "equipment_req": "MACHINE",
                "lumbar_load": "low",
                "score": 100,
                "score_reasons": ["primary target match"],
            }
        ], False

    monkeypatch.setattr(candidate_seed, "_scenario_candidates", fake_scenario_candidates)

    seed_path = tmp_path / "seed.jsonl"
    results_path = tmp_path / "results.json"
    seed_path.write_text(
        candidate_seed.json.dumps({
            "scenarioId": "lower-back-legs-001",
            "title": "요추 통증",
            "input": {"targetSplitLabel": "legs", "timeAvailableMin": 60, "currentPainAreas": ["lower-back"]},
            "expectedHardConstraints": [],
            "preferredPatterns": ["leg_press", "machine"],
            "failurePatterns": [],
            "qualityRubric": [],
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    results_path.write_text(
        candidate_seed.json.dumps({
            "scenarios": [{
                "scenarioId": "lower-back-legs-001",
                "selectedExerciseIds": ["leg_curl"],
                "rationaleText": "안전한 후보",
                "scoreEvidenceText": "",
                "warnings": [],
                "contractValid": True,
                "hardViolationCount": 0,
                "notes": "",
            }]
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    report = candidate_seed.build_candidate_report(
        seed_path=seed_path,
        output_dir=tmp_path / "out",
        run_id="candidate-gap-test",
        top_n=8,
        results_path=results_path,
    )

    scenario = report["scenarios"][0]
    assert scenario["scenarioId"] == "lower-back-legs-001"
    assert scenario["coverageStatus"] in {"candidate_supports_quality", "candidate_quality_gap"}
    if scenario["coverageStatus"] == "candidate_supports_quality":
        assert scenario["routingHint"] == "llm_selection_or_rationale_gap"


def test_full_v2_candidate_report_has_all_scenarios(monkeypatch):
    def fake_scenario_candidates(*, row, db, top_n):  # noqa: ARG001 - test seam
        req = candidate_seed.request_from_seed(row)
        return req, ["quadriceps"], [
            {
                "id": "leg_press",
                "name_kr": "레그프레스",
                "primary_muscle": "quadriceps",
                "equipment_req": "MACHINE",
                "score": 100,
                "score_reasons": ["primary target match"],
            }
        ], False

    monkeypatch.setattr(candidate_seed, "_scenario_candidates", fake_scenario_candidates)

    report = candidate_seed.build_candidate_report(
        seed_path=V2_SEED,
        output_dir=Path("tests/evaluation/.artifacts/candidate-quality-seed/test"),
        run_id="unit",
        top_n=6,
    )

    assert report["scenarioCount"] == 12
    assert len(report["scenarios"]) == 12
    assert report["statusCounts"]

import json
import time
from pathlib import Path

import pytest

from engines.schemas import RoutineDraftResponse
from scripts import run_gemma4_quality_seed as seed_runner
from scripts import run_gemma4_quality_seed_live as live


def _seed_row(scenario_id: str) -> dict:
    return {
        "scenarioId": scenario_id,
        "title": f"title {scenario_id}",
        "input": {"targetSplitLabel": "legs", "targetMuscles": ["quadriceps"]},
        "expectedHardConstraints": [],
        "preferredPatterns": [],
        "failurePatterns": [],
        "qualityRubric": [],
    }


def _write_seed(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )


def _routine_response(exercise_id: str = "leg_press") -> RoutineDraftResponse:
    return RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 35,
        "summaryTitle": "test",
        "rationaleSummary": ["machine option"],
        "routineBlocks": [
            {
                "order": 1,
                "exerciseId": exercise_id,
                "exerciseName": "레그프레스",
                "equipmentType": "MACHINE",
                "primaryMuscles": ["quadriceps"],
                "defaultRestSec": 90,
                "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
                "exerciseRationale": "quality seed response",
                "reasons": ["목표 주동근과 일치합니다."],
                "boosts": [],
                "penalties": [],
            }
        ],
        "warnings": [],
    })


class _FakeDb:
    def close(self) -> None:
        return None


def _patch_live_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(live, "SessionLocal", lambda: _FakeDb())
    monkeypatch.setattr(live, "get_user_profile_context", lambda db, user_id: None)
    monkeypatch.setattr(live, "get_recent_sets", lambda db, user_id: [])
    monkeypatch.setattr(live, "generate_smart_routine", lambda req, db, profile=None, recent_sets=None: _routine_response())


def test_request_from_seed_maps_public_shape_to_routine_request():
    row = {
        "scenarioId": "s",
        "input": {
            "targetSplitLabel": "legs",
            "timeAvailableMin": 45,
            "readinessLevel": "low",
            "currentPainAreas": ["lower-back"],
            "currentDoms": [{"bodyPart": "biceps", "level": 2}],
            "unavailableEquipment": ["BARBELL"],
            "mobilityLimits": ["limited_ankle_dorsiflexion"],
            "anthropometrySignals": ["long_femur"],
            "surgeryHistory": [{"bodyPart": "shoulder", "status": "needs_clearance"}],
        },
    }

    req = live.request_from_seed(row)

    assert req.target_split_label == "legs"
    assert req.time_available_min == 45
    assert req.readiness_level == "low"
    assert req.pain_areas[0].body_part == "lower-back"
    assert req.doms_data == {"biceps": 2}
    assert req.equipment == ["BARBELL"]
    assert req.mobility_limits == ["limited_ankle_dorsiflexion"]
    assert req.anthropometry_signals == {"signals": ["long_femur"]}
    assert req.condition_policies == [{
        "policyType": "surgeryHistory",
        "bodyPart": "shoulder",
        "status": "needs_clearance",
        "professionalClearance": False,
    }]


def test_live_result_row_is_score_file_compatible():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 40,
        "summaryTitle": "test",
        "rationaleSummary": ["lower-back context"],
        "routineBlocks": [
            {
                "order": 1,
                "exerciseId": "barbell_deadlift",
                "exerciseName": "Barbell Deadlift",
                "primaryMuscles": ["hamstring"],
                "defaultRestSec": 120,
                "prescription": [{"setIndex": 1, "targetReps": 5, "targetRestSec": 120}],
                "exerciseRationale": "high effect hinge",
                "reasons": ["요추 부하 조건을 고려한 후보입니다."],
                "boosts": [
                    {
                        "ruleCode": "primary_target_match",
                        "score": 20,
                        "reason": "목표 주동근과 일치합니다.",
                    }
                ],
                "penalties": [
                    {
                        "ruleCode": "lumbar_load_constraint_penalty",
                        "constraintCode": "lumbar_load",
                        "profileSignalCode": "pain",
                        "score": -35,
                        "reason": "허리 제약 조건에 따라 감점되었습니다.",
                    }
                ],
            }
        ],
        "warnings": [],
    })
    row = {
        "scenarioId": "lower-back-legs-001",
        "title": "요추 통증",
        "input": {"currentPainAreas": ["lower-back"]},
        "failurePatterns": ["barbell_deadlift_as_main"],
    }

    result = live.result_row(row, response, 123)

    assert result["scenarioId"] == "lower-back-legs-001"
    assert result["selectedExerciseIds"] == ["barbell_deadlift"]
    assert result["contractValid"] is True
    assert result["hardViolationCount"] == 1
    assert "barbell_deadlift_as_main" in result["notes"]
    assert "scoreEvidenceText" in result
    assert "primary_target_match" in result["scoreEvidenceText"]
    assert "lumbar_load_constraint_penalty" in result["scoreEvidenceText"]
    json.dumps({"scenarios": [result]}, ensure_ascii=False)


def test_live_result_row_exposes_preferred_quality_tags_for_score_file():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 35,
        "summaryTitle": "test",
        "rationaleSummary": ["허리 부담을 고려해 머신 하체 운동을 우선했습니다."],
        "routineBlocks": [
            {
                "order": 1,
                "exerciseId": "leg_press",
                "exerciseName": "레그프레스",
                "equipmentType": "MACHINE",
                "primaryMuscles": ["quadriceps"],
                "defaultRestSec": 90,
                "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
                "exerciseRationale": "lower-back 맥락에서 machine 후보를 선택했습니다.",
                "reasons": ["허리 제약 조건에서 더 안정적인 후보입니다."],
                "boosts": [
                    {
                        "ruleCode": "primary_target_match",
                        "score": 20,
                        "reason": "목표 주동근과 일치합니다.",
                    }
                ],
                "penalties": [],
            }
        ],
        "warnings": [],
    })
    seed_row = {
        "scenarioId": "lower-back-legs-001",
        "title": "요추 통증",
        "input": {"currentPainAreas": ["lower-back"]},
        "expectedHardConstraints": [],
        "preferredPatterns": ["leg_press", "machine"],
        "failurePatterns": [],
        "qualityRubric": [],
    }

    result = live.result_row(seed_row, response, 123)
    scored = seed_runner.score_result(seed_row, result, min_preferred_hits=1)

    assert "leg_press" in result["notes"]
    assert "machine" in result["notes"]
    assert scored["preferredQualityPassed"] is True
    assert scored["preferredHits"] == ["leg_press", "machine"]


def test_live_payload_rejects_forbidden_tokens(tmp_path):
    payload = {
        "runId": "x",
        "mode": "live-results",
        "seedPath": "seed",
        "scenarioCount": 1,
        "failedScenarioCount": 0,
        "failures": [],
        "scenarios": [{
            "scenarioId": "s",
            "selectedExerciseIds": [],
            "rationaleText": "",
            "warnings": [],
            "contractValid": True,
            "hardViolationCount": 0,
            "notes": "safe",
        }],
    }

    live.assert_no_forbidden_tokens(payload)


def test_scenario_timeout_raises_before_long_sleep():
    started = time.perf_counter()

    with pytest.raises(live.ScenarioTimeoutError):
        with live.scenario_timeout(0.01):
            time.sleep(1)

    assert time.perf_counter() - started < 0.5


def test_parse_scenario_ids_accepts_repeat_and_comma_values():
    assert live.parse_scenario_ids(["a,b", "b", " c "]) == ["a", "b", "c"]


def test_filter_seed_rows_keeps_seed_order_and_rejects_unknown_id():
    rows = [_seed_row("s1"), _seed_row("s2"), _seed_row("s3")]

    filtered = live.filter_seed_rows(rows, ["s3", "s1"])

    assert [row["scenarioId"] for row in filtered] == ["s1", "s3"]
    with pytest.raises(ValueError, match="scenarioId not found"):
        live.filter_seed_rows(rows, ["missing"])


def test_run_live_writes_partial_result_after_each_scenario(tmp_path, monkeypatch):
    _patch_live_dependencies(monkeypatch)
    seed_path = tmp_path / "seed.jsonl"
    output_path = tmp_path / "run" / "gemma4-quality-seed-results-filled.json"
    _write_seed(seed_path, [_seed_row("s1"), _seed_row("s2")])

    payload = live.run_live(
        seed_path,
        "unit-run",
        output_path=output_path,
        scenario_timeout_sec=0,
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["completedScenarioCount"] == 2
    assert saved["completedScenarioIds"] == ["s1", "s2"]
    assert saved["scenarioCount"] == 2
    assert len(saved["scenarios"]) == 2


def test_run_live_filters_scenario_ids_and_resume_skips_completed(tmp_path, monkeypatch):
    _patch_live_dependencies(monkeypatch)
    seed_path = tmp_path / "seed.jsonl"
    output_path = tmp_path / "run" / "gemma4-quality-seed-results-filled.json"
    _write_seed(seed_path, [_seed_row("s1"), _seed_row("s2")])
    existing = live.build_payload(
        run_id="unit-run",
        seed_path=seed_path,
        rows=[_seed_row("s1"), _seed_row("s2")],
        scenarios=[{"scenarioId": "s1", "title": "title s1", "selectedExerciseIds": []}],
        failures=[],
    )
    live.write_payload_snapshot(output_path, existing)

    payload = live.run_live(
        seed_path,
        "unit-run",
        output_path=output_path,
        scenario_ids=["s1", "s2"],
        scenario_timeout_sec=0,
        resume=True,
    )

    assert payload["completedScenarioIds"] == ["s1", "s2"]
    assert [row["scenarioId"] for row in payload["scenarios"]] == ["s1", "s2"]


def test_run_live_can_cool_down_between_scenarios(tmp_path, monkeypatch):
    _patch_live_dependencies(monkeypatch)
    slept: list[float] = []
    monkeypatch.setattr(live.time, "sleep", lambda seconds: slept.append(seconds))
    seed_path = tmp_path / "seed.jsonl"
    output_path = tmp_path / "run" / "gemma4-quality-seed-results-filled.json"
    _write_seed(seed_path, [_seed_row("s1"), _seed_row("s2"), _seed_row("s3")])

    live.run_live(
        seed_path,
        "unit-run",
        output_path=output_path,
        scenario_timeout_sec=0,
        cooldown_sec=1.5,
    )

    assert slept == [1.5, 1.5]

import json
import os
import sys
import time
from pathlib import Path

import pytest

from engines.schemas import AdjunctDecision, RoutineDraftResponse
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


@pytest.mark.parametrize("field", ["severity", "level"])
def test_pain_entries_rejects_legacy_injury_intensity_key(field):
    with pytest.raises(ValueError):
        live.pain_entries([{"bodyPart": "chest", field: None}])


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


def _quality_response(
    *,
    total_time: int,
    blocks: list[tuple[str, str, str, str]],
) -> RoutineDraftResponse:
    return RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": total_time,
        "summaryTitle": "quality gate",
        "rationaleSummary": ["요청 조건에 맞춰 운동 순서와 분량을 조정했습니다."],
        "routineBlocks": [
            {
                "order": index,
                "exerciseId": exercise_id,
                "exerciseName": exercise_name,
                "movementType": movement_type,
                "primaryMuscles": [primary_muscle],
                "defaultRestSec": 75,
                "prescription": [{
                    "setIndex": 1,
                    "targetReps": 10,
                    "targetRestSec": 75,
                }],
                "exerciseRationale": "목표 부위와 세션 구성을 고려한 운동입니다.",
                "reasons": [],
                "boosts": [],
                "penalties": [],
            }
            for index, (exercise_id, exercise_name, primary_muscle, movement_type)
            in enumerate(blocks, 1)
        ],
        "warnings": [],
    })


def test_rationale_summary_keeps_every_selected_exercise_for_human_review():
    response = _quality_response(
        total_time=45,
        blocks=[
            ("one", "운동 1", "chest", "COMPOUND"),
            ("two", "운동 2", "upper-back", "COMPOUND"),
            ("three", "운동 3", "quadriceps", "COMPOUND"),
            ("four", "운동 4", "biceps", "ISOLATION"),
            ("five", "운동 5", "triceps", "ISOLATION"),
        ],
    )

    summary = live.summarize_rationale(response)

    assert "운동 1:" in summary
    assert "운동 5:" in summary


def test_output_quality_flags_reject_normal_session_time_and_count_underfill():
    response = _quality_response(
        total_time=30,
        blocks=[
            ("bench_press", "바벨 벤치프레스", "chest", "COMPOUND"),
            ("machine_fly", "머신 플라이", "chest", "ISOLATION"),
            ("lateral_raise", "덤벨 레터럴 레이즈", "front-deltoids", "ISOLATION"),
            ("pushdown", "케이블 푸시다운", "triceps", "ISOLATION"),
        ],
    )

    flags = live.derived_output_quality_flags({
        "input": {
            "timeAvailableMin": 75,
            "readinessLevel": "normal",
            "currentPainAreas": [],
            "currentDoms": [],
        },
    }, response)

    assert "time_budget_underutilized" in flags
    assert "exercise_count_underfilled" in flags
    assert "duplicate_movement_family" not in flags


def test_output_quality_flags_reject_successful_accessory_only_push():
    response = _quality_response(
        total_time=17,
        blocks=[
            ("pushdown", "케이블 푸시다운", "triceps", "ISOLATION"),
            ("triceps_extension", "덤벨 라잉 트라이셉스 익스텐션", "triceps", "ISOLATION"),
        ],
    )

    flags = live.derived_output_quality_flags({
        "input": {
            "targetSplitLabel": "push",
            "timeAvailableMin": 45,
            "readinessLevel": "normal",
            "currentPainAreas": ["front-deltoids"],
            "currentDoms": [],
        },
    }, response)

    assert "required_main_pattern_missing" in flags


def test_output_quality_flags_accept_explicit_single_target_partial_session_when_injury_limits_scope():
    response = _quality_response(
        total_time=12,
        blocks=[
            ("machine_pushdown", "머신 푸시다운", "triceps", "ISOLATION"),
        ],
    )
    response.warnings = ["사용 가능한 장비를 추가하거나 목표 부위를 바꿔 다시 생성해 주세요."]

    flags = live.derived_output_quality_flags({
        "limitedScopeUnderfillAllowed": True,
        "input": {
            "targetSplitLabel": "custom",
            "targetMuscles": ["triceps"],
            "timeAvailableMin": 35,
            "readinessLevel": "normal",
            "currentPainAreas": [{"bodyPart": "front-deltoids"}],
            "currentDoms": [],
        },
    }, response)

    assert "required_main_pattern_missing" not in flags
    assert "exercise_count_underfilled" not in flags
    assert "time_budget_underutilized" not in flags


def test_output_quality_flags_do_not_accept_partial_session_without_warning():
    response = _quality_response(
        total_time=12,
        blocks=[
            ("machine_pushdown", "머신 푸시다운", "triceps", "ISOLATION"),
        ],
    )

    flags = live.derived_output_quality_flags({
        "limitedScopeUnderfillAllowed": True,
        "input": {
            "targetSplitLabel": "custom",
            "targetMuscles": ["triceps"],
            "timeAvailableMin": 35,
            "readinessLevel": "normal",
            "currentPainAreas": [{"bodyPart": "front-deltoids"}],
            "currentDoms": [],
        },
    }, response)

    assert "exercise_count_underfilled" in flags
    assert "time_budget_underutilized" in flags


def test_output_quality_flags_reject_equipment_outside_available_whitelist():
    response = _quality_response(
        total_time=30,
        blocks=[("band_press", "밴드 체스트 프레스", "chest", "COMPOUND")],
    )
    response.routine_blocks[0].equipment_type = "BAND"

    flags = live.derived_output_quality_flags({
        "input": {
            "targetSplitLabel": "push",
            "timeAvailableMin": 30,
            "readinessLevel": "normal",
            "availableEquipment": ["BODYWEIGHT", "DUMBBELL"],
            "currentPainAreas": [],
            "currentDoms": [],
        },
    }, response)

    assert "available_equipment_whitelist_violation" in flags


def test_output_quality_flags_treat_bodyweight_as_implicitly_available():
    response = _quality_response(
        total_time=30,
        blocks=[("calf_raise", "맨몸 카프 레이즈", "calves", "ISOLATION")],
    )
    response.routine_blocks[0].equipment_type = "BODYWEIGHT"

    flags = live.derived_output_quality_flags({
        "input": {
            "targetSplitLabel": "legs",
            "timeAvailableMin": 30,
            "readinessLevel": "normal",
            "availableEquipment": ["DUMBBELL", "MACHINE"],
            "currentPainAreas": [],
            "currentDoms": [],
        },
    }, response)

    assert "available_equipment_whitelist_violation" not in flags


def test_neutral_grip_press_tag_rejects_triceps_isolation_name_collision():
    response = _quality_response(
        total_time=15,
        blocks=[
            (
                "neutral_triceps_extension",
                "덤벨 라잉 뉴트럴 트라이셉스 익스텐션",
                "triceps",
                "ISOLATION",
            ),
        ],
    )
    row = {"scenarioId": "shoulder-push-001", "input": {}}

    assert "neutral_grip_press" not in live.derived_preferred_tags(row, response)


def test_neutral_grip_press_tag_accepts_compound_chest_press():
    response = _quality_response(
        total_time=20,
        blocks=[
            (
                "neutral_dumbbell_chest_press",
                "덤벨 뉴트럴 그립 체스트 프레스",
                "chest",
                "COMPOUND",
            ),
        ],
    )
    row = {"scenarioId": "shoulder-push-001", "input": {}}

    assert "neutral_grip_press" in live.derived_preferred_tags(row, response)


def test_output_quality_flags_accept_proven_candidate_exhaustion_only_for_time_floor():
    response = _quality_response(
        total_time=35,
        blocks=[
            ("lat_pulldown", "랫풀다운", "upper-back", "COMPOUND"),
            ("dumbbell_row", "덤벨 로우", "upper-back", "COMPOUND"),
            ("reverse_pec_deck", "리버스 펙덱", "back-deltoids", "ISOLATION"),
            ("hammer_curl", "해머 컬", "biceps", "ISOLATION"),
            ("face_pull", "페이스 풀", "back-deltoids", "ISOLATION"),
        ],
    )
    row = {
        "input": {
            "timeAvailableMin": 60,
            "targetSplitLabel": "pull",
            "readinessLevel": "normal",
            "sessionStructure": "standard",
            "currentPainAreas": [],
            "currentDoms": [],
        },
    }

    without_evidence = live.derived_output_quality_flags(row, response)
    with_evidence = live.derived_output_quality_flags(
        row,
        response,
        telemetry_summary={
            "productive_fill_exhausted": True,
            "productive_fill_exhaustion_reason": "candidate_policy_exhausted",
        },
    )

    assert "time_budget_underutilized" in without_evidence
    assert "time_budget_underutilized" not in with_evidence


def test_output_quality_flags_apply_recovery_limits_only_to_requested_targets():
    response = _quality_response(
        total_time=30,
        blocks=[
            ("bench_press", "바벨 벤치프레스", "chest", "COMPOUND"),
            ("machine_fly", "머신 플라이", "chest", "ISOLATION"),
            ("pushdown", "케이블 푸시다운", "triceps", "ISOLATION"),
        ],
    )
    base_input = {
        "targetSplitLabel": "push",
        "timeAvailableMin": 75,
        "readinessLevel": "normal",
        "sessionStructure": "standard",
        "currentPainAreas": [],
    }

    off_target_flags = live.derived_output_quality_flags({
        "input": {**base_input, "currentDoms": {"calves": 1}},
    }, response)
    target_flags = live.derived_output_quality_flags({
        "input": {**base_input, "currentDoms": {"chest": 1}},
    }, response)

    assert "time_budget_underutilized" in off_target_flags
    assert "exercise_count_underfilled" in off_target_flags
    assert "time_budget_underutilized" not in target_flags
    assert "exercise_count_underfilled" not in target_flags


def test_output_quality_flags_accept_productive_distinct_normal_session():
    response = _quality_response(
        total_time=52,
        blocks=[
            ("bench_press", "바벨 벤치프레스", "chest", "COMPOUND"),
            ("machine_fly", "머신 플라이", "chest", "ISOLATION"),
            ("lateral_raise", "덤벨 레터럴 레이즈", "front-deltoids", "ISOLATION"),
            ("pushdown", "케이블 푸시다운", "triceps", "ISOLATION"),
            ("overhead_extension", "덤벨 오버헤드 익스텐션", "triceps", "ISOLATION"),
            ("push_up", "푸시업", "chest", "COMPOUND"),
            ("doorway_stretch", "도어웨이 가슴 스트레칭", "chest", "STATIC"),
        ],
    )

    flags = live.derived_output_quality_flags({
        "input": {
            "timeAvailableMin": 75,
            "readinessLevel": "normal",
            "currentPainAreas": [],
            "currentDoms": [],
        },
    }, response)

    assert flags == []


def test_output_quality_flags_accept_one_below_target_when_blueprint_minimum_is_met():
    response = _quality_response(
        total_time=42,
        blocks=[
            ("bench_press", "바벨 벤치프레스", "chest", "COMPOUND"),
            ("incline_press", "덤벨 인클라인 프레스", "chest", "COMPOUND"),
            ("machine_fly", "머신 플라이", "chest", "ISOLATION"),
            ("lateral_raise", "덤벨 레터럴 레이즈", "front-deltoids", "ISOLATION"),
            ("pushdown", "케이블 푸시다운", "triceps", "ISOLATION"),
        ],
    )

    flags = live.derived_output_quality_flags({
        "scenarioId": "minimum-count-accepted-001",
        "input": {
            "targetSplitLabel": "push",
            "timeAvailableMin": 60,
            "goal": "hypertrophy",
            "readinessLevel": "normal",
            "currentPainAreas": [],
            "currentDoms": [],
        },
    }, response)

    assert "exercise_count_underfilled" not in flags


def test_output_quality_flags_accept_productive_session_below_target_exercise_count():
    response = _quality_response(
        total_time=40,
        blocks=[
            ("goblet_squat", "덤벨 고블릿 스쿼트", "quadriceps", "COMPOUND"),
            ("leg_press", "레그 프레스", "quadriceps", "COMPOUND"),
            ("leg_curl", "머신 레그 컬", "hamstring", "ISOLATION"),
            ("leg_extension", "머신 레그 익스텐션", "quadriceps", "ISOLATION"),
        ],
    )

    flags = live.derived_output_quality_flags({
        "input": {
            "targetSplitLabel": "legs",
            "timeAvailableMin": 50,
            "goal": "hypertrophy",
            "readinessLevel": "normal",
            "currentPainAreas": [],
            "currentDoms": [],
        },
    }, response)

    assert "time_budget_underutilized" not in flags
    assert "exercise_count_underfilled" not in flags


def test_output_quality_flags_count_only_main_blocks_when_adjuncts_are_suppressed():
    response = _quality_response(
        total_time=55,
        blocks=[
            ("bench", "벤치프레스", "chest", "COMPOUND"),
            ("row", "시티드 로우", "upper-back", "COMPOUND"),
            ("incline", "인클라인 프레스", "chest", "COMPOUND"),
            ("pulldown", "랫풀다운", "upper-back", "COMPOUND"),
            ("curl", "케이블 컬", "biceps", "ISOLATION"),
            ("pushdown", "케이블 푸시다운", "triceps", "ISOLATION"),
        ],
    )
    response.adjunct_decisions = [
        AdjunctDecision(
            option="mobility_warmup",
            status="suppressed",
            reason_code="insufficientTimeAfterMainMinimum",
            slot_id="adjunct_mobility_warmup",
        ),
        AdjunctDecision(
            option="prehab_accessory",
            status="suppressed",
            reason_code="insufficientTimeAfterMainMinimum",
            slot_id="adjunct_prehab_accessory",
        ),
    ]
    row = {
        "input": {
            "targetSplitLabel": "upper",
            "timeAvailableMin": 75,
            "readinessLevel": "normal",
            "goal": "hypertrophy",
            "sessionStructure": "standard",
            "adjunctOptions": ["mobility_warmup", "prehab_accessory"],
            "currentPainAreas": [],
            "currentDoms": [],
        }
    }

    flags = live.derived_output_quality_flags(row, response)

    assert "exercise_count_underfilled" not in flags


def test_output_quality_flags_inspect_fallback_main_underfill():
    response = _quality_response(
        total_time=50,
        blocks=[
            ("bench", "벤치프레스", "chest", "COMPOUND"),
            ("row", "시티드 로우", "upper-back", "COMPOUND"),
            ("incline", "인클라인 프레스", "chest", "COMPOUND"),
            ("pulldown", "랫풀다운", "upper-back", "COMPOUND"),
            ("curl", "케이블 컬", "biceps", "ISOLATION"),
        ],
    )
    response.generation_status = "fallback"
    response.status_reason_code = "networkError"
    response.is_fallback = True

    flags = live.derived_output_quality_flags({
        "input": {
            "targetSplitLabel": "upper",
            "timeAvailableMin": 75,
            "readinessLevel": "normal",
            "goal": "hypertrophy",
            "sessionStructure": "standard",
            "adjunctOptions": [],
            "currentPainAreas": [],
            "currentDoms": [],
        }
    }, response)

    assert "exercise_count_underfilled" in flags


def test_output_quality_flags_detect_coarse_pattern_over_cap():
    response = _quality_response(
        total_time=45,
        blocks=[
            ("flat_press", "바벨 플랫 벤치프레스", "chest", "COMPOUND"),
            ("incline_press", "덤벨 인클라인 벤치프레스", "chest", "COMPOUND"),
            ("decline_press", "머신 디클라인 벤치프레스", "chest", "COMPOUND"),
            ("machine_fly", "머신 플라이", "chest", "ISOLATION"),
            ("pushdown", "케이블 푸시다운", "triceps", "ISOLATION"),
        ],
    )

    flags = live.derived_output_quality_flags({
        "scenarioId": "coarse-pattern-over-cap-001",
        "input": {
            "targetSplitLabel": "push",
            "timeAvailableMin": 60,
            "goal": "hypertrophy",
            "readinessLevel": "normal",
        },
    }, response)

    assert "coarse_pattern_over_cap:horizontal_press" in flags


def test_output_quality_flags_require_scope_change_warning_for_pain_target_overlap():
    response = _quality_response(
        total_time=30,
        blocks=[
            ("pushdown", "케이블 푸시다운", "triceps", "ISOLATION"),
            ("lateral_raise", "덤벨 레터럴 레이즈", "front-deltoids", "ISOLATION"),
        ],
    )
    row = {
        "scenarioId": "pain-target-scope-warning-001",
        "input": {
            "targetSplitLabel": "push",
            "timeAvailableMin": 45,
            "goal": "hypertrophy",
            "readinessLevel": "normal",
            "currentPainAreas": [{"bodyPart": "chest"}],
        },
    }

    flags = live.derived_output_quality_flags(row, response)
    assert "missing_pain_target_scope_change_warning" in flags

    response.warnings = [
        "입력한 통증/부상 부위(가슴)와 직접 겹치는 목표 운동은 후보에서 제외했습니다."
    ]
    flags = live.derived_output_quality_flags(row, response)
    assert "missing_pain_target_scope_change_warning" not in flags


def test_output_quality_flags_do_not_apply_standard_duration_floor_to_full_density_session():
    response = _quality_response(
        total_time=45,
        blocks=[
            ("bench_press", "바벨 벤치프레스", "chest", "COMPOUND"),
            ("incline_press", "덤벨 인클라인 프레스", "chest", "COMPOUND"),
            ("machine_fly", "머신 플라이", "chest", "ISOLATION"),
            ("pushdown", "케이블 푸시다운", "triceps", "ISOLATION"),
            ("lateral_raise", "덤벨 레터럴 레이즈", "front-deltoids", "ISOLATION"),
            ("jm_press", "머신 JM 프레스", "triceps", "ISOLATION"),
            ("front_raise", "케이블 프론트 레이즈", "front-deltoids", "ISOLATION"),
            ("tate_press", "덤벨 테이트 프레스", "triceps", "ISOLATION"),
        ],
    )

    flags = live.derived_output_quality_flags({
        "input": {
            "timeAvailableMin": 120,
            "readinessLevel": "normal",
            "sessionStructure": "density",
            "currentPainAreas": [],
            "currentDoms": [],
        },
    }, response)

    assert "time_budget_underutilized" not in flags
    assert "exercise_count_underfilled" not in flags


def test_output_quality_flags_use_strength_specific_exercise_count():
    response = _quality_response(
        total_time=48,
        blocks=[
            ("back_squat", "바벨 백스쿼트", "quadriceps", "COMPOUND"),
            ("romanian_deadlift", "루마니안 데드리프트", "hamstring", "COMPOUND"),
            ("leg_curl", "머신 레그컬", "hamstring", "ISOLATION"),
            ("calf_raise", "머신 카프레이즈", "calves", "ISOLATION"),
        ],
    )

    flags = live.derived_output_quality_flags({
        "input": {
            "timeAvailableMin": 60,
            "goal": "strength",
            "readinessLevel": "normal",
            "currentPainAreas": [],
            "currentDoms": [],
        },
    }, response)

    assert "exercise_count_underfilled" not in flags
    assert "time_budget_underutilized" not in flags


def test_output_quality_flags_detect_equipment_variant_duplicate_family():
    response = _quality_response(
        total_time=40,
        blocks=[
            ("machine_pushdown", "머신 푸시다운", "triceps", "ISOLATION"),
            ("cable_pushdown", "케이블 푸시다운", "triceps", "ISOLATION"),
        ],
    )

    flags = live.derived_output_quality_flags({
        "input": {
            "timeAvailableMin": 45,
            "readinessLevel": "normal",
        },
    }, response)

    assert "duplicate_movement_family" in flags


def test_score_result_fails_when_live_output_quality_flag_is_present():
    seed_row = _seed_row("output-quality-fail-001")
    result = {
        "scenarioId": seed_row["scenarioId"],
        "selectedExerciseIds": ["leg_press"],
        "rationaleText": "목표 부위를 고려한 루틴입니다.",
        "scoreEvidenceText": "",
        "warnings": [],
        "contractValid": True,
        "hardViolationCount": 0,
        "outputQualityFlags": ["time_budget_underutilized"],
        "notes": "generationStatus=success isFallback=False elapsedMs=1000",
    }

    scored = seed_runner.score_result(seed_row, result)

    assert scored["hardSafetyPassed"] is True
    assert scored["outputQualityPassed"] is False
    assert scored["outputQualityFlags"] == ["time_budget_underutilized"]
    assert scored["passed"] is False


def test_score_result_fails_when_live_language_quality_flag_is_present():
    seed_row = _seed_row("language-quality-fail-001")
    result = {
        "scenarioId": seed_row["scenarioId"],
        "selectedExerciseIds": ["leg_press"],
        "rationaleText": "조건과 무관한 일반 설명입니다.",
        "scoreEvidenceText": "",
        "warnings": [],
        "contractValid": True,
        "hardViolationCount": 0,
        "languageQualityFlags": ["generic_rationale"],
        "notes": "generationStatus=success isFallback=False elapsedMs=1000",
    }

    scored = seed_runner.score_result(seed_row, result)

    assert scored["hardSafetyPassed"] is True
    assert scored["languageQualityPassed"] is False
    assert scored["languageQualityFlags"] == ["generic_rationale"]
    assert scored["passed"] is False


def test_holdout_rationale_tag_uses_only_user_visible_text():
    response = _routine_response()
    response.routine_blocks[0].reasons = ["limited_ankle_dorsiflexion penalty"]
    row = _seed_row("holdout-ankle-limited-legs-001")

    assert "ankle_mobility_reason" not in live.derived_preferred_tags(row, response)

    response.rationale_summary = ["발뒤꿈치를 붙인 깊은 앉기 동작의 어려움을 고려했습니다."]

    assert "ankle_mobility_reason" in live.derived_preferred_tags(row, response)


@pytest.mark.parametrize(
    ("profile_signal", "rationale", "expected_tag"),
    [
        (
            "hip_hinge_difficult",
            "엉덩이를 접어 숙이는 동작의 어려움을 고려했습니다.",
            "hip_hinge_difficulty_reason",
        ),
        (
            "overhead_reach_difficult",
            "팔을 머리 위로 올리는 동작의 어려움을 고려했습니다.",
            "overhead_reach_difficulty_reason",
        ),
        (
            "loaded_wrist_extension_difficult",
            "손목을 뒤로 젖혀 부하를 버티는 동작의 어려움을 고려했습니다.",
            "wrist_extension_difficulty_reason",
        ),
        (
            "self_reported_long_arms",
            "팔이 유난히 긴 편이라는 입력을 고려했습니다.",
            "long_arm_profile_reason",
        ),
        (
            "self_reported_long_thighs",
            "허벅지 구간이 유난히 긴 편이라는 입력을 고려했습니다.",
            "long_thigh_profile_reason",
        ),
    ],
)
def test_profile_signal_tags_require_user_visible_rationale(
    profile_signal,
    rationale,
    expected_tag,
):
    response = _routine_response()
    row = _seed_row(f"profile-v2-{profile_signal}")
    row["input"]["profileSignals"] = [profile_signal]

    assert expected_tag not in live.derived_preferred_tags(row, response)

    response.rationale_summary = [rationale]

    assert expected_tag in live.derived_preferred_tags(row, response)


def test_removed_long_lower_leg_signal_is_rejected_before_quality_evaluation():
    row = _seed_row("profile-v2-long-lower-leg")
    row["input"]["profileSignals"] = ["self_reported_long_lower_legs"]

    with pytest.raises(ValueError, match="unknown profile constraint signals"):
        live.request_from_seed(row)


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
            "availableEquipment": ["BODYWEIGHT", "DUMBBELL"],
            "trainingStyle": "yoga_pilates",
            "sessionStructure": "superset",
            "preferredExerciseIds": ["V24ROW-003058"],
            "unpreferredExerciseIds": ["V24ROW-003066"],
            "profileSignals": [
                "overhead_reach_difficult",
                "self_reported_long_arms",
            ],
            "mobilityLimits": ["limited_ankle_dorsiflexion"],
            "anthropometrySignals": ["long_femur"],
            "conditionPolicies": [{
                "conditionCode": "lumbar_herniation",
                "bodyPart": "lower-back",
                "status": "history",
            }],
            "injuryConditions": [{
                "areaCode": "ankle",
                "status": "managedWithRestrictions",
                "restrictedMovementPatterns": ["loadedPlantarFlexion"],
            }],
            "surgeryHistory": [{"bodyPart": "shoulder", "status": "needs_clearance"}],
        },
    }

    req = live.request_from_seed(row)

    assert req.target_split_label == "legs"
    assert req.time_available_min == 45
    assert req.readiness_level == "low"
    assert req.goal == "hypertrophy"
    assert req.experience_level == "intermediate"
    assert req.pain_areas[0].body_part == "lower-back"
    assert req.doms_data == {"biceps": 2}
    assert req.equipment == ["BARBELL"]
    assert req.available_equipment == ["BODYWEIGHT", "DUMBBELL"]
    assert req.training_style == "yoga_pilates"
    assert req.session_structure == "superset"
    assert req.preferred_exercise_ids == ["V24ROW-003058"]
    assert req.unpreferred_exercise_ids == ["V24ROW-003066"]
    assert req.profile_signals == [
        "overhead_reach_difficult",
        "self_reported_long_arms",
    ]
    assert req.mobility_limits == [
        "limited_ankle_dorsiflexion",
        "limited_shoulder_flexion",
    ]
    assert req.injury_conditions[0].area_code == "ankle"
    assert req.injury_conditions[0].status == "managedWithRestrictions"
    assert req.injury_conditions[0].restricted_movement_patterns == [
        "loadedPlantarFlexion"
    ]
    assert req.anthropometry_signals == {"signals": ["long_femur", "long_arm"]}
    assert req.condition_policies == [{
        "conditionCode": "lumbar_herniation",
        "bodyPart": "lower-back",
        "status": "history",
    }, {
        "policyType": "surgeryHistory",
        "bodyPart": "shoulder",
        "status": "needs_clearance",
        "professionalClearance": False,
    }]


def test_request_from_seed_preserves_explicit_goal_and_experience_level():
    row = {
        "scenarioId": "explicit-profile-values",
        "input": {
            "targetSplitLabel": "legs",
            "timeAvailableMin": 45,
            "goal": "strength",
            "experienceLevel": "advanced",
        },
    }

    req = live.request_from_seed(row)

    assert req.goal == "strength"
    assert req.experience_level == "advanced"


def test_output_quality_flags_reject_selected_managed_injury_overlap(monkeypatch):
    row = _seed_row("managed-ankle-overlap")
    row["input"]["injuryConditions"] = [{
        "areaCode": "ankle",
        "status": "managedWithRestrictions",
        "restrictedMovementPatterns": ["loadedPlantarFlexion"],
    }]
    response = _routine_response("calf_raise")
    monkeypatch.setattr(
        live,
        "_exercise_policy_candidate",
        lambda exercise_id: {
            "id": exercise_id,
            "ankle_region_load_level": "high",
            "deep_dorsiflexion_demand": "low",
            "plantar_flexion_load": "high",
            "single_leg_stability_demand": "low",
            "impact_demand": "low",
        },
    )

    flags = live.derived_output_quality_flags(row, response)

    assert "managed_injury_restricted_candidate_selected" in flags


def test_preferred_tags_record_safe_selection_and_user_context(monkeypatch):
    row = _seed_row("managed-ankle-safe")
    row["input"]["injuryConditions"] = [{
        "areaCode": "ankle",
        "status": "managedWithRestrictions",
        "restrictedMovementPatterns": ["loadedPlantarFlexion"],
    }]
    response = _routine_response("leg_extension")
    response.warnings = [
        "관리 중인 발목·발 상태에 따라 제한 동작을 후보 단계에서 제외했습니다."
    ]
    monkeypatch.setattr(
        live,
        "_exercise_policy_candidate",
        lambda exercise_id: {
            "id": exercise_id,
            "ankle_region_load_level": "low",
            "deep_dorsiflexion_demand": "low",
            "plantar_flexion_load": "low",
            "single_leg_stability_demand": "low",
            "impact_demand": "low",
        },
    )

    tags = live.derived_preferred_tags(row, response)

    assert "managed_injury_selection_safe" in tags
    assert "managed_injury_context_reason" in tags


def test_language_quality_uses_public_intermediate_default_when_seed_omits_level():
    response = _routine_response()
    response.rationale_summary = ["초보자 수준에 맞춰 안정적으로 구성했습니다."]
    row = _seed_row("default-intermediate-rationale")
    row["input"].pop("experienceLevel", None)

    flags = live.derived_language_quality_flags(response, row)

    assert "experience_level_contradiction" in flags


def test_language_quality_rejects_internal_selection_terms_count_mismatch_and_duplicate_equipment():
    response = _routine_response()
    response.rationale_summary = [
        "필수적인 6가지 동작을 포함했습니다.",
        "이 슬롯은 동일 ID라 다음 후보로 대체합니다.",
        "스미스 머신(스미스 머신)을 사용합니다.",
    ]

    flags = live.derived_language_quality_flags(response, _seed_row("public-text-contract"))

    assert "internal_implementation_term_in_user_text" in flags
    assert "exercise_count_claim_mismatch" in flags
    assert "equipment_name_duplication" in flags


def test_apply_local_llm_defaults_uses_long_enough_generation_budget(monkeypatch):
    for key in (
        "APP_ENV",
        "LLM_PROVIDER",
        "LOCAL_LLM_MODEL",
        "OLLAMA_BASE_URL",
        "ENABLE_LOCAL_RAW_JSON_INVOKE",
        "LOCAL_LLM_NUM_PREDICT",
        "LOCAL_LLM_NUM_CTX",
        "LOCAL_LLM_SEED",
        "LOCAL_LLM_REASONING",
        "ROUTINE_CANDIDATE_POOL_SIZE",
        "ROUTINE_COMPACT_CANDIDATE_PROMPT",
    ):
        monkeypatch.delenv(key, raising=False)

    live.apply_local_llm_defaults()

    assert os.environ["LOCAL_LLM_NUM_PREDICT"] == "4096"
    assert os.environ["LOCAL_LLM_NUM_CTX"] == "8192"
    assert os.environ["ROUTINE_CANDIDATE_POOL_SIZE"] == "12"


def test_generated_style_scenario_gets_style_specific_preferred_tag_from_boost():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 44,
        "summaryTitle": "요가 필라테스 하체",
        "rationaleSummary": ["요가/필라테스 스타일의 정렬과 안정성을 우선했습니다."],
        "routineBlocks": [
            {
                "order": 1,
                "exerciseId": "V24ROW-007381",
                "exerciseName": "필라테스 운다체어 고잉업 프론트",
                "equipmentType": "BODYWEIGHT",
                "primaryMuscles": ["quadriceps"],
                "defaultRestSec": 75,
                "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 75}],
                "exerciseRationale": "필라테스 스타일의 정렬 중심 하체 메인 운동입니다.",
                "reasons": ["요가/필라테스 스타일과 일치합니다."],
                "boosts": [
                    {
                        "ruleCode": "training_style_match_boost",
                        "score": 48,
                        "reason": "yoga_pilates training style match",
                    }
                ],
                "penalties": [],
            }
        ],
        "warnings": [],
    })
    row = {
        "scenarioId": "gen-sens-legs-trainingstyle-yoga-pilates-020",
        "title": "style",
        "input": {"trainingStyle": "yoga_pilates"},
        "expectedHardConstraints": [],
        "preferredPatterns": ["yoga_pilates_style_match"],
        "failurePatterns": [],
        "qualityRubric": [],
    }

    result = live.result_row(row, response, 123)
    scored = seed_runner.score_result(row, result, min_preferred_hits=1)

    assert "yoga_pilates_style_match" in result["notes"]
    assert "yoga_pilates_style_match" in scored["preferredHits"]
    assert scored["preferredQualityPassed"] is True


def test_required_preferred_patterns_are_scored_without_preferred_duplication():
    seed_row = {
        "scenarioId": "gen-sens-legs-trainingstyle-yoga-pilates-024",
        "title": "style required only",
        "input": {"trainingStyle": "yoga_pilates"},
        "expectedHardConstraints": [],
        "preferredPatterns": ["language_quality_ok", "live_success"],
        "requiredPreferredPatterns": ["live_success", "yoga_pilates_style_match"],
        "failurePatterns": [],
        "qualityRubric": [],
    }
    result = {
        "scenarioId": seed_row["scenarioId"],
        "selectedExerciseIds": ["V24ROW-008572"],
        "rationaleText": "요가/필라테스 스타일의 보조 운동을 포함했습니다.",
        "scoreEvidenceText": "",
        "warnings": [],
        "contractValid": True,
        "hardViolationCount": 0,
        "notes": "language_quality_ok live_success yoga_pilates_style_match",
    }

    scored = seed_runner.score_result(seed_row, result, min_preferred_hits=1)

    assert "yoga_pilates_style_match" in scored["preferredHits"]
    assert scored["missingRequiredPreferredPatterns"] == []
    assert scored["preferredQualityPassed"] is True


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
    assert result["generationStatus"] == "success"
    assert result["statusReasonCode"] == "none"
    assert result["isFallback"] is False
    assert result["totalEstimatedTime"] == 40
    assert result["elapsedMs"] == 123
    assert result["exerciseCount"] == 1
    assert result["selectedExerciseIds"] == ["barbell_deadlift"]
    assert result["contractValid"] is True
    assert result["hardViolationCount"] == 1
    assert "barbell_deadlift_as_main" in result["notes"]
    assert "scoreEvidenceText" in result
    assert "primary_target_match" in result["scoreEvidenceText"]
    assert "lumbar_load_constraint_penalty" in result["scoreEvidenceText"]
    json.dumps({"scenarios": [result]}, ensure_ascii=False)


def test_live_result_row_includes_safe_telemetry_summary_without_raw_output():
    response = _routine_response()
    seed_row = _seed_row("telemetry-001")
    telemetry_events = [{
        "fallback_reason": "schemaError",
        "parse_failure_subtype": "missing_required_field",
        "schema_validation_field_names": ["exercises.0.sets"],
        "schema_root_type": "dict",
        "schema_top_level_keys": ["exercises", "summary_title"],
        "schema_exercises_value_type": "str",
        "schema_exercises_nested_keys": [],
        "schema_exercises_list_length": None,
        "schema_first_exercise_item_type": "dict",
        "schema_first_exercise_keys": ["sets"],
        "schema_first_exercise_field_types": {"sets": "dict"},
        "schema_exercise_item_types": ["dict"],
        "schema_exercise_key_union": ["sets"],
        "schema_exercise_field_type_sets": {"sets": ["dict"]},
        "schema_validation_loc_summaries": ["exercises.*.sets"],
        "schema_validation_error_types": ["int_type"],
        "raw_output_recovery_failed_reason": "no_valid_json_object",
        "raw_model_output": "do not store this",
        "prompt": "do not store this",
    }]

    result = live.result_row(seed_row, response, 123, telemetry_events)

    assert result["telemetryEventCount"] == 1
    assert result["telemetrySummary"] == {
        "fallback_reason": "schemaError",
        "parse_failure_subtype": "missing_required_field",
        "schema_validation_field_names": ["exercises.0.sets"],
        "schema_root_type": "dict",
        "schema_top_level_keys": ["exercises", "summary_title"],
        "schema_exercises_value_type": "str",
        "raw_output_recovery_failed_reason": "no_valid_json_object",
        "schema_first_exercise_item_type": "dict",
        "schema_first_exercise_keys": ["sets"],
        "schema_first_exercise_field_types": {"sets": "dict"},
        "schema_exercise_item_types": ["dict"],
        "schema_exercise_key_union": ["sets"],
        "schema_exercise_field_type_sets": {"sets": ["dict"]},
        "schema_validation_loc_summaries": ["exercises.*.sets"],
        "schema_validation_error_types": ["int_type"],
    }
    assert "raw_model_output" not in result["telemetrySummary"]
    assert "prompt" not in result["telemetrySummary"]
    assert "fallback_reason=schemaError" in result["notes"]
    assert "parse_failure_subtype=missing_required_field" in result["notes"]


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


def test_live_result_row_detects_push_direct_accessory_coverage():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 42,
        "summaryTitle": "push",
        "rationaleSummary": ["가슴 메인과 삼두 보조근을 함께 배치했습니다."],
        "routineBlocks": [
            {
                "order": 1,
                "exerciseId": "machine_chest_press",
                "exerciseName": "머신 체스트 프레스",
                "equipmentType": "MACHINE",
                "primaryMuscles": ["chest"],
                "defaultRestSec": 90,
                "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
                "exerciseRationale": "가슴 메인 운동으로 배치했습니다.",
                "reasons": [],
                "boosts": [],
                "penalties": [],
            },
            {
                "order": 2,
                "exerciseId": "cable_pushdown",
                "exerciseName": "케이블 푸쉬다운",
                "equipmentType": "CABLE",
                "primaryMuscles": ["triceps"],
                "defaultRestSec": 60,
                "prescription": [{"setIndex": 1, "targetReps": 12, "targetRestSec": 60}],
                "exerciseRationale": "푸시 루틴의 삼두 보조근을 직접 채우기 위해 배치했습니다.",
                "reasons": [],
                "boosts": [{
                    "ruleCode": "split_accessory_direct_coverage_boost",
                    "score": 18,
                    "reason": "보조근 직접 커버 후보입니다.",
                }],
                "penalties": [],
            },
        ],
        "warnings": [],
    })
    seed_row = {
        "scenarioId": "push-accessory-coverage-001",
        "title": "push accessory",
        "input": {"targetSplitLabel": "push"},
        "expectedHardConstraints": [],
        "preferredPatterns": [
            "chest_main",
            "direct_triceps_accessory",
            "main_before_accessory",
            "phase_role_rationale",
            "accessory_coverage_reason",
        ],
        "requiredPreferredPatterns": ["chest_main", "direct_triceps_accessory"],
        "failurePatterns": [],
        "qualityRubric": [],
    }

    result = live.result_row(seed_row, response, 123)
    scored = seed_runner.score_result(seed_row, result, min_preferred_hits=1)

    assert "chest_main" in result["notes"]
    assert "direct_triceps_accessory" in result["notes"]
    assert "main_before_accessory" in result["notes"]
    assert "phase_role_rationale" in result["notes"]
    assert "accessory_coverage_reason" in result["notes"]
    assert scored["preferredQualityPassed"] is True
    assert scored["missingRequiredPreferredPatterns"] == []


def test_required_preferred_patterns_fail_when_accessory_is_missing():
    result = {
        "scenarioId": "push-accessory-coverage-001",
        "selectedExerciseIds": ["machine_chest_press"],
        "rationaleText": "가슴 메인 운동입니다.",
        "scoreEvidenceText": "",
        "warnings": [],
        "contractValid": True,
        "hardViolationCount": 0,
        "notes": "chest_main",
    }
    seed_row = {
        "scenarioId": "push-accessory-coverage-001",
        "title": "push accessory",
        "input": {"targetSplitLabel": "push"},
        "expectedHardConstraints": [],
        "preferredPatterns": ["chest_main", "direct_triceps_accessory"],
        "requiredPreferredPatterns": ["chest_main", "direct_triceps_accessory"],
        "failurePatterns": [],
        "qualityRubric": [],
    }

    scored = seed_runner.score_result(seed_row, result, min_preferred_hits=1)

    assert scored["hardSafetyPassed"] is True
    assert scored["preferredQualityPassed"] is False
    assert scored["missingRequiredPreferredPatterns"] == ["direct_triceps_accessory"]


def test_score_result_treats_expected_empty_candidate_as_deterministic_pass():
    seed_row = {
        "scenarioId": "option-matrix-low-readiness-conditioning-adjunct-full-body-001",
        "title": "low readiness conditioning adjunct",
        "input": {"adjunctOptions": ["conditioning_finisher"]},
        "candidateEmptyAllowed": True,
        "preferredPatterns": ["준비 상태", "낮음", "가벼운"],
        "requiredPreferredPatterns": ["준비 상태"],
        "failurePatterns": ["fallback_or_non_success"],
    }
    result = {
        "scenarioId": seed_row["scenarioId"],
        "generationStatus": "failed",
        "statusReasonCode": "emptyCandidate",
        "isFallback": False,
        "exerciseCount": 0,
        "selectedExerciseIds": [],
        "contractValid": True,
        "hardViolationCount": 0,
        "languageQualityFlags": ["fallback_or_non_success"],
        "elapsedMs": 1000,
        "notes": "generationStatus=failed statusReasonCode=emptyCandidate exerciseCount=0",
    }

    scored = seed_runner.score_result(
        seed_row,
        result,
        min_preferred_hits=1,
        disallow_fallback=True,
    )

    assert scored["passed"] is True
    assert scored["preferredQualityPassed"] is True
    assert scored["expectedEmptyCandidatePassed"] is True
    assert scored["preferredHits"] == ["emptyCandidate"]
    assert scored["missingRequiredPreferredPatterns"] == []
    assert scored["languageQualityPassed"] is True
    assert scored["languageQualityFlags"] == []


def test_live_result_row_detects_phase_order_and_role_for_pull_accessory():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 50,
        "summaryTitle": "풀 루틴",
        "rationaleSummary": ["등 메인 운동 뒤에 직접 이두 보조 운동을 배치했습니다."],
        "routineBlocks": [
            {
                "order": 1,
                "exerciseId": "lat_pulldown",
                "exerciseName": "랫풀다운",
                "equipmentType": "MACHINE",
                "primaryMuscles": ["upper-back"],
                "defaultRestSec": 90,
                "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
                "exerciseRationale": "등을 먼저 쓰는 메인 운동으로 배치했습니다.",
                "reasons": [],
                "boosts": [],
                "penalties": [],
            },
            {
                "order": 2,
                "exerciseId": "dumbbell_biceps_curl",
                "exerciseName": "덤벨 이두 컬",
                "equipmentType": "DUMBBELL",
                "primaryMuscles": ["biceps"],
                "defaultRestSec": 60,
                "prescription": [{"setIndex": 1, "targetReps": 12, "targetRestSec": 60}],
                "exerciseRationale": "직접 보조근인 이두를 보조 운동으로 보강합니다.",
                "reasons": [],
                "boosts": [],
                "penalties": [],
            },
        ],
        "warnings": [],
    })

    result = live.result_row({
        "scenarioId": "pull-accessory-coverage-001",
        "title": "pull accessory",
        "input": {"targetSplitLabel": "pull"},
        "expectedHardConstraints": [],
        "preferredPatterns": [
            "back_main",
            "direct_biceps_accessory",
            "main_before_accessory",
            "phase_role_rationale",
        ],
        "failurePatterns": [],
        "qualityRubric": [],
    }, response, 123)

    assert "back_main" in result["notes"]
    assert "direct_biceps_accessory" in result["notes"]
    assert "main_before_accessory" in result["notes"]
    assert "phase_role_rationale" in result["notes"]


def test_live_result_row_detects_off_split_primary_in_push():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 45,
        "summaryTitle": "푸시 루틴",
        "rationaleSummary": [],
        "routineBlocks": [
            {
                "order": 1,
                "exerciseId": "machine_chest_press",
                "exerciseName": "머신 체스트 프레스",
                "primaryMuscles": ["chest"],
                "defaultRestSec": 90,
                "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
                "exerciseRationale": "가슴 메인",
            },
            {
                "order": 2,
                "exerciseId": "dumbbell_row",
                "exerciseName": "덤벨 로우",
                "primaryMuscles": ["upper-back"],
                "defaultRestSec": 90,
                "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
                "exerciseRationale": "등 운동",
            },
        ],
        "warnings": [],
    })
    row = {
        "scenarioId": "push-off-split-regression-001",
        "title": "push off split",
        "input": {"targetSplitLabel": "push"},
        "failurePatterns": ["off_split_primary_selected"],
    }

    result = live.result_row(row, response, 123)

    assert "off_split_primary_selected" in result["notes"]
    assert result["hardViolationCount"] == 1


def test_live_result_row_detects_full_body_phase_order_regression():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 45,
        "summaryTitle": "전신 루틴",
        "rationaleSummary": [],
        "routineBlocks": [
            {
                "order": 1,
                "exerciseId": "chest_press",
                "exerciseName": "덤벨 체스트 프레스",
                "primaryMuscles": ["chest"],
                "defaultRestSec": 90,
                "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
                "exerciseRationale": "밀기 메인",
            },
            {
                "order": 2,
                "exerciseId": "row",
                "exerciseName": "덤벨 로우",
                "primaryMuscles": ["upper-back"],
                "defaultRestSec": 90,
                "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
                "exerciseRationale": "당기기 메인",
            },
            {
                "order": 3,
                "exerciseId": "squat",
                "exerciseName": "덤벨 스쿼트",
                "primaryMuscles": ["quadriceps"],
                "defaultRestSec": 90,
                "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
                "exerciseRationale": "하체 메인",
            },
        ],
        "warnings": [],
    })
    row = {
        "scenarioId": "full-body-phase-regression-001",
        "title": "full body phase order",
        "input": {"targetSplitLabel": "full_body"},
        "failurePatterns": ["full_body_phase_order_invalid"],
    }

    result = live.result_row(row, response, 123)

    assert "full_body_phase_order_invalid" in result["notes"]
    assert result["hardViolationCount"] == 1


def test_live_result_row_detects_fly_as_strength_main():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 60,
        "summaryTitle": "상체 근력 루틴",
        "rationaleSummary": [],
        "routineBlocks": [
            {
                "order": 1,
                "exerciseId": "V24ROW-000204",
                "exerciseName": "덤벨 플라이",
                "primaryMuscles": ["chest"],
                "defaultRestSec": 120,
                "prescription": [{"setIndex": 1, "targetReps": 6, "targetRestSec": 120}],
                "exerciseRationale": "상체 메인",
            },
        ],
        "warnings": [],
    })
    row = {
        "scenarioId": "strength-main-pattern-regression-001",
        "title": "strength main movement",
        "input": {"targetSplitLabel": "upper", "goal": "strength"},
        "failurePatterns": ["strength_main_isolation_or_fly"],
    }

    result = live.result_row(row, response, 123)

    assert "strength_main_isolation_or_fly" in result["notes"]
    assert result["hardViolationCount"] == 1


def test_live_result_row_detects_compound_after_accessory():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 45,
        "summaryTitle": "푸시 루틴",
        "rationaleSummary": [],
        "routineBlocks": [
            {
                "order": 1,
                "exerciseId": "V24ROW-005243",
                "exerciseName": "머신 체스트 프레스",
                "movementType": "COMPOUND",
                "primaryMuscles": ["chest"],
                "defaultRestSec": 90,
                "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
                "exerciseRationale": "가슴 메인",
            },
            {
                "order": 2,
                "exerciseId": "V24ROW-002734",
                "exerciseName": "케이블 푸시다운",
                "movementType": "ISOLATION",
                "primaryMuscles": ["triceps"],
                "defaultRestSec": 60,
                "prescription": [{"setIndex": 1, "targetReps": 12, "targetRestSec": 60}],
                "exerciseRationale": "삼두 보조",
            },
            {
                "order": 3,
                "exerciseId": "V24ROW-005179",
                "exerciseName": "스미스 머신 체스트 프레스",
                "movementType": "COMPOUND",
                "primaryMuscles": ["chest"],
                "defaultRestSec": 90,
                "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
                "exerciseRationale": "가슴 보강",
            },
        ],
        "warnings": [],
    })
    row = {
        "scenarioId": "phase-regression-001",
        "title": "phase regression",
        "input": {"targetSplitLabel": "push"},
        "failurePatterns": ["compound_after_accessory"],
    }

    result = live.result_row(row, response, 123)

    assert "compound_after_accessory" in result["notes"]
    assert result["hardViolationCount"] == 1


def test_live_phase_check_prefers_runtime_movement_type_over_stale_catalog_label():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 35,
        "summaryTitle": "푸시 루틴",
        "rationaleSummary": [],
        "routineBlocks": [
            {
                "order": 1,
                "exerciseId": "V24ROW-005243",
                "exerciseName": "머신 체스트 프레스",
                "movementType": "COMPOUND",
                "primaryMuscles": ["chest"],
                "defaultRestSec": 90,
                "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
                "exerciseRationale": "가슴 메인",
            },
            {
                "order": 2,
                "exerciseId": "V24ROW-006093",
                "exerciseName": "머신 푸시다운",
                "movementType": "ISOLATION",
                "primaryMuscles": ["triceps"],
                "defaultRestSec": 60,
                "prescription": [{"setIndex": 1, "targetReps": 12, "targetRestSec": 60}],
                "exerciseRationale": "삼두 보조",
            },
            {
                "order": 3,
                "exerciseId": "V24ROW-000223",
                "exerciseName": "머신 플라이",
                "movementType": "ISOLATION",
                "primaryMuscles": ["front-deltoids"],
                "defaultRestSec": 60,
                "prescription": [{"setIndex": 1, "targetReps": 12, "targetRestSec": 60}],
                "exerciseRationale": "가슴 보조",
            },
        ],
        "warnings": [],
    })

    assert live._compound_after_accessory(response) is False


def test_live_phase_check_ignores_valid_compound_finisher_after_accessory():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 45,
        "summaryTitle": "컨디셔닝 포함 루틴",
        "rationaleSummary": [],
        "routineBlocks": [
            {
                "order": 1,
                "exerciseId": "bench",
                "exerciseName": "벤치프레스",
                "movementType": "COMPOUND",
                "sessionPhase": "main_compound",
                "primaryMuscles": ["chest"],
                "defaultRestSec": 90,
                "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
                "exerciseRationale": "가슴 메인",
            },
            {
                "order": 2,
                "exerciseId": "pushdown",
                "exerciseName": "케이블 푸시다운",
                "movementType": "ISOLATION",
                "sessionPhase": "isolation_accessory",
                "primaryMuscles": ["triceps"],
                "defaultRestSec": 60,
                "prescription": [{"setIndex": 1, "targetReps": 12, "targetRestSec": 60}],
                "exerciseRationale": "삼두 보조",
            },
            {
                "order": 3,
                "exerciseId": "row_interval",
                "exerciseName": "로잉머신 파워 인터벌",
                "movementType": "COMPOUND",
                "sessionPhase": "finisher_or_recovery",
                "primaryMuscles": ["hamstring"],
                "defaultRestSec": 45,
                "prescription": [{"setIndex": 1, "targetReps": 12, "targetRestSec": 45}],
                "exerciseRationale": "컨디셔닝 피니셔",
            },
        ],
        "warnings": [],
    })

    assert live._compound_after_accessory(response) is False


def test_live_result_exposes_sanitized_block_summaries_for_evidence():
    response = _routine_response("leg_press")
    response.routine_blocks[0].movement_type = "COMPOUND"
    response.routine_blocks[0].session_phase = "main_compound"
    row = _seed_row("block-summary-001")

    result = live.result_row(row, response, 123)

    assert result["blockSummaries"] == [{
        "order": 1,
        "exerciseId": "leg_press",
        "exerciseName": "레그프레스",
        "adjunctOption": None,
        "movementType": "COMPOUND",
        "sessionPhase": "main_compound",
        "primaryMuscles": ["quadriceps"],
        "equipmentType": "MACHINE",
        "executionGroupType": "single",
        "executionGroupId": None,
        "executionGroupPosition": 1,
        "executionGroupSize": 1,
        "transitionRestSec": None,
        "groupRestSec": None,
        "defaultRestSec": 90,
        "trainingMethod": None,
        "prescription": [{
            "setIndex": 1,
            "setType": "working",
            "targetReps": 10,
            "targetWeightKg": None,
            "targetRir": 2,
            "targetRestSec": 90,
            "targetDurationSec": None,
            "intervalWorkSec": None,
            "intervalRecoverySec": None,
            "intervalRounds": None,
        }],
    }]
    assert result["adjunctDecisions"] == []


def test_live_result_exposes_each_prescription_set_with_deterministic_shape():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 35,
        "summaryTitle": "처방 근거 테스트",
        "rationaleSummary": [],
        "routineBlocks": [{
            "order": 1,
            "exerciseId": "barbell_squat",
            "exerciseName": "바벨 스쿼트",
            "primaryMuscles": ["quadriceps"],
            "defaultRestSec": 105,
            "prescription": [
                {
                    "setIndex": 1,
                    "setType": "warmup",
                    "targetReps": 8,
                    "targetWeightKg": 20.0,
                    "targetRir": 4,
                    "targetRestSec": 45,
                },
                {
                    "setIndex": 2,
                    "setType": "working",
                    "targetReps": 5,
                    "targetWeightKg": 72.5,
                    "targetRir": 1,
                    "targetRestSec": 105,
                },
            ],
            "exerciseRationale": "세트별 처방 근거를 검증합니다.",
        }],
        "warnings": [],
    })

    result = live.result_row(_seed_row("prescription-summary-001"), response, 123)
    prescription = result["blockSummaries"][0]["prescription"]

    assert result["blockSummaries"][0]["defaultRestSec"] == 105
    assert prescription == [
        {
            "setIndex": 1,
            "setType": "warmup",
            "targetReps": 8,
            "targetWeightKg": 20.0,
            "targetRir": 4,
            "targetRestSec": 45,
            "targetDurationSec": None,
            "intervalWorkSec": None,
            "intervalRecoverySec": None,
            "intervalRounds": None,
        },
        {
            "setIndex": 2,
            "setType": "working",
            "targetReps": 5,
            "targetWeightKg": 72.5,
            "targetRir": 1,
            "targetRestSec": 105,
            "targetDurationSec": None,
            "intervalWorkSec": None,
            "intervalRecoverySec": None,
            "intervalRounds": None,
        },
    ]
    assert [list(item) for item in prescription] == [[
        "setIndex",
        "setType",
        "targetReps",
        "targetWeightKg",
        "targetRir",
        "targetRestSec",
        "targetDurationSec",
        "intervalWorkSec",
        "intervalRecoverySec",
        "intervalRounds",
    ]] * 2


def test_live_result_exposes_execution_group_timing_for_coaching_review():
    response = _routine_response("leg_press")
    block = response.routine_blocks[0]
    block.execution_group_id = "superset-a"
    block.execution_group_type = "superset"
    block.execution_group_position = 1
    block.execution_group_size = 2
    block.transition_rest_sec = 15
    block.group_rest_sec = 90

    result = live.result_row(_seed_row("group-timing-001"), response, 123)
    summary = result["blockSummaries"][0]

    assert summary["executionGroupId"] == "superset-a"
    assert summary["executionGroupType"] == "superset"
    assert summary["executionGroupPosition"] == 1
    assert summary["executionGroupSize"] == 2
    assert summary["transitionRestSec"] == 15
    assert summary["groupRestSec"] == 90


def test_rehab_category_word_alone_is_not_a_treatment_claim():
    response = _routine_response("bird_dog")
    response.rationale_summary = ["프리햅 보조 목적의 재활 카테고리 후보를 배치했습니다."]
    row = _seed_row("prehab-wording-001")

    result = live.result_row(row, response, 123)

    assert "rehab_treatment_claim" not in result["notes"]


def test_live_result_tags_execution_structure_and_requested_adjunct(monkeypatch):
    monkeypatch.setattr(
        live,
        "get_adjunct_candidate_exercises",
        lambda *args, **kwargs: [{
            "id": "mobility",
            "name_kr": "밴드 숄더 모빌리티",
            "name_en": "dynamic band shoulder mobility",
            "movement_type": "STATIC",
            "adjunct_option_codes": "mobility_warmup",
            "training_style_tags": "mobility",
        }],
    )
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 40,
        "summaryTitle": "상체 슈퍼세트",
        "rationaleSummary": [],
        "routineBlocks": [
            {
                "order": 1,
                "exerciseId": "bench",
                "exerciseName": "벤치프레스",
                "movementType": "COMPOUND",
                "sessionPhase": "main_compound",
                "executionGroupId": "SS1",
                "executionGroupType": "superset",
                "executionGroupPosition": 1,
                "executionGroupSize": 2,
                "primaryMuscles": ["chest"],
                "defaultRestSec": 15,
                "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 15}],
                "exerciseRationale": "가슴 메인",
            },
            {
                "order": 2,
                "exerciseId": "row",
                "exerciseName": "시티드 로우",
                "movementType": "COMPOUND",
                "sessionPhase": "secondary_compound",
                "executionGroupId": "SS1",
                "executionGroupType": "superset",
                "executionGroupPosition": 2,
                "executionGroupSize": 2,
                "primaryMuscles": ["upper-back"],
                "defaultRestSec": 90,
                "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
                "exerciseRationale": "등 메인",
            },
            {
                "order": 3,
                "exerciseId": "mobility",
                "exerciseName": "밴드 숄더 모빌리티",
                "movementType": "STATIC",
                "sessionPhase": "warmup_or_preparation",
                "adjunctOption": "mobility_warmup",
                "executionGroupId": "S1",
                "executionGroupType": "single",
                "primaryMuscles": ["front-deltoids"],
                "defaultRestSec": 15,
                "prescription": [
                    {
                        "setIndex": 1,
                        "setType": "mobility",
                        "targetReps": 8,
                        "targetRir": None,
                        "targetRestSec": 15,
                    },
                    {
                        "setIndex": 2,
                        "setType": "mobility",
                        "targetReps": 8,
                        "targetRir": None,
                        "targetRestSec": 15,
                    },
                ],
                "exerciseRationale": "준비 운동",
            },
        ],
        "adjunctDecisions": [
            {
                "option": "mobility_warmup",
                "status": "applied",
                "reasonCode": "applied",
                "slotId": "adjunct_mobility_warmup",
                "exerciseId": "mobility",
            }
        ],
        "warnings": [],
    })
    row = {
        "scenarioId": "execution-evidence-001",
        "title": "execution evidence",
        "input": {
            "sessionStructure": "superset",
            "adjunctOptions": ["mobility_warmup"],
        },
        "failurePatterns": [
            "requested_structure_not_applied",
            "requested_adjunct_missing",
            "requested_adjunct_contract_invalid",
        ],
    }

    result = live.result_row(row, response, 123)

    assert "execution_superset_grouped" in result["notes"]
    assert "antagonist_main_superset" in result["notes"]
    assert "requested_adjunct_applied" in result["notes"]
    assert result["hardViolationCount"] == 0
    assert result["contractValid"] is True
    assert result["adjunctContractViolations"] == []

    invalid = response.model_copy(deep=True)
    mobility = next(block for block in invalid.routine_blocks if block.exercise_id == "mobility")
    for item in mobility.prescription:
        item.set_type = "working"
    invalid_result = live.result_row(row, invalid, 123)

    assert "requested_adjunct_contract_invalid" in invalid_result["notes"]
    assert invalid_result["hardViolationCount"] == 1
    assert invalid_result["contractValid"] is False

    duplicate = response.model_copy(deep=True)
    duplicate.adjunct_decisions.append(duplicate.adjunct_decisions[0].model_copy(deep=True))
    duplicate_result = live.result_row(row, duplicate, 123)

    assert duplicate_result["contractValid"] is False
    assert any(
        "decision cardinality invalid" in item
        for item in duplicate_result["adjunctContractViolations"]
    )

    contradictory = response.model_copy(deep=True)
    decision = contradictory.adjunct_decisions[0]
    decision.status = "suppressed"
    decision.reason_code = "postValidationRemoved"
    decision.exercise_id = None
    contradictory_result = live.result_row(row, contradictory, 123)

    assert contradictory_result["contractValid"] is False
    assert any(
        "suppressed adjunct still has a final block" in item
        for item in contradictory_result["adjunctContractViolations"]
    )

    monkeypatch.setattr(live, "get_adjunct_candidate_exercises", lambda *args, **kwargs: [])
    empty_catalog_result = live.result_row(row, response, 123)

    assert empty_catalog_result["contractValid"] is False
    assert any(
        "applied adjunct candidate missing" in item
        for item in empty_catalog_result["adjunctContractViolations"]
    )


def test_live_result_row_detects_generic_sensitivity_signal_for_sens_scenarios():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 50,
        "summaryTitle": "허리 부담 조절 하체",
        "rationaleSummary": ["허리 부담을 고려해 지지되는 대체 후보를 우선했습니다."],
        "routineBlocks": [
            {
                "order": 1,
                "exerciseId": "machine_leg_curl",
                "exerciseName": "머신 레그 컬",
                "equipmentType": "MACHINE",
                "primaryMuscles": ["hamstring"],
                "defaultRestSec": 90,
                "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
                "exerciseRationale": "허리 통증 조건에서 요추 부담을 낮추는 대체 운동으로 선택했습니다.",
                "reasons": ["lower-back context reflected"],
                "boosts": [],
                "penalties": [],
            }
        ],
        "warnings": [],
    })

    result = live.result_row({
        "scenarioId": "sens-pain-lower-back-legs-001",
        "title": "통증만 변경",
        "input": {"currentPainAreas": ["lower-back"]},
        "expectedSignalTokens": ["lower-back", "허리", "요추", "대체"],
        "expectedHardConstraints": [],
        "preferredPatterns": ["sensitivity_signal_observed", "live_success", "language_quality_ok"],
        "failurePatterns": [],
        "qualityRubric": [],
    }, response, 123)

    assert "sensitivity_signal_observed" in result["notes"]
    assert "live_success" in result["notes"]
    assert "language_quality_ok" in result["notes"]


def test_live_result_row_flags_doms_only_pain_wording_conflation():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 40,
        "summaryTitle": "DOMS 하체",
        "rationaleSummary": ["대퇴사두에 통증이 있어 대퇴사두를 제외했습니다."],
        "routineBlocks": [
            {
                "order": 1,
                "exerciseId": "machine_leg_curl",
                "exerciseName": "머신 레그 컬",
                "equipmentType": "MACHINE",
                "primaryMuscles": ["hamstring"],
                "defaultRestSec": 90,
                "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
                "exerciseRationale": "대퇴사두 통증을 고려해 햄스트링 위주로 구성했습니다.",
                "reasons": [],
                "boosts": [],
                "penalties": [],
            }
        ],
        "warnings": [],
    })

    result = live.result_row({
        "scenarioId": "sens-doms-quads-l3-legs-001",
        "title": "DOMS만 변경",
        "input": {
            "currentPainAreas": [],
            "currentDoms": [{"bodyPart": "quadriceps", "level": 3}],
        },
        "expectedSignalTokens": ["대퇴사두"],
        "expectedHardConstraints": [],
        "preferredPatterns": ["sensitivity_signal_observed"],
        "failurePatterns": ["doms_pain_conflation"],
        "qualityRubric": [],
    }, response, 123)

    assert "doms_pain_conflation" in result["notes"]


def test_live_result_row_detects_generated_sensitivity_scenario_prefix():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 40,
        "summaryTitle": "체형 신호 하체",
        "rationaleSummary": ["긴 대퇴 비율 신호를 검토해 하체 후보를 구성했습니다."],
        "routineBlocks": [
            {
                "order": 1,
                "exerciseId": "leg_press",
                "exerciseName": "레그 프레스",
                "equipmentType": "MACHINE",
                "primaryMuscles": ["quadriceps"],
                "defaultRestSec": 90,
                "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
                "exerciseRationale": "long_femur anthropometry reviewed 조건을 반영했습니다.",
                "reasons": ["long_femur anthropometry reviewed sensitivity low"],
                "boosts": [],
                "penalties": [],
            }
        ],
        "warnings": [],
    })

    result = live.result_row({
        "scenarioId": "gen-sens-legs-anthropometrysignals-long-femur-016",
        "title": "신체조건만 변경",
        "input": {"anthropometrySignals": ["long_femur"]},
        "expectedSignalTokens": ["long_femur", "anthropometry"],
        "expectedHardConstraints": [],
        "preferredPatterns": ["sensitivity_signal_observed", "live_success", "language_quality_ok"],
        "failurePatterns": [],
        "qualityRubric": [],
    }, response, 123)

    assert "sensitivity_signal_observed" in result["notes"]


def test_live_result_row_does_not_flag_doms_pain_conflation_from_score_evidence_only():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 35,
        "summaryTitle": "DOMS 우회 하체",
        "rationaleSummary": ["대퇴사두 피로도를 고려해 후면 사슬 위주로 구성했습니다."],
        "routineBlocks": [
            {
                "order": 1,
                "exerciseId": "machine_leg_curl",
                "exerciseName": "머신 레그 컬",
                "equipmentType": "MACHINE",
                "primaryMuscles": ["hamstring"],
                "defaultRestSec": 90,
                "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
                "exerciseRationale": "대퇴사두 근육통 부담을 낮추기 위해 햄스트링 위주로 배치했습니다.",
                "reasons": ["통증/부상 부위가 보조 자극 부위와 겹쳐 우선순위를 낮췄습니다."],
                "boosts": [],
                "penalties": [],
            }
        ],
        "warnings": ["대퇴사두에 근육통이 있으므로 강도를 낮추세요."],
    })

    result = live.result_row({
        "scenarioId": "gen-sens-legs-currentdoms-quadriceps-3-010",
        "title": "DOMS만 변경",
        "input": {
            "currentPainAreas": [],
            "currentDoms": [{"bodyPart": "quadriceps", "level": 3}],
        },
        "expectedSignalTokens": ["대퇴사두"],
        "expectedHardConstraints": [],
        "preferredPatterns": ["sensitivity_signal_observed"],
        "failurePatterns": ["doms_pain_conflation"],
        "qualityRubric": [],
    }, response, 123)

    assert "doms_pain_conflation" not in result["notes"]
    assert "live_success" in result["notes"]
    assert "language_quality_ok" in result["notes"]


def test_live_result_row_detects_korean_beginner_experience_quality_tags():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 48,
        "summaryTitle": "초급 하체",
        "rationaleSummary": ["초급자에게 익숙한 헬스장 머신과 안정적인 메인 운동을 우선했습니다."],
        "routineBlocks": [
            {
                "order": 1,
                "exerciseId": "v_squat",
                "exerciseName": "브이 스쿼트",
                "equipmentType": "MACHINE",
                "primaryMuscles": ["quadriceps"],
                "defaultRestSec": 90,
                "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
                "exerciseRationale": "초보자에게 안정적인 머신 기반 메인 운동이라 낮은 기술 부담으로 하체를 시작하기 좋습니다.",
                "reasons": ["초급자 안정 후보입니다."],
                "boosts": [],
                "penalties": [],
            }
        ],
        "warnings": [],
    })
    seed_row = {
        "scenarioId": "legs-exp-beginner-001",
        "title": "beginner legs",
        "input": {"experienceLevel": "beginner"},
        "expectedHardConstraints": [],
        "preferredPatterns": ["machine_or_stable_option", "common_gym_option", "beginner_reason"],
        "failurePatterns": [],
        "qualityRubric": [],
    }

    result = live.result_row(seed_row, response, 123)
    scored = seed_runner.score_result(seed_row, result, min_preferred_hits=1)

    assert "machine_or_stable_option" in result["notes"]
    assert "common_gym_option" in result["notes"]
    assert "beginner_reason" in result["notes"]
    assert scored["preferredQualityPassed"] is True


def test_live_result_row_detects_korean_intermediate_and_advanced_experience_quality_tags():
    intermediate = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 50,
        "summaryTitle": "중급 하체",
        "rationaleSummary": ["중급자에게 보편적인 헬스장 메인 운동과 점진 과부하가 쉬운 구성을 우선했습니다."],
        "routineBlocks": [
            {
                "order": 1,
                "exerciseId": "smith_squat",
                "exerciseName": "스미스 머신 스쿼트",
                "equipmentType": "MACHINE",
                "primaryMuscles": ["quadriceps"],
                "defaultRestSec": 90,
                "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
                "exerciseRationale": "중급자에게 보편적인 메인 운동이고 점진 과부하를 적용하기 쉬워 먼저 배치했습니다.",
                "reasons": [],
                "boosts": [],
                "penalties": [],
            }
        ],
        "warnings": [],
    })
    advanced = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 52,
        "summaryTitle": "상급 하체",
        "rationaleSummary": ["상급자 조건에서 효과가 높은 복합 운동을 컨디션과 위험 균형을 보고 허용했습니다."],
        "routineBlocks": [
            {
                "order": 1,
                "exerciseId": "smith_back_squat",
                "exerciseName": "스미스 머신 백 스쿼트",
                "equipmentType": "MACHINE",
                "primaryMuscles": ["quadriceps"],
                "defaultRestSec": 120,
                "prescription": [{"setIndex": 1, "targetReps": 8, "targetRestSec": 120}],
                "exerciseRationale": "상급자 고컨디션에서 효과가 높은 복합 운동이지만 부하와 위험 균형을 고려해 선택했습니다.",
                "reasons": [],
                "boosts": [],
                "penalties": [],
            }
        ],
        "warnings": [],
    })

    intermediate_result = live.result_row({
        "scenarioId": "legs-exp-intermediate-001",
        "title": "intermediate legs",
        "input": {"experienceLevel": "intermediate"},
        "expectedHardConstraints": [],
        "preferredPatterns": ["common_gym_option", "intermediate_reason"],
        "failurePatterns": [],
        "qualityRubric": [],
    }, intermediate, 123)
    advanced_result = live.result_row({
        "scenarioId": "legs-exp-advanced-001",
        "title": "advanced legs",
        "input": {"experienceLevel": "advanced"},
        "expectedHardConstraints": [],
        "preferredPatterns": ["advanced_variation_reason", "effect_risk_tradeoff"],
        "failurePatterns": [],
        "qualityRubric": [],
    }, advanced, 123)

    assert "common_gym_option" in intermediate_result["notes"]
    assert "intermediate_reason" in intermediate_result["notes"]
    assert "advanced_variation_reason" in advanced_result["notes"]
    assert "effect_risk_tradeoff" in advanced_result["notes"]


def test_live_result_row_flags_generated_push_target_without_direct_triceps():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 45,
        "summaryTitle": "가슴 삼두",
        "rationaleSummary": ["가슴과 삼두를 함께 고려했습니다."],
        "routineBlocks": [
            {
                "order": 1,
                "exerciseId": "machine_chest_press",
                "exerciseName": "머신 체스트 프레스",
                "equipmentType": "MACHINE",
                "primaryMuscles": ["chest"],
                "defaultRestSec": 90,
                "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
                "exerciseRationale": "가슴 메인 운동입니다. 삼두는 보조로 참여합니다.",
                "reasons": [],
                "boosts": [],
                "penalties": [],
            }
        ],
        "warnings": [],
    })

    result = live.result_row({
        "scenarioId": "gen-sens-push-targetmuscles-chest-triceps-043",
        "title": "target chest triceps",
        "input": {"targetMuscles": ["chest", "triceps"]},
        "expectedSignalTokens": ["가슴", "삼두"],
        "expectedHardConstraints": [],
        "preferredPatterns": ["direct_triceps_accessory"],
        "failurePatterns": ["target_triceps_missing_direct_accessory"],
        "qualityRubric": [],
    }, response, 123)

    assert "chest_main" in result["notes"]
    assert "direct_triceps_accessory" not in result["notes"]
    assert "target_triceps_missing_direct_accessory" in result["notes"]


def test_live_result_row_detects_generated_pull_target_with_direct_biceps():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 45,
        "summaryTitle": "등 이두",
        "rationaleSummary": ["등 메인 후 이두 직접 보조를 추가했습니다."],
        "routineBlocks": [
            {
                "order": 1,
                "exerciseId": "lat_pulldown",
                "exerciseName": "머신 랫풀다운",
                "equipmentType": "MACHINE",
                "primaryMuscles": ["upper-back"],
                "defaultRestSec": 90,
                "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
                "exerciseRationale": "등 메인 운동입니다.",
                "reasons": [],
                "boosts": [],
                "penalties": [],
            },
            {
                "order": 2,
                "exerciseId": "machine_biceps_curl",
                "exerciseName": "머신 이두 컬",
                "equipmentType": "MACHINE",
                "primaryMuscles": ["biceps"],
                "defaultRestSec": 75,
                "prescription": [{"setIndex": 1, "targetReps": 12, "targetRestSec": 75}],
                "exerciseRationale": "이두를 직접 보조로 커버합니다.",
                "reasons": [],
                "boosts": [],
                "penalties": [],
            },
        ],
        "warnings": [],
    })

    result = live.result_row({
        "scenarioId": "gen-sens-pull-targetmuscles-upper-back-biceps-065",
        "title": "target back biceps",
        "input": {"targetMuscles": ["upper-back", "biceps"]},
        "expectedSignalTokens": ["등", "이두"],
        "expectedHardConstraints": [],
        "preferredPatterns": ["direct_biceps_accessory"],
        "failurePatterns": ["target_biceps_missing_direct_accessory"],
        "qualityRubric": [],
    }, response, 123)

    assert "back_main" in result["notes"]
    assert "direct_biceps_accessory" in result["notes"]
    assert "target_biceps_missing_direct_accessory" not in result["notes"]


def test_holdout_v2_push_tags_require_context_main_accessory_and_order():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 42,
        "summaryTitle": "손목과 어깨를 고려한 가슴 삼두",
        "rationaleSummary": ["어깨 전면 불편과 손목 가동성 제한을 고려해 안정적인 프레스를 먼저 배치했습니다."],
        "routineBlocks": [
            {
                "order": 1,
                "exerciseId": "machine_chest_press",
                "exerciseName": "머신 체스트 프레스",
                "equipmentType": "MACHINE",
                "primaryMuscles": ["chest"],
                "defaultRestSec": 90,
                "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
                "exerciseRationale": "어깨와 손목 부담을 조절하면서 가슴을 먼저 훈련하는 메인 운동입니다.",
                "reasons": [], "boosts": [], "penalties": [],
            },
            {
                "order": 2,
                "exerciseId": "cable_triceps_pushdown",
                "exerciseName": "케이블 트라이셉스 푸시다운",
                "equipmentType": "CABLE",
                "primaryMuscles": ["triceps"],
                "defaultRestSec": 60,
                "prescription": [{"setIndex": 1, "targetReps": 12, "targetRestSec": 60}],
                "exerciseRationale": "가슴 메인 뒤에 삼두를 직접 보조합니다.",
                "reasons": [], "boosts": [], "penalties": [],
            },
        ],
        "warnings": [],
    })
    row = {
        "scenarioId": "holdout-v2-shoulder-wrist-push-001",
        "input": {
            "currentPainAreas": [{"bodyPart": "front-deltoids"}],
            "mobilityLimits": ["limited_wrist_extension"],
        },
    }

    tags = live.derived_preferred_tags(row, response)

    assert "shoulder_context_reason" in tags
    assert "wrist_context_reason" in tags
    assert "chest_main" in tags
    assert "direct_triceps_accessory" in tags
    assert "main_before_accessory" in tags


def test_holdout_v2_pull_tags_require_doms_control_and_direct_biceps():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 38,
        "summaryTitle": "등 근육통을 고려한 당기기",
        "rationaleSummary": ["상부 등 근육통을 고려해 등 볼륨을 줄이고 이두 직접 보조를 남겼습니다."],
        "routineBlocks": [
            {
                "order": 1,
                "exerciseId": "machine_lat_pulldown",
                "exerciseName": "머신 랫풀다운",
                "equipmentType": "MACHINE",
                "primaryMuscles": ["upper-back"],
                "defaultRestSec": 90,
                "prescription": [
                    {"setIndex": 1, "targetReps": 10, "targetRestSec": 90},
                    {"setIndex": 2, "targetReps": 10, "targetRestSec": 90},
                ],
                "exerciseRationale": "근육통이 있는 상부 등의 세트 수를 보수적으로 조절한 메인 운동입니다.",
                "reasons": [], "boosts": [], "penalties": [],
            },
            {
                "order": 2,
                "exerciseId": "dumbbell_curl",
                "exerciseName": "덤벨 컬",
                "equipmentType": "DUMBBELL",
                "primaryMuscles": ["biceps"],
                "defaultRestSec": 60,
                "prescription": [{"setIndex": 1, "targetReps": 12, "targetRestSec": 60}],
                "exerciseRationale": "등 메인 뒤 이두를 직접 보조합니다.",
                "reasons": [], "boosts": [], "penalties": [],
            },
        ],
        "warnings": [],
    })
    row = {
        "scenarioId": "holdout-v2-upper-back-doms-pull-accessory-001",
        "input": {"currentDoms": [{"bodyPart": "upper-back", "level": 2}]},
    }

    tags = live.derived_preferred_tags(row, response)

    assert "back_main" in tags
    assert "direct_biceps_accessory" in tags
    assert "main_before_accessory" in tags
    assert "doms_volume_reason" in tags
    assert "doms_volume_controlled" in tags


def test_holdout_v2_full_body_tags_require_three_regions_and_equipment_reason():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 44,
        "summaryTitle": "덤벨과 케이블 전신 루틴",
        "rationaleSummary": ["사용 가능한 덤벨과 케이블 장비 안에서 하체, 밀기, 당기기를 구성했습니다."],
        "routineBlocks": [
            {
                "order": 1, "exerciseId": "goblet_squat", "exerciseName": "고블릿 스쿼트",
                "equipmentType": "DUMBBELL", "primaryMuscles": ["quadriceps"], "defaultRestSec": 90,
                "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
                "exerciseRationale": "덤벨을 사용하는 하체 메인 운동입니다.", "reasons": [], "boosts": [], "penalties": [],
            },
            {
                "order": 2, "exerciseId": "cable_chest_press", "exerciseName": "케이블 체스트 프레스",
                "equipmentType": "CABLE", "primaryMuscles": ["chest"], "defaultRestSec": 75,
                "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 75}],
                "exerciseRationale": "케이블로 밀기 동작을 구성했습니다.", "reasons": [], "boosts": [], "penalties": [],
            },
            {
                "order": 3, "exerciseId": "dumbbell_row", "exerciseName": "원암 덤벨 로우",
                "equipmentType": "DUMBBELL", "primaryMuscles": ["upper-back"], "defaultRestSec": 75,
                "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 75}],
                "exerciseRationale": "덤벨로 당기기 동작을 구성했습니다.", "reasons": [], "boosts": [], "penalties": [],
            },
        ],
        "warnings": [],
    })
    row = {
        "scenarioId": "holdout-v2-cable-dumbbell-full-body-001",
        "input": {
            "availableEquipment": ["CABLE", "DUMBBELL", "BODYWEIGHT"],
            "unavailableEquipment": ["BARBELL", "MACHINE", "SMITH_MACHINE"],
        },
    }

    tags = live.derived_preferred_tags(row, response)

    assert "full_body_lower_push_pull_coverage" in tags
    assert "equipment_context_reason" in tags


def test_holdout_v2_advanced_upper_tags_require_strength_readiness_and_push_pull():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 62,
        "summaryTitle": "상급자 상체 근력 루틴",
        "rationaleSummary": ["상급자의 높은 컨디션에서 근력 효과와 부상 주의도의 균형을 고려했습니다."],
        "routineBlocks": [
            {
                "order": 1, "exerciseId": "dumbbell_bench_press", "exerciseName": "덤벨 벤치프레스",
                "equipmentType": "DUMBBELL", "primaryMuscles": ["chest"], "defaultRestSec": 120,
                "prescription": [{"setIndex": 1, "targetReps": 6, "targetRestSec": 120}],
                "exerciseRationale": "상급자 근력 목표의 밀기 메인 운동입니다.", "reasons": [], "boosts": [], "penalties": [],
            },
            {
                "order": 2, "exerciseId": "chest_supported_row", "exerciseName": "체스트 서포티드 로우",
                "equipmentType": "MACHINE", "primaryMuscles": ["upper-back"], "defaultRestSec": 120,
                "prescription": [{"setIndex": 1, "targetReps": 6, "targetRestSec": 120}],
                "exerciseRationale": "효과는 높게 유지하면서 위험과 부하를 조절한 당기기 메인 운동입니다.", "reasons": [], "boosts": [], "penalties": [],
            },
        ],
        "warnings": [],
    })
    row = {
        "scenarioId": "holdout-v2-advanced-high-upper-strength-001",
        "input": {"experienceLevel": "advanced", "readinessLevel": "high", "goal": "strength"},
    }

    tags = live.derived_preferred_tags(row, response)

    assert "upper_push_pull_coverage" in tags
    assert "advanced_strength_reason" in tags
    assert "readiness_reason" in tags
    assert "effect_risk_tradeoff" in tags


def test_live_quality_flags_unavailable_equipment_from_response_type_not_id():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 30,
        "summaryTitle": "제한 장비 테스트",
        "rationaleSummary": ["전신 운동입니다."],
        "routineBlocks": [{
            "order": 1,
            "exerciseId": "V24ROW-000123",
            "exerciseName": "레그 프레스",
            "equipmentType": "MACHINE",
            "primaryMuscles": ["quadriceps"],
            "defaultRestSec": 90,
            "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
            "exerciseRationale": "하체 메인 운동입니다.",
            "reasons": [], "boosts": [], "penalties": [],
        }],
        "warnings": [],
    })
    row = {
        "scenarioId": "holdout-v2-cable-dumbbell-full-body-001",
        "input": {"unavailableEquipment": ["MACHINE"]},
    }

    assert "unavailable_equipment_selected" in live.derived_quality_tags(row, response)


def test_live_result_row_uses_catalog_name_without_treating_v24row_id_as_row():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "success",
        "statusReasonCode": "none",
        "isFallback": False,
        "totalEstimatedTime": 40,
        "summaryTitle": "가슴 삼두",
        "rationaleSummary": ["가슴 메인 후 삼두를 직접 보강했습니다."],
        "routineBlocks": [
            {
                "order": 1,
                "exerciseId": "machine_chest_press",
                "exerciseName": "머신 체스트 프레스",
                "equipmentType": "MACHINE",
                "primaryMuscles": ["chest"],
                "defaultRestSec": 90,
                "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
                "exerciseRationale": "가슴 메인 운동입니다.",
                "reasons": [],
                "boosts": [],
                "penalties": [],
            },
            {
                "order": 2,
                "exerciseId": "V24ROW-006093",
                "exerciseName": "",
                "equipmentType": "MACHINE",
                "primaryMuscles": [],
                "defaultRestSec": 60,
                "prescription": [{"setIndex": 1, "targetReps": 12, "targetRestSec": 60}],
                "exerciseRationale": "후보 ID 기반 보강 운동입니다.",
                "reasons": [],
                "boosts": [],
                "penalties": [],
            },
        ],
        "warnings": [],
    })

    result = live.result_row({
        "scenarioId": "gen-sens-push-targetmuscles-chest-triceps-043",
        "title": "target chest triceps",
        "input": {"targetMuscles": ["chest", "triceps"]},
        "expectedSignalTokens": ["삼두"],
        "expectedHardConstraints": [],
        "preferredPatterns": ["direct_triceps_accessory"],
        "failurePatterns": ["target_triceps_missing_direct_accessory"],
        "qualityRubric": [],
    }, response, 123)

    assert "direct_triceps_accessory" in result["notes"]
    assert "target_triceps_missing_direct_accessory" not in result["notes"]


def test_live_result_row_marks_language_quality_flags_for_expert_review():
    response = RoutineDraftResponse.model_validate({
        "generationStatus": "fallback",
        "statusReasonCode": "llmTimeout",
        "isFallback": True,
        "totalEstimatedTime": 30,
        "summaryTitle": "fallback",
        "rationaleSummary": ["V24ROW-000001 quadriceps 후보를 사용했습니다."],
        "routineBlocks": [
            {
                "order": 1,
                "exerciseId": "V24ROW-000001",
                "exerciseName": "레그프레스",
                "equipmentType": "MACHINE",
                "primaryMuscles": ["quadriceps"],
                "defaultRestSec": 90,
                "prescription": [{"setIndex": 1, "targetReps": 10, "targetRestSec": 90}],
                "exerciseRationale": "목표 근육의 수축과 이완을 의식하며 천천히 수행하세요.",
                "reasons": [],
                "boosts": [],
                "penalties": [],
            }
        ],
        "warnings": [],
    })
    row = {
        "scenarioId": "lower-back-legs-001",
        "title": "요추 통증",
        "input": {},
        "failurePatterns": [],
    }

    result = live.result_row(row, response, 123)

    assert "languageQualityFlags" in result
    assert "fallback_or_non_success" in result["languageQualityFlags"]
    assert "catalog_id_in_user_text" in result["languageQualityFlags"]
    assert "english_schema_token_in_user_text" in result["languageQualityFlags"]
    assert "generic_rationale" in result["languageQualityFlags"]


def test_live_result_row_marks_experience_contradiction_and_prevention_claim():
    response = _routine_response("machine_press")
    response.rationale_summary = ["초급자도 부상을 예방하는 데 적합한 구성입니다."]
    row = {
        "scenarioId": "experience-language-001",
        "title": "숙련도 설명 검증",
        "input": {"experienceLevel": "intermediate"},
        "failurePatterns": [],
    }

    result = live.result_row(row, response, 123)

    assert "experience_level_contradiction" in result["languageQualityFlags"]
    assert "unsupported_injury_prevention_claim" in result["languageQualityFlags"]


def test_live_result_row_marks_malformed_prone_t_raise_display_name():
    response = _routine_response("prone_t_raise")
    response.routine_blocks[0].exercise_name = "덤벨 라잉 프론 t 레이즈"

    result = live.result_row(
        {
            "scenarioId": "display-name-001",
            "title": "표시명 검증",
            "input": {},
            "failurePatterns": [],
        },
        response,
        123,
    )

    assert "malformed_exercise_display_name" in result["languageQualityFlags"]


def test_live_result_row_marks_broken_hex_encoding_token():
    response = _routine_response("machine_press")
    response.rationale_summary = ["스트<0xEB>레칭을 포함했습니다."]

    result = live.result_row(
        {
            "scenarioId": "encoding-quality-001",
            "title": "깨진 인코딩 검증",
            "input": {},
            "failurePatterns": [],
        },
        response,
        123,
    )

    assert "malformed_text_encoding" in result["languageQualityFlags"]


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


def test_run_live_captures_routine_telemetry_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "SessionLocal", lambda: _FakeDb())
    monkeypatch.setattr(live, "get_user_profile_context", lambda db, user_id: None)
    monkeypatch.setattr(live, "get_recent_sets", lambda db, user_id: [])

    def fake_generate(req, db, profile=None, recent_sets=None):  # noqa: ARG001
        live.routine_telemetry.emit_routine_quality_telemetry({
            "fallback_reason": "schemaError",
            "schema_validation_error_category": "missing_field",
            "schema_validation_field_names": ["exercises.0.sets"],
            "raw_output_recovery_failed_reason": "no_valid_json_object",
        })
        return _routine_response()

    monkeypatch.setattr(live, "generate_smart_routine", fake_generate)
    seed_path = tmp_path / "seed.jsonl"
    output_path = tmp_path / "run" / "gemma4-quality-seed-results-filled.json"
    _write_seed(seed_path, [_seed_row("s1")])

    payload = live.run_live(
        seed_path,
        "unit-telemetry",
        output_path=output_path,
        scenario_timeout_sec=0,
    )

    scenario = payload["scenarios"][0]
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert scenario["telemetryEventCount"] == 1
    assert scenario["telemetrySummary"]["fallback_reason"] == "schemaError"
    assert scenario["telemetrySummary"]["schema_validation_error_category"] == "missing_field"
    assert saved["scenarios"][0]["telemetrySummary"] == scenario["telemetrySummary"]


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


def test_preflight_warmup_writes_report_and_blocks_on_failure(tmp_path, monkeypatch):
    def fake_report(**_kwargs):
        return {
            "readiness_status": "failed_warmup_timeout",
            "warmup_latency_ms": None,
            "probe_latency_ms": None,
            "next_action": "restart_or_prewarm_ollama_then_retry",
        }

    monkeypatch.setattr(live.local_readiness, "build_readiness_report", fake_report)
    output_path = tmp_path / "preflight.json"

    with pytest.raises(SystemExit) as exc:
        live.run_preflight_warmup(
            base_url="http://127.0.0.1:11434",
            model="gemma4:latest",
            timeout_sec=30,
            warmup_timeout_sec=180,
            warmup_num_predict=1,
            output_path=output_path,
        )

    assert exc.value.code == 2
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["readiness_status"] == "failed_warmup_timeout"


def test_main_preflight_runs_before_live_and_writes_readiness(tmp_path, monkeypatch):
    seed_path = tmp_path / "seed.jsonl"
    _write_seed(seed_path, [_seed_row("s1")])
    calls = []

    def fake_report(**kwargs):
        calls.append(("preflight", kwargs))
        return {
            "readiness_status": "pass",
            "warmup_latency_ms": 9440,
            "probe_latency_ms": 1009,
            "next_action": "ready_for_routine_quality_eval",
        }

    def fake_run_live(seed_path_arg, run_id, **kwargs):
        calls.append(("run_live", {"seed_path": seed_path_arg, "run_id": run_id, **kwargs}))
        return live.build_payload(
            run_id=run_id,
            seed_path=Path(seed_path_arg),
            rows=[_seed_row("s1")],
            scenarios=[],
            failures=[],
        )

    monkeypatch.setattr(live.local_readiness, "build_readiness_report", fake_report)
    monkeypatch.setattr(live, "run_live", fake_run_live)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_gemma4_quality_seed_live.py",
            "--seed",
            str(seed_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--run-id",
            "unit-preflight",
            "--preflight-warmup",
        ],
    )

    live.main()

    assert calls[0][0] == "preflight"
    assert calls[1][0] == "run_live"
    readiness_path = tmp_path / "out" / "unit-preflight" / "local-llm-readiness-preflight.json"
    assert json.loads(readiness_path.read_text(encoding="utf-8"))["readiness_status"] == "pass"

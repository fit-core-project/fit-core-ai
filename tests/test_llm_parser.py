from __future__ import annotations

import json

import pytest

from engines.llm_parser import (
    SchemaRepairError,
    classify_validation_error,
    normalize_llm_response,
    parse_json_candidate,
    summarize_validation_error_shapes,
    trim_routine_to_time_budget,
    validate_and_repair_routine_output,
)
from engines.prescription.adjuncts import apply_adjunct_prescription_profile
from engines.schemas import LLMExercisePlan, LLMRoutineOutput


def test_time_trim_preserves_required_exercise_when_optional_exercise_can_be_removed():
    output = LLMRoutineOutput(
        total_estimated_time=60,
        summary_title="보호 슬롯 테스트",
        rationale_summary=[],
        warnings=[],
        exercises=[
            LLMExercisePlan(
                exercise_id="required_triceps",
                exercise_name="케이블 푸시다운",
                primary_muscles=["triceps"],
                movement_type="ISOLATION",
                target_reps=10,
                sets=4,
                rest_time_sec=180,
                exercise_rationale="필수 직접 보조 슬롯",
            ),
            LLMExercisePlan(
                exercise_id="optional_raise",
                exercise_name="레터럴 레이즈",
                primary_muscles=["front-deltoids"],
                movement_type="ISOLATION",
                target_reps=12,
                sets=4,
                rest_time_sec=180,
                exercise_rationale="선택 보조 슬롯",
            ),
        ],
    )

    assert trim_routine_to_time_budget(
        output,
        7,
        protected_exercise_ids={"required_triceps"},
    )
    assert [exercise.exercise_id for exercise in output.exercises] == ["required_triceps"]


def test_validation_set_cap_excludes_reviewed_timed_adjunct_blocks():
    main = LLMExercisePlan(
        exercise_id="machine_chest_press",
        exercise_name="머신 체스트 프레스",
        primary_muscles=["chest"],
        movement_type="COMPOUND",
        target_reps=8,
        sets=3,
        rest_time_sec=90,
        exercise_rationale="가슴 메인 운동",
    )
    accessory = LLMExercisePlan(
        exercise_id="cable_pushdown",
        exercise_name="케이블 푸시다운",
        primary_muscles=["triceps"],
        movement_type="ISOLATION",
        target_reps=12,
        sets=3,
        rest_time_sec=60,
        exercise_rationale="삼두 직접 보조 운동",
    )
    cooldown = LLMExercisePlan(
        exercise_id="doorway_chest_stretch",
        exercise_name="문틀 가슴 스트레치",
        primary_muscles=["chest"],
        movement_type="STATIC",
        target_reps=12,
        sets=3,
        rest_time_sec=60,
        exercise_rationale="쿨다운 스트레치",
    )
    apply_adjunct_prescription_profile(cooldown, "cooldown_stretch")
    output = LLMRoutineOutput(
        total_estimated_time=0,
        summary_title="테스트",
        rationale_summary=[],
        warnings=[],
        exercises=[main, accessory, cooldown],
    )
    candidates = [
        {
            "id": "machine_chest_press",
            "name_kr": "머신 체스트 프레스",
            "primary_muscle": "chest",
            "movement_type": "COMPOUND",
            "equipment_req": "MACHINE",
        },
        {
            "id": "cable_pushdown",
            "name_kr": "케이블 푸시다운",
            "primary_muscle": "triceps",
            "movement_type": "ISOLATION",
            "equipment_req": "CABLE",
        },
        {
            "id": "doorway_chest_stretch",
            "name_kr": "문틀 가슴 스트레치",
            "primary_muscle": "chest",
            "movement_type": "STATIC",
            "equipment_req": "BODYWEIGHT",
        },
    ]

    repaired, reason = validate_and_repair_routine_output(
        output,
        candidates,
        blocked_equipment=[],
        pain_areas=[],
        max_total_sets=6,
        time_available_min=60,
    )

    assert reason == "none"
    assert repaired is not None
    assert [exercise.sets for exercise in repaired.exercises[:2]] == [3, 3]
    assert repaired.exercises[2]._prescription_set_type == "timed"
    assert repaired.exercises[2].sets == 2


def _valid_payload() -> dict:
    return {
        "total_estimated_time": 45,
        "summary_title": "Test routine",
        "rationale_summary": ["reason"],
        "warnings": [],
        "exercises": [
            {
                "exercise_id": "bench",
                "exercise_name": "Bench Press",
                "target_reps": 8,
                "sets": 3,
                "rest_time_sec": 90,
                "exercise_rationale": "matches target muscle",
            }
        ],
    }


def _valid_json() -> str:
    return json.dumps(_valid_payload())


def test_valid_json_object_parses():
    result = parse_json_candidate(_valid_json())

    assert result.success is True
    assert result.data["summary_title"] == "Test routine"


def test_markdown_fenced_json_parses_with_subtype():
    result = parse_json_candidate(f"```json\n{_valid_json()}\n```")

    assert result.success is True
    assert result.subtype == "markdown_fence_wrapped"


def test_leading_trailing_prose_balanced_json_extracts():
    result = parse_json_candidate(f"Here is the routine:\n{_valid_json()}\nDone.")

    assert result.success is True
    assert result.subtype == "leading_or_trailing_prose"
    assert result.data["exercises"][0]["exercise_id"] == "bench"


def test_leading_invalid_example_object_does_not_block_later_valid_json():
    raw = f"Example shape: {{\"summary_title\": }}\nActual answer:\n{_valid_json()}"

    result = parse_json_candidate(raw)

    assert result.success is True
    assert result.subtype == "leading_or_trailing_prose"
    assert result.data["exercises"][0]["exercise_id"] == "bench"


def test_missing_json_value_repairs_to_null_then_defaults_can_apply():
    payload = _valid_payload()
    payload["exercises"][0]["target_weight_kg"] = None
    raw = json.dumps(payload).replace('"target_weight_kg": null', '"target_weight_kg": ')

    result = parse_json_candidate(raw)

    assert result.success is True
    assert result.json_decode_recovery_attempted is True
    assert result.json_decode_recovery_succeeded is True
    assert result.json_decode_recovery_strategy == "missing_json_values_filled_null"
    assert result.data["exercises"][0]["target_weight_kg"] is None


def test_missing_json_value_repair_accepts_camel_case_routine_keys():
    raw = """
    {
      "totalEstimatedTime": 45,
      "summaryTitle": "Test routine",
      "rationaleSummary": ["reason"],
      "warnings": [],
      "exercises": [
        {
          "exerciseId": "bench",
          "exerciseName": "Bench Press",
          "targetWeightKg": ,
          "targetReps": 8,
          "sets": 3,
          "restTimeSec": 90,
          "exerciseRationale": "matches target muscle"
        }
      ]
    }
    """

    result = parse_json_candidate(raw)

    assert result.success is True
    assert result.json_decode_recovery_strategy == "missing_json_values_filled_null"
    assert result.data["exercises"][0]["targetWeightKg"] is None


def test_leading_prose_truncated_json_tail_repairs():
    truncated = _valid_json()[:-1]

    result = parse_json_candidate(f"Here is the routine:\n{truncated}")

    assert result.success is True
    assert result.json_decode_recovery_attempted is True
    assert result.json_decode_recovery_succeeded is True
    assert result.json_decode_recovery_strategy == "truncated_json_tail_repaired"
    assert result.data["exercises"][0]["exercise_id"] == "bench"


def test_empty_response_returns_empty_response_subtype():
    result = parse_json_candidate("   ")

    assert result.success is False
    assert result.subtype == "empty_response"


def test_non_object_json_returns_non_object_json_subtype():
    result = parse_json_candidate("[1, 2, 3]")

    assert result.success is False
    assert result.subtype == "non_object_json"


def test_malformed_json_returns_json_decode_error_subtype():
    result = parse_json_candidate('{"summary_title": }')

    assert result.success is False
    assert result.subtype == "json_decode_error"


def test_normalize_parses_leading_trailing_prose():
    output = normalize_llm_response(f"Result:\n{_valid_json()}\nThanks")

    assert isinstance(output, LLMRoutineOutput)
    assert output.exercises[0].exercise_id == "bench"


def test_normalize_repairs_exercises_object_with_numeric_keys():
    payload = _valid_payload()
    payload["exercises"] = {
        "1": payload["exercises"][0],
        "2": {
            **payload["exercises"][0],
            "exercise_id": "row",
            "exercise_name": "Cable Row",
            "primary_muscles": "upper-back",
        },
    }
    payload["warnings"] = "의료진 허가 범위에서 진행하세요."

    output = normalize_llm_response(json.dumps(payload, ensure_ascii=False))

    assert [exercise.exercise_id for exercise in output.exercises] == ["bench", "row"]
    assert output.exercises[1].primary_muscles == ["upper-back"]
    assert output.warnings == ["의료진 허가 범위에서 진행하세요."]


def test_normalize_repairs_exercises_object_items_list():
    payload = _valid_payload()
    payload["exercises"] = {"items": payload["exercises"]}

    output = normalize_llm_response(json.dumps(payload))

    assert output.exercises[0].exercise_id == "bench"


def test_normalize_repairs_common_local_model_contract_aliases():
    payload = _valid_payload()
    original = payload.pop("exercises")[0]
    payload["selected_exercises"] = [{
        "id": original["exercise_id"],
        "name": original["exercise_name"],
        "reasoning": original["exercise_rationale"],
    }]
    payload["recommendation_summary"] = "의료진 허가 범위에서 보수적으로 구성했습니다."
    del payload["rationale_summary"]
    del payload["summary_title"]

    output = normalize_llm_response(json.dumps(payload, ensure_ascii=False))

    assert output.summary_title == "맞춤형 AI 루틴"
    assert output.rationale_summary == ["의료진 허가 범위에서 보수적으로 구성했습니다."]
    assert output.exercises[0].exercise_id == "bench"
    assert output.exercises[0].exercise_name == "Bench Press"
    assert output.exercises[0].exercise_rationale == "matches target muscle"


def test_normalize_repairs_local_model_camel_case_and_loose_field_types():
    payload = {
        "totalEstimatedTime": "45분",
        "summaryTitle": "초급 하체 루틴",
        "rationaleSummary": "초급자에게 안정적인 머신 운동을 우선했습니다.",
        "warnings": "중량은 컨디션에 맞게 조절하세요.",
        "exercises": [
            {
                "exerciseId": "leg_press",
                "exerciseName": "레그 프레스",
                "primaryMuscles": {"main": "quadriceps"},
                "targetReps": ["10회"],
                "sets": {"value": 3},
                "restTimeSec": "90초",
                "targetWeightKg": "bodyweight",
                "targetRir": "2",
                "exerciseRationale": "초급자에게 안정적인 머신 기반 메인 운동입니다.",
                "substitutionCandidates": ["문자열 후보는 버립니다."],
            },
            {
                "exerciseName": "머신 레그 컬",
                "targetReps": "12회",
                "sets": 3.0,
                "restTimeSec": ["75초"],
                "exerciseRationale": "햄스트링 보조 볼륨입니다.",
                "substitutionCandidates": {
                    "1": {
                        "exercise_id": "lying_leg_curl",
                        "exercise_name": "라잉 레그 컬",
                        "reason": "동일 주동근 대체 후보입니다.",
                    }
                },
            },
        ],
    }

    output = normalize_llm_response(json.dumps(payload, ensure_ascii=False))

    assert output.total_estimated_time == 45
    assert output.summary_title == "초급 하체 루틴"
    assert output.rationale_summary == ["초급자에게 안정적인 머신 운동을 우선했습니다."]
    assert output.warnings == ["중량은 컨디션에 맞게 조절하세요."]
    assert output.exercises[0].exercise_id == "leg_press"
    assert output.exercises[0].primary_muscles == ["quadriceps"]
    assert output.exercises[0].target_reps == 10
    assert output.exercises[0].sets == 3
    assert output.exercises[0].rest_time_sec == 90
    assert output.exercises[0].target_weight_kg is None
    assert output.exercises[0].substitution_candidates == []
    assert output.exercises[1].exercise_id == "머신_레그_컬"
    assert output.exercises[1].rest_time_sec == 75
    assert output.exercises[1].substitution_candidates[0].exercise_id == "lying_leg_curl"


def test_normalize_repairs_optional_text_fields_from_lists():
    payload = _valid_payload()
    payload["exercises"][0]["equipment_type"] = ["MACHINE", "CABLE"]
    payload["exercises"][0]["movement_type"] = ["COMPOUND"]
    payload["exercises"][0]["movement_pattern"] = {"pattern": "squat"}

    output = normalize_llm_response(json.dumps(payload))

    assert output.exercises[0].equipment_type == "MACHINE, CABLE"
    assert output.exercises[0].movement_type == "COMPOUND"
    assert output.exercises[0].movement_pattern == "squat"


def test_normalize_repairs_leading_prose_truncated_json_tail():
    output = normalize_llm_response(f"Result:\n{_valid_json()[:-1]}")

    assert isinstance(output, LLMRoutineOutput)
    assert output.exercises[0].exercise_id == "bench"


def test_missing_required_field_category_stores_field_names_only():
    payload = _valid_payload()
    del payload["exercises"][0]["exercise_name"]

    with pytest.raises(SchemaRepairError) as exc_info:
        normalize_llm_response(json.dumps(payload))

    error = exc_info.value
    assert error.parse_failure_subtype == "pydantic_validation_error"
    assert error.schema_validation_error_category == "missing_required_field"
    assert error.schema_validation_field_names == ["exercise_name"]
    assert "Bench Press" not in json.dumps(error.schema_validation_field_names)


def test_wrong_type_category_stores_field_names_only():
    payload = _valid_payload()
    payload["exercises"][0]["exercise_name"] = {"bad": "shape"}

    with pytest.raises(SchemaRepairError) as exc_info:
        normalize_llm_response(json.dumps(payload))

    error = exc_info.value
    assert error.schema_validation_error_category == "wrong_field_type"
    assert error.schema_validation_field_names == ["exercise_name"]


def test_schema_shape_summary_does_not_store_raw_exercise_values():
    payload = _valid_payload()
    payload["exercises"] = "Bench Press private raw value"

    with pytest.raises(SchemaRepairError) as exc_info:
        normalize_llm_response(json.dumps(payload))

    error = exc_info.value
    assert error.schema_validation_error_category == "invalid_list_shape"
    assert error.schema_validation_field_names == ["exercises"]
    assert error.schema_root_type == "dict"
    assert "exercises" in error.schema_top_level_keys
    assert error.schema_exercises_value_type == "str"
    assert error.schema_exercises_nested_keys == []
    assert error.schema_exercises_list_length is None
    assert error.schema_first_exercise_item_type is None
    assert error.schema_first_exercise_keys == []
    assert error.schema_first_exercise_field_types == {}
    assert error.schema_exercise_item_types == []
    assert error.schema_exercise_key_union == []
    assert error.schema_exercise_field_type_sets == {}
    assert error.schema_validation_loc_summaries == ["exercises"]
    assert error.schema_validation_error_types
    shape_payload = json.dumps(
        {
            "root": error.schema_root_type,
            "top": error.schema_top_level_keys,
            "exerciseType": error.schema_exercises_value_type,
            "nested": error.schema_exercises_nested_keys,
            "length": error.schema_exercises_list_length,
            "firstItem": error.schema_first_exercise_item_type,
            "firstKeys": error.schema_first_exercise_keys,
            "firstTypes": error.schema_first_exercise_field_types,
            "itemTypes": error.schema_exercise_item_types,
            "keyUnion": error.schema_exercise_key_union,
            "typeSets": error.schema_exercise_field_type_sets,
            "locs": error.schema_validation_loc_summaries,
            "errorTypes": error.schema_validation_error_types,
        },
        ensure_ascii=False,
    )
    assert "Bench Press" not in shape_payload
    assert "private raw value" not in shape_payload


def test_schema_shape_summary_includes_first_exercise_types_without_values():
    payload = _valid_payload()
    payload["exercises"][0]["exercise_name"] = {"private": "do not store"}
    payload["exercises"][0]["exercise_rationale"] = {"bad": "shape"}

    with pytest.raises(SchemaRepairError) as exc_info:
        normalize_llm_response(json.dumps(payload))

    error = exc_info.value
    assert error.schema_validation_error_category == "wrong_field_type"
    assert error.schema_exercises_value_type == "list"
    assert error.schema_exercises_list_length == 1
    assert error.schema_first_exercise_item_type == "dict"
    assert "exercise_name" in error.schema_first_exercise_keys
    assert error.schema_first_exercise_field_types["exercise_name"] == "dict"
    assert error.schema_first_exercise_field_types["exercise_rationale"] == "dict"
    assert error.schema_exercise_item_types == ["dict"]
    assert "exercise_name" in error.schema_exercise_key_union
    assert error.schema_exercise_field_type_sets["exercise_name"] == ["dict"]
    assert error.schema_exercise_field_type_sets["exercise_rationale"] == ["dict"]
    assert "exercises.*.exercise_name" in error.schema_validation_loc_summaries
    assert "exercises.*.exercise_rationale" in error.schema_validation_loc_summaries
    shape_payload = json.dumps({
        "firstKeys": error.schema_first_exercise_keys,
        "firstTypes": error.schema_first_exercise_field_types,
        "itemTypes": error.schema_exercise_item_types,
        "keyUnion": error.schema_exercise_key_union,
        "typeSets": error.schema_exercise_field_type_sets,
        "locs": error.schema_validation_loc_summaries,
        "errorTypes": error.schema_validation_error_types,
    }, ensure_ascii=False)
    assert "private" not in shape_payload
    assert "do not store" not in shape_payload


def test_classify_validation_error_does_not_expose_raw_message():
    payload = _valid_payload()
    payload["exercises"][0]["rest_time_sec"] = {"private": "do not store"}

    try:
        LLMRoutineOutput(**payload)
    except Exception as exc:
        category, fields = classify_validation_error(exc)
        locs, error_types = summarize_validation_error_shapes(exc)

    assert category == "wrong_field_type"
    assert fields == ["rest_time_sec"]
    assert "private" not in json.dumps(fields)
    assert locs == ["exercises.*.rest_time_sec"]
    assert error_types

from engines.fallback import build_routine_draft
from engines.korean_text_guard import _is_english_heavy
from engines.schemas import LLMExercisePlan, LLMRoutineOutput


def _has_hangul(value: str) -> bool:
    return any("\uac00" <= char <= "\ud7a3" for char in value)


def test_generated_response_user_facing_text_is_guarded_to_korean():
    output = LLMRoutineOutput(
        total_estimated_time=45,
        summary_title="Push Hypertrophy Routine",
        rationale_summary=[
            "The plan focuses on hypertrophy while avoiding shoulder strain.",
            "Cable and dumbbell accessories maintain triceps volume safely.",
        ],
        warnings=["Reduce load if shoulder discomfort appears during the session."],
        exercises=[
            LLMExercisePlan(
                exercise_id="cable_pushdown",
                exercise_name="트라이셉스 로프 푸시다운",
                movement_type="ISOLATION",
                primary_muscles=["triceps"],
                equipment_type="CABLE",
                target_reps=12,
                sets=3,
                rest_time_sec=75,
                exercise_rationale=(
                    "This is a triceps isolation movement selected for hypertrophy "
                    "because cable equipment is available."
                ),
            )
        ],
    )

    response = build_routine_draft(output, "success", "none", False)

    assert _has_hangul(response.summary_title)
    assert not _is_english_heavy(response.summary_title)
    assert all(_has_hangul(item) and not _is_english_heavy(item) for item in response.rationale_summary)
    assert all(_has_hangul(item) and not _is_english_heavy(item) for item in response.warnings)
    assert all(
        _has_hangul(block.exercise_rationale) and not _is_english_heavy(block.exercise_rationale)
        for block in response.routine_blocks
    )

    block = response.routine_blocks[0]
    assert block.exercise_id == "cable_pushdown"
    assert block.primary_muscles == ["triceps"]
    assert block.equipment_type == "CABLE"
    assert block.prescription[0].set_type == "working"


def test_korean_text_with_schema_tokens_is_preserved():
    output = LLMRoutineOutput(
        total_estimated_time=30,
        summary_title="푸시 근비대 루틴",
        rationale_summary=["BARBELL 제한을 고려해 chest 후보 중 안전한 운동을 선택했습니다."],
        warnings=["통증이 있으면 세트를 줄이세요."],
        exercises=[
            LLMExercisePlan(
                exercise_id="pushup",
                exercise_name="푸시업",
                movement_type="COMPOUND",
                primary_muscles=["chest"],
                equipment_type="BODYWEIGHT",
                target_reps=10,
                sets=2,
                rest_time_sec=75,
                exercise_rationale="BODYWEIGHT 후보이며 chest 목표에 맞아 선택했습니다.",
            )
        ],
    )

    response = build_routine_draft(output, "success", "none", False)

    assert response.summary_title == "푸시 근비대 루틴"
    assert response.rationale_summary == ["BARBELL 제한을 고려해 chest 후보 중 안전한 운동을 선택했습니다."]
    assert response.warnings == ["통증이 있으면 세트를 줄이세요."]
    assert response.routine_blocks[0].exercise_rationale == "BODYWEIGHT 후보이며 chest 목표에 맞아 선택했습니다."

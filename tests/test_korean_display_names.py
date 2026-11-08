"""Korean exercise names and code-token translation for user-visible routine text.

Contract: `exerciseName` is the Korean catalog name (`exercise_tier.name_kr`, falling back to
`name_en`), and free-text fields never show raw contract tokens such as `push`, `DOMS moderate`
or `MACHINE`. Codes in structured fields (`primaryMuscles`, `equipmentType`, ...) are unchanged.
"""
import pytest

from engines.fallback import build_routine_draft
from engines.korean_text_guard import translate_code_tokens
from engines.llm_parser import (
    is_guard_replacement_warning,
    validate_and_repair_routine_output,
)
from engines.routine_telemetry import count_guard_repairs
from engines.schemas import LLMExercisePlan, LLMRoutineOutput, LLMSubstitutionCandidate


@pytest.fixture
def push_candidates():
    return [
        {
            "id": "30",
            "name_kr": "바벨 벤치프레스",
            "name_en": "Barbell Bench Press",
            "primary_muscle": "chest",
            "equipment_req": "BARBELL",
            "pain_triggers": None,
            "movement_type": "COMPOUND",
        },
        {
            "id": "31",
            "name_kr": "덤벨 플랫 프레스",
            "name_en": "Dumbbell Flat Press",
            "primary_muscle": "chest",
            "equipment_req": "DUMBBELL",
            "pain_triggers": None,
            "movement_type": "COMPOUND",
        },
        {
            "id": "117",
            "name_kr": "",
            "name_en": "Push-up",
            "primary_muscle": "chest",
            "equipment_req": "BODYWEIGHT",
            "pain_triggers": None,
            "movement_type": "COMPOUND",
        },
    ]


def _plan(exercise_id: str, name: str, substitutions=()) -> LLMExercisePlan:
    return LLMExercisePlan(
        exercise_id=exercise_id,
        exercise_name=name,
        primary_muscles=["chest"],
        sets=3,
        target_reps=10,
        rest_time_sec=90,
        exercise_rationale="가슴 중심 운동입니다.",
        substitution_candidates=[
            LLMSubstitutionCandidate(exercise_id=sid, exercise_name=sname, reason="대체 가능")
            for sid, sname in substitutions
        ],
    )


def _output(*plans: LLMExercisePlan) -> LLMRoutineOutput:
    return LLMRoutineOutput(
        total_estimated_time=40,
        summary_title="테스트 루틴",
        rationale_summary=["테스트"],
        warnings=[],
        exercises=list(plans),
    )


def test_matched_candidate_name_is_overwritten_with_catalog_korean_name(push_candidates):
    result, reason = validate_and_repair_routine_output(
        _output(_plan("30", "Barbell Bench Press")), push_candidates, [], []
    )

    assert reason == "none"
    assert result.exercises[0].exercise_name == "바벨 벤치프레스"


def test_empty_name_kr_falls_back_to_name_en(push_candidates):
    result, _ = validate_and_repair_routine_output(
        _output(_plan("117", "푸시업?")), push_candidates, [], []
    )

    assert result.exercises[0].exercise_name == "Push-up"


def test_substitutions_are_mapped_by_id_and_unknown_ids_dropped(push_candidates):
    plan = _plan(
        "30",
        "Barbell Bench Press",
        substitutions=[("31", "Dumbbell Flat Press"), ("ghost_press", "Ghost Press"), ("30", "Self")],
    )
    result, _ = validate_and_repair_routine_output(_output(plan), push_candidates, [], [])

    subs = result.exercises[0].substitution_candidates
    assert [(s.exercise_id, s.exercise_name) for s in subs] == [("31", "덤벨 플랫 프레스")]


def test_guard_replacement_warning_is_korean_and_counted(push_candidates):
    result, reason = validate_and_repair_routine_output(
        _output(_plan("ghost_press", "Ghost Press")), push_candidates, [], []
    )

    assert reason == "none"
    replaced = result.exercises[0]
    assert replaced.exercise_id == "30"
    assert replaced.exercise_name == "바벨 벤치프레스"
    assert "guard repair" not in replaced.exercise_rationale
    assert result.warnings == ["조건에 맞지 않는 운동을 안전한 후보인 '바벨 벤치프레스' 운동으로 바꿨습니다."]
    assert count_guard_repairs(result.warnings) == 1


def test_guard_counter_still_reads_legacy_warning_text():
    assert is_guard_replacement_warning("Guard: 'ghost' -> 'bench' (constraint violation replaced)")
    assert not is_guard_replacement_warning("Guard: informational line without replacement")
    assert not is_guard_replacement_warning("Guard diagnostics: repairCount=1, trimmedSetCount=0")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("어깨 부담을 낮춘 Push 루틴", "어깨 부담을 낮춘 밀기 루틴"),
        (
            "오늘 선택한 push 부위를 기준으로 가슴과 삼두 중심 루틴을 구성했다.",
            "오늘 선택한 밀기 부위를 기준으로 가슴과 삼두 중심 루틴을 구성했다.",
        ),
        (
            "가슴 DOMS moderate를 반영해 총 볼륨을 줄이고 어깨 부담이 큰 동작을 피했다.",
            "가슴 근육통 중간을 반영해 총 볼륨을 줄이고 어깨 부담이 큰 동작을 피했다.",
        ),
        (
            "사용 불가 장비 MACHINE을 제외하고 바벨, 덤벨, 맨몸 중심으로 구성했다.",
            "사용 불가 장비 머신을 제외하고 바벨, 덤벨, 맨몸 중심으로 구성했다.",
        ),
        (
            "COMPOUND 유형이고 primary_muscles가 chest인 후보입니다.",
            "복합 운동 유형이고 주동근이 가슴인 후보입니다.",
        ),
        ("SMITH_MACHINE 대신 BODYWEIGHT 후보", "스미스 머신 대신 맨몸 후보"),
        ("upper-back과 front-deltoids를 보완", "등과 전면 어깨를 보완"),
        ("DUMBBELL으로 대체", "덤벨로 대체"),
        ("readinessLevel low라서 세트를 줄였습니다", "컨디션 피곤함이라서 세트를 줄였습니다"),
        ("DOMS 2 반영", "근육통 2단계 반영"),
    ],
)
def test_translate_code_tokens(raw, expected):
    assert translate_code_tokens(raw) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Push-up 대신 벤치를 사용",  # exercise name with a hyphen is not a split code
        "ID 117 운동, RIR 2 유지",
        "pushdown 위주",  # token must be a whole word
        "바벨 벤치프레스",
    ],
)
def test_translate_code_tokens_leaves_non_tokens(text):
    assert translate_code_tokens(text) == text


def test_built_response_has_no_raw_tokens_in_user_text_but_keeps_codes_in_fields():
    plan = _plan("30", "바벨 벤치프레스", substitutions=[("31", "덤벨 플랫 프레스")])
    plan.exercise_rationale = "가슴 중심 push 루틴의 BARBELL 주운동입니다."
    plan.equipment_type = "BARBELL"
    plan.substitution_candidates[0].reason = "BARBELL 대신 DUMBBELL로 수평 밀기를 유지합니다."
    output = _output(plan)
    output.summary_title = "어깨 부담을 낮춘 Push 루틴"
    output.rationale_summary = ["가슴 DOMS moderate를 반영했습니다."]

    response = build_routine_draft(output, "success", "none", False)

    assert response.summary_title == "어깨 부담을 낮춘 밀기 루틴"
    assert response.rationale_summary == ["가슴 근육통 중간을 반영했습니다."]
    block = response.routine_blocks[0]
    assert block.exercise_rationale == "가슴 중심 밀기 루틴의 바벨 주운동입니다."
    assert block.substitution_candidates[0].reason == "바벨 대신 덤벨로 수평 밀기를 유지합니다."
    # Structured fields keep SSOT codes.
    assert block.equipment_type == "BARBELL"
    assert block.primary_muscles == ["chest"]

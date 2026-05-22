"""공통 fixture"""
import pytest
from engines.schemas import (
    LLMExercisePlan,
    LLMRoutineOutput,
    LLMSubstitutionCandidate,
    RoutineRequest,
    UserProfileContext,
)


@pytest.fixture
def sample_request():
    return RoutineRequest(
        user_id="test-user-001",
        target_split_label="push",
        readiness_level="normal",
        time_available_min=60,
        pain_areas=[],
        doms_data={"chest": 1},
        equipment=[],
    )


@pytest.fixture
def sample_profile():
    return UserProfileContext(
        goal_type="hypertrophy",
        split_type="PPL",
        split_label="push",
        experience_level="intermediate",
        strength_baseline={"bench_press": {"weight_kg": 80, "reps": 5}},
        equipment_access=["BARBELL", "DUMBBELL", "CABLE"],
        pain_areas=[],
    )


@pytest.fixture
def sample_llm_output():
    return LLMRoutineOutput(
        total_estimated_time=55,
        summary_title="Push 세션 추천 루틴",
        rationale_summary=["상체 밀기 패턴 중심", "DOMS 고려 볼륨 조정"],
        warnings=[],
        exercises=[
            LLMExercisePlan(
                exercise_id="barbell_bench_press",
                exercise_name="바벨 벤치프레스",
                movement_pattern="horizontalPush",
        primary_muscles=["chest"],
                equipment_type="barbell",
                target_weight_kg=80.0,
                target_reps=8,
                sets=4,
                rest_time_sec=120,
                target_rir=2,
                exercise_rationale="주요 복합 운동",
                substitution_candidates=[
                    LLMSubstitutionCandidate(
                        exercise_id="dumbbell_bench_press",
                        exercise_name="덤벨 벤치프레스",
                        reason="바벨 없을 때 대체",
                    )
                ],
            ),
        ],
    )


@pytest.fixture
def mock_candidates():
    return [
        {
            "id": "barbell_bench_press",
            "name_kr": "바벨 벤치프레스",
            "name_en": "Barbell Bench Press",
        "primary_muscle": "chest",
            "secondary_muscle": "triceps",
            "equipment_req": "BARBELL",
            "difficulty_tier": 3,
            "efficiency_tier": 5,
            "pain_triggers": None,
            "movement_type": "COMPOUND",
        },
        {
            "id": "dumbbell_fly",
            "name_kr": "덤벨 플라이",
            "name_en": "Dumbbell Fly",
        "primary_muscle": "chest",
            "secondary_muscle": None,
            "equipment_req": "DUMBBELL",
            "difficulty_tier": 2,
            "efficiency_tier": 3,
            "pain_triggers": None,
            "movement_type": "ISOLATION",
        },
        {
            "id": "pushup",
            "name_kr": "푸시업",
            "name_en": "Push-up",
        "primary_muscle": "chest",
            "secondary_muscle": "triceps",
            "equipment_req": "BODYWEIGHT",
            "difficulty_tier": 1,
            "efficiency_tier": 2,
            "pain_triggers": None,
            "movement_type": "COMPOUND",
        },
    ]


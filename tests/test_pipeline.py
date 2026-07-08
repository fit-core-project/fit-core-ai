"""
generate_smart_routine 통합 테스트
LLM과 DB를 mock해서 AI 파이프라인 경로를 검증한다.

[Mock 전략]
LangChain의 `prompt | structured_llm` 체인은 `|` 연산자로 RunnableSequence를 만든다.
MagicMock은 Runnable이 아니므로 RunnableLambda로 감싸져 .invoke() 대신 직접 호출된다.
→ with_structured_output 반환값을 RunnableLambda로 만들어야 체인 invoke가 올바로 동작한다.
"""
import asyncio
from unittest.mock import MagicMock, patch

import httpx
import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda
from sqlalchemy.orm import Session

from engines.db_queries import get_user_profile_context
from engines.routine_pipeline import generate_smart_routine
from engines.schemas import (
    LLMExercisePlan,
    LLMRoutineOutput,
    PainAreaEntry,
    RecentSetRecord,
    RoutineRequest,
)


def _make_db_mock():
    return MagicMock(spec=Session)


def _make_llm_mock(return_value=None, side_effect=None):
    """get_llm() 반환값 mock을 만든다. with_structured_output은 RunnableLambda를 반환한다."""
    mock_llm = MagicMock()

    if side_effect is not None:
        def _raise(_):
            raise side_effect
        structured = RunnableLambda(_raise)
    else:
        structured = RunnableLambda(lambda _: return_value)

    mock_llm.with_structured_output.return_value = structured
    return mock_llm


class _Row:
    def __init__(self, mapping):
        self._mapping = mapping


class _FetchOne:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class TestProfileContext:
    def test_strength_baseline_list_from_backend_is_indexed_by_exercise_id_and_name(self):
        db = _make_db_mock()
        db.execute.return_value = _FetchOne(_Row({
            "goal_type": "strength",
            "split_type": "pushPullLegs",
            "split_label": "push",
            "experience_level": "intermediate",
            "strength_baseline": [
                {
                    "exerciseId": "30",
                    "exerciseNameSnapshot": "Barbell Bench Press",
                    "workingWeightKg": 100,
                    "reps": 5,
                }
            ],
            "equipment_access": [],
            "pain_areas": [],
        }))

        profile = get_user_profile_context(db, "user-1")

        assert profile is not None
        assert profile.strength_baseline["30"]["weight_kg"] == 100.0
        assert profile.strength_baseline["Barbell Bench Press"]["reps"] == 5


class TestHappyPath:
    def test_success_returns_correct_status(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        db = _make_db_mock()
        mock_llm = _make_llm_mock(return_value=sample_llm_output)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                sample_request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.generation_status == "success"
        assert result.is_fallback is False
        assert result.status_reason_code == "none"
        assert len(result.routine_blocks) == 1
        assert result.routine_blocks[0].exercise_id == "barbell_bench_press"

    def test_professional_clearance_warning_is_deterministic_on_success(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        db = _make_db_mock()
        request = sample_request.model_copy(update={
            "condition_policies": [
                {
                    "policyType": "surgeryHistory",
                    "bodyPart": "shoulder",
                    "status": "needs_clearance",
                    "professionalClearance": False,
                }
            ]
        })
        llm_output = sample_llm_output.model_copy(update={"warnings": []})
        mock_llm = _make_llm_mock(return_value=llm_output)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.generation_status == "success"
        assert any("의료진 허가" in warning for warning in result.warnings)

    def test_professional_clearance_warning_is_deterministic_on_fallback(
        self, sample_request, sample_profile, mock_candidates
    ):
        db = _make_db_mock()
        request = sample_request.model_copy(update={
            "condition_policies": [
                {
                    "policyType": "surgeryHistory",
                    "bodyPart": "shoulder",
                    "status": "needs_clearance",
                    "professionalClearance": False,
                }
            ]
        })
        mock_llm = _make_llm_mock(side_effect=asyncio.TimeoutError())

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.generation_status == "fallback"
        assert any("의료진 허가" in warning for warning in result.warnings)

    def test_low_readiness_extreme_pain_context_does_not_force_normal_routine(self, sample_profile, mock_candidates):
        db = _make_db_mock()
        request = RoutineRequest(
            user_id="extreme-pain-user",
            target_split_label="full_body",
            readiness_level="low",
            time_available_min=45,
            pain_areas=[
                PainAreaEntry(body_part="lower-back"),
                PainAreaEntry(body_part="knees"),
                PainAreaEntry(body_part="front-deltoids"),
                PainAreaEntry(body_part="chest"),
                PainAreaEntry(body_part="upper-back"),
                PainAreaEntry(body_part="quadriceps"),
                PainAreaEntry(body_part="hamstring"),
            ],
            doms_data={},
            equipment=[],
            goal="hypertrophy",
        )

        with (
            patch("engines.routine_pipeline.get_llm") as get_llm_mock,
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                request, db, profile=sample_profile, recent_sets=[]
            )

        get_llm_mock.assert_not_called()
        assert result.generation_status == "failed"
        assert result.status_reason_code == "emptyCandidate"
        assert result.routine_blocks == []

    def test_success_routine_blocks_have_prescriptions(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        db = _make_db_mock()
        mock_llm = _make_llm_mock(return_value=sample_llm_output)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                sample_request, db, profile=sample_profile, recent_sets=[]
            )

        block = result.routine_blocks[0]
        assert len(block.prescription) == sample_llm_output.exercises[0].sets - 1
        assert block.prescription[0].set_index == 1

    def test_success_routine_reorders_compound_large_muscle_before_isolation(
        self, sample_request, sample_profile, mock_candidates
    ):
        db = _make_db_mock()
        output = LLMRoutineOutput(
            total_estimated_time=50,
            summary_title="Reordered push",
            rationale_summary=["case"],
            warnings=[],
            exercises=[
                LLMExercisePlan(
                    exercise_id="dumbbell_fly",
                    exercise_name="Dumbbell Fly",
                    movement_type="ISOLATION",
                    primary_muscles=["chest"],
                    equipment_type="DUMBBELL",
                    target_reps=12,
                    sets=3,
                    rest_time_sec=75,
                    exercise_rationale="accessory",
                ),
                LLMExercisePlan(
                    exercise_id="barbell_bench_press",
                    exercise_name="Barbell Bench Press",
                    movement_type="COMPOUND",
                    primary_muscles=["chest"],
                    equipment_type="BARBELL",
                    target_reps=8,
                    sets=3,
                    rest_time_sec=120,
                    exercise_rationale="main",
                ),
            ],
        )
        mock_llm = _make_llm_mock(return_value=output)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                sample_request.model_copy(update={"doms_data": {}}),
                db,
                profile=sample_profile,
                recent_sets=[],
            )

        assert result.routine_blocks[0].exercise_id == "barbell_bench_press"
        assert result.routine_blocks[0].order == 1
        assert result.routine_blocks[1].exercise_id == "dumbbell_fly"
        assert result.routine_blocks[1].order == 2

    def test_invalid_candidate_is_repaired_to_safe_candidate(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        db = _make_db_mock()
        hallucinated = sample_llm_output.model_copy(deep=True)
        hallucinated.exercises[0].exercise_id = "ghost_press"
        mock_llm = _make_llm_mock(return_value=hallucinated)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                sample_request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.generation_status == "success"
        assert result.routine_blocks[0].exercise_id == "barbell_bench_press"
        assert any("replaced" in warning for warning in result.warnings)

    def test_too_many_invalid_candidates_fall_back(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        db = _make_db_mock()
        bad = sample_llm_output.model_copy(deep=True)
        bad.exercises = [
            bad.exercises[0].model_copy(update={"exercise_id": f"ghost_{i}"})
            for i in range(5)
        ]
        mock_llm = _make_llm_mock(return_value=bad)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                sample_request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.generation_status == "fallback"
        assert result.is_fallback is True

    def test_recent_sets_override_llm_weight_and_reps(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        db = _make_db_mock()
        mock_llm = _make_llm_mock(return_value=sample_llm_output)
        recent_sets = [
            RecentSetRecord(
                exercise_name=sample_llm_output.exercises[0].exercise_name,
                weight_kg=100,
                reps=5,
            )
        ]

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                sample_request, db, profile=sample_profile, recent_sets=recent_sets
            )

        first_set = result.routine_blocks[0].prescription[0]
        assert first_set.target_reps == 8
        assert first_set.target_weight_kg == 85.0
        assert first_set.target_rest_sec == 120

    def test_recent_sets_match_by_exercise_id_before_name(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        db = _make_db_mock()
        mock_llm = _make_llm_mock(return_value=sample_llm_output)
        recent_sets = [
            RecentSetRecord(
                exercise_id=sample_llm_output.exercises[0].exercise_id,
                exercise_name="Unrelated Snapshot",
                weight_kg=100,
                reps=5,
            )
        ]

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                sample_request, db, profile=sample_profile, recent_sets=recent_sets
            )

        assert result.routine_blocks[0].prescription[0].target_weight_kg == 85.0

    def test_fat_loss_goal_uses_compound_conditioning_reps(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        db = _make_db_mock()
        mock_llm = _make_llm_mock(return_value=sample_llm_output)
        request = sample_request.model_copy(update={"goal": "fatLoss", "doms_data": {}})

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.routine_blocks[0].prescription[0].target_reps == 12

    def test_low_readiness_reduces_compound_sets_and_raises_rir(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        db = _make_db_mock()
        request = sample_request.model_copy(update={"readiness_level": "low", "doms_data": {}})
        output = sample_llm_output.model_copy(deep=True)
        output.exercises[0].sets = 4
        output.exercises[0].target_rir = 2
        mock_llm = _make_llm_mock(return_value=output)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                request, db, profile=sample_profile, recent_sets=[]
            )

        assert len(result.routine_blocks[0].prescription) == 3
        assert result.routine_blocks[0].prescription[0].target_rir == 4

    def test_high_readiness_only_lowers_rir(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        db = _make_db_mock()
        request = sample_request.model_copy(update={"readiness_level": "high", "doms_data": {}})
        output = sample_llm_output.model_copy(deep=True)
        output.exercises[0].sets = 4
        output.exercises[0].target_rir = 2
        mock_llm = _make_llm_mock(return_value=output)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                request, db, profile=sample_profile, recent_sets=[]
            )

        assert len(result.routine_blocks[0].prescription) == 4
        assert result.routine_blocks[0].prescription[0].target_rir == 1

    def test_pain_trigger_candidate_is_repaired_to_safe_candidate(
        self, sample_request, sample_profile
    ):
        db = _make_db_mock()
        request = sample_request.model_copy(update={
            "pain_areas": [PainAreaEntry(body_part="shoulder", side="left", severity="mild")],
            "doms_data": {},
        })
        candidates = [
            {
                "id": "painful_press",
                "name_kr": "Painful Press",
                "name_en": "Painful Press",
                "primary_muscle": "chest",
                "secondary_muscle": "front-deltoids",
                "equipment_req": "DUMBBELL",
                "difficulty_tier": 3,
                "efficiency_tier": 5,
                "pain_triggers": "shoulder",
                "movement_type": "COMPOUND",
            },
            {
                "id": "safe_pushup",
                "name_kr": "Safe Push-up",
                "name_en": "Safe Push-up",
                "primary_muscle": "chest",
                "secondary_muscle": "triceps",
                "equipment_req": "BODYWEIGHT",
                "difficulty_tier": 1,
                "efficiency_tier": 3,
                "pain_triggers": None,
                "movement_type": "COMPOUND",
            },
        ]
        output = LLMRoutineOutput(
            total_estimated_time=30,
            summary_title="Pain repair",
            rationale_summary=["case"],
            warnings=[],
            exercises=[
                LLMExercisePlan(
                    exercise_id="painful_press",
                    exercise_name="Painful Press",
                    movement_type="COMPOUND",
                    primary_muscles=["chest"],
                    equipment_type="DUMBBELL",
                    target_reps=8,
                    sets=3,
                    rest_time_sec=120,
                    target_rir=2,
                    exercise_rationale="case",
                ),
            ],
        )
        mock_llm = _make_llm_mock(return_value=output)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=candidates),
        ):
            result = generate_smart_routine(
                request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.generation_status == "success"
        assert result.routine_blocks[0].exercise_id == "safe_pushup"
        assert result.routine_blocks[0].prescription[0].target_weight_kg is None

    def test_big_four_strength_baseline_seeds_related_target_weight(
        self, sample_request, sample_profile
    ):
        db = _make_db_mock()
        request = sample_request.model_copy(update={
            "target_split_label": "legs",
            "target_muscles": ["quadriceps"],
            "doms_data": {},
            "goal": "hypertrophy",
        })
        profile = sample_profile.model_copy(update={
            "strength_baseline": {
                "98": {
                    "exercise_id": "98",
                    "exercise_name_snapshot": "Back Squat",
                    "weight_kg": 170,
                    "reps": 1,
                }
            }
        })
        candidates = [
            {
                "id": "leg_press",
                "name_kr": "Leg Press",
                "name_en": "Leg Press",
                "primary_muscle": "quadriceps",
                "secondary_muscle": "gluteal",
                "equipment_req": "MACHINE",
                "difficulty_tier": 2,
                "efficiency_tier": 4,
                "pain_triggers": None,
                "movement_type": "COMPOUND",
            }
        ]
        output = LLMRoutineOutput(
            total_estimated_time=40,
            summary_title="Legs",
            rationale_summary=["case"],
            warnings=[],
            exercises=[
                LLMExercisePlan(
                    exercise_id="leg_press",
                    exercise_name="Leg Press",
                    movement_type="COMPOUND",
                    primary_muscles=["quadriceps"],
                    equipment_type="MACHINE",
                    target_weight_kg=None,
                    target_reps=8,
                    sets=3,
                    rest_time_sec=120,
                    target_rir=2,
                    exercise_rationale="quadriceps target",
                ),
            ],
        )
        mock_llm = _make_llm_mock(return_value=output)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=candidates),
        ):
            result = generate_smart_routine(
                request, db, profile=profile, recent_sets=[]
            )

        assert result.generation_status == "success"
        assert result.routine_blocks[0].prescription[0].target_weight_kg == 80.0

    def test_push_split_caps_accessory_isolation_sets(
        self, sample_request, sample_profile, mock_candidates
    ):
        db = _make_db_mock()
        candidates = mock_candidates + [
            {
                "id": "cable_pushdown",
                "name_kr": "Cable Pushdown",
                "name_en": "Cable Pushdown",
                "primary_muscle": "triceps",
                "secondary_muscle": None,
                "equipment_req": "CABLE",
                "difficulty_tier": 2,
                "efficiency_tier": 4,
                "pain_triggers": "triceps",
                "movement_type": "ISOLATION",
            }
        ]
        output = LLMRoutineOutput(
            total_estimated_time=50,
            summary_title="Push accessory cap",
            rationale_summary=["case"],
            warnings=[],
            exercises=[
                LLMExercisePlan(
                    exercise_id="barbell_bench_press",
                    exercise_name="Barbell Bench Press",
                    movement_type="COMPOUND",
                    primary_muscles=["chest"],
                    equipment_type="BARBELL",
                    target_reps=8,
                    sets=3,
                    rest_time_sec=90,
                    target_rir=2,
                    exercise_rationale="main",
                ),
                LLMExercisePlan(
                    exercise_id="cable_pushdown",
                    exercise_name="Cable Pushdown",
                    movement_type="ISOLATION",
                    primary_muscles=["triceps"],
                    equipment_type="CABLE",
                    target_reps=10,
                    sets=4,
                    rest_time_sec=90,
                    target_rir=2,
                    exercise_rationale="accessory",
                ),
            ],
        )
        mock_llm = _make_llm_mock(return_value=output)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=candidates),
        ):
            result = generate_smart_routine(
                sample_request.model_copy(update={"doms_data": {}}),
                db,
                profile=sample_profile,
                recent_sets=[],
            )

        pushdown = next(block for block in result.routine_blocks if block.exercise_id == "cable_pushdown")
        assert len(pushdown.prescription) == 2

    def test_arm_split_allows_accessory_muscle_as_main_volume(
        self, sample_request, sample_profile
    ):
        db = _make_db_mock()
        request = sample_request.model_copy(update={"target_split_label": "arm", "doms_data": {}})
        candidates = [
            {
                "id": "cable_pushdown",
                "name_kr": "Cable Pushdown",
                "name_en": "Cable Pushdown",
                "primary_muscle": "triceps",
                "secondary_muscle": None,
                "equipment_req": "CABLE",
                "difficulty_tier": 2,
                "efficiency_tier": 4,
                "pain_triggers": "triceps",
                "movement_type": "ISOLATION",
            }
        ]
        output = LLMRoutineOutput(
            total_estimated_time=30,
            summary_title="Arm main",
            rationale_summary=["case"],
            warnings=[],
            exercises=[
                LLMExercisePlan(
                    exercise_id="cable_pushdown",
                    exercise_name="Cable Pushdown",
                    movement_type="ISOLATION",
                    primary_muscles=["triceps"],
                    equipment_type="CABLE",
                    target_reps=10,
                    sets=4,
                    rest_time_sec=90,
                    target_rir=2,
                    exercise_rationale="main arm work",
                ),
            ],
        )
        mock_llm = _make_llm_mock(return_value=output)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=candidates),
        ):
            result = generate_smart_routine(
                request, db, profile=sample_profile, recent_sets=[]
            )

        assert len(result.routine_blocks[0].prescription) == 4

    def test_moderate_doms_clamps_sets(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        db = _make_db_mock()
        request = sample_request.model_copy(update={"doms_data": {"chest": 2}})
        mock_llm = _make_llm_mock(return_value=sample_llm_output)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.generation_status == "success"
        assert len(result.routine_blocks[0].prescription) == 2

    def test_severe_doms_forces_fallback(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        db = _make_db_mock()
        request = sample_request.model_copy(update={"doms_data": {"chest": 3}})
        mock_llm = _make_llm_mock(return_value=sample_llm_output)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.generation_status == "failed"
        assert result.status_reason_code == "emptyCandidate"

    def test_over_budget_output_is_trimmed_before_fallback(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        db = _make_db_mock()
        bad = sample_llm_output.model_copy(deep=True)
        bad.exercises[0].sets = 20
        bad.exercises[0].rest_time_sec = 300
        mock_llm = _make_llm_mock(return_value=bad)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                sample_request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.generation_status == "success"
        assert result.total_estimated_time <= sample_request.time_available_min
        assert len(result.routine_blocks[0].prescription) < 20

    def test_short_time_realistic_time_model_trims_before_fallback(
        self, sample_request, sample_profile, mock_candidates
    ):
        db = _make_db_mock()
        request = sample_request.model_copy(update={"time_available_min": 30, "doms_data": {}})
        output = LLMRoutineOutput(
            total_estimated_time=30,
            summary_title="Short push",
            rationale_summary=["case"],
            warnings=[],
            exercises=[
                LLMExercisePlan(
                    exercise_id="barbell_bench_press",
                    exercise_name="Barbell Bench Press",
                    movement_type="COMPOUND",
                    primary_muscles=["chest"],
                    equipment_type="BARBELL",
                    target_reps=8,
                    sets=4,
                    rest_time_sec=120,
                    target_rir=2,
                    exercise_rationale="case",
                ),
                LLMExercisePlan(
                    exercise_id="pushup",
                    exercise_name="Push-up",
                    movement_type="COMPOUND",
                    primary_muscles=["chest"],
                    equipment_type="BODYWEIGHT",
                    target_reps=8,
                    sets=3,
                    rest_time_sec=120,
                    target_rir=2,
                    exercise_rationale="case",
                ),
                LLMExercisePlan(
                    exercise_id="dumbbell_fly",
                    exercise_name="Dumbbell Fly",
                    movement_type="ISOLATION",
                    primary_muscles=["chest"],
                    equipment_type="DUMBBELL",
                    target_reps=8,
                    sets=3,
                    rest_time_sec=120,
                    target_rir=2,
                    exercise_rationale="case",
                ),
            ],
        )
        mock_llm = _make_llm_mock(return_value=output)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.generation_status == "success"
        assert result.total_estimated_time <= request.time_available_min


class TestTimeoutFallback:
    def test_timeout_triggers_fallback(
        self, sample_request, sample_profile, mock_candidates
    ):
        db = _make_db_mock()
        mock_llm = _make_llm_mock(side_effect=httpx.TimeoutException("timeout"))

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                sample_request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.is_fallback is True
        assert result.generation_status == "fallback"
        assert result.status_reason_code == "llmTimeout"

    def test_asyncio_timeout_triggers_fallback(
        self, sample_request, sample_profile, mock_candidates
    ):
        db = _make_db_mock()
        mock_llm = _make_llm_mock(side_effect=asyncio.TimeoutError())

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                sample_request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.status_reason_code == "llmTimeout"


class TestSchemaErrorRecovery:
    def test_schema_error_with_valid_raw_text_recovers(
        self, sample_request, sample_profile, sample_llm_output, mock_candidates
    ):
        """OutputParserException + llm_output 속성에 유효한 JSON → normalize 복구 성공"""
        db = _make_db_mock()
        valid_raw = sample_llm_output.model_dump_json()

        parser_exc = OutputParserException("schema fail")
        parser_exc.llm_output = valid_raw

        mock_llm = _make_llm_mock(side_effect=parser_exc)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                sample_request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.generation_status == "success"
        assert result.is_fallback is False

    def test_schema_error_unrecoverable_falls_back(
        self, sample_request, sample_profile, mock_candidates
    ):
        """OutputParserException + 복구 불가 raw 텍스트 + LLM 재호출도 실패 → fallback"""
        db = _make_db_mock()

        parser_exc = OutputParserException("schema fail")
        parser_exc.llm_output = "완전히 깨진 JSON !!!"

        mock_llm = _make_llm_mock(side_effect=parser_exc)
        # 비구조화 재호출도 실패
        mock_llm.invoke.side_effect = RuntimeError("LLM도 죽음")

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                sample_request, db, profile=sample_profile, recent_sets=[]
            )

        assert result.is_fallback is True
        assert result.generation_status == "fallback"


class TestNoProfile:
    def test_works_without_profile(self, sample_request, sample_llm_output, mock_candidates):
        """profile=None이어도 파이프라인이 동작해야 한다"""
        db = _make_db_mock()
        mock_llm = _make_llm_mock(return_value=sample_llm_output)

        with (
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
        ):
            result = generate_smart_routine(
                sample_request, db, profile=None, recent_sets=[]
            )

        assert result.generation_status == "success"

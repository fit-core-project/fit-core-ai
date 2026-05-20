"""generate_fallback_routine — DB/LLM 없이 순수 로직 테스트"""
import pytest

from engines.routine_engine import RecentSetRecord, RoutineRequest, generate_fallback_routine


@pytest.fixture
def push_request():
    return RoutineRequest(
        user_id="test-user",
        target_split_label="push",
        time_available_min=60,
        pain_areas=[],
        doms_data={},
        equipment=[],
    )


class TestFallbackEmptyCandidates:
    def test_returns_failed_status_with_empty_candidate(self, push_request):
        result = generate_fallback_routine(push_request, candidates=[], max_total_sets=15)
        assert result.generation_status == "failed"
        assert result.status_reason_code == "emptyCandidate"
        assert result.is_fallback is False

    def test_empty_blocks_with_warning(self, push_request):
        result = generate_fallback_routine(push_request, candidates=[], max_total_sets=15)
        assert result.routine_blocks == []
        assert len(result.warnings) > 0


class TestFallbackDomsLogic:
    def test_doms_level3_skips_exercise(self, push_request, mock_candidates):
        doms_db = {"chest": 3}
        result = generate_fallback_routine(
            push_request, candidates=mock_candidates, max_total_sets=15, doms_db=doms_db
        )
        exercise_ids = [b.exercise_id for b in result.routine_blocks]
        assert "barbell_bench_press" not in exercise_ids
        assert "dumbbell_fly" not in exercise_ids

    def test_doms_level2_limits_sets_to_2(self, push_request, mock_candidates):
        doms_db = {"chest": 2}
        result = generate_fallback_routine(
            push_request, candidates=mock_candidates, max_total_sets=15, doms_db=doms_db
        )
        chest_blocks = [b for b in result.routine_blocks if "chest" in b.primary_muscles]
        assert chest_blocks
        for block in chest_blocks:
            assert len(block.prescription) == 1

    def test_doms_level1_reduces_sets_by_1(self, push_request, mock_candidates):
        doms_db = {"chest": 1}
        result = generate_fallback_routine(
            push_request, candidates=mock_candidates, max_total_sets=15, doms_db=doms_db, goal="hypertrophy"
        )
        # HYPERTROPHY 기본 3세트 → DOMS 1이면 max(1, 3-1)=2
        for block in result.routine_blocks:
            assert len(block.prescription) <= 2

    def test_chest_doms_level3_removes_all_chest_blocks(self, push_request, mock_candidates):
        result = generate_fallback_routine(
            push_request,
            candidates=mock_candidates,
            max_total_sets=15,
            doms_db={"chest": 3},
        )
        assert all("chest" not in block.primary_muscles for block in result.routine_blocks)


class TestFallbackMaxSets:
    def test_total_sets_does_not_exceed_max(self, push_request, mock_candidates):
        max_sets = 5
        result = generate_fallback_routine(
            push_request, candidates=mock_candidates, max_total_sets=max_sets
        )
        total = sum(len(b.prescription) for b in result.routine_blocks)
        assert total <= max_sets


class TestFallbackGoalParams:
    def test_strength_goal_uses_movement_specific_reps(self, push_request, mock_candidates):
        result = generate_fallback_routine(
            push_request, candidates=mock_candidates, max_total_sets=20, goal="strength"
        )
        compound = next(block for block in result.routine_blocks if block.exercise_id == "barbell_bench_press")
        isolation = next(block for block in result.routine_blocks if block.exercise_id == "dumbbell_fly")

        assert compound.prescription[0].target_reps == 5
        assert isolation.prescription[0].target_reps == 8

    def test_endurance_goal_uses_15_reps(self, push_request, mock_candidates):
        result = generate_fallback_routine(
            push_request, candidates=mock_candidates, max_total_sets=20, goal="endurance"
        )
        for block in result.routine_blocks:
            for s in block.prescription:
                assert s.target_reps == 15

    def test_same_muscle_recent_sets_seed_fallback_weight(self, push_request, mock_candidates):
        recent_sets = [
            RecentSetRecord(
                exercise_name="Different Chest Exercise",
                primary_muscle="chest",
                weight_kg=100,
                reps=5,
            )
        ]
        result = generate_fallback_routine(
            push_request,
            candidates=mock_candidates,
            max_total_sets=20,
            goal="hypertrophy",
            recent_sets=recent_sets,
        )
        fly = next(block for block in result.routine_blocks if block.exercise_id == "dumbbell_fly")

        assert fly.prescription[0].target_weight_kg == 72.5


class TestFallbackReadiness:
    def test_low_readiness_reduces_compound_sets_and_raises_rir(self, push_request, mock_candidates):
        request = push_request.model_copy(update={"readiness_level": "low"})
        result = generate_fallback_routine(
            request, candidates=mock_candidates, max_total_sets=20, goal="hypertrophy"
        )
        compound = next(block for block in result.routine_blocks if block.exercise_id == "barbell_bench_press")
        assert len(compound.prescription) == 2
        assert compound.prescription[0].target_rir == 4

    def test_high_readiness_keeps_sets_and_lowers_rir(self, push_request, mock_candidates):
        request = push_request.model_copy(update={"readiness_level": "high"})
        result = generate_fallback_routine(
            request, candidates=mock_candidates, max_total_sets=20, goal="hypertrophy"
        )
        first = result.routine_blocks[0]
        assert len(first.prescription) == 3
        assert first.prescription[0].target_rir == 1


class TestFallbackStatusReasonCode:
    def test_custom_status_reason_code_preserved(self, push_request, mock_candidates):
        result = generate_fallback_routine(
            push_request, candidates=mock_candidates, max_total_sets=15,
            status_reason_code="llmTimeout"
        )
        assert result.status_reason_code == "llmTimeout"

    def test_compound_exercises_come_before_isolation(self, push_request, mock_candidates):
        result = generate_fallback_routine(
            push_request, candidates=mock_candidates, max_total_sets=15
        )
        if len(result.routine_blocks) >= 2:
            # 첫 번째 블록은 COMPOUND(벤치프레스 or 푸시업)이어야 함
            first_id = result.routine_blocks[0].exercise_id
            compound_ids = {"barbell_bench_press", "pushup"}
            assert first_id in compound_ids


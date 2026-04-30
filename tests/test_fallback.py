"""generate_fallback_routine — DB/LLM 없이 순수 로직 테스트"""
import pytest

from engines.routine_engine import RoutineRequest, generate_fallback_routine


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
    def test_returns_fallback_status(self, push_request):
        result = generate_fallback_routine(push_request, candidates=[], max_total_sets=15)
        assert result.is_fallback is True
        assert result.generation_status == "fallback"

    def test_empty_blocks_with_warning(self, push_request):
        result = generate_fallback_routine(push_request, candidates=[], max_total_sets=15)
        assert result.routine_blocks == []
        assert len(result.warnings) > 0


class TestFallbackDomsLogic:
    def test_doms_level3_skips_exercise(self, push_request, mock_candidates):
        doms_db = {"CHEST_MID": 3}
        result = generate_fallback_routine(
            push_request, candidates=mock_candidates, max_total_sets=15, doms_db=doms_db
        )
        exercise_ids = [b.exercise_id for b in result.routine_blocks]
        assert "barbell_bench_press" not in exercise_ids
        assert "dumbbell_fly" not in exercise_ids

    def test_doms_level2_limits_sets_to_2(self, push_request, mock_candidates):
        doms_db = {"CHEST_MID": 2}
        result = generate_fallback_routine(
            push_request, candidates=mock_candidates, max_total_sets=15, doms_db=doms_db
        )
        for block in result.routine_blocks:
            assert len(block.prescription) <= 2

    def test_doms_level1_reduces_sets_by_1(self, push_request, mock_candidates):
        doms_db = {"CHEST_MID": 1}
        result = generate_fallback_routine(
            push_request, candidates=mock_candidates, max_total_sets=15, doms_db=doms_db, goal="HYPERTROPHY"
        )
        # HYPERTROPHY 기본 3세트 → DOMS 1이면 max(1, 3-1)=2
        for block in result.routine_blocks:
            assert len(block.prescription) <= 2


class TestFallbackMaxSets:
    def test_total_sets_does_not_exceed_max(self, push_request, mock_candidates):
        max_sets = 5
        result = generate_fallback_routine(
            push_request, candidates=mock_candidates, max_total_sets=max_sets
        )
        total = sum(len(b.prescription) for b in result.routine_blocks)
        assert total <= max_sets


class TestFallbackGoalParams:
    def test_strength_goal_uses_5_reps(self, push_request, mock_candidates):
        result = generate_fallback_routine(
            push_request, candidates=mock_candidates, max_total_sets=20, goal="STRENGTH"
        )
        for block in result.routine_blocks:
            for s in block.prescription:
                assert s.target_reps == 5

    def test_endurance_goal_uses_15_reps(self, push_request, mock_candidates):
        result = generate_fallback_routine(
            push_request, candidates=mock_candidates, max_total_sets=20, goal="ENDURANCE"
        )
        for block in result.routine_blocks:
            for s in block.prescription:
                assert s.target_reps == 15


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

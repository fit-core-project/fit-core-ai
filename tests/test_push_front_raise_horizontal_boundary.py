"""Reviewer feedback: horizontal presses also make generic front raises redundant."""
import pytest

from engines.fallback import generate_fallback_routine
from engines.routine_blueprint import build_routine_blueprint
from engines.schemas import RoutineRequest
from engines.routine_selection import is_redundant_push_front_raise
from tests.test_feedback_selection_completion_variants import candidate, select


def rows():
    values = [
        candidate("flat", "덤벨 플랫 체스트 프레스", "chest", "DUMBBELL", "COMPOUND"),
        candidate("incline", "덤벨 인클라인 체스트 프레스", "chest", "DUMBBELL", "COMPOUND"),
        candidate("front", "덤벨 프론트 레이즈", "front-deltoids", "DUMBBELL"),
        candidate("lateral", "덤벨 레터럴 레이즈", "side-deltoids", "DUMBBELL"),
        candidate("triceps", "덤벨 라잉 트라이셉스 익스텐션", "triceps", "DUMBBELL"),
        candidate("fly", "덤벨 플라이", "chest", "DUMBBELL"),
    ]
    for i, row in enumerate(values):
        row.update(score=200-i, difficulty_tier=2, efficiency_tier=3)
    return values


@pytest.mark.parametrize("minutes", [60, 90])
def test_push_selection_does_not_use_front_raise_to_fill_absent_vertical_press(minutes):
    result, _, _ = select("push", None, rows(), time=minutes)
    ids = {p.exercise_id for p in result.exercises}
    assert {"flat", "triceps"} <= ids
    assert "front" not in ids


def test_sparse_pool_keeps_front_raise_when_no_qualified_alternative_exists():
    sparse = [row for row in rows() if row["id"] in {"flat", "front", "triceps"}]
    assert not is_redundant_push_front_raise("push", None, "front_raise", {"horizontal_press"}, sparse)
    assert is_redundant_push_front_raise("push", None, "front_raise", {"horizontal_press"}, rows())
    assert not is_redundant_push_front_raise(
        "push", "frontDeltoidEmphasis", "front_raise", {"horizontal_press"}, rows(),
    )
    assert not is_redundant_push_front_raise("custom", None, "front_raise", {"horizontal_press"}, rows())


@pytest.mark.parametrize("use_blueprint", [False, True])
def test_push_fallback_does_not_use_front_raise_to_fill_absent_vertical_press(use_blueprint):
    request = RoutineRequest(user_id="synthetic", goal="hypertrophy", target_split_label="push",
                             experience_level="advanced", time_available_min=90,
                             doms_data={}, pain_areas=[])
    blueprint = build_routine_blueprint(
        target_split_label="push", target_muscles=[], goal="hypertrophy",
        experience_level="advanced", readiness_level="normal", time_available_min=90,
    )
    result = generate_fallback_routine(
        request, candidates=rows(), max_total_sets=30, doms_db={}, goal="hypertrophy",
        target_exercise_count=8, routine_blueprint=blueprint if use_blueprint else None,
    )
    ids = {p.exercise_id for p in result.routine_blocks}
    assert {"flat", "triceps"} <= ids
    assert "front" not in ids

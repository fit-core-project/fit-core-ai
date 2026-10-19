import pytest

from engines.fallback import generate_fallback_routine
from engines.routine_blueprint import build_routine_blueprint
from engines.schemas import RoutineRequest


def _rows():
    definitions = [
        ("flat", "덤벨 플랫 체스트 프레스", "chest", "COMPOUND"),
        ("incline", "덤벨 인클라인 체스트 프레스", "chest", "COMPOUND"),
        ("shoulder", "덤벨 숄더 프레스", "front-deltoids", "COMPOUND"),
        ("front", "덤벨 프론트 레이즈", "front-deltoids", "ISOLATION"),
        ("lateral", "덤벨 레터럴 레이즈", "side-deltoids", "ISOLATION"),
        ("triceps", "덤벨 라잉 트라이셉스 익스텐션", "triceps", "ISOLATION"),
        ("fly", "덤벨 플라이", "chest", "ISOLATION"),
    ]
    return [dict(id=eid, name_kr=name, name_en=name, primary_muscle=muscle,
                 movement_type=movement, equipment_req="DUMBBELL",
                 score=200-i, efficiency_tier=3, difficulty_tier=2,
                 main_routine_default_allowed=1, exercise_familiarity_tier="common",
                 injury_caution_level="low", failure_penalty="low")
            for i, (eid, name, muscle, movement) in enumerate(definitions)]


@pytest.mark.parametrize("use_blueprint", [False, True])
def test_long_push_fallback_does_not_reintroduce_redundant_front_raise(use_blueprint):
    request = RoutineRequest(user_id="synthetic", goal="hypertrophy", target_split_label="push",
                             experience_level="advanced", time_available_min=90, doms_data={}, pain_areas=[])
    blueprint = build_routine_blueprint(target_split_label="push", target_muscles=[], goal="hypertrophy",
                                        experience_level="advanced", readiness_level="normal", time_available_min=90)
    result = generate_fallback_routine(request, candidates=_rows(), max_total_sets=30,
        doms_db={}, goal="hypertrophy", target_exercise_count=8,
        routine_blueprint=blueprint if use_blueprint else None)
    ids = {b.exercise_id for b in result.routine_blocks}
    assert {"flat", "shoulder", "triceps"} <= ids
    assert "front" not in ids


def test_explicit_front_deltoid_focus_remains_available_in_fallback():
    request = RoutineRequest(user_id="synthetic", goal="hypertrophy", target_split_label="shoulder",
        experience_level="advanced", time_available_min=60, doms_data={}, pain_areas=[],
        specialization_focus="frontDeltoidEmphasis")
    result = generate_fallback_routine(request, candidates=[r for r in _rows() if r['id'] in {'shoulder','front','lateral'}],
        max_total_sets=20, doms_db={}, goal="hypertrophy", target_exercise_count=3)
    assert "front" in {b.exercise_id for b in result.routine_blocks}

import pytest

from engines.exercise_display import candidate_display_name
from engines.routine_blueprint import build_blueprint_candidate_bundles, build_routine_blueprint
from engines.routine_selection import adapt_slot_selection_output
from engines.schemas import LLMRoutineSelectionOutput


def candidate(eid, name, primary, equipment="MACHINE", movement="ISOLATION", **extra):
    return {"id": eid, "name_kr": name, "primary_muscle": primary,
            "equipment_req": equipment, "movement_type": movement,
            "main_routine_default_allowed": 1, "score": 100, **extra}


def select(split, focus, candidates, time=60):
    blueprint = build_routine_blueprint(
        target_split_label=split, target_muscles=[], goal="hypertrophy",
        experience_level="advanced", readiness_level="normal", time_available_min=time,
        specialization_focus=focus, method_mode="off",
    )
    bundles = build_blueprint_candidate_bundles(blueprint, candidates)
    selections = [{"slotId": b["slotId"], "exerciseId": b["candidates"][0]["exerciseId"]}
                  for b in bundles if b["candidates"] and b["slotId"] != "flex"]
    result = adapt_slot_selection_output(
        LLMRoutineSelectionOutput.model_validate({"selections": selections}),
        blueprint=blueprint, slot_candidate_bundles=bundles, ranked_candidates=candidates,
    )
    return result, blueprint, bundles


def test_mid_back_focus_can_keep_loadable_row_variants_but_default_deduplicates():
    rows = [candidate("db_row", "덤벨 로우", "upper-back", "DUMBBELL", "COMPOUND"),
            candidate("machine_row", "머신 로우", "upper-back", movement="COMPOUND"),
            candidate("pull", "랫풀다운", "upper-back", movement="COMPOUND"),
            candidate("curl", "머신 프리처 컬", "biceps")]
    focused, _, _ = select("pull", "midBackEmphasis", rows)
    plain, _, _ = select("pull", None, rows)
    assert {"db_row", "machine_row"} <= {p.exercise_id for p in focused.exercises}
    assert len({"db_row", "machine_row"} & {p.exercise_id for p in plain.exercises}) == 1


def test_lateral_focus_keeps_two_equipment_variants_only_when_explicit():
    rows = [candidate("press", "덤벨 숄더 프레스", "front-deltoids", "DUMBBELL", "COMPOUND"),
            candidate("machine_lateral", "머신 레터럴 레이즈", "side-deltoids"),
            candidate("db_lateral", "덤벨 레터럴 레이즈", "side-deltoids", "DUMBBELL"),
            candidate("rear", "머신 리버스 펙덱 플라이", "back-deltoids"),
            candidate("shrug", "머신 슈러그", "trapezius")]
    focused, _, _ = select("shoulder", "sideDeltoidEmphasis", rows)
    plain, _, _ = select("shoulder", None, rows)
    assert {"machine_lateral", "db_lateral"} <= {p.exercise_id for p in focused.exercises}
    assert len({"machine_lateral", "db_lateral"} & {p.exercise_id for p in plain.exercises}) == 1


@pytest.mark.parametrize(("source", "label"), [
    ("Machine Standing Calf Raise", "스탠딩"),
    ("Machine Seated Calf Raise", "시티드"),
])
def test_calf_display_retains_source_position_without_inventing_it(source, label):
    row = candidate("calf", "머신 카프 레이즈", "calves", name_en=source,
                    user_facing_display_name_kr="머신 카프 레이즈")
    assert label in candidate_display_name(row)
    row["name_en"] = "Machine Calf Raise"
    assert candidate_display_name(row) == "머신 카프 레이즈"


def test_biceps_focus_prefers_three_distinct_curl_patterns_over_extra_triceps():
    rows = [candidate("preacher", "머신 프리처 컬", "biceps"),
            candidate("preacher_db", "덤벨 프리처 컬", "biceps", "DUMBBELL"),
            candidate("hammer", "덤벨 해머 컬", "biceps", "DUMBBELL"),
            candidate("incline", "덤벨 인클라인 컬", "biceps", "DUMBBELL"),
            candidate("pushdown", "머신 푸시다운", "triceps"),
            candidate("lying", "덤벨 라잉 트라이셉스 익스텐션", "triceps", "DUMBBELL"),
            candidate("wrist", "머신 리스트 컬", "forearm")]
    focused, _, _ = select("arm", "bicepsEmphasis", rows)
    ids = {p.exercise_id for p in focused.exercises}
    assert {"preacher", "hammer", "incline"} <= ids
    assert "preacher_db" not in ids


def test_advanced_full_body_reserves_vertical_pull_before_optional_core():
    rows = [candidate("squat", "핵 스쿼트", "quadriceps", movement="COMPOUND"),
            candidate("press", "덤벨 플랫 체스트 프레스", "chest", "DUMBBELL", "COMPOUND"),
            candidate("row", "덤벨 로우", "upper-back", "DUMBBELL", "COMPOUND"),
            candidate("pull", "랫풀다운", "upper-back", movement="COMPOUND"),
            candidate("legcurl", "머신 레그 컬", "hamstring"),
            candidate("shoulder", "덤벨 숄더 프레스", "front-deltoids", "DUMBBELL", "COMPOUND"),
            candidate("crunch", "케이블 크런치", "abs", "CABLE")]
    result, blueprint, bundles = select("full_body", None, rows)
    assert "full_vertical_pull" in {b["slotId"] for b in bundles}
    assert {"row", "pull"} <= {p.exercise_id for p in result.exercises}
    assert "crunch" not in {p.exercise_id for p in result.exercises}
    assert blueprint.target_exercise_count == 6


def test_long_push_uses_direct_triceps_before_redundant_front_raise():
    rows = [candidate("flat", "덤벨 플랫 체스트 프레스", "chest", "DUMBBELL", "COMPOUND"),
            candidate("incline", "덤벨 인클라인 체스트 프레스", "chest", "DUMBBELL", "COMPOUND"),
            candidate("shoulder", "덤벨 숄더 프레스", "front-deltoids", "DUMBBELL", "COMPOUND"),
            candidate("front", "덤벨 프론트 레이즈", "front-deltoids", "DUMBBELL"),
            candidate("lateral", "머신 레터럴 레이즈", "side-deltoids"),
            candidate("pushdown", "머신 푸시다운", "triceps"),
            candidate("lying", "덤벨 라잉 트라이셉스 익스텐션", "triceps", "DUMBBELL"),
            candidate("fly", "머신 펙덱 플라이", "chest")]
    result, _, _ = select("push", None, rows, time=90)
    ids = {p.exercise_id for p in result.exercises}
    assert "front" not in ids
    assert {"pushdown", "lying"} <= ids

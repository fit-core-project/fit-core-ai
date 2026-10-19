from engines.candidate_ranker import (
    candidate_selection_family_key,
    candidate_selection_pattern_key,
)
from engines.fallback import generate_fallback_routine
from engines.routine_blueprint import build_routine_blueprint
from engines.schemas import RoutineRequest


def _candidate(
    exercise_id: str,
    name: str,
    primary: str,
    movement_type: str,
    score: int,
    equipment: str = "MACHINE",
) -> dict:
    return {
        "id": exercise_id,
        "name_kr": name,
        "primary_muscle": primary,
        "secondary_muscle": "",
        "movement_type": movement_type,
        "equipment_req": equipment,
        "score": score,
        "efficiency_tier": 2,
    }


def _request_and_blueprint(
    *,
    split: str,
    targets: list[str] | None = None,
    time_available_min: int = 50,
    experience: str = "intermediate",
    specialization_focus: str | None = None,
    pain_areas: list[dict] | None = None,
) -> tuple[RoutineRequest, object]:
    request = RoutineRequest(
        user_id="fallback-main-role-regression",
        target_split_label=split,
        target_muscles=targets or [],
        time_available_min=time_available_min,
        readiness_level="normal",
        pain_areas=pain_areas or [],
        doms_data={},
        equipment=[],
        goal="hypertrophy",
        experience_level=experience,
        training_style="gym",
        specialization_focus=specialization_focus,
    )
    blueprint = build_routine_blueprint(
        target_split_label=split,
        target_muscles=targets or [],
        goal="hypertrophy",
        experience_level=experience,
        readiness_level="normal",
        time_available_min=time_available_min,
        training_style="gym",
        specialization_focus=specialization_focus,
        pain_areas=pain_areas or [],
    )
    return request, blueprint


def _fallback(request: RoutineRequest, blueprint: object, candidates: list[dict]):
    return generate_fallback_routine(
        request,
        candidates=candidates,
        max_total_sets=blueprint.max_total_sets,
        goal="hypertrophy",
        status_reason_code="networkError",
        target_exercise_count=blueprint.target_exercise_count,
        minimum_main_exercise_count=blueprint.minimum_main_exercise_count,
        routine_blueprint=blueprint,
    )


def test_pull_fallback_preserves_required_roles_and_selection_diversity():
    request, blueprint = _request_and_blueprint(split="pull")
    candidates = [
        _candidate("row", "덤벨 로우", "upper-back", "COMPOUND", 200, "DUMBBELL"),
        _candidate("preacher_machine", "머신 프리처 컬", "biceps", "ISOLATION", 190),
        _candidate("preacher_barbell", "바벨 프리처 컬", "biceps", "ISOLATION", 180, "BARBELL"),
        _candidate("preacher_dumbbell", "덤벨 프리처 컬", "biceps", "ISOLATION", 170, "DUMBBELL"),
        _candidate("preacher_cable", "케이블 프리처 컬", "biceps", "ISOLATION", 160, "CABLE"),
        _candidate("lat_pulldown", "케이블 랫 풀다운", "upper-back", "COMPOUND", 120, "CABLE"),
        _candidate("rear_fly", "머신 리버스 플라이", "back-deltoids", "ISOLATION", 110),
        _candidate("wrist_curl", "머신 리스트 컬", "forearm", "ISOLATION", 100),
    ]

    result = _fallback(request, blueprint, candidates)
    selected = {
        block.exercise_id: next(item for item in candidates if item["id"] == block.exercise_id)
        for block in result.routine_blocks
    }
    patterns = [candidate_selection_pattern_key(item) for item in selected.values()]
    families = [candidate_selection_family_key(item) for item in selected.values()]

    assert result.generation_status == "fallback"
    assert result.status_reason_code == "networkError"
    assert "horizontal_row" in patterns
    assert "vertical_pull" in patterns
    assert patterns.count("biceps_curl") == 1
    assert len(families) == len(set(families))


def test_vertical_only_custom_back_fallback_is_valid():
    request, blueprint = _request_and_blueprint(
        split="custom",
        targets=["upper-back"],
        time_available_min=30,
    )
    candidates = [
        _candidate("assisted_pullup", "어시스트 풀업", "upper-back", "COMPOUND", 120),
    ]

    result = _fallback(request, blueprint, candidates)

    assert result.generation_status == "fallback"
    assert [block.exercise_id for block in result.routine_blocks] == ["assisted_pullup"]


def test_explicit_trapezius_focus_keeps_shrug_outside_back_main_role():
    request, blueprint = _request_and_blueprint(
        split="custom",
        targets=["upper-back", "trapezius"],
        experience="advanced",
        specialization_focus="trapeziusEmphasis",
    )
    candidates = [
        _candidate("row", "덤벨 로우", "upper-back", "COMPOUND", 140, "DUMBBELL"),
        _candidate("shrug", "머신 슈러그", "trapezius", "ISOLATION", 130),
        _candidate("curl", "케이블 컬", "biceps", "ISOLATION", 120, "CABLE"),
        _candidate("pulldown", "케이블 랫 풀다운", "upper-back", "COMPOUND", 110, "CABLE"),
    ]

    result = _fallback(request, blueprint, candidates)
    selected_ids = {block.exercise_id for block in result.routine_blocks}

    assert result.generation_status == "fallback"
    assert {"row", "shrug"} <= selected_ids


def test_missing_required_main_is_not_padded_with_accessories():
    request, blueprint = _request_and_blueprint(split="pull")
    candidates = [
        _candidate("curl", "케이블 컬", "biceps", "ISOLATION", 140, "CABLE"),
        _candidate("wrist_curl", "머신 리스트 컬", "forearm", "ISOLATION", 130),
        _candidate("rear_fly", "머신 리버스 플라이", "back-deltoids", "ISOLATION", 120),
    ]

    result = _fallback(request, blueprint, candidates)

    assert result.generation_status == "failed"
    assert result.status_reason_code == "emptyCandidate"
    assert result.is_fallback is False
    assert result.routine_blocks == []


def test_safety_limited_multi_main_blueprint_keeps_remaining_valid_main():
    request, blueprint = _request_and_blueprint(
        split="full_body",
        time_available_min=30,
        pain_areas=[{"bodyPart": "chest"}],
    )
    candidates = [
        _candidate("leg_press", "머신 레그 프레스", "quadriceps", "COMPOUND", 140),
        _candidate("leg_curl", "머신 레그 컬", "hamstring", "ISOLATION", 130),
    ]

    result = _fallback(request, blueprint, candidates)

    assert result.generation_status == "fallback"
    assert "leg_press" in {block.exercise_id for block in result.routine_blocks}


def test_short_fallback_reserves_set_budget_for_direct_accessory():
    request, blueprint = _request_and_blueprint(split="push", time_available_min=20)
    candidates = [
        _candidate("flat", "덤벨 플랫 체스트 프레스", "chest", "COMPOUND", 200, "DUMBBELL"),
        _candidate("incline", "덤벨 인클라인 체스트 프레스", "chest", "COMPOUND", 190, "DUMBBELL"),
        _candidate("pushdown", "케이블 푸시다운", "triceps", "ISOLATION", 180, "CABLE"),
    ]

    result = _fallback(request, blueprint, candidates)

    assert result.generation_status == "fallback"
    assert result.status_reason_code == "networkError"
    assert {block.exercise_id for block in result.routine_blocks} == {"flat", "incline", "pushdown"}
    assert sum(len(block.prescription) for block in result.routine_blocks) <= blueprint.max_total_sets
    assert result.total_estimated_time <= request.time_available_min


def test_unsuccessful_time_trim_does_not_mutate_accepted_prescriptions(monkeypatch):
    request, blueprint = _request_and_blueprint(split="push", time_available_min=20)
    candidates = [
        _candidate("flat", "덤벨 플랫 체스트 프레스", "chest", "COMPOUND", 200, "DUMBBELL"),
        _candidate("incline", "덤벨 인클라인 체스트 프레스", "chest", "COMPOUND", 190, "DUMBBELL"),
        _candidate("pushdown", "케이블 푸시다운", "triceps", "ISOLATION", 180, "CABLE"),
    ]
    attempts = []

    def partial_trim_that_cannot_fit(output, *_args, **_kwargs):
        attempts.append(True)
        for plan in output.exercises:
            plan.sets = 1
        return False

    monkeypatch.setattr("engines.fallback.trim_routine_to_time_budget", partial_trim_that_cannot_fit)
    result = _fallback(request, blueprint, candidates)

    assert attempts
    assert [(block.exercise_id, len(block.prescription)) for block in result.routine_blocks] == [
        ("flat", 3), ("incline", 2)
    ]
    assert result.total_estimated_time == 18

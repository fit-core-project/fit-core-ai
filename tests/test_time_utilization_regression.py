import pytest

from engines.fallback import generate_fallback_routine
from engines.prescription.adjustments import _fill_available_time
from engines.prescription.estimator import estimate_routine_time_min
from engines.prescription.underfill import resolve_underfill_decision
from engines.prescription.targets import apply_deterministic_targets
from engines.routine_blueprint import build_routine_blueprint
from engines.schemas import LLMExercisePlan, LLMRoutineOutput, RoutineRequest


def _plan(
    exercise_id: str,
    primary: str,
    *,
    movement_type: str = "COMPOUND",
    sets: int = 3,
    rir: int = 2,
) -> LLMExercisePlan:
    return LLMExercisePlan(
        exercise_id=exercise_id,
        exercise_name=exercise_id,
        movement_type=movement_type,
        primary_muscles=[primary],
        target_reps=10,
        sets=sets,
        rest_time_sec=90,
        target_rir=rir,
        exercise_rationale="test",
    )


def _output(*plans: LLMExercisePlan) -> LLMRoutineOutput:
    return LLMRoutineOutput(
        total_estimated_time=estimate_routine_time_min(list(plans)),
        summary_title="test",
        rationale_summary=[],
        exercises=list(plans),
    )


def _candidate(
    exercise_id: str,
    name: str,
    movement_type: str,
    equipment: str,
    score: int,
) -> dict:
    return {
        "id": exercise_id,
        "name_kr": name,
        "primary_muscle": "chest",
        "secondary_muscle": "triceps" if movement_type == "COMPOUND" else "",
        "movement_type": movement_type,
        "equipment_req": equipment,
        "score": score,
        "efficiency_tier": 3,
    }


def _advanced_chest_fallback(*, readiness: str = "normal", doms: dict | None = None):
    request = RoutineRequest(
        user_id="time-utilization-regression",
        target_split_label="chest",
        target_muscles=[],
        time_available_min=60,
        readiness_level=readiness,
        doms_data=doms or {},
        equipment=[],
        goal="hypertrophy",
        experience_level="advanced",
        training_style="gym",
    )
    blueprint = build_routine_blueprint(
        target_split_label="chest",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="advanced",
        readiness_level=readiness,
        time_available_min=60,
        training_style="gym",
        doms_data=doms or {},
    )
    candidates = [
        _candidate("flat", "덤벨 플랫 체스트 프레스", "COMPOUND", "DUMBBELL", 300),
        _candidate("incline", "덤벨 인클라인 체스트 프레스", "COMPOUND", "DUMBBELL", 290),
        _candidate("pec_deck", "머신 펙덱 플라이", "ISOLATION", "MACHINE", 280),
    ]
    result = generate_fallback_routine(
        request,
        candidates=candidates,
        max_total_sets=blueprint.max_total_sets,
        doms_db=doms or {},
        goal="hypertrophy",
        status_reason_code="networkError",
        target_exercise_count=blueprint.target_exercise_count,
        minimum_main_exercise_count=blueprint.minimum_main_exercise_count,
        routine_blueprint=blueprint,
    )
    return request, blueprint, candidates, result


def test_fallback_productive_fill_improves_time_without_changing_selection():
    request, blueprint, candidates, result = _advanced_chest_fallback()

    assert result.generation_status == "fallback"
    assert [block.exercise_id for block in result.routine_blocks] == [
        candidate["id"] for candidate in candidates
    ]
    assert sum(len(block.prescription) for block in result.routine_blocks) > 9
    assert sum(len(block.prescription) for block in result.routine_blocks) <= blueprint.max_total_sets
    assert 28 < result.total_estimated_time <= request.time_available_min
    assert all(
        len({
            (
                prescription.target_reps,
                prescription.target_rir,
                prescription.target_rest_sec,
            )
            for prescription in block.prescription
        }) == 1
        for block in result.routine_blocks
    )


def test_fill_available_time_honors_total_set_budget():
    output = _output(
        _plan("flat", "chest"),
        _plan("row", "upper-back"),
        _plan("leg_press", "quadriceps"),
    )

    diagnostics = _fill_available_time(
        output,
        60,
        "hypertrophy",
        "normal",
        "full_body",
        max_total_sets=10,
    )

    assert sum(plan.sets for plan in output.exercises) == 10
    assert estimate_routine_time_min(output.exercises) <= 60
    assert diagnostics['role_set_capacity_exhausted'] is False
    assert diagnostics['role_set_capacity_total_budget_exhausted'] is True


def test_role_capacity_requires_every_ordinary_working_role_and_resets_per_call():
    capped = _output(
        _plan("flat", "chest", sets=4),
        _plan("incline", "chest", sets=4),
        _plan("fly", "chest", movement_type="ISOLATION", sets=3),
    )
    capped_diagnostics = _fill_available_time(
        capped, 60, "hypertrophy", "normal", "chest", max_total_sets=20,
    )
    assert capped_diagnostics['role_set_capacity_exhausted'] is True

    mixed_doms = _output(
        _plan("flat", "chest", sets=4),
        _plan("triceps", "triceps", movement_type="ISOLATION", sets=2),
        _plan("fly", "chest", movement_type="ISOLATION", sets=3),
    )
    mixed_doms_diagnostics = _fill_available_time(
        mixed_doms,
        60,
        "hypertrophy",
        "normal",
        "chest",
        doms_db={"triceps": 1},
        doms_restricted_exercise_ids={"triceps"},
        max_total_sets=20,
    )
    assert mixed_doms_diagnostics['role_set_capacity_exhausted'] is False
    assert mixed_doms_diagnostics['role_set_capacity_restricted_working_role'] is True


def test_short_time_and_normal_target_cap_trim_do_not_keep_capacity_proof():
    short = _output(
        _plan("flat", "chest", sets=3),
        _plan("incline", "chest", sets=3),
    )
    short_diagnostics = _fill_available_time(
        short, 20, "hypertrophy", "normal", "chest", max_total_sets=20,
    )
    assert short_diagnostics['role_set_capacity_exhausted'] is False

    output = _output(
        _plan("flat", "chest", sets=4),
        _plan("incline", "chest", sets=4),
        _plan("fly", "chest", movement_type="ISOLATION", sets=3),
    )
    adjusted = apply_deterministic_targets(
        output,
        goal="hypertrophy",
        recent_sets=None,
        profile=None,
        readiness_level="normal",
        target_split_label="chest",
        target_muscles=["chest"],
        time_available_min=60,
        max_total_sets=9,
        experience_level="advanced",
    )
    assert sum(plan.sets for plan in adjusted.exercises) == 9
    assert adjusted._time_fill_diagnostics['role_set_capacity_exhausted'] is False
    assert adjusted._time_fill_diagnostics['role_set_capacity_total_budget_exhausted'] is True


def test_fill_does_not_restore_direct_or_secondary_doms_adjustments():
    direct_doms = _plan("direct_doms", "chest", sets=2, rir=3)
    secondary_doms = _plan("secondary_doms", "triceps", sets=1, rir=4)
    output = _output(direct_doms, secondary_doms)

    _fill_available_time(
        output,
        60,
        "hypertrophy",
        "normal",
        "push",
        doms_db={"chest": 1},
        doms_restricted_exercise_ids={"direct_doms", "secondary_doms"},
        max_total_sets=20,
    )

    assert [(plan.sets, plan.target_rir) for plan in output.exercises] == [(2, 3), (1, 4)]


def test_low_readiness_volume_is_not_refilled():
    # Historical ID retained: refill may reach reduced role capacity, never
    # the normal four-set cap. The blanket two-set oracle is superseded.
    plans = [_plan("flat", "chest", sets=2, rir=4), _plan("row", "upper-back", sets=2, rir=4)]
    output = _output(*plans)

    _fill_available_time(
        output,
        60,
        "hypertrophy",
        "low",
        "full_body",
        max_total_sets=20,
    )

    assert [(plan.sets, plan.target_rir) for plan in output.exercises] == [(3, 4), (3, 4)]


def test_timed_adjunct_prescription_is_unchanged_by_fill():
    working = _plan("flat", "chest", sets=3)
    timed = _plan("mobility", "abs", movement_type="STATIC", sets=1)
    timed._prescription_set_type = "timed"
    timed._target_duration_sec = 300
    output = _output(working, timed)

    _fill_available_time(
        output,
        60,
        "hypertrophy",
        "normal",
        "push",
        max_total_sets=4,
    )

    assert working.sets == 4
    assert timed.sets == 1
    assert timed._target_duration_sec == 300


def test_distance_work_suppresses_rep_role_capacity_claim():
    capped = _plan("flat", "chest", sets=4)
    distance = _plan("carry", "forearm", movement_type="COMPOUND", sets=1)
    distance._prescription_set_type = "distance"
    distance._target_distance_m = 30
    diagnostics = _fill_available_time(
        _output(capped, distance),
        60,
        "hypertrophy",
        "normal",
        "chest",
        max_total_sets=20,
    )

    assert diagnostics['role_set_capacity_unmeasured_working_role'] is True
    assert diagnostics['role_set_capacity_exhausted'] is False


@pytest.mark.parametrize("doms_level", [1, 2])
def test_fallback_never_refills_doms_adjusted_work(doms_level):
    _, _, _, result = _advanced_chest_fallback(doms={"chest": doms_level})

    expected_rir = 3 if doms_level == 1 else 4
    assert result.routine_blocks
    # v2: compound capacity 4 -> 3; isolation capacity 3 -> 2. No time refill.
    assert [len(block.prescription) for block in result.routine_blocks] == [3, 3, 2]
    assert all(
        prescription.target_rir == expected_rir
        for block in result.routine_blocks
        for prescription in block.prescription
    )


def test_doms_three_still_returns_zero_work_and_zero_time():
    _, _, _, result = _advanced_chest_fallback(doms={"chest": 3})

    assert result.generation_status == "failed"
    assert result.routine_blocks == []
    assert result.total_estimated_time == 0


def test_underfill_explanation_names_budget_plan_constraints_and_choices():
    decision = resolve_underfill_decision(
        actual_duration_min=20,
        available_duration_min=60,
        has_routine_blocks=True,
        injury_limited=True,
    )

    assert "요청하신 60분" in decision.user_summary_ko
    assert "예상 20분" in decision.user_summary_ko
    assert "등록된 부상 부위" in decision.user_summary_ko
    assert "이 계획을 그대로 진행" in decision.user_option_ko
    assert "제한과 겹치지 않는 다른 목표 부위" in decision.user_option_ko
    assert "동의" not in decision.user_option_ko

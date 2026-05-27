from __future__ import annotations

from engines.rule_critic import (
    RoutineEvalContext,
    RuleCriticResult,
    _balance_score,
    _candidate_lookup,
    _check_hard_constraints_sanity,
    _coverage_score,
    _diversity_score,
    _grade,
    _rationale_score,
    _repair_hygiene_score,
    _time_score,
    build_eval_context,
    evaluate_routine_quality,
)
from engines.schemas import (
    PainAreaEntry,
    RecentSetRecord,
    RoutineBlock,
    RoutineDraftResponse,
    RoutineRequest,
    SetPrescription,
)


def candidate(
    exercise_id: str = "bench",
    primary_muscle: str = "chest",
    *,
    equipment_req: str = "BARBELL",
    movement_type: str = "COMPOUND",
    pain_triggers: str | None = None,
) -> dict:
    return {
        "id": exercise_id,
        "name_kr": exercise_id,
        "name_en": exercise_id,
        "primary_muscle": primary_muscle,
        "secondary_muscle": None,
        "equipment_req": equipment_req,
        "movement_type": movement_type,
        "efficiency_tier": 4,
        "pain_triggers": pain_triggers,
    }


def block(
    exercise_id: str = "bench",
    primary_muscles: list[str] | None = None,
    *,
    equipment_type: str = "BARBELL",
    sets: int = 3,
    rationale: str = "matches the target muscle and goal",
    weight: float | None = None,
) -> RoutineBlock:
    return RoutineBlock(
        order=1,
        exercise_id=exercise_id,
        exercise_name=exercise_id,
        primary_muscles=primary_muscles or ["chest"],
        equipment_type=equipment_type,
        default_rest_sec=120,
        prescription=[
            SetPrescription(
                set_index=index + 1,
                target_reps=8,
                target_weight_kg=weight,
                target_rest_sec=120,
            )
            for index in range(sets)
        ],
        exercise_rationale=rationale,
    )


def response(
    blocks: list[RoutineBlock] | None = None,
    *,
    total_time: int = 60,
    warnings: list[str] | None = None,
    is_fallback: bool = False,
) -> RoutineDraftResponse:
    return RoutineDraftResponse(
        generation_status="fallback" if is_fallback else "success",
        status_reason_code="emptyCandidate" if is_fallback else "none",
        is_fallback=is_fallback,
        total_estimated_time=total_time,
        summary_title="Test routine",
        rationale_summary=["Deterministic test routine."],
        routine_blocks=blocks if blocks is not None else [block()],
        warnings=warnings or [],
    )


def context(**overrides) -> RoutineEvalContext:
    values = {
        "goal": "hypertrophy",
        "target_muscles": ["chest"],
        "duration_min": 60,
        "readiness_level": "normal",
        "available_equipment": ["BARBELL", "DUMBBELL", "CABLE", "BODYWEIGHT"],
        "unavailable_equipment": [],
        "current_pain_areas": [],
        "doms": {},
        "recent_exercises": [],
        "max_working_sets": 18,
        "overrides": {},
    }
    values.update(overrides)
    return RoutineEvalContext(**values)


def blocks_to_dicts(items: list[RoutineBlock]) -> list[dict]:
    return [item.model_dump(by_alias=False) for item in items]


def test_rule_critic_result_fields():
    result = RuleCriticResult(
        score=90,
        grade="PASS",
        hard_violations=[],
        warnings=[],
        metric_scores={"coverage": 25},
        should_rebuild=False,
        should_fallback=False,
        notes=["note"],
    )
    assert result.score == 90
    assert result.notes == ["note"]


def test_context_overrides_default_is_independent():
    first = context()
    second = context()
    first.overrides["coverage_exception"] = True
    assert second.overrides == {}


def test_should_rebuild_is_false_in_v1():
    result = evaluate_routine_quality(response(), [candidate()], context())
    assert result.should_rebuild is False


def test_should_fallback_is_false_in_v1():
    result = evaluate_routine_quality(response(), [candidate()], context())
    assert result.should_fallback is False


def test_candidate_outside_candidates_is_detected():
    violations = _check_hard_constraints_sanity(
        blocks_to_dicts([block("ghost")]),
        _candidate_lookup([candidate("bench")]),
        context(),
        18,
    )
    assert any("not in candidates" in item for item in violations)


def test_unavailable_equipment_is_detected():
    violations = _check_hard_constraints_sanity(
        blocks_to_dicts([block("bench")]),
        _candidate_lookup([candidate("bench", equipment_req="BARBELL")]),
        context(unavailable_equipment=["BARBELL"]),
        18,
    )
    assert any("blocked equipment" in item for item in violations)


def test_pain_trigger_is_detected():
    violations = _check_hard_constraints_sanity(
        blocks_to_dicts([block("press")]),
        _candidate_lookup([candidate("press", pain_triggers="shoulder")]),
        context(current_pain_areas=[PainAreaEntry(body_part="shoulder")]),
        18,
    )
    assert any("pain trigger" in item for item in violations)


def test_doms_level_3_is_detected():
    violations = _check_hard_constraints_sanity(
        blocks_to_dicts([block("bench", ["chest"])]),
        _candidate_lookup([candidate("bench", "chest")]),
        context(doms={"chest": 3}),
        18,
    )
    assert any("DOMS level 3" in item for item in violations)


def test_working_set_cap_is_detected():
    violations = _check_hard_constraints_sanity(
        blocks_to_dicts([block("bench", sets=4)]),
        _candidate_lookup([candidate("bench")]),
        context(),
        3,
    )
    assert any("working set cap exceeded" in item for item in violations)


def test_bodyweight_weight_is_detected():
    violations = _check_hard_constraints_sanity(
        blocks_to_dicts([block("pushup", equipment_type="BODYWEIGHT", weight=10.0)]),
        _candidate_lookup([candidate("pushup", equipment_req="BODYWEIGHT")]),
        context(),
        18,
    )
    assert any("BODYWEIGHT target_weight_kg" in item for item in violations)


def test_hard_violation_forces_fail_grade():
    result = evaluate_routine_quality(response([block("ghost")]), [candidate("bench")], context())
    assert result.grade == "FAIL"


def test_coverage_full_score():
    score, warnings = _coverage_score(
        blocks_to_dicts([block("bench", ["chest"]), block("pushdown", ["triceps"])]),
        _candidate_lookup([candidate("bench", "chest"), candidate("pushdown", "triceps")]),
        context(target_muscles=["chest", "triceps"]),
        False,
        2,
    )
    assert score == 25
    assert warnings == []


def test_coverage_partial_score():
    score, warnings = _coverage_score(
        blocks_to_dicts([block("bench", ["chest"])]),
        _candidate_lookup([candidate("bench", "chest")]),
        context(target_muscles=["chest", "triceps"]),
        False,
        3,
    )
    assert score == 12
    assert any("coverage below" in item for item in warnings)


def test_coverage_zero_warns_fail_without_exception():
    score, warnings = _coverage_score(
        blocks_to_dicts([block("row", ["upper-back"])]),
        _candidate_lookup([candidate("row", "upper-back")]),
        context(target_muscles=["chest"]),
        False,
        3,
    )
    assert score == 0
    assert any(item.startswith("FAIL coverage") for item in warnings)


def test_coverage_exception_removes_fail_prefix():
    score, warnings = _coverage_score(
        blocks_to_dicts([block("row", ["upper-back"])]),
        _candidate_lookup([candidate("row", "upper-back")]),
        context(target_muscles=["chest"], overrides={"coverage_exception": True}),
        False,
        3,
    )
    assert score == 0
    assert warnings and not warnings[0].startswith("FAIL ")


def test_time_within_five_minutes_gets_full_score():
    score, warnings = _time_score(response(total_time=56), context(duration_min=60), False, 4)
    assert score == 20
    assert warnings == []


def test_time_over_ten_minutes_gets_zero_and_fail_warning():
    score, warnings = _time_score(response(total_time=40), context(duration_min=60), False, 4)
    assert score == 0
    assert any(item.startswith("FAIL time") for item in warnings)


def test_allowed_underfill_gets_partial_score():
    score, warnings = _time_score(
        response(total_time=35),
        context(duration_min=60, overrides={"allowed_underfill": True}),
        False,
        4,
    )
    assert score == 10
    assert any("allowed underfill" in item for item in warnings)


def test_diversity_low_repetition_passes():
    score, warnings = _diversity_score(
        blocks_to_dicts([block("bench"), block("row", ["upper-back"])]),
        context(recent_exercises=[]),
        4,
    )
    assert score == 15
    assert warnings == []


def test_diversity_over_fifty_percent_warns():
    score, warnings = _diversity_score(
        blocks_to_dicts([block("bench"), block("row", ["upper-back"]), block("curl", ["biceps"])]),
        context(recent_exercises=["bench", "row"]),
        4,
    )
    assert score == 5
    assert any("recent repetition warning" in item for item in warnings)


def test_diversity_over_seventy_five_percent_fails():
    score, warnings = _diversity_score(
        blocks_to_dicts([block("bench"), block("row", ["upper-back"]), block("curl", ["biceps"])]),
        context(recent_exercises=["bench", "row", "curl"]),
        4,
    )
    assert score == 0
    assert any(item.startswith("FAIL recent repetition") for item in warnings)


def test_diversity_candidate_shortage_allows_repetition_warning():
    score, warnings = _diversity_score(
        blocks_to_dicts([block("bench"), block("row", ["upper-back"])]),
        context(recent_exercises=["bench", "row"], overrides={"candidate_shortage": True}),
        2,
    )
    assert score == 5
    assert not any(item.startswith("FAIL ") for item in warnings)


def test_strength_with_compound_passes_balance():
    score, warnings = _balance_score(
        blocks_to_dicts([block("squat", ["quadriceps"])]),
        _candidate_lookup([candidate("squat", "quadriceps", movement_type="COMPOUND")]),
        context(goal="strength"),
        3,
        False,
    )
    assert score == 15
    assert warnings == []


def test_strength_without_compound_fails_balance():
    score, warnings = _balance_score(
        blocks_to_dicts([block("curl", ["biceps"])]),
        _candidate_lookup([candidate("curl", "biceps", movement_type="ISOLATION")]),
        context(goal="strength"),
        3,
        False,
    )
    assert score == 0
    assert any(item.startswith("FAIL strength") for item in warnings)


def test_hypertrophy_mixed_balance_passes():
    score, warnings = _balance_score(
        blocks_to_dicts([block("bench"), block("fly", ["chest"])]),
        _candidate_lookup([
            candidate("bench", "chest", movement_type="COMPOUND"),
            candidate("fly", "chest", movement_type="ISOLATION"),
        ]),
        context(goal="hypertrophy"),
        3,
        False,
    )
    assert score == 15
    assert warnings == []


def test_min_compound_count_override_is_enforced():
    score, warnings = _balance_score(
        blocks_to_dicts([block("bench")]),
        _candidate_lookup([candidate("bench", movement_type="COMPOUND")]),
        context(overrides={"min_compound_count": 2}),
        3,
        False,
    )
    assert score == 0
    assert any("compound count below expected" in item for item in warnings)


def test_rationale_nonempty_passes():
    score, warnings = _rationale_score(blocks_to_dicts([block()]), context())
    assert score == 15
    assert warnings == []


def test_empty_rationale_fails_when_required():
    score, warnings = _rationale_score(blocks_to_dicts([block(rationale="")]), context())
    assert score == 0
    assert any("FAIL rationale" in item for item in warnings)


def test_rationale_nonempty_override_allows_missing_rationale():
    score, warnings = _rationale_score(
        blocks_to_dicts([block(rationale="")]),
        context(overrides={"rationale_must_be_nonempty": False}),
    )
    assert score == 0
    assert warnings


def test_rationale_detects_invented_history_or_medical_claim():
    score, warnings = _rationale_score(
        blocks_to_dicts([block(rationale="doctor said your injury history shows this is safe")]),
        context(),
    )
    assert score == 15
    assert any("unsupported medical claim" in item for item in warnings)


def test_repair_count_is_penalized_when_unexpected():
    score, warnings = _repair_hygiene_score(response(), 1, False, context())
    assert score == 7
    assert any("repair occurred" in item for item in warnings)


def test_fallback_used_is_penalized_when_unexpected():
    score, warnings = _repair_hygiene_score(response(is_fallback=True), 0, True, context())
    assert score == 0
    assert any("fallback mismatch" in item for item in warnings)


def test_expected_repair_count_uses_guard_replaced_only():
    warnings = [
        "Guard diagnostics: repairCount=1, trimmedSetCount=0, removedExerciseCount=0",
        "Guard: 'ghost' -> 'bench' (constraint violation replaced)",
        "Guard: informational line without replacement",
    ]
    repair_count = len([item for item in warnings if "Guard:" in item and "replaced" in item])
    assert repair_count == 1


def test_build_eval_context_uses_request_and_recent_sets():
    req = RoutineRequest(
        user_id="u1",
        target_muscles=["chest"],
        readiness_level="low",
        time_available_min=30,
        pain_areas=[PainAreaEntry(body_part="shoulder")],
        doms_data={"chest": 2},
        equipment=["BARBELL"],
        goal="fatloss",
    )
    built = build_eval_context(req, ["triceps"], [RecentSetRecord(exercise_id="bench", exercise_name="Bench", reps=8)], 8)
    assert built.goal == "fatloss"
    assert built.target_muscles == ["triceps"]
    assert built.recent_exercises == ["bench"]
    assert built.max_working_sets == 8


def test_evaluate_normal_routine_passes():
    blocks = [block("bench", ["chest"]), block("pushdown", ["triceps"])]
    candidates = [
        candidate("bench", "chest", movement_type="COMPOUND"),
        candidate("pushdown", "triceps", equipment_req="CABLE", movement_type="ISOLATION"),
    ]
    result = evaluate_routine_quality(
        response(blocks),
        candidates,
        context(target_muscles=["chest", "triceps"]),
    )
    assert result.grade == "PASS"
    assert result.score >= 80


def test_evaluate_low_score_routine_warns():
    result = evaluate_routine_quality(
        response([block("bench", ["chest"])], total_time=30),
        [
            candidate("bench", "chest", movement_type="COMPOUND"),
            candidate("fly", "chest", movement_type="ISOLATION"),
            candidate("pushdown", "triceps", equipment_req="CABLE", movement_type="ISOLATION"),
        ],
        context(target_muscles=["chest"], duration_min=60),
    )
    assert result.grade == "WARN"
    assert 65 <= result.score < 80


def test_evaluate_hard_violation_routine_fails():
    result = evaluate_routine_quality(
        response([block("ghost")]),
        [candidate("bench")],
        context(target_muscles=["chest"]),
    )
    assert result.grade == "FAIL"
    assert result.hard_violations


def test_grade_boundaries():
    assert _grade(80, []) == "PASS"
    assert _grade(65, []) == "WARN"
    assert _grade(64, []) == "FAIL"
    assert _grade(100, ["violation"]) == "FAIL"

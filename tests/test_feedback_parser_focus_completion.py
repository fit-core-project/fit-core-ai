from copy import deepcopy

import pytest

from engines.llm_parser import validate_and_repair_routine_output
from engines.schemas import LLMExercisePlan, LLMRoutineOutput, PainAreaEntry


def candidates(primary="upper-back", name="로우", movement="COMPOUND"):
    return [
        {"id": "db", "name_kr": f"덤벨 {name}", "name_en": f"Dumbbell {name}",
         "primary_muscle": primary, "equipment_req": "DUMBBELL", "movement_type": movement,
         "main_routine_default_allowed": 1, "pain_triggers": primary},
        {"id": "machine", "name_kr": f"머신 {name}", "name_en": f"Machine {name}",
         "primary_muscle": primary, "equipment_req": "MACHINE", "movement_type": movement,
         "main_routine_default_allowed": 1, "pain_triggers": primary},
    ]


def output(rows):
    return LLMRoutineOutput(
        total_estimated_time=20, summary_title="검증", rationale_summary=[], warnings=[],
        exercises=[LLMExercisePlan(exercise_id=c["id"], exercise_name=c["name_kr"],
                   primary_muscles=[c["primary_muscle"]], movement_type=c["movement_type"],
                   equipment_type=c["equipment_req"], target_reps=10, sets=2,
                   rest_time_sec=90, exercise_rationale="합성 검증") for c in rows],
    )


@pytest.mark.parametrize("primary,name,movement,focus", [
    ("upper-back", "로우", "COMPOUND", "midBackEmphasis"),
    ("side-deltoids", "레터럴 레이즈", "ISOLATION", "sideDeltoidEmphasis"),
    ("trapezius", "슈러그", "ISOLATION", "trapeziusEmphasis"),
])
def test_explicit_focus_keeps_reviewed_equipment_variants(primary, name, movement, focus):
    rows = candidates(primary, name, movement)
    result, reason = validate_and_repair_routine_output(
        output(rows), rows, [], [], specialization_focus=focus,
    )
    assert reason == "none" and result is not None
    assert [p.exercise_id for p in result.exercises] == ["db", "machine"]


def test_default_still_collapses_the_same_general_movement_family():
    rows = candidates()
    result, reason = validate_and_repair_routine_output(output(rows), rows, [], [])
    assert reason == "none" and result is not None
    assert [p.exercise_id for p in result.exercises] == ["db"]


@pytest.mark.parametrize("focus", [None, "unknown", "sideDeltoidEmphasis"])
def test_unrelated_or_absent_focus_does_not_allow_extra_rows(focus):
    rows = candidates()
    result, _ = validate_and_repair_routine_output(output(rows), rows, [], [], specialization_focus=focus)
    assert result is not None and len(result.exercises) == 1


def test_same_equipment_clone_still_collapses():
    rows = candidates()
    rows[1].update(equipment_req="DUMBBELL", name_kr="덤벨 시티드 로우", name_en="Dumbbell Seated Row")
    result, _ = validate_and_repair_routine_output(output(rows), rows, [], [], specialization_focus="midBackEmphasis")
    assert result is not None and len(result.exercises) == 1


def test_duplicate_id_is_not_a_focus_variant():
    rows = candidates()
    result, _ = validate_and_repair_routine_output(
        output([rows[0], rows[0]]), [rows[0]], [], [], specialization_focus="midBackEmphasis",
    )
    assert result is not None and [p.exercise_id for p in result.exercises] == ["db"]


def test_unreviewed_rows_do_not_gain_variant_identity():
    # Prevent focus from legitimizing unreviewed variants; P1 excludes all denied rows instead of deduplicating to one.
    rows = candidates()
    for row in rows:
        row["main_routine_default_allowed"] = 0
    result, reason = validate_and_repair_routine_output(output(rows), rows, [], [], specialization_focus="midBackEmphasis")
    assert result is None and reason == "emptyCandidate"


def test_focus_preserves_equipment_denial():
    rows = candidates()
    result, _ = validate_and_repair_routine_output(
        output(rows), rows, ["DUMBBELL"], [], specialization_focus="midBackEmphasis",
    )
    assert result is not None and all(p.exercise_id == "machine" for p in result.exercises)


def test_focus_does_not_override_injury_filter():
    rows = candidates()
    result, reason = validate_and_repair_routine_output(
        output(rows), rows, [], [PainAreaEntry(body_part="upper-back")],
        specialization_focus="midBackEmphasis",
    )
    assert result is None and reason == "emptyCandidate"


def test_focus_does_not_override_doms_stop():
    rows = candidates()
    result, reason = validate_and_repair_routine_output(
        output(rows), rows, [], [], doms_db={"upper-back": 3},
        specialization_focus="midBackEmphasis",
    )
    assert result is None and reason == "schemaError"


def test_replacement_uses_the_same_focus_aware_identity():
    rows = candidates()
    source = output(rows)
    source.exercises[1].exercise_id = "invalid"
    result, reason = validate_and_repair_routine_output(source, rows, [], [], specialization_focus="midBackEmphasis")
    assert reason == "none" and result is not None
    assert [p.exercise_id for p in result.exercises] == ["db", "machine"]
    assert source.exercises[1].exercise_id == "invalid"


def test_validator_does_not_mutate_candidates():
    rows = candidates()
    before = deepcopy(rows)
    validate_and_repair_routine_output(output(rows), rows, [], [], specialization_focus="midBackEmphasis")
    assert rows == before


@pytest.mark.parametrize("with_blueprint", [True, False])
def test_pipeline_passes_only_resolved_blueprint_focus(monkeypatch, with_blueprint):
    from engines import llm_parser, routine_pipeline
    from engines.routine_blueprint import build_routine_blueprint
    from engines.schemas import RoutineRequest

    rows = candidates()
    req = RoutineRequest(time_available_min=60, target_split_label="pull",
                         specialization_focus="midBackEmphasis", experience_level="advanced")
    blueprint = build_routine_blueprint(
        target_split_label="pull", target_muscles=[], goal="hypertrophy",
        experience_level="advanced", readiness_level="normal", time_available_min=60,
        specialization_focus="midBackEmphasis", method_mode="off",
    ) if with_blueprint else None
    seen = {}
    real_validate = llm_parser.validate_and_repair_routine_output

    def capture(*args, **kwargs):
        seen.update(kwargs)
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(llm_parser, "validate_and_repair_routine_output", capture)
    monkeypatch.setattr(routine_pipeline, "_deterministic_or_fallback", lambda result, *a, **k: result)
    result = routine_pipeline._validate_or_fallback(
        output(rows), req, rows, 20, {}, "hypertrophy", ["upper-back"], None, None,
        routine_blueprint=blueprint,
    )
    assert seen["specialization_focus"] == ("midBackEmphasis" if with_blueprint else None)
    assert len(result.exercises) == (2 if with_blueprint else 1)

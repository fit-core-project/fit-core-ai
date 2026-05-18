import json
from pathlib import Path

import pytest

from engines.routine_engine import (
    LLMExercisePlan,
    LLMRoutineOutput,
    PainAreaEntry,
    RoutineRequest,
    apply_deterministic_targets,
    generate_fallback_routine,
    score_candidate_exercises,
    split_label_to_muscles,
    get_mapped_targets,
    validate_and_repair_routine_output,
)


CASE_DIR = Path(__file__).parent / "prompt_cases"

CANDIDATE_CATALOG = {
    "barbell_bench_press": {
        "id": "barbell_bench_press",
        "name_kr": "Barbell Bench Press",
        "name_en": "Barbell Bench Press",
        "primary_muscle": "CHEST_MID",
        "secondary_muscle": "ARM_TRICEPS",
        "equipment_req": "BARBELL",
        "efficiency_tier": 5,
        "movement_type": "COMPOUND",
        "pain_triggers": None,
    },
    "pushup": {
        "id": "pushup",
        "name_kr": "Push-up",
        "name_en": "Push-up",
        "primary_muscle": "CHEST_MID",
        "secondary_muscle": "ARM_TRICEPS",
        "equipment_req": "BODYWEIGHT",
        "efficiency_tier": 3,
        "movement_type": "COMPOUND",
        "pain_triggers": None,
    },
    "painful_press": {
        "id": "painful_press",
        "name_kr": "Painful Press",
        "name_en": "Painful Press",
        "primary_muscle": "CHEST_MID",
        "secondary_muscle": "SHOULDER_FRONT",
        "equipment_req": "DUMBBELL",
        "efficiency_tier": 4,
        "movement_type": "COMPOUND",
        "pain_triggers": "shoulder",
    },
    "barbell_squat": {
        "id": "barbell_squat",
        "name_kr": "Barbell Squat",
        "name_en": "Barbell Squat",
        "primary_muscle": "LEG_QUADS",
        "secondary_muscle": "LEG_GLUTES",
        "equipment_req": "BARBELL",
        "efficiency_tier": 5,
        "movement_type": "COMPOUND",
        "pain_triggers": None,
    },
    "cable_pushdown": {
        "id": "cable_pushdown",
        "name_kr": "Cable Pushdown",
        "name_en": "Cable Pushdown",
        "primary_muscle": "ARM_TRICEPS",
        "secondary_muscle": None,
        "equipment_req": "CABLE",
        "efficiency_tier": 4,
        "movement_type": "ISOLATION",
        "pain_triggers": None,
    },
}


def _load_cases():
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(CASE_DIR.glob("*.json"))]


@pytest.mark.parametrize("case", _load_cases(), ids=lambda case: case["name"])
def test_prompt_case_constraints(case):
    req = RoutineRequest.model_validate(case["request"])
    candidates = [CANDIDATE_CATALOG[candidate_id] for candidate_id in case["candidates"]]
    expected = case["expected"]

    if not candidates:
        result = generate_fallback_routine(req, [], max_total_sets=10)
        assert result.generation_status == expected["status"]
        assert result.status_reason_code == expected["reason"]
        return

    if req.target_split_label:
        target_muscles = split_label_to_muscles(req.target_split_label)
    else:
        _, target_muscles = get_mapped_targets(req.target_muscles)

    pain_areas = [PainAreaEntry.model_validate(item) for item in case["request"].get("pain_areas", [])]
    ranked = score_candidate_exercises(
        candidates,
        target_muscles,
        doms_db=req.doms_data,
        blocked_equipment=req.equipment,
        pain_areas=pain_areas,
    )

    max_total_sets = max(1, int(req.time_available_min / ((60 + 90) / 60)))
    llm_case = case["llm"]
    llm_output = LLMRoutineOutput(
        total_estimated_time=req.time_available_min,
        summary_title=case["name"],
        rationale_summary=["case"],
        warnings=[],
        exercises=[
            LLMExercisePlan(
                exercise_id=llm_case["exercise_id"],
                exercise_name=llm_case["exercise_id"],
                primary_muscles=[],
                target_weight_kg=50,
                target_reps=8,
                sets=llm_case["sets"],
                rest_time_sec=llm_case["rest_time_sec"],
                exercise_rationale="case",
            )
        ],
    )
    validated = validate_and_repair_routine_output(
        llm_output,
        ranked,
        req.equipment,
        pain_areas,
        doms_db=req.doms_data,
        max_total_sets=max_total_sets,
        time_available_min=req.time_available_min,
    )

    if expected["status"] == "fallback":
        assert validated is None
        return

    assert validated is not None
    deterministic = apply_deterministic_targets(validated, req.goal or "hypertrophy", [], None)
    exercise = deterministic.exercises[0]
    assert expected["status"] == "success"
    if "exercise_id" in expected:
        assert exercise.exercise_id == expected["exercise_id"]
    if "max_sets" in expected:
        assert exercise.sets <= expected["max_sets"]
    if "rest_sec" in expected:
        assert exercise.rest_time_sec == expected["rest_sec"]
    if "primary_muscle" in expected:
        assert exercise.primary_muscles == [expected["primary_muscle"]]

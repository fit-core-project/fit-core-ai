from engines.routine_blueprint import build_routine_blueprint
from engines.routine_selection import (
    apply_blueprint_execution_policy,
    apply_blueprint_execution_policy_to_draft,
)
from engines.schemas import LLMRoutineOutput, RoutineDraftResponse


def _candidate(exercise_id, name, primary, movement_type, equipment="DUMBBELL"):
    return {
        "id": exercise_id,
        "name_kr": name,
        "primary_muscle": primary,
        "movement_type": movement_type,
        "equipment_req": equipment,
        "score": 100,
    }


def _plan(candidate, *, sets=4, rest=120):
    return {
        "exercise_id": candidate["id"],
        "exercise_name": candidate["name_kr"],
        "movement_type": candidate["movement_type"],
        "primary_muscles": [candidate["primary_muscle"]],
        "equipment_type": candidate["equipment_req"],
        "target_reps": 8,
        "sets": sets,
        "rest_time_sec": rest,
        "exercise_rationale": "test",
    }


def _output(candidates):
    return LLMRoutineOutput(
        total_estimated_time=0,
        summary_title="test",
        rationale_summary=[],
        warnings=[],
        exercises=[_plan(candidate) for candidate in candidates],
    )


def _draft(candidates):
    return RoutineDraftResponse.model_validate({
        "generation_status": "fallback",
        "status_reason_code": "networkError",
        "is_fallback": True,
        "summary_title": "test",
        "rationale_summary": [],
        "routine_blocks": [
            {
                "order": index,
                "exercise_id": candidate["id"],
                "exercise_name": candidate["name_kr"],
                "movement_type": candidate["movement_type"],
                "primary_muscles": [candidate["primary_muscle"]],
                "equipment_type": candidate["equipment_req"],
                "default_rest_sec": 120,
                "prescription": [
                    {
                        "set_index": set_index,
                        "target_reps": 8,
                        "target_rir": 2,
                        "target_rest_sec": 120,
                    }
                    for set_index in range(1, 5)
                ],
                "exercise_rationale": "test",
            }
            for index, candidate in enumerate(candidates, start=1)
        ],
    })


def _upper_blueprint(structure, *, readiness="normal", time=60):
    return build_routine_blueprint(
        target_split_label="upper",
        target_muscles=[],
        goal="strength",
        experience_level="advanced",
        readiness_level=readiness,
        time_available_min=time,
        session_structure=structure,
    )


def _upper_candidates():
    return [
        _candidate("push_main", "Dumbbell bench press", "chest", "COMPOUND"),
        _candidate("pull_main", "Dumbbell bent over row", "upper-back", "COMPOUND"),
        _candidate("optional_compound", "Dumbbell chest press", "chest", "COMPOUND"),
        _candidate("optional_accessory", "Dumbbell biceps curl", "biceps", "ISOLATION"),
    ]


def _assign_upper_roles(output):
    output._selection_phase_by_exercise_id = {
        "push_main": "main_compound",
        "pull_main": "secondary_compound",
        "optional_compound": "secondary_compound",
        "optional_accessory": "isolation_accessory",
    }
    output._selection_slot_by_exercise_id = {
        "push_main": "upper_push_main",
        "pull_main": "upper_pull_main",
        "optional_compound": "flex",
        "optional_accessory": "flex",
    }


def test_density_keeps_all_required_upper_main_roles_single_without_capping_them():
    candidates = _upper_candidates()
    output = _output(candidates)
    _assign_upper_roles(output)

    adjusted = apply_blueprint_execution_policy(output, _upper_blueprint("density"))
    metadata = adjusted._execution_metadata_by_exercise_id

    for exercise_id in ("push_main", "pull_main"):
        exercise = next(item for item in adjusted.exercises if item.exercise_id == exercise_id)
        assert metadata[exercise_id]["executionGroupType"] == "single"
        assert exercise.sets == 4
        assert exercise.rest_time_sec == 120
    assert metadata["optional_compound"]["executionGroupType"] == "density"
    assert metadata["optional_accessory"]["executionGroupType"] == "density"
    assert all(
        exercise.sets == 2
        for exercise in adjusted.exercises
        if exercise.exercise_id in {"optional_compound", "optional_accessory"}
    )


def test_fallback_reconstructs_required_upper_slots_before_density_grouping():
    candidates = _upper_candidates()
    draft = _draft(candidates)

    adjusted = apply_blueprint_execution_policy_to_draft(
        draft,
        _upper_blueprint("density"),
        candidates,
    )
    blocks = {block.exercise_id: block for block in adjusted.routine_blocks}

    for exercise_id in ("push_main", "pull_main"):
        assert blocks[exercise_id].execution_group_type == "single"
        assert len(blocks[exercise_id].prescription) == 4
        assert blocks[exercise_id].default_rest_sec == 120
    assert blocks["optional_compound"].execution_group_type == "density"
    assert blocks["optional_accessory"].execution_group_type == "density"


def test_fallback_reconstruction_uses_selected_non_top_ranked_required_candidates():
    selected_candidates = _upper_candidates()
    ranked_candidates = [
        _candidate("removed_push", "Dumbbell incline bench press", "chest", "COMPOUND"),
        _candidate("removed_pull", "Dumbbell single arm row", "upper-back", "COMPOUND"),
        *selected_candidates,
    ]

    adjusted = apply_blueprint_execution_policy_to_draft(
        _draft(selected_candidates),
        _upper_blueprint("density"),
        ranked_candidates,
    )
    blocks = {block.exercise_id: block for block in adjusted.routine_blocks}

    assert blocks["push_main"].execution_group_type == "single"
    assert blocks["pull_main"].execution_group_type == "single"
    assert blocks["optional_compound"].execution_group_type == "density"
    assert blocks["optional_accessory"].execution_group_type == "density"


def test_grouped_modes_report_no_group_when_only_required_mains_and_one_accessory_remain():
    candidates = _upper_candidates()[:2] + [_upper_candidates()[-1]]
    output = _output(candidates)
    output._selection_phase_by_exercise_id = {
        "push_main": "main_compound",
        "pull_main": "secondary_compound",
        "optional_accessory": "isolation_accessory",
    }
    output._selection_slot_by_exercise_id = {
        "push_main": "upper_push_main",
        "pull_main": "upper_pull_main",
        "optional_accessory": "flex",
    }

    for structure in ("density", "circuit", "superset"):
        adjusted = apply_blueprint_execution_policy(output.model_copy(deep=True), _upper_blueprint(structure))

        assert adjusted._selection_diagnostics["grouped_exercise_count"] == 0
        assert adjusted._selection_diagnostics["required_main_single_count"] == 2
        assert all(
            item["executionGroupType"] == "single"
            for item in adjusted._execution_metadata_by_exercise_id.values()
        )
        assert any("정책에 맞게 묶을 수 있는 운동 조합이 없어" in line for line in adjusted.rationale_summary)


def test_circuit_keeps_required_secondary_main_single_at_low_readiness():
    candidates = _upper_candidates()
    output = _output(candidates)
    _assign_upper_roles(output)

    adjusted = apply_blueprint_execution_policy(
        output,
        _upper_blueprint("circuit", readiness="low", time=30),
    )
    metadata = adjusted._execution_metadata_by_exercise_id

    assert metadata["push_main"]["executionGroupType"] == "single"
    assert metadata["pull_main"]["executionGroupType"] == "single"
    assert metadata["optional_compound"]["executionGroupType"] == "circuit"
    assert metadata["optional_accessory"]["executionGroupType"] == "circuit"


def test_fallback_reconstruction_protects_full_body_roles_without_reclassifying_extra_compounds():
    candidates = [
        _candidate("lower_main", "Dumbbell squat", "quadriceps", "COMPOUND"),
        _candidate("push_main", "Dumbbell bench press", "chest", "COMPOUND"),
        _candidate("pull_main", "Dumbbell bent over row", "upper-back", "COMPOUND"),
        _candidate("optional_chest", "Dumbbell chest press", "chest", "COMPOUND"),
        _candidate("optional_accessory", "Dumbbell biceps curl", "biceps", "ISOLATION"),
    ]
    blueprint = build_routine_blueprint(
        target_split_label="full_body",
        target_muscles=[],
        goal="strength",
        experience_level="advanced",
        readiness_level="normal",
        time_available_min=60,
        session_structure="density",
    )

    adjusted = apply_blueprint_execution_policy_to_draft(_draft(candidates), blueprint, candidates)
    blocks = {block.exercise_id: block for block in adjusted.routine_blocks}

    assert all(
        blocks[exercise_id].execution_group_type == "single"
        for exercise_id in ("lower_main", "push_main", "pull_main")
    )
    assert blocks["optional_chest"].execution_group_type == "density"
    assert blocks["optional_accessory"].execution_group_type == "density"
    assert blocks["optional_chest"].session_phase == "secondary_compound"


def test_explicit_main_antagonist_superset_remains_allowed_while_accessory_superset_is_eligible():
    candidates = _upper_candidates()
    output = _output(candidates)
    _assign_upper_roles(output)
    blueprint = build_routine_blueprint(
        target_split_label="chest_back",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="advanced",
        readiness_level="normal",
        time_available_min=60,
        session_structure="superset",
    )
    output._selection_slot_by_exercise_id["push_main"] = "upper_push_main"
    output._selection_slot_by_exercise_id["pull_main"] = "upper_pull_main"

    adjusted = apply_blueprint_execution_policy(output, blueprint)
    metadata = adjusted._execution_metadata_by_exercise_id

    assert metadata["push_main"]["executionGroupType"] == "superset"
    assert metadata["pull_main"]["executionGroupType"] == "superset"
    assert metadata["optional_compound"]["executionGroupType"] == "single"
    assert metadata["optional_accessory"]["executionGroupType"] == "single"


def test_standard_execution_preserves_sets_rest_and_order_for_required_mains():
    candidates = _upper_candidates()
    output = _output(candidates)
    _assign_upper_roles(output)

    adjusted = apply_blueprint_execution_policy(output, _upper_blueprint("standard"))

    assert [exercise.exercise_id for exercise in adjusted.exercises] == [
        "push_main",
        "pull_main",
        "optional_compound",
        "optional_accessory",
    ]
    assert all(exercise.sets == 4 for exercise in adjusted.exercises)
    assert all(exercise.rest_time_sec == 120 for exercise in adjusted.exercises)
    assert all(
        item["executionGroupType"] == "single"
        for item in adjusted._execution_metadata_by_exercise_id.values()
    )

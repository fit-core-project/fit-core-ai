"""Model ID selection must not choose a different base dose than fallback."""
import pytest

from engines.prescription.params import GOAL_PARAMS
from engines.prescription.targets import apply_deterministic_targets
from engines.routine_blueprint import build_blueprint_candidate_bundles, build_routine_blueprint
from engines.routine_selection import adapt_slot_selection_output
from engines.schemas import LLMRoutineSelectionOutput


def selected_plan(goal, experience, readiness):
    blueprint = build_routine_blueprint(
        target_split_label="chest", target_muscles=["chest"], goal=goal,
        experience_level=experience, readiness_level=readiness, time_available_min=60,
    )
    candidates = [{
        "id": "synthetic_chest_press", "name_kr": "머신 체스트 프레스",
        "primary_muscle": "chest", "movement_type": "COMPOUND",
        "movement_pattern": "horizontal_press", "equipment_req": "MACHINE", "score": 100,
    }]
    bundles = build_blueprint_candidate_bundles(blueprint, candidates, top_k_per_slot=1)
    selection = LLMRoutineSelectionOutput.model_validate({
        "summaryTitle": "합성 검증", "rationaleSummary": [], "warnings": [],
        "selections": [{"slotId": blueprint.slots[0].slot_id,
                        "exerciseId": "synthetic_chest_press", "exerciseRationale": "검증"}],
    })
    output = adapt_slot_selection_output(selection, blueprint=blueprint,
                                        slot_candidate_bundles=bundles, ranked_candidates=candidates)
    assert len(output.exercises) == 1
    return output


@pytest.mark.parametrize("goal", ["strength", "hypertrophy"])
@pytest.mark.parametrize("experience", ["beginner", "intermediate", "advanced"])
@pytest.mark.parametrize("readiness", ["low", "normal", "high"])
@pytest.mark.parametrize("doms", [0, 1, 2])
def test_slot_id_selection_uses_existing_goal_base_before_adjustments(goal, experience, readiness, doms):
    model_selected = selected_plan(goal, experience, readiness)
    goal_seeded = model_selected.model_copy(deep=True)
    goal_seeded.exercises[0].sets = GOAL_PARAMS[goal]["sets"]
    kwargs = dict(goal=goal, recent_sets=None, profile=None, readiness_level=readiness,
                  experience_level=experience, target_split_label="chest", target_muscles=["chest"],
                  doms_level_by_exercise_id={"synthetic_chest_press": doms})
    actual = apply_deterministic_targets(model_selected, **kwargs)
    expected = apply_deterministic_targets(goal_seeded, **kwargs)
    # Compare the existing goal policy, not a newly proposed larger dose.
    assert actual.exercises[0].model_dump() == expected.exercises[0].model_dump()

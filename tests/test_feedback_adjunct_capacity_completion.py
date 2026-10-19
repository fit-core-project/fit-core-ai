import pytest

from engines.routine_blueprint import build_routine_blueprint
from engines.routine_selection import adapt_slot_selection_output
from engines.schemas import LLMRoutineSelectionOutput


def blueprint(options=(), **changes):
    params = dict(
        target_split_label="full_body", target_muscles=[], goal="generalFitness",
        experience_level="intermediate", readiness_level="normal",
        time_available_min=60, adjunct_options=list(options),
    )
    params.update(changes)
    return build_routine_blueprint(**params)


@pytest.mark.parametrize("options", [
    ["conditioning_finisher"], ["cooldown_stretch"],
    ["mobility_warmup"], ["mobility_warmup", "cooldown_stretch"],
])
@pytest.mark.parametrize("experience", ["beginner", "intermediate", "advanced"])
def test_planned_adjunct_does_not_compete_with_excess_optional_main_slots(options, experience):
    bp = blueprint(options, experience_level=experience)
    main = [slot for slot in bp.slots if not slot.adjunct_option]
    assert len(main) <= bp.target_main_exercise_count
    assert len(bp.slots) <= bp.target_exercise_count
    assert len(main) >= bp.minimum_main_exercise_count
    original = blueprint(experience_level=experience)
    required = {s.slot_id for s in original.slots if s.requirement == "required_when_safe"}
    assert required <= {s.slot_id for s in main}
    assert {s.adjunct_option for s in bp.slots if s.adjunct_option} == set(bp.planned_adjunct_options)


def test_fullbody_conditioning_reserves_space_by_omitting_optional_core_slot():
    bp = blueprint(["conditioning_finisher"])
    ids = {s.slot_id for s in bp.slots}
    assert "adjunct_conditioning_finisher" in ids
    assert "full_core_accessory" not in ids
    assert {"full_lower_main", "full_push_main", "full_pull_main",
            "full_posterior_chain", "full_vertical_push"} <= ids
    assert bp.target_main_exercise_count == 5
    assert bp.target_exercise_count == 6


def test_no_adjunct_keeps_the_existing_fullbody_role_choices():
    bp = blueprint()
    assert {s.slot_id for s in bp.slots} == {
        "full_lower_main", "full_push_main", "full_pull_main",
        "full_posterior_chain", "full_vertical_push", "full_core_accessory",
    }
    assert bp.target_main_exercise_count == 6


def test_specialization_required_slot_is_not_removed_to_make_adjunct_space():
    bp = blueprint(["cooldown_stretch"], target_split_label="pull",
                   experience_level="advanced", specialization_focus="midBackEmphasis")
    assert "specialization_focus" in {s.slot_id for s in bp.slots}
    assert len(bp.slots) <= bp.target_exercise_count


def test_all_explicit_model_choices_keep_the_requested_finisher(monkeypatch):
    from engines import routine_selection

    bp = blueprint(["conditioning_finisher"])
    candidates, bundles, choices = [], [], []
    for i, slot in enumerate(bp.slots):
        eid = f"exercise-{i}"
        candidate = dict(id=eid, name_kr=eid, primary_muscle=next(iter(slot.target_muscles), "abs"),
                         equipment_req="DUMBBELL", movement_type="ISOLATION" if slot.adjunct_option else "COMPOUND")
        candidates.append(candidate)
        bundles.append(dict(slotId=slot.slot_id, candidates=[dict(exerciseId=eid)]))
        choices.append(dict(slotId=slot.slot_id, exerciseId=eid))
    # This test isolates count allocation; catalog/health validation is covered separately.
    monkeypatch.setattr(routine_selection, "_candidate_family", lambda c, _bp: c["id"])
    monkeypatch.setattr(routine_selection, "_candidate_pattern", lambda c: c["id"])
    result = adapt_slot_selection_output(
        LLMRoutineSelectionOutput.model_validate(dict(selections=choices)),
        blueprint=bp, slot_candidate_bundles=bundles, ranked_candidates=candidates,
    )
    assert len(result.exercises) <= bp.target_exercise_count
    assert set(result._adjunct_option_by_exercise_id.values()) == {"conditioning_finisher"}
    selected = {p.exercise_id.lower() for p in result.exercises}
    assert set(result._adjunct_option_by_exercise_id) <= selected


def test_health_blocked_required_role_is_not_reclassified_or_removed():
    bp = blueprint(["cooldown_stretch"], doms_data={"chest": 3})
    assert "chest" in bp.safety_limited_target_muscles
    slot = next(s for s in bp.slots if s.slot_id == "full_push_main")
    assert slot.requirement == "required_when_safe"

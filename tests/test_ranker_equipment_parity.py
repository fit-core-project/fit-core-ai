from itertools import product

import pytest

from engines.candidate_ranker import _candidate_is_safe, score_candidate_exercises
from engines.equipment_policy import select_equipment_option
from engines.schemas import PainAreaEntry


def _candidate(exercise_id, *, equipment, selected_equipment_code=None):
    candidate = {
        "id": exercise_id, "name_kr": "test chest press", "name_en": "Test Chest Press",
        "primary_muscle": "chest", "secondary_muscle": "triceps", "equipment_req": equipment,
        "difficulty_tier": 3, "efficiency_tier": 3, "movement_type": "COMPOUND",
        "pain_triggers": "", "lumbar_load": "low", "axial_load": "low", "knee_shear_load": "low",
        "shoulder_impingement_risk": "low", "wrist_stress": "low", "ankle_dorsiflexion_demand": "low",
        "overhead_position_required": 0, "technical_difficulty": "low", "balance_requirement": "low",
        "failure_penalty": "low", "injury_caution_level": "low", "hypertrophy_effect": "high",
        "strength_effect": "high", "stimulus_to_fatigue": "good",
    }
    if selected_equipment_code is not None:
        candidate["selected_equipment_code"] = selected_equipment_code
    return candidate


_RAW_EQUIPMENT = ("BAND,CABLE", "BODYWEIGHT,DUMBBELL", "SMITH_MACHINE,MACHINE", "CABLE")
_BLOCKED_SETS = ((), ("BAND",), ("CABLE",), ("DUMBBELL",), ("MACHINE",), ("SMITH_MACHINE",), ("BAND", "CABLE"), ("BODYWEIGHT", "DUMBBELL"))
_WHITELIST_SETS = ((), ("BAND",), ("CABLE",), ("DUMBBELL",), ("MACHINE",), ("SMITH_MACHINE",), ("BODYWEIGHT",), ("BAND", "CABLE"), ("MACHINE", "BODYWEIGHT"))


@pytest.mark.parametrize(
    ("equipment", "blocked", "available"),
    list(product(_RAW_EQUIPMENT, _BLOCKED_SETS, _WHITELIST_SETS)),
)
def test_candidate_safety_matches_equipment_option_policy_matrix(equipment, blocked, available):
    candidate = _candidate("equipment_matrix", equipment=equipment)
    expected = select_equipment_option(equipment, blocked=blocked, available=available)
    assert _candidate_is_safe(candidate, blocked, [], available) is (expected is not None)


def test_ranker_uses_current_cable_option_without_mutating_input():
    candidate = _candidate("band_or_cable", equipment="BAND,CABLE", selected_equipment_code="BAND")
    ranked = score_candidate_exercises([candidate], ["chest"], blocked_equipment=["BAND"], available_equipment=["CABLE"], top_n=1)
    assert candidate["selected_equipment_code"] == "BAND"
    assert ranked[0]["id"] == "band_or_cable"
    assert ranked[0]["selected_equipment_code"] == "CABLE"


def test_ranker_uses_current_bodyweight_option_without_loadable_bonus():
    candidate = _candidate("bodyweight_or_dumbbell", equipment="BODYWEIGHT,DUMBBELL", selected_equipment_code="DUMBBELL")
    ranked = score_candidate_exercises([candidate], ["chest"], blocked_equipment=["DUMBBELL"], available_equipment=["MACHINE"], top_n=1)
    assert ranked[0]["selected_equipment_code"] == "BODYWEIGHT"
    assert "loadable equipment priority" not in ranked[0]["score_reasons"]
    assert all(boost["rule_code"] != "loadable_equipment" for boost in ranked[0]["score_boosts"])


def test_ranker_uses_current_smith_option_not_stale_generic_machine_marker():
    candidate = _candidate("smith_only", equipment="SMITH_MACHINE,MACHINE", selected_equipment_code="MACHINE")
    ranked = score_candidate_exercises([candidate], ["chest"], available_equipment=["SMITH_MACHINE"], top_n=1)

    assert candidate["selected_equipment_code"] == "MACHINE"
    assert ranked[0]["selected_equipment_code"] == "SMITH_MACHINE"


def test_stale_selected_equipment_code_does_not_bypass_current_constraints():
    candidate = _candidate("stale_equipment", equipment="CABLE", selected_equipment_code="BODYWEIGHT")
    assert _candidate_is_safe(candidate, [], [], ["BODYWEIGHT"]) is False
    assert score_candidate_exercises([candidate], ["chest"], available_equipment=["BODYWEIGHT"], top_n=1) == []


def test_ranker_rejects_unrepresented_equipment_despite_stale_bodyweight_selection():
    candidate = _candidate("unrepresented", equipment="UNKNOWN", selected_equipment_code="BODYWEIGHT")
    assert _candidate_is_safe(candidate, [], [], ["BODYWEIGHT"]) is False
    assert score_candidate_exercises([candidate], ["chest"], available_equipment=["BODYWEIGHT"], top_n=1) == []


def test_injury_restriction_rejects_candidate_despite_viable_equipment_alternative():
    candidate = _candidate("injured_chest", equipment="BAND,CABLE")
    assert _candidate_is_safe(candidate, ["BAND"], [PainAreaEntry(body_part="chest")], ["CABLE"]) is False

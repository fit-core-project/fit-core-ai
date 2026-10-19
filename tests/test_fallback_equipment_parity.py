import copy

import pytest

from engines.fallback import _candidate_order_key, generate_fallback_routine
from engines.schemas import PainAreaEntry, RoutineRequest


def _candidate(raw, selected=None):
    return {
        "id": "equipment-parity",
        "name_kr": "Chest Press",
        "name_en": "Chest Press",
        "primary_muscle": "chest",
        "secondary_muscle": "triceps",
        "movement_type": "COMPOUND",
        "equipment_req": raw,
        "selected_equipment_code": selected,
        "efficiency_tier": 3,
        "difficulty_tier": 2,
        "pain_triggers": None,
    }


def _run(candidate, blocked, available, pain=None):
    before = copy.deepcopy(candidate)
    request = RoutineRequest(
        user_id="equipment-parity", goal="hypertrophy", time_available_min=35,
        target_muscles=["chest"], equipment=blocked,
        available_equipment=available, pain_areas=pain or [],
    )
    result = generate_fallback_routine(request, [candidate], max_total_sets=8)
    assert candidate == before
    return result


@pytest.mark.parametrize("raw,stale,blocked,available,expected", [
    ("BAND,CABLE", "BAND", ["BAND"], ["CABLE"], "CABLE"),
    ("DUMBBELL,BODYWEIGHT", "DUMBBELL", ["DUMBBELL"], ["MACHINE"], "BODYWEIGHT"),
    ("SMITH_MACHINE,MACHINE", "MACHINE", [], ["SMITH_MACHINE"], "SMITH_MACHINE"),
    ("CABLE", None, [], [], "CABLE"),
])
def test_fallback_emits_only_current_usable_equipment(raw, stale, blocked, available, expected):
    result = _run(_candidate(raw, stale), blocked, available)
    assert len(result.routine_blocks) == 1
    assert result.routine_blocks[0].equipment_type == expected


@pytest.mark.parametrize("raw,stale,blocked,available", [
    ("SMITH_MACHINE,MACHINE", "MACHINE", [], ["MACHINE"]),
    ("BAND,CABLE", "CABLE", ["BAND", "CABLE"], ["CABLE"]),
    ("DUMBBELL", "BODYWEIGHT", [], ["MACHINE"]),
])
def test_fallback_rejects_unavailable_options_despite_stale_selection(raw, stale, blocked, available):
    result = _run(_candidate(raw, stale), blocked, available)
    assert result.routine_blocks == []
    assert result.generation_status == "failed"
    assert result.status_reason_code == "emptyCandidate"
    assert result.total_estimated_time == 0


def test_fallback_cannot_bypass_unrepresented_equipment_with_bodyweight_selection():
    result = _run(_candidate("UNKNOWN", "BODYWEIGHT"), [], ["BODYWEIGHT"])
    assert result.routine_blocks == []
    assert result.generation_status == "failed"
    assert result.status_reason_code == "emptyCandidate"


def test_fallback_equipment_alternative_does_not_override_injury():
    result = _run(_candidate("BAND,CABLE"), ["BAND"], ["CABLE"], [PainAreaEntry(body_part="chest")])
    assert result.routine_blocks == []
    assert result.status_reason_code == "emptyCandidate"


def test_fallback_order_does_not_prioritize_an_unused_loadable_option():
    mixed = _candidate("DUMBBELL,BODYWEIGHT", "BODYWEIGHT")
    bodyweight = _candidate("BODYWEIGHT", "BODYWEIGHT")
    assert _candidate_order_key(mixed) == _candidate_order_key(bodyweight)

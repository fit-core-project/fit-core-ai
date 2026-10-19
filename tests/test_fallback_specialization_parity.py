import pytest

from engines.fallback import generate_fallback_routine
from engines.routine_blueprint import build_routine_blueprint
from engines.schemas import RoutineRequest


def _candidate(exercise_id, name_kr, movement_type, score):
    return {
        "id": exercise_id,
        "name_kr": name_kr,
        "name_en": name_kr,
        "primary_muscle": "chest",
        "secondary_muscle": "triceps",
        "equipment_req": "DUMBBELL",
        "difficulty_tier": 2,
        "efficiency_tier": 3,
        "movement_type": movement_type,
        "pain_triggers": None,
        "score": score,
    }


CHEST_CANDIDATES = [
    _candidate("incline", "덤벨 인클라인 체스트 프레스", "COMPOUND", 300),
    _candidate("flat", "덤벨 플랫 체스트 프레스", "COMPOUND", 290),
    _candidate("decline", "덤벨 디클라인 체스트 프레스", "COMPOUND", 280),
    _candidate("fly", "덤벨 플라이", "ISOLATION", 270),
]


def _fallback(*, focus=None, readiness="normal", doms=None, max_total_sets=None,
              candidates=None, time_available_min=60):
    request = RoutineRequest(
        user_id="fallback-specialization-parity",
        target_split_label="chest",
        target_muscles=[],
        time_available_min=time_available_min,
        goal="hypertrophy",
        experience_level="advanced",
        readiness_level=readiness,
        specialization_focus=focus,
        pain_areas=[],
        doms_data=doms or {},
        equipment=[],
    )
    blueprint = build_routine_blueprint(
        target_split_label="chest",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="advanced",
        readiness_level=readiness,
        time_available_min=time_available_min,
        specialization_focus=focus,
        doms_data=doms or {},
    )
    return generate_fallback_routine(
        request,
        candidates=CHEST_CANDIDATES if candidates is None else candidates,
        max_total_sets=blueprint.max_total_sets if max_total_sets is None else max_total_sets,
        doms_db=doms or {},
        goal="hypertrophy",
        target_exercise_count=3,
        minimum_main_exercise_count=blueprint.minimum_main_exercise_count,
        routine_blueprint=blueprint,
    )


def _sets_by_id(result):
    return {
        block.exercise_id: len(block.prescription)
        for block in result.routine_blocks
    }


def test_fallback_applies_upper_chest_transfer_after_productive_fill():
    balanced = _fallback()
    focused = _fallback(focus="upperChest")

    assert _sets_by_id(balanced) == {"flat": 4, "incline": 4, "fly": 3}
    assert _sets_by_id(focused) == {"incline": 5, "flat": 4, "fly": 2}
    assert sum(_sets_by_id(focused).values()) == sum(_sets_by_id(balanced).values())
    assert "윗가슴 집중" in next(
        block.exercise_rationale
        for block in focused.routine_blocks
        if block.exercise_id == "incline"
    )
    assert focused._fallback_selection_diagnostics['role_set_capacity_exhausted'] is True
    assert focused._fallback_selection_diagnostics['role_set_capacity_total_budget_exhausted'] is False


@pytest.mark.parametrize('readiness,doms', [
    ('low', {}), ('normal', {'triceps': 1}), ('normal', {'chest': 1}),
    ('normal', {'chest': 2}), ('normal', {'chest': 3}),
])
def test_fallback_focus_does_not_override_low_readiness_or_doms_restrictions(readiness, doms):
    balanced = _fallback(readiness=readiness, doms=doms)
    focused = _fallback(focus="upperChest", readiness=readiness, doms=doms)
    assert _sets_by_id(focused) == _sets_by_id(balanced)
    assert "직접 세트를 1세트" not in " ".join(
        block.exercise_rationale for block in focused.routine_blocks
    )
    normal_by_id = {b.exercise_id: b for b in balanced.routine_blocks}
    for block in focused.routine_blocks:
        assert block.prescription == normal_by_id[block.exercise_id].prescription


def test_fallback_focus_respects_total_set_cap_and_donor_remains_nonempty():
    focused = _fallback(focus="upperChest", max_total_sets=11)
    sets = _sets_by_id(focused)

    assert sum(sets.values()) <= 11
    assert sets["fly"] >= 1


@pytest.mark.parametrize('focus,exercise_id', [('midChest', 'flat'), ('lowerChest', 'decline')])
def test_other_chest_focus_changes_visible_distribution_once(focus, exercise_id):
    focused = _fallback(focus=focus)
    assert _sets_by_id(focused)[exercise_id] == 5
    assert sum(_sets_by_id(focused).values()) == 11
    assert len([b for b in focused.routine_blocks if '직접 세트를 1세트' in b.exercise_rationale]) == 1


def test_no_matching_focus_does_not_change_prescription():
    candidates = [c for c in CHEST_CANDIDATES if c['id'] != 'incline']
    balanced = _fallback(candidates=candidates)
    focused = _fallback(candidates=candidates, focus='upperChest')
    assert _sets_by_id(focused) == _sets_by_id(balanced)
    normal_by_id = {b.exercise_id: b for b in balanced.routine_blocks}
    for block in focused.routine_blocks:
        assert block.prescription == normal_by_id[block.exercise_id].prescription
    assert all('직접 세트를 1세트' not in b.exercise_rationale for b in focused.routine_blocks)


def test_focus_at_short_time_respects_request_budget():
    focused = _fallback(focus='upperChest', time_available_min=30)
    assert focused.total_estimated_time <= 30

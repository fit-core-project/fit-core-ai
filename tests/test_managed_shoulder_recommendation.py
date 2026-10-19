import pytest

from engines.candidate_ranker import score_candidate_exercises
from engines.injury_condition_policy import ResolvedInjuryCondition
from engines.schemas import InjuryConditionEntry


def _candidate(exercise_id: str, name: str, primary: str, *, plane: str = "unknown") -> dict:
    return {
        "id": exercise_id,
        "name_kr": name,
        "name_en": name,
        "primary_muscle": primary,
        "secondary_muscle": "",
        "equipment_req": "DUMBBELL",
        "difficulty_tier": 3,
        "efficiency_tier": 3,
        "pain_triggers": None,
        "movement_type": "ISOLATION",
        "shoulder_elevation_plane": plane,
        "shoulder_elevation_demand": "medium" if plane != "unknown" else "low",
        "shoulder_impingement_risk": "low",
        # An ordinary catalog admission flag must not override a constraint.
        "main_routine_default_allowed": 1,
    }


def _rank(candidates: list[dict], **kwargs: object) -> list[dict]:
    return score_candidate_exercises(
        candidates,
        ["front-deltoids", "side-deltoids", "chest"],
        top_n=12,
        **kwargs,
    )


def _managed_shoulder_condition(kind: str) -> object:
    payload = {
        "areaCode": "shoulder",
        "status": "managedWithRestrictions",
        "restrictedMovementPatterns": ["overheadPosition"],
    }
    if kind == "dict":
        return payload
    if kind == "pydantic":
        return InjuryConditionEntry.model_validate(payload)
    return ResolvedInjuryCondition(
        area_code="shoulder",
        status="managedWithRestrictions",
        restricted_movement_patterns=("overheadPosition",),
    )


@pytest.mark.parametrize("condition_kind", ["dict", "pydantic", "resolved"])
def test_effective_managed_shoulder_excludes_front_raise_for_all_condition_inputs(
    condition_kind: str,
):
    candidates = [
        _candidate("front_raise", "Dumbbell Front Raise", "front-deltoids", plane="sagittal"),
        _candidate("lateral_raise", "Dumbbell Lateral Raise", "side-deltoids", plane="frontal"),
        _candidate("chest_press", "Dumbbell Bench Press", "chest", plane="horizontal"),
    ]

    ranked = _rank(
        candidates,
        injury_conditions=[_managed_shoulder_condition(condition_kind)],
        shoulder_condition_status="recovered",
    )

    assert {candidate["id"] for candidate in ranked} == {
        "lateral_raise",
        "chest_press",
    }


def test_legacy_managed_shoulder_status_excludes_front_raise_without_blanket_upper_filter():
    candidates = [
        _candidate("front_raise", "Dumbbell Front Raise", "front-deltoids", plane="sagittal"),
        _candidate("lateral_raise", "Dumbbell Lateral Raise", "side-deltoids", plane="frontal"),
        _candidate("chest_press", "Dumbbell Bench Press", "chest", plane="horizontal"),
    ]

    ranked = _rank(
        candidates,
        shoulder_condition_status="managedWithRestrictions",
        restricted_movement_patterns=["overheadPosition"],
    )

    assert {candidate["id"] for candidate in ranked} == {
        "lateral_raise",
        "chest_press",
    }


@pytest.mark.parametrize("status", [None, "managedNoRestrictions", "recovered"])
def test_front_raise_remains_eligible_without_effective_managed_restrictions(status: str | None):
    front_raise = _candidate(
        "front_raise",
        "Dumbbell Front Raise",
        "front-deltoids",
        plane="sagittal",
    )

    ranked = _rank(
        [front_raise],
        shoulder_condition_status=status,
        specialization_focus="frontDeltoidEmphasis",
        doms_db={"chest": 3},
        injury_conditions=[
            {
                "areaCode": "knee",
                "status": "managedWithRestrictions",
                "restrictedMovementPatterns": ["deepKneeFlexion"],
            }
        ],
    )

    assert [candidate["id"] for candidate in ranked] == ["front_raise"]

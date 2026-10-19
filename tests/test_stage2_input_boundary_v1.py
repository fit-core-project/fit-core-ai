"""Approved DOMS boundary and existing input merge behavior, not HTTP integration."""
import pytest
from pydantic import ValidationError

from engines.public_contract import PublicRoutineGenerateRequest
from engines.registry import MUSCLE_SLUGS, PUBLIC_DOMS_BODY_PART_SLUGS, PUBLIC_PAIN_AREA_SLUGS
from engines.routine_pipeline import _merge_profile_injury_areas
from engines.schemas import PainAreaEntry, RoutineRequest


TRAINABLE = {
    "abductors", "abs", "adductor", "back-deltoids", "biceps", "calves",
    "chest", "forearm", "front-deltoids", "gluteal", "hamstring", "lower-back",
    "neck", "obliques", "quadriceps", "trapezius", "triceps", "upper-back",
}


def public_payload(**changes):
    return {"targetSplitLabel": "custom", "targetMuscles": ["calves"],
            "timeAvailableMin": 45, **changes}


def test_user_approved_doms_exception_keeps_other_domains_closed():
    assert MUSCLE_SLUGS == TRAINABLE
    assert PUBLIC_DOMS_BODY_PART_SLUGS == TRAINABLE | {"left-soleus", "right-soleus"}
    assert PUBLIC_PAIN_AREA_SLUGS == TRAINABLE | {"head", "knees", "left-soleus", "right-soleus"}


@pytest.mark.parametrize("part", sorted(TRAINABLE | {"left-soleus", "right-soleus"}))
@pytest.mark.parametrize("level", [1, 2, 3])
def test_doms_round_trip_preserves_area_level_and_never_creates_injury(part, level):
    # 2026-10-15 approval: the soleus exception is DOMS-only, not a new target.
    public = PublicRoutineGenerateRequest.model_validate(public_payload(
        currentDoms=[{"bodyPart": part, "level": level}]))
    request = RoutineRequest.model_validate(public.to_internal_request_payload())
    restored = RoutineRequest.model_validate_json(request.model_dump_json(by_alias=True))
    assert restored.doms_data == {part: level}
    assert restored.pain_areas == []
    assert restored.target_muscles == ["calves"]


@pytest.mark.parametrize("part", ["head", "knees", "side-deltoids", "left-soleus", "right-soleus"])
def test_doms_exception_does_not_expand_target(part):
    with pytest.raises(ValidationError):
        PublicRoutineGenerateRequest.model_validate(public_payload(targetMuscles=[part]))


@pytest.mark.parametrize("part", ["head", "knees", "side-deltoids"])
def test_non_doms_area_is_rejected(part):
    with pytest.raises(ValidationError):
        PublicRoutineGenerateRequest.model_validate(public_payload(currentDoms=[{"bodyPart": part, "level": 1}]))


def test_empty_doms_means_no_report_not_injury_or_zero_level():
    for changes in ({}, {"currentDoms": []}):
        public = PublicRoutineGenerateRequest.model_validate(public_payload(**changes))
        request = RoutineRequest.model_validate(public.to_internal_request_payload())
        assert request.doms_data == {} and request.pain_areas == []
    with pytest.raises(ValidationError):
        PublicRoutineGenerateRequest.model_validate(public_payload(currentDoms=[{"bodyPart": "calves", "level": 0}]))


def test_stored_goal_equipment_precedence_and_machine_smith_separation():
    public = PublicRoutineGenerateRequest.model_validate(public_payload(
        goal="hypertrophy", experienceLevel="advanced", availableEquipment=["MACHINE"]))
    mapped = public.to_internal_request_payload(stored_goal="strength",
        stored_experience_level="beginner", stored_available_equipment=["SMITH_MACHINE"])
    assert mapped["goal"] == "hypertrophy"
    assert mapped["experienceLevel"] == "advanced"
    assert mapped["availableEquipment"] == ["MACHINE"]
    # Existing [] semantics are profile fallback, not a claim of zero equipment.
    empty = PublicRoutineGenerateRequest.model_validate(public_payload(availableEquipment=[]))
    assert empty.to_internal_request_payload(stored_available_equipment=["DUMBBELL"])["availableEquipment"] == ["DUMBBELL"]


def test_empty_request_does_not_erase_stored_injury():
    stored = [PainAreaEntry(bodyPart="front-deltoids")]
    assert _merge_profile_injury_areas([], stored) == stored
    request = [PainAreaEntry(bodyPart="knees")]
    merged = _merge_profile_injury_areas(request, stored)
    assert {area.body_part for area in merged} == {"front-deltoids", "knees"}
    assert request[0].body_part == "knees" and stored[0].body_part == "front-deltoids"

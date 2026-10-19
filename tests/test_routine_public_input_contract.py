import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from engines.public_contract import (
    PUBLIC_ADJUNCT_OPTIONS,
    PUBLIC_CONTRACT_VERSION,
    PUBLIC_EQUIPMENT_CODES,
    PUBLIC_EXPERIENCE_LEVELS,
    PUBLIC_GOALS,
    PUBLIC_PROFILE_SIGNALS,
    PUBLIC_SPLIT_LABELS,
    PublicRoutineGenerateRequest,
)
from engines.registry import MUSCLE_SLUGS, PUBLIC_DOMS_BODY_PART_SLUGS, PUBLIC_PAIN_AREA_SLUGS
from engines.schemas import RoutineRequest


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "docs/contracts/routine-ai-core-public-generate-request-schema-v3-2026-07-22.json"


def full_public_payload() -> dict:
    return {
        "targetSplitLabel": "upper",
        "targetMuscles": ["chest", "upper-back", "biceps", "triceps"],
        "timeAvailableMin": 60,
        "readinessLevel": "normal",
        "currentPainAreas": [{"bodyPart": "knees"}],
        "currentDoms": [{"bodyPart": "chest", "level": 2}],
        "availableEquipment": ["DUMBBELL", "CABLE", "BODYWEIGHT", "BAND"],
        "unavailableEquipment": ["BARBELL", "SMITH_MACHINE"],
        "goal": "hypertrophy",
        "experienceLevel": "intermediate",
        "sessionStructure": "standard",
        "adjunctOptions": ["cooldown_stretch", "mobility_warmup"],
        "profileSignals": ["self_reported_long_arms"],
        "trainingStyle": "gym",
        "userNote": "어깨에 무리가 없는 범위에서 진행하고 싶어요.",
    }


def test_committed_public_schema_matches_executable_registries():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    assert PUBLIC_CONTRACT_VERSION in schema["$id"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["timeAvailableMin"]
    assert set(schema["properties"]) == {
        "targetSplitLabel",
        "targetMuscles",
        "timeAvailableMin",
        "readinessLevel",
        "currentPainAreas",
        "currentDoms",
        "availableEquipment",
        "unavailableEquipment",
        "goal",
        "experienceLevel",
        "sessionStructure",
        "adjunctOptions",
        "profileSignals",
        "trainingStyle",
        "userNote",
    }
    assert set(schema["properties"]["targetSplitLabel"]["enum"]) == {*PUBLIC_SPLIT_LABELS, None}
    assert set(schema["properties"]["goal"]["enum"]) == {*PUBLIC_GOALS, None}
    assert set(schema["properties"]["experienceLevel"]["enum"]) == {
        *PUBLIC_EXPERIENCE_LEVELS,
        None,
    }
    assert schema["properties"]["adjunctOptions"]["items"]["enum"] == list(PUBLIC_ADJUNCT_OPTIONS)
    assert set(schema["properties"]["profileSignals"]["items"]["enum"]) == set(PUBLIC_PROFILE_SIGNALS)
    assert set(schema["$defs"]["equipmentCode"]["enum"]) == set(PUBLIC_EQUIPMENT_CODES)
    assert set(schema["$defs"]["trainableMuscleSlug"]["enum"]) == MUSCLE_SLUGS
    assert set(schema["$defs"]["painAreaSlug"]["enum"]) == PUBLIC_PAIN_AREA_SLUGS
    assert set(schema["$defs"]["domsBodyPartSlug"]["enum"]) == PUBLIC_DOMS_BODY_PART_SLUGS
    assert schema["properties"]["timeAvailableMin"] == {
        "type": "integer",
        "minimum": 10,
        "maximum": 180,
    }
    for key in (
        "targetMuscles",
        "availableEquipment",
        "unavailableEquipment",
        "adjunctOptions",
        "profileSignals",
    ):
        assert schema["properties"][key]["uniqueItems"] is True
    assert schema["properties"]["currentPainAreas"]["x-fitcore-uniqueBy"] == "bodyPart"
    assert schema["properties"]["currentDoms"]["x-fitcore-uniqueBy"] == "bodyPart"
    assert schema["$defs"]["domsEntry"]["properties"]["level"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 3,
    }


def test_full_public_request_maps_to_valid_internal_request():
    public_request = PublicRoutineGenerateRequest.model_validate(full_public_payload())
    internal_payload = public_request.to_internal_request_payload(user_id="public-contract-test")
    internal_request = RoutineRequest.model_validate(internal_payload)

    assert internal_request.user_id == "public-contract-test"
    assert internal_request.pain_areas[0].body_part == "knees"
    assert internal_request.doms_data == {"chest": 2}
    assert internal_request.equipment == ["BARBELL", "SMITH_MACHINE"]
    assert internal_request.available_equipment == ["DUMBBELL", "CABLE", "BODYWEIGHT", "BAND"]
    assert internal_request.adjunct_options == ["mobility_warmup", "cooldown_stretch"]
    assert internal_request.profile_signals == ["self_reported_long_arms"]


def test_omitted_profile_values_have_one_explicit_fallback_chain():
    public_request = PublicRoutineGenerateRequest.model_validate({
        "targetSplitLabel": "push",
        "timeAvailableMin": 45,
    })

    stored = public_request.to_internal_request_payload(
        stored_goal="strength",
        stored_experience_level="advanced",
        stored_available_equipment=["BARBELL", "DUMBBELL"],
    )
    empty_profile = public_request.to_internal_request_payload()

    assert stored["goal"] == "strength"
    assert stored["experienceLevel"] == "advanced"
    assert stored["availableEquipment"] == ["BARBELL", "DUMBBELL"]
    assert empty_profile["goal"] == "hypertrophy"
    assert empty_profile["experienceLevel"] == "intermediate"
    assert empty_profile["availableEquipment"] == []


@pytest.mark.parametrize(
    "patch",
    [
        {"goal": "muscleGain"},
        {"experienceLevel": "expert"},
        {"targetSplitLabel": "random"},
        {"trainingStyle": "crossfit"},
        {"availableEquipment": ["UNKNOWN_TOOL"]},
        {"profileSignals": ["average_body"]},
        {"profileSignals": ["self_reported_long_lower_legs"]},
        {"currentPainAreas": [{"bodyPart": "knees", "severity": "mild"}]},
        {"currentDoms": {"chest": 2}},
        {"currentDoms": [{"bodyPart": "knees", "level": 2}]},
        {"timeAvailableMin": "45"},
        {"timeAvailableMin": 45.5},
        {"currentDoms": [{"bodyPart": "chest", "level": "2"}]},
        {"currentDoms": [{"bodyPart": "chest", "level": 2.5}]},
        {"targetMuscles": [" chest "]},
        {"currentPainAreas": [{"bodyPart": " knees "}]},
        {"currentDoms": [{"bodyPart": " chest ", "level": 2}]},
        {"profileSignals": [" self_reported_long_arms "]},
        {"profileSignals": None},
        {"unexpectedField": True},
    ],
)
def test_unknown_or_wrong_shape_public_values_are_rejected(patch):
    payload = {
        "targetSplitLabel": "push",
        "timeAvailableMin": 45,
        **patch,
    }

    with pytest.raises(ValidationError):
        PublicRoutineGenerateRequest.model_validate(payload)


@pytest.mark.parametrize(
    "patch",
    [
        {"targetSplitLabel": None, "targetMuscles": []},
        {"targetSplitLabel": "custom", "targetMuscles": []},
        {"availableEquipment": ["MACHINE"], "unavailableEquipment": ["MACHINE"]},
        {"adjunctOptions": ["conditioning_finisher", "endurance_support"]},
        {"readinessLevel": "low", "sessionStructure": "density"},
        {"readinessLevel": "low", "adjunctOptions": ["conditioning_finisher"]},
        {"goal": "strength", "sessionStructure": "circuit"},
        {
            "currentPainAreas": [
                {"bodyPart": "knees"},
                {"bodyPart": "knees"},
            ]
        },
        {
            "currentDoms": [
                {"bodyPart": "chest", "level": 1},
                {"bodyPart": "chest", "level": 2},
            ]
        },
        {"targetMuscles": ["chest", "chest"]},
        {"availableEquipment": ["MACHINE", "MACHINE"]},
        {"unavailableEquipment": ["CABLE", "CABLE"]},
        {"adjunctOptions": ["cooldown_stretch", "cooldown_stretch"]},
        {"profileSignals": ["self_reported_long_arms", "self_reported_long_arms"]},
    ],
)
def test_cross_field_public_contract_violations_are_rejected(patch):
    payload = {
        "targetSplitLabel": "push",
        "timeAvailableMin": 45,
        **patch,
    }

    with pytest.raises(ValidationError):
        PublicRoutineGenerateRequest.model_validate(payload)


def test_json_schema_integral_numbers_normalize_to_python_integers():
    request = PublicRoutineGenerateRequest.model_validate(
        {
            "targetSplitLabel": "push",
            "timeAvailableMin": 45.0,
            "currentDoms": [{"bodyPart": "chest", "level": 2.0}],
        }
    )

    assert request.time_available_min == 45
    assert type(request.time_available_min) is int
    assert request.current_doms[0].level == 2
    assert type(request.current_doms[0].level) is int


def test_stored_strength_fallback_is_revalidated_against_session_structure():
    public_request = PublicRoutineGenerateRequest.model_validate({
        "targetSplitLabel": "upper",
        "timeAvailableMin": 60,
        "sessionStructure": "circuit",
    })

    with pytest.raises(ValueError, match="resolved strength goal cannot use circuit"):
        public_request.to_internal_request_payload(stored_goal="strength")

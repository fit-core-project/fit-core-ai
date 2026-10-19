"""Executable FE/BE-facing request contract for the gym routine AI core.

The FastAPI endpoint still receives the internal ``RoutineRequest`` shape. This
module freezes the product JSON shape and its deterministic mapping so an
adapter can be verified without widening the AI endpoint contract.
"""

from __future__ import annotations

from typing import Iterable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.alias_generators import to_camel

from engines.equipment_policy import KNOWN_EQUIPMENT_CODES
from engines.option_resolver import (
    MAX_DISTINCT_ADJUNCT_OPTIONS,
    adjunct_option_conflict_code,
    canonical_adjunct_options,
)
from engines.profile_constraint_contract import resolve_profile_constraint_signals
from engines.registry import MUSCLE_SLUGS, PUBLIC_DOMS_BODY_PART_SLUGS, PUBLIC_PAIN_AREA_SLUGS


PUBLIC_CONTRACT_VERSION = "routine-ai-core-public-generate-request-v3-2026-07-22"
PUBLIC_SPLIT_LABELS = (
    "push",
    "pull",
    "legs",
    "upper",
    "chest_back",
    "lower",
    "chest",
    "back",
    "shoulder",
    "arm",
    "core",
    "neck",
    "full_body",
    "custom",
)
PUBLIC_GOALS = (
    "strength",
    "hypertrophy",
    "endurance",
    "fatLoss",
    "recomposition",
    "generalFitness",
)
PUBLIC_EXPERIENCE_LEVELS = ("beginner", "intermediate", "advanced")
PUBLIC_EQUIPMENT_CODES = tuple(sorted(KNOWN_EQUIPMENT_CODES))
PUBLIC_ADJUNCT_OPTIONS = (
    "mobility_warmup",
    "prehab_accessory",
    "conditioning_finisher",
    "endurance_support",
    "cooldown_stretch",
)
PUBLIC_PROFILE_SIGNALS = (
    "heel_down_deep_squat_difficult",
    "hip_hinge_difficult",
    "overhead_reach_difficult",
    "loaded_wrist_extension_difficult",
    "self_reported_long_arms",
    "self_reported_long_thighs",
)


def _normalize_json_schema_integer(value, *, field_name: str) -> int:
    """Match JSON Schema integer semantics without accepting numeric strings."""

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ValueError(f"{field_name} must be an integer")

TargetSplitLabel = Literal[
    "push",
    "pull",
    "legs",
    "upper",
    "chest_back",
    "lower",
    "chest",
    "back",
    "shoulder",
    "arm",
    "core",
    "neck",
    "full_body",
    "custom",
]
PublicGoal = Literal[
    "strength",
    "hypertrophy",
    "endurance",
    "fatLoss",
    "recomposition",
    "generalFitness",
]
PublicExperienceLevel = Literal["beginner", "intermediate", "advanced"]
PublicEquipmentCode = Literal[
    "BAND",
    "BARBELL",
    "BODYWEIGHT",
    "CABLE",
    "DUMBBELL",
    "KETTLEBELL",
    "LANDMINE",
    "MACHINE",
    "PLATE",
    "SMITH_MACHINE",
]
PublicSessionStructure = Literal["standard", "superset", "circuit", "density"]
PublicAdjunctOption = Literal[
    "mobility_warmup",
    "prehab_accessory",
    "conditioning_finisher",
    "endurance_support",
    "cooldown_stretch",
]


class PublicPainAreaEntry(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=False, extra="forbid")

    body_part: str

    @field_validator("body_part")
    @classmethod
    def validate_body_part(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("currentPainAreas.bodyPart must not contain surrounding whitespace")
        if value not in PUBLIC_PAIN_AREA_SLUGS:
            raise ValueError("currentPainAreas.bodyPart must use a locked public pain-area slug")
        return value


class PublicDomsEntry(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=False, extra="forbid")

    body_part: str
    level: int = Field(ge=1, le=3, strict=True)

    @field_validator("body_part")
    @classmethod
    def validate_body_part(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("currentDoms.bodyPart must not contain surrounding whitespace")
        if value not in PUBLIC_DOMS_BODY_PART_SLUGS:
            raise ValueError("currentDoms.bodyPart must use a locked muscle-soreness slug")
        return value

    @field_validator("level", mode="before")
    @classmethod
    def normalize_level(cls, value) -> int:
        return _normalize_json_schema_integer(value, field_name="currentDoms.level")


class PublicRoutineGenerateRequest(BaseModel):
    """Closed public request. Only camelCase product keys are accepted."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=False, extra="forbid")

    target_split_label: Optional[TargetSplitLabel] = None
    target_muscles: list[str] = Field(default_factory=list)
    time_available_min: int = Field(ge=10, le=180, strict=True)
    readiness_level: Literal["low", "normal", "high"] = "normal"
    current_pain_areas: list[PublicPainAreaEntry] = Field(default_factory=list)
    current_doms: list[PublicDomsEntry] = Field(default_factory=list)
    available_equipment: list[PublicEquipmentCode] = Field(default_factory=list)
    unavailable_equipment: list[PublicEquipmentCode] = Field(default_factory=list)
    goal: Optional[PublicGoal] = None
    experience_level: Optional[PublicExperienceLevel] = None
    session_structure: PublicSessionStructure = "standard"
    adjunct_options: list[PublicAdjunctOption] = Field(default_factory=list)
    profile_signals: list[str] = Field(default_factory=list)
    training_style: Literal["gym"] = "gym"
    user_note: Optional[str] = None

    @field_validator("time_available_min", mode="before")
    @classmethod
    def normalize_time_available_min(cls, value) -> int:
        return _normalize_json_schema_integer(value, field_name="timeAvailableMin")

    @field_validator("target_muscles")
    @classmethod
    def validate_target_muscles(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for raw_value in values:
            if raw_value != raw_value.strip():
                raise ValueError("targetMuscles must not contain surrounding whitespace")
            value = raw_value
            if value not in MUSCLE_SLUGS:
                raise ValueError("targetMuscles must use locked public trainable-muscle slugs")
            if value in normalized:
                raise ValueError("targetMuscles must not contain duplicate values")
            normalized.append(value)
        return normalized

    @field_validator("current_pain_areas")
    @classmethod
    def reject_duplicate_pain_areas(cls, values: list[PublicPainAreaEntry]) -> list[PublicPainAreaEntry]:
        body_parts = [entry.body_part for entry in values]
        if len(body_parts) != len(set(body_parts)):
            raise ValueError("currentPainAreas must contain each bodyPart at most once")
        return values

    @field_validator("current_doms")
    @classmethod
    def reject_duplicate_doms_areas(cls, values: list[PublicDomsEntry]) -> list[PublicDomsEntry]:
        body_parts = [entry.body_part for entry in values]
        if len(body_parts) != len(set(body_parts)):
            raise ValueError("currentDoms must contain each bodyPart at most once")
        return values

    @field_validator("available_equipment", "unavailable_equipment")
    @classmethod
    def reject_duplicate_equipment(cls, values: list[PublicEquipmentCode]) -> list[PublicEquipmentCode]:
        if len(values) != len(set(values)):
            raise ValueError("equipment lists must not contain duplicate values")
        return values

    @field_validator("adjunct_options")
    @classmethod
    def validate_adjunct_options(cls, values: list[PublicAdjunctOption]) -> list[PublicAdjunctOption]:
        if len(values) != len(set(values)):
            raise ValueError("adjunctOptions must not contain duplicate values")
        canonical = list(canonical_adjunct_options(values))
        conflict_code = adjunct_option_conflict_code(canonical)
        if conflict_code == "too_many_adjunct_options":
            raise ValueError(
                f"adjunctOptions accepts at most {MAX_DISTINCT_ADJUNCT_OPTIONS} distinct options"
            )
        if conflict_code == "conditioning_and_endurance_overlap":
            raise ValueError("conditioning_finisher and endurance_support cannot be combined")
        return canonical

    @field_validator("profile_signals", mode="before")
    @classmethod
    def validate_profile_signals(cls, values):
        if values is None:
            raise ValueError("profileSignals must be an array, not null")
        if isinstance(values, list) and all(isinstance(value, str) for value in values):
            if any(value != value.strip() for value in values):
                raise ValueError("profileSignals must not contain surrounding whitespace")
            if len(values) != len(set(values)):
                raise ValueError("profileSignals must not contain duplicate values")
        return list(resolve_profile_constraint_signals(values).selected_signals)

    @model_validator(mode="after")
    def validate_cross_field_rules(self):
        if not self.target_split_label and not self.target_muscles:
            raise ValueError("targetSplitLabel or targetMuscles is required")
        if self.target_split_label == "custom" and not self.target_muscles:
            raise ValueError("targetMuscles is required when targetSplitLabel is custom")
        overlap = set(self.available_equipment) & set(self.unavailable_equipment)
        if overlap:
            raise ValueError("equipment cannot be both available and unavailable")
        if self.readiness_level == "low" and self.session_structure in {"circuit", "density"}:
            raise ValueError("low readiness cannot use circuit or density")
        if self.readiness_level == "low" and {
            "conditioning_finisher",
            "endurance_support",
        }.intersection(self.adjunct_options):
            raise ValueError("low readiness cannot use conditioning or endurance adjuncts")
        if self.goal == "strength" and self.session_structure == "circuit":
            raise ValueError("strength goal cannot use circuit")
        return self

    def to_internal_request_payload(
        self,
        *,
        user_id: Optional[str] = None,
        stored_goal: Optional[str] = None,
        stored_experience_level: Optional[str] = None,
        stored_available_equipment: Iterable[str] = (),
    ) -> dict:
        """Map the public request to the internal AI request deterministically."""

        goal = self.goal or stored_goal or "hypertrophy"
        if goal not in PUBLIC_GOALS:
            raise ValueError("stored goal must use a locked public goal")
        experience_level = self.experience_level or stored_experience_level or "intermediate"
        if experience_level not in PUBLIC_EXPERIENCE_LEVELS:
            raise ValueError("stored experience level must use a locked public value")

        stored_equipment = [str(value).strip().upper() for value in stored_available_equipment]
        if any(value not in PUBLIC_EQUIPMENT_CODES for value in stored_equipment):
            raise ValueError("stored equipment must use locked public equipment codes")
        available_equipment = list(self.available_equipment) or list(dict.fromkeys(stored_equipment))
        overlap = set(available_equipment) & set(self.unavailable_equipment)
        if overlap:
            raise ValueError("resolved equipment cannot be both available and unavailable")
        if goal == "strength" and self.session_structure == "circuit":
            raise ValueError("resolved strength goal cannot use circuit")

        payload = {
            "targetSplitLabel": self.target_split_label,
            "targetMuscles": list(self.target_muscles),
            "readinessLevel": self.readiness_level,
            "timeAvailableMin": self.time_available_min,
            "painAreas": [
                {"bodyPart": entry.body_part}
                for entry in self.current_pain_areas
            ],
            "domsData": {
                entry.body_part: entry.level
                for entry in self.current_doms
            },
            "equipment": list(self.unavailable_equipment),
            "availableEquipment": available_equipment,
            "goal": goal,
            "experienceLevel": experience_level,
            "trainingStyle": "gym",
            "sessionStructure": self.session_structure,
            "adjunctOptions": list(self.adjunct_options),
            "profileSignals": list(self.profile_signals),
            "userNote": self.user_note,
        }
        if user_id is not None:
            payload["userId"] = user_id
        return payload

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from engines.supplement.query_understanding import EntityType


class InteractionType(str, Enum):
    ABSORPTION_INTERFERENCE = "absorption_interference"
    BLEEDING_RISK = "bleeding_risk"
    MEDICATION_TIMING_INTERFERENCE = "medication_timing_interference"
    DUPLICATE_OR_OVERLAP_RISK = "duplicate_or_overlap_risk"
    ORGAN_RISK = "organ_risk"
    UNKNOWN_OR_INSUFFICIENT_EVIDENCE = "unknown_or_insufficient_evidence"


class CautionLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("id", mode="after", check_fields=False)
    @classmethod
    def _id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("id must not be blank")
        return value


class TimingGuidance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    general: str = ""
    with_food: str = ""
    before_bed: str = ""
    exercise_related: str = ""


class KnowledgeEntityRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: EntityType
    canonical: str
    label: str

    @field_validator("canonical", "label", mode="after")
    @classmethod
    def _text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("entity fields must not be blank")
        return value


class IngredientProfile(_StrictModel):
    type: Literal["ingredient_profile"] = "ingredient_profile"
    id: str
    name: str
    aliases: list[str]
    category: str
    common_uses: list[str] = Field(default_factory=list)
    timing: TimingGuidance = Field(default_factory=TimingGuidance)
    spacing: list[str] = Field(default_factory=list)
    cautions: list[str] = Field(default_factory=list)
    high_risk_groups: list[str] = Field(default_factory=list)
    related_interactions: list[str] = Field(default_factory=list)

    @field_validator("name", "category", mode="after")
    @classmethod
    def _required_text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("required text fields must not be blank")
        return value

    @field_validator("aliases", mode="after")
    @classmethod
    def _aliases_must_not_be_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("aliases must not be empty")
        if any(not alias.strip() for alias in value):
            raise ValueError("aliases must not contain blank values")
        return value


class InteractionRule(_StrictModel):
    type: Literal["interaction_rule"] = "interaction_rule"
    id: str
    entity_a: KnowledgeEntityRef
    entity_b: KnowledgeEntityRef
    interaction_type: InteractionType
    mechanism: str = ""
    recommendation: str = ""
    spacing_guidance: str = ""
    caution_level: CautionLevel
    consultation_required_when: list[str] = Field(default_factory=list)


class SafetyRule(_StrictModel):
    type: Literal["safety_rule"] = "safety_rule"
    id: str
    trigger_conditions: list[str]
    affected_entities: list[str]
    risk: str
    recommendation: str
    caution_level: CautionLevel
    when_to_consult: str = ""
    evidence_summary: str = ""
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("trigger_conditions", "affected_entities", mode="after")
    @classmethod
    def _lists_must_not_be_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("rule lists must not be empty")
        if any(not item.strip() for item in value):
            raise ValueError("rule lists must not contain blank values")
        return value

    @field_validator("risk", "recommendation", mode="after")
    @classmethod
    def _required_rule_text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("required rule text must not be blank")
        return value

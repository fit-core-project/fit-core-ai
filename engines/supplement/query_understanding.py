from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EntityType(str, Enum):
    SUPPLEMENT_INGREDIENT = "supplement_ingredient"
    DRUG_OR_DRUG_CLASS = "drug_or_drug_class"
    FOOD_OR_COMPOUND = "food_or_compound"
    CONDITION = "condition"
    RISK_CONTEXT = "risk_context"


class IntentType(str, Enum):
    TIMING = "timing"
    SUPPLEMENT_INTERACTION = "supplement_interaction"
    DRUG_INTERACTION = "drug_interaction"
    FOOD_COMPOUND_INTERACTION = "food_compound_interaction"
    CONDITION_SAFETY = "condition_safety"
    PREGNANCY_OR_HIGH_DOSE_SAFETY = "pregnancy_or_high_dose_safety"
    MEDICATION_SAFETY = "medication_safety"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class AliasEntry:
    entity_type: EntityType
    canonical: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class ParsedEntity:
    type: EntityType
    canonical: str
    text: str


@dataclass(frozen=True)
class ParsedSupplementQuery:
    raw_query: str
    entities: tuple[ParsedEntity, ...]
    intents: tuple[IntentType, ...]


ALIAS_REGISTRY: tuple[AliasEntry, ...] = (
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "magnesium", ("마그네슘", "magnesium", "Mg")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "iron", ("철분", "iron", "Fe")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "vitamin d", ("비타민D", "비타민 D", "vitamin d")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "omega3", ("오메가3", "오메가-3", "omega3", "omega-3", "fish oil")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "creatine", ("크레아틴", "creatine")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "zinc", ("아연", "zinc", "Zn")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "calcium", ("칼슘", "calcium", "Ca")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "probiotics", ("유산균", "probiotics")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "protein", ("프로틴", "단백질 보충제", "whey protein")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "caffeine", ("카페인", "caffeine")),
    AliasEntry(EntityType.DRUG_OR_DRUG_CLASS, "antibiotics", ("항생제", "antibiotics")),
    AliasEntry(
        EntityType.DRUG_OR_DRUG_CLASS,
        "thyroid medication",
        ("갑상선약", "thyroid medication", "levothyroxine"),
    ),
    AliasEntry(EntityType.DRUG_OR_DRUG_CLASS, "anticoagulant", ("항응고제", "anticoagulant")),
    AliasEntry(EntityType.DRUG_OR_DRUG_CLASS, "warfarin", ("와파린", "warfarin")),
    AliasEntry(EntityType.DRUG_OR_DRUG_CLASS, "acetaminophen", ("타이레놀", "acetaminophen", "paracetamol")),
    AliasEntry(EntityType.DRUG_OR_DRUG_CLASS, "pain reliever", ("진통제", "pain reliever")),
    AliasEntry(EntityType.DRUG_OR_DRUG_CLASS, "blood pressure medication", ("혈압약", "blood pressure medication")),
    AliasEntry(EntityType.DRUG_OR_DRUG_CLASS, "diabetes medication", ("당뇨약", "diabetes medication")),
    AliasEntry(EntityType.FOOD_OR_COMPOUND, "coffee", ("커피", "coffee")),
    AliasEntry(EntityType.FOOD_OR_COMPOUND, "caffeine", ("카페인", "caffeine")),
    AliasEntry(EntityType.FOOD_OR_COMPOUND, "calcium", ("칼슘", "calcium")),
    AliasEntry(EntityType.FOOD_OR_COMPOUND, "alcohol", ("술", "알코올", "alcohol")),
    AliasEntry(EntityType.CONDITION, "kidney disease", ("신장질환", "kidney disease")),
    AliasEntry(EntityType.CONDITION, "liver disease", ("간질환", "liver disease")),
    AliasEntry(EntityType.CONDITION, "insomnia", ("불면", "insomnia")),
    AliasEntry(EntityType.CONDITION, "hypertension", ("고혈압", "hypertension")),
    AliasEntry(EntityType.RISK_CONTEXT, "pregnancy", ("임신", "pregnancy")),
    AliasEntry(EntityType.RISK_CONTEXT, "breastfeeding", ("수유", "breastfeeding")),
    AliasEntry(EntityType.RISK_CONTEXT, "high dose", ("고용량", "high dose")),
    AliasEntry(EntityType.RISK_CONTEXT, "surgery", ("수술", "surgery")),
    AliasEntry(EntityType.RISK_CONTEXT, "alcohol use", ("음주", "술", "알코올", "alcohol use")),
)

_TIMING_NEEDLES = ("언제", "먹는 시간", "복용 시간", "식후", "공복", "자기 전", "운동 전후", "운동 전", "운동 후")
_INTERACTION_NEEDLES = ("같이", "함께", "병용", "먹어도", "복용해도")


def parse_supplement_query(raw_query: str | None) -> ParsedSupplementQuery:
    query = (raw_query or "").strip()
    entities = _extract_entities(query)
    intents = _infer_intents(query, entities)
    return ParsedSupplementQuery(raw_query=query, entities=tuple(entities), intents=tuple(intents))


def _extract_entities(query: str) -> list[ParsedEntity]:
    lowered = query.lower()
    found: list[ParsedEntity] = []
    seen: set[tuple[EntityType, str]] = set()

    alias_candidates = []
    for entry in ALIAS_REGISTRY:
        for alias in entry.aliases:
            alias_candidates.append((entry, alias))
    alias_candidates.sort(key=lambda item: len(item[1]), reverse=True)

    for entry, alias in alias_candidates:
        haystack = lowered if _is_ascii(alias) else query
        needle = alias.lower() if _is_ascii(alias) else alias
        if needle not in haystack:
            continue
        key = (entry.entity_type, entry.canonical)
        if key in seen:
            continue
        found.append(ParsedEntity(type=entry.entity_type, canonical=entry.canonical, text=alias))
        seen.add(key)

    return found


def _infer_intents(query: str, entities: list[ParsedEntity]) -> list[IntentType]:
    intents: list[IntentType] = []
    supplement_entities = _entities_of_type(entities, EntityType.SUPPLEMENT_INGREDIENT)
    drug_entities = _entities_of_type(entities, EntityType.DRUG_OR_DRUG_CLASS)
    food_entities = _entities_of_type(entities, EntityType.FOOD_OR_COMPOUND)
    condition_entities = _entities_of_type(entities, EntityType.CONDITION)
    risk_entities = _entities_of_type(entities, EntityType.RISK_CONTEXT)
    canonical_set = {entity.canonical for entity in entities}

    if "acetaminophen" in canonical_set and (
        "liver disease" in canonical_set or "alcohol use" in canonical_set or "high dose" in canonical_set
    ):
        intents.append(IntentType.MEDICATION_SAFETY)

    if supplement_entities and any(needle in query for needle in _TIMING_NEEDLES):
        intents.append(IntentType.TIMING)

    if supplement_entities and drug_entities:
        intents.append(IntentType.DRUG_INTERACTION)

    if supplement_entities and food_entities:
        intents.append(IntentType.FOOD_COMPOUND_INTERACTION)

    if len(supplement_entities) >= 2 and any(needle in query for needle in _INTERACTION_NEEDLES):
        intents.append(IntentType.SUPPLEMENT_INTERACTION)

    if supplement_entities and condition_entities:
        intents.append(IntentType.CONDITION_SAFETY)

    if supplement_entities and any(entity.canonical in {"pregnancy", "high dose", "surgery", "alcohol use"} for entity in risk_entities):
        intents.append(IntentType.PREGNANCY_OR_HIGH_DOSE_SAFETY)

    return _dedupe_intents(intents) or [IntentType.UNKNOWN]


def _entities_of_type(entities: list[ParsedEntity], entity_type: EntityType) -> list[ParsedEntity]:
    return [entity for entity in entities if entity.type == entity_type]


def _dedupe_intents(intents: list[IntentType]) -> list[IntentType]:
    return list(dict.fromkeys(intents))


def _is_ascii(value: str) -> bool:
    return all(ord(ch) < 128 for ch in value)

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
    SIDE_EFFECT = "side_effect"
    LONG_TERM_USE = "long_term_use"
    GOAL_BASED_RECOMMENDATION = "goal_based_recommendation"
    SYMPTOM_AFTER_INTAKE = "symptom_after_intake"
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
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "multivitamin", ("종합비타민", "멀티비타민", "multivitamin", "multi vitamin")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "silymarin", ("실리마린", "밀크씨슬", "밀크시슬", "silymarin", "milk thistle")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "probiotics", ("유산균", "프로바이오틱스", "probiotic", "probiotics")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "zinc", ("아연", "zinc", "Zn")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "calcium", ("칼슘", "calcium", "Ca")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "protein", ("프로틴", "단백질 보충제", "웨이", "whey", "whey protein", "protein powder")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "caffeine", ("카페인", "caffeine")),
    AliasEntry(EntityType.DRUG_OR_DRUG_CLASS, "antibiotics", ("항생제", "antibiotics")),
    AliasEntry(EntityType.DRUG_OR_DRUG_CLASS, "medication", ("처방약", "복용 중인 약", "약물", "medication", "medicine")),
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
    AliasEntry(EntityType.CONDITION, "liver disease", ("간질환", "간수치", "간 기능", "간 기능 저하", "liver disease")),
    AliasEntry(EntityType.CONDITION, "insomnia", ("불면", "insomnia")),
    AliasEntry(EntityType.CONDITION, "hypertension", ("고혈압", "hypertension")),
    AliasEntry(EntityType.CONDITION, "kidney disease", ("신장 기능 저하", "renal impairment")),
    AliasEntry(EntityType.RISK_CONTEXT, "pregnancy", ("임신", "임산부", "pregnancy")),
    AliasEntry(EntityType.RISK_CONTEXT, "breastfeeding", ("수유", "breastfeeding")),
    AliasEntry(EntityType.RISK_CONTEXT, "high dose", ("고용량", "high dose")),
    AliasEntry(EntityType.RISK_CONTEXT, "long-term use", ("장기복용", "장기 복용", "long-term use")),
    AliasEntry(EntityType.RISK_CONTEXT, "surgery", ("수술", "surgery")),
    AliasEntry(EntityType.RISK_CONTEXT, "alcohol use", ("음주", "술", "알코올", "alcohol use")),
    AliasEntry(EntityType.RISK_CONTEXT, "immunocompromised", ("면역저하", "면역 저하", "항암", "중증질환", "중심정맥관")),
    AliasEntry(EntityType.RISK_CONTEXT, "digestive symptom", ("속이 안 좋음", "속이 안 좋은", "메스꺼움", "설사", "복통", "위장 불편")),
)

_TIMING_NEEDLES = ("언제", "먹는 시간", "복용 시간", "식후", "공복", "자기 전", "운동 전후", "운동 전", "운동 후")
_INTERACTION_NEEDLES = ("같이", "함께", "병용", "먹어도", "복용해도", "한번에", "동시에")
_SIDE_EFFECT_NEEDLES = ("부작용", "속이 안", "메스꺼움", "설사", "복통", "위장 불편", "불편")
_LONG_TERM_NEEDLES = ("장기복용", "장기 복용", "오래 먹", "계속 먹", "매일 먹")
_GOAL_NEEDLES = ("추천", "뭐 먹", "무엇을 먹", "좋은 거", "먹을 만한", "도움")
_SYMPTOM_AFTER_INTAKE_NEEDLES = ("먹고 속", "복용 후", "섭취 후", "먹었는데", "계속 먹어도")


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

    if any(needle in query for needle in _SYMPTOM_AFTER_INTAKE_NEEDLES):
        intents.append(IntentType.SYMPTOM_AFTER_INTAKE)

    if any(needle in query for needle in _SIDE_EFFECT_NEEDLES):
        intents.append(IntentType.SIDE_EFFECT)

    if any(needle in query for needle in _LONG_TERM_NEEDLES) or "long-term use" in canonical_set:
        intents.append(IntentType.LONG_TERM_USE)

    if any(needle in query for needle in _GOAL_NEEDLES):
        intents.append(IntentType.GOAL_BASED_RECOMMENDATION)

    if supplement_entities and drug_entities:
        intents.append(IntentType.DRUG_INTERACTION)

    if supplement_entities and food_entities:
        intents.append(IntentType.FOOD_COMPOUND_INTERACTION)

    if len(supplement_entities) >= 2 and any(needle in query for needle in _INTERACTION_NEEDLES):
        intents.append(IntentType.SUPPLEMENT_INTERACTION)

    if supplement_entities and condition_entities:
        intents.append(IntentType.CONDITION_SAFETY)

    if supplement_entities and any(entity.canonical in {"pregnancy", "high dose", "surgery", "alcohol use", "long-term use", "immunocompromised"} for entity in risk_entities):
        intents.append(IntentType.PREGNANCY_OR_HIGH_DOSE_SAFETY)

    return _dedupe_intents(intents) or [IntentType.UNKNOWN]


def _entities_of_type(entities: list[ParsedEntity], entity_type: EntityType) -> list[ParsedEntity]:
    return [entity for entity in entities if entity.type == entity_type]


def _dedupe_intents(intents: list[IntentType]) -> list[IntentType]:
    return list(dict.fromkeys(intents))


def _is_ascii(value: str) -> bool:
    return all(ord(ch) < 128 for ch in value)

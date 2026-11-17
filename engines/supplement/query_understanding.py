from __future__ import annotations

import difflib
import re
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
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "magnesium", ("마그네슘", "마그내슘", "마그네숨", "칼마디", "칼마", "magnesium", "Mg")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "iron", ("철분", "iron", "Fe")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "vitamin a", ("비타민A", "비타민 A", "vitamin a")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "vitamin c", ("비타민C", "비타민 C", "비타민씨", "비타민 씨", "아스코르브산", "아스코빅산", "vitamin c", "ascorbic acid")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "vitamin d", ("비타민D", "비타민 D", "비타민디", "비타민 디", "칼마디", "vitamin d", "d3", "vitamin d3")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "vitamin e", ("비타민E", "비타민 E", "vitamin e")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "omega3", ("오메가3", "오메가-3", "오메가 3", "오메가쓰리", "오메가 쓰리", "오매가3", "오매가", "오메가", "피쉬오일", "피시오일", "omega3", "omega-3", "fish oil")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "creatine", ("크레아틴", "creatine")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "multivitamin", ("종합비타민", "종합 비타민", "멀티비타민", "멀티 비타민", "multivitamin", "multi vitamin")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "silymarin", ("실리마린", "밀크씨슬", "밀크시슬", "silymarin", "milk thistle")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "probiotics", ("유산균", "프로바이오틱스", "프로바이오틱", "probiotic", "probiotics")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "zinc", ("아연", "zinc", "Zn")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "calcium", ("칼슘", "calcium", "Ca", "탄산칼슘", "구연산칼슘", "칼마디", "칼마")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "protein", ("프로틴", "프로테인", "단백질 보충제", "단백질 쉐이크", "단백질 파우더", "웨이", "whey", "whey protein", "protein powder")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "caffeine", ("카페인", "caffeine", "에너지드링크", "에너지 드링크", "핫식스", "레드불", "몬스터 에너지")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "melatonin", ("멜라토닌", "melatonin")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "vitamin b", ("비타민B", "비타민 B", "비타민비", "vitamin b", "b6", "b12", "비타민B6", "비타민B12")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "folate", ("엽산", "폴산", "folate", "folic acid")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "biotin", ("비오틴", "biotin")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "lutein", ("루테인", "마리골드꽃추출물", "lutein")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "coenzyme q10", ("코엔자임Q10", "코엔자임 Q10", "코엔자임큐텐", "코큐텐", "coenzyme q10", "coq10", "유비퀴놀", "유비퀴논", "ubiquinol", "ubiquinone")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "collagen", ("콜라겐", "collagen")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "arginine", ("아르기닌", "아르지닌", "arginine")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "bcaa", ("분지쇄아미노산", "비시에이에이", "비씨에이에이", "bcaa")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "glutamine", ("글루타민", "glutamine")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "selenium", ("셀레늄", "셀렌", "selenium")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "theanine", ("테아닌", "theanine")),
    AliasEntry(EntityType.SUPPLEMENT_INGREDIENT, "ashwagandha", ("아쉬와간다", "아슈와간다", "ashwagandha", "withania", "withania somnifera")),
    AliasEntry(EntityType.DRUG_OR_DRUG_CLASS, "antibiotics", ("항생제", "antibiotics")),
    AliasEntry(EntityType.DRUG_OR_DRUG_CLASS, "medication", ("처방약", "복용 중인 약", "약물", "medication", "medicine")),
    AliasEntry(
        EntityType.DRUG_OR_DRUG_CLASS,
        "thyroid medication",
        ("갑상선약", "thyroid medication", "levothyroxine"),
    ),
    AliasEntry(EntityType.DRUG_OR_DRUG_CLASS, "thyroid medication", ("갑상선 호르몬제", "갑상선호르몬제", "씬지로이드", "신지로이드", "synthroid")),
    AliasEntry(EntityType.DRUG_OR_DRUG_CLASS, "anticoagulant", ("플라빅스", "클로피도그렐", "항혈소판제", "엘리퀴스", "자렐토", "프라닥사", "eliquis", "xarelto")),
    AliasEntry(EntityType.DRUG_OR_DRUG_CLASS, "anticoagulant", ("항응고제", "혈액희석제", "anticoagulant", "blood thinner", "blood thinners")),
    AliasEntry(EntityType.DRUG_OR_DRUG_CLASS, "warfarin", ("와파린", "warfarin")),
    AliasEntry(EntityType.DRUG_OR_DRUG_CLASS, "aspirin", ("아스피린", "aspirin")),
    AliasEntry(EntityType.DRUG_OR_DRUG_CLASS, "acetaminophen", ("타이레놀", "아세트아미노펜", "acetaminophen", "paracetamol")),
    AliasEntry(EntityType.DRUG_OR_DRUG_CLASS, "pain reliever", ("진통제", "pain reliever")),
    AliasEntry(EntityType.DRUG_OR_DRUG_CLASS, "medication", ("감기약", "종합감기약", "cold medicine")),
    AliasEntry(EntityType.DRUG_OR_DRUG_CLASS, "blood pressure medication", ("혈압약", "blood pressure medication")),
    AliasEntry(EntityType.DRUG_OR_DRUG_CLASS, "diabetes medication", ("당뇨약", "diabetes medication")),
    AliasEntry(EntityType.FOOD_OR_COMPOUND, "coffee", ("커피", "coffee", "아아", "아메리카노", "카페라떼", "라떼")),
    AliasEntry(EntityType.FOOD_OR_COMPOUND, "caffeine", ("카페인", "caffeine")),
    AliasEntry(EntityType.FOOD_OR_COMPOUND, "calcium", ("칼슘", "calcium")),
    AliasEntry(EntityType.FOOD_OR_COMPOUND, "alcohol", ("술", "알코올", "alcohol")),
    AliasEntry(EntityType.CONDITION, "kidney disease", ("신장질환", "신부전", "투석", "신장 기능이 떨어", "신장기능이 떨어", "신장이 안 좋", "신장 안 좋", "콩팥", "kidney disease", "dialysis")),
    AliasEntry(EntityType.CONDITION, "liver disease", ("간질환", "간수치", "간 수치", "간 기능", "간 기능 저하", "간경화", "간경변", "지방간", "간염", "보균자", "liver disease")),
    AliasEntry(EntityType.CONDITION, "kidney disease", ("크레아티닌", "신장 수치", "신장수치", "단백뇨", "사구체")),
    AliasEntry(EntityType.CONDITION, "insomnia", ("불면", "insomnia")),
    AliasEntry(EntityType.CONDITION, "hypertension", ("고혈압", "hypertension")),
    AliasEntry(EntityType.CONDITION, "kidney disease", ("신장 기능 저하", "renal impairment")),
    AliasEntry(EntityType.RISK_CONTEXT, "pregnancy", ("임신", "임산부", "임테기", "아기 갖", "아이 갖", "아기 계획", "임신 준비", "임신 계획", "임신테스트", "입덧", "태아", "산모", "예비맘", "pregnancy", "pregnant")),
    AliasEntry(EntityType.RISK_CONTEXT, "breastfeeding", ("수유", "모유", "젖 먹이", "breastfeeding")),
    AliasEntry(EntityType.RISK_CONTEXT, "high dose", ("고용량", "고함량", "메가도스", "megadose", "high dose")),
    AliasEntry(EntityType.RISK_CONTEXT, "long-term use", ("장기복용", "장기 복용", "몇 달째", "몇달째", "개월째", "몇 년째", "몇년째", "long-term use")),
    AliasEntry(EntityType.RISK_CONTEXT, "surgery", ("수술", "surgery")),
    AliasEntry(EntityType.RISK_CONTEXT, "alcohol use", ("음주", "술", "알코올", "소주", "맥주", "와인", "alcohol use")),
    AliasEntry(EntityType.RISK_CONTEXT, "immunocompromised", ("면역저하", "면역 저하", "면역억제", "중환자실", "항암", "장기이식", "신장이식", "간이식", "이식 받", "이식받", "중증질환", "중심정맥관")),
    # the child is the one taking it ("아기한테", "고등학생 아들", "먹여도"); "아기 낳고 모유 먹이는데" is
    # about the mother (breastfeeding), so a bare 아기 is not enough
    AliasEntry(EntityType.RISK_CONTEXT, "child", (
        "아기한테", "아기에게", "애기한테", "애기에게", "아이한테", "아이에게", "영유아", "어린이", "초등학생",
        "중학생", "고등학생", "청소년", "소아", "돌 지난", "돌지난", "먹여도", "먹여요", "먹여볼", "먹일",
        "중1", "중2", "중3", "고1", "고2", "고3", "초등", "child", "kid", "teen")),
    AliasEntry(EntityType.RISK_CONTEXT, "digestive symptom", ("속이 안 좋음", "속이 안 좋은", "메스꺼움", "설사", "복통", "위장 불편")),
)

_TIMING_NEEDLES = (
    "언제", "먹는 시간", "복용 시간", "식후", "식전", "공복", "빈속", "자기 전", "자기전", "취침 전",
    "운동 전후", "운동 전", "운동 후", "운동 직후", "직후", "아침에", "저녁에", "밤에", "밥 먹고", "식사",
    "언제부터", "나서 먹", "늦나", "늦지", "타이밍", "아무 때나", "아무때나", "시간 상관", "몇 시", "before bed", "at night", "in the morning", "empty stomach", "with food", "after meal", "before meal", "when to take",
)
_INTERACTION_NEEDLES = ("같이", "함께", "병용", "먹어도", "복용해도", "한번에", "동시에", "둘 다", "둘다",
                        "궁합", "하나 더", "추가", "거기다", "들어있", "들어 있", "겹치", "가치", "따로", "너무 많", "너무 마니", "넘 마")
_SIDE_EFFECT_NEEDLES = (
    "부작용", "속이 안", "메스꺼", "속쓰", "설사", "복통", "위장 불편", "불편", "비린", "트림",
    "더부룩", "가스", "변비", "두드러기", "부글", "묽어", "묽게", "빵빵", "부풀", "울렁", "냄새",
    "여드름", "뾰루지", "가려", "붓", "구토", "화장실도 잘 못", "화장실을 못", "배가 아", "어지러", "두근",
    "꾸륵", "꾸르륵", "탈모", "배탈", "gas", "diarrhea", "bloat", "설사할까", "설사 할까",
)
_LONG_TERM_NEEDLES = ("장기복용", "장기 복용", "오래 먹", "계속 먹", "매일 먹")
_GOAL_NEEDLES = ("추천", "뭐 먹", "무엇을 먹", "좋은 거", "먹을 만한", "도움")
_SYMPTOM_AFTER_INTAKE_NEEDLES = ("먹고 속", "복용 후", "섭취 후", "먹었는데", "먹고 나서", "먹으면 속")


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
        if not _alias_matches(alias, query, lowered):
            continue
        key = (entry.entity_type, entry.canonical)
        if key in seen:
            continue
        found.append(ParsedEntity(type=entry.entity_type, canonical=entry.canonical, text=alias))
        seen.add(key)

    found_keys = {(e.type, e.canonical) for e in found}
    exact_aliases = {alias for _, alias in alias_candidates if not _is_ascii(alias)}
    for word in re.findall(r"[가-힣]{2,}", query):
        if any(word.startswith(a) for a in exact_aliases if len(a) >= 2):
            continue  # "비타민디" is vitamin D, not a typo of 비타민비
        for entry, alias in alias_candidates:
            if entry.entity_type != EntityType.SUPPLEMENT_INGREDIENT or _is_ascii(alias) or len(alias) < 2 or " " in alias:
                continue
            if (entry.entity_type, entry.canonical) in found_keys:
                continue
            stem = word[: len(alias)]
            if stem != alias and len(stem) == len(alias) and _jamo_distance(stem, alias) == 1 and stem[0] == alias[0]:
                found.append(ParsedEntity(type=entry.entity_type, canonical=entry.canonical, text=stem))
                found_keys.add((entry.entity_type, entry.canonical))
    for word in re.findall(r"[a-z][a-z0-9]{5,}", lowered):
        best = None
        for entry, alias in alias_candidates:
            if not _is_ascii(alias) or len(alias) < 6 or " " in alias:
                continue
            ratio = difflib.SequenceMatcher(None, word, alias.lower()).ratio()
            if ratio >= 0.85 and (best is None or ratio > best[0]):
                best = (ratio, entry)
        if best and (best[1].entity_type, best[1].canonical) not in found_keys:
            found.append(ParsedEntity(type=best[1].entity_type, canonical=best[1].canonical, text=word))
            found_keys.add((best[1].entity_type, best[1].canonical))
    supplements = [e for e in found if e.type == EntityType.SUPPLEMENT_INGREDIENT]
    # "프로틴 말고 콜라겐만": the negated supplement is not taken
    negated = {e.text for e in supplements if re.search(re.escape(e.text) + r"\s*(?:말고|대신|안\s*먹)", query)}
    if negated and len(negated) < len(supplements):
        found = [e for e in found if e.text not in negated]
    # any other prescription / medicine mention ("고지혈증약", "녹내장 안약", "이소트레티노인 먹는 중"):
    # a medication the pairing must be checked against
    if not any(e.type == EntityType.DRUG_OR_DRUG_CLASS for e in found):
        m = _GENERIC_DRUG_RE.search(query)
        if m:
            found.append(ParsedEntity(type=EntityType.DRUG_OR_DRUG_CLASS, canonical="medication", text=m.group(0).strip()))
    # "비타민D 먹다가 칼슘 수치 높대요": a lab value, not a second supplement being taken
    lab_only = [e for e in found if re.search(re.escape(e.text) + r"\s*수치", query)
                and query.count(e.text) == len(re.findall(re.escape(e.text) + r"\s*수치", query))]
    lab_texts = {e.text for e in lab_only}
    others = [e for e in found if e.type == EntityType.SUPPLEMENT_INGREDIENT and e.text not in lab_texts]
    if lab_only and others:
        found = [e for e in found if e.text not in lab_texts]
    return found


_GENERIC_DRUG_RE = re.compile(
    r"[가-힣A-Za-z]{1,10}\s?(?:약|안약|약물|호르몬제|억제제|치료제|진통제|소염제)(?=\s|을|를|이|가|은|는|도|만|이랑|랑|하고|과|와|,|$|먹|복용|처방|드시|넣)"
    r"|스타틴|아토르바스타틴|로수바스타틴|이소트레티노인|메트포르민|처방\s?받"
)
_UNKNOWN_PRODUCT_RE = re.compile(r"콘드로이친|홍국|글루코사민|홍삼|세인트존스워트|은행잎|쏘팔메토|프로폴리스|빌베리|여주|추출물|즙")
# "마그네슘 밤?", "글루타민 타이밍?": a one-line timing question
_SHORT_TIMING_RE = re.compile(r"(밤|아침|저녁|점심|공복|식후|식전|자기\s*전)\s*\??\s*$")


_LLM_PARSE_PROMPT = """You map a Korean supplement question to a fixed vocabulary. Return JSON only:
{{"entities": [...], "intents": [...]}}
Use ONLY these entity values (type:canonical):
{entities}
Use ONLY these intents: {intents}
Rules: list a supplement only if the user takes / asks about it (not "X 말고"); a drug or condition only if
the user has it; a context (pregnancy, breastfeeding, child, high dose, long-term use, surgery,
immunocompromised, alcohol use) only if clearly stated or clearly implied (임테기 두 줄 = pregnancy).
Intents: timing = when/with food; supplement_interaction = two supplements together or overlapping;
drug_interaction = a supplement with a drug; food_compound_interaction = with coffee/alcohol/food;
condition_safety = with kidney/liver/other disease; pregnancy_or_high_dose_safety = pregnancy, high dose,
surgery, chemo/immunocompromised; side_effect = a symptom after taking it; long_term_use = taking it for
a long time; medication_safety = a drug's own safety; goal_based_recommendation = what to take for a goal.
If nothing applies return empty lists.
Question: {question}"""


def llm_refine_parsed_query(parsed: ParsedSupplementQuery, invoke) -> ParsedSupplementQuery:
    """Fallback for wording the rules do not know: ask the LLM to map the question onto the KB
    vocabulary and MERGE its answer (rule hits are kept). `invoke(prompt) -> str`. Only values from
    the registry / intent enum are accepted, so the LLM cannot introduce unknown entities."""
    import json as _json
    vocab: dict[str, EntityType] = {}
    for entry in ALIAS_REGISTRY:
        vocab.setdefault(entry.canonical, entry.entity_type)
    allowed_intents = [i.value for i in IntentType if i != IntentType.UNKNOWN]
    prompt = _LLM_PARSE_PROMPT.format(
        entities=", ".join(f"{t.value}:{c}" for c, t in sorted(vocab.items(), key=lambda kv: kv[1].value)),
        intents=", ".join(allowed_intents), question=parsed.raw_query)
    try:
        raw = invoke(prompt)
        text = getattr(raw, "content", raw)
        m = re.search(r"\{.*\}", str(text), re.S)
        data = _json.loads(m.group(0)) if m else {}
    except Exception:
        return parsed
    entities = list(parsed.entities)
    have = {e.canonical for e in entities}
    rule_found_supplement = any(e.type == EntityType.SUPPLEMENT_INGREDIENT for e in entities)
    for value in data.get("entities") or []:
        canonical = str(value).split(":")[-1].strip().lower()
        if canonical in vocab and canonical not in have and not (
                rule_found_supplement and vocab[canonical] == EntityType.SUPPLEMENT_INGREDIENT):
            # rules already named the supplement: the LLM may add context, not more supplements
            entities.append(ParsedEntity(type=vocab[canonical], canonical=canonical, text=canonical))
            have.add(canonical)
    intents = [i for i in parsed.intents if i != IntentType.UNKNOWN]
    for value in data.get("intents") or []:
        try:
            intent = IntentType(str(value).strip().lower())
        except ValueError:
            continue
        if intent == IntentType.GOAL_BASED_RECOMMENDATION and any(
                e.type == EntityType.SUPPLEMENT_INGREDIENT for e in entities):
            continue  # "마그네슘 수면에 도움 돼요?" asks about a named supplement's effect, not for a recommendation
        if intent != IntentType.UNKNOWN and intent not in intents:
            intents.append(intent)
    return ParsedSupplementQuery(raw_query=parsed.raw_query, entities=tuple(entities),
                                 intents=tuple(intents or [IntentType.UNKNOWN]))


def needs_llm_parse(parsed: ParsedSupplementQuery) -> bool:
    has_supplement = any(e.type == EntityType.SUPPLEMENT_INGREDIENT for e in parsed.entities)
    return not has_supplement or list(parsed.intents) == [IntentType.UNKNOWN]


def _infer_intents(query: str, entities: list[ParsedEntity]) -> list[IntentType]:
    intents: list[IntentType] = []
    supplement_entities = _entities_of_type(entities, EntityType.SUPPLEMENT_INGREDIENT)
    drug_entities = _entities_of_type(entities, EntityType.DRUG_OR_DRUG_CLASS)
    food_entities = _entities_of_type(entities, EntityType.FOOD_OR_COMPOUND)
    condition_entities = _entities_of_type(entities, EntityType.CONDITION)
    risk_entities = _entities_of_type(entities, EntityType.RISK_CONTEXT)
    canonical_set = {entity.canonical for entity in entities}

    other_drugs = [e for e in drug_entities if e.canonical != "acetaminophen"]
    if "acetaminophen" in canonical_set and (
        "liver disease" in canonical_set or "alcohol use" in canonical_set or "high dose" in canonical_set
        or other_drugs  # 감기약 + 타이레놀: duplicate acetaminophen
    ):
        intents.append(IntentType.MEDICATION_SAFETY)

    if supplement_entities and (any(needle in query for needle in _TIMING_NEEDLES) or _SHORT_TIMING_RE.search(query)):
        intents.append(IntentType.TIMING)

    if any(needle in query for needle in _SYMPTOM_AFTER_INTAKE_NEEDLES) and (
        any(needle in query for needle in _SIDE_EFFECT_NEEDLES)
        or supplement_entities
        or any(e.canonical == "digestive symptom" for e in risk_entities)
    ):
        intents.append(IntentType.SYMPTOM_AFTER_INTAKE)
        if supplement_entities and any(needle in query for needle in _SIDE_EFFECT_NEEDLES):
            intents.append(IntentType.SIDE_EFFECT)

    if any(needle in query.lower() for needle in _SIDE_EFFECT_NEEDLES):
        intents.append(IntentType.SIDE_EFFECT)

    if any(needle in query for needle in _LONG_TERM_NEEDLES) or "long-term use" in canonical_set:
        intents.append(IntentType.LONG_TERM_USE)

    goal_hits = [needle for needle in _GOAL_NEEDLES if needle in query]
    # "마그네슘 수면에 도움 돼요?" asks whether a named supplement works (efficacy), not for a
    # recommendation; only "도움" with a named supplement is not a goal question.
    if goal_hits and not (supplement_entities and goal_hits == ["도움"]):
        intents.append(IntentType.GOAL_BASED_RECOMMENDATION)

    if drug_entities and (supplement_entities or _UNKNOWN_PRODUCT_RE.search(query)
                          or any(n in query for n in _INTERACTION_NEEDLES)):
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


# Short element symbols are matched case-sensitively so "500mg" / "30 mg" never become magnesium
# and "Caffeine" / "coffee" never become calcium / iron.
_CASE_SENSITIVE_ALIASES = {"Mg", "Fe", "Ca", "Zn"}
# Hangul syllables whose preceding syllable changes the word, e.g. "수술", "기술", "시술", "마술".
_HANGUL_PREFIX_GUARDED_ALIASES = {"술"}
# Korean aliases that are prefixes of unrelated words, e.g. "웨이" in "웨이트".
_KOREAN_SUFFIX_GUARDS = {"웨이": "트"}


def _alias_matches(alias: str, query: str, lowered: str) -> bool:
    if _is_ascii(alias):
        if alias in _CASE_SENSITIVE_ALIASES:
            haystack, needle = query, alias
        else:
            haystack, needle = lowered, alias.lower()
        # Digits may follow ("vitamin d3") but not precede ("500mg") an alias.
        pattern = rf"(?<![A-Za-z0-9]){re.escape(needle)}(?![A-Za-z])"
        return re.search(pattern, haystack) is not None

    if alias not in query:
        return False
    for match in re.finditer(re.escape(alias), query):
        start, end = match.start(), match.end()
        if alias in _HANGUL_PREFIX_GUARDED_ALIASES and start > 0 and _is_hangul_syllable(query[start - 1]):
            continue
        suffix_guard = _KOREAN_SUFFIX_GUARDS.get(alias)
        if suffix_guard and query.startswith(suffix_guard, end):
            continue
        return True
    return False


def _jamo(text: str) -> list[str]:
    out: list[str] = []
    for ch in text:
        code = ord(ch) - 0xAC00
        if 0 <= code < 11172:
            out += [str(code // 588), "v" + str((code % 588) // 28), "f" + str(code % 28)]
        else:
            out.append(ch)
    return out


def _jamo_distance(a: str, b: str) -> int:
    """Edit distance over jamo: "철뷴" ~ "철분" (one vowel), "철판" vs "철분" (two)."""
    x, y = _jamo(a), _jamo(b)
    prev = list(range(len(y) + 1))
    for i, cx in enumerate(x, 1):
        cur = [i]
        for j, cy in enumerate(y, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (cx != cy)))
        prev = cur
    return prev[-1]


def _is_hangul_syllable(ch: str) -> bool:
    return "가" <= ch <= "힣"

"""Diet-only log parser.

Extracts food items, meal_type, and time_of_day.
Does NOT output estimated_calories — BE calculates kcal from macros (4/4/9).
Macros are floats with 1 decimal place, or null when unknown.
"""

from __future__ import annotations

import logging
import os
import re
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from engines.llm_router import _parse_json_content, get_llm, resolve_llm_provider
from engines.log_redaction import sanitize_exception_for_log, summarize_text_for_log

from engines.food.food_units import (
    FOOD_UNIT_GRAMS as _SHARED_FOOD_UNIT_GRAMS,
    UNIT_DEFAULT_GRAMS as _SHARED_UNIT_DEFAULT_GRAMS,
    resolve_portion_grams,
    unit_to_grams as _shared_unit_to_grams,
)

try:
    from engines.food.food_engine import get_food_engine
except Exception:
    get_food_engine = None

logger = logging.getLogger(__name__)

_TRUE_VALUES = {"true", "1", "yes", "on"}

# ---------- Schema ----------

class DietParseItem(BaseModel):
    food_name: str
    amount: Optional[float] = None
    unit: Optional[str] = None
    protein_g: Optional[float] = None
    carbs_g: Optional[float] = None
    fat_g: Optional[float] = None
    kcal: Optional[float] = None
    source: Optional[str] = None
    meal_type: Optional[str] = None   # breakfast | lunch | dinner | snack | null
    time_of_day: Optional[str] = None  # "HH:mm" | null


class ParsedDietLog(BaseModel):
    items: List[DietParseItem] = Field(default_factory=list)


# ---------- Common food table (per 100 g) ----------

_FOOD_PER_100G: dict[str, dict[str, float]] = {
    "닭가슴살": {"protein_g": 31.0, "carbs_g": 0.0,  "fat_g": 3.6},
    "쌀밥":    {"protein_g": 2.4,  "carbs_g": 28.6, "fat_g": 0.3},
    "밥":      {"protein_g": 2.4,  "carbs_g": 28.6, "fat_g": 0.3},
    "잡곡밥":  {"protein_g": 3.1,  "carbs_g": 32.4, "fat_g": 0.7},
    "현미밥":  {"protein_g": 2.6,  "carbs_g": 23.5, "fat_g": 0.9},
    "계란":    {"protein_g": 12.6, "carbs_g": 0.7,  "fat_g": 9.5},
    "달걀":    {"protein_g": 12.6, "carbs_g": 0.7,  "fat_g": 9.5},
    "고구마":  {"protein_g": 1.6,  "carbs_g": 20.1, "fat_g": 0.1},
    "두부":    {"protein_g": 8.1,  "carbs_g": 1.9,  "fat_g": 4.8},
    "연어":    {"protein_g": 20.0, "carbs_g": 0.0,  "fat_g": 13.5},
    "오트밀":  {"protein_g": 16.9, "carbs_g": 66.3, "fat_g": 6.9},
    "삶은 달걀": {"protein_g": 12.6, "carbs_g": 0.7, "fat_g": 9.5},
    # DB(food_db) 에 행이 없는 흔한 항목 (per 100 g / 100 ml)
    "콜라":    {"protein_g": 0.0,  "carbs_g": 10.6, "fat_g": 0.0},
    "사이다":  {"protein_g": 0.0,  "carbs_g": 10.0, "fat_g": 0.0},
    "치킨":    {"protein_g": 19.0, "carbs_g": 9.0,  "fat_g": 15.0},
}

_GRAM_UNITS = {"g", "gram", "grams", "그램", "ml", "밀리리터", "cc"}

# ---------- Unit → gram conversion tables ----------
# 단일 출처: engines/food/food_units.py (food_engine 과 공유). 모든 값은 추정 기본값.
_UNIT_DEFAULT_GRAMS: dict[str, float] = _SHARED_UNIT_DEFAULT_GRAMS
_FOOD_UNIT_GRAMS: dict[str, dict[str, float]] = _SHARED_FOOD_UNIT_GRAMS


def _unit_to_grams(food_name: str, unit: str | None) -> float | None:
    """비그램 단위를 gram으로 환산. 변환 불가면 None 반환."""
    return _shared_unit_to_grams(food_name, unit)


# ---------- Regex patterns ----------

_MEAL_PATTERNS = [
    (re.compile(r"아침|조식"), "breakfast"),
    (re.compile(r"점심|중식"), "lunch"),
    (re.compile(r"저녁|석식"), "dinner"),
    (re.compile(r"간식|야식"), "snack"),
]

_TIME_RE = re.compile(r"(\d{1,2})시\s*(\d{1,2})?\s*분?|(\d{1,2}):(\d{2})")

_AMOUNT_UNIT_RE = re.compile(
    r"(?P<food>[가-힣A-Za-z][가-힣A-Za-z0-9\s]{0,20}?)"
    r"\s*(?P<amount>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>g|그램|ml|공기|개|알|인분|스쿱|조각|캔|병|그릇|봉지|팩|컵|잔|장|모|줄|쪽)?"
    r"(?![가-힣A-Za-z0-9])",
    re.IGNORECASE,
)

_CONNECTOR_RE = re.compile(r"랑|이랑|하고|그리고|와\s+|,\s*")
_MEAL_WORD_RE = re.compile(r"(?:아침|조식|점심|중식|저녁|석식|간식|야식)(?:으로는|에는|으론|엔|에|은|는|으로|로)?")
_TIME_WORD_RE = re.compile(r"\d{1,2}시\s*\d{0,2}분?|\d{1,2}:\d{2}")
_FILLER_RE = re.compile(
    r"(?:먹었|먹음|먹고|먹어|먹은|먹을|먹기|섭취|마셨|마심|마시고|드셨|드심)[가-힣]*"
)


# ---------- Helpers ----------

def _extract_meal_type(text: str) -> str | None:
    for pattern, meal_type in _MEAL_PATTERNS:
        if pattern.search(text):
            return meal_type
    return None


def _extract_time_of_day(text: str, meal_type: str | None = None) -> str | None:
    m = _TIME_RE.search(text)
    if not m:
        return None
    if m.group(1) is not None:
        hour = int(m.group(1))
        minute = int(m.group(2)) if m.group(2) else 0
        # 저녁 + hour <= 12 → PM heuristic
        if meal_type == "dinner" and hour <= 12:
            hour += 12
        return f"{hour:02d}:{minute:02d}"
    return f"{int(m.group(3)):02d}:{m.group(4)}"


def _r1(v: float) -> float:
    return round(v, 1)


def _macros_from_table(food_name: str, amount: float | None, unit: str | None) -> dict[str, float | None]:
    nutrition = _FOOD_PER_100G.get(food_name.strip())
    if not nutrition or not amount:
        return {"protein_g": None, "carbs_g": None, "fat_g": None}
    grams = resolve_portion_grams(food_name.strip(), amount, unit)
    if grams is None:
        return {"protein_g": None, "carbs_g": None, "fat_g": None}
    factor = grams / 100
    return {
        "protein_g": _r1(nutrition["protein_g"] * factor),
        "carbs_g":   _r1(nutrition["carbs_g"]   * factor),
        "fat_g":     _r1(nutrition["fat_g"]      * factor),
    }


# ---------- Deterministic fallback ----------

# Particles are stripped only at the END of a word, and never from words whose
# final syllable is part of the word itself (삶은 달걀, 포도, 아보카도).
_PARTICLE_RE = re.compile(r"(?<=[가-힣A-Za-z0-9])(?:에서|에는|에|으로|은|는|도|을|를)(?=\s|,|$)")
_PARTICLE_PROTECTED_WORDS = frozenset({
    "삶은", "구운", "볶은", "찐", "튀긴", "데친", "익은", "작은", "얇은", "굵은",
    "포도", "청포도", "적포도", "샤인머스캣포도", "아보카도", "오레가노",
    "망고", "애플망고", "소다", "크림소다", "식은", "남은",
    "고구마", "누룽지", "만두", "우동", "두부", "라면",
})
_WORD_RE = re.compile(r"[^\s,]+|[\s,]+")

# Non-food chatter tokens dropped by the connector fallback.
_CHATTER_WORDS = frozenset({
    "오늘", "어제", "그제", "방금", "아까", "지금", "너무", "진짜", "정말", "좀", "조금",
    "많이", "약간", "그냥", "그래도", "근데", "그런데", "그리고", "그래서", "하지만",
    "나", "난", "내가", "저", "전", "제가", "우리", "다", "또", "안", "못", "같이",
    "엔", "에", "때", "후", "전에", "후에", "다음", "먼저", "대충", "간단히", "든든하게",
    "아", "어", "음", "잘", "다시", "혼자", "많이", "배부르게",
})
# Workout / non-food words that may share a sentence with food.
_NON_FOOD_WORDS = frozenset({
    "운동", "헬스", "스쿼트", "벤치프레스", "데드리프트", "러닝", "달리기", "조깅", "걷기",
    "산책", "수영", "요가", "필라테스", "플랭크", "푸시업", "팔굽혀펴기", "턱걸이", "풀업",
    "런지", "사이클", "자전거", "등산", "스트레칭", "유산소", "웨이트",
})
_WORKOUT_QTY_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:kg|회|세트|분|시간|km|킬로|바퀴|층|보)")
# Only clear verb/adjective conjugations; bare endings like 음/고 are also food
# words (제육볶음, 망고), so they are not treated as predicates.
_PREDICATE_END_RE = re.compile(
    r"(?:했다|었다|았다|였다|했어|었어|았어|였어|해요|했고|하고|해서|한다|된다|됐다|"
    r"했는데|었는데|았는데|는데|지만|니까|어서|아서|었음|았음|했음|힘들다|피곤함|좋다|싫다|없다|있다)$"
)


def _strip_particles(text: str) -> str:
    out = []
    for piece in _WORD_RE.findall(text):
        if piece.strip(" ,") and piece not in _PARTICLE_PROTECTED_WORDS and piece not in _CHATTER_WORDS:
            stripped = _PARTICLE_RE.sub("", piece)
            piece = stripped if stripped else piece
        out.append(piece)
    return "".join(out)


def _strip_context_words(text: str) -> str:
    """Remove meal/time markers and particles so food regex starts at food names."""
    s = _MEAL_WORD_RE.sub(" ", text)
    s = _TIME_WORD_RE.sub(" ", s)
    s = _strip_particles(s)
    return re.sub(r"\s{2,}", " ", s).strip()


def _drop_chatter(part: str) -> str:
    """Keep only food-looking tokens from a free-text fragment."""
    kept = []
    for token in part.split():
        if token in _CHATTER_WORDS or token in _NON_FOOD_WORDS:
            continue
        if token not in _PARTICLE_PROTECTED_WORDS and _PREDICATE_END_RE.search(token):
            continue
        kept.append(token)
    return " ".join(kept)


def _deterministic_diet_parse(user_text: str) -> ParsedDietLog:
    text = (user_text or "").strip()
    if not text:
        return ParsedDietLog(items=[])

    meal_type = _extract_meal_type(text)
    time_of_day = _extract_time_of_day(text, meal_type)

    items: list[DietParseItem] = []

    text_for_food = _FILLER_RE.sub(" ", _strip_context_words(text))
    for segment in _CONNECTOR_RE.split(text_for_food):
        segment = segment.strip(" :;.!?~")
        if not segment or _WORKOUT_QTY_RE.search(segment):
            # workout clause ("벤치프레스 60kg 10회 3세트") — not a food
            continue
        # Pass 1: food + amount (+ unit)
        leftover = segment
        for m in _AMOUNT_UNIT_RE.finditer(segment):
            leftover = leftover.replace(m.group(0), " ", 1)
            food_name = _drop_chatter(m.group("food").strip())
            if not food_name:
                continue
            amount = float(m.group("amount"))
            unit = m.group("unit")
            if unit and unit.lower() == "그램":
                unit = "g"
            macros = _macros_from_table(food_name, amount, unit)
            items.append(DietParseItem(
                food_name=food_name,
                amount=amount,
                unit=unit,
                meal_type=meal_type,
                time_of_day=time_of_day,
                **macros,
            ))
        # Pass 2: foods without an amount ("..., 아메리카노", "저녁엔 샐러드 먹음")
        food_name = _drop_chatter(leftover.strip(" :;.!?~"))
        # leftover tokens with digits are unparsed quantities, not foods
        food_name = " ".join(t for t in food_name.split() if not re.search(r"\d", t))
        if food_name:
            macros = _macros_from_table(food_name, None, None)
            items.append(DietParseItem(
                food_name=food_name,
                meal_type=meal_type,
                time_of_day=time_of_day,
                **macros,
            ))

    return ParsedDietLog(items=items)


# ---------- Macro enrichment ----------

_MAX_KCAL_PER_GRAM = 9.0  # 순수 지방 = 9 kcal/g. 이를 넘는 추정은 물리적으로 불가능.


def _macro_kcal(protein, carbs, fat) -> float:
    return 4.0 * (protein or 0.0) + 4.0 * (carbs or 0.0) + 9.0 * (fat or 0.0)


def _sanity_check_llm_macros(item: DietParseItem) -> DietParseItem:
    """DB miss 시 LLM 매크로를 유지하되 섭취량 대비 kcal 밀도를 검증.

    >9 kcal/g (또는 매크로 질량 > 섭취량) 이면 per-100g 값을 portion 에 그대로
    쓴 것으로 보고 portion 으로 재환산한다. 재환산해도 불가능하면 매크로를 버린다.
    """
    if item.protein_g is None and item.carbs_g is None and item.fat_g is None and item.kcal is None:
        return item
    grams = resolve_portion_grams(item.food_name, item.amount, item.unit)
    if not grams or grams <= 0:
        return item
    macro_mass = (item.protein_g or 0.0) + (item.carbs_g or 0.0) + (item.fat_g or 0.0)
    kcal = max(_macro_kcal(item.protein_g, item.carbs_g, item.fat_g), item.kcal or 0.0)
    if kcal / grams <= _MAX_KCAL_PER_GRAM and macro_mass <= grams * 1.0001:
        return item
    factor = grams / 100.0
    scaled = {
        key: (_r1(value * factor) if value is not None else None)
        for key, value in (
            ("protein_g", item.protein_g),
            ("carbs_g", item.carbs_g),
            ("fat_g", item.fat_g),
            ("kcal", item.kcal),
        )
    }
    scaled_mass = sum(scaled[k] or 0.0 for k in ("protein_g", "carbs_g", "fat_g"))
    scaled_kcal = max(_macro_kcal(scaled["protein_g"], scaled["carbs_g"], scaled["fat_g"]), scaled["kcal"] or 0.0)
    if factor < 1.0 and scaled_kcal / grams <= _MAX_KCAL_PER_GRAM and scaled_mass <= grams:
        logger.info("[diet macros] implausible LLM density rescaled as per-100g values")
        return item.model_copy(update=scaled)
    logger.info("[diet macros] implausible LLM macros dropped")
    return item.model_copy(update={"protein_g": None, "carbs_g": None, "fat_g": None, "kcal": None})


def _enrich_macros(parsed: ParsedDietLog) -> ParsedDietLog:
    engine = get_food_engine() if get_food_engine is not None else None
    enriched = []
    for item in parsed.items:
        # Curated plain staples (닭가슴살, 쌀밥, 계란 …) beat vector search, which can land on a
        # composite row such as "샐러드 닭가슴살" (14g protein per 100g instead of 31g).
        if item.food_name.strip() in _FOOD_PER_100G:
            macros = _macros_from_table(item.food_name.strip(), item.amount, item.unit)
            if macros["protein_g"] is not None:
                enriched.append(item.model_copy(update={**macros, "source": "db"}))
                continue
        db_hit = None
        if engine is not None and getattr(engine, "available", False):
            try:
                db_hit = engine.search(item.food_name, item.amount, item.unit)
            except Exception:
                db_hit = None
        if db_hit is not None:
            update = {
                "protein_g": db_hit.get("protein_g"),
                "carbs_g": db_hit.get("carbs_g"),
                "fat_g": db_hit.get("fat_g"),
                "kcal": db_hit.get("kcal"),
                "source": "db",
            }
            # 수량 미기재 → DB 가 basis(예: 100g) 기준으로 계산했으므로 그 수량을 명시
            if item.amount is None and db_hit.get("resolved_amount") is not None:
                update["amount"] = db_hit.get("resolved_amount")
                update["unit"] = db_hit.get("resolved_unit")
            enriched.append(item.model_copy(update=update))
            continue
        food_key = item.food_name.strip()
        is_known = food_key in _FOOD_PER_100G
        all_null = item.protein_g is None and item.carbs_g is None and item.fat_g is None
        if is_known:
            macros = _macros_from_table(food_key, item.amount, item.unit)
            if macros["protein_g"] is not None:
                item = item.model_copy(update=macros)
            else:
                item = _sanity_check_llm_macros(item)
        elif all_null:
            macros = _macros_from_table(food_key, item.amount, item.unit)
            item = item.model_copy(update=macros)
        else:
            item = _sanity_check_llm_macros(item)
        enriched.append(item)
    return ParsedDietLog(items=enriched)


def _reconcile_amounts_with_text(parsed: ParsedDietLog, user_text: str) -> ParsedDietLog:
    """LLM 이 "계란 2개" 를 200 g 처럼 임의 환산한 경우 사용자가 쓴 수량/단위로 되돌린다."""
    literal = {}
    for item in _deterministic_diet_parse(user_text).items:
        if item.amount is not None:
            literal.setdefault("".join(item.food_name.split()), item)
    if not literal:
        return parsed
    fixed = []
    for item in parsed.items:
        source = literal.get("".join(item.food_name.split()))
        if source is not None and (item.amount != source.amount or (item.unit or None) != (source.unit or None)):
            update = {"amount": source.amount, "unit": source.unit}
            # LLM 매크로는 LLM 이 쓴 수량 기준 → 사용자 수량 기준으로 재환산 (불가 시 비움)
            llm_grams = resolve_portion_grams(item.food_name, item.amount, item.unit)
            user_grams = resolve_portion_grams(item.food_name, source.amount, source.unit)
            for key in ("protein_g", "carbs_g", "fat_g", "kcal"):
                value = getattr(item, key)
                if item.amount is None:
                    # LLM 이 수량 없이 준 매크로 = 사용자가 쓴 portion 기준으로 간주
                    update[key] = value
                elif value is not None and llm_grams and user_grams:
                    update[key] = _r1(value * user_grams / llm_grams)
                else:
                    update[key] = None
            item = item.model_copy(update=update)
        fixed.append(item)
    return ParsedDietLog(items=fixed)


# ---------- LLM path ----------

def _build_diet_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Korean diet log parser. Extract food items ONLY (no exercise).
Return JSON only. No markdown fences. No explanations.
Korean meal keywords: 아침/조식=breakfast, 점심/중식=lunch, 저녁/석식=dinner, 간식/야식=snack.
For time_of_day: extract ONLY if the user explicitly states a time (e.g. "7시", "19:00", "오후 7시"). Do NOT guess or infer.
Ignore non-food chatter (feelings, workouts, weather) but still extract every food mentioned anywhere in the text (e.g. "오늘 너무 피곤했다 그래도 저녁엔 샐러드 먹음" → one item 샐러드, meal_type dinner).
Keep amount and unit exactly as the user wrote them. Do NOT convert counts to grams (계란 2개 → amount 2.0, unit "개"; 콜라 1캔 → amount 1.0, unit "캔"). If no amount is given, use amount null and unit null.
Macros (protein_g, carbs_g, fat_g) MUST be estimated floats with 1 decimal place for the WHOLE PORTION eaten (amount × unit), not per 100 g (e.g. 닭가슴살 200g → protein_g 62.0, carbs_g 0.0, fat_g 7.2; 계란 2개 → protein_g 12.6, carbs_g 0.4, fat_g 9.5). NEVER return null — use 0.0 only if you have absolutely no data for the food.
Do NOT include an estimated_calories field.
Use this exact JSON shape:
{{"items":[{{"food_name":"string","amount":200.0,"unit":"g","protein_g":62.0,"carbs_g":0.0,"fat_g":7.2,"meal_type":"breakfast","time_of_day":null}}]}}
meal_type must be one of: breakfast, lunch, dinner, snack, or null.
time_of_day must be "HH:mm" (24-hour) or null.""",
        ),
        ("human", "Diet log: {text}"),
    ])


def _local_raw_json_enabled() -> bool:
    return os.getenv("ENABLE_LOCAL_RAW_JSON_INVOKE", "").strip().lower() in _TRUE_VALUES


def _parse_payload(payload) -> ParsedDietLog:
    if isinstance(payload, ParsedDietLog):
        return payload
    if isinstance(payload, dict):
        return ParsedDietLog.model_validate(payload)
    if isinstance(payload, str):
        return ParsedDietLog.model_validate(_parse_json_content(payload))
    content = getattr(payload, "content", None)
    if isinstance(content, str):
        return ParsedDietLog.model_validate(_parse_json_content(content))
    return ParsedDietLog.model_validate(payload)


# ---------- Public API ----------

def parse_diet_log(user_text: str) -> str:
    """Parse natural-language diet input into a JSON string (ParsedDietLog).

    Never logs raw user input. No estimated_calories in output.
    Falls back to deterministic regex parse if LLM is unavailable.
    """
    if not (user_text or "").strip():
        return ParsedDietLog(items=[]).model_dump_json()

    try:
        llm = get_llm("diet", temperature=0.1)
        logger.info("[LLM diet parsing start] %s", summarize_text_for_log(user_text))

        if resolve_llm_provider().effective_provider == "local" and _local_raw_json_enabled():
            parsed = _parse_payload((_build_diet_prompt() | llm).invoke({"text": user_text}))
        else:
            structured_llm = llm.with_structured_output(ParsedDietLog)
            if hasattr(structured_llm, "__ror__"):
                parsed = _parse_payload((_build_diet_prompt() | structured_llm).invoke({"text": user_text}))
            else:
                parsed = _parse_payload(structured_llm.invoke({"text": user_text}))

        if not parsed.items:
            # LLM 이 잡담 섞인 입력에서 음식을 놓친 경우 → 결정론 추출로 보강
            parsed = _deterministic_diet_parse(user_text)
        else:
            parsed = _reconcile_amounts_with_text(parsed, user_text)
        return _enrich_macros(parsed).model_dump_json()
    except Exception as exc:
        logger.warning("[LLM diet parsing fallback] %s", sanitize_exception_for_log(exc))
        return _deterministic_diet_parse(user_text).model_dump_json()

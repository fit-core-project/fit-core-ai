"""Diet-only log parser.

Extracts food items, meal_type, and time_of_day.
Does NOT output estimated_calories — BE calculates kcal from macros (4/4/9).
Macros are floats with 1 decimal place, or null when unknown.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from engines.llm_router import _parse_json_content, get_llm, resolve_llm_provider
from engines.log_redaction import sanitize_exception_for_log, summarize_text_for_log

_TRUE_VALUES = {"true", "1", "yes", "on"}

# ---------- Schema ----------

class DietParseItem(BaseModel):
    food_name: str
    amount: Optional[float] = None
    unit: Optional[str] = None
    protein_g: Optional[float] = None
    carbs_g: Optional[float] = None
    fat_g: Optional[float] = None
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
}

_GRAM_UNITS = {"g", "gram", "grams", "그램"}

# ---------- Unit → gram conversion tables ----------
# 모든 값은 추정 기본값. 식품/조리법에 따라 편차 존재. 추후 보정 가능.

# 식품 무관 단위 기본값 (편차 허용 범위 내 단위만 포함)
_UNIT_DEFAULT_GRAMS: dict[str, float] = {
    "공기": 210.0,   # 쌀밥 1공기 (한국 표준 계량)
    "그릇": 400.0,   # 국/찌개류 1그릇 평균
    "인분": 200.0,   # 고기류 1인분 평균
    "스쿱": 30.0,    # 단백질 파우더 1스쿱 평균
    "모": 300.0,     # 두부 1모 기준
    "컵": 200.0,     # 200ml 기준
}

# 식품별 단위 override ('개'처럼 식품마다 편차가 큰 단위)
_FOOD_UNIT_GRAMS: dict[str, dict[str, float]] = {
    "계란":      {"개": 50.0},   # 중란 기준
    "달걀":      {"개": 50.0},
    "삶은 달걀": {"개": 50.0},
    "바나나":    {"개": 120.0},  # 중간 크기
    "사과":      {"개": 200.0},
    "고구마":    {"개": 150.0},
    "감자":      {"개": 130.0},
    "오렌지":    {"개": 180.0},
}


def _unit_to_grams(food_name: str, unit: str | None) -> float | None:
    """비그램 단위를 gram으로 환산. 변환 불가면 None 반환."""
    if unit is None:
        return None
    u = unit.lower().strip()
    overrides = _FOOD_UNIT_GRAMS.get(food_name.strip(), {})
    if u in overrides:
        return overrides[u]
    return _UNIT_DEFAULT_GRAMS.get(u)

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
    r"\s+(?P<amount>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>g|그램|ml|공기|개|인분|스쿱)",
    re.IGNORECASE,
)

_CONNECTOR_RE = re.compile(r"랑|이랑|하고|그리고|와\s+|,\s*")
_MEAL_WORD_RE = re.compile(r"아침|조식|점심|중식|저녁|석식|간식|야식")
_TIME_WORD_RE = re.compile(r"\d{1,2}시\s*\d{0,2}분?|\d{1,2}:\d{2}")
_FILLER_RE = re.compile(r"에서?|으로|를|을|도|은|는|먹었\w*|먹음|섭취\w*|마셨\w*|드셨\w*")


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
    unit_lower = (unit or "").lower().strip()
    if unit_lower in _GRAM_UNITS:
        grams = amount
    else:
        per_unit = _unit_to_grams(food_name.strip(), unit)
        if per_unit is None:
            return {"protein_g": None, "carbs_g": None, "fat_g": None}
        grams = amount * per_unit
    factor = grams / 100
    return {
        "protein_g": _r1(nutrition["protein_g"] * factor),
        "carbs_g":   _r1(nutrition["carbs_g"]   * factor),
        "fat_g":     _r1(nutrition["fat_g"]      * factor),
    }


# ---------- Deterministic fallback ----------

_PARTICLE_RE = re.compile(r"에서?|으로|도(?=\s)|은|는")


def _strip_context_words(text: str) -> str:
    """Remove meal/time markers and particles so food regex starts at food names."""
    s = _MEAL_WORD_RE.sub("", text)
    s = _TIME_WORD_RE.sub("", s)
    s = _PARTICLE_RE.sub(" ", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def _deterministic_diet_parse(user_text: str) -> ParsedDietLog:
    text = (user_text or "").strip()
    if not text:
        return ParsedDietLog(items=[])

    meal_type = _extract_meal_type(text)
    time_of_day = _extract_time_of_day(text, meal_type)

    items: list[DietParseItem] = []

    # Pass 1: food + amount + unit (run on context-stripped text)
    text_for_food = _strip_context_words(text)
    for m in _AMOUNT_UNIT_RE.finditer(text_for_food):
        food_name = m.group("food").strip()
        amount = float(m.group("amount"))
        unit = m.group("unit")
        if unit.lower() == "그램":
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

    # Pass 2: if no structured match, split by connectors
    if not items:
        stripped = _FILLER_RE.sub(" ", text_for_food)
        parts = _CONNECTOR_RE.split(stripped)
        for part in parts:
            food_name = part.strip()
            if len(food_name) >= 1:
                macros = _macros_from_table(food_name, None, None)
                items.append(DietParseItem(
                    food_name=food_name,
                    meal_type=meal_type,
                    time_of_day=time_of_day,
                    **macros,
                ))

    return ParsedDietLog(items=items)


# ---------- Macro enrichment ----------

def _enrich_macros(parsed: ParsedDietLog) -> ParsedDietLog:
    """Enrich macros from food table.

    Policy:
    - known food (in _FOOD_PER_100G) + computable weight → always prefer table (overrides LLM)
    - unknown food, all macros null → try table (won't match, safe)
    - unknown food, macros present → keep LLM values
    """
    enriched = []
    for item in parsed.items:
        food_key = item.food_name.strip()
        is_known = food_key in _FOOD_PER_100G
        all_null = item.protein_g is None and item.carbs_g is None and item.fat_g is None
        if is_known:
            macros = _macros_from_table(food_key, item.amount, item.unit)
            if macros["protein_g"] is not None:
                item = item.model_copy(update=macros)
        elif all_null:
            macros = _macros_from_table(food_key, item.amount, item.unit)
            item = item.model_copy(update=macros)
        enriched.append(item)
    return ParsedDietLog(items=enriched)


# ---------- LLM path ----------

def _build_diet_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Korean diet log parser. Extract food items ONLY (no exercise).
Return JSON only. No markdown fences. No explanations.
Korean meal keywords: 아침/조식=breakfast, 점심/중식=lunch, 저녁/석식=dinner, 간식/야식=snack.
For time_of_day: extract ONLY if the user explicitly states a time (e.g. "7시", "19:00", "오후 7시"). Do NOT guess or infer.
Macros (protein_g, carbs_g, fat_g) MUST be estimated floats with 1 decimal place using your nutritional knowledge (e.g. 닭가슴살 100g → protein_g 31.0, carbs_g 0.0, fat_g 3.6). NEVER return null — use 0.0 only if you have absolutely no data for the food.
Do NOT include an estimated_calories field.
Use this exact JSON shape:
{{"items":[{{"food_name":"string","amount":200.0,"unit":"g","protein_g":31.0,"carbs_g":0.0,"fat_g":3.6,"meal_type":"breakfast","time_of_day":null}}]}}
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
        print("[LLM diet parsing start]", summarize_text_for_log(user_text))

        if resolve_llm_provider().effective_provider == "local" and _local_raw_json_enabled():
            parsed = _parse_payload((_build_diet_prompt() | llm).invoke({"text": user_text}))
        else:
            structured_llm = llm.with_structured_output(ParsedDietLog)
            if hasattr(structured_llm, "__ror__"):
                parsed = _parse_payload((_build_diet_prompt() | structured_llm).invoke({"text": user_text}))
            else:
                parsed = _parse_payload(structured_llm.invoke({"text": user_text}))

        return _enrich_macros(parsed).model_dump_json()
    except Exception as exc:
        print("[LLM diet parsing fallback]", sanitize_exception_for_log(exc))
        return _deterministic_diet_parse(user_text).model_dump_json()

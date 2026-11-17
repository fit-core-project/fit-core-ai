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

from engines.food.curated_products import find_processed_product, product_nutrition
try:
    from engines.food.food_engine import category_nutrition
except Exception:  # pragma: no cover - food engine optional
    def category_nutrition(*_args, **_kwargs):
        return None
from engines.food.food_units import (
    FOOD_UNIT_GRAMS as _SHARED_FOOD_UNIT_GRAMS,
    UNIT_DEFAULT_GRAMS as _SHARED_UNIT_DEFAULT_GRAMS,
    portion_confidence,
    resolve_portion_grams,
    serving_grams,
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
    # 주류: 알코올 열량(7 kcal/g)은 P/C/F(4/4/9)로 표현되지 않아 kcal 를 명시 (라벨 기준, per 100 ml).
    # 이 값이 없으면 LLM 이 매크로 0 을 써서 소주 1병이 0 kcal 로 기록됐다.
    "맥주":    {"protein_g": 0.4, "carbs_g": 3.5, "fat_g": 0.0, "kcal": 43.0},
    "생맥주":  {"protein_g": 0.4, "carbs_g": 3.5, "fat_g": 0.0, "kcal": 43.0},
    "생맥":    {"protein_g": 0.4, "carbs_g": 3.5, "fat_g": 0.0, "kcal": 43.0},
    "소주":    {"protein_g": 0.0, "carbs_g": 0.5, "fat_g": 0.0, "kcal": 113.0},
    "막걸리":  {"protein_g": 1.3, "carbs_g": 3.0, "fat_g": 0.0, "kcal": 50.0},
    "와인":    {"protein_g": 0.1, "carbs_g": 2.6, "fat_g": 0.0, "kcal": 83.0},
    "레드와인": {"protein_g": 0.1, "carbs_g": 2.6, "fat_g": 0.0, "kcal": 85.0},
    "화이트와인": {"protein_g": 0.1, "carbs_g": 2.6, "fat_g": 0.0, "kcal": 82.0},
    "소맥":    {"protein_g": 0.3, "carbs_g": 2.8, "fat_g": 0.0, "kcal": 60.0},
    "물":      {"protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0, "kcal": 0.0},
    # brand beers / makgeolli / spirits share the generic label values (per 100 ml)
    "카스":    {"protein_g": 0.4, "carbs_g": 3.5, "fat_g": 0.0, "kcal": 43.0},
    "테라":    {"protein_g": 0.4, "carbs_g": 3.5, "fat_g": 0.0, "kcal": 43.0},
    "하이트":  {"protein_g": 0.4, "carbs_g": 3.5, "fat_g": 0.0, "kcal": 43.0},
    "켈리":    {"protein_g": 0.4, "carbs_g": 3.5, "fat_g": 0.0, "kcal": 43.0},
    "장수막걸리": {"protein_g": 1.3, "carbs_g": 3.0, "fat_g": 0.0, "kcal": 50.0},
    "위스키":  {"protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0, "kcal": 250.0},
    "양주":    {"protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0, "kcal": 250.0},
    "하이볼":  {"protein_g": 0.0, "carbs_g": 5.0, "fat_g": 0.0, "kcal": 50.0},
    "논알콜맥주": {"protein_g": 0.2, "carbs_g": 5.0, "fat_g": 0.0, "kcal": 21.0},
    "생수":    {"protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0, "kcal": 0.0},
    # Alcohol: ethanol (7 kcal/g) is not in protein/carbs/fat, so an LLM macro estimate reads
    # ~0 kcal. Category label values per 100 ml (MFDS / 국가표준식품성분표 typical ranges).
    "청하":    {"protein_g": 0.2, "carbs_g": 4.0, "fat_g": 0.0, "kcal": 105.0},
    "청주":    {"protein_g": 0.2, "carbs_g": 4.0, "fat_g": 0.0, "kcal": 110.0},
    "사케":    {"protein_g": 0.2, "carbs_g": 4.0, "fat_g": 0.0, "kcal": 110.0},
    "정종":    {"protein_g": 0.2, "carbs_g": 4.0, "fat_g": 0.0, "kcal": 110.0},
    "고량주":  {"protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0, "kcal": 300.0},
    "백주":    {"protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0, "kcal": 300.0},
    "보드카":  {"protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0, "kcal": 230.0},
    "데킬라":  {"protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0, "kcal": 230.0},
    "진":      {"protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0, "kcal": 230.0},
    "럼":      {"protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0, "kcal": 230.0},
    "복분자주": {"protein_g": 0.2, "carbs_g": 12.0, "fat_g": 0.0, "kcal": 137.0},
    "매실주":  {"protein_g": 0.0, "carbs_g": 12.0, "fat_g": 0.0, "kcal": 150.0},
    "과실주":  {"protein_g": 0.0, "carbs_g": 12.0, "fat_g": 0.0, "kcal": 150.0},
    "과일소주": {"protein_g": 0.0, "carbs_g": 8.0, "fat_g": 0.0, "kcal": 105.0},
    # soju brands: MFDS labels (대선159 88, 새로 89, 좋은데이 89 kcal/100 ml)
    **{brand: {"protein_g": 0.0, "carbs_g": 0.5, "fat_g": 0.0, "kcal": 88.0} for brand in (
        "참이슬", "처음처럼", "진로", "진로이즈백", "대선", "새로", "좋은데이", "한라산", "잎새주",
        "c1", "시원", "맛있는참", "이제우린")},
    # flavoured soju (새로 다래 68) and canned soju highballs (순하리 레몬진/자몽진 9% 53-54)
    **{brand: {"protein_g": 0.0, "carbs_g": 6.0, "fat_g": 0.0, "kcal": 68.0} for brand in (
        "자몽에이슬", "청포도에이슬", "자두에이슬", "새로 다래", "새로다래", "새로 오미자")},
    **{brand: {"protein_g": 0.0, "carbs_g": 4.0, "fat_g": 0.0, "kcal": 53.0} for brand in (
        "순하리", "순하리 레몬진", "레몬진", "자몽진", "순하리 자몽진")},
    # drinks missing from the generic DB (label values per 100 ml); their names otherwise
    # land on look-alike rows (파워에이드 -> 라임에이드, 몬스터 -> 몬스터 아이스크림)
    "파워에이드": {"protein_g": 0.0, "carbs_g": 3.0, "fat_g": 0.0, "kcal": 12.0},
    "게토레이": {"protein_g": 0.0, "carbs_g": 6.0, "fat_g": 0.0, "kcal": 24.0},
    "몬스터":  {"protein_g": 0.0, "carbs_g": 11.5, "fat_g": 0.0, "kcal": 47.0},
    "몬스터 에너지": {"protein_g": 0.0, "carbs_g": 11.5, "fat_g": 0.0, "kcal": 47.0},
    "몬스터 울트라": {"protein_g": 0.0, "carbs_g": 0.9, "fat_g": 0.0, "kcal": 3.0},
    "레드불":  {"protein_g": 0.0, "carbs_g": 11.0, "fat_g": 0.0, "kcal": 46.0},
    # protein powders (label category values per 100 g; 1 스쿱 = 30 g)
    **{name: {"protein_g": 75.0, "carbs_g": 10.0, "fat_g": 6.0, "kcal": 390.0} for name in (
        "프로틴", "웨이 프로틴", "웨이프로틴", "프로틴 파우더", "단백질 파우더", "단백질 보충제", "WPI", "wpi")},
    # supplement capsules (per 100 g; one 알 ≈ 1 g, see food_units): fish-oil softgel ≈ 9 kcal
    "오메가3":  {"protein_g": 0.0, "carbs_g": 0.0, "fat_g": 100.0, "kcal": 900.0},
    "멀티비타민": {"protein_g": 0.0, "carbs_g": 60.0, "fat_g": 0.0, "kcal": 250.0},
    "종합비타민": {"protein_g": 0.0, "carbs_g": 60.0, "fat_g": 0.0, "kcal": 250.0},
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

_UNIT_WORDS = (
    "g|그램|ml|공기|개|알|인분|스쿱|조각|캔|병|그릇|봉지|팩|컵|잔|장|모|줄|쪽|"
    "줌|접시|마리|판|송이|통|봉|입|사발|대접|리터"
)
_AMOUNT_UNIT_RE = re.compile(
    r"(?P<food>[가-힣A-Za-z][가-힣A-Za-z0-9\s]{0,20}?)"
    # digits glued to a name are part of it ("오메가3", "비타민D3", "B12"), not an amount
    r"\s*(?<!오메가)(?<![A-Za-z])(?P<amount>\d+(?:\.\d+)?)\s*"
    rf"(?P<unit>{_UNIT_WORDS})?"
    # a particle may follow the unit ("반공기에", "1/3공기만", "2개씩")
    r"(?=[\s,.!?~)]|$|만|씩|정도|쯤|이랑|랑|에|을|를|은|는|도|하고|과|와|이|가)",
    re.IGNORECASE,
)

# Korean number words before a unit ("한 공기", "두 조각", "반 팩", "열댓 개").
_KOREAN_NUMBERS = {
    "한": 1, "하나": 1, "두": 2, "둘": 2, "세": 3, "셋": 3, "석": 3, "네": 4, "넷": 4, "넉": 4,
    "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10, "열댓": 12, "스무": 20,
    "반": 0.5,
}
_KOREAN_NUMBER_UNIT_RE = re.compile(
    r"(?<![가-힣])(?P<num>열댓|다섯|여섯|일곱|여덟|아홉|하나|스무|한|두|둘|세|셋|석|네|넷|넉|열|반)"
    rf"\s*(?P<unit>{_UNIT_WORDS})"
    r"(?=[\s,.!?~)]|$|만|씩|정도|쯤|이랑|랑|에|을|를|은|는|도|하고|과|와|이|가)"
)
# "한 열댓개" / "한 3개": 한 = "about", not one.
_APPROX_HAN_RE = re.compile(r"(?<![가-힣])한\s+(?=\d|열댓|다섯|여섯|일곱|여덟|아홉|스무|열)")
_FRACTION_UNIT_RE = re.compile(rf"(?P<a>\d+)\s*/\s*(?P<b>\d+)\s*(?P<unit>{_UNIT_WORDS})")
# "1개 반", "한 공기 반" -> +0.5
_AND_HALF_RE = re.compile(rf"(?P<n>\d+(?:\.\d+)?)\s*(?P<unit>{_UNIT_WORDS})\s*반(?![가-힣])")


def _fmt_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.3f}".rstrip("0")


def _normalize_number_words(text: str) -> str:
    """Rewrite Korean number words / fractions before a unit as digits for the regex parser."""
    text = _APPROX_HAN_RE.sub("", text)
    # Bare counting words without a unit ("햄버거 하나", "사과 둘") are pieces.
    text = re.sub(r"(?<![가-힣])(하나|둘|셋|넷)(?=[\s,.!?~)]|$|랑|이랑|하고|만|씩|도|를|을|는|은|와|과|에|만)",
                  lambda m: f"{_KOREAN_NUMBERS[m.group(1)]}개", text)
    text = _FRACTION_UNIT_RE.sub(
        lambda m: f"{_fmt_number(int(m['a']) / max(1, int(m['b'])))}{m['unit']}", text)
    text = _KOREAN_NUMBER_UNIT_RE.sub(
        lambda m: f"{_fmt_number(_KOREAN_NUMBERS[m['num']])}{m['unit']}", text)
    text = _AND_HALF_RE.sub(lambda m: f"{_fmt_number(float(m['n']) + 0.5)}{m['unit']}", text)
    return text

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


# Flavoured / branded drinks named by their category at the end ("용과망고 하이볼", "대선 샤인머스캣 소주",
# "필라이트 맥주"): the category's label values apply. Only drink categories: a food name ending in
# a food word is a different dish.
_DRINK_TABLE_KEYS = ("논알콜맥주", "생맥주", "맥주", "과일소주", "소주", "막걸리", "레드와인", "화이트와인", "와인",
                     "하이볼", "위스키", "양주", "청주", "사케", "고량주", "보드카", "소맥", "매실주", "복분자주")


def _curated_table_key(food_name: str | None) -> str | None:
    name = (food_name or "").strip()
    if name in _FOOD_PER_100G:
        return name
    compact = "".join(name.split())
    for key in _DRINK_TABLE_KEYS:
        if compact.endswith(key) and compact != key and key in _FOOD_PER_100G:
            return key
    return None


def _macros_from_table(food_name: str, amount: float | None, unit: str | None) -> dict[str, float | None]:
    nutrition = _FOOD_PER_100G.get(food_name.strip())
    if not nutrition:
        return {"protein_g": None, "carbs_g": None, "fat_g": None}
    if not amount:
        # 수량 미기재 → 1인분 ("콜라", "밥" 만 적어도 기본 한 잔 / 한 공기)
        grams = serving_grams(food_name.strip())
    else:
        grams = resolve_portion_grams(food_name.strip(), amount, unit)
    if grams is None:
        return {"protein_g": None, "carbs_g": None, "fat_g": None}
    factor = grams / 100
    macros = {
        "protein_g": _r1(nutrition["protein_g"] * factor),
        "carbs_g":   _r1(nutrition["carbs_g"]   * factor),
        "fat_g":     _r1(nutrition["fat_g"]      * factor),
    }
    if nutrition.get("kcal") is not None:
        macros["kcal"] = _r1(nutrition["kcal"] * factor)
    return macros


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
    "만", "씩", "정도", "쯤",
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

    text_for_food = _FILLER_RE.sub(" ", _strip_context_words(_normalize_number_words(text)))
    text_for_food = re.sub(r"\([^)]*\)", " ", text_for_food)          # "(라떼는 안 마심)"
    text_for_food = re.sub(r"\S+\s*말고\s+", " ", text_for_food)        # "라떼 말고 아인슈페너"
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


# When the portion is a blind guess (generic "개 = 100 g", or a head-noun suffix such as
# 잡채밥 -> 밥 = one rice bowl), the DB density x guessed grams can be several times off
# (약과 3개 = 1197 kcal, 잡채밥 = 315 kcal). The LLM estimated the whole portion it read; when
# the two disagree by more than this factor, keep the DB density but the LLM's portion size.
_PORTION_DISAGREEMENT = 2.0
_MIN_LLM_PORTION_KCAL = 30.0
_PIECE_COUNT_UNITS = frozenset({"", "개", "마리", "알", "piece", "pieces"})


def _reconcile_inferred_portion(item: DietParseItem, update: dict) -> dict:
    llm_kcal = _macro_kcal(item.protein_g, item.carbs_g, item.fat_g)
    db_kcal = update.get("kcal")
    if db_kcal is None:
        db_kcal = _macro_kcal(update.get("protein_g"), update.get("carbs_g"), update.get("fat_g"))
    if not db_kcal or llm_kcal < _MIN_LLM_PORTION_KCAL:
        return update
    amount = update.get("amount", item.amount)
    confidence = portion_confidence(item.food_name, amount, update.get("unit", item.unit))
    ratio = db_kcal / llm_kcal
    unit = (update.get("unit", item.unit) or "").strip().lower()
    if confidence == "default" and unit in _PIECE_COUNT_UNITS:
        # nothing known about this food's piece size ("약과 1개" is not 100 g): either direction.
        # Serving-like defaults (인분 200 g, 조각, 잔, 그릇) are kept: they are usually close.
        if 1 / _PORTION_DISAGREEMENT <= ratio <= _PORTION_DISAGREEMENT:
            return update
    else:
        return update
    factor = llm_kcal / db_kcal
    logger.info("[diet portion] inferred portion rescaled to the LLM portion (x%.2f)", factor)
    scaled = dict(update)
    for key in ("protein_g", "carbs_g", "fat_g", "kcal"):
        if scaled.get(key) is not None:
            scaled[key] = _r1(scaled[key] * factor)
    return scaled


def _enrich_with_trimmed_names(parsed: ParsedDietLog) -> ParsedDietLog:
    """Deterministic names can carry leading chatter ("부산 와서 대선", "더위사냥 반 갈라서"):
    when the full name matches nothing, retry with shorter word spans (last words first, then
    first words) and keep the first span that the DB / tables know."""
    enriched = _enrich_macros(parsed).items
    out = []
    for original, item in zip(parsed.items, enriched):
        words = (original.food_name or "").split()
        if _has_nutrition(item) or len(words) < 2:
            out.append(item)
            continue
        spans = [" ".join(words[i:]) for i in range(1, len(words))] + [" ".join(words[:j]) for j in range(len(words) - 1, 0, -1)]
        for span in spans:
            # only exact table / product hits: a vector-DB neighbour of a chatter word ("화이팅")
            # or a one-product line ("칼로리 …") is not a food the user logged
            if span.strip() not in _FOOD_PER_100G and product_nutrition(
                    span, original.amount, original.unit, allow_single_line=False) is None:
                continue
            retry = _enrich_macros(ParsedDietLog(items=[original.model_copy(update={"food_name": span})])).items[0]
            if _has_nutrition(retry):
                item = retry
                break
        out.append(item)
    return ParsedDietLog(items=out)


_JOINER_RE = re.compile(r"\d|이랑$|랑$|하고$|와$|과$|에$|에다$|및$|그리고$|먹고$|마시고$")


_SET_MENU_RE = re.compile(r"^(?P<main>.*(?:버거|와퍼|햄버거))\s*(?:세트|셋트|set|SET)$")
_HALF_HALF_CHICKEN_RE = re.compile(r"(후라이드|프라이드|양념|간장)\s*반\s*(양념|후라이드|프라이드|간장)\s*반|반반\s*치킨|치킨\s*반반")
_MEAL_SET_RE = re.compile(r"^(?P<main>.+?)\s*(?:정식|백반|한상|定食)$")
_SET_SIDE_WORDS = ("감자튀김", "후렌치", "프렌치", "프라이", "콜라", "사이다", "음료", "제로", "웨지")


# "두부김치 하나 시킴", "팔보채 시켜서", "간짜장 하나요": an ordered dish is one restaurant serving,
# not 100 g / one generic piece.
_ORDER_RE = re.compile(r"시켜|시킴|시켰|주문|배달|포장|하나요|1인분\s*(?:시|주)")


def _apply_ordered_dish_serving(parsed: ParsedDietLog, user_text: str) -> ParsedDietLog:
    if not _ORDER_RE.search(user_text or ""):
        return parsed
    literal_hundred = bool(_PLACEHOLDER_AMOUNT_RE.search(user_text or ""))
    out = []
    for item in parsed.items:
        unit = (item.unit or "").strip()
        # an amount the user wrote ("100g") always wins; only the LLM's placeholder 100 g is replaced
        placeholder = unit in ("g", "ml") and item.amount in (100, 100.0) and not literal_hundred
        bare = item.amount is None or placeholder or (unit in ("", "개") and item.amount in (1, 1.0))
        check_unit = "개" if (placeholder or not unit) else unit
        if bare and portion_confidence(item.food_name, 1, check_unit) == "default" and unit not in ("캔", "병", "봉"):
            item = item.model_copy(update={"amount": 1.0, "unit": "인분"})
        out.append(item)
    return ParsedDietLog(items=out)


def _apply_half_half_text(parsed: ParsedDietLog, user_text: str) -> ParsedDietLog:
    """"치킨 반반 시켜서" with the LLM item just "치킨 1인분": the order is one whole chicken."""
    if not _HALF_HALF_CHICKEN_RE.search(user_text or "") and "반반" not in (user_text or ""):
        return parsed
    out = []
    for item in parsed.items:
        name = item.food_name or ""
        if ("치킨" in name or _HALF_HALF_CHICKEN_RE.search(name)) and (item.unit or "") != "마리":
            item = item.model_copy(update={"food_name": "치킨", "amount": 1.0, "unit": "마리",
                                           "protein_g": None, "carbs_g": None, "fat_g": None, "kcal": None})
        out.append(item)
    return ParsedDietLog(items=out)


def _expand_set_menus(parsed: ParsedDietLog) -> ParsedDietLog:
    """"1955버거 세트": a fast-food set is the burger + a regular fries + a regular soft drink.
    A set with its own label row (빅맥 세트 in the curated table) stays as it is."""
    items = list(parsed.items)
    out = []
    has_side = any(any(w in (i.food_name or "") for w in _SET_SIDE_WORDS) for i in items)
    has_rice = any("밥" in (i.food_name or "") for i in items)
    for item in items:
        if _HALF_HALF_CHICKEN_RE.search(item.food_name or "") or (
                "반반" in (item.food_name or "") + " " + (item.unit or "") and "치킨" in (item.food_name or "")):
            # "후라이드 반 양념 반", "치킨 반반": one whole chicken, half of each flavour
            out.append(item.model_copy(update={"food_name": "치킨", "amount": item.amount if item.unit == "마리" and item.amount else 1.0,
                                               "unit": "마리", "protein_g": None, "carbs_g": None, "fat_g": None, "kcal": None}))
            continue
        meal = _MEAL_SET_RE.match((item.food_name or "").strip())
        if meal and meal.group("main").strip():
            # "생선까스 정식", "삼치구이 백반": the main dish (its own serving) + one bowl of rice;
            # the main dish's density must not be applied to the whole tray
            base = {"meal_type": item.meal_type, "time_of_day": item.time_of_day,
                    "protein_g": None, "carbs_g": None, "fat_g": None, "kcal": None}
            out.append(item.model_copy(update={**base, "food_name": meal.group("main").strip(), "amount": 1.0, "unit": "인분"}))
            if not has_rice:
                out.append(DietParseItem(food_name="쌀밥", amount=1.0, unit="공기", **base))
                has_rice = True
            continue
        m = _SET_MENU_RE.match((item.food_name or "").strip())
        if not m or product_nutrition(item.food_name, 1, "개", allow_single_line=False) is not None:
            out.append(item)
            continue
        count = item.amount if item.amount and item.unit not in _GRAM_UNITS and item.amount <= 5 else 1.0
        base = {"meal_type": item.meal_type, "time_of_day": item.time_of_day,
                "protein_g": None, "carbs_g": None, "fat_g": None, "kcal": None}
        out.append(item.model_copy(update={**base, "food_name": m.group("main").strip(), "amount": count, "unit": "개"}))
        if not has_side:
            out.append(DietParseItem(food_name="감자튀김", amount=count, unit="인분", **base))
            out.append(DietParseItem(food_name="콜라", amount=count, unit="잔", **base))
    return ParsedDietLog(items=out)


# Spoken self-corrections: "스콘 하나랑 … 아 아니다 스콘 두 개", "율무죽 한 그릇 아 반 그릇만": the last
# amount said for the item after the correction marker is the one eaten.
_CORRECTION_RE = re.compile(r"(?:아\s*)?아니(?:다|야|지|고)?\b|아\s+(?=반|\d|한|두|세|네)|정정|수정")


def _apply_spoken_correction(parsed: ParsedDietLog, user_text: str) -> ParsedDietLog:
    text = user_text or ""
    m = _CORRECTION_RE.search(text)
    if not m:
        return parsed
    tail = _normalize_number_words(text[m.end():])
    tail_items = _deterministic_diet_parse(tail).items
    out = []
    for item in parsed.items:
        name = "".join((item.food_name or "").split())
        hit = next((t for t in reversed(tail_items) if t.amount is not None and (
            name and ("".join(t.food_name.split()) in name or name in "".join(t.food_name.split())))), None)
        if hit is None and len(parsed.items) == 1:
            nums = list(_AMOUNT_UNIT_RE.finditer(tail)) or []
            half = re.search(r"반\s*(공기|그릇|접시|판|마리|인분|개|병|캔)", tail)
            if half:
                hit = DietParseItem(food_name=item.food_name, amount=0.5, unit=half.group(1))
            elif nums:
                last = nums[-1]
                hit = DietParseItem(food_name=item.food_name, amount=float(last.group("amount")), unit=last.group("unit"))
        if hit is not None and (hit.amount != item.amount or (hit.unit or item.unit) != item.unit):
            item = item.model_copy(update={"amount": hit.amount, "unit": hit.unit or item.unit,
                                           "protein_g": None, "carbs_g": None, "fat_g": None, "kcal": None})
        out.append(item)
    return ParsedDietLog(items=out)


def _restore_brand_prefix(parsed: ParsedDietLog, user_text: str) -> ParsedDietLog:
    """The LLM often drops the brand word the user wrote ("설레임 밀크쉐이크" -> 밀크쉐이크,
    "새로 다래" -> 다래). When the words right before the item name in the text form a known
    product / curated entry together with it, log that fuller name."""
    words = re.findall(r"[가-힣A-Za-z0-9]+", user_text or "")
    fixed = []
    for item in parsed.items:
        name = "".join((item.food_name or "").split())
        best = None
        if name:
            for idx in range(len(words)):
                joined = ""
                for end in range(idx, min(idx + 3, len(words))):
                    if end > idx and _JOINER_RE.search(words[end - 1]):
                        break  # "김밥 두 줄이랑 컵라면": another item, not a brand word
                    joined += words[end]
                    if joined == name or not joined.endswith(name) or len(joined) <= len(name):
                        continue
                    phrase = " ".join(words[idx:end + 1])
                    product = find_processed_product(phrase, allow_single_line=False)
                    exact = phrase.strip() in _FOOD_PER_100G or (
                        product is not None and product.unit_g < 2000)  # not a catering pack
                    if exact and (best is None or len(phrase) > len(best)):
                        best = phrase
        fixed.append(item.model_copy(update={"food_name": best}) if best else item)
    # "서울우유 커피" restored from 서울우유 + 커피 split items: drop the parts that only occur inside it
    text = "".join((user_text or "").split())
    keys = ["".join((i.food_name or "").split()) for i in fixed]
    kept = []
    for i, key in zip(fixed, keys):
        longer = [k for k in keys if k != key and key and key in k]
        if longer and all(text.count(key) <= text.count(k) for k in longer):
            continue
        kept.append(i)
    return ParsedDietLog(items=kept)


# Fried-chicken franchise menus are not in the generic DB under their brand names
# ("bhc 고추바사삭", "BBQ 황올", "교촌 허니콤보"): their macros are a fried chicken's.
_CHICKEN_BRAND_RE = re.compile(
    r"(bhc|bbq|교촌|굽네|네네|처갓집|푸라닭|호식이|멕시카나|페리카나|노랑통닭|자담|60계|가마치|맘스터치\s*치킨|"
    r"뿌링클|고추바사삭|황금올리브|황올|허니콤보|레드콤보|볼케이노|맛초킹|양념반\s*후라이드반)",
    re.IGNORECASE,
)


def _lookup_name(food_name: str) -> str:
    name = food_name or ""
    if _CHICKEN_BRAND_RE.search(name) and "버거" not in name:
        return "양념치킨" if "양념" in name or "레드" in name else "후라이드치킨"
    return name


def _enrich_macros(parsed: ParsedDietLog) -> ParsedDietLog:
    shown = [item.food_name for item in parsed.items]
    looked_up = [item.model_copy(update={"food_name": _lookup_name(item.food_name)}) for item in parsed.items]
    enriched = _enrich_macros_by_name(ParsedDietLog(items=looked_up)).items
    return ParsedDietLog(items=[i.model_copy(update={"food_name": n}) for i, n in zip(enriched, shown)])


def _enrich_macros_by_name(parsed: ParsedDietLog) -> ParsedDietLog:
    engine = get_food_engine() if get_food_engine is not None else None
    enriched = []
    for item in parsed.items:
        # Branded / packaged products (신라면, 바나나맛우유 …): per-unit label values first.
        product = product_nutrition(item.food_name, item.amount, item.unit)
        if product is not None:
            update = {k: product[k] for k in ("protein_g", "carbs_g", "fat_g", "kcal")}
            update["source"] = "db"
            if item.amount is None:
                update.update({"amount": 1.0, "unit": item.unit or "개"})
            enriched.append(item.model_copy(update=update))
            continue
        # Curated plain staples (닭가슴살, 쌀밥, 계란 …) beat vector search, which can land on a
        # composite row such as "샐러드 닭가슴살" (14g protein per 100g instead of 31g).
        table_key = _curated_table_key(item.food_name)
        if table_key is not None:
            macros = _macros_from_table(table_key, item.amount, item.unit)
            if macros["protein_g"] is not None:
                update = {**macros, "source": "db"}
                if item.amount is None:
                    update.update({"amount": 1.0, "unit": "인분"})
                enriched.append(item.model_copy(update=update))
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
            if os.getenv("DIET_LLM_PORTION_RESCALE", "0") == "1":
                update = _reconcile_inferred_portion(item, update)  # legacy: LLM-estimated portions
            enriched.append(item.model_copy(update=update))
            continue
        # No DB row for this exact food: the median density of its head-noun category
        # ("X볶음" -> 볶음 rows, "X라떼" -> 라떼 rows) x deterministic grams, before any LLM number.
        category = category_nutrition(item.food_name, item.amount, item.unit) if engine is not None else None
        if category is not None:
            update = {k: category[k] for k in ("protein_g", "carbs_g", "fat_g", "kcal")}
            update["source"] = "db"
            if item.amount is None:
                update.update({"amount": category["amount"], "unit": category["unit"]})
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


_SIZE_WORDS_RE = re.compile(r"^(?:작은|큰|중간|큰거|작은거|라지|레귤러|톨|그란데|벤티|곱빼기|특대|점보|대자|중자|소자)\s*")
_UNIT_WITH_NUMBER_RE = re.compile(r"^(?P<n>\d+(?:\.\d+)?)\s*(?P<unit>[^\d\s][^\d]*)$")
_COUNT_UNITS = frozenset(_UNIT_WORDS.split("|")) - {"g", "그램", "ml"}


_DISH_SIZE_UNITS = frozenset({"곱빼기", "특대", "점보", "대자", "중자", "소자"})
_CUP_SIZE_UNITS = frozenset({"라지", "레귤러", "톨", "그란데", "벤티"})
_DRINK_NAME_RE = re.compile(
    r"(커피|라떼|아메리카노|티$|차$|에이드|주스|스무디|쉐이크|프라푸치노|아샷추|음료|콜드브루|모카|마끼아또|버블티|밀크티|콜라|사이다|맥주)"
)
_UNIT_TYPOS = {"조깍": "조각", "조각이": "조각", "그륵": "그릇", "그룻": "그릇", "공귀": "공기", "게": "개", "갸": "개",
               "봉다리": "봉지", "잔씩": "잔", "개씩": "개"}
_HALF_UNIT_RE = re.compile(r"^반\s*(근|공기|그릇|접시|판|마리|인분|병|캔|개|봉지|팩|컵|잔)$")
_ENGLISH_UNITS = {
    "slice": "조각", "slices": "조각", "piece": "개", "pieces": "개", "pc": "개", "pcs": "개",
    "cup": "컵", "cups": "컵", "can": "캔", "cans": "캔", "bottle": "병", "bottles": "병",
    "glass": "잔", "glasses": "잔", "bowl": "그릇", "bowls": "그릇", "pack": "팩", "packs": "팩",
    "plate": "접시", "plates": "접시", "serving": "인분", "servings": "인분", "portion": "인분",
}


# The LLM sometimes returns the whole clause as the food name ("훠궈 먹으러 갔다 마라탕보다 매움 🌶️",
# "라조기 처음"): keep the food words, drop verbs, comparisons, time words and emoji.
_NAME_NOISE_WORDS = frozenset({"단품", "처음", "다음날", "오늘", "어제", "진짜", "완전", "너무", "엄청", "그냥", "막", "또"})
_NAME_NOISE_RE = re.compile(r"^(?:먹|마시|마셨|갔|왔|시켜|시킴)|(?:갔다|왔다|었다|았다|했다|먹음|마심|매움|맛있|보다|으러|하러|는데|어서|아서)$")


def _clean_llm_food_name(name: str) -> str:
    words = [w for w in re.findall(r"[가-힣A-Za-z0-9]+", name or "")]
    if len(words) < 2:
        return name
    kept = [w for w in words if w not in _NAME_NOISE_WORDS and w not in _CHATTER_WORDS and not _NAME_NOISE_RE.search(w)]
    if not kept or kept == words:
        return name
    return " ".join(kept)


_NAME_COUNT_RE = re.compile(r"\s*(반|한|두|세|\d+)\s*(마리|판|그릇|접시|인분|개|조각)$")


def _normalize_llm_units(parsed: ParsedDietLog) -> ParsedDietLog:
    """LLM 이 수량을 단위 문자열에 넣은 경우("한 컵", "반공기", "두 조각", "30cm") 분리한다."""
    fixed = []
    for item in parsed.items:
        cleaned = _clean_llm_food_name(item.food_name or "")
        if cleaned != item.food_name:
            item = item.model_copy(update={"food_name": cleaned})
        m = _NAME_COUNT_RE.search(item.food_name or "")
        if m and m.start() > 0:
            item = item.model_copy(update={"food_name": item.food_name[: m.start()].strip()})
        unit = (item.unit or "").strip()
        amount = item.amount
        if unit:
            # "곱빼기" / "라지" as the whole unit is a size of one serving (the size factor is
            # applied from the text later), not an unknown unit worth 100 g
            unit = _SIZE_WORDS_RE.sub("", unit).strip() or (
                "인분" if unit in _DISH_SIZE_UNITS
                else ("잔" if _DRINK_NAME_RE.search(item.food_name or "") else "인분") if unit in _CUP_SIZE_UNITS
                else unit)
            unit = _ENGLISH_UNITS.get(unit.lower(), unit)
            unit = _UNIT_TYPOS.get(unit, unit)
            half = _HALF_UNIT_RE.match(unit)
            if half and (amount is None or amount in (1, 1.0)):
                amount, unit = 0.5, half.group(1)  # "반근", "반공기", "반그릇" as the unit
            m = _UNIT_WITH_NUMBER_RE.match(_normalize_number_words(unit).replace(" ", ""))
            if m:
                amount, unit = float(m["n"]), m["unit"].strip()
            elif amount is None and unit in _COUNT_UNITS:
                amount = 1.0  # "쌀밥 공기", "녹차 잔": a unit without a number is one of it
        if amount != item.amount or unit != (item.unit or ""):
            item = item.model_copy(update={"amount": amount, "unit": unit or None})
        fixed.append(item)
    return ParsedDietLog(items=fixed)


_PLACEHOLDER_AMOUNT_RE = re.compile(r"100|백\s*(?:g|그램|ml|미리)")


def _drop_placeholder_amounts(parsed: ParsedDietLog, user_text: str) -> ParsedDietLog:
    """LLM 이 수량 없는 입력("된장찌개 먹었어")에 100 g/ml 를 채운 경우 수량 미기재로 되돌린다.

    그래야 enrichment 가 1인분 기준으로 계산한다. LLM 매크로(100g 기준)는 1인분으로 재환산."""
    if _PLACEHOLDER_AMOUNT_RE.search(user_text or ""):
        return parsed
    fixed = []
    for item in parsed.items:
        unit = (item.unit or "").strip().lower()
        if item.amount == 100 and unit in {"g", "ml", "그램"}:
            serving = serving_grams(item.food_name)
            update: dict = {"amount": None, "unit": None}
            for key in ("protein_g", "carbs_g", "fat_g", "kcal"):
                value = getattr(item, key)
                update[key] = _r1(value * serving / 100.0) if (value is not None and serving) else value
            item = item.model_copy(update=update)
        fixed.append(item)
    return ParsedDietLog(items=fixed)


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
        name_key = "".join(item.food_name.split())
        source = literal.get(name_key)
        if source is None and len(name_key) >= 2:
            # "중국집 고량주 세 잔": the deterministic name keeps leading words
            source = next((v for k, v in literal.items() if k.endswith(name_key)), None)
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

def _food_engine_available() -> bool:
    engine = get_food_engine() if get_food_engine is not None else None
    return bool(engine is not None and getattr(engine, "available", False))


# "둘이서 / 셋이서 / 반띵 / 나눠 먹음": the logged amount was shared.
_SHARE_RE = re.compile(
    r"(?<![가-힣0-9])(둘|두\s*명|2\s*명|셋|세\s*명|3\s*명|넷|네\s*명|4\s*명)이(?:서)?(?:\s*(?:나눠|노나|같이|먹))?"
    r"|반띵|반반\s*나눠|반씩\s*나눠"
    r"|(?:와이프|아내|남편|친구|동생|언니|오빠|누나|형|엄마|아빠|애인|여친|남친)(?:랑|하고|와|과)\s*(?:같이\s*)?나눠"
)
_SHARE_DIVISOR = {"둘": 2, "두": 2, "2": 2, "셋": 3, "세": 3, "3": 3, "넷": 4, "네": 4, "4": 4}
# Size words right next to a food ("짜장면 곱빼기", "라지 감자튀김", "바닐라라떼 그란데").
_SIZE_FACTORS = {
    "곱빼기": 1.5, "특대": 1.5, "특": 1.3, "라지": 1.4, "large": 1.4, "점보": 1.6,
    "그란데": 473 / 355, "벤티": 591 / 355, "스몰": 0.75, "숏": 0.67, "큰사이즈": 1.4, "빅사이즈": 1.4,
    # 족발·보쌈·찜·탕 포장 크기 (1인분 대비)
    "대자": 3.5, "중자": 2.5, "소자": 1.8, "小자": 1.8, "大자": 3.5, "中자": 2.5,
}
_SIZE_WORDS = "|".join(sorted(_SIZE_FACTORS, key=len, reverse=True))


def _share_divisor(user_text: str) -> float:
    m = _SHARE_RE.search(user_text or "")
    if not m:
        return 1.0
    if m.group(1) is None:
        return 2.0  # 반띵 / 반씩 나눠 / 와이프랑 나눠: two people
    head = m.group(1).replace(" ", "")[0]
    return float(_SHARE_DIVISOR.get(head, 1))


def _size_factor(food_name: str, user_text: str) -> float:
    name = re.escape("".join((food_name or "").split()))
    text = "".join((user_text or "").split())
    if not name or not text:
        return 1.0
    m = re.search(rf"{name}(?:으로|로)?({_SIZE_WORDS})|({_SIZE_WORDS})(?:사이즈)?{name}", text, re.IGNORECASE)
    if not m:
        return 1.0
    word = (m.group(1) or m.group(2) or "").lower()
    return _SIZE_FACTORS.get(word, 1.0)


def _scale_item(item: DietParseItem, factor: float) -> DietParseItem:
    if abs(factor - 1.0) < 1e-9:
        return item
    update = {k: _r1(getattr(item, k) * factor) for k in ("protein_g", "carbs_g", "fat_g", "kcal")
              if getattr(item, k) is not None}
    return item.model_copy(update=update)


# "반은 남김", "반만 먹음", "절반만 마심": the logged item was only half eaten.
_LEFTOVER_RE = re.compile(
    r"(?:반|절반)\s*(?:은|이상|정도|넘게)?\s*(?:남김|남겼|남기고|남기|버림|버렸|버리고)"
    r"|(?:반|절반)\s*만\s*(?:먹|마시|마심|먹음|먹었)"
)
_FRACTION_EATEN_RE = re.compile(r"(\d)\s*/\s*(\d)\s*만|(\d)\s*분의\s*(\d)\s*만")
_GAVE_ONE_AWAY_RE = re.compile(r"(?:하나|한\s*개|1\s*개)는\s*(?:\S+\s*){0,2}?(?:줌|줬|주고|줘|드림|드렸)")
_OTHER_ATE_HALF_RE = re.compile(
    r"반(?:은|이나)?\s*(?:동생|형|언니|오빠|누나|남편|아내|와이프|친구|엄마|아빠|애|아이|딸|아들)\S{0,2}\s*(?:이|가)?\s*(?:먹|줬|주)")
# "라면 끓이려다 그냥 굶음", "치킨 시키려다 참음": nothing (or only what follows) was eaten.
_NOTHING_EATEN_RE = re.compile(r"굶|안\s*먹었|안\s*먹음|못\s*먹었|참았|참음|패스했|스킵")


def _leftover_factor(item: DietParseItem, items: list[DietParseItem], user_text: str) -> float:
    text = user_text or ""
    if len(items) == 1:
        frac = _FRACTION_EATEN_RE.search(text)
        if frac:
            a, b = (int(frac.group(1)), int(frac.group(2))) if frac.group(1) else (int(frac.group(4)), int(frac.group(3)))
            if 0 < a < b:
                return a / b  # "1/3만 먹고", "3분의 1만 먹음"
        try:
            count = float(item.amount) if item.amount is not None else 0.0
        except (TypeError, ValueError):
            count = 0.0
        if count >= 2 and _GAVE_ONE_AWAY_RE.search(text):
            return (count - 1) / count  # "2개 끓여서 하나는 남편 줌"
        if _OTHER_ATE_HALF_RE.search(text):
            return 0.5  # "반은 동생이 먹음"
    m = _LEFTOVER_RE.search(text)
    if not m:
        return 1.0
    try:
        if item.amount is not None and float(item.amount) < 1.0:
            return 1.0  # the parser already logged the half ("반 그릇")
    except (TypeError, ValueError):
        pass
    if len(items) == 1:
        return 0.5
    # several items: the one named last before the phrase
    before = "".join((user_text or "")[: m.start()].split())
    positions = {id(i): before.rfind("".join((i.food_name or "").split())) for i in items}
    target = max(items, key=lambda i: positions[id(i)])
    return 0.5 if target is item and positions[id(item)] >= 0 else 1.0


def _apply_portion_context(parsed: ParsedDietLog, user_text: str) -> ParsedDietLog:
    """Size words scale the item next to them; sharing ("둘이서") divides every item;
    "반은 남김" halves the item it follows."""
    divisor = _share_divisor(user_text)
    items = [
        _scale_item(i, _size_factor(i.food_name, user_text) * _leftover_factor(i, parsed.items, user_text) / divisor)
        for i in parsed.items
    ]
    return ParsedDietLog(items=items)


def _has_nutrition(item: DietParseItem) -> bool:
    return item.source is not None or any(
        getattr(item, key) is not None for key in ("protein_g", "carbs_g", "fat_g", "kcal")
    )


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

        if not parsed.items and _NOTHING_EATEN_RE.search(user_text):
            return ParsedDietLog(items=[]).model_dump_json()  # "그냥 굶음": nothing to log
        if not parsed.items:
            # LLM 이 잡담 섞인 입력에서 음식을 놓친 경우 → 결정론 추출로 보강하되, DB/식품표에
            # 실제로 매칭되는 항목만 남긴다 ("점심 굶음", "ㅋㅋㅋ" 가 음식으로 기록되지 않도록).
            enriched = _apply_portion_context(
                _enrich_with_trimmed_names(_deterministic_diet_parse(user_text)), user_text)
            if not _food_engine_available():
                return enriched.model_dump_json()  # nothing to verify against: keep old behaviour
            return ParsedDietLog(items=[i for i in enriched.items if _has_nutrition(i)]).model_dump_json()
        parsed = _normalize_llm_units(parsed)
        parsed = _drop_placeholder_amounts(parsed, user_text)
        parsed = _reconcile_amounts_with_text(parsed, user_text)
        parsed = _apply_spoken_correction(parsed, user_text)
        parsed = _restore_brand_prefix(parsed, user_text)
        parsed = _apply_ordered_dish_serving(parsed, user_text)
        parsed = _apply_half_half_text(parsed, user_text)
        parsed = _expand_set_menus(parsed)
        return _apply_portion_context(_enrich_macros(parsed), user_text).model_dump_json()
    except Exception as exc:
        logger.warning("[LLM diet parsing fallback] %s", sanitize_exception_for_log(exc))
    # LLM down: deterministic parse still gets DB / curated macros (previously none at all).
    parsed = _deterministic_diet_parse(user_text)
    try:
        parsed = _enrich_macros(parsed)
    except Exception as exc:  # enrichment must never break the fallback
        logger.warning("[diet fallback enrich failed] %s", sanitize_exception_for_log(exc))
    return parsed.model_dump_json()

"""Portion unit -> gram/ml conversion (single source for food_engine + diet_parser).

All values are estimated defaults. 1 ml is treated as 1 g (DB rows with
basis_unit=ml are scaled the same way).

Lookup order for (food, unit):
  1. per-food override by exact / compact / alias name ("계란" == "달걀")
  2. per-food override by head-noun suffix ("양념치킨" -> "치킨")
  3. generic per-unit default (never silently 100 g regardless of count)

An empty unit with a bare count ("계란 2") is treated as pieces for countable
foods, and as grams otherwise when the number is large (> _BARE_COUNT_MAX).
"""
from __future__ import annotations

import re

GRAM_UNITS = {"g", "gram", "grams", "그램", "ml", "밀리리터", "cc", "mL"}
_KG_UNITS = {"kg", "킬로", "킬로그램"}
# 고기 "한 근" = 600 g (돼지·소 공통 관례)
_GEUN_UNITS = {"근"}
_LITER_UNITS = {"l", "리터"}
# Length sizes ("샌드위치 15cm", "30센치"): 15 cm is one standard piece, 30 cm two.
_LENGTH_UNITS = {"cm", "센치", "센티", "센티미터"}
_STANDARD_PIECE_CM = 15.0

# Generic per-unit defaults (used when no per-food override exists).
UNIT_DEFAULT_GRAMS: dict[str, float] = {
    "개": 100.0,
    "조각": 100.0,
    "캔": 355.0,
    "병": 500.0,
    "그릇": 400.0,
    "인분": 200.0,
    "봉지": 100.0,
    "팩": 200.0,
    "컵": 200.0,
    "잔": 250.0,
    "장": 20.0,
    "모": 300.0,
    "공기": 210.0,
    "스쿱": 30.0,
    "쪽": 10.0,
    "줄": 230.0,
    "알": 10.0,
    "접시": 300.0,
    "숟가락": 10.0, "숟갈": 10.0, "스푼": 10.0, "큰술": 12.0, "작은술": 4.0, "티스푼": 4.0,
}

# Per-food overrides. Keys are compact (no whitespace) names.
# 라면: DB "라면" rows are the cooked dish (noodles + broth, ~82 kcal/100g),
# so one pack (120 g dry + water) is expressed as ~600 g cooked -> ~500 kcal.
FOOD_UNIT_GRAMS: dict[str, dict[str, float]] = {
    "계란": {"개": 50.0, "알": 50.0},
    "달걀": {"개": 50.0, "알": 50.0},
    "메추리알": {"개": 10.0, "알": 10.0},
    "사과": {"개": 200.0},
    "배": {"개": 400.0, "쪽": 400.0},  # "배 반쪽" = 0.5 쪽 = half a pear
    "귤": {"개": 80.0},
    "오렌지": {"개": 180.0},
    "키위": {"개": 90.0},
    "복숭아": {"개": 200.0},
    "토마토": {"개": 150.0},
    "방울토마토": {"개": 15.0, "알": 15.0},
    "고구마": {"개": 150.0},
    "라면": {"개": 600.0, "봉지": 600.0, "그릇": 600.0, "인분": 600.0},
    "컵라면": {"개": 300.0},
    "치킨": {"조각": 100.0, "개": 100.0, "마리": 800.0},  # cooked delivery chicken, edible ~800 g
    "피자": {"조각": 120.0},
    "케이크": {"조각": 100.0},
    "닭가슴살": {"개": 100.0, "팩": 100.0, "덩이": 100.0},
    "콜라": {"캔": 355.0, "병": 500.0, "잔": 250.0, "컵": 200.0},
    "사이다": {"캔": 355.0, "병": 500.0, "잔": 250.0, "컵": 200.0},
    "맥주": {"캔": 355.0, "병": 500.0, "잔": 500.0},
    "소주": {"병": 360.0, "잔": 50.0},
    "막걸리": {"병": 750.0, "잔": 200.0, "사발": 300.0},
    "와인": {"잔": 150.0, "병": 750.0},
    "파워에이드": {"병": 600.0, "캔": 355.0, "개": 600.0},
    "게토레이": {"병": 600.0, "캔": 355.0, "개": 600.0},
    "몬스터": {"캔": 355.0, "개": 355.0},
    "몬스터에너지": {"캔": 355.0, "개": 355.0},
    "몬스터울트라": {"캔": 355.0, "개": 355.0},
    "레드불": {"캔": 250.0, "개": 250.0},
    "한라봉": {"개": 250.0},
    "천혜향": {"개": 220.0},
    # whole produce eaten as "N개" (edible part, common market size)
    "용과": {"개": 350.0}, "자몽": {"개": 300.0}, "석류": {"개": 250.0}, "당근": {"개": 150.0},
    "양파": {"개": 200.0}, "파프리카": {"개": 150.0}, "피망": {"개": 100.0}, "레몬": {"개": 100.0},
    "라임": {"개": 70.0}, "자두": {"개": 70.0}, "천도복숭아": {"개": 150.0}, "살구": {"개": 40.0},
    "무화과": {"개": 50.0}, "홍시": {"개": 180.0}, "연시": {"개": 180.0}, "곶감": {"개": 40.0},
    "대추": {"개": 5.0, "알": 5.0}, "체리": {"개": 8.0, "알": 8.0}, "방울토마토": {"개": 15.0, "알": 15.0},
    "블루베리": {"알": 1.5}, "밤": {"개": 15.0, "알": 15.0}, "애호박": {"개": 300.0},
    "가지": {"개": 150.0}, "브로콜리": {"개": 300.0, "송이": 300.0}, "옥수수": {"개": 150.0},
    "찰옥수수": {"개": 150.0}, "고추": {"개": 15.0}, "풋고추": {"개": 15.0}, "마늘": {"쪽": 5.0, "알": 5.0},
    "금귤": {"개": 15.0}, "멜론": {"개": 1500.0, "조각": 150.0},
    "청하": {"병": 300.0, "잔": 50.0}, "청주": {"병": 300.0, "잔": 50.0}, "사케": {"병": 300.0, "잔": 50.0},
    "정종": {"병": 300.0, "잔": 50.0}, "고량주": {"잔": 30.0, "병": 500.0}, "백주": {"잔": 30.0, "병": 500.0},
    "보드카": {"잔": 40.0, "샷": 30.0}, "데킬라": {"잔": 40.0, "샷": 30.0}, "럼": {"잔": 40.0, "샷": 30.0},
    "복분자주": {"잔": 50.0, "병": 375.0}, "매실주": {"잔": 50.0, "병": 375.0}, "과실주": {"잔": 50.0, "병": 375.0},
    "과일소주": {"병": 360.0, "잔": 50.0, "캔": 355.0},
    **{brand: {"병": 360.0, "잔": 50.0} for brand in (
        "참이슬", "처음처럼", "진로", "진로이즈백", "대선", "새로", "좋은데이", "한라산", "잎새주", "c1", "시원",
        "맛있는참", "이제우린")},
    **{brand: {"병": 360.0, "잔": 50.0, "캔": 355.0} for brand in (
        "자몽에이슬", "청포도에이슬", "자두에이슬", "새로다래", "새로오미자")},
    **{brand: {"캔": 355.0, "개": 355.0, "병": 355.0} for brand in ("순하리", "순하리레몬진", "레몬진", "자몽진", "순하리자몽진")},
    "크로플": {"개": 70.0},
    "오메가3": {"알": 1.0, "캡슐": 1.0, "개": 1.0},
    "오메가": {"알": 1.0, "캡슐": 1.0, "개": 1.0},
    "멀티비타민": {"알": 1.2, "정": 1.2, "개": 1.2},
    "종합비타민": {"알": 1.2, "정": 1.2, "개": 1.2},
    "위스키": {"잔": 40.0, "샷": 30.0, "병": 700.0},
    "양주": {"잔": 40.0, "샷": 30.0, "병": 700.0},
    "하이볼": {"잔": 350.0, "캔": 355.0},
    "소맥": {"잔": 200.0},
    "생맥": {"잔": 500.0},
    "생맥주": {"잔": 500.0},
    "우유": {"팩": 200.0, "개": 200.0, "컵": 200.0, "잔": 200.0},
    "아메리카노": {"잔": 355.0, "컵": 355.0, "개": 355.0},
    "두부": {"모": 300.0},
    "김밥": {"줄": 230.0},
    "만두": {"개": 30.0},
    "햄버거": {"개": 220.0},
    "샌드위치": {"개": 200.0},
    "식빵": {"장": 35.0, "쪽": 35.0},
    "치즈": {"장": 20.0},
    "김": {"장": 2.0},
    "요거트": {"개": 85.0},
    # A plain-rice "그릇" is one 공기 (210g), not the 400g soup-bowl default.
    "쌀밥": {"공기": 210.0, "그릇": 210.0, "개": 210.0},
    # Piece weights of small items (one piece, not one generic 100 g serving).
    "초밥": {"개": 30.0, "피스": 30.0},
    "유부초밥": {"개": 30.0},
    "교자": {"개": 35.0},
    "왕교자": {"개": 35.0},
    "마카롱": {"개": 25.0},
    "인절미": {"개": 20.0, "조각": 20.0},
    "떡": {"개": 25.0, "조각": 25.0, "줄": 100.0},
    # 전·부침개 "한 장" is a pan-sized pancake, not a sheet of seaweed (장 = 20 g default)
    "전": {"장": 200.0, "개": 40.0, "조각": 50.0}, "부침개": {"장": 250.0}, "빈대떡": {"장": 250.0},
    "지짐이": {"장": 200.0}, "팬케이크": {"장": 80.0, "개": 80.0}, "핫케이크": {"장": 80.0, "개": 80.0},
    "빵": {"조각": 35.0}, "찹쌀떡": {"개": 60.0}, "꿀떡": {"개": 15.0}, "호떡": {"개": 80.0}, "모찌": {"개": 50.0}, "소박이": {"개": 60.0},
    "월남쌈": {"개": 40.0}, "꼬치": {"개": 80.0},
    # piece weights of common bakery / traditional snacks (one item as sold)
    "붕어빵": {"개": 60.0, "마리": 60.0}, "약과": {"개": 40.0}, "송편": {"개": 30.0}, "경단": {"개": 15.0},
    "다쿠아즈": {"개": 40.0}, "마카롱": {"개": 20.0}, "마들렌": {"개": 30.0}, "휘낭시에": {"개": 40.0},
    "쿠키": {"개": 20.0}, "머핀": {"개": 100.0}, "스콘": {"개": 100.0}, "크루아상": {"개": 60.0},
    "크로와상": {"개": 60.0}, "베이글": {"개": 110.0}, "에그타르트": {"개": 60.0}, "꽈배기": {"개": 70.0},
    "찐빵": {"개": 90.0}, "호빵": {"개": 90.0}, "만쥬": {"개": 30.0}, "소시지": {"개": 50.0},
    "닭가슴살소시지": {"개": 100.0}, "닭가슴살 소시지": {"개": 100.0},
    # whole-chicken dishes ordered by the bird
    "닭": {"마리": 1000.0}, "백숙": {"마리": 1000.0}, "찜닭": {"마리": 1200.0},
    "아몬드": {"알": 1.2, "개": 1.2, "줌": 25.0},
    "땅콩": {"알": 1.0, "개": 1.0, "줌": 25.0},
    "호두": {"알": 4.0, "개": 4.0, "줌": 25.0},
    "달걀흰자": {"개": 33.0},
    "계란흰자": {"개": 33.0},
    "망고": {"개": 250.0},
    "오이": {"개": 200.0},
    "아보카도": {"개": 150.0},
    "크로와상": {"개": 60.0},
    "딸기": {"개": 15.0, "팩": 400.0},
    "블루베리": {"줌": 40.0, "컵": 150.0},
    "피자": {"조각": 120.0, "판": 950.0},
    "삼계탕": {"마리": 900.0, "그릇": 900.0},
    # Common piece / pack weights (식약처 1회 섭취참고량 · 일반 포장 기준).
    "새우": {"마리": 15.0, "개": 15.0},
    "너겟": {"조각": 17.0, "개": 17.0},
    "도넛": {"개": 60.0},
    "포도": {"송이": 400.0},
    "샤인머스캣": {"송이": 500.0, "알": 10.0},
    "수박": {"조각": 300.0},
    "멜론": {"조각": 150.0},
    "참외": {"개": 300.0},
    "두유": {"팩": 190.0, "개": 190.0},
    "오트밀": {"컵": 80.0},
    "버거": {"개": 190.0},
    "싸이버거": {"개": 230.0},
    "와퍼주니어": {"개": 180.0}, "와퍼쥬니어": {"개": 180.0},
    "와퍼": {"개": 280.0},
    "토스트": {"개": 40.0, "장": 40.0, "조각": 40.0},
    "라떼": {"잔": 355.0, "컵": 355.0},
    "버블티": {"잔": 500.0, "컵": 500.0},
    "김말이": {"개": 25.0},
    "새우튀김": {"마리": 25.0, "개": 25.0},
    "오징어튀김": {"개": 30.0},
    "파인애플": {"조각": 40.0},
    "복숭아": {"개": 200.0, "쪽": 200.0},
    "단감": {"개": 180.0},
    "감": {"개": 180.0},
    "감자": {"개": 130.0, "알": 130.0},
    "멜론": {"조각": 150.0, "통": 1500.0},
    "수박": {"조각": 300.0, "통": 5000.0},
    "단호박": {"통": 700.0, "개": 700.0},
    "바나나": {"개": 120.0, "송이": 720.0},
    "갈비": {"대": 100.0},
    "치즈스틱": {"개": 35.0},
    "삼각김밥": {"개": 110.0},
    "밥": {"공기": 210.0, "그릇": 210.0},
}

# One usual serving (g, ml treated as g) by dish-type head noun — restaurant / 1인분
# standard portions. The longest matching suffix wins ("떡국" beats "국").
SERVING_GRAMS_BY_SUFFIX: dict[str, float] = {
    # 국 · 찌개 · 탕
    "국": 350.0, "찌개": 400.0, "전골": 500.0, "탕": 700.0, "국밥": 700.0,
    "떡국": 600.0, "만두국": 600.0, "만둣국": 600.0, "육개장": 600.0, "해장국": 700.0,
    "순대국": 700.0, "순댓국": 700.0, "삼계탕": 900.0, "마라탕": 600.0,
    # 면
    "면": 600.0, "국수": 600.0, "우동": 600.0, "라면": 550.0, "라멘": 600.0, "소바": 400.0,
    "짬뽕": 800.0, "짜장면": 650.0, "자장면": 650.0, "냉면": 700.0, "칼국수": 750.0,
    "쌀국수": 700.0, "쫄면": 400.0, "파스타": 350.0, "스파게티": 350.0,
    # 밥 요리
    "비빔밥": 450.0, "덮밥": 450.0, "볶음밥": 350.0, "카레라이스": 450.0, "카레": 450.0,
    "오므라이스": 400.0, "김밥": 230.0, "쌀밥": 210.0, "밥": 210.0,
    "짬뽕밥": 750.0, "짜장밥": 500.0, "주먹밥": 120.0, "초밥": 250.0, "컵밥": 300.0, "솥밥": 450.0,
    # 일품 · 분식 · 고기
    "떡볶이": 300.0, "순대": 200.0, "정식": 500.0, "청국장": 300.0, "계란찜": 200.0, "수제비": 600.0,
    "찜": 400.0, "볶음탕": 500.0, "잡채": 200.0, "물회": 600.0, "프라푸치노": 473.0, "모카": 355.0,
    "마끼아또": 355.0, "녹차라떼": 355.0, "초코라떼": 355.0, "콜드브루": 473.0, "회": 250.0, "토스트": 40.0, "버블티": 600.0,
    "아몬드": 25.0, "견과": 25.0, "땅콩": 25.0, "호두": 25.0, "족발": 250.0, "보쌈": 250.0, "삼겹살": 200.0,
    "불고기": 200.0, "제육볶음": 250.0, "돈까스": 200.0, "돈가스": 200.0, "탕수육": 300.0,
    "꿔바로우": 300.0, "샐러드": 250.0, "치킨": 300.0, "감자튀김": 120.0,
    # 음료 (한 잔): 카페 에이드·스무디·차는 톨 사이즈보다 큰 컵이 기본
    "빙수": 500.0, "카푸치노": 355.0, "에이드": 450.0, "스무디": 400.0, "쉐이크": 400.0, "밀크티": 400.0, "차": 300.0, "식혜": 240.0,
    "아메리카노": 355.0, "커피": 355.0, "라떼": 355.0, "우유": 200.0, "콜라": 355.0,
}
# Units that mean "one serving of this dish" rather than a fixed weight.
_SERVING_UNITS = {"인분", "그릇", "접시", "개", "마리", "판", "공기", "대접", "뚝배기", "잔", "serving", "bowl", "plate"}


def serving_grams(food_name: str | None) -> float | None:
    """Grams in one usual serving of `food_name` (longest dish-type suffix), or None."""
    key = _food_key(food_name)
    if not key:
        return None
    try:  # search aliases name the same dish: "pho" -> 쌀국수, "카레" -> 카레라이스
        from engines.food.food_query_normalization import FOOD_SEARCH_ALIAS_RULES
        alias = FOOD_SEARCH_ALIAS_RULES.get((food_name or "").strip().lower()) or FOOD_SEARCH_ALIAS_RULES.get(key)
        if alias:
            key = _food_key(alias)
    except Exception:  # pragma: no cover - normalisation module always present in the app
        pass
    best = ""
    for suffix in SERVING_GRAMS_BY_SUFFIX:
        if key.endswith(suffix) and len(suffix) > len(best):
            best = suffix
    if best == "밥" and key != "밥" and not any(key[:-1].endswith(g) for g in _PLAIN_RICE_PREFIXES):
        # "잡채밥", "제육밥", "카레밥": a dish served over rice (덮밥 plate), not one bowl of rice
        return 400.0
    return SERVING_GRAMS_BY_SUFFIX.get(best)


# "보리밥", "현미밥", "햇반 같은 즉석밥", "찬밥": plain rice, one 공기
_PLAIN_RICE_PREFIXES = (
    "쌀", "흰", "백미", "보리", "현미", "잡곡", "흑미", "콩", "찹쌀", "오곡", "귀리", "기장", "수수", "율무",
    "공기", "공깃", "즉석", "햇", "찬", "식은", "곤약", "렌틸", "병아리콩", "오분도", "칠분도", "발아현미", "흑",
    "맨", "흰쌀", "누룽지", "곤드레", "영양", "약",
)


# Aliases resolved before the override lookup.
_FOOD_UNIT_ALIASES: dict[str, str] = {
    "삶은계란": "계란",
    "삶은달걀": "달걀",
    "구운계란": "계란",
    "계란후라이": "계란",
    "달걀후라이": "달걀",
    "흰쌀밥": "쌀밥",
    "공깃밥": "쌀밥",
    "공기밥": "쌀밥",
}

_BARE_COUNT_MAX = 20.0
_QUANTITY_SUFFIX_RE = re.compile(r"\s*\d+(?:\.\d+)?\s*[가-힣A-Za-z]{0,3}\s*$")


def _compact(value: str | None) -> str:
    return "".join((value or "").split())


def _food_key(food_name: str | None) -> str:
    key = _compact(food_name)
    stripped = _QUANTITY_SUFFIX_RE.sub("", key)
    if stripped:
        key = stripped
    return _FOOD_UNIT_ALIASES.get(key, key)


_ONE_SYLLABLE_HEADS = frozenset({"떡", "전", "빵", "닭"})


def food_unit_overrides(food_name: str | None) -> dict[str, float]:
    """Return per-food unit overrides (exact name first, then head-noun suffix)."""
    key = _food_key(food_name)
    if not key:
        return {}
    if key in FOOD_UNIT_GRAMS:
        return FOOD_UNIT_GRAMS[key]
    # Korean head noun is the last element: "양념치킨" -> "치킨", "삶은계란" -> "계란".
    best = ""
    for candidate in FOOD_UNIT_GRAMS:
        # one-syllable heads only for dish families whose head noun is one syllable (감자전, 찹쌀떡, 마늘빵)
        ok_len = len(candidate) >= 2 or candidate in _ONE_SYLLABLE_HEADS
        if ok_len and key.endswith(candidate) and len(candidate) > len(best):
            best = candidate
    return FOOD_UNIT_GRAMS.get(best, {})


def is_countable_food(food_name: str | None) -> bool:
    overrides = food_unit_overrides(food_name)
    return "개" in overrides or "알" in overrides


def unit_to_grams(food_name: str | None, unit: str | None) -> float | None:
    """Grams per one `unit` of `food_name`. None for gram-like or unknown units."""
    if unit is None:
        return None
    u = unit.strip().lower()
    if not u or u in GRAM_UNITS:
        return None
    overrides = food_unit_overrides(food_name)
    if u in overrides:
        return overrides[u]
    return UNIT_DEFAULT_GRAMS.get(u)


def resolve_portion_grams(food_name: str | None, amount, unit: str | None) -> float | None:
    """Total grams (ml treated as g) for an amount+unit. None when amount is missing."""
    if amount is None:
        return None
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return None
    u = (unit or "").strip().lower()
    if u in GRAM_UNITS:
        return amount
    if u in _KG_UNITS:
        return amount * 1000.0
    if u in _GEUN_UNITS:
        return amount * 600.0
    if u in _LITER_UNITS:
        return amount * 1000.0
    if u in _LENGTH_UNITS:
        piece = unit_to_grams(food_name, "개") or 200.0
        return max(1.0, amount / _STANDARD_PIECE_CM) * piece
    if not u:
        # Bare count: "계란 2" -> 2 pieces, not 2 g.
        if is_countable_food(food_name):
            return amount * unit_to_grams(food_name, "개")
        serving = serving_grams(food_name)
        if serving is not None and amount <= _BARE_COUNT_MAX:
            return amount * serving  # "된장찌개 1" = one bowl
        if amount <= _BARE_COUNT_MAX:
            return amount * UNIT_DEFAULT_GRAMS["개"]
        return amount
    overrides = food_unit_overrides(food_name)
    if u not in overrides and u in _SERVING_UNITS:
        serving = serving_grams(food_name)
        if serving is not None:
            return amount * serving  # "짬뽕 한 그릇", "칼국수 1인분", "돈까스 하나"
    per_unit = unit_to_grams(food_name, u)
    if per_unit is None:
        # Unknown count-like unit: one generic serving per count.
        per_unit = UNIT_DEFAULT_GRAMS["개"]
    return amount * per_unit


def portion_confidence(food_name: str | None, amount, unit: str | None) -> str:
    """How the grams of `resolve_portion_grams` were obtained.

    "explicit": the user gave a weight/volume (g, ml, kg, 근, L).
    "exact":    a per-food table entry for this exact food (FOOD_UNIT_GRAMS or a serving entry
                whose key is the whole name), e.g. 계란 1개, 짜장면 1그릇.
    "suffix":   a head-noun guess (양념치킨 -> 치킨 조각, 잡채밥 -> 밥 = one rice bowl).
    "default":  a generic unit default with nothing known about the food (개 = 100 g).
    """
    u = (unit or "").strip().lower()
    if u in GRAM_UNITS or u in _KG_UNITS or u in _GEUN_UNITS or u in _LITER_UNITS:
        return "explicit"
    key = _food_key(food_name)
    if not key:
        return "inferred"
    try:
        from engines.food.food_query_normalization import FOOD_SEARCH_ALIAS_RULES
        alias = FOOD_SEARCH_ALIAS_RULES.get((food_name or "").strip().lower()) or FOOD_SEARCH_ALIAS_RULES.get(key)
    except Exception:  # pragma: no cover
        alias = None
    keys = {key} | ({_food_key(alias)} if alias else set())
    for k in keys:
        overrides = FOOD_UNIT_GRAMS.get(k, {})
        if u and u in overrides:
            return "exact"
        if (not u or u in _SERVING_UNITS) and k in SERVING_GRAMS_BY_SUFFIX:
            return "exact"
        if not u and ("개" in overrides or "알" in overrides):
            return "exact"
    # head-noun guesses: "양념치킨" -> 치킨 조각, "잡채밥" -> 밥 = one bowl
    suffix_overrides = food_unit_overrides(food_name)
    if (u and u in suffix_overrides) or (not u and is_countable_food(food_name)):
        return "suffix"
    if (not u or u in _SERVING_UNITS) and serving_grams(food_name) is not None:
        return "suffix"
    return "default"


def preferred_basis_unit(unit: str | None) -> str:
    """'ml' for liquid units, 'g' otherwise (used to pick among g/ml duplicate rows)."""
    u = (unit or "").strip().lower()
    if u in {"ml", "밀리리터", "cc", "l", "리터", "캔", "병", "잔"}:
        return "ml"
    return "g"

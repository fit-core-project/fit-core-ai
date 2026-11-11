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
}

# Per-food overrides. Keys are compact (no whitespace) names.
# 라면: DB "라면" rows are the cooked dish (noodles + broth, ~82 kcal/100g),
# so one pack (120 g dry + water) is expressed as ~600 g cooked -> ~500 kcal.
FOOD_UNIT_GRAMS: dict[str, dict[str, float]] = {
    "계란": {"개": 50.0, "알": 50.0},
    "달걀": {"개": 50.0, "알": 50.0},
    "메추리알": {"개": 10.0, "알": 10.0},
    "바나나": {"개": 120.0},
    "사과": {"개": 200.0},
    "배": {"개": 400.0},
    "귤": {"개": 80.0},
    "오렌지": {"개": 180.0},
    "키위": {"개": 90.0},
    "복숭아": {"개": 200.0},
    "토마토": {"개": 150.0},
    "방울토마토": {"개": 15.0, "알": 15.0},
    "고구마": {"개": 150.0},
    "감자": {"개": 130.0},
    "라면": {"개": 600.0, "봉지": 600.0, "그릇": 600.0, "인분": 600.0},
    "컵라면": {"개": 300.0},
    "치킨": {"조각": 100.0, "개": 100.0, "마리": 1000.0},
    "피자": {"조각": 120.0},
    "케이크": {"조각": 100.0},
    "닭가슴살": {"개": 100.0, "팩": 100.0, "덩이": 100.0},
    "콜라": {"캔": 355.0, "병": 500.0, "잔": 250.0, "컵": 200.0},
    "사이다": {"캔": 355.0, "병": 500.0, "잔": 250.0, "컵": 200.0},
    "맥주": {"캔": 355.0, "병": 500.0, "잔": 500.0},
    "소주": {"병": 360.0, "잔": 50.0},
    "막걸리": {"병": 750.0, "잔": 200.0, "사발": 300.0},
    "와인": {"잔": 150.0, "병": 750.0},
    "소맥": {"잔": 200.0},
    "생맥": {"잔": 500.0},
    "생맥주": {"잔": 500.0},
    "우유": {"팩": 200.0, "개": 200.0, "컵": 200.0, "잔": 200.0},
    "아메리카노": {"잔": 355.0, "컵": 355.0, "개": 355.0},
    "두부": {"모": 300.0},
    "김밥": {"줄": 230.0},
    "삼각김밥": {"개": 110.0},
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
    "마카롱": {"개": 25.0},
    "인절미": {"개": 20.0, "조각": 20.0},
    "떡": {"개": 25.0, "조각": 25.0},
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
    # 일품 · 분식 · 고기
    "떡볶이": 300.0, "순대": 200.0, "족발": 250.0, "보쌈": 250.0, "삼겹살": 200.0,
    "불고기": 200.0, "제육볶음": 250.0, "돈까스": 200.0, "돈가스": 200.0, "탕수육": 300.0,
    "꿔바로우": 300.0, "샐러드": 250.0, "치킨": 300.0, "감자튀김": 120.0,
    # 음료 (한 잔)
    "아메리카노": 355.0, "커피": 355.0, "라떼": 355.0, "우유": 200.0, "콜라": 355.0,
}
# Units that mean "one serving of this dish" rather than a fixed weight.
_SERVING_UNITS = {"인분", "그릇", "접시", "개", "마리", "판", "공기"}


def serving_grams(food_name: str | None) -> float | None:
    """Grams in one usual serving of `food_name` (longest dish-type suffix), or None."""
    key = _food_key(food_name)
    if not key:
        return None
    best = ""
    for suffix in SERVING_GRAMS_BY_SUFFIX:
        if key.endswith(suffix) and len(suffix) > len(best):
            best = suffix
    return SERVING_GRAMS_BY_SUFFIX.get(best)


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
        if len(candidate) >= 2 and key.endswith(candidate) and len(candidate) > len(best):
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
    if u in _LITER_UNITS:
        return amount * 1000.0
    if u in _LENGTH_UNITS:
        piece = unit_to_grams(food_name, "개") or 200.0
        return max(1.0, amount / _STANDARD_PIECE_CM) * piece
    if not u:
        # Bare count: "계란 2" -> 2 pieces, not 2 g.
        if is_countable_food(food_name):
            return amount * unit_to_grams(food_name, "개")
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


def preferred_basis_unit(unit: str | None) -> str:
    """'ml' for liquid units, 'g' otherwise (used to pick among g/ml duplicate rows)."""
    u = (unit or "").strip().lower()
    if u in {"ml", "밀리리터", "cc", "l", "리터", "캔", "병", "잔"}:
        return "ml"
    return "g"

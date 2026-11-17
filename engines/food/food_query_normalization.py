"""Food search query normalization rules.

This module only rewrites search queries before vector retrieval. It does not
change API request/response contracts or user-visible food names.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal


FoodQueryIntent = Literal["ingredient", "dish", "ambiguous"]


@dataclass(frozen=True)
class FoodQueryAnalysis:
    raw_query: str
    cleaned_query: str
    normalized_query: str
    intent: FoodQueryIntent
    protected: bool
    alias_applied: bool
    cooking_state: str | None
    matched_dish_keyword: str | None
    reason: str


PROTECTED_FOOD_SEARCH_QUERIES: set[str] = {
    "계란빵",
    "볶음밥 계란",
    "김밥 계란",
    "닭가슴살",
    "샐러드 닭가슴살",
    "닭가슴살 샐러드",
    "샌드위치 닭가슴살",
}


FOOD_SEARCH_ALIAS_RULES: dict[str, str] = {
    "계란": "달걀",
    "계란 생것": "달걀 생것",
    "삶은 계란": "달걀 삶은것",
    "삶은계란": "달걀 삶은것",
    "계란 삶은것": "달걀 삶은것",
    "계란후라이": "달걀후라이",
    "닭가슴살 생것": "닭고기 가슴(껍질 제거) 생것",
    "닭 가슴살 생것": "닭고기 가슴(껍질 제거) 생것",
    "닭가슴살 삶은것": "닭고기 가슴(껍질 제거) 삶은것",
    "닭 가슴살 삶은것": "닭고기 가슴(껍질 제거) 삶은것",
    "삶은 닭가슴살": "닭고기 가슴(껍질 제거) 삶은것",
    "삶은 닭 가슴살": "닭고기 가슴(껍질 제거) 삶은것",
    "닭가슴살 구운것": "닭고기 가슴(껍질 제거) 구운것(팬)",
    "닭 가슴살 구운것": "닭고기 가슴(껍질 제거) 구운것(팬)",
    "구운 닭가슴살": "닭고기 가슴(껍질 제거) 구운것(팬)",
    "구운 닭 가슴살": "닭고기 가슴(껍질 제거) 구운것(팬)",
    # Everyday names for plain rice / coffee / dishes (2026-11-10 stage eval: these hit
    # 김밥 / 밤밥 / 에스프레소 / 떡볶이 카레 instead).
    "밥": "쌀밥", "공기밥": "쌀밥", "공깃밥": "쌀밥", "흰밥": "쌀밥",
    "햇반": "쌀밥", "즉석밥": "쌀밥",
    "커피": "커피 아메리카노", "아메리카노": "커피 아메리카노", "아아": "커피 아메리카노",
    "아이스 아메리카노": "커피 아메리카노", "아이스아메리카노": "커피 아메리카노",
    "라떼": "카페라떼", "카페라테": "카페라떼",
    "카레": "카레라이스",
    "coke": "콜라", "cola": "콜라", "코카콜라": "콜라",
    "pho": "쌀국수",
    "와퍼": "햄버거 와퍼",
    "김볶밥": "김치볶음밥",
    "돈까스": "돈가스", "돈까쓰": "돈가스",
    "toast": "토스트", "egg": "달걀", "eggs": "달걀", "banana": "바나나", "rice": "쌀밥",
    "아메리까노": "커피 아메리카노", "떡뽁이": "떡볶이",
    "물냉": "물냉면", "비냉": "비빔냉면", "삼김": "삼각김밥", "아아": "커피 아메리카노",
    "cheeseburger": "치즈버거", "fries": "감자튀김", "french fries": "감자튀김", "cold brew": "콜드브루",
    "맥너겟": "치킨너겟", "초코라떼": "더블초코라떼", "combination pizza": "콤비네이션 피자",
    # varieties without a raw row of their own -> the nearest plain raw row (not 에이드 / 샐러드)
    "샤인머스캣": "포도 마스캇함브르그 생것", "단호박": "호박 단호박 찐것",
    # English dish names (LLM keeps the user's English words); plural forms included
    "pad thai": "팟타이", "pad thais": "팟타이", "croquette": "고로케", "croquettes": "고로케", "cappuccino": "카푸치노", "cappuccinoes": "카푸치노", "fried rice": "볶음밥", "fried rices": "볶음밥", "kimchi stew": "김치찌개", "kimchi stews": "김치찌개", "bibimbap": "비빔밥", "bibimbaps": "비빔밥", "ramen": "라면", "ramens": "라면", "salad": "샐러드", "salads": "샐러드", "sandwich": "샌드위치", "sandwiches": "샌드위치", "bagel": "베이글", "bagels": "베이글", "latte": "카페라떼", "lattes": "카페라떼", "americano": "아메리카노", "americanoes": "아메리카노", "pizza": "피자", "pizzas": "피자", "burger": "햄버거", "burgers": "햄버거", "hamburger": "햄버거", "hamburgers": "햄버거", "fried chicken": "후라이드치킨", "fried chickens": "후라이드치킨", "tteokbokki": "떡볶이", "tteokbokkis": "떡볶이", "kimbap": "김밥", "kimbaps": "김밥", "gimbap": "김밥", "gimbaps": "김밥", "dumpling": "만두", "dumplings": "만두", "pasta": "파스타", "pastas": "파스타", "steak": "스테이크", "steaks": "스테이크", "sushi": "초밥", "sushis": "초밥", "udon": "우동", "udons": "우동", "curry": "카레라이스", "currys": "카레라이스", "oatmeal": "오트밀", "oatmeals": "오트밀", "yogurt": "요거트", "yogurts": "요거트", "omelette": "오믈렛", "omelettes": "오믈렛", "omelet": "오믈렛", "omelets": "오믈렛", "pancake": "팬케이크", "pancakes": "팬케이크", "waffle": "와플", "waffles": "와플", "donut": "도넛", "donuts": "도넛", "doughnut": "도넛", "doughnuts": "도넛", "cookie": "쿠키", "cookies": "쿠키", "cake": "케이크", "cakes": "케이크", "chicken breast": "닭가슴살", "chicken breasts": "닭가슴살", "milk": "우유", "milks": "우유", "orange juice": "오렌지주스", "orange juices": "오렌지주스", "beer": "맥주", "beers": "맥주", "wine": "와인", "wines": "와인", "soju": "소주", "sojus": "소주", "coke": "콜라", "cokes": "콜라", "protein shake": "단백질 쉐이크", "protein shakes": "단백질 쉐이크", "croissant": "크루아상", "croissants": "크루아상", "muffin": "머핀", "muffins": "머핀", "granola": "그래놀라", "granolas": "그래놀라", "smoothie": "스무디", "smoothies": "스무디", "hot dog": "핫도그", "hot dogs": "핫도그", "nachos": "나초", "taco": "타코", "tacoes": "타코", "burrito": "부리또", "burritoes": "부리또", "risotto": "리조또", "risottoes": "리조또", "carbonara": "까르보나라", "carbonaras": "까르보나라", "spaghetti": "스파게티", "spaghettis": "스파게티", "tiramisu": "티라미수", "tiramisus": "티라미수", "cheesecake": "치즈케이크", "cheesecakes": "치즈케이크", "brownie": "브라우니", "brownies": "브라우니", "apple": "사과", "apples": "사과", "banana": "바나나", "bananas": "바나나", "orange": "오렌지", "oranges": "오렌지", "sweet potato": "고구마", "sweet potatoes": "고구마", "potato": "감자", "potatoes": "감자", "boiled egg": "삶은 달걀", "boiled eggs": "삶은 달걀", "rice": "쌀밥", "rices": "쌀밥", "ramyeon": "라면", "ramyeons": "라면",
    "갈매기": "돼지고기 삼겹살(갈매기살) 생것", "갈매기살": "돼지고기 삼겹살(갈매기살) 생것",
    # meat cuts as ordered at a BBQ restaurant -> the cooked DB rows
    "목살": "돼지고기 목심 구운것", "돼지목살": "돼지고기 목심 구운것", "목살구이": "돼지고기 목심 구운것",
    "차돌": "차돌박이구이", "차돌박이": "차돌박이구이",
    "막창": "막창구이", "소막창": "막창구이 소고기", "돼지막창": "막창구이",
    "la갈비": "소갈비 구이", "LA갈비": "소갈비 구이", "엘에이갈비": "소갈비 구이",
    "계란 프라이": "달걀후라이", "계란프라이": "달걀후라이", "달걀프라이": "달걀후라이",
}


# Matches a trailing quantity token at the END of a query string.
# Examples: "계란 2개", "닭가슴살 100g", "바나나 1개", "우유 200ml"
# The strip is limited to the trailing position so that food names whose
# description begins with a number (rare in this DB) are not affected.
# Units: put multi-char units before single-char overlaps (kg before g, ml before l).
_QUANTITY_SUFFIX_RE = re.compile(
    r"\s*\d+(?:\.\d+)?\s*(?:개|kg|g|ml|l|인분|컵|조각|쪽|티스푼|스푼)\s*$",
    re.IGNORECASE,
)


def strip_trailing_quantity(text: str) -> str:
    """Remove a trailing quantity token (number + unit) from a query string.

    Returns the original string unchanged if stripping would leave an empty
    string, or if no quantity suffix is present.
    """
    stripped = _QUANTITY_SUFFIX_RE.sub("", text).strip()
    return stripped if stripped else text


_COOKING_STATE_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("삶은것", "삶은것"),
    ("삶은", "삶은것"),
    ("구운것", "구운것"),
    ("구운", "구운것"),
    ("생것", "생것"),
    ("후라이", "후라이"),
)

_DISH_KEYWORDS: tuple[str, ...] = (
    "샌드위치",
    "볶음밥",
    "샐러드",
    "김밥",
    "찌개",
    "튀김",
    "조림",
    "덮밥",
    "빵",
    "국",
)

# Amount-bearing queries were previously listed here for conservative
# protection. Quantity strip (strip_trailing_quantity) now handles them
# generically before the protected/alias checks, so this set is empty.
_CONSERVATIVE_PROTECTED_QUERIES: set[str] = {}


def clean_food_search_query(query: str | None) -> str:
    if not query:
        return ""
    return " ".join(str(query).strip().split())


def _detect_cooking_state(cleaned_query: str) -> str | None:
    for keyword, canonical in _COOKING_STATE_KEYWORDS:
        if keyword in cleaned_query:
            return canonical
    return None


def _detect_dish_keyword(cleaned_query: str) -> str | None:
    for keyword in _DISH_KEYWORDS:
        if keyword in cleaned_query:
            return keyword
    return None


def _is_protected_for_analysis(cleaned_query: str) -> bool:
    return (
        cleaned_query in PROTECTED_FOOD_SEARCH_QUERIES
        or cleaned_query in _CONSERVATIVE_PROTECTED_QUERIES
    )


def analyze_food_query_for_search(query: str | None) -> FoodQueryAnalysis:
    """Analyze a food search query without changing public search contracts."""
    raw_query = "" if query is None else str(query)
    cleaned_query = clean_food_search_query(query)
    # Strip trailing quantity token (e.g. "계란 2개" → "계란") before alias /
    # protected checks so that quantity-bearing queries reach the same
    # normalization path as their bare counterparts.
    cleaned_query = strip_trailing_quantity(cleaned_query)
    if not cleaned_query:
        return FoodQueryAnalysis(
            raw_query=raw_query,
            cleaned_query="",
            normalized_query="",
            intent="ambiguous",
            protected=False,
            alias_applied=False,
            cooking_state=None,
            matched_dish_keyword=None,
            reason="empty_query",
        )

    cooking_state = _detect_cooking_state(cleaned_query)
    matched_dish_keyword = _detect_dish_keyword(cleaned_query)
    protected = _is_protected_for_analysis(cleaned_query)

    if protected:
        intent: FoodQueryIntent = "dish" if matched_dish_keyword else "ambiguous"
        return FoodQueryAnalysis(
            raw_query=raw_query,
            cleaned_query=cleaned_query,
            normalized_query=cleaned_query,
            intent=intent,
            protected=True,
            alias_applied=False,
            cooking_state=cooking_state,
            matched_dish_keyword=matched_dish_keyword,
            reason="protected_dish" if intent == "dish" else "protected_ambiguous",
        )

    alias_normalized = FOOD_SEARCH_ALIAS_RULES.get(cleaned_query)
    if alias_normalized is not None:
        intent = "dish" if cooking_state == "후라이" else "ingredient"
        return FoodQueryAnalysis(
            raw_query=raw_query,
            cleaned_query=cleaned_query,
            normalized_query=alias_normalized,
            intent=intent,
            protected=False,
            alias_applied=alias_normalized != cleaned_query,
            cooking_state=cooking_state,
            matched_dish_keyword=matched_dish_keyword,
            reason="alias_applied",
        )

    if matched_dish_keyword:
        intent = "dish"
        reason = "dish_keyword"
    elif cooking_state:
        intent = "ingredient"
        reason = "cooking_state"
    else:
        intent = "ambiguous"
        reason = "no_rule"

    return FoodQueryAnalysis(
        raw_query=raw_query,
        cleaned_query=cleaned_query,
        normalized_query=cleaned_query,
        intent=intent,
        protected=False,
        alias_applied=False,
        cooking_state=cooking_state,
        matched_dish_keyword=matched_dish_keyword,
        reason=reason,
    )


def normalize_food_query_for_search(query: str | None) -> str:
    """Return the canonicalized query used for food vector retrieval."""
    return analyze_food_query_for_search(query).normalized_query


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def build_food_search_queries(query: str | None) -> list[str]:
    """Build ordered vector retrieval queries for a food search input."""
    analysis = analyze_food_query_for_search(query)
    if not analysis.cleaned_query:
        return []
    if analysis.protected:
        return [analysis.cleaned_query]
    if analysis.alias_applied:
        return _dedupe_preserve_order(
            [analysis.normalized_query, analysis.cleaned_query]
        )
    return [analysis.cleaned_query]

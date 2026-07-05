"""Food search query normalization rules.

This module only rewrites search queries before vector retrieval. It does not
change API request/response contracts or user-visible food names.
"""
from __future__ import annotations

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
}


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

# Amount-bearing raw ingredient queries are intentionally preserved until a
# later intent-aware retrieval step can safely decide how to handle quantities.
_CONSERVATIVE_PROTECTED_QUERIES: set[str] = {
    "닭가슴살 100g",
}


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

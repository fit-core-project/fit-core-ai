"""Food search query normalization rules.

This module only rewrites search queries before vector retrieval. It does not
change API request/response contracts or user-visible food names.
"""
from __future__ import annotations


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


def normalize_food_query_for_search(query: str | None) -> str:
    """Return the canonicalized query used for food vector retrieval."""
    if not query:
        return ""

    normalized = " ".join(str(query).strip().split())
    if normalized in PROTECTED_FOOD_SEARCH_QUERIES:
        return normalized

    return FOOD_SEARCH_ALIAS_RULES.get(normalized, normalized)

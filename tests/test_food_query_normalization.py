from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from engines.food.food_engine import FoodSearchEngine
from engines.food.food_query_normalization import (
    normalize_food_query_for_search,
    strip_trailing_quantity,
)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("계란", "달걀"),
        ("계란 생것", "달걀 생것"),
        ("삶은 계란", "달걀 삶은것"),
        ("삶은계란", "달걀 삶은것"),
        ("계란 삶은것", "달걀 삶은것"),
        ("계란후라이", "달걀후라이"),
        ("계란빵", "계란빵"),
        ("볶음밥 계란", "볶음밥 계란"),
        ("김밥 계란", "김밥 계란"),
        ("닭가슴살 생것", "닭고기 가슴(껍질 제거) 생것"),
        ("닭 가슴살 생것", "닭고기 가슴(껍질 제거) 생것"),
        ("삶은 닭가슴살", "닭고기 가슴(껍질 제거) 삶은것"),
        ("삶은 닭 가슴살", "닭고기 가슴(껍질 제거) 삶은것"),
        ("닭가슴살 삶은것", "닭고기 가슴(껍질 제거) 삶은것"),
        ("닭 가슴살 삶은것", "닭고기 가슴(껍질 제거) 삶은것"),
        ("구운 닭가슴살", "닭고기 가슴(껍질 제거) 구운것(팬)"),
        ("구운 닭 가슴살", "닭고기 가슴(껍질 제거) 구운것(팬)"),
        ("닭가슴살 구운것", "닭고기 가슴(껍질 제거) 구운것(팬)"),
        ("닭 가슴살 구운것", "닭고기 가슴(껍질 제거) 구운것(팬)"),
        ("닭가슴살", "닭가슴살"),
        ("샐러드 닭가슴살", "샐러드 닭가슴살"),
        ("닭가슴살 샐러드", "닭가슴살 샐러드"),
        ("샌드위치 닭가슴살", "샌드위치 닭가슴살"),
        (None, ""),
        ("", ""),
        ("  삶은   계란  ", "달걀 삶은것"),
        ("  닭가슴살   생것  ", "닭고기 가슴(껍질 제거) 생것"),
    ],
)
def test_normalize_food_query_for_search(query, expected):
    assert normalize_food_query_for_search(query) == expected


# ---------------------------------------------------------------------------
# strip_trailing_quantity unit tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Basic quantity strip
        ("계란 2개", "계란"),
        ("바나나 1개", "바나나"),
        ("닭가슴살 100g", "닭가슴살"),
        ("우유 200ml", "우유"),
        ("고구마 3.5kg", "고구마"),
        ("두부 1인분", "두부"),
        ("오트밀 2컵", "오트밀"),
        # No quantity suffix — unchanged
        ("계란", "계란"),
        ("닭가슴살", "닭가슴살"),
        ("계란빵", "계란빵"),
        ("달걀 생것", "달걀 생것"),
        ("삶은 계란", "삶은 계란"),
        # Non-trailing numbers or units — unchanged (strip is end-anchored)
        ("1인분 닭가슴살", "1인분 닭가슴살"),
        ("100g 달걀", "100g 달걀"),
        # Quantity-only input — returns original (guard against empty result)
        ("100g", "100g"),
        ("1개", "1개"),
        # Protected compound queries unchanged (no trailing digit+unit)
        ("샐러드 닭가슴살", "샐러드 닭가슴살"),
        ("닭가슴살 샐러드", "닭가슴살 샐러드"),
        # Idempotency: applying strip twice returns same result
    ],
)
def test_strip_trailing_quantity(text, expected):
    assert strip_trailing_quantity(text) == expected


def test_strip_trailing_quantity_is_idempotent():
    for query in ["계란 2개", "닭가슴살 100g", "바나나 1개", "계란빵", "계란"]:
        once = strip_trailing_quantity(query)
        twice = strip_trailing_quantity(once)
        assert once == twice, f"Not idempotent for {query!r}: {once!r} -> {twice!r}"


# ---------------------------------------------------------------------------
# normalize_food_query_for_search with quantity suffix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("query", "expected_normalized"),
    [
        # quantity strip → alias applies
        ("계란 2개", "달걀"),
        ("삶은 계란 1개", "달걀 삶은것"),
        # quantity strip → ambiguous (no alias for 바나나)
        ("바나나 1개", "바나나"),
        # quantity strip → strips to protected-ambiguous base
        ("닭가슴살 100g", "닭가슴살"),
        # protected dishes with no trailing quantity — unchanged
        ("계란빵", "계란빵"),
        ("샐러드 닭가슴살", "샐러드 닭가슴살"),
    ],
)
def test_normalize_with_quantity_suffix(query, expected_normalized):
    assert normalize_food_query_for_search(query) == expected_normalized


class _FakeVectorStore:
    def __init__(self):
        self.queries: list[str] = []

    def similarity_search_with_score(self, query, k):
        self.queries.append(query)
        doc = SimpleNamespace(
            metadata={
                "name": query,
                "protein_g": 10.0,
                "carb_g": 20.0,
                "fat_g": 3.0,
                "kcal": 147.0,
                "basis_value": 100.0,
                "basis_unit": "g",
            }
        )
        return [(doc, 0.1)]


def _engine_with_fake_vectorstore():
    engine = object.__new__(FoodSearchEngine)
    engine.available = True
    engine._vector_store = _FakeVectorStore()
    engine._lock = threading.Lock()
    return engine


@pytest.mark.parametrize(
    ("raw_query", "expected_query"),
    [
        ("삶은 계란", ["달걀 삶은것", "삶은 계란"]),
        ("계란빵", "계란빵"),
        ("샐러드 닭가슴살", "샐러드 닭가슴살"),
    ],
)
def test_food_search_engine_passes_normalized_query_to_vectorstore(raw_query, expected_query):
    engine = _engine_with_fake_vectorstore()

    result = engine.search(raw_query, 100, "g")

    assert result is not None
    expected_queries = (
        expected_query if isinstance(expected_query, list) else [expected_query]
    )
    assert engine._vector_store.queries == expected_queries

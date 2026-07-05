from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from engines.food.food_engine import FoodSearchEngine, TOP_K
from engines.food.food_query_normalization import build_food_search_queries


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (None, []),
        ("", []),
        ("  삶은   계란  ", ["달걀 삶은것", "삶은 계란"]),
        ("계란", ["달걀", "계란"]),
        ("계란 생것", ["달걀 생것", "계란 생것"]),
        ("삶은 계란", ["달걀 삶은것", "삶은 계란"]),
        ("삶은계란", ["달걀 삶은것", "삶은계란"]),
        ("계란 삶은것", ["달걀 삶은것", "계란 삶은것"]),
        ("계란후라이", ["달걀후라이", "계란후라이"]),
        ("닭가슴살 생것", ["닭고기 가슴(껍질 제거) 생것", "닭가슴살 생것"]),
        ("삶은 닭가슴살", ["닭고기 가슴(껍질 제거) 삶은것", "삶은 닭가슴살"]),
        ("닭가슴살 삶은것", ["닭고기 가슴(껍질 제거) 삶은것", "닭가슴살 삶은것"]),
        ("구운 닭가슴살", ["닭고기 가슴(껍질 제거) 구운것(팬)", "구운 닭가슴살"]),
        ("닭가슴살 구운것", ["닭고기 가슴(껍질 제거) 구운것(팬)", "닭가슴살 구운것"]),
        ("계란빵", ["계란빵"]),
        ("볶음밥 계란", ["볶음밥 계란"]),
        ("김밥 계란", ["김밥 계란"]),
        ("닭가슴살", ["닭가슴살"]),
        ("닭가슴살 100g", ["닭가슴살 100g"]),
        ("샐러드 닭가슴살", ["샐러드 닭가슴살"]),
        ("닭가슴살 샐러드", ["닭가슴살 샐러드"]),
        ("샌드위치 닭가슴살", ["샌드위치 닭가슴살"]),
        ("바나나", ["바나나"]),
        ("고구마", ["고구마"]),
    ],
)
def test_build_food_search_queries(query, expected):
    assert build_food_search_queries(query) == expected


class _FakeVectorStore:
    def __init__(self):
        self.queries: list[str] = []

    def similarity_search_with_score(self, query, k):
        self.queries.append(query)
        return [(_doc(query, query), 0.1)]


class _DuplicateVectorStore:
    def __init__(self):
        self.queries: list[str] = []
        self.result_groups = []

    def similarity_search_with_score(self, query, k):
        self.queries.append(query)
        if len(self.queries) == 1:
            results = [
                (_doc("shared", "normalized winner", protein=30.0), 0.1),
                (_doc("normalized-only", "normalized-only"), 0.2),
            ]
        else:
            results = [
                (_doc("shared", "raw duplicate", protein=1.0), 0.05),
                (_doc("raw-only", "raw-only"), 0.2),
                (_doc("raw-extra", "raw-extra"), 0.3),
            ]
        self.result_groups.append(results)
        return results


def _doc(food_id, name, protein=10.0):
    return SimpleNamespace(
        metadata={
            "food_id": food_id,
            "name": name,
            "protein_g": protein,
            "carb_g": 20.0,
            "fat_g": 3.0,
            "kcal": 147.0,
            "basis_value": 100.0,
            "basis_unit": "g",
        },
        page_content=name,
    )


def _engine_with_vectorstore(vectorstore):
    engine = object.__new__(FoodSearchEngine)
    engine.available = True
    engine._vector_store = vectorstore
    engine._lock = threading.Lock()
    return engine


@pytest.mark.parametrize(
    ("raw_query", "expected_queries"),
    [
        ("삶은 계란", ["달걀 삶은것", "삶은 계란"]),
        ("닭가슴살 생것", ["닭고기 가슴(껍질 제거) 생것", "닭가슴살 생것"]),
        ("계란빵", ["계란빵"]),
        ("샐러드 닭가슴살", ["샐러드 닭가슴살"]),
    ],
)
def test_food_search_engine_calls_vectorstore_with_multi_query_plan(
    raw_query,
    expected_queries,
):
    vectorstore = _FakeVectorStore()
    engine = _engine_with_vectorstore(vectorstore)

    result = engine.search(raw_query, 100, "g")

    assert result is not None
    assert vectorstore.queries == expected_queries


def test_food_search_engine_dedupes_candidates_preserving_normalized_priority():
    vectorstore = _DuplicateVectorStore()
    engine = _engine_with_vectorstore(vectorstore)

    result = engine.search("삶은 계란", 100, "g")
    deduped = engine._dedupe_results(vectorstore.result_groups)

    assert vectorstore.queries[:2] == ["달걀 삶은것", "삶은 계란"]
    assert result["matched_name"] == "normalized winner"
    assert result["protein_g"] == 30.0
    assert [doc.metadata["food_id"] for doc, _ in deduped] == [
        "shared",
        "normalized-only",
        "raw-only",
    ]
    assert len(deduped) <= TOP_K

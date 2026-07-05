from __future__ import annotations

import threading

import pytest

from engines.food.food_canonical_index import (
    FoodCanonicalIndex,
    clean_food_index_key,
)
from engines.food.food_engine import FoodSearchEngine
from engines.food.food_query_normalization import analyze_food_query_for_search


def _row(name, *, food_id=None, data_type="", rep_name="", protein=10.0):
    return {
        "food_id": food_id or name,
        "name": name,
        "rep_name": rep_name,
        "data_type": data_type,
        "protein_g": str(protein),
        "carb_g": "20.0",
        "fat_g": "3.0",
        "kcal": "147.0",
        "basis_value": "100.0",
        "basis_unit": "g",
        "embed_text": name,
    }


def _index():
    return FoodCanonicalIndex.from_rows(
        [
            _row("달걀 생것", data_type="원재료성 식품"),
            _row("달걀 삶은것", data_type="원재료성 식품"),
            _row("계란빵", data_type="음식"),
            _row("닭고기 가슴(껍질 제거) 생것", data_type="원재료성 식품"),
            _row("샐러드 닭가슴살", data_type="음식"),
        ]
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ""),
        ("", ""),
        ("  달걀   삶은것  ", "달걀 삶은것"),
        ("[원재료성 식품] 달걀 삶은것", "달걀 삶은것"),
        ("[음식] 계란빵", "계란빵"),
    ],
)
def test_clean_food_index_key(value, expected):
    assert clean_food_index_key(value) == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("달걀 생것", "달걀 생것"),
        ("달걀 삶은것", "달걀 삶은것"),
        ("계란빵", "계란빵"),
        ("닭고기 가슴(껍질 제거) 생것", "닭고기 가슴(껍질 제거) 생것"),
        ("샐러드 닭가슴살", "샐러드 닭가슴살"),
    ],
)
def test_canonical_index_exact_lookup(query, expected):
    results = _index().lookup(query)

    assert [record.metadata["name"] for record in results] == [expected]


def test_canonical_index_does_not_broad_substring_match():
    assert _index().lookup("달걀") == []


class _VectorStore:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query
        self.queries: list[str] = []

    def similarity_search_with_score(self, query, k):
        self.queries.append(query)
        return self.results_by_query.get(query, [])


def _engine(index, vectorstore):
    engine = object.__new__(FoodSearchEngine)
    engine.available = True
    engine._vector_store = vectorstore
    engine._canonical_index = index
    engine._lock = threading.Lock()
    return engine


def _doc_from_row(row):
    return _index().lookup(row["name"])[0].to_document()


def test_protected_query_uses_only_cleaned_lookup_for_canonical_injection():
    index = _index()
    vectorstore = _VectorStore({"계란빵": [(_doc_from_row(_row("달걀 생것")), 0.1)]})
    engine = _engine(index, vectorstore)

    result = engine.search("계란빵", 100, "g")

    assert result["matched_name"] == "계란빵"
    assert engine._canonical_lookup_queries(analyze_food_query_for_search("계란빵")) == ["계란빵"]


def test_protected_chicken_salad_does_not_inject_raw_chicken_breast():
    index = _index()
    vectorstore = _VectorStore(
        {"샐러드 닭가슴살": [(_doc_from_row(_row("닭고기 가슴(껍질 제거) 생것")), 0.1)]}
    )
    engine = _engine(index, vectorstore)

    result = engine.search("샐러드 닭가슴살", 100, "g")

    assert result["matched_name"] == "샐러드 닭가슴살"


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("삶은 계란", "달걀 삶은것"),
        ("닭가슴살 생것", "닭고기 가슴(껍질 제거) 생것"),
    ],
)
def test_alias_normalized_exact_candidate_injection(query, expected):
    index = _index()
    vectorstore = _VectorStore(
        {
            "달걀 삶은것": [(_doc_from_row(_row("계란빵")), 0.1)],
            "삶은 계란": [],
            "닭고기 가슴(껍질 제거) 생것": [(_doc_from_row(_row("샐러드 닭가슴살")), 0.1)],
            "닭가슴살 생것": [],
        }
    )
    engine = _engine(index, vectorstore)

    result = engine.search(query, 100, "g")

    assert result["matched_name"] == expected


def test_canonical_and_vector_duplicate_is_deduped():
    index = _index()
    duplicate = _doc_from_row(_row("달걀 삶은것"))
    vectorstore = _VectorStore({"달걀 삶은것": [(duplicate, 0.2)], "삶은 계란": []})
    engine = _engine(index, vectorstore)

    result = engine.search("삶은 계란", 100, "g")
    canonical = engine._lookup_canonical_candidates(analyze_food_query_for_search("삶은 계란"))
    vector = engine._collect_candidates([[(duplicate, 0.2)], []], ["달걀 삶은것", "삶은 계란"])
    deduped = engine._dedupe_candidates(canonical + vector)

    assert result["matched_name"] == "달걀 삶은것"
    assert [candidate.document.metadata["name"] for candidate in deduped].count("달걀 삶은것") == 1

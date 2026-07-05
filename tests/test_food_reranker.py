from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from engines.food.food_engine import FoodSearchEngine
from engines.food.food_query_normalization import analyze_food_query_for_search
from engines.food.food_reranker import (
    FoodSearchCandidate,
    build_food_candidate_dedupe_key,
    rerank_food_candidates,
)


def _doc(name, *, food_id=None, data_type="", rep_name="", protein=10.0):
    return SimpleNamespace(
        metadata={
            "food_id": food_id or name,
            "name": name,
            "rep_name": rep_name,
            "data_type": data_type,
            "protein_g": protein,
            "carb_g": 20.0,
            "fat_g": 3.0,
            "kcal": 147.0,
            "basis_value": 100.0,
            "basis_unit": "g",
        },
        page_content=name,
    )


def _candidate(document, rank):
    return FoodSearchCandidate(
        document=document,
        vector_score=0.1 + rank / 100,
        source_query="query",
        source_query_index=0,
        original_rank=rank,
        dedupe_key=build_food_candidate_dedupe_key(document),
    )


def _rerank_names(query, names):
    analysis = analyze_food_query_for_search(query)
    candidates = [_candidate(doc, rank) for rank, doc in enumerate(names)]
    return [
        candidate.document.metadata["name"]
        for candidate in rerank_food_candidates(analysis, candidates, final_k=len(candidates))
    ]


def test_reranker_prefers_ingredient_canonical_match_over_compound_food():
    names = _rerank_names(
        "계란",
        [
            _doc("계란빵", data_type="음식"),
            _doc("달걀 생것", data_type="원재료성 식품"),
        ],
    )

    assert names[0] == "달걀 생것"


def test_reranker_prefers_chicken_breast_canonical_match():
    names = _rerank_names(
        "닭가슴살 생것",
        [
            _doc("닭고기 토종 생것", data_type="원재료성 식품"),
            _doc("닭고기 가슴(껍질 제거) 생것", data_type="원재료성 식품"),
        ],
    )

    assert names[0] == "닭고기 가슴(껍질 제거) 생것"


@pytest.mark.parametrize(
    ("query", "candidate_names", "expected_first"),
    [
        ("삶은 계란", ["달걀 생것", "달걀 삶은것"], "달걀 삶은것"),
        (
            "삶은 닭가슴살",
            ["닭고기 가슴(껍질 제거) 생것", "닭고기 가슴(껍질 제거) 삶은것"],
            "닭고기 가슴(껍질 제거) 삶은것",
        ),
    ],
)
def test_reranker_prefers_cooking_state_match(query, candidate_names, expected_first):
    names = _rerank_names(
        query,
        [_doc(name, data_type="원재료성 식품") for name in candidate_names],
    )

    assert names[0] == expected_first


@pytest.mark.parametrize(
    ("query", "candidate_names", "expected_first"),
    [
        ("계란빵", ["달걀 생것", "계란빵"], "계란빵"),
        (
            "샐러드 닭가슴살",
            ["닭고기 가슴(껍질 제거) 생것", "샐러드 닭가슴살"],
            "샐러드 닭가슴살",
        ),
    ],
)
def test_reranker_preserves_protected_dish_exact_match(
    query,
    candidate_names,
    expected_first,
):
    names = _rerank_names(
        query,
        [_doc(name, data_type="음식") for name in candidate_names],
    )

    assert names[0] == expected_first


@pytest.mark.parametrize("query", ["닭가슴살", "닭가슴살 100g"])
def test_reranker_does_not_overcorrect_ambiguous_chicken_breast_query(query):
    names = _rerank_names(
        query,
        [
            _doc("샐러드 닭가슴살", data_type="음식"),
            _doc("닭고기 가슴(껍질 제거) 생것", data_type="원재료성 식품"),
        ],
    )

    assert names[0] == "샐러드 닭가슴살"


class _BadOrderVectorStore:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query
        self.queries: list[str] = []

    def similarity_search_with_score(self, query, k):
        self.queries.append(query)
        return self.results_by_query.get(query, [])


def _engine_with_vectorstore(vectorstore):
    engine = object.__new__(FoodSearchEngine)
    engine.available = True
    engine._vector_store = vectorstore
    engine._lock = threading.Lock()
    return engine


@pytest.mark.parametrize(
    ("query", "results_by_query", "expected_match"),
    [
        (
            "계란",
            {
                "달걀": [
                    (_doc("계란빵", data_type="음식"), 0.1),
                    (_doc("달걀 생것", data_type="원재료성 식품"), 0.2),
                ],
                "계란": [],
            },
            "달걀 생것",
        ),
        (
            "삶은 계란",
            {
                "달걀 삶은것": [
                    (_doc("계란빵", data_type="음식"), 0.1),
                    (_doc("땅콩 볶은것", data_type="원재료성 식품"), 0.2),
                    (_doc("달걀 삶은것", data_type="원재료성 식품"), 0.3),
                ],
                "삶은 계란": [],
            },
            "달걀 삶은것",
        ),
        (
            "계란빵",
            {
                "계란빵": [
                    (_doc("달걀 생것", data_type="원재료성 식품"), 0.1),
                    (_doc("계란빵", data_type="음식"), 0.2),
                ],
            },
            "계란빵",
        ),
        (
            "샐러드 닭가슴살",
            {
                "샐러드 닭가슴살": [
                    (_doc("닭고기 가슴(껍질 제거) 생것", data_type="원재료성 식품"), 0.1),
                    (_doc("샐러드 닭가슴살", data_type="음식"), 0.2),
                ],
            },
            "샐러드 닭가슴살",
        ),
    ],
)
def test_food_search_engine_applies_deterministic_reranker(
    query,
    results_by_query,
    expected_match,
):
    vectorstore = _BadOrderVectorStore(results_by_query)
    engine = _engine_with_vectorstore(vectorstore)

    result = engine.search(query, 100, "g")

    assert result is not None
    assert result["matched_name"] == expected_match

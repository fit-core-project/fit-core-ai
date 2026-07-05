from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

import engines.food.food_engine as food_engine
from engines.food.food_engine import (
    FOOD_SEARCH_FETCH_K,
    FOOD_SEARCH_FINAL_K,
    FoodSearchEngine,
)


class _VectorStore:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query
        self.calls: list[tuple[str, int]] = []

    def similarity_search_with_score(self, query, k):
        self.calls.append((query, k))
        return self.results_by_query.get(query, [])


def _doc(name, *, food_id=None, data_type="", protein=10.0):
    return SimpleNamespace(
        metadata={
            "food_id": food_id or name,
            "name": name,
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


def _engine_with_vectorstore(vectorstore):
    engine = object.__new__(FoodSearchEngine)
    engine.available = True
    engine._vector_store = vectorstore
    engine._canonical_index = None
    engine._lock = threading.Lock()
    return engine


def _results_with_target_after_previous_top_k(target_name):
    return [
        (_doc("계란빵", data_type="음식", protein=1.0), 0.1),
        (_doc("땅콩 삶은것", data_type="원재료성 식품", protein=2.0), 0.11),
        (_doc("메추리알 삶은것", data_type="원재료성 식품", protein=3.0), 0.12),
        (_doc(target_name, data_type="원재료성 식품", protein=30.0), 0.13),
        (_doc("달걀 생것", data_type="원재료성 식품", protein=4.0), 0.14),
    ]


def test_vectorstore_uses_fetch_k_for_alias_multi_query():
    vectorstore = _VectorStore(
        {
            "달걀 삶은것": [(_doc("달걀 삶은것"), 0.1)],
            "삶은 계란": [],
        }
    )
    engine = _engine_with_vectorstore(vectorstore)

    result = engine.search("삶은 계란", 100, "g")

    assert result is not None
    assert vectorstore.calls == [
        ("달걀 삶은것", FOOD_SEARCH_FETCH_K),
        ("삶은 계란", FOOD_SEARCH_FETCH_K),
    ]


def test_protected_query_uses_single_vector_call_with_fetch_k():
    vectorstore = _VectorStore({"계란빵": [(_doc("계란빵", data_type="음식"), 0.1)]})
    engine = _engine_with_vectorstore(vectorstore)

    result = engine.search("계란빵", 100, "g")

    assert result is not None
    assert result["matched_name"] == "계란빵"
    assert vectorstore.calls == [("계란빵", FOOD_SEARCH_FETCH_K)]


def test_reranker_receives_final_k_not_fetch_k(monkeypatch):
    observed_final_k = []
    real_reranker = food_engine.rerank_food_candidates

    def _spy_reranker(analysis, candidates, final_k):
        observed_final_k.append(final_k)
        return real_reranker(analysis, candidates, final_k)

    monkeypatch.setattr(food_engine, "rerank_food_candidates", _spy_reranker)
    vectorstore = _VectorStore(
        {"달걀 삶은것": _results_with_target_after_previous_top_k("달걀 삶은것"), "삶은 계란": []}
    )
    engine = _engine_with_vectorstore(vectorstore)

    result = engine.search("삶은 계란", 100, "g")

    assert result is not None
    assert observed_final_k == [FOOD_SEARCH_FINAL_K]
    assert FOOD_SEARCH_FETCH_K >= FOOD_SEARCH_FINAL_K


def test_candidate_beyond_previous_top_k_can_win_after_rerank():
    vectorstore = _VectorStore(
        {"달걀 삶은것": _results_with_target_after_previous_top_k("달걀 삶은것"), "삶은 계란": []}
    )
    engine = _engine_with_vectorstore(vectorstore)

    result = engine.search("삶은 계란", 100, "g")

    assert result is not None
    assert vectorstore.calls[0] == ("달걀 삶은것", FOOD_SEARCH_FETCH_K)
    assert result["matched_name"] == "달걀 삶은것"
    assert result["protein_g"] == 30.0


def test_dedupe_does_not_truncate_to_final_k_before_rerank():
    results = _results_with_target_after_previous_top_k("달걀 삶은것")
    vectorstore = _VectorStore({"달걀 삶은것": results, "삶은 계란": []})
    engine = _engine_with_vectorstore(vectorstore)

    result_groups = [results, []]
    candidates = engine._collect_candidates(result_groups, ["달걀 삶은것", "삶은 계란"])
    deduped = engine._dedupe_candidates(candidates)

    assert len(deduped) > FOOD_SEARCH_FINAL_K
    assert deduped[FOOD_SEARCH_FINAL_K].document.metadata["name"] == "달걀 삶은것"

    result = engine.search("삶은 계란", 100, "g")

    assert result is not None
    assert result["matched_name"] == "달걀 삶은것"

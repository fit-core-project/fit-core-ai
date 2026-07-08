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
    _canonical_match_score,
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


# ---------------------------------------------------------------------------
# _canonical_match_score: rep_name forward-only fix (2026-07-07)
# ---------------------------------------------------------------------------

def _score(query: str, name: str, rep_name: str) -> int:
    analysis = analyze_food_query_for_search(query)
    doc = _doc(name, rep_name=rep_name)
    candidate = _candidate(doc, rank=0)
    return _canonical_match_score(analysis, candidate)


@pytest.mark.parametrize(
    ("query", "name", "rep_name", "expected_score", "description"),
    [
        # [BUG FIX] rep_name "고구마" is a substring of query "고구마 조림" (reverse).
        # This was returning 2, causing raw-ingredient to outrank the correct dish.
        # Forward-only rep_name means rep_name ⊄ query → score=0.
        ("고구마 조림", "고구마 구운것", "고구마", 0, "rep_name reverse blocked"),
        # [PRESERVED] name "쌀밥" ⊂ query "흰쌀밥" — valid canonical equivalence.
        # name allows both directions, so this must still return 2.
        ("흰쌀밥", "쌀밥", "쌀밥", 2, "name reverse preserved (흰쌀밥→쌀밥)"),
        # [EXACT] normalized == name → 4
        ("달걀 삶은것", "달걀 삶은것", "달걀", 4, "exact name match"),
        # [COMPACT EXACT] whitespace-only difference is an exact name match.
        ("해물스파게티", "해물 스파게티", "해물 스파게티", 4, "compact exact name match"),
        # [COMPACT EXACT] whitespace-only difference is an exact rep_name match.
        ("해물스파게티", "치즈 스파게티 해물", "해물 스파게티", 3, "compact exact rep_name match"),
        # [PRESERVED] shorter substring candidate still scores 2, below compact exact 4.
        ("해물스파게티", "스파게티", "스파게티", 2, "shorter substring remains score 2"),
        # [PRESERVED] name reverse is kept for 흰쌀밥 → 쌀밥.
        ("흰쌀밥", "쌀밥", "쌀밥", 2, "name reverse preserved for 흰쌀밥"),
        # [PRESERVED] rep_name reverse is still blocked for 고구마 조림 → 고구마.
        ("고구마 조림", "고구마 구운것", "고구마", 0, "rep_name reverse remains blocked"),
        # [EXACT] normalized == rep_name → 3
        ("달걀", "달걀 생것", "달걀", 3, "exact rep_name match"),
        # [FORWARD name] normalized "닭가슴살" ⊂ name "샐러드 닭가슴살" → 2
        ("닭가슴살", "샐러드 닭가슴살", "샐러드", 2, "name forward: query in name"),
        # [FORWARD name] normalized "달걀" ⊂ name "달걀 삶은것" → 2
        ("달걀", "달걀 삶은것", "달걀", 3, "exact rep_name wins before forward name"),
        # [FORWARD rep_name] normalized "바나나 생것" contains rep_name "바나나" is false;
        # forward means normalized in rep_name. "바나나" in "바나나" is exact → already 3.
        # Demonstrate forward rep_name: "닭고기" in "닭고기 가슴(껍질 제거)" → 2
        ("닭고기", "닭고기 가슴(껍질 제거) 생것", "닭고기 가슴(껍질 제거)", 2, "rep_name forward: query in rep_name"),
        # [NO MATCH] neither name nor rep_name relates to query → 0
        ("오렌지", "사과 생것", "사과", 0, "no match"),
    ],
)
def test_canonical_match_score(query, name, rep_name, expected_score, description):
    assert _score(query, name, rep_name) == expected_score, description


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

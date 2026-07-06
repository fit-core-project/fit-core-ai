from __future__ import annotations

import os
import threading
from types import SimpleNamespace

import pytest

from engines.food.food_engine import FoodSearchEngine
from engines.food.food_query_normalization import normalize_food_query_for_search


EXPECTED_NORMALIZATION_CASES = [
    ("계란", "달걀"),
    ("계란 생것", "달걀 생것"),
    ("삶은 계란", "달걀 삶은것"),
    ("삶은계란", "달걀 삶은것"),
    ("계란 삶은것", "달걀 삶은것"),
    ("계란후라이", "달걀후라이"),
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
]

MUST_PRESERVE_CASES = [
    ("계란빵", "계란빵"),
    ("볶음밥 계란", "볶음밥 계란"),
    ("김밥 계란", "김밥 계란"),
    ("닭가슴살", "닭가슴살"),
    # quantity-strip: "닭가슴살 100g" -> "닭가슴살" (protected-ambiguous, not raw ingredient)
    ("닭가슴살 100g", "닭가슴살"),
    ("샐러드 닭가슴살", "샐러드 닭가슴살"),
    ("닭가슴살 샐러드", "닭가슴살 샐러드"),
    ("샌드위치 닭가슴살", "샌드위치 닭가슴살"),
]

WHITESPACE_AND_SAFETY_CASES = [
    (None, ""),
    ("", ""),
    ("  삶은   계란  ", "달걀 삶은것"),
    ("  닭가슴살   생것  ", "닭고기 가슴(껍질 제거) 생것"),
    ("  계란빵  ", "계란빵"),
    ("  샐러드   닭가슴살  ", "샐러드 닭가슴살"),
]

MUST_NOT_NORMALIZE_TO_CASES = [
    ("계란빵", "달걀"),
    ("계란빵", "달걀 생것"),
    ("볶음밥 계란", "볶음밥 달걀"),
    ("김밥 계란", "김밥 달걀"),
    ("닭가슴살", "닭고기 가슴(껍질 제거) 생것"),
    ("닭가슴살 100g", "닭고기 가슴(껍질 제거) 생것"),
    ("샐러드 닭가슴살", "닭고기 가슴(껍질 제거) 생것"),
    ("닭가슴살 샐러드", "닭고기 가슴(껍질 제거) 생것"),
    ("샌드위치 닭가슴살", "닭고기 가슴(껍질 제거) 생것"),
]


@pytest.mark.parametrize(("query", "expected"), EXPECTED_NORMALIZATION_CASES)
def test_food_search_expected_normalization_golden_queries(query, expected):
    assert normalize_food_query_for_search(query) == expected


@pytest.mark.parametrize(("query", "expected"), MUST_PRESERVE_CASES)
def test_food_search_must_preserve_golden_queries(query, expected):
    assert normalize_food_query_for_search(query) == expected


@pytest.mark.parametrize(("query", "expected"), WHITESPACE_AND_SAFETY_CASES)
def test_food_search_whitespace_and_safety_golden_queries(query, expected):
    assert normalize_food_query_for_search(query) == expected


@pytest.mark.parametrize(("query", "blocked"), MUST_NOT_NORMALIZE_TO_CASES)
def test_food_search_must_not_normalize_to_regression_queries(query, blocked):
    assert normalize_food_query_for_search(query) != blocked


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
        ("닭가슴살 생것", ["닭고기 가슴(껍질 제거) 생것", "닭가슴살 생것"]),
    ],
)
def test_food_search_engine_uses_golden_normalized_query_for_vectorstore(
    raw_query,
    expected_query,
):
    engine = _engine_with_fake_vectorstore()

    result = engine.search(raw_query, 100, "g")

    assert result is not None
    expected_queries = (
        expected_query if isinstance(expected_query, list) else [expected_query]
    )
    assert engine._vector_store.queries == expected_queries


@pytest.mark.chroma
@pytest.mark.smoke
@pytest.mark.skipif(
    os.getenv("RUN_FOOD_SEARCH_SMOKE") != "1",
    reason="Set RUN_FOOD_SEARCH_SMOKE=1 to run local Chroma food search smoke.",
)
@pytest.mark.parametrize(
    ("raw_query", "expected_normalized", "expected_top1"),
    [
        ("계란", "달걀", "달걀 생것"),
        ("삶은 계란", "달걀 삶은것", "달걀 삶은것"),
        ("계란후라이", "달걀후라이", "달걀후라이"),
        ("계란빵", "계란빵", "계란빵"),
        ("닭가슴살 생것", "닭고기 가슴(껍질 제거) 생것", "닭고기 가슴(껍질 제거) 생것"),
        ("삶은 닭가슴살", "닭고기 가슴(껍질 제거) 삶은것", "닭고기 가슴(껍질 제거) 삶은것"),
        ("구운 닭가슴살", "닭고기 가슴(껍질 제거) 구운것(팬)", "닭고기 가슴(껍질 제거) 구운것(팬)"),
        ("샐러드 닭가슴살", "샐러드 닭가슴살", "샐러드 닭가슴살"),
    ],
)
def test_food_search_rank_smoke_opt_in(raw_query, expected_normalized, expected_top1):
    from engines.food.food_engine import get_food_engine

    engine = get_food_engine()
    assert engine.available is True
    assert engine._vector_store is not None

    normalized = normalize_food_query_for_search(raw_query)
    assert normalized == expected_normalized

    results = engine._vector_store.similarity_search_with_score(normalized, k=3)
    top1 = (results[0][0].metadata or {}).get("name") if results else None

    assert top1 == expected_top1


# ---------------------------------------------------------------------------
# Full-pipeline smoke: engine.search() → canonical + reranker 포함 E2E
# ---------------------------------------------------------------------------
# 기존 test_food_search_rank_smoke_opt_in 은 normalization + vector 직접 호출만 검증.
# 아래 두 테스트는 전체 4단계 파이프라인(normalization→canonical→vector→reranker)을
# engine.search() 한 번으로 검증한다.

# 부류 A — 현재 올바른 동작 → 회귀 방지 고정 (2026-07-06 진단 기준)
_FULL_PIPELINE_REGRESSION_CASES = [
    # 닭가슴살은 의도적 ambiguous 보존 — 원재료로 강제되지 않는 것이 올바른 동작
    pytest.param("닭가슴살", "샐러드 닭가슴살", id="닭가슴살-ambiguous-preserved"),
    # 흰쌀밥 → 쌀밥: DB canonical 동치, 현재 올바른 매핑
    pytest.param("흰쌀밥", "쌀밥", id="흰쌀밥-canonical-equivalent"),
    # 계란 2개 → quantity strip → 계란 → 달걀 alias → canonical → 달걀 생것 (PR#1 승격)
    pytest.param("계란 2개", "달걀 생것", id="계란2개-quantity-strip-promoted"),
    # 바나나 1개 → quantity strip → 바나나 → canonical rep_name → 바나나 생것 (PR#1 승격)
    pytest.param("바나나 1개", "바나나 생것", id="바나나1개-quantity-strip-promoted"),
]

# 부류 B — 현재 틀린 동작 → xfail 개선 목표
# strict=True: 런타임 개선 후 통과하면 XPASS로 신호 발생
# 4/6 건(두부·오트밀·삶은두부·볶은두부)은 DB coverage gap이 전제 — pipeline fix 전에 DB 확장 필요
_FULL_PIPELINE_IMPROVEMENT_CASES = [
    # [정규화 레이어] 닭가슴살 100g → 닭가슴살 (protected-ambiguous) → 샐러드 닭가슴살
    # 목표: 수량 strip 후 ingredient 재분류 + 생것 alias 경로 필요 (별도 PR)
    pytest.param(
        "닭가슴살 100g", "닭고기 가슴(껍질 제거) 생것",
        marks=pytest.mark.xfail(strict=True, reason="normalization: 100g stripped to 닭가슴살 but protected-ambiguous; ingredient re-alias path not yet implemented"),
        id="닭가슴살100g-quantity-strip",
    ),
    # [reranker 레이어] cooking-state score가 vector top-1을 역전
    pytest.param(
        "고구마 조림", "고구마조림",
        marks=pytest.mark.xfail(strict=True, reason="reranker: cooking-state score demotes vector top-1 (고구마조림 dist=0.056) in favour of 고구마 구운것"),
        id="고구마조림-reranker",
    ),
    # [DB coverage gap] 원재료성 두부가 DB에 없음 + pipeline도 두부국(음식) 반환
    pytest.param(
        "두부", "두부",
        marks=pytest.mark.xfail(strict=True, reason="DB coverage gap: raw 두부(원재료) not in food_db; pipeline returns dish 두부국"),
        id="두부-db-gap",
    ),
    # [DB coverage gap] 원재료성 오트밀이 DB에 없음
    pytest.param(
        "오트밀", "오트밀",
        marks=pytest.mark.xfail(strict=True, reason="DB coverage gap: raw 오트밀 not in food_db; pipeline returns 라떼 오트밀 라떼 핫(HOT)"),
        id="오트밀-db-gap",
    ),
    # [DB coverage gap] 두부 삶은것이 DB에 없음 + reranker가 동부 삶은것 반환
    pytest.param(
        "삶은 두부", "두부 삶은것",
        marks=pytest.mark.xfail(strict=True, reason="DB coverage gap: 두부 삶은것 not in food_db; reranker returns 동부 삶은것"),
        id="삶은두부-db-gap",
    ),
    # [DB coverage gap] 두부볶음(단독)이 DB에 없고 두부볶음 돼지고기만 존재
    pytest.param(
        "볶은 두부", "두부볶음",
        marks=pytest.mark.xfail(strict=True, reason="DB coverage gap: plain 두부볶음 not in food_db (only 두부볶음 돼지고기); pipeline fix also needed"),
        id="볶은두부-db-gap",
    ),
]


@pytest.mark.chroma
@pytest.mark.smoke
@pytest.mark.skipif(
    os.getenv("RUN_FOOD_SEARCH_SMOKE") != "1",
    reason="Set RUN_FOOD_SEARCH_SMOKE=1 to run local Chroma food search smoke.",
)
@pytest.mark.parametrize(("raw_query", "expected_top1"), _FULL_PIPELINE_REGRESSION_CASES)
def test_food_search_full_pipeline_regression(raw_query, expected_top1):
    from engines.food.food_engine import get_food_engine

    engine = get_food_engine()
    assert engine.available is True
    result = engine.search(raw_query, 100, "g")
    matched = result.get("matched_name") if result else None
    assert matched == expected_top1


@pytest.mark.chroma
@pytest.mark.smoke
@pytest.mark.skipif(
    os.getenv("RUN_FOOD_SEARCH_SMOKE") != "1",
    reason="Set RUN_FOOD_SEARCH_SMOKE=1 to run local Chroma food search smoke.",
)
@pytest.mark.parametrize(("raw_query", "expected_top1"), _FULL_PIPELINE_IMPROVEMENT_CASES)
def test_food_search_full_pipeline_improvement_targets(raw_query, expected_top1):
    from engines.food.food_engine import get_food_engine

    engine = get_food_engine()
    assert engine.available is True
    result = engine.search(raw_query, 100, "g")
    matched = result.get("matched_name") if result else None
    assert matched == expected_top1

from __future__ import annotations

import pytest

from engines.food.food_query_normalization import (
    analyze_food_query_for_search,
    clean_food_search_query,
    normalize_food_query_for_search,
)
from tests.test_food_search_golden_queries import (
    EXPECTED_NORMALIZATION_CASES,
    MUST_PRESERVE_CASES,
    WHITESPACE_AND_SAFETY_CASES,
)


@pytest.mark.parametrize(
    "query",
    [
        *[query for query, _ in EXPECTED_NORMALIZATION_CASES],
        *[query for query, _ in MUST_PRESERVE_CASES],
        *[query for query, _ in WHITESPACE_AND_SAFETY_CASES],
    ],
)
def test_analysis_normalized_query_matches_public_normalizer(query):
    analysis = analyze_food_query_for_search(query)

    assert analysis.normalized_query == normalize_food_query_for_search(query)


@pytest.mark.parametrize(
    ("query", "expected_cleaned", "expected_normalized"),
    [
        (None, "", ""),
        ("", "", ""),
        ("  삶은   계란  ", "삶은 계란", "달걀 삶은것"),
        ("  샐러드   닭가슴살  ", "샐러드 닭가슴살", "샐러드 닭가슴살"),
    ],
)
def test_analysis_cleans_query_without_changing_normalization_contract(
    query,
    expected_cleaned,
    expected_normalized,
):
    analysis = analyze_food_query_for_search(query)

    assert clean_food_search_query(query) == expected_cleaned
    assert analysis.cleaned_query == expected_cleaned
    assert analysis.normalized_query == expected_normalized


@pytest.mark.parametrize(
    ("query", "dish_keyword", "intent"),
    [
        ("계란빵", "빵", "dish"),
        ("볶음밥 계란", "볶음밥", "dish"),
        ("김밥 계란", "김밥", "dish"),
        ("닭가슴살", None, "ambiguous"),
        ("샐러드 닭가슴살", "샐러드", "dish"),
        ("닭가슴살 샐러드", "샐러드", "dish"),
        ("샌드위치 닭가슴살", "샌드위치", "dish"),
    ],
)
def test_analysis_marks_protected_queries_without_aliasing(query, dish_keyword, intent):
    analysis = analyze_food_query_for_search(query)

    assert analysis.protected is True
    assert analysis.normalized_query == query
    assert analysis.intent == intent
    assert analysis.alias_applied is False
    assert analysis.matched_dish_keyword == dish_keyword


def test_analysis_quantity_bearing_protected_query_strips_to_base():
    # "닭가슴살 100g" strips to "닭가슴살" which is protected-ambiguous.
    # normalized_query reflects the stripped form, not the original.
    analysis = analyze_food_query_for_search("닭가슴살 100g")

    assert analysis.protected is True
    assert analysis.intent == "ambiguous"
    assert analysis.cleaned_query == "닭가슴살"
    assert analysis.normalized_query == "닭가슴살"
    assert analysis.alias_applied is False


@pytest.mark.parametrize(
    ("query", "expected_normalized", "cooking_state", "intent"),
    [
        ("계란", "달걀", None, "ingredient"),
        ("삶은 계란", "달걀 삶은것", "삶은것", "ingredient"),
        ("계란후라이", "달걀후라이", "후라이", "dish"),
        ("닭가슴살 생것", "닭고기 가슴(껍질 제거) 생것", "생것", "ingredient"),
        ("삶은 닭가슴살", "닭고기 가슴(껍질 제거) 삶은것", "삶은것", "ingredient"),
        ("구운 닭가슴살", "닭고기 가슴(껍질 제거) 구운것(팬)", "구운것", "ingredient"),
    ],
)
def test_analysis_marks_alias_applied_queries(
    query,
    expected_normalized,
    cooking_state,
    intent,
):
    analysis = analyze_food_query_for_search(query)

    assert analysis.protected is False
    assert analysis.alias_applied is True
    assert analysis.normalized_query == expected_normalized
    assert analysis.cooking_state == cooking_state
    assert analysis.intent == intent


@pytest.mark.parametrize("query", ["바나나", "고구마"])
def test_analysis_keeps_unknown_food_queries_ambiguous(query):
    analysis = analyze_food_query_for_search(query)

    assert analysis.normalized_query == query
    assert analysis.protected is False
    assert analysis.alias_applied is False
    assert analysis.intent == "ambiguous"

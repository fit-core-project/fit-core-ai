"""Unit tests for the general food lookup rules (composite variant, name tail, jamo spelling variants, 계란/달걀, lexical alias index, 빙과류 piece weight, English words)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from engines.food import curated_products as cp
from engines.food import food_reranker as fr
from engines.food.food_canonical_index import FoodCanonicalIndex
from engines.food.food_query_normalization import (
    analyze_food_query_for_search,
    build_food_search_queries,
    translate_english_food_name as tr,
)
from engines.food.food_reranker import (
    FoodSearchCandidate,
    build_food_candidate_dedupe_key,
    has_lexical_relation,
    is_implausible_food_candidate,
    rerank_food_candidates,
)


def _cand(name, *, rep_name="", data_type="음식", rank=0):
    doc = SimpleNamespace(
        metadata={"food_id": name, "name": name, "rep_name": rep_name, "data_type": data_type,
                  "kcal": 300.0, "protein_g": 5.0, "carb_g": 40.0, "fat_g": 10.0,
                  "basis_value": 100.0, "basis_unit": "g"},
        page_content=name,
    )
    return FoodSearchCandidate(document=doc, vector_score=0.1 + rank / 100, source_query="q",
                               source_query_index=0, original_rank=rank,
                               dedupe_key=build_food_candidate_dedupe_key(doc))


@pytest.fixture
def vocab(monkeypatch):
    """A fixed DB vocabulary: 치킨 heads standalone rows (닭튀김 후라이드 치킨), 브라우니 does not."""
    fixed = fr._FoodNameVocab(standalone_heads=frozenset({"치킨", "마카롱", "떡볶이"}),
                              rep_names=frozenset({"케이크", "햄버거", "마카롱", "떡볶이"}))
    monkeypatch.setattr(fr, "_food_name_vocab", lambda: fixed)
    return fixed


# ---------- composite-dish variant word (케이크 브라우니 for 브라우니) ----------

def test_composite_row_is_plausible_when_the_query_is_only_its_variant_word(vocab):
    analysis = analyze_food_query_for_search("브라우니")
    assert not is_implausible_food_candidate(analysis, _cand("케이크 브라우니", rep_name="케이크"))


def test_composite_row_stays_implausible_when_the_query_is_a_standalone_food(vocab):
    # 치킨 is a food on its own (닭튀김 ... 치킨): 햄버거 치킨 is a burger
    analysis = analyze_food_query_for_search("치킨")
    assert is_implausible_food_candidate(analysis, _cand("햄버거 치킨", rep_name="햄버거"))


def test_composite_variant_rule_is_off_without_the_db_vocabulary(monkeypatch):
    monkeypatch.setattr(fr, "_food_name_vocab", lambda: None)
    analysis = analyze_food_query_for_search("브라우니")
    assert is_implausible_food_candidate(analysis, _cand("케이크 브라우니", rep_name="케이크"))


# ---------- multi-word query that is the tail of the row name ----------

def test_multi_word_query_matching_the_name_tail_is_plausible(vocab):
    analysis = analyze_food_query_for_search("레드벨벳 케이크")
    assert not is_implausible_food_candidate(analysis, _cand("케이크 레드벨벳 케이크", rep_name="케이크"))


def test_multi_word_query_that_is_not_the_name_tail_is_not_rescued(vocab):
    analysis = analyze_food_query_for_search("레드벨벳 케이크")
    candidate = _cand("마카롱 레드벨벳 케이크 마카롱", rep_name="마카롱")
    assert not fr._query_is_name_tail(analysis, candidate)


def test_name_tail_ignores_size_and_temperature_markers():
    analysis = analyze_food_query_for_search("토마토 주스")
    assert fr._query_is_name_tail(analysis, _cand("과일주스 토마토 주스 아이스 L", rep_name="주스"))


# ---------- lexical relation: jamo spelling variants and 계란/달걀 ----------

@pytest.mark.parametrize("query, name", [
    ("배추젼", "배추전"), ("현미쥭", "현미죽"), ("깐쇼새오", "깐쇼새우"), ("계란국", "달걀국"),
    ("계란찜", "달걀찜 채소"),
])
def test_spelling_variants_and_egg_synonym_are_lexically_related(query, name):
    assert has_lexical_relation(query, name)


@pytest.mark.parametrize("query, name", [("소주", "소금"), ("맥주", "버거 맥치킨 버거"), ("콜라", "치커리")])
def test_short_or_unrelated_words_stay_unrelated(query, name):
    assert not has_lexical_relation(query, name)


def test_egg_query_also_searches_the_dalgyal_spelling():
    queries = build_food_search_queries("계란찜")
    assert queries[0] == "계란찜"
    assert "달걀찜" in queries


# ---------- lexical alias index ----------

def _index():
    rows = [
        {"food_id": "1", "name": "완자전 소고기(동그랑땡/육원전)", "rep_name": "완자전", "data_type": "음식"},
        {"food_id": "2", "name": "귤 한라봉 생것", "rep_name": "귤", "data_type": "원재료성 식품",
         "major_category": "과일류"},
        {"food_id": "3", "name": "돼지고기 삼겹살(삼겹살) 생것", "rep_name": "돼지고기",
         "data_type": "원재료성 식품", "major_category": "육류"},
        {"food_id": "4", "name": "옥수수 찰옥수수 삶은것", "rep_name": "옥수수", "data_type": "원재료성 식품",
         "major_category": "곡류"},
    ]
    return FoodCanonicalIndex.from_rows(rows)


def test_lexical_index_finds_dish_aliases_and_raw_produce_varieties():
    index = _index()
    assert [r.metadata["name"] for r in index.lookup_lexical("동그랑땡")] == ["완자전 소고기(동그랑땡/육원전)"]
    assert [r.metadata["name"] for r in index.lookup_lexical("한라봉")] == ["귤 한라봉 생것"]


def test_lexical_index_skips_raw_meat_aliases_and_cooked_grain_rows():
    index = _index()
    assert index.lookup_lexical("삼겹살") == []  # a bare 삼겹살 is eaten cooked
    assert index.lookup_lexical("찰옥수수") == []


# ---------- product piece weight ----------

def _product(category):
    return cp.IndexedProduct(key="하겐다즈미니컵딸기", name="하겐다즈 미니컵 딸기", kcal_100=220.0,
                             protein_100=4.0, carbs_100=20.0, fat_100=14.0, unit_g=100.0,
                             package_sizes=(100.0,), n_rows=1, category_lv3=category)


def test_ice_cream_product_piece_is_the_cup_not_the_fruit_weight():
    assert cp._indexed_grams(_product("빙과류"), "하겐다즈 미니컵 딸기", 1.0, "개") == pytest.approx(100.0)


def test_non_ice_cream_product_keeps_the_food_piece_weight():
    assert cp._indexed_grams(_product("과자류·빵류 또는 떡류"), "딸기", 1.0, "개") == pytest.approx(15.0)


# ---------- English food words ----------

@pytest.mark.parametrize("english, korean", [
    ("venti cold brew", "콜드브루"), ("mango smoothie", "망고스무디"), ("yogurt smoothie", "요거트스무디"),
    ("carrot cake", "당근케이크"), ("raisins", "포도 말린것"), ("iced americano", "아메리카노"),
    ("strawberry latte", "딸기라떼"),
])
def test_common_english_food_words_translate(english, korean):
    assert tr(english) == korean


@pytest.mark.parametrize("english", ["iced", "milk tea", "toowoomba pasta", "chocolate milk"])
def test_size_words_alone_and_non_composing_compounds_do_not_translate(english):
    assert tr(english) is None


# ---------- review regressions ----------

@pytest.fixture
def food_vocab(monkeypatch):
    """우유 / 베이컨 / 요거트 head standalone rows, so compounds ending in them are those foods."""
    fixed = fr._FoodNameVocab(standalone_heads=frozenset({"우유", "베이컨", "요거트", "치킨"}),
                              rep_names=frozenset({"케이크", "피자", "크로와상", "우유"}))
    monkeypatch.setattr(fr, "_food_name_vocab", lambda: fixed)
    return fixed


@pytest.mark.parametrize("query, name, rep", [
    ("딸기우유", "크로와상 딸기우유 크로와상", "크로와상"),
    ("감자베이컨", "피자 감자베이컨 피자 (L)", "피자"),
    ("생딸기요거트", "케이크 생딸기요거트 케이크", "케이크"),
])
def test_compound_whose_head_is_a_food_is_not_a_dessert_variant(food_vocab, query, name, rep):
    analysis = analyze_food_query_for_search(query)
    assert not fr._names_composite_variant(analysis, _cand(name, rep_name=rep))


def test_word_between_repeated_category_words_is_a_flavour_not_the_item(vocab):
    # 마카롱 브라우니 마카롱 is a brownie-flavoured macaron; 케이크 브라우니 is the brownie
    analysis = analyze_food_query_for_search("브라우니")
    assert not fr._names_composite_variant(analysis, _cand("마카롱 브라우니 마카롱", rep_name="마카롱"))
    assert fr._names_composite_variant(analysis, _cand("케이크 브라우니", rep_name="케이크"))


def test_drink_word_is_not_a_cafe_row_variant(vocab):
    analysis = analyze_food_query_for_search("쉐이크")
    assert not fr._names_composite_variant(analysis, _cand("스무디 커피 쉐이크", rep_name="스무디"))


@pytest.mark.parametrize("query, name", [
    ("녹두죽", "녹두묵"), ("미역죽", "미역국"), ("칼국수", "쌀국수"), ("새우깡", "새우탕"),
])
def test_a_changed_consonant_is_a_different_dish(query, name):
    assert not has_lexical_relation(query, name)


def test_ice_cream_named_after_its_piece_keeps_the_piece_weight(monkeypatch):
    monkeypatch.setattr(cp, "_produce_words", lambda: frozenset({"딸기"}))
    product = cp.IndexedProduct(key="미니아이스마카롱세트", name="미니아이스마카롱세트", kcal_100=293.0,
                                protein_100=4.0, carbs_100=40.0, fat_100=12.0, unit_g=200.0,
                                package_sizes=(200.0,), n_rows=1, category_lv3="빙과류")
    assert cp._indexed_grams(product, "미니 아이스 마카롱", 5.0, "개") == pytest.approx(100.0)
    assert cp._indexed_grams(_product("빙과류"), "하겐다즈 미니컵 딸기", 1.0, "개") == pytest.approx(100.0)


@pytest.mark.parametrize("english, korean", [
    ("hot chocolate", "핫초코"), ("hot cocoa", "핫초코"), ("iced tea", "아이스티"), ("ice cream", "아이스크림"),
])
def test_temperature_words_that_are_part_of_the_food_translate_as_phrases(english, korean):
    assert tr(english) == korean


@pytest.mark.parametrize("english", ["small pizza", "large pizza", "hot strawberry"])
def test_size_or_temperature_word_on_a_non_drink_does_not_translate(english):
    assert tr(english) is None


def test_lexical_index_skips_branded_menu_parentheticals_and_qualifier_words():
    rows = [
        {"food_id": "1", "name": "피자 차슈(불갈비) 피자 (M)", "rep_name": "피자", "data_type": "음식"},
        {"food_id": "2", "name": "케이크 우유듬뿍생크림 케이크(선샤인) 6호", "rep_name": "케이크", "data_type": "음식"},
        {"food_id": "3", "name": "효모빵 쫄깃한 천연효모빵 (미니)", "rep_name": "효모빵", "data_type": "음식"},
        {"food_id": "4", "name": "모싯잎송편 동부(국내산)", "rep_name": "모싯잎송편", "data_type": "음식"},
        {"food_id": "5", "name": "제육(돼지고기 수육)", "rep_name": "제육", "data_type": "음식"},
        {"food_id": "6", "name": "파프리카 빨간색 생것", "rep_name": "파프리카", "data_type": "원재료성 식품",
         "major_category": "채소류"},
        {"food_id": "7", "name": "토스트(식빵)", "rep_name": "토스트", "data_type": "음식"},
        {"food_id": "8", "name": "식빵", "rep_name": "식빵", "data_type": "음식"},
        {"food_id": "9", "name": "개피떡(바람떡)", "rep_name": "개피떡", "data_type": "음식"},
    ]
    index = FoodCanonicalIndex.from_rows(rows)
    for key in ("불갈비", "선샤인", "미니", "국내산", "돼지고기", "빨간색", "식빵"):
        assert index.lookup_lexical(key) == [], key
    assert [r.metadata["name"] for r in index.lookup_lexical("돼지고기 수육")] == ["제육(돼지고기 수육)"]
    assert [r.metadata["name"] for r in index.lookup_lexical("바람떡")] == ["개피떡(바람떡)"]


def test_lexical_candidates_carry_their_real_embedding_distance():
    import threading
    from engines.food.food_engine import FoodSearchEngine

    index = _index()
    calls = []

    class _Store:
        def similarity_search_with_score(self, query, k, filter=None):
            calls.append(filter)
            return [(index.lookup_lexical("동그랑땡")[0].to_document(), 0.52)]

    engine = FoodSearchEngine.__new__(FoodSearchEngine)
    engine._canonical_index, engine._vector_store, engine._lock = index, _Store(), threading.Lock()
    candidates = engine._lookup_lexical_candidates(analyze_food_query_for_search("동그랑땡"))
    assert [c.vector_score for c in candidates] == [pytest.approx(0.52)]
    assert calls == [{"food_id": {"$in": ["1"]}}]


def test_lexical_candidates_are_dropped_when_they_cannot_be_scored():
    import threading
    from engines.food.food_engine import FoodSearchEngine

    class _Store:
        def similarity_search_with_score(self, *args, **kwargs):
            raise RuntimeError("no filter support")

    engine = FoodSearchEngine.__new__(FoodSearchEngine)
    engine._canonical_index, engine._vector_store, engine._lock = _index(), _Store(), threading.Lock()
    assert engine._lookup_lexical_candidates(analyze_food_query_for_search("동그랑땡")) == []


def test_compound_ending_in_a_loanword_or_raw_food_head_is_not_a_variant(monkeypatch):
    fixed = fr._FoodNameVocab(standalone_heads=frozenset(), rep_names=frozenset({"샌드위치"}),
                              raw_rep_names=frozenset({"달걀"}))
    monkeypatch.setattr(fr, "_food_name_vocab", lambda: fixed)
    analysis = analyze_food_query_for_search("햄에그")
    assert not fr._names_composite_variant(analysis, _cand("샌드위치 햄에그 파니니", rep_name="샌드위치"))

"""Regression tests for diet-logging calorie accuracy (in-process, no LLM/Ollama).

Covers: substring false matches in the reranker (콜라 -> 쇼콜라 케이크,
치킨 -> 햄버거 치킨), piece/serving units, per-portion prompt macros, LLM macro
sanity checks, word-end particle stripping and food extraction from chatter.
"""
from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest
from langchain_core.runnables import RunnableLambda

from engines.food.food_engine import FoodSearchEngine
from engines.food.food_query_normalization import analyze_food_query_for_search
from engines.food.food_reranker import (
    FoodSearchCandidate,
    build_food_candidate_dedupe_key,
    rerank_food_candidates,
)
from engines.food.food_units import resolve_portion_grams
from engines.quicklog import diet_parser
from engines.quicklog.diet_parser import (
    DietParseItem,
    ParsedDietLog,
    _build_diet_prompt,
    _deterministic_diet_parse,
    _enrich_macros,
    _reconcile_amounts_with_text,
    _strip_context_words,
)


def _doc(name, rep_name="", *, kcal=147.0, protein=10.0, carb=20.0, fat=3.0,
         basis_unit="g", data_type="음식", food_id=None):
    return SimpleNamespace(
        metadata={
            "food_id": food_id or f"{name}|{basis_unit}",
            "name": name,
            "rep_name": rep_name,
            "data_type": data_type,
            "protein_g": protein,
            "carb_g": carb,
            "fat_g": fat,
            "kcal": kcal,
            "basis_value": 100.0,
            "basis_unit": basis_unit,
        },
        page_content=name,
    )


def _rerank(query, docs):
    analysis = analyze_food_query_for_search(query)
    candidates = [
        FoodSearchCandidate(
            document=doc,
            vector_score=0.3,
            source_query=query,
            source_query_index=0,
            original_rank=rank,
            dedupe_key=build_food_candidate_dedupe_key(doc),
        )
        for rank, doc in enumerate(docs)
    ]
    ranked = rerank_food_candidates(analysis, candidates, final_k=len(candidates))
    return [c.document.metadata["name"] for c in ranked]


class _FixedVectorStore:
    def __init__(self, docs, distance=0.3):
        self.docs = docs
        self.distance = distance

    def similarity_search_with_score(self, query, k):
        return [(doc, self.distance) for doc in self.docs[:k]]


def _engine(docs, distance=0.3):
    engine = object.__new__(FoodSearchEngine)
    engine.available = True
    engine._vector_store = _FixedVectorStore(docs, distance)
    engine._canonical_index = None
    engine._lock = threading.Lock()
    return engine


# ---------- 1. reranker substring false matches ----------

_COLA_LOOKALIKES = [
    _doc("케이크 쇼콜라 케이크", "케이크", kcal=376.0, protein=6.2),
    _doc("아이스크림 트로피컬 콜라다", "아이스크림", kcal=110.0),
    _doc("케이크 클래식쇼콜라", "케이크", kcal=429.0),
    _doc("콜라비 생것", "콜라비", kcal=22.0, data_type="원재료성 식품"),
]

_CHICKEN_LOOKALIKES = [
    _doc("햄버거 치킨", "햄버거", kcal=123.0),
    _doc("버거 치킨 버거", "버거", kcal=277.0),
    _doc("치킨가스", "치킨가스", kcal=296.0),
    _doc("피자 치킨 피자", "피자", kcal=139.0),
    _doc("치커리 치콘 생것", "치커리", kcal=15.0, data_type="원재료성 식품"),
]


def test_reranker_rejects_cola_inside_unrelated_dish_names():
    assert _rerank("콜라", _COLA_LOOKALIKES) == []


def test_reranker_rejects_chicken_as_trailing_modifier_of_other_dish():
    assert _rerank("치킨", _CHICKEN_LOOKALIKES) == []


def test_engine_returns_no_db_hit_for_cola_instead_of_chocolate_cake():
    assert _engine(_COLA_LOOKALIKES).search("콜라", 350, "ml") is None
    assert _engine(_COLA_LOOKALIKES).search("콜라", 1, "캔") is None


def test_engine_returns_no_db_hit_for_chicken_pieces_instead_of_burger():
    assert _engine(_CHICKEN_LOOKALIKES).search("치킨", 2, "조각") is None


def test_reranker_keeps_leading_token_match():
    names = _rerank("라면", [_doc("햄버거 라면", "햄버거"), _doc("라면 치즈", "라면"), _doc("라면", "라면")])
    assert names[0] == "라면"
    assert "햄버거 라면" not in names


def test_reranker_keeps_protected_chicken_breast_salad_mapping():
    names = _rerank("닭가슴살", [_doc("샐러드 닭가슴살", "샐러드"), _doc("닭튀김 닭가슴살", "닭튀김")])
    assert names[0] == "샐러드 닭가슴살"


def test_reranker_keeps_beverage_category_variant():
    names = _rerank("아메리카노", [_doc("케이크 아메리카노 케이크", "케이크"), _doc("커피 아메리카노", "커피", kcal=4.0)])
    assert names == ["커피 아메리카노"]


def test_reranker_keeps_head_noun_reverse_match():
    assert _rerank("흰쌀밥", [_doc("쌀밥", "쌀밥")]) == ["쌀밥"]


# ---------- 2. piece / serving units ----------

@pytest.mark.parametrize(
    ("food", "amount", "unit", "grams"),
    [
        ("라면", 1, "개", 600.0),
        ("라면", 1, "봉지", 600.0),
        ("계란", 2, "개", 100.0),
        ("계란 2개", 2, "개", 100.0),
        ("삶은 계란", 2, "개", 100.0),
        ("계란", 2, None, 100.0),        # bare count = pieces, not 2 g
        ("계란", 2, "", 100.0),
        ("치킨", 2, "조각", 200.0),
        ("양념치킨", 2, "조각", 200.0),  # head-noun suffix override
        ("콜라", 1, "캔", 355.0),
        ("콜라", 350, "ml", 350.0),
        ("바나나", 1, "개", 120.0),
        ("사과", 1, "개", 200.0),
        ("쌀밥", 1, "공기", 210.0),
        ("고구마", 150, "g", 150.0),
        ("닭가슴살", 200, "g", 200.0),
        ("두부", 1, "모", 300.0),
        ("모르는음식", 2, "조각", 200.0),  # generic per-unit fallback × count
        ("모르는음식", 3, "개", 300.0),
        ("모르는음식", 1, "그릇", 400.0),
        ("모르는음식", 1, "캔", 355.0),
        ("모르는음식", 250, None, 250.0),  # large bare number = grams
    ],
)
def test_resolve_portion_grams(food, amount, unit, grams):
    assert resolve_portion_grams(food, amount, unit) == pytest.approx(grams)


def test_ramen_one_pack_is_about_500_kcal_not_100g():
    engine = _engine([_doc("라면", "라면", kcal=82.0, protein=1.7, carb=11.0, fat=3.4)], distance=0.0)
    hit = engine.search("라면", 1, "개")
    assert 400.0 <= hit["kcal"] <= 600.0
    assert hit["resolved_grams"] == 600.0


def test_eggs_two_pieces_are_100g():
    engine = _engine([_doc("달걀 생것", "달걀", kcal=136.0, protein=12.4, data_type="원재료성 식품")], distance=0.0)
    for unit in ("개", None):
        hit = engine.search("계란", 2, unit)
        assert hit["resolved_grams"] == 100.0
        assert hit["kcal"] == pytest.approx(136.0)


def test_engine_prefers_gram_basis_row_for_count_units():
    engine = _engine(
        [_doc("라면", "라면", kcal=124.0, basis_unit="ml"), _doc("라면", "라면", kcal=82.0, basis_unit="g")],
        distance=0.0,
    )
    assert engine.search("라면", 1, "개")["basis_unit"] == "g"


# ---------- 3. prompt + LLM macro sanity ----------

def test_prompt_example_uses_per_portion_macros():
    system = _build_diet_prompt().messages[0].prompt.template
    assert '"amount":200.0,"unit":"g","protein_g":62.0' in system
    assert '"protein_g":31.0' not in system
    assert "WHOLE PORTION" in system
    assert "Do NOT convert counts to grams" in system


@pytest.fixture
def no_food_db(monkeypatch):
    monkeypatch.setattr(diet_parser, "get_food_engine", None)


def test_llm_macros_kept_when_plausible(no_food_db):
    item = DietParseItem(food_name="모르는음식", amount=200, unit="g", protein_g=20.0, carbs_g=30.0, fat_g=10.0)
    out = _enrich_macros(ParsedDietLog(items=[item])).items[0]
    assert (out.protein_g, out.carbs_g, out.fat_g) == (20.0, 30.0, 10.0)


def test_llm_per100g_macros_on_small_portion_are_rescaled(no_food_db):
    # 버터 10g with per-100g macros (81 g fat = 729 kcal in 10 g) is impossible.
    item = DietParseItem(food_name="버터", amount=10, unit="g", protein_g=0.9, carbs_g=0.1, fat_g=81.0)
    out = _enrich_macros(ParsedDietLog(items=[item])).items[0]
    assert out.fat_g == pytest.approx(8.1)
    kcal = 4 * out.protein_g + 4 * out.carbs_g + 9 * out.fat_g
    assert kcal / 10 <= 9.0


def test_llm_impossible_kcal_is_dropped(no_food_db):
    item = DietParseItem(food_name="모르는음식", amount=150, unit="g", protein_g=10.0, carbs_g=10.0, fat_g=10.0, kcal=5000.0)
    out = _enrich_macros(ParsedDietLog(items=[item])).items[0]
    assert out.kcal is None and out.fat_g is None


def test_db_hit_without_amount_reports_basis_amount(monkeypatch):
    engine = _engine([_doc("커피 아메리카노", "커피", kcal=4.0, basis_unit="ml")], distance=0.0)
    monkeypatch.setattr(diet_parser, "get_food_engine", lambda: engine)
    out = _enrich_macros(ParsedDietLog(items=[DietParseItem(food_name="아메리카노")])).items[0]
    assert out.source == "db"
    assert (out.amount, out.unit) == (100.0, "ml")


def test_llm_converted_egg_count_is_restored_to_pieces():
    llm = ParsedDietLog(items=[
        DietParseItem(food_name="계란", amount=200, unit="g", protein_g=25.2, carbs_g=1.4, fat_g=19.0),
        DietParseItem(food_name="바나나", amount=1, unit="개", protein_g=1.3, carbs_g=27.0, fat_g=0.4),
    ])
    out = _reconcile_amounts_with_text(llm, "아침: 계란 2개, 바나나 1개, 아메리카노").items
    assert (out[0].amount, out[0].unit) == (2.0, "개")
    assert out[0].protein_g == pytest.approx(12.6)  # rescaled 200 g -> 100 g
    assert (out[1].amount, out[1].unit, out[1].protein_g) == (1, "개", 1.3)


# ---------- 4. particle stripping ----------

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("삶은 달걀 2개", "삶은 달걀 2개"),
        ("포도도 먹음", "포도 먹음"),
        ("포도 먹음", "포도 먹음"),
        ("밥은 1공기", "밥 1공기"),
        ("아보카도", "아보카도"),
        ("닭가슴살을 200g", "닭가슴살 200g"),
    ],
)
def test_particles_stripped_only_at_word_end(text, expected):
    assert _strip_context_words(text) == expected


def test_deterministic_keeps_boiled_egg_name():
    items = _deterministic_diet_parse("삶은 달걀 2개 먹었어").items
    assert [(i.food_name, i.amount, i.unit) for i in items] == [("삶은 달걀", 2.0, "개")]


# ---------- 5. food extraction from chatter ----------

def test_deterministic_extracts_salad_from_chatter():
    items = _deterministic_diet_parse("오늘 너무 피곤했다 그래도 저녁엔 샐러드 먹음").items
    assert [(i.food_name, i.meal_type) for i in items] == [("샐러드", "dinner")]


@pytest.mark.parametrize("text", ["오늘 운동했다", "오늘 점심 잘 먹었어"])
def test_deterministic_chatter_without_food_yields_no_items(text):
    assert _deterministic_diet_parse(text).items == []


def test_deterministic_mixed_counts_and_bare_food():
    items = _deterministic_diet_parse("아침: 계란 2개, 바나나 1개, 아메리카노").items
    assert [(i.food_name, i.amount, i.unit) for i in items] == [
        ("계란", 2.0, "개"), ("바나나", 1.0, "개"), ("아메리카노", None, None),
    ]


def test_deterministic_new_units_and_bare_count():
    items = _deterministic_diet_parse("콜라 1캔이랑 치킨 2조각, 계란 2").items
    assert [(i.food_name, i.amount, i.unit) for i in items] == [
        ("콜라", 1.0, "캔"), ("치킨", 2.0, "조각"), ("계란", 2.0, None),
    ]


def test_deterministic_skips_workout_clause():
    items = _deterministic_diet_parse("벤치프레스 60kg 10회 3세트 하고 닭가슴살 200g 먹었어").items
    assert [(i.food_name, i.amount, i.unit) for i in items] == [("닭가슴살", 200.0, "g")]


def _mock_llm(monkeypatch, payload: ParsedDietLog):
    class _Llm:
        def with_structured_output(self, schema):
            return RunnableLambda(lambda _prompt: payload)

    monkeypatch.setattr(diet_parser, "get_llm", lambda *a, **kw: _Llm())
    monkeypatch.setattr(
        diet_parser, "resolve_llm_provider", lambda: SimpleNamespace(effective_provider="remote")
    )
    monkeypatch.setattr(diet_parser, "get_food_engine", None)


def test_parse_diet_log_falls_back_to_deterministic_when_llm_returns_no_items(monkeypatch):
    _mock_llm(monkeypatch, ParsedDietLog(items=[]))
    result = json.loads(diet_parser.parse_diet_log("오늘 너무 피곤했다 그래도 저녁엔 샐러드 먹음"))
    assert [(i["food_name"], i["meal_type"]) for i in result["items"]] == [("샐러드", "dinner")]


def test_parse_diet_log_restores_egg_count_from_llm_grams(monkeypatch):
    _mock_llm(monkeypatch, ParsedDietLog(items=[
        DietParseItem(food_name="계란", amount=200, unit="g", protein_g=25.2, carbs_g=1.4, fat_g=19.0),
    ]))
    item = json.loads(diet_parser.parse_diet_log("아침: 계란 2개"))["items"][0]
    assert (item["amount"], item["unit"]) == (2.0, "개")
    assert item["protein_g"] == pytest.approx(12.6)  # table: 12.6 g/100 g × 100 g


def test_parse_diet_log_cola_can_uses_355ml(monkeypatch):
    _mock_llm(monkeypatch, ParsedDietLog(items=[
        DietParseItem(food_name="콜라", amount=1, unit="캔", protein_g=0.0, carbs_g=10.6, fat_g=0.0),
    ]))
    item = json.loads(diet_parser.parse_diet_log("콜라 1캔"))["items"][0]
    assert item["carbs_g"] == pytest.approx(37.6)  # 10.6 g/100 ml × 355 ml


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("제육볶음 1인분", [("제육볶음", 1.0, "인분")]),
        ("저녁엔 김치볶음 먹음", [("김치볶음", None, None)]),
        ("망고 1개", [("망고", 1.0, "개")]),
    ],
)
def test_deterministic_keeps_foods_with_predicate_like_endings(text, expected):
    items = _deterministic_diet_parse(text).items
    assert [(i.food_name, i.amount, i.unit) for i in items] == expected

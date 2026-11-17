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
    # rescaled LLM macros (8.1 g) or, when the MFDS processed-food index is present, its butter row
    assert 7.5 <= out.fat_g <= 8.6
    kcal = 4 * out.protein_g + 4 * out.carbs_g + 9 * out.fat_g
    assert kcal / 10 <= 9.0


def test_llm_impossible_kcal_is_dropped(no_food_db):
    item = DietParseItem(food_name="모르는음식", amount=150, unit="g", protein_g=10.0, carbs_g=10.0, fat_g=10.0, kcal=5000.0)
    out = _enrich_macros(ParsedDietLog(items=[item])).items[0]
    assert out.kcal is None and out.fat_g is None


def test_db_hit_without_amount_reports_one_serving(monkeypatch):
    # 2026-11-10: no amount = one usual serving (a 355 ml cup), not the 100 ml DB basis.
    engine = _engine([_doc("커피 아메리카노", "커피", kcal=4.0, basis_unit="ml")], distance=0.0)
    monkeypatch.setattr(diet_parser, "get_food_engine", lambda: engine)
    out = _enrich_macros(ParsedDietLog(items=[DietParseItem(food_name="아메리카노")])).items[0]
    assert out.source == "db"
    assert (out.amount, out.unit) == (1.0, "인분")
    assert out.kcal == pytest.approx(14.2)


def test_food_without_serving_data_keeps_basis_amount(monkeypatch):
    engine = _engine([_doc("모시풀 잎", "모시풀", kcal=40.0)], distance=0.0)
    monkeypatch.setattr(diet_parser, "get_food_engine", lambda: engine)
    out = _enrich_macros(ParsedDietLog(items=[DietParseItem(food_name="모시풀 잎")])).items[0]
    assert (out.amount, out.unit) == (100.0, "g")


def test_llm_placeholder_100g_without_amount_in_text_becomes_one_serving():
    from engines.quicklog.diet_parser import _drop_placeholder_amounts
    parsed = ParsedDietLog(items=[DietParseItem(food_name="된장찌개", amount=100, unit="g", protein_g=3.0, carbs_g=4.0, fat_g=1.6)])
    out = _drop_placeholder_amounts(parsed, "된장찌개 먹었어").items[0]
    assert (out.amount, out.unit) == (None, None)
    assert out.protein_g == pytest.approx(12.0)  # 100 g estimate scaled to a 400 g serving
    kept = _drop_placeholder_amounts(parsed, "된장찌개 100g").items[0]
    assert kept.amount == 100


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


def test_rice_bowl_is_one_gonggi_not_soup_bowl():
    from engines.food.food_units import food_unit_overrides
    assert food_unit_overrides("공기밥").get("그릇") == 210.0
    assert food_unit_overrides("밥").get("그릇") == 210.0


def test_length_size_is_pieces_not_count():
    from engines.food.food_units import resolve_portion_grams
    # "터키 샌드위치 15cm" used to become 15 servings (1.5 kg, 3750 kcal).
    assert resolve_portion_grams("터키 샌드위치", 15, "cm") == 200.0
    assert resolve_portion_grams("터키 샌드위치", 30, "cm") == 400.0


@pytest.mark.parametrize(("text", "expected"), [
    ("밥 한 공기", [("밥", 1.0, "공기")]),
    ("우유 반 팩", [("우유", 0.5, "팩")]),
    ("밥 1/3공기만 먹었어요", [("밥", 0.333, "공기")]),
    ("잡곡밥 반공기에 계란후라이 2개", [("잡곡밥", 0.5, "공기"), ("계란후라이", 2.0, "개")]),
    ("방울토마토 한 열댓개", [("방울토마토", 12.0, "개")]),
    ("치킨 한 마리 반", [("치킨", 1.5, "마리")]),
    ("햄버거 하나랑 콜라", [("햄버거", 1.0, "개"), ("콜라", None, None)]),
])
def test_deterministic_parser_reads_korean_number_words(text, expected):
    items = diet_parser._deterministic_diet_parse(text).items
    assert [(i.food_name, i.amount, i.unit) for i in items] == expected


@pytest.mark.parametrize(("amount", "unit", "expected"), [
    (None, "한 컵", (1.0, "컵")),
    (1.0, "반공기", (0.5, "공기")),
    (None, "두 조각", (2.0, "조각")),
    (None, "30cm", (30.0, "cm")),
    (1.0, "작은 컵", (1.0, "컵")),
    (None, "공기", (1.0, "공기")),
    (200.0, "g", (200.0, "g")),
])
def test_llm_unit_strings_with_numbers_are_split(amount, unit, expected):
    parsed = ParsedDietLog(items=[DietParseItem(food_name="x", amount=amount, unit=unit)])
    item = diet_parser._normalize_llm_units(parsed).items[0]
    assert (item.amount, item.unit) == expected


def test_curated_product_uses_label_values_per_package():
    from engines.food.curated_products import product_nutrition
    assert product_nutrition("신라면", 1, "봉지")["kcal"] == 500.0
    assert product_nutrition("편의점 바나나우유", None, None)["kcal"] == 208.0  # alias inside the name
    assert product_nutrition("포카리", 250, "ml")["kcal"] == 60.0  # grams scale the package
    assert product_nutrition("빅맥 세트", 1, "")["kcal"] == 1105.0  # the set row, not the burger alone
    assert product_nutrition("빅맥", 1, "개")["kcal"] == 582.0
    assert product_nutrition("라면", 1, "개") is None  # generic dish stays on the DB path



def test_sharing_and_size_words_scale_the_logged_portion():
    from engines.quicklog.diet_parser import _share_divisor, _size_factor
    assert _share_divisor("치킨 한 마리를 셋이서 나눠먹음") == 3.0
    assert _share_divisor("라볶이 2인분 시켜서 친구랑 반띵") == 2.0
    assert _share_divisor("김밥 두 줄") == 1.0
    assert _size_factor("짜장면", "짜장면 곱빼기") == 1.5
    assert _size_factor("감자튀김", "감자튀김 라지") == 1.4
    assert _size_factor("특제 소스", "특제 소스") == 1.0  # "특" inside a word is not a size


def test_supplement_capsules_and_spirit_glasses_use_small_portions():
    from engines.food.food_units import unit_to_grams
    parsed = diet_parser._deterministic_diet_parse("오메가3 2알이랑 멀티비타민 1알")
    items = [(i.food_name, i.amount, i.unit) for i in parsed.items]
    assert items == [("오메가3", 2.0, "알"), ("멀티비타민", 1.0, "알")]  # "3" is part of the name
    assert parsed.items[0].kcal == 18.0  # ~1 g fish-oil softgel each, not a product named 오메가…
    assert unit_to_grams("위스키", "잔") == 40.0  # a shot glass, not a beer glass
    assert diet_parser._deterministic_diet_parse("닭가슴살 100g").items[0].amount == 100.0


def test_bbq_meat_cuts_resolve_to_cooked_rows():
    from engines.food.food_query_normalization import FOOD_SEARCH_ALIAS_RULES
    assert FOOD_SEARCH_ALIAS_RULES["목살"] == "돼지고기 목심 구운것"
    assert FOOD_SEARCH_ALIAS_RULES["차돌박이"] == "차돌박이구이"
    assert FOOD_SEARCH_ALIAS_RULES["LA갈비"] == "소갈비 구이"


def test_processed_index_single_serve_and_dish_variants():
    from engines.food import curated_products as cp
    if not cp._processed_index():
        pytest.skip("processed product index not built (generated file)")
    # "하나" of a drink / snack bag is the single-serve package, not a 50 ml sample or a serving
    assert cp.product_nutrition("아침햇살", 1, "개")["kcal"] > 90
    assert cp.product_nutrition("포스틱", 1, "개")["kcal"] > 300
    # a flavour word at the end of a drink name is not a piece ("…요구르트 딸기" is not 15 g)
    assert cp.product_nutrition("덴마크 드링킹 요구르트 딸기", 1, "개")["kcal"] > 150
    # menu dishes named "<variant><category>" stay on the generic dish path
    assert cp.find_processed_product("콤비네이션 피자") is None
    assert cp.find_processed_product("고구마피자") is None
    # brand prefix dropped by the user: "왕뚜껑" -> 팔도 왕뚜껑
    assert cp.find_processed_product("왕뚜껑") is not None


def test_set_meals_and_half_half_chicken_are_expanded():
    from engines.quicklog.diet_parser import _apply_half_half_text, _expand_set_menus
    p = ParsedDietLog(items=[DietParseItem(food_name="1955버거 세트", amount=100.0, unit="g")])
    assert [(i.food_name, i.unit) for i in _expand_set_menus(p).items] == [
        ("1955버거", "개"), ("감자튀김", "인분"), ("콜라", "잔")]
    p = ParsedDietLog(items=[DietParseItem(food_name="생선까스 정식", amount=1.0, unit="인분")])
    assert [i.food_name for i in _expand_set_menus(p).items] == ["생선까스", "쌀밥"]
    p = ParsedDietLog(items=[DietParseItem(food_name="후라이드 반 양념 반", amount=1.0, unit="마리")])
    assert [(i.food_name, i.unit) for i in _expand_set_menus(p).items] == [("치킨", "마리")]
    p = ParsedDietLog(items=[DietParseItem(food_name="치킨", amount=1.0, unit="인분")])
    assert _apply_half_half_text(p, "치킨 반반 시켜서 반씩 나눠먹음").items[0].unit == "마리"


def test_leftovers_and_nothing_eaten():
    from engines.quicklog.diet_parser import _NOTHING_EATEN_RE, _leftover_factor
    item = DietParseItem(food_name="진라면", amount=1.0, unit="개")
    assert _leftover_factor(item, [item], "진라면 하나 끓였는데 반은 남김") == 0.5
    half = DietParseItem(food_name="물냉면", amount=0.5, unit="그릇")
    assert _leftover_factor(half, [half], "물냉면 반 그릇만 먹음") == 1.0  # already logged as half
    assert _NOTHING_EATEN_RE.search("라면 끓이려다 그냥 굶음")


def test_alcohol_uses_label_values_not_llm_macros():
    from engines.food.food_units import unit_to_grams
    assert diet_parser._FOOD_PER_100G["청하"]["kcal"] > 90
    assert diet_parser._FOOD_PER_100G["대선"]["kcal"] == 88.0  # MFDS label
    assert unit_to_grams("고량주", "잔") == 30.0


def test_portion_confidence_levels():
    from engines.food.food_units import portion_confidence
    assert portion_confidence("닭가슴살", 200, "g") == "explicit"
    assert portion_confidence("계란", 2, "개") == "exact"
    assert portion_confidence("양념치킨", 3, "조각") == "suffix"
    assert portion_confidence("두부김치", 1, "개") == "default"  # no piece weight known


def test_dish_over_rice_and_drink_servings():
    from engines.food.food_units import serving_grams
    assert serving_grams("잡채밥") == 400.0  # dish over rice, not one bowl of rice (210)
    assert serving_grams("보리밥") == 210.0
    assert serving_grams("자몽에이드") == 450.0


def test_lexical_relation_accepts_spelling_variants_only():
    from engines.food.food_reranker import has_lexical_relation as rel
    assert rel("쭈꾸미볶음", "주꾸미볶음")
    assert rel("소대창", "대창구이 소고기")
    assert not rel("소주", "소금")
    assert not rel("맥주", "버거 맥치킨 버거")


def test_english_food_names_have_korean_aliases():
    from engines.food.food_query_normalization import FOOD_SEARCH_ALIAS_RULES as alias
    assert alias["pad thai"] == "팟타이"
    assert alias["croquettes"] == "고로케"


def test_gram_rules_precedence_a_to_f():
    from engines.food.food_units import resolve_portion_grams as grams
    assert grams("닭가슴살", 200, "g") == 200.0            # a explicit weight wins
    assert grams("삼겹살", 1, "근") == 600.0                # a 근
    assert grams("계란", 2, "개") == 100.0                  # b exact per-food table
    assert grams("감자전", 2, "장") == 400.0                # c head-noun table (전 장 = 200 g)
    assert grams("김치찌개", 1, "인분") == 400.0            # d serving suffix
    assert grams("잡채밥", 1, "그릇") == 400.0              # d dish over rice
    assert grams("미지의음식", 2, "개") == 200.0            # f generic default (개 = 100 g)
    assert grams("잣", 1, "숟가락") == 10.0                 # f spoon


def test_explicit_amount_beats_ordered_dish_serving():
    from engines.quicklog.diet_parser import _apply_ordered_dish_serving
    p = ParsedDietLog(items=[DietParseItem(food_name="미지의요리", amount=100.0, unit="g")])
    assert _apply_ordered_dish_serving(p, "미지의요리 100g 시켜먹음").items[0].unit == "g"
    assert _apply_ordered_dish_serving(p, "미지의요리 시켜먹음").items[0].unit == "인분"


def test_category_fallback_uses_median_of_one_basis():
    from engines.food.food_engine import category_nutrition
    hit = category_nutrition("쭈꾸미볶음", 200, "g")
    if hit is None:
        pytest.skip("food DB not available")
    assert hit["category"] == "볶음" and hit["basis"] == "g" and hit["category_rows"] >= 5
    assert hit["grams"] == 200.0
    assert category_nutrition("ㅋㅋ", 1, "개") is None

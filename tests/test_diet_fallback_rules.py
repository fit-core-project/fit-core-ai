"""General post-LLM / fallback / matching rules of the diet pipeline (no LLM, no vector index).

Each test pins one general rule (unit class, category, row structure), never a dataset case.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from engines.food.food_engine import FoodSearchEngine
from engines.food.food_units import category_portion_grams, resolve_portion_grams, serving_grams
from engines.quicklog import diet_parser
from engines.quicklog.diet_parser import DietParseItem, ParsedDietLog


def _meta(name, category, *, kcal=100.0, protein=3.0, data_type="음식"):
    return {"name": name, "rep_name": name.split()[0], "major_category": category, "data_type": data_type,
            "kcal": kcal, "protein_g": protein, "carb_g": 10.0, "fat_g": 3.0,
            "basis_value": 100.0, "basis_unit": "g", "food_id": name}


# ---------- unit classes ----------

def test_cutlet_sheet_and_whole_fish_units():
    assert category_portion_grams(_meta("돈가스 고구마", "튀김류", protein=12), "고구마돈가스", 1, "장", 20.0) == 130.0
    assert category_portion_grams(_meta("조기찜", "찜류", protein=18), "조기찜", 1, "마리", 400.0) == 120.0
    # poultry / crab dishes ordered by the animal keep their own weights
    assert category_portion_grams(_meta("통닭구이", "구이류", protein=20), "통닭구이", 1, "마리", 1000.0) is None
    assert category_portion_grams(_meta("꽃게찜", "찜류", protein=15), "꽃게찜", 1, "마리", 400.0) is None


def test_only_cutlets_get_the_cutlet_sheet_weight():
    assert category_portion_grams(_meta("생선까스", "튀김류", protein=12), "생선가스", 1, "장", 20.0) == 130.0
    for name in ("깻잎튀김", "김부각", "고추튀김"):
        assert category_portion_grams(_meta(name, "튀김류", protein=5), name, 2, "장", 40.0) is None


def test_pot_holds_four_servings():
    assert resolve_portion_grams("꽃게탕", 0.25, "냄비") == pytest.approx(serving_grams("꽃게탕"))


def test_pot_of_a_one_person_dish_is_one_serving():
    assert resolve_portion_grams("라면", 1, "냄비") == pytest.approx(serving_grams("라면"))
    assert resolve_portion_grams("떡볶이", 1, "냄비") == pytest.approx(serving_grams("떡볶이") or 400.0)


def test_half_unit_with_half_amount_is_one_half():
    parsed = ParsedDietLog(items=[DietParseItem(food_name="소안심스테이크", amount=0.5, unit="반근")])
    item = diet_parser._normalize_llm_units(parsed).items[0]
    assert (item.amount, item.unit) == (0.5, "근")


# ---------- LLM-side cleanup (post-LLM, deterministic) ----------

def test_byte_tokens_in_llm_names_are_decoded():
    parsed = diet_parser._parse_payload({"items": [{"food_name": "미고<0xEB><0xA0><0x9D>", "amount": 1, "unit": "봉"}]})
    assert parsed.items[0].food_name == "미고렝"
    assert diet_parser._decode_byte_tokens("치즈볼") == "치즈볼"


def test_negated_items_are_dropped():
    parsed = ParsedDietLog(items=[DietParseItem(food_name="나시고렝", amount=1, unit="개"), DietParseItem(food_name="콜라")])
    out = diet_parser._drop_negated_items(parsed, "야식: 나시고렝 1개 / 콜라 X")
    assert [i.food_name for i in out.items] == ["나시고렝"]
    # an X inside a word is not a negation
    box = ParsedDietLog(items=[DietParseItem(food_name="BOX"), DietParseItem(food_name="콜라")])
    assert len(diet_parser._drop_negated_items(box, "BOX 콜라").items) == 2


@pytest.mark.parametrize("text, names", [
    ("밥 1공기, 닭가슴살 x 2", ["밥", "닭가슴살"]),
    ("삶은계란 x 2, 고구마 1개", ["삶은계란", "고구마"]),
    ("아메리카노 X 2잔이랑 베이글", ["아메리카노", "베이글"]),
    ("닭가슴살 x 두 개, 밥 1공기", ["닭가슴살", "밥"]),
])
def test_x_before_a_count_is_a_multiplier_not_a_negation(text, names):
    parsed = ParsedDietLog(items=[DietParseItem(food_name=n) for n in names])
    assert [i.food_name for i in diet_parser._drop_negated_items(parsed, text).items] == names


def test_share_divides_only_the_shared_dishes():
    items = [DietParseItem(food_name="갈치조림", amount=2, unit="인분", kcal=600.0),
             DietParseItem(food_name="밥", amount=0.667, unit="공기", kcal=210.0)]
    out = diet_parser._apply_portion_context(ParsedDietLog(items=items), "갈치조림 2인분 3명이서 / 밥2/3공기").items
    assert out[0].kcal == pytest.approx(200.0)
    assert out[1].kcal == pytest.approx(210.0)
    # a single shared bottle is still divided
    soju = [DietParseItem(food_name="소주", amount=3, unit="병", kcal=1200.0)]
    assert diet_parser._apply_portion_context(ParsedDietLog(items=soju), "소주 3병 넷이서").items[0].kcal == pytest.approx(300.0)
    assert diet_parser._share_divisor("4인분을 다섯 명이 나누어 먹었다") == 5.0
    assert diet_parser._share_divisor("10명이서 나눠 먹음") == 10.0


def test_shared_bottles_are_divided_next_to_a_shared_dish():
    items = [DietParseItem(food_name="삼겹살", amount=3, unit="인분", kcal=2000.0),
             DietParseItem(food_name="소주", amount=2, unit="병", kcal=800.0)]
    out = diet_parser._apply_portion_context(ParsedDietLog(items=items), "삼겹살 3인분, 소주 2병 셋이서 나눠 먹음").items
    assert [i.kcal for i in out] == [pytest.approx(666.7, abs=0.1), pytest.approx(266.7, abs=0.1)]
    cola = [DietParseItem(food_name="피자", amount=1, unit="판", kcal=2000.0),
            DietParseItem(food_name="콜라", amount=1, unit="병", kcal=200.0)]
    out = diet_parser._apply_portion_context(ParsedDietLog(items=cola), "피자 1판이랑 콜라 1.25L 1병 둘이서 나눠 먹음").items
    assert out[1].kcal == pytest.approx(100.0)
    # a drinks-only share divides every drink
    drinks = [DietParseItem(food_name="소주", amount=1, unit="병", kcal=400.0),
              DietParseItem(food_name="맥주", amount=2, unit="캔", kcal=300.0)]
    out = diet_parser._apply_portion_context(ParsedDietLog(items=drinks), "소맥 말아서 소주 1병 맥주 2캔 둘이").items
    assert [i.kcal for i in out] == [pytest.approx(200.0), pytest.approx(150.0)]
    # cans are poured / drunk by one person
    can = [DietParseItem(food_name="치킨", amount=1, unit="마리", kcal=2000.0),
           DietParseItem(food_name="맥주", amount=2, unit="캔", kcal=300.0)]
    out = diet_parser._apply_portion_context(ParsedDietLog(items=can), "치킨 1마리 둘이서, 나는 맥주 2캔").items
    assert out[1].kcal == pytest.approx(300.0)


def test_size_word_keun_geo_scales_the_item():
    assert diet_parser._size_factor("두유 카페라떼", "두유 카페라떼 큰 거 하나") == pytest.approx(1.4)


def test_english_food_names_translate_word_by_word():
    from engines.food.food_query_normalization import translate_english_food_name as tr

    assert tr("zucchini jeon") == "애호박전"
    assert tr("braised lotus root") == "연근조림"
    assert tr("boiled eggs") == "삶은 계란"
    assert tr("KFC 징거타워") is None
    assert tr("saessak salad") is None  # an unknown word: leave it to search


@pytest.mark.parametrize("english, korean", [
    ("rice cake", "떡"), ("fish cake", "어묵"), ("soy milk", "두유"), ("fried rice", "볶음밥"),
    ("rice noodles", "쌀국수"), ("black coffee", "아메리카노"), ("bean sprouts", "콩나물"),
    ("black bean noodles", "짜장면"), ("spicy rice cake", "떡볶이"),
    # compounds whose head changes the meaning of the word before it: no word-by-word guess
    ("soy latte", "두유라떼"), ("banana pancake", None), ("milk tea", None), ("chicken noodles", None), ("fried rice cake", None),
])
def test_english_compounds_that_do_not_compose(english, korean):
    from engines.food.food_query_normalization import translate_english_food_name as tr

    assert tr(english) == korean


# ---------- implausible LLM macros on a DB miss ----------

def _zero(name, amount=1, unit="개"):
    return DietParseItem(food_name=name, amount=amount, unit=unit, protein_g=0.0, carbs_g=0.0, fat_g=0.0)


def test_zero_kcal_llm_macros_are_implausible_for_real_food_but_not_water():
    assert diet_parser._llm_macros_implausible(_zero("귤탕후루"))
    for name in ("물", "아메리카노", "콜라 제로", "크레아틴", "커피"):
        assert not diet_parser._llm_macros_implausible(_zero(name, unit="잔"))


@pytest.mark.parametrize("name", ["귤튀김", "고사리나물", "녹차라떼", "해물", "블랙타이거새우", "김튀김", "믹스커피", "물냉면"])
def test_zero_kcal_whitelist_is_anchored_on_the_whole_name(name):
    assert diet_parser._llm_macros_implausible(_zero(name))


@pytest.mark.parametrize("name", ["생수", "아이스 아메리카노", "venti cold brew", "스타벅스 아메리카노",
                                  "아메리카노 톨", "콜드브루 라지", "샷추가 아메리카노", "아이스 현미녹차", "아이스 커피", "녹차", "김", "미역", "오이", "제로콜라", "블랙커피"])
def test_zero_kcal_whitelist_keeps_real_zero_foods(name):
    assert not diet_parser._llm_macros_implausible(_zero(name))
    plausible = DietParseItem(food_name="모르는음식", amount=200, unit="g", protein_g=20.0, carbs_g=30.0, fat_g=10.0)
    assert not diet_parser._llm_macros_implausible(plausible)


def test_drink_denser_than_a_milkshake_is_implausible_unless_alcohol():
    item = DietParseItem(food_name="쏘굿 아몬드 언스위트", amount=1, unit="잔", protein_g=25.0, carbs_g=20.0, fat_g=55.0)
    assert diet_parser._llm_macros_implausible(item)
    whisky = DietParseItem(food_name="위스키", amount=100, unit="ml", protein_g=0.0, carbs_g=0.0, fat_g=0.0, kcal=250.0)
    assert not diet_parser._llm_macros_implausible(whisky)


def test_implausible_macros_use_the_product_line_density(monkeypatch):
    family = {"kcal_100": 107.0, "protein_100": 0.8, "carbs_100": 25.0, "fat_100": 0.2, "unit_g": 61.0,
              "rows": 1, "key": "귤탕후루"}
    monkeypatch.setattr(diet_parser, "product_family_density", lambda name: family)
    out = diet_parser._replace_implausible_llm_macros(_zero("귤탕후루"))
    assert out.kcal == pytest.approx(65.3)  # 107 kcal/100 g x one 61 g piece
    assert out.source == "db"


def test_implausible_drink_without_product_falls_back_to_drink_class(monkeypatch):
    monkeypatch.setattr(diet_parser, "product_family_density", lambda name: None)
    out = diet_parser._replace_implausible_llm_macros(_zero("오트 드링크", unit="잔"))
    assert 80 <= out.kcal <= 150  # ~45 kcal/100 ml x a 250 ml glass
    # nothing known about a non-drink: the LLM numbers stay
    snack = _zero("모르는과자")
    assert diet_parser._replace_implausible_llm_macros(snack) == snack


def test_misread_amount_is_not_used_as_a_portion():
    item = DietParseItem(food_name="곤약밥", amount=130, unit="팩", protein_g=1.0, carbs_g=10.0, fat_g=0.5)
    assert diet_parser._deterministic_grams(item) is None


def test_category_fallback_rejects_dessert_density_for_a_drink():
    item = DietParseItem(food_name="쏘굿 아몬드 언스위트", amount=1, unit="잔")
    assert diet_parser._category_density_implausible(item, {"kcal": 692.5, "grams": 250.0})
    assert not diet_parser._category_density_implausible(item, {"kcal": 40.0, "grams": 250.0})


# ---------- product index rules ----------

def _product(**kw):
    from engines.food.curated_products import IndexedProduct

    base = dict(key="p", name="p", kcal_100=100.0, protein_100=1.0, carbs_100=10.0, fat_100=1.0,
                unit_g=100.0, package_sizes=(), n_rows=1, category_lv3="", serving_g=0.0)
    base.update(kw)
    return IndexedProduct(**base)


def test_placeholder_label_serving_does_not_pick_a_multipack():
    from engines.food.curated_products import _indexed_grams

    bar = _product(key="누가바", unit_g=63.0, package_sizes=(63.0, 70.0, 160.0, 420.0), category_lv3="빙과류",
                   serving_g=100.0)
    assert _indexed_grams(bar, "누가바", 1.0, "개") == 63.0


def test_glass_of_a_bottled_drink_is_one_serving_package():
    from engines.food.curated_products import _indexed_grams

    drink = _product(unit_g=190.0, package_sizes=(190.0, 950.0), category_lv3="음료류", serving_g=190.0)
    assert _indexed_grams(drink, "아몬드음료", 1.0, "잔") == 190.0


def test_product_family_density_is_none_without_the_index(monkeypatch):
    from engines.food import curated_products

    monkeypatch.setattr(curated_products, "_processed_index", lambda: {})
    curated_products._family_rows.cache_clear()
    assert curated_products.product_family_density("귤탕후루") is None
    curated_products._family_rows.cache_clear()


# ---------- wrong-row rules ----------

def _cand(name, rep, data_type="원재료성 식품"):
    from engines.food.food_reranker import FoodSearchCandidate

    doc = SimpleNamespace(metadata={"name": name, "rep_name": rep, "data_type": data_type, "food_id": name},
                          page_content=name)
    return FoodSearchCandidate(document=doc, vector_score=0.2, source_query="q", source_query_index=0,
                               original_rank=0, dedupe_key=("food_id", name))


def test_plain_food_row_is_not_the_dish_it_modifies():
    from engines.food.food_query_normalization import analyze_food_query_for_search
    from engines.food.food_reranker import is_implausible_food_candidate

    assert is_implausible_food_candidate(analyze_food_query_for_search("두부면"), _cand("두부", "두부"))
    assert is_implausible_food_candidate(analyze_food_query_for_search("두부포케"), _cand("두부", "두부"))
    # a cooking suffix still names the same food (삼겹살구이 is 삼겹살)
    assert not is_implausible_food_candidate(analyze_food_query_for_search("삼겹살구이"), _cand("삼겹살", "삼겹살"))


def test_reordered_dish_key_matches_mfds_word_order():
    from engines.food.food_canonical_index import FoodCanonicalIndex

    rows = [
        {"food_id": "a", "name": "돈가스 고구마", "rep_name": "돈가스", "data_type": "음식", "kcal": "317"},
        {"food_id": "b", "name": "돈가스", "rep_name": "돈가스", "data_type": "음식", "kcal": "280"},
        {"food_id": "c", "name": "라떼 바닐라", "rep_name": "라떼", "data_type": "음식", "kcal": "60"},
        {"food_id": "d", "name": "라떼 바닐라라떼 핫(HOT)", "rep_name": "라떼", "data_type": "음식", "kcal": "44"},
    ]
    index = FoodCanonicalIndex.from_rows(rows)
    assert index.lookup_reordered("고구마돈가스").metadata["name"] == "돈가스 고구마"
    # a row already naming the query as a word wins over the reordered key
    assert index.lookup_reordered("바닐라라떼") is None


def test_dried_row_is_swapped_for_its_cooked_sibling():
    from engines.food.food_canonical_index import FoodCanonicalIndex

    rows = [
        {"food_id": "x", "name": "병아리콩 수입산(인도산) 말린것", "rep_name": "병아리콩",
         "data_type": "원재료성 식품", "kcal": "373"},
        {"food_id": "y", "name": "병아리콩 수입산(인도산) 말린것을 삶은 것", "rep_name": "병아리콩",
         "data_type": "원재료성 식품", "kcal": "183"},
    ]
    engine = object.__new__(FoodSearchEngine)
    engine._canonical_index = FoodCanonicalIndex.from_rows(rows)
    dried = engine._canonical_index.lookup_name(rows[0]["name"]).metadata
    assert engine._cooked_sibling(dried, "병아리콩")["kcal"] == 183.0
    assert engine._cooked_sibling(dried, "말린 병아리콩") is None

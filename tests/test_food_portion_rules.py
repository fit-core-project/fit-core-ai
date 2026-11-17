"""General portion / fallback rules of the formula-first diet pipeline (no LLM, no vector index).

Each test pins one category-level rule, never a single dataset case.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from engines.food.food_engine import FoodSearchEngine
from engines.food.food_units import category_portion_grams


def _meta(name, category, *, kcal=100.0, protein=3.0, data_type="음식"):
    return {"name": name, "rep_name": name.split()[0], "major_category": category, "data_type": data_type,
            "kcal": kcal, "protein_g": protein, "carb_g": 10.0, "fat_g": 3.0,
            "basis_value": 100.0, "basis_unit": "g", "food_id": name}


# ---------- category portion weights ----------

def test_gonggi_of_any_rice_row_is_one_rice_bowl():
    # "X밥 2/3공기": a 공기 is a rice bowl whatever the grain prefix (not the 400 g 덮밥 plate)
    assert category_portion_grams(_meta("차조밥", "밥류"), "차조밥", 2 / 3, "공기", 266.8) == pytest.approx(140.0)
    assert category_portion_grams(_meta("수수밥", "밥류"), "수수밥", 1, "공기", 400.0) == 210.0
    # 그릇 is untouched: 비빔밥 / 덮밥 rows are 밥류 too
    assert category_portion_grams(_meta("비빔밥", "밥류"), "비빔밥", 1, "그릇", 450.0) is None


def test_plate_weight_follows_side_dish_category():
    assert category_portion_grams(_meta("굴전", "전·적 및 부침류"), "굴전", 1, "접시", 300.0) == 200.0
    assert category_portion_grams(_meta("시금치나물", "나물·숙채류"), "시금치나물", 0.5, "접시", 150.0) == 50.0
    # vegetable stir-fry gets the side-dish plate; protein / starch stir-fries keep their serving
    assert category_portion_grams(_meta("미역줄기볶음", "볶음류", kcal=60, protein=1.5), "미역줄기볶음", 1, "접시", 300.0) == 120.0
    assert category_portion_grams(_meta("잡채", "볶음류", kcal=150, protein=4.0), "잡채", 1, "접시", 300.0) is None
    assert category_portion_grams(_meta("마파두부", "볶음류", kcal=110, protein=7.0), "마파두부", 1, "접시", 300.0) is None


def test_jjim_generic_serving_becomes_home_serving_unless_small():
    meta = _meta("돼지갈비찜", "찜류", protein=15.0)
    assert category_portion_grams(meta, "돼지갈비찜", None, None, 400.0) == 250.0
    assert category_portion_grams(meta, "돼지갈비찜", 1, "접시", 400.0) == 250.0
    assert category_portion_grams(meta, "돼지갈비찜", None, None, 400.0, small_portion=True) is None
    # a 1인분 order keeps the restaurant serving
    assert category_portion_grams(meta, "아구찜", 1, "인분", 400.0) is None


def test_jeon_piece_is_small_but_egg_rolls_are_excluded():
    assert category_portion_grams(_meta("호박전", "전·적 및 부침류"), "호박전", 2, "개", 80.0) == 50.0
    assert category_portion_grams(_meta("달걀말이", "전·적 및 부침류", protein=10), "계란말이", 2, "개", 100.0) is None


def test_pancake_type_jeon_piece_is_one_pan_sized_pancake():
    # 김치전 / 해물파전 / 빈대떡 1개 is one pancake (the 장 weight), not a 25 g bite
    assert category_portion_grams(_meta("김치전", "전·적 및 부침류"), "김치전", 1, "개", 40.0) == 200.0
    assert category_portion_grams(_meta("해물파전", "전·적 및 부침류"), "해물파전", 1, "개", 40.0) == 200.0
    assert category_portion_grams(_meta("빈대떡", "전·적 및 부침류"), "빈대떡", 1, "개", 100.0) == 250.0
    # bite-sized 전 keep the small piece
    assert category_portion_grams(_meta("동태전", "전·적 및 부침류"), "동태전", 3, "개", 120.0) == 75.0


def test_side_dish_without_amount_gets_one_banchan_portion_only_on_basis_fallback():
    meta = _meta("가지나물", "나물·숙채류", kcal=50, protein=2.0)
    assert category_portion_grams(meta, "가지나물", None, None, None, basis_fallback=True) == 50.0
    # a dish-suffix serving already resolved (샐러드, 떡볶이): not this rule
    assert category_portion_grams(meta, "가지나물", None, None, None, basis_fallback=False) is None
    # protein-dense banchan (멸치볶음, 장조림) keeps the basis
    assert category_portion_grams(_meta("멸치볶음", "볶음류", protein=30.0), "멸치볶음", None, None, None,
                                  basis_fallback=True) is None


def test_non_dish_rows_are_never_reweighted():
    meta = _meta("차조 생것", "밥류", data_type="원재료성 식품")
    assert category_portion_grams(meta, "차조", 1, "공기", 400.0) is None


# ---------- wired into FoodSearchEngine.search ----------

class _Store:
    def __init__(self, docs):
        self.docs = docs

    def similarity_search_with_score(self, query, k):
        return [(doc, 0.0) for doc in self.docs[:k]]


def _engine(meta):
    engine = object.__new__(FoodSearchEngine)
    engine.available = True
    engine._vector_store = _Store([SimpleNamespace(metadata=meta, page_content=meta["name"])])
    engine._canonical_index = None
    engine._lock = threading.Lock()
    return engine


def test_search_uses_rice_bowl_for_gonggi_of_grain_rice():
    hit = _engine(_meta("차조밥", "밥류", kcal=150.0)).search("차조밥", 1, "공기")
    assert hit["resolved_grams"] == 210.0
    assert hit["kcal"] == pytest.approx(315.0)


def test_search_small_portion_keeps_generic_jjim_serving():
    engine = _engine(_meta("단호박찜", "찜류", kcal=41.0, protein=1.0))
    assert engine.search("단호박찜")["resolved_grams"] == 250.0
    assert engine.search("단호박찜", small_portion=True)["resolved_grams"] == 400.0


def test_small_portion_names_follow_the_user_text():
    from engines.quicklog.diet_parser import DietParseItem, ParsedDietLog, _small_portion_names

    parsed = ParsedDietLog(items=[DietParseItem(food_name="단호박찜"), DietParseItem(food_name="현미죽", amount=1)])
    assert _small_portion_names(parsed, "현미죽1 단호박찜 조금") == frozenset({"단호박찜"})
    assert _small_portion_names(parsed, "현미죽1 단호박찜") == frozenset()

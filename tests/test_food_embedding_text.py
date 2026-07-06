from __future__ import annotations

from engines.food.food_embedding_text import (
    FOOD_EMBEDDING_ROW_ALIASES,
    build_food_embedding_text,
    clean_embedding_text_part,
    clean_food_embedding_key,
    dedupe_text_parts,
    get_food_embedding_aliases,
)
from engines.food.food_query_normalization import (
    FOOD_SEARCH_ALIAS_RULES,
    PROTECTED_FOOD_SEARCH_QUERIES,
)


# ---------------------------------------------------------------------------
# Registry guard — catches accidental re-introduction of document aliases
# ---------------------------------------------------------------------------

def test_food_embedding_row_aliases_registry_is_empty():
    # Document-side aliases dilute canonical row vectors without improving
    # recall (runtime normalization + canonical index already handle these).
    assert FOOD_EMBEDDING_ROW_ALIASES == {}


# ---------------------------------------------------------------------------
# Base-text fallback chain
# ---------------------------------------------------------------------------

def test_embedding_text_uses_embed_text_first():
    text = build_food_embedding_text({"embed_text": "달걀 삶은것", "name": "다른 이름"})

    assert text.startswith("달걀 삶은것")


def test_embedding_text_falls_back_to_name():
    assert build_food_embedding_text({"name": "바나나"}) == "바나나"


def test_embedding_text_falls_back_to_rep_name():
    assert build_food_embedding_text({"rep_name": "사과"}) == "사과"


def test_embedding_text_empty_row_returns_empty_string():
    assert build_food_embedding_text({}) == ""


# ---------------------------------------------------------------------------
# No document aliases appended — new contract (FOOD_EMBEDDING_ROW_ALIASES={})
# ---------------------------------------------------------------------------

def test_no_document_aliases_appended_to_egg_raw():
    # Registry is empty; only base text should be returned.
    assert build_food_embedding_text({"name": "달걀 생것"}) == "달걀 생것"


def test_no_document_aliases_appended_to_boiled_egg():
    assert build_food_embedding_text({"name": "달걀 삶은것"}) == "달걀 삶은것"


def test_no_document_aliases_appended_to_fried_egg():
    assert build_food_embedding_text({"name": "달걀후라이"}) == "달걀후라이"


def test_no_document_aliases_appended_to_raw_chicken_breast():
    assert build_food_embedding_text({"name": "닭고기 가슴(껍질 제거) 생것"}) == "닭고기 가슴(껍질 제거) 생것"


def test_no_document_aliases_appended_to_boiled_chicken_breast():
    assert build_food_embedding_text({"name": "닭고기 가슴(껍질 제거) 삶은것"}) == "닭고기 가슴(껍질 제거) 삶은것"


def test_no_document_aliases_appended_to_grilled_chicken_breast():
    assert build_food_embedding_text({"name": "닭고기 가슴(껍질 제거) 구운것(팬)"}) == "닭고기 가슴(껍질 제거) 구운것(팬)"


def test_no_document_aliases_appended_to_sweet_potato():
    assert build_food_embedding_text({"name": "고구마 찐것"}) == "고구마 찐것"
    assert build_food_embedding_text({"name": "고구마 구운것"}) == "고구마 구운것"


def test_no_document_aliases_appended_to_raw_fruit():
    assert build_food_embedding_text({"name": "바나나 생것"}) == "바나나 생것"
    assert build_food_embedding_text({"name": "사과 생것"}) == "사과 생것"


def test_get_food_embedding_aliases_returns_empty_when_registry_is_empty():
    # With an empty registry, get_food_embedding_aliases returns [] for any row.
    aliases = get_food_embedding_aliases({"name": "달걀 대표", "rep_name": "달걀 생것"})

    assert aliases == []


# ---------------------------------------------------------------------------
# Alias non-leakage to unrelated/neighbor rows (contract still holds)
# ---------------------------------------------------------------------------

def test_aliases_do_not_leak_to_neighbor_rows():
    assert build_food_embedding_text({"name": "고구마 생것"}) == "고구마 생것"
    assert build_food_embedding_text({"name": "고구마 말린것"}) == "고구마 말린것"
    assert build_food_embedding_text({"name": "바나나 우유"}) == "바나나 우유"
    assert build_food_embedding_text({"name": "사과 주스"}) == "사과 주스"


# ---------------------------------------------------------------------------
# Display-prefix handling
# ---------------------------------------------------------------------------

def test_clean_food_embedding_key_removes_display_prefix_only():
    assert clean_food_embedding_key("[원재료성 식품] 달걀 삶은것") == "달걀 삶은것"
    assert clean_food_embedding_key("[음식] 계란빵") == "계란빵"


def test_display_prefix_name_is_returned_as_base_text():
    # The display-prefix row yields only base text; no aliases are appended.
    text = build_food_embedding_text({"name": "[원재료성 식품] 달걀 삶은것"})

    assert "[원재료성 식품] 달걀 삶은것" in text
    assert "삶은 계란" not in text


# ---------------------------------------------------------------------------
# Protected compounds must not appear in ingredient rows
# ---------------------------------------------------------------------------

def test_protected_egg_compounds_do_not_leak_into_raw_egg_text():
    text = build_food_embedding_text({"name": "달걀 생것"})

    assert "계란빵" not in text
    assert "볶음밥 계란" not in text
    assert "김밥 계란" not in text


def test_protected_egg_compounds_do_not_leak_into_boiled_egg_text():
    text = build_food_embedding_text({"name": "달걀 삶은것"})

    assert "계란빵" not in text
    assert "볶음밥 계란" not in text
    assert "김밥 계란" not in text


def test_protected_chicken_compounds_do_not_leak_into_raw_chicken_text():
    text = build_food_embedding_text({"name": "닭고기 가슴(껍질 제거) 생것"})

    assert "샐러드 닭가슴살" not in text
    assert "닭가슴살 샐러드" not in text
    assert "샌드위치 닭가슴살" not in text


# ---------------------------------------------------------------------------
# Mutation safety
# ---------------------------------------------------------------------------

def test_build_food_embedding_text_does_not_mutate_row():
    row = {"name": "달걀 생것"}
    before = dict(row)

    build_food_embedding_text(row)

    assert row == before


# ---------------------------------------------------------------------------
# Deduplication and whitespace normalization
# ---------------------------------------------------------------------------

def test_embedding_text_dedupes_when_embed_text_and_name_are_identical():
    # embed_text and name carry the same value; only one instance should appear.
    text = build_food_embedding_text({"embed_text": "  달걀   삶은것  ", "name": "달걀 삶은것"})

    assert text == "달걀 삶은것"
    assert text.count("달걀 삶은것") == 1


def test_clean_embedding_text_part_normalizes_whitespace():
    assert clean_embedding_text_part("  삶은   계란  ") == "삶은 계란"


def test_dedupe_text_parts_preserves_order_and_removes_empty_parts():
    assert dedupe_text_parts(["", "달걀", " 계란 ", "달걀", None]) == ["달걀", "계란"]


# ---------------------------------------------------------------------------
# High-risk terms must never appear in ingredient rows
# ---------------------------------------------------------------------------

def test_unknown_sweet_potato_gets_no_broad_alias():
    text = build_food_embedding_text({"name": "고구마"})

    assert text == "고구마"
    assert "군고구마" not in text


def test_unknown_tofu_gets_no_broad_alias():
    text = build_food_embedding_text({"name": "두부"})

    assert text == "두부"
    assert "순두부" not in text
    assert "연두부" not in text


def test_high_risk_alias_candidates_are_not_added_to_ingredient_rows():
    sweet_potato_text = build_food_embedding_text({"name": "고구마 찐것"})
    tofu_text = build_food_embedding_text({"name": "두부"})
    milk_text = build_food_embedding_text({"name": "우유"})

    assert "군고구마" not in sweet_potato_text
    assert "순두부" not in tofu_text
    assert "연두부" not in tofu_text
    assert "두유" not in milk_text
    assert "콩우유" not in milk_text


# ---------------------------------------------------------------------------
# Runtime alias/protected query registries must not be affected
# ---------------------------------------------------------------------------

def test_runtime_alias_registry_is_not_changed_by_embedding_aliases():
    assert "찐 고구마" not in FOOD_SEARCH_ALIAS_RULES
    assert "구운 고구마" not in FOOD_SEARCH_ALIAS_RULES
    assert "바나나" not in FOOD_SEARCH_ALIAS_RULES
    assert "사과" not in FOOD_SEARCH_ALIAS_RULES
    assert "계란빵" in PROTECTED_FOOD_SEARCH_QUERIES

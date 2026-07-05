from __future__ import annotations

from engines.food.food_embedding_text import (
    build_food_embedding_text,
    clean_embedding_text_part,
    clean_food_embedding_key,
    dedupe_text_parts,
    get_food_embedding_aliases,
)


def test_embedding_text_uses_embed_text_first():
    text = build_food_embedding_text({"embed_text": "달걀 삶은것", "name": "다른 이름"})

    assert text.startswith("달걀 삶은것")


def test_embedding_text_falls_back_to_name():
    assert build_food_embedding_text({"name": "바나나"}) == "바나나"


def test_embedding_text_falls_back_to_rep_name():
    assert build_food_embedding_text({"rep_name": "사과"}) == "사과"


def test_embedding_text_empty_row_returns_empty_string():
    assert build_food_embedding_text({}) == ""


def test_egg_raw_aliases_are_appended():
    text = build_food_embedding_text({"name": "달걀 생것"})

    assert "달걀 생것" in text
    assert "계란" in text
    assert "계란 생것" in text


def test_boiled_egg_aliases_are_appended():
    text = build_food_embedding_text({"name": "달걀 삶은것"})

    assert "달걀 삶은것" in text
    assert "삶은 계란" in text
    assert "삶은계란" in text
    assert "계란 삶은것" in text


def test_fried_egg_aliases_are_appended():
    text = build_food_embedding_text({"name": "달걀후라이"})

    assert "달걀후라이" in text
    assert "계란후라이" in text


def test_raw_chicken_breast_aliases_are_appended():
    text = build_food_embedding_text({"name": "닭고기 가슴(껍질 제거) 생것"})

    assert "닭고기 가슴(껍질 제거) 생것" in text
    assert "닭가슴살 생것" in text
    assert "닭 가슴살 생것" in text


def test_boiled_chicken_breast_aliases_are_appended():
    text = build_food_embedding_text({"name": "닭고기 가슴(껍질 제거) 삶은것"})

    assert "닭고기 가슴(껍질 제거) 삶은것" in text
    assert "삶은 닭가슴살" in text
    assert "삶은 닭 가슴살" in text
    assert "닭가슴살 삶은것" in text
    assert "닭 가슴살 삶은것" in text


def test_grilled_chicken_breast_aliases_are_appended():
    text = build_food_embedding_text({"name": "닭고기 가슴(껍질 제거) 구운것(팬)"})

    assert "닭고기 가슴(껍질 제거) 구운것(팬)" in text
    assert "구운 닭가슴살" in text
    assert "구운 닭 가슴살" in text
    assert "닭가슴살 구운것" in text
    assert "닭 가슴살 구운것" in text


def test_display_prefix_name_can_match_alias_key():
    text = build_food_embedding_text({"name": "[원재료성 식품] 달걀 삶은것"})

    assert "[원재료성 식품] 달걀 삶은것" in text
    assert "삶은 계란" in text


def test_clean_food_embedding_key_removes_display_prefix_only():
    assert clean_food_embedding_key("[원재료성 식품] 달걀 삶은것") == "달걀 삶은것"
    assert clean_food_embedding_key("[음식] 계란빵") == "계란빵"


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


def test_build_food_embedding_text_does_not_mutate_row():
    row = {"name": "달걀 생것"}
    before = dict(row)

    build_food_embedding_text(row)

    assert row == before


def test_embedding_text_dedupes_base_and_alias_parts():
    text = build_food_embedding_text({"embed_text": "  달걀   삶은것  ", "name": "달걀 삶은것"})

    assert text.split(" | ").count("달걀 삶은것") == 1
    assert text.split(" | ").count("삶은 계란") == 1


def test_clean_embedding_text_part_normalizes_whitespace():
    assert clean_embedding_text_part("  삶은   계란  ") == "삶은 계란"


def test_dedupe_text_parts_preserves_order_and_removes_empty_parts():
    assert dedupe_text_parts(["", "달걀", " 계란 ", "달걀", None]) == ["달걀", "계란"]


def test_unknown_sweet_potato_gets_no_broad_alias():
    text = build_food_embedding_text({"name": "고구마"})

    assert text == "고구마"
    assert "군고구마" not in text


def test_unknown_tofu_gets_no_broad_alias():
    text = build_food_embedding_text({"name": "두부"})

    assert text == "두부"
    assert "순두부" not in text


def test_aliases_can_be_found_from_rep_name_when_name_has_no_rule():
    aliases = get_food_embedding_aliases({"name": "달걀 대표", "rep_name": "달걀 생것"})

    assert aliases == ["계란", "계란 생것"]

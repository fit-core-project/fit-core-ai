from engines.quicklog.diet_parser import _build_diet_prompt, _deterministic_diet_parse


def test_deterministic_diet_parse_handles_korean_quantity_units():
    parsed = _deterministic_diet_parse("점심에 밥 반 공기랑 계란 한 개 먹었어")

    rice = next(item for item in parsed.items if item.food_name == "밥")
    egg = next(item for item in parsed.items if item.food_name == "계란")

    assert rice.amount == 0.5
    assert rice.unit == "공기"
    assert rice.carbs_g and rice.carbs_g > 0
    assert egg.amount == 1.0
    assert egg.unit == "개"
    assert egg.protein_g and egg.protein_g > 0


def test_deterministic_diet_parse_strips_workout_noise_before_food():
    parsed = _deterministic_diet_parse("벤치프레스 60kg 10회 3세트 하고 닭가슴살 200g 먹었어")

    assert len(parsed.items) == 1
    item = parsed.items[0]
    assert item.food_name == "닭가슴살"
    assert item.amount == 200
    assert item.unit == "g"
    assert item.protein_g and item.protein_g > 0


def test_deterministic_diet_parse_estimates_banana_unit_macros():
    parsed = _deterministic_diet_parse("바나나 1개 먹었어")

    assert len(parsed.items) == 1
    item = parsed.items[0]
    assert item.food_name == "바나나"
    assert item.amount == 1
    assert item.unit == "개"
    assert item.carbs_g and item.carbs_g > 0


def test_deterministic_diet_parse_adds_known_food_mentioned_without_amount():
    parsed = _deterministic_diet_parse("오늘 운동하고 계란 2개랑 바나나 먹었어")

    assert [item.food_name for item in parsed.items] == ["계란", "바나나"]


def test_deterministic_diet_parse_filters_non_food_noise():
    assert _deterministic_diet_parse("오늘 점심 잘 먹었어").items == []
    assert _deterministic_diet_parse("asdfjkl").items == []
    assert _deterministic_diet_parse("내일 날씨 어때").items == []


def test_diet_prompt_preserves_user_quantity_units():
    prompt_text = _build_diet_prompt().messages[0].prompt.template

    assert "Preserve the user's original quantity and unit" in prompt_text
    assert "Do not convert" in prompt_text
    assert "계란 2개" in prompt_text

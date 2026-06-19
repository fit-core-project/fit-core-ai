from engines.supplement.query_understanding import EntityType, IntentType, parse_supplement_query


def _canonicals(parsed):
    return {entity.canonical for entity in parsed.entities}


def _entity_types(parsed):
    return {(entity.type, entity.canonical) for entity in parsed.entities}


def test_parse_compact_magnesium_timing_question():
    parsed = parse_supplement_query("\ub9c8\uadf8\ub124\uc298\uc740 \uc5b8\uc81c\uba39\ub294 \uac8c \uc88b\uc544?")

    assert (EntityType.SUPPLEMENT_INGREDIENT, "magnesium") in _entity_types(parsed)
    assert IntentType.TIMING in parsed.intents


def test_parse_antibiotics_and_iron_as_drug_interaction():
    parsed = parse_supplement_query("항생제 먹는데 철분 먹어도 돼?")

    assert (EntityType.SUPPLEMENT_INGREDIENT, "iron") in _entity_types(parsed)
    assert (EntityType.DRUG_OR_DRUG_CLASS, "antibiotics") in _entity_types(parsed)
    assert IntentType.DRUG_INTERACTION in parsed.intents


def test_parse_iron_and_coffee_as_food_compound_interaction():
    parsed = parse_supplement_query("철분은 커피랑 같이 먹어도 돼?")

    assert (EntityType.SUPPLEMENT_INGREDIENT, "iron") in _entity_types(parsed)
    assert (EntityType.FOOD_OR_COMPOUND, "coffee") in _entity_types(parsed)
    assert IntentType.FOOD_COMPOUND_INTERACTION in parsed.intents


def test_parse_kidney_disease_and_magnesium_as_condition_safety():
    parsed = parse_supplement_query("신장질환 있는데 마그네슘 먹어도 돼?")

    assert (EntityType.SUPPLEMENT_INGREDIENT, "magnesium") in _entity_types(parsed)
    assert (EntityType.CONDITION, "kidney disease") in _entity_types(parsed)
    assert IntentType.CONDITION_SAFETY in parsed.intents


def test_parse_magnesium_timing_question():
    parsed = parse_supplement_query("마그네슘은 언제 먹는 게 좋아?")

    assert _canonicals(parsed) == {"magnesium"}
    assert IntentType.TIMING in parsed.intents


def test_parse_pregnancy_and_high_dose_vitamin_d_as_safety():
    parsed = parse_supplement_query("임신 중인데 비타민D 고용량 먹어도 돼?")

    assert (EntityType.SUPPLEMENT_INGREDIENT, "vitamin d") in _entity_types(parsed)
    assert (EntityType.RISK_CONTEXT, "pregnancy") in _entity_types(parsed)
    assert (EntityType.RISK_CONTEXT, "high dose") in _entity_types(parsed)
    assert IntentType.PREGNANCY_OR_HIGH_DOSE_SAFETY in parsed.intents


def test_parse_warfarin_and_omega3_as_drug_interaction():
    parsed = parse_supplement_query("와파린 먹는데 오메가3 먹어도 돼?")

    assert (EntityType.SUPPLEMENT_INGREDIENT, "omega3") in _entity_types(parsed)
    assert (EntityType.DRUG_OR_DRUG_CLASS, "warfarin") in _entity_types(parsed)
    assert IntentType.DRUG_INTERACTION in parsed.intents


def test_parse_acetaminophen_and_alcohol_as_medication_safety():
    parsed = parse_supplement_query("타이레놀 자주 먹고 술도 마시는데 영양제 먹어도 돼?")

    assert (EntityType.DRUG_OR_DRUG_CLASS, "acetaminophen") in _entity_types(parsed)
    assert "alcohol" in _canonicals(parsed) or "alcohol use" in _canonicals(parsed)
    assert IntentType.MEDICATION_SAFETY in parsed.intents


def test_parse_creatine_exercise_timing_question():
    parsed = parse_supplement_query("크레아틴은 운동 전이 좋아 후가 좋아?")

    assert _canonicals(parsed) == {"creatine"}
    assert IntentType.TIMING in parsed.intents


def test_parse_unknown_when_no_known_entity_or_intent():
    parsed = parse_supplement_query("오늘 컨디션 어때?")

    assert parsed.raw_query == "오늘 컨디션 어때?"
    assert parsed.entities == ()
    assert parsed.intents == (IntentType.UNKNOWN,)


def test_parse_multivitamin_omega3_silymarin_combination():
    parsed = parse_supplement_query("종합비타민 오메가3 실리마린 같이 먹어도 돼?")

    assert {"multivitamin", "omega3", "silymarin"} <= _canonicals(parsed)
    assert IntentType.SUPPLEMENT_INTERACTION in parsed.intents


def test_parse_probiotic_antibiotics_as_drug_timing_interaction():
    parsed = parse_supplement_query("유산균 항생제랑 같이 먹어도 돼?")

    assert (EntityType.SUPPLEMENT_INGREDIENT, "probiotics") in _entity_types(parsed)
    assert (EntityType.DRUG_OR_DRUG_CLASS, "antibiotics") in _entity_types(parsed)
    assert IntentType.DRUG_INTERACTION in parsed.intents


def test_parse_zinc_iron_as_absorption_interaction():
    parsed = parse_supplement_query("아연이랑 철분 같이 먹어도 돼?")

    assert (EntityType.SUPPLEMENT_INGREDIENT, "zinc") in _entity_types(parsed)
    assert (EntityType.SUPPLEMENT_INGREDIENT, "iron") in _entity_types(parsed)
    assert IntentType.SUPPLEMENT_INTERACTION in parsed.intents


def test_parse_kidney_disease_and_protein_condition_safety():
    parsed = parse_supplement_query("신장질환 있는데 프로틴 먹어도 돼?")

    assert (EntityType.SUPPLEMENT_INGREDIENT, "protein") in _entity_types(parsed)
    assert (EntityType.CONDITION, "kidney disease") in _entity_types(parsed)
    assert IntentType.CONDITION_SAFETY in parsed.intents


def test_parse_pregnancy_and_multivitamin_context():
    parsed = parse_supplement_query("임산부 종합비타민 먹어도 돼?")

    assert (EntityType.SUPPLEMENT_INGREDIENT, "multivitamin") in _entity_types(parsed)
    assert (EntityType.RISK_CONTEXT, "pregnancy") in _entity_types(parsed)
    assert IntentType.PREGNANCY_OR_HIGH_DOSE_SAFETY in parsed.intents


def test_parse_symptom_after_intake_question():
    parsed = parse_supplement_query("먹고 속이 안 좋은데 계속 먹어도 돼?")

    assert IntentType.SYMPTOM_AFTER_INTAKE in parsed.intents
    assert IntentType.SIDE_EFFECT in parsed.intents

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from engines.supplement.kb_schema import (
    CautionLevel,
    CoverageStatus,
    IngredientProfile,
    InteractionRule,
    InteractionType,
    SafetyRule,
)
from engines.supplement.query_understanding import ALIAS_REGISTRY, ENTITY_LABELS, EntityType


def _roundtrip(model):
    return type(model).model_validate_json(model.model_dump_json())


def test_ingredient_profile_magnesium_fixture_roundtrips():
    profile = IngredientProfile.model_validate(
        {
            "type": "ingredient_profile",
            "id": "ING_MAGNESIUM",
            "name": "마그네슘",
            "canonical": "magnesium",
            "aliases": ["magnesium", "Mg"],
            "category": "mineral",
            "coverage_status": "covered",
            "source_refs": ["NIH_ODS_MAGNESIUM_HP"],
            "common_uses": ["일반적인 영양 보충"],
            "timing": {
                "general": "식후나 저녁 시간대 복용이 일반적으로 무난합니다.",
                "with_food": "속이 불편하면 식후 복용을 고려할 수 있습니다.",
                "before_bed": "저녁 시간대에 복용할 수 있습니다.",
                "exercise_related": "",
            },
            "spacing": ["갑상선약, 일부 항생제, 철분, 칼슘과는 간격을 둡니다."],
            "cautions": ["신장질환이 있으면 전문가 상담이 필요합니다."],
            "high_risk_groups": ["kidney disease"],
            "related_interactions": ["INT_MAGNESIUM_THYROID_MEDICATION"],
        }
    )

    assert profile.type == "ingredient_profile"
    assert profile.id == "ING_MAGNESIUM"
    assert profile.canonical == "magnesium"
    assert profile.coverage_status == CoverageStatus.COVERED
    assert profile.aliases == ["magnesium", "Mg"]
    assert _roundtrip(profile) == profile


def test_ingredient_profile_vitamin_d_fixture_roundtrips():
    profile = IngredientProfile.model_validate(
        {
            "type": "ingredient_profile",
            "id": "ING_VITAMIN_D",
            "name": "비타민D",
            "canonical": "vitamin d",
            "aliases": ["비타민 D", "vitamin d"],
            "category": "vitamin",
            "coverage_status": "covered",
            "source_refs": ["NIH_ODS_VITAMIN_D_HP"],
            "timing": {"general": "지방이 포함된 식사 후 복용하면 흡수에 유리할 수 있습니다."},
            "cautions": ["고용량 복용은 제품 라벨과 전문가 조언을 따르세요."],
        }
    )

    assert profile.timing.general
    assert profile.common_uses == []
    assert _roundtrip(profile) == profile


def test_interaction_rule_iron_caffeine_fixture_roundtrips():
    rule = InteractionRule.model_validate(
        {
            "type": "interaction_rule",
            "id": "INT_IRON_CAFFEINE",
            "entity_a": {"type": "supplement_ingredient", "canonical": "iron", "label": "철분"},
            "entity_b": {"type": "food_or_compound", "canonical": "caffeine", "label": "카페인"},
            "interaction_type": "absorption_interference",
            "coverage_status": "covered",
            "source_refs": ["NIH_ODS_IRON_HP", "MEDLINEPLUS_CAFFEINE"],
            "mechanism": "커피나 카페인 음료는 철분 흡수를 방해할 수 있습니다.",
            "recommendation": "함께 복용하지 말고 간격을 두는 방식을 고려합니다.",
            "spacing_guidance": "보통 2시간 이상 간격을 둡니다.",
            "caution_level": "moderate",
            "consultation_required_when": ["빈혈 치료 중", "임신 중"],
        }
    )

    assert rule.entity_a.type == EntityType.SUPPLEMENT_INGREDIENT
    assert rule.entity_b.type == EntityType.FOOD_OR_COMPOUND
    assert rule.interaction_type == InteractionType.ABSORPTION_INTERFERENCE
    assert rule.caution_level == CautionLevel.MODERATE
    assert _roundtrip(rule) == rule


def test_interaction_rule_omega3_warfarin_fixture_roundtrips():
    rule = InteractionRule.model_validate(
        {
            "type": "interaction_rule",
            "id": "INT_OMEGA3_WARFARIN",
            "entity_a": {"type": "supplement_ingredient", "canonical": "omega3", "label": "오메가3"},
            "entity_b": {"type": "drug_or_drug_class", "canonical": "warfarin", "label": "와파린"},
            "interaction_type": "bleeding_risk",
            "coverage_status": "covered",
            "source_refs": ["NIH_ODS_OMEGA3_HP"],
            "mechanism": "출혈 위험과 관련해 보수적으로 확인해야 합니다.",
            "recommendation": "복용 전 의사 또는 약사와 상담합니다.",
            "caution_level": "high",
            "consultation_required_when": ["항응고제 복용 중", "수술 예정"],
        }
    )

    assert rule.entity_b.type == EntityType.DRUG_OR_DRUG_CLASS
    assert rule.interaction_type == InteractionType.BLEEDING_RISK
    assert rule.caution_level == CautionLevel.HIGH
    assert _roundtrip(rule) == rule


def test_safety_rule_kidney_disease_magnesium_fixture_roundtrips():
    rule = SafetyRule.model_validate(
        {
            "type": "safety_rule",
            "id": "SAFE_KIDNEY_MAGNESIUM",
            "trigger_conditions": ["kidney disease", "신장질환"],
            "affected_entities": ["magnesium"],
            "risk": "신장 기능이 저하된 경우 마그네슘 축적 위험을 고려해야 합니다.",
            "recommendation": "복용 전 전문가와 상담합니다.",
            "coverage_status": "covered",
            "when_to_consult": "신장질환이 있거나 신장 기능 관련 진단을 받은 경우",
            "caution_level": "high",
            "evidence_summary": "일반 안전 guardrail 문서 기반",
            "source_refs": ["supplement_ingredient_profiles.json#ING_MAGNESIUM"],
        }
    )

    assert rule.caution_level == CautionLevel.HIGH
    assert rule.affected_entities == ["magnesium"]
    assert _roundtrip(rule) == rule


def test_safety_rule_pregnancy_high_dose_vitamin_d_fixture_roundtrips():
    rule = SafetyRule.model_validate(
        {
            "type": "safety_rule",
            "id": "SAFE_PREGNANCY_HIGH_DOSE_VITAMIN_D",
            "trigger_conditions": ["pregnancy", "임신", "high dose", "고용량"],
            "affected_entities": ["vitamin d"],
            "risk": "고용량 복용은 혈중 칼슘 상승 등 위험이 있을 수 있습니다.",
            "recommendation": "제품 라벨과 전문가 조언을 따릅니다.",
            "coverage_status": "partial",
            "when_to_consult": "임신 중이거나 고용량 복용을 고려하는 경우",
            "caution_level": "high",
        }
    )

    assert "pregnancy" in rule.trigger_conditions
    assert rule.affected_entities == ["vitamin d"]
    assert _roundtrip(rule) == rule


def test_type_is_fixed_per_schema():
    with pytest.raises(ValidationError):
        IngredientProfile.model_validate(
            {
                "type": "interaction_rule",
                "id": "ING_BAD",
                "name": "bad",
                "canonical": "bad",
                "aliases": ["bad"],
                "category": "bad",
            }
        )


def test_invalid_caution_level_fails_validation():
    with pytest.raises(ValidationError):
        SafetyRule.model_validate(
            {
                "type": "safety_rule",
                "id": "SAFE_BAD",
                "trigger_conditions": ["condition"],
                "affected_entities": ["entity"],
                "risk": "risk",
                "recommendation": "recommendation",
                "caution_level": "medium",
            }
        )


def test_blank_id_fails_validation():
    with pytest.raises(ValidationError):
        IngredientProfile.model_validate(
            {
                "type": "ingredient_profile",
                "id": " ",
                "name": "마그네슘",
                "canonical": "magnesium",
                "aliases": ["magnesium"],
                "category": "mineral",
            }
        )


def test_aliases_must_be_a_list():
    with pytest.raises(ValidationError):
        IngredientProfile.model_validate(
            {
                "type": "ingredient_profile",
                "id": "ING_BAD",
                "name": "마그네슘",
                "canonical": "magnesium",
                "aliases": "magnesium",
                "category": "mineral",
            }
        )


def test_entity_ref_requires_type_canonical_and_label():
    with pytest.raises(ValidationError):
        InteractionRule.model_validate(
            {
                "type": "interaction_rule",
                "id": "INT_BAD",
                "entity_a": {"type": "supplement_ingredient", "canonical": "iron"},
                "entity_b": {"type": "food_or_compound", "canonical": "caffeine", "label": "카페인"},
                "interaction_type": "absorption_interference",
                "caution_level": "moderate",
            }
        )


def test_supplement_ingredient_profile_corpus_validates_against_schema():
    corpus_path = Path("data/documents/supplement_ingredient_profiles.json")
    data = json.loads(corpus_path.read_text(encoding="utf-8"))

    profiles = [IngredientProfile.model_validate(item) for item in data]

    assert [profile.id for profile in profiles] == [
        "ING_MAGNESIUM",
        "ING_CREATINE",
        "ING_VITAMIN_D",
        "ING_OMEGA3",
        "ING_IRON",
        "ING_MULTIVITAMIN",
        "ING_SILYMARIN",
        "ING_PROBIOTIC",
        "ING_ZINC",
        "ING_PROTEIN",
        "ING_CALCIUM",
        "ING_VITAMIN_C",
        "ING_VITAMIN_E",
        "ING_VITAMIN_A",
        "ING_VITAMIN_B",
        "ING_THEANINE",
        "ING_MELATONIN",
        "ING_EAA",
        "ING_GLUTAMINE",
        "ING_BERBERINE",
        "ING_QUERCETIN",
        "ING_NAC",
    ]
    assert {profile.type for profile in profiles} == {"ingredient_profile"}
    assert all(profile.aliases for profile in profiles)
    assert all(profile.canonical for profile in profiles)
    assert all(profile.coverage_status in CoverageStatus for profile in profiles)
    assert all(profile.source_refs for profile in profiles)
    assert all(profile.timing.general for profile in profiles)
    assert all(profile.cautions for profile in profiles)


def test_supplement_interaction_rule_corpus_validates_against_schema():
    corpus_path = Path("data/documents/supplement_interaction_rules.json")
    data = json.loads(corpus_path.read_text(encoding="utf-8"))

    rules = [InteractionRule.model_validate(item) for item in data]

    assert [rule.id for rule in rules] == [
        "INT_IRON_CAFFEINE",
        "INT_IRON_CALCIUM",
        "INT_IRON_ANTIBIOTICS",
        "INT_MAGNESIUM_THYROID_MEDICATION",
        "INT_MAGNESIUM_ANTIBIOTICS",
        "INT_ZINC_ANTIBIOTICS",
        "INT_CALCIUM_IRON",
        "INT_OMEGA3_ANTICOAGULANTS",
        "INT_PROBIOTIC_ANTIBIOTICS",
        "INT_ZINC_IRON_CALCIUM",
        "INT_MULTIVITAMIN_MINERAL_OVERLAP",
        "INT_SILYMARIN_MEDICATIONS",
        "INT_VITAMIN_C_IRON",
        "INT_OMEGA3_VITAMIN_E",
        "INT_OMEGA3_ASPIRIN",
        "INT_MAGNESIUM_BLOOD_PRESSURE_MEDICATION",
        "INT_VITAMIN_D_DIABETES_MEDICATION",
        "INT_BERBERINE_DIABETES_MEDICATION",
        "INT_MELATONIN_BLOOD_PRESSURE_MEDICATION",
    ]
    assert {rule.type for rule in rules} == {"interaction_rule"}
    assert all(rule.entity_a.canonical for rule in rules)
    assert all(rule.entity_b.canonical for rule in rules)
    assert all(rule.mechanism for rule in rules)
    assert all(rule.recommendation for rule in rules)
    assert all(rule.coverage_status in CoverageStatus for rule in rules)
    assert all(rule.source_refs for rule in rules)
    assert all(rule.caution_level in CautionLevel for rule in rules)


def test_supplement_safety_rule_corpus_validates_against_schema():
    corpus_path = Path("data/documents/supplement_safety_rules.json")
    data = json.loads(corpus_path.read_text(encoding="utf-8"))

    rules = [SafetyRule.model_validate(item) for item in data]

    assert [rule.id for rule in rules] == [
        "SAFE_KIDNEY_MAGNESIUM",
        "SAFE_KIDNEY_CREATINE",
        "SAFE_PREGNANCY_HIGH_DOSE_VITAMIN_D",
        "SAFE_OMEGA3_SURGERY",
        "SAFE_ACETAMINOPHEN_LIVER_ALCOHOL",
        "SAFE_CAFFEINE_SLEEP",
        "SAFE_CAFFEINE_HYPERTENSION",
        "SAFE_KIDNEY_PROTEIN",
        "SAFE_IMMUNOCOMPROMISED_PROBIOTIC",
        "SAFE_HIGH_DOSE_ZINC",
        "SAFE_LIVER_SILYMARIN",
        "SAFE_PREGNANCY_MULTIVITAMIN",
        "SAFE_LIVER_VITAMIN_A",
        "SAFE_HIGH_DOSE_VITAMIN_E",
        "SAFE_DIGESTIVE_ZINC",
        "SAFE_BERBERINE_DIGESTIVE_MEDICATION",
        "SAFE_SLEEP_SUPPLEMENTS_MEDICATION",
    ]
    assert {rule.type for rule in rules} == {"safety_rule"}
    assert all(rule.trigger_conditions for rule in rules)
    assert all(rule.affected_entities for rule in rules)
    assert all(rule.risk for rule in rules)
    assert all(rule.recommendation for rule in rules)
    assert all(rule.coverage_status in CoverageStatus for rule in rules)
    assert all(rule.source_refs for rule in rules)
    assert all(rule.caution_level in CautionLevel for rule in rules)


def test_supplement_canonical_entities_corpus_is_parser_source():
    corpus_path = Path("data/documents/supplement_canonical_entities.json")
    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    entities = data["entities"]

    assert data["version"] == "supplement-canonical-entities-v1.1-2026-07-03"
    assert entities
    canonicals = [item["canonical"] for item in entities]
    assert len(canonicals) == len(set(canonicals))
    assert {"magnesium", "protein", "probiotics", "nmn"} <= set(canonicals)

    registry_keys = {(entry.entity_type.value, entry.canonical) for entry in ALIAS_REGISTRY}
    for item in entities:
        assert item["entityType"] in EntityType._value2member_map_
        assert item["aliases"]
        assert item["displayNameKo"]
        assert item["coverageStatus"] in {"covered", "partial", "unsupported"}
        assert "sourceRefs" in item
        assert (item["entityType"], item["canonical"]) in registry_keys

    assert ENTITY_LABELS["magnesium"] == "마그네슘"
    assert ENTITY_LABELS["nmn"] == "NMN"


def test_supplement_kb_source_refs_exist_in_active_source_map():
    canonical_path = Path("data/documents/supplement_canonical_entities.json")
    canonical_data = json.loads(canonical_path.read_text(encoding="utf-8"))
    source_map_path = Path(canonical_data["sourceMap"])
    source_map = json.loads(source_map_path.read_text(encoding="utf-8"))

    source_ids = {
        ref["id"]
        for entity in source_map["entities"]
        for ref in entity.get("sourceRefs", [])
    }
    assert source_ids

    for corpus_path in [
        Path("data/documents/supplement_ingredient_profiles.json"),
        Path("data/documents/supplement_interaction_rules.json"),
        Path("data/documents/supplement_safety_rules.json"),
    ]:
        for row in json.loads(corpus_path.read_text(encoding="utf-8")):
            assert row["source_refs"], row["id"]
            unknown = set(row["source_refs"]) - source_ids
            assert not unknown, f"{corpus_path}#{row['id']} unknown source refs: {sorted(unknown)}"

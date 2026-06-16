from engines.supplement.query_understanding import parse_supplement_query
from engines.supplement.response_composer import compose_supplement_response


class Doc:
    def __init__(self, doc_id, source, page_content):
        self.metadata = {"id": doc_id, "source": source}
        self.page_content = page_content


def _u(text):
    return text.encode("ascii").decode("unicode_escape")


def test_interaction_rule_caution_is_merged_and_answer_is_softened():
    doc = Doc(
        "INT_OMEGA3_ANTICOAGULANTS",
        "interaction_rule",
        "\n".join(
            [
                "[Interaction rule]",
                "ID: INT_OMEGA3_ANTICOAGULANTS",
                "Entity A: 오메가3 / omega3 / supplement_ingredient",
                "Entity B: 항응고제 / anticoagulant / drug_or_drug_class",
                "Interaction type: bleeding_risk",
                "Mechanism: 오메가3 보충제는 항응고제 복용자나 출혈 위험이 있는 상황에서 보수적으로 확인해야 합니다.",
                "Recommendation: 항응고제나 와파린을 복용 중이면 오메가3 보충제를 임의로 시작하지 말고 의료진과 상담하세요.",
                "Spacing guidance: 복용 간격만으로 위험이 해결된다고 단정하기 어렵습니다.",
                "Caution level: high",
                "Consultation required when:",
                "  - 항응고제 복용 중",
                "  - 와파린 복용 중",
            ]
        ),
    )

    result = compose_supplement_response(
        question="와파린 먹는데 오메가3 먹어도 돼?",
        parsed_query=parse_supplement_query("와파린 먹는데 오메가3 먹어도 돼?"),
        generated_answer="같이 먹어도 됩니다. 문제 없습니다.",
        generated_caution=None,
        selected_docs=[doc],
    )

    assert isinstance(result.answer, str)
    assert "먹어도 됩니다" not in result.answer
    assert "문제 없습니다" not in result.answer
    assert result.caution
    assert "항응고제" in result.caution or "출혈" in result.caution
    assert "상담" in result.caution


def test_safety_rule_caution_is_merged_for_kidney_magnesium():
    doc = Doc(
        "SAFE_KIDNEY_MAGNESIUM",
        "safety_rule",
        "\n".join(
            [
                "[Safety rule]",
                "ID: SAFE_KIDNEY_MAGNESIUM",
                "Trigger conditions:",
                "  - 신장질환",
                "Affected entities:",
                "  - magnesium",
                "Risk: 신장 기능이 저하된 경우 마그네슘 배출이 원활하지 않을 수 있습니다.",
                "Recommendation: 신장질환이 있으면 마그네슘 보충제를 임의로 시작하지 말고 의료진과 상담하세요.",
                "When to consult: 신장질환, 신장 기능 저하, 투석, 처방약 복용",
                "Caution level: high",
            ]
        ),
    )

    result = compose_supplement_response(
        question="신장질환 있는데 마그네슘 먹어도 돼?",
        parsed_query=parse_supplement_query("신장질환 있는데 마그네슘 먹어도 돼?"),
        generated_answer="마그네슘은 식후 복용할 수 있습니다.",
        generated_caution=None,
        selected_docs=[doc],
    )

    assert "안전합니다" not in result.answer
    assert result.caution
    assert "신장" in result.caution
    assert "상담" in result.caution or "의료진" in result.caution


def test_medication_safety_caution_is_merged_for_acetaminophen_alcohol():
    doc = Doc(
        "SAFE_ACETAMINOPHEN_LIVER_ALCOHOL",
        "safety_rule",
        "\n".join(
            [
                "[Safety rule]",
                "ID: SAFE_ACETAMINOPHEN_LIVER_ALCOHOL",
                "Trigger conditions:",
                "  - 간질환",
                "  - 음주",
                "Affected entities:",
                "  - acetaminophen",
                "Risk: 아세트아미노펜 성분은 간질환, 잦은 음주, 고용량 사용 상황에서 간 손상 위험을 확인해야 합니다.",
                "Recommendation: 타이레놀을 자주 복용하고 음주도 한다면 의사 또는 약사와 상담하세요.",
                "When to consult: 간질환, 잦은 음주, 고용량 또는 장기 복용",
                "Caution level: high",
            ]
        ),
    )

    result = compose_supplement_response(
        question="타이레놀 자주 먹고 술도 마시는데 괜찮아?",
        parsed_query=parse_supplement_query("타이레놀 자주 먹고 술도 마시는데 괜찮아?"),
        generated_answer="{}",
        generated_caution=None,
        selected_docs=[doc],
    )

    assert result.used_kb_fallback is True
    assert result.caution
    assert "간질환" in result.caution or "음주" in result.caution or "고용량" in result.caution
    assert "상담" in result.caution


def test_json_answer_is_not_exposed_and_kb_fallback_is_used():
    doc = Doc(
        "INT_OMEGA3_ANTICOAGULANTS",
        "interaction_rule",
        "\n".join(
            [
                "[Interaction rule]",
                "Recommendation: 항응고제 복용 중이면 의료진과 상담하세요.",
                "Caution level: high",
            ]
        ),
    )

    result = compose_supplement_response(
        question="와파린 먹는데 오메가3 먹어도 돼?",
        parsed_query=parse_supplement_query("와파린 먹는데 오메가3 먹어도 돼?"),
        generated_answer='{"answer":"{}","caution":"의사 또는 약사와 상담하세요."}',
        generated_caution=None,
        selected_docs=[doc],
    )

    assert result.used_kb_fallback is True
    assert result.answer != "{}"
    assert not result.answer.strip().startswith("{")
    assert result.caution
    assert "상담" in result.caution


def test_invalid_json_like_answer_uses_kb_fallback():
    doc = Doc(
        "SAFE_KIDNEY_MAGNESIUM",
        "safety_rule",
        "\n".join(
            [
                "[Safety rule]",
                "Risk: kidney disease requires caution",
                "Recommendation: consult a clinician before magnesium supplementation",
                "Caution level: high",
            ]
        ),
    )

    result = compose_supplement_response(
        question="신장질환 있는데 마그네슘 먹어도 돼?",
        parsed_query=parse_supplement_query("신장질환 있는데 마그네슘 먹어도 돼?"),
        generated_answer='{"**consult before use** this is not valid json',
        generated_caution=None,
        selected_docs=[doc],
    )

    assert result.used_kb_fallback is True
    assert not result.answer.strip().startswith("{")
    assert result.caution


def test_condition_safety_fallback_prefers_safety_rule_over_interaction_rule():
    safety = Doc(
        "SAFE_KIDNEY_MAGNESIUM",
        "safety_rule",
        "\n".join(
            [
                "[Safety rule]",
                "Risk: kidney disease requires caution",
                "Recommendation: consult a clinician before magnesium supplementation",
                "Caution level: high",
            ]
        ),
    )
    interaction = Doc(
        "INT_MAGNESIUM_ANTIBIOTICS",
        "interaction_rule",
        "\n".join(
            [
                "[Interaction rule]",
                "Recommendation: separate dosing from antibiotics",
                "Caution level: high",
            ]
        ),
    )

    result = compose_supplement_response(
        question="신장질환 있는데 마그네슘 먹어도 돼?",
        parsed_query=parse_supplement_query("신장질환 있는데 마그네슘 먹어도 돼?"),
        generated_answer="{}",
        generated_caution=None,
        selected_docs=[interaction, safety],
    )

    assert "검색된 안전 규칙" in result.answer
    assert "검색된 상호작용 규칙" not in result.answer


def test_recommendation_json_answer_is_promoted_without_key_flattening():
    result = compose_supplement_response(
        question="신장질환 있는데 마그네슘 먹어도 돼?",
        parsed_query=parse_supplement_query("신장질환 있는데 마그네슘 먹어도 돼?"),
        generated_answer='{"caution_level":"high","recommendation":"신장질환이 있으면 마그네슘 복용 전 의료진과 상담하세요.","details":[{"title":"주의"}]}',
        generated_caution=None,
        selected_docs=[],
    )

    assert result.answer == "신장질환이 있으면 마그네슘 복용 전 의료진과 상담하세요."
    assert not result.answer.startswith("caution_level")


def test_interaction_summary_json_answer_is_promoted_without_key_flattening():
    result = compose_supplement_response(
        question="와파린 먹는데 오메가3 먹어도 돼?",
        parsed_query=parse_supplement_query("와파린 먹는데 오메가3 먹어도 돼?"),
        generated_answer='{"Interaction_Summary":"와파린 복용 중이면 오메가3는 의료진과 상담하세요.","Detailed_Advice":[{"Risk":"bleeding"}]}',
        generated_caution=None,
        selected_docs=[],
    )

    assert result.answer == "와파린 복용 중이면 오메가3는 의료진과 상담하세요."
    assert not result.answer.startswith("Interaction_Summary")
    assert "Detailed_Advice" not in result.answer


def test_risk_caution_is_softened():
    query = _u("\\uc640\\ud30c\\ub9b0 \\uba39\\ub294\\ub370 \\uc624\\uba54\\uac003 \\uba39\\uc5b4\\ub3c4 \\ub3fc?")
    unsafe_caution = _u("\\uac19\\uc774 \\uba39\\uc5b4\\ub3c4 \\ub429\\ub2c8\\ub2e4. \\uc548\\uc804\\ud569\\ub2c8\\ub2e4.")

    result = compose_supplement_response(
        question=query,
        parsed_query=parse_supplement_query(query),
        generated_answer=_u("\\uac1c\\uc778 \\uc0c1\\ud0dc\\uc5d0 \\ub530\\ub77c \\ub2ec\\ub77c\\uc9c8 \\uc218 \\uc788\\uc2b5\\ub2c8\\ub2e4."),
        generated_caution=unsafe_caution,
        selected_docs=[],
    )

    assert result.caution
    assert _u("\\uba39\\uc5b4\\ub3c4 \\ub429\\ub2c8\\ub2e4") not in result.caution
    assert _u("\\uc548\\uc804\\ud569\\ub2c8\\ub2e4") not in result.caution
    assert _u("\\uc0c1\\ub2f4") in result.caution


def test_food_interaction_caution_is_softened():
    query = "철분은 커피랑 같이 먹어도 돼?"

    result = compose_supplement_response(
        question=query,
        parsed_query=parse_supplement_query(query),
        generated_answer="시간 간격을 두는 것이 좋습니다.",
        generated_caution="같이 먹어도 됩니다. 안전합니다.",
        selected_docs=[],
    )

    assert result.caution
    assert "먹어도 됩니다" not in result.caution
    assert "안전합니다" not in result.caution
    assert "상담" in result.caution


def test_timing_fallback_prefers_profile_over_safety_rule_for_answer():
    profile = Doc(
        "ING_VITAMIN_D",
        "ingredient_profile",
        "\n".join(
            [
                "[Ingredient profile]",
                "ID: ING_VITAMIN_D",
                "Name: vitamin d",
                "Cautions:",
                "  - high dose requires consultation",
            ]
        ),
    )
    safety = Doc(
        "SAFE_PREGNANCY_HIGH_DOSE_VITAMIN_D",
        "safety_rule",
        "\n".join(
            [
                "[Safety rule]",
                "Risk: high dose risk",
                "Recommendation: consult a clinician",
                "Caution level: high",
            ]
        ),
    )
    query = _u("\\ube44\\ud0c0\\ubbfcD\\ub294 \\uc2dd\\ud6c4\\uc5d0 \\uba39\\ub294 \\uac8c \\uc88b\\uc544?")

    result = compose_supplement_response(
        question=query,
        parsed_query=parse_supplement_query(query),
        generated_answer="{}",
        generated_caution=None,
        selected_docs=[safety, profile],
    )

    assert result.used_kb_fallback is True
    assert _u("\\ubcf5\\uc6a9 \\ud0c0\\uc774\\ubc0d") in result.answer
    assert _u("\\uc9c8\\ud658, \\uc784\\uc2e0/\\uc218\\uc720") not in result.answer


def test_timing_answer_keeps_plain_text_and_profile_caution():
    doc = Doc(
        "ING_MAGNESIUM",
        "ingredient_profile",
        "\n".join(
            [
                "[Ingredient profile]",
                "ID: ING_MAGNESIUM",
                "Name: 마그네슘",
                "Timing:",
                "  - General: 식후나 저녁 시간대 복용이 일반적으로 무난합니다.",
                "Cautions:",
                "  - 신장질환이 있으면 전문가 상담이 필요합니다.",
            ]
        ),
    )

    result = compose_supplement_response(
        question="마그네슘은 언제 먹는 게 좋아?",
        parsed_query=parse_supplement_query("마그네슘은 언제 먹는 게 좋아?"),
        generated_answer="마그네슘은 식후나 저녁 시간대에 복용하는 방식이 일반적으로 무난합니다.",
        generated_caution=None,
        selected_docs=[doc],
    )

    assert "마그네슘" in result.answer
    assert result.caution is None or "신장질환" in result.caution

def test_warfarin_omega3_caution_is_compacted():
    query = _u("\\uc640\\ud30c\\ub9b0 \\uba39\\ub294\\ub370 \\uc624\\uba54\\uac003 \\uba39\\uc5b4\\ub3c4 \\ub3fc?")
    interaction = Doc(
        "INT_OMEGA3_ANTICOAGULANTS",
        "interaction_rule",
        "\n".join(
            [
                "[Interaction rule]",
                "Recommendation: "
                + _u("\\uc640\\ud30c\\ub9b0 \\ub4f1 \\ud56d\\uc751\\uace0\\uc81c\\ub97c \\ubcf5\\uc6a9 \\uc911\\uc774\\uba74 \\uc624\\uba54\\uac003 \\ubcf4\\ucda9\\uc81c\\ub97c \\uc784\\uc758\\ub85c \\uc2dc\\uc791\\ud558\\uac70\\ub098 \\uace0\\uc6a9\\ub7c9\\uc73c\\ub85c \\ub298\\ub9ac\\uc9c0 \\ub9c8\\uc138\\uc694."),
                "Spacing guidance: "
                + _u("\\ucd9c\\ud608 \\uc704\\ud5d8\\uc774 \\uac1c\\uc778 \\uc0c1\\ud0dc\\uc640 \\uc57d\\ubb3c \\uc6a9\\ub7c9\\uc5d0 \\ub530\\ub77c \\ub2ec\\ub77c\\uc9c8 \\uc218 \\uc788\\uc73c\\ubbc0\\ub85c \\ubcf5\\uc6a9 \\uc804 \\uc758\\uc0ac \\ub610\\ub294 \\uc57d\\uc0ac\\uc640 \\uc0c1\\ub2f4\\ud558\\uc138\\uc694."),
                "Consultation required when:",
                "  - " + _u("\\uc218\\uc220 \\ub610\\ub294 \\uc2dc\\uc220 \\uc608\\uc815"),
                "  - " + _u("\\uba4d\\uc774\\ub098 \\ucd9c\\ud608\\uc774 \\uc798 \\uc0dd\\uae40"),
            ]
        ),
    )
    safety = Doc(
        "SAFE_OMEGA3_SURGERY",
        "safety_rule",
        "\n".join(
            [
                "[Safety rule]",
                "Risk: " + _u("\\uc218\\uc220\\u00b7\\uc2dc\\uc220 \\uc804\\ud6c4\\uc5d0\\ub294 \\ucd9c\\ud608 \\uc704\\ud5d8\\uc744 \\uc758\\ub8cc\\uc9c4\\uc5d0\\uac8c \\uc54c\\ub824\\uc57c \\ud569\\ub2c8\\ub2e4."),
                "Recommendation: " + _u("\\uc758\\ub8cc\\uc9c4\\uacfc \\uc0c1\\ub2f4\\ud558\\uc138\\uc694."),
            ]
        ),
    )
    profile = Doc(
        "ING_OMEGA3",
        "ingredient_profile",
        "\n".join(
            [
                "[Ingredient profile]",
                "Cautions:",
                "  - " + _u("\\ubcf5\\uc6a9 \\uc911\\uc778 \\uc57d\\uc774 \\uc788\\uc73c\\uba74 \\uc758\\uc0ac \\ub610\\ub294 \\uc57d\\uc0ac\\uc640 \\uc0c1\\ub2f4\\ud558\\uc138\\uc694."),
            ]
        ),
    )

    result = compose_supplement_response(
        question=query,
        parsed_query=parse_supplement_query(query),
        generated_answer=_u("\\uac1c\\uc778 \\uc0c1\\ud0dc\\uc5d0 \\ub530\\ub77c \\ub2ec\\ub77c\\uc9c8 \\uc218 \\uc788\\uc2b5\\ub2c8\\ub2e4."),
        generated_caution=_u("\\uc758\\ub8cc\\uc9c4\\uacfc \\uc0c1\\ub2f4\\ud558\\uc138\\uc694. \\uc758\\uc0ac \\ub610\\ub294 \\uc57d\\uc0ac\\uc640 \\uc0c1\\ub2f4\\ud558\\uc138\\uc694."),
        selected_docs=[interaction, safety, profile],
    )

    assert result.caution
    assert len(result.caution) <= 520
    assert _u("\\uc640\\ud30c\\ub9b0") in result.caution or _u("\\ud56d\\uc751\\uace0") in result.caution
    assert _u("\\ucd9c\\ud608") in result.caution
    assert _u("\\uc0c1\\ub2f4") in result.caution or _u("\\uc758\\ub8cc\\uc9c4") in result.caution
    assert _u("\\uc784\\uc758") in result.caution or _u("\\uace0\\uc6a9\\ub7c9") in result.caution
    assert result.caution.count(_u("\\uc0c1\\ub2f4")) <= 2


def test_magnesium_timing_caution_is_compacted():
    query = _u("\\ub9c8\\uadf8\\ub124\\uc298\\uc740 \\uc5b8\\uc81c\\uba39\\ub294 \\uac8c \\uc88b\\uc544?")
    profile = Doc(
        "ING_MAGNESIUM",
        "ingredient_profile",
        "\n".join(
            [
                "[Ingredient profile]",
                "Cautions:",
                "  - " + _u("\\uc784\\uc2e0 \\uc911\\uc774\\uac70\\ub098 \\ub2e4\\ub978 \\uc57d\\uc744 \\ubcf5\\uc6a9 \\uc911\\uc774\\uba74 \\uac1c\\uc778 \\uc0c1\\ud0dc\\uc5d0 \\ub9de\\ucdb0 \\ud655\\uc778\\ud558\\ub294 \\uac83\\uc774 \\uc548\\uc804\\ud569\\ub2c8\\ub2e4."),
            ]
        ),
    )
    kidney = Doc(
        "SAFE_KIDNEY_MAGNESIUM",
        "safety_rule",
        "\n".join(
            [
                "[Safety rule]",
                "Risk: " + _u("\\uc2e0\\uc7a5\\uc9c8\\ud658\\uc774\\ub098 \\uc2e0\\uc7a5 \\uae30\\ub2a5 \\uc800\\ud558\\uac00 \\uc788\\uc73c\\uba74 \\ub9c8\\uadf8\\ub124\\uc298 \\ubcf5\\uc6a9 \\uc804 \\uc758\\ub8cc\\uc9c4\\uacfc \\uc0c1\\ub2f4\\ud558\\uc138\\uc694."),
                "Recommendation: " + _u("\\uc2e0\\uc7a5\\uc9c8\\ud658\\uc774 \\uc788\\uc73c\\uba74 \\uc784\\uc758 \\ubcf5\\uc6a9\\uc744 \\ud53c\\ud558\\uc138\\uc694."),
            ]
        ),
    )
    thyroid = Doc(
        "INT_MAGNESIUM_THYROID_MEDICATION",
        "interaction_rule",
        "Recommendation: " + _u("\\uac11\\uc0c1\\uc120\\uc57d\\uc744 \\ubcf5\\uc6a9 \\uc911\\uc778 \\uacbd\\uc6b0 \\ud761\\uc218\\uc5d0 \\uc601\\ud5a5\\uc744 \\uc904 \\uc218 \\uc788\\uc73c\\ubbc0\\ub85c \\ubcf5\\uc6a9 \\uac04\\uaca9\\uc744 \\uc758\\uc0ac \\ub610\\ub294 \\uc57d\\uc0ac\\uc5d0\\uac8c \\ud655\\uc778\\ud558\\uc138\\uc694."),
    )
    antibiotics = Doc(
        "INT_MAGNESIUM_ANTIBIOTICS",
        "interaction_rule",
        "Recommendation: " + _u("\\uc77c\\ubd80 \\ud56d\\uc0dd\\uc81c\\ub294 \\ub9c8\\uadf8\\ub124\\uc298\\uacfc \\ubcf5\\uc6a9 \\uac04\\uaca9\\uc774 \\ud544\\uc694\\ud560 \\uc218 \\uc788\\uc2b5\\ub2c8\\ub2e4."),
    )

    result = compose_supplement_response(
        question=query,
        parsed_query=parse_supplement_query(query),
        generated_answer=_u("\\ub9c8\\uadf8\\ub124\\uc298\\uc740 \\uc2dd\\ud6c4\\ub098 \\uc800\\ub141 \\uc2dc\\uac04\\ub300\\uac00 \\ubb34\\ub09c\\ud569\\ub2c8\\ub2e4."),
        generated_caution=None,
        selected_docs=[profile, kidney, thyroid, antibiotics],
    )

    assert result.caution
    assert len(result.caution) <= 520
    assert _u("\\uc2e0\\uc7a5") in result.caution
    assert _u("\\uac11\\uc0c1\\uc120") in result.caution or _u("\\ud56d\\uc0dd\\uc81c") in result.caution
    assert _u("\\uac04\\uaca9") in result.caution or _u("\\uc0c1\\ub2f4") in result.caution


def test_kidney_magnesium_caution_prioritizes_high_risk_sentence():
    query = _u("\\uc2e0\\uc7a5\\uc9c8\\ud658 \\uc788\\ub294\\ub370 \\ub9c8\\uadf8\\ub124\\uc298 \\uba39\\uc5b4\\ub3c4 \\ub3fc?")
    interaction = Doc(
        "INT_MAGNESIUM_ANTIBIOTICS",
        "interaction_rule",
        "Recommendation: " + _u("\\ud56d\\uc0dd\\uc81c\\ub294 \\ubcf5\\uc6a9 \\uac04\\uaca9\\uc744 \\ud655\\uc778\\ud558\\uc138\\uc694."),
    )
    safety = Doc(
        "SAFE_KIDNEY_MAGNESIUM",
        "safety_rule",
        "\n".join(
            [
                "[Safety rule]",
                "Risk: " + _u("\\uc2e0\\uc7a5\\uc9c8\\ud658\\uc774\\ub098 \\uc2e0\\uc7a5 \\uae30\\ub2a5 \\uc800\\ud558\\uac00 \\uc788\\uc73c\\uba74 \\ub9c8\\uadf8\\ub124\\uc298 \\ubc30\\ucd9c\\uc774 \\uc5b4\\ub824\\uc6cc\\uc9c8 \\uc218 \\uc788\\uc2b5\\ub2c8\\ub2e4."),
                "Recommendation: " + _u("\\uc784\\uc758 \\ubcf5\\uc6a9\\uc744 \\ud53c\\ud558\\uace0 \\uc758\\ub8cc\\uc9c4\\uacfc \\uc0c1\\ub2f4\\ud558\\uc138\\uc694."),
            ]
        ),
    )

    result = compose_supplement_response(
        question=query,
        parsed_query=parse_supplement_query(query),
        generated_answer=_u("\\uac1c\\uc778 \\uc0c1\\ud0dc\\uc5d0 \\ub530\\ub77c \\ud655\\uc778\\uc774 \\ud544\\uc694\\ud569\\ub2c8\\ub2e4."),
        generated_caution=None,
        selected_docs=[interaction, safety],
    )

    assert result.caution
    first_sentence = result.caution.split(".")[0]
    assert _u("\\uc2e0\\uc7a5") in first_sentence
    assert _u("\\uba39\\uc5b4\\ub3c4 \\ub429\\ub2c8\\ub2e4") not in result.caution
    assert _u("\\uc548\\uc804\\ud569\\ub2c8\\ub2e4") not in result.caution


def test_compact_caution_keeps_none_when_no_caution_sources():
    query = _u("\\ud06c\\ub808\\uc544\\ud2f4\\uc740 \\uc5b8\\uc81c \\uba39\\ub294 \\uac8c \\uc88b\\uc544?")

    result = compose_supplement_response(
        question=query,
        parsed_query=parse_supplement_query(query),
        generated_answer=_u("\\ud06c\\ub808\\uc544\\ud2f4\\uc740 \\uafb8\\uc900\\ud55c \\ubcf5\\uc6a9\\uc774 \\uc911\\uc694\\ud569\\ub2c8\\ub2e4."),
        generated_caution=None,
        selected_docs=[],
    )

    assert isinstance(result.answer, str)
    assert result.caution is None

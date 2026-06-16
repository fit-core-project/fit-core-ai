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

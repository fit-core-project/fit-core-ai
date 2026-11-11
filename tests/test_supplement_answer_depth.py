from __future__ import annotations

import pytest

from engines.supplement.query_understanding import parse_supplement_query
from engines.supplement.response_composer import compose_supplement_response
from engines.supplement.supplement_engine import SupplementRAGEngine

FIXED_DISCLAIMER = "확보된 로컬 근거 문서만으로 답변합니다"
LABEL_NEEDLES = ("제품명", "성분표", "함량")


class Doc:
    def __init__(self, doc_id: str, source: str, content: str, **metadata):
        self.metadata = {"id": doc_id, "source": source, **metadata}
        self.page_content = content


ZINC_PROFILE = Doc(
    "ING_ZINC",
    "ingredient_profile",
    "\n".join(
        [
            "[Ingredient profile]",
            "ID: ING_ZINC",
            "Name: 아연",
            "Timing:",
            "  - General: 아연은 제품 지침에 따라 복용하고 공복에 속이 불편하면 식후 복용을 고려할 수 있습니다.",
            "  - With food: 공복 복용 시 메스꺼움이나 위장 불편이 있으면 식후 복용을 고려하세요.",
            "Spacing:",
            "  - 철분, 칼슘, 일부 항생제 등과 흡수 간섭 가능성이 있어 간격 확인이 필요합니다.",
            "Cautions:",
            "  - 고용량 장기 복용 시 구리 결핍이나 위장 불편 가능성이 있습니다.",
            "  - 공복 복용 시 위장 불편이 생길 수 있습니다.",
        ]
    ),
    canonical="zinc",
)
ZINC_IRON_CALCIUM = Doc(
    "INT_ZINC_IRON_CALCIUM",
    "interaction_rule",
    "[Interaction rule]\nEntity A: 아연 / zinc / supplement_ingredient\nEntity B: 철분 / iron / supplement_ingredient\n"
    "Recommendation: 아연과 철분 또는 칼슘은 흡수 경쟁이 생길 수 있어 간격 확인을 고려하세요.",
    entity_a_canonical="zinc",
    entity_b_canonical="iron",
)
PROTEIN_PROFILE = Doc(
    "ING_PROTEIN",
    "ingredient_profile",
    "\n".join(
        [
            "[Ingredient profile]",
            "Name: 단백질 보충제",
            "Timing:",
            "  - General: 단백질 보충제는 식단에서 부족한 단백질을 보완하는 용도로 총 섭취량을 함께 고려하세요.",
            "  - Exercise related: 운동 전후보다 하루 총 단백질 섭취량과 식단 균형을 우선 확인하세요.",
            "Cautions:",
            "  - 신장질환이 있으면 단백질 보충제 총량을 의료진과 확인하세요.",
        ]
    ),
    canonical="protein",
)
KIDNEY_PROTEIN = Doc(
    "SAFE_KIDNEY_PROTEIN",
    "safety_rule",
    "[Safety rule]\nTrigger conditions:\n  - 신장질환\nAffected entities:\n  - protein\n"
    "Risk: 신장질환이 있으면 단백질 보충제 총량 확인이 필요합니다.\nRecommendation: 의료진과 상담하세요.",
    affected_entities="protein, 프로틴",
)
MULTIVITAMIN_PROFILE = Doc(
    "ING_MULTIVITAMIN",
    "ingredient_profile",
    "\n".join(
        [
            "[Ingredient profile]",
            "Name: 종합비타민",
            "Timing:",
            "  - General: 종합비타민은 제품 지침에 따라 식사와 함께 복용하는 방식이 일반적으로 무난합니다.",
            "Spacing:",
            "  - 철분, 아연, 칼슘, 비타민A, 비타민D 등 고함량 성분이 다른 제품과 중복되는지 확인하세요.",
            "Cautions:",
            "  - 여러 비타민과 미네랄이 함께 들어 있어 다른 보충제와 중복 섭취를 확인해야 합니다.",
        ]
    ),
    canonical="multivitamin",
)
PREGNANCY_MULTIVITAMIN = Doc(
    "SAFE_PREGNANCY_MULTIVITAMIN",
    "safety_rule",
    "[Safety rule]\nTrigger conditions:\n  - 임신\n  - 임산부\nAffected entities:\n  - multivitamin\n"
    "Risk: 임신 중에는 비타민A 등 고함량 성분 주의가 필요합니다.\nRecommendation: 임산부용 제품인지 확인하고 상담하세요.",
    affected_entities="multivitamin, 종합비타민",
)
MAGNESIUM_PROFILE = Doc(
    "ING_MAGNESIUM",
    "ingredient_profile",
    "[Ingredient profile]\nName: 마그네슘\nTiming:\n  - General: 마그네슘은 식후나 저녁 시간대 복용이 일반적으로 무난합니다.",
    canonical="magnesium",
)
PROBIOTIC_PROFILE = Doc(
    "ING_PROBIOTIC",
    "ingredient_profile",
    "[Ingredient profile]\nName: 유산균\nSpacing:\n  - 항생제 복용 중이면 동시에 복용하지 말고 간격을 확인하세요.",
    canonical="probiotic",
)


def _compose(question: str, docs: list[Doc], answer: str = "WEB_SEARCH_REQUIRED"):
    return compose_supplement_response(
        question=question,
        parsed_query=parse_supplement_query(question),
        generated_answer=answer,
        generated_caution=None,
        selected_docs=docs,
    )


# --- 2. entity boundary matching -------------------------------------------------


@pytest.mark.parametrize(
    "question, forbidden",
    [
        ("철분 500mg 먹어도 돼?", "magnesium"),
        ("철분 30 mg 공복에 먹어도 돼?", "magnesium"),
        ("Caffeine 들어간 부스터 괜찮아?", "calcium"),
        ("coffee 마시고 아연 먹어도 돼?", "iron"),
        ("수술 전에 오메가3 먹어도 돼?", "alcohol use"),
        ("수술 전에 오메가3 먹어도 돼?", "alcohol"),
        ("웨이트 하고 크레아틴 먹어도 돼?", "protein"),
    ],
)
def test_short_aliases_do_not_match_inside_other_tokens(question, forbidden):
    canonicals = {entity.canonical for entity in parse_supplement_query(question).entities}
    assert forbidden not in canonicals


@pytest.mark.parametrize(
    "question, expected",
    [
        ("Mg랑 Zn 같이 먹어도 돼?", {"magnesium", "zinc"}),
        ("Fe 보충제 Ca랑 같이 먹어도 돼?", {"iron", "calcium"}),
        ("술 마신 다음날 아연 먹어도 돼?", {"alcohol use", "zinc"}),
        ("어제 술 마셨는데 타이레놀 괜찮아?", {"alcohol use", "acetaminophen"}),
        ("vitamin d3 언제 먹어?", {"vitamin d"}),
    ],
)
def test_short_aliases_still_match_on_token_boundaries(question, expected):
    canonicals = {entity.canonical for entity in parse_supplement_query(question).entities}
    assert expected <= canonicals


def test_surgery_still_detected_and_not_alcohol():
    parsed = parse_supplement_query("수술 앞두고 오메가3 먹어도 돼?")
    canonicals = {entity.canonical for entity in parsed.entities}
    assert "surgery" in canonicals
    assert not canonicals & {"alcohol", "alcohol use"}


# --- 1. WEB_SEARCH_REQUIRED -> doc-grounded answer ------------------------------------


class _FakePrompt:
    def __init__(self, answer):
        self.answer = answer

    def __or__(self, _other):
        return self

    def invoke(self, _payload):
        return self.answer


def _engine(answer: str, docs: list[Doc]) -> SupplementRAGEngine:
    engine = SupplementRAGEngine.degraded("test")
    engine.ready = True
    engine.device = "cpu"
    engine.db_dir = "test-db"
    engine.embeddings = type("E", (), {"embed_query": lambda self, q: [0.1]})()
    engine.vector_store = type("V", (), {"similarity_search_by_vector": lambda self, e, k: docs[:k]})()
    engine._tokenize_kiwi = lambda _text: []
    engine.original_docs = docs
    engine.reranker = type("R", (), {"predict": lambda self, pairs: [1.0 for _ in pairs]})()
    engine.local_prompt = _FakePrompt(answer)
    engine.llm = object()
    engine.StrOutputParser = lambda: object()
    engine.web_search = type("W", (), {"run": lambda self, q: (_ for _ in ()).throw(AssertionError("no web"))})()
    return engine


def test_engine_web_search_required_builds_answer_from_kb_docs(monkeypatch):
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("SUPPLEMENT_ENABLE_WEB_FALLBACK", "false")
    engine = _engine("WEB_SEARCH_REQUIRED", [ZINC_PROFILE, ZINC_IRON_CALCIUM])

    result = engine._answer_full("아연은 공복에 먹어도 돼?")

    assert FIXED_DISCLAIMER not in result["answer"]
    assert "WEB_SEARCH_REQUIRED" not in result["answer"]
    assert "식후" in result["answer"]
    assert "위장" in result["answer"]
    # The zinc+iron interaction answer must not fire when iron/calcium were not asked about.
    assert "흡수 경쟁" not in result["answer"]
    assert result["debugCounts"]["answerRecoveredFromKbDocs"] is True
    assert result["debugCounts"]["llmAnswerFallbackReason"] == "web_search_required"


def test_fixed_disclaimer_alone_is_treated_as_weak_answer():
    result = _compose(
        "아연은 공복에 먹어도 돼?",
        [ZINC_PROFILE],
        answer=f"{FIXED_DISCLAIMER}. 개인의 질환, 복용 중인 약, 증상에 따라 달라질 수 있으므로 보충제 복용 전에는 의사 또는 약사와 상담하세요.",
    )
    assert result.used_kb_fallback is True
    assert "식후" in result.answer


def test_protein_pre_workout_uses_exercise_timing_not_kidney_template():
    result = _compose("프로틴은 운동 전에 먹어도 돼?", [PROTEIN_PROFILE, KIDNEY_PROTEIN])
    assert "총 단백질" in result.answer
    assert not result.answer.startswith("신장질환이 있다면")


def test_multivitamin_daily_mentions_overlap_not_pregnancy_template():
    result = _compose("종합비타민 매일 먹어도 돼?", [PREGNANCY_MULTIVITAMIN, MULTIVITAMIN_PROFILE])
    assert "중복" in result.answer
    assert "임산부의" not in result.answer
    assert result.caution


def test_pregnancy_multivitamin_still_uses_pregnancy_answer_with_caution():
    result = _compose("임산부인데 종합비타민 먹어도 돼?", [PREGNANCY_MULTIVITAMIN, MULTIVITAMIN_PROFILE])
    assert "임산부" in result.answer
    assert "비타민A" in result.answer
    assert result.caution
    assert "복용해도 됩니다" not in f"{result.answer} {result.caution}"


def test_multi_supplement_web_search_required_splits_components():
    result = _compose("유산균이랑 마그네슘 같이 먹어도 돼?", [MAGNESIUM_PROFILE, PROBIOTIC_PROFILE])
    assert "유산균" in result.answer
    assert "마그네슘" in result.answer
    assert "성분별" in result.answer
    assert "단정" in result.answer
    assert "확인된 성분은" not in result.answer
    assert "수술 예정 여부" not in result.answer


def test_unknown_ingredient_asks_for_label_and_ignores_unrelated_docs():
    result = _compose("NMN 먹어도 돼?", [MAGNESIUM_PROFILE, ZINC_PROFILE])
    for needle in LABEL_NEEDLES:
        assert needle in result.answer
    assert "마그네슘" not in result.answer
    assert "아연" not in result.answer
    assert FIXED_DISCLAIMER not in result.answer


def test_partial_known_answers_known_part_and_requests_label_for_unknown():
    result = _compose("NAC랑 아연 같이 먹어도 돼?", [ZINC_PROFILE, ZINC_IRON_CALCIUM])
    assert "아연" in result.answer
    assert "흡수 간섭" in result.answer or "식후" in result.answer
    for needle in LABEL_NEEDLES:
        assert needle in result.answer


def test_goal_context_without_entities_is_not_treated_as_unknown():
    result = _compose("운동 전 보충제로 뭐가 좋아?", [MAGNESIUM_PROFILE])
    assert "추천" in result.answer


# --- 3. no mid-sentence splicing, consult dedupe ---------------------------------------


def test_soften_rewrites_recommendation_without_splicing():
    result = _compose(
        "와파린 먹는데 오메가3 먹어도 돼?",
        [],
        answer="와파린을 복용 중이면 오메가3는 의료진에게 먼저 확인하시는 것이 안전합니다.",
    )
    assert "확인하시는 것이 좋습니다" in result.answer
    assert "것이 개인 상태에 따라 달라질 수 있습니다" not in result.answer
    assert "안전합니다" not in result.answer


def test_soften_drops_unsafe_sentence_instead_of_splicing():
    result = _compose(
        "와파린 먹는데 오메가3 먹어도 돼?",
        [],
        answer="오메가3는 출혈 위험을 먼저 확인해야 합니다. 같이 먹어도 됩니다.",
    )
    assert "먹어도 됩니다" not in result.answer
    assert "개인 상태에 따라 달라질 수 있습니다" not in result.answer.split(".")[0]
    assert result.answer.startswith("오메가3는 출혈 위험을 먼저 확인해야 합니다.")
    assert "단정하기는 어렵습니다" in result.answer
    assert result.caution


def test_kb_recommendation_phrasing_is_softened_in_grounded_answer():
    doc = Doc(
        "ING_IRON",
        "ingredient_profile",
        "[Ingredient profile]\nTiming:\n  - General: 철분은 커피와 2~4시간 간격을 두는 방식이 안전합니다.",
        canonical="iron",
    )
    result = _compose("철분은 공복에 먹어도 돼?", [doc])
    assert "간격을 두는 방식이 좋습니다" in result.answer
    assert "안전합니다" not in result.answer


def test_consult_sentence_is_not_repeated_across_answer_and_caution():
    consult = "보충제 복용 전에는 의사 또는 약사와 상담하세요."
    result = _compose(
        "임산부인데 종합비타민 먹어도 돼?",
        [MULTIVITAMIN_PROFILE],
        answer=f"종합비타민은 임산부용 제품인지 확인하는 것이 중요합니다. {consult} {consult}",
    )
    combined = f"{result.answer} {result.caution or ''}"
    assert combined.count("의사 또는 약사와 상담하세요") <= 1
    assert result.caution
    assert any(word in result.caution for word in ("상담", "의사", "약사", "의료진"))
    assert result.answer.startswith("종합비타민은 임산부용 제품인지")


def test_caution_doc_relevance_uses_whole_tokens():
    from types import SimpleNamespace
    from engines.supplement.response_composer import _doc_is_relevant_for_caution
    doc = SimpleNamespace(metadata={"source": "interaction_rule", "id": "INT_IRON_CAFFEINE"}, page_content="")
    no_entities = SimpleNamespace(entities=())
    # "int" (rule prefix) and "iron" inside "environment" used to count as a match.
    assert not _doc_is_relevant_for_caution(doc, no_entities, "environment point mint")
    assert _doc_is_relevant_for_caution(doc, no_entities, "iron과 caffeine은 간격을 두세요")


def test_curated_kb_replaces_stale_index_copies():
    from pathlib import Path
    from types import SimpleNamespace
    from engines.supplement.kb_documents import merge_curated_kb_documents

    class Doc(SimpleNamespace):
        pass

    stale = [
        Doc(page_content="old magnesium", metadata={"source": "ingredient_profile", "id": "ING_MAGNESIUM"}),
        Doc(page_content="drug card", metadata={"source": "drugs", "id": "D1"}),
    ]
    merged = merge_curated_kb_documents(stale, Path("data/documents"), Doc)
    ids = [d.metadata["id"] for d in merged if d.metadata["source"] == "ingredient_profile"]
    assert "ING_ZINC" in ids and "ING_PROTEIN" in ids  # missing from the 2026-11 index
    assert ids.count("ING_MAGNESIUM") == 1
    assert not any(d.page_content == "old magnesium" for d in merged)
    assert any(d.metadata["id"] == "D1" for d in merged)


def _curated_docs():
    from pathlib import Path
    from types import SimpleNamespace
    from engines.supplement.kb_documents import load_curated_kb_records
    return [SimpleNamespace(page_content=t, metadata=m) for t, m, _ in load_curated_kb_records(Path("data/documents"))]


def test_efficacy_question_without_kb_evidence_says_so_first():
    from engines.supplement.query_understanding import parse_supplement_query
    from engines.supplement.response_composer import _apply_efficacy_policy
    q = "마그네슘 수면에 도움 돼요?"
    out = _apply_efficacy_policy("마그네슘은 숙면을 돕습니다.", q, parse_supplement_query(q), _curated_docs())
    assert out.startswith("확보된 근거 문서에서는 마그네슘의 그런 효과를 확인할 수 없습니다.")


def test_efficacy_question_with_documented_use_names_the_use_only():
    from engines.supplement.query_understanding import parse_supplement_query
    from engines.supplement.response_composer import _apply_efficacy_policy
    q = "비타민D 뼈 건강에 도움 되나요?"
    out = _apply_efficacy_policy("A", q, parse_supplement_query(q), _curated_docs())
    assert "'뼈 건강 관련 영양 보충' 용도로 안내" in out and "직접 확인한 근거는 없습니다" in out


def test_non_efficacy_question_is_untouched():
    from engines.supplement.query_understanding import parse_supplement_query
    from engines.supplement.response_composer import _apply_efficacy_policy
    q = "마그네슘 언제 먹어?"
    assert _apply_efficacy_policy("A", q, parse_supplement_query(q), _curated_docs()) == "A"

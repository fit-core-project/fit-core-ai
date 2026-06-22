from __future__ import annotations

from engines.supplement.query_understanding import parse_supplement_query
from engines.supplement.response_composer import compose_supplement_response
from engines.supplement.supplement_engine import SupplementRAGEngine


class Doc:
    def __init__(self, doc_id: str, source: str, content: str, **metadata):
        self.metadata = {"id": doc_id, "source": source, **metadata}
        self.page_content = content


class FakePrompt:
    def __init__(self, answer: str | Exception):
        self.answer = answer

    def __or__(self, _other):
        return self

    def invoke(self, _payload):
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


class FakeEmbeddings:
    def embed_query(self, _query):
        return [0.1, 0.2]


class FakeVectorStore:
    def __init__(self, docs):
        self.docs = docs

    def similarity_search_by_vector(self, _embedding, k):
        return self.docs[:k]


class FakeReranker:
    def predict(self, pairs):
        return [1.0 for _ in pairs]


def _engine_with_llm(answer: str | Exception, docs: list[Doc]) -> SupplementRAGEngine:
    engine = SupplementRAGEngine.degraded("test")
    engine.ready = True
    engine.device = "cpu"
    engine.db_dir = "test-db"
    engine.embeddings = FakeEmbeddings()
    engine.vector_store = FakeVectorStore(docs)
    engine._tokenize_kiwi = lambda _text: []
    engine.original_docs = docs
    engine.reranker = FakeReranker()
    engine.local_prompt = FakePrompt(answer)
    engine.llm = object()
    engine.StrOutputParser = lambda: object()
    return engine


def test_goal_based_recommendation_keeps_valid_llm_answer(monkeypatch):
    monkeypatch.setenv("APP_ENV", "local")
    docs = [
        Doc("ING_OMEGA3", "ingredient_profile", "Name: omega3\nCautions:\n- anticoagulant surgery caution", canonical="omega3"),
    ]
    engine = _engine_with_llm(
        "근육량 증가가 목표라면 우선순위는 식사 단백질을 채운 뒤 프로틴, 그다음 크레아틴을 보는 쪽입니다. EAA나 글루타민은 식사가 부족하거나 회복 이슈가 있을 때 조건부로 검토하세요.",
        docs,
    )

    result = engine._answer_full("근육량 증가 목표인데 영양제 및 보충제 추천해줘")

    assert result["mode"] == "full"
    assert result["answer"].startswith("근육량 증가가 목표라면")
    assert "확인된 성분은" not in result["answer"]
    assert "오메가3" not in (result.get("caution") or "")
    assert result["debugCounts"]["llmAnswerEnabled"] is True
    assert result["debugCounts"].get("answerRecoveredFromKbDocs") is not True


def test_multi_supplement_valid_llm_answer_is_not_replaced():
    result = compose_supplement_response(
        question="프로틴이랑 크레아틴 같이 먹어도 돼?",
        parsed_query=parse_supplement_query("프로틴이랑 크레아틴 같이 먹어도 돼?"),
        generated_answer="둘은 목적이 달라 함께 고려할 수 있지만, 프로틴은 하루 단백질 총량을 채우는 용도이고 크레아틴은 운동 수행 보조 목적이라 제품 권장량과 신장 상태를 확인하세요.",
        generated_caution=None,
        selected_docs=[],
    )

    assert result.answer.startswith("둘은 목적이 달라")
    assert "확인된 성분은" not in result.answer
    assert result.used_kb_fallback is False


def test_high_risk_keeps_llm_answer_and_adds_deterministic_caution():
    doc = Doc(
        "INT_OMEGA3_ANTICOAGULANTS",
        "interaction_rule",
        "Recommendation: 와파린 복용 중이면 오메가3 보충제를 임의로 시작하지 말고 의료진과 상담하세요.",
        entity_a_canonical="omega3",
        entity_b_canonical="warfarin",
    )

    result = compose_supplement_response(
        question="와파린 먹는데 오메가3 먹어도 돼?",
        parsed_query=parse_supplement_query("와파린 먹는데 오메가3 먹어도 돼?"),
        generated_answer="와파린을 복용 중이면 오메가3는 출혈 위험 관점에서 먼저 확인이 필요합니다.",
        generated_caution=None,
        selected_docs=[doc],
    )

    assert result.answer.startswith("와파린을 복용 중이면")
    assert result.caution
    assert "상담" in result.caution or "의료진" in result.caution or "의사" in result.caution or "약사" in result.caution


def test_unknown_partial_known_keeps_llm_answer_but_enforces_label_request():
    result = compose_supplement_response(
        question="NMN이랑 마그네슘 같이 먹어도 돼?",
        parsed_query=parse_supplement_query("NMN이랑 마그네슘 같이 먹어도 돼?"),
        generated_answer="마그네슘은 신장 기능이나 복용 중인 약에 따라 주의가 달라질 수 있습니다.",
        generated_caution=None,
        selected_docs=[Doc("ING_MAGNESIUM", "ingredient_profile", "Name: magnesium", canonical="magnesium")],
    )

    assert result.answer.startswith("마그네슘은")
    assert "제품명" in result.answer
    assert "성분표" in result.answer
    assert "함량" in result.answer


def test_llm_timeout_falls_back_to_deterministic_answer(monkeypatch):
    monkeypatch.setenv("APP_ENV", "local")
    docs = [Doc("ING_MAGNESIUM", "ingredient_profile", "Name: magnesium\nTiming:\n- with food", canonical="magnesium")]
    engine = _engine_with_llm(TimeoutError("llm timed out"), docs)

    result = engine._answer_full("마그네슘은 언제 먹는 게 좋아?")

    assert result["mode"] == "full"
    assert result["answer"]
    assert result["debugCounts"]["llmAnswerFallbackReason"] == "generation_error"
    assert result["debugCounts"]["answerRecoveredFromKbDocs"] is True

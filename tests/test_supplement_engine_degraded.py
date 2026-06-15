from engines.supplement.supplement_engine import SupplementRAGEngine, _is_unusable_generated_answer, _normalize_answer_payload


def test_supplement_ready_engine_uses_full_path(monkeypatch):
    engine = SupplementRAGEngine.degraded("test")
    engine.ready = True

    def answer_full(question):
        return {"answer": "full answer", "sources": [{"id": "doc-1"}], "mode": "full"}

    monkeypatch.setattr(engine, "_answer_full", answer_full)

    result = engine.answer_question("마그네슘")

    assert result["mode"] == "full"
    assert result["sources"] == [{"id": "doc-1"}]


def test_supplement_missing_db_degraded_answer():
    engine = SupplementRAGEngine(db_path="./missing-demo-db")

    result = engine.answer_question("마그네슘 먹어도 돼?")

    assert engine.ready is False
    assert result["answer"]
    assert result["sources"] == []
    assert result["mode"] == "degraded"
    assert "의사" in result["answer"] or "약사" in result["answer"]


def test_supplement_answer_question_never_raises_in_degraded_mode():
    engine = SupplementRAGEngine(db_path="./missing-demo-db")

    result = engine.answer_question("")

    assert result["answer"]
    assert isinstance(result["sources"], list)


def test_supplement_runtime_exception_falls_back(monkeypatch):
    engine = SupplementRAGEngine(db_path="./missing-demo-db")
    engine.ready = True

    def fail(question):
        raise RuntimeError("secret supplement stack")

    monkeypatch.setattr(engine, "_answer_full", fail)

    result = engine.answer_question("비타민D")

    assert result["mode"] == "degraded"
    assert result["degraded_reason"] == "runtime_error"


def test_supplement_query_optimizer_disabled_uses_deterministic_query(monkeypatch):
    engine = SupplementRAGEngine.degraded("test")

    class QueryOptimizerShouldNotRun:
        def invoke(self, prompt):
            raise AssertionError("query optimizer should not run")

    monkeypatch.setenv("SUPPLEMENT_ENABLE_LLM_QUERY_OPTIMIZER", "false")
    engine.query_optimizer = QueryOptimizerShouldNotRun()

    query = engine._expand_query("크레아틴 먹어도 되나요?")

    assert "크레아틴" in query
    assert "creatine" in query
    assert "운동 후" in query


def test_supplement_timing_queries_expand_to_retrieval_keywords(monkeypatch):
    engine = SupplementRAGEngine.degraded("test")
    monkeypatch.setenv("SUPPLEMENT_ENABLE_LLM_QUERY_OPTIMIZER", "false")

    magnesium_query = engine._expand_query("마그네슘은 언제 먹는 게 좋아?")
    iron_query = engine._expand_query("철분은 커피랑 같이 먹어도 돼?")

    assert "magnesium" in magnesium_query
    assert "복용 타이밍" in magnesium_query
    assert "식후" in magnesium_query
    assert "iron" in iron_query
    assert "카페인" in iron_query
    assert "흡수 방해" in iron_query
    assert "복용 간격" in iron_query


def test_supplement_existing_pass_queries_keep_expansion_keywords(monkeypatch):
    engine = SupplementRAGEngine.degraded("test")
    monkeypatch.setenv("SUPPLEMENT_ENABLE_LLM_QUERY_OPTIMIZER", "false")

    creatine_query = engine._expand_query("크레아틴은 언제 먹는 게 좋아?")
    vitamin_d_query = engine._expand_query("비타민D는 식후에 먹는 게 좋아?")
    omega3_query = engine._expand_query("오메가3 복용 시 주의사항 알려줘")

    assert "creatine" in creatine_query
    assert "운동 후" in creatine_query
    assert "vitamin D" in vitamin_d_query
    assert "복용 타이밍" in vitamin_d_query
    assert "omega-3" in omega3_query
    assert "EPA" in omega3_query


def test_supplement_priority_timing_docs_match_query_targets():
    engine = SupplementRAGEngine.degraded("test")

    class Doc:
        def __init__(self, doc_id):
            self.metadata = {"_source_file": "supplement_timing_guide.json", "id": doc_id}
            self.page_content = doc_id

    engine.original_docs = [
        Doc("SUPP_TIMING_MAGNESIUM"),
        Doc("SUPP_TIMING_IRON"),
        Doc("SUPP_TIMING_CREATINE"),
        Doc("SUPP_TIMING_VITAMIN_D"),
        Doc("SUPP_TIMING_OMEGA3"),
    ]

    assert [doc.metadata["id"] for doc in engine._priority_timing_docs("마그네슘은 언제 먹는 게 좋아?")] == [
        "SUPP_TIMING_MAGNESIUM"
    ]
    assert [doc.metadata["id"] for doc in engine._priority_timing_docs("철분은 커피랑 같이 먹어도 돼?")] == [
        "SUPP_TIMING_IRON"
    ]
    assert [doc.metadata["id"] for doc in engine._priority_timing_docs("크레아틴은 언제 먹는 게 좋아?")] == [
        "SUPP_TIMING_CREATINE"
    ]
    assert [doc.metadata["id"] for doc in engine._priority_timing_docs("비타민D는 식후에 먹는 게 좋아?")] == [
        "SUPP_TIMING_VITAMIN_D"
    ]
    assert [doc.metadata["id"] for doc in engine._priority_timing_docs("오메가3 복용 시 주의사항 알려줘")] == [
        "SUPP_TIMING_OMEGA3"
    ]


def test_supplement_unusable_generated_answer_detection():
    assert _is_unusable_generated_answer("{}") is True
    assert _is_unusable_generated_answer('{"cautions":["ask a clinician"]}') is True
    assert _is_unusable_generated_answer('{"answer":"Take with food.","caution":"Ask a clinician."}') is False
    assert _is_unusable_generated_answer("Take with food.") is False


def test_supplement_timing_doc_can_recover_plain_answer_and_caution():
    engine = SupplementRAGEngine.degraded("test")

    class Doc:
        page_content = "\n".join(
            [
                "[Supplement timing guide]",
                "Name: 철분 (Iron)",
                "Timing: 철분은 공복에 흡수가 유리할 수 있지만 속 불편이 있으면 음식과 함께 복용할 수 있습니다.",
                "Spacing: 커피, 차, 카페인, 칼슘과는 흡수 간섭 가능성이 있어 간격을 두는 것이 좋습니다.",
                "Cautions:",
                "  - 철분은 결핍 확인 없이 고용량으로 오래 복용하지 마세요.",
                "  - 임신 중이거나 빈혈 치료 중이면 전문가 지시에 따르세요.",
            ]
        )

    result = engine._answer_from_timing_doc(Doc())

    assert result == {
        "answer": "철분은 공복에 흡수가 유리할 수 있지만 속 불편이 있으면 음식과 함께 복용할 수 있습니다. 커피, 차, 카페인, 칼슘과는 흡수 간섭 가능성이 있어 간격을 두는 것이 좋습니다.",
        "caution": "철분은 결핍 확인 없이 고용량으로 오래 복용하지 마세요. 임신 중이거나 빈혈 치료 중이면 전문가 지시에 따르세요.",
    }


def test_supplement_web_fallback_disabled_keeps_local_sources(monkeypatch):
    from langchain_core.documents import Document

    engine = SupplementRAGEngine.degraded("test")
    engine.ready = True
    engine.device = "cpu"
    engine.db_dir = "local-db"
    engine.np = None
    engine.embeddings = type("Embeddings", (), {"embed_query": lambda self, query: [0.1, 0.2]})()
    engine.vector_store = type(
        "VectorStore",
        (),
        {
            "similarity_search_by_vector": lambda self, embedding, k: [
                Document(page_content="creatine local doc", metadata={"_source_file": "local.json", "id": "doc-1", "source": "supp"})
            ]
        },
    )()
    engine._tokenize_kiwi = lambda text: []
    engine.reranker = type("Reranker", (), {"predict": lambda self, pairs: [0.9]})()

    class LocalPrompt:
        def __or__(self, other):
            return self

        def invoke(self, payload):
            return "WEB_SEARCH_REQUIRED"

    class WebSearchShouldNotRun:
        def run(self, query):
            raise AssertionError("web search should not run")

    engine.local_prompt = LocalPrompt()
    engine.llm = object()
    engine.StrOutputParser = lambda: object()
    engine.web_search = WebSearchShouldNotRun()
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("SUPPLEMENT_ENABLE_WEB_FALLBACK", "false")
    monkeypatch.setenv("SUPPLEMENT_ENABLE_LLM_QUERY_OPTIMIZER", "false")

    result = engine._answer_full("크레아틴 먹어도 되나요?")

    assert result["mode"] == "full"
    assert result["sources"] == [{"file": "local.json", "id": "doc-1", "type": "supp"}]
    assert result["debugCounts"]["webSearchUsed"] is False
    assert result["debugCounts"]["webSearchEnabled"] is False


def test_old_supplement_import_shim_still_works():
    from engines.supplement_engine import SupplementRAGEngine as ShimSupplementRAGEngine

    assert ShimSupplementRAGEngine is SupplementRAGEngine


def test_supplement_json_string_answer_is_promoted_to_top_level_fields():
    payload = {
        "answer": '{"answer":"Take magnesium with food or in the evening.","caution":"Ask a clinician if you have kidney disease."}',
        "sources": [{"id": "doc-1"}],
        "mode": "full",
    }

    result = _normalize_answer_payload(payload)

    assert result["answer"] == "Take magnesium with food or in the evening."
    assert result["caution"] == "Ask a clinician if you have kidney disease."
    assert result["sources"] == [{"id": "doc-1"}]
    assert result["mode"] == "full"


def test_supplement_plain_text_answer_is_unchanged():
    payload = {"answer": "Plain text answer.", "sources": [], "mode": "full"}

    result = _normalize_answer_payload(payload)

    assert result == payload


def test_supplement_invalid_json_answer_is_kept_as_text():
    payload = {"answer": '{"answer": "missing end"', "sources": [], "mode": "full"}

    result = _normalize_answer_payload(payload)

    assert result == payload


def test_supplement_degraded_response_shape_is_unchanged():
    engine = SupplementRAGEngine.degraded("test")
    payload = engine.answer_question("magnesium")

    result = _normalize_answer_payload(payload)

    assert result == payload
    assert result["mode"] == "degraded"
    assert "caution" not in result

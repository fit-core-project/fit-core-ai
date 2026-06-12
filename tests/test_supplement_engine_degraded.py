from engines.supplement.supplement_engine import SupplementRAGEngine


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

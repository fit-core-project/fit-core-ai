from engines.supplement.query_understanding import parse_supplement_query
from engines.supplement.response_composer import compose_supplement_response
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


def test_supplement_timing_queries_prioritize_entity_profile_sources():
    engine = SupplementRAGEngine.degraded("test")

    class Doc:
        def __init__(self, doc_id, canonical):
            self.metadata = {"id": doc_id, "source": "ingredient_profile", "canonical": canonical}
            self.page_content = doc_id

    engine.original_docs = [
        Doc("ING_MAGNESIUM", "magnesium"),
        Doc("ING_CREATINE", "creatine"),
        Doc("ING_VITAMIN_D", "vitamin d"),
    ]

    cases = [
        ("마그네슘은 언제 먹는 게 좋아?", "magnesium", "ING_MAGNESIUM"),
        ("크레아틴은 언제 먹는 게 좋아?", "creatine", "ING_CREATINE"),
        ("비타민D는 식후에 먹는 게 좋아?", "vitamin d", "ING_VITAMIN_D"),
    ]

    for query, expected_entity, expected_id in cases:
        parsed = parse_supplement_query(query)
        docs = engine._priority_kb_docs(parsed)

        assert expected_entity in {entity.canonical for entity in parsed.entities}
        assert docs
        assert docs[0].metadata["source"] == "ingredient_profile"
        assert docs[0].metadata["canonical"] == expected_entity
        assert docs[0].metadata["id"] == expected_id


def test_supplement_timing_source_filter_keeps_only_entity_matched_timing_docs():
    engine = SupplementRAGEngine.degraded("test")
    parsed = parse_supplement_query("magnesium 언제 먹는 게 좋아?")

    class Doc:
        def __init__(self, doc_id, page_content):
            self.metadata = {"id": doc_id, "source": "supp_timing", "_source_file": "supplement_timing_guide.json"}
            self.page_content = page_content

    matched = Doc("SUPP_TIMING_MAGNESIUM", "Name: Magnesium\nTiming: take with food or in the evening.")
    unrelated = Doc("SUPP_TIMING_CREATINE", "Name: Creatine\nTiming: take consistently.")

    assert engine._is_unrelated_supp_timing_doc(parsed, matched) is False
    assert engine._is_unrelated_supp_timing_doc(parsed, unrelated) is True


def test_supplement_unusable_generated_answer_detection():
    assert _is_unusable_generated_answer("{}") is True
    assert _is_unusable_generated_answer('{"cautions":["ask a clinician"]}') is True
    assert _is_unusable_generated_answer('{"answer":"Take with food.","caution":"Ask a clinician."}') is False
    assert _is_unusable_generated_answer("Take with food.") is False


def test_supplement_composer_replaces_timing_doc_answer_recovery():
    class Doc:
        metadata = {"id": "ING_IRON", "source": "ingredient_profile", "canonical": "iron"}
        page_content = "\n".join(
            [
                "[Ingredient profile]",
                "ID: ING_IRON",
                "Name: 철분 (Iron)",
                "Timing:",
                "  - General: 철분은 공복에 흡수가 유리할 수 있지만 속 불편이 있으면 음식과 함께 복용할 수 있습니다.",
                "Spacing:",
                "  - 커피, 차, 카페인, 칼슘과는 흡수 간섭 가능성이 있어 간격을 두는 것이 좋습니다.",
                "Cautions:",
                "  - 철분은 결핍 확인 없이 고용량으로 오래 복용하지 마세요.",
                "  - 임신 중이거나 빈혈 치료 중이면 전문가 지시에 따르세요.",
            ]
        )

    result = compose_supplement_response(
        question="철분은 언제 먹는 게 좋아?",
        parsed_query=parse_supplement_query("철분은 언제 먹는 게 좋아?"),
        generated_answer="{}",
        generated_caution=None,
        selected_docs=[Doc()],
    )

    assert result.used_kb_fallback is True
    assert isinstance(result.answer, str)
    assert result.answer
    assert not result.answer.strip().startswith("{")
    assert result.caution
    assert "철분" in result.caution


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


def test_supplement_safety_check_json_answer_is_promoted_to_plain_text():
    payload = {
        "answer": '{"safety_check":{"summary":"Acetaminophen with alcohol needs caution.","recommendation":"Ask a pharmacist or physician before additional use.","risk":"Liver risk may be higher."}}',
        "sources": [{"id": "SAFE_ACETAMINOPHEN_LIVER_ALCOHOL"}],
        "mode": "full",
    }

    result = _normalize_answer_payload(payload)

    assert result["answer"] == "Acetaminophen with alcohol needs caution. Ask a pharmacist or physician before additional use."
    assert result["caution"] == "Liver risk may be higher."
    assert result["sources"] == [{"id": "SAFE_ACETAMINOPHEN_LIVER_ALCOHOL"}]


def test_supplement_unexpected_json_object_answer_is_flattened_to_plain_text():
    payload = {
        "answer": '{"Safety rule says":"Ask a clinician before using acetaminophen with alcohol."}',
        "sources": [{"id": "SAFE_ACETAMINOPHEN_LIVER_ALCOHOL"}],
        "mode": "full",
    }

    result = _normalize_answer_payload(payload)

    assert result["answer"] == "Safety rule says: Ask a clinician before using acetaminophen with alcohol."
    assert result["caution"] == "복용 중인 약, 질환, 음주 등 위험 요인이 있으면 의사 또는 약사와 상담하세요."


def test_supplement_degraded_response_shape_is_unchanged():
    engine = SupplementRAGEngine.degraded("test")
    payload = engine.answer_question("magnesium")

    result = _normalize_answer_payload(payload)

    assert result == payload
    assert result["mode"] == "degraded"
    assert "caution" not in result


def test_supplement_priority_kb_docs_route_risk_queries_by_entity_and_intent():
    engine = SupplementRAGEngine.degraded("test")

    class Doc:
        def __init__(self, doc_id, source, page_content="", **metadata):
            self.metadata = {"id": doc_id, "source": source, **metadata}
            self.page_content = page_content

    engine.original_docs = [
        Doc(
            "INT_OMEGA3_ANTICOAGULANTS",
            "interaction_rule",
            entity_a_canonical="omega3",
            entity_b_canonical="anticoagulant",
            interaction_type="bleeding_risk",
            caution_level="high",
        ),
        Doc(
            "SAFE_KIDNEY_MAGNESIUM",
            "safety_rule",
            "Trigger conditions: 신장질환 kidney disease\nAffected entities: magnesium 마그네슘",
            affected_entities="magnesium, 마그네슘",
            caution_level="high",
        ),
        Doc(
            "SAFE_ACETAMINOPHEN_LIVER_ALCOHOL",
            "safety_rule",
            "Trigger conditions: 간질환 liver disease 술 알코올 alcohol 음주\nAffected entities: acetaminophen paracetamol 타이레놀",
            affected_entities="acetaminophen, paracetamol, 타이레놀",
            caution_level="high",
        ),
        Doc(
            "ING_OMEGA3",
            "ingredient_profile",
            canonical="omega3",
            name="오메가3",
            category="fatty_acid",
        ),
    ]

    omega3_docs = engine._priority_kb_docs(parse_supplement_query("와파린 먹는데 오메가3 먹어도 돼?"))
    magnesium_docs = engine._priority_kb_docs(parse_supplement_query("신장질환 있는데 마그네슘 먹어도 돼?"))
    acetaminophen_docs = engine._priority_kb_docs(parse_supplement_query("타이레놀 자주 먹고 술도 마시는데 괜찮아?"))

    assert omega3_docs[0].metadata["id"] == "INT_OMEGA3_ANTICOAGULANTS"
    assert magnesium_docs[0].metadata["id"] == "SAFE_KIDNEY_MAGNESIUM"
    assert acetaminophen_docs[0].metadata["id"] == "SAFE_ACETAMINOPHEN_LIVER_ALCOHOL"


def test_supplement_priority_kb_docs_keep_existing_interaction_and_timing_sources():
    engine = SupplementRAGEngine.degraded("test")

    class Doc:
        def __init__(self, doc_id, source, page_content="", **metadata):
            self.metadata = {"id": doc_id, "source": source, **metadata}
            self.page_content = page_content or doc_id

    engine.original_docs = [
        Doc(
            "INT_IRON_CAFFEINE",
            "interaction_rule",
            entity_a_canonical="iron",
            entity_b_canonical="caffeine",
            interaction_type="absorption_interference",
            caution_level="moderate",
        ),
        Doc(
            "INT_IRON_ANTIBIOTICS",
            "interaction_rule",
            entity_a_canonical="iron",
            entity_b_canonical="antibiotics",
            interaction_type="medication_timing_interference",
            caution_level="high",
        ),
        Doc(
            "INT_MAGNESIUM_THYROID_MEDICATION",
            "interaction_rule",
            entity_a_canonical="magnesium",
            entity_b_canonical="thyroid medication",
            interaction_type="medication_timing_interference",
            caution_level="high",
        ),
        Doc("ING_MAGNESIUM", "ingredient_profile", canonical="magnesium"),
        Doc("ING_CREATINE", "ingredient_profile", canonical="creatine"),
    ]

    assert engine._priority_kb_docs(parse_supplement_query("철분은 커피랑 같이 먹어도 돼?"))[0].metadata["id"] == "INT_IRON_CAFFEINE"
    assert engine._priority_kb_docs(parse_supplement_query("항생제 먹는데 철분 먹어도 돼?"))[0].metadata["id"] == "INT_IRON_ANTIBIOTICS"
    assert (
        engine._priority_kb_docs(parse_supplement_query("갑상선약 먹는데 마그네슘 같이 먹어도 돼?"))[0].metadata["id"]
        == "INT_MAGNESIUM_THYROID_MEDICATION"
    )
    assert engine._priority_kb_docs(parse_supplement_query("마그네슘은 언제 먹는 게 좋아?"))[0].metadata["id"] == "ING_MAGNESIUM"
    assert engine._priority_kb_docs(parse_supplement_query("크레아틴은 언제 먹는 게 좋아?"))[0].metadata["id"] == "ING_CREATINE"

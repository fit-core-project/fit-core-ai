import json
from pathlib import Path
from types import SimpleNamespace

from engines.supplement.supplement_engine import SupplementRAGEngine


CORPUS_PATH = Path("data/documents/rag_data_supplement_timing.json")


def _load_timing_corpus() -> list[dict]:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def test_common_supplement_timing_corpus_covers_core_questions():
    corpus = _load_timing_corpus()
    text = json.dumps(corpus, ensure_ascii=False).lower()

    required_terms = [
        "마그네슘",
        "아연",
        "비타민d",
        "오메가3",
        "크레아틴",
        "철분",
        "칼슘",
        "유산균",
        "프로틴",
        "카페인",
    ]

    for term in required_terms:
        assert term in text


def test_common_supplement_timing_corpus_includes_guardrail_topics():
    corpus = _load_timing_corpus()
    text = json.dumps(corpus, ensure_ascii=False)

    required_guardrails = [
        "갑상선약",
        "항생제",
        "와파린",
        "신장질환",
        "임신",
        "항응고제",
        "전문가",
    ]

    for term in required_guardrails:
        assert term in text


def test_supplement_keyword_expansion_covers_common_questions(monkeypatch):
    engine = SupplementRAGEngine.degraded("test")
    monkeypatch.setenv("SUPPLEMENT_ENABLE_LLM_QUERY_OPTIMIZER", "false")

    cases = {
        "마그네슘은 언제 먹는 게 좋아?": ["magnesium", "식후", "신장질환"],
        "크레아틴은 언제 먹는 게 좋아?": ["creatine", "꾸준히", "수분"],
        "비타민D는 식후에 먹는 게 좋아?": ["vitamin d", "지용성"],
        "오메가3 복용 시 주의사항 알려줘": ["omega-3", "fish oil", "와파린"],
        "아연은 공복에 먹어도 돼?": ["zinc", "공복", "항생제"],
        "철분은 커피랑 같이 먹어도 돼?": ["iron", "커피", "칼슘"],
        "유산균은 언제 먹는 게 좋아?": ["probiotics", "항생제"],
        "프로틴은 운동 전이 좋아 후가 좋아?": ["whey protein", "총 단백질"],
    }

    for question, expected_terms in cases.items():
        expanded = engine._deterministic_query(question)
        for term in expected_terms:
            assert term in expanded


def test_supplement_keyword_expansion_covers_guardrail_questions(monkeypatch):
    engine = SupplementRAGEngine.degraded("test")
    monkeypatch.setenv("SUPPLEMENT_ENABLE_LLM_QUERY_OPTIMIZER", "false")

    cases = {
        "갑상선약 먹는데 마그네슘 같이 먹어도 돼?": ["마그네슘", "갑상선약", "간격"],
        "항생제 먹는데 아연 먹어도 돼?": ["아연", "항생제", "간격"],
        "와파린 먹는데 오메가3 먹어도 돼?": ["오메가3", "와파린", "출혈"],
        "신장질환 있는데 마그네슘 먹어도 돼?": ["마그네슘", "신장질환"],
        "임신 중인데 비타민D 고용량 먹어도 돼?": ["비타민D", "고용량", "임신"],
        "타이레놀이랑 오메가3 같이 먹어도 돼?": ["오메가3", "omega-3"],
    }

    for question, expected_terms in cases.items():
        expanded = engine._deterministic_query(question)
        for term in expected_terms:
            assert term in expanded


def test_supplement_timing_docs_are_promoted_for_matching_question():
    engine = SupplementRAGEngine.degraded("test")
    magnesium_doc = SimpleNamespace(
        page_content="Name: 마그네슘 Magnesium\nTiming: 식후나 저녁 시간대\nCautions: 신장질환",
        metadata={"source": "supp_timing", "id": "SUPP_TIMING_MAGNESIUM"},
    )
    unrelated_doc = SimpleNamespace(
        page_content="Name: 카페인 Caffeine\nTiming: 운동 전",
        metadata={"source": "supp_timing", "id": "SUPP_TIMING_CAFFEINE"},
    )
    engine.original_docs = [magnesium_doc, unrelated_doc]

    result = engine._supplement_timing_docs_for_query("마그네슘은 언제 먹는 게 좋아?")

    assert result == [magnesium_doc]

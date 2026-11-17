"""Contract tests for the supplement chat's optional `history` field and follow-up resolution."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import main
from engines.supplement.query_understanding import EntityType, parse_supplement_query
from engines.supplement_history import resolve_follow_up_question


def _supplement_canonicals(text: str) -> set[str]:
    return {e.canonical for e in parse_supplement_query(text).entities if e.type == EntityType.SUPPLEMENT_INGREDIENT}


# --- model validation -------------------------------------------------------


def test_history_defaults_to_empty_list():
    req = main.SupplementChatRequest(question="마그네슘 먹어도 돼?")
    assert req.history == []
    assert main.SupplementChatRequest.model_validate({"question": "q", "history": None}).history == []


def test_history_parses_turns():
    req = main.SupplementChatRequest.model_validate(
        {
            "question": "그럼 몇 시에 먹어요?",
            "history": [
                {"role": "user", "content": "마그네슘 먹어도 돼?"},
                {"role": "assistant", "content": "저녁에 많이 먹어요."},
            ],
        }
    )
    assert [t.role for t in req.history] == ["user", "assistant"]
    assert req.history[0].content == "마그네슘 먹어도 돼?"


@pytest.mark.parametrize("role", ["ai", "system", "", None])
def test_history_rejects_unknown_role(role):
    with pytest.raises(ValidationError):
        main.SupplementChatRequest.model_validate({"question": "q", "history": [{"role": role, "content": "x"}]})


def test_history_content_is_truncated_not_rejected():
    req = main.SupplementChatRequest.model_validate(
        {"question": "q", "history": [{"role": "user", "content": "가" * 5000}]}
    )
    assert len(req.history[0].content) == main.CHAT_TURN_MAX_CONTENT_CHARS


def test_history_keeps_only_most_recent_turns():
    turns = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"} for i in range(20)]
    req = main.SupplementChatRequest.model_validate({"question": "q", "history": turns})
    assert len(req.history) == main.CHAT_HISTORY_MAX_TURNS
    assert req.history[0].content == "turn 8"
    assert req.history[-1].content == "turn 19"


def test_bad_history_returns_422_without_echoing_content():
    client = TestClient(main.app)
    response = client.post(
        "/api/ai/supplement-chat",
        json={"question": "q", "history": [{"role": "bot", "content": "secret history text"}]},
    )
    assert response.status_code == 422
    assert "secret history text" not in response.text


# --- follow-up resolution ---------------------------------------------------


def _turns(*pairs):
    return [main.ChatTurn(role=role, content=content) for role, content in pairs]


def test_follow_up_without_entity_borrows_previous_supplement():
    history = _turns(("user", "마그네슘 먹어도 돼?"), ("assistant", "네, 보통 저녁에 먹어요."))
    effective = resolve_follow_up_question("그럼 몇 시에 먹어요?", history)
    assert effective == "마그네슘 관련 추가 질문: 그럼 몇 시에 먹어요?"
    # round trip: the rewritten question parses back to the same ingredient, so retrieval targets it
    assert _supplement_canonicals(effective) == {"magnesium"}


def test_follow_up_uses_most_recent_user_turn_with_entities():
    history = _turns(
        ("user", "오메가3 먹어도 돼?"),
        ("assistant", "식사와 함께 드세요."),
        ("user", "와파린 먹는 중인데 크레아틴은요?"),
        ("assistant", "의사와 상의하세요."),
        ("user", "고마워요"),
        ("assistant", "천만에요."),
    )
    effective = resolve_follow_up_question("공복에 먹어도 돼요?", history)
    assert effective.endswith("관련 추가 질문: 공복에 먹어도 돼요?")
    assert _supplement_canonicals(effective) == {"creatine"}
    drugs = {e.canonical for e in parse_supplement_query(effective).entities if e.type == EntityType.DRUG_OR_DRUG_CLASS}
    assert "warfarin" in drugs


def test_assistant_turns_are_not_used_as_entity_source():
    history = _turns(("user", "안녕하세요"), ("assistant", "마그네슘이나 오메가3에 대해 물어보세요."))
    assert resolve_follow_up_question("몇 시에 먹어요?", history) == "몇 시에 먹어요?"


def test_no_rewrite_when_question_has_own_supplement():
    history = _turns(("user", "마그네슘 먹어도 돼?"), ("assistant", "네."))
    question = "철분은 언제 먹어요?"
    assert resolve_follow_up_question(question, history) is question


def test_no_rewrite_with_empty_history():
    question = "  그럼 몇 시에 먹어요?  "
    assert resolve_follow_up_question(question, []) is question
    assert resolve_follow_up_question(question, None) is question


# --- endpoint wiring --------------------------------------------------------


class _RecordingEngine:
    def __init__(self):
        self.questions = []

    def answer_question(self, question):
        self.questions.append(question)
        return {"answer": "ok", "sources": [], "mode": "full"}


def test_endpoint_passes_question_unchanged_without_history(monkeypatch):
    engine = _RecordingEngine()
    monkeypatch.setattr(main, "supplement_rag", engine)
    question = "  그럼 몇 시에 먹어요? "
    assert main.api_supplement_chat(main.SupplementChatRequest(question=question)) == {
        "answer": "ok",
        "sources": [],
        "mode": "full",
    }
    assert engine.questions == [question]
    assert engine.questions[0] is question


def test_endpoint_passes_effective_question_with_history(monkeypatch):
    engine = _RecordingEngine()
    monkeypatch.setattr(main, "supplement_rag", engine)
    client = TestClient(main.app)
    response = client.post(
        "/api/ai/supplement-chat",
        json={
            "question": "그럼 몇 시에 먹어요?",
            "history": [
                {"role": "user", "content": "마그네슘 먹어도 돼?"},
                {"role": "assistant", "content": "네, 보통 저녁에 먹어요."},
            ],
        },
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "ok"
    assert engine.questions == ["마그네슘 관련 추가 질문: 그럼 몇 시에 먹어요?"]


def test_endpoint_degraded_path_also_uses_effective_question(monkeypatch):
    class _Failing:
        def answer_question(self, question):
            raise RuntimeError("boom")

    seen = []

    class _Degraded:
        def answer_question(self, question):
            seen.append(question)
            return {"answer": "degraded", "mode": "degraded"}

    monkeypatch.setattr(main, "supplement_rag", _Failing())
    monkeypatch.setattr(main.SupplementRAGEngine, "degraded", staticmethod(lambda reason: _Degraded()))
    req = main.SupplementChatRequest.model_validate(
        {"question": "몇 시에 먹어요?", "history": [{"role": "user", "content": "크레아틴 먹어요"}]}
    )
    assert main.api_supplement_chat(req)["mode"] == "degraded"
    assert seen == ["크레아틴 관련 추가 질문: 몇 시에 먹어요?"]


def test_new_topic_and_small_talk_are_not_rewritten():
    from types import SimpleNamespace

    from engines.supplement_history import resolve_follow_up_question

    history = [SimpleNamespace(role="user", content="마그네슘 먹어도 돼?")]
    for question in (
        "다른 영양제도 알려줘",
        "수면에 좋은 영양제 추천해줘",
        "피로에 좋은 거 추천해줘",
        "마그네슘 말고 다른 거 없어?",
        "어떤 영양제가 좋아요?",
        "고마워요",
        "감사합니다!",
    ):
        assert resolve_follow_up_question(question, history) == question, question


def test_ordinary_follow_ups_keep_the_ingredient():
    from types import SimpleNamespace

    from engines.supplement_history import resolve_follow_up_question

    history = [SimpleNamespace(role="user", content="마그네슘 먹어도 돼?")]
    for question in (
        "그 영양제 언제 먹어요?",
        "임신 중인데 괜찮아?",
        "다른 약이랑 같이 먹어도 돼?",
        "같이 뭐 먹으면 안 돼요?",
        "임신 중에도 먹어도 되는 영양제인가요?",
        "언제가 좋아요?",
        "공복에 먹는 게 좋아요?",
        "알겠어요 그럼 부작용은?",
        "커피 대신 물이랑 먹어야 해?",
        "추천 복용량이 얼마예요?",
        "하루 권장량 추천해줘",
        "괜찮은 영양제인가요?",
        "어떤 성분이 들어 있어요?",
    ):
        assert resolve_follow_up_question(question, history).startswith("마그네슘 관련 추가 질문"), question


def test_topic_change_resets_the_context():
    from types import SimpleNamespace

    from engines.supplement_history import resolve_follow_up_question

    history = [
        SimpleNamespace(role="user", content="마그네슘 먹어도 돼?"),
        SimpleNamespace(role="assistant", content="..."),
        SimpleNamespace(role="user", content="다른 영양제도 알려줘"),
        SimpleNamespace(role="assistant", content="..."),
    ]
    assert resolve_follow_up_question("그럼 언제 먹어요?", history) == "그럼 언제 먹어요?"
    # a new ingredient after the topic change becomes the context
    history.append(SimpleNamespace(role="user", content="오메가3는 어때?"))
    assert resolve_follow_up_question("그럼 언제 먹어요?", history).startswith("오메가3 관련 추가 질문")


def test_excluding_the_ingredient_is_a_topic_change():
    from types import SimpleNamespace

    from engines.supplement_history import resolve_follow_up_question

    history = [
        SimpleNamespace(role="user", content="마그네슘 먹어도 돼?"),
        SimpleNamespace(role="user", content="마그네슘 말고 다른 거 있어?"),
    ]
    assert resolve_follow_up_question("그럼 언제 먹어요?", history) == "그럼 언제 먹어요?"


def test_other_excluding_phrasings_reset_the_context():
    from types import SimpleNamespace

    from engines.supplement_history import resolve_follow_up_question

    for shift in ("오메가3 빼고 뭐가 있어?", "마그네슘 대신 다른 거 있어?", "마그네슘 외에 다른 영양제는?"):
        history = [SimpleNamespace(role="user", content="마그네슘 먹어도 돼?"), SimpleNamespace(role="user", content=shift)]
        assert resolve_follow_up_question("그럼 부작용은?", history) == "그럼 부작용은?", shift


def test_drug_only_follow_up_keeps_the_ingredient():
    from types import SimpleNamespace

    from engines.supplement_history import resolve_follow_up_question

    history = [
        SimpleNamespace(role="user", content="마그네슘 먹어도 돼?"),
        SimpleNamespace(role="user", content="와파린 먹는데 괜찮아?"),
    ]
    effective = resolve_follow_up_question("공복에는요?", history)
    assert effective.startswith("마그네슘, 와파린 관련 추가 질문"), effective

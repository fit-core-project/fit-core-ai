"""Follow-up resolution for the supplement chat's optional conversation history.

The engine answers one question at a time. When the current question names no supplement
("그럼 몇 시에 먹어요?"), this module borrows the supplement / drug mentions of the most recent
user turn that had some, so retrieval still targets the right ingredient. The engine itself is
not touched: it only receives a rewritten question string.
"""

from __future__ import annotations

import re
from typing import Iterable, Protocol

from engines.supplement.query_understanding import EntityType, parse_supplement_query

FOLLOW_UP_MARKER = "관련 추가 질문"

# A question starts a new topic only when it looks for a DIFFERENT or general supplement
# ("다른 영양제도", "마그네슘 말고 다른 거", "수면에 좋은 영양제 추천"), or when the whole message is
# small talk ("고마워요"). Ordinary follow-ups that merely contain "다른 약", "같이 뭐 먹으면",
# "먹어도 되는 영양제인가요" or "언제가 좋아요?" keep the previous ingredient.
_NEW_TOPIC_PATTERNS = (
    re.compile(r"다른\s*(?:영양제|보충제|성분|제품|비타민)"),
    re.compile(r"(?:말고|빼고|외에|이외에?|대신)\s*(?:다른|뭐|어떤|딴)"),
    re.compile(r"다른\s*(?:것|거|걸)\s*(?:있|없|알려|뭐|추천)"),
    re.compile(r"(?:어떤|무슨|뭔|어느)\s*(?:영양제|보충제)"),
    # "수면에 좋은 영양제", but not the confirmation "괜찮은 영양제인가요?"
    re.compile(r"(?:좋은|먹을\s*만한|괜찮은)\s*(?:영양제|보충제)(?!\s*(?:인가|이에|예요|인지|맞|일까|이야|이죠|인데))"),
)
# "추천" asks for something new ("수면에 좋은 거 추천해줘") unless it is about the dose or time
# of the current one ("추천 복용량이 얼마예요?", "하루 권장량 추천해줘").
_RECOMMEND_RE = re.compile(r"추천")
_EXCLUDING_RE = re.compile(r"말고|빼고|외에|이외|대신\s*다른")
_DOSE_WORDS_RE = re.compile(r"복용량|용량|권장량|섭취량|하루\s*몇|몇\s*(?:mg|알|정|번)|언제|시간|타이밍")
_SMALL_TALK_RE = re.compile(
    r"^(?:고마워요?|고맙습니다|감사(?:합니다|해요|요)?|안녕(?:하세요)?|알겠(?:어요|습니다|어)|수고(?:하세요|하셨어요)?"
    r"|넵|네|ㅇㅋ|오케이|ok|okay|thanks?(?:you)?)[!.~ㅎㅋ^]*$"
)


def is_topic_shift(question: str) -> bool:
    """The user asks for a different or general supplement: earlier turns are no longer the topic."""
    text = (question or "").strip().lower()
    if any(pattern.search(text) for pattern in _NEW_TOPIC_PATTERNS):
        return True
    return bool(_RECOMMEND_RE.search(text) and not _DOSE_WORDS_RE.search(text))


def is_new_topic(question: str) -> bool:
    """True for a question that should not inherit the previous turn's ingredient."""
    if is_topic_shift(question):
        return True
    text = (question or "").strip().lower()
    compact = "".join(text.split())
    return "?" not in compact and bool(_SMALL_TALK_RE.match(compact))


def resolve_follow_up_question(question: str, history: Iterable[_Turn] | None) -> str:
    """Return the question to send to the engine.

    Unchanged (same object, not stripped) when history is empty, when the question already names a
    supplement ingredient, or when no earlier user turn mentions a supplement or drug.
    """
    turns = list(history or [])
    if not turns:
        return question

    current = parse_supplement_query(question)
    if any(e.type == EntityType.SUPPLEMENT_INGREDIENT for e in current.entities):
        return question
    if is_new_topic(question):
        return question
    current_keys = {(e.type, e.canonical) for e in current.entities}

    # Walk back through user turns. A drug-only turn ("와파린 먹는데 괜찮아?") is collected and the walk
    # continues to the ingredient it was about, so later follow-ups keep both ("마그네슘, 와파린").
    drug_labels: list[str] = []

    def _rewrite(labels: list[str]) -> str:
        return f"{', '.join(labels)} {FOLLOW_UP_MARKER}: {question.strip()}"

    for turn in reversed(turns):
        if getattr(turn, "role", None) != "user":
            continue
        content = getattr(turn, "content", "") or ""
        parsed = parse_supplement_query(content)
        named = any(e.type in (EntityType.SUPPLEMENT_INGREDIENT, EntityType.DRUG_OR_DRUG_CLASS) for e in parsed.entities)
        # "마그네슘 말고 다른 거 있어?" names magnesium only to exclude it: that is a topic change too
        if is_topic_shift(content) and (not named or _EXCLUDING_RE.search(content)):
            # the user changed the subject ("다른 영양제도 알려줘"): nothing before it is the topic any more
            return _rewrite(drug_labels) if drug_labels else question
        supplement_labels: list[str] = []
        for entity in parsed.entities:
            if (entity.type, entity.canonical) in current_keys:
                continue
            # The alias text the user wrote parses back to the same entity, so retrieval keeps working.
            label = (entity.text or entity.canonical).strip()
            if not label:
                continue
            if entity.type == EntityType.SUPPLEMENT_INGREDIENT and label not in supplement_labels:
                supplement_labels.append(label)
            elif entity.type == EntityType.DRUG_OR_DRUG_CLASS and label not in drug_labels:
                drug_labels.append(label)
        if supplement_labels:
            return _rewrite(supplement_labels + [d for d in drug_labels if d not in supplement_labels])

    return _rewrite(drug_labels) if drug_labels else question

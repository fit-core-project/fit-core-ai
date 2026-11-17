from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass
from typing import Any

from engines.supplement.query_understanding import EntityType, IntentType, ParsedSupplementQuery


_RISK_INTENTS = {
    IntentType.DRUG_INTERACTION,
    IntentType.FOOD_COMPOUND_INTERACTION,
    IntentType.SUPPLEMENT_INTERACTION,
    IntentType.CONDITION_SAFETY,
    IntentType.PREGNANCY_OR_HIGH_DOSE_SAFETY,
    IntentType.MEDICATION_SAFETY,
    IntentType.SIDE_EFFECT,
    IntentType.LONG_TERM_USE,
    IntentType.SYMPTOM_AFTER_INTAKE,
}
_UNUSABLE_ANSWERS = {"", "{}", "[]", "null", "none"}
_WEAK_ANSWER_NEEDLES = (
    "정보가 부족",
    "자료 없음",
    "답변할 수 없습니다",
    "WEB_SEARCH_REQUIRED",
    "확보된 로컬 근거 문서만으로",
)
_UNSAFE_PHRASES = (
    "먹어도 됩니다",
    "복용해도 됩니다",
    "문제 없습니다",
    "안전합니다",
    "무조건 안전",
    "걱정하지 않아도 됩니다",
    "무조건 괜찮습니다",
)
_CONSULT_TEXT = "개인 상태, 복용 중인 약, 용량에 따라 달라질 수 있으므로 의사 또는 약사와 상담하세요."


_COMPACT_CONSULT_TEXT = "개인 상태, 복용 중인 약, 용량에 따라 달라질 수 있으므로 복용 전 의사 또는 약사와 상담하세요."
_MAX_CAUTION_SENTENCES = 3
_MAX_CAUTION_CHARS = 520


@dataclass(frozen=True)
class ComposedSupplementResponse:
    answer: str
    caution: str | None = None
    used_kb_fallback: bool = False


def compose_supplement_response(
    question: str,
    parsed_query: ParsedSupplementQuery,
    generated_answer: str | dict | None,
    generated_caution: str | None,
    selected_docs: list[Any],
    llm_refused: bool = False,
) -> ComposedSupplementResponse:
    answer, parsed_caution = _normalize_answer(generated_answer)
    caution_parts = [generated_caution, parsed_caution]
    caution_parts.extend(_caution_parts_from_docs(selected_docs, parsed_query, answer))

    # A medication in the question is a risk question even when the other ingredient is not in
    # the KB (아쉬와간다 + 갑상선약): no DRUG_INTERACTION intent is inferred then, but the user
    # still needs the consult advice.
    risk_query = bool(set(parsed_query.intents) & _RISK_INTENTS) or any(
        entity.type == EntityType.DRUG_OR_DRUG_CLASS for entity in parsed_query.entities
    )
    if risk_query:
        # Soften sentence-by-sentence before the weak check so an answer made only of unsafe
        # claims falls back to KB-grounded content instead of a spliced fragment.
        answer = _soften_unsafe_answer(answer)

    used_kb_fallback = False
    if _is_weak_answer(answer):
        answer = _fallback_answer_from_docs(selected_docs, parsed_query, question)
        # KB text phrases recommendations as "...것이 안전합니다"; keep the advice, drop the safety claim.
        answer = _soften_recommendation_phrasing(answer)
        used_kb_fallback = True

    if risk_query and not any(_is_consult_sentence(part) for part in _split_caution_sentences(caution_parts)):
        caution_parts.append(_CONSULT_TEXT)

    caution = _compact_caution(caution_parts, parsed_query, selected_docs, risk_query=risk_query)
    if caution:
        caution = _soften_unsafe_caution(caution)
    if risk_query and not caution:
        caution = _COMPACT_CONSULT_TEXT

    answer = _enhance_answer(
        answer or _fallback_answer_from_docs(selected_docs, parsed_query, question),
        question,
        parsed_query,
        selected_docs,
        allow_follow_up=used_kb_fallback,
    )
    answer = _apply_coverage_honesty(answer, question, parsed_query, selected_docs)
    answer = _apply_efficacy_policy(answer, question, parsed_query, selected_docs)
    answer, caution = _dedupe_answer_and_caution(answer, caution, parsed_query, risk_query=risk_query)
    if llm_refused and answer and not any(m in answer for m in _HONEST_LEAD_MARKERS):
        # the LLM judged the documents insufficient (WEB_SEARCH_REQUIRED); the fallback answer is
        # built from whatever KB text matched, so say first that the question itself is not covered
        answer = f"{_REFUSED_LEAD} {answer}"
        if _has_risk_context(parsed_query) and _CONSULT_TEXT not in answer:
            answer = f"{answer} {_CONSULT_TEXT}"
    if answer and _has_risk_context(parsed_query) and not _CONSULT_WORDS_RE.search(answer)             and not (caution and _CONSULT_WORDS_RE.search(caution)):
        answer = f"{answer} {_CONSULT_TEXT}"  # a drug / condition / risk context always gets the consult line
    answer = _plain_text(_strip_dose_numbers(answer))
    caution = _plain_text(_strip_dose_numbers(caution)) if caution else caution

    return ComposedSupplementResponse(
        answer=answer,
        caution=caution or None,
        used_kb_fallback=used_kb_fallback,
    )


# Policy: no concrete dose numbers in answers (the product label and a professional decide the
# amount). The LLM sometimes writes "하루 1000mg"; replace the number, keep the sentence.
_DOSE_NUMBER_RE = re.compile(
    r"(?:하루\s*)?\d+(?:[.,]\d+)?\s*(?:~|-|∼)?\s*(?:\d+(?:[.,]\d+)?\s*)?"
    r"(?:mg|㎎|밀리그램|mcg|µg|μg|IU|iu|그램|g)(?![A-Za-z])"
)


_CONSULT_WORDS_RE = re.compile(r"의사|약사|의료진|전문가|상담|상의")
_REFUSED_LEAD = "확보된 근거 문서에서는 질문하신 내용을 직접 확인할 수 없어, 아래는 확인된 관련 정보와 일반적인 참고 정보입니다."


def _has_risk_context(parsed_query: ParsedSupplementQuery) -> bool:
    return any(e.type in (EntityType.DRUG_OR_DRUG_CLASS, EntityType.CONDITION, EntityType.RISK_CONTEXT)
               for e in parsed_query.entities)


_MD_EMPHASIS_RE = re.compile(r"(\*\*|__)(.+?)\1")
_NUMBERED_RE = re.compile(r"^(\s*)(\d+)[.)]\s+")
_LABEL_LINE_RE = re.compile(r"^\s*\S[^:\n]{0,30}:\s")


def _plain_text(text: str | None) -> str | None:
    """The app renders plain text: drop markdown emphasis / headings / code marks, turn "* " bullets
    into "- ", and renumber each numbered list from 1 (an LLM list can lose its "1." when the first
    item is a bold label line)."""
    if not text:
        return text
    out = _MD_EMPHASIS_RE.sub(r"\2", text)
    out = out.replace("**", "").replace("`", "")
    lines = []
    for line in out.split("\n"):
        line = re.sub(r"^\s*#{1,6}\s+", "", line)
        line = re.sub(r"^(\s*)\*\s+", r"\1- ", line)
        lines.append(line)
    # renumber contiguous numbered blocks; a label line right before a block that starts at 2 is item 1
    numbered_idx = [i for i, l in enumerate(lines) if _NUMBERED_RE.match(l)]
    if numbered_idx:
        blocks: list[list[int]] = []
        for i in numbered_idx:
            if blocks and all(not lines[j].strip() for j in range(blocks[-1][-1] + 1, i)):
                blocks[-1].append(i)
            else:
                blocks.append([i])
        for block in blocks:
            first = int(_NUMBERED_RE.match(lines[block[0]]).group(2))
            prev = block[0] - 1
            while prev >= 0 and not lines[prev].strip():
                prev -= 1
            start = 1
            if first == 2 and prev >= 0 and _LABEL_LINE_RE.match(lines[prev]) and not _NUMBERED_RE.match(lines[prev]):
                lines[prev] = "1. " + lines[prev].lstrip()
                start = 2
            for n, i in enumerate(block, start):
                lines[i] = _NUMBERED_RE.sub(lambda m: f"{m.group(1)}{n}. ", lines[i], count=1)
    return "\n".join(lines)


def _strip_dose_numbers(text: str | None) -> str | None:
    if not text:
        return text
    out = _DOSE_NUMBER_RE.sub("제품에 표시된 양", text)
    return re.sub(r"(제품에 표시된 양)(\s*(?:이상|이하|정도|까지)?\s*,?\s*제품에 표시된 양)+", r"", out)


def _normalize_answer(value: str | dict | None) -> tuple[str, str | None]:
    if value is None:
        return "", None
    if isinstance(value, dict):
        return _answer_from_dict(value)
    if not isinstance(value, str):
        return str(value), None

    stripped = value.strip()
    if not stripped:
        return "", None
    try:
        parsed = json.loads(stripped)
    except (TypeError, json.JSONDecodeError):
        if stripped.startswith(("{", "[")):
            return "", None
        return stripped, None
    if isinstance(parsed, dict):
        return _answer_from_dict(parsed)
    if isinstance(parsed, str):
        return parsed.strip(), None
    if parsed in (None, [], {}):
        return "", None
    return str(parsed), None


def _answer_from_dict(data: dict[str, Any]) -> tuple[str, str | None]:
    caution = _string_or_none(_dict_get(data, "caution", "warning", "risk", "Risk"))
    answer = _string_or_none(_dict_get(data, "answer", "Answer"))
    if answer and not _is_weak_answer(answer):
        return answer, caution

    direct_answer = _dedupe_join(
        [
            _dict_get(data, "summary", "Summary", "interaction_summary", "Interaction_Summary", "Interaction Summary"),
            _dict_get(data, "recommendation", "Recommendation"),
        ],
        max_chars=700,
    )
    if direct_answer:
        return direct_answer, caution

    safety_check = _dict_get(data, "safety_check", "Safety check", "SafetyCheck")
    if isinstance(safety_check, dict):
        answer = _dedupe_join(
            [
                _dict_get(safety_check, "summary", "Summary"),
                _dict_get(safety_check, "recommendation", "Recommendation"),
            ],
            max_chars=600,
        )
        caution = caution or _string_or_none(_dict_get(safety_check, "caution", "Caution", "risk", "Risk"))
        if answer:
            return answer, caution

    flattened = []
    for key, item in data.items():
        if key in {"answer", "caution", "warning"}:
            continue
        if isinstance(item, str) and item.strip():
            flattened.append(f"{key}: {item.strip()}")
        elif isinstance(item, (dict, list)):
            flattened.append(f"{key}: {json.dumps(item, ensure_ascii=False)}")
    return _dedupe_join(flattened, max_chars=700), caution


def _caution_parts_from_docs(docs: list[Any], parsed_query: ParsedSupplementQuery, answer: str) -> list[str]:
    parts: list[str] = []
    for doc in docs:
        if not _doc_is_relevant_for_caution(doc, parsed_query, answer):
            continue
        meta = getattr(doc, "metadata", {}) or {}
        source = meta.get("source") or meta.get("type")
        text = getattr(doc, "page_content", "") or ""
        if source == "interaction_rule":
            parts.extend(
                [
                    _field(text, "Recommendation"),
                    _field(text, "Spacing guidance"),
                    _list_after(text, "Consultation required when"),
                ]
            )
        elif source == "safety_rule":
            parts.extend(
                [
                    _field(text, "Risk"),
                    _field(text, "Recommendation"),
                    _field(text, "When to consult"),
                ]
            )
        elif source == "ingredient_profile":
            parts.extend(
                [
                    _list_after(text, "Cautions"),
                    _list_after(text, "High risk groups"),
                    _list_after(text, "Spacing"),
                ]
            )
    return [part for part in parts if part]


def _supplement_canonicals() -> set[str]:
    from engines.supplement.query_understanding import ALIAS_REGISTRY
    return {e.canonical.lower() for e in ALIAS_REGISTRY if e.entity_type == EntityType.SUPPLEMENT_INGREDIENT}


def _doc_supplement_canonicals(meta: dict) -> set[str]:
    """Supplement ingredients a KB doc is about (profile canonical, rule entities, affected list)."""
    known = _supplement_canonicals()
    values: set[str] = set()
    for key in ("canonical", "entity_a_canonical", "entity_b_canonical", "affected_entities"):
        raw = meta.get(key)
        parts = raw.split(",") if isinstance(raw, str) else (raw or [])
        for part in parts:
            # profile ids say "probiotic", the parser says "probiotics": compare expanded forms
            values.update(_expand_canonical(str(part)))
    return {v for v in values if v in known}


def _doc_is_relevant_for_caution(doc: Any, parsed_query: ParsedSupplementQuery, answer: str) -> bool:
    meta = getattr(doc, "metadata", {}) or {}
    source = meta.get("source") or meta.get("type")
    if source not in {"interaction_rule", "safety_rule", "ingredient_profile"}:
        return False

    query_values = {entity.canonical.lower() for entity in parsed_query.entities}
    # A rule or profile about a specific ingredient only belongs in the caution when that
    # ingredient was asked about. Shared risk words ("장기 복용", "고용량") let
    # SAFE_HIGH_DOSE_SELENIUM into a creatine/kidney answer after the 2026-11-11 KB expansion.
    doc_supplements = _doc_supplement_canonicals(meta)
    if doc_supplements and not (doc_supplements & query_values):
        return False
    answer_text = (answer or "").lower()
    doc_values: set[str] = set()
    doc_id = meta.get("id")
    if doc_id:
        doc_values.update(part for part in str(doc_id).lower().replace("-", "_").split("_") if part)
    for key in ("canonical", "entity_a_canonical", "entity_b_canonical", "affected_entities"):
        value = meta.get(key)
        if isinstance(value, str):
            doc_values.update(part.strip().lower() for part in value.split(",") if part.strip())
        elif isinstance(value, list):
            doc_values.update(str(part).strip().lower() for part in value if str(part).strip())

    doc_values -= _GENERIC_DOC_ID_PARTS
    if query_values and doc_values & query_values:
        return True
    doc_text = (getattr(doc, "page_content", "") or "").lower()
    if query_values and any(_contains_token(doc_text, value) for value in query_values):
        return True
    return any(_contains_token(answer_text, value) for value in doc_values)


# Doc-id fragments that name the rule type, not an ingredient ("INT_IRON_CAFFEINE" -> "int").
_GENERIC_DOC_ID_PARTS = {
    "int", "rule", "rules", "safety", "interaction", "profile", "ingredient",
    "medication", "medications", "mineral", "overlap", "and", "with",
}


def _contains_token(text: str, value: str) -> bool:
    """Whole-token match so 'c' (vitamin_c) or 'iron' does not hit inside other words."""
    if not value or len(value) < 2:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(value)}(?![a-z0-9])", text) is not None


def _compact_caution(
    values: list[Any],
    parsed_query: ParsedSupplementQuery,
    docs: list[Any],
    *,
    risk_query: bool,
) -> str:
    sentences = _dedupe_caution_sentences(_split_caution_sentences(values))
    if not sentences:
        return ""

    ranked = sorted(
        enumerate(sentences),
        key=lambda item: (-_rank_caution_sentence(item[1], parsed_query, docs, risk_query=risk_query), item[0]),
    )

    selected: list[str] = []
    selected_categories: set[str] = set()
    for _idx, sentence in ranked:
        category = _caution_category(sentence)
        if category != "general" and category in selected_categories:
            continue
        candidate = " ".join([*selected, sentence]).strip()
        if selected and len(candidate) > _MAX_CAUTION_CHARS:
            continue
        selected.append(sentence)
        selected_categories.add(category)
        if len(selected) >= _MAX_CAUTION_SENTENCES:
            break

    if risk_query and not any(_contains_any(sentence, {"\uc0c1\ub2f4", "\uc758\ub8cc\uc9c4", "\uc758\uc0ac", "\uc57d\uc0ac", "consult"}) for sentence in selected):
        if len(selected) >= _MAX_CAUTION_SENTENCES:
            selected[-1] = _COMPACT_CONSULT_TEXT
        else:
            selected.append(_COMPACT_CONSULT_TEXT)

    compacted = " ".join(selected).strip()
    if len(compacted) <= _MAX_CAUTION_CHARS:
        return compacted
    return _truncate_by_sentence(compacted, _MAX_CAUTION_CHARS)


def _split_caution_sentences(values: list[Any]) -> list[str]:
    sentences: list[str] = []
    for value in values:
        text = _string_or_none(value)
        if not text:
            continue
        normalized = _normalize_caution_text(text)
        for part in re.split(r"(?<=[.!?。！？])\s+|\s{2,}", normalized):
            sentence = part.strip(" .;；")
            if not sentence or len(sentence) < 3:
                continue
            if sentence.lower() in {"low", "moderate", "high", "critical"}:
                continue
            if sentence[-1] not in ".!?。！？":
                sentence = f"{sentence}."
            sentences.append(sentence)
    return sentences


def _normalize_caution_text(text: str) -> str:
    normalized = text.replace("•", ". ").replace("- ", ". ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _dedupe_caution_sentences(sentences: list[str]) -> list[str]:
    seen_exact: set[str] = set()
    seen_categories: set[str] = set()
    deduped: list[str] = []
    for sentence in sentences:
        normalized = re.sub(r"\s+", " ", sentence).strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen_exact:
            continue
        category = _caution_category(normalized)
        if category in {"consult", "kidney"} and category in seen_categories:
            continue
        seen_exact.add(lowered)
        seen_categories.add(category)
        deduped.append(normalized)
    return deduped


def _rank_caution_sentence(
    sentence: str,
    parsed_query: ParsedSupplementQuery,
    docs: list[Any],
    *,
    risk_query: bool,
) -> int:
    score = 0
    text = sentence.lower()
    intents = set(parsed_query.intents)
    doc_types = {(getattr(doc, "metadata", {}) or {}).get("source") or (getattr(doc, "metadata", {}) or {}).get("type") for doc in docs}

    if risk_query:
        score += 20
    if "safety_rule" in doc_types and _contains_any(text, {"\uc2e0\uc7a5", "kidney", "renal", "\uac04\uc9c8\ud658", "liver", "\uc784\uc2e0", "pregnancy"}):
        score += 30
    if "interaction_rule" in doc_types and _contains_any(text, {"\uc640\ud30c\ub9b0", "warfarin", "\ud56d\uc751\uace0", "anticoagulant", "\ucd9c\ud608", "bleeding", "\uac11\uc0c1\uc120", "thyroid", "\ud56d\uc0dd\uc81c", "antibiotic", "\ud761\uc218", "\uac04\uaca9"}):
        score += 28
    if IntentType.CONDITION_SAFETY in intents and _contains_any(text, {"\uc2e0\uc7a5", "kidney", "renal", "\uac04\uc9c8\ud658", "liver"}):
        score += 25
    if IntentType.DRUG_INTERACTION in intents and _contains_any(text, {"\uc640\ud30c\ub9b0", "warfarin", "\ud56d\uc751\uace0", "anticoagulant", "\uac11\uc0c1\uc120", "thyroid", "\ud56d\uc0dd\uc81c", "antibiotic"}):
        score += 24
    if IntentType.FOOD_COMPOUND_INTERACTION in intents and _contains_any(text, {"\ucee4\ud53c", "coffee", "\uce74\ud398\uc778", "caffeine", "\uce7c\uc298", "calcium", "\ud761\uc218"}):
        score += 22
    if _contains_any(text, {"\ucd9c\ud608", "bleeding", "\uace0\uc6a9\ub7c9", "high dose", "\uc784\uc758", "\uc2dc\uc791", "\ub298\ub9ac\uc9c0", "\ud53c\ud558", "\uc8fc\uc758"}):
        score += 12
    if _contains_any(text, {"\uc0c1\ub2f4", "\uc758\ub8cc\uc9c4", "\uc758\uc0ac", "\uc57d\uc0ac", "consult"}):
        score += 8
    if _contains_any(text, {"\uc218\uc220", "\uc2dc\uc220", "surgery", "procedure", "\uba4d"}):
        score += 6
    return score


def _caution_category(sentence: str) -> str:
    text = sentence.lower()
    if _contains_any(text, {"\uc640\ud30c\ub9b0", "warfarin", "\ud56d\uc751\uace0", "anticoagulant"}):
        return "anticoagulant"
    if _contains_any(text, {"\ucd9c\ud608", "bleeding"}):
        return "bleeding"
    if _contains_any(text, {"\uc2e0\uc7a5", "kidney", "renal"}):
        return "kidney"
    if _contains_any(text, {"\uac11\uc0c1\uc120", "thyroid", "\ud56d\uc0dd\uc81c", "antibiotic", "\ud761\uc218", "\uac04\uaca9", "spacing"}):
        return "spacing"
    if _contains_any(text, {"\ucee4\ud53c", "coffee", "\uce74\ud398\uc778", "caffeine", "\uce7c\uc298", "calcium", "\ucca0\ubd84", "iron"}):
        return "absorption"
    if _contains_any(text, {"\uc218\uc220", "\uc2dc\uc220", "surgery", "procedure"}):
        return "surgery"
    if _contains_any(text, {"\uac04\uc9c8\ud658", "liver", "\uc74c\uc8fc", "alcohol"}):
        return "liver_alcohol"
    if _contains_any(text, {"\uc784\uc2e0", "pregnancy", "\uc218\uc720", "breastfeeding", "\uace0\uc6a9\ub7c9", "high dose"}):
        return "pregnancy_high_dose"
    if _contains_any(text, {"\uc0c1\ub2f4", "\uc758\ub8cc\uc9c4", "\uc758\uc0ac", "\uc57d\uc0ac", "consult"}):
        return "consult"
    return "general"


def _contains_any(text: str, needles: set[str]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def _truncate_by_sentence(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    kept: list[str] = []
    for sentence in _split_caution_sentences([text]):
        candidate = " ".join([*kept, sentence]).strip()
        if kept and len(candidate) > max_chars:
            break
        kept.append(sentence)
    if kept:
        return " ".join(kept).strip()
    return text[:max_chars].rsplit(" ", 1)[0].strip()


def _fallback_answer_from_docs(
    docs: list[Any],
    parsed_query: ParsedSupplementQuery | None = None,
    question: str = "",
) -> str:
    source_types = {(getattr(doc, "metadata", {}) or {}).get("source") for doc in docs}
    intents = set(parsed_query.intents) if parsed_query else set()
    if parsed_query:
        direct = _direct_answer_from_query_and_docs(parsed_query, docs)
        if direct:
            return direct
        grounded = _grounded_answer_from_entity_docs(question or parsed_query.raw_query, parsed_query, docs)
        if grounded:
            return grounded
        if not parsed_query.entities and IntentType.GOAL_BASED_RECOMMENDATION not in intents:
            if any(needle in (question or "") for needle in _GOAL_CONTEXT_NEEDLES):
                return _GOAL_RECOMMENDATION_ANSWER
            # Truly unknown ingredient/product: never build an answer from unrelated retrieved docs.
            return _UNKNOWN_INGREDIENT_ANSWER
    if IntentType.TIMING in intents and not (intents & _RISK_INTENTS) and "ingredient_profile" in source_types:
        return (
            "일반적인 복용 타이밍은 성분과 목적에 따라 달라질 수 있습니다. 검색된 성분 프로필의 복용 가이드와 "
            "주의사항을 기준으로 식사 여부, 운동 전후, 다른 약물과의 간격을 함께 확인하는 것이 좋습니다."
        )
    if source_types & {"safety_rule"} and intents & {
        IntentType.CONDITION_SAFETY,
        IntentType.PREGNANCY_OR_HIGH_DOSE_SAFETY,
        IntentType.MEDICATION_SAFETY,
    }:
        return (
            "해당 조건에서는 보충제나 약 복용을 임의로 결정하지 않는 것이 좋습니다. 검색된 안전 규칙에 따르면 질환, "
            "임신/수유, 수술 전후, 고용량 복용 같은 조건에서는 전문가 상담이 필요할 수 있습니다."
        )
    if "interaction_rule" in source_types:
        return (
            "해당 조합은 함께 복용할 때 주의가 필요합니다. 검색된 상호작용 규칙에 따르면 복용 간격, 약물 병용, "
            "출혈 위험 등 확인이 필요한 요소가 있을 수 있으므로 임의로 병용하지 말고 의사 또는 약사에게 안전성을 확인하는 것이 좋습니다."
        )
    if "safety_rule" in source_types:
        return (
            "해당 조건에서는 보충제나 약 복용을 임의로 결정하지 않는 것이 좋습니다. 검색된 안전 규칙에 따르면 질환, "
            "임신/수유, 수술 전후, 고용량 복용 같은 조건에서는 전문가 상담이 필요할 수 있습니다."
        )
    if "ingredient_profile" in source_types:
        return (
            "일반적인 복용 타이밍은 성분과 목적에 따라 달라질 수 있습니다. 검색된 성분 프로필의 복용 가이드와 "
            "주의사항을 기준으로 식사 여부, 운동 전후, 다른 약물과의 간격을 함께 확인하는 것이 좋습니다."
        )
    return _UNKNOWN_INGREDIENT_ANSWER


_UNKNOWN_INGREDIENT_ANSWER = (
    "확보된 근거 문서에서 확인되지 않는 성분이나 제품이라 복용 가능 여부를 단정하기 어렵습니다. "
    "제품명, 성분표, 함량, 복용량과 함께 복용 중인 약이나 질환 정보를 알려주면 확인 가능한 범위에서 나눠서 안내할 수 있습니다. "
    "복용 중인 약이 있다면 함께 먹기 전에 의사 또는 약사와 상담하세요."
)
_TIMING_LABEL_KEYS = (
    ("General:", "general"),
    ("With food:", "with_food"),
    ("Before bed:", "before_bed"),
    ("Exercise related:", "exercise_related"),
)
_MEAL_NEEDLES = ("공복", "식후", "식전", "식사", "밥")
_EXERCISE_NEEDLES = ("운동",)
_BEDTIME_NEEDLES = ("자기 전", "저녁", "밤", "수면", "잠")
_PAIRING_NEEDLES = ("같이", "함께", "병용", "동시에", "한번에", "간격")
_DAILY_NEEDLES = ("매일", "장기", "계속", "오래")
_COMPARISON_NEEDLES = ("중 뭐", "중에 뭐", "뭐가 나아", "뭘 봐야", "비교", "어떤 게 나")
_GOAL_CONTEXT_NEEDLES = ("근육", "근비대", "운동 전", "운동 후", "프리워크아웃", "잠", "수면", "피곤", "집중", "보충제로 뭐", "영양제로 뭐")
_CANONICAL_EQUIVALENTS = {
    "warfarin": {"anticoagulant"},
    "anticoagulant": {"warfarin"},
    "coffee": {"caffeine"},
    "caffeine": {"coffee"},
    "levothyroxine": {"thyroid medication"},
    "thyroid medication": {"levothyroxine"},
    "alcohol": {"alcohol use"},
    "alcohol use": {"alcohol"},
    "probiotics": {"probiotic"},
    "probiotic": {"probiotics"},
    "whey protein": {"protein"},
}


def _expand_canonical(value: str) -> set[str]:
    normalized = (value or "").strip().lower()
    if not normalized:
        return set()
    return {normalized, *_CANONICAL_EQUIVALENTS.get(normalized, set())}


def _query_canonicals(parsed_query: ParsedSupplementQuery, entity_types: set[EntityType] | None = None) -> set[str]:
    values: set[str] = set()
    for entity in parsed_query.entities:
        if entity_types is None or entity.type in entity_types:
            values.update(_expand_canonical(entity.canonical))
    return values


def _list_items(text: str, label: str) -> list[str]:
    items: list[str] = []
    in_block = False
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped == f"{label}:":
            in_block = True
            continue
        if in_block:
            if stripped.startswith("- "):
                items.append(stripped.removeprefix("- ").strip())
                continue
            if stripped and not line.startswith(" "):
                break
    return [item for item in items if item]


def _as_sentence(text: str) -> str:
    stripped = (text or "").strip()
    if stripped and stripped[-1] not in ".!?。":
        stripped = f"{stripped}."
    return stripped


def _profile_doc_for(canonical: str, docs: list[Any]) -> Any | None:
    keys = _expand_canonical(canonical)
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        if (meta.get("source") or meta.get("type")) != "ingredient_profile":
            continue
        doc_canonical = str(meta.get("canonical") or "").strip().lower()
        id_canonical = str(meta.get("id") or "").removeprefix("ING_").lower().replace("_", " ").strip()
        if (doc_canonical and _expand_canonical(doc_canonical) & keys) or (id_canonical and _expand_canonical(id_canonical) & keys):
            return doc
    return None


def _profile_timing(text: str) -> dict[str, str]:
    timing: dict[str, str] = {}
    for item in _list_items(text, "Timing"):
        for label, key in _TIMING_LABEL_KEYS:
            if item.startswith(label):
                value = item.removeprefix(label).strip()
                if value:
                    timing.setdefault(key, value)
                break
        else:
            timing.setdefault("general", item)
    return timing


def _select_timing_lines(question: str, timing: dict[str, str]) -> list[str]:
    keys: list[str] = []
    if any(needle in question for needle in _MEAL_NEEDLES):
        keys.extend(["general", "with_food"])
    if any(needle in question for needle in _EXERCISE_NEEDLES):
        keys.extend(["exercise_related", "general"])
    if any(needle in question for needle in _BEDTIME_NEEDLES):
        keys.extend(["general", "before_bed"])
    if not keys:
        keys.append("general")
    lines = [timing[key] for key in dict.fromkeys(keys) if timing.get(key)]
    return list(dict.fromkeys(lines))[:2]


def _caution_keywords(question: str, parsed_query: ParsedSupplementQuery) -> set[str]:
    keywords: set[str] = set()
    intents = set(parsed_query.intents)
    if any(needle in question for needle in _MEAL_NEEDLES):
        keywords.update({"공복", "위장", "속 불편", "식후"})
    if any(needle in question for needle in _EXERCISE_NEEDLES):
        keywords.update({"운동", "총 단백질", "총량"})
    if IntentType.LONG_TERM_USE in intents or any(needle in question for needle in _DAILY_NEEDLES):
        keywords.update({"중복", "고함량", "고용량", "장기"})
    risk_canonicals = _query_canonicals(parsed_query, {EntityType.RISK_CONTEXT, EntityType.CONDITION, EntityType.DRUG_OR_DRUG_CLASS})
    if "pregnancy" in risk_canonicals:
        keywords.update({"임신", "임산부"})
    if "kidney disease" in risk_canonicals:
        keywords.add("신장")
    if "liver disease" in risk_canonicals:
        keywords.add("간")
    if "antibiotics" in risk_canonicals:
        keywords.add("항생제")
    if "thyroid medication" in risk_canonicals:
        keywords.add("갑상선")
    return keywords


def _rule_entities(doc: Any, label: str, meta_key: str) -> set[str]:
    meta = getattr(doc, "metadata", {}) or {}
    values = _expand_canonical(str(meta.get(meta_key) or ""))
    match = re.search(rf"^{re.escape(label)}:\s*(.+)$", getattr(doc, "page_content", "") or "", re.MULTILINE)
    if match:
        for part in match.group(1).split("/"):
            values.update(_expand_canonical(part))
    return values


def _matched_rule_recommendations(question: str, parsed_query: ParsedSupplementQuery, docs: list[Any]) -> list[str]:
    supplement_canonicals = _query_canonicals(parsed_query, {EntityType.SUPPLEMENT_INGREDIENT})
    all_canonicals = _query_canonicals(parsed_query)
    lowered_question = (question or "").lower()
    recommendations: list[str] = []
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        source = meta.get("source") or meta.get("type")
        text = getattr(doc, "page_content", "") or ""
        if source == "interaction_rule":
            entity_a = _rule_entities(doc, "Entity A", "entity_a_canonical")
            entity_b = _rule_entities(doc, "Entity B", "entity_b_canonical")
            if entity_a & all_canonicals and entity_b & all_canonicals and (entity_a | entity_b) & supplement_canonicals:
                recommendations.append(_field(text, "Recommendation") or "")
        elif source == "safety_rule":
            affected: set[str] = set()
            for value in str(meta.get("affected_entities") or "").split(","):
                affected.update(_expand_canonical(value))
            for value in _list_items(text, "Affected entities"):
                affected.update(_expand_canonical(value))
            triggers = [item.lower() for item in _list_items(text, "Trigger conditions")]
            if affected & supplement_canonicals and any(trigger and trigger in lowered_question for trigger in triggers):
                recommendations.append(_field(text, "Recommendation") or "")
    return [_as_sentence(item) for item in dict.fromkeys(recommendations) if item]


def _grounded_answer_from_entity_docs(question: str, parsed_query: ParsedSupplementQuery, docs: list[Any]) -> str:
    """Build a doc-grounded answer from KB docs that match the detected entities."""
    entities = _supplement_entities(parsed_query)
    if not entities or not docs:
        return ""
    question = question or ""
    rule_lines = _matched_rule_recommendations(question, parsed_query, docs)
    keywords = _caution_keywords(question, parsed_query)
    wants_spacing = any(needle in question for needle in _PAIRING_NEEDLES) or IntentType.LONG_TERM_USE in set(parsed_query.intents)

    profile_lines: list[str] = []
    for entity in entities[:2]:
        profile = _profile_doc_for(entity.canonical, docs)
        if profile is None:
            continue
        text = getattr(profile, "page_content", "") or ""
        lines = _select_timing_lines(question, _profile_timing(text))
        matched_cautions = [item for item in _list_items(text, "Cautions") if keywords and _contains_any(item, keywords)]
        lines.extend(matched_cautions[: 2 if len(lines) < 2 else 1])
        if wants_spacing:
            lines.extend(_list_items(text, "Spacing")[:1])
        lines = [_as_sentence(line) for line in dict.fromkeys(lines) if line]
        if not lines:
            continue
        if len(entities) > 1:
            label = _entity_label(entity.canonical)
            profile_lines.append(f"{label}: " + " ".join(lines))
        else:
            profile_lines.extend(lines)

    if not rule_lines and not profile_lines:
        return ""
    lead: list[str] = []
    if any(needle in question for needle in _COMPARISON_NEEDLES):
        labels = ", ".join(_entity_label(entity.canonical) for entity in entities)
        lead.append(
            f"비교 후보 중 확보된 근거 문서로 확인할 수 있는 것은 {labels}이고, 나머지 후보는 제품 성분표와 함량을 확인한 뒤 비교하는 것이 좋습니다."
        )
    return " ".join([*lead, *rule_lines, *profile_lines][:6]).strip()


def _enhance_answer(
    answer: str,
    question: str,
    parsed_query: ParsedSupplementQuery,
    docs: list[Any],
    *,
    allow_follow_up: bool,
) -> str:
    enhanced = (answer or "").strip()
    if not enhanced:
        enhanced = _direct_answer_from_query_and_docs(parsed_query, docs)
    if _has_unknown_or_partial_known(question, parsed_query) and not _has_product_detail_request(enhanced):
        enhanced = (
            f"{enhanced} 확인되지 않은 제품이나 성분이 함께 있다면 안전하다고 단정하기 어렵습니다. "
            "제품명, 성분표, 함량, 복용량을 알려주면 확인 가능한 성분과 나눠서 볼 수 있습니다."
        ).strip()
    if allow_follow_up and _needs_follow_up(parsed_query, docs) and not _contains_any(enhanced, {"복용 중인 약", "질환", "제품명", "성분표"}):
        enhanced = (
            f"{enhanced} 복용 중인 약, 진단받은 질환, 제품명과 함량을 알려주면 더 안전하게 판단할 수 있습니다."
        ).strip()
    return enhanced


def _direct_answer_from_query_and_docs(parsed_query: ParsedSupplementQuery, docs: list[Any]) -> str:
    entities = _supplement_entities(parsed_query)
    doc_ids = _doc_ids(docs)
    intents = set(parsed_query.intents)
    canonicals = _query_canonicals(parsed_query)

    if IntentType.SYMPTOM_AFTER_INTAKE in intents:
        return (
            "먹고 속이 안 좋다면 계속 복용해도 된다고 단정하기 어렵습니다. 일단 제품명, 성분, 함량, 복용 시점과 증상 지속 시간을 확인해야 하며, "
            "증상이 심하거나 지속되거나 알레르기 의심 증상이 있으면 복용을 중단하고 의료진과 상담하세요."
        )

    # Doc-specific answers fire only when the question carries the co-entity the doc is about;
    # priority retrieval pulls these docs for any mention of the supplement alone.
    if "SAFE_KIDNEY_PROTEIN" in doc_ids and "kidney disease" in canonicals:
        return (
            "신장질환이 있다면 프로틴은 임의로 시작하거나 늘리기보다 하루 총 단백질 섭취량을 의료진과 확인하는 편이 좋습니다. "
            "식사 단백질, 보충제 1회 함량, 신장 기능 상태를 함께 봐야 합니다."
        )

    if len(entities) >= 2:
        checks = _component_checks(entities, doc_ids, docs)
        if checks:
            labels = ", ".join(_entity_label(entity.canonical) for entity in entities)
            return (
                f"{labels} 조합은 성분별로 확인할 점이 다르고, 제품 함량과 개인 상태에 따라 달라질 수 있어 안전하다고 단정하지 않습니다. "
                + " ".join(checks)
                + " 복용 중인 약이나 진단받은 질환이 있으면 우선 확인하세요."
            )

    if "SAFE_PREGNANCY_MULTIVITAMIN" in doc_ids and canonicals & {"pregnancy", "breastfeeding"}:
        return (
            "임산부의 종합비타민 복용은 제품에 따라 달라집니다. 임산부용 제품인지, 비타민A 등 고함량 지용성 비타민과 미네랄 중복이 없는지 확인하고 "
            "의료진 또는 약사와 상담하는 것이 좋습니다."
        )

    if "INT_PROBIOTIC_ANTIBIOTICS" in doc_ids and "antibiotics" in canonicals:
        return (
            "유산균은 항생제와 동시에 먹으면 효과가 줄 수 있어 복용 간격을 확인하는 편이 좋습니다. 항생제 치료를 대체하지 않으며, "
            "처방받은 항생제 종류와 유산균 제품 지침에 따라 의사 또는 약사에게 간격을 확인하세요."
        )

    if "INT_ZINC_IRON_CALCIUM" in doc_ids and "zinc" in canonicals and canonicals & {"iron", "calcium"}:
        return (
            "아연과 철분 또는 칼슘은 미네랄끼리 흡수 경쟁이 생길 수 있어 같은 시간에 고함량으로 몰아 먹는 것은 피하는 편이 좋습니다. "
            "둘 다 필요하면 제품 함량과 복용 목적을 확인해 간격을 두는 방식을 의사 또는 약사에게 확인하세요."
        )

    if IntentType.GOAL_BASED_RECOMMENDATION in intents:
        return _GOAL_RECOMMENDATION_ANSWER

    return ""


_GOAL_RECOMMENDATION_ANSWER = (
    "목적 기반 보충제 추천은 확보된 근거 문서 범위 안에서만 조심스럽게 볼 수 있습니다. 운동 전인지 운동 후인지, 수면 목적이면 카페인 민감도와 "
    "복용 중인 약, 질환, 현재 먹는 제품을 알려주면 안전 범위 안에서 후보를 좁혀볼 수 있습니다."
)


def _should_prefer_deterministic_answer(parsed_query: ParsedSupplementQuery, docs: list[Any]) -> bool:
    entities = _supplement_entities(parsed_query)
    doc_ids = _doc_ids(docs)
    if len(entities) >= 2:
        return True
    return bool(
        doc_ids
        & {
            "INT_PROBIOTIC_ANTIBIOTICS",
            "INT_ZINC_IRON_CALCIUM",
            "SAFE_KIDNEY_PROTEIN",
            "SAFE_PREGNANCY_MULTIVITAMIN",
        }
    )


_COMPONENT_CHECK_TEXT = {
    "multivitamin": "종합비타민은 비타민A, 비타민D, 철분, 아연 등 성분 중복과 고함량 여부를 확인하세요.",
    "omega3": "오메가3는 출혈 위험과 관련해 항응고제 복용이나 예정된 수술·시술이 있는지 확인하세요.",
    "silymarin": "실리마린은 간질환이나 처방약 복용 여부를 확인하세요.",
    "probiotics": "유산균은 항생제와 동시에 복용하면 효과가 줄 수 있어 간격 확인이 필요합니다.",
    "zinc": "아연은 철분, 칼슘, 일부 항생제와 흡수 경쟁이나 간섭 가능성이 있어 복용 간격을 확인하세요.",
    "protein": "프로틴은 식사 단백질까지 합친 하루 총량과 신장질환 여부를 확인하세요.",
    "creatine": "크레아틴은 신장 기능 이상 여부, 충분한 수분 섭취, 제품 권장량을 확인하세요.",
    "magnesium": "마그네슘은 신장질환 여부와 갑상선약·일부 항생제와의 복용 간격을 확인하세요.",
    "iron": "철분은 칼슘, 커피·차, 일부 항생제와 흡수 간섭이 있을 수 있어 복용 간격을 확인하세요.",
    "calcium": "칼슘은 철분, 아연 등 다른 미네랄과 흡수 경쟁이 있을 수 있어 복용 간격을 확인하세요.",
}


def _component_checks(entities: list[Any], doc_ids: set[str], docs: list[Any] | None = None) -> list[str]:
    checks: list[str] = []
    for canonical in dict.fromkeys(entity.canonical for entity in entities):
        text = _COMPONENT_CHECK_TEXT.get(canonical)
        if not text and docs:
            profile = _profile_doc_for(canonical, docs)
            if profile is not None:
                profile_text = getattr(profile, "page_content", "") or ""
                bullet = (_list_items(profile_text, "Spacing") or _list_items(profile_text, "Cautions") or [""])[0]
                if bullet:
                    text = f"{_entity_label(canonical)}: {_as_sentence(bullet)}"
        if text:
            checks.append(text)
    return checks


def _doc_answer_summaries(docs: list[Any]) -> list[str]:
    summaries: list[str] = []
    for doc in docs:
        text = getattr(doc, "page_content", "") or ""
        meta = getattr(doc, "metadata", {}) or {}
        source = meta.get("source") or meta.get("type")
        if source == "interaction_rule":
            summary = _field(text, "Recommendation") or _field(text, "Mechanism")
        elif source == "safety_rule":
            summary = _field(text, "Recommendation") or _field(text, "Risk")
        elif source == "ingredient_profile":
            summary = _field(text, "Name")
            timing = _list_after(text, "Timing")
            if timing:
                summary = f"{summary or '해당 성분'}: {timing}"
        else:
            summary = None
        if summary:
            summaries.append(summary)
        if len(summaries) >= 3:
            break
    return summaries


def _supplement_entities(parsed_query: ParsedSupplementQuery) -> list[Any]:
    return [entity for entity in parsed_query.entities if entity.type == EntityType.SUPPLEMENT_INGREDIENT]


def _doc_ids(docs: list[Any]) -> set[str]:
    return {str((getattr(doc, "metadata", {}) or {}).get("id") or "") for doc in docs}


def _entity_label(canonical: str) -> str:
    return {
        "multivitamin": "종합비타민",
        "omega3": "오메가3",
        "silymarin": "실리마린",
        "probiotics": "유산균",
        "zinc": "아연",
        "iron": "철분",
        "calcium": "칼슘",
        "protein": "프로틴",
        "magnesium": "마그네슘",
        "creatine": "크레아틴",
        "caffeine": "카페인",
        "vitamin a": "비타민A",
        "vitamin c": "비타민C",
        "vitamin d": "비타민D",
        "vitamin e": "비타민E",
        "melatonin": "멜라토닌",
        "vitamin b": "비타민B군",
        "folate": "엽산",
        "biotin": "비오틴",
        "lutein": "루테인",
        "coenzyme q10": "코엔자임Q10",
        "collagen": "콜라겐",
        "arginine": "아르기닌",
        "bcaa": "BCAA",
        "glutamine": "글루타민",
        "selenium": "셀레늄",
        "theanine": "테아닌",
        "ashwagandha": "아쉬와간다",
    }.get(canonical, canonical)


def _has_unknown_or_partial_known(question: str, parsed_query: ParsedSupplementQuery) -> bool:
    if not question:
        return False
    if not parsed_query.entities:
        return True
    known_text = question
    for entity in parsed_query.entities:
        known_text = known_text.replace(entity.text, " ")
    unknown_markers = ("처음 보는", "모르는", "새로 산", "제품", "보충제", "영양제")
    unsupported_ingredient_markers = ("nmn", "nac", "퀘르세틴", "quercetin", "베르베린", "berberine")
    connectors = ("랑", "이랑", "와", "과", "+", ",")
    known_text_lower = known_text.lower()
    has_unknown_marker = any(marker in known_text for marker in unknown_markers) or any(
        marker in known_text_lower for marker in unsupported_ingredient_markers
    )
    return has_unknown_marker and any(connector in question for connector in connectors)


def _has_product_detail_request(text: str) -> bool:
    if not text:
        return False
    if _contains_any(text, {"성분표", "함량", "복용량", "용량"}):
        return True
    return _contains_any(text, {"제품명"}) and _contains_any(text, {"성분", "함량", "복용량", "용량"})


def _needs_follow_up(parsed_query: ParsedSupplementQuery, docs: list[Any]) -> bool:
    intents = set(parsed_query.intents)
    if intents & {
        IntentType.SUPPLEMENT_INTERACTION,
        IntentType.DRUG_INTERACTION,
        IntentType.CONDITION_SAFETY,
        IntentType.PREGNANCY_OR_HIGH_DOSE_SAFETY,
        IntentType.GOAL_BASED_RECOMMENDATION,
        IntentType.SYMPTOM_AFTER_INTAKE,
    }:
        return True
    return not docs


_RECOMMENDATION_SAFE_RE = re.compile(r"(것이|편이|방식이|게)\s*안전합니다")
_ANSWER_HEDGE_TEXT = "다만 개인 상태와 제품 함량에 따라 달라질 수 있어 안전하다고 단정하기는 어렵습니다."
_CONSULT_SENTENCE_RE = re.compile(r"(의사|약사|의료진|전문가)[^.!?]{0,20}(상담|확인)")


def _split_answer_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?。])\s+", (text or "").strip()) if part.strip()]


def _soften_unsafe_answer(answer: str) -> str:
    """Soften unsafe claims sentence-by-sentence; never splice a phrase into the middle of a sentence."""
    text = _RECOMMENDATION_SAFE_RE.sub(lambda match: f"{match.group(1)} 좋습니다", answer or "")
    sentences = _split_answer_sentences(text)
    kept = [sentence for sentence in sentences if not any(phrase in sentence for phrase in _UNSAFE_PHRASES)]
    if len(kept) == len(sentences):
        return text.strip()
    if not kept:
        return ""
    return " ".join([*kept, _ANSWER_HEDGE_TEXT]).strip()


def _is_consult_sentence(sentence: str) -> bool:
    return bool(_CONSULT_SENTENCE_RE.search(sentence or "")) or _contains_any(sentence or "", {"상담하세요", "consult"})


def _is_boilerplate_consult(sentence: str, parsed_query: ParsedSupplementQuery) -> bool:
    """A consult-only sentence that carries no entity-specific content."""
    if not _is_consult_sentence(sentence) or _caution_category(sentence) != "consult":
        return False
    return not any(entity.text and entity.text.lower() in sentence.lower() for entity in parsed_query.entities)


def _normalized_sentence_key(sentence: str) -> str:
    return re.sub(r"[\s.!?。,]", "", sentence or "")


def _dedupe_answer_and_caution(
    answer: str,
    caution: str | None,
    parsed_query: ParsedSupplementQuery,
    *,
    risk_query: bool = False,
) -> tuple[str, str | None]:
    """Keep at most one generic consult sentence across answer + caution and drop caution
    sentences that merely repeat the answer."""
    answer_sentences = _split_answer_sentences(answer)
    caution_sentences = _split_answer_sentences(caution or "")
    caution_has_consult = any(_is_consult_sentence(sentence) for sentence in caution_sentences)

    kept_answer: list[str] = []
    seen_answer: set[str] = set()
    consult_seen = False
    for sentence in answer_sentences:
        key = _normalized_sentence_key(sentence)
        if key in seen_answer:
            continue
        if _is_boilerplate_consult(sentence, parsed_query):
            if consult_seen or caution_has_consult:
                continue
            consult_seen = True
        seen_answer.add(key)
        kept_answer.append(sentence)
    if not kept_answer:
        kept_answer = answer_sentences[:1]

    kept_caution: list[str] = []
    seen_caution: set[str] = set()
    caution_consult_seen = False
    for sentence in caution_sentences:
        key = _normalized_sentence_key(sentence)
        # Risk cautions keep their safety sentences even when the answer already states them.
        if key in seen_caution or (key in seen_answer and not risk_query):
            continue
        if _is_boilerplate_consult(sentence, parsed_query):
            if caution_consult_seen:
                continue
            caution_consult_seen = True
        seen_caution.add(key)
        kept_caution.append(sentence)
    if caution_sentences and not kept_caution:
        # High-risk cautions must stay non-empty; keep the first original sentence.
        kept_caution = caution_sentences[:1]

    deduped_answer = " ".join(kept_answer).strip() if answer_sentences else (answer or "")
    deduped_caution = " ".join(kept_caution).strip() if caution_sentences else caution
    return deduped_answer, deduped_caution or None


def _soften_recommendation_phrasing(text: str) -> str:
    return _RECOMMENDATION_SAFE_RE.sub(lambda match: f"{match.group(1)} 좋습니다", text or "")


def _soften_unsafe_caution(caution: str) -> str:
    sentences = _split_caution_sentences([_soften_recommendation_phrasing(caution)])
    softened_sentences: list[str] = []
    for sentence in sentences:
        if not any(phrase in sentence for phrase in _UNSAFE_PHRASES):
            softened_sentences.append(sentence)
            continue
        category = _caution_category(sentence)
        if category == "spacing":
            softened_sentences.append(
                "\uac11\uc0c1\uc120\uc57d\uc774\ub098 \uc77c\ubd80 \ud56d\uc0dd\uc81c\ub97c \ubcf5\uc6a9 \uc911\uc774\uba74 \ud761\uc218\uc5d0 \uc601\ud5a5\uc744 \uc904 \uc218 \uc788\uc73c\ubbc0\ub85c \ubcf5\uc6a9 \uac04\uaca9\uc744 \uc758\uc0ac \ub610\ub294 \uc57d\uc0ac\uc5d0\uac8c \ud655\uc778\ud558\uc138\uc694."
            )
        elif category in {"anticoagulant", "bleeding"}:
            softened_sentences.append(
                "\ud56d\uc751\uace0\uc81c\ub098 \uc640\ud30c\ub9b0\uc744 \ubcf5\uc6a9 \uc911\uc774\uba74 \ucd9c\ud608 \uc704\ud5d8\uc774 \ub2ec\ub77c\uc9c8 \uc218 \uc788\uc73c\ubbc0\ub85c \ubcf5\uc6a9 \uc804 \uc758\uc0ac \ub610\ub294 \uc57d\uc0ac\uc640 \uc0c1\ub2f4\ud558\uc138\uc694."
            )
        else:
            softened_sentences.append(_COMPACT_CONSULT_TEXT)
    return _truncate_by_sentence(" ".join(_dedupe_caution_sentences(softened_sentences)).strip(), _MAX_CAUTION_CHARS)


def _is_weak_answer(answer: str | None) -> bool:
    if not isinstance(answer, str):
        return True
    normalized = answer.strip()
    if normalized.lower() in _UNUSABLE_ANSWERS:
        return True
    return any(needle in normalized for needle in _WEAK_ANSWER_NEEDLES)


def _field(text: str, label: str) -> str | None:
    match = re.search(rf"^{re.escape(label)}:\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def _list_after(text: str, label: str) -> str | None:
    lines = text.splitlines()
    collected: list[str] = []
    in_block = False
    for line in lines:
        stripped = line.strip()
        if stripped == f"{label}:":
            in_block = True
            continue
        if in_block:
            if stripped.startswith("- "):
                collected.append(stripped.removeprefix("- ").strip())
                continue
            if stripped and not line.startswith(" "):
                break
    return ". ".join(collected).strip() or None


def _dedupe_join(values: list[Any], max_chars: int) -> str:
    seen: set[str] = set()
    parts: list[str] = []
    for value in values:
        text = _string_or_none(value)
        if not text:
            continue
        normalized = " ".join(text.split())
        if normalized in seen:
            continue
        seen.add(normalized)
        parts.append(normalized)
    joined = " ".join(parts).strip()
    if len(joined) <= max_chars:
        return joined
    return joined[:max_chars].rsplit(" ", 1)[0].strip()


def _string_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _dict_get(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    lowered = {str(key).lower(): value for key, value in data.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value is not None:
            return value
    return None


# ---------- Efficacy questions without KB evidence (policy 2026-11-11) ----------
# "마그네슘 수면에 도움 돼요?" / "비타민D 먹으면 골다공증 예방되나요?": the curated KB lists what an
# ingredient is used for, not whether it achieves a health outcome. The answer must say that
# first; anything after it is general reference information, not a documented effect.
_EFFICACY_NEEDLES = (
    "도움 돼", "도움돼", "도움 되", "도움되", "효과", "좋아지", "좋아져", "예방", "치료돼", "치료되",
    "치료 효과", "치료할", "치료해", "치료 되", "치료 돼", "완치",
    "낫나", "낫는", "나아지", "나아져", "나아요", "개선", "커지", "커져", "보호", "안 걸리", "잘 와", "회복", "빠지", "빠져",
    "늘어", "없어지", "없어져", "낮출", "낮춰", "낮아지", "낮아져", "줄어", "고쳐", "잘 타", "태워", "다이어트 돼", "다이어트 되",
)
_GENERIC_EFFICACY_NEEDLES = frozenset({"도움 돼", "도움돼", "도움 되", "도움되", "효과"})
_EFFICACY_STOPWORDS = frozenset({
    "먹으면", "먹고", "먹어도", "되나요", "돼요", "되죠", "되요", "하나요", "있나요", "정말", "확실히",
    "무조건", "바로", "많이", "계속", "매일", "같이", "그냥", "진짜", "효과", "도움",
})
_HONEST_LEAD_MARKERS = ("근거 문서에서", "근거 문서에는", "확인되지 않", "확인할 수 없")


def _josa(word: str, with_final: str, without_final: str) -> str:
    last = word[-1] if word else ""
    if "가" <= last <= "힣":
        return with_final if (ord(last) - 0xAC00) % 28 else without_final
    return without_final


# Outcome predicates ("주름 펴져요", "톤이 밝아지나요", "코로나 안 걸린다던데"): a claimed health
# result, whatever the verb. Used only when the question is not a timing / pairing question.
_OUTCOME_RE = re.compile(
    r"(?:아|어|여|펴|커|빠|꺼|쳐|켜|낮아|높아|밝아|좋아|나아|없어|줄어|늘어)\s*(?:지|져)"
    r"|안\s*걸|걸리지\s*않|막아\s*주|막아줘|막을|낮춰|높여|올려\s*주|줄여\s*주|생기지\s*않|된다던데|다던데"
    r"|멈추|멈출|키\s*(?:큼|크|커)|올라가|덜\s*피곤|(?:아|어)진다|좋다고|좋대|효과\s*있|내려가|내려감|내려갈"
    r"|\bhelps?\b|\bboost|\bprevent|\bcure|stay awake"
)


def _is_efficacy_question(question: str) -> bool:
    q = question or ""
    return any(needle in q for needle in _EFFICACY_NEEDLES) or bool(_OUTCOME_RE.search(q))


def _claim_words(question: str, entity_texts: set[str]) -> list[str]:
    words = re.findall(r"[가-힣]{2,}", question or "")
    out = []
    for word in words:
        if word in _EFFICACY_STOPWORDS or any(word in text or text in word for text in entity_texts if text):
            continue  # the ingredient's own name ("비타민" of 비타민D) is not a claimed effect
        if any(word.startswith(n) or n.startswith(word) for n in _EFFICACY_NEEDLES):
            continue
        out.append(re.sub(r"(에|에는|을|를|이|가|은|는|도|으로|로)$", "", word))
    return [w for w in out if len(w) >= 1]


def _apply_efficacy_policy(answer: str, question: str, parsed_query: ParsedSupplementQuery, docs: list[Any]) -> str:
    if not answer or not _is_efficacy_question(question):
        return answer
    # "공복에 먹어도 효과 있어요?", "커피랑 먹으면 효과 없어지나요?": timing / pairing questions that
    # happen to say 효과 are answered by the timing and interaction text, not as efficacy claims.
    # Only when 효과 / 도움 is the sole outcome word: "밤에 다리 쥐 나는 거 없어지나요" still claims one.
    # "커피랑 먹으면 효과 없어지나요" asks whether the supplement still works (a pairing question)
    stripped = re.sub(r"효과\s*(?:가|이|는)?\s*(?:없어지|없어져|떨어지|떨어져|줄어)", "효과", question or "")
    hits = [n for n in _EFFICACY_NEEDLES if n in stripped]
    # an outcome predicate alone counts as generic: pairing questions say "흡수가 떨어지나요"
    generic_only = all(n in _GENERIC_EFFICACY_NEEDLES for n in hits)
    if generic_only and set(parsed_query.intents) & {IntentType.TIMING, IntentType.SUPPLEMENT_INTERACTION,
                                                     IntentType.DRUG_INTERACTION, IntentType.FOOD_COMPOUND_INTERACTION}:
        return answer
    entities = _supplement_entities(parsed_query)
    if not entities:
        return answer  # out-of-KB ingredient: the unknown-ingredient template already says so
    if any(marker in answer for marker in _HONEST_LEAD_MARKERS):
        return answer
    entity = entities[0]
    label = _entity_label(entity.canonical)
    entity_texts = {getattr(e, "text", "") for e in parsed_query.entities}
    profile = _profile_doc_for(entity.canonical, docs)
    uses = _list_items(getattr(profile, "page_content", "") or "", "Common uses") if profile is not None else []
    claims = _claim_words(question, entity_texts)
    documented = [use for use in uses if any(claim and claim in use for claim in claims)]
    if documented:
        lead = (
            f"근거 문서에는 {label}{_josa(label, '이', '가')} '{documented[0]}' 용도로 안내되어 있지만, "
            "질문하신 효과를 직접 확인한 근거가 없습니다."
        )
    else:
        lead = f"확보된 근거 문서에서는 {label}의 그런 효과를 확인할 수 없습니다."
    return f"{lead} 아래 내용은 일반적인 참고 정보이며 개인에 따라 다를 수 있습니다. {answer}"


# ---------- Say what the KB does not cover (policy 2026-11-11) ----------
# An ingredient without a profile (비타민A/E), a risk context with no rule for that ingredient
# (밀크씨슬 + 임신), or a symptom the profile never mentions (유산균 + 더부룩): the LLM answers from
# general knowledge, so the answer must first say the curated documents do not cover it.
# Common supplements with no curated profile or rule (the parser does not know them either).
_UNCOVERED_SUPPLEMENT_NAMES = (
    "가르시니아", "글루코사민", "콘드로이친", "홍삼", "인삼", "프로폴리스", "쏘팔메토", "크릴오일", "스피루리나",
    "클로렐라", "히알루론산", "보스웰리아", "녹용", "흑마늘", "커큐민", "강황", "은행잎", "달맞이꽃",
    "감마리놀렌산", "녹차추출물", "카테킨", "키토산", "공액리놀레산", "알로에", "레시틴", "비타민k", "비타민 k",
    "세인트존스워트", "nac", "엔에이씨", "아세틸시스테인", "퀘르세틴", "nmn", "베르베린", "크롬", "요오드",
    "마카", "쏘팔", "타트체리", "엘더베리", "에키네시아", "비타민 젤리", "젤리 비타민", "빌베리", "여주",
)
_CONTEXT_WORDS = {
    "pregnancy": ("임신", "임산부"),
    "breastfeeding": ("수유",),
    "surgery": ("수술", "시술"),
    "child": ("어린이", "소아", "영유아", "유아", "청소년", "아이"),
    "immunocompromised": ("면역저하", "면역 저하", "면역력이 떨어", "항암", "이식", "중증"),
}
_CONTEXT_LABEL = {"pregnancy": "임신 중", "breastfeeding": "수유 중", "surgery": "수술 전후",
                  "child": "어린이·청소년", "immunocompromised": "항암·면역저하 상태"}
_TOPIC_WORDS = ("보관", "냉장", "실온", "유통기한", "상한", "산패")
_TOPICS = (
    (_TOPIC_WORDS, "보관 방법"),
    (("동물성", "식물성", "비건", "채식", "원료", "유래", "vegan", "wool", "gelatin", "젤라틴"), "원료·유래"),
)
_CONDITION_WORDS = {
    "kidney disease": ("신장", "투석", "신부전", "콩팥"),
    "liver disease": ("간 질환", "간질환", "간 기능", "간수치", "간"),
    "hypertension": ("혈압", "고혈압"),
}
_SYMPTOM_WORDS = ("가스", "더부룩", "메스꺼", "설사", "변비", "두드러기", "속쓰림", "복통", "트림", "두통", "어지러")


@functools.lru_cache(maxsize=1)
def _curated_kb() -> tuple[tuple[str, dict, str], ...]:
    """All curated profiles / rules (data/documents JSON) as (text, metadata, source)."""
    from pathlib import Path

    from engines.supplement.kb_documents import load_curated_kb_records

    root = Path(__file__).resolve().parents[2] / "data" / "documents"
    return tuple(load_curated_kb_records(root))


def _kb_text_for(canonical: str, sources: set[str]) -> str:
    """Curated KB text about this ingredient: the whole KB, not only what retrieval returned."""
    return "\n".join(
        text for text, meta, source in _curated_kb()
        if source in sources and canonical in _doc_supplement_canonicals(meta)
    )


def _doc_entity_canonicals(meta: dict) -> set[str]:
    """Every entity a rule is about (supplements and drugs / foods), expanded."""
    values: set[str] = set()
    for key in ("canonical", "entity_a_canonical", "entity_b_canonical", "affected_entities"):
        raw = meta.get(key)
        parts = raw.split(",") if isinstance(raw, str) else (raw or [])
        for part in parts:
            values.update(_expand_canonical(str(part)))
    return values


_UNSPECIFIC_MEDICATION_TEXTS = {"처방약", "복용 중인 약", "약물", "medication", "medicine", "처방 받", "처방받"}


def _uncovered_pair_lead(entity: Any, label: str, parsed_query: ParsedSupplementQuery) -> str | None:
    """"아연이랑 마그네슘 같이", "당뇨약 먹는데 프로틴": a combination question for which no curated
    interaction / safety rule names both sides. The LLM then answers from general knowledge."""
    if not set(parsed_query.intents) & {IntentType.SUPPLEMENT_INTERACTION, IntentType.DRUG_INTERACTION}:
        return None
    canonical = entity.canonical.lower()
    others = [
        e for e in parsed_query.entities
        if e is not entity and e.canonical.lower() != canonical and e.text != entity.text  # "칼마디" names both
        and e.type in (EntityType.SUPPLEMENT_INGREDIENT, EntityType.DRUG_OR_DRUG_CLASS)
        # "복용 중인 약" names no specific pair; "고지혈증약", "녹내장 안약" do
        and not (e.canonical.lower() == "medication" and e.text in _UNSPECIFIC_MEDICATION_TEXTS)
    ]
    for other in others:
        other_forms = _expand_canonical(other.canonical.lower())
        other_names = {other.text, _entity_label(other.canonical)} - {"", None}
        covered = any(
            canonical in _doc_supplement_canonicals(meta)
            and (
                # a rule naming both sides, or a profile that discusses the pairing
                (source in {"interaction_rule", "safety_rule"} and other_forms & _doc_entity_canonicals(meta))
                or (source in {"interaction_rule", "safety_rule", "ingredient_profile"}
                    and any(name in text for name in other_names))
            )
            for text, meta, source in _curated_kb()
        )
        if not covered:
            other_label = other.text or other.canonical
            return (
                f"확보된 근거 문서에는 {label}{_josa(label, '과', '와')} {other_label}의 병용에 대한 자료가 없어, "
                "아래 내용은 각 성분 자료와 일반적인 참고 정보입니다."
            )
    return None


def _apply_coverage_honesty(answer: str, question: str, parsed_query: ParsedSupplementQuery, docs: list[Any]) -> str:
    if not answer or any(marker in answer for marker in _HONEST_LEAD_MARKERS):
        return answer
    entities = [e for e in parsed_query.entities if e.type == EntityType.SUPPLEMENT_INGREDIENT]
    if not entities:
        named = next((name for name in _UNCOVERED_SUPPLEMENT_NAMES if name in (question or "").lower()), None)
        if named:
            return f"확보된 근거 문서에는 {named}에 대한 자료가 없어, 아래 내용은 일반적인 참고 정보입니다. {answer}"
        return answer
    entity = entities[0]
    label = _entity_label(entity.canonical)
    canonical = entity.canonical.lower()
    if not _curated_kb():
        return answer  # KB files unavailable: nothing to judge coverage against
    profile_text = _kb_text_for(canonical, {"ingredient_profile"})
    rule_text = _kb_text_for(canonical, {"safety_rule", "interaction_rule"})
    if not profile_text and not rule_text:
        lead = f"확보된 근거 문서에는 {label}에 대한 자료가 없어, 아래 내용은 일반적인 참고 정보입니다."
        return f"{lead} {answer}"
    covered_text = profile_text + "\n" + rule_text
    pair_lead = _uncovered_pair_lead(entity, label, parsed_query)
    if pair_lead:
        return f"{pair_lead} {answer}"
    contexts = [e.canonical.lower() for e in parsed_query.entities if e.canonical.lower() in _CONTEXT_WORDS]
    for context in contexts:
        if not any(word in covered_text for word in _CONTEXT_WORDS[context]):
            lead = (
                f"확보된 근거 문서에는 {label}의 {_CONTEXT_LABEL[context]} 안전성에 대한 구체적인 자료가 없습니다. "
                "복용 전 반드시 의사 또는 약사와 상담하세요."
            )
            return f"{lead} {answer}"
    for topic_words, topic_label in _TOPICS:
        asked_topics = [w for w in topic_words if w in (question or "")]
        if asked_topics and not any(w in covered_text for w in asked_topics):
            # "냉장 보관해야 돼요?", "비건인데 동물성 성분 들었나요?": topics the KB does not cover
            return (f"확보된 근거 문서에는 {label}의 {topic_label}에 대한 자료가 없어, 아래 내용은 일반적인 참고 정보입니다. "
                    f"{answer}")
    for entity_ in parsed_query.entities:
        words = _CONDITION_WORDS.get(entity_.canonical.lower())
        if entity_.type == EntityType.CONDITION and words and not any(w in covered_text for w in words):
            # "투석 중인데 셀레늄 괜찮나요?": the condition is not in any curated text for this ingredient
            return (f"확보된 근거 문서에는 {label}의 {words[0]} 관련 안전성 자료가 없습니다. "
                    f"복용 전 반드시 의사 또는 약사와 상담하세요. {answer}")
    asked_symptoms = [w for w in _SYMPTOM_WORDS if w in (question or "")]
    if asked_symptoms and not any(w in covered_text for w in asked_symptoms):
        lead = (
            f"확보된 근거 문서에서는 {label} 복용 후 이런 증상에 대한 자료를 확인할 수 없습니다. "
            "증상이 계속되면 복용을 중단하고 의사 또는 약사와 상담하세요."
        )
        return f"{lead} {answer}"
    return answer


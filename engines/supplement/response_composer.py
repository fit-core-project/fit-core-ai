from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from engines.supplement.query_understanding import IntentType, ParsedSupplementQuery


_RISK_INTENTS = {
    IntentType.DRUG_INTERACTION,
    IntentType.FOOD_COMPOUND_INTERACTION,
    IntentType.SUPPLEMENT_INTERACTION,
    IntentType.CONDITION_SAFETY,
    IntentType.PREGNANCY_OR_HIGH_DOSE_SAFETY,
    IntentType.MEDICATION_SAFETY,
}
_UNUSABLE_ANSWERS = {"", "{}", "[]", "null", "none"}
_WEAK_ANSWER_NEEDLES = (
    "정보가 부족",
    "자료 없음",
    "답변할 수 없습니다",
    "WEB_SEARCH_REQUIRED",
)
_UNSAFE_PHRASES = (
    "먹어도 됩니다",
    "문제 없습니다",
    "안전합니다",
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
) -> ComposedSupplementResponse:
    answer, parsed_caution = _normalize_answer(generated_answer)
    caution_parts = [generated_caution, parsed_caution]
    caution_parts.extend(_caution_parts_from_docs(selected_docs))

    used_kb_fallback = False
    if _is_weak_answer(answer):
        answer = _fallback_answer_from_docs(selected_docs, parsed_query)
        used_kb_fallback = True

    risk_query = bool(set(parsed_query.intents) & _RISK_INTENTS)
    if risk_query:
        answer = _soften_unsafe_answer(answer)
        caution_parts.append(_CONSULT_TEXT)

    caution = _compact_caution(caution_parts, parsed_query, selected_docs, risk_query=risk_query)
    if caution:
        caution = _soften_unsafe_caution(caution)
    if risk_query and not caution:
        caution = _COMPACT_CONSULT_TEXT

    return ComposedSupplementResponse(
        answer=answer or _fallback_answer_from_docs(selected_docs, parsed_query),
        caution=caution or None,
        used_kb_fallback=used_kb_fallback,
    )


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


def _caution_parts_from_docs(docs: list[Any]) -> list[str]:
    parts: list[str] = []
    for doc in docs:
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


def _fallback_answer_from_docs(docs: list[Any], parsed_query: ParsedSupplementQuery | None = None) -> str:
    source_types = {(getattr(doc, "metadata", {}) or {}).get("source") for doc in docs}
    intents = set(parsed_query.intents) if parsed_query else set()
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
    return "검색된 근거만으로는 충분하지 않습니다. 복용 중인 약이나 질환이 있다면 의사 또는 약사와 상담하세요."


def _soften_unsafe_answer(answer: str) -> str:
    softened = answer or ""
    found = False
    for phrase in _UNSAFE_PHRASES:
        if phrase in softened:
            softened = softened.replace(phrase, "개인 상태에 따라 달라질 수 있습니다")
            found = True
    if found and _CONSULT_TEXT not in softened:
        softened = f"{softened.strip()} {_CONSULT_TEXT}"
    return softened


def _soften_unsafe_caution(caution: str) -> str:
    sentences = _split_caution_sentences([caution])
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
    return " ".join(collected).strip() or None


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

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from engines.supplement.query_understanding import (
    EntityType,
    IntentType,
    ParsedSupplementQuery,
    entity_coverage_status,
    entity_label,
)


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
    "자료가 부족",
    "자료 없음",
    "답변할 수 없습니다",
    "WEB_SEARCH_REQUIRED",
    "의학적 조언을 제공할 수 없습니다",
    "의료 조언을 제공할 수 없습니다",
    "전문적인 의학적 조언",
    "진단할 수 없습니다",
    "처방할 수 없습니다",
)
_UNSAFE_PHRASES = (
    "먹어도 됩니다",
    "복용해도 됩니다",
    "계속 먹어도 됩니다",
    "문제 없습니다",
    "안전합니다",
    "걱정하지 않아도 됩니다",
    "무조건 괜찮습니다",
)
_CONSULT_TEXT = "개인 상태, 복용 중인 약, 용량에 따라 달라질 수 있으므로 의사 또는 약사와 상담하세요."


_COMPACT_CONSULT_TEXT = "개인 상태, 복용 중인 약, 용량에 따라 달라질 수 있으므로 복용 전 의사 또는 약사와 상담하세요."
_CONSULT_MARKERS = ("상담", "문의", "의료진", "의사", "약사", "전문가")
_ACTIONABLE_MARKERS = (
    "간격",
    "흡수",
    "출혈",
    "신장",
    "간질환",
    "고용량",
    "임신",
    "수유",
    "항생제",
    "와파린",
    "항응고제",
    "성분",
    "함량",
    "식후",
    "공복",
    "운동 전",
    "운동 후",
    "권장량",
    "복용량",
    "총량",
    "수분",
    "중복",
    "증상",
    "중단",
    "제품",
    "프로틴",
    "크레아틴",
    "오메가3",
    "마그네슘",
    "철분",
    "아연",
    "칼슘",
    "유산균",
)
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
    caution_parts.extend(_caution_parts_from_docs(selected_docs, parsed_query, answer))

    used_kb_fallback = False
    preferred_direct_answer = (
        _direct_answer_from_query_and_docs(parsed_query, selected_docs)
        if _should_prefer_deterministic_answer(parsed_query, selected_docs)
        else ""
    )
    if _is_weak_answer(answer):
        answer = preferred_direct_answer or _fallback_answer_from_docs(selected_docs, parsed_query)
        used_kb_fallback = True
    elif preferred_direct_answer and _should_replace_with_direct_answer(answer, parsed_query):
        answer = preferred_direct_answer
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

    answer = _enhance_answer(
        answer or _fallback_answer_from_docs(selected_docs, parsed_query),
        question,
        parsed_query,
        selected_docs,
        allow_follow_up=used_kb_fallback,
    )

    return ComposedSupplementResponse(
        answer=answer,
        caution=caution or None,
        used_kb_fallback=used_kb_fallback,
    )


def _should_replace_with_direct_answer(answer: str, parsed_query: ParsedSupplementQuery) -> bool:
    if any(phrase in (answer or "") for phrase in _UNSAFE_PHRASES):
        return True
    if _is_low_specificity_answer(answer):
        return True
    if _answer_misses_query_entities(answer, parsed_query):
        return True
    return False


def _answer_misses_query_entities(answer: str, parsed_query: ParsedSupplementQuery) -> bool:
    if not parsed_query.entities:
        return False
    normalized_answer = _normalize_match_text(answer)
    missing: list[str] = []
    for entity in parsed_query.entities:
        candidates = {
            entity.canonical,
            entity.text,
            _entity_label(entity.canonical),
        }
        if entity_coverage_status(entity.canonical) == "unsupported":
            candidates.add("근거 DB")
            candidates.add("성분표")
        if not any(_normalize_match_text(candidate) in normalized_answer for candidate in candidates if candidate):
            missing.append(entity.canonical)
    if not missing:
        return False

    intents = set(parsed_query.intents)
    if any(entity_coverage_status(entity.canonical) == "unsupported" for entity in parsed_query.entities):
        return True
    if intents & {IntentType.DRUG_INTERACTION, IntentType.CONDITION_SAFETY, IntentType.SUPPLEMENT_INTERACTION}:
        return True
    return False


def _normalize_match_text(value: str | None) -> str:
    return re.sub(r"[-_\\s]+", "", str(value or "").lower())


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


def _doc_is_relevant_for_caution(doc: Any, parsed_query: ParsedSupplementQuery, answer: str) -> bool:
    meta = getattr(doc, "metadata", {}) or {}
    source = meta.get("source") or meta.get("type")
    if source not in {"interaction_rule", "safety_rule", "ingredient_profile"}:
        return False

    query_values = {entity.canonical.lower() for entity in parsed_query.entities}
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

    if query_values and doc_values & query_values:
        return True
    doc_text = (getattr(doc, "page_content", "") or "").lower()
    if query_values and any(value and value in doc_text for value in query_values):
        return True
    return any(value and value in answer_text for value in doc_values)


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
    if parsed_query:
        direct = _direct_answer_from_query_and_docs(parsed_query, docs)
        if direct:
            return direct
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
        summary = _source_summary(docs, "safety_rule")
        if summary:
            return (
                f"검색된 안전 규칙에 따르면 {summary} "
                "개인 질환, 임신/수유, 수술 전후, 고용량 복용 같은 조건에서는 임의로 복용을 결정하지 않는 것이 좋습니다."
            )
        return (
            "해당 조건에서는 보충제나 약 복용을 임의로 결정하지 않는 것이 좋습니다. 검색된 안전 규칙에 따르면 질환, "
            "임신/수유, 수술 전후, 고용량 복용 같은 조건에서는 전문가 상담이 필요할 수 있습니다."
        )
    if "interaction_rule" in source_types:
        summary = _source_summary(docs, "interaction_rule")
        if summary:
            return (
                f"검색된 상호작용 규칙에 따르면 {summary} "
                "복용 간격만으로 해결되는 문제인지, 병용 자체를 피해야 하는 상황인지는 복용 중인 약과 용량에 따라 확인이 필요합니다."
            )
        return (
            "해당 조합은 함께 복용할 때 주의가 필요합니다. 검색된 상호작용 규칙에 따르면 복용 간격, 약물 병용, "
            "출혈 위험 등 확인이 필요한 요소가 있을 수 있으므로 임의로 병용하지 말고 의사 또는 약사에게 안전성을 확인하는 것이 좋습니다."
        )
    if "safety_rule" in source_types:
        summary = _source_summary(docs, "safety_rule")
        if summary:
            return (
                f"검색된 안전 규칙에 따르면 {summary} "
                "개인 질환, 복용 중인 약, 임신/수유, 수술 전후 여부에 따라 판단이 달라질 수 있습니다."
            )
        return (
            "해당 조건에서는 보충제나 약 복용을 임의로 결정하지 않는 것이 좋습니다. 검색된 안전 규칙에 따르면 질환, "
            "임신/수유, 수술 전후, 고용량 복용 같은 조건에서는 전문가 상담이 필요할 수 있습니다."
        )
    if "ingredient_profile" in source_types:
        summary = _source_summary(docs, "ingredient_profile")
        if summary:
            return (
                f"검색된 성분 프로필 기준으로는 {summary} "
                "식사 여부, 운동 전후, 다른 약물과의 간격, 개인 질환 여부를 함께 확인하는 것이 좋습니다."
            )
        return (
            "일반적인 복용 타이밍은 성분과 목적에 따라 달라질 수 있습니다. 검색된 성분 프로필의 복용 가이드와 "
            "주의사항을 기준으로 식사 여부, 운동 전후, 다른 약물과의 간격을 함께 확인하는 것이 좋습니다."
        )
    return "검색된 근거만으로는 충분하지 않습니다. 확인 가능한 제품명, 성분표, 함량, 복용 중인 약과 질환 정보를 알려주면 범위를 좁혀 안내할 수 있습니다."


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
    canonicals = {entity.canonical for entity in entities}
    all_canonicals = {entity.canonical for entity in parsed_query.entities}

    unsupported_entities = [
        entity for entity in parsed_query.entities if entity_coverage_status(entity.canonical) == "unsupported"
    ]
    if unsupported_entities:
        labels = ", ".join(_entity_label(entity.canonical) for entity in unsupported_entities)
        known_entities = [entity for entity in entities if entity_coverage_status(entity.canonical) != "unsupported"]
        known_text = ""
        if known_entities:
            known_labels = ", ".join(_entity_label(entity.canonical) for entity in known_entities)
            checks = _component_checks(known_entities, doc_ids)
            known_text = f" 확인된 다른 성분은 {known_labels}입니다. {' '.join(checks)}"
        return (
            f"{labels}은 현재 Fit-Core 근거 DB에서 안전성/상호작용 근거가 충분히 잠기지 않은 성분입니다. "
            "안전하다고 단정하지 않고, 제품명, 성분표, 함량, 복용량, 복용 중인 약과 질환 정보를 확인한 뒤 "
            f"의사 또는 약사에게 상담하는 쪽으로 안내해야 합니다.{known_text}"
        )

    dose_answer = _dose_or_label_context_answer(canonicals, all_canonicals, parsed_query.raw_query)
    if dose_answer:
        return dose_answer

    if len(entities) == 1 and IntentType.TIMING in intents:
        timing_answer = _single_ingredient_timing_answer(entities[0].canonical, docs)
        if timing_answer:
            return timing_answer

    if IntentType.FOOD_COMPOUND_INTERACTION in intents and "INT_IRON_CAFFEINE" in doc_ids:
        return (
            "철분은 커피나 카페인 음료와 동시에 먹으면 흡수에 방해가 될 수 있습니다. "
            "철분 보충제를 복용한다면 커피와는 시간 간격을 두는 편이 안전하며, 빈혈 치료나 고용량 철분 복용 중이면 "
            "구체적인 간격을 의사 또는 약사에게 확인하세요."
        )

    if IntentType.SIDE_EFFECT in intents and "creatine" in canonicals:
        return (
            "크레아틴은 사람에 따라 위장 불편, 체중 증가로 느껴지는 수분 저류, 복용량 관련 불편이 생길 수 있습니다. "
            "신장질환이 있거나 신장 기능을 추적 중이면 임의로 시작하지 말고 의료진과 상담하세요."
        )

    if "omega3" in canonicals and {"SAFE_OMEGA3_SURGERY", "INT_OMEGA3_ANTICOAGULANTS"} & doc_ids:
        return (
            "검색된 상호작용 규칙에 따르면 오메가3는 와파린 같은 항응고제나 항혈소판제 복용, 출혈 위험, 수술·시술 예정 여부가 있으면 먼저 확인해야 합니다. "
            "일반적인 보충 목적이라도 고용량으로 늘리기 전에는 제품 함량과 복용 중인 약을 함께 점검하세요."
        )

    if "omega3" in canonicals and "aspirin" in all_canonicals:
        return (
            "아스피린은 항혈소판제 맥락에 해당하므로 오메가3와 함께 볼 때 출혈 위험, 고용량 여부, 수술·시술 예정 여부를 확인해야 합니다. "
            "복용 시간을 띄우는 것만으로 안전하다고 단정하지 말고, 오메가3 제품 함량과 아스피린 복용 목적을 의사 또는 약사에게 확인하세요."
        )

    if {"omega3", "vitamin e"} <= canonicals:
        return (
            "오메가3와 비타민E는 일반 용량인지, 고용량인지, 항응고제·항혈소판제 복용이나 수술 예정이 있는지에 따라 판단이 달라집니다. "
            "둘을 함께 먹어도 된다고 단정하지 말고 제품 함량과 출혈 위험 맥락을 먼저 확인하세요."
        )

    if {"vitamin c", "iron"} <= canonicals:
        return (
            "비타민C는 철분 흡수에 도움이 될 수 있는 조합입니다. 다만 철분은 속 불편, 변비, 고용량 복용 문제가 있을 수 있고 "
            "커피·칼슘·마그네슘·아연 같은 다른 성분과는 간격 확인이 필요하므로, 치료 목적 철분이면 제품 함량과 복용법을 의사 또는 약사에게 확인하세요."
        )

    if "magnesium" in canonicals and "blood pressure medication" in all_canonicals:
        return (
            "혈압약을 복용 중이면 마그네슘을 안전하다고 단정하기 어렵습니다. 혈압약 종류, 용량, 혈압 조절 상태, "
            "마그네슘 제품 함량에 따라 판단이 달라지므로 의사 또는 약사에게 확인하세요."
        )

    if "vitamin d" in canonicals and "diabetes medication" in all_canonicals:
        return (
            "당뇨약을 복용 중이면 비타민D를 임의로 고용량 추가하지 말고 약 이름, 혈당 조절 상태, 비타민D 용량과 "
            "검사 수치를 함께 확인해야 합니다. 복용 간격보다 용량과 개인 상태 확인이 우선입니다."
        )

    if "vitamin a" in canonicals and "liver disease" in all_canonicals:
        return (
            "간질환이나 간수치 이상이 있으면 비타민A, 특히 고함량 preformed vitamin A 제품은 임의로 복용하지 않는 편이 안전합니다. "
            "종합비타민, 눈 건강 제품, 간유 제품 등과 비타민A가 중복될 수 있으니 제품 라벨과 총량을 의료진 또는 약사와 확인하세요."
        )

    if "zinc" in canonicals and "digestive symptom" in all_canonicals:
        return (
            "위가 약하거나 속이 불편한 편이라면 아연은 공복 복용 시 메스꺼움이나 위장 불편이 생길 수 있습니다. "
            "식후 복용, 제품 함량, 제형을 확인하고 증상이 지속되면 의사 또는 약사에게 상담하세요."
        )

    if IntentType.SYMPTOM_AFTER_INTAKE in intents:
        return (
            "먹고 속이 안 좋다면 계속 복용해도 된다고 단정하기 어렵습니다. 일단 제품명, 성분, 함량, 복용 시점과 증상 지속 시간을 확인해야 하며, "
            "증상이 심하거나 지속되거나 알레르기 의심 증상이 있으면 복용을 중단하고 의료진과 상담하세요."
        )

    if "SAFE_KIDNEY_PROTEIN" in doc_ids:
        return (
            "신장질환이 있다면 프로틴은 임의로 시작하거나 늘리기보다 하루 총 단백질 섭취량을 의료진과 확인하는 것이 안전합니다. "
            "식사 단백질, 보충제 1회 함량, 신장 기능 상태를 함께 봐야 합니다."
        )

    if len(entities) >= 2:
        goal_answer = _goal_based_direct_answer(canonicals)
        if IntentType.GOAL_BASED_RECOMMENDATION in intents and goal_answer:
            return goal_answer
        checks = _component_checks(entities, doc_ids)
        if checks:
            labels = ", ".join(_entity_label(entity.canonical) for entity in entities)
            return (
                f"확인된 성분은 {labels}입니다. 함께 복용 가능 여부는 제품 함량과 개인 상태에 따라 달라질 수 있어 안전하다고 단정하지 않습니다. "
                + " ".join(checks)
                + " 복용 중인 약, 질환, 수술 예정 여부가 있으면 먼저 확인하세요."
            )

    if "SAFE_PREGNANCY_MULTIVITAMIN" in doc_ids:
        return (
            "임산부의 종합비타민 복용은 제품에 따라 달라집니다. 임산부용 제품인지, 비타민A 등 고함량 지용성 비타민과 미네랄 중복이 없는지 확인하고 "
            "의료진 또는 약사와 상담하는 것이 좋습니다."
        )

    if "INT_PROBIOTIC_ANTIBIOTICS" in doc_ids:
        return (
            "유산균은 항생제와 동시에 먹으면 효과가 줄 수 있어 복용 간격을 확인하는 편이 좋습니다. 항생제 치료를 대체하지 않으며, "
            "처방받은 항생제 종류와 유산균 제품 지침에 따라 의사 또는 약사에게 간격을 확인하세요."
        )

    if "INT_ZINC_IRON_CALCIUM" in doc_ids:
        return (
            "아연과 철분 또는 칼슘은 미네랄끼리 흡수 경쟁이 생길 수 있어 같은 시간에 고함량으로 몰아 먹는 것은 피하는 편이 좋습니다. "
            "둘 다 필요하면 제품 함량과 복용 목적을 확인해 간격을 두는 방식을 의사 또는 약사에게 확인하세요."
        )

    if IntentType.GOAL_BASED_RECOMMENDATION in intents:
        goal_answer = _goal_based_direct_answer(canonicals)
        if goal_answer:
            return goal_answer
        return (
            "목적 기반 보충제 추천은 현재 확인된 corpus 범위 안에서만 조심스럽게 볼 수 있습니다. 운동 전인지 운동 후인지, 수면 목적이면 카페인 민감도와 "
            "복용 중인 약, 질환, 현재 먹는 제품을 알려주면 안전 범위 안에서 후보를 좁혀볼 수 있습니다."
        )

    return ""


def _dose_or_label_context_answer(canonicals: set[str], all_canonicals: set[str], question: str) -> str:
    if not canonicals or not _has_dose_or_label_context(question):
        return ""

    if "magnesium" in canonicals:
        return (
            "마그네슘 제품은 라벨의 1회 함량과 하루 총량, 원소량 기준인지 여부를 먼저 확인해야 합니다. "
            "신장질환, 설사·위장 불편, 혈압약이나 항생제·갑상선약 병용 여부가 있으면 임의로 늘리지 말고 의사 또는 약사에게 확인하세요."
        )
    if "vitamin d" in canonicals:
        return (
            "비타민D는 제품 라벨의 IU/μg 함량, 하루 총량, 다른 종합비타민과의 중복, 혈중 비타민D·칼슘 관련 검사 수치를 함께 봐야 합니다. "
            "고함량으로 매일 복용하려는 상황이거나 당뇨약·신장질환·임신/수유 맥락이 있으면 의료진 또는 약사에게 확인하세요."
        )
    if "zinc" in canonicals:
        return (
            "아연은 제품 함량과 하루 총량, 장기 복용 여부를 먼저 확인해야 합니다. "
            "고함량 또는 장기 복용에서는 위장 불편과 구리 결핍 가능성을 보수적으로 봐야 하므로, 철분·칼슘·항생제 병용이나 속 불편이 있으면 약사에게 확인하세요."
        )
    if "iron" in canonicals:
        return (
            "철분은 제품 함량과 복용 목적이 중요합니다. 결핍이나 빈혈 치료 목적이 아니라면 임의로 고함량을 계속 복용하지 않는 편이 안전하며, "
            "변비·속 불편, 커피·칼슘·마그네슘·아연·항생제와의 간격, 검사 수치를 함께 확인하세요."
        )
    if "omega3" in canonicals:
        return (
            "오메가3는 제품 라벨에서 EPA+DHA 총량과 하루 섭취량을 확인해야 합니다. "
            "아스피린·항응고제·항혈소판제 복용, 멍이나 출혈, 수술·시술 예정, 비타민E 고함량 병용이 있으면 복용 간격만으로 판단하지 말고 의료진 또는 약사에게 확인하세요."
        )
    if "vitamin a" in canonicals or "multivitamin" in canonicals:
        return (
            "종합비타민이나 비타민A 제품은 라벨에서 preformed vitamin A 함량, 비타민D·철분·아연 등 중복 성분, 하루 총량을 확인해야 합니다. "
            "임신 준비·임신/수유, 간질환·간수치 이상, 눈 건강 제품이나 간유 제품 병용이 있으면 임의 복용을 피하고 전문가에게 확인하세요."
        )
    if "vitamin e" in canonicals:
        return (
            "비타민E는 제품 함량과 고함량 여부, 다른 항산화 제품과의 중복을 확인해야 합니다. "
            "아스피린·항응고제·항혈소판제 복용, 출혈 병력, 수술·시술 예정이 있으면 안전하다고 단정하지 말고 의사 또는 약사에게 확인하세요."
        )
    if "berberine" in canonicals and "diabetes medication" in all_canonicals:
        return (
            "베르베린은 제품 함량과 목적, 혈당 기록, 복용 중인 당뇨약 종류를 함께 봐야 합니다. "
            "메트포르민·인슐린 등 혈당 관련 약을 복용 중이면 저혈당이나 위장 불편 가능성을 고려해 의료진 또는 약사에게 먼저 확인하세요."
        )

    labels = ", ".join(_entity_label(canonical) for canonical in sorted(canonicals))
    return (
        f"{labels} 제품은 제품명, 성분표, 1회 함량, 하루 복용량, 다른 제품과의 중복 성분을 먼저 확인해야 합니다. "
        "복용 중인 약, 질환, 임신/수유, 수술 예정 여부가 있으면 안전하다고 단정하지 말고 의사 또는 약사에게 확인하세요."
    )


_DOSE_OR_LABEL_PATTERN = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:mg|mcg|μg|ug|iu|g|그램|미리그램|마이크로그램|알|정|캡슐|스푼)",
    re.IGNORECASE,
)


def _has_dose_or_label_context(question: str) -> bool:
    if not question:
        return False
    if _DOSE_OR_LABEL_PATTERN.search(question):
        return True
    return any(
        needle in question
        for needle in (
            "제품",
            "라벨",
            "성분표",
            "함량",
            "복용량",
            "용량",
            "몇 알",
            "몇 정",
            "몇 캡슐",
            "하루 2",
            "하루 두",
            "하루 세",
            "EPA",
            "DHA",
            "고함량",
        )
    )


def _goal_based_direct_answer(canonicals: set[str]) -> str:
    if {"magnesium", "theanine", "melatonin"} <= canonicals:
        return (
            "수면 목적이라면 마그네슘, 테아닌, 멜라토닌은 성격이 다릅니다. 마그네슘은 결핍이나 근육 긴장 맥락을 확인하고, "
            "테아닌은 긴장 완화 쪽으로 조심스럽게 볼 수 있으며, 멜라토닌은 수면 리듬 문제에서 단기간 사용 맥락으로 보는 편이 안전합니다. "
            "불면이 지속되거나 수면제·진정제·혈압약·당뇨약을 복용 중이면 보충제로 해결하려고 하기보다 의료진 또는 약사에게 확인하세요."
        )
    if {"iron", "vitamin b", "caffeine"} <= canonicals:
        return (
            "피로와 집중 저하는 원인 확인이 먼저입니다. 철분은 결핍이나 빈혈 확인 없이 고용량으로 시작하지 않는 편이 좋고, "
            "비타민B군은 결핍 가능성이나 식사 상태를 함께 봐야 하며, 카페인은 단기 각성에는 도움이 될 수 있지만 불면·심박·혈압 문제를 악화시킬 수 있습니다. "
            "증상이 지속되면 검사와 생활요인을 먼저 확인하세요."
        )
    if {"protein", "creatine", "eaa", "glutamine"} <= canonicals:
        return (
            "근육량 증가 목적이라면 우선순위는 보통 식사 단백질과 프로틴으로 하루 총 단백질을 맞추고, 크레아틴은 꾸준히 복용 가능한지와 신장 관련 위험을 확인하는 쪽입니다. "
            "EAA는 식사 단백질이 부족한 상황에서 조건부 후보이고, 글루타민은 운동 성과 향상을 필수로 단정하기 어렵기 때문에 우선순위가 낮습니다. "
            "신장질환이 있으면 단백질·아미노산 보충 총량을 의료진과 확인하세요."
        )
    return ""


def _single_ingredient_timing_answer(canonical: str, docs: list[Any]) -> str:
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        if meta.get("source") != "ingredient_profile" or meta.get("canonical") != canonical:
            continue
        text = getattr(doc, "page_content", "") or ""
        timing = _list_after(text, "Timing")
        spacing = _list_after(text, "Spacing")
        cautions = _list_after(text, "Cautions")
        label = _entity_label(canonical)
        parts = []
        if timing:
            parts.append(f"{label} 복용 타이밍은 {_clean_doc_list_text(timing)}")
        if spacing:
            parts.append(f"다른 성분이나 약과 함께 먹는 경우에는 {_clean_doc_list_text(spacing)}")
        if cautions:
            parts.append(f"주의할 점은 {_clean_doc_list_text(cautions)}")
        if parts:
            return " ".join(parts)
    return ""


def _clean_doc_list_text(text: str) -> str:
    cleaned = re.sub(r"\b(?:General|With food|Before bed|Exercise related):\s*", "", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.replace("..", ".")
    return cleaned.strip(" .") + "."


def _should_prefer_deterministic_answer(parsed_query: ParsedSupplementQuery, docs: list[Any]) -> bool:
    entities = _supplement_entities(parsed_query)
    doc_ids = _doc_ids(docs)
    intents = set(parsed_query.intents)
    if any(entity_coverage_status(entity.canonical) == "unsupported" for entity in parsed_query.entities):
        return True
    if entities and _has_dose_or_label_context(parsed_query.raw_query):
        return True
    if intents & {IntentType.GOAL_BASED_RECOMMENDATION, IntentType.DRUG_INTERACTION, IntentType.CONDITION_SAFETY}:
        direct = _direct_answer_from_query_and_docs(parsed_query, docs)
        if direct:
            return True
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


def _component_checks(entities: list[Any], doc_ids: set[str]) -> list[str]:
    checks: list[str] = []
    canonicals = {entity.canonical for entity in entities}
    if "multivitamin" in canonicals:
        checks.append("종합비타민은 비타민A, 비타민D, 철분, 아연 등 성분 중복과 고함량 여부를 확인하세요.")
    if "omega3" in canonicals:
        checks.append("오메가3는 항응고제 복용, 출혈 위험, 수술 예정 여부를 확인하세요.")
    if "silymarin" in canonicals:
        checks.append("실리마린은 간질환이나 처방약 복용 여부를 확인하세요.")
    if "probiotics" in canonicals or "INT_PROBIOTIC_ANTIBIOTICS" in doc_ids:
        checks.append("유산균은 항생제와 동시에 복용하면 효과가 줄 수 있어 간격 확인이 필요합니다.")
    if "zinc" in canonicals:
        checks.append("아연은 철분, 칼슘, 일부 항생제와 흡수 경쟁이나 간섭 가능성이 있어 복용 간격을 확인하세요.")
    if "protein" in canonicals:
        checks.append("프로틴은 식사 단백질까지 합친 하루 총량과 신장질환 여부를 확인하세요.")
    if "creatine" in canonicals:
        checks.append("크레아틴은 신장 기능 이상 여부, 충분한 수분 섭취, 제품 권장량을 확인하세요.")
    if "vitamin c" in canonicals:
        checks.append("비타민C는 철분 흡수 보조 맥락이 있지만 고용량이나 위장 불편 여부를 확인하세요.")
    if "vitamin e" in canonicals:
        checks.append("비타민E는 고용량, 항응고제·항혈소판제, 수술 예정 여부를 확인하세요.")
    if "vitamin a" in canonicals:
        checks.append("비타민A는 임신 가능성, 간질환, 다른 제품과의 중복 함량을 확인하세요.")
    if "vitamin b" in canonicals:
        checks.append("비타민B군은 피로 원인과 고함량 B6 장기 복용 여부를 확인하세요.")
    if "calcium" in canonicals:
        checks.append("칼슘은 철분, 마그네슘, 아연, 일부 항생제·갑상선약과 간격 확인이 필요할 수 있습니다.")
    if "theanine" in canonicals:
        checks.append("테아닌은 수면장애 치료를 대체한다고 단정하지 말고 졸림이나 약물 병용 여부를 확인하세요.")
    if "melatonin" in canonicals:
        checks.append("멜라토닌은 수면 리듬 맥락의 단기 후보로 보고, 수면제·혈압약·당뇨약 병용 여부를 확인하세요.")
    if "eaa" in canonicals:
        checks.append("EAA는 식사 단백질이 부족한 경우의 조건부 후보이며 프로틴과 목적이 중복될 수 있습니다.")
    if "glutamine" in canonicals:
        checks.append("글루타민은 운동 성과 보충제로 필수라고 단정하기 어렵고 우선순위가 낮을 수 있습니다.")
    if "berberine" in canonicals:
        checks.append("베르베린은 당뇨약 등 처방약 복용과 위장 불편 여부를 확인하세요.")
    if "quercetin" in canonicals:
        checks.append("퀘르세틴은 근거 범위가 제한적이므로 제품 함량과 복용 중인 약을 확인하세요.")
    if "nac" in canonicals:
        checks.append("NAC는 제품 함량과 복용 목적, 처방약 병용 여부를 확인하세요.")
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


def _source_summary(docs: list[Any], source_type: str) -> str:
    filtered = [
        doc
        for doc in docs
        if ((getattr(doc, "metadata", {}) or {}).get("source") or (getattr(doc, "metadata", {}) or {}).get("type"))
        == source_type
    ]
    return _dedupe_join(_doc_answer_summaries(filtered), max_chars=520)


def _supplement_entities(parsed_query: ParsedSupplementQuery) -> list[Any]:
    return [entity for entity in parsed_query.entities if entity.type == EntityType.SUPPLEMENT_INGREDIENT]


def _doc_ids(docs: list[Any]) -> set[str]:
    return {str((getattr(doc, "metadata", {}) or {}).get("id") or "") for doc in docs}


def _entity_label(canonical: str) -> str:
    return entity_label(canonical)


def _has_unknown_or_partial_known(question: str, parsed_query: ParsedSupplementQuery) -> bool:
    if not question:
        return False
    if not parsed_query.entities:
        return True
    if any(entity_coverage_status(entity.canonical) == "unsupported" for entity in parsed_query.entities):
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
    if _is_low_specificity_answer(normalized):
        return True
    return any(needle in normalized for needle in _WEAK_ANSWER_NEEDLES)


def _is_low_specificity_answer(answer: str | None) -> bool:
    if not isinstance(answer, str):
        return True
    normalized = re.sub(r"\s+", " ", answer).strip()
    if not normalized:
        return True
    has_consult_marker = any(marker in normalized for marker in _CONSULT_MARKERS)
    has_actionable_marker = any(marker in normalized for marker in _ACTIONABLE_MARKERS)
    if has_consult_marker and not has_actionable_marker:
        return True
    if len(normalized) <= 90 and _contains_any(normalized, {"개인 상태에 따라", "상황에 따라", "전문가와 상담", "전문가에게 문의"}):
        return not has_actionable_marker
    return False


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

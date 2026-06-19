from __future__ import annotations

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
    preferred_answer = _direct_answer_from_query_and_docs(parsed_query, selected_docs)
    if preferred_answer and _should_prefer_deterministic_answer(parsed_query, selected_docs):
        answer = preferred_answer
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
    if _has_unknown_or_partial_known(question, parsed_query) and not _contains_any(
        enhanced,
        {"성분표", "제품명", "함량", "알려"},
    ):
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
        return (
            "목적 기반 보충제 추천은 현재 확인된 corpus 범위 안에서만 조심스럽게 볼 수 있습니다. 운동 전인지 운동 후인지, 수면 목적이면 카페인 민감도와 "
            "복용 중인 약, 질환, 현재 먹는 제품을 알려주면 안전 범위 안에서 후보를 좁혀볼 수 있습니다."
        )

    return ""


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
        "vitamin d": "비타민D",
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
    connectors = ("랑", "이랑", "와", "과", "+", ",")
    return any(marker in known_text for marker in unknown_markers) and any(connector in question for connector in connectors)


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

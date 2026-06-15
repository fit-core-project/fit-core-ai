from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from engines.supplement.query_understanding import IntentType, ParsedSupplementQuery


_RISK_INTENTS = {
    IntentType.DRUG_INTERACTION,
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
        answer = _fallback_answer_from_docs(selected_docs)
        used_kb_fallback = True

    risk_query = bool(set(parsed_query.intents) & _RISK_INTENTS)
    if risk_query:
        answer = _soften_unsafe_answer(answer)
        caution_parts.append(_CONSULT_TEXT)

    caution = _dedupe_join(caution_parts, max_chars=900)
    if risk_query and caution:
        caution = _soften_unsafe_answer(caution)
    if risk_query and not caution:
        caution = _CONSULT_TEXT

    return ComposedSupplementResponse(
        answer=answer or _fallback_answer_from_docs(selected_docs),
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
            _dict_get(data, "summary", "Summary"),
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
        source = meta.get("source")
        text = getattr(doc, "page_content", "") or ""
        if source == "interaction_rule":
            parts.extend(
                [
                    _field(text, "Recommendation"),
                    _field(text, "Spacing guidance"),
                    _field(text, "Caution level"),
                    _list_after(text, "Consultation required when"),
                ]
            )
        elif source == "safety_rule":
            parts.extend(
                [
                    _field(text, "Risk"),
                    _field(text, "Recommendation"),
                    _field(text, "When to consult"),
                    _field(text, "Caution level"),
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


def _fallback_answer_from_docs(docs: list[Any]) -> str:
    source_types = {(getattr(doc, "metadata", {}) or {}).get("source") for doc in docs}
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
    if "ingredient_profile" in source_types or "supp_timing" in source_types:
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

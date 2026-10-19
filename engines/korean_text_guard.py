"""Guard user-facing routine text against English-heavy LLM prose and raw schema tokens."""
from __future__ import annotations

import re

from typing import Any

from .schemas import LLMRoutineOutput, PainAreaEntry, RoutineDraftResponse


_SUMMARY_FALLBACK = "맞춤 운동 루틴"
_RATIONALE_FALLBACKS = (
    "요청한 목표와 컨디션을 반영해 구성했습니다.",
    "사용 가능한 후보 운동과 제한 조건을 기준으로 안전하게 조정했습니다.",
    "각 운동은 목표 근육과 장비 조건을 고려해 배치했습니다.",
)
_EXERCISE_RATIONALE_FALLBACK = "목표 근육, 장비 조건, 컨디션을 기준으로 선택한 운동입니다."
_WARNING_FALLBACK = "컨디션에 맞춰 중량과 반복 수를 조절하세요."
_RAW_EXERCISE_ID_RE = re.compile(r"\bV\d{2}ROW-\d+\b")
_HEX_BYTE_SEQUENCE_RE = re.compile(r"(?:<0x[0-9A-Fa-f]{2}>)+")
_HEX_BYTE_TOKEN_RE = re.compile(r"<0x([0-9A-Fa-f]{2})>")
_DOMS_ONLY_MEDICAL_WORD_REPLACEMENTS = {
    "부상 우려가": "근육통/피로도 부담이",
    "부상 우려는": "근육통/피로도 부담은",
    "부상 우려를": "근육통/피로도 부담을",
    "부상 우려로": "근육통/피로도 부담으로",
    "부상 우려": "근육통/피로도 부담",
    "부상 부담가": "근육통/피로도 부담이",
    "부상 부담는": "근육통/피로도 부담은",
    "부상 부담를": "근육통/피로도 부담을",
    "부상 부담로": "근육통/피로도 부담으로",
    "부상 부담": "근육통/피로도 부담",
    "부상 위험가": "근육통/피로도 부담이",
    "부상 위험는": "근육통/피로도 부담은",
    "부상 위험를": "근육통/피로도 부담을",
    "부상 위험로": "근육통/피로도 부담으로",
    "부상 위험": "근육통/피로도 부담",
    "부상": "근육통/피로도",
    "통증": "근육통",
    "손상": "피로 누적",
    "질환": "회복 상태",
}

_TOKEN_LABELS = {
    "primary_muscles": "주동근",
    "primary muscle": "주동근",
    "primary_muscle": "주동근",
    "secondary_muscle": "보조 근육",
    "secondary": "보조 근육",
    "target_muscles": "목표 근육",
    "target_split_label": "운동 분할",
    "readinessLevel": "컨디션",
    "timeAvailableMin": "가용 시간",
    "current_pain_areas": "현재 통증 부위",
    "mobility_limits": "가동성 제한",
    "condition_policies": "수술/질환 주의 조건",
    "anthropometry_signals": "체형 신호",
    "hypertrophy": "근비대",
    "strength": "근력",
    "스트렝스": "근력",
    "power": "파워",
    "beginner": "초보자",
    "intermediate": "중급자",
    "advanced": "상급자",
    "mild": "약함",
    "moderate": "중간",
    "severe": "심함",
    "비기너": "초보자",
    "인터미디에이트": "중급자",
    "어드밴스드": "상급자",
    "low readiness": "낮은 컨디션",
    "low": "낮음",
    "medium": "중간",
    "high": "높음",
    "elite": "매우 높음",
    "excellent": "매우 좋음",
    "good": "좋음",
    "fair": "보통",
    "poor": "낮음",
    "COMPOUND": "복합 운동",
    "ISOLATION": "고립 운동",
    "STATIC": "정적 운동",
    "BODYWEIGHT": "맨몸",
    "DUMBBELL": "덤벨",
    "BARBELL": "바벨",
    "MACHINE": "머신",
    "CABLE": "케이블",
    "SMITH_MACHINE": "스미스 머신",
    "KETTLEBELL": "케틀벨",
    "LANDMINE": "랜드마인",
    "BAND": "밴드",
    "PLATE": "원판",
    "chest": "가슴",
    "upper-back": "상부 등",
    "lower-back": "허리/요추",
    "front-deltoids": "전면 어깨",
    "back-deltoids": "후면 어깨",
    "side-deltoids": "측면 어깨",
    "triceps": "삼두",
    "biceps": "이두",
    "forearm": "전완",
    "trapezius": "승모",
    "abs": "복근",
    "obliques": "복사근",
    "quadriceps": "대퇴사두",
    "hamstring": "햄스트링",
    "gluteal": "둔근",
    "calves": "종아리",
    "left-soleus": "왼쪽 가자미근/종아리",
    "right-soleus": "오른쪽 가자미근/종아리",
    "soleus": "가자미근/종아리",
    "adductor": "내전근",
    "abductors": "외전근",
    "neck": "목",
    "knees": "무릎",
    "wrist": "손목",
}

_CAUTIOUS_TEXT_REPLACEMENTS = {
    "장골(긴 대퇴골)": "긴 대퇴 비율",
    "장골 신호": "긴 대퇴 비율 신호",
    "장골": "긴 대퇴 비율",
    "상부 승모근": "승모근",
    "중부 승모근": "승모근",
    "하부 승모근": "승모근",
    "100% 안전": "상대적으로 부담이 낮은 편",
    "완전히 안전": "상대적으로 부담이 낮은 편",
    "안전하게 고강도 훈련": "부담을 낮추는 방향으로 훈련",
    "안전하게 고볼륨": "부담을 낮추는 방향으로 볼륨을",
    "안전하게 수행": "부담을 낮춰 수행",
    "안전하게 진행": "부담을 낮춰 진행",
    "안전하게": "부담을 낮추는 방향으로",
    "안전한 환경": "동작이 안정적인 환경",
    "안전한 옵션": "상대적으로 부담이 낮은 옵션",
    "안전한 대안": "상대적으로 부담이 낮은 대안",
    "안전한 운동": "상대적으로 부담이 낮은 운동",
    "안전한 선택": "상대적으로 부담이 낮은 선택",
    "안전한 머신 기반": "부담을 조절하기 쉬운 머신 기반",
    "안전 후보": "제약을 통과한 후보",
    "통증이 없는": "통증 부담을 낮춘",
    "통증 없이": "통증 여부를 확인하며",
    "통증을 해결": "통증 부담을 고려",
    "치료합니다": "관리 관점에서 참고할 수 있습니다",
    "치료에 도움이 됩니다": "부담 관리에 참고할 수 있습니다",
    "회복을 돕습니다": "부담 관리에 참고할 수 있습니다",
    "부상 방지에 좋습니다": "부상 위험을 낮추는 방향입니다",
    "부상 방지": "부상 위험을 낮추는 방향",
    "부상을 예방하고": "운동 전 준비에 활용하고",
    "부상을 예방하는": "운동 전 준비에 활용하는",
    "부상 예방에": "운동 전 준비에",
    "부상 예방": "운동 전 준비",
    "효과를 극대화": "효과를 높이는 데 초점",
    "목표를 달성합니다": "목표에 맞춥니다",
}

_PHASE_LANGUAGE_REPLACEMENTS = {
    "warmup_or_preparation": {
        "메인 루틴 전후로": "메인 루틴 전에",
        "루틴 전후로": "루틴 전에",
        "운동 전후로": "운동 전에",
    },
    "conditioning_finisher": {
        "메인 루틴 전후로": "메인 루틴 후에",
        "루틴 전후로": "루틴 후에",
        "운동 전후로": "운동 후에",
    },
    "endurance_support": {
        "메인 루틴 전후로": "메인 루틴 후에",
        "루틴 전후로": "루틴 후에",
        "운동 전후로": "운동 후에",
    },
    "cooldown_recovery": {
        "메인 루틴 전후로": "메인 루틴 후에",
        "루틴 전후로": "루틴 후에",
        "운동 전후로": "운동 후에",
    },
    # Compatibility for historical evidence fixtures only.
    "finisher_or_recovery": {
        "메인 루틴 전후로": "메인 루틴 후에",
        "루틴 전후로": "루틴 후에",
        "운동 전후로": "운동 후에",
    },
}


def _has_final_consonant(value: str) -> bool:
    if not value:
        return False
    code = ord(value[-1])
    if not (0xAC00 <= code <= 0xD7A3):
        return False
    return (code - 0xAC00) % 28 != 0


def _label_with_particle(label: str, particle: str | None) -> str:
    if not particle:
        return label
    has_batchim = _has_final_consonant(label)
    if particle in {"이", "가"}:
        return label + ("이" if has_batchim else "가")
    if particle in {"은", "는"}:
        return label + ("은" if has_batchim else "는")
    if particle in {"을", "를"}:
        return label + ("을" if has_batchim else "를")
    if particle in {"과", "와"}:
        return label + ("과" if has_batchim else "와")
    if particle in {"으로", "로"}:
        final = ord(label[-1]) - 0xAC00 if label and 0xAC00 <= ord(label[-1]) <= 0xD7A3 else -1
        final_consonant = final % 28 if final >= 0 else 0
        return label + ("으로" if final_consonant and final_consonant != 8 else "로")
    return label + particle


def _is_english_heavy(text: str | None) -> bool:
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped:
        return False

    hangul_count = sum(1 for char in stripped if "\uac00" <= char <= "\ud7a3")
    ascii_alpha_count = sum(1 for char in stripped if ("A" <= char <= "Z") or ("a" <= char <= "z"))
    letter_count = hangul_count + ascii_alpha_count
    if letter_count == 0:
        return False

    # Keep mixed Korean text with enum/id tokens, but block English explanation prose.
    return ascii_alpha_count >= 12 and ascii_alpha_count / letter_count >= 0.60


def _replace_schema_tokens(text: str, exercise_labels: dict[str, str] | None = None) -> str:
    sanitized = text
    for exercise_id, exercise_name in sorted((exercise_labels or {}).items(), key=lambda item: len(item[0]), reverse=True):
        if exercise_id and exercise_name:
            sanitized = sanitized.replace(exercise_id, exercise_name)
    sanitized = _RAW_EXERCISE_ID_RE.sub("해당 운동", sanitized)
    for raw, label in sorted(_TOKEN_LABELS.items(), key=lambda item: len(item[0]), reverse=True):
        particle_pattern = rf"(?<![A-Za-z0-9_-]){re.escape(raw)}(이|가|은|는|을|를|과|와|으로|로)(?![A-Za-z0-9_\-\uac00-\ud7a3])"
        sanitized = re.sub(particle_pattern, lambda match: _label_with_particle(label, match.group(1)), sanitized)
        token_pattern = rf"(?<![A-Za-z0-9_-]){re.escape(raw)}(?![A-Za-z0-9_-])"
        sanitized = re.sub(token_pattern, label, sanitized)
    return sanitized


def _repair_malformed_text_encoding(text: str) -> str:
    """Decode textual UTF-8 byte markers emitted by some local model runs."""

    def decode_sequence(match: re.Match[str]) -> str:
        try:
            raw = bytes(
                int(value, 16)
                for value in _HEX_BYTE_TOKEN_RE.findall(match.group(0))
            )
            return raw.decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            return "\ufffd"

    repaired = _HEX_BYTE_SEQUENCE_RE.sub(decode_sequence, text)
    return _HEX_BYTE_TOKEN_RE.sub("\ufffd", repaired)


def _normalize_body_part_slug(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _doms_only_labels(
    doms_data: dict[str, int] | None,
    pain_areas: list[PainAreaEntry] | None,
) -> set[str]:
    pain_parts = {
        _normalize_body_part_slug(pain.body_part)
        for pain in (pain_areas or [])
        if pain.body_part
    }
    doms_parts = {
        _normalize_body_part_slug(part)
        for part, level in (doms_data or {}).items()
        if int(level or 0) > 0
    }
    return {
        _replace_schema_tokens(part)
        for part in (doms_parts - pain_parts)
        if part
    } | (doms_parts - pain_parts)


def _body_part_labels(parts: set[str]) -> set[str]:
    return {
        label
        for part in parts
        for label in (part, _replace_schema_tokens(part))
        if label
    }


def _replace_doms_only_medical_words(
    text: str,
    *,
    replace_globally: bool,
    labels: set[str],
    protected_labels: set[str],
    trusted_server_texts: set[str] | None = None,
) -> str:
    if not isinstance(text, str) or not text:
        return text
    if not replace_globally and not any(label and label in text for label in labels):
        return text

    all_labels = sorted(labels | protected_labels, key=len, reverse=True)
    mentions: list[tuple[int, int, str]] = []
    for label in all_labels:
        kind = "pain" if label in protected_labels else "doms"
        for match in re.finditer(re.escape(label), text):
            mentions.append((match.start(), match.end(), kind))

    replacements = sorted(
        _DOMS_ONLY_MEDICAL_WORD_REPLACEMENTS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    medical_pattern = re.compile("|".join(re.escape(raw) for raw, _ in replacements))
    replacement_by_raw = dict(replacements)
    trusted_spans = [
        (match.start(), match.end())
        for trusted_text in (trusted_server_texts or set())
        if trusted_text
        for match in re.finditer(re.escape(trusted_text), text)
    ]

    def is_trusted_server_span(match: re.Match[str]) -> bool:
        return any(start <= match.start() and match.end() <= end for start, end in trusted_spans)

    def replace_nearest_label(match: re.Match[str]) -> str:
        if is_trusted_server_span(match):
            return match.group(0)
        if replace_globally:
            return replacement_by_raw[match.group(0)]
        nearby: list[tuple[int, str, str, str]] = []
        for start, end, kind in mentions:
            if end <= match.start():
                distance = match.start() - end
                context = text[end:match.start()]
                direction = "preceding"
            elif start >= match.end():
                distance = start - match.end()
                context = text[match.end():start]
                direction = "following"
            else:
                continue
            if distance > 24 or re.search(r"[\n.!?;。！？]", context):
                continue
            if medical_pattern.search(context):
                continue
            nearby.append((distance, kind, direction, context))

        if not nearby:
            return match.group(0)

        nearest_distance = min(distance for distance, _, _, _ in nearby)
        nearest_kinds = {
            kind
            for distance, kind, _, _ in nearby
            if distance == nearest_distance
        }
        if len(nearest_kinds) != 1:
            return match.group(0)
        if nearest_kinds != {"doms"}:
            return match.group(0)
        return replacement_by_raw[match.group(0)]

    return medical_pattern.sub(replace_nearest_label, text)


def separate_doms_from_pain_wording_on_response(
    response: RoutineDraftResponse,
    *,
    doms_data: dict[str, int] | None,
    pain_areas: list[PainAreaEntry] | None,
    trusted_reasons_by_exercise_id: dict[str, set[str]] | None = None,
) -> RoutineDraftResponse:
    """Prevent DOMS-only context from being described as pain/injury in user text.

    DOMS levels mean soreness/fatigue. If the same body part is not also present
    in pain_areas, final user-facing text must not frame it as injury or acute
    pain. When there is no pain context at all, sanitize all visible text; when
    mixed pain and DOMS exist, sanitize only sentences that mention DOMS labels.
    Exact score reasons supplied by the server for the same exercise retain
    their exercise-level caution meaning; untrusted text is never exempted.
    """
    labels = _doms_only_labels(doms_data, pain_areas)
    if not labels:
        return response

    guarded = response.model_copy(deep=True)
    replace_globally = not pain_areas
    pain_parts = {
        _normalize_body_part_slug(pain.body_part)
        for pain in (pain_areas or [])
        if pain.body_part
    }
    protected_labels = _body_part_labels(pain_parts)
    guarded.summary_title = _replace_doms_only_medical_words(
        guarded.summary_title,
        replace_globally=replace_globally,
        labels=labels,
        protected_labels=protected_labels,
    )
    guarded.rationale_summary = [
        _replace_doms_only_medical_words(
            item,
            replace_globally=replace_globally,
            labels=labels,
            protected_labels=protected_labels,
        )
        for item in guarded.rationale_summary
    ]
    guarded.warnings = [
        _replace_doms_only_medical_words(
            item,
            replace_globally=replace_globally,
            labels=labels,
            protected_labels=protected_labels,
        )
        for item in guarded.warnings
    ]
    for block in guarded.routine_blocks:
        trusted_server_texts = (trusted_reasons_by_exercise_id or {}).get(
            str(block.exercise_id or "").strip().lower(),
            set(),
        )
        block.exercise_rationale = _replace_doms_only_medical_words(
            block.exercise_rationale,
            replace_globally=replace_globally,
            labels=labels,
            protected_labels=protected_labels,
            trusted_server_texts=trusted_server_texts,
        )
        block.reasons = [
            _replace_doms_only_medical_words(
                item,
                replace_globally=replace_globally,
                labels=labels,
                protected_labels=protected_labels,
                trusted_server_texts=trusted_server_texts,
            )
            for item in block.reasons
        ]
    return guarded


def _soften_absolute_safety_claims(text: str) -> str:
    softened = text
    for raw, replacement in _CAUTIOUS_TEXT_REPLACEMENTS.items():
        softened = softened.replace(raw, replacement)
    return softened


def _align_session_phase_language(text: str, session_phase: str | None) -> str:
    aligned = text
    replacements = _PHASE_LANGUAGE_REPLACEMENTS.get(
        str(session_phase or "").strip().lower(),
        {},
    )
    for raw, replacement in replacements.items():
        aligned = aligned.replace(raw, replacement)
    return aligned


def _sanitize_natural_language(
    text: str,
    fallback: str,
    exercise_labels: dict[str, str] | None = None,
) -> str:
    sanitized = _repair_malformed_text_encoding(text)
    if "\ufffd" in sanitized:
        return fallback
    sanitized = _replace_schema_tokens(sanitized, exercise_labels)
    sanitized = _soften_absolute_safety_claims(sanitized)
    return fallback if _is_english_heavy(sanitized) else sanitized


def _fallback_list(
    values: list[str],
    fallbacks: tuple[str, ...],
    exercise_labels: dict[str, str] | None = None,
) -> list[str]:
    if not values:
        return []
    return [
        value
        if isinstance(value, str) and value.startswith("Guard:")
        else _sanitize_natural_language(
            value,
            fallbacks[index % len(fallbacks)],
            exercise_labels,
        )
        for index, value in enumerate(values)
    ]


def _exercise_labels_from_output(output: LLMRoutineOutput) -> dict[str, str]:
    return {
        str(exercise.exercise_id): exercise.exercise_name
        for exercise in output.exercises
        if exercise.exercise_id and exercise.exercise_name
    }


def _exercise_labels_from_response(response: RoutineDraftResponse) -> dict[str, str]:
    return {
        str(block.exercise_id): block.exercise_name
        for block in response.routine_blocks
        if block.exercise_id and block.exercise_name
    }


def enforce_korean_user_text_on_output(output: LLMRoutineOutput) -> LLMRoutineOutput:
    """Return a copy whose user-visible natural-language fields avoid English prose."""
    guarded = output.model_copy(deep=True)
    exercise_labels = _exercise_labels_from_output(guarded)
    guarded.summary_title = _sanitize_natural_language(guarded.summary_title, _SUMMARY_FALLBACK, exercise_labels)
    guarded.rationale_summary = _fallback_list(guarded.rationale_summary, _RATIONALE_FALLBACKS, exercise_labels)
    guarded.warnings = _fallback_list(guarded.warnings, (_WARNING_FALLBACK,), exercise_labels)
    for exercise in guarded.exercises:
        exercise.exercise_rationale = _sanitize_natural_language(
            exercise.exercise_rationale,
            _EXERCISE_RATIONALE_FALLBACK,
            exercise_labels,
        )
    return guarded


def enforce_korean_user_text_on_response(response: RoutineDraftResponse) -> RoutineDraftResponse:
    """Apply the same guard to already-built fallback/response objects."""
    guarded = response.model_copy(deep=True)
    exercise_labels = _exercise_labels_from_response(guarded)
    guarded.summary_title = _sanitize_natural_language(guarded.summary_title, _SUMMARY_FALLBACK, exercise_labels)
    guarded.rationale_summary = _fallback_list(guarded.rationale_summary, _RATIONALE_FALLBACKS, exercise_labels)
    guarded.warnings = _fallback_list(guarded.warnings, (_WARNING_FALLBACK,), exercise_labels)
    for block in guarded.routine_blocks:
        block.exercise_rationale = _sanitize_natural_language(
            block.exercise_rationale,
            _EXERCISE_RATIONALE_FALLBACK,
            exercise_labels,
        )
        block.exercise_rationale = _align_session_phase_language(
            block.exercise_rationale,
            block.session_phase,
        )
    return guarded

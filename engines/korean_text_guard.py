"""Guard user-facing routine text against English-heavy LLM prose and raw schema tokens."""
from __future__ import annotations

import re

from .schemas import LLMRoutineOutput, RoutineDraftResponse


_SUMMARY_FALLBACK = "맞춤 운동 루틴"
_RATIONALE_FALLBACKS = (
    "요청한 목표와 컨디션을 반영해 구성했습니다.",
    "사용 가능한 후보 운동과 제한 조건을 기준으로 안전하게 조정했습니다.",
    "각 운동은 목표 근육과 장비 조건을 고려해 배치했습니다.",
)
_EXERCISE_RATIONALE_FALLBACK = "목표 근육, 장비 조건, 컨디션을 기준으로 선택한 운동입니다."
_WARNING_FALLBACK = "컨디션에 맞춰 중량과 반복 수를 조절하세요."
_RAW_EXERCISE_ID_RE = re.compile(r"\bV\d{2}ROW-\d+\b")

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
    "power": "파워",
    "beginner": "초보자",
    "intermediate": "중급자",
    "advanced": "상급자",
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
    "adductor": "내전근",
    "abductors": "외전근",
    "neck": "목",
    "knees": "무릎",
    "wrist": "손목",
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


def _sanitize_natural_language(
    text: str,
    fallback: str,
    exercise_labels: dict[str, str] | None = None,
) -> str:
    sanitized = _replace_schema_tokens(text, exercise_labels)
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
    return guarded

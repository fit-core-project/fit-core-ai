"""Guard user-facing routine text against English-heavy LLM prose."""
from __future__ import annotations

from .schemas import LLMRoutineOutput, RoutineDraftResponse


_SUMMARY_FALLBACK = "맞춤 운동 루틴"
_RATIONALE_FALLBACKS = (
    "요청한 목표와 컨디션을 반영해 구성했습니다.",
    "사용 가능한 후보 운동과 제한 조건을 기준으로 안전하게 조정했습니다.",
    "각 운동은 목표 근육과 장비 조건을 고려해 배치했습니다.",
)
_EXERCISE_RATIONALE_FALLBACK = "목표 근육, 장비 조건, 컨디션을 기준으로 선택한 운동입니다."
_WARNING_FALLBACK = "컨디션에 맞춰 중량과 반복 수를 조절하세요."


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


def _fallback_list(values: list[str], fallbacks: tuple[str, ...]) -> list[str]:
    if not values:
        return []
    return [
        value
        if isinstance(value, str) and value.startswith("Guard:")
        else fallbacks[index % len(fallbacks)] if _is_english_heavy(value) else value
        for index, value in enumerate(values)
    ]


def enforce_korean_user_text_on_output(output: LLMRoutineOutput) -> LLMRoutineOutput:
    """Return a copy whose user-visible natural-language fields avoid English prose."""
    guarded = output.model_copy(deep=True)
    if _is_english_heavy(guarded.summary_title):
        guarded.summary_title = _SUMMARY_FALLBACK
    guarded.rationale_summary = _fallback_list(guarded.rationale_summary, _RATIONALE_FALLBACKS)
    guarded.warnings = _fallback_list(guarded.warnings, (_WARNING_FALLBACK,))
    for exercise in guarded.exercises:
        if _is_english_heavy(exercise.exercise_rationale):
            exercise.exercise_rationale = _EXERCISE_RATIONALE_FALLBACK
    return guarded


def enforce_korean_user_text_on_response(response: RoutineDraftResponse) -> RoutineDraftResponse:
    """Apply the same guard to already-built fallback/response objects."""
    guarded = response.model_copy(deep=True)
    if _is_english_heavy(guarded.summary_title):
        guarded.summary_title = _SUMMARY_FALLBACK
    guarded.rationale_summary = _fallback_list(guarded.rationale_summary, _RATIONALE_FALLBACKS)
    guarded.warnings = _fallback_list(guarded.warnings, (_WARNING_FALLBACK,))
    for block in guarded.routine_blocks:
        if _is_english_heavy(block.exercise_rationale):
            block.exercise_rationale = _EXERCISE_RATIONALE_FALLBACK
    return guarded

"""Guard user-facing routine text against English-heavy LLM prose and raw code tokens."""
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


# ---------------------------------------------------------------------------
# Code token -> Korean display text.
#
# Display-only. The muscle, equipment, split, goal and readiness labels mirror
# fit-core-frontend/utils/displayLabels.ts so chips and prose read the same.
# This is NOT a vocabulary SSOT: request/response codes stay the spreadsheet
# slugs and enums; only free-text fields shown to the user go through here.
# ---------------------------------------------------------------------------

_MUSCLE_LABELS = {
    "chest": "가슴",
    "upper-back": "등",
    "lower-back": "허리",
    "trapezius": "승모근",
    "front-deltoids": "전면 어깨",
    "side-deltoids": "측면 어깨",
    "back-deltoids": "후면 어깨",
    "biceps": "이두",
    "triceps": "삼두",
    "forearm": "전완",
    "abs": "복부",
    "obliques": "옆구리",
    "gluteal": "둔근",
    "quadriceps": "앞 허벅지",
    "hamstring": "뒤 허벅지",
    "adductor": "내전근",
    "abductors": "외전근",
    "calves": "종아리",
    "neck": "목",
    # Pain-only slugs (exercise_tier.pain_triggers) that can echo into LLM text.
    "knees": "무릎",
    "left-soleus": "왼쪽 종아리",
    "right-soleus": "오른쪽 종아리",
    "head": "머리",
}

_EQUIPMENT_LABELS = {
    "SMITH_MACHINE": "스미스 머신",
    "BAND": "밴드",
    "BARBELL": "바벨",
    "BODYWEIGHT": "맨몸",
    "CABLE": "케이블",
    "DUMBBELL": "덤벨",
    "KETTLEBELL": "케틀벨",
    "LANDMINE": "랜드마인",
    "MACHINE": "머신",
    "PLATE": "원판",
}

_SPLIT_LABELS = {
    "push": "밀기",
    "pull": "당기기",
    "legs": "하체",
    "lower": "하체",
    "upper": "상체",
    "chest_back": "가슴·등",
    "full_body": "전신",
    "core": "코어",
    "shoulder": "어깨",
    "arm": "팔",
    "custom": "맞춤형",  # BE default label when the user picks muscles directly
}

_GOAL_LABELS = {
    "hypertrophy": "근비대",
    "strength": "근력",
    "endurance": "근지구력",
    "fatLoss": "체지방 감량",
    "recomposition": "바디 리컴포지션",
    "generalFitness": "건강 유지",
}

_MOVEMENT_TYPE_LABELS = {
    "COMPOUND": "복합 운동",
    "ISOLATION": "고립 운동",
    "STATIC": "정적 운동",
}

_LEVEL_LABELS = {
    "severe": "심함",
    "high": "심함",
    "moderate": "중간",
    "medium": "중간",
    "mild": "약함",
    "low": "약함",
}

_READINESS_LEVEL_LABELS = {"low": "피곤함", "normal": "보통", "high": "좋음"}

_FIELD_LABELS = {
    "primary_muscles": "주동근",
    "primary_muscle": "주동근",
    "primaryMuscles": "주동근",
    "secondary_muscle": "보조 근육",
    "movement_type": "운동 유형",
    "movementType": "운동 유형",
    "equipment_req": "필요 장비",
    "unavailable_equipment": "사용 불가 장비",
    "unavailableEquipment": "사용 불가 장비",
    "pain_triggers": "통증 유발 부위",
    "current_pain_areas": "통증 부위",
    "currentPainAreas": "통증 부위",
    "target_muscles": "목표 부위",
    "targetMuscles": "목표 부위",
    "target_split_label": "분할",
    "targetSplitLabel": "분할",
    "timeAvailableMin": "운동 가능 시간",
    "time_available_min": "운동 가능 시간",
    "targetExerciseCount": "목표 운동 수",
}


def _build_token_rules() -> list[tuple[re.Pattern[str], object]]:
    """Ordered (pattern, replacement) rules; longer/compound phrases first."""
    boundary_l = r"(?<![A-Za-z0-9_\-])"
    # The optional trailing particle is captured so it can be re-agreed with the Korean label.
    # Only a standalone particle is captured ("chest가 " yes, "push이후" no).
    boundary_r = r"(?![A-Za-z0-9_\-])(?:(?P<particle>이라서|라서|이라|라|이며|며|이고|고|으로|로|이|가|을|를|은|는|과|와)(?![가-힣]))?"
    rules: list[tuple[re.Pattern[str], object]] = []

    def add(token: str, label: object, *, ignore_case: bool) -> None:
        flags = re.IGNORECASE if ignore_case else 0
        rules.append((re.compile(boundary_l + re.escape(token) + boundary_r, flags), label))

    level_alt = "|".join(_LEVEL_LABELS)
    # "DOMS moderate", "DOMS: high", "DOMS level 2"
    rules.append((
        re.compile(boundary_l + r"DOMS\s*[:=]?\s*(" + level_alt + r")" + boundary_r, re.IGNORECASE),
        lambda m: f"근육통 {_LEVEL_LABELS[m.group(1).lower()]}",
    ))
    rules.append((
        re.compile(boundary_l + r"DOMS\s*(?:level\s*)?[:=]?\s*([0-3])" + boundary_r, re.IGNORECASE),
        lambda m: f"근육통 {m.group(1)}단계",
    ))
    add("DOMS", "근육통", ignore_case=True)
    rules.append((
        re.compile(
            boundary_l + r"(?:readinessLevel|readiness_level|readiness)\s*[:=]?\s*(low|normal|high)" + boundary_r,
            re.IGNORECASE,
        ),
        lambda m: f"컨디션 {_READINESS_LEVEL_LABELS[m.group(1).lower()]}",
    ))
    for token in ("readinessLevel", "readiness_level", "readiness"):
        add(token, "컨디션", ignore_case=True)

    for token, label in sorted(_FIELD_LABELS.items(), key=lambda item: -len(item[0])):
        add(token, label, ignore_case=False)
    for token, label in sorted(_EQUIPMENT_LABELS.items(), key=lambda item: -len(item[0])):
        add(token, label, ignore_case=True)
    for token, label in _MOVEMENT_TYPE_LABELS.items():
        add(token, label, ignore_case=True)
    for token, label in sorted(_MUSCLE_LABELS.items(), key=lambda item: -len(item[0])):
        add(token, label, ignore_case=True)
    for token, label in sorted(_SPLIT_LABELS.items(), key=lambda item: -len(item[0])):
        add(token, label, ignore_case=True)
    for token, label in _GOAL_LABELS.items():
        add(token, label, ignore_case=True)
    return rules


_TOKEN_RULES = _build_token_rules()

# Korean particle pairs: (after final consonant, after vowel).
_PARTICLE_PAIRS = (
    ("이라서", "라서"), ("이라", "라"), ("이며", "며"), ("이고", "고"),
    ("이", "가"), ("을", "를"), ("은", "는"), ("과", "와"), ("으로", "로"),
)


def _final_consonant(text: str) -> int | None:
    """Jongseong index of the last Hangul syllable (0 = none), or None if not Hangul."""
    if not text:
        return None
    last = text[-1]
    if not ("가" <= last <= "힣"):
        return None
    return (ord(last) - 0xAC00) % 28


def _fix_particle(label: str, particle: str) -> str:
    """Return the particle form that agrees with the Korean label's final consonant."""
    if not particle:
        return particle
    jong = _final_consonant(label)
    if jong is None:
        return particle
    has_final = jong != 0
    for with_final, without_final in _PARTICLE_PAIRS:
        if particle in (with_final, without_final):
            if with_final == "으로":
                # ㄹ final (jongseong index 8) takes "로".
                return "로" if (not has_final or jong == 8) else "으로"
            return with_final if has_final else without_final
    return particle


def translate_code_tokens(text: str | None) -> str | None:
    """Replace raw contract tokens (push, DOMS moderate, MACHINE, chest, ...) with Korean display text."""
    if not isinstance(text, str) or not text:
        return text
    result = text
    for pattern, label in _TOKEN_RULES:
        def _replace(match: re.Match[str], label: object = label) -> str:
            replacement = label(match) if callable(label) else str(label)
            return replacement + _fix_particle(replacement, match.group("particle") or "")

        result = pattern.sub(_replace, result)
    return result


def _is_english_heavy(text: str | None) -> bool:
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped:
        return False

    hangul_count = sum(1 for char in stripped if "가" <= char <= "힣")
    ascii_alpha_count = sum(1 for char in stripped if ("A" <= char <= "Z") or ("a" <= char <= "z"))
    letter_count = hangul_count + ascii_alpha_count
    if letter_count == 0:
        return False

    # Keep mixed Korean text with enum/id tokens, but block English explanation prose.
    return ascii_alpha_count >= 12 and ascii_alpha_count / letter_count >= 0.60


def _guard_text(value: str, fallback: str) -> str:
    if _is_english_heavy(value):
        return fallback
    return translate_code_tokens(value) or value


def _fallback_list(values: list[str], fallbacks: tuple[str, ...]) -> list[str]:
    if not values:
        return []
    return [
        value
        if isinstance(value, str) and value.startswith("Guard:")
        else _guard_text(value, fallbacks[index % len(fallbacks)])
        for index, value in enumerate(values)
    ]


def enforce_korean_user_text_on_output(output: LLMRoutineOutput) -> LLMRoutineOutput:
    """Return a copy whose user-visible natural-language fields avoid English prose and raw codes."""
    guarded = output.model_copy(deep=True)
    guarded.summary_title = _guard_text(guarded.summary_title, _SUMMARY_FALLBACK)
    guarded.rationale_summary = _fallback_list(guarded.rationale_summary, _RATIONALE_FALLBACKS)
    guarded.warnings = _fallback_list(guarded.warnings, (_WARNING_FALLBACK,))
    for exercise in guarded.exercises:
        exercise.exercise_rationale = _guard_text(exercise.exercise_rationale, _EXERCISE_RATIONALE_FALLBACK)
        for substitution in exercise.substitution_candidates:
            substitution.reason = _guard_text(substitution.reason, _EXERCISE_RATIONALE_FALLBACK)
    return guarded


def enforce_korean_user_text_on_response(response: RoutineDraftResponse) -> RoutineDraftResponse:
    """Apply the same guard to already-built fallback/response objects."""
    guarded = response.model_copy(deep=True)
    guarded.summary_title = _guard_text(guarded.summary_title, _SUMMARY_FALLBACK)
    guarded.rationale_summary = _fallback_list(guarded.rationale_summary, _RATIONALE_FALLBACKS)
    guarded.warnings = _fallback_list(guarded.warnings, (_WARNING_FALLBACK,))
    for block in guarded.routine_blocks:
        block.exercise_rationale = _guard_text(block.exercise_rationale, _EXERCISE_RATIONALE_FALLBACK)
        for substitution in block.substitution_candidates:
            substitution.reason = _guard_text(substitution.reason, _EXERCISE_RATIONALE_FALLBACK)
    return guarded

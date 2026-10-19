"""Closed, user-facing profile constraint contract for routine generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


PROFILE_INPUT_CONTRACT_VERSION = "routine-profile-input-v3-2026-07-22"


PROFILE_SIGNAL_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "code": "heel_down_deep_squat_difficult",
        "category": "functionalMovement",
        "ui_section": "movementChecks",
        "display_order": 10,
        "label_ko": "발뒤꿈치를 붙이고 깊게 앉기 어려움",
        "question_ko": "발뒤꿈치를 바닥에 붙인 채 편하게 깊이 앉기 어렵나요?",
        "help_text_ko": "통증이 없다면 동작 수행의 어려움만 선택합니다. 통증이 있으면 이 항목 대신 부상 부위를 선택합니다.",
        "input_stage": "profileOptional",
        "user_selectable": 1,
        "default_selected": 0,
        "safety_authority": "deterministicConstraint",
        "legacy_internal_code": "limited_ankle_dorsiflexion",
        "context_requirement": "no_registered_injury_required",
        "unchecked_semantics": "noKnownDifficultyReported",
        "pain_during_task_action": "routeToRegisteredInjury",
    },
    {
        "code": "hip_hinge_difficult",
        "category": "functionalMovement",
        "ui_section": "movementChecks",
        "display_order": 20,
        "label_ko": "엉덩이를 접어 숙이기 어려움",
        "question_ko": "무릎을 크게 굽히지 않고 엉덩이를 뒤로 빼며 숙이기 어렵나요?",
        "help_text_ko": "손이 바닥에 닿는지는 사용하지 않습니다. 통증이 있으면 이 항목 대신 부상 부위를 선택합니다.",
        "input_stage": "profileOptional",
        "user_selectable": 1,
        "default_selected": 0,
        "safety_authority": "deterministicConstraint",
        "legacy_internal_code": "limited_hip_flexion",
        "context_requirement": "no_registered_injury_required",
        "unchecked_semantics": "noKnownDifficultyReported",
        "pain_during_task_action": "routeToRegisteredInjury",
    },
    {
        "code": "overhead_reach_difficult",
        "category": "functionalMovement",
        "ui_section": "movementChecks",
        "display_order": 30,
        "label_ko": "팔을 머리 위로 올리기 어려움",
        "question_ko": "허리를 과하게 젖히지 않고 팔을 머리 위로 편하게 올리기 어렵나요?",
        "help_text_ko": "어깨 각도나 질환을 자가진단하지 않습니다. 통증이 있으면 이 항목 대신 부상 부위를 선택합니다.",
        "input_stage": "profileOptional",
        "user_selectable": 1,
        "default_selected": 0,
        "safety_authority": "deterministicConstraint",
        "legacy_internal_code": "limited_shoulder_flexion",
        "context_requirement": "no_registered_injury_required",
        "unchecked_semantics": "noKnownDifficultyReported",
        "pain_during_task_action": "routeToRegisteredInjury",
    },
    {
        "code": "loaded_wrist_extension_difficult",
        "category": "functionalMovement",
        "ui_section": "movementChecks",
        "display_order": 40,
        "label_ko": "손목을 뒤로 젖혀 버티기 어려움",
        "question_ko": "손바닥을 짚거나 손목을 뒤로 젖힌 상태에서 부하를 버티기 어렵나요?",
        "help_text_ko": "통증이 없다면 부하가 실린 손목 자세의 어려움만 선택합니다. 통증이 있으면 부상 부위를 선택합니다.",
        "input_stage": "profileOptional",
        "user_selectable": 1,
        "default_selected": 0,
        "safety_authority": "deterministicConstraint",
        "legacy_internal_code": "limited_wrist_extension",
        "context_requirement": "no_registered_injury_required",
        "unchecked_semantics": "noKnownDifficultyReported",
        "pain_during_task_action": "routeToRegisteredInjury",
    },
    {
        "code": "self_reported_long_arms",
        "category": "selfReportedAnthropometry",
        "ui_section": "bodyProportionChecks",
        "display_order": 110,
        "label_ko": "팔이 유난히 긴 편",
        "question_ko": "비슷한 키의 사람과 비교할 때 팔이 유난히 긴 편인가요?",
        "help_text_ko": "정밀 측정값이 아닌 자가 보고 신호이며, 운동 금지가 아니라 검수된 세팅과 후보 순위만 약하게 보조합니다.",
        "input_stage": "profileOptional",
        "user_selectable": 1,
        "default_selected": 0,
        "safety_authority": "advisoryOnly",
        "legacy_internal_code": "long_arm",
        "context_requirement": "none",
        "unchecked_semantics": "notSelfReportedAsNotable",
        "pain_during_task_action": "notApplicable",
    },
    {
        "code": "self_reported_long_thighs",
        "category": "selfReportedAnthropometry",
        "ui_section": "bodyProportionChecks",
        "display_order": 120,
        "label_ko": "허벅지 구간이 유난히 긴 편",
        "question_ko": "다리에서 종아리보다 허벅지 구간이 유난히 긴 편인가요?",
        "help_text_ko": "대퇴·경골 길이를 재지 않은 자가 보고 신호이며, 운동 금지가 아니라 검수된 세팅과 후보 순위만 약하게 보조합니다.",
        "input_stage": "profileOptional",
        "user_selectable": 1,
        "default_selected": 0,
        "safety_authority": "advisoryOnly",
        "legacy_internal_code": "long_femur",
        "context_requirement": "none",
        "unchecked_semantics": "notSelfReportedAsNotable",
        "pain_during_task_action": "notApplicable",
    },
)


_DEFINITION_BY_CODE = {item["code"]: item for item in PROFILE_SIGNAL_DEFINITIONS}
_ORDER_BY_CODE = {item["code"]: item["display_order"] for item in PROFILE_SIGNAL_DEFINITIONS}
PROFILE_SIGNAL_CODES = frozenset(_DEFINITION_BY_CODE)


@dataclass(frozen=True)
class ResolvedProfileConstraintSignals:
    selected_signals: tuple[str, ...]
    movement_signals: tuple[str, ...]
    body_proportion_signals: tuple[str, ...]
    mobility_limit_aliases: tuple[str, ...]
    anthropometry_signal_aliases: tuple[str, ...]

    def to_dict(self) -> dict[str, list[str]]:
        return {key: list(value) for key, value in asdict(self).items()}


def resolve_profile_constraint_signals(
    raw_signals: Iterable[str] | None,
) -> ResolvedProfileConstraintSignals:
    """Reject unknown values, deduplicate, and return stable contract order."""
    if raw_signals is None:
        values: list[str] = []
    elif isinstance(raw_signals, (str, bytes, dict)):
        raise ValueError("profile constraint signals must be an array of registered codes")
    else:
        values = []
        for raw in raw_signals:
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError("profile constraint signals must be non-empty strings")
            values.append(raw.strip())

    unknown = sorted(set(values) - PROFILE_SIGNAL_CODES)
    if unknown:
        raise ValueError(f"unknown profile constraint signals: {', '.join(unknown)}")

    selected = tuple(sorted(set(values), key=_ORDER_BY_CODE.__getitem__))
    movement = tuple(
        code for code in selected
        if _DEFINITION_BY_CODE[code]["category"] == "functionalMovement"
    )
    body_proportion = tuple(
        code for code in selected
        if _DEFINITION_BY_CODE[code]["category"] == "selfReportedAnthropometry"
    )
    mobility_aliases = tuple(_DEFINITION_BY_CODE[code]["legacy_internal_code"] for code in movement)
    anthropometry_aliases = tuple(
        _DEFINITION_BY_CODE[code]["legacy_internal_code"] for code in body_proportion
    )
    return ResolvedProfileConstraintSignals(
        selected_signals=selected,
        movement_signals=movement,
        body_proportion_signals=body_proportion,
        mobility_limit_aliases=mobility_aliases,
        anthropometry_signal_aliases=anthropometry_aliases,
    )


def build_profile_input_manifest() -> dict[str, Any]:
    """Return the versioned FE/BE manifest without exposing internal rule weights."""
    sections = []
    for section_code, title_ko, description_ko in (
        (
            "movementChecks",
            "동작 수행 확인",
            "통증이 없는 상태에서 동작이 편하게 가능한지만 선택합니다. 통증이 있으면 부상 부위 설정을 사용합니다.",
        ),
        (
            "bodyProportionChecks",
            "체형 특징",
            "본인이 뚜렷하게 인지하는 경우에만 선택합니다. 선택값은 운동 금지가 아닌 세팅과 후보 순위 보조에만 사용합니다.",
        ),
    ):
        options = []
        for item in PROFILE_SIGNAL_DEFINITIONS:
            if item["ui_section"] != section_code:
                continue
            options.append({
                "code": item["code"],
                "labelKo": item["label_ko"],
                "questionKo": item["question_ko"],
                "helpTextKo": item["help_text_ko"],
                "displayOrder": item["display_order"],
                "defaultSelected": bool(item["default_selected"]),
                "uncheckedSemantics": item["unchecked_semantics"],
                "painDuringTaskAction": item["pain_during_task_action"],
                "decisionAuthority": item["safety_authority"],
            })
        sections.append({
            "code": section_code,
            "titleKo": title_ko,
            "descriptionKo": description_ko,
            "selectionType": "multiCheckbox",
            "options": options,
        })

    return {
        "version": PROFILE_INPUT_CONTRACT_VERSION,
        "defaultBehavior": "allUnchecked",
        "emptyInputMeaning": "noConstraintOrNotableProportionReported",
        "priorityOrder": [
            "registeredInjury",
            "functionalMovement",
            "selfReportedAnthropometry",
            "userPreference",
            "llmSelection",
        ],
        "unknownCodeAction": "reject",
        "freeTextSafetyAuthority": "none",
        "sections": sections,
    }

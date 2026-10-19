"""Deterministic explanation policy for intentionally short routines."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

from ..registry import LARGE_MUSCLE_SLUGS
from .adjustments import (
    minimum_productive_duration_min,
    productive_session_reference_min,
    productive_time_utilization_pct,
    target_productive_duration_min,
)


UNDERFILL_POLICY_VERSION = "routine-underfill-v2-2026-09-14"


def planned_volume_complete_for_request(
    *,
    goal: str | None,
    experience_level: str | None,
    target_muscles: list[str] | tuple[str, ...],
    exercise_count: int,
    target_exercise_count: int,
) -> bool:
    """Return whether a bounded plan should not be padded to consume time."""
    targets = {
        str(muscle or "").strip().lower().replace("_", "-")
        for muscle in target_muscles
        if str(muscle or "").strip()
    }
    completed = int(exercise_count) >= max(1, int(target_exercise_count))
    if not completed or not targets:
        return False
    beginner_single_target_strength = (
        str(goal or "").strip().lower().replace("_", "") == "strength"
        and str(experience_level or "").strip().lower() == "beginner"
        and len(targets) == 1
    )
    small_muscle_only = not bool(targets & LARGE_MUSCLE_SLUGS)
    return beginner_single_target_strength or small_muscle_only


@dataclass(frozen=True)
class UnderfillDecision:
    status: str
    reason_code: Optional[str]
    actual_duration_min: int
    available_duration_min: int
    reference_duration_min: int
    minimum_productive_duration_min: int
    target_productive_duration_min: int
    utilization_pct: float
    planned_volume_complete: bool = False
    role_set_capacity_exhausted: bool = False
    user_summary_ko: Optional[str] = None
    user_option_ko: Optional[str] = None

    @property
    def underfilled(self) -> bool:
        return self.status == "underfilled"

    @property
    def severe_underfill(self) -> bool:
        return self.underfilled and self.utilization_pct < 50.0

    def to_diagnostics(self) -> dict[str, Any]:
        return {
            "policy_version": UNDERFILL_POLICY_VERSION,
            **asdict(self),
            "underfilled": self.underfilled,
            "severe_underfill": self.severe_underfill,
        }


_REASON_COPY: dict[str, tuple[str, str]] = {
    "registered_injury_constraint": (
        "등록된 부상 부위와 겹치는 운동을 제외한 결과입니다.",
        "이 계획을 그대로 진행하거나, 등록된 제한과 겹치지 않는 다른 목표 부위를 선택해 다시 생성할 수 있습니다.",
    ),
    "severe_doms_constraint": (
        "회복이 필요한 3단계 근육통 부위의 운동을 제외한 결과입니다.",
        "이 계획을 그대로 진행하거나, 3단계 근육통과 겹치지 않는 다른 목표 부위를 선택해 다시 생성할 수 있습니다.",
    ),
    "recovery_constraint": (
        "표시한 근육통을 반영해 해당 운동의 세트 수와 강도를 보수적으로 조정한 결과입니다.",
        "이 계획을 그대로 진행하거나, 근육통과 겹치지 않는 다른 목표 부위를 선택해 다시 생성할 수 있습니다.",
    ),
    "low_readiness": (
        "오늘 컨디션을 반영해 세트 수와 강도를 조절한 결과입니다.",
        "이 계획을 그대로 진행하거나, 현재 컨디션에 맞는 다른 목표 부위를 선택해 다시 생성할 수 있습니다.",
    ),
    "condition_policy_constraint": (
        "등록된 수술·질환 관련 제한 조건을 우선한 결과입니다.",
        "이 계획을 그대로 진행하거나, 등록된 제한을 충족하는 다른 목표 부위를 선택해 다시 생성할 수 있습니다.",
    ),
    "equipment_shortage": (
        "현재 사용 가능한 장비 조건에서 적합한 운동을 더 추가하기 어려운 결과입니다.",
        "이 계획을 그대로 진행하거나, 사용 가능한 장비 또는 허용된 목표 부위를 바꿔 다시 생성할 수 있습니다.",
    ),
    "candidate_policy_exhausted": (
        "중복과 현재 루틴 조건을 지키면서 추가할 적합한 운동이 없는 결과입니다.",
        "이 계획을 그대로 진행하거나, 허용된 목표 부위 또는 보조 목적을 조정해 다시 생성할 수 있습니다.",
    ),
    "candidate_composition_limit": (
        "현재 후보 구성에서 중복과 동작 패턴 상한을 지키며 최소 운동 수를 채우지 못한 결과입니다.",
        "이 계획을 그대로 진행하거나, 허용된 목표 부위 또는 보조 목적을 조정해 다시 생성할 수 있습니다.",
    ),
    "candidate_shortage": (
        "현재 입력 조건을 모두 만족하는 운동 후보가 부족한 결과입니다.",
        "이 계획을 그대로 진행하거나, 장비 또는 허용된 목표 부위 조건을 조정해 다시 생성할 수 있습니다.",
    ),
    "role_set_capacity_exhausted": (
        "선택한 운동의 역할별 세트 상한에 도달해 현재 분량에서 마쳤습니다. 남은 시간을 채우려고 같은 부위의 세트를 더 늘리지는 않았습니다.",
        "이 계획을 그대로 진행하거나, 허용된 목표 부위를 다시 선택해 생성할 수 있습니다.",
    ),
    "fallback_baseline": (
        "AI 응답 대신 검증된 기본 후보와 기존 운동량 상한으로 구성한 결과입니다.",
        "이 계획을 그대로 진행하거나, 허용된 목표 부위를 다시 선택해 생성할 수 있습니다.",
    ),
    "unexplained_underfill": (
        "현재 입력 조건을 유지하면서 적합한 운동을 더 추가하지 못한 결과입니다.",
        "이 계획을 그대로 진행하거나, 허용된 목표 부위·장비·컨디션 입력을 조정해 다시 생성할 수 있습니다.",
    ),
}


def _resolve_reason_code(
    *,
    readiness_level: str,
    injury_limited: bool,
    severe_doms_limited: bool,
    recovery_limited: bool,
    condition_policy_limited: bool,
    equipment_constrained: bool,
    candidate_count: int,
    minimum_exercise_count: int,
    fallback_used: bool,
    selection_diagnostics: Mapping[str, Any],
) -> str:
    if injury_limited:
        return "registered_injury_constraint"
    if severe_doms_limited:
        return "severe_doms_constraint"
    if recovery_limited:
        return "recovery_constraint"
    if str(readiness_level or "normal").strip().lower() == "low":
        return "low_readiness"
    if condition_policy_limited:
        return "condition_policy_constraint"
    if equipment_constrained:
        return "equipment_shortage"
    if (
        fallback_used
        and bool(selection_diagnostics.get("fallback_selection_measured", False))
        and bool(selection_diagnostics.get("fallback_composition_limit_observed", False))
        and int(selection_diagnostics.get("fallback_selection_shortfall") or 0) > 0
    ):
        return "candidate_composition_limit"
    if bool(selection_diagnostics.get("productive_fill_exhausted", False)):
        return "candidate_policy_exhausted"
    if 0 <= int(candidate_count) < max(1, int(minimum_exercise_count)):
        return "candidate_shortage"
    if bool(selection_diagnostics.get("role_set_capacity_exhausted", False)):
        return "role_set_capacity_exhausted"
    if fallback_used:
        return "fallback_baseline"
    return "unexplained_underfill"


def resolve_underfill_decision(
    *,
    actual_duration_min: int,
    available_duration_min: int,
    has_routine_blocks: bool,
    readiness_level: str = "normal",
    injury_limited: bool = False,
    severe_doms_limited: bool = False,
    recovery_limited: bool = False,
    condition_policy_limited: bool = False,
    equipment_constrained: bool = False,
    candidate_count: int = 0,
    minimum_exercise_count: int = 1,
    fallback_used: bool = False,
    planned_volume_complete: bool = False,
    selection_diagnostics: Optional[Mapping[str, Any]] = None,
) -> UnderfillDecision:
    actual = max(0, int(actual_duration_min))
    available = max(1, int(available_duration_min))
    reference = productive_session_reference_min(available)
    floor = minimum_productive_duration_min(available)
    target = target_productive_duration_min(available)
    utilization = productive_time_utilization_pct(actual, available)

    if not has_routine_blocks:
        status = "empty"
    elif actual >= target:
        status = "within_target"
    elif actual >= floor:
        status = "within_floor"
    elif planned_volume_complete and utilization >= 50.0:
        # Available time is an upper bound. A completed bounded plan should not
        # be padded with redundant work merely to consume the clock.
        status = "within_plan"
    else:
        status = "underfilled"

    reason_code: Optional[str] = None
    summary: Optional[str] = None
    option: Optional[str] = None
    if status == "underfilled":
        reason_code = _resolve_reason_code(
            readiness_level=readiness_level,
            injury_limited=injury_limited,
            severe_doms_limited=severe_doms_limited,
            recovery_limited=recovery_limited,
            condition_policy_limited=condition_policy_limited,
            equipment_constrained=equipment_constrained,
            candidate_count=candidate_count,
            minimum_exercise_count=minimum_exercise_count,
            fallback_used=fallback_used,
            selection_diagnostics=selection_diagnostics or {},
        )
        reason_summary, option = _REASON_COPY[reason_code]
        summary = (
            f"요청하신 {available}분 중 조건을 반영해 선택된 계획은 예상 {actual}분입니다. "
            f"{reason_summary}"
        )

    return UnderfillDecision(
        status=status,
        reason_code=reason_code,
        actual_duration_min=actual,
        available_duration_min=available,
        reference_duration_min=reference,
        minimum_productive_duration_min=floor,
        target_productive_duration_min=target,
        utilization_pct=utilization,
        planned_volume_complete=planned_volume_complete,
        role_set_capacity_exhausted=bool(
            (selection_diagnostics or {}).get("role_set_capacity_exhausted", False)
        ),
        user_summary_ko=summary,
        user_option_ko=option,
    )

"""Conservative, confirmation-only routine personalization suggestions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional, Sequence

from .schemas import RecentSetRecord


_MUSCLE_FOCUS_CODES = {
    "quadriceps": "quadricepsEmphasis",
    "hamstring": "hamstringEmphasis",
    "gluteal": "gluteEmphasis",
    "adductor": "adductorEmphasis",
    "abductors": "abductorEmphasis",
    "calves": "calvesEmphasis",
    "trapezius": "trapeziusEmphasis",
    "biceps": "bicepsEmphasis",
    "triceps": "tricepsEmphasis",
    "forearm": "forearmEmphasis",
    "front-deltoids": "frontDeltoidEmphasis",
    "side-deltoids": "sideDeltoidEmphasis",
    "back-deltoids": "rearDeltoidEmphasis",
}

_MUSCLE_LABELS_KO = {
    "quadriceps": "대퇴사두",
    "hamstring": "햄스트링",
    "gluteal": "둔근",
    "adductor": "내전근",
    "abductors": "외전근",
    "calves": "종아리",
    "trapezius": "승모근",
    "biceps": "이두",
    "triceps": "삼두",
    "forearm": "전완",
    "front-deltoids": "전면 어깨",
    "side-deltoids": "측면 어깨",
    "back-deltoids": "후면 어깨",
}


@dataclass(frozen=True)
class SpecializationSuggestion:
    focus_code: str
    reason_ko: str
    evidence_set_count: int
    comparison_max_set_count: int
    requires_confirmation: bool = True


def suggest_specialization_from_recent_sets(
    recent_sets: Sequence[RecentSetRecord],
    *,
    target_muscles: Sequence[str],
    experience_level: Optional[str],
    explicit_focus: Optional[str],
) -> Optional[SpecializationSuggestion]:
    """Suggest one under-covered direct target without silently applying it.

    The current runtime history contains recent working sets but not enough
    longitudinal detail to diagnose a plateau or a precise sticking point.
    This rule therefore reports only a clear recent direct-set imbalance.
    """

    if explicit_focus:
        return None
    if str(experience_level or "").strip().lower() != "advanced":
        return None

    eligible_targets = [
        str(target).strip()
        for target in target_muscles
        if str(target).strip() in _MUSCLE_FOCUS_CODES
    ]
    if len(eligible_targets) < 2:
        return None

    counts = Counter(
        str(record.primary_muscle or "").strip()
        for record in recent_sets
        if str(record.primary_muscle or "").strip() in eligible_targets
    )
    total_direct_sets = sum(counts[target] for target in eligible_targets)
    if total_direct_sets < 8:
        return None

    lowest_target = min(eligible_targets, key=lambda target: (counts[target], target))
    lowest_count = counts[lowest_target]
    highest_count = max(counts[target] for target in eligible_targets)
    if highest_count - lowest_count < 4:
        return None
    if highest_count > 0 and lowest_count / highest_count > 0.5:
        return None

    label = _MUSCLE_LABELS_KO[lowest_target]
    return SpecializationSuggestion(
        focus_code=_MUSCLE_FOCUS_CODES[lowest_target],
        reason_ko=(
            f"최근 기록에서 {label} 직접 세트가 함께 선택한 다른 부위보다 적었습니다. "
            "이번 세션에서 우선 집중할지 확인이 필요합니다."
        ),
        evidence_set_count=lowest_count,
        comparison_max_set_count=highest_count,
    )

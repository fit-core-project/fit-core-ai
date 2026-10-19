"""Canonical user-facing exercise-name selection."""

from __future__ import annotations

import re
from typing import Any, Mapping


_GENERIC_DISPLAY_NAMES = {
    "컬",
    "밴드컬",
    "덤벨컬",
    "케이블컬",
    "머신컬",
    "curl",
    "프레스",
    "press",
    "로우",
    "row",
    "레이즈",
    "raise",
    "익스텐션",
    "extension",
}

_USER_FACING_EXERCISE_ALIASES = {
    # Keep a searchable standard exercise name while explaining the movement
    # in Korean. This row is the catalog's "Thread the Needle" variation.
    "V24ROW-023032": "스레드 더 니들(네발기기 흉추 회전)",
}

_MEANINGFUL_VARIATION_TOKENS = (
    "와이드",
    "스모",
    "고블릿",
    "불가리안",
    "스플릿",
    "프론트",
    "백",
    "인클라인",
    "디클라인",
    "플랫",
    "해머",
    "크로스바디",
    "프리처",
    "스콧",
    "스파이더",
    "드래그",
    "베이지안",
    "리버스",
    "정지",
    "wide",
    "sumo",
    "goblet",
    "bulgarian",
    "split",
    "front",
    "back",
    "incline",
    "decline",
    "flat",
    "hammer",
    "crossbody",
    "preacher",
    "scott",
    "spider",
    "drag",
    "bayesian",
    "reverse",
    "paused",
    "pause",
)


def _compact(value: Any) -> str:
    return re.sub(r"[\s/_()·,.-]+", "", str(value or "").strip().lower())


def is_generic_candidate_display_text(value: Any) -> bool:
    return _compact(value) in {_compact(token) for token in _GENERIC_DISPLAY_NAMES}


def candidate_display_name(candidate: Mapping[str, Any]) -> str:
    """Prefer concise names unless they hide a meaningful exercise variation."""
    exercise_id = str(candidate.get("id") or "").strip()
    if exercise_id in _USER_FACING_EXERCISE_ALIASES:
        return _USER_FACING_EXERCISE_ALIASES[exercise_id]

    primary = str(candidate.get("primary_muscle") or "").strip().lower()
    user_facing = str(candidate.get("user_facing_display_name_kr") or "").strip()
    detailed = str(candidate.get("display_name_ko") or candidate.get("name_kr") or "").strip()
    source_name = str(candidate.get("name_en") or "").lower()
    base_name = user_facing or detailed
    if primary == "calves" and "카프 레이즈" in base_name:
        if not any(token in base_name for token in ("시티드", "스탠딩", "앉아", "서서")):
            for token, label in (("seated", "시티드"), ("standing", "스탠딩")):
                if re.search(rf"\b{token}\b", source_name):
                    return base_name.replace("카프 레이즈", f"{label} 카프 레이즈")

    # Default routine rows are system-curated representatives of a familiar
    # public exercise concept. Human familiarity review remains a separate gate.
    # Their source row may still contain catalog-only stance or grip detail,
    # which belongs on the encyclopedia detail page rather than in the title.
    if candidate.get("main_routine_default_allowed") in {True, 1, "1"} and user_facing:
        return user_facing

    # Advanced training methods are displayed as a separate instruction on the
    # routine block. Keep the title recognizable instead of turning the catalog
    # variation string into a new exercise name.
    if candidate.get("method_modifier_separate") and user_facing:
        return user_facing

    if (
        primary == "back-deltoids"
        and user_facing
        and detailed
        and ("펙덱" in user_facing or "pec" in user_facing.lower())
        and not any(
            token in user_facing.lower()
            for token in ("리버스", "리어", "후면", "reverse", "rear")
        )
        and any(
            token in detailed.lower()
            for token in ("리버스", "리어", "후면", "reverse", "rear")
        )
    ):
        return detailed

    if user_facing and detailed and is_generic_candidate_display_text(user_facing):
        return detailed

    if user_facing and detailed:
        user_facing_lower = user_facing.lower()
        detailed_lower = detailed.lower()
        if any(
            token in detailed_lower and token not in user_facing_lower
            for token in _MEANINGFUL_VARIATION_TOKENS
        ):
            return detailed

    return str(
        candidate.get("user_facing_display_name_kr")
        or candidate.get("display_name_ko")
        or candidate.get("base_exercise_name_kr")
        or candidate.get("name_kr")
        or candidate.get("name_en")
        or candidate.get("id")
        or ""
    ).strip()

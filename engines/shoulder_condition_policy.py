"""Deterministic shoulder-condition policy for routine candidate filtering.

The public product adapter is not locked to these fields yet. The AI core
accepts the closed values here so the policy can be completed and verified
before FE/BE integration work starts.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional


SHOULDER_CONDITION_STATUSES = (
    "activeUncleared",
    "managedWithRestrictions",
    "managedNoRestrictions",
    "recovered",
)
SHOULDER_RESTRICTED_MOVEMENT_PATTERNS = (
    "overheadPosition",
    "horizontalPush",
    "horizontalPull",
    "armRaise",
    "behindBackRotation",
    "unknown",
)

_ACTIVE_UNCLEARED = "activeUncleared"
_MANAGED_WITH_RESTRICTIONS = "managedWithRestrictions"

_UPPER_BODY_SAFE_STOP_SPLITS = {
    "push",
    "pull",
    "upper",
    "chest-back",
    "chest",
    "back",
    "shoulder",
}
_SHOULDER_DEPENDENT_TARGETS = {
    "chest",
    "upper-back",
    "front-deltoids",
    "back-deltoids",
    "side-deltoids",
}
_SHOULDER_DIRECT_MUSCLES = {
    "front-deltoids",
    "back-deltoids",
    "side-deltoids",
}
_SHOULDER_COMPOUND_MUSCLES = {
    "chest",
    "upper-back",
    "trapezius",
    *_SHOULDER_DIRECT_MUSCLES,
}
_HORIZONTAL_PUSH_MUSCLES = {"chest", "triceps", "front-deltoids"}
_HORIZONTAL_PULL_MUSCLES = {
    "upper-back",
    "back-deltoids",
    "trapezius",
    "biceps",
}

_OVERHEAD_NAME_TOKENS = {
    "overhead",
    "오버헤드",
    "militarypress",
    "밀리터리프레스",
    "shoulderpress",
    "숄더프레스",
    "pushpress",
    "푸시프레스",
    "pulldown",
    "풀다운",
    "pullup",
    "풀업",
    "chinup",
    "친업",
    "snatch",
    "스내치",
    "jerk",
    "저크",
}
_EXPLICIT_OVERHEAD_POSITION_NAME_TOKENS = {
    "overhead",
    "오버헤드",
    "머리위",
}
_HORIZONTAL_PUSH_NAME_TOKENS = {
    "benchpress",
    "벤치프레스",
    "chestpress",
    "체스트프레스",
    "pushup",
    "푸시업",
    "푸쉬업",
    "pecdeck",
    "펙덱",
    "butterfly",
    "버터플라이",
    "crossover",
    "크로스오버",
}
_HORIZONTAL_PULL_NAME_TOKENS = {
    "reversefly",
    "리버스플라이",
    "reversedfly",
    "rearfly",
    "리어플라이",
    "facepull",
    "페이스풀",
}
_ARM_RAISE_NAME_TOKENS = {
    "frontraise",
    "프론트레이즈",
    "lateralraise",
    "레터럴레이즈",
    "rearraise",
    "리어레이즈",
    "scaption",
    "스캐프션",
    "uprightrow",
    "업라이트로우",
    "yraise",
    "traise",
    "wraise",
    "luraise",
}
_BEHIND_BACK_NAME_TOKENS = {
    "behindback",
    "비하인드백",
    "behindtheback",
    "behindneck",
    "비하인드넥",
    "kickback",
    "킥백",
}
_SHOULDER_RACK_POSITION_NAME_TOKENS = {
    "backsquat",
    "백스쿼트",
    "lowbarsquat",
    "로우바스쿼트",
    "highbarsquat",
    "하이바스쿼트",
    "frontsquat",
    "프론트스쿼트",
}
_SHOULDER_WEIGHT_BEARING_NAME_TOKENS = {
    "plank",
    "플랭크",
    "bearcrawl",
    "베어크롤",
    "mountainclimber",
    "마운틴클라이머",
    "quadruped",
    "쿼드러펫",
    "bird-dog",
    "birddog",
    "버드독",
}


def _normalize_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _compact(value: Any) -> str:
    return re.sub(r"[\s_\-/]+", "", str(value or "").strip().lower())


def _candidate_names(candidate: dict) -> str:
    return " ".join(
        _compact(candidate.get(key))
        for key in (
            "name_kr",
            "name_en",
            "display_name_ko",
            "base_exercise_name_kr",
            "variation_name_kr",
            "user_facing_display_name_kr",
            "movement_pattern",
        )
    )


def _candidate_raw_names(candidate: dict) -> str:
    return " ".join(
        str(candidate.get(key) or "").strip().lower()
        for key in (
            "name_kr",
            "name_en",
            "display_name_ko",
            "base_exercise_name_kr",
            "variation_name_kr",
            "user_facing_display_name_kr",
        )
    )


def _has_row_name(candidate: dict) -> bool:
    raw_names = _candidate_raw_names(candidate)
    compact_names = _compact(raw_names)
    if re.search(r"(?<![a-z])row(?![a-z])", raw_names):
        return True
    korean_compact = compact_names.replace("스로우", "").replace("플로우", "")
    return "로우" in korean_compact


def _has_dip_name(candidate: dict) -> bool:
    raw_names = _candidate_raw_names(candidate)
    if re.search(r"(?<![a-z])dips?(?![a-z])", raw_names):
        return True
    for match in re.finditer(r"(?<![가-힣])딥스?(?![가-힣])", raw_names):
        # "딥 스쿼트" is a depth cue, not a dip exercise.
        if not re.match(r"\s*스쿼트", raw_names[match.end():]):
            return True
    return False


def _candidate_muscles(candidate: dict) -> set[str]:
    values: set[str] = set()
    for key in (
        "primary_muscle",
        "secondary_muscle",
        "canonical_primary_muscle",
        "canonical_secondary_muscles",
        "catalog_source_primary_muscle",
        "catalog_source_secondary_muscle",
    ):
        raw = candidate.get(key)
        if isinstance(raw, list):
            parts = raw
        else:
            parts = str(raw or "").replace(";", ",").split(",")
        values.update(
            _normalize_token(part)
            for part in parts
            if str(part or "").strip()
        )
    return values


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _level_at_least_medium(value: Any) -> bool:
    return str(value or "").strip().lower() in {"medium", "high", "elite"}


def normalize_restricted_movement_patterns(values: Iterable[str]) -> tuple[str, ...]:
    """Return unique values in the locked product order."""

    selected = {str(value or "").strip() for value in values if str(value or "").strip()}
    return tuple(
        pattern
        for pattern in SHOULDER_RESTRICTED_MOVEMENT_PATTERNS
        if pattern in selected
    )


def validate_shoulder_condition_combination(
    status: Optional[str],
    restricted_patterns: Iterable[str],
) -> tuple[str, ...]:
    """Validate the closed cross-field contract and return canonical patterns."""

    raw_patterns = [str(value or "").strip() for value in restricted_patterns]
    patterns = normalize_restricted_movement_patterns(raw_patterns)
    if len(raw_patterns) != len(set(raw_patterns)):
        raise ValueError("restrictedMovementPatterns must not contain duplicate values")
    if any(value not in SHOULDER_RESTRICTED_MOVEMENT_PATTERNS for value in raw_patterns):
        raise ValueError("restrictedMovementPatterns contains an unsupported value")
    if status is not None and status not in SHOULDER_CONDITION_STATUSES:
        raise ValueError("shoulderConditionStatus contains an unsupported value")
    if status == _MANAGED_WITH_RESTRICTIONS and not patterns:
        raise ValueError(
            "managedWithRestrictions requires at least one restrictedMovementPatterns value"
        )
    if status != _MANAGED_WITH_RESTRICTIONS and patterns:
        raise ValueError(
            "restrictedMovementPatterns is allowed only with managedWithRestrictions"
        )
    if "unknown" in patterns and len(patterns) != 1:
        raise ValueError("unknown must be the only restrictedMovementPatterns value")
    return patterns


def resolve_shoulder_condition_context(
    request_status: Optional[str],
    request_restricted_patterns: Iterable[str],
    profile_status: Optional[str],
    profile_restricted_patterns: Iterable[str],
) -> tuple[Optional[str], tuple[str, ...]]:
    """Merge request and profile state without allowing a weaker override.

    The persistent profile is the normal source, while request values support
    internal scenarios and a future adapter. When both exist, the strictest
    deterministic interpretation wins.
    """

    request_patterns = validate_shoulder_condition_combination(
        request_status,
        request_restricted_patterns,
    )
    profile_patterns = validate_shoulder_condition_combination(
        profile_status,
        profile_restricted_patterns,
    )

    statuses = {status for status in (request_status, profile_status) if status}
    if _ACTIVE_UNCLEARED in statuses:
        return _ACTIVE_UNCLEARED, ()

    if _MANAGED_WITH_RESTRICTIONS in statuses:
        combined = {*request_patterns, *profile_patterns}
        if "unknown" in combined:
            return _MANAGED_WITH_RESTRICTIONS, ("unknown",)
        return (
            _MANAGED_WITH_RESTRICTIONS,
            normalize_restricted_movement_patterns(combined),
        )

    if "managedNoRestrictions" in statuses:
        return "managedNoRestrictions", ()
    if "recovered" in statuses:
        return "recovered", ()
    return None, ()


def active_uncleared_requires_safe_stop(
    status: Optional[str],
    target_split_label: Optional[str],
    target_muscles: Iterable[str],
) -> bool:
    """Return whether the requested target itself requires shoulder-loaded work."""

    if status != _ACTIVE_UNCLEARED:
        return False
    split = _normalize_token(target_split_label)
    if split in _UPPER_BODY_SAFE_STOP_SPLITS:
        return True
    targets = {
        _normalize_token(target)
        for target in target_muscles
        if str(target or "").strip()
    }
    return bool(targets & _SHOULDER_DEPENDENT_TARGETS)


def candidate_shoulder_movement_patterns(candidate: dict) -> frozenset[str]:
    """Classify shoulder-loaded movement patterns from reviewed metadata first."""

    patterns: set[str] = set()
    compact_names = _candidate_names(candidate)
    explicit_pattern = _compact(candidate.get("movement_pattern"))
    plane = str(candidate.get("shoulder_elevation_plane") or "").strip().lower()
    muscles = _candidate_muscles(candidate)
    shoulder_context = bool(muscles & _SHOULDER_COMPOUND_MUSCLES) or any(
        _level_at_least_medium(candidate.get(field))
        for field in (
            "shoulder_elevation_demand",
            "shoulder_flexion_demand",
            "shoulder_impingement_risk",
        )
    )

    if (
        _is_true(candidate.get("overhead_position_required"))
        or plane == "overhead"
        or "overheadposition" in explicit_pattern
        or any(
            token in compact_names
            for token in _EXPLICIT_OVERHEAD_POSITION_NAME_TOKENS
        )
        or (
            shoulder_context
            and any(token in compact_names for token in _OVERHEAD_NAME_TOKENS)
        )
    ):
        patterns.add("overheadPosition")

    horizontal_push = (
        "horizontalpush" in explicit_pattern
        or (
            bool(muscles & _HORIZONTAL_PUSH_MUSCLES)
            and any(token in compact_names for token in _HORIZONTAL_PUSH_NAME_TOKENS)
        )
        or (
            plane == "horizontal"
            and "chest" in muscles
            and any(token in compact_names for token in {"press", "프레스", "fly", "플라이"})
        )
    )
    if horizontal_push:
        patterns.add("horizontalPush")

    horizontal_pull = (
        "horizontalpull" in explicit_pattern
        or (
            bool(muscles & _HORIZONTAL_PULL_MUSCLES)
            and (
                _has_row_name(candidate)
                or any(token in compact_names for token in _HORIZONTAL_PULL_NAME_TOKENS)
            )
        )
        or (
            plane == "horizontal"
            and bool(muscles & {"upper-back", "back-deltoids", "trapezius"})
        )
    )
    if horizontal_pull:
        patterns.add("horizontalPull")

    if (
        "armraise" in explicit_pattern
        or plane in {"frontal", "sagittal"}
        or any(token in compact_names for token in _ARM_RAISE_NAME_TOKENS)
        or (
            bool(muscles & _SHOULDER_COMPOUND_MUSCLES)
            and ("raise" in compact_names or "레이즈" in compact_names)
        )
    ):
        patterns.add("armRaise")

    if (
        "behindbackrotation" in explicit_pattern
        or _has_dip_name(candidate)
        or (
            shoulder_context
            and any(token in compact_names for token in _BEHIND_BACK_NAME_TOKENS)
        )
    ):
        patterns.add("behindBackRotation")

    if _candidate_has_shoulder_load_base(candidate) and not patterns:
        patterns.add("unknown")
    return frozenset(patterns)


def _candidate_has_shoulder_load_base(candidate: dict) -> bool:
    muscles = _candidate_muscles(candidate)
    movement_type = str(
        candidate.get("canonical_movement_type")
        or candidate.get("movement_type")
        or ""
    ).strip().upper()
    compact_names = _candidate_names(candidate)
    if muscles & _SHOULDER_DIRECT_MUSCLES:
        return True
    if (
        muscles & _SHOULDER_COMPOUND_MUSCLES
        and (
            movement_type == "COMPOUND"
            or any(
                token in compact_names
                for token in (
                    *_OVERHEAD_NAME_TOKENS,
                    *_HORIZONTAL_PUSH_NAME_TOKENS,
                    *_HORIZONTAL_PULL_NAME_TOKENS,
                    *_ARM_RAISE_NAME_TOKENS,
                )
            )
        )
    ):
        return True
    if any(token in compact_names for token in _SHOULDER_RACK_POSITION_NAME_TOKENS):
        return True
    if any(token in compact_names for token in _SHOULDER_WEIGHT_BEARING_NAME_TOKENS):
        return True
    return bool(
        muscles
        & {
            *_SHOULDER_COMPOUND_MUSCLES,
            "biceps",
            "triceps",
            "forearm",
        }
        and any(
            _level_at_least_medium(candidate.get(field))
            for field in (
                "shoulder_elevation_demand",
                "shoulder_flexion_demand",
                "shoulder_impingement_risk",
            )
        )
    )


def candidate_has_shoulder_load(
    candidate: dict,
    *,
    preclassified_patterns: Optional[set[str]] = None,
) -> bool:
    """Conservatively identify every known or unclassified shoulder load."""

    patterns = (
        set(candidate_shoulder_movement_patterns(candidate))
        if preclassified_patterns is None
        else set(preclassified_patterns)
    )
    return bool(patterns) or _candidate_has_shoulder_load_base(candidate)


def candidate_restricted_movement_overlap(
    candidate: dict,
    status: Optional[str],
    restricted_patterns: Iterable[str],
) -> frozenset[str]:
    """Return restriction codes that make this candidate ineligible."""

    if status not in {_ACTIVE_UNCLEARED, _MANAGED_WITH_RESTRICTIONS}:
        return frozenset()

    detected = set(candidate_shoulder_movement_patterns(candidate))
    has_shoulder_load = candidate_has_shoulder_load(
        candidate,
        preclassified_patterns=detected,
    )
    if status == _ACTIVE_UNCLEARED:
        return frozenset({"unknown"}) if has_shoulder_load else frozenset()
    selected = set(normalize_restricted_movement_patterns(restricted_patterns))
    if "unknown" in selected:
        return frozenset({"unknown"}) if has_shoulder_load else frozenset()

    overlap = detected & selected
    if overlap:
        return frozenset(overlap)
    if "unknown" in detected:
        return frozenset({"unknown"})
    return frozenset()


def shoulder_restriction_labels_ko(patterns: Iterable[str]) -> list[str]:
    labels = {
        "overheadPosition": "머리 위 동작",
        "horizontalPush": "수평 밀기",
        "horizontalPull": "수평 당기기",
        "armRaise": "팔 들어 올리기",
        "behindBackRotation": "팔을 등 뒤로 보내는 동작",
        "unknown": "제한 동작 미확인",
    }
    return [
        labels[pattern]
        for pattern in normalize_restricted_movement_patterns(patterns)
    ]

"""Deterministic managed-injury policy shared across body regions.

``painAreas`` remains the binary channel for an active registered injury.
This module handles a different case: a previously assessed condition whose
allowed movement families are known. It never diagnoses a condition or infers
medical clearance from free text.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Optional

from .injury_movement_metadata import effective_injury_movement_metadata
from .shoulder_condition_policy import (
    SHOULDER_CONDITION_STATUSES,
    SHOULDER_RESTRICTED_MOVEMENT_PATTERNS,
    candidate_has_shoulder_load,
    candidate_restricted_movement_overlap,
    candidate_shoulder_movement_patterns,
)


INJURY_CONDITION_STATUSES = SHOULDER_CONDITION_STATUSES
INJURY_AREA_PATTERN_ORDER: dict[str, tuple[str, ...]] = {
    "shoulder": SHOULDER_RESTRICTED_MOVEMENT_PATTERNS,
    "knee": (
        "deepKneeFlexion",
        "loadedKneeExtension",
        "loadedKneeFlexion",
        "singleLegWeightBearing",
        "impactJumpRun",
        "unknown",
    ),
    "lumbar": (
        "loadedSpinalFlexion",
        "hipHinge",
        "axialLoading",
        "unsupportedBentOver",
        "trunkRotation",
        "unknown",
    ),
    "wrist": (
        "loadedWristExtension",
        "fixedBarGrip",
        "loadedGrip",
        "pronationSupination",
        "weightBearingOnHands",
        "unknown",
    ),
    "ankle": (
        "deepDorsiflexion",
        "loadedPlantarFlexion",
        "singleLegStability",
        "impactJumpRun",
        "unknown",
    ),
    "hip": (
        "deepHipFlexion",
        "hipHinge",
        "loadedAbduction",
        "loadedAdduction",
        "hipRotation",
        "singleLegWeightBearing",
        "unknown",
    ),
    "elbow": (
        "loadedElbowFlexion",
        "loadedElbowExtension",
        "loadedGrip",
        "pronationSupination",
        "weightBearingOnArms",
        "unknown",
    ),
    "neck": (
        "directNeckLoading",
        "overheadAxialLoading",
        "sustainedUnsupportedHeadPosition",
        "impactBallistic",
        "unknown",
    ),
}
SUPPORTED_INJURY_CONDITION_AREAS = tuple(INJURY_AREA_PATTERN_ORDER)

_AREA_ALIASES = {
    "shoulder": "shoulder",
    "shoulders": "shoulder",
    "knee": "knee",
    "knees": "knee",
    "lumbar": "lumbar",
    "lower-back": "lumbar",
    "lowerback": "lumbar",
    "wrist": "wrist",
    "wrists": "wrist",
    "ankle": "ankle",
    "ankles": "ankle",
    "foot": "ankle",
    "feet": "ankle",
    "achilles": "ankle",
    "hip": "hip",
    "hips": "hip",
    "groin": "hip",
    "elbow": "elbow",
    "elbows": "elbow",
    "neck": "neck",
    "cervical": "neck",
}
_ACTIVE_UNCLEARED = "activeUncleared"
_MANAGED_WITH_RESTRICTIONS = "managedWithRestrictions"

_LOWER_BODY_MUSCLES = {
    "quadriceps",
    "hamstring",
    "gluteal",
    "calves",
    "adductor",
    "abductors",
    "soleus",
    "left-soleus",
    "right-soleus",
}
_UPPER_BODY_MUSCLES = {
    "chest",
    "upper-back",
    "lower-back",
    "trapezius",
    "front-deltoids",
    "side-deltoids",
    "back-deltoids",
    "biceps",
    "triceps",
    "forearm",
    "neck",
}
_KNEE_EXTENSION_TOKENS = {
    "squat",
    "스쿼트",
    "lunge",
    "런지",
    "legpress",
    "레그프레스",
    "legextension",
    "레그익스텐션",
    "stepup",
    "스텝업",
}
_KNEE_FLEXION_TOKENS = {
    "legcurl",
    "레그컬",
    "hamstringcurl",
    "햄스트링컬",
    "nordiccurl",
    "노르딕컬",
    "glutehamraise",
    "글루트햄레이즈",
}
_DEEP_KNEE_FLEXION_TOKENS = {
    "deepsquat",
    "딥스쿼트",
    "atgsquat",
    "atg스쿼트",
    "atgsplitsquat",
    "atg스플릿스쿼트",
    "ass-to-grass",
    "시시스쿼트",
}
_SINGLE_LEG_TOKENS = {
    "singleleg",
    "single-leg",
    "싱글레그",
    "한발",
    "한쪽다리",
    "unilateral",
    "bulgarian",
    "불가리안",
    "splitsquat",
    "스플릿스쿼트",
    "lunge",
    "런지",
    "stepup",
    "스텝업",
    "pistol",
    "피스톨",
}
_IMPACT_TOKENS = {
    "jump",
    "점프",
    "hop",
    "바운드",
    "run",
    "러닝",
    "sprint",
    "스프린트",
    "burpee",
    "버피",
    "skipping",
    "줄넘기",
}
_HIP_HINGE_TOKENS = {
    "deadlift",
    "데드리프트",
    "romanian",
    "루마니안",
    "rdl",
    "goodmorning",
    "굿모닝",
    "hiphinge",
    "힙힌지",
    "kettlebellswing",
    "케틀벨스윙",
    "backextension",
    "백익스텐션",
    "hyperextension",
    "하이퍼익스텐션",
}
_SPINAL_FLEXION_TOKENS = {
    "jeffersoncurl",
    "제퍼슨컬",
    "weightedcrunch",
    "웨이티드크런치",
    "cablecrunch",
    "케이블크런치",
    "situp",
    "싯업",
}
_UNSUPPORTED_BENT_OVER_TOKENS = {
    "bentover",
    "벤트오버",
    "barbellrow",
    "바벨로우",
    "pendlayrow",
    "펜들레이로우",
    "tbarrow",
    "티바로우",
}
_TRUNK_ROTATION_TOKENS = {
    "russiantwist",
    "러시안트위스트",
    "woodchop",
    "우드초퍼",
    "rotation",
    "로테이션",
    "twist",
    "트위스트",
}
_WRIST_WEIGHT_BEARING_TOKENS = {
    "pushup",
    "push-up",
    "푸시업",
    "푸쉬업",
    "plank",
    "플랭크",
    "bearcrawl",
    "베어크롤",
    "mountainclimber",
    "마운틴클라이머",
    "quadruped",
    "쿼드러펫",
}
_HEAVY_GRIP_TOKENS = {
    "deadlift",
    "데드리프트",
    "row",
    "로우",
    "pullup",
    "pull-up",
    "풀업",
    "chinup",
    "친업",
    "farmers",
    "farmer",
    "파머스",
    "carry",
    "캐리",
}
_PRONATION_SUPINATION_TOKENS = {
    "reversecurl",
    "리버스컬",
    "pronation",
    "회내",
    "supination",
    "회외",
    "wristrotation",
    "손목회전",
}
_FIXED_BAR_EQUIPMENT = {"BARBELL", "SMITH_MACHINE", "PULLUP_BAR"}
_LOADED_GRIP_EQUIPMENT = {
    "BARBELL",
    "DUMBBELL",
    "KETTLEBELL",
    "CABLE",
    "BAND",
    "MACHINE",
    "SMITH_MACHINE",
    "LANDMINE",
    "PLATE",
    "PULLUP_BAR",
}


@dataclass(frozen=True)
class ResolvedInjuryCondition:
    area_code: str
    status: str
    restricted_movement_patterns: tuple[str, ...] = ()


def normalize_injury_area_code(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("_", "-")
    return _AREA_ALIASES.get(normalized, normalized)


def normalize_injury_patterns(
    area_code: str,
    values: Iterable[str],
) -> tuple[str, ...]:
    allowed = INJURY_AREA_PATTERN_ORDER[normalize_injury_area_code(area_code)]
    selected = {
        str(value or "").strip()
        for value in values
        if str(value or "").strip()
    }
    return tuple(pattern for pattern in allowed if pattern in selected)


def validate_injury_condition_combination(
    area_code: str,
    status: str,
    restricted_patterns: Iterable[str],
) -> tuple[str, ...]:
    area = normalize_injury_area_code(area_code)
    if area not in INJURY_AREA_PATTERN_ORDER:
        raise ValueError("injuryConditions areaCode contains an unsupported value")
    if status not in INJURY_CONDITION_STATUSES:
        raise ValueError("injuryConditions status contains an unsupported value")

    raw_patterns = [str(value or "").strip() for value in restricted_patterns]
    patterns = normalize_injury_patterns(area, raw_patterns)
    allowed = set(INJURY_AREA_PATTERN_ORDER[area])
    if len(raw_patterns) != len(set(raw_patterns)):
        raise ValueError(
            "injuryConditions restrictedMovementPatterns must not contain duplicates"
        )
    if any(pattern not in allowed for pattern in raw_patterns):
        raise ValueError(
            "injuryConditions restrictedMovementPatterns contains a value "
            "unsupported for the selected areaCode"
        )
    if status == _MANAGED_WITH_RESTRICTIONS and not patterns:
        raise ValueError(
            "managedWithRestrictions requires at least one restrictedMovementPatterns value"
        )
    if status != _MANAGED_WITH_RESTRICTIONS and patterns:
        raise ValueError(
            "restrictedMovementPatterns is allowed only with managedWithRestrictions"
        )
    if "unknown" in patterns and len(patterns) != 1:
        raise ValueError(
            "unknown must be the only injuryConditions restrictedMovementPatterns value"
        )
    return patterns


def _entry_value(entry: Any, snake_name: str, camel_name: str) -> Any:
    if isinstance(entry, dict):
        return entry.get(snake_name, entry.get(camel_name))
    return getattr(entry, snake_name, None)


def _canonical_entry(entry: Any) -> ResolvedInjuryCondition:
    area = normalize_injury_area_code(_entry_value(entry, "area_code", "areaCode"))
    status = str(_entry_value(entry, "status", "status") or "").strip()
    patterns = _entry_value(
        entry,
        "restricted_movement_patterns",
        "restrictedMovementPatterns",
    ) or ()
    return ResolvedInjuryCondition(
        area_code=area,
        status=status,
        restricted_movement_patterns=validate_injury_condition_combination(
            area,
            status,
            patterns,
        ),
    )


def resolve_injury_condition_contexts(
    request_conditions: Iterable[Any],
    profile_conditions: Iterable[Any],
    *,
    legacy_shoulder_contexts: Iterable[tuple[Optional[str], Iterable[str]]] = (),
) -> tuple[ResolvedInjuryCondition, ...]:
    """Merge request/profile contexts without allowing a weaker override."""

    grouped: dict[str, list[ResolvedInjuryCondition]] = {}
    for entry in (*tuple(request_conditions), *tuple(profile_conditions)):
        canonical = _canonical_entry(entry)
        grouped.setdefault(canonical.area_code, []).append(canonical)

    for status, patterns in legacy_shoulder_contexts:
        if status is None:
            continue
        grouped.setdefault("shoulder", []).append(
            ResolvedInjuryCondition(
                area_code="shoulder",
                status=status,
                restricted_movement_patterns=validate_injury_condition_combination(
                    "shoulder",
                    status,
                    patterns,
                ),
            )
        )

    resolved: list[ResolvedInjuryCondition] = []
    for area in SUPPORTED_INJURY_CONDITION_AREAS:
        contexts = grouped.get(area, [])
        if not contexts:
            continue
        statuses = {context.status for context in contexts}
        if _ACTIVE_UNCLEARED in statuses:
            resolved.append(ResolvedInjuryCondition(area, _ACTIVE_UNCLEARED))
            continue
        if _MANAGED_WITH_RESTRICTIONS in statuses:
            combined = {
                pattern
                for context in contexts
                for pattern in context.restricted_movement_patterns
            }
            patterns = (
                ("unknown",)
                if "unknown" in combined
                else normalize_injury_patterns(area, combined)
            )
            resolved.append(
                ResolvedInjuryCondition(
                    area,
                    _MANAGED_WITH_RESTRICTIONS,
                    patterns,
                )
            )
            continue
        if "managedNoRestrictions" in statuses:
            resolved.append(ResolvedInjuryCondition(area, "managedNoRestrictions"))
        elif "recovered" in statuses:
            resolved.append(ResolvedInjuryCondition(area, "recovered"))
    return tuple(resolved)


def active_uncleared_neck_requires_safe_stop(
    conditions: Iterable[Any],
    target_muscles: Iterable[str],
) -> bool:
    """Stop upper-body training while a neck injury remains uncleared."""
    has_active_neck_condition = any(
        normalize_injury_area_code(getattr(condition, "area_code", None)) == "neck"
        and getattr(condition, "status", None) == _ACTIVE_UNCLEARED
        for condition in conditions
    )
    if not has_active_neck_condition:
        return False
    normalized_targets = {
        _normalize_muscle(value)
        for value in target_muscles
        if str(value or "").strip()
    }
    return bool(normalized_targets & _UPPER_BODY_MUSCLES)


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


def _normalize_muscle(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _candidate_muscles(candidate: dict) -> set[str]:
    muscles: set[str] = set()
    for key in (
        "primary_muscle",
        "secondary_muscle",
        "canonical_primary_muscle",
        "canonical_secondary_muscles",
        "catalog_source_primary_muscle",
        "catalog_source_secondary_muscle",
    ):
        raw = candidate.get(key)
        parts = raw if isinstance(raw, list) else str(raw or "").replace(";", ",").split(",")
        muscles.update(
            _normalize_muscle(part)
            for part in parts
            if str(part or "").strip()
        )
    return muscles


def _equipment(candidate: dict) -> set[str]:
    return {
        token.strip().upper()
        for token in str(candidate.get("equipment_req") or "").replace(";", ",").split(",")
        if token.strip()
    }


def _level_at_least(value: Any, threshold: str) -> bool:
    ranks = {"none": 0, "low": 1, "medium": 2, "high": 3, "elite": 4}
    return ranks.get(str(value or "").strip().lower(), 0) >= ranks[threshold]


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _has_any(names: str, tokens: Iterable[str]) -> bool:
    return any(_compact(token) in names for token in tokens)


def candidate_knee_movement_patterns(candidate: dict) -> frozenset[str]:
    names = _candidate_names(candidate)
    patterns: set[str] = set()
    if (
        _level_at_least(candidate.get("knee_flexion_depth_demand"), "high")
        or _is_true(candidate.get("deep_squat_position_required"))
        or _has_any(names, _DEEP_KNEE_FLEXION_TOKENS)
    ):
        patterns.add("deepKneeFlexion")
    if _has_any(names, _KNEE_EXTENSION_TOKENS):
        patterns.add("loadedKneeExtension")
    if _has_any(names, _KNEE_FLEXION_TOKENS):
        patterns.add("loadedKneeFlexion")
    if _has_any(names, _SINGLE_LEG_TOKENS):
        patterns.add("singleLegWeightBearing")
    if _has_any(names, _IMPACT_TOKENS):
        patterns.add("impactJumpRun")
    if candidate_has_knee_load(candidate) and not patterns:
        patterns.add("unknown")
    return frozenset(patterns)


def candidate_has_knee_load(candidate: dict) -> bool:
    names = _candidate_names(candidate)
    muscles = _candidate_muscles(candidate)
    movement_type = str(candidate.get("movement_type") or "").strip().upper()
    return bool(
        _level_at_least(candidate.get("knee_shear_load"), "medium")
        or _level_at_least(candidate.get("knee_flexion_depth_demand"), "medium")
        or _has_any(
            names,
            (
                *_KNEE_EXTENSION_TOKENS,
                *_KNEE_FLEXION_TOKENS,
                *_SINGLE_LEG_TOKENS,
                *_IMPACT_TOKENS,
            ),
        )
        or (
            "quadriceps" in muscles
            and movement_type in {"COMPOUND", "ISOLATION"}
        )
    )


def candidate_lumbar_movement_patterns(candidate: dict) -> frozenset[str]:
    names = _candidate_names(candidate)
    patterns: set[str] = set()
    if _has_any(names, _SPINAL_FLEXION_TOKENS):
        patterns.add("loadedSpinalFlexion")
    if _has_any(names, _HIP_HINGE_TOKENS):
        patterns.add("hipHinge")
    if _level_at_least(candidate.get("axial_load"), "medium"):
        patterns.add("axialLoading")
    if _has_any(names, _UNSUPPORTED_BENT_OVER_TOKENS):
        patterns.add("unsupportedBentOver")
    if _has_any(names, _TRUNK_ROTATION_TOKENS):
        patterns.add("trunkRotation")
    if candidate_has_lumbar_load(candidate) and not patterns:
        patterns.add("unknown")
    return frozenset(patterns)


def candidate_has_lumbar_load(candidate: dict) -> bool:
    names = _candidate_names(candidate)
    muscles = _candidate_muscles(candidate)
    return bool(
        "lower-back" in muscles
        or _level_at_least(candidate.get("lumbar_load"), "medium")
        or _level_at_least(candidate.get("axial_load"), "medium")
        or _has_any(
            names,
            (
                *_HIP_HINGE_TOKENS,
                *_SPINAL_FLEXION_TOKENS,
                *_UNSUPPORTED_BENT_OVER_TOKENS,
                *_TRUNK_ROTATION_TOKENS,
            ),
        )
    )


def candidate_wrist_movement_patterns(candidate: dict) -> frozenset[str]:
    names = _candidate_names(candidate)
    equipment = _equipment(candidate)
    patterns: set[str] = set()
    weight_bearing = _has_any(names, _WRIST_WEIGHT_BEARING_TOKENS)
    if (
        _level_at_least(candidate.get("wrist_loaded_extension_demand"), "medium")
        or _level_at_least(candidate.get("wrist_extension_demand"), "medium")
        or weight_bearing
    ):
        patterns.add("loadedWristExtension")
    if equipment & _FIXED_BAR_EQUIPMENT:
        patterns.add("fixedBarGrip")
    if (
        _level_at_least(candidate.get("wrist_stress"), "high")
        or _has_any(names, _HEAVY_GRIP_TOKENS)
        or (
            bool(_candidate_muscles(candidate) & _UPPER_BODY_MUSCLES)
            and bool(equipment & _LOADED_GRIP_EQUIPMENT)
        )
    ):
        patterns.add("loadedGrip")
    if _has_any(names, _PRONATION_SUPINATION_TOKENS):
        patterns.add("pronationSupination")
    if weight_bearing:
        patterns.add("weightBearingOnHands")
    if candidate_has_wrist_load(candidate) and not patterns:
        patterns.add("unknown")
    return frozenset(patterns)


def candidate_has_wrist_load(candidate: dict) -> bool:
    muscles = _candidate_muscles(candidate)
    equipment = _equipment(candidate)
    names = _candidate_names(candidate)
    return bool(
        "forearm" in muscles
        or bool(equipment & _FIXED_BAR_EQUIPMENT)
        or _level_at_least(candidate.get("wrist_stress"), "medium")
        or _level_at_least(candidate.get("wrist_loaded_extension_demand"), "medium")
        or _level_at_least(candidate.get("wrist_extension_demand"), "medium")
        or _has_any(
            names,
            (*_WRIST_WEIGHT_BEARING_TOKENS, *_HEAVY_GRIP_TOKENS, *_PRONATION_SUPINATION_TOKENS),
        )
        or (
            bool(muscles & _UPPER_BODY_MUSCLES)
            and bool(equipment & _LOADED_GRIP_EQUIPMENT)
        )
    )


def candidate_ankle_movement_patterns(candidate: dict) -> frozenset[str]:
    metadata = effective_injury_movement_metadata(candidate)
    patterns: set[str] = set()
    if _level_at_least(metadata["deep_dorsiflexion_demand"], "medium"):
        patterns.add("deepDorsiflexion")
    if _level_at_least(metadata["plantar_flexion_load"], "medium"):
        patterns.add("loadedPlantarFlexion")
    if _level_at_least(metadata["single_leg_stability_demand"], "medium"):
        patterns.add("singleLegStability")
    if _level_at_least(metadata["impact_demand"], "medium"):
        patterns.add("impactJumpRun")
    if _level_at_least(metadata["ankle_region_load_level"], "medium") and not patterns:
        patterns.add("unknown")
    return frozenset(patterns)


def candidate_has_ankle_load(candidate: dict) -> bool:
    metadata = effective_injury_movement_metadata(candidate)
    return _level_at_least(metadata["ankle_region_load_level"], "medium")


def candidate_hip_movement_patterns(candidate: dict) -> frozenset[str]:
    metadata = effective_injury_movement_metadata(candidate)
    patterns: set[str] = set()
    if _level_at_least(metadata["deep_hip_flexion_demand"], "medium"):
        patterns.add("deepHipFlexion")
    if _level_at_least(metadata["hip_hinge_load"], "medium"):
        patterns.add("hipHinge")
    if _level_at_least(metadata["hip_abduction_load"], "medium"):
        patterns.add("loadedAbduction")
    if _level_at_least(metadata["hip_adduction_load"], "medium"):
        patterns.add("loadedAdduction")
    if _level_at_least(metadata["hip_rotation_demand"], "medium"):
        patterns.add("hipRotation")
    if _level_at_least(metadata["single_leg_stability_demand"], "medium"):
        patterns.add("singleLegWeightBearing")
    if _level_at_least(metadata["hip_region_load_level"], "medium") and not patterns:
        patterns.add("unknown")
    return frozenset(patterns)


def candidate_has_hip_load(candidate: dict) -> bool:
    metadata = effective_injury_movement_metadata(candidate)
    return _level_at_least(metadata["hip_region_load_level"], "medium")


def candidate_elbow_movement_patterns(candidate: dict) -> frozenset[str]:
    metadata = effective_injury_movement_metadata(candidate)
    patterns: set[str] = set()
    if _level_at_least(metadata["elbow_flexion_load"], "medium"):
        patterns.add("loadedElbowFlexion")
    if _level_at_least(metadata["elbow_extension_load"], "medium"):
        patterns.add("loadedElbowExtension")
    if _level_at_least(metadata["loaded_grip_demand"], "medium"):
        patterns.add("loadedGrip")
    if _level_at_least(metadata["forearm_rotation_load"], "medium"):
        patterns.add("pronationSupination")
    if _level_at_least(metadata["upper_limb_weight_bearing_load"], "medium"):
        patterns.add("weightBearingOnArms")
    if _level_at_least(metadata["elbow_region_load_level"], "medium") and not patterns:
        patterns.add("unknown")
    return frozenset(patterns)


def candidate_has_elbow_load(candidate: dict) -> bool:
    metadata = effective_injury_movement_metadata(candidate)
    return _level_at_least(metadata["elbow_region_load_level"], "medium")


def candidate_neck_movement_patterns(candidate: dict) -> frozenset[str]:
    metadata = effective_injury_movement_metadata(candidate)
    patterns: set[str] = set()
    if _level_at_least(metadata["direct_neck_load"], "medium"):
        patterns.add("directNeckLoading")
    if _level_at_least(metadata["overhead_axial_load"], "medium"):
        patterns.add("overheadAxialLoading")
    if _level_at_least(
        metadata["sustained_neck_position_demand"],
        "medium",
    ):
        patterns.add("sustainedUnsupportedHeadPosition")
    if _level_at_least(metadata["ballistic_impact_demand"], "medium"):
        patterns.add("impactBallistic")
    if _level_at_least(metadata["neck_region_load_level"], "medium") and not patterns:
        patterns.add("unknown")
    return frozenset(patterns)


def candidate_has_neck_load(candidate: dict) -> bool:
    metadata = effective_injury_movement_metadata(candidate)
    return _level_at_least(metadata["neck_region_load_level"], "medium")


def candidate_injury_movement_patterns(
    candidate: dict,
    area_code: str,
) -> frozenset[str]:
    area = normalize_injury_area_code(area_code)
    if area == "shoulder":
        return candidate_shoulder_movement_patterns(candidate)
    if area == "knee":
        return candidate_knee_movement_patterns(candidate)
    if area == "lumbar":
        return candidate_lumbar_movement_patterns(candidate)
    if area == "wrist":
        return candidate_wrist_movement_patterns(candidate)
    if area == "ankle":
        return candidate_ankle_movement_patterns(candidate)
    if area == "hip":
        return candidate_hip_movement_patterns(candidate)
    if area == "elbow":
        return candidate_elbow_movement_patterns(candidate)
    if area == "neck":
        return candidate_neck_movement_patterns(candidate)
    return frozenset()


def candidate_has_injury_area_load(candidate: dict, area_code: str) -> bool:
    area = normalize_injury_area_code(area_code)
    if area == "shoulder":
        return candidate_has_shoulder_load(candidate)
    if area == "knee":
        return candidate_has_knee_load(candidate)
    if area == "lumbar":
        return candidate_has_lumbar_load(candidate)
    if area == "wrist":
        return candidate_has_wrist_load(candidate)
    if area == "ankle":
        return candidate_has_ankle_load(candidate)
    if area == "hip":
        return candidate_has_hip_load(candidate)
    if area == "elbow":
        return candidate_has_elbow_load(candidate)
    if area == "neck":
        return candidate_has_neck_load(candidate)
    return False


def candidate_condition_restriction_overlap(
    candidate: dict,
    condition: Any,
) -> frozenset[str]:
    canonical = _canonical_entry(condition)
    if canonical.status not in {_ACTIVE_UNCLEARED, _MANAGED_WITH_RESTRICTIONS}:
        return frozenset()
    if canonical.area_code == "shoulder":
        return candidate_restricted_movement_overlap(
            candidate,
            canonical.status,
            canonical.restricted_movement_patterns,
        )

    detected = set(
        candidate_injury_movement_patterns(candidate, canonical.area_code)
    )
    has_region_load = candidate_has_injury_area_load(candidate, canonical.area_code)
    if canonical.status == _ACTIVE_UNCLEARED:
        return frozenset(detected or ({"unknown"} if has_region_load else set()))

    selected = set(canonical.restricted_movement_patterns)
    if "unknown" in selected:
        return frozenset({"unknown"}) if has_region_load else frozenset()
    overlap = detected & selected
    if overlap:
        return frozenset(overlap)
    if "unknown" in detected:
        return frozenset({"unknown"})
    return frozenset()


def candidate_injury_condition_overlaps(
    candidate: dict,
    conditions: Iterable[Any],
) -> dict[str, frozenset[str]]:
    overlaps: dict[str, frozenset[str]] = {}
    for condition in conditions:
        canonical = _canonical_entry(condition)
        matched = candidate_condition_restriction_overlap(candidate, canonical)
        if matched:
            overlaps[canonical.area_code] = matched
    return overlaps


def injury_condition_labels_ko(condition: Any) -> list[str]:
    canonical = _canonical_entry(condition)
    labels = {
        "overheadPosition": "머리 위 동작",
        "horizontalPush": "수평 밀기",
        "horizontalPull": "수평 당기기",
        "armRaise": "팔 들어 올리기",
        "behindBackRotation": "팔을 등 뒤로 보내는 동작",
        "deepKneeFlexion": "깊은 무릎 굽힘",
        "loadedKneeExtension": "무릎을 펴며 힘을 쓰는 동작",
        "loadedKneeFlexion": "무릎을 굽히며 힘을 쓰는 동작",
        "singleLegWeightBearing": "한 발 체중 지지",
        "impactJumpRun": "점프·달리기 충격",
        "loadedSpinalFlexion": "부하가 있는 허리 굽힘",
        "hipHinge": "힙힌지",
        "axialLoading": "척추 수직 부하",
        "unsupportedBentOver": "지지 없는 숙인 자세",
        "trunkRotation": "몸통 회전",
        "loadedWristExtension": "부하가 있는 손목 젖힘",
        "fixedBarGrip": "고정 바 그립",
        "loadedGrip": "부하가 있는 그립",
        "pronationSupination": "손목·전완 회전",
        "weightBearingOnHands": "손으로 체중 지지",
        "deepDorsiflexion": "깊은 발목 배굴",
        "loadedPlantarFlexion": "부하가 있는 발목 저측굴곡",
        "singleLegStability": "한 발 균형 지지",
        "deepHipFlexion": "깊은 고관절 굽힘",
        "loadedAbduction": "부하가 있는 고관절 벌림",
        "loadedAdduction": "부하가 있는 고관절 모음",
        "hipRotation": "고관절 회전",
        "loadedElbowFlexion": "부하가 있는 팔꿈치 굽힘",
        "loadedElbowExtension": "부하가 있는 팔꿈치 폄",
        "weightBearingOnArms": "팔로 체중 지지",
        "directNeckLoading": "목에 직접 부하",
        "overheadAxialLoading": "머리 위 척추 수직 부하",
        "sustainedUnsupportedHeadPosition": "머리를 지지하지 않은 자세 유지",
        "impactBallistic": "충격·탄도성 동작",
        "unknown": "제한 동작 미확인",
    }
    return [labels[pattern] for pattern in canonical.restricted_movement_patterns]


def injury_area_label_ko(area_code: str) -> str:
    return {
        "shoulder": "어깨",
        "knee": "무릎",
        "lumbar": "허리",
        "wrist": "손목",
        "ankle": "발목·발",
        "hip": "고관절·사타구니",
        "elbow": "팔꿈치",
        "neck": "목",
    }[normalize_injury_area_code(area_code)]


def format_injury_conditions_for_prompt(conditions: Iterable[Any]) -> str:
    values: list[str] = []
    for condition in conditions:
        canonical = _canonical_entry(condition)
        patterns = ",".join(canonical.restricted_movement_patterns) or "none"
        values.append(
            f"{canonical.area_code}:{canonical.status}:restricted={patterns}"
        )
    return "; ".join(values) if values else "none"

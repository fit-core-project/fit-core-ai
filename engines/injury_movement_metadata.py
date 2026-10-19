"""Versioned objective movement-demand projection for managed injury policy.

The projection describes what an exercise asks the body to do. It does not
diagnose an injury, prescribe rehabilitation, or decide medical clearance.
"""

from __future__ import annotations

import re
from typing import Any, Iterable


PROFILE_VERSION = "managed-injury-movement-profile-v3-2026-07-27"
DEMAND_LEVELS = ("low", "medium", "high")

PROFILE_FIELDS = (
    "ankle_region_load_level",
    "deep_dorsiflexion_demand",
    "plantar_flexion_load",
    "single_leg_stability_demand",
    "impact_demand",
    "hip_region_load_level",
    "deep_hip_flexion_demand",
    "hip_hinge_load",
    "hip_abduction_load",
    "hip_adduction_load",
    "hip_rotation_demand",
    "elbow_region_load_level",
    "elbow_flexion_load",
    "elbow_extension_load",
    "loaded_grip_demand",
    "forearm_rotation_load",
    "upper_limb_weight_bearing_load",
    "neck_region_load_level",
    "direct_neck_load",
    "overhead_axial_load",
    "sustained_neck_position_demand",
    "ballistic_impact_demand",
)

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
_ANKLE_MUSCLES = {"calves", "soleus", "left-soleus", "right-soleus"}
_HIP_MUSCLES = {"gluteal", "adductor", "abductors", "hamstring"}
_ELBOW_MUSCLES = {"biceps", "triceps", "forearm"}
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

_SINGLE_LEG_TOKENS = (
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
    "stepdown",
    "stepdescent",
    "스텝다운",
    "pistol",
    "피스톨",
)
_STEP_DESCENT_TOKENS = (
    "stepdown",
    "stepdescent",
    "스텝다운",
)
_STEP_ASCENT_TOKENS = (
    "stepup",
    "스텝업",
)
_LOADED_LOCOMOTION_TOKENS = (
    "sled",
    "슬레드",
)
_HIGH_IMPACT_TOKENS = (
    "boxjump",
    "박스점프",
    "depthjump",
    "드롭점프",
    "broadjump",
    "멀리뛰기",
    "tuckjump",
    "턱점프",
    "sprint",
    "스프린트",
    "burpee",
    "버피",
    "jumprope",
    "줄넘기",
    "plyometric",
    "플라이오메트릭",
)
_MEDIUM_IMPACT_TOKENS = (
    "jump",
    "점프",
    "hop",
    "바운드",
    "run",
    "러닝",
    "jog",
    "조깅",
    "skipping",
    "마운틴클라이머",
    "shuffle",
    "셔플",
)
_PLANTAR_FLEXION_TOKENS = (
    "calfraise",
    "카프레이즈",
    "heelraise",
    "힐레이즈",
    "plantarflex",
    "플랜터플렉션",
    "toe press",
    "토프레스",
)
_HIP_HINGE_TOKENS = (
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
)
_HIP_ABDUCTION_TOKENS = (
    "hipabduction",
    "힙어브덕션",
    "고관절외전",
    "sidelegraise",
    "사이드레그레이즈",
    "clamshell",
    "클램셸",
    "클램쉘",
)
_HIP_ADDUCTION_TOKENS = (
    "hipadduction",
    "힙어덕션",
    "고관절내전",
    "adductor",
    "어덕터",
    "내전근",
)
_HIP_ROTATION_TOKENS = (
    "hiprotation",
    "고관절회전",
    "internalrotation",
    "내회전",
    "externalrotation",
    "외회전",
    "90/90",
    "9090",
    "piriformis",
    "이상근",
)
_ELBOW_FLEXION_TOKENS = (
    "curl",
    "컬",
    "chinup",
    "chin-up",
    "친업",
)
_LOADED_PULL_ELBOW_FLEXION_TOKENS = (
    "latpulldown",
    "pulldown",
    "랫풀다운",
    "풀다운",
    "facepull",
    "페이스풀",
    "pullup",
    "pull-up",
    "풀업",
)
_ELBOW_EXTENSION_TOKENS = (
    "tricep",
    "트라이셉",
    "삼두",
    "pushdown",
    "푸시다운",
    "skullcrusher",
    "스컬크러셔",
    "dip",
    "딥스",
    "pushup",
    "push-up",
    "푸시업",
    "푸쉬업",
)
_ELBOW_PRESS_TOKENS = ("press", "프레스")
_FOREARM_ROTATION_TOKENS = (
    "pronation",
    "회내",
    "supination",
    "회외",
    "reversecurl",
    "리버스컬",
    "wristrotation",
    "손목회전",
)
_LOADED_GRIP_TOKENS = (
    "deadlift",
    "데드리프트",
    "pullup",
    "pull-up",
    "풀업",
    "chinup",
    "친업",
    "farmer",
    "파머",
    "carry",
    "캐리",
    "curl",
    "컬",
)
_UPPER_LIMB_WEIGHT_BEARING_TOKENS = (
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
    "handstand",
    "핸드스탠드",
)
_DIRECT_NECK_TOKENS = (
    "neckflex",
    "넥플렉션",
    "neckextension",
    "넥익스텐션",
    "neckrotation",
    "넥로테이션",
    "neckharness",
    "넥하네스",
    "cervical",
    "경추",
    "목운동",
)
_SUSTAINED_NECK_POSITION_TOKENS = (
    "pronelying",
    "프론라잉",
    "복와위",
    "superman",
    "슈퍼맨",
    "bentover",
    "벤트오버",
    "plank",
    "플랭크",
    "bearcrawl",
    "베어크롤",
)


def _compact(value: Any) -> str:
    return re.sub(r"[\s_\-/]+", "", str(value or "").strip().lower())


def _text(candidate: dict[str, Any]) -> str:
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


def _movement_words(candidate: dict[str, Any]) -> set[str]:
    raw = " ".join(
        str(candidate.get(key) or "").lower()
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
    return set(re.findall(r"[a-z]+|[가-힣]+", raw))


def _contains(text: str, tokens: Iterable[str]) -> bool:
    return any(_compact(token) in text for token in tokens)


def _normalize_muscle(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _muscles(candidate: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in (
        "primary_muscle",
        "secondary_muscle",
        "canonical_primary_muscle",
        "canonical_secondary_muscles",
        "corrected_primary_muscle",
    ):
        raw = candidate.get(key)
        parts = raw if isinstance(raw, list) else str(raw or "").replace(";", ",").split(",")
        values.update(
            _normalize_muscle(part)
            for part in parts
            if str(part or "").strip()
        )
    return values


def _primary_muscles(candidate: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in (
        "canonical_primary_muscle",
        "corrected_primary_muscle",
        "primary_muscle",
    ):
        raw = candidate.get(key)
        parts = raw if isinstance(raw, list) else str(raw or "").replace(";", ",").split(",")
        values.update(
            _normalize_muscle(part)
            for part in parts
            if str(part or "").strip()
        )
    return values


def _source_primary_muscles(candidate: dict[str, Any]) -> set[str]:
    return {
        _normalize_muscle(part)
        for part in str(candidate.get("primary_muscle") or "").replace(";", ",").split(",")
        if str(part or "").strip()
    }


def _equipment(candidate: dict[str, Any]) -> set[str]:
    return {
        token.strip().upper()
        for token in str(candidate.get("equipment_req") or "").replace(";", ",").split(",")
        if token.strip()
    }


def _level_rank(value: Any) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(
        str(value or "").strip().lower(),
        0,
    )


def _level(*, high: bool = False, medium: bool = False) -> str:
    if high:
        return "high"
    if medium:
        return "medium"
    return "low"


def _max_level(*values: Any) -> str:
    return max(
        (str(value or "low").strip().lower() for value in values),
        key=_level_rank,
    )


def derive_injury_movement_metadata(candidate: dict[str, Any]) -> dict[str, Any]:
    """Project repeatable movement/load fields from catalog-visible metadata."""

    text = _text(candidate)
    muscles = _muscles(candidate)
    primary_muscles = _primary_muscles(candidate)
    source_primary_muscles = _source_primary_muscles(candidate)
    equipment = _equipment(candidate)
    movement_type = str(candidate.get("movement_type") or "").strip().upper()
    compound = movement_type == "COMPOUND"

    dorsiflexion = str(
        candidate.get("ankle_dorsiflexion_demand") or "low"
    ).strip().lower()
    if _contains(text, (*_STEP_ASCENT_TOKENS, *_STEP_DESCENT_TOKENS)):
        dorsiflexion = _max_level(dorsiflexion, "medium")
    hip_flexion = str(
        candidate.get("hip_flexion_demand") or "low"
    ).strip().lower()

    high_impact = _contains(text, _HIGH_IMPACT_TOKENS)
    medium_impact = high_impact or _contains(text, _MEDIUM_IMPACT_TOKENS)
    impact = _level(high=high_impact, medium=medium_impact)
    single_leg = _contains(text, _SINGLE_LEG_TOKENS)
    loaded_locomotion = _contains(text, _LOADED_LOCOMOTION_TOKENS)
    lower_body_compound = compound and bool(muscles & _LOWER_BODY_MUSCLES)

    plantar_flexion = _level(
        high=_contains(text, _PLANTAR_FLEXION_TOKENS),
        medium=bool(muscles & _ANKLE_MUSCLES),
    )
    ankle_region = _level(
        high=(
            _level_rank(plantar_flexion) >= 2
            or high_impact
            or (single_leg and compound)
        ),
        medium=(
            bool(muscles & _ANKLE_MUSCLES)
            or _level_rank(dorsiflexion) >= 1
            or medium_impact
            or single_leg
            or loaded_locomotion
            or lower_body_compound
        ),
    )

    hip_hinge = _level(
        high=_contains(text, _HIP_HINGE_TOKENS),
        medium=compound and bool(primary_muscles & {"hamstring", "gluteal"}),
    )
    hip_abduction = _level(
        high=(
            "abductors" in muscles
            or _contains(text, _HIP_ABDUCTION_TOKENS)
        ),
        medium="gluteal" in muscles and single_leg,
    )
    hip_adduction = _level(
        high=(
            "adductor" in muscles
            or _contains(text, _HIP_ADDUCTION_TOKENS)
        ),
    )
    hip_rotation = _level(
        high=_contains(text, _HIP_ROTATION_TOKENS),
    )
    hip_region = _level(
        high=(
            compound
            and bool(muscles & _HIP_MUSCLES)
        )
        or any(
            _level_rank(value) >= 2
            for value in (hip_hinge, hip_abduction, hip_adduction, hip_rotation)
        ),
        medium=(
            bool(muscles & _HIP_MUSCLES)
            or (compound and bool(muscles & _LOWER_BODY_MUSCLES))
            or _level_rank(hip_flexion) >= 1
            or single_leg
        ),
    )

    lower_body_curl = _contains(
        text,
        ("legcurl", "레그컬", "hamstringcurl", "햄스트링컬"),
    )
    movement_words = _movement_words(candidate)
    loaded_pull_elbow_flexion = (
        "row" in movement_words
        or "로우" in movement_words
        or _contains(text, _LOADED_PULL_ELBOW_FLEXION_TOKENS)
    )
    elbow_flexion = _level(
        high=(
            "biceps" in primary_muscles
            or (
                _contains(text, _ELBOW_FLEXION_TOKENS)
                and not lower_body_curl
            )
        ),
        medium=loaded_pull_elbow_flexion
        or (
            compound
            and bool(
                (source_primary_muscles | muscles)
                & {"upper-back", "back-deltoids"}
            )
        ),
    )
    lower_body_press = _contains(text, ("legpress", "레그프레스"))
    elbow_extension = _level(
        high=(
            "triceps" in primary_muscles
            or _contains(text, _ELBOW_EXTENSION_TOKENS)
        ),
        medium=(
            _contains(text, _ELBOW_PRESS_TOKENS)
            and not lower_body_press
        ),
    )
    forearm_rotation = _level(
        high=_contains(text, _FOREARM_ROTATION_TOKENS),
        medium="forearm" in muscles,
    )
    loaded_grip = _level(
        high=(
            _contains(text, _LOADED_GRIP_TOKENS)
            and not lower_body_curl
        ),
        medium=bool(muscles & {"biceps", "forearm", "upper-back"}),
    )
    upper_limb_weight_bearing = _level(
        high=_contains(text, _UPPER_LIMB_WEIGHT_BEARING_TOKENS),
    )
    elbow_region = _level(
        high=any(
            _level_rank(value) >= 2
            for value in (
                elbow_flexion,
                elbow_extension,
                loaded_grip,
                forearm_rotation,
                upper_limb_weight_bearing,
            )
        ),
        medium=(
            any(
                _level_rank(value) >= 1
                for value in (
                    elbow_flexion,
                    elbow_extension,
                    loaded_grip,
                    forearm_rotation,
                    upper_limb_weight_bearing,
                )
            )
            or (bool(muscles & _UPPER_BODY_MUSCLES) and compound)
        ),
    )

    direct_neck = _level(
        high=(
            "neck" in muscles
            or _contains(text, _DIRECT_NECK_TOKENS)
        ),
    )
    sustained_neck = _level(
        high=_contains(text, _SUSTAINED_NECK_POSITION_TOKENS),
    )
    weighted_overhead = (
        bool(candidate.get("overhead_position_required"))
        and compound
        and bool(
            equipment
            & {
                "BARBELL",
                "DUMBBELL",
                "KETTLEBELL",
                "MACHINE",
                "SMITH_MACHINE",
                "CABLE",
                "LANDMINE",
                "PLATE",
            }
        )
    )
    overhead_axial = _level(
        high=(
            weighted_overhead
            and bool(equipment & {"BARBELL", "DUMBBELL", "KETTLEBELL", "SMITH_MACHINE"})
        ),
        medium=weighted_overhead,
    )
    ballistic = _max_level(
        impact,
        _level(
            high=_contains(text, ("combat", "boxing", "복싱", "격투", "contact")),
        ),
    )
    neck_region = _level(
        high=(
            _level_rank(direct_neck) >= 2
            or _level_rank(overhead_axial) >= 2
            or _level_rank(ballistic) >= 2
        ),
        medium=(
            "neck" in muscles
            or _level_rank(overhead_axial) >= 1
            or _level_rank(sustained_neck) >= 1
            or _level_rank(ballistic) >= 1
        ),
    )

    return {
        "ankle_region_load_level": ankle_region,
        "deep_dorsiflexion_demand": dorsiflexion
        if dorsiflexion in DEMAND_LEVELS else "low",
        "plantar_flexion_load": plantar_flexion,
        "single_leg_stability_demand": _level(
            high=single_leg and compound,
            medium=single_leg,
        ),
        "impact_demand": impact,
        "hip_region_load_level": hip_region,
        "deep_hip_flexion_demand": hip_flexion
        if hip_flexion in DEMAND_LEVELS else "low",
        "hip_hinge_load": hip_hinge,
        "hip_abduction_load": hip_abduction,
        "hip_adduction_load": hip_adduction,
        "hip_rotation_demand": hip_rotation,
        "elbow_region_load_level": elbow_region,
        "elbow_flexion_load": elbow_flexion,
        "elbow_extension_load": elbow_extension,
        "loaded_grip_demand": loaded_grip,
        "forearm_rotation_load": forearm_rotation,
        "upper_limb_weight_bearing_load": upper_limb_weight_bearing,
        "neck_region_load_level": neck_region,
        "direct_neck_load": direct_neck,
        "overhead_axial_load": overhead_axial,
        "sustained_neck_position_demand": sustained_neck,
        "ballistic_impact_demand": ballistic,
        "source_type": "deterministicCatalogProjection",
        "confidence": "medium",
        "review_required": 1,
        "version": PROFILE_VERSION,
    }


def effective_injury_movement_metadata(candidate: dict[str, Any]) -> dict[str, Any]:
    """Use persisted profile values when present, otherwise derive them."""

    persisted = {
        field: str(candidate.get(field) or "").strip().lower()
        for field in PROFILE_FIELDS
    }
    if all(value in DEMAND_LEVELS for value in persisted.values()):
        return persisted

    derived = derive_injury_movement_metadata(candidate)
    for field in PROFILE_FIELDS:
        value = persisted[field]
        if value in DEMAND_LEVELS:
            derived[field] = value
    return derived

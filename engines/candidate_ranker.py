"""후보 운동 점수화 및 프롬프트 포맷터."""
import os
import re
from typing import Any, Dict, List, Optional

from .schemas import PainAreaEntry, RecentSetRecord
from .option_resolver import (
    canonical_adjunct_options,
    candidate_adjunct_option_codes,
    candidate_matches_adjunct_option,
    derive_exercise_risk_profiles,
    infer_medical_clearance,
    matching_adjunct_options,
    resolve_candidate_policy_for_request,
)
from .registry import LARGE_MUSCLE_SLUGS, ACCESSORY_MUSCLE_SLUGS, LOADED_EQUIPMENT_TOKENS
from .exercise_display import candidate_display_name, is_generic_candidate_display_text
from .equipment_policy import select_equipment_option
from .experience_policy import main_role_eligibility, automatic_recommendation_eligibility, BEGINNER_EXPERIENCE_ALIASES
from .adjunct_admission import admission_purposes
from .injury_condition_policy import (
    candidate_injury_condition_overlaps,
    resolve_injury_condition_contexts,
)
from .muscle_taxonomy import recommendation_taxonomy_is_eligible
from .shoulder_condition_policy import candidate_restricted_movement_overlap
from .specialization_policy import (
    candidate_matches_specialization_focus,
    candidate_matches_training_method,
    focus_label_ko,
    legacy_specialization_values,
    training_method_instruction,
)


def _normalize_tokens(raw: Optional[str]) -> set[str]:
    if not raw:
        return set()
    return {token.strip().upper() for token in str(raw).split(",") if token.strip()}


def _is_loadable_equipment(raw: Optional[str]) -> bool:
    return bool(_normalize_tokens(raw) & LOADED_EQUIPMENT_TOKENS)


def _is_bodyweight_only(raw: Optional[str]) -> bool:
    tokens = _normalize_tokens(raw)
    return bool(tokens) and tokens <= {"BODYWEIGHT"}


_LEVEL_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3, "elite": 4}
_EFFECT_SCORE = {"low": -4, "medium": 3, "high": 10, "elite": 16}
_STF_SCORE = {"poor": -8, "fair": 0, "good": 6, "excellent": 10}
_MAX_CANDIDATES_PER_FAMILY = 1
_SELECTION_FAMILY_EQUIPMENT_TOKENS = (
    "스미스 머신",
    "스미스머신",
    "메디신 볼",
    "메디신볼",
    "바벨",
    "덤벨",
    "케이블",
    "머신",
    "밴드",
    "케틀벨",
    "플레이트",
    "랜드마인",
    "짐볼",
    "smith machine",
    "medicine ball",
    "barbell",
    "dumbbell",
    "cable",
    "machine",
    "band",
    "kettlebell",
    "plate",
    "landmine",
    "bodyweight",
    "trx",
)
_SPORT_OR_CARDIO_CONTEXT_TOKENS = {
    "농구",
    "배드민턴",
    "사이클",
    "러닝",
    "보강",
    "착지",
    "테니스",
    "축구",
    "등산",
    "하이킹",
    "골프",
    "러닝머신",
    "걷기",
    "야구",
    "treadmill",
    "walking",
    "running",
    "cycling",
    "basketball",
    "badminton",
    "hiking",
    "golf",
    "tennis",
    "baseball",
    "soccer",
}
_BALLISTIC_OR_HIGH_COORDINATION_TOKENS = {
    "스윙",
    "파워",
    "점프",
    "착지",
    "바운드",
    "스프린트",
    "스킵",
    "에어플레인",
    "swing",
    "power",
    "jump",
    "landing",
    "bound",
    "sprint",
    "skip",
    "airplane",
}
_MAINSTREAM_GYM_PATTERNS = {
    "벤치프레스",
    "benchpress",
    "체스트프레스",
    "chestpress",
    "숄더프레스",
    "shoulderpress",
    "오버헤드프레스",
    "overheadpress",
    "랫풀다운",
    "latpulldown",
    "시티드로우",
    "seatedrow",
    "케이블로우",
    "cablerow",
    "덤벨로우",
    "dumbbellrow",
    "레그프레스",
    "legpress",
    "레그익스텐션",
    "legextension",
    "레그컬",
    "legcurl",
    "스쿼트",
    "squat",
    "런지",
    "lunge",
    "힙쓰러스트",
    "hipthrust",
    "레터럴레이즈",
    "lateralraise",
    "푸쉬다운",
    "pushdown",
    "킥백",
    "kickback",
    "바벨컬",
    "barbellcurl",
    "덤벨컬",
    "dumbbellcurl",
    "케이블컬",
    "cablecurl",
    "머신컬",
    "machinecurl",
    "해머컬",
    "hammercurl",
    "프리처컬",
    "preachercurl",
    "드래그컬",
    "dragcurl",
    "바이셉스컬",
    "bicepscurl",
    "페이스풀",
    "facepull",
    "펙덱",
    "pecdeck",
    "플라이",
    "fly",
    "카프레이즈",
    "calfraise",
    "플랭크",
    "plank",
    "크런치",
    "crunch",
}
_UNCOMMON_DEFAULT_GYM_TOKENS = {
    "링",
    "ring",
    "trx",
    "스내치",
    "snatch",
    "클린",
    "clean",
    "저크",
    "jerk",
    "스러스터",
    "thruster",
    "월볼",
    "wallball",
    "키핑",
    "kipping",
    "피터슨",
    "peterson",
    "시시",
    "sissy",
    "제퍼슨",
    "jefferson",
    "코사크",
    "cossack",
    "에어플레인",
    "airplane",
    "싱글레그",
    "singleleg",
    "단일다리",
    "교대",
    "가동범위",
    "발끝",
    "전경사",
    "상체세움",
    "상체 세움",
    "박스터치",
    "박스 터치",
    "워리어",
    "warrior",
    "미니볼",
    "mini ball",
    "소도구",
    "펠빅컬",
    "pelvic curl",
    "정렬",
    "alignment",
    "요가",
    "yoga",
    "필라테스",
    "pilates",
    "템포",
    "정지",
    "내로우",
    "narrow",
}
_SPECIALTY_MOBILITY_DEFAULT_GYM_TOKENS = {
    "워리어",
    "warrior",
    "미니볼",
    "mini ball",
    "소도구",
    "펠빅컬",
    "pelvic curl",
    "정렬",
    "alignment",
    "요가",
    "yoga",
    "필라테스",
    "pilates",
}
_CROSSFIT_STYLE_TOKENS = {
    "crossfit",
    "크로스핏",
    "wod",
    "스내치",
    "snatch",
    "클린",
    "clean",
    "저크",
    "jerk",
    "스러스터",
    "thruster",
    "버피",
    "burpee",
    "월볼",
    "wallball",
    "키핑",
    "kipping",
}
_GYM_DEFAULT_STYLES = {"", "gym", "fitness", "bodybuilding", "hypertrophy", "health_club", "헬스장", "헬스", "보디빌딩"}
_TRAINING_STYLE_ALIASES = {
    "pilates_yoga": "yoga_pilates",
    "yoga": "yoga_pilates",
    "pilates": "yoga_pilates",
    "프리햅": "rehab_prehab",
    "prehab": "rehab_prehab",
    "rehab": "rehab_prehab",
    "재활": "rehab_prehab",
    "모빌리티": "mobility",
    "요가": "yoga_pilates",
    "필라테스": "yoga_pilates",
    "홈트": "home",
}
# Legacy entry modes still use the gym blueprint and its candidate admission gate.
_GYM_BASE_ADJUNCT_STYLES = {"crossfit", "athletic", "home", "mobility", "yoga_pilates", "rehab_prehab"}
_DEFAULT_GYM_EXPLICIT_ONLY_STYLES = {
    "athletic",
    "cardio",
    "conditioning",
    "crossfit",
    "endurance",
    "mobility",
    "rehab_prehab",
    "yoga_pilates",
    "home_specific",
}
_ADJUNCT_OPTION_ALLOWED_STYLES = {
    "conditioning_finisher": {"athletic", "cardio", "conditioning", "crossfit"},
    "endurance_support": {"athletic", "cardio", "conditioning", "crossfit", "endurance"},
    "mobility_warmup": {"mobility", "yoga_pilates"},
    "cooldown_stretch": {"mobility", "yoga_pilates"},
    "prehab_accessory": {"rehab_prehab"},
}
_SPLIT_ACCESSORY_DIRECT_BOOSTS = {
    "push": {"triceps": 18, "front-deltoids": 8, "side-deltoids": 8},
    "pull": {"biceps": 20, "forearm": 6, "back-deltoids": 4, "trapezius": 4},
    "legs": {"calves": 12, "adductor": 5, "abductors": 5},
    "upper": {"biceps": 12, "triceps": 12, "front-deltoids": 6, "back-deltoids": 6, "trapezius": 4},
}
_SPLIT_PRIMARY_TARGETS = {
    "push": {"chest", "front-deltoids", "side-deltoids", "triceps"},
    "pull": {"upper-back", "lower-back", "back-deltoids", "trapezius", "biceps", "forearm"},
    "legs": {"quadriceps", "hamstring", "gluteal", "calves", "adductor", "abductors"},
    "lower": {"quadriceps", "hamstring", "gluteal", "calves", "adductor", "abductors"},
    "upper": {"chest", "upper-back", "front-deltoids", "back-deltoids", "trapezius", "triceps", "biceps", "forearm"},
    "core": {"abs", "obliques"},
}
_TIBIALIS_ACCESSORY_TOKENS = {
    "tibialis",
    "티비알리스",
    "전경골",
    "shinraise",
}
_PULLDOWN_TOKENS = {"랫풀다운", "풀다운", "latpulldown", "pulldown"}
_CURL_TOKENS = {"컬", "curl", "bicepscurl"}


def _split_primary_targets(target_split_label: Optional[str]) -> set[str]:
    return _SPLIT_PRIMARY_TARGETS.get(
        str(target_split_label or "").strip().lower().replace("-", "_").replace(" ", "_"),
        set(),
    )


def _is_tibialis_accessory_candidate(candidate: dict) -> bool:
    compact = _candidate_name_compact(candidate)
    return any(token.replace(" ", "").lower() in compact for token in _TIBIALIS_ACCESSORY_TOKENS)


def _is_biceps_labeled_pulldown_candidate(candidate: dict) -> bool:
    compact = _candidate_name_compact(candidate)
    has_pulldown = any(token.replace(" ", "").lower() in compact for token in _PULLDOWN_TOKENS)
    has_curl = any(token.replace(" ", "").lower() in compact for token in _CURL_TOKENS)
    return has_pulldown and not has_curl


def _level(value: Any) -> str:
    return str(value or "").strip().lower()


def _level_rank(value: Any) -> int:
    return _LEVEL_RANK.get(_level(value), 0)


def _dict_get(raw: Any, *keys: str) -> Any:
    if not isinstance(raw, dict):
        return None
    for key in keys:
        if key in raw:
            return raw[key]
    return None


def _is_false(value: Any) -> bool:
    if isinstance(value, bool):
        return not value
    if value is None:
        return False
    return str(value).strip().lower() in {"0", "false", "no", "n", "off"}


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_sport_or_cardio_context_candidate(candidate: dict) -> bool:
    name = " ".join([
        str(candidate.get("name_kr") or ""),
        str(candidate.get("name_en") or ""),
    ]).lower()
    compact = re.sub(r"[\s/_()·,.-]+", "", name)
    return any(token.replace(" ", "") in compact for token in _SPORT_OR_CARDIO_CONTEXT_TOKENS)


def _is_ballistic_or_high_coordination_candidate(candidate: dict) -> bool:
    name = " ".join([
        str(candidate.get("name_kr") or ""),
        str(candidate.get("name_en") or ""),
    ]).lower()
    compact = re.sub(r"[\s/_()·,.-]+", "", name)
    return any(token.replace(" ", "") in compact for token in _BALLISTIC_OR_HIGH_COORDINATION_TOKENS)


def _candidate_name_compact(candidate: dict) -> str:
    name = " ".join([
        str(candidate.get("name_kr") or ""),
        str(candidate.get("name_en") or ""),
    ]).lower()
    return re.sub(r"[\s/_()·,.-]+", "", name)


def _candidate_name_has_any(candidate: dict, tokens: set[str]) -> bool:
    compact = _candidate_name_compact(candidate)
    return any(token.replace(" ", "").lower() in compact for token in tokens)


def _optional_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _metadata_style_tags(candidate: dict) -> set[str]:
    raw = candidate.get("training_style_tags")
    if not raw:
        return set()
    return {
        str(token).strip().lower().replace("-", "_").replace(" ", "_")
        for token in str(raw).split(",")
        if str(token).strip()
    }


def _is_mainstream_gym_candidate(candidate: dict) -> bool:
    from .focus_catalog_admission import REVIEW_VERSION, REVIEWED_IDS

    if (candidate.get("id") in REVIEWED_IDS
            and candidate.get("core_admission_review_version") == REVIEW_VERSION
            and _is_true(candidate.get("main_routine_default_allowed"))):
        return True
    has_explicit_mainstream_name = _candidate_name_has_any(candidate, _MAINSTREAM_GYM_PATTERNS)
    score = _optional_int(candidate.get("mainstream_gym_score"))
    if score is not None:
        tier = str(candidate.get("exercise_familiarity_tier") or "").strip().lower()
        return (
            has_explicit_mainstream_name
            or score >= 80
            or tier == "common"
        )
    return has_explicit_mainstream_name


def _is_uncommon_default_gym_candidate(candidate: dict) -> bool:
    if _is_false(candidate.get("main_routine_default_allowed")):
        return True
    score = _optional_int(candidate.get("mainstream_gym_score"))
    if score is not None:
        tier = str(candidate.get("exercise_familiarity_tier") or "").strip().lower()
        novelty = str(candidate.get("novelty_level") or "").strip().lower()
        return (
            score <= 55
            or tier in {"specialized", "obscure"}
            or novelty in {"medium", "high"}
            or _candidate_name_has_any(candidate, _UNCOMMON_DEFAULT_GYM_TOKENS)
        )
    return _candidate_name_has_any(candidate, _UNCOMMON_DEFAULT_GYM_TOKENS)


def _is_explicitly_reviewed_mainstream_candidate(candidate: dict) -> bool:
    score = _optional_int(candidate.get("mainstream_gym_score"))
    familiarity = str(
        candidate.get("exercise_familiarity_tier") or ""
    ).strip().lower()
    novelty = str(candidate.get("novelty_level") or "").strip().lower()
    return (
        _is_true(candidate.get("main_routine_default_allowed"))
        and score is not None
        and score >= 80
        and familiarity == "common"
        and novelty not in {"medium", "high"}
    )


def _is_common_default_gym_candidate(candidate: dict) -> bool:
    score = _optional_int(candidate.get("mainstream_gym_score"))
    familiarity = str(
        candidate.get("exercise_familiarity_tier") or ""
    ).strip().lower()
    novelty = str(candidate.get("novelty_level") or "").strip().lower()
    return (
        not _is_false(candidate.get("main_routine_default_allowed"))
        and score is not None
        and score >= 80
        and familiarity == "common"
        and novelty not in {"medium", "high"}
    )


def _requires_explicit_deep_squat_position(candidate: dict) -> bool:
    if _is_true(candidate.get("deep_squat_position_required")):
        return True
    compact_name = _candidate_name_compact(candidate)
    is_squat_family = "스쿼트" in compact_name or "squat" in compact_name
    return is_squat_family and any(
        marker in compact_name
        for marker in ("atg", "딥", "deep", "풀", "full")
    )


def _requires_overhead_position(candidate: dict) -> bool:
    if _is_true(candidate.get("overhead_position_required")):
        return True
    compact_name = _candidate_name_compact(candidate)
    return "오버헤드" in compact_name or "overhead" in compact_name


def _is_specialty_mobility_default_gym_candidate(candidate: dict) -> bool:
    return _candidate_name_has_any(candidate, _SPECIALTY_MOBILITY_DEFAULT_GYM_TOKENS)


def _is_crossfit_style_candidate(candidate: dict) -> bool:
    if "crossfit" in _metadata_style_tags(candidate):
        return True
    return _candidate_name_has_any(candidate, _CROSSFIT_STYLE_TOKENS)


def _normalize_training_style(value: Optional[str]) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _TRAINING_STYLE_ALIASES.get(normalized, normalized)


def _is_default_gym_style(value: Optional[str]) -> bool:
    return _normalize_training_style(value) in _GYM_DEFAULT_STYLES


def _uses_gym_base_formula(value: Optional[str]) -> bool:
    normalized = _normalize_training_style(value)
    return normalized in _GYM_DEFAULT_STYLES or normalized in _GYM_BASE_ADJUNCT_STYLES


def _normalize_constraint_tokens(values: Optional[List[str]]) -> set[str]:
    return {
        str(value).strip().lower().replace("-", "_")
        for value in (values or [])
        if str(value).strip()
    }


def _normalize_body_part(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _candidate_family_key(candidate: dict) -> str:
    """Return the candidate-pool family while preserving requested adjunct roles.

    Final selection still collapses equipment variants into one movement family.
    The ranked pool, however, must retain both a low-load adjunct variant and a
    normal working-set variant when they serve different blueprint slots.
    """
    base_key = candidate_selection_family_key(candidate)
    adjunct_codes = _candidate_requested_adjunct_codes(candidate)
    if adjunct_codes:
        return f"{base_key}:adjunct:{','.join(sorted(adjunct_codes))}"
    return base_key


def _candidate_requested_adjunct_codes(candidate: dict) -> set[str]:
    adjunct_codes: set[str] = set()
    for boost in candidate.get("score_boosts") or []:
        if not isinstance(boost, dict):
            continue
        rule_code = str(boost.get("rule_code") or boost.get("ruleCode") or "").strip()
        if rule_code != "adjunct_option_match_boost":
            continue
        constraint_code = str(
            boost.get("constraint_code") or boost.get("constraintCode") or ""
        ).strip()
        if constraint_code:
            adjunct_codes.add(constraint_code)
    return adjunct_codes


def candidate_selection_family_key(candidate: dict) -> str:
    """Return an equipment-agnostic movement family for one final routine.

    Candidate pools may intentionally retain equipment alternatives. The final
    routine should not spend two slots on the same movement merely because one
    version uses a cable and another uses a machine.
    """
    raw_name = candidate_display_name(candidate).strip().lower()
    compact = re.sub(r"[\s/_()·,.-]+", "", raw_name)
    primary = str(candidate.get("primary_muscle") or "").strip().lower()

    def has(*patterns: str) -> bool:
        return any(re.sub(r"\s+", "", pattern.lower()) in compact for pattern in patterns)

    angle = ""
    if has("인클라인", "incline"):
        angle = "incline"
    elif has("디클라인", "decline"):
        angle = "decline"
    elif has("플랫", "flat"):
        angle = "flat"

    if has("푸시다운", "pushdown", "push down"):
        return "triceps_pushdown"
    if primary == "biceps" and has("컬", "curl"):
        if has("해머", "hammer", "뉴트럴", "neutral", "크로스바디", "crossbody"):
            curl_pattern = "hammer"
        elif has("리버스", "reverse"):
            curl_pattern = "reverse"
        elif has("프리처", "스콧", "preacher", "scott"):
            curl_pattern = "preacher"
        elif has("인클라인", "incline"):
            curl_pattern = "incline"
        elif has("스파이더", "spider"):
            curl_pattern = "spider"
        elif has("콘센트레이션", "concentration"):
            curl_pattern = "concentration"
        elif has("드래그", "drag"):
            curl_pattern = "drag"
        elif has("베이지안", "bayesian"):
            curl_pattern = "bayesian"
        else:
            curl_pattern = "standard"
        return f"biceps_curl:{curl_pattern}"
    if primary == "forearm" and has(
        "회내/회외", "회내회외", "pronationsupination", "pronation supination"
    ):
        return "forearm_rotation:pronation_supination"
    if has("백익스텐션", "백 익스텐션", "backextension", "back extension"):
        return "back_extension"
    if has("플라이", "크로스오버", "fly", "crossover"):
        direction = "reverse" if has("리버스", "리어", "후면", "reverse", "rear") else "forward"
        return f"fly:{direction}:{angle or 'unspecified'}"
    if has("벤치프레스", "벤치 프레스", "체스트프레스", "체스트 프레스", "benchpress", "bench press", "chestpress", "chest press"):
        return f"horizontal_press:{angle or 'unspecified'}"
    if has("숄더프레스", "숄더 프레스", "오버헤드프레스", "오버헤드 프레스", "밀리터리프레스", "shoulderpress", "shoulder press", "overheadpress", "overhead press", "militarypress"):
        return "vertical_press"
    if has("프론트레이즈", "프론트 레이즈", "frontraise", "front raise"):
        return "front_raise"
    if has("레그컬", "레그 컬", "legcurl", "leg curl"):
        position = "seated" if has("시티드", "seated") else "lying" if has("라잉", "lying", "prone") else "standing" if has("스탠딩", "standing") else "unspecified"
        return f"leg_curl:{position}"
    if has("레그익스텐션", "레그 익스텐션", "legextension", "leg extension"):
        return "leg_extension"
    if has("랫풀다운", "랫 풀다운", "latpulldown", "lat pulldown"):
        return "lat_pulldown"
    if (
        primary in {"upper-back", "trapezius"}
        and has("로우", "row")
        and not has("로우투하이 플라이", "lowtohighfly", "low to high fly")
    ):
        return "horizontal_row"
    if has("불가리안스플릿스쿼트", "스플릿스쿼트", "bulgariansplitsquat", "splitsquat"):
        return "split_squat"
    if has("사이드런지", "레터럴런지", "sidelunge", "laterallunge"):
        return "lunge:side"
    if has("리버스런지", "백워드런지", "reverselunge", "backwardlunge"):
        return "lunge:reverse"
    if has("런지", "lunge"):
        return "lunge:standard"
    if has("스쿼트", "squat"):
        # A routine should not spend two slots on Smith, dumbbell-wide, or
        # barbell versions of the same bilateral squat pattern. Unilateral
        # split squats and lunges are handled above as separate families.
        return "squat"

    normalized = raw_name
    for token in _SELECTION_FAMILY_EQUIPMENT_TOKENS:
        normalized = normalized.replace(token, " ")
    normalized = re.sub(r"v\d+[-_]\d+", " ", normalized)
    normalized = re.sub(r"[\s/_()·,.-]+", "", normalized)
    movement_type = str(candidate.get("movement_type") or "").strip().lower()
    return f"{primary}:{movement_type}:{normalized or compact}"


def candidate_selection_pattern_key(candidate: dict) -> str:
    """Return a coarse movement pattern used to prevent redundant routines.

    ``candidate_selection_family_key`` intentionally keeps useful variations,
    such as flat and incline presses, separate. Final routine assembly also
    needs a broader guard so those variations do not consume most of a short
    session merely to hit an exercise-count target.
    """
    family = candidate_selection_family_key(candidate)
    raw_name = candidate_display_name(candidate).strip().lower()
    compact = re.sub(r"[\s/_()·,.-]+", "", raw_name)

    def has(*patterns: str) -> bool:
        return any(re.sub(r"\s+", "", pattern.lower()) in compact for pattern in patterns)

    if family.startswith("horizontal_press:"):
        return "horizontal_press"
    if family.startswith("biceps_curl:"):
        return "biceps_curl"
    if family.startswith("leg_curl:"):
        return "leg_curl"
    if family.startswith("fly:forward:"):
        return "chest_fly"
    if family.startswith("fly:reverse:"):
        return "rear_delt_fly"
    if family == "lat_pulldown" or has(
        "풀업",
        "친업",
        "턱걸이",
        "pullup",
        "pull-up",
        "chinup",
        "chin-up",
    ):
        return "vertical_pull"
    if has(
        "레터럴레이즈",
        "레터럴 레이즈",
        "사이드레이즈",
        "사이드 레이즈",
        "lateralraise",
        "lateral raise",
        "sideraise",
        "side raise",
    ):
        return "lateral_raise"
    if family == "front_raise":
        return "front_raise"
    return family


def _candidate_display_name(candidate: dict) -> str:
    return candidate_display_name(candidate)


def _has_generic_display_text(value: str) -> bool:
    return is_generic_candidate_display_text(value)


def _preserve_full_body_flat_press_equipment_choice(
    selected: List[dict], scored: List[dict],
) -> List[dict]:
    """Trade a redundant chest angle for an eligible free-weight alternative.

    Full-body has one horizontal-push role. Preserve its choice without taking
    capacity from another role or changing the final routine's family limits.
    Both lists must already have passed request safety/equipment filtering.
    """
    flat_family = "horizontal_press:flat"

    def eligible(candidate: dict) -> bool:
        return (
            candidate_selection_family_key(candidate) == flat_family
            and str(candidate.get("primary_muscle") or "") == "chest"
            and str(candidate.get("movement_type") or "") == "COMPOUND"
            and _is_true(candidate.get("main_routine_default_allowed"))
            and main_role_eligibility(candidate, "intermediate", "hypertrophy").eligible
        )

    available = {
        str(candidate.get("equipment_req") or "").upper()
        for candidate in selected if eligible(candidate)
    } & {"BARBELL", "DUMBBELL"}
    if len(available) != 1:
        return selected
    missing = ({"BARBELL", "DUMBBELL"} - available).pop()
    selected_ids = {str(candidate.get("id")) for candidate in selected}
    alternative = next((candidate for candidate in scored
        if str(candidate.get("id")) not in selected_ids
        and str(candidate.get("equipment_req") or "").upper() == missing
        and eligible(candidate)), None)
    donor_index = next((index for index in range(len(selected) - 1, -1, -1)
        if str(selected[index].get("primary_muscle") or "") == "chest"
        and candidate_selection_family_key(selected[index])
        in {"horizontal_press:incline", "horizontal_press:decline"}
        and not _candidate_requested_adjunct_codes(selected[index])), None)
    if alternative is None or donor_index is None:
        return selected
    result = list(selected)
    result[donor_index] = alternative
    return sorted(result, key=lambda candidate: (
        -int(candidate.get("score") or 0),
        int(candidate.get("efficiency_tier") or 99),
        str(candidate.get("id") or ""),
    ))


def _has_generic_display_name(candidate: dict) -> bool:
    name = _candidate_display_name(candidate).strip().lower()
    return _has_generic_display_text(name)


def _target_primary_cap(top_n: int, target_muscles: Optional[List[str]]) -> int:
    target_count = len([target for target in (target_muscles or []) if str(target).strip()])
    if target_count >= 5:
        return 2
    if target_count >= 3:
        return max(2, top_n // 3)
    return top_n


def _diversify_ranked_candidates(
    candidates: List[dict],
    top_n: int,
    target_muscles: Optional[List[str]] = None,
) -> List[dict]:
    if top_n <= 0:
        return []
    selected: List[dict] = []
    overflow: List[dict] = []
    family_counts: Dict[str, int] = {}
    primary_counts: Dict[str, int] = {}
    selected_ids: set[str] = set()
    target_order = [str(target).strip() for target in (target_muscles or []) if str(target).strip()]
    target_set = set(target_order)
    primary_cap = _target_primary_cap(top_n, target_order)

    def can_take(
        candidate: dict,
        *,
        enforce_primary_cap: bool = True,
        enforce_family_cap: bool = True,
    ) -> bool:
        candidate_id = str(candidate.get("id") or "")
        if candidate_id in selected_ids:
            return False
        family_key = _candidate_family_key(candidate)
        if enforce_family_cap and family_counts.get(family_key, 0) >= _MAX_CANDIDATES_PER_FAMILY:
            return False
        primary = str(candidate.get("primary_muscle") or "")
        if enforce_primary_cap and len(target_set) > 1 and primary in target_set:
            if primary_counts.get(primary, 0) >= primary_cap:
                return False
        return True

    def take(candidate: dict) -> None:
        selected.append(candidate)
        selected_ids.add(str(candidate.get("id") or ""))
        family_key = _candidate_family_key(candidate)
        family_counts[family_key] = family_counts.get(family_key, 0) + 1
        primary = str(candidate.get("primary_muscle") or "")
        if not _candidate_requested_adjunct_codes(candidate):
            primary_counts[primary] = primary_counts.get(primary, 0) + 1

    for target in target_order:
        if len(selected) >= top_n:
            break
        for candidate in candidates:
            if str(candidate.get("primary_muscle") or "") != target:
                continue
            if can_take(candidate, enforce_primary_cap=False):
                take(candidate)
                break

    for candidate in candidates:
        if len(selected) >= top_n:
            break
        family_key = _candidate_family_key(candidate)
        count = family_counts.get(family_key, 0)
        if count < _MAX_CANDIDATES_PER_FAMILY and can_take(candidate):
            take(candidate)
        else:
            overflow.append(candidate)
    if len(selected) < top_n:
        for candidate in overflow:
            if len(selected) >= top_n:
                break
            if can_take(candidate, enforce_primary_cap=False, enforce_family_cap=False):
                take(candidate)
    return sorted(
        selected[:top_n],
        key=lambda ex: (
            -int(ex.get("score") or 0),
            int(ex.get("efficiency_tier") or 99),
            str(ex.get("id") or ""),
        ),
    )


def _ensure_requested_style_adjunct(
    *,
    selected: List[dict],
    sorted_candidates: List[dict],
    normalized_training_style: str,
    top_n: int,
) -> List[dict]:
    """Keep Fit-Core gym-based while preserving one requested adjunct style option.

    The product's current routine formula remains gym strength/hypertrophy.
    When the user explicitly asks for CrossFit, mobility, yoga/Pilates, or
    rehab/prehab as an option, at least one safe style-matched candidate should
    reach the LLM candidate pool so it can be used as an accessory/substitution.
    """
    if (
        top_n <= 0
        or normalized_training_style not in _GYM_BASE_ADJUNCT_STYLES
        or any(normalized_training_style in _metadata_style_tags(candidate) for candidate in selected)
    ):
        return selected[:top_n]

    selected_ids = {str(candidate.get("id") or "") for candidate in selected}
    style_candidate = next(
        (
            candidate
            for candidate in sorted_candidates
            if str(candidate.get("id") or "") not in selected_ids
            and normalized_training_style in _metadata_style_tags(candidate)
        ),
        None,
    )
    if style_candidate is None:
        return selected[:top_n]

    if len(selected) < top_n:
        return (selected + [style_candidate])[:top_n]

    if top_n == 1:
        return selected[:top_n]

    preserved = selected[: top_n - 1]
    return (preserved + [style_candidate])[:top_n]


def _ensure_specialization_focus_candidate(
    *,
    selected: List[dict],
    sorted_candidates: List[dict],
    specialization_focus: Optional[str],
    adjunct_options: Optional[List[str]],
    target_muscles: Optional[List[str]],
    top_n: int,
) -> List[dict]:
    """Reserve reviewed, distinct focus patterns without displacing coverage."""

    focus_code = str(specialization_focus or "").strip()
    if top_n <= 0 or not focus_code:
        return selected[:top_n]
    requested_adjuncts = list(canonical_adjunct_options(adjunct_options or []))
    result = list(selected[:top_n])

    def candidate_id(candidate: dict) -> str:
        return str(candidate.get("id") or "").strip()

    def focus_pattern(candidate: dict) -> Optional[str]:
        if not candidate_matches_specialization_focus(candidate, focus_code):
            return None

        family = candidate_selection_family_key(candidate)
        if focus_code == "hamstringEmphasis":
            movement_type = str(
                candidate.get("canonical_movement_type")
                or candidate.get("movement_type")
                or ""
            ).strip().upper()
            hinge_load = _level(candidate.get("hip_hinge_load"))
            if movement_type == "COMPOUND" and hinge_load == "high":
                return "hamstring:direct_hinge"
            return f"hamstring:{family}"
        if focus_code == "bicepsEmphasis":
            return family if family.startswith("biceps_curl:") else None
        if focus_code == "forearmEmphasis":
            name = _candidate_name_compact(candidate)
            has_wrist_curl = any(token in name for token in ("리스트컬", "wristcurl"))
            if has_wrist_curl and any(token in name for token in ("리버스", "reverse")):
                return "forearm:reverse_wrist_curl"
            if has_wrist_curl:
                return "forearm:wrist_curl"
            if any(
                token in name
                for token in (
                    "회내/회외",
                    "회내회외",
                    "pronationsupination",
                )
            ):
                return "forearm:pronation_supination"
            if any(token in name for token in ("조트만", "zottman")):
                return "forearm:zottman_curl"
            return f"forearm:{family}"
        return family

    def is_reservable(candidate: dict) -> bool:
        reviewed_specialized_focus = (
            _is_true(candidate.get("core_forearm_pattern_reviewed"))
            and candidate.get("core_admission_review_version")
            == "forearm-pattern-admission-v1-2026-09-17"
            and candidate.get("core_admission_scope") == focus_code
        )
        return (
            candidate_matches_specialization_focus(candidate, focus_code)
            and _is_true(candidate.get("main_routine_default_allowed"))
            and (
                _is_mainstream_gym_candidate(candidate)
                or reviewed_specialized_focus
            )
        )

    available_by_pattern: dict[str, dict] = {}
    for candidate in sorted_candidates:
        pattern = focus_pattern(candidate)
        if pattern and is_reservable(candidate):
            available_by_pattern.setdefault(pattern, candidate)

    if not available_by_pattern:
        return result

    if focus_code == "hamstringEmphasis":
        desired_patterns = ["hamstring:direct_hinge"]
        desired_patterns.extend(
            pattern
            for pattern in available_by_pattern
            if pattern != "hamstring:direct_hinge"
        )
        desired_patterns = [
            pattern for pattern in desired_patterns if pattern in available_by_pattern
        ][:2]
    elif focus_code == "bicepsEmphasis":
        desired_patterns = list(available_by_pattern)[:3]
    elif focus_code == "forearmEmphasis":
        desired_patterns = [
            pattern
            for pattern in (
                "forearm:wrist_curl",
                "forearm:reverse_wrist_curl",
                "forearm:pronation_supination",
            )
            if pattern in available_by_pattern
        ]
    else:
        # Preserve the existing one-candidate behavior for other focus codes.
        desired_patterns = [next(iter(available_by_pattern))]

    desired_candidates = [available_by_pattern[pattern] for pattern in desired_patterns]
    desired_ids = {candidate_id(candidate) for candidate in desired_candidates}
    focus_targets = {
        "hamstringEmphasis": {"hamstring"},
        "bicepsEmphasis": {"biceps"},
        "forearmEmphasis": {"forearm"},
    }.get(focus_code, set())
    protected_targets = {
        str(target or "").strip().lower().replace("_", "-")
        for target in (target_muscles or [])
        if str(target or "").strip()
    } - focus_targets

    def is_adjunct(candidate: dict) -> bool:
        return any(
            candidate_matches_adjunct_option(candidate, option)
            for option in requested_adjuncts
        )

    def preserves_required_target(index: int) -> bool:
        primary = str(result[index].get("primary_muscle") or "").strip().lower()
        if primary not in protected_targets:
            return True
        if focus_code == "forearmEmphasis" and primary == "biceps":
            eligible = [
                (position, candidate_selection_family_key(candidate))
                for position, candidate in enumerate(result)
                if str(candidate.get("primary_muscle") or "").strip().lower() == primary
                and _is_true(candidate.get("main_routine_default_allowed"))
            ]
            families = {family for _, family in eligible}
            remaining = {family for position, family in eligible if position != index}
            if len(remaining) < min(2, len(families)):
                return False
        return sum(
            str(candidate.get("primary_muscle") or "").strip().lower() == primary
            for candidate in result
        ) > 1

    def donor_index() -> Optional[int]:
        # Replace a focus equipment clone first, then only a non-protected donor.
        for require_duplicate_focus in (True, False):
            for index in range(len(result) - 1, -1, -1):
                candidate = result[index]
                candidate_pattern = focus_pattern(candidate)
                if candidate_id(candidate) in desired_ids or is_adjunct(candidate):
                    continue
                if require_duplicate_focus:
                    if not candidate_pattern or candidate_pattern not in desired_patterns:
                        continue
                elif candidate_pattern and candidate_pattern in desired_patterns:
                    continue
                if preserves_required_target(index):
                    return index
        return None

    for candidate in desired_candidates:
        if candidate_id(candidate) in {candidate_id(item) for item in result}:
            continue
        if len(result) < top_n:
            result.append(candidate)
            continue
        replace_index = donor_index()
        if replace_index is None:
            continue
        result[replace_index] = candidate

    return result[:top_n]


def _ensure_requested_adjunct_option(
    *,
    selected: List[dict],
    sorted_candidates: List[dict],
    adjunct_options: Optional[List[str]],
    top_n: int,
) -> List[dict]:
    options = list(canonical_adjunct_options(adjunct_options or []))
    if top_n <= 0 or not options:
        return selected[:top_n]
    result = list(selected[:top_n])
    reserved_ids: set[str] = set()

    def candidate_id(candidate: dict) -> str:
        return str(candidate.get("id") or "").strip()

    def best_distinct_match(
        pool: List[dict],
        option: str,
        *,
        explicit_only: bool,
    ) -> Optional[dict]:
        available = [
            candidate
            for candidate in pool
            if candidate_id(candidate) not in reserved_ids
        ]
        explicit = next(
            (
                candidate
                for candidate in available
                if option in candidate_adjunct_option_codes(candidate)
                and candidate_matches_adjunct_option(candidate, option)
            ),
            None,
        )
        if explicit is not None or explicit_only:
            return explicit
        return next(
            (
                candidate
                for candidate in available
                if candidate_matches_adjunct_option(candidate, option)
            ),
            None,
        )

    for option in options:
        existing = best_distinct_match(result, option, explicit_only=True)
        if existing is not None:
            reserved_ids.add(candidate_id(existing))
            continue

        result_ids = {candidate_id(candidate) for candidate in result}
        candidate = best_distinct_match(
            [
                item
                for item in sorted_candidates
                if candidate_id(item) not in result_ids
            ],
            option,
            explicit_only=True,
        )
        if candidate is None:
            existing = best_distinct_match(result, option, explicit_only=False)
            if existing is not None:
                reserved_ids.add(candidate_id(existing))
                continue
            candidate = best_distinct_match(
                [
                    item
                    for item in sorted_candidates
                    if candidate_id(item) not in result_ids
                ],
                option,
                explicit_only=False,
            )
            if candidate is None:
                continue
        if len(result) < top_n:
            result.append(candidate)
            reserved_ids.add(candidate_id(candidate))
            continue
        replace_index = next(
            (
                index
                for index in range(len(result) - 1, -1, -1)
                if candidate_id(result[index]) not in reserved_ids
            ),
            None,
        )
        if replace_index is None:
            continue
        result[replace_index] = candidate
        reserved_ids.add(candidate_id(candidate))
    return result[:top_n]


def _ensure_pull_back_choices(
    *, selected: List[dict], sorted_candidates: List[dict], top_n: int,
) -> List[dict]:
    """Reserve back alternatives inside the pool, never extra routine exercises.

    Input is already filtered. Keep existing primary coverage and the best
    direct curl; replace surplus accessory alternatives, not safety decisions.
    """
    if top_n < 4:
        return selected[:top_n]
    result = list(selected[:top_n])
    from .pull_composition import reviewed_row_setup

    eligible = [c for c in sorted_candidates
                if _is_true(c.get("main_routine_default_allowed"))
                and (_is_mainstream_gym_candidate(c)
                     or reviewed_row_setup(c) == "chest_supported")]
    reserved: list[dict] = []
    vertical_families: set[str] = set()
    for pattern in ("horizontal_row", "vertical_pull", "biceps_curl"):
        for candidate in eligible:
            primary = str(candidate.get("primary_muscle") or "")
            if primary not in ({"biceps"} if pattern == "biceps_curl" else {"upper-back", "trapezius"}):
                continue
            if candidate_selection_pattern_key(candidate) != pattern:
                continue
            family = candidate_selection_family_key(candidate)
            if pattern == "vertical_pull" and family in vertical_families:
                continue
            reserved.append(candidate)
            if pattern != "vertical_pull":
                break
            vertical_families.add(family)
            if len(vertical_families) == 2:
                break
    protected = {str(c.get("id")) for c in reserved}
    # Only the scoped, identity-reviewed pair can reserve an additional row.
    setup_choices = {
        setup: next((c for c in eligible if reviewed_row_setup(c) == setup), None)
        for setup in ("bent_over", "chest_supported")
    }
    if all(setup_choices.values()):
        reserved.extend(setup_choices.values())
        protected.update(str(c.get("id")) for c in setup_choices.values())
    for candidate in reserved:
        if any(c.get("id") == candidate.get("id") for c in result):
            continue
        if len(result) < top_n:
            result.append(candidate)
            continue
        # Count again after every replacement so no primary's last candidate
        # can be displaced by successive reservations.
        primary_counts: Dict[str, int] = {}
        for item in result:
            primary = str(item.get("primary_muscle") or "")
            primary_counts[primary] = primary_counts.get(primary, 0) + 1
        donor = next((i for i in range(len(result) - 1, -1, -1)
                      if str(result[i].get("id")) not in protected
                      and primary_counts[str(result[i].get("primary_muscle") or "")] > 1), None)
        if donor is not None:
            result[donor] = candidate
    return sorted(result, key=lambda c: (
        -int(c.get("score") or 0), int(c.get("efficiency_tier") or 99), str(c.get("id") or ""),
    ))


def _ensure_full_body_vertical_press_candidate(
    *,
    selected: List[dict],
    sorted_candidates: List[dict],
    target_muscles: Optional[List[str]],
    target_split_label: Optional[str],
    specialization_focus: Optional[str],
    adjunct_options: Optional[List[str]],
    top_n: int,
) -> List[dict]:
    """Preserve one qualified front-delt vertical press in full-body pools."""
    if (
        top_n <= 0
        or str(target_split_label or "").strip().lower().replace("-", "_") != "full_body"
        or "front-deltoids" not in {
            str(target or "").strip().lower().replace("_", "-")
            for target in (target_muscles or [])
        }
    ):
        return selected[:top_n]

    result = list(selected[:top_n])
    candidate_id = lambda candidate: str(candidate.get("id") or "").strip()
    selected_ids = {candidate_id(candidate) for candidate in result}
    if any(
        candidate_selection_pattern_key(candidate) == "vertical_press"
        and str(candidate.get("primary_muscle") or "").strip().lower() == "front-deltoids"
        for candidate in result
    ):
        return result

    press = next(
        (
            candidate
            for candidate in sorted_candidates
            if candidate_id(candidate) not in selected_ids
            and str(candidate.get("primary_muscle") or "").strip().lower() == "front-deltoids"
            and candidate_selection_pattern_key(candidate) == "vertical_press"
            and _is_true(candidate.get("main_routine_default_allowed"))
            and _is_mainstream_gym_candidate(candidate)
        ),
        None,
    )
    if press is None:
        return result
    if len(result) < top_n:
        return (result + [press])[:top_n]

    requested_adjuncts = list(canonical_adjunct_options(adjunct_options or []))
    focus_code = str(specialization_focus or "").strip()
    for index in range(len(result) - 1, -1, -1):
        donor = result[index]
        if candidate_id(donor) == candidate_id(press):
            continue
        if any(candidate_matches_adjunct_option(donor, option) for option in requested_adjuncts):
            continue
        if focus_code and candidate_matches_specialization_focus(donor, focus_code):
            continue
        if str(donor.get("primary_muscle") or "").strip().lower() != "front-deltoids":
            continue
        if candidate_selection_pattern_key(donor) not in {"front_raise", "vertical_press"}:
            continue
        if candidate_selection_pattern_key(donor) == "vertical_press":
            continue
        result[index] = press
        return result[:top_n]
    return result


def _pain_tokens_from_entries(pain_areas: List[PainAreaEntry]) -> set[str]:
    tokens = {
        _normalize_body_part(p.body_part)
        for p in pain_areas
        if isinstance(p, PainAreaEntry) and p.body_part
    }
    if tokens & {"left-soleus", "right-soleus", "soleus"}:
        tokens.update({"soleus", "calves", "ankle"})
    if tokens & {"front-deltoids", "back-deltoids", "side-deltoids"}:
        tokens.update({"shoulder"})
    if "shoulder" in tokens:
        tokens.update({"front-deltoids", "back-deltoids", "side-deltoids"})
    return tokens


def _primary_muscle_conflicts_with_pain(candidate: dict, pain_tokens: set[str]) -> bool:
    primary = _normalize_body_part(candidate.get("primary_muscle"))
    return bool(primary and primary in pain_tokens)


def _secondary_muscle_pain_overlap(candidate: dict, pain_tokens: set[str]) -> set[str]:
    secondary = {
        _normalize_body_part(part)
        for part in str(candidate.get("secondary_muscle") or "").split(",")
        if part.strip()
    }
    return secondary & pain_tokens


_KNEE_ARTICULATION_NAME_TOKENS = {
    "legcurl", "레그컬", "hamstringcurl", "햄스트링컬",
    "legextension", "레그익스텐션", "kneeextension", "kneeflexion",
    "squat", "스쿼트", "lunge", "런지", "legpress", "레그프레스",
    "stepup", "스텝업", "splitsquat", "스플릿스쿼트",
    "deadlift", "데드리프트", "romaniandeadlift", "루마니안데드리프트", "rdl",
}
_ANKLE_ARTICULATION_NAME_TOKENS = {
    "calfraise", "카프레이즈", "tibialis", "티비알리스", "soleus", "가자미근",
    "squat", "스쿼트", "lunge", "런지", "stepup", "스텝업",
}
_SHOULDER_ARTICULATION_NAME_TOKENS = {
    "press", "프레스", "overhead", "오버헤드",
    "chestpress", "체스트프레스", "benchpress", "벤치프레스",
    "shoulderpress", "숄더프레스", "overheadpress", "오버헤드프레스",
    "jmpress", "jm프레스",
    "fly", "플라이", "crossover", "크로스오버", "pulldown", "풀다운",
    "pullup", "풀업", "row", "로우", "raise", "레이즈",
    "pushup", "푸시업", "dip", "딥", "pullover", "풀오버",
    "pecdeck", "펙덱", "butterfly", "버터플라이", "kickback", "킥백",
}
_LUMBAR_INVOLVEMENT_NAME_TOKENS = {
    "deadlift", "데드리프트", "goodmorning", "굿모닝",
    "swing", "스윙", "hiphinge", "힙힌지",
    "hipthrust", "힙쓰러스트", "glutebridge", "글루트브릿지",
    "backextension", "백익스텐션", "hyperextension", "하이퍼익스텐션",
    "bentover", "벤트오버", "bentoverrow", "벤트오버로우",
    "barbellrow", "바벨로우",
}
_UPPER_BODY_INVOLVEMENT_MUSCLES = {
    "chest", "upper-back", "trapezius", "front-deltoids", "back-deltoids",
    "side-deltoids", "biceps", "triceps", "forearm", "neck",
}
_LOWER_BODY_INVOLVEMENT_MUSCLES = {
    "quadriceps", "hamstring", "gluteal", "calves", "adductor", "abductors",
    "soleus", "left-soleus", "right-soleus",
}


def _compact_candidate_names(candidate: dict) -> str:
    return " ".join(
        re.sub(r"[\s_\-/]+", "", str(candidate.get(key) or "").strip().lower())
        for key in (
            "name_kr",
            "name_en",
            "display_name_ko",
            "base_exercise_name_kr",
            "variation_name_kr",
            "catalog_original_name_kr",
            "movement_pattern",
        )
    )


def _candidate_joint_involvement(candidate: dict) -> set[str]:
    """Infer direct joint articulation, not injury risk severity.

    Risk fields describe load magnitude and cannot prove non-involvement when
    their value is ``low``. Explicit metadata wins; compact movement-family
    names cover high-confidence catalog gaps such as leg curls and chest flys.
    """
    joints = {
        _normalize_body_part(token)
        for key in ("joint_involvement", "joint_involvement_tokens", "involved_joints")
        for token in str(candidate.get(key) or "").replace(";", ",").split(",")
        if token.strip()
    }
    compact_names = _compact_candidate_names(candidate)
    muscles = {
        _normalize_body_part(part)
        for key in (
            "primary_muscle",
            "secondary_muscle",
            "catalog_source_primary_muscle",
            "catalog_source_secondary_muscle",
        )
        for part in str(candidate.get(key) or "").replace(";", ",").split(",")
        if part.strip()
    }
    movement_type = str(candidate.get("movement_type") or "").strip().upper()
    if (
        muscles & _LOWER_BODY_INVOLVEMENT_MUSCLES
        and (
            movement_type == "COMPOUND"
            or any(token in compact_names for token in _KNEE_ARTICULATION_NAME_TOKENS)
        )
    ):
        joints.add("knees")
    if (
        muscles & _LOWER_BODY_INVOLVEMENT_MUSCLES
        and any(token in compact_names for token in _ANKLE_ARTICULATION_NAME_TOKENS)
    ):
        joints.add("ankle")
    if (
        muscles & _UPPER_BODY_INVOLVEMENT_MUSCLES
        and (
            movement_type == "COMPOUND"
            or any(token in compact_names for token in _SHOULDER_ARTICULATION_NAME_TOKENS)
        )
    ):
        joints.add("shoulder")
    # The source catalog contains loaded press/curl rows whose equipment is
    # incorrectly recorded as BODYWEIGHT. Under the locked conservative injury
    # policy, upper-body work is therefore treated as wrist/forearm involvement
    # instead of trusting equipment metadata to prove non-involvement.
    if muscles & _UPPER_BODY_INVOLVEMENT_MUSCLES:
        joints.add("wrist")
    if any(token in compact_names for token in _LUMBAR_INVOLVEMENT_NAME_TOKENS):
        joints.add("lumbar")
    return joints


def _candidate_injury_overlap(candidate: dict, injury_tokens: set[str]) -> set[str]:
    """Return registered injury tokens touched by muscles or reviewed load metadata."""
    involved = {
        _normalize_body_part(candidate.get("primary_muscle")),
        *{
            _normalize_body_part(part)
            for part in str(candidate.get("secondary_muscle") or "").replace(";", ",").split(",")
            if part.strip()
        },
        *{
            _normalize_body_part(part)
            for part in str(candidate.get("pain_triggers") or "").replace(";", ",").split(",")
            if part.strip()
        },
        *{
            _normalize_body_part(part)
            for key in ("catalog_source_primary_muscle", "catalog_source_secondary_muscle")
            for part in str(candidate.get(key) or "").replace(";", ",").split(",")
            if part.strip()
        },
    }
    involved.discard("")
    overlaps = involved & injury_tokens
    joint_involvement = _candidate_joint_involvement(candidate)

    for injury in injury_tokens:
        if injury in {"knees", "knee"} and (
            bool(joint_involvement & {"knees", "knee"})
            or
            "quadriceps" in involved
            or _level_rank(candidate.get("knee_shear_load")) >= _LEVEL_RANK["medium"]
        ):
            overlaps.add(injury)
        elif injury in {"wrist", "forearm"} and (
            bool(joint_involvement & {"wrist"})
            or "forearm" in involved
            or _level_rank(candidate.get("wrist_stress")) >= _LEVEL_RANK["medium"]
        ):
            overlaps.add(injury)
        elif injury in {"ankle", "calves", "soleus", "left-soleus", "right-soleus"} and (
            bool(joint_involvement & {"ankle"})
            or bool(involved & {"calves", "soleus", "left-soleus", "right-soleus"})
            or _level_rank(candidate.get("ankle_dorsiflexion_demand")) >= _LEVEL_RANK["medium"]
        ):
            overlaps.add(injury)
        elif injury in {"lower-back", "back", "lumbar"} and (
            bool(joint_involvement & {"lumbar"})
            or "lower-back" in involved
            or _level_rank(candidate.get("lumbar_load")) >= _LEVEL_RANK["medium"]
        ):
            overlaps.add(injury)
        elif injury in {"shoulder", "front-deltoids", "back-deltoids", "side-deltoids"} and (
            bool(joint_involvement & {"shoulder"})
            or bool(involved & {"shoulder", "front-deltoids", "back-deltoids", "side-deltoids"})
            or _level_rank(candidate.get("shoulder_impingement_risk")) >= _LEVEL_RANK["medium"]
        ):
            overlaps.add(injury)

    return overlaps


def _infer_candidate_application_phase(candidate: dict, target_set: set[str], movement_type: str) -> str:
    primary = _normalize_body_part(candidate.get("primary_muscle"))
    style_tags = _metadata_style_tags(candidate)
    if style_tags & _GYM_BASE_ADJUNCT_STYLES and primary not in target_set:
        return "finisher"
    if movement_type == "COMPOUND" and primary in target_set:
        return "main_lift"
    if primary in ACCESSORY_MUSCLE_SLUGS or movement_type == "ISOLATION":
        return "accessory"
    return "main_lift" if primary in target_set else "accessory"


def _infer_intensity_intent(goal: Optional[str], readiness_level: Optional[str], training_style: Optional[str]) -> str:
    normalized_goal = str(goal or "").strip().lower()
    normalized_readiness = str(readiness_level or "").strip().lower()
    normalized_style = _normalize_training_style(training_style)
    if normalized_goal == "strength" and normalized_readiness != "low":
        return "heavy"
    if normalized_style == "crossfit":
        return "high_density"
    return "normal"


_DOMS_IMPLIED_MUSCLE_BY_FAMILY = {
    "quadriceps": {
        "스쿼트",
        "squat",
        "런지",
        "lunge",
        "스플릿스쿼트",
        "split squat",
        "불가리안",
        "bulgarian",
        "스텝업",
        "step up",
        "스텝다운",
        "step down",
        "레그프레스",
        "leg press",
        "프레스/스쿼트",
    },
    "chest": {
        "벤치프레스",
        "bench press",
        "체스트프레스",
        "chest press",
        "플라이",
        "fly",
        "크로스오버",
        "crossover",
        "푸쉬업",
        "push up",
        "push-up",
    },
    "upper-back": {
        "랫풀다운",
        "lat pulldown",
        "풀다운",
        "pulldown",
        "로우",
        "row",
        "풀업",
        "pull up",
        "pull-up",
    },
    "biceps": {
        "컬",
        "curl",
    },
    "triceps": {
        "트라이셉스",
        "triceps",
        "푸쉬다운",
        "pushdown",
        "익스텐션",
        "extension",
        "딥스",
        "dip",
    },
    "hamstring": {
        "레그컬",
        "leg curl",
        "루마니안데드리프트",
        "romanian deadlift",
        "rdl",
    },
}


def _candidate_doms_overlap(candidate: dict, doms: Dict[str, int]) -> Dict[str, int]:
    """Return DOMS body parts that this candidate directly or strongly loads.

    DB primary muscle alone is not enough for DOMS. Some broad gym movements
    are stored under a variant-specific primary muscle, but still strongly load
    another sore muscle. Example: a wide dumbbell squat may be stored as
    adductor-primary while still being a quadriceps-dominant squat pattern.
    """
    if not doms:
        return {}

    candidate_parts = {
        _normalize_body_part(candidate.get("primary_muscle")),
        *{
            _normalize_body_part(part)
            for part in str(candidate.get("secondary_muscle") or "").split(",")
            if part.strip()
        },
        *{
            _normalize_body_part(part)
            for part in str(candidate.get("pain_triggers") or "").split(",")
            if part.strip()
        },
    }
    candidate_parts.discard("")

    compact_name = _candidate_name_compact(candidate)
    for body_part, patterns in _DOMS_IMPLIED_MUSCLE_BY_FAMILY.items():
        if any(pattern.replace(" ", "").lower() in compact_name for pattern in patterns):
            candidate_parts.add(body_part)

    return {
        body_part: level
        for raw_part, level in doms.items()
        for body_part in [_normalize_body_part(raw_part)]
        if body_part in candidate_parts
    }


def _candidate_doms_prescription_signal(candidate: dict, doms: Dict[str, int]) -> Dict[str, int]:
    """Split direct/strong DOMS loading from secondary involvement."""
    if not doms:
        return {"direct_level": 0, "secondary_level": 0}

    direct_parts = {_normalize_body_part(candidate.get("primary_muscle"))}
    secondary_parts = {
        _normalize_body_part(part)
        for field in ("secondary_muscle", "pain_triggers")
        for part in str(candidate.get(field) or "").split(",")
        if part.strip()
    }
    compact_name = _candidate_name_compact(candidate)
    for body_part, patterns in _DOMS_IMPLIED_MUSCLE_BY_FAMILY.items():
        if any(pattern.replace(" ", "").lower() in compact_name for pattern in patterns):
            direct_parts.add(body_part)
    direct_parts.discard("")
    secondary_parts -= direct_parts

    normalized_doms = {
        _normalize_body_part(raw_part): int(level or 0)
        for raw_part, level in doms.items()
    }
    return {
        "direct_level": max(
            (normalized_doms.get(part, 0) for part in direct_parts),
            default=0,
        ),
        "secondary_level": max(
            (normalized_doms.get(part, 0) for part in secondary_parts),
            default=0,
        ),
    }


def _apply_level_penalty(
    *,
    candidate: dict,
    field: str,
    high_delta: int,
    medium_delta: int,
    high_reason: str,
    medium_reason: str,
    rule_code: str,
    constraint_code: str,
    profile_signal_code: str,
    score_reasons: List[str],
    display_reasons: List[str],
    boosts: List[dict],
    penalties: List[dict],
) -> int:
    rank = _level_rank(candidate.get(field))
    if rank >= 3:
        return _record_score_delta(
            delta=high_delta,
            internal_reason=f"{field} high penalty",
            public_reason=high_reason,
            rule_code=rule_code,
            constraint_code=constraint_code,
            profile_signal_code=profile_signal_code,
            score_reasons=score_reasons,
            display_reasons=display_reasons,
            boosts=boosts,
            penalties=penalties,
        )
    if rank == 2:
        return _record_score_delta(
            delta=medium_delta,
            internal_reason=f"{field} medium penalty",
            public_reason=medium_reason,
            rule_code=rule_code,
            constraint_code=constraint_code,
            profile_signal_code=profile_signal_code,
            score_reasons=score_reasons,
            display_reasons=display_reasons,
            boosts=boosts,
            penalties=penalties,
        )
    return 0


def _condition_policy_matches_lumbar(condition_policies: List[dict]) -> bool:
    for policy in condition_policies:
        code = str(_dict_get(policy, "conditionCode", "condition_code") or "").lower()
        body_part = _normalize_body_part(_dict_get(policy, "bodyPart", "body_part"))
        if "lumbar" in code or "lower_back" in code or body_part in {"lower-back", "lumbar", "back"}:
            return True
    return False


def _matching_approved_condition_actions(
    candidate: dict,
    condition_policies: List[dict],
) -> List[dict]:
    """Return reviewed catalog actions matching the user's condition and phase."""
    active: Dict[str, set[str]] = {}
    for policy in condition_policies:
        code = str(_dict_get(policy, "conditionCode", "condition_code") or "").strip().lower()
        if not code:
            continue
        phase = str(_dict_get(policy, "phase") or "").strip().lower()
        active.setdefault(code, set()).add(phase)

    matches: List[dict] = []
    for action in candidate.get("approved_condition_policies") or []:
        if not isinstance(action, dict):
            continue
        if bool(action.get("review_required", 1)):
            continue
        code = str(action.get("condition_code") or "").strip().lower()
        if code not in active:
            continue
        action_phase = str(action.get("phase") or "").strip().lower()
        active_phases = active[code]
        if action_phase not in {"", "all", "any", "general"} and action_phase not in active_phases:
            continue
        matches.append(action)
    return matches


def _professional_clearance_missing(condition_policies: List[dict]) -> bool:
    return any(
        _is_false(_dict_get(policy, "professionalClearance", "professional_clearance"))
        for policy in condition_policies
    )


def _condition_policy_requires_clearance(condition_policies: List[dict]) -> bool:
    clearance_statuses = {
        "needs_clearance",
        "requires_clearance",
        "medical_clearance_required",
        "doctor_clearance_required",
        "post_surgery",
        "post-op",
        "postop",
        "수술후",
        "의료진허가필요",
    }
    for policy in condition_policies:
        if _is_false(_dict_get(policy, "professionalClearance", "professional_clearance")):
            return True
        status = str(_dict_get(policy, "status", "phase") or "").strip().lower().replace(" ", "_")
        if status in clearance_statuses:
            return True
        policy_type = str(_dict_get(policy, "policyType", "policy_type") or "").strip().lower()
        if policy_type == "surgeryhistory" and status:
            return True
    return False


def _is_beginner_experience(experience_level: Optional[str]) -> bool:
    normalized = str(experience_level or "").strip().lower()
    return normalized in BEGINNER_EXPERIENCE_ALIASES


def _is_intermediate_experience(experience_level: Optional[str]) -> bool:
    normalized = str(experience_level or "").strip().lower()
    return normalized in {
        "intermediate",
        "trained",
        "regular",
        "중급",
        "중급자",
    }


def _is_advanced_experience(experience_level: Optional[str]) -> bool:
    normalized = str(experience_level or "").strip().lower()
    return normalized in {
        "advanced",
        "expert",
        "athlete",
        "상급",
        "상급자",
        "고급",
        "고급자",
    }


def _is_supported_or_stable_candidate(candidate: dict) -> bool:
    equipment = _normalize_tokens(candidate.get("equipment_req"))
    if equipment & {"MACHINE", "SMITH_MACHINE", "CABLE", "BODYWEIGHT"}:
        return True
    name = " ".join([
        str(candidate.get("name_kr") or ""),
        str(candidate.get("name_en") or ""),
    ]).lower()
    compact = re.sub(r"[\s/_()·,.-]+", "", name)
    return any(token in compact for token in ("supported", "assist", "assisted", "서포티드", "어시스트", "고정식"))


def _has_progression_option(candidate: dict) -> bool:
    for option in candidate.get("substitution_options") or []:
        relation_type = str(_dict_get(option, "relation_type", "relationType") or "").strip().lower()
        if relation_type in {"regression", "progression", "mobilityalternative", "safetyalternative"}:
            return True
    return False


def _is_low_risk_clearance_alternative(candidate: dict) -> bool:
    return (
        _level_rank(candidate.get("injury_caution_level")) <= 1
        and _level_rank(candidate.get("pain_risk_level")) <= 1
        and _level_rank(candidate.get("lumbar_load")) <= 1
        and _level_rank(candidate.get("axial_load")) <= 1
        and _level_rank(candidate.get("shoulder_impingement_risk")) <= 1
        and _level_rank(candidate.get("wrist_stress")) <= 1
        and _level_rank(candidate.get("failure_penalty")) <= 1
        and _is_supported_or_stable_candidate(candidate)
        and not _is_sport_or_cardio_context_candidate(candidate)
        and not _is_ballistic_or_high_coordination_candidate(candidate)
    )


def _is_beginner_low_load_baseline_candidate(candidate: dict) -> bool:
    return (
        _is_supported_or_stable_candidate(candidate)
        and _level_rank(candidate.get("pain_risk_level")) <= 1
        and _level_rank(candidate.get("injury_caution_level")) <= 1
        and _level_rank(candidate.get("lumbar_load")) <= 1
        and _level_rank(candidate.get("axial_load")) <= 1
        and _level_rank(candidate.get("knee_shear_load")) <= 1
        and _level_rank(candidate.get("technical_difficulty")) <= 1
        and _level_rank(candidate.get("failure_penalty")) <= 1
        and not _is_ballistic_or_high_coordination_candidate(candidate)
        and not _is_sport_or_cardio_context_candidate(candidate)
    )


def _has_advanced_progression_ceiling(candidate: dict) -> bool:
    return _candidate_name_has_any(
        candidate,
        {
            "goblet squat",
            "고블릿 스쿼트",
            "assisted pull-up",
            "assisted pullup",
            "어시스트 풀업",
        },
    )


def _has_beginner_moderate_compound_load_caution(candidate: dict, movement_type: str) -> bool:
    if movement_type != "COMPOUND":
        return False
    return any(
        _level_rank(candidate.get(field)) >= 2
        for field in (
            "pain_risk_level",
            "injury_caution_level",
            "lumbar_load",
            "axial_load",
            "knee_shear_load",
        )
    )


def _goal_effect_field(goal: Optional[str]) -> Optional[str]:
    normalized = str(goal or "").strip().lower()
    if normalized in {"hypertrophy", "muscle_gain", "muscle", "근비대"}:
        return "hypertrophy_effect"
    if normalized in {"strength", "max_strength", "powerlifting", "근력"}:
        return "strength_effect"
    if normalized in {"power", "explosive", "파워"}:
        return "power_effect"
    return None


def _candidate_is_safe(
    candidate: dict,
    blocked_equipment: List[str],
    pain_areas: List[PainAreaEntry],
    available_equipment: Optional[List[str]] = None,
    shoulder_condition_status: Optional[str] = None,
    restricted_movement_patterns: Optional[List[str]] = None,
    injury_conditions: Optional[List[Any]] = None,
) -> bool:
    if select_equipment_option(
        candidate.get("equipment_req"), blocked_equipment, available_equipment
    ) is None:
        return False

    injury_tokens = _pain_tokens_from_entries(pain_areas)
    if _candidate_injury_overlap(candidate, injury_tokens):
        return False
    if candidate_injury_condition_overlaps(candidate, injury_conditions or []):
        return False
    return not candidate_restricted_movement_overlap(
        candidate,
        shoulder_condition_status,
        restricted_movement_patterns or [],
    )


def _breakdown_item(
    *,
    score: int,
    reason: str,
    rule_code: str,
    constraint_code: Optional[str] = None,
    profile_signal_code: Optional[str] = None,
) -> dict:
    item = {
        "score": score,
        "reason": reason,
        "rule_code": rule_code,
    }
    if constraint_code:
        item["constraint_code"] = constraint_code
    if profile_signal_code:
        item["profile_signal_code"] = profile_signal_code
    return item


def _record_score_delta(
    *,
    delta: int,
    internal_reason: str,
    public_reason: str,
    rule_code: str,
    score_reasons: List[str],
    display_reasons: List[str],
    boosts: List[dict],
    penalties: List[dict],
    constraint_code: Optional[str] = None,
    profile_signal_code: Optional[str] = None,
) -> int:
    score_reasons.append(internal_reason)
    if delta == 0:
        return 0

    display_reasons.append(public_reason)
    item = _breakdown_item(
        score=delta,
        reason=public_reason,
        rule_code=rule_code,
        constraint_code=constraint_code,
        profile_signal_code=profile_signal_code,
    )
    if delta > 0:
        boosts.append(item)
    else:
        penalties.append(item)
    return delta


def score_candidate_exercises(
    candidates: List[dict],
    target_muscles: List[str],
    doms_db: Optional[Dict[str, int]] = None,
    blocked_equipment: Optional[List[str]] = None,
    available_equipment: Optional[List[str]] = None,
    pain_areas: Optional[List[PainAreaEntry]] = None,
    recent_sets: Optional[List[RecentSetRecord]] = None,
    preferred_exercise_ids: Optional[List[str]] = None,
    unpreferred_exercise_ids: Optional[List[str]] = None,
    top_n: int = 12,
    feedback_adjustments: Optional[dict] = None,
    readiness_level: Optional[str] = None,
    goal: Optional[str] = None,
    mobility_limits: Optional[List[str]] = None,
    condition_policies: Optional[List[dict]] = None,
    anthropometry_signals: Optional[Dict[str, Any]] = None,
    experience_level: Optional[str] = None,
    advanced_chest_focus: Optional[str] = None,
    specialization_focus: Optional[str] = None,
    training_method: Optional[str] = None,
    training_style: Optional[str] = None,
    target_split_label: Optional[str] = None,
    session_structure: Optional[str] = None,
    adjunct_options: Optional[List[str]] = None,
    intensity_intent: Optional[str] = None,
    time_available_min: Optional[int] = None,
    shoulder_condition_status: Optional[str] = None,
    restricted_movement_patterns: Optional[List[str]] = None,
    injury_conditions: Optional[List[Any]] = None,
) -> List[dict]:
    """후보 운동을 deterministic하게 점수화해 상위 N개를 반환한다."""
    target_set = set(target_muscles)
    scored: List[dict] = []

    doms = doms_db or {}
    blocked_equipment = blocked_equipment or []
    available_equipment = available_equipment or []
    pain_areas = pain_areas or []
    preferred_set = {eid.strip().lower() for eid in (preferred_exercise_ids or [])}
    unpreferred_set = {eid.strip().lower() for eid in (unpreferred_exercise_ids or [])}
    mobility_limit_set = _normalize_constraint_tokens(mobility_limits)
    condition_policies = condition_policies or []
    anthropometry_signals = anthropometry_signals or {}
    anthropometry_signal_set = _normalize_constraint_tokens(
        anthropometry_signals.get("signals")
        if isinstance(anthropometry_signals.get("signals"), list)
        else []
    )
    readiness = str(readiness_level or "normal").strip().lower()
    resolved_experience_level = experience_level or "intermediate"
    is_beginner = _is_beginner_experience(resolved_experience_level)
    is_intermediate = _is_intermediate_experience(resolved_experience_level)
    is_advanced = _is_advanced_experience(resolved_experience_level)
    normalized_advanced_chest_focus = str(
        advanced_chest_focus or "balanced"
    ).strip()
    legacy_focus, legacy_method = legacy_specialization_values(
        normalized_advanced_chest_focus
    )
    normalized_specialization_focus = str(
        specialization_focus or legacy_focus or ""
    ).strip() or None
    normalized_training_method = str(
        training_method or legacy_method or ""
    ).strip() or None
    normalized_training_style = _normalize_training_style(training_style)
    normalized_adjunct_options = list(canonical_adjunct_options(adjunct_options or []))
    explicitly_requested_specialty_styles = set().union(*(
        _ADJUNCT_OPTION_ALLOWED_STYLES.get(option, set())
        for option in normalized_adjunct_options
    )) if normalized_adjunct_options else set()
    resolved_intensity_intent = intensity_intent or _infer_intensity_intent(goal, readiness_level, training_style)
    resolved_session_structure = session_structure or "standard"
    default_gym_style = _is_default_gym_style(training_style)
    gym_base_formula_style = _uses_gym_base_formula(training_style)
    split_accessory_boosts = _SPLIT_ACCESSORY_DIRECT_BOOSTS.get(
        str(target_split_label or "").strip().lower(),
        {},
    )
    split_primary_targets = _split_primary_targets(target_split_label)
    goal_effect_field = _goal_effect_field(goal)
    pain_body_parts = _pain_tokens_from_entries(pain_areas)
    knee_injury_present = bool(pain_body_parts & {"knees", "knee"})
    has_lumbar_constraint = (
        bool(pain_body_parts & {"lower-back", "back", "lumbar"})
        or _condition_policy_matches_lumbar(condition_policies)
    )
    clearance_state = infer_medical_clearance(condition_policies)
    clearance_required = clearance_state in {
        "not_cleared",
        "unknown",
        "cleared_with_restrictions",
    }
    resolved_injury_conditions = resolve_injury_condition_contexts(
        injury_conditions or (),
        (),
        legacy_shoulder_contexts=(
            (shoulder_condition_status, restricted_movement_patterns or ()),
        ),
    )
    effective_shoulder_condition = next(
        (
            condition
            for condition in resolved_injury_conditions
            if condition.area_code == "shoulder"
        ),
        None,
    )
    effective_shoulder_status = (
        effective_shoulder_condition.status
        if effective_shoulder_condition is not None
        else None
    )
    effective_shoulder_patterns = (
        effective_shoulder_condition.restricted_movement_patterns
        if effective_shoulder_condition is not None
        else ()
    )
    recent_names = {
        record.exercise_name.strip().lower()
        for record in (recent_sets or [])
        if record.exercise_name
    }
    from .pull_composition import complement_scope

    allow_row_complement = complement_scope(
        split=str(target_split_label or "").lower(), experience=str(experience_level or "").lower(),
        goal=str(goal or "").lower(), readiness=readiness, minutes=time_available_min,
        structure=resolved_session_structure, style=normalized_training_style,
        focus=normalized_specialization_focus, method=training_method,
        adjuncts=normalized_adjunct_options, pain=pain_areas, injuries=resolved_injury_conditions,
        doms=doms, constraints=bool(condition_policies or mobility_limit_set or restricted_movement_patterns),
    )
    from .beginner_machine_press import SCOPE as BEGINNER_PRESS_SCOPE, admission_scope

    allow_beginner_press = admission_scope(
        split=str(target_split_label or "").lower(), experience=str(experience_level or "").lower(),
        goal=str(goal or "").lower(), readiness=readiness,
        structure=resolved_session_structure, style=normalized_training_style,
        pain=pain_areas, injuries=resolved_injury_conditions, doms=doms,
        constraints=bool(condition_policies or mobility_limit_set or restricted_movement_patterns),
        adjuncts=normalized_adjunct_options, focus=normalized_specialization_focus,
        method=training_method,
    )

    allowed_purposes = admission_purposes(normalized_adjunct_options, normalized_training_style)
    for candidate in candidates:
        if not automatic_recommendation_eligibility(candidate, experience_level,
            adjunct_options=allowed_purposes).eligible:
            continue
        if not recommendation_taxonomy_is_eligible(candidate):
            continue
        if candidate.get("core_admission_scope") == BEGINNER_PRESS_SCOPE and not allow_beginner_press:
            continue
        if candidate.get("core_admission_scope") == "balancedPullRowComplement" and not allow_row_complement:
            continue
        if candidate.get("core_admission_scope") == "forearmEmphasis" and (
            normalized_specialization_focus != "forearmEmphasis" or is_beginner
        ):
            continue
        if candidate.get("core_admission_scope") == "lowerChest" and (
            normalized_specialization_focus != "lowerChest"
            or not is_advanced
            or readiness not in {"normal", "high"}
            or goal != "hypertrophy"
            or resolved_session_structure != "standard"
            or pain_areas or resolved_injury_conditions or condition_policies
            or mobility_limit_set
            or any(int(level or 0) > 0 for level in doms.values())
        ):
            # Bounded product admission, not clinical clearance or a global
            # promotion of previously held catalog variants.
            continue
        if (
            effective_shoulder_status == "managedWithRestrictions"
            and candidate_selection_family_key(candidate) == "front_raise"
        ):
            # This recommendation boundary is independent of the selected
            # movement restrictions and does not classify front raises as unsafe.
            continue
        if (
            is_beginner
            and _is_true(candidate.get("core_main_role_reviewed"))
            and _is_false(candidate.get("beginner_default_allowed"))
            and _is_false(
                candidate.get("catalog_original_main_routine_default_allowed")
            )
        ):
            # Identity-bound main-role reviews retain their declared experience
            # scope at admission; a preserved beginner denial is not a score hint.
            continue
        candidate_keys = {
            str(candidate.get("id") or "").strip().lower(),
            str(candidate.get("name_en") or "").strip().lower(),
        }
        if candidate_keys & unpreferred_set:
            continue

        exclusion_reasons: List[str] = []
        if not _candidate_is_safe(
            candidate,
            blocked_equipment,
            pain_areas,
            available_equipment,
            effective_shoulder_status,
            list(effective_shoulder_patterns),
            injury_conditions,
        ):
            if select_equipment_option(
                candidate.get("equipment_req"), blocked_equipment, available_equipment
            ) is None:
                exclusion_reasons.append("excluded unavailable equipment options")
            injury_overlap = sorted(_candidate_injury_overlap(candidate, pain_body_parts))
            if injury_overlap:
                exclusion_reasons.append(
                    "excluded registered injury overlap: " + ",".join(injury_overlap)
                )
            shoulder_overlap = sorted(candidate_restricted_movement_overlap(
                candidate,
                effective_shoulder_status,
                effective_shoulder_patterns,
            ))
            if shoulder_overlap:
                exclusion_reasons.append(
                    "excluded shoulder movement restriction: " + ",".join(shoulder_overlap)
                )
            managed_injury_overlap = candidate_injury_condition_overlaps(
                candidate,
                injury_conditions or [],
            )
            if managed_injury_overlap:
                details = ";".join(
                    f"{area}={','.join(sorted(patterns))}"
                    for area, patterns in sorted(managed_injury_overlap.items())
                )
                exclusion_reasons.append(
                    "excluded managed injury movement restriction: " + details
                )
            continue

        # Recompute from current constraints; cached selections are not authority.
        candidate = {
            **candidate,
            "selected_equipment_code": select_equipment_option(
                candidate.get("equipment_req"), blocked_equipment, available_equipment
            ),
        }
        if (
            "limited_ankle_dorsiflexion" in mobility_limit_set
            and _requires_explicit_deep_squat_position(candidate)
        ):
            # The closed user signal describes difficulty keeping the heels down
            # in a deep squat. Rows that explicitly require that position are
            # outside the selected movement range; ordinary squat patterns still
            # remain eligible and are ranked by ankle demand below.
            continue

        if (
            "limited_shoulder_flexion" in mobility_limit_set
            and _requires_overhead_position(candidate)
        ):
            # This is a movement-difficulty product boundary, not an injury claim.
            # Explicit overhead rows are outside the selected movement range even
            # when legacy posture metadata is stale.
            continue

        if "limited_shoulder_flexion" in mobility_limit_set:
            bench_angle = str(candidate.get("bench_angle_band") or "").strip().lower()
            shoulder_plane = str(
                candidate.get("shoulder_elevation_plane") or ""
            ).strip().lower()
            shoulder_demand = _level(
                candidate.get("shoulder_elevation_demand")
                if candidate.get("shoulder_elevation_demand") is not None
                else candidate.get("shoulder_flexion_demand")
            )
            if bench_angle.startswith("incline"):
                # This is a conservative product-scope rule for a selected
                # functional movement difficulty, not a diagnosis or treatment
                # claim. Flat/horizontal presses remain available.
                continue
            if shoulder_plane == "sagittal" and shoulder_demand in {"medium", "high"}:
                # Front-raise-like sagittal elevation duplicates the reported
                # difficult movement and is not needed for the base gym formula.
                continue

        raw_candidate_names = " ".join(
            str(candidate.get(key) or "").strip().lower()
            for key in (
                "name_kr",
                "name_en",
                "variation_name_kr",
                "catalog_name_kr",
                "catalog_name_en",
            )
        )
        candidate_bench_angle = str(
            candidate.get("bench_angle_band") or ""
        ).strip().lower()
        candidate_is_paused_press = (
            str(candidate.get("primary_muscle") or "").strip().lower() == "chest"
            and any(token in raw_candidate_names for token in ("정지", "paused", "pause"))
            and (
                candidate_bench_angle == "flat"
                or any(token in raw_candidate_names for token in ("플랫", "flat"))
            )
        )
        specialization_focus_match = (
            not is_beginner
            and candidate_matches_specialization_focus(
                candidate,
                normalized_specialization_focus,
            )
        )
        explicit_training_method_match = (
            is_advanced
            and candidate_matches_training_method(
                candidate,
                normalized_training_method,
            )
            and _level(candidate.get("failure_penalty")) in {"", "low", "medium"}
        )
        style_tags = _metadata_style_tags(candidate)
        candidate_specialty_styles = style_tags & _DEFAULT_GYM_EXPLICIT_ONLY_STYLES
        matched_adjunct_options = matching_adjunct_options(candidate, normalized_adjunct_options)
        requested_style_match = bool(
            normalized_training_style
            and not default_gym_style
            and normalized_training_style in style_tags
        )
        if (
            default_gym_style
            and candidate_specialty_styles
            and not candidate_specialty_styles & explicitly_requested_specialty_styles
            and not _is_explicitly_reviewed_mainstream_candidate(candidate)
        ):
            continue
        if (
            gym_base_formula_style
            and not requested_style_match
            and not matched_adjunct_options
            and _is_false(candidate.get("main_routine_default_allowed"))
        ):
            # The default gym formula must stay recognizable at every experience
            # level. Specialized rows are available only through an explicit
            # style/adjunct request, not as a score-based surprise.
            continue

        score = 0
        reasons: List[str] = list(candidate.get("score_reasons") or [])
        display_reasons: List[str] = []
        boosts: List[dict] = []
        penalties: List[dict] = []
        if "mapped substitute for unavailable equipment" in reasons:
            score += _record_score_delta(
                delta=12,
                internal_reason="mapped substitute for unavailable equipment",
                public_reason="사용 가능한 장비 조건에 맞춘 대체 후보입니다.",
                rule_code="mapped_substitute",
                constraint_code="equipment_alternative",
                score_reasons=[],
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        efficiency = int(candidate.get("efficiency_tier") or 7)
        efficiency_delta = max(0, 8 - efficiency) * 10
        score += _record_score_delta(
            delta=efficiency_delta,
            internal_reason=f"efficiency {efficiency}",
            public_reason=f"운동 효율 등급 {efficiency}이 추천 점수에 반영되었습니다.",
            rule_code="efficiency_tier",
            score_reasons=reasons,
            display_reasons=display_reasons,
            boosts=boosts,
            penalties=penalties,
        )

        movement_type = str(candidate.get("movement_type") or "").upper()
        if movement_type == "COMPOUND":
            score += _record_score_delta(
                delta=30,
                internal_reason="compound priority",
                public_reason="복합 운동이라 메인 운동 우선순위가 올라갔습니다.",
                rule_code="movement_type_compound",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )
        elif movement_type == "ISOLATION":
            score += _record_score_delta(
                delta=5,
                internal_reason="isolation support",
                public_reason="보조 볼륨을 채우는 고립 운동으로 반영되었습니다.",
                rule_code="movement_type_isolation",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        application_phase = _infer_candidate_application_phase(candidate, target_set, movement_type)
        approved_condition_actions = _matching_approved_condition_actions(
            candidate,
            condition_policies,
        )
        if any(
            str(action.get("policy") or "").strip().lower() == "avoid"
            for action in approved_condition_actions
        ):
            continue

        if any(
            str(action.get("policy") or "").strip().lower() == "caution"
            for action in approved_condition_actions
        ):
            score += _record_score_delta(
                delta=-18,
                internal_reason="approved condition policy caution",
                public_reason="검수 승인된 질환 정책에서 주의 대상으로 분류되어 우선순위를 낮췄습니다.",
                rule_code="approved_condition_policy_caution",
                constraint_code="condition_policy",
                profile_signal_code="condition_policy",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        option_policy = resolve_candidate_policy_for_request(
            candidate,
            goal=goal or "hypertrophy",
            experience_level=experience_level or "intermediate",
            readiness_level=readiness,
            time_available_min=time_available_min or 60,
            session_structure=resolved_session_structure,
            adjunct_options=adjunct_options,
            pain_areas=pain_areas,
            condition_policies=condition_policies,
            intensity_intent=resolved_intensity_intent,
            application_phase=application_phase,
        )
        if not option_policy.allowed:
            continue
        if option_policy.severity == "warn":
            score += _record_score_delta(
                delta=-8,
                internal_reason=f"option policy warning {option_policy.root_reason_code}",
                public_reason=option_policy.user_message_ko or "현재 조건을 반영해 보수적으로 우선순위를 낮췄습니다.",
                rule_code="option_policy_warning",
                constraint_code=option_policy.root_reason_code,
                profile_signal_code="option_policy",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        equipment_req = candidate.get("selected_equipment_code")
        if _is_loadable_equipment(equipment_req):
            score += _record_score_delta(
                delta=18,
                internal_reason="loadable equipment priority",
                public_reason="중량 조절이 가능한 장비 운동이라 우선순위가 올라갔습니다.",
                rule_code="loadable_equipment",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )
        elif _is_bodyweight_only(equipment_req):
            score += _record_score_delta(
                delta=-18,
                internal_reason="bodyweight deprioritized",
                public_reason="중량 증량이 제한적인 맨몸 운동이라 우선순위가 낮아졌습니다.",
                rule_code="bodyweight_deprioritized",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        if matched_adjunct_options:
            matched_adjunct = matched_adjunct_options[0]
            score += _record_score_delta(
                delta=60,
                internal_reason=f"requested adjunct option match {matched_adjunct}",
                public_reason="요청한 보조 목적과 운동 분류가 일치해 보조 슬롯 후보로 보존했습니다.",
                rule_code="adjunct_option_match_boost",
                constraint_code=matched_adjunct,
                profile_signal_code="adjunct_option",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )
        if gym_base_formula_style and _is_mainstream_gym_candidate(candidate):
            score += _record_score_delta(
                delta=14 if not is_beginner else 20,
                internal_reason="mainstream gym exercise priority",
                public_reason="헬스장에서 보편적으로 수행되는 운동이라 우선순위가 올라갔습니다.",
                rule_code="mainstream_gym_priority",
                profile_signal_code="training_style",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )
        if gym_base_formula_style and _is_common_default_gym_candidate(candidate):
            score += _record_score_delta(
                delta=12,
                internal_reason="common default gym familiarity priority",
                public_reason="동일 목적 후보 중 익숙한 표준 운동을 먼저 제안하도록 우선순위를 높였습니다.",
                rule_code="common_default_gym_familiarity_priority",
                profile_signal_code="training_style",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        if gym_base_formula_style and not requested_style_match and _is_uncommon_default_gym_candidate(candidate):
            score += _record_score_delta(
                delta=-18 if not is_beginner else -28,
                internal_reason="uncommon default gym variation penalty",
                public_reason="일반 헬스장 루틴에서는 생소하거나 고협응 변형 운동이라 우선순위를 낮췄습니다.",
                rule_code="uncommon_default_gym_variation_penalty",
                profile_signal_code="training_style",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        if gym_base_formula_style and not requested_style_match and _is_specialty_mobility_default_gym_candidate(candidate):
            score += _record_score_delta(
                delta=-55 if not is_beginner else -70,
                internal_reason="specialty mobility default gym suppression",
                public_reason="기본 헬스장 루틴에서는 요가/필라테스/소도구 정렬성 운동보다 표준 저항운동을 우선합니다.",
                rule_code="specialty_mobility_default_gym_suppression",
                profile_signal_code="training_style",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        if gym_base_formula_style and not requested_style_match and _is_false(candidate.get("main_routine_default_allowed")):
            score += _record_score_delta(
                delta=-12 if not is_beginner else -18,
                internal_reason="not main routine default allowed",
                public_reason="기본 루틴에서는 대표 운동보다 세부 변형/특수 운동이라 우선순위를 낮췄습니다.",
                rule_code="main_routine_default_allowed_penalty",
                profile_signal_code="training_style",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        if _has_generic_display_name(candidate):
            score += _record_score_delta(
                delta=-35,
                internal_reason="generic display name penalty",
                public_reason="사용자에게 운동 종류가 충분히 드러나지 않는 일반명이라 우선순위를 낮췄습니다.",
                rule_code="generic_display_name_penalty",
                constraint_code="display_name_quality",
                profile_signal_code="catalog_quality",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        if is_beginner and _is_false(candidate.get("beginner_default_allowed")):
            score += _record_score_delta(
                delta=-16,
                internal_reason="not beginner default allowed",
                public_reason="초보자 기본 루틴에서는 대표 운동으로 보기 어려워 우선순위를 낮췄습니다.",
                rule_code="beginner_default_allowed_penalty",
                profile_signal_code="experience_level",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        if gym_base_formula_style and normalized_training_style != "crossfit" and _is_crossfit_style_candidate(candidate):
            score += _record_score_delta(
                delta=-24,
                internal_reason="non gym style exercise penalty",
                public_reason="크로스핏/스포츠형 운동은 헬스장 기본 루틴 후보보다 낮게 배치했습니다.",
                rule_code="training_style_mismatch_penalty",
                profile_signal_code="training_style",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        if gym_base_formula_style and normalized_training_style != "rehab_prehab" and ({"rehab", "rehab_prehab"} & style_tags) and not condition_policies:
            score += _record_score_delta(
                delta=-22,
                internal_reason="rehab style default gym penalty",
                public_reason="재활/프리햅 성격의 운동은 검수된 질환·재활 목적이 없을 때 일반 루틴 후보보다 낮게 배치했습니다.",
                rule_code="rehab_style_default_gym_penalty",
                profile_signal_code="training_style",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )
        if gym_base_formula_style and "cardio" in style_tags and goal_effect_field in {"hypertrophy_effect", "strength_effect"}:
            score += _record_score_delta(
                delta=-28,
                internal_reason="cardio style strength hypertrophy penalty",
                public_reason="근비대/근력 루틴에서는 유산소성 후보를 메인 운동보다 낮게 배치했습니다.",
                rule_code="cardio_style_strength_hypertrophy_penalty",
                profile_signal_code="goal",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        injury_level = _level(candidate.get("injury_caution_level"))
        if injury_level == "high":
            if is_beginner:
                risk_delta = -42
            elif is_advanced and readiness == "high":
                risk_delta = -18
            else:
                risk_delta = -32
            score += _record_score_delta(
                delta=risk_delta,
                internal_reason="general high injury caution penalty",
                public_reason="부상 주의도가 높은 운동은 기본 추천 우선순위를 낮췄습니다.",
                rule_code="general_injury_caution_penalty",
                constraint_code="injury_caution_level",
                profile_signal_code="service_safety_policy",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        if requested_style_match:
            score += _record_score_delta(
                delta=8,
                internal_reason=f"{normalized_training_style} adjunct style match",
                public_reason="요청한 보조 운동 스타일과 맞는 후보라 대체/마무리 후보 우선순위가 올라갔습니다.",
                rule_code="training_style_match_boost",
                constraint_code=normalized_training_style,
                profile_signal_code="training_style",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        if goal_effect_field in {"hypertrophy_effect", "strength_effect"} and _is_sport_or_cardio_context_candidate(candidate):
            score += _record_score_delta(
                delta=-40,
                internal_reason="sport cardio context accessory penalty",
                public_reason="근비대/근력 루틴에서는 스포츠 보강·유산소성 후보를 메인 운동보다 낮게 배치했습니다.",
                rule_code="sport_cardio_context_penalty",
                constraint_code=goal_effect_field,
                profile_signal_code="goal",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        primary = str(candidate.get("primary_muscle") or "").strip()
        secondary = {
            part.strip()
            for part in str(candidate.get("secondary_muscle") or "").split(",")
            if part.strip()
        }
        if primary and primary in target_set:
            score += _record_score_delta(
                delta=20,
                internal_reason="primary target match",
                public_reason="오늘 목표 주동근과 직접 일치합니다.",
                rule_code="primary_target_match",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )
            if primary in ACCESSORY_MUSCLE_SLUGS and movement_type == "ISOLATION":
                score += _record_score_delta(
                    delta=50,
                    internal_reason="direct target accessory coverage boost",
                    public_reason="사용자가 직접 지정한 작은 목표 근육을 고립 운동으로 채우는 후보라 우선순위를 올렸습니다.",
                    rule_code="direct_target_accessory_coverage_boost",
                    constraint_code="target_muscles",
                    profile_signal_code="target_muscles",
                    score_reasons=reasons,
                    display_reasons=display_reasons,
                    boosts=boosts,
                    penalties=penalties,
                )
        if secondary & target_set:
            score += _record_score_delta(
                delta=8,
                internal_reason="secondary target match",
                public_reason="보조 자극 부위가 오늘 목표와 겹칩니다.",
                rule_code="secondary_target_match",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )
        if (
            split_primary_targets
            and primary
            and primary not in split_primary_targets
            and primary not in target_set
            and not (secondary & split_primary_targets)
            and not (secondary & target_set)
        ):
            score += _record_score_delta(
                delta=-55,
                internal_reason=f"{target_split_label} split non target primary penalty",
                public_reason="선택한 분할 루틴의 목표 부위와 직접 관련이 낮아 후보 우선순위를 크게 낮췄습니다.",
                rule_code="split_non_target_primary_penalty",
                constraint_code=str(target_split_label or "").strip().lower(),
                profile_signal_code="target_split_label",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )
        if primary in split_accessory_boosts and movement_type == "ISOLATION":
            score += _record_score_delta(
                delta=split_accessory_boosts[primary],
                internal_reason=f"{target_split_label} direct accessory coverage boost",
                public_reason="분할 루틴의 보조근을 직접 채우는 고립 운동이라 후보에 포함되도록 우선순위를 올렸습니다.",
                rule_code="split_accessory_direct_coverage_boost",
                constraint_code=str(target_split_label or "").strip().lower(),
                profile_signal_code="target_split_label",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )
        if (
            str(target_split_label or "").strip().lower() == "pull"
            and primary == "biceps"
            and movement_type != "ISOLATION"
            and _is_biceps_labeled_pulldown_candidate(candidate)
        ):
            score += _record_score_delta(
                delta=-32,
                internal_reason="pull biceps accessory pulldown ambiguity penalty",
                public_reason="이두 보조 운동으로 보기 어려운 풀다운형 후보라 직접 컬 계열보다 낮게 배치했습니다.",
                rule_code="pull_biceps_pulldown_ambiguity_penalty",
                constraint_code="direct_biceps_accessory",
                profile_signal_code="target_split_label",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )
        if (
            str(target_split_label or "").strip().lower() in {"legs", "lower"}
            and _is_tibialis_accessory_candidate(candidate)
        ):
            tibialis_delta = -55 if is_beginner else -30
            tibialis_reason = (
                "beginner legs tibialis accessory-only penalty"
                if is_beginner
                else "legs tibialis accessory-only penalty"
            )
            tibialis_rule_code = (
                "beginner_legs_tibialis_accessory_penalty"
                if is_beginner
                else "legs_tibialis_accessory_penalty"
            )
            score += _record_score_delta(
                delta=tibialis_delta,
                internal_reason=tibialis_reason,
                public_reason="하체 기본 루틴에서는 정강이 보조 운동을 메인 운동보다 낮게 배치했습니다.",
                rule_code=tibialis_rule_code,
                constraint_code="accessory_only",
                profile_signal_code="experience_level",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )
        if primary in LARGE_MUSCLE_SLUGS:
            score += _record_score_delta(
                delta=10,
                internal_reason="large muscle priority",
                public_reason="대근육 주동근 운동이라 메인 볼륨으로 우선 반영되었습니다.",
                rule_code="large_muscle_priority",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )
        elif primary in ACCESSORY_MUSCLE_SLUGS:
            score += _record_score_delta(
                delta=-4,
                internal_reason="accessory volume guard",
                public_reason="소근육 보조 운동이라 메인 볼륨보다 낮게 배치되었습니다.",
                rule_code="accessory_volume_guard",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        doms_overlap = _candidate_doms_overlap(candidate, doms)
        doms_level = max(doms_overlap.values(), default=0)
        if doms_level >= 3:
            continue
        if doms_level == 2:
            score += _record_score_delta(
                delta=-25,
                internal_reason="doms moderate penalty",
                public_reason="해당 주동근의 근육통이 강해 볼륨 우선순위가 낮아졌습니다.",
                rule_code="doms_penalty",
                profile_signal_code="doms",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )
        elif doms_level == 1:
            score += _record_score_delta(
                delta=-10,
                internal_reason="doms mild penalty",
                public_reason="해당 주동근의 가벼운 근육통이 반영되어 우선순위가 조금 낮아졌습니다.",
                rule_code="doms_penalty",
                profile_signal_code="doms",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        if has_lumbar_constraint:
            score += _apply_level_penalty(
                candidate=candidate,
                field="lumbar_load",
                high_delta=-35,
                medium_delta=-15,
                high_reason="허리 제약 조건에 비해 요추 부하가 큰 운동이라 우선순위가 크게 낮아졌습니다.",
                medium_reason="허리 제약 조건에 따라 요추 부하가 중간 수준인 운동의 추천 우선순위를 낮췄습니다.",
                rule_code="lumbar_load_constraint_penalty",
                constraint_code="lumbar_load",
                profile_signal_code="registered_injury_or_condition",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )
            score += _apply_level_penalty(
                candidate=candidate,
                field="axial_load",
                high_delta=-20,
                medium_delta=-8,
                high_reason="척추 수직 압박 부담이 큰 운동이라 허리 제약 조건에서 우선순위가 낮아졌습니다.",
                medium_reason="척추 수직 압박 부담이 중간 수준이라 허리 제약 조건에서 감점되었습니다.",
                rule_code="axial_load_constraint_penalty",
                constraint_code="axial_load",
                profile_signal_code="registered_injury_or_condition",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        if pain_body_parts & {"knees", "knee"}:
            score += _apply_level_penalty(
                candidate=candidate,
                field="knee_shear_load",
                high_delta=-55 if knee_injury_present else -30,
                medium_delta=-35 if knee_injury_present else -12,
                high_reason="등록된 무릎 부상과 충돌할 수 있어 무릎 전단 부하가 큰 운동의 우선순위를 크게 낮췄습니다.",
                medium_reason="등록된 무릎 부상과 충돌할 수 있어 무릎 전단 부하가 중간인 운동도 추천 우선순위를 크게 낮췄습니다.",
                rule_code="knee_load_constraint_penalty",
                constraint_code="knee_shear_load",
                profile_signal_code="registered_injury",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        if pain_body_parts & {"shoulder", "front-deltoids", "back-deltoids"}:
            score += _apply_level_penalty(
                candidate=candidate,
                field="shoulder_impingement_risk",
                high_delta=-30,
                medium_delta=-12,
                high_reason="등록된 어깨 부상과 충돌할 수 있어 어깨 충돌 위험 지표가 높은 운동의 우선순위를 낮췄습니다.",
                medium_reason="등록된 어깨 부상과 충돌할 수 있어 어깨 부담이 중간인 운동의 추천 우선순위를 낮췄습니다.",
                rule_code="shoulder_impingement_constraint_penalty",
                constraint_code="shoulder_impingement_risk",
                profile_signal_code="registered_injury",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        if pain_body_parts & {"wrist", "forearm"}:
            score += _apply_level_penalty(
                candidate=candidate,
                field="wrist_stress",
                high_delta=-25,
                medium_delta=-10,
                high_reason="등록된 손목/전완 부상과 충돌할 수 있어 손목 부담이 큰 운동의 우선순위를 낮췄습니다.",
                medium_reason="등록된 손목/전완 부상과 충돌할 수 있어 손목 부담이 중간인 운동의 추천 우선순위를 낮췄습니다.",
                rule_code="wrist_stress_constraint_penalty",
                constraint_code="wrist_stress",
                profile_signal_code="registered_injury",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        if pain_body_parts & {"ankle", "calves", "soleus", "left-soleus", "right-soleus"}:
            score += _apply_level_penalty(
                candidate=candidate,
                field="ankle_dorsiflexion_demand",
                high_delta=-24,
                medium_delta=-10,
                high_reason="등록된 발목/가자미근 부상과 충돌할 수 있어 발목 요구도가 높은 운동의 우선순위를 낮췄습니다.",
                medium_reason="등록된 발목/가자미근 부상과 충돌할 수 있어 발목 요구도가 중간인 운동의 추천 우선순위를 낮췄습니다.",
                rule_code="ankle_soleus_pain_demand_penalty",
                constraint_code="ankle_dorsiflexion_demand",
                profile_signal_code="registered_injury",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        # A registered injury is a binary, body-part-scoped exclusion. It must not
        # globally perturb otherwise unrelated candidates. Reviewed condition
        # policies may still apply broader caution because they encode that scope.
        if condition_policies:
            condition_context_label = "condition_policy"
            injury_level = _level(candidate.get("injury_caution_level"))
            if injury_level == "high":
                score += _record_score_delta(
                    delta=-14,
                    internal_reason=f"condition context {condition_context_label} high injury caution penalty",
                    public_reason="검수된 질환·수술 주의 맥락에서 일반 부상 주의도가 높은 운동은 우선순위를 낮췄습니다.",
                    rule_code="condition_context_injury_caution_penalty",
                    constraint_code="injury_caution_level",
                    profile_signal_code="condition_policy",
                    score_reasons=reasons,
                    display_reasons=display_reasons,
                    boosts=boosts,
                    penalties=penalties,
                )
            elif injury_level == "medium":
                score += _record_score_delta(
                    delta=-5,
                    internal_reason=f"condition context {condition_context_label} medium injury caution penalty",
                    public_reason="검수된 질환·수술 주의 맥락에서 일반 부상 주의도가 중간인 운동의 추천 우선순위를 낮췄습니다.",
                    rule_code="condition_context_injury_caution_penalty",
                    constraint_code="injury_caution_level",
                    profile_signal_code="condition_policy",
                    score_reasons=reasons,
                    display_reasons=display_reasons,
                    boosts=boosts,
                    penalties=penalties,
                )

        if condition_policies and _is_ballistic_or_high_coordination_candidate(candidate):
            score += _record_score_delta(
                delta=-14,
                internal_reason="condition context ballistic coordination penalty",
                public_reason="검수된 질환·수술 주의 맥락에서는 탄성·착지·고협응 보강 운동을 메인 후보보다 낮게 배치했습니다.",
                rule_code="condition_context_ballistic_coordination_penalty",
                constraint_code="movement_context",
                profile_signal_code="condition_policy",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        if (
            has_lumbar_constraint
            and primary in target_set
            and _level(candidate.get("lumbar_load")) == "low"
            and _level(candidate.get("axial_load")) == "low"
            and _level(candidate.get("injury_caution_level")) == "low"
            and (movement_type == "ISOLATION" or "MACHINE" in _normalize_tokens(candidate.get("equipment_req")))
            and not _is_sport_or_cardio_context_candidate(candidate)
            and not _is_ballistic_or_high_coordination_candidate(candidate)
        ):
            score += _record_score_delta(
                delta=18,
                internal_reason="lumbar safe low load alternative boost",
                public_reason="허리 제약 조건에서 요추/축방향 부하와 일반 부상 주의도가 낮은 안전 대체 후보입니다.",
                rule_code="lumbar_safe_low_load_alternative_boost",
                constraint_code="lumbar_safe_alternative",
                profile_signal_code="registered_injury_or_condition",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        shoulder_demand_field = (
            "shoulder_elevation_demand"
            if candidate.get("shoulder_elevation_demand") is not None
            else "shoulder_flexion_demand"
        )
        mobility_penalties = [
            (
                "limited_ankle_dorsiflexion",
                "ankle_dorsiflexion_demand",
                "발뒤꿈치를 붙인 깊은 앉기 동작이 어렵다고 입력되어 발목 요구도가 높은 운동의 우선순위를 낮췄습니다.",
                "발뒤꿈치를 붙인 깊은 앉기 동작이 어렵다고 입력되어 발목 요구도가 중간인 운동의 추천 우선순위를 낮췄습니다.",
                "limited_ankle_dorsiflexion",
            ),
            (
                "limited_hip_flexion",
                "hip_flexion_demand",
                "엉덩이를 접어 숙이는 동작이 어렵다고 입력되어 고관절 요구도가 높은 운동의 우선순위를 낮췄습니다.",
                "엉덩이를 접어 숙이는 동작이 어렵다고 입력되어 고관절 요구도가 중간인 운동의 추천 우선순위를 낮췄습니다.",
                "limited_hip_flexion",
            ),
            (
                "limited_shoulder_flexion",
                shoulder_demand_field,
                "팔을 머리 위로 올리는 동작이 어렵다고 입력되어 어깨 요구도가 높은 운동의 우선순위를 낮췄습니다.",
                "팔을 머리 위로 올리는 동작이 어렵다고 입력되어 어깨 요구도가 중간인 운동의 추천 우선순위를 낮췄습니다.",
                "limited_shoulder_flexion",
            ),
            (
                "limited_wrist_extension",
                "wrist_extension_demand",
                "손목을 뒤로 젖혀 부하를 버티는 동작이 어렵다고 입력되어 손목 요구도가 높은 운동의 우선순위를 낮췄습니다.",
                "손목을 뒤로 젖혀 부하를 버티는 동작이 어렵다고 입력되어 손목 요구도가 중간인 운동의 추천 우선순위를 낮췄습니다.",
                "limited_wrist_extension",
            ),
        ]
        for limit_code, field, high_reason, medium_reason, constraint_code in mobility_penalties:
            if limit_code not in mobility_limit_set:
                continue
            score += _apply_level_penalty(
                candidate=candidate,
                field=field,
                high_delta=-20,
                medium_delta=-8,
                high_reason=high_reason,
                medium_reason=medium_reason,
                rule_code="mobility_constraint_penalty",
                constraint_code=constraint_code,
                profile_signal_code="mobility_limit",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )
            if (
                limit_code == "limited_shoulder_flexion"
                and _level_rank(candidate.get(field)) <= 1
                and (
                    primary in {"chest", "front-deltoids", "side-deltoids", "triceps"}
                    or bool(
                        _normalize_tokens(candidate.get("secondary_muscle"))
                        & {"CHEST", "FRONT-DELTOIDS", "SIDE-DELTOIDS", "TRICEPS"}
                    )
                    or _level_rank(candidate.get("shoulder_impingement_risk")) >= 2
                )
            ):
                score += _record_score_delta(
                    delta=0,
                    internal_reason="limited_shoulder_flexion mobility reviewed low demand",
                    public_reason="팔을 머리 위로 올리는 동작의 어려움을 검토했으며, 이 후보는 어깨 요구도가 낮거나 직접 제한 신호가 크지 않습니다.",
                    rule_code="mobility_constraint_review",
                    constraint_code="limited_shoulder_flexion",
                    profile_signal_code="mobility_limit",
                    score_reasons=reasons,
                    display_reasons=display_reasons,
                    boosts=boosts,
                    penalties=penalties,
                )
            if (
                limit_code == "limited_shoulder_flexion"
                and _level_rank(candidate.get(field)) <= 1
                and str(candidate.get("path_type") or "").strip() in {"free", "semiFree"}
                and str(candidate.get("bilateral_coupling") or "").strip()
                in {"independent", "independentCapable"}
                and str(candidate.get("grip_freedom") or "").strip() == "high"
            ):
                score += _record_score_delta(
                    delta=4,
                    internal_reason="overhead reach difficulty independent setup preference",
                    public_reason="양팔을 독립적으로 움직이고 그립을 조절하기 쉬운 저요구도 세팅 후보라 약하게 우선했습니다.",
                    rule_code="mobility_setup_preference",
                    constraint_code="limited_shoulder_flexion",
                    profile_signal_code="mobility_limit",
                    score_reasons=reasons,
                    display_reasons=display_reasons,
                    boosts=boosts,
                    penalties=penalties,
                )
            if (
                limit_code == "limited_wrist_extension"
                and _level_rank(candidate.get(field)) <= 1
                and _level_rank(candidate.get("wrist_stress")) <= 1
                and (
                    primary
                    in {
                        "chest",
                        "front-deltoids",
                        "side-deltoids",
                        "back-deltoids",
                        "triceps",
                        "biceps",
                        "forearm",
                        "upper-back",
                        "trapezius",
                    }
                    or bool(
                        _normalize_tokens(candidate.get("secondary_muscle"))
                        & {
                            "CHEST",
                            "FRONT-DELTOIDS",
                            "SIDE-DELTOIDS",
                            "BACK-DELTOIDS",
                            "TRICEPS",
                            "BICEPS",
                            "FOREARM",
                            "UPPER-BACK",
                            "TRAPEZIUS",
                        }
                    )
                )
            ):
                score += _record_score_delta(
                    delta=0,
                    internal_reason="limited_wrist_extension mobility reviewed low demand",
                    public_reason="손목을 뒤로 젖혀 부하를 버티는 동작의 어려움을 검토했으며, 이 후보는 손목 요구도와 손목 스트레스가 낮습니다.",
                    rule_code="mobility_constraint_review",
                    constraint_code="limited_wrist_extension",
                    profile_signal_code="mobility_limit",
                    score_reasons=reasons,
                    display_reasons=display_reasons,
                    boosts=boosts,
                    penalties=penalties,
                )

        if clearance_required and (
            _level_rank(candidate.get("lumbar_load")) >= 3
            or _level_rank(candidate.get("axial_load")) >= 3
            or _level_rank(candidate.get("failure_penalty")) >= 3
            or _level_rank(candidate.get("shoulder_impingement_risk")) >= 3
            or _level_rank(candidate.get("injury_caution_level")) >= 3
        ):
            score += _record_score_delta(
                delta=-25,
                internal_reason="professional clearance guard",
                public_reason="전문가 허가가 확인되지 않은 재활/질환 조건이라 고부하 후보의 추천 우선순위를 낮췄습니다.",
                rule_code="professional_clearance_guard",
                constraint_code="professional_clearance",
                profile_signal_code="condition_policy",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        if clearance_required and _is_low_risk_clearance_alternative(candidate):
            score += _record_score_delta(
                delta=12,
                internal_reason="professional clearance safe alternative candidate",
                public_reason="전문가 허가가 필요한 조건에서 낮은 부하와 낮은 주의도의 안전 대체 후보입니다.",
                rule_code="professional_clearance_safe_alternative",
                constraint_code="professional_clearance",
                profile_signal_code="condition_policy",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        experience_equipment = _normalize_tokens(
            candidate.get("selected_equipment_code")
            or candidate.get("equipment_req")
        )
        is_smith_candidate = "SMITH_MACHINE" in experience_equipment
        is_generic_machine_candidate = (
            "MACHINE" in experience_equipment and not is_smith_candidate
        )
        is_free_weight_candidate = bool(
            experience_equipment
            & {"BARBELL", "DUMBBELL", "KETTLEBELL", "LANDMINE"}
        )
        experience_equipment_preference_allowed = (
            default_gym_style
            and readiness in {"normal", "high", ""}
        )

        if is_beginner:
            if (
                experience_equipment_preference_allowed
                and movement_type == "COMPOUND"
                and (is_generic_machine_candidate or is_smith_candidate)
            ):
                score += _record_score_delta(
                    delta=6 if is_generic_machine_candidate else 3,
                    internal_reason="beginner stable equipment preference",
                    public_reason="초보자 조건에서 동작 경로와 세팅이 비교적 안정적인 장비 운동을 우선했습니다.",
                    rule_code="beginner_stable_equipment_preference",
                    constraint_code="experience_level",
                    profile_signal_code="experience_level",
                    score_reasons=reasons,
                    display_reasons=display_reasons,
                    boosts=boosts,
                    penalties=penalties,
                )
            if _has_beginner_moderate_compound_load_caution(candidate, movement_type):
                score += _record_score_delta(
                    delta=-12,
                    internal_reason="beginner moderate compound load caution",
                    public_reason="초보자 조건에서는 통증/관절/요추 부하가 중간 이상인 복합 운동보다 더 안정적인 후보를 우선합니다.",
                    rule_code="beginner_moderate_compound_load_caution",
                    constraint_code="experience_level",
                    profile_signal_code="experience_level",
                    score_reasons=reasons,
                    display_reasons=display_reasons,
                    boosts=boosts,
                    penalties=penalties,
                )
            if _level_rank(candidate.get("failure_penalty")) >= 3:
                score += _record_score_delta(
                    delta=-14,
                    internal_reason="beginner high failure penalty",
                    public_reason="초보자 조건에서 실패 시 부담이 큰 운동이라 우선순위를 낮췄습니다.",
                    rule_code="beginner_failure_penalty",
                    constraint_code="failure_penalty",
                    profile_signal_code="experience_level",
                    score_reasons=reasons,
                    display_reasons=display_reasons,
                    boosts=boosts,
                    penalties=penalties,
                )
            if _level_rank(candidate.get("technical_difficulty")) >= 3:
                score += _record_score_delta(
                    delta=-12,
                    internal_reason="beginner high technical difficulty penalty",
                    public_reason="초보자 조건에서 기술 난이도가 높은 운동이라 우선순위를 낮췄습니다.",
                    rule_code="beginner_technical_difficulty_penalty",
                    constraint_code="technical_difficulty",
                    profile_signal_code="experience_level",
                    score_reasons=reasons,
                    display_reasons=display_reasons,
                    boosts=boosts,
                    penalties=penalties,
                )
            if (
                _level_rank(candidate.get("technical_difficulty")) <= 1
                and _level_rank(candidate.get("failure_penalty")) <= 1
                and _is_supported_or_stable_candidate(candidate)
                and not _is_ballistic_or_high_coordination_candidate(candidate)
            ):
                score += _record_score_delta(
                    delta=10,
                    internal_reason="beginner low technique risk candidate",
                    public_reason="초보자 조건에서 기술 난이도와 실패 부담이 낮은 안정 후보입니다.",
                    rule_code="beginner_low_technique_risk_boost",
                    constraint_code="technical_difficulty",
                    profile_signal_code="experience_level",
                    score_reasons=reasons,
                    display_reasons=display_reasons,
                    boosts=boosts,
                    penalties=penalties,
                )
            if _is_beginner_low_load_baseline_candidate(candidate):
                score += _record_score_delta(
                    delta=12,
                    internal_reason="beginner low load baseline candidate",
                    public_reason="초보자 조건에서 통증 위험과 관절/요추 부하가 낮은 안정 후보입니다.",
                    rule_code="beginner_low_load_baseline_boost",
                    constraint_code="experience_level",
                    profile_signal_code="experience_level",
                    score_reasons=reasons,
                    display_reasons=display_reasons,
                    boosts=boosts,
                    penalties=penalties,
                )
            if (
                (_has_progression_option(candidate) or _is_supported_or_stable_candidate(candidate))
                and _level_rank(candidate.get("failure_penalty")) <= 1
                and not _is_ballistic_or_high_coordination_candidate(candidate)
            ):
                score += _record_score_delta(
                    delta=4,
                    internal_reason="beginner progression path candidate",
                    public_reason="초보자 조건에서 단계적으로 난이도를 올리기 쉬운 progression 후보입니다.",
                    rule_code="beginner_progression_path_boost",
                    constraint_code="progression_path",
                    profile_signal_code="experience_level",
                    score_reasons=reasons,
                    display_reasons=display_reasons,
                    boosts=boosts,
                    penalties=penalties,
                )

        if is_intermediate:
            if (
                experience_equipment_preference_allowed
                and movement_type == "COMPOUND"
                and is_free_weight_candidate
            ):
                free_weight_delta = (
                    12
                    if "DUMBBELL" in experience_equipment
                    else 10
                    if "BARBELL" in experience_equipment
                    else 6
                )
                score += _record_score_delta(
                    delta=free_weight_delta,
                    internal_reason="intermediate free weight progression preference",
                    public_reason="중급자 조건에서 보편적이고 점진 과부하를 적용하기 쉬운 프리웨이트 복합 운동을 우선했습니다.",
                    rule_code="intermediate_free_weight_progression_preference",
                    constraint_code="experience_level",
                    profile_signal_code="experience_level",
                    score_reasons=reasons,
                    display_reasons=display_reasons,
                    boosts=boosts,
                    penalties=penalties,
                )
            if (
                experience_equipment_preference_allowed
                and goal_effect_field == "strength_effect"
                and movement_type == "COMPOUND"
                and "BARBELL" in experience_equipment
                and _is_explicitly_reviewed_mainstream_candidate(candidate)
            ):
                score += _record_score_delta(
                    delta=44,
                    internal_reason="intermediate strength barbell progression preference",
                    public_reason="중급자 근력 목표에서 점진 과부하를 적용하기 쉬운 검수된 바벨 복합 운동을 우선했습니다.",
                    rule_code="intermediate_strength_barbell_progression_preference",
                    constraint_code="experience_level",
                    profile_signal_code="experience_level",
                    score_reasons=reasons,
                    display_reasons=display_reasons,
                    boosts=boosts,
                    penalties=penalties,
                )
            if (
                experience_equipment_preference_allowed
                and goal_effect_field == "strength_effect"
                and movement_type == "COMPOUND"
                and _has_advanced_progression_ceiling(candidate)
            ):
                score += _record_score_delta(
                    delta=-14,
                    internal_reason="intermediate strength progression ceiling penalty",
                    public_reason="중급자 근력 메인 운동으로는 중량 확장 한계가 빨리 오는 후보라 보조 또는 대체 운동으로 낮췄습니다.",
                    rule_code="intermediate_strength_progression_ceiling_penalty",
                    constraint_code="experience_level",
                    profile_signal_code="experience_level",
                    score_reasons=reasons,
                    display_reasons=display_reasons,
                    boosts=boosts,
                    penalties=penalties,
                )
            if (
                default_gym_style
                and _is_mainstream_gym_candidate(candidate)
                and _level_rank(candidate.get("technical_difficulty")) <= 2
                and _level_rank(candidate.get("failure_penalty")) <= 2
            ):
                score += _record_score_delta(
                    delta=12 if movement_type == "COMPOUND" else 8,
                    internal_reason="intermediate mainstream progression candidate",
                    public_reason="중급자 조건에서 보편적이고 점진 과부하를 적용하기 쉬운 운동이라 우선순위를 올렸습니다.",
                    rule_code="intermediate_mainstream_progression_boost",
                    constraint_code="experience_level",
                    profile_signal_code="experience_level",
                    score_reasons=reasons,
                    display_reasons=display_reasons,
                    boosts=boosts,
                    penalties=penalties,
                )
            if (
                _level_rank(candidate.get("technical_difficulty")) >= 3
                and (
                    not _is_mainstream_gym_candidate(candidate)
                    or str(candidate.get("exercise_familiarity_tier") or "").strip().lower()
                    in {"specialized", "obscure"}
                    or str(candidate.get("novelty_level") or "").strip().lower()
                    in {"medium", "high"}
                )
            ):
                score += _record_score_delta(
                    delta=-10,
                    internal_reason="intermediate unfamiliar high technique penalty",
                    public_reason="중급자 기본 루틴에서는 생소한 고기술 운동이라 우선순위를 낮췄습니다.",
                    rule_code="intermediate_unfamiliar_high_technique_penalty",
                    constraint_code="technical_difficulty",
                    profile_signal_code="experience_level",
                    score_reasons=reasons,
                    display_reasons=display_reasons,
                    boosts=boosts,
                    penalties=penalties,
                )

        if specialization_focus_match:
            focus_label = focus_label_ko(normalized_specialization_focus)
            score += _record_score_delta(
                delta=24,
                internal_reason="specialization focus match",
                public_reason=(
                    f"사용자가 선택한 {focus_label} 집중 목표에 직접 맞는 후보입니다."
                ),
                rule_code="specialization_focus_match",
                constraint_code=normalized_specialization_focus,
                profile_signal_code="specialization_focus",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        if (
            is_advanced
            and str(target_split_label or "").strip().lower() == "arm"
            and primary == "triceps"
            and any(
                token in _compact_candidate_names(candidate)
                for token in ("킥백", "kickback")
            )
        ):
            score += _record_score_delta(
                delta=-18,
                internal_reason="advanced arm kickback accessory preference",
                public_reason="상급자 팔 루틴의 메인 후보보다 킥백 보조 후보의 우선순위를 낮췄습니다.",
                rule_code="advanced_kickback_accessory_penalty",
                constraint_code="advanced_arm_overload_order",
                profile_signal_code="experience_level",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        if explicit_training_method_match:
            method_instruction = training_method_instruction(
                normalized_training_method
            ) or {}
            score += _record_score_delta(
                delta=48,
                internal_reason="advanced training method match",
                public_reason=(
                    "상급자 세션에서 사용자가 선택한 "
                    f"{method_instruction.get('labelKo') or '훈련 방법'}을 적용할 수 있는 "
                    "검증 후보입니다."
                ),
                rule_code="training_method_match",
                constraint_code=normalized_training_method,
                profile_signal_code="training_method",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        if (
            is_advanced
            and readiness in {"normal", "high", ""}
            and not condition_policies
            and not mobility_limit_set
        ):
            if movement_type == "COMPOUND" and is_free_weight_candidate:
                free_weight_delta = (
                    20
                    if "BARBELL" in experience_equipment
                    else 16
                    if "DUMBBELL" in experience_equipment
                    else 10
                )
                score += _record_score_delta(
                    delta=free_weight_delta,
                    internal_reason="advanced free weight compound preference",
                    public_reason="상급자의 중량 조절과 점진적 과부하 목표에 맞춰 프리웨이트 복합 운동의 우선순위를 높였습니다.",
                    rule_code="advanced_free_weight_compound_preference",
                    constraint_code="experience_level",
                    profile_signal_code="experience_level",
                    score_reasons=reasons,
                    display_reasons=display_reasons,
                    boosts=boosts,
                    penalties=penalties,
                )
            if (
                not specialization_focus_match
                and not explicit_training_method_match
                and movement_type == "COMPOUND"
                and (
                is_generic_machine_candidate or is_smith_candidate
                )
            ):
                score += _record_score_delta(
                    delta=-6 if is_smith_candidate else -4,
                    internal_reason="advanced guided equipment alternative penalty",
                    public_reason="상급자 기본 루틴의 장비 선호 기준에 따라 고정 경로 장비를 프리웨이트 메인 운동의 대안으로 배치했습니다.",
                    rule_code="advanced_guided_equipment_alternative_penalty",
                    constraint_code="experience_level",
                    profile_signal_code="experience_level",
                    score_reasons=reasons,
                    display_reasons=display_reasons,
                    boosts=boosts,
                    penalties=penalties,
                )
            if (
                experience_equipment_preference_allowed
                and movement_type == "COMPOUND"
                and _has_advanced_progression_ceiling(candidate)
            ):
                score += _record_score_delta(
                    delta=-20,
                    internal_reason="advanced progression ceiling penalty",
                    public_reason="상급자 메인 운동으로는 중량 확장 한계가 빨리 오는 후보라 보조 또는 대체 운동으로 낮췄습니다.",
                    rule_code="advanced_progression_ceiling_penalty",
                    constraint_code="experience_level",
                    profile_signal_code="experience_level",
                    score_reasons=reasons,
                    display_reasons=display_reasons,
                    boosts=boosts,
                    penalties=penalties,
                )
            if (
                experience_equipment_preference_allowed
                and goal_effect_field == "strength_effect"
                and movement_type == "COMPOUND"
                and "BARBELL" in experience_equipment
                and _is_explicitly_reviewed_mainstream_candidate(candidate)
            ):
                score += _record_score_delta(
                    delta=36,
                    internal_reason="advanced strength barbell progression preference",
                    public_reason="상급자 근력 목표에서 높은 중량 확장성을 가진 검수된 바벨 복합 운동을 우선했습니다.",
                    rule_code="advanced_strength_barbell_progression_preference",
                    constraint_code="experience_level",
                    profile_signal_code="experience_level",
                    score_reasons=reasons,
                    display_reasons=display_reasons,
                    boosts=boosts,
                    penalties=penalties,
                )
            if (
                goal_effect_field
                and _level(candidate.get(goal_effect_field)) in {"high", "elite"}
                and _level(candidate.get("stimulus_to_fatigue")) in {"good", "excellent", ""}
                and _level_rank(candidate.get("injury_caution_level")) <= 2
                and _level_rank(candidate.get("failure_penalty")) <= 2
            ):
                score += _record_score_delta(
                    delta=8 if _level(candidate.get(goal_effect_field)) == "high" else 12,
                    internal_reason="advanced high effect exercise boost",
                    public_reason="상급자 조건에서 목표 효과가 높은 운동이라 우선순위를 조금 올렸습니다.",
                    rule_code="advanced_high_effect_boost",
                    constraint_code=goal_effect_field,
                    profile_signal_code="experience_level",
                    score_reasons=reasons,
                    display_reasons=display_reasons,
                    boosts=boosts,
                    penalties=penalties,
                )
            if (
                movement_type == "COMPOUND"
                and is_free_weight_candidate
                and _level_rank(candidate.get("technical_difficulty")) >= 2
                and _level_rank(candidate.get("failure_penalty")) <= 2
                and _level_rank(candidate.get("injury_caution_level")) <= 2
                and not _is_ballistic_or_high_coordination_candidate(candidate)
            ):
                score += _record_score_delta(
                    delta=8,
                    internal_reason="advanced compound variation capacity boost",
                    public_reason="상급자 조건에서 기술적으로 수행 가능한 복합 운동 변형이라 우선순위를 조금 올렸습니다.",
                    rule_code="advanced_compound_variation_boost",
                    constraint_code="experience_level",
                    profile_signal_code="experience_level",
                    score_reasons=reasons,
                    display_reasons=display_reasons,
                    boosts=boosts,
                    penalties=penalties,
                )

        if readiness == "low":
            injury_level = _level(candidate.get("injury_caution_level"))
            if injury_level == "high":
                score += _record_score_delta(
                    delta=-18,
                    internal_reason="low readiness high injury caution penalty",
                    public_reason="오늘 컨디션이 낮아 부상 주의도가 높은 운동은 우선순위를 낮췄습니다.",
                    rule_code="readiness_injury_caution_penalty",
                    constraint_code="injury_caution_level",
                    profile_signal_code="readiness",
                    score_reasons=reasons,
                    display_reasons=display_reasons,
                    boosts=boosts,
                    penalties=penalties,
                )
            elif injury_level == "medium":
                score += _record_score_delta(
                    delta=-8,
                    internal_reason="low readiness medium injury caution penalty",
                    public_reason="오늘 컨디션이 낮아 부상 주의도가 중간인 운동의 추천 우선순위를 낮췄습니다.",
                    rule_code="readiness_injury_caution_penalty",
                    constraint_code="injury_caution_level",
                    profile_signal_code="readiness",
                    score_reasons=reasons,
                    display_reasons=display_reasons,
                    boosts=boosts,
                    penalties=penalties,
                )
            if _level(candidate.get("failure_penalty")) == "high":
                score += _record_score_delta(
                    delta=-12,
                    internal_reason="low readiness failure penalty",
                    public_reason="실패 시 부담이 큰 운동이라 낮은 컨디션에서는 우선순위를 낮췄습니다.",
                    rule_code="readiness_failure_penalty",
                    constraint_code="failure_penalty",
                    profile_signal_code="readiness",
                    score_reasons=reasons,
                    display_reasons=display_reasons,
                    boosts=boosts,
                    penalties=penalties,
                )
            if _level(candidate.get("technical_difficulty")) == "high":
                score += _record_score_delta(
                    delta=-8,
                    internal_reason="low readiness technical difficulty penalty",
                    public_reason="기술 난이도가 높은 운동이라 낮은 컨디션에서는 추천 우선순위를 낮췄습니다.",
                    rule_code="readiness_technical_difficulty_penalty",
                    constraint_code="technical_difficulty",
                    profile_signal_code="readiness",
                    score_reasons=reasons,
                    display_reasons=display_reasons,
                    boosts=boosts,
                    penalties=penalties,
                )

        if goal_effect_field:
            effect_level = _level(candidate.get(goal_effect_field))
            effect_delta = _EFFECT_SCORE.get(effect_level, 0)
            if effect_delta:
                score += _record_score_delta(
                    delta=effect_delta,
                    internal_reason=f"{goal_effect_field} {effect_level}",
                    public_reason=f"현재 목표와 관련된 효과 지표({effect_level})가 추천 점수에 반영되었습니다.",
                    rule_code="goal_effect_match",
                    constraint_code=goal_effect_field,
                    profile_signal_code="goal",
                    score_reasons=reasons,
                    display_reasons=display_reasons,
                    boosts=boosts,
                    penalties=penalties,
                )

        stf_level = _level(candidate.get("stimulus_to_fatigue"))
        stf_delta = _STF_SCORE.get(stf_level, 0)
        if stf_delta:
            score += _record_score_delta(
                delta=stf_delta,
                internal_reason=f"stimulus_to_fatigue {stf_level}",
                public_reason=f"자극 대비 피로도 지표({stf_level})가 추천 점수에 반영되었습니다.",
                rule_code="stimulus_to_fatigue",
                constraint_code="stimulus_to_fatigue",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        femur_level = str(
            anthropometry_signals.get("femurLengthLevel")
            or anthropometry_signals.get("femur_length_level")
            or ("long" if "long_femur" in anthropometry_signal_set else "")
            or ""
        ).strip().lower()
        if femur_level == "long" and _level_rank(candidate.get("long_femur_sensitivity")) >= 2:
            delta = -14 if _level(candidate.get("long_femur_sensitivity")) == "high" else -6
            score += _record_score_delta(
                delta=delta,
                internal_reason=f"long femur sensitivity {candidate.get('long_femur_sensitivity')}",
                public_reason="대퇴 길이 신호와 운동의 체형 민감도가 맞물려 세팅 난이도를 보수적으로 반영했습니다.",
                rule_code="anthropometry_sensitivity_penalty",
                constraint_code="long_femur_sensitivity",
                profile_signal_code="long_femur",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )
        elif femur_level == "long" and primary in {"quadriceps", "hamstring", "gluteal", "adductor", "abductors"}:
            score += _record_score_delta(
                delta=0,
                internal_reason=f"long_femur anthropometry reviewed sensitivity {candidate.get('long_femur_sensitivity') or 'unknown'}",
                public_reason="긴 대퇴 비율 신호를 검토했으며, 이 후보는 체형 민감도가 낮거나 중립적인 후보입니다.",
                rule_code="anthropometry_sensitivity_review",
                constraint_code="long_femur_sensitivity",
                profile_signal_code="long_femur",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        arm_level = str(
            anthropometry_signals.get("armLengthLevel")
            or anthropometry_signals.get("arm_length_level")
            or ("long" if "long_arm" in anthropometry_signal_set else "")
            or ""
        ).strip().lower()
        if arm_level == "long" and _level(candidate.get("long_arm_sensitivity")) == "high":
            score += _record_score_delta(
                delta=-10,
                internal_reason="long arm sensitivity high",
                public_reason="팔 길이 신호와 운동의 체형 민감도가 맞물려 세팅 난이도를 보수적으로 반영했습니다.",
                rule_code="anthropometry_sensitivity_penalty",
                constraint_code="long_arm_sensitivity",
                profile_signal_code="long_arm",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        recent_key_candidates = {
            str(candidate.get("name_kr") or "").strip().lower(),
            str(candidate.get("name_en") or "").strip().lower(),
            str(candidate.get("id") or "").strip().lower(),
        }
        if recent_key_candidates & recent_names:
            score += _record_score_delta(
                delta=-6,
                internal_reason="recent repetition penalty",
                public_reason="최근 수행한 운동과 겹쳐 반복 노출을 줄였습니다.",
                rule_code="recent_repetition_penalty",
                profile_signal_code="recent_sets",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        if candidate_keys & preferred_set:
            score += _record_score_delta(
                delta=20,
                internal_reason="user preferred exercise",
                public_reason="사용자가 선호한 운동이라 우선순위가 올라갔습니다.",
                rule_code="user_preference_boost",
                profile_signal_code="user_preference",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        if feedback_adjustments is not None:
            candidate_id = str(candidate.get("id") or "").strip().lower()
            feedback_score = _resolve_feedback_score(feedback_adjustments.get(candidate_id))
            if feedback_score != 0.0:
                feedback_delta = int(round(feedback_score))
                score += _record_score_delta(
                    delta=feedback_delta,
                    internal_reason=f"feedback adj {feedback_score:+.1f}",
                    public_reason=f"과거 피드백 점수({feedback_score:+.1f})가 추천 순위에 반영되었습니다.",
                    rule_code="feedback_adjustment",
                    profile_signal_code="user_feedback",
                    score_reasons=reasons,
                    display_reasons=display_reasons,
                    boosts=boosts,
                    penalties=penalties,
                )

        output_candidate = dict(candidate)
        scored.append({
            **output_candidate,
            "score": score,
            "score_reasons": reasons,
            "score_display_reasons": display_reasons,
            "score_boosts": boosts,
            "score_penalties": penalties,
            "option_policy_severity": option_policy.severity,
            "option_policy_allowed": option_policy.allowed,
            "option_policy_root_reason_code": option_policy.root_reason_code,
            "option_policy_user_message_ko": option_policy.user_message_ko,
            "option_policy_evidence_codes": option_policy.evidence_codes,
            "exercise_risk_profiles": sorted(derive_exercise_risk_profiles(candidate)),
            "score_breakdown_version": "v2",
        })

    sorted_candidates = sorted(
        scored,
        key=lambda ex: (
            -ex["score"],
            int(ex.get("efficiency_tier") or 99),
            str(ex.get("id") or ""),
        ),
    )
    from .pull_composition import reviewed_row_setup

    if not any(reviewed_row_setup(c) == "bent_over" for c in sorted_candidates):
        # This admission is for a complementary role, not an unreviewed general
        # substitute when equipment or another filter removed the main setup.
        sorted_candidates = [c for c in sorted_candidates
                             if reviewed_row_setup(c) != "chest_supported"]
    diversified = _diversify_ranked_candidates(sorted_candidates, top_n, target_muscles)
    if (
        is_intermediate and target_split_label == "full_body"
        and goal == "hypertrophy" and default_gym_style
        and resolved_session_structure == "standard"
        and readiness in {"normal", "high"}
        and not normalized_specialization_focus and not normalized_training_method
        and not normalized_adjunct_options and not preferred_set
    ):
        diversified = _preserve_full_body_flat_press_equipment_choice(
            diversified, sorted_candidates,
        )
    with_style_adjunct = (
        diversified
        if normalized_adjunct_options
        else _ensure_requested_style_adjunct(
            selected=diversified,
            sorted_candidates=sorted_candidates,
            normalized_training_style=normalized_training_style,
            top_n=top_n,
        )
    )
    with_requested_adjunct = _ensure_requested_adjunct_option(
        selected=with_style_adjunct,
        sorted_candidates=sorted_candidates,
        adjunct_options=normalized_adjunct_options,
        top_n=top_n,
    )
    with_focus = _ensure_specialization_focus_candidate(
        selected=with_requested_adjunct,
        sorted_candidates=sorted_candidates,
        specialization_focus=normalized_specialization_focus,
        adjunct_options=normalized_adjunct_options,
        target_muscles=target_muscles,
        top_n=top_n,
    )
    with_full_body_press = _ensure_full_body_vertical_press_candidate(
        selected=with_focus,
        sorted_candidates=sorted_candidates,
        target_muscles=target_muscles,
        target_split_label=target_split_label,
        specialization_focus=normalized_specialization_focus,
        adjunct_options=normalized_adjunct_options,
        top_n=top_n,
    )
    balanced_pull_choices = (
        str(target_split_label or "").strip().lower() == "pull"
        and str(experience_level or "").strip().lower() in {"intermediate", "advanced"}
        and str(goal or "").strip().lower() == "hypertrophy"
        and readiness in {"normal", "high"}
        and int(time_available_min or 0) >= 50
        and str(session_structure or "standard").strip().lower() == "standard"
        and normalized_training_style == "gym"
        and not normalized_specialization_focus
        and not training_method
        and not normalized_adjunct_options
        and not pain_areas
        and not injury_conditions
        and not condition_policies
        and not mobility_limit_set
        and not restricted_movement_patterns
        and str(shoulder_condition_status or "none").strip().lower() == "none"
        and not any(int(level or 0) > 0 for level in doms.values())
    )
    return (
        _ensure_pull_back_choices(
            selected=with_full_body_press, sorted_candidates=sorted_candidates, top_n=top_n,
        ) if balanced_pull_choices else with_full_body_press
    )


def _resolve_feedback_score(adjustment) -> float:
    if adjustment is None:
        return 0.0
    raw = getattr(adjustment, "blended_adjustment", adjustment)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return max(-10.0, min(10.0, value))


def _compact_candidate_prompt_enabled() -> bool:
    return str(os.getenv("ROUTINE_COMPACT_CANDIDATE_PROMPT", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _candidate_reasons_for_prompt(ex: dict, *, compact: bool) -> str:
    if compact:
        reasons = list(ex.get("score_display_reasons") or [])[:3]
        if not reasons:
            reasons = list(ex.get("score_reasons") or [])[:3]
    else:
        reasons = list(ex.get("score_reasons") or [])
    return "; ".join(str(reason) for reason in reasons if str(reason).strip()) or "server ranked"


def _format_compact_candidate_for_prompt(ex: dict) -> str:
    display_name = _candidate_display_name(ex)
    secondary = str(ex.get("secondary_muscle") or "none").strip()
    equipment = str(ex.get("selected_equipment_code") or ex.get("equipment_req") or "unknown").strip()
    pain_triggers = str(ex.get("pain_triggers") or "none").strip()
    substitutions = "none"
    substitution_options = ex.get("substitution_options") or []
    if substitution_options:
        first = substitution_options[0]
        substitutions = (
            f"{first.get('exercise_name')}[{first.get('relation_type')}:{first.get('constraint_code')}]"
        )
    adjunct_purpose = str(ex.get("adjunct_purpose_ko") or "none").strip()
    return (
        f"- {display_name} (ID:{ex['id']}, score:{ex.get('score', 0)}, "
        f"type:{ex.get('movement_type') or 'UNKNOWN'}, primary:{ex.get('primary_muscle') or 'unknown'}, "
        f"secondary:{secondary}, equipment:{equipment}, painTriggers:{pain_triggers}, "
        f"injuryCaution:{ex.get('injury_caution_level') or 'unknown'}, "
        f"painRisk:{ex.get('pain_risk_level') or 'unknown'}, "
        f"load:lumbar={ex.get('lumbar_load') or 'unknown'}/axial={ex.get('axial_load') or 'unknown'}/"
        f"knee={ex.get('knee_shear_load') or 'unknown'}/shoulder={ex.get('shoulder_impingement_risk') or 'unknown'}/"
        f"wrist={ex.get('wrist_stress') or 'unknown'}, "
        f"mobility:ankle={ex.get('ankle_dorsiflexion_demand') or 'unknown'}/"
        f"hip={ex.get('hip_flexion_demand') or 'unknown'}/"
        f"shoulder={ex.get('shoulder_flexion_demand') or 'unknown'}/"
        f"wrist={ex.get('wrist_extension_demand') or 'unknown'}, "
        f"effect:H={ex.get('hypertrophy_effect') or 'unknown'}/S={ex.get('strength_effect') or 'unknown'}/"
        f"STF={ex.get('stimulus_to_fatigue') or 'unknown'}, "
        f"difficulty:tech={ex.get('technical_difficulty') or 'unknown'}/"
        f"balance={ex.get('balance_requirement') or 'unknown'}/failure={ex.get('failure_penalty') or 'unknown'}, "
        f"adjunctPurpose:{adjunct_purpose}, substitute:{substitutions}, "
        f"reason:{_candidate_reasons_for_prompt(ex, compact=True)})"
    )


def format_candidates_for_prompt(candidates: List[dict], compact: Optional[bool] = None) -> str:
    if not candidates:
        return "선택된 후보 운동이 없습니다."
    use_compact = _compact_candidate_prompt_enabled() if compact is None else compact
    lines = []
    for ex in candidates:
        display_name = _candidate_display_name(ex)
        if use_compact:
            lines.append(_format_compact_candidate_for_prompt(ex))
            continue
        secondary = str(ex.get("secondary_muscle") or "none").strip()
        pain_triggers = str(ex.get("pain_triggers") or "none").strip()
        equipment = str(ex.get("selected_equipment_code") or ex.get("equipment_req") or "unknown").strip()
        reasons = _candidate_reasons_for_prompt(ex, compact=False)
        caution = (
            f"caution=injury:{ex.get('injury_caution_level') or 'unknown'}, "
            f"painRisk:{ex.get('pain_risk_level') or 'unknown'}"
        )
        joint_load = (
            f"jointLoad=lumbar:{ex.get('lumbar_load') or 'unknown'}, "
            f"axial:{ex.get('axial_load') or 'unknown'}, "
            f"knee:{ex.get('knee_shear_load') or 'unknown'}, "
            f"shoulder:{ex.get('shoulder_impingement_risk') or 'unknown'}, "
            f"wrist:{ex.get('wrist_stress') or 'unknown'}"
        )
        mobility = (
            f"mobility=ankle:{ex.get('ankle_dorsiflexion_demand') or 'unknown'}, "
            f"hip:{ex.get('hip_flexion_demand') or 'unknown'}, "
            f"shoulder:{ex.get('shoulder_flexion_demand') or 'unknown'}, "
            f"wrist:{ex.get('wrist_extension_demand') or 'unknown'}"
        )
        effect = (
            f"effect=hypertrophy:{ex.get('hypertrophy_effect') or 'unknown'}, "
            f"strength:{ex.get('strength_effect') or 'unknown'}, "
            f"stf:{ex.get('stimulus_to_fatigue') or 'unknown'}"
        )
        difficulty = (
            f"difficulty=technical:{ex.get('technical_difficulty') or 'unknown'}, "
            f"balance:{ex.get('balance_requirement') or 'unknown'}, "
            f"failure:{ex.get('failure_penalty') or 'unknown'}"
        )
        adjunct_purpose = str(ex.get("adjunct_purpose_ko") or "none").strip()
        substitution_options = ex.get("substitution_options") or []
        substitutions = "none"
        if substitution_options:
            substitutions = "; ".join(
                f"{item.get('exercise_name')}[{item.get('relation_type')}:{item.get('constraint_code')}]"
                for item in substitution_options[:2]
            )
        lines.append(
            f"- {display_name} (ID: {ex['id']}, score={ex.get('score', 0)}, "
            f"movement_type={ex.get('movement_type') or 'UNKNOWN'}, "
            f"primary={ex.get('primary_muscle') or 'unknown'}, secondary={secondary}, "
            f"equipment={equipment}, pain_triggers={pain_triggers}, "
            f"{caution}, {joint_load}, {mobility}, {effect}, {difficulty}, "
            f"adjunctPurpose={adjunct_purpose}, substitutions={substitutions}, reason={reasons})"
        )
    return "\n".join(lines)

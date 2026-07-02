"""후보 운동 점수화 및 프롬프트 포맷터."""
import re
from typing import Any, Dict, List, Optional

from .schemas import PainAreaEntry, RecentSetRecord
from .registry import LARGE_MUSCLE_SLUGS, ACCESSORY_MUSCLE_SLUGS, LOADED_EQUIPMENT_TOKENS


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
_MAX_CANDIDATES_PER_FAMILY = 2
_FAMILY_VARIATION_TOKENS = {
    "v",
    "파워",
    "템포",
    "정점정지",
    "하단",
    "상단",
    "정지",
    "느린",
    "이완",
    "3초",
    "5초",
    "싱글암",
    "양측",
}
_EQUIPMENT_AGNOSTIC_FAMILIES = {
    "deadlift",
    "step_up_down",
    "swing",
    "pushup_plus",
    "split_squat",
    "leg_press",
}
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


def _normalize_constraint_tokens(values: Optional[List[str]]) -> set[str]:
    return {
        str(value).strip().lower().replace("-", "_")
        for value in (values or [])
        if str(value).strip()
    }


def _normalize_body_part(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _candidate_family_key(candidate: dict) -> str:
    """동일 운동의 미세 변형이 후보 상단을 독점하지 않도록 묶는 키."""
    raw_name = str(candidate.get("name_kr") or candidate.get("name_en") or candidate.get("id") or "")
    normalized = re.sub(r"v\d+[-_]\d+", "", raw_name.lower())
    compact = re.sub(r"[\s/_()·,.-]+", "", normalized)
    movement_family = ""
    family_patterns = [
        ("deadlift", ("데드리프트", "deadlift", "rdl")),
        ("step_up_down", ("스텝다운", "스텝업", "stepdown", "stepup", "step down", "step up")),
        ("swing", ("스윙", "swing")),
        ("pushup_plus", ("푸쉬업플러스", "푸쉬업 플러스", "pushupplus", "push up plus")),
        ("split_squat", ("스플릿스쿼트", "스플릿 스쿼트", "불가리안", "split squat", "bulgarian")),
        ("lunge", ("런지", "lunge")),
        ("squat", ("스쿼트", "squat")),
        ("leg_press", ("레그프레스", "프레스/스쿼트", "프레스스쿼트", "legpress", "leg press")),
        ("leg_curl", ("레그컬", "레그 컬", "legcurl", "leg curl")),
        ("lat_pulldown", ("랫풀다운", "latpulldown", "lat pulldown")),
        ("row", ("로우", "row")),
        ("chest_press", ("체스트프레스", "체스트 프레스", "chestpress", "chest press")),
        ("fly", ("플라이", "크로스오버", "fly", "crossover")),
    ]
    for family, patterns in family_patterns:
        if any(pattern.replace(" ", "") in compact for pattern in patterns):
            movement_family = family
            break
    tokens = [
        token
        for token in re.split(r"[\s/_()·,.-]+", normalized)
        if token and token not in _FAMILY_VARIATION_TOKENS and not token.isdigit()
    ]
    stem = " ".join(tokens[:5]) if tokens else normalized[:32]
    equipment_family = ""
    if movement_family not in _EQUIPMENT_AGNOSTIC_FAMILIES:
        equipment_family = ",".join(sorted(_normalize_tokens(candidate.get("equipment_req")))[:2])
    return ":".join([
        str(candidate.get("primary_muscle") or ""),
        str(candidate.get("movement_type") or ""),
        equipment_family,
        movement_family or stem,
    ]).lower()


def _diversify_ranked_candidates(candidates: List[dict], top_n: int) -> List[dict]:
    if top_n <= 0:
        return []
    selected: List[dict] = []
    overflow: List[dict] = []
    family_counts: Dict[str, int] = {}
    for candidate in candidates:
        family_key = _candidate_family_key(candidate)
        count = family_counts.get(family_key, 0)
        if count < _MAX_CANDIDATES_PER_FAMILY:
            selected.append(candidate)
            family_counts[family_key] = count + 1
        else:
            overflow.append(candidate)
        if len(selected) >= top_n:
            break
    if len(selected) < top_n:
        selected.extend(overflow[: top_n - len(selected)])
    return selected[:top_n]


def _pain_tokens_from_entries(pain_areas: List[PainAreaEntry]) -> set[str]:
    return {
        _normalize_body_part(p.body_part)
        for p in pain_areas
        if isinstance(p, PainAreaEntry) and p.body_part
    }


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
    return normalized in {"beginner", "novice", "newbie", "entry", "초보", "초보자", "입문"}


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
) -> bool:
    blocked = {item.strip().upper() for item in blocked_equipment}
    candidate_equipment = _normalize_tokens(candidate.get("equipment_req")) - {"BODYWEIGHT"}
    if candidate_equipment & blocked:
        return False

    pain_tokens = {p.body_part.lower() for p in pain_areas if p.body_part}
    candidate_pain = str(candidate.get("pain_triggers") or "").lower()
    normalized_pain_tokens = _pain_tokens_from_entries(pain_areas)
    if any(token in candidate_pain for token in pain_tokens):
        return False
    return not _primary_muscle_conflicts_with_pain(candidate, normalized_pain_tokens)


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
) -> List[dict]:
    """후보 운동을 deterministic하게 점수화해 상위 N개를 반환한다."""
    target_set = set(target_muscles)
    scored: List[dict] = []

    doms = doms_db or {}
    blocked_equipment = blocked_equipment or []
    pain_areas = pain_areas or []
    preferred_set = {eid.strip().lower() for eid in (preferred_exercise_ids or [])}
    unpreferred_set = {eid.strip().lower() for eid in (unpreferred_exercise_ids or [])}
    mobility_limit_set = _normalize_constraint_tokens(mobility_limits)
    condition_policies = condition_policies or []
    anthropometry_signals = anthropometry_signals or {}
    readiness = str(readiness_level or "normal").strip().lower()
    is_beginner = _is_beginner_experience(experience_level)
    goal_effect_field = _goal_effect_field(goal)
    pain_body_parts = _pain_tokens_from_entries(pain_areas)
    has_lumbar_constraint = (
        bool(pain_body_parts & {"lower-back", "back", "lumbar"})
        or _condition_policy_matches_lumbar(condition_policies)
    )
    clearance_required = (
        _professional_clearance_missing(condition_policies)
        or _condition_policy_requires_clearance(condition_policies)
    )
    recent_names = {
        record.exercise_name.strip().lower()
        for record in (recent_sets or [])
        if record.exercise_name
    }

    for candidate in candidates:
        candidate_keys = {
            str(candidate.get("id") or "").strip().lower(),
            str(candidate.get("name_en") or "").strip().lower(),
        }
        if candidate_keys & unpreferred_set:
            continue

        exclusion_reasons: List[str] = []
        if not _candidate_is_safe(candidate, blocked_equipment, pain_areas):
            candidate_equipment = _normalize_tokens(candidate.get("equipment_req")) - {"BODYWEIGHT"}
            blocked = {item.strip().upper() for item in blocked_equipment}
            if candidate_equipment & blocked:
                exclusion_reasons.append("excluded unavailable equipment")
            pain_tokens = {p.body_part.lower() for p in pain_areas if p.body_part}
            candidate_pain = str(candidate.get("pain_triggers") or "").lower()
            if any(token in candidate_pain for token in pain_tokens):
                exclusion_reasons.append("excluded pain trigger")
            if _primary_muscle_conflicts_with_pain(candidate, pain_body_parts):
                exclusion_reasons.append("excluded direct pain primary muscle")
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

        equipment_req = candidate.get("equipment_req")
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

        if goal_effect_field in {"hypertrophy_effect", "strength_effect"} and _is_sport_or_cardio_context_candidate(candidate):
            score += _record_score_delta(
                delta=-22,
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
        secondary_pain_overlap = _secondary_muscle_pain_overlap(candidate, pain_body_parts)
        if secondary_pain_overlap:
            score += _record_score_delta(
                delta=-18,
                internal_reason=f"secondary pain area overlap {', '.join(sorted(secondary_pain_overlap))}",
                public_reason="통증/부상 부위가 보조 자극 부위와 겹쳐 우선순위를 낮췄습니다.",
                rule_code="secondary_pain_area_penalty",
                constraint_code="current_pain_areas",
                profile_signal_code="pain",
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

        doms_level = doms.get(primary, 0)
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
                medium_reason="허리 제약 조건에 따라 요추 부하가 중간 수준인 운동은 보수적으로 낮췄습니다.",
                rule_code="lumbar_load_constraint_penalty",
                constraint_code="lumbar_load",
                profile_signal_code="pain_or_condition",
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
                profile_signal_code="pain_or_condition",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        if pain_body_parts & {"knees", "knee"}:
            score += _apply_level_penalty(
                candidate=candidate,
                field="knee_shear_load",
                high_delta=-30,
                medium_delta=-12,
                high_reason="무릎 통증 신호에 비해 무릎 전단 부하가 큰 운동이라 우선순위가 낮아졌습니다.",
                medium_reason="무릎 통증 신호가 있어 무릎 전단 부하가 중간인 운동을 보수적으로 낮췄습니다.",
                rule_code="knee_load_constraint_penalty",
                constraint_code="knee_shear_load",
                profile_signal_code="pain",
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
                high_reason="어깨 통증 신호에 비해 어깨 충돌 위험 지표가 높아 우선순위가 낮아졌습니다.",
                medium_reason="어깨 통증 신호가 있어 어깨 부담이 중간인 운동을 보수적으로 낮췄습니다.",
                rule_code="shoulder_impingement_constraint_penalty",
                constraint_code="shoulder_impingement_risk",
                profile_signal_code="pain",
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
                high_reason="손목/전완 통증 신호에 비해 손목 부담이 큰 운동이라 우선순위가 낮아졌습니다.",
                medium_reason="손목/전완 통증 신호가 있어 손목 부담이 중간인 운동을 보수적으로 낮췄습니다.",
                rule_code="wrist_stress_constraint_penalty",
                constraint_code="wrist_stress",
                profile_signal_code="pain",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        if pain_body_parts or condition_policies:
            injury_level = _level(candidate.get("injury_caution_level"))
            if injury_level == "high":
                score += _record_score_delta(
                    delta=-14,
                    internal_reason="pain context high injury caution penalty",
                    public_reason="통증/주의 맥락이 있어 일반 부상 주의도가 높은 운동은 우선순위를 낮췄습니다.",
                    rule_code="pain_context_injury_caution_penalty",
                    constraint_code="injury_caution_level",
                    profile_signal_code="pain_or_condition",
                    score_reasons=reasons,
                    display_reasons=display_reasons,
                    boosts=boosts,
                    penalties=penalties,
                )
            elif injury_level == "medium":
                score += _record_score_delta(
                    delta=-5,
                    internal_reason="pain context medium injury caution penalty",
                    public_reason="통증/주의 맥락이 있어 일반 부상 주의도가 중간인 운동을 보수적으로 낮췄습니다.",
                    rule_code="pain_context_injury_caution_penalty",
                    constraint_code="injury_caution_level",
                    profile_signal_code="pain_or_condition",
                    score_reasons=reasons,
                    display_reasons=display_reasons,
                    boosts=boosts,
                    penalties=penalties,
                )

        if (pain_body_parts or condition_policies) and _is_ballistic_or_high_coordination_candidate(candidate):
            score += _record_score_delta(
                delta=-14,
                internal_reason="pain context ballistic coordination penalty",
                public_reason="통증/주의 맥락에서는 탄성·착지·고협응 보강 운동을 메인 후보보다 낮게 배치했습니다.",
                rule_code="pain_context_ballistic_coordination_penalty",
                constraint_code="movement_context",
                profile_signal_code="pain_or_condition",
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
                profile_signal_code="pain_or_condition",
                score_reasons=reasons,
                display_reasons=display_reasons,
                boosts=boosts,
                penalties=penalties,
            )

        mobility_penalties = [
            (
                "limited_ankle_dorsiflexion",
                "ankle_dorsiflexion_demand",
                "발목 배굴 가동성 제한에 비해 발목 요구도가 높은 운동이라 우선순위가 낮아졌습니다.",
                "발목 배굴 가동성 제한이 있어 발목 요구도가 중간인 운동을 보수적으로 낮췄습니다.",
                "limited_ankle_dorsiflexion",
            ),
            (
                "limited_hip_flexion",
                "hip_flexion_demand",
                "고관절 굴곡 가동성 제한에 비해 고관절 요구도가 높은 운동이라 우선순위가 낮아졌습니다.",
                "고관절 굴곡 가동성 제한이 있어 고관절 요구도가 중간인 운동을 보수적으로 낮췄습니다.",
                "limited_hip_flexion",
            ),
            (
                "limited_shoulder_flexion",
                "shoulder_flexion_demand",
                "어깨 굴곡 가동성 제한에 비해 어깨 요구도가 높은 운동이라 우선순위가 낮아졌습니다.",
                "어깨 굴곡 가동성 제한이 있어 어깨 요구도가 중간인 운동을 보수적으로 낮췄습니다.",
                "limited_shoulder_flexion",
            ),
            (
                "limited_wrist_extension",
                "wrist_extension_demand",
                "손목 신전 가동성 제한에 비해 손목 요구도가 높은 운동이라 우선순위가 낮아졌습니다.",
                "손목 신전 가동성 제한이 있어 손목 요구도가 중간인 운동을 보수적으로 낮췄습니다.",
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
                public_reason="전문가 허가가 확인되지 않은 재활/질환 조건이라 고부하 후보를 더 보수적으로 낮췄습니다.",
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

        if is_beginner:
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
                    public_reason="오늘 컨디션이 낮아 부상 주의도가 중간인 운동을 보수적으로 낮췄습니다.",
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
                    public_reason="기술 난이도가 높은 운동이라 낮은 컨디션에서는 보수적으로 낮췄습니다.",
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

        arm_level = str(
            anthropometry_signals.get("armLengthLevel")
            or anthropometry_signals.get("arm_length_level")
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

        scored.append({
            **candidate,
            "score": score,
            "score_reasons": reasons,
            "score_display_reasons": display_reasons,
            "score_boosts": boosts,
            "score_penalties": penalties,
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
    return _diversify_ranked_candidates(sorted_candidates, top_n)


def _resolve_feedback_score(adjustment) -> float:
    if adjustment is None:
        return 0.0
    raw = getattr(adjustment, "blended_adjustment", adjustment)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return max(-10.0, min(10.0, value))


def format_candidates_for_prompt(candidates: List[dict]) -> str:
    if not candidates:
        return "선택된 후보 운동이 없습니다."
    lines = []
    for ex in candidates:
        secondary = str(ex.get("secondary_muscle") or "none").strip()
        pain_triggers = str(ex.get("pain_triggers") or "none").strip()
        equipment = str(ex.get("equipment_req") or "unknown").strip()
        reasons = "; ".join(ex.get("score_reasons", [])) or "server ranked"
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
        substitution_options = ex.get("substitution_options") or []
        substitutions = "none"
        if substitution_options:
            substitutions = "; ".join(
                f"{item.get('exercise_name')}[{item.get('relation_type')}:{item.get('constraint_code')}]"
                for item in substitution_options[:2]
            )
        lines.append(
            f"- {ex['name_kr']} (ID: {ex['id']}, score={ex.get('score', 0)}, "
            f"movement_type={ex.get('movement_type') or 'UNKNOWN'}, "
            f"primary={ex.get('primary_muscle') or 'unknown'}, secondary={secondary}, "
            f"equipment={equipment}, pain_triggers={pain_triggers}, "
            f"{caution}, {joint_load}, {mobility}, {effect}, {difficulty}, "
            f"substitutions={substitutions}, reason={reasons})"
        )
    return "\n".join(lines)

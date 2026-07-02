"""후보 운동 점수화 및 프롬프트 포맷터."""
from typing import Dict, List, Optional

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
    return not any(token in candidate_pain for token in pain_tokens)


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
) -> List[dict]:
    """후보 운동을 deterministic하게 점수화해 상위 N개를 반환한다."""
    target_set = set(target_muscles)
    scored: List[dict] = []

    doms = doms_db or {}
    blocked_equipment = blocked_equipment or []
    pain_areas = pain_areas or []
    preferred_set = {eid.strip().lower() for eid in (preferred_exercise_ids or [])}
    unpreferred_set = {eid.strip().lower() for eid in (unpreferred_exercise_ids or [])}
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
            "score_breakdown_version": "v1",
        })

    return sorted(
        scored,
        key=lambda ex: (
            -ex["score"],
            int(ex.get("efficiency_tier") or 99),
            str(ex.get("id") or ""),
        ),
    )[:top_n]


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
        lines.append(
            f"- {ex['name_kr']} (ID: {ex['id']}, score={ex.get('score', 0)}, "
            f"movement_type={ex.get('movement_type') or 'UNKNOWN'}, "
            f"primary={ex.get('primary_muscle') or 'unknown'}, secondary={secondary}, "
            f"equipment={equipment}, pain_triggers={pain_triggers}, reason={reasons})"
        )
    return "\n".join(lines)

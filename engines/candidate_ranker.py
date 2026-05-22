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
        reasons: List[str] = []

        efficiency = int(candidate.get("efficiency_tier") or 0)
        score += efficiency * 10
        reasons.append(f"efficiency {efficiency}")

        movement_type = str(candidate.get("movement_type") or "").upper()
        if movement_type == "COMPOUND":
            score += 15
            reasons.append("compound priority")
        elif movement_type == "ISOLATION":
            score += 5
            reasons.append("isolation support")

        equipment_req = candidate.get("equipment_req")
        if _is_loadable_equipment(equipment_req):
            score += 18
            reasons.append("loadable equipment priority")
        elif _is_bodyweight_only(equipment_req):
            score -= 18
            reasons.append("bodyweight deprioritized")

        primary = str(candidate.get("primary_muscle") or "").strip()
        secondary = {
            part.strip()
            for part in str(candidate.get("secondary_muscle") or "").split(",")
            if part.strip()
        }
        if primary and primary in target_set:
            score += 20
            reasons.append("primary target match")
        if secondary & target_set:
            score += 8
            reasons.append("secondary target match")
        if primary in LARGE_MUSCLE_SLUGS:
            score += 10
            reasons.append("large muscle priority")
        elif primary in ACCESSORY_MUSCLE_SLUGS:
            score -= 4
            reasons.append("accessory volume guard")

        doms_level = doms.get(primary, 0)
        if doms_level >= 3:
            continue
        if doms_level == 2:
            score -= 25
            reasons.append("doms moderate penalty")
        elif doms_level == 1:
            score -= 10
            reasons.append("doms mild penalty")

        recent_key_candidates = {
            str(candidate.get("name_kr") or "").strip().lower(),
            str(candidate.get("name_en") or "").strip().lower(),
            str(candidate.get("id") or "").strip().lower(),
        }
        if recent_key_candidates & recent_names:
            score -= 6
            reasons.append("recent repetition penalty")

        if candidate_keys & preferred_set:
            score += 20
            reasons.append("user preferred exercise")

        scored.append({
            **candidate,
            "score": score,
            "score_reasons": reasons,
        })

    return sorted(
        scored,
        key=lambda ex: (
            -ex["score"],
            -int(ex.get("efficiency_tier") or 0),
            str(ex.get("id") or ""),
        ),
    )[:top_n]


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

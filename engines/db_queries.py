"""DB 조회 함수 — user_profiles, workout_sets, exercise_tier."""
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session
from sqlalchemy import text

from .schemas import UserProfileContext, RecentSetRecord, PainAreaEntry


def get_user_profile_context(db: Session, user_id: str) -> Optional[UserProfileContext]:
    import json

    row = db.execute(
        text("""
            SELECT goal_type, split_type, split_label, experience_level,
                   strength_baseline, equipment_access, pain_areas
            FROM user_profiles
            WHERE user_id = :uid
        """),
        {"uid": user_id}
    ).fetchone()

    if row is None:
        return None

    r = dict(row._mapping)

    def _parse(val):
        if isinstance(val, str):
            return json.loads(val)
        return val if val is not None else {}

    def _parse_list(val):
        parsed = _parse(val)
        return parsed if isinstance(parsed, list) else []

    def _parse_strength_baseline(val):
        parsed = _parse(val)
        if isinstance(parsed, dict):
            return parsed
        if not isinstance(parsed, list):
            return {}

        baseline: Dict[str, Dict[str, Any]] = {}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            exercise_id = item.get("exerciseId") or item.get("exercise_id")
            exercise_name = item.get("exerciseNameSnapshot") or item.get("exercise_name_snapshot")
            weight = item.get("workingWeightKg") or item.get("working_weight_kg") or item.get("weight_kg")
            reps = item.get("reps")
            if not weight or not reps:
                continue
            value = {
                "exercise_id": str(exercise_id) if exercise_id else None,
                "exercise_name_snapshot": exercise_name,
                "weight_kg": float(weight),
                "reps": int(reps),
            }
            if exercise_id:
                baseline[str(exercise_id)] = value
            if exercise_name:
                baseline[str(exercise_name)] = value
        return baseline

    return UserProfileContext(
        goal_type=r["goal_type"],
        split_type=r["split_type"],
        split_label=r.get("split_label"),
        experience_level=r.get("experience_level"),
        strength_baseline=_parse_strength_baseline(r.get("strength_baseline")),
        equipment_access=_parse_list(r.get("equipment_access")),
        pain_areas=_parse_list(r.get("pain_areas")),
    )


def get_recent_sets(db: Session, user_id: str, limit: int = 30) -> List[RecentSetRecord]:
    """최근 3회 세션의 working 세트를 최신 순으로 반환한다."""
    session_rows = db.execute(
        text("""
            SELECT workout_session_id
            FROM workout_sessions
            WHERE user_id = :uid
            ORDER BY workout_date DESC
            LIMIT 3
        """),
        {"uid": user_id}
    ).fetchall()

    if not session_rows:
        return []

    session_ids = [r[0] for r in session_rows]
    sid_placeholders = ", ".join([f":sid{i}" for i in range(len(session_ids))])
    params: dict = {f"sid{i}": sid for i, sid in enumerate(session_ids)}
    params["limit"] = limit

    rows = db.execute(
        text(f"""
            SELECT ws.exercise_id, ws.exercise_name_snapshot, et.primary_muscle,
                   ws.weight_kg, ws.reps, ws.rir, ws.rpe, ws.is_failure
            FROM workout_sets ws
            LEFT JOIN exercise_tier et
              ON CAST(et.id AS CHAR CHARACTER SET utf8mb4) COLLATE utf8mb4_unicode_ci = ws.exercise_id COLLATE utf8mb4_unicode_ci
                 OR et.name_kr COLLATE utf8mb4_unicode_ci = ws.exercise_name_snapshot COLLATE utf8mb4_unicode_ci
                 OR et.name_en COLLATE utf8mb4_unicode_ci = ws.exercise_name_snapshot COLLATE utf8mb4_unicode_ci
            WHERE ws.workout_session_id IN ({sid_placeholders})
              AND ws.set_type = 'working'
            ORDER BY ws.created_at DESC
            LIMIT :limit
        """),
        params
    ).fetchall()

    return [
        RecentSetRecord(
            exercise_id=str(r[0]) if r[0] is not None else None,
            exercise_name=r[1],
            primary_muscle=str(r[2]) if r[2] is not None else None,
            weight_kg=float(r[3]) if r[3] is not None else None,
            reps=int(r[4]),
            rir=float(r[5]) if r[5] is not None else None,
            rpe=float(r[6]) if r[6] is not None else None,
            is_failure=bool(r[7]) if r[7] is not None else False,
        )
        for r in rows
    ]


def get_candidate_exercises(
    db: Session,
    target_muscles: List[str],
    unavailable_equipment: List[str],
    pain_areas: List[PainAreaEntry],
) -> List[dict]:
    """
    근육 부위 필터 + 통증 부위 제외 + 사용 불가 장비 블랙리스트 필터.
    - unavailable_equipment: 블랙리스트 방식 (기존 화이트리스트에서 변경)
    - BODYWEIGHT는 unavailable_equipment 여부와 무관하게 항상 허용
    """
    if not target_muscles:
        return []

    muscle_placeholders = ", ".join([f":m{i}" for i in range(len(target_muscles))])
    params: dict = {f"m{i}": m for i, m in enumerate(target_muscles)}

    pain_body_parts = [
        p.body_part for p in pain_areas
        if isinstance(p, PainAreaEntry) and p.body_part
    ]
    if pain_body_parts:
        not_like_parts = " AND ".join(
            [f"pain_triggers NOT LIKE :p{i}" for i in range(len(pain_body_parts))]
        )
        pain_filter = f"AND (pain_triggers IS NULL OR ({not_like_parts}))"
        for i, part in enumerate(pain_body_parts):
            params[f"p{i}"] = f"%{part}%"
    else:
        pain_filter = ""

    query = text(f"""
        SELECT id, name_kr, name_en, primary_muscle, secondary_muscle,
               equipment_req, difficulty_tier, efficiency_tier,
               pain_triggers, movement_type
        FROM exercise_tier
        WHERE primary_muscle IN ({muscle_placeholders})
        {pain_filter}
        ORDER BY efficiency_tier DESC
    """)

    rows = db.execute(query, params).fetchall()

    blocked = {e.strip().upper() for e in unavailable_equipment}

    candidates = []
    for row in rows:
        row_dict = dict(row._mapping)
        equip_str = row_dict.get("equipment_req")
        if equip_str is None:
            candidates.append(row_dict)
            continue
        exercise_equip = {e.strip().upper() for e in equip_str.split(",")}
        non_bw = exercise_equip - {"BODYWEIGHT"}
        # 비-맨몸 장비가 모두 블랙리스트에 없거나, 맨몸 운동인 경우 포함
        if not non_bw or not (non_bw & blocked):
            candidates.append(row_dict)

    return candidates

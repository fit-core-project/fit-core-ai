"""DB 조회 함수 — user_profiles, workout_sets, exercise_tier."""
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from sqlalchemy.orm import Session
from sqlalchemy import text

from .schemas import UserProfileContext, RecentSetRecord, PainAreaEntry


SQLITE_DB_PATH = Path(
    os.getenv(
        "FITCORE_SQLITE_DB_PATH",
        str(Path(__file__).resolve().parents[1] / "fit_core.sqlite"),
    )
)


def _connect_local_catalog_db() -> sqlite3.Connection:
    if not SQLITE_DB_PATH.exists():
        raise FileNotFoundError(f"Local exercise catalog DB not found: {SQLITE_DB_PATH}")
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_exercise_details_from_sqlite(ids: List[str]) -> List[dict]:
    """운동 백과 상세 필드 조회.

    운영 루틴 DB와 별개로, 로컬 고도화 카탈로그(`fit_core.sqlite`)의
    가동성/관절부하/효과/난이도/체형 민감도 정보를 프론트 백과 API에 제공한다.
    """
    normalized_ids = [str(item).strip() for item in ids if str(item).strip()]
    if not normalized_ids:
        return []

    placeholders = ", ".join(["?"] * len(normalized_ids))
    query = f"""
        SELECT
            et.id,
            et.name_kr,
            et.name_en,
            et.primary_muscle,
            mr.ankle_dorsiflexion_demand,
            mr.hip_flexion_demand,
            mr.shoulder_flexion_demand,
            mr.wrist_extension_demand,
            mr.overhead_position_required,
            jl.lumbar_load,
            jl.axial_load,
            jl.knee_shear_load,
            jl.shoulder_impingement_risk,
            jl.wrist_stress,
            ep.hypertrophy_effect,
            ep.strength_effect,
            ep.power_effect,
            ep.rehab_utility,
            ep.loadability,
            ep.progression_ceiling,
            ep.mobility_effect,
            ep.uniqueness,
            ep.stimulus_to_fatigue,
            dp.technical_difficulty,
            dp.balance_requirement,
            dp.failure_penalty,
            ap.limb_length_sensitivity,
            ap.long_femur_sensitivity,
            ap.long_arm_sensitivity,
            ap.torso_angle_demand,
            ap.machine_adjustability,
            ap.setup_modification_available
        FROM exercise_tier et
        LEFT JOIN exercise_mobility_requirement mr
            ON mr.exercise_id = et.id
        LEFT JOIN exercise_joint_load_profile jl
            ON jl.exercise_id = et.id
        LEFT JOIN exercise_effect_profile ep
            ON ep.exercise_id = et.id
        LEFT JOIN exercise_difficulty_profile dp
            ON dp.exercise_id = et.id
        LEFT JOIN exercise_anthropometry_sensitivity_profile ap
            ON ap.exercise_id = et.id
        WHERE et.id IN ({placeholders})
    """

    with _connect_local_catalog_db() as conn:
        rows = conn.execute(query, normalized_ids).fetchall()

    by_id = {str(row["id"]): dict(row) for row in rows}
    return [by_id[item] for item in normalized_ids if item in by_id]


def get_exercise_substitution_map_from_sqlite(
    source_exercise_id: Optional[str] = None,
    relation_type: Optional[str] = None,
    constraint_code: Optional[str] = None,
    limit: int = 5000,
) -> List[dict]:
    """운동 대체/회귀 관계 맵 조회.

    프론트 백과 화면은 이 값을 relation_type별로 묶어 대체 운동,
    장비 대체, 재활 후보, 변형 운동 후보로 표시한다.
    """
    where_clauses: List[str] = []
    params: List[Any] = []

    if source_exercise_id:
        where_clauses.append("m.source_exercise_id = ?")
        params.append(source_exercise_id)
    if relation_type:
        where_clauses.append("m.relation_type = ?")
        params.append(relation_type)
    if constraint_code:
        where_clauses.append("m.constraint_code = ?")
        params.append(constraint_code)

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    safe_limit = max(1, min(int(limit or 5000), 5000))

    query = f"""
        SELECT
            m.source_exercise_id,
            src.name_kr AS source_name_kr,
            src.name_en AS source_name_en,
            m.target_exercise_id,
            tgt.name_kr AS target_name_kr,
            tgt.name_en AS target_name_en,
            m.relation_type,
            m.constraint_code,
            m.reason,
            m.source_type,
            m.confidence,
            m.review_required
        FROM exercise_regression_progression_map m
        LEFT JOIN exercise_tier src
            ON src.id = m.source_exercise_id
        LEFT JOIN exercise_tier tgt
            ON tgt.id = m.target_exercise_id
        {where_sql}
        ORDER BY m.source_exercise_id, m.relation_type, m.target_exercise_id
        LIMIT ?
    """
    params.append(safe_limit)

    with _connect_local_catalog_db() as conn:
        rows = conn.execute(query, params).fetchall()

    return [dict(row) for row in rows]


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
               pain_triggers, movement_type, substitute_exercise_ids
        FROM exercise_tier
        WHERE primary_muscle IN ({muscle_placeholders})
        {pain_filter}
        ORDER BY efficiency_tier ASC
    """)

    rows = db.execute(query, params).fetchall()

    blocked = {e.strip().upper() for e in unavailable_equipment}

    candidates = []
    substitute_ids: Set[str] = set()
    for row in rows:
        row_dict = dict(row._mapping)
        equip_str = row_dict.get("equipment_req")
        if _is_equipment_allowed(equip_str, blocked):
            candidates.append(row_dict)
            continue

        substitute_ids.update(_parse_id_tokens(row_dict.get("substitute_exercise_ids")))

    for candidate in candidates:
        if str(candidate.get("id")) in substitute_ids:
            candidate["score_reasons"] = ["mapped substitute for unavailable equipment"]

    existing_ids = {str(candidate.get("id")) for candidate in candidates}
    missing_substitute_ids = substitute_ids - existing_ids
    pain_tokens = {part.lower() for part in pain_body_parts}
    if missing_substitute_ids:
        substitute_placeholders = ", ".join([f":sid{i}" for i in range(len(missing_substitute_ids))])
        substitute_params = {f"sid{i}": sid for i, sid in enumerate(sorted(missing_substitute_ids))}
        substitute_query = text(f"""
            SELECT id, name_kr, name_en, primary_muscle, secondary_muscle,
                   equipment_req, difficulty_tier, efficiency_tier,
                   pain_triggers, movement_type, substitute_exercise_ids
            FROM exercise_tier
            WHERE CAST(id AS CHAR) IN ({substitute_placeholders})
        """)
        for row in db.execute(substitute_query, substitute_params).fetchall():
            row_dict = dict(row._mapping)
            if (
                _is_equipment_allowed(row_dict.get("equipment_req"), blocked)
                and _is_pain_allowed(row_dict.get("pain_triggers"), pain_tokens)
            ):
                row_dict["score_reasons"] = ["mapped substitute for unavailable equipment"]
                candidates.append(row_dict)

    return candidates


def _parse_id_tokens(raw: Optional[str]) -> Set[str]:
    if not raw:
        return set()
    return {token.strip() for token in str(raw).split(",") if token.strip()}


def _is_equipment_allowed(raw: Optional[str], blocked: Set[str]) -> bool:
    if raw is None:
        return True
    exercise_equip = {e.strip().upper() for e in str(raw).split(",") if e.strip()}
    non_bw = exercise_equip - {"BODYWEIGHT"}
    return not non_bw or not (non_bw & blocked)


def _is_pain_allowed(raw: Optional[str], pain_tokens: Set[str]) -> bool:
    if not pain_tokens:
        return True
    candidate_pain = str(raw or "").lower()
    return not any(token in candidate_pain for token in pain_tokens)

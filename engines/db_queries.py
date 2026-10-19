"""DB 조회 함수 — user_profiles, workout_sets, exercise_tier."""
import csv
import hashlib
import io
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set

from sqlalchemy.orm import Session
from sqlalchemy import inspect, text

from .schemas import UserProfileContext, RecentSetRecord, PainAreaEntry
from .equipment_policy import (
    canonical_equipment_req,
    select_equipment_option,
    source_equipment_identity_matches,
    _SUPPORTED_SOURCE_SHA256,
    verified_source_equipment_fallback,
    verified_source_snapshot_available,
)
from .muscle_taxonomy import (
    filter_candidates_to_canonical_targets,
    recommendation_taxonomy_is_eligible,
    source_query_muscles_for_canonical_targets,
)
from .prescription.metadata import classify_exercise_prescription_metadata
from .catalog_identity import load_identity_edges, resolve_catalog_identity


SQLITE_DB_PATH = Path(
    os.getenv(
        "FITCORE_SQLITE_DB_PATH",
        str(Path(__file__).resolve().parents[1] / "fit_core.sqlite"),
    )
)
ROUTINE_ADJUNCT_CATALOG_PATH = Path(
    os.getenv(
        "FITCORE_ROUTINE_ADJUNCT_CATALOG_PATH",
        str(Path(__file__).resolve().parent / "data" / "routine_adjunct_catalog_v1.json"),
    )
)
CANONICAL_SOURCE_CATALOG_PATH = Path(
    os.getenv(
        "FITCORE_CANONICAL_SOURCE_CATALOG_PATH",
        str(
            Path(__file__).resolve().parents[1]
            / "data"
            / "provenance"
            / "v25_canonical_filter_core.csv"
        ),
    )
)
CORE_ADMISSION_REVIEW_PATH = Path(
    os.getenv(
        "FITCORE_CORE_ADMISSION_REVIEW_PATH",
        str(
            Path(__file__).resolve().parents[1]
            / "data"
            / "provenance"
            / "shoulder-main-admission-review-v1-2026-09-14.json"
        ),
    )
)
FOREARM_PATTERN_REVIEW_PATH = Path(
    os.getenv(
        "FITCORE_FOREARM_PATTERN_REVIEW_PATH",
        str(
            Path(__file__).resolve().parents[1]
            / "data"
            / "provenance"
            / "forearm-pattern-admission-review-v1-2026-09-17.json"
        ),
    )
)
FOREARM_PATTERN_REVIEW_VERSION = "forearm-pattern-admission-v1-2026-09-17"
_SUPPORTED_FOREARM_PATTERN_REVIEW_SHA256 = (
    "23d9b2fede3ea89a408c5b6b9042d25ffb08520983c8b66c59754aebf6e190f2"
)
_SUPPORTED_CORE_ADMISSION_REVIEW_SHA256 = (
    "9d4d7b6fe5a3a8b4bd2b4d31f16d882d15ad2316a0d09a938f488e2af1436768"
)
_CORE_ADMISSION_DERIVED_FIELDS = {
    "core_admission_original_beginner_default_allowed",
    "core_admission_original_primary_muscle",
    "catalog_original_main_routine_default_allowed",
    "core_admission_original_secondary_muscle",
    "core_admission_review_version",
    "core_admission_reviewed_involvement_muscles",
    "core_admission_source_sha256",
    "core_admission_scope",
    "core_forearm_pattern_reviewed",
    "core_chest_focus_reviewed",
    "core_main_role_reviewed",
    "core_pull_row_setup",
    "core_pull_row_review_version",
}


def _revoke_derived_core_admission(candidate: dict) -> dict:
    original_beginner = candidate.get("core_admission_original_beginner_default_allowed")
    original_primary = candidate.get("core_admission_original_primary_muscle")
    original_secondary = candidate.get("core_admission_original_secondary_muscle")
    sanitized = {
        key: value
        for key, value in candidate.items()
        if key not in _CORE_ADMISSION_DERIVED_FIELDS
    }
    if any(key in candidate for key in _CORE_ADMISSION_DERIVED_FIELDS):
        sanitized["main_routine_default_allowed"] = 0
    if original_primary is not None:
        sanitized["primary_muscle"] = original_primary
    if original_secondary is not None:
        sanitized["secondary_muscle"] = original_secondary
    if original_beginner is not None:
        sanitized["beginner_default_allowed"] = original_beginner
    if candidate.get("core_chest_focus_reviewed"):
        sanitized.pop("specialization_focus_codes", None)
    return sanitized

_CATALOG_EXTENSION_COLUMNS = [
    "ankle_dorsiflexion_demand",
    "hip_flexion_demand",
    "shoulder_flexion_demand",
    "wrist_extension_demand",
    "overhead_position_required",
    "lumbar_load",
    "axial_load",
    "knee_shear_load",
    "shoulder_impingement_risk",
    "wrist_stress",
    "hypertrophy_effect",
    "strength_effect",
    "power_effect",
    "rehab_utility",
    "loadability",
    "progression_ceiling",
    "mobility_effect",
    "uniqueness",
    "stimulus_to_fatigue",
    "technical_difficulty",
    "balance_requirement",
    "failure_penalty",
    "limb_length_sensitivity",
    "long_femur_sensitivity",
    "long_arm_sensitivity",
    "torso_angle_demand",
    "machine_adjustability",
    "setup_modification_available",
    "display_name_ko",
    "base_exercise_name_kr",
    "variation_name_kr",
    "user_facing_display_name_kr",
    "corrected_primary_muscle",
    "canonical_primary_muscle",
    "canonical_secondary_muscles",
    "canonical_pain_triggers",
    "classification_rule_code",
    "taxonomy_version",
    "taxonomy_review_status",
    "taxonomy_confidence",
    "taxonomy_review_required",
    "canonical_movement_type",
    "prescription_mode",
    "preferred_session_phase",
    "allowed_session_phases",
    "default_duration_sec",
    "default_distance_m",
    "default_interval_work_sec",
    "default_interval_recovery_sec",
    "default_interval_rounds",
    "prescription_default_rest_sec",
    "prescription_rule_code",
    "prescription_metadata_version",
    "prescription_review_status",
    "prescription_confidence",
    "prescription_review_required",
    "knee_flexion_depth_demand",
    "shoulder_elevation_demand",
    "shoulder_elevation_plane",
    "wrist_loaded_extension_demand",
    "deep_squat_position_required",
    "bench_angle_band",
    "movement_profile_confidence",
    "movement_profile_review_required",
    "path_type",
    "bilateral_coupling",
    "grip_freedom",
    "seat_adjustability",
    "start_position_adjustability",
    "pad_axis_adjustability",
    "handle_reach_adjustability",
    "machine_model_dependency",
    "setup_profile_confidence",
    "setup_profile_review_required",
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
    "injury_movement_profile_confidence",
    "injury_movement_profile_review_required",
    "mainstream_gym_score",
    "exercise_familiarity_tier",
    "commonness_level",
    "novelty_level",
    "training_style_tags",
    "user_facing_priority_score",
    "beginner_default_allowed",
    "main_routine_default_allowed",
    "pain_risk_level",
    "quality_score",
]

_GENERIC_USER_FACING_DISPLAY_NAMES = {
    "컬",
    "밴드컬",
    "덤벨컬",
    "케이블컬",
    "머신컬",
    "프레스",
    "로우",
    "레이즈",
    "익스텐션",
}


def _compact_korean_name(value: Any) -> str:
    return "".join(str(value or "").lower().split())


def _select_user_display_name(row: sqlite3.Row) -> str:
    """Pick a user-facing name without losing important variation context."""
    user_facing = str(row.get("user_facing_display_name_kr") or "").strip()
    detailed = str(row.get("display_name_ko") or row.get("catalog_name_kr") or "").strip()
    compact_user_facing = _compact_korean_name(user_facing)
    compact_detailed = _compact_korean_name(detailed)
    directional_tokens = ("로우투하이", "하이투로우", "lowtohigh", "hightolow")
    if (
        user_facing
        and detailed
        and any(token in compact_detailed for token in directional_tokens)
        and not any(token in compact_user_facing for token in directional_tokens)
    ):
        return detailed
    if (
        user_facing
        and compact_user_facing in _GENERIC_USER_FACING_DISPLAY_NAMES
        and detailed
    ):
        return detailed
    return user_facing or detailed

_CATALOG_ENRICH_SELECT_SQL = """
    SELECT
        et.id AS catalog_exercise_id,
        et.name_kr AS catalog_name_kr,
        et.name_en AS catalog_name_en,
        et.primary_muscle AS catalog_primary_muscle,
        et.secondary_muscle AS catalog_secondary_muscle,
        et.equipment_req AS catalog_equipment_req,
        et.movement_type AS catalog_movement_type,
        et.pain_risk_level,
        et.quality_score,
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
        ap.setup_modification_available,
        rm.display_name_ko,
        rm.base_exercise_name_kr,
        rm.variation_name_kr,
        rm.user_facing_display_name_kr,
        rm.corrected_primary_muscle,
        rm.mainstream_gym_score,
        rm.exercise_familiarity_tier,
        rm.commonness_level,
        rm.novelty_level,
        rm.training_style_tags,
        rm.user_facing_priority_score,
        rm.beginner_default_allowed,
        rm.main_routine_default_allowed,
        mt.canonical_primary_muscle,
        mt.canonical_secondary_muscles,
        mt.canonical_pain_triggers,
        mt.classification_rule_code,
        mt.taxonomy_version,
        mt.review_status AS taxonomy_review_status,
        mt.confidence AS taxonomy_confidence,
        mt.review_required AS taxonomy_review_required,
        pm.canonical_movement_type,
        pm.prescription_mode,
        pm.preferred_session_phase,
        pm.allowed_session_phases,
        pm.default_duration_sec,
        pm.default_distance_m,
        pm.default_interval_work_sec,
        pm.default_interval_recovery_sec,
        pm.default_interval_rounds,
        pm.default_rest_sec AS prescription_default_rest_sec,
        pm.classification_rule_code AS prescription_rule_code,
        pm.metadata_version AS prescription_metadata_version,
        pm.review_status AS prescription_review_status,
        pm.confidence AS prescription_confidence,
        pm.review_required AS prescription_review_required,
        (
            SELECT GROUP_CONCAT(focus_code)
            FROM exercise_specialization_focus_map_v1 focus_map
            WHERE focus_map.exercise_id = et.id
              AND focus_map.review_required = 0
        ) AS specialization_focus_codes,
        (
            SELECT GROUP_CONCAT(method_code)
            FROM exercise_training_method_compatibility_v1 method_map
            WHERE method_map.exercise_id = et.id
              AND method_map.compatibility = 'allowed'
              AND method_map.review_required = 0
        ) AS training_method_codes,
        mp.knee_flexion_depth_demand,
        mp.shoulder_elevation_demand,
        mp.shoulder_elevation_plane,
        mp.wrist_loaded_extension_demand,
        mp.deep_squat_position_required,
        mp.bench_angle_band,
        mp.confidence AS movement_profile_confidence,
        mp.review_required AS movement_profile_review_required,
        sf.path_type,
        sf.bilateral_coupling,
        sf.grip_freedom,
        sf.seat_adjustability,
        sf.start_position_adjustability,
        sf.pad_axis_adjustability,
        sf.handle_reach_adjustability,
        sf.machine_model_dependency,
        sf.confidence AS setup_profile_confidence,
        sf.review_required AS setup_profile_review_required,
        im.ankle_region_load_level,
        im.deep_dorsiflexion_demand,
        im.plantar_flexion_load,
        im.single_leg_stability_demand,
        im.impact_demand,
        im.hip_region_load_level,
        im.deep_hip_flexion_demand,
        im.hip_hinge_load,
        im.hip_abduction_load,
        im.hip_adduction_load,
        im.hip_rotation_demand,
        im.elbow_region_load_level,
        im.elbow_flexion_load,
        im.elbow_extension_load,
        im.loaded_grip_demand,
        im.forearm_rotation_load,
        im.upper_limb_weight_bearing_load,
        im.neck_region_load_level,
        im.direct_neck_load,
        im.overhead_axial_load,
        im.sustained_neck_position_demand,
        im.ballistic_impact_demand,
        im.confidence AS injury_movement_profile_confidence,
        im.review_required AS injury_movement_profile_review_required
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
    LEFT JOIN exercise_recommendation_metadata rm
        ON rm.exercise_id = et.id
    LEFT JOIN exercise_muscle_taxonomy_v2 mt
        ON mt.exercise_id = et.id
    LEFT JOIN exercise_prescription_metadata_v2 pm
        ON pm.exercise_id = et.id
    LEFT JOIN exercise_movement_requirement_profile_v2 mp
        ON mp.exercise_id = et.id
    LEFT JOIN exercise_setup_fit_profile_v2 sf
        ON sf.exercise_id = et.id
    LEFT JOIN exercise_managed_injury_movement_profile_v1 im
        ON im.exercise_id = et.id
"""
_LOCAL_CATALOG_ENRICHMENT_CACHE: Optional[dict] = None
_LOCAL_CATALOG_ENRICHMENT_CACHE_KEY: Optional[tuple[str, int]] = None
_SOURCE_EQUIPMENT_CACHE: Optional[dict[str, str]] = None
_SOURCE_EQUIPMENT_CACHE_KEY: Optional[tuple[str, int]] = None
_CORE_ADMISSION_REVIEW_CACHE: Optional[dict[str, dict]] = None
_CORE_ADMISSION_REVIEW_CACHE_KEY: Optional[tuple[str, int, str, int]] = None
_FOREARM_PATTERN_REVIEW_CACHE: Optional[dict[str, dict]] = None
_FOREARM_PATTERN_REVIEW_CACHE_KEY: Optional[tuple[str, int, str, int]] = None


def _connect_local_catalog_db() -> sqlite3.Connection:
    if not SQLITE_DB_PATH.exists():
        raise FileNotFoundError(f"Local exercise catalog DB not found: {SQLITE_DB_PATH}")
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _safe_local_catalog_connection() -> Optional[sqlite3.Connection]:
    try:
        return _connect_local_catalog_db()
    except (FileNotFoundError, sqlite3.Error):
        return None


def _source_query_muscles_from_local_taxonomy(
    target_muscles: List[str],
) -> List[str]:
    """Return source slugs needed to retrieve reviewed canonical targets.

    The operational exercise table still contains known source-label errors
    (for example, calf raises stored as ``abs``). Querying only the source slug
    would discard those rows before the local taxonomy can correct them.
    """

    normalized_targets = [
        str(target).strip()
        for target in target_muscles
        if str(target).strip()
    ]
    if not normalized_targets:
        return []
    conn = _safe_local_catalog_connection()
    if conn is None:
        return []
    try:
        placeholders = _placeholders(normalized_targets)
        rows = conn.execute(
            f"""
            SELECT DISTINCT source_primary_muscle
            FROM exercise_muscle_taxonomy_v2
            WHERE canonical_primary_muscle IN ({placeholders})
              AND COALESCE(review_required, 1) = 0
              AND COALESCE(source_primary_muscle, '') != ''
            ORDER BY source_primary_muscle
            """,
            normalized_targets,
        ).fetchall()
        return [str(row[0]).strip() for row in rows if str(row[0] or "").strip()]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def _placeholders(values: List[Any]) -> str:
    return ", ".join(["?"] * len(values))


def _chunked(values: List[Any], size: int) -> List[List[Any]]:
    return [values[index:index + size] for index in range(0, len(values), size)]


def _normalize_key(value: Any) -> str:
    return str(value or "").strip().lower()


def _compact_name(value: Any) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("/", "")
    )


def _correct_candidate_primary_from_name(name: Any, primary: Any) -> str:
    compact = _compact_name(name)
    primary_slug = str(primary or "").strip()
    if "어브덕션" in compact or "abduction" in compact:
        return "abductors"
    if "어덕션" in compact or "adduction" in compact:
        return "adductor"
    if primary_slug == "front-deltoids" and ("로우" in compact or "row" in compact):
        return "back-deltoids" if "팔꿈치벌림" in compact else "upper-back"
    return ""


def _correct_candidate_movement_type_from_name(name: Any, movement_type: Any) -> str:
    current = str(movement_type or "").strip().upper()
    if current != "COMPOUND":
        return ""

    raw_name = str(name or "").strip().lower()
    compact = _compact_name(raw_name)
    is_fly_or_crossover = (
        ("플라이" in compact and "플라이오" not in compact)
        or "크로스오버" in compact
        or re.search(r"\bfly\b", raw_name) is not None
        or re.search(r"\bcrossover\b", raw_name) is not None
    )
    return "ISOLATION" if is_fly_or_crossover else ""


def _truthy_sqlite_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _derive_injury_caution_level(row: Dict[str, Any]) -> str:
    """운동 자체의 일반 주의도를 보수적으로 산출한다.

    통증 유발 부위(pain_triggers)와 별개로, 운동 자체가 요구하는 관절 부하,
    실패 페널티, 기술 난이도, 기존 pain_risk_level을 합성한 표시용 지표다.
    """
    high_signals = [
        row.get("pain_risk_level") == "high",
        row.get("lumbar_load") == "high",
        row.get("axial_load") == "high",
        row.get("knee_shear_load") == "high",
        row.get("shoulder_impingement_risk") == "high",
        row.get("wrist_stress") == "high",
        row.get("failure_penalty") == "high",
        row.get("technical_difficulty") == "high",
    ]
    medium_signals = [
        row.get("pain_risk_level") == "medium",
        row.get("lumbar_load") == "medium",
        row.get("axial_load") == "medium",
        row.get("knee_shear_load") == "medium",
        row.get("shoulder_impingement_risk") == "medium",
        row.get("wrist_stress") == "medium",
        row.get("failure_penalty") == "medium",
        row.get("technical_difficulty") == "medium",
    ]
    if any(high_signals):
        return "high"
    if any(medium_signals):
        return "medium"
    return "low"


def _fetch_catalog_substitution_options(
    conn: sqlite3.Connection,
    source_ids: List[str],
    per_source_limit: int = 3,
) -> Dict[str, List[dict]]:
    normalized_ids = [str(item).strip() for item in source_ids if str(item).strip()]
    if not normalized_ids:
        return {}

    query_template = """
        SELECT
            m.source_exercise_id,
            m.target_exercise_id,
            tgt.name_kr AS target_name_kr,
            tgt.name_en AS target_name_en,
            m.relation_type,
            m.constraint_code,
            m.reason,
            m.confidence,
            m.review_required
        FROM exercise_regression_progression_map m
        LEFT JOIN exercise_tier tgt
            ON tgt.id = m.target_exercise_id
        LEFT JOIN exercise_muscle_taxonomy_v2 target_taxonomy
            ON target_taxonomy.exercise_id = m.target_exercise_id
        WHERE m.source_exercise_id IN ({placeholders})
          AND COALESCE(target_taxonomy.review_required, 0) = 0
        ORDER BY
            m.source_exercise_id,
            CASE m.relation_type
                WHEN 'painAlternative' THEN 1
                WHEN 'safetyAlternative' THEN 2
                WHEN 'mobilityAlternative' THEN 3
                WHEN 'regression' THEN 4
                WHEN 'equipmentAlternative' THEN 5
                ELSE 9
            END,
            m.confidence DESC,
            m.target_exercise_id
    """
    rows = []
    for batch in _chunked(normalized_ids, 800):
        query = query_template.format(placeholders=_placeholders(batch))
        rows.extend(conn.execute(query, batch).fetchall())
    grouped: Dict[str, List[dict]] = {}
    for row in rows:
        source_id = str(row["source_exercise_id"])
        items = grouped.setdefault(source_id, [])
        if len(items) >= per_source_limit:
            continue
        items.append({
            "exercise_id": str(row["target_exercise_id"]),
            "exercise_name": row["target_name_kr"] or row["target_name_en"] or str(row["target_exercise_id"]),
            "relation_type": row["relation_type"],
            "constraint_code": row["constraint_code"],
            "reason": row["reason"],
        })
    return grouped


def _fetch_approved_condition_policies(
    conn: sqlite3.Connection,
) -> Dict[str, List[dict]]:
    """Load only condition policies that passed the explicit review gate."""
    try:
        rows = conn.execute(
            """
            SELECT
                condition_code,
                exercise_id,
                phase,
                policy,
                modification,
                replacement_exercise_id,
                requires_professional_clearance,
                confidence,
                review_required
            FROM exercise_condition_policy
            WHERE COALESCE(review_required, 1) = 0
            ORDER BY condition_code, exercise_id, phase
            """
        ).fetchall()
    except sqlite3.Error:
        return {}

    grouped: Dict[str, List[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["exercise_id"]), []).append(dict(row))
    return grouped


def _local_catalog_cache_key() -> Optional[tuple[str, int]]:
    try:
        return (str(SQLITE_DB_PATH), SQLITE_DB_PATH.stat().st_mtime_ns)
    except OSError:
        return None


def _source_equipment_cache_key() -> Optional[tuple[str, int]]:
    try:
        return (str(CANONICAL_SOURCE_CATALOG_PATH), CANONICAL_SOURCE_CATALOG_PATH.stat().st_mtime_ns)
    except OSError:
        return None


def _load_source_equipment_by_id() -> Optional[dict[str, dict[str, str]]]:
    global _SOURCE_EQUIPMENT_CACHE, _SOURCE_EQUIPMENT_CACHE_KEY

    cache_key = _source_equipment_cache_key()
    if cache_key is None:
        return None
    if _SOURCE_EQUIPMENT_CACHE is not None and _SOURCE_EQUIPMENT_CACHE_KEY == cache_key:
        return _SOURCE_EQUIPMENT_CACHE
    try:
        raw = CANONICAL_SOURCE_CATALOG_PATH.read_bytes()
        if hashlib.sha256(raw).hexdigest() not in _SUPPORTED_SOURCE_SHA256:
            return None
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
        if not reader.fieldnames or not {"v24_row_uid", "equipment", "name", "english"}.issubset(reader.fieldnames):
            return None
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            exercise_id = str(row.get("v24_row_uid") or "").strip()
            equipment = str(row.get("equipment") or "").strip()
            name_kr = str(row.get("name") or "").strip()
            name_en = str(row.get("english") or "").strip()
            if not exercise_id or exercise_id in rows:
                return None
            rows[exercise_id] = {
                "equipment": equipment,
                "name_kr": name_kr,
                "name_en": name_en,
            }
    except (OSError, UnicodeError, csv.Error):
        return None
    _SOURCE_EQUIPMENT_CACHE = rows
    _SOURCE_EQUIPMENT_CACHE_KEY = cache_key
    return rows


def _load_core_admission_reviews() -> dict[str, dict]:
    global _CORE_ADMISSION_REVIEW_CACHE, _CORE_ADMISSION_REVIEW_CACHE_KEY

    try:
        cache_key = (
            str(CORE_ADMISSION_REVIEW_PATH),
            CORE_ADMISSION_REVIEW_PATH.stat().st_mtime_ns,
            str(CANONICAL_SOURCE_CATALOG_PATH),
            CANONICAL_SOURCE_CATALOG_PATH.stat().st_mtime_ns,
        )
    except OSError:
        return {}
    if (
        _CORE_ADMISSION_REVIEW_CACHE is not None
        and _CORE_ADMISSION_REVIEW_CACHE_KEY == cache_key
    ):
        return _CORE_ADMISSION_REVIEW_CACHE

    try:
        review_raw = CORE_ADMISSION_REVIEW_PATH.read_bytes()
        if hashlib.sha256(review_raw).hexdigest() != _SUPPORTED_CORE_ADMISSION_REVIEW_SHA256:
            return {}
        payload = json.loads(review_raw.decode("utf-8"))
        source_raw = CANONICAL_SOURCE_CATALOG_PATH.read_bytes()
        source_sha = hashlib.sha256(source_raw).hexdigest()
        if (
            payload.get("schemaVersion") != "routine-core-admission-review-v1"
            or payload.get("sourceCatalogSha256") != source_sha
            or source_sha not in _SUPPORTED_SOURCE_SHA256
            or not isinstance(payload.get("reviews"), list)
        ):
            return {}
        source_rows = {
            str(row.get("v24_row_uid") or "").strip(): row
            for row in csv.DictReader(io.StringIO(source_raw.decode("utf-8-sig")))
        }
        reviews: dict[str, dict] = {}
        for review in payload["reviews"]:
            exercise_id = str(review.get("catalogExerciseId") or "").strip()
            source_identity = review.get("sourceIdentity")
            catalog_identity = review.get("catalogIdentity")
            decision = review.get("decision")
            source_row = source_rows.get(exercise_id)
            if (
                not exercise_id
                or exercise_id in reviews
                or not isinstance(source_identity, dict)
                or not isinstance(catalog_identity, dict)
                or not isinstance(decision, dict)
                or source_row is None
                or source_identity.get("v24_row_uid") != exercise_id
                or any(source_row.get(key) != value for key, value in source_identity.items())
                or decision != {
                    "mainRoutineDefaultAllowed": True,
                    "mainRoleReviewed": True,
                    "beginnerDefaultAllowed": False,
                }
                or review.get("reviewedFacts") != {
                    "standardGymIdentity": True,
                    "seatedBenchSupported": True,
                    "overheadPositionRequired": True,
                    "conservativeSafetyInvolvementMuscles": ["side-deltoids"],
                    "directLateralRaiseCredit": False,
                }
            ):
                return {}
            reviews[exercise_id] = {
                **review,
                "decisionVersion": payload.get("decisionVersion"),
                "sourceCatalogSha256": source_sha,
            }
    except (OSError, UnicodeError, csv.Error, json.JSONDecodeError, AttributeError):
        return {}

    _CORE_ADMISSION_REVIEW_CACHE = reviews
    _CORE_ADMISSION_REVIEW_CACHE_KEY = cache_key
    return reviews


def _load_forearm_pattern_admission_reviews() -> dict[str, dict]:
    global _FOREARM_PATTERN_REVIEW_CACHE, _FOREARM_PATTERN_REVIEW_CACHE_KEY

    try:
        cache_key = (
            str(FOREARM_PATTERN_REVIEW_PATH),
            FOREARM_PATTERN_REVIEW_PATH.stat().st_mtime_ns,
            str(CANONICAL_SOURCE_CATALOG_PATH),
            CANONICAL_SOURCE_CATALOG_PATH.stat().st_mtime_ns,
        )
    except OSError:
        return {}
    if (
        _FOREARM_PATTERN_REVIEW_CACHE is not None
        and _FOREARM_PATTERN_REVIEW_CACHE_KEY == cache_key
    ):
        return _FOREARM_PATTERN_REVIEW_CACHE

    try:
        review_raw = FOREARM_PATTERN_REVIEW_PATH.read_bytes()
        source_raw = CANONICAL_SOURCE_CATALOG_PATH.read_bytes()
        source_sha = hashlib.sha256(source_raw).hexdigest()
        if (
            hashlib.sha256(review_raw).hexdigest()
            != _SUPPORTED_FOREARM_PATTERN_REVIEW_SHA256
            or source_sha not in _SUPPORTED_SOURCE_SHA256
        ):
            return {}
        payload = json.loads(review_raw.decode("utf-8"))
        if (
            payload.get("schemaVersion") != "routine-forearm-pattern-admission-v1"
            or payload.get("decisionVersion") != FOREARM_PATTERN_REVIEW_VERSION
            or payload.get("sourceCatalogSha256") != source_sha
            or payload.get("accessedAt") != "2026-09-17"
            or not isinstance(payload.get("anatomyEvidence"), list)
            or not isinstance(payload.get("reviews"), list)
        ):
            return {}
        source_rows = {
            str(row.get("v24_row_uid") or "").strip(): row
            for row in csv.DictReader(io.StringIO(source_raw.decode("utf-8-sig")))
        }
        reviews: dict[str, dict] = {}
        for review in payload["reviews"]:
            exercise_id = str(review.get("catalogExerciseId") or "").strip()
            source_identity = review.get("sourceIdentity")
            source_row = source_rows.get(exercise_id)
            if (
                exercise_id != "V24ROW-004724"
                or exercise_id in reviews
                or not isinstance(source_identity, dict)
                or source_row is None
                or source_identity.get("v24_row_uid") != exercise_id
                or any(source_row.get(key) != value for key, value in source_identity.items())
                or review.get("decision")
                != {
                    "effectivePrimaryMuscle": "forearm",
                    "mainRoutineDefaultAllowed": True,
                    "beginnerDefaultAllowed": False,
                    "specializationFocusOnly": "forearmEmphasis",
                }
                or review.get("reviewKind")
                != "engineering_product_admission_and_anatomical_identity_not_clinical_clearance"
                or review.get("preservedLimitations")
                != {
                    "exerciseFamiliarityTier": "specialized",
                    "commonnessLevel": "specialized",
                    "noveltyLevel": "medium",
                    "injuryMovementProfileReviewRequired": 1,
                    "noNewSafetyAttributes": True,
                }
            ):
                return {}
            reviews[exercise_id] = {
                **review,
                "decisionVersion": payload["decisionVersion"],
                "sourceCatalogSha256": source_sha,
            }
        if set(reviews) != {"V24ROW-004724"}:
            return {}
    except (OSError, UnicodeError, csv.Error, json.JSONDecodeError, AttributeError):
        return {}

    _FOREARM_PATTERN_REVIEW_CACHE = reviews
    _FOREARM_PATTERN_REVIEW_CACHE_KEY = cache_key
    return reviews


def _forearm_pattern_admission_patch(
    exercise_id: str,
    catalog_identity: dict,
    effective_identity: dict,
) -> dict:
    if exercise_id != "V24ROW-004724":
        return {}
    review = _load_forearm_pattern_admission_reviews().get(exercise_id)
    if (
        not review
        or catalog_identity != review.get("catalogIdentity")
        or effective_identity != review.get("effectiveIdentityBeforeReview")
    ):
        return {}
    return {
        "catalog_original_primary_muscle": effective_identity["primary_muscle"],
        "core_admission_original_primary_muscle": effective_identity["primary_muscle"],
        "primary_muscle": "forearm",
        "catalog_original_main_routine_default_allowed": 0,
        "main_routine_default_allowed": 1,
        "specialization_focus_codes": "forearmEmphasis",
        "core_admission_scope": "forearmEmphasis",
        "core_forearm_pattern_reviewed": 1,
        "core_admission_review_version": FOREARM_PATTERN_REVIEW_VERSION,
        "core_admission_source_sha256": review["sourceCatalogSha256"],
    }


def clear_local_catalog_enrichment_cache() -> None:
    global _LOCAL_CATALOG_ENRICHMENT_CACHE, _LOCAL_CATALOG_ENRICHMENT_CACHE_KEY
    global _SOURCE_EQUIPMENT_CACHE, _SOURCE_EQUIPMENT_CACHE_KEY
    global _CORE_ADMISSION_REVIEW_CACHE, _CORE_ADMISSION_REVIEW_CACHE_KEY
    global _FOREARM_PATTERN_REVIEW_CACHE, _FOREARM_PATTERN_REVIEW_CACHE_KEY
    _LOCAL_CATALOG_ENRICHMENT_CACHE = None
    _LOCAL_CATALOG_ENRICHMENT_CACHE_KEY = None
    _SOURCE_EQUIPMENT_CACHE = None
    _SOURCE_EQUIPMENT_CACHE_KEY = None
    _CORE_ADMISSION_REVIEW_CACHE = None
    _CORE_ADMISSION_REVIEW_CACHE_KEY = None
    _FOREARM_PATTERN_REVIEW_CACHE = None
    _FOREARM_PATTERN_REVIEW_CACHE_KEY = None


def _load_local_catalog_enrichment_cache() -> Optional[dict]:
    global _LOCAL_CATALOG_ENRICHMENT_CACHE, _LOCAL_CATALOG_ENRICHMENT_CACHE_KEY

    cache_key = _local_catalog_cache_key()
    if cache_key is None:
        return None
    if (
        _LOCAL_CATALOG_ENRICHMENT_CACHE is not None
        and _LOCAL_CATALOG_ENRICHMENT_CACHE_KEY == cache_key
    ):
        return _LOCAL_CATALOG_ENRICHMENT_CACHE

    conn = _safe_local_catalog_connection()
    if conn is None:
        return None
    try:
        rows = [dict(row) for row in conn.execute(_CATALOG_ENRICH_SELECT_SQL).fetchall()]
        try:
            equipment_rows = conn.execute(
                """
                SELECT exercise_id, equipment_code
                FROM exercise_equipment_option_v2
                ORDER BY exercise_id, equipment_code
                """
            ).fetchall()
        except sqlite3.Error:
            equipment_rows = []
        substitutions = _fetch_catalog_substitution_options(
            conn,
            [str(row["catalog_exercise_id"]) for row in rows],
        )
        approved_condition_policies = _fetch_approved_condition_policies(conn)
    except sqlite3.Error:
        return None
    finally:
        conn.close()

    equipment_by_id: dict[str, list[str]] = {}
    for row in equipment_rows:
        equipment_by_id.setdefault(str(row["exercise_id"]), []).append(
            str(row["equipment_code"])
        )

    cache = {
        "by_id": {_normalize_key(row.get("catalog_exercise_id")): row for row in rows},
        "equipment_by_id": equipment_by_id,
        "substitutions": substitutions,
        "approved_condition_policies": approved_condition_policies,
    }
    _LOCAL_CATALOG_ENRICHMENT_CACHE = cache
    _LOCAL_CATALOG_ENRICHMENT_CACHE_KEY = cache_key
    return cache


def _enrich_candidates_with_local_catalog(candidates: List[dict]) -> List[dict]:
    """루틴 후보에 운동 백과 확장 필드를 붙인다.

    카탈로그 자체 ID 또는 검수된 명시적 연결만 사용한다.
    이름 일치는 검수 후보 생성에만 사용하며 런타임 식별자로 사용하지 않는다.
    """
    if not candidates:
        return candidates

    candidates = [_revoke_derived_core_admission(candidate) for candidate in candidates]

    cache = _load_local_catalog_enrichment_cache()
    if cache is None:
        return [candidate for candidate in candidates
                if not candidate.get("catalog_exercise_id")
                or str(candidate.get("id") or "").startswith("V24ROW-")]

    by_id = cache["by_id"]
    identity_edges = load_identity_edges()
    substitutions = cache["substitutions"]
    approved_condition_policies = cache["approved_condition_policies"]
    core_admission_reviews = _load_core_admission_reviews()
    source_equipment_by_id = _load_source_equipment_by_id()
    if source_equipment_by_id is None and not verified_source_snapshot_available():
        # Neither immutable provenance asset is trustworthy. Returning no
        # candidates fails closed without changing unrelated catalog health data.
        return []

    enriched: List[dict] = []
    for candidate in candidates:
        row = resolve_catalog_identity(candidate, by_id, identity_edges)
        if not row:
            if candidate.get("catalog_exercise_id"):
                if not str(candidate.get("id") or "").startswith("V24ROW-"):
                    # External rows must be freshly read after approval changes;
                    # their already-overlaid source fields cannot be trusted.
                    continue
                # Keep the established native-catalog admission-revocation
                # behavior, but never retain another row's catalog identity.
                candidate = dict(candidate)
                candidate.pop("catalog_exercise_id", None)
            enriched.append(candidate)
            continue

        extension = {
            key: row.get(key)
            for key in _CATALOG_EXTENSION_COLUMNS
            if key in row
        }
        catalog_id = str(row.get("catalog_exercise_id") or "")
        extension["catalog_exercise_id"] = catalog_id
        extension["catalog_source_primary_muscle"] = row.get("catalog_primary_muscle")
        extension["catalog_source_secondary_muscle"] = row.get("catalog_secondary_muscle")
        display_name = _select_user_display_name(row)
        if display_name:
            extension["catalog_original_name_kr"] = candidate.get("name_kr")
            extension["name_kr"] = display_name
        corrected_primary = str(row.get("canonical_primary_muscle") or "").strip()
        corrected_primary = corrected_primary or str(row.get("corrected_primary_muscle") or "").strip()
        corrected_primary = corrected_primary or _correct_candidate_primary_from_name(
            display_name or candidate.get("name_kr") or row.get("catalog_name_kr"),
            candidate.get("primary_muscle"),
        )
        if corrected_primary:
            extension["catalog_original_primary_muscle"] = candidate.get("primary_muscle")
            extension["primary_muscle"] = corrected_primary
        canonical_secondary = str(row.get("canonical_secondary_muscles") or "").strip()
        if canonical_secondary:
            extension["catalog_original_secondary_muscle"] = candidate.get("secondary_muscle")
            extension["secondary_muscle"] = canonical_secondary
        canonical_pain = str(row.get("canonical_pain_triggers") or "").strip()
        if canonical_pain:
            extension["catalog_original_pain_triggers"] = candidate.get("pain_triggers")
            extension["pain_triggers"] = canonical_pain
        source_record = (
            source_equipment_by_id.get(catalog_id)
            if source_equipment_by_id is not None
            else None
        )
        from .muscle_taxonomy import exact_id_primary_taxonomy_patch

        extension.update(exact_id_primary_taxonomy_patch(row, source_record))
        source_equipment = (
            source_record["equipment"]
            if source_record
            and source_equipment_identity_matches(
                source_name_kr=source_record["name_kr"],
                source_name_en=source_record["name_en"],
                catalog_name_kr=row.get("catalog_name_kr"),
                catalog_name_en=row.get("catalog_name_en"),
            )
            else None
        ) or verified_source_equipment_fallback(
            catalog_id,
            name_kr=row.get("catalog_name_kr"),
            name_en=row.get("catalog_name_en"),
        )
        equipment_options = cache["equipment_by_id"].get(catalog_id, [])
        effective_equipment = (
            ", ".join(equipment_options)
            if equipment_options
            else candidate.get("equipment_req")
        )
        provenance_equipment = canonical_equipment_req(
            effective_equipment,
            name_kr=row.get("catalog_name_kr"),
            name_en=row.get("catalog_name_en"),
            detail_subpart=row.get("detail_subpart"),
            source_equipment=source_equipment,
        )
        canonical_equipment = provenance_equipment
        if canonical_equipment != str(candidate.get("equipment_req") or "").strip():
            extension["catalog_original_equipment_req"] = candidate.get("equipment_req")
            extension["equipment_req"] = canonical_equipment
        canonical_movement_type = str(row.get("canonical_movement_type") or "").strip().upper()
        corrected_movement_type = canonical_movement_type or _correct_candidate_movement_type_from_name(
            display_name or candidate.get("name_kr") or row.get("catalog_name_kr"),
            candidate.get("movement_type"),
        )
        if corrected_movement_type and corrected_movement_type != str(candidate.get("movement_type") or "").strip().upper():
            extension["catalog_original_movement_type"] = candidate.get("movement_type")
            extension["movement_type"] = corrected_movement_type
        # Apply the reviewed classification without rebuilding the shared catalog.
        # Use verified source identity, never a caller-supplied display name.
        if (
            source_record
            and source_equipment_identity_matches(
                source_name_kr=source_record["name_kr"],
                source_name_en=source_record["name_en"],
                catalog_name_kr=row.get("catalog_name_kr"),
                catalog_name_en=row.get("catalog_name_en"),
            )
            and canonical_movement_type == "ISOLATION"
            and row.get("prescription_mode") == "reps"
            and (extension.get("primary_muscle") or candidate.get("primary_muscle")) == "upper-back"
        ):
            decision = classify_exercise_prescription_metadata(
                name_kr=row.get("catalog_name_kr"),
                name_en=row.get("catalog_name_en"),
                source_movement_type=canonical_movement_type,
            )
            if decision.classification_rule_code == "standard_lat_pulldown_compound_correction":
                extension.update({
                    "catalog_original_movement_type": canonical_movement_type,
                    "movement_type": decision.canonical_movement_type,
                    "canonical_movement_type": decision.canonical_movement_type,
                    "prescription_mode": decision.prescription_mode,
                    "preferred_session_phase": decision.preferred_session_phase,
                    "allowed_session_phases": decision.allowed_session_phases,
                    "prescription_rule_code": decision.classification_rule_code,
                    "prescription_metadata_version": decision.metadata_version,
                    "prescription_confidence": decision.confidence,
                    "prescription_review_status": decision.review_status,
                    "prescription_review_required": decision.review_required,
                })
        admission_review = core_admission_reviews.get(catalog_id)
        if admission_review and str(candidate.get("id") or "").strip() == catalog_id:
            expected_identity = admission_review["catalogIdentity"]
            actual_identity = {
                "catalog_name_kr": str(row.get("catalog_name_kr") or "").strip(),
                "catalog_name_en": str(row.get("catalog_name_en") or "").strip(),
                "catalog_primary_muscle": str(row.get("catalog_primary_muscle") or "").strip(),
                "catalog_secondary_muscle": str(row.get("catalog_secondary_muscle") or "").strip(),
                "equipment_req": canonical_equipment,
                "canonical_movement_type": canonical_movement_type,
            }
            if (
                actual_identity == expected_identity
                and row.get("beginner_default_allowed") == 0
                and row.get("main_routine_default_allowed") == 0
            ):
                extension["catalog_original_main_routine_default_allowed"] = row.get(
                    "main_routine_default_allowed"
                )
                original_secondary = str(
                    extension.get("secondary_muscle")
                    or candidate.get("secondary_muscle")
                    or ""
                ).strip()
                secondary_parts = [
                    part.strip()
                    for part in original_secondary.split(",")
                    if part.strip()
                ]
                extension["core_admission_original_secondary_muscle"] = (
                    original_secondary
                )
                extension["core_admission_reviewed_involvement_muscles"] = (
                    "side-deltoids"
                )
                if "side-deltoids" not in secondary_parts:
                    secondary_parts.append("side-deltoids")
                extension["secondary_muscle"] = ", ".join(secondary_parts)
                extension["main_routine_default_allowed"] = 1
                extension["core_admission_review_version"] = admission_review[
                    "decisionVersion"
                ]
                extension["core_admission_source_sha256"] = admission_review[
                    "sourceCatalogSha256"
                ]
                extension["core_main_role_reviewed"] = 1
        from .focus_catalog_admission import admission_patch

        extension.update(admission_patch(
            str(candidate.get("id") or ""),
            {
                "name_kr": row.get("catalog_name_kr"),
                "name_en": row.get("catalog_name_en"),
                "primary_muscle": row.get("catalog_primary_muscle"),
                "secondary_muscle": row.get("catalog_secondary_muscle"),
                "equipment_req": canonical_equipment,
                "movement_type": canonical_movement_type,
            },
            {**candidate, **extension},
            CANONICAL_SOURCE_CATALOG_PATH,
        ))
        effective = {**candidate, **extension}
        from .pull_composition import row_setup_patch

        extension.update(row_setup_patch(
            str(candidate.get("id") or ""), row, effective, source_record,
        ))
        from .beginner_machine_press import admission_patch as beginner_press_patch

        extension.update(beginner_press_patch(
            str(candidate.get("id") or ""), row, {**candidate, **extension}, source_record,
        ))
        effective = {**candidate, **extension}
        extension.update(_forearm_pattern_admission_patch(
            str(candidate.get("id") or ""),
            {
                "name_kr": str(row.get("catalog_name_kr") or "").strip(),
                "name_en": str(row.get("catalog_name_en") or "").strip(),
                "primary_muscle": str(row.get("catalog_primary_muscle") or "").strip(),
                "secondary_muscle": str(row.get("catalog_secondary_muscle") or "").strip(),
                "equipment_req": canonical_equipment,
                "movement_type": canonical_movement_type,
                "prescription_mode": str(row.get("prescription_mode") or "").strip(),
            },
            {
                "primary_muscle": str(effective.get("primary_muscle") or "").strip(),
                "secondary_muscle": str(effective.get("secondary_muscle") or "").strip(),
                "equipment_req": str(effective.get("equipment_req") or "").strip(),
                "movement_type": str(effective.get("movement_type") or "").strip(),
                "prescription_mode": str(effective.get("prescription_mode") or "").strip(),
                "mainstream_gym_score": effective.get("mainstream_gym_score"),
                "exercise_familiarity_tier": str(effective.get("exercise_familiarity_tier") or "").strip(),
                "commonness_level": str(effective.get("commonness_level") or "").strip(),
                "novelty_level": str(effective.get("novelty_level") or "").strip(),
                "beginner_default_allowed": effective.get("beginner_default_allowed"),
                "main_routine_default_allowed": effective.get("main_routine_default_allowed"),
            },
        ))
        extension["injury_caution_level"] = _derive_injury_caution_level(row)
        extension["injury_caution_source"] = "catalog_composite_v1"
        extension["substitution_options"] = substitutions.get(catalog_id, [])
        extension["approved_condition_policies"] = approved_condition_policies.get(catalog_id, [])
        enriched.append({**candidate, **extension})

    return enriched


def _load_routine_adjunct_catalog() -> dict:
    try:
        payload = json.loads(ROUTINE_ADJUNCT_CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict) or not isinstance(payload.get("options"), dict):
        return {}
    return payload


def get_adjunct_candidate_exercises(
    adjunct_options: List[str],
    unavailable_equipment: List[str],
    pain_areas: List[PainAreaEntry],
    available_equipment: Optional[List[str]] = None,
) -> List[dict]:
    """Load only Core-reviewed exercises for an explicitly applied adjunct slot."""
    normalized_options = [
        str(option or "").strip().lower().replace("-", "_").replace(" ", "_")
        for option in adjunct_options
        if str(option or "").strip()
    ]
    if not normalized_options:
        return []

    catalog = _load_routine_adjunct_catalog()
    option_rows = catalog.get("options") or {}
    requested_entries: List[tuple[str, dict]] = []
    for option in normalized_options:
        for entry in option_rows.get(option) or []:
            if isinstance(entry, dict) and str(entry.get("exerciseId") or "").strip():
                requested_entries.append((option, entry))
    if not requested_entries:
        return []

    ordered_ids = list(dict.fromkeys(
        str(entry["exerciseId"]).strip()
        for _, entry in requested_entries
    ))
    conn = _safe_local_catalog_connection()
    if conn is None:
        return []
    try:
        rows = conn.execute(
            f"""
            SELECT id, name_kr, name_en, primary_muscle, secondary_muscle,
                   equipment_req, difficulty_tier, efficiency_tier,
                   pain_triggers, movement_type, NULL AS substitute_exercise_ids
            FROM exercise_tier
            WHERE id IN ({_placeholders(ordered_ids)})
            """,
            ordered_ids,
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()

    row_by_id = {str(row["id"]): dict(row) for row in rows}
    blocked = {str(item).strip().upper() for item in unavailable_equipment if str(item).strip()}
    available = {
        str(item).strip().upper()
        for item in (available_equipment or [])
        if str(item).strip()
    }
    pain_tokens = {
        pain.body_part.strip().lower()
        for pain in pain_areas
        if isinstance(pain, PainAreaEntry) and pain.body_part
    }
    selected: List[dict] = []
    by_selected_id: dict[str, dict] = {}
    for option, entry in requested_entries:
        exercise_id = str(entry["exerciseId"]).strip()
        row = row_by_id.get(exercise_id)
        if not row:
            continue
        selected_equipment = select_equipment_option(
            row.get("equipment_req"),
            blocked,
            available,
        )
        if selected_equipment is None:
            continue
        if not _is_pain_allowed(row.get("pain_triggers"), pain_tokens):
            continue
        candidate = by_selected_id.get(exercise_id)
        if candidate is None:
            candidate = {
                **row,
                "selected_equipment_code": selected_equipment,
                "adjunct_option_codes": [],
                "adjunct_catalog_version": catalog.get("catalogVersion"),
                "adjunct_catalog_review_status": catalog.get("reviewStatus"),
                "adjunct_display_name_ko": str(entry.get("displayNameKo") or "").strip(),
                "adjunct_purpose_ko": str(entry.get("purposeKo") or "").strip(),
                "score_reasons": ["core-reviewed adjunct candidate"],
            }
            by_selected_id[exercise_id] = candidate
            selected.append(candidate)
        if option not in candidate["adjunct_option_codes"]:
            candidate["adjunct_option_codes"].append(option)

    enriched = [
        candidate
        for candidate in _enrich_candidates_with_local_catalog(selected)
        if recommendation_taxonomy_is_eligible(candidate)
    ]
    for candidate in enriched:
        display_name = str(candidate.get("adjunct_display_name_ko") or "").strip()
        if display_name:
            candidate["name_kr"] = display_name
            candidate["user_facing_display_name_kr"] = display_name
    return enriched


@dataclass(frozen=True)
class ConstraintProfileLookup:
    status: Literal["found", "not_found", "error"]
    profile: dict


def _empty_constraint_profile() -> dict:
    return {"mobility_limits": [], "anthropometry_signals": {}, "condition_policies": []}


def get_user_constraint_profile_from_sqlite(user_id: Optional[str]) -> dict:
    """Legacy best-effort projection; checked v4 uses the typed lookup instead."""
    return lookup_user_constraint_profile_from_sqlite(user_id).profile


def lookup_user_constraint_profile_from_sqlite(user_id: Optional[str]) -> ConstraintProfileLookup:
    """루틴 추천용 사용자 제약 프로필을 로컬 카탈로그 DB에서 읽는다.

    이 값은 제품 결정이 아니라 deterministic scoring의 입력 재료다. 의료 메모는
    프롬프트에 직접 노출하지 않고 코드/부위/단계/허가 여부만 전달한다.
    """
    if not user_id:
        return ConstraintProfileLookup("not_found", _empty_constraint_profile())

    conn = _safe_local_catalog_connection()
    if conn is None:
        return ConstraintProfileLookup("error", _empty_constraint_profile())

    try:
        mobility_row = conn.execute(
            """
            SELECT ankle_dorsiflexion_level, hip_flexion_level,
                   shoulder_flexion_level, wrist_extension_level
            FROM user_mobility_profile
            WHERE user_id = ?
            """,
            [user_id],
        ).fetchone()
        anthro_row = conn.execute(
            """
            SELECT femur_length_level, arm_length_level, torso_length_level
            FROM user_anthropometry_profile
            WHERE user_id = ?
            """,
            [user_id],
        ).fetchone()
        condition_rows = conn.execute(
            """
            SELECT condition_code, body_part, side, phase, professional_clearance
            FROM user_condition_profile
            WHERE user_id = ?
            """,
            [user_id],
        ).fetchall()
    except sqlite3.Error:
        return ConstraintProfileLookup("error", _empty_constraint_profile())
    finally:
        conn.close()

    mobility_limits: List[str] = []
    if mobility_row:
        mapping = {
            "ankle_dorsiflexion_level": "limited_ankle_dorsiflexion",
            "hip_flexion_level": "limited_hip_flexion",
            "shoulder_flexion_level": "limited_shoulder_flexion",
            "wrist_extension_level": "limited_wrist_extension",
        }
        row_dict = dict(mobility_row)
        for field, code in mapping.items():
            if str(row_dict.get(field) or "").strip().lower() == "limited":
                mobility_limits.append(code)

    anthropometry_signals = dict(anthro_row) if anthro_row else {}
    condition_policies = [
        {
            "conditionCode": row["condition_code"],
            "bodyPart": str(row["body_part"] or "").replace("_", "-"),
            "side": row["side"],
            "phase": row["phase"],
            "professionalClearance": _truthy_sqlite_bool(row["professional_clearance"]),
        }
        for row in condition_rows
    ]

    return ConstraintProfileLookup("found" if mobility_row or anthro_row or condition_rows else "not_found", {
        "mobility_limits": mobility_limits,
        "anthropometry_signals": anthropometry_signals,
        "condition_policies": condition_policies,
    })


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
            et.primary_muscle AS source_primary_muscle,
            COALESCE(
                mt.canonical_primary_muscle,
                rm.corrected_primary_muscle,
                et.primary_muscle
            ) AS primary_muscle,
            et.secondary_muscle AS source_secondary_muscle,
            COALESCE(
                mt.canonical_secondary_muscles,
                et.secondary_muscle
            ) AS secondary_muscle,
            et.equipment_req,
            et.pain_triggers,
            et.movement_type,
            et.difficulty_tier,
            et.efficiency_tier,
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
            ap.setup_modification_available,
            rm.display_name_ko,
            rm.base_exercise_name_kr,
            rm.variation_name_kr,
            rm.user_facing_display_name_kr,
            rm.corrected_primary_muscle,
            rm.mainstream_gym_score,
            rm.exercise_familiarity_tier,
            rm.commonness_level,
            rm.novelty_level,
            rm.training_style_tags,
            rm.user_facing_priority_score,
            rm.beginner_default_allowed,
            rm.main_routine_default_allowed,
            pm.canonical_movement_type,
            pm.prescription_mode,
            pm.preferred_session_phase,
            pm.allowed_session_phases,
            (
                SELECT GROUP_CONCAT(focus_code)
                FROM exercise_specialization_focus_map_v1 focus_map
                WHERE focus_map.exercise_id = et.id
                  AND focus_map.review_required = 0
            ) AS specialization_focus_codes,
            (
                SELECT GROUP_CONCAT(method_code)
                FROM exercise_training_method_compatibility_v1 method_map
                WHERE method_map.exercise_id = et.id
                  AND method_map.compatibility = 'allowed'
                  AND method_map.review_required = 0
            ) AS training_method_codes,
            mp.bench_angle_band
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
        LEFT JOIN exercise_recommendation_metadata rm
            ON rm.exercise_id = et.id
        LEFT JOIN exercise_muscle_taxonomy_v2 mt
            ON mt.exercise_id = et.id
        LEFT JOIN exercise_prescription_metadata_v2 pm
            ON pm.exercise_id = et.id
        LEFT JOIN exercise_movement_requirement_profile_v2 mp
            ON mp.exercise_id = et.id
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
    where_clauses: List[str] = [
        "COALESCE(source_taxonomy.review_required, 0) = 0",
        "COALESCE(target_taxonomy.review_required, 0) = 0",
    ]
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
        LEFT JOIN exercise_muscle_taxonomy_v2 source_taxonomy
            ON source_taxonomy.exercise_id = m.source_exercise_id
        LEFT JOIN exercise_muscle_taxonomy_v2 target_taxonomy
            ON target_taxonomy.exercise_id = m.target_exercise_id
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
    available_equipment: Optional[List[str]] = None,
) -> List[dict]:
    """
    근육 부위 필터 + 통증 부위 제외 + 장비 필터.
    - unavailable_equipment: 블랙리스트 방식 (기존 화이트리스트에서 변경)
    - available_equipment: 값이 있으면 사용 가능 장비 화이트리스트도 함께 강제
    - BODYWEIGHT는 unavailable_equipment 여부와 무관하게 항상 허용
    """
    if not target_muscles:
        return []

    substitute_column_expr = (
        "substitute_exercise_ids"
        if _table_has_column(db, "exercise_tier", "substitute_exercise_ids")
        else "NULL AS substitute_exercise_ids"
    )

    query_target_muscles = list(dict.fromkeys([
        *source_query_muscles_for_canonical_targets(target_muscles),
        *_source_query_muscles_from_local_taxonomy(target_muscles),
    ]))
    muscle_placeholders = ", ".join([f":m{i}" for i in range(len(query_target_muscles))])
    params: dict = {f"m{i}": m for i, m in enumerate(query_target_muscles)}

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
               pain_triggers, movement_type, {substitute_column_expr}
        FROM exercise_tier
        WHERE primary_muscle IN ({muscle_placeholders})
        {pain_filter}
        ORDER BY efficiency_tier ASC
    """)

    rows = db.execute(query, params).fetchall()

    blocked = {e.strip().upper() for e in unavailable_equipment}
    available = {e.strip().upper() for e in (available_equipment or []) if e and e.strip()}

    source_candidates = [dict(row._mapping) for row in rows]
    normalized_targets = {
        str(target or "").strip().lower().replace("_", "-")
        for target in target_muscles
    }
    if "forearm" in normalized_targets:
        reviewed_ids = sorted(_load_forearm_pattern_admission_reviews())
        existing_source_ids = {
            str(candidate.get("id") or "") for candidate in source_candidates
        }
        missing_reviewed_ids = [
            exercise_id
            for exercise_id in reviewed_ids
            if exercise_id not in existing_source_ids
        ]
        if missing_reviewed_ids:
            placeholders = ", ".join(
                f":reviewed{i}" for i in range(len(missing_reviewed_ids))
            )
            reviewed_rows = db.execute(
                text(f"""
                    SELECT id, name_kr, name_en, primary_muscle, secondary_muscle,
                           equipment_req, difficulty_tier, efficiency_tier,
                           pain_triggers, movement_type, {substitute_column_expr}
                    FROM exercise_tier
                    WHERE CAST(id AS CHAR) IN ({placeholders})
                """),
                {
                    f"reviewed{i}": exercise_id
                    for i, exercise_id in enumerate(missing_reviewed_ids)
                },
            ).fetchall()
            source_candidates.extend(dict(row._mapping) for row in reviewed_rows)
    enriched_candidates = [
        candidate
        for candidate in _enrich_candidates_with_local_catalog(source_candidates)
        if recommendation_taxonomy_is_eligible(candidate)
    ]
    pain_tokens = {part.lower() for part in pain_body_parts}
    candidates = []
    substitute_ids: Set[str] = set()
    for row_dict in enriched_candidates:
        equip_str = row_dict.get("equipment_req")
        selected_equipment = select_equipment_option(equip_str, blocked, available)
        if selected_equipment is not None:
            if _is_pain_allowed(row_dict.get("pain_triggers"), pain_tokens):
                row_dict["selected_equipment_code"] = selected_equipment
                candidates.append(row_dict)
            continue

        substitute_ids.update(_parse_id_tokens(row_dict.get("substitute_exercise_ids")))

    for candidate in candidates:
        if str(candidate.get("id")) in substitute_ids:
            candidate["score_reasons"] = ["mapped substitute for unavailable equipment"]

    existing_ids = {str(candidate.get("id")) for candidate in candidates}
    missing_substitute_ids = substitute_ids - existing_ids
    if missing_substitute_ids:
        substitute_placeholders = ", ".join([f":sid{i}" for i in range(len(missing_substitute_ids))])
        substitute_params = {f"sid{i}": sid for i, sid in enumerate(sorted(missing_substitute_ids))}
        substitute_query = text(f"""
            SELECT id, name_kr, name_en, primary_muscle, secondary_muscle,
                   equipment_req, difficulty_tier, efficiency_tier,
                   pain_triggers, movement_type, {substitute_column_expr}
            FROM exercise_tier
            WHERE CAST(id AS CHAR) IN ({substitute_placeholders})
        """)
        substitute_rows = [
            dict(row._mapping)
            for row in db.execute(substitute_query, substitute_params).fetchall()
        ]
        for row_dict in _enrich_candidates_with_local_catalog(substitute_rows):
            if not recommendation_taxonomy_is_eligible(row_dict):
                continue
            selected_equipment = select_equipment_option(
                row_dict.get("equipment_req"),
                blocked,
                available,
            )
            if selected_equipment is not None and _is_pain_allowed(
                row_dict.get("pain_triggers"),
                pain_tokens,
            ):
                row_dict["selected_equipment_code"] = selected_equipment
                row_dict["score_reasons"] = ["mapped substitute for unavailable equipment"]
                candidates.append(row_dict)

    # Source-slug expansion may intentionally load rows from unrelated legacy
    # buckets so their reviewed canonical taxonomy can recover them. Always
    # apply the canonical target filter after enrichment; limiting this step to
    # deltoids leaks unrelated rows when calves/adductors/biceps need recovery.
    return filter_candidates_to_canonical_targets(candidates, target_muscles)


def _table_has_column(db: Session, table_name: str, column_name: str) -> bool:
    try:
        columns = inspect(db.get_bind()).get_columns(table_name)
    except Exception:
        return True
    return any(str(column.get("name")) == column_name for column in columns)


def _parse_id_tokens(raw: Optional[str]) -> Set[str]:
    if not raw:
        return set()
    return {token.strip() for token in str(raw).split(",") if token.strip()}


def _is_equipment_allowed(raw: Optional[str], blocked: Set[str], available: Optional[Set[str]] = None) -> bool:
    return select_equipment_option(raw, blocked, available) is not None


def _is_pain_allowed(raw: Optional[str], pain_tokens: Set[str]) -> bool:
    if not pain_tokens:
        return True
    candidate_pain = str(raw or "").lower()
    return not any(token in candidate_pain for token in pain_tokens)

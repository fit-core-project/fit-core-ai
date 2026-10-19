"""Narrow product admission, not a catalog release or clinical clearance."""
from __future__ import annotations

VERSION = "beginner-machine-press-v1-2026-10-01"
SCOPE = "beginnerFullBodyVerticalPress"
EXERCISE_ID = "V24ROW-000365"
NAME_KO = "머신 시티드 숄더 뉴트럴 그립 프레스"
NAME_EN = "Machine 시티드 숄더 뉴트럴 그립 양측 Shoulder Press"


def admission_patch(exercise_id: str, row: dict, effective: dict, verified_source: dict | None) -> dict:
    """Use only db_queries' hash-verified source record and exact DB identity."""
    if (exercise_id != EXERCISE_ID or not verified_source
            or row.get("catalog_exercise_id") != EXERCISE_ID
            or row.get("catalog_name_kr") != NAME_KO
            or row.get("catalog_name_en") != NAME_EN
            or verified_source.get("name_kr") != NAME_KO
            or verified_source.get("name_en") != NAME_EN
            or verified_source.get("equipment") != "머신"
            or row.get("catalog_primary_muscle") != "front-deltoids"
            or row.get("catalog_secondary_muscle") != "upper-back, triceps, trapezius"
            or effective.get("primary_muscle") != "front-deltoids"
            or effective.get("equipment_req") != "MACHINE"
            or effective.get("movement_type") != "COMPOUND"
            or effective.get("prescription_mode") != "reps"
            or effective.get("main_routine_default_allowed") != 0
            or effective.get("beginner_default_allowed") != 0):
        return {}
    # Preserve pain/quality/licence markers and treat side delts conservatively
    # as involved muscles, not as direct lateral-raise set credit.
    original_secondary = str(effective.get("secondary_muscle") or "")
    secondary = [part.strip() for part in original_secondary.split(",") if part.strip()]
    if "side-deltoids" not in secondary:
        secondary.append("side-deltoids")
    return {
        "catalog_original_main_routine_default_allowed": 0,
        "core_admission_original_beginner_default_allowed": 0,
        "core_admission_original_secondary_muscle": original_secondary,
        "main_routine_default_allowed": 1,
        "beginner_default_allowed": 1,
        "core_main_role_reviewed": 1,
        "core_admission_scope": SCOPE,
        "core_admission_review_version": VERSION,
        "secondary_muscle": ", ".join(secondary),
        "name_kr": "머신 숄더 프레스 (뉴트럴 그립)",
        "user_facing_display_name_kr": "머신 숄더 프레스 (뉴트럴 그립)",
        # The original DB video links to a barbell press, not this machine.
        "video_url": None,
    }


def admission_scope(*, split, experience, goal, readiness, structure, style,
                    pain=None, injuries=None, doms=None, constraints=False,
                    adjuncts=None, focus=None, method=None) -> bool:
    return bool(
        split == "full_body" and experience == "beginner"
        and goal == "hypertrophy" and readiness in {"normal", "high"}
        and structure == "standard" and style == "gym"
        and not pain and not injuries and not constraints
        and not adjuncts and not focus and not method
        and not any(int(v or 0) > 0 for v in (doms or {}).values())
    )

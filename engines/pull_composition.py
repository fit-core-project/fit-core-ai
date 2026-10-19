"""Bounded product composition, not medical clearance or a new dose ceiling."""
from __future__ import annotations

VERSION = "pull-row-setup-v1-2026-10-01"
BENT_OVER_ID = "V24ROW-001379"
SUPPORTED_ID = "V24ROW-001609"
REVIEWED = {
    BENT_OVER_ID: (
        "덤벨 벤트오버 뉴트럴 팔꿈치 45도 로우",
        "Dumbbell 벤트오버 뉴트럴 팔꿈치 45도 Row", "DUMBBELL", "bent_over", 1,
    ),
    SUPPORTED_ID: (
        "로우 머신 체스트 서포트 뉴트럴 팔꿈치 45도 로우",
        "Row Machine 체스트 서포트 뉴트럴 팔꿈치 45도 Row", "MACHINE", "chest_supported", 0,
    ),
}


def row_setup_patch(exercise_id: str, row: dict, effective: dict, verified_source: dict | None) -> dict:
    """Called only with the hash-verified source record from db_queries.

    Identity checks do not infer chest support from MACHINE or a generic row
    title. The original catalog and all load/health metadata remain unchanged.
    """
    expected = REVIEWED.get(exercise_id)
    if not expected or not verified_source:
        return {}
    ko, en, equipment, setup, main_allowed = expected
    if (
        row.get("catalog_exercise_id") != exercise_id
        or row.get("catalog_name_kr") != ko or row.get("catalog_name_en") != en
        or verified_source.get("name_kr") != ko or verified_source.get("name_en") != en
        or row.get("catalog_primary_muscle") != "upper-back"
        or row.get("catalog_secondary_muscle") != "biceps, trapezius"
        or effective.get("primary_muscle") != "upper-back"
        or effective.get("equipment_req") != equipment
        or effective.get("movement_type") != "COMPOUND"
        or effective.get("prescription_mode") != "reps"
        or effective.get("main_routine_default_allowed") != main_allowed
    ):
        return {}
    patch = {"core_pull_row_setup": setup, "core_pull_row_review_version": VERSION}
    if exercise_id == SUPPORTED_ID:
        if effective.get("beginner_default_allowed") != 0:
            return {}
        patch.update(
            catalog_original_main_routine_default_allowed=0,
            main_routine_default_allowed=1,
            core_main_role_reviewed=1,
            core_admission_scope="balancedPullRowComplement",
            core_admission_review_version=VERSION,
            name_kr="머신 체스트 서포티드 로우",
            user_facing_display_name_kr="머신 체스트 서포티드 로우",
        )
    return patch


def reviewed_row_setup(candidate: dict) -> str | None:
    expected = REVIEWED.get(str(candidate.get("id") or ""))
    if (expected and candidate.get("core_pull_row_review_version") == VERSION
            and candidate.get("core_pull_row_setup") == expected[3]
            and candidate.get("primary_muscle") == "upper-back"
            and candidate.get("equipment_req") == expected[2]
            and candidate.get("movement_type") == "COMPOUND"):
        return expected[3]
    return None


def complement_scope(*, split, experience, goal, readiness, minutes, structure,
                     style="gym", focus=None, method=None, adjuncts=None,
                     pain=None, injuries=None, doms=None, constraints=False) -> bool:
    return bool(
        split == "pull" and experience == "advanced" and goal == "hypertrophy"
        and readiness in {"normal", "high"} and int(minutes or 0) >= 60
        and structure == "standard" and style == "gym"
        and not focus and not method and not adjuncts and not pain and not injuries
        and not constraints and not any(int(v or 0) > 0 for v in (doms or {}).values())
    )


def complementary_pair(candidates) -> bool:
    candidates = tuple(candidates)
    return len(candidates) == 2 and {reviewed_row_setup(c) for c in candidates} == {"bent_over", "chest_supported"}


def reviewed_pull_family_key(candidate: dict, family: str, *, enabled: bool) -> str:
    if enabled and family == "horizontal_row" and reviewed_row_setup(candidate) == "chest_supported":
        return f"{family}:chest_supported"
    return family

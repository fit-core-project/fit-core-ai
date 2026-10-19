"""Narrow glute-focus role relation; never rewrites taxonomy or admission."""
from __future__ import annotations

SUPPORT_SLOT = "glute_abduction_support"
SUPPORT_ID = "V24ROW-003682"
RELATION_VERSION = "glute-support-relation-v1-2026-10-10"


def is_reviewed_glute_support(candidate: dict) -> bool:
    expected = {
        "id": SUPPORT_ID,
        "catalog_exercise_id": SUPPORT_ID,
        "catalog_original_name_kr": "힙 어브덕션 머신 시티드 힙 어브덕션",
        "primary_muscle": "abductors",
        "canonical_primary_muscle": "abductors",
        "movement_type": "ISOLATION",
        "canonical_movement_type": "ISOLATION",
        "equipment_req": "MACHINE",
        "prescription_mode": "reps",
        "injury_caution_level": "low",
    }
    return (
        all(candidate.get(key) == value for key, value in expected.items())
        and candidate.get("main_routine_default_allowed") == 1
    )


def get_glute_support_candidates(db, blueprint, unavailable_equipment, pain_areas,
                                 available_equipment=None) -> list[dict]:
    if (blueprint.target_split_label != "custom"
            or blueprint.target_muscles != ("gluteal",)
            or not any(slot.slot_id == SUPPORT_SLOT for slot in blueprint.slots)):
        return []
    from .db_queries import get_candidate_exercises

    # Reuse normal equipment/pain/taxonomy filters, then retain the exact variant.
    # The common ranker still owns DOMS, medical, preference and admission gates.
    return [row for row in get_candidate_exercises(
        db, ["abductors"], unavailable_equipment, pain_areas, available_equipment,
    ) if is_reviewed_glute_support(row)]

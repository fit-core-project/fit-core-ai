from copy import deepcopy
import sqlite3

import pytest

from engines import db_queries as dbq
from engines.prescription.metadata import (
    LAT_PULLDOWN_CORRECTION_VERSION,
    apply_candidate_prescription_metadata,
    classify_exercise_prescription_metadata,
)
from engines.prescription.experience import apply_experience_prescription
from engines.schemas import LLMExercisePlan


def decision(en, kr=""):
    return classify_exercise_prescription_metadata(
        name_en=en, name_kr=kr, source_movement_type="ISOLATION")


@pytest.mark.parametrize("equipment", ["Machine", "Cable", "Band"])
@pytest.mark.parametrize("grip", ["Pronated", "Supinated", "Neutral", "Wide", "Close"])
@pytest.mark.parametrize("variant", ["", "Single Arm "])
def test_standard_names_correct_movement_and_phase(equipment, grip, variant):
    value = decision(f"{equipment} {grip} {variant}Lat Pulldown")
    assert value.canonical_movement_type == "COMPOUND"
    assert value.prescription_mode == "reps"
    assert value.preferred_session_phase == "main_compound"
    assert value.allowed_session_phases == "main_compound,secondary_compound"
    assert value.metadata_version == LAT_PULLDOWN_CORRECTION_VERSION


@pytest.mark.parametrize("name", [
    "Machine Neutral Straight Arm Pulldown", "Machine Neutral Lat Pullover",
    "Machine Neutral Scapular Pulldown", "Machine Neutral Lat Pulldown / Pullover",
    "Machine Neutral Half Kneeling Pulldown", "Lat Pulldown Machine Row",
    "Cable Triceps Pulldown", "Unknown Lat Pulldown", "", None,
])
def test_ambiguous_and_single_joint_names_are_not_promoted(name):
    assert decision(name).canonical_movement_type == "ISOLATION"


@pytest.mark.parametrize("kr", ["스트레이트암 랫 풀다운", "풀오버", "스캐풀라 풀다운", "팔을 펴 랫 풀다운"])
def test_conflicting_korean_identity_prevents_correction(kr):
    assert decision("Machine Neutral Lat Pulldown", kr).canonical_movement_type == "ISOLATION"


def raw_candidate():
    with sqlite3.connect(f"file:{dbq.SQLITE_DB_PATH}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        return dict(conn.execute("SELECT * FROM exercise_tier WHERE id='V24ROW-005645'").fetchone())


def test_real_catalog_runtime_correction_is_idempotent_and_not_admission():
    dbq.clear_local_catalog_enrichment_cache()
    raw = raw_candidate()
    before = deepcopy(raw)
    enriched = dbq._enrich_candidates_with_local_catalog([raw])[0]
    assert raw == before and raw["movement_type"] == "ISOLATION"
    assert enriched["movement_type"] == enriched["canonical_movement_type"] == "COMPOUND"
    assert enriched["catalog_original_movement_type"] == "ISOLATION"
    assert enriched["main_routine_default_allowed"] == 1
    assert enriched["prescription_metadata_version"] == LAT_PULLDOWN_CORRECTION_VERSION
    again = dbq._enrich_candidates_with_local_catalog([enriched])[0]
    for key in ("movement_type", "canonical_movement_type", "catalog_original_movement_type",
                "prescription_metadata_version", "allowed_session_phases"):
        assert again[key] == enriched[key]
    assert raw_candidate() == before


def test_missing_verified_source_does_not_correct_from_display_name(monkeypatch):
    dbq.clear_local_catalog_enrichment_cache()
    monkeypatch.setattr(dbq, "_load_source_equipment_by_id", lambda: None)
    enriched = dbq._enrich_candidates_with_local_catalog([raw_candidate()])[0]
    assert enriched["movement_type"] == enriched["canonical_movement_type"] == "ISOLATION"


def test_source_identity_mismatch_does_not_promote(monkeypatch):
    dbq.clear_local_catalog_enrichment_cache()
    source = deepcopy(dbq._load_source_equipment_by_id())
    source['V24ROW-005645']['name_en'] = 'Machine Neutral Straight Arm Pulldown'
    source['V24ROW-005645']['name_kr'] = 'unreviewed identity'
    monkeypatch.setattr(dbq, "_load_source_equipment_by_id", lambda: source)
    enriched = dbq._enrich_candidates_with_local_catalog([raw_candidate()])[0]
    assert enriched["canonical_movement_type"] == "ISOLATION"


def test_corrected_candidate_uses_existing_compound_prescription():
    dbq.clear_local_catalog_enrichment_cache()
    candidate = dbq._enrich_candidates_with_local_catalog([raw_candidate()])[0]
    exercise = LLMExercisePlan(exercise_id=candidate['id'], exercise_name=candidate['name_kr'],
        movement_type='ISOLATION', primary_muscles=['upper-back'], target_reps=12,
        sets=3, rest_time_sec=75, exercise_rationale='test')
    apply_candidate_prescription_metadata(exercise, candidate)
    apply_experience_prescription(exercise, experience_level='advanced', goal='hypertrophy',
                                  base_reps=12, base_rest_sec=75)
    assert exercise.movement_type == 'COMPOUND'
    assert exercise.rest_time_sec >= 120
    assert exercise.target_reps == 8
    assert exercise.sets <= 4

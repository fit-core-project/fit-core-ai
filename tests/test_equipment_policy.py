import ast
import csv
import json
import os
from pathlib import Path
import sqlite3

import engines.db_queries as db_queries
import engines.equipment_policy as equipment_policy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from engines.db_queries import (
    _enrich_candidates_with_local_catalog,
    clear_local_catalog_enrichment_cache,
    get_candidate_exercises,
)
from engines.equipment_policy import (
    _verified_source_snapshot_by_id,
    canonical_equipment_req,
    equipment_option_profiles,
    parse_equipment_options,
    select_equipment_option,
    source_equipment_identity_matches,
    superset_equipment_compatible,
    verified_source_equipment_fallback,
)


DB_PATH = Path(__file__).resolve().parents[1] / "fit_core.sqlite"


def test_smith_machine_category_token_is_not_an_alternative_option():
    assert parse_equipment_options("SMITH_MACHINE, MACHINE") == ("SMITH_MACHINE",)
    assert select_equipment_option(
        "SMITH_MACHINE, MACHINE",
        available={"MACHINE"},
    ) is None
    assert select_equipment_option(
        "SMITH_MACHINE, MACHINE",
        available={"SMITH_MACHINE"},
    ) == "SMITH_MACHINE"


def test_blocked_alternative_does_not_remove_an_available_option():
    assert select_equipment_option(
        "CABLE, DUMBBELL",
        blocked={"DUMBBELL"},
        available={"CABLE"},
    ) == "CABLE"


def test_bodyweight_alternative_remains_available_without_selected_equipment():
    assert select_equipment_option(
        "DUMBBELL, BODYWEIGHT",
        blocked={"DUMBBELL"},
        available={"MACHINE"},
    ) == "BODYWEIGHT"


def test_missing_or_unknown_equipment_does_not_default_to_bodyweight():
    assert select_equipment_option(None) is None
    assert select_equipment_option("UNKNOWN", available={"BODYWEIGHT"}) is None
    assert select_equipment_option("UNKNOWN,CABLE", available={"CABLE"}) == "CABLE"
    assert select_equipment_option("UNKNOWN,CABLE", blocked={"CABLE"}) is None


def test_source_provenance_marks_unrepresented_equipment_unknown_without_name_inference():
    assert canonical_equipment_req(
        "BODYWEIGHT",
        source_equipment="EZ바",
    ) == "UNKNOWN"
    assert canonical_equipment_req(
        "BODYWEIGHT",
        source_equipment="짐볼",
    ) == "UNKNOWN"
    assert canonical_equipment_req(
        "BODYWEIGHT",
        name_kr="EZ바 라잉 익스텐션",
    ) == "BODYWEIGHT"
    profiles = equipment_option_profiles(
        "V24ROW-002628",
        "BODYWEIGHT",
        source_equipment="EZ바",
    )
    assert profiles[0]["equipment_code"] == "UNKNOWN"
    assert profiles[0]["review_required"] == 1


def test_source_provenance_quarantines_unresolved_band_stability_ball_conflict():
    assert canonical_equipment_req("BAND", source_equipment="짐볼") == "UNKNOWN"
    assert select_equipment_option("UNKNOWN", available={"BAND"}) is None


def test_source_provenance_requires_matching_source_and_catalog_identity():
    assert source_equipment_identity_matches(
        source_name_kr="짐볼 밴드 저항 짐볼 백 익스텐션",
        source_name_en="Stability Ball 밴드 저항 짐볼 백 익스텐션",
        catalog_name_kr="짐볼 밴드 저항 짐볼 백 익스텐션",
        catalog_name_en="Stability Ball 밴드 저항 짐볼 백 익스텐션",
    ) is True
    assert source_equipment_identity_matches(
        source_name_kr="reassigned row",
        source_name_en="Stability Ball 밴드 저항 짐볼 백 익스텐션",
        catalog_name_kr="짐볼 밴드 저항 짐볼 백 익스텐션",
        catalog_name_en="Stability Ball 밴드 저항 짐볼 백 익스텐션",
    ) is False


def test_generated_snapshot_covers_all_verified_rows_and_rejects_identity_drift():
    source = DB_PATH.parent / "data" / "provenance" / "v25_canonical_filter_core.csv"
    with source.open(encoding="utf-8-sig", newline="") as handle:
        special_rows = [
            row
            for row in csv.DictReader(handle)
            if row["equipment"].replace(" ", "").lower() in {"ez바", "짐볼", "swissball"}
        ]

    _verified_source_snapshot_by_id.cache_clear()
    snapshot = _verified_source_snapshot_by_id()
    assert len(special_rows) == 168
    assert len(snapshot) == len(special_rows)
    for row in special_rows:
        fallback_equipment = verified_source_equipment_fallback(
            row["v24_row_uid"],
            name_kr=row["name"],
            name_en=row["english"],
        )
        assert fallback_equipment == row["equipment"]
        assert canonical_equipment_req("BODYWEIGHT", source_equipment=fallback_equipment) == "UNKNOWN"
    assert verified_source_equipment_fallback(
        "V24ROW-002201",
        name_kr="changed identity",
        name_en="Stability Ball 밴드 저항 짐볼 백 익스텐션",
    ) is None


def test_generated_snapshot_is_reproducible_from_current_source(tmp_path):
    from scripts.generate_equipment_provenance_snapshot import build_snapshot

    source = DB_PATH.parent / "data" / "provenance" / "v25_canonical_filter_core.csv"
    generated = build_snapshot(source)
    checked_in = json.loads(
        (
            DB_PATH.parent
            / "data"
            / "provenance"
            / "equipment-provenance-snapshot-v1-2026-09-14.json"
        ).read_text(encoding="utf-8")
    )
    assert generated == checked_in


def test_snapshot_rejects_truncated_but_structurally_valid_payload(monkeypatch, tmp_path):
    snapshot_path = (
        DB_PATH.parent
        / "data"
        / "provenance"
        / "equipment-provenance-snapshot-v1-2026-09-14.json"
    )
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    payload["records"].pop()
    payload["recordCount"] = len(payload["records"])
    payload["recordsSha256"] = equipment_policy._records_hash(payload["records"])
    truncated = tmp_path / "truncated.json"
    truncated.write_text(json.dumps(payload), encoding="utf-8")

    with monkeypatch.context() as patch:
        patch.setattr(equipment_policy, "_VERIFIED_SOURCE_SNAPSHOT_PATH", truncated)
        _verified_source_snapshot_by_id.cache_clear()
        assert _verified_source_snapshot_by_id() == {}
    _verified_source_snapshot_by_id.cache_clear()


def test_missing_csv_and_invalid_snapshot_fail_candidate_enrichment_closed(monkeypatch, tmp_path):
    missing = tmp_path / "missing.csv"
    invalid_snapshot = tmp_path / "invalid-snapshot.json"
    invalid_snapshot.write_text("[]", encoding="utf-8")

    with monkeypatch.context() as patch:
        patch.setattr(db_queries, "CANONICAL_SOURCE_CATALOG_PATH", missing)
        patch.setattr(equipment_policy, "_VERIFIED_SOURCE_SNAPSHOT_PATH", invalid_snapshot)
        _verified_source_snapshot_by_id.cache_clear()
        clear_local_catalog_enrichment_cache()
        assert _enrich_candidates_with_local_catalog([
            {
                "id": "V24ROW-003510",
                "name_kr": "짐볼 라잉 브릿지 레그 컬",
                "name_en": "Stability Ball 라잉 브릿지 양측 Leg Curl",
                "equipment_req": "BODYWEIGHT",
                "primary_muscle": "hamstring",
            }
        ]) == []
    _verified_source_snapshot_by_id.cache_clear()
    clear_local_catalog_enrichment_cache()


def test_runtime_provenance_overrides_stale_bodyweight_before_selection():
    clear_local_catalog_enrichment_cache()
    candidates = _enrich_candidates_with_local_catalog([
        {
            "id": "V24ROW-003510",
            "name_kr": "짐볼 라잉 브릿지 레그 컬",
            "name_en": "Stability Ball 라잉 브릿지 양측 Leg Curl",
            "equipment_req": "BODYWEIGHT",
            "primary_muscle": "hamstring",
        }
    ])
    corrected = candidates[0]
    assert corrected["catalog_original_equipment_req"] == "BODYWEIGHT"
    assert corrected["equipment_req"] == "UNKNOWN"
    assert select_equipment_option(corrected["equipment_req"], available={"BODYWEIGHT"}) is None
    enriched_twice = _enrich_candidates_with_local_catalog(candidates)
    assert enriched_twice[0]["equipment_req"] == "UNKNOWN"
    assert enriched_twice[0]["catalog_original_equipment_req"] == "BODYWEIGHT"


def test_runtime_uses_verified_fallback_without_disabling_catalog_enrichment(monkeypatch, tmp_path):
    missing = tmp_path / "missing.csv"
    monkeypatch.setattr(db_queries, "CANONICAL_SOURCE_CATALOG_PATH", missing)
    clear_local_catalog_enrichment_cache()
    candidates = _enrich_candidates_with_local_catalog([
        {
            "id": "V24ROW-003510",
            "name_kr": "짐볼 라잉 브릿지 레그 컬",
            "name_en": "Stability Ball 라잉 브릿지 양측 Leg Curl",
            "equipment_req": "BODYWEIGHT",
            "primary_muscle": "hamstring",
        },
        {
            "id": "V24ROW-003053",
            "name_kr": "레그프레스 중앙 프레스/스쿼트",
            "name_en": "Leg Press Center Leg Press Squat",
            "equipment_req": "BODYWEIGHT",
            "primary_muscle": "quadriceps",
        },
    ])
    assert candidates[0]["equipment_req"] == "UNKNOWN"
    assert candidates[1]["catalog_exercise_id"] == "V24ROW-003053"
    assert candidates[1]["equipment_req"] == "MACHINE"


def test_source_loader_rejects_malformed_and_duplicate_rows(monkeypatch, tmp_path):
    malformed = tmp_path / "malformed.csv"
    malformed.write_text("v24_row_uid,name\nV24ROW-002628,ez\n", encoding="utf-8")
    monkeypatch.setattr(db_queries, "CANONICAL_SOURCE_CATALOG_PATH", malformed)
    clear_local_catalog_enrichment_cache()
    assert db_queries._load_source_equipment_by_id() is None

    invalid_utf8 = tmp_path / "invalid.csv"
    invalid_utf8.write_bytes(b"v24_row_uid,equipment\n\xff")
    monkeypatch.setattr(db_queries, "CANONICAL_SOURCE_CATALOG_PATH", invalid_utf8)
    clear_local_catalog_enrichment_cache()
    assert db_queries._load_source_equipment_by_id() is None

    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text(
        "v24_row_uid,equipment\nV24ROW-002628,EZ바\nV24ROW-002628,짐볼\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(db_queries, "CANONICAL_SOURCE_CATALOG_PATH", duplicate)
    clear_local_catalog_enrichment_cache()
    assert db_queries._load_source_equipment_by_id() is None


def test_source_loader_cache_key_includes_path_when_mtime_matches(monkeypatch, tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    canonical_source = DB_PATH.parent / "data" / "provenance" / "v25_canonical_filter_core.csv"
    first.write_bytes(canonical_source.read_bytes())
    second.write_bytes(canonical_source.read_bytes() + b"\n")
    shared_mtime_ns = first.stat().st_mtime_ns
    os.utime(second, ns=(shared_mtime_ns, shared_mtime_ns))
    monkeypatch.setattr(db_queries, "CANONICAL_SOURCE_CATALOG_PATH", first)
    clear_local_catalog_enrichment_cache()
    assert db_queries._load_source_equipment_by_id()["V24ROW-002628"]["equipment"] == "EZ바"
    monkeypatch.setattr(db_queries, "CANONICAL_SOURCE_CATALOG_PATH", second)
    assert db_queries._load_source_equipment_by_id() is None


def test_partial_source_uses_verified_fallback_for_missing_known_id(monkeypatch, tmp_path):
    header_only = tmp_path / "header-only.csv"
    header_only.write_text("v24_row_uid,equipment\n", encoding="utf-8")
    monkeypatch.setattr(db_queries, "CANONICAL_SOURCE_CATALOG_PATH", header_only)
    clear_local_catalog_enrichment_cache()
    candidate = _enrich_candidates_with_local_catalog([
        {
            "id": "V24ROW-002628",
            "name_kr": "EZ바 라잉 양손 라잉 트라이셉스 익스텐션",
            "name_en": "EZ Bar 라잉 양손 라잉 트라이셉스 익스텐션",
            "equipment_req": "BODYWEIGHT",
            "primary_muscle": "triceps",
        }
    ])[0]
    assert candidate["equipment_req"] == "UNKNOWN"
    assert candidate["catalog_exercise_id"] == "V24ROW-002628"


def test_missing_source_quarantines_band_stability_ball_conflict(monkeypatch, tmp_path):
    missing = tmp_path / "missing.csv"
    monkeypatch.setattr(db_queries, "CANONICAL_SOURCE_CATALOG_PATH", missing)
    clear_local_catalog_enrichment_cache()
    candidate = _enrich_candidates_with_local_catalog([
        {
            "id": "V24ROW-002201",
            "name_kr": "짐볼 밴드 저항 짐볼 백 익스텐션",
            "name_en": "Stability Ball 밴드 저항 짐볼 백 익스텐션",
            "equipment_req": "BAND",
            "primary_muscle": "lower-back",
        }
    ])[0]
    assert candidate["equipment_req"] == "UNKNOWN"
    assert select_equipment_option(candidate["equipment_req"], available={"BAND"}) is None


def test_builder_passes_raw_source_equipment_into_derived_options_without_execution():
    builder = Path(__file__).resolve().parents[1] / "scripts" / "build_constraints_db.py"
    tree = ast.parse(builder.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "equipment_option_profiles"
    ]
    assert len(calls) == 1
    source_argument = next(
        keyword.value
        for keyword in calls[0].keywords
        if keyword.arg == "source_equipment"
    )
    assert isinstance(source_argument, ast.Call)
    assert isinstance(source_argument.func, ast.Attribute)
    assert source_argument.func.attr == "get"


def test_superset_policy_allows_free_weights_but_not_two_generic_machines():
    assert superset_equipment_compatible("DUMBBELL", "BARBELL") is True
    assert superset_equipment_compatible("CABLE", "CABLE") is True
    assert superset_equipment_compatible("MACHINE", "MACHINE") is False
    assert superset_equipment_compatible("SMITH_MACHINE", "MACHINE") is False


def test_leg_press_catalog_equipment_is_corrected_without_mutating_source():
    assert canonical_equipment_req(
        "BODYWEIGHT",
        name_kr="레그프레스 중앙 프레스/스쿼트",
        name_en="Leg Press Center Leg Press Squat",
        detail_subpart="머신 프레스",
    ) == "MACHINE"
    assert canonical_equipment_req(
        "SMITH_MACHINE, MACHINE",
        name_kr="스미스 머신 중앙 프레스/스쿼트",
        name_en="Smith Machine Center Leg Press Squat",
        detail_subpart="머신 프레스",
    ) == "SMITH_MACHINE, MACHINE"


def test_named_band_exercise_is_not_treated_as_equipment_free_bodyweight():
    assert canonical_equipment_req(
        "BODYWEIGHT",
        name_kr="밴드 스탠딩 로프 푸시다운",
        name_en="Band Standing Rope Triceps Pushdown",
        detail_subpart="푸시다운/외측두",
    ) == "BAND"
    assert select_equipment_option(
        "BAND",
        available={"BODYWEIGHT", "DUMBBELL"},
    ) is None
    assert select_equipment_option("BAND", available={"BAND"}) == "BAND"


def test_assisted_pullup_is_not_treated_as_equipment_free_bodyweight():
    assert canonical_equipment_req(
        "BODYWEIGHT",
        name_kr="기본 어시스트 풀업",
        name_en="Standard Assisted Pull-up",
        detail_subpart="수직 당기기",
    ) == "MACHINE"
    assert select_equipment_option(
        "MACHINE",
        available={"BODYWEIGHT", "DUMBBELL"},
    ) is None


def test_candidate_filter_uses_corrected_leg_press_equipment_and_movement():
    clear_local_catalog_enrichment_cache()
    engine = create_engine(f"sqlite:///{DB_PATH}")
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        bodyweight_only = get_candidate_exercises(
            session,
            ["quadriceps"],
            [],
            [],
            ["BODYWEIGHT"],
        )
        machine_available = get_candidate_exercises(
            session,
            ["quadriceps"],
            [],
            [],
            ["MACHINE"],
        )
    finally:
        session.close()
        engine.dispose()

    assert all(item["id"] != "V24ROW-003053" for item in bodyweight_only)
    corrected = next(item for item in machine_available if item["id"] == "V24ROW-003053")
    assert corrected["equipment_req"] == "MACHINE"
    assert corrected["selected_equipment_code"] == "MACHINE"
    assert corrected["movement_type"] == "COMPOUND"


def test_home_equipment_filter_excludes_assisted_pullup():
    clear_local_catalog_enrichment_cache()
    engine = create_engine(f"sqlite:///{DB_PATH}")
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        candidates = get_candidate_exercises(
            session,
            ["upper-back"],
            [],
            [],
            ["BODYWEIGHT", "DUMBBELL"],
        )
    finally:
        session.close()
        engine.dispose()

    assert all(item["id"] != "V24ROW-005709" for item in candidates)


def test_equipment_option_table_covers_catalog_and_keeps_smith_required():
    with sqlite3.connect(DB_PATH) as conn:
        exercise_count = conn.execute("SELECT COUNT(*) FROM exercise_tier").fetchone()[0]
        coverage = conn.execute(
            "SELECT COUNT(DISTINCT exercise_id) FROM exercise_equipment_option_v2"
        ).fetchone()[0]
        smith_count = conn.execute("""
            SELECT COUNT(*)
            FROM exercise_tier
            WHERE UPPER(COALESCE(equipment_req, '')) LIKE '%SMITH_MACHINE%'
        """).fetchone()[0]
        smith_option_count = conn.execute("""
            SELECT COUNT(*)
            FROM exercise_equipment_option_v2 eo
            JOIN exercise_tier et ON et.id = eo.exercise_id
            WHERE UPPER(COALESCE(et.equipment_req, '')) LIKE '%SMITH_MACHINE%'
              AND eo.equipment_code = 'SMITH_MACHINE'
        """).fetchone()[0]
        smith_other_count = conn.execute("""
            SELECT COUNT(*)
            FROM exercise_equipment_option_v2 eo
            JOIN exercise_tier et ON et.id = eo.exercise_id
            WHERE UPPER(COALESCE(et.equipment_req, '')) LIKE '%SMITH_MACHINE%'
              AND eo.equipment_code <> 'SMITH_MACHINE'
        """).fetchone()[0]

    assert coverage == exercise_count == 24653
    assert smith_option_count == smith_count == 553
    assert smith_other_count == 0


def test_band_pushdown_uses_band_equipment_truth_in_runtime_catalog():
    clear_local_catalog_enrichment_cache()
    engine = create_engine(f"sqlite:///{DB_PATH}")
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        home_without_band = get_candidate_exercises(
            session,
            ["triceps"],
            [],
            [],
            ["BODYWEIGHT", "DUMBBELL"],
        )
        band_available = get_candidate_exercises(
            session,
            ["triceps"],
            [],
            [],
            ["BAND"],
        )
    finally:
        session.close()
        engine.dispose()

    assert all(item["id"] != "V24ROW-002726" for item in home_without_band)
    corrected = next(item for item in band_available if item["id"] == "V24ROW-002726")
    assert corrected["equipment_req"] == "BAND"
    assert corrected["selected_equipment_code"] == "BAND"

"""Pinned collaboration-seed boundary audit, not a live Java/FE/LLM test.

Name matches are candidates, NEVER approved identity mappings. Existing source,
catalog and golden fixtures are read-only; SQL seed INSERT runs only in memory.
Run this file with --inventory to emit reproducible matching evidence as JSON.
"""
import hashlib
import json
import re
import sqlite3
import subprocess
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import pytest


CORE = Path(__file__).resolve().parents[1]
ROOT = CORE.parents[1]
BE_SHA = "1490c91a08b674291754ced722af8dd979cf116c"
SEED_PATH = "src/main/resources/db/migration/V6__seed_exercise_catalog.sql"
JAVA = "src/main/java/com/fitcore/api/"
COLUMNS = (
    "id", "name_kr", "name_en", "primary_muscle", "secondary_muscle",
    "equipment_req", "difficulty_tier", "efficiency_tier", "pain_triggers",
    "movement_type", "substitute_exercise_ids",
)
SAMPLE_IDS = (7, 14, 26, 30, 40, 75, 98, 106, 115, 151)


@lru_cache(maxsize=None)
def be_source(path):
    # Pin the same BE bytes locally so a standalone AI clone runs this oracle.
    baseline = CORE / 'tests/fixtures/backend_boundary_v1'
    manifest = json.loads((baseline / 'manifest.json').read_text())
    assert manifest['commit'] == BE_SHA
    data = (baseline / path).read_bytes()
    assert hashlib.sha256(data).hexdigest() == manifest['sha256'][path]
    return data.decode('utf-8')


def seed_rows():
    sql = be_source(SEED_PATH)
    insert = sql[sql.index("INSERT INTO `exercise_tier`"):].strip()
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        defs = ", ".join(f"{col} {'INTEGER' if col in {'id', 'difficulty_tier', 'efficiency_tier'} else 'TEXT'}" for col in COLUMNS)
        conn.execute(f"CREATE TABLE exercise_tier ({defs})")
        # execute, not executescript: trailing unrelated statements are rejected.
        conn.execute(insert)
        return [dict(r) for r in conn.execute("SELECT * FROM exercise_tier ORDER BY id")]
    finally:
        conn.close()


def catalog_rows():
    conn = sqlite3.connect((CORE / "fit_core.sqlite").as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(
            "SELECT id,name_kr,name_en,primary_muscle,equipment_req,movement_type "
            "FROM exercise_tier ORDER BY id"
        )]
    finally:
        conn.close()


def normalized_name(value):
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return "".join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith("P"))


def tokens(value):
    return {v.strip().upper() for v in str(value or "").split(",") if v.strip()}


def classify(seed, catalog):
    indexes = {kind: {lang: defaultdict(list) for lang in ("name_kr", "name_en")}
               for kind in ("exact", "normalized")}
    for row in catalog:
        for lang in ("name_kr", "name_en"):
            if row[lang]:
                indexes["exact"][lang][row[lang]].append(row)
                indexes["normalized"][lang][normalized_name(row[lang])].append(row)
    results = []
    for source in seed:
        matches = {}
        category = "unmatched"
        for kind in ("exact", "normalized"):
            for lang in ("name_kr", "name_en"):
                key = source[lang] if kind == "exact" else normalized_name(source[lang])
                if key:
                    for row in indexes[kind][lang].get(key, []):
                        matches[row["id"]] = row
            if matches:
                category = kind
                break
        candidates = []
        for row in sorted(matches.values(), key=lambda r: r["id"]):
            conflicts = []
            if source["primary_muscle"] != row["primary_muscle"]:
                conflicts.append("source_primary_muscle_diff")
            if not tokens(source["equipment_req"]) & tokens(row["equipment_req"]):
                conflicts.append("equipment_no_overlap")
            if source["movement_type"] != row["movement_type"]:
                conflicts.append("movement_type_diff")
            candidates.append({**row, "conflicts": conflicts})
        results.append({"numericId": source["id"], "name": source["name_kr"],
                        "english": source["name_en"], "category": category,
                        "ambiguous": len(candidates) > 1, "approved": False,
                        "candidates": candidates})
    return results


def inventory():
    rows = classify(seed_rows(), catalog_rows())
    counts = Counter(row["category"] for row in rows)
    return {"backendCommit": BE_SHA, "seedPath": SEED_PATH,
            "seedSha256": hashlib.sha256(be_source(SEED_PATH).encode()).hexdigest(),
            "catalogSha256": hashlib.sha256((CORE / "fit_core.sqlite").read_bytes()).hexdigest(),
            "normalization": "same-language NFKC + casefold - whitespace - Unicode punctuation; no translation/fuzzy matching",
            "denominator": len(rows), "catalogRows": len(catalog_rows()),
            "counts": dict(counts),
            "ambiguous": sum(r["ambiguous"] for r in rows),
            "approvedMappings": 0, "rows": rows}


@pytest.fixture(scope="module")
def readonly_core():
    import engines.db_queries as dbq

    def connect():
        conn = sqlite3.connect((CORE / "fit_core.sqlite").as_uri() + "?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(dbq, "_connect_local_catalog_db", connect)
        patch.setattr(dbq, "SQLITE_DB_PATH", CORE / "fit_core.sqlite")
        dbq.clear_local_catalog_enrichment_cache()
        try:
            yield dbq
        finally:
            dbq.clear_local_catalog_enrichment_cache()


def public_request(source):
    return {"targetSplitLabel": "custom", "targetMuscles": [source["primary_muscle"]],
            "readinessLevel": "normal", "timeAvailableMin": 60,
            "currentPainAreas": [], "currentDoms": [], "unavailableEquipment": [],
            "goal": "hypertrophy", "userNote": ""}


def adapter_projection(public, numeric_id):
    """Test-side mirror of pinned RoutineService.mapToAiRequest, not Java execution."""
    levels = {"mild": 1, "moderate": 2, "severe": 3}
    doms = {}
    for item in public["currentDoms"]:
        key = item["bodyPart"].strip()
        doms[key] = max(doms.get(key, 0), levels.get(item["level"].lower(), 1))
    return {"userId": "synthetic-numeric-boundary", "targetSplitLabel": public["targetSplitLabel"],
            "targetMuscles": [v.strip() for v in public["targetMuscles"]],
            "readinessLevel": public["readinessLevel"], "timeAvailableMin": public["timeAvailableMin"],
            "painAreas": [{"bodyPart": v} for v in public["currentPainAreas"]],
            "domsData": doms, "equipment": public["unavailableEquipment"],
            "goal": public["goal"], "userNote": public["userNote"],
            "preferredExerciseIds": [str(numeric_id)], "unpreferredExerciseIds": []}


def core_response(candidate, modality="working"):
    from engines.schemas import RoutineBlock, RoutineDraftResponse, SetPrescription, TrainingMethodInstruction
    extras = {
        "working": {}, "timed": {"target_duration_sec": 30},
        "distance": {"target_distance_m": 20},
        "interval": {"target_duration_sec": 60, "interval_work_sec": 20,
                     "interval_recovery_sec": 10, "interval_rounds": 2},
    }[modality]
    block = RoutineBlock(order=1, exercise_id=str(candidate["id"]),
                         exercise_name=candidate["name_kr"],
                         primary_muscles=[candidate["primary_muscle"]],
                         default_rest_sec=90, exercise_rationale="Synthetic boundary only",
                         execution_group_id="pair-1", execution_group_type="superset",
                         execution_group_size=2, execution_group_position=1, group_rest_sec=90,
                         training_method=TrainingMethodInstruction(code="pausedReps", label_ko="pause",
                             purpose_ko="serialization", instruction_ko="synthetic", pause_sec=1),
                         prescription=[SetPrescription(set_index=1, set_type=modality,
                             target_reps=1 if modality != "working" else 8,
                             target_weight_kg=None, target_rir=3, target_rest_sec=90, **extras)])
    return RoutineDraftResponse(generation_status="success", status_reason_code="none",
        is_fallback=False, total_estimated_time=5, summary_title="Boundary fixture",
        rationale_summary=["Not a recommended workout"], routine_blocks=[block])


def java_fields(path):
    return set(re.findall(r"^\s*private\s+[^;=]+?\s+(\w+)\s*;", be_source(JAVA + path), re.M))


def be_response_projection(payload):
    """Lossy declared-field projection only, NOT a Jackson runtime emulator."""
    top = java_fields("infrastructure/ai/dto/AiRoutineResponse.java")
    block = java_fields("domain/routine/dto/RoutineBlock.java")
    prescription = java_fields("domain/routine/dto/Prescription.java")
    result = {k: v for k, v in payload.items() if k in top}
    result["routineBlocks"] = [{k: v for k, v in b.items() if k in block} for b in payload["routineBlocks"]]
    for b in result["routineBlocks"]:
        b["prescription"] = [{k: v for k, v in p.items() if k in prescription} for p in b["prescription"]]
    return result


def test_seed_parser_and_partition():
    source = seed_rows()
    assert [r["id"] for r in source] == list(range(1, 175))
    assert source[9]["name_en"] == "Farmer's Walk"
    rows = classify(source, catalog_rows())
    assert len(rows) == 174
    assert all(r["approved"] is False for r in rows)
    assert all((r["category"] == "unmatched") == (not r["candidates"]) for r in rows)


def test_normalization_does_not_translate_or_drop_equipment():
    assert normalized_name("Barbell Bench-Press") == normalized_name("barbell bench press")
    assert normalized_name("Barbell Bench Press") != normalized_name("Dumbbell Bench Press")
    assert normalized_name("Bench Press") != normalized_name("Incline Bench Press")


def test_adapter_projection_preserves_pain_doms_and_numeric_preferences():
    from engines.schemas import RoutineRequest
    source = next(r for r in seed_rows() if r["id"] == 30)
    public = public_request(source)
    public.update(currentPainAreas=["front-deltoids"],
                  currentDoms=[{"bodyPart": "chest", "level": "moderate"}],
                  unavailableEquipment=["DUMBBELL"])
    req = RoutineRequest.model_validate(adapter_projection(public, 30))
    assert req.pain_areas[0].body_part == "front-deltoids"
    assert req.doms_data == {"chest": 2}
    assert req.equipment == ["DUMBBELL"]
    assert req.preferred_exercise_ids == ["30"]
    service = be_source(JAVA + "domain/routine/service/RoutineService.java")
    assert '.map(area -> Map.of("bodyPart", area))' in service
    assert ".equipment(request.getUnavailableEquipment())" in service


@pytest.mark.parametrize("numeric_id", SAMPLE_IDS)
def test_numeric_id_core_roundtrip_and_history_materialization(readonly_core, numeric_id):
    from engines.schemas import RoutineDraftResponse, RoutineRequest
    source = next(r for r in seed_rows() if r["id"] == numeric_id)
    req = RoutineRequest.model_validate(adapter_projection(public_request(source), numeric_id))
    candidate = readonly_core._enrich_candidates_with_local_catalog([source])[0]
    assert candidate["id"] == numeric_id
    draft = core_response(candidate)
    payload = json.loads(draft.model_dump_json(by_alias=True))
    restored = RoutineDraftResponse.model_validate(payload)
    assert restored.routine_blocks[0].exercise_id == req.preferred_exercise_ids[0]

    class QueryResult:
        def __init__(self, rows): self.rows = rows
        def fetchall(self): return self.rows

    class RecordedHistoryDb:
        def __init__(self): self.calls = []
        def execute(self, query, params):
            self.calls.append((str(query), params))
            if len(self.calls) == 1:
                return QueryResult([("synthetic-session",)])
            return QueryResult([(payload["routineBlocks"][0]["exerciseId"], source["name_kr"],
                                 source["primary_muscle"], 20.0, 8, 3, None, False)])

    history_db = RecordedHistoryDb()
    history = readonly_core.get_recent_sets(history_db, req.user_id)
    assert history[0].exercise_id == str(numeric_id)
    assert history_db.calls[0][1]["uid"] == req.user_id
    assert "LEFT JOIN exercise_tier" in history_db.calls[1][0]


def test_ambiguous_name_requires_explicit_identity_mapping(readonly_core):
    audit = classify(seed_rows(), catalog_rows())
    ambiguous_ids = {r["numericId"] for r in audit if r["ambiguous"]}
    assert ambiguous_ids, "Pinned input changed: expected ambiguity control"
    sources = [r for r in seed_rows() if r["id"] in ambiguous_ids]
    # 2026-10-08 approved 151 explicitly; names alone must still never resolve it.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(readonly_core, "load_identity_edges", lambda: [])
        candidates = readonly_core._enrich_candidates_with_local_catalog(sources)
    silently_resolved = {str(r["id"]): r["catalog_exercise_id"]
                         for r in candidates if r.get("catalog_exercise_id")}
    assert not silently_resolved, f"Ambiguous names resolved without reviewed ID bridge: {silently_resolved}"
    candidates = readonly_core._enrich_candidates_with_local_catalog(sources)
    explicitly_resolved = {str(r["id"]): r["catalog_exercise_id"]
                           for r in candidates if r.get("catalog_exercise_id")}
    assert explicitly_resolved == {"151": "V24ROW-004710"}
    approval, = [e for e in readonly_core.load_identity_edges()
                 if e["sourceId"] == "151" and e["status"] == "approved"]
    assert approval["catalogId"] == "V24ROW-004710"
    assert approval["review"]["reviewedBy"] == "user"
    assert approval["review"]["reviewedAt"] == "2026-10-08"


@pytest.mark.parametrize("target", ["chest", "upper-back", "quadriceps"])
def test_numeric_seed_candidate_query_boundary(readonly_core, target):
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session
    from engines.schemas import RoutineRequest

    # Disposable SQL database: exercise selection is real CORE code, no MySQL access.
    engine = create_engine("sqlite://")
    try:
        with engine.begin() as conn:
            defs = ", ".join(f"{col} {'INTEGER' if col in {'id', 'difficulty_tier', 'efficiency_tier'} else 'TEXT'}" for col in COLUMNS)
            conn.exec_driver_sql(f"CREATE TABLE exercise_tier ({defs})")
            conn.execute(text(f"INSERT INTO exercise_tier VALUES ({', '.join(':'+c for c in COLUMNS)})"), seed_rows())
        request = RoutineRequest.model_validate(adapter_projection(
            public_request({"primary_muscle": target}), 30))
        with Session(engine) as session:
            candidates = readonly_core.get_candidate_exercises(
                session, request.target_muscles, request.equipment, request.pain_areas)
        print("NUMERIC_CANDIDATES", target, [(str(r["id"]), r.get("catalog_exercise_id")) for r in candidates])
        assert candidates, f"Unrestricted numeric seed has no eligible {target} candidates; bridge not ready"
        assert all(str(r["id"]).isdigit() for r in candidates)
        for row in candidates:
            payload = json.loads(core_response(row).model_dump_json(by_alias=True))
            assert payload["routineBlocks"][0]["exerciseId"] == str(row["id"])
    finally:
        engine.dispose()


@pytest.mark.parametrize("modality", ["timed", "distance", "interval"])
def test_extended_prescription_survives_declared_be_dto(modality):
    source = next(r for r in seed_rows() if r["id"] == 26)
    payload = json.loads(core_response(source, modality).model_dump_json(by_alias=True))
    expected = payload["routineBlocks"][0]["prescription"][0]
    observed = be_response_projection(payload)["routineBlocks"][0]["prescription"][0]
    assert observed == expected, f"BE declared DTO loses fields: {sorted(expected.keys() - observed.keys())}"


def test_execution_group_and_method_survive_declared_be_dto():
    source = next(r for r in seed_rows() if r["id"] == 30)
    payload = json.loads(core_response(source).model_dump_json(by_alias=True))
    projected = be_response_projection(payload)["routineBlocks"][0]
    required = {"executionGroupId", "executionGroupType", "executionGroupSize", "groupRestSec", "trainingMethod"}
    assert required <= projected.keys(), f"BE declared DTO loses fields: {sorted(required - projected.keys())}"


if __name__ == "__main__":
    import sys
    if sys.argv[1:] != ["--inventory"]:
        raise SystemExit("Use pytest, or --inventory for JSON matching evidence")
    print(json.dumps(inventory(), ensure_ascii=False, indent=2))

import json
from pathlib import Path
import sqlite3

import pytest

from engines import db_queries
from engines.candidate_ranker import candidate_selection_family_key
from engines.focus_catalog_admission import REVIEWED_IDS, admission_patch
from engines.routine_blueprint import build_blueprint_candidate_bundles, build_routine_blueprint
from engines.routine_selection import adapt_slot_selection_output
from engines.schemas import LLMRoutineSelectionOutput
from engines.specialization_policy import focus_variant_family_key


ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "data/provenance/focus-basic-admission-review-v2-2026-10-01.json"
SOURCE_PATH = ROOT / "data/provenance/v25_canonical_filter_core.csv"


def _review_payload():
    return json.loads(REVIEW_PATH.read_text(encoding="utf-8"))


def _review_row(exercise_id):
    return next(row for row in _review_payload()["reviews"] if row["catalogExerciseId"] == exercise_id)


def _catalog_candidates(ids=REVIEWED_IDS):
    with sqlite3.connect(f"file:{ROOT / 'fit_core.sqlite'}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        return [
            dict(connection.execute("SELECT * FROM exercise_tier WHERE id=?", (exercise_id,)).fetchone())
            for exercise_id in ids
        ]


def _candidate(exercise_id, name, primary, equipment="MACHINE", movement="ISOLATION", **extra):
    return {
        "id": exercise_id,
        "name_kr": name,
        "primary_muscle": primary,
        "equipment_req": equipment,
        "movement_type": movement,
        "main_routine_default_allowed": 1,
        "exercise_familiarity_tier": "common",
        "failure_penalty": "low",
        "injury_caution_level": "low",
        "score": 100,
        **extra,
    }


def _select_all_available(blueprint, candidates):
    bundles = build_blueprint_candidate_bundles(blueprint, candidates)
    selections = [
        {"slotId": bundle["slotId"], "exerciseId": bundle["candidates"][0]["exerciseId"]}
        for bundle in bundles
        if bundle["candidates"] and bundle["slotId"] != "flex"
    ]
    result = adapt_slot_selection_output(
        LLMRoutineSelectionOutput.model_validate({"selections": selections}),
        blueprint=blueprint,
        slot_candidate_bundles=bundles,
        ranked_candidates=candidates,
    )
    return result, bundles


def test_exact_reviewed_population_requires_canonical_and_effective_identity_match():
    enriched = db_queries._enrich_candidates_with_local_catalog(_catalog_candidates())

    promoted = {
        row["id"]
        for row in enriched
        if row.get("core_admission_review_version") == "focus-basic-admission-v2-2026-10-01"
    }
    assert promoted == REVIEWED_IDS
    for row in enriched:
        review = _review_row(row["id"])
        assert row["main_routine_default_allowed"] == 1
        assert row["beginner_default_allowed"] == 0
        assert {key: row.get(key) for key in review["effectiveIdentity"]} == review["effectiveIdentity"]


@pytest.mark.parametrize("tamper_target", ["manifest", "source"])
def test_admission_fails_closed_when_immutable_admission_inputs_are_tampered(
    monkeypatch, tmp_path, tamper_target
):
    from engines import focus_catalog_admission as policy

    review = _review_row("V24ROW-006682")
    if tamper_target == "manifest":
        tampered_manifest = tmp_path / "review.json"
        tampered_manifest.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(policy, "REVIEW_PATH", tampered_manifest)
        source_path = SOURCE_PATH
    else:
        source_path = tmp_path / "source.csv"
        source_path.write_bytes(SOURCE_PATH.read_bytes() + b"\n")
    policy._load.cache_clear()

    assert policy.admission_patch(
        review["catalogExerciseId"],
        review["catalogIdentity"],
        {**review["effectiveIdentity"], "beginner_default_allowed": 0, "main_routine_default_allowed": 0},
        source_path,
    ) == {}


@pytest.mark.parametrize("field", ["name_kr", "equipment_req", "movement_type"])
def test_admission_rejects_catalog_or_effective_identity_substitution(field):
    review = _review_row("V24ROW-006024")
    catalog = dict(review["catalogIdentity"])
    effective = {
        **review["effectiveIdentity"],
        "beginner_default_allowed": 0,
        "main_routine_default_allowed": 0,
    }
    if field in catalog:
        catalog[field] = "tampered"
    else:
        effective[field] = "tampered"

    assert admission_patch(review["catalogExerciseId"], catalog, effective, SOURCE_PATH) == {}


def test_similarly_named_unmapped_row_cannot_inherit_reviewed_admission():
    review = _review_row("V24ROW-006121")
    effective = {
        **review["effectiveIdentity"],
        "beginner_default_allowed": 0,
        "main_routine_default_allowed": 0,
    }

    assert admission_patch("V24ROW-006121-copy", review["catalogIdentity"], effective, SOURCE_PATH) == {}


@pytest.mark.parametrize(
    ("focus", "primary", "name", "base_family"),
    [
        ("midBackEmphasis", "upper-back", "머신 로우", "horizontal_row"),
        ("sideDeltoidEmphasis", "side-deltoids", "머신 레터럴 레이즈", "lateral_raise"),
        ("trapeziusEmphasis", "trapezius", "머신 슈러그", "shrug"),
    ],
)
def test_focus_variant_cap_counts_equipment_setups_not_same_equipment_clones(
    focus, primary, name, base_family
):
    machine = _candidate("machine", name, primary, "MACHINE", specialization_focus_codes=focus)
    machine_clone = _candidate("machine_clone", name + " 변형", primary, "MACHINE", specialization_focus_codes=focus)
    cable = _candidate("cable", name, primary, "CABLE", specialization_focus_codes=focus)

    assert focus_variant_family_key(machine, focus, base_family) == focus_variant_family_key(
        machine_clone, focus, base_family
    )
    assert focus_variant_family_key(machine, focus, base_family) != focus_variant_family_key(
        cable, focus, base_family
    )


@pytest.mark.parametrize("readiness", ["normal", "low"])
def test_focus_equipment_variants_do_not_disable_default_duplicate_guard(readiness):
    blueprint = build_routine_blueprint(
        target_split_label="pull",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="advanced",
        readiness_level=readiness,
        time_available_min=60,
        specialization_focus="midBackEmphasis",
        method_mode="off",
    )
    rows = [
        _candidate("machine_row", "머신 로우", "upper-back", "MACHINE", "COMPOUND"),
        _candidate("machine_row_clone", "머신 로우 변형", "upper-back", "MACHINE", "COMPOUND"),
        _candidate("cable_row", "케이블 로우", "upper-back", "CABLE", "COMPOUND"),
        _candidate("pulldown", "랫풀다운", "upper-back", "MACHINE", "COMPOUND"),
    ]
    _, bundles = _select_all_available(blueprint, rows)
    row_ids = {
        candidate["exerciseId"]
        for bundle in bundles
        for candidate in bundle["candidates"]
        if bundle["slotId"] != "flex"
    }

    assert not {"machine_row", "machine_row_clone"} <= row_ids
    if readiness == "low":
        assert "specialization_focus" in {bundle["slotId"] for bundle in bundles}
        assert blueprint.training_method is None


def test_advanced_full_body_vertical_pull_preserves_mandatory_roles_and_rejects_restricted_row():
    blueprint = build_routine_blueprint(
        target_split_label="full_body",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="advanced",
        readiness_level="normal",
        time_available_min=60,
        specialization_focus=None,
        method_mode="off",
    )
    rows = [
        _candidate("squat", "핵 스쿼트", "quadriceps", movement="COMPOUND"),
        _candidate("press", "덤벨 플랫 체스트 프레스", "chest", "DUMBBELL", "COMPOUND"),
        _candidate("row", "덤벨 로우", "upper-back", "DUMBBELL", "COMPOUND"),
        _candidate("restricted_pull", "제한 랫풀다운", "upper-back", movement="COMPOUND", main_routine_default_allowed=0, score=500),
        _candidate("pull", "랫풀다운", "upper-back", movement="COMPOUND"),
        _candidate("legcurl", "머신 레그 컬", "hamstring"),
        _candidate("shoulder", "덤벨 숄더 프레스", "front-deltoids", "DUMBBELL", "COMPOUND"),
    ]
    result, bundles = _select_all_available(blueprint, rows)
    by_slot = {bundle["slotId"]: bundle for bundle in bundles}
    selected = {plan.exercise_id for plan in result.exercises}

    assert {"full_lower_main", "full_push_main", "full_pull_main", "full_vertical_pull"} <= set(by_slot)
    assert all(by_slot[slot]["candidates"] for slot in ("full_lower_main", "full_push_main", "full_pull_main", "full_vertical_pull"))
    assert {"squat", "press", "row", "pull"} <= selected
    assert "restricted_pull" not in selected
    assert "restricted_pull" not in {
        candidate["exerciseId"]
        for bundle in bundles
        for candidate in bundle["candidates"]
    }

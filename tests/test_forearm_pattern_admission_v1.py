import sqlite3
import json
import socket
from collections import Counter
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from engines import db_queries
from engines.candidate_ranker import (
    _ensure_specialization_focus_candidate,
    candidate_selection_family_key,
    score_candidate_exercises,
)
from engines.schemas import PainAreaEntry
from engines.routine_blueprint import (
    build_blueprint_candidate_bundles,
    build_routine_blueprint,
)


ROTATION_ID = "V24ROW-004724"
_ARM_CANDIDATE_CACHE = None


def _session():
    engine = create_engine(
        "sqlite://",
        creator=lambda: sqlite3.connect(
            f"file:{db_queries.SQLITE_DB_PATH}?mode=ro", uri=True
        ),
    )
    return engine, Session(engine)


def _arm_candidates(*, equipment=None):
    global _ARM_CANDIDATE_CACHE
    if equipment is None and _ARM_CANDIDATE_CACHE is not None:
        return [dict(row) for row in _ARM_CANDIDATE_CACHE]
    engine, session = _session()
    try:
        rows = db_queries.get_candidate_exercises(
            session,
            ["forearm", "biceps", "triceps"],
            [],
            [],
            available_equipment=equipment
            or ["BARBELL", "DUMBBELL", "MACHINE", "CABLE"],
        )
        if equipment is None:
            _ARM_CANDIDATE_CACHE = tuple(dict(row) for row in rows)
        return rows
    finally:
        session.close()
        engine.dispose()


def _rank(rows, **overrides):
    options = {
        "available_equipment": ["BARBELL", "DUMBBELL", "MACHINE", "CABLE"],
        "blocked_equipment": [],
        "pain_areas": [],
        "doms_db": {},
        "readiness_level": "normal",
        "goal": "hypertrophy",
        "experience_level": "advanced",
        "training_style": "gym",
        "target_split_label": "arm",
        "specialization_focus": "forearmEmphasis",
        "time_available_min": 60,
        "top_n": 12,
    }
    options.update(overrides)
    return score_candidate_exercises(
        rows, ["forearm", "biceps", "triceps"], **options
    )


def _candidate(exercise_id, name, primary, score, equipment="DUMBBELL"):
    return {
        "id": exercise_id,
        "name_kr": name,
        "name_en": name,
        "primary_muscle": primary,
        "equipment_req": equipment,
        "movement_type": "ISOLATION",
        "main_routine_default_allowed": 1,
        "mainstream_gym_score": 90,
        "exercise_familiarity_tier": "common",
        "novelty_level": "low",
        "score": score,
        "efficiency_tier": 3,
    }


def test_identity_locked_rotation_overlay_preserves_catalog_quality_and_prescription():
    rotation = next(row for row in _arm_candidates() if row["id"] == ROTATION_ID)

    assert rotation["name_kr"] == "덤벨 전완 회내/회외"
    assert rotation["primary_muscle"] == "forearm"
    assert rotation["catalog_original_primary_muscle"] == "upper-back"
    assert rotation["equipment_req"] == "DUMBBELL"
    assert rotation["movement_type"] == "ISOLATION"
    assert rotation["prescription_mode"] == "reps"
    assert rotation["mainstream_gym_score"] == 57
    assert rotation["exercise_familiarity_tier"] == "specialized"
    assert rotation["commonness_level"] == "specialized"
    assert rotation["novelty_level"] == "medium"
    assert rotation["beginner_default_allowed"] == 0
    assert rotation["core_admission_scope"] == "forearmEmphasis"
    assert rotation["core_admission_review_version"] == db_queries.FOREARM_PATTERN_REVIEW_VERSION


def test_rotation_review_fails_closed_for_tampered_review_or_source(monkeypatch, tmp_path):
    tampered_review = tmp_path / "review.json"
    tampered_review.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(db_queries, "FOREARM_PATTERN_REVIEW_PATH", tampered_review)
    db_queries.clear_local_catalog_enrichment_cache()
    assert db_queries._load_forearm_pattern_admission_reviews() == {}

    monkeypatch.undo()
    tampered_source = tmp_path / "catalog.csv"
    tampered_source.write_bytes(
        Path(db_queries.CANONICAL_SOURCE_CATALOG_PATH).read_bytes() + b"\n"
    )
    monkeypatch.setattr(db_queries, "CANONICAL_SOURCE_CATALOG_PATH", tampered_source)
    db_queries.clear_local_catalog_enrichment_cache()
    assert db_queries._load_forearm_pattern_admission_reviews() == {}


def test_rotation_review_rejects_catalog_or_effective_identity_drift():
    review = db_queries._load_forearm_pattern_admission_reviews()[ROTATION_ID]
    catalog = dict(review["catalogIdentity"])
    effective = dict(review["effectiveIdentityBeforeReview"])

    assert db_queries._forearm_pattern_admission_patch(
        ROTATION_ID, catalog, effective
    )["primary_muscle"] == "forearm"
    for field in ("name_kr", "equipment_req", "prescription_mode"):
        changed = dict(catalog)
        changed[field] = "tampered"
        assert db_queries._forearm_pattern_admission_patch(
            ROTATION_ID, changed, effective
        ) == {}
    for field in (
        "primary_muscle",
        "mainstream_gym_score",
        "main_routine_default_allowed",
    ):
        changed = dict(effective)
        changed[field] = "tampered"
        assert db_queries._forearm_pattern_admission_patch(
            ROTATION_ID, catalog, changed
        ) == {}


def test_unrelated_candidate_never_loads_forearm_review(monkeypatch):
    def unexpected_loader():
        raise AssertionError("unrelated candidates must bypass forearm review I/O")

    monkeypatch.setattr(
        db_queries, "_load_forearm_pattern_admission_reviews", unexpected_loader
    )
    assert db_queries._forearm_pattern_admission_patch(
        "V24ROW-unrelated", {}, {}
    ) == {}


def test_actual_073_pool_exposes_three_patterns_without_reclassifying_hammer_curl():
    ranked = _rank(_arm_candidates())
    by_id = {row["id"]: row for row in ranked}

    assert ROTATION_ID in by_id
    rotation = by_id[ROTATION_ID]
    boost_codes = {
        item.get("rule_code") for item in rotation.get("score_boosts", [])
    }
    assert "mainstream_gym_priority" not in boost_codes
    assert "common_default_gym_familiarity_priority" not in boost_codes
    assert not any(
        "보편적으로 수행" in reason
        for reason in rotation.get("score_display_reasons", [])
    )
    assert any("Reverse Wrist Curl" in str(row.get("name_en")) for row in ranked)
    assert any(
        "Wrist Curl" in str(row.get("name_en"))
        and "Reverse" not in str(row.get("name_en"))
        for row in ranked
    )
    assert {"forearm", "biceps", "triceps"} <= {
        row.get("primary_muscle") for row in ranked
    }
    assert all(
        row.get("primary_muscle") == "biceps"
        for row in ranked
        if "해머 컬" in str(row.get("name_kr"))
    )


def test_rotation_equipment_variants_collapse_but_distinct_forearm_patterns_do_not():
    wrist_machine = _candidate("wrist-machine", "머신 리스트 컬", "forearm", 130, "MACHINE")
    wrist_barbell = _candidate("wrist-barbell", "바벨 리스트 컬", "forearm", 125, "BARBELL")
    reverse_wrist = _candidate("reverse-wrist", "덤벨 리버스 리스트 컬", "forearm", 115)
    rotation_dumbbell = _candidate("rotation-db", "덤벨 전완 회내/회외", "forearm", 105)
    rotation_cable = _candidate("rotation-cable", "케이블 전완 회내/회외", "forearm", 100, "CABLE")
    preacher = _candidate("preacher", "머신 프리처 컬", "biceps", 120, "MACHINE")
    pushdown = _candidate("pushdown", "머신 푸시다운", "triceps", 110, "MACHINE")

    result = _ensure_specialization_focus_candidate(
        selected=[wrist_machine, wrist_barbell, preacher, pushdown],
        sorted_candidates=[
            wrist_machine,
            wrist_barbell,
            preacher,
            reverse_wrist,
            pushdown,
            rotation_dumbbell,
            rotation_cable,
        ],
        specialization_focus="forearmEmphasis",
        adjunct_options=[],
        target_muscles=["forearm", "biceps", "triceps"],
        top_n=5,
    )

    ids = {row["id"] for row in result}
    assert {"wrist-machine", "reverse-wrist", "rotation-db", "preacher", "pushdown"} == ids
    assert candidate_selection_family_key(rotation_dumbbell) == candidate_selection_family_key(rotation_cable)


def test_short_pool_never_spends_protected_biceps_or_triceps_slot_on_rotation():
    wrist = _candidate("wrist", "머신 리스트 컬", "forearm", 130, "MACHINE")
    reverse_wrist = _candidate("reverse", "덤벨 리버스 리스트 컬", "forearm", 120)
    rotation = _candidate("rotation", "덤벨 전완 회내/회외", "forearm", 100)
    preacher = _candidate("preacher", "머신 프리처 컬", "biceps", 115, "MACHINE")
    pushdown = _candidate("pushdown", "머신 푸시다운", "triceps", 110, "MACHINE")

    result = _ensure_specialization_focus_candidate(
        selected=[wrist, preacher, pushdown],
        sorted_candidates=[wrist, reverse_wrist, preacher, pushdown, rotation],
        specialization_focus="forearmEmphasis",
        adjunct_options=[],
        target_muscles=["forearm", "biceps", "triceps"],
        top_n=3,
    )

    assert {row["primary_muscle"] for row in result} == {"forearm", "biceps", "triceps"}
    assert "rotation" not in {row["id"] for row in result}


def test_rotation_is_focus_only_and_balanced_arms_remain_unchanged():
    rows = _arm_candidates()
    focused = _rank(rows)
    balanced = _rank(rows, specialization_focus=None)

    assert ROTATION_ID in {row["id"] for row in focused}
    assert ROTATION_ID not in {row["id"] for row in balanced}


def test_rotation_respects_doms_low_readiness_injury_and_equipment_boundaries():
    rows = _arm_candidates()
    doms = _rank(rows, doms_db={"forearm": 2})
    low = _rank(rows, readiness_level="low")
    injured = _rank(rows, pain_areas=[PainAreaEntry(body_part="forearm")])
    no_dumbbell = _rank(
        rows,
        available_equipment=["BODYWEIGHT"],
    )

    assert ROTATION_ID in {row["id"] for row in doms}
    assert ROTATION_ID in {row["id"] for row in low}
    assert ROTATION_ID not in {row["id"] for row in injured}
    assert ROTATION_ID not in {row["id"] for row in no_dumbbell}


def test_beginner_and_unrelated_core_queries_do_not_admit_rotation():
    rows = _arm_candidates()
    beginner = _rank(rows, experience_level="beginner")
    engine, session = _session()
    try:
        core_rows = db_queries.get_candidate_exercises(
            session, ["abs"], [], [], available_equipment=["DUMBBELL"]
        )
    finally:
        session.close()
        engine.dispose()

    assert ROTATION_ID not in {row["id"] for row in beginner}
    assert ROTATION_ID not in {row["id"] for row in core_rows}


def test_no_static_grip_or_carry_is_admitted_through_rotation_review():
    reviewed = [
        row
        for row in _arm_candidates()
        if row.get("core_admission_review_version")
        == db_queries.FOREARM_PATTERN_REVIEW_VERSION
    ]

    assert [row["id"] for row in reviewed] == [ROTATION_ID]
    assert reviewed[0]["movement_type"] == "ISOLATION"
    assert reviewed[0]["prescription_mode"] == "reps"
    assert not any(
        token in (str(reviewed[0].get("name_kr")) + str(reviewed[0].get("name_en"))).lower()
        for token in ("grip", "carry", "hold", "hang", "그립", "캐리", "홀드", "행")
    )


def _blueprint(focus, *, experience="advanced"):
    return build_routine_blueprint(
        target_split_label="arm",
        target_muscles=[],
        goal="hypertrophy",
        experience_level=experience,
        readiness_level="normal",
        time_available_min=60,
        specialization_focus=focus,
        method_mode="off",
    )


def _bundle_ids(bundles, slot_id):
    bundle = next(row for row in bundles if row["slotId"] == slot_id)
    return [row["exerciseId"] for row in bundle["candidates"]]


def test_actual_073_slots_expose_rotation_as_optional_focus_alternative_only():
    ranked = _rank(_arm_candidates())
    bundles = build_blueprint_candidate_bundles(
        _blueprint("forearmEmphasis"), ranked
    )

    dedicated = _bundle_ids(bundles, "specialization_focus")
    ordinary = _bundle_ids(bundles, "target_3_forearm")
    assert dedicated == ["V24ROW-006181", ROTATION_ID]
    assert ordinary == ["V24ROW-006121"]
    assert len(dedicated) == len(set(dedicated))
    assert _bundle_ids(bundles, "target_1_biceps")
    assert _bundle_ids(bundles, "target_2_triceps")


def test_focus_slot_widening_does_not_apply_to_beginner_balanced_or_other_focus():
    ranked = _rank(_arm_candidates())
    with pytest.raises(ValueError, match="specialization focus requires advanced"):
        _blueprint("forearmEmphasis", experience="beginner")
    other = build_blueprint_candidate_bundles(_blueprint("bicepsEmphasis"), ranked)
    balanced = build_blueprint_candidate_bundles(_blueprint(None), ranked)

    assert len(_bundle_ids(other, "specialization_focus")) <= 1
    assert all(row["slotId"] != "specialization_focus" for row in balanced)


def _actual_073_pipeline(monkeypatch, branch):
    from engines import routine_pipeline
    from scripts.run_gemma4_quality_seed_live import request_from_seed
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    fixture = Path(__file__).resolve().parents[1] / "docs/reviews/fixtures/routine-ai-core-current-comprehensive-expert-review-v3.jsonl"
    row = next(json.loads(line) for line in fixture.read_text().splitlines()
               if json.loads(line)["scenarioId"] == "expert-current-v3-073")
    request = request_from_seed(row)
    for key, value in {
        "LLM_PROVIDER": "local", "APP_ENV": "development",
        "ENABLE_LOCAL_RAW_JSON_INVOKE": "true",
        "ROUTINE_CANDIDATE_POOL_SIZE": "12",
        "ROUTINE_COMPACT_CANDIDATE_PROMPT": "true",
    }.items():
        monkeypatch.setenv(key, value)
    trace = {"networkAttempts": 0}
    original = routine_pipeline.build_blueprint_candidate_bundles

    def bundles(blueprint, ranked, **kwargs):
        result = original(blueprint, ranked, **kwargs)
        trace.update(ranked=ranked, bundles=result)
        selections = []
        for bundle in result:
            if bundle["slotId"] == "flex" or not bundle["candidates"]:
                continue
            chosen = bundle["candidates"][0]["exerciseId"]
            if branch == "rotation" and bundle["slotId"] == "specialization_focus":
                assert ROTATION_ID in _bundle_ids(result, "specialization_focus")
                chosen = ROTATION_ID
            selections.append({"slotId": bundle["slotId"], "exerciseId": chosen})
        trace["selections"] = selections
        if branch == "legacy":
            trace["selections"] = [
                {"slotId": slot, "exerciseId": exercise_id}
                for slot, exercise_id in (
                    ("target_1_biceps", "V24ROW-006042"),
                    ("target_2_triceps", "V24ROW-006093"),
                    ("target_3_forearm", "V24ROW-006181"),
                    ("flex", "V24ROW-006048"),
                    ("flex", "V24ROW-002639"),
                )
            ]
        return result

    def model(_):
        if branch == "fallback":
            raise RuntimeError("synthetic forced fallback")
        return AIMessage(content=json.dumps({"selections": trace["selections"]}))

    def deny(*args, **kwargs):
        trace["networkAttempts"] += 1
        raise AssertionError("network forbidden in forearm acceptance test")

    monkeypatch.setattr(routine_pipeline, "build_blueprint_candidate_bundles", bundles)
    monkeypatch.setattr(routine_pipeline, "get_llm", lambda *args, **kwargs: RunnableLambda(model))
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    engine, session = _session()
    try:
        response = routine_pipeline.generate_smart_routine(
            request, session, profile=None, recent_sets=[]
        )
    finally:
        session.close()
        engine.dispose()
    return response, trace


@pytest.mark.parametrize("branch", ["classic", "rotation", "fallback", "legacy"])
def test_actual_073_final_allocation_keeps_biceps_complement(monkeypatch, branch):
    response, trace = _actual_073_pipeline(monkeypatch, branch)
    blocks = response.routine_blocks
    counts = Counter(block.primary_muscles[0] for block in blocks)
    sets = Counter()
    for block in blocks:
        sets[block.primary_muscles[0]] += len(block.prescription)
    assert counts == {"forearm": 2, "biceps": 2, "triceps": 1}
    assert sets == {"forearm": 10, "biceps": 7, "triceps": 3}
    assert len(blocks) == 5
    assert response.total_estimated_time == 46
    assert response.generation_status == ("fallback" if branch == "fallback" else "success")
    assert (ROTATION_ID in {block.exercise_id for block in blocks}) == (branch == "rotation")
    biceps = [candidate_selection_family_key(row) for row in trace["ranked"]
              if row.get("primary_muscle") == "biceps"]
    assert len(set(biceps)) >= 2
    assert ROTATION_ID not in _bundle_ids(trace["bundles"], "flex")
    assert trace["networkAttempts"] == 0


def test_rotation_reservation_preserves_two_eligible_biceps_families():
    wrist = _candidate("wrist", "Machine Wrist Curl", "forearm", 130)
    reverse = _candidate("reverse", "Reverse Wrist Curl", "forearm", 120)
    preacher = _candidate("preacher", "Machine Preacher Curl", "biceps", 115)
    preacher_clone = _candidate("preacher-clone", "Dumbbell Preacher Curl", "biceps", 114)
    incline = _candidate("incline", "Dumbbell Incline Curl", "biceps", 100)
    pushdown = _candidate("pushdown", "Machine Pushdown", "triceps", 110)
    rotation = _candidate("rotation", "Dumbbell Pronation Supination", "forearm", 90)
    selected = [wrist, reverse, preacher, preacher_clone, pushdown, incline]
    result = _ensure_specialization_focus_candidate(
        selected=selected, sorted_candidates=selected + [rotation],
        specialization_focus="forearmEmphasis", adjunct_options=[],
        target_muscles=["forearm", "biceps", "triceps"], top_n=6,
    )
    assert {candidate_selection_family_key(row) for row in result
            if row["primary_muscle"] == "biceps"} == {
        candidate_selection_family_key(preacher), candidate_selection_family_key(incline)
    }
    assert "rotation" in {row["id"] for row in result}


def test_unexposed_rotation_cannot_enter_flex_when_focus_choices_are_capped_at_one():
    bundles = build_blueprint_candidate_bundles(
        _blueprint("forearmEmphasis"), _rank(_arm_candidates()), top_k_per_slot=1
    )
    assert len(_bundle_ids(bundles, "specialization_focus")) == 1
    assert ROTATION_ID not in _bundle_ids(bundles, "specialization_focus")
    assert ROTATION_ID not in _bundle_ids(bundles, "flex")

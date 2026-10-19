import json
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import engines.db_queries as db_queries
from engines.candidate_ranker import score_candidate_exercises
from engines.db_queries import (
    _enrich_candidates_with_local_catalog,
    clear_local_catalog_enrichment_cache,
    get_candidate_exercises,
)
from engines.experience_policy import main_role_eligibility
from engines.fallback import generate_fallback_routine
from engines.registry import SPLIT_LABEL_TO_MUSCLES
from engines.schemas import PainAreaEntry, RoutineRequest
from engines.routine_blueprint import (
    RoutineBlueprintSlot,
    build_blueprint_candidate_bundles,
    build_routine_blueprint,
)


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "fit_core.sqlite"
EXERCISE_ID = "V24ROW-005748"
TARGETS = ["front-deltoids", "side-deltoids", "back-deltoids", "trapezius"]


def _candidates(*, available=None, unavailable=None):
    engine = create_engine(f"sqlite:///{DB_PATH}")
    try:
        with Session(engine) as session:
            return get_candidate_exercises(
                session,
                TARGETS,
                unavailable or [],
                [],
                available_equipment=available,
            )
    finally:
        engine.dispose()


def _press_candidate():
    return next(row for row in _candidates(available=["DUMBBELL", "MACHINE", "CABLE"])
                if row["id"] == EXERCISE_ID)


def _rank(candidate, **overrides):
    kwargs = {
        "available_equipment": ["DUMBBELL", "MACHINE", "CABLE"],
        "blocked_equipment": [],
        "pain_areas": [],
        "doms_db": {},
        "readiness_level": "normal",
        "goal": "hypertrophy",
        "experience_level": "advanced",
        "training_style": "gym",
        "target_split_label": "shoulder",
        "session_structure": "standard",
        "adjunct_options": [],
        "top_n": 12,
    }
    kwargs.update(overrides)
    return score_candidate_exercises([candidate], TARGETS, **kwargs)


def _rank_all(**overrides):
    kwargs = {
        "available_equipment": ["DUMBBELL", "MACHINE", "CABLE"],
        "blocked_equipment": [],
        "pain_areas": [],
        "doms_db": {},
        "readiness_level": "normal",
        "goal": "hypertrophy",
        "experience_level": "advanced",
        "training_style": "gym",
        "target_split_label": "shoulder",
        "session_structure": "standard",
        "adjunct_options": [],
        "top_n": 20,
    }
    kwargs.update(overrides)
    return score_candidate_exercises(
        _candidates(available=["DUMBBELL", "MACHINE", "CABLE"]),
        TARGETS,
        **kwargs,
    )


def test_reviewed_exact_dumbbell_row_is_admitted_without_metadata_relaxation():
    candidate = _press_candidate()

    assert candidate["main_routine_default_allowed"] == 1
    assert candidate["catalog_original_main_routine_default_allowed"] == 0
    assert candidate["beginner_default_allowed"] == 0
    assert candidate["pain_risk_level"] == "high"
    assert candidate["injury_caution_level"] == "high"
    assert candidate["overhead_position_required"] == 1
    assert candidate["shoulder_elevation_demand"] == "high"
    assert candidate["movement_profile_review_required"] == 1
    assert candidate["core_main_role_reviewed"] == 1
    assert candidate["primary_muscle"] == "front-deltoids"
    assert "side-deltoids" in candidate["secondary_muscle"].split(", ")
    assert candidate["core_admission_original_secondary_muscle"] == (
        "upper-back, triceps, trapezius"
    )
    assert candidate["core_admission_reviewed_involvement_muscles"] == (
        "side-deltoids"
    )
    assert [row["id"] for row in _rank(candidate)] == [EXERCISE_ID]


def test_reviewed_press_has_optional_actual_main_role_for_advanced_shoulder():
    ranked = _rank(_press_candidate())
    blueprint = build_routine_blueprint(
        target_split_label="shoulder",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="advanced",
        readiness_level="normal",
        time_available_min=60,
        training_style="gym",
    )
    bundles = build_blueprint_candidate_bundles(blueprint, ranked)
    main = next(row for row in bundles if row["slotId"] == "shoulder_compound_main")

    assert main["phase"] == "main_compound"
    assert main["requirement"] == "preferred_when_safe"
    assert [row["exerciseId"] for row in main["candidates"]] == [EXERCISE_ID]
    assert not any(
        slot.slot_id.endswith("front_deltoids") for slot in blueprint.slots
    )


def test_reviewed_press_occupies_main_bundle_with_complete_ranked_pool():
    ranked = _rank_all()
    blueprint = build_routine_blueprint(
        target_split_label="shoulder",
        target_muscles=TARGETS,
        goal="hypertrophy",
        experience_level="advanced",
        readiness_level="normal",
        time_available_min=60,
        training_style="gym",
    )
    bundles = build_blueprint_candidate_bundles(blueprint, ranked)
    main = next(row for row in bundles if row["slotId"] == "shoulder_compound_main")

    assert EXERCISE_ID in [row["id"] for row in ranked]
    assert [row["exerciseId"] for row in main["candidates"]] == [EXERCISE_ID]
    assert not any(row["slotId"].endswith("front_deltoids") for row in bundles)


def test_exact_row_admission_does_not_broaden_beginner_main_role():
    candidate = _press_candidate()
    ranked = _rank(candidate, experience_level="beginner")
    blueprint = build_routine_blueprint(
        target_split_label="shoulder",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="beginner",
        readiness_level="normal",
        time_available_min=60,
        training_style="gym",
    )
    main = next(
        row for row in build_blueprint_candidate_bundles(blueprint, ranked)
        if row["slotId"] == "shoulder_compound_main"
    )

    assert candidate["beginner_default_allowed"] == 0
    assert main["candidates"] == []


def test_exact_reviewed_row_is_not_admitted_to_beginner_ranked_pool():
    assert _rank(_press_candidate(), experience_level="beginner") == []


def test_full_body_fallback_preserves_real_posterior_chain_before_optional_press():
    targets = SPLIT_LABEL_TO_MUSCLES["full_body"]
    unavailable = ["BARBELL", "MACHINE", "CABLE", "SMITH_MACHINE"]
    available = ["DUMBBELL", "BODYWEIGHT"]
    engine = create_engine(f"sqlite:///{DB_PATH}")
    try:
        with Session(engine) as session:
            candidates = get_candidate_exercises(
                session,
                targets,
                unavailable,
                [],
                available_equipment=available,
            )
    finally:
        engine.dispose()

    ranked = score_candidate_exercises(
        candidates,
        targets,
        available_equipment=available,
        blocked_equipment=unavailable,
        pain_areas=[],
        doms_db={},
        readiness_level="normal",
        goal="hypertrophy",
        experience_level="intermediate",
        training_style="gym",
        target_split_label="full_body",
        session_structure="standard",
        adjunct_options=[],
        time_available_min=50,
        top_n=12,
    )
    blueprint = build_routine_blueprint(
        target_split_label="full_body",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=50,
        training_style="gym",
    )
    bundles = build_blueprint_candidate_bundles(blueprint, ranked)
    by_slot = {bundle["slotId"]: bundle for bundle in bundles}
    posterior_candidates = by_slot["full_posterior_chain"]["candidates"]
    vertical_ids = {
        row["exerciseId"]
        for row in by_slot["full_vertical_push"]["candidates"]
    }

    assert any(
        row["primaryMuscle"] in {"hamstring", "gluteal"}
        for row in posterior_candidates
    )
    assert EXERCISE_ID in vertical_ids

    request = RoutineRequest(
        user_id="shoulder-admission-integration",
        target_split_label="full_body",
        target_muscles=[],
        time_available_min=50,
        readiness_level="normal",
        pain_areas=[],
        doms_data={},
        available_equipment=available,
        equipment=unavailable,
        goal="hypertrophy",
        experience_level="intermediate",
        session_structure="standard",
        adjunct_options=[],
        training_style="gym",
    )
    draft = generate_fallback_routine(
        request,
        ranked,
        15,
        doms_db={},
        goal="hypertrophy",
        target_exercise_count=blueprint.target_exercise_count,
        minimum_main_exercise_count=blueprint.minimum_main_exercise_count,
        applied_adjunct_options=blueprint.planned_adjunct_options,
        routine_blueprint=blueprint,
    )

    assert any(
        set(block.primary_muscles) & {"hamstring", "gluteal"}
        for block in draft.routine_blocks
    )


def test_later_target_reservation_requires_available_current_and_compatible_future():
    base = build_routine_blueprint(
        target_split_label="full_body",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=50,
        training_style="gym",
    )

    def slot(slot_id, requirement, targets, movements):
        return RoutineBlueprintSlot(
            slot_id=slot_id,
            phase="secondary_compound",
            requirement=requirement,
            target_muscles=tuple(targets),
            preferred_movement_types=tuple(movements),
            preferred_selection_patterns=(),
            purpose=slot_id,
        )

    blueprint = replace(
        base,
        slots=(
            slot("prior_chest", "required_when_safe", ["chest"], ["COMPOUND"]),
            slot(
                "current_compound",
                "required_when_safe",
                ["chest", "triceps"],
                ["COMPOUND"],
            ),
            slot(
                "future_isolation",
                "preferred_when_safe",
                ["triceps"],
                ["ISOLATION"],
            ),
        ),
    )
    candidates = [
        {
            "id": "prior-chest",
            "name_en": "Chest Press",
            "primary_muscle": "chest",
            "movement_type": "COMPOUND",
            "score": 30,
        },
        {
            "id": "current-triceps-compound",
            "name_en": "Triceps Compound",
            "primary_muscle": "triceps",
            "movement_type": "COMPOUND",
            "score": 20,
        },
        {
            "id": "future-triceps-isolation",
            "name_en": "Triceps Isolation",
            "primary_muscle": "triceps",
            "movement_type": "ISOLATION",
            "score": 10,
        },
    ]

    bundles = build_blueprint_candidate_bundles(
        blueprint,
        candidates,
        top_k_per_slot=1,
    )
    by_slot = {bundle["slotId"]: bundle for bundle in bundles}

    assert [
        row["exerciseId"] for row in by_slot["current_compound"]["candidates"]
    ] == ["current-triceps-compound"]
    assert [
        row["exerciseId"] for row in by_slot["future_isolation"]["candidates"]
    ] == ["future-triceps-isolation"]


@pytest.mark.parametrize("exclusive_candidate_state", ["free", "prior", "intervening"])
def test_target_reservation_does_not_displace_independent_posterior_candidate(
    exclusive_candidate_state,
):
    base = build_routine_blueprint(
        target_split_label="legs", target_muscles=[], goal="hypertrophy",
        experience_level="intermediate", readiness_level="normal",
        time_available_min=50, training_style="gym",
    )

    def slot(slot_id, targets, movements):
        return RoutineBlueprintSlot(
            slot_id=slot_id, phase="secondary_compound",
            requirement="required_when_safe", target_muscles=tuple(targets),
            preferred_movement_types=tuple(movements),
            preferred_selection_patterns=(), purpose=slot_id,
        )

    current = slot("current", ["quadriceps", "gluteal"], ["COMPOUND"])
    posterior = slot("posterior", ["hamstring", "gluteal"], ["COMPOUND", "ISOLATION"])
    competing = slot("competing", ["hamstring"], ["ISOLATION"])
    slots = [current, posterior]
    if exclusive_candidate_state == "prior":
        slots.insert(0, competing)
    elif exclusive_candidate_state == "intervening":
        slots.insert(1, competing)
    candidates = [
        {"id": "quads", "name_en": "Squat", "primary_muscle": "quadriceps",
         "movement_type": "COMPOUND", "score": 30},
        {"id": "glutes", "name_en": "Hip Thrust", "primary_muscle": "gluteal",
         "movement_type": "COMPOUND", "score": 20},
        {"id": "hamstrings", "name_en": "Leg Curl", "primary_muscle": "hamstring",
         "movement_type": "ISOLATION", "score": 10},
    ]
    bundles = build_blueprint_candidate_bundles(
        replace(base, slots=tuple(slots)), candidates, top_k_per_slot=2,
    )
    by_slot = {b["slotId"]: [c["exerciseId"] for c in b["candidates"]] for b in bundles}
    if exclusive_candidate_state == "free":
        assert by_slot["current"] == ["quads", "glutes"]
        assert by_slot["posterior"] == ["hamstrings"]
    else:
        assert by_slot["current"] == ["quads"]
        assert by_slot["posterior"] == ["glutes"]


def test_live_seed_case_008_pipeline_fallback_preserves_posterior_chain(
    monkeypatch,
):
    from langchain_core.runnables import RunnableLambda

    from engines import routine_pipeline
    from scripts.run_gemma4_quality_seed_live import request_from_seed

    seed_path = (
        ROOT
        / "docs/reviews/fixtures"
        / "routine-ai-core-current-comprehensive-expert-review-v3.jsonl"
    )
    seed_row = next(
        json.loads(line)
        for line in seed_path.read_text(encoding="utf-8").splitlines()
        if '"scenarioId":"expert-current-v3-008"' in line
    )
    request = request_from_seed(seed_row)

    class ForcedFailureModel:
        def with_structured_output(self, *_args, **_kwargs):
            def fail(_payload):
                raise ConnectionError("forced offline fallback")

            return RunnableLambda(fail)

        def invoke(self, *_args, **_kwargs):
            raise ConnectionError("forced offline fallback")

    monkeypatch.setattr(
        routine_pipeline,
        "get_llm",
        lambda *_args, **_kwargs: ForcedFailureModel(),
    )
    engine = create_engine(f"sqlite:///{DB_PATH}")
    try:
        with Session(engine) as session:
            result = routine_pipeline.generate_smart_routine(
                request,
                session,
                profile=None,
                recent_sets=[],
            )
    finally:
        engine.dispose()

    assert result.is_fallback is True
    assert any(
        set(block.primary_muscles) & {"hamstring", "gluteal"}
        for block in result.routine_blocks
    )


def test_exact_row_respects_equipment_exclusions_and_machine_is_not_dumbbell():
    assert EXERCISE_ID not in {
        row["id"] for row in _candidates(available=["MACHINE"])
    }
    assert EXERCISE_ID not in {
        row["id"]
        for row in _candidates(
            available=["DUMBBELL", "MACHINE"],
            unavailable=["DUMBBELL"],
        )
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"shoulder_condition_status": "activeUncleared"},
        {
            "shoulder_condition_status": "managedWithRestrictions",
            "restricted_movement_patterns": ["overheadPosition"],
        },
        {"doms_db": {"front-deltoids": 3}},
    ],
)
def test_exact_row_never_overrides_shoulder_or_severe_doms_boundaries(overrides):
    assert _rank(_press_candidate(), **overrides) == []


def test_exact_row_respects_source_backed_side_deltoid_health_boundaries():
    candidate = _press_candidate()

    assert _rank(candidate, doms_db={"side-deltoids": 3}) == []
    assert _rank(
        candidate,
        pain_areas=[PainAreaEntry(body_part="side-deltoids")],
    ) == []
    assert [
        row["id"]
        for row in _rank(candidate, experience_level="intermediate")
    ] == [EXERCISE_ID]


def test_overlay_source_missing_or_changed_fails_closed(monkeypatch, tmp_path):
    raw_candidate = {
        "id": EXERCISE_ID,
        "name_kr": "덤벨 시티드 덤벨 숄더 프레스",
        "name_en": "Dumbbell Seated Dumbbell Shoulder Press",
        "primary_muscle": "front-deltoids",
        "secondary_muscle": "upper-back, triceps, trapezius",
        "equipment_req": "DUMBBELL",
        "difficulty_tier": 1,
        "efficiency_tier": 0,
        "pain_triggers": "front-deltoids",
        "movement_type": "COMPOUND",
    }
    source = db_queries.CANONICAL_SOURCE_CATALOG_PATH
    changed = tmp_path / "changed.csv"
    changed.write_text(
        source.read_text(encoding="utf-8-sig").replace(
            "Dumbbell Seated Dumbbell Shoulder Press",
            "changed identity",
            1,
        ),
        encoding="utf-8",
    )

    for invalid_source in (tmp_path / "missing.csv", changed):
        monkeypatch.setattr(db_queries, "CANONICAL_SOURCE_CATALOG_PATH", invalid_source)
        clear_local_catalog_enrichment_cache()
        enriched = _enrich_candidates_with_local_catalog([raw_candidate])[0]
        assert enriched["main_routine_default_allowed"] == 0
        assert "core_main_role_reviewed" not in enriched
    monkeypatch.setattr(db_queries, "CANONICAL_SOURCE_CATALOG_PATH", source)
    clear_local_catalog_enrichment_cache()


def test_reenrichment_revokes_stale_admission_when_catalog_is_unavailable(
    monkeypatch,
):
    candidate = _press_candidate()
    monkeypatch.setattr(db_queries, "_load_local_catalog_enrichment_cache", lambda: None)

    revoked = _enrich_candidates_with_local_catalog([candidate])[0]

    assert revoked["main_routine_default_allowed"] == 0
    assert not set(db_queries._CORE_ADMISSION_DERIVED_FIELDS).intersection(revoked)


def test_reenrichment_revokes_stale_admission_when_no_catalog_row_matches():
    candidate = {
        **_press_candidate(),
        "id": "V24ROW-UNREVIEWED",
        "name_kr": "unmatched identity",
        "name_en": "unmatched identity",
    }

    revoked = _enrich_candidates_with_local_catalog([candidate])[0]

    assert revoked["main_routine_default_allowed"] == 0
    assert not set(db_queries._CORE_ADMISSION_DERIVED_FIELDS).intersection(revoked)


def test_role_review_relaxation_requires_exact_verified_provenance():
    candidate = _press_candidate()
    assert main_role_eligibility(candidate, "advanced", "hypertrophy").eligible

    for changed in (
        {**candidate, "id": "V24ROW-UNREVIEWED"},
        {**candidate, "catalog_exercise_id": "V24ROW-UNREVIEWED"},
        {**candidate, "core_admission_source_sha256": "changed"},
        {**candidate, "core_admission_review_version": "changed"},
    ):
        decision = main_role_eligibility(changed, "advanced", "hypertrophy")
        assert not decision.eligible
        assert decision.reason_code == "advanced_nonstandard_main_role_gate"


def test_shoulder_press_slot_respects_explicit_targets_and_short_capacity():
    side_rear = build_routine_blueprint(
        target_split_label="shoulder",
        target_muscles=["side-deltoids", "back-deltoids"],
        goal="hypertrophy",
        experience_level="advanced",
        readiness_level="normal",
        time_available_min=30,
        training_style="gym",
    )
    assert [slot.slot_id for slot in side_rear.slots] == [
        "target_1_side_deltoids",
        "target_2_back_deltoids",
    ]
    assert all(
        "front-deltoids" not in slot.target_muscles for slot in side_rear.slots
    )
    assert len(side_rear.slots) <= side_rear.target_main_exercise_count
    side_rear_bundles = build_blueprint_candidate_bundles(side_rear, _rank_all())
    assert EXERCISE_ID not in {
        candidate["exerciseId"]
        for bundle in side_rear_bundles
        for candidate in bundle["candidates"]
    }

    short_front = build_routine_blueprint(
        target_split_label="shoulder",
        target_muscles=["front-deltoids", "side-deltoids", "back-deltoids"],
        goal="hypertrophy",
        experience_level="advanced",
        readiness_level="normal",
        time_available_min=30,
        training_style="gym",
    )
    assert [slot.slot_id for slot in short_front.slots].count(
        "shoulder_compound_main"
    ) == 1
    assert not any(
        slot.slot_id.endswith("front_deltoids") for slot in short_front.slots
    )
    assert len(short_front.slots) <= short_front.target_main_exercise_count


def test_overlay_requires_exact_candidate_id_and_repeated_enrichment_is_stable():
    candidate = _press_candidate()
    repeated = _enrich_candidates_with_local_catalog([candidate])[0]
    assert {
        key: repeated.get(key)
        for key in (
            "main_routine_default_allowed",
            "beginner_default_allowed",
            "pain_risk_level",
            "overhead_position_required",
            "core_admission_review_version",
            "core_admission_source_sha256",
            "core_main_role_reviewed",
        )
    } == {
        key: candidate.get(key)
        for key in (
            "main_routine_default_allowed",
            "beginner_default_allowed",
            "pain_risk_level",
            "overhead_position_required",
            "core_admission_review_version",
            "core_admission_source_sha256",
            "core_main_role_reviewed",
        )
    }

    mismatched = {**candidate, "id": "V24ROW-UNREVIEWED"}
    mismatched = _enrich_candidates_with_local_catalog([mismatched])[0]
    assert mismatched.get("main_routine_default_allowed") == 0
    assert "core_main_role_reviewed" not in mismatched

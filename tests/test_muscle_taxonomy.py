from pathlib import Path
import sqlite3

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from engines.candidate_ranker import score_candidate_exercises
from engines.db_queries import (
    clear_local_catalog_enrichment_cache,
    get_candidate_exercises,
    get_exercise_substitution_map_from_sqlite,
)
from engines.muscle_mapping import resolve_request_target_muscles
from engines.muscle_taxonomy import (
    classify_exercise_muscle_taxonomy,
    recommendation_taxonomy_is_eligible,
    source_query_muscles_for_canonical_targets,
)


DB_PATH = Path(__file__).resolve().parents[1] / "fit_core.sqlite"


def test_explicit_deltoid_raise_taxonomy_is_directional():
    lateral = classify_exercise_muscle_taxonomy(
        name_kr="덤벨 스탠딩 레터럴 레이즈",
        name_en="Dumbbell Standing Lateral Raise",
        source_primary_muscle="front-deltoids",
        source_pain_triggers="front-deltoids",
    )
    front = classify_exercise_muscle_taxonomy(
        name_kr="덤벨 프론트 레이즈",
        name_en="Dumbbell Front Raise",
        source_primary_muscle="back-deltoids",
        source_pain_triggers="back-deltoids",
    )
    rear = classify_exercise_muscle_taxonomy(
        name_kr="머신 리버스 펙덱",
        name_en="Machine Reverse Pec Deck",
        source_primary_muscle="front-deltoids",
        source_pain_triggers="front-deltoids",
    )

    assert lateral.canonical_primary_muscle == "side-deltoids"
    assert lateral.canonical_pain_triggers == "side-deltoids"
    assert front.canonical_primary_muscle == "front-deltoids"
    assert front.canonical_pain_triggers == "front-deltoids"
    assert rear.canonical_primary_muscle == "back-deltoids"


def test_prone_generic_lateral_raise_remains_posterior_deltoid():
    decision = classify_exercise_muscle_taxonomy(
        name_kr="덤벨 엎드림 레터럴 레이즈",
        name_en="Dumbbell Prone Lateral Raise",
        source_primary_muscle="back-deltoids",
        source_pain_triggers="back-deltoids",
    )

    assert decision.canonical_primary_muscle == "back-deltoids"
    assert decision.classification_rule_code == "prone_generic_lateral_raise"


def test_explicit_deltoid_patterns_leave_the_review_queue():
    cases = [
        ("케이블 페이스 풀", "Cable Face Pull", "back-deltoids", "back-deltoids"),
        ("밴드 풀어파트", "Band Pull Apart", "back-deltoids", "back-deltoids"),
        ("덤벨 파월 레이즈", "Dumbbell Powell Raise", "back-deltoids", "back-deltoids"),
        ("덤벨 프론 T 레이즈", "Dumbbell Front T Raise", "front-deltoids", "front-deltoids"),
        ("머신 숄더 프레스", "Machine Shoulder Press", "front-deltoids", "front-deltoids"),
    ]

    for name_kr, name_en, source_primary, expected_primary in cases:
        decision = classify_exercise_muscle_taxonomy(
            name_kr=name_kr,
            name_en=name_en,
            source_primary_muscle=source_primary,
            source_pain_triggers=source_primary,
            detail_subpart="삼각근",
        )

        assert decision.canonical_primary_muscle == expected_primary
        assert decision.review_required == 0
        assert decision.review_status == "autoClassified"
        assert decision.confidence == "high"


def test_wrist_curl_and_shrug_use_canonical_forearm_and_trapezius_targets():
    wrist_curl = classify_exercise_muscle_taxonomy(
        name_kr="덤벨 리스트 컬",
        name_en="Dumbbell Wrist Curl",
        source_primary_muscle="upper-back",
        source_pain_triggers="upper-back",
    )
    shrug = classify_exercise_muscle_taxonomy(
        name_kr="덤벨 스탠딩 뉴트럴 슈러그",
        name_en="Dumbbell Standing Neutral Shrug",
        source_primary_muscle="upper-back",
        source_pain_triggers="upper-back",
    )

    assert wrist_curl.canonical_primary_muscle == "forearm"
    assert wrist_curl.canonical_pain_triggers == "forearm"
    assert wrist_curl.classification_rule_code == "explicit_wrist_curl"
    assert shrug.canonical_primary_muscle == "trapezius"
    assert shrug.canonical_pain_triggers == "trapezius"
    assert shrug.classification_rule_code == "explicit_shrug"


def test_ambiguous_deltoid_taxonomy_is_preserved_but_not_recommendation_eligible():
    decision = classify_exercise_muscle_taxonomy(
        name_kr="리포머 Y 레이즈",
        name_en="Reformer Y Raise",
        source_primary_muscle="back-deltoids",
        source_pain_triggers="back-deltoids",
        detail_subpart="삼각근",
    )

    assert decision.review_required == 1
    assert decision.review_status == "reviewRequired"
    assert not recommendation_taxonomy_is_eligible(decision.to_dict())
    assert recommendation_taxonomy_is_eligible({})


def test_ranker_quarantines_review_required_taxonomy_even_for_in_memory_candidates():
    candidates = [
        {
            "id": "unresolved",
            "name_kr": "미검수 운동",
            "name_en": "Unresolved Exercise",
            "primary_muscle": "back-deltoids",
            "equipment_req": "DUMBBELL",
            "efficiency_tier": 1,
            "movement_type": "ISOLATION",
            "pain_triggers": "",
            "taxonomy_review_required": 1,
        },
        {
            "id": "approved",
            "name_kr": "검수 운동",
            "name_en": "Approved Exercise",
            "primary_muscle": "back-deltoids",
            "equipment_req": "DUMBBELL",
            "efficiency_tier": 3,
            "movement_type": "ISOLATION",
            "pain_triggers": "",
            "taxonomy_review_required": 0,
        },
    ]

    ranked = score_candidate_exercises(
        candidates,
        ["back-deltoids"],
        top_n=2,
    )

    assert [candidate["id"] for candidate in ranked] == ["approved"]


def test_deltoid_source_query_aliases_are_internal_only():
    assert source_query_muscles_for_canonical_targets(["side-deltoids"]) == [
        "side-deltoids",
        "front-deltoids",
        "back-deltoids",
    ]
    assert resolve_request_target_muscles("shoulder", []) == [
        "front-deltoids",
        "side-deltoids",
        "back-deltoids",
        "trapezius",
    ]
    assert resolve_request_target_muscles("push", []) == [
        "chest",
        "front-deltoids",
        "side-deltoids",
        "triceps",
    ]
    assert source_query_muscles_for_canonical_targets(["forearm"]) == [
        "forearm",
        "upper-back",
    ]
    assert source_query_muscles_for_canonical_targets(["trapezius"]) == [
        "trapezius",
        "upper-back",
    ]


def test_taxonomy_table_preserves_source_and_covers_catalog():
    with sqlite3.connect(DB_PATH) as conn:
        exercise_count = conn.execute("SELECT COUNT(*) FROM exercise_tier").fetchone()[0]
        coverage = conn.execute(
            "SELECT COUNT(*) FROM exercise_muscle_taxonomy_v2"
        ).fetchone()[0]
        rows = {
            row[0]: row[1:]
            for row in conn.execute("""
                SELECT exercise_id, source_primary_muscle,
                       canonical_primary_muscle, canonical_pain_triggers,
                       classification_rule_code
                FROM exercise_muscle_taxonomy_v2
                WHERE exercise_id IN (
                    'V24ROW-000656',
                    'V24ROW-005798',
                    'V24ROW-005801',
                    'V24ROW-005810'
                )
            """)
        }

    assert coverage == exercise_count == 24653
    assert rows["V24ROW-000656"][:2] == ("front-deltoids", "side-deltoids")
    assert rows["V24ROW-005798"][:3] == (
        "back-deltoids",
        "side-deltoids",
        "side-deltoids",
    )
    assert rows["V24ROW-005801"][1] == "back-deltoids"
    assert rows["V24ROW-005810"][1] == "front-deltoids"


def test_side_deltoid_target_returns_only_canonical_side_candidates():
    clear_local_catalog_enrichment_cache()
    engine = create_engine(f"sqlite:///{DB_PATH}")
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        candidates = get_candidate_exercises(
            session,
            ["side-deltoids"],
            [],
            [],
            ["DUMBBELL", "CABLE", "MACHINE", "BODYWEIGHT"],
        )
    finally:
        session.close()
        engine.dispose()

    assert candidates
    assert {candidate["primary_muscle"] for candidate in candidates} == {"side-deltoids"}
    assert any("레터럴 레이즈" in candidate["name_kr"] for candidate in candidates)


def test_calf_target_recovers_reviewed_rows_with_incorrect_source_primary():
    clear_local_catalog_enrichment_cache()
    engine = create_engine(f"sqlite:///{DB_PATH}")
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        candidates = get_candidate_exercises(
            session,
            ["calves"],
            [],
            [],
            ["DUMBBELL", "MACHINE", "BODYWEIGHT"],
        )
    finally:
        session.close()
        engine.dispose()

    assert candidates
    assert {candidate["primary_muscle"] for candidate in candidates} == {"calves"}
    assert any(
        candidate["id"] == "V24ROW-006846"
        for candidate in candidates
    )


def test_review_required_taxonomy_rows_do_not_exit_candidate_query():
    clear_local_catalog_enrichment_cache()
    engine = create_engine(f"sqlite:///{DB_PATH}")
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        candidates = get_candidate_exercises(
            session,
            ["back-deltoids"],
            [],
            [],
            ["DUMBBELL", "CABLE", "MACHINE", "BAND", "BODYWEIGHT"],
        )
    finally:
        session.close()
        engine.dispose()

    assert candidates
    assert all(
        int(candidate.get("taxonomy_review_required") or 0) == 0
        for candidate in candidates
    )


def test_substitution_surface_exposes_only_reviewed_taxonomy():
    substitutions = get_exercise_substitution_map_from_sqlite(limit=5000)
    exposed_ids = {
        str(item[key])
        for item in substitutions
        for key in ("source_exercise_id", "target_exercise_id")
    }

    with sqlite3.connect(DB_PATH) as conn:
        unresolved = {
            row[0]
            for row in conn.execute(
                "SELECT exercise_id FROM exercise_muscle_taxonomy_v2 WHERE review_required = 1"
            )
        }

    assert exposed_ids.isdisjoint(unresolved)

import pytest

from engines.candidate_ranker import format_candidates_for_prompt, score_candidate_exercises


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("raw,blocked,available,chosen", [
    ("BAND,CABLE", ["BAND"], ["CABLE"], "CABLE"),
    ("DUMBBELL,BODYWEIGHT", ["DUMBBELL"], ["MACHINE"], "BODYWEIGHT"),
    ("SMITH_MACHINE,MACHINE", [], ["SMITH_MACHINE"], "SMITH_MACHINE"),
])
def test_prompt_equipment_is_the_current_selected_option(compact, raw, blocked, available, chosen):
    candidate = {
        "id": "selected-equipment", "name_kr": "Press", "name_en": "Press",
        "primary_muscle": "chest", "movement_type": "COMPOUND",
        "equipment_req": raw, "efficiency_tier": 3,
    }
    ranked = score_candidate_exercises([candidate], ["chest"], blocked_equipment=blocked,
                                      available_equipment=available)
    assert len(ranked) == 1
    assert "selected_equipment_code" not in candidate
    prompt = format_candidates_for_prompt(ranked, compact=compact)
    marker = "equipment:" if compact else "equipment="
    assert marker + chosen + "," in prompt
    assert marker + raw + "," not in prompt


@pytest.mark.parametrize("raw,blocked,available,chosen", [
    ("BAND,CABLE", ["BAND"], ["CABLE"], "CABLE"),
    ("DUMBBELL,BODYWEIGHT", ["DUMBBELL"], ["MACHINE"], "BODYWEIGHT"),
    ("SMITH_MACHINE,MACHINE", [], ["SMITH_MACHINE"], "SMITH_MACHINE"),
    ("SMITH_MACHINE,MACHINE", [], ["MACHINE"], None),
    ("BAND,CABLE", ["BAND", "CABLE"], ["CABLE"], None),
])
def test_query_rank_prompt_and_fallback_share_equipment_choice(monkeypatch, raw, blocked, available, chosen):
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session
    from engines import db_queries
    from engines.fallback import generate_fallback_routine
    from engines.schemas import RoutineRequest

    monkeypatch.setattr(db_queries, "_load_local_catalog_enrichment_cache", lambda: None)
    monkeypatch.setattr(db_queries, "_source_query_muscles_from_local_taxonomy", lambda _: [])
    engine = create_engine("sqlite:///:memory:")
    try:
        with engine.begin() as connection:
            connection.execute(text("""CREATE TABLE exercise_tier (
                id TEXT PRIMARY KEY, name_kr TEXT, name_en TEXT,
                primary_muscle TEXT, secondary_muscle TEXT, equipment_req TEXT,
                difficulty_tier INTEGER, efficiency_tier INTEGER,
                pain_triggers TEXT, movement_type TEXT)"""))
            connection.execute(text("""INSERT INTO exercise_tier VALUES (
                'equipment-flow', 'Press', 'Press', 'chest', 'triceps',
                :equipment, 2, 3, NULL, 'COMPOUND')"""), {"equipment": raw})
        with Session(engine) as db:
            queried = db_queries.get_candidate_exercises(db, ["chest"], blocked, [], available)
        ranked = score_candidate_exercises(queried, ["chest"], blocked_equipment=blocked,
                                          available_equipment=available)
        req = RoutineRequest(user_id="equipment-flow", target_muscles=["chest"],
                             goal="hypertrophy", time_available_min=35,
                             equipment=blocked, available_equipment=available)
        result = generate_fallback_routine(req, ranked, max_total_sets=8)
        if chosen is None:
            assert queried == ranked == result.routine_blocks == []
            assert result.status_reason_code == "emptyCandidate"
        else:
            assert len(queried) == len(ranked) == len(result.routine_blocks) == 1
            assert queried[0]["selected_equipment_code"] == ranked[0]["selected_equipment_code"] == chosen
            assert "equipment=" + chosen + "," in format_candidates_for_prompt(ranked, compact=False)
            assert result.routine_blocks[0].equipment_type == chosen
    finally:
        engine.dispose()

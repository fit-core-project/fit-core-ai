from engines.routine_engine import (
    PainAreaEntry,
    RecentSetRecord,
    UserProfileContext,
    _build_system_prompt,
    format_candidates_for_prompt,
    score_candidate_exercises,
)


def test_scoring_prefers_compound_primary_target_match(mock_candidates):
    ranked = score_candidate_exercises(mock_candidates, ["CHEST_MID"])
    assert ranked[0]["id"] == "barbell_bench_press"
    assert "compound priority" in ranked[0]["score_reasons"]
    assert "primary target match" in ranked[0]["score_reasons"]


def test_scoring_limits_prompt_candidates_to_top_n(mock_candidates):
    expanded = [
        {**mock_candidates[i % len(mock_candidates)], "id": f"candidate_{i}"}
        for i in range(20)
    ]
    ranked = score_candidate_exercises(expanded, ["CHEST_MID"], top_n=5)
    assert len(ranked) == 5


def test_scoring_penalizes_doms_and_excludes_blocked_candidates(mock_candidates):
    candidates = [
        {**mock_candidates[0], "id": "moderate_chest"},
        {**mock_candidates[1], "id": "blocked_equipment", "equipment_req": "MACHINE"},
        {**mock_candidates[2], "id": "safe_bodyweight"},
    ]
    ranked = score_candidate_exercises(
        candidates,
        ["CHEST_MID"],
        doms_db={"CHEST_MID": 2},
        blocked_equipment=["MACHINE"],
    )
    ids = [candidate["id"] for candidate in ranked]
    assert "blocked_equipment" not in ids
    assert "doms moderate penalty" in ranked[0]["score_reasons"]


def test_prompt_v2_mentions_hard_constraints_and_ranked_candidates():
    prompt = _build_system_prompt(profile=None, recent_sets=None)
    assert "[ROLE]" in prompt
    assert "[HARD CONSTRAINTS]" in prompt
    assert "[OUTPUT SCHEMA]" in prompt
    assert "[PROHIBITED BEHAVIOR]" in prompt
    assert "[RANKED CANDIDATES]" in prompt
    assert "at or below {max_sets}" in prompt


def test_prompt_includes_profile_pain_and_recent_sets_context():
    profile = UserProfileContext(
        goal_type="hypertrophy",
        split_type="PPL",
        split_label="push",
        experience_level="intermediate",
        strength_baseline={"barbell_bench_press": {"weight_kg": 80, "reps": 5}},
        equipment_access=["BARBELL"],
        pain_areas=[PainAreaEntry(body_part="shoulder", side="left", severity="mild")],
    )
    recent_sets = [RecentSetRecord(exercise_name="barbell_bench_press", weight_kg=80, reps=5)]
    prompt = _build_system_prompt(profile=profile, recent_sets=recent_sets)
    assert "[USER PROFILE]" in prompt
    assert "shoulder(left, mild)" in prompt
    assert "[RECENT SETS - reference only]" in prompt


def test_prompt_candidate_text_exposes_score_reasons(mock_candidates):
    ranked = score_candidate_exercises(mock_candidates, ["CHEST_MID"])
    rendered = format_candidates_for_prompt(ranked)
    assert "score=" in rendered
    assert "compound priority" in rendered

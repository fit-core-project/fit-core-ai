from engines.routine_engine import (
    PainAreaEntry,
    RecentSetRecord,
    UserProfileContext,
    _build_system_prompt,
    _format_request_pain_areas,
    _target_exercise_count,
    format_candidates_for_prompt,
    score_candidate_exercises,
)
from langchain_core.prompts import ChatPromptTemplate


def test_scoring_prefers_compound_primary_target_match(mock_candidates):
    ranked = score_candidate_exercises(mock_candidates, ["chest"])
    assert ranked[0]["id"] == "barbell_bench_press"
    assert "compound priority" in ranked[0]["score_reasons"]
    assert "primary target match" in ranked[0]["score_reasons"]


def test_scoring_limits_prompt_candidates_to_top_n(mock_candidates):
    expanded = [
        {**mock_candidates[i % len(mock_candidates)], "id": f"candidate_{i}"}
        for i in range(20)
    ]
    ranked = score_candidate_exercises(expanded, ["chest"], top_n=5)
    assert len(ranked) == 5


def test_scoring_penalizes_doms_and_excludes_blocked_candidates(mock_candidates):
    candidates = [
        {**mock_candidates[0], "id": "moderate_chest"},
        {**mock_candidates[1], "id": "blocked_equipment", "equipment_req": "MACHINE"},
        {**mock_candidates[2], "id": "safe_bodyweight"},
    ]
    ranked = score_candidate_exercises(
        candidates,
        ["chest"],
        doms_db={"chest": 2},
        blocked_equipment=["MACHINE"],
    )
    ids = [candidate["id"] for candidate in ranked]
    assert "blocked_equipment" not in ids
    assert "doms moderate penalty" in ranked[0]["score_reasons"]


def test_scoring_prefers_loadable_equipment_over_bodyweight():
    candidates = [
        {
            "id": "high_efficiency_pushup",
            "name_kr": "Push-up",
            "name_en": "Push-up",
            "primary_muscle": "chest",
            "secondary_muscle": "triceps",
            "equipment_req": "BODYWEIGHT",
            "efficiency_tier": 5,
            "movement_type": "COMPOUND",
            "pain_triggers": None,
        },
        {
            "id": "dumbbell_bench_press",
            "name_kr": "Dumbbell Bench Press",
            "name_en": "Dumbbell Bench Press",
            "primary_muscle": "chest",
            "secondary_muscle": "triceps",
            "equipment_req": "DUMBBELL",
            "efficiency_tier": 4,
            "movement_type": "COMPOUND",
            "pain_triggers": None,
        },
    ]

    ranked = score_candidate_exercises(candidates, ["chest"])

    assert ranked[0]["id"] == "dumbbell_bench_press"
    assert "loadable equipment priority" in ranked[0]["score_reasons"]
    assert "bodyweight deprioritized" in ranked[1]["score_reasons"]


def test_prompt_v2_mentions_hard_constraints_and_ranked_candidates():
    prompt = _build_system_prompt(profile=None, recent_sets=None)
    assert "[ROLE]" in prompt
    assert "[REQUEST CONTEXT]" in prompt
    assert "timeAvailableMin: {time_available_min}" in prompt
    assert "targetExerciseCount: {target_exercise_count}" in prompt
    assert "readinessLevel: {readiness_level}" in prompt
    assert "unavailable_equipment: {unavailable_equipment}" in prompt
    assert "target_split_label: {target_split_label}" in prompt
    assert "target_muscles: {target_muscles}" in prompt
    assert "current_pain_areas: {current_pain_areas}" in prompt
    assert "candidate_count: {candidate_count}" in prompt
    assert "[HARD CONSTRAINTS]" in prompt
    assert "[READINESS POLICY]" in prompt
    assert "[TIME POLICY]" in prompt
    assert "[EQUIPMENT POLICY]" in prompt
    assert "[SET ALLOCATION POLICY]" in prompt
    assert "[EXERCISE ORDER POLICY]" in prompt
    assert "[RATIONALE POLICY]" in prompt
    assert "Every exercise_rationale must cite at least one concrete input value" in prompt
    assert "Avoid generic explanations" in prompt
    assert "[OUTPUT SCHEMA]" in prompt
    assert "[PROHIBITED BEHAVIOR]" in prompt
    assert "[RANKED CANDIDATES]" in prompt
    assert "at or below {max_sets}" in prompt


def test_target_exercise_count_uses_time_bands():
    assert _target_exercise_count(30) == 4
    assert _target_exercise_count(45) == 5
    assert _target_exercise_count(60) == 6
    assert _target_exercise_count(75) == 7
    assert _target_exercise_count(90) == 8


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
    ranked = score_candidate_exercises(mock_candidates, ["chest"])
    rendered = format_candidates_for_prompt(ranked)
    assert "score=" in rendered
    assert "primary=chest" in rendered
    assert "secondary=triceps" in rendered
    assert "equipment=BARBELL" in rendered
    assert "pain_triggers=none" in rendered
    assert "movement_type=COMPOUND" in rendered
    assert "compound priority" in rendered
    assert "large muscle priority" in rendered


def test_rendered_prompt_snapshot_for_low_readiness_pain_and_short_time(mock_candidates):
    candidates = [
        {
            **mock_candidates[0],
            "pain_triggers": "chest, front-deltoids, triceps",
        },
        {
            **mock_candidates[1],
            "pain_triggers": "chest",
        },
        {
            **mock_candidates[2],
            "pain_triggers": "chest, triceps",
        },
    ]
    pain_areas = [PainAreaEntry(body_part="chest", side="left", severity="mild")]
    ranked = score_candidate_exercises(
        candidates,
        ["chest", "front-deltoids", "triceps"],
        pain_areas=pain_areas,
    )
    candidate_text = format_candidates_for_prompt(ranked)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _build_system_prompt(profile=None, recent_sets=None)),
        ("human", "User note: {user_note}\nUse only the request context and ranked candidates above."),
    ])

    rendered = prompt.format(
        goal="hypertrophy",
        max_sets=8,
        doms_instructions="none",
        candidate_exercises=candidate_text,
        user_note="가볍게 진행",
        time_available_min=30,
        target_exercise_count=4,
        readiness_level="low",
        unavailable_equipment="BARBELL",
        target_split_label="push",
        target_muscles="chest, front-deltoids, triceps",
        current_pain_areas=_format_request_pain_areas(pain_areas),
        candidate_count=len(ranked),
    )

    assert "timeAvailableMin: 30" in rendered
    assert "targetExerciseCount: 4" in rendered
    assert "readinessLevel: low" in rendered
    assert "unavailable_equipment: BARBELL" in rendered
    assert "target_muscles: chest, front-deltoids, triceps" in rendered
    assert "current_pain_areas: (chest, side=left, severity=mild)" in rendered
    assert "candidate_count: 0" in rendered
    assert "[READINESS POLICY]" in rendered
    assert "[TIME POLICY]" in rendered
    assert "Use only the request context and ranked candidates above." in rendered


def test_pain_slug_filter_excludes_chest_upper_back_and_trapezius():
    candidates = [
        {
            "id": "bench",
            "name_kr": "Bench",
            "primary_muscle": "chest",
            "secondary_muscle": "triceps",
            "equipment_req": "BARBELL",
            "efficiency_tier": 5,
            "movement_type": "COMPOUND",
            "pain_triggers": "chest, front-deltoids, triceps",
        },
        {
            "id": "row",
            "name_kr": "Row",
            "primary_muscle": "upper-back",
            "secondary_muscle": "trapezius, biceps",
            "equipment_req": "CABLE",
            "efficiency_tier": 4,
            "movement_type": "COMPOUND",
            "pain_triggers": "upper-back, trapezius, biceps",
        },
        {
            "id": "curl",
            "name_kr": "Curl",
            "primary_muscle": "biceps",
            "secondary_muscle": "forearm",
            "equipment_req": "DUMBBELL",
            "efficiency_tier": 3,
            "movement_type": "ISOLATION",
            "pain_triggers": "biceps, forearm",
        },
    ]

    ranked = score_candidate_exercises(
        candidates,
        ["chest", "upper-back", "trapezius", "biceps"],
        pain_areas=[
            PainAreaEntry(body_part="chest"),
            PainAreaEntry(body_part="upper-back"),
            PainAreaEntry(body_part="trapezius"),
        ],
    )

    assert [candidate["id"] for candidate in ranked] == ["curl"]


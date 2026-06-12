from engines.candidate_ranker import (
    format_candidates_for_prompt,
    score_candidate_exercises,
)
from engines.prescription.adjustments import _target_exercise_count
from engines.prompt_builder import (
    _format_request_pain_areas,
    build_system_prompt as _build_system_prompt,
)
from engines.schemas import (
    PainAreaEntry,
    RecentSetRecord,
    UserProfileContext,
)
from langchain_core.prompts import ChatPromptTemplate


def _section(prompt: str, title: str) -> str:
    start = prompt.index(title)
    next_start = prompt.find("\n\n[", start + len(title))
    return prompt[start:] if next_start == -1 else prompt[start:next_start]


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
    assert "[LANGUAGE POLICY]" in prompt
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


def test_prompt_requires_korean_user_facing_text():
    prompt = _build_system_prompt(profile=None, recent_sets=None)
    language_policy = _section(prompt, "[LANGUAGE POLICY]")

    assert "All user-facing natural language fields must be Korean." in language_policy
    assert "summary_title, rationale_summary, exercise_rationale, and warnings must be Korean." in language_policy
    assert "Do not output English explanations unless the field is an enum, id, equipment key, or muscle key." in language_policy


def test_prompt_weight_policy_delegates_prescription_to_server():
    prompt = _build_system_prompt(profile=None, recent_sets=None)

    assert "[WEIGHT PRESCRIPTION POLICY]" in prompt
    assert "Do not perform 1RM percentage calculations" in prompt
    assert "server overrides target_reps and rest_time_sec" in prompt
    assert "Always set target_weight_kg to null for BODYWEIGHT equipment" in prompt
    assert "target_reps, sets, and rest_time_sec are required schema fields" in prompt
    assert "your values serve as structural hints only" in prompt
    for removed in (
        "80-90 % of 1RM",
        "65-80 % of 1RM",
        "50-70 % of 1RM",
        "Round prescribed weights to the nearest 2.5 kg",
    ):
        assert removed not in prompt
    # These damaged tokens are intentionally listed only as regression sentinels.
    for damaged in ("??", "�", "횞", "80??0", "3??", "65??0", "8??2", "50??0", "12??0"):
        assert damaged not in prompt


def test_prohibited_behavior_removes_duplicate_hard_constraints():
    prompt = _build_system_prompt(profile=None, recent_sets=None)
    hard_constraints = _section(prompt, "[HARD CONSTRAINTS]")
    prohibited = _section(prompt, "[PROHIBITED BEHAVIOR]")

    assert "Use only exercise_id values that appear in [RANKED CANDIDATES]." in hard_constraints
    assert "Treat DOMS, pain, and equipment restrictions as hard constraints" in hard_constraints
    assert "Keep total working sets at or below {max_sets}." in hard_constraints

    assert "Never fabricate medical advice, user history, or unavailable rationale." in prohibited
    assert "Never choose an exercise outside [RANKED CANDIDATES]." not in prohibited
    assert "Never reintroduce excluded equipment or pain-triggering movements." not in prohibited
    assert "Never ignore DOMS instructions or exceed the working-set cap." not in prohibited


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
        user_note="light session",
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


def test_few_shot_section_exists():
    prompt = _build_system_prompt(profile=None, recent_sets=None)
    assert "[EXAMPLES]" in prompt
    examples = _section(prompt, "[EXAMPLES]")
    assert "Example 1" in examples
    assert "Example 2" in examples


def test_few_shot_candidate_id_isolation_note():
    prompt = _build_system_prompt(profile=None, recent_sets=None)
    examples = _section(prompt, "[EXAMPLES]")
    reminder = _section(prompt, "[FINAL SELECTION REMINDER]")
    assert "EXAMPLE_A" in examples
    assert "EXAMPLE_B" in examples
    assert "EXAMPLE_C" in examples
    assert "use only IDs from [RANKED CANDIDATES]" in examples
    assert "never use EXAMPLE_* ids" in reminder
    assert "Use only exercise_id values from [RANKED CANDIDATES]" in reminder


def test_few_shot_no_1rm_calculation():
    prompt = _build_system_prompt(profile=None, recent_sets=None)
    examples = _section(prompt, "[EXAMPLES]")
    assert "1RM" not in examples
    assert "%" not in examples


def test_few_shot_bodyweight_target_weight_null():
    prompt = _build_system_prompt(profile=None, recent_sets=None)
    examples = _section(prompt, "[EXAMPLES]")
    assert "BODYWEIGHT" in examples
    assert '"target_weight_kg":null' in examples


def test_few_shot_required_schema_fields_present():
    prompt = _build_system_prompt(profile=None, recent_sets=None)
    examples = _section(prompt, "[EXAMPLES]")
    for field in (
        "exercise_id",
        "exercise_name",
        "primary_muscles",
        "target_reps",
        "sets",
        "rest_time_sec",
        "exercise_rationale",
    ):
        assert field in examples, f"missing field: {field}"

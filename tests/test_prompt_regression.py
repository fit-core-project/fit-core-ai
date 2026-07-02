from engines.candidate_ranker import (
    format_candidates_for_prompt,
    score_candidate_exercises,
)
from engines.db_queries import get_user_constraint_profile_from_sqlite
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


def test_scoring_diversifies_near_duplicate_exercise_families():
    candidates = [
        {
            "id": f"kettlebell_deadlift_{index}",
            "name_kr": f"케틀벨 스탠딩 정면 싱글 데드리프트 v7-{index}",
            "name_en": f"Kettlebell Standing Single Deadlift v7-{index}",
            "primary_muscle": "hamstring",
            "secondary_muscle": "gluteal",
            "equipment_req": "KETTLEBELL",
            "efficiency_tier": 3,
            "movement_type": "COMPOUND",
            "pain_triggers": None,
            "hypertrophy_effect": "elite",
            "stimulus_to_fatigue": "good",
        }
        for index in range(8)
    ]
    candidates.extend([
        {
            "id": "leg_press_machine",
            "name_kr": "레그프레스 발 높게 프레스/스쿼트",
            "name_en": "Leg Press",
            "primary_muscle": "quadriceps",
            "secondary_muscle": "gluteal",
            "equipment_req": "MACHINE",
            "efficiency_tier": 3,
            "movement_type": "COMPOUND",
            "pain_triggers": None,
            "hypertrophy_effect": "high",
            "stimulus_to_fatigue": "good",
        },
        {
            "id": "seated_leg_curl",
            "name_kr": "시티드 레그 컬",
            "name_en": "Seated Leg Curl",
            "primary_muscle": "hamstring",
            "secondary_muscle": "",
            "equipment_req": "MACHINE",
            "efficiency_tier": 3,
            "movement_type": "ISOLATION",
            "pain_triggers": None,
            "hypertrophy_effect": "high",
            "stimulus_to_fatigue": "good",
        },
    ])

    ranked = score_candidate_exercises(candidates, ["quadriceps", "hamstring"], top_n=4)
    ids = [candidate["id"] for candidate in ranked]

    assert sum(candidate_id.startswith("kettlebell_deadlift") for candidate_id in ids) <= 2
    assert "leg_press_machine" in ids
    assert "seated_leg_curl" in ids


def test_scoring_diversifies_step_variants_across_equipment():
    candidates = []
    for equipment in ("SMITH_MACHINE, MACHINE", "DUMBBELL", "BARBELL", "CABLE", "KETTLEBELL"):
        for variant in ("상체 세움", "상체 전경사"):
            for movement in ("스텝다운", "스텝업"):
                candidates.append({
                    "id": f"{equipment}_{variant}_{movement}".replace(" ", "_"),
                    "name_kr": f"{equipment} {variant} {movement}",
                    "primary_muscle": "quadriceps",
                    "secondary_muscle": "gluteal",
                    "equipment_req": equipment,
                    "efficiency_tier": 3,
                    "movement_type": "COMPOUND",
                    "pain_triggers": None,
                    "injury_caution_level": "medium",
                    "lumbar_load": "low",
                    "axial_load": "low",
                    "hypertrophy_effect": "high",
                    "stimulus_to_fatigue": "excellent",
                })
    candidates.extend([
        {
            "id": "machine_leg_press",
            "name_kr": "브이 스쿼트 머신 발 높게 프레스/스쿼트",
            "primary_muscle": "quadriceps",
            "secondary_muscle": "gluteal",
            "equipment_req": "MACHINE",
            "efficiency_tier": 3,
            "movement_type": "COMPOUND",
            "pain_triggers": None,
            "injury_caution_level": "medium",
            "lumbar_load": "low",
            "axial_load": "low",
            "hypertrophy_effect": "high",
            "stimulus_to_fatigue": "excellent",
        },
        {
            "id": "seated_leg_curl",
            "name_kr": "시티드 레그 컬",
            "primary_muscle": "hamstring",
            "secondary_muscle": "",
            "equipment_req": "MACHINE",
            "efficiency_tier": 3,
            "movement_type": "ISOLATION",
            "pain_triggers": None,
            "injury_caution_level": "low",
            "lumbar_load": "low",
            "axial_load": "low",
            "hypertrophy_effect": "high",
            "stimulus_to_fatigue": "excellent",
        },
    ])

    ranked = score_candidate_exercises(
        candidates,
        ["quadriceps", "hamstring"],
        pain_areas=[PainAreaEntry(body_part="lower-back")],
        goal="hypertrophy",
        top_n=6,
    )
    ids = [candidate["id"] for candidate in ranked]

    assert sum("스텝" in candidate["name_kr"] for candidate in ranked[:4]) <= 2
    assert "machine_leg_press" in ids
    assert "seated_leg_curl" in ids


def test_scoring_penalizes_general_injury_caution_when_pain_context_exists():
    candidates = [
        {
            "id": "high_caution_stepdown",
            "name_kr": "고난도 스텝다운",
            "primary_muscle": "quadriceps",
            "secondary_muscle": "gluteal",
            "equipment_req": "DUMBBELL",
            "efficiency_tier": 3,
            "movement_type": "COMPOUND",
            "pain_triggers": "quadriceps",
            "injury_caution_level": "high",
            "lumbar_load": "low",
            "hypertrophy_effect": "high",
        },
        {
            "id": "low_caution_machine_press",
            "name_kr": "머신 레그 프레스",
            "primary_muscle": "quadriceps",
            "secondary_muscle": "gluteal",
            "equipment_req": "MACHINE",
            "efficiency_tier": 3,
            "movement_type": "COMPOUND",
            "pain_triggers": "quadriceps",
            "injury_caution_level": "low",
            "lumbar_load": "low",
            "hypertrophy_effect": "high",
        },
    ]

    ranked = score_candidate_exercises(
        candidates,
        ["quadriceps"],
        pain_areas=[PainAreaEntry(body_part="lower-back")],
        top_n=2,
    )

    assert ranked[0]["id"] == "low_caution_machine_press"
    high_caution = next(candidate for candidate in ranked if candidate["id"] == "high_caution_stepdown")
    assert "pain context high injury caution penalty" in high_caution["score_reasons"]


def test_scoring_demotes_sport_or_cardio_context_for_hypertrophy_main_lifts():
    candidates = [
        {
            "id": "treadmill_walk",
            "name_kr": "러닝머신 평지 걷기",
            "primary_muscle": "hamstring",
            "secondary_muscle": "",
            "equipment_req": "MACHINE",
            "efficiency_tier": 3,
            "movement_type": "COMPOUND",
            "pain_triggers": None,
            "injury_caution_level": "low",
            "hypertrophy_effect": "elite",
            "stimulus_to_fatigue": "excellent",
        },
        {
            "id": "basketball_landing",
            "name_kr": "농구 보강 싱글레그 착지 - 덤벨 부하",
            "primary_muscle": "quadriceps",
            "secondary_muscle": "",
            "equipment_req": "DUMBBELL",
            "efficiency_tier": 3,
            "movement_type": "COMPOUND",
            "pain_triggers": None,
            "injury_caution_level": "low",
            "hypertrophy_effect": "elite",
            "stimulus_to_fatigue": "excellent",
        },
        {
            "id": "machine_leg_press",
            "name_kr": "머신 레그 프레스",
            "primary_muscle": "quadriceps",
            "secondary_muscle": "gluteal",
            "equipment_req": "MACHINE",
            "efficiency_tier": 3,
            "movement_type": "COMPOUND",
            "pain_triggers": None,
            "injury_caution_level": "low",
            "hypertrophy_effect": "high",
            "stimulus_to_fatigue": "good",
        },
    ]

    ranked = score_candidate_exercises(candidates, ["quadriceps", "hamstring"], goal="hypertrophy", top_n=3)

    assert ranked[0]["id"] == "machine_leg_press"
    treadmill = next(candidate for candidate in ranked if candidate["id"] == "treadmill_walk")
    basketball = next(candidate for candidate in ranked if candidate["id"] == "basketball_landing")
    assert "sport cardio context accessory penalty" in treadmill["score_reasons"]
    assert "sport cardio context accessory penalty" in basketball["score_reasons"]


def test_scoring_promotes_lumbar_safe_low_load_alternatives_over_ballistic_context():
    candidates = [
        {
            "id": "treadmill_walk",
            "name_kr": "러닝머신 평지 걷기",
            "primary_muscle": "hamstring",
            "secondary_muscle": "",
            "equipment_req": "MACHINE",
            "efficiency_tier": 3,
            "movement_type": "COMPOUND",
            "pain_triggers": None,
            "injury_caution_level": "low",
            "lumbar_load": "low",
            "axial_load": "low",
            "hypertrophy_effect": "high",
            "stimulus_to_fatigue": "excellent",
        },
        {
            "id": "kettlebell_swing",
            "name_kr": "케틀벨 스탠딩 정면 싱글 스윙",
            "primary_muscle": "hamstring",
            "secondary_muscle": "gluteal",
            "equipment_req": "KETTLEBELL",
            "efficiency_tier": 3,
            "movement_type": "COMPOUND",
            "pain_triggers": None,
            "injury_caution_level": "medium",
            "lumbar_load": "low",
            "axial_load": "low",
            "hypertrophy_effect": "high",
            "stimulus_to_fatigue": "good",
        },
        {
            "id": "machine_leg_curl",
            "name_kr": "머신 라잉 레그 컬",
            "primary_muscle": "hamstring",
            "secondary_muscle": "",
            "equipment_req": "MACHINE",
            "efficiency_tier": 3,
            "movement_type": "ISOLATION",
            "pain_triggers": None,
            "injury_caution_level": "low",
            "lumbar_load": "low",
            "axial_load": "low",
            "hypertrophy_effect": "high",
            "stimulus_to_fatigue": "excellent",
        },
    ]

    ranked = score_candidate_exercises(
        candidates,
        ["hamstring"],
        pain_areas=[PainAreaEntry(body_part="lower-back")],
        goal="hypertrophy",
    )

    assert ranked[0]["id"] == "machine_leg_curl"
    swing = next(candidate for candidate in ranked if candidate["id"] == "kettlebell_swing")
    leg_curl = next(candidate for candidate in ranked if candidate["id"] == "machine_leg_curl")
    treadmill = next(candidate for candidate in ranked if candidate["id"] == "treadmill_walk")
    assert "pain context ballistic coordination penalty" in swing["score_reasons"]
    assert "lumbar safe low load alternative boost" in leg_curl["score_reasons"]
    assert "lumbar safe low load alternative boost" not in treadmill["score_reasons"]


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


def test_scoring_exposes_structured_boosts_and_penalties():
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

    ranked = score_candidate_exercises(candidates, ["chest"], doms_db={"chest": 1})
    dumbbell = next(candidate for candidate in ranked if candidate["id"] == "dumbbell_bench_press")
    pushup = next(candidate for candidate in ranked if candidate["id"] == "high_efficiency_pushup")

    assert dumbbell["score_breakdown_version"] == "v2"
    assert any(item["rule_code"] == "primary_target_match" and item["score"] == 20 for item in dumbbell["score_boosts"])
    assert any(item["rule_code"] == "doms_penalty" and item["profile_signal_code"] == "doms" for item in dumbbell["score_penalties"])
    assert any(item["rule_code"] == "bodyweight_deprioritized" and item["score"] == -18 for item in pushup["score_penalties"])
    assert all("priority" not in item["reason"] for item in dumbbell["score_boosts"])


def test_prompt_v2_mentions_hard_constraints_and_ranked_candidates():
    prompt = _build_system_prompt(profile=None, recent_sets=None)
    assert "[ROLE]" in prompt
    assert "[REQUEST CONTEXT]" in prompt
    assert "timeAvailableMin: {time_available_min}" in prompt
    assert "targetExerciseCount: {target_exercise_count}" in prompt
    assert "readinessLevel: {readiness_level}" in prompt
    assert "experienceLevel: {experience_level}" in prompt
    assert "unavailable_equipment: {unavailable_equipment}" in prompt
    assert "target_split_label: {target_split_label}" in prompt
    assert "target_muscles: {target_muscles}" in prompt
    assert "current_pain_areas: {current_pain_areas}" in prompt
    assert "mobility_limits: {mobility_limits}" in prompt
    assert "condition_policies: {condition_policies}" in prompt
    assert "anthropometry_signals: {anthropometry_signals}" in prompt
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


def test_prompt_v2_mentions_quality_selection_policy():
    prompt = _build_system_prompt(profile=None, recent_sets=None)
    quality_policy = _section(prompt, "[QUALITY SELECTION POLICY]")

    assert "score reasons" in quality_policy
    assert "near-duplicate exercise variations" in quality_policy
    assert "penalty/boost tradeoff" in quality_policy
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
    assert "Treat current_pain_areas and unavailable equipment as hard constraints" in hard_constraints
    assert "Treat DOMS level 3 as an exclusion constraint" in hard_constraints
    assert "DOMS level 1/2 as volume and priority constraints" in hard_constraints
    assert "Keep total working sets at or below {max_sets}." in hard_constraints

    assert "Never fabricate medical advice, user history, or unavailable rationale." in prohibited
    assert "Never choose an exercise outside [RANKED CANDIDATES]." not in prohibited
    assert "Never reintroduce excluded equipment or pain-triggering movements." not in prohibited
    assert "Never ignore DOMS instructions or exceed the working-set cap." not in prohibited


def test_target_exercise_count_uses_time_bands():
    assert _target_exercise_count(30) == 3
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


def test_scoring_uses_lumbar_condition_policy():
    candidates = [
        {
            "id": "barbell_good_morning",
            "name_kr": "바벨 굿모닝",
            "name_en": "Barbell Good Morning",
            "primary_muscle": "hamstring",
            "secondary_muscle": "gluteal",
            "equipment_req": "BARBELL",
            "efficiency_tier": 3,
            "movement_type": "COMPOUND",
            "pain_triggers": None,
            "lumbar_load": "high",
            "axial_load": "medium",
            "failure_penalty": "high",
        },
        {
            "id": "seated_leg_curl",
            "name_kr": "시티드 레그 컬",
            "name_en": "Seated Leg Curl",
            "primary_muscle": "hamstring",
            "secondary_muscle": None,
            "equipment_req": "MACHINE",
            "efficiency_tier": 3,
            "movement_type": "ISOLATION",
            "pain_triggers": None,
            "lumbar_load": "low",
            "axial_load": "low",
            "failure_penalty": "low",
        },
    ]

    ranked = score_candidate_exercises(
        candidates,
        ["hamstring"],
        condition_policies=[
            {
                "conditionCode": "lumbar_herniation",
                "bodyPart": "lower-back",
                "phase": "rehab",
                "professionalClearance": False,
            }
        ],
    )

    assert ranked[0]["id"] == "seated_leg_curl"
    penalized = next(candidate for candidate in ranked if candidate["id"] == "barbell_good_morning")
    assert any(item["rule_code"] == "lumbar_load_constraint_penalty" for item in penalized["score_penalties"])
    assert any(item["rule_code"] == "professional_clearance_guard" for item in penalized["score_penalties"])


def test_scoring_uses_beginner_experience_for_low_technique_candidates():
    candidates = [
        {
            "id": "barbell_back_squat",
            "name_kr": "바벨 백스쿼트",
            "name_en": "Barbell Back Squat",
            "primary_muscle": "quadriceps",
            "secondary_muscle": "gluteal",
            "equipment_req": "BARBELL",
            "efficiency_tier": 3,
            "movement_type": "COMPOUND",
            "pain_triggers": None,
            "technical_difficulty": "high",
            "failure_penalty": "high",
            "strength_effect": "elite",
            "stimulus_to_fatigue": "good",
        },
        {
            "id": "v_squat_machine",
            "name_kr": "브이 스쿼트 머신",
            "name_en": "V Squat Machine",
            "primary_muscle": "quadriceps",
            "secondary_muscle": "gluteal",
            "equipment_req": "MACHINE",
            "efficiency_tier": 3,
            "movement_type": "COMPOUND",
            "pain_triggers": None,
            "technical_difficulty": "low",
            "balance_requirement": "low",
            "failure_penalty": "low",
            "strength_effect": "high",
            "stimulus_to_fatigue": "excellent",
        },
    ]

    ranked = score_candidate_exercises(
        candidates,
        ["quadriceps"],
        goal="strength",
        experience_level="beginner",
    )

    assert ranked[0]["id"] == "v_squat_machine"
    machine = ranked[0]
    squat = next(candidate for candidate in ranked if candidate["id"] == "barbell_back_squat")
    assert any(item["rule_code"] == "beginner_low_technique_risk_boost" for item in machine["score_boosts"])
    assert any(item["rule_code"] == "beginner_progression_path_boost" for item in machine["score_boosts"])
    assert any(item["rule_code"] == "beginner_failure_penalty" for item in squat["score_penalties"])
    assert any(item["rule_code"] == "beginner_technical_difficulty_penalty" for item in squat["score_penalties"])


def test_scoring_marks_clearance_safe_alternative_candidates():
    candidates = [
        {
            "id": "overhead_press",
            "name_kr": "바벨 오버헤드 프레스",
            "name_en": "Barbell Overhead Press",
            "primary_muscle": "front-deltoids",
            "secondary_muscle": "triceps",
            "equipment_req": "BARBELL",
            "efficiency_tier": 3,
            "movement_type": "COMPOUND",
            "pain_triggers": None,
            "injury_caution_level": "high",
            "shoulder_impingement_risk": "high",
            "failure_penalty": "high",
            "hypertrophy_effect": "high",
            "stimulus_to_fatigue": "good",
        },
        {
            "id": "neutral_machine_row",
            "name_kr": "뉴트럴 머신 로우",
            "name_en": "Neutral Machine Row",
            "primary_muscle": "upper-back",
            "secondary_muscle": "back-deltoids",
            "equipment_req": "MACHINE",
            "efficiency_tier": 3,
            "movement_type": "COMPOUND",
            "pain_triggers": None,
            "injury_caution_level": "low",
            "pain_risk_level": "low",
            "lumbar_load": "low",
            "axial_load": "low",
            "shoulder_impingement_risk": "low",
            "wrist_stress": "low",
            "failure_penalty": "low",
            "hypertrophy_effect": "elite",
            "stimulus_to_fatigue": "excellent",
        },
    ]

    ranked = score_candidate_exercises(
        candidates,
        ["front-deltoids", "upper-back", "back-deltoids"],
        goal="hypertrophy",
        condition_policies=[{
            "policyType": "surgeryHistory",
            "bodyPart": "shoulder",
            "status": "needs_clearance",
            "professionalClearance": False,
        }],
    )

    assert ranked[0]["id"] == "neutral_machine_row"
    safe = ranked[0]
    overhead = next(candidate for candidate in ranked if candidate["id"] == "overhead_press")
    assert any(item["rule_code"] == "professional_clearance_safe_alternative" for item in safe["score_boosts"])
    assert any(item["rule_code"] == "professional_clearance_guard" for item in overhead["score_penalties"])


def test_scoring_uses_mobility_limit_goal_effect_and_anthropometry():
    candidates = [
        {
            "id": "deep_free_squat",
            "name_kr": "딥 프리 스쿼트",
            "name_en": "Deep Free Squat",
            "primary_muscle": "quadriceps",
            "secondary_muscle": "gluteal",
            "equipment_req": "BARBELL",
            "efficiency_tier": 3,
            "movement_type": "COMPOUND",
            "pain_triggers": None,
            "ankle_dorsiflexion_demand": "high",
            "long_femur_sensitivity": "high",
            "hypertrophy_effect": "high",
            "stimulus_to_fatigue": "fair",
        },
        {
            "id": "leg_press",
            "name_kr": "레그프레스",
            "name_en": "Leg Press",
            "primary_muscle": "quadriceps",
            "secondary_muscle": "gluteal",
            "equipment_req": "MACHINE",
            "efficiency_tier": 3,
            "movement_type": "COMPOUND",
            "pain_triggers": None,
            "ankle_dorsiflexion_demand": "low",
            "long_femur_sensitivity": "low",
            "hypertrophy_effect": "elite",
            "stimulus_to_fatigue": "excellent",
        },
    ]

    ranked = score_candidate_exercises(
        candidates,
        ["quadriceps"],
        goal="hypertrophy",
        mobility_limits=["limited_ankle_dorsiflexion"],
        anthropometry_signals={"femurLengthLevel": "long"},
    )

    assert ranked[0]["id"] == "leg_press"
    squat = next(candidate for candidate in ranked if candidate["id"] == "deep_free_squat")
    leg_press = next(candidate for candidate in ranked if candidate["id"] == "leg_press")
    assert any(item["rule_code"] == "mobility_constraint_penalty" for item in squat["score_penalties"])
    assert any(item["rule_code"] == "anthropometry_sensitivity_penalty" for item in squat["score_penalties"])
    assert any(item["rule_code"] == "goal_effect_match" for item in leg_press["score_boosts"])
    assert any(item["rule_code"] == "stimulus_to_fatigue" for item in leg_press["score_boosts"])


def test_prompt_candidate_text_exposes_catalog_extensions_and_substitutions():
    ranked = score_candidate_exercises(
        [
            {
                "id": "leg_press",
                "name_kr": "레그프레스",
                "name_en": "Leg Press",
                "primary_muscle": "quadriceps",
                "secondary_muscle": "gluteal",
                "equipment_req": "MACHINE",
                "efficiency_tier": 3,
                "movement_type": "COMPOUND",
                "pain_triggers": "knees",
                "injury_caution_level": "medium",
                "pain_risk_level": "low",
                "lumbar_load": "low",
                "axial_load": "low",
                "knee_shear_load": "medium",
                "shoulder_impingement_risk": "low",
                "wrist_stress": "low",
                "ankle_dorsiflexion_demand": "low",
                "hip_flexion_demand": "medium",
                "shoulder_flexion_demand": "low",
                "wrist_extension_demand": "low",
                "hypertrophy_effect": "elite",
                "strength_effect": "high",
                "stimulus_to_fatigue": "excellent",
                "technical_difficulty": "low",
                "balance_requirement": "low",
                "failure_penalty": "medium",
                "substitution_options": [
                    {
                        "exercise_name": "시티드 레그 프레스",
                        "relation_type": "mobilityAlternative",
                        "constraint_code": "limited_ankle_dorsiflexion",
                    }
                ],
            }
        ],
        ["quadriceps"],
    )

    rendered = format_candidates_for_prompt(ranked)

    assert "caution=injury:" in rendered
    assert "jointLoad=lumbar:low" in rendered
    assert "mobility=ankle:low" in rendered
    assert "effect=hypertrophy:elite" in rendered
    assert "difficulty=technical:low" in rendered
    assert "시티드 레그 프레스[mobilityAlternative:limited_ankle_dorsiflexion]" in rendered


def test_user_constraint_profile_reads_sqlite_seed_user():
    profile = get_user_constraint_profile_from_sqlite("test_user")

    assert "limited_ankle_dorsiflexion" in profile["mobility_limits"]
    assert profile["anthropometry_signals"]["femur_length_level"] == "long"
    assert profile["condition_policies"][0]["conditionCode"] == "lumbar_herniation"
    assert profile["condition_policies"][0]["professionalClearance"] is False


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
        target_exercise_count=3,
        readiness_level="low",
        experience_level="none",
        unavailable_equipment="BARBELL",
        target_split_label="push",
        target_muscles="chest, front-deltoids, triceps",
        current_pain_areas=_format_request_pain_areas(pain_areas),
        mobility_limits="none",
        condition_policies="none",
        anthropometry_signals="none",
        candidate_count=len(ranked),
    )

    assert "timeAvailableMin: 30" in rendered
    assert "targetExerciseCount: 3" in rendered
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


def test_pain_slug_filter_excludes_direct_primary_even_without_pain_triggers():
    candidates = [
        {
            "id": "machine_fly",
            "name_kr": "Machine Fly",
            "primary_muscle": "chest",
            "secondary_muscle": "front-deltoids",
            "equipment_req": "MACHINE",
            "efficiency_tier": 3,
            "movement_type": "COMPOUND",
            "pain_triggers": None,
        },
        {
            "id": "pushdown",
            "name_kr": "Pushdown",
            "primary_muscle": "triceps",
            "secondary_muscle": "chest",
            "equipment_req": "CABLE",
            "efficiency_tier": 3,
            "movement_type": "ISOLATION",
            "pain_triggers": None,
        },
    ]

    ranked = score_candidate_exercises(
        candidates,
        ["chest", "triceps"],
        pain_areas=[PainAreaEntry(body_part="chest")],
        top_n=5,
    )

    assert [candidate["id"] for candidate in ranked] == ["pushdown"]
    assert any(
        "secondary pain area overlap chest" in reason
        for reason in ranked[0]["score_reasons"]
    )


def test_doms_1_and_2_are_penalties_but_doms_3_is_exclusion():
    candidates = [
        {
            "id": "chest_press",
            "name_kr": "체스트 프레스",
            "primary_muscle": "chest",
            "secondary_muscle": "triceps",
            "equipment_req": "MACHINE",
            "efficiency_tier": 3,
            "movement_type": "COMPOUND",
            "pain_triggers": None,
        }
    ]

    mild = score_candidate_exercises(candidates, ["chest"], doms_db={"chest": 1})
    moderate = score_candidate_exercises(candidates, ["chest"], doms_db={"chest": 2})
    severe = score_candidate_exercises(candidates, ["chest"], doms_db={"chest": 3})

    assert [candidate["id"] for candidate in mild] == ["chest_press"]
    assert [candidate["id"] for candidate in moderate] == ["chest_press"]
    assert "doms mild penalty" in mild[0]["score_reasons"]
    assert "doms moderate penalty" in moderate[0]["score_reasons"]
    assert severe == []


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

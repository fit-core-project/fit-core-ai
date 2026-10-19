import json

from engines.candidate_ranker import (
    _diversify_ranked_candidates,
    _ensure_requested_adjunct_option,
    candidate_selection_family_key,
    candidate_selection_pattern_key,
)
from engines.exercise_display import candidate_display_name
from engines.prescription.estimator import estimate_routine_time_min
from engines.routine_blueprint import build_blueprint_candidate_bundles, build_routine_blueprint
from engines.routine_selection import (
    apply_blueprint_execution_policy,
    adapt_slot_selection_output,
    adapt_structured_routine_response,
    ensure_antagonist_superset_pair,
    normalize_routine_selection_response,
    required_main_pattern_targets,
    routine_output_has_required_main_pattern,
    selection_pattern_limit,
)
from engines.schemas import LLMRoutineOutput, LLMRoutineSelectionOutput


def _candidate(
    exercise_id: str,
    name: str,
    primary: str,
    movement_type: str,
    score: int,
    equipment_req: str = "MACHINE",
) -> dict:
    return {
        "id": exercise_id,
        "name_kr": name,
        "primary_muscle": primary,
        "movement_type": movement_type,
        "equipment_req": equipment_req,
        "score": score,
    }


def _push_context():
    blueprint = build_routine_blueprint(
        target_split_label="push",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="beginner",
        readiness_level="normal",
        time_available_min=30,
    )
    candidates = [
        _candidate("bench_press", "머신 체스트 프레스", "chest", "COMPOUND", 120),
        _candidate("cable_pushdown", "케이블 푸시다운", "triceps", "ISOLATION", 110),
        _candidate("front_raise", "머신 프론트 레이즈", "front-deltoids", "ISOLATION", 90),
    ]
    bundles = build_blueprint_candidate_bundles(blueprint, candidates, top_k_per_slot=1)
    return blueprint, candidates, bundles


def test_required_main_pattern_rejects_accessory_only_push_selection():
    blueprint, candidates, _bundles = _push_context()
    accessory_only = LLMRoutineOutput(
        total_estimated_time=15,
        summary_title="보조 운동만 있는 잘못된 루틴",
        rationale_summary=[],
        warnings=[],
        exercises=[],
    )
    accessory_only.exercises = [
        adapt_slot_selection_output(
            LLMRoutineSelectionOutput.model_validate({
                "summaryTitle": "보조 운동",
                "rationaleSummary": [],
                "selections": [{
                    "slotId": "triceps_direct",
                    "exerciseId": "cable_pushdown",
                    "exerciseRationale": "삼두 보조",
                }],
                "warnings": [],
            }),
            blueprint=blueprint,
            slot_candidate_bundles=build_blueprint_candidate_bundles(
                blueprint,
                candidates[1:],
                top_k_per_slot=1,
            ),
            ranked_candidates=candidates[1:],
        ).exercises[0]
    ]

    assert required_main_pattern_targets(blueprint) == ("chest",)
    assert routine_output_has_required_main_pattern(
        accessory_only,
        blueprint,
        candidates,
    ) is False


def test_required_main_pattern_accepts_candidate_owned_chest_compound():
    blueprint, candidates, bundles = _push_context()
    output = adapt_slot_selection_output(
        LLMRoutineSelectionOutput.model_validate({
            "summaryTitle": "정상 푸시 루틴",
            "rationaleSummary": [],
            "selections": [{
                "slotId": "push_main",
                "exerciseId": "bench_press",
                "exerciseRationale": "가슴 메인",
            }],
            "warnings": [],
        }),
        blueprint=blueprint,
        slot_candidate_bundles=bundles,
        ranked_candidates=candidates,
    )

    assert routine_output_has_required_main_pattern(output, blueprint, candidates) is True


def test_required_main_pattern_accepts_blueprint_allowed_posterior_isolation():
    blueprint = build_routine_blueprint(
        target_split_label="legs",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="beginner",
        readiness_level="low",
        time_available_min=45,
    )
    candidates = [
        _candidate("frog_pump", "프로그 펌프", "gluteal", "ISOLATION", 120),
    ]
    bundles = build_blueprint_candidate_bundles(blueprint, candidates, top_k_per_slot=1)
    output = adapt_slot_selection_output(
        LLMRoutineSelectionOutput.model_validate({
            "summaryTitle": "무릎 부담을 피한 하체 루틴",
            "rationaleSummary": [],
            "selections": [{
                "slotId": "posterior_chain",
                "exerciseId": "frog_pump",
                "exerciseRationale": "둔근 중심 후면 사슬 운동",
            }],
            "warnings": [],
        }),
        blueprint=blueprint,
        slot_candidate_bundles=bundles,
        ranked_candidates=candidates,
    )

    assert routine_output_has_required_main_pattern(output, blueprint, candidates) is True


def test_final_selection_family_ignores_equipment_and_grip_variants():
    machine_pushdown = _candidate(
        "machine_pushdown",
        "머신 로프 푸시다운",
        "triceps",
        "ISOLATION",
        120,
    )
    cable_pushdown = _candidate(
        "cable_pushdown",
        "케이블 스탠딩 V바 푸시다운",
        "triceps",
        "ISOLATION",
        110,
    )
    machine_fly = _candidate(
        "machine_fly",
        "머신 인클라인 플라이",
        "front-deltoids",
        "COMPOUND",
        100,
    )
    dumbbell_fly = _candidate(
        "dumbbell_fly",
        "덤벨 인클라인 뉴트럴 그립 플라이",
        "front-deltoids",
        "COMPOUND",
        90,
    )
    flat_press = _candidate(
        "flat_press",
        "스미스 머신 플랫 체스트 프레스",
        "chest",
        "COMPOUND",
        80,
    )
    incline_press = _candidate(
        "incline_press",
        "덤벨 인클라인 체스트 프레스",
        "chest",
        "COMPOUND",
        70,
    )
    smith_squat = _candidate(
        "smith_squat",
        "스미스 머신 스쿼트",
        "quadriceps",
        "COMPOUND",
        60,
    )
    wide_squat = _candidate(
        "wide_squat",
        "덤벨 와이드 스쿼트",
        "adductor",
        "COMPOUND",
        50,
    )
    side_lunge = _candidate(
        "side_lunge",
        "덤벨 사이드 런지",
        "adductor",
        "COMPOUND",
        40,
    )
    dumbbell_row = _candidate(
        "dumbbell_row",
        "덤벨 벤트오버 뉴트럴 팔꿈치 45도 로우",
        "upper-back",
        "COMPOUND",
        30,
    )
    smith_row = _candidate(
        "smith_row",
        "스미스 머신 벤트오버 뉴트럴 로우투하이 로우",
        "upper-back",
        "COMPOUND",
        20,
    )
    machine_hammer_curl = _candidate(
        "machine_hammer_curl",
        "머신 해머 컬",
        "biceps",
        "ISOLATION",
        15,
    )
    dumbbell_crossbody_curl = _candidate(
        "dumbbell_crossbody_curl",
        "덤벨 크로스바디 해머 컬",
        "biceps",
        "ISOLATION",
        10,
    )
    machine_standard_curl = _candidate(
        "machine_standard_curl",
        "머신 컬",
        "biceps",
        "ISOLATION",
        9,
    )
    machine_back_extension = _candidate(
        "machine_back_extension",
        "머신 백 익스텐션",
        "lower-back",
        "ISOLATION",
        8,
    )
    dumbbell_back_extension = _candidate(
        "dumbbell_back_extension",
        "덤벨 백 익스텐션",
        "lower-back",
        "COMPOUND",
        5,
    )

    assert candidate_selection_family_key(machine_pushdown) == candidate_selection_family_key(cable_pushdown)
    assert candidate_selection_family_key(machine_fly) == candidate_selection_family_key(dumbbell_fly)
    assert candidate_selection_family_key(flat_press) != candidate_selection_family_key(incline_press)
    assert candidate_selection_family_key(smith_squat) == candidate_selection_family_key(wide_squat)
    assert candidate_selection_family_key(smith_squat) != candidate_selection_family_key(side_lunge)
    assert candidate_selection_family_key(dumbbell_row) == candidate_selection_family_key(smith_row)
    assert candidate_selection_family_key(machine_hammer_curl) == candidate_selection_family_key(dumbbell_crossbody_curl)
    assert candidate_selection_family_key(machine_hammer_curl) != candidate_selection_family_key(machine_standard_curl)
    assert candidate_selection_family_key(machine_back_extension) == candidate_selection_family_key(dumbbell_back_extension)


def test_coarse_pattern_groups_variations_that_should_not_fill_a_routine():
    flat_press = _candidate("flat", "머신 플랫 체스트 프레스", "chest", "COMPOUND", 120)
    incline_press = _candidate("incline", "덤벨 인클라인 체스트 프레스", "chest", "COMPOUND", 110)
    hammer_curl = _candidate("hammer", "덤벨 해머 컬", "biceps", "ISOLATION", 100)
    preacher_curl = _candidate("preacher", "머신 프리처 컬", "biceps", "ISOLATION", 90)

    assert candidate_selection_family_key(flat_press) != candidate_selection_family_key(incline_press)
    assert candidate_selection_pattern_key(flat_press) == "horizontal_press"
    assert candidate_selection_pattern_key(incline_press) == "horizontal_press"
    assert candidate_selection_pattern_key(hammer_curl) == "biceps_curl"
    assert candidate_selection_pattern_key(preacher_curl) == "biceps_curl"


def test_flex_fill_prefers_explainable_underfill_over_a_third_chest_press():
    blueprint = build_routine_blueprint(
        target_split_label="push",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=50,
    )
    candidates = [
        _candidate("flat", "머신 플랫 체스트 프레스", "chest", "COMPOUND", 130),
        _candidate("incline", "덤벨 인클라인 체스트 프레스", "chest", "COMPOUND", 125),
        _candidate("decline", "스미스 머신 디클라인 체스트 프레스", "chest", "COMPOUND", 120),
        _candidate("pushdown", "케이블 푸시다운", "triceps", "ISOLATION", 115),
    ]
    bundles = build_blueprint_candidate_bundles(blueprint, candidates, top_k_per_slot=4)
    output = adapt_slot_selection_output(
        LLMRoutineSelectionOutput.model_validate({
            "selections": [
                {"slotId": "push_main", "exerciseId": "flat"},
                {"slotId": "push_secondary", "exerciseId": "incline"},
                {"slotId": "triceps_direct", "exerciseId": "pushdown"},
                {"slotId": "flex", "exerciseId": "decline"},
            ]
        }),
        blueprint=blueprint,
        slot_candidate_bundles=bundles,
        ranked_candidates=candidates,
    )

    horizontal_press_count = sum(
        candidate_selection_pattern_key(next(item for item in candidates if item["id"] == exercise.exercise_id))
        == "horizontal_press"
        for exercise in output.exercises
    )
    assert horizontal_press_count == 2
    assert len(output.exercises) < blueprint.minimum_exercise_count
    assert output._selection_diagnostics["rejected_selection_count"] >= 1


def test_chest_specialization_allows_flat_and_incline_press_roles():
    blueprint = build_routine_blueprint(
        target_split_label="chest",
        target_muscles=["chest"],
        goal="strength",
        experience_level="advanced",
        readiness_level="normal",
        time_available_min=60,
    )

    assert selection_pattern_limit("horizontal_press", blueprint) == 2


def test_two_adjunct_slots_select_distinct_candidates_in_phase_order():
    blueprint = build_routine_blueprint(
        target_split_label="upper",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=60,
        adjunct_options=["mobility_warmup", "cooldown_stretch"],
    )
    candidates = [
        _candidate("press", "머신 체스트 프레스", "chest", "COMPOUND", 140),
        _candidate("row", "머신 시티드 로우", "upper-back", "COMPOUND", 130),
        {
            **_candidate("mobility", "동적 어깨 모빌리티", "front-deltoids", "STATIC", 120),
            "adjunct_option_codes": ["mobility_warmup"],
            "adjunct_catalog_review_status": "core-reviewed",
        },
        {
            **_candidate("cooldown", "상체 쿨다운 스트레치", "chest", "STATIC", 110),
            "adjunct_option_codes": ["cooldown_stretch"],
            "adjunct_catalog_review_status": "core-reviewed",
        },
    ]
    bundles = build_blueprint_candidate_bundles(blueprint, candidates, top_k_per_slot=1)
    selections = [
        {
            "slotId": bundle["slotId"],
            "exerciseId": bundle["candidates"][0]["exerciseId"],
        }
        for bundle in bundles
        if bundle["candidates"] and bundle["slotId"] != "flex"
    ]

    output = adapt_slot_selection_output(
        LLMRoutineSelectionOutput.model_validate({"selections": selections}),
        blueprint=blueprint,
        slot_candidate_bundles=bundles,
        ranked_candidates=candidates,
    )

    selected_ids = [exercise.exercise_id for exercise in output.exercises]
    assert "mobility" in selected_ids
    assert "cooldown" in selected_ids
    assert output._selection_phase_by_exercise_id["mobility"] == "warmup_or_preparation"
    assert output._selection_phase_by_exercise_id["cooldown"] == "cooldown_recovery"
    assert output._adjunct_option_by_exercise_id == {
        "mobility": "mobility_warmup",
        "cooldown": "cooldown_stretch",
    }


def test_candidate_pool_reserves_one_distinct_candidate_for_each_planned_adjunct():
    main = _candidate("main", "머신 체스트 프레스", "chest", "COMPOUND", 150)
    filler = _candidate("filler", "케이블 플라이", "chest", "ISOLATION", 140)
    mobility = {
        **_candidate("mobility", "동적 어깨 모빌리티", "front-deltoids", "STATIC", 80),
        "adjunct_option_codes": ["mobility_warmup"],
        "adjunct_catalog_review_status": "core-reviewed",
    }
    cooldown = {
        **_candidate("cooldown", "상체 쿨다운 스트레치", "chest", "STATIC", 70),
        "adjunct_option_codes": ["cooldown_stretch"],
        "adjunct_catalog_review_status": "core-reviewed",
    }

    selected = _ensure_requested_adjunct_option(
        selected=[main, filler],
        sorted_candidates=[main, filler, mobility, cooldown],
        adjunct_options=["cooldown_stretch", "mobility_warmup"],
        top_n=4,
    )

    assert {candidate["id"] for candidate in selected} == {
        "main",
        "filler",
        "mobility",
        "cooldown",
    }


def test_candidate_pool_preserves_adjunct_and_working_set_variants_of_same_family():
    adjunct_curl = _candidate(
        "prehab_hammer_curl",
        "손목 팔꿈치 프리햅 해머 컬 이소메트릭",
        "biceps",
        "ISOLATION",
        200,
    )
    adjunct_curl["score_boosts"] = [
        {
            "rule_code": "adjunct_option_match_boost",
            "constraint_code": "prehab_accessory",
        }
    ]
    working_curl = _candidate(
        "machine_hammer_curl",
        "머신 해머 컬",
        "biceps",
        "ISOLATION",
        180,
    )

    selected = _diversify_ranked_candidates(
        [adjunct_curl, working_curl],
        top_n=2,
        target_muscles=["biceps"],
    )

    assert [candidate["id"] for candidate in selected] == [
        "prehab_hammer_curl",
        "machine_hammer_curl",
    ]


def test_requested_adjunct_does_not_consume_main_primary_distribution_cap():
    adjunct_curl = _candidate(
        "prehab_hammer_curl",
        "프리행 해머 컬 이소메트릭",
        "biceps",
        "ISOLATION",
        220,
    )
    adjunct_curl["score_boosts"] = [
        {
            "rule_code": "adjunct_option_match_boost",
            "constraint_code": "prehab_accessory",
        }
    ]
    candidates = [
        adjunct_curl,
        _candidate("machine_hammer", "머신 해머 컬", "biceps", "ISOLATION", 210),
        _candidate("machine_drag", "머신 드래그 컬", "biceps", "ISOLATION", 200),
        _candidate("lat_pull", "머신 랫풀다운", "upper-back", "COMPOUND", 190),
        _candidate("rear_fly", "머신 리버스 플라이", "back-deltoids", "ISOLATION", 180),
        _candidate("shrug", "덤벨 슈러그", "trapezius", "COMPOUND", 170),
    ]

    selected = _diversify_ranked_candidates(
        candidates,
        top_n=6,
        target_muscles=["upper-back", "trapezius", "biceps", "forearm", "back-deltoids"],
    )

    assert {"prehab_hammer_curl", "machine_hammer", "machine_drag"} <= {
        candidate["id"] for candidate in selected
    }


def test_candidate_display_name_keeps_meaningful_curl_variant():
    candidate = _candidate(
        "machine_drag_curl",
        "머신 시티드 드래그 컬",
        "biceps",
        "ISOLATION",
        180,
    )
    candidate["user_facing_display_name_kr"] = "머신 컬"

    assert candidate_display_name(candidate) == "머신 시티드 드래그 컬"


def test_candidate_display_name_keeps_explicit_paused_press_method():
    candidate = _candidate(
        "paused_press",
        "바벨 플랫 일반그립 하단 정지 체스트 프레스",
        "chest",
        "COMPOUND",
        180,
        "BARBELL",
    )
    candidate["user_facing_display_name_kr"] = "바벨 플랫 체스트 프레스"

    assert candidate_display_name(candidate) == "바벨 플랫 일반그립 하단 정지 체스트 프레스"


def test_candidate_display_name_uses_reviewed_public_name_for_default_routine_row():
    candidate = _candidate(
        "reviewed_smith_squat",
        "스미스 머신 스모 패럴렐 백 스쿼트",
        "quadriceps",
        "COMPOUND",
        180,
    )
    candidate["user_facing_display_name_kr"] = "스미스 머신 백 스쿼트"
    candidate["main_routine_default_allowed"] = 1

    assert candidate_display_name(candidate) == "스미스 머신 백 스쿼트"


def test_candidate_display_name_keeps_searchable_thread_the_needle_alias():
    candidate = {
        "id": "V24ROW-023032",
        "name_kr": "스레드 더 니들 - 동적 반복",
        "user_facing_display_name_kr": "네발기기 흉추 회전 스트레칭",
        "primary_muscle": "upper-back",
    }

    assert candidate_display_name(candidate) == "스레드 더 니들(네발기기 흉추 회전)"


def test_family_key_uses_detailed_name_when_public_name_is_generic():
    hammer = _candidate("hammer", "머신 해머 컬", "biceps", "ISOLATION", 180)
    hammer["user_facing_display_name_kr"] = "머신 컬"
    drag = _candidate("drag", "머신 드래그 컬", "biceps", "ISOLATION", 170)
    drag["user_facing_display_name_kr"] = "머신 컬"

    assert candidate_selection_family_key(hammer) == "biceps_curl:hammer"
    assert candidate_selection_family_key(drag) == "biceps_curl:drag"


def test_blueprint_candidate_bundle_hides_same_family_equipment_alternatives():
    blueprint = build_routine_blueprint(
        target_split_label="push",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=45,
    )
    candidates = [
        _candidate("bench", "머신 체스트 프레스", "chest", "COMPOUND", 200),
        _candidate("machine_pushdown", "머신 푸시다운", "triceps", "ISOLATION", 190),
        _candidate("cable_pushdown", "케이블 푸시다운", "triceps", "ISOLATION", 180),
        _candidate("킥백", "덤벨 킥백", "triceps", "ISOLATION", 170),
    ]

    bundles = build_blueprint_candidate_bundles(blueprint, candidates, top_k_per_slot=3)
    triceps = next(bundle for bundle in bundles if bundle["slotId"] == "triceps_direct")

    assert [candidate["exerciseId"] for candidate in triceps["candidates"]] == [
        "machine_pushdown",
        "킥백",
    ]
    flex = next(bundle for bundle in bundles if bundle["slotId"] == "flex")
    assert [candidate["exerciseId"] for candidate in flex["candidates"]].count(
        "machine_pushdown"
    ) == 1
    assert "cable_pushdown" not in {
        candidate["exerciseId"] for candidate in flex["candidates"]
    }


def test_target_fill_skips_duplicate_movement_family_and_uses_distinct_flex_candidates():
    blueprint = build_routine_blueprint(
        target_split_label="push",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=45,
    )
    candidates = [
        _candidate("bench", "스미스 머신 플랫 체스트 프레스", "chest", "COMPOUND", 140),
        _candidate("incline_press", "덤벨 인클라인 체스트 프레스", "chest", "COMPOUND", 130),
        _candidate("machine_pushdown", "머신 로프 푸시다운", "triceps", "ISOLATION", 120),
        _candidate("cable_pushdown", "케이블 스탠딩 V바 푸시다운", "triceps", "ISOLATION", 115),
        _candidate("front_raise", "머신 프론트 레이즈", "front-deltoids", "ISOLATION", 110),
        _candidate("lateral_raise", "덤벨 레터럴 레이즈", "front-deltoids", "ISOLATION", 105),
    ]
    bundles = build_blueprint_candidate_bundles(blueprint, candidates, top_k_per_slot=3)
    selection = LLMRoutineSelectionOutput.model_validate({
        "selections": [
            {"slotId": "push_main", "exerciseId": "bench"},
            {"slotId": "push_secondary", "exerciseId": "incline_press"},
            {"slotId": "triceps_direct", "exerciseId": "machine_pushdown"},
            {"slotId": "flex", "exerciseId": "cable_pushdown"},
        ]
    })

    output = adapt_slot_selection_output(
        selection,
        blueprint=blueprint,
        slot_candidate_bundles=bundles,
        ranked_candidates=candidates,
    )
    ids = [exercise.exercise_id for exercise in output.exercises]

    assert len(ids) == blueprint.minimum_exercise_count
    assert "machine_pushdown" in ids
    assert "cable_pushdown" not in ids
    assert output._selection_diagnostics["duplicate_family_skip_count"] >= 1


def test_candidate_pool_prefers_distinct_patterns_before_equipment_variants():
    candidates = [
        _candidate("machine_pushdown", "머신 로프 푸시다운", "triceps", "ISOLATION", 150),
        _candidate("cable_pushdown", "케이블 V바 푸시다운", "triceps", "ISOLATION", 145),
        _candidate("machine_fly", "머신 인클라인 플라이", "front-deltoids", "COMPOUND", 140),
        _candidate("dumbbell_fly", "덤벨 인클라인 플라이", "front-deltoids", "COMPOUND", 135),
        _candidate("bench", "스미스 머신 플랫 체스트 프레스", "chest", "COMPOUND", 130),
        _candidate("front_raise", "덤벨 프론트 레이즈", "front-deltoids", "ISOLATION", 125),
        _candidate("overhead_press", "머신 숄더 프레스", "front-deltoids", "COMPOUND", 120),
        _candidate("triceps_extension", "덤벨 라잉 트라이셉스 익스텐션", "triceps", "ISOLATION", 115),
    ]

    selected = _diversify_ranked_candidates(
        candidates,
        top_n=5,
        target_muscles=["chest", "front-deltoids", "triceps"],
    )
    selected_ids = {item["id"] for item in selected}

    assert "bench" in selected_ids
    assert "front_raise" in selected_ids
    assert len(selected_ids & {"machine_pushdown", "cable_pushdown"}) == 1
    assert len(selected_ids & {"machine_fly", "dumbbell_fly"}) == 1


def test_push_flex_fill_balances_chest_shoulders_and_triceps():
    blueprint = build_routine_blueprint(
        target_split_label="push",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=60,
    )
    candidates = [
        _candidate("flat_press", "스미스 머신 플랫 체스트 프레스", "chest", "COMPOUND", 150),
        _candidate("incline_fly", "머신 인클라인 플라이", "front-deltoids", "ISOLATION", 145),
        _candidate("pushdown", "머신 로프 푸시다운", "triceps", "ISOLATION", 140),
        _candidate("pushup", "스미스 머신 푸시업", "chest", "COMPOUND", 130),
        _candidate("lateral_raise", "덤벨 레터럴 레이즈", "side-deltoids", "ISOLATION", 125),
        _candidate("jm_press", "머신 JM 프레스", "triceps", "ISOLATION", 120),
        _candidate("tate_press", "머신 테이트 프레스", "triceps", "ISOLATION", 115),
    ]
    bundles = build_blueprint_candidate_bundles(blueprint, candidates, top_k_per_slot=3)
    output = adapt_slot_selection_output(
        LLMRoutineSelectionOutput.model_validate({
            "selections": [
                {"slotId": "push_main", "exerciseId": "flat_press"},
                {"slotId": "push_secondary", "exerciseId": "incline_fly"},
                {"slotId": "triceps_direct", "exerciseId": "pushdown"},
            ]
        }),
        blueprint=blueprint,
        slot_candidate_bundles=bundles,
        ranked_candidates=candidates,
    )
    primary_counts: dict[str, int] = {}
    for exercise in output.exercises:
        primary = exercise.primary_muscles[0]
        primary_counts[primary] = primary_counts.get(primary, 0) + 1

    assert len(output.exercises) == blueprint.minimum_exercise_count
    assert set(primary_counts) == {
        "chest",
        "front-deltoids",
        "side-deltoids",
        "triceps",
    }
    assert max(primary_counts.values()) <= 2
    assert "tate_press" not in {exercise.exercise_id for exercise in output.exercises}
    assert output._selection_diagnostics["distribution_cap_relax_count"] == 0


def test_strength_flex_fill_prefers_distinct_compound_before_isolation_accessory():
    blueprint = build_routine_blueprint(
        target_split_label="legs",
        target_muscles=[],
        goal="strength",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=60,
    )
    candidates = [
        _candidate("smith_squat", "스미스 머신 스쿼트", "quadriceps", "COMPOUND", 150),
        _candidate("wide_squat", "덤벨 와이드 스쿼트", "adductor", "COMPOUND", 149),
        _candidate("leg_curl", "머신 레그 컬", "hamstring", "ISOLATION", 145),
        _candidate("frog_pump", "프로그 펌프", "gluteal", "ISOLATION", 140),
        _candidate("calf_raise", "머신 카프 레이즈", "calves", "ISOLATION", 135),
        _candidate("side_lunge", "덤벨 사이드 런지", "adductor", "COMPOUND", 110),
    ]
    bundles = build_blueprint_candidate_bundles(blueprint, candidates, top_k_per_slot=3)
    output = adapt_slot_selection_output(
        LLMRoutineSelectionOutput.model_validate({
            "selections": [
                {"slotId": "lower_main", "exerciseId": "smith_squat"},
                {"slotId": "posterior_chain", "exerciseId": "leg_curl"},
            ]
        }),
        blueprint=blueprint,
        slot_candidate_bundles=bundles,
        ranked_candidates=candidates,
    )

    ids = {exercise.exercise_id for exercise in output.exercises}
    compound_count = sum(exercise.movement_type == "COMPOUND" for exercise in output.exercises)

    assert len(output.exercises) == blueprint.minimum_exercise_count
    assert "side_lunge" in ids
    assert "wide_squat" not in ids
    assert "frog_pump" not in ids
    assert "calf_raise" not in ids
    assert compound_count == 2
    assert output._selection_phase_by_exercise_id["leg_curl"] == "isolation_accessory"
    assert output._selection_diagnostics["duplicate_family_skip_count"] >= 1


def test_slot_selection_is_emitted_in_blueprint_order_even_when_model_reverses_it():
    blueprint, candidates, bundles = _push_context()
    selection = LLMRoutineSelectionOutput.model_validate(
        {
            "summaryTitle": "초보자 푸시 루틴",
            "rationaleSummary": ["안정적인 머신 운동을 우선했습니다."],
            "selections": [
                {
                    "slotId": "triceps_direct",
                    "exerciseId": "cable_pushdown",
                    "exerciseRationale": "삼두 직접 운동입니다.",
                },
                {
                    "slotId": "push_main",
                    "exerciseId": "bench_press",
                    "exerciseRationale": "가슴 메인 운동입니다.",
                },
            ],
        }
    )

    output = adapt_slot_selection_output(
        selection,
        blueprint=blueprint,
        slot_candidate_bundles=bundles,
        ranked_candidates=candidates,
    )

    assert [exercise.exercise_id for exercise in output.exercises] == [
        "bench_press",
        "cable_pushdown",
        "front_raise",
    ]
    assert output.summary_title == "맞춤 운동 루틴"


def test_standard_execution_keeps_every_exercise_in_a_single_group():
    blueprint, candidates, bundles = _push_context()
    output = adapt_slot_selection_output(
        LLMRoutineSelectionOutput(selections=[]),
        blueprint=blueprint,
        slot_candidate_bundles=bundles,
        ranked_candidates=candidates,
    )

    adjusted = apply_blueprint_execution_policy(output, blueprint)
    metadata = adjusted._execution_metadata_by_exercise_id

    assert all(item["executionGroupType"] == "single" for item in metadata.values())
    assert [item["executionGroupId"] for item in metadata.values()] == ["S1", "S2", "S3"]


def test_superset_execution_pairs_upper_antagonist_main_lifts_and_sets_group_rest():
    blueprint = build_routine_blueprint(
        target_split_label="chest_back",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=45,
        session_structure="superset",
    )
    candidates = [
        _candidate("chest_press", "덤벨 체스트 프레스", "chest", "COMPOUND", 120, "DUMBBELL"),
        _candidate("dumbbell_row", "덤벨 로우", "upper-back", "COMPOUND", 115, "DUMBBELL"),
    ]
    bundles = build_blueprint_candidate_bundles(blueprint, candidates, top_k_per_slot=1)
    output = adapt_slot_selection_output(
        LLMRoutineSelectionOutput(selections=[]),
        blueprint=blueprint,
        slot_candidate_bundles=bundles,
        ranked_candidates=candidates,
    )

    adjusted = apply_blueprint_execution_policy(output, blueprint)
    metadata = adjusted._execution_metadata_by_exercise_id

    assert metadata["chest_press"]["executionGroupId"] == "SS1"
    assert metadata["dumbbell_row"]["executionGroupId"] == "SS1"
    assert metadata["chest_press"]["executionGroupPosition"] == 1
    assert metadata["dumbbell_row"]["executionGroupPosition"] == 2

    grouped_time = estimate_routine_time_min(
        output.exercises,
        execution_metadata=output._execution_metadata_by_exercise_id,
    )
    ungrouped_time = estimate_routine_time_min(output.exercises)
    assert grouped_time < ungrouped_time
    assert adjusted.exercises[0].rest_time_sec == 15
    assert adjusted.exercises[1].rest_time_sec == 90


def test_superset_execution_limits_default_session_to_one_pair():
    blueprint = build_routine_blueprint(
        target_split_label="chest_back",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=60,
        session_structure="superset",
    )
    output = LLMRoutineOutput.model_validate({
        "total_estimated_time": 60,
        "summary_title": "가슴·등 슈퍼세트",
        "rationale_summary": [],
        "warnings": [],
        "exercises": [
            {
                "exercise_id": "chest_press",
                "exercise_name": "덤벨 체스트 프레스",
                "movement_type": "COMPOUND",
                "primary_muscles": ["chest"],
                "equipment_type": "DUMBBELL",
                "target_reps": 8,
                "sets": 3,
                "rest_time_sec": 90,
                "exercise_rationale": "가슴 메인",
            },
            {
                "exercise_id": "dumbbell_row",
                "exercise_name": "덤벨 로우",
                "movement_type": "COMPOUND",
                "primary_muscles": ["upper-back"],
                "equipment_type": "DUMBBELL",
                "target_reps": 8,
                "sets": 3,
                "rest_time_sec": 90,
                "exercise_rationale": "등 메인",
            },
            {
                "exercise_id": "cable_fly",
                "exercise_name": "케이블 플라이",
                "movement_type": "ISOLATION",
                "primary_muscles": ["chest"],
                "equipment_type": "CABLE",
                "target_reps": 12,
                "sets": 3,
                "rest_time_sec": 60,
                "exercise_rationale": "가슴 보조",
            },
            {
                "exercise_id": "cable_pullover",
                "exercise_name": "케이블 풀오버",
                "movement_type": "ISOLATION",
                "primary_muscles": ["upper-back"],
                "equipment_type": "CABLE",
                "target_reps": 12,
                "sets": 3,
                "rest_time_sec": 60,
                "exercise_rationale": "등 보조",
            },
        ],
    })
    output._selection_phase_by_exercise_id = {
        "chest_press": "main_compound",
        "dumbbell_row": "main_compound",
        "cable_fly": "isolation_accessory",
        "cable_pullover": "isolation_accessory",
    }

    adjusted = apply_blueprint_execution_policy(output, blueprint)
    metadata = adjusted._execution_metadata_by_exercise_id

    assert {
        metadata["chest_press"]["executionGroupId"],
        metadata["dumbbell_row"]["executionGroupId"],
    } == {"SS1"}
    assert metadata["cable_fly"]["executionGroupType"] == "single"
    assert metadata["cable_pullover"]["executionGroupType"] == "single"
    assert sum(
        item["executionGroupType"] == "superset"
        for item in metadata.values()
    ) == 2
    assert any("핵심 한 쌍만 묶고" in line for line in adjusted.rationale_summary)


def test_superset_execution_downgrades_different_generic_machines_to_single_sets():
    blueprint = build_routine_blueprint(
        target_split_label="chest_back",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=45,
        session_structure="superset",
    )
    candidates = [
        _candidate("chest_press", "머신 체스트 프레스", "chest", "COMPOUND", 120),
        _candidate("seated_row", "머신 시티드 로우", "upper-back", "COMPOUND", 115),
    ]
    bundles = build_blueprint_candidate_bundles(blueprint, candidates, top_k_per_slot=1)
    output = adapt_slot_selection_output(
        LLMRoutineSelectionOutput(selections=[]),
        blueprint=blueprint,
        slot_candidate_bundles=bundles,
        ranked_candidates=candidates,
    )

    adjusted = apply_blueprint_execution_policy(output, blueprint)
    metadata = adjusted._execution_metadata_by_exercise_id

    assert metadata["chest_press"]["executionGroupType"] == "single"
    assert metadata["seated_row"]["executionGroupType"] == "single"
    assert adjusted._selection_diagnostics["grouped_exercise_count"] == 0
    assert any("모두 일반 세트" in line for line in adjusted.rationale_summary)
    assert not any("핵심 한 쌍만 묶고" in line for line in adjusted.rationale_summary)


def test_superset_execution_repairs_selected_machines_with_safe_compatible_candidate():
    blueprint = build_routine_blueprint(
        target_split_label="chest_back",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=45,
        session_structure="superset",
    )
    candidates = [
        _candidate("machine_press", "머신 체스트 프레스", "chest", "COMPOUND", 130),
        _candidate("machine_row", "머신 시티드 로우", "upper-back", "COMPOUND", 125),
        _candidate("dumbbell_press", "덤벨 체스트 프레스", "chest", "COMPOUND", 120, "DUMBBELL"),
        _candidate("dumbbell_row", "덤벨 로우", "upper-back", "COMPOUND", 115, "DUMBBELL"),
    ]
    output = LLMRoutineOutput(
        total_estimated_time=45,
        summary_title="가슴·등 슈퍼세트",
        rationale_summary=[],
        warnings=[],
        exercises=[
            {
                "exercise_id": "machine_press",
                "exercise_name": "머신 체스트 프레스",
                "movement_type": "COMPOUND",
                "primary_muscles": ["chest"],
                "equipment_type": "MACHINE",
                "target_reps": 10,
                "sets": 3,
                "rest_time_sec": 90,
                "exercise_rationale": "가슴 메인 운동",
            },
            {
                "exercise_id": "dumbbell_row",
                "exercise_name": "덤벨 로우",
                "movement_type": "COMPOUND",
                "primary_muscles": ["upper-back"],
                "equipment_type": "DUMBBELL",
                "target_reps": 10,
                "sets": 3,
                "rest_time_sec": 90,
                "exercise_rationale": "등 메인 운동",
            },
        ],
    )
    output._selection_phase_by_exercise_id = {
        "machine_press": "main_compound",
        "dumbbell_row": "secondary_compound",
    }
    output._selection_slot_by_exercise_id = {
        "machine_press": "upper_push_main",
        "dumbbell_row": "upper_pull_main",
    }

    repaired = ensure_antagonist_superset_pair(output, blueprint, candidates)
    adjusted = apply_blueprint_execution_policy(repaired, blueprint)
    metadata = adjusted._execution_metadata_by_exercise_id

    assert "machine_press" not in {item.exercise_id for item in adjusted.exercises}
    assert {item.exercise_id for item in adjusted.exercises} == {
        "dumbbell_press",
        "dumbbell_row",
    }
    assert metadata["dumbbell_press"]["executionGroupId"] == "SS1"
    assert metadata["dumbbell_row"]["executionGroupId"] == "SS1"
    assert adjusted._selection_slot_by_exercise_id["dumbbell_press"] == "upper_push_main"
    assert adjusted._selection_diagnostics["superset_pair_repair_count"] == 1


def test_superset_execution_repairs_both_machines_when_single_replacement_cannot_pair():
    blueprint = build_routine_blueprint(
        target_split_label="chest_back",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=45,
        session_structure="superset",
    )
    candidates = [
        _candidate("machine_press", "머신 체스트 프레스", "chest", "COMPOUND", 130),
        _candidate("machine_row", "머신 시티드 로우", "upper-back", "COMPOUND", 125),
        _candidate("barbell_press", "바벨 벤치 프레스", "chest", "COMPOUND", 120, "BARBELL"),
        _candidate("dumbbell_row", "덤벨 로우", "upper-back", "COMPOUND", 115, "DUMBBELL"),
    ]
    output = LLMRoutineOutput(
        total_estimated_time=45,
        summary_title="가슴·등 슈퍼세트",
        rationale_summary=[],
        warnings=[],
        exercises=[
            {
                "exercise_id": candidate["id"],
                "exercise_name": candidate["name_kr"],
                "movement_type": candidate["movement_type"],
                "primary_muscles": [candidate["primary_muscle"]],
                "equipment_type": candidate["equipment_req"],
                "target_reps": 10,
                "sets": 3,
                "rest_time_sec": 90,
                "exercise_rationale": "검증 운동",
            }
            for candidate in candidates[:2]
        ],
    )
    output._selection_phase_by_exercise_id = {
        "machine_press": "main_compound",
        "machine_row": "secondary_compound",
    }
    output._selection_slot_by_exercise_id = {
        "machine_press": "upper_push_main",
        "machine_row": "upper_pull_main",
    }

    repaired = ensure_antagonist_superset_pair(output, blueprint, candidates)
    adjusted = apply_blueprint_execution_policy(repaired, blueprint)
    metadata = adjusted._execution_metadata_by_exercise_id

    assert [item.exercise_id for item in adjusted.exercises] == [
        "barbell_press",
        "dumbbell_row",
    ]
    assert metadata["barbell_press"]["executionGroupId"] == "SS1"
    assert metadata["dumbbell_row"]["executionGroupId"] == "SS1"
    assert adjusted._selection_phase_by_exercise_id == {
        "barbell_press": "main_compound",
        "dumbbell_row": "secondary_compound",
    }
    assert adjusted._selection_slot_by_exercise_id == {
        "barbell_press": "upper_push_main",
        "dumbbell_row": "upper_pull_main",
    }
    assert adjusted._selection_diagnostics["superset_pair_repair_count"] == 2


def test_superset_execution_does_not_relax_two_machine_station_policy_without_alternative():
    blueprint = build_routine_blueprint(
        target_split_label="chest_back",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=45,
        session_structure="superset",
    )
    candidates = [
        _candidate("machine_press", "머신 체스트 프레스", "chest", "COMPOUND", 130),
        _candidate("machine_row", "머신 시티드 로우", "upper-back", "COMPOUND", 125),
    ]
    output = LLMRoutineOutput(
        total_estimated_time=45,
        summary_title="가슴·등 루틴",
        rationale_summary=[],
        warnings=[],
        exercises=[
            {
                "exercise_id": candidate["id"],
                "exercise_name": candidate["name_kr"],
                "movement_type": candidate["movement_type"],
                "primary_muscles": [candidate["primary_muscle"]],
                "equipment_type": candidate["equipment_req"],
                "target_reps": 10,
                "sets": 3,
                "rest_time_sec": 90,
                "exercise_rationale": "검증 운동",
            }
            for candidate in candidates
        ],
    )
    output._selection_phase_by_exercise_id = {
        "machine_press": "main_compound",
        "machine_row": "secondary_compound",
    }

    repaired = ensure_antagonist_superset_pair(output, blueprint, candidates)
    adjusted = apply_blueprint_execution_policy(repaired, blueprint)

    assert [item.exercise_id for item in adjusted.exercises] == [
        "machine_press",
        "machine_row",
    ]
    assert all(
        item["executionGroupType"] == "single"
        for item in adjusted._execution_metadata_by_exercise_id.values()
    )


def test_circuit_execution_keeps_first_main_single_and_caps_grouped_sets():
    blueprint = build_routine_blueprint(
        target_split_label="full_body",
        target_muscles=[],
        goal="generalfitness",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=60,
        session_structure="circuit",
    )
    candidates = [
        _candidate("squat", "머신 스쿼트", "quadriceps", "COMPOUND", 130),
        _candidate("chest_press", "체스트 프레스", "chest", "COMPOUND", 120),
        _candidate("seated_row", "시티드 로우", "upper-back", "COMPOUND", 115),
        _candidate("leg_curl", "덤벨 레그 컬", "hamstring", "ISOLATION", 100, "DUMBBELL"),
        _candidate("shoulder_press", "덤벨 숄더 프레스", "front-deltoids", "COMPOUND", 95, "DUMBBELL"),
    ]
    bundles = build_blueprint_candidate_bundles(blueprint, candidates, top_k_per_slot=1)
    output = adapt_slot_selection_output(
        LLMRoutineSelectionOutput(
            selections=[{
                "slotId": "full_posterior_chain",
                "exerciseId": "leg_curl",
            }, {
                "slotId": "full_vertical_push",
                "exerciseId": "shoulder_press",
            }]
        ),
        blueprint=blueprint,
        slot_candidate_bundles=bundles,
        ranked_candidates=candidates,
    )
    for exercise in output.exercises:
        exercise.sets = 4

    adjusted = apply_blueprint_execution_policy(output, blueprint)
    metadata = adjusted._execution_metadata_by_exercise_id

    assert metadata["squat"]["executionGroupType"] == "single"
    assert metadata["chest_press"]["executionGroupType"] == "single"
    assert metadata["seated_row"]["executionGroupType"] == "single"
    grouped = [
        exercise
        for exercise in adjusted.exercises
        if metadata[exercise.exercise_id]["executionGroupType"] == "circuit"
    ]
    assert {exercise.exercise_id for exercise in grouped} == {"leg_curl", "shoulder_press"}
    assert all(exercise.sets <= 2 for exercise in grouped)


def test_missing_required_slots_are_repaired_only_from_matching_bundles():
    blueprint, candidates, bundles = _push_context()

    output = adapt_slot_selection_output(
        LLMRoutineSelectionOutput(selections=[]),
        blueprint=blueprint,
        slot_candidate_bundles=bundles,
        ranked_candidates=candidates,
    )

    assert [exercise.exercise_id for exercise in output.exercises] == [
        "bench_press",
        "cable_pushdown",
        "front_raise",
    ]
    assert any("누락된 운동" in warning for warning in output.warnings)
    assert output._selection_diagnostics["required_slot_repair_count"] == 2


def test_normal_minimum_flex_fill_is_telemetry_not_user_warning():
    blueprint = build_routine_blueprint(
        target_split_label="push",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=45,
    )
    candidates = [
        _candidate("bench_press", "머신 체스트 프레스", "chest", "COMPOUND", 120),
        _candidate("incline_press", "인클라인 체스트 프레스", "chest", "COMPOUND", 115),
        _candidate("cable_pushdown", "케이블 푸시다운", "triceps", "ISOLATION", 110),
        _candidate("front_raise", "머신 프론트 레이즈", "front-deltoids", "ISOLATION", 90),
    ]
    bundles = build_blueprint_candidate_bundles(blueprint, candidates, top_k_per_slot=1)
    selection = LLMRoutineSelectionOutput.model_validate(
        {
            "selections": [
                {"slotId": "push_main", "exerciseId": "bench_press"},
                {"slotId": "push_secondary", "exerciseId": "incline_press"},
                {"slotId": "triceps_direct", "exerciseId": "cable_pushdown"},
            ]
        }
    )

    output = adapt_slot_selection_output(
        selection,
        blueprint=blueprint,
        slot_candidate_bundles=bundles,
        ranked_candidates=candidates,
    )

    assert output.warnings == []
    assert output._selection_diagnostics == {
        "contract_version": "slot-selection-v3",
        "specialization_focus_unavailable": False,
        "model_selection_count": 3,
        "model_accepted_count": 3,
        "rejected_selection_count": 0,
        "required_slot_repair_count": 0,
        "requested_slot_repair_count": 0,
        "minimum_fill_count": 1,
        "duplicate_key_repair_count": 0,
        "duplicate_family_skip_count": 0,
        "coarse_pattern_skip_count": 0,
        "distribution_cap_relax_count": 0,
    }


def test_flex_compound_is_ordered_before_direct_accessory():
    blueprint = build_routine_blueprint(
        target_split_label="push",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=45,
    )
    candidates = [
        _candidate("bench_press", "머신 체스트 프레스", "chest", "COMPOUND", 120),
        _candidate("incline_press", "인클라인 체스트 프레스", "chest", "COMPOUND", 115),
        _candidate("cable_pushdown", "케이블 푸시다운", "triceps", "ISOLATION", 110),
    ]
    bundles = build_blueprint_candidate_bundles(blueprint, candidates, top_k_per_slot=1)
    selection = LLMRoutineSelectionOutput.model_validate({
        "selections": [
            {"slotId": "push_main", "exerciseId": "bench_press"},
            {"slotId": "triceps_direct", "exerciseId": "cable_pushdown"},
            {"slotId": "flex", "exerciseId": "incline_press"},
        ]
    })

    output = adapt_slot_selection_output(
        selection,
        blueprint=blueprint,
        slot_candidate_bundles=bundles,
        ranked_candidates=candidates,
    )

    assert [exercise.exercise_id for exercise in output.exercises] == [
        "bench_press",
        "incline_press",
        "cable_pushdown",
    ]


def test_flex_fill_skips_another_variant_from_an_already_selected_family():
    blueprint = build_routine_blueprint(
        target_split_label="push",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=45,
    )
    candidates = [
        {
            **_candidate("bench_press", "머신 체스트 프레스", "chest", "COMPOUND", 120),
            "base_exercise_name_kr": "체스트 프레스",
        },
        {
            **_candidate("incline_press", "인클라인 체스트 프레스", "chest", "COMPOUND", 115),
            "base_exercise_name_kr": "인클라인 프레스",
        },
        _candidate("cable_pushdown", "케이블 푸시다운", "triceps", "ISOLATION", 110),
        {
            **_candidate("press_variant", "다른 체스트 프레스", "chest", "COMPOUND", 100),
            "base_exercise_name_kr": "체스트 프레스",
        },
        _candidate("front_raise", "머신 프론트 레이즈", "front-deltoids", "ISOLATION", 90),
    ]
    bundles = build_blueprint_candidate_bundles(blueprint, candidates, top_k_per_slot=1)
    selection = LLMRoutineSelectionOutput.model_validate(
        {
            "selections": [
                {"slotId": "push_main", "exerciseId": "bench_press"},
                {"slotId": "push_secondary", "exerciseId": "incline_press"},
                {"slotId": "triceps_direct", "exerciseId": "cable_pushdown"},
            ]
        }
    )

    output = adapt_slot_selection_output(
        selection,
        blueprint=blueprint,
        slot_candidate_bundles=bundles,
        ranked_candidates=candidates,
    )

    assert [exercise.exercise_id for exercise in output.exercises] == [
        "bench_press",
        "incline_press",
        "cable_pushdown",
        "front_raise",
    ]
    assert output._selection_diagnostics["duplicate_family_skip_count"] == 1
    assert output._selection_diagnostics["minimum_fill_count"] == 1


def test_safety_limited_blueprint_does_not_fill_back_to_base_count():
    blueprint = build_routine_blueprint(
        target_split_label="push",
        target_muscles=[],
        goal="hypertrophy",
        experience_level="intermediate",
        readiness_level="normal",
        time_available_min=60,
        pain_areas=[{"bodyPart": "chest"}],
    )
    candidates = [
        _candidate("front_raise", "머신 프론트 레이즈", "front-deltoids", "ISOLATION", 120),
        _candidate("lateral_raise", "머신 레터럴 레이즈", "side-deltoids", "ISOLATION", 115),
        _candidate("pushdown", "케이블 푸시다운", "triceps", "ISOLATION", 110),
        _candidate("jm_press", "머신 JM 프레스", "triceps", "COMPOUND", 105),
        _candidate("tate_press", "덤벨 테이트 프레스", "triceps", "ISOLATION", 100),
    ]
    bundles = build_blueprint_candidate_bundles(blueprint, candidates, top_k_per_slot=2)
    output = adapt_slot_selection_output(
        LLMRoutineSelectionOutput(selections=[]),
        blueprint=blueprint,
        slot_candidate_bundles=bundles,
        ranked_candidates=candidates,
    )

    assert blueprint.base_target_exercise_count == 6
    assert blueprint.target_exercise_count == 5
    assert len(output.exercises) == blueprint.minimum_exercise_count
    assert all("chest" not in exercise.primary_muscles for exercise in output.exercises)


def test_cross_bundle_unknown_and_duplicate_selections_are_rejected():
    blueprint, candidates, bundles = _push_context()
    selection = LLMRoutineSelectionOutput.model_validate(
        {
            "selections": [
                {"slotId": "push_main", "exerciseId": "cable_pushdown"},
                {"slotId": "unknown_slot", "exerciseId": "bench_press"},
                {"slotId": "flex", "exerciseId": "bench_press"},
            ]
        }
    )

    output = adapt_slot_selection_output(
        selection,
        blueprint=blueprint,
        slot_candidate_bundles=bundles,
        ranked_candidates=candidates,
    )

    assert [exercise.exercise_id for exercise in output.exercises] == [
        "bench_press",
        "cable_pushdown",
        "front_raise",
    ]
    assert any("현재 조건 또는 루틴 구성과 맞지 않는 운동 3개" in warning for warning in output.warnings)


def test_raw_camel_case_v2_selection_is_parsed_and_adapted():
    blueprint, candidates, bundles = _push_context()
    raw = json.dumps(
        {
            "summaryTitle": "선택 결과",
            "rationaleSummary": ["후보 번들 안에서 골랐습니다."],
            "warnings": [],
            "selections": [
                {
                    "slotId": "push_main",
                    "exerciseId": "bench_press",
                    "exerciseRationale": "초보자가 쓰기 쉬운 메인 운동입니다.",
                },
                {
                    "slotId": "triceps_direct",
                    "exerciseId": "cable_pushdown",
                    "exerciseRationale": "직접 삼두 볼륨을 보완합니다.",
                },
            ],
        },
        ensure_ascii=False,
    )

    output = normalize_routine_selection_response(
        raw,
        blueprint=blueprint,
        slot_candidate_bundles=bundles,
        ranked_candidates=candidates,
    )

    assert isinstance(output, LLMRoutineOutput)
    assert output.summary_title == "맞춤 운동 루틴"
    assert [exercise.exercise_id for exercise in output.exercises] == [
        "bench_press",
        "cable_pushdown",
        "front_raise",
    ]


def test_duplicate_selection_keys_are_split_without_a_second_llm_call():
    blueprint, candidates, bundles = _push_context()
    raw = (
        '{"summaryTitle":"중복 키 복구","rationaleSummary":[],"warnings":[],"selections":['
        '{"slotId":"push_main","exerciseId":"bench_press","exerciseRationale":"가슴 메인",'
        '"slotId":"triceps_direct","exerciseId":"cable_pushdown","exerciseRationale":"삼두 직접"}'
        "]}"
    )

    output = normalize_routine_selection_response(
        raw,
        blueprint=blueprint,
        slot_candidate_bundles=bundles,
        ranked_candidates=candidates,
    )

    assert [exercise.exercise_id for exercise in output.exercises] == [
        "bench_press",
        "cable_pushdown",
        "front_raise",
    ]
    assert output._selection_diagnostics["duplicate_key_repair_count"] == 1
    assert output._selection_diagnostics["required_slot_repair_count"] == 0
    assert output._selection_diagnostics["model_accepted_count"] == 2


def test_small_model_selection_text_shapes_are_repaired_without_fallback():
    blueprint, candidates, bundles = _push_context()
    raw = json.dumps(
        {
            "summaryTitle": ["선택 결과"],
            "rationaleSummary": [{"reason": "후보 번들 안에서 골랐습니다."}],
            "warnings": "중량을 보수적으로 조절하세요.",
            "selections": {
                "first": {
                    "slotId": ["push_main"],
                    "exerciseId": {"id": "bench_press"},
                    "exerciseRationale": {"text": "가슴 메인 운동입니다."},
                },
                "second": {
                    "slot_id": "triceps_direct",
                    "exercise_id": "cable_pushdown",
                    "exercise_rationale": ["삼두 직접 볼륨을 보완합니다."],
                },
            },
        },
        ensure_ascii=False,
    )

    output = normalize_routine_selection_response(
        raw,
        blueprint=blueprint,
        slot_candidate_bundles=bundles,
        ranked_candidates=candidates,
    )

    assert output.summary_title == "맞춤 운동 루틴"
    assert output.rationale_summary == [
        "초급자 숙련도와 근비대 목표를 반영해 운동 순서와 구성을 조정했습니다."
    ]
    assert "중량을 보수적으로 조절하세요." not in output.warnings
    assert [exercise.exercise_id for exercise in output.exercises] == [
        "bench_press",
        "cable_pushdown",
        "front_raise",
    ]
    assert output._selection_diagnostics["selection_contract_type_repair_count"] >= 1


def test_invalid_small_model_selection_items_are_dropped_for_server_slot_repair():
    blueprint, candidates, bundles = _push_context()
    raw = json.dumps(
        {
            "summaryTitle": "선택 결과",
            "rationaleSummary": "서버가 누락 슬롯을 보완합니다.",
            "warnings": [],
            "selections": [
                {"slotId": "push_main", "exerciseId": "bench_press"},
                {"slotId": {"unexpected": []}, "exerciseId": None},
                "not-an-object",
            ],
        },
        ensure_ascii=False,
    )

    output = normalize_routine_selection_response(
        raw,
        blueprint=blueprint,
        slot_candidate_bundles=bundles,
        ranked_candidates=candidates,
    )

    assert output.rationale_summary == [
        "초급자 숙련도와 근비대 목표를 반영해 운동 순서와 구성을 조정했습니다."
    ]
    assert [exercise.exercise_id for exercise in output.exercises] == [
        "bench_press",
        "cable_pushdown",
        "front_raise",
    ]
    assert output._selection_diagnostics["required_slot_repair_count"] == 1


def test_model_rationale_that_contradicts_selected_exercise_is_not_user_visible():
    blueprint, candidates, bundles = _push_context()
    selection = LLMRoutineSelectionOutput.model_validate({
        "summaryTitle": "6가지 운동을 선택했습니다.",
        "rationaleSummary": ["머신 로우를 선택했습니다."],
        "warnings": ["이 슬롯을 건너뛰세요."],
        "selections": [
            {
                "slotId": "push_main",
                "exerciseId": "bench_press",
                "exerciseRationale": "이 슬롯은 건너뛰었고 동일 ID라 다음 후보로 대체합니다.",
            },
            {
                "slotId": "triceps_direct",
                "exerciseId": "cable_pushdown",
                "exerciseRationale": "선택하지 않았습니다.",
            },
        ],
    })

    output = adapt_slot_selection_output(
        selection,
        blueprint=blueprint,
        slot_candidate_bundles=bundles,
        ranked_candidates=candidates,
    )
    visible_text = " ".join([
        output.summary_title,
        *output.rationale_summary,
        *output.warnings,
        *(exercise.exercise_rationale for exercise in output.exercises),
    ])

    assert output.summary_title == "맞춤 운동 루틴"
    assert "머신 로우" not in visible_text
    assert "슬롯" not in visible_text
    assert "동일 ID" not in visible_text
    assert "건너뛰" not in visible_text


def test_raw_legacy_v1_output_remains_supported_during_transition():
    blueprint, candidates, bundles = _push_context()
    raw = json.dumps(
        {
            "total_estimated_time": 30,
            "summary_title": "기존 응답",
            "rationale_summary": ["기존 계약 호환 테스트입니다."],
            "warnings": [],
            "exercises": [
                {
                    "exercise_id": "bench_press",
                    "exercise_name": "머신 체스트 프레스",
                    "primary_muscles": ["chest"],
                    "target_reps": 10,
                    "sets": 3,
                    "rest_time_sec": 90,
                    "exercise_rationale": "가슴 메인 운동입니다.",
                }
            ],
        },
        ensure_ascii=False,
    )

    output = normalize_routine_selection_response(
        raw,
        blueprint=blueprint,
        slot_candidate_bundles=bundles,
        ranked_candidates=candidates,
    )

    assert output.summary_title == "기존 응답"
    assert [exercise.exercise_id for exercise in output.exercises] == ["bench_press"]


def test_structured_legacy_model_remains_supported_for_provider_transition():
    blueprint, candidates, bundles = _push_context()
    legacy = LLMRoutineOutput.model_validate(
        {
            "total_estimated_time": 30,
            "summary_title": "구조화 기존 응답",
            "rationale_summary": ["이행기 호환입니다."],
            "warnings": [],
            "exercises": [
                {
                    "exercise_id": "bench_press",
                    "exercise_name": "머신 체스트 프레스",
                    "primary_muscles": ["chest"],
                    "target_reps": 10,
                    "sets": 3,
                    "rest_time_sec": 90,
                    "exercise_rationale": "가슴 메인 운동입니다.",
                }
            ],
        }
    )

    output = adapt_structured_routine_response(
        legacy,
        blueprint=blueprint,
        slot_candidate_bundles=bundles,
        ranked_candidates=candidates,
    )

    assert output is legacy

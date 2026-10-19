"""generate_fallback_routine — DB/LLM 없이 순수 로직 테스트"""
import pytest

from engines.fallback import generate_fallback_routine
from engines.prescription.estimator import estimate_routine_time_min
from engines.routine_blueprint import build_routine_blueprint
from engines.schemas import (
    InjuryConditionEntry,
    LLMExercisePlan,
    RecentSetRecord,
    RoutineRequest,
)


@pytest.fixture
def push_request():
    return RoutineRequest(
        user_id="test-user",
        target_split_label="push",
        time_available_min=60,
        pain_areas=[],
        doms_data={},
        equipment=[],
    )


class TestFallbackEmptyCandidates:
    def test_returns_failed_status_with_empty_candidate(self, push_request):
        result = generate_fallback_routine(push_request, candidates=[], max_total_sets=15)
        assert result.generation_status == "failed"
        assert result.status_reason_code == "emptyCandidate"
        assert result.is_fallback is False

    def test_empty_blocks_with_warning(self, push_request):
        result = generate_fallback_routine(push_request, candidates=[], max_total_sets=15)
        assert result.routine_blocks == []
        assert len(result.warnings) > 0


class TestFallbackDomsLogic:
    def test_doms_level3_skips_exercise(self, push_request, mock_candidates):
        doms_db = {"chest": 3}
        result = generate_fallback_routine(
            push_request, candidates=mock_candidates, max_total_sets=15, doms_db=doms_db
        )
        exercise_ids = [b.exercise_id for b in result.routine_blocks]
        assert "barbell_bench_press" not in exercise_ids
        assert "dumbbell_fly" not in exercise_ids

    def test_doms_level2_limits_sets_to_2(self, push_request, mock_candidates):
        doms_db = {"chest": 2}
        result = generate_fallback_routine(
            push_request, candidates=mock_candidates, max_total_sets=15, doms_db=doms_db
        )
        chest_blocks = [b for b in result.routine_blocks if "chest" in b.primary_muscles]
        assert chest_blocks
        for block in chest_blocks:
            # Historical ID retained: v2 uses role capacity rather than 1/2 globally.
            assert len(block.prescription) == (3 if block.movement_type == "COMPOUND" else 2)

    def test_doms_level1_reduces_sets_by_1(self, push_request, mock_candidates):
        doms_db = {"chest": 1}
        result = generate_fallback_routine(
            push_request, candidates=mock_candidates, max_total_sets=15, doms_db=doms_db, goal="hypertrophy"
        )
        # v2: role/experience capacity, with RIR providing the effort distinction.
        for block in result.routine_blocks:
            assert len(block.prescription) <= 3

    def test_chest_doms_level3_removes_all_chest_blocks(self, push_request, mock_candidates):
        result = generate_fallback_routine(
            push_request,
            candidates=mock_candidates,
            max_total_sets=15,
            doms_db={"chest": 3},
        )
        assert all("chest" not in block.primary_muscles for block in result.routine_blocks)

    def test_explicit_safety_adjusted_target_count_caps_fallback(self, push_request, mock_candidates):
        result = generate_fallback_routine(
            push_request,
            candidates=mock_candidates,
            max_total_sets=15,
            target_exercise_count=2,
        )

        assert len(result.routine_blocks) == 2

    def test_level3_doms_removes_movement_family_that_strongly_loads_sore_muscle(self):
        request = RoutineRequest(
            user_id="legs-doms",
            target_split_label="legs",
            time_available_min=50,
            pain_areas=[],
            doms_data={"quadriceps": 3},
            equipment=[],
        )
        candidates = [
            {
                "id": "wide_dumbbell_squat_adductor_primary",
                "name_kr": "덤벨 와이드 스쿼트",
                "name_en": "Dumbbell Wide Squat",
                "primary_muscle": "adductor",
                "secondary_muscle": None,
                "equipment_req": "DUMBBELL",
                "efficiency_tier": 3,
                "movement_type": "COMPOUND",
                "pain_triggers": None,
            },
            {
                "id": "machine_leg_curl",
                "name_kr": "머신 레그 컬",
                "name_en": "Machine Leg Curl",
                "primary_muscle": "hamstring",
                "secondary_muscle": "",
                "equipment_req": "MACHINE",
                "efficiency_tier": 3,
                "movement_type": "ISOLATION",
                "pain_triggers": None,
            },
        ]

        result = generate_fallback_routine(
            request,
            candidates=candidates,
            max_total_sets=10,
            doms_db=request.doms_data,
            goal="hypertrophy",
        )

        exercise_ids = [block.exercise_id for block in result.routine_blocks]
        assert "wide_dumbbell_squat_adductor_primary" not in exercise_ids
        assert "machine_leg_curl" in exercise_ids


class TestFallbackMaxSets:
    def test_total_sets_does_not_exceed_max(self, push_request, mock_candidates):
        max_sets = 5
        result = generate_fallback_routine(
            push_request, candidates=mock_candidates, max_total_sets=max_sets
        )
        total = sum(len(b.prescription) for b in result.routine_blocks)
        assert total <= max_sets

    def test_explicit_focus_uses_same_volume_transfer_policy_in_fallback(self):
        request = RoutineRequest(
            user_id="focused-fallback",
            target_split_label="legs",
            target_muscles=[],
            time_available_min=60,
            goal="hypertrophy",
            experience_level="advanced",
            readiness_level="normal",
            specialization_focus="hamstringEmphasis",
            pain_areas=[],
            doms_data={},
            equipment=[],
        )
        candidates = [
            {
                "id": "leg_press",
                "name_kr": "레그 프레스",
                "primary_muscle": "quadriceps",
                "secondary_muscle": "",
                "equipment_req": "MACHINE",
                "difficulty_tier": 2,
                "efficiency_tier": 2,
                "movement_type": "COMPOUND",
                "pain_triggers": None,
            },
            {
                "id": "leg_curl",
                "name_kr": "머신 레그 컬",
                "primary_muscle": "hamstring",
                "secondary_muscle": "",
                "equipment_req": "MACHINE",
                "difficulty_tier": 2,
                "efficiency_tier": 3,
                "movement_type": "ISOLATION",
                "pain_triggers": None,
            },
            {
                "id": "calf_raise",
                "name_kr": "머신 카프 레이즈",
                "primary_muscle": "calves",
                "secondary_muscle": "",
                "equipment_req": "MACHINE",
                "difficulty_tier": 2,
                "efficiency_tier": 4,
                "movement_type": "ISOLATION",
                "pain_triggers": None,
            },
        ]
        blueprint = build_routine_blueprint(
            target_split_label="legs",
            target_muscles=[],
            goal="hypertrophy",
            experience_level="advanced",
            readiness_level="normal",
            time_available_min=60,
            specialization_focus="hamstringEmphasis",
        )
        default_request = request.model_copy(
            update={"specialization_focus": None}
        )
        default_blueprint = build_routine_blueprint(
            target_split_label="legs",
            target_muscles=[],
            goal="hypertrophy",
            experience_level="intermediate",
            readiness_level="normal",
            time_available_min=60,
        )
        default_result = generate_fallback_routine(
            default_request,
            candidates=candidates,
            max_total_sets=9,
            goal="hypertrophy",
            target_exercise_count=3,
            routine_blueprint=default_blueprint,
        )

        result = generate_fallback_routine(
            request,
            candidates=candidates,
            max_total_sets=9,
            goal="hypertrophy",
            target_exercise_count=3,
            routine_blueprint=blueprint,
        )

        default_by_id = {
            block.exercise_id: block for block in default_result.routine_blocks
        }
        by_id = {block.exercise_id: block for block in result.routine_blocks}
        assert sum(len(block.prescription) for block in result.routine_blocks) == sum(
            len(block.prescription) for block in default_result.routine_blocks
        )
        assert len(by_id["leg_curl"].prescription) == (
            len(default_by_id["leg_curl"].prescription) + 1
        )
        # Fund the focus from a four-set main, not a two-set accessory.
        assert len(by_id["calf_raise"].prescription) == len(default_by_id["calf_raise"].prescription)
        assert len(by_id["leg_press"].prescription) == len(default_by_id["leg_press"].prescription) - 1
        assert len(by_id["leg_press"].prescription) >= 3
        assert "햄스트링 집중" in by_id["leg_curl"].exercise_rationale

    def test_short_time_fallback_caps_exercise_count_even_when_doms_splits_sets(self):
        request = RoutineRequest(
            user_id="short-pull",
            target_split_label="pull",
            time_available_min=30,
            pain_areas=[],
            doms_data={"upper-back": 2, "biceps": 2},
            equipment=[],
        )
        candidates = [
            {
                "id": f"pull_{idx}",
                "name_kr": f"풀 운동 {idx}",
                "name_en": f"Pull {idx}",
                "primary_muscle": "upper-back" if idx % 2 else "biceps",
                "secondary_muscle": None,
                "equipment_req": "MACHINE",
                "difficulty_tier": 2,
                "efficiency_tier": idx,
                "pain_triggers": None,
                "movement_type": "ISOLATION",
            }
            for idx in range(1, 11)
        ]

        result = generate_fallback_routine(
            request,
            candidates=candidates,
            max_total_sets=10,
            doms_db=request.doms_data,
            goal="hypertrophy",
            status_reason_code="llmTimeout",
        )

        assert result.is_fallback is True
        assert len(result.routine_blocks) <= 3
        assert result.total_estimated_time <= request.time_available_min
        fallback_plans = [
            LLMExercisePlan(
                exercise_id=block.exercise_id,
                exercise_name=block.exercise_name,
                movement_pattern=block.movement_pattern,
                primary_muscles=block.primary_muscles,
                equipment_type=block.equipment_type,
                target_reps=block.prescription[0].target_reps,
                sets=len(block.prescription),
                rest_time_sec=block.default_rest_sec,
                target_rir=block.prescription[0].target_rir,
                exercise_rationale=block.exercise_rationale,
            )
            for block in result.routine_blocks
            if block.prescription
        ]
        assert estimate_routine_time_min(fallback_plans) <= request.time_available_min


class TestFallbackGoalParams:
    def test_strength_goal_uses_movement_specific_reps(self, push_request, mock_candidates):
        result = generate_fallback_routine(
            push_request, candidates=mock_candidates, max_total_sets=20, goal="strength"
        )
        compound = next(block for block in result.routine_blocks if block.exercise_id == "barbell_bench_press")
        isolation = next(block for block in result.routine_blocks if block.exercise_id == "dumbbell_fly")

        assert compound.prescription[0].target_reps == 5
        assert isolation.prescription[0].target_reps == 8
        assert len(result.routine_blocks) <= 4

    def test_endurance_goal_uses_15_reps(self, push_request, mock_candidates):
        result = generate_fallback_routine(
            push_request, candidates=mock_candidates, max_total_sets=20, goal="endurance"
        )
        for block in result.routine_blocks:
            for s in block.prescription:
                assert s.target_reps == 15

    def test_same_muscle_recent_sets_seed_fallback_weight(self, push_request, mock_candidates):
        recent_sets = [
            RecentSetRecord(
                exercise_name="Different Chest Exercise",
                primary_muscle="chest",
                weight_kg=100,
                reps=5,
            )
        ]
        result = generate_fallback_routine(
            push_request,
            candidates=mock_candidates,
            max_total_sets=20,
            goal="hypertrophy",
            recent_sets=recent_sets,
        )
        fly = next(block for block in result.routine_blocks if block.exercise_id == "dumbbell_fly")

        assert fly.prescription[0].target_weight_kg is None


class TestFallbackReadiness:
    def test_low_readiness_reduces_compound_sets_and_raises_rir(self, push_request, mock_candidates):
        request = push_request.model_copy(update={"readiness_level": "low"})
        result = generate_fallback_routine(
            request, candidates=mock_candidates, max_total_sets=20, goal="hypertrophy"
        )
        compound = next(block for block in result.routine_blocks if block.exercise_id == "barbell_bench_press")
        assert len(compound.prescription) == 2
        assert compound.prescription[0].target_rir == 4

    def test_high_readiness_keeps_sets_without_crossing_the_evidence_floor(self, push_request, mock_candidates):
        request = push_request.model_copy(update={"readiness_level": "high"})
        result = generate_fallback_routine(
            request, candidates=mock_candidates, max_total_sets=20, goal="hypertrophy"
        )
        first = result.routine_blocks[0]
        assert len(first.prescription) == 3
        assert first.prescription[0].target_rir == 2


class TestFallbackStatusReasonCode:
    def test_custom_status_reason_code_preserved(self, push_request, mock_candidates):
        result = generate_fallback_routine(
            push_request, candidates=mock_candidates, max_total_sets=15,
            status_reason_code="llmTimeout"
        )
        assert result.status_reason_code == "llmTimeout"

    def test_compound_exercises_come_before_isolation(self, push_request, mock_candidates):
        result = generate_fallback_routine(
            push_request, candidates=mock_candidates, max_total_sets=15
        )
        if len(result.routine_blocks) >= 2:
            # 첫 번째 블록은 COMPOUND(벤치프레스 or 푸시업)이어야 함
            first_id = result.routine_blocks[0].exercise_id
            compound_ids = {"barbell_bench_press", "pushup"}
            assert first_id in compound_ids


class TestFallbackPainContext:
    def test_knee_injury_hard_excludes_all_direct_quadriceps_fallback_candidates(self):
        request = RoutineRequest(
            user_id="knee-fallback",
            target_split_label="legs",
            time_available_min=45,
            pain_areas=[{"body_part": "knees"}],
            doms_data={},
            equipment=[],
        )
        candidates = [
            {
                "id": "sissy_squat",
                "name_kr": "시시 스쿼트",
                "primary_muscle": "quadriceps",
                "secondary_muscle": "",
                "equipment_req": "BODYWEIGHT",
                "efficiency_tier": 1,
                "movement_type": "COMPOUND",
                "knee_shear_load": "high",
                "score": 130,
            },
            {
                "id": "leg_extension",
                "name_kr": "레그 익스텐션",
                "primary_muscle": "quadriceps",
                "secondary_muscle": "",
                "equipment_req": "MACHINE",
                "efficiency_tier": 3,
                "movement_type": "ISOLATION",
                "knee_shear_load": "low",
                "score": 100,
            },
        ]

        result = generate_fallback_routine(
            request,
            candidates=candidates,
            max_total_sets=8,
            goal="hypertrophy",
            status_reason_code="schemaError",
        )

        assert result.generation_status == "failed"
        assert result.status_reason_code == "emptyCandidate"
        assert result.routine_blocks == []
        assert any("목표 부위를 바꾸거나 운동을 보류" in warning for warning in result.warnings)

    def test_clearance_policy_adds_medical_warning_in_fallback(self):
        request = RoutineRequest(
            user_id="medical-fallback",
            target_split_label="legs",
            time_available_min=45,
            pain_areas=[],
            doms_data={},
            equipment=[],
            condition_policies=[
                {"policyType": "surgeryHistory", "bodyPart": "lower-back", "status": "needs_clearance"}
            ],
        )
        candidates = [
            {
                "id": "leg_curl",
                "name_kr": "머신 레그 컬",
                "primary_muscle": "hamstring",
                "secondary_muscle": "",
                "equipment_req": "MACHINE",
                "efficiency_tier": 3,
                "movement_type": "ISOLATION",
                "score": 100,
            }
        ]

        result = generate_fallback_routine(
            request,
            candidates=candidates,
            max_total_sets=8,
            goal="hypertrophy",
            status_reason_code="schemaError",
        )

        assert any("의료진 허가" in warning for warning in result.warnings)

    @pytest.mark.parametrize(
        ("area", "pattern", "blocked_candidate", "safe_candidate"),
        [
            (
                "ankle",
                "deepDorsiflexion",
                {
                    "id": "deep_squat",
                    "name_kr": "딥 스쿼트",
                    "primary_muscle": "quadriceps",
                    "equipment_req": "BARBELL",
                    "movement_type": "COMPOUND",
                    "ankle_region_load_level": "high",
                    "deep_dorsiflexion_demand": "high",
                },
                {
                    "id": "leg_extension",
                    "name_kr": "머신 레그 익스텐션",
                    "primary_muscle": "quadriceps",
                    "equipment_req": "MACHINE",
                    "movement_type": "ISOLATION",
                    "ankle_region_load_level": "low",
                    "deep_dorsiflexion_demand": "low",
                },
            ),
            (
                "hip",
                "hipHinge",
                {
                    "id": "deadlift",
                    "name_kr": "바벨 데드리프트",
                    "primary_muscle": "hamstring",
                    "equipment_req": "BARBELL",
                    "movement_type": "COMPOUND",
                    "hip_region_load_level": "high",
                    "hip_hinge_load": "high",
                },
                {
                    "id": "seated_leg_curl",
                    "name_kr": "머신 시티드 레그 컬",
                    "primary_muscle": "hamstring",
                    "equipment_req": "MACHINE",
                    "movement_type": "ISOLATION",
                    "hip_region_load_level": "low",
                    "hip_hinge_load": "low",
                },
            ),
            (
                "elbow",
                "loadedElbowFlexion",
                {
                    "id": "barbell_curl",
                    "name_kr": "바벨 컬",
                    "primary_muscle": "biceps",
                    "equipment_req": "BARBELL",
                    "movement_type": "ISOLATION",
                    "elbow_region_load_level": "high",
                    "elbow_flexion_load": "high",
                },
                {
                    "id": "leg_press",
                    "name_kr": "머신 레그 프레스",
                    "primary_muscle": "quadriceps",
                    "equipment_req": "MACHINE",
                    "movement_type": "COMPOUND",
                    "elbow_region_load_level": "low",
                    "elbow_flexion_load": "low",
                },
            ),
            (
                "neck",
                "overheadAxialLoading",
                {
                    "id": "overhead_press",
                    "name_kr": "바벨 오버헤드 프레스",
                    "primary_muscle": "front-deltoids",
                    "equipment_req": "BARBELL",
                    "movement_type": "COMPOUND",
                    "neck_region_load_level": "high",
                    "overhead_axial_load": "high",
                },
                {
                    "id": "leg_press",
                    "name_kr": "머신 레그 프레스",
                    "primary_muscle": "quadriceps",
                    "equipment_req": "MACHINE",
                    "movement_type": "COMPOUND",
                    "neck_region_load_level": "low",
                    "overhead_axial_load": "low",
                },
            ),
        ],
    )
    def test_managed_injury_filter_is_reapplied_inside_fallback(
        self,
        area,
        pattern,
        blocked_candidate,
        safe_candidate,
    ):
        request = RoutineRequest(
            user_id=f"{area}-fallback",
            target_split_label="full_body",
            target_muscles=[
                blocked_candidate["primary_muscle"],
                safe_candidate["primary_muscle"],
            ],
            time_available_min=45,
            injury_conditions=[
                InjuryConditionEntry(
                    area_code=area,
                    status="managedWithRestrictions",
                    restricted_movement_patterns=[pattern],
                )
            ],
        )

        result = generate_fallback_routine(
            request,
            candidates=[blocked_candidate, safe_candidate],
            max_total_sets=8,
            goal="hypertrophy",
            status_reason_code="schemaError",
        )

        selected_ids = {block.exercise_id for block in result.routine_blocks}
        assert blocked_candidate["id"] not in selected_ids
        assert safe_candidate["id"] in selected_ids


class TestFallbackTrainingStyleAdjunct:
    def test_requested_crossfit_style_is_kept_as_adjunct_not_main_formula(self):
        request = RoutineRequest(
            user_id="legs-crossfit-adjunct",
            target_split_label="legs",
            training_style="crossfit",
            time_available_min=30,
            pain_areas=[],
            doms_data={},
            equipment=[],
        )
        candidates = [
            {
                "id": "machine_v_squat",
                "name_kr": "브이 스쿼트",
                "primary_muscle": "quadriceps",
                "secondary_muscle": "gluteal",
                "equipment_req": "MACHINE",
                "efficiency_tier": 1,
                "movement_type": "COMPOUND",
                "score": 120,
                "training_style_tags": ["machine_gym"],
            },
            {
                "id": "smith_squat",
                "name_kr": "스미스 머신 스쿼트",
                "primary_muscle": "quadriceps",
                "secondary_muscle": "gluteal",
                "equipment_req": "SMITH_MACHINE",
                "efficiency_tier": 2,
                "movement_type": "COMPOUND",
                "score": 110,
                "training_style_tags": ["machine_gym"],
            },
            {
                "id": "leg_curl",
                "name_kr": "머신 레그 컬",
                "primary_muscle": "hamstring",
                "secondary_muscle": "",
                "equipment_req": "MACHINE",
                "efficiency_tier": 2,
                "movement_type": "ISOLATION",
                "score": 100,
                "training_style_tags": ["machine_gym"],
            },
            {
                "id": "frog_pump",
                "name_kr": "프로그 펌프",
                "primary_muscle": "gluteal",
                "secondary_muscle": "",
                "equipment_req": "BODYWEIGHT",
                "efficiency_tier": 3,
                "movement_type": "ISOLATION",
                "score": 95,
                "training_style_tags": ["bodybuilding_gym"],
            },
            {
                "id": "amrap_thruster",
                "name_kr": "덤벨 AMRAP 스러스터",
                "primary_muscle": "quadriceps",
                "secondary_muscle": "front-deltoids, triceps",
                "equipment_req": "DUMBBELL",
                "efficiency_tier": 4,
                "movement_type": "COMPOUND",
                "score": 60,
                "training_style_tags": ["crossfit"],
                "score_boosts": [
                    {
                        "ruleCode": "training_style_match_boost",
                        "constraintCode": "crossfit",
                        "score": 8,
                        "reason": "요청한 보조 운동 스타일과 맞는 후보라 대체/마무리 후보 우선순위가 올라갔습니다.",
                    }
                ],
            },
        ]

        result = generate_fallback_routine(
            request,
            candidates=candidates,
            max_total_sets=10,
            goal="hypertrophy",
            status_reason_code="schemaError",
        )

        exercise_ids = [block.exercise_id for block in result.routine_blocks]
        assert result.generation_status == "fallback"
        assert exercise_ids[0] in {"machine_v_squat", "smith_squat"}
        assert "amrap_thruster" in exercise_ids
        assert exercise_ids.index("amrap_thruster") > 0
        adjunct_block = next(block for block in result.routine_blocks if block.exercise_id == "amrap_thruster")
        assert any(
            getattr(boost, "rule_code", None) == "training_style_match_boost"
            for boost in adjunct_block.boosts
        )

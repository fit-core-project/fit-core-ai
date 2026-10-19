"""
post_validation_guard + validate_and_repair_routine_output — hard constraint guard tests.

Scenarios:
  - legs 분할 요청인데 LLM이 chest 운동을 추천 → hallucination 위반 감지, chest 제거
  - 무릎 부상(knees) 인데 스쿼트 추천 → pain_area 위반 감지, 스쿼트 차단
  - 안전한 운동만 포함된 출력 → 위반 없음(clean)
  - 안전 후보가 전혀 없는 경우 → (None, "emptyCandidate")
"""
import pytest

from engines.llm_parser import validate_and_repair_routine_output
from engines.post_validation_guard import GuardReport, Violation, audit_routine_output
from engines.schemas import (
    LLMExercisePlan,
    LLMRoutineOutput,
    LLMSubstitutionCandidate,
    InjuryConditionEntry,
    PainAreaEntry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def legs_candidates():
    """legs 분할에 적합한 후보 운동 3종 (squat은 무릎 pain trigger 포함)."""
    return [
        {
            "id": "squat",
            "name_kr": "스쿼트",
            "name_en": "Squat",
            "primary_muscle": "quadriceps",
            "equipment_req": "BARBELL",
            "pain_triggers": "knees",
            "movement_type": "COMPOUND",
        },
        {
            "id": "leg_press",
            "name_kr": "레그프레스",
            "name_en": "Leg Press",
            "primary_muscle": "quadriceps",
            "equipment_req": "MACHINE",
            "pain_triggers": None,
            "movement_type": "COMPOUND",
        },
        {
            "id": "leg_curl",
            "name_kr": "레그컬",
            "name_en": "Leg Curl",
            "primary_muscle": "hamstrings",
            "equipment_req": "MACHINE",
            "pain_triggers": None,
            "movement_type": "ISOLATION",
        },
    ]


def _make_output(*exercise_ids_and_names: tuple) -> LLMRoutineOutput:
    """LLMRoutineOutput 헬퍼 — (exercise_id, exercise_name) 쌍을 받아 생성."""
    exercises = [
        LLMExercisePlan(
            exercise_id=eid,
            exercise_name=ename,
            movement_pattern="compoundPush",
            primary_muscles=["chest"],
            sets=3,
            target_reps=10,
            rest_time_sec=90,
            target_rir=2,
            exercise_rationale="test",
        )
        for eid, ename in exercise_ids_and_names
    ]
    return LLMRoutineOutput(
        total_estimated_time=45,
        summary_title="테스트 루틴",
        rationale_summary=["테스트"],
        warnings=[],
        exercises=exercises,
    )


# ---------------------------------------------------------------------------
# audit_routine_output — 순수 감사 테스트
# ---------------------------------------------------------------------------

class TestAuditRoutineOutput:
    def test_chest_exercise_in_legs_candidates_is_hallucination(self, legs_candidates):
        """legs 후보에 없는 chest 운동 → hallucination 위반"""
        output = _make_output(("barbell_bench_press", "바벨 벤치프레스"))
        report = audit_routine_output(output, legs_candidates, blocked_equipment=[], pain_areas=[])

        assert report.has_violations
        assert len(report.violations) == 1
        assert report.violations[0].constraint == "hallucination"
        assert report.violations[0].exercise_id == "barbell_bench_press"
        assert report.level == "blocked"

    def test_squat_with_knees_injury_is_pain_area_violation(self, legs_candidates):
        """무릎 부상 + 스쿼트(pain_triggers=knees) → pain_area 위반"""
        output = _make_output(("squat", "스쿼트"))
        pain_areas = [PainAreaEntry(body_part="knees")]
        report = audit_routine_output(output, legs_candidates, blocked_equipment=[], pain_areas=pain_areas)

        assert report.has_violations
        assert report.violations[0].constraint == "pain_area"
        assert report.violations[0].exercise_id == "squat"
        assert "knees" in report.violations[0].detail

    def test_knee_injury_detects_direct_quadriceps_even_without_trigger_text(self, legs_candidates):
        """부상 판정은 pain_triggers 문자열에만 의존하지 않는다."""
        output = _make_output(("leg_press", "레그프레스"))
        report = audit_routine_output(
            output,
            legs_candidates,
            blocked_equipment=[],
            pain_areas=[PainAreaEntry(body_part="knees")],
        )

        assert report.has_violations
        assert report.violations[0].constraint == "pain_area"
        assert "knees" in report.violations[0].detail

    def test_clean_output_produces_no_violations(self, legs_candidates):
        """안전한 운동(leg_press)만 포함 → 위반 없음"""
        output = _make_output(("leg_press", "레그프레스"))
        report = audit_routine_output(output, legs_candidates, blocked_equipment=[], pain_areas=[])

        assert not report.has_violations
        assert report.level == "clean"

    def test_equipment_violation_detected(self, legs_candidates):
        """barbell 장비 차단 상태에서 스쿼트(BARBELL) 추천 → equipment 위반"""
        output = _make_output(("squat", "스쿼트"))
        report = audit_routine_output(output, legs_candidates, blocked_equipment=["BARBELL"], pain_areas=[])

        assert report.has_violations
        assert report.violations[0].constraint == "equipment"
        assert report.violations[0].exercise_id == "squat"

    def test_restricted_shoulder_movement_is_detected_even_with_unfiltered_candidates(self):
        candidates = [{
            "id": "overhead_press",
            "name_kr": "덤벨 오버헤드 프레스",
            "name_en": "Dumbbell Overhead Press",
            "primary_muscle": "front-deltoids",
            "equipment_req": "DUMBBELL",
            "pain_triggers": None,
            "movement_type": "COMPOUND",
            "overhead_position_required": 1,
            "shoulder_elevation_plane": "overhead",
        }]
        output = _make_output(("overhead_press", "덤벨 오버헤드 프레스"))

        report = audit_routine_output(
            output,
            candidates,
            blocked_equipment=[],
            pain_areas=[],
            shoulder_condition_status="managedWithRestrictions",
            restricted_movement_patterns=["overheadPosition"],
        )

        assert report.has_violations
        assert report.violations[0].constraint == "shoulder_movement"
        assert "overheadPosition" in report.violations[0].detail

    def test_restricted_shoulder_substitution_is_detected(self):
        candidates = [
            {
                "id": "supported_curl",
                "name_kr": "프리처 덤벨 컬",
                "primary_muscle": "biceps",
                "equipment_req": "DUMBBELL",
                "pain_triggers": None,
                "movement_type": "ISOLATION",
            },
            {
                "id": "bench_press",
                "name_kr": "덤벨 벤치프레스",
                "primary_muscle": "chest",
                "equipment_req": "DUMBBELL",
                "pain_triggers": None,
                "movement_type": "COMPOUND",
            },
        ]
        output = _make_output(("supported_curl", "프리처 덤벨 컬"))
        output.exercises[0].substitution_candidates = [
            LLMSubstitutionCandidate(
                exercise_id="bench_press",
                exercise_name="덤벨 벤치프레스",
                reason="대체",
            )
        ]

        report = audit_routine_output(
            output,
            candidates,
            blocked_equipment=[],
            pain_areas=[],
            shoulder_condition_status="managedWithRestrictions",
            restricted_movement_patterns=["horizontalPush"],
        )

        assert len(report.violations) == 1
        assert report.violations[0].constraint == "shoulder_movement"
        assert "substitution candidate" in report.violations[0].detail

    def test_managed_lumbar_movement_is_detected_after_llm_selection(self):
        candidates = [{
            "id": "deadlift",
            "name_kr": "바벨 데드리프트",
            "primary_muscle": "hamstring",
            "equipment_req": "BARBELL",
            "pain_triggers": None,
            "movement_type": "COMPOUND",
            "lumbar_load": "high",
        }]
        output = _make_output(("deadlift", "바벨 데드리프트"))

        report = audit_routine_output(
            output,
            candidates,
            blocked_equipment=[],
            pain_areas=[],
            injury_conditions=[
                InjuryConditionEntry(
                    area_code="lumbar",
                    status="managedWithRestrictions",
                    restricted_movement_patterns=["hipHinge"],
                )
            ],
        )

        assert report.has_violations
        assert report.violations[0].constraint == "managed_injury_movement"
        assert "hipHinge" in report.violations[0].detail

    def test_hallucinated_substitution_is_detected(self, legs_candidates):
        output = _make_output(("leg_press", "레그프레스"))
        output.exercises[0].substitution_candidates = [
            LLMSubstitutionCandidate(
                exercise_id="ghost_leg_press",
                exercise_name="유령 레그프레스",
                reason="대체",
            )
        ]

        report = audit_routine_output(
            output,
            legs_candidates,
            blocked_equipment=[],
            pain_areas=[],
        )

        assert len(report.violations) == 1
        assert report.violations[0].constraint == "hallucination"
        assert "substitution candidate" in report.violations[0].detail


# ---------------------------------------------------------------------------
# validate_and_repair_routine_output — 차단 및 교체 테스트
# ---------------------------------------------------------------------------

class TestValidateAndRepair:
    def test_chest_exercise_absent_from_legs_output_after_repair(self, legs_candidates):
        """legs 후보에 없는 chest 운동 → 안전 후보로 교체, chest 완전 제거"""
        output = _make_output(("barbell_bench_press", "바벨 벤치프레스"))
        result, reason = validate_and_repair_routine_output(
            output, legs_candidates, blocked_equipment=[], pain_areas=[]
        )

        assert result is not None, "안전 후보가 존재하므로 repair 성공해야 함"
        exercise_ids = [e.exercise_id for e in result.exercises]
        assert "barbell_bench_press" not in exercise_ids, "chest 운동이 최종 출력에 남아있으면 안 됨"

    def test_squat_blocked_when_knees_injury_present(self, legs_candidates):
        """무릎 부상 시 사두근 슬롯을 다른 근육 운동으로 억지 교체하지 않는다."""
        output = _make_output(("squat", "스쿼트"))
        pain_areas = [PainAreaEntry(body_part="knees")]
        result, reason = validate_and_repair_routine_output(
            output, legs_candidates, blocked_equipment=[], pain_areas=pain_areas
        )

        assert result is None
        assert reason == "emptyCandidate"

    def test_no_safe_candidates_returns_empty_candidate(self):
        """안전 후보가 하나도 없으면 (None, 'emptyCandidate') 반환"""
        output = _make_output(("barbell_bench_press", "바벨 벤치프레스"))
        result, reason = validate_and_repair_routine_output(
            output, candidates=[], blocked_equipment=[], pain_areas=[]
        )

        assert result is None
        assert reason == "emptyCandidate"

    def test_warning_added_to_payload_when_repair_occurs(self, legs_candidates):
        """repair 발생 시 response payload warnings에 guard 메시지 포함"""
        output = _make_output(("barbell_bench_press", "바벨 벤치프레스"))
        result, reason = validate_and_repair_routine_output(
            output, legs_candidates, blocked_equipment=[], pain_areas=[]
        )

        assert result is not None
        assert any("현재 조건을 반영해" in w for w in result.warnings), "사용자용 교체 안내가 warnings에 없음"
        assert reason == "none"

    def test_repair_cannot_reintroduce_restricted_shoulder_movement(self):
        candidates = [
            {
                "id": "overhead_press",
                "name_kr": "덤벨 오버헤드 프레스",
                "name_en": "Dumbbell Overhead Press",
                "primary_muscle": "front-deltoids",
                "equipment_req": "DUMBBELL",
                "pain_triggers": None,
                "movement_type": "COMPOUND",
                "overhead_position_required": 1,
                "shoulder_elevation_plane": "overhead",
            },
            {
                "id": "lateral_raise",
                "name_kr": "덤벨 레터럴 레이즈",
                "name_en": "Dumbbell Lateral Raise",
                "primary_muscle": "front-deltoids",
                "equipment_req": "DUMBBELL",
                "pain_triggers": None,
                "movement_type": "ISOLATION",
                "overhead_position_required": 0,
                "shoulder_elevation_plane": "frontal",
            },
        ]
        output = _make_output(("overhead_press", "덤벨 오버헤드 프레스"))

        result, reason = validate_and_repair_routine_output(
            output,
            candidates,
            blocked_equipment=[],
            pain_areas=[],
            shoulder_condition_status="managedWithRestrictions",
            restricted_movement_patterns=["overheadPosition"],
        )

        assert result is not None
        assert reason == "none"
        assert [exercise.exercise_id for exercise in result.exercises] == ["lateral_raise"]

    def test_repair_keeps_only_safe_related_substitution_candidates(self):
        candidates = [
            {
                "id": "supported_curl",
                "name_kr": "프리처 덤벨 컬",
                "primary_muscle": "biceps",
                "equipment_req": "DUMBBELL",
                "pain_triggers": None,
                "movement_type": "ISOLATION",
            },
            {
                "id": "cable_curl",
                "name_kr": "케이블 컬",
                "primary_muscle": "biceps",
                "equipment_req": "CABLE",
                "pain_triggers": None,
                "movement_type": "ISOLATION",
            },
            {
                "id": "bench_press",
                "name_kr": "덤벨 벤치프레스",
                "primary_muscle": "chest",
                "equipment_req": "DUMBBELL",
                "pain_triggers": None,
                "movement_type": "COMPOUND",
            },
        ]
        output = _make_output(("supported_curl", "프리처 덤벨 컬"))
        output.exercises[0].substitution_candidates = [
            LLMSubstitutionCandidate(
                exercise_id="cable_curl",
                exercise_name="잘못된 이름",
                reason="LLM 원문",
            ),
            LLMSubstitutionCandidate(
                exercise_id="bench_press",
                exercise_name="덤벨 벤치프레스",
                reason="제한 운동",
            ),
            LLMSubstitutionCandidate(
                exercise_id="ghost_curl",
                exercise_name="유령 컬",
                reason="후보 밖 운동",
            ),
        ]

        result, reason = validate_and_repair_routine_output(
            output,
            candidates,
            blocked_equipment=[],
            pain_areas=[],
            shoulder_condition_status="managedWithRestrictions",
            restricted_movement_patterns=["horizontalPush"],
        )

        assert result is not None
        assert reason == "none"
        assert [
            substitution.exercise_id
            for substitution in result.exercises[0].substitution_candidates
        ] == ["cable_curl"]
        assert result.exercises[0].substitution_candidates[0].exercise_name == "케이블 컬"
        assert any("대체 운동 2개를 제외" in warning for warning in result.warnings)

    def test_repair_accepts_explicit_relation_or_same_primary_substitution(self):
        candidates = [
            {
                "id": "supported_curl",
                "name_kr": "프리처 덤벨 컬",
                "primary_muscle": "biceps",
                "equipment_req": "DUMBBELL",
                "pain_triggers": None,
                "movement_type": "ISOLATION",
                "substitution_options": [
                    {"exercise_id": "cable_curl", "relation_type": "equipmentAlternative"},
                ],
            },
            {
                "id": "cable_curl",
                "name_kr": "케이블 컬",
                "primary_muscle": "biceps",
                "equipment_req": "CABLE",
                "pain_triggers": None,
                "movement_type": "ISOLATION",
            },
            {
                "id": "hammer_curl",
                "name_kr": "덤벨 해머 컬",
                "primary_muscle": "biceps",
                "equipment_req": "DUMBBELL",
                "pain_triggers": None,
                "movement_type": "ISOLATION",
            },
        ]
        output = _make_output(("supported_curl", "프리처 덤벨 컬"))
        output.exercises[0].substitution_candidates = [
            LLMSubstitutionCandidate(
                exercise_id="cable_curl",
                exercise_name="케이블 컬",
                reason="명시 관계",
            ),
            LLMSubstitutionCandidate(
                exercise_id="hammer_curl",
                exercise_name="덤벨 해머 컬",
                reason="같은 주동근",
            ),
        ]

        result, reason = validate_and_repair_routine_output(
            output,
            candidates,
            blocked_equipment=[],
            pain_areas=[],
        )

        assert result is not None
        assert reason == "none"
        assert [
            substitution.exercise_id
            for substitution in result.exercises[0].substitution_candidates
        ] == ["cable_curl", "hammer_curl"]

    def test_repair_removes_substitution_blocked_by_equipment_and_injury(self):
        candidates = [
            {
                "id": "cable_curl",
                "name_kr": "케이블 컬",
                "primary_muscle": "biceps",
                "equipment_req": "CABLE",
                "pain_triggers": None,
                "movement_type": "ISOLATION",
            },
            {
                "id": "barbell_bench_press",
                "name_kr": "바벨 벤치프레스",
                "primary_muscle": "chest",
                "equipment_req": "BARBELL",
                "pain_triggers": "front-deltoids",
                "movement_type": "COMPOUND",
            },
        ]
        output = _make_output(("cable_curl", "케이블 컬"))
        output.exercises[0].substitution_candidates = [
            LLMSubstitutionCandidate(
                exercise_id="barbell_bench_press",
                exercise_name="바벨 벤치프레스",
                reason="대체",
            )
        ]

        result, reason = validate_and_repair_routine_output(
            output,
            candidates,
            blocked_equipment=["BARBELL"],
            pain_areas=[PainAreaEntry(body_part="front-deltoids")],
        )

        assert result is not None
        assert reason == "none"
        assert result.exercises[0].substitution_candidates == []
        assert any("대체 운동 1개를 제외" in warning for warning in result.warnings)

    def test_duplicate_exercise_id_is_repaired_to_next_same_primary_candidate(self, legs_candidates):
        """LLM이 같은 exercise_id를 반복하면 두 번째 항목은 같은 주동근 안전 후보로 교체"""
        candidates = [
            *legs_candidates,
            {
                "id": "leg_extension",
                "name_kr": "레그 익스텐션",
                "name_en": "Leg Extension",
                "primary_muscle": "quadriceps",
                "equipment_req": "MACHINE",
                "pain_triggers": None,
                "movement_type": "ISOLATION",
            },
        ]
        output = _make_output(("leg_press", "레그프레스"), ("leg_press", "레그프레스"))

        result, reason = validate_and_repair_routine_output(
            output, candidates, blocked_equipment=["BARBELL"], pain_areas=[]
        )

        assert result is not None
        exercise_ids = [e.exercise_id for e in result.exercises]
        assert len(exercise_ids) == len(set(exercise_ids))
        assert exercise_ids[0] == "leg_press"
        assert exercise_ids[1] == "leg_extension"
        assert any("현재 조건을 반영해" in w for w in result.warnings)
        assert reason == "none"

    def test_duplicate_exercise_id_without_same_primary_drops_instead_of_unrelated_replacement(self, legs_candidates):
        """같은 주동근 대체가 없으면 pull/legs 루틴에 무관한 운동을 끼워 넣지 않고 제외한다."""
        output = _make_output(("leg_press", "레그프레스"), ("leg_press", "레그프레스"))

        result, reason = validate_and_repair_routine_output(
            output, legs_candidates, blocked_equipment=["BARBELL"], pain_areas=[]
        )

        assert result is not None
        exercise_ids = [e.exercise_id for e in result.exercises]
        assert exercise_ids == ["leg_press"]
        assert "leg_curl" not in exercise_ids
        assert any("비슷한 운동이 겹치지 않도록" in warning for warning in result.warnings)
        assert reason == "none"

    def test_duplicate_user_facing_display_name_is_repaired(self):
        """서로 다른 catalog ID라도 사용자 표시명이 같으면 다음 안전 후보로 교체"""
        candidates = [
            {
                "id": "smith_chest_press_flat",
                "name_kr": "스미스 머신 플랫 체스트 프레스",
                "user_facing_display_name_kr": "스미스 머신 체스트 프레스",
                "primary_muscle": "chest",
                "equipment_req": "MACHINE",
                "pain_triggers": None,
                "movement_type": "COMPOUND",
            },
            {
                "id": "smith_chest_press_flat_duplicate",
                "name_kr": "스미스 머신 플랫 체스트 프레스",
                "user_facing_display_name_kr": "스미스 머신 체스트 프레스",
                "primary_muscle": "chest",
                "equipment_req": "MACHINE",
                "pain_triggers": None,
                "movement_type": "COMPOUND",
            },
            {
                "id": "machine_chest_press",
                "name_kr": "머신 체스트 프레스",
                "user_facing_display_name_kr": "머신 체스트 프레스",
                "primary_muscle": "chest",
                "equipment_req": "MACHINE",
                "pain_triggers": None,
                "movement_type": "COMPOUND",
            },
        ]
        output = _make_output(
            ("smith_chest_press_flat", "스미스 머신 체스트 프레스"),
            ("smith_chest_press_flat_duplicate", "스미스 머신 체스트 프레스"),
        )

        result, reason = validate_and_repair_routine_output(
            output, candidates, blocked_equipment=[], pain_areas=[]
        )

        assert result is not None
        exercise_names = [exercise.exercise_name for exercise in result.exercises]
        exercise_ids = [exercise.exercise_id for exercise in result.exercises]
        assert exercise_names == ["스미스 머신 플랫 체스트 프레스", "머신 체스트 프레스"]
        assert exercise_ids == ["smith_chest_press_flat", "machine_chest_press"]
        assert len(exercise_names) == len(set(exercise_names))
        assert any("현재 조건을 반영해" in w for w in result.warnings)
        assert reason == "none"

    def test_reverse_pec_deck_keeps_rear_delt_context_in_display_name(self):
        """후면삼각 리버스 펙덱을 일반 가슴 펙덱처럼 보이게 축약하지 않는다."""
        candidates = [
            {
                "id": "reverse_pec_deck",
                "name_kr": "머신 리버스 펙덱 뉴트럴 리어 델트 레이즈",
                "display_name_ko": "머신 리버스 펙덱 뉴트럴 리어 델트 레이즈",
                "user_facing_display_name_kr": "머신 펙덱 플라이",
                "primary_muscle": "back-deltoids",
                "equipment_req": "MACHINE",
                "pain_triggers": None,
                "movement_type": "ISOLATION",
            },
        ]
        output = _make_output(("reverse_pec_deck", "머신 펙덱 플라이"))

        result, reason = validate_and_repair_routine_output(
            output, candidates, blocked_equipment=[], pain_areas=[]
        )

        assert result is not None
        assert result.exercises[0].exercise_name == "머신 리버스 펙덱 뉴트럴 리어 델트 레이즈"
        assert reason == "none"

    def test_duplicate_display_name_without_replacement_uses_item_level_warning(self):
        """중복 표시명 후보 하나를 제외할 때 운동 전체 제거처럼 오해되는 문구를 쓰지 않는다."""
        candidates = [
            {
                "id": "smith_chest_press_flat",
                "name_kr": "스미스 머신 플랫 체스트 프레스",
                "user_facing_display_name_kr": "스미스 머신 체스트 프레스",
                "primary_muscle": "chest",
                "equipment_req": "MACHINE",
                "pain_triggers": None,
                "movement_type": "COMPOUND",
            },
            {
                "id": "smith_chest_press_flat_duplicate",
                "name_kr": "스미스 머신 플랫 체스트 프레스",
                "user_facing_display_name_kr": "스미스 머신 체스트 프레스",
                "primary_muscle": "chest",
                "equipment_req": "MACHINE",
                "pain_triggers": None,
                "movement_type": "COMPOUND",
            },
        ]
        output = _make_output(
            ("smith_chest_press_flat", "스미스 머신 체스트 프레스"),
            ("smith_chest_press_flat_duplicate", "스미스 머신 체스트 프레스"),
        )

        result, reason = validate_and_repair_routine_output(
            output, candidates, blocked_equipment=[], pain_areas=[]
        )

        assert result is not None
        assert [exercise.exercise_id for exercise in result.exercises] == ["smith_chest_press_flat"]
        assert any("비슷한 운동이 겹치지 않도록" in warning for warning in result.warnings)
        assert reason == "none"

    def test_duplicate_movement_family_is_removed_even_when_name_and_equipment_differ(self):
        candidates = [
            {
                "id": "machine_hammer_curl",
                "name_kr": "머신 해머 컬",
                "primary_muscle": "biceps",
                "equipment_req": "MACHINE",
                "movement_type": "ISOLATION",
            },
            {
                "id": "dumbbell_crossbody_hammer_curl",
                "name_kr": "덤벨 크로스바디 해머 컬",
                "primary_muscle": "biceps",
                "equipment_req": "DUMBBELL",
                "movement_type": "ISOLATION",
            },
        ]
        output = _make_output(
            ("machine_hammer_curl", "머신 해머 컬"),
            ("dumbbell_crossbody_hammer_curl", "덤벨 크로스바디 해머 컬"),
        )

        result, reason = validate_and_repair_routine_output(
            output, candidates, blocked_equipment=[], pain_areas=[]
        )

        assert result is not None
        assert [exercise.exercise_id for exercise in result.exercises] == ["machine_hammer_curl"]
        assert any("비슷한 운동이 겹치지 않도록" in warning for warning in result.warnings)
        assert reason == "none"

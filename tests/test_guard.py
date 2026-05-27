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
from engines.schemas import LLMExercisePlan, LLMRoutineOutput, PainAreaEntry


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
        """무릎 부상 시 스쿼트 → 차단 후 안전 후보(leg_press 또는 leg_curl)로 교체"""
        output = _make_output(("squat", "스쿼트"))
        pain_areas = [PainAreaEntry(body_part="knees")]
        result, reason = validate_and_repair_routine_output(
            output, legs_candidates, blocked_equipment=[], pain_areas=pain_areas
        )

        assert result is not None, "무릎 부상 없는 후보가 있으므로 repair 성공해야 함"
        exercise_ids = [e.exercise_id for e in result.exercises]
        assert "squat" not in exercise_ids, "스쿼트가 최종 출력에 남아있으면 안 됨"
        assert reason == "none"

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
        assert any("Guard" in w for w in result.warnings), "guard 교체 경고가 warnings에 없음"
        assert reason == "none"

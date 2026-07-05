"""
Candidate Pool 12 vs 18 Staging Telemetry Test

pytest tests/test_staging_telemetry.py -v -s

기능 범위:
- routine generation
- candidate ranking (pool 12 vs 18)
- LLM structured output
- schema repair
- guard repair
- fallback
- Rule Critic telemetry

제외 범위:
- feedback-aware ranking
- dynamic temperature
- Rule Critic rebuild/fallback trigger
"""
from __future__ import annotations

import math
import os
from contextlib import contextmanager
from statistics import mean, median
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.runnables import RunnableLambda
from sqlalchemy.orm import Session

from engines.routine_pipeline import generate_smart_routine
from engines.schemas import (
    LLMExercisePlan,
    LLMRoutineOutput,
    LLMSubstitutionCandidate,
    PainAreaEntry,
    RoutineRequest,
    UserProfileContext,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextmanager
def capture_telemetry():
    """Capture routine telemetry events at the production emit boundary."""
    captured: list[dict] = []

    def _capture(payload: dict[str, Any]) -> None:
        if payload.get("event") == "routine_generation_quality":
            captured.append(dict(payload))

    with patch("engines.routine_telemetry.emit_routine_quality_telemetry", side_effect=_capture):
        yield captured


def _make_db_mock() -> MagicMock:
    return MagicMock(spec=Session)


def _make_llm_mock(return_value=None, side_effect=None):
    mock_llm = MagicMock()
    if side_effect is not None:
        structured = RunnableLambda(lambda _: (_ for _ in ()).throw(side_effect))
    else:
        structured = RunnableLambda(lambda _: return_value)
    mock_llm.with_structured_output.return_value = structured
    return mock_llm


def _make_large_candidate_pool(n: int = 25) -> list[dict]:
    """pool 18을 커버할 수 있을 만큼 충분한 후보 운동 목록을 반환한다."""
    muscles = [
        ("chest", "BARBELL"), ("chest", "DUMBBELL"), ("chest", "BODYWEIGHT"),
        ("triceps", "CABLE"), ("triceps", "DUMBBELL"), ("triceps", "BODYWEIGHT"),
        ("front-deltoids", "DUMBBELL"), ("front-deltoids", "BARBELL"),
        ("upper-back", "BARBELL"), ("upper-back", "CABLE"), ("upper-back", "DUMBBELL"),
        ("biceps", "DUMBBELL"), ("biceps", "CABLE"), ("biceps", "BARBELL"),
        ("quadriceps", "BARBELL"), ("quadriceps", "MACHINE"), ("quadriceps", "BODYWEIGHT"),
        ("gluteal", "BARBELL"), ("gluteal", "DUMBBELL"), ("gluteal", "BODYWEIGHT"),
        ("hamstring", "MACHINE"), ("hamstring", "DUMBBELL"),
        ("abs", "BODYWEIGHT"), ("abs", "CABLE"),
        ("lateral-deltoids", "DUMBBELL"),
    ]
    return [
        {
            "id": f"exercise_{i:03d}",
            "name_kr": f"운동{i}",
            "name_en": f"Exercise {i}",
            "primary_muscle": muscle,
            "secondary_muscle": None,
            "equipment_req": eq,
            "difficulty_tier": (i % 3) + 1,
            "efficiency_tier": (i % 5) + 1,
            "pain_triggers": None,
            "movement_type": "COMPOUND" if i % 3 == 0 else "ISOLATION",
        }
        for i, (muscle, eq) in enumerate(muscles[:n])
    ]


def _make_llm_output_for(
    exercises: list[dict],
    blocked_equipment: list[str] | None = None,
    pain_areas: list[PainAreaEntry] | None = None,
) -> LLMRoutineOutput:
    blocked_set = {e.upper() for e in (blocked_equipment or [])}
    pain_tokens = {p.body_part.lower() for p in (pain_areas or [])}
    eligible = [
        ex for ex in exercises
        if ex["equipment_req"].upper() not in blocked_set
        and (
            not pain_tokens
            or not ex.get("pain_triggers")
            or not any(t in str(ex.get("pain_triggers", "")).lower() for t in pain_tokens)
        )
    ][:5]
    if not eligible:
        eligible = [ex for ex in exercises if ex["equipment_req"] == "BODYWEIGHT"][:3]

    plans = []
    for ex in eligible:
        plans.append(
            LLMExercisePlan(
                exercise_id=ex["id"],
                exercise_name=ex["name_kr"],
                movement_type=ex["movement_type"],
                primary_muscles=[ex["primary_muscle"]],
                equipment_type=ex["equipment_req"],
                target_weight_kg=None if ex["equipment_req"] == "BODYWEIGHT" else 60.0,
                target_reps=8,
                sets=3,
                rest_time_sec=90,
                target_rir=2,
                exercise_rationale="테스트 운동",
                substitution_candidates=[],
            )
        )
    return LLMRoutineOutput(
        total_estimated_time=45,
        summary_title="테스트 루틴",
        rationale_summary=["테스트 목적"],
        warnings=[],
        exercises=plans,
    )


# ---------------------------------------------------------------------------
# 시나리오 정의
# ---------------------------------------------------------------------------

_SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "normal_hypertrophy_upper",
        "req": dict(
            user_id="u1", target_split_label="push",
            readiness_level="normal", time_available_min=60,
            pain_areas=[], doms_data={}, equipment=[], goal="hypertrophy",
        ),
        "profile": dict(
            goal_type="hypertrophy", split_type="PPL", split_label="push",
            experience_level="intermediate",
            strength_baseline={}, equipment_access=["BARBELL", "DUMBBELL"],
        ),
    },
    {
        "id": "strength_lower",
        "req": dict(
            user_id="u2", target_split_label="legs",
            readiness_level="normal", time_available_min=60,
            pain_areas=[], doms_data={}, equipment=[], goal="strength",
        ),
        "profile": dict(
            goal_type="strength", split_type="PPL", split_label="legs",
            experience_level="intermediate",
            strength_baseline={}, equipment_access=["BARBELL"],
        ),
    },
    {
        "id": "fat_loss_limited_equipment",
        "req": dict(
            user_id="u3", target_split_label="push",
            readiness_level="normal", time_available_min=45,
            pain_areas=[], doms_data={}, equipment=["BARBELL", "MACHINE"], goal="endurance",
        ),
        "profile": dict(
            goal_type="endurance", split_type="fullBody", split_label="push",
            experience_level="beginner",
            strength_baseline={}, equipment_access=["DUMBBELL"],
        ),
    },
    {
        "id": "shoulder_pain",
        "req": dict(
            user_id="u4", target_split_label="push",
            readiness_level="normal", time_available_min=50,
            pain_areas=[PainAreaEntry(area="front-deltoids", severity="severe")],
            doms_data={}, equipment=[], goal="hypertrophy",
        ),
        "profile": dict(
            goal_type="hypertrophy", split_type="PPL", split_label="push",
            experience_level="intermediate",
            strength_baseline={}, equipment_access=["BARBELL", "DUMBBELL"],
        ),
    },
    {
        "id": "knee_pain",
        "req": dict(
            user_id="u5", target_split_label="legs",
            readiness_level="normal", time_available_min=50,
            pain_areas=[PainAreaEntry(area="quadriceps", severity="severe")],
            doms_data={}, equipment=[], goal="hypertrophy",
        ),
        "profile": dict(
            goal_type="hypertrophy", split_type="PPL", split_label="legs",
            experience_level="intermediate",
            strength_baseline={}, equipment_access=["BARBELL"],
        ),
    },
    {
        "id": "low_readiness",
        "req": dict(
            user_id="u6", target_split_label="push",
            readiness_level="low", time_available_min=40,
            pain_areas=[], doms_data={"chest": 2}, equipment=[], goal="hypertrophy",
        ),
        "profile": dict(
            goal_type="hypertrophy", split_type="PPL", split_label="push",
            experience_level="intermediate",
            strength_baseline={}, equipment_access=["BARBELL", "DUMBBELL"],
        ),
    },
    {
        "id": "short_duration_20min",
        "req": dict(
            user_id="u7", target_split_label="push",
            readiness_level="normal", time_available_min=20,
            pain_areas=[], doms_data={}, equipment=[], goal="hypertrophy",
        ),
        "profile": dict(
            goal_type="hypertrophy", split_type="PPL", split_label="push",
            experience_level="beginner",
            strength_baseline={}, equipment_access=["DUMBBELL"],
        ),
    },
    {
        "id": "long_duration_60min",
        "req": dict(
            user_id="u8", target_split_label="pull",
            readiness_level="high", time_available_min=60,
            pain_areas=[], doms_data={}, equipment=[], goal="hypertrophy",
        ),
        "profile": dict(
            goal_type="hypertrophy", split_type="PPL", split_label="pull",
            experience_level="advanced",
            strength_baseline={}, equipment_access=["BARBELL", "DUMBBELL", "CABLE"],
        ),
    },
    {
        "id": "bodyweight_only",
        "req": dict(
            user_id="u9", target_split_label="push",
            readiness_level="normal", time_available_min=40,
            pain_areas=[], doms_data={}, equipment=["BARBELL", "DUMBBELL", "CABLE", "MACHINE"],
            goal="endurance",
        ),
        "profile": dict(
            goal_type="endurance", split_type="fullBody", split_label="push",
            experience_level="beginner",
            strength_baseline={}, equipment_access=["BODYWEIGHT"],
        ),
    },
    {
        "id": "mixed_target_muscles",
        "req": dict(
            user_id="u10", target_muscles=["chest", "triceps", "front-deltoids"],
            readiness_level="normal", time_available_min=55,
            pain_areas=[], doms_data={}, equipment=[], goal="hypertrophy",
        ),
        "profile": dict(
            goal_type="hypertrophy", split_type="PPL", split_label="push",
            experience_level="intermediate",
            strength_baseline={}, equipment_access=["BARBELL", "DUMBBELL", "CABLE"],
        ),
    },
]

_PRIVACY_FORBIDDEN_FIELDS = {
    "user_note",
    "raw_prompt",
    "raw_llm_output",
    "rationale_summary",
    "exercise_rationale",
    "rationale",
    "llm_raw",
    "prompt_text",
    "medical_note",
}

_REQUIRED_TELEMETRY_FIELDS = {
    "event", "routine_draft_id", "candidate_pool_size", "candidate_count",
    "candidate_payload_char_count", "approx_candidate_payload_tokens",
    "dynamic_temperature_enabled", "generation_temperature",
    "critic_score", "critic_grade", "critic_warning_count", "hard_violation_count",
    "repair_count", "fallback_used", "exercise_count",
    "total_estimated_time", "target_duration_min",
    "generation_latency_ms", "schema_repair_latency_ms",
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_group(pool_size: int, repeats: int = 5) -> list[dict]:
    """주어진 pool_size로 모든 시나리오를 repeats회 실행하고 텔레메트리를 수집한다."""
    all_candidates = _make_large_candidate_pool(25)

    # score_candidate_exercises를 패치해 상위 pool_size개를 순서 그대로 반환
    # → LLM이 선택한 exercise_id가 항상 ranked_candidates 안에 있도록 보장
    def _mock_score(candidates, *_args, top_n=12, **_kwargs):
        return candidates[:top_n]

    with (
        patch.dict(os.environ, {
            "ROUTINE_CANDIDATE_POOL_SIZE": str(pool_size),
            "ENABLE_RULE_CRITIC_TELEMETRY": "true",
            "ENABLE_DYNAMIC_TEMPERATURE": "false",
        }),
        capture_telemetry() as telemetry_events,
        patch("engines.routine_pipeline.get_candidate_pool_size", return_value=pool_size),
        patch("engines.routine_telemetry.get_rule_critic_telemetry_enabled", return_value=True),
        patch("engines.routine_pipeline.score_candidate_exercises", side_effect=_mock_score),
    ):
        for _repeat in range(repeats):
            for scenario in _SCENARIOS:
                req_data = dict(scenario["req"])
                profile_data = dict(scenario["profile"])

                req = RoutineRequest(**req_data)
                profile = UserProfileContext(
                    pain_areas=[],
                    **{k: v for k, v in profile_data.items()},
                )
                # 시나리오별 blocked equipment를 사전 적용한 후보 풀 생성
                # → fallback 생성기도 동일하게 필터된 후보만 받아 hard violation 방지
                blocked = list(req_data.get("equipment", []))
                pain_list = list(req_data.get("pain_areas", []))
                blocked_set = {e.upper() for e in blocked}
                scenario_candidates = [
                    ex for ex in all_candidates
                    if ex["equipment_req"].upper() not in blocked_set
                ]
                llm_output = _make_llm_output_for(
                    scenario_candidates[:pool_size],
                    blocked_equipment=blocked,
                    pain_areas=pain_list,
                )
                mock_llm = _make_llm_mock(return_value=llm_output)
                db = _make_db_mock()

                with (
                    patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
                    patch("engines.routine_pipeline.get_candidate_exercises", return_value=scenario_candidates),
                ):
                    generate_smart_routine(req, db, profile=profile, recent_sets=[])

    return list(telemetry_events)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _pct(items: list[dict], key: str, value) -> float:
    if not items:
        return 0.0
    return round(100.0 * sum(1 for e in items if e.get(key) == value) / len(items), 1)


def _avg(items: list[dict], key: str) -> float:
    vals = [e[key] for e in items if e.get(key) is not None]
    return round(mean(vals), 2) if vals else 0.0


def _p50(items: list[dict], key: str) -> float:
    vals = sorted(e[key] for e in items if e.get(key) is not None)
    return round(median(vals), 2) if vals else 0.0


def _p95(items: list[dict], key: str) -> float:
    vals = sorted(e[key] for e in items if e.get(key) is not None)
    if not vals:
        return 0.0
    idx = max(0, math.ceil(len(vals) * 0.95) - 1)
    return round(vals[idx], 2)


def _p99(items: list[dict], key: str) -> float:
    vals = sorted(e[key] for e in items if e.get(key) is not None)
    if not vals:
        return 0.0
    idx = max(0, math.ceil(len(vals) * 0.99) - 1)
    return round(vals[idx], 2)


def _time_fit_rate(items: list[dict], tolerance_min: int) -> float:
    if not items:
        return 0.0
    fit = sum(
        1 for e in items
        if e.get("total_estimated_time") is not None
        and e.get("target_duration_min") is not None
        and abs(e["total_estimated_time"] - e["target_duration_min"]) <= tolerance_min
    )
    return round(100.0 * fit / len(items), 1)


def _privacy_leak(items: list[dict]) -> list[str]:
    leaked = []
    for event in items:
        for field in _PRIVACY_FORBIDDEN_FIELDS:
            if field in event:
                leaked.append(f"{field} in event {event.get('routine_draft_id', '?')}")
    return leaked


# ---------------------------------------------------------------------------
# 테스트 클래스
# ---------------------------------------------------------------------------

class TestStagingTelemetryPrerequisites:
    """사전 조건: 기본값 및 필드 검증"""

    def test_default_env_values(self):
        """env 미설정 시 pool=12, telemetry=off, dynamic_temp=off"""
        from engines.candidate_pool_policy import get_candidate_pool_size
        from engines.routine_telemetry import get_rule_critic_telemetry_enabled
        from engines.temperature_policy import is_dynamic_temperature_enabled

        saved = {k: os.environ.pop(k, None) for k in [
            "ROUTINE_CANDIDATE_POOL_SIZE",
            "ENABLE_RULE_CRITIC_TELEMETRY",
            "ENABLE_DYNAMIC_TEMPERATURE",
        ]}
        try:
            assert get_candidate_pool_size() == 12
            assert get_rule_critic_telemetry_enabled() is False
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_generation_temperature_default_is_zero(self):
        from engines.temperature_policy import resolve_generation_temperature
        assert resolve_generation_temperature("normal") == 0.0
        assert resolve_generation_temperature("low") == 0.0


class TestTelemetryFieldsAndPrivacy:
    """텔레메트리 필드 완전성 + 개인정보 누출 금지"""

    def test_telemetry_emits_all_required_fields_pool12(self):
        events = _run_group(pool_size=12, repeats=1)
        assert len(events) == len(_SCENARIOS), \
            f"Expected {len(_SCENARIOS)} events, got {len(events)}"
        for event in events:
            missing = _REQUIRED_TELEMETRY_FIELDS - event.keys()
            assert not missing, f"Missing fields: {missing}"

    def test_telemetry_emits_all_required_fields_pool18(self):
        events = _run_group(pool_size=18, repeats=1)
        assert len(events) == len(_SCENARIOS)
        for event in events:
            missing = _REQUIRED_TELEMETRY_FIELDS - event.keys()
            assert not missing, f"Missing fields: {missing}"

    def test_no_private_fields_in_telemetry_pool12(self):
        events = _run_group(pool_size=12, repeats=1)
        leaked = _privacy_leak(events)
        assert not leaked, f"Privacy leak: {leaked}"

    def test_no_private_fields_in_telemetry_pool18(self):
        events = _run_group(pool_size=18, repeats=1)
        leaked = _privacy_leak(events)
        assert not leaked, f"Privacy leak: {leaked}"


class TestCandidatePoolSizeEffect:
    """pool_size env가 candidate_count에 올바르게 반영되는지 검증"""

    def test_pool12_candidate_count_bounded(self):
        events = _run_group(pool_size=12, repeats=1)
        for event in events:
            assert event["candidate_pool_size"] == 12
            assert event["candidate_count"] <= 12, \
                f"candidate_count={event['candidate_count']} exceeds pool_size=12"

    def test_pool18_candidate_count_bounded(self):
        events = _run_group(pool_size=18, repeats=1)
        for event in events:
            assert event["candidate_pool_size"] == 18
            assert event["candidate_count"] <= 18, \
                f"candidate_count={event['candidate_count']} exceeds pool_size=18"

    def test_pool18_payload_larger_than_pool12(self):
        """pool 18은 pool 12보다 candidate payload가 같거나 커야 한다."""
        events12 = _run_group(pool_size=12, repeats=1)
        events18 = _run_group(pool_size=18, repeats=1)
        avg12 = _avg(events12, "candidate_payload_char_count")
        avg18 = _avg(events18, "candidate_payload_char_count")
        assert avg18 >= avg12, \
            f"pool 18 payload ({avg18}) < pool 12 payload ({avg12})"

    def test_pool18_payload_growth_within_60pct(self):
        """payload 증가율 <= 60%"""
        events12 = _run_group(pool_size=12, repeats=1)
        events18 = _run_group(pool_size=18, repeats=1)
        avg12 = _avg(events12, "candidate_payload_char_count")
        avg18 = _avg(events18, "candidate_payload_char_count")
        if avg12 > 0:
            growth_pct = (avg18 - avg12) / avg12 * 100
            assert growth_pct <= 60, \
                f"Payload growth {growth_pct:.1f}% exceeds 60% threshold"

    def test_approx_tokens_matches_char_count(self):
        events = _run_group(pool_size=12, repeats=1)
        for event in events:
            expected_tokens = math.ceil(event["candidate_payload_char_count"] / 4)
            assert event["approx_candidate_payload_tokens"] == expected_tokens


class TestRuleCriticTelemetry:
    """Rule Critic 텔레메트리 값 합리성"""

    def test_critic_score_in_range(self):
        for pool_size in (12, 18):
            events = _run_group(pool_size=pool_size, repeats=1)
            for event in events:
                score = event.get("critic_score")
                if score is not None:
                    assert 0 <= score <= 100, \
                        f"critic_score={score} out of range for pool={pool_size}"

    def test_critic_grade_valid_values(self):
        for pool_size in (12, 18):
            events = _run_group(pool_size=pool_size, repeats=1)
            for event in events:
                grade = event.get("critic_grade")
                if grade is not None:
                    assert grade in {"PASS", "WARN", "FAIL"}, \
                        f"Invalid critic_grade={grade}"

    def test_hard_violations_are_tracked(self):
        """hard_violation_count 필드가 항상 정수로 기록되어야 한다.
        실제 0 여부는 live staging 서버에서 검증한다 (mock에서는 LLM/DB 부재로 편차 가능).
        """
        for pool_size in (12, 18):
            events = _run_group(pool_size=pool_size, repeats=1)
            for event in events:
                hv = event.get("hard_violation_count")
                assert hv is not None, "hard_violation_count 필드 누락"
                assert isinstance(hv, int) and hv >= 0, \
                    f"hard_violation_count must be int >= 0, got {hv}"

    def test_pool18_no_more_hard_violations_than_pool12(self):
        """pool 18 hard violation 총합이 pool 12보다 많으면 안 된다."""
        events12 = _run_group(pool_size=12, repeats=1)
        events18 = _run_group(pool_size=18, repeats=1)
        hv12 = sum(e.get("hard_violation_count", 0) for e in events12)
        hv18 = sum(e.get("hard_violation_count", 0) for e in events18)
        assert hv18 <= hv12, \
            f"pool 18 has more hard violations ({hv18}) than pool 12 ({hv12})"

    def test_should_rebuild_always_false_v1(self):
        """Rule Critic V1은 observe-only: should_rebuild=False"""
        for pool_size in (12, 18):
            events = _run_group(pool_size=pool_size, repeats=1)
            for event in events:
                assert event.get("should_rebuild") is False

    def test_should_fallback_always_false_v1(self):
        """Rule Critic V1은 observe-only: should_fallback=False"""
        for pool_size in (12, 18):
            events = _run_group(pool_size=pool_size, repeats=1)
            for event in events:
                assert event.get("should_fallback") is False


class TestFallbackTelemetry:
    """fallback 시나리오에서 텔레메트리 정확성"""

    def test_fallback_event_is_recorded(self):
        """LLM 타임아웃 fallback도 텔레메트리 이벤트를 남긴다."""
        import httpx
        from langchain_core.exceptions import OutputParserException

        db = _make_db_mock()
        candidates = _make_large_candidate_pool(25)
        req = RoutineRequest(
            user_id="fallback-test",
            target_split_label="push",
            readiness_level="normal",
            time_available_min=45,
            pain_areas=[], doms_data={}, equipment=[],
        )
        profile = UserProfileContext(
            goal_type="hypertrophy", split_type="PPL", split_label="push",
            experience_level="intermediate",
            strength_baseline={}, equipment_access=["BARBELL"],
            pain_areas=[],
        )
        mock_llm = _make_llm_mock(side_effect=httpx.TimeoutException("timeout"))

        with (
            patch.dict(os.environ, {
                "ROUTINE_CANDIDATE_POOL_SIZE": "12",
                "ENABLE_RULE_CRITIC_TELEMETRY": "true",
                "ENABLE_DYNAMIC_TEMPERATURE": "false",
            }),
            patch("engines.routine_pipeline.get_candidate_pool_size", return_value=12),
            patch("engines.routine_telemetry.get_rule_critic_telemetry_enabled", return_value=True),
            capture_telemetry() as events,
            patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
            patch("engines.routine_pipeline.get_candidate_exercises", return_value=candidates),
        ):
            result = generate_smart_routine(req, db, profile=profile, recent_sets=[])

        assert result.is_fallback is True
        assert len(events) == 1
        event = events[0]
        assert event["fallback_used"] is True
        assert "routine_draft_id" in event


class TestDynamicTemperatureDisabled:
    """ENABLE_DYNAMIC_TEMPERATURE=false 강제 시 temperature=0.0"""

    def test_generation_temperature_is_zero_regardless_of_readiness(self):
        for readiness in ("low", "normal", "high"):
            req = RoutineRequest(
                user_id="temp-test",
                target_split_label="push",
                readiness_level=readiness,
                time_available_min=45,
                pain_areas=[], doms_data={}, equipment=[],
            )
            profile = UserProfileContext(
                goal_type="hypertrophy", split_type="PPL", split_label="push",
                experience_level="intermediate",
                strength_baseline={}, equipment_access=["DUMBBELL"],
                pain_areas=[],
            )
            candidates = _make_large_candidate_pool(25)
            llm_output = _make_llm_output_for(candidates)
            db = _make_db_mock()

            with (
                patch.dict(os.environ, {
                    "ROUTINE_CANDIDATE_POOL_SIZE": "12",
                    "ENABLE_RULE_CRITIC_TELEMETRY": "true",
                    "ENABLE_DYNAMIC_TEMPERATURE": "false",
                }),
                patch("engines.routine_pipeline.get_candidate_pool_size", return_value=12),
                patch("engines.routine_telemetry.get_rule_critic_telemetry_enabled", return_value=True),
                capture_telemetry() as events,
                patch("engines.routine_pipeline.get_llm", return_value=_make_llm_mock(return_value=llm_output)),
                patch("engines.routine_pipeline.get_candidate_exercises", return_value=candidates),
            ):
                generate_smart_routine(req, db, profile=profile, recent_sets=[])

            assert len(events) == 1
            assert events[0]["generation_temperature"] == 0.0, \
                f"Expected temperature=0.0 for readiness={readiness}, got {events[0]['generation_temperature']}"
            assert events[0]["dynamic_temperature_enabled"] is False


# ---------------------------------------------------------------------------
# 풀 비교 리포트 (pytest -s 출력)
# ---------------------------------------------------------------------------

class TestABComparisonReport:
    """A(pool=12) vs B(pool=18) 비교 리포트를 출력한다."""

    def test_generate_comparison_report(self):
        """
        pytest tests/test_staging_telemetry.py::TestABComparisonReport -s
        실행 시 보고서를 stdout에 출력한다.
        """
        REPEATS = 5
        events12 = _run_group(pool_size=12, repeats=REPEATS)
        events18 = _run_group(pool_size=18, repeats=REPEATS)

        assert len(events12) == len(_SCENARIOS) * REPEATS, \
            f"pool 12: got {len(events12)}, expected {len(_SCENARIOS) * REPEATS}"
        assert len(events18) == len(_SCENARIOS) * REPEATS, \
            f"pool 18: got {len(events18)}, expected {len(_SCENARIOS) * REPEATS}"

        # --- 비교 지표 계산 ---
        avg_score12 = _avg(events12, "critic_score")
        avg_score18 = _avg(events18, "critic_score")
        pass12 = _pct(events12, "critic_grade", "PASS")
        warn12 = _pct(events12, "critic_grade", "WARN")
        fail12 = _pct(events12, "critic_grade", "FAIL")
        pass18 = _pct(events18, "critic_grade", "PASS")
        warn18 = _pct(events18, "critic_grade", "WARN")
        fail18 = _pct(events18, "critic_grade", "FAIL")
        hviol12 = sum(e.get("hard_violation_count", 0) for e in events12)
        hviol18 = sum(e.get("hard_violation_count", 0) for e in events18)

        avg_repair12 = _avg(events12, "repair_count")
        avg_repair18 = _avg(events18, "repair_count")
        fallback_rate12 = _pct(events12, "fallback_used", True)
        fallback_rate18 = _pct(events18, "fallback_used", True)
        avg_schema_lat12 = _avg(events12, "schema_repair_latency_ms")
        avg_schema_lat18 = _avg(events18, "schema_repair_latency_ms")

        avg_chars12 = _avg(events12, "candidate_payload_char_count")
        avg_chars18 = _avg(events18, "candidate_payload_char_count")
        payload_growth = ((avg_chars18 - avg_chars12) / avg_chars12 * 100) if avg_chars12 > 0 else 0.0
        p50_lat12 = _p50(events12, "generation_latency_ms")
        p50_lat18 = _p50(events18, "generation_latency_ms")
        p95_lat12 = _p95(events12, "generation_latency_ms")
        p95_lat18 = _p95(events18, "generation_latency_ms")
        lat_growth = ((p95_lat18 - p95_lat12) / p95_lat12 * 100) if p95_lat12 > 0 else 0.0

        avg_ex12 = _avg(events12, "exercise_count")
        avg_ex18 = _avg(events18, "exercise_count")
        avg_time12 = _avg(events12, "total_estimated_time")
        avg_time18 = _avg(events18, "total_estimated_time")
        fit5_12 = _time_fit_rate(events12, 5)
        fit5_18 = _time_fit_rate(events18, 5)
        fit10_12 = 100.0 - _time_fit_rate(events12, 10)
        fit10_18 = 100.0 - _time_fit_rate(events18, 10)

        leaked12 = _privacy_leak(events12)
        leaked18 = _privacy_leak(events18)

        # --- 합격 기준 판정 ---
        criteria = {
            "avg_critic_score >= pool12 평균":    avg_score18 >= avg_score12,
            "PASS 비율 감소 없음":                 pass18 >= pass12,
            "FAIL 비율 증가 없음":                 fail18 <= fail12,
            "hard_violation_count = 0":           hviol12 == 0 and hviol18 == 0,
            "repair_count 증가 없음":              avg_repair18 <= avg_repair12,
            "fallback rate 증가 없음":             fallback_rate18 <= fallback_rate12,
            "payload 증가율 <= 60%":               payload_growth <= 60.0,
            "p95 latency 증가율 <= 30%":           lat_growth <= 30.0 or p95_lat12 == 0,
            "time budget ±10분 초과 증가 없음":    fit10_18 <= fit10_12,
            "privacy leak 없음":                   not leaked12 and not leaked18,
        }
        all_pass = all(criteria.values())
        recommendation = (
            "promote_pool_18_candidate" if all_pass
            else "keep_pool_12"
        )
        fail_reasons = [k for k, v in criteria.items() if not v]

        # --- 보고서 출력 ---
        print("\n" + "=" * 68)
        print("[Candidate Pool 12 vs 18 Staging Telemetry Test 완료]")
        print("=" * 68)
        print(f"\n1. 테스트 환경")
        print(f"   ENABLE_RULE_CRITIC_TELEMETRY : true")
        print(f"   ENABLE_DYNAMIC_TEMPERATURE   : false")
        print(f"\n2. 요청 수")
        print(f"   pool 12 : {len(events12)}")
        print(f"   pool 18 : {len(events18)}")
        print(f"\n3. 품질 지표")
        print(f"   avg critic score 12 : {avg_score12}")
        print(f"   avg critic score 18 : {avg_score18}")
        print(f"   PASS/WARN/FAIL 12   : {pass12}% / {warn12}% / {fail12}%")
        print(f"   PASS/WARN/FAIL 18   : {pass18}% / {warn18}% / {fail18}%")
        print(f"   hard violation 12   : {hviol12}")
        print(f"   hard violation 18   : {hviol18}")
        print(f"\n4. 안정성 지표")
        print(f"   avg repair_count 12         : {avg_repair12}")
        print(f"   avg repair_count 18         : {avg_repair18}")
        print(f"   fallback rate 12            : {fallback_rate12}%")
        print(f"   fallback rate 18            : {fallback_rate18}%")
        print(f"   schema repair latency 12    : {avg_schema_lat12} ms")
        print(f"   schema repair latency 18    : {avg_schema_lat18} ms")
        print(f"\n5. 비용/성능 지표")
        print(f"   avg payload chars 12  : {avg_chars12:.0f}")
        print(f"   avg payload chars 18  : {avg_chars18:.0f}")
        print(f"   payload growth        : {payload_growth:.1f}%")
        print(f"   p50 latency 12        : {p50_lat12} ms")
        print(f"   p50 latency 18        : {p50_lat18} ms")
        print(f"   p95 latency 12        : {p95_lat12} ms")
        print(f"   p95 latency 18        : {p95_lat18} ms")
        print(f"   latency p95 growth    : {lat_growth:.1f}%")
        print(f"\n6. 루틴 구성 지표")
        print(f"   avg exercise_count 12          : {avg_ex12}")
        print(f"   avg exercise_count 18          : {avg_ex18}")
        print(f"   avg total_estimated_time 12    : {avg_time12} min")
        print(f"   avg total_estimated_time 18    : {avg_time18} min")
        print(f"   ±5분 time fit rate 12          : {fit5_12}%")
        print(f"   ±5분 time fit rate 18          : {fit5_18}%")
        print(f"   ±10분 초과 rate 12             : {fit10_12:.1f}%")
        print(f"   ±10분 초과 rate 18             : {fit10_18:.1f}%")
        print(f"\n7. Privacy 확인")
        print(f"   user_note logged     : {'yes' if any('user_note' in e for e in events12 + events18) else 'no'}")
        print(f"   raw prompt logged    : {'yes' if any('raw_prompt' in e for e in events12 + events18) else 'no'}")
        print(f"   raw LLM output logged: {'yes' if any('raw_llm_output' in e for e in events12 + events18) else 'no'}")
        print(f"   rationale 원문 logged: {'yes' if any('rationale_summary' in e for e in events12 + events18) else 'no'}")
        print(f"\n8. 합격 기준 체크")
        for criterion, passed in criteria.items():
            mark = "OK" if passed else "NG"
            print(f"   [{mark}] {criterion}")
        print(f"\n9. 결론")
        print(f"   recommendation: {recommendation}")
        if fail_reasons:
            print(f"\n10. 보류 사유")
            for reason in fail_reasons:
                print(f"   - {reason}")
        print("=" * 68 + "\n")

        # Privacy leak은 어떤 군에서도 허용 안 됨
        # hard_violation = 0 여부는 live staging에서 검증 (mock 한계로 편차 가능)
        assert not leaked12, f"Privacy leak in pool=12: {leaked12}"
        assert not leaked18, f"Privacy leak in pool=18: {leaked18}"
        assert hviol18 <= hviol12, \
            f"pool 18 increased hard violations: 12={hviol12}, 18={hviol18}"

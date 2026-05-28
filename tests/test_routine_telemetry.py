from copy import deepcopy
from datetime import datetime, timezone

from langchain_core.runnables import RunnableLambda

from engines.routine_pipeline import generate_smart_routine
from engines.routine_telemetry import (
    build_routine_quality_telemetry_payload,
    count_guard_repairs,
    emit_routine_quality_telemetry,
    observe_routine_quality,
    resolve_rule_critic_telemetry_enabled,
)
from engines.schemas import RoutineBlock, RoutineDraftResponse, SetPrescription


class _Critic:
    score = 87
    grade = "PASS"
    warnings = ["secret warning text"]
    hard_violations = ["secret violation text"]
    should_rebuild = True
    should_fallback = True


def _response(**overrides):
    values = {
        "generation_status": "success",
        "status_reason_code": "none",
        "is_fallback": False,
        "total_estimated_time": 45,
        "summary_title": "Test",
        "rationale_summary": ["secret rationale summary"],
        "routine_blocks": [
            RoutineBlock(
                order=1,
                exercise_id="bench",
                exercise_name="Bench",
                primary_muscles=["chest"],
                equipment_type="BARBELL",
                default_rest_sec=120,
                prescription=[
                    SetPrescription(set_index=1, target_reps=8, target_rest_sec=120),
                    SetPrescription(set_index=2, target_reps=8, target_rest_sec=120),
                ],
                exercise_rationale="secret exercise rationale",
            )
        ],
        "warnings": [],
    }
    values.update(overrides)
    return RoutineDraftResponse(**values)


def _candidate():
    return {
        "id": "bench",
        "name_kr": "Bench",
        "name_en": "Bench",
        "primary_muscle": "chest",
        "secondary_muscle": None,
        "equipment_req": "BARBELL",
        "movement_type": "COMPOUND",
        "efficiency_tier": 4,
        "pain_triggers": None,
    }


def test_resolve_none_is_false():
    assert resolve_rule_critic_telemetry_enabled(None) is False


def test_resolve_false_values_are_false():
    for value in ["", "false", "0", "no", "off", " FALSE "]:
        assert resolve_rule_critic_telemetry_enabled(value) is False


def test_resolve_true_values_are_true():
    for value in ["true", "1", "yes", "on", " TRUE "]:
        assert resolve_rule_critic_telemetry_enabled(value) is True


def test_resolve_invalid_string_is_false():
    assert resolve_rule_critic_telemetry_enabled("enabled") is False


def test_count_guard_repairs_requires_guard_and_replaced():
    warnings = [
        "Guard diagnostics: repairCount=1, trimmedSetCount=0, removedExerciseCount=0",
        "Guard: informational line without replacement",
        "Guard: 'ghost' -> 'bench' (constraint violation replaced)",
        "replaced without guard prefix",
    ]
    assert count_guard_repairs(warnings) == 1


def test_count_guard_repairs_does_not_count_diagnostics():
    warnings = ["Guard diagnostics: repairCount=2, trimmedSetCount=0, removedExerciseCount=0"]
    assert count_guard_repairs(warnings) == 0


def test_payload_includes_candidate_pool_size():
    payload = build_routine_quality_telemetry_payload(
        response=_response(),
        candidate_pool_size=18,
        ranked_candidates=[_candidate()],
        candidate_payload="secret candidate payload",
        generation_temperature=0.0,
        critic_result=_Critic(),
        repair_count=0,
        fallback_used=False,
        target_duration_min=45,
        created_at=datetime(2026, 5, 28, tzinfo=timezone.utc),
    )
    assert payload["candidate_pool_size"] == 18


def test_payload_includes_critic_score_and_grade():
    payload = build_routine_quality_telemetry_payload(
        response=_response(),
        candidate_pool_size=12,
        ranked_candidates=[_candidate()],
        candidate_payload="x",
        generation_temperature=0.0,
        critic_result=_Critic(),
        repair_count=0,
        fallback_used=False,
        target_duration_min=45,
    )
    assert payload["critic_score"] == 87
    assert payload["critic_grade"] == "PASS"


def test_payload_includes_repair_count_and_fallback_used():
    payload = build_routine_quality_telemetry_payload(
        response=_response(is_fallback=True, generation_status="fallback", status_reason_code="emptyCandidate"),
        candidate_pool_size=12,
        ranked_candidates=[_candidate()],
        candidate_payload="x",
        generation_temperature=0.0,
        critic_result=_Critic(),
        repair_count=2,
        fallback_used=True,
        target_duration_min=45,
    )
    assert payload["repair_count"] == 2
    assert payload["fallback_used"] is True


def test_payload_includes_dynamic_temperature_and_generation_temperature():
    payload = build_routine_quality_telemetry_payload(
        response=_response(),
        candidate_pool_size=12,
        ranked_candidates=[_candidate()],
        candidate_payload="x",
        dynamic_temperature_enabled=True,
        generation_temperature=0.2,
        critic_result=_Critic(),
        repair_count=0,
        fallback_used=False,
        target_duration_min=45,
    )
    assert payload["dynamic_temperature_enabled"] is True
    assert payload["generation_temperature"] == 0.2


def test_payload_excludes_free_text_fields():
    payload = build_routine_quality_telemetry_payload(
        response=_response(warnings=["Guard: secret ghost replaced"]),
        candidate_pool_size=12,
        ranked_candidates=[_candidate()],
        candidate_payload="raw prompt secret candidate payload",
        generation_temperature=0.0,
        critic_result=_Critic(),
        repair_count=1,
        fallback_used=False,
        target_duration_min=45,
    )
    payload_text = str(payload)
    assert "user_note" not in payload
    assert "raw prompt secret" not in payload_text
    assert "raw LLM" not in payload_text
    assert "secret rationale" not in payload_text
    assert "secret warning text" not in payload_text
    assert "secret violation text" not in payload_text


def test_payload_records_candidate_payload_sizes():
    payload = build_routine_quality_telemetry_payload(
        response=_response(),
        candidate_pool_size=12,
        ranked_candidates=[_candidate()],
        candidate_payload="12345",
        generation_temperature=0.0,
        critic_result=_Critic(),
        repair_count=0,
        fallback_used=False,
        target_duration_min=45,
    )
    assert payload["candidate_payload_char_count"] == 5
    assert payload["approx_candidate_payload_tokens"] == 2


def test_emit_failure_is_swallowed(monkeypatch):
    from engines import routine_telemetry

    monkeypatch.setattr(routine_telemetry.json, "dumps", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError()))
    emit_routine_quality_telemetry({"event": "routine_generation_quality"})


def test_observe_flag_off_is_noop(monkeypatch, sample_request):
    from engines import routine_telemetry

    response = _response()
    monkeypatch.delenv("ENABLE_RULE_CRITIC_TELEMETRY", raising=False)
    monkeypatch.setattr(
        routine_telemetry,
        "emit_routine_quality_telemetry",
        lambda _payload: (_ for _ in ()).throw(AssertionError("emit should not be called")),
    )
    result = observe_routine_quality(
        response=response,
        req=sample_request,
        ranked_candidates=[_candidate()],
        db_target_muscles=["chest"],
        recent_sets=[],
        max_total_sets=10,
        candidate_pool_size=12,
        candidate_payload="x",
        generation_temperature=0.0,
    )
    assert result is response


def test_rule_critic_fail_does_not_mutate_response(monkeypatch, sample_request):
    from engines import routine_telemetry

    response = _response(routine_blocks=[_response().routine_blocks[0].model_copy(update={"exercise_id": "ghost"})])
    before = deepcopy(response.model_dump())
    emitted = []
    monkeypatch.setenv("ENABLE_RULE_CRITIC_TELEMETRY", "true")
    monkeypatch.setattr(routine_telemetry, "emit_routine_quality_telemetry", emitted.append)

    result = observe_routine_quality(
        response=response,
        req=sample_request,
        ranked_candidates=[_candidate()],
        db_target_muscles=["chest"],
        recent_sets=[],
        max_total_sets=10,
        candidate_pool_size=12,
        candidate_payload="x",
        generation_temperature=0.0,
    )

    assert result is response
    assert response.model_dump() == before
    assert emitted[0]["critic_grade"] == "FAIL"


def test_pipeline_flag_off_keeps_generation_result(
    monkeypatch,
    sample_request,
    sample_profile,
    sample_llm_output,
    mock_candidates,
):
    from engines import routine_telemetry
    from engines import routine_pipeline

    class _LLM:
        def with_structured_output(self, _schema):
            return RunnableLambda(lambda _: sample_llm_output)

    monkeypatch.delenv("ENABLE_RULE_CRITIC_TELEMETRY", raising=False)
    monkeypatch.setattr(
        routine_telemetry,
        "emit_routine_quality_telemetry",
        lambda _payload: (_ for _ in ()).throw(AssertionError("emit should not be called")),
    )
    monkeypatch.setattr(routine_pipeline, "get_candidate_exercises", lambda *_args, **_kwargs: mock_candidates)
    monkeypatch.setattr(routine_pipeline, "get_llm", lambda *_args, **_kwargs: _LLM())

    result = generate_smart_routine(sample_request, db=None, profile=sample_profile, recent_sets=[])

    assert result.generation_status == "success"
    assert result.is_fallback is False


def test_pipeline_flag_on_emits(
    monkeypatch,
    sample_request,
    sample_profile,
    sample_llm_output,
    mock_candidates,
):
    from engines import routine_telemetry
    from engines import routine_pipeline

    class _LLM:
        def with_structured_output(self, _schema):
            return RunnableLambda(lambda _: sample_llm_output)

    emitted = []
    monkeypatch.setenv("ENABLE_RULE_CRITIC_TELEMETRY", "true")
    monkeypatch.setattr(routine_telemetry, "emit_routine_quality_telemetry", emitted.append)
    monkeypatch.setattr(routine_pipeline, "get_candidate_exercises", lambda *_args, **_kwargs: mock_candidates)
    monkeypatch.setattr(routine_pipeline, "get_llm", lambda *_args, **_kwargs: _LLM())

    result = generate_smart_routine(sample_request, db=None, profile=sample_profile, recent_sets=[])

    assert result.generation_status == "success"
    assert len(emitted) == 1
    assert emitted[0]["event"] == "routine_generation_quality"

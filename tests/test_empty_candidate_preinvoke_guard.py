import pytest

from langchain_core.runnables import RunnableLambda

from engines.routine_pipeline import generate_smart_routine


def test_empty_ranked_candidates_skip_llm_and_preserve_failed_contract(
    monkeypatch,
    sample_request,
    sample_profile,
):
    from engines import routine_pipeline, routine_telemetry

    llm_constructions = 0
    runnable_invocations = 0
    emitted = []

    class _LLM:
        def with_structured_output(self, _schema):
            def invoke(_kwargs):
                nonlocal runnable_invocations
                runnable_invocations += 1
                raise AssertionError("empty candidate pool must not invoke the LLM")

            return RunnableLambda(invoke)

    def get_llm_spy(*_args, **_kwargs):
        nonlocal llm_constructions
        llm_constructions += 1
        return _LLM()

    monkeypatch.setenv("ENABLE_RULE_CRITIC_TELEMETRY", "true")
    monkeypatch.setattr(routine_telemetry, "emit_routine_quality_telemetry", emitted.append)
    monkeypatch.setattr(routine_pipeline, "get_candidate_exercises", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(routine_pipeline, "get_adjunct_candidate_exercises", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(routine_pipeline, "get_llm", get_llm_spy)

    result = generate_smart_routine(sample_request, db=None, profile=sample_profile, recent_sets=[])

    assert runnable_invocations == 0
    assert llm_constructions == 0
    assert result.generation_status == "failed"
    assert result.status_reason_code == "emptyCandidate"
    assert result.is_fallback is False
    assert result.routine_blocks == []
    assert result.total_estimated_time == 0
    assert result.warnings
    assert len(emitted) == 1
    assert emitted[0]["candidate_count"] == 0
    assert emitted[0]["exercise_count"] == 0
    assert emitted[0]["total_estimated_time"] == 0
    assert emitted[0]["generation_latency_ms"] is None


def test_nonempty_ranked_candidates_still_invoke_llm(
    monkeypatch,
    sample_request,
    sample_profile,
    sample_llm_output,
    mock_candidates,
):
    from engines import routine_pipeline

    runnable_invocations = 0

    class _LLM:
        def with_structured_output(self, _schema):
            def invoke(_kwargs):
                nonlocal runnable_invocations
                runnable_invocations += 1
                return sample_llm_output

            return RunnableLambda(invoke)

    monkeypatch.setattr(routine_pipeline, "get_candidate_exercises", lambda *_args, **_kwargs: mock_candidates)
    monkeypatch.setattr(routine_pipeline, "get_adjunct_candidate_exercises", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(routine_pipeline, "get_llm", lambda *_args, **_kwargs: _LLM())

    result = generate_smart_routine(sample_request, db=None, profile=sample_profile, recent_sets=[])

    assert runnable_invocations == 1
    assert result.generation_status == "success"
    assert result.routine_blocks


@pytest.mark.parametrize('split,targets', [
    ('core', ['abs']), ('arm', ['biceps', 'triceps']), ('push', []),
])
def test_empty_pool_preserves_existing_request_specific_failure(
    monkeypatch, sample_request, sample_profile, split, targets,
):
    from engines import routine_pipeline as pipeline

    request = sample_request.model_copy(update={
        'target_split_label': split, 'target_muscles': targets,
    })
    monkeypatch.setattr(pipeline, 'get_candidate_exercises', lambda *a, **kw: [])
    monkeypatch.setattr(pipeline, 'get_adjunct_candidate_exercises', lambda *a, **kw: [])
    monkeypatch.setattr(pipeline, 'get_llm', lambda *a, **kw: pytest.fail('LLM must not be constructed'))
    observed = []
    original = pipeline._fallback

    def capture_existing_failure(*args, **kwargs):
        draft = original(*args, **kwargs)
        observed.append(draft.model_copy(deep=True))
        return draft

    monkeypatch.setattr(pipeline, '_fallback', capture_existing_failure)
    result = pipeline.generate_smart_routine(
        request, db=None, profile=sample_profile, recent_sets=[],
    )
    assert len(observed) == 1
    assert result.summary_title == observed[0].summary_title
    assert set(observed[0].warnings) <= set(result.warnings)
    assert result.routine_blocks == []
    assert result.status_reason_code == 'emptyCandidate'
    assert result.total_estimated_time == 0

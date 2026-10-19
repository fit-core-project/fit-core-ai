import inspect

from langchain_core.runnables import RunnableLambda

from engines.candidate_pool_policy import (
    DEFAULT_CANDIDATE_POOL_SIZE,
    ENV_CANDIDATE_POOL_SIZE,
    resolve_candidate_pool_size,
)
from engines.candidate_ranker import score_candidate_exercises
from engines.routine_pipeline import generate_smart_routine


def test_missing_value_uses_default():
    assert resolve_candidate_pool_size(None) == 12


def test_twelve_is_allowed():
    assert resolve_candidate_pool_size("12") == 12


def test_eighteen_is_allowed():
    assert resolve_candidate_pool_size("18") == 18


def test_invalid_string_uses_default():
    assert resolve_candidate_pool_size("large") == 12


def test_empty_string_uses_default():
    assert resolve_candidate_pool_size("") == 12


def test_negative_number_uses_default():
    assert resolve_candidate_pool_size("-1") == 12


def test_zero_uses_default():
    assert resolve_candidate_pool_size("0") == 12


def test_over_max_uses_default():
    assert resolve_candidate_pool_size("999") == 12


def test_whitespace_is_trimmed():
    assert resolve_candidate_pool_size(" 18 ") == 18


def test_candidate_ranker_default_top_n_remains_twelve():
    signature = inspect.signature(score_candidate_exercises)
    assert signature.parameters["top_n"].default == DEFAULT_CANDIDATE_POOL_SIZE


def test_default_gym_excludes_unrequested_specialty_style_candidates():
    candidates = [
        {
            "id": "cable_crunch",
            "name_kr": "케이블 크런치",
            "primary_muscle": "abs",
            "movement_type": "ISOLATION",
            "equipment_req": "CABLE",
            "training_style_tags": "bodybuilding,gym",
        },
        {
            "id": "child_pose",
            "name_kr": "차일드 포즈 스트레치",
            "primary_muscle": "lower-back",
            "movement_type": "STATIC",
            "equipment_req": "BODYWEIGHT",
            "training_style_tags": "mobility,yoga_pilates",
        },
    ]

    ranked = score_candidate_exercises(
        candidates,
        ["abs", "lower-back"],
        training_style="gym",
        adjunct_options=[],
    )

    assert [candidate["id"] for candidate in ranked] == ["cable_crunch"]


def test_default_gym_allows_specialty_style_when_matching_adjunct_is_requested():
    candidates = [
        {
            "id": "child_pose",
            "name_kr": "차일드 포즈 스트레치",
            "primary_muscle": "lower-back",
            "movement_type": "STATIC",
            "equipment_req": "BODYWEIGHT",
            "training_style_tags": "mobility,yoga_pilates",
        },
    ]

    ranked = score_candidate_exercises(
        candidates,
        ["lower-back"],
        training_style="gym",
        adjunct_options=["cooldown_stretch"],
    )

    assert [candidate["id"] for candidate in ranked] == ["child_pose"]


def test_routine_pipeline_passes_configured_pool_size(
    monkeypatch,
    sample_request,
    sample_profile,
    sample_llm_output,
    mock_candidates,
):
    from engines import routine_pipeline

    captured = {}
    original_score_candidate_exercises = routine_pipeline.score_candidate_exercises

    def spy_score_candidate_exercises(*args, **kwargs):
        captured["top_n"] = kwargs.get("top_n")
        return original_score_candidate_exercises(*args, **kwargs)

    class _LLM:
        def with_structured_output(self, _schema):
            return RunnableLambda(lambda _: sample_llm_output)

    monkeypatch.setenv(ENV_CANDIDATE_POOL_SIZE, "18")
    monkeypatch.setattr(routine_pipeline, "score_candidate_exercises", spy_score_candidate_exercises)
    monkeypatch.setattr(routine_pipeline, "get_candidate_exercises", lambda *_args, **_kwargs: mock_candidates)
    monkeypatch.setattr(routine_pipeline, "get_llm", lambda *_args, **_kwargs: _LLM())

    result = generate_smart_routine(sample_request, db=None, profile=sample_profile, recent_sets=[])

    assert captured["top_n"] == 18
    assert result.generation_status == "success"

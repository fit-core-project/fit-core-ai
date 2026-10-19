"""Check the real choice -> pipeline -> ranker boundary, not a generator stub."""
import pytest

from engines import db_queries, routine_generation_choice as choice, routine_pipeline as pipeline
from tests.test_db_health_choice_preflight import _health_db, _request


@pytest.mark.parametrize('selection_code', ['normalVolume', 'reducedVolume'])
def test_persisted_snapshot_reaches_real_ranker_once(tmp_path, monkeypatch, selection_code):
    path = tmp_path / 'health.sqlite'
    _health_db(path)
    monkeypatch.setattr(db_queries, 'SQLITE_DB_PATH', path)
    request = _request()
    original = request.model_dump(mode='json')
    preflight = choice.preflight_routine_generation(request, resolve_persisted_context=True)
    observed = []

    class ReachedRanker(Exception):
        pass

    def capture_ranker(*args, **kwargs):
        observed.append(kwargs)
        raise ReachedRanker

    def fail_reload(*args, **kwargs):
        raise AssertionError('pipeline must reuse the bound snapshot')

    monkeypatch.setattr(pipeline, 'get_user_constraint_profile_from_sqlite', fail_reload)
    monkeypatch.setattr(pipeline, 'get_candidate_exercises', lambda *a, **kw: [])
    monkeypatch.setattr(pipeline, 'get_adjunct_candidate_exercises', lambda *a, **kw: [])
    monkeypatch.setattr(pipeline, '_get_feedback_adjustments_with_status', lambda *a, **kw: ({}, False, None))
    monkeypatch.setattr(pipeline, 'score_candidate_exercises', capture_ranker)
    monkeypatch.setattr(pipeline, 'get_llm', lambda *a, **kw: pytest.fail('ranker boundary must stop before model construction'))
    with pytest.raises(ReachedRanker):
        choice.generate_routine_with_choice(
            request, object(), selection=choice.make_routine_generation_selection(preflight, selection_code))

    assert len(observed) == 1
    assert observed[0]['condition_policies'] == preflight.constraints.condition_policies
    assert len(observed[0]['condition_policies']) == 1
    assert observed[0]['mobility_limits'] == ['limited_shoulder_flexion']
    assert request.model_dump(mode='json') == original

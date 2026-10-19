from copy import deepcopy

import pytest

from engines import candidate_ranker as ranker
from engines.routine_blueprint import build_blueprint_candidate_bundles, build_routine_blueprint
from engines.routine_selection import adapt_slot_selection_output
from engines.schemas import LLMRoutineSelectionOutput


def candidate(cid, name, primary, score=100, equipment="MACHINE"):
    return dict(id=cid, name_kr=name, primary_muscle=primary, score=score,
                name_en=name, equipment_req=equipment, main_routine_default_allowed=1,
                movement_type="ISOLATION" if primary == "biceps" else "COMPOUND",
                efficiency_tier=3, difficulty_tier=2, technical_difficulty="low",
                failure_penalty="low", injury_caution_level="low", pain_triggers="",
                exercise_familiarity_tier="common", mainstream_gym_score=90)


def pool():
    return [candidate("curl", "머신 프리처 컬", "biceps", 220),
            candidate("curl_alt", "덤벨 프리처 컬", "biceps", 219, "DUMBBELL"),
            candidate("wrist", "머신 리스트 컬", "forearm", 215),
            candidate("row", "덤벨 로우", "upper-back", 210, "DUMBBELL"),
            candidate("shrug", "머신 슈러그", "trapezius", 208),
            candidate("rear", "머신 리버스 펙덱 플라이", "back-deltoids", 207),
            candidate("pullup", "어시스트 풀업", "upper-back", 200),
            candidate("lat", "머신 랫풀다운", "upper-back", 190)]


def test_back_choice_reservation_replaces_surplus_not_primary_coverage():
    source = pool()
    before = deepcopy(source)
    selected = source[:-1]
    result = ranker._ensure_pull_back_choices(selected=selected, sorted_candidates=source, top_n=7)
    assert {c['id'] for c in result} == {c['id'] for c in source} - {'curl_alt'}
    assert {c['primary_muscle'] for c in result} == {c['primary_muscle'] for c in selected}
    assert source == before
    assert all(c in source for c in result)
    assert result == ranker._ensure_pull_back_choices(selected=result, sorted_candidates=source, top_n=7)


@pytest.mark.parametrize('size', range(1, 19))
def test_pool_bound_uniqueness_no_fabrication(size):
    source = pool()
    result = ranker._ensure_pull_back_choices(selected=source[:size], sorted_candidates=source, top_n=size)
    assert len(result) <= size
    assert len({c['id'] for c in result}) == len(result)
    assert all(c in source for c in result)


def test_no_reservation_reintroduces_filtered_or_unadmitted_candidate():
    source = pool()
    source[-1]['main_routine_default_allowed'] = 0
    result = ranker._ensure_pull_back_choices(selected=source[:-1], sorted_candidates=source, top_n=7)
    assert 'lat' not in {c['id'] for c in result}
    result = ranker._ensure_pull_back_choices(selected=source[:-1], sorted_candidates=source[:-1], top_n=7)
    assert 'lat' not in {c['id'] for c in result}


def test_no_last_primary_displaced_when_pool_is_small():
    source = pool()
    selected = [c for c in source if c['id'] in {'curl', 'wrist', 'shrug', 'rear'}]
    result = ranker._ensure_pull_back_choices(selected=selected, sorted_candidates=source, top_n=4)
    assert {c['primary_muscle'] for c in selected} <= {c['primary_muscle'] for c in result}


DEFAULTS = dict(target_split_label='pull', experience_level='intermediate', goal='hypertrophy',
                readiness_level='normal', time_available_min=50, session_structure='standard',
                training_style='gym')


@pytest.mark.parametrize('delta,expected', [
    ({}, True), ({'experience_level': 'advanced'}, True), ({'readiness_level': 'high'}, True),
    ({'experience_level': 'beginner'}, False), ({'target_split_label': 'full_body'}, False),
    ({'goal': 'strength'}, False), ({'time_available_min': 49}, False),
    ({'readiness_level': 'low'}, False), ({'session_structure': 'density'}, False),
    ({'adjunct_options': ['cooldownRecovery']}, False),
    ({'specialization_focus': 'latEmphasis'}, False),
    ({'doms_db': {'upper-back': 1}}, False),
    ({'mobility_limits': ['limited_shoulder_flexion']}, False),
    ({'shoulder_condition_status': 'activeUncleared'}, False),
])
def test_scope_gate(monkeypatch, delta, expected):
    calls = []
    original = ranker._ensure_pull_back_choices
    def spy(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(ranker, '_ensure_pull_back_choices', spy)
    ranker.score_candidate_exercises(pool(), ['upper-back', 'trapezius', 'biceps', 'forearm', 'back-deltoids'],
                                    **{**DEFAULTS, **delta})
    assert bool(calls) == expected


@pytest.mark.parametrize('kwargs', [
    {'unpreferred_exercise_ids': ['lat']},
    {'blocked_equipment': ['MACHINE']},
    {'available_equipment': ['DUMBBELL']},
])
def test_full_ranker_reservation_cannot_restore_excluded_lat(kwargs):
    result = ranker.score_candidate_exercises(pool(), ['upper-back', 'biceps'], **DEFAULTS, **kwargs)
    assert 'lat' not in {c['id'] for c in result}


def test_vertical_alternatives_are_choices_not_two_extra_exercises():
    candidates = ranker._ensure_pull_back_choices(selected=pool()[:-1], sorted_candidates=pool(), top_n=7)
    bp = build_routine_blueprint(target_split_label='pull', target_muscles=[], goal='hypertrophy',
                                experience_level='intermediate', readiness_level='normal', time_available_min=50)
    bundles = build_blueprint_candidate_bundles(bp, candidates)
    vertical = next(b for b in bundles if b['slotId'] == 'pull_vertical_main')
    assert {c['exerciseId'] for c in vertical['candidates']} == {'pullup', 'lat'}
    selection = LLMRoutineSelectionOutput.model_validate({'selections': [
        {'slotId': 'pull_vertical_main', 'exerciseId': 'lat', 'exerciseRationale': '테스트'},
    ]})
    result = adapt_slot_selection_output(selection, blueprint=bp, slot_candidate_bundles=bundles,
                                         ranked_candidates=candidates)
    chosen = {str(c.exercise_id) for c in result.exercises}
    assert 'lat' in chosen
    assert 'pullup' not in chosen
    assert 'row' in chosen and 'curl' in chosen

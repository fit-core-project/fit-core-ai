from copy import deepcopy

import pytest

from engines.candidate_ranker import score_candidate_exercises


def candidate(**updates):
    value = {
        'id': 'copy-check', 'name_kr': '머신 체스트 프레스',
        'name_en': 'Machine Chest Press', 'primary_muscle': 'chest',
        'secondary_muscle': 'triceps', 'equipment_req': 'MACHINE',
        'difficulty_tier': 3, 'efficiency_tier': 3,
        'movement_type': 'COMPOUND', 'pain_triggers': '',
        'lumbar_load': 'low', 'axial_load': 'low',
        'injury_caution_level': 'medium', 'technical_difficulty': 'low',
        'failure_penalty': 'low',
    }
    return {**value, **updates}


@pytest.mark.parametrize(('fields', 'options', 'rule', 'delta'), [
    ({}, {'readiness_level': 'low'}, 'readiness_injury_caution_penalty', -8),
    ({}, {'condition_policies': [{'condition_code': 'reviewed_context'}]},
     'condition_context_injury_caution_penalty', -5),
    ({'injury_caution_level': 'high'}, {}, 'general_injury_caution_penalty', -32),
    ({'technical_difficulty': 'high'}, {'readiness_level': 'low'},
     'readiness_technical_difficulty_penalty', -8),
])
def test_penalty_copy_identifies_priority_not_risk_or_dose_change(fields, options, rule, delta):
    source = candidate(**fields)
    before = deepcopy(source)
    rows = score_candidate_exercises([source], ['chest'], **options)
    assert len(rows) == 1
    entries = [entry for entry in rows[0]['score_penalties'] if entry['rule_code'] == rule]
    assert len(entries) == 1
    assert entries[0]['score'] == delta
    assert '추천 우선순위' in entries[0]['reason']
    assert entries[0]['reason'] in rows[0]['score_display_reasons']
    assert source == before
    assert rows[0]['injury_caution_level'] == fields.get('injury_caution_level', 'medium')

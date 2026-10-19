"""P1 exclusion is not the same decision as main-lift role eligibility."""
import copy

import pytest

from engines.experience_policy import automatic_recommendation_eligibility, main_role_eligibility
from engines.fallback import generate_fallback_routine
from engines.routine_blueprint import build_blueprint_candidate_bundles, build_routine_blueprint
from engines.schemas import RoutineRequest, UserProfileContext, LLMRoutineSelectionOutput
from engines.routine_selection import adapt_slot_selection_output, _replacement_matches_destination
from tests.test_routine_selection import _push_context
from tests.test_barbell_admission_boundary_v1 import catalog_candidates, rank


@pytest.fixture(scope='module')
def squat():
    return catalog_candidates()['V24ROW-003251']


def blueprint(level):
    return build_routine_blueprint(target_split_label='full_body', target_muscles=[],
        goal='hypertrophy', experience_level=level, readiness_level='normal', time_available_min=90)


@pytest.mark.parametrize('flag', [False, 0, '0', 'false', 'no', 'blocked'])
@pytest.mark.parametrize('level', ['beginner', 'intermediate', 'advanced'])
def test_catalog_denial_is_independent_of_role_and_experience(flag, level):
    assert not automatic_recommendation_eligibility({'main_routine_default_allowed': flag}, level).eligible


@pytest.mark.parametrize('flag', [False, 0, '0', 'false', 'no', 'blocked'])
def test_beginner_denial_is_not_an_intermediate_or_advanced_ban(flag):
    row = {'main_routine_default_allowed': 1, 'beginner_default_allowed': flag,
           'technical_difficulty': 'high', 'failure_penalty': 'high', 'injury_caution_level': 'high'}
    assert not automatic_recommendation_eligibility(row, 'beginner').eligible
    for level in ('intermediate', 'advanced'):
        assert automatic_recommendation_eligibility(row, level).eligible


@pytest.mark.parametrize('level', ['beginner', 'intermediate', 'advanced'])
def test_missing_extension_metadata_preserves_legacy_boundary(level):
    assert automatic_recommendation_eligibility({}, level).eligible


def test_real_beginner_squat_removed_before_ranking_and_flex(squat):
    assert not rank(squat, experience_level='beginner', time_available_min=90)
    bundles = build_blueprint_candidate_bundles(blueprint('beginner'), [squat])
    assert not any(b['candidates'] for b in bundles)


@pytest.mark.parametrize('level', ['intermediate', 'advanced'])
def test_real_high_squat_remains_flex_but_not_main(squat, level):
    assert rank(squat, experience_level=level, time_available_min=90)
    assert not main_role_eligibility(squat, level, 'hypertrophy').eligible
    bundles = build_blueprint_candidate_bundles(blueprint(level), [squat])
    assert any(c['exerciseId'] == squat['id'] for b in bundles if b['slotId'] == 'flex' for c in b['candidates'])


@pytest.mark.parametrize('level', ['beginner', 'intermediate', 'advanced'])
def test_hold_cannot_escape_via_flex_or_preference(squat, level):
    row = dict(squat, main_routine_default_allowed=0, beginner_default_allowed=1,
               core_main_role_reviewed=True, mainstream_gym_score=100)
    assert not rank(row, experience_level=level, preferred_exercise_ids=[row['id']])
    assert not any(b['candidates'] for b in build_blueprint_candidate_bundles(blueprint(level), [row]))


@pytest.mark.parametrize('denial', ['beginner', 'catalog'])
@pytest.mark.parametrize('with_blueprint', [False, True])
def test_forced_fallback_cannot_reintroduce_explicit_denials(squat, denial, with_blueprint):
    row = copy.deepcopy(squat)
    if denial == 'catalog':
        row.update(main_routine_default_allowed=0, beginner_default_allowed=1)
    req = RoutineRequest(userId='p1-synthetic', targetSplitLabel='full_body', targetMuscles=[],
                         timeAvailableMin=90, experienceLevel='beginner', availableEquipment=['BARBELL'])
    result = generate_fallback_routine(req, [row], 20,
        routine_blueprint=blueprint('beginner') if with_blueprint else None)
    assert result.generation_status == 'failed'
    assert result.status_reason_code == 'emptyCandidate'
    assert result.routine_blocks == []


def test_no_mutation_of_candidate_metadata(squat):
    before = copy.deepcopy(squat)
    automatic_recommendation_eligibility(squat, 'beginner')
    assert squat == before


@pytest.mark.parametrize('alias', ['novice', 'newbie', 'entry', '초보', '초보자', '입문', ' Beginner '])
def test_beginner_profile_aliases_cannot_bypass_exclusion(squat, alias):
    assert not automatic_recommendation_eligibility(squat, alias).eligible
    req = RoutineRequest(userId='p1-profile-only', timeAvailableMin=60,
                         targetSplitLabel='legs', availableEquipment=['BARBELL'])
    result = generate_fallback_routine(req, [squat], 20,
        profile=UserProfileContext(experience_level=alias, goalType='hypertrophy',
                                   splitType='pushPullLegs'))
    assert result.routine_blocks == []


def _selection():
    return LLMRoutineSelectionOutput.model_validate({'summaryTitle': '선택 검증',
        'rationaleSummary': [], 'warnings': [], 'selections': [
            {'slotId': 'push_main', 'exerciseId': 'bench_press', 'exerciseRationale': '메인'}]})


def test_adapter_rejects_denied_id_even_with_stale_slot_bundle():
    bp, rows, stale_bundles = _push_context()
    rows[0]['beginner_default_allowed'] = 0
    output = adapt_slot_selection_output(_selection(), blueprint=bp,
        ranked_candidates=rows, slot_candidate_bundles=stale_bundles)
    assert 'bench_press' not in {e.exercise_id for e in output.exercises}


def test_mixed_fallback_retains_allowed_candidates_not_denied(squat):
    _, rows, _ = _push_context()
    req = RoutineRequest(userId='p1-mixed', experienceLevel='beginner',
        targetSplitLabel='push', timeAvailableMin=30,
        availableEquipment=['BARBELL', 'MACHINE'])
    result = generate_fallback_routine(req, [squat, *rows], 12)
    assert result.routine_blocks
    assert squat['id'] not in {b.exercise_id for b in result.routine_blocks}


@pytest.mark.parametrize('level', ['intermediate', 'advanced'])
def test_repair_cannot_promote_automatically_allowed_high_candidate_to_main(level):
    from dataclasses import replace
    bp, rows, bundles = _push_context()
    bp = replace(bp, experience_level=level)
    output = adapt_slot_selection_output(_selection(), blueprint=bp,
        ranked_candidates=rows, slot_candidate_bundles=bundles)
    index = next(i for i, e in enumerate(output.exercises) if e.exercise_id == 'bench_press')
    replacement = dict(rows[0], id='high_bench', main_routine_default_allowed=1,
        beginner_default_allowed=0, injury_caution_level='high', failure_penalty='high',
        exercise_familiarity_tier='common', mainstream_gym_score=100)
    assert automatic_recommendation_eligibility(replacement, level).eligible
    assert not _replacement_matches_destination(output, bp, index, replacement)

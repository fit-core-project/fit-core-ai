"""Exact catalog identity is required for the purpose-only P1 exception."""
import copy

import pytest

from engines.experience_policy import automatic_recommendation_eligibility
from engines.candidate_ranker import score_candidate_exercises
from engines.routine_blueprint import build_blueprint_candidate_bundles
from engines.routine_selection import adapt_slot_selection_output
from engines.schemas import LLMRoutineSelectionOutput
from engines.routine_blueprint import build_routine_blueprint
from engines.fallback import generate_fallback_routine
from engines.schemas import RoutineRequest
from tests.test_routine_selection import _push_context


def reviewed():
    return dict(id='V24ROW-023032', name_kr='네발기기 흉추 회전 스트레칭',
        primary_muscle='upper-back', secondary_muscle='', equipment_req='BODYWEIGHT',
        name_en='Thread the Needle - 동적 반복',
        movement_type='ISOLATION', efficiency_tier=3, pain_triggers='',
        main_routine_default_allowed=0, beginner_default_allowed=0,
        training_style_tags='mobility,yoga_pilates', adjunct_option_codes=['mobility_warmup'])


def test_reviewed_purpose_request_allows_only_the_catalog_denial_exception():
    row = reviewed()
    assert automatic_recommendation_eligibility(row, 'intermediate',
        adjunct_options=['mobility_warmup']).eligible
    assert not automatic_recommendation_eligibility(row, 'intermediate').eligible
    assert not automatic_recommendation_eligibility(row, 'intermediate',
        adjunct_options=['cooldown_stretch']).eligible
    assert not automatic_recommendation_eligibility(row, 'beginner',
        adjunct_options=['mobility_warmup']).eligible


@pytest.mark.parametrize('eid', ['band_shoulder_mobility', 'reviewed_dynamic_stretch', 'warrior_2', '30'])
def test_names_and_claimed_review_metadata_do_not_approve_an_id(eid):
    row = dict(reviewed(), id=eid, catalog_exercise_id='V24ROW-023032',
        adjunct_catalog_review_status='core-reviewed', adjunct_catalog_version='routine-adjunct-catalog-v1')
    assert not automatic_recommendation_eligibility(row, 'advanced',
        adjunct_options=['mobility_warmup']).eligible


def test_mismatching_catalog_identity_is_not_an_exception():
    row = dict(reviewed(), catalog_exercise_id='V24ROW-000379')
    assert not automatic_recommendation_eligibility(row, 'advanced',
        adjunct_options=['mobility_warmup']).eligible


def test_exact_reviewed_yoga_style_candidate_can_enter_pool_but_not_unknown_id():
    row = reviewed()
    kwargs = dict(experience_level='intermediate', training_style='yoga_pilates', top_n=5)
    assert score_candidate_exercises([row], ['upper-back'], **kwargs)
    assert not score_candidate_exercises([dict(row, id='warrior_2')], ['upper-back'], **kwargs)


def test_unrequested_reviewed_candidate_cannot_enter_pool():
    assert not score_candidate_exercises([reviewed()], ['upper-back'],
        experience_level='intermediate', training_style='gym', top_n=5)


def test_missing_or_wrong_review_catalog_fails_closed(monkeypatch, tmp_path, caplog):
    from engines import adjunct_admission as admission
    monkeypatch.setattr(admission, 'CATALOG_PATH', tmp_path/'missing.json')
    admission.reviewed_purpose_index.cache_clear()
    try:
        assert not automatic_recommendation_eligibility(reviewed(), 'advanced',
            adjunct_options=['mobility_warmup']).eligible
        assert 'reviewed adjunct admission catalog unavailable' in caplog.text
    finally:
        admission.reviewed_purpose_index.cache_clear()


def test_no_metadata_mutation():
    row = reviewed()
    before = copy.deepcopy(row)
    automatic_recommendation_eligibility(row, 'advanced', adjunct_options=['mobility_warmup'])
    assert row == before


def purpose_context():
    bp = build_routine_blueprint(target_split_label='push', target_muscles=[],
        goal='hypertrophy', experience_level='intermediate', readiness_level='normal',
        time_available_min=60, adjunct_options=['mobility_warmup'])
    _, rows, _ = _push_context()
    rows.append(dict(reviewed(), score=130))
    return bp, rows, build_blueprint_candidate_bundles(bp, rows, top_k_per_slot=3)


def test_reviewed_id_is_bundled_only_in_its_requested_purpose_slot():
    bp, rows, bundles = purpose_context()
    destinations = {b['slotId'] for b in bundles
        if any(c['exerciseId'] == reviewed()['id'] for c in b['candidates'])}
    assert destinations == {s.slot_id for s in bp.slots if s.adjunct_option == 'mobility_warmup'}
    assert destinations


@pytest.mark.parametrize('slot_id', ['push_main', 'flex'])
def test_adapter_rechecks_actual_destination_even_with_stale_bundles(slot_id):
    bp, rows, bundles = purpose_context()
    eid = reviewed()['id']
    # An explicit stale selection may not repurpose an approved adjunct as a main lift.
    stale = copy.deepcopy(bundles)
    for bundle in stale:
        if bundle['slotId'] == slot_id:
            bundle['candidates'].append({'exerciseId': eid})
    output = adapt_slot_selection_output(LLMRoutineSelectionOutput.model_validate({
        'summaryTitle': '경계 검증', 'rationaleSummary': [], 'warnings': [],
        'selections': [{'slotId': slot_id, 'exerciseId': eid, 'exerciseRationale': '검증'}]}),
        blueprint=bp, ranked_candidates=rows, slot_candidate_bundles=stale)
    if eid in {e.exercise_id for e in output.exercises}:
        assert output._adjunct_option_by_exercise_id[eid.lower()] == 'mobility_warmup'


@pytest.mark.parametrize('with_blueprint', [False, True])
def test_fallback_retains_reviewed_purpose_but_not_forged_identity(with_blueprint):
    bp, rows, _ = purpose_context()
    req = RoutineRequest(userId='purpose-test', experienceLevel='intermediate',
        targetSplitLabel='push', timeAvailableMin=60, adjunctOptions=['mobility_warmup'],
        availableEquipment=['MACHINE', 'BODYWEIGHT'])
    kwargs = dict(routine_blueprint=bp) if with_blueprint else {}
    result = generate_fallback_routine(req, rows, 20, **kwargs)
    assert reviewed()['id'] in {b.exercise_id for b in result.routine_blocks}
    rows[-1]['id'] = 'reviewed_dynamic_stretch'
    result = generate_fallback_routine(req, rows, 20, **kwargs)
    assert 'reviewed_dynamic_stretch' not in {b.exercise_id for b in result.routine_blocks}


def test_explicitly_suppressed_purpose_cannot_be_reintroduced_by_fallback():
    _, rows, _ = purpose_context()
    req = RoutineRequest(userId='suppression-test', experienceLevel='intermediate',
        targetSplitLabel='push', timeAvailableMin=60, adjunctOptions=['mobility_warmup'])
    result = generate_fallback_routine(req, rows, 20, applied_adjunct_options=[])
    assert reviewed()['id'] not in {b.exercise_id for b in result.routine_blocks}


@pytest.mark.parametrize('defect', ['unknown_option', 'invalid_id', 'wrong_review'])
def test_malformed_catalog_fails_closed(monkeypatch, tmp_path, defect):
    import json
    from engines import adjunct_admission as admission
    data = json.loads(admission.CATALOG_PATH.read_text())
    if defect == 'unknown_option':
        data['options']['unreviewed_purpose'] = data['options']['mobility_warmup']
    elif defect == 'invalid_id':
        data['options']['mobility_warmup'][0]['exerciseId'] = 'V24ROW-fake'
    else:
        data['reviewStatus'] = 'candidate'
    path = tmp_path/'catalog.json'
    path.write_text(json.dumps(data))
    monkeypatch.setattr(admission, 'CATALOG_PATH', path)
    admission.reviewed_purpose_index.cache_clear()
    try:
        assert admission.reviewed_purpose_index() == {}
    finally:
        admission.reviewed_purpose_index.cache_clear()

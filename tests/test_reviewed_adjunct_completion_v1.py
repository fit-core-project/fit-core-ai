"""Independent-review reproductions: completion, legacy repair, purpose reservation."""
import pytest

from engines.experience_policy import automatic_recommendation_eligibility
from engines.routine_blueprint import build_routine_blueprint
from engines.routine_selection import _accessory_group_options, _candidate_plan, adapt_structured_routine_response
from engines.schemas import LLMRoutineOutput
from engines.prescription.adjuncts import apply_requested_adjunct_prescription
from engines.llm_parser import validate_and_repair_routine_output
from engines.fallback import _ensure_requested_option_adjuncts_for_fallback
from tests.test_reviewed_adjunct_admission_v1 import reviewed, purpose_context


def dynamic_chest():
    return dict(reviewed(), id='V24ROW-022960', name_kr='문틀 흉근 동적 스트레치',
        name_en='Doorway Chest Stretch - dynamic', primary_muscle='chest')


def static_chest():
    return dict(dynamic_chest(), id='V24ROW-022956', name_kr='문틀 흉근 스트레치',
        name_en='Doorway Chest Stretch', adjunct_option_codes=['cooldown_stretch'])


@pytest.mark.parametrize('structure', ['circuit', 'density'])
def test_completion_cannot_promote_denied_adjunct_from_stale_flex(structure):
    bp = build_routine_blueprint(target_split_label='full_body', target_muscles=[],
        goal='hypertrophy', experience_level='intermediate', readiness_level='normal',
        time_available_min=60, adjunct_options=['mobility_warmup'], session_structure=structure)
    a = reviewed()
    b = dict(id='ordinary_chest', name_kr='가슴 운동', primary_muscle='chest',
        movement_type='ISOLATION', equipment_req='BODYWEIGHT')
    bundles = [{'slotId': 'flex', 'candidates': [{'exerciseId': c['id']} for c in (a, b)]}]
    pairs = _accessory_group_options(bp, bundles, [a, b])
    assert not any(c['id'] == a['id'] for pair in pairs for _, c in pair)
    assert _accessory_group_options(bp, bundles, [dict(a, main_routine_default_allowed=1), b])


def test_claimed_review_flags_with_self_consistent_unknown_identity_do_not_approve():
    row = dict(reviewed(), id='unreviewed', catalog_exercise_id='unreviewed',
        adjunct_catalog_review_status='core-reviewed', adjunct_catalog_version='routine-adjunct-catalog-v1')
    assert not automatic_recommendation_eligibility(row, 'advanced', adjunct_options=['mobility_warmup']).eligible


@pytest.mark.parametrize('as_dict', [False, True])
def test_legacy_response_cannot_keep_second_purpose_only_id_as_general_work(as_dict):
    bp, ordinary, bundles = purpose_context()
    rows = [*ordinary, dynamic_chest()]
    output = LLMRoutineOutput(total_estimated_time=30, summary_title='경계 검증',
        rationale_summary=[], warnings=[], exercises=[
            _candidate_plan(c, '검증', goal='hypertrophy') for c in [ordinary[0], reviewed(), dynamic_chest()]])
    output = adapt_structured_routine_response(output.model_dump() if as_dict else output,
        blueprint=bp, slot_candidate_bundles=bundles, ranked_candidates=rows)
    output = apply_requested_adjunct_prescription(output, ['mobility_warmup'], rows)
    validated, reason = validate_and_repair_routine_output(output, rows, [], [],
        experience_level='intermediate', adjunct_options=['mobility_warmup'])
    assert validated is not None, reason
    by_id = {c['id']: c for c in rows}
    for ex in validated.exercises:
        option = validated._adjunct_option_by_exercise_id.get(ex.exercise_id.lower())
        assert automatic_recommendation_eligibility(by_id[ex.exercise_id], 'intermediate',
            adjunct_options=[option] if option else []).eligible
    assert sum(ex.exercise_id in {reviewed()['id'], dynamic_chest()['id']} for ex in validated.exercises) == 1


def test_general_repair_cannot_consume_an_unused_purpose_only_candidate():
    bp, rows, _ = purpose_context()
    extras = dict(dynamic_chest(), primary_muscle='chest')
    plans = [_candidate_plan(rows[0], '검증', goal='hypertrophy'),
             _candidate_plan(dict(rows[0], id='unknown'), '검증', goal='hypertrophy')]
    output = LLMRoutineOutput(total_estimated_time=20, summary_title='경계 검증',
        rationale_summary=[], warnings=[], exercises=plans)
    validated, reason = validate_and_repair_routine_output(output, [extras, rows[0]], [], [],
        experience_level='intermediate', adjunct_options=['mobility_warmup'])
    assert validated is not None, reason
    assert extras['id'] not in {ex.exercise_id for ex in validated.exercises}


def test_fallback_reservation_matches_exact_reviewed_purpose_before_capacity_trim():
    ordinary = [dict(id=f'ordinary_{i}', name_kr='일반 운동', primary_muscle='chest',
        movement_type='COMPOUND', equipment_req='MACHINE') for i in range(5)]
    result = _ensure_requested_option_adjuncts_for_fallback(
        [*ordinary, static_chest(), dynamic_chest()], ['mobility_warmup', 'cooldown_stretch'], 6, 4)
    assert {static_chest()['id'], dynamic_chest()['id']} <= {c['id'] for c in result[:6]}


def test_prescription_does_not_assign_static_cooldown_to_dynamic_purpose():
    rows = [static_chest(), dynamic_chest()]
    output = LLMRoutineOutput(total_estimated_time=10, summary_title='경계 검증',
        rationale_summary=[], warnings=[], exercises=[_candidate_plan(c, '검증', goal='hypertrophy') for c in rows])
    output = apply_requested_adjunct_prescription(output, ['mobility_warmup', 'cooldown_stretch'], rows)
    assert output._adjunct_option_by_exercise_id == {
        static_chest()['id'].lower(): 'cooldown_stretch',
        dynamic_chest()['id'].lower(): 'mobility_warmup'}

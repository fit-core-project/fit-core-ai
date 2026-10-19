"""Final-path regression cases from the second independent review."""
from engines.routine_selection import _candidate_plan
from engines.schemas import LLMRoutineOutput, LLMSubstitutionCandidate, RoutineRequest
from engines.prescription.adjuncts import apply_requested_adjunct_prescription
from engines.llm_parser import validate_and_repair_routine_output
from engines.routine_pipeline import _ensure_requested_adjunct_style_candidate
from tests.test_reviewed_adjunct_admission_v1 import reviewed, purpose_context
from tests.test_reviewed_adjunct_completion_v1 import dynamic_chest


def output(rows):
    return LLMRoutineOutput(total_estimated_time=20, summary_title='최종 경계 검증',
        rationale_summary=[], warnings=[], exercises=[
            _candidate_plan(c, '검증', goal='hypertrophy') for c in rows])


def test_duplicate_purpose_id_does_not_lose_mapping_or_gain_second_purpose_exercise():
    _, main, _ = purpose_context()
    a = dynamic_chest()
    b = dict(a, id='V24ROW-022964', name_kr='광배 벤치 동적 스트레치')
    # Use another actual mobility-reviewed ID with the same primary for the repair trap.
    b['id'] = 'V24ROW-022976'
    rows = [main[0], a, b]
    value = apply_requested_adjunct_prescription(output([main[0], a, a]), ['mobility_warmup'], rows)
    validated, reason = validate_and_repair_routine_output(value, rows, [], [],
        experience_level='intermediate', adjunct_options=['mobility_warmup'])
    assert validated is not None, reason
    assert [ex.exercise_id for ex in validated.exercises].count(a['id']) == 1
    assert b['id'] not in {ex.exercise_id for ex in validated.exercises}
    assert validated._adjunct_option_by_exercise_id[a['id'].lower()] == 'mobility_warmup'


def test_legacy_style_fill_does_not_add_second_unassigned_purpose_exercise():
    _, main, _ = purpose_context()
    a = dict(reviewed(), score_boosts=[{'rule_code': 'training_style_match_boost', 'constraint_code': 'yoga_pilates'}])
    b = dynamic_chest()
    rows = [main[0], b, a]
    value = apply_requested_adjunct_prescription(output([main[0], b]), ['mobility_warmup'], rows)
    req = RoutineRequest(userId='style-test', experienceLevel='intermediate', timeAvailableMin=60,
                         trainingStyle='yoga_pilates')
    result = _ensure_requested_adjunct_style_candidate(value, req, rows, 20, 6)
    assert a['id'] not in {ex.exercise_id for ex in result.exercises}


def test_legacy_style_fill_assigns_reviewed_purpose_and_honors_suppression():
    _, main, _ = purpose_context()
    a = dict(reviewed(), score_boosts=[{'rule_code': 'training_style_match_boost', 'constraint_code': 'yoga_pilates'}])
    rows = [main[0], a]
    req = RoutineRequest(userId='style-test', experienceLevel='intermediate', timeAvailableMin=60,
                         trainingStyle='yoga_pilates')
    result = _ensure_requested_adjunct_style_candidate(output([main[0]]), req, rows, 20, 6)
    assert result._adjunct_option_by_exercise_id[a['id'].lower()] == 'mobility_warmup'
    result = _ensure_requested_adjunct_style_candidate(output([main[0]]), req, rows, 20, 6,
                                                     applied_adjunct_options=[])
    assert a['id'] not in {ex.exercise_id for ex in result.exercises}


def test_main_exercise_substitution_cannot_advertise_purpose_only_candidate():
    _, main, _ = purpose_context()
    special = dynamic_chest()
    value = output([main[0]])
    value.exercises[0].substitution_candidates = [LLMSubstitutionCandidate(
        exercise_id=special['id'], exercise_name=special['name_kr'], reason='같은 가슴')]
    validated, reason = validate_and_repair_routine_output(value, [main[0], special], [], [],
        experience_level='intermediate', adjunct_options=['mobility_warmup'])
    assert validated is not None, reason
    assert not validated.exercises[0].substitution_candidates

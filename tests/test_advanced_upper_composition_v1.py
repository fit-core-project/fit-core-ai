from engines.routine_blueprint import build_blueprint_candidate_bundles, build_routine_blueprint
from engines.routine_selection import adapt_slot_selection_output, apply_blueprint_execution_policy
from engines.schemas import LLMRoutineSelectionOutput
from tests.test_accessory_group_planning import candidate
import pytest


def complete_pool():
    return [candidate('bench', 'Barbell Bench Press', 'chest', 'COMPOUND', 'BARBELL'),
            candidate('row', 'Dumbbell Row', 'upper-back', 'COMPOUND', 'DUMBBELL'),
            candidate('lat', 'Machine Lat Pulldown', 'upper-back', 'COMPOUND', 'MACHINE'),
            candidate('curl', 'Dumbbell Curl', 'biceps', 'ISOLATION', 'DUMBBELL'),
            candidate('kickback', 'Dumbbell Kickback', 'triceps', 'ISOLATION', 'DUMBBELL')]


@pytest.mark.parametrize('structure', ['standard', 'density'])
def test_fallback_budgets_all_five_roles_before_spending_time(structure):
    from engines.fallback import generate_fallback_routine
    from engines.schemas import RoutineRequest
    bp = blueprint(session_structure=structure)
    req = RoutineRequest(time_available_min=60, target_split_label='upper',
                         goal='strength', experience_level='advanced',
                         session_structure=structure,
                         available_equipment=['BARBELL', 'DUMBBELL', 'MACHINE'])
    result = generate_fallback_routine(req, complete_pool(), 19, goal='strength',
        target_exercise_count=5, minimum_main_exercise_count=4, routine_blueprint=bp)
    by_id = {b.exercise_id: b for b in result.routine_blocks}
    assert set(by_id) == {'bench', 'row', 'lat', 'curl', 'kickback'}
    assert all(len(by_id[e].prescription) >= 2 for e in ('curl', 'kickback'))
    assert all(len(by_id[e].prescription) >= 3 for e in ('bench', 'row', 'lat'))
    assert 0 < result.total_estimated_time <= 60
    assert sum(len(b.prescription) for b in result.routine_blocks) <= 19
    assert by_id['bench'].execution_group_type == 'single'
    assert by_id['row'].execution_group_type == 'single'
    if structure == 'density':
        assert by_id['curl'].execution_group_type == 'density'
        assert by_id['kickback'].execution_group_type == 'density'


def prescribed_output(structure='standard'):
    bp = blueprint(session_structure=structure)
    pool = complete_pool()
    out = adapt_slot_selection_output(LLMRoutineSelectionOutput(selections=[]),
        blueprint=bp, ranked_candidates=pool,
        slot_candidate_bundles=build_blueprint_candidate_bundles(bp, pool))
    for p in out.exercises:
        p.sets = 5 if p.movement_type == 'COMPOUND' else 2
        p.rest_time_sec = 180 if p.movement_type == 'COMPOUND' else 90
    return bp, apply_blueprint_execution_policy(out, bp)


@pytest.mark.parametrize('structure', ['standard', 'density'])
@pytest.mark.parametrize('set_cap', [13, 16, 19])
def test_joint_allocator_preserves_roles_caps_rest_and_determinism(structure, set_cap):
    from engines.prescription.upper_role_allocation import allocate_upper_role_sets
    from engines.prescription.estimator import estimate_routine_time_min
    bp, out = prescribed_output(structure)
    before = {p.exercise_id: (p.sets, p.rest_time_sec) for p in out.exercises}
    assert allocate_upper_role_sets(out, bp, 60, set_cap)
    for p in out.exercises:
        assert (3 if p.movement_type == 'COMPOUND' else 2) <= p.sets <= before[p.exercise_id][0]
        assert p.rest_time_sec == before[p.exercise_id][1]
    assert sum(p.sets for p in out.exercises) <= set_cap
    assert out.total_estimated_time == estimate_routine_time_min(
        out.exercises, execution_metadata=out._execution_metadata_by_exercise_id) <= 60
    first = [p.model_dump() for p in out.exercises]
    assert allocate_upper_role_sets(out, bp, 60, set_cap)
    assert [p.model_dump() for p in out.exercises] == first


@pytest.mark.parametrize('minutes,cap', [(5, 19), (60, 12)])
def test_impossible_joint_budget_leaves_input_unchanged(minutes, cap):
    from engines.prescription.upper_role_allocation import allocate_upper_role_sets
    bp, out = prescribed_output()
    before = out.model_dump()
    assert not allocate_upper_role_sets(out, bp, minutes, cap)
    assert out.model_dump() == before


def test_joint_allocator_does_not_raise_an_existing_lower_role_cap():
    from engines.prescription.upper_role_allocation import allocate_upper_role_sets
    bp, out = prescribed_output()
    next(p for p in out.exercises if p.exercise_id == 'curl').sets = 1
    assert allocate_upper_role_sets(out, bp, 60, 19)
    assert next(p for p in out.exercises if p.exercise_id == 'curl').sets == 1


@pytest.mark.parametrize('method,extra', [('pausedReps', {'pause_sec': 2}),
                                         ('tempoEccentric', {'eccentric_sec': 4})])
def test_fallback_draft_regrouping_preserves_training_method_time(method, extra):
    from engines.routine_selection import apply_blueprint_execution_policy_to_draft
    from engines.schemas import RoutineDraftResponse, RoutineBlock, SetPrescription, TrainingMethodInstruction
    draft = RoutineDraftResponse(generation_status='fallback', status_reason_code='networkError',
        is_fallback=True, total_estimated_time=23, summary_title='Synthetic timing check', rationale_summary=[],
        routine_blocks=[RoutineBlock(order=1, exercise_id='bench', exercise_name='Bench Press',
            movement_type='COMPOUND', primary_muscles=['chest'], equipment_type='BARBELL',
            default_rest_sec=180, exercise_rationale='Synthetic timing check',
            prescription=[SetPrescription(set_index=i+1, target_reps=6, target_rest_sec=180)
                          for i in range(5)],
            training_method=TrainingMethodInstruction(code=method, label_ko='test',
                purpose_ko='test', instruction_ko='test', **extra))])
    result = apply_blueprint_execution_policy_to_draft(draft, blueprint(session_structure='standard'), complete_pool())
    assert result.total_estimated_time == 23
    result = apply_blueprint_execution_policy_to_draft(result, blueprint(session_structure='standard'), complete_pool())
    assert result.total_estimated_time == 23


@pytest.mark.parametrize('route', ['adapted', 'fallback'])
def test_shared_cable_station_cannot_group_the_vertical_back_role(route):
    from engines.fallback import generate_fallback_routine
    from engines.schemas import RoutineRequest
    pool = complete_pool()
    for row in pool:
        if row['id'] in {'lat', 'curl', 'kickback'}:
            row['equipment_req'] = 'CABLE'
    bp = blueprint()
    if route == 'adapted':
        out = adapt_slot_selection_output(LLMRoutineSelectionOutput(selections=[]),
            blueprint=bp, ranked_candidates=pool,
            slot_candidate_bundles=build_blueprint_candidate_bundles(bp, pool))
        out = apply_blueprint_execution_policy(out, bp)
        groups = {eid: m['executionGroupType'] for eid, m in out._execution_metadata_by_exercise_id.items()}
        counts = {p.exercise_id: p.sets for p in out.exercises}
    else:
        req = RoutineRequest(time_available_min=60, target_split_label='upper', goal='strength',
            experience_level='advanced', session_structure='density',
            available_equipment=['BARBELL', 'DUMBBELL', 'CABLE'])
        out = generate_fallback_routine(req, pool, 19, goal='strength', target_exercise_count=5,
            minimum_main_exercise_count=4, routine_blueprint=bp)
        groups = {b.exercise_id: b.execution_group_type for b in out.routine_blocks}
        counts = {b.exercise_id: len(b.prescription) for b in out.routine_blocks}
        assert 0 < out.total_estimated_time <= 60
    assert set(groups) == {'bench', 'row', 'lat', 'curl', 'kickback'}
    assert all(groups[eid] == 'single' for eid in ('bench', 'row', 'lat'))
    assert counts['lat'] >= 3
    assert groups['curl'] == groups['kickback'] == 'density'


def blueprint(**delta):
    values = dict(target_split_label='upper', target_muscles=[], goal='strength',
                  experience_level='advanced', readiness_level='normal',
                  time_available_min=60, session_structure='density')
    values.update(delta)
    return build_routine_blueprint(**values)


def test_long_advanced_upper_has_distinct_back_roles_and_room_for_accessories():
    bp = blueprint()
    assert bp.target_main_exercise_count == 5
    roles = {slot.slot_id: slot for slot in bp.slots}
    assert roles['upper_pull_main'].preferred_selection_patterns == ('horizontal_row',)
    assert roles['upper_vertical_pull'].preferred_selection_patterns == ('vertical_pull',)
    assert roles['upper_vertical_pull'].requirement == 'preferred_when_requested_and_safe'
    assert bp.execution_policy.main_lift_group_type == 'single'


def test_vertical_pull_cannot_fill_both_back_roles():
    bp = blueprint()
    pool = [candidate('bench', 'Barbell Bench Press', 'chest', 'COMPOUND', 'BARBELL'),
            candidate('row', 'Dumbbell Row', 'upper-back', 'COMPOUND', 'DUMBBELL'),
            candidate('lat', 'Machine Lat Pulldown', 'upper-back', 'COMPOUND', 'MACHINE'),
            candidate('curl', 'Dumbbell Curl', 'biceps', 'ISOLATION', 'DUMBBELL'),
            candidate('kickback', 'Dumbbell Kickback', 'triceps', 'ISOLATION', 'DUMBBELL')]
    bundles = build_blueprint_candidate_bundles(bp, pool)
    by_slot = {b['slotId']: b for b in bundles}
    assert [c['exerciseId'] for c in by_slot['upper_pull_main']['candidates']] == ['row']
    assert [c['exerciseId'] for c in by_slot['upper_vertical_pull']['candidates']] == ['lat']
    output = adapt_slot_selection_output(LLMRoutineSelectionOutput(selections=[]),
        blueprint=bp, ranked_candidates=pool, slot_candidate_bundles=bundles)
    output = apply_blueprint_execution_policy(output, bp)
    assert {p.exercise_id for p in output.exercises} == {'bench', 'row', 'lat', 'curl', 'kickback'}
    for eid in ('bench', 'row', 'lat'):
        assert output._execution_metadata_by_exercise_id[eid]['executionGroupType'] == 'single'
    assert output._execution_metadata_by_exercise_id['curl']['executionGroupType'] == 'density'
    assert output._execution_metadata_by_exercise_id['kickback']['executionGroupType'] == 'density'


def test_missing_vertical_candidate_does_not_duplicate_row_or_create_an_exercise():
    bp = blueprint()
    pool = [candidate('bench', 'Barbell Bench Press', 'chest', 'COMPOUND', 'BARBELL'),
            candidate('row', 'Dumbbell Row', 'upper-back', 'COMPOUND', 'DUMBBELL')]
    bundles = build_blueprint_candidate_bundles(bp, pool)
    output = adapt_slot_selection_output(LLMRoutineSelectionOutput(selections=[]),
        blueprint=bp, ranked_candidates=pool, slot_candidate_bundles=bundles)
    assert [p.exercise_id for p in output.exercises] == ['bench', 'row']


def test_short_beginner_recovery_and_explicit_single_target_keep_old_roles():
    for delta in ({'time_available_min': 45}, {'experience_level': 'intermediate'},
                  {'experience_level': 'beginner'}, {'readiness_level': 'low'},
                  {'doms_data': {'upper-back': 2}}, {'pain_areas': ['front-deltoids']},
                  {'target_muscles': ['chest']}):
        bp = blueprint(**delta)
        assert 'upper_vertical_pull' not in {s.slot_id for s in bp.slots}


def test_standard_advanced_upper_keeps_all_main_lifts_single():
    bp = blueprint(session_structure='standard')
    assert bp.target_main_exercise_count == 5
    assert bp.execution_policy.main_lift_group_type == 'single'


def test_explicit_arm_roles_prevent_front_raise_displacing_biceps_in_density():
    bp = blueprint()
    pool = [candidate('bench', 'Barbell Bench Press', 'chest', 'COMPOUND', 'BARBELL'),
            candidate('row', 'Dumbbell Row', 'upper-back', 'COMPOUND', 'DUMBBELL'),
            candidate('lat', 'Machine Lat Pulldown', 'upper-back', 'COMPOUND', 'MACHINE'),
            candidate('front', 'Dumbbell Front Raise', 'front-deltoids', 'ISOLATION', 'DUMBBELL', 400),
            candidate('curl', 'Dumbbell Curl', 'biceps', 'ISOLATION', 'DUMBBELL'),
            candidate('kickback', 'Dumbbell Kickback', 'triceps', 'ISOLATION', 'DUMBBELL')]
    bundles = build_blueprint_candidate_bundles(bp, pool)
    output = adapt_slot_selection_output(LLMRoutineSelectionOutput(selections=[]),
        blueprint=bp, ranked_candidates=pool, slot_candidate_bundles=bundles)
    assert {p.exercise_id for p in output.exercises} == {'bench', 'row', 'lat', 'curl', 'kickback'}

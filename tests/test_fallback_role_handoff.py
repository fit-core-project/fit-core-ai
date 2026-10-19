"""The selected role must survive the fallback list boundary."""
from engines.fallback import _blueprint_fallback_candidates
from engines.routine_blueprint import build_routine_blueprint
from engines.schemas import LLMExercisePlan, LLMRoutineOutput
from engines.specialization_policy import apply_blueprint_specialization_volume
from engines.fallback import generate_fallback_routine
from engines.schemas import RoutineRequest


def blueprint():
    return build_routine_blueprint(
        target_split_label="legs", target_muscles=[], goal="hypertrophy",
        experience_level="advanced", readiness_level="normal", time_available_min=60,
        specialization_focus="calvesEmphasis",
    )


def rows():
    return [dict(id=eid, name_kr=name, primary_muscle=muscle, movement_type=movement,
                 equipment_req="MACHINE", main_routine_default_allowed=1,
                 score=100, efficiency_tier=3, difficulty_tier=2,
                 quality_score=80, strength_effect="high", hypertrophy_effect="high",
                 failure_penalty="low", mainstream_gym_score=80,
                 exercise_familiarity_tier="common", injury_caution_level="low",
                 prescription_mode="reps")
            for eid, name, muscle, movement in (
                ("main", "레그 프레스", "quadriceps", "COMPOUND"),
                ("curl", "머신 레그 컬", "hamstring", "ISOLATION"),
                ("calf", "머신 카프 레이즈", "calves", "ISOLATION"),
            )]


def test_fallback_candidate_list_keeps_selected_role_mapping():
    from engines.experience_policy import main_role_eligibility
    assert main_role_eligibility(rows()[0], "advanced", "hypertrophy").eligible
    selected = _blueprint_fallback_candidates(blueprint(), rows())
    assert selected._fallback_slot_by_exercise_id == {
        "main": "lower_main", "curl": "posterior_chain", "calf": "calves_accessory",
    }


def test_fallback_preserves_adapter_rationale_for_each_selected_role():
    candidates = rows()
    candidates[1]['score_display_reasons'] = ['근육통 부담을 반영해 우선순위를 조정했습니다.']
    selected = _blueprint_fallback_candidates(blueprint(), candidates)
    rationales = selected._fallback_rationale_by_exercise_id
    assert set(rationales) == {'main', 'curl', 'calf'}
    assert '근육통 부담' in rationales['curl']
    assert '종아리' in rationales['calf']
    assert rationales['main'] != rationales['curl']


def test_final_fallback_retains_selected_role_and_candidate_context():
    candidates = rows()
    candidates[1]['score_display_reasons'] = ['근육통 부담을 반영해 우선순위를 조정했습니다.']
    request = RoutineRequest(target_split_label='legs', time_available_min=60,
                             experience_level='advanced', specialization_focus='calvesEmphasis')
    selected = _blueprint_fallback_candidates(blueprint(), candidates)
    result = generate_fallback_routine(request, candidates, 20, routine_blueprint=blueprint())
    assert result.generation_status == 'fallback'
    assert result.status_reason_code == 'networkError'
    assert result.routine_blocks
    for block in result.routine_blocks:
        assert selected._fallback_rationale_by_exercise_id[block.exercise_id] in block.exercise_rationale
        assert '정확한 자세를 유지하세요' not in block.exercise_rationale
        assert '목표 근육의 수축과 이완을 의식' not in block.exercise_rationale
    assert '근육통 부담' in next(b for b in result.routine_blocks if b.exercise_id == 'curl').exercise_rationale


def test_legacy_fallback_uses_candidate_facts_without_inventing_a_role():
    result = generate_fallback_routine(
        RoutineRequest(target_split_label='legs', time_available_min=60), rows(), 20,
    )
    assert result.routine_blocks
    text = next(b for b in result.routine_blocks if b.exercise_id == 'curl').exercise_rationale
    assert '뒤 허벅지' in text or '햄스트링' in text
    assert '머신' in text
    assert '목표 부위 구성을 보완' in text


def test_isolation_used_as_required_main_keeps_three_sets_during_focus_transfer():
    plans = [LLMExercisePlan(
        exercise_id=row['id'], exercise_name=row['name_kr'],
        primary_muscles=[row['primary_muscle']], movement_type=row['movement_type'],
        sets=sets, target_reps=10, rest_time_sec=90, exercise_rationale='fixture',
    ) for row, sets in zip(rows(), (4, 3, 3))]
    result = LLMRoutineOutput(total_estimated_time=30, summary_title='fixture',
                              rationale_summary=[], exercises=plans)
    result._selection_slot_by_exercise_id = {
        "main": "lower_main", "curl": "posterior_chain", "calf": "calves_accessory",
    }
    apply_blueprint_specialization_volume(result, blueprint(), rows())
    assert {p.exercise_id: p.sets for p in result.exercises} == {"main": 3, "curl": 3, "calf": 4}

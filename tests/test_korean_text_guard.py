from engines.fallback import build_routine_draft
from engines.korean_text_guard import _is_english_heavy, separate_doms_from_pain_wording_on_response
from engines.schemas import LLMExercisePlan, LLMRoutineOutput, PainAreaEntry


def _has_hangul(value: str) -> bool:
    return any("\uac00" <= char <= "\ud7a3" for char in value)


def test_generated_response_user_facing_text_is_guarded_to_korean():
    output = LLMRoutineOutput(
        total_estimated_time=45,
        summary_title="Push Hypertrophy Routine",
        rationale_summary=[
            "The plan focuses on hypertrophy while avoiding shoulder strain.",
            "Cable and dumbbell accessories maintain triceps volume safely.",
        ],
        warnings=["Reduce load if shoulder discomfort appears during the session."],
        exercises=[
            LLMExercisePlan(
                exercise_id="cable_pushdown",
                exercise_name="트라이셉스 로프 푸시다운",
                movement_type="ISOLATION",
                primary_muscles=["triceps"],
                equipment_type="CABLE",
                target_reps=12,
                sets=3,
                rest_time_sec=75,
                exercise_rationale=(
                    "This is a triceps isolation movement selected for hypertrophy "
                    "because cable equipment is available."
                ),
            )
        ],
    )

    response = build_routine_draft(output, "success", "none", False)

    assert _has_hangul(response.summary_title)
    assert not _is_english_heavy(response.summary_title)
    assert all(_has_hangul(item) and not _is_english_heavy(item) for item in response.rationale_summary)
    assert all(_has_hangul(item) and not _is_english_heavy(item) for item in response.warnings)
    assert all(
        _has_hangul(block.exercise_rationale) and not _is_english_heavy(block.exercise_rationale)
        for block in response.routine_blocks
    )

    block = response.routine_blocks[0]
    assert block.exercise_id == "cable_pushdown"
    assert block.primary_muscles == ["triceps"]
    assert block.equipment_type == "CABLE"
    assert block.prescription[0].set_type == "working"


def test_korean_text_with_schema_tokens_is_preserved():
    output = LLMRoutineOutput(
        total_estimated_time=30,
        summary_title="푸시 근비대 루틴",
        rationale_summary=["BARBELL 제한을 고려해 chest 후보 중 안전한 운동을 선택했습니다."],
        warnings=["통증이 있으면 세트를 줄이세요."],
        exercises=[
            LLMExercisePlan(
                exercise_id="pushup",
                exercise_name="푸시업",
                movement_type="COMPOUND",
                primary_muscles=["chest"],
                equipment_type="BODYWEIGHT",
                target_reps=10,
                sets=2,
                rest_time_sec=75,
                exercise_rationale="BODYWEIGHT 후보이며 chest 목표에 맞아 선택했습니다.",
            )
        ],
    )

    response = build_routine_draft(output, "success", "none", False)

    assert response.summary_title == "푸시 근비대 루틴"
    assert response.rationale_summary == ["바벨 제한을 고려해 가슴 후보 중 상대적으로 부담이 낮은 운동을 선택했습니다."]
    assert response.warnings == ["통증이 있으면 세트를 줄이세요."]
    assert response.routine_blocks[0].exercise_rationale == "맨몸 후보이며 가슴 목표에 맞아 선택했습니다."


def test_user_facing_text_repairs_utf8_hex_markers_from_local_model():
    output = LLMRoutineOutput(
        total_estimated_time=30,
        summary_title="상체 루틴",
        rationale_summary=["메인 스트<0xEB><0xA0><0x9D>스 운동을 먼저 배치했습니다."],
        warnings=["깨진 문자 \ufffd가 보이면 제거합니다."],
        exercises=[
            LLMExercisePlan(
                exercise_id="machine_press",
                exercise_name="머신 체스트 프레스",
                movement_type="COMPOUND",
                primary_muscles=["chest"],
                equipment_type="MACHINE",
                target_reps=10,
                sets=2,
                rest_time_sec=90,
                exercise_rationale="스트<0xEB><0xA0><0x9D>스 중심 메인 운동입니다.",
            )
        ],
    )

    response = build_routine_draft(output, "success", "none", False)
    text = " ".join([
        *response.rationale_summary,
        *response.warnings,
        response.routine_blocks[0].exercise_rationale,
    ])

    assert "<0x" not in text
    assert "\ufffd" not in text
    assert "스트렝스" not in text
    assert "근력" in text


def test_user_facing_text_repairs_multi_character_utf8_hex_marker_sequence():
    output = LLMRoutineOutput(
        total_estimated_time=30,
        summary_title="하체 루틴",
        rationale_summary=[
            "메인 <0xEC><0x9A><0xB4><0xEB><0x8F><0x99>을 먼저 배치했습니다."
        ],
        warnings=[],
        exercises=[],
    )

    response = build_routine_draft(output, "success", "none", False)

    assert response.rationale_summary == ["메인 운동을 먼저 배치했습니다."]


def test_user_facing_text_uses_safe_fallback_when_hex_marker_is_not_valid_utf8():
    output = LLMRoutineOutput(
        total_estimated_time=30,
        summary_title="깨진 <0xFF> 제목",
        rationale_summary=[],
        warnings=[],
        exercises=[],
    )

    response = build_routine_draft(output, "success", "none", False)

    assert response.summary_title == "맞춤 운동 루틴"


def test_user_facing_text_replaces_exercise_ids_with_names():
    output = LLMRoutineOutput(
        total_estimated_time=30,
        summary_title="하체 루틴",
        rationale_summary=["V24ROW-003179는 quadriceps 목표에 맞습니다."],
        warnings=["V24ROW-003179 수행 중 통증이 있으면 중단하세요."],
        exercises=[
            LLMExercisePlan(
                exercise_id="V24ROW-003179",
                exercise_name="머신 내로우 박스 터치 박스 스쿼트",
                movement_type="COMPOUND",
                primary_muscles=["quadriceps"],
                equipment_type="MACHINE",
                target_reps=10,
                sets=2,
                rest_time_sec=90,
                exercise_rationale="V24ROW-003179는 primary_muscles가 quadriceps이고 MACHINE 후보라 선택했습니다.",
            )
        ],
    )

    response = build_routine_draft(output, "success", "none", False)
    text = " ".join([
        *response.rationale_summary,
        *response.warnings,
        response.routine_blocks[0].exercise_rationale,
    ])

    assert "V24ROW-003179" not in text
    assert "quadriceps" not in text
    assert "primary_muscles" not in text
    assert "MACHINE" not in text
    assert "머신 내로우 박스 터치 박스 스쿼트" in text
    assert "대퇴사두" in text
    assert "주동근" in text
    assert "머신" in text
    assert response.routine_blocks[0].exercise_rationale == (
        "머신 내로우 박스 터치 박스 스쿼트는 주동근이 대퇴사두이고 머신 후보라 선택했습니다."
    )


def test_user_facing_text_softens_absolute_safety_or_medical_claims():
    output = LLMRoutineOutput(
        total_estimated_time=30,
        summary_title="통증 고려 루틴",
        rationale_summary=[
            "이 운동은 통증 없이 안전하게 수행할 수 있고 치료합니다.",
            "현재 조건에서는 안전한 선택입니다.",
            "안전한 머신 기반의 동작을 선택했습니다.",
        ],
        warnings=["100% 안전한 운동은 아니므로 통증이 있으면 중단하세요."],
        exercises=[
            LLMExercisePlan(
                exercise_id="pushup",
                exercise_name="푸시업",
                movement_type="COMPOUND",
                primary_muscles=["chest"],
                equipment_type="BODYWEIGHT",
                target_reps=10,
                sets=2,
                rest_time_sec=75,
                exercise_rationale="안전한 옵션이며 부상 방지에 좋습니다.",
            )
        ],
    )

    response = build_routine_draft(output, "success", "none", False)
    text = " ".join([
        *response.rationale_summary,
        *response.warnings,
        response.routine_blocks[0].exercise_rationale,
    ])

    assert "통증 없이" not in text
    assert "100% 안전" not in text
    assert "안전한 옵션" not in text
    assert "안전한 선택" not in text
    assert "안전한 머신 기반" not in text
    assert "치료합니다" not in text
    assert "부상 방지" not in text
    assert "통증 여부를 확인하며" in text
    assert "상대적으로 부담이 낮은 옵션" in text
    assert "상대적으로 부담이 낮은 선택" in text
    assert "부담을 조절하기 쉬운 머신 기반" in text
    assert "부상 위험을 낮추는 방향입니다" in text


def test_user_facing_text_replaces_injury_prevention_claim_with_preparation_wording():
    output = LLMRoutineOutput(
        total_estimated_time=20,
        summary_title="준비 운동",
        rationale_summary=["부상을 예방하는 동작으로 시작합니다."],
        warnings=[],
        exercises=[
            LLMExercisePlan(
                exercise_id="dynamic_stretch",
                exercise_name="문틀 흉근 동적 스트레치",
                movement_type="STATIC",
                primary_muscles=["chest"],
                equipment_type="BODYWEIGHT",
                target_reps=8,
                sets=1,
                rest_time_sec=30,
                exercise_rationale="부상을 예방하고 메인 운동을 준비합니다.",
            )
        ],
    )

    response = build_routine_draft(output, "success", "none", False)
    text = " ".join([*response.rationale_summary, response.routine_blocks[0].exercise_rationale])

    assert "부상 예방" not in text
    assert "부상을 예방" not in text
    assert "운동 전 준비" in text


def test_user_facing_text_normalizes_transliterated_experience_and_safe_candidate_wording():
    output = LLMRoutineOutput(
        total_estimated_time=20,
        summary_title="인터미디에이트 루틴",
        rationale_summary=["인터미디에이트 레벨의 안전 후보를 우선했습니다."],
        warnings=[],
        exercises=[
            LLMExercisePlan(
                exercise_id="machine_press",
                exercise_name="머신 체스트 프레스",
                movement_type="COMPOUND",
                primary_muscles=["chest"],
                equipment_type="MACHINE",
                target_reps=10,
                sets=2,
                rest_time_sec=90,
                exercise_rationale="인터미디에이트 레벨에서 점진적으로 수행합니다.",
            )
        ],
    )

    response = build_routine_draft(output, "success", "none", False)
    text = " ".join([
        response.summary_title,
        *response.rationale_summary,
        response.routine_blocks[0].exercise_rationale,
    ])

    assert "인터미디에이트" not in text
    assert "안전 후보" not in text
    assert "중급자" in text
    assert "제약을 통과한 후보" in text


def test_doms_only_text_is_not_framed_as_pain_or_injury():
    output = LLMRoutineOutput(
        total_estimated_time=30,
        summary_title="하체 루틴",
        rationale_summary=["대퇴사두 부상 우려가 있어 부담을 낮췄습니다."],
        warnings=["대퇴사두 통증이 있으면 강도를 줄이세요."],
        exercises=[
            LLMExercisePlan(
                exercise_id="leg_curl",
                exercise_name="레그 컬",
                movement_type="ISOLATION",
                primary_muscles=["hamstring"],
                equipment_type="MACHINE",
                target_reps=12,
                sets=2,
                rest_time_sec=75,
                exercise_rationale="대퇴사두 통증과 부상 우려를 고려해 선택했습니다.",
            )
        ],
    )
    response = build_routine_draft(output, "success", "none", False)

    guarded = separate_doms_from_pain_wording_on_response(
        response,
        doms_data={"quadriceps": 3},
        pain_areas=[],
    )
    text = " ".join([
        *guarded.rationale_summary,
        *guarded.warnings,
        guarded.routine_blocks[0].exercise_rationale,
    ])

    assert "부상" not in text
    assert "통증" not in text
    assert "부담가" not in text
    assert "근육통" in text
    assert "피로도" in text


def test_doms_only_guard_preserves_only_exact_server_ranker_caution_reason():
    trusted_reason = "부상 주의도가 높은 운동은 기본 추천 우선순위를 낮췄습니다."
    output = LLMRoutineOutput(
        total_estimated_time=30,
        summary_title="하체 루틴",
        rationale_summary=[],
        warnings=[],
        exercises=[
            LLMExercisePlan(
                exercise_id="leg_curl",
                exercise_name="레그 컬",
                movement_type="ISOLATION",
                primary_muscles=["hamstring"],
                equipment_type="MACHINE",
                target_reps=12,
                sets=2,
                rest_time_sec=75,
                exercise_rationale=(
                    "대퇴사두 부상 우려를 고려했습니다. " + trusted_reason
                ),
            )
        ],
    )
    response = build_routine_draft(output, "success", "none", False)
    response.routine_blocks[0].reasons = [
        trusted_reason,
        "대퇴사두 부상 우려가 있어 강도를 낮췄습니다.",
    ]
    original = response.model_dump()

    guarded = separate_doms_from_pain_wording_on_response(
        response,
        doms_data={"quadriceps": 2},
        pain_areas=[],
        trusted_reasons_by_exercise_id={"leg_curl": {trusted_reason}},
    )

    assert response.model_dump() == original
    assert guarded.routine_blocks[0].reasons == [
        trusted_reason,
        "대퇴사두 근육통/피로도 부담이 있어 강도를 낮췄습니다.",
    ]
    assert guarded.routine_blocks[0].exercise_rationale == (
        "대퇴사두 근육통/피로도 부담을 고려했습니다. " + trusted_reason
    )
    assert separate_doms_from_pain_wording_on_response(
        guarded,
        doms_data={"quadriceps": 2},
        pain_areas=[],
        trusted_reasons_by_exercise_id={"leg_curl": {trusted_reason}},
    ).model_dump() == guarded.model_dump()


def test_doms_only_guard_does_not_trust_unverified_injury_caution_copy():
    output = LLMRoutineOutput(
        total_estimated_time=30,
        summary_title="상체 루틴",
        rationale_summary=["이두 부상 주의도가 높은 운동을 피했습니다."],
        warnings=[],
        exercises=[],
    )
    response = build_routine_draft(output, "success", "none", False)

    guarded = separate_doms_from_pain_wording_on_response(
        response,
        doms_data={"biceps": 2},
        pain_areas=[],
        trusted_reasons_by_exercise_id={"other_exercise": {
            "부상 주의도가 높은 운동은 기본 추천 우선순위를 낮췄습니다."
        }},
    )

    assert guarded.rationale_summary == ["이두 근육통/피로도 주의도가 높은 운동을 피했습니다."]


def test_doms_guard_leaves_no_doms_or_pain_context_unchanged():
    output = LLMRoutineOutput(
        total_estimated_time=30,
        summary_title="상체 루틴",
        rationale_summary=["부상 주의도를 고려했습니다."],
        warnings=[],
        exercises=[],
    )
    response = build_routine_draft(output, "success", "none", False)

    assert separate_doms_from_pain_wording_on_response(
        response,
        doms_data={},
        pain_areas=[],
    ) is response


def test_actual_pain_text_is_preserved_when_same_part_is_pain_area():
    output = LLMRoutineOutput(
        total_estimated_time=30,
        summary_title="하체 루틴",
        rationale_summary=["대퇴사두 통증/부상 입력을 고려해 부담을 낮췄습니다."],
        warnings=[],
        exercises=[
            LLMExercisePlan(
                exercise_id="leg_curl",
                exercise_name="레그 컬",
                movement_type="ISOLATION",
                primary_muscles=["hamstring"],
                equipment_type="MACHINE",
                target_reps=12,
                sets=2,
                rest_time_sec=75,
                exercise_rationale="대퇴사두 통증과 부상 우려를 고려해 선택했습니다.",
            )
        ],
    )
    response = build_routine_draft(output, "success", "none", False)

    guarded = separate_doms_from_pain_wording_on_response(
        response,
        doms_data={"quadriceps": 3},
        pain_areas=[PainAreaEntry(body_part="quadriceps")],
    )
    text = " ".join([
        *guarded.rationale_summary,
        guarded.routine_blocks[0].exercise_rationale,
    ])

    assert "통증" in text
    assert "부상" in text


def test_mixed_pain_and_doms_sentence_only_rewrites_the_doms_part():
    output = LLMRoutineOutput(
        total_estimated_time=30,
        summary_title="상체 루틴",
        rationale_summary=["가슴 통증과 이두 통증을 함께 고려했습니다."],
        warnings=[],
        exercises=[],
    )
    response = build_routine_draft(output, "success", "none", False)

    guarded = separate_doms_from_pain_wording_on_response(
        response,
        doms_data={"biceps": 2},
        pain_areas=[PainAreaEntry(body_part="chest")],
    )

    assert guarded.rationale_summary == ["가슴 통증과 이두 근육통을 함께 고려했습니다."]


def test_mixed_pain_and_doms_sentence_uses_following_nearest_body_label():
    output = LLMRoutineOutput(
        total_estimated_time=30,
        summary_title="상체 루틴",
        rationale_summary=["가슴 통증과 통증이 있는 이두를 함께 고려했습니다."],
        warnings=[],
        exercises=[],
    )
    response = build_routine_draft(output, "success", "none", False)

    guarded = separate_doms_from_pain_wording_on_response(
        response,
        doms_data={"biceps": 2},
        pain_areas=[PainAreaEntry(body_part="chest")],
    )

    assert guarded.rationale_summary == [
        "가슴 통증과 근육통이 있는 이두를 함께 고려했습니다."
    ]


def test_mixed_pain_and_doms_sentence_preserves_ambiguous_equal_distance_wording():
    output = LLMRoutineOutput(
        total_estimated_time=30,
        summary_title="상체 루틴",
        rationale_summary=["가슴 통증 이두를 함께 고려했습니다."],
        warnings=[],
        exercises=[],
    )
    response = build_routine_draft(output, "success", "none", False)

    guarded = separate_doms_from_pain_wording_on_response(
        response,
        doms_data={"biceps": 2},
        pain_areas=[PainAreaEntry(body_part="chest")],
    )

    assert guarded.rationale_summary == ["가슴 통증 이두를 함께 고려했습니다."]


def test_mixed_pain_and_doms_sentence_keeps_closer_preceding_pain_label():
    output = LLMRoutineOutput(
        total_estimated_time=30,
        summary_title="상체 루틴",
        rationale_summary=["가슴 통증이 있는 이두 운동을 검토했습니다."],
        warnings=[],
        exercises=[],
    )
    response = build_routine_draft(output, "success", "none", False)

    guarded = separate_doms_from_pain_wording_on_response(
        response,
        doms_data={"biceps": 2},
        pain_areas=[PainAreaEntry(body_part="chest")],
    )

    assert guarded.rationale_summary == ["가슴 통증이 있는 이두 운동을 검토했습니다."]


def test_long_femur_user_text_does_not_use_ilium_term():
    output = LLMRoutineOutput(
        total_estimated_time=30,
        summary_title="하체 루틴",
        rationale_summary=["장골(긴 대퇴골) 신호와 근비대 목표를 고려했습니다."],
        warnings=[],
        exercises=[
            LLMExercisePlan(
                exercise_id="v_squat",
                exercise_name="브이 스쿼트",
                movement_type="COMPOUND",
                primary_muscles=["quadriceps"],
                equipment_type="MACHINE",
                target_reps=10,
                sets=2,
                rest_time_sec=90,
                exercise_rationale="장골 신호를 검토해 메인 운동으로 배치했습니다.",
            )
        ],
    )

    response = build_routine_draft(output, "success", "none", False)
    text = " ".join([*response.rationale_summary, response.routine_blocks[0].exercise_rationale])

    assert "장골" not in text
    assert "긴 대퇴 비율" in text


def test_response_aligns_before_and_after_wording_with_session_phase():
    output = LLMRoutineOutput(
        total_estimated_time=10,
        summary_title="준비와 마무리",
        rationale_summary=["요청한 보조 목적을 반영했습니다."],
        warnings=[],
        exercises=[
            LLMExercisePlan(
                exercise_id="warmup",
                exercise_name="동적 스트레치",
                movement_type="STATIC",
                primary_muscles=["chest"],
                target_reps=8,
                sets=2,
                rest_time_sec=15,
                target_rir=None,
                exercise_rationale="메인 루틴 전후로 부드럽게 준비합니다.",
            ),
            LLMExercisePlan(
                exercise_id="cooldown",
                exercise_name="정적 스트레치",
                movement_type="STATIC",
                primary_muscles=["chest"],
                target_reps=1,
                sets=1,
                rest_time_sec=15,
                target_rir=None,
                exercise_rationale="운동 전후로 가볍게 이완합니다.",
            ),
        ],
    )
    output._execution_metadata_by_exercise_id = {
        "warmup": {"sessionPhase": "warmup_or_preparation"},
        "cooldown": {"sessionPhase": "finisher_or_recovery"},
    }

    response = build_routine_draft(
        output,
        "success",
        "none",
        False,
        preserve_input_order=True,
    )

    assert response.routine_blocks[0].exercise_rationale == "메인 루틴 전에 부드럽게 준비합니다."
    assert response.routine_blocks[1].exercise_rationale == "운동 후에 가볍게 이완합니다."


def test_response_does_not_invent_trapezius_subregion_beyond_public_contract():
    output = LLMRoutineOutput(
        total_estimated_time=20,
        summary_title="당기기 루틴",
        rationale_summary=["등과 승모근을 함께 고려했습니다."],
        warnings=[],
        exercises=[
            LLMExercisePlan(
                exercise_id="lat_pulldown",
                exercise_name="머신 랫풀다운",
                movement_type="COMPOUND",
                primary_muscles=["upper-back"],
                target_reps=12,
                sets=3,
                rest_time_sec=75,
                exercise_rationale="상부 승모근 발달에 효과적인 운동입니다.",
            )
        ],
    )

    response = build_routine_draft(output, "success", "none", False)

    assert "상부 승모근" not in response.routine_blocks[0].exercise_rationale
    assert "승모근 발달" in response.routine_blocks[0].exercise_rationale

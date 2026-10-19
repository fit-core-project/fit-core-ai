from engines.routine_selection import _candidate_public_rationale


def test_candidate_rationale_uses_candidate_context_and_keeps_health_caution():
    candidate = {
        "id": "V24ROW-000123",
        "primary_muscle": "chest",
        "selected_equipment_code": "MACHINE",
        "score_display_reasons": [
            "헬스장에서 보편적으로 수행되는 운동이라 우선순위가 올라갔습니다.",
            "동일 목적 후보 중 익숙한 표준 운동을 먼저 제안하도록 우선순위를 높였습니다.",
            "중량 조절이 가능한 장비 운동이라 우선순위가 올라갔습니다.",
            "오늘 컨디션과 부상 이력을 반영해 우선순위를 조정했습니다.",
        ],
    }

    rationale = _candidate_public_rationale(candidate, purpose="가슴 메인 운동")

    assert rationale == _candidate_public_rationale(candidate, purpose="가슴 메인 운동")
    assert "가슴 중심의 머신 운동" in rationale
    assert "가슴 메인 운동에 맞춰" in rationale
    assert "오늘 컨디션과 부상 이력을 반영해 우선순위를 조정했습니다." in rationale
    assert "헬스장에서 보편적" not in rationale
    assert "익숙한 표준 운동" not in rationale
    assert "중량 조절이 가능한 장비 운동" not in rationale
    assert "V24ROW" not in rationale


def test_candidate_rationale_omits_unknown_metadata_instead_of_guessing():
    rationale = _candidate_public_rationale({
        "id": "unknown_exercise_id",
        "primary_muscle": "UNKNOWN_MUSCLE",
        "selected_equipment_code": "MYSTERY_EQUIPMENT",
        "score_display_reasons": [
            "헬스장에서 보편적으로 수행되는 운동이라 우선순위가 올라갔습니다.",
        ],
    }, purpose="role_id")

    assert rationale == "선택된 운동으로 목표 부위 구성을 보완하도록 배치했습니다."
    assert "UNKNOWN" not in rationale
    assert "MYSTERY" not in rationale
    assert "role_id" not in rationale


def test_candidate_rationale_prioritizes_late_health_caution_over_constraints():
    rationale = _candidate_public_rationale({
        "primary_muscle": "side-deltoids",
        "selected_equipment_code": "CABLE",
        "score_display_reasons": [
            "어깨 주동근을 직접 자극하는 운동이라 우선순위를 높였습니다.",
            "사용 가능한 장비 조건을 반영했습니다.",
            "최근 수행한 운동과 겹쳐 반복 노출을 줄였습니다.",
            "근육통 부담을 반영해 우선순위를 조정했습니다.",
        ],
    }, purpose="측면 어깨 보조 운동")

    assert "근육통 부담을 반영해 우선순위를 조정했습니다." in rationale
    assert "사용 가능한 장비 조건을 반영했습니다." in rationale
    assert "최근 수행한 운동과 겹쳐 반복 노출을 줄였습니다." not in rationale
    assert "어깨 주동근을 직접 자극하는 운동" not in rationale


def test_candidate_rationale_replaces_legacy_safe_alternative_phrase():
    rationale = _candidate_public_rationale({
        "primary_muscle": "quadriceps",
        "score_display_reasons": [
            "부상 이력을 반영한 안전 대체 후보로 우선순위를 조정했습니다.",
        ],
    })

    assert "부상 이력을 반영한 부담이 낮은 대체 후보" in rationale
    assert "안전 대체 후보" not in rationale

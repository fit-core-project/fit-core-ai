from collections import Counter

from scripts.build_current_comprehensive_expert_review_fixture_v3 import (
    CURRENT_PREINTEGRATION_KEYS,
    CURRENT_PUBLIC_KEYS,
    FORBIDDEN_LEGACY_KEYS,
    build_fixture,
)
from scripts.generate_current_comprehensive_expert_review_v1 import _comparison_table
from scripts.run_gemma4_quality_seed_live import request_from_seed


def test_v3_fixture_is_fresh_current_contract_matrix():
    rows = build_fixture()

    assert len(rows) == 81
    assert [row["scenarioId"] for row in rows] == [
        f"expert-current-v3-{index:03d}" for index in range(1, 82)
    ]
    assert [row["reviewOrder"] for row in rows] == list(range(1, 82))
    assert {row["expertDocumentVersion"] for row in rows} == {
        "routine-ai-core-current-comprehensive-expert-v3"
    }
    assert Counter(row["reviewSection"] for row in rows) == {
        "1. 기본 루틴과 운동 경력": 5,
        "2. 목표·시간·컨디션·장비": 13,
        "3. 근육통과 등록 부상": 7,
        "4. 관리 중 부상과 제한 동작": 9,
        "5. 동작 수행 어려움과 체형 특징": 6,
        "6. 세션 진행 방식": 4,
        "7. 추가 목적 운동": 6,
        "8. 상급자 세부 집중": 23,
        "9. 상급자 수행법": 8,
    }

    for row in rows:
        keys = set(row["input"])
        assert not keys.intersection(FORBIDDEN_LEGACY_KEYS)
        assert keys <= CURRENT_PUBLIC_KEYS | CURRENT_PREINTEGRATION_KEYS
        assert isinstance(row["input"]["currentDoms"], list)
        assert row["input"]["trainingStyle"] == "gym"
        request_from_seed(row)

    readiness_rows = [
        row for row in rows if row.get("comparisonGroup") == "readiness"
    ]
    assert {row["input"]["experienceLevel"] for row in readiness_rows} == {"beginner"}

    specialization_rows = [
        row
        for row in rows
        if str(row.get("comparisonGroup") or "").startswith("specialization_")
    ]
    assert {
        row["input"].get("specializationFocus")
        for row in specialization_rows
        if row["input"].get("specializationFocus")
    } == {
        "upperChest",
        "midChest",
        "lowerChest",
        "latEmphasis",
        "midBackEmphasis",
        "trapeziusEmphasis",
        "quadricepsEmphasis",
        "hamstringEmphasis",
        "gluteEmphasis",
        "adductorEmphasis",
        "abductorEmphasis",
        "calvesEmphasis",
        "frontDeltoidEmphasis",
        "sideDeltoidEmphasis",
        "rearDeltoidEmphasis",
        "bicepsEmphasis",
        "tricepsEmphasis",
        "forearmEmphasis",
    }
    assert all(
        "Emphasis" not in str(row.get("comparisonLabel") or "")
        and str(row.get("comparisonLabel") or "") not in {
            "upperChest",
            "midChest",
            "lowerChest",
        }
        for row in specialization_rows
    )


def test_v3_fixture_changes_only_declared_axis_in_core_control_groups():
    rows = build_fixture()
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        group = row.get("comparisonGroup")
        if group:
            grouped.setdefault(group, []).append(row)

    axis_by_group = {
        "experience": {"experienceLevel"},
        "goal": {"goal"},
        "readiness": {"readinessLevel"},
        "doms": {"currentDoms"},
        "ankle_status": {"injuryConditions"},
        "specialization_chest": {"specializationFocus"},
        "specialization_back": {"specializationFocus"},
        "specialization_legs": {"specializationFocus"},
        "specialization_shoulder": {"specializationFocus"},
        "specialization_arm": {"specializationFocus"},
        "advanced_method": {
            "targetSplitLabel",
            "goal",
            "readinessLevel",
            "currentDoms",
            "methodMode",
            "requestedMethod",
            "recentTrainingMethodUses",
        },
    }
    for group, allowed_changes in axis_by_group.items():
        inputs = [row["input"] for row in grouped[group]]
        all_keys = set().union(*(set(request) for request in inputs))
        changing = {
            key
            for key in all_keys
            if len({repr(request.get(key)) for request in inputs}) > 1
        }
        assert changing <= allowed_changes, (group, changing)


def test_v3_comparison_table_states_fixed_and_changed_conditions():
    fixture = {
        "scenarioId": "expert-current-v3-001",
        "reviewTitle": "초급자의 50분 전신 근비대",
        "comparisonLabel": "초급",
        "comparisonFixedConditions": "전신 · 근비대 · 50분",
    }
    result = {
        "exerciseCount": 2,
        "totalEstimatedTime": 42,
        "blockSummaries": [],
    }

    rendered = "\n".join(
        _comparison_table(
            "운동 경력 비교",
            [fixture],
            {fixture["scenarioId"]: result},
        )
    )

    assert "고정 조건: 전신 · 근비대 · 50분" in rendered
    assert "| 변경한 조건 |" in rendered
    assert "| 초급 | 2개 |" in rendered

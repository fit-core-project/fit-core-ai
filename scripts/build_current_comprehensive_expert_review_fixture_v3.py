#!/usr/bin/env python3
"""Build the v3 clean, current-contract expert review fixture.

This builder does not load or copy historical fixture rows. Every
scenario is declared from the current product inputs so a fresh live run can be
traced back to one canonical condition matrix.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "docs/reviews/fixtures/"
    "routine-ai-core-current-comprehensive-expert-review-v3.jsonl"
)

FULL_GYM = [
    "BARBELL",
    "DUMBBELL",
    "MACHINE",
    "CABLE",
    "SMITH_MACHINE",
    "BODYWEIGHT",
]

SPLIT_REVIEW_LABELS = {
    "chest": "가슴",
    "pull": "당기기",
    "legs": "하체",
    "shoulder": "어깨",
    "arm": "팔",
}

CURRENT_PUBLIC_KEYS = {
    "targetSplitLabel",
    "targetMuscles",
    "timeAvailableMin",
    "readinessLevel",
    "currentPainAreas",
    "currentDoms",
    "availableEquipment",
    "unavailableEquipment",
    "goal",
    "experienceLevel",
    "sessionStructure",
    "adjunctOptions",
    "profileSignals",
    "trainingStyle",
    "userNote",
}
CURRENT_PREINTEGRATION_KEYS = {
    "injuryConditions",
    "specializationFocus",
    "methodMode",
    "requestedMethod",
    "recentTrainingMethodUses",
}
FORBIDDEN_LEGACY_KEYS = {
    "advancedChestFocus",
    "anthropometrySignals",
    "conditionPolicies",
    "doms",
    "domsData",
    "mobilityLimits",
    "painAreas",
    "shoulderConditionStatus",
    "surgeryHistory",
    "trainingMethod",
}

COMMON_FAILURE_PATTERNS = [
    "fallback_or_non_success",
    "unavailable_equipment_selected",
    "catalog_id_in_user_text",
    "english_schema_token_in_user_text",
    "absolute_safety_claim",
    "medical_treatment_claim",
    "pain_free_claim",
    "rehab_treatment_claim",
]

SECTION_FOCUS = {
    "1. 기본 루틴과 운동 경력": [
        "운동 이름과 기술 난도가 해당 운동 경력에 맞는가",
        "큰 근육의 메인 운동부터 직접 보조 운동으로 이어지는 순서가 자연스러운가",
        "세트·반복·RIR·휴식이 혼자 수행하기 적절한가",
    ],
    "2. 목표·시간·컨디션·장비": [
        "변경한 한 가지 조건이 종목·처방·운동 수에 납득 가능한 차이를 만들었는가",
        "요청 시간과 실제 예상 시간이 현장에서 수용 가능한가",
        "허용하지 않은 장비나 과도한 볼륨이 포함되지 않았는가",
    ],
    "3. 근육통과 등록 부상": [
        "근육통 단계와 등록 부상을 서로 다른 정책으로 처리했는가",
        "관련 없는 근육까지 과도하게 막거나 안전 후보가 없는데 억지 추천하지 않았는가",
        "주의 안내가 치료·안전 보장처럼 보이지 않는가",
    ],
    "4. 관리 중 부상과 제한 동작": [
        "운동 미허가와 확인된 제한 동작을 구분했는가",
        "제한한 동작만 제외하고 가능한 목표는 보존했는가",
        "회복·제한 없음 상태를 현재 부상처럼 과도하게 처리하지 않았는가",
    ],
    "5. 동작 수행 어려움과 체형 특징": [
        "통증이 없는 수행 어려움과 부상을 혼동하지 않았는가",
        "운동 금지보다 후보 순위·세팅·가동범위 안내에 적절히 반영했는가",
        "체형 특징을 진단이나 절대적 제한으로 과장하지 않았는가",
    ],
    "6. 세션 진행 방식": [
        "메인 운동을 보호하고 보조 운동만 현실적으로 묶었는가",
        "같은 장비 또는 현장에서 이동 가능한 조합인가",
        "그룹 순서와 휴식 안내만으로 혼자 실행할 수 있는가",
    ],
    "7. 추가 목적 운동": [
        "추가 목적이 메인 헬스장 루틴을 밀어내지 않았는가",
        "준비·보조·마무리 위치와 처방 단위가 목적에 맞는가",
        "두 옵션을 함께 쓸 때 시간과 메인 운동의 최소 구성이 보존됐는가",
    ],
    "8. 상급자 세부 집중": [
        "집중 부위가 체감할 만큼 반영되면서 전체 균형을 해치지 않았는가",
        "생소함만 늘린 운동 대신 수행 가치가 있는 종목을 선택했는가",
        "집중 부위 직접 세트와 낮은 우선순위 세트의 교환이 자연스러운가",
    ],
    "9. 상급자 수행법": [
        "수행법을 붙인 운동과 목표의 조합이 적절한가",
        "최근 사용·낮은 컨디션·근육통에 따른 억제가 보수적으로 작동했는가",
        "수행 안내가 짧고 실제로 따라 하기 쉬운가",
    ],
}


def _request(
    *,
    split: str,
    target_muscles: list[str] | None = None,
    time_min: int = 50,
    goal: str = "hypertrophy",
    experience: str = "intermediate",
    readiness: str = "normal",
    equipment: list[str] | None = None,
    unavailable: list[str] | None = None,
    pain: list[dict[str, Any]] | None = None,
    doms: list[dict[str, Any]] | None = None,
    profile_signals: list[str] | None = None,
    session_structure: str = "standard",
    adjunct_options: list[str] | None = None,
    **extensions: Any,
) -> dict[str, Any]:
    request = {
        "targetSplitLabel": split,
        "targetMuscles": list(target_muscles or []),
        "timeAvailableMin": time_min,
        "readinessLevel": readiness,
        "currentPainAreas": deepcopy(pain or []),
        "currentDoms": deepcopy(doms or []),
        "availableEquipment": list(equipment or FULL_GYM),
        "unavailableEquipment": list(unavailable or []),
        "goal": goal,
        "experienceLevel": experience,
        "sessionStructure": session_structure,
        "adjunctOptions": list(adjunct_options or []),
        "profileSignals": list(profile_signals or []),
        "trainingStyle": "gym",
    }
    request.update(
        {
            key: deepcopy(value)
            for key, value in extensions.items()
            if value is not None
        }
    )
    return request


def _split_label_for_review(split: str) -> str:
    return SPLIT_REVIEW_LABELS.get(split, split)


def _case(
    rows: list[dict[str, Any]],
    *,
    section: str,
    title: str,
    request: dict[str, Any],
    comparison_group: str | None = None,
    comparison_label: str | None = None,
    comparison_fixed_conditions: str | None = None,
    focus: list[str] | None = None,
    expected: list[str] | None = None,
    candidate_empty_allowed: bool = False,
    required_patterns: list[str] | None = None,
) -> None:
    order = len(rows) + 1
    patterns = ["live_success", "language_quality_ok"]
    if candidate_empty_allowed:
        patterns = ["expected_policy_stop", "language_quality_ok"]
    if required_patterns:
        patterns.extend(required_patterns)
    rows.append(
        {
            "scenarioId": f"expert-current-v3-{order:03d}",
            "title": title,
            "input": request,
            "expectedHardConstraints": list(expected or []),
            "preferredPatterns": list(dict.fromkeys(patterns)),
            "requiredPreferredPatterns": list(dict.fromkeys(patterns)),
            "failurePatterns": list(COMMON_FAILURE_PATTERNS),
            "qualityRubric": [
                "current_expert_review_v3",
                "fresh_live_result_required",
                "valid_contract_json",
            ],
            "candidateEmptyAllowed": candidate_empty_allowed,
            "reviewOrder": order,
            "reviewSection": section,
            "reviewTitle": title,
            "reviewFocus": list(focus or SECTION_FOCUS[section]),
            "comparisonGroup": comparison_group,
            "comparisonLabel": comparison_label,
            "comparisonFixedConditions": comparison_fixed_conditions,
            "expertDocumentVersion": "routine-ai-core-current-comprehensive-expert-v3",
        }
    )


def build_fixture() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # 1. Basic routine, experience, and direct target editing.
    experience_fixed = "전신 · 근비대 · 50분 · 컨디션 보통 · 전체 헬스장 장비 · 통증·부상 없음"
    for level, label in (
        ("beginner", "초급"),
        ("intermediate", "중급"),
        ("advanced", "상급"),
    ):
        _case(
            rows,
            section="1. 기본 루틴과 운동 경력",
            title=f"{label}자의 50분 전신 근비대",
            request=_request(split="full_body", experience=level),
            comparison_group="experience",
            comparison_label=label,
            comparison_fixed_conditions=experience_fixed,
        )
    _case(
        rows,
        section="1. 기본 루틴과 운동 경력",
        title="중급자의 50분 당기기와 직접 이두 보강",
        request=_request(split="pull"),
        required_patterns=["direct_biceps_curl"],
        focus=[
            "등 메인 운동과 직접 이두 운동이 모두 포함되었는가",
            "후면 어깨·승모·전완이 직접 이두를 밀어내지 않았는가",
            "한 세션으로 운동 수와 순서가 자연스러운가",
        ],
    )
    _case(
        rows,
        section="1. 기본 루틴과 운동 경력",
        title="가슴 근육통 3단계에서 어깨·삼두만 직접 지정한 편집 루틴",
        request=_request(
            split="custom",
            target_muscles=["front-deltoids", "back-deltoids", "triceps"],
            doms=[{"bodyPart": "chest", "level": 3}],
        ),
        focus=[
            "직접 고른 어깨·삼두 범위를 정확히 지켰는가",
            "가슴 근육통 때문에 관련 없는 어깨·삼두까지 과도하게 막지 않았는가",
            "프리셋 편집이라는 제품 흐름으로 이해하기 쉬운가",
        ],
    )

    # 2. Time, equipment, goal, and readiness.
    _case(
        rows,
        section="2. 목표·시간·컨디션·장비",
        title="20분만 가능한 중급자의 밀기 루틴",
        request=_request(split="push", time_min=20),
        comparison_group="time_equipment",
        comparison_label="20분 밀기",
        comparison_fixed_conditions="중급 · 근비대 · 컨디션 보통 · 통증·부상 없음",
    )
    _case(
        rows,
        section="2. 목표·시간·컨디션·장비",
        title="90분 가능한 중급자의 밀기 근비대",
        request=_request(split="push", time_min=90),
        comparison_group="time_equipment",
        comparison_label="90분 밀기",
        comparison_fixed_conditions="중급 · 근비대 · 컨디션 보통 · 통증·부상 없음",
    )
    _case(
        rows,
        section="2. 목표·시간·컨디션·장비",
        title="덤벨과 맨몸만 가능한 50분 전신 루틴",
        request=_request(
            split="full_body",
            equipment=["DUMBBELL", "BODYWEIGHT"],
            unavailable=["BARBELL", "MACHINE", "CABLE", "SMITH_MACHINE"],
        ),
        comparison_group="time_equipment",
        comparison_label="덤벨·맨몸만",
        comparison_fixed_conditions="중급 · 근비대 · 컨디션 보통 · 통증·부상 없음",
    )
    _case(
        rows,
        section="2. 목표·시간·컨디션·장비",
        title="일반 머신·케이블만 허용하고 스미스 머신은 제외한 밀기 루틴",
        request=_request(
            split="push",
            equipment=["MACHINE", "CABLE"],
            unavailable=["SMITH_MACHINE"],
        ),
        comparison_group="time_equipment",
        comparison_label="머신·케이블, 스미스 제외",
        comparison_fixed_conditions="중급 · 근비대 · 50분 · 컨디션 보통 · 통증·부상 없음",
        expected=["MACHINE does not imply SMITH_MACHINE"],
    )

    goal_fixed = "하체 · 중급 · 50분 · 컨디션 보통 · 전체 헬스장 장비 · 통증·부상 없음"
    for code, label in (
        ("strength", "근력"),
        ("hypertrophy", "근비대"),
        ("endurance", "근지구력"),
        ("fatLoss", "체지방 감소"),
        ("recomposition", "체성분 개선"),
        ("generalFitness", "일반 체력"),
    ):
        _case(
            rows,
            section="2. 목표·시간·컨디션·장비",
            title=f"같은 하체 조건 비교: 목표 {label}",
            request=_request(split="legs", goal=code),
            comparison_group="goal",
            comparison_label=label,
            comparison_fixed_conditions=goal_fixed,
        )

    readiness_fixed = "전신 · 초급 · 근비대 · 50분 · 전체 헬스장 장비 · 통증·부상 없음"
    for code, label in (("low", "낮음"), ("normal", "보통"), ("high", "좋음")):
        _case(
            rows,
            section="2. 목표·시간·컨디션·장비",
            title=f"같은 초급 전신 조건에서 오늘 컨디션만 {label}으로 변경",
            request=_request(
                split="full_body",
                experience="beginner",
                readiness=code,
            ),
            comparison_group="readiness",
            comparison_label=label,
            comparison_fixed_conditions=readiness_fixed,
        )

    # 3. DOMS and binary registered injury.
    doms_fixed = "밀기 · 중급 · 근비대 · 50분 · 컨디션 보통 · 전체 헬스장 장비"
    _case(
        rows,
        section="3. 근육통과 등록 부상",
        title="비교 기준: 부상·근육통 없는 50분 밀기 루틴",
        request=_request(split="push"),
        comparison_group="doms",
        comparison_label="근육통 없음",
        comparison_fixed_conditions=doms_fixed,
        required_patterns=["direct_triceps"],
    )
    for level in (1, 2, 3):
        _case(
            rows,
            section="3. 근육통과 등록 부상",
            title=f"가슴 근육통 {level}단계의 밀기 요청",
            request=_request(
                split="push",
                doms=[{"bodyPart": "chest", "level": level}],
            ),
            comparison_group="doms",
            comparison_label=f"가슴 {level}단계",
            comparison_fixed_conditions=doms_fixed,
            candidate_empty_allowed=level == 3,
        )
    _case(
        rows,
        section="3. 근육통과 등록 부상",
        title="허리·요추 부상을 등록한 하체 요청",
        request=_request(
            split="legs",
            pain=[{"bodyPart": "lower-back"}],
        ),
        comparison_group="injury_boundary",
        comparison_label="허리 등록 부상 + 하체",
        comparison_fixed_conditions="중급 · 근비대 · 컨디션 보통 · 전체 헬스장 장비",
    )
    _case(
        rows,
        section="3. 근육통과 등록 부상",
        title="전면 어깨 부상을 등록한 전체 밀기 요청",
        request=_request(
            split="push",
            pain=[{"bodyPart": "front-deltoids"}],
        ),
        comparison_group="injury_boundary",
        comparison_label="전면 어깨 등록 부상 + 전체 밀기",
        comparison_fixed_conditions="중급 · 근비대 · 50분 · 컨디션 보통 · 전체 헬스장 장비",
        candidate_empty_allowed=True,
    )
    _case(
        rows,
        section="3. 근육통과 등록 부상",
        title="전면 어깨 부상에서 삼두만 직접 선택한 요청",
        request=_request(
            split="custom",
            target_muscles=["triceps"],
            time_min=35,
            equipment=["CABLE", "MACHINE"],
            pain=[{"bodyPart": "front-deltoids"}],
        ),
        comparison_group="injury_boundary",
        comparison_label="전면 어깨 등록 부상 + 삼두만",
        comparison_fixed_conditions="중급 · 근비대 · 컨디션 보통",
    )

    # 4. Managed injury states and movement restrictions.
    managed_fixed = "중급 · 근비대 · 50분 · 컨디션 보통 · 전체 헬스장 장비"
    for title, split, area, pattern in (
        ("관리 중 어깨: 머리 위 동작 제한", "push", "shoulder", "overheadPosition"),
        ("관리 중 무릎: 깊은 무릎 굽힘 제한", "legs", "knee", "deepKneeFlexion"),
        ("관리 중 허리: 힙힌지 제한", "legs", "lumbar", "hipHinge"),
        ("관리 중 손목: 부하가 실린 손목 신전 제한", "push", "wrist", "loadedWristExtension"),
    ):
        _case(
            rows,
            section="4. 관리 중 부상과 제한 동작",
            title=title,
            request=_request(
                split=split,
                injuryConditions=[
                    {
                        "areaCode": area,
                        "status": "managedWithRestrictions",
                        "restrictedMovementPatterns": [pattern],
                    }
                ],
            ),
            comparison_group="managed_restriction",
            comparison_label=title.replace("관리 중 ", ""),
            comparison_fixed_conditions=managed_fixed,
            expected=["restricted movement pattern remains excluded"],
        )

    ankle_targets = ["quadriceps", "hamstring", "gluteal", "calves"]
    ankle_fixed = "하체 4개 근육 직접 지정 · 중급 · 근비대 · 50분 · 컨디션 보통 · 전체 헬스장 장비"
    ankle_states = (
        ("발목 상태를 입력하지 않은 하체 기준선", None, [], "입력 없음"),
        ("발목 활성 부상·운동 미허가 상태", "activeUncleared", [], "활성·미허가"),
        (
            "발목 깊은 배굴만 제한된 관리 상태",
            "managedWithRestrictions",
            ["deepDorsiflexion"],
            "관리 중·깊은 배굴 제한",
        ),
        (
            "관리 중이지만 운동 제한은 없는 발목 상태",
            "managedNoRestrictions",
            [],
            "관리 중·제한 없음",
        ),
        ("회복 완료로 기록된 발목 상태", "recovered", [], "회복 완료"),
    )
    for title, status, patterns, label in ankle_states:
        extensions: dict[str, Any] = {}
        if status:
            extensions["injuryConditions"] = [
                {
                    "areaCode": "ankle",
                    "status": status,
                    "restrictedMovementPatterns": patterns,
                }
            ]
        _case(
            rows,
            section="4. 관리 중 부상과 제한 동작",
            title=title,
            request=_request(
                split="legs",
                target_muscles=ankle_targets,
                **extensions,
            ),
            comparison_group="ankle_status",
            comparison_label=label,
            comparison_fixed_conditions=ankle_fixed,
        )

    # 5. Current profile signals.
    profile_cases = (
        ("발뒤꿈치를 붙이고 깊게 앉기 어려운 하체 요청", "legs", "heel_down_deep_squat_difficult"),
        ("엉덩이를 접어 숙이기 어려운 하체 요청", "legs", "hip_hinge_difficult"),
        ("팔을 머리 위로 올리기 어려운 밀기 요청", "push", "overhead_reach_difficult"),
        ("손목을 뒤로 젖혀 버티기 어려운 밀기 요청", "push", "loaded_wrist_extension_difficult"),
        ("팔이 유난히 길다고 선택한 밀기 요청", "push", "self_reported_long_arms"),
        ("허벅지 구간이 유난히 길다고 선택한 하체 요청", "legs", "self_reported_long_thighs"),
    )
    for title, split, signal in profile_cases:
        targets = (
            ["quadriceps", "hamstring", "gluteal", "calves"]
            if split == "legs"
            else ["chest", "front-deltoids", "triceps"]
        )
        _case(
            rows,
            section="5. 동작 수행 어려움과 체형 특징",
            title=title,
            request=_request(
                split=split,
                target_muscles=targets,
                time_min=60,
                profile_signals=[signal],
            ),
            comparison_group="profile_signal",
            comparison_label=title.replace(" 요청", ""),
            comparison_fixed_conditions="중급 · 근비대 · 60분 · 컨디션 보통 · 전체 헬스장 장비 · 통증·부상 없음",
        )

    # 6. Execution structures.
    structure_cases = (
        ("일반 세트 방식의 밀기 루틴", "push", 50, "hypertrophy", "intermediate", "standard", FULL_GYM, []),
        ("가슴·등 길항근 슈퍼세트", "chest_back", 60, "hypertrophy", "intermediate", "superset", FULL_GYM, []),
        (
            "전신 서킷: 메인 운동 보호 후 보조 운동만 묶기",
            "full_body",
            45,
            "generalFitness",
            "intermediate",
            "circuit",
            ["MACHINE", "CABLE", "DUMBBELL", "BODYWEIGHT"],
            ["BARBELL"],
        ),
        (
            "상체 근력 밀도형: 메인은 단독, 보조만 밀도형",
            "upper",
            60,
            "strength",
            "advanced",
            "density",
            ["MACHINE", "CABLE", "DUMBBELL", "BARBELL"],
            [],
        ),
    )
    for title, split, minutes, goal, experience, structure, equipment, unavailable in structure_cases:
        _case(
            rows,
            section="6. 세션 진행 방식",
            title=title,
            request=_request(
                split=split,
                time_min=minutes,
                goal=goal,
                experience=experience,
                equipment=equipment,
                unavailable=unavailable,
                session_structure=structure,
            ),
            comparison_group="session_structure",
            comparison_label=title,
            comparison_fixed_conditions="각 방식이 허용되는 대표 헬스장 조건",
        )

    # 7. Adjunct options.
    adjunct_cases = (
        ("메인 헬스장 루틴에 모빌리티 워밍업 추가", "push", 60, ["mobility_warmup"], ["MACHINE", "CABLE", "DUMBBELL", "BAND", "BODYWEIGHT"]),
        ("메인 헬스장 루틴에 컨디셔닝 마무리 추가", "full_body", 60, ["conditioning_finisher"], ["MACHINE", "CABLE", "DUMBBELL", "KETTLEBELL", "BODYWEIGHT"]),
        ("메인 상체 루틴에 프리햅 보조 추가", "upper", 70, ["prehab_accessory"], FULL_GYM),
        ("메인 상체 루틴에 지구력 보조 추가", "upper", 70, ["endurance_support"], FULL_GYM),
        ("메인 상체 루틴에 쿨다운 스트레칭 추가", "upper", 70, ["cooldown_stretch"], FULL_GYM),
        (
            "모빌리티 워밍업과 쿨다운 스트레칭을 함께 선택",
            "upper",
            75,
            ["mobility_warmup", "cooldown_stretch"],
            FULL_GYM,
        ),
    )
    for title, split, minutes, options, equipment in adjunct_cases:
        goal = "generalFitness" if options == ["conditioning_finisher"] else "hypertrophy"
        _case(
            rows,
            section="7. 추가 목적 운동",
            title=title,
            request=_request(
                split=split,
                time_min=minutes,
                goal=goal,
                equipment=equipment,
                unavailable=["BARBELL"] if options == ["conditioning_finisher"] else [],
                adjunct_options=options,
            ),
            comparison_group="adjunct",
            comparison_label=" + ".join(options),
            comparison_fixed_conditions="중급 · 컨디션 보통 · 통증·부상 없음 · 메인 헬스장 루틴 유지",
        )

    # 8. Advanced specialization. Each body-region family has its own balanced
    # baseline so the comparison changes only the local focus.
    specialization_groups = (
        (
            "specialization_chest",
            "chest",
            FULL_GYM,
            (
                (None, "가슴 균형형"),
                ("upperChest", "윗가슴 집중"),
                ("midChest", "가슴 중앙 집중"),
                ("lowerChest", "아랫가슴 집중"),
            ),
        ),
        (
            "specialization_back",
            "pull",
            ["BARBELL", "DUMBBELL", "MACHINE", "CABLE", "BODYWEIGHT"],
            (
                (None, "등 균형형"),
                ("latEmphasis", "광배근 집중"),
                ("midBackEmphasis", "등 중앙·두께 집중"),
                ("trapeziusEmphasis", "승모근 집중"),
            ),
        ),
        (
            "specialization_legs",
            "legs",
            FULL_GYM,
            (
                (None, "하체 균형형"),
                ("quadricepsEmphasis", "대퇴사두 집중"),
                ("hamstringEmphasis", "햄스트링 집중"),
                ("gluteEmphasis", "둔근 집중"),
                ("adductorEmphasis", "내전근 집중"),
                ("abductorEmphasis", "외전근 집중"),
                ("calvesEmphasis", "종아리 집중"),
            ),
        ),
        (
            "specialization_shoulder",
            "shoulder",
            ["DUMBBELL", "MACHINE", "CABLE"],
            (
                (None, "어깨 균형형"),
                ("frontDeltoidEmphasis", "전면 어깨 집중"),
                ("sideDeltoidEmphasis", "측면 어깨 집중"),
                ("rearDeltoidEmphasis", "후면 어깨 집중"),
            ),
        ),
        (
            "specialization_arm",
            "arm",
            ["BARBELL", "DUMBBELL", "MACHINE", "CABLE"],
            (
                (None, "팔 균형형"),
                ("bicepsEmphasis", "이두 집중"),
                ("tricepsEmphasis", "삼두 집중"),
                ("forearmEmphasis", "전완 집중"),
            ),
        ),
    )
    for group, split, equipment, focus_cases in specialization_groups:
        for focus, label in focus_cases:
            title = f"상급자 {label} 근비대"
            _case(
                rows,
                section="8. 상급자 세부 집중",
                title=title,
                request=_request(
                    split=split,
                    time_min=60,
                    experience="advanced",
                    equipment=equipment,
                    methodMode="off",
                    specializationFocus=focus,
                ),
                comparison_group=group,
                comparison_label=label,
                comparison_fixed_conditions=(
                    f"상급 · {_split_label_for_review(split)} · 근비대 · 60분 · "
                    "컨디션 보통 · 통증·부상 없음 · 고급 수행법 끔"
                ),
            )

    # 9. Advanced method auto/manual policy.
    method_equipment = ["BARBELL", "DUMBBELL", "MACHINE", "CABLE"]
    method_cases = (
        ("상급 근력: 자동 정지 반복", "chest", "strength", "normal", "auto", None, [], []),
        ("상급 근비대: 자동 3초 하강", "legs", "hypertrophy", "normal", "auto", None, [], []),
        (
            "최근 같은 분할에서 같은 수행법 사용 후 자동 억제",
            "chest",
            "strength",
            "normal",
            "auto",
            None,
            [],
            [{"methodCode": "pausedReps", "targetSplitLabel": "chest", "sessionsAgo": 1}],
        ),
        (
            "다른 분할의 최근 사용은 자동 수행법 허용",
            "chest",
            "strength",
            "normal",
            "auto",
            None,
            [],
            [{"methodCode": "pausedReps", "targetSplitLabel": "legs", "sessionsAgo": 1}],
        ),
        ("낮은 컨디션에서 자동 수행법 억제", "chest", "strength", "low", "auto", None, [], []),
        (
            "목표 부위 근육통 2단계에서 자동 수행법 억제",
            "chest",
            "strength",
            "normal",
            "auto",
            None,
            [{"bodyPart": "chest", "level": 2}],
            [],
        ),
        ("사용자가 고급 수행법을 끈 경우", "chest", "strength", "normal", "off", None, [], []),
        (
            "근육통 2단계에서 수동 수행법도 억제",
            "chest",
            "strength",
            "normal",
            "manual",
            "pausedReps",
            [{"bodyPart": "chest", "level": 2}],
            [],
        ),
    )
    for title, split, goal, readiness, mode, requested, doms, recent in method_cases:
        _case(
            rows,
            section="9. 상급자 수행법",
            title=title,
            request=_request(
                split=split,
                time_min=60,
                goal=goal,
                experience="advanced",
                readiness=readiness,
                equipment=method_equipment,
                doms=doms,
                methodMode=mode,
                requestedMethod=requested,
                recentTrainingMethodUses=recent,
            ),
            comparison_group="advanced_method",
            comparison_label=title,
            comparison_fixed_conditions="상급 · 60분 · 해당 목표 · 통증·등록 부상 없음",
        )

    if len(rows) != 81:
        raise ValueError(f"expected 81 clean scenarios, got {len(rows)}")
    for row in rows:
        keys = set(row["input"])
        legacy = sorted(keys & FORBIDDEN_LEGACY_KEYS)
        unknown = sorted(keys - CURRENT_PUBLIC_KEYS - CURRENT_PREINTEGRATION_KEYS)
        if legacy or unknown:
            raise ValueError(
                f"{row['scenarioId']} invalid input keys: legacy={legacy}, unknown={unknown}"
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = build_fixture()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    sections: dict[str, int] = {}
    for row in rows:
        section = str(row["reviewSection"])
        sections[section] = sections.get(section, 0) + 1
    print(
        json.dumps(
            {
                "status": "pass",
                "output": str(args.output),
                "scenarioCount": len(rows),
                "sections": sections,
                "sourceFixtureCount": 0,
                "legacyInputKeyCount": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

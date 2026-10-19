#!/usr/bin/env python3
"""Generate the full, non-technical current Routine AI expert review pack."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT
DEFAULT_FIXTURE = (
    ROOT
    / "docs/reviews/fixtures/"
    "routine-ai-core-current-comprehensive-expert-review-v1.jsonl"
)
DEFAULT_RESULT = (
    ROOT
    / "docs/evidence/v3-finalization/current-comprehensive-expert-review-v1/"
    "live/current-comprehensive-expert-review-v1-live-20260730/"
    "gemma4-quality-seed-results-filled.json"
)
DEFAULT_CANDIDATE_REPORT = (
    ROOT
    / "docs/evidence/v3-finalization/current-comprehensive-expert-review-v1/"
    "candidate/current-comprehensive-expert-review-v1-candidate/"
    "candidate-quality-seed-report.json"
)
DEFAULT_CANDIDATE_LIVE_REPORT = (
    ROOT
    / "docs/evidence/v3-finalization/current-comprehensive-expert-review-v1/"
    "candidate-with-live/current-comprehensive-expert-review-v1-candidate-with-live/"
    "candidate-quality-seed-report.json"
)
DEFAULT_REVIEW = (
    PROJECT_ROOT
    / "docs/reviews/"
    "routine-ai-core-current-comprehensive-expert-review-2026-07-30.md"
)
DEFAULT_RESPONSE = (
    PROJECT_ROOT
    / "docs/reviews/"
    "routine-ai-core-current-comprehensive-expert-response-sheet-2026-07-30.md"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "docs/evidence/"
    "routine-ai-core-current-comprehensive-expert-review-manifest-2026-07-30.json"
)


SPLIT_LABELS = {
    "push": "밀기(가슴·어깨·삼두)",
    "pull": "당기기(등·이두)",
    "legs": "하체",
    "upper": "상체",
    "chest_back": "가슴·등",
    "lower": "하체",
    "chest": "가슴",
    "back": "등",
    "shoulder": "어깨",
    "arm": "팔",
    "core": "코어",
    "neck": "목",
    "full_body": "전신",
    "custom": "직접 선택",
}
MUSCLE_LABELS = {
    "abductors": "외전근",
    "abs": "복근",
    "adductor": "내전근",
    "back-deltoids": "후면 어깨",
    "biceps": "이두",
    "calves": "종아리",
    "chest": "가슴",
    "forearm": "전완",
    "front-deltoids": "전면 어깨",
    "side-deltoids": "측면 어깨",
    "gluteal": "둔근",
    "hamstring": "햄스트링",
    "lower-back": "허리·요추",
    "neck": "목",
    "obliques": "복사근",
    "quadriceps": "대퇴사두",
    "trapezius": "승모근",
    "triceps": "삼두",
    "upper-back": "등 상부",
    "head": "머리",
    "knees": "무릎",
    "left-soleus": "왼쪽 가자미근",
    "right-soleus": "오른쪽 가자미근",
}
GOAL_LABELS = {
    "strength": "근력",
    "hypertrophy": "근비대",
    "endurance": "근지구력",
    "fatLoss": "체지방 감소",
    "fatloss": "체지방 감소",
    "recomposition": "체성분 개선",
    "generalFitness": "일반 체력",
    "generalfitness": "일반 체력",
}
EXPERIENCE_LABELS = {
    "beginner": "초급",
    "intermediate": "중급",
    "advanced": "상급",
}
READINESS_LABELS = {
    "low": "낮음",
    "normal": "보통",
    "high": "좋음",
}
EQUIPMENT_LABELS = {
    "BAND": "밴드",
    "BARBELL": "바벨",
    "BODYWEIGHT": "맨몸",
    "CABLE": "케이블",
    "DUMBBELL": "덤벨",
    "KETTLEBELL": "케틀벨",
    "LANDMINE": "랜드마인",
    "MACHINE": "일반 머신",
    "PLATE": "원판",
    "SMITH_MACHINE": "스미스 머신",
}
SESSION_LABELS = {
    "standard": "일반 세트",
    "superset": "슈퍼세트",
    "circuit": "서킷",
    "density": "밀도형 세션",
}
ADJUNCT_LABELS = {
    "mobility_warmup": "모빌리티 워밍업",
    "prehab_accessory": "프리햅 보조",
    "conditioning_finisher": "컨디셔닝 마무리",
    "endurance_support": "지구력 보조",
    "cooldown_stretch": "쿨다운 스트레칭",
}
PROFILE_SIGNAL_LABELS = {
    "heel_down_deep_squat_difficult": "발뒤꿈치를 붙이고 깊게 앉기 어려움",
    "hip_hinge_difficult": "엉덩이를 접어 숙이기 어려움",
    "overhead_reach_difficult": "팔을 머리 위로 올리기 어려움",
    "loaded_wrist_extension_difficult": "손목을 뒤로 젖혀 버티기 어려움",
    "self_reported_long_arms": "팔이 유난히 긴 편",
    "self_reported_long_thighs": "허벅지 구간이 유난히 긴 편",
}
INJURY_AREA_LABELS = {
    "shoulder": "어깨",
    "knee": "무릎",
    "lumbar": "허리·요추",
    "wrist": "손목",
    "ankle": "발목·발",
    "hip": "고관절",
    "elbow": "팔꿈치",
    "neck": "목",
}
INJURY_STATUS_LABELS = {
    "activeUncleared": "활성 부상·운동 미허가",
    "managedWithRestrictions": "관리 중·제한 동작 있음",
    "managedNoRestrictions": "관리 중·현재 운동 제한 없음",
    "recovered": "회복 완료",
}
MOVEMENT_PATTERN_LABELS = {
    "overheadPosition": "팔을 머리 위에 두는 동작",
    "horizontalPush": "수평 밀기",
    "horizontalPull": "수평 당기기",
    "armRaise": "팔 들어 올리기",
    "behindBackRotation": "등 뒤로 팔 돌리기",
    "deepKneeFlexion": "무릎 깊게 굽히기",
    "loadedKneeExtension": "부하를 건 무릎 펴기",
    "loadedKneeFlexion": "부하를 건 무릎 굽히기",
    "singleLegWeightBearing": "한쪽 다리 체중 지지",
    "impactJumpRun": "충격이 있는 점프·달리기",
    "loadedSpinalFlexion": "부하를 건 척추 굽힘",
    "hipHinge": "힙힌지",
    "axialLoading": "척추 수직 압박",
    "unsupportedBentOver": "지지 없는 상체 숙임",
    "trunkRotation": "몸통 회전",
    "loadedWristExtension": "부하가 실린 손목 신전",
    "fixedBarGrip": "고정 바 그립",
    "loadedGrip": "고중량 그립",
    "pronationSupination": "회내·회외",
    "weightBearingOnHands": "손으로 체중 지지",
    "deepDorsiflexion": "발목 깊은 배굴",
    "loadedPlantarFlexion": "부하를 건 발목 저측굴곡",
    "singleLegStability": "한쪽 다리 안정성",
    "deepHipFlexion": "고관절 깊게 굽히기",
    "loadedAbduction": "부하를 건 외전",
    "loadedAdduction": "부하를 건 내전",
    "hipRotation": "고관절 회전",
    "loadedElbowFlexion": "부하를 건 팔꿈치 굽힘",
    "loadedElbowExtension": "부하를 건 팔꿈치 펴기",
    "weightBearingOnArms": "팔로 체중 지지",
    "directNeckLoading": "목 직접 부하",
    "overheadAxialLoading": "머리 위 축성 부하",
    "sustainedUnsupportedHeadPosition": "머리 지지 없는 자세 유지",
    "impactBallistic": "충격·탄도성 동작",
    "unknown": "제한 동작을 아직 모름",
}
FOCUS_LABELS = {
    "upperChest": "윗가슴",
    "midChest": "가슴 중앙",
    "lowerChest": "아랫가슴",
    "latEmphasis": "광배근",
    "midBackEmphasis": "등 중앙·두께",
    "trapeziusEmphasis": "승모근",
    "quadricepsEmphasis": "대퇴사두",
    "hamstringEmphasis": "햄스트링",
    "gluteEmphasis": "둔근",
    "adductorEmphasis": "내전근",
    "abductorEmphasis": "외전근",
    "calvesEmphasis": "종아리",
    "frontDeltoidEmphasis": "전면 어깨",
    "sideDeltoidEmphasis": "측면 어깨",
    "rearDeltoidEmphasis": "후면 어깨",
    "bicepsEmphasis": "이두",
    "tricepsEmphasis": "삼두",
    "forearmEmphasis": "전완",
}
METHOD_LABELS = {
    "pausedReps": "정지 반복",
    "tempoEccentric": "3초 하강",
}
METHOD_MODE_LABELS = {
    "auto": "자동 추천",
    "manual": "직접 선택",
    "off": "사용 안 함",
}
PHASE_LABELS = {
    "warmup_or_preparation": "준비 운동",
    "power_if_explicit": "파워 운동",
    "main_compound": "메인 복합 운동",
    "secondary_compound": "두 번째 복합 운동",
    "isolation_accessory": "고립·보조 운동",
    "prehab_accessory": "프리햅 보조",
    "conditioning_finisher": "컨디셔닝 마무리",
    "endurance_support": "지구력 보조",
    "cooldown_recovery": "쿨다운·회복",
}
GROUP_LABELS = {
    "single": "개별 수행",
    "superset": "슈퍼세트",
    "circuit": "서킷",
    "density": "밀도형 그룹",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label(mapping: dict[str, str], value: Any, default: str = "없음") -> str:
    if value is None or str(value).strip() == "":
        return default
    return mapping.get(str(value), str(value))


def _labels(mapping: dict[str, str], values: Iterable[Any]) -> str:
    rendered = [_label(mapping, value) for value in values]
    return ", ".join(rendered) if rendered else "없음"


def _doms_items(raw: Any) -> list[tuple[str, int]]:
    if isinstance(raw, dict):
        return [(str(key), int(value)) for key, value in raw.items()]
    result: list[tuple[str, int]] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        body_part = item.get("bodyPart") or item.get("body_part") or item.get("area")
        if body_part:
            result.append((str(body_part), int(item.get("level") or 1)))
    return result


def _pain_items(raw: Any) -> list[str]:
    result: list[str] = []
    for item in raw or []:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            value = item.get("bodyPart") or item.get("body_part") or item.get("area")
            if value:
                result.append(str(value))
    return result


def _working_rows(block: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in block.get("prescription") or []
        if str(row.get("setType") or "working") == "working"
    ]


def _working_set_total(result: dict[str, Any]) -> int:
    return sum(len(_working_rows(block)) for block in result.get("blockSummaries") or [])


def _direct_set_total(result: dict[str, Any], muscle: str) -> int:
    return sum(
        len(_working_rows(block))
        for block in result.get("blockSummaries") or []
        if muscle in (block.get("primaryMuscles") or [])
    )


def _direct_set_distribution(result: dict[str, Any]) -> str:
    totals: dict[str, int] = {}
    for block in result.get("blockSummaries") or []:
        set_count = len(_working_rows(block))
        for muscle in block.get("primaryMuscles") or []:
            totals[str(muscle)] = totals.get(str(muscle), 0) + set_count
    if not totals:
        return "직접 작업 세트 없음"
    ordered = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(
        f"{_label(MUSCLE_LABELS, muscle)} {set_count}세트"
        for muscle, set_count in ordered
    )


def _exercise_names(result: dict[str, Any]) -> str:
    names = [
        str(block.get("exerciseName") or "").strip()
        for block in result.get("blockSummaries") or []
        if str(block.get("exerciseName") or "").strip()
    ]
    return ", ".join(names) if names else "추천 운동 없음"


def _numeric_range(values: Iterable[Any], suffix: str) -> str:
    numbers = sorted(
        {
            int(value)
            for value in values
            if value is not None and str(value).strip() != ""
        }
    )
    if not numbers:
        return "해당 없음"
    if len(numbers) == 1:
        return f"{numbers[0]}{suffix}"
    return f"{numbers[0]}~{numbers[-1]}{suffix}"


def _prescription_summary(result: dict[str, Any]) -> str:
    rows = [
        row
        for block in result.get("blockSummaries") or []
        for row in _working_rows(block)
    ]
    if not rows:
        return "처방 없음"
    repetitions = _numeric_range(
        (row.get("targetReps") for row in rows),
        "회",
    )
    durations = _numeric_range(
        (row.get("targetDurationSec") for row in rows),
        "초",
    )
    rir = _numeric_range((row.get("targetRir") for row in rows), "")
    rests = _numeric_range((row.get("targetRestSec") for row in rows), "초")
    load = durations if repetitions == "해당 없음" else repetitions
    return f"{load} · RIR {rir} · 휴식 {rests}"


def _execution_summary(result: dict[str, Any]) -> str:
    grouped: dict[str, dict[str, Any]] = {}
    for block in result.get("blockSummaries") or []:
        group_type = str(block.get("executionGroupType") or "single")
        if group_type == "single":
            continue
        group_id = str(block.get("executionGroupId") or "?")
        key = f"{group_type}:{group_id}"
        grouped.setdefault(
            key,
            {
                "type": group_type,
                "names": [],
                "rest": block.get("groupRestSec"),
            },
        )
        grouped[key]["names"].append(str(block.get("exerciseName") or "이름 없음"))
    if not grouped:
        return "모든 운동 단독 수행"
    parts = []
    for group in grouped.values():
        label = _label(GROUP_LABELS, group["type"])
        rest = (
            f", 묶음 뒤 {group['rest']}초 휴식"
            if group.get("rest") is not None
            else ""
        )
        parts.append(f"{label}: {' + '.join(group['names'])}{rest}")
    return "; ".join(parts)


def _method_summary(result: dict[str, Any]) -> str:
    methods = []
    for block in result.get("blockSummaries") or []:
        method = block.get("trainingMethod")
        if not method:
            continue
        label = method.get("labelKo") or _label(METHOD_LABELS, method.get("code"))
        methods.append(f"{block.get('exerciseName')}: {label}")
    return "; ".join(methods) if methods else "적용 안 됨"


def _adjunct_summary(result: dict[str, Any]) -> str:
    adjunct_blocks = [
        block
        for block in result.get("blockSummaries") or []
        if block.get("adjunctOption")
    ]
    if not adjunct_blocks:
        return "추가 운동 없음"
    return "; ".join(
        f"{_label(ADJUNCT_LABELS, block.get('adjunctOption'))}: "
        f"{block.get('exerciseName') or '이름 없음'}"
        for block in adjunct_blocks
    )


def _format_duration(seconds: Any) -> str:
    try:
        value = int(seconds)
    except (TypeError, ValueError):
        return ""
    if value % 60 == 0:
        return f"{value // 60}분"
    return f"{value}초"


def _format_prescription(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "처방 정보 없음"
    first = rows[0]
    if first.get("intervalRounds") is not None:
        return (
            f"{first['intervalRounds']}라운드, 운동 {first.get('intervalWorkSec', '?')}초, "
            f"회복 {first.get('intervalRecoverySec', '?')}초"
        )
    if first.get("targetDurationSec") is not None and first.get("targetReps") is None:
        return f"{len(rows)}세트, 세트당 {_format_duration(first['targetDurationSec'])}"
    same = all(
        row.get("targetReps") == first.get("targetReps")
        and row.get("targetRir") == first.get("targetRir")
        and row.get("targetRestSec") == first.get("targetRestSec")
        and row.get("targetDurationSec") == first.get("targetDurationSec")
        for row in rows
    )
    if same:
        bits = [f"{len(rows)}세트"]
        if first.get("targetReps") is not None:
            bits.append(f"{first['targetReps']}회")
        if first.get("targetDurationSec") is not None:
            bits.append(_format_duration(first["targetDurationSec"]))
        if first.get("targetRir") is not None:
            bits.append(f"RIR {first['targetRir']}")
        if first.get("targetRestSec") is not None:
            bits.append(f"휴식 {first['targetRestSec']}초")
        return " · ".join(bits)
    output: list[str] = []
    for row in rows:
        bits = [f"{row.get('setIndex', '?')}세트"]
        if row.get("targetReps") is not None:
            bits.append(f"{row['targetReps']}회")
        if row.get("targetDurationSec") is not None:
            bits.append(_format_duration(row["targetDurationSec"]))
        if row.get("targetRir") is not None:
            bits.append(f"RIR {row['targetRir']}")
        if row.get("targetRestSec") is not None:
            bits.append(f"휴식 {row['targetRestSec']}초")
        output.append(" · ".join(bits))
    return "; ".join(output)


def _render_input(request: dict[str, Any]) -> list[str]:
    target_muscles = request.get("targetMuscles") or []
    lines = [
        f"- 운동 부위: {_label(SPLIT_LABELS, request.get('targetSplitLabel'))}",
    ]
    if target_muscles:
        lines.append(f"- 직접 지정 근육: {_labels(MUSCLE_LABELS, target_muscles)}")
    lines.extend(
        [
            f"- 운동 가능 시간: {request.get('timeAvailableMin')}분",
            f"- 운동 목표: {_label(GOAL_LABELS, request.get('goal'))}",
            f"- 운동 경력: {_label(EXPERIENCE_LABELS, request.get('experienceLevel'))}",
            f"- 오늘 컨디션: {_label(READINESS_LABELS, request.get('readinessLevel') or 'normal')}",
            f"- 세션 진행 방식: {_label(SESSION_LABELS, request.get('sessionStructure') or 'standard')}",
            f"- 사용 가능 장비: {_labels(EQUIPMENT_LABELS, request.get('availableEquipment') or [])}",
            f"- 제외 장비: {_labels(EQUIPMENT_LABELS, request.get('unavailableEquipment') or [])}",
        ]
    )
    pain = _pain_items(request.get("currentPainAreas") or request.get("painAreas"))
    lines.append(f"- 등록 부상: {_labels(MUSCLE_LABELS, pain)}")
    doms = _doms_items(
        request.get("currentDoms")
        or request.get("domsData")
        or request.get("doms")
    )
    lines.append(
        "- 근육통: "
        + (
            ", ".join(
                f"{_label(MUSCLE_LABELS, body_part)} {level}단계"
                for body_part, level in doms
            )
            if doms
            else "없음"
        )
    )
    profile_signals = request.get("profileSignals") or []
    if profile_signals:
        lines.append(
            f"- 동작·체형 특징: {_labels(PROFILE_SIGNAL_LABELS, profile_signals)}"
        )
    conditions = request.get("injuryConditions") or []
    if conditions:
        rendered_conditions = []
        for condition in conditions:
            area = _label(INJURY_AREA_LABELS, condition.get("areaCode"))
            status = _label(INJURY_STATUS_LABELS, condition.get("status"))
            patterns = _labels(
                MOVEMENT_PATTERN_LABELS,
                condition.get("restrictedMovementPatterns") or [],
            )
            rendered_conditions.append(f"{area} / {status} / 제한: {patterns}")
        lines.append(f"- 관리 중 부상 상태: {'; '.join(rendered_conditions)}")
    adjuncts = request.get("adjunctOptions") or []
    if adjuncts:
        lines.append(f"- 추가 목적 운동: {_labels(ADJUNCT_LABELS, adjuncts)}")
    focus = request.get("specializationFocus")
    if focus:
        lines.append(f"- 상급자 세부 집중: {_label(FOCUS_LABELS, focus)}")
    method_mode = request.get("methodMode")
    if method_mode:
        lines.append(f"- 상급자 수행법 설정: {_label(METHOD_MODE_LABELS, method_mode)}")
    requested_method = request.get("requestedMethod") or request.get("trainingMethod")
    if requested_method:
        lines.append(f"- 직접 선택한 수행법: {_label(METHOD_LABELS, requested_method)}")
    recent_method_uses = request.get("recentTrainingMethodUses") or []
    if recent_method_uses:
        rendered = []
        for item in recent_method_uses:
            method = _label(METHOD_LABELS, item.get("methodCode"))
            split = _label(SPLIT_LABELS, item.get("targetSplitLabel"))
            rendered.append(f"{split}에서 {method}, {item.get('sessionsAgo')}회 전")
        lines.append(f"- 최근 고급 수행법 이력: {'; '.join(rendered)}")
    return lines


def _render_exercises(result: dict[str, Any]) -> list[str]:
    blocks = result.get("blockSummaries") or []
    if not blocks:
        return ["- 추천 운동 없음"]
    lines: list[str] = []
    for block in blocks:
        name = str(block.get("exerciseName") or "운동 이름 없음")
        phase = _label(PHASE_LABELS, block.get("sessionPhase"), "단계 정보 없음")
        muscles = _labels(MUSCLE_LABELS, block.get("primaryMuscles") or [])
        equipment = _label(EQUIPMENT_LABELS, block.get("equipmentType"), "장비 정보 없음")
        prescription = _format_prescription(block.get("prescription") or [])
        group_type = str(block.get("executionGroupType") or "single")
        group_note = ""
        if group_type != "single":
            group_note = (
                f" · {_label(GROUP_LABELS, group_type)} "
                f"{block.get('executionGroupId') or ''}"
            )
            if block.get("groupRestSec") is not None:
                group_note += f" · 묶음 뒤 휴식 {block['groupRestSec']}초"
        method = block.get("trainingMethod")
        method_note = ""
        if method:
            method_note = f" · **{method.get('labelKo') or _label(METHOD_LABELS, method.get('code'))}**"
        lines.append(
            f"{block.get('order')}. **{name}** [{phase}] "
            f"({muscles}, {equipment}) - {prescription}{group_note}{method_note}"
        )
        if method and method.get("instructionKo"):
            lines.append(f"   - 수행 안내: {method['instructionKo']}")
    return lines


def _render_rationale(text: Any) -> list[str]:
    parts = [part.strip() for part in str(text or "").split("|") if part.strip()]
    return [f"> {part}" for part in parts] if parts else ["> 별도 설명 없음"]


def _outcome_label(result: dict[str, Any]) -> str:
    status = result.get("generationStatus")
    if status == "success":
        return "정상 추천"
    if status == "fallback":
        return "대체 경로로 추천"
    if status == "failed" and result.get("statusReasonCode") == "emptyCandidate":
        return "안전 후보가 없어 추천 중단"
    return f"추천 중단({result.get('statusReasonCode') or '이유 미상'})"


def _render_case(
    fixture: dict[str, Any],
    result: dict[str, Any],
) -> list[str]:
    requested = int(fixture.get("input", {}).get("timeAvailableMin") or 0)
    estimated = int(result.get("totalEstimatedTime") or 0)
    time_note = ""
    if requested and estimated:
        gap = requested - estimated
        if gap >= 10:
            time_note = f" (요청보다 {gap}분 짧음)"
        elif gap < 0:
            time_note = f" (요청보다 {-gap}분 김)"
    lines = [
        f"#### 사례 {fixture['reviewOrder']}. {fixture['reviewTitle']}",
        "",
        "**사용자가 고른 조건**",
        "",
        *_render_input(fixture["input"]),
        "",
        "**결과 요약**",
        "",
        f"- 판정: {_outcome_label(result)}",
        f"- 운동 수: {result.get('exerciseCount', 0)}개",
        f"- 작업 세트: {_working_set_total(result)}세트",
        f"- 예상 시간: {estimated}분{time_note}",
        "",
        "**실제 추천 운동**",
        "",
        *_render_exercises(result),
        "",
        "**사용자에게 보이는 추천 이유**",
        "",
        *_render_rationale(result.get("rationaleText")),
        "",
        "**사용자에게 보이는 주의 안내**",
        "",
    ]
    warnings = result.get("warnings") or []
    lines.extend(f"- {warning}" for warning in warnings)
    if not warnings:
        lines.append("- 별도 주의 안내 없음")
    suggestions = result.get("personalizationSuggestions") or []
    if suggestions:
        lines.extend(["", "**개인화 제안**", ""])
        for suggestion in suggestions:
            if isinstance(suggestion, dict):
                text = (
                    suggestion.get("messageKo")
                    or suggestion.get("message")
                    or suggestion.get("reasonKo")
                    or json.dumps(suggestion, ensure_ascii=False)
                )
            else:
                text = str(suggestion)
            lines.append(f"- {text}")
    lines.extend(["", "**코치님께 확인받고 싶은 점**", ""])
    lines.extend(f"- {item}" for item in fixture.get("reviewFocus") or [])
    lines.extend(
        [
            "",
            "평가: `적절 / 조건부 적절 / 수정 필요 / 판단하기 어려움`",
            "",
            "의견:",
            "",
            "---",
            "",
        ]
    )
    return lines


def _comparison_table(
    title: str,
    fixtures: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    *,
    extra: callable | None = None,
    detail_header: str | None = None,
    detail: callable | None = None,
) -> list[str]:
    fixed_conditions = list(
        dict.fromkeys(
            str(fixture.get("comparisonFixedConditions") or "").strip()
            for fixture in fixtures
            if str(fixture.get("comparisonFixedConditions") or "").strip()
        )
    )
    lines = [f"### {title}", ""]
    if fixed_conditions:
        if len(fixed_conditions) == 1:
            rendered_fixed = fixed_conditions[0]
        else:
            token_lists = [
                [token.strip() for token in value.split(" · ") if token.strip()]
                for value in fixed_conditions
            ]
            common_tokens = [
                token
                for token in token_lists[0]
                if all(token in tokens for tokens in token_lists[1:])
            ]
            rendered_fixed = (
                " · ".join(common_tokens)
                if common_tokens
                else "행마다 달라지는 세부 조건은 첫 열과 전체 사례에 명시"
            )
        lines.append(f"- 고정 조건: {rendered_fixed}")
        lines.append("- 아래 표의 첫 열만 바꾸고 현재 AI 코어에 새로 입력했습니다.")
        lines.append("")
    header = "| 변경한 조건 | 운동 수 | 작업 세트 | 예상 시간 |"
    divider = "|---|---:|---:|---:|"
    if detail_header:
        header += f" {detail_header} |"
        divider += "---|"
    header += " 실제 추천 운동 |"
    divider += "---|"
    lines.extend([header, divider])
    for fixture in fixtures:
        result = results[fixture["scenarioId"]]
        label = fixture.get("comparisonLabel") or fixture["reviewTitle"]
        if extra is not None:
            label = f"{label}{extra(fixture, result)}"
        row = (
            f"| {label} | {result.get('exerciseCount', 0)}개 | "
            f"{_working_set_total(result)}세트 | "
            f"{result.get('totalEstimatedTime', 0)}분 |"
        )
        if detail_header and detail is not None:
            row += f" {detail(fixture, result)} |"
        row += f" {_exercise_names(result)} |"
        lines.append(row)
    lines.append("")
    return lines


def _render_input_dictionary() -> list[str]:
    lines = [
        "## 1. 이 AI가 받는 입력",
        "",
        "아래 표는 전문가가 각 사례의 조건을 해석하기 위한 사용자 관점 설명입니다. "
        "영문 변수명이나 개발 구조를 알 필요는 없습니다.",
        "",
        "`공개 입력`은 현재 AI 입력 계약이 받을 수 있는 값이라는 뜻입니다. "
        "실제 배포 웹 화면에서의 배치와 연결 완료 여부는 별도 FE 검수 대상입니다. "
        "`코어 준비 완료·화면 연결 전`은 이번 AI 검증에는 포함됐지만 아직 사용자 "
        "화면에서 선택할 수 없는 최신 입력입니다.",
        "",
        "| 입력 묶음 | 사용자가 선택하거나 제공하는 내용 | 현재 상태 | 루틴에 미치는 영향 |",
        "|---|---|---|---|",
        "| 운동 부위 | 밀기, 당기기, 하체, 상체, 가슴·등, 가슴, 등, 어깨, 팔, 코어, 목, 전신, 직접 선택 | 공개 입력 | 메인 근육과 필수 운동 구성을 정함 |",
        "| 운동 시간 | 10~180분 | 공개 입력 | 운동 수·세트 수·보조 목적 수용 범위를 정함 |",
        "| 운동 목표 | 근력, 근비대, 근지구력, 체지방 감소, 체성분 개선, 일반 체력 | 공개 입력 | 반복 수·세트·RIR·휴식과 운동 효과 우선순위를 바꿈 |",
        "| 운동 경력 | 초급, 중급, 상급 | 공개 입력 | 운동 난도·익숙함·자유중량 허용 범위와 처방 용량을 바꿈 |",
        "| 오늘 컨디션 | 낮음, 보통, 좋음 | 공개 입력 | 낮음은 강도·볼륨을 보수화하고 고밀도 옵션을 제한함 |",
        "| 장비 | 바벨, 덤벨, 일반 머신, 스미스 머신, 케이블, 밴드, 맨몸, 케틀벨, 랜드마인, 원판 | 공개 입력 | 허용 장비 밖 운동을 제외함. 일반 머신은 스미스 머신을 포함하지 않음 |",
        "| 근육통 | 근육별 1·2·3단계 | 공개 입력 | 1단계는 소폭, 2단계는 강하게 감량, 3단계는 해당 근육 직접 운동을 제거함 |",
        "| 등록 부상 | 부위만 선택하며 강도 선택 없음 | 공개 입력 | 관련 운동을 안전 후보에서 제외함 |",
        "| 동작 수행 어려움 | 깊게 앉기, 힙힌지, 머리 위 팔 올리기, 손목 신전 지지 | 공개 입력 | 통증이 없다는 전제에서 요구도가 높은 운동을 낮추고 대체 후보를 올림 |",
        "| 체형 특징 | 팔이 유난히 긺, 허벅지 구간이 유난히 긺 | 공개 입력 | 운동 금지가 아니라 세팅·가동범위 설명과 후보 순위를 약하게 조정함 |",
        "| 세션 방식 | 일반 세트, 슈퍼세트, 서킷, 밀도형 | 공개 입력 | 메인 운동은 보호하고 호환되는 운동만 제한적으로 묶음 |",
        "| 추가 목적 | 모빌리티 워밍업, 프리햅, 컨디셔닝, 지구력, 쿨다운 | 공개 입력 | 헬스장 메인 루틴을 유지한 채 최대 2개 보조 슬롯을 추가함 |",
        "| 관리 중 부상 | 8개 부위의 상태와 이미 확인된 제한 동작 | 코어 준비 완료·화면 연결 전 | 전부 금지 또는 제한 동작만 제외하는 보수적 제어에 사용 |",
        "| 상급자 세부 집중 | 가슴·등·하체·어깨·팔의 18개 세부 부위 | 코어 준비 완료·화면 연결 전 | 균형 루틴 안에서 직접 세트 1세트를 재배분함 |",
        "| 상급자 수행법 | 자동, 직접 선택, 사용 안 함; 정지 반복, 3초 하강 | 코어 준비 완료·화면 연결 전 | 한 개의 익숙한 운동에만 수행법을 붙이며 안전 신호가 있으면 억제함 |",
        "| 운동 이력 | 최근 세트, 중량 기준, 최근 수행법 사용 | 프로필·이력에서 자동 | 중량 참고, 반복 수행 방지, 개인화 설명에 사용함 |",
        "| 자유 메모 | 짧은 선호 설명 | 보조 입력 | 추천 이유 문맥에만 쓰며 부상·안전·운동 자격을 바꾸지 못함 |",
        "",
        "### 운동 목표가 뜻하는 것",
        "",
        "- **근력**: 큰 중량을 안정적으로 다루는 능력을 높이는 목표입니다. 복합 운동, 낮은 반복 수, 긴 휴식을 우선합니다.",
        "- **근비대**: 근육 성장에 필요한 직접 작업량과 자극을 확보하는 목표입니다.",
        "- **근지구력**: 비교적 가벼운 부하를 더 오래 반복하는 능력을 높이는 목표입니다.",
        "- **체지방 감소**: 저항 운동만으로 감량을 보장하지 않습니다. 근육을 보존하면서 세션 밀도와 활동량을 높이는 보조 목표로 사용합니다.",
        "- **체성분 개선**: 체지방을 줄이면서 근육량을 유지하거나 늘리는 장기 목표입니다. 종목은 근비대와 겹칠 수 있지만 휴식·반복·중량 기준을 조금 더 혼합형으로 조정합니다.",
        "- **일반 체력**: 특정 기록이나 근육 성장에 치우치지 않고 기본 근력·근지구력·동작 경험을 균형 있게 유지하는 목표입니다. 중간 반복 수와 중간 휴식을 사용합니다.",
        "",
        "운동 목표가 달라도 안전하고 효율적인 기본 운동 종목은 같을 수 있습니다. "
        "특히 체성분 개선·일반 체력은 종목 이름보다 반복 수, 휴식, 세트 밀도와 "
        "중량 기준에서 차이가 나타나는 것이 정상입니다.",
        "",
        "### 운동 부위 직접 선택",
        "",
        "프리셋을 고른 뒤 사용자가 실제 운동 근육을 편집할 수 있습니다. "
        "직접 선택 가능한 근육은 다음과 같습니다.",
        "",
        "- 가슴, 등 상부, 승모근, 전면·후면 어깨",
        "- 이두, 삼두, 전완",
        "- 대퇴사두, 햄스트링, 둔근, 내전근, 외전근, 종아리",
        "- 복근, 복사근, 허리·요추, 목",
        "",
        "예를 들어 가슴 근육통 3단계이더라도 사용자가 어깨와 삼두만 직접 "
        "선택하면 가슴 운동 없이 그 범위의 루틴을 만들 수 있습니다. 반대로 "
        "가슴을 포함한 전체 밀기를 선택했다면 가슴 메인을 다른 부위 운동으로 "
        "위장하지 않고 중단합니다.",
        "",
        "### 등록 부상과 관리 중 부상의 차이",
        "",
        "- **등록 부상**: 현재 운동해서는 안 되는 부위를 단순하게 표시합니다. 강도는 받지 않으며 관련 운동을 제외합니다.",
        "- **관리 중 부상**: 이미 상태와 제한 동작을 알고 있는 경우에만 씁니다. 어깨, 무릎, 허리·요추, 손목, 발목·발, 고관절, 팔꿈치, 목을 지원합니다.",
        "- 상태는 `활성 부상·운동 미허가`, `관리 중·제한 동작 있음`, `관리 중·현재 제한 없음`, `회복 완료`로 구분합니다.",
        "- 제한 동작을 아직 모르면 같은 부위 부하를 넓게 허용하지 않습니다.",
        "- **회복 완료는 루틴 화면의 지속 선택값이 아닙니다.** 프로필 이력에는 회복 전환을 남기되, 루틴 생성 시에는 현재 제한이 없는 일반 상태로 정규화합니다.",
        "- 이 입력은 진단이나 재활 처방이 아니라, 이미 확인된 제한을 루틴이 어기지 않게 하는 장치입니다.",
        "",
        "### 추가 목적 운동의 선택 방식",
        "",
        "추가 목적은 고정 운동 하나를 무조건 붙이는 기능이 아닙니다. 사용자가 목적을 "
        "선택하면 서버가 메인 루틴을 먼저 보존하고, 목적별로 검수된 후보군에서 현재 "
        "운동 부위·장비·부상·시간 조건을 통과한 운동 하나를 준비·보조·마무리 슬롯에 "
        "배치합니다.",
        "",
        "- 모빌리티 워밍업: 검수 후보 8개",
        "- 프리햅 보조: 검수 후보 8개",
        "- 컨디셔닝 마무리: 검수 후보 4개",
        "- 지구력 보조: 검수 후보 6개",
        "- 쿨다운 스트레칭: 검수 후보 7개",
        "",
        "목적 슬롯을 만들지 여부, 배치 단계, 안전·시간 검사는 결정론적 규칙이 "
        "소유합니다. 작은 언어 모델은 통과한 슬롯 후보 안에서만 운동을 선택하며, "
        "누락하면 서버가 같은 후보 묶음 안에서 복구합니다.",
        "",
        "### 상급자 세부 집중 선택",
        "",
        "- 가슴: 윗가슴, 가슴 중앙, 아랫가슴",
        "- 등: 광배근, 등 중앙·두께, 승모근",
        "- 하체: 대퇴사두, 햄스트링, 둔근, 내전근, 외전근, 종아리",
        "- 어깨·팔: 전면·측면·후면 어깨, 이두, 삼두, 전완",
        "",
        "지원되는 세부 집중은 총 18가지입니다. 뒤의 비교표는 가슴·등·하체·어깨·팔 "
        "각 그룹에서 균형형 기준선과 모든 세부 집중을 각각 다시 실행해 보여줍니다.",
        "",
        "집중을 고르지 않으면 균형형으로 생성합니다. 집중을 골라도 전체 세션 "
        "볼륨을 단순 증가시키지 않고, 집중 부위 직접 세트를 한 세트 늘리는 대신 "
        "낮은 우선순위 세트를 줄이는 방식이 기본입니다.",
        "",
        "### RIR 읽는 법",
        "",
        "`RIR 3`은 해당 세트를 끝냈을 때 약 3회를 더 할 여유, `RIR 2`는 "
        "약 2회를 더 할 여유가 있다는 뜻입니다. 이 문서에서는 사용자가 "
        "처방 강도를 판단할 수 있도록 RIR을 그대로 표시합니다.",
        "",
    ]
    return lines


def _render_priority_rules() -> list[str]:
    return [
        "## 2. 입력이 겹칠 때의 우선순위",
        "",
        "AI가 여러 조건을 동시에 받으면 아래 순서로 판단합니다.",
        "",
        "1. 등록 부상과 운동 미허가 상태",
        "2. 사용 불가 장비와 제한 동작",
        "3. 근육통 3단계와 기타 안전 제한",
        "4. 사용자가 선택한 운동 부위와 반드시 필요한 메인 운동",
        "5. 운동 경력과 오늘 컨디션",
        "6. 목표에 따른 효과·반복·세트·휴식",
        "7. 세션 방식과 추가 목적 운동",
        "8. 상급자 세부 집중과 수행법",
        "9. 자유 메모와 선호",
        "",
        "즉 효과가 높거나 상급자에게 좋은 운동이어도 부상·장비·동작 제한을 "
        "넘을 수 없습니다. 안전 후보가 없으면 억지로 비슷한 운동을 고르지 않고 "
        "추천을 중단합니다.",
        "",
        "### 입력 단계에서 막는 조합",
        "",
        "- 오늘 컨디션이 낮은데 서킷 또는 밀도형 세션을 선택",
        "- 오늘 컨디션이 낮은데 컨디셔닝 마무리 또는 지구력 보조를 선택",
        "- 근력 목표와 서킷을 함께 선택",
        "- 컨디셔닝 마무리와 지구력 보조를 함께 선택",
        "- 추가 목적 운동을 3개 이상 선택",
        "- 같은 장비를 사용 가능과 사용 불가에 동시에 선택",
        "- 상급자가 아닌데 세부 집중이나 고급 수행법을 직접 선택",
        "- 수행법 사용 안 함과 특정 수행법을 동시에 선택",
        "- 정지 반복을 근력 외 목표에 직접 선택하거나 3초 하강을 근비대·체성분 개선 외 목표에 직접 선택",
        "- 같은 부위를 등록 부상과 근육통으로 동시에 입력",
        "- 등록 부상에 약함·보통·심함 같은 강도를 입력",
        "",
        "### 특히 주의해서 볼 조합",
        "",
        "- **근육통과 운동 부위**: 근육통 1·2단계는 운동 금지가 아니라 직접 세트와 강도를 줄이는 신호입니다. 3단계에서 해당 근육이 루틴의 필수 메인이면 다른 근육으로 정상 루틴처럼 꾸미지 않고 중단합니다.",
        "- **등록 부상과 직접 근육 선택**: 부상 부위가 포함된 프리셋은 중단될 수 있지만, 부상과 겹치지 않는 근육만 사용자가 직접 선택하면 그 범위 안에서 다시 구성할 수 있습니다.",
        "- **관리 중 부상과 제한 동작**: 운동 허가가 없으면 해당 부위 부하를 넓게 제외합니다. 이미 확인된 제한 동작만 있는 경우에는 그 동작만 제외하고 나머지 목표를 최대한 보존합니다.",
        "- **동작 수행 어려움과 부상**: 깊게 앉기나 머리 위 팔 올리기 어려움은 통증·부상 진단이 아니라 대체 운동과 세팅 설명을 위한 신호입니다. 등록 부상보다 약한 우선순위로 처리합니다.",
        "- **컨디션과 고밀도 옵션**: 컨디션이 낮으면 서킷·밀도형·컨디셔닝·지구력 보조를 받지 않으며, 일반 세트에서도 작업 세트와 강도를 보수화합니다.",
        "- **세션 방식과 장비 이동**: 슈퍼세트·서킷·밀도형은 모든 운동을 묶지 않습니다. 메인 운동은 단독으로 보호하고, 같은 장비 또는 프리웨이트처럼 현장에서 이어 하기 쉬운 보조 운동만 제한적으로 묶습니다.",
        "- **상급자 세부 집중과 수행법**: 세부 집중은 한 세트 재배분이고 수행법은 한 운동에만 적용합니다. 낮은 컨디션, 목표 부위 근육통 2단계 이상, 부상·관리 제한, 최근 같은 분할에서 같은 수행법 사용이 있으면 자동 적용을 억제합니다.",
        "- **추가 목적 두 가지**: 서로 충돌하지 않으면 최대 두 가지를 함께 쓸 수 있습니다. 준비 운동은 앞, 프리햅은 보조 구간, 컨디셔닝·지구력·쿨다운은 뒤에 배치하며 메인 헬스장 루틴을 대체하지 않습니다.",
        "",
        "### 헬스장 루틴이라는 제품 경계",
        "",
        "이 코어는 헬스장 근력 운동이 중심입니다. 요가, 크로스핏, 재활, 독립 "
        "스트레칭 루틴을 만들지 않습니다. 검수된 관련 동작은 사용자가 추가 "
        "목적을 고른 경우에만 준비·보조·마무리 한 슬롯으로 들어갈 수 있습니다.",
        "",
    ]


def _render_comparisons(
    fixtures: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
) -> list[str]:
    def group(name: str) -> list[dict[str, Any]]:
        return sorted(
            [
                fixture
                for fixture in fixtures
                if fixture.get("comparisonGroup") == name
            ],
            key=lambda fixture: int(fixture["reviewOrder"]),
        )

    lines = [
        "## 3. 먼저 보는 핵심 비교표",
        "",
        "아래 비교는 같은 조건에서 한 항목만 바꾸거나, 같은 정책의 상태 단계만 "
        "바꾼 결과입니다. 세부 추천은 뒤의 전체 사례에서 볼 수 있습니다.",
        "",
    ]
    lines.extend(
        _comparison_table(
            "운동 경력만 초급·중급·상급으로 변경",
            group("experience"),
            results,
            detail_header="대표 처방 범위",
            detail=lambda _fixture, result: _prescription_summary(result),
        )
    )
    lines.extend(
        _comparison_table(
            "운동 시간과 장비 경계 조건",
            group("time_equipment"),
            results,
            detail_header="대표 처방 범위",
            detail=lambda _fixture, result: _prescription_summary(result),
        )
    )
    lines.extend(
        _comparison_table(
            "운동 목표만 6가지로 변경",
            group("goal"),
            results,
            detail_header="대표 처방 범위",
            detail=lambda _fixture, result: _prescription_summary(result),
        )
    )
    lines.extend(
        _comparison_table(
            "초급 전신 루틴에서 오늘 컨디션만 낮음·보통·좋음으로 변경",
            group("readiness"),
            results,
            detail_header="대표 처방 범위",
            detail=lambda _fixture, result: _prescription_summary(result),
        )
    )
    lines.extend(
        [
            "중급 근비대의 기본 처방은 이미 RIR 2이므로 `보통`과 `좋음`이 같아질 수 "
            "있습니다. 비교표는 컨디션 변화가 실제 처방 차이로 드러나는 초급 기준으로 "
            "바꿨습니다. `좋음`은 무조건 운동 수나 볼륨을 늘리라는 신호가 아니라, "
            "보수적인 기본 강도를 안전 범위 안에서 한 단계 조정할 수 있다는 신호입니다.",
            "",
        ]
    )
    doms_fixtures = group("doms")
    lines.extend(
        [
            "### 가슴 근육통 단계별 밀기 루틴",
            "",
            f"- 고정 조건: {doms_fixtures[0].get('comparisonFixedConditions')}",
            "- 아래 표에서는 가슴 근육통 단계만 바꾸고 현재 AI 코어에 새로 입력했습니다.",
            "",
            "| 변경한 조건 | 직접 가슴 세트 | 다른 부위 세트 | 전체 세트 | 예상 시간 | 결과 |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for fixture in doms_fixtures:
        result = results[fixture["scenarioId"]]
        direct = _direct_set_total(result, "chest")
        total = _working_set_total(result)
        doms = _doms_items(fixture["input"].get("currentDoms"))
        label = "없음" if not doms else f"{doms[0][1]}단계"
        lines.append(
            f"| {label} | {direct}세트 | {total - direct}세트 | {total}세트 | "
            f"{result.get('totalEstimatedTime', 0)}분 | {_outcome_label(result)} |"
        )
    lines.append("")
    lines.extend(
        _comparison_table(
            "등록 부상과 직접 선택 범위의 경계",
            group("injury_boundary"),
            results,
        )
    )
    lines.extend(
        _comparison_table(
            "관리 중 부위별 제한 동작",
            group("managed_restriction"),
            results,
        )
    )
    lines.extend(
        _comparison_table(
            "발목 상태 기록의 단계별 차이",
            group("ankle_status"),
            results,
        )
    )
    lines.extend(
        [
            "`회복 완료`는 이력 전환과 잔여 제한 제거를 확인하기 위한 회귀 사례입니다. "
            "실제 루틴 생성 요청에서는 회복 이력을 현재 제약으로 전달하지 않으며, "
            "`입력 없음`과 같은 일반 상태로 처리합니다.",
            "",
        ]
    )
    lines.extend(
        _comparison_table(
            "세션 진행 방식 4가지",
            group("session_structure"),
            results,
            detail_header="실제 묶음 방식",
            detail=lambda _fixture, result: _execution_summary(result),
        )
    )
    lines.extend(
        _comparison_table(
            "동작 수행 어려움과 체형 특징의 반영 결과",
            group("profile_signal"),
            results,
            detail_header="대표 처방 범위",
            detail=lambda _fixture, result: _prescription_summary(result),
        )
    )
    lines.extend(
        _comparison_table(
            "추가 목적 운동이 메인 루틴에 붙는 방식",
            group("adjunct"),
            results,
            detail_header="추가된 준비·보조·마무리",
            detail=lambda _fixture, result: _adjunct_summary(result),
        )
    )
    for group_name, title in (
        ("specialization_chest", "상급자 가슴 균형형과 세부 집중 3가지"),
        ("specialization_back", "상급자 등 균형형과 세부 집중 3가지"),
        ("specialization_legs", "상급자 하체 균형형과 세부 집중 6가지"),
        ("specialization_shoulder", "상급자 어깨 균형형과 세부 집중 3가지"),
        ("specialization_arm", "상급자 팔 균형형과 세부 집중 3가지"),
    ):
        lines.extend(
            _comparison_table(
                title,
                group(group_name),
                results,
                detail_header="직접 세트 분포",
                detail=lambda _fixture, result: _direct_set_distribution(result),
            )
        )
    lines.extend(
        _comparison_table(
            "상급자 수행법 자동 적용과 억제",
            group("advanced_method"),
            results,
            detail_header="실제 수행법",
            detail=lambda _fixture, result: _method_summary(result),
        )
    )
    return lines


def _render_review(
    *,
    fixture_path: Path,
    result_path: Path,
    candidate_report_path: Path,
    live_quality_pass_count: int,
    fixtures: list[dict[str, Any]],
    result_payload: dict[str, Any],
    results: dict[str, dict[str, Any]],
    document_version: str,
    document_title: str,
) -> str:
    success_count = sum(
        result.get("generationStatus") == "success" for result in results.values()
    )
    failed_count = sum(
        result.get("generationStatus") == "failed" for result in results.values()
    )
    fallback_count = sum(
        result.get("generationStatus") == "fallback" for result in results.values()
    )
    constrained_short_cases = []
    for fixture in fixtures:
        result = results[fixture["scenarioId"]]
        output_flags = set(result.get("outputQualityFlags") or [])
        if not output_flags.intersection(
            {"exercise_count_underfilled", "time_budget_underutilized"}
        ):
            continue
        constrained_short_cases.append(
            f"- 사례 {fixture['reviewOrder']} `{fixture['reviewTitle']}`: "
            f"{result.get('exerciseCount', 0)}개 운동, "
            f"{_working_set_total(result)}세트, "
            f"{result.get('totalEstimatedTime', 0)}분"
        )
    lines = [
        f"# {document_title}",
        "",
        f"- 문서 버전: `{document_version}`",
        "- 작성 기준일: 2026-07-30",
        "- 대상: 운동 코칭·현장 실행성 검토자",
        "",
        "> 이전에 전달한 v1 이후 입력과 추천 규칙이 크게 바뀌었습니다. 이 문서는 "
        "이전 문서를 안 읽어도 되도록 현재 AI를 처음부터 설명합니다. 개발 방식이나 "
        "프롬프트를 평가하실 필요는 없습니다. 사용자가 어떤 조건을 골랐고, 추천 "
        "운동·순서·세트·강도·안내가 코칭 관점에서 자연스러운지만 봐주시면 됩니다.",
        "",
        "## 검토 목적",
        "",
        "1. 각 입력이 실제 추천에 올바른 방향으로 영향을 주는지 확인",
        "2. 여러 입력이 겹쳤을 때 안전과 운동 목표의 우선순위가 자연스러운지 확인",
        "3. 초급·중급·상급, 근육통·부상, 체형·동작 제한, 세션 방식, 보조 목적, 상급 집중·수행법이 현장에서 납득되는지 확인",
        "4. 사용자에게 보이는 추천 이유와 주의 안내가 이해하기 쉽고 과장되지 않았는지 확인",
        "",
        "## 이번 검토 결과의 기술적 상태",
        "",
        f"- 현재 Gemma4 실제 실행 사례: {len(fixtures)}개",
        f"- 정상 추천: {success_count}개",
        f"- 안전 후보 부족으로 정책상 추천 중단: {failed_count}개",
        f"- 모델 장애 대체 경로 사용: {fallback_count}개",
        f"- 실행 예외: {result_payload.get('failedScenarioCount', 0)}개",
        f"- 사례별 기대 품질 신호 확인: {live_quality_pass_count}/{len(fixtures)}개",
        "- 모든 추천은 먼저 결정론적 안전·장비·구성 규칙을 통과한 후보 안에서만 선택되었습니다.",
        "- 이 수치는 프로그램 작동 증거이며 운동 코칭 적절성의 전문가 승인을 뜻하지 않습니다.",
        "",
        "### 먼저 확인할 짧은 구성",
        "",
        *(
            constrained_short_cases
            if constrained_short_cases
            else ["- 별도로 짧게 구성된 정상 추천 사례는 없습니다."]
        ),
        "",
        "위 사례는 오류나 fallback으로 숨기지 않았습니다. 안전 조건·직접 선택 범위·"
        "중복 억제를 지킨 결과이지만, 실제 코칭에서 이 정도로 짧게 끝내는 것이 "
        "타당한지는 전체 사례에서 확인해 주세요.",
        "",
        "## 검토 방법",
        "",
        "- 먼저 1~3부에서 입력 종류, 충돌 규칙, 핵심 비교표를 읽어 주세요.",
        "- 이후 4부의 사례를 관심 분야부터 확인해도 됩니다.",
        "- 각 사례는 `적절 / 조건부 적절 / 수정 필요 / 판단하기 어려움` 중 하나로 평가해 주세요.",
        "- 취향 차이와 안전·구성 결함을 구분해 주시면 수정 우선순위를 잡는 데 도움이 됩니다.",
        "",
        *_render_input_dictionary(),
        *_render_priority_rules(),
        *_render_comparisons(fixtures, results),
        "## 4. 전체 실제 추천 사례",
        "",
    ]
    current_section = None
    for fixture in fixtures:
        section = str(fixture["reviewSection"])
        if section != current_section:
            lines.extend([f"### {section}", ""])
            current_section = section
        lines.extend(_render_case(fixture, results[fixture["scenarioId"]]))

    lines.extend(
        [
            "## 5. 최종 종합 질문",
            "",
            "1. 기본 헬스장 루틴의 운동 순서가 큰 근육·고기술 메인 운동에서 보조·마무리 운동으로 자연스럽게 이어집니까?",
            "2. 같은 조건에서 운동 경력만 바꾼 초급·중급·상급 결과가 종목 난도와 처방에서 의미 있게 구분됩니까?",
            "3. 근력·근비대·근지구력·체지방 감소·체성분 개선·일반 체력의 처방 차이가 현장 기준으로 납득됩니까?",
            "4. 오늘 컨디션이 낮음·보통·좋음일 때 세트·RIR·운동 난도 변화가 과하거나 부족하지 않습니까?",
            "5. 근육통 1·2·3단계의 직접 운동량 감소와 3단계 안전 중단 범위가 적절합니까?",
            "6. 등록 부상은 관련 운동을 제외하고, 이미 관리 중인 부상은 확인된 제한 동작만 제외하는 구분이 실용적입니까?",
            "7. 깊은 앉기·힙힌지·머리 위 팔 올리기·손목 신전 어려움과 긴 팔·긴 허벅지 신호가 운동 선택과 세팅 설명에 적절히 반영됩니까?",
            "8. 일반 세트·슈퍼세트·서킷·밀도형의 묶음 수, 장비 이동, 휴식이 혼자 운동하는 환경에서 현실적입니까?",
            "9. 모빌리티·프리햅·컨디셔닝·지구력·쿨다운이 메인 헬스장 루틴을 해치지 않고 보조 역할에 머뭅니까?",
            "10. 상급자 세부 집중의 한 세트 재배분이 체감할 만큼 유용하면서 전체 균형을 지나치게 해치지 않습니까?",
            "11. 자동 정지 반복과 3초 하강의 대상 운동, 수행 안내, 최근 사용·컨디션·근육통에 따른 억제가 적절합니까?",
            "12. 운동 이름, 처방 단위, 추천 이유, 주의 안내 중 사용자가 오해하거나 실행하기 어려운 표현이 있습니까?",
            "",
            "## 6. 검증 범위와 한계",
            "",
            f"- 실제 실행 결과: `{result_path}`",
            f"- 입력 fixture: `{fixture_path}`",
            f"- 후보·필수 슬롯 사전 검사: `{candidate_report_path}`",
            f"- 이 문서는 {len(fixtures)}개 대표 조합을 다루며 가능한 모든 조합을 전부 나열하지 않습니다.",
        "- 비교표와 전체 사례는 이 문서용으로 새로 선언한 동일 fixture와 동일 live 실행 결과에서 생성했습니다.",
        "- 과거 전문가 문서의 추천 결과나 과거 fixture 행을 복사해 사례로 사용하지 않았습니다.",
            "- 전체 운동 DB의 모든 행을 전문가가 승인했다는 뜻이 아닙니다.",
            "- 의료 진단, 치료, 재활 처방, 모든 사용자에게 최적이라는 주장을 하지 않습니다.",
            "- 관리 중 부상, 상급자 세부 집중·수행법 등 일부 최신 입력은 AI 코어에서 검증된 상태이며 실제 FE·BE 화면 연결 전입니다.",
            "- 전문가 검토에서 수정이 필요한 사례가 나오면 해당 조건만 재현해 정책·DB·설명을 구분해 수정합니다.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_response(
    fixtures: list[dict[str, Any]],
    document_version: str = "current-comprehensive-expert-v1",
) -> str:
    sections: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for fixture in fixtures:
        sections.setdefault(str(fixture["reviewSection"]), []).append(fixture)
    lines = [
        "# Fit-Core 운동 루틴 AI 현재판 종합 코칭 답변지",
        "",
        f"- 대응 검토 문서 버전: `{document_version}`",
        "",
        "검토 자료의 개발 구조는 보지 않으셔도 됩니다. 실제 사용자 조건과 추천 "
        "운동·순서·처방·안내만 평가해 주세요.",
        "",
        "평가 선택: `적절 / 조건부 적절 / 수정 필요 / 판단하기 어려움`",
        "",
        "## 영역별 총평",
        "",
        "| 영역 | 평가 | 잘 된 점 | 반드시 고칠 점 | 적용 조건·이유 |",
        "|---|---|---|---|---|",
    ]
    for section in sections:
        lines.append(f"| {section} |  |  |  |  |")
    lines.extend(
        [
            "",
            "## 사례별 답변",
            "",
            "| 사례 | 평가 | 바꿀 운동·순서·세트·강도 | 설명·주의 안내 의견 | 이유 또는 적용 조건 |",
            "|---|---|---|---|---|",
        ]
    )
    for fixture in fixtures:
        lines.append(
            f"| {fixture['reviewOrder']}. {fixture['reviewTitle']} |  |  |  |  |"
        )
    lines.extend(
        [
            "",
            "## 최종 질문 답변",
            "",
            "### 1. 기본 운동 순서와 세션 구성",
            "",
            "의견:",
            "",
            "### 2. 초급·중급·상급 구분",
            "",
            "의견:",
            "",
            "### 3. 운동 목표 6종의 처방 차이",
            "",
            "의견:",
            "",
            "### 4. 컨디션·시간·장비 조절",
            "",
            "의견:",
            "",
            "### 5. 근육통 1·2·3단계",
            "",
            "의견:",
            "",
            "### 6. 등록 부상과 관리 중 부상",
            "",
            "의견:",
            "",
            "### 7. 동작 수행 어려움과 체형 특징",
            "",
            "의견:",
            "",
            "### 8. 일반 세트·슈퍼세트·서킷·밀도형",
            "",
            "의견:",
            "",
            "### 9. 모빌리티·프리햅·컨디셔닝·지구력·쿨다운",
            "",
            "의견:",
            "",
            "### 10. 상급자 세부 집중과 수행법",
            "",
            "의견:",
            "",
            "### 11. 사용자 설명과 주의 안내",
            "",
            "의견:",
            "",
            "## 가장 먼저 수정해야 할 사례",
            "",
            "사례 번호·이름:",
            "",
            "문제:",
            "",
            "권장 수정:",
            "",
            "## 추가로 보고 싶은 사용자 조건",
            "",
            "조건:",
            "",
            "확인하려는 이유:",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_result(
    fixtures: list[dict[str, Any]],
    payload: dict[str, Any],
) -> dict[str, Any]:
    rows = payload.get("scenarios") or []
    fixture_by_id = {str(row["scenarioId"]): row for row in fixtures}
    result_by_id = {str(row["scenarioId"]): row for row in rows}
    if set(fixture_by_id) != set(result_by_id):
        raise ValueError(
            "scenario mismatch: "
            f"fixture_only={sorted(set(fixture_by_id) - set(result_by_id))}, "
            f"result_only={sorted(set(result_by_id) - set(fixture_by_id))}"
        )
    unexpected_failed = []
    fallback_ids = []
    contract_invalid = []
    hard_violations = []
    quality_flags: dict[str, list[str]] = {}
    for scenario_id, result in result_by_id.items():
        fixture = fixture_by_id[scenario_id]
        if (
            result.get("generationStatus") == "failed"
            and not fixture.get("candidateEmptyAllowed")
        ):
            unexpected_failed.append(scenario_id)
        if result.get("generationStatus") == "fallback" or result.get("isFallback"):
            fallback_ids.append(scenario_id)
        if result.get("contractValid") is not True:
            contract_invalid.append(scenario_id)
        if int(result.get("hardViolationCount") or 0):
            hard_violations.append(scenario_id)
        flags = [
            *(result.get("languageQualityFlags") or []),
            *(result.get("outputQualityFlags") or []),
        ]
        if (
            fixture.get("candidateEmptyAllowed")
            and result.get("generationStatus") == "failed"
            and result.get("statusReasonCode") == "emptyCandidate"
        ):
            flags = [flag for flag in flags if flag != "fallback_or_non_success"]
        if flags:
            quality_flags[scenario_id] = flags
    if payload.get("failedScenarioCount"):
        raise ValueError(f"live runner exceptions: {payload.get('failures')}")
    if unexpected_failed or fallback_ids or contract_invalid or hard_violations:
        raise ValueError(
            json.dumps(
                {
                    "unexpectedFailed": unexpected_failed,
                    "fallback": fallback_ids,
                    "contractInvalid": contract_invalid,
                    "hardViolations": hard_violations,
                },
                ensure_ascii=False,
            )
        )
    return {
        "scenarioCount": len(rows),
        "successCount": sum(row.get("generationStatus") == "success" for row in rows),
        "expectedPolicyStopCount": sum(
            row.get("generationStatus") == "failed" for row in rows
        ),
        "fallbackCount": len(fallback_ids),
        "runnerExceptionCount": int(payload.get("failedScenarioCount") or 0),
        "contractInvalidCount": len(contract_invalid),
        "hardViolationScenarioCount": len(hard_violations),
        "qualityFlagScenarioCount": len(quality_flags),
        "qualityFlags": quality_flags,
    }


def _validate_rendered_documents(
    review: str,
    response: str,
    fixtures: list[dict[str, Any]],
) -> dict[str, Any]:
    case_heading_count = review.count("\n#### 사례 ")
    response_row_count = sum(
        f"| {fixture['reviewOrder']}. {fixture['reviewTitle']} |" in response
        for fixture in fixtures
    )
    forbidden_tokens = (
        "V24ROW-",
        "targetSplitLabel",
        "experienceLevel",
        "readinessLevel",
        "currentPainAreas",
        "currentDoms",
        "SMITH_MACHINE",
        "specializationFocus",
        "methodMode",
        "injuryConditions",
    )
    leaked_tokens = [token for token in forbidden_tokens if token in review]
    missing_titles = [
        fixture["reviewTitle"]
        for fixture in fixtures
        if fixture["reviewTitle"] not in review
    ]
    if (
        case_heading_count != len(fixtures)
        or response_row_count != len(fixtures)
        or leaked_tokens
        or missing_titles
    ):
        raise ValueError(
            json.dumps(
                {
                    "caseHeadingCount": case_heading_count,
                    "responseRowCount": response_row_count,
                    "leakedTokens": leaked_tokens,
                    "missingTitles": missing_titles,
                },
                ensure_ascii=False,
            )
        )
    return {
        "caseHeadingCount": case_heading_count,
        "responseRowCount": response_row_count,
        "technicalTokenLeakCount": len(leaked_tokens),
        "missingTitleCount": len(missing_titles),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument(
        "--candidate-report",
        type=Path,
        default=DEFAULT_CANDIDATE_REPORT,
    )
    parser.add_argument(
        "--candidate-live-report",
        type=Path,
        default=DEFAULT_CANDIDATE_LIVE_REPORT,
    )
    parser.add_argument("--review-output", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--response-output", type=Path, default=DEFAULT_RESPONSE)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--document-version",
        default="current-comprehensive-expert-v1",
    )
    parser.add_argument(
        "--document-title",
        default="Fit-Core 운동 루틴 AI 현재판 종합 코칭 검토 자료",
    )
    args = parser.parse_args()

    fixtures = sorted(
        _load_jsonl(args.fixture),
        key=lambda row: int(row.get("reviewOrder") or 0),
    )
    result_payload = _load_json(args.result)
    result_rows = result_payload.get("scenarios") or []
    results = {str(row["scenarioId"]): row for row in result_rows}
    validation = _validate_result(fixtures, result_payload)
    candidate_report = _load_json(args.candidate_report)
    candidate_summary = candidate_report.get("report") or candidate_report
    blueprint = candidate_summary.get("blueprintSlotCoverage") or {}
    adjunct = candidate_summary.get("adjunctSlotCoverage") or {}
    if blueprint.get("status") != "pass" or adjunct.get("status") != "pass":
        raise ValueError("candidate required-slot or adjunct-slot coverage did not pass")
    candidate_live_report = _load_json(args.candidate_live_report)
    candidate_live_summary = (
        candidate_live_report.get("report") or candidate_live_report
    )
    candidate_live_rows = candidate_live_summary.get("scenarios") or []
    live_quality_pass_count = sum(
        row.get("livePreferredQualityPassed") is True
        for row in candidate_live_rows
    )
    if (
        len(candidate_live_rows) != len(fixtures)
        or live_quality_pass_count != len(fixtures)
    ):
        raise ValueError(
            "candidate/live integrated quality did not pass for every scenario"
        )

    review = _render_review(
        fixture_path=args.fixture,
        result_path=args.result,
        candidate_report_path=args.candidate_report,
        live_quality_pass_count=live_quality_pass_count,
        fixtures=fixtures,
        result_payload=result_payload,
        results=results,
        document_version=args.document_version,
        document_title=args.document_title,
    )
    response = _render_response(fixtures, args.document_version)
    document_validation = _validate_rendered_documents(
        review,
        response,
        fixtures,
    )

    for path in (args.review_output, args.response_output, args.manifest_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.review_output.write_text(review, encoding="utf-8")
    args.response_output.write_text(response, encoding="utf-8")

    manifest = {
        "schemaVersion": f"routine-{args.document_version}",
        "generatedAt": "2026-07-30",
        "status": "passed",
        "validation": validation,
        "documentValidation": document_validation,
        "candidateCoverage": {
            "eligibleRequiredSlotCount": blueprint.get("eligibleRequiredSlotCount"),
            "eligibleRequiredSlotCoveredCount": blueprint.get(
                "eligibleRequiredSlotCoveredCount"
            ),
            "unexpectedUnfilledScenarioCount": blueprint.get(
                "unexpectedUnfilledScenarioCount"
            ),
            "adjunctAppliedScenarioCount": adjunct.get("appliedScenarioCount"),
            "adjunctFilledScenarioCount": adjunct.get("filledScenarioCount"),
            "livePreferredQualityPassCount": live_quality_pass_count,
        },
        "files": {
            "fixture": {
                "path": str(args.fixture),
                "sha256": _sha256(args.fixture),
            },
            "liveResult": {
                "path": str(args.result),
                "sha256": _sha256(args.result),
            },
            "candidateReport": {
                "path": str(args.candidate_report),
                "sha256": _sha256(args.candidate_report),
            },
            "candidateLiveReport": {
                "path": str(args.candidate_live_report),
                "sha256": _sha256(args.candidate_live_report),
            },
            "review": {
                "path": str(args.review_output),
                "sha256": _sha256(args.review_output),
            },
            "response": {
                "path": str(args.response_output),
                "sha256": _sha256(args.response_output),
            },
        },
        "claimBoundary": [
            "human_coaching_usability_review_not_medical_validation",
            "representative_scenarios_not_all_combinations",
            "latest_core_inputs_include_pre_integration_fields",
            "no_developer_handoff",
            "no_commit_push_deploy_or_notion_edit",
        ],
    }
    args.manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

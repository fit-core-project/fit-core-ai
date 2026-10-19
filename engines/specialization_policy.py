"""Closed specialization-focus and advanced training-method policy.

The exercise catalog may contain method variants as separate rows, but the
product contract treats the method as a modifier of a recognizable base
exercise. This module keeps those concepts separate and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

if TYPE_CHECKING:
    from .routine_blueprint import RoutineBlueprint
    from .schemas import LLMExercisePlan, LLMRoutineOutput


SPECIALIZATION_FOCUS_TARGETS: dict[str, frozenset[str]] = {
    "upperChest": frozenset({"chest"}),
    "midChest": frozenset({"chest"}),
    "lowerChest": frozenset({"chest"}),
    "latEmphasis": frozenset({"upper-back"}),
    "midBackEmphasis": frozenset({"upper-back"}),
    "trapeziusEmphasis": frozenset({"trapezius", "upper-back"}),
    "quadricepsEmphasis": frozenset({"quadriceps"}),
    "hamstringEmphasis": frozenset({"hamstring"}),
    "gluteEmphasis": frozenset({"gluteal"}),
    "adductorEmphasis": frozenset({"adductor"}),
    "abductorEmphasis": frozenset({"abductors"}),
    "calvesEmphasis": frozenset({"calves"}),
    "frontDeltoidEmphasis": frozenset({"front-deltoids"}),
    "sideDeltoidEmphasis": frozenset({"side-deltoids"}),
    "rearDeltoidEmphasis": frozenset({"back-deltoids"}),
    "bicepsEmphasis": frozenset({"biceps"}),
    "tricepsEmphasis": frozenset({"triceps"}),
    "forearmEmphasis": frozenset({"forearm"}),
}

TRAINING_METHOD_GOALS: dict[str, frozenset[str]] = {
    "pausedReps": frozenset({"strength"}),
    "tempoEccentric": frozenset({"hypertrophy", "recomposition"}),
}

LEGACY_CHEST_FOCUS_MAP: dict[str, tuple[Optional[str], Optional[str]]] = {
    "balanced": (None, None),
    "upperChest": ("upperChest", None),
    "lowerChest": ("lowerChest", None),
    "pausedPress": (None, "pausedReps"),
}

_DIRECT_PRIMARY_FOCUS: dict[str, str] = {
    "quadricepsEmphasis": "quadriceps",
    "hamstringEmphasis": "hamstring",
    "gluteEmphasis": "gluteal",
    "adductorEmphasis": "adductor",
    "abductorEmphasis": "abductors",
    "calvesEmphasis": "calves",
    "frontDeltoidEmphasis": "front-deltoids",
    "sideDeltoidEmphasis": "side-deltoids",
    "rearDeltoidEmphasis": "back-deltoids",
    "bicepsEmphasis": "biceps",
    "tricepsEmphasis": "triceps",
    "forearmEmphasis": "forearm",
}

_FOCUS_LABELS_KO: dict[str, str] = {
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

_METHOD_INSTRUCTIONS: dict[str, dict[str, Any]] = {
    "pausedReps": {
        "code": "pausedReps",
        "labelKo": "정지 반복",
        "purposeKo": "반동을 줄이고 정지 지점에서 힘과 동작 통제를 훈련합니다.",
        "instructionKo": "각 반복의 지정 지점에서 2초 정지한 뒤 자세를 유지하며 다음 구간을 수행하세요.",
        "pauseSec": 2,
    },
    "tempoEccentric": {
        "code": "tempoEccentric",
        "labelKo": "3초 하강",
        "purposeKo": "내려가는 구간을 통제해 목표 근육의 긴장과 자세 재현성을 높입니다.",
        "instructionKo": "각 반복의 내려가는 구간을 3초 동안 일정하게 통제하세요.",
        "eccentricSec": 3,
    },
}


@dataclass(frozen=True)
class TrainingMethodResolution:
    method_code: Optional[str]
    source: Optional[str]
    reason_code: Optional[str]


def _entry_value(entry: Any, *keys: str) -> Any:
    if isinstance(entry, Mapping):
        for key in keys:
            if key in entry:
                return entry[key]
        return None
    for key in keys:
        if hasattr(entry, key):
            return getattr(entry, key)
    return None


def resolve_training_method(
    *,
    method_mode: Optional[str],
    requested_method: Optional[str],
    goal: Optional[str],
    experience_level: Optional[str],
    readiness_level: Optional[str],
    target_muscles: Sequence[str],
    target_split_label: Optional[str] = None,
    recent_method_uses: Optional[Sequence[Any]] = None,
    pain_areas: Optional[Sequence[Any]] = None,
    doms_data: Optional[Mapping[str, Any]] = None,
    injury_conditions: Optional[Sequence[Any]] = None,
) -> TrainingMethodResolution:
    """Resolve one optional advanced method without delegating the policy to the LLM."""

    mode = str(method_mode or ("manual" if requested_method else "auto")).strip()
    if mode == "off":
        return TrainingMethodResolution(None, None, "disabledByUser")
    if mode not in {"auto", "manual"}:
        raise ValueError("unsupported method mode")

    experience = str(experience_level or "").strip().lower()
    readiness = str(readiness_level or "normal").strip().lower()
    normalized_goal = str(goal or "").strip().lower().replace("_", "")
    if experience != "advanced":
        return TrainingMethodResolution(None, None, "advancedExperienceRequired")
    if readiness == "low":
        return TrainingMethodResolution(None, None, "lowReadiness")

    targets = {
        str(target or "").strip().lower().replace("_", "-")
        for target in target_muscles
        if str(target or "").strip()
    }
    registered_injuries = {
        str(_entry_value(entry, "body_part", "bodyPart", "area") or "")
        .strip()
        .lower()
        .replace("_", "-")
        for entry in (pain_areas or ())
    }
    if targets & registered_injuries:
        return TrainingMethodResolution(None, None, "targetInjury")

    for entry in injury_conditions or ():
        status = str(
            _entry_value(entry, "status") or ""
        ).strip()
        if status in {"activeUncleared", "managedWithRestrictions"}:
            return TrainingMethodResolution(None, None, "managedInjuryRestriction")

    for body_part, raw_level in (doms_data or {}).items():
        normalized_part = str(body_part or "").strip().lower().replace("_", "-")
        try:
            level = int(raw_level)
        except (TypeError, ValueError):
            continue
        if normalized_part in targets and level >= 2:
            return TrainingMethodResolution(None, None, "targetDoms")

    if mode == "manual":
        method_code = str(requested_method or "").strip() or None
        if not method_code:
            raise ValueError("manual method mode requires requested method")
        return TrainingMethodResolution(
            method_code,
            "manual",
            "manualOverride",
        )

    if normalized_goal == "strength":
        method_code = "pausedReps"
        reason_code = "advancedStrengthMainControl"
    elif normalized_goal in {"hypertrophy", "recomposition"}:
        method_code = "tempoEccentric"
        reason_code = "advancedHypertrophyTempoControl"
    else:
        return TrainingMethodResolution(None, None, "unsupportedGoal")

    normalized_split = (
        str(target_split_label or "").strip().lower().replace("-", "_").replace(" ", "_")
    )
    for entry in recent_method_uses or ():
        recent_method = str(
            _entry_value(entry, "method_code", "methodCode") or ""
        ).strip()
        recent_split = str(
            _entry_value(entry, "target_split_label", "targetSplitLabel") or ""
        ).strip().lower().replace("-", "_").replace(" ", "_")
        try:
            sessions_ago = int(_entry_value(entry, "sessions_ago", "sessionsAgo"))
        except (TypeError, ValueError):
            continue
        same_scope = not recent_split or not normalized_split or recent_split == normalized_split
        if recent_method == method_code and same_scope and sessions_ago <= 2:
            return TrainingMethodResolution(None, None, "recentMethodCooldown")

    return TrainingMethodResolution(method_code, "auto", reason_code)


def _searchable_candidate_text(candidate: Mapping[str, Any]) -> str:
    return " ".join(
        str(candidate.get(key) or "").strip().lower()
        for key in (
            "name_kr",
            "name_en",
            "variation_name_kr",
            "catalog_name_kr",
            "catalog_name_en",
            "detail_subpart",
        )
    )


def focus_label_ko(focus_code: Optional[str]) -> str:
    return _FOCUS_LABELS_KO.get(str(focus_code or ""), "")


def focus_variant_family_key(candidate: Mapping[str, Any], focus_code: Optional[str], base_family: str) -> str:
    """Allow one loaded setup per explicit focus, not arbitrary duplicate IDs."""
    if not candidate_matches_specialization_focus(candidate, focus_code):
        return base_family
    if candidate.get("main_routine_default_allowed") in {False, 0, "0"}:
        return base_family
    primary = str(candidate.get("primary_muscle") or "")
    supported = (
        (focus_code == "midBackEmphasis" and base_family == "horizontal_row")
        or (focus_code == "sideDeltoidEmphasis" and primary == "side-deltoids")
        or (focus_code == "trapeziusEmphasis" and primary == "trapezius")
    )
    if not supported:
        return base_family
    equipment = str(candidate.get("equipment_req") or candidate.get("equipment") or "").strip().upper()
    if equipment not in {"MACHINE", "CABLE", "DUMBBELL", "BARBELL"}:
        return base_family
    return f"{base_family}:focus_setup:{equipment}"


def legacy_specialization_values(
    advanced_chest_focus: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    return LEGACY_CHEST_FOCUS_MAP.get(
        str(advanced_chest_focus or "balanced").strip(),
        (None, None),
    )


def focus_matches_targets(
    focus_code: Optional[str],
    target_muscles: list[str] | tuple[str, ...] | set[str],
) -> bool:
    if not focus_code:
        return True
    allowed_targets = SPECIALIZATION_FOCUS_TARGETS.get(str(focus_code))
    return bool(allowed_targets and allowed_targets.intersection(target_muscles))


def candidate_matches_specialization_focus(
    candidate: Mapping[str, Any],
    focus_code: Optional[str],
) -> bool:
    if not focus_code:
        return False
    declared_raw = candidate.get("specialization_focus_codes")
    if declared_raw:
        declared = {
            token.strip()
            for token in str(declared_raw).split(",")
            if token.strip()
        }
        return str(focus_code) in declared

    primary = str(candidate.get("primary_muscle") or "").strip().lower()
    text = _searchable_candidate_text(candidate)
    bench_angle = str(candidate.get("bench_angle_band") or "").strip().lower()

    if focus_code == "upperChest":
        return primary == "chest" and (
            bench_angle.startswith("incline")
            or "인클라인" in text
            or "incline" in text
        )
    if focus_code == "midChest":
        return primary == "chest" and (
            bench_angle == "flat"
            or "플랫" in text
            or "flat" in text
        )
    if focus_code == "lowerChest":
        return primary == "chest" and (
            bench_angle == "decline"
            or "디클라인" in text
            or "decline" in text
        )
    if focus_code == "latEmphasis":
        return primary == "upper-back" and any(
            token in text
            for token in (
                "수직 당기기",
                "수직당기기",
                "광배",
                "랫풀",
                "lat pull",
                "latpull",
                "pullup",
                "pull-up",
                "풀업",
            )
        )
    if focus_code == "midBackEmphasis":
        return primary == "upper-back" and any(
            token in text
            for token in (
                "수평 당기기",
                "수평당기기",
                "등두께",
                "능형",
                "중부승모",
                "row",
                "로우",
            )
        )
    if focus_code == "trapeziusEmphasis":
        return primary == "trapezius"

    direct_primary = _DIRECT_PRIMARY_FOCUS.get(str(focus_code))
    return bool(direct_primary and primary == direct_primary)


def candidate_matches_training_method(
    candidate: Mapping[str, Any],
    method_code: Optional[str],
) -> bool:
    """Return whether a familiar base exercise can receive the method.

    Catalog rows may encode tempo/pause as separate variations, but routine
    generation treats the method as server-owned prescription metadata. This
    prevents obscure variation names from replacing recognizable exercises.
    """
    if not method_code:
        return False
    declared = {
        token.strip()
        for token in str(candidate.get("training_method_codes") or "").split(",")
        if token.strip()
    }

    text = _searchable_candidate_text(candidate)
    movement_type = str(
        candidate.get("canonical_movement_type")
        or candidate.get("movement_type")
        or ""
    ).strip().upper()
    equipment = {
        token.strip().upper()
        for token in str(candidate.get("equipment_req") or "").split(",")
        if token.strip()
    }
    loadable = bool(
        equipment
        & {
            "BARBELL",
            "DUMBBELL",
            "MACHINE",
            "CABLE",
            "SMITH_MACHINE",
            "KETTLEBELL",
            "LANDMINE",
            "PLATE",
        }
    )
    ballistic = any(
        token in text
        for token in (
            "스윙",
            "점프",
            "스내치",
            "클린",
            "저크",
            "throw",
            "swing",
            "jump",
            "snatch",
            "clean",
            "jerk",
        )
    )
    incompatible_context = any(
        token in text
        for token in (
            "crossfit",
            "크로스핏",
            "메트콘",
            "metcon",
            "amrap",
            "emom",
            "벤치마크 준비",
            "기술연습",
            "요가",
            "yoga",
            "필라테스",
            "pilates",
            "재활",
            "rehab",
            "스트레칭",
            "stretch",
            "파머스 워크",
            "farmer",
            "캐리",
            "carry",
        )
    )
    if incompatible_context:
        return False

    metadata_keys = {
        "main_routine_default_allowed",
        "mainstream_gym_score",
        "exercise_familiarity_tier",
        "training_style_tags",
        "prescription_mode",
    }
    if metadata_keys.intersection(candidate):
        main_default = candidate.get("main_routine_default_allowed") in {
            True,
            1,
            "1",
        }
        try:
            mainstream_score = int(candidate.get("mainstream_gym_score") or 0)
        except (TypeError, ValueError):
            mainstream_score = 0
        familiarity = str(
            candidate.get("exercise_familiarity_tier") or ""
        ).strip().lower()
        novelty = str(candidate.get("novelty_level") or "").strip().lower()
        style_tags = {
            token.strip().lower()
            for token in str(candidate.get("training_style_tags") or "").split(",")
            if token.strip()
        }
        prescription_mode = str(
            candidate.get("prescription_mode") or ""
        ).strip().lower()
        if not (
            main_default
            and mainstream_score >= 75
            and familiarity in {"common", "standard"}
            and novelty not in {"medium", "high"}
            and style_tags.intersection({"gym", "bodybuilding"})
            and prescription_mode == "reps"
        ):
            return False

    if declared and str(method_code) not in declared:
        return False
    if method_code == "pausedReps":
        return movement_type == "COMPOUND" and loadable and not ballistic
    if method_code == "tempoEccentric":
        return movement_type in {"COMPOUND", "ISOLATION"} and loadable and not ballistic
    return False


def candidate_encodes_training_method(
    candidate: Mapping[str, Any],
    method_code: Optional[str],
) -> bool:
    """Return whether the source catalog name itself encodes the method."""
    if not method_code:
        return False
    text = _searchable_candidate_text(candidate)
    if method_code == "pausedReps":
        return any(token in text for token in ("정지", "paused", "pause"))
    if method_code == "tempoEccentric":
        return any(
            token in text
            for token in (
                "템포",
                "tempo",
                "3초 하강",
                "3초 이완",
                "3-second eccentric",
                "slow eccentric",
            )
        )
    return False


def training_method_instruction(
    method_code: Optional[str],
    *,
    source: Optional[str] = None,
    reason_code: Optional[str] = None,
    candidate: Optional[Mapping[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    instruction = _METHOD_INSTRUCTIONS.get(str(method_code or ""))
    if not instruction:
        return None
    resolved = dict(instruction)
    text = _searchable_candidate_text(candidate or {})
    if method_code == "pausedReps" and candidate:
        if any(token in text for token in ("프레스", "press")):
            resolved["instructionKo"] = (
                "중량이 가슴 또는 어깨 쪽 하단 지점에 도달했을 때 반동 없이 "
                "2초 정지한 뒤 자세를 유지하며 밀어 올리세요."
            )
        elif any(token in text for token in ("스쿼트", "squat", "레그 프레스", "leg press")):
            resolved["instructionKo"] = (
                "통제 가능한 최하단 자세에서 긴장을 유지한 채 2초 정지한 뒤 "
                "반동 없이 일어나세요."
            )
        elif any(token in text for token in ("로우", "row", "풀다운", "pulldown", "풀업", "pull-up")):
            resolved["instructionKo"] = (
                "손잡이나 중량을 몸쪽으로 당긴 수축 지점에서 2초 정지한 뒤 "
                "천천히 시작 자세로 돌아가세요."
            )
    resolved["source"] = str(source or "manual")
    if reason_code:
        resolved["reasonCode"] = str(reason_code)
    return resolved


def training_method_supports_goal(
    method_code: Optional[str],
    goal: Optional[str],
) -> bool:
    if not method_code:
        return True
    allowed_goals = TRAINING_METHOD_GOALS.get(str(method_code))
    normalized_goal = str(goal or "").strip().lower().replace("_", "")
    return bool(allowed_goals and normalized_goal in allowed_goals)


def apply_training_method_to_plan(
    exercise: "LLMExercisePlan",
    method_code: Optional[str],
    *,
    source: Optional[str] = None,
    reason_code: Optional[str] = None,
    candidate: Optional[Mapping[str, Any]] = None,
) -> "LLMExercisePlan":
    instruction = training_method_instruction(
        method_code,
        source=source,
        reason_code=reason_code,
        candidate=candidate,
    )
    if instruction:
        exercise._training_method_code = str(method_code)
        exercise._training_method_instruction = instruction
        from .prescription.weight import invalidate_weight_for_training_method

        invalidate_weight_for_training_method(exercise)
    return exercise


def primary_focus_candidate_ids(
    blueprint: Optional["RoutineBlueprint"],
    candidates: Sequence[dict],
) -> set[str]:
    """Identify admitted focus roles, without promoting or admitting candidates."""
    if (
        blueprint is None
        or blueprint.experience_level != "advanced"
        or blueprint.readiness_level == "low"
        or not blueprint.specialization_focus
    ):
        return set()
    return {
        str(candidate.get("id") or "").strip().lower()
        for candidate in candidates
        if str(candidate.get("id") or "").strip()
        and str(candidate.get("prescription_mode") or "reps").lower() == "reps"
        and candidate_matches_specialization_focus(candidate, blueprint.specialization_focus)
    }


def _allocate_forearm_session(
    output: "LLMRoutineOutput",
    blueprint: "RoutineBlueprint",
    candidate_by_id: Mapping[str, dict],
    restricted_ids: set[str],
) -> bool:
    """Finite product-policy allocation; the 3:2:1 shares are not a dose claim."""
    if not (
        blueprint.specialization_focus == "forearmEmphasis"
        and blueprint.experience_level == "advanced"
        and blueprint.target_split_label == "arm"
        and set(blueprint.target_muscles) == {"forearm", "biceps", "triceps"}
        and blueprint.goal in {"hypertrophy", "recomposition"}
        and blueprint.session_structure == "standard"
        and 1 <= len(output.exercises) <= 8
    ):
        return False

    from .prescription.estimator import _work_seconds_for_exercise

    muscles, ranges, costs = [], [], []
    original = tuple(plan.sets for plan in output.exercises)
    slots = {slot.slot_id: slot for slot in blueprint.slots}
    for plan in output.exercises:
        key = str(plan.exercise_id or "").strip().lower()
        candidate = candidate_by_id.get(key)
        primary = str((candidate or {}).get("primary_muscle") or "").strip().lower()
        if (
            primary not in {"forearm", "biceps", "triceps"}
            or str(plan.movement_type or "").upper() != "ISOLATION"
            or getattr(plan, "_prescription_set_type", "working") != "working"
            or getattr(plan, "_target_duration_sec", None)
            or getattr(plan, "_target_distance_m", None)
            or str((candidate or {}).get("prescription_mode") or "reps").lower() != "reps"
            or not 1 <= plan.sets <= 5
        ):
            return False
        muscles.append(primary)
        costs.append(_work_seconds_for_exercise(plan) + max(0, plan.rest_time_sec))
        slot = slots.get((output._selection_slot_by_exercise_id or {}).get(key))
        floor = 3 if slot and slot.requirement == "required_when_safe" and slot.phase in {
            "main_compound", "secondary_compound",
        } else 2
        if key in restricted_ids:
            ranges.append((plan.sets,))
        elif primary == "forearm":
            ranges.append(range(plan.sets, 6))
        else:
            ranges.append(range(min(plan.sets, floor), plan.sets + 1))

    # All active exercises remain, so warmup, transitions and final-rest terms
    # cancel. This exact marginal time bound cannot lengthen the original plan.
    total = sum(original)
    seconds = sum(n * cost for n, cost in zip(original, costs))
    weights = {"forearm": 3, "biceps": 2, "triceps": 1}

    def objective(counts: tuple[int, ...]) -> tuple:
        sums = {muscle: sum(n for n, m in zip(counts, muscles) if m == muscle) for muscle in weights}
        loss = sum((6 * sums[muscle] - total * weight) ** 2 for muscle, weight in weights.items())
        return loss, sum(abs(a - b) for a, b in zip(counts, original)), tuple(-n for n in counts)

    best = original
    best_key = objective(best)
    for counts in product(*ranges):
        if sum(counts) != total or sum(n * cost for n, cost in zip(counts, costs)) > seconds:
            continue
        key = objective(counts)
        if key < best_key:
            best, best_key = counts, key
    for plan, count in zip(output.exercises, best):
        plan.sets = count
    focus_sets = sum(n for n, muscle in zip(best, muscles) if muscle == "forearm")
    focus_plan = next((plan for plan, muscle in zip(output.exercises, muscles) if muscle == "forearm"), None)
    if focus_plan is not None:
        note = f"전완 집중은 전체 분량을 늘리지 않고 직접 작업 {focus_sets}세트로 배분했습니다. 종목별 상한과 다른 요청 부위의 기본 분량을 유지했습니다."
        if note not in focus_plan.exercise_rationale:
            focus_plan.exercise_rationale = f"{focus_plan.exercise_rationale} {note}".strip()
    output._selection_diagnostics["forearm_session_allocation"] = {
        "policy": "forearm-session-shares-v1", "directSetWeights": weights,
        "totalWorkingSets": total, "forearmWorkingSets": focus_sets,
        "receiverCap": 5, "targetShareShortfall": max(0, (total + 1) // 2 - focus_sets),
        "limitedByExistingRolesDoseAndTime": True,
    }
    return True


def apply_blueprint_specialization_volume(
    output: "LLMRoutineOutput",
    blueprint: "RoutineBlueprint",
    ranked_candidates: Sequence[dict],
    *,
    restricted_exercise_ids: Optional[set[str]] = None,
) -> "LLMRoutineOutput":
    """Make an explicit focus visible without inflating total session volume.

    Selection focus should not be represented by obscure catalog variations.
    Forearm arm sessions use a bounded whole-session allocation; other focuses
    retain the one-set transfer. Safety and low readiness take precedence.
    """

    focus_code = blueprint.specialization_focus
    focus_slot_id = blueprint.specialization_slot_id
    if (
        not focus_code
        or not focus_slot_id
        or blueprint.readiness_level == "low"
    ):
        return output

    restricted_ids = {
        str(exercise_id or "").strip().lower()
        for exercise_id in (restricted_exercise_ids or set())
        if str(exercise_id or "").strip()
    }
    candidate_by_id = {
        str(candidate.get("id") or "").strip().lower(): candidate
        for candidate in ranked_candidates
        if str(candidate.get("id") or "").strip()
    }
    if _allocate_forearm_session(output, blueprint, candidate_by_id, restricted_ids):
        return output
    slot_by_exercise_id = {
        str(exercise_id or "").strip().lower(): str(slot_id or "").strip()
        for exercise_id, slot_id in dict(
            output._selection_slot_by_exercise_id or {}
        ).items()
    }
    blueprint_slot_by_id = {slot.slot_id: slot for slot in blueprint.slots}

    def is_working_repetition_plan(exercise: "LLMExercisePlan", candidate: Mapping[str, Any]) -> bool:
        return (
            getattr(exercise, "_prescription_set_type", "working") == "working"
            and not getattr(exercise, "_target_duration_sec", None)
            and not getattr(exercise, "_target_distance_m", None)
            and str(candidate.get("prescription_mode") or "reps").lower() == "reps"
        )

    focus_plans = []
    for exercise in output.exercises:
        exercise_id = str(exercise.exercise_id or "").strip().lower()
        candidate = candidate_by_id.get(exercise_id)
        if (
            exercise_id not in restricted_ids
            and candidate is not None
            and is_working_repetition_plan(exercise, candidate)
            and candidate_matches_specialization_focus(candidate, focus_code)
        ):
            focus_plans.append(exercise)

    if not focus_plans:
        return output

    focus_plans.sort(key=lambda exercise: (
        slot_by_exercise_id.get(str(exercise.exercise_id or "").strip().lower()) != focus_slot_id
    ))
    # Keep the dedicated focus role first, but its cap must not hide another
    # already-selected recipient. Transfer at most the existing single set.
    focus_plan = next((exercise for exercise in focus_plans if exercise.sets < 5), focus_plans[0])

    focus_id = str(focus_plan.exercise_id or "").strip().lower()
    label = focus_label_ko(focus_code)

    def append_focus_note(note: str) -> None:
        current_rationale = str(focus_plan.exercise_rationale or "").strip()
        if note not in current_rationale:
            focus_plan.exercise_rationale = f"{current_rationale} {note}".strip()

    if focus_plan.sets >= 5:
        append_focus_note(
            f"{label} 집중 목표를 반영해 해당 운동과 수행 각도를 우선 배치했습니다."
        )
        return output

    def donor_priority(
        item: tuple[int, "LLMExercisePlan"],
    ) -> tuple[int, int, int]:
        index, exercise = item
        exercise_id = str(exercise.exercise_id or "").strip().lower()
        candidate = candidate_by_id.get(exercise_id)
        same_focus = int(
            candidate is not None
            and candidate_matches_specialization_focus(candidate, focus_code)
        )
        slot_id = slot_by_exercise_id.get(exercise_id, "")
        slot = blueprint_slot_by_id.get(slot_id)
        if not slot_id or slot_id.startswith("flex"):
            tier = 0
        elif slot and slot.requirement == "optional":
            tier = 1
        elif slot and slot.requirement == "preferred_when_safe":
            tier = 2
        elif slot and slot.phase in {"isolation_accessory", "prehab_accessory"}:
            tier = 3
        elif slot and slot.phase == "secondary_compound":
            tier = 4
        else:
            tier = 5
        # A focus transfer must change the user's direct-set distribution.
        # Prefer a non-focus donor before reducing another exercise that serves
        # the same focus (for example, another triceps or rear-delt exercise).
        return (same_focus, tier, -index)

    donor_candidates: list[tuple[int, "LLMExercisePlan"]] = []
    for index, exercise in enumerate(output.exercises):
        exercise_id = str(exercise.exercise_id or "").strip().lower()
        if exercise_id == focus_id or exercise_id in restricted_ids:
            continue
        candidate = candidate_by_id.get(exercise_id)
        if not is_working_repetition_plan(exercise, candidate or {}):
            continue
        if candidate is not None and candidate_matches_specialization_focus(candidate, focus_code):
            continue
        slot = blueprint_slot_by_id.get(slot_by_exercise_id.get(exercise_id, ""))
        required_main = bool(
            slot and slot.requirement == "required_when_safe"
            and slot.phase in {"main_compound", "secondary_compound"}
        )
        # A required main can be isolation (for example a posterior-chain curl).
        # Preserve its role floor, not just the compound/isolation label.
        minimum_sets = (
            3 if blueprint.experience_level != "beginner"
            and (required_main or str(exercise.movement_type or "").upper() == "COMPOUND") else 2
        )
        if exercise.sets > minimum_sets:
            donor_candidates.append((index, exercise))

    donor = (
        min(donor_candidates, key=donor_priority)[1]
        if donor_candidates
        else None
    )
    if donor is None:
        append_focus_note(
            f"{label} 집중 목표를 반영해 해당 운동과 수행 각도를 우선 배치했습니다."
        )
        return output

    focus_plan.sets += 1
    if donor is not None:
        donor.sets -= 1

    note = f"{label} 집중 목표를 반영해 직접 세트를 1세트 우선 배정했습니다."
    append_focus_note(note)
    return output


def apply_blueprint_training_method(
    output: "LLMRoutineOutput",
    blueprint: "RoutineBlueprint",
    ranked_candidates: Sequence[dict],
) -> "LLMRoutineOutput":
    """Apply one method modifier after selection/fill has stabilized."""

    method_code = blueprint.training_method
    method_slot_id = blueprint.training_method_slot_id
    if not method_code or not method_slot_id:
        return output

    candidate_by_id = {
        str(candidate.get("id") or "").strip().lower(): candidate
        for candidate in ranked_candidates
        if str(candidate.get("id") or "").strip()
    }
    slot_by_id = dict(output._selection_slot_by_exercise_id or {})
    method_slot = next(
        (slot for slot in blueprint.slots if slot.slot_id == method_slot_id),
        None,
    )
    method_targets = set(method_slot.target_muscles if method_slot else ())

    selected_plan = None
    for plan in output.exercises:
        exercise_id = str(plan.exercise_id or "").strip().lower()
        candidate = candidate_by_id.get(exercise_id)
        if (
            candidate
            and slot_by_id.get(exercise_id) == method_slot_id
            and candidate_matches_training_method(candidate, method_code)
        ):
            selected_plan = plan
            break

    if selected_plan is None:
        for plan in output.exercises:
            exercise_id = str(plan.exercise_id or "").strip().lower()
            candidate = candidate_by_id.get(exercise_id)
            primary = str(
                (candidate or {}).get("primary_muscle") or ""
            ).strip().lower().replace("_", "-")
            if (
                candidate
                and (not method_targets or primary in method_targets)
                and candidate_matches_training_method(candidate, method_code)
            ):
                selected_plan = plan
                break

    if selected_plan is not None:
        selected_candidate = candidate_by_id.get(
            str(selected_plan.exercise_id or "").strip().lower()
        )
        apply_training_method_to_plan(
            selected_plan,
            method_code,
            source=blueprint.training_method_source,
            reason_code=blueprint.training_method_reason_code,
            candidate=selected_candidate,
        )
    return output

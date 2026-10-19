"""Adapt the small-model slot selection contract into the stable routine schema."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import json
import math
import re
from typing import Any, Mapping, Sequence

from pydantic import ValidationError

from .candidate_ranker import (
    candidate_selection_family_key,
    candidate_selection_pattern_key,
)
from .exercise_display import candidate_display_name
from .equipment_policy import (
    EQUIPMENT_POLICY_VERSION,
    parse_equipment_options,
    superset_equipment_compatible,
)
from .llm_parser import (
    SchemaRepairError,
    classify_validation_error,
    normalize_llm_response,
    parse_json_candidate,
    summarize_validation_error_shapes,
)
from .option_resolver import candidate_matches_adjunct_option
from .prescription.estimator import estimate_routine_time_min
from .prescription.unilateral import restore_unilateral_from_block
from .prescription.metadata import apply_candidate_prescription_metadata
from .prescription.params import GOAL_PARAMS, _goal_key
from .registry import LARGE_MUSCLE_SLUGS
from .routine_blueprint import RoutineBlueprint, build_blueprint_candidate_bundles, _candidate_matches_slot
from .experience_policy import automatic_recommendation_eligibility, main_role_eligibility
from .specialization_policy import SPECIALIZATION_FOCUS_TARGETS, focus_variant_family_key
from .schemas import (
    LLMExercisePlan,
    LLMRoutineOutput,
    LLMRoutineSelectionOutput,
    RoutineDraftResponse,
)


_PHASE_PRIORITY = {
    "warmup_or_preparation": 0,
    "power_if_explicit": 1,
    "main_compound": 2,
    "secondary_compound": 3,
    "isolation_accessory": 4,
    "prehab_accessory": 5,
    "conditioning_finisher": 6,
    "endurance_support": 7,
    "cooldown_recovery": 8,
    # Read-only compatibility for old fixtures; active code must not emit it.
    "finisher_or_recovery": 9,
    "cooldown": 10,
}
_MOVEMENT_PRIORITY = {
    "COMPOUND": 0,
    "ISOLATION": 1,
    "CARRY": 2,
    "STATIC": 3,
    "MOBILITY": 4,
    "CONDITIONING": 5,
}
_NON_MAIN_PHASES = {
    "warmup_or_preparation",
    "prehab_accessory",
    "conditioning_finisher",
    "endurance_support",
    "cooldown_recovery",
    "finisher_or_recovery",
    "cooldown",
}
_REQUIRED_MAIN_PHASES = {"main_compound", "secondary_compound"}

SESSION_EXECUTION_POLICY_VERSION = "session-execution-v1"
SLOT_SELECTION_CONTRACT_VERSION = "slot-selection-v3"

_EXPERIENCE_LABELS = {
    "beginner": "초급자",
    "intermediate": "중급자",
    "advanced": "상급자",
}
_GOAL_LABELS = {
    "hypertrophy": "근비대",
    "strength": "근력",
    "generalfitness": "일반 체력",
    "general_fitness": "일반 체력",
    "power": "파워",
}
_PUBLIC_MUSCLE_LABELS = {
    "chest": "가슴",
    "upper-back": "상부 등",
    "lower-back": "허리/요추",
    "front-deltoids": "전면 어깨",
    "back-deltoids": "후면 어깨",
    "side-deltoids": "측면 어깨",
    "triceps": "삼두",
    "biceps": "이두",
    "forearm": "전완",
    "trapezius": "승모",
    "abs": "복근",
    "obliques": "복사근",
    "quadriceps": "대퇴사두",
    "hamstring": "햄스트링",
    "gluteal": "둔근",
    "calves": "종아리",
    "adductor": "내전근",
    "abductors": "외전근",
    "neck": "목",
}
_PUBLIC_EQUIPMENT_LABELS = {
    "BODYWEIGHT": "맨몸",
    "DUMBBELL": "덤벨",
    "BARBELL": "바벨",
    "MACHINE": "머신",
    "CABLE": "케이블",
    "SMITH_MACHINE": "스미스 머신",
    "KETTLEBELL": "케틀벨",
    "LANDMINE": "랜드마인",
    "BAND": "밴드",
    "PLATE": "원판",
}
_HEALTH_RATIONALE_TOKENS = (
    "부상",
    "근육통",
    "통증",
    "가동",
    "관절",
    "요추",
    "허리",
    "무릎",
    "손목",
    "팔꿈치",
    "발목",
)
_MATERIAL_RATIONALE_TOKENS = (
    "체형",
    "대퇴 길이",
    "컨디션",
    "사용 가능한 장비",
    "장비 제약",
    "제한 조건",
    "최근 수행",
    "반복 노출",
)


class _JsonObjectPairs(list):
    """Preserve duplicate JSON keys for deterministic small-model repair."""


def _convert_object_pairs(value: Any) -> Any:
    if isinstance(value, _JsonObjectPairs):
        return {str(key): _convert_object_pairs(item) for key, item in value}
    if isinstance(value, list):
        return [_convert_object_pairs(item) for item in value]
    return value


def _selection_text(value: Any) -> str:
    """Coerce common small-model text shapes without trusting their structure."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    if isinstance(value, dict):
        for key in ("text", "value", "reason", "rationale", "summary", "title", "name", "id"):
            if key in value:
                text = _selection_text(value.get(key))
                if text:
                    return text
        parts = [_selection_text(item) for item in value.values()]
        return " ".join(part for part in parts if part).strip()
    if isinstance(value, (list, tuple, set)):
        parts = [_selection_text(item) for item in value]
        return " ".join(part for part in parts if part).strip()
    return ""


def _selection_text_list(value: Any) -> list[str]:
    items = value if isinstance(value, list) else ([] if value is None else [value])
    normalized: list[str] = []
    for item in items:
        text = _selection_text(item)
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _normalize_selection_contract_data(data: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Repair bounded type drift in the small-model selection-only contract."""
    normalized = dict(data)
    repair_count = 0

    for camel_key, snake_key in (
        ("summaryTitle", "summary_title"),
        ("rationaleSummary", "rationale_summary"),
        ("warnings", "warnings"),
    ):
        source_key = camel_key if camel_key in normalized else snake_key
        if source_key not in normalized:
            continue
        original = normalized.get(source_key)
        repaired = (
            _selection_text_list(original)
            if camel_key in {"rationaleSummary", "warnings"}
            else _selection_text(original)
        )
        if repaired != original:
            repair_count += 1
        normalized[camel_key] = repaired

    raw_selections = normalized.get("selections")
    if isinstance(raw_selections, dict):
        if any(key in raw_selections for key in ("slotId", "slot_id", "exerciseId", "exercise_id")):
            selection_items: list[Any] = [raw_selections]
        else:
            selection_items = [item for item in raw_selections.values() if isinstance(item, dict)]
        repair_count += 1
    elif isinstance(raw_selections, list):
        selection_items = raw_selections
    elif raw_selections is None:
        selection_items = []
    else:
        selection_items = []
        repair_count += 1

    repaired_selections: list[dict[str, str]] = []
    for item in selection_items:
        if not isinstance(item, dict):
            repair_count += 1
            continue
        slot_raw = item.get("slotId", item.get("slot_id"))
        exercise_raw = item.get("exerciseId", item.get("exercise_id"))
        rationale_raw = item.get("exerciseRationale", item.get("exercise_rationale"))
        slot_id = _selection_text(slot_raw)
        exercise_id = _selection_text(exercise_raw)
        rationale = _selection_text(rationale_raw)
        if not slot_id or not exercise_id:
            repair_count += 1
            continue
        repaired = {
            "slotId": slot_id,
            "exerciseId": exercise_id,
            "exerciseRationale": rationale,
        }
        if repaired != item:
            repair_count += 1
        repaired_selections.append(repaired)
    normalized["selections"] = repaired_selections
    return normalized, repair_count


def _recover_duplicate_selection_objects(raw_text: str) -> tuple[dict[str, Any] | None, int]:
    """Split repeated slot/exercise key triplets that a small model put in one object."""
    text = str(raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    if not text.startswith("{"):
        return None, 0
    try:
        root = json.loads(text, object_pairs_hook=_JsonObjectPairs)
    except json.JSONDecodeError:
        return None, 0
    if not isinstance(root, _JsonObjectPairs):
        return None, 0

    root_data: dict[str, Any] = {}
    repair_count = 0
    for key, value in root:
        if str(key) != "selections" or not isinstance(value, list):
            root_data[str(key)] = _convert_object_pairs(value)
            continue
        recovered: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, _JsonObjectPairs):
                converted = _convert_object_pairs(item)
                if isinstance(converted, dict):
                    recovered.append(converted)
                continue
            current: dict[str, Any] = {}
            for item_key, item_value in item:
                key_text = str(item_key)
                starts_new_selection = key_text in {"slotId", "slot_id"} and bool(current)
                repeats_current_key = key_text in current
                if starts_new_selection or repeats_current_key:
                    recovered.append(current)
                    current = {}
                    repair_count += 1
                current[key_text] = _convert_object_pairs(item_value)
            if current:
                recovered.append(current)
        root_data["selections"] = recovered
    return (root_data, repair_count) if repair_count else (None, 0)


def _candidate_name(candidate: dict[str, Any]) -> str:
    return candidate_display_name(candidate)


def _candidate_equipment(candidate: dict[str, Any]) -> str | None:
    selected = str(candidate.get("selected_equipment_code") or "").strip().upper()
    if selected:
        return selected
    options = parse_equipment_options(candidate.get("equipment_req"))
    return options[0] if options else None


def _candidate_family(candidate: dict[str, Any], blueprint: RoutineBlueprint | None = None) -> str:
    family = candidate_selection_family_key(candidate)
    if blueprint is None or blueprint.readiness_level == "low":
        return family
    from .pull_composition import reviewed_pull_family_key

    family = reviewed_pull_family_key(
        candidate, family, enabled=blueprint.formula_version == "gym-routine-blueprint-v11"
    )
    return focus_variant_family_key(candidate, blueprint.specialization_focus, family)


def _candidate_pattern(candidate: dict[str, Any]) -> str:
    return candidate_selection_pattern_key(candidate)


def is_redundant_push_front_raise(
    split: str | None,
    focus: str | None,
    candidate_pattern: str,
    selected_patterns: Sequence[str] | set[str],
    candidates: Sequence[dict[str, Any]] = (),
) -> bool:
    """Composition preference, not a medical exclusion of front raises."""
    alternative_available = any(
        (_candidate_pattern(candidate), candidate.get("primary_muscle"))
        in {("lateral_raise", "side-deltoids"), ("chest_fly", "chest")}
        and str(candidate.get("prescription_mode") or "reps").lower() == "reps"
        for candidate in candidates
    )
    return bool(
        split == "push" and focus != "frontDeltoidEmphasis"
        and candidate_pattern == "front_raise"
        and ("vertical_press" in selected_patterns
             or ("horizontal_press" in selected_patterns and alternative_available))
    )


def align_candidate_bundles_with_composition(
    blueprint: RoutineBlueprint,
    bundles: Sequence[dict[str, Any]],
    ranked_candidates: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Hide offers rejected after every possible choice in a prior required slot.

    Optional or mixed prior slots are not proof of redundancy. Flex stays
    server-owned and retains its selection-time composition checks.
    """
    by_id = {str(c.get("id") or "").strip().lower(): c for c in ranked_candidates}
    specs = {slot.slot_id: slot for slot in blueprint.slots}
    required_choices: list[list[dict[str, Any]]] = []
    aligned = []
    for bundle in bundles:
        slot_id = bundle.get("slotId")
        offers = list(bundle.get("candidates") or [])
        if slot_id != "flex":
            offers = [offer for offer in offers if not any(
                all(is_redundant_push_front_raise(
                    blueprint.target_split_label, blueprint.specialization_focus,
                    _candidate_pattern(by_id.get(str(offer.get("exerciseId") or "").strip().lower(), {})),
                    {_candidate_pattern(prior)}, ranked_candidates,
                ) for prior in prior_choices)
                for prior_choices in required_choices
            )]
        aligned_bundle = {**bundle, "candidates": offers}
        if "slotUnfilled" in bundle:
            aligned_bundle["slotUnfilled"] = not offers
        aligned.append(aligned_bundle)
        spec = specs.get(slot_id)
        if spec and spec.requirement == "required_when_safe" and offers:
            required_choices.append([
                by_id.get(str(offer.get("exerciseId") or "").strip().lower(), {})
                for offer in offers
            ])
    return aligned


def selection_pattern_limit(pattern: str, blueprint: RoutineBlueprint) -> int | None:
    """Return the product diversity cap for a coarse movement pattern.

    These caps prevent near-duplicate variations from being added merely to
    satisfy a count heuristic. They are routine-composition defaults, not
    claims that a second variation is physiologically unsafe.
    """
    if pattern == "horizontal_press":
        return (
            2
            if blueprint.target_split_label in {"push", "upper", "chest_back", "chest"}
            else 1
        )
    if pattern == "horizontal_row" and getattr(blueprint, "specialization_focus", None) == "midBackEmphasis":
        return 2
    if pattern == "biceps_curl":
        if getattr(blueprint, "specialization_focus", None) == "bicepsEmphasis" and getattr(blueprint, "readiness_level", "normal") != "low":
            return 3
        return 2 if blueprint.target_split_label == "arm" else 1
    if pattern == "lateral_raise" and getattr(blueprint, "specialization_focus", None) == "sideDeltoidEmphasis" and getattr(blueprint, "readiness_level", "normal") != "low":
        return 2
    if pattern in {
        "chest_fly",
        "rear_delt_fly",
        "lateral_raise",
        "front_raise",
        "vertical_pull",
        "leg_curl",
    }:
        return 1
    return None


def _candidate_plan(candidate: dict[str, Any], rationale: str, *, goal: str) -> LLMExercisePlan:
    primary = str(candidate.get("primary_muscle") or "").strip()
    plan = LLMExercisePlan(
        exercise_id=str(candidate.get("id") or ""),
        exercise_name=_candidate_name(candidate),
        movement_pattern=str(candidate.get("movement_pattern") or "") or None,
        movement_type=str(candidate.get("movement_type") or "") or None,
        primary_muscles=[primary] if primary else [],
        equipment_type=_candidate_equipment(candidate),
        target_weight_kg=None,
        target_reps=10,
        # IDs-only model selection uses the same base dose as fallback before
        # experience, readiness, DOMS, and budget adjustments are applied.
        sets=GOAL_PARAMS.get(_goal_key(goal), GOAL_PARAMS["hypertrophy"])["sets"],
        rest_time_sec=90,
        target_rir=2,
        exercise_rationale=rationale,
        substitution_candidates=[],
    )
    return apply_candidate_prescription_metadata(plan, candidate)


def _candidate_public_rationale(
    candidate: dict[str, Any],
    *,
    purpose: str | None = None,
) -> str:
    """Build a factual rationale from the accepted candidate, not model prose."""
    primary_key = str(candidate.get("primary_muscle") or "").strip().lower()
    primary = _PUBLIC_MUSCLE_LABELS.get(primary_key)
    equipment = _PUBLIC_EQUIPMENT_LABELS.get(_candidate_equipment(candidate) or "")
    purpose_text = str(purpose or "").strip()
    trusted_purpose = purpose_text if re.search(r"[가-힣]", purpose_text) else ""

    if primary and equipment:
        exercise_context = f"{primary} 중심의 {equipment} 운동"
    elif primary:
        exercise_context = f"{primary} 중심 운동"
    elif equipment:
        exercise_context = f"{equipment}을 사용하는 운동"
    else:
        exercise_context = "선택된 운동"
    context = (
        f"{exercise_context}으로 {trusted_purpose}에 맞춰 배치했습니다."
        if trusted_purpose
        else f"{exercise_context}으로 목표 부위 구성을 보완하도록 배치했습니다."
    )

    # Keep health cautions ahead of other constraints; target labels alone are not cautions.
    health_reasons: list[str] = []
    constraint_reasons: list[str] = []
    for item in candidate.get("score_display_reasons") or []:
        reason = str(item).strip()
        if not reason or reason.startswith("운동 효율 등급"):
            continue
        reason = reason.replace("안전 대체 후보", "부담이 낮은 대체 후보")
        if any(token in reason for token in _HEALTH_RATIONALE_TOKENS):
            if reason not in health_reasons:
                health_reasons.append(reason)
        elif any(token in reason for token in _MATERIAL_RATIONALE_TOKENS):
            if reason not in constraint_reasons:
                constraint_reasons.append(reason)
    display_reasons = health_reasons + constraint_reasons
    if not display_reasons:
        return context
    return " ".join([context, *display_reasons[:2]])


def _deterministic_rationale_summary(blueprint: RoutineBlueprint) -> list[str]:
    experience = _EXPERIENCE_LABELS.get(
        str(blueprint.experience_level or "").strip().lower(),
        "사용자",
    )
    goal_key = re.sub(r"[-\s]+", "_", str(blueprint.goal or "").strip().lower())
    goal = _GOAL_LABELS.get(goal_key, "운동")
    return [
        f"{experience} 숙련도와 {goal} 목표를 반영해 운동 순서와 구성을 조정했습니다."
    ]


def _inferred_candidate_phase(candidate: dict[str, Any]) -> str:
    preferred_phase = str(candidate.get("preferred_session_phase") or "").strip()
    if preferred_phase:
        return preferred_phase
    movement_type = str(candidate.get("movement_type") or "").strip().upper()
    if movement_type == "COMPOUND":
        return "secondary_compound"
    if movement_type == "ISOLATION":
        return "isolation_accessory"
    if movement_type == "CONDITIONING":
        return "conditioning_finisher"
    if movement_type in {"STATIC", "MOBILITY"}:
        return "cooldown_recovery"
    return "isolation_accessory"


def _resolved_slot_phase(candidate: dict[str, Any], slot_phase: str | None) -> str:
    """Keep a slot choice inside the candidate's DB-approved phase contract."""
    requested = str(slot_phase or "").strip()
    if not requested:
        return _inferred_candidate_phase(candidate)
    movement = str(candidate.get("movement_type") or "").strip().upper()
    if requested in {"main_compound", "secondary_compound"} and movement != "COMPOUND":
        return _inferred_candidate_phase(candidate)
    allowed = {
        item.strip()
        for item in str(candidate.get("allowed_session_phases") or "").split(",")
        if item.strip()
    }
    if not allowed or requested in allowed:
        return requested
    inferred = _inferred_candidate_phase(candidate)
    return inferred if inferred in allowed else requested


def order_routine_output_by_selection_phase(
    output: LLMRoutineOutput, *, group_primary_muscles: bool = False,
) -> LLMRoutineOutput:
    """Keep phase/movement priority and group standard work by primary muscle.

    Primary grouping is opt-in after dose/time planning, so it cannot affect
    selection, trimming, or prospective superset/circuit composition.
    """
    phase_by_id = dict(getattr(output, "_selection_phase_by_exercise_id", {}) or {})

    execution_metadata = dict(
        getattr(output, "_execution_metadata_by_exercise_id", {}) or {}
    )
    if group_primary_muscles and any(
        str(metadata.get("executionGroupType") or "single").strip().lower()
        in {"superset", "circuit", "density"}
        for metadata in execution_metadata.values()
        if isinstance(metadata, Mapping)
    ):
        return output

    def resolved_phase(exercise: LLMExercisePlan) -> str:
        exercise_id = str(exercise.exercise_id or "").strip().lower()
        return phase_by_id.get(exercise_id) or _inferred_candidate_phase(
            {"movement_type": exercise.movement_type}
        )

    def unambiguous_primary(exercise: LLMExercisePlan) -> str | None:
        primaries = [
            str(primary or "").strip().lower().replace("_", "-")
            for primary in (exercise.primary_muscles or [])
            if str(primary or "").strip()
        ]
        if len(primaries) != 1 or primaries[0] in {"unknown", "none", "n/a", "na"}:
            return None
        return primaries[0]

    # Use source positions as group priorities so focus ordering encoded in
    # selection order remains stable, while ambiguous primaries stay distinct.
    primary_group_first_index: dict[tuple[str, str, str], int] = {}
    ordering_context: dict[int, tuple[str, int]] = {}
    for original_index, exercise in enumerate(output.exercises):
        phase = resolved_phase(exercise)
        primary = unambiguous_primary(exercise)
        group_index = original_index
        if group_primary_muscles and primary:
            group_index = primary_group_first_index.setdefault(
                (phase, str(exercise.movement_type or "").strip().upper(), primary),
                original_index,
            )
        ordering_context[original_index] = (phase, group_index)

    def order_key(item: tuple[int, LLMExercisePlan]) -> tuple[int, int, int, int]:
        original_index, exercise = item
        movement_type = str(exercise.movement_type or "").strip().upper()
        phase, primary_group_index = ordering_context[original_index]
        return (
            _PHASE_PRIORITY.get(phase, 99),
            _MOVEMENT_PRIORITY.get(movement_type, 99),
            primary_group_index,
            original_index,
        )

    output.exercises = [
        exercise
        for _, exercise in sorted(enumerate(output.exercises), key=order_key)
    ]
    if group_primary_muscles:
        for index, exercise in enumerate(output.exercises, 1):
            metadata = output._execution_metadata_by_exercise_id.get(
                str(exercise.exercise_id or "").strip().lower()
            )
            if metadata and metadata.get("executionGroupType", "single") == "single":
                metadata["executionGroupId"] = f"S{index}"
    return output


def _exercise_primary(exercise: LLMExercisePlan) -> str:
    return str((exercise.primary_muscles or [""])[0] or "").strip().lower().replace("_", "-")


def _execution_phase(output: LLMRoutineOutput, exercise: LLMExercisePlan) -> str:
    phase_by_id = dict(output._selection_phase_by_exercise_id or {})
    exercise_id = str(exercise.exercise_id or "").strip().lower()
    return phase_by_id.get(exercise_id) or _inferred_candidate_phase(
        {
            "movement_type": exercise.movement_type,
        }
    )


def _required_main_execution_indices(
    output: LLMRoutineOutput,
    blueprint: RoutineBlueprint,
) -> set[int]:
    """Return selected required-main slots that the policy keeps as singles."""
    if blueprint.execution_policy.main_lift_group_type != "single":
        return set()
    required_main_slots = {
        slot.slot_id
        for slot in blueprint.slots
        if (
            (slot.requirement == "required_when_safe"
             or (blueprint.formula_version == "gym-routine-blueprint-v10"
                 and slot.slot_id == "upper_vertical_pull"))
            and slot.phase in _REQUIRED_MAIN_PHASES
        )
    }
    slot_by_id = dict(output._selection_slot_by_exercise_id or {})
    return {
        index
        for index, exercise in enumerate(output.exercises)
        if str(exercise.exercise_id or "").strip().lower() in slot_by_id
        and slot_by_id[str(exercise.exercise_id or "").strip().lower()]
        in required_main_slots
    }


def _find_antagonist_main_pair(
    output: LLMRoutineOutput,
    phases: list[str],
) -> tuple[int, int] | None:
    chest_indices = [
        index
        for index, exercise in enumerate(output.exercises)
        if phases[index] in {"main_compound", "secondary_compound"}
        and _exercise_primary(exercise) == "chest"
    ]
    back_indices = [
        index
        for index, exercise in enumerate(output.exercises)
        if phases[index] in {"main_compound", "secondary_compound"}
        and _exercise_primary(exercise) in {"upper-back", "trapezius"}
    ]
    for chest_index in chest_indices:
        for back_index in back_indices:
            if superset_equipment_compatible(
                output.exercises[chest_index].equipment_type,
                output.exercises[back_index].equipment_type,
            ):
                return (min(chest_index, back_index), max(chest_index, back_index))
    return None


def _effective_grouped_sets(exercise: LLMExercisePlan, grouped_set_cap: int | None) -> int:
    """Return the rounds an exercise would retain if it entered a group."""

    sets = int(exercise.sets or 0)
    return min(sets, grouped_set_cap) if grouped_set_cap is not None else sets


def _execution_group_is_feasible(
    output: LLMRoutineOutput,
    indices: Sequence[int],
    *,
    grouped_set_cap: int | None,
) -> bool:
    """Require compatible equipment and one positive round count for a group."""

    if len(indices) < 2:
        return False
    effective_sets = [
        _effective_grouped_sets(output.exercises[index], grouped_set_cap)
        for index in indices
    ]
    if effective_sets[0] <= 0 or len(set(effective_sets)) != 1:
        return False
    return all(
        superset_equipment_compatible(
            output.exercises[first].equipment_type,
            output.exercises[second].equipment_type,
        )
        for position, first in enumerate(indices)
        for second in indices[position + 1 :]
    )


def _contiguous_feasible_accessory_groups(
    output: LLMRoutineOutput,
    phases: Sequence[str],
    protected_main_indices: set[int],
    *,
    max_group_size: int,
    grouped_set_cap: int | None,
) -> list[list[int]]:
    """Build stable, non-interleaving accessory groups from adjacent exercises."""

    def is_groupable(index: int) -> bool:
        return (
            index not in protected_main_indices
            and phases[index] in {"secondary_compound", "isolation_accessory"}
        )

    groups: list[list[int]] = []
    index = 0
    while index < len(output.exercises):
        if not is_groupable(index):
            index += 1
            continue
        candidate_indices = [index]
        next_index = index + 1
        while (
            next_index < len(output.exercises)
            and len(candidate_indices) < max_group_size
            and is_groupable(next_index)
            and _execution_group_is_feasible(
                output,
                [*candidate_indices, next_index],
                grouped_set_cap=grouped_set_cap,
            )
        ):
            candidate_indices.append(next_index)
            next_index += 1
        if len(candidate_indices) >= 2:
            groups.append(candidate_indices)
            index = next_index
        else:
            index += 1
    return groups


def _candidate_execution_primary(candidate: dict[str, Any]) -> str:
    return str(candidate.get("primary_muscle") or "").strip().lower().replace("_", "-")


def _accessory_group_options(blueprint, bundles, ranked_candidates):
    """Bounded pair proposals from admitted slots, not new exercise admission."""
    if blueprint.execution_policy.structure not in {"circuit", "density"}:
        return []
    pool = {str(c.get("id") or "").strip().lower(): c for c in ranked_candidates}
    slots = {slot.slot_id: slot for slot in blueprint.slots}
    reserved_ids = {
        str(item.get("exerciseId") or "").strip().lower()
        for bundle in bundles
        if bundle.get("requirement") == "required_when_safe"
        for item in _bundle_candidates(bundle)
    }
    choices = []
    for bundle in bundles:
        slot_id = str(bundle.get("slotId") or "")
        slot = slots.get(slot_id)
        if slot_id != "flex" and not (
            slot and slot.requirement in {"preferred_when_safe", "preferred_when_requested_and_safe"}
            and slot.phase in {"secondary_compound", "isolation_accessory"}
            and not slot.adjunct_option
            and slot_id not in {blueprint.specialization_slot_id, blueprint.training_method_slot_id}
        ):
            continue
        # Explicit bundles already have their own small limit. Flex remains
        # server-owned and only uses the first 64 admitted isolation candidates.
        for item in _bundle_candidates(bundle)[:64]:
            id = str(item.get("exerciseId") or "").strip().lower()
            candidate = pool.get(id)
            if not candidate or id in reserved_ids:
                continue
            if not automatic_recommendation_eligibility(candidate, blueprint.experience_level).eligible:
                continue
            movement = str(candidate.get("movement_type") or "").upper()
            if movement not in {"COMPOUND", "ISOLATION"}:
                continue
            primary = _candidate_execution_primary(candidate)
            if primary in blueprint.safety_limited_target_muscles:
                continue
            if slot_id == "flex" and (movement != "ISOLATION" or primary not in blueprint.target_muscles):
                continue
            choices.append((slot_id, candidate))
    pairs = []
    for first, second in combinations(choices, 2):
        slot_a, a = first
        slot_b, b = second
        if slot_a == slot_b and slot_a != "flex":
            continue
        if (str(a['id']).lower() == str(b['id']).lower()
                or _candidate_execution_primary(a) == _candidate_execution_primary(b)
                or (_candidate_family(a, blueprint)
                    and _candidate_family(a, blueprint) == _candidate_family(b, blueprint))):
            continue
        if superset_equipment_compatible(_candidate_equipment(a), _candidate_equipment(b)):
            pairs.append((first, second))
    return sorted(pairs, key=lambda pair: (
        # Upper-session main presses/rows do not replace direct arm work.
        # Prefer the requested arm targets before an extra anterior raise.
        -len({_candidate_execution_primary(c) for _, c in pair}
             & {'biceps', 'triceps'} & set(blueprint.target_muscles))
        if blueprint.target_split_label == 'upper' else 0,
        sum(slot == 'flex' for slot, _ in pair),
        -sum(int(candidate.get('score') or 0) for _, candidate in pair),
        tuple((slot, str(candidate['id'])) for slot, candidate in pair),
    ))


def render_accessory_group_plan(blueprint, bundles, ranked_candidates) -> str:
    pairs = _accessory_group_options(blueprint, bundles, ranked_candidates)
    if not pairs:
        return ""
    examples = [[{'slotId': slot, 'exerciseId': candidate['id']}
                 for slot, candidate in pair] for pair in pairs[:3]]
    return (
        "[ACCESSORY GROUP OPTIONS]\n"
        "Prefer one compatible pair below after required main lifts. Required main lifts stay single. "
        "Flex is server-owned; do not emit flex selections. These are candidate proposals, "
        "not final groups: server dose, time and equipment validation remains authoritative.\n"
        + json.dumps(examples, ensure_ascii=False, separators=(',', ':'))
    )


def complete_accessory_group_selection(output, blueprint, bundles, ranked_candidates):
    """Choose the least disruptive pair before dose/time enforcement, or leave singles."""
    output = order_routine_output_by_selection_phase(output)
    if blueprint.execution_policy.structure not in {"circuit", "density"}:
        return output
    policy = blueprint.execution_policy

    def has_group(value):
        return bool(_contiguous_feasible_accessory_groups(
            value, [_execution_phase(value, ex) for ex in value.exercises],
            _required_main_execution_indices(value, blueprint),
            max_group_size=policy.max_group_size, grouped_set_cap=policy.grouped_set_cap,
        ))

    if has_group(output):
        return output
    pool = {str(c.get('id') or '').strip().lower(): c for c in ranked_candidates}
    if any(str(ex.exercise_id).strip().lower() not in pool for ex in output.exercises):
        return output
    slots = {slot.slot_id: slot for slot in blueprint.slots}
    slot_by_id = dict(output._selection_slot_by_exercise_id or {})
    flex_indices = [i for i, ex in enumerate(output.exercises)
                    if slot_by_id.get(str(ex.exercise_id).strip().lower(), 'flex') == 'flex'
                    and _execution_phase(output, ex) == 'isolation_accessory'
                    and str(ex.exercise_id).strip().lower() not in output._adjunct_option_by_exercise_id]
    best = None
    for pair in _accessory_group_options(blueprint, bundles, ranked_candidates):
        trial = output.model_copy(deep=True)
        replaced, additions, replacements = set(), 0, 0
        for slot_id, candidate in pair:
            id = str(candidate['id']).strip().lower()
            existing = next((i for i, ex in enumerate(trial.exercises)
                             if str(ex.exercise_id).strip().lower() == id), None)
            if existing is not None:
                continue
            if slot_id == 'flex':
                pair_ids = {str(c['id']).strip().lower() for _, c in pair}
                index = next((i for i in flex_indices if i not in replaced
                              and str(trial.exercises[i].exercise_id).strip().lower() not in pair_ids), None)
            else:
                index = next((i for i, ex in enumerate(trial.exercises)
                              if slot_by_id.get(str(ex.exercise_id).strip().lower()) == slot_id), None)
            plan = _candidate_plan(candidate, _candidate_public_rationale(candidate,
                purpose='요청한 보조 운동 묶음의 장비 호환성 보완'), goal=blueprint.goal)
            if index is None:
                trial.exercises.append(plan)
                additions += 1
            else:
                old_id = str(trial.exercises[index].exercise_id).strip().lower()
                for attr in ('_selection_phase_by_exercise_id', '_selection_slot_by_exercise_id'):
                    getattr(trial, attr).pop(old_id, None)
                trial.exercises[index] = plan
                replaced.add(index)
                replacements += 1
            trial._selection_phase_by_exercise_id[id] = _resolved_slot_phase(
                candidate, slots[slot_id].phase if slot_id in slots else None)
            trial._selection_slot_by_exercise_id[id] = slot_id
        if len(trial.exercises) > blueprint.target_exercise_count:
            continue
        trial_ids = [str(ex.exercise_id).strip().lower() for ex in trial.exercises]
        if len(trial_ids) != len(set(trial_ids)) or not _replacement_preserves_selection_diversity(
            trial, blueprint, pool, set(), [],
        ):
            continue
        patterns = {_candidate_pattern(pool[id]) for id in trial_ids}
        if 'front_raise' in patterns and is_redundant_push_front_raise(
            blueprint.target_split_label, blueprint.specialization_focus, 'front_raise', patterns,
            ranked_candidates,
        ):
            continue
        trial = order_routine_output_by_selection_phase(trial)
        if not has_group(trial):
            continue
        cost = (replacements + additions, replacements, additions)
        if best is None or cost < best[0]:
            best = (cost, trial, replacements, additions)
    if best is None:
        return output
    _, trial, replacements, additions = best
    trial._selection_diagnostics.update(
        accessory_group_replacement_count=replacements,
        accessory_group_addition_count=additions,
    )
    trial.warnings = list(dict.fromkeys([*trial.warnings,
        '요청한 보조 운동 묶음을 구성하기 위해 허용된 후보 안에서 장비 조합을 보완했습니다. '
        '실제 묶음과 분량은 최종 세트·시간 검사를 거쳐 결정됩니다.']))
    return trial


def _is_antagonist_main_candidate(candidate: dict[str, Any]) -> bool:
    return (
        str(candidate.get("movement_type") or "").strip().upper() == "COMPOUND"
        and _candidate_execution_primary(candidate) in {"chest", "upper-back", "trapezius"}
    )


def prioritize_antagonist_superset_candidates(
    candidates: Sequence[dict[str, Any]],
    blueprint: RoutineBlueprint,
) -> list[dict[str, Any]]:
    """Put one executable chest/back pair inside the fallback main window.

    Candidate safety and availability have already been decided upstream. This
    helper only changes ordering when the user explicitly requested the
    chest/back antagonist superset contract.
    """
    ordered = list(candidates)
    policy = blueprint.execution_policy
    if not (
        policy.structure == "superset"
        and policy.pair_antagonist_main_lifts
    ):
        return ordered

    chest_candidates = [
        candidate
        for candidate in ordered
        if _is_antagonist_main_candidate(candidate)
        and _candidate_execution_primary(candidate) == "chest"
    ]
    back_candidates = [
        candidate
        for candidate in ordered
        if _is_antagonist_main_candidate(candidate)
        and _candidate_execution_primary(candidate) in {"upper-back", "trapezius"}
    ]
    for chest in chest_candidates:
        for back in back_candidates:
            if not superset_equipment_compatible(
                _candidate_equipment(chest),
                _candidate_equipment(back),
            ):
                continue
            promoted_ids = {
                str(chest.get("id") or "").strip().lower(),
                str(back.get("id") or "").strip().lower(),
            }
            return [
                chest,
                back,
                *[
                    candidate
                    for candidate in ordered
                    if str(candidate.get("id") or "").strip().lower()
                    not in promoted_ids
                ],
            ]
    return ordered


def _replacement_preserves_selection_diversity(
    output: LLMRoutineOutput,
    blueprint: RoutineBlueprint,
    candidate_by_id: dict[str, dict[str, Any]],
    replacement_indices: set[int],
    replacements: Sequence[dict[str, Any]],
) -> bool:
    remaining_candidates = [
        candidate_by_id.get(str(exercise.exercise_id or "").strip().lower())
        for index, exercise in enumerate(output.exercises)
        if index not in replacement_indices
    ]
    combined = [
        candidate
        for candidate in [*remaining_candidates, *replacements]
        if candidate
    ]
    families = [family for family in (_candidate_family(item, blueprint) for item in combined) if family]
    if len(families) != len(set(families)):
        return False

    pattern_counts: dict[str, int] = defaultdict(int)
    for candidate in combined:
        pattern = _candidate_pattern(candidate)
        if pattern:
            pattern_counts[pattern] += 1
    return all(
        limit is None or pattern_counts.get(pattern, 0) <= limit
        for pattern in pattern_counts
        for limit in [selection_pattern_limit(pattern, blueprint)]
    )


def _replace_selected_exercise(
    output: LLMRoutineOutput,
    index: int,
    candidate: dict[str, Any],
    *,
    goal: str,
) -> tuple[str, str]:
    old_id = str(output.exercises[index].exercise_id or "").strip().lower()
    new_id = str(candidate.get("id") or "").strip().lower()
    output.exercises[index] = _candidate_plan(
        candidate,
        "요청한 가슴·등 슈퍼세트를 실제 장비 동선 안에서 구성하기 위해 "
        "장비 호환성을 확인한 운동으로 조정했습니다.",
        goal=goal,
    )
    for attr_name in (
        "_selection_phase_by_exercise_id",
        "_selection_slot_by_exercise_id",
        "_adjunct_option_by_exercise_id",
    ):
        values = dict(getattr(output, attr_name, {}) or {})
        previous = values.pop(old_id, None)
        if previous is not None:
            values[new_id] = previous
        setattr(output, attr_name, values)
    return old_id, new_id


def _replacement_matches_destination(output, blueprint, index, candidate):
    if not automatic_recommendation_eligibility(candidate, blueprint.experience_level).eligible:
        return False
    old_id = str(output.exercises[index].exercise_id or "").strip().lower()
    slot_id = (getattr(output, "_selection_slot_by_exercise_id", {}) or {}).get(old_id)
    slot = next((slot for slot in blueprint.slots if slot.slot_id == slot_id), None)
    if slot is not None:
        return _candidate_matches_slot(candidate, slot,
            experience_level=blueprint.experience_level, goal=blueprint.goal,
            advanced_chest_focus=blueprint.advanced_chest_focus,
            specialization_focus=blueprint.specialization_focus,
            training_method=blueprint.training_method,
            specialization_slot_id=blueprint.specialization_slot_id,
            training_method_slot_id=blueprint.training_method_slot_id)
    return (
        _execution_phase(output, output.exercises[index]) != "main_compound"
        or main_role_eligibility(candidate, blueprint.experience_level, blueprint.goal).eligible
    )


def ensure_antagonist_superset_pair(
    output: LLMRoutineOutput,
    blueprint: RoutineBlueprint,
    ranked_candidates: Sequence[dict[str, Any]],
) -> LLMRoutineOutput:
    """Repair a requested chest/back superset with the fewest safe replacements.

    The LLM selects exercises, but an executable grouping contract is
    server-owned. When selected main lifts require two unrelated machines, a
    compatible already-filtered candidate replaces one selected lift. We do
    not relax the conservative station policy.
    """
    policy = blueprint.execution_policy
    if not (
        policy.structure == "superset"
        and policy.pair_antagonist_main_lifts
        and output.exercises
    ):
        return output

    phases = [_execution_phase(output, exercise) for exercise in output.exercises]
    if _find_antagonist_main_pair(output, phases):
        return output

    candidate_by_id = {
        str(candidate.get("id") or "").strip().lower(): candidate
        for candidate in ranked_candidates
        if str(candidate.get("id") or "").strip()
    }
    selected_ids = {
        str(exercise.exercise_id or "").strip().lower()
        for exercise in output.exercises
    }
    chest_indices = [
        index
        for index, exercise in enumerate(output.exercises)
        if phases[index] in {"main_compound", "secondary_compound"}
        and _exercise_primary(exercise) == "chest"
    ]
    back_indices = [
        index
        for index, exercise in enumerate(output.exercises)
        if phases[index] in {"main_compound", "secondary_compound"}
        and _exercise_primary(exercise) in {"upper-back", "trapezius"}
    ]
    if not chest_indices or not back_indices:
        return output

    one_replacement: tuple[int, dict[str, Any]] | None = None
    for candidate in ranked_candidates:
        candidate_id = str(candidate.get("id") or "").strip().lower()
        if (
            not candidate_id
            or candidate_id in selected_ids
            or not _is_antagonist_main_candidate(candidate)
        ):
            continue
        primary = _candidate_execution_primary(candidate)
        replacement_indices = chest_indices if primary == "chest" else back_indices
        opposite_indices = back_indices if primary == "chest" else chest_indices
        for opposite_index in opposite_indices:
            if not superset_equipment_compatible(
                _candidate_equipment(candidate),
                output.exercises[opposite_index].equipment_type,
            ):
                continue
            for replacement_index in replacement_indices:
                if not _replacement_matches_destination(output, blueprint, replacement_index, candidate):
                    continue
                if _replacement_preserves_selection_diversity(
                    output,
                    blueprint,
                    candidate_by_id,
                    {replacement_index},
                    [candidate],
                ):
                    one_replacement = (replacement_index, candidate)
                    break
            if one_replacement:
                break
        if one_replacement:
            break

    replacements: list[tuple[int, dict[str, Any]]] = []
    if one_replacement:
        replacements.append(one_replacement)
    else:
        available_chest = [
            candidate
            for candidate in ranked_candidates
            if str(candidate.get("id") or "").strip().lower() not in selected_ids
            and _is_antagonist_main_candidate(candidate)
            and _candidate_execution_primary(candidate) == "chest"
        ]
        available_back = [
            candidate
            for candidate in ranked_candidates
            if str(candidate.get("id") or "").strip().lower() not in selected_ids
            and _is_antagonist_main_candidate(candidate)
            and _candidate_execution_primary(candidate) in {"upper-back", "trapezius"}
        ]
        for chest in available_chest:
            for back in available_back:
                if not (
                    _replacement_matches_destination(output, blueprint, chest_indices[0], chest)
                    and _replacement_matches_destination(output, blueprint, back_indices[0], back)
                ):
                    continue
                if not superset_equipment_compatible(
                    _candidate_equipment(chest),
                    _candidate_equipment(back),
                ):
                    continue
                indices = {chest_indices[0], back_indices[0]}
                if _replacement_preserves_selection_diversity(
                    output,
                    blueprint,
                    candidate_by_id,
                    indices,
                    [chest, back],
                ):
                    replacements = [
                        (chest_indices[0], chest),
                        (back_indices[0], back),
                    ]
                    break
            if replacements:
                break

    if not replacements:
        return output

    replacement_evidence = []
    for index, candidate in replacements:
        old_id, new_id = _replace_selected_exercise(output, index, candidate, goal=blueprint.goal)
        replacement_evidence.append(
            {"replacedExerciseId": old_id, "replacementExerciseId": new_id}
        )
    output._selection_diagnostics = {
        **dict(output._selection_diagnostics or {}),
        "superset_pair_repair_count": len(replacements),
        "superset_pair_replacements": replacement_evidence,
    }
    return order_routine_output_by_selection_phase(output)


def apply_blueprint_execution_policy(
    output: LLMRoutineOutput,
    blueprint: RoutineBlueprint,
) -> LLMRoutineOutput:
    """Apply server-owned grouping, rest, and grouped-volume rules.

    The LLM chooses exercises only. It never decides whether a main lift is
    paired, how a circuit is grouped, or how much rest a group receives.
    """
    output = order_routine_output_by_selection_phase(output)
    policy = blueprint.execution_policy
    phases = [_execution_phase(output, exercise) for exercise in output.exercises]
    protected_main_indices = _required_main_execution_indices(output, blueprint)
    assigned: set[int] = set()
    groups: list[tuple[str, list[int]]] = []

    if policy.structure == "superset" and policy.pair_antagonist_main_lifts:
        main_pair = _find_antagonist_main_pair(output, phases)
        if main_pair and _execution_group_is_feasible(
            output,
            main_pair,
            grouped_set_cap=policy.grouped_set_cap,
        ):
            groups.append(("superset", list(main_pair)))
            assigned.update(main_pair)

    # A superset option means one deliberate pair by default, not that the
    # entire session should be converted into consecutive pairs. Preserve all
    # remaining work as ordinary sets after the main antagonist pair. If a
    # main pair is unavailable, select only the first compatible accessory pair.
    if policy.structure == "superset" and not groups:
        accessory_indices = [
            index
            for index, phase in enumerate(phases)
            if index not in assigned
            and index not in protected_main_indices
            and phase in {"secondary_compound", "isolation_accessory"}
        ]
        cursor = 0
        while cursor < len(accessory_indices) - 1:
            first = accessory_indices[cursor]
            second = accessory_indices[cursor + 1]
            if (
                _exercise_primary(output.exercises[first]) != _exercise_primary(output.exercises[second])
                and _execution_group_is_feasible(
                    output,
                    [first, second],
                    grouped_set_cap=policy.grouped_set_cap,
                )
            ):
                groups.append(("superset", [first, second]))
                assigned.update({first, second})
                break
            else:
                cursor += 1
    elif policy.structure in {"circuit", "density"}:
        for group in _contiguous_feasible_accessory_groups(
            output,
            phases,
            protected_main_indices,
            max_group_size=policy.max_group_size,
            grouped_set_cap=policy.grouped_set_cap,
        ):
            groups.append((policy.accessory_group_type, group))
            assigned.update(group)

    for index in range(len(output.exercises)):
        if index not in assigned:
            groups.append(("single", [index]))

    groups.sort(key=lambda item: min(item[1]))
    type_counters = {"single": 0, "superset": 0, "circuit": 0, "density": 0}
    prefixes = {"single": "S", "superset": "SS", "circuit": "C", "density": "D"}
    metadata: dict[str, dict[str, Any]] = {}

    for group_type, indices in groups:
        type_counters[group_type] += 1
        group_id = f"{prefixes[group_type]}{type_counters[group_type]}"
        group_size = len(indices)
        for position, index in enumerate(indices, start=1):
            exercise = output.exercises[index]
            transition_rest = None
            group_rest = None
            if group_type != "single":
                if policy.grouped_set_cap is not None:
                    exercise.sets = min(exercise.sets, policy.grouped_set_cap)
                if position < group_size:
                    transition_rest = policy.transition_rest_sec
                    exercise.rest_time_sec = int(policy.transition_rest_sec or 0)
                else:
                    group_rest = policy.group_rest_sec
                    exercise.rest_time_sec = int(policy.group_rest_sec or exercise.rest_time_sec)
            exercise_id = str(exercise.exercise_id or "").strip().lower()
            metadata[exercise_id] = {
                "sessionPhase": phases[index],
                "executionGroupId": group_id,
                "executionGroupType": group_type,
                "executionGroupPosition": position,
                "executionGroupSize": group_size,
                "transitionRestSec": transition_rest,
                "groupRestSec": group_rest,
            }

    output._execution_metadata_by_exercise_id = metadata
    output._selection_diagnostics = {
        **dict(output._selection_diagnostics or {}),
        "execution_policy_version": SESSION_EXECUTION_POLICY_VERSION,
        "session_structure": policy.structure,
        "execution_group_count": len(groups),
        "grouped_exercise_count": sum(len(indices) for group_type, indices in groups if group_type != "single"),
        "required_main_single_count": len(protected_main_indices),
        "superset_station_policy_version": EQUIPMENT_POLICY_VERSION,
    }
    if policy.structure != "standard":
        structure_labels = {
            "superset": "슈퍼세트",
            "circuit": "서킷",
            "density": "밀도형",
        }
        grouped_exercise_count = output._selection_diagnostics["grouped_exercise_count"]
        if not grouped_exercise_count:
            message = (
                f"{structure_labels[policy.structure]} 구조를 요청했지만 이번 구성에서는 "
                "정책에 맞게 묶을 수 있는 운동 조합이 없어 모두 일반 세트로 진행합니다."
            )
        elif policy.structure == "superset":
            message = (
                "슈퍼세트는 시간을 아끼기 위해 선택한 진행 방식입니다. 장비 이동·점유 "
                "조건, 그룹 적용 후 세트 수, 근육 조합이 모두 맞는 핵심 한 쌍만 묶고 나머지는 일반 세트로 "
                "유지했으며, 일반 세트보다 체감 피로가 높을 수 있습니다."
            )
        else:
            message = (
                f"{structure_labels[policy.structure]} 구조에서는 메인 운동은 각각 일반 세트로 진행하고, "
                "장비 이동 조건과 그룹 적용 후 세트 수가 맞는 보조 운동만 함께 진행합니다."
            )
        output.rationale_summary = list(dict.fromkeys([*output.rationale_summary, message]))
    return output


def apply_blueprint_execution_policy_to_draft(
    draft: RoutineDraftResponse,
    blueprint: RoutineBlueprint,
    ranked_candidates: Sequence[dict[str, Any]] | None = None,
) -> RoutineDraftResponse:
    """Apply the same execution contract to deterministic fallback responses."""
    if not draft.routine_blocks:
        return draft
    candidate_by_id = {
        str(candidate.get("id") or "").strip().lower(): candidate
        for candidate in (ranked_candidates or [])
        if str(candidate.get("id") or "").strip()
    }
    plans: list[LLMExercisePlan] = []
    phase_by_id: dict[str, str] = {}
    slot_by_id: dict[str, str] = {}
    compound_seen = 0
    draft_exercise_ids = {
        str(block.exercise_id or "").strip().lower()
        for block in draft.routine_blocks
        if str(block.exercise_id or "").strip()
    }
    selected_ranked_candidates = [
        candidate
        for candidate in (ranked_candidates or [])
        if str(candidate.get("id") or "").strip().lower() in draft_exercise_ids
    ]
    assigned_reconstructed_ids: set[str] = set()
    for bundle in build_blueprint_candidate_bundles(
        blueprint,
        selected_ranked_candidates,
        top_k_per_slot=max(1, len(selected_ranked_candidates)),
    ):
        slot_id = str(bundle.get("slotId") or "").strip()
        if (
            not slot_id
            or not (bundle.get("requirement") == "required_when_safe"
                    or (blueprint.formula_version == "gym-routine-blueprint-v10"
                        and slot_id == "upper_vertical_pull"))
            or bundle.get("phase") not in _REQUIRED_MAIN_PHASES
        ):
            continue
        exercise_id = next(
            (
                str(item.get("exerciseId") or "").strip().lower()
                for item in _bundle_candidates(bundle)
                if str(item.get("exerciseId") or "").strip().lower()
                in draft_exercise_ids - assigned_reconstructed_ids
            ),
            "",
        )
        if exercise_id:
            slot_by_id[exercise_id] = slot_id
            assigned_reconstructed_ids.add(exercise_id)
    adjunct_phase_by_option = {
        slot.adjunct_option: slot.phase
        for slot in blueprint.slots
        if slot.adjunct_option
    }
    used_adjunct_options: set[str] = set()
    adjunct_option_by_id: dict[str, str] = {}
    for block in draft.routine_blocks:
        first_set = block.prescription[0] if block.prescription else None
        movement_type = str(
            block.movement_type
            or candidate_by_id.get(str(block.exercise_id or "").strip().lower(), {}).get("movement_type")
            or ""
        ).upper() or None
        plan = LLMExercisePlan(
            exercise_id=block.exercise_id,
            exercise_name=block.exercise_name,
            movement_pattern=block.movement_pattern,
            movement_type=movement_type,
            primary_muscles=list(block.primary_muscles),
            equipment_type=block.equipment_type,
            target_weight_kg=first_set.target_weight_kg if first_set else None,
            target_reps=first_set.target_reps if first_set else 10,
            sets=max(1, len(block.prescription)),
            rest_time_sec=block.default_rest_sec,
            target_rir=first_set.target_rir if first_set else 2,
            exercise_rationale=block.exercise_rationale,
            substitution_candidates=[],
        )
        if block.training_method is not None:
            plan._training_method_instruction = block.training_method.model_dump(by_alias=True)
        if first_set is not None:
            plan._prescription_set_type = first_set.set_type
            plan._target_duration_sec = first_set.target_duration_sec
            plan._interval_work_sec = first_set.interval_work_sec
            plan._interval_recovery_sec = first_set.interval_recovery_sec
            plan._interval_rounds = first_set.interval_rounds
            plan._target_distance_m = first_set.target_distance_m
        restore_unilateral_from_block(plan, block)
        plans.append(plan)
        exercise_id = str(block.exercise_id or "").strip().lower()
        candidate = candidate_by_id.get(exercise_id, {})
        matched_adjunct = block.adjunct_option or next(
            (
                option
                for option in blueprint.planned_adjunct_options
                if option not in used_adjunct_options
                if candidate_matches_adjunct_option(candidate, option)
            ),
            None,
        )
        if matched_adjunct:
            used_adjunct_options.add(matched_adjunct)
            adjunct_option_by_id[exercise_id] = matched_adjunct
            block.adjunct_option = matched_adjunct
            phase_by_id[exercise_id] = (
                adjunct_phase_by_option.get(matched_adjunct)
                or _inferred_candidate_phase(candidate)
            )
        elif movement_type == "COMPOUND":
            phase_by_id[exercise_id] = "main_compound" if compound_seen == 0 else "secondary_compound"
            compound_seen += 1
        elif movement_type == "ISOLATION":
            phase_by_id[exercise_id] = "isolation_accessory"
        else:
            phase_by_id[exercise_id] = _inferred_candidate_phase(candidate)

    output = LLMRoutineOutput(
        total_estimated_time=draft.total_estimated_time,
        summary_title=draft.summary_title,
        rationale_summary=list(draft.rationale_summary),
        exercises=plans,
        warnings=list(draft.warnings),
    )
    output._selection_phase_by_exercise_id = phase_by_id
    output._selection_slot_by_exercise_id = slot_by_id
    output._adjunct_option_by_exercise_id = adjunct_option_by_id
    output = apply_blueprint_execution_policy(output, blueprint)
    if blueprint.execution_policy.structure == "standard":
        output = order_routine_output_by_selection_phase(output, group_primary_muscles=True)
    block_by_id = {
        str(block.exercise_id or "").strip().lower(): block
        for block in draft.routine_blocks
    }
    ordered_blocks = []
    for order, exercise in enumerate(output.exercises, start=1):
        exercise_id = str(exercise.exercise_id or "").strip().lower()
        block = block_by_id[exercise_id]
        block.order = order
        block.movement_type = exercise.movement_type
        block.default_rest_sec = exercise.rest_time_sec
        block.prescription = block.prescription[: exercise.sets]
        for set_index, prescription in enumerate(block.prescription, start=1):
            prescription.set_index = set_index
            if not (block.unilateral_prescription is not None
                    and blueprint.execution_policy.structure == 'standard'):
                prescription.target_rest_sec = exercise.rest_time_sec
        restore_unilateral_from_block(exercise, block)
        metadata = output._execution_metadata_by_exercise_id.get(exercise_id, {})
        block.session_phase = metadata.get("sessionPhase")
        block.execution_group_id = metadata.get("executionGroupId")
        block.execution_group_type = metadata.get("executionGroupType", "single")
        block.execution_group_position = int(metadata.get("executionGroupPosition") or 1)
        block.execution_group_size = int(metadata.get("executionGroupSize") or 1)
        block.transition_rest_sec = metadata.get("transitionRestSec")
        block.group_rest_sec = metadata.get("groupRestSec")
        ordered_blocks.append(block)

    draft.routine_blocks = ordered_blocks
    draft.total_estimated_time = estimate_routine_time_min(
        output.exercises,
        execution_metadata=output._execution_metadata_by_exercise_id,
    )
    draft.rationale_summary = output.rationale_summary
    draft.warnings = output.warnings
    return draft


def _bundle_candidates(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in (bundle.get("candidates") or []) if isinstance(item, dict)]


def required_main_pattern_targets(blueprint: RoutineBlueprint) -> tuple[str, ...]:
    """Return primary targets that require at least one real compound main lift."""

    targets: list[str] = []
    for slot in blueprint.slots:
        if (
            slot.requirement != "required_when_safe"
            or slot.phase not in _REQUIRED_MAIN_PHASES
            or "COMPOUND" not in slot.preferred_movement_types
        ):
            continue
        for muscle in slot.target_muscles:
            normalized = str(muscle or "").strip().lower().replace("_", "-")
            if normalized and normalized not in targets:
                targets.append(normalized)
    return tuple(targets)


def required_main_pattern_slot_specs(
    blueprint: RoutineBlueprint,
) -> tuple[dict[str, tuple[str, ...]], ...]:
    """Expose the required main-slot facts used by runtime and evaluators."""

    specs: list[dict[str, tuple[str, ...]]] = []
    for slot in blueprint.slots:
        if (
            slot.requirement != "required_when_safe"
            or slot.phase not in _REQUIRED_MAIN_PHASES
        ):
            continue
        specs.append({
            "target_muscles": tuple(
                str(muscle or "").strip().lower().replace("_", "-")
                for muscle in slot.target_muscles
                if str(muscle or "").strip()
            ),
            "preferred_movement_types": tuple(
                str(movement or "").strip().upper()
                for movement in slot.preferred_movement_types
                if str(movement or "").strip()
            ),
        })
    return tuple(specs)


def routine_items_have_required_main_pattern(
    items: Sequence[Any],
    required_targets: Sequence[str],
    *,
    candidates: Sequence[dict[str, Any]] | None = None,
    phase_by_exercise_id: Mapping[str, str] | None = None,
    required_slot_specs: Sequence[Mapping[str, Sequence[str]]] | None = None,
) -> bool:
    """Check a response against candidate-owned movement and primary-muscle facts."""

    normalized_targets = {
        str(item or "").strip().lower().replace("_", "-")
        for item in required_targets
        if str(item or "").strip()
    }
    if not normalized_targets:
        return True
    candidate_by_id = {
        str(candidate.get("id") or "").strip().lower(): candidate
        for candidate in (candidates or [])
        if str(candidate.get("id") or "").strip()
    }
    phases = {
        str(exercise_id or "").strip().lower(): str(phase or "").strip()
        for exercise_id, phase in (phase_by_exercise_id or {}).items()
    }

    def value(item: Any, key: str, default: Any = None) -> Any:
        if isinstance(item, Mapping):
            return item.get(key, default)
        return getattr(item, key, default)

    normalized_slot_specs = [
        (
            {
                str(target or "").strip().lower().replace("_", "-")
                for target in (spec.get("target_muscles") or [])
                if str(target or "").strip()
            },
            {
                str(movement or "").strip().upper()
                for movement in (spec.get("preferred_movement_types") or [])
                if str(movement or "").strip()
            },
        )
        for spec in (required_slot_specs or [])
    ]

    for item in items:
        exercise_id = str(
            value(item, "exercise_id", "")
            or value(item, "id", "")
            or ""
        ).strip().lower()
        candidate = candidate_by_id.get(exercise_id, {})
        movement = str(
            candidate.get("movement_type")
            or value(item, "movement_type", "")
            or ""
        ).strip().upper()
        candidate_primary = str(candidate.get("primary_muscle") or "").strip()
        raw_primaries = (
            [candidate_primary]
            if candidate_primary
            else value(item, "primary_muscles", [])
            or value(item, "primary_muscle", [])
            or []
        )
        if isinstance(raw_primaries, str):
            raw_primaries = [raw_primaries]
        primaries = {
            str(primary or "").strip().lower().replace("_", "-")
            for primary in raw_primaries
            if str(primary or "").strip()
        }
        if normalized_slot_specs:
            if any(
                movement in accepted_movements and bool(primaries & slot_targets)
                for slot_targets, accepted_movements in normalized_slot_specs
            ):
                return True
            continue
        if movement != "COMPOUND":
            continue
        if not (primaries & normalized_targets):
            continue
        phase = str(
            value(item, "session_phase", "")
            or phases.get(exercise_id)
            or _inferred_candidate_phase(candidate or {"movement_type": movement})
        ).strip()
        if phase in _REQUIRED_MAIN_PHASES:
            return True
    return False


def candidates_have_required_main_pattern(
    blueprint: RoutineBlueprint,
    candidates: Sequence[dict[str, Any]],
) -> bool:
    """Only stop early when even the full allowed pool fails the final gate."""
    return routine_items_have_required_main_pattern(
        candidates,
        required_main_pattern_targets(blueprint),
        candidates=candidates,
        required_slot_specs=required_main_pattern_slot_specs(blueprint),
    )


def routine_output_has_required_main_pattern(
    output: LLMRoutineOutput,
    blueprint: RoutineBlueprint,
    candidates: Sequence[dict[str, Any]] | None = None,
) -> bool:
    return routine_items_have_required_main_pattern(
        output.exercises,
        required_main_pattern_targets(blueprint),
        candidates=candidates,
        phase_by_exercise_id=dict(output._selection_phase_by_exercise_id or {}),
        required_slot_specs=required_main_pattern_slot_specs(blueprint),
    )


def routine_draft_has_required_main_pattern(
    draft: RoutineDraftResponse,
    blueprint: RoutineBlueprint,
    candidates: Sequence[dict[str, Any]] | None = None,
) -> bool:
    return routine_items_have_required_main_pattern(
        draft.routine_blocks,
        required_main_pattern_targets(blueprint),
        candidates=candidates,
        required_slot_specs=required_main_pattern_slot_specs(blueprint),
    )


def adapt_slot_selection_output(
    selection_output: LLMRoutineSelectionOutput,
    *,
    blueprint: RoutineBlueprint,
    slot_candidate_bundles: Sequence[dict[str, Any]],
    ranked_candidates: Sequence[dict[str, Any]],
) -> LLMRoutineOutput:
    """Repair selections only within deterministic bundles and emit stable v1 schema."""
    ranked_candidates = tuple(
        candidate for candidate in ranked_candidates
        if automatic_recommendation_eligibility(candidate, blueprint.experience_level,
            adjunct_options=blueprint.planned_adjunct_options).eligible
    )
    candidate_by_id = {
        str(candidate.get("id") or "").strip().lower(): candidate
        for candidate in ranked_candidates
        if str(candidate.get("id") or "").strip()
    }
    bundles_by_slot = {
        str(bundle.get("slotId") or "").strip(): bundle
        for bundle in slot_candidate_bundles
        if str(bundle.get("slotId") or "").strip()
    }
    allowed_ids_by_slot = {
        slot_id: {
            str(item.get("exerciseId") or "").strip().lower()
            for item in _bundle_candidates(bundle)
            if str(item.get("exerciseId") or "").strip()
        }
        for slot_id, bundle in bundles_by_slot.items()
    }
    selections_by_slot: dict[str, list] = defaultdict(list)
    for selection in selection_output.selections:
        selections_by_slot[str(selection.slot_id or "").strip()].append(selection)

    used_ids: set[str] = set()
    used_families: set[str] = set()
    plans: list[LLMExercisePlan] = []
    # The small model is only authoritative for candidate IDs. User-facing
    # warnings are generated later from deterministic request and safety rules.
    warnings: list[str] = []
    required_repaired_slots: list[str] = []
    requested_repaired_slots: list[str] = []
    minimum_fill_count = 0
    model_accepted_count = 0
    rejected_selection_count = 0
    duplicate_family_skip_count = sum(
        max(0, int(bundle.get("duplicateFamilySuppressedCount") or 0))
        for bundle in slot_candidate_bundles
    )
    duplicate_family_skip_ids: set[str] = set()
    coarse_pattern_skip_ids: set[str] = set()
    distribution_cap_relax_count = 0
    phase_by_exercise_id: dict[str, str] = {}
    slot_by_exercise_id: dict[str, str] = {}
    adjunct_option_by_exercise_id: dict[str, str] = {}
    adjunct_option_by_slot_id = {
        slot.slot_id: slot.adjunct_option
        for slot in blueprint.slots
        if slot.adjunct_option
    }
    primary_counts: dict[str, int] = defaultdict(int)
    pattern_counts: dict[str, int] = defaultdict(int)
    target_order = list(blueprint.target_muscles)
    target_index = {muscle: index for index, muscle in enumerate(target_order)}
    accessory_primary_split = blueprint.target_split_label in {"arm", "core", "shoulder", "neck"}
    primary_weights = {
        muscle: 1 if accessory_primary_split or muscle not in LARGE_MUSCLE_SLUGS else 2
        for muscle in target_order
    }
    focus_targets = SPECIALIZATION_FOCUS_TARGETS.get(str(blueprint.specialization_focus or ""), frozenset())
    # Low readiness limits dose and methods, not the requested muscle priority.
    for muscle in focus_targets & set(primary_weights):
        primary_weights[muscle] = max(3, primary_weights[muscle])
    main_capacity = max(1, blueprint.minimum_main_exercise_count)
    weight_total = max(1, sum(primary_weights.values()))
    primary_caps = {
        muscle: max(1, math.ceil(main_capacity * weight / weight_total))
        for muscle, weight in primary_weights.items()
    }

    def append_candidate(
        exercise_id: str,
        rationale: str,
        phase: str | None = None,
        slot_id: str | None = None,
    ) -> bool:
        nonlocal duplicate_family_skip_count
        normalized_id = exercise_id.strip().lower()
        candidate = candidate_by_id.get(normalized_id)
        if not candidate or normalized_id in used_ids:
            return False
        option = adjunct_option_by_slot_id.get(slot_id)
        if not automatic_recommendation_eligibility(candidate, blueprint.experience_level,
            adjunct_options=(option,) if option else ()).eligible:
            return False
        family = _candidate_family(candidate, blueprint)
        if family and family in used_families:
            if normalized_id not in duplicate_family_skip_ids:
                duplicate_family_skip_ids.add(normalized_id)
                duplicate_family_skip_count += 1
            return False
        pattern = _candidate_pattern(candidate)
        if pattern == "horizontal_row" and blueprint.formula_version == "gym-routine-blueprint-v11":
            from .pull_composition import complementary_pair

            existing_rows = [candidate_by_id[eid] for eid in used_ids
                             if _candidate_pattern(candidate_by_id[eid]) == "horizontal_row"]
            if existing_rows and not complementary_pair([*existing_rows, candidate]):
                coarse_pattern_skip_ids.add(normalized_id)
                return False
        if is_redundant_push_front_raise(
            blueprint.target_split_label, blueprint.specialization_focus, pattern,
            {key for key, count in pattern_counts.items() if count > 0},
            ranked_candidates,
        ):
            coarse_pattern_skip_ids.add(normalized_id)
            return False
        pattern_limit = selection_pattern_limit(pattern, blueprint)
        if pattern_limit is not None and pattern_counts.get(pattern, 0) >= pattern_limit:
            coarse_pattern_skip_ids.add(normalized_id)
            return False
        used_ids.add(normalized_id)
        if family:
            used_families.add(family)
        if pattern:
            pattern_counts[pattern] += 1
        plans.append(_candidate_plan(candidate, rationale, goal=blueprint.goal))
        resolved_phase = _resolved_slot_phase(candidate, phase)
        phase_by_exercise_id[normalized_id] = resolved_phase
        if slot_id:
            slot_by_exercise_id[normalized_id] = slot_id
            adjunct_option = adjunct_option_by_slot_id.get(slot_id)
            if adjunct_option:
                adjunct_option_by_exercise_id[normalized_id] = adjunct_option
        primary = str(candidate.get("primary_muscle") or "").strip().lower().replace("_", "-")
        if primary in primary_weights and resolved_phase not in _NON_MAIN_PHASES:
            primary_counts[primary] += 1
        return True

    def flex_candidate_priority(candidate: dict[str, Any]) -> tuple[int, float, int, int, int, str]:
        primary = str(candidate.get("primary_muscle") or "").strip().lower().replace("_", "-")
        weight = max(1, primary_weights.get(primary, 1))
        ratio = primary_counts.get(primary, 0) / weight
        movement_priority = 0 if str(candidate.get("movement_type") or "").strip().upper() == "COMPOUND" else 1
        target_priority = target_index.get(primary, len(target_index))
        redundant_front_raise = int(
            blueprint.target_split_label == "push"
            and blueprint.specialization_focus != "frontDeltoidEmphasis"
            and _candidate_pattern(candidate) == "front_raise"
            and pattern_counts.get("vertical_press", 0) > 0
        )
        if str(blueprint.goal or "").strip().lower().replace("_", "") == "strength":
            return (
                redundant_front_raise,
                ratio,
                movement_priority,
                target_priority,
                -int(candidate.get("score") or 0),
                str(candidate.get("id") or ""),
            )
        return (
            redundant_front_raise,
            ratio,
            target_priority,
            movement_priority,
            -int(candidate.get("score") or 0),
            str(candidate.get("id") or ""),
        )

    def flex_within_cap(candidate: dict[str, Any]) -> bool:
        primary = str(candidate.get("primary_muscle") or "").strip().lower().replace("_", "-")
        cap = primary_caps.get(primary)
        return cap is None or primary_counts.get(primary, 0) < cap

    def first_available_from_bundle(slot_id: str) -> str | None:
        bundle = bundles_by_slot.get(slot_id) or {}
        for item in _bundle_candidates(bundle):
            exercise_id = str(item.get("exerciseId") or "").strip()
            normalized_id = exercise_id.lower()
            candidate = candidate_by_id.get(normalized_id)
            if not exercise_id or normalized_id in used_ids or not candidate:
                continue
            if _candidate_family(candidate, blueprint) in used_families:
                continue
            pattern = _candidate_pattern(candidate)
            pattern_limit = selection_pattern_limit(pattern, blueprint)
            if pattern_limit is not None and pattern_counts.get(pattern, 0) >= pattern_limit:
                coarse_pattern_skip_ids.add(normalized_id)
                continue
            return exercise_id
        return None

    ordered_slot_ids = [slot.slot_id for slot in blueprint.slots]
    for slot in blueprint.slots:
        allowed_ids = allowed_ids_by_slot.get(slot.slot_id, set())
        selected_for_slot = selections_by_slot.get(slot.slot_id, [])
        accepted = False
        for selection in selected_for_slot:
            exercise_id = str(selection.exercise_id or "").strip()
            if exercise_id.lower() not in allowed_ids:
                rejected_selection_count += 1
                continue
            candidate = candidate_by_id.get(exercise_id.lower())
            rationale = _candidate_public_rationale(candidate or {}, purpose=slot.purpose)
            if append_candidate(exercise_id, rationale, slot.phase, slot.slot_id):
                accepted = True
                model_accepted_count += 1
                break
            rejected_selection_count += 1

        if accepted:
            continue
        should_repair = slot.requirement == "required_when_safe"
        if slot.requirement == "preferred_when_requested_and_safe":
            should_repair = True
        if should_repair:
            replacement_id = first_available_from_bundle(slot.slot_id)
            replacement_candidate = candidate_by_id.get(str(replacement_id or "").lower())
            if replacement_id and append_candidate(
                replacement_id,
                _candidate_public_rationale(
                    replacement_candidate or {},
                    purpose=slot.purpose,
                ),
                slot.phase,
                slot.slot_id,
            ):
                if slot.requirement == "required_when_safe":
                    required_repaired_slots.append(slot.slot_id)
                else:
                    requested_repaired_slots.append(slot.slot_id)

    flex_bundle = bundles_by_slot.get("flex") or {}
    flex_allowed_ids = allowed_ids_by_slot.get("flex", set())
    for selection in selections_by_slot.get("flex", []):
        if len(plans) >= blueprint.target_exercise_count:
            break
        exercise_id = str(selection.exercise_id or "").strip()
        if exercise_id.lower() not in flex_allowed_ids:
            rejected_selection_count += 1
            continue
        candidate = candidate_by_id.get(exercise_id.lower())
        if candidate and not flex_within_cap(candidate):
            rejected_selection_count += 1
            continue
        rationale = _candidate_public_rationale(candidate or {})
        if append_candidate(exercise_id, rationale):
            model_accepted_count += 1
        else:
            rejected_selection_count += 1

    def selected_main_count() -> int:
        return sum(
            str(plan.exercise_id or "").strip().lower()
            not in adjunct_option_by_exercise_id
            for plan in plans
        )

    if selected_main_count() < blueprint.minimum_main_exercise_count:
        flex_candidates = [
            candidate_by_id[str(item.get("exerciseId") or "").strip().lower()]
            for item in _bundle_candidates(flex_bundle)
            if str(item.get("exerciseId") or "").strip().lower() in candidate_by_id
        ]
        while (
            selected_main_count() < blueprint.minimum_main_exercise_count
            and len(plans) < blueprint.target_exercise_count
        ):
            for candidate in flex_candidates:
                normalized_id = str(candidate.get("id") or "").strip().lower()
                if (
                    normalized_id
                    and normalized_id not in used_ids
                    and _candidate_family(candidate, blueprint) in used_families
                    and normalized_id not in duplicate_family_skip_ids
                ):
                    duplicate_family_skip_ids.add(normalized_id)
                    duplicate_family_skip_count += 1
            available = [
                candidate
                for candidate in flex_candidates
                if str(candidate.get("id") or "").strip().lower() not in used_ids
                and _candidate_family(candidate, blueprint) not in used_families
                and (
                    selection_pattern_limit(_candidate_pattern(candidate), blueprint) is None
                    or pattern_counts.get(_candidate_pattern(candidate), 0)
                    < selection_pattern_limit(_candidate_pattern(candidate), blueprint)
                )
            ]
            if not available:
                break
            within_cap = [candidate for candidate in available if flex_within_cap(candidate)]
            pool = within_cap or available
            if not within_cap:
                distribution_cap_relax_count += 1
            chosen = min(pool, key=flex_candidate_priority)
            if append_candidate(
                str(chosen.get("id") or ""),
                _candidate_public_rationale(chosen),
            ):
                minimum_fill_count += 1
            else:
                break

    # Unknown slot IDs are never accepted, but keep their count visible without raw payloads.
    known_slot_ids = set(ordered_slot_ids) | {"flex"}
    rejected_selection_count += sum(
        len(items)
        for slot_id, items in selections_by_slot.items()
        if slot_id not in known_slot_ids
    )
    if required_repaired_slots:
        warnings.append(
            "운동 부위와 세션 구성을 맞추기 위해 누락된 운동을 안전 조건 안에서 보완했습니다."
        )
    if rejected_selection_count:
        warnings.append(
            f"현재 조건 또는 루틴 구성과 맞지 않는 운동 {rejected_selection_count}개를 제외했습니다."
        )

    unique_warnings = list(dict.fromkeys(str(item).strip() for item in warnings if str(item).strip()))
    output = LLMRoutineOutput(
        total_estimated_time=0,
        summary_title="맞춤 운동 루틴",
        rationale_summary=_deterministic_rationale_summary(blueprint),
        exercises=plans[: blueprint.target_exercise_count],
        warnings=unique_warnings,
    )
    output._selection_diagnostics = {
        "contract_version": SLOT_SELECTION_CONTRACT_VERSION,
        "specialization_focus_unavailable": any(
            bundle.get("specializationFocusUnavailable", False)
            for bundle in slot_candidate_bundles
        ),
        "model_selection_count": len(selection_output.selections),
        "model_accepted_count": model_accepted_count,
        "rejected_selection_count": rejected_selection_count,
        "required_slot_repair_count": len(required_repaired_slots),
        "requested_slot_repair_count": len(requested_repaired_slots),
        "minimum_fill_count": minimum_fill_count,
        "duplicate_key_repair_count": 0,
        "duplicate_family_skip_count": duplicate_family_skip_count,
        "coarse_pattern_skip_count": len(coarse_pattern_skip_ids),
        "distribution_cap_relax_count": distribution_cap_relax_count,
    }
    output._selection_phase_by_exercise_id = phase_by_exercise_id
    output._selection_slot_by_exercise_id = slot_by_exercise_id
    output._adjunct_option_by_exercise_id = adjunct_option_by_exercise_id
    return complete_accessory_group_selection(
        output, blueprint, slot_candidate_bundles, ranked_candidates,
    )


def adapt_structured_routine_response(
    response: Any,
    *,
    blueprint: RoutineBlueprint,
    slot_candidate_bundles: Sequence[dict[str, Any]],
    ranked_candidates: Sequence[dict[str, Any]],
) -> LLMRoutineOutput:
    """Accept the internal v2 response while preserving legacy v1 providers/mocks."""
    if isinstance(response, LLMRoutineOutput):
        return response
    if isinstance(response, dict) and "exercises" in response and "selections" not in response:
        return LLMRoutineOutput.model_validate(response)
    selection_output = (
        response
        if isinstance(response, LLMRoutineSelectionOutput)
        else LLMRoutineSelectionOutput.model_validate(response)
    )
    return adapt_slot_selection_output(
        selection_output,
        blueprint=blueprint,
        slot_candidate_bundles=slot_candidate_bundles,
        ranked_candidates=ranked_candidates,
    )


def normalize_routine_selection_response(
    raw_text: str,
    *,
    blueprint: RoutineBlueprint,
    slot_candidate_bundles: Sequence[dict[str, Any]],
    ranked_candidates: Sequence[dict[str, Any]],
    llm=None,
) -> LLMRoutineOutput:
    """Parse v2 slot selection, with v1 exercises as a compatibility fallback."""
    duplicate_repaired_data, duplicate_key_repair_count = _recover_duplicate_selection_objects(raw_text)
    parse_result = parse_json_candidate(raw_text)
    if not parse_result.success:
        if llm is not None:
            try:
                from langchain.output_parsers import OutputFixingParser
                from langchain_core.output_parsers import PydanticOutputParser

                base_parser = PydanticOutputParser(pydantic_object=LLMRoutineSelectionOutput)
                fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)
                repaired = fixing_parser.parse(str(raw_text or ""))
                return adapt_slot_selection_output(
                    repaired,
                    blueprint=blueprint,
                    slot_candidate_bundles=slot_candidate_bundles,
                    ranked_candidates=ranked_candidates,
                )
            except Exception as exc:
                raise SchemaRepairError(
                    "slot selection JSON repair failed",
                    parse_failure_subtype=parse_result.subtype,
                    repair_failure_reason=parse_result.subtype,
                    json_decode_error_category=parse_result.json_decode_error_category,
                    json_decode_recovery_attempted=parse_result.json_decode_recovery_attempted,
                    json_decode_recovery_succeeded=parse_result.json_decode_recovery_succeeded,
                    json_decode_recovery_strategy=parse_result.json_decode_recovery_strategy,
                ) from exc
        raise SchemaRepairError(
            "slot selection JSON parse failed",
            parse_failure_subtype=parse_result.subtype,
            repair_failure_reason=parse_result.subtype,
            json_decode_error_category=parse_result.json_decode_error_category,
            json_decode_recovery_attempted=parse_result.json_decode_recovery_attempted,
            json_decode_recovery_succeeded=parse_result.json_decode_recovery_succeeded,
            json_decode_recovery_strategy=parse_result.json_decode_recovery_strategy,
        )

    data = duplicate_repaired_data or parse_result.data or {}
    if isinstance(data, dict) and "exercises" in data and "selections" not in data:
        return normalize_llm_response(raw_text, llm=llm)
    if isinstance(data, dict) and "selections" not in data:
        for alias in ("slotSelections", "selectedExercises", "selected_exercises"):
            if alias in data:
                data["selections"] = data[alias]
                break
    selection_contract_type_repair_count = 0
    if isinstance(data, dict):
        data, selection_contract_type_repair_count = _normalize_selection_contract_data(data)
    try:
        parsed = LLMRoutineSelectionOutput.model_validate(data)
    except ValidationError as validation_error:
        category, fields = classify_validation_error(validation_error)
        loc_summaries, error_types = summarize_validation_error_shapes(validation_error)
        raise SchemaRepairError(
            "slot selection schema validation failed",
            parse_failure_subtype="pydantic_validation_error",
            schema_validation_error_category=category,
            schema_validation_field_names=fields,
            schema_validation_loc_summaries=loc_summaries,
            schema_validation_error_types=error_types,
            repair_failure_reason=category,
        ) from validation_error
    output = adapt_slot_selection_output(
        parsed,
        blueprint=blueprint,
        slot_candidate_bundles=slot_candidate_bundles,
        ranked_candidates=ranked_candidates,
    )
    output._selection_diagnostics["duplicate_key_repair_count"] = duplicate_key_repair_count
    output._selection_diagnostics[
        "selection_contract_type_repair_count"
    ] = selection_contract_type_repair_count
    return output

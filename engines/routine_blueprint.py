"""Deterministic gym-routine structure passed to the constrained LLM layer."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

from .option_resolver import (
    ADJUNCT_OPTION_PHASE,
    HIGH_DENSITY_ADJUNCTS,
    LEGACY_STYLE_TO_ADJUNCT_OPTION,
    MAX_DISTINCT_ADJUNCT_OPTIONS,
    MUTUALLY_EXCLUSIVE_ADJUNCT_GROUPS,
    canonical_adjunct_options,
    candidate_adjunct_option_codes,
    candidate_matches_adjunct_option,
)
from .prescription.adjustments import (
    _calculate_max_total_sets,
    target_exercise_count_for_session,
)
from .registry import LARGE_MUSCLE_SLUGS, SPLIT_LABEL_TO_MUSCLES
from .exercise_display import candidate_display_name
from .candidate_ranker import (
    candidate_selection_family_key,
    candidate_selection_pattern_key,
)
from .experience_policy import main_role_eligibility, automatic_recommendation_eligibility
from .specialization_policy import (
    SPECIALIZATION_FOCUS_TARGETS,
    candidate_matches_specialization_focus,
    candidate_matches_training_method,
    focus_matches_targets,
    focus_variant_family_key,
    legacy_specialization_values,
    resolve_training_method,
    training_method_supports_goal,
)


class RoutineBlueprintValidationError(ValueError):
    """Structured invariant codes; retains the existing ValueError contract."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(errors)
        super().__init__(f"invalid routine blueprint: {', '.join(self.errors)}")


PHASE_ORDER = (
    "warmup_or_preparation",
    "power_if_explicit",
    "main_compound",
    "secondary_compound",
    "isolation_accessory",
    "prehab_accessory",
    "conditioning_finisher",
    "endurance_support",
    "cooldown_recovery",
)

# A focus slot uses an observable movement pattern only when the catalog has a
# stable pattern key for it. Other focus areas keep the catalog's safety and
# equipment filters instead of inferring a clinical or equipment-specific rule.
_FOCUS_SLOT_SELECTION_PATTERNS: dict[str, tuple[str, ...]] = {
    "midBackEmphasis": ("horizontal_row",),
    "sideDeltoidEmphasis": ("lateral_raise",),
}

EXPERIENCE_PREFERENCES = {
    "beginner": (
        "mainstream_gym",
        "stable_setup",
        "low_technical_difficulty",
        "low_failure_penalty",
    ),
    "intermediate": (
        "mainstream_gym",
        "progressive_overload_friendly",
        "moderate_technical_difficulty",
    ),
    "advanced": (
        "effect_to_risk_tradeoff",
        "higher_technical_difficulty_when_unconstrained",
        "progressive_overload_friendly",
    ),
}

ADJUNCT_STYLES = {"crossfit", "athletic", "mobility", "yoga_pilates", "rehab_prehab"}


@dataclass(frozen=True)
class RoutineBlueprintSlot:
    slot_id: str
    phase: str
    requirement: str
    target_muscles: tuple[str, ...]
    preferred_movement_types: tuple[str, ...]
    preferred_selection_patterns: tuple[str, ...]
    purpose: str
    max_exercises: int = 1
    adjunct_option: str | None = None


@dataclass(frozen=True)
class RoutineExecutionPolicy:
    structure: str
    main_lift_group_type: str
    accessory_group_type: str
    max_group_size: int
    transition_rest_sec: int | None
    group_rest_sec: int | None
    grouped_set_cap: int | None
    pair_antagonist_main_lifts: bool


@dataclass(frozen=True)
class RoutineBlueprint:
    formula_version: str
    target_split_label: str
    target_muscles: tuple[str, ...]
    goal: str
    experience_level: str
    advanced_chest_focus: str
    specialization_focus: str | None
    training_method_mode: str
    training_method: str | None
    training_method_source: str | None
    training_method_reason_code: str | None
    specialization_slot_id: str | None
    training_method_slot_id: str | None
    readiness_level: str
    session_structure: str
    base_target_exercise_count: int
    target_main_exercise_count: int
    target_exercise_count: int
    minimum_main_exercise_count: int
    minimum_exercise_count: int
    safety_limited_target_muscles: tuple[str, ...]
    max_total_sets: int
    phase_order: tuple[str, ...]
    candidate_preferences: tuple[str, ...]
    requested_adjunct_options: tuple[str, ...]
    planned_adjunct_options: tuple[str, ...]
    suppressed_adjunct_options: tuple[str, ...]
    adjunct_suppression_reasons: tuple[tuple[str, str], ...]
    execution_policy: RoutineExecutionPolicy
    slots: tuple[RoutineBlueprintSlot, ...]
    flex_slot_count: int
    llm_role: str

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def applied_adjunct_options(self) -> tuple[str, ...]:
        """Compatibility alias; blueprint values are plans, not final evidence."""
        return self.planned_adjunct_options


def _norm(value: str | None, default: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return normalized or default


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        token = str(value or "").strip().lower().replace("_", "-")
        if token and token not in seen:
            seen.add(token)
            result.append(token)
    return tuple(result)


def _unique_codes(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        token = _norm(value, "")
        if token and token not in seen:
            seen.add(token)
            result.append(token)
    return tuple(result)


def _body_part_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _expand_safety_body_parts(values: Iterable[Any]) -> set[str]:
    expanded = {_body_part_token(value) for value in values if _body_part_token(value)}
    if expanded & {"left-soleus", "right-soleus", "soleus"}:
        expanded.update({"left-soleus", "right-soleus", "soleus", "calves"})
    if "shoulder" in expanded:
        expanded.update({"front-deltoids", "back-deltoids", "side-deltoids"})
    if expanded & {"front-deltoids", "back-deltoids", "side-deltoids"}:
        expanded.add("shoulder")
    return expanded


def _pain_body_parts(pain_areas: Sequence[Any] | None) -> set[str]:
    values: list[Any] = []
    for item in pain_areas or []:
        if isinstance(item, Mapping):
            values.append(item.get("bodyPart", item.get("body_part", item.get("area"))))
        else:
            values.append(getattr(item, "body_part", None) or getattr(item, "area", None))
    return _expand_safety_body_parts(values)


def _severe_doms_body_parts(doms_data: Mapping[str, Any] | None) -> set[str]:
    return _expand_safety_body_parts(
        body_part
        for body_part, level in (doms_data or {}).items()
        if int(level or 0) >= 3
    )


def _safety_limited_targets(
    target_muscles: Sequence[str],
    *,
    pain_areas: Sequence[Any] | None,
    doms_data: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    blocked = _pain_body_parts(pain_areas) | _severe_doms_body_parts(doms_data)
    return tuple(muscle for muscle in target_muscles if muscle in blocked)


def _slot(
    slot_id: str,
    phase: str,
    requirement: str,
    targets: Sequence[str],
    movements: Sequence[str],
    purpose: str,
    *,
    selection_patterns: Sequence[str] = (),
    adjunct_option: str | None = None,
) -> RoutineBlueprintSlot:
    return RoutineBlueprintSlot(
        slot_id=slot_id,
        phase=phase,
        requirement=requirement,
        target_muscles=_unique(targets),
        preferred_movement_types=tuple(movements),
        preferred_selection_patterns=tuple(selection_patterns),
        purpose=purpose,
        adjunct_option=adjunct_option,
    )


def _execution_policy(
    structure: str,
    *,
    split: str,
    goal: str,
    readiness: str,
) -> RoutineExecutionPolicy:
    if structure == "superset":
        return RoutineExecutionPolicy(
            structure=structure,
            main_lift_group_type="superset" if split in {"upper", "chest_back"} and goal != "strength" and readiness != "low" else "single",
            accessory_group_type="superset",
            max_group_size=2,
            transition_rest_sec=15,
            group_rest_sec=90,
            grouped_set_cap=3,
            pair_antagonist_main_lifts=split in {"upper", "chest_back"} and goal != "strength" and readiness != "low",
        )
    if structure == "circuit":
        return RoutineExecutionPolicy(
            structure=structure,
            main_lift_group_type="single",
            accessory_group_type="circuit",
            max_group_size=4,
            transition_rest_sec=20,
            group_rest_sec=90,
            grouped_set_cap=2,
            pair_antagonist_main_lifts=False,
        )
    if structure == "density":
        return RoutineExecutionPolicy(
            structure=structure,
            main_lift_group_type="single",
            accessory_group_type="density",
            max_group_size=3,
            transition_rest_sec=30,
            group_rest_sec=60,
            grouped_set_cap=2,
            pair_antagonist_main_lifts=False,
        )
    return RoutineExecutionPolicy(
        structure="standard",
        main_lift_group_type="single",
        accessory_group_type="single",
        max_group_size=1,
        transition_rest_sec=None,
        group_rest_sec=None,
        grouped_set_cap=None,
        pair_antagonist_main_lifts=False,
    )


def _direct_accessory_requirement(goal: str, training_style: str) -> str:
    """Keep direct arm work mandatory only when it is central to the request.

    Direct arm isolation can be useful, but current evidence does not support
    treating it as a universal requirement on top of compound push/pull work.
    """
    if goal == "hypertrophy" or training_style == "bodybuilding":
        return "required_when_safe"
    return "preferred_when_safe"


def _split_slots(
    split: str,
    target_count: int,
    target_muscles: tuple[str, ...],
    *,
    goal: str,
    training_style: str,
) -> list[RoutineBlueprintSlot]:
    direct_accessory_requirement = _direct_accessory_requirement(goal, training_style)
    if split == "chest":
        slots = [
            _slot(
                "chest_main",
                "main_compound",
                "required_when_safe",
                ["chest"],
                ["COMPOUND"],
                "가슴 목표의 우선순위가 가장 높은 주력 프레스",
            )
        ]
        if target_count >= 2:
            slots.append(
                _slot(
                    "chest_secondary",
                    "secondary_compound",
                    "required_when_safe",
                    ["chest"],
                    ["COMPOUND"],
                    "주력 프레스와 각도·자극을 보완하는 두 번째 프레스",
                )
            )
        if target_count >= 3:
            slots.append(
                _slot(
                    "chest_isolation",
                    "isolation_accessory",
                    "required_when_safe",
                    ["chest"],
                    ["ISOLATION"],
                    "복합 운동 이후 가슴을 직접 보완하는 고립 운동",
                )
            )
        return slots
    if split == "push":
        slots = [
            _slot("push_main", "main_compound", "required_when_safe", ["chest"], ["COMPOUND"], "가슴 중심 메인 운동"),
            _slot("triceps_direct", "isolation_accessory", direct_accessory_requirement, ["triceps"], ["ISOLATION"], "직접 삼두 보조 볼륨"),
        ]
        if target_count >= 4:
            slots.insert(1, _slot(
                "push_secondary",
                "secondary_compound",
                "preferred_when_safe",
                ["chest", "front-deltoids", "side-deltoids"],
                ["COMPOUND", "ISOLATION"],
                "가슴/전면·측면 어깨 보완",
            ))
        return slots
    if split == "pull":
        slots = [
            _slot(
                "pull_main",
                "main_compound",
                "required_when_safe",
                ["upper-back", "trapezius"],
                ["COMPOUND", "ISOLATION"],
                "등 중심 당기기 메인 운동",
            ),
            _slot("biceps_direct", "isolation_accessory", direct_accessory_requirement, ["biceps"], ["ISOLATION"], "직접 이두 보조 볼륨"),
        ]
        if target_count >= 4:
            slots.insert(1, _slot(
                "pull_vertical_main",
                "secondary_compound",
                "preferred_when_safe",
                ["upper-back", "trapezius"],
                ["COMPOUND", "ISOLATION"],
                "광배 중심 수직 당기기 보완",
                selection_patterns=["vertical_pull"],
            ))
        return slots
    if split in {"legs", "lower"}:
        return [
            _slot(
                "lower_main",
                "main_compound",
                "required_when_safe",
                ["quadriceps", "gluteal"],
                ["COMPOUND"],
                "큰 하체 근육 중심 메인 운동",
            ),
            _slot(
                "posterior_chain",
                "secondary_compound",
                "required_when_safe",
                ["hamstring", "gluteal"],
                ["COMPOUND", "ISOLATION"],
                "햄스트링/둔근 보완",
            ),
            *(
                [
                    _slot(
                        "calves_accessory",
                        "isolation_accessory",
                        "optional",
                        ["calves"],
                        ["ISOLATION"],
                        "메인 하체 커버 이후 종아리 보조",
                    )
                ]
                if target_count >= 5
                else []
            ),
        ]
    if split in {"upper", "chest_back"}:
        return [
            _slot("upper_push_main", "main_compound", "required_when_safe", ["chest"], ["COMPOUND"], "상체 밀기 메인"),
            _slot("upper_pull_main", "secondary_compound", "required_when_safe", ["upper-back", "trapezius"], ["COMPOUND"], "상체 당기기 메인"),
        ]
    if split == "full_body":
        slots = [
            _slot(
                "full_lower_main",
                "main_compound",
                "required_when_safe",
                ["quadriceps", "gluteal"],
                ["COMPOUND"],
                "전신 루틴 하체 메인",
            ),
            _slot(
                "full_push_main",
                "main_compound",
                "required_when_safe",
                ["chest"],
                ["COMPOUND"],
                "전신 루틴 수평 밀기 메인",
                selection_patterns=["horizontal_press"],
            ),
            _slot("full_pull_main", "secondary_compound", "required_when_safe", ["upper-back", "trapezius"], ["COMPOUND"], "전신 루틴 당기기 메인"),
        ]
        if target_count >= 4:
            slots.append(
                _slot(
                    "full_posterior_chain",
                    "secondary_compound",
                    "preferred_when_safe",
                    ["hamstring", "gluteal"],
                    ["COMPOUND", "ISOLATION"],
                    "전신 루틴의 후면 사슬 보완",
                )
            )
        if target_count >= 5 and "front-deltoids" in target_muscles:
            slots.append(
                _slot(
                    "full_vertical_push",
                    "secondary_compound",
                    "preferred_when_safe",
                    ["front-deltoids", "side-deltoids"],
                    ["COMPOUND"],
                    "전신 루틴의 수직 밀기 보완",
                    selection_patterns=["vertical_press"],
                )
            )
        if target_count >= 6 and "abs" in target_muscles:
            slots.append(
                _slot(
                    "full_core_accessory",
                    "isolation_accessory",
                    "preferred_when_safe",
                    ["abs", "obliques"],
                    ["STATIC", "ISOLATION", "COMPOUND"],
                    "전신 루틴의 몸통 안정성 보완",
                )
            )
        return slots
    if split == "core":
        slots = [
            _slot(
                "core_main",
                "isolation_accessory",
                "required_when_safe",
                ["abs", "obliques"],
                ["ISOLATION", "STATIC", "COMPOUND"],
                "복근/복사근 중심 코어 메인 운동",
            )
        ]
        if target_count >= 4 and "lower-back" in target_muscles:
            slots.append(
                _slot(
                    "core_posterior_support",
                    "isolation_accessory",
                    "preferred_when_safe",
                    ["lower-back"],
                    ["ISOLATION", "STATIC"],
                    "안전할 때만 포함하는 후면 코어 보완",
                )
            )
        return slots

    if split == "shoulder":
        explicit_targets = list(target_muscles[: max(1, target_count)])
        slots: list[RoutineBlueprintSlot] = []
        if "front-deltoids" in explicit_targets:
            slots.append(
                _slot(
                    "shoulder_compound_main",
                    "main_compound",
                    "preferred_when_safe",
                    ["front-deltoids"],
                    ["COMPOUND"],
                    "안전 조건과 승인 후보가 모두 충족될 때 우선하는 어깨 복합 운동",
                    selection_patterns=["vertical_press"],
                )
            )
        slots.extend(
            _slot(
                f"target_{index + 1}_{muscle.replace('-', '_')}",
                "isolation_accessory",
                "required_when_safe",
                [muscle],
                ["ISOLATION", "COMPOUND", "STATIC"],
                f"사용자가 직접 선택한 {muscle} 부위 운동",
            )
            for index, muscle in enumerate(explicit_targets)
            if muscle != "front-deltoids"
        )
        return slots

    # Explicit custom targets are authoritative. Give each selected muscle its
    # own required slot (within session capacity) so a shoulder+triceps request
    # cannot collapse into triceps-only work.
    explicit_targets = list(target_muscles[: max(1, target_count)])
    if len(explicit_targets) > 1:
        return [
            _slot(
                f"target_{index + 1}_{muscle.replace('-', '_')}",
                "main_compound"
                if muscle in LARGE_MUSCLE_SLUGS
                else "isolation_accessory",
                "required_when_safe",
                [muscle],
                ["COMPOUND", "ISOLATION"]
                if muscle in LARGE_MUSCLE_SLUGS
                else ["ISOLATION", "COMPOUND", "STATIC"],
                f"사용자가 직접 선택한 {muscle} 부위 운동",
            )
            for index, muscle in enumerate(explicit_targets)
        ]

    large_targets = [
        muscle for muscle in target_muscles if muscle in LARGE_MUSCLE_SLUGS
    ]
    main_targets = large_targets or list(target_muscles)
    return [
        _slot(
            "target_main",
            "main_compound" if large_targets else "isolation_accessory",
            "required_when_safe",
            main_targets,
            ["COMPOUND", "ISOLATION"]
            if large_targets
            else ["ISOLATION", "COMPOUND", "STATIC"],
            "지정 부위 메인 운동",
        )
    ]


def _infer_split_from_targets(target_muscles: tuple[str, ...]) -> str:
    targets = set(target_muscles)
    if targets <= {"abs", "obliques", "lower-back"} and targets & {"abs", "obliques"}:
        return "core"
    if "chest" in targets and "upper-back" in targets:
        return "chest_back"
    if "chest" in targets and (targets & {"triceps", "front-deltoids"}):
        return "push"
    if "upper-back" in targets and (targets & {"biceps", "trapezius", "back-deltoids"}):
        return "pull"
    if {"quadriceps", "hamstring"} <= targets or (
        "quadriceps" in targets and "gluteal" in targets
    ):
        return "legs"
    return "custom"


def _specialization_slot_id(
    slots: Sequence[RoutineBlueprintSlot],
    focus_code: str | None,
) -> str | None:
    focus_targets = SPECIALIZATION_FOCUS_TARGETS.get(str(focus_code or ""))
    if not focus_targets:
        return None
    direct_slot = next(
        (slot for slot in slots if slot.slot_id == "specialization_focus"),
        None,
    )
    if direct_slot is not None:
        return direct_slot.slot_id
    if focus_code == "latEmphasis":
        vertical_slot = next(
            (slot for slot in slots if "vertical_pull" in slot.preferred_selection_patterns),
            None,
        )
        if vertical_slot is not None:
            return vertical_slot.slot_id
    matched = [
        slot
        for slot in slots
        if not slot.adjunct_option and focus_targets.intersection(slot.target_muscles)
    ]
    if not matched:
        return None
    matched.sort(
        key=lambda slot: (
            PHASE_ORDER.index(slot.phase),
            slot.slot_id,
        )
    )
    return matched[0].slot_id


def _is_direct_hamstring_hinge(candidate: dict) -> bool:
    """Keep the advanced hamstring focus role distinct from a curl or thrust."""
    if _candidate_primary(candidate) != "hamstring":
        return False
    hinge_load = str(candidate.get("hip_hinge_load") or "").strip().lower()
    if hinge_load == "high":
        return True
    name = _candidate_name(candidate).lower()
    return any(token in name for token in (
        "루마니안", "데드리프트", "rdl", "romanian", "deadlift", "good morning",
    ))


def _training_method_slot_id(
    slots: Sequence[RoutineBlueprintSlot],
    *,
    specialization_slot_id: str | None,
) -> str | None:
    if specialization_slot_id:
        focused = next(
            (slot for slot in slots if slot.slot_id == specialization_slot_id),
            None,
        )
        if focused and focused.phase in {"main_compound", "secondary_compound"}:
            return focused.slot_id
    return next(
        (
            slot.slot_id
            for slot in slots
            if not slot.adjunct_option and slot.phase == "main_compound"
        ),
        None,
    )


def blueprint_invariant_errors(blueprint: RoutineBlueprint) -> list[str]:
    errors: list[str] = []
    slot_ids = [slot.slot_id for slot in blueprint.slots]
    safety_limited = set(blueprint.safety_limited_target_muscles)
    required_slots = [
        slot
        for slot in blueprint.slots
        if slot.requirement == "required_when_safe"
        and not (slot.target_muscles and set(slot.target_muscles) <= safety_limited)
    ]
    if len(slot_ids) != len(set(slot_ids)):
        errors.append("duplicate_slot_id")
    if blueprint.minimum_main_exercise_count < 1:
        errors.append("minimum_main_count_below_one")
    if blueprint.minimum_main_exercise_count > blueprint.target_main_exercise_count:
        errors.append("minimum_main_count_above_main_target")
    if blueprint.minimum_exercise_count < 1:
        errors.append("minimum_count_below_one")
    if blueprint.minimum_exercise_count > blueprint.target_exercise_count:
        errors.append("minimum_count_above_target")
    if len(required_slots) > blueprint.target_main_exercise_count:
        errors.append("required_slots_above_main_target")
    if blueprint.target_exercise_count != (
        blueprint.target_main_exercise_count + len(blueprint.planned_adjunct_options)
    ):
        errors.append("total_target_does_not_match_main_plus_adjunct")
    if blueprint.minimum_exercise_count != (
        blueprint.minimum_main_exercise_count + len(blueprint.planned_adjunct_options)
    ):
        errors.append("total_minimum_does_not_match_main_plus_adjunct")
    if blueprint.flex_slot_count < 0:
        errors.append("negative_flex_slot_count")
    if blueprint.max_total_sets < len(required_slots):
        errors.append("set_budget_below_required_slots")
    if any(not slot.target_muscles for slot in blueprint.slots if not slot.adjunct_option):
        errors.append("empty_slot_target")
    return errors


def build_routine_blueprint(
    *,
    target_split_label: str | None,
    target_muscles: Sequence[str] | None,
    goal: str | None,
    experience_level: str | None,
    readiness_level: str | None,
    time_available_min: int,
    advanced_chest_focus: str | None = "balanced",
    specialization_focus: str | None = None,
    training_method: str | None = None,
    method_mode: str | None = None,
    requested_method: str | None = None,
    recent_training_method_uses: Sequence[Any] | None = None,
    training_style: str | None = "gym",
    session_structure: str | None = "standard",
    adjunct_options: Sequence[str] | None = None,
    pain_areas: Sequence[Any] | None = None,
    doms_data: Mapping[str, Any] | None = None,
    injury_conditions: Sequence[Any] | None = None,
) -> RoutineBlueprint:
    requested_split = _norm(target_split_label, "custom")
    muscles = _unique(target_muscles or SPLIT_LABEL_TO_MUSCLES.get(requested_split, []))
    split = requested_split if requested_split != "custom" else _infer_split_from_targets(muscles)
    normalized_goal = _norm(goal, "hypertrophy").replace("_", "")
    experience = _norm(experience_level, "intermediate")
    chest_focus = str(advanced_chest_focus or "balanced").strip()
    if chest_focus not in {"balanced", "upperChest", "lowerChest", "pausedPress"}:
        raise ValueError("unsupported advanced chest focus")
    legacy_focus, legacy_method = legacy_specialization_values(chest_focus)
    resolved_focus = str(specialization_focus or legacy_focus or "").strip() or None
    explicit_method = str(
        requested_method or training_method or legacy_method or ""
    ).strip() or None
    resolved_method_mode = str(
        method_mode or ("manual" if explicit_method else "auto")
    ).strip()
    if explicit_method and resolved_method_mode != "manual":
        raise ValueError("requested training method requires manual method mode")
    method_resolution = resolve_training_method(
        method_mode=resolved_method_mode,
        requested_method=explicit_method,
        goal=normalized_goal,
        experience_level=experience,
        readiness_level=readiness_level,
        target_muscles=muscles,
        target_split_label=split,
        recent_method_uses=recent_training_method_uses,
        pain_areas=pain_areas,
        doms_data=doms_data,
        injury_conditions=injury_conditions,
    )
    resolved_method = method_resolution.method_code
    if resolved_focus and experience != "advanced":
        raise ValueError("specialization focus requires advanced experience")
    if resolved_focus and not focus_matches_targets(resolved_focus, muscles):
        raise ValueError("specialization focus does not match the selected target muscles")
    if resolved_method and experience != "advanced":
        raise ValueError("training method requires advanced experience")
    readiness = _norm(readiness_level, "normal")
    if resolved_method and readiness == "low":
        raise ValueError("training method is unavailable when readiness is low")
    if resolved_method and not training_method_supports_goal(
        resolved_method,
        normalized_goal,
    ):
        raise ValueError("training method does not support the selected goal")
    structure = _norm(session_structure, "standard")
    style = _norm(training_style, "gym")
    base_target_count = target_exercise_count_for_session(
        int(time_available_min),
        normalized_goal,
        experience,
    )
    # A single-muscle strength session is not a general full-session split.
    # Three distinct roles (priority compound, secondary compound, isolation)
    # provide enough work without padding the plan with redundant variations.
    if normalized_goal == "strength" and len(muscles) == 1:
        single_target_cap = 2 if int(time_available_min) <= 30 else 3
        base_target_count = min(base_target_count, single_target_cap)
    safety_limited = _safety_limited_targets(
        muscles,
        pain_areas=pain_areas,
        doms_data=doms_data,
    )
    safe_target_count = max(0, len(muscles) - len(safety_limited))
    target_count = base_target_count
    if muscles and safety_limited:
        target_count = max(
            1,
            math.ceil(base_target_count * safe_target_count / len(muscles)),
        )
    max_sets = _calculate_max_total_sets(int(time_available_min), normalized_goal)
    slots = _split_slots(
        split,
        target_count,
        muscles,
        goal=normalized_goal,
        training_style=style,
    )
    from .pull_composition import complement_scope

    expanded_pull = complement_scope(
        split=split, experience=experience, goal=normalized_goal, readiness=readiness,
        minutes=time_available_min, structure=structure, style=style,
        focus=resolved_focus, method=resolved_method, adjuncts=adjunct_options,
        pain=pain_areas, injuries=injury_conditions, doms=doms_data,
    )
    if expanded_pull:
        slots = [replace(slot, preferred_selection_patterns=("horizontal_row",))
                 if slot.slot_id == "pull_main" else
                 replace(slot, requirement="preferred_when_requested_and_safe")
                 if slot.slot_id == "pull_vertical_main" else slot for slot in slots]
        slots.insert(2, _slot(
            "pull_supported_row", "secondary_compound", "preferred_when_requested_and_safe",
            ["upper-back"], ["COMPOUND"],
            "벤트오버 로우 이후 가슴 지지 셋업으로 등 당기기를 보완",
            selection_patterns=["horizontal_row"],
        ))
    # v10: reserve distinct back work within the nominal gym-session capacity,
    # not by shortening heavy-set rests or increasing per-exercise set caps.
    expanded_upper_strength = (
        split == "upper"
        and {"chest", "upper-back", "biceps", "triceps"} <= set(muscles)
        and experience == "advanced"
        and normalized_goal == "strength"
        and readiness in {"normal", "high"}
        and structure in {"standard", "density"}
        and int(time_available_min) >= 60
        and not pain_areas
        and not injury_conditions
        and not safety_limited
        and not any(int(level or 0) > 0 for level in (doms_data or {}).values())
    )
    if expanded_upper_strength:
        target_count = max(target_count, 5)
        slots = [
            replace(slot, preferred_selection_patterns=("horizontal_row",))
            if slot.slot_id == "upper_pull_main" else slot
            for slot in slots
        ]
        slots.append(_slot(
            "upper_vertical_pull", "secondary_compound",
            "preferred_when_requested_and_safe",
            ["upper-back", "trapezius"], ["COMPOUND", "ISOLATION"],
            "상급 상체 근력 세션: 수평 로우를 보완하는 수직 당기기",
            selection_patterns=["vertical_pull"],
        ))
        for muscle, label in (("biceps", "이두"), ("triceps", "삼두")):
            slots.append(_slot(
                f"upper_{muscle}_direct", "isolation_accessory",
                "preferred_when_requested_and_safe", [muscle], ["ISOLATION"],
                f"상체 메인 운동 이후 직접 {label} 보완",
            ))
    if split == "full_body" and experience == "advanced" and target_count >= 6:
        slots = [
            _slot("full_vertical_pull", "secondary_compound", "required_when_safe",
                  ["upper-back", "trapezius"], ["COMPOUND", "ISOLATION"],
                  "상급 전신 루틴의 수평 로우와 구분되는 수직 당기기",
                  selection_patterns=["vertical_pull"])
            if slot.slot_id == "full_core_accessory" else slot
            for slot in slots
        ]
    if split == "chest" and set(muscles) == {"chest"}:
        # A time budget cannot require more chest exercises than the declared
        # press/press/isolation roles permit. This is independent of DB supply;
        # missing safe candidates must still fail coverage/underfill checks.
        target_count = min(target_count, sum(slot.max_exercises for slot in slots))
    focus_targets = SPECIALIZATION_FOCUS_TARGETS.get(str(resolved_focus or ""))
    main_slot_count_before_focus = sum(not slot.adjunct_option for slot in slots)
    # An advanced specialization exchanges one optional/flex place for direct
    # focus work. It never extends the time-derived exercise capacity.
    reserve_direct_focus_slot = bool(
        focus_targets
        and experience == "advanced"
        and main_slot_count_before_focus < target_count
        and not (
            resolved_focus == "latEmphasis"
            and any("vertical_pull" in slot.preferred_selection_patterns for slot in slots)
        )
    )
    if reserve_direct_focus_slot:
        slots.append(
            _slot(
                "specialization_focus",
                "secondary_compound",
                "required_when_safe",
                sorted(focus_targets),
                ["ISOLATION", "COMPOUND"],
                "사용자가 명시적으로 선택한 국소 부위를 직접 보완하는 운동",
                selection_patterns=_FOCUS_SLOT_SELECTION_PATTERNS.get(
                    str(resolved_focus or ""),
                    (),
                ),
            )
        )
    elif focus_targets and not any(
        focus_targets.intersection(slot.target_muscles)
        for slot in slots
        if not slot.adjunct_option
    ):
        slots.append(
            _slot(
                "specialization_focus",
                "isolation_accessory",
                "required_when_safe",
                sorted(focus_targets),
                ["ISOLATION", "COMPOUND"],
                "사용자가 명시적으로 선택한 국소 부위를 직접 보완하는 운동",
            )
        )
    # A requested focus needs an explicit role before generic flex filling.
    # Reuse existing capacity, not a new exercise-count or volume allowance.
    sore_parts = _expand_safety_body_parts(
        part for part, level in (doms_data or {}).items() if int(level or 0) > 0
    )
    complement_allowed = (
        resolved_focus in {"calvesEmphasis", "trapeziusEmphasis"}
        and experience == "advanced"
        and normalized_goal in {"hypertrophy", "recomposition"}
        and readiness in {"normal", "high"}
        and structure == "standard"
        and int(time_available_min) >= 50
        and target_count >= 5
        and not pain_areas
        and not injury_conditions
        and not safety_limited
        and not (set(focus_targets or ()) & sore_parts)
    )
    if complement_allowed and resolved_focus == "calvesEmphasis":
        slots = [
            replace(
                slot,
                requirement="preferred_when_requested_and_safe",
                purpose="종아리 집중: 첫 직접 운동과 무릎 자세가 다른 카프 운동을 후보가 있을 때 보완",
            ) if slot.slot_id == "calves_accessory" else slot
            for slot in slots
        ]
    elif complement_allowed and len(slots) < target_count:
        slots.append(_slot(
            "specialization_complement",
            "isolation_accessory",
            "preferred_when_requested_and_safe",
            ["trapezius"],
            ["ISOLATION"],
            "승모 집중: 같은 운동 중복을 피하고 허용된 다른 부하 도구로 직접 운동량을 보완",
        ))

    # Glute focus may use one already-admitted abduction accessory. This consumes
    # existing flex capacity, not a new main target or a taxonomy remapping.
    if (
        split == "custom"
        and muscles == ("gluteal",)
        and normalized_goal == "hypertrophy"
        and style == "gym"
        and readiness in {"normal", "high"}
        and structure == "standard"
        and int(time_available_min) >= 45
        and not resolved_focus
        and not adjunct_options
        and not pain_areas
        and not injury_conditions
        and not safety_limited
        and not ({"gluteal", "abductors"} & sore_parts)
        and len(slots) < target_count
    ):
        from .glute_support import SUPPORT_SLOT

        slots.append(_slot(
            SUPPORT_SLOT, "isolation_accessory", "preferred_when_requested_and_safe",
            ["abductors"], ["ISOLATION"],
            "둔근 집중 요청에서 메인 신전 운동 뒤에 엉덩이 옆쪽 외전근 운동을 보완",
        ))

    pain_body_parts = _pain_body_parts(pain_areas)
    if pain_body_parts & {"lower-back", "back", "lumbar"}:
        # A registered lumbar injury can remove every compound lower-body row.
        # Preserve the intended quadriceps/gluteal slot while allowing a
        # supported isolation replacement rather than forcing an unsafe lift or
        # silently omitting the main lower-body pattern.
        slots = [
            _slot(
                slot.slot_id,
                slot.phase,
                slot.requirement,
                slot.target_muscles,
                ("COMPOUND", "ISOLATION")
                if slot.slot_id in {"lower_main", "full_lower_main"}
                else slot.preferred_movement_types,
                "허리 부상 제한을 반영한 지지형 하체 메인 운동"
                if slot.slot_id in {"lower_main", "full_lower_main"}
                else slot.purpose,
                adjunct_option=slot.adjunct_option,
            )
            for slot in slots
        ]

    requested_adjuncts = canonical_adjunct_options(adjunct_options or [])
    if not requested_adjuncts and style in ADJUNCT_STYLES:
        legacy_option = LEGACY_STYLE_TO_ADJUNCT_OPTION.get(style)
        requested_adjuncts = (legacy_option,) if legacy_option else ()

    suppressed_adjuncts: list[str] = []
    suppression_reasons: dict[str, str] = {}
    selected_adjuncts: list[str] = []
    for option in requested_adjuncts:
        if option not in ADJUNCT_OPTION_PHASE:
            suppressed_adjuncts.append(option)
            suppression_reasons[option] = "postValidationRemoved"
            continue
        if readiness == "low" and option in HIGH_DENSITY_ADJUNCTS:
            suppressed_adjuncts.append(option)
            suppression_reasons[option] = "postValidationRemoved"
            continue
        if any(
            option in group and bool(set(selected_adjuncts) & group)
            for group in MUTUALLY_EXCLUSIVE_ADJUNCT_GROUPS
        ):
            suppressed_adjuncts.append(option)
            suppression_reasons[option] = "postValidationRemoved"
            continue
        if len(selected_adjuncts) >= MAX_DISTINCT_ADJUNCT_OPTIONS:
            suppressed_adjuncts.append(option)
            suppression_reasons[option] = "slotCapacityAfterMainMinimum"
            continue
        selected_adjuncts.append(option)

    safety_limited_set = set(safety_limited)
    required_count = sum(
        slot.requirement == "required_when_safe"
        and not (slot.target_muscles and set(slot.target_muscles) <= safety_limited_set)
        for slot in slots
    )
    main_target_count = max(target_count, required_count, 1)
    # Low readiness reduces per-exercise dose/effort, not role coverage again.
    normal_main_minimum = (
        main_target_count if main_target_count <= 3 else main_target_count - 1
    )
    minimum_main_count = min(
        main_target_count,
        max(required_count, normal_main_minimum),
    )
    adjuncts_to_plan = tuple(selected_adjuncts)
    if adjuncts_to_plan:
        # Adjunct options share the session's exercise-count budget whenever
        # optional main capacity exists. They may never displace the safe main
        # minimum or required split slots.
        main_target_count = max(
            minimum_main_count,
            main_target_count - len(adjuncts_to_plan),
        )
        # Reserve actual slot capacity as well as the numeric count. Otherwise
        # selecting every explicit slot can truncate the requested finisher.
        eligible_main_slots = [
            (index, slot)
            for index, slot in enumerate(slots)
            if not slot.adjunct_option
            and not (slot.target_muscles and set(slot.target_muscles) <= safety_limited_set)
        ]
        overflow = max(0, len(eligible_main_slots) - main_target_count)
        omitted_optional_slots = set()
        for index, slot in reversed(eligible_main_slots):
            if overflow and slot.requirement == "preferred_when_safe":
                omitted_optional_slots.add(index)
                overflow -= 1
        slots = [slot for index, slot in enumerate(slots) if index not in omitted_optional_slots]

    for option in adjuncts_to_plan:
        slots.append(_slot(
            f"adjunct_{option}",
            ADJUNCT_OPTION_PHASE[option],
            "preferred_when_requested_and_safe",
            [],
            ["ISOLATION", "STATIC", "COMPOUND"],
            "헬스장 메인 루틴의 필수 구성을 보존하면서 요청 목적을 보완하는 운동",
            adjunct_option=option,
        ))
    slots.sort(key=lambda slot: (
        PHASE_ORDER.index(slot.phase),
        0 if slot.slot_id == "specialization_complement" else 1,
    ))
    focus_slot_id = _specialization_slot_id(slots, resolved_focus)
    method_slot_id = (
        _training_method_slot_id(
            slots,
            specialization_slot_id=focus_slot_id,
        )
        if resolved_method
        else None
    )
    target_count = main_target_count + len(adjuncts_to_plan)
    minimum_count = minimum_main_count + len(adjuncts_to_plan)
    main_slot_count = sum(not slot.adjunct_option for slot in slots)
    flex_count = max(0, main_target_count - main_slot_count)
    blueprint = RoutineBlueprint(
        formula_version=("gym-routine-blueprint-v11" if expanded_pull else
                         "gym-routine-blueprint-v10" if expanded_upper_strength else "gym-routine-blueprint-v9"),
        target_split_label=split,
        target_muscles=muscles,
        goal=normalized_goal,
        experience_level=experience,
        advanced_chest_focus=chest_focus,
        specialization_focus=resolved_focus,
        training_method_mode=resolved_method_mode,
        training_method=resolved_method,
        training_method_source=method_resolution.source,
        training_method_reason_code=method_resolution.reason_code,
        specialization_slot_id=focus_slot_id,
        training_method_slot_id=method_slot_id,
        readiness_level=readiness,
        session_structure=structure,
        base_target_exercise_count=base_target_count,
        target_main_exercise_count=main_target_count,
        target_exercise_count=target_count,
        minimum_main_exercise_count=minimum_main_count,
        minimum_exercise_count=minimum_count,
        safety_limited_target_muscles=safety_limited,
        max_total_sets=max_sets,
        phase_order=PHASE_ORDER,
        candidate_preferences=EXPERIENCE_PREFERENCES.get(experience, EXPERIENCE_PREFERENCES["intermediate"]),
        requested_adjunct_options=requested_adjuncts,
        planned_adjunct_options=adjuncts_to_plan,
        suppressed_adjunct_options=tuple(suppressed_adjuncts),
        adjunct_suppression_reasons=tuple(suppression_reasons.items()),
        execution_policy=_execution_policy(
            structure,
            split=split,
            goal=normalized_goal,
            readiness=readiness,
        ),
        slots=tuple(slots),
        flex_slot_count=flex_count,
        llm_role="choose_exercise_id_from_each_explicit_slot_bundle_only",
    )
    errors = blueprint_invariant_errors(blueprint)
    if errors:
        raise RoutineBlueprintValidationError(errors)
    return blueprint


def render_blueprint_for_prompt(blueprint: RoutineBlueprint) -> str:
    return json.dumps(blueprint.to_dict(), ensure_ascii=False, separators=(",", ":"))


def _candidate_id(candidate: dict) -> str:
    return str(candidate.get("id") or "").strip()


def _candidate_name(candidate: dict) -> str:
    return candidate_display_name(candidate)


def _candidate_primary(candidate: dict) -> str:
    return str(candidate.get("primary_muscle") or "").strip().lower().replace("_", "-")


def _candidate_movement(candidate: dict) -> str:
    return str(candidate.get("movement_type") or "UNKNOWN").strip().upper()


def _candidate_is_back_main_pull(candidate: dict) -> bool:
    pattern = candidate_selection_pattern_key(candidate)
    if pattern not in {"horizontal_row", "vertical_pull"}:
        return False
    identity = " ".join(
        str(candidate.get(key) or "").strip().lower()
        for key in (
            "name_kr",
            "name_en",
            "display_name_ko",
            "base_exercise_name_kr",
            "variation_name_kr",
            "user_facing_display_name_kr",
        )
    )
    compact_identity = re.sub(r"[\s/_()·,.-]+", "", identity)
    return not any(
        token in compact_identity
        for token in ("업라이트로우", "uprightrow")
    )


def _is_adjunct_candidate(candidate: dict) -> bool:
    searchable = " ".join(
        [
            " ".join(str(item) for item in (candidate.get("score_reasons") or [])),
            " ".join(str(item.get("rule_code") or "") for item in (candidate.get("score_boosts") or [])),
            str(candidate.get("training_style_tags") or ""),
        ]
    ).lower()
    return any(
        token in searchable
        for token in [
            "training_style_match_boost",
            "adjunct style match",
            "mobility",
            "yoga_pilates",
            "rehab_prehab",
            "conditioning",
        ]
    )


def _candidate_matches_slot(
    candidate: dict,
    slot: RoutineBlueprintSlot,
    *,
    experience_level: str,
    goal: str,
    advanced_chest_focus: str,
    specialization_focus: str | None,
    training_method: str | None,
    specialization_slot_id: str | None,
    training_method_slot_id: str | None,
) -> bool:
    if not automatic_recommendation_eligibility(candidate, experience_level,
        adjunct_options=(slot.adjunct_option,) if slot.adjunct_option else ()).eligible:
        return False
    if not _candidate_id(candidate):
        return False
    from .glute_support import SUPPORT_SLOT, is_reviewed_glute_support

    if slot.slot_id == SUPPORT_SLOT and not is_reviewed_glute_support(candidate):
        return False
    from .pull_composition import reviewed_row_setup

    setup = reviewed_row_setup(candidate)
    if slot.slot_id == "pull_supported_row" and setup != "chest_supported":
        return False
    if setup == "chest_supported" and slot.slot_id != "pull_supported_row":
        return False
    if slot.adjunct_option:
        return bool(
            slot.adjunct_option
            and candidate_matches_adjunct_option(candidate, slot.adjunct_option)
        )
    if str(candidate.get("main_routine_default_allowed", "")).lower() in {"0", "false"}:
        return False
    primary = _candidate_primary(candidate)
    movement = _candidate_movement(candidate)
    if primary not in set(slot.target_muscles):
        return False
    if slot.preferred_movement_types and movement not in set(slot.preferred_movement_types):
        return False
    if (
        slot.slot_id in {"pull_main", "upper_pull_main", "full_pull_main"}
        and not _candidate_is_back_main_pull(candidate)
    ):
        return False
    if (
        slot.preferred_selection_patterns
        and candidate_selection_pattern_key(candidate)
        not in set(slot.preferred_selection_patterns)
    ):
        return False
    searchable_name = " ".join(
        str(candidate.get(key) or "").strip().lower()
        for key in (
            "name_kr",
            "name_en",
            "variation_name_kr",
            "catalog_name_kr",
            "catalog_name_en",
        )
    )
    bench_angle = str(candidate.get("bench_angle_band") or "").strip().lower()
    is_incline = bench_angle.startswith("incline") or any(
        token in searchable_name for token in ("인클라인", "incline")
    )
    is_decline = bench_angle == "decline" or any(
        token in searchable_name for token in ("디클라인", "decline")
    )
    is_flat = bench_angle == "flat" or any(
        token in searchable_name for token in ("플랫", "flat")
    )
    is_paused = any(
        token in searchable_name for token in ("정지", "paused", "pause")
    )
    resolved_chest_focus = (
        specialization_focus
        if specialization_focus in {"upperChest", "lowerChest"}
        else "balanced"
    )
    if slot.slot_id == "chest_main":
        if resolved_chest_focus == "upperChest" and not is_incline:
            return False
        if resolved_chest_focus == "lowerChest" and not is_decline:
            return False
        if resolved_chest_focus == "balanced" and (is_incline or is_decline):
            return False
    if slot.slot_id == "chest_secondary":
        if resolved_chest_focus == "upperChest" and is_incline:
            return False
        if resolved_chest_focus == "lowerChest" and is_decline:
            return False
        if resolved_chest_focus == "balanced" and not (is_incline or is_decline):
            return False

    if (
        specialization_focus
        and slot.slot_id == specialization_slot_id
        and not candidate_matches_specialization_focus(candidate, specialization_focus)
    ):
        return False
    if (
        specialization_focus == "hamstringEmphasis"
        and slot.slot_id == specialization_slot_id
        and not _is_direct_hamstring_hinge(candidate)
    ):
        return False

    method_match = candidate_matches_training_method(candidate, training_method)
    if training_method and slot.slot_id == training_method_slot_id and not method_match:
        return False

    explicit_method_main = (
        slot.slot_id == training_method_slot_id
        and method_match
        and experience_level == "advanced"
        and str(candidate.get("failure_penalty") or "").strip().lower()
        in {"", "low", "medium"}
    )
    if (
        slot.phase == "main_compound"
        and movement == "COMPOUND"
        and not explicit_method_main
        and not main_role_eligibility(
            candidate,
            experience_level,
            goal,
        ).eligible
    ):
        return False
    return True


def build_blueprint_candidate_bundles(
    blueprint: RoutineBlueprint,
    ranked_candidates: Sequence[dict],
    *,
    top_k_per_slot: int = 3,
) -> list[dict]:
    """Create disjoint candidate choices for each explicit blueprint slot."""
    effective_focus = blueprint.specialization_focus
    focus_slot = next(
        (slot for slot in blueprint.slots
         if slot.slot_id == blueprint.specialization_slot_id), None,
    )
    focus_target_muscles = set(focus_slot.target_muscles) if focus_slot else set()
    focus_unavailable = bool(
        blueprint.target_split_label == "chest"
        and effective_focus in {"upperChest", "midChest", "lowerChest"}
        and focus_slot is not None
        and not any(
            _candidate_matches_slot(
                candidate, focus_slot,
                experience_level=blueprint.experience_level,
                goal=blueprint.goal,
                advanced_chest_focus=blueprint.advanced_chest_focus,
                specialization_focus=effective_focus,
                training_method=blueprint.training_method,
                specialization_slot_id=blueprint.specialization_slot_id,
                training_method_slot_id=blueprint.training_method_slot_id,
            )
            for candidate in ranked_candidates
        )
    )
    if focus_unavailable:
        # Focus is optional emphasis, not permission to drop an otherwise valid
        # main role. Reuse only already-filtered candidates and keep method guards.
        effective_focus = None
    requested_adjunct_options = {
        slot.adjunct_option
        for slot in blueprint.slots
        if slot.adjunct_option
    }
    explicitly_available_adjunct_options = {
        option
        for option in requested_adjunct_options
        if any(
            option in candidate_adjunct_option_codes(candidate)
            and candidate_matches_adjunct_option(candidate, option)
            for candidate in ranked_candidates
        )
    }
    reserved_adjunct_ids = {
        _candidate_id(candidate).lower()
        for candidate in ranked_candidates
        if _candidate_id(candidate)
        and any(
            candidate_matches_adjunct_option(candidate, option)
            for option in requested_adjunct_options
        )
    }
    used_ids: set[str] = set()
    claimed_focus_families: set[str] = set()
    bundles: list[dict] = []
    for slot_index, slot in enumerate(blueprint.slots):
        matched: list[dict] = []
        matched_family_keys: set[str] = set()
        matched_display_keys: set[str] = set()
        duplicate_family_suppressed_count = 0
        experience_role_rejected_count = 0
        # Candidate bundles are disjoint. Keep one compatible alternative in
        # each ordinary focus-target slot so the dedicated focus role can make
        # the requested pattern visible instead of being starved by a broad
        # top-three bundle earlier in the plan.
        focus_bound_slot = bool(
            focus_slot
            and (
                slot.slot_id == focus_slot.slot_id
                or bool(focus_target_muscles.intersection(slot.target_muscles))
            )
        )
        reviewed_forearm_focus_alternatives = bool(
            focus_slot
            and slot.slot_id == focus_slot.slot_id
            and effective_focus == "forearmEmphasis"
            and blueprint.experience_level != "beginner"
        )
        slot_candidate_limit = (
            min(2, max(1, top_k_per_slot))
            if reviewed_forearm_focus_alternatives
            else 1
            if focus_bound_slot
            else max(1, top_k_per_slot)
        )
        later_selection_patterns = {
            pattern
            for later_slot in blueprint.slots[slot_index + 1:]
            for pattern in later_slot.preferred_selection_patterns
        }
        later_slots = tuple(
            later_slot
            for later_slot in blueprint.slots[slot_index + 1:]
            if not later_slot.adjunct_option
        )
        later_target_muscles = {
            muscle
            for later_slot in later_slots
            for muscle in later_slot.target_muscles
        }
        overlapping_later_targets = set(slot.target_muscles) & later_target_muscles
        current_unique_targets = set(slot.target_muscles) - later_target_muscles
        reserve_later_patterns = bool(
            slot.requirement == "required_when_safe"
            and not slot.preferred_selection_patterns
            and later_selection_patterns
            and any(
                _candidate_matches_slot(
                    candidate,
                    slot,
                    experience_level=blueprint.experience_level,
                    goal=blueprint.goal,
                    advanced_chest_focus=blueprint.advanced_chest_focus,
                    specialization_focus=effective_focus,
                    training_method=blueprint.training_method,
                    specialization_slot_id=blueprint.specialization_slot_id,
                    training_method_slot_id=blueprint.training_method_slot_id,
                )
                and candidate_selection_pattern_key(candidate)
                not in later_selection_patterns
                for candidate in ranked_candidates
            )
        )
        reserve_later_targets = bool(
            slot.requirement == "required_when_safe"
            and overlapping_later_targets
            and current_unique_targets
            and any(
                _candidate_id(candidate).lower() not in used_ids
                and _candidate_id(candidate).lower() not in reserved_adjunct_ids
                and _candidate_primary(candidate) in current_unique_targets
                and (
                    not reserve_later_patterns
                    or candidate_selection_pattern_key(candidate)
                    not in later_selection_patterns
                )
                and _candidate_matches_slot(
                    candidate,
                    slot,
                    experience_level=blueprint.experience_level,
                    goal=blueprint.goal,
                    advanced_chest_focus=blueprint.advanced_chest_focus,
                    specialization_focus=effective_focus,
                    training_method=blueprint.training_method,
                    specialization_slot_id=blueprint.specialization_slot_id,
                    training_method_slot_id=blueprint.training_method_slot_id,
                )
                for candidate in ranked_candidates
            )
        )
        independently_covered_later_slots: set[str] = set()
        if reserve_later_targets:
            def matches_role(candidate: dict, role: RoutineBlueprintSlot) -> bool:
                return _candidate_matches_slot(
                    candidate,
                    role,
                    experience_level=blueprint.experience_level,
                    goal=blueprint.goal,
                    advanced_chest_focus=blueprint.advanced_chest_focus,
                    specialization_focus=effective_focus,
                    training_method=blueprint.training_method,
                    specialization_slot_id=blueprint.specialization_slot_id,
                    training_method_slot_id=blueprint.training_method_slot_id,
                )

            # Reserve only threatened roles; a dedicated future candidate needs no substitute.
            for later_index in range(slot_index + 1, len(blueprint.slots)):
                later_slot = blueprint.slots[later_index]
                if later_slot.adjunct_option:
                    continue
                if any(
                    _candidate_id(candidate).lower() not in used_ids
                    and _candidate_id(candidate).lower() not in reserved_adjunct_ids
                    and matches_role(candidate, later_slot)
                    and not any(
                        matches_role(candidate, earlier_slot)
                        for earlier_slot in blueprint.slots[slot_index:later_index]
                    )
                    for candidate in ranked_candidates
                ):
                    independently_covered_later_slots.add(later_slot.slot_id)
        ordered_candidates = list(ranked_candidates)
        if focus_slot and focus_target_muscles.intersection(slot.target_muscles):
            ordered_candidates.sort(key=lambda candidate: (
                focus_variant_family_key(candidate, effective_focus, candidate_selection_family_key(candidate))
                in claimed_focus_families,
            ))
        if slot.slot_id == "chest_isolation" and effective_focus == "lowerChest":
            # Prefer a reviewed focus alternative among already filtered
            # candidates. Keep the generic isolation when none is available.
            ordered_candidates.sort(key=lambda candidate: not (
                candidate.get("core_chest_focus_reviewed")
                and candidate_matches_specialization_focus(candidate, effective_focus)
            ))
        for candidate in ordered_candidates:
            exercise_id = _candidate_id(candidate)
            normalized_id = exercise_id.lower()
            if normalized_id in used_ids:
                continue
            if not slot.adjunct_option and normalized_id in reserved_adjunct_ids:
                continue
            if (
                reviewed_forearm_focus_alternatives
                and matched
                and not candidate.get("core_forearm_pattern_reviewed")
            ):
                continue
            if (
                reserve_later_patterns
                and candidate_selection_pattern_key(candidate)
                in later_selection_patterns
            ):
                continue
            if (
                reserve_later_targets
                and _candidate_primary(candidate) in overlapping_later_targets
                and any(
                    later_slot.slot_id not in independently_covered_later_slots
                    and _candidate_matches_slot(
                        candidate,
                        later_slot,
                        experience_level=blueprint.experience_level,
                        goal=blueprint.goal,
                        advanced_chest_focus=blueprint.advanced_chest_focus,
                        specialization_focus=effective_focus,
                        training_method=blueprint.training_method,
                        specialization_slot_id=blueprint.specialization_slot_id,
                        training_method_slot_id=blueprint.training_method_slot_id,
                    )
                    for later_slot in later_slots
                )
            ):
                continue
            if (
                slot.adjunct_option
                and slot.adjunct_option in explicitly_available_adjunct_options
                and slot.adjunct_option not in candidate_adjunct_option_codes(candidate)
            ):
                continue
            if not _candidate_matches_slot(
                candidate,
                slot,
                experience_level=blueprint.experience_level,
                goal=blueprint.goal,
                advanced_chest_focus=blueprint.advanced_chest_focus,
                specialization_focus=effective_focus,
                training_method=blueprint.training_method,
                specialization_slot_id=blueprint.specialization_slot_id,
                training_method_slot_id=blueprint.training_method_slot_id,
            ):
                if (
                    slot.phase == "main_compound"
                    and _candidate_movement(candidate) == "COMPOUND"
                    and _candidate_primary(candidate) in set(slot.target_muscles)
                    and _candidate_movement(candidate) in set(slot.preferred_movement_types)
                ):
                    experience_role_rejected_count += 1
                continue
            family_key = focus_variant_family_key(
                candidate, effective_focus, candidate_selection_family_key(candidate)
            )
            choice_family_key = family_key
            if (
                slot.slot_id == "full_push_main"
                and blueprint.experience_level == "intermediate"
                and blueprint.goal == "hypertrophy"
                and blueprint.session_structure == "standard"
                and blueprint.readiness_level in {"normal", "high"}
                and not effective_focus and not blueprint.training_method
                and not blueprint.requested_adjunct_options
                and family_key == "horizontal_press:flat"
                and candidate.get("equipment_req") in {"BARBELL", "DUMBBELL"}
            ):
                # These are choices for one slot, not permission to select both.
                choice_family_key = f"{family_key}:{candidate['equipment_req']}"
            elif (
                slot.slot_id == "target_main"
                and set(blueprint.target_muscles) == {"gluteal"}
                and _candidate_primary(candidate) == "gluteal"
                and _candidate_movement(candidate) == "COMPOUND"
                and main_role_eligibility(
                    candidate, blueprint.experience_level, blueprint.goal
                ).eligible
                and blueprint.goal == "hypertrophy"
                and blueprint.session_structure == "standard"
                and blueprint.readiness_level in {"normal", "high"}
                and not effective_focus
                and not blueprint.requested_adjunct_options
                and str(candidate.get("catalog_exercise_id") or candidate.get("id"))
                in {"V24ROW-023880", "V24ROW-023882"}
                and candidate.get("equipment_req") in {"BODYWEIGHT", "DUMBBELL"}
            ):
                # Restore admitted equipment choices; final family dedup stays unchanged.
                choice_family_key = f"{family_key}:{candidate['equipment_req']}"
            display_key = re.sub(
                r"[\s/_()·,.-]+",
                "",
                candidate_display_name(candidate).lower(),
            )
            if choice_family_key in matched_family_keys or display_key in matched_display_keys:
                duplicate_family_suppressed_count += 1
                continue
            matched.append(
                {
                    "exerciseId": exercise_id,
                    "exerciseName": _candidate_name(candidate),
                    "score": int(candidate.get("score") or 0),
                    "primaryMuscle": _candidate_primary(candidate),
                    "movementType": _candidate_movement(candidate),
                }
            )
            matched_family_keys.add(choice_family_key)
            matched_display_keys.add(display_key)
            used_ids.add(normalized_id)
            if focus_slot and focus_target_muscles.intersection(slot.target_muscles):
                claimed_focus_families.add(family_key)
            if len(matched) >= slot_candidate_limit:
                break
        bundles.append(
            {
                "slotId": slot.slot_id,
                "phase": slot.phase,
                "requirement": slot.requirement,
                "purpose": slot.purpose,
                "adjunctOption": slot.adjunct_option,
                "candidates": matched,
                "slotUnfilled": not matched,
                "duplicateFamilySuppressedCount": duplicate_family_suppressed_count,
                "experienceRoleRejectedCount": experience_role_rejected_count,
                "specializationFocusUnavailable": focus_unavailable,
            }
        )

    # Flex is server-owned. It may reuse unselected explicit-slot alternatives;
    # the adapter removes selected IDs and duplicate exercise families at fill time.
    exclusive_focus_families: set[str] = set()
    if effective_focus == "forearmEmphasis" and focus_slot:
        focus_ids = {
            item["exerciseId"]
            for bundle in bundles
            if bundle["slotId"] == focus_slot.slot_id
            for item in bundle["candidates"]
        }
        focus_choices = [
            candidate for candidate in ranked_candidates
            if _candidate_id(candidate) in focus_ids
        ]
        if any(candidate.get("core_forearm_pattern_reviewed") for candidate in focus_choices):
            # Reviewed rotation replaces this role; neither branch may become
            # a third direct-forearm exercise through minimum flex fill.
            exclusive_focus_families = {
                focus_variant_family_key(
                    candidate, effective_focus, candidate_selection_family_key(candidate)
                )
                for candidate in focus_choices
            }
    flex_limit = len(ranked_candidates) if blueprint.flex_slot_count else 0
    if flex_limit:
        flex_candidates: list[dict] = []
        flex_family_keys: set[str] = set()
        flex_display_keys: set[str] = set()
        flex_duplicate_family_suppressed_count = 0
        allowed_flex_targets = set(blueprint.target_muscles)
        for candidate in ranked_candidates:
            exercise_id = _candidate_id(candidate)
            if (
                not exercise_id
                or not automatic_recommendation_eligibility(candidate, blueprint.experience_level).eligible
                or str(candidate.get("main_routine_default_allowed", "")).lower() in {"0", "false"}
                or exercise_id.lower() in reserved_adjunct_ids
                or _candidate_primary(candidate) not in allowed_flex_targets
                or (
                    effective_focus == "forearmEmphasis"
                    and candidate.get("core_forearm_pattern_reviewed")
                )
            ):
                continue
            family_key = focus_variant_family_key(
                candidate, effective_focus, candidate_selection_family_key(candidate)
            )
            if family_key in exclusive_focus_families:
                continue
            display_key = re.sub(
                r"[\s/_()·,.-]+",
                "",
                candidate_display_name(candidate).lower(),
            )
            if family_key in flex_family_keys or display_key in flex_display_keys:
                flex_duplicate_family_suppressed_count += 1
                continue
            flex_candidates.append(
                {
                    "exerciseId": exercise_id,
                    "exerciseName": _candidate_name(candidate),
                    "score": int(candidate.get("score") or 0),
                    "primaryMuscle": _candidate_primary(candidate),
                    "movementType": _candidate_movement(candidate),
                }
            )
            flex_family_keys.add(family_key)
            flex_display_keys.add(display_key)
            if len(flex_candidates) >= flex_limit:
                break
        bundles.append(
            {
                "slotId": "flex",
                "phase": "after_required_slots",
                "requirement": "optional",
                "purpose": "필수 슬롯 이후 남은 안전 후보",
                "candidates": flex_candidates,
                "slotUnfilled": not flex_candidates,
                "duplicateFamilySuppressedCount": flex_duplicate_family_suppressed_count,
            }
        )
    return bundles


def render_candidate_bundles_for_prompt(bundles: Sequence[dict]) -> str:
    return json.dumps(list(bundles), ensure_ascii=False, separators=(",", ":"))

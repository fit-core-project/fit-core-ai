"""Internal pre-generation injury warning and volume-choice facade.

The existing routine API remains unchanged. Callers that opt into this facade
must resolve an injury warning before generation; a choice never weakens the
request's readiness, DOMS, pain-area, or managed-injury constraints.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Callable, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from .db_queries import get_user_constraint_profile_from_sqlite
from .injury_condition_policy import resolve_injury_condition_contexts
from .routine_pipeline import (
    _normalize_execution_groups_after_block_removal,
    generate_smart_routine,
)
from .prescription.estimator import estimate_routine_time_min
from .prescription.unilateral import restore_unilateral_from_block
from .schemas import (
    LLMExercisePlan,
    RecentSetRecord,
    RoutineDraftResponse,
    RoutineRequest,
    UserProfileContext,
)


CHOICE_SCHEMA_VERSION = "routine-generation-choice-v1"
CHOICE_POLICY_VERSION = "injury-volume-choice-2026-09-11"
RoutineGenerationChoiceCode = Literal[
    "reducedVolume",
    "normalVolume",
    "requestReevaluation",
    "cancel",
]
RoutineGenerationChoiceStatus = Literal[
    "ready",
    "choiceRequired",
    "generated",
    "generationFailed",
    "reevaluationRequired",
    "canceled",
]
WorkingSetReductionStatus = Literal[
    "notRequested",
    "applied",
    "floorLimited",
    "noFurtherReduction",
    "noWorkingSets",
]

_AVAILABLE_CHOICES: tuple[RoutineGenerationChoiceCode, ...] = (
    "reducedVolume",
    "normalVolume",
    "requestReevaluation",
    "cancel",
)
_PREFLIGHT_WARNING = (
    "허용된 운동 범위가 불명확하거나 현재 부상 부위에 통증이 있다면 해당 부위 운동을 보류하고 "
    "의료 전문가에게 먼저 확인하세요. 진행 선택은 의료진 허가를 대신하지 않습니다. "
    "등록된 부상 및 관리 제한을 적용한 뒤에만 루틴을 생성합니다. "
    "볼륨 선택은 제외된 동작을 다시 허용하지 않습니다."
)
_REDUCED_VOLUME_WARNING = (
    "선택에 따라 기존 안전 제한을 유지한 채 작업 세트만 추가로 줄였습니다."
)
_REDUCED_VOLUME_RATIONALE = (
    "선택한 감소 볼륨을 반영해 안전 필터를 통과한 운동의 작업 세트를 조정했습니다."
)
_NO_FURTHER_REDUCTION_WARNING = (
    "선택된 각 작업 운동의 최소 1세트를 유지해 추가 세트 감소를 적용하지 않았습니다."
)
_FLOOR_LIMITED_REDUCTION_WARNING = (
    "선택된 각 작업 운동의 최소 1세트를 유지해 목표한 세트 감소를 모두 적용하지 못했습니다."
)
_REEVALUATION_MESSAGE = (
    "부상 상태와 확인된 금지 동작을 다시 확인해 주세요. "
    "동의만으로 제외 동작을 허용하지 않습니다."
)


class RoutineGenerationChoiceError(ValueError):
    """Raised when a selection does not belong to the current preflight."""


class RoutineGenerationSelection(BaseModel):
    """Closed selection bound to one exact request and policy revision."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )

    schema_version: Literal["routine-generation-choice-v1"] = CHOICE_SCHEMA_VERSION
    policy_version: Literal["injury-volume-choice-2026-09-11"] = CHOICE_POLICY_VERSION
    request_fingerprint: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    choice: RoutineGenerationChoiceCode


class EffectiveInjuryCondition(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    area_code: str
    status: str
    restricted_movement_patterns: list[str] = Field(default_factory=list)


class RoutineGenerationConstraintSummary(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    pain_areas: list[str] = Field(default_factory=list)
    injury_conditions: list[EffectiveInjuryCondition] = Field(default_factory=list)
    condition_policies: list[dict[str, Any]] = Field(default_factory=list)
    active_uncleared_count: int = 0
    managed_restriction_count: int = 0
    explicit_prohibited_pattern_count: int = 0
    condition_policy_count: int = 0


class RoutineGenerationChoiceMetrics(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    generation_invocation_count: int = 0
    requested_time_min: int
    estimated_time_min: Optional[int] = None
    estimated_time_before_choice_min: Optional[int] = None
    reported_estimated_time_before_choice_min: Optional[int] = None
    generated_exercise_count_before_choice: int = 0
    generated_exercise_count_after_choice: int = 0
    working_set_count_before_choice: int = 0
    working_set_count_after_choice: int = 0
    optional_working_set_reduction_target: int = 0
    optional_working_sets_removed: int = 0
    optional_working_set_reduction_status: WorkingSetReductionStatus = "notRequested"


class RoutineGenerationChoiceResult(BaseModel):
    """Internal result; intentionally separate from the stable public response."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    status: RoutineGenerationChoiceStatus
    choice_required: bool
    request_fingerprint: str
    policy_version: str = CHOICE_POLICY_VERSION
    schema_version: str = CHOICE_SCHEMA_VERSION
    available_choices: list[RoutineGenerationChoiceCode] = Field(default_factory=list)
    warning: Optional[str] = None
    selected_choice: Optional[RoutineGenerationChoiceCode] = None
    reevaluation_message: Optional[str] = None
    constraints: RoutineGenerationConstraintSummary
    metrics: RoutineGenerationChoiceMetrics
    routine: Optional[RoutineDraftResponse] = None


def _model_payload(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True, exclude_none=False)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_model_payload(item) for item in value]
    if value is None:
        return None
    raise TypeError(f"unsupported choice-binding input: {type(value).__name__}")


def _request_fingerprint(
    request: RoutineRequest,
    profile: Optional[UserProfileContext],
    recent_sets: Optional[Sequence[RecentSetRecord]],
) -> str:
    payload = {
        "policyVersion": CHOICE_POLICY_VERSION,
        "schemaVersion": CHOICE_SCHEMA_VERSION,
        "request": _model_payload(request),
        "profile": _model_payload(profile),
        "recentSets": _model_payload(list(recent_sets or ())),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _constraint_summary(
    request: RoutineRequest,
    profile: Optional[UserProfileContext],
) -> RoutineGenerationConstraintSummary:
    pain_keys = {
        (entry.body_part, entry.side)
        for entry in (
            *(request.pain_areas or ()),
            *((profile.pain_areas or ()) if profile else ()),
        )
    }
    pain_areas = sorted(
        f"{body_part}:{side}" if side else body_part
        for body_part, side in pain_keys
    )
    conditions = resolve_injury_condition_contexts(
        request.injury_conditions,
        profile.injury_conditions if profile else (),
        legacy_shoulder_contexts=(
            (request.shoulder_condition_status, request.restricted_movement_patterns),
            (
                profile.shoulder_condition_status if profile else None,
                profile.restricted_movement_patterns if profile else (),
            ),
        ),
    )
    effective = [
        EffectiveInjuryCondition(
            area_code=condition.area_code,
            status=condition.status,
            restricted_movement_patterns=list(condition.restricted_movement_patterns),
        )
        for condition in conditions
        if condition.status != "recovered"
    ]
    return RoutineGenerationConstraintSummary(
        pain_areas=pain_areas,
        injury_conditions=effective,
        condition_policies=[dict(item) for item in request.condition_policies],
        active_uncleared_count=sum(
            condition.status == "activeUncleared" for condition in effective
        ),
        managed_restriction_count=sum(
            condition.status == "managedWithRestrictions" for condition in effective
        ),
        explicit_prohibited_pattern_count=sum(
            len(condition.restricted_movement_patterns) for condition in effective
        ),
        condition_policy_count=len(request.condition_policies),
    )


def _resolve_persisted_generation_context(
    request: RoutineRequest,
    local_constraint_profile: Optional[Mapping[str, Any]] = None,
) -> tuple[RoutineRequest, dict[str, Any]]:
    """Bind the choice flow to the same local health policy input as generation."""

    local_constraint_profile = dict(
        local_constraint_profile
        if local_constraint_profile is not None
        else get_user_constraint_profile_from_sqlite(request.user_id)
    )
    persisted_policies = list(
        local_constraint_profile.get("condition_policies", []) or []
    )
    mobility_limits: list[str] = []
    seen_mobility_limits: set[str] = set()
    for value in (
        *(request.mobility_limits or []),
        *(local_constraint_profile.get("mobility_limits", []) or []),
    ):
        text = str(value).strip()
        if text and text.lower() not in seen_mobility_limits:
            seen_mobility_limits.add(text.lower())
            mobility_limits.append(text)
    return request.model_copy(deep=True, update={
        "condition_policies": [
            *(request.condition_policies or []),
            *persisted_policies,
        ],
        "mobility_limits": mobility_limits,
        "anthropometry_signals": {
            **(local_constraint_profile.get("anthropometry_signals", {}) or {}),
            **(request.anthropometry_signals or {}),
        },
    }), local_constraint_profile


def _metrics(
    request: RoutineRequest,
    *,
    invocation_count: int = 0,
    routine: Optional[RoutineDraftResponse] = None,
    before_exercise_count: Optional[int] = None,
    before_working_sets: Optional[int] = None,
    before_estimated_time: Optional[int] = None,
    reported_before_estimated_time: Optional[int] = None,
    reduction_target: int = 0,
    removed_working_sets: int = 0,
    reduction_status: WorkingSetReductionStatus = "notRequested",
) -> RoutineGenerationChoiceMetrics:
    blocks = list(routine.routine_blocks) if routine else []
    after_working_sets = sum(
        prescription.set_type == "working"
        for block in blocks
        for prescription in block.prescription
    )
    return RoutineGenerationChoiceMetrics(
        generation_invocation_count=invocation_count,
        requested_time_min=request.time_available_min,
        estimated_time_min=routine.total_estimated_time if routine else None,
        estimated_time_before_choice_min=(
            routine.total_estimated_time
            if routine and before_estimated_time is None
            else before_estimated_time
        ),
        reported_estimated_time_before_choice_min=(
            routine.total_estimated_time
            if routine and reported_before_estimated_time is None
            else reported_before_estimated_time
        ),
        generated_exercise_count_before_choice=(
            len(blocks) if before_exercise_count is None else before_exercise_count
        ),
        generated_exercise_count_after_choice=len(blocks),
        working_set_count_before_choice=(
            after_working_sets if before_working_sets is None else before_working_sets
        ),
        working_set_count_after_choice=after_working_sets,
        optional_working_set_reduction_target=reduction_target,
        optional_working_sets_removed=removed_working_sets,
        optional_working_set_reduction_status=reduction_status,
    )


def preflight_routine_generation(
    request: RoutineRequest,
    *,
    profile: Optional[UserProfileContext] = None,
    recent_sets: Optional[Sequence[RecentSetRecord]] = None,
    local_constraint_profile: Optional[Mapping[str, Any]] = None,
    resolve_persisted_context: bool = False,
) -> RoutineGenerationChoiceResult:
    """Inspect inputs, optionally resolving the local persisted health snapshot."""

    if local_constraint_profile is not None or resolve_persisted_context:
        request, _ = _resolve_persisted_generation_context(
            request,
            local_constraint_profile,
        )

    fingerprint = _request_fingerprint(request, profile, recent_sets)
    constraints = _constraint_summary(request, profile)
    warning_required = bool(
        constraints.pain_areas
        or constraints.injury_conditions
        or constraints.condition_policies
    )
    return RoutineGenerationChoiceResult(
        status="choiceRequired" if warning_required else "ready",
        choice_required=warning_required,
        request_fingerprint=fingerprint,
        available_choices=list(_AVAILABLE_CHOICES) if warning_required else [],
        warning=_PREFLIGHT_WARNING if warning_required else None,
        constraints=constraints,
        metrics=_metrics(request),
    )


def make_routine_generation_selection(
    preflight: RoutineGenerationChoiceResult,
    choice: RoutineGenerationChoiceCode,
) -> RoutineGenerationSelection:
    if not preflight.choice_required:
        raise RoutineGenerationChoiceError("this preflight does not require a choice")
    return RoutineGenerationSelection(
        request_fingerprint=preflight.request_fingerprint,
        choice=choice,
    )


def _working_set_count(routine: RoutineDraftResponse) -> int:
    return sum(
        prescription.set_type == "working"
        for block in routine.routine_blocks
        for prescription in block.prescription
    )


def _estimate_choice_routine_blocks_time(blocks: Sequence[Any]) -> int:
    """Adapt final blocks to the canonical estimator without losing metadata."""

    plans: list[LLMExercisePlan] = []
    execution_metadata: dict[str, dict[str, Any]] = {}
    for block in blocks:
        if not block.prescription:
            continue
        first_set = block.prescription[0]
        plan = LLMExercisePlan(
            exercise_id=block.exercise_id,
            exercise_name=block.exercise_name,
            movement_pattern=block.movement_pattern,
            movement_type=block.movement_type,
            primary_muscles=list(block.primary_muscles),
            equipment_type=block.equipment_type,
            target_weight_kg=first_set.target_weight_kg,
            target_reps=first_set.target_reps,
            sets=len(block.prescription),
            rest_time_sec=block.default_rest_sec,
            target_rir=first_set.target_rir,
            exercise_rationale=block.exercise_rationale,
            substitution_candidates=[],
        )
        plan._prescription_set_type = first_set.set_type
        plan._target_duration_sec = first_set.target_duration_sec
        plan._interval_work_sec = first_set.interval_work_sec
        plan._interval_recovery_sec = first_set.interval_recovery_sec
        plan._interval_rounds = first_set.interval_rounds
        plan._target_distance_m = first_set.target_distance_m
        if block.training_method is not None:
            plan._training_method_code = block.training_method.code
            plan._training_method_instruction = block.training_method.model_dump(
                mode="json",
                by_alias=True,
                exclude_none=True,
            )
        restore_unilateral_from_block(plan, block)
        plans.append(plan)
        exercise_id = str(block.exercise_id or "").strip().lower()
        execution_metadata[exercise_id] = {
            "executionGroupId": block.execution_group_id,
            "executionGroupType": block.execution_group_type,
            "executionGroupPosition": block.execution_group_position,
            "executionGroupSize": block.execution_group_size,
            "transitionRestSec": block.transition_rest_sec,
            "groupRestSec": block.group_rest_sec,
        }
    if not plans:
        return 0
    return estimate_routine_time_min(plans, execution_metadata=execution_metadata)


_PRESCRIPTION_QUANTITY = re.compile(
    r"(?i)(?<![A-Za-z0-9])\d+(?:\.\d+)?\s*(?:세트|분|sets?|minutes?|mins?)(?![A-Za-z])"
)


def _is_old_plan_total_claim(text: str, *, working_sets: int, time_min: int) -> bool:
    value = str(text or "")
    volume_claim = bool(
        re.search(rf"(?<!\d){working_sets}\s*(?:세트|sets?)(?![A-Za-z])", value, re.I)
        and any(token in value.lower() for token in ("총", "전체", "구성", "볼륨", "total"))
    )
    time_claim = bool(
        re.search(rf"(?<!\d){time_min}\s*(?:분|minutes?|mins?)(?![A-Za-z])", value, re.I)
        and any(token in value.lower() for token in ("총", "예상", "소요", "구성", "total", "estimated"))
    )
    return volume_claim or time_claim


def _annotate_reduced_prescription(
    routine: RoutineDraftResponse,
    *,
    before_working_sets: int,
    after_working_sets: int,
    before_time_min: int,
    reported_before_time_min: int,
    after_time_min: int,
    requested_time_min: int,
) -> None:
    selection_reasons = list(routine.rationale_summary)
    routine.rationale_summary = [
        item
        for item in selection_reasons
        if not _is_old_plan_total_claim(
            item,
            working_sets=before_working_sets,
            time_min=reported_before_time_min,
        )
    ]
    routine.rationale_summary.extend([
        _REDUCED_VOLUME_RATIONALE,
        (
            f"현재 처방은 작업 세트 {after_working_sets}세트, 예상 {after_time_min}분입니다. "
            f"요청 시간은 {requested_time_min}분이며, 조정 전 계획은 작업 세트 "
            f"{before_working_sets}세트, 예상 {before_time_min}분이었습니다."
        ),
    ])
    routine.summary_title = (
        f"감소 볼륨 루틴: 작업 세트 {after_working_sets}세트, 예상 {after_time_min}분"
    )
    routine.warnings = [
        (
            f"요청 {requested_time_min}분 기준 현재 처방은 작업 세트 "
            f"{after_working_sets}세트, 예상 {after_time_min}분입니다. "
            f"조정 전 안내: {warning}"
            if _is_old_plan_total_claim(
                warning,
                working_sets=before_working_sets,
                time_min=reported_before_time_min,
            )
            else warning
        )
        for warning in routine.warnings
    ]
    for block in routine.routine_blocks:
        working_sets = sum(
            prescription.set_type == "working" for prescription in block.prescription
        )
        if working_sets:
            selection_reason = str(block.exercise_rationale or "")
            # A matching number may describe history or a limit, not this plan.
            if _PRESCRIPTION_QUANTITY.search(selection_reason):
                block.exercise_rationale = (
                    f"현재 작업 세트는 {working_sets}세트입니다. "
                    f"감량 전 운동 선택 설명: {selection_reason}"
                )
            else:
                block.exercise_rationale = (
                    f"{selection_reason} 현재 작업 세트 {working_sets}세트."
                )
    routine._underfill_diagnostics = {}


def _reduce_working_volume(
    routine: RoutineDraftResponse,
    *,
    requested_time_min: int,
) -> tuple[
    RoutineDraftResponse,
    int,
    int,
    WorkingSetReductionStatus,
    int,
]:
    """Remove about 25% of working sets without changing adjunct prescriptions."""

    reduced = routine.model_copy(deep=True)
    reported_before_time_min = reduced.total_estimated_time
    original_blocks = list(reduced.routine_blocks)
    working_sets = _working_set_count(reduced)
    before_time_min = _estimate_choice_routine_blocks_time(reduced.routine_blocks)
    if working_sets == 0:
        return reduced, 0, 0, "noWorkingSets", before_time_min

    target_removals = min(working_sets - 1, max(1, math.ceil(working_sets * 0.25)))
    if target_removals == 0:
        if _NO_FURTHER_REDUCTION_WARNING not in reduced.warnings:
            reduced.warnings.append(_NO_FURTHER_REDUCTION_WARNING)
        return reduced, 0, 0, "noFurtherReduction", before_time_min
    removed = 0
    while removed < target_removals:
        changed = False
        for block in reversed(reduced.routine_blocks):
            working_indexes = [
                index
                for index, prescription in enumerate(block.prescription)
                if prescription.set_type == "working"
            ]
            if len(working_indexes) <= 1:
                continue
            del block.prescription[working_indexes[-1]]
            removed += 1
            changed = True
            if removed >= target_removals:
                break
        if not changed:
            break

    _normalize_execution_groups_after_block_removal(
        original_blocks,
        reduced.routine_blocks,
    )
    for order, block in enumerate(reduced.routine_blocks, start=1):
        block.order = order
        for set_index, prescription in enumerate(block.prescription, start=1):
            prescription.set_index = set_index
    reduced.total_estimated_time = _estimate_choice_routine_blocks_time(
        reduced.routine_blocks
    )
    if removed:
        if _REDUCED_VOLUME_WARNING not in reduced.warnings:
            reduced.warnings.append(_REDUCED_VOLUME_WARNING)
        _annotate_reduced_prescription(
            reduced,
            before_working_sets=working_sets,
            after_working_sets=_working_set_count(reduced),
            before_time_min=before_time_min,
            reported_before_time_min=reported_before_time_min,
            after_time_min=reduced.total_estimated_time,
            requested_time_min=requested_time_min,
        )
    status: WorkingSetReductionStatus = (
        "noFurtherReduction"
        if removed == 0
        else "applied"
        if removed == target_removals
        else "floorLimited"
    )
    if status == "floorLimited":
        reduced.warnings.append(
            f"목표 {target_removals}세트 중 {removed}세트만 줄였습니다. "
            f"{_FLOOR_LIMITED_REDUCTION_WARNING}"
        )
    elif (
        status == "noFurtherReduction"
        and _NO_FURTHER_REDUCTION_WARNING not in reduced.warnings
    ):
        reduced.warnings.append(_NO_FURTHER_REDUCTION_WARNING)
    return reduced, removed, target_removals, status, before_time_min


def _parse_selection(
    selection: RoutineGenerationSelection | Mapping[str, Any],
) -> RoutineGenerationSelection:
    if isinstance(selection, RoutineGenerationSelection):
        selection = selection.model_dump(mode="json", by_alias=True)
    return RoutineGenerationSelection.model_validate(selection)


def generate_routine_with_choice(
    request: RoutineRequest,
    db: Any,
    *,
    profile: Optional[UserProfileContext] = None,
    recent_sets: Optional[Sequence[RecentSetRecord]] = None,
    selection: Optional[RoutineGenerationSelection | Mapping[str, Any]] = None,
    _generator: Optional[Callable[..., RoutineDraftResponse]] = None,
) -> RoutineGenerationChoiceResult:
    """Run preflight, enforce a bound choice, then call the existing generator."""

    resolved_request, local_constraint_profile = _resolve_persisted_generation_context(
        request
    )
    preflight = preflight_routine_generation(
        resolved_request,
        profile=profile,
        recent_sets=recent_sets,
    )
    parsed_selection = _parse_selection(selection) if selection is not None else None

    if preflight.choice_required:
        if parsed_selection is None:
            return preflight
        if parsed_selection.request_fingerprint != preflight.request_fingerprint:
            raise RoutineGenerationChoiceError(
                "selection does not match the current request, profile, or recent-set context"
            )
        if parsed_selection.choice == "cancel":
            return preflight.model_copy(update={
                "status": "canceled",
                "choice_required": False,
                "selected_choice": "cancel",
                "available_choices": [],
            })
        if parsed_selection.choice == "requestReevaluation":
            return preflight.model_copy(update={
                "status": "reevaluationRequired",
                "choice_required": False,
                "selected_choice": "requestReevaluation",
                "available_choices": [],
                "reevaluation_message": _REEVALUATION_MESSAGE,
            })
    elif parsed_selection is not None:
        raise RoutineGenerationChoiceError("selection supplied when no warning choice is required")

    selected_choice: Optional[RoutineGenerationChoiceCode] = (
        parsed_selection.choice if parsed_selection else None
    )
    request_copy = resolved_request.model_copy(deep=True)
    profile_copy = profile.model_copy(deep=True) if profile else None
    recent_sets_copy = [item.model_copy(deep=True) for item in (recent_sets or ())]
    if _generator is None:
        routine = generate_smart_routine(
            request_copy,
            db,
            profile=profile_copy,
            recent_sets=recent_sets_copy,
            local_constraint_profile=local_constraint_profile,
            local_constraint_profile_is_resolved=True,
        )
    else:
        routine = _generator(
            request_copy,
            db,
            profile=profile_copy,
            recent_sets=recent_sets_copy,
        )
    before_exercise_count = len(routine.routine_blocks)
    before_working_sets = _working_set_count(routine)
    before_estimated_time = routine.total_estimated_time
    reported_before_estimated_time = routine.total_estimated_time
    reduction_target = 0
    removed_working_sets = 0
    reduction_status: WorkingSetReductionStatus = "notRequested"
    if selected_choice == "reducedVolume":
        (
            routine,
            removed_working_sets,
            reduction_target,
            reduction_status,
            before_estimated_time,
        ) = _reduce_working_volume(
            routine,
            requested_time_min=request.time_available_min,
        )

    return RoutineGenerationChoiceResult(
        status=(
            "generationFailed"
            if routine.generation_status == "failed" or not routine.routine_blocks
            else "generated"
        ),
        choice_required=False,
        request_fingerprint=preflight.request_fingerprint,
        selected_choice=selected_choice,
        warning=preflight.warning,
        constraints=preflight.constraints,
        metrics=_metrics(
            request,
            invocation_count=1,
            routine=routine,
            before_exercise_count=before_exercise_count,
            before_working_sets=before_working_sets,
            before_estimated_time=before_estimated_time,
            reported_before_estimated_time=reported_before_estimated_time,
            reduction_target=reduction_target,
            removed_working_sets=removed_working_sets,
            reduction_status=reduction_status,
        ),
        routine=routine,
    )

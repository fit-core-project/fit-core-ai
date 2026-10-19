"""Opt-in draft transport; legacy RoutineDraftResponse remains unchanged."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .schemas import RoutineDraftResponse, RoutineRequest, UserProfileContext, to_camel

CONTRACT_VERSION = 'routine-core-v4-draft-v3'
_EXPANSIONS = (
    ('addQuadriceps', 'quadriceps', '앞 허벅지를 추가해 다시 생성'),
    ('addHamstring', 'hamstring', '뒤 허벅지를 추가해 다시 생성'),
    ('addCalves', 'calves', '종아리를 추가해 다시 생성'),
)


class TargetExpansionOption(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra='forbid')
    code: Literal['addQuadriceps', 'addHamstring', 'addCalves']
    label_ko: str
    proposed_target_muscles: list[str]
    requires_confirmation: Literal[True] = True
    requires_regeneration: Literal[True] = True
    availability: Literal['notEvaluated'] = 'notEvaluated'


class ShortRoutineNotice(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra='forbid')
    is_short_routine: bool
    requested_time_min: int = Field(ge=10, le=180)
    planned_time_min: int = Field(ge=0)
    threshold_ratio: Literal[0.5] = 0.5
    message_ko: str | None = None
    target_expansion_options: list[TargetExpansionOption] = Field(default_factory=list)


class RoutineResponseV4(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra='forbid')
    contract_version: Literal['routine-core-v4-draft-v3'] = CONTRACT_VERSION
    routine: RoutineDraftResponse
    short_routine_notice: ShortRoutineNotice


def _can_offer_glute_expansion(req, profile, local_constraint_profile):
    # Unknown local context is not evidence of unrestricted movement. Options
    # are new target requests, never a clearance or a promise of available rows.
    if local_constraint_profile is None or any(local_constraint_profile.values()):
        return False
    if profile and (profile.pain_areas or profile.injury_conditions
                    or profile.shoulder_condition_status or profile.restricted_movement_patterns):
        return False
    return (
        req.target_split_label == 'custom' and req.target_muscles == ['gluteal']
        and req.goal == 'hypertrophy' and req.training_style == 'gym'
        and req.session_structure == 'standard' and req.readiness_level in {'normal', 'high'}
        and not any((req.pain_areas, req.doms_data, req.injury_conditions,
                     req.shoulder_condition_status, req.restricted_movement_patterns,
                     req.condition_policies, req.mobility_limits, req.profile_signals,
                     req.anthropometry_signals, req.specialization_focus, req.adjunct_options,
                     req.user_note))
    )


def build_v4_response(req: RoutineRequest, routine: RoutineDraftResponse, *,
                      profile: UserProfileContext | None = None,
                      local_constraint_profile: dict | None = None) -> RoutineResponseV4:
    actual = routine.total_estimated_time
    has_plan = routine.generation_status in {'success', 'fallback'} and bool(routine.routine_blocks)
    if has_plan and actual <= 0:
        raise ValueError('A nonempty routine requires a positive estimated duration')
    short = has_plan and 2 * actual < req.time_available_min
    notice = ShortRoutineNotice(is_short_routine=short, requested_time_min=req.time_available_min,
                                planned_time_min=actual)
    if short:
        notice.message_ko = (
            f'요청하신 {req.time_available_min}분보다 짧은 예상 {actual}분 루틴입니다. '
            '시간을 채우기 위해 운동 부위나 세트 수를 자동으로 늘리지 않았습니다. '
            '현재 루틴을 유지하거나 목표 부위를 직접 바꿔 다시 요청할 수 있습니다. '
            '재생성에서도 등록된 부상·장비 등의 제한은 유지됩니다.'
        )
        if _can_offer_glute_expansion(req, profile, local_constraint_profile):
            notice.target_expansion_options = [TargetExpansionOption(
                code=code, label_ko=label, proposed_target_muscles=['gluteal', muscle],
            ) for code, muscle, label in _EXPANSIONS]
            notice.message_ko += ' 추가 부위의 운동 가능 여부와 분량은 기존 제한을 유지한 재생성에서 확인합니다.'
    return RoutineResponseV4(routine=routine.model_copy(deep=True), short_routine_notice=notice)


def build_confirmed_expansion_request(req: RoutineRequest, routine: RoutineDraftResponse, *,
        option_code: str, confirmed: bool, profile: UserProfileContext | None = None,
        local_constraint_profile: dict | None = None) -> RoutineRequest:
    """Use server-held current request/response, not a client-supplied envelope."""
    if confirmed is not True:
        raise ValueError('Explicit confirmation required')
    envelope = build_v4_response(req, routine, profile=profile,
                                 local_constraint_profile=local_constraint_profile)
    option = next((o for o in envelope.short_routine_notice.target_expansion_options
                   if o.code == option_code), None)
    if option is None:
        raise ValueError('Target option is not available for the current request')
    data = req.model_dump()
    data.update(target_split_label='custom', target_muscles=list(option.proposed_target_muscles))
    return RoutineRequest.model_validate(data)


def generate_smart_routine_v4(req, db, profile=None, recent_sets=None,
        local_constraint_profile=None, local_constraint_profile_is_resolved=False):
    from . import routine_pipeline

    verified_context = local_constraint_profile
    if local_constraint_profile is None:
        local_constraint_profile = routine_pipeline.get_user_constraint_profile_from_sqlite(req.user_id)
    routine = routine_pipeline.generate_smart_routine(
        req, db, profile, recent_sets, local_constraint_profile, local_constraint_profile_is_resolved,
    )
    # Legacy lookup collapses DB errors to empty context. Do not interpret that
    # best-effort fallback as verified absence of restrictions for new advice.
    return build_v4_response(req, routine, profile=profile,
                              local_constraint_profile=verified_context)

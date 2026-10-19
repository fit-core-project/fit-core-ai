"""
Routine generation pipeline orchestrator.

This module currently exceeds the 100-line threshold, but the size is
intentional for now: it contains orchestration only, while domain logic is
delegated to focused modules.

Structure:
- generate_smart_routine: end-to-end orchestration (public).
- _fallback: delegates fallback routine response generation.
- _finalize_success: converts a successful LLM output into a draft response.
- _deterministic_or_fallback: applies deterministic prescription and time checks.
- _validate_or_fallback: validates/repairs LLM output before deterministic handling.

Domain logic is delegated to candidate_ranker, prompt_builder, prescription.*,
llm_parser, fallback, muscle_mapping, and llm_router.

Revisit trigger: if the LLM interaction block (around line 210-249) grows more
complex or takes on a separate responsibility, consider splitting it into
pipeline_llm.py.
"""
import asyncio
import logging
import os
import re
import time
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Sequence

import httpx
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError
from sqlalchemy.orm import Session

from .candidate_pool_policy import get_candidate_pool_size
from .candidate_ranker import (
    _candidate_doms_overlap,
    _candidate_doms_prescription_signal,
    _candidate_injury_overlap,
    candidate_selection_family_key,
    candidate_selection_pattern_key,
    format_candidates_for_prompt,
    score_candidate_exercises,
)
from .exercise_display import candidate_display_name
from .equipment_policy import parse_equipment_options
from .db_queries import (
    get_adjunct_candidate_exercises,
    get_candidate_exercises,
    get_user_constraint_profile_from_sqlite,
)
from .fallback import (
    _debug_print_draft,
    _debug_print_llm_output,
    _debug_print_prompt,
    build_routine_draft,
    generate_fallback_routine,
)
from .llm_router import LocalJsonChatModel, StatusReasonCode, classify_llm_error, get_llm, map_llm_error, resolve_llm_provider
from .routine_selection_schema import build_selection_decoding_schema
from .llm_parser import (
    SchemaRepairError,
    classify_validation_error,
    normalize_llm_response,
    extract_raw_llm_output_from_exception,
    parse_json_candidate,
    summarize_validation_error_shapes,
    trim_routine_to_time_budget,
)
from .korean_text_guard import _replace_schema_tokens, separate_doms_from_pain_wording_on_response
from .injury_condition_policy import (
    active_uncleared_neck_requires_safe_stop,
    format_injury_conditions_for_prompt,
    injury_area_label_ko,
    injury_condition_labels_ko,
    resolve_injury_condition_contexts,
)
from .log_redaction import sanitize_exception_for_log, summarize_text_for_log
from .muscle_mapping import (
    resolve_request_target_muscles,
    resolve_request_target_split_label,
)
from .prescription.adjustments import (
    _calculate_max_total_sets,
    _max_sets_for_exercise,
    enforce_total_set_cap,
    minimum_productive_duration_min,
    target_exercise_count_for_session,
)
from .prescription.estimator import estimate_routine_time_min
from .prescription.unilateral import restore_unilateral_from_block
from .prescription.experience import experience_set_cap
from .prescription.focus_capacity import complete_lower_chest_focus_capacity
from .prescription.role_budget import role_time_set_ceiling
from .prescription.upper_role_allocation import allocate_upper_role_sets, uses_joint_upper_budget
from .prescription.metadata import apply_candidate_prescription_metadata
from .personalization_history import suggest_specialization_from_recent_sets
from .prescription.underfill import (
    planned_volume_complete_for_request,
    resolve_underfill_decision,
)
from .prescription.targets import apply_deterministic_targets
from .prescription.adjuncts import (
    adjunct_block_matches_profile,
    apply_requested_adjunct_prescription,
    is_working_prescription,
    prescription_set_type,
)
from .option_resolver import canonical_adjunct_options, candidate_matches_adjunct_option
from .adjunct_admission import admission_purposes
from .experience_policy import automatic_recommendation_eligibility
from .prompt_builder import _format_request_pain_areas, build_compiled_system_prompt
from .schemas import (
    LLMExercisePlan,
    LLMRoutineOutput,
    LLMRoutineSelectionOutput,
    RecentSetRecord,
    RoutineDraftResponse,
    AdjunctDecision,
    InjuryConditionEntry,
    RoutineRequest,
    UserProfileContext,
    PersonalizationSuggestion,
)
from .routine_selection import (
    align_candidate_bundles_with_composition,
    candidates_have_required_main_pattern,
    is_redundant_push_front_raise,
    apply_blueprint_execution_policy,
    apply_blueprint_execution_policy_to_draft,
    adapt_structured_routine_response,
    ensure_antagonist_superset_pair,
    render_accessory_group_plan,
    normalize_routine_selection_response,
    order_routine_output_by_selection_phase,
    required_main_pattern_targets,
    routine_draft_has_required_main_pattern,
    routine_output_has_required_main_pattern,
    selection_pattern_limit,
)
from .temperature_policy import resolve_generation_temperature
from .routine_telemetry import classify_fallback_reason, observe_routine_quality
from .routine_blueprint import (
    RoutineBlueprint,
    RoutineBlueprintValidationError,
    build_blueprint_candidate_bundles,
    build_routine_blueprint,
    render_blueprint_for_prompt,
    render_candidate_bundles_for_prompt,
)
from .shoulder_condition_policy import (
    active_uncleared_requires_safe_stop,
    candidate_has_shoulder_load,
    shoulder_restriction_labels_ko,
)
from .specialization_policy import (
    primary_focus_candidate_ids,
    apply_blueprint_specialization_volume,
    apply_blueprint_training_method,
)

logger = logging.getLogger(__name__)

_TRUE_VALUES = {"true", "1", "yes", "on"}
_BROAD_INJURY_TARGET_OVERLAP_RATIO = 0.5
_BROAD_INJURY_MIN_BODY_PARTS = 5
_PROFESSIONAL_CLEARANCE_WARNING = "수술/질환 이력이 있는 부위는 의료진 허가 범위 안에서만 진행하세요."
_CLEARANCE_REQUIRED_STATUSES = {
    "needs_clearance",
    "requires_clearance",
    "medical_clearance_required",
    "doctor_clearance_required",
    "post_surgery",
}
_ADJUNCT_TRAINING_STYLES = {"crossfit", "mobility", "yoga_pilates", "rehab_prehab"}
_ADJUNCT_OPTION_LABELS = {
    "mobility_warmup": "모빌리티 워밍업",
    "cooldown_stretch": "쿨다운 스트레칭",
    "conditioning_finisher": "컨디셔닝 마무리",
    "endurance_support": "지구력 보강",
    "prehab_accessory": "프리햅 보조운동",
}
_MOBILITY_CONTEXT_LABELS = {
    "limited_ankle_dorsiflexion": "발뒤꿈치를 붙인 깊은 앉기 동작의 어려움",
    "limited_hip_flexion": "엉덩이를 접어 숙이는 동작의 어려움",
    "limited_shoulder_flexion": "팔을 머리 위로 올리는 동작의 어려움",
    "limited_wrist_extension": "손목을 뒤로 젖혀 부하를 버티는 동작의 어려움",
}
_MOBILITY_CONTEXT_MATCH_TOKENS = {
    "limited_ankle_dorsiflexion": ("발뒤꿈치", "깊은 앉기", "발목", "배굴"),
    "limited_hip_flexion": ("엉덩이를 접어", "힙힌지", "고관절 굴곡"),
    "limited_shoulder_flexion": ("머리 위로", "오버헤드", "어깨 굴곡"),
    "limited_wrist_extension": ("손목을 뒤로", "손목 신전", "손목 가동"),
}
_MOBILITY_DEMAND_WARNING_POLICY = {
    "limited_ankle_dorsiflexion": (
        "ankle_dorsiflexion_demand",
        "발뒤꿈치를 붙인 깊은 앉기 동작의 어려움",
    ),
    "limited_hip_flexion": (
        "hip_flexion_demand",
        "엉덩이를 접어 숙이는 동작의 어려움",
    ),
    "limited_shoulder_flexion": (
        "shoulder_flexion_demand",
        "팔을 머리 위로 올리는 동작의 어려움",
    ),
    "limited_wrist_extension": (
        "wrist_extension_demand",
        "손목을 뒤로 젖혀 부하를 버티는 동작의 어려움",
    ),
}
_PROFILE_SIGNAL_CONTEXTS = {
    "self_reported_long_arms": {
        "targets": {
            "chest", "upper-back", "trapezius", "front-deltoids",
            "back-deltoids", "biceps", "triceps", "forearm",
        },
        "tokens": ("팔이 유난히 긴", "긴 팔", "팔 길이 신호"),
        "message": (
            "팔이 유난히 긴 편이라는 입력을 고려해 손잡이·벤치 위치와 통증 없는 "
            "가동범위를 먼저 맞추고, 고정 경로가 불편할 가능성이 큰 후보보다 "
            "세팅하거나 좌우를 독립적으로 조절하기 쉬운 후보를 우선했습니다."
        ),
    },
    "self_reported_long_thighs": {
        "targets": {"quadriceps", "hamstring", "gluteal", "adductor", "abductors"},
        "tokens": ("허벅지 구간이 유난히 긴", "긴 허벅지", "대퇴 길이 신호"),
        "message": (
            "허벅지 구간이 유난히 긴 편이라는 입력을 고려해 발 위치·스탠스·가동범위를 "
            "먼저 조절하고, 대퇴 길이 민감도가 높은 후보보다 세팅하기 쉬운 후보를 "
            "우선했습니다."
        ),
    },
}
_MOBILITY_RELEVANT_TARGETS = {
    "limited_ankle_dorsiflexion": {
        "quadriceps", "hamstring", "gluteal", "calves", "adductor", "abductors",
    },
    "limited_hip_flexion": {
        "quadriceps", "hamstring", "gluteal", "lower-back", "adductor", "abductors",
    },
    "limited_shoulder_flexion": {
        "chest", "upper-back", "trapezius", "front-deltoids", "back-deltoids", "triceps",
    },
    "limited_wrist_extension": {
        "chest", "upper-back", "trapezius", "front-deltoids", "back-deltoids",
        "biceps", "triceps", "forearm",
    },
}
_EXPERIENCE_LABELS = {
    "beginner": "초보자",
    "intermediate": "중급자",
    "advanced": "상급자",
}
_EXPERIENCE_TOKENS = {
    "beginner": ("초보자", "초급자", "입문자"),
    "intermediate": ("중급자", "중간 수준의 숙련자"),
    "advanced": ("상급자", "고급자"),
}
_MUSCLE_COACHING_LABELS = {
    "chest": "가슴",
    "front-deltoids": "어깨",
    "back-deltoids": "후면 어깨",
    "triceps": "삼두",
    "biceps": "이두",
    "upper-back": "등",
    "trapezius": "승모근",
    "quadriceps": "대퇴사두근",
    "hamstring": "햄스트링",
    "gluteal": "둔근",
    "calves": "종아리",
    "adductor": "내전근",
    "abductors": "외전근",
    "abs": "복근",
}
_COMPOSITION_CLAIM_TOKENS = ("중심", "위주", "구성")
_CONSTRAINT_CONTEXT_TOKENS = ("부상", "부담", "제외", "피해", "근육통", "피로", "고려")


def _local_raw_json_invoke_flag_enabled() -> bool:
    return os.getenv("ENABLE_LOCAL_RAW_JSON_INVOKE", "").strip().lower() in _TRUE_VALUES


def _should_use_local_raw_json_invoke() -> bool:
    return (
        _local_raw_json_invoke_flag_enabled()
        and resolve_llm_provider().effective_provider == "local"
    )


def _extract_llm_message_content(message) -> str:
    content, _shape = _extract_llm_message_content_with_shape(message)
    return content


def _bind_local_selection_decoder(llm, slot_candidate_bundles):
    if isinstance(llm, LocalJsonChatModel):
        return llm.bind(format=build_selection_decoding_schema(slot_candidate_bundles))
    return llm


def _format_slot_candidate_details(
    slot_candidate_bundles: list[dict[str, Any]],
    ranked_candidates: list[dict[str, Any]],
) -> str:
    """Render candidate facts under their authoritative slot, without a second flat list."""
    candidate_by_id = {
        str(candidate.get("id") or "").strip().lower(): candidate
        for candidate in ranked_candidates
        if str(candidate.get("id") or "").strip()
    }
    sections: list[str] = []
    for bundle in slot_candidate_bundles:
        slot_id = str(bundle.get("slotId") or "").strip()
        if slot_id == "flex":
            continue
        candidates = [
            candidate_by_id[exercise_id]
            for item in bundle.get("candidates") or []
            if (exercise_id := str(item.get("exerciseId") or "").strip().lower()) in candidate_by_id
        ]
        details = format_candidates_for_prompt(candidates) if candidates else "- no safe candidate"
        sections.append(
            f"SLOT {slot_id} | requirement={bundle.get('requirement')} | purpose={bundle.get('purpose')}\n"
            f"{details}"
        )
    return "\n\n".join(sections)


def _content_length_bucket(value: str) -> str:
    length = len(value or "")
    if length == 0:
        return "empty"
    if length < 100:
        return "short_lt_100"
    if length < 1000:
        return "medium_lt_1000"
    return "long_gte_1000"


def _safe_metadata_category(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) <= 64 and all(char.isalnum() or char in "_.:-" for char in text):
        return text
    return "present"


def _text_from_content_blocks(blocks: list[Any]) -> str:
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _extract_response_metadata(message) -> dict[str, Any]:
    metadata = getattr(message, "response_metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _extract_llm_message_content_with_shape(message) -> tuple[str, dict[str, Any]]:
    response_class = type(message).__name__
    content_value = getattr(message, "content", None)
    content_present = content_value is not None
    content_type = type(content_value).__name__ if content_present else "missing"
    text: Optional[str] = None

    if isinstance(content_value, str):
        text = content_value
    elif isinstance(content_value, list):
        text = _text_from_content_blocks(content_value)
    else:
        nested_message = getattr(message, "message", None)
        nested_content = getattr(nested_message, "content", None) if nested_message is not None else None
        if isinstance(nested_content, str):
            text = nested_content
            content_present = True
            content_type = "message.content:str"
        elif isinstance(nested_content, list):
            text = _text_from_content_blocks(nested_content)
            content_present = True
            content_type = "message.content:list"
        else:
            text_method = getattr(message, "text", None)
            if callable(text_method):
                try:
                    method_text = text_method()
                    if isinstance(method_text, str):
                        text = method_text
                        content_present = True
                        content_type = "text_method:str"
                except Exception:
                    text = None

    if text is None:
        text = ""

    metadata = _extract_response_metadata(message)
    usage_metadata = getattr(message, "usage_metadata", None)
    shape = {
        "raw_invoke_response_class": response_class,
        "raw_invoke_content_present": content_present,
        "raw_invoke_content_type": content_type,
        "raw_invoke_content_length_bucket": _content_length_bucket(text),
        "raw_invoke_content_stripped_empty": not bool(text.strip()),
        "raw_invoke_has_response_metadata": bool(metadata),
        "raw_invoke_finish_reason": _safe_metadata_category(
            metadata.get("finish_reason") or metadata.get("done_reason")
        ),
        "raw_invoke_done_reason": _safe_metadata_category(
            metadata.get("done_reason") or metadata.get("done")
        ),
        "raw_invoke_error_category": "present" if metadata.get("error") else None,
        "raw_invoke_usage_present": usage_metadata is not None,
    }
    return text, shape


def _invoke_local_raw_json_output_with_shape(prompt, llm, invoke_kwargs: dict) -> tuple[LLMRoutineOutput, dict[str, Any]]:
    raw_resp = (prompt | llm).invoke(invoke_kwargs)
    raw_text, shape = _extract_llm_message_content_with_shape(raw_resp)
    return normalize_llm_response(raw_text), shape


def _invoke_local_raw_json_output(prompt, llm, invoke_kwargs: dict) -> LLMRoutineOutput:
    output, _shape = _invoke_local_raw_json_output_with_shape(prompt, llm, invoke_kwargs)
    return output


def _safe_positive_int_env(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw.strip())
    except ValueError:
        return None
    return value if value > 0 else None


def _is_false_value(value: Any) -> bool:
    if value is False:
        return True
    if value is None:
        return False
    return str(value).strip().lower() in {"false", "0", "no", "off"}


def _policy_requires_professional_clearance_notice(condition_policies: List[dict]) -> bool:
    for policy in condition_policies or []:
        if not isinstance(policy, dict):
            continue
        status = str(policy.get("status") or policy.get("phase") or "").strip().lower()
        policy_type = str(policy.get("policyType") or policy.get("policy_type") or "").strip().lower()
        clearance = policy.get("professionalClearance", policy.get("professional_clearance"))
        requires_clearance = policy.get(
            "requiresProfessionalClearance",
            policy.get("requires_professional_clearance"),
        )
        if _is_false_value(clearance) or requires_clearance is True:
            return True
        if status in _CLEARANCE_REQUIRED_STATUSES:
            return True
        if policy_type == "surgeryhistory" and status:
            return True
    return False


def _apply_deterministic_request_warnings(
    response: RoutineDraftResponse,
    condition_policies: List[dict],
) -> RoutineDraftResponse:
    if not _policy_requires_professional_clearance_notice(condition_policies):
        return response
    response.warnings = _merge_unique_strings(response.warnings, [_PROFESSIONAL_CLEARANCE_WARNING])
    return response


def _apply_adjunct_application_warnings(
    response: RoutineDraftResponse,
    blueprint: RoutineBlueprint,
    ranked_candidates: List[dict],
) -> RoutineDraftResponse:
    candidate_by_id = {
        str(candidate.get("id") or "").strip().lower(): candidate
        for candidate in ranked_candidates
        if str(candidate.get("id") or "").strip()
    }
    slot_id_by_option = {
        slot.adjunct_option: slot.slot_id
        for slot in blueprint.slots
        if slot.adjunct_option
    }
    suppression_reasons = {
        **dict(blueprint.adjunct_suppression_reasons),
        **dict(response._adjunct_suppression_reason_by_option or {}),
    }
    used_exercise_ids: set[str] = set()
    decisions: list[AdjunctDecision] = []
    missing: list[str] = []
    for option in blueprint.requested_adjunct_options:
        if option not in _ADJUNCT_OPTION_LABELS:
            continue
        matched_block = next(
            (
                block
                for block in response.routine_blocks
                if str(block.exercise_id or "").strip().lower() not in used_exercise_ids
                and block.adjunct_option == option
                and adjunct_block_matches_profile(
                    block,
                    option,
                    candidate=candidate_by_id.get(
                        str(block.exercise_id or "").strip().lower()
                    ),
                )
            ),
            None,
        )
        if matched_block is None:
            matched_block = next(
                (
                    block
                    for block in response.routine_blocks
                    if str(block.exercise_id or "").strip().lower()
                    not in used_exercise_ids
                    and adjunct_block_matches_profile(
                        block,
                        option,
                        candidate=candidate_by_id.get(
                            str(block.exercise_id or "").strip().lower()
                        ),
                    )
                ),
                None,
            )
        if matched_block is not None:
            exercise_id = str(matched_block.exercise_id or "").strip().lower()
            used_exercise_ids.add(exercise_id)
            matched_block.adjunct_option = option
            decisions.append(AdjunctDecision(
                option=option,
                status="applied",
                reason_code="applied",
                slot_id=slot_id_by_option.get(option),
                exercise_id=matched_block.exercise_id,
            ))
            continue

        reason_code = suppression_reasons.get(option)
        if reason_code is None:
            has_eligible_candidate = any(
                candidate_matches_adjunct_option(candidate, option)
                for candidate in ranked_candidates
            )
            reason_code = (
                "postValidationRemoved"
                if has_eligible_candidate
                else "noEligibleCandidate"
            )
        decisions.append(AdjunctDecision(
            option=option,
            status="suppressed",
            reason_code=reason_code,
            slot_id=slot_id_by_option.get(option),
        ))
        missing.append(option)

    response.adjunct_decisions = decisions
    if not missing:
        return response
    warnings = [
        (
            f"요청한 {_ADJUNCT_OPTION_LABELS.get(option, option)}은(는) "
            "메인 운동 최소 구성, 안전 후보와 시간 예산을 우선해 이번 루틴에 반영하지 못했습니다."
        )
        for option in missing
    ]
    response.warnings = _merge_unique_strings(response.warnings, warnings)
    return response


def _candidate_warning_lookup(candidates: List[dict]) -> dict[str, dict]:
    return {
        str(candidate.get("id") or "").strip().lower(): candidate
        for candidate in candidates or []
        if str(candidate.get("id") or "").strip()
    }


def _split_slug_tokens(value: Any) -> set[str]:
    return {
        item.strip().lower()
        for item in str(value or "").replace("_", "-").split(",")
        if item.strip()
    }


def _expand_body_part_aliases(parts: set[str]) -> set[str]:
    expanded = set(parts)
    if expanded & {"left-soleus", "right-soleus", "soleus"}:
        expanded.update({"soleus", "calves", "ankle"})
    if expanded & {"ankle", "calves"}:
        expanded.update({"ankle", "calves"})
    if expanded & {"front-deltoids", "back-deltoids", "side-deltoids"}:
        expanded.add("shoulder")
    if "shoulder" in expanded:
        expanded.update({"front-deltoids", "back-deltoids", "side-deltoids"})
    return expanded


def _expand_doms_levels(doms_data: Dict[str, int] | dict[str, Any] | None) -> dict[str, int]:
    expanded: dict[str, int] = {}
    for part, raw_level in (doms_data or {}).items():
        level = int(raw_level or 0)
        if level <= 0:
            continue
        normalized = _normalize_body_part_slug(part)
        for alias in _expand_body_part_aliases({normalized}):
            expanded[alias] = max(level, expanded.get(alias, 0))
    return expanded


def has_target_relevant_recovery_constraint(
    req: RoutineRequest,
    target_muscles: List[str],
) -> bool:
    """Return whether recovery inputs constrain the requested target scope."""
    target_parts = _expand_body_part_aliases({
        _normalize_body_part_slug(part)
        for part in target_muscles
        if str(part).strip()
    })
    pain_parts = _expand_body_part_aliases({
        _normalize_body_part_slug(pain.body_part)
        for pain in (req.pain_areas or [])
        if pain.body_part
    })
    doms_parts = set(_expand_doms_levels(req.doms_data))
    lower_targets = {
        "quadriceps",
        "hamstring",
        "gluteal",
        "calves",
        "adductor",
        "abductors",
        "soleus",
    }
    indirect_pain_constraint = (
        bool(target_parts & lower_targets)
        and bool(pain_parts & {"lower-back", "back", "lumbar", "knees", "knee"})
    )
    return (
        bool(req.condition_policies)
        or bool(target_parts & (pain_parts | doms_parts))
        or indirect_pain_constraint
        or _has_target_relevant_managed_injury_condition(req, target_muscles)
    )


def _has_target_relevant_managed_injury_condition(
    req: RoutineRequest,
    target_muscles: List[str],
) -> bool:
    guarded_areas = {
        condition.area_code
        for condition in req.injury_conditions
        if condition.status in {"activeUncleared", "managedWithRestrictions"}
    }
    if not guarded_areas:
        return False

    targets = {
        _normalize_body_part_slug(part)
        for part in target_muscles
        if str(part).strip()
    }
    lower_targets = {
        "quadriceps",
        "hamstring",
        "gluteal",
        "calves",
        "adductor",
        "abductors",
        "soleus",
        "left-soleus",
        "right-soleus",
    }
    upper_targets = {
        "chest",
        "upper-back",
        "lower-back",
        "trapezius",
        "front-deltoids",
        "side-deltoids",
        "back-deltoids",
        "biceps",
        "triceps",
        "forearm",
        "neck",
    }
    area_target_groups = {
        "ankle": lower_targets,
        "knee": lower_targets,
        "hip": lower_targets,
        "lumbar": lower_targets
        | {"lower-back", "upper-back", "trapezius", "abs", "obliques"},
        "wrist": upper_targets,
        "elbow": upper_targets,
        "shoulder": upper_targets,
        "neck": upper_targets,
    }
    return any(
        bool(targets & area_target_groups.get(area, set()))
        for area in guarded_areas
    )


def _target_recovery_constraint_flags(
    req: RoutineRequest,
    target_muscles: List[str],
) -> tuple[bool, bool, bool]:
    """Return target-relevant injury, severe DOMS, and level 1/2 DOMS flags."""
    target_parts = _expand_body_part_aliases({
        _normalize_body_part_slug(part)
        for part in target_muscles
        if str(part).strip()
    })
    injury_parts = _expand_body_part_aliases({
        _normalize_body_part_slug(pain.body_part)
        for pain in (req.pain_areas or [])
        if pain.body_part
    })
    doms_levels = _expand_doms_levels(req.doms_data)
    severe_doms_parts = {part for part, level in doms_levels.items() if level >= 3}
    recovery_doms_parts = {part for part, level in doms_levels.items() if level in {1, 2}}
    return (
        bool(target_parts & injury_parts),
        bool(target_parts & severe_doms_parts),
        bool(target_parts & recovery_doms_parts),
    )


def _apply_underfill_explanation(
    response: RoutineDraftResponse,
    req: RoutineRequest,
    ranked_candidates: List[dict],
    routine_blueprint: RoutineBlueprint,
    selection_diagnostics: dict[str, Any],
) -> RoutineDraftResponse:
    if selection_diagnostics.get("specialization_focus_unavailable"):
        response.warnings = _merge_unique_strings(
            response.warnings,
            ["요청한 가슴 세부 집중에 맞는 후보가 없어, 현재 제한을 지키는 기본 가슴 구성으로 제안합니다. 세부 집중은 적용되지 않았습니다."],
        )
    injury_limited, severe_doms_limited, recovery_limited = (
        _target_recovery_constraint_flags(req, list(routine_blueprint.target_muscles))
    )
    # Requesting or listing equipment is not evidence that equipment caused a
    # shortage. The candidate pipeline does not preserve that causal proof.
    equipment_constrained = False
    fallback_used = response.is_fallback or response.generation_status == "fallback"
    planned_volume_complete = planned_volume_complete_for_request(
        goal=req.goal,
        experience_level=req.experience_level,
        target_muscles=routine_blueprint.target_muscles,
        exercise_count=len(response.routine_blocks),
        target_exercise_count=routine_blueprint.target_exercise_count,
    )
    decision = resolve_underfill_decision(
        actual_duration_min=response.total_estimated_time,
        available_duration_min=req.time_available_min,
        has_routine_blocks=bool(response.routine_blocks),
        readiness_level=req.readiness_level,
        injury_limited=injury_limited,
        severe_doms_limited=severe_doms_limited,
        recovery_limited=recovery_limited,
        condition_policy_limited=(
            bool(req.condition_policies)
            or _has_target_relevant_managed_injury_condition(
                req,
                list(routine_blueprint.target_muscles),
            )
        ),
        equipment_constrained=equipment_constrained,
        candidate_count=len(ranked_candidates),
        minimum_exercise_count=routine_blueprint.minimum_exercise_count,
        fallback_used=fallback_used,
        planned_volume_complete=planned_volume_complete,
        selection_diagnostics=selection_diagnostics,
    )
    diagnostics = decision.to_diagnostics()
    response._underfill_diagnostics = diagnostics
    selection_diagnostics.update({
        f"underfill_{key}": value
        for key, value in diagnostics.items()
    })
    if decision.underfilled:
        response.rationale_summary = _merge_unique_strings(
            response.rationale_summary,
            [decision.user_summary_ko or ""],
        )
        response.warnings = _merge_unique_strings(
            response.warnings,
            [decision.user_option_ko or ""],
        )
    # Different adjustment paths can translate to the same user-facing warning.
    response.warnings = _merge_unique_strings(response.warnings)
    return response


def _format_body_part_labels(parts: List[str] | set[str]) -> str:
    return ", ".join(
        _replace_schema_tokens(str(part))
        for part in parts
        if str(part).strip()
    )


def _apply_pain_and_doms_warnings(
    response: RoutineDraftResponse,
    req: RoutineRequest,
    ranked_candidates: List[dict],
    target_muscles: Optional[List[str]] = None,
) -> RoutineDraftResponse:
    reported_pain_parts = {
        _normalize_body_part_slug(pain.body_part)
        for pain in (req.pain_areas or [])
        if pain.body_part
    }
    pain_parts = _expand_body_part_aliases(reported_pain_parts)
    doms = _expand_doms_levels(req.doms_data)
    if not pain_parts and not doms:
        return response

    guarded = response.model_copy(deep=True)
    candidates_by_id = _candidate_warning_lookup(ranked_candidates)
    warnings: list[str] = []
    resolved_target_muscles = target_muscles or resolve_request_target_muscles(
        req.target_split_label,
        req.target_muscles,
    )
    target_parts = _expand_body_part_aliases({
        _normalize_body_part_slug(part)
        for part in resolved_target_muscles
        if str(part).strip()
    })
    excluded_reported_parts = sorted(
        part
        for part in reported_pain_parts
        if _expand_body_part_aliases({part}) & target_parts
    )
    if excluded_reported_parts:
        labels = _format_body_part_labels(excluded_reported_parts)
        warnings.append(
            f"등록된 부상 부위({labels})와 직접 겹치는 목표 운동은 후보에서 제외했습니다. "
            "루틴이 짧아졌다면 다른 부위를 선택하거나 운동을 보류하세요."
        )
    original_blocks = list(guarded.routine_blocks)
    retained_blocks = []
    removed_injury_blocks: list[str] = []
    for block in original_blocks:
        candidate = candidates_by_id.get(str(block.exercise_id or "").strip().lower())
        candidate_primary = _normalize_body_part_slug(candidate.get("primary_muscle")) if candidate else ""
        candidate_secondary = _split_slug_tokens(candidate.get("secondary_muscle")) if candidate else set()
        candidate_pain_triggers = _split_slug_tokens(candidate.get("pain_triggers")) if candidate else set()
        block_primary = {
            _normalize_body_part_slug(part)
            for part in (block.primary_muscles or [])
            if str(part).strip()
        }
        involved = _expand_body_part_aliases(
            ({candidate_primary} if candidate_primary else set()) | candidate_secondary | block_primary
        )
        candidate_pain_triggers = _expand_body_part_aliases(candidate_pain_triggers)

        candidate_for_injury_guard = candidate or {
            "primary_muscle": ",".join(block.primary_muscles or []),
            "secondary_muscle": "",
            "pain_triggers": "",
        }
        pain_overlap = sorted(
            _candidate_injury_overlap(candidate_for_injury_guard, pain_parts)
            | (block_primary & pain_parts)
        )
        if pain_overlap:
            pain_labels = _format_body_part_labels(pain_overlap)
            removed_injury_blocks.append(f"{block.exercise_name}({pain_labels})")
            continue

        retained_blocks.append(block)

        doms_overlap = sorted(part for part in involved if int(doms.get(part, 0) or 0) in {1, 2})
        if doms_overlap:
            strong = [part for part in doms_overlap if int(doms.get(part, 0) or 0) == 2]
            if strong:
                strong_labels = _format_body_part_labels(strong)
                warnings.append(
                    f"{block.exercise_name}: 최근 강한 피로 부위({strong_labels})를 사용할 수 있습니다. 세트 수와 강도를 낮춰 진행하세요."
                )
            else:
                doms_labels = _format_body_part_labels(doms_overlap)
                warnings.append(
                    f"{block.exercise_name}: 최근 피로 부위({doms_labels})를 사용할 수 있습니다. 컨디션에 따라 볼륨을 줄이세요."
                )

    if removed_injury_blocks:
        removed_labels = ", ".join(removed_injury_blocks)
        warnings.append(
            f"등록된 부상 부위와 겹치는 운동을 안전 규칙에 따라 제외했습니다: {removed_labels}."
        )
        guarded.routine_blocks = retained_blocks
        _normalize_execution_groups_after_block_removal(original_blocks, guarded.routine_blocks)
        for order, block in enumerate(guarded.routine_blocks, start=1):
            block.order = order
        guarded.total_estimated_time = _estimate_routine_blocks_time(guarded.routine_blocks)

        if original_blocks and not guarded.routine_blocks:
            guarded.generation_status = "failed"
            guarded.status_reason_code = "emptyCandidate"
            guarded.is_fallback = False
            guarded.total_estimated_time = 0
            guarded.summary_title = "등록된 부상 부위를 피해 구성할 수 없어요"
            guarded.rationale_summary = [
                "등록된 부상 부위에 관여하는 운동을 모두 제외한 뒤 안전한 후보가 남지 않았습니다."
            ]
            warnings.append("오늘 목표 부위를 바꾸거나 운동을 보류해 주세요.")

    if warnings:
        guarded.warnings = _merge_unique_strings(guarded.warnings, warnings)
    return guarded


def _normalize_execution_groups_after_block_removal(
    original_blocks: List[Any],
    retained_blocks: List[Any],
) -> None:
    original_counts: dict[str, int] = {}
    retained_counts: dict[str, int] = {}

    def group_key(block: Any) -> str:
        return str(block.execution_group_id or f"single:{block.exercise_id}")

    for block in original_blocks:
        key = group_key(block)
        original_counts[key] = original_counts.get(key, 0) + 1
    for block in retained_blocks:
        key = group_key(block)
        retained_counts[key] = retained_counts.get(key, 0) + 1

    for block in retained_blocks:
        key = group_key(block)
        if retained_counts.get(key, 0) >= original_counts.get(key, 0):
            continue
        block.execution_group_id = None
        block.execution_group_type = "single"
        block.execution_group_position = 1
        block.execution_group_size = 1
        block.transition_rest_sec = None
        block.group_rest_sec = None


def _estimate_routine_blocks_time(blocks: List[Any]) -> int:
    plans: list[LLMExercisePlan] = []
    execution_metadata: dict[str, dict[str, Any]] = {}
    for block in blocks:
        first_set = block.prescription[0] if block.prescription else None
        plan = LLMExercisePlan(
            exercise_id=block.exercise_id,
            exercise_name=block.exercise_name,
            movement_pattern=block.movement_pattern,
            movement_type=block.movement_type,
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
        execution_metadata[exercise_id] = {
            "executionGroupId": block.execution_group_id,
            "executionGroupType": block.execution_group_type,
            "executionGroupPosition": block.execution_group_position,
            "executionGroupSize": block.execution_group_size,
            "transitionRestSec": block.transition_rest_sec,
            "groupRestSec": block.group_rest_sec,
        }
    return estimate_routine_time_min(plans, execution_metadata=execution_metadata)


def _normalize_body_part_slug(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _should_block_broad_injury_context(req: RoutineRequest, target_muscles: List[str]) -> bool:
    if str(req.readiness_level or "").strip().lower() != "low":
        return False
    pain_parts = {
        _normalize_body_part_slug(pain.body_part)
        for pain in (req.pain_areas or [])
        if pain.body_part
    }
    target_parts = {
        _normalize_body_part_slug(part)
        for part in target_muscles
        if str(part).strip()
    }
    if len(pain_parts) < _BROAD_INJURY_MIN_BODY_PARTS or not target_parts:
        return False
    target_overlap = pain_parts & target_parts
    return (len(target_overlap) / len(target_parts)) >= _BROAD_INJURY_TARGET_OVERLAP_RATIO


def _broad_injury_failed_response(
    adjunct_options: Optional[List[str]] = None,
) -> RoutineDraftResponse:
    return RoutineDraftResponse(
        generation_status="failed",
        status_reason_code="emptyCandidate",
        is_fallback=False,
        total_estimated_time=0,
        summary_title="오늘은 루틴 생성을 보류했어요",
        rationale_summary=[
            "낮은 컨디션과 여러 등록 부상 부위가 오늘 목표 근육과 겹쳐, 안전한 루틴을 억지로 구성하지 않았습니다.",
        ],
        routine_blocks=[],
        warnings=[
            "등록된 부상 부위를 운동하지 않도록 목표 부위를 바꾸거나 오늘 운동을 보류해 주세요.",
            "부상 부위 운동 재개 여부는 필요한 경우 의료 전문가와 상의해 주세요.",
        ],
        adjunct_decisions=[
            AdjunctDecision(
                option=option,
                status="suppressed",
                reason_code="noEligibleCandidate",
                slot_id=f"adjunct_{option}",
            )
            for option in canonical_adjunct_options(adjunct_options)
        ],
    )


def _active_uncleared_shoulder_failed_response(
    adjunct_options: Optional[List[str]] = None,
) -> RoutineDraftResponse:
    return RoutineDraftResponse(
        generation_status="failed",
        status_reason_code="emptyCandidate",
        is_fallback=False,
        total_estimated_time=0,
        summary_title="현재 어깨 상태로는 요청한 상체 루틴을 만들 수 없어요",
        rationale_summary=[
            "의료진 허가 전으로 등록된 어깨 부상 상태에서는 어깨에 부하가 걸리는 가슴·등·어깨 운동을 추천하지 않습니다.",
        ],
        routine_blocks=[],
        warnings=[
            "오늘은 하체 또는 어깨가 개입하지 않는 코어 운동으로 목표를 바꿔 주세요.",
            "상체 운동 재개 여부와 허용 동작은 의료 전문가에게 확인해 주세요.",
        ],
        adjunct_decisions=[
            AdjunctDecision(
                option=option,
                status="suppressed",
                reason_code="noEligibleCandidate",
                slot_id=f"adjunct_{option}",
            )
            for option in canonical_adjunct_options(adjunct_options)
        ],
    )


def _active_uncleared_neck_failed_response(
    adjunct_options: Optional[List[str]] = None,
) -> RoutineDraftResponse:
    return RoutineDraftResponse(
        generation_status="failed",
        status_reason_code="emptyCandidate",
        is_fallback=False,
        total_estimated_time=0,
        summary_title="현재 목 상태로는 요청한 상체 루틴을 만들 수 없어요",
        rationale_summary=[
            "의료진 허가 전으로 등록된 목 부상 상태에서는 목의 안정화가 필요한 상체 운동을 추천하지 않습니다.",
        ],
        routine_blocks=[],
        warnings=[
            "오늘은 목에 부담이 적은 다른 목표로 바꾸거나 운동을 보류해 주세요.",
            "상체 운동 재개 여부와 허용 동작은 의료 전문가에게 확인해 주세요.",
        ],
        adjunct_decisions=[
            AdjunctDecision(
                option=option,
                status="suppressed",
                reason_code="noEligibleCandidate",
                slot_id=f"adjunct_{option}",
            )
            for option in canonical_adjunct_options(adjunct_options)
        ],
    )


def _apply_shoulder_condition_warnings(
    response: RoutineDraftResponse,
    req: RoutineRequest,
) -> RoutineDraftResponse:
    shoulder = next((
        condition for condition in resolve_injury_condition_contexts(
            req.injury_conditions, [], legacy_shoulder_contexts=[
                (req.shoulder_condition_status, req.restricted_movement_patterns)
            ]
        ) if condition.area_code == "shoulder"
    ), None)
    status = shoulder.status if shoulder else None
    warning: Optional[str] = None
    if status == "activeUncleared":
        warning = "의료진 허가 전 어깨 부상 상태에 따라 어깨 부하 운동을 후보 단계에서 제외했습니다."
    elif status == "managedWithRestrictions":
        labels = shoulder_restriction_labels_ko(shoulder.restricted_movement_patterns)
        warning = (
            "관리 중인 어깨 상태에 따라 제한 동작"
            f"({', '.join(labels)})을 후보 단계에서 제외했습니다."
        )
    if warning and warning not in response.warnings:
        response.warnings.append(warning)
    if status == "managedWithRestrictions":
        response.warnings = _merge_unique_strings(response.warnings, [
            "운동 진행 선택은 의료진 허가를 대신하지 않습니다. 어깨 운동의 허용 범위가 불명확하거나 "
            "현재 통증이 있다면 해당 부위 운동을 보류하고 의료 전문가에게 확인하세요."
        ])
    return response


def _apply_injury_condition_warnings(
    response: RoutineDraftResponse,
    req: RoutineRequest,
) -> RoutineDraftResponse:
    for condition in req.injury_conditions:
        if condition.area_code == "shoulder":
            continue
        area_label = injury_area_label_ko(condition.area_code)
        warning: Optional[str] = None
        if condition.status == "activeUncleared":
            warning = (
                f"의료진 허가 전 {area_label} 부상 상태에 따라 "
                "해당 부위에 부하가 걸리는 운동을 후보 단계에서 제외했습니다."
            )
        elif condition.status == "managedWithRestrictions":
            labels = injury_condition_labels_ko(condition)
            warning = (
                f"관리 중인 {area_label} 상태에 따라 제한 동작"
                f"({', '.join(labels)})을 후보 단계에서 제외했습니다."
            )
        if warning and warning not in response.warnings:
            response.warnings.append(warning)
    return response


def _required_main_pattern_failed_response(
    blueprint: RoutineBlueprint,
) -> RoutineDraftResponse:
    return RoutineDraftResponse(
        generation_status="failed",
        status_reason_code="emptyCandidate",
        is_fallback=False,
        total_estimated_time=0,
        summary_title="요청한 메인 운동을 안전하게 구성할 수 없어요",
        rationale_summary=[
            "현재 부상·장비·운동 조건을 모두 적용한 뒤 목표 부위의 메인 복합 운동 후보가 남지 않았습니다."
        ],
        routine_blocks=[],
        warnings=[
            "보조 운동만으로 정상 루틴을 대신하지 않았습니다. 목표 부위나 장비 조건을 바꾸거나 오늘 운동을 보류해 주세요."
        ],
        adjunct_decisions=[
            AdjunctDecision(
                option=option,
                status="suppressed",
                reason_code="noEligibleCandidate",
                slot_id=f"adjunct_{option}",
            )
            for option in canonical_adjunct_options(blueprint.planned_adjunct_options)
        ],
    )


def _merge_unique_strings(*groups: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group or []:
            text = str(item).strip()
            key = text.lower()
            if not text or key in seen:
                continue
            seen.add(key)
            merged.append(text)
    return merged


def _merge_profile_injury_areas(
    request_areas: Optional[List[Any]],
    profile_areas: Optional[List[Any]],
) -> List[Any]:
    """Union binary injury areas so a request cannot erase profile safety state."""

    merged: List[Any] = []
    seen: set[tuple[str, str]] = set()
    for group in (profile_areas or [], request_areas or []):
        for area in group:
            body_part = _normalize_body_part_slug(getattr(area, "body_part", ""))
            side = str(getattr(area, "side", "") or "").strip().lower()
            key = (body_part, side)
            if not body_part or key in seen:
                continue
            seen.add(key)
            merged.append(area)
    return merged


def _merge_candidate_sources(*groups: Optional[List[dict]]) -> List[dict]:
    merged: List[dict] = []
    by_id: dict[str, dict] = {}
    for group in groups:
        for candidate in group or []:
            exercise_id = str(candidate.get("id") or "").strip()
            key = exercise_id.lower()
            if not key:
                continue
            existing = by_id.get(key)
            if existing is None:
                copied = dict(candidate)
                copied["adjunct_option_codes"] = list(dict.fromkeys(
                    copied.get("adjunct_option_codes") or []
                ))
                by_id[key] = copied
                merged.append(copied)
                continue
            codes = list(dict.fromkeys([
                *(existing.get("adjunct_option_codes") or []),
                *(candidate.get("adjunct_option_codes") or []),
            ]))
            existing["adjunct_option_codes"] = codes
            for field in (
                "adjunct_catalog_version",
                "adjunct_catalog_review_status",
                "adjunct_display_name_ko",
                "adjunct_purpose_ko",
            ):
                if candidate.get(field):
                    existing[field] = candidate[field]
            reviewed_display_name = str(
                candidate.get("adjunct_display_name_ko")
                or candidate.get("user_facing_display_name_kr")
                or ""
            ).strip()
            if candidate.get("adjunct_option_codes") and reviewed_display_name:
                existing["name_kr"] = reviewed_display_name
                existing["display_name_ko"] = reviewed_display_name
                existing["user_facing_display_name_kr"] = reviewed_display_name
    return merged


def _user_visible_rationale_text(response: RoutineDraftResponse) -> str:
    parts: list[str] = [
        response.summary_title or "",
        *(response.rationale_summary or []),
        *(response.warnings or []),
    ]
    for block in response.routine_blocks:
        parts.append(block.exercise_rationale or "")
    return " ".join(str(item).strip().lower() for item in parts if str(item).strip())


def _enforce_experience_rationale_consistency(
    response: RoutineDraftResponse,
    experience_level: Optional[str],
) -> RoutineDraftResponse:
    """Keep user-facing coaching text consistent with the structured level input."""
    expected = str(experience_level or "").strip().lower()
    expected_label = _EXPERIENCE_LABELS.get(expected)
    if not expected_label:
        return response

    conflicting_tokens = {
        token
        for level, tokens in _EXPERIENCE_TOKENS.items()
        if level != expected
        for token in tokens
    }

    def normalize(text: str) -> str:
        normalized = str(text or "")
        for token in sorted(conflicting_tokens, key=len, reverse=True):
            normalized = normalized.replace(token, expected_label)
        return normalized

    guarded = response.model_copy(deep=True)
    guarded.summary_title = normalize(guarded.summary_title)
    guarded.rationale_summary = [normalize(item) for item in guarded.rationale_summary]
    guarded.warnings = [normalize(item) for item in guarded.warnings]
    for block in guarded.routine_blocks:
        block.exercise_rationale = normalize(block.exercise_rationale)
    return guarded


def _is_lateral_raise_name(value: Any) -> bool:
    compact = re.sub(r"[\s/_()·,.-]+", "", str(value or "").strip().lower())
    return "레터럴레이즈" in compact or "lateralraise" in compact


def _enforce_response_fact_consistency(
    response: RoutineDraftResponse,
    req: RoutineRequest,
) -> RoutineDraftResponse:
    """Repair narrow user-facing claims that contradict structured output facts.

    The public muscle contract currently collapses lateral-deltoid catalog rows
    into the legacy front-deltoid slug. Keep that contract untouched here, while
    preventing the Korean coaching text from calling a lateral raise a front-
    deltoid or chest exercise.
    """
    guarded = response.model_copy(deep=True)
    normalized_goal = re.sub(r"[-_\s]+", "", str(req.goal or "").strip().lower())

    def normalize_goal(text: str) -> str:
        value = str(text or "")
        if normalized_goal == "generalfitness":
            value = value.replace("전반적인 근력 향상", "전반적인 체력 향상")
        return value

    guarded.summary_title = normalize_goal(guarded.summary_title)
    guarded.rationale_summary = [normalize_goal(item) for item in guarded.rationale_summary]
    guarded.warnings = [normalize_goal(item) for item in guarded.warnings]
    for block in guarded.routine_blocks:
        block.exercise_rationale = normalize_goal(block.exercise_rationale)

    lateral_blocks = [
        block for block in guarded.routine_blocks
        if _is_lateral_raise_name(block.exercise_name)
    ]
    for block in lateral_blocks:
        focus_note = (
            " 측면 어깨 집중 목표를 반영해 해당 운동과 수행 각도를 우선 배치했습니다."
            if "집중 목표를 반영" in str(block.exercise_rationale or "")
            else ""
        )
        block.exercise_rationale = (
            "측면 어깨 보조 볼륨을 채우는 고립 운동으로 배치했습니다."
            f"{focus_note}"
        )
        if block._weight_guidance:
            block.exercise_rationale += f" {block._weight_guidance}"

    has_upper_body_press = any(
        ("프레스" in str(block.exercise_name or "").lower()
         or "press" in str(block.exercise_name or "").lower())
        and bool(set(block.primary_muscles or []) & {"chest", "front-deltoids"})
        for block in guarded.routine_blocks
    )
    has_other_front_isolation = any(
        str(block.movement_type or "").strip().upper() == "ISOLATION"
        and "front-deltoids" in set(block.primary_muscles or [])
        and not _is_lateral_raise_name(block.exercise_name)
        for block in guarded.routine_blocks
    )
    if lateral_blocks:
        def normalize_lateral_summary(text: str) -> str:
            value = str(text or "")
            should_repair = (
                not has_upper_body_press
                or (
                    not has_other_front_isolation
                    and ("고립" in value or "레터럴" in value)
                )
            )
            if not should_repair:
                return value
            value = value.replace("전면 삼각근", "측면 어깨")
            value = value.replace("전면 어깨", "측면 어깨")
            value = value.replace("측면 어깨와 가슴 보조 볼륨", "측면 어깨 보조 볼륨")
            value = value.replace("가슴과 측면 어깨를 보조", "측면 어깨를 보조")
            value = re.sub(
                r"측면 어깨(?:와|과)\s*가슴의\s*보조적인\s*자극",
                "측면 어깨의 보조적인 자극",
                value,
            )
            value = value.replace("측면 어깨과", "측면 어깨와")
            return value

        guarded.summary_title = normalize_lateral_summary(guarded.summary_title)
        guarded.rationale_summary = [
            normalize_lateral_summary(item) for item in guarded.rationale_summary
        ]
        guarded.warnings = [normalize_lateral_summary(item) for item in guarded.warnings]

    selected_equipment = {
        token.strip().upper()
        for block in guarded.routine_blocks
        for token in str(block.equipment_type or "").split(",")
        if token.strip()
    }
    if "DUMBBELL" in selected_equipment:
        equipment_only_claim = re.compile(
            r"(?:머신\s*(?:및|과)\s*케이블|케이블\s*(?:및|과)\s*머신)\s*위주로"
        )

        def normalize_equipment_claim(text: str) -> str:
            return equipment_only_claim.sub("선택된 장비를 사용하도록", str(text or ""))

        guarded.summary_title = normalize_equipment_claim(guarded.summary_title)
        guarded.rationale_summary = [
            normalize_equipment_claim(item) for item in guarded.rationale_summary
        ]
        guarded.warnings = [normalize_equipment_claim(item) for item in guarded.warnings]

    if req.pain_areas and guarded.routine_blocks:
        requested_muscles = resolve_request_target_muscles(
            req.target_split_label,
            req.target_muscles,
        )
        selected_muscles = {
            str(muscle or "").strip().lower()
            for block in guarded.routine_blocks
            for muscle in (block.primary_muscles or [])
            if str(muscle or "").strip()
        }
        suppressed_muscles = {
            muscle for muscle in requested_muscles if muscle not in selected_muscles
        }
        suppressed_labels = {
            _MUSCLE_COACHING_LABELS[muscle]
            for muscle in suppressed_muscles
            if muscle in _MUSCLE_COACHING_LABELS
        }

        def claims_suppressed_composition(text: str) -> bool:
            value = str(text or "")
            return (
                any(label in value for label in suppressed_labels)
                and any(token in value for token in _COMPOSITION_CLAIM_TOKENS)
                and not any(token in value for token in _CONSTRAINT_CONTEXT_TOKENS)
            )

        retained_summary = [
            item for item in guarded.rationale_summary
            if not claims_suppressed_composition(item)
        ]
        if len(retained_summary) != len(guarded.rationale_summary):
            selected_labels = [
                _MUSCLE_COACHING_LABELS[muscle]
                for muscle in requested_muscles
                if muscle in selected_muscles and muscle in _MUSCLE_COACHING_LABELS
            ]
            if selected_labels:
                composition_line = (
                    "등록 부상과 겹치는 목표 운동을 제외하고 "
                    f"{'·'.join(dict.fromkeys(selected_labels))} 중심으로 구성했습니다."
                )
                retained_summary = _merge_unique_strings(
                    [composition_line],
                    retained_summary,
                )
            guarded.rationale_summary = retained_summary

    return guarded


def _mobility_limit_is_relevant_to_request(
    req: RoutineRequest,
    raw_limit: Any,
) -> bool:
    limit = str(raw_limit or "").strip().lower()
    relevant_targets = _MOBILITY_RELEVANT_TARGETS.get(limit)
    if not relevant_targets:
        return False
    request_targets = set(resolve_request_target_muscles(
        req.target_split_label,
        req.target_muscles,
    ))
    return bool(request_targets & relevant_targets)


def _apply_active_context_rationale(
    response: RoutineDraftResponse,
    req: RoutineRequest,
    *,
    mobility_limits: Optional[List[str]] = None,
    experience_level: Optional[str] = None,
    ranked_candidates: Optional[List[dict[str, Any]]] = None,
) -> RoutineDraftResponse:
    """Keep active request constraints visible even when a small model omits them."""
    existing_text = _user_visible_rationale_text(response)
    context_lines: list[str] = []

    pain_parts = sorted({
        _normalize_body_part_slug(pain.body_part)
        for pain in (req.pain_areas or [])
        if pain.body_part
    })
    if pain_parts:
        pain_labels = _format_body_part_labels(pain_parts)
        if "등록 부상" not in existing_text:
            context_lines.append(
                f"입력한 등록 부상 부위({pain_labels})와 직접 겹치는 운동은 추천 후보에서 제외했습니다."
            )

    for raw_limit in mobility_limits or req.mobility_limits or []:
        limit = str(raw_limit or "").strip().lower()
        label = _MOBILITY_CONTEXT_LABELS.get(limit)
        if not label or not _mobility_limit_is_relevant_to_request(req, limit):
            continue
        label_tokens = _MOBILITY_CONTEXT_MATCH_TOKENS.get(limit, ())
        if not any(token in existing_text for token in label_tokens):
            context_lines.append(
                f"입력한 {label}을 고려해 가동범위 요구도가 낮거나 안정적인 후보를 우선했습니다."
            )

    request_targets = set(resolve_request_target_muscles(
        req.target_split_label,
        req.target_muscles,
    ))
    selected_profile_signals = set(req.profile_signals or [])
    for signal_code, context in _PROFILE_SIGNAL_CONTEXTS.items():
        if signal_code not in selected_profile_signals:
            continue
        if not (request_targets & context["targets"]):
            continue
        if not any(token in existing_text for token in context["tokens"]):
            context_lines.append(context["message"])

    if str(req.readiness_level or "").strip().lower() == "low" and not any(
        token in existing_text for token in ("컨디션", "준비 상태", "피로", "readiness")
    ):
        context_lines.append("오늘 컨디션이 낮아 운동 강도와 분량을 보수적으로 계획했습니다.")

    if req.time_available_min <= 30 and not any(
        token in existing_text for token in ("시간", "짧은", f"{req.time_available_min}분")
    ):
        context_lines.append(
            f"{req.time_available_min}분 안에 마칠 수 있도록 핵심 메인 운동과 필요한 보조 운동을 우선했습니다."
        )

    effective_experience = str(experience_level or req.experience_level or "").strip().lower()
    if effective_experience == "beginner" and not any(
        token in existing_text for token in ("초보", "초급", "입문", "beginner")
    ):
        context_lines.append("초보자 숙련도를 고려해 기술 난도가 낮고 안정적인 운동을 우선했습니다.")
    if effective_experience == "intermediate" and not any(
        token in existing_text for token in ("중급", "중간 수준", "intermediate")
    ):
        context_lines.append("중급자 숙련도에 맞춰 점진적 과부하를 적용하기 쉬운 보편 운동을 우선했습니다.")
    if effective_experience == "advanced" and not any(
        token in existing_text for token in ("상급", "고급", "advanced")
    ):
        context_lines.append("상급자 숙련도에 맞춰 효과와 기술 난도의 균형을 검토했습니다.")

    is_advanced_high_strength = (
        effective_experience == "advanced"
        and str(req.readiness_level or "").strip().lower() == "high"
        and str(req.goal or "").strip().lower() == "strength"
    )
    if is_advanced_high_strength:
        selected_ids = {
            str(block.exercise_id or "").strip().lower()
            for block in response.routine_blocks
            if block.exercise_id
        }
        selected_metadata = [
            candidate
            for candidate in (ranked_candidates or [])
            if str(candidate.get("id") or "").strip().lower() in selected_ids
        ]
        has_high_strength_effect = any(
            str(candidate.get("strength_effect") or "").strip().lower() in {"high", "elite"}
            for candidate in selected_metadata
        )
        has_risk_and_load_metadata = any(
            str(candidate.get("injury_caution_level") or "").strip()
            and str(candidate.get("lumbar_load") or "").strip()
            and str(candidate.get("axial_load") or "").strip()
            for candidate in selected_metadata
        )
        already_explains_tradeoff = (
            any(token in existing_text for token in ("효과", "effect"))
            and any(token in existing_text for token in ("부상 주의", "위험", "부하", "risk"))
        )
        if has_high_strength_effect and has_risk_and_load_metadata and not already_explains_tradeoff:
            context_lines.append(
                "상급자의 높은 컨디션과 근력 목표에 맞춰 근력 효과가 높은 후보를 허용하되, 부상 주의도와 요추/축방향 부하를 함께 검토했습니다."
            )

    if context_lines:
        response.rationale_summary = _merge_unique_strings(
            context_lines,
            response.rationale_summary,
        )
    return response


def _apply_selected_mobility_demand_warnings(
    response: RoutineDraftResponse,
    req: RoutineRequest,
    *,
    mobility_limits: Optional[List[str]] = None,
    ranked_candidates: Optional[List[dict[str, Any]]] = None,
) -> RoutineDraftResponse:
    """Disclose selected movement demands and bounded setup instructions."""
    candidate_by_id = {
        str(candidate.get("id") or "").strip().lower(): candidate
        for candidate in ranked_candidates or []
        if str(candidate.get("id") or "").strip()
    }
    if not candidate_by_id or not response.routine_blocks:
        return response

    selected_candidates: list[tuple[RoutineBlock, dict[str, Any], str, str]] = []
    for block in response.routine_blocks:
        exercise_id = str(block.exercise_id or "").strip().lower()
        candidate = candidate_by_id.get(exercise_id)
        if not candidate:
            continue
        name_text = " ".join(
            str(value or "").strip().lower()
            for value in (
                block.exercise_name,
                candidate.get("name_kr"),
                candidate.get("name_en"),
            )
        )
        compact_name = re.sub(r"[\s_\-/]+", "", name_text)
        selected_candidates.append(
            (block, candidate, compact_name, candidate_selection_pattern_key(candidate))
        )

    def selected_names_matching(
        predicate: Callable[[dict[str, Any], str, str], bool],
    ) -> list[str]:
        names: list[str] = []
        for block, candidate, compact_name, selection_pattern in selected_candidates:
            if not predicate(candidate, compact_name, selection_pattern):
                continue
            exercise_name = str(block.exercise_name or "").strip()
            if exercise_name and exercise_name not in names:
                names.append(exercise_name)
        return names

    def has_name_token(compact_name: str, *tokens: str) -> bool:
        return any(
            re.sub(r"\s+", "", token.lower()) in compact_name
            for token in tokens
        )

    upper_press_muscles = {
        "chest",
        "front-deltoids",
        "side-deltoids",
        "triceps",
    }

    def upper_press_names() -> list[str]:
        return selected_names_matching(
            lambda candidate, compact_name, selection_pattern: (
                str(candidate.get("primary_muscle") or "").strip().lower()
                in upper_press_muscles
                and (
                    selection_pattern in {"horizontal_press", "vertical_press"}
                    or has_name_token(compact_name, "푸시업", "pushup", "push-up")
                )
            )
        )

    def squat_or_leg_press_names() -> list[str]:
        return selected_names_matching(
            lambda candidate, compact_name, _selection_pattern: (
                str(candidate.get("primary_muscle") or "").strip().lower()
                in {"quadriceps", "gluteal"}
                and has_name_token(
                    compact_name,
                    "레그프레스",
                    "legpress",
                    "스쿼트",
                    "squat",
                )
            )
        )

    def lunge_names() -> list[str]:
        return selected_names_matching(
            lambda candidate, compact_name, _selection_pattern: (
                str(candidate.get("primary_muscle") or "").strip().lower()
                in {"quadriceps", "gluteal"}
                and has_name_token(compact_name, "런지", "lunge")
            )
        )

    warnings: list[str] = []
    for raw_limit in mobility_limits or req.mobility_limits or []:
        limit = str(raw_limit or "").strip().lower()
        policy = _MOBILITY_DEMAND_WARNING_POLICY.get(limit)
        if not policy or not _mobility_limit_is_relevant_to_request(req, limit):
            continue
        field, label = policy
        high_demand_names: list[str] = []
        for block in response.routine_blocks:
            exercise_id = str(block.exercise_id or "").strip().lower()
            candidate = candidate_by_id.get(exercise_id)
            if not candidate or str(candidate.get(field) or "").strip().lower() != "high":
                continue
            exercise_name = str(block.exercise_name or "").strip()
            if exercise_name and exercise_name not in high_demand_names:
                high_demand_names.append(exercise_name)
        if high_demand_names:
            names = "·".join(high_demand_names[:3])
            warnings.append(
                f"{label} 입력과 관련해 가동범위 요구도가 높은 운동({names})이 포함되었습니다. "
                "편하게 수행 가능한 범위만 사용하고, 통증이나 뚜렷한 불편이 나타나면 "
                "해당 운동을 제외하고 낮은 요구도의 대체 운동을 선택하세요."
            )

        if limit == "limited_wrist_extension":
            press_names = upper_press_names()
            if press_names:
                warnings.append(
                    f"손목 젖힘이 어려운 입력을 반영해 { '·'.join(press_names[:3]) }에서는 "
                    "손목을 중립에 가깝게 유지하고, 뒤로 과도하게 젖혀지기 전까지만 "
                    "그립과 가동범위를 조정하세요. 유지하기 어렵다면 머신이나 중립 그립 대체 운동을 선택하세요."
                )
        elif limit == "limited_ankle_dorsiflexion":
            lower_names = squat_or_leg_press_names()
            if lower_names:
                warnings.append(
                    f"발목 가동범위 입력을 반영해 { '·'.join(lower_names[:3]) }에서는 "
                    "발뒤꿈치가 들리거나 제한된 깊이로 들어가기 전까지만 수행하고, 발 위치와 가동범위를 먼저 조정하세요."
                )
            selected_lunges = lunge_names()
            if selected_lunges:
                warnings.append(
                    f"발목 가동범위 입력을 반영해 { '·'.join(selected_lunges[:3]) }에서는 "
                    "앞쪽 지지 발이 바닥에서 들리기 전까지만 깊이를 사용하고, 보폭과 앞발 위치를 먼저 조정하세요."
                )

    managed_patterns = {
        pattern
        for condition in req.injury_conditions or []
        if str(condition.status or "").strip() == "managedWithRestrictions"
        for pattern in condition.restricted_movement_patterns
    }
    managed_shoulder = any(
        condition.area_code == "shoulder" and condition.status == "managedWithRestrictions"
        for condition in req.injury_conditions or []
    ) or req.shoulder_condition_status == "managedWithRestrictions"
    if managed_shoulder:
        instruction = (
            "관리 중인 어깨의 허용된 가동범위 안에서만 수행하세요. 통증이 생기면 해당 운동을 중단하고 "
            "계속하거나 대체하기 전에 의료 전문가에게 허용 동작을 확인하세요."
        )
        for block, candidate, _name, _pattern in selected_candidates:
            if not candidate_has_shoulder_load(candidate):
                continue
            if instruction not in block.exercise_rationale:
                block.exercise_rationale = f"{block.exercise_rationale} {instruction}".strip()
            warnings.append(f"{block.exercise_name}: {instruction}")
    if "deepKneeFlexion" in managed_patterns:
        lower_names = squat_or_leg_press_names()
        if lower_names:
            warnings.append(
                f"관리 중인 깊은 무릎 굽힘 제한을 반영해 { '·'.join(lower_names[:3]) }에서는 "
                "제한된 깊이에 들어가기 전에 멈추도록 발 위치와 가동범위를 설정하세요."
            )
        selected_lunges = lunge_names()
        if selected_lunges:
            warnings.append(
                f"관리 중인 깊은 무릎 굽힘 제한을 반영해 { '·'.join(selected_lunges[:3]) }에서는 "
                "앞쪽 지지 다리의 제한된 굽힘 깊이에 들어가기 전에 멈추도록 보폭과 가동범위를 설정하세요."
            )
    if "loadedWristExtension" in managed_patterns:
        press_names = upper_press_names()
        if press_names:
            warnings.append(
                f"관리 중인 손목 젖힘 제한을 반영해 { '·'.join(press_names[:3]) }에서는 "
                "손목을 중립에 가깝게 유지하고 뒤로 과도하게 젖혀지기 전까지만 그립과 가동범위를 조정하세요."
            )

    if warnings:
        response.warnings = _merge_unique_strings(response.warnings, warnings)
    return response


def _enforce_same_area_injury_precedence_wording(
    response: RoutineDraftResponse,
    req: RoutineRequest,
) -> RoutineDraftResponse:
    """Keep same-area injury + DOMS copy aligned with injury precedence."""
    reported_pain_parts = {
        _normalize_body_part_slug(pain.body_part)
        for pain in (req.pain_areas or [])
        if pain.body_part
    }
    expanded_doms = set(_expand_doms_levels(req.doms_data))
    expanded_pain = _expand_body_part_aliases(reported_pain_parts)
    overlapping_reported_parts = {
        part
        for part in reported_pain_parts
        if _expand_body_part_aliases({part}) & expanded_doms
    }
    if not overlapping_reported_parts:
        return response

    labels = _format_body_part_labels(sorted(overlapping_reported_parts))
    label_tokens = {
        token.lower()
        for part in overlapping_reported_parts
        for token in (part, _replace_schema_tokens(part))
        if token
    }
    doms_terms = ("근육통", "피로", "DOMS", "doms", "뻐근")
    all_doms_covered_by_injury = bool(expanded_doms) and expanded_doms <= expanded_pain
    precedence_line = (
        f"등록된 부상 부위({labels})는 근육통 단계와 관계없이 해당 부위에 관여하는 운동을 제외했습니다."
    )

    def repair(text: str) -> str:
        value = str(text or "")
        lowered = value.lower()
        has_doms_word = any(term in value for term in doms_terms)
        if has_doms_word and (
            all_doms_covered_by_injury
            or any(token in lowered for token in label_tokens)
        ):
            return precedence_line
        return value

    guarded = response.model_copy(deep=True)
    guarded.rationale_summary = _merge_unique_strings(
        [precedence_line],
        [repair(item) for item in guarded.rationale_summary],
    )
    guarded.warnings = _merge_unique_strings(
        [],
        [repair(item) for item in guarded.warnings],
    )
    for block in guarded.routine_blocks:
        block.exercise_rationale = repair(block.exercise_rationale)
        block.reasons = _merge_unique_strings([], [repair(item) for item in block.reasons])
    return guarded


def _format_prompt_mapping(value: Any) -> str:
    if not value:
        return "none"
    if isinstance(value, dict):
        return ", ".join(f"{key}={val}" for key, val in value.items())
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(", ".join(f"{key}={val}" for key, val in item.items() if val is not None))
            else:
                parts.append(str(item))
        return " | ".join(part for part in parts if part) or "none"
    return str(value)


def _rendered_prompt_char_count(prompt, invoke_kwargs: dict) -> int:
    try:
        messages = prompt.format_messages(**invoke_kwargs)
    except Exception:
        return 0
    return sum(len(str(message)) for message in messages)


def _fallback(
    req: RoutineRequest,
    ranked_candidates: List[dict],
    max_total_sets: int,
    doms_db: Dict[str, int],
    goal: str,
    status_reason_code: StatusReasonCode,
    recent_sets: Optional[List[RecentSetRecord]],
    profile: Optional[UserProfileContext],
) -> RoutineDraftResponse:
    fallback_targets = resolve_request_target_muscles(
        req.target_split_label,
        req.target_muscles,
    )
    blueprint = build_routine_blueprint(
        target_split_label=req.target_split_label,
        target_muscles=fallback_targets,
        goal=goal,
        experience_level=req.experience_level or (profile.experience_level if profile else None),
        advanced_chest_focus=req.advanced_chest_focus,
        specialization_focus=req.specialization_focus,
        training_method=req.training_method,
        method_mode=req.method_mode,
        requested_method=req.requested_method,
        recent_training_method_uses=req.recent_training_method_uses,
        readiness_level=req.readiness_level,
        time_available_min=req.time_available_min,
        training_style=req.training_style,
        session_structure=req.session_structure,
        adjunct_options=req.adjunct_options,
        pain_areas=req.pain_areas,
        doms_data=req.doms_data,
        injury_conditions=req.injury_conditions,
    )
    blueprint = replace(blueprint, max_total_sets=max_total_sets)
    draft = generate_fallback_routine(
        req,
        ranked_candidates,
        max_total_sets,
        doms_db,
        goal=goal,
        status_reason_code=status_reason_code,
        recent_sets=recent_sets,
        profile=profile,
        target_exercise_count=blueprint.target_exercise_count,
        minimum_main_exercise_count=blueprint.minimum_main_exercise_count,
        applied_adjunct_options=blueprint.planned_adjunct_options,
        routine_blueprint=blueprint,
    )
    draft = apply_blueprint_execution_policy_to_draft(
        draft,
        blueprint,
        ranked_candidates,
    )
    if not routine_draft_has_required_main_pattern(
        draft,
        blueprint,
        ranked_candidates,
    ):
        return _required_main_pattern_failed_response(blueprint)
    return draft


def _get_feedback_adjustments_if_enabled(
    db,
    user_id: Optional[str],
    candidates: List[dict],
) -> Optional[dict]:
    adjustments, _, _ = _get_feedback_adjustments_with_status(db, user_id, candidates)
    return adjustments


def _get_feedback_adjustments_with_status(
    db,
    user_id: Optional[str],
    candidates: List[dict],
) -> tuple[Optional[dict], bool, Optional[str]]:
    from .feedback_aggregation import (
        get_feedback_adjustments,
        get_feedback_aware_ranking_enabled,
    )

    if not get_feedback_aware_ranking_enabled():
        return None, False, None

    exercise_ids = [
        exercise_id
        for candidate in candidates
        if (exercise_id := str(candidate.get("id") or "").strip().lower())
    ]
    if not exercise_ids:
        return None, False, None

    try:
        return get_feedback_adjustments(
            db,
            user_id=user_id,
            exercise_ids=exercise_ids,
        ), False, None
    except Exception as exc:
        err = sanitize_exception_for_log(exc)
        logger.warning(
            "[FeedbackRanker] adjustment fetch failed type=%s category=%s",
            err['type'],
            err['category'],
        )
        return None, True, err["category"]


def _compute_feedback_stats(
    ranked: List[dict],
    adjustments: Optional[dict],
) -> dict:
    if not adjustments:
        return {"adjusted": 0, "positive": 0, "negative": 0, "abs_avg": 0.0}

    values: List[float] = []
    for candidate in ranked:
        candidate_id = str(candidate.get("id") or "").strip().lower()
        if not candidate_id:
            continue
        adjustment = adjustments.get(candidate_id)
        if adjustment is None:
            continue
        raw = getattr(adjustment, "blended_adjustment", adjustment)
        try:
            value = max(-10.0, min(10.0, float(raw)))
        except (TypeError, ValueError):
            continue
        if value != 0.0:
            values.append(value)

    if not values:
        return {"adjusted": 0, "positive": 0, "negative": 0, "abs_avg": 0.0}

    return {
        "adjusted": len(values),
        "positive": sum(1 for value in values if value > 0),
        "negative": sum(1 for value in values if value < 0),
        "abs_avg": round(sum(abs(value) for value in values) / len(values), 3),
    }


def _finalize_success(
    output: LLMRoutineOutput,
    label: str,
    ranked_candidates: Optional[List[dict]] = None,
) -> RoutineDraftResponse:
    draft = build_routine_draft(
        output,
        "success",
        "none",
        False,
        ranked_candidates=ranked_candidates,
        preserve_input_order=True,
    )
    _debug_print_draft(label, draft)
    return draft


def _candidate_exercise_name(candidate: dict) -> str:
    return candidate_display_name(candidate)


def _candidate_primary_muscles(candidate: dict) -> List[str]:
    primary = str(candidate.get("primary_muscle") or "").strip()
    return [primary] if primary else []


def _candidate_equipment_type(candidate: dict) -> Optional[str]:
    selected = str(candidate.get("selected_equipment_code") or "").strip().upper()
    if selected:
        return selected
    options = parse_equipment_options(candidate.get("equipment_req"))
    return options[0] if options else None


def _candidate_role_set_budget(
    req: RoutineRequest,
    blueprint: RoutineBlueprint,
    candidates: List[dict],
    doms_db: Dict[str, int],
) -> int:
    """Relax the old all-compound estimate, never existing role/health caps."""
    roles = []
    signals = {}
    seen = set()
    focus_ids = primary_focus_candidate_ids(blueprint, candidates)
    for candidate in candidates:
        candidate_id = str(candidate.get("id") or "").strip().lower()
        if not candidate_id or candidate_id in seen:
            continue
        seen.add(candidate_id)
        if str(candidate.get("movement_type") or "").upper() not in {"COMPOUND", "ISOLATION"}:
            continue
        if str(candidate.get("prescription_mode") or "reps").lower() != "reps":
            continue
        if any(candidate_matches_adjunct_option(candidate, option)
               for option in blueprint.planned_adjunct_options):
            continue
        signal = _candidate_doms_prescription_signal(candidate, doms_db)
        if max(_candidate_doms_overlap(candidate, doms_db).values(), default=0) >= 3:
            continue
        plan = LLMExercisePlan(
            exercise_id=candidate_id, exercise_name=_candidate_exercise_name(candidate),
            movement_type=candidate.get("movement_type"),
            primary_muscles=_candidate_primary_muscles(candidate),
            sets=1, target_reps=1, rest_time_sec=0, exercise_rationale="budget only",
        )
        plan.sets = min(
            _max_sets_for_exercise(
                plan, blueprint.goal, blueprint.target_split_label,
                primary_focus=candidate_id in focus_ids,
            ),
            experience_set_cap(plan, blueprint.goal, blueprint.experience_level, blueprint.target_split_label),
        )
        roles.append(plan)
        signals[candidate_id] = signal
    prescribed = apply_deterministic_targets(
        LLMRoutineOutput(total_estimated_time=0, summary_title="budget only",
                         rationale_summary=[], exercises=roles),
        goal=blueprint.goal, recent_sets=None, profile=None,
        readiness_level=blueprint.readiness_level,
        target_split_label=blueprint.target_split_label,
        target_muscles=list(blueprint.target_muscles),
        experience_level=blueprint.experience_level,
        primary_focus_exercise_ids=focus_ids,
        doms_level_by_exercise_id={key: value["direct_level"] for key, value in signals.items()},
        secondary_doms_level_by_exercise_id={key: value["secondary_level"] for key, value in signals.items()},
    )
    ceiling = role_time_set_ceiling(
        prescribed.exercises, req.time_available_min, blueprint.target_main_exercise_count,
        grouped=blueprint.session_structure != "standard",
    )
    # This wave removes demonstrated over-trimming only. The old estimate
    # remains a floor so a sparse pool cannot newly suppress required slots.
    return max(blueprint.max_total_sets, ceiling)


def _add_underfilled_safe_candidates(
    validated: LLMRoutineOutput,
    req: RoutineRequest,
    ranked_candidates: List[dict],
    max_total_sets: int,
    db_target_muscles: List[str],
    routine_blueprint: RoutineBlueprint,
) -> LLMRoutineOutput:
    min_expected = routine_blueprint.minimum_main_exercise_count
    adjunct_option_by_id = dict(
        getattr(validated, "_adjunct_option_by_exercise_id", {}) or {}
    )

    def selected_main_count() -> int:
        return sum(
            str(exercise.exercise_id or "").strip().lower()
            not in adjunct_option_by_id
            for exercise in validated.exercises
        )

    productive_floor_min = minimum_productive_duration_min(req.time_available_min)
    estimated_time_before = estimate_routine_time_min(validated.exercises)
    injury_limited, severe_doms_limited, _ = _target_recovery_constraint_flags(
        req,
        db_target_muscles,
    )
    productive_fill_allowed = (
        str(req.readiness_level or "normal").strip().lower() != "low"
        and str(req.session_structure or "standard").strip().lower() == "standard"
        and not injury_limited
        and not severe_doms_limited
        and not req.condition_policies
        and not routine_blueprint.safety_limited_target_muscles
    )
    desired_count = min_expected
    if (
        productive_fill_allowed
        and estimated_time_before < productive_floor_min
    ):
        desired_count = routine_blueprint.target_main_exercise_count
    productive_fill_requested = (
        productive_fill_allowed
        and req.time_available_min >= 45
        and estimated_time_before < productive_floor_min
        and selected_main_count() < desired_count
    )
    if productive_fill_requested:
        validated._selection_diagnostics = {
            **dict(validated._selection_diagnostics or {}),
            "productive_fill_requested": True,
            "productive_fill_target_count": desired_count,
            "productive_fill_initial_count": selected_main_count(),
            "productive_fill_added_count": 0,
            "productive_fill_exhausted": False,
            "productive_fill_exhaustion_reason": None,
            "productive_fill_floor_min": productive_floor_min,
            "productive_fill_estimated_time_before": estimated_time_before,
            "productive_fill_estimated_time_after": estimated_time_before,
        }
    if (
        req.time_available_min < 45
        or str(req.readiness_level or "normal").strip().lower() == "low"
        or selected_main_count() >= desired_count
        or (not productive_fill_requested and len(ranked_candidates) < desired_count)
    ):
        return validated

    candidate_by_id = {
        str(candidate.get("id") or "").strip().lower(): candidate
        for candidate in ranked_candidates
        if str(candidate.get("id") or "").strip()
    }
    selected_ids = {str(exercise.exercise_id or "").strip().lower() for exercise in validated.exercises}
    selected_families = {
        candidate_selection_family_key(candidate_by_id[exercise_id])
        for exercise_id in selected_ids
        if exercise_id in candidate_by_id
    }
    selected_pattern_counts: Dict[str, int] = {}
    for exercise_id in selected_ids:
        candidate = candidate_by_id.get(exercise_id)
        if not candidate:
            continue
        pattern = candidate_selection_pattern_key(candidate)
        if pattern:
            selected_pattern_counts[pattern] = selected_pattern_counts.get(pattern, 0) + 1
    target_set = {str(muscle).strip() for muscle in (db_target_muscles or req.target_muscles or []) if str(muscle).strip()}
    current_sets = sum(max(0, int(exercise.sets or 0)) for exercise in validated.exercises)
    added = 0
    exhaustion_reason = "candidate_policy_exhausted"
    for candidate in ranked_candidates:
        if selected_main_count() >= desired_count:
            break
        candidate_id = str(candidate.get("id") or "").strip()
        if not candidate_id or candidate_id.lower() in selected_ids:
            continue
        if str(candidate.get("prescription_mode") or "reps").strip().lower() != "reps":
            continue
        if any(
            candidate_matches_adjunct_option(candidate, option)
            for option in routine_blueprint.planned_adjunct_options
        ):
            continue
        candidate_family = candidate_selection_family_key(candidate)
        if candidate_family and candidate_family in selected_families:
            continue
        candidate_pattern = candidate_selection_pattern_key(candidate)
        if is_redundant_push_front_raise(
            routine_blueprint.target_split_label, routine_blueprint.specialization_focus,
            candidate_pattern, {key for key, count in selected_pattern_counts.items() if count > 0},
            ranked_candidates,
        ):
            continue
        pattern_limit = selection_pattern_limit(candidate_pattern, routine_blueprint)
        if (
            pattern_limit is not None
            and selected_pattern_counts.get(candidate_pattern, 0) >= pattern_limit
        ):
            continue
        candidate_primary = str(candidate.get("primary_muscle") or "").strip()
        candidate_secondary = {
            part.strip()
            for part in str(candidate.get("secondary_muscle") or "").split(",")
            if part.strip()
        }
        if target_set and candidate_primary not in target_set and not (candidate_secondary & target_set):
            continue
        if current_sets + 2 > max_total_sets:
            exhaustion_reason = "set_cap_reached"
            break
        exercise_name = _candidate_exercise_name(candidate)
        if not exercise_name:
            continue
        primary_muscles = _candidate_primary_muscles(candidate)
        primary_label = ", ".join(primary_muscles) if primary_muscles else "목표 부위"
        plan = LLMExercisePlan(
                exercise_id=candidate_id,
                exercise_name=exercise_name,
                movement_type=str(candidate.get("movement_type") or "") or None,
                primary_muscles=primary_muscles,
                equipment_type=_candidate_equipment_type(candidate),
                target_weight_kg=None,
                target_reps=10,
                sets=2,
                rest_time_sec=90,
                exercise_rationale=(
                    f"현재 조건과 세션 시간을 고려해 {primary_label} 보조 볼륨을 "
                    "채우는 운동으로 추가했습니다."
                ),
            )
        validated.exercises.append(apply_candidate_prescription_metadata(plan, candidate))
        selected_ids.add(candidate_id.lower())
        if candidate_family:
            selected_families.add(candidate_family)
        if candidate_pattern:
            selected_pattern_counts[candidate_pattern] = (
                selected_pattern_counts.get(candidate_pattern, 0) + 1
            )
        current_sets += 2
        added += 1

    if added:
        validated.warnings = _merge_unique_strings(
            validated.warnings,
            [f"루틴 구성을 균형 있게 맞추기 위해 보조 운동 {added}개를 추가했습니다."],
        )
    if productive_fill_requested:
        estimated_time_after = estimate_routine_time_min(validated.exercises)
        exhausted = (
            selected_main_count() < desired_count
            and estimated_time_after < productive_floor_min
        )
        if exhausted and not ranked_candidates:
            exhaustion_reason = "ranked_candidate_shortage"
        validated._selection_diagnostics = {
            **dict(validated._selection_diagnostics or {}),
            "productive_fill_added_count": added,
            "productive_fill_exhausted": exhausted,
            "productive_fill_exhaustion_reason": exhaustion_reason if exhausted else None,
            "productive_fill_final_count": selected_main_count(),
            "productive_fill_estimated_time_after": estimated_time_after,
        }
        if exhausted:
            validated.warnings = _merge_unique_strings(
                validated.warnings,
                [
                    "중복과 현재 루틴 조건을 지키면서 추가할 적합한 운동 후보가 없어, "
                    "루틴이 이용 가능 시간보다 짧게 구성되었습니다."
                ],
            )
    return validated


def _normalize_training_style(value: Optional[str]) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _candidate_has_score_boost(candidate: dict, rule_code: str, constraint_code: Optional[str] = None) -> bool:
    for item in candidate.get("score_boosts") or []:
        current_rule = str(item.get("rule_code") or item.get("ruleCode") or "")
        current_constraint = str(item.get("constraint_code") or item.get("constraintCode") or "")
        if current_rule != rule_code:
            continue
        if constraint_code is not None and current_constraint != constraint_code:
            continue
        return True
    return False


def _candidate_to_llm_exercise_plan(candidate: dict, rationale: str) -> LLMExercisePlan:
    plan = LLMExercisePlan(
        exercise_id=str(candidate.get("id") or ""),
        exercise_name=_candidate_exercise_name(candidate),
        movement_type=str(candidate.get("movement_type") or "") or None,
        primary_muscles=_candidate_primary_muscles(candidate),
        equipment_type=_candidate_equipment_type(candidate),
        target_weight_kg=None,
        target_reps=10,
        sets=2,
        rest_time_sec=90,
        exercise_rationale=rationale,
    )
    return apply_candidate_prescription_metadata(plan, candidate)


def _ensure_requested_adjunct_style_candidate(
    validated: LLMRoutineOutput,
    req: RoutineRequest,
    ranked_candidates: List[dict],
    max_total_sets: int,
    target_exercise_count: Optional[int] = None,
    applied_adjunct_options: Optional[Sequence[str]] = None,
) -> LLMRoutineOutput:
    """Preserve one requested adjunct-style candidate after LLM selection.

    Fit-Core's current formula is still gym-based. Non-gym styles are accessory
    options, but small local models can drop the one style-matched candidate
    even when the ranker preserved it. Keep this deterministic and conservative:
    never touch the first main movement, and use only ranker-marked candidates.
    """
    normalized_style = _normalize_training_style(req.training_style)
    if normalized_style not in _ADJUNCT_TRAINING_STYLES:
        return validated

    purposes = (admission_purposes(req.adjunct_options, normalized_style)
        if applied_adjunct_options is None else canonical_adjunct_options(applied_adjunct_options))
    if normalized_style == 'yoga_pilates' and not purposes:
        return validated

    selected_ids = {
        str(exercise.exercise_id or "").strip().lower()
        for exercise in validated.exercises
        if str(exercise.exercise_id or "").strip()
    }
    candidates_by_id = {
        str(candidate.get("id") or "").strip().lower(): candidate
        for candidate in ranked_candidates
        if str(candidate.get("id") or "").strip()
    }
    if normalized_style == 'yoga_pilates' and any(
        option in purposes and exercise_id in selected_ids
        for exercise_id, option in validated._adjunct_option_by_exercise_id.items()
    ):
        return validated
    if any(
        _candidate_has_score_boost(
            candidates_by_id.get(exercise_id, {}),
            "training_style_match_boost",
            normalized_style,
        )
        for exercise_id in selected_ids
    ):
        return validated

    style_candidate = next(
        (
            candidate
            for candidate in ranked_candidates
            if str(candidate.get("id") or "").strip().lower() not in selected_ids
            and _candidate_has_score_boost(candidate, "training_style_match_boost", normalized_style)
            and _candidate_exercise_name(candidate)
            and automatic_recommendation_eligibility(candidate, req.experience_level,
                adjunct_options=purposes if normalized_style == 'yoga_pilates' else ()).eligible
            and (normalized_style != 'yoga_pilates' or any(
                candidate_matches_adjunct_option(candidate, option)
                and automatic_recommendation_eligibility(candidate, req.experience_level,
                    adjunct_options=(option,)).eligible for option in purposes))
        ),
        None,
    )
    if style_candidate is None:
        return validated

    current_sets = sum(max(0, int(exercise.sets or 0)) for exercise in validated.exercises)
    rationale = (
        "요청한 보조 운동 스타일과 맞는 후보라, 헬스장 메인 루틴을 유지한 채 "
        "마무리/보조 운동으로 보존했습니다."
    )
    style_plan = _candidate_to_llm_exercise_plan(style_candidate, rationale)
    exercise_count_cap = target_exercise_count or target_exercise_count_for_session(
        req.time_available_min,
        req.goal or "hypertrophy",
    )
    if current_sets + style_plan.sets <= max_total_sets and len(validated.exercises) < exercise_count_cap:
        validated.exercises.append(style_plan)
    elif len(validated.exercises) > 1:
        replace_index = len(validated.exercises) - 1
        style_primary = set(_candidate_primary_muscles(style_candidate))
        for index in range(len(validated.exercises) - 1, 0, -1):
            exercise_primary = set(validated.exercises[index].primary_muscles or [])
            if style_primary and exercise_primary & style_primary:
                replace_index = index
                break
        validated.exercises[replace_index] = style_plan
    else:
        return validated

    if normalized_style == 'yoga_pilates':
        purpose = next(option for option in purposes
            if candidate_matches_adjunct_option(style_candidate, option)
            and automatic_recommendation_eligibility(style_candidate, req.experience_level,
                adjunct_options=(option,)).eligible)
        eid = str(style_candidate['id']).strip().lower()
        validated._adjunct_option_by_exercise_id[eid] = purpose
        validated._selection_slot_by_exercise_id[eid] = f'adjunct_{purpose}'
        validated = apply_requested_adjunct_prescription(validated, purposes, ranked_candidates)

    validated.warnings = _merge_unique_strings(
        validated.warnings,
        ["요청한 보조 운동 스타일은 메인 운동이 아니라 보조/마무리 운동 1개로만 반영했습니다."],
    )
    return validated


def _canonicalize_llm_exercise_names(
    output: LLMRoutineOutput,
    ranked_candidates: List[dict],
) -> LLMRoutineOutput:
    candidates_by_id = {
        str(candidate.get("id") or "").strip().lower(): candidate
        for candidate in ranked_candidates
        if str(candidate.get("id") or "").strip()
    }
    for exercise in output.exercises:
        candidate = candidates_by_id.get(str(exercise.exercise_id or "").strip().lower())
        if not candidate:
            continue
        canonical_name = _candidate_exercise_name(candidate)
        if canonical_name:
            exercise.exercise_name = canonical_name
        equipment_type = _candidate_equipment_type(candidate)
        if equipment_type:
            exercise.equipment_type = equipment_type
    return output


def _deterministic_or_fallback(
    validated: LLMRoutineOutput,
    req: RoutineRequest,
    ranked_candidates: List[dict],
    max_total_sets: int,
    doms_db: Dict[str, int],
    goal: str,
    db_target_muscles: List[str],
    recent_sets: Optional[List[RecentSetRecord]],
    profile: Optional[UserProfileContext],
    output_label: str,
    draft_label: str,
    on_time_budget_exceeded: Optional[Callable[[], None]] = None,
    routine_blueprint: Optional[RoutineBlueprint] = None,
    selection_diagnostics: Optional[dict[str, Any]] = None,
) -> RoutineDraftResponse:
    if routine_blueprint is None:
        routine_blueprint = build_routine_blueprint(
            target_split_label=req.target_split_label,
            target_muscles=db_target_muscles,
            goal=goal,
            experience_level=req.experience_level or (profile.experience_level if profile else None),
            advanced_chest_focus=req.advanced_chest_focus,
            specialization_focus=req.specialization_focus,
            training_method=req.training_method,
            method_mode=req.method_mode,
            requested_method=req.requested_method,
            recent_training_method_uses=req.recent_training_method_uses,
            readiness_level=req.readiness_level,
            time_available_min=req.time_available_min,
            training_style=req.training_style,
            session_structure=req.session_structure,
            adjunct_options=req.adjunct_options,
            pain_areas=req.pain_areas,
            doms_data=req.doms_data,
            injury_conditions=req.injury_conditions,
        )
    validated = _canonicalize_llm_exercise_names(validated, ranked_candidates)
    validated = order_routine_output_by_selection_phase(validated)
    if not routine_output_has_required_main_pattern(
        validated,
        routine_blueprint,
        ranked_candidates,
    ):
        if selection_diagnostics is not None:
            selection_diagnostics.clear()
            selection_diagnostics.update({
                "required_main_pattern_missing": True,
                "required_main_pattern_targets": list(
                    required_main_pattern_targets(routine_blueprint)
                ),
            })
        return _required_main_pattern_failed_response(routine_blueprint)
    validated = _ensure_requested_adjunct_style_candidate(
        validated,
        req,
        ranked_candidates,
        max_total_sets,
        routine_blueprint.target_exercise_count,
        applied_adjunct_options=routine_blueprint.planned_adjunct_options,
    )
    validated = ensure_antagonist_superset_pair(
        validated,
        routine_blueprint,
        ranked_candidates,
    )
    safety_limited = set(routine_blueprint.safety_limited_target_muscles)
    protected_slot_ids = {
        slot.slot_id
        for slot in routine_blueprint.slots
        if slot.requirement == "required_when_safe"
        and not (slot.target_muscles and set(slot.target_muscles) <= safety_limited)
    }
    protected_exercise_ids = {
        exercise_id
        for exercise_id, slot_id in dict(
            getattr(validated, "_selection_slot_by_exercise_id", {}) or {}
        ).items()
        if slot_id in protected_slot_ids
    }
    doms_signal_by_exercise_id = {
        str(candidate.get("id") or "").strip().lower(): _candidate_doms_prescription_signal(
            candidate,
            doms_db,
        )
        for candidate in ranked_candidates
        if str(candidate.get("id") or "").strip()
    }
    deterministic = apply_deterministic_targets(
        validated,
        goal=goal,
        recent_sets=recent_sets,
        profile=profile,
        readiness_level=req.readiness_level,
        target_split_label=req.target_split_label,
        target_muscles=db_target_muscles,
        doms_db=doms_db,
        time_available_min=req.time_available_min,
        max_total_sets=None,
        protected_exercise_ids=protected_exercise_ids,
        doms_level_by_exercise_id={
            exercise_id: signal["direct_level"]
            for exercise_id, signal in doms_signal_by_exercise_id.items()
        },
        secondary_doms_level_by_exercise_id={
            exercise_id: signal["secondary_level"]
            for exercise_id, signal in doms_signal_by_exercise_id.items()
        },
        experience_level=req.experience_level,
        primary_focus_exercise_ids=primary_focus_candidate_ids(routine_blueprint, ranked_candidates),
    )
    time_fill_diagnostics = dict(
        getattr(deterministic, "_time_fill_diagnostics", {}) or {}
    )
    deterministic = apply_requested_adjunct_prescription(
        deterministic,
        routine_blueprint.planned_adjunct_options,
        ranked_candidates,
    )
    exercise_count_before_productive_fill = len(deterministic.exercises)
    deterministic = _add_underfilled_safe_candidates(
        deterministic,
        req,
        ranked_candidates,
        max_total_sets,
        db_target_muscles,
        routine_blueprint,
    )
    if len(deterministic.exercises) > exercise_count_before_productive_fill:
        # The productive-fill decision must use the prescribed duration, and a
        # newly added candidate must receive the same deterministic policy as
        # every model-selected exercise.
        deterministic = apply_deterministic_targets(
            deterministic,
            goal=goal,
            recent_sets=recent_sets,
            profile=profile,
            readiness_level=req.readiness_level,
            target_split_label=req.target_split_label,
            target_muscles=db_target_muscles,
            doms_db=doms_db,
            time_available_min=req.time_available_min,
            max_total_sets=None,
            protected_exercise_ids=protected_exercise_ids,
            doms_level_by_exercise_id={
                exercise_id: signal["direct_level"]
                for exercise_id, signal in doms_signal_by_exercise_id.items()
            },
            secondary_doms_level_by_exercise_id={
                exercise_id: signal["secondary_level"]
                for exercise_id, signal in doms_signal_by_exercise_id.items()
            },
            experience_level=req.experience_level,
            primary_focus_exercise_ids=primary_focus_candidate_ids(routine_blueprint, ranked_candidates),
        )
        time_fill_diagnostics = dict(
            getattr(deterministic, "_time_fill_diagnostics", {}) or {}
        )
        deterministic = apply_requested_adjunct_prescription(
            deterministic,
            routine_blueprint.planned_adjunct_options,
            ranked_candidates,
        )
    fill_working_ids = {
        str(exercise.exercise_id or "").strip().lower()
        for exercise in deterministic.exercises
        if is_working_prescription(exercise)
    }
    fill_working_sets = sum(
        max(0, exercise.sets)
        for exercise in deterministic.exercises
        if is_working_prescription(exercise)
    )
    pre_focus_sets = {
        str(exercise.exercise_id or "").strip().lower(): exercise.sets
        for exercise in deterministic.exercises
    }
    deterministic = apply_blueprint_specialization_volume(
        deterministic,
        routine_blueprint,
        ranked_candidates,
        restricted_exercise_ids={
            exercise_id
            for exercise_id, signal in doms_signal_by_exercise_id.items()
            if signal["direct_level"] > 0 or signal["secondary_level"] > 0
        },
    )
    focus_capacity = complete_lower_chest_focus_capacity(
        deterministic, routine_blueprint, req, ranked_candidates, pre_focus_sets,
        doms_db=doms_db,
        restricted_exercise_ids={
            exercise_id for exercise_id, signal in doms_signal_by_exercise_id.items()
            if signal["direct_level"] > 0 or signal["secondary_level"] > 0
        },
        max_total_sets=max_total_sets,
    )
    if focus_capacity is not None:
        time_fill_diagnostics = focus_capacity
        fill_working_sets = sum(
            max(0, exercise.sets) for exercise in deterministic.exercises
            if is_working_prescription(exercise)
        )
    deterministic = apply_blueprint_training_method(
        deterministic,
        routine_blueprint,
        ranked_candidates,
    )
    if selection_diagnostics is not None:
        selection_diagnostics.clear()
        selection_diagnostics.update(
            dict(getattr(deterministic, "_selection_diagnostics", {}) or {})
        )
    if uses_joint_upper_budget(routine_blueprint, deterministic._selection_slot_by_exercise_id):
        deterministic = apply_blueprint_execution_policy(deterministic, routine_blueprint)
        allocate_upper_role_sets(
            deterministic, routine_blueprint, req.time_available_min, max_total_sets
        )
    set_cap_enforced = enforce_total_set_cap(
        deterministic,
        max_total_sets,
        db_target_muscles,
        protected_exercise_ids=protected_exercise_ids,
    )
    deterministic = apply_blueprint_execution_policy(deterministic, routine_blueprint)
    final_working_ids = {
        str(exercise.exercise_id or "").strip().lower()
        for exercise in deterministic.exercises
        if is_working_prescription(exercise)
    }
    final_working_sets = sum(
        max(0, exercise.sets)
        for exercise in deterministic.exercises
        if is_working_prescription(exercise)
    )
    final_main_count = sum(
        str(exercise.exercise_id or "").strip().lower()
        not in dict(deterministic._adjunct_option_by_exercise_id or {})
        for exercise in deterministic.exercises
        if (
            prescription_set_type(exercise) == "working"
            and not getattr(exercise, "_target_duration_sec", None)
            and not getattr(exercise, "_target_distance_m", None)
        )
    )
    allocation_conserved = (
        fill_working_ids == final_working_ids
        and fill_working_sets == final_working_sets
    )
    if selection_diagnostics is not None:
        selection_diagnostics.update({
            "role_set_capacity_exhausted": bool(
                time_fill_diagnostics.get("role_set_capacity_exhausted", False)
                and allocation_conserved
                and final_main_count >= routine_blueprint.minimum_main_exercise_count
            ),
            "role_set_capacity_total_budget_exhausted": bool(
                time_fill_diagnostics.get("role_set_capacity_total_budget_exhausted", False)
                or set_cap_enforced
            ),
            "role_set_capacity_eligible_count": int(
                time_fill_diagnostics.get("role_set_capacity_eligible_count", 0)
            ),
            "role_set_capacity_restricted_working_role": bool(
                time_fill_diagnostics.get(
                    "role_set_capacity_restricted_working_role", False
                )
            ),
            "role_set_capacity_unmeasured_working_role": bool(
                time_fill_diagnostics.get(
                    "role_set_capacity_unmeasured_working_role", False
                )
            ),
        })
    _debug_print_llm_output(output_label, deterministic)
    execution_metadata = dict(deterministic._execution_metadata_by_exercise_id or {})
    if estimate_routine_time_min(
        deterministic.exercises,
        execution_metadata=execution_metadata,
    ) > req.time_available_min:
        logger.warning("[AI post-deterministic] routine exceeds time budget -> trying trim")
        trimmed_to_time_budget = trim_routine_to_time_budget(
            deterministic,
            req.time_available_min,
            protected_exercise_ids=protected_exercise_ids,
            execution_metadata=execution_metadata,
            adjunct_option_by_exercise_id=dict(
                deterministic._adjunct_option_by_exercise_id or {}
            ),
            minimum_main_exercise_count=(
                routine_blueprint.minimum_main_exercise_count
            ),
        )
        if not trimmed_to_time_budget:
            logger.warning("[AI post-deterministic] trim failed -> fallback")
            if on_time_budget_exceeded is not None:
                on_time_budget_exceeded()
            if selection_diagnostics is not None:
                selection_diagnostics.clear()
            return _fallback(
                req,
                ranked_candidates,
                max_total_sets,
                doms_db,
                goal,
                "schemaError",
                recent_sets,
                profile,
            )
        if selection_diagnostics is not None:
            # The fill proof describes the pre-trim allocation only.
            selection_diagnostics["role_set_capacity_exhausted"] = False
    if routine_blueprint.execution_policy.structure == "standard":
        deterministic = order_routine_output_by_selection_phase(
            deterministic, group_primary_muscles=True,
        )
    return _finalize_success(deterministic, draft_label, ranked_candidates)


def _validate_or_fallback(
    output: LLMRoutineOutput,
    req: RoutineRequest,
    ranked_candidates: List[dict],
    max_total_sets: int,
    doms_db: Dict[str, int],
    goal: str,
    db_target_muscles: List[str],
    recent_sets: Optional[List[RecentSetRecord]],
    profile: Optional[UserProfileContext],
    normalized: bool = False,
    on_validation_failed: Optional[Callable[[], None]] = None,
    on_time_budget_exceeded: Optional[Callable[[], None]] = None,
    routine_blueprint: Optional[RoutineBlueprint] = None,
    selection_diagnostics: Optional[dict[str, Any]] = None,
) -> RoutineDraftResponse:
    from .llm_parser import validate_and_repair_routine_output

    output = apply_requested_adjunct_prescription(
        output,
        (
            routine_blueprint.planned_adjunct_options
            if routine_blueprint is not None
            else req.adjunct_options
        ),
        ranked_candidates,
    )
    validated, repair_reason = validate_and_repair_routine_output(
        output,
        ranked_candidates,
        req.equipment,
        req.pain_areas,
        doms_db=doms_db,
        max_total_sets=max_total_sets,
        time_available_min=req.time_available_min,
        minimum_main_exercise_count=(
            routine_blueprint.minimum_main_exercise_count
            if routine_blueprint is not None
            else 1
        ),
        shoulder_condition_status=req.shoulder_condition_status,
        restricted_movement_patterns=req.restricted_movement_patterns,
        injury_conditions=req.injury_conditions,
        specialization_focus=(
            routine_blueprint.specialization_focus
            if routine_blueprint is not None
            else None
        ),
        allow_reviewed_pull_pair=(
            routine_blueprint is not None
            and routine_blueprint.formula_version == "gym-routine-blueprint-v11"
        ),
        experience_level=(routine_blueprint.experience_level if routine_blueprint is not None
            else req.experience_level or (profile.experience_level if profile else None)),
        adjunct_options=list(routine_blueprint.planned_adjunct_options
            if routine_blueprint is not None else req.adjunct_options or []),
    )
    if validated is None:
        if on_validation_failed is not None:
            on_validation_failed()
        logger.warning("[AI post-validation] output unrecoverable (reason=%s) -> fallback", repair_reason)
        return _fallback(req, ranked_candidates, max_total_sets, doms_db, goal, repair_reason, recent_sets, profile)
    return _deterministic_or_fallback(
        validated,
        req,
        ranked_candidates,
        max_total_sets,
        doms_db,
        goal,
        db_target_muscles,
        recent_sets,
        profile,
        "NORMALIZED DETERMINISTIC OUTPUT" if normalized else "DETERMINISTIC OUTPUT",
        "FINAL NORMALIZED DRAFT RESPONSE" if normalized else "FINAL DRAFT RESPONSE",
        on_time_budget_exceeded=on_time_budget_exceeded,
        routine_blueprint=routine_blueprint,
        selection_diagnostics=selection_diagnostics,
    )


def generate_smart_routine(
    req: RoutineRequest,
    db: Session,
    profile: Optional[UserProfileContext] = None,
    recent_sets: Optional[List[RecentSetRecord]] = None,
    local_constraint_profile: Optional[dict[str, Any]] = None,
    local_constraint_profile_is_resolved: bool = False,
) -> RoutineDraftResponse:
    merged_pain_areas = _merge_profile_injury_areas(
        req.pain_areas,
        profile.pain_areas if profile else None,
    )
    if merged_pain_areas != req.pain_areas:
        req = req.model_copy(update={"pain_areas": merged_pain_areas})

    resolved_injury_conditions = resolve_injury_condition_contexts(
        req.injury_conditions,
        profile.injury_conditions if profile else (),
        legacy_shoulder_contexts=(
            (req.shoulder_condition_status, req.restricted_movement_patterns),
            (
                profile.shoulder_condition_status if profile else None,
                profile.restricted_movement_patterns if profile else (),
            ),
        ),
    )
    resolved_condition_models = [
        InjuryConditionEntry(
            area_code=condition.area_code,
            status=condition.status,
            restricted_movement_patterns=list(
                condition.restricted_movement_patterns
            ),
        )
        for condition in resolved_injury_conditions
    ]
    shoulder_condition = next(
        (
            condition
            for condition in resolved_injury_conditions
            if condition.area_code == "shoulder"
        ),
        None,
    )
    shoulder_condition_status = (
        shoulder_condition.status if shoulder_condition else None
    )
    restricted_movement_patterns = (
        shoulder_condition.restricted_movement_patterns
        if shoulder_condition
        else ()
    )
    if (
        resolved_condition_models != req.injury_conditions
        or shoulder_condition_status != req.shoulder_condition_status
        or list(restricted_movement_patterns) != req.restricted_movement_patterns
    ):
        req = req.model_copy(update={
            "injury_conditions": resolved_condition_models,
            "shoulder_condition_status": shoulder_condition_status,
            "restricted_movement_patterns": list(restricted_movement_patterns),
        })

    db_target_muscles = resolve_request_target_muscles(
        req.target_split_label,
        req.target_muscles,
    )
    effective_target_split_label = resolve_request_target_split_label(
        req.target_split_label,
        req.target_muscles,
    )
    if effective_target_split_label != req.target_split_label:
        logger.info(
            "[매핑] split_label=%s -> effective_split_label=%s",
            req.target_split_label,
            effective_target_split_label,
        )
        req = req.model_copy(update={
            "target_split_label": effective_target_split_label,
        })
    if db_target_muscles:
        logger.info(
            "[매핑] split_label=%s explicit_targets=%s -> %s",
            req.target_split_label,
            req.target_muscles,
            db_target_muscles,
        )
    else:
        logger.info("[매핑] 타겟 근육 없음 -> 빈 후보 리스트로 진행")

    doms_db = req.doms_data
    goal = req.goal or (profile.goal_type if profile else "hypertrophy")
    pain_areas = req.pain_areas
    if local_constraint_profile is None:
        local_constraint_profile = get_user_constraint_profile_from_sqlite(req.user_id)
    mobility_limits = (
        list(req.mobility_limits or [])
        if local_constraint_profile_is_resolved
        else _merge_unique_strings(
            req.mobility_limits,
            local_constraint_profile.get("mobility_limits", []),
        )
    )
    condition_policies = list(req.condition_policies or [])
    if not local_constraint_profile_is_resolved:
        condition_policies.extend(
            local_constraint_profile.get("condition_policies", []) or []
        )
    anthropometry_signals = (
        dict(req.anthropometry_signals or {})
        if local_constraint_profile_is_resolved
        else {
            **(local_constraint_profile.get("anthropometry_signals", {}) or {}),
            **(req.anthropometry_signals or {}),
        }
    )
    logger.info("[매핑] doms -> %s", doms_db)

    if active_uncleared_requires_safe_stop(
        req.shoulder_condition_status,
        req.target_split_label,
        db_target_muscles,
    ):
        logger.warning("[Safety] active uncleared shoulder target -> failed emptyCandidate")
        response = _active_uncleared_shoulder_failed_response(list(req.adjunct_options))
        response = _apply_deterministic_request_warnings(response, condition_policies)
        response = _apply_shoulder_condition_warnings(response, req)
        return _apply_injury_condition_warnings(response, req)

    if active_uncleared_neck_requires_safe_stop(
        req.injury_conditions,
        db_target_muscles,
    ):
        logger.warning("[Safety] active uncleared neck upper target -> failed emptyCandidate")
        response = _active_uncleared_neck_failed_response(list(req.adjunct_options))
        response = _apply_deterministic_request_warnings(response, condition_policies)
        return _apply_injury_condition_warnings(response, req)

    if _should_block_broad_injury_context(req, db_target_muscles):
        logger.warning("[Safety] broad injury context -> failed emptyCandidate")
        return _apply_deterministic_request_warnings(
            _broad_injury_failed_response(list(req.adjunct_options)),
            condition_policies,
        )

    max_total_sets = _calculate_max_total_sets(req.time_available_min, goal)
    try:
        routine_blueprint = build_routine_blueprint(
            target_split_label=req.target_split_label,
            target_muscles=db_target_muscles,
            goal=goal,
            experience_level=req.experience_level or (profile.experience_level if profile else None),
            advanced_chest_focus=req.advanced_chest_focus,
            specialization_focus=req.specialization_focus,
            training_method=req.training_method,
            method_mode=req.method_mode,
            requested_method=req.requested_method,
            recent_training_method_uses=req.recent_training_method_uses,
            readiness_level=req.readiness_level,
            time_available_min=req.time_available_min,
            training_style=req.training_style,
            session_structure=req.session_structure,
            adjunct_options=req.adjunct_options,
            pain_areas=req.pain_areas,
            doms_data=req.doms_data,
            injury_conditions=req.injury_conditions,
        )
    except RoutineBlueprintValidationError as exc:
        if exc.errors != ("set_budget_below_required_slots",):
            raise
        logger.warning("[Blueprint] insufficient time for required roles")
        return RoutineDraftResponse(
            generation_status="failed", status_reason_code="schemaError", is_fallback=False,
            total_estimated_time=0, summary_title="운동 시간을 조정해 주세요",
            rationale_summary=[
                f"선택한 부위의 필수 운동 구성을 {req.time_available_min}분 안에 배치할 수 없습니다."
            ],
            routine_blocks=[],
            warnings=["운동 시간을 늘리거나 운동 부위를 줄여 다시 생성해 주세요."],
        )
    target_exercise_count = routine_blueprint.target_exercise_count
    candidates = get_candidate_exercises(
        db,
        db_target_muscles,
        req.equipment,
        pain_areas,
        req.available_equipment,
    )
    from .glute_support import get_glute_support_candidates

    glute_support_candidates = get_glute_support_candidates(
        db, routine_blueprint, req.equipment, pain_areas, req.available_equipment,
    )
    candidates = _merge_candidate_sources(candidates, glute_support_candidates)
    adjunct_candidates = get_adjunct_candidate_exercises(
        list(routine_blueprint.planned_adjunct_options),
        req.equipment,
        pain_areas,
        req.available_equipment,
    )
    candidates = _merge_candidate_sources(candidates, adjunct_candidates)
    candidate_pool_size = get_candidate_pool_size()
    from .feedback_aggregation import get_feedback_aware_ranking_enabled

    feedback_enabled = get_feedback_aware_ranking_enabled()
    feedback_query_latency_ms: Optional[int] = None
    feedback_query_failed = False
    feedback_query_error_category: Optional[str] = None
    feedback_started = time.perf_counter()
    feedback_adjustments, feedback_query_failed, feedback_query_error_category = _get_feedback_adjustments_with_status(
        db,
        req.user_id,
        candidates,
    )
    if feedback_enabled:
        feedback_query_latency_ms = round((time.perf_counter() - feedback_started) * 1000)
    ranked_candidates = score_candidate_exercises(
        candidates,
        db_target_muscles,
        doms_db=doms_db,
        blocked_equipment=req.equipment,
        available_equipment=req.available_equipment,
        pain_areas=pain_areas,
        recent_sets=recent_sets,
        preferred_exercise_ids=req.preferred_exercise_ids,
        unpreferred_exercise_ids=req.unpreferred_exercise_ids,
        top_n=candidate_pool_size,
        feedback_adjustments=feedback_adjustments,
        readiness_level=req.readiness_level,
        goal=goal,
        mobility_limits=mobility_limits,
        condition_policies=condition_policies,
        anthropometry_signals=anthropometry_signals,
        experience_level=req.experience_level or (profile.experience_level if profile else None),
        advanced_chest_focus=req.advanced_chest_focus,
        specialization_focus=req.specialization_focus,
        training_method=routine_blueprint.training_method,
        training_style=req.training_style or "gym",
        target_split_label=req.target_split_label,
        session_structure=req.session_structure,
        adjunct_options=list(routine_blueprint.planned_adjunct_options),
        time_available_min=req.time_available_min,
        shoulder_condition_status=req.shoulder_condition_status,
        restricted_movement_patterns=req.restricted_movement_patterns,
        injury_conditions=req.injury_conditions,
    )
    feedback_stats = _compute_feedback_stats(ranked_candidates, feedback_adjustments)
    feedback_adjusted_candidate_count = feedback_stats["adjusted"]
    feedback_positive_count = feedback_stats["positive"]
    feedback_negative_count = feedback_stats["negative"]
    feedback_abs_adjustment_avg = feedback_stats["abs_avg"]
    max_total_sets = _candidate_role_set_budget(req, routine_blueprint, ranked_candidates, doms_db)
    routine_blueprint = replace(routine_blueprint, max_total_sets=max_total_sets)
    slot_candidate_bundles = build_blueprint_candidate_bundles(
        routine_blueprint,
        ranked_candidates,
    )
    slot_candidate_bundles = align_candidate_bundles_with_composition(
        routine_blueprint, slot_candidate_bundles, ranked_candidates,
    )
    prompt_slot_candidate_bundles = [
        bundle for bundle in slot_candidate_bundles if bundle.get("slotId") != "flex"
    ]
    candidate_payload = _format_slot_candidate_details(
        slot_candidate_bundles,
        ranked_candidates,
    )
    accessory_plan = render_accessory_group_plan(
        routine_blueprint, slot_candidate_bundles, ranked_candidates,
    )
    if accessory_plan:
        candidate_payload += "\n\n" + accessory_plan
    label_map = {
        1: "약간 근육통 (가벼운 볼륨 감소)",
        2: "강한 근육통 (서버가 경력·역할별 분량을 줄이고 RIR4 이상 적용, 고급 수행법 제외)",
        3: "매우 강한 근육 피로 (해당 부위 운동 제외)",
    }
    doms_instructions = (
        "\n".join(f"- {part}: {label_map.get(level, '')}" for part, level in doms_db.items())
        if doms_db
        else "현재 근육통 없음. 정상 볼륨으로 진행."
    )
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            build_compiled_system_prompt(
                profile,
                recent_sets,
                has_doms=bool(doms_db),
                has_medical_context=bool(
                    pain_areas
                    or condition_policies
                    or req.injury_conditions
                    or req.shoulder_condition_status
                ),
                readiness_level=req.readiness_level,
                training_style=req.training_style,
                session_structure=req.session_structure,
                has_adjunct_options=bool(req.adjunct_options),
            ),
        ),
        ("human", "User note: {user_note}\nUse only the request context and slot candidate details above."),
    ])
    invoke_kwargs = {
        "goal": goal,
        "max_sets": max_total_sets,
        "doms_instructions": doms_instructions,
        "candidate_exercises": candidate_payload,
        "user_note": req.user_note or "없음",
        "time_available_min": req.time_available_min,
        "target_exercise_count": target_exercise_count,
        "readiness_level": req.readiness_level or "normal",
        "experience_level": req.experience_level or (profile.experience_level if profile else "none"),
        "training_style": req.training_style or "gym",
        "session_structure": req.session_structure or "standard",
        "adjunct_options": ", ".join(routine_blueprint.planned_adjunct_options)
        if routine_blueprint.planned_adjunct_options else "none",
        "available_equipment": ", ".join(req.available_equipment) if req.available_equipment else "none",
        "unavailable_equipment": ", ".join(req.equipment) if req.equipment else "none",
        "target_split_label": req.target_split_label or "none",
        "target_muscles": ", ".join(db_target_muscles) if db_target_muscles else "none",
        "current_pain_areas": _format_request_pain_areas(pain_areas),
        "injury_conditions": format_injury_conditions_for_prompt(
            req.injury_conditions
        ),
        "shoulder_condition_status": req.shoulder_condition_status or "none",
        "restricted_movement_patterns": ", ".join(req.restricted_movement_patterns)
        if req.restricted_movement_patterns else "none",
        "mobility_limits": ", ".join(mobility_limits) if mobility_limits else "no_known_limit_reported",
        "condition_policies": _format_prompt_mapping(condition_policies),
        "anthropometry_signals": _format_prompt_mapping(anthropometry_signals),
        "candidate_count": len(ranked_candidates),
        "routine_blueprint": render_blueprint_for_prompt(routine_blueprint),
        "slot_candidate_bundles": render_candidate_bundles_for_prompt(prompt_slot_candidate_bundles),
    }
    _debug_print_prompt(prompt, invoke_kwargs)
    prompt_char_count = _rendered_prompt_char_count(prompt, invoke_kwargs)
    prompt_approx_tokens = (prompt_char_count + 3) // 4 if prompt_char_count else 0
    local_llm_num_predict = _safe_positive_int_env("LOCAL_LLM_NUM_PREDICT")
    local_llm_num_ctx = _safe_positive_int_env("LOCAL_LLM_NUM_CTX")

    reason: StatusReasonCode = "networkError"
    generation_temp = resolve_generation_temperature(req.readiness_level)
    generation_latency_ms: Optional[int] = None
    schema_repair_latency_ms: Optional[int] = None
    llm_error_type: Optional[str] = None
    schema_repair_attempted = False
    schema_repair_succeeded = False
    parse_failure_subtype: Optional[str] = None
    schema_validation_error_category: Optional[str] = None
    schema_validation_field_names: list[str] = []
    repair_failure_reason: Optional[str] = None
    raw_output_recovery_attempted = False
    raw_output_recovery_succeeded = False
    raw_output_recovery_source: Optional[str] = None
    raw_output_recovery_failed_reason: Optional[str] = None
    json_decode_error_category: Optional[str] = None
    json_decode_recovery_attempted = False
    json_decode_recovery_succeeded = False
    json_decode_recovery_strategy: Optional[str] = None
    schema_root_type: Optional[str] = None
    schema_top_level_keys: list[str] = []
    schema_exercises_value_type: Optional[str] = None
    schema_exercises_nested_keys: list[str] = []
    schema_exercises_list_length: Optional[int] = None
    schema_first_exercise_item_type: Optional[str] = None
    schema_first_exercise_keys: list[str] = []
    schema_first_exercise_field_types: dict[str, str] = {}
    schema_exercise_item_types: list[str] = []
    schema_exercise_key_union: list[str] = []
    schema_exercise_field_type_sets: dict[str, list[str]] = {}
    schema_validation_loc_summaries: list[str] = []
    schema_validation_error_types: list[str] = []
    local_raw_json_invoke_enabled = _local_raw_json_invoke_flag_enabled()
    local_raw_json_invoke_used = False
    local_raw_json_invoke_succeeded = False
    local_raw_json_invoke_failed_reason: Optional[str] = None
    structured_output_bypassed = False
    raw_invoke_response_class: Optional[str] = None
    raw_invoke_content_present = False
    raw_invoke_content_type: Optional[str] = None
    raw_invoke_content_length_bucket: Optional[str] = None
    raw_invoke_content_stripped_empty = False
    raw_invoke_has_response_metadata = False
    raw_invoke_finish_reason: Optional[str] = None
    raw_invoke_done_reason: Optional[str] = None
    raw_invoke_error_category: Optional[str] = None
    raw_invoke_usage_present = False
    selection_diagnostics: dict[str, Any] = {}
    validation_failed = False
    time_budget_exceeded = False

    def _mark_validation_failed() -> None:
        nonlocal validation_failed
        validation_failed = True

    def _mark_time_budget_exceeded() -> None:
        nonlocal time_budget_exceeded
        time_budget_exceeded = True

    def _observe(response: RoutineDraftResponse) -> RoutineDraftResponse:
        fallback_selection_diagnostics = dict(
            getattr(response, "_fallback_selection_diagnostics", {}) or {}
        )
        if response.is_fallback or response.generation_status == "fallback":
            # A synthesized fallback has no accepted LLM selections. Replace,
            # rather than merge, failed-model diagnostics before telemetry.
            selection_diagnostics.clear()
            selection_diagnostics.update(fallback_selection_diagnostics)
        response = _apply_deterministic_request_warnings(response, condition_policies)
        response = _apply_shoulder_condition_warnings(response, req)
        response = _apply_injury_condition_warnings(response, req)
        response = _apply_pain_and_doms_warnings(
            response,
            req,
            ranked_candidates,
            db_target_muscles,
        )
        # Safety post-processing may remove an applied adjunct block. Build the
        # public ledger only after that final block mutation has completed.
        response = _apply_adjunct_application_warnings(
            response,
            routine_blueprint,
            ranked_candidates,
        )
        response = _enforce_experience_rationale_consistency(
            response,
            req.experience_level or (profile.experience_level if profile else None),
        )
        response = _enforce_response_fact_consistency(response, req)
        response = _apply_selected_mobility_demand_warnings(
            response,
            req,
            mobility_limits=mobility_limits,
            ranked_candidates=ranked_candidates,
        )
        response = _apply_active_context_rationale(
            response,
            req,
            mobility_limits=mobility_limits,
            experience_level=req.experience_level or (profile.experience_level if profile else None),
            ranked_candidates=ranked_candidates,
        )
        response = separate_doms_from_pain_wording_on_response(
            response,
            doms_data=req.doms_data,
            pain_areas=req.pain_areas,
            trusted_reasons_by_exercise_id={
                str(candidate.get("id") or "").strip().lower(): {
                    str(reason)
                    for reason in (candidate.get("score_display_reasons") or [])
                    if isinstance(reason, str) and reason
                }
                for candidate in ranked_candidates
                if str(candidate.get("id") or "").strip()
            },
        )
        response = _enforce_same_area_injury_precedence_wording(response, req)
        response = _apply_underfill_explanation(
            response,
            req,
            ranked_candidates,
            routine_blueprint,
            selection_diagnostics,
        )
        history_suggestion = suggest_specialization_from_recent_sets(
            recent_sets or [],
            target_muscles=db_target_muscles,
            experience_level=(
                req.experience_level
                or (profile.experience_level if profile else None)
            ),
            explicit_focus=req.specialization_focus,
        )
        if history_suggestion and response.routine_blocks:
            response.personalization_suggestions = [
                PersonalizationSuggestion(
                    suggestion_type="specializationFocus",
                    code=history_suggestion.focus_code,
                    reason_ko=history_suggestion.reason_ko,
                    evidence_set_count=history_suggestion.evidence_set_count,
                    comparison_max_set_count=(
                        history_suggestion.comparison_max_set_count
                    ),
                    requires_confirmation=True,
                )
            ]
        fallback_used = response.is_fallback or response.generation_status == "fallback"
        fallback_reason = classify_fallback_reason(
            fallback_used=fallback_used,
            llm_error_type=llm_error_type,
            schema_repair_attempted=schema_repair_attempted,
            schema_repair_succeeded=schema_repair_succeeded,
            validation_failed=validation_failed,
            time_budget_exceeded=time_budget_exceeded,
        )
        return observe_routine_quality(
            response=response,
            req=req,
            ranked_candidates=ranked_candidates,
            db_target_muscles=db_target_muscles,
            recent_sets=recent_sets,
            max_total_sets=max_total_sets,
            candidate_pool_size=candidate_pool_size,
            candidate_payload=candidate_payload,
            prompt_char_count=prompt_char_count,
            prompt_approx_tokens=prompt_approx_tokens,
            generation_temperature=generation_temp,
            local_llm_num_predict=local_llm_num_predict,
            local_llm_num_ctx=local_llm_num_ctx,
            generation_latency_ms=generation_latency_ms,
            schema_repair_latency_ms=schema_repair_latency_ms,
            fallback_reason=fallback_reason,
            llm_error_type=llm_error_type,
            schema_repair_attempted=schema_repair_attempted,
            schema_repair_succeeded=schema_repair_succeeded,
            parse_failure_subtype=parse_failure_subtype,
            schema_validation_error_category=schema_validation_error_category,
            schema_validation_field_names=schema_validation_field_names,
            repair_failure_reason=repair_failure_reason,
            raw_output_recovery_attempted=raw_output_recovery_attempted,
            raw_output_recovery_succeeded=raw_output_recovery_succeeded,
            raw_output_recovery_source=raw_output_recovery_source,
            raw_output_recovery_failed_reason=raw_output_recovery_failed_reason,
            json_decode_error_category=json_decode_error_category,
            json_decode_recovery_attempted=json_decode_recovery_attempted,
            json_decode_recovery_succeeded=json_decode_recovery_succeeded,
            json_decode_recovery_strategy=json_decode_recovery_strategy,
            schema_root_type=schema_root_type,
            schema_top_level_keys=schema_top_level_keys,
            schema_exercises_value_type=schema_exercises_value_type,
            schema_exercises_nested_keys=schema_exercises_nested_keys,
            schema_exercises_list_length=schema_exercises_list_length,
            schema_first_exercise_item_type=schema_first_exercise_item_type,
            schema_first_exercise_keys=schema_first_exercise_keys,
            schema_first_exercise_field_types=schema_first_exercise_field_types,
            schema_exercise_item_types=schema_exercise_item_types,
            schema_exercise_key_union=schema_exercise_key_union,
            schema_exercise_field_type_sets=schema_exercise_field_type_sets,
            schema_validation_loc_summaries=schema_validation_loc_summaries,
            schema_validation_error_types=schema_validation_error_types,
            local_raw_json_invoke_enabled=local_raw_json_invoke_enabled,
            local_raw_json_invoke_used=local_raw_json_invoke_used,
            local_raw_json_invoke_succeeded=local_raw_json_invoke_succeeded,
            local_raw_json_invoke_failed_reason=local_raw_json_invoke_failed_reason,
            structured_output_bypassed=structured_output_bypassed,
            raw_invoke_response_class=raw_invoke_response_class,
            raw_invoke_content_present=raw_invoke_content_present,
            raw_invoke_content_type=raw_invoke_content_type,
            raw_invoke_content_length_bucket=raw_invoke_content_length_bucket,
            raw_invoke_content_stripped_empty=raw_invoke_content_stripped_empty,
            raw_invoke_has_response_metadata=raw_invoke_has_response_metadata,
            raw_invoke_finish_reason=raw_invoke_finish_reason,
            raw_invoke_done_reason=raw_invoke_done_reason,
            raw_invoke_error_category=raw_invoke_error_category,
            raw_invoke_usage_present=raw_invoke_usage_present,
            feedback_enabled=feedback_enabled,
            feedback_adjusted_candidate_count=feedback_adjusted_candidate_count,
            feedback_positive_count=feedback_positive_count,
            feedback_negative_count=feedback_negative_count,
            feedback_abs_adjustment_avg=feedback_abs_adjustment_avg,
            feedback_query_latency_ms=feedback_query_latency_ms,
            feedback_query_failed=feedback_query_failed,
            feedback_query_error_category=feedback_query_error_category,
            selection_diagnostics=selection_diagnostics,
        )

    if not ranked_candidates:
        logger.warning("[Routine] no eligible ranked candidates -> failed emptyCandidate without LLM invoke")
        return _observe(_fallback(
            req, ranked_candidates, max_total_sets, doms_db, goal,
            "emptyCandidate", recent_sets, profile,
        ))

    if not candidates_have_required_main_pattern(routine_blueprint, ranked_candidates):
        selection_diagnostics.update({
            "required_main_pattern_missing": True,
            "required_main_pattern_targets": list(required_main_pattern_targets(routine_blueprint)),
        })
        logger.info("[Routine] required main unavailable -> emptyCandidate without LLM invoke")
        return _observe(_required_main_pattern_failed_response(routine_blueprint))

    try:
        llm = get_llm("routine", temperature=generation_temp)
        use_local_raw_json_invoke = _should_use_local_raw_json_invoke()
        generation_started = time.perf_counter()
        try:
            if use_local_raw_json_invoke:
                local_raw_json_invoke_used = True
                structured_output_bypassed = True
                llm = _bind_local_selection_decoder(llm, slot_candidate_bundles)
                try:
                    raw_resp = (prompt | llm).invoke(invoke_kwargs)
                    raw_text, raw_shape = _extract_llm_message_content_with_shape(raw_resp)
                    raw_invoke_response_class = raw_shape["raw_invoke_response_class"]
                    raw_invoke_content_present = raw_shape["raw_invoke_content_present"]
                    raw_invoke_content_type = raw_shape["raw_invoke_content_type"]
                    raw_invoke_content_length_bucket = raw_shape["raw_invoke_content_length_bucket"]
                    raw_invoke_content_stripped_empty = raw_shape["raw_invoke_content_stripped_empty"]
                    raw_invoke_has_response_metadata = raw_shape["raw_invoke_has_response_metadata"]
                    raw_invoke_finish_reason = raw_shape["raw_invoke_finish_reason"]
                    raw_invoke_done_reason = raw_shape["raw_invoke_done_reason"]
                    raw_invoke_error_category = raw_shape["raw_invoke_error_category"]
                    raw_invoke_usage_present = raw_shape["raw_invoke_usage_present"]
                    response = normalize_routine_selection_response(
                        raw_text,
                        blueprint=routine_blueprint,
                        slot_candidate_bundles=slot_candidate_bundles,
                        ranked_candidates=ranked_candidates,
                    )
                    selection_diagnostics = dict(
                        getattr(response, "_selection_diagnostics", {}) or {}
                    )
                    local_raw_json_invoke_succeeded = True
                except SchemaRepairError as raw_e:
                    local_raw_json_invoke_failed_reason = raw_e.repair_failure_reason or raw_e.parse_failure_subtype
                    parse_failure_subtype = raw_e.parse_failure_subtype or parse_failure_subtype
                    schema_validation_error_category = (
                        raw_e.schema_validation_error_category or schema_validation_error_category
                    )
                    schema_validation_field_names = (
                        raw_e.schema_validation_field_names or schema_validation_field_names
                    )
                    repair_failure_reason = raw_e.repair_failure_reason
                    json_decode_error_category = raw_e.json_decode_error_category or json_decode_error_category
                    json_decode_recovery_attempted = (
                        raw_e.json_decode_recovery_attempted or json_decode_recovery_attempted
                    )
                    json_decode_recovery_succeeded = (
                        raw_e.json_decode_recovery_succeeded or json_decode_recovery_succeeded
                    )
                    json_decode_recovery_strategy = raw_e.json_decode_recovery_strategy or json_decode_recovery_strategy
                    schema_root_type = raw_e.schema_root_type or schema_root_type
                    schema_top_level_keys = raw_e.schema_top_level_keys or schema_top_level_keys
                    schema_exercises_value_type = (
                        raw_e.schema_exercises_value_type or schema_exercises_value_type
                    )
                    schema_exercises_nested_keys = (
                        raw_e.schema_exercises_nested_keys or schema_exercises_nested_keys
                    )
                    schema_exercises_list_length = (
                        raw_e.schema_exercises_list_length
                        if raw_e.schema_exercises_list_length is not None
                        else schema_exercises_list_length
                    )
                    schema_first_exercise_item_type = (
                        raw_e.schema_first_exercise_item_type or schema_first_exercise_item_type
                    )
                    schema_first_exercise_keys = raw_e.schema_first_exercise_keys or schema_first_exercise_keys
                    schema_first_exercise_field_types = (
                        raw_e.schema_first_exercise_field_types or schema_first_exercise_field_types
                    )
                    schema_exercise_item_types = raw_e.schema_exercise_item_types or schema_exercise_item_types
                    schema_exercise_key_union = raw_e.schema_exercise_key_union or schema_exercise_key_union
                    schema_exercise_field_type_sets = (
                        raw_e.schema_exercise_field_type_sets or schema_exercise_field_type_sets
                    )
                    schema_validation_loc_summaries = (
                        raw_e.schema_validation_loc_summaries or schema_validation_loc_summaries
                    )
                    schema_validation_error_types = (
                        raw_e.schema_validation_error_types or schema_validation_error_types
                    )
                    raise OutputParserException("local raw JSON invoke failed") from raw_e
            else:
                selection_response = (
                    prompt | llm.with_structured_output(LLMRoutineSelectionOutput)
                ).invoke(invoke_kwargs)
                response = adapt_structured_routine_response(
                    selection_response,
                    blueprint=routine_blueprint,
                    slot_candidate_bundles=slot_candidate_bundles,
                    ranked_candidates=ranked_candidates,
                )
                selection_diagnostics = dict(
                    getattr(response, "_selection_diagnostics", {}) or {}
                )
        finally:
            generation_latency_ms = round((time.perf_counter() - generation_started) * 1000)
        _debug_print_llm_output("LLM STRUCTURED OUTPUT", response)
        draft = _validate_or_fallback(
            response,
            req,
            ranked_candidates,
            max_total_sets,
            doms_db,
            goal,
            db_target_muscles,
            recent_sets,
            profile,
            on_validation_failed=_mark_validation_failed,
            on_time_budget_exceeded=_mark_time_budget_exceeded,
            routine_blueprint=routine_blueprint,
            selection_diagnostics=selection_diagnostics,
        )
        return _observe(draft)
    except (asyncio.TimeoutError, httpx.TimeoutException) as e:
        llm_error_type = classify_llm_error(e)
        reason = map_llm_error(e)
        logger.warning("[AI 실패 - TIMEOUT] %s -> fallback 루틴으로 전환", sanitize_exception_for_log(e))
    except (OutputParserException, ValidationError) as e:
        llm_error_type = classify_llm_error(e)
        schema_repair_attempted = True
        if isinstance(e, ValidationError):
            schema_validation_error_category, schema_validation_field_names = classify_validation_error(e)
            schema_validation_loc_summaries, schema_validation_error_types = summarize_validation_error_shapes(e)
            parse_failure_subtype = "pydantic_validation_error"
        elif parse_failure_subtype is None:
            original_raw_text, _original_source = extract_raw_llm_output_from_exception(e)
            parse_failure_subtype = parse_json_candidate(original_raw_text or "").subtype
        logger.warning("[AI - SCHEMA 오류] 정제 어댑터로 복구 시도... (%s)", type(e).__name__)
        try:
            repair_llm = get_llm("routine", temperature=0)
            if _should_use_local_raw_json_invoke():
                repair_llm = _bind_local_selection_decoder(repair_llm, slot_candidate_bundles)
            raw_output_recovery_attempted = True
            raw_text, raw_output_recovery_source = extract_raw_llm_output_from_exception(e)
            if raw_text:
                raw_output_recovery_succeeded = True
            else:
                raw_output_recovery_failed_reason = "no_raw_content"
                logger.warning("[정제 어댑터] raw 텍스트 없음 -> LLM 비구조화 재호출")
                raw_resp = (prompt | repair_llm).invoke(invoke_kwargs)
                raw_text = raw_resp.content if hasattr(raw_resp, "content") else str(raw_resp)
                if raw_text and raw_text.strip():
                    raw_output_recovery_succeeded = True
                    raw_output_recovery_source = "raw_retry"
                    raw_output_recovery_failed_reason = None
                else:
                    raw_output_recovery_source = "none"
                    raw_output_recovery_failed_reason = "no_raw_content"
                raw_summary = summarize_text_for_log(raw_text)
                logger.info(
                    "[정제 어댑터] raw LLM output 수신 %s",
                    {
                        "redacted": True,
                        "llm_output_char_count": raw_summary["char_count"],
                        "llm_output_approx_tokens": raw_summary["approx_tokens"],
                    },
                )
            raw_output_recovery_source = raw_output_recovery_source or "none"
            repair_started = time.perf_counter()
            try:
                pre_parse_result = parse_json_candidate(raw_text or "")
                json_decode_error_category = pre_parse_result.json_decode_error_category
                json_decode_recovery_attempted = pre_parse_result.json_decode_recovery_attempted
                json_decode_recovery_succeeded = pre_parse_result.json_decode_recovery_succeeded
                json_decode_recovery_strategy = pre_parse_result.json_decode_recovery_strategy
                normalized = normalize_routine_selection_response(
                    raw_text,
                    blueprint=routine_blueprint,
                    slot_candidate_bundles=slot_candidate_bundles,
                    ranked_candidates=ranked_candidates,
                    llm=repair_llm,
                )
                selection_diagnostics = dict(
                    getattr(normalized, "_selection_diagnostics", {}) or {}
                )
            finally:
                schema_repair_latency_ms = round((time.perf_counter() - repair_started) * 1000)
            _debug_print_llm_output("NORMALIZED LLM OUTPUT", normalized)
            draft = _validate_or_fallback(
                normalized,
                req,
                ranked_candidates,
                max_total_sets,
                doms_db,
                goal,
                db_target_muscles,
                recent_sets,
                profile,
                normalized=True,
                on_validation_failed=_mark_validation_failed,
                on_time_budget_exceeded=_mark_time_budget_exceeded,
                routine_blueprint=routine_blueprint,
                selection_diagnostics=selection_diagnostics,
            )
            if not validation_failed and not draft.is_fallback and draft.generation_status != "fallback":
                schema_repair_succeeded = True
            return _observe(draft)
        except Exception as norm_e:
            reason = map_llm_error(e)
            if isinstance(norm_e, SchemaRepairError):
                parse_failure_subtype = norm_e.parse_failure_subtype or parse_failure_subtype
                schema_validation_error_category = (
                    norm_e.schema_validation_error_category or schema_validation_error_category
                )
                schema_validation_field_names = norm_e.schema_validation_field_names or schema_validation_field_names
                repair_failure_reason = norm_e.repair_failure_reason
                json_decode_error_category = norm_e.json_decode_error_category or json_decode_error_category
                json_decode_recovery_attempted = (
                    norm_e.json_decode_recovery_attempted or json_decode_recovery_attempted
                )
                json_decode_recovery_succeeded = (
                    norm_e.json_decode_recovery_succeeded or json_decode_recovery_succeeded
                )
                json_decode_recovery_strategy = norm_e.json_decode_recovery_strategy or json_decode_recovery_strategy
                schema_root_type = norm_e.schema_root_type or schema_root_type
                schema_top_level_keys = norm_e.schema_top_level_keys or schema_top_level_keys
                schema_exercises_value_type = norm_e.schema_exercises_value_type or schema_exercises_value_type
                schema_exercises_nested_keys = (
                    norm_e.schema_exercises_nested_keys or schema_exercises_nested_keys
                )
                schema_exercises_list_length = (
                    norm_e.schema_exercises_list_length
                    if norm_e.schema_exercises_list_length is not None
                    else schema_exercises_list_length
                )
                schema_first_exercise_item_type = (
                    norm_e.schema_first_exercise_item_type or schema_first_exercise_item_type
                )
                schema_first_exercise_keys = norm_e.schema_first_exercise_keys or schema_first_exercise_keys
                schema_first_exercise_field_types = (
                    norm_e.schema_first_exercise_field_types or schema_first_exercise_field_types
                )
                schema_exercise_item_types = norm_e.schema_exercise_item_types or schema_exercise_item_types
                schema_exercise_key_union = norm_e.schema_exercise_key_union or schema_exercise_key_union
                schema_exercise_field_type_sets = (
                    norm_e.schema_exercise_field_type_sets or schema_exercise_field_type_sets
                )
                schema_validation_loc_summaries = (
                    norm_e.schema_validation_loc_summaries or schema_validation_loc_summaries
                )
                schema_validation_error_types = (
                    norm_e.schema_validation_error_types or schema_validation_error_types
                )
            else:
                repair_failure_reason = type(norm_e).__name__
            logger.warning("[정제 어댑터] 복구 실패: %s -> fallback 루틴으로 전환", sanitize_exception_for_log(norm_e))
    except Exception as e:
        llm_error_type = classify_llm_error(e)
        reason = map_llm_error(e)
        logger.warning("[AI 실패 - %s] %s -> fallback 루틴으로 전환", reason.upper(), sanitize_exception_for_log(e))

    draft = _fallback(req, ranked_candidates, max_total_sets, doms_db, goal, reason, recent_sets, profile)
    return _observe(draft)

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
from typing import Dict, List, Optional

import httpx
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError
from sqlalchemy.orm import Session

from .candidate_ranker import format_candidates_for_prompt, score_candidate_exercises
from .db_queries import get_candidate_exercises
from .fallback import (
    _debug_print_draft,
    _debug_print_llm_output,
    _debug_print_prompt,
    build_routine_draft,
    generate_fallback_routine,
)
from .llm_router import StatusReasonCode, get_llm, map_llm_error
from .llm_parser import normalize_llm_response, trim_routine_to_time_budget
from .muscle_mapping import get_mapped_targets, split_label_to_muscles
from .prescription.adjustments import _calculate_max_total_sets, _target_exercise_count
from .prescription.estimator import estimate_routine_time_min
from .prescription.targets import apply_deterministic_targets
from .prompt_builder import _format_request_pain_areas, build_system_prompt
from .schemas import LLMRoutineOutput, RecentSetRecord, RoutineDraftResponse, RoutineRequest, UserProfileContext


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
    return generate_fallback_routine(
        req,
        ranked_candidates,
        max_total_sets,
        doms_db,
        goal=goal,
        status_reason_code=status_reason_code,
        recent_sets=recent_sets,
        profile=profile,
    )


def _finalize_success(
    output: LLMRoutineOutput,
    label: str,
) -> RoutineDraftResponse:
    draft = build_routine_draft(output, "success", "none", False)
    _debug_print_draft(label, draft)
    return draft


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
) -> RoutineDraftResponse:
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
    )
    _debug_print_llm_output(output_label, deterministic)
    if estimate_routine_time_min(deterministic.exercises) > req.time_available_min:
        print("[AI post-deterministic] routine exceeds time budget -> trying trim")
        if not trim_routine_to_time_budget(deterministic, req.time_available_min):
            print("[AI post-deterministic] trim failed -> fallback")
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
    return _finalize_success(deterministic, draft_label)


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
) -> RoutineDraftResponse:
    from .llm_parser import validate_and_repair_routine_output

    validated, repair_reason = validate_and_repair_routine_output(
        output,
        ranked_candidates,
        req.equipment,
        req.pain_areas,
        doms_db=doms_db,
        max_total_sets=max_total_sets,
        time_available_min=req.time_available_min,
    )
    if validated is None:
        print(f"[AI post-validation] output unrecoverable (reason={repair_reason}) -> fallback")
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
    )


def generate_smart_routine(
    req: RoutineRequest,
    db: Session,
    profile: Optional[UserProfileContext] = None,
    recent_sets: Optional[List[RecentSetRecord]] = None,
) -> RoutineDraftResponse:
    if req.target_split_label:
        db_target_muscles = split_label_to_muscles(req.target_split_label)
        print(f"[매핑] split_label={req.target_split_label} -> {db_target_muscles}")
    elif req.target_muscles:
        _, db_target_muscles = get_mapped_targets(req.target_muscles)
        print(f"[매핑] DB 타겟 -> {db_target_muscles}")
    else:
        db_target_muscles = []
        print("[매핑] 타겟 근육 없음 -> 빈 후보 리스트로 진행")

    doms_db = req.doms_data
    goal = req.goal or (profile.goal_type if profile else "hypertrophy")
    pain_areas = req.pain_areas
    print(f"[매핑] doms -> {doms_db}")

    candidates = get_candidate_exercises(db, db_target_muscles, req.equipment, pain_areas)
    ranked_candidates = score_candidate_exercises(
        candidates,
        db_target_muscles,
        doms_db=doms_db,
        blocked_equipment=req.equipment,
        pain_areas=pain_areas,
        recent_sets=recent_sets,
        preferred_exercise_ids=req.preferred_exercise_ids,
        unpreferred_exercise_ids=req.unpreferred_exercise_ids,
    )
    max_total_sets = _calculate_max_total_sets(req.time_available_min, goal)
    target_exercise_count = _target_exercise_count(req.time_available_min)
    label_map = {
        1: "약간 근육통 (볼륨 20% 감소)",
        2: "매우 근육통 (가벼운 자극 1~2세트만)",
        3: "통증/부상 우려 (해당 부위 운동 금지)",
    }
    doms_instructions = (
        "\n".join(f"- {part}: {label_map.get(level, '')}" for part, level in doms_db.items())
        if doms_db
        else "현재 근육통 없음. 정상 볼륨으로 진행."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", build_system_prompt(profile, recent_sets)),
        ("human", "User note: {user_note}\nUse only the request context and ranked candidates above."),
    ])
    invoke_kwargs = {
        "goal": goal,
        "max_sets": max_total_sets,
        "doms_instructions": doms_instructions,
        "candidate_exercises": format_candidates_for_prompt(ranked_candidates),
        "user_note": req.user_note or "없음",
        "time_available_min": req.time_available_min,
        "target_exercise_count": target_exercise_count,
        "readiness_level": req.readiness_level or "normal",
        "unavailable_equipment": ", ".join(req.equipment) if req.equipment else "none",
        "target_split_label": req.target_split_label or "none",
        "target_muscles": ", ".join(db_target_muscles) if db_target_muscles else "none",
        "current_pain_areas": _format_request_pain_areas(pain_areas),
        "candidate_count": len(ranked_candidates),
    }
    _debug_print_prompt(prompt, invoke_kwargs)

    reason: StatusReasonCode = "networkError"
    try:
        llm = get_llm("routine")
        response: LLMRoutineOutput = (prompt | llm.with_structured_output(LLMRoutineOutput)).invoke(invoke_kwargs)
        _debug_print_llm_output("LLM STRUCTURED OUTPUT", response)
        return _validate_or_fallback(
            response, req, ranked_candidates, max_total_sets, doms_db, goal, db_target_muscles, recent_sets, profile
        )
    except (asyncio.TimeoutError, httpx.TimeoutException) as e:
        reason = map_llm_error(e)
        print(f"[AI 실패 - TIMEOUT] {e} -> fallback 루틴으로 전환")
    except (OutputParserException, ValidationError) as e:
        print(f"[AI - SCHEMA 오류] 정제 어댑터로 복구 시도... ({type(e).__name__})")
        try:
            raw_text: str | None = getattr(e, "llm_output", None)
            if raw_text is None:
                print("[정제 어댑터] raw 텍스트 없음 -> LLM 비구조화 재호출")
                raw_resp = (prompt | llm).invoke(invoke_kwargs)
                raw_text = raw_resp.content if hasattr(raw_resp, "content") else str(raw_resp)
            normalized = normalize_llm_response(raw_text, llm=llm)
            _debug_print_llm_output("NORMALIZED LLM OUTPUT", normalized)
            return _validate_or_fallback(
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
            )
        except Exception as norm_e:
            reason = map_llm_error(e)
            print(f"[정제 어댑터] 복구 실패: {norm_e} -> fallback 루틴으로 전환")
    except Exception as e:
        reason = map_llm_error(e)
        print(f"[AI 실패 - {reason.upper()}] {e} -> fallback 루틴으로 전환")

    return _fallback(req, ranked_candidates, max_total_sets, doms_db, goal, reason, recent_sets, profile)

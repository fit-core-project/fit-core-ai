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
import time
from typing import Callable, Dict, List, Optional

import httpx
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError
from sqlalchemy.orm import Session

from .candidate_pool_policy import get_candidate_pool_size
from .candidate_ranker import format_candidates_for_prompt, score_candidate_exercises
from .db_queries import get_candidate_exercises
from .fallback import (
    _debug_print_draft,
    _debug_print_llm_output,
    _debug_print_prompt,
    build_routine_draft,
    generate_fallback_routine,
)
from .llm_router import StatusReasonCode, classify_llm_error, get_llm, map_llm_error
from .llm_parser import normalize_llm_response, trim_routine_to_time_budget
from .log_redaction import sanitize_exception_for_log, summarize_text_for_log
from .muscle_mapping import get_mapped_targets, split_label_to_muscles
from .prescription.adjustments import _calculate_max_total_sets, _target_exercise_count
from .prescription.estimator import estimate_routine_time_min
from .prescription.targets import apply_deterministic_targets
from .prompt_builder import _format_request_pain_areas, build_system_prompt
from .schemas import LLMRoutineOutput, RecentSetRecord, RoutineDraftResponse, RoutineRequest, UserProfileContext
from .temperature_policy import resolve_generation_temperature
from .routine_telemetry import classify_fallback_reason, observe_routine_quality


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
        print(
            f"[FeedbackRanker] adjustment fetch failed "
            f"type={err['type']} category={err['category']}"
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
    on_time_budget_exceeded: Optional[Callable[[], None]] = None,
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
            if on_time_budget_exceeded is not None:
                on_time_budget_exceeded()
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
    on_validation_failed: Optional[Callable[[], None]] = None,
    on_time_budget_exceeded: Optional[Callable[[], None]] = None,
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
        if on_validation_failed is not None:
            on_validation_failed()
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
        on_time_budget_exceeded=on_time_budget_exceeded,
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
        pain_areas=pain_areas,
        recent_sets=recent_sets,
        preferred_exercise_ids=req.preferred_exercise_ids,
        unpreferred_exercise_ids=req.unpreferred_exercise_ids,
        top_n=candidate_pool_size,
        feedback_adjustments=feedback_adjustments,
    )
    feedback_stats = _compute_feedback_stats(ranked_candidates, feedback_adjustments)
    feedback_adjusted_candidate_count = feedback_stats["adjusted"]
    feedback_positive_count = feedback_stats["positive"]
    feedback_negative_count = feedback_stats["negative"]
    feedback_abs_adjustment_avg = feedback_stats["abs_avg"]
    max_total_sets = _calculate_max_total_sets(req.time_available_min, goal)
    target_exercise_count = _target_exercise_count(req.time_available_min)
    candidate_payload = format_candidates_for_prompt(ranked_candidates)
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
        "candidate_exercises": candidate_payload,
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
    generation_temp = resolve_generation_temperature(req.readiness_level)
    generation_latency_ms: Optional[int] = None
    schema_repair_latency_ms: Optional[int] = None
    llm_error_type: Optional[str] = None
    schema_repair_attempted = False
    schema_repair_succeeded = False
    validation_failed = False
    time_budget_exceeded = False

    def _mark_validation_failed() -> None:
        nonlocal validation_failed
        validation_failed = True

    def _mark_time_budget_exceeded() -> None:
        nonlocal time_budget_exceeded
        time_budget_exceeded = True

    def _observe(response: RoutineDraftResponse) -> RoutineDraftResponse:
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
            generation_temperature=generation_temp,
            generation_latency_ms=generation_latency_ms,
            schema_repair_latency_ms=schema_repair_latency_ms,
            fallback_reason=fallback_reason,
            llm_error_type=llm_error_type,
            schema_repair_attempted=schema_repair_attempted,
            schema_repair_succeeded=schema_repair_succeeded,
            feedback_enabled=feedback_enabled,
            feedback_adjusted_candidate_count=feedback_adjusted_candidate_count,
            feedback_positive_count=feedback_positive_count,
            feedback_negative_count=feedback_negative_count,
            feedback_abs_adjustment_avg=feedback_abs_adjustment_avg,
            feedback_query_latency_ms=feedback_query_latency_ms,
            feedback_query_failed=feedback_query_failed,
            feedback_query_error_category=feedback_query_error_category,
        )

    try:
        llm = get_llm("routine", temperature=generation_temp)
        generation_started = time.perf_counter()
        try:
            response: LLMRoutineOutput = (prompt | llm.with_structured_output(LLMRoutineOutput)).invoke(invoke_kwargs)
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
        )
        return _observe(draft)
    except (asyncio.TimeoutError, httpx.TimeoutException) as e:
        llm_error_type = classify_llm_error(e)
        reason = map_llm_error(e)
        print("[AI 실패 - TIMEOUT]", sanitize_exception_for_log(e), "-> fallback 루틴으로 전환")
    except (OutputParserException, ValidationError) as e:
        llm_error_type = classify_llm_error(e)
        schema_repair_attempted = True
        print(f"[AI - SCHEMA 오류] 정제 어댑터로 복구 시도... ({type(e).__name__})")
        try:
            repair_llm = get_llm("routine", temperature=0)
            raw_text: str | None = getattr(e, "llm_output", None)
            if raw_text is None:
                print("[정제 어댑터] raw 텍스트 없음 -> LLM 비구조화 재호출")
                raw_resp = (prompt | repair_llm).invoke(invoke_kwargs)
                raw_text = raw_resp.content if hasattr(raw_resp, "content") else str(raw_resp)
                raw_summary = summarize_text_for_log(raw_text)
                print(
                    "[정제 어댑터] raw LLM output 수신",
                    {
                        "redacted": True,
                        "llm_output_char_count": raw_summary["char_count"],
                        "llm_output_approx_tokens": raw_summary["approx_tokens"],
                    },
                )
            repair_started = time.perf_counter()
            try:
                normalized = normalize_llm_response(raw_text, llm=repair_llm)
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
            )
            if not validation_failed and not draft.is_fallback and draft.generation_status != "fallback":
                schema_repair_succeeded = True
            return _observe(draft)
        except Exception as norm_e:
            reason = map_llm_error(e)
            print("[정제 어댑터] 복구 실패:", sanitize_exception_for_log(norm_e), "-> fallback 루틴으로 전환")
    except Exception as e:
        llm_error_type = classify_llm_error(e)
        reason = map_llm_error(e)
        print(f"[AI 실패 - {reason.upper()}]", sanitize_exception_for_log(e), "-> fallback 루틴으로 전환")

    draft = _fallback(req, ranked_candidates, max_total_sets, doms_db, goal, reason, recent_sets, profile)
    return _observe(draft)

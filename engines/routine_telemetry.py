"""Observe-only telemetry for routine generation quality."""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Optional

from .schemas import RecentSetRecord, RoutineDraftResponse, RoutineRequest

ENV_RULE_CRITIC_TELEMETRY = "ENABLE_RULE_CRITIC_TELEMETRY"
_TRUE_VALUES = {"true", "1", "yes", "on"}
_FALSE_VALUES = {"", "false", "0", "no", "off"}


def resolve_rule_critic_telemetry_enabled(value: Optional[str] = None) -> bool:
    normalized = (value or "").strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return False


def get_rule_critic_telemetry_enabled() -> bool:
    return resolve_rule_critic_telemetry_enabled(os.getenv(ENV_RULE_CRITIC_TELEMETRY))


def count_guard_repairs(warnings: list[str]) -> int:
    return len([w for w in warnings if "Guard:" in w and "replaced" in w])


def classify_fallback_reason(
    *,
    fallback_used: bool,
    llm_error_type: Optional[str],
    schema_repair_attempted: bool,
    schema_repair_succeeded: bool,
    validation_failed: bool,
    time_budget_exceeded: bool,
) -> str:
    if not fallback_used:
        return "none"
    if llm_error_type == "timeout":
        return "timeout"
    if llm_error_type == "quota_exhausted":
        return "quota_exhausted"
    if llm_error_type in ("schema_parse_error", "validation_error") and schema_repair_attempted:
        if schema_repair_succeeded:
            return "schema_error_recovered"
        return "schema_error_unrecoverable"
    if validation_failed:
        return "validation_failed"
    if time_budget_exceeded:
        return "time_budget_exceeded"
    if llm_error_type == "network_error":
        return "network_error"
    if llm_error_type == "unknown":
        return "unknown_llm_error"
    return "unknown_llm_error"


def _dynamic_temperature_enabled() -> bool:
    from .temperature_policy import is_dynamic_temperature_enabled

    return is_dynamic_temperature_enabled()


def build_routine_quality_telemetry_payload(
    *,
    response: RoutineDraftResponse,
    candidate_pool_size: int,
    ranked_candidates: list[dict],
    candidate_payload: str,
    generation_temperature: float,
    critic_result: Any,
    repair_count: int,
    fallback_used: bool,
    target_duration_min: int,
    dynamic_temperature_enabled: Optional[bool] = None,
    generation_latency_ms: Optional[int] = None,
    schema_repair_latency_ms: Optional[int] = None,
    fallback_reason: str = "none",
    llm_error_type: Optional[str] = None,
    schema_repair_attempted: bool = False,
    schema_repair_succeeded: bool = False,
    feedback_enabled: bool = False,
    feedback_adjusted_candidate_count: int = 0,
    feedback_positive_count: int = 0,
    feedback_negative_count: int = 0,
    feedback_abs_adjustment_avg: float = 0.0,
    feedback_query_latency_ms: Optional[int] = None,
    feedback_query_failed: bool = False,
    feedback_query_error_category: Optional[str] = None,
    created_at: Optional[datetime] = None,
) -> dict[str, Any]:
    payload_created_at = created_at or datetime.now(timezone.utc)
    candidate_payload_char_count = len(candidate_payload)
    return {
        "event": "routine_generation_quality",
        "routine_draft_id": response.routine_draft_id,
        "candidate_pool_size": candidate_pool_size,
        "candidate_count": len(ranked_candidates),
        "candidate_payload_char_count": candidate_payload_char_count,
        "approx_candidate_payload_tokens": math.ceil(candidate_payload_char_count / 4),
        "dynamic_temperature_enabled": (
            _dynamic_temperature_enabled()
            if dynamic_temperature_enabled is None
            else dynamic_temperature_enabled
        ),
        "generation_temperature": generation_temperature,
        "critic_score": getattr(critic_result, "score", None),
        "critic_grade": getattr(critic_result, "grade", None),
        "critic_warning_count": len(getattr(critic_result, "warnings", []) or []),
        "hard_violation_count": len(getattr(critic_result, "hard_violations", []) or []),
        "repair_count": repair_count,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "llm_error_type": llm_error_type,
        "schema_repair_attempted": schema_repair_attempted,
        "schema_repair_succeeded": schema_repair_succeeded,
        "feedback_enabled": feedback_enabled,
        "feedback_adjusted_candidate_count": feedback_adjusted_candidate_count,
        "feedback_positive_count": feedback_positive_count,
        "feedback_negative_count": feedback_negative_count,
        "feedback_abs_adjustment_avg": round(float(feedback_abs_adjustment_avg), 3),
        "feedback_query_latency_ms": feedback_query_latency_ms,
        "feedback_query_failed": feedback_query_failed,
        "feedback_query_error_category": feedback_query_error_category,
        "exercise_count": len(response.routine_blocks),
        "total_estimated_time": response.total_estimated_time,
        "target_duration_min": target_duration_min,
        "created_at": payload_created_at.isoformat().replace("+00:00", "Z"),
        "should_rebuild": bool(getattr(critic_result, "should_rebuild", False)),
        "should_fallback": bool(getattr(critic_result, "should_fallback", False)),
        "generation_latency_ms": generation_latency_ms,
        "schema_repair_latency_ms": schema_repair_latency_ms,
    }


def emit_routine_quality_telemetry(payload: dict[str, Any]) -> None:
    try:
        print("[Telemetry] " + json.dumps(payload, ensure_ascii=False, sort_keys=True))
    except Exception:
        pass


def observe_routine_quality(
    *,
    response: RoutineDraftResponse,
    req: RoutineRequest,
    ranked_candidates: list[dict],
    db_target_muscles: list[str],
    recent_sets: Optional[list[RecentSetRecord]],
    max_total_sets: int,
    candidate_pool_size: int,
    candidate_payload: str,
    generation_temperature: float,
    generation_latency_ms: Optional[int] = None,
    schema_repair_latency_ms: Optional[int] = None,
    fallback_reason: str = "none",
    llm_error_type: Optional[str] = None,
    schema_repair_attempted: bool = False,
    schema_repair_succeeded: bool = False,
    feedback_enabled: bool = False,
    feedback_adjusted_candidate_count: int = 0,
    feedback_positive_count: int = 0,
    feedback_negative_count: int = 0,
    feedback_abs_adjustment_avg: float = 0.0,
    feedback_query_latency_ms: Optional[int] = None,
    feedback_query_failed: bool = False,
    feedback_query_error_category: Optional[str] = None,
) -> RoutineDraftResponse:
    if not get_rule_critic_telemetry_enabled():
        return response

    try:
        from .rule_critic import build_eval_context, evaluate_routine_quality

        repair_count = count_guard_repairs(response.warnings)
        fallback_used = response.is_fallback or response.generation_status == "fallback"
        context = build_eval_context(req, db_target_muscles, recent_sets, max_total_sets)
        critic_result = evaluate_routine_quality(
            response,
            ranked_candidates,
            context,
            repair_count=repair_count,
            fallback_used=fallback_used,
        )
        payload = build_routine_quality_telemetry_payload(
            response=response,
            candidate_pool_size=candidate_pool_size,
            ranked_candidates=ranked_candidates,
            candidate_payload=candidate_payload,
            generation_temperature=generation_temperature,
            critic_result=critic_result,
            repair_count=repair_count,
            fallback_used=fallback_used,
            target_duration_min=req.time_available_min,
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
        emit_routine_quality_telemetry(payload)
    except Exception as exc:
        print(f"[Telemetry] emit skipped: {type(exc).__name__}")
    return response

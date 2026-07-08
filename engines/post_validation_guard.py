"""
Post-LLM validation guard — secondary observability layer.

Policy:
  Primary enforcement: DB query-level filtering + score_candidate_exercises().
  Guard role: detect constraint leakage in the final LLM output and emit
              structured audit logs. Violations trigger explicit warnings in
              the response payload; silent fallback is prohibited.

Three hard constraints audited:
  1. Hallucination  — exercise_id absent from ranked candidates
  2. Pain area      — exercise pain_triggers match current injury areas
  3. Equipment      — exercise requires a blocked equipment token

Policy #4: when no safe replacement exists for a violation, the caller must
return generation_status="failed" with status_reason_code="emptyCandidate".
"""
import logging
from dataclasses import dataclass, field
from typing import List

from .candidate_ranker import _candidate_is_safe, _normalize_body_part
from .schemas import LLMRoutineOutput, PainAreaEntry

logger = logging.getLogger(__name__)


@dataclass
class Violation:
    exercise_id: str
    exercise_name: str
    constraint: str  # "hallucination" | "pain_area" | "equipment"
    detail: str


@dataclass
class GuardReport:
    violations: List[Violation] = field(default_factory=list)

    @property
    def has_violations(self) -> bool:
        return bool(self.violations)

    @property
    def level(self) -> str:
        return "clean" if not self.violations else "blocked"


def audit_routine_output(
    llm_output: LLMRoutineOutput,
    candidates: List[dict],
    blocked_equipment: List[str],
    pain_areas: List[PainAreaEntry],
) -> GuardReport:
    """
    Audit LLM output against hard constraints. Pure diagnostic — no mutation.
    Returns a GuardReport describing every violation found.
    """
    candidate_by_id = {str(c["id"]): c for c in candidates}
    safe_ids = {
        str(c["id"])
        for c in candidates
        if _candidate_is_safe(c, blocked_equipment, pain_areas)
    }
    violations: List[Violation] = []

    for exercise in llm_output.exercises:
        eid = str(exercise.exercise_id)
        candidate = candidate_by_id.get(eid)

        if candidate is None:
            violations.append(Violation(
                exercise_id=eid,
                exercise_name=exercise.exercise_name,
                constraint="hallucination",
                detail=f"exercise_id '{eid}' not present in ranked candidates",
            ))
            continue

        if eid in safe_ids:
            continue

        blocked = {item.strip().upper() for item in blocked_equipment}
        equip_tokens = (
            {t.strip().upper() for t in str(candidate.get("equipment_req") or "").split(",") if t.strip()}
            - {"BODYWEIGHT"}
        )
        pain_tokens = {p.body_part.lower() for p in pain_areas if p.body_part}
        candidate_pain = str(candidate.get("pain_triggers") or "").lower()
        normalized_pain_tokens = {
            _normalize_body_part(p.body_part)
            for p in pain_areas
            if p.body_part
        }
        primary_muscle = _normalize_body_part(candidate.get("primary_muscle"))

        if equip_tokens & blocked:
            violations.append(Violation(
                exercise_id=eid,
                exercise_name=exercise.exercise_name,
                constraint="equipment",
                detail=f"requires blocked equipment: {equip_tokens & blocked}",
            ))
        elif any(token in candidate_pain for token in pain_tokens):
            matched = sorted(t for t in pain_tokens if t in candidate_pain)
            violations.append(Violation(
                exercise_id=eid,
                exercise_name=exercise.exercise_name,
                constraint="pain_area",
                detail=f"pain_triggers matches injury area(s): {matched}",
            ))
        elif primary_muscle and primary_muscle in normalized_pain_tokens:
            violations.append(Violation(
                exercise_id=eid,
                exercise_name=exercise.exercise_name,
                constraint="pain_area",
                detail=f"primary_muscle directly targets pain area: {primary_muscle}",
            ))
        else:
            violations.append(Violation(
                exercise_id=eid,
                exercise_name=exercise.exercise_name,
                constraint="equipment",
                detail="unsafe candidate (equipment, pain trigger, or primary pain area conflict)",
            ))

    return GuardReport(violations=violations)


def log_guard_report(report: GuardReport) -> None:
    """Emit structured guard audit log to stdout."""
    if not report.has_violations:
        logger.info("[Guard] ✓ 제약 위반 없음 (clean)")
        return
    logger.warning("[Guard] ⚠ 위반 감지 — level=%s, count=%s", report.level, len(report.violations))
    for v in report.violations:
        logger.warning("  [%s] %s (%s): %s", v.constraint.upper(), v.exercise_id, v.exercise_name, v.detail)

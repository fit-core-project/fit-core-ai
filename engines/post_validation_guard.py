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
from typing import Any, List

from .candidate_ranker import (
    _candidate_injury_overlap,
    _candidate_is_safe,
    _pain_tokens_from_entries,
)
from .schemas import LLMRoutineOutput, PainAreaEntry
from .injury_condition_policy import candidate_injury_condition_overlaps
from .shoulder_condition_policy import candidate_restricted_movement_overlap

logger = logging.getLogger(__name__)


@dataclass
class Violation:
    exercise_id: str
    exercise_name: str
    constraint: str  # hallucination | pain_area | equipment | managed_injury_movement
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
    shoulder_condition_status: str | None = None,
    restricted_movement_patterns: List[str] | None = None,
    injury_conditions: List[Any] | None = None,
) -> GuardReport:
    """
    Audit LLM output against hard constraints. Pure diagnostic — no mutation.
    Returns a GuardReport describing every violation found.
    """
    candidate_by_id = {str(c["id"]): c for c in candidates}
    safe_ids = {
        str(c["id"])
        for c in candidates
        if _candidate_is_safe(
            c,
            blocked_equipment,
            pain_areas,
            shoulder_condition_status=shoulder_condition_status,
            restricted_movement_patterns=restricted_movement_patterns,
            injury_conditions=injury_conditions,
        )
    }
    violations: List[Violation] = []

    def audit_candidate_reference(
        eid: str,
        exercise_name: str,
        *,
        role: str,
    ) -> None:
        candidate = candidate_by_id.get(eid)
        detail_prefix = "substitution candidate " if role == "substitution" else ""

        if candidate is None:
            violations.append(Violation(
                exercise_id=eid,
                exercise_name=exercise_name,
                constraint="hallucination",
                detail=(
                    f"{detail_prefix}exercise_id '{eid}' not present in ranked candidates"
                ),
            ))
            return

        if eid in safe_ids:
            return

        blocked = {item.strip().upper() for item in blocked_equipment}
        equip_tokens = (
            {t.strip().upper() for t in str(candidate.get("equipment_req") or "").split(",") if t.strip()}
            - {"BODYWEIGHT"}
        )
        injury_tokens = _pain_tokens_from_entries(pain_areas)
        injury_overlap = _candidate_injury_overlap(candidate, injury_tokens)
        shoulder_overlap = candidate_restricted_movement_overlap(
            candidate,
            shoulder_condition_status,
            restricted_movement_patterns or [],
        )
        managed_injury_overlap = candidate_injury_condition_overlaps(
            candidate,
            injury_conditions or [],
        )

        if equip_tokens & blocked:
            violations.append(Violation(
                exercise_id=eid,
                exercise_name=exercise_name,
                constraint="equipment",
                detail=(
                    f"{detail_prefix}requires blocked equipment: "
                    f"{equip_tokens & blocked}"
                ),
            ))
        elif injury_overlap:
            violations.append(Violation(
                exercise_id=eid,
                exercise_name=exercise_name,
                constraint="pain_area",
                detail=(
                    f"{detail_prefix}candidate involves registered injury area(s): "
                    f"{sorted(injury_overlap)}"
                ),
            ))
        elif shoulder_overlap:
            violations.append(Violation(
                exercise_id=eid,
                exercise_name=exercise_name,
                constraint="shoulder_movement",
                detail=(
                    f"{detail_prefix}candidate matches restricted shoulder movement(s): "
                    f"{sorted(shoulder_overlap)}"
                ),
            ))
        elif managed_injury_overlap:
            violations.append(Violation(
                exercise_id=eid,
                exercise_name=exercise_name,
                constraint="managed_injury_movement",
                detail=(
                    f"{detail_prefix}candidate matches managed injury restriction(s): "
                    f"{managed_injury_overlap}"
                ),
            ))
        else:
            violations.append(Violation(
                exercise_id=eid,
                exercise_name=exercise_name,
                constraint="equipment",
                detail=(
                    f"{detail_prefix}unsafe candidate "
                    "(equipment, pain trigger, or primary pain area conflict)"
                ),
            ))

    for exercise in llm_output.exercises:
        audit_candidate_reference(
            str(exercise.exercise_id),
            exercise.exercise_name,
            role="selected",
        )
        for substitution in exercise.substitution_candidates:
            audit_candidate_reference(
                str(substitution.exercise_id),
                substitution.exercise_name,
                role="substitution",
            )

    return GuardReport(violations=violations)


def log_guard_report(report: GuardReport) -> None:
    """Emit structured guard audit log to stdout."""
    if not report.has_violations:
        logger.info("[Guard] ✓ 제약 위반 없음 (clean)")
        return
    logger.warning("[Guard] ⚠ 위반 감지 — level=%s, count=%s", report.level, len(report.violations))
    for v in report.violations:
        logger.warning("  [%s] %s (%s): %s", v.constraint.upper(), v.exercise_id, v.exercise_name, v.detail)

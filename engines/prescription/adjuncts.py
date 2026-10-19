"""Deterministic prescription profiles for non-working gym adjunct blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from ..option_resolver import ADJUNCT_OPTION_PHASE, candidate_matches_adjunct_option
from ..schemas import LLMExercisePlan, LLMRoutineOutput
from ..experience_policy import automatic_recommendation_eligibility


@dataclass(frozen=True)
class AdjunctPrescriptionProfile:
    set_type: str
    sets: int
    target_reps: int
    target_rir: Optional[int]
    rest_sec: int
    target_duration_sec: Optional[int] = None
    interval_work_sec: Optional[int] = None
    interval_recovery_sec: Optional[int] = None
    interval_rounds: Optional[int] = None


# These are conservative product defaults, not medical or rehabilitation
# prescriptions. Timed/interval metadata prevents FE from presenting cardio or
# stretching blocks as ordinary hypertrophy sets.
ADJUNCT_PRESCRIPTION_PROFILES = {
    "mobility_warmup": AdjunctPrescriptionProfile(
        set_type="mobility",
        sets=2,
        target_reps=8,
        target_rir=None,
        rest_sec=15,
    ),
    "cooldown_stretch": AdjunctPrescriptionProfile(
        set_type="timed",
        sets=2,
        target_reps=1,
        target_rir=None,
        rest_sec=15,
        target_duration_sec=30,
    ),
    "conditioning_finisher": AdjunctPrescriptionProfile(
        set_type="interval",
        sets=1,
        target_reps=1,
        target_rir=None,
        rest_sec=0,
        target_duration_sec=480,
        interval_work_sec=30,
        interval_recovery_sec=60,
        interval_rounds=6,
    ),
    "endurance_support": AdjunctPrescriptionProfile(
        set_type="timed",
        sets=1,
        target_reps=1,
        target_rir=None,
        rest_sec=0,
        target_duration_sec=600,
    ),
    "prehab_accessory": AdjunctPrescriptionProfile(
        set_type="prehab",
        sets=2,
        target_reps=12,
        target_rir=3,
        rest_sec=45,
    ),
}


def prescription_set_type(exercise: LLMExercisePlan) -> str:
    return str(getattr(exercise, "_prescription_set_type", "working") or "working").lower()


def is_working_prescription(exercise: LLMExercisePlan) -> bool:
    # Loaded carries are productive work expressed by distance rather than reps.
    return prescription_set_type(exercise) in {"working", "distance"}


def _field(value: Any, snake_name: str, camel_name: str | None = None) -> Any:
    if isinstance(value, dict):
        if snake_name in value:
            return value.get(snake_name)
        return value.get(camel_name) if camel_name else None
    return getattr(value, snake_name, None)


def is_adjunct_block(block: Any) -> bool:
    """Return whether a final block belongs outside the main exercise count."""

    option = str(_field(block, "adjunct_option", "adjunctOption") or "").strip().lower()
    # Phase alone is insufficient: a normal core finisher or recovery block can
    # use the same phase without being requested as an adjunct. The public tag
    # is authoritative; missing tags are caught separately by the ledger check.
    return bool(option)


def count_main_blocks(blocks: Sequence[Any] | None) -> int:
    return sum(1 for block in (blocks or []) if not is_adjunct_block(block))


def adjunct_block_matches_profile(
    block: Any,
    adjunct_option: str,
    *,
    candidate: dict | None = None,
) -> bool:
    """Validate adjunct identity, phase, and serialized prescription semantics."""

    option = str(adjunct_option or "").strip().lower()
    profile = ADJUNCT_PRESCRIPTION_PROFILES.get(option)
    if profile is None:
        return False
    if candidate is not None and not candidate_matches_adjunct_option(candidate, option):
        return False
    if _field(block, "session_phase", "sessionPhase") != ADJUNCT_OPTION_PHASE.get(option):
        return False
    if int(_field(block, "default_rest_sec", "defaultRestSec") or 0) != profile.rest_sec:
        return False

    prescriptions = list(_field(block, "prescription") or [])
    if len(prescriptions) != profile.sets:
        return False
    for item in prescriptions:
        if str(_field(item, "set_type", "setType") or "working").lower() != profile.set_type:
            return False
        if int(_field(item, "target_reps", "targetReps") or 0) != profile.target_reps:
            return False
        if _field(item, "target_rir", "targetRir") != profile.target_rir:
            return False
        if int(_field(item, "target_rest_sec", "targetRestSec") or 0) != profile.rest_sec:
            return False
        if _field(item, "target_duration_sec", "targetDurationSec") != profile.target_duration_sec:
            return False
        if _field(item, "interval_work_sec", "intervalWorkSec") != profile.interval_work_sec:
            return False
        if _field(item, "interval_recovery_sec", "intervalRecoverySec") != profile.interval_recovery_sec:
            return False
        if _field(item, "interval_rounds", "intervalRounds") != profile.interval_rounds:
            return False
    return True


def validate_adjunct_decision_contract(
    requested_options: Sequence[str] | None,
    blocks: Sequence[Any] | None,
    decisions: Sequence[Any] | None,
    *,
    candidates_by_id: Mapping[str, dict] | None = None,
) -> list[str]:
    """Cross-check the public adjunct ledger against the final routine blocks.

    A requested option must have exactly one decision. Applied decisions must
    point to one distinct, explicitly tagged block with the exact deterministic
    prescription profile. Suppressed decisions must not leave a matching block
    behind. When the catalog lookup is supplied, candidate eligibility is also
    part of the contract check.
    """

    requested = list(dict.fromkeys(
        str(option or "").strip().lower()
        for option in (requested_options or [])
        if str(option or "").strip()
    ))
    final_blocks = list(blocks or [])
    final_decisions = list(decisions or [])
    catalog_supplied = candidates_by_id is not None
    candidate_lookup = {
        str(exercise_id or "").strip().lower(): candidate
        for exercise_id, candidate in (candidates_by_id or {}).items()
    }
    violations: list[str] = []

    decisions_by_option: dict[str, list[Any]] = {}
    for decision in final_decisions:
        option = str(_field(decision, "option") or "").strip().lower()
        decisions_by_option.setdefault(option, []).append(decision)
        if option not in requested:
            violations.append(f"unexpected adjunct decision: {option or '<empty>'}")

    applied_exercise_ids: list[str] = []
    for option in requested:
        option_decisions = decisions_by_option.get(option, [])
        if len(option_decisions) != 1:
            violations.append(
                f"adjunct decision cardinality invalid: option={option}, actual={len(option_decisions)}"
            )
            continue

        decision = option_decisions[0]
        status = str(_field(decision, "status") or "").strip().lower()
        reason_code = str(
            _field(decision, "reason_code", "reasonCode") or ""
        ).strip()
        slot_id = str(_field(decision, "slot_id", "slotId") or "").strip()
        decision_exercise_id = str(
            _field(decision, "exercise_id", "exerciseId") or ""
        ).strip().lower()
        expected_slot_id = f"adjunct_{option}"

        if slot_id != expected_slot_id:
            violations.append(
                f"adjunct decision slot mismatch: option={option}, "
                f"actual={slot_id or '<empty>'}, expected={expected_slot_id}"
            )

        tagged_blocks = [
            block
            for block in final_blocks
            if str(
                _field(block, "adjunct_option", "adjunctOption") or ""
            ).strip().lower() == option
        ]
        profile_blocks = [
            block
            for block in final_blocks
            if adjunct_block_matches_profile(block, option)
        ]

        if status == "suppressed":
            if reason_code == "applied":
                violations.append(
                    f"suppressed adjunct uses applied reason: option={option}"
                )
            if decision_exercise_id:
                violations.append(
                    f"suppressed adjunct retains exercise id: option={option}, "
                    f"exerciseId={decision_exercise_id}"
                )
            if tagged_blocks or profile_blocks:
                violations.append(
                    f"suppressed adjunct still has a final block: option={option}"
                )
            continue

        if status != "applied":
            violations.append(
                f"adjunct decision status invalid: option={option}, status={status or '<empty>'}"
            )
            continue
        if reason_code != "applied":
            violations.append(
                f"applied adjunct reason invalid: option={option}, reasonCode={reason_code or '<empty>'}"
            )
        if len(tagged_blocks) != 1:
            violations.append(
                f"applied adjunct block cardinality invalid: option={option}, actual={len(tagged_blocks)}"
            )
            continue

        matched_block = tagged_blocks[0]
        block_exercise_id = str(
            _field(matched_block, "exercise_id", "exerciseId") or ""
        ).strip().lower()
        candidate = candidate_lookup.get(block_exercise_id) if catalog_supplied else None
        if catalog_supplied:
            if candidate is None:
                violations.append(
                    f"applied adjunct candidate missing: option={option}, exerciseId={block_exercise_id}"
                )
            elif not candidate_matches_adjunct_option(candidate, option):
                violations.append(
                    f"applied adjunct candidate ineligible: option={option}, exerciseId={block_exercise_id}"
                )
        if not adjunct_block_matches_profile(matched_block, option):
            violations.append(
                f"applied adjunct profile invalid: option={option}, exerciseId={block_exercise_id}"
            )
        if not decision_exercise_id or decision_exercise_id != block_exercise_id:
            violations.append(
                f"applied adjunct exercise mismatch: option={option}, "
                f"decision={decision_exercise_id or '<empty>'}, block={block_exercise_id or '<empty>'}"
            )
        if block_exercise_id:
            applied_exercise_ids.append(block_exercise_id)

    if len(applied_exercise_ids) != len(set(applied_exercise_ids)):
        violations.append("applied adjunct options must use distinct final blocks")
    return violations


def apply_adjunct_prescription_profile(
    exercise: LLMExercisePlan,
    adjunct_option: str,
) -> bool:
    profile = ADJUNCT_PRESCRIPTION_PROFILES.get(str(adjunct_option or "").strip().lower())
    if profile is None:
        return False

    exercise.sets = profile.sets
    exercise.target_reps = profile.target_reps
    exercise.target_rir = profile.target_rir
    exercise.rest_time_sec = profile.rest_sec
    exercise.target_weight_kg = None
    from .weight import _set_weight_guidance

    _set_weight_guidance(exercise, "")
    exercise._prescription_set_type = profile.set_type
    exercise._target_duration_sec = profile.target_duration_sec
    exercise._interval_work_sec = profile.interval_work_sec
    exercise._interval_recovery_sec = profile.interval_recovery_sec
    exercise._interval_rounds = profile.interval_rounds
    return True


def apply_requested_adjunct_prescription(
    output: LLMRoutineOutput,
    adjunct_options: Sequence[str] | None,
    ranked_candidates: Sequence[dict] | None,
) -> LLMRoutineOutput:
    """Apply each requested adjunct profile to one distinct reviewed candidate."""

    requested = list(dict.fromkeys(
        str(option or "").strip().lower()
        for option in (adjunct_options or [])
        if str(option or "").strip().lower() in ADJUNCT_PRESCRIPTION_PROFILES
    ))
    if not requested:
        return output

    candidate_by_id = {
        str(candidate.get("id") or "").strip().lower(): candidate
        for candidate in (ranked_candidates or [])
        if str(candidate.get("id") or "").strip()
    }
    explicit_option_by_id = dict(
        getattr(output, "_adjunct_option_by_exercise_id", {}) or {}
    )
    selection_phase_by_id = dict(
        getattr(output, "_selection_phase_by_exercise_id", {}) or {}
    )
    used_exercise_ids: set[str] = set()
    for option in requested:
        explicit_exercise = next(
            (
                exercise
                for exercise in output.exercises
                if explicit_option_by_id.get(
                    str(exercise.exercise_id or "").strip().lower()
                ) == option
            ),
            None,
        )
        if explicit_exercise is not None:
            exercise_id = str(explicit_exercise.exercise_id or "").strip().lower()
            candidate = candidate_by_id.get(exercise_id)
            if (candidate and candidate_matches_adjunct_option(candidate, option)
                    and automatic_recommendation_eligibility(candidate, None, adjunct_options=(option,)).eligible):
                apply_adjunct_prescription_profile(explicit_exercise, option)
                selection_phase_by_id[exercise_id] = ADJUNCT_OPTION_PHASE[option]
                used_exercise_ids.add(exercise_id)
                continue
        for exercise in output.exercises:
            exercise_id = str(exercise.exercise_id or "").strip().lower()
            if not exercise_id or exercise_id in used_exercise_ids:
                continue
            candidate = candidate_by_id.get(exercise_id)
            if (candidate and candidate_matches_adjunct_option(candidate, option)
                    and automatic_recommendation_eligibility(candidate, None, adjunct_options=(option,)).eligible):
                apply_adjunct_prescription_profile(exercise, option)
                used_exercise_ids.add(exercise_id)
                explicit_option_by_id[exercise_id] = option
                selection_phase_by_id[exercise_id] = ADJUNCT_OPTION_PHASE[option]
                break
    output._adjunct_option_by_exercise_id = explicit_option_by_id
    output._selection_phase_by_exercise_id = selection_phase_by_id
    return output

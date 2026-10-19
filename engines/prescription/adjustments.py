"""운동 세트/중량/시간 조정 로직."""
import math
from typing import Dict, List, Optional

from ..schemas import LLMExercisePlan, LLMRoutineOutput
from ..registry import ACCESSORY_MUSCLE_SLUGS, LARGE_MUSCLE_SLUGS, ACCESSORY_PRIMARY_SPLITS
from .adjuncts import is_working_prescription, prescription_set_type
from .estimator import _movement_type_key, estimate_routine_time_min, _work_seconds_for_movement
from .params import _goal_key, GOAL_PARAMS
from .experience import experience_set_cap, normalize_experience_level


PRODUCTIVE_SESSION_CAP_MIN = 75
MIN_PRODUCTIVE_UTILIZATION_RATIO = 0.65
TARGET_PRODUCTIVE_UTILIZATION_RATIO = 0.85
DOMS_DOSE_POLICY_VERSION = "routine-recovery-dose-v2-2026-09-17"


def productive_session_reference_min(time_available_min: int) -> int:
    """Cap the useful target window so long availability never forces junk volume."""
    return max(1, min(int(time_available_min), PRODUCTIVE_SESSION_CAP_MIN))


def minimum_productive_duration_min(time_available_min: int) -> int:
    return max(
        1,
        math.ceil(
            productive_session_reference_min(time_available_min)
            * MIN_PRODUCTIVE_UTILIZATION_RATIO
        ),
    )


def target_productive_duration_min(time_available_min: int) -> int:
    return max(
        1,
        math.ceil(
            productive_session_reference_min(time_available_min)
            * TARGET_PRODUCTIVE_UTILIZATION_RATIO
        ),
    )


def productive_time_utilization_pct(actual_min: int, time_available_min: int) -> float:
    reference = productive_session_reference_min(time_available_min)
    return round(max(0, int(actual_min)) / reference * 100.0, 1)


def _apply_readiness_to_exercise(exercise: LLMExercisePlan, readiness_level: Optional[str]) -> None:
    readiness = (readiness_level or "normal").strip().lower()
    movement_type = _movement_type_key(exercise.movement_type)
    base_rir = exercise.target_rir if exercise.target_rir is not None else 2

    if readiness == "low":
        if movement_type == "COMPOUND" and exercise.sets > 2:
            exercise.sets = max(2, exercise.sets - 1)
        exercise.target_rir = max(4, min(5, base_rir + 2))
    elif readiness == "high":
        # High readiness may move a conservative beginner prescription from
        # 3 RIR to 2 RIR, but normal product generation does not prescribe
        # momentary failure. Explicit failure work is outside this core.
        exercise.target_rir = max(2, base_rir - 1)


def _uses_large_muscle_guard(
    target_split_label: Optional[str],
    target_muscles: Optional[List[str]],
) -> bool:
    split = (target_split_label or "").strip().lower()
    if split in ACCESSORY_PRIMARY_SPLITS:
        return False
    if target_muscles and not (set(target_muscles) & LARGE_MUSCLE_SLUGS):
        return False
    return True


def _apply_large_muscle_volume_guard(
    exercise: LLMExercisePlan,
    target_split_label: Optional[str],
    target_muscles: Optional[List[str]],
    *,
    primary_focus: bool = False,
) -> None:
    if primary_focus or not _uses_large_muscle_guard(target_split_label, target_muscles):
        return

    primary = exercise.primary_muscles[0] if exercise.primary_muscles else ""
    movement_type = _movement_type_key(exercise.movement_type)
    if primary in ACCESSORY_MUSCLE_SLUGS and movement_type != "COMPOUND":
        exercise.sets = min(exercise.sets, 2)


def _apply_doms_volume_policy(
    exercise: LLMExercisePlan,
    goal: str,
    doms_level: int,
    secondary_doms_level: int = 0,
    *,
    experience_level: Optional[str] = "intermediate",
    target_split_label: Optional[str] = None,
) -> None:
    """Make DOMS volume and effort deterministic and monotonic.

    The local model may emit different raw set counts for otherwise matched
    scenarios. Use the normal role/experience capacity, not a model-proposed
    set count. Percentages are versioned product heuristics, not clinical cutoffs.
    """
    if (prescription_set_type(exercise) != "working"
            or getattr(exercise, "_target_duration_sec", None)
            or getattr(exercise, "_target_distance_m", None)):
        return
    base_sets = min(
        _max_sets_for_exercise(exercise, goal, target_split_label),
        experience_set_cap(exercise, goal, experience_level, target_split_label),
    )
    mild_cap = max(1, round(base_sets * 0.8))
    if doms_level == 1:
        exercise.sets = min(exercise.sets, mild_cap)
        exercise.target_rir = max(3, int(exercise.target_rir or 0))
    elif doms_level == 2:
        exercise.sets = min(exercise.sets, mild_cap, max(1, math.ceil(base_sets * 0.6)))
        exercise.target_rir = max(4, int(exercise.target_rir or 0))
    if secondary_doms_level == 2:
        exercise.sets = min(exercise.sets, max(1, base_sets - 1))
        exercise.target_rir = max(3, int(exercise.target_rir or 0))


def _max_sets_for_exercise(
    exercise: LLMExercisePlan, goal: str, target_split_label: Optional[str],
    *, primary_focus: bool = False,
) -> int:
    goal_key = _goal_key(goal)
    movement_type = _movement_type_key(exercise.movement_type)
    primary = exercise.primary_muscles[0] if exercise.primary_muscles else ""
    if movement_type == "COMPOUND":
        return 5 if goal_key == "strength" else 4
    if (target_split_label or "").lower() in ACCESSORY_PRIMARY_SPLITS:
        return 4
    if primary_focus or primary in LARGE_MUSCLE_SLUGS:
        return 3
    return 2


def _calculate_max_total_sets(time_available_min: int, goal: str) -> int:
    rest_sec = GOAL_PARAMS.get(goal.lower(), GOAL_PARAMS["hypertrophy"])["rest_sec"]
    available_sec = max(0, time_available_min - 5) * 60
    per_set_sec = _work_seconds_for_movement("COMPOUND") + rest_sec
    return max(1, int(available_sec / per_set_sec))


def _target_exercise_count(time_available_min: int) -> int:
    if time_available_min <= 30:
        return 3
    if time_available_min <= 45:
        return 5
    if time_available_min <= 60:
        return 6
    if time_available_min <= 75:
        return 7
    return 8


def target_exercise_count_for_session(
    time_available_min: int,
    goal: str,
    experience_level: str | None = "intermediate",
) -> int:
    """Return the deterministic exercise-count target for a gym session.

    Most goals use the product's time bands. Strength sessions need fewer
    movements because their compound sets carry substantially longer rests.
    Derive that target from the same working-set budget used by prescription
    logic, while retaining at least three movements for basic session coverage.
    """
    base_target = _target_exercise_count(time_available_min)
    if _goal_key(goal) != "strength":
        return base_target

    max_total_sets = _calculate_max_total_sets(time_available_min, "strength")
    experience = normalize_experience_level(experience_level)
    # Beginner working sets are capped at three per compound movement.  Using
    # the intermediate four-set divisor here can produce a three-exercise plan
    # that cannot reach even the productive-duration floor after prescription.
    per_movement_capacity = 3 if experience == "beginner" else 4
    # Round up so the selected movement slots can actually absorb the working-set
    # budget without forcing every exercise above the normal four-set planning
    # reference. At 60 minutes, for example, 13 strength sets need four slots.
    strength_target = max(3, math.ceil(max_total_sets / per_movement_capacity))
    # A normal 50-minute intermediate strength session can otherwise stop at
    # one main lift plus two accessories. Their role-specific set caps do not
    # absorb the nominal set budget, leaving a visibly short session. Keep
    # 45-minute sessions compact, but open a fourth distinct movement from
    # 50 minutes onward so productive fill can use the available time.
    if experience != "beginner" and time_available_min >= 50:
        strength_target = max(strength_target, 4)
    return min(base_target, strength_target)


def _fill_available_time(
    output: LLMRoutineOutput,
    time_available_min: int,
    goal: str,
    readiness_level: Optional[str],
    target_split_label: Optional[str],
    doms_db: Optional[Dict[str, int]] = None,
    experience_level: Optional[str] = "intermediate",
    doms_restricted_exercise_ids: Optional[set[str]] = None,
    max_total_sets: Optional[int] = None,
    primary_focus_exercise_ids: Optional[set[str]] = None,
) -> dict[str, int | bool]:
    diagnostics: dict[str, int | bool] = {
        "role_set_capacity_exhausted": False,
        "role_set_capacity_total_budget_exhausted": False,
        "role_set_capacity_eligible_count": 0,
        "role_set_capacity_restricted_working_role": False,
        "role_set_capacity_unmeasured_working_role": False,
    }
    low_readiness = (readiness_level or "normal").strip().lower() == "low"
    target_min = target_productive_duration_min(time_available_min)
    current = estimate_routine_time_min(output.exercises)
    if current >= target_min:
        return diagnostics

    focus_ids = {
        str(value or "").strip().lower()
        for value in (primary_focus_exercise_ids or set())
    } if normalize_experience_level(experience_level) == "advanced" and not low_readiness else set()

    def role_fill_cap(exercise: LLMExercisePlan) -> int:
        cap = min(
            _max_sets_for_exercise(
                exercise, goal, target_split_label,
                primary_focus=str(exercise.exercise_id or "").strip().lower() in focus_ids,
            ),
            experience_set_cap(exercise, goal, experience_level, target_split_label),
        )
        # Preserve the existing one-set compound reduction, but apply it to
        # bounded role capacity instead of disabling all remaining planning.
        if low_readiness and _movement_type_key(exercise.movement_type) == "COMPOUND":
            cap = max(2, cap - 1)
        return cap

    candidates = sorted(
        [exercise for exercise in output.exercises if is_working_prescription(exercise)],
        key=lambda ex: (
            0 if str(ex.exercise_id or "").strip().lower() in focus_ids else 1,
            0 if _movement_type_key(ex.movement_type) == "COMPOUND" else 1,
            0 if (ex.primary_muscles and ex.primary_muscles[0] in LARGE_MUSCLE_SLUGS) else 1,
        ),
    )
    doms = doms_db or {}
    set_budget = None if max_total_sets is None else max(0, int(max_total_sets))
    total_working_sets = sum(
        max(0, exercise.sets)
        for exercise in output.exercises
        if is_working_prescription(exercise)
    )
    changed = True
    while changed and current < target_min:
        changed = False
        for exercise in candidates:
            if set_budget is not None and total_working_sets >= set_budget:
                break
            if low_readiness and (
                prescription_set_type(exercise) != "working"
                or getattr(exercise, "_target_duration_sec", None)
                or getattr(exercise, "_target_distance_m", None)
            ):
                continue
            exercise_id = str(exercise.exercise_id or "").strip().lower()
            if exercise_id in (doms_restricted_exercise_ids or set()):
                continue
            primary = exercise.primary_muscles[0] if exercise.primary_muscles else ""
            if doms.get(primary, 0) > 0:
                continue
            max_sets = role_fill_cap(exercise)
            if exercise.sets >= max_sets:
                continue
            exercise.sets += 1
            next_time = estimate_routine_time_min(output.exercises)
            if next_time > time_available_min:
                exercise.sets -= 1
                continue
            current = next_time
            total_working_sets += 1
            changed = True
            if current >= target_min:
                break

    working_roles = [
        exercise
        for exercise in output.exercises
        if (
            prescription_set_type(exercise) == "working"
            and not getattr(exercise, "_target_duration_sec", None)
            and not getattr(exercise, "_target_distance_m", None)
        )
    ]
    ordinary_working_role_ids = {id(exercise) for exercise in working_roles}
    unmeasured_working_role = any(
        is_working_prescription(exercise)
        and id(exercise) not in ordinary_working_role_ids
        for exercise in output.exercises
    )
    restricted_working_role = any(
        str(exercise.exercise_id or "").strip().lower()
        in (doms_restricted_exercise_ids or set())
        or doms.get(exercise.primary_muscles[0] if exercise.primary_muscles else "", 0)
        for exercise in working_roles
    )
    eligible_roles = [
        exercise
        for exercise in working_roles
        if str(exercise.exercise_id or "").strip().lower()
        not in (doms_restricted_exercise_ids or set())
        and not doms.get(exercise.primary_muscles[0] if exercise.primary_muscles else "", 0)
    ]
    diagnostics["role_set_capacity_eligible_count"] = len(eligible_roles)
    diagnostics["role_set_capacity_restricted_working_role"] = restricted_working_role
    diagnostics["role_set_capacity_unmeasured_working_role"] = unmeasured_working_role
    diagnostics["role_set_capacity_total_budget_exhausted"] = bool(
        set_budget is not None and total_working_sets >= set_budget
    )
    if current < target_min and eligible_roles and not restricted_working_role and not unmeasured_working_role and not diagnostics[
        "role_set_capacity_total_budget_exhausted"
    ]:
        diagnostics["role_set_capacity_exhausted"] = all(
            exercise.sets >= role_fill_cap(exercise)
            for exercise in eligible_roles
        )
    return diagnostics


def enforce_total_set_cap(
    output: LLMRoutineOutput,
    max_total_sets: int,
    target_muscles: Optional[List[str]] = None,
    protected_exercise_ids: Optional[set[str]] = None,
) -> bool:
    """Trim deterministic prescriptions so final working sets never exceed the cap."""
    if max_total_sets < 1:
        max_total_sets = 1

    total_sets = sum(
        max(0, exercise.sets)
        for exercise in output.exercises
        if is_working_prescription(exercise)
    )
    if total_sets <= max_total_sets:
        return False

    target_set = set(target_muscles or [])
    protected_ids = {
        str(exercise_id or "").strip().lower()
        for exercise_id in (protected_exercise_ids or set())
        if str(exercise_id or "").strip()
    }

    def priority(index: int, exercise: LLMExercisePlan) -> tuple[int, int, int, int]:
        protected_bonus = 4 if str(exercise.exercise_id or "").strip().lower() in protected_ids else 0
        movement_bonus = 1 if _movement_type_key(exercise.movement_type) == "COMPOUND" else 0
        primary = exercise.primary_muscles[0] if exercise.primary_muscles else ""
        target_bonus = 1 if primary in target_set else 0
        return (protected_bonus + movement_bonus + target_bonus, protected_bonus, movement_bonus, -index)

    trim_order = sorted(
        (
            (index, exercise)
            for index, exercise in enumerate(output.exercises)
            if is_working_prescription(exercise)
        ),
        key=lambda item: priority(item[0], item[1]),
    )
    overflow = total_sets - max_total_sets

    for _, exercise in trim_order:
        if overflow <= 0:
            break
        reducible = max(0, exercise.sets - 1)
        reduction = min(reducible, overflow)
        exercise.sets -= reduction
        overflow -= reduction

    while overflow > 0 and sum(
        1 for exercise in output.exercises if is_working_prescription(exercise)
    ) > 1:
        removable_index, removable = min(
            (
                (index, exercise)
                for index, exercise in enumerate(output.exercises)
                if is_working_prescription(exercise)
            ),
            key=lambda item: priority(item[0], item[1]),
        )
        overflow -= max(1, removable.sets)
        output.exercises.pop(removable_index)

    output.warnings = [*output.warnings, "set cap enforced"]
    return True

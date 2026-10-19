"""Joint time allocation for the five-role advanced upper-strength blueprint."""
from itertools import product

from .estimator import estimate_routine_time_min


UPPER_ROLE_ORDER = (
    "upper_push_main", "upper_pull_main", "upper_vertical_pull",
    "upper_biceps_direct", "upper_triceps_direct",
)


def uses_joint_upper_budget(blueprint, slot_by_id):
    return bool(
        blueprint is not None
        and blueprint.formula_version == "gym-routine-blueprint-v10"
        and not blueprint.planned_adjunct_options
        and len(slot_by_id) == len(UPPER_ROLE_ORDER)
        and set(slot_by_id.values()) == set(UPPER_ROLE_ORDER)
    )


def allocate_upper_role_sets(output, blueprint, time_available_min, max_total_sets):
    """Reduce existing caps jointly; never add sets, exercises, or shorter rests.

    Three main sets and two direct-arm sets are a product coverage preference,
    not a universal physiological minimum. Existing lower caps take precedence.
    Return False without mutation when this scoped allocation cannot fit.
    """
    slots = dict(output._selection_slot_by_exercise_id or {})
    if not uses_joint_upper_budget(blueprint, slots):
        return False
    by_slot = {slots.get(str(p.exercise_id).strip().lower()): p for p in output.exercises}
    if set(by_slot) != set(UPPER_ROLE_ORDER) or len(output.exercises) != 5:
        return False
    plans = [by_slot[s] for s in UPPER_ROLE_ORDER]
    caps = [p.sets for p in plans]
    floors = [min(cap, 3 if i < 3 else 2) for i, cap in enumerate(caps)]
    if any(cap < 1 or cap > 8 for cap in caps):
        return False
    trial = output.model_copy(deep=True)
    trial_by_id = {p.exercise_id: p for p in trial.exercises}
    best = None
    for counts in product(*(range(low, high + 1) for low, high in zip(floors, caps))):
        if sum(counts) > max_total_sets:
            continue
        for plan, count in zip(plans, counts):
            trial_by_id[plan.exercise_id].sets = count
        minutes = estimate_routine_time_min(
            trial.exercises, execution_metadata=trial._execution_metadata_by_exercise_id)
        if minutes > time_available_min:
            continue
        # Once all roles retain coverage, favor volume then the main slot order.
        score = (sum(counts), counts)
        if best is None or score > best[0]:
            best = (score, counts, minutes)
    if best is None:
        return False
    for plan, count in zip(plans, best[1]):
        plan.sets = count
    output.total_estimated_time = best[2]
    output._selection_diagnostics = {
        **dict(output._selection_diagnostics or {}),
        "upper_role_allocation_version": "joint-upper-roles-v1",
        "upper_role_allocation_changed": tuple(caps) != best[1],
    }
    return True

"""Bounded post-transfer capacity for reviewed lower-chest sessions."""
from .adjustments import _max_sets_for_exercise, target_productive_duration_min
from .estimator import estimate_routine_time_min
from .experience import experience_set_cap


POLICY_VERSION = "lower-chest-post-focus-capacity-v1"


def complete_lower_chest_focus_capacity(
    output, blueprint, request, candidates, pre_focus_sets, *,
    doms_db, restricted_exercise_ids, max_total_sets,
):
    """Restore at most one donor set, within its pre-transfer role ceiling.

    This is a scoped product allocation, not evidence that 12 chest sets are
    optimal. The focused main/isolation, exercise count, RIR and rest are fixed.
    Return final capacity diagnostics, or None outside the reviewed scope.
    """
    if (
        blueprint is None
        or blueprint.target_split_label != "chest"
        or blueprint.experience_level != "advanced"
        or blueprint.goal != "hypertrophy"
        or blueprint.specialization_focus != "lowerChest"
        or blueprint.readiness_level not in {"normal", "high"}
        or blueprint.session_structure != "standard"
        or blueprint.training_method is not None
        or blueprint.requested_adjunct_options
        or blueprint.safety_limited_target_muscles
        or request.time_available_min < 50
        or request.pain_areas or request.injury_conditions or request.condition_policies
        or request.mobility_limits or request.restricted_movement_patterns
        or request.shoulder_condition_status not in {None, "none"}
        or restricted_exercise_ids
        or any(int(level or 0) > 0 for level in (doms_db or {}).values())
    ):
        return None
    plans = {str(p.exercise_id).strip().lower(): p for p in output.exercises}
    slots = dict(output._selection_slot_by_exercise_id or {})
    required_slots = {"chest_main", "chest_secondary", "chest_isolation"}
    if (len(output.exercises) != 3 or len(plans) != 3 or set(plans) != set(pre_focus_sets)
            or set(slots) != set(plans) or set(slots.values()) != required_slots):
        return None
    by_id = {str(c.get("id") or "").strip().lower(): c for c in candidates}
    isolation_id = next(eid for eid in plans if slots[eid] == "chest_isolation")
    isolation = by_id.get(isolation_id, {})
    if not (isolation.get("core_chest_focus_reviewed") == 1
            and isolation.get("core_admission_scope") == "lowerChest"):
        return None
    if any(p.primary_muscles != ["chest"]
           or getattr(p, "_prescription_set_type", "working") != "working"
           or getattr(p, "_target_duration_sec", None)
           or getattr(p, "_target_distance_m", None)
           or getattr(p, "_training_method_instruction", None)
           for p in plans.values()):
        return None
    deltas = {eid: p.sets - pre_focus_sets[eid] for eid, p in plans.items()}
    receivers = [eid for eid, delta in deltas.items() if delta == 1]
    donors = [eid for eid, delta in deltas.items() if delta == -1]
    if (len(receivers) != 1 or len(donors) != 1 or sum(deltas.values()) != 0
            or slots[receivers[0]] != "chest_main" or slots[donors[0]] != "chest_secondary"
            or any(abs(delta) > 1 for delta in deltas.values())):
        return None
    # Preserve the already-approved focus increment, not an arbitrary LLM count.
    caps = {
        eid: min(_max_sets_for_exercise(p, blueprint.goal, "chest"),
                 experience_set_cap(p, blueprint.goal, "advanced", "chest"))
        for eid, p in plans.items()
    }
    receiver = receivers[0]
    if plans[receiver].sets > 5 or pre_focus_sets[receiver] > caps[receiver]:
        return None
    caps[receiver] = max(caps[receiver], plans[receiver].sets)
    donor = plans[donors[0]]
    before = {eid: p.sets for eid, p in plans.items()}
    total = sum(before.values())
    target_min = target_productive_duration_min(request.time_available_min)
    added = False
    if (total < max_total_sets
            and donor.sets < min(caps[donors[0]], pre_focus_sets[donors[0]])
            and estimate_routine_time_min(output.exercises) < target_min):
        donor.sets += 1
        if estimate_routine_time_min(output.exercises) <= request.time_available_min:
            added = True
        else:
            donor.sets -= 1
    total += int(added)
    duration = estimate_routine_time_min(output.exercises)
    diagnostics = {
        "role_set_capacity_exhausted": bool(
            duration < target_min and total < max_total_sets
            and all(p.sets >= caps[eid] for eid, p in plans.items())
        ),
        "role_set_capacity_total_budget_exhausted": total >= max_total_sets,
        "role_set_capacity_eligible_count": 3,
        "role_set_capacity_restricted_working_role": False,
        "role_set_capacity_unmeasured_working_role": False,
    }
    output._selection_diagnostics = {
        **dict(output._selection_diagnostics or {}),
        "focus_capacity_allocation": {
            "policy": POLICY_VERSION, "before": before,
            "after": {eid: p.sets for eid, p in plans.items()},
            "addedSets": int(added), "roleCaps": caps,
            "requestedMinutes": request.time_available_min, "estimatedMinutes": duration,
        },
    }
    return diagnostics

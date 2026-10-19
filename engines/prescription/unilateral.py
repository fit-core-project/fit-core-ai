"""Server-reviewed units only; names and model output never enable laterality."""
from types import MappingProxyType

from ..unilateral_contract import UnilateralPrescription

# No exercise is approved by this infrastructure change, including kickback.
REVIEWED_UNILATERAL_PRESCRIPTIONS = MappingProxyType({})


def has_reviewed_unilateral_metadata(candidate):
    exact_id = candidate.get('id')
    return exact_id == candidate.get('catalog_exercise_id') and exact_id in REVIEWED_UNILATERAL_PRESCRIPTIONS


def apply_reviewed_unilateral_metadata(plan, candidate):
    plan._unilateral_prescription = None
    plan._unilateral_set_overrides = None
    exact_id = candidate.get('id')
    if exact_id != candidate.get('catalog_exercise_id') or exact_id != plan.exercise_id:
        return plan
    value = REVIEWED_UNILATERAL_PRESCRIPTIONS.get(exact_id)
    if value is not None:
        plan._unilateral_prescription = UnilateralPrescription.model_validate(value)
        unilateral_work_seconds(plan, 0)
    return plan


def restore_unilateral_from_block(plan, block):
    units = getattr(block, 'unilateral_prescription', None)
    if units is None:
        return plan
    plan._unilateral_prescription = units
    plan._unilateral_set_overrides = [s.model_dump() for s in block.prescription]
    plan.sets = len(block.prescription)
    if block.training_method is not None:
        plan._training_method_instruction = block.training_method.model_dump(by_alias=True)
    return plan


def _set_values(plan, index):
    overrides = plan._unilateral_set_overrides
    if overrides is not None:
        if len(overrides) != plan.sets:
            raise ValueError('Unilateral set overrides must match the current set count')
        return overrides[index]
    return dict(set_type=plan._prescription_set_type, target_reps=plan.target_reps,
                target_duration_sec=plan._target_duration_sec, target_rest_sec=plan.rest_time_sec)


def unilateral_work_seconds(plan, index):
    units = plan._unilateral_prescription
    if units is None:
        return None
    value = _set_values(plan, index)
    if units.quantity_mode == 'durationPerSide':
        duration = value['target_duration_sec']
        if value['set_type'] != 'timed' or type(duration) is not int or duration <= 0:
            raise ValueError('Unilateral duration requires positive timed seconds per side')
        if plan._training_method_instruction:
            raise ValueError('Timed unilateral work cannot include repetition methods')
        per_side = duration
    else:
        reps = value['target_reps']
        if value['set_type'] != 'working' or type(reps) is not int or reps <= 0:
            raise ValueError('Unilateral repetitions require positive working reps per side')
        method = plan._training_method_instruction or {}
        # Preserve the established method increment, on each side exactly once.
        extra = max(0, int(method.get('pauseSec') or 0)) + max(0, int(method.get('eccentricSec') or 0) - 2)
        per_side = reps * (units.seconds_per_rep + extra)
    return 2 * per_side + units.side_switch_sec


def unilateral_rest_seconds(plan, index):
    if plan._unilateral_prescription is None:
        return max(0, plan.rest_time_sec)
    return max(0, _set_values(plan, index)['target_rest_sec'])

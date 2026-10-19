"""Pending approval only. No imports from this package in runtime engines."""
from dataclasses import dataclass

EXERCISE_ID = 'V24ROW-003678'
APPROVAL_STATUS = 'pending'


@dataclass(frozen=True)
class PreparedReview:
    ready_if_approved: bool
    unmet_conditions: tuple[str, ...]
    runtime_enabled: bool = False


def evaluate_prepared_conditions(row, req, setup):
    reasons = []
    expected = {
        'id': EXERCISE_ID, 'catalog_exercise_id': EXERCISE_ID,
        'catalog_original_name_kr': '케이블 스탠딩 싱글레그 글루트 킥백',
        'equipment_req': 'CABLE', 'canonical_primary_muscle': 'gluteal',
        'canonical_movement_type': 'ISOLATION', 'prescription_mode': 'reps',
        'injury_caution_level': 'low', 'main_routine_default_allowed': 0,
    }
    if any(row.get(k) != v for k, v in expected.items()):
        reasons.append('exact_variant_or_review_metadata_mismatch')
    if req.experience_level not in {'intermediate', 'advanced'}:
        reasons.append('experience')
    if 'CABLE' not in req.available_equipment or 'CABLE' in req.equipment:
        reasons.append('equipment')
    if EXERCISE_ID in req.unpreferred_exercise_ids:
        reasons.append('unpreferred')
    if (req.target_muscles != ['gluteal'] or req.target_split_label != 'custom'
            or req.goal != 'hypertrophy' or req.training_style != 'gym'
            or req.session_structure != 'standard' or req.readiness_level == 'low'
            or req.specialization_focus or req.adjunct_options or req.requested_method):
        reasons.append('unsupported_request_scope')
    if any((req.pain_areas, req.doms_data, req.injury_conditions, req.condition_policies,
            req.mobility_limits, req.profile_signals, req.anthropometry_signals,
            req.shoulder_condition_status, req.restricted_movement_patterns, req.user_note)):
        reasons.append('restriction_context')
    for key in ('lowPulleyAvailable', 'ankleCuffAvailable', 'stableHandSupportAvailable',
                'exactVariantFamiliar', 'restrictionContextResolved'):
        if setup.get(key) is not True:
            reasons.append(key)
    return PreparedReview(not reasons, tuple(reasons))


def preview_bilateral_set(*, reps_per_side: int, seconds_per_rep: int, side_switch_sec: int):
    """Arithmetic preview, not a production prescription or clinical formula."""
    if any(type(v) is not int for v in (reps_per_side, seconds_per_rep, side_switch_sec)):
        raise ValueError('Integer units required')
    if not (1 <= reps_per_side <= 30 and 1 <= seconds_per_rep <= 10 and 0 <= side_switch_sec <= 60):
        raise ValueError('Outside the prepared unit bounds')
    return {'repsPerSide': reps_per_side, 'sideCount': 2, 'setUnit': 'bothSides',
            'workAndSwitchSec': 2 * reps_per_side * seconds_per_rep + side_switch_sec,
            'restPlacement': 'afterBothSides', 'runtimeEnabled': False}

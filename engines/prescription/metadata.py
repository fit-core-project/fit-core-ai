"""Exercise-level prescription unit and session-phase metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Optional

from ..schemas import LLMExercisePlan
from ..specialization_policy import training_method_instruction
from .unilateral import apply_reviewed_unilateral_metadata


PRESCRIPTION_METADATA_VERSION = "routine-prescription-metadata-v3-2026-07-30"
LAT_PULLDOWN_CORRECTION_VERSION = "lat-pulldown-metadata-v1-2026-10-01"


def is_standard_lat_pulldown(name_en: object) -> bool:
    # The catalog also calls straight-arm, scapular and mixed movements pulldowns.
    return re.fullmatch(
        r"(?:machine|cable|band) (?:pronated|supinated|neutral|wide|close) "
        r"(?:single arm )?lat pulldown",
        " ".join(str(name_en or "").lower().split()),
    ) is not None


def _compact(value: object) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("/", "")
    )


@dataclass(frozen=True)
class PrescriptionMetadataDecision:
    source_movement_type: str
    canonical_movement_type: str
    prescription_mode: str
    preferred_session_phase: str
    allowed_session_phases: str
    default_duration_sec: Optional[int]
    default_distance_m: Optional[int]
    default_interval_work_sec: Optional[int]
    default_interval_recovery_sec: Optional[int]
    default_interval_rounds: Optional[int]
    default_rest_sec: Optional[int]
    classification_rule_code: str
    metadata_version: str
    source_type: str
    confidence: str
    review_status: str
    review_required: int

    def to_dict(self) -> dict:
        return asdict(self)


_DYNAMIC_PLANK_TOKENS = (
    "푸시업",
    "푸쉬업",
    "pushup",
    "숄더탭",
    "shouldertap",
    "레그리프트",
    "leglift",
    "힙어브덕션",
    "hipabduction",
    "힙딥",
    "hipdip",
    "니드라이브",
    "kneedrive",
    "마운틴",
    "mountain",
    "잭",
    "jack",
    "로우",
    "row",
    "리치",
    "reach",
    "워크",
    "walk",
    "로테이션",
    "rotation",
    "교대",
    "alternating",
)
_INTERVAL_MODALITY_TOKENS = (
    "마운틴클라이머",
    "mountainclimber",
    "버피",
    "burpee",
    "로잉머신",
    "rowingerg",
    "트레드밀",
    "treadmill",
    "바이크",
    "bike",
    "줄넘기",
    "jumprope",
)
_INTERVAL_FORMAT_TOKENS = (
    "인터벌",
    "interval",
    "amrap",
    "emom",
    "타바타",
    "tabata",
    "스프린트",
    "sprint",
    "고강도",
    "highintensity",
)


def classify_exercise_prescription_metadata(
    *,
    name_kr: object,
    name_en: object,
    source_movement_type: object,
) -> PrescriptionMetadataDecision:
    text = _compact(f"{name_kr or ''} {name_en or ''}")
    source_type = str(source_movement_type or "").strip().upper() or "UNKNOWN"

    values = {
        "source_movement_type": source_type,
        "canonical_movement_type": source_type,
        "prescription_mode": "reps",
        "preferred_session_phase": (
            "secondary_compound" if source_type == "COMPOUND" else "isolation_accessory"
        ),
        "allowed_session_phases": (
            "power_if_explicit,main_compound,secondary_compound"
            if source_type == "COMPOUND"
            else "isolation_accessory"
        ),
        "default_duration_sec": None,
        "default_distance_m": None,
        "default_interval_work_sec": None,
        "default_interval_recovery_sec": None,
        "default_interval_rounds": None,
        "default_rest_sec": None,
        "classification_rule_code": "source_movement_repetition",
        "metadata_version": PRESCRIPTION_METADATA_VERSION,
        "source_type": "deterministicNameAndMovementRule",
        "confidence": "medium",
        "review_status": "sourceDerived",
        "review_required": 0,
    }

    is_interval_modality = any(token in text for token in _INTERVAL_MODALITY_TOKENS)
    is_interval_format = any(token in text for token in _INTERVAL_FORMAT_TOKENS)
    is_static_stretch = (
        any(token in text for token in ("스트레치", "stretch"))
        and not any(token in text for token in ("동적", "dynamic"))
    )
    is_dynamic_stretch = (
        any(token in text for token in ("스트레치", "stretch"))
        and any(token in text for token in ("동적", "dynamic"))
    )
    is_plank = "플랭크" in text or "plank" in text
    is_dynamic_plank = is_plank and any(token in text for token in _DYNAMIC_PLANK_TOKENS)
    is_fly_or_crossover = (
        ("플라이" in text and "플라이오" not in text)
        or "크로스오버" in text
        or "chestfly" in text
        or "pecfly" in text
        or "crossover" in text
    )
    is_leg_press_or_machine_squat = (
        "프레스스쿼트" in text
        or "legpresssquat" in text
    )
    is_hip_thrust_or_glute_bridge = any(
        token in text
        for token in (
            "힙쓰러스트",
            "hipthrust",
            "글루트브리지",
            "glutebridge",
        )
    )
    is_static_hold = any(
        token in text
        for token in (
            "정적유지",
            "statichold",
            "isometric",
            "아이소메트릭",
            "월싯",
            "wallsit",
        )
    ) or (is_plank and not is_dynamic_plank)
    is_distance_carry = any(
        token in text
        for token in ("파머스워크", "farmerswalk", "farmercarry", "수트케이스캐리", "suitcasecarry")
    )
    is_steady_endurance = any(
        token in text
        for token in ("지속주", "steady", "걷기", "walking", "조깅", "jogging")
    ) and is_interval_modality

    if is_interval_modality and is_interval_format:
        values.update({
            "canonical_movement_type": "CONDITIONING",
            "prescription_mode": "interval",
            "preferred_session_phase": "conditioning_finisher",
            "allowed_session_phases": "conditioning_finisher",
            "default_duration_sec": 480,
            "default_interval_work_sec": 30,
            "default_interval_recovery_sec": 60,
            "default_interval_rounds": 6,
            "default_rest_sec": 0,
            "classification_rule_code": "explicit_conditioning_interval",
            "confidence": "high",
            "review_status": "autoClassified",
        })
    elif is_steady_endurance:
        values.update({
            "canonical_movement_type": "CONDITIONING",
            "prescription_mode": "duration",
            "preferred_session_phase": "endurance_support",
            "allowed_session_phases": "endurance_support",
            "default_duration_sec": 600,
            "default_rest_sec": 0,
            "classification_rule_code": "explicit_steady_endurance",
            "confidence": "high",
            "review_status": "autoClassified",
        })
    elif is_static_stretch:
        values.update({
            "canonical_movement_type": "STATIC",
            "prescription_mode": "duration",
            "preferred_session_phase": "cooldown_recovery",
            "allowed_session_phases": "cooldown_recovery",
            "default_duration_sec": 30,
            "default_rest_sec": 15,
            "classification_rule_code": "static_stretch_duration",
            "confidence": "high",
            "review_status": "autoClassified",
        })
    elif is_dynamic_stretch:
        values.update({
            "canonical_movement_type": "MOBILITY",
            "prescription_mode": "reps",
            "preferred_session_phase": "warmup_or_preparation",
            "allowed_session_phases": "warmup_or_preparation",
            "default_rest_sec": 15,
            "classification_rule_code": "dynamic_stretch_repetitions",
            "confidence": "high",
            "review_status": "autoClassified",
        })
    elif is_static_hold:
        values.update({
            "canonical_movement_type": "STATIC",
            "prescription_mode": "duration",
            "preferred_session_phase": "isolation_accessory",
            "allowed_session_phases": "isolation_accessory,prehab_accessory",
            "default_duration_sec": 30,
            "default_rest_sec": 45,
            "classification_rule_code": "static_hold_duration",
            "confidence": "high",
            "review_status": "autoClassified",
        })
    elif is_distance_carry:
        values.update({
            "canonical_movement_type": "CARRY",
            "prescription_mode": "distance",
            "preferred_session_phase": "isolation_accessory",
            "allowed_session_phases": "isolation_accessory,conditioning_finisher",
            "default_distance_m": 20,
            "default_rest_sec": 60,
            "classification_rule_code": "loaded_carry_distance",
            "confidence": "high",
            "review_status": "autoClassified",
        })
    elif is_dynamic_plank:
        values.update({
            "canonical_movement_type": "COMPOUND",
            "prescription_mode": "reps",
            "preferred_session_phase": "isolation_accessory",
            "allowed_session_phases": "isolation_accessory,conditioning_finisher",
            "classification_rule_code": "dynamic_plank_repetitions",
            "confidence": "high",
            "review_status": "autoClassified",
        })
    elif is_standard_lat_pulldown(name_en) and not any(
        token in text for token in (
            "straightarm", "stiffarm", "pullover", "scapular",
            "스트레이트암", "풀오버", "스캐풀라", "팔펴", "팔을펴",
        )
    ):
        values.update({
            "canonical_movement_type": "COMPOUND",
            "prescription_mode": "reps",
            "preferred_session_phase": "main_compound",
            "allowed_session_phases": "main_compound,secondary_compound",
            "classification_rule_code": "standard_lat_pulldown_compound_correction",
            "metadata_version": LAT_PULLDOWN_CORRECTION_VERSION,
            "confidence": "high",
            "review_status": "autoClassified",
        })
    elif is_leg_press_or_machine_squat or is_hip_thrust_or_glute_bridge:
        values.update({
            "canonical_movement_type": "COMPOUND",
            "prescription_mode": "reps",
            "preferred_session_phase": "main_compound",
            "allowed_session_phases": "main_compound,secondary_compound",
            "classification_rule_code": (
                "hip_thrust_glute_bridge_compound_correction"
                if is_hip_thrust_or_glute_bridge
                else "leg_press_machine_squat_compound_correction"
            ),
            "confidence": "high",
            "review_status": "autoClassified",
        })
    elif is_fly_or_crossover:
        values.update({
            "canonical_movement_type": "ISOLATION",
            "prescription_mode": "reps",
            "preferred_session_phase": "isolation_accessory",
            "allowed_session_phases": "isolation_accessory",
            "classification_rule_code": "fly_crossover_isolation_correction",
            "confidence": "high",
            "review_status": "autoClassified",
        })

    return PrescriptionMetadataDecision(**values)


def apply_candidate_prescription_metadata(
    exercise: LLMExercisePlan,
    candidate: dict,
) -> LLMExercisePlan:
    """Apply server-owned units to a selected plan before readiness/DOMS logic."""

    canonical_movement = str(candidate.get("canonical_movement_type") or "").strip()
    if canonical_movement:
        exercise.movement_type = canonical_movement
    mode = str(candidate.get("prescription_mode") or "reps").strip().lower()
    rest = candidate.get("prescription_default_rest_sec")
    if rest is not None:
        exercise.rest_time_sec = max(0, int(rest))

    if mode == "duration":
        exercise._prescription_set_type = "timed"
        exercise._target_duration_sec = int(candidate.get("default_duration_sec") or 30)
        exercise.target_reps = 1
        exercise.target_rir = None
        exercise.target_weight_kg = None
    elif mode == "interval":
        exercise._prescription_set_type = "interval"
        exercise._target_duration_sec = int(candidate.get("default_duration_sec") or 480)
        exercise._interval_work_sec = int(candidate.get("default_interval_work_sec") or 30)
        exercise._interval_recovery_sec = int(candidate.get("default_interval_recovery_sec") or 60)
        exercise._interval_rounds = int(candidate.get("default_interval_rounds") or 6)
        exercise.sets = 1
        exercise.target_reps = 1
        exercise.target_rir = None
        exercise.target_weight_kg = None
    elif mode == "distance":
        exercise._prescription_set_type = "distance"
        exercise._target_distance_m = int(candidate.get("default_distance_m") or 20)
        exercise.target_reps = 1

    method_code = str(candidate.get("training_method_code") or "").strip() or None
    method_instruction = candidate.get("training_method_instruction")
    if method_code:
        exercise._training_method_code = method_code
        exercise._training_method_instruction = (
            dict(method_instruction)
            if isinstance(method_instruction, dict)
            else training_method_instruction(method_code)
        )
    return apply_reviewed_unilateral_metadata(exercise, candidate)

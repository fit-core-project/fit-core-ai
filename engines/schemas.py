"""루틴 엔진 Pydantic 모델 정의 (Request / Response / LLM I/O / Context)."""
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator, model_serializer
from pydantic.alias_generators import to_camel

from engines.llm_router import StatusReasonCode
from engines.option_resolver import (
    MAX_DISTINCT_ADJUNCT_OPTIONS,
    adjunct_option_conflict_code,
    canonical_adjunct_options,
)
from engines.profile_constraint_contract import resolve_profile_constraint_signals
from engines.registry import MUSCLE_SLUGS, PUBLIC_DOMS_BODY_PART_SLUGS, PUBLIC_PAIN_AREA_SLUGS
from engines.injury_condition_policy import validate_injury_condition_combination
from engines.shoulder_condition_policy import validate_shoulder_condition_combination
from engines.unilateral_contract import UnilateralPrescription

GenerationStatus = Literal["success", "fallback", "failed"]
# StatusReasonCode는 engines.llm_router에서 단일 정의 후 re-export됨

ReadinessLevel = Literal["low", "normal", "high"]
ExperienceLevel = Literal["beginner", "intermediate", "advanced"]
AdvancedChestFocus = Literal[
    "balanced",
    "upperChest",
    "lowerChest",
    "pausedPress",
]
SpecializationFocus = Literal[
    "upperChest",
    "midChest",
    "lowerChest",
    "latEmphasis",
    "midBackEmphasis",
    "trapeziusEmphasis",
    "quadricepsEmphasis",
    "hamstringEmphasis",
    "gluteEmphasis",
    "adductorEmphasis",
    "abductorEmphasis",
    "calvesEmphasis",
    "frontDeltoidEmphasis",
    "sideDeltoidEmphasis",
    "rearDeltoidEmphasis",
    "bicepsEmphasis",
    "tricepsEmphasis",
    "forearmEmphasis",
]
TrainingMethod = Literal["pausedReps", "tempoEccentric"]
TrainingMethodMode = Literal["auto", "manual", "off"]
TrainingStyle = Literal[
    "gym",
    "bodybuilding",
    "crossfit",
    "athletic",
    "mobility",
    "yoga_pilates",
    "rehab_prehab",
    "home",
]
SessionStructure = Literal["standard", "superset", "circuit", "density"]
AdjunctOption = Literal[
    "conditioning_finisher",
    "endurance_support",
    "mobility_warmup",
    "cooldown_stretch",
    "prehab_accessory",
]
AdjunctDecisionStatus = Literal["applied", "suppressed"]
AdjunctDecisionReasonCode = Literal[
    "applied",
    "slotCapacityAfterMainMinimum",
    "noEligibleCandidate",
    "insufficientTimeAfterMainMinimum",
    "postValidationRemoved",
]
MobilityLimitCode = Literal[
    "limited_ankle_dorsiflexion",
    "limited_hip_flexion",
    "limited_shoulder_flexion",
    "limited_wrist_extension",
]
ShoulderConditionStatus = Literal[
    "activeUncleared",
    "managedWithRestrictions",
    "managedNoRestrictions",
    "recovered",
]
ShoulderRestrictedMovementPattern = Literal[
    "overheadPosition",
    "horizontalPush",
    "horizontalPull",
    "armRaise",
    "behindBackRotation",
    "unknown",
]
InjuryConditionAreaCode = Literal[
    "shoulder",
    "knee",
    "lumbar",
    "wrist",
    "ankle",
    "hip",
    "elbow",
    "neck",
]
InjuryConditionStatus = Literal[
    "activeUncleared",
    "managedWithRestrictions",
    "managedNoRestrictions",
    "recovered",
]
PrescriptionSetType = Literal[
    "working",
    "warmup",
    "mobility",
    "timed",
    "interval",
    "prehab",
    "distance",
]

_CLEARANCE_STATUS_ALIASES = {
    "notneeded": "not_needed",
    "not_needed": "not_needed",
    "cleared": "cleared",
    "clearedwithrestrictions": "cleared_with_restrictions",
    "cleared_with_restrictions": "cleared_with_restrictions",
    "notcleared": "not_cleared",
    "not_cleared": "not_cleared",
    "unknown": "unknown",
}
_REVIEWED_RUNTIME_ANTHROPOMETRY_ALIASES = {"long_arm", "long_femur"}

# ==========================================
# Request 모델
# ==========================================

class DomEntry(BaseModel):
    """Legacy DOMS-array adapter; active RoutineRequest uses integer domsData."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    body_part: str
    level: str  # Legacy DOMS label only, never registered-injury severity.


class PainAreaEntry(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")
    body_part: str = Field(validation_alias=AliasChoices("area", "bodyPart", "body_part"))
    side: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def reject_injury_severity(cls, value: Any) -> Any:
        """Registered injury areas are binary; soreness intensity belongs to DOMS."""
        if isinstance(value, dict):
            if {"severity", "level", "painLevel", "pain_level", "intensity"} & set(value):
                raise ValueError(
                    "painAreas is a severity-free injury list; use domsData/currentDoms for soreness levels"
                )
        return value


class InjuryConditionEntry(BaseModel):
    """Closed, previously assessed movement restrictions for one body region."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")
    area_code: InjuryConditionAreaCode
    status: InjuryConditionStatus
    restricted_movement_patterns: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_area_condition(self):
        self.restricted_movement_patterns = list(
            validate_injury_condition_combination(
                self.area_code,
                self.status,
                self.restricted_movement_patterns,
            )
        )
        return self


class RecentTrainingMethodUse(BaseModel):
    """Small, closed history signal used only to avoid repeated auto methods."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")

    method_code: TrainingMethod
    target_split_label: Optional[str] = Field(default=None, min_length=1, max_length=64)
    sessions_ago: int = Field(ge=1, le=12)

    @field_validator("target_split_label")
    @classmethod
    def normalize_target_split_label(cls, value):
        if value is None:
            return None
        return str(value).strip().lower().replace("-", "_").replace(" ", "_")


class RoutineRequest(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    user_id: Optional[str] = None
    target_split_label: Optional[str] = None         # "push" | "pull" | "legs" | ... (없으면 target_muscles 직접 사용)
    target_muscles: List[str] = Field(default_factory=list)  # spreadsheet slug 직접 지정 시 사용
    readiness_level: ReadinessLevel = "normal"
    time_available_min: int = Field(ge=10, le=180)
    pain_areas: List[PainAreaEntry] = Field(default_factory=list)  # 강도 없는 장기 부상·관리 부위
    doms_data: Dict[str, int] = Field(default_factory=dict)   # Java가 매핑한 {spreadsheet_slug: level(1~3)}
    equipment: List[str] = Field(default_factory=list)         # 사용 불가 장비 블랙리스트
    available_equipment: List[str] = Field(default_factory=list)  # 명시된 경우 사용 가능 장비 화이트리스트
    goal: Optional[str] = None                                 # 미전달 시 프로필의 goal_type 사용
    experience_level: Optional[ExperienceLevel] = None         # beginner | intermediate | advanced
    specialization_focus: Optional[SpecializationFocus] = None  # 선택하지 않으면 균형형
    method_mode: TrainingMethodMode = "auto"                    # 자동 추천 | 직접 선택 | 사용 안 함
    requested_method: Optional[TrainingMethod] = None           # 상급자 상세 설정의 수동 선택
    training_method: Optional[TrainingMethod] = None             # legacy input/internal compatibility
    recent_training_method_uses: List[RecentTrainingMethodUse] = Field(
        default_factory=list
    )
    advanced_chest_focus: AdvancedChestFocus = "balanced"       # legacy internal adapter
    training_style: TrainingStyle = "gym"                      # 사용자 UI 기본값은 gym, 나머지는 legacy/internal adjunct 호환
    session_structure: SessionStructure = "standard"           # standard | superset | circuit | density
    adjunct_options: List[AdjunctOption] = Field(default_factory=list)
    user_note: Optional[str] = None
    preferred_exercise_ids: List[str] = Field(default_factory=list)
    unpreferred_exercise_ids: List[str] = Field(default_factory=list)
    profile_signals: List[str] = Field(default_factory=list)
    mobility_limits: List[MobilityLimitCode] = Field(default_factory=list)
    injury_conditions: List[InjuryConditionEntry] = Field(default_factory=list)
    shoulder_condition_status: Optional[ShoulderConditionStatus] = None
    restricted_movement_patterns: List[ShoulderRestrictedMovementPattern] = Field(
        default_factory=list
    )
    condition_policies: List[Dict[str, Any]] = Field(default_factory=list)  # 질환/수술/재활 정책 입력
    anthropometry_signals: Dict[str, Any] = Field(default_factory=dict)     # 체형 민감도 입력

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_advanced_chest_focus(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        migrated = dict(value)
        legacy = migrated.get(
            "advancedChestFocus",
            migrated.get("advanced_chest_focus", "balanced"),
        )
        legacy_focus_map = {
            "upperChest": "upperChest",
            "lowerChest": "lowerChest",
        }
        legacy_method_map = {"pausedPress": "pausedReps"}
        legacy_focus = legacy_focus_map.get(str(legacy or "balanced"))
        legacy_method = legacy_method_map.get(str(legacy or "balanced"))

        current_focus = migrated.get(
            "specializationFocus",
            migrated.get("specialization_focus"),
        )
        requested_method = migrated.get(
            "requestedMethod",
            migrated.get("requested_method"),
        )
        legacy_training_method = migrated.get(
            "trainingMethod",
            migrated.get("training_method"),
        )
        if (
            requested_method
            and legacy_training_method
            and requested_method != legacy_training_method
        ):
            raise ValueError("requestedMethod conflicts with legacy trainingMethod")
        current_method = requested_method or legacy_training_method
        if legacy_focus and current_focus and current_focus != legacy_focus:
            raise ValueError(
                "advancedChestFocus conflicts with specializationFocus"
            )
        if legacy_method and current_method and current_method != legacy_method:
            raise ValueError(
                "advancedChestFocus conflicts with trainingMethod"
            )
        if legacy_focus and not current_focus:
            migrated["specialization_focus"] = legacy_focus
        if legacy_method and not current_method:
            current_method = legacy_method
        if current_method:
            migrated["requested_method"] = current_method
            migrated["training_method"] = current_method
            if "methodMode" not in migrated and "method_mode" not in migrated:
                migrated["method_mode"] = "manual"
        return migrated

    @field_validator("mobility_limits")
    @classmethod
    def deduplicate_mobility_codes(cls, values):
        return list(dict.fromkeys(values))

    @field_validator("adjunct_options")
    @classmethod
    def validate_adjunct_options(cls, values):
        deduplicated = list(canonical_adjunct_options(values))
        conflict_code = adjunct_option_conflict_code(deduplicated)
        if conflict_code == "too_many_adjunct_options":
            raise ValueError(
                f"adjunctOptions accepts at most {MAX_DISTINCT_ADJUNCT_OPTIONS} distinct options"
            )
        if conflict_code == "conditioning_and_endurance_overlap":
            raise ValueError(
                "conditioningFinisher and enduranceSupport cannot be requested together"
            )
        return deduplicated

    @field_validator("recent_training_method_uses")
    @classmethod
    def validate_recent_training_method_uses(cls, values):
        if len(values) > 12:
            raise ValueError("recentTrainingMethodUses accepts at most 12 entries")
        keys = [
            (entry.method_code, entry.target_split_label, entry.sessions_ago)
            for entry in values
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("recentTrainingMethodUses must not contain duplicates")
        return values

    @field_validator("target_muscles")
    @classmethod
    def validate_target_muscle_codes(cls, values):
        normalized = []
        for raw_value in values:
            value = str(raw_value or "").strip()
            if value not in MUSCLE_SLUGS:
                raise ValueError("targetMuscles must use locked public trainable-muscle slugs")
            normalized.append(value)
        return list(dict.fromkeys(normalized))

    @field_validator("pain_areas")
    @classmethod
    def validate_pain_area_codes(cls, values):
        for entry in values:
            if entry.body_part not in PUBLIC_PAIN_AREA_SLUGS:
                raise ValueError("painAreas bodyPart must use a locked public pain-area slug")
        return values

    @field_validator("injury_conditions")
    @classmethod
    def validate_and_normalize_injury_conditions(cls, values):
        area_codes = [entry.area_code for entry in values]
        if len(area_codes) != len(set(area_codes)):
            raise ValueError("injuryConditions must contain at most one entry per areaCode")
        # Recovery is a profile/history transition, not an active generation
        # constraint. Accept it for adapter compatibility, then remove it from
        # the runtime request so the candidate ranker and LLM see the same
        # state as a user with no current restriction.
        return [entry for entry in values if entry.status != "recovered"]

    @field_validator("profile_signals", mode="before")
    @classmethod
    def validate_profile_signals(cls, values):
        return list(resolve_profile_constraint_signals(values).selected_signals)

    @field_validator("doms_data", mode="before")
    @classmethod
    def validate_doms_levels(cls, values):
        if not isinstance(values, dict):
            raise ValueError("domsData must be an object keyed by body-part slug")
        normalized: Dict[str, int] = {}
        for raw_body_part, raw_level in values.items():
            body_part = str(raw_body_part or "").strip()
            if body_part not in PUBLIC_DOMS_BODY_PART_SLUGS:
                raise ValueError("domsData keys must use locked muscle-soreness slugs")
            if isinstance(raw_level, bool) or not isinstance(raw_level, int) or raw_level not in {1, 2, 3}:
                raise ValueError("domsData levels must be integers from 1 to 3")
            normalized[body_part] = raw_level
        return normalized

    @field_validator("condition_policies")
    @classmethod
    def validate_condition_clearance_status(cls, policies):
        for policy in policies:
            if not isinstance(policy, dict):
                raise ValueError("conditionPolicies entries must be objects")
            raw_status = policy.get("clearanceStatus", policy.get("clearance_status"))
            if raw_status is None:
                continue
            normalized = str(raw_status).strip().replace("-", "_").replace(" ", "_").lower()
            compact = normalized.replace("_", "")
            if normalized not in _CLEARANCE_STATUS_ALIASES and compact not in _CLEARANCE_STATUS_ALIASES:
                raise ValueError(
                    "clearanceStatus must be one of notNeeded, cleared, "
                    "clearedWithRestrictions, notCleared, unknown"
                )
        return policies

    @model_validator(mode="after")
    def validate_option_combinations(self):
        self.restricted_movement_patterns = list(
            validate_shoulder_condition_combination(
                self.shoulder_condition_status,
                self.restricted_movement_patterns,
            )
        )
        resolved_profile = resolve_profile_constraint_signals(self.profile_signals)
        self.mobility_limits = list(dict.fromkeys([
            *self.mobility_limits,
            *resolved_profile.mobility_limit_aliases,
        ]))

        anthropometry = dict(self.anthropometry_signals or {})
        existing_signals = anthropometry.get("signals", [])
        if existing_signals is None:
            existing_signals = []
        if not isinstance(existing_signals, list) or any(
            not isinstance(value, str) or not value.strip()
            for value in existing_signals
        ):
            raise ValueError("anthropometrySignals.signals must be an array of non-empty strings")
        if "signals" in anthropometry or existing_signals or resolved_profile.anthropometry_signal_aliases:
            reviewed_profile_aliases = [
                value
                for value in resolved_profile.anthropometry_signal_aliases
                if value in _REVIEWED_RUNTIME_ANTHROPOMETRY_ALIASES
            ]
            anthropometry["signals"] = list(dict.fromkeys([
                *(value.strip() for value in existing_signals),
                *reviewed_profile_aliases,
            ]))
            if not anthropometry["signals"]:
                anthropometry.pop("signals")
        self.anthropometry_signals = anthropometry

        high_density_structures = {"circuit", "density"}
        high_density_adjuncts = {"conditioning_finisher", "endurance_support"}
        if self.readiness_level == "low" and self.session_structure in high_density_structures:
            raise ValueError(
                "low readiness cannot be combined with circuit or density; use standard or superset"
            )
        if self.readiness_level == "low" and high_density_adjuncts.intersection(self.adjunct_options):
            raise ValueError(
                "low readiness cannot be combined with conditioning or endurance adjuncts"
            )
        if str(self.goal or "").strip().lower() == "strength" and self.session_structure == "circuit":
            raise ValueError("strength goal cannot use circuit session structure")
        if (
            self.specialization_focus
            and self.experience_level not in {None, "advanced"}
        ):
            raise ValueError(
                "specializationFocus is available to advanced users only"
            )
        selected_method = self.requested_method or self.training_method
        if (
            self.requested_method
            and self.training_method
            and self.requested_method != self.training_method
        ):
            raise ValueError("requestedMethod conflicts with legacy trainingMethod")
        if self.method_mode == "off" and selected_method:
            raise ValueError("methodMode off cannot include requestedMethod")
        if self.method_mode == "manual" and not selected_method:
            raise ValueError("methodMode manual requires requestedMethod")
        if self.method_mode == "auto" and selected_method:
            raise ValueError(
                "methodMode auto cannot include requestedMethod; use manual mode"
            )
        if selected_method and self.experience_level not in {None, "advanced"}:
            raise ValueError(
                "requestedMethod requires advanced experience level"
            )
        if selected_method and self.readiness_level == "low":
            raise ValueError(
                "requestedMethod cannot be combined with low readiness"
            )
        normalized_goal = str(self.goal or "").strip().lower().replace("_", "")
        method_goals = {
            "pausedReps": {"strength"},
            "tempoEccentric": {"hypertrophy", "recomposition"},
        }
        if (
            selected_method
            and normalized_goal
            and normalized_goal not in method_goals[selected_method]
        ):
            raise ValueError(
                "requestedMethod is not supported for the selected goal"
            )
        self.requested_method = selected_method
        self.training_method = selected_method
        return self


# ==========================================
# DB 컨텍스트 모델 (내부)
# ==========================================

class RecentSetRecord(BaseModel):
    exercise_id: Optional[str] = None
    exercise_name: str
    primary_muscle: Optional[str] = None
    weight_kg: Optional[float] = None
    reps: int
    rir: Optional[float] = None
    rpe: Optional[float] = None
    is_failure: bool = False


class UserProfileContext(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    goal_type: str
    split_type: str
    split_label: Optional[str] = None
    experience_level: Optional[str] = None
    strength_baseline: Dict[str, Any] = Field(default_factory=dict)
    equipment_access: List[str] = Field(default_factory=list)
    pain_areas: List[PainAreaEntry] = Field(default_factory=list)
    injury_conditions: List[InjuryConditionEntry] = Field(default_factory=list)
    shoulder_condition_status: Optional[ShoulderConditionStatus] = None
    restricted_movement_patterns: List[ShoulderRestrictedMovementPattern] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_shoulder_condition(self):
        area_codes = [entry.area_code for entry in self.injury_conditions]
        if len(area_codes) != len(set(area_codes)):
            raise ValueError("injuryConditions must contain at most one entry per areaCode")
        self.restricted_movement_patterns = list(
            validate_shoulder_condition_combination(
                self.shoulder_condition_status,
                self.restricted_movement_patterns,
            )
        )
        return self


# ==========================================
# LLM 구조화 출력 모델 (내부)
# ==========================================

class LLMSubstitutionCandidate(BaseModel):
    exercise_id: str = Field(description="대체 운동 ID (snake_case)")
    exercise_name: str = Field(description="대체 운동 이름")
    reason: str = Field(description="대체 이유")


class LLMExercisePlan(BaseModel):
    _unilateral_prescription: Optional[UnilateralPrescription] = PrivateAttr(default=None)
    _unilateral_set_overrides: Optional[List[Dict[str, Any]]] = PrivateAttr(default=None)
    _prescription_set_type: str = PrivateAttr(default="working")
    _target_duration_sec: Optional[int] = PrivateAttr(default=None)
    _interval_work_sec: Optional[int] = PrivateAttr(default=None)
    _interval_recovery_sec: Optional[int] = PrivateAttr(default=None)
    _interval_rounds: Optional[int] = PrivateAttr(default=None)
    _target_distance_m: Optional[int] = PrivateAttr(default=None)
    _training_method_code: Optional[str] = PrivateAttr(default=None)
    _training_method_instruction: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    _weight_guidance: Optional[str] = PrivateAttr(default=None)

    exercise_id: str = Field(description="운동 ID — 반드시 운동 목록의 ID 그대로 사용")
    exercise_name: str = Field(description="운동 이름 (한글)")
    movement_pattern: Optional[str] = Field(default=None, description="동작 패턴 (예: horizontalPush)")
    movement_type: Optional[str] = Field(default=None, description="운동 유형 (COMPOUND | ISOLATION | STATIC)")
    primary_muscles: List[str] = Field(default_factory=list, description="주동근 목록")
    equipment_type: Optional[str] = Field(default=None, description="사용 장비 (예: barbell)")
    target_weight_kg: Optional[float] = Field(default=None, description="목표 중량(kg). 맨몸이면 null")
    target_reps: int = Field(description="세트당 반복 횟수")
    sets: int = Field(description="총 세트 수")
    rest_time_sec: int = Field(description="세트 간 휴식 시간(초)")
    target_rir: Optional[int] = Field(default=2, description="목표 RIR (Reps In Reserve)")
    exercise_rationale: str = Field(description="이 운동을 선택한 이유")
    substitution_candidates: List[LLMSubstitutionCandidate] = Field(
        default_factory=list, description="대체 운동 후보 (최대 2개)"
    )


class LLMRoutineOutput(BaseModel):
    _selection_diagnostics: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _selection_phase_by_exercise_id: Dict[str, str] = PrivateAttr(default_factory=dict)
    _selection_slot_by_exercise_id: Dict[str, str] = PrivateAttr(default_factory=dict)
    _adjunct_option_by_exercise_id: Dict[str, str] = PrivateAttr(default_factory=dict)
    _adjunct_suppression_reason_by_option: Dict[str, str] = PrivateAttr(default_factory=dict)
    _execution_metadata_by_exercise_id: Dict[str, Dict[str, Any]] = PrivateAttr(default_factory=dict)

    total_estimated_time: int = Field(description="루틴 예상 소요 시간(분)")
    summary_title: str = Field(description="루틴 요약 제목 (예: '다음 push 세션 추천')")
    rationale_summary: List[str] = Field(description="루틴 생성 근거 요약 (2~3개 항목)")
    exercises: List[LLMExercisePlan]
    warnings: List[str] = Field(default_factory=list, description="주의사항 목록")


class LLMSlotSelection(BaseModel):
    """Small-model internal selection contract; prescription stays server-owned."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    slot_id: str = Field(description="deterministic blueprint slot ID")
    exercise_id: str = Field(description="exercise ID from the matching slot candidate bundle")
    exercise_rationale: str = Field(default="", description="짧은 한국어 선택 이유")


class LLMRoutineSelectionOutput(BaseModel):
    """Internal v2 output used before adapting to the stable routine pipeline."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    summary_title: str = Field(default="맞춤 운동 루틴")
    rationale_summary: List[str] = Field(default_factory=list)
    selections: List[LLMSlotSelection] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


# ==========================================
# Response 모델 (프론트엔드 ↔ API)
# ==========================================

class SetPrescription(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    set_index: int = Field(ge=1)
    set_type: PrescriptionSetType = "working"
    target_reps: int = Field(ge=1)
    target_weight_kg: Optional[float] = None
    target_rir: Optional[int] = 2
    target_rest_sec: int = Field(ge=0)
    target_duration_sec: Optional[int] = Field(
        default=None,
        ge=1,
        exclude_if=lambda value: value is None,
    )
    interval_work_sec: Optional[int] = Field(
        default=None,
        ge=1,
        exclude_if=lambda value: value is None,
    )
    interval_recovery_sec: Optional[int] = Field(
        default=None,
        ge=0,
        exclude_if=lambda value: value is None,
    )
    interval_rounds: Optional[int] = Field(
        default=None,
        ge=1,
        exclude_if=lambda value: value is None,
    )
    target_distance_m: Optional[int] = Field(
        default=None,
        ge=1,
        exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def validate_modality_fields(self):
        if self.set_type == "timed" and self.target_duration_sec is None:
            raise ValueError("timed prescription requires targetDurationSec")
        if self.set_type == "interval" and any(
            value is None
            for value in (
                self.target_duration_sec,
                self.interval_work_sec,
                self.interval_recovery_sec,
                self.interval_rounds,
            )
        ):
            raise ValueError(
                "interval prescription requires duration, work, recovery, and rounds"
            )
        if self.set_type == "distance" and self.target_distance_m is None:
            raise ValueError("distance prescription requires targetDistanceM")
        return self


class SubstitutionCandidate(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    exercise_id: str
    exercise_name: str
    reason: str


class ScoreBreakdownItem(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    constraint_code: Optional[str] = None
    rule_code: Optional[str] = None
    profile_signal_code: Optional[str] = None
    score: int
    reason: str


class TrainingMethodInstruction(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    code: TrainingMethod
    label_ko: str
    purpose_ko: str
    instruction_ko: str
    pause_sec: Optional[int] = Field(default=None, ge=1)
    eccentric_sec: Optional[int] = Field(default=None, ge=1)
    source: Literal["auto", "manual", "legacy"] = "manual"
    reason_code: Optional[str] = None


class RoutineBlock(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    _weight_guidance: Optional[str] = PrivateAttr(default=None)
    unilateral_prescription: Optional[UnilateralPrescription] = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    order: int
    exercise_id: str
    exercise_name: str
    movement_pattern: Optional[str] = None
    movement_type: Optional[str] = None
    primary_muscles: List[str] = Field(default_factory=list)
    primary_muscle_details: List[str] = Field(default_factory=list, exclude_if=lambda value: not value)
    equipment_type: Optional[str] = None
    default_rest_sec: int
    prescription: List[SetPrescription]
    exercise_rationale: str
    substitution_candidates: List[SubstitutionCandidate] = Field(default_factory=list)
    reasons: List[str] = Field(default_factory=list)
    boosts: List[ScoreBreakdownItem] = Field(default_factory=list)
    penalties: List[ScoreBreakdownItem] = Field(default_factory=list)
    session_phase: Optional[str] = None
    execution_group_id: Optional[str] = None
    execution_group_type: str = "single"
    execution_group_position: int = 1
    execution_group_size: int = 1
    transition_rest_sec: Optional[int] = None
    group_rest_sec: Optional[int] = None
    adjunct_option: Optional[AdjunctOption] = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    training_method: Optional[TrainingMethodInstruction] = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    @model_validator(mode="before")
    @classmethod
    def restore_internal_primary_muscles(cls, value):
        if isinstance(value, dict) and "primaryMuscleDetails" in value:
            value = dict(value)
            muscles = value.get("primaryMuscles", [])
            details = value["primaryMuscleDetails"]
            if (isinstance(muscles, list) and isinstance(details, list)
                    and all(isinstance(m, str) for m in muscles + details)):
                value["primaryMuscles"] = list(dict.fromkeys(muscles + details))
        return value

    @model_validator(mode='after')
    def validate_unilateral_prescription(self):
        if self.unilateral_prescription is not None:
            expected = 'working' if self.unilateral_prescription.quantity_mode == 'repsPerSide' else 'timed'
            if any(s.set_type != expected for s in self.prescription):
                raise ValueError('Unilateral quantity mode must match every prescribed set')
            if expected == 'timed' and self.training_method is not None:
                raise ValueError('Timed unilateral work cannot include repetition methods')
        return self

    @model_serializer(mode="wrap")
    def serialize_public_muscles(self, handler, info):
        data = handler(self)
        # Snake-case dumps remain internal. Camel-case transport never expands
        # the locked public enum; side-deltoids has no honest public equivalent.
        if info.by_alias and "primaryMuscles" in data:
            data["primaryMuscles"] = [m for m in self.primary_muscles if m in MUSCLE_SLUGS]
            details = list(dict.fromkeys(self.primary_muscle_details +
                [m for m in self.primary_muscles if m not in MUSCLE_SLUGS]))
            if details:
                data["primaryMuscleDetails"] = details
        return data


class AdjunctDecision(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    option: AdjunctOption
    status: AdjunctDecisionStatus
    reason_code: AdjunctDecisionReasonCode
    slot_id: Optional[str] = Field(default=None, exclude_if=lambda value: value is None)
    exercise_id: Optional[str] = Field(default=None, exclude_if=lambda value: value is None)


class PersonalizationSuggestion(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    suggestion_type: Literal["specializationFocus"]
    code: SpecializationFocus
    reason_ko: str
    evidence_set_count: int = Field(ge=0)
    comparison_max_set_count: int = Field(ge=0)
    requires_confirmation: bool = True


class RoutineDraftResponse(BaseModel):
    _underfill_diagnostics: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _fallback_selection_diagnostics: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _adjunct_suppression_reason_by_option: Dict[str, str] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    routine_draft_id: str = Field(default_factory=lambda: str(uuid4()))
    generation_status: GenerationStatus
    status_reason_code: StatusReasonCode
    is_fallback: bool
    total_estimated_time: int = 0
    summary_title: str
    rationale_summary: List[str]
    routine_blocks: List[RoutineBlock]
    warnings: List[str] = Field(default_factory=list)
    adjunct_decisions: List[AdjunctDecision] = Field(
        default_factory=list,
        exclude_if=lambda value: not value,
    )
    personalization_suggestions: List[PersonalizationSuggestion] = Field(
        default_factory=list,
        exclude_if=lambda value: not value,
    )


EditReason = Literal[
    "too_heavy",
    "too_easy",
    "pain",
    "no_equipment",
    "dislike",
    "duplicate",
    "other",
]


class EditedExercise(BaseModel):
    original_exercise_id: str = Field(..., min_length=1, max_length=128)
    replacement_exercise_id: Optional[str] = Field(default=None, max_length=128)
    reason: EditReason

    @field_validator("original_exercise_id", "replacement_exercise_id", mode="before")
    @classmethod
    def strip_exercise_id(cls, value):
        if value is None:
            return None
        return str(value).strip()


class RoutineFeedbackRequest(BaseModel):
    routine_draft_id: str = Field(..., min_length=1, max_length=128)
    user_id: Optional[str] = Field(default=None, max_length=128)
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    completed: Optional[bool] = None
    accepted_without_edits: Optional[bool] = None
    skipped_exercises: List[str] = Field(default_factory=list)
    edited_exercises: List[EditedExercise] = Field(default_factory=list)
    user_note: Optional[str] = None

    @field_validator("routine_draft_id", "user_id", mode="before")
    @classmethod
    def strip_optional_text(cls, value):
        if value is None:
            return None
        return str(value).strip()

    @field_validator("user_note", mode="before")
    @classmethod
    def strip_user_note(cls, value):
        if value is None:
            return None
        stripped = str(value).strip()
        if not stripped:
            return None
        if len(stripped) > 500:
            raise ValueError("user_note must be at most 500 characters after strip")
        return stripped

    @field_validator("skipped_exercises")
    @classmethod
    def validate_skipped_exercises(cls, value):
        stripped = [item.strip() for item in value]
        if any(not item for item in stripped):
            raise ValueError("skipped_exercises must contain non-empty exercise_id strings")
        if any(len(item) > 128 for item in stripped):
            raise ValueError("skipped_exercises exercise_id values must be at most 128 characters")
        return stripped

    @model_validator(mode="after")
    def at_least_one_feedback_field(self) -> "RoutineFeedbackRequest":
        has_payload = any([
            self.rating is not None,
            self.completed is not None,
            self.accepted_without_edits is not None,
            bool(self.skipped_exercises),
            bool(self.edited_exercises),
            bool(self.user_note),
        ])
        if not has_payload:
            raise ValueError(
                "At least one of rating, completed, accepted_without_edits, "
                "skipped_exercises, edited_exercises, user_note must be provided."
            )
        return self


class RoutineFeedbackResponse(BaseModel):
    ok: bool = True
    feedback_id: str
    stored_at: str

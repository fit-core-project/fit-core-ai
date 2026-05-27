"""루틴 엔진 Pydantic 모델 정의 (Request / Response / LLM I/O / Context)."""
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict, AliasChoices, field_validator, model_validator
from pydantic.alias_generators import to_camel

from engines.llm_router import StatusReasonCode

GenerationStatus = Literal["success", "fallback", "failed"]
# StatusReasonCode는 engines.llm_router에서 단일 정의 후 re-export됨

# ==========================================
# Request 모델
# ==========================================

class DomEntry(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    body_part: str
    level: str  # "mild" | "moderate" | "severe"


class PainAreaEntry(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    body_part: str = Field(validation_alias=AliasChoices("area", "bodyPart", "body_part"))
    side: Optional[str] = None
    severity: Optional[str] = None


class RoutineRequest(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    user_id: Optional[str] = None
    target_split_label: Optional[str] = None         # "push" | "pull" | "legs" | ... (없으면 target_muscles 직접 사용)
    target_muscles: List[str] = Field(default_factory=list)  # spreadsheet slug 직접 지정 시 사용
    readiness_level: Optional[str] = "normal"
    time_available_min: int
    pain_areas: List[PainAreaEntry] = Field(default_factory=list)  # 장기 부상 부위 객체 배열
    doms_data: Dict[str, int] = Field(default_factory=dict)   # Java가 매핑한 {spreadsheet_slug: level(1~3)}
    equipment: List[str] = Field(default_factory=list)         # 사용 불가 장비 블랙리스트
    goal: Optional[str] = None                                 # 미전달 시 프로필의 goal_type 사용
    user_note: Optional[str] = None
    preferred_exercise_ids: List[str] = Field(default_factory=list)
    unpreferred_exercise_ids: List[str] = Field(default_factory=list)


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
    goal_type: str
    split_type: str
    split_label: Optional[str] = None
    experience_level: Optional[str] = None
    strength_baseline: Dict[str, Any] = Field(default_factory=dict)
    equipment_access: List[str] = Field(default_factory=list)
    pain_areas: List[PainAreaEntry] = Field(default_factory=list)


# ==========================================
# LLM 구조화 출력 모델 (내부)
# ==========================================

class LLMSubstitutionCandidate(BaseModel):
    exercise_id: str = Field(description="대체 운동 ID (snake_case)")
    exercise_name: str = Field(description="대체 운동 이름")
    reason: str = Field(description="대체 이유")


class LLMExercisePlan(BaseModel):
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
    total_estimated_time: int = Field(description="루틴 예상 소요 시간(분)")
    summary_title: str = Field(description="루틴 요약 제목 (예: '다음 push 세션 추천')")
    rationale_summary: List[str] = Field(description="루틴 생성 근거 요약 (2~3개 항목)")
    exercises: List[LLMExercisePlan]
    warnings: List[str] = Field(default_factory=list, description="주의사항 목록")


# ==========================================
# Response 모델 (프론트엔드 ↔ API)
# ==========================================

class SetPrescription(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    set_index: int
    set_type: str = "working"
    target_reps: int
    target_weight_kg: Optional[float] = None
    target_rir: Optional[int] = 2
    target_rest_sec: int


class SubstitutionCandidate(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    exercise_id: str
    exercise_name: str
    reason: str


class RoutineBlock(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    order: int
    exercise_id: str
    exercise_name: str
    movement_pattern: Optional[str] = None
    primary_muscles: List[str] = Field(default_factory=list)
    equipment_type: Optional[str] = None
    default_rest_sec: int
    prescription: List[SetPrescription]
    exercise_rationale: str
    substitution_candidates: List[SubstitutionCandidate] = Field(default_factory=list)


class RoutineDraftResponse(BaseModel):
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

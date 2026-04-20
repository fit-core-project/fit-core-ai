import re
import json
import math
import asyncio
import httpx
from typing import Any, List, Optional, Dict, Literal
from collections import defaultdict
from uuid import uuid4
from pydantic import BaseModel, Field, ConfigDict, ValidationError
from pydantic.alias_generators import to_camel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from sqlalchemy.orm import Session
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. 공통 타입 정의
# ==========================================

GenerationStatus = Literal["success", "fallback", "failed"]
StatusReasonCode = Literal["none", "llmTimeout", "schemaError", "networkError"]

DOMS_LEVEL_MAP: Dict[str, int] = {
    "mild": 1,
    "moderate": 2,
    "severe": 3,
}

# ==========================================
# 2. Request 모델
# ==========================================

class DomEntry(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    body_part: str
    level: str  # "mild" | "moderate" | "severe"


class RoutineRequest(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    user_id: Optional[str] = None
    target_split_label: str                          # "push" | "pull" | "legs" | ...
    readiness_level: Optional[str] = "normal"
    time_available_min: int
    current_pain_areas: List[str] = Field(default_factory=list)
    doms: List[DomEntry] = Field(default_factory=list)
    unavailable_equipment: List[str] = Field(default_factory=list)
    goal: Optional[str] = None                       # 미전달 시 프로필의 goal_type 사용
    user_note: Optional[str] = None


# ==========================================
# 3. DB 컨텍스트 모델 (내부)
# ==========================================

class RecentSetRecord(BaseModel):
    exercise_name: str
    weight_kg: Optional[float] = None
    reps: int


class UserProfileContext(BaseModel):
    goal_type: str
    split_type: str
    split_label: Optional[str] = None
    experience_level: Optional[str] = None
    strength_baseline: Dict[str, Any] = Field(default_factory=dict)
    equipment_access: List[str] = Field(default_factory=list)
    pain_areas: List[str] = Field(default_factory=list)


# ==========================================
# 4. LLM 구조화 출력 모델 (내부)
# ==========================================

class LLMSubstitutionCandidate(BaseModel):
    exercise_id: str = Field(description="대체 운동 ID (snake_case)")
    exercise_name: str = Field(description="대체 운동 이름")
    reason: str = Field(description="대체 이유")


class LLMExercisePlan(BaseModel):
    exercise_id: str = Field(description="운동 ID — 반드시 운동 목록의 ID 그대로 사용")
    exercise_name: str = Field(description="운동 이름 (한글)")
    movement_pattern: Optional[str] = Field(default=None, description="동작 패턴 (예: horizontalPush)")
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
# 5. Response 모델 (프론트엔드 ↔ API)
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
    summary_title: str
    rationale_summary: List[str]
    routine_blocks: List[RoutineBlock]
    warnings: List[str] = Field(default_factory=list)


# ==========================================
# 6. Split Label → DB 근육 Enum 매핑
# ==========================================

SPLIT_LABEL_TO_MUSCLES: Dict[str, List[str]] = {
    "push":      ["CHEST_UPPER", "CHEST_MID", "CHEST_LOWER", "SHOULDER_FRONT", "SHOULDER_SIDE", "ARM_TRICEPS"],
    "pull":      ["BACK_UPPER", "BACK_LATS", "BACK_LOWER", "ARM_BICEPS", "ARM_FOREARMS"],
    "legs":      ["LEG_QUADS", "LEG_HAMSTRINGS", "LEG_GLUTES", "LEG_CALVES", "LEG_ADDUCTORS"],
    "upper":     ["CHEST_UPPER", "CHEST_MID", "BACK_UPPER", "BACK_LATS", "SHOULDER_FRONT", "SHOULDER_SIDE", "ARM_BICEPS", "ARM_TRICEPS"],
    "lower":     ["LEG_QUADS", "LEG_HAMSTRINGS", "LEG_GLUTES", "LEG_CALVES", "LEG_ADDUCTORS", "LEG_ABDUCTORS"],
    "chest":     ["CHEST_UPPER", "CHEST_MID", "CHEST_LOWER"],
    "back":      ["BACK_UPPER", "BACK_LATS", "BACK_LOWER"],
    "shoulder":  ["SHOULDER_FRONT", "SHOULDER_SIDE", "SHOULDER_REAR"],
    "arm":       ["ARM_BICEPS", "ARM_TRICEPS", "ARM_FOREARMS"],
    "core":      ["CORE_ABS", "CORE_OBLIQUES"],
    "full_body": ["CHEST_UPPER", "BACK_LATS", "SHOULDER_FRONT", "LEG_QUADS", "LEG_HAMSTRINGS", "CORE_ABS"],
}

# react-body-highlighter 키 → DB Enum (DOMS bodyPart 역매핑용)
FRONTEND_TO_DB_MAP: Dict[str, List[str]] = {
    "chest":          ["CHEST_UPPER", "CHEST_MID", "CHEST_LOWER"],
    "upper-back":     ["BACK_UPPER"],
    "trapezius":      ["BACK_UPPER"],
    "lats":           ["BACK_LATS"],
    "lower-back":     ["BACK_LOWER"],
    "front-deltoids": ["SHOULDER_FRONT"],
    "back-deltoids":  ["SHOULDER_REAR"],
    "deltoids":       ["SHOULDER_FRONT", "SHOULDER_REAR", "SHOULDER_SIDE"],
    "side-deltoids":  ["SHOULDER_SIDE"],
    "biceps":         ["ARM_BICEPS"],
    "triceps":        ["ARM_TRICEPS"],
    "forearm":        ["ARM_FOREARMS"],
    "abs":            ["CORE_ABS"],
    "obliques":       ["CORE_OBLIQUES"],
    "glutes":         ["LEG_GLUTES"],
    "hamstring":      ["LEG_HAMSTRINGS"],
    "quadriceps":     ["LEG_QUADS"],
    "calves":         ["LEG_CALVES"],
    "adductors":      ["LEG_ADDUCTORS"],
    "abductors":      ["LEG_ABDUCTORS"],
    "knees":          ["LEG_QUADS", "LEG_HAMSTRINGS"],
    "neck":           ["NECK"],
    "head":           [],
}


def split_label_to_muscles(label: str) -> List[str]:
    """targetSplitLabel("push" 등)을 DB 근육 Enum 리스트로 변환한다."""
    return SPLIT_LABEL_TO_MUSCLES.get(label.lower(), [])


def map_doms_to_db(doms: List[DomEntry]) -> Dict[str, int]:
    """DomEntry 배열을 {DB_MUSCLE_ENUM: level_int} 딕셔너리로 변환한다."""
    result: Dict[str, int] = {}
    for entry in doms:
        level_int = DOMS_LEVEL_MAP.get(entry.level.lower(), 1)
        mapped = FRONTEND_TO_DB_MAP.get(entry.body_part.lower())
        targets = mapped if mapped is not None else [entry.body_part]
        for db_muscle in targets:
            result[db_muscle] = max(result.get(db_muscle, 0), level_int)
    return result


# ==========================================
# 7. DB 조회 함수
# ==========================================

def get_user_profile_context(db: Session, user_id: str) -> Optional[UserProfileContext]:
    import json

    row = db.execute(
        text("""
            SELECT goal_type, split_type, split_label, experience_level,
                   strength_baseline, equipment_access, pain_areas
            FROM user_profiles
            WHERE user_id = :uid
        """),
        {"uid": user_id}
    ).fetchone()

    if row is None:
        return None

    r = dict(row._mapping)

    def _parse(val):
        if isinstance(val, str):
            return json.loads(val)
        return val if val is not None else {}

    def _parse_list(val):
        parsed = _parse(val)
        return parsed if isinstance(parsed, list) else []

    return UserProfileContext(
        goal_type=r["goal_type"],
        split_type=r["split_type"],
        split_label=r.get("split_label"),
        experience_level=r.get("experience_level"),
        strength_baseline=_parse(r.get("strength_baseline")),
        equipment_access=_parse_list(r.get("equipment_access")),
        pain_areas=_parse_list(r.get("pain_areas")),
    )


def get_recent_sets(db: Session, user_id: str, limit: int = 30) -> List[RecentSetRecord]:
    """최근 3회 세션의 working 세트를 최신 순으로 반환한다."""
    session_rows = db.execute(
        text("""
            SELECT workout_session_id
            FROM workout_sessions
            WHERE user_id = :uid
            ORDER BY workout_date DESC
            LIMIT 3
        """),
        {"uid": user_id}
    ).fetchall()

    if not session_rows:
        return []

    session_ids = [r[0] for r in session_rows]
    sid_placeholders = ", ".join([f":sid{i}" for i in range(len(session_ids))])
    params: dict = {f"sid{i}": sid for i, sid in enumerate(session_ids)}
    params["limit"] = limit

    rows = db.execute(
        text(f"""
            SELECT exercise_name_snapshot, weight_kg, reps
            FROM workout_sets
            WHERE workout_session_id IN ({sid_placeholders})
              AND set_type = 'working'
            ORDER BY created_at DESC
            LIMIT :limit
        """),
        params
    ).fetchall()

    return [
        RecentSetRecord(
            exercise_name=r[0],
            weight_kg=float(r[1]) if r[1] is not None else None,
            reps=int(r[2]),
        )
        for r in rows
    ]


def get_candidate_exercises(
    db: Session,
    target_muscles: List[str],
    unavailable_equipment: List[str],
    pain_areas: List[str],
) -> List[dict]:
    """
    근육 부위 필터 + 통증 부위 제외 + 사용 불가 장비 블랙리스트 필터.
    - unavailable_equipment: 블랙리스트 방식 (기존 화이트리스트에서 변경)
    - BODYWEIGHT는 unavailable_equipment 여부와 무관하게 항상 허용
    """
    if not target_muscles:
        return []

    muscle_placeholders = ", ".join([f":m{i}" for i in range(len(target_muscles))])
    params: dict = {f"m{i}": m for i, m in enumerate(target_muscles)}

    if pain_areas:
        not_like_parts = " AND ".join(
            [f"pain_triggers NOT LIKE :p{i}" for i in range(len(pain_areas))]
        )
        pain_filter = f"AND (pain_triggers IS NULL OR ({not_like_parts}))"
        for i, pain in enumerate(pain_areas):
            params[f"p{i}"] = f"%{pain}%"
    else:
        pain_filter = ""

    query = text(f"""
        SELECT id, name_kr, name_en, primary_muscle, secondary_muscle,
               equipment_req, difficulty_tier, efficiency_tier,
               pain_triggers, movement_type
        FROM exercise_tier
        WHERE primary_muscle IN ({muscle_placeholders})
        {pain_filter}
        ORDER BY efficiency_tier DESC
    """)

    rows = db.execute(query, params).fetchall()

    blocked = {e.strip().upper() for e in unavailable_equipment}

    candidates = []
    for row in rows:
        row_dict = dict(row._mapping)
        equip_str = row_dict.get("equipment_req")
        if equip_str is None:
            candidates.append(row_dict)
            continue
        exercise_equip = {e.strip().upper() for e in equip_str.split(",")}
        non_bw = exercise_equip - {"BODYWEIGHT"}
        # 비-맨몸 장비가 모두 블랙리스트에 없거나, 맨몸 운동인 경우 포함
        if not non_bw or not (non_bw & blocked):
            candidates.append(row_dict)

    return candidates


# ==========================================
# 후보 목록 → 프롬프트용 문자열
# ==========================================

def format_candidates_for_prompt(candidates: List[dict]) -> str:
    if not candidates:
        return "해당 조건에 맞는 운동이 없습니다."
    lines = [
        f"- {ex['name_kr']} (ID: {ex['id']}, {ex['movement_type']}, efficiency={ex['efficiency_tier']})"
        for ex in candidates
    ]
    return "\n".join(lines)


def _format_strength_baseline(baseline: Dict[str, Any]) -> str:
    lines = []
    for exercise, data in baseline.items():
        if isinstance(data, dict):
            parts = []
            if "weight_kg" in data:
                parts.append(f"{data['weight_kg']}kg")
            if "reps" in data:
                parts.append(f"{data['reps']}회")
            value_str = " × ".join(parts) if parts else str(data)
        else:
            value_str = str(data)
        lines.append(f"- {exercise}: {value_str}")
    return "\n".join(lines)


def _format_recent_sets(sets: List[RecentSetRecord]) -> str:
    groups: Dict[str, List[RecentSetRecord]] = defaultdict(list)
    for s in sets:
        groups[s.exercise_name].append(s)
    lines = []
    for name, records in groups.items():
        set_strs = [
            f"{r.weight_kg}kg×{r.reps}" if r.weight_kg else f"{r.reps}회"
            for r in records[:3]
        ]
        lines.append(f"- {name}: {', '.join(set_strs)}")
    return "\n".join(lines)


def _build_system_prompt(
    profile: Optional[UserProfileContext],
    recent_sets: Optional[List[RecentSetRecord]],
) -> str:
    sections = []

    sections.append(
        "너는 운동 생리학 지식이 풍부한 피트니스 AI 코치야.\n"
        "반드시 아래 [선택 가능한 운동 목록]에 있는 운동만 사용해서 루틴을 구성해.\n\n"
        "[하드 제약 조건]\n"
        "- 훈련 목표: {goal}\n"
        "- 전체 총 세트 수 합계는 {max_sets}세트를 초과하지 말 것\n"
        "- 중량은 {goal} 목표에 맞게 추정할 것 (맨몸 운동은 null)\n"
        "- exercise_id는 반드시 운동 목록의 ID를 그대로 사용할 것"
    )

    if profile:
        lines = ["[유저 프로필]"]
        split_str = profile.split_type + (f" ({profile.split_label})" if profile.split_label else "")
        lines.append(f"- 훈련 목표: {profile.goal_type}")
        lines.append(f"- 분할 방식: {split_str}")
        if profile.experience_level:
            lines.append(f"- 경험 수준: {profile.experience_level}")
        if profile.pain_areas:
            lines.append(f"- 만성 통증 이력: {', '.join(profile.pain_areas)}")
        if profile.strength_baseline:
            lines.append("\n[강도 기준 (strength_baseline) — 중량 추정 시 참고]")
            lines.append(_format_strength_baseline(profile.strength_baseline))
        sections.append("\n".join(lines))

    sections.append("[오늘의 근육통(DOMS) 상태]\n{doms_instructions}")

    if recent_sets:
        history_str = _format_recent_sets(recent_sets)
        if history_str:
            sections.append("[최근 운동 기록 (중량 산정 참고)]\n" + history_str)

    sections.append(
        "[선택 가능한 운동 목록] (efficiency 높은 순, 반드시 이 목록에서만 선택)\n"
        "{candidate_exercises}"
    )

    return "\n\n".join(sections)


# ==========================================
# 8. 핵심 연산
# ==========================================

def calculate_1rm(weight: int, reps: int) -> float:
    """1RM 계산 (Epley 공식)"""
    if reps <= 1:
        return float(weight)
    return weight * (1 + reps / 30.0)


# ==========================================
# 9. Fallback 루틴 (규칙 기반)
# ==========================================

_GOAL_PARAMS = {
    "STRENGTH":    {"sets": 5, "reps": 5,  "rest_sec": 180},
    "HYPERTROPHY": {"sets": 3, "reps": 10, "rest_sec": 90},
    "ENDURANCE":   {"sets": 3, "reps": 15, "rest_sec": 60},
}

_MOVEMENT_PRIORITY = {"COMPOUND": 0, "ISOLATION": 1, "STATIC": 2}

_FALLBACK_TIPS = {
    "COMPOUND":  "주동근에 집중하며 정확한 자세를 유지하세요.",
    "ISOLATION": "목표 근육의 수축과 이완을 의식하며 천천히 수행하세요.",
    "STATIC":    "호흡을 멈추지 말고 코어를 조여 자세를 유지하세요.",
}


def _build_prescription(
    sets: int,
    reps: int,
    weight_kg: Optional[float],
    rest_sec: int,
    target_rir: int = 2,
) -> List[SetPrescription]:
    return [
        SetPrescription(
            set_index=i + 1,
            set_type="working",
            target_reps=reps,
            target_weight_kg=weight_kg,
            target_rir=target_rir,
            target_rest_sec=rest_sec,
        )
        for i in range(sets)
    ]


def _build_routine_draft(
    llm_output: LLMRoutineOutput,
    generation_status: GenerationStatus,
    status_reason_code: StatusReasonCode,
    is_fallback: bool,
) -> RoutineDraftResponse:
    blocks = []
    for order, ex in enumerate(llm_output.exercises, start=1):
        blocks.append(RoutineBlock(
            order=order,
            exercise_id=ex.exercise_id,
            exercise_name=ex.exercise_name,
            movement_pattern=ex.movement_pattern,
            primary_muscles=ex.primary_muscles,
            equipment_type=ex.equipment_type,
            default_rest_sec=ex.rest_time_sec,
            prescription=_build_prescription(
                sets=ex.sets,
                reps=ex.target_reps,
                weight_kg=ex.target_weight_kg,
                rest_sec=ex.rest_time_sec,
                target_rir=ex.target_rir or 2,
            ),
            exercise_rationale=ex.exercise_rationale,
            substitution_candidates=[
                SubstitutionCandidate(
                    exercise_id=s.exercise_id,
                    exercise_name=s.exercise_name,
                    reason=s.reason,
                )
                for s in ex.substitution_candidates
            ],
        ))

    return RoutineDraftResponse(
        generation_status=generation_status,
        status_reason_code=status_reason_code,
        is_fallback=is_fallback,
        summary_title=llm_output.summary_title,
        rationale_summary=llm_output.rationale_summary,
        routine_blocks=blocks,
        warnings=llm_output.warnings,
    )


# ==========================================
# normalize adapter v1
# ==========================================

_MARKDOWN_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```")

def _try_repair_json(text: str) -> dict:
    """
    불완전한 JSON 문자열에 닫는 괄호를 순차적으로 보충해 파싱을 재시도한다.
    마지막 유효 토큰 위치까지만 잘라내는 전처리도 병행한다.
    """
    for suffix in ("}", "}]}", "}]}}", "}]}}]}"):
        try:
            return json.loads(text + suffix)
        except json.JSONDecodeError:
            pass
    # 마지막 완전한 `}` 위치까지만 잘라내어 재시도
    last_brace = text.rfind("}")
    if last_brace != -1:
        try:
            return json.loads(text[: last_brace + 1])
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError("JSON 자가 복구 실패", text, 0)


def _inject_defaults(data: dict) -> dict:
    """
    Pydantic 검증 전 누락된 필수 필드에 기본값을 주입한다 (Graceful Degradation).
    """
    data.setdefault("summary_title", "맞춤형 AI 루틴")
    data.setdefault("rationale_summary", ["회원님의 데이터 기반으로 생성된 루틴입니다."])
    data.setdefault("warnings", [])
    data.setdefault("total_estimated_time", 45)
    data.setdefault("exercises", [])

    for ex in data["exercises"]:
        if not isinstance(ex, dict):
            continue
        ex.setdefault(
            "exercise_id",
            re.sub(r"\s+", "_", ex.get("exercise_name", "unknown")).lower(),
        )
        ex.setdefault("primary_muscles", [])
        ex.setdefault("target_rir", 2)
        ex.setdefault("substitution_candidates", [])
        ex.setdefault("exercise_rationale", "AI 추천 운동")
        ex.setdefault("rest_time_sec", 90)
        ex.setdefault("sets", 3)
        ex.setdefault("target_reps", 10)

    return data


def normalize_llm_response(raw_text: str, llm=None) -> LLMRoutineOutput:
    """
    LLM raw output → LLMRoutineOutput 정제 어댑터.

    처리 순서:
    1. 마크다운 코드펜스(```json ... ```) 제거
    2. json.loads 파싱 시도
    3. 실패 시 자가 복구(_try_repair_json) 재시도
    4. 여전히 실패하고 llm이 주어진 경우 OutputFixingParser로 재시도
    5. 누락 필드 기본값 주입(_inject_defaults)
    6. Pydantic 검증
    """
    cleaned = _MARKDOWN_FENCE_RE.sub("", raw_text).strip()

    # JSON 파싱
    data: dict | None = None
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            data = _try_repair_json(cleaned)
            print("[정제 어댑터] 자가 복구(bracket repair) 성공")
        except json.JSONDecodeError:
            if llm is not None:
                print("[정제 어댑터] OutputFixingParser로 재시도...")
                try:
                    from langchain.output_parsers import OutputFixingParser
                    from langchain_core.output_parsers import PydanticOutputParser

                    base_parser = PydanticOutputParser(pydantic_object=LLMRoutineOutput)
                    fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)
                    return fixing_parser.parse(cleaned)
                except Exception as fix_e:
                    raise ValueError(f"OutputFixingParser 실패: {fix_e}") from fix_e
            raise

    # 기본값 주입 후 Pydantic 검증
    data = _inject_defaults(data)
    return LLMRoutineOutput(**data)


def generate_fallback_routine(
    req: RoutineRequest,
    candidates: List[dict],
    max_total_sets: int,
    doms_db: Optional[Dict[str, int]] = None,
    goal: str = "HYPERTROPHY",
    status_reason_code: StatusReasonCode = "networkError",
) -> RoutineDraftResponse:
    print("[Fallback] 규칙 기반 루틴 생성 시작")

    if not candidates:
        return RoutineDraftResponse(
            generation_status="fallback",
            status_reason_code=status_reason_code,
            is_fallback=True,
            summary_title="기본 루틴",
            rationale_summary=["선택한 조건에 맞는 운동이 없습니다."],
            routine_blocks=[],
            warnings=["타겟 근육 또는 장비 조건을 변경해 주세요."],
        )

    params = _GOAL_PARAMS.get(goal, _GOAL_PARAMS["HYPERTROPHY"])
    doms = doms_db or {}

    sorted_candidates = sorted(
        candidates,
        key=lambda x: (_MOVEMENT_PRIORITY.get(x["movement_type"], 1), -x["efficiency_tier"])
    )

    blocks: List[RoutineBlock] = []
    remaining_sets = max_total_sets
    order = 1

    for ex in sorted_candidates:
        if remaining_sets <= 0:
            break

        doms_level = doms.get(ex["primary_muscle"], 0)
        if doms_level >= 3:
            continue

        sets = params["sets"]
        if doms_level == 2:
            sets = min(sets, 2)
        elif doms_level == 1:
            sets = max(1, sets - 1)
        sets = min(sets, remaining_sets)
        remaining_sets -= sets

        exercise_id = str(ex.get("id", ex["name_kr"].replace(" ", "_").lower()))
        blocks.append(RoutineBlock(
            order=order,
            exercise_id=exercise_id,
            exercise_name=ex["name_kr"],
            default_rest_sec=params["rest_sec"],
            prescription=_build_prescription(
                sets=sets,
                reps=params["reps"],
                weight_kg=None,
                rest_sec=params["rest_sec"],
            ),
            exercise_rationale=_FALLBACK_TIPS.get(ex["movement_type"], "정확한 자세로 수행하세요."),
            substitution_candidates=[],
        ))
        order += 1

    return RoutineDraftResponse(
        generation_status="fallback",
        status_reason_code=status_reason_code,
        is_fallback=True,
        summary_title=f"기본 {req.target_split_label} 루틴",
        rationale_summary=["AI 코치 연결이 원활하지 않아 기본 루틴으로 대체되었습니다."],
        routine_blocks=blocks,
        warnings=["중량은 본인의 컨디션에 맞게 조절하세요."],
    )


# ==========================================
# 10. AI 파이프라인
# ==========================================

def generate_smart_routine(
    req: RoutineRequest,
    db: Session,
    profile: Optional[UserProfileContext] = None,
    recent_sets: Optional[List[RecentSetRecord]] = None,
) -> RoutineDraftResponse:

    # 0. targetSplitLabel → DB 근육 Enum, DOMS 변환
    target_muscles = split_label_to_muscles(req.target_split_label)
    doms_db = map_doms_to_db(req.doms)
    goal = req.goal or (profile.goal_type if profile else "HYPERTROPHY")
    pain_areas = req.current_pain_areas

    print(f"[매핑] split_label={req.target_split_label} → {target_muscles}")
    print(f"[매핑] doms → {doms_db}")

    # 1. 후보 운동 DB 조회
    candidates = get_candidate_exercises(
        db=db,
        target_muscles=target_muscles,
        unavailable_equipment=req.unavailable_equipment,
        pain_areas=pain_areas,
    )
    candidate_str = format_candidates_for_prompt(candidates)
    print(f"[후보 운동] {len(candidates)}개 조회됨")

    # 2. 최대 세트 수 계산
    rest_sec = _GOAL_PARAMS.get(goal, _GOAL_PARAMS["HYPERTROPHY"])["rest_sec"]
    safe_time = max(req.time_available_min, 15)
    max_total_sets = int(safe_time / ((60 + rest_sec) / 60))

    # 3. DOMS 프롬프트 문자열 생성
    if doms_db:
        label_map = {1: "약간 뻐근함 (볼륨 20% 감소)", 2: "매우 뻐근함 (가벼운 자극 1~2세트만)", 3: "통증/부상 우려 (해당 부위 운동 금지)"}
        doms_instructions = "\n".join(
            f"- {part}: {label_map.get(level, '')}" for part, level in doms_db.items()
        )
    else:
        doms_instructions = "현재 근육통 없음. 정상 볼륨으로 진행."

    # 4. LLM 호출
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    structured_llm = llm.with_structured_output(LLMRoutineOutput)

    system_prompt = _build_system_prompt(profile, recent_sets)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "유저 코멘트: {user_note}")
    ])

    invoke_kwargs = {
        "goal": goal,
        "max_sets": max_total_sets,
        "doms_instructions": doms_instructions,
        "candidate_exercises": candidate_str,
        "user_note": req.user_note or "없음",
    }

    reason: StatusReasonCode = "networkError"
    try:
        chain = prompt | structured_llm
        response: LLMRoutineOutput = chain.invoke(invoke_kwargs)
        return _build_routine_draft(response, "success", "none", False)

    except (asyncio.TimeoutError, httpx.TimeoutException) as e:
        reason = "llmTimeout"
        print(f"[AI 실패 - TIMEOUT] {e} → fallback 루틴으로 전환")

    except (OutputParserException, ValidationError) as e:
        # ── normalize adapter v1 ──────────────────────────────────────
        # structured_output 파싱 실패 시 바로 fallback 하지 않고,
        # raw 텍스트를 추출하거나 LLM을 비구조화로 재호출해 복구를 시도한다.
        print(f"[AI - SCHEMA 오류] 정제 어댑터로 복구 시도... ({type(e).__name__})")
        try:
            # OutputParserException은 llm_output 속성에 raw 텍스트를 담고 있을 수 있다.
            raw_text: str | None = getattr(e, "llm_output", None)

            if raw_text is None:
                # raw 텍스트를 얻지 못한 경우 비구조화 LLM 재호출로 획득
                print("[정제 어댑터] raw 텍스트 없음 → LLM 비구조화 재호출")
                raw_resp = (prompt | llm).invoke(invoke_kwargs)
                raw_text = raw_resp.content if hasattr(raw_resp, "content") else str(raw_resp)

            normalized = normalize_llm_response(raw_text, llm=llm)
            print("[정제 어댑터] 복구 성공!")
            return _build_routine_draft(normalized, "success", "none", False)

        except Exception as norm_e:
            reason = "schemaError"
            print(f"[정제 어댑터] 복구 실패: {norm_e} → fallback 루틴으로 전환")
        # ─────────────────────────────────────────────────────────────

    except Exception as e:
        reason = "networkError"
        print(f"[AI 실패 - NETWORK] {e} → fallback 루틴으로 전환")

    return generate_fallback_routine(
        req, candidates, max_total_sets, doms_db,
        goal=goal, status_reason_code=reason,
    )


# ==========================================
# 테스트 실행
# ==========================================
if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from database import SessionLocal

    TEST_USER_ID = "test-user-001"

    mock_payload = RoutineRequest(
        user_id=TEST_USER_ID,
        target_split_label="push",
        readiness_level="normal",
        time_available_min=70,
        current_pain_areas=[],
        doms=[DomEntry(body_part="chest", level="mild")],
        unavailable_equipment=["smith_machine"],
    )

    db = SessionLocal()
    try:
        print("[테스트] 프로필 및 최근 기록 조회 중...\n")
        profile = get_user_profile_context(db, TEST_USER_ID)
        recent_sets = get_recent_sets(db, TEST_USER_ID)
        print(f"[프로필] {profile}")
        print(f"[최근 세트] {len(recent_sets)}개\n")

        print("[테스트] 루틴 생성 중...\n")
        result = generate_smart_routine(mock_payload, db, profile=profile, recent_sets=recent_sets)
        print(result.model_dump_json(by_alias=True, indent=2))
    finally:
        db.close()

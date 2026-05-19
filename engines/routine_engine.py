import re
import json
import math
import asyncio
import httpx
from dataclasses import dataclass
from typing import Any, List, Optional, Dict, Literal, Tuple
from collections import defaultdict
from uuid import uuid4
from pydantic import BaseModel, Field, ConfigDict, ValidationError, AliasChoices
from pydantic.alias_generators import to_camel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from sqlalchemy.orm import Session
from sqlalchemy import text
from dotenv import load_dotenv

from engines.llm_router import get_llm, map_llm_error, StatusReasonCode

load_dotenv()

# ==========================================
# 1. 공통 타입 정의
# ==========================================

GenerationStatus = Literal["success", "fallback", "failed"]
# StatusReasonCode는 engines.llm_router에서 단일 정의 후 re-export됨

DOMS_LEVEL_MAP: Dict[str, int] = {
    "mild": 1,
    "moderate": 2,
    "severe": 3,
}

# scripts/exercise_tier.xlsx primary_muscle 기준 slug SSOT
MUSCLE_SLUGS = {
    "abductors",
    "abs",
    "adductor",
    "back-deltoids",
    "biceps",
    "calves",
    "chest",
    "forearm",
    "front-deltoids",
    "gluteal",
    "hamstring",
    "lower-back",
    "neck",
    "obliques",
    "quadriceps",
    "trapezius",
    "triceps",
    "upper-back",
}

LARGE_MUSCLE_SLUGS = {
    "chest",
    "upper-back",
    "lower-back",
    "quadriceps",
    "hamstring",
    "gluteal",
}

ACCESSORY_MUSCLE_SLUGS = {
    "abductors",
    "abs",
    "adductor",
    "back-deltoids",
    "biceps",
    "calves",
    "forearm",
    "front-deltoids",
    "neck",
    "obliques",
    "trapezius",
    "triceps",
}

ACCESSORY_PRIMARY_SPLITS = {"arm", "core", "shoulder", "neck"}

# ==========================================
# 2. Request 모델
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


# ==========================================
# 3. DB 컨텍스트 모델 (내부)
# ==========================================

class RecentSetRecord(BaseModel):
    exercise_id: Optional[str] = None
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
    pain_areas: List[PainAreaEntry] = Field(default_factory=list)


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
    total_estimated_time: int = 0
    summary_title: str
    rationale_summary: List[str]
    routine_blocks: List[RoutineBlock]
    warnings: List[str] = Field(default_factory=list)


# ==========================================
# 6. Split Label → spreadsheet slug 매핑
# ==========================================

SPLIT_LABEL_TO_MUSCLES: Dict[str, List[str]] = {
    "push":      ["chest", "front-deltoids", "triceps"],
    "pull":      ["upper-back", "trapezius", "biceps", "forearm", "back-deltoids"],
    "legs":      ["quadriceps", "hamstring", "gluteal", "calves", "adductor", "abductors"],
    "upper":     ["chest", "upper-back", "trapezius", "front-deltoids", "back-deltoids", "biceps", "triceps"],
    "lower":     ["quadriceps", "hamstring", "gluteal", "calves", "adductor", "abductors"],
    "chest":     ["chest"],
    "back":      ["upper-back", "trapezius", "lower-back"],
    "shoulder":  ["front-deltoids", "back-deltoids", "trapezius"],
    "arm":       ["biceps", "triceps", "forearm"],
    "core":      ["abs", "lower-back", "obliques"],
    "neck":      ["neck"],
    "full_body": ["chest", "upper-back", "front-deltoids", "quadriceps", "hamstring", "gluteal", "abs"],
}

@dataclass
class MuscleMapping:
    ai_targets: List[str]
    db_enums: List[str]

# react-body-highlighter 키와 spreadsheet slug 기준 단일 진실 공급원 (SSOT)
MUSCLE_REGISTRY: Dict[str, MuscleMapping] = {
    "chest":          MuscleMapping(["chest"],                              ["chest"]),
    "upper-back":     MuscleMapping(["upperBack"],                          ["upper-back"]),
    "trapezius":      MuscleMapping(["upperBack"],                          ["trapezius"]),
    "lats":           MuscleMapping(["lats"],                               ["upper-back"]),
    "lower-back":     MuscleMapping(["lowerBack"],                          ["lower-back"]),
    "front-deltoids": MuscleMapping(["frontDelts"],                         ["front-deltoids"]),
    "back-deltoids":  MuscleMapping(["rearDelts"],                          ["back-deltoids"]),
    "deltoids":       MuscleMapping(["frontDelts", "rearDelts"],            ["front-deltoids", "back-deltoids"]),
    "rotator-cuff":   MuscleMapping(["rotatorCuff"],                        ["trapezius"]),
    "biceps":         MuscleMapping(["biceps"],                             ["biceps"]),
    "triceps":        MuscleMapping(["triceps"],                            ["triceps"]),
    "forearm":        MuscleMapping(["forearm"],                            ["forearm"]),
    "abs":            MuscleMapping(["abs"],                                ["abs"]),
    "obliques":       MuscleMapping(["obliques"],                           ["obliques"]),
    "glutes":         MuscleMapping(["glutes"],                             ["gluteal"]),
    "gluteal":        MuscleMapping(["glutes"],                             ["gluteal"]),
    "hamstring":      MuscleMapping(["hamstrings"],                         ["hamstring"]),
    "quadriceps":     MuscleMapping(["quads"],                              ["quadriceps"]),
    "calves":         MuscleMapping(["calves"],                             ["calves"]),
    "adductors":      MuscleMapping(["adductors"],                          ["adductor"]),
    "adductor":       MuscleMapping(["adductors"],                          ["adductor"]),
    "abductors":      MuscleMapping(["abductors"],                          ["abductors"]),
    "knees":          MuscleMapping(["quads", "hamstrings"],                ["quadriceps", "hamstring"]),
    "neck":           MuscleMapping(["neck"],                               ["neck"]),
    "head":           MuscleMapping([],                                     []),
}


def split_label_to_muscles(label: str) -> List[str]:
    """targetSplitLabel("push" 등)을 spreadsheet slug 리스트로 변환한다."""
    return SPLIT_LABEL_TO_MUSCLES.get(label.lower(), [])


def get_mapped_targets(raw_muscles: List[str]) -> Tuple[List[str], List[str]]:
    """
    프론트엔드 target_muscles 리스트를 MUSCLE_REGISTRY 기반으로 변환한다.
    반환: (ai_targets, db_enums) — 각각 중복 제거된 리스트. db_enums는 legacy 이름이며 spreadsheet slug를 담는다.
    """
    ai_seen: set = set()
    db_seen: set = set()
    for raw in raw_muscles:
        mapping = MUSCLE_REGISTRY.get(raw.lower())
        if mapping:
            ai_seen.update(mapping.ai_targets)
            db_seen.update(mapping.db_enums)
        else:
            normalized = raw.strip()
            if normalized in MUSCLE_SLUGS:
                db_seen.add(normalized)
            else:
                ai_seen.add(raw)
    return list(ai_seen), list(db_seen)


def map_doms_to_db(doms: List[DomEntry]) -> Dict[str, int]:
    """DomEntry 배열을 {spreadsheet_slug: level_int} 딕셔너리로 변환한다."""
    result: Dict[str, int] = {}
    for entry in doms:
        level_int = DOMS_LEVEL_MAP.get(entry.level.lower(), 1)
        mapping = MUSCLE_REGISTRY.get(entry.body_part.lower())
        normalized = entry.body_part.strip()
        targets = mapping.db_enums if mapping is not None else [
            normalized if normalized in MUSCLE_SLUGS else entry.body_part
        ]
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

    def _parse_strength_baseline(val):
        parsed = _parse(val)
        if isinstance(parsed, dict):
            return parsed
        if not isinstance(parsed, list):
            return {}

        baseline: Dict[str, Dict[str, Any]] = {}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            exercise_id = item.get("exerciseId") or item.get("exercise_id")
            exercise_name = item.get("exerciseNameSnapshot") or item.get("exercise_name_snapshot")
            weight = item.get("workingWeightKg") or item.get("working_weight_kg") or item.get("weight_kg")
            reps = item.get("reps")
            if not weight or not reps:
                continue
            value = {
                "exercise_id": str(exercise_id) if exercise_id else None,
                "exercise_name_snapshot": exercise_name,
                "weight_kg": float(weight),
                "reps": int(reps),
            }
            if exercise_id:
                baseline[str(exercise_id)] = value
            if exercise_name:
                baseline[str(exercise_name)] = value
        return baseline

    return UserProfileContext(
        goal_type=r["goal_type"],
        split_type=r["split_type"],
        split_label=r.get("split_label"),
        experience_level=r.get("experience_level"),
        strength_baseline=_parse_strength_baseline(r.get("strength_baseline")),
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
            SELECT exercise_id, exercise_name_snapshot, weight_kg, reps
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
            exercise_id=str(r[0]) if r[0] is not None else None,
            exercise_name=r[1],
            weight_kg=float(r[2]) if r[2] is not None else None,
            reps=int(r[3]),
        )
        for r in rows
    ]


def get_candidate_exercises(
    db: Session,
    target_muscles: List[str],
    unavailable_equipment: List[str],
    pain_areas: List[PainAreaEntry],
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

    pain_body_parts = [
        p.body_part for p in pain_areas
        if isinstance(p, PainAreaEntry) and p.body_part
    ]
    if pain_body_parts:
        not_like_parts = " AND ".join(
            [f"pain_triggers NOT LIKE :p{i}" for i in range(len(pain_body_parts))]
        )
        pain_filter = f"AND (pain_triggers IS NULL OR ({not_like_parts}))"
        for i, part in enumerate(pain_body_parts):
            params[f"p{i}"] = f"%{part}%"
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
        return "?? ??? ?? ??? ????."
    lines = []
    for ex in candidates:
        secondary = str(ex.get("secondary_muscle") or "none").strip()
        pain_triggers = str(ex.get("pain_triggers") or "none").strip()
        equipment = str(ex.get("equipment_req") or "unknown").strip()
        reasons = "; ".join(ex.get("score_reasons", [])) or "server ranked"
        lines.append(
            f"- {ex['name_kr']} (ID: {ex['id']}, score={ex.get('score', 0)}, "
            f"movement_type={ex.get('movement_type') or 'UNKNOWN'}, "
            f"primary={ex.get('primary_muscle') or 'unknown'}, secondary={secondary}, "
            f"equipment={equipment}, pain_triggers={pain_triggers}, reason={reasons})"
        )
    return "\n".join(lines)


def score_candidate_exercises(
    candidates: List[dict],
    target_muscles: List[str],
    doms_db: Optional[Dict[str, int]] = None,
    blocked_equipment: Optional[List[str]] = None,
    pain_areas: Optional[List[PainAreaEntry]] = None,
    recent_sets: Optional[List[RecentSetRecord]] = None,
    top_n: int = 12,
) -> List[dict]:
    """?? ??? deterministic?? ????? ?? N?? ????."""
    target_set = set(target_muscles)
    scored: List[dict] = []

    doms = doms_db or {}
    blocked_equipment = blocked_equipment or []
    pain_areas = pain_areas or []
    recent_names = {
        record.exercise_name.strip().lower()
        for record in (recent_sets or [])
        if record.exercise_name
    }

    for candidate in candidates:
        exclusion_reasons: List[str] = []
        if not _candidate_is_safe(candidate, blocked_equipment, pain_areas):
            candidate_equipment = _normalize_tokens(candidate.get("equipment_req")) - {"BODYWEIGHT"}
            blocked = {item.strip().upper() for item in blocked_equipment}
            if candidate_equipment & blocked:
                exclusion_reasons.append("excluded unavailable equipment")
            pain_tokens = {p.body_part.lower() for p in pain_areas if p.body_part}
            candidate_pain = str(candidate.get("pain_triggers") or "").lower()
            if any(token in candidate_pain for token in pain_tokens):
                exclusion_reasons.append("excluded pain trigger")
            continue

        score = 0
        reasons: List[str] = []

        efficiency = int(candidate.get("efficiency_tier") or 0)
        score += efficiency * 10
        reasons.append(f"efficiency {efficiency}")

        movement_type = str(candidate.get("movement_type") or "").upper()
        if movement_type == "COMPOUND":
            score += 15
            reasons.append("compound priority")
        elif movement_type == "ISOLATION":
            score += 5
            reasons.append("isolation support")

        primary = str(candidate.get("primary_muscle") or "").strip()
        secondary = {
            part.strip()
            for part in str(candidate.get("secondary_muscle") or "").split(",")
            if part.strip()
        }
        if primary and primary in target_set:
            score += 20
            reasons.append("primary target match")
        if secondary & target_set:
            score += 8
            reasons.append("secondary target match")
        if primary in LARGE_MUSCLE_SLUGS:
            score += 10
            reasons.append("large muscle priority")
        elif primary in ACCESSORY_MUSCLE_SLUGS:
            score -= 4
            reasons.append("accessory volume guard")

        doms_level = doms.get(primary, 0)
        if doms_level >= 3:
            continue
        if doms_level == 2:
            score -= 25
            reasons.append("doms moderate penalty")
        elif doms_level == 1:
            score -= 10
            reasons.append("doms mild penalty")

        recent_key_candidates = {
            str(candidate.get("name_kr") or "").strip().lower(),
            str(candidate.get("name_en") or "").strip().lower(),
            str(candidate.get("id") or "").strip().lower(),
        }
        if recent_key_candidates & recent_names:
            score -= 6
            reasons.append("recent repetition penalty")

        scored.append({
            **candidate,
            "score": score,
            "score_reasons": reasons,
        })

    return sorted(
        scored,
        key=lambda ex: (
            -ex["score"],
            -int(ex.get("efficiency_tier") or 0),
            str(ex.get("id") or ""),
        ),
    )[:top_n]


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


def _format_request_pain_areas(pain_areas: List[PainAreaEntry]) -> str:
    if not pain_areas:
        return "none"
    values = []
    for pain in pain_areas:
        if not isinstance(pain, PainAreaEntry) or not pain.body_part:
            continue
        details = [pain.body_part]
        if pain.side:
            details.append(f"side={pain.side}")
        if pain.severity:
            details.append(f"severity={pain.severity}")
        values.append("(" + ", ".join(details) + ")")
    return ", ".join(values) if values else "none"


def _build_system_prompt(
    profile: Optional[UserProfileContext],
    recent_sets: Optional[List[RecentSetRecord]],
) -> str:
    sections = []

    sections.append(
        "[ROLE]\n"
        "You are Fit-Core's routine composer. Build a safe, time-bounded workout plan from the supplied candidates only.\n\n"
        "[REQUEST CONTEXT]\n"
        "- goal: {goal}\n"
        "- timeAvailableMin: {time_available_min}\n"
        "- readinessLevel: {readiness_level}\n"
        "- unavailable_equipment: {unavailable_equipment}\n"
        "- target_split_label: {target_split_label}\n"
        "- target_muscles: {target_muscles}\n"
        "- current_pain_areas: {current_pain_areas}\n"
        "- candidate_count: {candidate_count}\n"
        "- Use this context to write specific exercise_rationale text; server-side filters remain authoritative.\n\n"
        "[HARD CONSTRAINTS]\n"
        "- Use only exercise_id values that appear in [RANKED CANDIDATES].\n"
        "- Respect the user's goal: {goal}.\n"
        "- Keep total working sets at or below {max_sets}.\n"
        "- Treat DOMS, pain, and equipment restrictions as hard constraints; never reintroduce excluded exercises.\n"
        "- Return stable ordering and valid integer sets/reps/rest values.\n"
        "- Weight and reps will be finalized by deterministic server logic, so prefer sensible structure over speculative numbers.\n\n"
        "[READINESS POLICY]\n"
        "- If readinessLevel is low, avoid overloading the plan with heavy COMPOUND choices when reasonable alternatives exist.\n"
        "- If readinessLevel is low and a COMPOUND exercise is selected, keep sets conservative; the server will raise RIR and may reduce sets.\n"
        "- If readinessLevel is high, do not inflate volume beyond the set cap; the server may only make a small RIR adjustment.\n\n"
        "[TIME POLICY]\n"
        "- The server time model includes warmup, transition buffer, movement type, and rest time.\n"
        "- For short timeAvailableMin values, prefer fewer exercises with clear priorities instead of many low-value additions.\n"
        "- total_estimated_time is recomputed by the server, but the exercise list must still be plausible within the set cap.\n\n"
        "[SET ALLOCATION POLICY]\n"
        "- For push, pull, legs, upper, lower, and full_body splits, allocate main working sets to large-muscle targets first.\n"
        "- Use small-muscle isolation work as accessory volume after the main targets are covered.\n"
        "- For arm, core, shoulder, and neck splits, the named small muscle group may be treated as the main target.\n\n"
        "[RATIONALE POLICY]\n"
        "- The candidates are pre-ranked by the server.\n"
        "- When writing exercise_rationale, use only visible candidate fields: primary, secondary, equipment, movement_type, pain_triggers, and reason.\n"
        "- Mention readiness, time pressure, target muscles, DOMS, pain, or equipment only when those values are present in [REQUEST CONTEXT] or [DOMS].\n"
        "- Do not claim a selected exercise is pain-free; say it was selected from server-filtered candidates when pain context matters.\n"
        "- Do not invent unsupported medical claims or selection reasons.\n\n"
        "[OUTPUT SCHEMA]\n"
        "- Return one JSON object matching LLMRoutineOutput exactly.\n"
        "- Required top-level fields: total_estimated_time, summary_title, rationale_summary, warnings, exercises.\n"
        "- Each exercise must include: exercise_id, exercise_name, primary_muscles, target_reps, sets, rest_time_sec, exercise_rationale.\n"
        "- Keep exercises as an ordered list; do not wrap the response in markdown or prose.\n\n"
        "[PROHIBITED BEHAVIOR]\n"
        "- Never choose an exercise outside [RANKED CANDIDATES].\n"
        "- Never reintroduce excluded equipment or pain-triggering movements.\n"
        "- Never ignore DOMS instructions or exceed the working-set cap.\n"
        "- Never fabricate medical advice, user history, or unavailable rationale."
    )

    if profile:
        lines = ["[USER PROFILE]"]
        split_str = profile.split_type + (f" ({profile.split_label})" if profile.split_label else "")
        lines.append(f"- goal: {profile.goal_type}")
        lines.append(f"- split: {split_str}")
        if profile.experience_level:
            lines.append(f"- experience: {profile.experience_level}")
        if profile.pain_areas:
            pain_strs = [
                f"{p.body_part}({p.side or ''}, {p.severity or ''})"
                for p in profile.pain_areas
                if isinstance(p, PainAreaEntry) and p.body_part
            ]
            if pain_strs:
                lines.append(f"- chronic pain history: {', '.join(pain_strs)}")
        if profile.strength_baseline:
            lines.append("\n[STRENGTH BASELINE - reference only]")
            lines.append(_format_strength_baseline(profile.strength_baseline))
        sections.append("\n".join(lines))

    sections.append("[DOMS]\n{doms_instructions}")

    if recent_sets:
        history_str = _format_recent_sets(recent_sets)
        if history_str:
            sections.append("[RECENT SETS - reference only]\n" + history_str)

    sections.append(
        "[RANKED CANDIDATES]\n"
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


def _normalize_tokens(raw: Optional[str]) -> set[str]:
    if not raw:
        return set()
    return {token.strip().upper() for token in str(raw).split(",") if token.strip()}


def _candidate_is_safe(
    candidate: dict,
    blocked_equipment: List[str],
    pain_areas: List[PainAreaEntry],
) -> bool:
    blocked = {item.strip().upper() for item in blocked_equipment}
    candidate_equipment = _normalize_tokens(candidate.get("equipment_req")) - {"BODYWEIGHT"}
    if candidate_equipment & blocked:
        return False

    pain_tokens = {p.body_part.lower() for p in pain_areas if p.body_part}
    candidate_pain = str(candidate.get("pain_triggers") or "").lower()
    return not any(token in candidate_pain for token in pain_tokens)


def _candidate_to_plan(candidate: dict, template: LLMExercisePlan) -> LLMExercisePlan:
    return template.model_copy(update={
        "exercise_id": str(candidate["id"]),
        "exercise_name": candidate.get("name_kr") or candidate.get("name_en") or template.exercise_name,
        "movement_pattern": candidate.get("movement_pattern") or template.movement_pattern,
        "movement_type": candidate.get("movement_type") or template.movement_type,
        "primary_muscles": [candidate["primary_muscle"]] if candidate.get("primary_muscle") else [],
        "equipment_type": candidate.get("equipment_req"),
        "exercise_rationale": f"{template.exercise_rationale} (후검증 repair: 안전 후보로 교체)",
    })


def validate_and_repair_routine_output(
    llm_output: LLMRoutineOutput,
    candidates: List[dict],
    blocked_equipment: List[str],
    pain_areas: List[PainAreaEntry],
    doms_db: Optional[Dict[str, int]] = None,
    max_total_sets: Optional[int] = None,
    time_available_min: Optional[int] = None,
    max_repairs: int = 2,
) -> Optional[LLMRoutineOutput]:
    """
    LLM success output을 후보군 기준으로 재검증한다.
    - 후보 밖 exercise_id 또는 현재 제약에 안전하지 않은 후보는 repair 시도
    - 위반이 많거나 대체 후보가 없으면 None을 반환해 fallback으로 넘긴다
    """
    safe_candidates = [
        c for c in candidates if _candidate_is_safe(c, blocked_equipment, pain_areas)
    ]
    safe_by_id = {str(c["id"]): c for c in safe_candidates}
    if not safe_candidates:
        return None

    repaired = llm_output.model_copy(deep=True)
    used_ids: set[str] = set()
    violations = 0
    doms = doms_db or {}

    for index, exercise in enumerate(repaired.exercises):
        candidate = safe_by_id.get(exercise.exercise_id)
        if candidate is not None:
            used_ids.add(exercise.exercise_id)
            if candidate.get("movement_type"):
                exercise.movement_type = candidate["movement_type"]
            if candidate.get("primary_muscle"):
                exercise.primary_muscles = [candidate["primary_muscle"]]
            if candidate.get("equipment_req"):
                exercise.equipment_type = candidate["equipment_req"]
        else:
            violations += 1
            if violations > max_repairs:
                return None

            replacement = next(
                (c for c in safe_candidates if str(c["id"]) not in used_ids),
                None,
            )
            if replacement is None:
                return None

            repaired.exercises[index] = _candidate_to_plan(replacement, exercise)
            used_ids.add(str(replacement["id"]))
            exercise = repaired.exercises[index]
            candidate = replacement

        primary = str(candidate.get("primary_muscle") or "").strip()
        doms_level = doms.get(primary, 0)
        if doms_level >= 3:
            return None
        if exercise.sets < 1 or exercise.target_reps < 1 or exercise.rest_time_sec < 0:
            return None
        if doms_level == 2:
            exercise.sets = min(exercise.sets, 2)
        elif doms_level == 1:
            exercise.sets = max(1, exercise.sets - 1)

    if max_total_sets is not None:
        total_sets = sum(exercise.sets for exercise in repaired.exercises)
        overflow = total_sets - max_total_sets
        if overflow > 0:
            for exercise in reversed(repaired.exercises):
                reducible = max(0, exercise.sets - 1)
                reduction = min(reducible, overflow)
                exercise.sets -= reduction
                overflow -= reduction
                if overflow == 0:
                    break
            if overflow > 0:
                return None

    if time_available_min is not None:
        if estimate_routine_time_min(repaired.exercises) > time_available_min:
            return None

    if violations:
        repaired.warnings = [
            *repaired.warnings,
            f"LLM 결과 중 {violations}개 운동을 안전 후보로 교체했습니다.",
        ]
    return repaired


_GOAL_INTENSITY = {
    "strength": 0.85,
    "hypertrophy": 0.72,
    "endurance": 0.60,
    "fatloss": 0.60,
    "recomposition": 0.72,
    "generalfitness": 0.65,
}


def _movement_type_key(value: Optional[str]) -> str:
    normalized = (value or "").strip().upper()
    return normalized if normalized in {"COMPOUND", "ISOLATION", "STATIC"} else "UNKNOWN"


def _work_seconds_for_movement(movement_type: Optional[str]) -> int:
    return {
        "COMPOUND": 60,
        "ISOLATION": 40,
        "STATIC": 45,
    }.get(_movement_type_key(movement_type), 45)


def estimate_routine_time_min(exercises: List[LLMExercisePlan]) -> int:
    """
    서버 기준 예상 시간. LLM 산수는 신뢰하지 않고 최종 루틴 블록에서 재계산한다.
    - warmup: 5분
    - 운동 간 전환: 각 2분
    - set 수행 시간: compound 60초, isolation 40초, static/unknown 45초
    - 휴식은 같은 운동의 세트 사이에만 계산한다.
    """
    active = [exercise for exercise in exercises if exercise.sets > 0]
    if not active:
        return 0

    total_sec = 5 * 60
    total_sec += max(0, len(active) - 1) * 2 * 60
    for exercise in active:
        sets = max(0, exercise.sets)
        work_sec = _work_seconds_for_movement(exercise.movement_type)
        rest_sec = max(0, exercise.rest_time_sec)
        total_sec += sets * work_sec
        total_sec += max(0, sets - 1) * rest_sec
    return max(1, math.ceil(total_sec / 60))


def _calculate_max_total_sets(time_available_min: int, goal: str) -> int:
    rest_sec = _GOAL_PARAMS.get(goal.lower(), _GOAL_PARAMS["hypertrophy"])["rest_sec"]
    available_sec = max(0, time_available_min - 5) * 60
    per_set_sec = _work_seconds_for_movement("COMPOUND") + rest_sec
    return max(1, int(available_sec / per_set_sec))


def _apply_readiness_to_exercise(exercise: LLMExercisePlan, readiness_level: Optional[str]) -> None:
    readiness = (readiness_level or "normal").strip().lower()
    movement_type = _movement_type_key(exercise.movement_type)
    base_rir = exercise.target_rir if exercise.target_rir is not None else 2

    if readiness == "low" and movement_type == "COMPOUND":
        if exercise.sets > 2:
            exercise.sets = max(2, exercise.sets - 1)
        exercise.target_rir = min(5, base_rir + 2)
    elif readiness == "high":
        exercise.target_rir = max(0, base_rir - 1)


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
) -> None:
    if not _uses_large_muscle_guard(target_split_label, target_muscles):
        return

    primary = exercise.primary_muscles[0] if exercise.primary_muscles else ""
    movement_type = _movement_type_key(exercise.movement_type)
    if primary in ACCESSORY_MUSCLE_SLUGS and movement_type != "COMPOUND":
        exercise.sets = min(exercise.sets, 2)


def _round_to_nearest_2_5(value: float) -> float:
    return round(value / 2.5) * 2.5


def _find_recent_reference(
    exercise: LLMExercisePlan,
    recent_sets: Optional[List[RecentSetRecord]],
) -> Optional[RecentSetRecord]:
    if not recent_sets:
        return None
    exercise_id = exercise.exercise_id.strip().lower()
    for record in recent_sets:
        if record.exercise_id and record.exercise_id.strip().lower() == exercise_id and record.weight_kg:
            return record

    normalized_names = {exercise.exercise_name.strip().lower(), exercise_id}
    matching = [
        record for record in recent_sets
        if record.exercise_name.strip().lower() in normalized_names and record.weight_kg
    ]
    return matching[0] if matching else None


def _find_baseline_reference(
    exercise: LLMExercisePlan,
    profile: Optional[UserProfileContext],
) -> Optional[Tuple[float, int]]:
    if not profile or not profile.strength_baseline:
        return None
    normalized_keys = {exercise.exercise_name.strip().lower(), exercise.exercise_id.strip().lower()}
    for key, value in profile.strength_baseline.items():
        if not isinstance(value, dict):
            continue
        value_keys = {
            key.strip().lower(),
            str(value.get("exercise_id") or "").strip().lower(),
            str(value.get("exerciseId") or "").strip().lower(),
            str(value.get("exercise_name_snapshot") or "").strip().lower(),
            str(value.get("exerciseNameSnapshot") or "").strip().lower(),
        }
        if not (value_keys & normalized_keys):
            continue
        weight = value.get("weight_kg")
        if weight is None:
            weight = value.get("workingWeightKg") or value.get("working_weight_kg")
        reps = value.get("reps")
        if weight and reps:
            return float(weight), int(reps)
    return None


def apply_deterministic_targets(
    llm_output: LLMRoutineOutput,
    goal: str,
    recent_sets: Optional[List[RecentSetRecord]],
    profile: Optional[UserProfileContext],
    readiness_level: Optional[str] = "normal",
    target_split_label: Optional[str] = None,
    target_muscles: Optional[List[str]] = None,
) -> LLMRoutineOutput:
    """
    LLM의 kg/reps는 hint로만 두고 최종 값은 서버가 확정한다.
    최근 세트 > strength_baseline > LLM hint 순으로 참조한다.
    """
    adjusted = llm_output.model_copy(deep=True)
    goal_key = goal.lower()
    deterministic_reps = _GOAL_PARAMS.get(goal_key, _GOAL_PARAMS["hypertrophy"])["reps"]
    intensity = _GOAL_INTENSITY.get(goal_key, _GOAL_INTENSITY["hypertrophy"])

    for exercise in adjusted.exercises:
        exercise.target_reps = deterministic_reps
        exercise.rest_time_sec = _GOAL_PARAMS.get(goal_key, _GOAL_PARAMS["hypertrophy"])["rest_sec"]
        _apply_readiness_to_exercise(exercise, readiness_level)
        _apply_large_muscle_volume_guard(exercise, target_split_label, target_muscles)

        equipment = (exercise.equipment_type or "").upper()
        if "BODYWEIGHT" in equipment:
            exercise.target_weight_kg = None
            continue

        recent = _find_recent_reference(exercise, recent_sets)
        if recent and recent.weight_kg:
            one_rm = calculate_1rm(recent.weight_kg, recent.reps)
            exercise.target_weight_kg = _round_to_nearest_2_5(one_rm * intensity)
            continue

        baseline = _find_baseline_reference(exercise, profile)
        if baseline:
            one_rm = calculate_1rm(baseline[0], baseline[1])
            exercise.target_weight_kg = _round_to_nearest_2_5(one_rm * intensity)
            continue

        if exercise.target_weight_kg is not None:
            exercise.target_weight_kg = max(0, _round_to_nearest_2_5(exercise.target_weight_kg))

    return adjusted


# ==========================================
# 9. Fallback 루틴 (규칙 기반)
# ==========================================

_GOAL_PARAMS = {
    "strength":    {"sets": 5, "reps": 5,  "rest_sec": 180},
    "hypertrophy": {"sets": 3, "reps": 10, "rest_sec": 90},
    "endurance":   {"sets": 3, "reps": 15, "rest_sec": 60},
    "fatloss":     {"sets": 3, "reps": 15, "rest_sec": 60},
    "recomposition": {"sets": 3, "reps": 10, "rest_sec": 90},
    "generalfitness": {"sets": 3, "reps": 12, "rest_sec": 75},
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
                target_rir=ex.target_rir if ex.target_rir is not None else 2,
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
        total_estimated_time=estimate_routine_time_min(llm_output.exercises),
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
    goal: str = "hypertrophy",
    status_reason_code: StatusReasonCode = "networkError",
) -> RoutineDraftResponse:
    print("[Fallback] 규칙 기반 루틴 생성 시작")

    if not candidates:
        return RoutineDraftResponse(
            generation_status="failed",
            status_reason_code="emptyCandidate",
            is_fallback=False,
            total_estimated_time=0,
            summary_title="기본 루틴",
            rationale_summary=["선택한 조건에 맞는 운동이 없습니다."],
            routine_blocks=[],
            warnings=["타겟 근육 또는 장비 조건을 변경해 주세요."],
        )

    params = _GOAL_PARAMS.get(goal.lower(), _GOAL_PARAMS["hypertrophy"])
    doms = doms_db or {}
    if req.target_split_label:
        fallback_target_muscles = split_label_to_muscles(req.target_split_label)
    elif req.target_muscles:
        _, fallback_target_muscles = get_mapped_targets(req.target_muscles)
    else:
        fallback_target_muscles = []

    sorted_candidates = sorted(
        candidates,
        key=lambda x: (_MOVEMENT_PRIORITY.get(x["movement_type"], 1), -x["efficiency_tier"])
    )

    blocks: List[RoutineBlock] = []
    fallback_plans: List[LLMExercisePlan] = []
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
            sets = max(1, sets // 2)   # 절반 이하로 제한 (level 1보다 항상 낮음)
        elif doms_level == 1:
            sets = max(1, sets - 1)
        sets = min(sets, remaining_sets)

        plan = LLMExercisePlan(
            exercise_id=str(ex.get("id", ex["name_kr"].replace(" ", "_").lower())),
            exercise_name=ex["name_kr"],
            movement_pattern=ex.get("movement_pattern"),
            movement_type=ex.get("movement_type"),
            primary_muscles=[ex["primary_muscle"]] if ex.get("primary_muscle") else [],
            equipment_type=ex.get("equipment_req"),
            target_weight_kg=None,
            target_reps=params["reps"],
            sets=sets,
            rest_time_sec=params["rest_sec"],
            target_rir=2,
            exercise_rationale=_FALLBACK_TIPS.get(ex["movement_type"], "정확한 자세로 수행하세요."),
        )
        _apply_readiness_to_exercise(plan, req.readiness_level)
        _apply_large_muscle_volume_guard(plan, req.target_split_label, fallback_target_muscles)
        sets = plan.sets
        remaining_sets -= sets

        blocks.append(RoutineBlock(
            order=order,
            exercise_id=plan.exercise_id,
            exercise_name=plan.exercise_name,
            movement_pattern=plan.movement_pattern,
            primary_muscles=plan.primary_muscles,
            equipment_type=plan.equipment_type,
            default_rest_sec=params["rest_sec"],
            prescription=_build_prescription(
                sets=plan.sets,
                reps=plan.target_reps,
                weight_kg=None,
                rest_sec=plan.rest_time_sec,
                target_rir=plan.target_rir if plan.target_rir is not None else 2,
            ),
            exercise_rationale=plan.exercise_rationale,
            substitution_candidates=[],
        ))
        fallback_plans.append(plan)
        order += 1
        if remaining_sets <= 0:
            break

    if not blocks:
        return RoutineDraftResponse(
            generation_status="failed",
            status_reason_code="emptyCandidate",
            is_fallback=False,
            total_estimated_time=0,
            summary_title="기본 루틴",
            rationale_summary=["모든 후보 운동이 DOMS 제약으로 제외되었습니다."],
            routine_blocks=[],
            warnings=["컨디션이 회복된 후 다시 시도해 주세요."],
        )

    return RoutineDraftResponse(
        generation_status="fallback",
        status_reason_code=status_reason_code,
        is_fallback=True,
        total_estimated_time=estimate_routine_time_min(fallback_plans),
        summary_title=f"기본 {req.target_split_label or '맞춤형'} 루틴",
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

    # 0. 타겟 근육 결정: splitLabel 우선 → target_muscles 직접 지정 → 빈 리스트
    if req.target_split_label:
        db_target_muscles = split_label_to_muscles(req.target_split_label)
        print(f"[매핑] split_label={req.target_split_label} → {db_target_muscles}")
    elif req.target_muscles:
        _, db_target_muscles = get_mapped_targets(req.target_muscles)
        print(f"[매핑] DB 타겟 → {db_target_muscles}")
    else:
        db_target_muscles = []
        print("[매핑] 타겟 근육 없음 — 빈 후보 리스트로 진행")

    doms_db = req.doms_data
    goal = req.goal or (profile.goal_type if profile else "hypertrophy")
    pain_areas = req.pain_areas
    print(f"[매핑] doms → {doms_db}")

    # 1. 후보 운동 DB 조회
    candidates = get_candidate_exercises(
        db=db,
        target_muscles=db_target_muscles,   # spreadsheet slug 형식 (예: chest, upper-back)
        unavailable_equipment=req.equipment,
        pain_areas=pain_areas,
    )
    ranked_candidates = score_candidate_exercises(
        candidates,
        db_target_muscles,
        doms_db=doms_db,
        blocked_equipment=req.equipment,
        pain_areas=pain_areas,
        recent_sets=recent_sets,
    )
    candidate_str = format_candidates_for_prompt(ranked_candidates)
    current_pain_areas = _format_request_pain_areas(pain_areas)
    print(f"[후보 운동] {len(candidates)}개 조회됨")

    # 2. 최대 세트 수 계산
    max_total_sets = _calculate_max_total_sets(req.time_available_min, goal)

    # 3. DOMS 프롬프트 문자열 생성
    if doms_db:
        label_map = {1: "약간 뻐근함 (볼륨 20% 감소)", 2: "매우 뻐근함 (가벼운 자극 1~2세트만)", 3: "통증/부상 우려 (해당 부위 운동 금지)"}
        doms_instructions = "\n".join(
            f"- {part}: {label_map.get(level, '')}" for part, level in doms_db.items()
        )
    else:
        doms_instructions = "현재 근육통 없음. 정상 볼륨으로 진행."

    # 4. LLM 호출 (provider는 LLM_PROVIDER 환경 변수로 결정)
    system_prompt = _build_system_prompt(profile, recent_sets)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "User note: {user_note}\nUse only the request context and ranked candidates above.")
    ])

    invoke_kwargs = {
        "goal": goal,
        "max_sets": max_total_sets,
        "doms_instructions": doms_instructions,
        "candidate_exercises": candidate_str,
        "user_note": req.user_note or "없음",
        "time_available_min": req.time_available_min,
        "readiness_level": req.readiness_level or "normal",
        "unavailable_equipment": ", ".join(req.equipment) if req.equipment else "none",
        "target_split_label": req.target_split_label or "none",
        "target_muscles": ", ".join(db_target_muscles) if db_target_muscles else "none",
        "current_pain_areas": current_pain_areas,
        "candidate_count": len(ranked_candidates),
    }

    reason: StatusReasonCode = "networkError"
    try:
        llm = get_llm("routine")
        structured_llm = llm.with_structured_output(LLMRoutineOutput)
        chain = prompt | structured_llm
        response: LLMRoutineOutput = chain.invoke(invoke_kwargs)
        validated = validate_and_repair_routine_output(
            response,
            ranked_candidates,
            req.equipment,
            pain_areas,
            doms_db=doms_db,
            max_total_sets=max_total_sets,
            time_available_min=req.time_available_min,
        )
        if validated is None:
            print("[AI post-validation] unrecoverable violation -> fallback")
            return generate_fallback_routine(
                req, ranked_candidates, max_total_sets, doms_db,
                goal=goal, status_reason_code="schemaError",
            )
        deterministic = apply_deterministic_targets(
            validated,
            goal,
            recent_sets,
            profile,
            req.readiness_level,
            req.target_split_label,
            db_target_muscles,
        )
        if estimate_routine_time_min(deterministic.exercises) > req.time_available_min:
            print("[AI post-deterministic] final routine exceeds time budget -> fallback")
            return generate_fallback_routine(
                req, ranked_candidates, max_total_sets, doms_db,
                goal=goal, status_reason_code="schemaError",
            )
        return _build_routine_draft(deterministic, "success", "none", False)

    except (asyncio.TimeoutError, httpx.TimeoutException) as e:
        reason = map_llm_error(e)
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
            validated = validate_and_repair_routine_output(
                normalized,
                ranked_candidates,
                req.equipment,
                pain_areas,
                doms_db=doms_db,
                max_total_sets=max_total_sets,
                time_available_min=req.time_available_min,
            )
            if validated is None:
                print("[AI post-validation] normalized output unrecoverable -> fallback")
                return generate_fallback_routine(
                    req, ranked_candidates, max_total_sets, doms_db,
                    goal=goal, status_reason_code="schemaError",
                )
            deterministic = apply_deterministic_targets(
                validated,
                goal,
                recent_sets,
                profile,
                req.readiness_level,
                req.target_split_label,
                db_target_muscles,
            )
            if estimate_routine_time_min(deterministic.exercises) > req.time_available_min:
                print("[AI post-deterministic] normalized routine exceeds time budget -> fallback")
                return generate_fallback_routine(
                    req, ranked_candidates, max_total_sets, doms_db,
                    goal=goal, status_reason_code="schemaError",
                )
            print("[정제 어댑터] 복구 성공!")
            return _build_routine_draft(deterministic, "success", "none", False)

        except Exception as norm_e:
            reason = map_llm_error(e)  # 원본 schema 에러 기준으로 분류
            print(f"[정제 어댑터] 복구 실패: {norm_e} → fallback 루틴으로 전환")
        # ─────────────────────────────────────────────────────────────

    except Exception as e:
        reason = map_llm_error(e)
        print(f"[AI 실패 - {reason.upper()}] {e} → fallback 루틴으로 전환")

    return generate_fallback_routine(
        req, ranked_candidates, max_total_sets, doms_db,
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
        pain_areas=[],
        doms_data={"chest": 1},   # mild=1
        equipment=["smith_machine"],
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

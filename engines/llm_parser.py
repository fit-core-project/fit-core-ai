"""LLM raw output → LLMRoutineOutput 정제 어댑터."""
import re
import json
from typing import Dict, List, Optional

from .candidate_ranker import _candidate_is_safe
from .prescription.estimator import estimate_routine_time_min
from .schemas import LLMExercisePlan, LLMRoutineOutput, PainAreaEntry

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
    if isinstance(data.get("rationale_summary"), str):
        data["rationale_summary"] = [data["rationale_summary"]]
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
        ex["exercise_id"] = str(ex["exercise_id"])
        ex.setdefault("primary_muscles", [])
        if isinstance(ex["primary_muscles"], str):
            ex["primary_muscles"] = [ex["primary_muscles"]]
        ex.setdefault("target_rir", 2)
        ex.setdefault("substitution_candidates", [])
        ex.setdefault("exercise_rationale", "AI 추천 운동")
        ex.setdefault("rest_time_sec", 90)
        ex.setdefault("sets", 3)
        ex.setdefault("target_reps", 10)
        for numeric_key in ("target_reps", "sets", "rest_time_sec", "target_rir"):
            value = ex.get(numeric_key)
            if isinstance(value, str):
                match = re.search(r"\d+", value)
                if match:
                    ex[numeric_key] = int(match.group())

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
    - 위반이 많거나 대체 후보가 없으면 None을 반환해 fallback으로 넘긴다.
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

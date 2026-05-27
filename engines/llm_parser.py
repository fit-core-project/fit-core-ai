"""Normalize raw LLM output into LLMRoutineOutput."""
import json
import re
from typing import Dict, List, Optional, Tuple

from .candidate_ranker import _candidate_is_safe
from .llm_router import StatusReasonCode
from .post_validation_guard import audit_routine_output, log_guard_report
from .prescription.estimator import estimate_routine_time_min
from .schemas import LLMExercisePlan, LLMRoutineOutput, PainAreaEntry

_MARKDOWN_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```")


def _try_repair_json(text: str) -> dict:
    """Repair common truncated JSON output by appending or trimming closing braces."""
    for suffix in ("}", "}]}", "}]}}", "}]}}]}"):
        try:
            return json.loads(text + suffix)
        except json.JSONDecodeError:
            pass

    # If trailing text is present, retry up to the final complete object boundary.
    last_brace = text.rfind("}")
    if last_brace != -1:
        try:
            return json.loads(text[: last_brace + 1])
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError("JSON auto repair failed", text, 0)


def _inject_defaults(data: dict) -> dict:
    """Inject required defaults before Pydantic validation."""
    data.setdefault("summary_title", "맞춤형 AI 루틴")
    if isinstance(data.get("rationale_summary"), str):
        data["rationale_summary"] = [data["rationale_summary"]]
    data.setdefault("rationale_summary", ["회원 데이터 기반으로 생성한 루틴입니다."])
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
    Normalize raw LLM output into LLMRoutineOutput.

    Steps:
    1. Remove markdown code fences.
    2. Try json.loads.
    3. Retry with local JSON repair.
    4. If provided, retry with OutputFixingParser.
    5. Inject missing defaults.
    6. Validate with Pydantic.
    """
    cleaned = _MARKDOWN_FENCE_RE.sub("", raw_text).strip()

    data: dict | None = None
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            data = _try_repair_json(cleaned)
            print("[LLM parser] JSON bracket repair succeeded")
        except json.JSONDecodeError:
            if llm is not None:
                print("[LLM parser] retrying with OutputFixingParser")
                try:
                    from langchain.output_parsers import OutputFixingParser
                    from langchain_core.output_parsers import PydanticOutputParser

                    base_parser = PydanticOutputParser(pydantic_object=LLMRoutineOutput)
                    fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)
                    return fixing_parser.parse(cleaned)
                except Exception as fix_e:
                    raise ValueError(f"OutputFixingParser failed: {fix_e}") from fix_e
            raise

    # Inject defaults before Pydantic validation so loose local-model output can be salvaged.
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
        "exercise_rationale": f"{template.exercise_rationale} (guard repair: replaced with safe candidate)",
    })


def trim_routine_to_time_budget(output: LLMRoutineOutput, time_available_min: int) -> bool:
    """Reduce sets, then trailing exercises, until the routine fits the time budget."""
    if estimate_routine_time_min(output.exercises) <= time_available_min:
        return True

    changed = True
    while changed and estimate_routine_time_min(output.exercises) > time_available_min:
        changed = False
        for exercise in reversed(output.exercises):
            if exercise.sets > 1:
                exercise.sets -= 1
                changed = True
                if estimate_routine_time_min(output.exercises) <= time_available_min:
                    output.warnings = [
                        *output.warnings,
                        "시간 예산에 맞추기 위해 일부 세트 수를 줄였습니다.",
                    ]
                    return True

    while len(output.exercises) > 1 and estimate_routine_time_min(output.exercises) > time_available_min:
        removed = output.exercises.pop()
        output.warnings = [
            *output.warnings,
            f"시간 예산에 맞추기 위해 {removed.exercise_name} 운동을 제외했습니다.",
        ]

    return bool(output.exercises) and estimate_routine_time_min(output.exercises) <= time_available_min


def _has_minimum_success_quality(output: LLMRoutineOutput) -> bool:
    if not output.exercises:
        return False
    total_sets = sum(exercise.sets for exercise in output.exercises)
    covered_muscles = {
        muscle
        for exercise in output.exercises
        for muscle in exercise.primary_muscles
        if muscle
    }
    return total_sets >= 2 and bool(covered_muscles)


def validate_and_repair_routine_output(
    llm_output: LLMRoutineOutput,
    candidates: List[dict],
    blocked_equipment: List[str],
    pain_areas: List[PainAreaEntry],
    doms_db: Optional[Dict[str, int]] = None,
    max_total_sets: Optional[int] = None,
    time_available_min: Optional[int] = None,
    max_repairs: Optional[int] = None,
) -> Tuple[Optional[LLMRoutineOutput], StatusReasonCode]:
    """
    Validate LLM success output against the safe candidate set.
    - Repair missing or unsafe exercise_id values when safe candidates exist.
    - Return (None, "emptyCandidate") when no replacement candidate exists.
    - Return (None, "schemaError") for DOMS, set, or time rule violations.
    - Return (repaired, "none") on success.

    Guard audit results are explicitly recorded in response warnings and logs.
    """
    # Guard audit: observe constraint violations before repair.
    report = audit_routine_output(llm_output, candidates, blocked_equipment, pain_areas)
    log_guard_report(report)

    safe_candidates = [
        c for c in candidates if _candidate_is_safe(c, blocked_equipment, pain_areas)
    ]
    safe_by_id = {str(c["id"]): c for c in safe_candidates}
    if not safe_candidates:
        return None, "emptyCandidate"

    repaired = llm_output.model_copy(deep=True)
    used_ids: set[str] = set()
    violations = 0
    doms = doms_db or {}
    guard_warning_messages: List[str] = []
    repair_limit = max_repairs if max_repairs is not None else max(2, len(repaired.exercises))

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
            if violations > repair_limit:
                print(f"[Guard] repair attempt limit exceeded ({violations} > {repair_limit}) -> fallback")
                return None, "emptyCandidate"

            replacement = next(
                (c for c in safe_candidates if str(c["id"]) not in used_ids),
                None,
            )
            if replacement is None:
                print("[Guard] safe candidate exhausted -> fallback")
                return None, "emptyCandidate"

            original_id = exercise.exercise_id
            repaired.exercises[index] = _candidate_to_plan(replacement, exercise)
            used_ids.add(str(replacement["id"]))
            exercise = repaired.exercises[index]
            candidate = replacement
            guard_warning_messages.append(
                f"Guard: '{original_id}' -> '{replacement['id']}' (constraint violation replaced)"
            )

        primary = str(candidate.get("primary_muscle") or "").strip()
        doms_level = doms.get(primary, 0)
        if doms_level >= 3:
            print(f"[Guard] DOMS level 3 detected: {primary} -> fallback")
            return None, "schemaError"
        if exercise.sets < 1 or exercise.target_reps < 1 or exercise.rest_time_sec < 0:
            print(f"[Guard] invalid set/reps/rest value: {exercise.exercise_id} -> fallback")
            return None, "schemaError"
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
                print("[Guard] set budget overflow cannot be reduced -> fallback")
                return None, "schemaError"

    pre_time_trim_exercise_count = len(repaired.exercises)
    pre_time_trim_set_count = sum(exercise.sets for exercise in repaired.exercises)
    time_trim_attempted = False
    if time_available_min is not None:
        if estimate_routine_time_min(repaired.exercises) > time_available_min:
            time_trim_attempted = True
            print("[Guard] time budget exceeded -> trying set/exercise trim")
            if not trim_routine_to_time_budget(repaired, time_available_min):
                print("[Guard] time budget trim failed -> fallback")
                return None, "schemaError"

    if not _has_minimum_success_quality(repaired):
        print("[Guard] trimmed/repaired routine below minimum quality -> fallback")
        return None, "schemaError"

    final_exercise_count = len(repaired.exercises)
    final_set_count = sum(exercise.sets for exercise in repaired.exercises)
    trimmed_sets = max(0, pre_time_trim_set_count - final_set_count)
    removed_exercises = max(0, pre_time_trim_exercise_count - final_exercise_count)
    if violations or time_trim_attempted:
        print(
            "[Guard diagnostics] "
            f"repair_count={violations} trimmed_set_count={trimmed_sets} "
            f"removed_exercise_count={removed_exercises}"
        )
        repaired.warnings = [
            *repaired.warnings,
            (
                "Guard diagnostics: "
                f"repairCount={violations}, trimmedSetCount={trimmed_sets}, "
                f"removedExerciseCount={removed_exercises}"
            ),
        ]

    if guard_warning_messages:
        repaired.warnings = [*repaired.warnings, *guard_warning_messages]
    return repaired, "none"

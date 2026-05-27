"""LLM raw output ??LLMRoutineOutput ?뺤젣 ?대뙌??"""
import re
import json
from typing import Dict, List, Optional, Tuple

from .candidate_ranker import _candidate_is_safe
from .llm_router import StatusReasonCode
from .post_validation_guard import audit_routine_output, log_guard_report
from .prescription.estimator import estimate_routine_time_min
from .schemas import LLMExercisePlan, LLMRoutineOutput, PainAreaEntry

_MARKDOWN_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```")


def _try_repair_json(text: str) -> dict:
    """
    遺덉셿?꾪븳 JSON 臾몄옄?댁뿉 ?ル뒗 愿꾪샇瑜??쒖감?곸쑝濡?蹂댁땐???뚯떛???ъ떆?꾪븳??
    留덉?留??좏슚 ?좏겙 ?꾩튂源뚯?留??섎씪?대뒗 ?꾩쿂由щ룄 蹂묓뻾?쒕떎.
    """
    for suffix in ("}", "}]}", "}]}}", "}]}}]}"):
        try:
            return json.loads(text + suffix)
        except json.JSONDecodeError:
            pass
    # 留덉?留??꾩쟾??`}` ?꾩튂源뚯?留??섎씪?댁뼱 ?ъ떆??    last_brace = text.rfind("}")
    if last_brace != -1:
        try:
            return json.loads(text[: last_brace + 1])
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError("JSON ?먭? 蹂듦뎄 ?ㅽ뙣", text, 0)


def _inject_defaults(data: dict) -> dict:
    """
    Pydantic 寃利????꾨씫???꾩닔 ?꾨뱶??湲곕낯媛믪쓣 二쇱엯?쒕떎 (Graceful Degradation).
    """
    data.setdefault("summary_title", "맞춤형 AI 루틴")
    if isinstance(data.get("rationale_summary"), str):
        data["rationale_summary"] = [data["rationale_summary"]]
    data.setdefault("rationale_summary", ["?뚯썝?섏쓽 ?곗씠??湲곕컲?쇰줈 ?앹꽦??猷⑦떞?낅땲??"])
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
    LLM raw output ??LLMRoutineOutput ?뺤젣 ?대뙌??

    泥섎━ ?쒖꽌:
    1. 留덊겕?ㅼ슫 肄붾뱶?쒖뒪(```json ... ```) ?쒓굅
    2. json.loads ?뚯떛 ?쒕룄
    3. ?ㅽ뙣 ???먭? 蹂듦뎄(_try_repair_json) ?ъ떆??    4. ?ъ쟾???ㅽ뙣?섍퀬 llm??二쇱뼱吏?寃쎌슦 OutputFixingParser濡??ъ떆??    5. ?꾨씫 ?꾨뱶 湲곕낯媛?二쇱엯(_inject_defaults)
    6. Pydantic 寃利?    """
    cleaned = _MARKDOWN_FENCE_RE.sub("", raw_text).strip()

    # JSON ?뚯떛
    data: dict | None = None
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            data = _try_repair_json(cleaned)
            print("[?뺤젣 ?대뙌?? ?먭? 蹂듦뎄(bracket repair) ?깃났")
        except json.JSONDecodeError:
            if llm is not None:
                print("[?뺤젣 ?대뙌?? OutputFixingParser濡??ъ떆??..")
                try:
                    from langchain.output_parsers import OutputFixingParser
                    from langchain_core.output_parsers import PydanticOutputParser

                    base_parser = PydanticOutputParser(pydantic_object=LLMRoutineOutput)
                    fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)
                    return fixing_parser.parse(cleaned)
                except Exception as fix_e:
                    raise ValueError(f"OutputFixingParser ?ㅽ뙣: {fix_e}") from fix_e
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
        "exercise_rationale": f"{template.exercise_rationale} (?꾧?利?repair: ?덉쟾 ?꾨낫濡?援먯껜)",
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
                        "?쒓컙 ?덉궛??留욎텛湲??꾪빐 ?쇰? ?명듃 ?섎? 以꾩??듬땲??",
                    ]
                    return True

    while len(output.exercises) > 1 and estimate_routine_time_min(output.exercises) > time_available_min:
        removed = output.exercises.pop()
        output.warnings = [
            *output.warnings,
            f"?쒓컙 ?덉궛??留욎텛湲??꾪빐 {removed.exercise_name} ?대룞???쒖쇅?덉뒿?덈떎.",
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
    LLM success output???꾨낫援?湲곗??쇰줈 ?ш?利앺븳??
    - ?꾨낫 諛?exercise_id ?먮뒗 ?꾩옱 ?쒖빟???덉쟾?섏? ?딆? ?꾨낫??repair ?쒕룄
    - ?꾨컲??留롪굅???泥??꾨낫媛 ?놁쑝硫?(None, "emptyCandidate") 諛섑솚
    - DOMS/?명듃/?쒓컙 洹쒖튃 ?꾨컲 ??(None, "schemaError") 諛섑솚
    - ?깃났 ??(repaired, "none") 諛섑솚

    guard audit 寃곌낵??response payload??warnings 諛??쒕쾭 濡쒓렇??紐낆떆?곸쑝濡?湲곕줉?쒕떎.
    """
    # --- Guard: ?쒖빟 ?꾨컲 媛먯궗 (?쒖닔 愿李? 蹂???놁쓬) ---
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
                print("[Guard] ?덉쟾 ?꾨낫 ?뚯쭊 -> fallback")
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
            print(f"[Guard] DOMS level 3 媛먯?: {primary} -> fallback")
            return None, "schemaError"
        if exercise.sets < 1 or exercise.target_reps < 1 or exercise.rest_time_sec < 0:
            print(f"[Guard] ?좏슚?섏? ?딆? ?명듃/諛섎났 媛? {exercise.exercise_id} -> fallback")
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
                print(f"[Guard] ?명듃 ?덉궛 珥덇낵 ?댁냼 遺덇? -> fallback")
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

"""Normalize raw LLM output into LLMRoutineOutput."""
import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from pydantic import ValidationError

from .candidate_ranker import _candidate_is_safe
from .llm_router import StatusReasonCode
from .post_validation_guard import audit_routine_output, log_guard_report
from .prescription.estimator import estimate_routine_time_min
from .schemas import LLMExercisePlan, LLMRoutineOutput, PainAreaEntry

logger = logging.getLogger(__name__)

_MARKDOWN_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```")
_ALLOWED_VALIDATION_FIELDS = {
    "routine_blocks",
    "exercises",
    "exercise_id",
    "exercise_name",
    "target_reps",
    "sets",
    "rest_time_sec",
    "target_weight_kg",
    "exercise_rationale",
    "summary_title",
    "rationale_summary",
    "total_estimated_time",
    "warnings",
}


@dataclass(frozen=True)
class ParseResult:
    success: bool
    data: Optional[dict] = None
    subtype: str = "unknown"
    sanitized_error_category: Optional[str] = None
    json_decode_error_category: Optional[str] = None
    json_decode_recovery_attempted: bool = False
    json_decode_recovery_succeeded: bool = False
    json_decode_recovery_strategy: str = "none"


class SchemaRepairError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        parse_failure_subtype: str = "unknown",
        schema_validation_error_category: Optional[str] = None,
        schema_validation_field_names: Optional[list[str]] = None,
        repair_failure_reason: Optional[str] = None,
        json_decode_error_category: Optional[str] = None,
        json_decode_recovery_attempted: bool = False,
        json_decode_recovery_succeeded: bool = False,
        json_decode_recovery_strategy: str = "none",
    ):
        super().__init__(message)
        self.parse_failure_subtype = parse_failure_subtype
        self.schema_validation_error_category = schema_validation_error_category
        self.schema_validation_field_names = schema_validation_field_names or []
        self.repair_failure_reason = repair_failure_reason or parse_failure_subtype
        self.json_decode_error_category = json_decode_error_category
        self.json_decode_recovery_attempted = json_decode_recovery_attempted
        self.json_decode_recovery_succeeded = json_decode_recovery_succeeded
        self.json_decode_recovery_strategy = json_decode_recovery_strategy


def _non_empty_text(value) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def extract_raw_llm_output_from_exception(exc: BaseException | None) -> tuple[str | None, str]:
    """Recover raw LLM text for internal repair without logging or storing it."""
    seen: set[int] = set()

    def visit(current: BaseException | None) -> tuple[str | None, str]:
        if current is None or id(current) in seen:
            return None, "none"
        seen.add(id(current))

        for attr, source in (
            ("llm_output", "exception_llm_output"),
            ("observation", "exception_observation"),
        ):
            text = _non_empty_text(getattr(current, attr, None))
            if text is not None:
                return text, source

        for arg in getattr(current, "args", ()) or ():
            text = _non_empty_text(arg)
            if text is not None:
                return text, "exception_args"

        text, source = visit(getattr(current, "__cause__", None))
        if text is not None:
            return text, source
        return visit(getattr(current, "__context__", None))

    return visit(exc)


def _strip_markdown_fence(text: str) -> tuple[str, bool]:
    stripped = text.strip()
    without_fence = _MARKDOWN_FENCE_RE.sub("", stripped).strip()
    return without_fence, without_fence != stripped


def _extract_first_balanced_json_object(text: str) -> tuple[Optional[str], bool]:
    start = text.find("{")
    if start == -1:
        return None, False

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1], (start != 0 or index != len(text) - 1)
    return None, False


def _has_balanced_delimiters(text: str, open_char: str, close_char: str) -> bool:
    depth = 0
    in_string = False
    escape = False
    for char in text:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth < 0:
                return False
    return depth == 0 and not in_string


def _maybe_trailing_comma_outside_string(text: str) -> bool:
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == ",":
            remainder = text[index + 1 :].lstrip()
            if remainder.startswith("}") or remainder.startswith("]"):
                return True
    return False


def _remove_trailing_commas_outside_strings(text: str) -> str:
    chars: list[str] = []
    in_string = False
    escape = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            chars.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            chars.append(char)
            index += 1
            continue
        if char == ",":
            lookahead = index + 1
            while lookahead < len(text) and text[lookahead].isspace():
                lookahead += 1
            if lookahead < len(text) and text[lookahead] in ("}", "]"):
                index += 1
                continue
        chars.append(char)
        index += 1
    return "".join(chars)


def _contains_python_literal_outside_strings(text: str) -> bool:
    scrubbed = []
    in_string = False
    escape = False
    for char in text:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            scrubbed.append(" ")
            continue
        if char == '"':
            in_string = True
            scrubbed.append(" ")
        else:
            scrubbed.append(char)
    return bool(re.search(r"\b(None|True|False)\b", "".join(scrubbed)))


def _single_quotes_suspected(text: str) -> bool:
    return "'" in text and '"' not in text[: max(len(text), 1)]


def classify_json_decode_error(error: json.JSONDecodeError, text: str) -> str:
    message = str(error.msg or "").lower()
    if "unterminated string" in message:
        return "json_decode_unterminated_string"
    if "invalid control character" in message:
        return "json_decode_invalid_control_character"
    if "extra data" in message:
        return "json_decode_extra_data"
    if "expecting property name" in message:
        if _maybe_trailing_comma_outside_string(text):
            return "json_decode_trailing_comma"
        if _single_quotes_suspected(text):
            return "json_decode_single_quotes_suspected"
        return "json_decode_expect_property_name"
    if "expecting value" in message:
        if _contains_python_literal_outside_strings(text):
            return "json_decode_python_literal"
        return "json_decode_expect_value"
    if not _has_balanced_delimiters(text, "{", "}"):
        return "json_decode_unbalanced_braces"
    if not _has_balanced_delimiters(text, "[", "]"):
        return "json_decode_unbalanced_brackets"
    if "```" in text:
        return "json_decode_markdown_or_prose_residual"
    return "json_decode_unknown"


def _json_decode_failure_result(error: json.JSONDecodeError, text: str) -> ParseResult:
    category = classify_json_decode_error(error, text)
    return ParseResult(
        False,
        subtype="json_decode_error",
        sanitized_error_category="json_decode_error",
        json_decode_error_category=category,
    )


def _loads_with_safe_recovery(candidate: str) -> ParseResult:
    try:
        data = json.loads(candidate)
        if not isinstance(data, dict):
            return ParseResult(False, subtype="non_object_json", sanitized_error_category="non_object_json")
        return ParseResult(True, data=data)
    except json.JSONDecodeError as first_error:
        if _maybe_trailing_comma_outside_string(candidate):
            repaired = _remove_trailing_commas_outside_strings(candidate)
            try:
                data = json.loads(repaired)
                if not isinstance(data, dict):
                    return ParseResult(False, subtype="non_object_json", sanitized_error_category="non_object_json")
                return ParseResult(
                    True,
                    data=data,
                    subtype="json_decode_error",
                    json_decode_error_category="json_decode_trailing_comma",
                    json_decode_recovery_attempted=True,
                    json_decode_recovery_succeeded=True,
                    json_decode_recovery_strategy="trailing_comma_removed",
                )
            except json.JSONDecodeError:
                return ParseResult(
                    False,
                    subtype="json_decode_error",
                    sanitized_error_category="json_decode_error",
                    json_decode_error_category=classify_json_decode_error(first_error, candidate),
                    json_decode_recovery_attempted=True,
                    json_decode_recovery_succeeded=False,
                    json_decode_recovery_strategy="trailing_comma_removed",
                )
        return _json_decode_failure_result(first_error, candidate)


def parse_json_candidate(raw_text: str) -> ParseResult:
    if raw_text is None or not str(raw_text).strip():
        return ParseResult(False, subtype="empty_response", sanitized_error_category="empty_response")

    cleaned, had_fence = _strip_markdown_fence(str(raw_text))
    candidate = cleaned
    had_prose = False
    if candidate.startswith("{"):
        result = _loads_with_safe_recovery(candidate)
        if result.success:
            subtype = result.subtype
            if had_fence and subtype == "unknown":
                subtype = "markdown_fence_wrapped"
            return ParseResult(
                True,
                data=result.data,
                subtype=subtype,
                json_decode_error_category=result.json_decode_error_category,
                json_decode_recovery_attempted=result.json_decode_recovery_attempted,
                json_decode_recovery_succeeded=result.json_decode_recovery_succeeded,
                json_decode_recovery_strategy=result.json_decode_recovery_strategy,
            )
        if result.subtype != "json_decode_error":
            return result
        try:
            data = _try_repair_json(candidate)
            return ParseResult(
                True,
                data=data,
                subtype="json_decode_error",
                json_decode_error_category=result.json_decode_error_category,
            )
        except json.JSONDecodeError:
            return result

    if not candidate.startswith("{") or not candidate.endswith("}"):
        try:
            parsed = json.loads(candidate)
            if not isinstance(parsed, dict):
                return ParseResult(False, subtype="non_object_json", sanitized_error_category="non_object_json")
        except json.JSONDecodeError as decode_error:
            decode_category = classify_json_decode_error(decode_error, candidate)
            if "```" in candidate and not had_fence:
                return ParseResult(
                    False,
                    subtype="json_decode_error",
                    sanitized_error_category="json_decode_error",
                    json_decode_error_category="json_decode_markdown_or_prose_residual",
                )
            pass
        extracted, had_prose = _extract_first_balanced_json_object(candidate)
        if extracted is None:
            return ParseResult(
                False,
                subtype="json_decode_error",
                sanitized_error_category="json_decode_error",
                json_decode_error_category=decode_category if "decode_category" in locals() else "json_decode_unknown",
            )
        candidate = extracted

    result = _loads_with_safe_recovery(candidate)
    if not result.success:
        if result.subtype != "json_decode_error":
            return result
        try:
            data = _try_repair_json(candidate)
        except json.JSONDecodeError:
            return result
    else:
        data = result.data

    if not isinstance(data, dict):
        return ParseResult(False, subtype="non_object_json", sanitized_error_category="non_object_json")

    subtype = result.subtype if "result" in locals() and result.subtype != "unknown" else "unknown"
    if had_fence:
        subtype = "markdown_fence_wrapped"
    elif had_prose:
        subtype = "leading_or_trailing_prose"
    return ParseResult(
        True,
        data=data,
        subtype=subtype,
        json_decode_error_category=result.json_decode_error_category if "result" in locals() else None,
        json_decode_recovery_attempted=result.json_decode_recovery_attempted if "result" in locals() else False,
        json_decode_recovery_succeeded=result.json_decode_recovery_succeeded if "result" in locals() else False,
        json_decode_recovery_strategy=result.json_decode_recovery_strategy if "result" in locals() else "none",
    )


def classify_validation_error(error: ValidationError) -> tuple[str, list[str]]:
    categories: set[str] = set()
    fields: set[str] = set()
    for item in error.errors():
        loc = item.get("loc") or ()
        leaf = next(
            (part for part in reversed(loc) if isinstance(part, str) and part in _ALLOWED_VALIDATION_FIELDS),
            None,
        )
        if leaf is not None:
            fields.add(leaf)
        error_type = str(item.get("type") or "")
        if error_type == "missing":
            categories.add("missing_required_field")
        elif "list" in error_type:
            categories.add("invalid_list_shape")
        elif any(token in error_type for token in ("int", "float", "string", "bool", "dict", "model_type")):
            categories.add("wrong_field_type")
        else:
            categories.add("unknown_validation_error")

    if "missing_required_field" in categories:
        category = "missing_required_field"
    elif "wrong_field_type" in categories:
        category = "wrong_field_type"
    elif "invalid_list_shape" in categories:
        category = "invalid_list_shape"
    else:
        category = "unknown_validation_error"
    return category, sorted(fields)


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

    if not isinstance(data["exercises"], list):
        return data

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
    parse_result = parse_json_candidate(raw_text)
    if not parse_result.success:
        if llm is not None:
            logger.warning("[LLM parser] retrying with OutputFixingParser")
            cleaned, _had_fence = _strip_markdown_fence(str(raw_text or ""))
            try:
                from langchain.output_parsers import OutputFixingParser
                from langchain_core.output_parsers import PydanticOutputParser

                base_parser = PydanticOutputParser(pydantic_object=LLMRoutineOutput)
                fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)
                return fixing_parser.parse(cleaned)
            except Exception as fix_e:
                raise SchemaRepairError(
                    "OutputFixingParser failed",
                    parse_failure_subtype=parse_result.subtype,
                    repair_failure_reason=parse_result.subtype,
                    json_decode_error_category=parse_result.json_decode_error_category,
                    json_decode_recovery_attempted=parse_result.json_decode_recovery_attempted,
                    json_decode_recovery_succeeded=parse_result.json_decode_recovery_succeeded,
                    json_decode_recovery_strategy=parse_result.json_decode_recovery_strategy,
                ) from fix_e
        raise SchemaRepairError(
            "LLM output JSON parse failed",
            parse_failure_subtype=parse_result.subtype,
            repair_failure_reason=parse_result.subtype,
            json_decode_error_category=parse_result.json_decode_error_category,
            json_decode_recovery_attempted=parse_result.json_decode_recovery_attempted,
            json_decode_recovery_succeeded=parse_result.json_decode_recovery_succeeded,
            json_decode_recovery_strategy=parse_result.json_decode_recovery_strategy,
        )

    # Inject defaults before Pydantic validation so loose local-model output can be salvaged.
    data = _inject_defaults(parse_result.data or {})
    try:
        return LLMRoutineOutput(**data)
    except ValidationError as validation_error:
        category, fields = classify_validation_error(validation_error)
        raise SchemaRepairError(
            "LLM output schema validation failed",
            parse_failure_subtype="pydantic_validation_error",
            schema_validation_error_category=category,
            schema_validation_field_names=fields,
            repair_failure_reason=category,
        ) from validation_error


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
                logger.warning("[Guard] repair attempt limit exceeded (%s > %s) -> fallback", violations, repair_limit)
                return None, "emptyCandidate"

            replacement = next(
                (c for c in safe_candidates if str(c["id"]) not in used_ids),
                None,
            )
            if replacement is None:
                logger.warning("[Guard] safe candidate exhausted -> fallback")
                return None, "emptyCandidate"

            original_id = exercise.exercise_id
            repaired.exercises[index] = _candidate_to_plan(replacement, exercise)
            used_ids.add(str(replacement["id"]))
            exercise = repaired.exercises[index]
            candidate = replacement
            guard_warning_messages.append(
                f"Guard: '{original_id}' -> '{replacement['id']}' 제약 위반으로 replaced 처리했습니다."
            )

        primary = str(candidate.get("primary_muscle") or "").strip()
        doms_level = doms.get(primary, 0)
        if doms_level >= 3:
            logger.warning("[Guard] DOMS level 3 detected: %s -> fallback", primary)
            return None, "schemaError"
        if exercise.sets < 1 or exercise.target_reps < 1 or exercise.rest_time_sec < 0:
            logger.warning("[Guard] invalid set/reps/rest value: %s -> fallback", exercise.exercise_id)
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
                logger.warning("[Guard] set budget overflow cannot be reduced -> fallback")
                return None, "schemaError"

    pre_time_trim_exercise_count = len(repaired.exercises)
    pre_time_trim_set_count = sum(exercise.sets for exercise in repaired.exercises)
    time_trim_attempted = False
    if time_available_min is not None:
        if estimate_routine_time_min(repaired.exercises) > time_available_min:
            time_trim_attempted = True
            logger.warning("[Guard] time budget exceeded -> trying set/exercise trim")
            if not trim_routine_to_time_budget(repaired, time_available_min):
                logger.warning("[Guard] time budget trim failed -> fallback")
                return None, "schemaError"

    if not _has_minimum_success_quality(repaired):
        logger.warning("[Guard] trimmed/repaired routine below minimum quality -> fallback")
        return None, "schemaError"

    final_exercise_count = len(repaired.exercises)
    final_set_count = sum(exercise.sets for exercise in repaired.exercises)
    trimmed_sets = max(0, pre_time_trim_set_count - final_set_count)
    removed_exercises = max(0, pre_time_trim_exercise_count - final_exercise_count)
    if violations or time_trim_attempted:
        logger.info(
            "[Guard diagnostics] repair_count=%s trimmed_set_count=%s removed_exercise_count=%s",
            violations,
            trimmed_sets,
            removed_exercises,
        )
        repaired.warnings = [
            *repaired.warnings,
            (
                "Guard 진단: "
                f"repairCount={violations}, trimmedSetCount={trimmed_sets}, "
                f"removedExerciseCount={removed_exercises}"
            ),
        ]

    if guard_warning_messages:
        repaired.warnings = [*repaired.warnings, *guard_warning_messages]
    return repaired, "none"

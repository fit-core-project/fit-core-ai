"""Normalize raw LLM output into LLMRoutineOutput."""
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from .candidate_ranker import _candidate_is_safe, candidate_selection_family_key, candidate_selection_pattern_key
from .exercise_display import candidate_display_name
from .llm_router import StatusReasonCode
from .option_resolver import ADJUNCT_OPTION_PRIORITY, candidate_matches_adjunct_option, canonical_adjunct_options
from .experience_policy import automatic_recommendation_eligibility
from .post_validation_guard import audit_routine_output, log_guard_report
from .prescription.adjuncts import is_working_prescription
from .prescription.estimator import estimate_routine_time_min
from .prescription.unilateral import apply_reviewed_unilateral_metadata, has_reviewed_unilateral_metadata
from .pull_composition import complementary_pair, reviewed_pull_family_key
from .specialization_policy import focus_variant_family_key
from .schemas import (
    LLMExercisePlan,
    LLMRoutineOutput,
    LLMSubstitutionCandidate,
    PainAreaEntry,
)

logger = logging.getLogger(__name__)

_MARKDOWN_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```")
_ALLOWED_VALIDATION_FIELDS = {
    "routine_blocks",
    "exercises",
    "exercise_id",
    "exercise_name",
    "movement_pattern",
    "movement_type",
    "primary_muscles",
    "equipment_type",
    "target_reps",
    "sets",
    "rest_time_sec",
    "target_weight_kg",
    "target_rir",
    "exercise_rationale",
    "substitution_candidates",
    "reason",
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
        schema_root_type: Optional[str] = None,
        schema_top_level_keys: Optional[list[str]] = None,
        schema_exercises_value_type: Optional[str] = None,
        schema_exercises_nested_keys: Optional[list[str]] = None,
        schema_exercises_list_length: Optional[int] = None,
        schema_first_exercise_item_type: Optional[str] = None,
        schema_first_exercise_keys: Optional[list[str]] = None,
        schema_first_exercise_field_types: Optional[dict[str, str]] = None,
        schema_exercise_item_types: Optional[list[str]] = None,
        schema_exercise_key_union: Optional[list[str]] = None,
        schema_exercise_field_type_sets: Optional[dict[str, list[str]]] = None,
        schema_validation_loc_summaries: Optional[list[str]] = None,
        schema_validation_error_types: Optional[list[str]] = None,
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
        self.schema_root_type = schema_root_type
        self.schema_top_level_keys = schema_top_level_keys or []
        self.schema_exercises_value_type = schema_exercises_value_type
        self.schema_exercises_nested_keys = schema_exercises_nested_keys or []
        self.schema_exercises_list_length = schema_exercises_list_length
        self.schema_first_exercise_item_type = schema_first_exercise_item_type
        self.schema_first_exercise_keys = schema_first_exercise_keys or []
        self.schema_first_exercise_field_types = schema_first_exercise_field_types or {}
        self.schema_exercise_item_types = schema_exercise_item_types or []
        self.schema_exercise_key_union = schema_exercise_key_union or []
        self.schema_exercise_field_type_sets = schema_exercise_field_type_sets or {}
        self.schema_validation_loc_summaries = schema_validation_loc_summaries or []
        self.schema_validation_error_types = schema_validation_error_types or []


_SCHEMA_SHAPE_KEY_ALLOWLIST = _ALLOWED_VALIDATION_FIELDS | {
    "selected_exercises",
    "recommendation_summary",
    "items",
    "list",
    "plans",
    "routine",
    "routine_blocks",
    "routineBlocks",
}


def _shape_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, str):
        return "str"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _shape_keys(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    keys: list[str] = []
    has_other = False
    for key in value:
        key_text = str(key)
        if key_text in _SCHEMA_SHAPE_KEY_ALLOWLIST:
            keys.append(key_text)
        else:
            has_other = True
    if has_other:
        keys.append("__other__")
    return sorted(set(keys))


def summarize_llm_schema_shape(data: Any) -> dict[str, Any]:
    """Return structural schema diagnostics without storing raw LLM values."""
    summary: dict[str, Any] = {
        "schema_root_type": _shape_type_name(data),
        "schema_top_level_keys": _shape_keys(data),
        "schema_exercises_value_type": None,
        "schema_exercises_nested_keys": [],
        "schema_exercises_list_length": None,
        "schema_first_exercise_item_type": None,
        "schema_first_exercise_keys": [],
        "schema_first_exercise_field_types": {},
        "schema_exercise_item_types": [],
        "schema_exercise_key_union": [],
        "schema_exercise_field_type_sets": {},
    }
    if not isinstance(data, dict):
        return summary

    exercises = data.get("exercises")
    summary["schema_exercises_value_type"] = _shape_type_name(exercises)
    if isinstance(exercises, list):
        summary["schema_exercises_list_length"] = len(exercises)
        summary["schema_exercise_item_types"] = sorted(set(_shape_type_name(item) for item in exercises))
        key_union: list[str] = []
        field_type_sets: dict[str, set[str]] = {}
        for item in exercises:
            if not isinstance(item, dict):
                continue
            key_union.extend(_shape_keys(item))
            for key in _ALLOWED_VALIDATION_FIELDS:
                if key in item:
                    field_type_sets.setdefault(key, set()).add(_shape_type_name(item.get(key)))
        summary["schema_exercise_key_union"] = sorted(set(key_union))
        summary["schema_exercise_field_type_sets"] = {
            key: sorted(values)
            for key, values in sorted(field_type_sets.items())
        }
        if exercises:
            first_item = exercises[0]
            summary["schema_first_exercise_item_type"] = _shape_type_name(first_item)
            if isinstance(first_item, dict):
                summary["schema_first_exercise_keys"] = _shape_keys(first_item)
                summary["schema_first_exercise_field_types"] = {
                    key: _shape_type_name(first_item.get(key))
                    for key in sorted(_ALLOWED_VALIDATION_FIELDS)
                    if key in first_item
                }
    elif isinstance(exercises, dict):
        nested_keys = _shape_keys(exercises)
        if exercises and all(isinstance(item, dict) for item in exercises.values()):
            nested_keys = sorted(set(nested_keys + ["__dict_of_objects__"]))
        summary["schema_exercises_nested_keys"] = nested_keys
    return summary


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


def _iter_balanced_json_objects(text: str) -> List[tuple[str, bool]]:
    """Yield balanced JSON-looking object substrings in textual order.

    Local models sometimes include a small example object before the actual
    answer. Parsing only the first balanced ``{...}`` can therefore fail even
    when a later object is valid. This helper keeps raw text out of telemetry
    while allowing deterministic retry over candidate object spans.
    """
    objects: List[tuple[str, bool]] = []
    search_from = 0
    while search_from < len(text):
        start = text.find("{", search_from)
        if start == -1:
            break

        depth = 0
        in_string = False
        escape = False
        found_end: Optional[int] = None
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
                    found_end = index
                    break
        if found_end is None:
            break
        objects.append((text[start : found_end + 1], (start != 0 or found_end != len(text) - 1)))
        search_from = start + 1
    return objects


def _extract_json_tail_from_first_object_start(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    tail = text[start:].strip()
    return tail or None


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


def _fill_missing_json_values_outside_strings(text: str) -> str:
    """Replace empty JSON values such as ``"key": ,`` with ``"key": null,``."""
    chars: list[str] = []
    in_string = False
    escape = False
    index = 0
    while index < len(text):
        char = text[index]
        chars.append(char)
        if in_string:
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
            index += 1
            continue
        if char == ":":
            lookahead = index + 1
            while lookahead < len(text) and text[lookahead].isspace():
                lookahead += 1
            if lookahead < len(text) and text[lookahead] in ",}]":
                chars.append(" null")
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
        first_category = classify_json_decode_error(first_error, candidate)
        if first_category == "json_decode_expect_value" and any(
            token in candidate
            for token in (
                '"exercises"',
                '"routine_blocks"',
                '"routineBlocks"',
                '"exercise_id"',
                '"exerciseId"',
            )
        ):
            repaired = _fill_missing_json_values_outside_strings(candidate)
            if repaired != candidate:
                try:
                    data = json.loads(repaired)
                    if not isinstance(data, dict):
                        return ParseResult(False, subtype="non_object_json", sanitized_error_category="non_object_json")
                    return ParseResult(
                        True,
                        data=data,
                        subtype="json_decode_error",
                        json_decode_error_category=first_category,
                        json_decode_recovery_attempted=True,
                        json_decode_recovery_succeeded=True,
                        json_decode_recovery_strategy="missing_json_values_filled_null",
                    )
                except json.JSONDecodeError:
                    return ParseResult(
                        False,
                        subtype="json_decode_error",
                        sanitized_error_category="json_decode_error",
                        json_decode_error_category=first_category,
                        json_decode_recovery_attempted=True,
                        json_decode_recovery_succeeded=False,
                        json_decode_recovery_strategy="missing_json_values_filled_null",
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
        balanced_objects = _iter_balanced_json_objects(candidate)
        extracted, had_prose = balanced_objects[0] if balanced_objects else (None, False)
        if extracted is None:
            tail = _extract_json_tail_from_first_object_start(candidate)
            if tail is not None:
                try:
                    data = _try_repair_json(tail)
                    if isinstance(data, dict):
                        return ParseResult(
                            True,
                            data=data,
                            subtype="json_decode_error",
                            json_decode_error_category=(
                                decode_category if "decode_category" in locals() else "json_decode_unbalanced_braces"
                            ),
                            json_decode_recovery_attempted=True,
                            json_decode_recovery_succeeded=True,
                            json_decode_recovery_strategy="truncated_json_tail_repaired",
                        )
                except json.JSONDecodeError:
                    pass
            return ParseResult(
                False,
                subtype="json_decode_error",
                sanitized_error_category="json_decode_error",
                json_decode_error_category=decode_category if "decode_category" in locals() else "json_decode_unknown",
            )
        for candidate_object, object_had_prose in balanced_objects:
            object_result = _loads_with_safe_recovery(candidate_object)
            if object_result.success and isinstance(object_result.data, dict):
                return ParseResult(
                    True,
                    data=object_result.data,
                    subtype="leading_or_trailing_prose" if object_had_prose else object_result.subtype,
                    json_decode_error_category=object_result.json_decode_error_category,
                    json_decode_recovery_attempted=object_result.json_decode_recovery_attempted,
                    json_decode_recovery_succeeded=object_result.json_decode_recovery_succeeded,
                    json_decode_recovery_strategy=object_result.json_decode_recovery_strategy,
                )
            if object_result.subtype == "json_decode_error":
                try:
                    data = _try_repair_json(candidate_object)
                    if isinstance(data, dict):
                        return ParseResult(
                            True,
                            data=data,
                            subtype="json_decode_error",
                            json_decode_error_category=object_result.json_decode_error_category,
                            json_decode_recovery_attempted=True,
                            json_decode_recovery_succeeded=True,
                            json_decode_recovery_strategy="balanced_object_repaired",
                        )
                except json.JSONDecodeError:
                    continue
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


def summarize_validation_error_shapes(error: ValidationError) -> tuple[list[str], list[str]]:
    loc_summaries: set[str] = set()
    error_types: set[str] = set()
    for item in error.errors():
        error_type = str(item.get("type") or "unknown")
        error_types.add(error_type)
        loc = item.get("loc") or ()
        parts: list[str] = []
        for part in loc:
            if isinstance(part, int):
                parts.append("*")
            elif isinstance(part, str) and part in _ALLOWED_VALIDATION_FIELDS:
                parts.append(part)
            elif isinstance(part, str) and part in _SCHEMA_SHAPE_KEY_ALLOWLIST:
                parts.append(part)
            else:
                parts.append("__other__")
        loc_summaries.add(".".join(parts) if parts else "__root__")
    return sorted(loc_summaries), sorted(error_types)


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
    top_level_aliases = {
        "totalEstimatedTime": "total_estimated_time",
        "summaryTitle": "summary_title",
        "rationaleSummary": "rationale_summary",
    }
    for alias, canonical in top_level_aliases.items():
        if canonical not in data and alias in data:
            data[canonical] = data[alias]

    if "exercises" not in data and isinstance(data.get("selected_exercises"), (list, dict)):
        data["exercises"] = data["selected_exercises"]
    if "rationale_summary" not in data and "recommendation_summary" in data:
        summary = data.get("recommendation_summary")
        data["rationale_summary"] = summary if isinstance(summary, list) else [str(summary)]
    data.setdefault("summary_title", "맞춤형 AI 루틴")
    if isinstance(data.get("rationale_summary"), str):
        data["rationale_summary"] = [data["rationale_summary"]]
    data.setdefault("rationale_summary", ["회원 데이터 기반으로 생성한 루틴입니다."])
    data.setdefault("warnings", [])
    if isinstance(data.get("warnings"), str):
        data["warnings"] = [data["warnings"]]
    data.setdefault("total_estimated_time", 45)
    if isinstance(data.get("total_estimated_time"), str):
        match = re.search(r"\d+(?:\.\d+)?", data["total_estimated_time"])
        data["total_estimated_time"] = int(float(match.group())) if match else 45
    elif isinstance(data.get("total_estimated_time"), float):
        data["total_estimated_time"] = int(data["total_estimated_time"])
    elif isinstance(data.get("total_estimated_time"), (dict, list)):
        data["total_estimated_time"] = 45
    data.setdefault("exercises", [])

    if isinstance(data["exercises"], dict):
        for key in ("items", "list", "plans", "routine", "routine_blocks", "routineBlocks"):
            value = data["exercises"].get(key)
            if isinstance(value, list):
                data["exercises"] = value
                break
        else:
            dict_values = list(data["exercises"].values())
            if dict_values and all(isinstance(item, dict) for item in dict_values):
                data["exercises"] = dict_values

    if not isinstance(data["exercises"], list):
        return data

    exercise_aliases = {
        "exerciseId": "exercise_id",
        "exerciseName": "exercise_name",
        "movementPattern": "movement_pattern",
        "movementType": "movement_type",
        "primaryMuscles": "primary_muscles",
        "equipmentType": "equipment_type",
        "targetWeightKg": "target_weight_kg",
        "targetReps": "target_reps",
        "restTimeSec": "rest_time_sec",
        "targetRir": "target_rir",
        "exerciseRationale": "exercise_rationale",
        "substitutionCandidates": "substitution_candidates",
    }

    normalized_exercises: list[dict[str, Any]] = []
    for ex in data["exercises"]:
        if not isinstance(ex, dict):
            if isinstance(ex, str) and ex.strip():
                ex = {"exercise_name": ex.strip()}
            else:
                continue
        for alias, canonical in exercise_aliases.items():
            if canonical not in ex and alias in ex:
                ex[canonical] = ex[alias]
        if "exercise_id" not in ex and "id" in ex:
            ex["exercise_id"] = ex["id"]
        if "exercise_name" not in ex and "name" in ex:
            ex["exercise_name"] = ex["name"]
        if "exercise_rationale" not in ex and "reasoning" in ex:
            ex["exercise_rationale"] = ex["reasoning"]
        exercise_name_for_id = ex.get("exercise_name")
        if not isinstance(exercise_name_for_id, str) or not exercise_name_for_id.strip():
            exercise_name_for_id = "unknown"
        ex.setdefault(
            "exercise_id",
            re.sub(r"\s+", "_", exercise_name_for_id).lower(),
        )
        ex["exercise_id"] = str(ex["exercise_id"])
        ex.setdefault("primary_muscles", [])
        if isinstance(ex["primary_muscles"], str):
            ex["primary_muscles"] = [ex["primary_muscles"]]
        elif isinstance(ex["primary_muscles"], dict):
            ex["primary_muscles"] = [str(item) for item in ex["primary_muscles"].values() if str(item).strip()]
        elif isinstance(ex["primary_muscles"], list):
            ex["primary_muscles"] = [str(item) for item in ex["primary_muscles"] if str(item).strip()]
        else:
            ex["primary_muscles"] = []
        ex.setdefault("target_rir", 2)
        ex.setdefault("substitution_candidates", [])
        if isinstance(ex["substitution_candidates"], dict):
            values = list(ex["substitution_candidates"].values())
            ex["substitution_candidates"] = values if all(isinstance(item, dict) for item in values) else []
        elif isinstance(ex["substitution_candidates"], list):
            ex["substitution_candidates"] = [
                item
                for item in ex["substitution_candidates"]
                if isinstance(item, dict)
            ]
        else:
            ex["substitution_candidates"] = []
        ex.setdefault("exercise_rationale", "AI 추천 운동")
        ex.setdefault("rest_time_sec", 90)
        ex.setdefault("sets", 3)
        ex.setdefault("target_reps", 10)
        for optional_text_key in ("movement_pattern", "movement_type", "equipment_type"):
            value = ex.get(optional_text_key)
            if isinstance(value, list):
                text_items = [str(item).strip() for item in value if str(item).strip()]
                ex[optional_text_key] = ", ".join(text_items) if text_items else None
            elif isinstance(value, dict):
                text_items = [str(item).strip() for item in value.values() if str(item).strip()]
                ex[optional_text_key] = ", ".join(text_items) if text_items else None
        for numeric_key in ("target_reps", "sets", "rest_time_sec", "target_rir"):
            value = ex.get(numeric_key)
            if isinstance(value, str):
                match = re.search(r"\d+", value)
                if match:
                    ex[numeric_key] = int(match.group())
            elif isinstance(value, float):
                ex[numeric_key] = int(value)
            elif isinstance(value, dict):
                ex[numeric_key] = {"target_reps": 10, "sets": 3, "rest_time_sec": 90, "target_rir": 2}[numeric_key]
            elif isinstance(value, list):
                first_number = None
                for item in value:
                    match = re.search(r"\d+(?:\.\d+)?", str(item))
                    if match:
                        first_number = int(float(match.group()))
                        break
                ex[numeric_key] = first_number or {"target_reps": 10, "sets": 3, "rest_time_sec": 90, "target_rir": 2}[numeric_key]
        if isinstance(ex.get("target_weight_kg"), str):
            lowered = ex["target_weight_kg"].strip().lower()
            if lowered in {"", "none", "null", "bodyweight", "맨몸"}:
                ex["target_weight_kg"] = None
            else:
                match = re.search(r"\d+(?:\.\d+)?", lowered)
                ex["target_weight_kg"] = float(match.group()) if match else None
        elif isinstance(ex.get("target_weight_kg"), dict):
            ex["target_weight_kg"] = None
        normalized_exercises.append(ex)

    data["exercises"] = normalized_exercises

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
        loc_summaries, error_types = summarize_validation_error_shapes(validation_error)
        shape_summary = summarize_llm_schema_shape(data)
        raise SchemaRepairError(
            "LLM output schema validation failed",
            parse_failure_subtype="pydantic_validation_error",
            schema_validation_error_category=category,
            schema_validation_field_names=fields,
            schema_validation_loc_summaries=loc_summaries,
            schema_validation_error_types=error_types,
            repair_failure_reason=category,
            **shape_summary,
        ) from validation_error


def _candidate_to_plan(candidate: dict, template: LLMExercisePlan) -> LLMExercisePlan:
    display_name = _candidate_display_name(candidate) or template.exercise_name
    plan = template.model_copy(update={
        "exercise_id": str(candidate["id"]),
        "exercise_name": display_name,
        "movement_pattern": candidate.get("movement_pattern") or template.movement_pattern,
        "movement_type": candidate.get("movement_type") or template.movement_type,
        "primary_muscles": [candidate["primary_muscle"]] if candidate.get("primary_muscle") else [],
        "equipment_type": candidate.get("equipment_req"),
        "exercise_rationale": f"{display_name}은(는) 비슷한 운동 중 현재 조건에 더 잘 맞아 대체 운동으로 선택했습니다.",
    })
    if template._unilateral_prescription is not None or has_reviewed_unilateral_metadata(candidate):
        from .prescription.metadata import apply_candidate_prescription_metadata

        # A new exercise must not inherit another modality or its tempo.
        was_working = plan._prescription_set_type == 'working'
        plan._prescription_set_type = 'working'
        plan._target_duration_sec = None
        plan._interval_work_sec = None
        plan._interval_recovery_sec = None
        plan._interval_rounds = None
        plan._target_distance_m = None
        plan._training_method_code = None
        plan._training_method_instruction = None
        plan._unilateral_set_overrides = None
        if not was_working:
            # Same base repetition/RIR seed as _candidate_plan, not a carried
            # duration placeholder. Goal/readiness adjustment stays downstream.
            plan.target_reps = 10
            plan.target_rir = 2
            plan.target_weight_kg = None
        return apply_candidate_prescription_metadata(plan, candidate)
    return apply_reviewed_unilateral_metadata(plan, candidate)


def _candidate_display_name(candidate: dict) -> str:
    return candidate_display_name(candidate)


def _candidate_display_key(candidate: dict) -> str:
    return re.sub(r"[\s/_()·,.-]+", "", _candidate_display_name(candidate).lower())


def _candidate_primary(candidate: Optional[dict]) -> str:
    if candidate is None:
        return ""
    return str(candidate.get("primary_muscle") or "").strip()


def _exercise_primary(exercise: LLMExercisePlan) -> str:
    if exercise.primary_muscles:
        return str(exercise.primary_muscles[0] or "").strip()
    return ""


def _sanitize_substitution_candidates(
    exercise: LLMExercisePlan,
    main_candidate: Optional[dict],
    safe_by_id: Dict[str, dict],
) -> tuple[List[LLMSubstitutionCandidate], int]:
    """Keep only safe, related candidates from the server-ranked pool."""

    explicit_options = {
        str(
            option.get("exercise_id")
            or option.get("exerciseId")
            or option.get("target_exercise_id")
            or ""
        ).strip(): option
        for option in (main_candidate or {}).get("substitution_options", [])
        if isinstance(option, dict)
    }
    explicit_options.pop("", None)
    main_primary = _candidate_primary(main_candidate)
    sanitized: List[LLMSubstitutionCandidate] = []
    seen_ids: set[str] = set()

    for substitution in exercise.substitution_candidates:
        substitution_id = str(substitution.exercise_id or "").strip()
        candidate = safe_by_id.get(substitution_id)
        if (
            not substitution_id
            or candidate is None
            or substitution_id == exercise.exercise_id
            or substitution_id in seen_ids
        ):
            continue

        has_explicit_relation = substitution_id in explicit_options
        has_same_primary = bool(
            main_primary and _candidate_primary(candidate) == main_primary
        )
        if not (has_explicit_relation or has_same_primary):
            continue

        seen_ids.add(substitution_id)
        sanitized.append(
            LLMSubstitutionCandidate(
                exercise_id=substitution_id,
                exercise_name=_candidate_display_name(candidate),
                reason="현재 조건에서 같은 운동 목표를 유지하는 검증된 대체 후보입니다.",
            )
        )
        if len(sanitized) >= 2:
            break

    return sanitized, max(0, len(exercise.substitution_candidates) - len(sanitized))


def _guard_warning_label_for_exercise(exercise: LLMExercisePlan) -> str:
    return str(exercise.exercise_name or exercise.exercise_id or "").strip()


def _guard_unreplaced_warning(exercise: LLMExercisePlan, violation_reason: str) -> str:
    label = _guard_warning_label_for_exercise(exercise)
    if violation_reason in {"duplicate exercise_id", "duplicate display name", "duplicate movement family"}:
        return f"비슷한 운동이 겹치지 않도록 '{label}' 항목 하나를 제외했습니다."
    return f"현재 통증·장비·루틴 구성 조건에 맞지 않아 '{label}' 항목을 제외했습니다."


def trim_routine_to_time_budget(
    output: LLMRoutineOutput,
    time_available_min: int,
    *,
    protected_exercise_ids: Optional[set[str]] = None,
    execution_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    adjunct_option_by_exercise_id: Optional[Dict[str, str]] = None,
    minimum_main_exercise_count: int = 1,
) -> bool:
    """Fit the time budget without silently deleting required blueprint slots."""
    protected_ids = {
        str(exercise_id or "").strip().lower()
        for exercise_id in (protected_exercise_ids or set())
        if str(exercise_id or "").strip()
    }
    adjunct_by_id = dict(
        adjunct_option_by_exercise_id
        if adjunct_option_by_exercise_id is not None
        else (output._adjunct_option_by_exercise_id or {})
    )
    priority_by_option = {
        option: index for index, option in enumerate(ADJUNCT_OPTION_PRIORITY)
    }

    def estimate() -> int:
        return estimate_routine_time_min(
            output.exercises,
            execution_metadata=execution_metadata,
        )

    if estimate() <= time_available_min:
        return True

    adjunct_indices = sorted(
        (
            index
            for index, exercise in enumerate(output.exercises)
            if str(exercise.exercise_id or "").strip().lower() in adjunct_by_id
        ),
        key=lambda index: (
            priority_by_option.get(
                adjunct_by_id.get(
                    str(output.exercises[index].exercise_id or "").strip().lower(),
                    "",
                ),
                999,
            ),
            index,
        ),
        reverse=True,
    )
    for index in adjunct_indices:
        if estimate() <= time_available_min:
            break
        removed = output.exercises.pop(index)
        exercise_id = str(removed.exercise_id or "").strip().lower()
        option = adjunct_by_id.get(exercise_id)
        if option:
            output._adjunct_suppression_reason_by_option = {
                **dict(output._adjunct_suppression_reason_by_option or {}),
                option: "insufficientTimeAfterMainMinimum",
            }
        output.warnings = [
            *output.warnings,
            f"메인 운동과 시간 예산을 우선해 {removed.exercise_name} 보조 블록을 제외했습니다.",
        ]

    changed = True
    while changed and estimate() > time_available_min:
        changed = False
        reduction_order = sorted(
            (
                exercise
                for exercise in reversed(output.exercises)
                if is_working_prescription(exercise)
            ),
            key=lambda exercise: (
                str(exercise.exercise_id or "").strip().lower() in protected_ids,
            ),
        )
        for exercise in reduction_order:
            if exercise.sets > 1:
                exercise.sets -= 1
                changed = True
                if estimate() <= time_available_min:
                    output.warnings = [
                        *output.warnings,
                        "시간 예산에 맞추기 위해 일부 세트 수를 줄였습니다.",
                    ]
                    return True

    def main_exercise_count() -> int:
        return sum(
            str(exercise.exercise_id or "").strip().lower() not in adjunct_by_id
            for exercise in output.exercises
        )

    while (
        main_exercise_count() > max(1, int(minimum_main_exercise_count))
        and estimate() > time_available_min
    ):
        removable_index = next(
            (
                index
                for index in range(len(output.exercises) - 1, -1, -1)
                if is_working_prescription(output.exercises[index])
                if str(output.exercises[index].exercise_id or "").strip().lower()
                not in protected_ids
            ),
            None,
        )
        if removable_index is None:
            return False
        removed = output.exercises.pop(removable_index)
        output.warnings = [
            *output.warnings,
            f"시간 예산에 맞추기 위해 {removed.exercise_name} 운동을 제외했습니다.",
        ]

    return bool(output.exercises) and estimate() <= time_available_min


def _is_working_exercise(exercise: LLMExercisePlan) -> bool:
    return is_working_prescription(exercise)


def _working_set_count(exercises: List[LLMExercisePlan]) -> int:
    return sum(
        max(0, exercise.sets)
        for exercise in exercises
        if _is_working_exercise(exercise)
    )


def _has_minimum_success_quality(output: LLMRoutineOutput) -> bool:
    if not output.exercises:
        return False
    total_sets = _working_set_count(output.exercises)
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
    minimum_main_exercise_count: int = 1,
    max_repairs: Optional[int] = None,
    shoulder_condition_status: Optional[str] = None,
    restricted_movement_patterns: Optional[List[str]] = None,
    injury_conditions: Optional[List[Any]] = None,
    specialization_focus: Optional[str] = None,
    allow_reviewed_pull_pair: bool = False,
    experience_level: Optional[str] = None,
    adjunct_options: Optional[List[str]] = None,
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
    report = audit_routine_output(
        llm_output,
        candidates,
        blocked_equipment,
        pain_areas,
        shoulder_condition_status=shoulder_condition_status,
        restricted_movement_patterns=restricted_movement_patterns,
        injury_conditions=injury_conditions,
    )
    log_guard_report(report)

    requested_purposes = set(canonical_adjunct_options(adjunct_options or []))
    assigned_purposes = dict(llm_output._adjunct_option_by_exercise_id or {})

    def destination_allowed(candidate, option=None):
        if option is not None and (
            option not in requested_purposes or not candidate_matches_adjunct_option(candidate, option)
        ):
            return False
        return automatic_recommendation_eligibility(candidate, experience_level,
            adjunct_options=(option,) if option else ()).eligible

    safe_candidates = [
        c
        for c in candidates
        if _candidate_is_safe(
            c,
            blocked_equipment,
            pain_areas,
            shoulder_condition_status=shoulder_condition_status,
            restricted_movement_patterns=restricted_movement_patterns,
            injury_conditions=injury_conditions,
        )
    ]
    candidates_by_id = {str(c["id"]): c for c in candidates if c.get("id")}
    safe_by_id = {str(c["id"]): c for c in safe_candidates}
    if not safe_candidates:
        return None, "emptyCandidate"

    repaired = llm_output.model_copy(deep=True)
    has_original_safe_candidate = any(
        exercise.exercise_id in safe_by_id
        for exercise in repaired.exercises
    )
    used_ids: set[str] = set()
    used_display_keys: set[str] = set()
    used_family_keys: set[str] = set()
    violations = 0
    doms = doms_db or {}
    guard_warning_messages: List[str] = []
    repair_limit = max_repairs if max_repairs is not None else max(2, len(repaired.exercises))

    removed_indexes: set[int] = set()

    # Only the reviewed two-row output gains distinct identities; generic,
    # unsafe or third rows cannot use the composition flag as a dedup bypass.
    selected_rows = [
        safe_by_id[exercise.exercise_id]
        for exercise in repaired.exercises
        if exercise.exercise_id in safe_by_id
        and candidate_selection_pattern_key(safe_by_id[exercise.exercise_id]) == "horizontal_row"
    ]
    keep_reviewed_pair = allow_reviewed_pull_pair and complementary_pair(selected_rows)

    def selection_family(candidate: dict) -> str:
        # Keep reviewed focus variants consistent with blueprint and selection.
        family = reviewed_pull_family_key(
            candidate, candidate_selection_family_key(candidate), enabled=keep_reviewed_pair
        )
        return focus_variant_family_key(
            candidate, specialization_focus, family
        )

    for index, exercise in enumerate(repaired.exercises):
        candidate = safe_by_id.get(exercise.exercise_id)
        destination_option = assigned_purposes.get(str(exercise.exercise_id).strip().lower())
        if candidate is not None and not destination_allowed(candidate, destination_option):
            candidate = None
        display_key = _candidate_display_key(candidate) if candidate is not None else ""
        family_key = selection_family(candidate) if candidate is not None else ""
        if (
            candidate is not None
            and exercise.exercise_id not in used_ids
            and (not display_key or display_key not in used_display_keys)
            and (not family_key or family_key not in used_family_keys)
        ):
            used_ids.add(exercise.exercise_id)
            if display_key:
                used_display_keys.add(display_key)
            if family_key:
                used_family_keys.add(family_key)
            if candidate.get("movement_type"):
                exercise.movement_type = candidate["movement_type"]
            if candidate.get("primary_muscle"):
                exercise.primary_muscles = [candidate["primary_muscle"]]
            if candidate.get("equipment_req"):
                exercise.equipment_type = candidate["equipment_req"]
            display_name = _candidate_display_name(candidate)
            if display_name:
                exercise.exercise_name = display_name
        else:
            violations += 1
            original_id = exercise.exercise_id
            if candidate is not None and original_id in used_ids:
                violation_reason = "duplicate exercise_id"
            elif candidate is not None and display_key and display_key in used_display_keys:
                violation_reason = "duplicate display name"
            elif candidate is not None and family_key and family_key in used_family_keys:
                violation_reason = "duplicate movement family"
            else:
                violation_reason = "constraint violation"
            if violations > repair_limit:
                logger.warning("[Guard] repair attempt limit exceeded (%s > %s) -> fallback", violations, repair_limit)
                return None, "emptyCandidate"

            # An already assigned purpose has one owner; replacing its duplicate
            # would steal that owner's metadata and create a second purpose block.
            if destination_option is not None and original_id in used_ids:
                removed_indexes.add(index)
                guard_warning_messages.append(_guard_unreplaced_warning(exercise, "duplicate exercise_id"))
                continue

            replacement_pool = [
                c for c in safe_candidates
                if str(c["id"]) not in used_ids
                and destination_allowed(c, destination_option)
                and _candidate_display_key(c) not in used_display_keys
                and selection_family(c) not in used_family_keys
            ]
            original_candidate = candidates_by_id.get(exercise.exercise_id)
            desired_primary = _candidate_primary(candidate) or _candidate_primary(original_candidate)
            if candidate is not None and not desired_primary:
                desired_primary = _exercise_primary(exercise)
            same_primary_replacement = next(
                (
                    c for c in replacement_pool
                    if desired_primary and _candidate_primary(c) == desired_primary
                ),
                None,
            )
            replacement = same_primary_replacement
            if replacement is None and not desired_primary and violation_reason != "duplicate display name":
                replacement = next(iter(replacement_pool), None)
            if replacement is None:
                logger.warning("[Guard] safe candidate exhausted -> drop invalid exercise")
                removed_indexes.add(index)
                guard_warning_messages.append(_guard_unreplaced_warning(exercise, violation_reason))
                continue

            repaired.exercises[index] = _candidate_to_plan(replacement, exercise)
            if destination_option is not None:
                old_key, new_key = str(original_id).strip().lower(), str(replacement['id']).strip().lower()
                for attr in ('_adjunct_option_by_exercise_id', '_selection_phase_by_exercise_id',
                             '_selection_slot_by_exercise_id'):
                    mapping = getattr(repaired, attr)
                    if old_key in mapping:
                        mapping[new_key] = mapping.pop(old_key)
            used_ids.add(str(replacement["id"]))
            replacement_display_key = _candidate_display_key(replacement)
            if replacement_display_key:
                used_display_keys.add(replacement_display_key)
            replacement_family_key = selection_family(replacement)
            if replacement_family_key:
                used_family_keys.add(replacement_family_key)
            exercise = repaired.exercises[index]
            candidate = replacement
            guard_warning_messages.append(
                f"현재 조건을 반영해 '{_guard_warning_label_for_exercise(llm_output.exercises[index])}' 대신 "
                f"'{_candidate_display_name(replacement)}' 운동으로 조정했습니다."
            )

        primary = str(candidate.get("primary_muscle") or "").strip()
        doms_level = doms.get(primary, 0)
        if doms_level >= 3:
            logger.warning("[Guard] DOMS level 3 detected: %s -> fallback", primary)
            return None, "schemaError"
        if exercise.sets < 1 or exercise.target_reps < 1 or exercise.rest_time_sec < 0:
            logger.warning("[Guard] invalid set/reps/rest value: %s -> fallback", exercise.exercise_id)
            return None, "schemaError"
        # Dose is owned by deterministic prescription after candidate repair;
        # this layer must not apply a second, goal-independent soreness clamp.

    if removed_indexes:
        if not has_original_safe_candidate:
            logger.warning("[Guard] all original exercises were invalid -> fallback")
            return None, "emptyCandidate"
        repaired.exercises = [
            exercise
            for index, exercise in enumerate(repaired.exercises)
            if index not in removed_indexes
        ]
        if not _has_minimum_success_quality(repaired):
            logger.warning("[Guard] repaired routine below minimum quality after dropping invalid exercises -> fallback")
            return None, "emptyCandidate"

    removed_substitution_count = 0
    for exercise in repaired.exercises:
        purpose = repaired._adjunct_option_by_exercise_id.get(str(exercise.exercise_id).strip().lower())
        sanitized_substitutions, removed_count = _sanitize_substitution_candidates(
            exercise,
            safe_by_id.get(exercise.exercise_id),
            {eid: candidate for eid, candidate in safe_by_id.items()
             if destination_allowed(candidate, purpose)},
        )
        exercise.substitution_candidates = sanitized_substitutions
        removed_substitution_count += removed_count
    if removed_substitution_count:
        guard_warning_messages.append(
            "현재 부상·장비·동작 제한 또는 후보 관계를 검증할 수 없는 "
            f"대체 운동 {removed_substitution_count}개를 제외했습니다."
        )

    if max_total_sets is not None:
        total_sets = _working_set_count(repaired.exercises)
        overflow = total_sets - max_total_sets
        if overflow > 0:
            for exercise in reversed(repaired.exercises):
                if not _is_working_exercise(exercise):
                    continue
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
            if not trim_routine_to_time_budget(
                repaired,
                time_available_min,
                minimum_main_exercise_count=minimum_main_exercise_count,
            ):
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

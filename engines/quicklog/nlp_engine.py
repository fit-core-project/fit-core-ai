from __future__ import annotations

import os
import re
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from engines.llm_router import _parse_json_content, get_llm, resolve_llm_provider
from engines.log_redaction import sanitize_exception_for_log, summarize_text_for_log

load_dotenv()


class DietItem(BaseModel):
    food_name: str = Field(description="Food name from the user log.")
    amount: Optional[float] = Field(None, description="Food amount from the user log when present.")
    unit: Optional[str] = Field(None, description="Food amount unit from the user log when present.")
    estimated_calories: int = Field(description="Estimated calories in kcal.")
    protein_g: int = Field(description="Estimated protein in grams.")
    carbs_g: int = Field(description="Estimated carbohydrates in grams.")
    fat_g: int = Field(description="Estimated fat in grams.")


class WorkoutItem(BaseModel):
    exercise_name: str = Field(description="Exercise name from the user log.")
    weight_kg: Optional[float] = Field(None, description="Performed weight in kg when present.")
    sets: Optional[int] = Field(None, description="Set count when present.")
    reps: Optional[int] = Field(None, description="Rep count when present.")


class ParsedDailyLog(BaseModel):
    diet_logs: List[DietItem] = Field(default_factory=list)
    workout_logs: List[WorkoutItem] = Field(default_factory=list)
    overall_summary: str


_FALLBACK_SUMMARY = "기록을 저장했어요. 세부 분석은 잠시 후 다시 시도해 주세요."
_EMPTY_SUMMARY = "기록할 내용이 비어 있어요. 식단이나 운동 내용을 입력하면 저장할 수 있어요."
_TRUE_VALUES = {"true", "1", "yes", "on"}


def _quicklog_summary(workout_logs: list[WorkoutItem], diet_logs: list[DietItem]) -> str:
    parts: list[str] = []
    if workout_logs:
        item = workout_logs[0]
        detail = item.exercise_name
        if item.weight_kg is not None:
            detail += f" {item.weight_kg:g}kg"
        if item.reps is not None:
            detail += f" {item.reps}회"
        if item.sets is not None:
            detail += f" {item.sets}세트"
        parts.append(detail)
    if diet_logs:
        item = diet_logs[0]
        detail = item.food_name
        if item.amount is not None and item.unit:
            detail += f" {item.amount:g}{item.unit}"
        parts.append(detail)
    return "와 ".join(parts) + "을 기록했습니다." if parts else _FALLBACK_SUMMARY


def _estimate_diet_item(food_name: str, amount: float | None, unit: str | None) -> DietItem:
    normalized = food_name.strip()
    amount_value = amount or 1
    unit_value = unit or "serving"
    if "닭가슴살" in normalized:
        factor = amount_value / 100 if unit_value.lower() == "g" else amount_value
        return DietItem(
            food_name="닭가슴살",
            amount=amount,
            unit=unit,
            estimated_calories=round(165 * factor),
            protein_g=round(31 * factor),
            carbs_g=0,
            fat_g=round(4 * factor),
        )
    if "계란" in normalized or "달걀" in normalized:
        return DietItem(
            food_name="계란",
            amount=amount,
            unit=unit,
            estimated_calories=round(70 * amount_value),
            protein_g=round(6 * amount_value),
            carbs_g=round(amount_value),
            fat_g=round(5 * amount_value),
        )
    if normalized in {"밥", "공기밥"}:
        return DietItem(
            food_name="밥",
            amount=amount,
            unit=unit,
            estimated_calories=round(300 * amount_value),
            protein_g=round(6 * amount_value),
            carbs_g=round(65 * amount_value),
            fat_g=round(amount_value),
        )
    if "프로틴" in normalized:
        return DietItem(
            food_name="프로틴",
            amount=amount,
            unit=unit,
            estimated_calories=round(120 * amount_value),
            protein_g=round(24 * amount_value),
            carbs_g=round(3 * amount_value),
            fat_g=round(2 * amount_value),
        )
    return DietItem(
        food_name=normalized,
        amount=amount,
        unit=unit,
        estimated_calories=0,
        protein_g=0,
        carbs_g=0,
        fat_g=0,
    )


def _deterministic_parse(user_text: str | None, *, summary: str | None = None) -> ParsedDailyLog:
    text = (user_text or "").strip()
    if not text:
        return ParsedDailyLog(diet_logs=[], workout_logs=[], overall_summary=_EMPTY_SUMMARY)

    workout_logs: list[WorkoutItem] = []
    workout_pattern = re.compile(
        r"(?P<exercise>[가-힣A-Za-z][가-힣A-Za-z0-9\s]{0,30}?)\s*"
        r"(?:(?P<weight>\d+(?:\.\d+)?)\s*kg\s*)?"
        r"(?P<reps>\d+)\s*(?:회|rep|reps)\s*"
        r"(?P<sets>\d+)\s*(?:세트|set|sets)",
        flags=re.IGNORECASE,
    )
    for match in workout_pattern.finditer(text):
        exercise_name = re.sub(r"^(오늘|그리고|하고|또)\s+", "", match.group("exercise").strip())
        workout_logs.append(
            WorkoutItem(
                exercise_name=exercise_name,
                weight_kg=float(match.group("weight")) if match.group("weight") else None,
                reps=int(match.group("reps")),
                sets=int(match.group("sets")),
            )
        )
    if not workout_logs:
        weight_match = re.search(r"(?P<weight>\d+(?:\.\d+)?)\s*kg", text, flags=re.IGNORECASE)
        reps_match = re.search(r"(?P<reps>\d+)\s*(?:회|rep|reps)", text, flags=re.IGNORECASE)
        sets_match = re.search(r"(?P<sets>\d+)\s*(?:세트|set|sets)", text, flags=re.IGNORECASE)
        if weight_match or reps_match or sets_match:
            exercise_name = text[: weight_match.start()].strip() if weight_match else text.split()[0]
            workout_logs.append(
                WorkoutItem(
                    exercise_name=exercise_name or "unknown",
                    weight_kg=float(weight_match.group("weight")) if weight_match else None,
                    reps=int(reps_match.group("reps")) if reps_match else None,
                    sets=int(sets_match.group("sets")) if sets_match else None,
                )
            )

    diet_logs: list[DietItem] = []
    diet_pattern = re.compile(
        r"(?P<food>닭가슴살|밥|공기밥|계란|달걀|프로틴)\s*"
        r"(?P<amount>\d+(?:\.\d+)?)\s*"
        r"(?P<unit>g|그램|공기|개|스쿱)",
        flags=re.IGNORECASE,
    )
    for match in diet_pattern.finditer(text):
        unit = match.group("unit")
        normalized_unit = "g" if unit == "그램" else unit
        diet_logs.append(
            _estimate_diet_item(
                match.group("food"),
                float(match.group("amount")),
                normalized_unit,
            )
        )

    return ParsedDailyLog(
        diet_logs=diet_logs,
        workout_logs=workout_logs,
        overall_summary=summary or _quicklog_summary(workout_logs, diet_logs),
    )


def _fallback_log(user_text: str | None, *, summary: str = _FALLBACK_SUMMARY) -> ParsedDailyLog:
    return _deterministic_parse(user_text, summary=summary)


def _build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a provider-neutral fitness and nutrition log parser.
Return JSON only. Do not use markdown fences. Do not add explanations.
Extract diet and workout records from the user's natural-language text.
Preserve numbers and units from the input.
Use this exact JSON shape:
{{"diet_logs":[{{"food_name":"string","amount":200,"unit":"g","estimated_calories":0,"protein_g":0,"carbs_g":0,"fat_g":0}}],"workout_logs":[{{"exercise_name":"string","weight_kg":60.0,"sets":3,"reps":10}}],"overall_summary":"string"}}
When nutrition macros are uncertain, use 0 rather than inventing precise values.
When a workout field is unknown, use null.""",
            ),
            ("human", "Daily log text: {text}"),
        ]
    )


def _local_raw_json_enabled() -> bool:
    return os.getenv("ENABLE_LOCAL_RAW_JSON_INVOKE", "").strip().lower() in _TRUE_VALUES


def _parse_llm_payload(payload) -> ParsedDailyLog:
    if isinstance(payload, ParsedDailyLog):
        return payload
    if isinstance(payload, dict):
        return ParsedDailyLog.model_validate(payload)
    if isinstance(payload, str):
        return ParsedDailyLog.model_validate(_parse_json_content(payload))
    content = getattr(payload, "content", None)
    if isinstance(content, str):
        return ParsedDailyLog.model_validate(_parse_json_content(content))
    return ParsedDailyLog.model_validate(payload)


def parse_natural_language_log(user_text: str) -> str:
    """Parse a natural-language quicklog into a JSON string.

    This function always returns a ParsedDailyLog-compatible JSON string and
    never logs raw user input or raw LLM output.
    """
    if not (user_text or "").strip():
        return _fallback_log(user_text).model_dump_json()

    try:
        llm = get_llm("nlp", temperature=0.1)
        print("[LLM quicklog parsing start]", summarize_text_for_log(user_text))
        if resolve_llm_provider().effective_provider == "local" and _local_raw_json_enabled():
            parsed_result = _parse_llm_payload((_build_prompt() | llm).invoke({"text": user_text}))
        else:
            structured_llm = llm.with_structured_output(ParsedDailyLog)
            if hasattr(structured_llm, "__ror__"):
                parsed_result = _parse_llm_payload((_build_prompt() | structured_llm).invoke({"text": user_text}))
            else:
                parsed_result = _parse_llm_payload(structured_llm.invoke({"text": user_text}))
        return parsed_result.model_dump_json()
    except Exception as exc:
        print("[LLM quicklog parsing fallback]", sanitize_exception_for_log(exc))
        return _deterministic_parse(user_text).model_dump_json()

"""규칙 기반 Fallback 루틴 생성 + 루틴 빌드·디버그 유틸리티."""
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate

from engines.llm_router import StatusReasonCode
from .schemas import (
    GenerationStatus,
    LLMExercisePlan,
    LLMRoutineOutput,
    RoutineDraftResponse,
    RoutineBlock,
    RoutineRequest,
    RecentSetRecord,
    SetPrescription,
    SubstitutionCandidate,
    UserProfileContext,
)
from .registry import LARGE_MUSCLE_SLUGS, ACCESSORY_MUSCLE_SLUGS
from .prescription.estimator import estimate_routine_time_min, _movement_type_key
from .prescription.params import _prescription_params_for_exercise
from .prescription.weight import _resolve_target_weight
from .prescription.adjustments import _apply_readiness_to_exercise, _apply_large_muscle_volume_guard
from .candidate_ranker import _is_loadable_equipment
from .muscle_mapping import get_mapped_targets, split_label_to_muscles
from .llm_parser import trim_routine_to_time_budget

# ==========================================
# 9. Fallback 루틴 (규칙 기반)  →  fallback.py 로 이동됨
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


def _primary_muscle_priority(primary: Optional[str]) -> int:
    if primary in LARGE_MUSCLE_SLUGS:
        return 0
    if primary in ACCESSORY_MUSCLE_SLUGS:
        return 1
    return 2


def _planned_exercise_order_key(item: Tuple[int, LLMExercisePlan]) -> Tuple[int, int, float, int]:
    original_order, exercise = item
    primary = exercise.primary_muscles[0] if exercise.primary_muscles else None
    weight = exercise.target_weight_kg or 0
    return (
        _MOVEMENT_PRIORITY.get(_movement_type_key(exercise.movement_type), 3),
        _primary_muscle_priority(primary),
        -weight,
        original_order,
    )


def _candidate_order_key(candidate: dict) -> Tuple[int, int, int, int, str]:
    return (
        0 if _is_loadable_equipment(candidate.get("equipment_req")) else 1,
        _MOVEMENT_PRIORITY.get(_movement_type_key(candidate.get("movement_type")), 3),
        _primary_muscle_priority(candidate.get("primary_muscle")),
        int(candidate.get("efficiency_tier") or 99),
        str(candidate.get("id") or ""),
    )


def _order_exercises_for_training(exercises: List[LLMExercisePlan]) -> List[LLMExercisePlan]:
    return [
        exercise
        for _, exercise in sorted(
            enumerate(exercises),
            key=_planned_exercise_order_key,
        )
    ]

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


def build_routine_draft(
    llm_output: LLMRoutineOutput,
    generation_status: GenerationStatus,
    status_reason_code: StatusReasonCode,
    is_fallback: bool,
) -> RoutineDraftResponse:
    blocks = []
    ordered_exercises = _order_exercises_for_training(llm_output.exercises)
    for order, ex in enumerate(ordered_exercises, start=1):
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
        total_estimated_time=estimate_routine_time_min(ordered_exercises),
        summary_title=llm_output.summary_title,
        rationale_summary=llm_output.rationale_summary,
        routine_blocks=blocks,
        warnings=llm_output.warnings,
    )


def _debug_print_prompt(prompt: ChatPromptTemplate, invoke_kwargs: Dict[str, Any]) -> None:
    try:
        rendered = prompt.format(**invoke_kwargs)
    except Exception as exc:
        print(f"[AI DEBUG][PROMPT] render failed: {exc}")
        print(f"[AI DEBUG][PROMPT VARS] {invoke_kwargs}")
        return

    print("\n========== [AI DEBUG] RENDERED ROUTINE PROMPT ==========")
    print(rendered)
    print("========== [AI DEBUG] END ROUTINE PROMPT ==========\n")


def _debug_print_llm_output(label: str, output: LLMRoutineOutput) -> None:
    print(f"\n========== [AI DEBUG] {label} ==========")
    print(
        f"summary={output.summary_title} | llm_total_estimated_time={output.total_estimated_time} | "
        f"server_estimated_time={estimate_routine_time_min(output.exercises)}"
    )
    for index, exercise in enumerate(output.exercises, start=1):
        print(
            f"{index}. id={exercise.exercise_id} name={exercise.exercise_name} "
            f"type={exercise.movement_type} primary={exercise.primary_muscles} "
            f"sets={exercise.sets} reps={exercise.target_reps} "
            f"weight={exercise.target_weight_kg}kg rir={exercise.target_rir} "
            f"rest={exercise.rest_time_sec}s rationale={exercise.exercise_rationale}"
        )
    print(f"========== [AI DEBUG] END {label} ==========\n")


def _debug_print_draft(label: str, draft: RoutineDraftResponse) -> None:
    print(f"\n========== [AI DEBUG] {label} ==========")
    print(
        f"status={draft.generation_status} reason={draft.status_reason_code} "
        f"is_fallback={draft.is_fallback} total_estimated_time={draft.total_estimated_time}"
    )
    for block in draft.routine_blocks:
        first_set = block.prescription[0] if block.prescription else None
        print(
            f"{block.order}. id={block.exercise_id} name={block.exercise_name} "
            f"primary={block.primary_muscles} sets={len(block.prescription)} "
            f"reps={first_set.target_reps if first_set else None} "
            f"weight={first_set.target_weight_kg if first_set else None}kg "
            f"rir={first_set.target_rir if first_set else None} "
            f"rest={first_set.target_rest_sec if first_set else None}s"
        )
    print(f"========== [AI DEBUG] END {label} ==========\n")


def generate_fallback_routine(
    req: RoutineRequest,
    candidates: List[dict],
    max_total_sets: int,
    doms_db: Optional[Dict[str, int]] = None,
    goal: str = "hypertrophy",
    status_reason_code: StatusReasonCode = "networkError",
    recent_sets: Optional[List[RecentSetRecord]] = None,
    profile: Optional[UserProfileContext] = None,
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
        key=_candidate_order_key,
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
        target_reps, rest_sec = _prescription_params_for_exercise(
            LLMExercisePlan(
                exercise_id=str(ex.get("id", "")),
                exercise_name=str(ex.get("name_kr", "")),
                movement_type=ex.get("movement_type"),
                primary_muscles=[ex["primary_muscle"]] if ex.get("primary_muscle") else [],
                target_reps=params["reps"],
                sets=sets,
                rest_time_sec=params["rest_sec"],
                exercise_rationale="",
            ),
            goal,
        )

        plan = LLMExercisePlan(
            exercise_id=str(ex.get("id", ex["name_kr"].replace(" ", "_").lower())),
            exercise_name=ex["name_kr"],
            movement_pattern=ex.get("movement_pattern"),
            movement_type=ex.get("movement_type"),
            primary_muscles=[ex["primary_muscle"]] if ex.get("primary_muscle") else [],
            equipment_type=ex.get("equipment_req"),
            target_weight_kg=None,
            target_reps=target_reps,
            sets=sets,
            rest_time_sec=rest_sec,
            target_rir=2,
            exercise_rationale=_FALLBACK_TIPS.get(ex["movement_type"], "정확한 자세로 수행하세요."),
        )
        _apply_readiness_to_exercise(plan, req.readiness_level)
        _apply_large_muscle_volume_guard(plan, req.target_split_label, fallback_target_muscles)
        plan.target_weight_kg = _resolve_target_weight(plan, goal, recent_sets, profile)
        sets = plan.sets
        remaining_sets -= sets

        blocks.append(RoutineBlock(
            order=order,
            exercise_id=plan.exercise_id,
            exercise_name=plan.exercise_name,
            movement_pattern=plan.movement_pattern,
            primary_muscles=plan.primary_muscles,
            equipment_type=plan.equipment_type,
            default_rest_sec=plan.rest_time_sec,
            prescription=_build_prescription(
                sets=plan.sets,
                reps=plan.target_reps,
                weight_kg=plan.target_weight_kg,
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

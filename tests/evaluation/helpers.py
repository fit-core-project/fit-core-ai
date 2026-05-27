from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from engines.candidate_ranker import score_candidate_exercises
from engines.prescription.adjustments import _calculate_max_total_sets
from engines.routine_pipeline import _validate_or_fallback
from engines.rule_critic import (
    RoutineEvalContext,
    _check_hard_constraints_sanity,
    build_eval_context,
    evaluate_routine_quality,
)
from engines.schemas import (
    LLMRoutineOutput,
    PainAreaEntry,
    RecentSetRecord,
    RoutineDraftResponse,
    RoutineRequest,
)


@dataclass
class ScenarioResult:
    scenario_id: str
    passed: bool
    total_score: int
    hard_violations: int
    hard_violation_details: list[str]
    metric_scores: dict[str, int]
    warnings: list[str]
    fallback_used: bool
    repair_used: bool
    routine_blocks: list[dict[str, Any]]
    total_estimated_time: int
    requested_duration: int
    notes: list[str]


def check_hard_constraints(
    blocks: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    scenario: dict[str, Any],
    max_working_sets: int,
) -> list[str]:
    """Backward-compatible wrapper used by offline harness tests."""
    pain_areas = [PainAreaEntry(**item) for item in scenario.get("current_pain_areas", [])]
    context = RoutineEvalContext(
        goal=scenario.get("goal", "hypertrophy"),
        target_muscles=scenario.get("target_muscles", []),
        duration_min=scenario.get("duration_min", 60),
        readiness_level=scenario.get("readiness_level", "normal"),
        available_equipment=scenario.get("available_equipment", []),
        unavailable_equipment=scenario.get("unavailable_equipment", []),
        current_pain_areas=pain_areas,
        doms=scenario.get("doms", {}),
        recent_exercises=scenario.get("recent_exercises", []),
        max_working_sets=max_working_sets,
        overrides=scenario.get("expected", {}),
    )
    candidates_by_id = {str(candidate["id"]): candidate for candidate in candidates}
    return _check_hard_constraints_sanity(blocks, candidates_by_id, context, max_working_sets)


class EvalHarness:
    def run(self, scenario: dict[str, Any]) -> ScenarioResult:
        expected = scenario.get("expected", {})
        pain_areas = [PainAreaEntry(**item) for item in scenario.get("current_pain_areas", [])]
        recent_sets = [
            RecentSetRecord(
                exercise_id=exercise_id,
                exercise_name=exercise_id,
                reps=8,
            )
            for exercise_id in scenario.get("recent_exercises", [])
        ]
        req = RoutineRequest(
            user_id=f"eval-{scenario['id']}",
            target_split_label=scenario.get("target_split_label"),
            target_muscles=scenario.get("target_muscles", []),
            readiness_level=scenario.get("readiness_level", "normal"),
            time_available_min=scenario["duration_min"],
            pain_areas=pain_areas,
            doms_data=scenario.get("doms", {}),
            equipment=scenario.get("unavailable_equipment", []),
            goal=scenario.get("goal"),
            user_note=scenario.get("name"),
        )
        ranked_candidates = score_candidate_exercises(
            scenario["candidates"],
            scenario.get("target_muscles", []),
            doms_db=scenario.get("doms", {}),
            blocked_equipment=scenario.get("unavailable_equipment", []),
            pain_areas=pain_areas,
            recent_sets=recent_sets,
        )
        output = LLMRoutineOutput.model_validate(scenario["mock_llm_output"])
        default_max_sets = _calculate_max_total_sets(scenario["duration_min"], scenario["goal"])
        max_working_sets = int(expected.get("max_working_sets") or default_max_sets)

        response: RoutineDraftResponse = _validate_or_fallback(
            output=output,
            req=req,
            ranked_candidates=ranked_candidates,
            max_total_sets=max_working_sets,
            doms_db=scenario.get("doms", {}),
            goal=scenario["goal"],
            db_target_muscles=scenario.get("target_muscles", []),
            recent_sets=recent_sets,
            profile=None,
        )

        context = build_eval_context(
            req,
            scenario.get("target_muscles", []),
            recent_sets=recent_sets,
            max_working_sets=max_working_sets,
        )
        context.available_equipment = scenario.get("available_equipment", [])
        context.overrides = dict(expected)
        repair_count = len([
            warning
            for warning in response.warnings
            if "Guard:" in warning and "replaced" in warning
        ])
        fallback_used = response.is_fallback or response.generation_status == "fallback"
        critic = evaluate_routine_quality(
            response,
            ranked_candidates,
            context,
            repair_count=repair_count,
            fallback_used=fallback_used,
        )
        blocks = [block.model_dump(by_alias=False) for block in response.routine_blocks]
        fail_warnings = [warning for warning in critic.warnings if warning.startswith("FAIL ")]
        passed = (
            not critic.hard_violations
            and critic.score >= 80
            and not fail_warnings
            and fallback_used == bool(expected.get("is_fallback", False))
            and (repair_count > 0) == bool(expected.get("expected_repair", False))
        )
        if expected.get("must_pass", True) is False:
            passed = bool(expected.get("expected_violation")) and (
                bool(critic.hard_violations) or bool(fail_warnings)
            )

        return ScenarioResult(
            scenario_id=scenario["id"],
            passed=passed,
            total_score=critic.score,
            hard_violations=len(critic.hard_violations),
            hard_violation_details=critic.hard_violations,
            metric_scores=critic.metric_scores,
            warnings=list(response.warnings) + critic.warnings,
            fallback_used=fallback_used,
            repair_used=repair_count > 0,
            routine_blocks=blocks,
            total_estimated_time=response.total_estimated_time,
            requested_duration=scenario["duration_min"],
            notes=critic.notes,
        )


def save_report(results: list[ScenarioResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "total_scenarios": len(results),
        "passed": sum(1 for result in results if result.passed),
        "failed": sum(1 for result in results if not result.passed),
        "scenarios": [asdict(result) for result in results],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

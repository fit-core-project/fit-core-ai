from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_gemma4_quality_seed as seed_runner
from scripts.routine_eval_provenance import build_provenance


DEFAULT_SEED_PATH = Path("tests/fixtures/gemma4_routine_quality_eval_seed_v2.jsonl")
DEFAULT_OUTPUT_DIR = Path("tests/evaluation/.artifacts/gemma4-quality-stability")


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("gemma4-stability-%Y%m%dT%H%M%SZ")


def sanitize_run_id(value: str | None) -> str:
    raw = value or utc_run_id()
    sanitized = re.sub(r"[^A-Za-z0-9_.:-]", "-", raw).strip("-")
    return sanitized or utc_run_id()


def selection_signature(result: dict[str, Any]) -> str:
    selected = [str(item) for item in result.get("selectedExerciseIds", [])]
    return "|".join(selected)


def canonical_generation_status(value: str) -> str:
    normalized = str(value or "").strip()
    lowered = normalized.lower()
    if lowered in {"success", "fallback", "failed", "error"}:
        return lowered
    return normalized or "unknown"


def canonical_status_reason_code(value: str) -> str:
    normalized = str(value or "").strip()
    lowered = normalized.lower()
    known = {
        "none": "none",
        "llmtimeout": "llmTimeout",
        "schemaerror": "schemaError",
        "networkerror": "networkError",
        "emptycandidate": "emptyCandidate",
        "liverunnererror": "liveRunnerError",
    }
    return known.get(lowered, normalized or "unknown")


def result_field(result: dict[str, Any], field: str, note_key: str, default: str = "unknown") -> str:
    value = result.get(field)
    if value is not None:
        return str(value)
    note_fields = seed_runner.parse_note_fields(result.get("notes"))
    return str(note_fields.get(note_key, default))


def result_is_fallback(result: dict[str, Any]) -> bool | None:
    if "isFallback" in result:
        return seed_runner._parse_bool(result.get("isFallback"))
    note_fields = seed_runner.parse_note_fields(result.get("notes"))
    return seed_runner._parse_bool(note_fields.get("isfallback"))


def scenario_action_items(
    item: dict[str, Any],
    *,
    require_exact_selection_stability: bool = True,
) -> list[str]:
    actions: list[str] = []
    if item["missingRunIndexes"]:
        actions.append("rerun missing result files before judging stability")
    if item["passCount"] < item["scoredRunCount"]:
        actions.append("inspect hard safety, latency, or fallback gate failures")
    if item["qualityPassCount"] < item["scoredRunCount"]:
        actions.append("review preferred pattern gaps and candidate-to-LLM handoff")
    if item["outputQualityPassCount"] < item["scoredRunCount"]:
        actions.append("inspect time utilization, exercise count, and movement-family diversity")
    if item["languageQualityPassCount"] < item["scoredRunCount"]:
        actions.append("inspect user-facing Korean wording and unsupported coaching claims")
    if item["latencyPassCount"] < item["scoredRunCount"]:
        actions.append("reduce prompt/candidate pressure or raise timeout deliberately")
    if item["fallbackPassCount"] < item["scoredRunCount"]:
        actions.append("decide whether fallback is acceptable for this scenario")
    if item.get("fallbackRunCount", 0):
        actions.append("inspect fallback output quality and fallback trigger reason")
    if item.get("timeoutRunCount", 0):
        actions.append("review local LLM latency and compact prompt settings")
    if item["uniqueSelectionCount"] > 1 and require_exact_selection_stability:
        actions.append("compare selected exercise signatures across runs")
    return actions


def scenario_informational_notes(
    item: dict[str, Any],
    *,
    require_exact_selection_stability: bool = True,
) -> list[str]:
    if item["uniqueSelectionCount"] > 1 and not require_exact_selection_stability:
        return ["valid exercise selection varied while every quality gate passed"]
    return []


def parse_scenario_ids(values: list[str] | None) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        for raw_item in str(value).split(","):
            scenario_id = raw_item.strip()
            if not scenario_id or scenario_id in seen:
                continue
            ids.append(scenario_id)
            seen.add(scenario_id)
    return ids


def filter_seed_rows(rows: list[dict[str, Any]], scenario_ids: list[str] | None = None) -> list[dict[str, Any]]:
    if not scenario_ids:
        return rows
    requested = set(scenario_ids)
    found = {str(row["scenarioId"]) for row in rows if str(row["scenarioId"]) in requested}
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"scenarioId not found in seed: {', '.join(missing)}")
    return [row for row in rows if str(row["scenarioId"]) in requested]


def build_stability_report(
    *,
    seed_path: Path,
    results_paths: list[Path],
    output_dir: Path,
    run_id: str,
    min_preferred_hits: int = seed_runner.DEFAULT_MIN_PREFERRED_HITS,
    max_elapsed_ms: int | None = None,
    disallow_fallback: bool = False,
    require_preferred_quality: bool = False,
    require_exact_selection_stability: bool = True,
    scenario_ids: list[str] | None = None,
) -> dict[str, Any]:
    rows = filter_seed_rows(seed_runner.load_seed(seed_path), scenario_ids)
    result_maps = [seed_runner.load_results(path) for path in results_paths]
    scenarios: list[dict[str, Any]] = []
    report_generation_counts: Counter[str] = Counter()
    report_reason_counts: Counter[str] = Counter()
    total_fallback_runs = 0
    total_non_success_runs = 0
    total_timeout_runs = 0
    for row in rows:
        scenario_id = str(row["scenarioId"])
        scored_runs: list[dict[str, Any]] = []
        missing_run_indexes: list[int] = []
        signatures: list[str] = []
        elapsed_values: list[int] = []
        generation_counts: Counter[str] = Counter()
        reason_counts: Counter[str] = Counter()
        fallback_run_count = 0
        non_success_run_count = 0
        timeout_run_count = 0
        for index, result_map in enumerate(result_maps, 1):
            result = result_map.get(scenario_id)
            if not result:
                missing_run_indexes.append(index)
                continue
            generation_status = canonical_generation_status(
                result_field(result, "generationStatus", "generationstatus")
            )
            status_reason_code = canonical_status_reason_code(
                result_field(result, "statusReasonCode", "statusreasoncode")
            )
            is_fallback = result_is_fallback(result)
            generation_counts[generation_status] += 1
            reason_counts[status_reason_code] += 1
            report_generation_counts[generation_status] += 1
            report_reason_counts[status_reason_code] += 1
            if is_fallback is True:
                fallback_run_count += 1
                total_fallback_runs += 1
            if generation_status != "success":
                non_success_run_count += 1
                total_non_success_runs += 1
            if status_reason_code == "llmTimeout":
                timeout_run_count += 1
                total_timeout_runs += 1
            scored = seed_runner.score_result(
                row,
                result,
                min_preferred_hits=min_preferred_hits,
                max_elapsed_ms=max_elapsed_ms,
                disallow_fallback=disallow_fallback,
                require_preferred_quality=require_preferred_quality,
            )
            scored_runs.append(scored)
            signatures.append(selection_signature(result))
            if scored.get("elapsedMs") is not None:
                elapsed_values.append(int(scored["elapsedMs"]))
        pass_count = sum(1 for item in scored_runs if item["passed"])
        quality_pass_count = sum(1 for item in scored_runs if item["preferredQualityPassed"])
        output_quality_pass_count = sum(1 for item in scored_runs if item["outputQualityPassed"])
        language_quality_pass_count = sum(1 for item in scored_runs if item["languageQualityPassed"])
        latency_pass_count = sum(1 for item in scored_runs if item["latencyPassed"])
        fallback_pass_count = sum(1 for item in scored_runs if item["fallbackPassed"])
        complete_count = len(scored_runs)
        unique_selection_count = len(set(signatures))
        all_runs_passed = complete_count == len(results_paths) and pass_count == complete_count
        exact_selection_stable = unique_selection_count <= 1
        stability_gate_passed = all_runs_passed and (
            exact_selection_stable or not require_exact_selection_stability
        )
        scenario_report = {
            "scenarioId": scenario_id,
            "title": row["title"],
            "runCount": len(results_paths),
            "scoredRunCount": complete_count,
            "missingRunIndexes": missing_run_indexes,
            "passCount": pass_count,
            "qualityPassCount": quality_pass_count,
            "outputQualityPassCount": output_quality_pass_count,
            "languageQualityPassCount": language_quality_pass_count,
            "latencyPassCount": latency_pass_count,
            "fallbackPassCount": fallback_pass_count,
            "fallbackRunCount": fallback_run_count,
            "nonSuccessRunCount": non_success_run_count,
            "timeoutRunCount": timeout_run_count,
            "generationStatusCounts": dict(sorted(generation_counts.items())),
            "statusReasonCodeCounts": dict(sorted(reason_counts.items())),
            "uniqueSelectionCount": unique_selection_count,
            "selectionStable": unique_selection_count <= 1,
            "exactSelectionStable": exact_selection_stable,
            "allRunsPassed": all_runs_passed,
            "allRunsStable": stability_gate_passed,
            "stabilityGatePassed": stability_gate_passed,
            "elapsedMsMin": min(elapsed_values) if elapsed_values else None,
            "elapsedMsMax": max(elapsed_values) if elapsed_values else None,
        }
        scenario_report["actionItems"] = scenario_action_items(
            scenario_report,
            require_exact_selection_stability=require_exact_selection_stability,
        )
        scenario_report["informationalNotes"] = scenario_informational_notes(
            scenario_report,
            require_exact_selection_stability=require_exact_selection_stability,
        )
        scenarios.append(scenario_report)
    stable_passed = sum(1 for item in scenarios if item["allRunsStable"])
    unstable_or_failed = len(scenarios) - stable_passed
    scenario_action_count = sum(len(item["actionItems"]) for item in scenarios)
    if unstable_or_failed:
        recommendation = "needs_review"
    elif total_fallback_runs or total_timeout_runs:
        recommendation = "pass_with_runtime_observation"
    elif scenario_action_count:
        recommendation = "pass_with_notes"
    else:
        recommendation = "pass"
    return {
        "runId": run_id,
        "mode": "quality-stability",
        "seedPath": str(seed_path),
        "outputDir": str(output_dir),
        "resultsPaths": [str(path) for path in results_paths],
        "runCount": len(results_paths),
        "scenarioCount": len(scenarios),
        "stablePassed": stable_passed,
        "unstableOrFailed": unstable_or_failed,
        "minPreferredHits": min_preferred_hits,
        "maxElapsedMs": max_elapsed_ms,
        "disallowFallback": disallow_fallback,
        "requirePreferredQuality": require_preferred_quality,
        "requireExactSelectionStability": require_exact_selection_stability,
        "selectedScenarioIds": scenario_ids or [],
        "generationStatusCounts": dict(sorted(report_generation_counts.items())),
        "statusReasonCodeCounts": dict(sorted(report_reason_counts.items())),
        "totalFallbackRuns": total_fallback_runs,
        "totalNonSuccessRuns": total_non_success_runs,
        "totalTimeoutRuns": total_timeout_runs,
        "scenarioActionCount": scenario_action_count,
        "recommendation": recommendation,
        "provenance": build_provenance(
            runner="scripts/summarize_gemma4_quality_stability.py",
            input_paths=[seed_path, *results_paths],
        ),
        "scenarios": scenarios,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Gemma4 Quality Stability Report",
        "",
        f"- run_id: `{report['runId']}`",
        f"- seed: `{report['seedPath']}`",
        f"- run_count: `{report['runCount']}`",
        f"- stable_passed: `{report['stablePassed']}` / `{report['scenarioCount']}`",
        f"- unstable_or_failed: `{report['unstableOrFailed']}`",
        f"- recommendation: `{report['recommendation']}`",
        f"- generation_status_counts: `{report.get('generationStatusCounts', {})}`",
        f"- status_reason_code_counts: `{report.get('statusReasonCodeCounts', {})}`",
        f"- fallback_runs: `{report.get('totalFallbackRuns', 0)}`",
        f"- timeout_runs: `{report.get('totalTimeoutRuns', 0)}`",
        f"- scenario_action_count: `{report.get('scenarioActionCount', 0)}`",
        "",
        "## Scenarios",
        "",
        "| Scenario | Pass | Preferred Quality | Output Quality | Language Quality | Latency | Fallback Gate | Fallback Runs | Timeout Runs | Status | Reason | Unique Selection | Elapsed Range | Missing Runs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|---|",
    ]
    for item in report["scenarios"]:
        missing = ", ".join(str(index) for index in item["missingRunIndexes"]) or "-"
        elapsed_range = (
            f"{item['elapsedMsMin']}..{item['elapsedMsMax']}"
            if item["elapsedMsMin"] is not None
            else "-"
        )
        generation_status = ", ".join(
            f"{key}:{value}" for key, value in item.get("generationStatusCounts", {}).items()
        ) or "-"
        status_reason = ", ".join(
            f"{key}:{value}" for key, value in item.get("statusReasonCodeCounts", {}).items()
        ) or "-"
        lines.append(
            f"| `{item['scenarioId']}` | `{item['passCount']}/{item['runCount']}` | "
            f"`{item['qualityPassCount']}/{item['runCount']}` | "
            f"`{item['outputQualityPassCount']}/{item['runCount']}` | "
            f"`{item['languageQualityPassCount']}/{item['runCount']}` | "
            f"`{item['latencyPassCount']}/{item['runCount']}` | "
            f"`{item['fallbackPassCount']}/{item['runCount']}` | "
            f"`{item.get('fallbackRunCount', 0)}` | "
            f"`{item.get('timeoutRunCount', 0)}` | "
            f"{generation_status} | {status_reason} | "
            f"`{item['uniqueSelectionCount']}` | {elapsed_range} | {missing} |"
        )
        if item.get("actionItems"):
            for action in item["actionItems"]:
                lines.append(f"  - action: {action}")
        if item.get("informationalNotes"):
            for note in item["informationalNotes"]:
                lines.append(f"  - note: {note}")
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    seed_runner.assert_no_forbidden_tokens(report)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "gemma4-quality-stability-report.json"
    md_path = output_dir / "gemma4-quality-stability-report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize repeated Gemma4 quality seed runs.")
    parser.add_argument("--seed", default=str(DEFAULT_SEED_PATH))
    parser.add_argument("--results", action="append", required=True, help="Result JSON. Repeat for each run.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-id")
    parser.add_argument("--min-preferred-hits", type=int, default=seed_runner.DEFAULT_MIN_PREFERRED_HITS)
    parser.add_argument("--max-elapsed-ms", type=int)
    parser.add_argument("--disallow-fallback", action="store_true")
    parser.add_argument("--require-preferred-quality", action="store_true")
    parser.add_argument(
        "--allow-valid-selection-variation",
        action="store_true",
        help="Treat exact exercise-id variation as informational when every quality gate passes.",
    )
    parser.add_argument("--scenario-id", action="append", help="Summarize only selected scenario ids. Repeat or comma-separate.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_id = sanitize_run_id(args.run_id)
    output_dir = Path(args.output_dir) / run_id
    report = build_stability_report(
        seed_path=Path(args.seed),
        results_paths=[Path(path) for path in args.results],
        output_dir=output_dir,
        run_id=run_id,
        min_preferred_hits=args.min_preferred_hits,
        max_elapsed_ms=args.max_elapsed_ms,
        disallow_fallback=args.disallow_fallback,
        require_preferred_quality=args.require_preferred_quality,
        require_exact_selection_stability=not args.allow_valid_selection_variation,
        scenario_ids=parse_scenario_ids(args.scenario_id),
    )
    paths = write_report(report, output_dir)
    print(json.dumps({"report": report, "paths": paths}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

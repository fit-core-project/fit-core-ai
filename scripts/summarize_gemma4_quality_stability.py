from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_gemma4_quality_seed as seed_runner


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
    scenario_ids: list[str] | None = None,
) -> dict[str, Any]:
    rows = filter_seed_rows(seed_runner.load_seed(seed_path), scenario_ids)
    result_maps = [seed_runner.load_results(path) for path in results_paths]
    scenarios: list[dict[str, Any]] = []
    for row in rows:
        scenario_id = str(row["scenarioId"])
        scored_runs: list[dict[str, Any]] = []
        missing_run_indexes: list[int] = []
        signatures: list[str] = []
        elapsed_values: list[int] = []
        for index, result_map in enumerate(result_maps, 1):
            result = result_map.get(scenario_id)
            if not result:
                missing_run_indexes.append(index)
                continue
            scored = seed_runner.score_result(
                row,
                result,
                min_preferred_hits=min_preferred_hits,
                max_elapsed_ms=max_elapsed_ms,
                disallow_fallback=disallow_fallback,
            )
            scored_runs.append(scored)
            signatures.append(selection_signature(result))
            if scored.get("elapsedMs") is not None:
                elapsed_values.append(int(scored["elapsedMs"]))
        pass_count = sum(1 for item in scored_runs if item["passed"])
        quality_pass_count = sum(1 for item in scored_runs if item["preferredQualityPassed"])
        latency_pass_count = sum(1 for item in scored_runs if item["latencyPassed"])
        fallback_pass_count = sum(1 for item in scored_runs if item["fallbackPassed"])
        complete_count = len(scored_runs)
        unique_selection_count = len(set(signatures))
        all_runs_passed = complete_count == len(results_paths) and pass_count == complete_count
        all_runs_stable = all_runs_passed and unique_selection_count <= 1
        scenarios.append({
            "scenarioId": scenario_id,
            "title": row["title"],
            "runCount": len(results_paths),
            "scoredRunCount": complete_count,
            "missingRunIndexes": missing_run_indexes,
            "passCount": pass_count,
            "qualityPassCount": quality_pass_count,
            "latencyPassCount": latency_pass_count,
            "fallbackPassCount": fallback_pass_count,
            "uniqueSelectionCount": unique_selection_count,
            "selectionStable": unique_selection_count <= 1,
            "allRunsPassed": all_runs_passed,
            "allRunsStable": all_runs_stable,
            "elapsedMsMin": min(elapsed_values) if elapsed_values else None,
            "elapsedMsMax": max(elapsed_values) if elapsed_values else None,
        })
    stable_passed = sum(1 for item in scenarios if item["allRunsStable"])
    unstable_or_failed = len(scenarios) - stable_passed
    recommendation = "pass" if unstable_or_failed == 0 else "needs_review"
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
        "selectedScenarioIds": scenario_ids or [],
        "recommendation": recommendation,
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
        "",
        "## Scenarios",
        "",
        "| Scenario | Pass | Quality | Latency | Fallback | Unique Selection | Elapsed Range | Missing Runs |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in report["scenarios"]:
        missing = ", ".join(str(index) for index in item["missingRunIndexes"]) or "-"
        elapsed_range = (
            f"{item['elapsedMsMin']}..{item['elapsedMsMax']}"
            if item["elapsedMsMin"] is not None
            else "-"
        )
        lines.append(
            f"| `{item['scenarioId']}` | `{item['passCount']}/{item['runCount']}` | "
            f"`{item['qualityPassCount']}/{item['runCount']}` | "
            f"`{item['latencyPassCount']}/{item['runCount']}` | "
            f"`{item['fallbackPassCount']}/{item['runCount']}` | "
            f"`{item['uniqueSelectionCount']}` | {elapsed_range} | {missing} |"
        )
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
        scenario_ids=parse_scenario_ids(args.scenario_id),
    )
    paths = write_report(report, output_dir)
    print(json.dumps({"report": report, "paths": paths}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

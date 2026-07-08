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


DEFAULT_OUTPUT_DIR = Path("tests/evaluation/.artifacts/gemma4-quality-seed")
RESULT_FILENAME = "gemma4-quality-seed-results-filled.json"


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("gemma4-combined-%Y%m%dT%H%M%SZ")


def sanitize_run_id(value: str | None) -> str:
    raw = value or utc_run_id()
    sanitized = re.sub(r"[^A-Za-z0-9_.:-]", "-", raw).strip("-")
    return sanitized or utc_run_id()


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"results payload must be a JSON object: {path}")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError(f"results payload must contain scenarios[]: {path}")
    return payload


def _scenario_ids_from_seed(seed_path: Path | None) -> list[str]:
    if not seed_path:
        return []
    return [str(row["scenarioId"]) for row in seed_runner.load_seed(seed_path)]


def combine_payloads(
    *,
    result_paths: list[Path],
    run_id: str,
    seed_path: Path | None = None,
) -> dict[str, Any]:
    if not result_paths:
        raise ValueError("at least one --results path is required")

    scenario_by_id: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    source_runs: list[dict[str, Any]] = []
    duplicate_ids: list[str] = []
    selected_ids_in_source_order: list[str] = []

    for path in result_paths:
        payload = load_payload(path)
        source_runs.append({
            "runId": payload.get("runId"),
            "mode": payload.get("mode"),
            "path": str(path),
            "scenarioCount": payload.get("scenarioCount"),
            "completedScenarioCount": payload.get("completedScenarioCount"),
            "failedScenarioCount": payload.get("failedScenarioCount"),
        })
        for failure in payload.get("failures") or []:
            if isinstance(failure, dict):
                copied_failure = dict(failure)
                copied_failure.setdefault("sourceResultPath", str(path))
                failures.append(copied_failure)
        for scenario in payload.get("scenarios") or []:
            if not isinstance(scenario, dict):
                continue
            scenario_id = str(scenario.get("scenarioId") or "")
            if not scenario_id:
                continue
            if scenario_id in scenario_by_id:
                duplicate_ids.append(scenario_id)
                continue
            scenario_by_id[scenario_id] = scenario
            selected_ids_in_source_order.append(scenario_id)

    if duplicate_ids:
        unique_duplicates = ", ".join(sorted(set(duplicate_ids)))
        raise ValueError(f"duplicate scenarioId across result files: {unique_duplicates}")

    seed_order = _scenario_ids_from_seed(seed_path)
    if seed_order:
        selected_ids = [scenario_id for scenario_id in seed_order if scenario_id in scenario_by_id]
    else:
        selected_ids = selected_ids_in_source_order

    missing_seed_ids = [scenario_id for scenario_id in seed_order if scenario_id not in scenario_by_id]
    scenarios = [scenario_by_id[scenario_id] for scenario_id in selected_ids]
    payload = {
        "runId": run_id,
        "mode": "live-results-combined",
        "seedPath": str(seed_path) if seed_path else None,
        "scenarioCount": len(selected_ids),
        "completedScenarioCount": len(scenarios),
        "failedScenarioCount": len(failures),
        "selectedScenarioIds": selected_ids,
        "completedScenarioIds": [str(item.get("scenarioId")) for item in scenarios if item.get("scenarioId")],
        "missingSeedScenarioIds": missing_seed_ids,
        "sourceResultPaths": [str(path) for path in result_paths],
        "sourceRuns": source_runs,
        "failures": failures,
        "scenarios": scenarios,
    }
    seed_runner.assert_no_forbidden_tokens(payload)
    return payload


def write_payload(payload: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / RESULT_FILENAME
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(out_path)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combine Gemma4 live quality seed result batches.")
    parser.add_argument("--results", action="append", required=True, help="Result JSON path. Repeat for each batch.")
    parser.add_argument("--seed", help="Optional seed path. When provided, combined scenarios follow seed order.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-id")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_id = sanitize_run_id(args.run_id)
    output_dir = Path(args.output_dir) / run_id
    payload = combine_payloads(
        result_paths=[Path(path) for path in args.results],
        run_id=run_id,
        seed_path=Path(args.seed) if args.seed else None,
    )
    out_path = write_payload(payload, output_dir)
    print(json.dumps({
        "runId": run_id,
        "output": str(out_path),
        "scenarioCount": payload["scenarioCount"],
        "completedScenarioCount": payload["completedScenarioCount"],
        "failedScenarioCount": payload["failedScenarioCount"],
        "missingSeedScenarioIds": payload["missingSeedScenarioIds"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

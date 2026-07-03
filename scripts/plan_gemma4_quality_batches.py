from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_gemma4_quality_seed as seed_runner


DEFAULT_SEED_PATH = Path("tests/fixtures/gemma4_routine_quality_eval_seed_v3.jsonl")
DEFAULT_OUTPUT_DIR = Path("tests/evaluation/.artifacts/gemma4-quality-batches")
DEFAULT_LIVE_OUTPUT_DIR = Path("tests/evaluation/.artifacts/gemma4-quality-seed-live-v4")
DEFAULT_SCORE_OUTPUT_DIR = Path("tests/evaluation/.artifacts/gemma4-quality-seed-v4-score")
DEFAULT_MAX_ELAPSED_MS = 120_000
DEFAULT_SCENARIO_TIMEOUT_SEC = 240
DEFAULT_BATCHES = [
    {
        "batchId": "safety-core",
        "purpose": "Core pain/mobility/push safety smoke.",
        "scenarioIds": [
            "lower-back-legs-001",
            "limited-ankle-legs-001",
            "shoulder-push-001",
        ],
    },
    {
        "batchId": "time-equipment",
        "purpose": "Short time and equipment restriction behavior.",
        "scenarioIds": [
            "short-time-push-001",
            "home-equipment-001",
            "low-readiness-home-short-fullbody-001",
        ],
    },
    {
        "batchId": "beginner-joint",
        "purpose": "Beginner, knee, ankle, and wrist constraints.",
        "scenarioIds": [
            "beginner-lower-skill-001",
            "knee-pain-limited-ankle-beginner-001",
            "wrist-limited-push-001",
        ],
    },
    {
        "batchId": "medical-profile",
        "purpose": "Surgery, profile injury, DOMS, and hard-stop policies.",
        "scenarioIds": [
            "post-surgery-caution-001",
            "shoulder-surgery-home-push-001",
            "profile-injury-plus-doms-001",
            "all-major-pain-extreme-001",
        ],
    },
    {
        "batchId": "advanced-effect",
        "purpose": "Pull, effect-vs-risk, anthropometry, and advanced readiness.",
        "scenarioIds": [
            "upper-back-doms-lower-back-pull-short-001",
            "high-risk-high-effect-substitution-001",
            "long-femur-squat-setup-001",
            "low-readiness-fullbody-001",
            "advanced-high-readiness-effect-safe-001",
        ],
    },
]


def utc_run_prefix() -> str:
    return datetime.now(timezone.utc).strftime("gemma4-v4-batched-%Y%m%dT%H%M%SZ")


def sanitize_run_id(value: str | None) -> str:
    raw = value or utc_run_prefix()
    cleaned = "".join(char if char.isalnum() or char in "_.:-" else "-" for char in raw).strip("-")
    return cleaned or utc_run_prefix()


def _shell_join(args: list[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in args)


def _scenario_arg(scenario_ids: list[str]) -> str:
    return ",".join(scenario_ids)


def validate_batches(seed_rows: list[dict[str, Any]], batches: list[dict[str, Any]]) -> None:
    seed_ids = [str(row["scenarioId"]) for row in seed_rows]
    seed_set = set(seed_ids)
    seen: list[str] = []
    for batch in batches:
        for scenario_id in batch["scenarioIds"]:
            if scenario_id not in seed_set:
                raise ValueError(f"batch {batch['batchId']} references unknown scenarioId: {scenario_id}")
            seen.append(scenario_id)
    duplicates = sorted({scenario_id for scenario_id in seen if seen.count(scenario_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate batched scenarioId values: {', '.join(duplicates)}")
    missing = [scenario_id for scenario_id in seed_ids if scenario_id not in set(seen)]
    if missing:
        raise ValueError(f"seed scenarioIds missing from batches: {', '.join(missing)}")


def build_batch_plan(
    *,
    seed_path: Path,
    output_dir: Path,
    run_prefix: str,
    live_output_dir: Path,
    score_output_dir: Path,
    max_elapsed_ms: int,
    scenario_timeout_sec: int,
    cooldown_sec: float,
    resume: bool,
    selected_batch_ids: list[str] | None = None,
) -> dict[str, Any]:
    seed_rows = seed_runner.load_seed(seed_path)
    batches = [
        batch for batch in DEFAULT_BATCHES
        if not selected_batch_ids or batch["batchId"] in selected_batch_ids
    ]
    if not batches:
        raise ValueError("no batches selected")
    validate_batches(seed_rows, DEFAULT_BATCHES)

    plan_batches: list[dict[str, Any]] = []
    for batch in batches:
        batch_id = batch["batchId"]
        scenario_ids = list(batch["scenarioIds"])
        run_id = f"{run_prefix}-{batch_id}"
        score_run_id = f"{run_id}-score"
        result_path = live_output_dir / run_id / "gemma4-quality-seed-results-filled.json"
        scenario_arg = _scenario_arg(scenario_ids)
        live_args = [
            "python3",
            "scripts/run_gemma4_quality_seed_live.py",
            "--seed",
            str(seed_path),
            "--output-dir",
            str(live_output_dir),
            "--run-id",
            run_id,
            "--scenario-id",
            scenario_arg,
            "--scenario-timeout-sec",
            str(scenario_timeout_sec),
            "--cooldown-sec",
            str(cooldown_sec),
        ]
        if resume:
            live_args.append("--resume")
        score_args = [
            "python3",
            "scripts/run_gemma4_quality_seed.py",
            "score-file",
            "--seed",
            str(seed_path),
            "--results",
            str(result_path),
            "--output-dir",
            str(score_output_dir),
            "--run-id",
            score_run_id,
            "--scenario-id",
            scenario_arg,
            "--max-elapsed-ms",
            str(max_elapsed_ms),
        ]
        plan_batches.append({
            "batchId": batch_id,
            "purpose": batch["purpose"],
            "scenarioIds": scenario_ids,
            "runId": run_id,
            "scoreRunId": score_run_id,
            "resultPath": str(result_path),
            "liveCommand": live_args,
            "liveShell": _shell_join(live_args),
            "scoreCommand": score_args,
            "scoreShell": _shell_join(score_args),
        })

    return {
        "runPrefix": run_prefix,
        "mode": "gemma4-quality-batch-plan",
        "seedPath": str(seed_path),
        "outputDir": str(output_dir),
        "liveOutputDir": str(live_output_dir),
        "scoreOutputDir": str(score_output_dir),
        "maxElapsedMs": max_elapsed_ms,
        "scenarioTimeoutSec": scenario_timeout_sec,
        "cooldownSec": cooldown_sec,
        "resume": resume,
        "seedScenarioCount": len(seed_rows),
        "selectedBatchCount": len(plan_batches),
        "selectedScenarioCount": sum(len(batch["scenarioIds"]) for batch in plan_batches),
        "batches": plan_batches,
    }


def render_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# Gemma4 Routine Quality Batch Plan",
        "",
        f"- run_prefix: `{plan['runPrefix']}`",
        f"- seed: `{plan['seedPath']}`",
        f"- selected batches: `{plan['selectedBatchCount']}`",
        f"- selected scenarios: `{plan['selectedScenarioCount']}` / `{plan['seedScenarioCount']}`",
        f"- max elapsed gate: `{plan['maxElapsedMs']} ms`",
        "",
        "## Batches",
        "",
    ]
    for batch in plan["batches"]:
        lines.extend([
            f"### {batch['batchId']}",
            "",
            batch["purpose"],
            "",
            "Scenarios:",
            "",
        ])
        lines.extend(f"- `{scenario_id}`" for scenario_id in batch["scenarioIds"])
        lines.extend([
            "",
            "Live:",
            "",
            "```bash",
            batch["liveShell"],
            "```",
            "",
            "Score:",
            "",
            "```bash",
            batch["scoreShell"],
            "```",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def write_plan(plan: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "gemma4-quality-batch-plan.json"
    md_path = output_dir / "gemma4-quality-batch-plan.md"
    json_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(plan), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def parse_batch_ids(values: list[str] | None) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        for raw in str(value).split(","):
            batch_id = raw.strip()
            if batch_id and batch_id not in seen:
                ids.append(batch_id)
                seen.add(batch_id)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a bounded Gemma4 routine quality live-eval batch plan.")
    parser.add_argument("--seed", default=str(DEFAULT_SEED_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-prefix")
    parser.add_argument("--live-output-dir", default=str(DEFAULT_LIVE_OUTPUT_DIR))
    parser.add_argument("--score-output-dir", default=str(DEFAULT_SCORE_OUTPUT_DIR))
    parser.add_argument("--max-elapsed-ms", type=int, default=DEFAULT_MAX_ELAPSED_MS)
    parser.add_argument("--scenario-timeout-sec", type=int, default=DEFAULT_SCENARIO_TIMEOUT_SEC)
    parser.add_argument("--cooldown-sec", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-id", action="append", help="Plan only selected batch IDs. Repeat or comma-separate.")
    args = parser.parse_args()

    run_prefix = sanitize_run_id(args.run_prefix)
    output_dir = Path(args.output_dir) / run_prefix
    plan = build_batch_plan(
        seed_path=Path(args.seed),
        output_dir=output_dir,
        run_prefix=run_prefix,
        live_output_dir=Path(args.live_output_dir),
        score_output_dir=Path(args.score_output_dir),
        max_elapsed_ms=args.max_elapsed_ms,
        scenario_timeout_sec=args.scenario_timeout_sec,
        cooldown_sec=args.cooldown_sec,
        resume=args.resume,
        selected_batch_ids=parse_batch_ids(args.batch_id),
    )
    paths = write_plan(plan, output_dir)
    print(json.dumps({"plan": plan, "paths": paths}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


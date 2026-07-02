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

from scripts import check_local_llm_readiness as readiness

DEFAULT_SEED_PATH = Path("tests/fixtures/gemma4_routine_quality_eval_seed.jsonl")
DEFAULT_OUTPUT_DIR = Path("tests/evaluation/.artifacts/gemma4-quality-seed")
REQUIRED_FIELDS = {
    "scenarioId",
    "title",
    "input",
    "expectedHardConstraints",
    "preferredPatterns",
    "failurePatterns",
    "qualityRubric",
}
FORBIDDEN_REPORT_TOKENS = {
    "raw_prompt",
    "raw prompt",
    "raw_llm_output",
    "raw model response",
    "api_key",
    "GOOGLE_API_KEY",
    "authorization",
    "bearer ",
}


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("gemma4-quality-seed-%Y%m%dT%H%M%SZ")


def sanitize_run_id(value: str | None) -> str:
    raw = value or utc_run_id()
    sanitized = re.sub(r"[^A-Za-z0-9_.:-]", "-", raw).strip("-")
    return sanitized or utc_run_id()


def load_seed(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = sorted(REQUIRED_FIELDS - set(row))
        if missing:
            raise ValueError(f"seed row {line_no} missing fields: {', '.join(missing)}")
        rows.append(row)
    if not rows:
        raise ValueError("seed file is empty")
    ids = [str(row["scenarioId"]) for row in rows]
    duplicates = sorted([scenario_id for scenario_id, count in Counter(ids).items() if count > 1])
    if duplicates:
        raise ValueError(f"duplicate scenarioId values: {', '.join(duplicates)}")
    return rows


def summarize_seed(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "scenario_count": len(rows),
        "scenario_ids": [row["scenarioId"] for row in rows],
        "hard_constraint_count": sum(len(row.get("expectedHardConstraints", [])) for row in rows),
        "failure_pattern_count": sum(len(row.get("failurePatterns", [])) for row in rows),
        "rubric_item_count": sum(len(row.get("qualityRubric", [])) for row in rows),
    }


def build_plan_report(
    *,
    seed_path: Path,
    output_dir: Path,
    run_id: str,
    model: str,
    ollama_base_url: str,
    timeout_sec: float,
    probe: bool,
) -> dict[str, Any]:
    rows = load_seed(seed_path)
    readiness_report = readiness.build_readiness_report(
        base_url=ollama_base_url,
        model=model,
        timeout_sec=timeout_sec,
        probe=probe,
    )
    ready = readiness_report.get("readiness_status") == "pass"
    return {
        "run_id": run_id,
        "mode": "plan",
        "seed_path": str(seed_path),
        "output_dir": str(output_dir),
        "model": model,
        "ollama_readiness": {
            key: readiness_report.get(key)
            for key in (
                "readiness_status",
                "error_category",
                "next_action",
                "ollama_reachable",
                "model_installed",
                "windows_install_detected",
                "windows_executable_detected",
                "base_url_host",
            )
        },
        "seed_summary": summarize_seed(rows),
        "can_run_live_eval": ready,
        "next_actions": next_actions_for_plan(readiness_report),
    }


def next_actions_for_plan(readiness_report: dict[str, Any]) -> list[str]:
    status = readiness_report.get("readiness_status")
    action = readiness_report.get("next_action")
    if status == "pass":
        return [
            "run live routine generation for each seed scenario",
            "save sanitized model outputs",
            "score outputs with score-file",
            "compare base Gemma vs tuned Gemma before accepting fine-tuning",
        ]
    if action == "windows_ollama_detected_but_http_unreachable":
        return [
            "open Windows Ollama host for WSL/AI server access",
            "set OLLAMA_BASE_URL to the reachable Windows host IP",
            "rerun readiness with probe enabled",
        ]
    if action == "pull_model_or_update_LOCAL_LLM_MODEL":
        return [
            "run ollama list",
            "pull the target model or update LOCAL_LLM_MODEL",
            "rerun readiness with probe enabled",
        ]
    return [
        "start or install Ollama",
        "confirm /api/tags is reachable",
        "rerun readiness with probe enabled",
    ]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(normalize_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(f"{key} {normalize_text(val)}" for key, val in value.items())
    return str(value).lower()


def score_result(seed_row: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    selected_ids = [str(item).lower() for item in result.get("selectedExerciseIds", [])]
    output_text = normalize_text({
        "selectedExerciseIds": selected_ids,
        "rationaleText": result.get("rationaleText"),
        "warnings": result.get("warnings"),
        "notes": result.get("notes"),
    })
    failure_hits = [
        pattern
        for pattern in seed_row.get("failurePatterns", [])
        if str(pattern).lower() in output_text
    ]
    preferred_hits = [
        pattern
        for pattern in seed_row.get("preferredPatterns", [])
        if str(pattern).lower() in output_text
    ]
    hard_constraint_notes = seed_row.get("expectedHardConstraints", [])
    contract_valid = bool(result.get("contractValid", True))
    hard_violation_count = int(result.get("hardViolationCount", 0) or 0)
    passed = contract_valid and hard_violation_count == 0 and not failure_hits
    return {
        "scenarioId": seed_row["scenarioId"],
        "title": seed_row["title"],
        "passed": passed,
        "contractValid": contract_valid,
        "hardViolationCount": hard_violation_count,
        "failureHits": failure_hits,
        "preferredHits": preferred_hits,
        "expectedHardConstraints": hard_constraint_notes,
        "missingPreferredPatternCount": max(0, len(seed_row.get("preferredPatterns", [])) - len(preferred_hits)),
    }


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("scenarios", payload if isinstance(payload, list) else [])
    if not isinstance(rows, list):
        raise ValueError("results file must be a list or contain scenarios[]")
    result_map: dict[str, dict[str, Any]] = {}
    for row in rows:
        scenario_id = row.get("scenarioId")
        if scenario_id:
            result_map[str(scenario_id)] = row
    return result_map


def build_score_report(*, seed_path: Path, results_path: Path, output_dir: Path, run_id: str) -> dict[str, Any]:
    rows = load_seed(seed_path)
    result_map = load_results(results_path)
    scenario_reports = []
    missing_results = []
    for row in rows:
        result = result_map.get(str(row["scenarioId"]))
        if not result:
            missing_results.append(row["scenarioId"])
            continue
        scenario_reports.append(score_result(row, result))
    passed = sum(1 for item in scenario_reports if item["passed"])
    failed = len(scenario_reports) - passed
    return {
        "run_id": run_id,
        "mode": "score-file",
        "seed_path": str(seed_path),
        "results_path": str(results_path),
        "output_dir": str(output_dir),
        "total_scenarios": len(rows),
        "scored_scenarios": len(scenario_reports),
        "passed": passed,
        "failed": failed,
        "missing_results": missing_results,
        "scenario_reports": scenario_reports,
        "recommendation": "pass" if failed == 0 and not missing_results else "needs_review",
    }


def build_results_template(*, seed_path: Path, output_dir: Path, run_id: str) -> dict[str, Any]:
    rows = load_seed(seed_path)
    return {
        "runId": run_id,
        "mode": "results-template",
        "seedPath": str(seed_path),
        "outputDir": str(output_dir),
        "instructions": [
            "Fill selectedExerciseIds with the final model-selected exercise ids only.",
            "Use rationaleText for a short sanitized rationale summary, not an unredacted prompt or full model transcript.",
            "Set contractValid and hardViolationCount after schema/contract validation.",
            "Keep notes short and evidence-oriented so score-file can compare runs consistently.",
        ],
        "scenarios": [
            {
                "scenarioId": row["scenarioId"],
                "title": row["title"],
                "inputSummary": {
                    "targetSplitLabel": row.get("input", {}).get("targetSplitLabel"),
                    "timeAvailableMin": row.get("input", {}).get("timeAvailableMin"),
                    "readinessLevel": row.get("input", {}).get("readinessLevel"),
                    "currentPainAreas": row.get("input", {}).get("currentPainAreas", []),
                    "currentDoms": row.get("input", {}).get("currentDoms", []),
                    "mobilityLimits": row.get("input", {}).get("mobilityLimits", []),
                    "availableEquipment": row.get("input", {}).get("availableEquipment", []),
                    "unavailableEquipment": row.get("input", {}).get("unavailableEquipment", []),
                },
                "expectedHardConstraints": row.get("expectedHardConstraints", []),
                "preferredPatterns": row.get("preferredPatterns", []),
                "failurePatterns": row.get("failurePatterns", []),
                "qualityRubric": row.get("qualityRubric", []),
                "selectedExerciseIds": [],
                "rationaleText": "",
                "warnings": [],
                "contractValid": False,
                "hardViolationCount": 0,
                "notes": "",
            }
            for row in rows
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Gemma4 Routine Quality Seed Report",
        "",
        f"- run_id: `{report['run_id']}`",
        f"- mode: `{report['mode']}`",
        f"- seed: `{report['seed_path']}`",
        "",
    ]
    if report["mode"] == "plan":
        readiness_report = report["ollama_readiness"]
        lines.extend([
            "## Readiness",
            "",
            f"- status: `{readiness_report.get('readiness_status')}`",
            f"- next_action: `{readiness_report.get('next_action')}`",
            f"- windows_install_detected: `{readiness_report.get('windows_install_detected')}`",
            f"- windows_executable_detected: `{readiness_report.get('windows_executable_detected')}`",
            "",
            "## Seed Summary",
            "",
            f"- scenarios: `{report['seed_summary']['scenario_count']}`",
            f"- hard constraints: `{report['seed_summary']['hard_constraint_count']}`",
            f"- failure patterns: `{report['seed_summary']['failure_pattern_count']}`",
            "",
            "## Next Actions",
            "",
        ])
        lines.extend(f"- {item}" for item in report["next_actions"])
    else:
        lines.extend([
            "## Score Summary",
            "",
            f"- scored: `{report['scored_scenarios']}` / `{report['total_scenarios']}`",
            f"- passed: `{report['passed']}`",
            f"- failed: `{report['failed']}`",
            f"- recommendation: `{report['recommendation']}`",
            "",
            "## Scenario Results",
            "",
        ])
        for item in report["scenario_reports"]:
            lines.append(
                f"- `{item['scenarioId']}`: {'PASS' if item['passed'] else 'FAIL'} "
                f"(failures={len(item['failureHits'])}, hardViolations={item['hardViolationCount']})"
            )
        if report["missing_results"]:
            lines.append("")
            lines.append("## Missing Results")
            lines.extend(f"- `{item}`" for item in report["missing_results"])
    return "\n".join(lines) + "\n"


def assert_no_forbidden_tokens(payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False).lower()
    leaked = sorted(token for token in FORBIDDEN_REPORT_TOKENS if token.lower() in text)
    if leaked:
        raise ValueError(f"forbidden report tokens found: {', '.join(leaked)}")


def write_report(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    assert_no_forbidden_tokens(report)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "gemma4-quality-seed-report.json"
    md_path = output_dir / "gemma4-quality-seed-report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def write_results_template(template: dict[str, Any], output_dir: Path) -> dict[str, str]:
    assert_no_forbidden_tokens(template)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "gemma4-quality-seed-results-template.json"
    json_path.write_text(json.dumps(template, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"json": str(json_path)}


def cmd_validate(args: argparse.Namespace) -> None:
    rows = load_seed(Path(args.seed))
    print(json.dumps(summarize_seed(rows), indent=2, ensure_ascii=False))


def cmd_plan(args: argparse.Namespace) -> None:
    run_id = sanitize_run_id(args.run_id)
    output_dir = Path(args.output_dir) / run_id
    report = build_plan_report(
        seed_path=Path(args.seed),
        output_dir=output_dir,
        run_id=run_id,
        model=args.model,
        ollama_base_url=args.ollama_base_url,
        timeout_sec=args.timeout_sec,
        probe=args.probe,
    )
    paths = write_report(report, output_dir)
    print(json.dumps({"report": report, "paths": paths}, indent=2, ensure_ascii=False))


def cmd_template(args: argparse.Namespace) -> None:
    run_id = sanitize_run_id(args.run_id)
    output_dir = Path(args.output_dir) / run_id
    template = build_results_template(
        seed_path=Path(args.seed),
        output_dir=output_dir,
        run_id=run_id,
    )
    paths = write_results_template(template, output_dir)
    print(json.dumps({"template": template, "paths": paths}, indent=2, ensure_ascii=False))


def cmd_score_file(args: argparse.Namespace) -> None:
    run_id = sanitize_run_id(args.run_id)
    output_dir = Path(args.output_dir) / run_id
    report = build_score_report(
        seed_path=Path(args.seed),
        results_path=Path(args.results),
        output_dir=output_dir,
        run_id=run_id,
    )
    paths = write_report(report, output_dir)
    print(json.dumps({"report": report, "paths": paths}, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run lightweight Gemma4 routine quality seed checks.")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("--seed", default=str(DEFAULT_SEED_PATH))
    validate.set_defaults(func=cmd_validate)

    plan = sub.add_parser("plan")
    plan.add_argument("--seed", default=str(DEFAULT_SEED_PATH))
    plan.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    plan.add_argument("--run-id")
    plan.add_argument("--model", default="gemma4:latest")
    plan.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    plan.add_argument("--timeout-sec", type=float, default=2)
    plan.add_argument("--probe", action=argparse.BooleanOptionalAction, default=False)
    plan.set_defaults(func=cmd_plan)

    template = sub.add_parser("template")
    template.add_argument("--seed", default=str(DEFAULT_SEED_PATH))
    template.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    template.add_argument("--run-id")
    template.set_defaults(func=cmd_template)

    score_file = sub.add_parser("score-file")
    score_file.add_argument("--seed", default=str(DEFAULT_SEED_PATH))
    score_file.add_argument("--results", required=True)
    score_file.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    score_file.add_argument("--run-id")
    score_file.set_defaults(func=cmd_score_file)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

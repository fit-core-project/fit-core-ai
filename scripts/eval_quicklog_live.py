from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIOS = ROOT / "tests" / "evaluation" / "fixtures" / "quicklog_scenarios.json"
DEFAULT_OUT_DIR = ROOT / "tests" / "evaluation" / ".artifacts" / "quicklog-live"


class _DietPipeFailure:
    def invoke(self, payload: Any) -> None:  # noqa: ARG002
        raise RuntimeError("LLM failure mock")

    def __ror__(self, other: Any) -> "_DietPipeFailure":  # noqa: ARG002
        return self

    def __or__(self, other: Any) -> "_DietPipeFailure":  # noqa: ARG002
        return self


class _LlmDietFailure:
    def with_structured_output(self, schema: Any) -> _DietPipeFailure:  # noqa: ARG002
        return _DietPipeFailure()

    def __or__(self, other: Any) -> _DietPipeFailure:  # noqa: ARG002
        return _DietPipeFailure()

    def __ror__(self, other: Any) -> _DietPipeFailure:  # noqa: ARG002
        return _DietPipeFailure()

    def invoke(self, payload: Any) -> None:  # noqa: ARG002
        raise RuntimeError("LLM failure mock")


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("quicklog-live-%Y%m%dT%H%M%SZ")


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["scenarios"])


def select_scenarios(
    scenarios: list[dict[str, Any]],
    case_ids: list[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected = scenarios
    if case_ids:
        wanted = set(case_ids)
        selected = [scenario for scenario in scenarios if scenario["id"] in wanted]
        missing = sorted(wanted - {scenario["id"] for scenario in selected})
        if missing:
            raise SystemExit(f"case id not found: {', '.join(missing)}")
    if limit is not None:
        selected = selected[:limit]
    return selected


def child_main(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(ROOT))
    from engines.quicklog import diet_parser
    from engines.quicklog.diet_parser import parse_diet_log

    scenario = next(
        scenario for scenario in load_scenarios(Path(args.scenarios)) if scenario["id"] == args.single_case_id
    )
    if args.mode == "fallback":
        diet_parser.get_llm = lambda *a, **kw: _LlmDietFailure()
    else:
        os.environ.setdefault("APP_ENV", "development")
        os.environ.setdefault("LLM_PROVIDER", "local")
        os.environ.setdefault("LOCAL_LLM_MODEL", "gemma4:latest")
        os.environ.setdefault("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        os.environ.setdefault("ENABLE_LOCAL_RAW_JSON_INVOKE", "true")
        os.environ.setdefault("LOCAL_LLM_NUM_PREDICT", str(args.num_predict))
        os.environ.setdefault("LOCAL_LLM_NUM_CTX", str(args.num_ctx))

    print(parse_diet_log(scenario["input"]))
    return 0


def score_case(scenario: dict[str, Any], result_json: str, elapsed_ms: float) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from tests.evaluation.test_quicklog_eval import _score_case

    return _score_case(scenario, result_json, elapsed_ms)


def run_single_case(args: argparse.Namespace, scenario: dict[str, Any]) -> dict[str, Any]:
    if args.mode == "fallback":
        sys.path.insert(0, str(ROOT))
        from engines.quicklog import diet_parser
        from engines.quicklog.diet_parser import parse_diet_log

        diet_parser.get_llm = lambda *a, **kw: _LlmDietFailure()
        started = time.perf_counter()
        try:
            result_json = parse_diet_log(scenario["input"])
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            scored = score_case(scenario, result_json, elapsed_ms)
            return {
                "id": scenario["id"],
                "input": scenario["input"],
                "mode": args.mode,
                "status": "pass" if scored.get("overall_pass") else "fail",
                **scored,
            }
        except Exception as exc:  # noqa: BLE001 - evaluation harness should record, not crash.
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            return {
                "id": scenario["id"],
                "input": scenario["input"],
                "mode": args.mode,
                "status": "error",
                "overall_pass": False,
                "elapsed_ms": elapsed_ms,
                "error": str(exc),
            }

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-case-id",
        scenario["id"],
        "--mode",
        args.mode,
        "--scenarios",
        str(args.scenarios),
        "--num-predict",
        str(args.num_predict),
        "--num-ctx",
        str(args.num_ctx),
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            text=True,
            capture_output=True,
            timeout=args.timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        return {
            "id": scenario["id"],
            "input": scenario["input"],
            "mode": args.mode,
            "status": "timeout",
            "overall_pass": False,
            "elapsed_ms": elapsed_ms,
            "error": f"case timed out after {args.timeout_sec}s",
        }

    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    stdout = completed.stdout.strip()
    if completed.returncode != 0:
        return {
            "id": scenario["id"],
            "input": scenario["input"],
            "mode": args.mode,
            "status": "error",
            "overall_pass": False,
            "elapsed_ms": elapsed_ms,
            "error": completed.stderr.strip() or stdout,
        }
    try:
        scored = score_case(scenario, stdout, elapsed_ms)
    except Exception as exc:  # noqa: BLE001 - evaluation harness should record, not crash.
        return {
            "id": scenario["id"],
            "input": scenario["input"],
            "mode": args.mode,
            "status": "score_error",
            "overall_pass": False,
            "elapsed_ms": elapsed_ms,
            "error": str(exc),
            "raw": stdout,
        }
    return {
        "id": scenario["id"],
        "input": scenario["input"],
        "mode": args.mode,
        "status": "pass" if scored.get("overall_pass") else "fail",
        **scored,
    }


def write_outputs(args: argparse.Namespace, results: list[dict[str, Any]]) -> tuple[Path, Path]:
    run_id = args.run_id or utc_run_id()
    out_dir = Path(args.out_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "quicklog_live_eval_results.json"
    md_path = out_dir / "quicklog_live_eval_summary.md"
    total = len(results)
    pass_count = sum(1 for result in results if result.get("overall_pass"))
    timeout_count = sum(1 for result in results if result.get("status") == "timeout")
    error_count = sum(1 for result in results if result.get("status") in {"error", "score_error"})

    json_path.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = [
        "# QuickLog Live Eval Summary",
        "",
        f"- mode: `{args.mode}`",
        f"- total: {total}",
        f"- pass: {pass_count}",
        f"- timeout: {timeout_count}",
        f"- error: {error_count}",
        f"- timeoutSec: {args.timeout_sec}",
        "",
        "| ID | Status | Foods | Elapsed ms | Error |",
        "|---|---|---|---:|---|",
    ]
    for result in results:
        foods = ", ".join(result.get("actual_food_names") or [])
        error = str(result.get("error") or "").replace("\n", " ")[:160]
        rows.append(
            f"| {result['id']} | {result.get('status')} | {foods} | {result.get('elapsed_ms')} | {error} |"
        )
    md_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded QuickLog live/fallback evaluation with per-case timeout.")
    parser.add_argument("--mode", choices=["real", "fallback"], default="fallback")
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--run-id")
    parser.add_argument("--timeout-sec", type=float, default=45.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--num-predict", type=int, default=512)
    parser.add_argument("--num-ctx", type=int, default=2048)
    parser.add_argument("--single-case-id")
    args = parser.parse_args()
    if args.single_case_id:
        return args
    return args


def main() -> int:
    args = parse_args()
    if args.single_case_id:
        return child_main(args)

    scenarios = select_scenarios(load_scenarios(Path(args.scenarios)), args.case_id, args.limit)
    results = [run_single_case(args, scenario) for scenario in scenarios]
    json_path, md_path = write_outputs(args, results)
    print(f"results: {json_path}")
    print(f"summary: {md_path}")
    return 1 if any(not result.get("overall_pass") for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())

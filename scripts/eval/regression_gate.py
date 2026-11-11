"""Stage-eval regression gate: replay committed LLM outputs and fail if accuracy drops.

Profiles
  ci    diet harness, curated table only (no vector index, no MySQL, no GPU). Runs in CI.
  full  diet (with food vector index) + supplement (Chroma + local embeddings) + routine
        (MySQL exercise_tier). Run locally before pushing AI changes.

Every run uses `--llm cache` on tests/evaluation/llm_cache/*.json, so the score only moves
when deterministic code (parsing, matching, units, repair, composer) changes.

  python scripts/eval/regression_gate.py --profile ci
  python scripts/eval/regression_gate.py --profile full
  python scripts/eval/regression_gate.py --profile full --update-baseline   # after an intended change

A metric fails when it falls below its baseline value. Improvements pass and print a reminder
to refresh the baseline. A cache miss (prompt changed) fails the gate: refill it with
`--llm live` for the listed ids, then commit the updated cache.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "tests" / "evaluation" / ".artifacts"
BASELINE = ROOT / "tests" / "evaluation" / "baselines" / "stage_eval_baseline.json"
PY = sys.executable


def _run(args: list[str], env_extra: dict | None = None) -> None:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "AI_EMBEDDING_DEVICE": "cpu", **(env_extra or {})}
    print("$", " ".join(args), flush=True)
    proc = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise SystemExit(f"harness failed: {' '.join(args)}")


def diet_metrics(no_food_engine: bool) -> dict:
    out = ARTIFACTS / ("gate_diet_curated.json" if no_food_engine else "gate_diet_full.json")
    args = [PY, "scripts/eval/diet_stage_eval.py", "--llm", "cache", "--out-json", str(out),
            "--out-md", str(out.with_suffix(".md"))]
    if no_food_engine:
        args.append("--no-food-engine")
    _run(args)
    report = json.loads(out.read_text(encoding="utf-8"))
    summary = report["summary"]
    no_cache = sum(1 for case in report["cases"] if case.get("status") == "no_cache")
    return {
        "pass": summary["overall"]["pass"],
        "test_pass": summary["by_split"]["test"]["pass"],
        "total_kcal_acc_pct": round(summary["overall"]["total_kcal_acc"] * 100, 1),
        "parse_ok": summary["overall"]["n"] - summary["overall"]["first_fail"].get("S1_parse", 0),
        "_cache_misses": no_cache,
    }


def supplement_metrics() -> dict:
    _run([PY, "scripts/eval/supplement_stage_eval.py", "--llm", "cache", "--out-prefix", "gate_supplement"])
    report = json.loads((ARTIFACTS / "gate_supplement.json").read_text(encoding="utf-8"))
    summary = report["summary"]
    misses = sum(1 for case in report["cases"] if "llm_cache_miss" in ((case.get("s0") or {}).get("reasons") or []))
    return {
        "pass": summary["all"]["pass"],
        "test_pass": summary["test"]["pass"],
        "parse_ok": summary["all"]["stage_pass_independent"]["s1"],
        "_cache_misses": misses,
    }


def routine_metrics() -> dict:
    _run([PY, "scripts/eval/routine_stage_eval.py", "--llm", "cache", "--quiet"])
    report = json.loads((ARTIFACTS / "routine_stage_report.json").read_text(encoding="utf-8"))
    cases = report["cases"]

    def final_ok(case: dict) -> bool:
        return all(case["stages"].get(stage, {}).get("status") != "fail" for stage in ("S3", "S4", "S5"))

    return {
        "final_pass": sum(final_ok(c) for c in cases),
        "test_final_pass": sum(final_ok(c) for c in cases if c["split"] == "test"),
        "pool_ok": sum(c["stages"].get("S1", {}).get("status") != "fail" for c in cases),
        "_cache_misses": len(report["summary"].get("cacheMissCases") or []),
    }


PROFILES = {
    "ci": {"diet_curated": lambda: diet_metrics(no_food_engine=True)},
    "full": {
        "diet_full": lambda: diet_metrics(no_food_engine=False),
        "supplement": supplement_metrics,
        "routine": routine_metrics,
    },
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="ci")
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args()

    baseline = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}
    current = {name: fn() for name, fn in PROFILES[args.profile].items()}

    failures, improvements = [], []
    for name, metrics in current.items():
        misses = metrics.pop("_cache_misses", 0)
        if misses:
            failures.append(f"{name}: {misses} LLM cache misses (prompt changed) - refill with --llm live")
        for key, value in metrics.items():
            base = baseline.get(name, {}).get(key)
            mark = ""
            if base is None:
                mark = "(new)"
            elif value < base:
                failures.append(f"{name}.{key}: {value} < baseline {base}")
                mark = "REGRESSION"
            elif value > base:
                improvements.append(f"{name}.{key}: {base} -> {value}")
                mark = "improved"
            print(f"  {name:13s} {key:20s} {value!s:>7} (baseline {base}) {mark}")

    if args.update_baseline:
        if failures and any("cache misses" in f for f in failures):
            print("refusing to update baseline with cache misses")
            return 1
        merged = {**baseline, **current}
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"baseline updated: {BASELINE}")
        return 0
    if improvements:
        print("improved (refresh with --update-baseline):", *improvements, sep="\n  ")
    if failures:
        print("REGRESSION GATE FAILED:", *failures, sep="\n  ")
        return 1
    print("regression gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

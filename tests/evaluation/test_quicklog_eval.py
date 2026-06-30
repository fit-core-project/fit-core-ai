"""QuickLog 식단 파싱 평가 harness.

평가 대상: parse_diet_log (diet_parser.py) — 현 QuickLog UI가 호출하는 함수.
작업서에 기재된 parse_natural_language_log는 구버전(nlp_engine.py)이며,
현재 /api/ai/parse-diet 엔드포인트는 parse_diet_log를 사용함.

LLM 경로: REAL(gemma4 via Ollama, raw JSON invoke) + deterministic fallback 둘 다 측정.
ENABLE_LOCAL_RAW_JSON_INVOKE=true → 프로덕션 경로와 일치.

구조:
  Phase 1 — REAL 전체 (20건): get_llm 패치 없음, monkeypatch.setenv만
  Phase 2 — FALLB 전체 (20건): get_llm을 _LlmDietFailure로 일괄 패치
  → setattr 누적 오염 방지
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from engines.quicklog import diet_parser
from engines.quicklog.diet_parser import ParsedDietLog, parse_diet_log

SCENARIOS_PATH = Path(__file__).parent / "fixtures" / "quicklog_scenarios.json"
ARTIFACTS_DIR = Path(__file__).parent / ".artifacts"
REPORT_PATH = ARTIFACTS_DIR / "quicklog_eval_report.json"


# ---------- Fallback mock ----------

class _DietPipeFailure:
    """LangChain pipe chain 말단에 붙어 TypeError를 발생시켜 fallback 유도."""
    def invoke(self, payload: Any) -> None:
        raise RuntimeError("LLM failure mock — fallback path forced")

    def __ror__(self, other: Any) -> "_DietPipeFailure":
        return self

    def __or__(self, other: Any) -> "_DietPipeFailure":
        return self


class _LlmDietFailure:
    def with_structured_output(self, schema: Any) -> _DietPipeFailure:
        return _DietPipeFailure()

    def __or__(self, other: Any) -> _DietPipeFailure:
        return _DietPipeFailure()

    def __ror__(self, other: Any) -> _DietPipeFailure:
        return _DietPipeFailure()

    def invoke(self, payload: Any) -> None:
        raise RuntimeError("LLM failure mock")


# ---------- Metric helpers ----------

def _compute_kcal(item: dict) -> float | None:
    p, c, f = item.get("protein_g"), item.get("carbs_g"), item.get("fat_g")
    if any(v is None for v in (p, c, f)):
        return None
    return p * 4 + c * 4 + f * 9


def _kcal_error_pct(actual: float | None, ref: float) -> float | None:
    if actual is None or ref == 0:
        return None
    return abs(actual - ref) / ref * 100


def _macro_error(actual: float | None, ref: float) -> float | None:
    return None if actual is None else abs(actual - ref)


def _food_name_match(actual_names: list[str], expected_names: list[str]) -> bool:
    al = [n.lower() for n in actual_names]
    return any(
        any(exp.lower() in name or name in exp.lower() for name in al)
        for exp in expected_names
    )


def _amount_match(items: list[dict], exp_amount: float | None, exp_unit: str | None) -> bool:
    if exp_amount is None:
        return True
    for item in items:
        if item.get("amount") == exp_amount:
            if exp_unit is None:
                return True
            au = (item.get("unit") or "").lower()
            if exp_unit.lower() in au or au in exp_unit.lower():
                return True
    return False


def _score_case(scenario: dict, result_json: str, elapsed_ms: float) -> dict:
    exp = scenario["expected"]
    macros_mode = exp.get("macros_expected", "both")

    try:
        parsed = ParsedDietLog.model_validate_json(result_json)
        parse_success = True
    except Exception as e:
        return {
            "parse_success": False,
            "parse_error": str(e),
            "elapsed_ms": elapsed_ms,
            "actual_raw": result_json,
            "overall_pass": False,
        }

    items = [i.model_dump() for i in parsed.items]
    actual_names = [i["food_name"] for i in items]

    items_count_ok = len(items) >= exp.get("items_min", 0)
    food_match = (
        _food_name_match(actual_names, exp["food_names_include"])
        if exp.get("food_names_include") else None
    )
    amount_ok = _amount_match(items, exp.get("amount"), exp.get("unit"))

    best_kcal = kcal_err_pct = kcal_within_tol = None
    if "kcal_ref" in exp and items and macros_mode != "none":
        valid_kcals = [k for k in (_compute_kcal(i) for i in items) if k is not None]
        best_kcal = sum(valid_kcals) if valid_kcals else None
        kcal_ref = exp["kcal_ref"]
        tol_pct = exp.get("kcal_tol_pct", 15)
        kcal_err_pct = _kcal_error_pct(best_kcal, kcal_ref)
        kcal_within_tol = (kcal_err_pct is not None and kcal_err_pct <= tol_pct)

    protein_err = carbs_err = fat_err = macro_within_tol = None
    if macros_mode != "none" and items:
        tol = exp.get("macro_tol_g_known", 8) if macros_mode == "both" else exp.get("macro_tol_g_llm", 15)
        ps = [i["protein_g"] for i in items if i["protein_g"] is not None]
        cs = [i["carbs_g"]   for i in items if i["carbs_g"]   is not None]
        fs = [i["fat_g"]     for i in items if i["fat_g"]     is not None]
        tp = sum(ps) if ps else None
        tc = sum(cs) if cs else None
        tf = sum(fs) if fs else None
        if "protein_g_ref" in exp:
            protein_err = _macro_error(tp, exp["protein_g_ref"])
        if "carbs_g_ref" in exp:
            carbs_err = _macro_error(tc, exp["carbs_g_ref"])
        if "fat_g_ref" in exp:
            fat_err = _macro_error(tf, exp["fat_g_ref"])
        errs = [e for e in [protein_err, carbs_err, fat_err] if e is not None]
        macro_within_tol = all(e <= tol for e in errs) if errs else None

    meal_type_ok = None
    if "meal_type_expected" in exp and items:
        actual_meals = [i.get("meal_type") for i in items]
        meal_type_ok = exp["meal_type_expected"] in actual_meals

    workout_noise_ok = None
    if exp.get("workout_noise"):
        keywords = ["벤치프레스", "벤치", "스쿼트", "운동", "세트", "ȸ"]
        workout_noise_ok = not any(
            any(kw in n.lower() for kw in keywords) for n in actual_names
        )

    overall_pass = all([
        parse_success,
        items_count_ok,
        food_match if food_match is not None else True,
        amount_ok,
        kcal_within_tol if kcal_within_tol is not None else True,
        macro_within_tol if macro_within_tol is not None else True,
        meal_type_ok if meal_type_ok is not None else True,
        workout_noise_ok if workout_noise_ok is not None else True,
    ])

    return {
        "parse_success": parse_success,
        "items_count": len(items),
        "items_count_ok": items_count_ok,
        "actual_food_names": actual_names,
        "food_match": food_match,
        "amount_ok": amount_ok,
        "computed_kcal_total": round(best_kcal, 1) if best_kcal is not None else None,
        "kcal_ref": exp.get("kcal_ref"),
        "kcal_err_pct": round(kcal_err_pct, 1) if kcal_err_pct is not None else None,
        "kcal_within_tol": kcal_within_tol,
        "protein_err_g": round(protein_err, 1) if protein_err is not None else None,
        "carbs_err_g": round(carbs_err, 1) if carbs_err is not None else None,
        "fat_err_g": round(fat_err, 1) if fat_err is not None else None,
        "macro_within_tol": macro_within_tol,
        "meal_type_ok": meal_type_ok,
        "workout_noise_ok": workout_noise_ok,
        "macros_mode": macros_mode,
        "overall_pass": overall_pass,
        "elapsed_ms": round(elapsed_ms, 1),
        "actual_items": items,
    }


def _run_single(scenario: dict) -> dict:
    t0 = time.perf_counter()
    try:
        result_json = parse_diet_log(scenario["input"])
        elapsed_ms = (time.perf_counter() - t0) * 1000
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "parse_success": False,
            "exception": str(exc),
            "elapsed_ms": round(elapsed_ms, 1),
            "overall_pass": False,
            "actual_food_names": [],
            "actual_items": [],
        }
    return _score_case(scenario, result_json, (time.perf_counter() - t0) * 1000 + elapsed_ms - elapsed_ms)


def _run_single_timed(scenario: dict) -> dict:
    t0 = time.perf_counter()
    try:
        result_json = parse_diet_log(scenario["input"])
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "parse_success": False,
            "exception": str(exc),
            "elapsed_ms": round(elapsed_ms, 1),
            "overall_pass": False,
            "actual_food_names": [],
            "actual_items": [],
        }
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return _score_case(scenario, result_json, elapsed_ms)


# ---------- pytest test ----------

@pytest.mark.eval
@pytest.mark.llm
def test_quicklog_eval(monkeypatch):
    """QuickLog 식단 파싱 전체 평가 harness.

    assert는 invalid/graceful 케이스 예외 미발생만 강제.
    정확도 지표는 측정·기록만.
    """
    raw = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    scenarios = raw["scenarios"]

    # ── Phase 1: REAL LLM (get_llm 패치 없음) ──────────────────────────────
    monkeypatch.setenv("ENABLE_LOCAL_RAW_JSON_INVOKE", "true")
    print("\n\n" + "="*70)
    print("PHASE 1: REAL LLM (gemma4, raw JSON invoke)")
    print("="*70)

    real_results: dict[str, dict] = {}
    latencies_real: list[float] = []
    for sc in scenarios:
        print(f"\n[{sc['id']}] {sc['category']} | {sc['input']!r}")
        r = _run_single_timed(sc)
        real_results[sc["id"]] = r
        if r.get("parse_success") and r["elapsed_ms"] > 50:
            latencies_real.append(r["elapsed_ms"])
        print(f"  → foods={r.get('actual_food_names')} kcal={r.get('computed_kcal_total')} "
              f"kcal_err%={r.get('kcal_err_pct')} pass={r.get('overall_pass')} {r['elapsed_ms']:.0f}ms")

    # ── Phase 2: FALLBACK (get_llm → _LlmDietFailure) ──────────────────────
    monkeypatch.setattr(diet_parser, "get_llm", lambda *a, **kw: _LlmDietFailure())
    print("\n\n" + "="*70)
    print("PHASE 2: DETERMINISTIC FALLBACK (get_llm mocked)")
    print("="*70)

    fb_results: dict[str, dict] = {}
    for sc in scenarios:
        print(f"\n[{sc['id']}] {sc['category']} | {sc['input']!r}")
        r = _run_single_timed(sc)
        fb_results[sc["id"]] = r
        print(f"  → foods={r.get('actual_food_names')} kcal={r.get('computed_kcal_total')} "
              f"pass={r.get('overall_pass')} {r['elapsed_ms']:.0f}ms")

    # ── Invalid/graceful assert ─────────────────────────────────────────────
    for sc in scenarios:
        if sc["expected"].get("no_exception") or sc["expected"].get("graceful"):
            sid = sc["id"]
            assert real_results[sid].get("parse_success"), (
                f"[{sid}] graceful 케이스 REAL 예외: {real_results[sid].get('exception')}"
            )
            assert fb_results[sid].get("parse_success"), (
                f"[{sid}] graceful 케이스 FALLB 예외: {fb_results[sid].get('exception')}"
            )

    # ── 집계 ────────────────────────────────────────────────────────────────
    latencies_sorted = sorted(latencies_real)
    n = len(latencies_sorted)
    p50 = latencies_sorted[max(0, int(n * 0.50) - 1)] if n else None
    p95 = latencies_sorted[max(0, int(n * 0.95) - 1)] if n else None

    def _pct(key: str, bucket: dict[str, dict]) -> float:
        vals = [v[key] for v in bucket.values() if v.get(key) is not None]
        return round(sum(1 for v in vals if v is True) / len(vals) * 100, 1) if vals else 0.0

    def _cnt(key: str, bucket: dict[str, dict]) -> int:
        return sum(1 for v in bucket.values() if v.get(key) is True)

    summary = {
        "total_cases": len(scenarios),
        "latency_real_p50_ms": round(p50, 1) if p50 else None,
        "latency_real_p95_ms": round(p95, 1) if p95 else None,
        "llm_actually_ran_count": len(latencies_real),
        "real": {
            "parse_success_rate_pct": _pct("parse_success", real_results),
            "food_match_pct": _pct("food_match", real_results),
            "amount_ok_pct": _pct("amount_ok", real_results),
            "kcal_within_tol_pct": _pct("kcal_within_tol", real_results),
            "macro_within_tol_pct": _pct("macro_within_tol", real_results),
            "meal_type_ok_pct": _pct("meal_type_ok", real_results),
            "workout_noise_ok_pct": _pct("workout_noise_ok", real_results),
            "overall_pass_count": _cnt("overall_pass", real_results),
        },
        "fallback": {
            "parse_success_rate_pct": _pct("parse_success", fb_results),
            "food_match_pct": _pct("food_match", fb_results),
            "amount_ok_pct": _pct("amount_ok", fb_results),
            "kcal_within_tol_pct": _pct("kcal_within_tol", fb_results),
            "macro_within_tol_pct": _pct("macro_within_tol", fb_results),
            "meal_type_ok_pct": _pct("meal_type_ok", fb_results),
            "workout_noise_ok_pct": _pct("workout_noise_ok", fb_results),
            "overall_pass_count": _cnt("overall_pass", fb_results),
        },
    }

    print("\n\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"REAL    parse={summary['real']['parse_success_rate_pct']}% "
          f"food={summary['real']['food_match_pct']}% "
          f"kcal_tol={summary['real']['kcal_within_tol_pct']}% "
          f"macro_tol={summary['real']['macro_within_tol_pct']}% "
          f"overall={summary['real']['overall_pass_count']}/{len(scenarios)} "
          f"p50={summary['latency_real_p50_ms']}ms p95={summary['latency_real_p95_ms']}ms")
    print(f"FALLB   parse={summary['fallback']['parse_success_rate_pct']}% "
          f"food={summary['fallback']['food_match_pct']}% "
          f"kcal_tol={summary['fallback']['kcal_within_tol_pct']}% "
          f"macro_tol={summary['fallback']['macro_within_tol_pct']}% "
          f"overall={summary['fallback']['overall_pass_count']}/{len(scenarios)}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "meta": {
            "evaluated_function": "parse_diet_log (engines/quicklog/diet_parser.py)",
            "note": "작업서 기재 parse_natural_language_log는 구버전(nlp_engine.py). 현 QuickLog UI는 parse_diet_log 호출.",
            "llm_path": "REAL(gemma4:latest via Ollama, ENABLE_LOCAL_RAW_JSON_INVOKE=true) + deterministic fallback",
            "llm_model": "gemma4:latest",
            "raw_json_invoke": True,
        },
        "summary": summary,
        "cases": [
            {
                "id": sc["id"],
                "category": sc["category"],
                "input": sc["input"],
                "expected": sc["expected"],
                "note": sc.get("note", ""),
                "real": real_results[sc["id"]],
                "fallback": fb_results[sc["id"]],
            }
            for sc in scenarios
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[REPORT] → {REPORT_PATH}")

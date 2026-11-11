"""Stage-instrumented evaluation of the diet-log parser (engines/quicklog/diet_parser.py).

Drives the REAL `parse_diet_log(text)` and captures every intermediate stage by wrapping
module globals / the food-engine singleton (engine code is never edited):

  S1 parse   : items entering `_enrich_macros` (LLM parse after `_reconcile_amounts_with_text`,
               or the deterministic fallback). Checks items found, names, literal amount+unit,
               extra / hallucinated / negated items.
  S2 match   : per item source -> curated table / DB row (food_id of the top candidate the
               engine used) / LLM estimate / nothing. Checked against gold_food_ids.
  S3 grams   : grams used to scale DB/curated macros (n/a for LLM-estimated items).
  S4 kcal    : backend-effective kcal (mirrors DietLogService.resolveKcal) within range.
  total      : case-level total kcal (reported separately; also a bucket when all items pass).

LLM modes
  --llm live   real LLM in-process (llm_router.get_llm), raw responses cached to
               tests/evaluation/llm_cache/diet_llm_cache.json (exceptions cached too)
  --llm cache  replay cache only; texts without an entry are reported as "no_cache"
  --llm none   no LLM.  --none-mode empty (default): LLM "returns no items" -> deterministic
               parse + _enrich_macros (food DB).  --none-mode raise: LLM "is down" -> the
               production exception branch (deterministic parse only, NO enrichment).

Examples (cwd = fit-core-ai):
  ./venv/Scripts/python scripts/eval/diet_stage_eval.py --llm none
  ./venv/Scripts/python scripts/eval/diet_stage_eval.py --llm none --none-mode raise
  ./venv/Scripts/python scripts/eval/diet_stage_eval.py --llm live      # needs Ollama/Gemini
  ./venv/Scripts/python scripts/eval/diet_stage_eval.py --llm cache --split dev
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ---- env must be set before any engines.* import -------------------------------------
os.environ.setdefault("AI_EMBEDDING_DEVICE", "cpu")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if not os.environ.get("FOOD_EMBEDDING_PATH"):
    _snap_root = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "hub" / "models--dragonkue--BGE-m3-ko" / "snapshots"
    _snaps = sorted((p for p in _snap_root.glob("*") if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    if _snaps:
        os.environ["FOOD_EMBEDDING_PATH"] = str(_snaps[0])
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET = ROOT / "tests" / "evaluation" / "datasets" / "diet_eval_v1.jsonl"
ARTIFACTS = ROOT / "tests" / "evaluation" / ".artifacts"
# Committed so any machine can replay the measured LLM outputs (--llm cache).
CACHE_PATH = ROOT / "tests" / "evaluation" / "llm_cache" / "diet_llm_cache.json"
STAGES = ["S1_parse", "S2_match", "S3_grams", "S4_kcal", "total"]
GRAM_UNITS = {"g", "gram", "grams", "그램", "ml", "mL", "밀리리터", "cc"}
LLM_TOKEN = "llm_estimate_ok"
COUNT_UNITS = {"개", "알", "피스", "ea", "pc"}


# =========================================================================================
# helpers
# =========================================================================================

def compact(s) -> str:
    return "".join(str(s or "").split()).lower()


def name_score(pred: str, names: list[str]) -> int:
    p = compact(pred)
    best = 0
    for n in names:
        n = compact(n)
        if not n or not p:
            continue
        if p == n:
            return 3
        if len(n) >= 2 and n in p:
            best = max(best, 2)
        elif len(p) >= 2 and p in n:
            best = max(best, 1)
    return best


def macro_kcal(it: dict):
    p, c, f = it.get("protein_g"), it.get("carbs_g"), it.get("fat_g")
    if p is None and c is None and f is None:
        return None
    return round(4 * (p or 0) + 4 * (c or 0) + 9 * (f or 0), 1)


def effective_kcal(it: dict):
    """Mirror of fit-core-backend DietLogService.resolveKcal (db kcal > 0, else macros 4/4/9, else kcal)."""
    kcal = it.get("kcal")
    if it.get("source") == "db" and kcal is not None and kcal > 0:
        return round(float(kcal), 1)
    mk = macro_kcal(it)
    if mk is not None:
        return mk
    if kcal is not None:
        return round(float(kcal), 1)
    return None


def in_range(v, rng, slack=0.0) -> bool:
    return v is not None and rng[0] - slack <= v <= rng[1] + slack


def prompt_hash(dp) -> str:
    try:
        msgs = dp._build_diet_prompt().messages
        raw = "\n".join(getattr(getattr(m, "prompt", None), "template", str(m)) for m in msgs)
    except Exception as exc:  # pragma: no cover
        raw = repr(exc)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


# =========================================================================================
# instrumentation
# =========================================================================================

class Capture:
    def __init__(self):
        self.reset("")

    def reset(self, text):
        self.text = text
        self.llm_calls = 0
        self.llm_items = None          # items from _parse_payload (raw LLM parse)
        self.reconciled = None         # after _reconcile_amounts_with_text
        self.det_fallback = None       # _deterministic_diet_parse called by parse_diet_log itself
        self.enrich_in = None
        self.enrich_out = None
        self.enrich_started = False
        self.search_log: list[dict] = []
        self._in_reconcile = False
        self.top = None
        self.sanity_log: list[dict] = []


class NoCache(Exception):
    pass


def install(args, cap: Capture, cache: dict):
    import engines.quicklog.diet_parser as dp
    from engines import llm_router
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.runnables import RunnableLambda
    from pydantic import BaseModel

    p_hash = prompt_hash(dp)
    real_get_llm = llm_router.get_llm

    def model_meta():
        try:
            cfg = llm_router.resolve_llm_provider()
            return {"provider": cfg.effective_provider, "model": cfg.model_name,
                    "raw_json": dp._local_raw_json_enabled()}
        except Exception as exc:
            return {"provider": "unknown", "error": repr(exc)}

    def store(entry):
        entry.update({"prompt_hash": p_hash, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **model_meta()})
        cache[cap.text] = entry

    def replay(kind):
        entry = cache.get(cap.text)
        if entry is None:
            raise NoCache(cap.text)
        if entry.get("error"):
            raise RuntimeError(f"cached LLM error: {entry['error']}")
        payload = entry.get("payload")
        if isinstance(payload, str):
            return AIMessage(content=payload)
        return payload

    def call(kind, prompt_value, task, temperature):
        cap.llm_calls += 1
        if args.llm == "none":
            if args.none_mode == "raise":
                raise RuntimeError("LLM disabled (--llm none --none-mode raise)")
            return {"items": []}
        if args.llm == "cache":
            return replay(kind)
        # live
        real = real_get_llm(task, temperature=temperature)
        runnable = real.with_structured_output(dp.ParsedDietLog) if kind == "structured" else real
        try:
            out = runnable.invoke(prompt_value)
        except Exception as exc:
            store({"kind": kind, "error": f"{type(exc).__name__}: {exc}"[:500]})
            raise
        if isinstance(out, BaseMessage):
            payload = out.content if isinstance(out.content, str) else json.dumps(out.content, ensure_ascii=False)
        elif isinstance(out, BaseModel):
            payload = out.model_dump()
        elif isinstance(out, dict):
            payload = out
        else:
            payload = getattr(out, "content", None)
            if not isinstance(payload, str):
                payload = str(out)
        store({"kind": kind, "payload": payload})
        return out

    class FakeLLM(RunnableLambda):
        def __init__(self, task, temperature):
            super().__init__(lambda pv: call("raw", pv, task, temperature))
            self._task, self._temp = task, temperature

        def with_structured_output(self, schema, **kwargs):
            return RunnableLambda(lambda pv: call("structured", pv, self._task, self._temp))

    def fake_get_llm(engine_type, temperature=0, **kwargs):
        return FakeLLM(engine_type, temperature)

    dp.get_llm = fake_get_llm

    # ---- stage wrappers (module globals are looked up at call time) ----
    o_parse_payload = dp._parse_payload
    o_reconcile = dp._reconcile_amounts_with_text
    o_det = dp._deterministic_diet_parse
    o_enrich = dp._enrich_macros

    def w_parse_payload(payload):
        res = o_parse_payload(payload)
        cap.llm_items = [i.model_dump() for i in res.items]
        return res

    def w_reconcile(parsed, user_text):
        cap._in_reconcile = True
        try:
            res = o_reconcile(parsed, user_text)
        finally:
            cap._in_reconcile = False
        cap.reconciled = [i.model_dump() for i in res.items]
        return res

    def w_det(user_text):
        res = o_det(user_text)
        if not cap._in_reconcile:
            cap.det_fallback = [i.model_dump() for i in res.items]
        return res

    def w_enrich(parsed):
        cap.enrich_started = True
        cap.enrich_in = [i.model_dump() for i in parsed.items]
        res = o_enrich(parsed)
        cap.enrich_out = [i.model_dump() for i in res.items]
        return res

    dp._parse_payload = w_parse_payload
    dp._reconcile_amounts_with_text = w_reconcile
    dp._deterministic_diet_parse = w_det
    dp._enrich_macros = w_enrich

    o_sanity = dp._sanity_check_llm_macros

    def w_sanity(item):
        res = o_sanity(item)
        keys = ("protein_g", "carbs_g", "fat_g", "kcal")
        before = {k: getattr(item, k) for k in keys}
        after = {k: getattr(res, k) for k in keys}
        if before == after:
            outcome = "kept" if any(v is not None for v in before.values()) else "no_llm_macros"
        elif all(v is None for v in after.values()):
            outcome = "dropped"
        else:
            outcome = "rescaled"
        cap.sanity_log.append({"food_name": item.food_name, "outcome": outcome, "before": before, "after": after})
        return res

    dp._sanity_check_llm_macros = w_sanity

    # ---- food engine ----
    engine = None
    threshold = None
    need_engine = not (args.llm == "none" and args.none_mode == "raise")
    if args.no_food_engine or not need_engine:
        dp.get_food_engine = lambda: None
    else:
        from engines.food import food_engine as fe
        threshold = fe.SIMILARITY_THRESHOLD
        t0 = time.time()
        engine = fe.get_food_engine()
        print(f"[harness] food engine available={engine.available} load={time.time() - t0:.1f}s "
              f"embedding={os.environ.get('FOOD_EMBEDDING_PATH')}", flush=True)
        o_prefer = fe.FoodSearchEngine.__dict__["_prefer_basis_unit"].__func__

        def w_prefer(candidates, unit):
            out = o_prefer(candidates, unit)
            if out:
                c = out[0]
                meta = dict(c.document.metadata or {})
                cap.top = {
                    "food_id": meta.get("food_id"), "name": meta.get("name") or meta.get("rep_name"),
                    "basis_unit": meta.get("basis_unit"), "kcal_per_basis": meta.get("kcal"),
                    "similarity": round(1.0 - float(c.vector_score), 3),
                    "source_query": c.source_query,
                    "from_canonical": c.source_query_index < 0,
                }
            return out

        fe.FoodSearchEngine._prefer_basis_unit = staticmethod(w_prefer)
        o_search = engine.search

        from engines.food.food_query_normalization import build_food_search_queries

        def w_search(food_name, amount=None, unit=None):
            cap.top = None
            try:
                queries = list(build_food_search_queries(food_name))
            except Exception:
                queries = None
            try:
                res = o_search(food_name, amount, unit)
            except Exception as exc:
                cap.search_log.append({"query": food_name, "amount": amount, "unit": unit,
                                       "outcome": f"exception:{type(exc).__name__}", "top": cap.top,
                                       "resolved_grams": None, "matched_name": None})
                raise
            top = cap.top
            if res is not None:
                outcome = "hit"
            elif top is None:
                outcome = "no_search_queries" if not queries else "no_candidates_after_rerank"
            elif top["similarity"] < threshold:
                outcome = "below_threshold"
            else:
                outcome = "empty_macros"
            cap.search_log.append({
                "query": food_name, "amount": amount, "unit": unit, "outcome": outcome, "top": top,
                "search_queries": queries,
                "resolved_grams": (res or {}).get("resolved_grams"),
                "matched_name": (res or {}).get("matched_name"),
            })
            return res

        engine.search = w_search
    return dp, engine, threshold


# =========================================================================================
# per-case evaluation
# =========================================================================================

def attribute_sources(dp, cap: Capture, engine, final_items: list[dict]) -> list[dict]:
    """Per final item: source (curated|db|llm|none), food_id, grams used for scaling, search record."""
    from engines.food.food_units import resolve_portion_grams

    infos = []
    engine_on = engine is not None and getattr(engine, "available", False)
    if cap.enrich_out is None:
        # exception branch: deterministic output, macros only from the curated table
        for it in final_items:
            name = (it.get("food_name") or "").strip()
            has = any(it.get(k) is not None for k in ("protein_g", "carbs_g", "fat_g"))
            if has and name in dp._FOOD_PER_100G:
                infos.append({"source": "curated", "food_id": None,
                              "grams": resolve_portion_grams(name, it.get("amount"), it.get("unit")), "search": None})
            else:
                infos.append({"source": "none", "food_id": None, "grams": None, "search": None})
        return infos
    log = list(cap.search_log)
    sanity = list(cap.sanity_log)

    def pop_sanity(name):
        for k, rec in enumerate(sanity):
            if (rec["food_name"] or "").strip() == name:
                return sanity.pop(k)["outcome"]
        return None

    for before, it in zip(cap.enrich_in, final_items):
        name = (before.get("food_name") or "").strip()
        if name in dp._FOOD_PER_100G and dp._macros_from_table(name, before.get("amount"), before.get("unit"))["protein_g"] is not None:
            infos.append({"source": "curated", "food_id": None,
                          "grams": resolve_portion_grams(name, before.get("amount"), before.get("unit")), "search": None})
            continue
        rec = log.pop(0) if engine_on and log else None
        if rec is not None and rec["outcome"] == "hit":
            infos.append({"source": "db", "food_id": (rec["top"] or {}).get("food_id"),
                          "grams": rec["resolved_grams"], "search": rec})
            continue
        has = any(it.get(k) is not None for k in ("protein_g", "carbs_g", "fat_g", "kcal"))
        if has and name in dp._FOOD_PER_100G and dp._macros_from_table(name, it.get("amount"), it.get("unit"))["protein_g"] is not None:
            src = "curated"
        else:
            src = "llm" if has else "none"
        infos.append({"source": src, "food_id": None,
                      "grams": resolve_portion_grams(name, it.get("amount"), it.get("unit")) if src == "curated" else None,
                      "search": rec, "sanity": pop_sanity(name)})
    return infos


def align(expected: list[dict], parsed: list[dict]):
    pairs = []
    for ei, e in enumerate(expected):
        for pi, p in enumerate(parsed):
            s = name_score(p.get("food_name"), e["names"])
            if s:
                pairs.append((-s, ei, pi))
    pairs.sort()
    e_used, p_used, match = set(), set(), {}
    for _, ei, pi in pairs:
        if ei in e_used or pi in p_used:
            continue
        e_used.add(ei)
        p_used.add(pi)
        match[ei] = pi
    return match, [pi for pi in range(len(parsed)) if pi not in p_used]


def check_amount(e: dict, p: dict):
    if e["amount"] is None:
        return True, "no_literal_amount"
    amt, unit = p.get("amount"), (p.get("unit") or "").strip()
    if amt is None:
        return False, "amount_missing"
    lo, hi = e["amount"]
    units_ok = {u.lower() for u in e["units"]}
    eps = 0.02 * max(1.0, abs(hi))
    if unit.lower() in units_ok and lo - eps <= float(amt) <= hi + eps:
        return True, "literal"
    if not unit and units_ok & COUNT_UNITS and lo - eps <= float(amt) <= hi + eps:
        return True, "bare_count"
    if unit in GRAM_UNITS or unit.lower() in GRAM_UNITS:
        if e["grams"][0] * 0.95 <= float(amt) <= e["grams"][1] * 1.05:
            return True, "converted_to_grams"
    return False, f"amount_mismatch(expected {lo:g}-{hi:g} {'/'.join(e['units'])}, got {amt} {unit or '-'})"


NAME_RESIDUAL_MAX = 6


def check_name(e: dict, p: dict, others: list[dict]):
    """Name acceptable = matches an accepted name and does not swallow other foods / long chatter."""
    pred = compact(p.get("food_name"))
    residual = pred
    for n in sorted((compact(x) for x in e["names"]), key=len, reverse=True):
        if n and n in residual:
            residual = residual.replace(n, "", 1)
            break
    for o in others:
        for n in o["names"]:
            n = compact(n)
            hit = (n in residual) if len(n) >= 2 else (residual.startswith(n) or residual.endswith(n))
            if n and hit:
                return False, f"merged_items(name '{p.get('food_name')}' also contains '{n}')"
    if len(residual) > NAME_RESIDUAL_MAX:
        return False, f"noisy_name('{p.get('food_name')}')"
    return True, ""


def evaluate_case(case: dict, parsed: list[dict], final: list[dict], infos: list[dict],
                  llm_items=None, branch: str = "llm") -> dict:
    expected = case["expected_items"]
    match, extra = align(expected, parsed)
    llm_path = branch == "llm" and llm_items is not None
    raw_match = align(expected, llm_items)[0] if llm_path else {}
    items = []
    for ei, e in enumerate(expected):
        row = {"key": e["key"], "optional": e["optional"], "expected_grams": e["grams"], "expected_kcal": e["kcal"],
               "gold": e["gold_food_ids"], "stages": {}, "first_fail": None}
        pi = match.get(ei)
        if pi is None:
            row["pred"] = None
            if not e["optional"]:
                row["stages"]["S1_parse"] = "fail:missing"
                row["first_fail"] = "S1_parse"
                row["s1_cause"] = "llm" if llm_path else "deterministic"
            else:
                row["stages"]["S1_parse"] = "skip:optional_absent"
            items.append(row)
            continue
        p, f, info = parsed[pi], final[pi], infos[pi]
        ek = effective_kcal(f)
        row["pred"] = {"food_name": p.get("food_name"), "amount": p.get("amount"), "unit": p.get("unit"),
                       "final_amount": f.get("amount"), "final_unit": f.get("unit"),
                       "source": info["source"], "food_id": info["food_id"],
                       "matched_name": (info["search"] or {}).get("matched_name"),
                       "search_outcome": (info["search"] or {}).get("outcome"),
                       "top": (info["search"] or {}).get("top"),
                       "grams": info["grams"], "kcal_field": f.get("kcal"), "macro_kcal": macro_kcal(f),
                       "effective_kcal": ek}
        # S1: name quality, then literal amount/unit
        ok, why = check_name(e, p, [x for j, x in enumerate(expected) if j != ei])
        if ok:
            ok, why = check_amount(e, p)
        row["stages"]["S1_parse"] = ("pass:" if ok else "fail:") + why
        if llm_path:
            ri = raw_match.get(ei)
            raw = llm_items[ri] if ri is not None else None
            row["llm_raw"] = None if raw is None else {k: raw.get(k) for k in ("food_name", "amount", "unit", "protein_g", "carbs_g", "fat_g", "kcal")}
            row["reconcile_changed"] = raw is not None and (raw.get("amount"), raw.get("unit")) != (p.get("amount"), p.get("unit"))
            if not ok:
                raw_ok = False
                if raw is not None:
                    raw_ok, _ = check_name(e, raw, [x for j, x in enumerate(expected) if j != ei])
                    if raw_ok:
                        raw_ok, _ = check_amount(e, raw)
                row["s1_cause"] = "reconcile" if raw_ok else "llm"
        elif not ok:
            row["s1_cause"] = "deterministic"
        if not ok:
            row["first_fail"] = "S1_parse"
        if not e["optional"] and row["first_fail"] is None:
            # S2
            gold = set(e["gold_food_ids"])
            src = info["source"]
            if src == "curated":
                # No correct DB row (llm_estimate_ok) -> a curated label value is also a
                # legitimate source; S4 still checks its kcal against the label range.
                ok2, why2 = ("curated" in gold or LLM_TOKEN in gold), "curated"
            elif src == "db":
                ok2, why2 = info["food_id"] in gold, f"db:{info['food_id']}"
            elif src == "llm":
                outcome = (info["search"] or {}).get("outcome") or "no_engine"
                ok2, why2 = LLM_TOKEN in gold, f"llm_estimate(db {outcome}; sanity {info.get('sanity') or '-'})"
            else:
                outcome = (info["search"] or {}).get("outcome") or "no_engine"
                ok2, why2 = False, f"no_macros(db {outcome}; sanity {info.get('sanity') or '-'})"
            row["stages"]["S2_match"] = ("pass:" if ok2 else "fail:") + why2
            if not ok2:
                row["first_fail"] = "S2_match"
        if not e["optional"] and row["first_fail"] is None:
            # S3 (only when grams drive the macros)
            if info["source"] in ("db", "curated"):
                g = info["grams"]
                if g is None:
                    row["stages"]["S3_grams"] = "unknown:grams_unresolved"
                elif in_range(g, e["grams"], slack=0.5):
                    row["stages"]["S3_grams"] = f"pass:{g:g}g"
                else:
                    row["stages"]["S3_grams"] = f"fail:{g:g}g not in {e['grams']}"
                    row["first_fail"] = "S3_grams"
            else:
                row["stages"]["S3_grams"] = "n/a:llm_estimate"
        if row["first_fail"] is None:
            ok4 = in_range(ek, e["kcal"])
            row["stages"]["S4_kcal"] = ("pass:" if ok4 else "fail:") + (f"{ek}kcal" if ek is not None else "no_kcal") + f" vs {e['kcal']}" \
                + (f" (llm macros, sanity {info.get('sanity') or '-'})" if info["source"] == "llm" else "")
            if not ok4:
                row["first_fail"] = "S4_kcal"
        items.append(row)

    extra_items = [{"food_name": parsed[pi].get("food_name"), "amount": parsed[pi].get("amount"),
                    "unit": parsed[pi].get("unit"), "effective_kcal": effective_kcal(final[pi])} for pi in extra]
    cnt = case["expected_item_count"]
    case_s1 = []
    if extra_items:
        case_s1.append("extra_items:" + ("llm" if llm_path else "deterministic"))
    if len(parsed) > cnt["max"]:
        case_s1.append(f"too_many({len(parsed)}>{cnt['max']})")
    total = round(sum((effective_kcal(f) or 0.0) for f in final), 1)
    total_ok = in_range(total, case["total_kcal"])
    order = {s: i for i, s in enumerate(STAGES)}
    fails = [r["first_fail"] for r in items if r["first_fail"]]
    if case_s1:
        fails.append("S1_parse")
    first = min(fails, key=order.get) if fails else ("total" if not total_ok else None)
    return {"items": items, "extra_items": extra_items, "case_s1": case_s1, "total_kcal": total,
            "total_ok": total_ok, "first_fail": first}


# =========================================================================================
# reporting
# =========================================================================================

def summarize(results: list[dict]) -> dict:
    def block(rs):
        n = len(rs)
        scored = [r for r in rs if r["status"] == "scored"]
        passed = sum(1 for r in scored if r["first_fail"] is None)
        ff = Counter(r["first_fail"] or "pass" for r in scored)
        ff.update(r["status"] for r in rs if r["status"] != "scored")
        tot = sum(1 for r in scored if r["total_ok"])
        return {"n": n, "scored": len(scored), "pass": passed,
                "acc": round(passed / len(scored), 3) if scored else None,
                "total_kcal_acc": round(tot / len(scored), 3) if scored else None,
                "first_fail": dict(sorted(ff.items()))}

    out = {"overall": block(results), "by_split": {}, "by_tag": {}, "item_stage": {}, "branches": {}}
    for sp in ("dev", "test"):
        out["by_split"][sp] = block([r for r in results if r["split"] == sp])
    tags = sorted({t for r in results for t in r["tags"]})
    for t in tags:
        out["by_tag"][t] = block([r for r in results if t in r["tags"]])
    # item-level stage pass rates (required items only)
    st = defaultdict(Counter)
    for r in results:
        if r["status"] != "scored":
            continue
        for it in r["eval"]["items"]:
            if it["optional"]:
                continue
            for s, v in it["stages"].items():
                st[s][v.split(":")[0]] += 1
    out["item_stage"] = {s: dict(c) for s, c in st.items()}
    causes, reasons = Counter(), Counter()
    for r in results:
        if r["status"] != "scored":
            continue
        for it in r["eval"]["items"]:
            if it.get("s1_cause"):
                causes[it["s1_cause"]] += 1
                reasons[it["stages"]["S1_parse"].split("(")[0]] += 1
        for c in r["eval"]["case_s1"]:
            causes["case:" + c.split("(")[0]] += 1
    out["s1_causes"] = dict(causes)
    out["s1_reasons"] = dict(reasons)
    out["reconcile_changed_items"] = sum(1 for r in results if r["status"] == "scored"
                                         for it in r["eval"]["items"] if it.get("reconcile_changed"))
    s2 = Counter()
    for r in results:
        if r["status"] == "scored":
            for it in r["eval"]["items"]:
                v = it["stages"].get("S2_match", "")
                if v.startswith("fail"):
                    head = v.split(":", 1)[1].split("(")[0]
                    if "(db " in v:
                        head += "(" + v.split("(db ")[1].split(";")[0] + ")"
                    elif head == "db":
                        head = "wrong_db_row"
                    s2[head] += 1
    out["s2_fail_reasons"] = dict(s2)
    out["branches"] = dict(Counter(r.get("branch") for r in results))
    src = Counter()
    for r in results:
        if r["status"] == "scored":
            for it in r["eval"]["items"]:
                if it["pred"]:
                    src[it["pred"]["source"]] += 1
    out["item_sources"] = dict(src)
    return out


def write_md(path: Path, meta: dict, summ: dict, results: list[dict], max_fail: int):
    L = []
    L.append(f"# Diet stage eval — {meta['dataset']}\n")
    L.append(f"- llm: `{meta['llm']}`" + (f" (none-mode `{meta['none_mode']}`)" if meta["llm"] == "none" else ""))
    L.append(f"- food engine: {meta['food_engine']}  | cases: {meta['n_cases']} | elapsed: {meta['elapsed_s']}s")
    L.append(f"- branches: {summ['branches']}  | item sources: {summ.get('item_sources')}\n")
    L.append("## Accuracy\n")
    L.append("| slice | n | scored | pass | acc | total-kcal acc | first failing stage |")
    L.append("|---|---|---|---|---|---|---|")

    def row(name, b):
        ff = ", ".join(f"{k}={v}" for k, v in b["first_fail"].items())
        L.append(f"| {name} | {b['n']} | {b['scored']} | {b['pass']} | {b['acc']} | {b['total_kcal_acc']} | {ff} |")

    row("overall", summ["overall"])
    for sp, b in summ["by_split"].items():
        row(sp, b)
    L.append("\n## Item-level stage outcomes (required items)\n")
    for s, c in summ["item_stage"].items():
        L.append(f"- {s}: {c}")
    L.append(f"- S1 failure causes (llm / reconcile / deterministic; case-level extras): {summ['s1_causes']}")
    L.append(f"- S1 failure reasons: {summ['s1_reasons']}")
    L.append(f"- items whose amount/unit were changed by _reconcile_amounts_with_text: {summ['reconcile_changed_items']}")
    L.append(f"- S2 failure reasons: {summ['s2_fail_reasons']}")
    L.append("- note: S4 kcal ranges are derived from the same gold rows x grams, so S4 can only fail for "
             "LLM-estimated items, rows with inconsistent macros, or kcal leaks; "
             "S4=0 does not by itself prove kcal is right beyond S2+S3.")
    L.append("\n## By tag\n")
    L.append("| tag | n | acc | total-kcal acc | first failing stage |")
    L.append("|---|---|---|---|---|")
    for t, b in summ["by_tag"].items():
        ff = ", ".join(f"{k}={v}" for k, v in b["first_fail"].items())
        L.append(f"| {t} | {b['n']} | {b['acc']} | {b['total_kcal_acc']} | {ff} |")
    L.append(f"\n## Failures (first {max_fail})\n")
    shown = 0
    for r in results:
        if r["status"] == "scored" and r["first_fail"] is None:
            continue
        if shown >= max_fail:
            break
        shown += 1
        L.append(f"### {r['id']} [{r['split']}] `{r['first_fail'] or r['status']}` — {r['text']}")
        if r["status"] != "scored":
            L.append(f"- {r['status']}: {r.get('error', '')}")
            continue
        ev = r["eval"]
        L.append(f"- branch `{r['branch']}`; parsed: " + ", ".join(
            f"{p.get('food_name')}({p.get('amount')} {p.get('unit') or ''})" for p in r["parsed"]))
        if ev["case_s1"]:
            L.append(f"- case S1: {ev['case_s1']}; extra: {ev['extra_items']}")
        for it in ev["items"]:
            pred = it["pred"]
            desc = "missing" if pred is None else (
                f"`{pred['food_name']}` {pred['amount']} {pred['unit'] or ''} src={pred['source']}"
                f"{'/' + str(pred['food_id']) if pred['food_id'] else ''}"
                f"{' (' + str(pred['matched_name']) + ')' if pred['matched_name'] else ''}"
                f" grams={pred['grams']} kcal={pred['effective_kcal']}")
            extra_info = ""
            if it.get("s1_cause"):
                extra_info += f" [S1 cause: {it['s1_cause']}]"
            if it.get("llm_raw") and it.get("reconcile_changed"):
                extra_info += f" [reconcile changed llm {it['llm_raw']['amount']} {it['llm_raw']['unit']}]"
            L.append(f"  - {it['key']}{' (opt)' if it['optional'] else ''}: {desc} → "
                     + "; ".join(f"{k}={v}" for k, v in it["stages"].items()) + extra_info)
        L.append(f"  - total {ev['total_kcal']} vs {r['total_kcal_expected']} ({'ok' if ev['total_ok'] else 'out'})")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# =========================================================================================
# main
# =========================================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", choices=["live", "cache", "none"], default="cache")
    ap.add_argument("--none-mode", choices=["empty", "raise"], default="empty",
                    help="--llm none: 'empty' = LLM returns no items (deterministic + DB enrich); "
                         "'raise' = LLM down (production exception branch, no enrichment)")
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--split", choices=["all", "dev", "test"], default="all")
    ap.add_argument("--ids", default="", help="comma-separated case ids")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cache-path", default=str(CACHE_PATH))
    ap.add_argument("--no-food-engine", action="store_true", help="disable vector food search (curated table only)")
    ap.add_argument("--out-json", default=str(ARTIFACTS / "diet_stage_report.json"))
    ap.add_argument("--out-md", default=str(ARTIFACTS / "diet_stage_report.md"))
    ap.add_argument("--max-fail-md", type=int, default=400)
    args = ap.parse_args()

    os.chdir(ROOT)
    try:
        from dotenv import load_dotenv  # same as main.py
        load_dotenv(ROOT / ".env")
    except Exception:
        pass
    import logging
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("engines.quicklog.diet_parser").setLevel(logging.ERROR)

    cases = [json.loads(l) for l in Path(args.dataset).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.split != "all":
        cases = [c for c in cases if c["split"] == args.split]
    if args.ids:
        want = {s.strip() for s in args.ids.split(",") if s.strip()}
        cases = [c for c in cases if c["id"] in want]
    if args.limit:
        cases = cases[: args.limit]

    cache_path = Path(args.cache_path)
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    cap = Capture()
    dp, engine, threshold = install(args, cap, cache)
    cur_hash = prompt_hash(dp)

    t_start = time.time()
    results = []
    stale = 0
    for n, case in enumerate(cases, 1):
        text = case["text"]
        base = {"id": case["id"], "split": case["split"], "tags": case["tags"], "text": text,
                "total_kcal_expected": case["total_kcal"], "label_confidence": case["label_confidence"]}
        if args.llm == "cache":
            entry = cache.get(text)
            if entry is None:
                results.append({**base, "status": "no_cache", "first_fail": None, "branch": None})
                continue
            if entry.get("prompt_hash") != cur_hash:
                stale += 1
        cap.reset(text)
        t0 = time.time()
        try:
            final = json.loads(dp.parse_diet_log(text))["items"]
        except Exception as exc:
            results.append({**base, "status": "error", "error": f"{type(exc).__name__}: {exc}", "first_fail": None, "branch": None})
            continue
        if cap.enrich_out is not None:
            parsed = cap.enrich_in
            if cap.det_fallback is not None:
                branch = "llm_empty->deterministic+enrich"
            else:
                branch = "llm"
        elif cap.det_fallback is not None:
            parsed = cap.det_fallback
            branch = "exception->deterministic(no enrich)" + ("[after_llm]" if cap.llm_items else "")
            if cap.enrich_started:
                branch += "[enrich_crashed]"
        else:
            parsed = final
            branch = "empty_input"
        if len(parsed) != len(final):
            parsed = final  # should not happen; keep alignment safe
        infos = attribute_sources(dp, cap, engine, final)
        ev = evaluate_case(case, parsed, final, infos, llm_items=cap.llm_items, branch=branch)
        results.append({**base, "status": "scored", "branch": branch, "first_fail": ev["first_fail"],
                        "total_ok": ev["total_ok"], "llm_items": cap.llm_items, "parsed": parsed, "final": final,
                        "eval": ev, "secs": round(time.time() - t0, 2)})
        if args.llm == "live":
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        if n % 25 == 0:
            print(f"[harness] {n}/{len(cases)} cases, {time.time() - t_start:.1f}s", flush=True)

    elapsed = round(time.time() - t_start, 1)
    summ = summarize(results)
    meta = {"dataset": str(Path(args.dataset).relative_to(ROOT)) if Path(args.dataset).is_relative_to(ROOT) else args.dataset,
            "llm": args.llm, "none_mode": args.none_mode,
            "food_engine": ("disabled" if engine is None else f"available={engine.available} thr={threshold}"),
            "n_cases": len(cases), "elapsed_s": elapsed, "prompt_hash": cur_hash, "stale_cache_entries": stale,
            "split": args.split, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out_json, out_md = Path(args.out_json), Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"meta": meta, "summary": summ, "cases": results}, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    write_md(out_md, meta, summ, results, args.max_fail_md)

    o = summ["overall"]
    print(f"[harness] done in {elapsed}s  llm={args.llm}{'/' + args.none_mode if args.llm == 'none' else ''}  "
          f"acc={o['acc']} ({o['pass']}/{o['scored']})  total-kcal acc={o['total_kcal_acc']}")
    for sp, b in summ["by_split"].items():
        print(f"  {sp}: acc={b['acc']} ({b['pass']}/{b['scored']}) first_fail={b['first_fail']}")
    print(f"  overall first_fail={o['first_fail']}")
    print(f"  branches={summ['branches']} item_sources={summ['item_sources']}")
    if stale:
        print(f"  WARNING: {stale} cache entries were recorded with a different prompt hash")
    print(f"  report: {out_json}\n          {out_md}")


if __name__ == "__main__":
    main()

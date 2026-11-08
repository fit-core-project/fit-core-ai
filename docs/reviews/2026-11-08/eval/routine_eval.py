"""Live routine-generation eval against the local AI server (gemma4).

Scores hard constraints programmatically and dumps full responses for manual reading.
Run with the fit-core-ai venv python, cwd = fit-core-ai.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, ".")
from database import SessionLocal  # noqa: E402
from sqlalchemy import text  # noqa: E402

OUT = Path(sys.argv[1])
REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
URL = "http://127.0.0.1:8000/api/ai/generate-routine"
DEMO = "70f84fa9-83d4-4584-8d60-9c2f4f006a94"  # profile: strength, lower-back pain, has workout history

SPLIT_MUSCLES = {
    "push": {"chest", "front-deltoids", "triceps"},
    "pull": {"upper-back", "biceps", "back-deltoids", "trapezius", "forearm", "lower-back"},
    "legs": {"quadriceps", "hamstring", "gluteal", "calves", "adductor", "abductors"},
    "core": {"abs", "obliques", "lower-back"},
}

CASES = [
    {"id": "R1-push-basic", "req": {"userId": DEMO, "targetSplitLabel": "push", "timeAvailableMin": 60}},
    {"id": "R2-golden-shoulder-pain", "req": {"userId": DEMO, "targetSplitLabel": "push", "targetMuscles": ["chest", "front-deltoids", "triceps"],
        "timeAvailableMin": 70, "painAreas": [{"bodyPart": "front-deltoids"}], "domsData": {"chest": 2, "upper-back": 1},
        "equipment": ["MACHINE"], "goal": "hypertrophy", "userNote": "오늘은 어깨 무리 없이 진행하고 싶음"}},
    {"id": "R3-pull-30min", "req": {"userId": DEMO, "targetSplitLabel": "pull", "timeAvailableMin": 30}},
    {"id": "R4-legs-no-barbell-machine", "req": {"userId": DEMO, "targetSplitLabel": "legs", "timeAvailableMin": 50, "equipment": ["BARBELL", "MACHINE"]}},
    {"id": "R5-legs-doms3-quads", "req": {"userId": DEMO, "targetSplitLabel": "legs", "timeAvailableMin": 60, "domsData": {"quadriceps": 3}}},
    {"id": "R6-bodyweight-only-push", "req": {"userId": DEMO, "targetSplitLabel": "push", "timeAvailableMin": 40,
        "equipment": ["BARBELL", "DUMBBELL", "MACHINE", "CABLE", "SMITH_MACHINE", "PLATE", "BAND", "BALL", "BOX"]}},
    {"id": "R7-custom-arms", "req": {"userId": DEMO, "targetMuscles": ["biceps", "triceps"], "timeAvailableMin": 45}},
    {"id": "R8-tired-core", "req": {"userId": DEMO, "targetSplitLabel": "core", "timeAvailableMin": 20, "readinessLevel": "low"}},
    {"id": "R9-new-user-push", "req": {"userId": "eval-new-user-x", "targetSplitLabel": "push", "timeAvailableMin": 60, "goal": "hypertrophy"}},
    {"id": "R11-custom-arms-label", "req": {"userId": DEMO, "targetSplitLabel": "custom", "targetMuscles": ["biceps", "triceps"], "timeAvailableMin": 45}},
    {"id": "R10-note-knee", "req": {"userId": DEMO, "targetSplitLabel": "legs", "timeAvailableMin": 60, "painAreas": [{"bodyPart": "quadriceps"}],
        "userNote": "무릎이 좀 시큰해서 무릎 부담 적은 걸로"}},
]

ALLOWED_LATIN = {"kg", "rir", "rpe", "lbs", "min", "x"}
CODE_TOKENS = re.compile(r"\b(push|pull|legs|core|doms|DOMS|MACHINE|BARBELL|DUMBBELL|CABLE|BODYWEIGHT|hypertrophy|strength|"
                         r"quadriceps|hamstring|gluteal|front-deltoids|upper-back|triceps|biceps|chest|abs|none|fallback)\b", re.I)


def load_catalog():
    s = SessionLocal()
    rows = s.execute(text("select id,name_kr,name_en,primary_muscle,secondary_muscle,equipment_req,pain_triggers from exercise_tier")).mappings().all()
    return {str(r["id"]): dict(r) for r in rows}


def blocked_by_equipment(raw, blocked):
    eq = {e.strip().upper() for e in str(raw or "").split(",") if e.strip()} - {"BODYWEIGHT"}
    return bool(eq) and eq <= blocked  # truly unusable: every option blocked


def any_equipment_blocked(raw, blocked):
    eq = {e.strip().upper() for e in str(raw or "").split(",") if e.strip()} - {"BODYWEIGHT"}
    return bool(eq & blocked)


def score(case, resp, cat, latency):
    req = case["req"]
    issues, notes = [], []
    blocks = resp.get("routineBlocks") or []
    status = resp.get("generationStatus")
    if status not in ("success", "fallback"):
        issues.append(f"status={status}")
    if resp.get("isFallback"):
        issues.append(f"fallback:{resp.get('statusReasonCode')}")
    targets = set(req.get("targetMuscles") or []) or SPLIT_MUSCLES.get(req.get("targetSplitLabel"), set())
    pains = {p["bodyPart"] for p in req.get("painAreas", [])}
    blocked = {e.upper() for e in req.get("equipment", [])}
    doms3 = {m for m, v in req.get("domsData", {}).items() if v >= 3}
    total_sets = 0
    for b in blocks:
        ex = cat.get(str(b.get("exerciseId")))
        name = b.get("exerciseName")
        if ex is None:
            issues.append(f"unknown exerciseId {b.get('exerciseId')} ({name})")
            continue
        pm = ex["primary_muscle"]
        if targets and pm not in targets:
            issues.append(f"off-target: {name} primary={pm}")
        trig = str(ex.get("pain_triggers") or "").lower()
        if pains and (pm in pains or any(p in trig for p in pains)):
            issues.append(f"pain-violation: {name} (primary={pm}, triggers={trig})")
        if blocked and blocked_by_equipment(ex["equipment_req"], blocked):
            issues.append(f"equipment-violation: {name} needs {ex['equipment_req']}")
        if doms3 and pm in doms3:
            notes.append(f"doms3-muscle used as primary: {name}")
        sets = b.get("prescription") or []
        total_sets += len(sets)
        idx = [s.get("setIndex") for s in sets]
        if idx != list(range(1, len(sets) + 1)):
            issues.append(f"setIndex not 1..n for {name}: {idx}")
        for s in sets:
            r = s.get("targetReps")
            if r is None or r <= 0 or r > 30:
                issues.append(f"odd reps {r} in {name}")
            w = s.get("targetWeightKg")
            if w is not None and (w < 0 or w > 300):
                issues.append(f"odd weight {w} in {name}")
    tet = resp.get("totalEstimatedTime") or 0
    if tet > req["timeAvailableMin"]:
        issues.append(f"over time {tet}>{req['timeAvailableMin']}")
    if tet < req["timeAvailableMin"] * 0.5:
        notes.append(f"under-uses time {tet}/{req['timeAvailableMin']}")
    texts = [resp.get("summaryTitle") or ""] + list(resp.get("rationaleSummary") or []) + list(resp.get("warnings") or [])
    texts += [b.get("exerciseRationale") or "" for b in blocks] + [b.get("exerciseName") or "" for b in blocks]
    leaks = sorted({m.group(0) for t in texts for m in CODE_TOKENS.finditer(t)})
    if leaks:
        issues.append(f"code/english tokens in user text: {leaks}")
    guard = [w for w in resp.get("warnings") or [] if "Guard" in w or "guard" in w]
    if guard:
        issues.append(f"internal guard text leaked into warnings: {guard[:2]}")
    weights = [s.get("targetWeightKg") for b in blocks for s in (b.get("prescription") or [])]
    return {
        "id": case["id"], "latency_s": round(latency, 1), "status": status, "reason": resp.get("statusReasonCode"),
        "n_ex": len(blocks), "total_sets": total_sets, "time": tet,
        "weights_null_ratio": round(sum(w is None for w in weights) / max(len(weights), 1), 2),
        "issues": issues, "notes": notes,
        "exercises": [f"{b.get('exerciseName')}[{(cat.get(str(b.get('exerciseId'))) or {}).get('primary_muscle')}] "
                      + " / ".join(f"{s.get('targetWeightKg')}x{s.get('targetReps')}" for s in b.get("prescription") or [])
                      for b in blocks],
        "summaryTitle": resp.get("summaryTitle"), "rationale": resp.get("rationaleSummary"), "warnings": resp.get("warnings"),
    }


def main():
    cat = load_catalog()
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for rep in range(REPEATS):
        for case in CASES:
            t0 = time.time()
            body = json.dumps(case["req"]).encode()
            try:
                r = urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=300)
                resp = json.loads(r.read())
            except Exception as e:  # noqa: BLE001
                resp = {"generationStatus": f"http_error:{e}"}
            lat = time.time() - t0
            (OUT / f"{case['id']}-r{rep}.json").write_text(json.dumps(resp, ensure_ascii=False, indent=2), encoding="utf-8")
            res = score(case, resp, cat, lat)
            res["rep"] = rep
            results.append(res)
            print(json.dumps({k: res[k] for k in ("id", "rep", "latency_s", "status", "n_ex", "time", "issues")}, ensure_ascii=False), flush=True)
    (OUT / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Stage-instrumented routine generation eval.

Runs every case of tests/evaluation/datasets/routine_eval_v1.jsonl through the real
in-process pipeline (engines.routine_pipeline.generate_smart_routine, real DB catalog,
real candidate pool) and records the FIRST stage at which a routine goes wrong:

  S1 candidates/pool   pool holds every required muscle the catalog allows, no forbidden id
  S2 LLM selection     raw LLM picks before repair (off-candidate, duplicates, coverage, sets)
  S3 repair/validation validate_and_repair output (constraints, coverage, set cap, no reject)
  S4 prescription      weights/reps/sets plausibility, setIndex 1..n, final time usage
  S5 response          status/reason, render-ready fields, Korean user text

Engine code is never edited; intermediate values are captured by patching module
attributes for the duration of each case.

LLM modes:
  --llm none   deterministic stub picker (pool order) - no GPU; S2 is reported as skipped
  --llm cache  replay raw LLM outputs from tests/evaluation/llm_cache/routine_llm_cache.json
               (a miss is a case error; no network call is made)
  --llm live   real get_llm("routine") in process; raw outputs are written to the cache

Dataset: --build-dataset regenerates the jsonl. Expectations are derived from catalog rows
(own SQL) and written policy rules, never from get_candidate_exercises / the ranker.

Run from fit-core-ai/:
  ./venv/Scripts/python scripts/eval/routine_stage_eval.py --llm none
  ./venv/Scripts/python scripts/eval/routine_stage_eval.py --llm live   (GPU / Ollama)
  ./venv/Scripts/python scripts/eval/routine_stage_eval.py --llm cache
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")  # same as main.py: LLM_PROVIDER / ENABLE_LOCAL_RAW_JSON_INVOKE decide the code path
os.environ["ENABLE_FEEDBACK_AWARE_RANKING"] = "false"  # deterministic pool

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

DATASET_PATH = ROOT / "tests" / "evaluation" / "datasets" / "routine_eval_v1.jsonl"
ARTIFACTS = ROOT / "tests" / "evaluation" / ".artifacts"
# Committed so any machine can replay the measured LLM outputs (--llm cache).
CACHE_PATH = ROOT / "tests" / "evaluation" / "llm_cache" / "routine_llm_cache.json"
REPORT_JSON = ARTIFACTS / "routine_stage_report.json"
REPORT_MD = ARTIFACTS / "routine_stage_report.md"

DEMO_USER = "70f84fa9-83d4-4584-8d60-9c2f4f006a94"
NEW_USER = "eval-new-user-0001"
STAGES = ["S1", "S2", "S3", "S4", "S5"]

# ---------------------------------------------------------------------------
# Policy (written here on purpose; not imported from engines)
# ---------------------------------------------------------------------------

# Split label -> target slugs. Mirrors the BE-synced registry table (product policy).
SPLIT_TARGETS: Dict[str, List[str]] = {
    "push": ["chest", "front-deltoids", "triceps"],
    "pull": ["upper-back", "trapezius", "biceps", "forearm", "back-deltoids"],
    "legs": ["quadriceps", "hamstring", "gluteal", "calves", "adductor", "abductors"],
    "upper": ["chest", "upper-back", "trapezius", "front-deltoids", "back-deltoids", "biceps", "triceps"],
    "lower": ["quadriceps", "hamstring", "gluteal", "calves", "adductor", "abductors"],
    "chest": ["chest"],
    "back": ["upper-back", "trapezius", "lower-back"],
    "shoulder": ["front-deltoids", "back-deltoids", "trapezius"],
    "arm": ["biceps", "triceps", "forearm"],
    "core": ["abs", "lower-back", "obliques"],
    "neck": ["neck"],
    "full_body": ["chest", "upper-back", "front-deltoids", "quadriceps", "hamstring", "gluteal", "abs"],
}
# Muscles a split should touch at least once (minor ones only count when picked directly).
MAJOR_MUSCLES = {
    "chest", "front-deltoids", "triceps", "upper-back", "biceps", "back-deltoids",
    "quadriceps", "hamstring", "gluteal", "abs", "obliques",
}
ALL_EQUIPMENT_TOKENS = [
    "BARBELL", "DUMBBELL", "MACHINE", "CABLE", "SMITH_MACHINE", "KETTLEBELL", "PLATE", "LANDMINE",
    "BAND", "BALL", "BOX",
]
NON_LOAD_TOKENS = {"BODYWEIGHT", "BAND", "BALL", "BOX"}
# Ceiling of a working weight relative to the barbell equivalent for the same muscle
# (dumbbell/kettlebell = per hand).
LOAD_CAP = {
    "BARBELL": 1.0, "SMITH_MACHINE": 1.0, "MACHINE": 1.0, "CABLE": 0.75,
    "DUMBBELL": 0.5, "KETTLEBELL": 0.5, "PLATE": 0.4, "LANDMINE": 0.6,
}
# Absolute ceilings used when the user has no reference lift for the muscle.
ABS_CAP = {
    "BARBELL": 300.0, "SMITH_MACHINE": 300.0, "MACHINE": 400.0, "CABLE": 120.0,
    "DUMBBELL": 60.0, "KETTLEBELL": 48.0, "PLATE": 45.0, "LANDMINE": 120.0,
}
SMALL_MUSCLES = {"biceps", "triceps", "forearm", "front-deltoids", "back-deltoids", "neck", "abs", "obliques", "calves", "trapezius"}
# muscle -> (reference big-four lift, share of that lift's 1RM a barbell compound for the muscle can carry)
MUSCLE_REF = {
    "chest": ("bench", 1.0), "triceps": ("bench", 0.9),
    "front-deltoids": ("ohp", 1.0), "back-deltoids": ("ohp", 0.7),
    "trapezius": ("deadlift", 0.8), "upper-back": ("deadlift", 0.65),
    "lower-back": ("deadlift", 1.0), "hamstring": ("deadlift", 0.7),
    "gluteal": ("squat", 1.3), "quadriceps": ("squat", 1.0), "calves": ("squat", 1.0),
    "adductor": ("squat", 0.8), "abductors": ("squat", 0.8),
}
BIG_FOUR_IDS = {"bench": {"30"}, "ohp": {"75"}, "deadlift": {"7"}, "squat": {"98"}}
GOAL_REST_SEC = {"strength": 180, "hypertrophy": 90, "endurance": 60, "fatloss": 60, "recomposition": 90, "generalfitness": 75}
# Plausible rep ranges per goal and movement type (STATIC = seconds).
REP_RANGES = {
    "strength": {"COMPOUND": (1, 6), "ISOLATION": (5, 12)},
    "hypertrophy": {"COMPOUND": (5, 12), "ISOLATION": (8, 20)},
    "endurance": {"COMPOUND": (8, 20), "ISOLATION": (10, 25)},
    "fatloss": {"COMPOUND": (8, 20), "ISOLATION": (10, 25)},
}
MIN_TIME_USAGE = 0.5
RENDER_FIELDS = [
    "routineDraftId", "generationStatus", "statusReasonCode", "isFallback", "totalEstimatedTime",
    "summaryTitle", "rationaleSummary", "warnings", "routineBlocks",
]
_CODE_WORDS = sorted({
    *[m for v in SPLIT_TARGETS.values() for m in v], "knees", "neck", "obliques", "abs",
    "push", "pull", "legs", "upper", "lower", "full_body", "core", "custom",
    "strength", "hypertrophy", "endurance", "fatLoss", "fatloss", "recomposition", "generalFitness",
    "COMPOUND", "ISOLATION", "STATIC", "BODYWEIGHT", *ALL_EQUIPMENT_TOKENS,
    "DOMS", "none", "fallback", "Guard", "schemaError", "emptyCandidate", "readinessLevel",
    "primary_muscles", "movement_type", "exercise_id",
}, key=len, reverse=True)
CODE_TOKEN_RE = re.compile(r"(?<![A-Za-z_-])(" + "|".join(re.escape(w) for w in _CODE_WORDS) + r")(?![A-Za-z_-])", re.I)


def goal_key(goal: Optional[str]) -> str:
    return (goal or "hypertrophy").replace("_", "").lower()


def target_exercise_count(t: int) -> int:
    return 4 if t <= 30 else 5 if t <= 45 else 6 if t <= 60 else 7 if t <= 75 else 8


def max_total_sets(t: int, goal: str) -> int:
    rest = GOAL_REST_SEC.get(goal_key(goal), 90)
    return max(1, int(max(0, t - 5) * 60 / (60 + rest)))


def tokens(raw: Any) -> set:
    return {t.strip().upper() for t in str(raw or "").split(",") if t.strip()}


def pain_tokens(raw: Any) -> set:
    return {t.strip().lower() for t in str(raw or "").split(",") if t.strip()}


def equipment_blocked(raw: Any, blocked: set) -> bool:
    return bool((tokens(raw) - {"BODYWEIGHT"}) & blocked)


def usable_load_tokens(raw: Any, blocked: set) -> List[str]:
    return [t for t in tokens(raw) if t in LOAD_CAP and t not in blocked]


def is_pure_non_load(raw: Any) -> bool:
    return tokens(raw) <= NON_LOAD_TOKENS


def e1rm(weight: float, reps: float) -> float:
    return weight if reps <= 1 else weight * (1 + reps / 30.0)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

def load_catalog(db) -> Dict[str, dict]:
    from sqlalchemy import text

    rows = db.execute(text(
        "SELECT id, name_kr, name_en, primary_muscle, secondary_muscle, equipment_req, "
        "movement_type, efficiency_tier, pain_triggers, substitute_exercise_ids FROM exercise_tier"
    )).mappings().all()
    return {str(r["id"]): dict(r) for r in rows}


def catalog_fingerprint(catalog: Dict[str, dict]) -> str:
    rows = sorted(
        (k, v["primary_muscle"], v["equipment_req"], v["pain_triggers"], v["movement_type"])
        for k, v in catalog.items()
    )
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, default=str).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

EQUIP_SETS = {
    "none": [],
    "barbell+machine": ["BARBELL", "MACHINE"],
    "bodyweight-only": list(ALL_EQUIPMENT_TOKENS),
    "no-cable": ["CABLE"],
    "machine": ["MACHINE"],
}

SYNTHETIC_PROFILES = {
    "syn-hyp": {
        "userId": "eval-synthetic-hyp-0001",
        "profile": {
            "goal_type": "hypertrophy", "split_type": "upperLower", "experience_level": "intermediate",
            "strength_baseline": {
                "30": {"exercise_id": "30", "exercise_name_snapshot": "Barbell Bench Press", "weight_kg": 80.0, "reps": 5},
                "75": {"exercise_id": "75", "exercise_name_snapshot": "Barbell Overhead Press", "weight_kg": 50.0, "reps": 5},
                "7": {"exercise_id": "7", "exercise_name_snapshot": "Deadlift", "weight_kg": 140.0, "reps": 5},
                "98": {"exercise_id": "98", "exercise_name_snapshot": "Back Squat", "weight_kg": 100.0, "reps": 5},
            },
            "equipment_access": ["BARBELL", "DUMBBELL", "MACHINE", "CABLE", "BODYWEIGHT"],
            "pain_areas": [],
        },
        "recentSets": [],
    },
    "syn-knee": {
        "userId": "eval-synthetic-knee-0001",
        "profile": {
            "goal_type": "hypertrophy", "split_type": "pushPullLegs", "experience_level": "beginner",
            "strength_baseline": {},
            "equipment_access": ["DUMBBELL", "MACHINE", "CABLE", "BODYWEIGHT"],
            "pain_areas": [{"body_part": "knees"}],
        },
        "recentSets": [],
    },
}

# (id, user, split, targets, time, equip, pains, doms, readiness, goal, note, extra)
CASE_SPECS: List[tuple] = [
    # push
    ("push-demo-60-base", "demo", "push", None, 60, "none", [], {}, "normal", None, None, {}),
    ("push-demo-30", "demo", "push", None, 30, "none", [], {}, "normal", "strength", None, {}),
    ("push-demo-90-high-hyp", "demo", "push", None, 90, "none", [], {}, "high", "hypertrophy", None, {}),
    ("push-demo-45-bm-shoulderpain", "demo", "push", None, 45, "barbell+machine", ["front-deltoids"], {}, "normal", "hypertrophy", "어깨가 불편해요", {}),
    ("push-demo-45-bodyweight", "demo", "push", None, 45, "bodyweight-only", [], {}, "normal", "hypertrophy", "집에서 맨몸으로만", {}),
    ("push-new-60-hyp", "new", "push", None, 60, "none", [], {}, "normal", "hypertrophy", None, {}),
    ("push-new-20-low-endurance-nocable", "new", "push", None, 20, "no-cable", [], {}, "low", "endurance", None, {}),
    ("push-syn-60-doms3-triceps", "syn-hyp", "push", None, 60, "none", [], {"triceps": 3}, "normal", "hypertrophy", None, {}),
    ("push-demo-30-nocable-doms1-chest-fatloss", "demo", "push", None, 30, "no-cable", [], {"chest": 1}, "normal", "fatLoss", None, {}),
    ("push-syn-45-bm-doms2-shoulder", "syn-hyp", "push", None, 45, "barbell+machine", [], {"front-deltoids": 2}, "normal", None, None, {}),
    ("push-demo-60-chest-shoulder-pain", "demo", "push", None, 60, "none", ["chest", "front-deltoids"], {}, "normal", None, None, {}),
    # pull
    ("pull-demo-60-strength", "demo", "pull", None, 60, "none", [], {}, "normal", "strength", None, {}),
    ("pull-demo-30", "demo", "pull", None, 30, "none", [], {}, "normal", None, None, {}),
    ("pull-demo-45-bm", "demo", "pull", None, 45, "barbell+machine", [], {}, "normal", "hypertrophy", None, {}),
    ("pull-demo-45-bodyweight", "demo", "pull", None, 45, "bodyweight-only", [], {}, "normal", "hypertrophy", None, {}),
    ("pull-new-60-hyp", "new", "pull", None, 60, "none", [], {}, "normal", "hypertrophy", None, {}),
    ("pull-syn-90-high-doms2-biceps", "syn-hyp", "pull", None, 90, "none", [], {"biceps": 2}, "high", "hypertrophy", None, {}),
    ("pull-demo-20-low", "demo", "pull", None, 20, "none", [], {}, "low", None, "오늘 너무 피곤함", {}),
    ("pull-new-30-nocable-fatloss", "new", "pull", None, 30, "no-cable", [], {}, "normal", "fatLoss", None, {}),
    ("pull-demo-60-doms3-upperback", "demo", "pull", None, 60, "none", [], {"upper-back": 3}, "normal", "hypertrophy", None, {}),
    # legs
    ("legs-demo-60-strength", "demo", "legs", None, 60, "none", [], {}, "normal", "strength", None, {}),
    ("legs-demo-45-bm", "demo", "legs", None, 45, "barbell+machine", [], {}, "normal", None, None, {}),
    ("legs-demo-60-doms3-quads", "demo", "legs", None, 60, "none", [], {"quadriceps": 3}, "normal", "hypertrophy", None, {}),
    ("legs-demo-45-bodyweight", "demo", "legs", None, 45, "bodyweight-only", [], {}, "normal", "hypertrophy", None, {}),
    ("legs-demo-60-quadpain-note", "demo", "legs", None, 60, "none", ["quadriceps"], {}, "normal", None, "무릎이 좀 시큰해서 무릎 부담 적은 걸로", {}),
    ("legs-syn-60-kneepain", "syn-hyp", "legs", None, 60, "none", ["knees"], {}, "normal", "hypertrophy", None, {}),
    ("legs-new-30-hyp", "new", "legs", None, 30, "none", [], {}, "normal", "hypertrophy", None, {}),
    ("legs-synknee-45-profilepain", "syn-knee", "legs", None, 45, "none", [], {}, "normal", None, None, {}),
    ("legs-demo-90-high-doms1-ham", "demo", "legs", None, 90, "none", [], {"hamstring": 1}, "high", "hypertrophy", None, {}),
    ("legs-new-20-low-endurance", "new", "legs", None, 20, "none", [], {}, "low", "endurance", None, {}),
    # core
    ("core-demo-20-low", "demo", "core", None, 20, "none", [], {}, "low", None, None, {}),
    ("core-demo-30", "demo", "core", None, 30, "none", [], {}, "normal", "hypertrophy", None, {}),
    ("core-new-30-bodyweight", "new", "core", None, 30, "bodyweight-only", [], {}, "normal", "hypertrophy", None, {}),
    ("core-syn-45-doms2-abs", "syn-hyp", "core", None, 45, "none", [], {"abs": 2}, "normal", None, None, {}),
    ("core-new-20-lowbackpain", "new", "core", None, 20, "none", ["lower-back"], {}, "normal", "fatLoss", None, {}),
    # upper
    ("upper-demo-60", "demo", "upper", None, 60, "none", [], {}, "normal", None, None, {}),
    ("upper-demo-90-hyp", "demo", "upper", None, 90, "none", [], {}, "normal", "hypertrophy", None, {}),
    ("upper-new-45-nocable", "new", "upper", None, 45, "no-cable", [], {}, "normal", "hypertrophy", None, {}),
    ("upper-syn-30-bm", "syn-hyp", "upper", None, 30, "barbell+machine", [], {}, "normal", None, None, {}),
    ("upper-demo-60-shoulderpain-doms2-chest", "demo", "upper", None, 60, "none", ["front-deltoids"], {"chest": 2}, "normal", "hypertrophy", None, {}),
    ("upper-new-60-bodyweight", "new", "upper", None, 60, "bodyweight-only", [], {}, "high", "endurance", None, {}),
    # lower
    ("lower-demo-45", "demo", "lower", None, 45, "none", [], {}, "normal", None, None, {}),
    ("lower-new-60-bm-fatloss", "new", "lower", None, 60, "barbell+machine", [], {}, "normal", "fatLoss", None, {}),
    ("lower-syn-90-high-strength", "syn-hyp", "lower", None, 90, "none", [], {}, "high", "strength", None, {}),
    ("lower-demo-30-low-doms2-glute", "demo", "lower", None, 30, "none", [], {"gluteal": 2}, "low", "hypertrophy", None, {}),
    # full body
    ("full-demo-60", "demo", "full_body", None, 60, "none", [], {}, "normal", None, None, {}),
    ("full-new-45-hyp", "new", "full_body", None, 45, "none", [], {}, "normal", "hypertrophy", None, {}),
    ("full-syn-90-hyp", "syn-hyp", "full_body", None, 90, "none", [], {}, "normal", "hypertrophy", None, {}),
    ("full-demo-30-bodyweight", "demo", "full_body", None, 30, "bodyweight-only", [], {}, "normal", "hypertrophy", None, {}),
    ("full-new-60-nocable-kneepain", "new", "full_body", None, 60, "no-cable", ["knees"], {}, "normal", "hypertrophy", None, {}),
    # accessory split labels
    ("arm-demo-45", "demo", "arm", None, 45, "none", [], {}, "normal", "hypertrophy", None, {}),
    ("shoulder-new-30-shoulderpain", "new", "shoulder", None, 30, "none", ["front-deltoids"], {}, "normal", "hypertrophy", None, {}),
    # custom
    ("custom-demo-45-bi-tri", "demo", "custom", ["biceps", "triceps"], 45, "none", [], {}, "normal", None, None, {}),
    ("custom-demo-20-biceps", "demo", "custom", ["biceps"], 20, "none", [], {}, "normal", "hypertrophy", None, {}),
    ("custom-new-30-chest", "new", "custom", ["chest"], 30, "none", [], {}, "normal", "hypertrophy", None, {}),
    ("custom-syn-45-glute-ham", "syn-hyp", "custom", ["gluteal", "hamstring"], 45, "none", [], {}, "normal", None, None, {}),
    ("custom-demo-30-bm-delts", "demo", "custom", ["front-deltoids", "back-deltoids"], 30, "barbell+machine", [], {}, "normal", "hypertrophy", None, {}),
    ("custom-new-20-bodyweight-calves", "new", "custom", ["calves"], 20, "bodyweight-only", [], {}, "normal", "endurance", None, {}),
    ("custom-demo-30-nocable-bi-forearm", "demo", "custom", ["biceps", "forearm"], 30, "no-cable", [], {}, "normal", "hypertrophy", None, {}),
    ("custom-new-30-high-abs-obliques", "new", "custom", ["abs", "obliques"], 30, "none", [], {}, "high", "hypertrophy", None, {}),
    ("custom-demo-45-quads-kneepain", "demo", "custom", ["quadriceps"], 45, "none", ["knees"], {}, "normal", "hypertrophy", None, {}),
    ("custom-nolabel-new-45-chest-back", "new", None, ["chest", "upper-back"], 45, "none", [], {}, "normal", "hypertrophy", None, {}),
    # expected empty (filters leave nothing)
    ("empty-demo-30-lowerback-profilepain", "demo", "custom", ["lower-back"], 30, "none", [], {}, "normal", None, None, {}),
    ("empty-demo-45-chest-doms3", "demo", "custom", ["chest"], 45, "none", [], {"chest": 3}, "normal", "hypertrophy", None, {}),
    ("empty-new-20-neck-neckpain", "new", "custom", ["neck"], 20, "none", ["neck"], {}, "normal", "hypertrophy", None, {}),
    ("empty-new-30-custom-no-targets", "new", "custom", [], 30, "none", [], {}, "normal", "hypertrophy", None, {}),
    # stub faults: exercise the repair / fallback path without an LLM (ignored in live/cache)
    ("fault-demo-60-push-offcandidate", "demo", "push", None, 60, "none", [], {}, "normal", None, None, {"stubFault": "offCandidate"}),
    ("fault-new-45-pull-duplicate", "new", "pull", None, 45, "none", [], {}, "normal", "hypertrophy", None, {"stubFault": "duplicate"}),
    ("fault-new-45-custom-empty-exercises", "new", "custom", ["biceps", "triceps"], 45, "none", [], {}, "normal", "hypertrophy", None, {"stubFault": "emptyExercises"}),
    ("fault-demo-30-legs-badjson", "demo", "legs", None, 30, "none", [], {}, "normal", "hypertrophy", None, {"stubFault": "badJson"}),
]


def _ref_lifts_from_baseline(baseline: Dict[str, Any]) -> Dict[str, float]:
    refs: Dict[str, float] = {}
    for lift, ids in BIG_FOUR_IDS.items():
        for key, value in (baseline or {}).items():
            if not isinstance(value, dict):
                continue
            vid = str(value.get("exercise_id") or key)
            if vid in ids and value.get("weight_kg") and value.get("reps"):
                refs[lift] = round(e1rm(float(value["weight_kg"]), int(value["reps"])), 1)
    return refs


def _history_e1rm(recent_sets: List[dict]) -> Dict[str, float]:
    best: Dict[str, float] = {}
    for record in recent_sets:
        eid, w = record.get("exercise_id"), record.get("weight_kg")
        if not eid or not w:
            continue
        rir = 0.0 if record.get("is_failure") else float(record.get("rir") or 0.0)
        best[str(eid)] = round(max(best.get(str(eid), 0.0), e1rm(float(w), record["reps"] + rir)), 1)
    return best


def _history_by_muscle(recent_sets: List[dict]) -> Dict[str, dict]:
    """Heaviest recent working set per primary muscle (soft check for borrowed loads)."""
    best: Dict[str, dict] = {}
    for record in recent_sets:
        muscle, w = record.get("primary_muscle"), record.get("weight_kg")
        if not muscle or not w:
            continue
        if w > best.get(muscle, {}).get("weightKg", 0):
            best[muscle] = {"weightKg": float(w), "reps": record["reps"], "exerciseId": record.get("exercise_id"),
                            "equipment": record.get("equipment_type")}
    return best


def derive_expectations(case: dict, catalog: Dict[str, dict], profile: Optional[dict], recent: List[dict]) -> dict:
    req = case["request"]
    split = (req.get("targetSplitLabel") or "").lower()
    targets = SPLIT_TARGETS.get(split) or list(dict.fromkeys(m.strip() for m in req.get("targetMuscles") or [] if m.strip()))
    pains = {p["bodyPart"].strip().lower() for p in req.get("painAreas") or []}
    profile_pains = {str(p.get("body_part") or p.get("bodyPart") or "").strip().lower() for p in (profile or {}).get("pain_areas") or []}
    effective_pains = sorted((pains | profile_pains) - {""})
    blocked = {e.strip().upper() for e in req.get("equipment") or []}
    doms = req.get("domsData") or {}
    t = req["timeAvailableMin"]
    goal = req.get("goal") or (profile or {}).get("goal_type") or "hypertrophy"
    readiness = (req.get("readinessLevel") or "normal").lower()

    forbidden: Dict[str, List[str]] = {"pain": [], "equipment": [], "doms3": []}
    safe_by_muscle: Dict[str, List[str]] = defaultdict(list)
    for eid, ex in catalog.items():
        primary = ex["primary_muscle"]
        if primary not in targets:
            continue
        bad = False
        if pain_tokens(ex["pain_triggers"]) & set(effective_pains):
            forbidden["pain"].append(eid)
            bad = True
        if equipment_blocked(ex["equipment_req"], blocked):
            forbidden["equipment"].append(eid)
            bad = True
        if doms.get(primary, 0) >= 3:
            forbidden["doms3"].append(eid)
            bad = True
        if not bad:
            safe_by_muscle[primary].append(eid)
    for k in forbidden:
        forbidden[k] = sorted(forbidden[k], key=int)
    forbidden_ids = sorted({i for v in forbidden.values() for i in v}, key=int)

    coverage_pool = [m for m in targets if m in MAJOR_MUSCLES] if len(targets) > 3 else list(targets)
    available = [m for m in coverage_pool if doms.get(m, 0) < 3 and safe_by_muscle.get(m) and m not in effective_pains]
    uncertain: List[str] = []
    budget = min(target_exercise_count(t), max(1, max_total_sets(t, goal) // 2))
    if t < 30:
        required, min_distinct = [], min(1, len(available))
    elif len(available) <= budget:
        required, min_distinct = list(available), len(available)
    else:
        required, min_distinct = [], budget
        uncertain.append(f"{len(available)} coverage muscles but budget {budget} exercises: only minDistinct={budget} required")
    if any(doms.get(m, 0) == 2 for m in required):
        uncertain.append("DOMS-2 muscle still required (policy: light 1-2 sets allowed)")
    if goal_key(goal) == "endurance":
        uncertain.append("goal 'endurance' is not a BE GoalType value")
    if any(doms.get(m, 0) >= 3 for m in doms if m not in targets):
        uncertain.append("DOMS-3 on a non-target muscle: secondary-muscle DOMS is not enforced")

    total_safe = sum(len(v) for v in safe_by_muscle.values())
    expect_empty = total_safe == 0
    min_time_usage = 0.0 if readiness == "low" else MIN_TIME_USAGE
    if 0 < total_safe < 3 and min_time_usage:
        min_time_usage = 0.0
        uncertain.append(f"only {total_safe} safe catalog exercises: time-usage floor dropped")
    exp = {
        "targetMuscles": targets,
        "allowedPrimaryMuscles": [m for m in targets if doms.get(m, 0) < 3],
        "effectivePainAreas": effective_pains,
        "blockedEquipment": sorted(blocked),
        "forbiddenExerciseIds": forbidden_ids,
        "forbiddenReasons": forbidden,
        "safeCatalogCountByMuscle": {m: len(v) for m, v in sorted(safe_by_muscle.items())},
        "poolRequiredMuscles": available,
        "requiredMuscles": required,
        "minDistinctRequiredMuscles": min_distinct,
        "doms2Muscles": sorted(m for m, v in doms.items() if v == 2),
        "maxSetsPerDoms2Exercise": 2,
        "effectiveGoal": goal,
        "maxTotalSets": max_total_sets(t, goal),
        "targetExerciseCount": target_exercise_count(t),
        "minTimeUsage": min_time_usage,
        "maxTimeMin": t,
        "expectedStatus": ["failed"] if expect_empty else ["success"],
        "expectedReason": "emptyCandidate" if expect_empty else "none",
        "weightPolicy": {
            "refOneRmByLift": _ref_lifts_from_baseline((profile or {}).get("strength_baseline") or {}),
            "historyE1rmById": _history_e1rm(recent),
            "historyByMuscle": _history_by_muscle(recent),
            "nullAllowedWhenNoReference": True,
        },
        "uncertain": uncertain,
    }
    fault = case.get("harness", {}).get("stubFault")
    if fault == "emptyExercises":
        exp["expectedStatusUnderStubFault"] = ["fallback"]
        exp["expectedReasonUnderStubFault"] = ["schemaError"]
    elif fault == "badJson":
        # Unparseable JSON: an LLM repair to success or a render-ready fallback are both acceptable.
        exp["expectedStatusUnderStubFault"] = ["success", "fallback"]
        exp["expectedReasonUnderStubFault"] = ["none", "schemaError"]
    elif fault:
        exp["expectedStatusUnderStubFault"] = ["success"]
        exp["expectedReasonUnderStubFault"] = ["none"]
    return exp


def stable_split(case_id: str) -> str:
    return "test" if int(hashlib.sha1(case_id.encode()).hexdigest(), 16) % 10 < 3 else "dev"


def build_dataset(db) -> List[dict]:
    from engines.db_queries import get_recent_sets, get_user_profile_context

    catalog = load_catalog(db)
    fingerprint = catalog_fingerprint(catalog)
    demo_profile = get_user_profile_context(db, DEMO_USER)
    demo_recent = [r.model_dump() for r in get_recent_sets(db, DEMO_USER)]
    new_profile = get_user_profile_context(db, NEW_USER)
    if new_profile is not None or get_recent_sets(db, NEW_USER):
        raise SystemExit(f"{NEW_USER} unexpectedly has profile/history")

    cases: List[dict] = []
    golden = json.loads((ROOT.parent / "docs" / "ops" / "golden-examples" / "ai-adapter.request.golden.json").read_text(encoding="utf-8"))
    specs = list(CASE_SPECS)
    for spec in specs:
        cid, user, split, targets, t, equip, pains, doms, readiness, goal, note, extra = spec
        harness: Dict[str, Any] = {"profileSource": "db"}
        if user == "demo":
            user_id, profile, recent = DEMO_USER, demo_profile.model_dump() if demo_profile else None, demo_recent
        elif user == "new":
            user_id, profile, recent = NEW_USER, None, []
        else:
            syn = SYNTHETIC_PROFILES[user]
            user_id, profile, recent = syn["userId"], syn["profile"], syn["recentSets"]
            harness = {"profileSource": "override", "profile": syn["profile"], "recentSets": syn["recentSets"]}
        harness.update(extra)
        request = {
            "userId": user_id,
            "targetSplitLabel": split,
            "targetMuscles": list(targets or []),
            "timeAvailableMin": t,
            "painAreas": [{"bodyPart": p} for p in pains],
            "domsData": dict(doms),
            "equipment": list(EQUIP_SETS[equip]),
            "goal": goal,
            "readinessLevel": readiness,
            "userNote": note,
        }
        tags = [
            f"split:{split or 'none'}" if split != "custom" else "split:custom",
            f"time:{t}", f"equip:{equip}", f"user:{'synthetic' if user.startswith('syn') else user}",
            f"readiness:{readiness}", f"goal:{goal or 'profile-default'}",
            "pain:none" if not pains else "pain:" + "+".join(pains),
            "doms:none" if not doms else "doms:" + "+".join(f"{m}={v}" for m, v in doms.items()),
        ]
        if split == "custom" or split is None:
            tags.append(f"custom-n:{len(targets or [])}")
        if note:
            tags.append("note")
        if extra.get("stubFault"):
            tags.append("stub-fault")
        case = {"id": cid, "split": stable_split(cid), "tags": tags, "request": request, "harness": harness}
        case["expectations"] = derive_expectations(case, catalog, profile, recent)
        if case["expectations"]["expectedStatus"] == ["failed"]:
            tags.append("expect:empty")
        if user == "demo":
            tags.append("profile-pain:lower-back")
        cases.append(case)

    golden_case = {
        "id": "golden-ai-adapter-request", "split": stable_split("golden-ai-adapter-request"),
        "tags": ["golden", "split:push", "time:70", "equip:machine", "user:new", "readiness:normal",
                 "goal:hypertrophy", "pain:front-deltoids", "doms:chest=2+upper-back=1", "note"],
        "request": golden, "harness": {"profileSource": "db"},
    }
    golden_profile = get_user_profile_context(db, golden["userId"])
    golden_case["expectations"] = derive_expectations(
        golden_case, catalog, golden_profile.model_dump() if golden_profile else None,
        [r.model_dump() for r in get_recent_sets(db, golden["userId"])],
    )
    cases.append(golden_case)
    for case in cases:
        case["catalogFingerprint"] = fingerprint
    return cases


# ---------------------------------------------------------------------------
# LLM fakes (stub / cache / live)
# ---------------------------------------------------------------------------

class CacheMiss(RuntimeError):
    pass


class LlmCache:
    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, dict] = {}
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
        self.dirty = False

    def save(self) -> None:
        if self.dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self.path)
            self.dirty = False


def _prompt_text(value: Any) -> str:
    if hasattr(value, "to_string"):
        return value.to_string()
    if isinstance(value, list):
        return "\n".join(str(getattr(m, "content", m)) for m in value)
    return str(value)


def _cache_key(path_kind: str, temperature: float, prompt_text: str) -> str:
    model = f"{os.getenv('LLM_PROVIDER', '')}:{os.getenv('LOCAL_LLM_MODEL', '')}"
    return hashlib.sha256(json.dumps([model, float(temperature), path_kind, prompt_text], ensure_ascii=False).encode()).hexdigest()


def build_stub_output(capture: "CaseCapture") -> dict:
    pool = capture.pool or []
    req = capture.request
    n = target_exercise_count(req.time_available_min)
    picks = pool[:n]
    exercises = [{
        "exercise_id": str(c["id"]),
        "exercise_name": c.get("name_kr") or str(c["id"]),
        "movement_type": c.get("movement_type"),
        "primary_muscles": [c.get("primary_muscle")] if c.get("primary_muscle") else [],
        "equipment_type": c.get("equipment_req"),
        "target_weight_kg": None,
        "target_reps": 10,
        "sets": 3,
        "rest_time_sec": 90,
        "target_rir": 2,
        "exercise_rationale": "목표 부위를 자극하도록 후보 목록에서 고른 운동입니다.",
        "substitution_candidates": [],
    } for c in picks]
    fault = capture.stub_fault
    if fault == "offCandidate" and len(exercises) >= 2:
        exercises[1]["exercise_id"] = "999999"
        exercises[1]["exercise_name"] = "존재하지 않는 운동"
    elif fault == "duplicate" and len(exercises) >= 2:
        exercises[1] = copy.deepcopy(exercises[0])
    elif fault == "emptyExercises":
        exercises = []
    return {
        "total_estimated_time": req.time_available_min,
        "summary_title": "오늘의 맞춤 루틴",
        "rationale_summary": ["목표 근육 위주로 구성했습니다.", "서버가 고른 후보 순서대로 배치했습니다."],
        "exercises": exercises,
        "warnings": [],
    }


class FakeChatModel:
    """Stands in for get_llm(...) output. Must be a Runnable for `prompt | llm`."""

    def __new__(cls, mode: str, temperature: float, capture: "CaseCapture", cache: Optional[LlmCache], real_get_llm):
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda

        def call_raw(prompt_value):
            text_prompt = _prompt_text(prompt_value)
            key = _cache_key("raw", temperature, text_prompt)
            capture.llm_calls.append({"path": "raw", "key": key[:16]})
            if mode == "none":
                if capture.stub_fault == "badJson" and not capture.bad_json_sent:
                    capture.bad_json_sent = True
                    text = '{"summary_title": "오늘의 루틴", "exercises": [ {"exercise_id": '
                else:
                    text = json.dumps(build_stub_output(capture), ensure_ascii=False)
            elif mode == "cache":
                entry = cache.data.get(key) if cache else None
                if entry is None:
                    capture.cache_misses += 1
                    raise CacheMiss(f"cache miss {key[:16]}")
                text = entry["text"]
            else:
                started = time.perf_counter()
                real = real_get_llm("routine", temperature=temperature)
                from engines.routine_pipeline import _extract_llm_message_content

                message = real.invoke(prompt_value)
                text = _extract_llm_message_content(message)
                capture.llm_latency_ms += round((time.perf_counter() - started) * 1000)
                cache.data[key] = {"kind": "raw", "text": text, "caseId": capture.case_id, "savedAt": time.strftime("%Y-%m-%dT%H:%M:%S")}
                cache.dirty = True
            capture.raw_llm_texts.append(text)
            return AIMessage(content=text)

        runnable = RunnableLambda(call_raw)

        def with_structured_output(schema, **kwargs):
            def call_structured(prompt_value):
                from langchain_core.exceptions import OutputParserException
                from engines.llm_parser import extract_raw_llm_output_from_exception

                text_prompt = _prompt_text(prompt_value)
                key = _cache_key("structured", temperature, text_prompt)
                capture.llm_calls.append({"path": "structured", "key": key[:16]})
                if mode == "none":
                    out = schema.model_validate(build_stub_output(capture))
                    capture.raw_llm_texts.append(out.model_dump_json())
                    return out
                if mode == "cache":
                    entry = cache.data.get(key) if cache else None
                    if entry is None:
                        capture.cache_misses += 1
                        raise CacheMiss(f"cache miss {key[:16]}")
                else:
                    real = real_get_llm("routine", temperature=temperature)
                    started = time.perf_counter()
                    try:
                        out = real.with_structured_output(schema, **kwargs).invoke(prompt_value)
                        entry = {"kind": "structured", "text": out.model_dump_json()}
                    except (OutputParserException, ValueError) as exc:
                        raw, _src = extract_raw_llm_output_from_exception(exc)
                        entry = {"kind": "error", "error": type(exc).__name__, "text": raw}
                    finally:
                        capture.llm_latency_ms += round((time.perf_counter() - started) * 1000)
                    entry.update(caseId=capture.case_id, savedAt=time.strftime("%Y-%m-%dT%H:%M:%S"))
                    cache.data[key] = entry
                    cache.dirty = True
                capture.raw_llm_texts.append(entry.get("text") or "")
                if entry["kind"] == "error":
                    raise OutputParserException("cached structured-output failure", llm_output=entry.get("text") or "")
                return schema.model_validate_json(entry["text"])

            return RunnableLambda(call_structured)

        runnable.with_structured_output = with_structured_output  # type: ignore[attr-defined]
        return runnable


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

class CaseCapture:
    def __init__(self, case_id: str, request, stub_fault: Optional[str]):
        self.case_id = case_id
        self.request = request
        self.stub_fault = stub_fault
        self.bad_json_sent = False
        self.raw_candidates: Optional[List[dict]] = None
        self.pool: Optional[List[dict]] = None
        self.llm_calls: List[dict] = []
        self.raw_llm_texts: List[str] = []
        self.normalize_events: List[dict] = []
        self.validate_inputs: List[Any] = []
        self.validate_outputs: List[tuple] = []
        self.deterministic_outputs: List[Any] = []
        self.fallback_calls: List[dict] = []
        self.cache_misses = 0
        self.llm_latency_ms = 0
        self.logs: List[str] = []


class _ListHandler(logging.Handler):
    def __init__(self, sink: List[str]):
        super().__init__(level=logging.INFO)
        self.sink = sink

    def emit(self, record):
        if len(self.sink) < 80:
            self.sink.append(f"{record.levelname} {record.name}: {record.getMessage()}"[:300])


def run_case(case: dict, db, mode: str, cache: Optional[LlmCache]) -> dict:
    import engines.llm_parser as llm_parser
    import engines.routine_pipeline as rp
    from engines.db_queries import get_recent_sets, get_user_profile_context
    from engines.schemas import RecentSetRecord, RoutineRequest, UserProfileContext

    req = RoutineRequest.model_validate(case["request"])
    harness = case.get("harness") or {}
    stub_fault = harness.get("stubFault") if mode == "none" else None
    cap = CaseCapture(case["id"], req, stub_fault)

    if harness.get("profileSource") == "override":
        profile = UserProfileContext.model_validate(harness["profile"]) if harness.get("profile") else None
        recent = [RecentSetRecord.model_validate(r) for r in harness.get("recentSets") or []]
    else:
        profile = get_user_profile_context(db, req.user_id) if req.user_id else None
        recent = get_recent_sets(db, req.user_id) if req.user_id else []

    real_get_candidates = rp.get_candidate_exercises
    real_score = rp.score_candidate_exercises
    real_normalize = rp.normalize_llm_response
    real_validate = llm_parser.validate_and_repair_routine_output
    real_deterministic = rp.apply_deterministic_targets
    real_fallback = rp.generate_fallback_routine
    real_get_llm = rp.get_llm

    def w_candidates(*a, **k):
        out = real_get_candidates(*a, **k)
        cap.raw_candidates = copy.deepcopy(out)
        return out

    def w_score(*a, **k):
        out = real_score(*a, **k)
        cap.pool = copy.deepcopy(out)
        return out

    def w_normalize(raw_text, llm=None):
        event = {"raw": (raw_text or "")[:4000], "repairLlm": llm is not None}
        try:
            out = real_normalize(raw_text, llm=llm)
            event["ok"] = True
            event["parsed"] = out.model_dump()
            return out
        except Exception as exc:
            event["ok"] = False
            event["error"] = type(exc).__name__
            raise
        finally:
            cap.normalize_events.append(event)

    def w_validate(llm_output, *a, **k):
        cap.validate_inputs.append(llm_output.model_copy(deep=True))
        out, reason = real_validate(llm_output, *a, **k)
        cap.validate_outputs.append((out.model_copy(deep=True) if out is not None else None, reason))
        return out, reason

    def w_deterministic(*a, **k):
        out = real_deterministic(*a, **k)
        cap.deterministic_outputs.append(out.model_copy(deep=True))
        return out

    def w_fallback(req_, candidates, *a, **k):
        reason = k.get("status_reason_code") or (a[3] if len(a) > 3 else None)
        cap.fallback_calls.append({"reason": reason, "candidateCount": len(candidates or [])})
        return real_fallback(req_, candidates, *a, **k)

    def w_get_llm(engine_type, temperature=0):
        return FakeChatModel(mode, temperature, cap, cache, real_get_llm)

    patches = [
        mock.patch.object(rp, "get_candidate_exercises", w_candidates),
        mock.patch.object(rp, "score_candidate_exercises", w_score),
        mock.patch.object(rp, "normalize_llm_response", w_normalize),
        mock.patch.object(llm_parser, "validate_and_repair_routine_output", w_validate),
        mock.patch.object(rp, "apply_deterministic_targets", w_deterministic),
        mock.patch.object(rp, "generate_fallback_routine", w_fallback),
        mock.patch.object(rp, "get_llm", w_get_llm),
    ]
    if mode == "none":
        patches.append(mock.patch.object(rp, "_should_use_local_raw_json_invoke", lambda: True))

    handler = _ListHandler(cap.logs)
    engines_logger = logging.getLogger("engines")
    engines_logger.addHandler(handler)
    started = time.perf_counter()
    response = None
    error = None
    try:
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            draft = rp.generate_smart_routine(req, db, profile=profile, recent_sets=recent)
        response = draft.model_dump(by_alias=True)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {str(exc)[:300]}"
        try:
            db.rollback()
        except Exception:
            pass
    finally:
        engines_logger.removeHandler(handler)
    latency = round((time.perf_counter() - started) * 1000)
    return {"capture": cap, "response": response, "error": error, "latencyMs": latency, "profile": profile}


# ---------------------------------------------------------------------------
# Stage checks
# ---------------------------------------------------------------------------

def _primary(catalog: Dict[str, dict], eid: str, fallback: Optional[str] = None) -> Optional[str]:
    row = catalog.get(str(eid))
    return row["primary_muscle"] if row else fallback


def _coverage_issues(prefix: str, muscles: List[str], exp: dict) -> List[str]:
    issues = []
    covered = set(muscles)
    required = exp["requiredMuscles"]
    missing = [m for m in required if m not in covered]
    if missing:
        issues.append(f"{prefix}missing required muscles {missing}")
    candidates = exp["poolRequiredMuscles"]
    k = exp["minDistinctRequiredMuscles"]
    if k and len(covered & set(candidates)) < k:
        issues.append(f"{prefix}covers {len(covered & set(candidates))} of required-distinct {k} ({sorted(covered & set(candidates))})")
    return issues


def check_s1(cap: CaseCapture, exp: dict, catalog: Dict[str, dict]) -> dict:
    pool = cap.pool or []
    issues: List[str] = []
    ids = [str(c["id"]) for c in pool]
    forbidden = set(exp["forbiddenExerciseIds"])
    allowed = set(exp["allowedPrimaryMuscles"])
    bad = [i for i in ids if i in forbidden]
    if bad:
        issues.append(f"forbidden ids in pool {bad}")
    off = [f"{i}:{_primary(catalog, i)}" for i in ids if _primary(catalog, i) not in allowed]
    if off:
        issues.append(f"pool primary outside allowed targets {off}")
    pool_muscles = Counter(_primary(catalog, i) for i in ids)
    missing = [m for m in exp["poolRequiredMuscles"] if m not in pool_muscles]
    if missing:
        issues.append(f"pool lacks required muscles the catalog allows {missing}")
    expect_empty = exp["expectedStatus"] == ["failed"]
    if expect_empty and pool:
        issues.append(f"expected empty pool, got {len(pool)}")
    if not expect_empty and not pool:
        issues.append("pool empty although catalog has safe candidates")
    return {
        "status": "fail" if issues else "pass", "issues": issues,
        "rawCandidateCount": len(cap.raw_candidates or []), "poolSize": len(pool),
        "poolMuscles": dict(pool_muscles), "poolIds": ids,
    }


def check_s2(cap: CaseCapture, exp: dict, catalog: Dict[str, dict], mode: str) -> dict:
    out: Dict[str, Any] = {"llmCalls": len(cap.llm_calls), "cacheMisses": cap.cache_misses}
    if exp["expectedStatus"] == ["failed"] and not cap.llm_calls:
        out.update(status="n/a", issues=[])
        return out
    issues: List[str] = []
    picks = cap.validate_inputs[0] if cap.validate_inputs else None
    if cap.cache_misses:
        issues.append(f"llm cache miss x{cap.cache_misses}")
    if picks is None:
        failed_parse = [e.get("error") for e in cap.normalize_events if not e.get("ok")]
        issues.append(f"no usable LLM output (parse errors {failed_parse})" if failed_parse else "no usable LLM output")
    else:
        pool_ids = {str(c["id"]) for c in cap.pool or []}
        ids = [e.exercise_id for e in picks.exercises]
        off = [i for i in ids if i not in pool_ids]
        if off:
            issues.append(f"off-candidate ids {off}")
        dups = [i for i, n in Counter(ids).items() if n > 1]
        if dups:
            issues.append(f"duplicate ids {dups}")
        forb = [i for i in ids if i in set(exp["forbiddenExerciseIds"])]
        if forb:
            issues.append(f"forbidden ids picked {forb}")
        issues += _coverage_issues("raw picks ", [_primary(catalog, i) for i in ids if i in pool_ids], exp)
        total_sets = sum(e.sets for e in picks.exercises)
        if total_sets > exp["maxTotalSets"]:
            issues.append(f"raw total sets {total_sets} > cap {exp['maxTotalSets']}")
        if not picks.exercises:
            issues.append("no exercises")
        elif abs(len(ids) - exp["targetExerciseCount"]) > 2:
            issues.append(f"exercise count {len(ids)} vs target {exp['targetExerciseCount']}")
        out["rawPicks"] = [f"{e.exercise_id}:{_primary(catalog, e.exercise_id, '?')}x{e.sets}" for e in picks.exercises]
        out["rawWeights"] = [e.target_weight_kg for e in picks.exercises]
    if mode == "none":
        out.update(status="skipped", issues=issues, note="stub picker; S2 not scored")
        return out
    out.update(status="fail" if issues else "pass", issues=issues)
    return out


def check_s3(cap: CaseCapture, exp: dict, catalog: Dict[str, dict], expected_status: List[str]) -> dict:
    if exp["expectedStatus"] == ["failed"] and not cap.validate_outputs:
        return {"status": "n/a", "issues": []}
    if not cap.validate_outputs:
        return {"status": "n/a", "issues": [], "note": "validate never ran (LLM stage failed)"}
    validated, reason = cap.validate_outputs[-1]
    if validated is None:
        if "success" not in expected_status:
            return {"status": "pass", "issues": [], "note": f"expected reject under stub fault (reason={reason})"}
        return {"status": "fail", "issues": [f"repair rejected output -> fallback (reason={reason})"], "reason": reason}
    issues: List[str] = []
    pool_ids = {str(c["id"]) for c in cap.pool or []}
    ids = [e.exercise_id for e in validated.exercises]
    off = [i for i in ids if i not in pool_ids]
    if off:
        issues.append(f"ids outside pool after repair {off}")
    dups = [i for i, n in Counter(ids).items() if n > 1]
    if dups:
        issues.append(f"duplicates after repair {dups}")
    forb = [i for i in ids if i in set(exp["forbiddenExerciseIds"])]
    if forb:
        issues.append(f"forbidden ids after repair {forb}")
    muscles = [_primary(catalog, i) for i in ids]
    offt = [f"{i}:{m}" for i, m in zip(ids, muscles) if m not in set(exp["allowedPrimaryMuscles"])]
    if offt:
        issues.append(f"off-target after repair {offt}")
    issues += _coverage_issues("", muscles, exp)
    total_sets = sum(e.sets for e in validated.exercises)
    if total_sets > exp["maxTotalSets"]:
        issues.append(f"total sets {total_sets} > cap {exp['maxTotalSets']}")
    d2 = set(exp["doms2Muscles"])
    over = [f"{e.exercise_id}x{e.sets}" for e, m in zip(validated.exercises, muscles) if m in d2 and e.sets > 2]
    if over:
        issues.append(f"DOMS-2 muscle exercise over 2 sets {over}")
    from engines.prescription.estimator import estimate_routine_time_min

    est = estimate_routine_time_min(validated.exercises)
    return {
        "status": "fail" if issues else "pass", "issues": issues,
        "exercises": [f"{i}:{m}x{e.sets}" for i, m, e in zip(ids, muscles, validated.exercises)],
        "estimatedTimeMin": est, "underUsed": est < exp["maxTimeMin"] * exp["minTimeUsage"],
    }


def weight_bounds(row: dict, reps: int, exp: dict, profile_has_refs: bool) -> dict:
    """Return {'mustBeNull': bool, 'lo': float|None, 'hi': float|None, 'basis': str}."""
    blocked = set(exp["blockedEquipment"])
    if is_pure_non_load(row["equipment_req"]):
        return {"mustBeNull": True, "basis": "non-load equipment"}
    usable = usable_load_tokens(row["equipment_req"], blocked)
    if not usable:
        return {"mustBeNull": False, "lo": None, "hi": None, "basis": "bodyweight variant", "nullOk": True}
    movement = str(row.get("movement_type") or "").upper()
    mv_cap = 1.0 if movement == "COMPOUND" else 0.6
    eff_reps = min(int(reps), 12)
    denom = 1 + eff_reps / 30.0
    primary = row["primary_muscle"]
    refs = exp["weightPolicy"]["refOneRmByLift"]
    history = exp["weightPolicy"]["historyE1rmById"]
    lift = MUSCLE_REF.get(primary)
    eq_hi = max(LOAD_CAP[t] for t in usable)
    eq_lo = min(LOAD_CAP[t] for t in usable)
    if movement == "COMPOUND" and "MACHINE" in usable and primary in {"quadriceps", "gluteal", "calves"}:
        eq_hi = max(eq_hi, 2.0)  # leg press / hack squat stacks exceed the squat
    lo = hi = None
    basis = []
    null_ok = True
    can_bodyweight = "BODYWEIGHT" in tokens(row["equipment_req"])  # e.g. Nordic curl, split squat: null is fine
    if lift and lift[0] in refs:
        base = refs[lift[0]] * lift[1]
        hi = base * eq_hi * mv_cap / denom * 1.10
        lo = 0.15 * base * eq_lo * mv_cap / denom
        basis.append(f"{lift[0]} 1RM {refs[lift[0]]}x{lift[1]}")
        null_ok = can_bodyweight
    hist = history.get(str(row["id"]))
    if hist:
        h = hist / denom
        hi = max(hi or 0.0, h * 1.15)
        lo = min(lo, h * 0.5) if lo is not None else h * 0.5
        basis.append(f"history e1RM {hist}")
        null_ok = can_bodyweight
    if hi is None:
        hi = max(ABS_CAP[t] for t in usable) * (0.5 if movement != "COMPOUND" and primary in SMALL_MUSCLES else 1.0)
        lo = 0.0
        basis.append("absolute cap (no reference)")
    return {"mustBeNull": False, "lo": round(lo, 1), "hi": round(hi, 1), "basis": "; ".join(basis), "nullOk": null_ok}


def check_s4(cap: CaseCapture, exp: dict, catalog: Dict[str, dict], response: Optional[dict], s3: dict) -> dict:
    if not response or not response.get("routineBlocks"):
        return {"status": "n/a", "issues": []}
    issues: List[str] = []
    gk = goal_key(exp["effectiveGoal"])
    ranges = REP_RANGES.get(gk, REP_RANGES["hypertrophy"])
    d2 = set(exp["doms2Muscles"])
    total_sets = 0
    weights = []
    muscles = []
    light_notes: List[str] = []
    for block in response["routineBlocks"]:
        eid = str(block.get("exerciseId"))
        row = catalog.get(eid)
        sets = block.get("prescription") or []
        total_sets += len(sets)
        idx = [s.get("setIndex") for s in sets]
        if idx != list(range(1, len(sets) + 1)):
            issues.append(f"{eid}: setIndex {idx} not 1..n")
        if not sets:
            issues.append(f"{eid}: no sets")
        if len(sets) > 6:
            issues.append(f"{eid}: {len(sets)} sets")
        if row is None:
            issues.append(f"{eid}: not in catalog")
            continue
        muscles.append(row["primary_muscle"])
        if row["primary_muscle"] in d2 and len(sets) > 2:
            issues.append(f"{eid}: DOMS-2 muscle {row['primary_muscle']} has {len(sets)} sets")
        movement = str(row.get("movement_type") or "").upper()
        for s in sets:
            reps = s.get("targetReps")
            w = s.get("targetWeightKg")
            if movement in ranges:
                lo_r, hi_r = ranges[movement]
                if reps is None or not (lo_r <= reps <= hi_r):
                    issues.append(f"{eid}: reps {reps} outside {gk}/{movement} {lo_r}-{hi_r}")
                    break
            elif reps is None or not (1 <= reps <= 90):
                issues.append(f"{eid}: reps {reps} implausible")
                break
        first = sets[0] if sets else {}
        w = first.get("targetWeightKg")
        bounds = weight_bounds(row, first.get("targetReps") or 10, exp, bool(exp["weightPolicy"]["refOneRmByLift"]))
        weights.append(f"{eid}:{row['equipment_req']}:{w}")
        if bounds["mustBeNull"]:
            if w is not None:
                issues.append(f"{eid}: weight {w} on non-load exercise ({row['equipment_req']})")
        elif w is None:
            if not bounds.get("nullOk", True):
                issues.append(f"{eid}: null weight although reference exists ({bounds['basis']})")
        else:
            if bounds["hi"] is not None and w > bounds["hi"]:
                issues.append(f"{eid}: weight {w} > plausible {bounds['hi']} ({row['equipment_req']}, {bounds['basis']})")
            if bounds["lo"] and w < bounds["lo"]:
                issues.append(f"{eid}: weight {w} < plausible {bounds['lo']} ({row['equipment_req']}, {bounds['basis']})")
        same = exp["weightPolicy"].get("historyByMuscle", {}).get(row["primary_muscle"])
        # Only compare loads on a shared implement (history rows can be multi-token, so this is a soft note).
        if (w is not None and same and str(same.get("exerciseId")) != eid
                and (tokens(row["equipment_req"]) & tokens(same.get("equipment")) - NON_LOAD_TOKENS)
                and w < 0.4 * same["weightKg"]):
            light_notes.append(f"{eid}: {w}kg < 40% of same-muscle recent {same['weightKg']}kg (ex {same['exerciseId']}, {same['equipment']})")
        if len({s.get('targetWeightKg') for s in sets}) > 1:
            issues.append(f"{eid}: mixed weights across working sets")
    if total_sets > exp["maxTotalSets"]:
        issues.append(f"final total sets {total_sets} > cap {exp['maxTotalSets']}")
    # Soft observations (reported, not scored).
    notes: List[str] = list(light_notes)
    single = [str(b.get("exerciseId")) for b in response["routineBlocks"]
              if len(b.get("prescription") or []) == 1 and _primary(catalog, b.get("exerciseId")) not in d2]
    if single:
        notes.append(f"single-set exercises {single}")
    if len(muscles) >= 4 and len(exp["poolRequiredMuscles"]) >= 2:
        top, n = Counter(muscles).most_common(1)[0]
        if n / len(muscles) > 0.6:
            notes.append(f"muscle dominance {top} {n}/{len(muscles)} exercises")
    tet = response.get("totalEstimatedTime") or 0
    time_issue = None
    if tet > exp["maxTimeMin"]:
        time_issue = f"final time {tet} > {exp['maxTimeMin']}"
    elif tet < exp["maxTimeMin"] * exp["minTimeUsage"]:
        time_issue = f"final time {tet} < {int(exp['minTimeUsage'] * 100)}% of {exp['maxTimeMin']}"
    # Coverage lost after repair is a prescription-stage regression.
    if s3.get("status") == "pass" and response.get("generationStatus") == "success":
        lost = _coverage_issues("after prescription ", muscles, exp)
        issues += lost
    return {
        "status": "fail" if issues or time_issue else "pass",
        "issues": issues + ([time_issue] if time_issue else []),
        "notes": notes, "timeIssue": time_issue, "totalSets": total_sets, "totalEstimatedTime": tet, "weights": weights,
    }


def _korean_text_issues(response: dict) -> List[str]:
    texts = [("summaryTitle", response.get("summaryTitle") or "")]
    texts += [("rationaleSummary", t) for t in response.get("rationaleSummary") or []]
    texts += [("warnings", t) for t in response.get("warnings") or []]
    for block in response.get("routineBlocks") or []:
        texts.append((f"exerciseRationale[{block.get('exerciseId')}]", block.get("exerciseRationale") or ""))
        for sub in block.get("substitutionCandidates") or []:
            texts.append((f"substitutionReason[{block.get('exerciseId')}]", sub.get("reason") or ""))
    issues = []
    for field, text in texts:
        leaks = sorted({m.group(0) for m in CODE_TOKEN_RE.finditer(text)})
        if leaks:
            issues.append(f"code tokens in {field}: {leaks} :: {text[:80]}")
        hangul = len(re.findall(r"[가-힣]", text))
        latin = len(re.findall(r"[A-Za-z]", text))
        if text.strip() and latin > max(8, hangul):
            issues.append(f"English-heavy {field}: {text[:80]}")
    return issues


def check_s5(response: Optional[dict], exp: dict, mode: str, case: dict, error: Optional[str]) -> dict:
    if error:
        return {"status": "fail", "issues": [f"pipeline exception {error}"]}
    issues: List[str] = []
    fault = (case.get("harness") or {}).get("stubFault") if mode == "none" else None
    want_status = exp.get("expectedStatusUnderStubFault") if fault else exp["expectedStatus"]
    want_reason = exp.get("expectedReasonUnderStubFault") if fault else exp["expectedReason"]
    status = response.get("generationStatus")
    if status not in want_status:
        issues.append(f"status {status}/{response.get('statusReasonCode')} expected {want_status}/{want_reason}")
    elif response.get("statusReasonCode") not in ([want_reason] if isinstance(want_reason, str) else want_reason):
        issues.append(f"reason {response.get('statusReasonCode')} expected {want_reason}")
    missing = [f for f in RENDER_FIELDS if f not in response or response[f] is None]
    if missing:
        issues.append(f"missing render fields {missing}")
    blocks = response.get("routineBlocks") or []
    if status in ("success", "fallback"):
        if not blocks:
            issues.append(f"{status} with empty routineBlocks")
        if not (response.get("summaryTitle") or "").strip() or not response.get("rationaleSummary"):
            issues.append("empty summaryTitle/rationaleSummary")
        if (status == "fallback") != bool(response.get("isFallback")):
            issues.append(f"isFallback={response.get('isFallback')} inconsistent with {status}")
        for b in blocks:
            for f in ("order", "exerciseId", "exerciseName", "defaultRestSec", "prescription", "exerciseRationale"):
                if b.get(f) in (None, "", []):
                    issues.append(f"block {b.get('exerciseId')} missing {f}")
        orders = [b.get("order") for b in blocks]
        if orders != list(range(1, len(blocks) + 1)):
            issues.append(f"block order {orders} not 1..n")
    elif status == "failed" and blocks:
        issues.append("failed with routineBlocks")
    forb = [b.get("exerciseId") for b in blocks if str(b.get("exerciseId")) in set(exp["forbiddenExerciseIds"])]
    if forb:
        issues.append(f"forbidden ids in final response {forb}")
    issues += _korean_text_issues(response)
    return {"status": "fail" if issues else "pass", "issues": issues}


def evaluate_case(case: dict, run: dict, catalog: Dict[str, dict], mode: str) -> dict:
    exp = case["expectations"]
    cap: CaseCapture = run["capture"]
    response = run["response"]
    stages: Dict[str, dict] = {}
    stages["S1"] = check_s1(cap, exp, catalog)
    stages["S2"] = check_s2(cap, exp, catalog, mode)
    fault = (case.get("harness") or {}).get("stubFault") if mode == "none" else None
    expected_status = exp.get("expectedStatusUnderStubFault") if fault else exp["expectedStatus"]
    stages["S3"] = check_s3(cap, exp, catalog, expected_status)
    stages["S4"] = check_s4(cap, exp, catalog, response, stages["S3"])
    stages["S5"] = check_s5(response, exp, mode, case, run["error"])
    # Final time under-use that already existed after repair belongs to S3 (repair did not fill).
    s4 = stages["S4"]
    if s4.get("timeIssue") and "<" in s4["timeIssue"] and stages["S3"].get("underUsed") and stages["S3"]["status"] == "pass":
        stages["S3"]["status"] = "fail"
        stages["S3"]["issues"].append(f"under-used time after repair ({stages['S3']['estimatedTimeMin']} min) and not recovered: {s4['timeIssue']}")
        s4["issues"].remove(s4["timeIssue"])
        if not s4["issues"]:
            s4["status"] = "pass"
    first = next((s for s in STAGES if stages[s]["status"] == "fail"), None)
    if run["error"] and first is None:
        first = "S5"
    return {
        "id": case["id"], "split": case["split"], "tags": case["tags"],
        "passed": first is None, "firstFailingStage": first, "cacheMiss": cap.cache_misses > 0,
        "generationStatus": (response or {}).get("generationStatus"),
        "statusReasonCode": (response or {}).get("statusReasonCode"),
        "fallbackCalls": cap.fallback_calls,
        "latencyMs": run["latencyMs"], "llmLatencyMs": cap.llm_latency_ms,
        "stages": stages,
        "finalBlocks": [
            f"{b.get('exerciseId')}:{_primary(catalog, b.get('exerciseId'), '?')}:{b.get('equipmentType')} "
            + " / ".join(f"{s.get('targetWeightKg')}x{s.get('targetReps')}" for s in b.get("prescription") or [])
            for b in (response or {}).get("routineBlocks") or []
        ],
        "summaryTitle": (response or {}).get("summaryTitle"),
        "warnings": (response or {}).get("warnings"),
        "uncertainExpectations": exp.get("uncertain", []),
        "guardLogs": [l for l in cap.logs if "Guard" in l or "fallback" in l.lower()][:20],
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def summarize(results: List[dict]) -> dict:
    miss_cases = [r["id"] for r in results if r.get("cacheMiss")]
    results = [r for r in results if not r.get("cacheMiss")]

    def rate(rs):
        return {"n": len(rs), "passed": sum(r["passed"] for r in rs),
                "accuracy": round(sum(r["passed"] for r in rs) / len(rs), 3) if rs else None}

    by_split = {s: rate([r for r in results if r["split"] == s]) for s in ("dev", "test")}
    first = Counter(r["firstFailingStage"] or "pass" for r in results)
    stage_fail_any = {s: sum(1 for r in results if r["stages"][s]["status"] == "fail") for s in STAGES}
    def note_kind(n: str) -> str:
        return "light vs same-muscle history" if "same-muscle" in n else " ".join(n.split(" ")[:2])

    notes = Counter(note_kind(n) for r in results for n in r["stages"]["S4"].get("notes") or [])
    tag_stats: Dict[str, dict] = {}
    by_tag: Dict[str, List[dict]] = defaultdict(list)
    for r in results:
        for t in r["tags"]:
            by_tag[t].append(r)
    for t, rs in sorted(by_tag.items()):
        st = rate(rs)
        st["firstFailingStage"] = dict(Counter(r["firstFailingStage"] or "pass" for r in rs))
        tag_stats[t] = st
    return {"overall": rate(results), "bySplit": by_split, "firstFailingStage": dict(first),
            "cacheMissCases": miss_cases, "stageFailAnyCount": stage_fail_any, "softNotes": dict(notes), "byTag": tag_stats}


def write_report(results: List[dict], meta: dict) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    summary = summarize(results)
    REPORT_JSON.write_text(json.dumps({"meta": meta, "summary": summary, "cases": results}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    lines = [f"# Routine stage eval ({meta['llmMode']})", ""]
    lines += [f"- {k}: `{v}`" for k, v in meta.items()]
    o = summary["overall"]
    lines += ["", f"**Overall** {o['passed']}/{o['n']} ({o['accuracy']})", ""]
    lines += ["| split | n | passed | accuracy |", "|---|---|---|---|"]
    for s, v in summary["bySplit"].items():
        lines.append(f"| {s} | {v['n']} | {v['passed']} | {v['accuracy']} |")
    if summary["cacheMissCases"]:
        lines += ["", f"**Excluded (LLM cache miss)**: {len(summary['cacheMissCases'])} cases {summary['cacheMissCases']}"]
    lines += ["", "## First failing stage", "", "| stage | cases | (any-fail count) |", "|---|---|---|"]
    for s in STAGES + ["pass"]:
        lines.append(f"| {s} | {summary['firstFailingStage'].get(s, 0)} | {summary['stageFailAnyCount'].get(s, '')} |")
    lines += ["", "## Soft notes (not scored)", ""]
    for r in results:
        if r["stages"]["S4"].get("notes"):
            lines.append(f"- {r['id']}: " + "; ".join(r["stages"]["S4"]["notes"]))
    lines += ["", "## By tag", "", "| tag | n | passed | first-fail stages |", "|---|---|---|---|"]
    for t, v in summary["byTag"].items():
        ff = ", ".join(f"{k}:{n}" for k, n in sorted(v["firstFailingStage"].items()) if k != "pass")
        lines.append(f"| {t} | {v['n']} | {v['passed']} | {ff} |")
    lines += ["", "## Failures", ""]
    for r in results:
        if r["passed"]:
            continue
        lines.append(f"### {r['id']} ({r['split']}) - first fail {r['firstFailingStage']} - {r['generationStatus']}/{r['statusReasonCode']}")
        for s in STAGES:
            st = r["stages"][s]
            if st.get("issues"):
                tag = st["status"]
                lines += [f"- {s} [{tag}]: " + "; ".join(st["issues"])]
        lines.append(f"- pool: {r['stages']['S1'].get('poolMuscles')}")
        lines.append(f"- final: {r['finalBlocks']}")
        if r["uncertainExpectations"]:
            lines.append(f"- uncertain expectations: {r['uncertainExpectations']}")
        lines.append("")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--llm", choices=["none", "cache", "live"], default="none")
    ap.add_argument("--dataset", type=Path, default=DATASET_PATH)
    ap.add_argument("--cache", type=Path, default=CACHE_PATH)
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all")
    ap.add_argument("--cases", default="", help="comma-separated case ids (substring match)")
    ap.add_argument("--tag", default="", help="only cases carrying this tag")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--build-dataset", action="store_true", help="regenerate the dataset jsonl from the catalog and exit")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, stream=io.StringIO())  # engine logs go to per-case capture only
    logging.getLogger("engines").setLevel(logging.INFO)
    logging.getLogger("engines").propagate = False

    from database import SessionLocal

    db = SessionLocal()
    try:
        if args.build_dataset:
            cases = build_dataset(db)
            args.dataset.parent.mkdir(parents=True, exist_ok=True)
            with args.dataset.open("w", encoding="utf-8", newline="\n") as fh:
                for c in cases:
                    fh.write(json.dumps(c, ensure_ascii=False) + "\n")
            print(f"wrote {len(cases)} cases -> {args.dataset}")
            print("split:", dict(Counter(c["split"] for c in cases)))
            return 0

        cases = [json.loads(l) for l in args.dataset.read_text(encoding="utf-8").splitlines() if l.strip()]
        if args.split != "all":
            cases = [c for c in cases if c["split"] == args.split]
        if args.cases:
            wanted = [w.strip() for w in args.cases.split(",") if w.strip()]
            cases = [c for c in cases if any(w in c["id"] for w in wanted)]
        if args.tag:
            cases = [c for c in cases if args.tag in c["tags"]]
        if args.limit:
            cases = cases[: args.limit]

        catalog = load_catalog(db)
        fp = catalog_fingerprint(catalog)
        drift = sorted({c.get("catalogFingerprint") for c in cases} - {fp})
        if drift:
            print(f"WARNING catalog fingerprint {fp} differs from dataset {drift}; expectations may be stale (rerun --build-dataset)")

        cache = LlmCache(args.cache) if args.llm in ("cache", "live") else None
        from engines.candidate_pool_policy import get_candidate_pool_size
        from engines.llm_router import resolve_llm_provider

        provider = resolve_llm_provider()
        import subprocess

        engine_hash = hashlib.sha256()
        for f in sorted((ROOT / "engines").rglob("*.py")):
            engine_hash.update(f.relative_to(ROOT).as_posix().encode() + f.read_bytes())
        try:
            head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
            dirty = subprocess.run(["git", "status", "--porcelain", "--", "engines"], cwd=ROOT, capture_output=True, text=True).stdout.split()
        except Exception:  # noqa: BLE001
            head, dirty = "unknown", []
        meta = {
            "gitHead": head, "enginesSha256": engine_hash.hexdigest()[:16],
            "enginesDirtyFiles": [d for d in dirty if d.endswith(".py")],
            "llmMode": args.llm, "dataset": str(args.dataset.relative_to(ROOT)) if args.dataset.is_relative_to(ROOT) else str(args.dataset),
            "cases": len(cases), "catalogFingerprint": fp, "catalogRows": len(catalog),
            "llmProvider": provider.effective_provider, "llmModel": provider.model_name,
            "localRawJsonInvoke": os.getenv("ENABLE_LOCAL_RAW_JSON_INVOKE", ""),
            "ROUTINE_CANDIDATE_POOL_SIZE": os.getenv("ROUTINE_CANDIDATE_POOL_SIZE", "(default)") + f" -> {get_candidate_pool_size()}",
            "ENABLE_FEEDBACK_AWARE_RANKING": os.getenv("ENABLE_FEEDBACK_AWARE_RANKING"),
            "startedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        results = []
        for i, case in enumerate(cases, 1):
            run = run_case(case, db, args.llm, cache)
            if cache:
                cache.save()
            res = evaluate_case(case, run, catalog, args.llm)
            results.append(res)
            if not args.quiet:
                stage = res["firstFailingStage"] or "PASS"
                first_issue = ""
                if res["firstFailingStage"]:
                    first_issue = (res["stages"][res["firstFailingStage"]]["issues"] or [""])[0][:140]
                print(f"[{i:02d}/{len(cases)}] {stage:4} {res['id']:<45} {res['generationStatus']}/{res['statusReasonCode']} {first_issue}", flush=True)
        meta["finishedAt"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        write_report(results, meta)
        s = summarize(results)
        print(json.dumps({"overall": s["overall"], "bySplit": s["bySplit"], "firstFailingStage": s["firstFailingStage"]}, ensure_ascii=False))
        print(f"report -> {REPORT_JSON}\n         {REPORT_MD}")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())

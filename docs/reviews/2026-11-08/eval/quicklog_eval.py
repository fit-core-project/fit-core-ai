"""Live quicklog eval: extra hard diet cases (/parse-diet) and workout log cases (/parse-log)."""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

OUT = Path(sys.argv[1])
BASE = "http://127.0.0.1:8000/api/ai"

# kcal ranges are generous public-DB style estimates; the point is gross errors, not decimals.
DIET = [
    {"id": "D1-two-items", "text": "점심에 김치찌개 한 그릇이랑 공기밥 먹었어", "names": [["김치찌개"], ["밥", "공기밥"]], "kcal": (400, 900)},
    {"id": "D2-half-bowl", "text": "밥 반 공기", "names": [["밥"]], "kcal": (100, 180)},
    {"id": "D3-pieces", "text": "치킨 두 조각이랑 콜라 한 캔", "names": [["치킨"], ["콜라"]], "kcal": (450, 900)},
    {"id": "D4-ml", "text": "우유 300ml 마셨어", "names": [["우유"]], "amount": 300, "unit": "ml", "kcal": (150, 230)},
    {"id": "D5-typo-spacing", "text": "닭가슴 살 150그램", "names": [["닭가슴살", "닭가슴 살"]], "amount": 150, "kcal": (160, 270)},
    {"id": "D6-protein-scoop", "text": "운동 끝나고 프로틴 한 스쿱", "names": [["프로틴", "단백질"]], "kcal": (90, 160)},
    {"id": "D7-three-items", "text": "아침: 계란 2개, 바나나 1개, 아메리카노", "names": [["계란", "달걀"], ["바나나"], ["아메리카노", "커피"]], "kcal": (200, 330)},
    {"id": "D8-snack-brand", "text": "편의점 삼각김밥 참치마요 하나", "names": [["삼각김밥", "김밥"]], "kcal": (180, 280)},
    {"id": "D9-ramen", "text": "라면 하나 끓여 먹음", "names": [["라면"]], "kcal": (450, 600)},
    {"id": "D10-noise", "text": "오늘 너무 피곤했다 그래도 저녁엔 샐러드 먹음", "names": [["샐러드"]], "kcal": (30, 400)},
    {"id": "D11-negation", "text": "저녁은 안 먹고 사과 하나만 먹었어", "names": [["사과"]], "kcal": (50, 130), "max_items": 1},
    {"id": "D12-english-mix", "text": "subway 터키 샌드위치 15cm", "names": [["샌드위치", "써브웨이", "subway"]], "kcal": (250, 450)},
]

WORKOUT = [
    {"id": "W1-basic", "text": "벤치프레스 60키로 10개 3세트", "ex": [("벤치", 60, 3, 10)]},
    {"id": "W2-two-ex", "text": "스쿼트 100kg 5회 5세트 하고 레그프레스 150kg 12회 3세트", "ex": [("스쿼트", 100, 5, 5), ("레그", 150, 3, 12)]},
    {"id": "W3-bodyweight", "text": "턱걸이 8개씩 4세트", "ex": [("턱걸이", None, 4, 8)]},
    {"id": "W4-lbs", "text": "덤벨컬 25lbs 12개 3세트", "ex": [("컬", 11.3, 3, 12)]},
    {"id": "W5-slang", "text": "데드 140 5개 3셋", "ex": [("데드", 140, 3, 5)]},
    {"id": "W6-mixed-diet", "text": "오늘 벤치 80kg 5x5 했고 저녁에 닭가슴살 200g 먹음", "ex": [("벤치", 80, 5, 5)], "diet": ["닭가슴살"]},
    {"id": "W7-english", "text": "OHP 40kg 8 reps 4 sets", "ex": [("OHP", 40, 4, 8)]},
    {"id": "W8-pyramid", "text": "벤치 60kg 12개, 70kg 10개, 80kg 8개", "ex": [("벤치", None, None, None)], "note": "pyramid: check whether 3 weights survive"},
]


def post(path, text):
    body = json.dumps({"text": text}).encode()
    t0 = time.time()
    r = urllib.request.urlopen(urllib.request.Request(f"{BASE}/{path}", body, {"Content-Type": "application/json"}), timeout=300)
    return json.loads(r.read()), time.time() - t0


def score_diet(c, resp):
    items = resp.get("diet_logs") or resp.get("items") or []
    issues = []
    names = [str(i.get("food_name", "")) for i in items]
    for alts in c["names"]:
        if not any(any(a in n for a in alts) for n in names):
            issues.append(f"missing item {alts[0]}")
    if c.get("max_items") and len(items) > c["max_items"]:
        issues.append(f"extra items {names}")
    kcal = 0.0
    for i in items:
        k = i.get("kcal")
        if k is None:
            p, cb, f = (i.get("protein_g") or 0), (i.get("carbs_g") or 0), (i.get("fat_g") or 0)
            k = p * 4 + cb * 4 + f * 9
        kcal += k or 0
    lo, hi = c["kcal"]
    if not (lo <= kcal <= hi):
        issues.append(f"kcal {kcal:.0f} outside {lo}-{hi}")
    if "amount" in c and not any(i.get("amount") == c["amount"] for i in items):
        issues.append(f"amount {c['amount']} lost: {[(i.get('amount'), i.get('unit')) for i in items]}")
    return issues, round(kcal)


def score_workout(c, resp):
    logs = resp.get("workout_logs") or []
    issues = []
    for key, w, sets, reps in c["ex"]:
        hit = [l for l in logs if key.lower() in str(l.get("exercise_name", "")).lower()]
        if not hit:
            issues.append(f"missing exercise {key}")
            continue
        l = hit[0]
        if w is not None and (l.get("weight_kg") is None or abs(float(l["weight_kg"]) - w) > 1.0):
            issues.append(f"{key} weight {l.get('weight_kg')} != {w}")
        if sets is not None and l.get("sets") != sets:
            issues.append(f"{key} sets {l.get('sets')} != {sets}")
        if reps is not None and l.get("reps") != reps:
            issues.append(f"{key} reps {l.get('reps')} != {reps}")
    for d in c.get("diet", []):
        if not any(d in str(x.get("food_name", "")) for x in resp.get("diet_logs") or []):
            issues.append(f"missing diet {d}")
    return issues


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in DIET:
        resp, lat = post("parse-diet", c["text"])
        issues, kcal = score_diet(c, resp)
        rows.append({"kind": "diet", "id": c["id"], "text": c["text"], "latency_s": round(lat, 1), "kcal": kcal, "issues": issues, "resp": resp})
        print(json.dumps({k: rows[-1][k] for k in ("id", "latency_s", "kcal", "issues")}, ensure_ascii=False), flush=True)
    for c in WORKOUT:
        resp, lat = post("parse-log", c["text"])
        issues = score_workout(c, resp)
        rows.append({"kind": "workout", "id": c["id"], "text": c["text"], "latency_s": round(lat, 1), "issues": issues, "resp": resp})
        print(json.dumps({k: rows[-1][k] for k in ("id", "latency_s", "issues")}, ensure_ascii=False), flush=True)
    (OUT / "quicklog_extra.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

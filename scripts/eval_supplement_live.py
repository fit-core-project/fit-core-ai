from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "tests" / "fixtures" / "supplement_live_eval_cases.json"
DEFAULT_OUT_DIR = ROOT / "tests" / "evaluation" / ".artifacts" / "supplement-live"


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("supplement-live-%Y%m%dT%H%M%SZ")


def load_cases(path: Path) -> list[dict[str, Any]]:
    return list(json.loads(path.read_text(encoding="utf-8")))


def select_cases(cases: list[dict[str, Any]], case_ids: list[str] | None, limit: int | None) -> list[dict[str, Any]]:
    selected = cases
    if case_ids:
        wanted = set(case_ids)
        selected = [case for case in cases if case["id"] in wanted]
        missing = sorted(wanted - {case["id"] for case in selected})
        if missing:
            raise SystemExit(f"case id not found: {', '.join(missing)}")
    if limit is not None:
        selected = selected[:limit]
    return selected


def post_json(url: str, payload: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:  # noqa: S310 - local/user-provided eval URL.
        return json.loads(response.read().decode("utf-8"))


def has_any(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return any(str(needle).lower() in lowered for needle in needles if str(needle).strip())


_EXPECTED_ENTITY_EQUIVALENTS = {
    "aspirin": ["aspirin", "아스피린", "antiplatelet", "항혈소판제"],
    "berberine": ["berberine", "베르베린"],
    "blood pressure medication": [
        "blood pressure medication",
        "혈압약",
        "혈압 약",
        "고혈압약",
        "ACE 억제제",
        "ARB",
        "베타차단제",
        "칼슘채널차단제",
        "이뇨제",
    ],
    "coffee": ["coffee", "커피", "caffeine", "카페인"],
    "caffeine": ["caffeine", "카페인", "coffee", "커피"],
    "diabetes medication": ["diabetes medication", "당뇨약", "당뇨 약", "메트포르민", "인슐린", "SGLT2", "GLP-1"],
    "digestive symptom": ["digestive symptom", "위장", "위가", "속이", "메스꺼움", "설사", "복통", "변비"],
    "eaa": ["eaa", "필수아미노산", "essential amino acids"],
    "glutamine": ["glutamine", "글루타민"],
    "kidney disease": ["kidney disease", "신장질환", "신장 기능", "신장기능", "renal", "투석"],
    "liver disease": ["liver disease", "간질환", "간수치", "간 기능", "간기능"],
    "melatonin": ["melatonin", "멜라토닌"],
    "nac": ["nac", "엔아세틸시스테인", "n-acetylcysteine"],
    "nmn": ["nmn"],
    "omega3": ["omega3", "omega-3", "오메가3", "오메가"],
    "quercetin": ["quercetin", "퀘르세틴"],
    "theanine": ["theanine", "테아닌", "l-theanine", "L-테아닌"],
    "vitamin a": ["vitamin a", "vitamina", "비타민a", "비타민A"],
    "vitamin b": ["vitamin b", "vitaminb", "비타민b", "비타민B", "비타민B군"],
    "vitamin c": ["vitamin c", "vitaminc", "비타민c", "비타민C"],
    "vitamin d": ["vitamin d", "vitamind", "비타민d", "비타민D"],
    "vitamin e": ["vitamin e", "vitamine", "비타민e", "비타민E"],
    "probiotics": ["probiotics", "probiotic", "유산균", "프로바이오틱스"],
    "thyroid medication": ["thyroid medication", "levothyroxine", "갑상선약"],
    "warfarin": ["warfarin", "와파린", "anticoagulant", "항응고제"],
    "high dose": ["high dose", "고용량", "고함량", "함량", "총량"],
    "pregnancy": ["pregnancy", "임신", "임산부", "임신 준비"],
    "surgery": ["surgery", "수술", "시술"],
}


def expected_entity_evidenced(expected: str, evidence_norm: str) -> bool:
    expected_raw = str(expected)
    candidates = _EXPECTED_ENTITY_EQUIVALENTS.get(expected_raw.lower(), [expected_raw])
    for candidate in candidates:
        candidate_norm = str(candidate).lower().replace("-", "").replace("_", "").replace(" ", "")
        if candidate_norm and candidate_norm in evidence_norm:
            return True
    return False


def score_case(
    case: dict[str, Any],
    payload: dict[str, Any],
    elapsed_ms: float,
    *,
    allow_degraded: bool,
) -> dict[str, Any]:
    answer = str(payload.get("answer") or "")
    caution = str(payload.get("caution") or "")
    mode = str(payload.get("mode") or "")
    sources = payload.get("sources") or []
    combined = f"{answer} {caution}"
    source_text = json.dumps(sources, ensure_ascii=False).lower() if isinstance(sources, list) else ""
    failures: list[str] = []
    warnings: list[str] = []

    if not answer.strip():
        failures.append("answer is empty")
    if case.get("risk_level") == "high" and not caution.strip():
        failures.append("high-risk case has no caution")
    for needle in case.get("must_not_include") or []:
        if needle and needle in combined:
            failures.append(f"must_not_include found: {needle}")
    if not sources:
        if allow_degraded:
            warnings.append("sources is empty")
        else:
            failures.append("sources is empty")
    if mode and mode != "full":
        if allow_degraded:
            warnings.append(f"mode is {mode}")
        else:
            failures.append(f"mode is {mode}; expected full RAG mode")
    for expected in case.get("expected_entities") or []:
        evidence_norm = f"{combined} {source_text}".lower().replace("-", "").replace("_", "").replace(" ", "")
        if not expected_entity_evidenced(str(expected), evidence_norm):
            failures.append(f"expected entity not evidenced: {expected}")

    category = case.get("category")
    if category == "multi_supplement":
        labels = ["성분", "함께", "확인", "오메가3", "철분", "아연", "마그네슘", "프로틴", "크레아틴"]
        if not has_any(combined, labels):
            failures.append("multi-supplement answer does not expose component checks")
    if category in {"drug_interaction", "condition_safety", "condition_risk"}:
        risk_words = ["상담", "의료진", "의사", "약사", "간격", "위험", "주의", "확인"]
        if not has_any(combined, risk_words):
            failures.append("risk answer lacks caution/action words")
    if category == "unknown_partial_known":
        detail_words = ["제품명", "성분표", "함량", "복용량", "용량"]
        if not has_any(combined, detail_words):
            failures.append("unknown product answer does not request product details")

    return {
        "id": case["id"],
        "category": case.get("category"),
        "risk_level": case.get("risk_level"),
        "question": case["question"],
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "warnings": warnings,
        "elapsed_ms": elapsed_ms,
        "mode": mode,
        "sources_count": len(sources) if isinstance(sources, list) else 0,
        "sources": sources if isinstance(sources, list) else [],
        "answer": answer,
        "caution": caution,
    }


def run_case(args: argparse.Namespace, case: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        payload = post_json(
            f"{args.base_url.rstrip('/')}/api/ai/supplement-chat",
            {"question": case["question"]},
            args.timeout_sec,
        )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {
            "id": case["id"],
            "category": case.get("category"),
            "risk_level": case.get("risk_level"),
            "question": case["question"],
            "status": "error",
            "failures": [f"HTTP error: {exc}"],
            "warnings": [],
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "answer": "",
            "caution": "",
        }
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    return score_case(case, payload, elapsed_ms, allow_degraded=args.allow_degraded)


def write_outputs(args: argparse.Namespace, results: list[dict[str, Any]]) -> tuple[Path, Path]:
    run_id = args.run_id or utc_run_id()
    out_dir = Path(args.out_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "supplement_live_eval_results.json"
    md_path = out_dir / "supplement_live_eval_summary.md"
    total = len(results)
    pass_count = sum(1 for result in results if result.get("status") == "pass")
    error_count = sum(1 for result in results if result.get("status") == "error")
    json_path.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = [
        "# Supplement Live Eval Summary",
        "",
        f"- baseUrl: `{args.base_url}`",
        f"- total: {total}",
        f"- pass: {pass_count}",
        f"- error: {error_count}",
        f"- timeoutSec: {args.timeout_sec}",
        "",
        "| ID | Status | Mode | Sources | Elapsed ms | Failures |",
        "|---|---|---|---:|---:|---|",
    ]
    for result in results:
        failures = "; ".join(result.get("failures") or [])[:180].replace("\n", " ")
        rows.append(
            f"| {result['id']} | {result.get('status')} | {result.get('mode', '')} | "
            f"{result.get('sources_count', 0)} | {result.get('elapsed_ms')} | {failures} |"
        )
    md_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded supplement endpoint live evaluation.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--run-id")
    parser.add_argument("--timeout-sec", type=float, default=20.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument(
        "--allow-degraded",
        action="store_true",
        help="Treat degraded mode as endpoint-survival smoke instead of quality failure.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = select_cases(load_cases(Path(args.cases)), args.case_id, args.limit)
    results = [run_case(args, case) for case in cases]
    json_path, md_path = write_outputs(args, results)
    print(f"results: {json_path}")
    print(f"summary: {md_path}")
    return 1 if any(result.get("status") != "pass" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())

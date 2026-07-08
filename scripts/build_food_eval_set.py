from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
FOOD_CSV_PATH = PROJECT_ROOT / "data" / "food_db" / "food_db_clean.csv"
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
EVALSET_PATH = EVAL_DIR / "food_search_evalset_v1.jsonl"
RESULTS_PATH = EVAL_DIR / "food_search_baseline_v1_results.jsonl"
REPORT_PATH = EVAL_DIR / "food_search_baseline_v1_report.md"
EVALSET_V2_PATH = EVAL_DIR / "food_search_evalset_v2.jsonl"
RESULTS_V2_PATH = EVAL_DIR / "food_search_baseline_v2_results.jsonl"
REPORT_V2_PATH = EVAL_DIR / "food_search_baseline_v2_report.md"

TYPE_TARGETS = {"A": 200, "B": 200, "C": 200, "D": 200, "E": 200}
RAW_DATA_TYPE = "원재료성 식품"
DISH_DATA_TYPE = "음식"
COOKING_STATES = ("삶은것", "구운것", "찐것", "데친것", "볶은것")
COOKING_PREFIX = {
    "삶은것": "삶은",
    "구운것": "구운",
    "찐것": "찐",
    "데친것": "데친",
    "볶은것": "볶은",
}
QUANTITY_SUFFIXES = (" 100g", " 200g", " 1개", " 2개", " 한컵")
PROCESSED_OR_COMPOUND_TERMS = (
    "젓갈",
    "맛탕",
    "샐러드",
    "샌드위치",
    "김밥",
    "찌개",
    "전골",
    "볶음밥",
    "덮밥",
    "튀김",
    "장아찌",
    "절임",
)


def read_food_rows() -> list[dict[str, str]]:
    with FOOD_CSV_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_spaces(value: str) -> str:
    return " ".join((value or "").strip().split())


def compact_text(value: str | None) -> str:
    return normalize_spaces(value or "").replace(" ", "")


def base_compare_text(value: str | None) -> str:
    text = normalize_spaces(value or "")
    for token in (
        "생것",
        "삶은것",
        "구운것",
        "찐것",
        "데친것",
        "볶은것",
        "조린것",
        "말린것",
        "대표",
        "평균",
        "국산",
        "수입",
    ):
        text = text.replace(token, "")
    return compact_text(text)


def row_key(row: dict[str, str]) -> tuple[str, str]:
    return (row.get("food_id", ""), row.get("name", ""))


def has_macro(row: dict[str, str]) -> bool:
    return any((row.get(field) or "").strip() for field in ("protein_g", "carb_g", "fat_g", "kcal"))


def detect_cooking_state(name: str) -> str | None:
    for state in COOKING_STATES:
        if state in name:
            return state
    return None


def strip_cooking_state(name: str) -> str:
    stripped = name
    for state in COOKING_STATES:
        stripped = stripped.replace(state, "")
    return normalize_spaces(stripped)


def ambiguity_base(name: str) -> str:
    base = strip_cooking_state(name)
    for token in ("생것", "말린것", "냉동", "대표", "평균", "국산", "수입"):
        base = base.replace(token, "")
    return normalize_spaces(base)


def is_processed_or_compound_name(name: str | None) -> bool:
    text = normalize_spaces(name or "")
    return any(term in text for term in PROCESSED_OR_COMPOUND_TERMS)


def is_same_base_name(base: str | None, name: str | None) -> bool:
    base_compact = base_compare_text(base)
    name_compact = base_compare_text(name)
    return bool(base_compact and name_compact and name_compact.startswith(base_compact))


def same_base_allowed_names(base: str, rows: list[dict[str, str]]) -> list[str]:
    return sorted(
        {
            row["name"]
            for row in rows
            if row.get("name")
            and row.get("data_type") == RAW_DATA_TYPE
            and is_same_base_name(base, row["name"])
            and not is_processed_or_compound_name(row["name"])
        }
    )


def refine_evalset_v2(items: list[dict[str, Any]], rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    refined = [dict(item) for item in items]
    for item in refined:
        if item.get("type") != "E":
            continue
        if item.get("expected_kind") == "preserve_ambiguous":
            item["expected"] = {
                "allowed_base": item["query"],
                "disallow": "different_base_or_processed_or_compound_or_no_match",
                "blocked_terms": list(PROCESSED_OR_COMPOUND_TERMS),
            }
            item["note"] = (
                "v2: ambiguous 단일 재료명은 같은 base 생것/기본형, 조리형, provenance 변형을 허용하고 "
                "다른 재료/가공품/복합요리/no-match만 실패 처리"
            )
        elif item.get("expected_kind") == "allowed_set":
            allowed = same_base_allowed_names(item["query"], rows)
            if allowed:
                item["expected"] = allowed
                item["note"] = (
                    "v2: CSV에 실재하는 같은 base 행 중 가공품/복합요리를 제외한 생것/조리형/provenance 변형 허용"
                )
    return refined


def round_robin_by_category(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: (r.get("major_category", ""), r.get("name", ""), r.get("food_id", ""))):
        buckets[row.get("major_category") or "미분류"].append(row)
    selected: list[dict[str, str]] = []
    categories = sorted(buckets)
    index = 0
    while len(selected) < limit and categories:
        category = categories[index % len(categories)]
        bucket = buckets[category]
        if bucket:
            selected.append(bucket.pop(0))
        categories = [cat for cat in categories if buckets[cat]]
        index += 1
    if len(selected) < limit:
        raise RuntimeError(f"Not enough rows to select {limit}; got {len(selected)}")
    return selected


def add_item(
    items: list[dict[str, Any]],
    used_queries: set[str],
    item_type: str,
    query: str,
    row: dict[str, str],
    expected_kind: str,
    expected: Any,
    note: str,
) -> bool:
    query = normalize_spaces(query)
    if not query or query in used_queries:
        return False
    used_queries.add(query)
    items.append(
        {
            "id": f"eval_{len(items) + 1:04d}",
            "type": item_type,
            "query": query,
            "source_food_id": row.get("food_id") or None,
            "source_name": row.get("name") or None,
            "major_category": row.get("major_category") or "미분류",
            "expected_kind": expected_kind,
            "expected": expected,
            "note": note,
        }
    )
    return True


def build_evalset(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = [row for row in rows if row.get("name") and row.get("food_id") and has_macro(row)]
    raw_rows = [row for row in rows if row.get("data_type") == RAW_DATA_TYPE]
    dish_rows = [row for row in rows if row.get("data_type") == DISH_DATA_TYPE]
    cooking_rows = [row for row in raw_rows if detect_cooking_state(row["name"])]
    plain_raw_rows = [row for row in raw_rows if len(row["name"]) <= 28]
    dish_sample_rows = [row for row in dish_rows if len(row["name"]) <= 28]

    items: list[dict[str, Any]] = []
    used_queries: set[str] = set()

    for row in round_robin_by_category(plain_raw_rows, TYPE_TARGETS["A"]):
        add_item(items, used_queries, "A", row["name"], row, "exact", row["name"], "CSV 원재료성 식품 name exact")

    for row in round_robin_by_category(cooking_rows, TYPE_TARGETS["B"]):
        state = detect_cooking_state(row["name"])
        ingredient = strip_cooking_state(row["name"])
        query = f"{COOKING_PREFIX[state]} {ingredient}" if state else row["name"]
        if not add_item(items, used_queries, "B", query, row, "exact", row["name"], f"조리상태 '{state}' 사용자식 어순 변형"):
            add_item(items, used_queries, "B", f"{ingredient} {COOKING_PREFIX[state]}", row, "exact", row["name"], f"조리상태 '{state}' 사용자식 후치 변형")

    c_count = 0
    for row in round_robin_by_category(dish_sample_rows, TYPE_TARGETS["C"] * 3):
        if c_count >= TYPE_TARGETS["C"]:
            break
        if add_item(items, used_queries, "C", row["name"], row, "exact", row["name"], "CSV 음식 name exact"):
            c_count += 1
    if c_count < TYPE_TARGETS["C"]:
        raise RuntimeError(f"Not enough C items; got {c_count}")

    d_sources = round_robin_by_category(plain_raw_rows[:300] + cooking_rows[:300] + dish_sample_rows[:500], TYPE_TARGETS["D"] * 3)
    d_count = 0
    for row in d_sources:
        if d_count >= TYPE_TARGETS["D"]:
            break
        name = row["name"]
        variants = [
            name.replace(" ", ""),
            f"{name}{QUANTITY_SUFFIXES[d_count % len(QUANTITY_SUFFIXES)]}",
            f"{name} {QUANTITY_SUFFIXES[(d_count + 1) % len(QUANTITY_SUFFIXES)].strip()}",
        ]
        for query in variants:
            if query != name and add_item(items, used_queries, "D", query, row, "exact", name, "띄어쓰기/수량 접미 사용자 변형"):
                d_count += 1
                break
    if d_count < TYPE_TARGETS["D"]:
        raise RuntimeError(f"Not enough D items; got {d_count}")

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        base = ambiguity_base(row["name"])
        if base and len(base) >= 2:
            groups[base].append(row)
    ambiguous_groups = [
        (base, group)
        for base, group in groups.items()
        if len({r["name"] for r in group}) >= 2
    ]
    ambiguous_groups.sort(key=lambda pair: (pair[1][0].get("major_category", ""), pair[0]))

    e_allowed = 0
    e_preserve = 0
    for base, group in ambiguous_groups:
        if e_allowed + e_preserve >= TYPE_TARGETS["E"]:
            break
        names = sorted({row["name"] for row in group})
        source = sorted(group, key=lambda r: (r.get("major_category", ""), r.get("name", "")))[0]
        raw_candidates = [name for name in names if "생것" in name]
        if e_preserve < TYPE_TARGETS["E"] // 2 and raw_candidates:
            expected = {"must_not_force": "raw ingredient", "blocked_names": raw_candidates}
            if add_item(items, used_queries, "E", base, source, "preserve_ambiguous", expected, "동일 base의 생것/조리형 복수 존재: raw 강제 금지"):
                e_preserve += 1
                continue
        if e_allowed < TYPE_TARGETS["E"] // 2:
            if add_item(items, used_queries, "E", base, source, "allowed_set", names[:12], "동일 base 파생 행 허용 집합"):
                e_allowed += 1

    if e_allowed + e_preserve < TYPE_TARGETS["E"]:
        raise RuntimeError(f"Not enough E items; got {e_allowed + e_preserve}")

    counts = Counter(item["type"] for item in items)
    if len(items) != 1000 or counts != Counter(TYPE_TARGETS):
        raise RuntimeError(f"Unexpected evalset distribution: total={len(items)} counts={dict(counts)}")
    return items


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def doc_name(document: Any) -> str | None:
    return (getattr(document, "metadata", None) or {}).get("name")


def doc_food_id(document: Any) -> str | None:
    return (getattr(document, "metadata", None) or {}).get("food_id")


def candidate_record(candidate: Any) -> dict[str, Any]:
    meta = getattr(candidate.document, "metadata", None) or {}
    return {
        "food_id": meta.get("food_id"),
        "name": meta.get("name"),
        "rep_name": meta.get("rep_name"),
        "data_type": meta.get("data_type"),
        "major_category": meta.get("major_category"),
        "vector_score": round(float(candidate.vector_score), 6),
        "source_query": candidate.source_query,
        "source_query_index": candidate.source_query_index,
        "original_rank": candidate.original_rank,
    }


def trace_search(engine: Any, query: str) -> dict[str, Any]:
    from engines.food.food_engine import FOOD_SEARCH_FETCH_K, FOOD_SEARCH_FINAL_K
    from engines.food.food_query_normalization import analyze_food_query_for_search, build_food_search_queries
    from engines.food.food_reranker import rerank_food_candidates

    analysis = analyze_food_query_for_search(query)
    search_queries = build_food_search_queries(query)
    result_groups = []
    if search_queries:
        with engine._lock:
            result_groups = [
                engine._vector_store.similarity_search_with_score(search_query, k=FOOD_SEARCH_FETCH_K)
                for search_query in search_queries
            ]
    canonical_candidates = engine._lookup_canonical_candidates(analysis)
    vector_candidates = engine._collect_candidates(result_groups, search_queries)
    candidates = engine._dedupe_candidates(canonical_candidates + vector_candidates)
    reranked = rerank_food_candidates(analysis, candidates, final_k=FOOD_SEARCH_FINAL_K)
    result = engine.search(query, 100, "g")
    vector_flat = [
        {"name": doc_name(doc), "food_id": doc_food_id(doc), "distance": round(float(distance), 6), "source_query": search_queries[group_index]}
        for group_index, group in enumerate(result_groups)
        for doc, distance in group[:3]
    ]
    return {
        "matched_name": result.get("matched_name") if result else None,
        "similarity": result.get("similarity") if result else None,
        "source": result.get("source") if result else None,
        "normalized_query": analysis.normalized_query,
        "cleaned_query": analysis.cleaned_query,
        "intent": analysis.intent,
        "cooking_state": analysis.cooking_state,
        "normalization_reason": analysis.reason,
        "canonical_top1": candidate_record(canonical_candidates[0]) if canonical_candidates else None,
        "canonical_candidates": [candidate_record(candidate) for candidate in canonical_candidates[:5]],
        "vector_top1": vector_flat[0] if vector_flat else None,
        "vector_topk": vector_flat[:10],
        "reranker_top1": candidate_record(reranked[0]) if reranked else None,
        "reranker_topk": [candidate_record(candidate) for candidate in reranked[:5]],
    }


def is_pass(item: dict[str, Any], trace: dict[str, Any]) -> bool:
    actual = trace.get("matched_name")
    expected_kind = item["expected_kind"]
    expected = item["expected"]
    if expected_kind == "exact":
        return actual == expected
    if expected_kind == "allowed_set":
        return actual in set(expected)
    if expected_kind == "preserve_ambiguous":
        if not actual:
            return False
        allowed_base = expected.get("allowed_base")
        if allowed_base:
            return is_same_base_name(allowed_base, actual) and not is_processed_or_compound_name(actual)
        blocked = set(expected.get("blocked_names", []))
        return actual not in blocked
    raise ValueError(f"Unknown expected_kind: {expected_kind}")


def classify_failure(item: dict[str, Any], trace: dict[str, Any], csv_names: set[str]) -> tuple[str, str]:
    actual = trace.get("matched_name")
    expected = item["expected"]
    if item["expected_kind"] == "exact" and expected not in csv_names:
        return "데이터", "expected name is not present in CSV"
    allowed = set(expected if isinstance(expected, list) else [expected]) if item["expected_kind"] != "preserve_ambiguous" else set()
    canonical_names = {candidate.get("name") for candidate in trace.get("canonical_candidates", []) if candidate.get("name")}
    vector_names = {item.get("name") for item in trace.get("vector_topk", []) if item.get("name")}
    rerank_names = {candidate.get("name") for candidate in trace.get("reranker_topk", []) if candidate.get("name")}
    if allowed and allowed & canonical_names and actual not in allowed:
        return "reranker", "expected appears in canonical candidates but reranker top-1 differs"
    if allowed and allowed & vector_names and actual not in allowed:
        return "reranker", "expected appears in vector top-k but reranker top-1 differs"
    if allowed and not (allowed & canonical_names):
        if item["expected_kind"] in {"exact", "allowed_set"} and trace.get("normalized_query") != item["query"] and trace.get("canonical_top1") is None:
            return "정규화", "normalized query changed and expected was not found canonically"
        return "canonical", "expected was not returned by canonical lookup"
    if allowed and not (allowed & vector_names):
        return "vector", "expected was absent from vector top-k"
    if item["expected_kind"] == "preserve_ambiguous":
        allowed_base = expected.get("allowed_base")
        if allowed_base:
            if not actual:
                return "canonical", "ambiguous query returned no top-1 match"
            if is_processed_or_compound_name(actual):
                return "reranker", "ambiguous query matched a processed or compound food"
            if not is_same_base_name(allowed_base, actual):
                return "vector", "ambiguous query matched a different base ingredient"
            return "기대값의심", "preserve_ambiguous expected is inconsistent with same-base actual"
        blocked = set(expected.get("blocked_names", []))
        if actual in blocked:
            return "정규화", "ambiguous query resolved to a blocked raw ingredient"
    if actual and item["expected_kind"] == "exact" and item["query"].replace(" ", "") == actual.replace(" ", ""):
        return "기대값의심", "actual differs only by whitespace from generated exact expectation"
    return "미분류", "no deterministic stage rule matched"


def run_baseline(evalset: list[dict[str, Any]], rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    from engines.food.food_engine import get_food_engine

    engine = get_food_engine()
    if not engine.available or engine._vector_store is None:
        raise RuntimeError("FoodSearchEngine is not available; check local Chroma DB and embedding model")
    csv_names = {row["name"] for row in rows if row.get("name")}
    results = []
    started = time.perf_counter()
    for index, item in enumerate(evalset, start=1):
        trace = trace_search(engine, item["query"])
        passed = is_pass(item, trace)
        stage, reason = ("PASS", "matched expectation") if passed else classify_failure(item, trace, csv_names)
        results.append({**item, "pass": passed, "failure_stage": None if passed else stage, "failure_reason": reason, **trace})
        if index % 50 == 0:
            elapsed = time.perf_counter() - started
            print(f"{index}/{len(evalset)} complete ({elapsed:.1f}s)")
    print(f"baseline complete: {len(evalset)} queries in {time.perf_counter() - started:.1f}s")
    return results


def summarize_percent(passed: int, total: int) -> str:
    return f"{passed}/{total} ({passed / total * 100:.1f}%)" if total else "0/0 (0.0%)"


def make_report(results: list[dict[str, Any]], version: str = "v1", v1_results: list[dict[str, Any]] | None = None) -> str:
    total = len(results)
    passed = sum(1 for row in results if row["pass"])
    lines = [
        f"# Food Search Baseline {version}",
        "",
        f"- Evalset: {total} queries",
        f"- Overall accuracy: {summarize_percent(passed, total)}",
    ]
    if v1_results:
        v1_total = len(v1_results)
        v1_passed = sum(1 for row in v1_results if row["pass"])
        lines.append(f"- v1 comparison: {summarize_percent(v1_passed, v1_total)} -> {summarize_percent(passed, total)}")
    if version == "v2":
        lines.extend(
            [
                "- v2 refinement: preserve_ambiguous now passes same-base raw/basic ingredient matches.",
                "- allowed_set expansion: E allowed sets include same-base CSV rows, excluding processed or compound terms.",
                "- Still-fail contract: no-match, different-base, processed, and compound foods remain failures.",
            ]
        )
    lines.extend(["", "## Accuracy by Type", "", "| Type | Pass | Total | Accuracy |", "|---|---:|---:|---:|"])
    for item_type in sorted({row["type"] for row in results}):
        subset = [row for row in results if row["type"] == item_type]
        ok = sum(1 for row in subset if row["pass"])
        lines.append(f"| {item_type} | {ok} | {len(subset)} | {ok / len(subset) * 100:.1f}% |")

    lines.extend(["", "## Accuracy by Major Category", "", "| Major category | Pass | Total | Accuracy |", "|---|---:|---:|---:|"])
    by_category = sorted(Counter(row["major_category"] for row in results).items(), key=lambda pair: (-pair[1], pair[0]))
    for category, count in by_category:
        subset = [row for row in results if row["major_category"] == category]
        ok = sum(1 for row in subset if row["pass"])
        lines.append(f"| {category} | {ok} | {count} | {ok / count * 100:.1f}% |")

    failures = [row for row in results if not row["pass"]]
    stage_counts = Counter(row["failure_stage"] for row in failures)
    lines.extend(["", "## Failure Stage Distribution", "", "| Stage | Count |", "|---|---:|"])
    for stage in ("정규화", "canonical", "vector", "reranker", "데이터", "기대값의심", "미분류"):
        lines.append(f"| {stage} | {stage_counts.get(stage, 0)} |")

    lines.extend(["", "## Representative Failures", ""])
    for stage in ("정규화", "canonical", "vector", "reranker", "데이터", "기대값의심", "미분류"):
        examples = [row for row in failures if row["failure_stage"] == stage][:5]
        if not examples:
            continue
        lines.extend([f"### {stage}", "", "| Query | Expected | Actual | Evidence |", "|---|---|---|---|"])
        for row in examples:
            expected = json.dumps(row["expected"], ensure_ascii=False)
            evidence = row.get("failure_reason") or ""
            actual = row.get("matched_name") or "None"
            lines.append(f"| {row['query']} | {expected} | {actual} | {evidence} |")
        lines.append("")

    suspected = stage_counts.get("기대값의심", 0)
    lines.extend(
        [
            "## Evalset Quality Signal",
            "",
            f"- 기대값의심: {suspected} cases. These should be reviewed before using a future v2 evalset.",
            "",
            "## Improvement Priority",
            "",
            "| Priority | Stage | Potential XPASS | Rationale |",
            "|---:|---|---:|---|",
        ]
    )
    ranked = [(stage, count) for stage, count in stage_counts.most_common() if stage not in {"기대값의심", "데이터"}]
    for rank, (stage, count) in enumerate(ranked, start=1):
        lines.append(f"| {rank} | {stage} | {count} | Largest current deterministic failure bucket. |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build food search evalset and optional baseline report.")
    parser.add_argument("--baseline", action="store_true", help="Run FoodSearchEngine over the evalset and write results/report.")
    parser.add_argument("--version", choices=("v1", "v2"), default="v1", help="Evalset version to build.")
    args = parser.parse_args()

    rows = read_food_rows()
    evalset = build_evalset(rows)
    evalset_path = EVALSET_PATH
    results_path = RESULTS_PATH
    report_path = REPORT_PATH
    if args.version == "v2":
        evalset = refine_evalset_v2(evalset, rows)
        evalset_path = EVALSET_V2_PATH
        results_path = RESULTS_V2_PATH
        report_path = REPORT_V2_PATH

    write_jsonl(evalset_path, evalset)
    print(f"wrote {evalset_path} ({len(evalset)} rows)")
    print("type distribution:", dict(Counter(item["type"] for item in evalset)))
    print("category distribution:", dict(Counter(item["major_category"] for item in evalset).most_common()))

    if args.baseline:
        results = run_baseline(evalset, rows)
        write_jsonl(results_path, results)
        v1_results = load_jsonl(RESULTS_PATH) if args.version == "v2" and RESULTS_PATH.exists() else None
        report_path.write_text(
            make_report(results, version=args.version, v1_results=v1_results),
            encoding="utf-8",
            newline="\n",
        )
        print(f"wrote {results_path}")
        print(f"wrote {report_path}")


if __name__ == "__main__":
    main()

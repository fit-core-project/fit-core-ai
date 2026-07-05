#!/usr/bin/env python3
"""Read-only food alias candidate discovery.

This script inspects food_db_clean.csv and emits review candidates for future
embedding alias enrichment. It does not modify CSV files, Chroma indexes, or
runtime alias registries.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "food_db" / "food_db_clean.csv"

SAFE_EXACT_SYNONYM = "SAFE_EXACT_SYNONYM"
SAFE_SPACING_VARIANT = "SAFE_SPACING_VARIANT"
SAFE_COOKING_STATE_VARIANT = "SAFE_COOKING_STATE_VARIANT"
REVIEW_REP_NAME_GROUP = "REVIEW_REP_NAME_GROUP"
HIGH_RISK_COMPOUND = "HIGH_RISK_COMPOUND"
HIGH_RISK_AMBIGUOUS_INGREDIENT = "HIGH_RISK_AMBIGUOUS_INGREDIENT"
HIGH_RISK_PRODUCT_OR_BRAND = "HIGH_RISK_PRODUCT_OR_BRAND"
DO_NOT_LINK_PROTECTED_COMPOUND = "DO_NOT_LINK_PROTECTED_COMPOUND"

COMPOUND_KEYWORDS: tuple[str, ...] = (
    "샌드위치",
    "볶음밥",
    "샐러드",
    "김밥",
    "덮밥",
    "찌개",
    "튀김",
    "조림",
    "빵",
    "국",
)

PROTECTED_COMPOUND_EXAMPLES: tuple[str, ...] = (
    "계란빵",
    "볶음밥 계란",
    "김밥 계란",
    "샐러드 닭가슴살",
    "닭가슴살 샐러드",
    "샌드위치 닭가슴살",
)

SYNONYM_REVIEW_RULES: tuple[tuple[str, str], ...] = (
    ("달걀", "계란"),
    ("쇠고기", "소고기"),
    ("돼지고기", "돈육"),
    ("우유", "밀크"),
    ("두유", "콩우유"),
)

COOKING_STATE_RULES: tuple[tuple[str, str], ...] = (
    ("생것", "생"),
    ("삶은것", "삶은"),
    ("구운것", "구운"),
    ("찐것", "찐"),
    ("볶은것", "볶은"),
)


@dataclass(frozen=True)
class FoodAliasCandidate:
    source_alias: str
    target_name: str
    target_rep_name: str | None
    data_type: str | None
    major_category: str | None
    candidate_type: str
    risk_label: str
    reason: str


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def load_rows(csv_path: Path = DEFAULT_CSV_PATH, limit: int | None = None) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append(dict(row))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _row_value(row: Mapping[str, object], field: str) -> str:
    return clean_text(row.get(field))


def _candidate(
    row: Mapping[str, object],
    *,
    source_alias: str,
    candidate_type: str,
    risk_label: str,
    reason: str,
) -> FoodAliasCandidate:
    return FoodAliasCandidate(
        source_alias=clean_text(source_alias),
        target_name=_row_value(row, "name"),
        target_rep_name=_row_value(row, "rep_name") or None,
        data_type=_row_value(row, "data_type") or None,
        major_category=_row_value(row, "major_category") or None,
        candidate_type=candidate_type,
        risk_label=risk_label,
        reason=reason,
    )


def _is_compound_name(name: str) -> bool:
    return any(keyword in name for keyword in COMPOUND_KEYWORDS)


def _dedupe_candidates(
    candidates: Iterable[FoodAliasCandidate],
) -> list[FoodAliasCandidate]:
    seen: set[tuple[str, str, str, str]] = set()
    deduped: list[FoodAliasCandidate] = []
    for candidate in candidates:
        if not candidate.source_alias or not candidate.target_name:
            continue
        key = (
            candidate.source_alias,
            candidate.target_name,
            candidate.candidate_type,
            candidate.risk_label,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def discover_row_alias_candidates(row: Mapping[str, object]) -> list[FoodAliasCandidate]:
    name = _row_value(row, "name")
    rep_name = _row_value(row, "rep_name")
    data_type = _row_value(row, "data_type")
    candidates: list[FoodAliasCandidate] = []

    if not name:
        return []

    if name in PROTECTED_COMPOUND_EXAMPLES:
        candidates.append(
            _candidate(
                row,
                source_alias=name,
                candidate_type="protected_compound",
                risk_label=DO_NOT_LINK_PROTECTED_COMPOUND,
                reason="Protected compound query must not be linked to an ingredient row.",
            )
        )
    elif _is_compound_name(name):
        candidates.append(
            _candidate(
                row,
                source_alias=name,
                candidate_type="compound_food_name",
                risk_label=HIGH_RISK_COMPOUND,
                reason="Compound food keyword detected; do not use as ingredient alias without review.",
            )
        )

    if "닭고기 가슴" in name:
        alias = name.replace("닭고기 가슴(껍질 제거)", "닭가슴살")
        spaced_alias = name.replace("닭고기 가슴(껍질 제거)", "닭 가슴살")
        for source_alias in (alias, spaced_alias):
            candidates.append(
                _candidate(
                    row,
                    source_alias=source_alias,
                    candidate_type="spacing_variant",
                    risk_label=SAFE_SPACING_VARIANT,
                    reason="Chicken breast spacing/name variant for exact canonical row review.",
                )
            )

    for canonical, alias_word in SYNONYM_REVIEW_RULES:
        if canonical in name:
            risk = SAFE_EXACT_SYNONYM if data_type == "원재료성 식품" else HIGH_RISK_AMBIGUOUS_INGREDIENT
            candidates.append(
                _candidate(
                    row,
                    source_alias=name.replace(canonical, alias_word),
                    candidate_type="synonym_review",
                    risk_label=risk,
                    reason=f"Review common synonym candidate: {alias_word} -> {canonical}.",
                )
            )

    for canonical_state, prefix_state in COOKING_STATE_RULES:
        if name.endswith(f" {canonical_state}") and rep_name:
            candidates.append(
                _candidate(
                    row,
                    source_alias=f"{prefix_state} {rep_name}",
                    candidate_type="cooking_state_variant",
                    risk_label=SAFE_COOKING_STATE_VARIANT,
                    reason=f"Review cooking-state phrase order for {canonical_state}.",
                )
            )
        if canonical_state in name and name != rep_name:
            candidates.append(
                _candidate(
                    row,
                    source_alias=name.replace(canonical_state, prefix_state),
                    candidate_type="cooking_state_variant",
                    risk_label=SAFE_COOKING_STATE_VARIANT,
                    reason=f"Review shortened cooking-state spelling for {canonical_state}.",
                )
            )

    return _dedupe_candidates(candidates)


def discover_rep_name_group_candidates(rows: Sequence[Mapping[str, object]], limit: int = 40) -> list[FoodAliasCandidate]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        rep_name = _row_value(row, "rep_name")
        if rep_name:
            grouped[rep_name].append(row)

    candidates: list[FoodAliasCandidate] = []
    for rep_name, group in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(group) < 2:
            continue
        data_types = {_row_value(row, "data_type") for row in group}
        names = [_row_value(row, "name") for row in group[:5]]
        risk = REVIEW_REP_NAME_GROUP
        reason = f"rep_name group has {len(group)} rows; sample names: {', '.join(names)}"
        if "음식" in data_types and "원재료성 식품" in data_types:
            risk = HIGH_RISK_AMBIGUOUS_INGREDIENT
            reason += "; mixed dish/ingredient data types."
        candidates.append(
            FoodAliasCandidate(
                source_alias=rep_name,
                target_name=f"{rep_name} group",
                target_rep_name=rep_name,
                data_type=", ".join(sorted(dt for dt in data_types if dt)) or None,
                major_category=None,
                candidate_type="rep_name_group",
                risk_label=risk,
                reason=reason,
            )
        )
        if len(candidates) >= limit:
            break
    return candidates


def discover_alias_candidates(
    rows: Sequence[Mapping[str, object]],
    *,
    row_limit: int | None = None,
    rep_group_limit: int = 40,
) -> list[FoodAliasCandidate]:
    selected_rows = list(rows[:row_limit] if row_limit is not None else rows)
    candidates: list[FoodAliasCandidate] = []
    for row in selected_rows:
        candidates.extend(discover_row_alias_candidates(row))
    candidates.extend(discover_rep_name_group_candidates(selected_rows, limit=rep_group_limit))
    return _dedupe_candidates(candidates)


def _markdown_table(candidates: Sequence[FoodAliasCandidate], limit: int) -> str:
    rows = [
        "| source_alias | target_name | target_rep_name | data_type | major_category | candidate_type | risk_label | reason |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for candidate in candidates[:limit]:
        rows.append(
            "| "
            + " | ".join(
                _escape_markdown(value)
                for value in (
                    candidate.source_alias,
                    candidate.target_name,
                    candidate.target_rep_name or "",
                    candidate.data_type or "",
                    candidate.major_category or "",
                    candidate.candidate_type,
                    candidate.risk_label,
                    candidate.reason,
                )
            )
            + " |"
        )
    return "\n".join(rows)


def _escape_markdown(value: object) -> str:
    return clean_text(value).replace("|", "\\|")


def format_markdown_report(
    candidates: Sequence[FoodAliasCandidate],
    *,
    title: str = "Food Alias Candidate Discovery Report",
    limit: int = 80,
) -> str:
    safe_candidates = [
        candidate
        for candidate in candidates
        if candidate.risk_label.startswith("SAFE_") or candidate.risk_label == REVIEW_REP_NAME_GROUP
    ]
    high_risk_candidates = [
        candidate
        for candidate in candidates
        if candidate.risk_label.startswith("HIGH_RISK_")
        or candidate.risk_label == DO_NOT_LINK_PROTECTED_COMPOUND
    ]

    return "\n\n".join(
        [
            f"# {title}",
            (
                "이 보고서는 자동 적용 목록이 아닙니다. 자동 적용 금지. "
                "후보는 사람이 검토한 뒤 별도 PR에서 embedding alias registry와 테스트에 반영해야 합니다."
            ),
            "## Summary",
            "\n".join(
                [
                    f"- Total candidates: {len(candidates)}",
                    f"- Safe/review candidates: {len(safe_candidates)}",
                    f"- High-risk/do-not-link candidates: {len(high_risk_candidates)}",
                    "- Chroma rebuild is not part of this report.",
                    "- Runtime alias rules are not changed by this report.",
                ]
            ),
            "## Safe Or Review Candidates",
            _markdown_table(safe_candidates, limit),
            "## High-Risk Or Do-Not-Link Candidates",
            _markdown_table(high_risk_candidates, limit),
            "## Protected Compound Rule",
            (
                "`계란빵`, `볶음밥 계란`, `김밥 계란`, `샐러드 닭가슴살`, "
                "`닭가슴살 샐러드`, `샌드위치 닭가슴살` must not be linked to raw ingredient rows."
            ),
            "## Next Step",
            (
                "Select a small reviewed subset, add row-specific aliases to "
                "`FOOD_EMBEDDING_ROW_ALIASES` with tests, then run the approved Chroma rebuild plan."
            ),
        ]
    ) + "\n"


def write_report(candidates: Sequence[FoodAliasCandidate], output: Path, *, limit: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(format_markdown_report(candidates, limit=limit), encoding="utf-8")


def emit_report(candidates: Sequence[FoodAliasCandidate], stream: TextIO, *, limit: int) -> None:
    stream.write(format_markdown_report(candidates, limit=limit))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover food alias candidates without applying them.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit CSV rows read for exploration.")
    parser.add_argument("--report-limit", type=int, default=80, help="Limit rows shown per report section.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv, limit=args.limit)
    candidates = discover_alias_candidates(rows)
    if args.output:
        write_report(candidates, args.output, limit=args.report_limit)
    else:
        import sys

        emit_report(candidates, sys.stdout, limit=args.report_limit)


if __name__ == "__main__":
    main()

"""Generated food embedding text helpers.

This module prepares row-level embedding text for future food DB rebuilds. It
does not mutate source CSV rows, write files, or affect runtime search.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any


_DISPLAY_PREFIX_RE = re.compile(r"^\[[^\]]+\]\s*")


# Intentionally empty. Document-side embedding aliases were found to be a
# no-op-to-harmful in this pipeline: runtime FOOD_SEARCH_ALIAS_RULES already
# normalizes queries to canonical names before Chroma, and rep_name-based
# canonical index injection (vector_score=0.0) decides fruit/veg matches.
# Enriching document text only diluted canonical rows' vectors and broke
# exact-match (see egg-family smoke regression, rebuild rollback 2026-07-06).
# Do NOT reintroduce document aliases without re-validating against a rebuild.
FOOD_EMBEDDING_ROW_ALIASES: dict[str, list[str]] = {}


def clean_embedding_text_part(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def clean_food_embedding_key(value: Any) -> str:
    cleaned = clean_embedding_text_part(value)
    if not cleaned:
        return ""
    return _DISPLAY_PREFIX_RE.sub("", cleaned).strip()


def dedupe_text_parts(parts: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for part in parts:
        cleaned = clean_embedding_text_part(part)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped


def get_food_embedding_aliases(row: Mapping[str, Any]) -> list[str]:
    keys = dedupe_text_parts(
        clean_food_embedding_key(row.get(field))
        for field in ("name", "rep_name")
    )
    aliases: list[str] = []
    for key in keys:
        aliases.extend(FOOD_EMBEDDING_ROW_ALIASES.get(key, []))
    return dedupe_text_parts(aliases)


def build_food_embedding_text(row: Mapping[str, Any]) -> str:
    base_text = ""
    for field in ("embed_text", "name", "rep_name"):
        base_text = clean_embedding_text_part(row.get(field))
        if base_text:
            break

    parts = [base_text, *get_food_embedding_aliases(row)]
    return " | ".join(dedupe_text_parts(parts))

"""Generated food embedding text helpers.

This module prepares row-level embedding text for future food DB rebuilds. It
does not mutate source CSV rows, write files, or affect runtime search.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any


_DISPLAY_PREFIX_RE = re.compile(r"^\[[^\]]+\]\s*")


# Reviewed safe aliases only. Protected compounds and broad dish/product names
# must not be added to ingredient rows.
FOOD_EMBEDDING_ROW_ALIASES: dict[str, list[str]] = {
    "달걀 생것": ["계란", "계란 생것"],
    "달걀 삶은것": ["삶은 계란", "삶은계란", "계란 삶은것"],
    "달걀후라이": ["계란후라이"],
    "고구마 찐것": ["찐 고구마", "고구마 찐"],
    "고구마 구운것": ["구운 고구마", "고구마 구운"],
    "바나나 생것": ["바나나"],
    "사과 생것": ["사과"],
    "닭고기 가슴(껍질 제거) 생것": ["닭가슴살 생것", "닭 가슴살 생것"],
    "닭고기 가슴(껍질 제거) 삶은것": [
        "삶은 닭가슴살",
        "삶은 닭 가슴살",
        "닭가슴살 삶은것",
        "닭 가슴살 삶은것",
    ],
    "닭고기 가슴(껍질 제거) 구운것(팬)": [
        "구운 닭가슴살",
        "구운 닭 가슴살",
        "닭가슴살 구운것",
        "닭 가슴살 구운것",
    ],
}


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

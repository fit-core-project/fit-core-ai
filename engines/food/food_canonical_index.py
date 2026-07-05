"""Read-only runtime canonical food lookup index."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


_DISPLAY_PREFIX_RE = re.compile(r"^\[[^\]]+\]\s*")
_NUMERIC_FIELDS = (
    "basis_value",
    "kcal",
    "carb_g",
    "protein_g",
    "fat_g",
    "sugar_g",
    "fiber_g",
    "sodium_mg",
    "satfat_g",
)
_TEXT_FIELDS = (
    "food_id",
    "name",
    "rep_name",
    "basis_unit",
    "major_category",
    "data_type",
)


@dataclass(frozen=True)
class FoodCanonicalRecord:
    metadata: dict
    page_content: str

    def to_document(self):
        return SimpleNamespace(
            metadata=dict(self.metadata),
            page_content=self.page_content,
        )


def clean_food_index_key(value: str | None) -> str:
    if not value:
        return ""
    cleaned = " ".join(str(value).strip().split())
    return _DISPLAY_PREFIX_RE.sub("", cleaned).strip()


def _to_float(value):
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _append_unique(target: list[FoodCanonicalRecord], record: FoodCanonicalRecord):
    record_id = record.metadata.get("food_id") or (
        record.metadata.get("name"),
        record.metadata.get("rep_name"),
        record.metadata.get("data_type"),
    )
    for existing in target:
        existing_id = existing.metadata.get("food_id") or (
            existing.metadata.get("name"),
            existing.metadata.get("rep_name"),
            existing.metadata.get("data_type"),
        )
        if existing_id == record_id:
            return
    target.append(record)


class FoodCanonicalIndex:
    def __init__(self, records: list[FoodCanonicalRecord]):
        self.by_key: dict[str, list[FoodCanonicalRecord]] = {}
        for record in records:
            self._add_record(record)

    @classmethod
    def from_rows(cls, rows) -> "FoodCanonicalIndex":
        records = []
        for row in rows:
            metadata = {}
            for field in _TEXT_FIELDS:
                value = (row.get(field) or "").strip()
                if value:
                    metadata[field] = value
            for field in _NUMERIC_FIELDS:
                value = _to_float(row.get(field))
                if value is not None:
                    metadata[field] = value
            metadata.setdefault("basis_value", 100.0)
            metadata.setdefault("basis_unit", "g")
            name = metadata.get("name") or ""
            rep_name = metadata.get("rep_name") or ""
            page_content = (row.get("embed_text") or name or rep_name).strip()
            if not name and not rep_name:
                continue
            records.append(FoodCanonicalRecord(metadata=metadata, page_content=page_content))
        return cls(records)

    def _add_record(self, record: FoodCanonicalRecord):
        keys = {
            clean_food_index_key(record.metadata.get("name")),
            clean_food_index_key(record.metadata.get("rep_name")),
            clean_food_index_key(record.page_content),
        }
        for key in keys:
            if not key:
                continue
            bucket = self.by_key.setdefault(key, [])
            _append_unique(bucket, record)

    def lookup(self, query: str | None, limit: int = 5) -> list[FoodCanonicalRecord]:
        key = clean_food_index_key(query)
        if not key:
            return []
        return list(self.by_key.get(key, []))[:limit]


def load_food_canonical_index(csv_path: Path) -> FoodCanonicalIndex:
    if not csv_path.exists():
        return FoodCanonicalIndex([])
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return FoodCanonicalIndex.from_rows(csv.DictReader(handle))

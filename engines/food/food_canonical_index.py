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


_PAREN_TEXT_RE = re.compile(r"\([^()]*\)")


def _compact(value: str | None) -> str:
    return "".join((value or "").split())


def reordered_dish_key(name: str | None, rep_name: str | None) -> str:
    """MFDS names dishes "<rep> <modifier>" ("돈가스 고구마", "비빔밥 멍게"); users write the
    Korean compound "<modifier><rep>" (고구마돈가스). Key = compact(modifier) + compact(rep), or ""
    when the row is not a two-word "<rep> <modifier>" name."""
    words = _PAREN_TEXT_RE.sub(" ", name or "").split()
    rep = _compact(rep_name)
    if len(words) != 2 or not rep or words[0] != rep or len(words[1]) < 2:
        return ""
    return words[1] + rep


class FoodCanonicalIndex:
    def __init__(self, records: list[FoodCanonicalRecord]):
        self.by_key: dict[str, list[FoodCanonicalRecord]] = {}
        self.by_name: dict[str, FoodCanonicalRecord] = {}
        reorder: dict[str, list[FoodCanonicalRecord]] = {}
        for record in records:
            self._add_record(record)
            name = record.metadata.get("name") or ""
            self.by_name.setdefault(name, record)
            if (record.metadata.get("data_type") or "") == "음식":
                key = reordered_dish_key(name, record.metadata.get("rep_name"))
                if key:
                    reorder.setdefault(key, []).append(record)
        exact = {_compact(k) for k in self.by_key}
        # a name word equal to the query ("라떼 바닐라라떼 핫" for 바닐라라떼) already names it
        exact |= {w for r in records for w in _PAREN_TEXT_RE.sub(" ", r.metadata.get("name") or "").split()}
        # unique, and never shadowing a row that already carries that compact name
        self.reordered: dict[str, FoodCanonicalRecord] = {
            key: rows[0] for key, rows in reorder.items()
            if key not in exact and len({r.metadata.get("name") for r in rows}) == 1
        }

    def lookup_reordered(self, query: str | None) -> FoodCanonicalRecord | None:
        return self.reordered.get(_compact(clean_food_index_key(query)))

    def lookup_name(self, name: str | None) -> FoodCanonicalRecord | None:
        return self.by_name.get(name or "")

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

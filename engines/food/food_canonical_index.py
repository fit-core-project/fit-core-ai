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
# produce whose raw row is what a bare name means (한라봉 = 귤 한라봉 생것); a bare 삼겹살 is
# eaten cooked, so meat / grain varieties are not indexed
_EATEN_RAW_CATEGORIES = frozenset({"과일류", "채소류"})


def _compact(value: str | None) -> str:
    return "".join((value or "").split())


# Parenthetical words that qualify a row instead of naming it: origin (국내산, 수입산, 일본),
# count / form (조각, 낱개, 마른것) and "X만" (멸치볶음(멸치만): only the anchovies).
_ALIAS_QUALIFIER_RE = re.compile(r"^(?:.+산|일본|중국|미국|조각|낱개|마른것|.+만)$")


# Second words of raw produce rows that name a colour, plant part, growing method or grade
# (파프리카 빨간색 생것, 무 잎줄기 생것, 부추 재래종 생것), not a variety someone eats by name.
_VARIETY_QUALIFIER_RE = re.compile(
    r"^(?:.+색|.+종|.+질|껍질|과육|뿌리|.*줄기|열매|어린잎|전체|구근|야생|재배|유기재배|하우스|노지|미숙|완숙"
    r"|그린|골드|화이트|레드)$"
)


def _dish_alias_keys(name: str) -> set[str]:
    """Other names of a plain dish row given as its closing parenthetical: 완자전 소고기(동그랑땡/
    육원전) -> 동그랑땡, 육원전; 개피떡(바람떡) -> 바람떡. Only a short (one or two word) dish name
    whose single parenthetical ends the name: in a branded menu row (피자 차슈(불갈비) 피자 (M),
    케이크 ...(선샤인) 6호, 아이스크림 미니 스노우모찌(그린티 스트로베리 망고)) the parenthetical is
    a topping, flavour, size or brand word. Each "/"-separated alias is one key, never a single
    word of a multi-word alias (제육(돼지고기 수육) is not 돼지고기)."""
    parens = re.findall(r"\(([^()]*)\)", name)
    text = name.rstrip()
    if len(parens) != 1 or not text.endswith(")"):
        return set()
    before = text[:text.rindex("(")]
    if not before or before.endswith(" ") or len(before.split()) > 2:
        return set()
    keys = set()
    for alias in re.split(r"[/,·]+", parens[0]):
        key = _compact(alias)
        if len(key) >= 2 and re.fullmatch(r"[가-힣]+", key) and not _ALIAS_QUALIFIER_RE.match(key):
            keys.add(key)
    return keys


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

        # lexical retrieval keys for rows the embedding search misses:
        # - a plain dish's parenthetical alias: 완자전 소고기(동그랑땡/육원전) for 동그랑땡
        # - the variety word of a raw fruit / vegetable row: 귤 한라봉 생것 for 한라봉
        self.lexical: dict[str, list[FoodCanonicalRecord]] = {}
        generic_names = {_compact(r.metadata.get("rep_name")) for r in records} | {
            _compact(_PAREN_TEXT_RE.sub(" ", r.metadata.get("name") or "")) for r in records}
        for record in records:
            name = record.metadata.get("name") or ""
            keys = set()
            raw = (record.metadata.get("data_type") or "").startswith("원재료")
            # raw cuts name their alias in parentheses too (돼지고기 삼겹살(삼겹살) 생것), but a
            # bare 삼겹살 is eaten cooked: aliases only index prepared dishes
            if not raw:
                # an alias that is itself a food the DB lists (토스트(식빵), 쑥전(부침개)) names a
                # broader food: the regular lookup finds that food's own rows
                keys.update(k for k in _dish_alias_keys(name) if k not in generic_names)
            if raw and record.metadata.get("major_category") in _EATEN_RAW_CATEGORIES:
                words = _PAREN_TEXT_RE.sub(" ", name).split()
                if (len(words) == 3 and words[2] == "생것" and len(words[1]) >= 2
                        and re.fullmatch(r"[가-힣]+", words[1]) and not _VARIETY_QUALIFIER_RE.match(words[1])):
                    keys.add(words[1])
            for key in keys:
                _append_unique(self.lexical.setdefault(key, []), record)

    def lookup_lexical(self, query: str | None, limit: int = 5) -> list[FoodCanonicalRecord]:
        return list(self.lexical.get(_compact(clean_food_index_key(query)), []))[:limit]

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

"""Generate the checked-in fallback for unrepresentable source equipment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path


SCHEMA_VERSION = "fit-core-equipment-provenance-snapshot-v1"
SOURCE_HEADERS = ("v24_row_uid", "name", "english", "equipment")
UNREPRESENTED = {"ez바", "짐볼", "swissball"}


def compact(value: object) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("/", "")
    )


def identity_hash(exercise_id: str, name_kr: str, name_en: str) -> str:
    payload = json.dumps(
        {"exerciseId": exercise_id, "nameEn": name_en, "nameKr": name_kr},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def records_hash(records: list[dict]) -> str:
    payload = json.dumps(
        records,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_snapshot(source: Path) -> dict:
    raw = source.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if not reader.fieldnames or not set(SOURCE_HEADERS).issubset(reader.fieldnames):
        raise ValueError("source CSV is missing required provenance columns")
    records = []
    source_ids = set()
    for row in reader:
        exercise_id = str(row.get("v24_row_uid") or "").strip()
        if not exercise_id or exercise_id in source_ids:
            raise ValueError("source rows require unique non-empty v24_row_uid values")
        source_ids.add(exercise_id)
        equipment = str(row.get("equipment") or "").strip()
        if compact(equipment) not in UNREPRESENTED:
            continue
        name_kr = str(row.get("name") or "").strip()
        name_en = str(row.get("english") or "").strip()
        if not name_kr or not name_en:
            raise ValueError("source snapshot rows require non-empty identities")
        records.append(
            {
                "exerciseId": exercise_id,
                "nameKr": name_kr,
                "nameEn": name_en,
                "sourceEquipment": equipment,
                "identitySha256": identity_hash(exercise_id, name_kr, name_en),
            }
        )
    records.sort(key=lambda item: item["exerciseId"])
    return {
        "schemaVersion": SCHEMA_VERSION,
        "sourceSha256": hashlib.sha256(raw).hexdigest(),
        "sourceSchema": list(SOURCE_HEADERS),
        "recordCount": len(records),
        "recordsSha256": records_hash(records),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    snapshot = build_snapshot(args.source)
    args.output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

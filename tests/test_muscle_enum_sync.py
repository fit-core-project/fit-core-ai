from pathlib import Path

import pandas as pd

from engines.routine_engine import (
    MUSCLE_SLUGS,
    DomEntry,
    MUSCLE_REGISTRY,
    SPLIT_LABEL_TO_MUSCLES,
    get_mapped_targets,
    map_doms_to_db,
)


def _exercise_tier_enums() -> set[str]:
    source = Path(__file__).resolve().parents[1] / "scripts" / "exercise_tier.xlsx"
    df = pd.read_excel(source, sheet_name=0)
    enums = set(df["primary_muscle"].dropna().astype(str).str.strip())
    for raw in df["secondary_muscle"].dropna().astype(str):
        enums.update(part.strip() for part in raw.split(",") if part.strip())
    return enums


def test_db_muscle_enum_constants_match_exercise_tier_ssot():
    assert MUSCLE_SLUGS == _exercise_tier_enums()


def test_registry_covers_every_db_muscle_enum():
    registry_enums = {enum for mapping in MUSCLE_REGISTRY.values() for enum in mapping.db_enums}
    assert registry_enums == MUSCLE_SLUGS


def test_split_labels_cover_every_db_muscle_enum():
    split_enums = {enum for enums in SPLIT_LABEL_TO_MUSCLES.values() for enum in enums}
    assert split_enums == MUSCLE_SLUGS


def test_frontend_chest_slug_preserves_spreadsheet_slug():
    _, db_enums = get_mapped_targets(["chest"])
    assert db_enums == ["chest"]


def test_direct_db_enum_inputs_are_normalized_and_preserved():
    _, db_enums = get_mapped_targets(["chest"])
    assert db_enums == ["chest"]

    doms = map_doms_to_db([DomEntry(body_part="glutes", level="mild")])
    assert doms == {"gluteal": 1}


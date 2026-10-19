from pathlib import Path

import pandas as pd

from engines.muscle_mapping import (
    get_mapped_targets,
    map_doms_to_db,
    resolve_request_target_muscles,
    resolve_request_target_split_label,
)
from engines.registry import (
    INTERNAL_CANONICAL_MUSCLE_SLUGS,
    MUSCLE_SLUGS,
    MUSCLE_REGISTRY,
    SPLIT_LABEL_TO_MUSCLES,
)
from engines.schemas import DomEntry


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
    assert MUSCLE_SLUGS <= split_enums
    assert split_enums <= INTERNAL_CANONICAL_MUSCLE_SLUGS
    assert "side-deltoids" in SPLIT_LABEL_TO_MUSCLES["shoulder"]


def test_frontend_chest_slug_preserves_spreadsheet_slug():
    _, db_enums = get_mapped_targets(["chest"])
    assert db_enums == ["chest"]


def test_direct_db_enum_inputs_are_normalized_and_preserved():
    _, db_enums = get_mapped_targets(["chest"])
    assert db_enums == ["chest"]

    doms = map_doms_to_db([DomEntry(body_part="gluteal", level="mild")])
    assert doms == {"gluteal": 1}


def test_target_and_doms_values_are_not_alias_mapped():
    _, db_enums = get_mapped_targets(["glutes", "knees"])
    assert db_enums == ["glutes", "knees"]

    doms = map_doms_to_db([DomEntry(body_part="knees", level="severe")])
    assert doms == {"knees": 3}


def test_custom_split_uses_explicit_target_muscles():
    assert resolve_request_target_muscles(
        "custom",
        ["chest", "upper-back", "chest"],
    ) == ["chest", "upper-back"]


def test_explicit_targets_override_a_stale_preset_label():
    assert resolve_request_target_muscles("push", ["quadriceps"]) == ["quadriceps"]
    assert resolve_request_target_split_label("push", ["quadriceps"]) == "custom"


def test_edited_push_targets_do_not_restore_chest():
    explicit = ["front-deltoids", "triceps"]
    assert resolve_request_target_muscles("push", explicit) == explicit
    assert resolve_request_target_split_label("push", explicit) == "custom"


def test_complete_frontend_push_preset_keeps_push_semantics():
    explicit = ["chest", "front-deltoids", "triceps"]
    assert resolve_request_target_muscles("push", explicit) == explicit
    assert resolve_request_target_split_label("push", explicit) == "push"


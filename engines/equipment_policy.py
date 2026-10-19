"""Deterministic equipment semantics for filtering and execution grouping."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional


EQUIPMENT_POLICY_VERSION = "routine-equipment-semantics-v5-2026-09-14"
KNOWN_EQUIPMENT_CODES = {
    "BAND",
    "BODYWEIGHT",
    "KETTLEBELL",
    "DUMBBELL",
    "MACHINE",
    "BARBELL",
    "CABLE",
    "LANDMINE",
    "SMITH_MACHINE",
    "PLATE",
}
UNSUPPORTED_EQUIPMENT_CODE = "UNKNOWN"

# These source equipment values are explicit, but have no selectable code in
# the current equipment contract. Keep the source fact and fail closed rather
# than treating a stale BODYWEIGHT value as an equipment-free exercise.
_UNREPRESENTED_SOURCE_EQUIPMENT = {"ez바", "짐볼", "swissball"}
_VERIFIED_SOURCE_SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "provenance"
    / "equipment-provenance-snapshot-v1-2026-09-14.json"
)
_SOURCE_SNAPSHOT_SCHEMA_VERSION = "fit-core-equipment-provenance-snapshot-v1"
_SOURCE_SNAPSHOT_SOURCE_SCHEMA = ["v24_row_uid", "name", "english", "equipment"]
_SUPPORTED_SOURCE_SHA256 = {
    "22e7b98199b3ee8b1d4e880e88fe76a5ad7a8dc14430c7f94ed0143f4f234ff2",
}
_SUPPORTED_SOURCE_RECORD_COUNT = 168
_SUPPORTED_SOURCE_RECORDS_SHA256 = "12f84cab17475234f947415a6d56232ebbe0db168b20735b20443a07dfbf0199"

_OPTION_PROFILES = {
    "BAND": ("resistance_band", "open_floor", "portable", "quick"),
    "BODYWEIGHT": ("bodyweight", "open_floor", "portable", "quick"),
    "DUMBBELL": ("free_weight", "free_weight_area", "portable", "quick"),
    "KETTLEBELL": ("free_weight", "free_weight_area", "portable", "quick"),
    "PLATE": ("free_weight", "free_weight_area", "portable", "quick"),
    "BARBELL": ("free_weight", "barbell_station", "semi_portable", "moderate"),
    "CABLE": ("cable_machine", "cable_station", "fixed", "moderate"),
    "LANDMINE": ("free_weight", "landmine_station", "fixed", "moderate"),
    "MACHINE": ("machine", "generic_machine", "fixed", "slow"),
    "SMITH_MACHINE": ("machine", "smith_station", "fixed", "slow"),
}
_FREE_WEIGHT_SUPERSET_CODES = {
    "BAND",
    "BODYWEIGHT",
    "DUMBBELL",
    "KETTLEBELL",
    "PLATE",
    "BARBELL",
}
_SAME_STATION_COMPATIBLE_CODES = {
    "BAND",
    "BODYWEIGHT",
    "DUMBBELL",
    "KETTLEBELL",
    "PLATE",
    "BARBELL",
    "CABLE",
    "LANDMINE",
    "SMITH_MACHINE",
}


def normalize_equipment_codes(values: Iterable[str]) -> set[str]:
    return {
        str(value).strip().upper()
        for value in values
        if str(value or "").strip()
    }


def parse_equipment_options(raw: Optional[str]) -> tuple[str, ...]:
    """Return actual alternative equipment options in stable source order.

    The legacy pair ``SMITH_MACHINE, MACHINE`` describes one Smith exercise and
    its broad category. It is not an either/or choice.
    """

    options: list[str] = []
    for value in str(raw or "").split(","):
        code = value.strip().upper()
        if not code or code in options:
            continue
        options.append(code)
    if "SMITH_MACHINE" in options and "MACHINE" in options:
        options.remove("MACHINE")
    return tuple(options)


def _compact(value: object) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("/", "")
    )


def _identity_hash(exercise_id: str, name_kr: str, name_en: str) -> str:
    payload = json.dumps(
        {
            "exerciseId": exercise_id,
            "nameEn": name_en,
            "nameKr": name_kr,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _records_hash(records: list[dict]) -> str:
    payload = json.dumps(
        records,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _verified_source_snapshot_by_id() -> dict[str, dict[str, str]]:
    """Load the generated fallback only when its schema and identities verify."""

    try:
        payload = json.loads(_VERIFIED_SOURCE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("schemaVersion") != _SOURCE_SNAPSHOT_SCHEMA_VERSION
            or payload.get("sourceSha256") not in _SUPPORTED_SOURCE_SHA256
            or payload.get("sourceSchema") != _SOURCE_SNAPSHOT_SOURCE_SCHEMA
            or not isinstance(payload.get("records"), list)
            or payload.get("recordCount") != len(payload["records"])
            or payload.get("recordCount") != _SUPPORTED_SOURCE_RECORD_COUNT
            or payload.get("recordsSha256") != _SUPPORTED_SOURCE_RECORDS_SHA256
            or _records_hash(payload["records"]) != _SUPPORTED_SOURCE_RECORDS_SHA256
        ):
            return {}
        rows: dict[str, dict[str, str]] = {}
        for record in payload["records"]:
            if not isinstance(record, dict):
                return {}
            exercise_id = str(record.get("exerciseId") or "").strip()
            name_kr = str(record.get("nameKr") or "").strip()
            name_en = str(record.get("nameEn") or "").strip()
            equipment = str(record.get("sourceEquipment") or "").strip()
            if (
                not exercise_id
                or not name_kr
                or not name_en
                or _compact(equipment) not in _UNREPRESENTED_SOURCE_EQUIPMENT
                or exercise_id in rows
                or record.get("identitySha256") != _identity_hash(exercise_id, name_kr, name_en)
            ):
                return {}
            rows[exercise_id] = {
                "nameKr": name_kr,
                "nameEn": name_en,
                "sourceEquipment": equipment,
            }
        return rows
    except (OSError, TypeError, ValueError, UnicodeError):
        return {}


def verified_source_snapshot_available() -> bool:
    return bool(_verified_source_snapshot_by_id())


def source_equipment_identity_matches(
    *,
    source_name_kr: object,
    source_name_en: object,
    catalog_name_kr: object,
    catalog_name_en: object,
) -> bool:
    """Require both source identity fields before applying source equipment."""

    return (
        bool(_compact(source_name_kr))
        and bool(_compact(source_name_en))
        and _compact(source_name_kr) == _compact(catalog_name_kr)
        and _compact(source_name_en) == _compact(catalog_name_en)
    )


def canonical_equipment_req(
    raw: Optional[str],
    *,
    name_kr: object = None,
    name_en: object = None,
    detail_subpart: object = None,
    source_equipment: object = None,
) -> str:
    """Correct catalog equipment only when the movement name is unambiguous.

    Correct only explicit catalog evidence. Named band movements and the
    leg-press family retain their existing rules. Source equipment that the
    public contract cannot represent is marked UNKNOWN, so it cannot silently
    become BODYWEIGHT.
    """

    text = _compact(f"{name_kr or ''} {name_en or ''}")
    detail = _compact(detail_subpart)
    raw_value = str(raw or "").strip()
    raw_compact = _compact(raw_value)
    source_compact = _compact(source_equipment)
    if source_compact in _UNREPRESENTED_SOURCE_EQUIPMENT:
        # The source requires equipment outside the public contract. A stale
        # selectable value (including BAND) is an unresolved conflict, not an
        # approved alternative.
        return UNSUPPORTED_EQUIPMENT_CODE
    if raw_value.upper() == "BODYWEIGHT" and (
        "밴드" in text
        or "band" in text
        or raw_compact in {"밴드", "band"}
    ):
        return "BAND"

    if any(
        token in text
        for token in ("어시스트풀업", "assistedpullup", "보조풀업", "assistedchinup")
    ):
        # An assisted pull-up needs a machine, band, or another explicit assist
        # device. The current public contract has no pull-up-bar code and this
        # catalog row does not identify a band, so keep it out of equipment-free
        # and home dumbbell/bodyweight routines.
        return "MACHINE"

    leg_press_family = any(
        token in text
        for token in ("레그프레스", "legpress", "브이스쿼트", "핵스쿼트", "hacksquat")
    )
    machine_press_pattern = (
        "프레스스쿼트" in text
        or "legpresssquat" in text
        or detail == "머신프레스"
    )
    if leg_press_family and machine_press_pattern:
        if "스미스머신" in text or "smithmachine" in text:
            return "SMITH_MACHINE, MACHINE"
        return "MACHINE"
    return raw_value


def verified_source_equipment_fallback(
    exercise_id: object,
    *,
    name_kr: object = None,
    name_en: object = None,
) -> Optional[str]:
    """Return source equipment only when the generated snapshot identity matches."""

    record = _verified_source_snapshot_by_id().get(str(exercise_id or "").strip())
    if not record:
        return None
    if (
        _compact(name_kr) != _compact(record["nameKr"])
        or _compact(name_en) != _compact(record["nameEn"])
    ):
        return None
    return record["sourceEquipment"]


def select_equipment_option(
    raw: Optional[str],
    blocked: Iterable[str] = (),
    available: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """Choose one usable option or return ``None`` when none is feasible."""

    options = parse_equipment_options(raw)
    if not options:
        return None

    options = tuple(code for code in options if code in KNOWN_EQUIPMENT_CODES)
    if not options:
        return None

    blocked_codes = normalize_equipment_codes(blocked) - {"BODYWEIGHT"}
    viable = [code for code in options if code not in blocked_codes]
    if not viable:
        return None

    available_codes = normalize_equipment_codes(available or ())
    if not available_codes:
        return viable[0]
    for code in viable:
        if code == "BODYWEIGHT" or code in available_codes:
            return code
    return None


def equipment_option_profile(
    exercise_id: str,
    equipment_code: str,
    *,
    source_type: str = "derivedFromExerciseTierEquipmentReq",
) -> dict:
    code = str(equipment_code or "").strip().upper()
    category, station, portability, transition = _OPTION_PROFILES.get(
        code,
        ("unknown", "unknown", "unknown", "unknown"),
    )
    return {
        "exercise_id": str(exercise_id),
        "equipment_code": code,
        "equipment_category": category,
        "station_class": station,
        "portability": portability,
        "setup_transition_class": transition,
        "source_type": source_type,
        "confidence": "high" if code in KNOWN_EQUIPMENT_CODES else "low",
        "review_required": int(code not in KNOWN_EQUIPMENT_CODES),
        "version": EQUIPMENT_POLICY_VERSION,
    }


def equipment_option_profiles(
    exercise_id: str,
    raw: Optional[str],
    *,
    name_kr: object = None,
    name_en: object = None,
    detail_subpart: object = None,
    source_equipment: object = None,
) -> list[dict]:
    canonical = canonical_equipment_req(
        raw,
        name_kr=name_kr,
        name_en=name_en,
        detail_subpart=detail_subpart,
        source_equipment=source_equipment,
    )
    source_type = (
        "deterministicCatalogEquipmentCorrection"
        if canonical != str(raw or "").strip()
        else "derivedFromExerciseTierEquipmentReq"
    )
    return [
        equipment_option_profile(exercise_id, code, source_type=source_type)
        for code in parse_equipment_options(canonical)
    ]


def superset_equipment_compatible(first: Optional[str], second: Optional[str]) -> bool:
    """Conservative default when exact gym station identity is unavailable."""

    first_code = str(first or "").strip().upper()
    second_code = str(second or "").strip().upper()
    if not first_code or not second_code:
        return False
    if first_code in _FREE_WEIGHT_SUPERSET_CODES and second_code in _FREE_WEIGHT_SUPERSET_CODES:
        return True
    if first_code == second_code:
        return first_code in _SAME_STATION_COMPATIBLE_CODES
    return False

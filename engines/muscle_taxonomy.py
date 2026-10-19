"""Versioned internal muscle taxonomy derived without overwriting source data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional


MUSCLE_TAXONOMY_VERSION = "routine-muscle-taxonomy-v4-2026-07-30"
EXACT_ID_TAXONOMY_VERSION = "routine-taxonomy-exact-id-v1-2026-10-08"
# 004724 retains its existing provenance-checked forearm admission correction.
EXACT_ID_PRIMARY_CORRECTIONS = {
    "V24ROW-004721": ("Wrist Roller", "upper-back", "forearm"),
    "V24ROW-004722": ("Plate Pinch Hold", "upper-back", "forearm"),
    "V24ROW-004723": ("Hand Gripper", "upper-back", "forearm"),
    "V24ROW-006126": ("Dumbbell Wrist Roller", "upper-back", "forearm"),
    "V24ROW-006135": ("Barbell Wrist Roller", "upper-back", "forearm"),
    "V24ROW-006146": ("Cable Wrist Roller", "upper-back", "forearm"),
    "V24ROW-006155": ("Band Wrist Roller", "upper-back", "forearm"),
    "V24ROW-006164": ("Plate Wrist Roller", "upper-back", "forearm"),
    "V24ROW-006176": ("Kettlebell Wrist Roller", "upper-back", "forearm"),
    "V24ROW-006187": ("Machine Wrist Roller", "upper-back", "forearm"),
    "V24ROW-004709": ("Banded Lateral Walk", "adductor", "abductors"),
    "V24ROW-004710": ("Clamshell", "adductor", "abductors"),
    "V24ROW-004711": ("Hip Airplane", "adductor", "abductors"),
}


def exact_id_primary_taxonomy_patch(
    catalog: Mapping[str, Any], source: Optional[Mapping[str, Any]],
) -> dict:
    """Correct reviewed primary labels only; no safety or admission promotion."""
    correction = EXACT_ID_PRIMARY_CORRECTIONS.get(str(catalog.get("catalog_exercise_id") or ""))
    if correction is None or not source:
        return {}
    english, original, corrected = correction
    if (
        catalog.get("catalog_name_en") != english
        or source.get("name_en") != english
        or source.get("name_kr") != catalog.get("catalog_name_kr")
        or catalog.get("catalog_primary_muscle") != original
        or catalog.get("canonical_primary_muscle") != original
        or catalog.get("taxonomy_review_status") != "sourceRetained"
    ):
        return {}
    return {
        "primary_muscle": corrected,
        "canonical_primary_muscle": corrected,
        "core_taxonomy_original_primary_muscle": original,
        "core_taxonomy_review_version": EXACT_ID_TAXONOMY_VERSION,
    }

DELTOID_CANONICAL_SLUGS = {
    "front-deltoids",
    "side-deltoids",
    "back-deltoids",
}


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


def _split_tokens(value: object) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def recommendation_taxonomy_is_eligible(candidate: Mapping[str, Any]) -> bool:
    """Keep unresolved taxonomy visible in the catalog but out of recommendations.

    Legacy candidates without taxonomy metadata remain eligible so an unavailable
    local enrichment DB does not disable the existing operational catalog.
    """

    if "taxonomy_review_required" in candidate:
        return not _is_truthy(candidate.get("taxonomy_review_required"))
    if "review_required" in candidate:
        return not _is_truthy(candidate.get("review_required"))
    return True


def _replace_source_token(
    value: object,
    source_primary: str,
    canonical_primary: str,
) -> str:
    tokens = _split_tokens(value)
    if source_primary != canonical_primary:
        tokens = [canonical_primary if token == source_primary else token for token in tokens]
    return ", ".join(dict.fromkeys(tokens))


@dataclass(frozen=True)
class MuscleTaxonomyDecision:
    source_primary_muscle: str
    canonical_primary_muscle: str
    source_secondary_muscles: str
    canonical_secondary_muscles: str
    source_pain_triggers: str
    canonical_pain_triggers: str
    classification_rule_code: str
    taxonomy_version: str
    source_type: str
    confidence: str
    review_status: str
    review_required: int

    def to_dict(self) -> dict:
        return asdict(self)


def classify_exercise_muscle_taxonomy(
    *,
    name_kr: object,
    name_en: object,
    source_primary_muscle: object,
    source_secondary_muscles: object = None,
    source_pain_triggers: object = None,
    detail_subpart: object = None,
) -> MuscleTaxonomyDecision:
    """Return a conservative canonical classification and preserve every source field.

    Explicit front/rear/lateral raise wording is sufficiently bounded for automatic
    classification. Generic shoulder raises without a direction remain unchanged
    and are sent to the review queue.
    """

    source_primary = str(source_primary_muscle or "").strip()
    combined = _compact(f"{name_kr or ''} {name_en or ''}")
    detail = _compact(detail_subpart)
    canonical_primary = source_primary
    rule_code = "source_primary_retained"
    confidence = "high"
    review_status = "sourceRetained"
    review_required = 0

    is_front_raise = any(
        token in combined
        for token in (
            "프론트레이즈",
            "frontraise",
            "프론t레이즈",
            "fronttraise",
        )
    )
    is_shoulder_press = "숄더프레스" in combined or "shoulderpress" in combined
    is_lateral_raise = any(
        token in combined
        for token in ("레터럴레이즈", "lateralraise", "사이드레이즈", "sideraise")
    )
    is_rear_deltoid = any(
        token in combined
        for token in (
            "리어델트",
            "reardelt",
            "후면삼각",
            "리버스펙덱",
            "reversepecdeck",
            "리버스플라이",
            "reversefly",
            "페이스풀",
            "facepull",
            "밴드풀어파트",
            "bandpullapart",
            "파월레이즈",
            "powellraise",
        )
    )
    is_prone_generic_lateral = is_lateral_raise and any(
        token in combined for token in ("엎드림", "prone")
    ) and not any(token in combined for token in ("사이드레터럴", "sidelateral"))
    is_wrist_curl = any(
        token in combined
        for token in (
            "리스트컬",
            "wristcurl",
            "손목컬",
        )
    )
    is_shrug = any(
        token in combined
        for token in (
            "슈러그",
            "shrug",
        )
    )

    if is_wrist_curl:
        canonical_primary = "forearm"
        rule_code = "explicit_wrist_curl"
        review_status = "autoClassified"
    elif is_shrug:
        canonical_primary = "trapezius"
        rule_code = "explicit_shrug"
        review_status = "autoClassified"
    elif is_rear_deltoid:
        canonical_primary = "back-deltoids"
        rule_code = "explicit_rear_deltoid_movement"
        review_status = "autoClassified"
    elif is_front_raise or is_shoulder_press:
        canonical_primary = "front-deltoids"
        rule_code = "explicit_front_raise"
        review_status = "autoClassified"
    elif is_prone_generic_lateral:
        canonical_primary = "back-deltoids"
        rule_code = "prone_generic_lateral_raise"
        review_status = "autoClassified"
    elif is_lateral_raise:
        canonical_primary = "side-deltoids"
        rule_code = "explicit_lateral_raise"
        review_status = "autoClassified"
    elif source_primary in {"front-deltoids", "back-deltoids"} and (
        "레이즈" in combined or "raise" in combined or "삼각" in detail
    ):
        rule_code = "ambiguous_deltoid_raise_source_retained"
        confidence = "low"
        review_status = "reviewRequired"
        review_required = 1

    canonical_secondary = _replace_source_token(
        source_secondary_muscles,
        source_primary,
        canonical_primary,
    )
    canonical_pain = _replace_source_token(
        source_pain_triggers,
        source_primary,
        canonical_primary,
    )
    return MuscleTaxonomyDecision(
        source_primary_muscle=source_primary,
        canonical_primary_muscle=canonical_primary,
        source_secondary_muscles=str(source_secondary_muscles or "").strip(),
        canonical_secondary_muscles=canonical_secondary,
        source_pain_triggers=str(source_pain_triggers or "").strip(),
        canonical_pain_triggers=canonical_pain,
        classification_rule_code=rule_code,
        taxonomy_version=MUSCLE_TAXONOMY_VERSION,
        source_type="deterministicNameAndSourceRule",
        confidence=confidence,
        review_status=review_status,
        review_required=review_required,
    )


def source_query_muscles_for_canonical_targets(targets: Iterable[str]) -> list[str]:
    """Expand only the source query, not the public/internal canonical contract."""

    normalized = [str(target).strip() for target in targets if str(target).strip()]
    expanded = list(dict.fromkeys(normalized))
    if set(normalized) & DELTOID_CANONICAL_SLUGS:
        for source_slug in ("front-deltoids", "back-deltoids"):
            if source_slug not in expanded:
                expanded.append(source_slug)
    # The source catalog groups wrist curls and shrugs under upper-back. The
    # canonical layer corrects those rows, so the source query must include the
    # legacy bucket before the canonical post-filter runs.
    if set(normalized) & {"forearm", "trapezius"} and "upper-back" not in expanded:
        expanded.append("upper-back")
    for _, original, corrected in EXACT_ID_PRIMARY_CORRECTIONS.values():
        if corrected in normalized and original not in expanded:
            expanded.append(original)
    return expanded


def filter_candidates_to_canonical_targets(
    candidates: Iterable[dict],
    targets: Iterable[str],
) -> list[dict]:
    target_set = {str(target).strip() for target in targets if str(target).strip()}
    if not target_set:
        return []
    return [
        candidate
        for candidate in candidates
        if str(candidate.get("primary_muscle") or "").strip() in target_set
    ]

"""Deterministic experience-level role eligibility for routine composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from .adjunct_admission import reviewed_for_purpose


EXPERIENCE_ROLE_POLICY_VERSION = "routine-experience-role-v3-2026-07-27"
BEGINNER_EXPERIENCE_ALIASES = frozenset({"beginner", "novice", "newbie", "entry", "초보", "초보자", "입문"})
_IDENTITY_REVIEWED_MAIN_ID = "V24ROW-005748"
_IDENTITY_REVIEWED_MAIN_VERSION = "shoulder-main-admission-v1-2026-09-14"
_IDENTITY_REVIEWED_MAIN_SOURCE_SHA256 = (
    "22e7b98199b3ee8b1d4e880e88fe76a5ad7a8dc14430c7f94ed0143f4f234ff2"
)


def _level(value: Any) -> str:
    return str(value or "").strip().lower()


def _rank(value: Any) -> int:
    return {
        "": 0,
        "unknown": 0,
        "low": 1,
        "fair": 1,
        "medium": 2,
        "moderate": 2,
        "good": 2,
        "high": 3,
        "excellent": 3,
        "elite": 4,
    }.get(_level(value), 0)


def _optional_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    normalized = _level(value)
    if normalized in {"true", "yes", "y", "on", "allowed"}:
        return True
    if normalized in {"false", "no", "n", "off", "blocked"}:
        return False
    return None


def _optional_int(value: Any) -> int | None:
    try:
        return None if value in {None, ""} else int(value)
    except (TypeError, ValueError):
        return None


def _goal_effect_field(goal: str | None) -> str:
    normalized = str(goal or "hypertrophy").strip().lower().replace("_", "")
    return {
        "strength": "strength_effect",
        "hypertrophy": "hypertrophy_effect",
        "recomposition": "hypertrophy_effect",
        "fatloss": "stimulus_to_fatigue",
        "endurance": "stimulus_to_fatigue",
        "generalfitness": "stimulus_to_fatigue",
    }.get(normalized, "hypertrophy_effect")


@dataclass(frozen=True)
class ExperienceRoleDecision:
    eligible: bool
    reason_code: str
    policy_version: str = EXPERIENCE_ROLE_POLICY_VERSION


def automatic_recommendation_eligibility(
    candidate: dict, experience_level: str | None, *, adjunct_options=(),
) -> ExperienceRoleDecision:
    """Apply explicit admission denials independently of main/accessory roles.

    Missing extension flags retain the legacy boundary. Risk, familiarity and
    main-lift role thresholds are deliberately not global exclusion criteria.
    Only exact reviewed purpose IDs bypass catalog denial; beginner denial remains.
    """
    for field, reason in (
        ("main_routine_default_allowed", "automatic_catalog_denied"),
        ("beginner_default_allowed", "automatic_beginner_denied"),
    ):
        if field == "beginner_default_allowed" and _level(experience_level) not in BEGINNER_EXPERIENCE_ALIASES:
            continue
        value = candidate.get(field)
        if _optional_bool(value) is False or _level(value) == "0":
            if field == "main_routine_default_allowed" and reviewed_for_purpose(candidate, adjunct_options):
                continue
            return ExperienceRoleDecision(False, reason, "automatic-admission-v1-2026-10-07")
    return ExperienceRoleDecision(True, "automatic_admission_allowed", "automatic-admission-v1-2026-10-07")


def main_role_eligibility(
    candidate: dict,
    experience_level: str | None,
    goal: str | None,
) -> ExperienceRoleDecision:
    """Decide whether a candidate may occupy a main-lift blueprint slot.

    Missing extension metadata preserves compatibility with legacy test/remote
    candidate stores. The active 24k catalog is fully enriched, so production
    candidates take the explicit branch below.
    """

    experience = _level(experience_level) or "intermediate"
    metadata_present = "main_routine_default_allowed" in candidate
    if not metadata_present:
        return ExperienceRoleDecision(True, "legacy_metadata_missing_allow")

    main_default = _optional_bool(candidate.get("main_routine_default_allowed")) is True
    beginner_default = _optional_bool(candidate.get("beginner_default_allowed")) is True
    familiarity = _level(candidate.get("exercise_familiarity_tier"))
    mainstream_score = _optional_int(candidate.get("mainstream_gym_score")) or 0
    technical_rank = _rank(candidate.get("technical_difficulty"))
    failure_rank = _rank(candidate.get("failure_penalty"))
    caution_rank = _rank(candidate.get("injury_caution_level"))
    identity_reviewed_main = all(
        (
            str(candidate.get("id") or "").strip() == _IDENTITY_REVIEWED_MAIN_ID,
            str(candidate.get("catalog_exercise_id") or "").strip()
            == _IDENTITY_REVIEWED_MAIN_ID,
            candidate.get("core_admission_review_version")
            == _IDENTITY_REVIEWED_MAIN_VERSION,
            candidate.get("core_admission_source_sha256")
            == _IDENTITY_REVIEWED_MAIN_SOURCE_SHA256,
            _optional_bool(candidate.get("core_main_role_reviewed")) is True,
            _optional_bool(
                candidate.get("catalog_original_main_routine_default_allowed")
            )
            is False,
        )
    )

    if experience == "beginner":
        eligible = (
            main_default
            and beginner_default
            and familiarity in {"common", "standard"}
            and mainstream_score >= 65
            and technical_rank <= 2
            and failure_rank <= 2
            and caution_rank <= 2
        )
        return ExperienceRoleDecision(
            eligible,
            "beginner_stable_main_eligible" if eligible else "beginner_main_role_gate",
        )

    if (
        identity_reviewed_main
        and main_default
        and familiarity in {"common", "standard"}
    ):
        return ExperienceRoleDecision(True, "identity_reviewed_main_eligible")

    if experience == "intermediate":
        eligible = (
            main_default
            and familiarity in {"common", "standard"}
            and mainstream_score >= 55
            and failure_rank <= 2
            and caution_rank <= 2
        )
        return ExperienceRoleDecision(
            eligible,
            "intermediate_standard_main_eligible" if eligible else "intermediate_main_role_gate",
        )

    eligible = (
        main_default
        and familiarity in {"common", "standard"}
        and failure_rank <= 2
        and caution_rank <= 2
    )
    return ExperienceRoleDecision(
        eligible,
        "advanced_default_main_eligible"
        if eligible
        else "advanced_nonstandard_main_role_gate",
    )

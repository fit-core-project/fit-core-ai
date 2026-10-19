"""Deterministic routine option policy resolver.

This module resolves hard blocks and warnings before candidate ranking or LLM
prompting. It is intentionally independent from candidate scoring so the safety
surface can be tested without invoking a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


LOADED_PHASES = {"main_lift", "accessory", "finisher"}
RECOVERY_PHASES = {"warmup", "cooldown"}
HIGH_DENSITY_STRUCTURES = {"circuit", "density"}
HIGH_DENSITY_ADJUNCTS = {"conditioning_finisher", "endurance_support"}
MAX_DISTINCT_ADJUNCT_OPTIONS = 2
MUTUALLY_EXCLUSIVE_ADJUNCT_GROUPS = (
    frozenset({"conditioning_finisher", "endurance_support"}),
)
ADJUNCT_OPTION_PRIORITY = (
    "mobility_warmup",
    "prehab_accessory",
    "conditioning_finisher",
    "endurance_support",
    "cooldown_stretch",
)


def canonical_adjunct_options(adjunct_options: Iterable[Any] | None) -> tuple[str, ...]:
    """Return distinct, stable option order independent of request permutation."""
    distinct = set(_norm_options(adjunct_options))
    known = [option for option in ADJUNCT_OPTION_PRIORITY if option in distinct]
    unknown = sorted(distinct - set(ADJUNCT_OPTION_PRIORITY))
    return tuple([*known, *unknown])


def adjunct_option_conflict_code(adjunct_options: Iterable[Any] | None) -> str | None:
    """Return the first public hard-conflict code for a normalized request."""
    normalized = canonical_adjunct_options(adjunct_options)
    if len(normalized) > MAX_DISTINCT_ADJUNCT_OPTIONS:
        return "too_many_adjunct_options"
    selected = set(normalized)
    if any(group <= selected for group in MUTUALLY_EXCLUSIVE_ADJUNCT_GROUPS):
        return "conditioning_and_endurance_overlap"
    return None

ADJUNCT_OPTION_STYLE_TAGS = {
    "mobility_warmup": {"mobility", "yoga_pilates"},
    "cooldown_stretch": {"mobility", "yoga_pilates"},
    "conditioning_finisher": {"conditioning", "cardio"},
    "endurance_support": {"endurance", "conditioning", "cardio"},
    "prehab_accessory": {"prehab", "rehab_prehab"},
}

ADJUNCT_OPTION_NAME_TOKENS = {
    "mobility_warmup": {
        "모빌리티", "mobility", "동적반복", "dynamic", "오픈북", "openbook",
        "스레드더니들", "threadtheneedle", "월드그레이티스트", "worldsgreatest",
        "프라임스트레치", "primestretch", "스트레치", "stretch",
    },
    "cooldown_stretch": {"스트레치", "stretch"},
    "conditioning_finisher": {
        "스프린트", "sprint", "인터벌", "interval", "고강도", "highintensity",
        "마운틴클라이머", "mountainclimber",
    },
    "endurance_support": {
        "걷기", "walk", "조깅", "jog", "steady", "지속", "저강도",
        "lowintensity", "저충격", "lowimpact", "endurance",
    },
    "prehab_accessory": {
        "외회전", "externalrotation", "페이스풀", "facepull", "데드버그", "deadbug",
        "버드독", "birddog", "프리햅", "prehab", "회전근개", "rotatorcuff",
    },
}

ADJUNCT_OPTION_MOVEMENT_TYPES = {
    "mobility_warmup": {"STATIC", "ISOLATION", "UNKNOWN"},
    "cooldown_stretch": {"STATIC", "ISOLATION", "UNKNOWN"},
    "conditioning_finisher": {"COMPOUND", "ISOLATION", "UNKNOWN"},
    "endurance_support": {"COMPOUND", "ISOLATION", "UNKNOWN"},
    "prehab_accessory": {"STATIC", "ISOLATION", "UNKNOWN"},
}

ADJUNCT_OPTION_PHASE = {
    "mobility_warmup": "warmup_or_preparation",
    "prehab_accessory": "prehab_accessory",
    "conditioning_finisher": "conditioning_finisher",
    "endurance_support": "endurance_support",
    "cooldown_stretch": "cooldown_recovery",
}

LEGACY_STYLE_TO_ADJUNCT_OPTION = {
    "crossfit": "conditioning_finisher",
    "athletic": "conditioning_finisher",
    "mobility": "mobility_warmup",
    "yoga_pilates": "mobility_warmup",
    "rehab_prehab": "prehab_accessory",
}

LOADED_OR_COMPLEX_PROFILES = {
    "high_lumbar_load",
    "high_axial_load",
    "overhead_shoulder",
    "high_knee_shear",
    "high_wrist_stress",
    "high_ankle_demand",
    "ballistic_complex",
}

RISK_PROFILE_REGIONS = {
    "high_lumbar_load": {"lower_back"},
    "high_axial_load": {"lower_back", "neck"},
    "overhead_shoulder": {"shoulder", "neck"},
    "high_knee_shear": {"knee"},
    "high_wrist_stress": {"wrist"},
    "high_ankle_demand": {"ankle", "knee"},
    "ballistic_complex": {"shoulder", "lower_back", "knee", "wrist", "ankle"},
}

ROOT_REASON_PRIORITY = {
    "medical_not_cleared_loaded_or_complex_block": 10,
    "medical_unknown_active_pain_loaded_or_complex_block": 20,
    "acute_direct_region_loaded_phase_block": 30,
    "active_direct_region_main_lift_block": 40,
    "recorded_injury_direct_region_exercise_block": 45,
    "lower_back_heavy_spinal_loading_block": 50,
    "shoulder_overhead_loaded_block": 60,
    "knee_high_shear_main_lift_block": 70,
    "wrist_loaded_stress_block": 80,
    "ankle_high_demand_main_lift_block": 90,
    "beginner_ballistic_complex_block": 100,
    "low_readiness_high_density_block": 105,
    "low_readiness_high_density_loaded_phase_block": 110,
    "strength_high_density_main_lift_block": 120,
    "strength_circuit_block": 125,
    "medical_unknown_requires_caution_copy": 200,
    "medical_restricted_requires_caution_copy": 205,
    "pain_status_without_region_needs_region_prompt": 210,
    "pain_region_without_pain_status_copy_guard": 220,
    "mild_direct_discomfort_requires_downgrade": 230,
    "direct_region_recovery_work_needs_non_medical_copy": 240,
    "pain_context_recommends_low_intensity_preparation_or_cooldown": 300,
}

POLICY_ROOT_KIND = {
    "medical_not_cleared_loaded_or_complex_block": "block",
    "medical_unknown_active_pain_loaded_or_complex_block": "block",
    "acute_direct_region_loaded_phase_block": "block",
    "active_direct_region_main_lift_block": "block",
    "recorded_injury_direct_region_exercise_block": "block",
    "lower_back_heavy_spinal_loading_block": "block",
    "shoulder_overhead_loaded_block": "block",
    "knee_high_shear_main_lift_block": "block",
    "wrist_loaded_stress_block": "block",
    "ankle_high_demand_main_lift_block": "block",
    "beginner_ballistic_complex_block": "block",
    "low_readiness_high_density_block": "block",
    "low_readiness_high_density_loaded_phase_block": "block",
    "strength_high_density_main_lift_block": "block",
    "strength_circuit_block": "block",
    "medical_unknown_requires_caution_copy": "warn",
    "medical_restricted_requires_caution_copy": "warn",
    "pain_status_without_region_needs_region_prompt": "warn",
    "pain_region_without_pain_status_copy_guard": "warn",
    "mild_direct_discomfort_requires_downgrade": "warn",
    "direct_region_recovery_work_needs_non_medical_copy": "warn",
    "pain_context_recommends_low_intensity_preparation_or_cooldown": "recommend",
}

POLICY_ROOT_RUNTIME_ACTION = {
    **{
        root: "exclude_candidate_or_option"
        for root, kind in POLICY_ROOT_KIND.items()
        if kind == "block"
    },
    "medical_unknown_requires_caution_copy": "rank_penalty_and_caution_copy",
    "medical_restricted_requires_caution_copy": "rank_penalty_and_caution_copy",
    "pain_status_without_region_needs_region_prompt": "input_review_and_caution_copy",
    "pain_region_without_pain_status_copy_guard": "input_review_and_caution_copy",
    "mild_direct_discomfort_requires_downgrade": "rank_penalty_and_caution_copy",
    "direct_region_recovery_work_needs_non_medical_copy": "caution_copy",
    "pain_context_recommends_low_intensity_preparation_or_cooldown": "recommend_recovery_phase",
}

# Current public contract: currentPainAreas is a binary registered-injury list.
# Legacy same-day pain intensity roots remain readable for historical artifacts,
# but active matrices and release evidence must use only this registry subset.
ACTIVE_BINARY_INJURY_POLICY_ROOTS = {
    "medical_not_cleared_loaded_or_complex_block",
    "recorded_injury_direct_region_exercise_block",
    "beginner_ballistic_complex_block",
    "low_readiness_high_density_block",
    "low_readiness_high_density_loaded_phase_block",
    "strength_high_density_main_lift_block",
    "strength_circuit_block",
    "medical_unknown_requires_caution_copy",
    "medical_restricted_requires_caution_copy",
}

LEGACY_PAIN_INTENSITY_POLICY_ROOTS = set(POLICY_ROOT_KIND) - ACTIVE_BINARY_INJURY_POLICY_ROOTS


@dataclass(frozen=True)
class RoutineOptionCase:
    goal: str = "hypertrophy"
    experience_level: str = "intermediate"
    readiness_level: str = "normal"
    time_available_min: int = 60
    session_structure: str = "standard"
    adjunct_options: tuple[str, ...] = ()
    pain_status: str = "none"
    medical_clearance: str = "not_needed"
    pain_region: str = "none"
    intensity_intent: str = "normal"
    application_phase: str = "main_lift"
    exercise_risk_profile: str = "normal"


@dataclass(frozen=True)
class OptionPolicyHit:
    kind: str
    root_reason_code: str
    target: str
    user_message_ko: str
    evidence_code: str


@dataclass
class OptionPolicyResolution:
    blocked: list[OptionPolicyHit] = field(default_factory=list)
    warnings: list[OptionPolicyHit] = field(default_factory=list)
    recommended: list[OptionPolicyHit] = field(default_factory=list)
    forced: list[OptionPolicyHit] = field(default_factory=list)

    @property
    def allowed(self) -> bool:
        return not self.blocked

    @property
    def severity(self) -> str:
        if self.blocked:
            return "block"
        if self.warnings:
            return "warn"
        if self.recommended or self.forced:
            return "recommend"
        return "ok"

    @property
    def root_reason_code(self) -> Optional[str]:
        hit = highest_priority_hit(self.blocked or self.warnings or self.forced or self.recommended)
        return hit.root_reason_code if hit else None

    @property
    def user_message_ko(self) -> Optional[str]:
        hit = highest_priority_hit(self.blocked or self.warnings or self.forced or self.recommended)
        return hit.user_message_ko if hit else None

    @property
    def evidence_codes(self) -> list[str]:
        hits = [*self.blocked, *self.warnings, *self.recommended, *self.forced]
        return [hit.evidence_code for hit in hits]


def _norm(value: Any, default: str = "") -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return text or default


def _norm_options(values: Iterable[Any] | None) -> tuple[str, ...]:
    return tuple(token for token in (_norm(value, "") for value in (values or [])) if token)


def candidate_style_tags(candidate: dict[str, Any]) -> set[str]:
    raw = candidate.get("training_style_tags")
    if isinstance(raw, str):
        tags = {_norm(token, "") for token in raw.split(",") if _norm(token, "")}
    elif isinstance(raw, (list, tuple, set)):
        tags = {_norm(token, "") for token in raw if _norm(token, "")}
    else:
        tags = set()
    for boost in candidate.get("score_boosts") or []:
        if not isinstance(boost, dict):
            continue
        rule_code = _norm(boost.get("rule_code") or boost.get("ruleCode"), "")
        if rule_code not in {"training_style_match_boost", "adjunct_option_match_boost"}:
            continue
        constraint = _norm(boost.get("constraint_code") or boost.get("constraintCode"), "")
        if constraint:
            tags.add(constraint)
    return tags


def candidate_adjunct_option_codes(candidate: dict[str, Any]) -> set[str]:
    raw = candidate.get("adjunct_option_codes") or candidate.get("adjunctOptionCodes")
    if isinstance(raw, str):
        values = raw.split(",")
    elif isinstance(raw, (list, tuple, set)):
        values = raw
    else:
        values = []
    return {_norm(value, "") for value in values if _norm(value, "")}


def candidate_semantically_matches_adjunct_option(
    candidate: dict[str, Any],
    adjunct_option: str,
) -> bool:
    option = _norm(adjunct_option, "")
    accepted_movements = ADJUNCT_OPTION_MOVEMENT_TYPES.get(option, set())
    movement_type = _norm(candidate.get("movement_type"), "unknown").upper()
    if not accepted_movements or movement_type not in accepted_movements:
        return False

    searchable = "".join(
        str(candidate.get(key) or "").strip().lower()
        for key in (
            "user_facing_display_name_kr",
            "display_name_ko",
            "base_exercise_name_kr",
            "name_kr",
            "name_en",
        )
    ).replace(" ", "").replace("-", "").replace("_", "").replace("'", "")
    if option == "cooldown_stretch" and any(
        token in searchable for token in ("동적반복", "dynamic")
    ):
        return False
    return any(
        token.replace(" ", "").replace("-", "").replace("_", "").replace("'", "") in searchable
        for token in ADJUNCT_OPTION_NAME_TOKENS.get(option, set())
    )


def candidate_matches_adjunct_option(candidate: dict[str, Any], adjunct_option: str) -> bool:
    option = _norm(adjunct_option, "")
    accepted_tags = ADJUNCT_OPTION_STYLE_TAGS.get(option, set())
    explicitly_curated = option in candidate_adjunct_option_codes(candidate)
    metadata_match = bool(accepted_tags and candidate_style_tags(candidate) & accepted_tags)
    if not explicitly_curated and not metadata_match:
        return False
    return candidate_semantically_matches_adjunct_option(candidate, option)


def matching_adjunct_options(
    candidate: dict[str, Any],
    adjunct_options: Iterable[Any] | None,
) -> tuple[str, ...]:
    return tuple(
        option
        for option in _norm_options(adjunct_options)
        if candidate_matches_adjunct_option(candidate, option)
    )


def _hit(kind: str, root_reason_code: str, target: str, user_message_ko: str, evidence_code: str) -> OptionPolicyHit:
    return OptionPolicyHit(
        kind=kind,
        root_reason_code=root_reason_code,
        target=target,
        user_message_ko=user_message_ko,
        evidence_code=evidence_code,
    )


def highest_priority_hit(hits: Iterable[OptionPolicyHit]) -> Optional[OptionPolicyHit]:
    hits = list(hits)
    if not hits:
        return None
    return min(hits, key=lambda hit: ROOT_REASON_PRIORITY.get(hit.root_reason_code, 9999))


def risk_profile_matches_pain_region(exercise_risk_profile: str, pain_region: str) -> bool:
    region = _norm(pain_region, "none")
    if region == "none":
        return False
    return region in RISK_PROFILE_REGIONS.get(_norm(exercise_risk_profile, "normal"), set())


def derive_exercise_risk_profiles(candidate: dict[str, Any]) -> set[str]:
    """Map DB exercise attributes to resolver risk profile tags."""
    profiles: set[str] = set()

    def level(field: str) -> str:
        return _norm(candidate.get(field), "")

    if level("lumbar_load") == "high":
        profiles.add("high_lumbar_load")
    if level("axial_load") == "high":
        profiles.add("high_axial_load")
    if level("knee_shear_load") == "high":
        profiles.add("high_knee_shear")
    if level("wrist_stress") == "high":
        profiles.add("high_wrist_stress")
    if level("ankle_dorsiflexion_demand") == "high":
        profiles.add("high_ankle_demand")

    overhead_required = str(candidate.get("overhead_position_required") or "").strip().lower()
    overhead = overhead_required in {"1", "true", "yes", "y", "on"}
    if overhead and level("shoulder_impingement_risk") in {"medium", "high"}:
        profiles.add("overhead_shoulder")
    elif level("shoulder_impingement_risk") == "high":
        profiles.add("overhead_shoulder")

    compact_text = " ".join(
        str(candidate.get(key) or "")
        for key in ["name_kr", "name_en", "movement_pattern", "pattern", "exercise_family"]
    ).lower()
    high_skill_name = any(
        token in compact_text
        for token in [
            "snatch",
            "clean",
            "jerk",
            "thruster",
            "스내치",
            "클린",
            "저크",
            "스러스터",
            "점프",
            "jump",
            "ballistic",
        ]
    )
    if high_skill_name or level("balance_requirement") == "high":
        profiles.add("ballistic_complex")

    return profiles or {"normal"}


def resolve_routine_option_policy(case: RoutineOptionCase) -> OptionPolicyResolution:
    goal = _norm(case.goal, "hypertrophy")
    experience = _norm(case.experience_level, "intermediate")
    readiness = _norm(case.readiness_level, "normal")
    session_structure = _norm(case.session_structure, "standard")
    adjunct_options = set(_norm_options(case.adjunct_options))
    pain_status = _norm(case.pain_status, "none")
    medical_clearance = _norm(case.medical_clearance, "not_needed")
    pain_region = _norm(case.pain_region, "none")
    intensity_intent = _norm(case.intensity_intent, "normal")
    application_phase = _norm(case.application_phase, "main_lift")
    risk_profile = _norm(case.exercise_risk_profile, "normal")

    resolution = OptionPolicyResolution()
    direct_region_risk = risk_profile_matches_pain_region(risk_profile, pain_region)
    loaded_or_complex = risk_profile in LOADED_OR_COMPLEX_PROFILES
    loaded_phase = application_phase in LOADED_PHASES
    recovery_phase = application_phase in RECOVERY_PHASES
    high_density_adjunct_selected = bool(adjunct_options & HIGH_DENSITY_ADJUNCTS)
    high_density_intent = (
        intensity_intent == "high_density"
        or session_structure in HIGH_DENSITY_STRUCTURES
        or high_density_adjunct_selected
    )

    if readiness == "low" and (
        session_structure in HIGH_DENSITY_STRUCTURES
        or intensity_intent == "high_density"
        or high_density_adjunct_selected
    ):
        target = "high_density_adjunct" if high_density_adjunct_selected else "high_density_session"
        evidence_code = (
            "low_readiness_blocks_conditioning_or_endurance_adjunct"
            if high_density_adjunct_selected
            else "low_readiness_blocks_high_density_session"
        )
        resolution.blocked.append(
            _hit(
                "block",
                "low_readiness_high_density_block",
                target,
                "컨디션이 낮은 날에는 서킷/고밀도 진행을 제외하고 표준 루틴으로 낮춥니다.",
                evidence_code,
            )
        )

    if goal == "strength" and session_structure == "circuit":
        resolution.blocked.append(
            _hit(
                "block",
                "strength_circuit_block",
                "circuit_session",
                "근력 목표에서는 메인 리프트 품질과 휴식이 중요해 서킷 진행을 제외합니다.",
                "strength_blocks_circuit_main_work",
            )
        )

    if pain_status == "none" and pain_region != "none":
        resolution.warnings.append(
            _hit(
                "warn",
                "pain_region_without_pain_status_copy_guard",
                "pain_copy",
                "통증 상태가 없음인데 통증 부위가 선택되어 있어 입력 상태 확인이 필요합니다.",
                "pain_region_without_pain_status_copy_guard",
            )
        )

    if pain_status != "none" and pain_region == "none":
        resolution.warnings.append(
            _hit(
                "warn",
                "pain_status_without_region_needs_region_prompt",
                "pain_region",
                "통증 상태가 있으므로 통증 부위를 다시 확인하거나 전체 강도를 낮춥니다.",
                "pain_status_without_region_needs_region_prompt",
            )
        )

    if medical_clearance == "not_cleared" and loaded_or_complex and loaded_phase:
        resolution.blocked.append(
            _hit(
                "block",
                "medical_not_cleared_loaded_or_complex_block",
                "loaded_or_complex_exercise",
                "의료진 허가가 확인되지 않아 부하가 크거나 복잡한 운동은 제외합니다.",
                "medical_not_cleared_blocks_loaded_or_complex_work",
            )
        )
    elif medical_clearance == "unknown" and loaded_or_complex and loaded_phase and pain_status in {"active_pain", "acute_pain"}:
        resolution.blocked.append(
            _hit(
                "block",
                "medical_unknown_active_pain_loaded_or_complex_block",
                "loaded_or_complex_exercise",
                "현재 통증이 있고 의료진 허가 상태가 불명확해 고위험 운동은 제외합니다.",
                "medical_unknown_with_active_pain_blocks_loaded_or_complex_work",
            )
        )
    elif medical_clearance == "unknown" and loaded_or_complex and loaded_phase:
        resolution.warnings.append(
            _hit(
                "warn",
                "medical_unknown_requires_caution_copy",
                "medical_copy",
                "의료진 허가 상태가 불명확해 강도를 보수적으로 조절합니다.",
                "medical_unknown_requires_caution_copy",
            )
        )
    elif medical_clearance == "cleared_with_restrictions" and loaded_or_complex and loaded_phase:
        resolution.warnings.append(
            _hit(
                "warn",
                "medical_restricted_requires_caution_copy",
                "medical_copy",
                "의료진 허가는 확인됐지만 제한 조건이 있어 부하를 보수적으로 조절합니다.",
                "medical_restricted_requires_caution_copy",
            )
        )

    if pain_status == "acute_pain" and direct_region_risk and loaded_phase:
        resolution.blocked.append(
            _hit(
                "block",
                "acute_direct_region_loaded_phase_block",
                "direct_region_loading",
                "급성 통증 부위와 직접 충돌하는 부하 운동은 제외합니다.",
                "acute_direct_region_pain_blocks_loaded_phase",
            )
        )
    elif pain_status == "recorded_injury" and direct_region_risk:
        resolution.blocked.append(
            _hit(
                "block",
                "recorded_injury_direct_region_exercise_block",
                "direct_region_exercise",
                "등록된 부상·관리 부위와 직접 충돌하는 운동은 단계와 강도에 관계없이 제외합니다.",
                "recorded_injury_area_blocks_direct_exercise",
            )
        )
    elif pain_status == "active_pain" and direct_region_risk and application_phase == "main_lift":
        resolution.blocked.append(
            _hit(
                "block",
                "active_direct_region_main_lift_block",
                "direct_region_main_lift",
                "현재 통증 부위와 직접 충돌하는 메인 운동은 제외합니다.",
                "active_direct_region_pain_blocks_main_lift",
            )
        )
    elif pain_status in {"active_pain", "acute_pain"} and direct_region_risk and recovery_phase:
        resolution.warnings.append(
            _hit(
                "warn",
                "direct_region_recovery_work_needs_non_medical_copy",
                "recovery_copy",
                "통증 부위 관련 준비/정리 동작은 치료 표현 없이 낮은 강도로만 안내합니다.",
                "direct_region_recovery_work_needs_non_medical_copy",
            )
        )
    elif pain_status == "mild_discomfort" and direct_region_risk and loaded_phase:
        resolution.warnings.append(
            _hit(
                "warn",
                "mild_direct_discomfort_requires_downgrade",
                "direct_region_loading",
                "가벼운 불편감이 있는 부위와 겹치는 부하 운동은 강도와 우선순위를 낮춥니다.",
                "mild_direct_discomfort_requires_downgrade",
            )
        )

    if pain_region == "lower_back" and pain_status in {"active_pain", "acute_pain"}:
        if risk_profile in {"high_lumbar_load", "high_axial_load"} and intensity_intent == "heavy" and loaded_phase:
            resolution.blocked.append(
                _hit(
                    "block",
                    "lower_back_heavy_spinal_loading_block",
                    "heavy_spinal_loading",
                    "요추 통증이 있어 고중량 요추 전단/수직압박 계열은 제외합니다.",
                    "lower_back_pain_blocks_heavy_spinal_loading",
                )
            )

    if pain_region == "shoulder" and pain_status in {"active_pain", "acute_pain"}:
        if risk_profile == "overhead_shoulder" and loaded_phase:
            resolution.blocked.append(
                _hit(
                    "block",
                    "shoulder_overhead_loaded_block",
                    "overhead_loaded_work",
                    "어깨 통증이 있어 오버헤드성 부하 운동은 제외합니다.",
                    "shoulder_pain_blocks_overhead_loaded_work",
                )
            )

    if pain_region == "knee" and pain_status in {"active_pain", "acute_pain"}:
        if risk_profile == "high_knee_shear" and application_phase == "main_lift":
            resolution.blocked.append(
                _hit(
                    "block",
                    "knee_high_shear_main_lift_block",
                    "high_knee_shear_main_lift",
                    "무릎 통증이 있어 전단 부담이 큰 메인 운동은 제외합니다.",
                    "knee_pain_blocks_high_shear_main_lift",
                )
            )

    if pain_region == "wrist" and pain_status in {"active_pain", "acute_pain"}:
        if risk_profile == "high_wrist_stress" and loaded_phase:
            resolution.blocked.append(
                _hit(
                    "block",
                    "wrist_loaded_stress_block",
                    "loaded_wrist_stress",
                    "손목 통증이 있어 손목 부담이 큰 부하 운동은 제외합니다.",
                    "wrist_pain_blocks_loaded_wrist_stress",
                )
            )

    if pain_region == "ankle" and pain_status in {"active_pain", "acute_pain"}:
        if risk_profile == "high_ankle_demand" and application_phase == "main_lift":
            resolution.blocked.append(
                _hit(
                    "block",
                    "ankle_high_demand_main_lift_block",
                    "high_ankle_demand_main_lift",
                    "발목 통증/제한이 있어 발목 요구도가 큰 메인 운동은 제외합니다.",
                    "ankle_pain_blocks_high_demand_main_lift",
                )
            )

    if experience == "beginner" and risk_profile == "ballistic_complex":
        if application_phase in {"main_lift", "finisher"} or high_density_intent:
            resolution.blocked.append(
                _hit(
                    "block",
                    "beginner_ballistic_complex_block",
                    "ballistic_complex",
                    "초보자에게 복잡한 탄도성/고기술 운동은 메인 또는 고밀도 구성에서 제외합니다.",
                    "beginner_blocks_ballistic_complex_main_or_density",
                )
            )

    if readiness == "low" and high_density_intent and application_phase in {"main_lift", "finisher"}:
        resolution.blocked.append(
            _hit(
                "block",
                "low_readiness_high_density_loaded_phase_block",
                "high_density_loaded_phase",
                "컨디션이 낮아 고밀도 본운동/마무리 구성은 제외합니다.",
                "low_readiness_blocks_high_density_loaded_phase",
            )
        )

    # A density session keeps the strength main lift as a normal single group and
    # applies density only to accessories. Block the main lift only when the
    # caller explicitly asks to perform that lift with high-density intent.
    if goal == "strength" and intensity_intent == "high_density" and application_phase == "main_lift":
        resolution.blocked.append(
            _hit(
                "block",
                "strength_high_density_main_lift_block",
                "high_density_main_lift",
                "근력 목표의 메인 운동은 고밀도 진행과 충돌하므로 제외합니다.",
                "strength_blocks_high_density_main_lift",
            )
        )

    if pain_status in {"active_pain", "acute_pain"} and recovery_phase:
        resolution.recommended.append(
            _hit(
                "recommend",
                "pain_context_recommends_low_intensity_preparation_or_cooldown",
                "low_intensity_recovery_phase",
                "통증 맥락에서는 낮은 강도의 준비/정리 동작을 우선합니다.",
                "pain_context_recommends_low_intensity_preparation_or_cooldown",
            )
        )

    return resolution


def resolve_candidate_option_policy(
    candidate: dict[str, Any],
    *,
    goal: str = "hypertrophy",
    experience_level: str = "intermediate",
    readiness_level: str = "normal",
    time_available_min: int = 60,
    session_structure: str = "standard",
    adjunct_options: Optional[Iterable[Any]] = None,
    pain_status: str = "none",
    medical_clearance: str = "not_needed",
    pain_region: str = "none",
    intensity_intent: str = "normal",
    application_phase: str = "main_lift",
) -> OptionPolicyResolution:
    """Resolve policy using the highest-severity profile derived from a candidate."""
    profiles = derive_exercise_risk_profiles(candidate)
    resolutions = [
        resolve_routine_option_policy(
            RoutineOptionCase(
                goal=goal,
                experience_level=experience_level,
                readiness_level=readiness_level,
                time_available_min=time_available_min,
                session_structure=session_structure,
                adjunct_options=_norm_options(adjunct_options),
                pain_status=pain_status,
                medical_clearance=medical_clearance,
                pain_region=pain_region,
                intensity_intent=intensity_intent,
                application_phase=application_phase,
                exercise_risk_profile=profile,
            )
        )
        for profile in profiles
    ]
    if not resolutions:
        return resolve_routine_option_policy(RoutineOptionCase())
    return sorted(
        resolutions,
        key=lambda item: (
            {"block": 0, "warn": 1, "recommend": 2, "ok": 3}.get(item.severity, 9),
            ROOT_REASON_PRIORITY.get(item.root_reason_code or "", 9999),
        ),
    )[0]


def infer_medical_clearance(condition_policies: Iterable[dict[str, Any]] | None) -> str:
    policies = list(condition_policies or [])
    if not policies:
        return "not_needed"

    normalized_aliases = {
        "notneeded": "not_needed",
        "not_needed": "not_needed",
        "cleared": "cleared",
        "clearedwithrestrictions": "cleared_with_restrictions",
        "cleared_with_restrictions": "cleared_with_restrictions",
        "notcleared": "not_cleared",
        "not_cleared": "not_cleared",
        "unknown": "unknown",
    }

    def normalize_clearance(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, bool):
            return "cleared" if value else "not_cleared"
        token = _norm(value, "")
        compact = token.replace("_", "")
        if token in {"true", "1", "yes", "on"}:
            return "cleared"
        if token in {"false", "0", "no", "off"}:
            return "not_cleared"
        return normalized_aliases.get(token) or normalized_aliases.get(compact)

    states: list[str] = []
    for policy in policies:
        if not isinstance(policy, dict):
            continue
        status = _norm(policy.get("status") or policy.get("phase"), "")
        clearance = policy.get(
            "clearanceStatus",
            policy.get(
                "clearance_status",
                policy.get("professionalClearance", policy.get("professional_clearance")),
            ),
        )
        requires_clearance = policy.get(
            "requiresProfessionalClearance",
            policy.get("requires_professional_clearance"),
        )
        explicit_state = normalize_clearance(clearance)
        if explicit_state:
            states.append(explicit_state)
            continue
        if requires_clearance is True or status in {
            "needs_clearance",
            "requires_clearance",
            "medical_clearance_required",
            "doctor_clearance_required",
            "post_surgery",
        }:
            states.append("not_cleared")
        else:
            states.append("unknown")

    if "not_cleared" in states:
        return "not_cleared"
    if "unknown" in states:
        return "unknown"
    if "cleared_with_restrictions" in states:
        return "cleared_with_restrictions"
    if "cleared" in states:
        return "cleared"
    if states and all(state == "not_needed" for state in states):
        return "not_needed"
    return "unknown"


def infer_pain_region(body_part: Any) -> str:
    token = _norm(body_part, "none")
    if token in {"none", ""}:
        return "none"
    if token in {"lower_back", "lumbar", "back", "low_back"}:
        return "lower_back"
    if token in {"shoulder", "front_deltoids", "back_deltoids", "side_deltoids", "deltoids"}:
        return "shoulder"
    if token in {"knee", "knees"}:
        return "knee"
    if token in {"wrist", "forearm"}:
        return "wrist"
    if token in {"ankle", "calves", "left_soleus", "right_soleus", "soleus"}:
        return "ankle"
    if token in {"neck", "head", "trapezius"}:
        return "neck"
    return token


def _iter_pain_contexts(pain_areas: Iterable[Any] | None) -> list[tuple[str, str]]:
    contexts: list[tuple[str, str]] = []
    for item in pain_areas or []:
        if isinstance(item, dict):
            body_part = item.get("bodyPart", item.get("body_part", item.get("area")))
        else:
            body_part = getattr(item, "body_part", None) or getattr(item, "area", None)
        region = infer_pain_region(body_part)
        # Public painAreas is a binary persisted injury/managed-area profile.
        # Intensity is represented only by DOMS and must not weaken this block.
        if region != "none":
            contexts.append((region, "recorded_injury"))
    return contexts or [("none", "none")]


def resolve_candidate_policy_for_request(
    candidate: dict[str, Any],
    *,
    goal: str = "hypertrophy",
    experience_level: str = "intermediate",
    readiness_level: str = "normal",
    time_available_min: int = 60,
    session_structure: str = "standard",
    adjunct_options: Optional[Iterable[Any]] = None,
    pain_areas: Iterable[Any] | None = None,
    condition_policies: Iterable[dict[str, Any]] | None = None,
    intensity_intent: str = "normal",
    application_phase: str = "main_lift",
) -> OptionPolicyResolution:
    medical_clearance = infer_medical_clearance(condition_policies)
    resolutions: list[OptionPolicyResolution] = []
    for pain_region, pain_status in _iter_pain_contexts(pain_areas):
        resolutions.append(
            resolve_candidate_option_policy(
                candidate,
                goal=goal,
                experience_level=experience_level,
                readiness_level=readiness_level,
                time_available_min=time_available_min,
                session_structure=session_structure,
                adjunct_options=adjunct_options,
                pain_status=pain_status,
                medical_clearance=medical_clearance,
                pain_region=pain_region,
                intensity_intent=intensity_intent,
                application_phase=application_phase,
            )
        )
    return sorted(
        resolutions,
        key=lambda item: (
            {"block": 0, "warn": 1, "recommend": 2, "ok": 3}.get(item.severity, 9),
            ROOT_REASON_PRIORITY.get(item.root_reason_code or "", 9999),
        ),
    )[0]


def annotate_candidates_with_option_policy(
    candidates: Iterable[dict[str, Any]],
    **policy_kwargs: Any,
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for candidate in candidates:
        resolution = resolve_candidate_policy_for_request(candidate, **policy_kwargs)
        item = dict(candidate)
        item["option_policy_severity"] = resolution.severity
        item["option_policy_allowed"] = resolution.allowed
        item["option_policy_root_reason_code"] = resolution.root_reason_code
        item["option_policy_user_message_ko"] = resolution.user_message_ko
        item["option_policy_evidence_codes"] = resolution.evidence_codes
        item["exercise_risk_profiles"] = sorted(derive_exercise_risk_profiles(candidate))
        annotated.append(item)
    return annotated


def filter_candidates_with_option_policy(
    candidates: Iterable[dict[str, Any]],
    **policy_kwargs: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    annotated = annotate_candidates_with_option_policy(candidates, **policy_kwargs)
    allowed = [candidate for candidate in annotated if candidate["option_policy_allowed"]]
    blocked = [candidate for candidate in annotated if not candidate["option_policy_allowed"]]
    return allowed, blocked

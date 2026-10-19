"""Checked core-only integration entrypoint; no HTTP or new public error schema."""
from __future__ import annotations

from dataclasses import dataclass

from . import db_queries, routine_pipeline
from .routine_response_v4 import RoutineResponseV4, build_v4_response
from .schemas import RoutineDraftResponse, RoutineRequest, UserProfileContext


@dataclass(frozen=True)
class CapabilityCheck:
    unsupported_fields: tuple[str, ...]

    @property
    def supported(self) -> bool:
        return not self.unsupported_fields


def check_legacy_request(request: RoutineRequest) -> CapabilityCheck:
    """Conservative internal pre-check; not an HTTP/BE adapter approval."""
    unsupported = []
    if request.session_structure != "standard":
        unsupported.append("sessionStructure")
    # Auto methods can be chosen only after ranking: response check is also required.
    for field in ("adjunct_options", "specialization_focus", "requested_method", "training_method",
                  "profile_signals", "mobility_limits", "injury_conditions", "condition_policies",
                  "anthropometry_signals", "shoulder_condition_status", "restricted_movement_patterns",
                  "recent_training_method_uses"):
        if getattr(request, field):
            unsupported.append(RoutineRequest.model_fields[field].alias)
    if request.advanced_chest_focus != "balanced":
        unsupported.append("advancedChestFocus")
    if request.training_style != "gym":
        unsupported.append("trainingStyle")
    if set(request.doms_data) & {"left-soleus", "right-soleus"}:
        unsupported.append("domsData")
    return CapabilityCheck(tuple(unsupported))


def check_legacy_response(response: RoutineDraftResponse | RoutineResponseV4) -> CapabilityCheck:
    """Detect lost meaning without dropping fields or mutating a response."""
    if isinstance(response, RoutineResponseV4):
        return CapabilityCheck(("contractVersion", "shortRoutineNotice"))
    payload = response.model_dump(mode="json", by_alias=True)
    unsupported = []
    for field in ("adjunctDecisions", "personalizationSuggestions"):
        if payload.get(field):
            unsupported.append(field)
    for index, block in enumerate(payload["routineBlocks"]):
        prefix = f"routineBlocks[{index}]."
        for field in ("primaryMuscleDetails", "trainingMethod", "adjunctOption", "unilateralPrescription"):
            if block.get(field):
                unsupported.append(prefix + field)
        if block.get("sessionPhase") not in (None, "main"):
            unsupported.append(prefix + "sessionPhase")
        if (block.get("executionGroupType", "single") != "single" or block.get("executionGroupId")
                or block.get("executionGroupSize", 1) != 1 or block.get("executionGroupPosition", 1) != 1
                or block.get("groupRestSec") is not None or block.get("transitionRestSec") is not None):
            unsupported.append(prefix + "executionGroup")
        for position, prescription in enumerate(block["prescription"]):
            at = prefix + f"prescription[{position}]."
            if prescription["setType"] not in {"working", "warmup"}:
                unsupported.append(at + "setType")
            for field in ("targetDurationSec", "targetDistanceM", "intervalWorkSec", "intervalRecoverySec", "intervalRounds"):
                if prescription.get(field) is not None:
                    unsupported.append(at + field)
    return CapabilityCheck(tuple(unsupported))


def generate_checked_v4(request: RoutineRequest, db, *, profile: UserProfileContext | None = None,
                        recent_sets=None) -> RoutineResponseV4:
    """New integration boundary. Server owns user/profile identity, never the LLM."""
    lookup = db_queries.lookup_user_constraint_profile_from_sqlite(request.user_id)
    if lookup.status == "error":
        failed = RoutineDraftResponse(generation_status="failed", status_reason_code="schemaError",
            is_fallback=False, total_estimated_time=0, summary_title="사용자 제한 정보 확인 불가",
            rationale_summary=[], warnings=["저장된 제한 정보를 확인하지 못해 루틴을 생성하지 않았습니다."],
            routine_blocks=[])
        return build_v4_response(request, failed, profile=profile)
    routine = routine_pipeline.generate_smart_routine(
        request.model_copy(deep=True), db, profile=profile, recent_sets=recent_sets,
        local_constraint_profile=lookup.profile, local_constraint_profile_is_resolved=False)
    # Absence of a stored profile is not a verified unrestricted user.
    return build_v4_response(request, routine, profile=profile,
        local_constraint_profile=lookup.profile if lookup.status == "found" else None)

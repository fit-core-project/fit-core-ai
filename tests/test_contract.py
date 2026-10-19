"""
계약 회귀(Contract Regression) 테스트
- RoutineRequest: 프론트엔드가 보내는 camelCase JSON이 올바르게 파싱되는지
- RoutineDraftResponse: 응답이 camelCase로 직렬화되는지 (FE와의 약속)
"""
import pytest
from itertools import combinations, permutations
from pydantic import ValidationError

from engines.schemas import (
    PainAreaEntry,
    RoutineDraftResponse,
    RoutineRequest,
)
from engines.registry import MUSCLE_SLUGS, PUBLIC_DOMS_BODY_PART_SLUGS, PUBLIC_PAIN_AREA_SLUGS


EXPECTED_PUBLIC_TRAINABLE_MUSCLE_SLUGS = {
    "abductors",
    "abs",
    "adductor",
    "back-deltoids",
    "biceps",
    "calves",
    "chest",
    "forearm",
    "front-deltoids",
    "gluteal",
    "hamstring",
    "lower-back",
    "neck",
    "obliques",
    "quadriceps",
    "trapezius",
    "triceps",
    "upper-back",
}
EXPECTED_PUBLIC_PAIN_AREA_SLUGS = EXPECTED_PUBLIC_TRAINABLE_MUSCLE_SLUGS | {
    "head",
    "knees",
    "left-soleus",
    "right-soleus",
}
EXPECTED_PUBLIC_DOMS_BODY_PART_SLUGS = EXPECTED_PUBLIC_TRAINABLE_MUSCLE_SLUGS | {
    "left-soleus",
    "right-soleus",
}
ADJUNCT_OPTIONS_IN_CANONICAL_ORDER = [
    "mobility_warmup",
    "prehab_accessory",
    "conditioning_finisher",
    "endurance_support",
    "cooldown_stretch",
]
ADJUNCT_PAIR_PERMUTATIONS = [
    ordered
    for pair in combinations(ADJUNCT_OPTIONS_IN_CANONICAL_ORDER, 2)
    for ordered in permutations(pair)
]


class TestRoutineRequestParsing:
    """프론트엔드 → 백엔드 입력 계약"""

    def test_camel_case_json_parsed(self):
        payload = {
            "userId": "u-123",
            "targetSplitLabel": "push",
            "timeAvailableMin": 60,
            "readinessLevel": "normal",
            "painAreas": [],
            "domsData": {},
            "equipment": [],
        }
        req = RoutineRequest.model_validate(payload)
        assert req.user_id == "u-123"
        assert req.target_split_label == "push"
        assert req.time_available_min == 60

    def test_public_body_part_allowlists_are_exactly_locked(self):
        assert MUSCLE_SLUGS == EXPECTED_PUBLIC_TRAINABLE_MUSCLE_SLUGS
        assert len(MUSCLE_SLUGS) == 18
        assert PUBLIC_PAIN_AREA_SLUGS == EXPECTED_PUBLIC_PAIN_AREA_SLUGS
        assert len(PUBLIC_PAIN_AREA_SLUGS) == 22
        assert PUBLIC_DOMS_BODY_PART_SLUGS == EXPECTED_PUBLIC_DOMS_BODY_PART_SLUGS
        assert len(PUBLIC_DOMS_BODY_PART_SLUGS) == 20

    def test_internal_pain_alias_tolerance_does_not_widen_public_request_boundary(self):
        internal_entry = PainAreaEntry(bodyPart="shoulder")
        assert internal_entry.body_part == "shoulder"

        with pytest.raises(ValidationError):
            RoutineRequest.model_validate({
                "timeAvailableMin": 45,
                "painAreas": [{"bodyPart": "shoulder"}],
            })

    def test_snake_case_also_accepted(self):
        payload = {
            "user_id": "u-123",
            "target_split_label": "legs",
            "time_available_min": 45,
        }
        req = RoutineRequest.model_validate(payload)
        assert req.target_split_label == "legs"

    def test_doms_data_camel_case_parsed(self):
        payload = {
            "timeAvailableMin": 60,
            "domsData": {"chest": 1, "triceps": 2},
        }
        req = RoutineRequest.model_validate(payload)
        assert req.doms_data["chest"] == 1
        assert req.doms_data["triceps"] == 2

    @pytest.mark.parametrize("minutes", [10, 180])
    def test_time_available_boundary_is_accepted(self, minutes):
        req = RoutineRequest.model_validate({"timeAvailableMin": minutes})
        assert req.time_available_min == minutes

    @pytest.mark.parametrize("minutes", [9, 181])
    def test_time_available_outside_internal_contract_is_rejected(self, minutes):
        with pytest.raises(ValidationError):
            RoutineRequest.model_validate({"timeAvailableMin": minutes})

    @pytest.mark.parametrize("level", [1, 2, 3])
    def test_doms_levels_one_to_three_are_accepted(self, level):
        req = RoutineRequest.model_validate({
            "timeAvailableMin": 45,
            "domsData": {"quadriceps": level},
        })
        assert req.doms_data == {"quadriceps": level}

    @pytest.mark.parametrize("level", [0, 4, -1, True])
    def test_invalid_doms_level_is_rejected(self, level):
        with pytest.raises(ValidationError):
            RoutineRequest.model_validate({
                "timeAvailableMin": 45,
                "domsData": {"quadriceps": level},
            })

    def test_blank_doms_body_part_is_rejected(self):
        with pytest.raises(ValidationError):
            RoutineRequest.model_validate({
                "timeAvailableMin": 45,
                "domsData": {" ": 1},
            })

    @pytest.mark.parametrize("body_part", ["chest", "quadriceps", "left-soleus", "right-soleus"])
    def test_locked_doms_body_part_is_accepted(self, body_part):
        req = RoutineRequest.model_validate({
            "timeAvailableMin": 45,
            "domsData": {body_part: 1},
        })

        assert req.doms_data == {body_part: 1}

    @pytest.mark.parametrize("body_part", ["head", "knees", "side-deltoids", "unknown-muscle"])
    def test_non_doms_body_part_is_rejected(self, body_part):
        with pytest.raises(ValidationError):
            RoutineRequest.model_validate({
                "timeAvailableMin": 45,
                "domsData": {body_part: 1},
            })

    def test_injury_area_is_binary_and_accepts_no_severity(self):
        req = RoutineRequest.model_validate({
            "timeAvailableMin": 45,
            "painAreas": [{"bodyPart": "chest"}],
        })

        assert req.pain_areas[0].body_part == "chest"
        assert not hasattr(req.pain_areas[0], "severity")

    @pytest.mark.parametrize("body_part", ["chest", "head", "knees", "left-soleus", "right-soleus"])
    def test_locked_pain_area_is_accepted(self, body_part):
        req = RoutineRequest.model_validate({
            "timeAvailableMin": 45,
            "painAreas": [{"bodyPart": body_part}],
        })

        assert req.pain_areas[0].body_part == body_part

    @pytest.mark.parametrize("body_part", ["side-deltoids", "shoulder", "unknown-area"])
    def test_unknown_or_unversioned_pain_area_is_rejected(self, body_part):
        with pytest.raises(ValidationError):
            RoutineRequest.model_validate({
                "timeAvailableMin": 45,
                "painAreas": [{"bodyPart": body_part}],
            })

    def test_target_muscles_use_locked_public_trainable_slugs(self):
        req = RoutineRequest.model_validate({
            "timeAvailableMin": 45,
            "targetMuscles": ["chest", "triceps", "chest"],
        })

        assert req.target_muscles == ["chest", "triceps"]

    @pytest.mark.parametrize("body_part", ["head", "knees", "side-deltoids", "unknown-muscle"])
    def test_pain_only_unknown_or_unversioned_target_muscle_is_rejected(self, body_part):
        with pytest.raises(ValidationError):
            RoutineRequest.model_validate({
                "timeAvailableMin": 45,
                "targetMuscles": [body_part],
            })

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("severity", "mild"),
            ("severity", "moderate"),
            ("severity", "severe"),
            ("severity", 1),
            ("severity", None),
            ("severity", ""),
            ("level", "mild"),
            ("level", None),
            ("level", ""),
            ("painLevel", "mild"),
            ("pain_level", 2),
            ("intensity", "high"),
        ],
    )
    def test_injury_severity_key_is_rejected(self, field, value):
        with pytest.raises(ValidationError):
            RoutineRequest.model_validate({
                "timeAvailableMin": 45,
                "painAreas": [{"bodyPart": "chest", field: value}],
            })

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            RoutineRequest.model_validate({})  # time_available_min 필수

    def test_defaults_applied(self):
        req = RoutineRequest.model_validate({"timeAvailableMin": 30})
        assert req.readiness_level == "normal"
        assert req.target_muscles == []
        assert req.pain_areas == []
        assert req.doms_data == {}
        assert req.equipment == []
        assert req.experience_level is None
        assert req.training_style == "gym"
        assert req.mobility_limits == []
        assert req.injury_conditions == []
        assert req.shoulder_condition_status is None
        assert req.restricted_movement_patterns == []
        assert req.condition_policies == []
        assert req.anthropometry_signals == {}

    def test_routine_quality_inputs_parse_camel_case(self):
        payload = {
            "timeAvailableMin": 60,
            "experienceLevel": "beginner",
            "trainingStyle": "bodybuilding",
            "mobilityLimits": ["limited_ankle_dorsiflexion"],
            "conditionPolicies": [
                {
                    "conditionCode": "lumbar_herniation",
                    "bodyPart": "lower-back",
                    "phase": "rehab",
                    "professionalClearance": False,
                }
            ],
            "anthropometrySignals": {
                "femurLengthLevel": "long",
                "armLengthLevel": "average",
            },
        }
        req = RoutineRequest.model_validate(payload)

        assert req.mobility_limits == ["limited_ankle_dorsiflexion"]
        assert req.experience_level == "beginner"
        assert req.training_style == "bodybuilding"
        assert req.condition_policies[0]["conditionCode"] == "lumbar_herniation"
        assert req.anthropometry_signals["femurLengthLevel"] == "long"

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("readinessLevel", "exhausted"),
            ("experienceLevel", "expert"),
            ("trainingStyle", "pilates_class"),
            ("sessionStructure", "random"),
            ("adjunctOptions", ["anything"]),
            ("mobilityLimits", ["limited_spine_rotation"]),
        ],
    )
    def test_unknown_routine_option_value_is_rejected(self, field, value):
        with pytest.raises(ValidationError):
            RoutineRequest.model_validate({"timeAvailableMin": 45, field: value})

    def test_mobility_limits_are_binary_codes_and_deduplicated(self):
        req = RoutineRequest.model_validate({
            "timeAvailableMin": 45,
            "mobilityLimits": [
                "limited_ankle_dorsiflexion",
                "limited_ankle_dorsiflexion",
                "limited_wrist_extension",
            ],
        })

        assert req.mobility_limits == [
            "limited_ankle_dorsiflexion",
            "limited_wrist_extension",
        ]

        with pytest.raises(ValidationError):
            RoutineRequest.model_validate({
                "timeAvailableMin": 45,
                "mobilityLimits": [{"code": "limited_ankle_dorsiflexion", "severity": "high"}],
            })

    def test_adjunct_options_accept_one_distinct_code_and_deduplicate_identical_values(self):
        req = RoutineRequest.model_validate({
            "timeAvailableMin": 45,
            "adjunctOptions": ["cooldown_stretch", "cooldown_stretch"],
        })

        assert req.adjunct_options == ["cooldown_stretch"]

    def test_two_compatible_adjunct_options_are_accepted_before_model(self):
        req = RoutineRequest.model_validate({
            "timeAvailableMin": 45,
            "adjunctOptions": ["mobility_warmup", "cooldown_stretch"],
        })

        assert req.adjunct_options == ["mobility_warmup", "cooldown_stretch"]

    def test_more_than_two_distinct_adjunct_options_are_rejected_before_model(self):
        with pytest.raises(ValidationError):
            RoutineRequest.model_validate({
                "timeAvailableMin": 45,
                "adjunctOptions": [
                    "mobility_warmup",
                    "cooldown_stretch",
                    "prehab_accessory",
                ],
            })

    def test_conditioning_and_endurance_adjuncts_are_rejected_as_overlapping(self):
        with pytest.raises(ValidationError):
            RoutineRequest.model_validate({
                "timeAvailableMin": 60,
                "adjunctOptions": ["conditioning_finisher", "endurance_support"],
            })

    @pytest.mark.parametrize("ordered_pair", ADJUNCT_PAIR_PERMUTATIONS)
    def test_all_adjunct_pair_permutations_have_stable_public_boundary(self, ordered_pair):
        is_conflict = set(ordered_pair) == {
            "conditioning_finisher",
            "endurance_support",
        }
        payload = {
            "timeAvailableMin": 60,
            "adjunctOptions": list(ordered_pair),
        }
        if is_conflict:
            with pytest.raises(ValidationError):
                RoutineRequest.model_validate(payload)
            return

        request = RoutineRequest.model_validate(payload)
        assert request.adjunct_options == [
            option
            for option in ADJUNCT_OPTIONS_IN_CANONICAL_ORDER
            if option in ordered_pair
        ]

    def test_unknown_clearance_status_is_rejected(self):
        with pytest.raises(ValidationError):
            RoutineRequest.model_validate({
                "timeAvailableMin": 45,
                "conditionPolicies": [{
                    "conditionCode": "lumbar_herniation",
                    "clearanceStatus": "probablyCleared",
                }],
            })

    @pytest.mark.parametrize(
        "payload",
        [
            {
                "targetSplitLabel": "legs",
                "timeAvailableMin": 45,
                "readinessLevel": "low",
                "sessionStructure": "circuit",
            },
            {
                "targetSplitLabel": "legs",
                "timeAvailableMin": 45,
                "readinessLevel": "low",
                "adjunctOptions": ["conditioning_finisher"],
            },
            {
                "targetSplitLabel": "upper",
                "timeAvailableMin": 60,
                "goal": "strength",
                "sessionStructure": "circuit",
            },
        ],
    )
    def test_unsafe_option_combinations_are_rejected(self, payload):
        with pytest.raises(ValidationError):
            RoutineRequest.model_validate(payload)


class TestRoutineDraftResponseSerialization:
    """백엔드 → 프론트엔드 출력 계약"""

    def test_top_level_keys_are_camel_case(self, sample_llm_output):
        from engines.fallback import build_routine_draft as _build_routine_draft

        response = _build_routine_draft(sample_llm_output, "success", "none", False)
        dumped = response.model_dump(by_alias=True)

        assert "routineDraftId" in dumped
        assert "generationStatus" in dumped
        assert "statusReasonCode" in dumped
        assert "isFallback" in dumped
        assert "summaryTitle" in dumped
        assert "rationaleSum" not in dumped  # 오타 방지
        assert "rationale_summary" not in dumped  # snake_case 유출 방지
        assert "rationaleSum mary" not in dumped

    def test_routine_blocks_keys_are_camel_case(self, sample_llm_output):
        from engines.fallback import build_routine_draft as _build_routine_draft

        response = _build_routine_draft(sample_llm_output, "success", "none", False)
        dumped = response.model_dump(by_alias=True)

        block = dumped["routineBlocks"][0]
        assert "exerciseId" in block
        assert "exerciseName" in block
        assert "movementPattern" in block
        assert "primaryMuscles" in block
        assert "equipmentType" in block
        assert "defaultRestSec" in block
        assert "exerciseRationale" in block
        assert "substitutionCandidates" in block
        assert "movementType" in block
        assert "sessionPhase" in block
        assert "executionGroupId" in block
        assert "executionGroupType" in block
        assert "executionGroupPosition" in block
        assert "executionGroupSize" in block

    def test_prescription_keys_are_camel_case(self, sample_llm_output):
        from engines.fallback import build_routine_draft as _build_routine_draft

        response = _build_routine_draft(sample_llm_output, "success", "none", False)
        dumped = response.model_dump(by_alias=True)

        prescription = dumped["routineBlocks"][0]["prescription"][0]
        assert "setIndex" in prescription
        assert "setType" in prescription
        assert "targetReps" in prescription
        assert "targetWeightKg" in prescription
        assert "targetRir" in prescription
        assert "targetRestSec" in prescription

    def test_is_fallback_false_on_success(self, sample_llm_output):
        from engines.fallback import build_routine_draft as _build_routine_draft

        response = _build_routine_draft(sample_llm_output, "success", "none", False)
        assert response.is_fallback is False

    def test_is_fallback_true_on_fallback(self, sample_llm_output):
        from engines.fallback import build_routine_draft as _build_routine_draft

        response = _build_routine_draft(sample_llm_output, "fallback", "llmTimeout", True)
        assert response.is_fallback is True
        assert response.status_reason_code == "llmTimeout"

    def test_routine_draft_id_is_uuid_string(self, sample_llm_output):
        import re
        from engines.fallback import build_routine_draft as _build_routine_draft

        response = _build_routine_draft(sample_llm_output, "success", "none", False)
        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )
        assert uuid_pattern.match(response.routine_draft_id)

    def test_total_estimated_time_in_response(self, sample_llm_output):
        from engines.fallback import build_routine_draft as _build_routine_draft
        from engines.prescription.estimator import estimate_routine_time_min

        response = _build_routine_draft(sample_llm_output, "success", "none", False)
        dumped = response.model_dump(by_alias=True)

        assert "totalEstimatedTime" in dumped
        assert dumped["totalEstimatedTime"] == estimate_routine_time_min(sample_llm_output.exercises)

    def test_is_fallback_serialized_as_camel_case(self, sample_llm_output):
        from engines.fallback import build_routine_draft as _build_routine_draft

        response = _build_routine_draft(sample_llm_output, "success", "none", False)
        dumped = response.model_dump(by_alias=True)

        assert "isFallback" in dumped
        assert "is_fallback" not in dumped

    def test_score_breakdown_serialized_as_camel_case(self, sample_llm_output, mock_candidates):
        from engines.candidate_ranker import score_candidate_exercises
        from engines.fallback import build_routine_draft as _build_routine_draft

        ranked = score_candidate_exercises(mock_candidates, ["chest"], doms_db={"chest": 1})
        response = _build_routine_draft(
            sample_llm_output,
            "success",
            "none",
            False,
            ranked_candidates=ranked,
        )
        dumped = response.model_dump(by_alias=True)
        block = dumped["routineBlocks"][0]

        assert "reasons" in block
        assert "boosts" in block
        assert "penalties" in block
        assert block["boosts"]
        assert block["penalties"]
        assert "ruleCode" in block["boosts"][0]
        assert "rule_code" not in block["boosts"][0]
        assert any(item["profileSignalCode"] == "doms" for item in block["penalties"])

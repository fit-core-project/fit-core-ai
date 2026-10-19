"""Independent arithmetic for the frozen DOMS1 sample, not a runtime dump."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import pytest

from engines.prescription.adjustments import _apply_doms_volume_policy, DOMS_DOSE_POLICY_VERSION
from engines.prescription.estimator import estimate_routine_time_min
from engines.schemas import LLMExercisePlan


GOLDEN = Path(__file__).parent / "golden"
HISTORICAL_HASHES = {
    "main_path_with_deterministic.json": "ebddfa0677bd1397194b33fa2138167cc3cdfc78dcfa07358fc43fa55f72a6b0",
    "main_path_with_deterministic_v2.json": "bfc3823f279a863f3430bb5f4de0e49804568c261ee85a5c64bb7ad73ae7cd30",
    "main_path_with_deterministic_v3.json": "4811f2f737e9cba34bf8c81f1543be4b1390f354ba8b6fcb2e7aba054bbc2291",
}


def _expected_v4():
    expected = json.loads((GOLDEN / "main_path_with_deterministic_v3.json").read_text())
    # Policy: compound role cap4 -> round(4*.8)=3. Isolation cap3 -> round(3*.8)=2.
    for block in expected["block_contract"][:2]:
        assert len(block["prescription"]) == 2
        block["prescription"].append(deepcopy(block["prescription"][0]))
    seconds = 300 + 2 * 120 + 2 * (3 * 60 + 2 * 120) + 2 * 40 + 75
    assert seconds == 1535
    expected["total_estimated_time"] = math.ceil(seconds / 60)
    expected["rationale_summary"][-1] = (
        "요청하신 60분 중 조건을 반영해 선택된 계획은 예상 26분입니다. "
        "표시한 근육통을 반영해 해당 운동의 세트 수와 강도를 보수적으로 조정한 결과입니다."
    )
    return expected


def _assert_v4_oracle(value):
    assert value == _expected_v4(), "Unexpected change outside the approved recovery-dose delta"


@pytest.mark.parametrize("filename,expected", HISTORICAL_HASHES.items())
def test_historical_goldens_are_immutable(filename, expected):
    assert hashlib.sha256((GOLDEN / filename).read_bytes()).hexdigest() == expected


def test_v4_changes_only_two_compound_sets_time_and_matching_rationale():
    _assert_v4_oracle(json.loads((GOLDEN / "main_path_with_deterministic_v4.json").read_text()))


@pytest.mark.parametrize("mutation", ["old_sets", "excess_sets", "rir", "rest", "time", "ranking"])
def test_oracle_rejects_underfill_overfill_and_unrelated_changes(mutation):
    value = _expected_v4()
    first = value["block_contract"][0]
    if mutation == "old_sets":
        first["prescription"].pop()
    elif mutation == "excess_sets":
        first["prescription"].append(deepcopy(first["prescription"][0]))
    elif mutation == "rir":
        first["prescription"][0]["target_rir"] = 2
    elif mutation == "rest":
        first["prescription"][0]["target_rest_sec"] = 90
    elif mutation == "time":
        value["total_estimated_time"] = 20
    else:
        first["boost_rule_codes"].append("unapproved_change")
    with pytest.raises(AssertionError, match="approved recovery-dose delta"):
        _assert_v4_oracle(value)


@pytest.mark.parametrize("movement,level,start_sets,expected_sets,expected_rir", [
    ("COMPOUND", 1, 4, 3, 3), ("COMPOUND", 1, 2, 2, 3),
    ("COMPOUND", 2, 4, 3, 4), ("ISOLATION", 1, 3, 2, 3),
    ("ISOLATION", 2, 3, 2, 4),
])
def test_role_based_dose_and_no_restoration(movement, level, start_sets, expected_sets, expected_rir):
    assert DOMS_DOSE_POLICY_VERSION == "routine-recovery-dose-v2-2026-09-17"
    exercise = LLMExercisePlan(exercise_id="sample", exercise_name="sample",
        primary_muscles=["chest"], movement_type=movement, sets=start_sets,
        target_reps=8, rest_time_sec=120, target_rir=2,
        exercise_rationale="Frozen recovery-dose test input")
    _apply_doms_volume_policy(exercise, "hypertrophy", level,
        experience_level="intermediate", target_split_label="push")
    assert (exercise.sets, exercise.target_rir) == (expected_sets, expected_rir)
    assert exercise.rest_time_sec == 120 and exercise.target_reps == 8


@pytest.mark.parametrize("compound_sets,seconds,minutes", [(2, 1175, 20), (3, 1535, 26)])
def test_time_model_matches_independent_seconds(compound_sets, seconds, minutes):
    exercises = [LLMExercisePlan(exercise_id=str(i), exercise_name="sample",
        primary_muscles=["chest"], movement_type=movement, sets=sets,
        target_reps=reps, rest_time_sec=rest,
        exercise_rationale="Frozen time-model test input")
        for i, (movement, sets, reps, rest) in enumerate([
            ("COMPOUND", compound_sets, 8, 120),
            ("COMPOUND", compound_sets, 8, 120),
            ("ISOLATION", 2, 12, 75)])]
    manual = 300 + 2 * 120 + 2 * (compound_sets * 60 + (compound_sets - 1) * 120) + 2 * 40 + 75
    assert manual == seconds
    assert math.ceil(manual / 60) == minutes
    assert estimate_routine_time_min(exercises) == minutes

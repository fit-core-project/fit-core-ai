import json
import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.runnables import RunnableLambda
from sqlalchemy.orm import Session

from engines.routine_pipeline import generate_smart_routine


GOLDEN_DIR = Path(__file__).parent / "golden"
V2_GOLDEN_PATH = GOLDEN_DIR / "main_path_with_deterministic_v2.json"
V3_GOLDEN_PATH = GOLDEN_DIR / "main_path_with_deterministic_v3.json"
V4_GOLDEN_PATH = GOLDEN_DIR / "main_path_with_deterministic_v4.json"
V2_GOLDEN_SHA256 = "bfc3823f279a863f3430bb5f4de0e49804568c261ee85a5c64bb7ad73ae7cd30"


def _make_db_mock():
    return MagicMock(spec=Session)


def _make_llm_mock(return_value=None):
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = RunnableLambda(lambda _: return_value)
    return mock_llm


def _pretty_diff(expected: dict, actual: dict) -> str:
    expected_text = json.dumps(expected, ensure_ascii=False, indent=2, sort_keys=True)
    actual_text = json.dumps(actual, ensure_ascii=False, indent=2, sort_keys=True)
    return f"expected:\n{expected_text}\n\nactual:\n{actual_text}"


def _underfill_option_disclosed(warnings: list[str]) -> bool:
    return any(
        "이 계획을 그대로 진행하거나" in item
        and "다시 생성할 수 있습니다" in item
        for item in warnings
    )


def _behavior_projection(actual: dict) -> dict:
    blocks = actual["routine_blocks"]
    warnings = actual["warnings"]
    return {
        "generation_status": actual["generation_status"],
        "status_reason_code": actual["status_reason_code"],
        "is_fallback": actual["is_fallback"],
        "total_estimated_time": actual["total_estimated_time"],
        "summary_title": actual["summary_title"],
        "rationale_summary": actual["rationale_summary"],
        "block_contract": [
            {
                "exercise_id": block["exercise_id"],
                "exercise_name": block["exercise_name"],
                "movement_type": block["movement_type"],
                "equipment_type": block["equipment_type"],
                "session_phase": block["session_phase"],
                "execution_group_id": block["execution_group_id"],
                "execution_group_type": block["execution_group_type"],
                "prescription": [
                    {
                        "set_type": item["set_type"],
                        "target_reps": item["target_reps"],
                        "target_rir": item["target_rir"],
                        "target_rest_sec": item["target_rest_sec"],
                    }
                    for item in block["prescription"]
                ],
                "boost_rule_codes": [item["rule_code"] for item in block["boosts"]],
                "penalty_rule_codes": [item["rule_code"] for item in block["penalties"]],
                "substitution_ids": [
                    item["exercise_id"] for item in block["substitution_candidates"]
                ],
            }
            for block in blocks
        ],
        "warning_evidence": {
            "productive_fill_added": any("보조 운동 2개" in item for item in warnings),
            "candidate_exhaustion_disclosed": any(
                "추가할 적합한 운동 후보가 없어" in item for item in warnings
            ),
            "doms_warning_count": sum("최근 피로 부위" in item for item in warnings),
            "underfill_option_disclosed": _underfill_option_disclosed(warnings),
        },
    }


def test_main_path_with_deterministic_matches_pre_refactor_baseline(
    sample_request,
    sample_profile,
    sample_llm_output,
    mock_candidates,
):
    """
    Regression guard: 회복 분량 v2 정책을 반영한 v4 main-path 출력을 고정한다.
    점수 근거, DOMS 경고, 세션 실행 메타데이터가 의도치 않게 바뀌면 실패한다.
    """
    db = _make_db_mock()
    mock_llm = _make_llm_mock(return_value=sample_llm_output)

    with (
        patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
        patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
    ):
        result = generate_smart_routine(sample_request, db, profile=sample_profile, recent_sets=[])

    masked = result.model_copy(update={"routine_draft_id": "<UUID>"})
    actual = _behavior_projection(masked.model_dump(mode="json"))
    # Policy v2 (2026-09-17); preserve v3 history. See docs/reviews/golden-recovery-reconciliation-v1-2026-10-19.md.
    expected = json.loads(V4_GOLDEN_PATH.read_text(encoding="utf-8"))

    assert actual == expected, f"Behavior regression detected: {_pretty_diff(expected, actual)}"


def test_v3_golden_changes_only_the_approved_underfill_rationale():
    v2_bytes = V2_GOLDEN_PATH.read_bytes()
    assert hashlib.sha256(v2_bytes).hexdigest() == V2_GOLDEN_SHA256

    v2 = json.loads(v2_bytes)
    v3 = json.loads(V3_GOLDEN_PATH.read_text(encoding="utf-8"))
    assert v3["rationale_summary"][-1] == (
        "요청하신 60분 중 조건을 반영해 선택된 계획은 예상 20분입니다. "
        "표시한 근육통을 반영해 해당 운동의 세트 수와 강도를 보수적으로 조정한 결과입니다."
    )
    v2["rationale_summary"] = v3["rationale_summary"]
    assert v2 == v3


def test_underfill_option_oracle_rejects_obsolete_or_incomplete_copy():
    assert _underfill_option_disclosed([
        "이 계획을 그대로 진행하거나, 근육통과 겹치지 않는 다른 목표 부위를 선택해 다시 생성할 수 있습니다.",
    ])
    assert not _underfill_option_disclosed([
        "회복 후 다시 시도할 수 있습니다.",
    ])
    assert not _underfill_option_disclosed([
        "이 계획을 그대로 진행할 수 있습니다.",
    ])

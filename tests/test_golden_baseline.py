import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.runnables import RunnableLambda
from sqlalchemy.orm import Session

from engines.routine_pipeline import generate_smart_routine


GOLDEN_PATH = Path(__file__).parent / "golden" / "main_path_with_deterministic.json"


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


def test_main_path_with_deterministic_matches_pre_refactor_baseline(
    sample_request,
    sample_profile,
    sample_llm_output,
    mock_candidates,
):
    """
    Regression guard: 분리 작업(398a9d4 → 현재) 시점에 캡처된 출력을 baseline으로 고정.
    generate_smart_routine의 main path 동작이 의도치 않게 변경되면 이 테스트가 실패한다.
    """
    db = _make_db_mock()
    mock_llm = _make_llm_mock(return_value=sample_llm_output)

    with (
        patch("engines.routine_pipeline.get_llm", return_value=mock_llm),
        patch("engines.routine_pipeline.get_candidate_exercises", return_value=mock_candidates),
    ):
        result = generate_smart_routine(sample_request, db, profile=sample_profile, recent_sets=[])

    masked = result.model_copy(update={"routine_draft_id": "<UUID>"})
    actual = masked.model_dump(mode="json")
    expected = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    assert actual == expected, f"Behavior regression detected: {_pretty_diff(expected, actual)}"

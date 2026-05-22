"""
Live DB smoke test for routine generation.

사용: python scripts/smoke_routine.py
주의: 실제 DB 연결 필요. CI나 자동 테스트가 아니라 수동 smoke 용도.
결정적 회귀 검증은 tests/test_golden_baseline.py에서 수행됨.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import SessionLocal
from engines.db_queries import get_recent_sets, get_user_profile_context
from engines.routine_pipeline import generate_smart_routine
from engines.schemas import RoutineRequest


def main() -> None:
    TEST_USER_ID = "test-user-001"

    mock_payload = RoutineRequest(
        user_id=TEST_USER_ID,
        target_split_label="push",
        readiness_level="normal",
        time_available_min=70,
        pain_areas=[],
        doms_data={"chest": 1},
        equipment=["smith_machine"],
    )

    db = SessionLocal()
    try:
        print("[테스트] 프로필 및 최근 기록 조회 중...\n")
        profile = get_user_profile_context(db, TEST_USER_ID)
        recent_sets = get_recent_sets(db, TEST_USER_ID)
        print(f"[프로필] {profile}")
        print(f"[최근 세트] {len(recent_sets)}개\n")

        print("[테스트] 루틴 생성 중...\n")
        result = generate_smart_routine(
            mock_payload, db, profile=profile, recent_sets=recent_sets
        )
        print(result.model_dump_json(by_alias=True, indent=2))
    finally:
        db.close()


if __name__ == "__main__":
    main()

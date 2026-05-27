# Routine Feedback API — 설계 스펙

**버전**: 1.0  
**대상 레포**: `fit-core-ai`  
**현재 테스트 상태**: 171 passed  
**관련 P-task**: P1-3  
**전제**: P1-1 Evaluation Harness, P1-2 Rule Critic 완료

---

## 1. 설계 요약

사용자가 AI가 생성한 루틴에 대해 rating / completed / skipped / edited / note를 제출하는 수집 전용 API를 추가한다.

**V1 범위**: 저장만 수행. ranking 반영·analytics aggregation은 P2.  
**기존 generation path 변경 없음.** Rule Critic과 연결하지 않는다.

### 명시적 설계 결정 (Codex 임의 선택 금지)

| # | 결정 사항 | 선택 | 이유 |
|---|-----------|------|------|
| 1 | ORM 모델 vs raw SQL | **ORM 모델** (`RoutineFeedback` 선언) | `database.py:20`에 `Base = declarative_base()` 인프라 이미 있음; 새 테이블은 ORM으로 선언해 테스트 용이성 확보. 기존 raw SQL 패턴과 의도적 분리 |
| 2 | `routine_draft_id` 검증 | **format check만** (non-empty string) | 서버에 draft persistence 없음. FK/존재 확인 없음. 모르는 ID도 그대로 저장. Codex가 `exercise_tier` 등에서 lookup 시도 금지 |
| 3 | `exercise_id` 타입 | **`list[str]`** (사용자 초안 `list[int]` 교정) | `RoutineBlock.exercise_id: str` 기존 API 계약과 일치; `exercise_tier.id`는 DB INT지만 API 레이어에서 항상 str로 처리됨 |
| 4 | 중복 제출 | **N개 허용** (`created_at`으로 최신 우선) | 사용자가 운동 후 추가 코멘트 가능; `(user_id, routine_draft_id)` UNIQUE 없음 |
| 5 | `user_note` 로깅 | **절대 print 금지** | `install_stdout_capture()`가 stdout 캡처 중; 자유 텍스트 누출 방지 |
| 6 | 테스트 DB | **SQLite in-memory** + `Base.metadata.create_all` | 기존 171개 테스트가 DB 미사용; 신규 테스트는 격리된 in-memory fixture 사용. `JSON` 컬럼은 `sqlalchemy.JSON` (dialect-neutral) |
| 7 | "최소 1개 필드" validation | **`model_validator(mode="after")`** | Pydantic V2 패턴; Codex 구현 예시 아래 제공 |
| 8 | Migration | **raw SQL DDL + `Base.metadata.create_all` 옵션** | alembic 없음; DDL을 스펙에 첨부, 수동 실행 또는 lifespan create_all 선택 |

---

## 2. API Contract

### 엔드포인트

```
POST /api/ai/routine-feedback
Content-Type: application/json
```

### Request Schema

```json
{
  "routine_draft_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user-123",
  "rating": 4,
  "completed": true,
  "accepted_without_edits": false,
  "skipped_exercises": ["barbell_bench_press", "dumbbell_fly"],
  "edited_exercises": [
    {
      "original_exercise_id": "barbell_bench_press",
      "replacement_exercise_id": "dumbbell_bench_press",
      "reason": "too_heavy"
    }
  ],
  "user_note": "벤치가 너무 무거웠어요"
}
```

### Response Schema (성공)

```json
{
  "ok": true,
  "feedback_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "stored_at": "2026-05-27T10:30:00.000Z"
}
```

### Error Response

```json
{
  "detail": "..."
}
```

| HTTP 코드 | 사유 |
|-----------|------|
| `201 Created` | 저장 성공 |
| `422 Unprocessable Entity` | Pydantic validation 실패 |
| `500 Internal Server Error` | DB 저장 실패 |

---

## 3. Request / Response Pydantic 모델

### `RoutineFeedbackRequest`

```python
# engines/schemas.py 또는 별도 feedback_schemas.py에 추가

from __future__ import annotations
from typing import List, Literal, Optional
from uuid import uuid4
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator, model_validator

EditReason = Literal[
    "too_heavy", "too_easy", "pain", "no_equipment",
    "dislike", "duplicate", "other"
]


class EditedExercise(BaseModel):
    original_exercise_id: str
    replacement_exercise_id: str
    reason: EditReason


class RoutineFeedbackRequest(BaseModel):
    routine_draft_id: str = Field(..., min_length=1, max_length=128)
    user_id: Optional[str] = Field(default=None, max_length=128)

    # Soft payload — 최소 1개 필수
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    completed: Optional[bool] = None
    accepted_without_edits: Optional[bool] = None
    skipped_exercises: List[str] = Field(default_factory=list)
    edited_exercises: List[EditedExercise] = Field(default_factory=list)
    user_note: Optional[str] = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def at_least_one_feedback_field(self) -> "RoutineFeedbackRequest":
        has_payload = any([
            self.rating is not None,
            self.completed is not None,
            self.accepted_without_edits is not None,
            bool(self.skipped_exercises),
            bool(self.edited_exercises),
            bool(self.user_note and self.user_note.strip()),
        ])
        if not has_payload:
            raise ValueError(
                "At least one of rating, completed, accepted_without_edits, "
                "skipped_exercises, edited_exercises, user_note must be provided."
            )
        return self
```

### `RoutineFeedbackResponse`

```python
class RoutineFeedbackResponse(BaseModel):
    ok: bool = True
    feedback_id: str
    stored_at: str  # ISO 8601
```

---

## 4. DB Schema

### ORM 모델 (`database.py` 또는 `engines/feedback_model.py`)

```python
from sqlalchemy import Column, String, Integer, Boolean, Text, DateTime
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.sql import func
import uuid

from database import Base


class RoutineFeedback(Base):
    __tablename__ = "routine_feedback"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(128), nullable=True, index=True)
    routine_draft_id = Column(String(128), nullable=False, index=True)

    # Soft payload
    rating = Column(Integer, nullable=True)                  # 1~5
    completed = Column(Boolean, nullable=True)
    accepted_without_edits = Column(Boolean, nullable=True)
    skipped_exercise_ids = Column(JSON, nullable=True)       # list[str]
    edited_exercises = Column(JSON, nullable=True)           # list[dict]
    user_note = Column(Text(500), nullable=True)             # 최대 500자 (저장 시 truncate 아님; validation에서 차단)

    # Metadata
    created_at = Column(DateTime, server_default=func.now(), nullable=False, index=True)
    source = Column(String(32), nullable=False, default="api")
```

> **중요**: `skipped_exercise_ids`, `edited_exercises`는 `sqlalchemy.JSON` 타입 사용  
> (MariaDB native JSON 컬럼. SQLite 테스트 환경에서도 dialect-neutral하게 동작)

### Raw SQL DDL (수동 마이그레이션용)

```sql
CREATE TABLE IF NOT EXISTS routine_feedback (
    id                   VARCHAR(36)   NOT NULL PRIMARY KEY,
    user_id              VARCHAR(128)  NULL,
    routine_draft_id     VARCHAR(128)  NOT NULL,
    rating               TINYINT       NULL CHECK (rating BETWEEN 1 AND 5),
    completed            TINYINT(1)    NULL,
    accepted_without_edits TINYINT(1)  NULL,
    skipped_exercise_ids JSON          NULL,
    edited_exercises     JSON          NULL,
    user_note            TEXT          NULL,
    created_at           DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    source               VARCHAR(32)   NOT NULL DEFAULT 'api',

    INDEX idx_rf_user_id         (user_id),
    INDEX idx_rf_routine_draft   (routine_draft_id),
    INDEX idx_rf_created_at      (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

### 인덱스 설계 근거

| 인덱스 | 이유 |
|--------|------|
| `user_id` | P2 개인화: 사용자별 피드백 이력 조회 |
| `routine_draft_id` | 특정 루틴에 대한 피드백 집계 |
| `created_at` | 기간별 분석, 최신 피드백 우선 정렬 |

> **P2 고려**: `skipped_exercise_ids`에서 exercise_id를 추출해 별도 `routine_feedback_exercise` 테이블 정규화는 P2에서 검토.

---

## 5. Validation Rules

| 필드 | 규칙 | 위반 시 |
|------|------|---------|
| `routine_draft_id` | 필수, 1~128자 non-empty string | 422 |
| `user_id` | Optional, 최대 128자 | 422 |
| `rating` | null 또는 1~5 정수 | 422 |
| `skipped_exercises` | `list[str]` (사용자 초안 `list[int]` 교정); 각 항목 최대 128자 | 422 |
| `edited_exercises[].reason` | `EditReason` enum 6종 + "other" | 422 |
| `user_note` | 최대 500자; raw free text | 422 |
| Payload 최소 1개 | `model_validator`로 강제 | 422 |
| `routine_draft_id` 존재 여부 | **검증 없음** — format check만; unknown ID 그대로 저장 | — |
| `skipped_exercises` 운동 존재 여부 | **검증 없음** (V1) — `exercise_tier` lookup 없음 | — |

### `user_note` 처리 방침

- 최대 500자 (Pydantic `max_length=500`으로 API 레이어에서 차단)
- 저장 전 `.strip()` 적용; 빈 문자열은 `None`으로 처리
- **절대 print/log 금지** — `install_stdout_capture()`가 stdout을 `dev_log_buffer`에 캡처 중. `user_note`는 free text이므로 어떤 로그에도 포함하지 않는다

---

## 6. Duplicate 정책

**N개 허용 (append-only).**

- `(user_id, routine_draft_id)` UNIQUE 제약 없음
- 사용자가 운동 후 추가 코멘트를 제출할 수 있어야 함
- P2 집계 시 최신 `created_at` 우선 또는 모든 제출 평균 사용
- Response에 기존 `feedback_id` 미노출 (중복 여부 클라이언트에 알리지 않음)

---

## 7. Privacy / Data Minimization

| 항목 | 방침 |
|------|------|
| `user_id` | 기존 시스템(`RoutineRequest`)과 동일 방식 — request body에서 수신, nullable |
| 익명 제출 | `user_id=null` 허용 (로그인 여부 미검사; V1) |
| `user_note` | 자유 텍스트; 로그 노출 금지; DB 저장만 |
| PII | `user_note` 외 수집 항목 없음; exercise_id, rating은 행동 데이터 |
| `context_snapshot_json` | **V1에서 미사용** — 루틴 스냅샷 저장은 P2에서 별도 검토 |
| 데이터 보존 정책 | 스펙 범위 외; P2에서 정의 |

---

## 8. 파일 구조

```
fit-core-ai/
├── main.py                            # MODIFIED: 새 엔드포인트 추가
├── database.py                        # UNCHANGED
│
├── engines/
│   └── schemas.py                     # MODIFIED: RoutineFeedbackRequest, RoutineFeedbackResponse, EditedExercise 추가
│
├── models/
│   └── routine_feedback.py            # NEW: RoutineFeedback SQLAlchemy ORM 모델
│
└── tests/
    ├── conftest.py                    # MODIFIED: in-memory DB fixture 추가
    └── test_routine_feedback.py       # NEW: 피드백 API 테스트
```

> **`models/` 디렉토리 신규 생성.** 기존 raw SQL 패턴과 의도적으로 분리; 새 테이블은 ORM 모델로 관리.

---

## 9. 테스트 계획

### Fixture 설계 (`tests/conftest.py` 추가)

```python
# tests/conftest.py 추가 부분

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import Base
from models.routine_feedback import RoutineFeedback  # import로 메타데이터 등록

@pytest.fixture(scope="function")
def test_db():
    """SQLite in-memory DB; 각 테스트마다 초기화."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(engine)

@pytest.fixture
def feedback_client(test_db):
    """FastAPI TestClient with DB dependency override."""
    from fastapi.testclient import TestClient
    from main import app
    from database import get_db

    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()
```

### 테스트 목록 (`tests/test_routine_feedback.py`)

```python
# -- 정상 저장 --
def test_full_feedback_stored():
    """rating + completed + skipped + edited + note 전체 제출 → 201 + feedback_id"""

def test_rating_only_stored():
    """rating만 제출 → 201"""

def test_completed_only_stored():
    """completed만 제출 → 201"""

def test_skipped_exercises_stored():
    """skipped_exercises만 제출 → 201, DB에 list 저장 확인"""

def test_edited_exercises_stored():
    """edited_exercises만 제출 → 201, original/replacement/reason 저장 확인"""

def test_user_note_only_stored():
    """user_note만 제출 → 201"""

def test_anonymous_feedback_allowed():
    """user_id=null → 201 (익명 허용)"""

def test_duplicate_submission_allowed():
    """같은 (user_id, routine_draft_id) 두 번 제출 → 둘 다 201; DB에 2건"""

# -- Validation reject --
def test_missing_routine_draft_id_rejected():
    """routine_draft_id 없음 → 422"""

def test_empty_routine_draft_id_rejected():
    """routine_draft_id="" → 422"""

def test_invalid_rating_zero_rejected():
    """rating=0 → 422"""

def test_invalid_rating_six_rejected():
    """rating=6 → 422"""

def test_empty_feedback_rejected():
    """모든 payload 필드 없음 → 422 (model_validator)"""

def test_user_note_length_limit():
    """user_note 501자 → 422"""

def test_user_note_500_chars_allowed():
    """user_note 500자 정확히 → 201"""

def test_invalid_reason_enum_rejected():
    """reason='unknown_value' → 422"""

def test_skipped_exercises_type_str():
    """skipped_exercises=['abc', 'def'] (str list) → 201"""

# -- DB persistence --
def test_feedback_persisted_in_db(test_db):
    """201 반환 후 test_db에 해당 feedback_id 존재 확인"""

def test_feedback_id_is_uuid(feedback_client):
    """response.feedback_id가 UUID 형식"""

def test_stored_at_is_iso8601(feedback_client):
    """response.stored_at이 ISO 8601 형식"""

def test_source_field_defaults_to_api(test_db):
    """DB source 컬럼 = 'api'"""

# -- 기존 테스트 영향 없음 --
def test_generate_routine_endpoint_unaffected(feedback_client):
    """POST /api/ai/routine-feedback 추가 후 generate-routine 엔드포인트 정상"""
```

> 목표 신규 테스트: **20개 이상**. 기존 171 + 20 = 191+ passed.

---

## 10. Migration Plan

### Option A: 수동 실행 (권장)

1. Section 4의 DDL을 MariaDB에서 직접 실행
2. 실행 후 `pytest tests/test_routine_feedback.py -v` 통과 확인

### Option B: lifespan에서 자동 생성 (개발 환경 전용)

```python
# main.py lifespan에 추가 (개발 환경만)
from models.routine_feedback import RoutineFeedback  # metadata 등록
from database import engine, Base

@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("APP_ENV", "").strip().lower() != "production":
        Base.metadata.create_all(bind=engine)  # 신규 테이블만 생성; 기존 테이블 변경 없음
    ...
    yield
    ...
```

> **Production 주의**: `create_all`은 `IF NOT EXISTS` 동작이므로 기존 테이블을 덮어쓰지 않음. 단, production에서는 Option A(수동 DDL) 권장.

### 마이그레이션 순서

1. `models/routine_feedback.py` 생성 (ORM 모델)
2. DDL 실행 (MariaDB)
3. `engines/schemas.py`에 Request/Response 모델 추가
4. `main.py`에 엔드포인트 추가
5. `tests/conftest.py`에 in-memory DB fixture 추가
6. `tests/test_routine_feedback.py` 작성 및 실행

---

## 11. 리스크

| 리스크 | 내용 | 대응 |
|--------|------|------|
| **`routine_draft_id` orphan** | 루틴 draft가 서버에 저장 안 됨 → 어떤 ID도 valid; 잘못된 ID가 들어올 수 있음 | V1 허용; P2에서 루틴 persistence 추가 시 FK 제약 도입 |
| **`user_note` PII 누출** | 자유 텍스트에 이름·전화번호 포함 가능 | 로그 노출 금지 명시; 저장만 수행; 별도 검열은 P2 |
| **SQLite JSON 타입** | `sqlalchemy.JSON`은 SQLite에서 `TEXT`로 fallback | 테스트에서는 동작하지만 `JSON_EXTRACT` 쿼리 사용 시 주의 (V1 쿼리 없으므로 안전) |
| **`skipped_exercises` 정합성** | 루틴에 없는 exercise_id 제출 가능 | V1 허용; P2에서 routine_draft persistence 도입 후 검증 |
| **익명 사용자 남용** | `user_id=null` 무제한 제출 | V1 rate limit 없음; P2에서 IP 기반 제한 고려 |
| **`edited_exercises` 중첩 depth** | JSON 구조가 깊어지면 쿼리 어려움 | V1은 flat object로 제한 (spec 구조 유지) |
| **기존 테스트 영향** | `main.py` lifespan 수정 가능성 | Option B 사용 시 `APP_ENV` 조건부로 기존 lifespan 동작 보장 |

---

## 12. Codex 구현용 최종 지시문

```
작업명: P1-3 Routine Feedback API + Schema 구현
대상 레포: fit-core-ai
스펙 파일: docs/routine_feedback_api_spec.md (전체 참조)
현재 테스트: 171 passed — 이 숫자를 깨지 않는다

────────────────────────────────────────────
Step 1. models/routine_feedback.py 생성
────────────────────────────────────────────
스펙 섹션 4 ORM 모델 참조.

- `database.Base` 상속
- `id`: VARCHAR(36) PK, default=uuid4
- `user_id`: nullable, indexed
- `routine_draft_id`: NOT NULL, indexed
- `rating`: nullable int
- `completed`, `accepted_without_edits`: nullable bool
- `skipped_exercise_ids`: sqlalchemy.JSON (nullable)
- `edited_exercises`: sqlalchemy.JSON (nullable)
- `user_note`: Text(500) nullable
- `created_at`: server_default=func.now(), indexed
- `source`: String(32), default="api"

────────────────────────────────────────────
Step 2. engines/schemas.py 에 모델 추가
────────────────────────────────────────────
스펙 섹션 3 전체 참조.

추가할 클래스:
  - EditReason (Literal)
  - EditedExercise (BaseModel)
  - RoutineFeedbackRequest (BaseModel) — model_validator 포함
  - RoutineFeedbackResponse (BaseModel)

타입 주의:
  - skipped_exercises: List[str]  ← list[int] 아님
  - user_note: max_length=500
  - model_validator(mode="after"): at_least_one_feedback_field

────────────────────────────────────────────
Step 3. main.py 에 엔드포인트 추가
────────────────────────────────────────────
기존 엔드포인트 변경 없이 아래 추가:

@app.post("/api/ai/routine-feedback", response_model=RoutineFeedbackResponse, status_code=201)
def api_routine_feedback(req: RoutineFeedbackRequest, db: Session = Depends(get_db)):
    ...

구현 주의:
  - user_note를 절대 print/log 하지 않는다 (stdout capture 때문)
  - DB 저장 실패 시 500, rollback
  - RoutineFeedback ORM 인스턴스 생성 후 db.add() / db.commit() / db.refresh()
  - skipped_exercise_ids에 req.skipped_exercises 저장
  - edited_exercises에 [e.model_dump() for e in req.edited_exercises] 저장
  - user_note: req.user_note.strip() if req.user_note else None; 빈 문자열은 None

────────────────────────────────────────────
Step 4. tests/conftest.py 에 fixture 추가
────────────────────────────────────────────
스펙 섹션 9 Fixture 설계 참조.

  - test_db(): SQLite in-memory, Base.metadata.create_all
  - feedback_client(test_db): TestClient + get_db override
  - 기존 fixture (sample_request, sample_llm_output, mock_candidates) 유지

────────────────────────────────────────────
Step 5. tests/test_routine_feedback.py 생성
────────────────────────────────────────────
스펙 섹션 9 테스트 목록 전체 구현.
목표: 20개 이상 테스트.

────────────────────────────────────────────
Step 6. DB 테이블 생성
────────────────────────────────────────────
스펙 섹션 10 Migration Plan 참조.
개발 환경: lifespan Option B 또는 DDL 수동 실행.

────────────────────────────────────────────
Step 7. 검증 (반드시 모두 통과 후 완료 선언)
────────────────────────────────────────────
# 기존 테스트 영향 없음 확인 (171개)
pytest tests/test_prompt_regression.py tests/test_guard.py \
       tests/test_contract.py tests/test_fallback.py tests/test_rule_critic.py -q

# 신규 피드백 테스트 (20개 이상)
pytest tests/test_routine_feedback.py -v --tb=short

# 전체
pytest tests/ -q

────────────────────────────────────────────
제약 조건 (위반 시 즉시 중단)
────────────────────────────────────────────
- engines/routine_pipeline.py 변경 없음
- Rule Critic 연결 없음
- user_note를 print/log 금지
- ranking 로직 변경 없음
- skipped_exercises 타입: list[str] (list[int] 아님)
- routine_draft_id: format check만, exercise_tier lookup 없음
- should_rebuild/should_fallback 트리거 없음
- LLM API 호출 없음
```

---

*이 스펙은 P1-3의 산출물이다. 구현은 Codex가 담당한다.*

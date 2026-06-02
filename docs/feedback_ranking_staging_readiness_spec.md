# P2-6 Feedback Ranking Staging Readiness Hardening Spec

## 1. 설계 요약

P2-6은 P2-5 feedback-aware ranking staging test에서 드러난 `routine_feedback` DB readiness와 seed 안정성 문제를 보완하기 위한 구현 스펙이다. 이 단계의 구현 목표는 staging 실험 스크립트가 실행 전에 DB 상태를 명확히 검증하고, seed를 반복 실행해도 feedback adjustment가 왜곡되지 않도록 만들며, readiness/seed 상태를 report에 남기는 것이다.

권장 구현 방향:

- 기본 정책은 `routine_feedback` 테이블이 없으면 fail fast.
- 테이블 생성은 dev/staging/test 전용 명시 옵션 `--allow-create-tables`에서만 허용.
- production-like 환경에서는 스크립트 기반 `create_all()`을 금지.
- seed row는 deterministic `routine_draft_id` prefix `staging-seed-`로만 관리.
- staging seed는 기본적으로 기존 seed row를 중복 삽입하지 않는다.
- 명시 옵션 `--reset-seed`가 있을 때만 `staging-seed-*` row를 삭제 후 재삽입한다.
- 실제 사용자 feedback row는 절대 삭제하지 않는다.
- `user_note`는 staging seed payload, readiness report, telemetry, logs에 계속 포함하지 않는다.

P2-6은 production ranking behavior, Gemini quota, candidate pool promotion, UI를 변경하지 않는다.

## 2. 현재 문제 분석

검토 대상:

- `scripts/run_feedback_ranking_staging_test.py`
- `models/routine_feedback.py`
- `database.py`
- `main.py`
- `tests/test_routine_feedback.py`
- `tests/test_feedback_aggregation.py`
- `tests/test_feedback_ranking_telemetry.py`

현재 상태:

- `models/routine_feedback.py`에는 `RoutineFeedback` ORM 모델이 있고 `__tablename__ = "routine_feedback"`이다.
- `database.py`는 `DATABASE_URL` 기반 SQLAlchemy `engine`, `SessionLocal`, `Base`를 제공한다.
- `main.py`는 `RoutineFeedback`을 import하고 `/api/ai/routine-feedback`에서 row를 저장하지만 startup에서 `Base.metadata.create_all()`을 실행하지 않는다.
- 현재 저장 API는 중복 `routine_draft_id` 제출을 허용한다. `tests/test_routine_feedback.py`도 duplicate submission이 두 row로 저장되는 append-only 계약을 검증한다.
- 테스트 DB는 `tests/conftest.py`의 in-memory SQLite fixture에서만 `Base.metadata.create_all(engine)`과 `drop_all(engine)`을 수행한다.
- 별도 Alembic 또는 migration framework 파일은 현재 코드베이스에서 확인되지 않는다.
- 기존 스펙 `docs/routine_feedback_api_spec.md`는 Alembic 없음, raw SQL DDL 또는 수동 생성/선택적 `create_all`을 언급하지만, 현재 runtime startup 자동 생성은 구현되어 있지 않다.
- `scripts/run_feedback_ranking_staging_test.py seed`는 `SEED_ROWS`를 API로 POST한다. 같은 `staging-seed-*` draft id를 반복 seed하면 API 계약상 row가 누적될 수 있다.
- `scripts/run_feedback_ranking_staging_test.py run`과 `compare` report에는 DB readiness 또는 seed readiness 상태가 없다.

P2-5 staging 결과에서 드러난 blocker:

- `routine_feedback` 테이블이 없어 `Base.metadata.create_all()`로 수동 생성해야 했다.
- Gemini quota exhaustion으로 ranking 품질 판단은 불가했지만, DB readiness 문제는 quota와 무관한 실험 준비도 문제다.

결론:

- `routine_feedback` 생성 주체가 명확하지 않다.
- staging script가 실험 전에 DB readiness를 검증하지 않는다.
- seed가 append-only API를 사용하면서 deterministic id를 쓰기 때문에 반복 실행 시 feedback aggregation이 과대표집될 수 있다.
- report가 DB/seed 상태를 남기지 않아 나중에 staging 결과의 신뢰도를 판단하기 어렵다.

## 3. DB Readiness Policy

### 기본 정책

`scripts/run_feedback_ranking_staging_test.py`는 `seed`, `run`, `verify-seed` 실행 전에 `routine_feedback` 테이블 존재 여부를 확인해야 한다.

- 테이블 존재: readiness pass.
- 테이블 없음 + `--allow-create-tables` 없음: fail fast.
- 테이블 없음 + `--allow-create-tables` 있음 + dev/staging/test 환경: `RoutineFeedback.__table__.create(bind=engine, checkfirst=True)` 또는 `Base.metadata.create_all(bind=engine, tables=[RoutineFeedback.__table__])`로 해당 테이블만 생성.
- 테이블 없음 + `--allow-create-tables` 있음 + production-like 환경: fail fast.

테이블 생성은 전체 metadata create가 아니라 `routine_feedback` 단일 테이블에 한정한다. 이 스크립트의 책임은 staging readiness 보완이지 전체 schema migration이 아니다.

### Production-like 차단

스크립트는 다음 중 하나라도 production-like로 판단되면 `--allow-create-tables`를 거부해야 한다.

- `APP_ENV=production`
- `ENV=production`
- `FIT_CORE_ENV=production`
- `DATABASE_URL`에 운영 DB로 식별 가능한 host/name이 들어있는 경우는 자동 판정이 어렵기 때문에, P2-6에서는 env 기반 차단을 우선 구현한다.

운영 DB 식별 휴리스틱은 오탐/미탐 위험이 있으므로 optional warning 수준으로만 둔다. 강제 차단은 명시 환경변수 기반으로 시작한다.

### DB 확인 방식

권장 구현:

- script가 `database.engine`을 import한다.
- `sqlalchemy.inspect(engine).has_table("routine_feedback")`로 존재 여부를 확인한다.
- DB connection 실패는 readiness failure로 처리하고 원본 connection string, credentials, user id는 출력하지 않는다.

직접 HTTP `/api/ai/routine-feedback` probe로 readiness를 확인하지 않는다. 이유:

- readiness 확인이 seed row를 생성하면 안 된다.
- missing table과 API validation/runtime 오류를 명확히 구분하기 어렵다.
- DB readiness는 스크립트 실행 전제이므로 DB 직접 확인이 더 정확하다.

### 앱 Startup 정책

P2-6에서는 `main.py` startup behavior를 변경하지 않는다.

- production behavior 변경 금지.
- lifespan에 `Base.metadata.create_all()`을 추가하지 않는다.
- DB 생성은 migration/manual DDL 또는 staging script의 명시 dev/staging 옵션으로만 수행한다.

## 4. Seed Idempotency Policy

### Prefix

staging seed row는 deterministic `routine_draft_id` prefix `staging-seed-`를 사용한다.

현재 `SEED_ROWS`의 `routine_draft_id`:

- `staging-seed-001`
- `staging-seed-002`
- `staging-seed-003`
- `staging-seed-004`
- `staging-seed-005`
- `staging-seed-006`
- `staging-seed-007`

P2-6 구현은 `SEED_PREFIX = "staging-seed-"`와 `EXPECTED_SEED_COUNT = len(SEED_ROWS)`를 명시 상수로 둔다.

### 기본 seed 정책

기본 `seed`는 idempotent skip 정책을 사용한다.

- 기존 `routine_draft_id in SEED_ROWS` row가 없으면 삽입한다.
- 이미 존재하면 삽입하지 않고 `seed_rows_existing`에 집계한다.
- 같은 seed id에 여러 row가 이미 존재하면 추가 삽입하지 않고 readiness warning 또는 failure로 표시한다.

권장 판정:

- 각 expected seed id별 row count가 `0` 또는 `1`: seed 가능.
- 어떤 expected seed id라도 count가 `> 1`: `seed_readiness_status = "failed_duplicate_seed_rows"`로 fail fast. 사용자는 `--reset-seed`로 복구한다.

이 정책은 append-only production feedback API 계약을 바꾸지 않으면서 staging seed만 안전하게 만든다.

### Reset 정책

`--reset-seed`가 있으면 script는 seed 삽입 전에 staging seed row만 삭제한다.

삭제 대상:

- `RoutineFeedback.routine_draft_id.like("staging-seed-%")`

삭제 금지:

- `routine_draft_id`가 `staging-seed-`로 시작하지 않는 모든 row
- 실제 사용자 feedback
- staging user id만 보고 삭제하는 방식

삭제 기준을 `user_id`가 아니라 `routine_draft_id` prefix로 제한하는 이유:

- staging run request user id와 feedback seed user id가 같거나 달라질 수 있다.
- user id는 개인정보 성격이 있어 report에 남기지 않는다.
- deterministic seed fixture의 소유권은 `routine_draft_id` prefix가 가장 명확하다.

### Append-only 유지 금지

P2-6 staging script에서는 seed append-only 누적을 허용하지 않는다.

이유:

- feedback aggregation은 row count를 신호 강도로 사용한다.
- 동일 seed를 반복 삽입하면 `feedback_count`, skip/edit count, confidence가 인위적으로 증가한다.
- P2-5 staging 결과의 `avg_feedback_adjusted_count`가 seed 실행 횟수에 따라 달라질 수 있다.

단, production API 자체의 duplicate submission allowed 계약은 변경하지 않는다.

## 5. Script Command Design

대상 파일: `scripts/run_feedback_ranking_staging_test.py`

### `check-db`

목적: staging 실행 전 DB readiness를 단독 확인한다.

예시:

```bash
python scripts/run_feedback_ranking_staging_test.py check-db
python scripts/run_feedback_ranking_staging_test.py check-db --allow-create-tables
python scripts/run_feedback_ranking_staging_test.py check-db --output tests/evaluation/.artifacts/db-readiness.json
```

옵션:

- `--allow-create-tables`: dev/staging/test에서 missing `routine_feedback` 테이블 생성 허용.
- `--output`: readiness JSON report 저장.
- `--dry-run`: 생성 가능 여부만 판단하고 실제 생성하지 않음.

출력 JSON 필드:

```json
{
  "db_readiness_status": "pass",
  "db_table_exists": true,
  "db_table_created": false,
  "allow_create_tables": false,
  "production_like_blocked": false,
  "error_category": null
}
```

상태값:

- `pass`
- `created`
- `failed_missing_table`
- `failed_connection`
- `failed_production_like_create_blocked`

### `seed`

목적: deterministic staging feedback row를 안전하게 준비한다.

예시:

```bash
python scripts/run_feedback_ranking_staging_test.py seed --base-url http://127.0.0.1:8000
python scripts/run_feedback_ranking_staging_test.py seed --reset-seed
python scripts/run_feedback_ranking_staging_test.py seed --reset-seed --dry-run
```

옵션:

- `--base-url`: 기존 API seed 전송 endpoint.
- `--user-id`: seed에 사용할 staging user id. report에는 원문 저장 금지.
- `--allow-create-tables`: seed 전 DB readiness에서만 사용.
- `--reset-seed`: `staging-seed-*` row 삭제 후 재삽입.
- `--dry-run`: 삭제/삽입 없이 planned action만 출력.
- `--output`: seed readiness JSON 저장.

동작 순서:

1. DB readiness 확인.
2. expected seed id별 기존 row count 조회.
3. duplicate seed row가 있고 `--reset-seed`가 없으면 fail fast.
4. `--reset-seed`가 있으면 `staging-seed-*` row만 삭제.
5. 기존 row가 있는 seed id는 skip.
6. 없는 seed id만 API로 POST.
7. `verify-seed`와 동일한 검증을 수행.
8. JSON 결과를 stdout 및 `--output`에 기록.

`user_note`는 seed payload에 넣지 않는다.

### `verify-seed`

목적: seed row가 기대한 형태로 준비되었는지 확인한다.

예시:

```bash
python scripts/run_feedback_ranking_staging_test.py verify-seed
python scripts/run_feedback_ranking_staging_test.py verify-seed --output tests/evaluation/.artifacts/seed-readiness.json
```

검증 항목:

- expected seed id 7개가 각각 정확히 1 row인지 확인.
- expected exercise id signal이 row JSON에 존재하는지 확인.
- duplicate seed row가 없는지 확인.
- `user_note`가 seed row에 없거나 null인지 확인.
- report에 raw `user_id`를 저장하지 않는지 확인.

expected exercise signal 예:

- skipped: `34`, `114`
- edited originals: `34`, `37`, `76`, `99`, `126`
- edited replacements: `40`, `101`, `107`

상태값:

- `pass`
- `failed_missing_rows`
- `failed_duplicate_rows`
- `failed_unexpected_payload`
- `failed_user_note_present`

### `run`

`run`은 기존 group A/B 실행 전 DB와 seed readiness를 확인해야 한다.

- DB table missing이면 fail fast.
- group B 또는 seeded scenarios가 포함되는 모든 실행에서는 `verify-seed` pass를 요구한다.
- `run` 자체가 seed를 자동 삽입하지 않는다. seed는 명시 subcommand 책임으로 유지한다.
- `run` output에 readiness summary를 포함한다.

### `compare`

`compare`는 A/B raw report의 readiness 필드를 읽어 최종 report에 병합한다.

- A 또는 B 중 하나라도 `db_readiness_status`가 pass/created가 아니면 `recommendation = "inconclusive_readiness_failed"`.
- B가 feedback-enabled인데 `seed_verification_passed=false`이면 `recommendation = "inconclusive_seed_not_ready"`.
- privacy leak, hard violation 등 기존 keep-off 판단은 유지한다.

## 6. Report Additions

`run` raw output에 추가:

```json
{
  "db_readiness_status": "pass",
  "db_table_exists": true,
  "db_table_created": false,
  "seed_readiness_status": "pass",
  "seed_verification_passed": true,
  "seed_rows_expected": 7,
  "seed_rows_existing": 7,
  "seed_rows_inserted": 0,
  "seed_rows_deleted": 0,
  "seed_reset_performed": false
}
```

`seed` output에 추가:

```json
{
  "seed_readiness_status": "pass",
  "seed_rows_expected": 7,
  "seed_rows_existing": 0,
  "seed_rows_inserted": 7,
  "seed_rows_deleted": 0,
  "seed_reset_performed": false,
  "seed_verification_passed": true,
  "dry_run": false
}
```

`compare` final report에 추가:

```json
{
  "db_readiness_status_a": "pass",
  "db_readiness_status_b": "pass",
  "seed_readiness_status_a": "pass",
  "seed_readiness_status_b": "pass",
  "seed_verification_passed_b": true,
  "readiness_blocked": false
}
```

Markdown report에도 최소 필드를 추가한다.

- DB readiness: A/B
- Seed readiness: A/B
- Seed verification passed: B
- Recommendation

민감 정보 금지:

- raw `user_id`
- `DATABASE_URL`
- DB credentials
- `user_note`
- raw skipped/edited per-user maps

## 7. Privacy Policy

P2-6은 기존 privacy posture를 유지한다.

- `user_note`는 seed payload에 넣지 않는다.
- `user_note`는 readiness check, seed verification, report, telemetry에 포함하지 않는다.
- report에는 raw `user_id`를 저장하지 않는다.
- DB connection error report에는 sanitized `error_category`와 짧은 message만 남긴다. credentials 또는 full DSN을 남기지 않는다.
- privacy leak detector의 forbidden tokens는 기존 목록을 유지하고, readiness/seed report에도 적용한다.
- seed verification은 expected exercise id signal 존재만 확인하고 per-user feedback map을 저장하지 않는다.

## 8. Test Plan

### Unit tests for DB readiness

1. `routine_feedback` table exists -> readiness pass.
2. `routine_feedback` table missing + `allow_create=false` -> fail fast with `failed_missing_table`.
3. `routine_feedback` table missing + `allow_create=true` + non-production env -> table created and status `created`.
4. `routine_feedback` table missing + `allow_create=true` + `APP_ENV=production` -> fail fast with `failed_production_like_create_blocked`.
5. DB connection/inspection error -> sanitized `failed_connection`; no raw `DATABASE_URL`.

### Unit tests for seed idempotency

6. first seed inserts 7 rows.
7. repeated seed without reset inserts 0 rows and does not duplicate.
8. repeated seed with reset deletes only `staging-seed-*` rows and reinserts 7.
9. reset does not delete non-seed feedback rows.
10. duplicate existing rows for one seed id without reset fails with `failed_duplicate_seed_rows`.
11. dry-run seed reports planned inserts/deletes without mutating DB.

### Unit tests for verify-seed

12. verify-seed passes after seed.
13. verify-seed fails when expected rows are missing.
14. verify-seed fails when duplicate seed rows exist.
15. verify-seed fails when expected exercise ids are missing from skipped/edited payload.
16. verify-seed fails if a seed row has `user_note`.

### Script/CLI tests

17. `tests/test_feedback_ranking_telemetry.py::test_staging_script_help_commands` includes `check-db` and `verify-seed`.
18. `seed --help` includes `--reset-seed`, `--allow-create-tables`, `--dry-run`, `--output`.
19. `check-db --help` includes `--allow-create-tables`, `--dry-run`, `--output`.
20. `verify-seed --help` includes `--output`.

### Report tests

21. `run` output includes DB/seed readiness fields.
22. `compare` output includes A/B DB/seed readiness fields.
23. readiness failure in A or B results in `inconclusive_readiness_failed`.
24. B seed verification failure results in `inconclusive_seed_not_ready`.
25. privacy detector still passes after readiness fields are added.

### Existing regression tests

26. `tests/test_routine_feedback.py` continues passing, including duplicate submission append-only API behavior.
27. `tests/test_feedback_aggregation.py` continues passing.
28. `tests/test_feedback_ranking_telemetry.py` continues passing.
29. `tests/test_log_redaction.py` continues passing.
30. full `pytest` passes.

## 9. Risk Analysis

### Risk: Script creates tables in production-like DB

Mitigation:

- Default fail fast.
- `--allow-create-tables` required.
- production-like env blocks creation.
- Create only `RoutineFeedback.__table__`, not all metadata.
- Do not change `main.py` startup.

### Risk: Seed reset deletes real user feedback

Mitigation:

- Delete only `routine_draft_id LIKE 'staging-seed-%'`.
- Do not delete by `user_id`.
- Dry-run support shows row counts before mutation.
- Tests cover non-seed row preservation.

### Risk: Repeated seed inflates feedback aggregation

Mitigation:

- Default skip existing rows.
- Duplicate existing seed rows fail without reset.
- `--reset-seed` gives deterministic recovery.

### Risk: Report leaks private data

Mitigation:

- No raw user id in report.
- No `user_note`.
- No DB URL.
- Existing forbidden token privacy scan applies to run results.

### Risk: Readiness checks depend on DB direct access while seed uses API

Mitigation:

- This is intentional: table existence is a DB concern, while seed insertion should still exercise the public feedback API storage path.
- Tests should isolate DB helper logic so CLI behavior remains deterministic.

### Risk: No migration framework exists

Mitigation:

- P2-6 does not introduce a large migration framework.
- The readiness policy documents current state and makes staging script behavior explicit.
- Future migration adoption can replace manual/table-create path without changing seed policy.

## 10. Codex 구현용 최종 지시문

Implement P2-6 in `fit-core-ai` only. Do not change production ranking behavior, Gemini quota handling, candidate pool promotion, backend, or frontend.

1. Update `scripts/run_feedback_ranking_staging_test.py`.
2. Add DB readiness helpers using `sqlalchemy.inspect(database.engine).has_table("routine_feedback")`.
3. Add `check-db` subcommand with `--allow-create-tables`, `--dry-run`, and `--output`.
4. Missing table must fail fast by default.
5. `--allow-create-tables` may create only `RoutineFeedback.__table__` and must be blocked when `APP_ENV`, `ENV`, or `FIT_CORE_ENV` is `production`.
6. Do not add `Base.metadata.create_all()` to `main.py` startup.
7. Make `seed` idempotent:
   - Keep using the feedback API for inserts.
   - Use deterministic seed ids with prefix `staging-seed-`.
   - Insert only missing expected seed ids.
   - Fail on duplicate existing seed ids unless `--reset-seed` is supplied.
   - With `--reset-seed`, delete only rows whose `routine_draft_id` starts with `staging-seed-`, then reinsert expected rows.
   - Add `--dry-run`, `--reset-seed`, `--allow-create-tables`, and `--output`.
8. Add `verify-seed` subcommand:
   - expected 7 seed rows, one per expected seed id.
   - expected skipped/edited exercise ids present.
   - no seed row has `user_note`.
   - report must not contain raw `user_id`.
9. Make `run` check DB readiness and seed readiness before executing scenarios. Do not auto-seed from `run`.
10. Add readiness fields to raw run output and final compare output:
    - `db_table_exists`
    - `db_table_created`
    - `db_readiness_status`
    - `seed_rows_expected`
    - `seed_rows_existing`
    - `seed_rows_inserted`
    - `seed_rows_deleted`
    - `seed_reset_performed`
    - `seed_verification_passed`
    - `seed_readiness_status`
11. Update `compare` recommendation logic:
    - readiness failure -> `inconclusive_readiness_failed`
    - B seed verification failure -> `inconclusive_seed_not_ready`
    - keep existing privacy, hard violation, fallback, latency, critic score, and quota rules.
12. Extend tests:
    - DB readiness pass/missing/create/prod-blocked.
    - seed first run/repeat skip/reset/non-seed preservation/duplicate failure/dry-run.
    - verify-seed pass/missing/duplicate/payload mismatch/user_note failure.
    - CLI help includes `check-db` and `verify-seed`.
    - report contains readiness fields.
    - privacy/user_note tests continue passing.
13. Run focused tests first:
    - `pytest tests/test_feedback_ranking_telemetry.py tests/test_routine_feedback.py tests/test_feedback_aggregation.py -q`
14. Then run full `pytest`.


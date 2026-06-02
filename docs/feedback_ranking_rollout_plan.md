# P2-7 Feedback-aware Ranking Rollout Plan Spec

## 1. 설계 요약

P2-7은 P2-6까지 구현 및 검증된 feedback-aware ranking을 production default로 켜기 위한 스펙이 아니다. 이 문서는 staging limited rollout을 운영하기 위한 절차, flag 전략, telemetry 관측 기준, rollback 기준, quota 회복 후 재검증 계획, production promotion gate를 정의한다.

현재 완료 상태:

- P2-1 Feedback Aggregation Spec 완료.
- P2-2 Feedback Aggregation Helper 완료.
- P2-3 Feedback-aware Candidate Ranker Integration 완료.
- P2-4 Feedback-aware Ranking Evaluation 완료.
- P2-5 Feedback-aware Ranking Staging Rollout 구현 완료.
- P2-6 Feedback Ranking Staging Readiness Hardening 완료.
- 전체 pytest: `408 passed`.
- `ENABLE_FEEDBACK_AWARE_RANKING=false` 기본값 유지.
- candidate pool default는 `12` 유지.
- production behavior 변경 없음.

최근 P2-6 적용 후 staging 재실행 결과:

- Group A feedback off 완료.
- Group B feedback on 완료.
- Group B request count: `36`.
- failed requests: `0`.
- telemetry count: `36`.
- feedback enabled rate: `1.0`.
- candidate pool size: `12`.
- avg feedback adjusted count: `2.8333`.
- feedback query failed rate: `0.0`.
- p95 feedback query latency: `2ms`.
- avg critic score: `61.1667`.
- fallback rate: `1.0`.
- hard violation count: `0`.
- privacy leak detected: `false`.
- DB readiness: `pass`.
- seed readiness: `pass`.
- seed verification passed: `true`.
- readiness blocked: `false`.
- compare recommendation: `promote_feedback_ranking_candidate`.

해석:

- feedback-aware ranking mechanism은 offline, harness, staging readiness를 통과했으므로 staging limited rollout 후보로 승격 가능하다.
- 단, 최근 staging run의 `fallback_rate=1.0`은 최종 LLM 품질 개선 판단이 아직 부족하다는 caveat다.
- Gemini quota exhaustion 또는 fallback 지배 상황에서는 production default ON이 금지된다.
- 다음 단계는 production rollout이 아니라 staging internal/test user 대상 제한 rollout이다.

## 2. Rollout Scope

### 포함 범위

- Staging only.
- Internal/test users only.
- `ROUTINE_CANDIDATE_POOL_SIZE=12` 고정.
- `ENABLE_DYNAMIC_TEMPERATURE=false`.
- `ENABLE_RULE_CRITIC_TELEMETRY=true`.
- Treatment 환경에서만 `ENABLE_FEEDBACK_AWARE_RANKING=true`.
- P2-6 readiness hardening 사용:
  - `check-db`.
  - `seed --reset-seed`.
  - `verify-seed`.
  - run/compare readiness report fields.

### 제외 범위

- Production enable.
- Production default ON.
- Candidate pool 18 실험.
- Dynamic temperature rollout.
- LLM retry/backoff 변경.
- `user_note` NLP 분석 또는 ranking 반영.
- ML/LTR ranking.
- Admin dashboard.
- UI 변경.
- Migration framework 대규모 도입.

## 3. Environment Matrix

| Environment | `ENABLE_FEEDBACK_AWARE_RANKING` | `ROUTINE_CANDIDATE_POOL_SIZE` | `ENABLE_RULE_CRITIC_TELEMETRY` | `ENABLE_DYNAMIC_TEMPERATURE` | 목적 |
|---|---:|---:|---:|---:|---|
| Staging control | `false` | `12` | `true` | `false` | Group A baseline |
| Staging treatment | `true` | `12` | `true` | `false` | Group B limited rollout |
| Production | `false` | `12` | optional/current policy | `false` unless separately approved | 기존 behavior 유지 |

운영 원칙:

- Production에서 `ENABLE_FEEDBACK_AWARE_RANKING` 기본값은 계속 `false`.
- Candidate pool size는 P2-7 범위에서 항상 `12`.
- Percentage rollout 또는 user targeting 기능은 현재 명시 구현되어 있지 않다. 따라서 P2-7 limited rollout은 env-level staging 분리 또는 staging internal/test traffic으로만 제한한다.
- Production canary는 user targeting capability가 생기기 전까지 실행하지 않는다.

## 4. Preflight Checklist

실행 전 필수 확인:

- [ ] `pytest tests/ -q` 통과.
- [ ] `/docs` reachable.
- [ ] `/api/dev/logs` reachable. Production에서는 `/api/dev/logs`가 404일 수 있으므로 staging에서만 확인한다.
- [ ] `python scripts/run_feedback_ranking_staging_test.py check-db` 결과 `db_readiness_status=pass`.
- [ ] `python scripts/run_feedback_ranking_staging_test.py seed --reset-seed --base-url http://127.0.0.1:8000` 결과 `seed_verification_passed=true`.
- [ ] `python scripts/run_feedback_ranking_staging_test.py verify-seed` 결과 `seed_readiness_status=pass`.
- [ ] Telemetry probe에서 Group A는 `feedback_enabled=false`.
- [ ] Telemetry probe에서 Group B는 `feedback_enabled=true`.
- [ ] Telemetry probe에서 양쪽 모두 `candidate_pool_size=12`.
- [ ] Privacy forbidden token scan pass:
  - raw `user_id` 없음.
  - `user_note` 없음.
  - raw feedback row 없음.
  - per-exercise feedback map 없음.
- [ ] 중복 server process 없음.
- [ ] old uvicorn process 정리 완료.
- [ ] Gemini quota 상태 확인.
- [ ] `routine_feedback` 테이블은 migration/manual DDL 또는 P2-6 dev/staging 전용 allow-create로 준비되어 있음.

중단 조건:

- `check-db`가 `failed_missing_table`이면 기본 중단.
- `--allow-create-tables`는 dev/staging에서만 명시 승인 후 사용.
- `APP_ENV`, `ENV`, `FIT_CORE_ENV` 중 하나라도 `production`이면 create 허용 금지.
- `seed_verification_passed=false`이면 run 금지.
- `candidate_pool_size != 12`이면 run 금지.
- probe의 `feedback_enabled`가 기대값과 다르면 run 금지.

## 5. Staging Execution Runbook

### Shared Readiness

```bash
python scripts/run_feedback_ranking_staging_test.py check-db
python scripts/run_feedback_ranking_staging_test.py seed --reset-seed --base-url http://127.0.0.1:8000
python scripts/run_feedback_ranking_staging_test.py verify-seed
```

기대 결과:

- `db_readiness_status=pass`.
- `db_table_exists=true`.
- `db_table_created=false` unless explicitly created in dev/staging.
- `seed_rows_expected=7`.
- `seed_verification_passed=true`.
- `missing_seed_count=0`.

### Group A: Staging Control

Server env:

```bash
ENABLE_FEEDBACK_AWARE_RANKING=false
ROUTINE_CANDIDATE_POOL_SIZE=12
ENABLE_RULE_CRITIC_TELEMETRY=true
ENABLE_DYNAMIC_TEMPERATURE=false
```

Run:

```bash
python scripts/run_feedback_ranking_staging_test.py run --group a --base-url http://127.0.0.1:8000 --output tests/evaluation/.artifacts/feedback-ranking-a.json
```

Gate:

- `feedback_enabled_rate_a=0.0`.
- All telemetry events have `candidate_pool_size=12`.
- `failed_requests=0`.
- `db_readiness_status=pass`.
- `seed_verification_passed=true`.

### Group B: Staging Treatment

Server env:

```bash
ENABLE_FEEDBACK_AWARE_RANKING=true
ROUTINE_CANDIDATE_POOL_SIZE=12
ENABLE_RULE_CRITIC_TELEMETRY=true
ENABLE_DYNAMIC_TEMPERATURE=false
```

Run:

```bash
python scripts/run_feedback_ranking_staging_test.py run --group b --base-url http://127.0.0.1:8000 --output tests/evaluation/.artifacts/feedback-ranking-b.json
```

Gate:

- `feedback_enabled_rate_b=1.0`.
- All telemetry events have `candidate_pool_size=12`.
- `failed_requests=0`.
- `avg_feedback_adjusted_count_b >= 1` on seeded/test users.
- `feedback_query_failed_rate_b=0.0`.
- `hard_violation_count_b=0`.
- `db_readiness_status=pass`.
- `seed_verification_passed=true`.

### Compare

```bash
python scripts/run_feedback_ranking_staging_test.py compare --a tests/evaluation/.artifacts/feedback-ranking-a.json --b tests/evaluation/.artifacts/feedback-ranking-b.json --output tests/evaluation/.artifacts/feedback-ranking-report.json --markdown tests/evaluation/.artifacts/feedback-ranking-report.md
```

Expected artifacts:

- `tests/evaluation/.artifacts/feedback-ranking-a.json`.
- `tests/evaluation/.artifacts/feedback-ranking-b.json`.
- `tests/evaluation/.artifacts/feedback-ranking-report.json`.
- `tests/evaluation/.artifacts/feedback-ranking-report.md`.

Acceptable recommendations for staging limited rollout review:

- `promote_feedback_ranking_candidate`: mechanism eligible for staging limited rollout.
- `inconclusive_quota_exhausted`: mechanism may still be operationally safe, but quality promotion requires quota revalidation.
- `inconclusive_no_adjustment`: seed/test traffic did not exercise adjustment; rerun with seed/readiness investigation.

Blocking recommendations:

- `keep_feedback_off`.
- `inconclusive_readiness_failed`.
- `inconclusive_seed_not_ready`.

## 6. Telemetry Monitoring Plan

Source:

- Staging `/api/dev/logs`.
- `[Telemetry]` events with `event="routine_generation_quality"`.
- Script artifacts under `tests/evaluation/.artifacts/`.

Required fields:

- `feedback_enabled`.
- `feedback_adjusted_candidate_count`.
- `feedback_positive_count`.
- `feedback_negative_count`.
- `feedback_abs_adjustment_avg`.
- `feedback_query_latency_ms`.
- `feedback_query_failed`.
- `feedback_query_error_category`.
- `hard_violation_count`.
- `fallback_used`.
- `fallback_reason`.
- `llm_error_type`.
- `repair_count`.
- `critic_score`.
- `critic_grade`.
- `candidate_pool_size`.
- `generation_latency_ms`.
- `privacy_leak_detected`.

Aggregation windows:

- 24h window: first-pass operational stability.
- 72h window: limited rollout stability before broader staging exposure.

Aggregate metrics:

- p50/p95 `feedback_query_latency_ms`.
- avg `feedback_adjusted_candidate_count`.
- avg `feedback_positive_count`.
- avg `feedback_negative_count`.
- avg `feedback_abs_adjustment_avg`.
- `feedback_query_failed_rate`.
- total `hard_violation_count`.
- `fallback_rate` and treatment-control delta.
- `quota_fallback_rate`.
- avg `critic_score` and treatment-control delta.
- `privacy_leak_count`.
- server 500 count.
- telemetry missing rate.
- `candidate_pool_size` distribution.

Monitoring cadence:

- During first 2 hours: inspect every 30 minutes.
- 24h window: inspect at least 3 times.
- 72h window: inspect at least daily and after any server restart.
- After each restart: run a probe and verify expected `feedback_enabled` and `candidate_pool_size`.

## 7. Pass Criteria

Staging limited rollout passes only if all conditions hold:

- `hard_violation_count = 0`.
- `privacy_leak_detected = false`.
- `feedback_query_failed_rate = 0`.
- p95 `feedback_query_latency_ms <= 50ms`.
- Treatment `fallback_rate` does not increase vs control by more than 5 percentage points.
- Treatment avg `critic_score` does not drop vs control by more than 5 points.
- `avg_feedback_adjusted_candidate_count >= 1` on seeded/test users.
- DB readiness and seed readiness pass.
- No server 500 increase.
- No blocked equipment hard constraint bypass.
- No pain hard constraint bypass.
- No DOMS/readiness hard constraint regression.
- No `candidate_pool_size != 12` telemetry.
- No `feedback_enabled` probe mismatch.

Current latest run status:

- Mechanism pass: yes.
- Readiness pass: yes.
- Privacy pass: yes.
- Latency pass: yes.
- Hard violation pass: yes.
- Quality pass: not fully established because recent run had `fallback_rate=1.0`.

## 8. Rollback Criteria

Immediate rollback if any condition occurs:

- `hard_violation_count > 0`.
- Privacy leak detected.
- `feedback_query_failed_rate > 0`.
- p95 `feedback_query_latency_ms > 100ms` sustained for 30 minutes or more.
- Treatment `fallback_rate` increase > 10 percentage points vs control.
- Treatment avg `critic_score` drop > 10 points vs control.
- Server 500 increase after enabling treatment.
- Probe `feedback_enabled` mismatch.
- DB readiness failure.
- Seed readiness failure in staging test flows.
- `candidate_pool_size != 12`.
- Any raw `user_note` in logs/report.
- Any raw `user_id` in report.
- Any raw feedback row in report.
- Any per-exercise feedback map in logs/report.
- Any hard constraint bypass involving pain, equipment, or DOMS/readiness.

Rollback procedure:

1. Set `ENABLE_FEEDBACK_AWARE_RANKING=false`.
2. Restart the staging server.
3. Run probe with:

```bash
python scripts/run_feedback_ranking_staging_test.py run --group a --base-url http://127.0.0.1:8000 --output tests/evaluation/.artifacts/feedback-ranking-rollback-probe.json
```

4. Confirm `feedback_enabled=false`.
5. Confirm `candidate_pool_size=12`.
6. Inspect at least 10 post-rollback telemetry events.
7. Confirm no new `feedback_query_failed` events.
8. Write an incident note containing:
   - trigger metric.
   - start/end time.
   - affected environment.
   - rollback commit/config.
   - follow-up action.

## 9. Quota Caveat / LLM Quality Revalidation

The latest staging run had `fallback_rate=1.0`, so it cannot prove LLM output quality improvement. It does prove the mechanism is active, fast, privacy-preserving in report shape, and does not introduce hard violations in the tested scenarios.

Quota caveat:

- Gemini quota exhaustion or fallback-dominated traffic can hide quality deltas.
- A `promote_feedback_ranking_candidate` recommendation in a fallback-dominated run should be interpreted as mechanism readiness, not production quality approval.
- Production default ON remains blocked until LLM-backed quality revalidation passes.

Revalidation after quota recovery:

- Rerun A/B with at least 50 requests per group; 100 requests per group preferred.
- Keep `candidate_pool_size=12`.
- Keep dynamic temperature off unless a separate rollout has approved it.
- Require `quota_fallback_rate < 0.2`.
- Prefer low overall fallback rate in both groups.
- Require `hard_violation_count=0`.
- Require privacy leak count `0`.
- Require `feedback_query_failed_rate=0`.
- Require p95 `feedback_query_latency_ms <= 50ms`.
- Require treatment avg `critic_score` no worse than control by more than 5.
- Require treatment fallback rate no higher than control by more than 5 percentage points.
- Require `avg_feedback_adjusted_candidate_count >= 1` for seeded/test users.

If revalidation is inconclusive:

- Do not promote to production.
- Keep staging treatment only.
- Increase scenario count or add more realistic internal/test feedback.
- Investigate LLM quota/fallback reason distribution before rerun.

## 10. Production Promotion Gate

Production ON is not allowed until every gate below is satisfied.

Required gates:

- Staging limited rollout stable for 24h minimum; 72h preferred.
- Quota recovered and A/B revalidation completed.
- `hard_violation_count=0`.
- privacy leak count `0`.
- `feedback_query_failed_rate=0`.
- p95 `feedback_query_latency_ms <= 50ms`.
- Treatment avg `critic_score` drop <= 5 vs control.
- Treatment fallback increase <= 5 percentage points vs control.
- No server 500 increase.
- No hard constraint bypass.
- Rollback drill completed.
- Feature flag canary strategy approved.
- Observability owner assigned.
- Support/operator runbook acknowledged.

Production rollout phases:

- Phase 0: default false. Current state.
- Phase 1: staging internal users only.
- Phase 2: staging broader test users.
- Phase 3: production canary for internal accounts only, only after user targeting exists.
- Phase 4: small percentage rollout, only if percentage rollout infrastructure exists.
- Phase 5: default on only after review.

Current system limitation:

- Percentage rollout and user targeting are not available yet in this code path.
- Until targeting exists, production canary and percentage rollout are not available.
- P2-7 is limited to env-level staging rollout.

## 11. Risk Analysis

### Quota Exhaustion

Risk: fallback-dominated runs can create false positive or false negative quality conclusions.

Mitigation:

- Treat fallback-dominated results as mechanism readiness only.
- Require quota revalidation before production promotion.

### Seed and Candidate ID Mismatch

Risk: seeded exercise IDs may not align with current candidate set or target scenarios.

Mitigation:

- Continue `verify-seed`.
- Monitor `avg_feedback_adjusted_candidate_count`.
- Fail if seeded/test users produce no adjustments.

### DB Table / Migration Readiness

Risk: `routine_feedback` table may be missing in a new staging DB.

Mitigation:

- Run `check-db` before every staging run.
- Fail fast by default.
- Allow table creation only in dev/staging with explicit approval.
- Do not add startup `create_all()`.

### Query Latency Increase

Risk: feedback aggregation query adds latency under real traffic.

Mitigation:

- Track p50/p95 `feedback_query_latency_ms`.
- Roll back if p95 exceeds sustained threshold.

### Feedback Overfitting

Risk: small seed/test sample overweights artificial preferences.

Mitigation:

- Keep rollout internal/test.
- Require broader staging test before production gate.
- Keep score adjustments bounded.

### Global Feedback Dilutes User Preference

Risk: global feedback may reduce personalization quality.

Mitigation:

- Monitor positive/negative counts and critic deltas.
- Require quality revalidation after quota recovery.

### Privacy Leak

Risk: feedback rows, user IDs, or user notes leak into telemetry/report.

Mitigation:

- Keep forbidden token scans.
- Never include `user_note`.
- Never report raw feedback rows or per-exercise maps.

### Hard Constraint Bypass

Risk: ranking adjustment could surface exercises blocked by pain, equipment, or DOMS/readiness.

Mitigation:

- Ranking adjustment must remain after hard filters.
- Monitor hard violations.
- Keep existing guard/regression tests.

### Process / Env Mismatch

Risk: old uvicorn process or wrong env produces false Group A/B results.

Mitigation:

- Check process ownership before run.
- Use probe telemetry.
- Abort on `feedback_enabled` mismatch.

### Duplicate Uvicorn Process

Risk: multiple servers on same or adjacent ports confuse test traffic.

Mitigation:

- Verify port `8000` listener.
- Stop stale staging processes before manual operator run.
- Re-probe after restart.

## 12. Artifact / Report Requirements

Required artifact:

- `docs/feedback_ranking_rollout_plan.md`.

Required run artifacts per staging rollout:

- Group A raw JSON:
  - `tests/evaluation/.artifacts/feedback-ranking-a.json`.
- Group B raw JSON:
  - `tests/evaluation/.artifacts/feedback-ranking-b.json`.
- Compare JSON:
  - `tests/evaluation/.artifacts/feedback-ranking-report.json`.
- Compare Markdown:
  - `tests/evaluation/.artifacts/feedback-ranking-report.md`.
- Optional incident note if rollback occurs.

Report must include:

- request counts.
- failed requests.
- telemetry counts.
- feedback enabled rates.
- candidate pool size.
- avg feedback adjusted count.
- feedback query failed rate.
- p50/p95 feedback query latency.
- avg critic score.
- fallback rate.
- hard violation count.
- privacy leak status.
- DB readiness status.
- seed readiness status.
- recommendation.

Report must not include:

- raw `user_id`.
- `user_note`.
- raw feedback rows.
- raw skipped/edited per-user maps.
- DB credentials.
- full `DATABASE_URL`.

## 13. Codex / Operator Checklist

Copy this checklist for each staging limited rollout.

### Preflight

- [ ] Confirm this is staging, not production.
- [ ] Confirm `ENABLE_FEEDBACK_AWARE_RANKING` production default remains `false`.
- [ ] Confirm candidate pool remains `12`.
- [ ] Run:

```bash
pytest tests/ -q
```

- [ ] Start Group A server with:

```bash
ENABLE_FEEDBACK_AWARE_RANKING=false
ROUTINE_CANDIDATE_POOL_SIZE=12
ENABLE_RULE_CRITIC_TELEMETRY=true
ENABLE_DYNAMIC_TEMPERATURE=false
```

- [ ] Check `/docs`.
- [ ] Run:

```bash
python scripts/run_feedback_ranking_staging_test.py check-db
python scripts/run_feedback_ranking_staging_test.py seed --reset-seed --base-url http://127.0.0.1:8000
python scripts/run_feedback_ranking_staging_test.py verify-seed
```

### Group A

- [ ] Run:

```bash
python scripts/run_feedback_ranking_staging_test.py run --group a --base-url http://127.0.0.1:8000 --output tests/evaluation/.artifacts/feedback-ranking-a.json
```

- [ ] Confirm `feedback_enabled=false`.
- [ ] Confirm `candidate_pool_size=12`.
- [ ] Confirm `failed_requests=0`.

### Group B

- [ ] Restart staging server with:

```bash
ENABLE_FEEDBACK_AWARE_RANKING=true
ROUTINE_CANDIDATE_POOL_SIZE=12
ENABLE_RULE_CRITIC_TELEMETRY=true
ENABLE_DYNAMIC_TEMPERATURE=false
```

- [ ] Check `/docs`.
- [ ] Run:

```bash
python scripts/run_feedback_ranking_staging_test.py check-db
python scripts/run_feedback_ranking_staging_test.py verify-seed
python scripts/run_feedback_ranking_staging_test.py run --group b --base-url http://127.0.0.1:8000 --output tests/evaluation/.artifacts/feedback-ranking-b.json
```

- [ ] Confirm `feedback_enabled=true`.
- [ ] Confirm `candidate_pool_size=12`.
- [ ] Confirm `avg_feedback_adjusted_count_b >= 1`.
- [ ] Confirm `feedback_query_failed_rate_b=0`.
- [ ] Confirm `hard_violation_count_b=0`.

### Compare

- [ ] Run:

```bash
python scripts/run_feedback_ranking_staging_test.py compare --a tests/evaluation/.artifacts/feedback-ranking-a.json --b tests/evaluation/.artifacts/feedback-ranking-b.json --output tests/evaluation/.artifacts/feedback-ranking-report.json --markdown tests/evaluation/.artifacts/feedback-ranking-report.md
```

- [ ] Confirm `readiness_blocked=false`.
- [ ] Confirm `privacy_leak_detected=false`.
- [ ] Confirm recommendation is not `keep_feedback_off`.
- [ ] If recommendation is `promote_feedback_ranking_candidate`, approve staging limited rollout only.
- [ ] If recommendation is `inconclusive_quota_exhausted`, keep staging only and schedule quota revalidation.

### 24h / 72h Monitoring

- [ ] Collect 24h telemetry window.
- [ ] Collect 72h telemetry window if 24h is stable.
- [ ] Track p50/p95 feedback query latency.
- [ ] Track feedback query failed rate.
- [ ] Track fallback delta.
- [ ] Track critic score delta.
- [ ] Track hard violations.
- [ ] Track privacy leak count.

### Rollback

- [ ] If rollback criterion triggers, set:

```bash
ENABLE_FEEDBACK_AWARE_RANKING=false
```

- [ ] Restart staging server.
- [ ] Confirm probe `feedback_enabled=false`.
- [ ] Inspect 10 post-rollback telemetry events.
- [ ] Write incident note.

### Promotion

- [ ] Do not enable production by default during P2-7.
- [ ] Wait for quota recovery.
- [ ] Rerun A/B with at least 50 requests per group; 100 preferred.
- [ ] Confirm all production promotion gates pass.
- [ ] Confirm user targeting or canary infrastructure exists before production canary.


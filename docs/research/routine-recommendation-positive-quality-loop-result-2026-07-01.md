# Routine Recommendation Positive Quality Loop Result

Date: 2026-07-01
Status: pass
Owner: Planner / Codex routine recommendation owner

## Summary

두 번째 루프에서는 안전성 통과와 추천 품질 통과를 분리했다.

이전 리포트는 `contractValid=true`, `hardViolationCount=0`, `failurePatterns=0`이면 대부분 통과로 보였다. 하지만 실제 제품 품질에서는 “위험하지 않다”와 “좋은 추천이다”가 다르다.

이번 변경으로 Gemma4 품질 리포트는 아래를 분리한다.

- `hardSafetyPassed`: 계약/하드 제약/금지 패턴 통과 여부
- `preferredQualityPassed`: 시나리오별 선호 패턴이 최소 기준 이상 보이는지
- `recommendation=pass_with_quality_gaps`: 안전은 통과했지만 추천 품질 고도화가 필요한 상태

## Changes Made

1. `scripts/run_gemma4_quality_seed.py`
   - `score-file` 결과에 preferred-pattern quality gate를 추가했다.
   - `passed/failed`는 hard safety gate로 유지했다.
   - `quality_passed/quality_failed`를 별도 집계한다.
   - `recommendation`을 `pass`, `pass_with_quality_gaps`, `needs_review`로 세분화했다.
   - `scoreEvidenceText`도 scoring 대상에 포함했다.

2. `scripts/run_gemma4_quality_seed_live.py`
   - live result row에 `scoreEvidenceText`를 추가했다.
   - routine block의 score reasons, boost rule codes, penalty rule codes, constraint codes, profile signal codes를 sanitized evidence로 요약한다.
   - 12개 Gemma4 seed scenario의 preferred pattern을 일부 deterministic tag로 자동 산출한다.

3. `tests/test_gemma4_quality_seed_v2.py`
   - hard safety는 통과하지만 preferred quality가 부족한 결과를 `pass_with_quality_gaps`로 표시하는 테스트를 추가했다.
   - preferred pattern이 보이면 `recommendation=pass`가 되는 테스트를 추가했다.

4. `tests/test_gemma4_quality_seed_live_runner.py`
   - live result row가 score evidence를 포함하는지 검증했다.
   - live result row의 preferred tags가 score-file에서 preferred hit로 잡히는지 검증했다.

5. `docs/gemma4_routine_quality_loop_spec.md`
   - hard safety와 preferred quality를 분리하는 기준을 루프 spec에 반영했다.

## Evidence

Seed / live runner scoring tests:

```bash
python3 -m pytest -q -s --capture=no \
  tests/test_gemma4_quality_seed_live_runner.py \
  tests/test_gemma4_quality_seed_v2.py
```

Result:

```text
9 passed
```

Contract / prompt / pipeline / guard tests:

```bash
python3 -m pytest -q -s --capture=no \
  tests/test_contract.py \
  tests/test_prompt_regression.py \
  tests/test_pipeline.py \
  tests/test_guard.py
```

Result:

```text
73 passed
```

Offline eval harness:

```bash
python3 -m pytest -q -s --capture=no tests/evaluation/test_offline_eval_harness.py
```

Result:

```text
17 passed
```

Compile check:

```bash
python3 -m compileall -q engines scripts/run_gemma4_quality_seed.py scripts/run_gemma4_quality_seed_live.py
```

Result:

```text
passed
```

Existing live result re-score:

```bash
python3 scripts/run_gemma4_quality_seed.py score-file \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v2.jsonl \
  --results tests/evaluation/.artifacts/gemma4-quality-seed-live/gemma4-v2-live-full-after-safety-guard-2026-07-01/gemma4-quality-seed-results-filled.json \
  --run-id gemma4-v2-live-rescore-preferred-quality-2026-07-01
```

Result:

```text
passed=12
failed=0
quality_passed=1
quality_failed=11
recommendation=pass_with_quality_gaps
```

Generated report:

- `tests/evaluation/.artifacts/gemma4-quality-seed/gemma4-v2-live-rescore-preferred-quality-2026-07-01/gemma4-quality-seed-report.json`
- `tests/evaluation/.artifacts/gemma4-quality-seed/gemma4-v2-live-rescore-preferred-quality-2026-07-01/gemma4-quality-seed-report.md`

## Interpretation

The existing saved live result remains safe, but it is not yet a strong positive-quality recommendation set.

This is expected because that saved result was generated before `scoreEvidenceText` and preferred tag derivation were added. Future live runs will expose positive quality patterns more clearly.

## Next Actions

1. Re-run live Gemma4 seed after the deterministic preferred tags are in place.
2. Compare:
   - hard safety pass rate
   - preferred quality pass rate
   - timeout rate
   - fallback/failed rate
3. Improve ranking/prompt only for scenarios that remain quality gaps.
4. Keep fine-tuning gated until failures are repeated and LLM-specific.

## Approval / Safety Notes

- No commit was created.
- No push was performed.
- No DB migration was applied.
- No active public contract was changed.
- This loop only improved evaluation, evidence, and internal quality reporting.

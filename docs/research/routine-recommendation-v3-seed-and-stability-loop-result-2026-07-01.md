# Routine Recommendation v3 Seed and Stability Loop Result

Date: 2026-07-01
Status: pass for deterministic candidate layer, all 6 new v3 targeted live smoke scenarios, and full 18-scenario v3 batched live run; single-process continuous live run is not recommended on the current local Gemma4 runtime
Owner: Planner / Codex routine recommendation owner

## Summary

이번 루프는 v2 품질 seed를 더 현실적인 복합 조건으로 확장하고, live Gemma4 평가를 반복 가능한 형태로 만들기 위한 작업이다.

핵심 결론:

- v3 quality seed를 18개 시나리오로 확장했다.
- 새 복합 시나리오는 무릎 통증 + 발목 제한 + 초보자, 손목 제한 push, 낮은 컨디션 + 홈트 + 짧은 시간, 어깨 수술 이력 + 홈트, 요추 통증 + 등/이두 DOMS + 짧은 pull, 상급자 고컨디션 + 효과/안전 tradeoff를 포함한다.
- candidate/ranking layer strict check는 `candidate_supports_quality 16`, `no_candidate_level_patterns 1`, `hard_stop 1`로 통과권이다.
- live runner와 scorer를 보강해 부분 scenario 실행, resume, latency/fallback 기준, partial scoring을 지원한다.
- 짧은 pull live smoke에서 처음에는 `llmTimeout` fallback이 10개 운동을 내려 짧은 시간 정책을 위반했다.
- fallback 생성기를 시간 예산과 보수적 운동 수 cap에 맞게 보정했다.
- 30분 이하 루틴의 기본 `targetExerciseCount`를 4에서 3으로 낮춰 짧은 루틴 정책을 명확히 했다.
- 보정 후 targeted live v3 smoke 신규 6개는 `6/6 pass`, `quality_passed 6/6`, `latency_passed 6/6`, `fallback_passed 6/6`이다.
- v3 18개 전체 live run은 단일 연속 프로세스에서는 로컬 Gemma4 runtime이 연속 timeout/fallback으로 불안정했다.
- 동일 18개를 3개 시나리오 단위 batch + cooldown 방식으로 실행하면 `18/18 pass`, `quality_passed 18/18`, `latency_passed 18/18`, `fallback_passed 18/18`이다.
- 수술/질환 이력의 의료진 허가 안내는 LLM-only rationale이 아니라 deterministic response warning으로 승격했다.
- 취약 시나리오 3개를 3회 반복한 stability run은 `3/3 stable pass`, `recommendation pass`이다.

## Loop Boundary

Objective:

- 운동 루틴 추천 AI가 여러 개인 조건이 겹친 상황에서도 안전하고 설명 가능한 후보를 받을 수 있는지 검증한다.
- live Gemma4 실행이 느리거나 fallback으로 떨어질 때도 결과를 잃지 않고, 실패 원인을 hard safety / preferred quality / latency / fallback으로 나눠 확인한다.

Inputs:

- `tests/fixtures/gemma4_routine_quality_eval_seed_v3.jsonl`
- local SQLite catalog: `fit_core.sqlite`
- local Gemma4 via Ollama: `gemma4:latest`

Success criteria:

- seed schema validation pass
- candidate layer strict check에서 `candidate_quality_gap=0`
- targeted live smoke에서 hard safety, latency, fallback 기준 pass
- 짧은 시간 시나리오에서 30분 이하 루틴은 3개 이하 운동 중심으로 구성

Stop criteria:

- live scenario timeout
- fallback used when `--disallow-fallback` is active
- hard violation detected
- latency above threshold
- candidate gap found in strict candidate layer

Approval gates:

- commit / push / deploy 없음
- DB migration 없음
- active public contract 변경 없음

## Changes Made

1. `tests/fixtures/gemma4_routine_quality_eval_seed_v3.jsonl`
   - v2 12개 시나리오에 6개 복합 시나리오를 추가했다.
   - 30분 이하 short-time 시나리오에 `short_time_over_3_exercises` failure pattern을 추가했다.

2. `scripts/run_gemma4_quality_seed.py`
   - `--min-preferred-hits`로 preferred pattern 최소 hit 기준을 조절한다.
   - `--max-elapsed-ms`로 latency 기준을 채점한다.
   - `--disallow-fallback`으로 fallback 사용을 실패로 처리한다.
   - `--scenario-id`로 부분 실행 결과만 채점할 수 있게 했다.
   - score report에 hard safety, preferred quality, latency, fallback pass/fail을 분리해 기록한다.

3. `scripts/run_gemma4_quality_seed_live.py`
   - v3 신규 시나리오의 preferred/failure tag를 산출한다.
   - 30분 이하 루틴에서 4개 이상 운동이 나오면 `short_time_over_3_exercises`로 기록한다.
   - 부분 실행, snapshot 저장, resume 실행을 지원한다.
   - `--cooldown-sec`로 scenario 사이 pause를 둘 수 있게 했다. 현재 로컬 Gemma4 runtime에서는 3개 단위 batch와 cooldown이 단일 연속 실행보다 안정적이다.

4. `scripts/combine_gemma4_quality_results.py`
   - 여러 live batch result를 하나의 `gemma4-quality-seed-results-filled.json`으로 병합한다.
   - seed order를 기준으로 scenario 순서를 복원하고, duplicate scenarioId와 missing seed scenario를 명시한다.

5. `scripts/run_candidate_quality_seed.py`
   - v3 신규 시나리오의 candidate-level quality tag를 추가했다.
   - strict 후보층 평가에서 후보와 출력 전용 패턴을 분리한다.

6. `scripts/summarize_gemma4_quality_stability.py`
   - 여러 live result 파일을 비교해 반복 실행 안정성을 요약한다.
   - selection drift, pass count, quality count, latency, fallback을 scenario 단위로 비교한다.

7. `engines/prescription/adjustments.py`
   - 30분 이하 루틴의 기본 target exercise count를 3으로 낮췄다.

8. `engines/fallback.py`
   - fallback 루틴은 일반 LLM 목표보다 더 보수적인 운동 수 cap을 사용한다.
   - 25분 이하는 최대 2개, 30분 이하는 최대 3개 운동으로 제한한다.
   - 후보를 추가하기 전에 서버 시간 모델로 `timeAvailableMin` 초과 여부를 확인하고, 필요 시 세트 수를 줄인다.

9. Tests
   - v3 seed validation, candidate tag coverage, partial live score, fallback short-time cap, latency/fallback scorer, stability summary를 회귀테스트로 고정했다.

10. `engines/routine_pipeline.py`
   - `condition_policies`에 수술 이력, `professionalClearance=false`, `needs_clearance` 류 상태가 있으면 최종 응답 `warnings`에 deterministic clearance notice를 추가한다.
   - 적용 문구: `수술/질환 이력이 있는 부위는 의료진 허가 범위 안에서만 진행하세요.`
   - 이 warning은 success와 fallback 경로 모두에서 붙는다.

## Evidence

Seed validation:

```bash
python3 scripts/run_gemma4_quality_seed.py validate \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v3.jsonl
```

Result:

```text
scenario_count: 18
hard_constraint_count: 42
failure_pattern_count: 66
rubric_item_count: 55
```

Candidate layer strict check:

```bash
python3 scripts/run_candidate_quality_seed.py \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v3.jsonl \
  --output-dir tests/evaluation/.artifacts/candidate-quality-seed \
  --run-id candidate-quality-v3-strict2-target3-20260701 \
  --min-preferred-hits 2
```

Result:

```text
candidate_supports_quality: 16
no_candidate_level_patterns: 1
hard_stop: 1

candidate_layer_ok: 17
deterministic_guard_ok: 1
```

Report:

- `tests/evaluation/.artifacts/candidate-quality-seed/candidate-quality-v3-strict2-target3-20260701/candidate-quality-seed-report.json`
- `tests/evaluation/.artifacts/candidate-quality-seed/candidate-quality-v3-strict2-target3-20260701/candidate-quality-seed-report.md`

Gemma4 readiness:

```bash
python3 scripts/check_local_llm_readiness.py --timeout-sec 120 --strict
```

Result:

```text
readiness_status: pass
probe_success: true
probe_latency_ms: 30594
```

Initial targeted live smoke before fallback/time cap fix:

```bash
python3 scripts/run_gemma4_quality_seed_live.py \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v3.jsonl \
  --output-dir tests/evaluation/.artifacts/gemma4-quality-seed \
  --run-id gemma4-v3-new-scenarios-smoke-20260701 \
  --scenario-id knee-pain-limited-ankle-beginner-001 \
  --scenario-id shoulder-surgery-home-push-001 \
  --scenario-id upper-back-doms-lower-back-pull-short-001 \
  --scenario-timeout-sec 240
```

Finding:

```text
knee-pain-limited-ankle-beginner-001: success, exerciseCount=5, elapsedMs=82617
shoulder-surgery-home-push-001: success, exerciseCount=5, elapsedMs=75602
upper-back-doms-lower-back-pull-short-001: fallback, reason=llmTimeout, exerciseCount=10, elapsedMs=240033
```

This exposed a deterministic fallback policy gap: timeout fallback could ignore short-time exercise count expectations.

Targeted live smoke after target count and fallback fix:

```bash
python3 scripts/run_gemma4_quality_seed_live.py \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v3.jsonl \
  --output-dir tests/evaluation/.artifacts/gemma4-quality-seed \
  --run-id gemma4-v3-new-scenarios-smoke-target3-20260701 \
  --scenario-id knee-pain-limited-ankle-beginner-001 \
  --scenario-id shoulder-surgery-home-push-001 \
  --scenario-id upper-back-doms-lower-back-pull-short-001 \
  --scenario-timeout-sec 240
```

Result:

```text
knee-pain-limited-ankle-beginner-001: success, exerciseCount=5, elapsedMs=62305
shoulder-surgery-home-push-001: success, exerciseCount=5, elapsedMs=63764
upper-back-doms-lower-back-pull-short-001: success, exerciseCount=3, elapsedMs=62139
completedScenarioCount: 3
failedScenarioCount: 0
```

Strict score for targeted live smoke:

```bash
python3 scripts/run_gemma4_quality_seed.py score-file \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v3.jsonl \
  --results tests/evaluation/.artifacts/gemma4-quality-seed/gemma4-v3-new-scenarios-smoke-target3-20260701/gemma4-quality-seed-results-filled.json \
  --output-dir tests/evaluation/.artifacts/gemma4-quality-seed \
  --run-id gemma4-v3-new-scenarios-smoke-target3-20260701-score \
  --scenario-id knee-pain-limited-ankle-beginner-001 \
  --scenario-id shoulder-surgery-home-push-001 \
  --scenario-id upper-back-doms-lower-back-pull-short-001 \
  --min-preferred-hits 1 \
  --max-elapsed-ms 180000 \
  --disallow-fallback
```

Result:

```text
passed: 3
failed: 0
quality_passed: 3
quality_failed: 0
latency_passed: 3
latency_failed: 0
fallback_passed: 3
fallback_failed: 0
recommendation: pass
```

Report:

- `tests/evaluation/.artifacts/gemma4-quality-seed/gemma4-v3-new-scenarios-smoke-target3-20260701-score/gemma4-quality-seed-report.json`
- `tests/evaluation/.artifacts/gemma4-quality-seed/gemma4-v3-new-scenarios-smoke-target3-20260701-score/gemma4-quality-seed-report.md`

Remaining new v3 live smoke:

```bash
python3 scripts/run_gemma4_quality_seed_live.py \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v3.jsonl \
  --output-dir tests/evaluation/.artifacts/gemma4-quality-seed \
  --run-id gemma4-v3-remaining-new-scenarios-smoke-20260701 \
  --scenario-id wrist-limited-push-001 \
  --scenario-id low-readiness-home-short-fullbody-001 \
  --scenario-id advanced-high-readiness-effect-safe-001 \
  --scenario-timeout-sec 240
```

Result:

```text
wrist-limited-push-001: success, exerciseCount=5, elapsedMs=51808
low-readiness-home-short-fullbody-001: success, exerciseCount=3, elapsedMs=49541
advanced-high-readiness-effect-safe-001: success, exerciseCount=6, elapsedMs=80068
completedScenarioCount: 3
failedScenarioCount: 0
```

Combined score for all 6 new v3 scenarios:

```text
passed: 6
failed: 0
quality_passed: 6
quality_failed: 0
latency_passed: 6
latency_failed: 0
fallback_passed: 6
fallback_failed: 0
recommendation: pass
```

Combined reports:

- `tests/evaluation/.artifacts/gemma4-quality-seed/gemma4-v3-new-six-scenarios-combined-20260701/gemma4-quality-seed-results-filled.json`
- `tests/evaluation/.artifacts/gemma4-quality-seed/gemma4-v3-new-six-scenarios-combined-20260701-score/gemma4-quality-seed-report.json`
- `tests/evaluation/.artifacts/gemma4-quality-seed/gemma4-v3-new-six-scenarios-combined-20260701-score/gemma4-quality-seed-report.md`

Full single-process v3 live run attempt:

```bash
python3 scripts/run_gemma4_quality_seed_live.py \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v3.jsonl \
  --output-dir tests/evaluation/.artifacts/gemma4-quality-seed \
  --run-id gemma4-v3-full-live-20260701 \
  --scenario-timeout-sec 240 \
  --resume
```

Observed stop condition:

```text
lower-back-legs-001: success, elapsedMs=102104
limited-ankle-legs-001: fallback, reason=llmTimeout, elapsedMs=240021
shoulder-push-001: fallback, reason=llmTimeout, elapsedMs=240010
low-readiness-fullbody-001: fallback, reason=llmTimeout, elapsedMs=240028
```

Partial score:

```text
passed: 1
failed: 3
quality_passed: 4
quality_failed: 0
latency_failed: 3
fallback_failed: 3
recommendation: needs_review
```

Interpretation: hard safety and quality tags were acceptable, but the local Gemma4 runtime did not tolerate one long continuous 18-scenario process under the 240-second per-scenario timeout.

Full v3 batched live run:

```bash
python3 scripts/run_gemma4_quality_seed_live.py \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v3.jsonl \
  --output-dir tests/evaluation/.artifacts/gemma4-quality-seed \
  --run-id gemma4-v3-old-batch-01-20260701 \
  --scenario-timeout-sec 240 \
  --cooldown-sec 20 \
  --scenario-id lower-back-legs-001 \
  --scenario-id limited-ankle-legs-001 \
  --scenario-id shoulder-push-001
```

The same 3-scenario batch pattern was repeated for the remaining old v3 scenarios, then combined with the already-passing 6 new v3 scenarios:

```bash
python3 scripts/combine_gemma4_quality_results.py \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v3.jsonl \
  --output-dir tests/evaluation/.artifacts/gemma4-quality-seed \
  --run-id gemma4-v3-full-batched-combined-20260701 \
  --results tests/evaluation/.artifacts/gemma4-quality-seed/gemma4-v3-old-batch-01-20260701/gemma4-quality-seed-results-filled.json \
  --results tests/evaluation/.artifacts/gemma4-quality-seed/gemma4-v3-old-batch-02-20260701/gemma4-quality-seed-results-filled.json \
  --results tests/evaluation/.artifacts/gemma4-quality-seed/gemma4-v3-old-batch-03-20260701/gemma4-quality-seed-results-filled.json \
  --results tests/evaluation/.artifacts/gemma4-quality-seed/gemma4-v3-old-batch-04-20260701/gemma4-quality-seed-results-filled.json \
  --results tests/evaluation/.artifacts/gemma4-quality-seed/gemma4-v3-new-six-scenarios-combined-20260701/gemma4-quality-seed-results-filled.json
```

Combined result:

```text
scenarioCount: 18
completedScenarioCount: 18
failedScenarioCount: 0
missingSeedScenarioIds: []
```

Strict score for full v3 batched live run:

```bash
python3 scripts/run_gemma4_quality_seed.py score-file \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v3.jsonl \
  --results tests/evaluation/.artifacts/gemma4-quality-seed/gemma4-v3-full-batched-combined-20260701/gemma4-quality-seed-results-filled.json \
  --output-dir tests/evaluation/.artifacts/gemma4-quality-seed \
  --run-id gemma4-v3-full-batched-combined-20260701-score \
  --min-preferred-hits 1 \
  --max-elapsed-ms 180000 \
  --disallow-fallback
```

Result:

```text
passed: 18
failed: 0
quality_passed: 18
quality_failed: 0
latency_passed: 18
latency_failed: 0
fallback_passed: 18
fallback_failed: 0
recommendation: pass
```

Reports:

- `tests/evaluation/.artifacts/gemma4-quality-seed/gemma4-v3-full-batched-combined-20260701/gemma4-quality-seed-results-filled.json`
- `tests/evaluation/.artifacts/gemma4-quality-seed/gemma4-v3-full-batched-combined-20260701-score/gemma4-quality-seed-report.json`
- `tests/evaluation/.artifacts/gemma4-quality-seed/gemma4-v3-full-batched-combined-20260701-score/gemma4-quality-seed-report.md`

Deterministic professional clearance warning:

```text
condition_policies에 surgeryHistory / needs_clearance / professionalClearance=false가 있으면
LLM 출력과 무관하게 response.warnings에 아래 문구를 추가한다.

수술/질환 이력이 있는 부위는 의료진 허가 범위 안에서만 진행하세요.
```

Regression tests:

```bash
pytest -s --capture=no -q \
  tests/test_pipeline.py::TestHappyPath::test_professional_clearance_warning_is_deterministic_on_success \
  tests/test_pipeline.py::TestHappyPath::test_professional_clearance_warning_is_deterministic_on_fallback \
  tests/test_gemma4_quality_seed_live_runner.py::test_live_result_row_exposes_preferred_quality_tags_for_score_file
```

Result:

```text
3 passed
```

Fragile scenario repeat stability:

```bash
python3 scripts/run_gemma4_quality_seed_live.py \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v3.jsonl \
  --output-dir tests/evaluation/.artifacts/gemma4-quality-seed \
  --run-id gemma4-v3-fragile-repeat-01-20260701 \
  --scenario-timeout-sec 240 \
  --cooldown-sec 20 \
  --scenario-id shoulder-surgery-home-push-001 \
  --scenario-id upper-back-doms-lower-back-pull-short-001 \
  --scenario-id low-readiness-home-short-fullbody-001
```

The same command shape was repeated for `repeat-02` and `repeat-03`, then summarized with selected scenario filtering:

```bash
python3 scripts/summarize_gemma4_quality_stability.py \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v3.jsonl \
  --output-dir tests/evaluation/.artifacts/gemma4-quality-seed \
  --run-id gemma4-v3-fragile-repeat-stability-selected-20260701 \
  --scenario-id shoulder-surgery-home-push-001 \
  --scenario-id upper-back-doms-lower-back-pull-short-001 \
  --scenario-id low-readiness-home-short-fullbody-001 \
  --results tests/evaluation/.artifacts/gemma4-quality-seed/gemma4-v3-fragile-repeat-01-20260701/gemma4-quality-seed-results-filled.json \
  --results tests/evaluation/.artifacts/gemma4-quality-seed/gemma4-v3-fragile-repeat-02-20260701/gemma4-quality-seed-results-filled.json \
  --results tests/evaluation/.artifacts/gemma4-quality-seed/gemma4-v3-fragile-repeat-03-20260701/gemma4-quality-seed-results-filled.json \
  --min-preferred-hits 1 \
  --max-elapsed-ms 180000 \
  --disallow-fallback
```

Result:

```text
runCount: 3
scenarioCount: 3
stablePassed: 3
unstableOrFailed: 0
recommendation: pass
```

Reports:

- `tests/evaluation/.artifacts/gemma4-quality-seed/gemma4-v3-fragile-repeat-stability-selected-20260701/gemma4-quality-stability-report.json`
- `tests/evaluation/.artifacts/gemma4-quality-seed/gemma4-v3-fragile-repeat-stability-selected-20260701/gemma4-quality-stability-report.md`

Regression tests:

```bash
python3 -m pytest -q -s --capture=no \
  tests/test_fallback.py \
  tests/test_prompt_regression.py \
  tests/test_pipeline.py \
  tests/test_contract.py \
  tests/test_gemma4_quality_seed_v2.py \
  tests/test_gemma4_quality_seed_v3.py \
  tests/test_gemma4_quality_stability.py \
  tests/test_candidate_quality_seed.py \
  tests/test_combine_gemma4_quality_results.py \
  tests/test_gemma4_quality_seed_live_runner.py
```

Result:

```text
124 passed
```

Compile and diff hygiene:

```bash
python3 -m compileall -q \
  engines/fallback.py \
  engines/prescription/adjustments.py \
  scripts/run_gemma4_quality_seed.py \
  scripts/run_gemma4_quality_seed_live.py \
  scripts/combine_gemma4_quality_results.py \
  scripts/run_candidate_quality_seed.py \
  scripts/summarize_gemma4_quality_stability.py \
  tests/test_fallback.py \
  tests/test_prompt_regression.py \
  tests/test_gemma4_quality_seed_v2.py \
  tests/test_gemma4_quality_seed_v3.py \
  tests/test_gemma4_quality_stability.py
```

```bash
git diff --check
```

Result:

```text
passed
```

## Current Interpretation

The candidate/ranking layer is now strong enough to support the v3 complex scenarios. The main residual risk is not candidate availability or prompt quality, but local LLM runtime stability under long continuous runs.

Observed improvement:

- The original short pull fallback failure was caused by deterministic fallback not enforcing short-time exercise count/time budget strongly enough.
- After reducing 30-minute `targetExerciseCount` to 3 and enforcing fallback caps, the same scenario generated a 3-exercise live result and passed strict scoring.

Observed remaining quality gaps:

- `shoulder-surgery-home-push-001` passed hard safety, but did not always surface `medical_clearance_notice` in the live preferred tags.
- `upper-back-doms-lower-back-pull-short-001` passed hard safety, but did not always include `supported_row`.

These are not immediate blockers, but they are good candidates for the next prompt/rationale refinement loop.

Operational rule:

- Do not use one long local Gemma4 process as the default proof path for full-seed evaluation.
- Use 3-scenario batches with `--cooldown-sec 20`, then combine and score the batch outputs.
- Treat a single-process continuous timeout as a runtime stability signal, not as direct evidence that the recommendation policy is unsafe.
- Treat professional clearance / surgery-history warnings as deterministic response policy, not as LLM-only copy.

## Next Actions

1. Consider tightening live preferred quality thresholds per scenario after two repeated runs are available.

2. Keep the v3 full batched run as the current release-quality proof path unless a faster local model runtime or tuned Gemma4 serving profile is introduced.

3. If the product UI wants stronger user-facing safety, expose deterministic `warnings` more prominently in the routine draft screen.

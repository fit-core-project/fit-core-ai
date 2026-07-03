# Routine Recommendation v4 Batch Harness Result

Date: 2026-07-03
Owner: Planner / AI routine core owner
Scope: v4 evaluation harness improvement after baseline

## Summary

The v4 loop now separates candidate-layer evidence from live LLM output quality and provides a bounded batch execution plan for Gemma4 live evaluation.

Key improvements:

1. Added `candidatePreferredPatterns` for candidate-layer-only quality evidence.
2. Added `scripts/plan_gemma4_quality_batches.py` to split the 18-scenario v3 seed into bounded live/score batches.
3. Confirmed latency-aware scoring works with `--max-elapsed-ms`.
4. Re-ran candidate quality after the short-time push update.

## Why This Matters

The previous v4 baseline showed two issues:

- Full 18-scenario live Gemma4 evaluation is too slow as one uninterrupted run.
- `short-time-push-001` had no candidate-level preferred pattern because all preferred patterns were output-only behaviors.

The fix keeps responsibilities clean:

- `candidatePreferredPatterns`: candidate/ranking layer evidence.
- `preferredPatterns`: final live output quality.
- `failurePatterns`: hard safety, contract, and behavior violations.

This prevents candidate diagnostics from polluting live model scoring.

## Changes

### 1. Candidate-Layer Pattern Separation

`tests/fixtures/gemma4_routine_quality_eval_seed_v3.jsonl` now allows scenario rows to include:

```json
"candidatePreferredPatterns": ["short_time_candidate_pool"]
```

The candidate-quality runner uses `candidatePreferredPatterns` when present. Otherwise it falls back to the old behavior of filtering `preferredPatterns` by output-only tags.

### 2. Short-Time Push Candidate Support

`scripts/run_candidate_quality_seed.py` now derives:

- `short_time_candidate_pool`
- `main_compound_then_accessory`

for `short-time-push-001` when the request is a short-time push scenario and ranked candidates exist.

This does not mean the final routine is good. The final live output must still pass:

- `two_to_three_exercises`
- `main_compound_then_accessory`
- `short_time_reason`
- no `short_time_over_3_exercises`
- no `six_exercise_plan`

### 3. Batch Plan Tool

New script:

```bash
python3 scripts/plan_gemma4_quality_batches.py \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v3.jsonl \
  --run-prefix gemma4-v4-batched-2026-07-03 \
  --max-elapsed-ms 120000 \
  --scenario-timeout-sec 240 \
  --resume
```

Output:

- `tests/evaluation/.artifacts/gemma4-quality-batches/gemma4-v4-batched-2026-07-03/gemma4-quality-batch-plan.json`
- `tests/evaluation/.artifacts/gemma4-quality-batches/gemma4-v4-batched-2026-07-03/gemma4-quality-batch-plan.md`

Default batches:

| Batch | Scenarios | Purpose |
|---|---:|---|
| `safety-core` | 3 | Core pain/mobility/push safety smoke |
| `time-equipment` | 3 | Short time and equipment restriction behavior |
| `beginner-joint` | 3 | Beginner, knee, ankle, and wrist constraints |
| `medical-profile` | 4 | Surgery, profile injury, DOMS, and hard-stop policies |
| `advanced-effect` | 5 | Pull, effect-vs-risk, anthropometry, advanced readiness |

## Evidence

### Seed Validation

```bash
python3 scripts/run_gemma4_quality_seed.py validate \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v3.jsonl
```

Result:

- scenario count: `18`
- hard constraints: `42`
- failure patterns: `66`
- rubric items: `55`

### Candidate Quality After Update

```bash
python3 scripts/run_candidate_quality_seed.py \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v3.jsonl \
  --output-dir tests/evaluation/.artifacts/candidate-quality-v4-after-batch-plan \
  --run-id candidate-quality-v4-after-batch-plan-2026-07-03
```

Result:

- `candidate_supports_quality`: `17`
- `hard_stop`: `1`
- `candidate_layer_ok`: `17`
- `deterministic_guard_ok`: `1`

Report:

- `tests/evaluation/.artifacts/candidate-quality-v4-after-batch-plan/candidate-quality-v4-after-batch-plan-2026-07-03/candidate-quality-seed-report.md`

### Latency-Gated Partial Live Score

```bash
python3 scripts/run_gemma4_quality_seed.py score-file \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v3.jsonl \
  --results tests/evaluation/.artifacts/gemma4-quality-seed-live-v4-baseline/gemma4-v4-baseline-live-2026-07-03/gemma4-quality-seed-results-filled.json \
  --output-dir tests/evaluation/.artifacts/gemma4-quality-seed-v4-score \
  --run-id gemma4-v4-baseline-partial-latency-gated-2026-07-03 \
  --scenario-id lower-back-legs-001,limited-ankle-legs-001,shoulder-push-001 \
  --max-elapsed-ms 120000
```

Result:

- scored: `3 / 3`
- hard safety passed: `3 / 3`
- preferred quality passed: `3 / 3`
- latency passed: `2 / 3`
- latency failed: `1 / 3`
- failed scenario due to latency: `lower-back-legs-001` at `196462 ms`

Report:

- `tests/evaluation/.artifacts/gemma4-quality-seed-v4-score/gemma4-v4-baseline-partial-latency-gated-2026-07-03/gemma4-quality-seed-report.md`

## Test Evidence

```bash
python3 -m pytest -q -s --capture=no \
  tests/test_candidate_quality_seed.py \
  tests/test_gemma4_quality_batch_plan.py \
  tests/test_gemma4_quality_seed_v3.py \
  tests/test_gemma4_quality_seed_v2.py \
  tests/test_gemma4_quality_seed_live_runner.py \
  tests/test_gemma4_quality_stability.py \
  tests/test_combine_gemma4_quality_results.py
```

Result:

- `38 passed`

## Findings

### F1. Candidate layer no longer has the short-time blind spot

`short-time-push-001` now has candidate-layer evidence via `short_time_candidate_pool`, while live output quality remains judged by output-level preferred patterns.

Category: `eval_harness_fix`

### F2. Latency gate is now actionable

The same partial live run that passed safety and preferred quality now correctly fails one scenario when `--max-elapsed-ms 120000` is applied.

Category: `harness_runtime_gate`

### F3. Full local Gemma4 live evaluation should remain batched

Local Gemma4 can pass quality checks, but long latency makes full-suite runs impractical as one command.

Category: `operational_constraint`

## Next Actions

1. Run `safety-core` batch with the generated command and score it.
2. If `safety-core` passes except latency, decide whether the latency threshold should be:
   - strict model gate
   - local-environment warning
   - tuned-model comparison metric
3. Run `time-equipment` batch next because it includes `short-time-push-001`.
4. Only after batch score gaps are clear, change prompt/rules/model.

## Developer-Copy Summary

Routine AI v4 harness improved.

- Candidate-layer evidence is now separated from live output quality via `candidatePreferredPatterns`.
- Added batch plan generator for 18 scenario Gemma4 live eval.
- Candidate quality improved from `16 support + 1 no-candidate + 1 hard-stop` to `17 support + 1 hard-stop`.
- Latency gate confirmed: one scenario fails at 196s when `--max-elapsed-ms 120000` is used.
- No FE/BE contract change is required from this harness-only update.

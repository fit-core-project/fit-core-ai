# Routine Recommendation v4 Baseline Loop Result

Date: 2026-07-03
Owner: Planner / AI routine core owner
Scope: Fit-Core routine recommendation AI baseline before v4 prompt/rule/model changes

## Summary

Routine recommendation v4 should proceed from the current v3 branch, but the first baseline shows that the next bottleneck is not hard-safety quality. The DB/candidate layer is mostly ready, and the first partial live Gemma4 run passed hard-safety and preferred-quality checks for the completed scenarios.

The immediate v4 priority is evaluation/runtime control:

1. Run live evaluations in smaller batches instead of all 18 scenarios at once.
2. Use `--max-elapsed-ms` in score-file so slow outputs are visible as quality risks.
3. Use `candidatePreferredPatterns` for candidate-layer-only evidence, keeping `preferredPatterns` for live output quality.
4. Continue live scoring only after the batch strategy is stable.

## Inputs

- Branch: `codex/workout-gym-db-integration-2026-06-30`
- Base: latest `origin/develop`
- Seed: `tests/fixtures/gemma4_routine_quality_eval_seed_v3.jsonl`
- Local model: `gemma4:latest`
- SQLite DB: `fit_core.sqlite`

## Commands

```bash
python3 scripts/run_gemma4_quality_seed.py validate \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v3.jsonl

python3 scripts/run_gemma4_quality_seed.py plan \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v3.jsonl \
  --model gemma4:latest \
  --output-dir tests/evaluation/.artifacts/gemma4-quality-seed-v4-baseline

python3 scripts/run_candidate_quality_seed.py \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v3.jsonl \
  --output-dir tests/evaluation/.artifacts/candidate-quality-v4-baseline \
  --run-id candidate-quality-v4-baseline-2026-07-03

python3 scripts/run_gemma4_quality_seed_live.py \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v3.jsonl \
  --output-dir tests/evaluation/.artifacts/gemma4-quality-seed-live-v4-baseline \
  --run-id gemma4-v4-baseline-live-2026-07-03 \
  --scenario-timeout-sec 240 \
  --cooldown-sec 1

python3 scripts/run_gemma4_quality_seed.py score-file \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v3.jsonl \
  --results tests/evaluation/.artifacts/gemma4-quality-seed-live-v4-baseline/gemma4-v4-baseline-live-2026-07-03/gemma4-quality-seed-results-filled.json \
  --output-dir tests/evaluation/.artifacts/gemma4-quality-seed-v4-baseline-score \
  --run-id gemma4-v4-baseline-partial-score-2026-07-03
```

## Evidence

### Seed / Readiness

- Scenario count: `18`
- Hard constraints: `42`
- Failure patterns: `66`
- Rubric items: `55`
- Ollama reachable: `true`
- `gemma4:latest` installed: `true`
- Can run live eval: `true`

Report:

- `tests/evaluation/.artifacts/gemma4-quality-seed-v4-baseline/gemma4-quality-seed-20260703T000712Z/gemma4-quality-seed-report.md`

### DB Population

SQLite extension tables are populated.

| Table | Rows |
|---|---:|
| `exercise_tier` | 24653 |
| `exercise_mobility_requirement` | 24653 |
| `exercise_joint_load_profile` | 24653 |
| `exercise_effect_profile` | 24653 |
| `exercise_difficulty_profile` | 24653 |
| `exercise_anthropometry_sensitivity_profile` | 24653 |
| `exercise_regression_progression_map` | 206 |
| `exercise_condition_policy` | 817 |

### Candidate Quality

Candidate-layer result:

- `candidate_supports_quality`: `16`
- `no_candidate_level_patterns`: `1`
- `hard_stop`: `1`
- `candidate_layer_ok`: `17`
- `deterministic_guard_ok`: `1`

Attention scenarios:

| Scenario | Status | Interpretation |
|---|---|---|
| `short-time-push-001` | `no_candidate_level_patterns` | Candidate layer does not expose preferred short-time quality patterns strongly enough. |
| `all-major-pain-extreme-001` | `hard_stop` | Expected deterministic safety guard. This is not a defect. |

Report:

- `tests/evaluation/.artifacts/candidate-quality-v4-baseline/candidate-quality-v4-baseline-2026-07-03/candidate-quality-seed-report.md`

### Partial Live Gemma4 Result

Full 18-scenario live evaluation was stopped because local Gemma4 runtime was too slow for a single uninterrupted run.

Completed scenarios: `3 / 18`

| Scenario | Result | Elapsed |
|---|---|---:|
| `lower-back-legs-001` | pass | 196462 ms |
| `limited-ankle-legs-001` | pass | 84733 ms |
| `shoulder-push-001` | pass | 76817 ms |

Score summary for completed scenarios:

- passed: `3`
- failed: `0`
- hard safety passed: `3`
- preferred quality passed: `3`
- fallback failed: `0`
- contract valid: `3`

The score-file recommendation is `needs_review` only because 15 seed scenarios are missing from the interrupted live run.

Reports:

- `tests/evaluation/.artifacts/gemma4-quality-seed-live-v4-baseline/gemma4-v4-baseline-live-2026-07-03/gemma4-quality-seed-results-filled.json`
- `tests/evaluation/.artifacts/gemma4-quality-seed-v4-baseline-score/gemma4-v4-baseline-partial-score-2026-07-03/gemma4-quality-seed-report.md`

## Findings

### F1. Candidate layer is mostly ready

16 of 18 scenarios have candidate-level evidence for preferred quality. This supports continuing with live scoring and prompt/rule refinement rather than redesigning the DB layer first.

Primary category: `candidate_layer_ok`

### F2. Extreme pain hard stop is behaving as intended

`all-major-pain-extreme-001` returns a hard stop candidate status. This matches the product policy: severe overlapping pain and low readiness should not force a routine.

Primary category: `deterministic_guard_ok`

### F3. Short-time push needs better positive-quality evidence

`short-time-push-001` has candidates, but no candidate-level preferred pattern hit. The likely v4 improvement is not a medical/safety fix. It is a short-time composition quality issue.

Likely action:

- Add or derive candidate tags for short-time push cases.
- Prefer compact, high-yield, low-transition exercises.
- Make the scorer expect `two_to_three_exercises`, `short_time_reason`, or similar candidate-visible evidence.

Primary category: `python_rule_gap` or `eval_scenario_gap`

### F4. Full live suite is too slow as a single run

The completed live scenarios passed, but latency was high:

- 196s
- 85s
- 77s

This makes a full 18-scenario live run operationally expensive. v4 should support smaller scenario batches and latency scoring before using full live runs as a routine gate.

Primary category: `harness_runtime_gap`

## Recommended Next Actions

1. Generate a batch plan with `scripts/plan_gemma4_quality_batches.py`.
2. Score each live batch with `--scenario-id` and `--max-elapsed-ms`.
3. Run live eval in bounded batches:
   - safety batch
   - mobility batch
   - equipment/time batch
   - profile/condition batch
   - advanced/effect batch
4. Improve `short-time-push-001` candidate-quality tags or rubric without polluting live output preferred-pattern scoring.
5. After batching, run the remaining 15 live scenarios using `--resume`.
6. Only after live score gaps are categorized, decide whether v4 needs:
   - DB material patch
   - Python rule patch
   - prompt payload patch
   - fine-tuning dataset example

## Developer-Copy Summary

Routine recommendation v4 baseline started.

- DB extension tables are populated.
- v3 seed expanded to 18 scenarios and validates.
- Candidate layer supports preferred quality in 16/18 scenarios.
- Extreme pain scenario hard-stops as intended.
- Partial live Gemma4 run passed 3/3 completed scenarios for hard safety and preferred quality.
- Main issue is live evaluation runtime: first 3 scenarios took about 196s, 85s, and 77s.

Next work should focus on batched live evaluation and latency-aware scoring before changing prompt/model behavior.

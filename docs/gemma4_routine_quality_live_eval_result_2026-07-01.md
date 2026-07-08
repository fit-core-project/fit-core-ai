# Gemma4 Routine Quality Live Eval Result

Date: 2026-07-01

## Objective

운동 DB 고도화가 루틴 추천에 결합된 상태에서 Gemma4 기반 루틴 생성이 안전 제약, 시간 제약, 장비 제약, 통증/부상 제약을 실제 라이브 파이프라인으로 통과하는지 검증한다.

## Inputs

- Seed: `tests/fixtures/gemma4_routine_quality_eval_seed_v2.jsonl`
- Live runner: `scripts/run_gemma4_quality_seed_live.py`
- Scorer: `scripts/run_gemma4_quality_seed.py score-file`
- Local model: `gemma4:latest` via Ollama
- Local DB: `fit_core.sqlite`

## Changes Made

1. Live evaluation runner hardening
   - Added per-scenario timeout.
   - Added flushed progress logs per scenario.
   - Kept output sanitized: no raw prompt, no raw model output.

2. Direct pain primary-muscle safety rule
   - If `currentPainAreas` directly matches a candidate `primary_muscle`, the candidate is excluded.
   - If `currentPainAreas` overlaps a candidate `secondary_muscle`, the candidate is penalized rather than hard-excluded.

3. Extreme pain context guard
   - If readiness is `low`, pain covers at least 5 body parts, and pain overlaps at least half of target muscles, the pipeline returns:
     - `generationStatus=failed`
     - `statusReasonCode=emptyCandidate`
     - `routineBlocks=[]`
   - This prevents forcing a normal full-body routine when the user's current condition makes the request unsafe or contradictory.

## Final Live Eval Evidence

- Live output:
  - `tests/evaluation/.artifacts/gemma4-quality-seed-live/gemma4-v2-live-full-after-safety-guard-2026-07-01/gemma4-quality-seed-results-filled.json`
- Score report:
  - `tests/evaluation/.artifacts/gemma4-quality-seed/gemma4-v2-live-full-after-safety-guard-score-2026-07-01/gemma4-quality-seed-report.json`
  - `tests/evaluation/.artifacts/gemma4-quality-seed/gemma4-v2-live-full-after-safety-guard-score-2026-07-01/gemma4-quality-seed-report.md`

## Final Score

- Total scenarios: 12
- Scored scenarios: 12
- Passed: 12
- Failed: 0
- Recommendation: `pass`

## Regression Evidence

- Targeted tests:
  - `python3 -m pytest -q -s --capture=no tests/test_pipeline.py tests/test_prompt_regression.py tests/test_guard.py tests/test_feedback_ranker_integration.py tests/test_gemma4_quality_seed_live_runner.py`
  - Result: 78 passed
- Full tests:
  - `python3 -m pytest -q -s --capture=no`
  - Result: 690 passed, 1 skipped, 1 warning
- Compile check:
  - `python3 -m compileall -q engines scripts/run_gemma4_quality_seed_live.py`
  - Result: passed

## Remaining Improvement Candidates

1. Speed
   - Most live Gemma4 scenarios take about 60-110 seconds.
   - The new timeout prevents indefinite stalls, but prompt size / candidate payload / model options still need tuning.

2. Preferred-pattern quality
   - All hard constraints pass, but many scenarios still miss preferred patterns.
   - Next quality loop should improve positive recommendation quality, not just safety.

3. Fine-tuning readiness
   - The current evidence supports using this v2 seed as a baseline eval harness.
   - Fine-tuning should only be accepted if tuned Gemma improves preferred-pattern hits without increasing hard violations, fallback rate, timeout rate, or privacy risk.

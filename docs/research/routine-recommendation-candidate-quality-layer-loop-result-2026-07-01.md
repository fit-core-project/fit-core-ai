# Routine Recommendation Candidate Quality Layer Loop Result

Date: 2026-07-01
Status: pass; live Gemma4 v2 seed re-run complete
Owner: Planner / Codex routine recommendation owner

## Summary

이번 루프는 Gemma4 결과가 마음에 들지 않을 때 곧바로 프롬프트나 파인튜닝을 탓하지 않기 위해, 추천 품질 문제를 두 층으로 분리했다.

1. DB / candidate query / ranking이 좋은 후보를 실제로 LLM에게 올려주는가
2. LLM이 그 후보를 골라서 좋은 루틴과 설명으로 조립하는가

결론:

- 후보 품질 진단 스크립트를 추가했다.
- 후보군이 너무 비슷한 운동 변형으로 도배되는 문제를 줄였다.
- 통증/주의 맥락에서 일반 부상 주의도, 탄성/착지/고협응 후보, 스포츠 보강성 후보를 더 보수적으로 처리했다.
- 허리 제약 케이스에서는 요추/축방향 부하가 낮은 머신/고립 대체 후보가 상단에 보이도록 보강했다.
- 엄격 기준에서 `candidate_quality_gap`은 0건으로 닫혔다.
- live Gemma4 v2 seed 12개를 새 코드 기준으로 재실행했다.
- live score 결과는 `12/12 pass`, `quality_passed 12/12`, `recommendation=pass`다.
- 후보군 비교 결과는 `candidate_layer_ok 11`, `deterministic_guard_ok 1`이다.
- 현재 기준에서 남은 즉시 P0/P1 수정 대상은 없다. 다음 고도화는 seed coverage 확대와 더 엄격한 preferred hit 기준으로 진행한다.

## Live Re-Run Result

이번 재실행에서는 live runner가 중간 종료 시 결과를 잃지 않도록 먼저 보강했다.

- `scripts/run_gemma4_quality_seed_live.py`
  - 시나리오 1개가 끝날 때마다 `gemma4-quality-seed-results-filled.json`을 즉시 저장한다.
  - `--scenario-id`로 특정 scenario만 선택 실행할 수 있다.
  - `--resume`으로 이미 저장된 scenario를 건너뛰고 이어 실행할 수 있다.
  - 결과 payload에 `completedScenarioCount`, `completedScenarioIds`, `selectedScenarioIds`를 추가했다.
- `tests/test_gemma4_quality_seed_live_runner.py`
  - scenario id parsing
  - seed row filtering
  - partial snapshot write
  - resume skip
  를 회귀테스트로 고정했다.

Live execution result:

```text
runId: gemma4-live-full-20260701-resumable
scenarioCount: 12
completedScenarioCount: 12
failedScenarioCount: 0
```

Live score result:

```text
total_scenarios: 12
scored_scenarios: 12
passed: 12
failed: 0
quality_passed: 12
quality_failed: 0
recommendation: pass
```

Candidate + live routing result:

```text
candidate_supports_quality: 10
no_candidate_level_patterns: 1
hard_stop: 1

candidate_layer_ok: 11
deterministic_guard_ok: 1
```

## Changes Made

1. `scripts/run_candidate_quality_seed.py`
   - v2 quality seed를 LLM 호출 없이 DB 후보와 ranker에 통과시킨다.
   - live result JSON이 있으면 후보층 gap과 LLM selection/rationale gap을 분리한다.
   - JSON/Markdown report를 `tests/evaluation/.artifacts/candidate-quality-seed/` 아래에 저장한다.

2. `engines/candidate_ranker.py`
   - 같은 운동의 미세 변형이 후보 상단을 독점하지 않도록 family diversity를 추가했다.
   - step up/down, swing, push-up plus, split squat, leg press 계열은 장비만 바뀌어도 같은 family로 묶는다.
   - 통증/주의 맥락에서 high/medium `injury_caution_level`에 deterministic penalty를 준다.
   - hypertrophy/strength 루틴에서 스포츠 보강/유산소성 후보를 메인 운동보다 낮게 배치한다.
   - 통증/주의 맥락에서 swing, power, landing, sprint, skip 같은 탄성/고협응 후보를 보수적으로 낮춘다.
   - 허리 제약 케이스에서 low lumbar/low axial/low injury caution을 가진 머신/고립 대체 후보를 boost한다.
   - 단, 러닝머신/스포츠 보강/탄성 후보에는 이 safe alternative boost를 주지 않는다.
   - `experienceLevel=beginner`를 반영해 고기술/고실패 후보를 감점하고, 낮은 기술 난이도/낮은 실패 부담/안정 장비 후보를 boost한다.
   - 수술 이력/의료 허가 필요 조건에서 저부하·저주의도·안정 장비 후보를 `professional_clearance_safe_alternative`로 boost한다.

3. `engines/prompt_builder.py`
   - `[QUALITY SELECTION POLICY]`를 추가했다.
   - LLM에게 최고 점수 후보만 기계적으로 고르지 말고, active constraints와 score reasons가 잘 맞는 균형 잡힌 후보를 선택하게 했다.
   - 근거에는 penalty/boost tradeoff와 active context를 연결하도록 했다.
   - request context에 `experienceLevel`을 추가해 초보자/상급자 맥락이 LLM 설명에도 전달되게 했다.

4. `tests/test_candidate_quality_seed.py`
   - 후보 품질 tag 산출과 live-result 비교 report 생성을 테스트한다.
   - beginner technique risk / progression path와 professional clearance safe alternative tag를 회귀테스트로 고정했다.

5. `tests/test_prompt_regression.py`
   - deadlift/step 변형 도배 방지
   - 스포츠/유산소/보강 후보 demotion
   - 허리 제약에서 저부하 대체 후보 boost
   - 초보자 경험 수준에서 low technique risk / progression path boost
   - 수술/의료 허가 필요 조건에서 safe alternative boost
   - prompt quality selection policy 유지
   를 회귀테스트로 고정했다.

6. `engines/schemas.py`, `scripts/run_gemma4_quality_seed_live.py`, `engines/routine_pipeline.py`
   - `RoutineRequest.experience_level`을 보존한다.
   - v2 seed의 `experienceLevel`을 request로 매핑한다.
   - `surgeryHistory.status=needs_clearance`를 `professionalClearance=false`로 명시화해 랭커가 의료 허가 필요 맥락을 deterministic하게 읽게 했다.
   - routine pipeline에서 request/profile experience를 ranker와 prompt에 전달한다.
   - live runner가 시나리오별 partial snapshot을 저장하고 `--scenario-id`, `--resume` 실행을 지원한다.

7. `docs/gemma4_routine_quality_loop_spec.md`
   - candidate/ranking layer 진단 명령과 interpretation을 표준 루프 명령에 추가했다.

8. `engines/db_queries.py`
   - local catalog enrichment cache를 추가했다.
   - cache key는 `fit_core.sqlite` 경로와 파일 수정시각이다.
   - DB를 다시 빌드하면 다음 호출에서 자동으로 새 enrichment cache가 생성된다.

## Evidence

Targeted candidate/ranking tests:

```bash
python3 -m pytest -q -s --capture=no \
  tests/test_prompt_regression.py::test_scoring_promotes_lumbar_safe_low_load_alternatives_over_ballistic_context \
  tests/test_prompt_regression.py::test_scoring_demotes_sport_or_cardio_context_for_hypertrophy_main_lifts \
  tests/test_prompt_regression.py::test_scoring_diversifies_step_variants_across_equipment
```

Result:

```text
3 passed
```

Routine quality regression bundle:

```bash
python3 -m pytest -q -s --capture=no \
  tests/test_candidate_quality_seed.py \
  tests/test_prompt_regression.py \
  tests/test_pipeline.py \
  tests/test_guard.py \
  tests/test_gemma4_quality_seed_live_runner.py \
  tests/test_gemma4_quality_seed_v2.py \
  tests/test_contract.py
```

Result:

```text
100 passed
```

Compile check:

```bash
python3 -m compileall -q engines scripts/run_candidate_quality_seed.py scripts/run_gemma4_quality_seed.py scripts/run_gemma4_quality_seed_live.py
```

Result:

```text
passed
```

Live Gemma4 readiness:

```bash
python3 scripts/check_local_llm_readiness.py \
  --base-url http://127.0.0.1:11434 \
  --model gemma4:latest \
  --timeout-sec 180 \
  --strict
```

Result:

```text
readiness_status: pass
probe_success: true
probe_latency_ms: 62656
```

Live Gemma4 v2 seed run:

```bash
python3 scripts/run_gemma4_quality_seed_live.py \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v2.jsonl \
  --output-dir tests/evaluation/.artifacts/gemma4-quality-seed \
  --run-id gemma4-live-full-20260701-resumable \
  --scenario-timeout-sec 240
```

Result:

```text
scenarioCount: 12
completedScenarioCount: 12
failedScenarioCount: 0
```

Generated live result:

- `tests/evaluation/.artifacts/gemma4-quality-seed/gemma4-live-full-20260701-resumable/gemma4-quality-seed-results-filled.json`

Live score:

```bash
python3 scripts/run_gemma4_quality_seed.py score-file \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v2.jsonl \
  --results tests/evaluation/.artifacts/gemma4-quality-seed/gemma4-live-full-20260701-resumable/gemma4-quality-seed-results-filled.json \
  --output-dir tests/evaluation/.artifacts/gemma4-quality-seed/gemma4-live-full-20260701-resumable-score \
  --run-id gemma4-live-full-20260701-resumable-score-v2
```

Result:

```text
passed: 12
failed: 0
quality_passed: 12
quality_failed: 0
recommendation: pass
```

Generated score report:

- `tests/evaluation/.artifacts/gemma4-quality-seed/gemma4-live-full-20260701-resumable-score/gemma4-live-full-20260701-resumable-score-v2/gemma4-quality-seed-report.json`
- `tests/evaluation/.artifacts/gemma4-quality-seed/gemma4-live-full-20260701-resumable-score/gemma4-live-full-20260701-resumable-score-v2/gemma4-quality-seed-report.md`

Candidate + live comparison:

```bash
python3 scripts/run_candidate_quality_seed.py \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v2.jsonl \
  --results tests/evaluation/.artifacts/gemma4-quality-seed/gemma4-live-full-20260701-resumable/gemma4-quality-seed-results-filled.json \
  --output-dir tests/evaluation/.artifacts/candidate-quality-seed \
  --run-id candidate-quality-v2-with-live-full-20260701 \
  --min-preferred-hits 1
```

Result:

```text
candidate_supports_quality: 10
no_candidate_level_patterns: 1
hard_stop: 1

candidate_layer_ok: 11
deterministic_guard_ok: 1
```

Generated candidate comparison report:

- `tests/evaluation/.artifacts/candidate-quality-seed/candidate-quality-v2-with-live-full-20260701/candidate-quality-seed-report.json`
- `tests/evaluation/.artifacts/candidate-quality-seed/candidate-quality-v2-with-live-full-20260701/candidate-quality-seed-report.md`

Strict candidate quality report:

```bash
python3 scripts/run_candidate_quality_seed.py \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v2.jsonl \
  --min-preferred-hits 2 \
  --run-id candidate-quality-v2-strict-after-experience-clearance-2026-07-01
```

Result summary:

```text
candidate_supports_quality: 10
candidate_quality_gap: 0
no_candidate_level_patterns: 1
hard_stop: 1

candidate_layer_ok: 11
deterministic_guard_ok: 1
```

Runtime:

```text
elapsed=0:26.62
```

Generated report:

- `tests/evaluation/.artifacts/candidate-quality-seed/candidate-quality-v2-strict-after-experience-clearance-2026-07-01/candidate-quality-seed-report.json`
- `tests/evaluation/.artifacts/candidate-quality-seed/candidate-quality-v2-strict-after-experience-clearance-2026-07-01/candidate-quality-seed-report.md`

Historical candidate quality report with previous live Gemma4 result attached:

```bash
python3 scripts/run_candidate_quality_seed.py \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v2.jsonl \
  --results tests/evaluation/.artifacts/gemma4-quality-seed-live/gemma4-v2-live-full-after-safety-guard-2026-07-01/gemma4-quality-seed-results-filled.json \
  --min-preferred-hits 2 \
  --run-id candidate-quality-v2-strict-after-experience-clearance-with-live-2026-07-01
```

Historical result summary, before the current ranker/prompt/live-runner changes:

```text
candidate_supports_quality: 10
candidate_quality_gap: 0
no_candidate_level_patterns: 1
hard_stop: 1

llm_selection_or_rationale_gap: 11
deterministic_guard_ok: 1
```

Interpretation: the attached live result predates the latest ranker/prompt/live-runner changes. It is retained only as historical routing evidence and must not be used as the current model-quality result. The current result is the `gemma4-live-full-20260701-resumable` run above, which passed 12/12.

## Interpretation

The candidate layer now exposes enough preferred candidate-level signals under strict mode.

Notable improvement:

- `lower-back-legs-001` now exposes `leg_curl` and `machine` signals under strict mode.
- `high-risk-high-effect-substitution-001` now exposes `risk_adjusted_substitution`, `effect_vs_risk_reason`, and `lower_lumbar_alternative`.
- `beginner-lower-skill-001` now exposes machine/stable, technique-risk, and progression-path signals.
- `post-surgery-caution-001` now exposes safe-alternative and neutral/supported signals.
- The extreme all-major-pain case remains correctly blocked by the deterministic guard.
- Candidate report runtime is now acceptable for local iteration after enrichment caching.

## Next Actions

1. Keep the current candidate/ranker/prompt state as the new local baseline.
   - Current seed v2 pass criteria is satisfied.
   - Do not fine-tune just to fix this seed set; there is no repeated live LLM failure left in the current evidence.

2. Expand seed coverage before deeper tuning.
   - Add more realistic multi-condition profiles.
   - Add stricter substitution quality checks.
   - Add latency/fallback thresholds.
   - Add repeated-run stability checks for the same scenario.

3. Fine-tuning remains gated.
   - Fine-tuning is justified only if new expanded seed cases show repeated LLM-specific selection/rationale failures after DB/ranker/prompt are already correct.
   - Current evidence says deterministic DB/ranker/prompt control is doing the main work correctly.

## Approval / Safety Notes

- No commit was created.
- No push was performed.
- No DB migration was applied.
- No active public contract was changed.
- This iteration only changed internal ranking, prompt guidance, local catalog read caching, evaluation script, tests, and documentation.

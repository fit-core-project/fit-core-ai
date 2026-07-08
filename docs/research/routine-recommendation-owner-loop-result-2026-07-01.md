# Routine Recommendation Owner Loop Result

Date: 2026-07-01
Status: pass
Owner: Planner / Codex routine recommendation owner

## Summary

운동 루틴 추천 기능을 직접 소유하는 전제로 첫 품질 루프를 실행했다.

이번 반복의 목적은 새 기능을 크게 더하는 것이 아니라, 앞으로 고도화의 기준이 될 안전 경계를 테스트와 프롬프트에 고정하는 것이었다.

결론:

- `currentPainAreas`는 부상/통증 hard safety input이다.
- 직접 주동근이 `currentPainAreas`와 겹치는 후보는 LLM 전에 제외한다.
- 보조근만 겹치는 후보는 제외하지 않고 deterministic penalty를 준다.
- `currentDoms` 1/2단계는 제외가 아니라 우선순위/볼륨 조정이다.
- `currentDoms` 3단계만 hard exclusion boundary로 둔다.
- Gemma4는 최종 안전 판단자가 아니라, DB/Python 필터를 통과한 후보를 조립하고 설명하는 역할로 제한한다.

## Inputs

- Repo: `/mnt/d/project-fit-core/worktrees/fit-core-ai-workout-gym-db`
- Branch: `codex/workout-gym-db-integration-2026-06-30`
- Loop spec: `docs/gemma4_routine_quality_loop_spec.md`
- Prompt builder: `engines/prompt_builder.py`
- Candidate ranker: `engines/candidate_ranker.py`
- Regression tests: `tests/test_prompt_regression.py`
- Offline eval harness: `tests/evaluation/test_offline_eval_harness.py`
- Gemma4 seed tests:
  - `tests/test_gemma4_quality_seed_v2.py`
  - `tests/test_gemma4_quality_seed_live_runner.py`

## Changes Made In This Iteration

1. Prompt hard constraint wording corrected.
   - Old meaning: DOMS, pain, equipment were all described as hard constraints.
   - New meaning: pain/equipment are hard constraints; DOMS level 3 is exclusion; DOMS 1/2 are volume/priority constraints.

2. Prompt regression tests updated.
   - The prompt must now explicitly preserve the DOMS 1/2 vs 3 boundary.

3. DOMS boundary regression test added.
   - DOMS 1 keeps the candidate and adds `doms mild penalty`.
   - DOMS 2 keeps the candidate and adds `doms moderate penalty`.
   - DOMS 3 removes the candidate.

4. Loop spec updated.
   - Added active safety/recovery policy so future changes do not re-open the same ambiguity.

## Verification Evidence

Seed integrity:

```bash
python3 -m pytest -q -s --capture=no tests/test_gemma4_quality_seed_v2.py
```

Result:

```text
2 passed
```

Targeted routine tests:

```bash
python3 -m pytest -q -s --capture=no \
  tests/test_prompt_regression.py \
  tests/test_pipeline.py \
  tests/test_guard.py \
  tests/test_gemma4_quality_seed_live_runner.py
```

Result:

```text
62 passed
```

Offline scenario eval:

```bash
python3 -m pytest -q -s --capture=no tests/evaluation/test_offline_eval_harness.py
```

Result:

```text
17 passed
```

Contract / feedback regression tests:

```bash
python3 -m pytest -q -s --capture=no tests/test_contract.py tests/test_feedback_ranker_integration.py
```

Result:

```text
32 passed
```

Compile check:

```bash
python3 -m compileall -q engines scripts/run_gemma4_quality_seed_live.py
```

Result:

```text
passed
```

## Findings

1. The current codebase already has a useful deterministic safety architecture.
   - direct primary pain exclusion
   - secondary pain penalty
   - DOMS 1/2 penalty
   - DOMS 3 exclusion
   - guard/repair/fallback
   - offline and live evaluation harnesses

2. The main immediate mismatch was prompt wording, not the deterministic ranker.
   - The prompt still implied all DOMS are hard constraints.
   - That could teach the LLM and future maintainers the wrong behavior.

3. The next quality work should focus on positive recommendation quality.
   - Hard safety currently has good coverage.
   - Remaining value is better substitutions, score explanations, and scenario-specific preferred patterns.

## Next Actions

1. Add preferred-pattern scoring tests for the 12 Gemma4 quality scenarios.
   - Example: lower-back legs should prefer supported lower-body alternatives over high lumbar hinge work.

2. Add a compact score explanation contract.
   - Each routine block should be able to expose why it was selected:
     - target match
     - equipment availability
     - safety penalty avoided
     - mobility/body condition fit
     - effect / stimulus-to-fatigue

3. Run live Gemma/Ollama evaluation only after the deterministic tests pass.
   - Live eval is slower and should be used as a second layer, not the first debugging loop.

4. Fine-tuning remains gated.
   - Only use fine-tuning when repeated failures are LLM-specific and cannot be solved by DB, deterministic rule, validator, or prompt cleanup.

## Approval / Safety Notes

- No commit was created.
- No push was performed.
- No DB migration was applied.
- No active public contract was changed.
- This iteration only tightened internal prompt/test/spec alignment around the already-agreed pain/DOMS boundary.

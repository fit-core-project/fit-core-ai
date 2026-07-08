# Gemma4 Routine Recommendation Quality Loop

Status: active draft  
Created: 2026-07-01  
Owner: Planner / AI routine quality reviewer  
Scope: routine recommendation quality evaluation before prompt, rule, DB, or fine-tune changes

## 1. Objective

Evaluate whether the current Gemma4 routine recommendation stack produces safe, contract-valid, explainable workout drafts under fixed scenario pressure, then route each failure to the right improvement layer.

This loop is not a fine-tuning job by itself. It is the evidence loop that decides whether a failure should be solved by DB data, deterministic Python rules, prompt changes, validator/repair logic, or a later fine-tune dataset.

## 2. Non-Goals

- Do not fine-tune before the failure class is proven by repeated scenario evidence.
- Do not use the LLM to enforce hard safety rules that Python/DB can enforce deterministically.
- Do not change active public contracts in this loop.
- Do not store raw prompts, raw model outputs, API keys, user notes, or credential-bearing URLs in reports.

## 3. Active Baseline

- Model: `gemma4:latest`
- Provider: local Ollama
- Ollama URL: `http://127.0.0.1:11434`
- Quality seed v1: `tests/fixtures/gemma4_routine_quality_eval_seed.jsonl`
- Quality seed v2: `tests/fixtures/gemma4_routine_quality_eval_seed_v2.jsonl`
- Runner: `scripts/run_gemma4_quality_seed.py`
- Readiness check: `scripts/check_local_llm_readiness.py`

## 4. Improvement Routing Rule

Use this rule before deciding to fine-tune.

| Failure Type | First Fix Layer | Fine-Tune Candidate? |
|---|---|---|
| Unsafe exercise enters candidates | DB / candidate ranking / hard filter | No |
| Injury, pain, DOMS, mobility, equipment constraints ignored after candidate filtering | Python validator / repair / hard constraint tests | No |
| Candidate ranking prefers bad substitutes despite correct DB fields | scoring weights / substitution map | Usually no |
| Prompt omits available context or gives unclear role/constraints | prompt builder | Maybe later |
| Output JSON contract breaks repeatedly | parser / deterministic repair first | Yes, if repeated after repair |
| Rationale is generic or does not cite visible inputs | prompt/rubric first | Yes, if repeated |
| Coaching tone is awkward but structurally correct | prompt first | Yes, if repeated |
| Model repeatedly misuses visible candidate fields | prompt + examples first | Yes, if repeated |

## 4-1. Active Safety / Recovery Policy

This loop treats the following as the current active routine recommendation policy.

- `currentPainAreas` means current injury / pain / medical caution input. A candidate whose direct `primary_muscle` matches a current pain area must be excluded before the LLM sees the final candidate list.
- If a current pain area overlaps only `secondary_muscle`, the candidate may remain but must receive a deterministic penalty and a visible score reason.
- `currentDoms` level 1 means mild soreness. It should reduce priority/volume, not remove all matching candidates.
- `currentDoms` level 2 means stronger soreness. It should strongly reduce priority/volume, not remove all matching candidates by default.
- `currentDoms` level 3 is the exclusion boundary and must behave like a hard candidate exclusion for that muscle.
- Gemma4 must not be the primary safety gate. DB/Python filters, score penalties, validators, and fallback/failed guards own hard safety behavior.

## 5. Loop Steps

1. Observe
   - Confirm Ollama and `gemma4:latest` readiness.
   - Validate seed file shape.
   - Confirm DB/backend readiness if running live service evaluation.

2. Run Baseline
   - Run readiness and seed plan.
   - If live service is available, run the same scenarios through the AI server and fill the sanitized result template.

3. Score
   - Score selected exercise IDs, rationale summary, score evidence summary, warning summary, contract validity, and hard violation count.
   - Separate hard safety pass from preferred-pattern quality pass.
   - Do not store raw model transcript.

4. Classify Failures
   - Mark each failure as DB, deterministic rule, prompt, validator/repair, or fine-tune candidate.

5. Improve
   - Apply only the smallest needed layer.
   - Prefer DB/Python determinism for safety and eligibility.
   - Use prompt/fine-tune for explanation quality, JSON discipline, and coaching style.

6. Re-run
   - Re-run the same seed.
   - Compare before/after pass rate and scenario-level deltas.

7. Stop
   - Stop when pass criteria are met, the same failure repeats twice, or a human decision is needed.

## 6. Success Criteria

- Readiness:
  - `ollama_reachable=true`
  - `model_installed=true`
  - `probe_success=true`
  - `readiness_status=pass`

- Seed integrity:
  - v2 seed validates.
  - No duplicate `scenarioId`.
  - All rows include required fields.

- Quality:
  - No hard constraint violations in must-pass scenarios.
  - No forbidden/failure pattern hits.
  - Preferred-pattern hits are reported separately from hard safety.
  - A result can be `pass_with_quality_gaps` when hard safety passes but preferred-pattern quality is weak.
  - Valid public contract JSON.
  - Rationale mentions relevant visible context when pain, DOMS, equipment, mobility, or readiness is present.

- Evidence:
  - JSON report and markdown report are saved under `tests/evaluation/.artifacts/`.
  - Report contains summarized outputs only.

## 7. Stop Criteria

Stop and ask for human review if:

- a public contract change is needed
- a DB migration is needed
- live service cannot reach DB/backend
- the same scenario fails twice after the intended layer is changed
- safety policy or medical wording is ambiguous
- fine-tune data would require raw private user data

## 8. Standard Commands

Readiness:

```bash
python3 scripts/check_local_llm_readiness.py \
  --base-url http://127.0.0.1:11434 \
  --model gemma4:latest \
  --timeout-sec 180 \
  --strict
```

Validate v2 seed:

```bash
python3 scripts/run_gemma4_quality_seed.py validate \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v2.jsonl
```

Create a bounded v4 batch plan:

```bash
python3 scripts/plan_gemma4_quality_batches.py \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v3.jsonl \
  --run-prefix gemma4-v4-batched-YYYY-MM-DD \
  --max-elapsed-ms 120000 \
  --scenario-timeout-sec 240 \
  --resume
```

Use the generated `gemma4-quality-batch-plan.md` as the execution checklist. Run one batch at a time, then score that same batch with `--scenario-id` and `--max-elapsed-ms`. Do not use a full 18-scenario live run as the default local Gemma4 gate; it is too slow and makes failures harder to isolate.

Create a run plan:

```bash
python3 scripts/run_gemma4_quality_seed.py plan \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v2.jsonl \
  --model gemma4:latest \
  --ollama-base-url http://127.0.0.1:11434 \
  --timeout-sec 180 \
  --probe
```

Create a sanitized result template:

```bash
python3 scripts/run_gemma4_quality_seed.py template \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v2.jsonl
```

Score a filled result file:

```bash
python3 scripts/run_gemma4_quality_seed.py score-file \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v2.jsonl \
  --results tests/evaluation/.artifacts/gemma4-quality-seed/<run-id>/gemma4-quality-seed-results-filled.json \
  --min-preferred-hits 1
```

Diagnose candidate/ranking layer before blaming the LLM:

```bash
python3 scripts/run_candidate_quality_seed.py \
  --seed tests/fixtures/gemma4_routine_quality_eval_seed_v2.jsonl \
  --results tests/evaluation/.artifacts/gemma4-quality-seed/<run-id>/gemma4-quality-seed-results-filled.json \
  --min-preferred-hits 2 \
  --run-id candidate-quality-v2-YYYY-MM-DD
```

Interpretation:

- `candidate_supports_quality`: DB/ranking exposed enough preferred candidate-level signals.
- `candidate_quality_gap`: improve DB tags, candidate query, scoring weights, substitution map, or diversity before prompt/fine-tune.
- `no_candidate_level_patterns`: scenario is mainly output/rationale/format behavior.
- `hard_stop`: deterministic guard blocked the scenario before candidate ranking.
- `routingHint=db_or_ranking_gap`: do not fine-tune yet.
- `routingHint=llm_selection_or_rationale_gap`: candidate layer is adequate; re-run live LLM after prompt/ranking changes.

Interpretation:

- `passed` / `failed`: hard safety, contract, and failure-pattern gate.
- `quality_passed` / `quality_failed`: preferred-pattern visibility gate.
- `recommendation=pass`: hard safety and preferred quality both pass.
- `recommendation=pass_with_quality_gaps`: hard safety passes, but positive recommendation quality needs improvement.
- `recommendation=needs_review`: hard safety, contract, failure-pattern, or missing-result issue exists.

## 9. Fine-Tune Gate

Only create fine-tune data when all are true:

- the failure is repeated across at least two related scenarios
- DB and deterministic rules cannot solve the issue cleanly
- prompt change alone is insufficient or makes other scenarios worse
- the desired answer can be represented as safe, contract-valid target output
- the dataset contains no raw private user note or credential-bearing content

Fine-tune candidates should be stored as synthetic or sanitized pairs:

```json
{
  "scenarioId": "lower-back-legs-001",
  "inputSummary": "legs, 60min, lower-back pain, normal readiness",
  "expectedBehavior": "avoid high lumbar-load barbell hinge, prefer supported lower-body alternatives, explain lumbar-load reason",
  "targetOutputShape": "RoutineDraftResponse-compatible JSON"
}
```

## 10. First Scenario Families

- Lumbar pain lower-body recommendation
- Limited ankle dorsiflexion legs recommendation
- Shoulder pain push recommendation
- Low readiness volume control
- Home equipment restriction
- Short time budget
- Beginner high-skill lift avoidance
- Post-surgery / medical clearance wording guard
- All-major-pain extreme fallback/failed behavior
- High-effect but high-risk substitution decision
- Body proportion setup sensitivity
- Long-term profile injury plus acute DOMS conflict

## 11. Developer Shareable Summary

The routine AI quality process is:

```text
Baseline -> fixed scenario pressure test -> failure classification -> smallest-layer fix -> re-test -> fine-tune only for repeated LLM-specific failures.
```

Safety, eligibility, and substitution control should stay in DB/Python rules whenever possible. Gemma4 should mainly handle structured composition, contextual rationale, and coaching tone after candidates are already safe.

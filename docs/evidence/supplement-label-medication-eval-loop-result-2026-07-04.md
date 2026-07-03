# Supplement Label / Medication Eval Loop Result

Date: 2026-07-04

Target: Fit-Core AI supplement chat

Branch: `codex/workout-gym-db-integration-2026-06-30`

## Objective

Harden supplement AI behavior for product-label, numeric dose, and medication-class questions without weakening the existing 57-case supplement safety regression.

## Inputs

- `data/documents/supplement_canonical_entities.json`
- `engines/supplement/query_understanding.py`
- `engines/supplement/response_composer.py`
- `scripts/eval_supplement_live.py`
- `tests/fixtures/supplement_live_eval_cases.json`
- `tests/fixtures/supplement_label_medication_eval_cases.json`

## Changes

1. Medication-class aliases expanded.
   - Blood pressure medication now catches user-facing terms such as `ACE 억제제`, `ARB`, `베타차단제`, `칼슘채널차단제`, and `이뇨제`.
   - Diabetes medication now catches `메트포르민`, `인슐린`, `SGLT2`, `GLP-1`, and `설폰요소제`.

2. Numeric dose / product-label context added.
   - Questions containing `mg`, `mcg`, `IU`, `g`, `알`, `정`, `캡슐`, `스푼`, or product-label wording are treated as safety-sensitive dose context.
   - The system does not recommend a dose. It asks for/uses product label, serving amount, daily total, overlap, medications, conditions, and consult context.

3. Short ASCII alias false-positive fixed.
   - `Mg` no longer matches the `mg` unit inside `50mg` or `2000mg`.
   - Short element-symbol aliases now require token boundaries.

4. Unsafe permissive wording expanded.
   - `복용해도 됩니다` and `계속 먹어도 됩니다` are treated like `먹어도 됩니다` and replaced/softened in risk contexts.

5. New live eval fixture added.
   - `tests/fixtures/supplement_label_medication_eval_cases.json`
   - 21 cases covering label dose, blood pressure medication classes, diabetes medication classes, aspirin/warfarin, surgery, pregnancy, mineral-antibiotic spacing, unknown product labels, immunocompromised probiotic, kidney/protein/creatine, and sleep stack contexts.

## Validation

### JSON validation

```bash
python3 -m json.tool tests/fixtures/supplement_label_medication_eval_cases.json >/tmp/supplement_label_medication_eval_cases.validated.json
python3 -m json.tool data/documents/supplement_canonical_entities.json >/tmp/supplement_canonical_entities.validated.json
```

Result: pass

### Unit / regression tests

```bash
python3 -m pytest -q -s --capture=no \
  tests/test_supplement_query_understanding.py \
  tests/test_supplement_response_composer.py \
  tests/test_supplement_kb_schema.py
```

Result: `75 passed`

```bash
python3 -m pytest -q -s --capture=no \
  tests/test_supplement_query_understanding.py \
  tests/test_supplement_response_composer.py \
  tests/test_supplement_llm_first_policy.py \
  tests/test_supplement_engine_degraded.py \
  tests/test_supplement_api_smoke.py
```

Result: `88 passed, 1 skipped`

Note: pytest emitted a shutdown hang in Python threading after the pass summary. This is the same local test-environment behavior seen in prior supplement evaluations; the pass summary was captured before interrupting the stuck shutdown.

### Live evaluation environment

Server:

```bash
SUPPLEMENT_LLM_TIMEOUT_SEC=1 \
SUPPLEMENT_ENABLE_WEB_FALLBACK=false \
SUPPLEMENT_ENABLE_LLM_QUERY_OPTIMIZER=false \
python3 -m uvicorn main:app --host 127.0.0.1 --port 8000
```

The server reported:

- `Supplement RAG full mode ready`
- `curated JSON fallback KB: chroma_db_missing`
- per-case full responses with sources

### Existing 57-case supplement regression

```bash
python3 scripts/eval_supplement_live.py \
  --timeout-sec 8 \
  --run-id supplement-live-57case-after-label-med-v1-20260704
```

Result: `57/57 pass`

Artifacts:

- `tests/evaluation/.artifacts/supplement-live/supplement-live-57case-after-label-med-v1-20260704/supplement_live_eval_results.json`
- `tests/evaluation/.artifacts/supplement-live/supplement-live-57case-after-label-med-v1-20260704/supplement_live_eval_summary.md`

### New label / medication-class evaluation

```bash
python3 scripts/eval_supplement_live.py \
  --timeout-sec 8 \
  --cases tests/fixtures/supplement_label_medication_eval_cases.json \
  --run-id supplement-live-label-med-v1-20260704
```

Result: `21/21 pass`

Artifacts:

- `tests/evaluation/.artifacts/supplement-live/supplement-live-label-med-v1-20260704/supplement_live_eval_results.json`
- `tests/evaluation/.artifacts/supplement-live/supplement-live-label-med-v1-20260704/supplement_live_eval_summary.md`

## Notes

- Default local Gemma timeout (`45s`) makes full 57-case live evaluation slow. For bounded regression loops, use the documented evaluation mode above.
- The current eval intentionally validates deterministic KB-backed safety behavior, not long-form conversational polish.
- Chroma DB was not rebuilt in this loop. The server still returned full-mode responses through curated JSON fallback KB. Rebuild Chroma only when testing semantic retrieval quality specifically.

## Next Actions

1. Add a pharmacist-review checklist scenario pack if real pharmacist feedback is available.
2. Decide whether numeric dose should stay mapped to `high dose` internally or become a separate `dose specified` risk context in a future contract version.
3. If production uses Chroma retrieval, run one dedicated Chroma rebuild and semantic retrieval drift check after the supplement corpus stabilizes.

# Supplement AI 57-Case Eval Loop Result

Date: 2026-07-03

## Objective

영양제 AI가 기존 curated JSON KB fallback 환경에서도 주요 사용자 질문에 대해 안전하고 직접적인 한국어 답변을 제공하는지 검증한다.

이번 루프의 목표는 의학적 정답 보장이 아니라, 현재 정의한 57개 제품 안전/계약 평가셋에서 아래 조건을 만족하는 것이다.

- 알려진 성분 단건 질문에 직접 답변한다.
- 다중 영양제 조합에서 핵심 성분을 빠뜨리지 않는다.
- 약물 병용, 질환, 임신/수술/면역저하 등 고위험 맥락에서 상담/확인 안내를 누락하지 않는다.
- unsupported 또는 partial 성분은 안전하다고 단정하지 않고 제품명, 성분표, 함량, 복용량을 요청한다.
- LLM timeout 환경에서도 결정론 fallback이 엉뚱한 성분으로 답변하지 않는다.

## Inputs

- Fixture: `tests/fixtures/supplement_live_eval_cases.json`
- Eval script: `scripts/eval_supplement_live.py`
- Server mode:
  - `APP_ENV=local`
  - `SUPPLEMENT_LLM_TIMEOUT_SEC=1`
  - `SUPPLEMENT_ENABLE_WEB_FALLBACK=false`
  - `SUPPLEMENT_ENABLE_LLM_QUERY_OPTIMIZER=false`
  - `WHISPER_PRELOAD=false`
- Retrieval mode observed: `curated_json`

## Changes

### KB Coverage

Expanded `data/documents/supplement_ingredient_profiles.json` with profiles for:

- calcium
- vitamin C
- vitamin E
- vitamin A
- vitamin B
- theanine
- melatonin
- EAA
- glutamine
- berberine
- quercetin
- NAC

Expanded `data/documents/supplement_interaction_rules.json` with targeted rules for:

- vitamin C + iron
- omega3 + vitamin E
- omega3 + aspirin
- magnesium + blood pressure medication
- vitamin D + diabetes medication
- berberine + diabetes medication
- melatonin + blood pressure medication

Expanded `data/documents/supplement_safety_rules.json` with targeted rules for:

- liver disease + vitamin A
- high-dose vitamin E / bleeding context
- digestive symptom + zinc
- berberine digestive/medication caution
- sleep supplement medication caution

### Query and Response Logic

- Added goal-comparison Korean expressions such as `뭐가 나아`, `뭘 봐야`.
- Added deterministic direct answers for unsupported/partial/high-risk cases.
- Preserved good LLM answers when they already mention the user’s core entities.
- Replaced low-specificity or entity-missing answers with deterministic answers.
- Added Korean/entity aliases to live eval expectations.
- Fixed `condition_risk` category to receive high-risk caution/action validation.

## Results

### Before

Run: `supplement-live-57case-kb-v2-20260703`

- Total: 57
- Pass: 40
- Fail: 17

Failure clusters:

- multi-supplement entity omissions
- drug/condition entity omissions
- Korean condition labels not recognized by evaluator
- unsupported/partial ingredient drift
- natural comparison questions falling back to irrelevant KB

### Intermediate

Run: `supplement-live-57case-kb-v3-20260703`

- Total: 57
- Pass: 55
- Fail: 2

Remaining issue:

- goal comparison phrases were not parsed as goal-based recommendation intent by the running server.

### Final

Run: `supplement-live-57case-kb-v3-2-20260703`

- Total: 57
- Pass: 57
- Fail: 0
- Error: 0

Result files:

- `tests/evaluation/.artifacts/supplement-live/supplement-live-57case-kb-v3-2-20260703/supplement_live_eval_results.json`
- `tests/evaluation/.artifacts/supplement-live/supplement-live-57case-kb-v3-2-20260703/supplement_live_eval_summary.md`

## Verification Commands

```bash
python3 -m json.tool data/documents/supplement_canonical_entities.json >/dev/null
python3 -m json.tool data/documents/supplement_ingredient_profiles.json >/dev/null
python3 -m json.tool data/documents/supplement_interaction_rules.json >/dev/null
python3 -m json.tool data/documents/supplement_safety_rules.json >/dev/null
```

Result: `json-ok`

```bash
python3 -m pytest -q -s --capture=no \
  tests/test_supplement_kb_schema.py \
  tests/test_supplement_query_understanding.py \
  tests/test_supplement_response_composer.py \
  tests/test_supplement_llm_first_policy.py \
  tests/test_supplement_engine_degraded.py \
  tests/test_supplement_api_smoke.py
```

Result: `97 passed, 1 skipped`

```bash
python3 scripts/eval_supplement_live.py \
  --timeout-sec 15 \
  --run-id supplement-live-57case-kb-v3-2-20260703
```

Result: `57 passed, 0 failed, 0 errors`

## Limitations

- This is not a medical validation suite.
- Source references are evidence locators and conservative policy anchors; ingredient-specific clinical strength still varies by source quality.
- `curated_json` fallback was used because Chroma DB was not available in this local run.
- LLM timeout was intentionally short to stress deterministic fallback quality.
- Broader real-user phrasing, product-label OCR input, and dose-specific reasoning still need separate evaluation loops.

## Next Recommended Loop

1. Add dose/product-label scenarios.
2. Add medication class specificity scenarios.
3. Add pharmacist-review checklist for high-risk cases.
4. Rebuild full vector DB and rerun the same 57-case suite in `chroma` retrieval mode.

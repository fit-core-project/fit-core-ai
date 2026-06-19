# Test Profiles

Fit-Core AI uses explicit pytest markers and small wrapper scripts so normal development does not require a full test run.

## Fast

Purpose: normal development loop for pure unit and regression tests.

Expected time: 3-10 minutes in slower local environments.

Command:

```powershell
scripts/test_fast.ps1
```

Equivalent pytest:

```powershell
pytest -m "not slow and not integration and not llm and not chroma and not smoke"
```

## Supplement Regression

Purpose: Supplement KB, query understanding, response composer, and degraded-engine routing changes.

Expected time: 5-15 minutes in slower local environments.

Command:

```powershell
scripts/test_supplement.ps1
```

## Full

Purpose: release/deploy readiness, shared engine changes, model/index behavior changes, or broad refactors.

Expected time: long-running in production-like local setups.

Command:

```powershell
scripts/test_full.ps1
```

Equivalent pytest:

```powershell
pytest
```

## Slow And Integration

Purpose: subprocess, local-service, model, Chroma, or broader integration checks outside the normal fast loop.

Command:

```powershell
pytest -m "slow or integration or llm or chroma"
```

## Smoke

Purpose: manual/local check against a running AI server.

Command:

```powershell
scripts/smoke_supplement.ps1
```

Smoke tests depend on a running AI server and are not part of the fast pytest loop.

Pytest smoke profile is opt-in:

```powershell
$env:RUN_SUPPLEMENT_SMOKE = "true"
pytest -m smoke
```

## Operating Rules

- Small logic change: run `scripts/test_fast.ps1`.
- Supplement change: run `scripts/test_supplement.ps1`.
- Corpus change: run KB schema and supplement regression, then reindex once.
- ResponseComposer change: run response composer tests and supplement regression.
- Common engine, model, index, or routing change: run full pytest.
- Before release/deploy: run full pytest and smoke.

## Marker Policy

- `slow`: long-running tests not required for normal development.
- `integration`: subprocess, multi-component, or local-service tests.
- `llm`: tests that require model loading or LLM inference.
- `chroma`: tests that access ChromaDB or embedding indexes.
- `smoke`: manual tests against a running service.
- `eval`: offline evaluation harness tests.

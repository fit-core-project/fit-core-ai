# Gemma4 Evaluation Suite Runbook

This runbook describes the pre-cutover Gemma4-only evaluation suite. The suite does not start or stop backend servers, does not start Ollama, does not pull models, and does not change production defaults.

## Commands

Preflight:

```text
python scripts/run_gemma4_evaluation_suite.py preflight --base-url http://127.0.0.1:8000 --ollama-base-url http://127.0.0.1:11434 --model gemma4:latest --output-dir tests/evaluation/.artifacts/gemma4-suite --strict
```

Provider policy check:

```text
python scripts/run_gemma4_evaluation_suite.py provider-check --production-like --strict
```

Group A run:

```text
python scripts/run_gemma4_evaluation_suite.py run-a --base-url http://127.0.0.1:8000 --ollama-base-url http://127.0.0.1:11434 --model gemma4:latest --output-dir tests/evaluation/.artifacts/gemma4-suite --repeats 5 --request-timeout 240
```

Group B run:

```text
python scripts/run_gemma4_evaluation_suite.py run-b --base-url http://127.0.0.1:8000 --ollama-base-url http://127.0.0.1:11434 --model gemma4:latest --output-dir tests/evaluation/.artifacts/gemma4-suite --repeats 5 --request-timeout 240
```

Compare:

```text
python scripts/run_gemma4_evaluation_suite.py compare --output-dir tests/evaluation/.artifacts/gemma4-suite --repeats 5 --request-timeout 240 --min-requests-per-group 50
```

## Two-Phase Server Workflow

The suite intentionally uses separate `run-a` and `run-b` commands.

Group A server env:

```text
LLM_PROVIDER=local
LOCAL_LLM_MODEL=gemma4:latest
OLLAMA_BASE_URL=http://127.0.0.1:11434
ENABLE_FEEDBACK_AWARE_RANKING=false
ROUTINE_CANDIDATE_POOL_SIZE=12
ENABLE_RULE_CRITIC_TELEMETRY=true
ENABLE_DYNAMIC_TEMPERATURE=false
```

Group B server env:

```text
LLM_PROVIDER=local
LOCAL_LLM_MODEL=gemma4:latest
OLLAMA_BASE_URL=http://127.0.0.1:11434
ENABLE_FEEDBACK_AWARE_RANKING=true
ROUTINE_CANDIDATE_POOL_SIZE=12
ENABLE_RULE_CRITIC_TELEMETRY=true
ENABLE_DYNAMIC_TEMPERATURE=false
```

Restarting the backend between A and B is operator-owned. The suite runner only sends requests to the already running service.

## Production-Like Opt-In

For a production-like provider check:

```text
APP_ENV=production
LLM_PROVIDER=local
ALLOW_LOCAL_LLM_IN_PRODUCTION=true
LOCAL_LLM_MODEL=gemma4:latest
```

Then run:

```text
python scripts/run_gemma4_evaluation_suite.py provider-check --production-like --strict
```

Without `ALLOW_LOCAL_LLM_IN_PRODUCTION=true`, production local provider should remain blocked.

## Recommendation Interpretation

- `promote_gemma4_only_candidate`: suite gates passed.
- `keep_gemma4_only_off`: safety or quality blocker found.
- `inconclusive_gemma4_readiness_failed`: readiness or provider mismatch.
- `inconclusive_gemma4_timeout`: timeout or timeout-like failed requests.
- `inconclusive_no_feedback_adjustment`: feedback-aware ranking did not affect enough candidates.
- `inconclusive_insufficient_sample`: request count below configured sample floor.
- `inconclusive`: non-blocking uncertainty remains.

The suite-level recommendation is authoritative for this evaluation flow; lower-level staging recommendations are included as supporting context.

## Privacy

Reports do not include raw prompts, raw LLM output, raw model responses, user notes, raw feedback rows, API keys, full environment dumps, or credential-bearing URLs.

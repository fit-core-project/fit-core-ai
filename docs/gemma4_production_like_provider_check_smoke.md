# P3-7a Gemma4 Production-like Provider-check Smoke

This document records the production-like provider policy smoke for the Gemma4 local provider. It is not a production cutover. It does not change provider defaults, start or stop servers, start Ollama, pull models, or modify the database.

## 1. Summary

Provider policy smoke passed for the production-like opt-in guard:

- Positive provider-check: passed with `APP_ENV=production`, `LLM_PROVIDER=local`, and `ALLOW_LOCAL_LLM_IN_PRODUCTION=true`.
- Negative provider-check: blocked local provider when `ALLOW_LOCAL_LLM_IN_PRODUCTION` was unset.
- Ollama/Gemma4 readiness: passed.
- Existing AI server `/docs`: reachable.
- Small generation smoke: not executed in this run because the already running server showed recent telemetry with `feedback_enabled=true`, while the P3-7a baseline smoke requires `ENABLE_FEEDBACK_AWARE_RANKING=false`. Server restart is operator-owned and was intentionally not performed.

Overall P3-7a status: `partial_pass_operator_generation_smoke_pending`.

## 2. Purpose

The goal is to verify that the Gemma4 local provider opt-in guard behaves as intended in a production-like environment:

- Production local provider is allowed only with explicit opt-in.
- Production local provider is blocked without opt-in.
- The existing Gemini fallback/override behavior remains available.
- Ollama and `gemma4:latest` are ready before any generation smoke.
- A small generation smoke can be run with production-like env after an operator starts the server with the exact env.

## 3. Preconditions

Required env for the verified Gemma4 local path:

```text
LLM_PROVIDER=local
LOCAL_LLM_MODEL=gemma4:latest
OLLAMA_BASE_URL=http://127.0.0.1:11434
ENABLE_LOCAL_RAW_JSON_INVOKE=true
LOCAL_LLM_NUM_PREDICT=4096
LOCAL_LLM_NUM_CTX=8192
ROUTINE_CANDIDATE_POOL_SIZE=12
ENABLE_RULE_CRITIC_TELEMETRY=true
ENABLE_DYNAMIC_TEMPERATURE=false
```

Production-like opt-in env:

```text
APP_ENV=production
LLM_PROVIDER=local
ALLOW_LOCAL_LLM_IN_PRODUCTION=true
```

Do not store raw prompts, raw LLM output, raw model responses, `user_note`, API keys, credential-bearing URLs, or full environment dumps in artifacts.

## 4. Positive Provider-check

Command:

```powershell
$env:APP_ENV="production"
$env:LLM_PROVIDER="local"
$env:ALLOW_LOCAL_LLM_IN_PRODUCTION="true"
$env:LOCAL_LLM_MODEL="gemma4:latest"
python scripts\run_gemma4_evaluation_suite.py provider-check --production-like --output-dir tests\evaluation\.artifacts\gemma4-production-like-smoke-positive --strict
```

Observed CLI result:

```json
{
  "app_env": "production",
  "requested_provider": "local",
  "effective_provider": "local",
  "local_allowed_in_production": true,
  "provider_check_status": "pass",
  "blocked_reason": null
}
```

Observed resolver fields:

```json
{
  "requested_provider": "local",
  "effective_provider": "local",
  "local_allowed_in_production": true,
  "model_name": "gemma4:latest",
  "is_production": true,
  "blocked_reason": null
}
```

Result: pass.

## 5. Negative Provider-check

Command:

```powershell
$env:APP_ENV="production"
$env:LLM_PROVIDER="local"
Remove-Item Env:\ALLOW_LOCAL_LLM_IN_PRODUCTION -ErrorAction SilentlyContinue
$env:LOCAL_LLM_MODEL="gemma4:latest"
python scripts\run_gemma4_evaluation_suite.py provider-check --production-like --output-dir tests\evaluation\.artifacts\gemma4-production-like-smoke-negative
```

Observed CLI result:

```json
{
  "app_env": "production",
  "requested_provider": "local",
  "effective_provider": "gemini",
  "local_allowed_in_production": false,
  "provider_check_status": "failed_production_local_opt_in_missing",
  "blocked_reason": "production_local_not_allowed"
}
```

Observed resolver fields:

```json
{
  "requested_provider": "local",
  "effective_provider": "gemini",
  "local_allowed_in_production": false,
  "model_name": "gemma4:latest",
  "is_production": true,
  "blocked_reason": "production_local_not_allowed"
}
```

Result: expected block. The negative command is intentionally non-strict because the expected status is a blocked production local provider, not `pass`.

## 6. Ollama Readiness

Command:

```text
python scripts/check_local_llm_readiness.py --base-url http://127.0.0.1:11434 --model gemma4:latest --timeout-sec 240 --strict
```

Observed result:

```json
{
  "ollama_reachable": true,
  "model_installed": true,
  "model": "gemma4:latest",
  "base_url_host": "127.0.0.1",
  "tags_latency_ms": 5,
  "probe_enabled": true,
  "probe_success": true,
  "probe_latency_ms": 11498,
  "json_parse_success": true,
  "readiness_status": "pass",
  "error_category": null
}
```

Result: pass.

Failure statuses to report if readiness fails:

- `failed_ollama_unreachable`
- `failed_model_missing`
- `failed_probe_timeout`
- `failed_probe_error`
- `failed_json_parse`

## 7. Production-like Server Env

This is a local production-like smoke env, not production cutover.

Operator-owned PowerShell env:

```powershell
$env:APP_ENV="production"
$env:LLM_PROVIDER="local"
$env:ALLOW_LOCAL_LLM_IN_PRODUCTION="true"
$env:LOCAL_LLM_MODEL="gemma4:latest"
$env:OLLAMA_BASE_URL="http://127.0.0.1:11434"
$env:ENABLE_LOCAL_RAW_JSON_INVOKE="true"
$env:LOCAL_LLM_NUM_PREDICT="4096"
$env:LOCAL_LLM_NUM_CTX="8192"
$env:ENABLE_FEEDBACK_AWARE_RANKING="false"
$env:ROUTINE_CANDIDATE_POOL_SIZE="12"
$env:ENABLE_RULE_CRITIC_TELEMETRY="true"
$env:ENABLE_DYNAMIC_TEMPERATURE="false"
$env:PYTHONIOENCODING="utf-8"
uvicorn main:app --host 127.0.0.1 --port 8000
```

Expected server behavior:

- `/docs` reachable.
- Provider log includes `production local provider enabled via explicit opt-in`.
- Provider log includes `local -> ChatOllama(model=gemma4:latest, format=json, engine=routine)`.
- No Gemini override.
- Telemetry includes `local_llm_num_predict=4096` and `local_llm_num_ctx=8192`.

Observed during this run:

- `/docs` on `http://127.0.0.1:8000` returned HTTP `200`.
- Recent dev logs showed `local -> ChatOllama(model=gemma4:latest, format=json, engine=routine)`.
- Recent telemetry showed `local_raw_json_invoke_used=true`, `local_raw_json_invoke_succeeded=true`, `raw_invoke_finish_reason=stop`, `local_llm_num_predict=4096`, and `local_llm_num_ctx=8192`.
- Recent telemetry also showed `feedback_enabled=true`, so it was not the exact P3-7a baseline env.

## 8. Small Generation Smoke

Use the suite runner for a focused 1-repeat smoke after the operator starts the server with the exact production-like env from section 7.

Do not run the default `preflight` in an environment where DB seed/reset is prohibited, because the current preflight path includes DB and seed checks. If DB/seed checks are approved for the target staging environment, the operator may run:

```text
python scripts/run_gemma4_evaluation_suite.py preflight --base-url http://127.0.0.1:8000 --ollama-base-url http://127.0.0.1:11434 --model gemma4:latest --output-dir tests/evaluation/.artifacts/gemma4-production-like-smoke --no-seed-reset --strict
```

Focused generation smoke:

```text
python scripts/run_gemma4_evaluation_suite.py run-a --base-url http://127.0.0.1:8000 --ollama-base-url http://127.0.0.1:11434 --model gemma4:latest --output-dir tests/evaluation/.artifacts/gemma4-production-like-smoke --repeats 1 --request-timeout 240 --scenario-ids normal_hypertrophy_upper,strength_lower --strict
```

Expected result:

- `request_count=2`.
- `failed_requests=0`.
- `timeout_count=0`.
- `telemetry_missing_count=0`.
- `feedback_enabled_rate=0.0`.
- `candidate_pool_size=12`.
- `llm_provider=local`.
- `local_llm_model=gemma4:latest`.
- `provider_mismatch_detected=false`.
- `local_raw_json_invoke_used_count=2`.
- `local_raw_json_invoke_succeeded_count=2`.
- `raw_invoke_finish_reason_counts={"stop": 2}`.
- `fallback_reason_counts={"none": 2}`.
- `hard_violation_count=0`.
- `privacy_leak_detected=false`.
- `local_llm_num_predict=4096`.
- `local_llm_num_ctx=8192`.

This run was not executed in this P3-7a pass because the active server env did not match the required baseline and server restart is intentionally operator-owned.

## 9. Observed Result Fields

Provider fields:

- `requested_provider`
- `effective_provider`
- `local_allowed_in_production`
- `blocked_reason`
- `model_name`
- `is_production`
- `provider_check_status`

Readiness fields:

- `ollama_reachable`
- `model_installed`
- `probe_success`
- `json_parse_success`
- `readiness_status`
- `error_category`

Generation smoke fields:

- `request_count`
- `failed_requests`
- `timeout_count`
- `telemetry_missing_count`
- `feedback_enabled_rate`
- `candidate_pool_size`
- `provider_mismatch_detected`
- `local_raw_json_invoke_used_count`
- `local_raw_json_invoke_succeeded_count`
- `raw_invoke_finish_reason_counts`
- `raw_invoke_stripped_empty_count`
- `fallback_rate`
- `fallback_reason_counts`
- `hard_violation_count`
- `privacy_leak_detected`
- `local_llm_num_predict`
- `local_llm_num_ctx`

## 10. Pass Criteria

P3-7a is fully pass only when all of these are true:

- Positive provider-check passes.
- Negative provider-check blocks local provider without opt-in.
- Ollama readiness passes.
- Production-like server starts with explicit opt-in.
- `/docs` is reachable.
- Effective provider is local.
- No Gemini override occurs.
- Small smoke `request_count >= 1`.
- `failed_requests=0`.
- `timeout_count=0`.
- `fallback_rate=0`.
- `hard_violation_count=0`.
- `privacy_leak_detected=false`.
- `local_raw_json_invoke_succeeded_count == request_count`.
- Raw invoke finish reason is `stop`.
- `local_llm_num_predict=4096`.
- `local_llm_num_ctx=8192`.

Current run status:

- Provider positive check: pass.
- Provider negative check: pass as expected block.
- Ollama readiness: pass.
- Server `/docs`: reachable on an already running process.
- Small generation smoke: pending operator-run with exact env.

## 11. Fail / Rollback Criteria

Stop or fail the smoke if:

- Positive provider-check does not resolve to local.
- Negative provider-check allows local without opt-in.
- Ollama readiness fails.
- Server starts but provider is Gemini.
- `fallback_rate > 0`.
- `finish_reason=length`.
- `raw_invoke_stripped_empty_count > 0`.
- `hard_violation_count > 0`.
- `privacy_leak_detected=true`.
- Timeout occurs.
- Provider mismatch is detected.
- Required telemetry is missing.

Rollback actions for a production-like server process:

- Remove `ALLOW_LOCAL_LLM_IN_PRODUCTION`, or set it to `false`.
- Set `LLM_PROVIDER=gemini` if returning to the previous effective provider is required.
- Set `ENABLE_FEEDBACK_AWARE_RANKING=false` if treatment-specific degradation is observed.
- Restart `fit-core-ai`.
- Verify effective provider state before any further traffic.

## 12. Risks

- The current `provider-check` CLI reports the guard status but not `model_name` or `is_production`; use `resolve_llm_provider()` directly when those fields must be recorded.
- The current `preflight` command includes DB and seed checks. In no-DB-change contexts, use readiness plus provider-check first, and run `preflight --no-seed-reset` only when DB/seed verification is approved.
- Reusing an already running server can mix env from previous phases. Confirm `feedback_enabled`, provider logs, and budget telemetry before generation smoke.
- Negative provider-check should not be followed by generation smoke in the same shell without resetting env.
- Ollama readiness can pass while production-like server env is still wrong; both checks are required.

## 13. Next Step

Run the pending small generation smoke after an operator starts `fit-core-ai` with the exact production-like positive env and `ENABLE_FEEDBACK_AWARE_RANKING=false`.

Recommended follow-up:

- P3-7b Internal canary runbook rehearsal after the focused generation smoke passes.

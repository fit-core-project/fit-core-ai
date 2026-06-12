# P3-7 Gemma4-only Controlled Rollout Plan

This document describes the controlled rollout plan for running the Gemma4-only local provider in staging or production-like environments. It is an operator runbook and handoff document. It does not declare that production cutover has happened.

## 1. Executive Summary

The Gemma4-only path is a controlled rollout candidate after the P3-6j full suite passed with 60 requests per group.

The strategy is to reduce dependency on Gemini quota by validating a local Ollama-backed provider for routine generation. The current state is ready for controlled rollout planning, not a completed production migration.

P3-6j key results:

| Metric | Result |
|---|---:|
| Group A requests | `60` |
| Group B requests | `60` |
| Failed requests | `0` / `0` |
| Timeout count | `0` / `0` |
| Telemetry missing count | `0` / `0` |
| Fallback rate | `0.0` / `0.0` |
| Hard violation count | `0` / `0` |
| Privacy leak detected | `false` |
| Feedback query failed rate, Group B | `0.0` |
| Critic score delta, B - A | `+0.8333` |
| Recommendation | `promote_gemma4_only_candidate` |

## 2. Rollout Scope

Included:

- `fit-core-ai` routine generation path.
- Gemma4 through the Ollama local provider.
- Raw JSON invoke path.
- Feedback-aware ranking evaluation path.
- Telemetry, readiness checks, and reporting.
- Staging and production-like opt-in validation.

Excluded:

- `fit-core-backend` DB or Flyway changes.
- Frontend changes.
- Gemini deletion.
- Production default provider changes.
- Dynamic temperature rollout.
- Candidate pool size `18`.
- Prompt overhaul.
- ML or LTR ranking.

## 3. Current Verified Configuration

| Env | Verified value | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `local` | Selects the local LLM provider path. |
| `LOCAL_LLM_MODEL` | `gemma4:latest` | Ollama model name used by ChatOllama. |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama HTTP endpoint. |
| `ENABLE_LOCAL_RAW_JSON_INVOKE` | `true` | Uses raw ChatOllama invoke plus the internal parser and validator instead of the LangChain `structured_output` wrapper. |
| `LOCAL_LLM_NUM_PREDICT` | `4096` | Output token budget. Required to prevent length-finish truncation. |
| `LOCAL_LLM_NUM_CTX` | `8192` | Context budget for the local model request. |
| `ROUTINE_CANDIDATE_POOL_SIZE` | `12` | Candidate pool size used by the verified suite. |
| `ENABLE_RULE_CRITIC_TELEMETRY` | `true` | Emits Rule Critic telemetry required for rollout evaluation. |
| `ENABLE_DYNAMIC_TEMPERATURE` | `false` | Keeps temperature policy stable during rollout. |
| `ENABLE_FEEDBACK_AWARE_RANKING` | `false` or `true` by phase | Disabled for control/baseline phases, enabled only for treatment phases. |
| `APP_ENV` + `ALLOW_LOCAL_LLM_IN_PRODUCTION` | `production` + `true` only for production-like opt-in | Explicit safety guard that allows local provider use in production-like mode. |

`ALLOW_LOCAL_LLM_IN_PRODUCTION` must not default to true. Production-like validation requires an explicit opt-in.

## 4. Why These Settings Are Required

The P3-6d through P3-6j sequence isolated the Gemma4 failure mode:

- Initial failures showed `schema_error_unrecoverable`, `schema_parse_error`, and excessive fallback.
- P3-6d confirmed `empty_response`.
- P3-6e proved raw output recovery could succeed.
- P3-6f identified `json_decode_expect_value`.
- P3-6g implemented structured-output bypass.
- P3-6h showed `AIMessage` content existed, but responses were affected by `finish_reason=length` and `stripped_empty`.
- P3-6i added `LOCAL_LLM_NUM_PREDICT=4096` and `LOCAL_LLM_NUM_CTX=8192`.
- P3-6j reached fallback `0`, with `finish_reason=stop` for `60/60` Group A and `60/60` Group B requests.

Core conclusion: the Gemma4 failure was primarily output budget exhaustion, not a safety violation. The controlled rollout requires raw JSON invoke plus sufficient output and context budgets.

## 5. Environment Matrix

### A. Local or Staging Control

```text
APP_ENV=development
LLM_PROVIDER=local
LOCAL_LLM_MODEL=gemma4:latest
OLLAMA_BASE_URL=http://127.0.0.1:11434
ENABLE_LOCAL_RAW_JSON_INVOKE=true
LOCAL_LLM_NUM_PREDICT=4096
LOCAL_LLM_NUM_CTX=8192
ENABLE_FEEDBACK_AWARE_RANKING=false
ROUTINE_CANDIDATE_POOL_SIZE=12
ENABLE_RULE_CRITIC_TELEMETRY=true
ENABLE_DYNAMIC_TEMPERATURE=false
```

`APP_ENV=staging` is also acceptable for staging validation.

### B. Local or Staging Treatment

Same as control, except:

```text
ENABLE_FEEDBACK_AWARE_RANKING=true
```

### C. Production-like Opt-in

```text
APP_ENV=production
LLM_PROVIDER=local
ALLOW_LOCAL_LLM_IN_PRODUCTION=true
LOCAL_LLM_MODEL=gemma4:latest
OLLAMA_BASE_URL=http://127.0.0.1:11434
ENABLE_LOCAL_RAW_JSON_INVOKE=true
LOCAL_LLM_NUM_PREDICT=4096
LOCAL_LLM_NUM_CTX=8192
ROUTINE_CANDIDATE_POOL_SIZE=12
ENABLE_RULE_CRITIC_TELEMETRY=true
ENABLE_DYNAMIC_TEMPERATURE=false
ENABLE_FEEDBACK_AWARE_RANKING=false
```

Set `ENABLE_FEEDBACK_AWARE_RANKING=true` only after staged validation passes.

### D. Negative Safety Check

```text
APP_ENV=production
LLM_PROVIDER=local
ALLOW_LOCAL_LLM_IN_PRODUCTION=false
```

Expected result: local provider is blocked, or the effective provider follows the existing non-local policy. This confirms the opt-in guard is working.

## 6. Preflight Checklist

Ollama:

- Ollama process is running.
- `http://127.0.0.1:11434/api/tags` is reachable.
- `gemma4:latest` is installed.
- Readiness command passes:

```text
python scripts/check_local_llm_readiness.py --base-url http://127.0.0.1:11434 --model gemma4:latest --timeout-sec 240 --strict
```

- `probe_success=true`.
- `json_parse_success=true`.
- Probe latency is acceptable for the rollout phase.

`fit-core-ai`:

- Virtual environment is active.
- Correct environment variables are loaded.
- Uvicorn is running at `http://127.0.0.1:8000`.
- `/docs` is reachable.
- `/api/dev/logs` is reachable in dev/staging.
- Provider log shows local provider routed to ChatOllama.
- No old Uvicorn process is still bound to port `8000`.
- No port collision exists.

DB and seed:

- `check-db` passes.
- Seed readiness passes.
- `verify-seed` passes.
- No accidental production DB usage.

Tests:

- `pytest tests/ -q` passes.
- Local provider policy tests pass.
- Gemma4 suite tests pass.

Privacy:

- Dev logs do not contain raw prompt, raw output, or `user_note`.
- Suite report has `privacy_leak_detected=false`.

## 7. Rollout Phases

### Phase 0 - No Production Change

- Keep production default unchanged.
- Perform only docs and local/staging validation.
- This is the current state.

### Phase 1 - Staging Gemma4 Provider Smoke

- Use `APP_ENV=staging`.
- Set `ENABLE_FEEDBACK_AWARE_RANKING=false`.
- Run readiness.
- Run a small focused smoke.
- Expected: fallback `0`, `finish_reason=stop`, hard violation `0`.

### Phase 2 - Staging Feedback-aware Treatment

- Set `ENABLE_FEEDBACK_AWARE_RANKING=true`.
- Run the 60-request-per-group suite.
- Expected: `promote_gemma4_only_candidate`, `avg_feedback_adjusted_count_b >= 1`, and `feedback_query_failed_rate_b=0`.

### Phase 3 - Production-like Opt-in Dry Run

- Set `APP_ENV=production`, `LLM_PROVIDER=local`, and `ALLOW_LOCAL_LLM_IN_PRODUCTION=true`.
- Use no real user traffic, or controlled internal traffic only.
- Verify effective provider is local.
- Run a small smoke.
- Verify there is no Gemini override.

### Phase 4 - Limited Internal Canary

- Restrict to internal or test users.
- Keep traffic low.
- Keep monitoring enabled.
- Keep rollback ready.
- Leave feedback-aware ranking disabled initially, or guard it with a controlled flag.

### Phase 5 - Controlled Rollout

- Use gradual percentage rollout or a manual cohort.
- Enable feedback-aware ranking only after the Gemma4 baseline is stable.
- Monitor fallback, latency, and hard violations.

### Phase 6 - General Availability Candidate

- Proceed only after multiple clean runs.
- Confirm the production incident plan is ready.
- Confirm resource and memory stability.

## 8. Execution Runbook

### A. Start Ollama

Start Ollama manually with the Ollama app or:

```text
ollama serve
```

The application must not auto-start Ollama.

### B. Check Readiness

```text
python scripts/check_local_llm_readiness.py --base-url http://127.0.0.1:11434 --model gemma4:latest --timeout-sec 240 --strict
```

### C. Start `fit-core-ai` Group A

PowerShell:

```powershell
$env:APP_ENV="development"
$env:LLM_PROVIDER="local"
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

### D. Run Preflight

```text
python scripts/run_gemma4_evaluation_suite.py preflight --base-url http://127.0.0.1:8000 --ollama-base-url http://127.0.0.1:11434 --model gemma4:latest --output-dir tests/evaluation/.artifacts/gemma4-suite --strict
```

### E. Run Group A

```text
python scripts/run_gemma4_evaluation_suite.py run-a --base-url http://127.0.0.1:8000 --ollama-base-url http://127.0.0.1:11434 --model gemma4:latest --output-dir tests/evaluation/.artifacts/gemma4-suite --repeats 5 --request-timeout 240
```

### F. Restart Group B

Set:

```powershell
$env:ENABLE_FEEDBACK_AWARE_RANKING="true"
```

Keep the raw invoke and budget env unchanged, then restart Uvicorn.

### G. Run Group B

```text
python scripts/run_gemma4_evaluation_suite.py run-b --base-url http://127.0.0.1:8000 --ollama-base-url http://127.0.0.1:11434 --model gemma4:latest --output-dir tests/evaluation/.artifacts/gemma4-suite --repeats 5 --request-timeout 240
```

### H. Compare

```text
python scripts/run_gemma4_evaluation_suite.py compare --output-dir tests/evaluation/.artifacts/gemma4-suite --repeats 5 --request-timeout 240 --min-requests-per-group 50 --strict
```

### I. Extract Summary

```text
python -c "import json, pathlib; p=pathlib.Path('tests/evaluation/.artifacts/gemma4-suite/gemma4-suite-report.json'); r=json.loads(p.read_text(encoding='utf-8')); keys=['recommendation','fallback_rate_a','fallback_rate_b','hard_violation_count_a','hard_violation_count_b','privacy_leak_detected','raw_invoke_finish_reason_counts_a','raw_invoke_finish_reason_counts_b','local_llm_num_predict_a','local_llm_num_ctx_a','avg_critic_score_a','avg_critic_score_b','blockers','warnings']; print(json.dumps({k:r.get(k) for k in keys}, indent=2))"
```

## 9. Monitoring Plan

LLM and provider:

- `effective_provider`
- `local_raw_json_invoke_used`
- `local_raw_json_invoke_succeeded`
- `raw_invoke_finish_reason`
- `raw_invoke_done_reason`
- `raw_invoke_stripped_empty`
- `local_llm_num_predict`
- `local_llm_num_ctx`
- `raw_invoke_error_category`

Quality and safety:

- `hard_violation_count`
- `hard_violation_category_counts`
- `avg_critic_score`
- `critic_score_delta`
- `fallback_rate`
- `fallback_reason_counts`
- `schema_parse_error`
- `schema_repair_attempted`
- `schema_repair_succeeded`
- `parse_failure_subtype_counts`

Latency:

- `avg_generation_latency_ms`
- `p95_generation_latency_ms`
- `max_generation_latency_ms`
- Ollama readiness probe latency.

Feedback:

- `feedback_enabled`
- `avg_feedback_adjusted_count`
- `feedback_query_failed_rate`
- `p95_feedback_query_latency_ms`

Privacy:

- `privacy_leak_detected`
- Forbidden token scan result.
- Dev log redaction status.

Resource:

- System memory.
- Ollama process uptime.
- Model load time.
- CPU or GPU usage when available.

## 10. Pass Criteria

Promotion requires:

- `readiness_status=pass`.
- `request_count >= 50` per group.
- `failed_requests=0`.
- `timeout_count=0`.
- `telemetry_missing_count=0`.
- `fallback_rate <= 0.05`, preferably `0`.
- `hard_violation_count=0`.
- `privacy_leak_detected=false`.
- `raw_invoke_finish_reason` is dominated by `stop`.
- `raw_invoke_stripped_empty_count=0`.
- `local_raw_json_invoke_succeeded_count` approximately equals request count.
- `quota_fallback_rate=0`.
- `feedback_query_failed_rate=0`.
- `avg_feedback_adjusted_count_b >= 1` when feedback-aware ranking is enabled.
- `critic_score_b >= critic_score_a - 5`.
- No blockers.
- No unresolved warnings.

## 11. Rollback Criteria

Rollback immediately if:

- `hard_violation_count > 0`.
- `privacy_leak_detected=true`.
- `fallback_rate > 0.05` sustained.
- `finish_reason=length` spikes.
- `raw_invoke_stripped_empty_count > 0` sustained.
- `timeout_count > 0` above tolerance.
- Ollama readiness fails.
- Model memory error occurs.
- `feedback_query_failed_rate > 0`.
- Critic score drops significantly.
- User-visible routine generation failures increase.
- Provider mismatch is detected.
- `ALLOW_LOCAL_LLM_IN_PRODUCTION` is misconfigured.
- Dev logs contain raw prompt, raw output, or `user_note`.

Rollback actions:

- Set `LLM_PROVIDER` back to the previous effective provider, or disable local provider opt-in.
- Remove or disable `ALLOW_LOCAL_LLM_IN_PRODUCTION`.
- Set `ENABLE_FEEDBACK_AWARE_RANKING=false` if needed.
- Restart `fit-core-ai`.
- Verify effective provider state.
- Preserve incident artifacts.
- Do not delete logs before redaction review.

## 12. Failure Mode Playbook

### A. Ollama Unreachable

- Check the Ollama process.
- Check port `11434`.
- Restart Ollama manually.
- Rerun readiness.

### B. Model Missing

- Run `ollama list`.
- Pull the model manually.
- Rerun readiness.

### C. Memory Error

- Close other processes.
- Reduce load.
- Consider a smaller model.
- Do not proceed to rollout.

### D. `finish_reason=length`

- Verify `LOCAL_LLM_NUM_PREDICT=4096`.
- Verify `LOCAL_LLM_NUM_CTX=8192`.
- Confirm telemetry sees both values.
- If length finishes continue, consider prompt or payload reduction in a separate change.

### E. `schema_parse_error`

- Inspect `parse_failure_subtype_counts`.
- Inspect `json_decode_error_category_counts`.
- Keep raw output private.

### F. Hard Violation

- Stop rollout.
- Inspect hard violation category.
- Do not relax Rule Critic.

### G. Fallback Spike

- Inspect `fallback_reason_counts`.
- Inspect `raw_invoke_finish_reason`.
- Inspect schema repair metrics.

### H. Feedback Treatment Degradation

- Set `ENABLE_FEEDBACK_AWARE_RANKING=false`.
- Keep Gemma4 provider only if the baseline is stable.
- Analyze feedback aggregation separately.

## 13. Security / Privacy Policy

Must not store:

- Raw prompt.
- Raw LLM output.
- Raw model response.
- `user_note`.
- Raw feedback rows.
- API keys.
- Credential-bearing URLs.
- Full environment dumps.
- Stack traces with local variables.

Allowed:

- Aggregate counts.
- Enum categories.
- Latency stats.
- Character or token counts.
- Provider and model safe identifiers.
- `finish_reason` or `done_reason` safe tokens.
- Length buckets.

Existing controls:

- `log_redaction.py` allowlist.
- Dev log redaction.
- Structured telemetry only.

## 14. Operational Risks

| Risk | Mitigation |
|---|---|
| Local hardware dependency | Validate on target hardware and monitor resource headroom. |
| Ollama process crash | Use readiness checks and an operator-owned restart process. |
| Memory pressure | Monitor memory and stop rollout on model memory errors. |
| Model load latency | Warm up before canary and track readiness probe latency. |
| Windows port/process confusion | Check port ownership before starting Uvicorn or Ollama. |
| Long generation latency | Track p95 and max generation latency. |
| Budget misconfiguration | Require telemetry for `LOCAL_LLM_NUM_PREDICT` and `LOCAL_LLM_NUM_CTX`. |
| Feedback-aware ranking prompt complexity | Enable only after stable baseline. |
| Suite overfitting to current 60 requests | Require repeated clean runs and canary monitoring. |
| Production opt-in misconfiguration | Run positive and negative provider checks. |
| Rollback not rehearsed | Perform a rollback drill before wider rollout. |

## 15. Backend / AI Server Separation

- `fit-core-ai`: Python FastAPI service, port `8000`, Gemma4 suite target.
- `fit-core-backend`: Spring Boot business API, port `8080`.
- Ollama: local LLM server, port `11434`.
- The P3-6j suite targeted `fit-core-ai`.
- The backend Flyway V6 issue was a local backend startup issue and is not directly related to the Gemma4 suite.
- Operators must not confuse the AI server, backend server, and Ollama process during rollout.

## 16. Artifact Requirements

Preserve these artifacts for each rollout run:

- `preflight.json`
- `group_a.json`
- `group_b.json`
- `feedback-ranking-compare.json`
- `gemma4-suite-report.json`
- `gemma4-suite-report.md`
- Pytest output.
- Readiness output.
- Environment matrix snapshot without secrets.

Path convention:

```text
tests/evaluation/.artifacts/gemma4-suite-<date-or-phase>
```

## 17. Production-like Opt-in Validation

Positive check:

```text
APP_ENV=production
LLM_PROVIDER=local
ALLOW_LOCAL_LLM_IN_PRODUCTION=true
python scripts/run_gemma4_evaluation_suite.py provider-check --production-like --strict
```

Expected: `effective_provider=local`.

Negative check:

```text
APP_ENV=production
LLM_PROVIDER=local
ALLOW_LOCAL_LLM_IN_PRODUCTION=false
python scripts/run_gemma4_evaluation_suite.py provider-check --production-like --strict
```

Expected: local provider is blocked or replaced according to the existing provider policy.

## 18. Decision Record

- Gemma4-only candidate is accepted for controlled rollout planning.
- Local raw JSON invoke is required.
- `LOCAL_LLM_NUM_PREDICT=4096` is required for the verified path.
- `LOCAL_LLM_NUM_CTX=8192` is required for the verified path.
- Feedback-aware ranking can proceed only with the same raw invoke and budget settings.
- Production default remains unchanged until the controlled rollout gate is explicitly passed.
- Gemini code is retained as a rollback and compatibility path.

## 19. Next Steps

- P3-7a Production-like provider-check smoke.
- P3-7b Internal canary runbook rehearsal.
- P3-7c Monitoring dashboard or report checklist.
- P3-7d Rollback drill.
- P3-8 Actual controlled rollout decision if all gates pass.

## 20. Interview / Explanation Notes

Why move from Gemini to Gemma4:

- Gemini quota dependency can block or degrade routine generation. A local provider gives the team an operator-controlled path for staging and production-like validation.

Why raw JSON invoke was needed:

- The structured-output wrapper path produced unrecoverable schema and parse failures. Raw ChatOllama invoke preserved model content so the internal parser and validator could recover valid JSON safely.

Why `num_predict` and `num_ctx` were needed:

- P3-6h showed content existed but responses were truncated with length finishes. Increasing output and context budgets resolved the truncation pattern in P3-6j.

How safety is preserved:

- Rule Critic remains unchanged.
- Hard violation telemetry is required.
- Fallback thresholds are not relaxed.
- Raw prompt and raw output are not stored.
- Production local provider remains behind explicit opt-in.

How rollout risk is reduced:

- The plan uses staging first, then production-like opt-in dry run, then limited internal canary, then gradual controlled rollout.
- Rollback criteria are explicit.
- Gemini remains available as compatibility and rollback path.

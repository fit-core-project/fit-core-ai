# Gemma4-Only Provider Policy

Fit Core is moving routine generation toward a Gemma4 via Ollama provider path. Gemini support remains in the codebase for compatibility and rollback, but production use of the local provider must be explicit.

## Provider Selection

Default behavior remains unchanged:

- `LLM_PROVIDER` defaults to `gemini`.
- `APP_ENV=production` with `LLM_PROVIDER=local` is blocked by default and falls back to the Gemini provider path.
- `ENABLE_FEEDBACK_AWARE_RANKING` keeps its existing default behavior.
- `ROUTINE_CANDIDATE_POOL_SIZE` policy is unchanged.

Production local provider opt-in:

```text
APP_ENV=production
LLM_PROVIDER=local
ALLOW_LOCAL_LLM_IN_PRODUCTION=true
LOCAL_LLM_MODEL=gemma4:latest
OLLAMA_BASE_URL=http://127.0.0.1:11434
ROUTINE_CANDIDATE_POOL_SIZE=12
ENABLE_RULE_CRITIC_TELEMETRY=true
ENABLE_DYNAMIC_TEMPERATURE=false
```

`ALLOW_LOCAL_LLM_IN_PRODUCTION` accepts `true`, `1`, `yes`, or `on`. Any other value is treated as false.

## Safety Guard

If production requests the local provider without opt-in, the router logs a sanitized warning:

```text
[LLM Router] WARNING: local provider blocked in production; set ALLOW_LOCAL_LLM_IN_PRODUCTION=true to enable
```

If production local is explicitly enabled, the router logs:

```text
[LLM Router] production local provider enabled via explicit opt-in
```

The router does not log API keys, raw environment dumps, prompts, raw model output, user notes, or full Ollama URLs.

## Ollama Readiness

Before enabling production local provider:

```text
ollama pull gemma4:latest
ollama list
```

Confirm the configured `LOCAL_LLM_MODEL` exists and Ollama is reachable from the AI service runtime. For container deployments, `OLLAMA_BASE_URL` must point to a host reachable from inside the container.

Run the operational preflight before production opt-in:

```text
python scripts/check_local_llm_readiness.py --base-url http://127.0.0.1:11434 --model gemma4:latest --strict
```

See `docs/gemma4_operational_readiness.md` for report fields and failure handling.

## Rollback

Rollback options:

- Remove `ALLOW_LOCAL_LLM_IN_PRODUCTION`.
- Set `ALLOW_LOCAL_LLM_IN_PRODUCTION=false`.
- Set `LLM_PROVIDER=gemini`.

With `APP_ENV=production`, any missing or false opt-in returns to the existing Gemini override behavior.

## Known Limitations

- The local provider depends on Ollama process availability and installed model state.
- Model latency can be substantially higher than Gemini; staging runs should continue to use explicit request timeouts.
- Provider selection is internal runtime configuration and does not alter public API response schemas.

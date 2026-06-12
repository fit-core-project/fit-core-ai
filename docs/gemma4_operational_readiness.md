# Gemma4 Operational Readiness

This document covers preflight checks for running Gemma4 through Ollama. The readiness tool only checks state. It does not start Ollama, pull models, change provider defaults, or modify production behavior.

## Preconditions

Install Ollama and make the target model available before enabling the local provider:

```text
ollama pull gemma4:latest
ollama serve
```

For a containerized AI service, `OLLAMA_BASE_URL` must point to an Ollama endpoint reachable from inside the container.

## Readiness Command

Default check:

```text
python scripts/check_local_llm_readiness.py
```

Explicit Gemma4 check:

```text
python scripts/check_local_llm_readiness.py --base-url http://127.0.0.1:11434 --model gemma4:latest --strict
```

Write a JSON report:

```text
python scripts/check_local_llm_readiness.py --output tests/evaluation/.artifacts/local-llm-readiness.json
```

Skip the generation probe and only verify `/api/tags` plus model presence:

```text
python scripts/check_local_llm_readiness.py --no-probe
```

## Report Shape

The report contains structured readiness fields only:

```json
{
  "ollama_reachable": true,
  "model_installed": true,
  "model": "gemma4:latest",
  "base_url_host": "127.0.0.1",
  "tags_latency_ms": 12,
  "probe_enabled": true,
  "probe_success": true,
  "probe_latency_ms": 1345,
  "json_parse_success": true,
  "readiness_status": "pass",
  "error_category": null
}
```

The tool does not store raw prompts, raw generated content, raw environment dumps, user notes, API keys, or credential-bearing URLs.

## Failure Statuses

- `failed_ollama_unreachable`: `/api/tags` could not be reached.
- `failed_model_missing`: Ollama is reachable, but the configured model is absent from `/api/tags`.
- `failed_probe_timeout`: the generation probe timed out.
- `failed_probe_error`: the generation probe failed for a non-timeout reason.
- `failed_json_parse`: the probe returned content that was not valid JSON.

`--strict` exits non-zero for any status other than `pass`. Without `--strict`, the command prints the report and exits zero unless the script itself crashes.

## Production Example

Production local provider remains opt-in:

```text
APP_ENV=production
LLM_PROVIDER=local
ALLOW_LOCAL_LLM_IN_PRODUCTION=true
LOCAL_LLM_MODEL=gemma4:latest
OLLAMA_BASE_URL=http://127.0.0.1:11434
ROUTINE_CANDIDATE_POOL_SIZE=12
ENABLE_RULE_CRITIC_TELEMETRY=true
ENABLE_DYNAMIC_TEMPERATURE=false
ENABLE_FEEDBACK_AWARE_RANKING=false
```

Set `ENABLE_FEEDBACK_AWARE_RANKING=true` only when the rollout step calls for it.

## Rollback

Rollback options:

- Set `LLM_PROVIDER=gemini`.
- Remove `ALLOW_LOCAL_LLM_IN_PRODUCTION`.
- Set `ALLOW_LOCAL_LLM_IN_PRODUCTION=false`.

When `APP_ENV=production`, missing or false opt-in keeps the local provider blocked.

## Limitations

- The script does not start the Ollama server.
- The script does not pull or install missing models.
- Probe latency depends on local hardware, model warmup state, and current Ollama load.

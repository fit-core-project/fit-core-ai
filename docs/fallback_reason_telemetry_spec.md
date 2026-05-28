# P1.8 Fallback Reason / LLM Error Telemetry Spec

## Goal

Record structured fallback causes in routine generation telemetry without changing runtime behavior, response contracts, ranking, rebuild triggers, or fallback triggers.

## LLM Error Type

`engines.llm_router.classify_llm_error(exc)` returns one of:

- `timeout`
- `quota_exhausted`
- `schema_parse_error`
- `validation_error`
- `network_error`
- `unknown`

The classifier is never-raise and does not expose raw exception messages in telemetry.

## Fallback Reason

`engines.routine_telemetry.classify_fallback_reason(...)` returns one of:

- `none`
- `timeout`
- `quota_exhausted`
- `schema_error_unrecoverable`
- `schema_error_recovered`
- `validation_failed`
- `time_budget_exceeded`
- `network_error`
- `unknown_llm_error`

## Telemetry Fields

The existing `fallback_used` field remains. P1.8 adds:

- `fallback_reason`
- `llm_error_type`
- `schema_repair_attempted`
- `schema_repair_succeeded`

No raw prompt, raw LLM output, raw exception message, rationale text, user note, or feedback free text is recorded.

## Pipeline Integration

`generate_smart_routine()` tracks LLM error type, schema repair attempt/success, validation fallback, and time-budget fallback through local variables and callbacks. These values are passed into `observe_routine_quality()` only for telemetry.

The pipeline does not branch on these telemetry fields.

## Live A/B Report

`scripts/run_live_candidate_pool_ab_test.py` aggregates:

- `fallback_reason_counts_12` / `fallback_reason_counts_18`
- `llm_error_type_counts_12` / `llm_error_type_counts_18`
- quota fallback counts/rates
- schema repair attempted/succeeded counts and success rates

If quota fallback rate exceeds `0.5` in either group, recommendation becomes `inconclusive_quota_exhausted`.

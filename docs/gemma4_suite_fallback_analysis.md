# Gemma4 Suite Fallback Analysis

## 1. Summary

The Gemma4-only suite rerun did not fail because of request errors, timeouts, privacy leakage, quota fallback, feedback query failures, or hard rule violations. The blocking signal is high routine fallback usage in the local Gemma4 generation path.

The control group already has a high fallback rate: 23/60 = 38.33%. Treatment is worse: 44/60 = 73.33%. Every recorded fallback in both groups is classified as `schema_error_unrecoverable`, with `llm_error_type=schema_parse_error`. Schema repair was attempted for every fallback and succeeded zero times.

The current strongest root cause is Gemma4 structured-output/schema adherence instability. Feedback-aware ranking appears to amplify the problem in several scenarios, but the artifact does not contain enough candidate ordering/source detail to prove that feedback ranking itself is the direct cause.

## 2. Current suite result

- Output directory: `tests/evaluation/.artifacts/gemma4-suite-rerun`
- Suite recommendation: `inconclusive`
- Suite blockers: `["control_fallback_rate"]`
- Lower-level recommendation: `keep_feedback_off`
- Provider/model: `local` / `gemma4:latest`
- Request count: A=60, B=60
- Failed requests: A=0, B=0
- Timeouts: A=0, B=0
- Telemetry missing: A=0, B=0
- Hard violations: A=0, B=0
- Quota fallback rate: A=0.0, B=0.0
- Feedback query failed rate B: 0.0
- Privacy leak detected: false

The suite-level recommendation is `inconclusive` because `scripts/run_gemma4_evaluation_suite.py` checks `fallback_rate_a > 0.05` before treatment fallback spike. Since control fallback rate is 0.3833, it records `control_fallback_rate` and returns `inconclusive`.

The lower-level staging comparison returns `keep_feedback_off` because treatment fallback rate is higher than control by more than 0.05.

## 3. Why recommendation is inconclusive/keep_feedback_off

The decisive metrics are:

| Metric | Group A | Group B |
| --- | ---: | ---: |
| fallback count | 23/60 | 44/60 |
| fallback rate | 0.3833 | 0.7333 |
| schema repair attempted | 23/60 | 44/60 |
| schema repair succeeded | 0/60 | 0/60 |
| llm error type | 23 schema parse | 44 schema parse |
| hard violation count | 0 | 0 |
| timeout count | 0 | 0 |

This means the suite is not rejecting Gemma4 for unsafe routines. It is rejecting or blocking the rollout decision because Gemma4 frequently fails to produce a parseable schema-compatible routine, then the repair path cannot recover the output.

## 4. Artifact schema analysis

Analyzed files:

- `group_a.json`
- `group_b.json`
- `group_a_summary.json`
- `group_b_summary.json`
- `feedback-ranking-compare.json`
- `gemma4-suite-report.json`

`group_a.json` and `group_b.json` are top-level objects with run metadata and a `results` list. The group label is top-level as `group`; individual result records do not repeat a `group` field.

Each result record has:

- `scenario_id`
- `response`
- `telemetry`

The `response` summary contains:

- `routineDraftId`
- `generationStatus`
- `isFallback`
- `routineBlockCount`

The `telemetry` summary contains the important fallback analysis fields:

- `fallback_used`
- `fallback_reason`
- `llm_error_type`
- `repair_count`
- `schema_repair_attempted`
- `schema_repair_succeeded`
- `schema_repair_latency_ms`
- `critic_score`
- `critic_grade`
- `candidate_pool_size`
- `candidate_count`
- `candidate_payload_char_count`
- `approx_candidate_payload_tokens`
- `feedback_enabled`
- `feedback_adjusted_candidate_count`
- `feedback_positive_count`
- `feedback_negative_count`
- `feedback_abs_adjustment_avg`
- `feedback_query_latency_ms`
- `feedback_query_failed`
- `feedback_query_error_category`

Fields not present in the artifact:

- `fallback_reason_category`
- explicit response source or fallback source beyond `fallback_used` / `generationStatus`
- candidate IDs before and after feedback ranking
- selected exercise IDs from successful LLM output
- sanitized parse failure subtype, such as malformed JSON vs Pydantic field mismatch
- sanitized repair failure subtype

The current artifact is sufficient to identify schema parse/repair failure as the broad fallback cause. It is not sufficient to determine the exact malformed schema pattern or prove a specific feedback-ranked exercise/candidate ordering caused the parse failures.

## 5. Fallback breakdown

Fallback reason counts:

| Reason | Group A | Group B |
| --- | ---: | ---: |
| `none` | 37 | 16 |
| `schema_error_unrecoverable` | 23 | 44 |

LLM error type counts:

| Error type | Group A | Group B |
| --- | ---: | ---: |
| `None` | 37 | 16 |
| `schema_parse_error` | 23 | 44 |

Generation status counts:

| Status | Group A | Group B |
| --- | ---: | ---: |
| `success` | 37 | 16 |
| `fallback` | 23 | 44 |

Schema repair:

| Metric | Group A | Group B |
| --- | ---: | ---: |
| attempted count | 23 | 44 |
| attempted rate | 0.3833 | 0.7333 |
| succeeded count | 0 | 0 |
| succeeded rate | 0.0 | 0.0 |

Fallback by scenario:

| Scenario | Group A | Group B | Delta |
| --- | ---: | ---: | ---: |
| `bodyweight_only` | 0/5 | 0/5 | 0 |
| `fatloss_limited_equipment` | 1/5 | 5/5 | +4 |
| `knee_pain_present` | 5/5 | 4/5 | -1 |
| `long_duration_60m` | 5/5 | 5/5 | 0 |
| `low_readiness` | 5/5 | 5/5 | 0 |
| `mixed_target_muscles` | 1/5 | 5/5 | +4 |
| `normal_hypertrophy_upper` | 0/5 | 5/5 | +5 |
| `normal_hypertrophy_upper_fb` | 0/5 | 0/5 | 0 |
| `short_duration_20m` | 1/5 | 0/5 | -1 |
| `shoulder_pain_present` | 5/5 | 5/5 | 0 |
| `strength_lower` | 0/5 | 5/5 | +5 |
| `strength_lower_fb` | 0/5 | 5/5 | +5 |

Critic grade distribution, fallback vs non-fallback:

| Group | Subset | FAIL | WARN | PASS |
| --- | --- | ---: | ---: | ---: |
| A | fallback | 17 | 6 | 0 |
| A | non-fallback | 4 | 33 | 0 |
| B | fallback | 39 | 5 | 0 |
| B | non-fallback | 0 | 15 | 1 |

The critic grade distribution is mostly a consequence of fallback use: fallback records are graded worse, while non-fallback B records are mostly WARN plus one PASS.

## 6. Group A vs Group B comparison

Group B has feedback enabled on every request and averages 2.8333 adjusted candidates per request. Feedback query latency is effectively irrelevant here: p95 is 1 ms and failure rate is 0.0.

Payload size is not a large global change because both groups use candidate pool size 12. However, fallback records in Group B have a larger average candidate payload than non-fallback records:

| Group/subset | Avg payload chars | Avg approx tokens | Avg adjusted candidates | Avg latency ms |
| --- | ---: | ---: | ---: | ---: |
| A fallback | 3765.91 | 941.70 | 0.00 | 20760.61 |
| A non-fallback | 3648.35 | 912.46 | 0.00 | 26863.22 |
| B fallback | 3889.82 | 972.73 | 3.14 | 18966.39 |
| B non-fallback | 3355.19 | 839.06 | 2.00 | 25934.19 |

The biggest B-only regressions are concentrated in:

- `normal_hypertrophy_upper`: 0/5 to 5/5 fallback
- `strength_lower`: 0/5 to 5/5 fallback
- `strength_lower_fb`: 0/5 to 5/5 fallback
- `fatloss_limited_equipment`: 1/5 to 5/5 fallback
- `mixed_target_muscles`: 1/5 to 5/5 fallback

Scenarios already unstable in control remained unstable in treatment:

- `long_duration_60m`: 5/5 in both
- `low_readiness`: 5/5 in both
- `shoulder_pain_present`: 5/5 in both

This pattern suggests two layers:

1. Gemma4 has baseline schema-output instability on several scenario families even without feedback.
2. Feedback-aware ranking or its resulting candidate ordering/payload likely increases schema parse failures in otherwise stable scenarios.

## 7. Root cause hypotheses

Most likely:

- Gemma4 structured output is frequently malformed or schema-incompatible for this prompt and `LLMRoutineOutput` contract.
- The current repair path does not recover these failures. The artifact shows zero successful schema repairs in 67 attempted repairs across both groups.
- Feedback-aware ranking increases the probability of schema parse failure in several scenarios, possibly by changing candidate order, candidate details, or exercise mix enough for Gemma4 to drift from the strict schema.

Plausible but not proven from current artifacts:

- Candidate payload complexity matters. Group B fallback records have higher average payload chars/tokens than Group B non-fallback records.
- Feedback adjustment intensity matters. Group B fallback records average 3.14 adjusted candidates, while non-fallback records average 2.00.
- Specific feedback-ranked exercise combinations may trigger malformed JSON or field mismatches.

Less likely based on current data:

- Timeout: zero timeout count and p95 generation latency is below request timeout.
- Quota: quota fallback rate is 0.0 in both groups.
- Feedback query service: query failure rate is 0.0 and p95 query latency is 1 ms.
- Rule Critic strictness: hard violation count is zero, and the fallback reason is schema parse, not hard violation.
- Provider mismatch: provider is local Gemma4 in both groups.

## 8. Report observability gaps

The lower-level `feedback-ranking-compare.json` already contains useful aggregate fields:

- `fallback_reason_distribution_a/b`
- `llm_error_type_distribution_a/b`
- `schema_repair_attempted_count_a/b`
- `schema_repair_succeeded_count_a/b`

The suite-level `gemma4-suite-report.json` does not carry all of those forward. It has fallback rates and schema repair rates, but not reason/error distributions or scenario-level fallback concentration.

Missing report fields that should be added:

- `fallback_reason_counts_a`
- `fallback_reason_counts_b`
- `llm_error_type_counts_a`
- `llm_error_type_counts_b`
- `fallback_count_by_scenario_a`
- `fallback_count_by_scenario_b`
- `fallback_rate_by_scenario_a`
- `fallback_rate_by_scenario_b`
- `schema_repair_attempted_count_a`
- `schema_repair_attempted_count_b`
- `schema_repair_succeeded_count_a`
- `schema_repair_succeeded_count_b`
- optional sanitized `schema_parse_failure_type_counts_a/b`
- optional sanitized `schema_repair_failure_type_counts_a/b`

The staging compare report should also expose fallback count by scenario, not only reason/error distributions. This can be computed from existing per-result telemetry without storing raw prompts, raw LLM output, user notes, or secrets.

## 9. Recommended next implementation task

Recommended next task: P3-6b Fallback Breakdown Reporting.

Scope:

- Add fallback reason/error distributions to the suite-level JSON and Markdown report.
- Add fallback count/rate by scenario to both staging compare and suite-level reports.
- Add schema repair attempted/succeeded counts next to existing rates.
- Preserve privacy constraints: only aggregate scenario IDs, categories, and counts.
- Use existing artifact/telemetry fields first; no long rerun is required to implement or test the report builder.

After reporting is strengthened, choose the next hardening task based on the breakdown:

- P3-6c if schema/repair failures dominate, as they do in this rerun.
- P3-6d if B-only failures correlate with candidate payload/order/feedback-adjusted candidates.
- P3-6e if sanitized parse failure subtype shows malformed JSON dominates.

Given this rerun, P3-6c is likely needed after P3-6b, but P3-6b should come first so the next run can explain failures without manual artifact mining.

## 10. Whether rerun is needed

No immediate long Gemma4 suite rerun is needed for this analysis. The existing artifact is enough to identify the broad root cause: schema parse failures with unrecovered repair attempts.

A rerun is useful only after adding fallback breakdown reporting or after implementing schema/repair hardening. The next rerun should verify:

- control fallback rate falls below the suite threshold,
- treatment fallback does not spike over control,
- schema repair either succeeds at a measurable rate or parse failures decrease,
- hard violations, timeout, quota fallback, and privacy remain clean.

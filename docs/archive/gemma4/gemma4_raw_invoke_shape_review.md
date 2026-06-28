# P3-6h Review: Local Raw Invoke Response Shape Analysis

**Review date:** 2026-06-04  
**Reviewer:** Claude Code (independent re-analysis)  
**Original implementation:** Codex (P3-6h)  
**Baseline:** P3-6g focused smoke — `empty_response` dominant failure, `json_decode_expect_value` secondary

---

## 1. Summary

P3-6h adds safe structural telemetry for the ChatOllama/Gemma4 raw invoke path (`ENABLE_LOCAL_RAW_JSON_INVOKE=true`) and hardens content extraction to handle list content blocks and nested message objects. All 10 required `raw_invoke_*` fields are confirmed present in pipeline, telemetry, log allowlist, staging runner, and evaluation suite. 10 new tests were added. **No raw model content is stored anywhere.** Two minor gaps are noted: staging report markdown does not render `raw_invoke_*` fields, and 4 of the 10 shape fields are not aggregated in suite compare outputs. These are non-blocking for P3-6h smoke.

---

## 2. Why P3-6h Was Needed

P3-6g (focused, 8 scenarios, repeats=1) produced:

| metric | Group A | Group B |
|---|---:|---:|
| `local_raw_json_invoke_used` | 8/8 | 8/8 |
| `local_raw_json_invoke_succeeded` | 2/8 | 0/8 |
| `local_raw_json_invoke_failed_reason: empty_response` | 6 | 8 |
| `json_decode_error_category: json_decode_expect_value` | 6 | 8 |
| `raw_output_recovery_source: exception_args` | 6 | 8 |

**Core question:** Was `empty_response` caused by content extraction missing list/nested formats, or was Gemma4 genuinely returning empty content?

**Secondary problem:** No structural telemetry existed to answer this — the code only tracked that the failure happened, not what the response object looked like.

---

## 3. Codex Implementation Review

### 3.1 engines/routine_pipeline.py — CONFIRMED COMPLETE

| Function | Lines | Status |
|---|---|---|
| `_content_length_bucket(value: str) -> str` | 82–91 | ✓ implemented |
| `_safe_metadata_category(value: Any) -> Optional[str]` | 93–102 | ✓ implemented |
| `_text_from_content_blocks(blocks: list) -> str` | 104–113 | ✓ implemented |
| `_extract_response_metadata(message) -> dict` | 116–118 | ✓ implemented |
| `_extract_llm_message_content_with_shape(message) -> tuple[str, dict]` | 121–176 | ✓ implemented |
| `_invoke_local_raw_json_output_with_shape(...)` | 179–182 | ✓ implemented |
| `_extract_llm_message_content()` delegates to `_with_shape` | 77–79 | ✓ updated |
| 10 `raw_invoke_*` variables initialized in `generate_smart_routine` | 504–513 | ✓ present |
| Shape dict read in local raw invoke branch | 598–608 | ✓ present |
| Shape variables passed to `_observe` → `observe_routine_quality` | 568–577 | ✓ present |

### 3.2 engines/routine_telemetry.py — CONFIRMED COMPLETE

| Location | Status |
|---|---|
| 10 `raw_invoke_*` params in `build_routine_quality_telemetry_payload` | ✓ lines 147–156 |
| 10 fields in returned dict | ✓ lines 213–222 |
| 10 `raw_invoke_*` params in `observe_routine_quality` | ✓ lines 283–292 |
| 10 fields passed through to `build_...` call | ✓ lines 351–360 |

### 3.3 engines/log_redaction.py — CONFIRMED COMPLETE

All 10 `raw_invoke_*` field names are in `SAFE_DEV_LOG_FIELDS` (lines 38–53). Dev log sanitization will allow these fields through.

### 3.4 scripts/run_feedback_ranking_staging_test.py — MOSTLY COMPLETE

| Component | Status |
|---|---|
| 10 `raw_invoke_*` keys in `_telemetry_summary()` | ✓ lines 731–740 |
| 6 `raw_invoke_*` aggregate fields in `cmd_compare()` | ✓ lines 1231–1242 |
| `raw_invoke_*` fields in `_markdown()` report | **MISSING** — markdown only shows standard fields |

Aggregated in compare JSON: `response_class_counts`, `content_present_count`, `stripped_empty_count`, `finish_reason_counts`, `done_reason_counts`, `error_category_counts`.  
NOT aggregated: `content_type`, `content_length_bucket`, `has_response_metadata`, `usage_present` (per-event only).

### 3.5 scripts/run_gemma4_evaluation_suite.py — CONFIRMED COMPLETE

| Component | Status |
|---|---|
| 6 `raw_invoke_*` aggregate fields in `build_fallback_breakdown()` | ✓ lines 422–457 |
| "Raw invoke response shape" markdown table section | ✓ lines 871–879 |

Same 4 fields are not aggregated here either (same subset as staging runner).

---

## 4. Response Object / Content Extraction Analysis

### 4.1 What LocalJsonChatModel.invoke() returns

```
LocalJsonChatModel.invoke(input)
  → self._llm.invoke(input)   # self._llm is ChatOllama(format="json")
  → AIMessage(
        content: str,                  # JSON string (or empty string)
        response_metadata: dict,       # done_reason, done, model, total_duration, ...
        usage_metadata: dict | None,   # input_tokens, output_tokens (LangChain ≥0.1.4)
    )
```

`LocalJsonChatModel` is a transparent pass-through; it adds no wrapping layer.

### 4.2 Content extraction order (implemented)

```
_extract_llm_message_content_with_shape(message) → (text: str, shape: dict)
```

| Step | Condition | Action | ChatOllama path? |
|---|---|---|---|
| 1 | `content_value` is `str` | `text = content_value` | **YES — normal path** |
| 2 | `content_value` is `list` | `text = _text_from_content_blocks(content_value)` | unlikely (multimodal) |
| 3a | `message.message.content` is `str` | `text = nested_content` | unlikely (wrapped) |
| 3b | `message.message.content` is `list` | `text = _text_from_content_blocks(...)` | unlikely |
| 4 | `message.text()` callable | `text = message.text()` | unlikely |
| 5 | none of above | `text = ""` | fallback |

For ChatOllama/Gemma4, **Step 1 is always taken**. Steps 2–5 serve future-proofing and multimodal scenarios.

### 4.3 What _text_from_content_blocks does

Iterates each block:
- `str` block → appended directly
- `dict` block with `"text"` key → `block["text"]` appended
- other → skipped (no exception)

Concatenated without separator. Empty list → `""`.

### 4.4 Shape fields populated

```python
shape = {
    "raw_invoke_response_class": type(message).__name__,   # "AIMessage"
    "raw_invoke_content_present": content_value is not None,
    "raw_invoke_content_type": type(content_value).__name__,  # "str", "list", "missing"
    "raw_invoke_content_length_bucket": _content_length_bucket(text),
    "raw_invoke_content_stripped_empty": not bool(text.strip()),
    "raw_invoke_has_response_metadata": bool(metadata),
    "raw_invoke_finish_reason": _safe_metadata_category(
        metadata.get("finish_reason") or metadata.get("done_reason")
    ),
    "raw_invoke_done_reason": _safe_metadata_category(
        metadata.get("done_reason") or metadata.get("done")
    ),
    "raw_invoke_error_category": "present" if metadata.get("error") else None,
    "raw_invoke_usage_present": usage_metadata is not None,
}
```

### 4.5 _content_length_bucket thresholds

| Bucket | Condition |
|---|---|
| `"empty"` | `len == 0` |
| `"short_lt_100"` | `0 < len < 100` |
| `"medium_lt_1000"` | `100 ≤ len < 1000` |
| `"long_gte_1000"` | `len ≥ 1000` |

### 4.6 _safe_metadata_category sanitization

Allows alphanumeric + `_.:-` up to 64 chars. Longer or non-safe strings → `"present"`. `None`/empty → `None`.  
**No raw metadata text is stored.**

### 4.7 Raw content storage — NONE

`raw_invoke_*` shape fields contain only:
- Class name (structural, not content)
- Booleans (present/absent)
- Enum-like category strings (length bucket, sanitized reason tokens)
- No character of model output is stored

---

## 5. Telemetry / Report Field Verification

### 5.1 Per-request telemetry (all 10 fields)

| Field | Pipeline variable | In payload dict | In `observe_routine_quality` | In `SAFE_DEV_LOG_FIELDS` |
|---|---|---|---|---|
| `raw_invoke_response_class` | ✓ | ✓ | ✓ | ✓ |
| `raw_invoke_content_present` | ✓ | ✓ | ✓ | ✓ |
| `raw_invoke_content_type` | ✓ | ✓ | ✓ | ✓ |
| `raw_invoke_content_length_bucket` | ✓ | ✓ | ✓ | ✓ |
| `raw_invoke_content_stripped_empty` | ✓ | ✓ | ✓ | ✓ |
| `raw_invoke_has_response_metadata` | ✓ | ✓ | ✓ | ✓ |
| `raw_invoke_finish_reason` | ✓ | ✓ | ✓ | ✓ |
| `raw_invoke_done_reason` | ✓ | ✓ | ✓ | ✓ |
| `raw_invoke_error_category` | ✓ | ✓ | ✓ | ✓ |
| `raw_invoke_usage_present` | ✓ | ✓ | ✓ | ✓ |

### 5.2 Suite report aggregate (6 of 10)

| Field | Evaluation suite | Staging runner compare |
|---|---|---|
| `raw_invoke_response_class_counts_a/b` | ✓ | ✓ |
| `raw_invoke_content_present_count_a/b` | ✓ | ✓ |
| `raw_invoke_stripped_empty_count_a/b` | ✓ | ✓ |
| `raw_invoke_finish_reason_counts_a/b` | ✓ | ✓ |
| `raw_invoke_done_reason_counts_a/b` | ✓ | ✓ |
| `raw_invoke_error_category_counts_a/b` | ✓ | ✓ |
| `raw_invoke_content_type_counts_a/b` | **NOT aggregated** | **NOT aggregated** |
| `raw_invoke_content_length_bucket_counts_a/b` | **NOT aggregated** | **NOT aggregated** |
| `raw_invoke_has_response_metadata_count_a/b` | **NOT aggregated** | **NOT aggregated** |
| `raw_invoke_usage_present_count_a/b` | **NOT aggregated** | **NOT aggregated** |

**Assessment:** 4 missing aggregations are non-blocking for smoke. The 6 aggregated fields are sufficient to determine whether content was present and empty, which is the key diagnostic question.  
`content_type` and `content_length_bucket` are per-event and can be analyzed manually from the group_a/group_b JSON if needed.

### 5.3 Staging report markdown

`_markdown()` in `run_feedback_ranking_staging_test.py` does **not** include `raw_invoke_*` fields. The JSON output (`cmd_compare` → compare JSON) includes all 6 aggregated fields, so this is a cosmetic gap only.

---

## 6. Privacy Review

### 6.1 What is stored (safe)

- `raw_invoke_response_class`: Python type name (`"AIMessage"`)
- `raw_invoke_content_present`: boolean
- `raw_invoke_content_type`: Python type name (`"str"`, `"list"`, `"missing"`)
- `raw_invoke_content_length_bucket`: enum string (`"empty"`, `"short_lt_100"`, `"medium_lt_1000"`, `"long_gte_1000"`)
- `raw_invoke_content_stripped_empty`: boolean
- `raw_invoke_has_response_metadata`: boolean
- `raw_invoke_finish_reason`: sanitized token ≤64 chars, alphanumeric/`_.:-` only, or `"present"` if complex, or `None`
- `raw_invoke_done_reason`: same sanitization
- `raw_invoke_error_category`: `"present"` or `None` (not the error text)
- `raw_invoke_usage_present`: boolean

### 6.2 What is NOT stored (confirmed absent)

| Sensitive item | Stored? |
|---|---|
| Raw LLM output text | NO |
| Raw model response content | NO |
| Full response_metadata dict | NO — only boolean presence and 2 sanitized tokens |
| user_note | NO |
| Prompt/candidate_payload text | NO |
| API key or credential | NO |
| Full exception message | NO |

`_safe_metadata_category()` truncates any unsafe value to `"present"`, so long provider-specific strings (e.g., model path, error detail) are not stored.

### 6.3 Known false positive telemetry (out of scope for P3-6h)

When `OutputParserException("local raw JSON invoke failed")` is raised, the repair path's `extract_raw_llm_output_from_exception()` picks up `"local raw JSON invoke failed"` from `exc.args` and sets `raw_output_recovery_succeeded=True`, `raw_output_recovery_source="exception_args"`. This causes misleading telemetry (P3-6g showed `exception_args: 6/8`). **No user data is exposed**, but the telemetry indicates false recovery. This is a known issue, not in scope for P3-6h.

---

## 7. Test Coverage Review

### 7.1 Tests in test_local_raw_json_invoke.py (17 total, 10 new)

| Test | What it covers | New? |
|---|---|---|
| `test_flag_off_uses_existing_structured_output_path` | `ENABLE_LOCAL_RAW_JSON_INVOKE` off → `_should_use` returns False | existing |
| `test_non_local_provider_does_not_use_raw_path_even_when_flag_true` | Gemini provider + flag=true → False | existing |
| `test_local_provider_with_flag_uses_raw_path` | local + flag=true → True | existing |
| `test_raw_invoke_valid_json_succeeds` | valid JSON → LLMRoutineOutput success | existing |
| `test_raw_invoke_fenced_and_prose_json_uses_existing_normalizer` | fence-wrapped JSON normalized | existing |
| `test_raw_invoke_malformed_json_preserves_fallback_failure_path` | malformed JSON → SchemaRepairError | existing |
| `test_raw_content_not_stored_in_safe_local_invoke_metadata` | metadata dict has no model text | existing |
| `test_response_content_str_extracts_shape` | str content → shape fields correct, finish_reason, usage | **new** |
| `test_response_content_whitespace_records_stripped_empty` | whitespace-only content → stripped_empty=True | **new** |
| `test_response_content_list_text_blocks_extracts_text` | list content with dict text block → text extracted | **new** |
| `test_response_message_content_fallback_extracts_text` | nested message.content fallback → text extracted | **new** |
| `test_response_metadata_done_reason_and_error_are_categorized` | done_reason safe token, error→"present", no raw text in shape | **new** |
| `test_invoke_with_shape_returns_output_and_shape` | `_invoke_local_raw_json_output_with_shape` returns (output, shape) | **new** |
| `test_content_length_bucket_boundaries` | all 4 bucket boundaries exact | **new** |
| `test_response_missing_metadata_and_usage_records_false` | no response_metadata → has_metadata=False, finish_reason=None | **new** |
| `test_response_content_none_records_missing_and_empty` | content=None → present=False, type="missing", bucket="empty" | **new** |
| `test_response_content_list_str_blocks_concatenated` | list of str blocks → concatenated text | **new** |

### 7.2 Gaps

| Uncovered case | Risk |
|---|---|
| `response.text()` method fallback (Step 4) | Low — not a ChatOllama response pattern |
| `_safe_metadata_category` with >64 char string → `"present"` | Low — tested implicitly via error text in test_response_metadata_done_reason |
| Shape propagation from SchemaRepairError in pipeline integration | Medium — pipeline unit test would confirm `raw_invoke_*` vars updated on failure |

These gaps do not block P3-6h smoke.

### 7.3 Tests in test_fallback_reason_telemetry.py (confirmed)

`test_payload_contains_new_fallback_fields` (lines 161–234) explicitly verifies all 10 `raw_invoke_*` params are accepted and returned in payload. Passes.

### 7.4 Tests in test_gemma4_evaluation_suite.py (confirmed)

`test_suite_report_includes_parse_and_repair_failure_breakdown` asserts on all 6 aggregate `raw_invoke_*` fields. Passes. `_event()` helper includes all 10 per-event defaults.

---

## 8. Focused Smoke Commands

### Prerequisites

Server must be running with:
```bash
ENABLE_RULE_CRITIC_TELEMETRY=true
LLM_PROVIDER=local
LOCAL_LLM_MODEL=gemma4:latest
OLLAMA_BASE_URL=http://127.0.0.1:11434
ROUTINE_CANDIDATE_POOL_SIZE=12
```

Group A control: `ENABLE_LOCAL_RAW_JSON_INVOKE=false`, `ENABLE_FEEDBACK_AWARE_RANKING=false`  
Group B treatment: `ENABLE_LOCAL_RAW_JSON_INVOKE=true`, `ENABLE_FEEDBACK_AWARE_RANKING=true`

### Step 1 — Preflight (with seed reset)

```bash
python scripts/run_gemma4_evaluation_suite.py preflight \
  --output-dir tests/evaluation/.artifacts/gemma4-suite-focused-p3-6h \
  --run-id gemma4-suite-p3-6h
```

### Step 2 — Run A (ENABLE_LOCAL_RAW_JSON_INVOKE=false)

Restart server with `ENABLE_LOCAL_RAW_JSON_INVOKE=false`, `ENABLE_FEEDBACK_AWARE_RANKING=false`, then:

```bash
python scripts/run_gemma4_evaluation_suite.py run-a \
  --output-dir tests/evaluation/.artifacts/gemma4-suite-focused-p3-6h \
  --run-id gemma4-suite-p3-6h \
  --repeats 1 \
  --request-timeout 240 \
  --scenario-ids normal_hypertrophy_upper,strength_lower,strength_lower_fb,fatloss_limited_equipment,mixed_target_muscles,long_duration_60m,low_readiness,shoulder_pain_present
```

### Step 3 — Run B (ENABLE_LOCAL_RAW_JSON_INVOKE=true)

Restart server with `ENABLE_LOCAL_RAW_JSON_INVOKE=true`, `ENABLE_FEEDBACK_AWARE_RANKING=true`, then:

```bash
python scripts/run_gemma4_evaluation_suite.py run-b \
  --output-dir tests/evaluation/.artifacts/gemma4-suite-focused-p3-6h \
  --run-id gemma4-suite-p3-6h \
  --repeats 1 \
  --request-timeout 240 \
  --scenario-ids normal_hypertrophy_upper,strength_lower,strength_lower_fb,fatloss_limited_equipment,mixed_target_muscles,long_duration_60m,low_readiness,shoulder_pain_present
```

### Step 4 — Compare

```bash
python scripts/run_gemma4_evaluation_suite.py compare \
  --output-dir tests/evaluation/.artifacts/gemma4-suite-focused-p3-6h \
  --run-id gemma4-suite-p3-6h \
  --markdown tests/evaluation/.artifacts/gemma4-suite-focused-p3-6h/gemma4-suite-report.md
```

### Step 5 — Extract diagnostic fields

```bash
python -c "
import json, pathlib
r = json.loads(pathlib.Path('tests/evaluation/.artifacts/gemma4-suite-focused-p3-6h/gemma4-suite-report.json').read_text())
keys = [
  'recommendation',
  'request_count_a', 'request_count_b',
  'fallback_reason_counts_a', 'fallback_reason_counts_b',
  'llm_error_type_counts_a', 'llm_error_type_counts_b',
  'parse_failure_subtype_counts_a', 'parse_failure_subtype_counts_b',
  'json_decode_error_category_counts_a', 'json_decode_error_category_counts_b',
  'local_raw_json_invoke_used_count_a', 'local_raw_json_invoke_used_count_b',
  'local_raw_json_invoke_succeeded_count_a', 'local_raw_json_invoke_succeeded_count_b',
  'local_raw_json_invoke_failed_reason_counts_a', 'local_raw_json_invoke_failed_reason_counts_b',
  'structured_output_bypassed_count_a', 'structured_output_bypassed_count_b',
  'raw_invoke_response_class_counts_a', 'raw_invoke_response_class_counts_b',
  'raw_invoke_content_present_count_a', 'raw_invoke_content_present_count_b',
  'raw_invoke_stripped_empty_count_a', 'raw_invoke_stripped_empty_count_b',
  'raw_invoke_finish_reason_counts_a', 'raw_invoke_finish_reason_counts_b',
  'raw_invoke_done_reason_counts_a', 'raw_invoke_done_reason_counts_b',
  'raw_invoke_error_category_counts_a', 'raw_invoke_error_category_counts_b',
  'schema_repair_attempted_count_a', 'schema_repair_attempted_count_b',
  'schema_repair_succeeded_count_a', 'schema_repair_succeeded_count_b',
  'privacy_leak_detected',
]
print(json.dumps({k: r.get(k) for k in keys}, indent=2))
"
```

---

## 9. Result Interpretation Matrix

### Case A — Content extraction fix helped

**Signal:** `raw_invoke_content_present_count_b > raw_invoke_stripped_empty_count_b`, AND `local_raw_json_invoke_succeeded_count_b > 0`, AND fallback rate lower than P3-6g Group B (100%)

**Meaning:** Gemma4 was returning non-empty content that the old extractor was missing (list or nested format).  
**Next action:** Full suite run (`--repeats 5`, all scenarios). Content extraction fix was the root cause.

---

### Case B — Content present but JSON still malformed

**Signal:** `raw_invoke_content_present_count_b` high, `raw_invoke_stripped_empty_count_b` low, but `parse_failure_subtype_counts_b` still shows `json_decode_error` or `json_decode_expect_value`

**Meaning:** Gemma4 is returning non-empty, non-JSON text (markdown prose, partial JSON, preamble). Content extraction is not the bottleneck.  
**Next action:** Investigate prompt minimal JSON hint, stricter `format=json` enforcement, or schema shape repair (next task P3-7x).

---

### Case C — Content genuinely empty

**Signal:** `raw_invoke_content_present_count_b` high but `raw_invoke_stripped_empty_count_b` also high (≥50% of requests), AND `raw_invoke_content_length_bucket` (per-event) shows `"empty"` or `"short_lt_100"`

**Meaning:** ChatOllama/Gemma4 is returning `content=""` or near-empty strings on most requests. Not a content extraction problem.  
**Next action:** Test Ollama directly via `curl http://localhost:11434/api/generate -d '{"model":"gemma4","format":"json","prompt":"..."}'` to confirm Ollama-level behavior. Then evaluate payload size / prompt truncation.

---

### Case D — Provider/runtime error

**Signal:** `raw_invoke_error_category_counts_b` shows `"present"` for a significant fraction, AND/OR `raw_invoke_finish_reason_counts_b` does not contain `"stop"` but shows unexpected tokens

**Meaning:** Ollama/Gemma4 runtime is returning error metadata. GPU OOM, model loading issue, or context length overflow.  
**Next action:** Check Ollama server logs. Compare `response_metadata.done_reason` across scenarios.

---

### Case E — Group B worse than Group A

**Signal:** `raw_invoke_stripped_empty_count_b > raw_invoke_stripped_empty_count_a` despite Group A using `structured_output` path

**Meaning:** Feedback ranking is passing a larger/different candidate payload, or the `ENABLE_FEEDBACK_AWARE_RANKING=true` probe query adds latency that interferes with Gemma4 generation.  
**Next action:** Compare `candidate_payload_char_count` distribution A vs B. Evaluate if feedback payload size is a factor.

---

### Case F — No change vs P3-6g

**Signal:** Results are statistically identical to P3-6g: `succeeded=0/8`, `failed_reason: empty_response: 8`

**Meaning:** The content extraction fix did not help because the actual content was always `str` (Step 1 path) and was always empty. P3-6h confirmed that the bug is **not** extraction-related.  
**Next action:** Treat Case C as confirmed. Move to Ollama-level diagnosis.

---

## 10. Gaps and Recommended Checks Before Smoke

### Non-blocking gaps

1. **Staging report markdown** (`_markdown()` in `run_feedback_ranking_staging_test.py`) does not render `raw_invoke_*` fields. JSON output is complete. If markdown readability matters, add 6 lines. Not needed for P3-6h smoke.

2. **4 shape fields not aggregated** in compare outputs: `content_type`, `content_length_bucket`, `has_response_metadata`, `usage_present`. Per-event data is complete. Manual extraction from `group_a.json`/`group_b.json` is possible if needed.

3. **`exception_args` recovery false positive** (out of scope for P3-6h): `raw_output_recovery_source=exception_args` will still appear in telemetry for the repair path. This is cosmetically misleading but does not cause wrong behavior.

### Confirmed before smoke

- 549 unit tests passing (confirmed by test run in same session)
- No server restart needed for test run
- `ENABLE_LOCAL_RAW_JSON_INVOKE` env var correctly controls the code path
- Shape dict initialized to safe defaults when local raw invoke is not used (all `raw_invoke_*` vars initialized to `None`/`False` in `generate_smart_routine`)
- Shape fields set only when `use_local_raw_json_invoke is True` in the pipeline branch

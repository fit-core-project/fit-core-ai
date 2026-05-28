# P1.7 Rule Critic Telemetry Integration Spec

## 1. 설계 요약

P1.7은 AI 루틴 생성 pipeline에 observe-only quality telemetry를 추가하기 위한 구현 스펙이다. 구현 목표는 dev/staging에서 `ROUTINE_CANDIDATE_POOL_SIZE=18` 운영 검증에 필요한 품질 지표를 남기되, production behavior와 response contract는 변경하지 않는 것이다.

핵심 정책:

- 기본값은 `ENABLE_RULE_CRITIC_TELEMETRY=false`이다.
- flag off일 때는 현재 `generate_smart_routine()` 동작, latency, import surface를 최대한 그대로 유지한다.
- flag on일 때만 Rule Critic을 실행하고 structured telemetry event를 emit한다.
- Rule Critic 결과는 기록만 한다. `should_rebuild`, `should_fallback`, `grade=FAIL`은 response를 변경하거나 rebuild/fallback을 트리거하지 않는다.
- `user_note`, feedback free text, raw prompt, raw LLM output, medical free text는 telemetry에 기록하지 않는다.
- feedback API 저장 로직과 ranking score 로직은 변경하지 않는다.

## 2. Integration Point 분석

### 현재 pipeline에서 이미 확보 가능한 값

`engines/routine_pipeline.py::generate_smart_routine()` 안에서 다음 값이 이미 존재한다.

| 값 | 현재 위치 | telemetry 사용 |
| --- | --- | --- |
| `candidate_pool_size` | candidate ranking 직전 | `candidate_pool_size` |
| `ranked_candidates` | `score_candidate_exercises()` 결과 | `candidate_count`, Rule Critic candidates |
| candidate prompt string | `format_candidates_for_prompt(ranked_candidates)` | `candidate_payload_char_count`, token estimate |
| `generation_temp` | LLM 호출 직전 | `generation_temperature` |
| dynamic temperature flag | `temperature_policy` | `dynamic_temperature_enabled` |
| `max_total_sets` | `_calculate_max_total_sets()` 결과 | `build_eval_context(..., max_working_sets=...)` |
| `db_target_muscles` | target mapping 이후 | `build_eval_context()` |
| `recent_sets` | function arg | `build_eval_context()` |
| final response | success/fallback return 직전 | routine stats, repair/fallback stats |

### 권장 삽입 지점

V1에서는 response가 만들어지는 모든 경로를 한 곳에서 감싸기 위해 `generate_smart_routine()` 내부에 small wrapper를 둔다.

권장 구조:

```python
def _emit_quality_telemetry_if_enabled(
    response: RoutineDraftResponse,
    *,
    req: RoutineRequest,
    ranked_candidates: list[dict],
    db_target_muscles: list[str],
    recent_sets: list[RecentSetRecord] | None,
    max_total_sets: int,
    candidate_pool_size: int,
    candidate_payload: str,
    generation_temperature: float,
    llm_latency_ms: int | None,
    schema_repair_latency_ms: int | None,
) -> None:
    ...
```

그리고 현재 `return _validate_or_fallback(...)`, schema repair 경로의 `return _validate_or_fallback(...)`, 마지막 `_fallback(...)` 반환을 다음 패턴으로 바꾼다.

```python
response = _validate_or_fallback(...)
_emit_quality_telemetry_if_enabled(response, ...)
return response
```

fallback branch도 동일하게:

```python
response = _fallback(...)
_emit_quality_telemetry_if_enabled(response, ...)
return response
```

주의:

- `_validate_or_fallback()`, `_deterministic_or_fallback()`, fallback generator의 response shape는 변경하지 않는다.
- Rule Critic 결과로 response를 mutate하지 않는다.
- telemetry emit 실패는 반드시 swallow한다.

## 3. Telemetry Payload Schema

Event name: `routine_generation_quality`

V1 payload는 JSON-serializable dict로 제한한다.

```json
{
  "event": "routine_generation_quality",
  "routine_draft_id": "string",
  "candidate_pool_size": 18,
  "candidate_count": 18,
  "candidate_payload_char_count": 12345,
  "approx_candidate_payload_tokens": 3087,
  "dynamic_temperature_enabled": false,
  "generation_temperature": 0.0,
  "critic_score": 87,
  "critic_grade": "PASS",
  "critic_warning_count": 1,
  "hard_violation_count": 0,
  "repair_count": 0,
  "fallback_used": false,
  "llm_generation_latency_ms": 840,
  "schema_repair_latency_ms": null,
  "exercise_count": 6,
  "total_estimated_time": 47,
  "target_duration_min": 45,
  "created_at": "2026-05-28T12:34:56.789Z"
}
```

### Field policy

| Field | Source | Required | Notes |
| --- | --- | --- | --- |
| `event` | constant | yes | `routine_generation_quality` |
| `routine_draft_id` | `response.routine_draft_id` | yes | feedback join key |
| `candidate_pool_size` | `candidate_pool_size` | yes | configured requested pool size |
| `candidate_count` | `len(ranked_candidates)` | yes | actual post-filter count |
| `candidate_payload_char_count` | `len(candidate_payload)` | yes | candidate section only |
| `approx_candidate_payload_tokens` | `ceil(char_count / 4)` | yes | approximate only |
| `dynamic_temperature_enabled` | temperature policy resolver | yes | see section 4 |
| `generation_temperature` | `generation_temp` | yes | value passed to `get_llm()` |
| `critic_score` | `RuleCriticResult.score` | flag-on yes | `None` only if critic failed internally |
| `critic_grade` | `RuleCriticResult.grade` | flag-on yes | `PASS`, `WARN`, `FAIL` |
| `critic_warning_count` | `len(result.warnings)` | yes | warning text is not logged |
| `hard_violation_count` | `len(result.hard_violations)` | yes | violation text is not logged in telemetry |
| `repair_count` | response warnings | yes | strict formula below |
| `fallback_used` | `response.is_fallback` or `generation_status == "fallback"` | yes | `failed` response is also not success |
| `llm_generation_latency_ms` | around first LLM invoke | yes nullable | local monotonic timer |
| `schema_repair_latency_ms` | around normalize/repair branch | yes nullable | only schema repair path |
| `exercise_count` | `len(response.routine_blocks)` | yes | final renderable response |
| `total_estimated_time` | `response.total_estimated_time` | yes | final response value |
| `target_duration_min` | `req.time_available_min` | yes | numeric only |
| `created_at` | UTC iso timestamp | yes | use timezone-aware UTC |

### Excluded fields

Do not include:

- `req.user_note`
- raw prompt
- rendered full prompt
- raw LLM response
- `response.rationale_summary`
- `exercise_rationale` text
- feedback `user_note`
- pain free text beyond aggregate counts
- Rule Critic warning strings or hard violation strings in V1 telemetry payload

## 4. Feature Flag / Config 정책

Add `engines/routine_telemetry.py` or `engines/telemetry_policy.py`.

Recommended constants:

```python
ENV_RULE_CRITIC_TELEMETRY = "ENABLE_RULE_CRITIC_TELEMETRY"
```

Resolver:

```python
def resolve_rule_critic_telemetry_enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() == "true"

def is_rule_critic_telemetry_enabled() -> bool:
    return resolve_rule_critic_telemetry_enabled(os.getenv(ENV_RULE_CRITIC_TELEMETRY))
```

Policy:

- unset -> `False`
- `"false"` -> `False`
- `"true"` -> `True`
- whitespace/case-insensitive true, e.g. `" TRUE "` -> `True`
- all other values -> `False`

Dynamic temperature enabled 여부는 existing `engines/temperature_policy.py`의 `_DYNAMIC_ENABLED`와 같은 환경 값을 기준으로 기록해야 한다. 구현 시 private module variable 직접 참조를 피하려면 다음 public helper를 추가한다.

```python
def is_dynamic_temperature_enabled() -> bool:
    return _DYNAMIC_ENABLED
```

또는 telemetry module에서 `os.getenv("ENABLE_DYNAMIC_TEMPERATURE", "false")`를 같은 resolver 방식으로 읽는다. 기존 temperature mapping은 변경하지 않는다.

## 5. Rule Critic 연결 방식

현재 `engines/rule_critic.py`는 V1 observe-only 구조를 이미 가진다.

사용 함수:

```python
from engines.rule_critic import build_eval_context, evaluate_routine_quality
```

연결 절차:

1. final `RoutineDraftResponse` 생성 후 telemetry flag 확인
2. flag on이면 `build_eval_context(req, db_target_muscles, recent_sets, max_total_sets)` 호출
3. `repair_count` 계산
4. `fallback_used` 계산
5. `evaluate_routine_quality(response, ranked_candidates, context, repair_count, fallback_used)` 호출
6. 결과를 telemetry payload에 넣고 emit

Required repair count formula:

```python
repair_count = len([
    warning
    for warning in response.warnings
    if "Guard:" in warning and "replaced" in warning
])
```

금지:

- `"Guard:"` 단독 조건으로 repair count 계산 금지
- `"Guard diagnostics"`를 repair count에 포함 금지
- `result.should_rebuild`나 `result.should_fallback`로 pipeline branch 변경 금지
- `critic_grade == "FAIL"`로 fallback 전환 금지

`fallback_used` 권장 계산:

```python
fallback_used = response.is_fallback or response.generation_status == "fallback"
```

`failed` 상태는 fallback은 아니지만 success도 아니므로 별도 field가 필요하면 V2에서 `generation_status`를 추가한다. V1 필수 payload에는 response contract 변경 없이 `fallback_used`만 둔다.

## 6. Privacy Policy

Telemetry는 운영 검증용 aggregate quality signal만 남긴다.

반드시 지킬 정책:

- `user_note`는 payload, stdout, dev log, structured logger에 기록하지 않는다.
- feedback free text는 payload, stdout, dev log, structured logger에 기록하지 않는다.
- raw prompt와 raw LLM output은 telemetry payload에 기록하지 않는다.
- candidate payload는 원문을 기록하지 않고 char/token count만 기록한다.
- rationale 원문은 기록하지 않는다. 필요하면 V2에서 `rationale_missing_count`, `avg_rationale_length` 같은 aggregate만 추가한다.
- pain/medical free text는 기록하지 않는다. `current_pain_area_count` 같은 aggregate도 V1 필수 범위가 아니므로 생략한다.
- Rule Critic warning/hard violation strings는 free text를 포함할 수 있으므로 V1 payload에는 count만 기록한다.

주의: 현재 `install_stdout_capture()`가 stdout을 `dev_log_buffer`에 캡처한다. telemetry emit을 `print(json.dumps(payload))`로 구현할 경우 payload에 free text가 없어야 한다.

## 7. Failure Handling

Telemetry는 generation path를 절대 실패시키면 안 된다.

권장 구현:

```python
def emit_routine_quality_telemetry(payload: dict) -> None:
    try:
        print("[Telemetry] " + json.dumps(payload, ensure_ascii=False, sort_keys=True))
    except Exception:
        pass
```

Pipeline wrapper:

```python
try:
    ...
except Exception as exc:
    print(f"[Telemetry] emit skipped: {type(exc).__name__}")
```

정책:

- telemetry build 실패 -> response 그대로 반환
- Rule Critic evaluate 실패 -> response 그대로 반환
- sink append/print 실패 -> response 그대로 반환
- telemetry failure detail에 request free text를 포함하지 않는다.

## 8. 파일 구조 제안

권장 파일: `engines/routine_telemetry.py`

이름을 `routine_telemetry.py`로 권장하는 이유는 telemetry가 generic app telemetry가 아니라 routine generation quality event에 특화되어 있기 때문이다.

Required functions:

```python
ENV_RULE_CRITIC_TELEMETRY = "ENABLE_RULE_CRITIC_TELEMETRY"

def resolve_rule_critic_telemetry_enabled(value: str | None) -> bool: ...

def is_rule_critic_telemetry_enabled() -> bool: ...

def count_guard_repairs(warnings: list[str]) -> int: ...

def build_routine_quality_telemetry_payload(
    *,
    response: RoutineDraftResponse,
    candidate_pool_size: int,
    ranked_candidates: list[dict],
    candidate_payload: str,
    dynamic_temperature_enabled: bool,
    generation_temperature: float,
    critic_result: RuleCriticResult | None,
    repair_count: int,
    fallback_used: bool,
    target_duration_min: int,
    llm_generation_latency_ms: int | None = None,
    schema_repair_latency_ms: int | None = None,
    created_at: datetime | None = None,
) -> dict: ...

def emit_routine_quality_telemetry(payload: dict) -> None: ...
```

Optional helper:

```python
def observe_routine_generation_quality(...) -> None:
    """Feature-flagged wrapper that builds context, runs critic, builds payload, emits."""
```

Pipeline import strategy:

- If flag resolver is cheap, import `is_rule_critic_telemetry_enabled` at module import is acceptable.
- To minimize flag-off overhead, import Rule Critic only inside the flag-on branch.

Example:

```python
if not is_rule_critic_telemetry_enabled():
    return

from .rule_critic import build_eval_context, evaluate_routine_quality
```

## 9. 테스트 계획

Add `tests/test_routine_telemetry.py`.

Required tests:

1. `resolve_rule_critic_telemetry_enabled(None)` returns `False`
2. resolver accepts `"true"` and `" TRUE "`
3. resolver rejects invalid strings
4. flag off -> pipeline does not call telemetry emit
5. flag on -> pipeline calls telemetry emit once for success response
6. telemetry payload includes `candidate_pool_size`
7. telemetry payload includes `critic_score` and `critic_grade`
8. `count_guard_repairs()` counts only warnings containing both `"Guard:"` and `"replaced"`
9. `count_guard_repairs()` does not count `"Guard diagnostics"` or `"Guard:"` informational warnings
10. payload records `fallback_used=True` for fallback response
11. payload does not contain `user_note`
12. payload does not contain raw rationale text
13. payload records `dynamic_temperature_enabled`
14. payload records `generation_temperature`
15. payload records candidate char count and approximate token count
16. Rule Critic `FAIL` does not mutate response and does not change fallback behavior
17. telemetry emit failure is swallowed and generation still returns original response
18. schema repair latency is nullable on normal path
19. `routine_draft_id` is present in payload
20. existing `tests/test_rule_critic.py` still pass
21. existing `tests/test_candidate_pool_policy.py` still pass
22. full `pytest tests/ -q` remains green

Implementation-specific pipeline tests can follow the existing monkeypatch pattern in `tests/test_candidate_pool_policy.py`:

- patch `engines.routine_pipeline.get_candidate_exercises`
- patch `engines.routine_pipeline.get_llm`
- set `ENABLE_RULE_CRITIC_TELEMETRY=true`
- monkeypatch `engines.routine_telemetry.emit_routine_quality_telemetry` or wrapper function

Validation commands:

```bash
pytest tests/test_routine_telemetry.py -q
pytest tests/test_rule_critic.py -q
pytest tests/test_candidate_pool_policy.py -q
pytest tests/evaluation -q
pytest tests/test_routine_feedback.py -q
pytest tests/test_temperature_policy.py -q
pytest tests/test_prompt_regression.py tests/test_guard.py tests/test_contract.py tests/test_fallback.py -q
pytest tests/ -q
```

## 10. 리스크

| Risk | Impact | Mitigation |
| --- | --- | --- |
| flag-on path accidentally changes response | high | telemetry wrapper returns `None`; response object is never mutated |
| Rule Critic FAIL triggers fallback | high | explicit tests for FAIL observe-only |
| free text leaks through payload | high | payload builder allowlist fields only |
| logging warning strings leaks medical/rationale text | medium | V1 logs only counts, not warning strings |
| flag off still imports/evaluates critic | low/medium | lazy import critic inside enabled branch |
| latency measurement adds complexity | low | use `time.perf_counter()` only around existing calls |
| schema repair latency hard to thread through | medium | V1 nullable; add timing only around existing repair branch |
| fallback paths miss telemetry | medium | wrap every return path in `generate_smart_routine()` |
| dev log stdout sink unavailable in tests | low | emit function is monkeypatchable and swallows failures |

## 11. Codex 구현용 최종 지시문

작업명: P1.7 Rule Critic Telemetry Integration 구현

구현 범위:

1. `engines/routine_telemetry.py`를 추가한다.
2. `ENABLE_RULE_CRITIC_TELEMETRY` flag resolver를 추가한다. 기본값은 false이다.
3. `count_guard_repairs(warnings)`를 추가한다. 반드시 `"Guard:" in warning and "replaced" in warning` 조건만 repair로 센다.
4. `build_routine_quality_telemetry_payload()` 순수 함수를 추가한다.
5. `emit_routine_quality_telemetry(payload)`를 추가한다. V1 sink는 structured JSON stdout/dev log로 충분하다.
6. telemetry emit 실패는 generation 실패로 전파하지 않는다.
7. `engines/routine_pipeline.py`에서 final response 생성 후 flag-on일 때만 Rule Critic을 실행하고 telemetry를 emit한다.
8. Rule Critic은 `build_eval_context(req, db_target_muscles, recent_sets, max_total_sets)`와 `evaluate_routine_quality(response, ranked_candidates, context, repair_count, fallback_used)`로 연결한다.
9. `candidate_payload = format_candidates_for_prompt(ranked_candidates)`를 한 번만 만들고 prompt와 telemetry count에 재사용한다.
10. LLM first generation latency를 `time.perf_counter()`로 측정한다.
11. schema repair latency는 repair branch에서만 측정하고 normal path는 `None`으로 둔다.
12. Dynamic Temperature 로직과 flag 기본값은 변경하지 않는다.
13. Candidate ranking score, `candidate_ranker.py`, Rule Critic scoring formula, fallback/repair trigger는 변경하지 않는다.
14. response schema와 public API contract는 변경하지 않는다.
15. telemetry payload에는 `user_note`, raw prompt, raw LLM output, rationale 원문, feedback free text를 절대 넣지 않는다.

필수 payload fields:

- `event`
- `routine_draft_id`
- `candidate_pool_size`
- `candidate_count`
- `candidate_payload_char_count`
- `approx_candidate_payload_tokens`
- `dynamic_temperature_enabled`
- `generation_temperature`
- `critic_score`
- `critic_grade`
- `critic_warning_count`
- `hard_violation_count`
- `repair_count`
- `fallback_used`
- `llm_generation_latency_ms`
- `schema_repair_latency_ms`
- `exercise_count`
- `total_estimated_time`
- `target_duration_min`
- `created_at`

필수 테스트:

- flag off no emit
- flag on emit
- payload schema field assertions
- repair count strict formula
- fallback_used recording
- privacy allowlist test: `user_note` and rationale text absent
- dynamic temperature enabled/value recorded
- candidate payload char/token counts recorded
- Rule Critic FAIL observe-only
- emit failure swallowed
- full regression green

완료 조건:

- `pytest tests/test_routine_telemetry.py -q` 통과
- `pytest tests/test_rule_critic.py -q` 통과
- `pytest tests/test_candidate_pool_policy.py -q` 통과
- `pytest tests/evaluation -q` 통과
- `pytest tests/test_routine_feedback.py -q` 통과
- `pytest tests/test_temperature_policy.py -q` 통과
- `pytest tests/test_prompt_regression.py tests/test_guard.py tests/test_contract.py tests/test_fallback.py -q` 통과
- `pytest tests/ -q` 통과
- production default behavior 변경 없음

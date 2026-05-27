# Rule Critic Score — 설계 스펙

**버전**: 1.0  
**대상 레포**: `fit-core-ai`  
**현재 테스트 상태**: 133 passed  
**관련 P-task**: P1-2  
**전제**: P1-1 Offline Evaluation Harness 구현 완료

---

## 1. 설계 요약

`Rule Critic Score`는 AI 루틴 생성 파이프라인에서 **최종 `RoutineDraftResponse`의 품질을 runtime에 결정론적으로 평가**하는 모듈이다.

**핵심 목적**:
- Guard(hard constraint 방어)가 통과한 루틴의 **soft quality**를 정량화
- 낮은 품질 루틴을 production에서 감지·기록하고, 추후 rebuild/fallback 트리거의 기반 마련
- `tests/evaluation/helpers.py`의 metric 로직을 production 코드로 승격시켜 single source of truth 확보

**V1 범위**: 계산 + 로그 기록만. Rebuild/fallback 트리거는 feature flag로 opt-in.

---

## 2. post_validation_guard와 Rule Critic 역할 경계

```
[LLM Output]
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  post_validation_guard  (engines/post_validation_guard.py)   │
│  + validate_and_repair_routine_output (engines/llm_parser.py)│
│                                                              │
│  담당: HARD CONSTRAINT 탐지 및 repair                         │
│  - exercise_id ∉ candidates → repair or fallback            │
│  - unavailable equipment 사용 → repair or fallback           │
│  - pain trigger 운동 → repair or fallback                    │
│  - DOMS level 3 근육 → fallback                              │
│  - set/reps/rest 유효성 (스키마 레벨)                          │
│  - working set cap 초과 → set 감소 또는 fallback              │
│  - time budget 초과 → trim 또는 fallback                     │
│                                                              │
│  출력: RoutineDraftResponse (success | fallback | failed)    │
└─────────────────────────────────────────────────────────────┘
     │ (guard 통과 후)
     ▼
┌─────────────────────────────────────────────────────────────┐
│  Rule Critic Score  (engines/rule_critic.py)  ◄── NEW       │
│                                                              │
│  담당: SOFT QUALITY 평가                                      │
│  - hard constraint 누락 감시 (guard 우회 감지, sanity check)  │
│  - target muscle coverage                                    │
│  - time budget fit                                           │
│  - recent exercise 반복률 (diversity)                         │
│  - compound/isolation balance                                │
│  - rationale 비어있음 여부                                     │
│  - suspicious fallback/repair 패턴                           │
│                                                              │
│  출력: RuleCriticResult (score, grade, warnings, rebuild?)   │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
[최종 RoutineDraftResponse 반환 / 로그 기록]
```

### 역할 분리 원칙

| 구분 | Guard | Rule Critic |
|------|-------|-------------|
| 실행 위치 | `validate_and_repair_routine_output()` 내부 | `_finalize_success()` 이후 |
| 입력 | `LLMRoutineOutput` | `RoutineDraftResponse` |
| 위반 시 동작 | repair 시도 → 불가 시 fallback | V1: log only / V2: rebuild opt-in |
| hard constraint | 주 담당 | sanity check (누락 감시) |
| soft quality | 미담당 | 주 담당 |
| DB 접근 | 없음 | 없음 |
| latency | < 5ms | < 5ms |

---

## 3. RuleCriticResult 모델

```python
# engines/rule_critic.py

from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class RuleCriticResult:
    score: int                          # 0~100
    grade: str                          # "PASS" | "WARN" | "FAIL"
    hard_violations: List[str]          # guard 누락 감지 시 1건 이상; 정상이면 []
    warnings: List[str]                 # soft quality 경고 ("FAIL " prefix = 점수 0 영역)
    metric_scores: Dict[str, int]       # {"coverage": 25, "time": 20, ...}
    should_rebuild: bool                # V1: 항상 False; V2: grade == "FAIL" 시 True
    should_fallback: bool               # V1: 항상 False; V2: rebuild 실패 후 True
    notes: List[str] = field(default_factory=list)
```

### 필드 설명

| 필드 | 설명 |
|------|------|
| `score` | hard_violations > 0이면 0; 아니면 metric_scores 합산 |
| `grade` | PASS ≥ 80 / WARN 65~79 / FAIL < 65 또는 hard_violation |
| `hard_violations` | 이미 guard가 처리했어야 하는 항목이 최종 결과에 남아있으면 기록 |
| `warnings` | "FAIL " prefix = score 0 처리된 metric; 없으면 info 경고 |
| `should_rebuild` | V1: `False` 고정; V2: `CRITIC_REBUILD_ENABLED=true` 시 활성화 |
| `should_fallback` | V1: `False` 고정; V2: rebuild도 FAIL 시 활성화 |

---

## 4. 함수 시그니처

### 메인 함수

```python
def evaluate_routine_quality(
    response: RoutineDraftResponse,
    candidates: List[dict],
    context: "RoutineEvalContext",
    repair_count: int = 0,
    fallback_used: bool = False,
) -> RuleCriticResult:
    """
    Guard 통과 후 최종 RoutineDraftResponse의 soft quality를 평가한다.

    Args:
        response: pipeline이 반환한 최종 RoutineDraftResponse
        candidates: score_candidate_exercises() 결과 (scored 상태)
        context: 요청 컨텍스트 (goal, duration, equipment, pain, doms, recent_ids)
        repair_count: guard repair 발생 횟수 (validate_and_repair_routine_output 내부 violations 카운트)
        fallback_used: response.is_fallback 또는 generation_status == "fallback"

    Returns:
        RuleCriticResult
    """
```

### 컨텍스트 모델

```python
@dataclass
class RoutineEvalContext:
    goal: str                           # "hypertrophy" | "strength" | ...
    time_available_min: int             # req.time_available_min
    target_muscles: List[str]           # db_target_muscles (split_label → muscles)
    blocked_equipment: List[str]        # req.equipment (차단 목록)
    pain_areas: List[PainAreaEntry]     # req.pain_areas
    doms_db: Dict[str, int]             # req.doms_data
    recent_exercise_ids: List[str]      # recent_sets에서 exercise_id 추출
    readiness_level: str = "normal"     # req.readiness_level
    max_working_sets: Optional[int] = None  # None이면 _calculate_max_total_sets() 사용
    overrides: dict = field(default_factory=dict)  # harness: scenario["expected"] 플래그 주입; pipeline: 항상 {}
```

### 헬퍼 함수 (public)

```python
def build_eval_context(
    req: RoutineRequest,
    db_target_muscles: List[str],
    recent_sets: Optional[List[RecentSetRecord]] = None,
    max_working_sets: Optional[int] = None,
) -> RoutineEvalContext:
    """RoutineRequest → RoutineEvalContext 변환 헬퍼 (pipeline에서 호출)."""
```

---

## 5. Metric별 계산 방식

### 공통 전제

- `blocks`: `[block.model_dump(by_alias=False) for block in response.routine_blocks]`
- `candidates_by_id`: `{c["id"]: c for c in candidates}`
- `movement_type` 조회: `candidates_by_id[block["exercise_id"]]["movement_type"]` (없으면 "UNKNOWN")
- `primary_muscles` 조회: `block["primary_muscles"]` (없으면 candidates에서 fallback)

---

### A. Hard Constraint Sanity (gate — 점수 없음)

Guard가 이미 처리했어야 하는 항목이 최종 결과에 남아있는지 재확인한다.  
**이 체크가 위반을 감지하면 guard 누락 버그이므로 즉시 로그에 `[CRITIC][GUARD_BYPASS]` 기록.**

```python
def _check_hard_constraints_sanity(
    blocks: List[dict],
    candidates_by_id: Dict[str, dict],
    context: RoutineEvalContext,
    max_working_sets: int,
) -> List[str]:
    violations = []

    for block in blocks:
        eid = block.get("exercise_id", "")
        candidate = candidates_by_id.get(eid)

        # 1. hallucination
        if candidate is None:
            violations.append(f"[GUARD_BYPASS] exercise_id '{eid}' not in candidates")
            continue

        # 2. equipment
        equip_tokens = {t.strip().upper() for t in str(candidate.get("equipment_req") or "").split(",") if t.strip()} - {"BODYWEIGHT"}
        blocked = {e.strip().upper() for e in context.blocked_equipment}
        if equip_tokens & blocked:
            violations.append(f"[GUARD_BYPASS] blocked equipment in {eid}: {equip_tokens & blocked}")

        # 3. pain
        pain_tokens = {p.body_part.lower() for p in context.pain_areas if p.body_part}
        pain_triggers = str(candidate.get("pain_triggers") or "").lower()
        matched = sorted(t for t in pain_tokens if t and t in pain_triggers)
        if matched:
            violations.append(f"[GUARD_BYPASS] pain trigger in {eid}: {matched}")

        # 4. DOMS level 3
        for muscle in (block.get("primary_muscles") or []):
            if int(context.doms_db.get(str(muscle), 0) or 0) >= 3:
                violations.append(f"[GUARD_BYPASS] DOMS level 3 muscle in {eid}: {muscle}")

    # 5. working set cap
    total_sets = sum(len(block.get("prescription") or []) for block in blocks)
    if total_sets > max_working_sets:
        violations.append(f"[GUARD_BYPASS] set cap exceeded: actual={total_sets}, max={max_working_sets}")

    return violations
```

---

### B. Coverage Score (0~25점)

```python
covered = {muscle for block in blocks for muscle in block.get("primary_muscles", [])} & set(context.target_muscles)
ratio = len(covered) / len(context.target_muscles) if context.target_muscles else 1.0
score = round(ratio * 25)
```

| ratio | 점수 | 판정 |
|-------|------|------|
| 1.0 | 25 | — |
| 0.7~1.0 | 18~24 | warning |
| < 0.7 | < 18 | "FAIL " prefix (후보 충분 시) |

> **예외 조건** (후보 부족, low readiness, fallback_used): < 0.7도 warning으로 처리.  
> 예외 여부는 `len(candidates) <= 2 or context.readiness_level == "low" or fallback_used`로 판단.

---

### C. Time Budget Score (0~20점)

```python
actual = response.total_estimated_time  # 재계산 금지; pipeline이 이미 계산한 값 사용
error = abs(actual - context.time_available_min)
```

| error | 점수 | 판정 |
|-------|------|------|
| ≤ 5분 | 20 | — |
| 6~10분 | 10 | warning |
| > 10분 | 0 | "FAIL " prefix |

> **under-fill 예외**: `actual < context.time_available_min and (fallback_used or readiness_level == "low" or len(candidates) <= 2)` → error 10분 초과도 warning으로.

---

### D. Diversity Score (0~15점)

```python
routine_ids = {block.get("exercise_id", "") for block in blocks}
recent_set = set(context.recent_exercise_ids)
repeat_ratio = len(routine_ids & recent_set) / len(routine_ids) if routine_ids else 0.0
```

| repeat_ratio | 점수 | 판정 |
|--------------|------|------|
| ≤ 0.25 | 15 | — |
| 0.25~0.50 | 10 | warning |
| 0.50~0.75 | 5 | warning |
| > 0.75 | 0 | "FAIL " prefix (후보 충분 시) |

> **예외**: `len(candidates) <= 2` → 0.75 초과도 warning.

---

### E. Balance Score (0~15점)

```python
compound_count = sum(1 for b in blocks if candidates_by_id.get(b.get("exercise_id",{}) or "",{}).get("movement_type","").upper() == "COMPOUND")
ratio = compound_count / len(blocks) if blocks else 0.0
goal_key = (context.goal or "hypertrophy").replace("_","").lower()
```

| goal | PASS 범위 | WARN 범위 | FAIL |
|------|----------|-----------|------|
| `strength` | ratio ≥ 0.6 | 0.4 ≤ ratio < 0.6 | compound_count == 0 |
| `hypertrophy` / `generalfitness` | 0.3 ≤ ratio ≤ 0.7 | 외부 | — |
| `fatloss` / `endurance` / `recomposition` | 0.2 ≤ ratio ≤ 0.8 | 외부 | — |

| 판정 | 점수 |
|------|------|
| PASS 범위 | 15 |
| WARN 범위 | 8 |
| FAIL | 0 (+ "FAIL " prefix) |

> **예외**: `len(candidates) <= 2 or fallback_used` → FAIL → warning.

---

### F. Rationale Score (0~15점)

```python
total = len(blocks)
filled = sum(1 for b in blocks if str(b.get("exercise_rationale") or "").strip())
ratio = filled / total if total else 0.0
```

| ratio | 점수 | 판정 |
|-------|------|------|
| 1.0 | 15 | — |
| 0.5~1.0 | 8 | warning |
| < 0.5 | 0 | "FAIL " prefix |

---

### G. Repair/Fallback Hygiene Score (0~10점)

```python
# repair_count: validate_and_repair_routine_output()의 violations 카운터
# fallback_used: response.is_fallback
```

| 상황 | 점수 | 판정 |
|------|------|------|
| success, repair_count == 0 | 10 | — |
| success, repair_count > 0 | 7 | warning (예상 가능한 repair) |
| success, repair_count > 2 | 3 | warning (과도한 repair) |
| fallback, repair_count == 0 | 7 | warning |
| fallback, repair_count > 0 | 3 | warning |
| generation_status == "failed" | 0 | "FAIL " prefix |

> Guard의 repair 여부는 `response.warnings` 내 `"Guard:" in w and "replaced" in w`로 확인.  
> `"Guard:" in w` 단독 조건은 per-repair 메시지와 diagnostics 요약 라인을 모두 매칭해 이중 카운트되므로 사용 금지.

---

## 6. Grade / Threshold 정책

```python
def _grade(score: int, hard_violations: List[str]) -> str:
    if hard_violations:
        return "FAIL"
    if score >= 80:
        return "PASS"
    if score >= 65:
        return "WARN"
    return "FAIL"
```

| Grade | 조건 |
|-------|------|
| **PASS** | `hard_violations == 0 AND score >= 80` |
| **WARN** | `hard_violations == 0 AND 65 <= score < 80` |
| **FAIL** | `hard_violations > 0 OR score < 65` |

---

## 7. Rebuild / Fallback 적용 정책

### V1 — 로그 기록만 (기본값)

```python
CRITIC_REBUILD_ENABLED = False  # 환경 변수 또는 settings로 관리

result = evaluate_routine_quality(response, candidates, context, repair_count, fallback_used)
log_critic_result(result)          # stdout (structured)
# response는 grade에 관계없이 그대로 반환
```

### V2 — Opt-in rebuild (향후)

```python
if CRITIC_REBUILD_ENABLED:
    if result.grade == "FAIL" and not fallback_used:
        # 1회 rebuild 시도 (LLM 재호출)
        result.should_rebuild = True
    if result.should_rebuild and rebuild_failed:
        result.should_fallback = True
```

> **V1에서 `should_rebuild`, `should_fallback`은 항상 `False`.**  
> 필드는 모델에 포함시켜 V2 전환 시 API 변경 최소화.

### Runtime 정책 요약

| Grade | V1 동작 | V2 동작 (opt-in) |
|-------|---------|-----------------|
| PASS | 그대로 반환 | 그대로 반환 |
| WARN | 반환 + dev log | 반환 + dev log |
| FAIL | 반환 + dev log | rebuild 1회 → 여전히 FAIL이면 fallback |

---

## 8. 파일 구조

```
fit-core-ai/
├── engines/
│   └── rule_critic.py                 # NEW — production critic
│
└── tests/
    ├── test_rule_critic.py            # NEW — unit tests for rule_critic
    └── evaluation/
        ├── helpers.py                 # MODIFIED — metric 함수를 rule_critic에서 import
        └── test_offline_eval_harness.py  # UNCHANGED
```

### `engines/rule_critic.py` 내부 구조

```python
# RoutineEvalContext dataclass
# RuleCriticResult dataclass
# build_eval_context()
# evaluate_routine_quality()        ← 메인 함수
# log_critic_result()               ← stdout structured log
# _check_hard_constraints_sanity()  ← private
# _coverage_score()                 ← private
# _time_score()                     ← private
# _diversity_score()                ← private
# _balance_score()                  ← private
# _rationale_score()                ← private
# _repair_hygiene_score()           ← private
# _grade()                          ← private
```

---

## 9. Offline Evaluation Harness와의 통합 방식

### 현재 상태 (구현 전)

```
tests/evaluation/helpers.py
    - check_hard_constraints()          ← 독립 구현
    - compute_coverage_score()          ← 독립 구현
    - compute_time_score()              ← 독립 구현
    - compute_diversity_score()         ← 독립 구현
    - compute_balance_score()           ← 독립 구현
    - compute_rationale_score()         ← 독립 구현
    - compute_fallback_score()          ← 독립 구현
    - EvalHarness.run()                 ← 직접 metric 호출
```

### 목표 상태 (구현 후)

```
engines/rule_critic.py
    - _check_hard_constraints_sanity()  ← production version
    - _coverage_score()                 ← production version
    - ... (모든 metric)
    - evaluate_routine_quality()        ← 메인 함수

tests/evaluation/helpers.py
    - from engines.rule_critic import (
          RoutineEvalContext, RuleCriticResult,
          build_eval_context, evaluate_routine_quality,
      )
    - EvalHarness.run():
        1. _validate_or_fallback() 호출 → RoutineDraftResponse
        2. build_eval_context(req, ...) → RoutineEvalContext
        3. evaluate_routine_quality(response, candidates, context, ...) → RuleCriticResult
        4. ScenarioResult 생성 (harness-specific fields 추가)
```

### ScenarioResult vs RuleCriticResult

| 필드 | `RuleCriticResult` | `ScenarioResult` (harness) |
|------|--------------------|--------------------------|
| `score` | ✓ | ✓ (total_score) |
| `grade` | ✓ | ✗ (harness는 passed bool) |
| `hard_violations` | ✓ (list) | ✓ (count + details) |
| `warnings` | ✓ | ✓ |
| `metric_scores` | ✓ | ✓ |
| `should_rebuild` | ✓ | ✗ |
| `fallback_used` | ✗ (input) | ✓ (output field) |
| `requested_duration` | ✗ (context) | ✓ |
| `scenario_id` | ✗ | ✓ |

### 마이그레이션 전략

1. `engines/rule_critic.py` 구현 (metric 함수 재구현; 기존 helpers.py 로직 기반)
2. `tests/evaluation/helpers.py` 수정:
   - `compute_*_score()` 함수 제거
   - `check_hard_constraints()` 제거
   - `engines.rule_critic` import 추가
   - `EvalHarness.run()` 내부에서 `evaluate_routine_quality()` 호출
3. 기존 53개 테스트 (`tests/test_prompt_regression.py` 등) 영향 없음
4. harness 15개 시나리오 재실행으로 regression 확인

---

## 10. Logging / Observability 설계

### `log_critic_result()` 출력 형식

```
[Critic] grade=PASS score=88 hard_violations=0 repair_count=0 fallback=False
[Critic] metrics: coverage=25 time=18 diversity=15 balance=15 rationale=15 hygiene=0
[Critic] warnings: ["unexpected repair occurred"]
```

FAIL 시:
```
[Critic][FAIL] grade=FAIL score=55 hard_violations=1 repair_count=2 fallback=False
[Critic][GUARD_BYPASS] exercise_id 'squat' not in candidates
[Critic][FAIL_REASON] time error too large: actual=15, requested=60, error=45
```

### stdout 구조화 원칙

- prefix: `[Critic]` (grep-friendly)
- hard_violation: `[Critic][GUARD_BYPASS]` (알람 트리거 용이)
- 한 줄 summary + 한 줄 metrics (개행 최소화)
- JSON 출력 모드: 추후 `CRITIC_LOG_FORMAT=json` 환경 변수로 opt-in

---

## 11. 테스트 계획

### `tests/test_rule_critic.py` 구조

```python
# -- RoutineEvalContext 빌드 테스트 --
def test_build_eval_context_from_request():
    """req + recent_sets → RoutineEvalContext 정상 변환"""

# -- 정상 케이스 --
def test_evaluate_routine_quality_pass_on_clean_output():
    """hard_violations=0, score≥80 → grade=PASS"""

def test_evaluate_routine_quality_warn_on_borderline_score():
    """score=70, hard_violations=0 → grade=WARN"""

# -- Hard Constraint Sanity --
def test_sanity_detects_hallucinated_exercise_id():
    """candidates에 없는 exercise_id → hard_violation"""

def test_sanity_detects_blocked_equipment_bypass():
    """BARBELL 차단인데 barbell 운동 포함 → hard_violation"""

def test_sanity_detects_pain_trigger_bypass():
    """pain=knees인데 squat 포함 → hard_violation"""

def test_sanity_detects_doms_level3_bypass():
    """chest DOMS=3인데 chest 운동 포함 → hard_violation"""

def test_sanity_detects_set_cap_exceeded():
    """total_sets > max_working_sets → hard_violation"""

# -- Coverage --
def test_coverage_score_full_coverage():
def test_coverage_score_partial_coverage_warning():
def test_coverage_score_fail_when_candidate_sufficient():
def test_coverage_exception_allows_low_coverage_with_warning():

# -- Time --
def test_time_score_within_5min():
def test_time_score_10min_error_warning():
def test_time_score_over_10min_fail():
def test_time_score_underfill_exception_low_readiness():

# -- Diversity --
def test_diversity_score_no_repetition():
def test_diversity_score_high_repetition_fail():
def test_diversity_exception_candidate_shortage():

# -- Balance --
def test_balance_score_strength_compound_priority():
def test_balance_score_hypertrophy_mix():
def test_balance_fail_strength_no_compound():

# -- Rationale --
def test_rationale_score_all_filled():
def test_rationale_score_partially_empty_warning():
def test_rationale_score_mostly_empty_fail():

# -- Hygiene --
def test_hygiene_score_no_repair():
def test_hygiene_score_single_repair_warning():
def test_hygiene_fail_on_generation_failed():

# -- Grade 결정 --
def test_grade_pass_on_high_score():
def test_grade_warn_on_borderline():
def test_grade_fail_on_low_score():
def test_grade_fail_on_hard_violation():

# -- rebuild/fallback flag --
def test_should_rebuild_false_in_v1():
def test_should_fallback_false_in_v1():
```

> 목표 신규 테스트: **30개 이상**. 기존 133 + 30 = 163+ passed.

---

## 12. 리스크

| 리스크 | 내용 | 대응 |
|--------|------|------|
| **`movement_type` 조회 불일치** | `RoutineBlock`에 `movement_type` 필드 없음; candidates에서 lookup 필요. candidates에 없으면 "UNKNOWN" | fallback lookup 명시, UNKNOWN은 balance score WARN으로 처리 |
| **`repair_count` 노출** | `validate_and_repair_routine_output()` 내부 `violations` 카운터가 현재 반환값에 없음 | pipeline에서 `response.warnings` 내 `"Guard:"` 포함 횟수로 근사 (V1); V2에서 정확한 카운터 추가 |
| **helpers.py 마이그레이션 시 harness 깨짐** | 마이그레이션 중 기존 15개 시나리오 테스트가 실패할 수 있음 | 순서: (1) rule_critic.py 구현 → (2) helpers.py import 교체 → (3) harness 재실행 |
| **under-fill 예외 기준 모호** | fallback_used / readiness / candidate_shortage 중 어느 조건이면 예외인지 | 3개 조건 중 하나라도 충족하면 예외 처리 (spec 섹션 5 기준 사용) |
| **production latency** | `evaluate_routine_quality()` 자체는 < 1ms 예상. log 출력이 latency에 영향 | `log_critic_result()` async 또는 별도 thread 고려; V1에서는 sync stdout |
| **rebuild 도입 시 LLM 재호출** | V2에서 FAIL → rebuild는 LLM 재호출 의미. 비용/latency 위험 | V2 scope; V1에서 flag 비활성화 |

---

## 13. Codex 구현용 최종 지시문

### 작업 범위

`fit-core-ai/engines/rule_critic.py` 신규 생성 + `tests/evaluation/helpers.py` 리팩터링 + `tests/test_rule_critic.py` 신규 생성.  
**`engines/routine_pipeline.py` 등 운영 코드는 변경하지 않는다** (pipeline 통합은 V2).

---

### Step 1: `engines/rule_critic.py` 생성

#### 1-1. import

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from engines.schemas import PainAreaEntry, RecentSetRecord, RoutineDraftResponse, RoutineRequest
```

#### 1-2. `RoutineEvalContext` dataclass (섹션 4 스펙 참조)

#### 1-3. `RuleCriticResult` dataclass (섹션 3 스펙 참조)

#### 1-4. `build_eval_context()` 구현

```python
def build_eval_context(
    req: RoutineRequest,
    db_target_muscles: List[str],
    recent_sets: Optional[List[RecentSetRecord]] = None,
    max_working_sets: Optional[int] = None,
) -> RoutineEvalContext:
    recent_ids = [
        r.exercise_id for r in (recent_sets or []) if r.exercise_id
    ]
    return RoutineEvalContext(
        goal=req.goal or "hypertrophy",
        time_available_min=req.time_available_min,
        target_muscles=db_target_muscles,
        blocked_equipment=req.equipment,
        pain_areas=req.pain_areas,
        doms_db=req.doms_data,
        recent_exercise_ids=recent_ids,
        readiness_level=req.readiness_level or "normal",
        max_working_sets=max_working_sets,
    )
```

#### 1-5. Private metric 함수들 구현 (섹션 5 상세 참조)

각 함수: `(score: int, warnings: List[str])` 반환.

- `_check_hard_constraints_sanity(blocks, candidates_by_id, context, max_working_sets) -> List[str]`
- `_coverage_score(blocks, candidates_by_id, context, fallback_used, candidate_count) -> (int, List[str])`
- `_time_score(response, context, fallback_used, candidate_count) -> (int, List[str])`
- `_diversity_score(blocks, context, candidate_count) -> (int, List[str])`
- `_balance_score(blocks, candidates_by_id, context, candidate_count, fallback_used) -> (int, List[str])`
- `_rationale_score(blocks) -> (int, List[str])`
- `_repair_hygiene_score(response, repair_count, fallback_used) -> (int, List[str])`
- `_grade(score, hard_violations) -> str`

#### 1-6. `evaluate_routine_quality()` 구현 (섹션 4 시그니처 참조)

```python
def evaluate_routine_quality(
    response: RoutineDraftResponse,
    candidates: List[dict],
    context: RoutineEvalContext,
    repair_count: int = 0,
    fallback_used: bool = False,
) -> RuleCriticResult:
    from engines.prescription.adjustments import _calculate_max_total_sets

    blocks = [block.model_dump(by_alias=False) for block in response.routine_blocks]
    candidates_by_id = {str(c["id"]): c for c in candidates}
    candidate_count = len(candidates)
    max_ws = context.max_working_sets or _calculate_max_total_sets(
        context.time_available_min, context.goal
    )

    hard_violations = _check_hard_constraints_sanity(blocks, candidates_by_id, context, max_ws)

    warnings: List[str] = []
    metric_scores: Dict[str, int] = {}

    metric_scores["coverage"], w = _coverage_score(blocks, candidates_by_id, context, fallback_used, candidate_count)
    warnings.extend(w)
    metric_scores["time"], w = _time_score(response, context, fallback_used, candidate_count)
    warnings.extend(w)
    metric_scores["diversity"], w = _diversity_score(blocks, context, candidate_count)
    warnings.extend(w)
    metric_scores["balance"], w = _balance_score(blocks, candidates_by_id, context, candidate_count, fallback_used)
    warnings.extend(w)
    metric_scores["rationale"], w = _rationale_score(blocks)
    warnings.extend(w)
    metric_scores["hygiene"], w = _repair_hygiene_score(response, repair_count, fallback_used)
    warnings.extend(w)

    score = 0 if hard_violations else sum(metric_scores.values())
    grade = _grade(score, hard_violations)

    return RuleCriticResult(
        score=score,
        grade=grade,
        hard_violations=hard_violations,
        warnings=warnings,
        metric_scores=metric_scores,
        should_rebuild=False,   # V1: 항상 False
        should_fallback=False,  # V1: 항상 False
    )
```

#### 1-7. `log_critic_result()` 구현

```python
def log_critic_result(result: RuleCriticResult) -> None:
    prefix = "[Critic]" if result.grade != "FAIL" else "[Critic][FAIL]"
    print(
        f"{prefix} grade={result.grade} score={result.score} "
        f"hard_violations={len(result.hard_violations)} "
        f"should_rebuild={result.should_rebuild}"
    )
    metrics_str = " ".join(f"{k}={v}" for k, v in result.metric_scores.items())
    print(f"[Critic] metrics: {metrics_str}")
    for v in result.hard_violations:
        print(f"[Critic][GUARD_BYPASS] {v}")
    for w in result.warnings:
        print(f"[Critic] warning: {w}")
```

---

### Step 2: `tests/evaluation/helpers.py` 리팩터링

**목표**: metric 함수를 `engines.rule_critic`에서 import. 기존 `ScenarioResult`, `EvalHarness`, `save_report()` 유지.

```python
# helpers.py 변경 후 import 섹션
from engines.rule_critic import (
    RoutineEvalContext,
    RuleCriticResult,
    build_eval_context,
    evaluate_routine_quality,
    _check_hard_constraints_sanity as _sanity_check,
)
```

`EvalHarness.run()` 변경:
1. `_validate_or_fallback()` 호출 → `RoutineDraftResponse`
2. `build_eval_context(req, target_muscles, recent_sets)` → `RoutineEvalContext`
3. `context.overrides` = `scenario.get("expected", {})` (coverage_exception, allowed_underfill, candidate_shortage, min_compound_count, rationale_must_be_nonempty 등 harness 전용 플래그 주입)
4. `repair_count` = `len([w for w in response.warnings if "Guard:" in w and "replaced" in w])`
5. `evaluate_routine_quality(response, ranked_candidates, context, repair_count, fallback_used)` → `RuleCriticResult`
6. `ScenarioResult` 필드를 `RuleCriticResult`에서 채움

> **pipeline**은 항상 `overrides={}` (기본값) 유지. `_coverage_score()`, `_balance_score()` 등 내부 함수는 `context.overrides.get("coverage_exception", False)` 패턴으로 플래그 참조.  
> 런타임 heuristic(`len(candidates) <= 2`, `readiness_level == "low"`)은 `overrides` 값보다 후순위로 적용.

`check_hard_constraints()` 함수: `test_offline_eval_harness.py:93`에서 직접 호출되므로 **정확한 시그니처를 유지**한 backward-compat 래퍼로 존치:
```python
def check_hard_constraints(
    blocks: list[dict],
    candidates: list[dict],
    scenario: dict,
    max_working_sets: int,
) -> list[str]:
    """Backward-compat wrapper: scenario dict → RoutineEvalContext → _sanity_check() 위임."""
    from engines.schemas import PainAreaEntry
    pain_areas = [PainAreaEntry(**item) for item in scenario.get("current_pain_areas", [])]
    context = RoutineEvalContext(
        goal=scenario.get("goal", "hypertrophy"),
        time_available_min=scenario.get("duration_min", 60),
        target_muscles=scenario.get("target_muscles", []),
        blocked_equipment=scenario.get("unavailable_equipment", []),
        pain_areas=pain_areas,
        doms_db=scenario.get("doms", {}),
        recent_exercise_ids=[],
        overrides=scenario.get("expected", {}),
    )
    candidates_by_id = {str(c["id"]): c for c in candidates}
    # forbidden_exercise_ids 체크도 유지 (테스트 기대값)
    violations = []
    ids = {str(b.get("exercise_id") or "") for b in blocks}
    for forbidden_id in scenario.get("expected", {}).get("forbidden_exercise_ids", []):
        if forbidden_id in ids:
            violations.append(f"forbidden exercise present: {forbidden_id}")
    violations.extend(_sanity_check(blocks, candidates_by_id, context, max_working_sets))
    return violations
```

---

### Step 3: `tests/test_rule_critic.py` 생성

섹션 11의 테스트 목록 구현. 각 테스트는:
- `RoutineDraftResponse` fixture를 직접 생성 (DB/LLM 없이)
- `RoutineEvalContext` 직접 생성
- `evaluate_routine_quality()` 호출
- `RuleCriticResult` 검증

**중요**: 기존 `conftest.py`의 `sample_llm_output`, `mock_candidates` fixture 재사용 가능.

---

### Step 4: 검증

```bash
# 기존 테스트 깨지지 않음 (133 통과 확인)
pytest tests/test_prompt_regression.py tests/test_guard.py tests/test_contract.py tests/test_fallback.py -q

# 새 critic 테스트
pytest tests/test_rule_critic.py -v --tb=short

# harness 15개 시나리오 여전히 통과
pytest tests/evaluation/ -q

# 전체
pytest tests/ -q
```

### 제약 조건

- LLM API 호출 절대 금지
- DB 연결 없음
- `engines/routine_pipeline.py` 변경 없음 (pipeline 통합은 별도 PR)
- `should_rebuild`, `should_fallback` V1에서 항상 `False`
- `repair_count` 근사 허용 (V1): `len([w for w in response.warnings if "Guard:" in w and "replaced" in w])`  
  (`"Guard:" in w` 단독 사용 금지 — diagnostics 요약 라인과 이중 카운트됨)
- `engines/rule_critic.py`는 `tests/` 디렉토리를 import하지 않는다

---

*이 스펙은 P1-2의 산출물이다. 구현은 P1-3(또는 P1-2-impl)에서 Codex가 담당한다.*

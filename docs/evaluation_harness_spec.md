# Offline Evaluation Harness — 설계 스펙

**버전**: 1.0  
**대상 레포**: `fit-core-ai`  
**현재 테스트 상태**: 52 passed (P0 완료 기준)  
**관련 P-task**: P1-1

---

## 1. 설계 요약

AI 루틴 생성기의 품질을 정량 평가하는 **Offline Evaluation Harness**다.  
LLM API 없이 CI에서 안정적으로 실행되는 regression baseline을 목표로 한다.

### 핵심 설계 원칙

| 원칙 | 내용 |
|------|------|
| **No LLM calls** | LLM output을 fixture로 주입; Gemini/Claude API 호출 없음 |
| **Deterministic** | 같은 입력 → 항상 같은 결과; flaky test 금지 |
| **Pipeline 재사용** | `_validate_or_fallback()` → `apply_deterministic_targets()` → 결과 평가 |
| **Guard 중복 금지** | 기존 `test_guard.py` 단위 테스트와 겹치지 않음; 시나리오 단위 e2e |
| **Rule-based scoring** | LLM-as-Judge는 P2로 미룸; 이번은 순수 결정론적 점수 산식 |

### 테스트 진입점

```
fixture LLMRoutineOutput
    ↓
_validate_or_fallback(output, req, candidates, ...)
    ↓ guard + repair
apply_deterministic_targets(...)
    ↓ prescription override
RoutineDraftResponse
    ↓
EvalHarness.evaluate(scenario, response)
    → ScenarioResult
```

---

## 2. 평가 범위 / 비범위

### 평가 범위 (IN SCOPE)

- Hard constraint 위반 (hallucination, equipment, pain, DOMS)
- Working set cap 준수
- Target muscle coverage
- Time budget 오차
- Diversity (최근 운동 반복 억제)
- Compound/isolation 균형 (goal별)
- Rationale 비어있음 여부
- Fallback / schema repair 발생 여부
- `RoutineDraftResponse` 필드 완전성

### 비범위 (OUT OF SCOPE)

- LLM 프롬프트 품질 평가 (→ `test_prompt_regression.py`)
- 개별 방어 로직 단위 테스트 (→ `test_guard.py`, `test_fallback.py`)
- 운동 과학적 최적성 (→ P2 LLM-as-Judge)
- API 계약 직렬화 (→ `test_contract.py`)
- 실시간 성능/latency

---

## 3. 파일 구조

```
fit-core-ai/
└── tests/
    └── evaluation/
        ├── __init__.py
        ├── conftest.py                    # pytest_addoption (--eval-report), 공유 fixture
        ├── test_offline_eval_harness.py   # pytest 진입점 (parametrize)
        ├── harness.py                     # EvalHarness, ScenarioResult, 점수 산식
        ├── metrics.py                     # 개별 metric 함수 (pure functions)
        └── fixtures/
            ├── scenarios.json             # 전체 시나리오 배열 (S01~S15)
            └── candidates_library.json    # 재사용 가능 후보 운동 풀
```

> **운영 코드 변경 없음**: harness는 `engines/` 내부 함수를 직접 import하지만,
> 운영 코드를 수정하지 않는다.

---

## 4. 시나리오 스키마 (JSON)

```jsonc
{
  "id": "S01",
  "name": "정상 hypertrophy push — 제약 없음",
  "goal": "hypertrophy",
  "target_split_label": "push",
  "target_muscles": ["chest", "front-deltoids", "triceps"],
  "duration_min": 60,
  "readiness_level": "normal",
  "available_equipment": ["BARBELL", "DUMBBELL", "CABLE"],
  "unavailable_equipment": [],
  "current_pain_areas": [],
  "doms": {},
  "recent_exercises": [],           // exercise_id 문자열 배열 (e.g. ["barbell_bench_press"])
  "candidates": [
    {
      "id": "barbell_bench_press",
      "name_kr": "바벨 벤치프레스",
      "name_en": "Barbell Bench Press",
      "primary_muscle": "chest",
      "secondary_muscle": "triceps",
      "equipment_req": "BARBELL",
      "efficiency_tier": 5,
      "movement_type": "COMPOUND",
      "pain_triggers": null
    }
    // ... 실제 scenarios.json에는 4~8개 후보를 기재해야 함 (아래 주의사항 참고)
  ],
  // ⚠️ mock_llm_output.exercises는 반드시 duration_min의 ~80%를 채우도록 복수 운동 포함해야 함.
  // 단일 운동만 있으면 estimate_routine_time_min()이 duration_min을 크게 미달해
  // time_score=0이 되어 total_score < 80이 된다.
  // 각 시나리오의 exercises 배열을 3~5개로 채울 것.
  "mock_llm_output": {
    "total_estimated_time": 55,
    "summary_title": "Push Hypertrophy",
    "rationale_summary": ["바벨 벤치프레스 중심 compound 루틴"],
    "warnings": [],
    "exercises": [
      {
        "exercise_id": "barbell_bench_press",
        "exercise_name": "바벨 벤치프레스",
        "primary_muscles": ["chest"],
        "movement_type": "COMPOUND",
        "target_reps": 8,
        "sets": 3,
        "rest_time_sec": 120,
        "target_weight_kg": null,
        "exercise_rationale": "chest primary, COMPOUND, barbell available, hypertrophy goal"
      }
      // + 추가 3~4개 운동 (시나리오 duration에 맞게)
    ]
  },
  "expected": {
    "must_pass": true,
    "generation_status": "success",
    "is_fallback": false,
    "expected_repair": false,
    "coverage_exception": false,      // true면 required_muscle_coverage < 70% 허용 (warning 처리)
    "forbidden_exercise_ids": [],
    "required_muscle_coverage": ["chest"],
    "min_compound_count": 1,
    "max_time_error_min": 5,
    "max_working_sets": null,
    "rationale_must_be_nonempty": true,
    "allowed_warnings": [],
    "notes": "정상 경로; hard violation 0건 필수"
  }
}
```

### 필드 설명

| 필드 | 타입 | 설명 |
|------|------|------|
| `id` | str | `S01`~`S15` 형식 |
| `goal` | str | `hypertrophy` / `strength` / `fatloss` / `endurance` / `generalfitness` |
| `target_split_label` | str? | `push` / `pull` / `legs` / `upper` / `lower` / `full_body` / null |
| `target_muscles` | list[str] | registry.py의 muscle slug |
| `candidates` | list[dict] | `score_candidate_exercises()` input 형식 그대로 |
| `mock_llm_output` | dict | `LLMRoutineOutput` 직렬화; fixture로 주입 |
| `expected.must_pass` | bool | false면 실패가 예상되는 시나리오 (negative test) |
| `expected.is_fallback` | bool | fallback 루틴이 나와야 하면 true |
| `expected.expected_repair` | bool | guard repair가 발생해야 하면 true |
| `expected.forbidden_exercise_ids` | list | 최종 루틴에 절대 없어야 할 exercise_id |
| `expected.required_muscle_coverage` | list | 최종 루틴에 반드시 있어야 할 primary_muscle slug |
| `expected.min_compound_count` | int? | COMPOUND 운동 최소 개수 |
| `expected.max_time_error_min` | int | 허용 시간 오차 (분) |
| `expected.max_working_sets` | int? | null이면 계산된 cap 사용 |
| `expected.coverage_exception` | bool | true면 coverage < 70%도 warning (후보 부족/제약 시나리오) |
| `recent_exercises` | list[str] | exercise_id 문자열 배열; diversity metric의 반복 기준 |

---

## 5. Metric별 판정 기준

### A. Hard Constraint Gate (pass/fail — 점수 없음)

모두 통과해야 이후 metric 채점 진행. 하나라도 위반 시 **전체 시나리오 실패**.

| 체크 | 판정 |
|------|------|
| 최종 루틴에 `forbidden_exercise_ids` 포함 여부 | 포함 → 즉시 실패 |
| 최종 루틴 `exercise_id`가 candidates에 없는 경우 | 1건 이상 → 즉시 실패 |
| unavailable equipment 사용 | 1건 이상 → 즉시 실패 |
| current_pain_areas 운동 포함 | 1건 이상 → 즉시 실패 |
| DOMS level 3 근육 운동 포함 | 1건 이상 → 즉시 실패 |
| total_working_sets > max_working_sets | → 즉시 실패 |

> `max_working_sets`는 `_calculate_max_total_sets(duration_min, goal)` 결과 사용.  
> scenario `expected.max_working_sets`가 null이 아니면 그 값을 우선 사용.

---

### B. Coverage Score (0~25점)

```python
covered = set(primary_muscles_in_routine) & set(required_muscle_coverage)
ratio = len(covered) / len(required_muscle_coverage)  # required가 빈 경우 1.0
score = round(ratio * 25)
```

| ratio | 점수 | 판정 |
|-------|------|------|
| 1.0 | 25 | pass |
| ≥ 0.7 | 18~24 | warning |
| < 0.7 | < 18 | fail (후보 충분한 시나리오에서) |

> 후보 부족 시나리오(S11)와 low readiness 시나리오(S08)에서는 < 0.7도 허용.  
> `expected.coverage_exception: true` 필드로 명시.

---

### C. Time Budget Score (0~20점)

```python
actual_time = estimate_routine_time_min(routine_blocks_as_exercises)
error_min = abs(actual_time - duration_min)
```

| error | 점수 | 판정 |
|-------|------|------|
| ≤ 5분 | 20 | pass |
| 6~10분 | 10 | warning |
| > 10분 | 0 | fail |

> `expected.max_time_error_min`으로 시나리오별 override 가능.  
> under-fill은 low readiness / 후보 부족 시나리오에서 허용 (warning으로 처리).

---

### D. Diversity Score (0~15점)

```python
recent_ids = set(recent_exercises)  # scenario의 recent_exercises 필드
routine_ids = {block.exercise_id for block in routine_blocks}
repeat_count = len(routine_ids & recent_ids)
repeat_ratio = repeat_count / len(routine_ids) if routine_ids else 0
```

| repeat_ratio | 점수 | 판정 |
|--------------|------|------|
| ≤ 0.25 | 15 | pass |
| 0.25~0.50 | 10 | pass |
| 0.50~0.75 | 5 | warning |
| > 0.75 | 0 | fail |

> 후보가 1~2개뿐인 시나리오(S11)에서는 repeat_ratio 관계없이 warning으로 처리.

---

### E. Balance Score (0~15점)

```python
compound_count = sum(1 for b in routine_blocks if b.movement_type == "COMPOUND")
total = len(routine_blocks)
compound_ratio = compound_count / total if total > 0 else 0
```

목표별 기대 compound_ratio 범위:

| goal | 기대 범위 | 벗어날 시 |
|------|----------|----------|
| `strength` | ≥ 0.6 | warning if < 0.4, fail if 0 |
| `hypertrophy` | 0.3~0.7 | warning if outside |
| `fatloss` / `endurance` | 0.2~0.8 | 대부분 허용 |
| `generalfitness` | 0.3~0.7 | warning if outside |

| 판정 | 점수 |
|------|------|
| 기대 범위 내 | 15 |
| warning 범위 | 8 |
| fail 조건 충족 | 0 |

> 후보가 모두 ISOLATION인 시나리오(S11)에서는 compound=0도 warning으로 처리.  
> `expected.min_compound_count` 필드로 시나리오별 최소값 명시 가능.

---

### F. Rationale Quality Score (0~15점)

```python
exercises_with_rationale = [b for b in routine_blocks if b.exercise_rationale.strip()]
ratio = len(exercises_with_rationale) / len(routine_blocks) if routine_blocks else 1.0
```

| ratio | 점수 | 판정 |
|-------|------|------|
| 1.0 | 15 | pass |
| 0.5~1.0 | 8 | warning |
| < 0.5 | 0 | fail |

> rationale이 완전히 비어있는 운동이 있으면 항상 warning 이상.  
> `expected.rationale_must_be_nonempty: true` 시 ratio < 1.0 → fail.

---

### G. Fallback / Repair Score (0~10점)

| 상황 | 점수 |
|------|------|
| success, no repair | 10 |
| success, with repair (예상된 경우) | 10 |
| success, with repair (예상 안 된 경우) | 5 (warning) |
| fallback (예상된 경우) | 10 |
| fallback (예상 안 된 경우) | 0 (fail) |
| failed status | 0 |

> `expected.is_fallback: true` / `expected.expected_repair: true`로 시나리오별 예상 상태 명시.

---

## 6. 점수 산식

```
Hard Constraint Gate → 통과 못하면 total_score = 0, hard_violations > 0 → 시나리오 실패

total_score (100점 만점) =
    coverage_score    (0~25)
  + time_score        (0~20)
  + diversity_score   (0~15)
  + balance_score     (0~15)
  + rationale_score   (0~15)
  + fallback_score    (0~10)

시나리오 pass 기준:
  1. hard_violations == 0
  2. total_score >= 80
  3. 시나리오별 required checks 통과
     (forbidden_exercise_ids 없음, required_muscle_coverage 달성 등)
```

---

## 7. 초기 시나리오 목록 (15개)

| ID | 이름 | goal | split | duration | readiness | 특수 조건 | 예상 결과 |
|----|------|------|-------|----------|-----------|----------|----------|
| S01 | 정상 hypertrophy push | hypertrophy | push | 60 | normal | 없음 | success, score≥80 |
| S02 | strength lower body barbell | strength | legs | 60 | normal | barbell 포함 | success, compound≥2 |
| S03 | fatLoss 장비 제한 | fatloss | full_body | 45 | normal | barbell 차단 | success, no BARBELL |
| S04 | shoulder pain present | hypertrophy | push | 45 | normal | pain=front-deltoids | success, no shoulder trigger |
| S05 | knee pain present | strength | legs | 60 | normal | pain=knees | success, no squat |
| S06 | DOMS level 3 target muscle | hypertrophy | push | 45 | normal | chest DOMS=3 | success, no chest exercise |
| S07 | unavailable equipment blocked | hypertrophy | push | 60 | normal | BARBELL 차단 | success, bodyweight/dumbbell only |
| S08 | low readiness volume reduction | hypertrophy | upper | 60 | low | 없음 | success, COMPOUND sets≤2 |
| S09 | short time budget 20min | fatloss | push | 20 | normal | 없음 | success, total_time≤25 |
| S10 | long time budget 60min | hypertrophy | pull | 60 | normal | 없음 | success, exercise_count≥4 |
| S11 | candidate shortage | hypertrophy | push | 45 | normal | 후보 2개만 | success or fallback (expected) |
| S12 | recent exercise repetition risk | hypertrophy | push | 45 | normal | 최근 동일 ID | warning, diversity warning |
| S13 | bodyweight-only routine | generalfitness | full_body | 30 | normal | 모든 장비 차단 | success, BODYWEIGHT only |
| S14 | mixed target muscles (upper+core) | hypertrophy | upper | 45 | normal | abs+chest 혼합 | success, coverage≥2 muscle groups |
| S15 | malformed LLM output — schema repair | hypertrophy | push | 60 | normal | hallucination ID 포함 | expected_repair=true |

---

> **⚠️ mock_llm_output 구현 주의**: 아래 상세 예시의 `exercises` 배열은 설명 목적으로 축약되어 있다.
> 실제 `scenarios.json` 구현 시 각 시나리오의 `duration_min`의 ~80%를 채우도록
> 3~5개 운동을 포함해야 한다. 단일 운동(~12분)으로는 time_score가 0이 되어
> `total_score < 80`이 되며 시나리오가 의도치 않게 실패한다.

### 시나리오 상세: S04 (shoulder pain)

```jsonc
{
  "id": "S04",
  "name": "shoulder pain — front-deltoid trigger 운동 차단",
  "goal": "hypertrophy",
  "target_split_label": "push",
  "target_muscles": ["chest", "front-deltoids", "triceps"],
  "duration_min": 45,
  "readiness_level": "normal",
  "current_pain_areas": [{"body_part": "front-deltoids", "side": "left", "severity": "mild"}],
  "doms": {},
  "recent_exercises": [],
  "candidates": [
    {"id": "overhead_press", "name_kr": "오버헤드프레스", "primary_muscle": "front-deltoids",
     "equipment_req": "BARBELL", "efficiency_tier": 5, "movement_type": "COMPOUND",
     "pain_triggers": "front-deltoids, shoulder"},
    {"id": "barbell_bench_press", "name_kr": "바벨 벤치프레스", "primary_muscle": "chest",
     "equipment_req": "BARBELL", "efficiency_tier": 5, "movement_type": "COMPOUND",
     "pain_triggers": null},
    {"id": "tricep_pushdown", "name_kr": "트라이셉 푸시다운", "primary_muscle": "triceps",
     "equipment_req": "CABLE", "efficiency_tier": 3, "movement_type": "ISOLATION",
     "pain_triggers": null}
  ],
  "mock_llm_output": {
    "total_estimated_time": 40,
    "summary_title": "Push — Shoulder Safe",
    "rationale_summary": ["front-deltoid pain: overhead press 제외"],
    "warnings": ["오버헤드프레스 pain trigger로 제외"],
    "exercises": [
      {"exercise_id": "overhead_press", "exercise_name": "오버헤드프레스",
       "primary_muscles": ["front-deltoids"], "movement_type": "COMPOUND",
       "target_reps": 8, "sets": 3, "rest_time_sec": 120, "target_weight_kg": null,
       "exercise_rationale": "front-deltoid primary; however pain is present"},
      {"exercise_id": "barbell_bench_press", "exercise_name": "바벨 벤치프레스",
       "primary_muscles": ["chest"], "movement_type": "COMPOUND",
       "target_reps": 8, "sets": 3, "rest_time_sec": 120, "target_weight_kg": null,
       "exercise_rationale": "chest compound; no pain trigger"}
    ]
  },
  "expected": {
    "must_pass": true,
    "generation_status": "success",
    "is_fallback": false,
    "expected_repair": true,
    "forbidden_exercise_ids": ["overhead_press"],
    "required_muscle_coverage": ["chest"],
    "min_compound_count": 1,
    "max_time_error_min": 10,
    "rationale_must_be_nonempty": true,
    "notes": "LLM이 pain trigger 운동을 포함했으나 guard가 repair해야 함"
  }
}
```

### 시나리오 상세: S06 (DOMS level 3)

```jsonc
{
  "id": "S06",
  "name": "DOMS level 3 — chest 운동 완전 제외",
  "goal": "hypertrophy",
  "target_split_label": "push",
  "target_muscles": ["chest", "front-deltoids", "triceps"],
  "duration_min": 45,
  "doms": {"chest": 3},
  "current_pain_areas": [],
  "candidates": [
    {"id": "barbell_bench_press", "primary_muscle": "chest", "equipment_req": "BARBELL",
     "movement_type": "COMPOUND", "pain_triggers": null, "efficiency_tier": 5},
    {"id": "dumbbell_fly", "primary_muscle": "chest", "equipment_req": "DUMBBELL",
     "movement_type": "ISOLATION", "pain_triggers": null, "efficiency_tier": 3},
    {"id": "overhead_press", "primary_muscle": "front-deltoids", "equipment_req": "BARBELL",
     "movement_type": "COMPOUND", "pain_triggers": null, "efficiency_tier": 5},
    {"id": "tricep_pushdown", "primary_muscle": "triceps", "equipment_req": "CABLE",
     "movement_type": "ISOLATION", "pain_triggers": null, "efficiency_tier": 3}
  ],
  "mock_llm_output": {
    "exercises": [
      {"exercise_id": "barbell_bench_press", "primary_muscles": ["chest"],
       "movement_type": "COMPOUND", "sets": 3, "target_reps": 8, "rest_time_sec": 120,
       "target_weight_kg": null, "exercise_rationale": "chest compound"},
      {"exercise_id": "overhead_press", "primary_muscles": ["front-deltoids"],
       "movement_type": "COMPOUND", "sets": 3, "target_reps": 8, "rest_time_sec": 120,
       "target_weight_kg": null, "exercise_rationale": "shoulder compound"}
    ]
  },
  "expected": {
    "must_pass": true,
    "is_fallback": false,
    "expected_repair": true,
    "forbidden_exercise_ids": ["barbell_bench_press", "dumbbell_fly"],
    "required_muscle_coverage": ["front-deltoids"],
    "coverage_exception": false,
    "notes": "DOMS 3 chest → score_candidate_exercises가 이미 chest 제거; LLM이 실수로 포함하면 repair"
  }
}
```

### 시나리오 상세: S11 (candidate shortage)

```jsonc
{
  "id": "S11",
  "name": "candidate shortage — 후보 2개 제한",
  "goal": "hypertrophy",
  "target_split_label": "push",
  "target_muscles": ["chest", "front-deltoids", "triceps"],
  "duration_min": 45,
  "candidates": [
    {"id": "pushup", "primary_muscle": "chest", "equipment_req": "BODYWEIGHT",
     "movement_type": "COMPOUND", "pain_triggers": null, "efficiency_tier": 2},
    {"id": "tricep_dip", "primary_muscle": "triceps", "equipment_req": "BODYWEIGHT",
     "movement_type": "COMPOUND", "pain_triggers": null, "efficiency_tier": 2}
  ],
  "mock_llm_output": {
    "exercises": [
      {"exercise_id": "pushup", "primary_muscles": ["chest"], "movement_type": "COMPOUND",
       "sets": 3, "target_reps": 12, "rest_time_sec": 90, "target_weight_kg": null,
       "exercise_rationale": "only available chest compound; bodyweight due to candidate shortage"},
      {"exercise_id": "tricep_dip", "primary_muscles": ["triceps"], "movement_type": "COMPOUND",
       "sets": 3, "target_reps": 12, "rest_time_sec": 90, "target_weight_kg": null,
       "exercise_rationale": "only available triceps compound"}
    ]
  },
  "expected": {
    "must_pass": true,
    "is_fallback": false,
    "coverage_exception": true,
    "required_muscle_coverage": ["chest"],
    "max_time_error_min": 10,
    "notes": "후보 부족으로 front-deltoids 미커버 허용; coverage_exception=true로 60% 미만도 warning 처리"
  }
}
```

### 시나리오 상세: S15 (schema repair)

```jsonc
{
  "id": "S15",
  "name": "malformed LLM output — hallucination ID repair",
  "goal": "hypertrophy",
  "target_split_label": "push",
  "target_muscles": ["chest", "triceps"],
  "duration_min": 45,
  "candidates": [
    {"id": "barbell_bench_press", "primary_muscle": "chest", "equipment_req": "BARBELL",
     "movement_type": "COMPOUND", "pain_triggers": null, "efficiency_tier": 5},
    {"id": "tricep_overhead", "primary_muscle": "triceps", "equipment_req": "DUMBBELL",
     "movement_type": "ISOLATION", "pain_triggers": null, "efficiency_tier": 3}
  ],
  "mock_llm_output": {
    "exercises": [
      {"exercise_id": "HALLUCINATED_EXERCISE_99", "exercise_name": "존재하지않는운동",
       "primary_muscles": ["chest"], "movement_type": "COMPOUND",
       "sets": 3, "target_reps": 8, "rest_time_sec": 120, "target_weight_kg": null,
       "exercise_rationale": "hallucinated exercise"},
      {"exercise_id": "barbell_bench_press", "exercise_name": "바벨 벤치프레스",
       "primary_muscles": ["chest"], "movement_type": "COMPOUND",
       "sets": 3, "target_reps": 8, "rest_time_sec": 120, "target_weight_kg": null,
       "exercise_rationale": "valid exercise"}
    ]
  },
  "expected": {
    "must_pass": true,
    "is_fallback": false,
    "expected_repair": true,
    "forbidden_exercise_ids": ["HALLUCINATED_EXERCISE_99"],
    "required_muscle_coverage": ["chest"],
    "notes": "Guard가 hallucination 감지 후 교체; 최종 루틴에 hallucinated ID 없어야 함"
  }
}
```

---

## 8. pytest 실행 방식

### 기본 실행

```bash
# 전체 evaluation suite
pytest tests/evaluation/test_offline_eval_harness.py -v

# 특정 시나리오
pytest tests/evaluation/test_offline_eval_harness.py -k "S04 or S06"

# JSON 리포트 포함
pytest tests/evaluation/test_offline_eval_harness.py --eval-report=eval_report.json
```

### `test_offline_eval_harness.py` 구조

```python
import json
import pytest
from pathlib import Path
from tests.evaluation.harness import EvalHarness, ScenarioResult

SCENARIOS_PATH = Path(__file__).parent / "fixtures" / "scenarios.json"
SCENARIOS = json.loads(SCENARIOS_PATH.read_text())


def pytest_configure(config):
    config.addinivalue_line("markers", "eval: offline evaluation scenario")


@pytest.fixture(scope="session")
def harness():
    return EvalHarness()


@pytest.mark.eval
@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["id"] for s in SCENARIOS])
def test_scenario(harness, scenario):
    result: ScenarioResult = harness.run(scenario)

    # 1. Hard gate
    assert result.hard_violations == 0, (
        f"[{scenario['id']}] Hard violations: {result.hard_violation_details}"
    )

    # 2. Score gate
    if scenario["expected"].get("must_pass", True):
        assert result.total_score >= 80, (
            f"[{scenario['id']}] total_score={result.total_score} < 80\n"
            f"  metrics={result.metrics}"
        )

    # 3. Scenario-specific checks
    for forbidden_id in scenario["expected"].get("forbidden_exercise_ids", []):
        actual_ids = [b["exercise_id"] for b in result.routine_blocks]
        assert forbidden_id not in actual_ids, (
            f"[{scenario['id']}] Forbidden exercise_id '{forbidden_id}' found in output"
        )

    # 4. Fallback expectation
    expected_fallback = scenario["expected"].get("is_fallback", False)
    assert result.is_fallback == expected_fallback, (
        f"[{scenario['id']}] is_fallback={result.is_fallback}, expected={expected_fallback}"
    )

    # 5. Repair expectation
    if not scenario["expected"].get("expected_repair", False):
        assert not result.repair_occurred, (
            f"[{scenario['id']}] Unexpected repair occurred: {result.repair_reason}"
        )
```

---

## 9. JSON 리포트 형식

```jsonc
{
  "run_at": "2026-05-27T12:00:00Z",
  "total_scenarios": 15,
  "passed": 14,
  "failed": 1,
  "scenarios": [
    {
      "scenario_id": "S01",
      "scenario_name": "정상 hypertrophy push",
      "passed": true,
      "total_score": 92,
      "hard_violations": 0,
      "hard_violation_details": [],
      "metrics": {
        "coverage_score": 25,
        "time_score": 20,
        "diversity_score": 15,
        "balance_score": 15,
        "rationale_score": 15,
        "fallback_score": 10
      },
      "warnings": [],
      "is_fallback": false,
      "repair_occurred": false,
      "routine_blocks_count": 5,
      "total_estimated_time": 57,
      "requested_duration": 60
    }
  ]
}
```

---

## 10. 구현 리스크

| 리스크 | 내용 | 대응 |
|--------|------|------|
| **pipeline 진입점** | `_validate_or_fallback()`이 private 함수; import 경로 변경 시 harness 깨짐 | harness.py에서 import, 문서화 |
| **mock_llm_output과 실제 LLM 괴리** | fixture가 실제 LLM 출력 패턴을 못 반영할 수 있음 | 주기적 fixture 갱신 (P2 단계) |
| **DOMS 처리 경계** | `score_candidate_exercises()`가 DOMS 3 후보를 이미 제거; LLM fixture에 포함하면 guard에서 다시 판정 | S06처럼 명시적으로 이 동작을 테스트하는 시나리오 설계 |
| **estimate_routine_time_min() 정밀도** | 5분 warmup + 2분 transition 고정값; 일부 시나리오에서 time_error > 5분 가능 | `max_time_error_min` 시나리오별 조정 |
| **repair 감지** | `validate_and_repair_routine_output()`의 repair 여부를 외부에서 감지하는 방법 | repair_reason 반환값 (`"none"` vs 다른 값) 또는 warnings 내 Guard 메시지 확인 |
| **JSON fixture 유지보수** | scenarios.json이 커지면 관리 어려움 | 나중에 `scenarios/S01.json` 분리; 첫 버전은 단일 파일 |

---

## 11. `harness.py` 인터페이스 스펙

```python
# tests/evaluation/harness.py

@dataclass
class ScenarioResult:
    scenario_id: str
    passed: bool
    total_score: int
    hard_violations: int
    hard_violation_details: list[str]
    metrics: dict[str, int]   # coverage, time, diversity, balance, rationale, fallback
    warnings: list[str]
    is_fallback: bool
    repair_occurred: bool
    repair_reason: str
    routine_blocks: list[dict]   # RoutineBlock.model_dump(by_alias=False) 결과
    total_estimated_time: int    # response.total_estimated_time 직접 사용 (재계산 금지)


class EvalHarness:
    def run(self, scenario: dict) -> ScenarioResult:
        """
        1. scenario에서 RoutineRequest, LLMRoutineOutput, candidates 빌드
        2. score_candidate_exercises()로 candidates scoring
        3. _validate_or_fallback() 호출 (LLM 호출 없음)
        4. RoutineDraftResponse.routine_blocks → list[dict] 변환 (model_dump 사용)
        5. metrics 계산
        """
        ...
```

### 파이프라인 호출 방식

```python
from engines.routine_pipeline import _validate_or_fallback
from engines.prescription.adjustments import _calculate_max_total_sets
from engines.schemas import LLMRoutineOutput, RoutineRequest, PainAreaEntry

req = RoutineRequest(
    time_available_min=scenario["duration_min"],
    target_split_label=scenario.get("target_split_label"),
    pain_areas=[PainAreaEntry(**p) for p in scenario["current_pain_areas"]],
    doms_data=scenario["doms"],
    equipment=scenario["unavailable_equipment"],
    goal=scenario["goal"],
    readiness_level=scenario.get("readiness_level", "normal"),
)
mock_output = LLMRoutineOutput.model_validate(scenario["mock_llm_output"])
max_total_sets = _calculate_max_total_sets(scenario["duration_min"], scenario["goal"])

response = _validate_or_fallback(
    output=mock_output,
    req=req,
    ranked_candidates=scenario["candidates"],  # 이미 scored 상태로 가정 or raw candidates
    max_total_sets=max_total_sets,
    doms_db=scenario["doms"],
    goal=scenario["goal"],
    db_target_muscles=scenario["target_muscles"],
    recent_sets=None,
    profile=None,
)
```

> **주의**: `ranked_candidates`는 `score_candidate_exercises()` 통과 후 결과를 사용해야 함.  
> candidates 필드에 `score`, `score_reasons` 없으면 `score_candidate_exercises()` 먼저 실행.

---

## 12. `metrics.py` 함수 스펙

```python
# tests/evaluation/metrics.py

def compute_coverage_score(
    routine_blocks: list[dict],
    required_muscle_coverage: list[str],
    coverage_exception: bool = False,
) -> tuple[int, list[str]]:
    """returns (score, warnings)"""

def compute_time_score(
    actual_time_min: int,          # ScenarioResult.total_estimated_time 사용 (response에서 직접)
    requested_duration_min: int,
    max_time_error_min: int = 5,
) -> tuple[int, list[str]]:
    """response.total_estimated_time과 duration_min 비교; estimate 재계산 금지"""

def compute_diversity_score(
    routine_blocks: list[dict],
    recent_exercise_ids: list[str],
    candidate_shortage: bool = False,
) -> tuple[int, list[str]]:

def compute_balance_score(
    routine_blocks: list[dict],
    goal: str,
    min_compound_count: int = 0,
) -> tuple[int, list[str]]:

def compute_rationale_score(
    routine_blocks: list[dict],
    rationale_must_be_nonempty: bool = True,
) -> tuple[int, list[str]]:

def compute_fallback_score(
    is_fallback: bool,
    repair_occurred: bool,
    expected_fallback: bool,
    expected_repair: bool,
) -> tuple[int, list[str]]:

def check_hard_constraints(
    routine_blocks: list[dict],
    candidates: list[dict],
    forbidden_exercise_ids: list[str],
    unavailable_equipment: list[str],
    pain_areas: list[dict],
    doms: dict[str, int],
    max_working_sets: int,
) -> tuple[int, list[str]]:
    """returns (violation_count, violation_details)"""
```

---

## 13. Codex 구현용 최종 지시문

### 작업 범위

`fit-core-ai/tests/evaluation/` 디렉토리를 신규 생성하고 아래 파일을 구현한다.  
운영 코드(`engines/`) 변경 없음. 기존 테스트 수정 없음.

### 구현 순서

**Step 1**: `tests/evaluation/__init__.py` (빈 파일)

**Step 2**: `tests/evaluation/fixtures/scenarios.json`  
- 이 스펙 문서의 시나리오 표(S01~S15)를 JSON 배열로 작성  
- 각 시나리오는 이 스펙의 섹션 4 스키마를 따름  
- S04, S06, S11, S15 상세 예시는 스펙 섹션 7에 있음  
- S01~S03, S05, S07~S10, S12~S14는 스펙 표의 `특수 조건` 컬럼을 기반으로 작성  
- `candidates` 필드는 `conftest.py`의 `mock_candidates` 스타일로 작성  
  (필수 필드: id, name_kr, name_en, primary_muscle, secondary_muscle, equipment_req, efficiency_tier, movement_type, pain_triggers)

**Step 3**: `tests/evaluation/metrics.py`  
- 섹션 12의 함수 시그니처를 구현  
- 각 함수는 `(score: int, warnings: list[str])` 튜플 반환  
- `estimate_routine_time_min()`은 `engines.prescription.estimator`에서 import  
- `LLMExercisePlan`을 routine_blocks dict로 변환하는 헬퍼 포함  
  (`RoutineBlock`은 `movement_type` 필드가 없으므로 candidates에서 lookup 필요)

**Step 4**: `tests/evaluation/harness.py`  
- `ScenarioResult` dataclass 구현 (섹션 11)  
- `EvalHarness.run(scenario)` 구현 (섹션 11 파이프라인 호출 방식)  
- `engines.routine_pipeline._validate_or_fallback` import  
- `score_candidate_exercises()`로 candidates 먼저 scoring  
- repair 감지: `response.warnings`에 "Guard" 문자열 포함 여부  
- JSON 리포트 생성 헬퍼 `save_report(results, path)` 포함

**Step 5**: `tests/evaluation/test_offline_eval_harness.py`  
- 섹션 8의 pytest parametrize 구조 그대로 구현  
- `--eval-report` CLI 옵션 추가 (선택; `pytest_addoption`은 `tests/evaluation/conftest.py`에 구현)  
- 마커: `@pytest.mark.eval`

### 제약 조건

- LLM API 호출 절대 금지 (`get_llm()` 호출 없음)
- DB 연결 없음 (`get_candidate_exercises()` 호출 없음)
- `mock_llm_output`이 `LLMRoutineOutput.model_validate()` 실패하면 `pytest.skip()` + 경고
- 시나리오 실패 메시지에 항상 `scenario_id` 포함
- `candidates` 필드에 `score`/`score_reasons`가 없으면 `score_candidate_exercises()` 자동 실행
- `_validate_or_fallback`이 import 불가면 `pytest.skip("pipeline import failed")`

### 실행 검증

구현 완료 후 아래 명령이 모두 통과해야 한다:

```bash
# 기존 테스트 깨지지 않음
pytest tests/test_prompt_regression.py tests/test_guard.py tests/test_contract.py tests/test_fallback.py -q

# 새 harness 실행
pytest tests/evaluation/ -v --tb=short

# 전체 (52 + 15 이상 통과)
pytest tests/ -q --ignore=tests/evaluation  # 기존 52 확인
pytest tests/evaluation/ -q                  # 신규 15 확인
```

---

*이 스펙은 P1-1의 산출물이다. 구현은 P1-2에서 Codex가 담당한다.*

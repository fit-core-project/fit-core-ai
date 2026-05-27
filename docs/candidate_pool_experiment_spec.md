# P1-4 Candidate Pool Size 12 → 18 실험 스펙

**버전:** 1.0  
**작성일:** 2026-05-27  
**단계:** P1-4 (실험 설계 — production 변경 없음)  
**상태:** Codex 구현 대기

---

## 1. 설계 요약

AI 루틴 생성기의 ranked candidate pool 크기를 12개에서 18개로 확대했을 때 품질, 다양성, fallback/repair 안정성, prompt token 비용에 어떤 영향이 있는지를 **실제 LLM API 호출 없이** deterministic하게 측정한다.

이번 단계의 목표:
- production default(`top_n=12`)를 유지하면서 실험용 `candidate_limit` 파라미터를 주입한다.
- Offline Evaluation Harness와 새로운 실험 픽스처를 활용해 12 vs 18을 비교한다.
- proxy metric(후보 구성, prompt 크기, Rule Critic score 등)으로 default 변경 여부를 판단한다.

---

## 2. Candidate Limit 위치 분석

### 2-1. 현재 하드코딩 위치

| 파일 | 라인 | 내용 |
|------|------|------|
| `engines/candidate_ranker.py` | 47 | `top_n: int = 12` (함수 파라미터 기본값) |
| `engines/candidate_ranker.py` | 164 | `[:top_n]` (정렬 후 슬라이싱) |
| `engines/routine_pipeline.py` | 186~195 | `score_candidate_exercises(...)` 호출 시 `top_n` 미전달 → 기본값 12 사용 |

### 2-2. 관련 파일 구조

```
engines/
├── candidate_ranker.py       ← top_n 정의 및 슬라이싱
├── routine_pipeline.py       ← score_candidate_exercises() 호출 (top_n 미전달)
├── prompt_builder.py         ← format_candidates_for_prompt() 소비
├── llm_parser.py             ← LLM 출력 파싱 및 guard repair
├── post_validation_guard.py  ← exercise_id 환각/통증/장비 검증
├── rule_critic.py            ← 루틴 품질 평가
└── fallback.py               ← fallback 루틴 생성

tests/evaluation/
├── helpers.py                ← EvalHarness (score_candidate_exercises 직접 호출)
├── fixtures/scenarios.json   ← 15개 시나리오 픽스처
└── experiment/               ← [신규] P1-4 실험 전용 디렉토리
    ├── pool_size_fixtures.json
    └── test_candidate_pool_size.py
```

### 2-3. 후보 payload 구조 (prompt 내)

`format_candidates_for_prompt()` 출력 예시 (후보 1개당 1줄):

```
- 바벨 벤치프레스 (ID: barbell_bench_press, score=95, movement_type=COMPOUND,
  primary=chest, secondary=triceps, equipment=BARBELL, pain_triggers=none,
  reason=compound priority; primary target match; loadable equipment)
```

**후보 1개당 평균 문자 수:** 약 200~280자  
**12개 기준 candidate section:** ~2,400~3,360자  
**18개 기준 candidate section:** ~3,600~5,040자 (예상 증가율 50%)

---

## 3. 실험 범위 / 비범위

### In Scope

- `score_candidate_exercises()` 호출 시 `top_n` 파라미터를 12 또는 18로 주입
- candidate pool 구성(근육군, 장비, 운동 유형) 분포 비교
- `format_candidates_for_prompt()` 출력 문자 수 / 토큰 수 비교
- Rule Critic score 비교 (동일 mock_llm_output 기준)
- guard repair count 비교 (동일 mock_llm_output 기준)
- fallback 유발 시나리오에서 pool 크기별 후보 충분성 비교
- 실험 전용 fixtures(18개 후보 포함)로 candidate shortage 개선 효과 측정

### Out of Scope

- production `top_n` 기본값 변경 (이번 단계 금지)
- 실제 LLM API 호출
- ranking score 로직 변경
- guard/fallback 로직 변경
- feedback 데이터의 ranking 반영
- Rule Critic의 `should_rebuild`/`should_fallback` 활성화
- 외부 네트워크 호출

---

## 4. 12 vs 18 비교 Metric 정의

### A. Candidate Pool Metrics

| Metric | 설명 | 측정 방법 |
|--------|------|-----------|
| `candidate_count` | ranked 후보 총 수 | `len(ranked_candidates)` |
| `target_muscle_candidate_count` | 각 target muscle별 후보 수 | primary_muscle 그룹별 집계 |
| `compound_count` | COMPOUND 운동 수 | movement_type == "COMPOUND" 집계 |
| `isolation_count` | ISOLATION 운동 수 | movement_type == "ISOLATION" 집계 |
| `equipment_distribution` | 장비별 후보 분포 | equipment_req 그룹별 집계 |
| `pain_safe_alternative_count` | pain_triggers=none인 후보 수 | pain_triggers 필드 기준 |
| `recent_exercise_overlap_count` | recent_exercises와 겹치는 후보 수 | id 교집합 크기 |
| `bodyweight_only_count` | BODYWEIGHT 전용 후보 수 | equipment_req == "BODYWEIGHT" 집계 |

### B. Prompt Cost Metrics

| Metric | 설명 | 측정 방법 |
|--------|------|-----------|
| `candidate_payload_char_count` | `format_candidates_for_prompt()` 출력 문자 수 | `len(formatted_string)` |
| `approximate_token_count` | 토큰 추정 수 | `char_count // 4` (근사값) |
| `payload_size_increase_ratio` | 18 대비 12 문자 증가율 | `(18_chars - 12_chars) / 12_chars` |

### C. Quality Proxy Metrics

| Metric | 설명 | 측정 방법 |
|--------|------|-----------|
| `rule_critic_score` | Rule Critic 총점 | `critic.score` |
| `coverage_potential` | target muscle 당 후보 수 (최대 coverage 가능성) | `target_muscle_candidate_count` 기반 |
| `diversity_potential` | 고유 movement_type 수 × 고유 equipment_req 수 | 카테시안 다양성 지수 |
| `compound_ratio` | 전체 후보 중 COMPOUND 비율 | `compound_count / candidate_count` |
| `candidate_shortage_rate` | target muscle 중 후보 0개인 근육군 비율 | 근육별 후보 수 0인 경우 |

### D. Stability Metrics

| Metric | 설명 | 측정 방법 |
|--------|------|-----------|
| `guard_repair_count` | Guard repair 횟수 | `"Guard:" + "replaced"` 포함 warning 수 |
| `fallback_count` | fallback 유발 횟수 | `response.is_fallback == True` 집계 |
| `hard_violation_count` | hard constraint 위반 수 | `len(critic.hard_violations)` |
| `scenario_pass_rate` | 전체 시나리오 pass 비율 | `passed_count / total_count` |

---

## 5. 실험 Fixture 및 테스트 구조

### 5-1. 기존 15개 시나리오의 한계

기존 `scenarios.json`의 각 시나리오는 후보 수가 4~6개로 설정되어 있다.  
`top_n=12`와 `top_n=18`을 비교하려면 **12개 이상의 후보를 가진 시나리오**가 필요하다.  
따라서 실험 전용 픽스처를 별도로 작성한다.

### 5-2. 실험 전용 Fixture: `tests/evaluation/experiment/pool_size_fixtures.json`

```json
[
  {
    "id": "EXP01",
    "name": "large_pool_upper_hypertrophy",
    "description": "18개 이상의 상체 hypertrophy 후보로 12 vs 18 pool 비교",
    "goal": "hypertrophy",
    "target_split_label": "push",
    "target_muscles": ["chest", "front-deltoids", "triceps", "side-deltoids"],
    "duration_min": 60,
    "readiness_level": "normal",
    "unavailable_equipment": [],
    "current_pain_areas": [],
    "doms": {},
    "recent_exercises": [],
    "candidates": [
      // ← 20개 이상의 후보를 포함
      // COMPOUND + ISOLATION, 다양한 equipment (BARBELL, DUMBBELL, CABLE, BODYWEIGHT, MACHINE)
      // 모든 target muscle 커버
    ],
    "mock_llm_output": {
      // ← 12개 후보 기준으로 유효한 선택 (top-12 내의 exercise_id 사용)
    },
    "expected": {
      "must_pass": true,
      "is_fallback": false,
      "expected_repair": false,
      "required_muscle_coverage": ["chest", "front-deltoids", "triceps"],
      "min_compound_count": 2,
      "max_working_sets": 18
    }
  },
  {
    "id": "EXP02",
    "name": "shortage_scenario_lower",
    "description": "하체 후보 부족 시나리오 — 18개 pool이 shortage를 완화하는지 확인",
    "goal": "strength",
    "target_muscles": ["quadriceps", "hamstring", "gluteal", "calves"],
    "duration_min": 60,
    "unavailable_equipment": ["BARBELL"],
    "candidates": [
      // ← 20개 이상, BARBELL 제외 후보 다수 포함
    ]
  },
  {
    "id": "EXP03",
    "name": "pain_area_large_pool",
    "description": "통증 부위 있을 때 18개 pool이 대체 후보를 충분히 제공하는지 확인",
    "goal": "hypertrophy",
    "target_muscles": ["chest", "front-deltoids"],
    "current_pain_areas": [{"area": "shoulder", "severity": "mild"}],
    "candidates": [
      // ← 20개 이상, shoulder pain_triggers 있는 후보 + 없는 대체 후보 혼합
    ]
  },
  {
    "id": "EXP04",
    "name": "doms_large_pool",
    "description": "DOMS 패널티 시 18개 pool이 더 많은 유효 후보를 남기는지 확인",
    "goal": "hypertrophy",
    "target_muscles": ["chest", "triceps"],
    "doms": {"chest": 2, "triceps": 1},
    "candidates": [
      // ← 20개 이상, DOMS 패널티 적용 후에도 충분한 후보 잔류 확인
    ]
  }
]
```

**각 픽스처의 후보 수 요구사항:**
- 원시 후보(ranking 전): 20~25개
- `top_n=12` 적용 후: 12개
- `top_n=18` 적용 후: 18개
- `mock_llm_output`: top-12 범위 내 유효한 exercise_id 사용 (두 조건 공통 유효)

### 5-3. 실험 테스트 파일: `tests/evaluation/experiment/test_candidate_pool_size.py`

```python
"""
P1-4: Candidate Pool Size 12 vs 18 실험

목적:
- production default(top_n=12)를 유지하면서 top_n=18 실험군과 비교
- LLM API 호출 없이 deterministic하게 proxy metric 측정
- 결과를 JSON 리포트로 저장
"""
import json
from pathlib import Path
from typing import Any

import pytest

from engines.candidate_ranker import format_candidates_for_prompt, score_candidate_exercises
from engines.rule_critic import build_eval_context, evaluate_routine_quality
from engines.schemas import LLMRoutineOutput, PainAreaEntry, RecentSetRecord, RoutineRequest
from engines.prescription.adjustments import _calculate_max_total_sets
from engines.routine_pipeline import _validate_or_fallback

FIXTURES_PATH = Path(__file__).parent / "pool_size_fixtures.json"
REPORT_PATH = Path(__file__).parent / "pool_size_experiment_report.json"

POOL_SIZES = [12, 18]


def load_fixtures() -> list[dict[str, Any]]:
    with FIXTURES_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def run_single_scenario(
    scenario: dict[str, Any],
    pool_size: int,
) -> dict[str, Any]:
    """단일 시나리오를 지정된 pool_size로 실행하고 metric을 반환한다."""
    pain_areas = [PainAreaEntry(**item) for item in scenario.get("current_pain_areas", [])]
    recent_sets = [
        RecentSetRecord(exercise_id=eid, exercise_name=eid, reps=8)
        for eid in scenario.get("recent_exercises", [])
    ]

    # pool_size 주입: production default를 변경하지 않고 실험 파라미터로 전달
    ranked_candidates = score_candidate_exercises(
        scenario["candidates"],
        scenario.get("target_muscles", []),
        doms_db=scenario.get("doms", {}),
        blocked_equipment=scenario.get("unavailable_equipment", []),
        pain_areas=pain_areas,
        recent_sets=recent_sets,
        top_n=pool_size,  # ← 실험 파라미터 주입 지점
    )

    # --- Candidate Pool Metrics ---
    candidate_count = len(ranked_candidates)
    target_muscles = scenario.get("target_muscles", [])
    target_muscle_dist = {
        m: sum(1 for c in ranked_candidates if c.get("primary_muscle") == m)
        for m in target_muscles
    }
    compound_count = sum(1 for c in ranked_candidates if c.get("movement_type") == "COMPOUND")
    isolation_count = sum(1 for c in ranked_candidates if c.get("movement_type") == "ISOLATION")
    equipment_dist: dict[str, int] = {}
    for c in ranked_candidates:
        eq = c.get("equipment_req", "UNKNOWN")
        equipment_dist[eq] = equipment_dist.get(eq, 0) + 1
    pain_safe_count = sum(
        1 for c in ranked_candidates
        if not c.get("pain_triggers") or str(c.get("pain_triggers")).lower() in ("none", "null", "")
    )
    recent_names = {r.exercise_name.lower() for r in recent_sets}
    recent_overlap_count = sum(
        1 for c in ranked_candidates
        if str(c.get("id", "")).lower() in recent_names
    )
    bodyweight_count = sum(
        1 for c in ranked_candidates if c.get("equipment_req") == "BODYWEIGHT"
    )
    shortage_muscles = [m for m, cnt in target_muscle_dist.items() if cnt == 0]

    # --- Prompt Cost Metrics ---
    formatted_payload = format_candidates_for_prompt(ranked_candidates)
    payload_char_count = len(formatted_payload)
    approx_token_count = payload_char_count // 4

    # --- Quality Proxy (Rule Critic) ---
    output = LLMRoutineOutput.model_validate(scenario["mock_llm_output"])
    req = RoutineRequest(
        user_id=f"exp-{scenario['id']}-pool{pool_size}",
        target_split_label=scenario.get("target_split_label"),
        target_muscles=target_muscles,
        readiness_level=scenario.get("readiness_level", "normal"),
        time_available_min=scenario["duration_min"],
        pain_areas=pain_areas,
        doms_data=scenario.get("doms", {}),
        equipment=scenario.get("unavailable_equipment", []),
        goal=scenario.get("goal"),
    )
    expected = scenario.get("expected", {})
    default_max_sets = _calculate_max_total_sets(scenario["duration_min"], scenario["goal"])
    max_working_sets = int(expected.get("max_working_sets") or default_max_sets)

    response = _validate_or_fallback(
        output=output,
        req=req,
        ranked_candidates=ranked_candidates,
        max_total_sets=max_working_sets,
        doms_db=scenario.get("doms", {}),
        goal=scenario["goal"],
        db_target_muscles=target_muscles,
        recent_sets=recent_sets,
        profile=None,
    )

    repair_count = len([
        w for w in response.warnings
        if "Guard:" in w and "replaced" in w
    ])
    fallback_used = response.is_fallback or response.generation_status == "fallback"

    context = build_eval_context(req, target_muscles, recent_sets=recent_sets, max_working_sets=max_working_sets)
    context.available_equipment = scenario.get("available_equipment", [])
    context.overrides = dict(expected)

    critic = evaluate_routine_quality(
        response,
        ranked_candidates,
        context,
        repair_count=repair_count,
        fallback_used=fallback_used,
    )

    fail_warnings = [w for w in critic.warnings if w.startswith("FAIL ")]
    passed = (
        not critic.hard_violations
        and critic.score >= 80
        and not fail_warnings
        and fallback_used == bool(expected.get("is_fallback", False))
        and (repair_count > 0) == bool(expected.get("expected_repair", False))
    )

    # --- Diversity Potential ---
    unique_movement_types = len({c.get("movement_type") for c in ranked_candidates})
    unique_equipment = len({c.get("equipment_req") for c in ranked_candidates})
    diversity_potential = unique_movement_types * unique_equipment

    return {
        "scenario_id": scenario["id"],
        "scenario_name": scenario["name"],
        "pool_size": pool_size,
        # Candidate Pool
        "candidate_count": candidate_count,
        "target_muscle_distribution": target_muscle_dist,
        "compound_count": compound_count,
        "isolation_count": isolation_count,
        "equipment_distribution": equipment_dist,
        "pain_safe_alternative_count": pain_safe_count,
        "recent_exercise_overlap_count": recent_overlap_count,
        "bodyweight_only_count": bodyweight_count,
        "shortage_muscles": shortage_muscles,
        "candidate_shortage_rate": len(shortage_muscles) / max(len(target_muscles), 1),
        # Prompt Cost
        "candidate_payload_char_count": payload_char_count,
        "approximate_token_count": approx_token_count,
        # Quality Proxy
        "rule_critic_score": critic.score,
        "hard_violation_count": len(critic.hard_violations),
        "hard_violation_details": critic.hard_violations,
        "metric_scores": critic.metric_scores,
        "diversity_potential": diversity_potential,
        "compound_ratio": compound_count / max(candidate_count, 1),
        # Stability
        "guard_repair_count": repair_count,
        "fallback_used": fallback_used,
        "scenario_passed": passed,
    }


class TestCandidatePoolSizeExperiment:
    """P1-4 Candidate Pool Size 12 vs 18 실험."""

    @pytest.fixture(scope="class")
    def fixtures(self) -> list[dict[str, Any]]:
        return load_fixtures()

    @pytest.fixture(scope="class")
    def results(self, fixtures) -> dict[str, list[dict[str, Any]]]:
        all_results: dict[str, list[dict[str, Any]]] = {}
        for scenario in fixtures:
            scenario_results = []
            for pool_size in POOL_SIZES:
                result = run_single_scenario(scenario, pool_size)
                scenario_results.append(result)
            all_results[scenario["id"]] = scenario_results
        return all_results

    def test_pool_18_candidate_count_is_larger(self, results):
        """top_n=18은 top_n=12보다 항상 같거나 많은 후보를 반환한다."""
        for scenario_id, scenario_results in results.items():
            r12 = next(r for r in scenario_results if r["pool_size"] == 12)
            r18 = next(r for r in scenario_results if r["pool_size"] == 18)
            assert r18["candidate_count"] >= r12["candidate_count"], (
                f"{scenario_id}: pool=18이 pool=12보다 후보가 적음"
            )

    def test_pool_18_no_scenario_pass_regression(self, results):
        """pool=18에서 기존 pass 시나리오가 fail로 바뀌지 않는다."""
        for scenario_id, scenario_results in results.items():
            r12 = next(r for r in scenario_results if r["pool_size"] == 12)
            r18 = next(r for r in scenario_results if r["pool_size"] == 18)
            if r12["scenario_passed"]:
                assert r18["scenario_passed"], (
                    f"{scenario_id}: pool=12 pass였으나 pool=18에서 fail"
                )

    def test_pool_18_rule_critic_score_not_worse(self, results):
        """pool=18의 Rule Critic score가 pool=12 대비 감소하지 않는다."""
        for scenario_id, scenario_results in results.items():
            r12 = next(r for r in scenario_results if r["pool_size"] == 12)
            r18 = next(r for r in scenario_results if r["pool_size"] == 18)
            assert r18["rule_critic_score"] >= r12["rule_critic_score"], (
                f"{scenario_id}: pool=18 critic score({r18['rule_critic_score']}) < "
                f"pool=12 critic score({r12['rule_critic_score']})"
            )

    def test_pool_18_fallback_count_not_increased(self, results):
        """pool=18에서 fallback이 새로 발생하지 않는다."""
        for scenario_id, scenario_results in results.items():
            r12 = next(r for r in scenario_results if r["pool_size"] == 12)
            r18 = next(r for r in scenario_results if r["pool_size"] == 18)
            if not r12["fallback_used"]:
                assert not r18["fallback_used"], (
                    f"{scenario_id}: pool=12에서 fallback 없었으나 pool=18에서 fallback 발생"
                )

    def test_pool_18_payload_size_increase_within_limit(self, results):
        """candidate payload 문자 수 증가율이 60% 이하여야 한다."""
        for scenario_id, scenario_results in results.items():
            r12 = next(r for r in scenario_results if r["pool_size"] == 12)
            r18 = next(r for r in scenario_results if r["pool_size"] == 18)
            if r12["candidate_payload_char_count"] > 0:
                increase_ratio = (
                    r18["candidate_payload_char_count"] - r12["candidate_payload_char_count"]
                ) / r12["candidate_payload_char_count"]
                assert increase_ratio <= 0.60, (
                    f"{scenario_id}: payload 증가율 {increase_ratio:.1%} > 60% 허용 한도"
                )

    def test_shortage_scenario_improves_with_pool_18(self, results):
        """candidate shortage 시나리오에서 pool=18이 shortage_rate를 낮춰야 한다."""
        shortage_scenarios = [
            sid for sid, scenario_results in results.items()
            if any(r["candidate_shortage_rate"] > 0 for r in scenario_results if r["pool_size"] == 12)
        ]
        for scenario_id in shortage_scenarios:
            scenario_results = results[scenario_id]
            r12 = next(r for r in scenario_results if r["pool_size"] == 12)
            r18 = next(r for r in scenario_results if r["pool_size"] == 18)
            assert r18["candidate_shortage_rate"] <= r12["candidate_shortage_rate"], (
                f"{scenario_id}: shortage scenario에서 pool=18이 shortage를 개선하지 못함"
            )

    def test_generate_json_report(self, results):
        """실험 결과를 JSON 리포트로 저장한다."""
        report = _build_experiment_report(results)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with REPORT_PATH.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        assert REPORT_PATH.exists()


def _build_experiment_report(
    results: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """실험 결과를 구조화된 JSON 리포트로 변환한다."""
    scenario_comparisons = []
    aggregate_12: list[dict] = []
    aggregate_18: list[dict] = []

    for scenario_id, scenario_results in results.items():
        r12 = next(r for r in scenario_results if r["pool_size"] == 12)
        r18 = next(r for r in scenario_results if r["pool_size"] == 18)
        aggregate_12.append(r12)
        aggregate_18.append(r18)

        payload_increase_ratio = 0.0
        if r12["candidate_payload_char_count"] > 0:
            payload_increase_ratio = (
                r18["candidate_payload_char_count"] - r12["candidate_payload_char_count"]
            ) / r12["candidate_payload_char_count"]

        scenario_comparisons.append({
            "scenario_id": scenario_id,
            "scenario_name": r12["scenario_name"],
            "pool_12": {k: v for k, v in r12.items() if k not in ("scenario_id", "scenario_name", "pool_size")},
            "pool_18": {k: v for k, v in r18.items() if k not in ("scenario_id", "scenario_name", "pool_size")},
            "delta": {
                "candidate_count": r18["candidate_count"] - r12["candidate_count"],
                "rule_critic_score": r18["rule_critic_score"] - r12["rule_critic_score"],
                "guard_repair_count": r18["guard_repair_count"] - r12["guard_repair_count"],
                "candidate_payload_char_count": r18["candidate_payload_char_count"] - r12["candidate_payload_char_count"],
                "approximate_token_count": r18["approximate_token_count"] - r12["approximate_token_count"],
                "payload_increase_ratio": payload_increase_ratio,
                "diversity_potential": r18["diversity_potential"] - r12["diversity_potential"],
                "candidate_shortage_rate": r18["candidate_shortage_rate"] - r12["candidate_shortage_rate"],
                "pass_regression": r12["scenario_passed"] and not r18["scenario_passed"],
            },
        })

    total = len(aggregate_12)
    pass_rate_12 = sum(1 for r in aggregate_12 if r["scenario_passed"]) / max(total, 1)
    pass_rate_18 = sum(1 for r in aggregate_18 if r["scenario_passed"]) / max(total, 1)
    avg_score_12 = sum(r["rule_critic_score"] for r in aggregate_12) / max(total, 1)
    avg_score_18 = sum(r["rule_critic_score"] for r in aggregate_18) / max(total, 1)
    avg_payload_12 = sum(r["candidate_payload_char_count"] for r in aggregate_12) / max(total, 1)
    avg_payload_18 = sum(r["candidate_payload_char_count"] for r in aggregate_18) / max(total, 1)
    fallback_count_12 = sum(1 for r in aggregate_12 if r["fallback_used"])
    fallback_count_18 = sum(1 for r in aggregate_18 if r["fallback_used"])
    repair_count_12 = sum(r["guard_repair_count"] for r in aggregate_12)
    repair_count_18 = sum(r["guard_repair_count"] for r in aggregate_18)

    verdict = _determine_verdict(
        pass_rate_12=pass_rate_12,
        pass_rate_18=pass_rate_18,
        avg_score_12=avg_score_12,
        avg_score_18=avg_score_18,
        avg_payload_12=avg_payload_12,
        avg_payload_18=avg_payload_18,
        fallback_count_12=fallback_count_12,
        fallback_count_18=fallback_count_18,
        repair_count_12=repair_count_12,
        repair_count_18=repair_count_18,
    )

    return {
        "experiment_id": "P1-4-candidate-pool-size",
        "pool_sizes_tested": POOL_SIZES,
        "total_scenarios": total,
        "summary": {
            "pass_rate_pool_12": pass_rate_12,
            "pass_rate_pool_18": pass_rate_18,
            "avg_rule_critic_score_12": avg_score_12,
            "avg_rule_critic_score_18": avg_score_18,
            "avg_payload_char_12": avg_payload_12,
            "avg_payload_char_18": avg_payload_18,
            "payload_increase_ratio": (avg_payload_18 - avg_payload_12) / max(avg_payload_12, 1),
            "avg_approx_token_12": avg_payload_12 // 4,
            "avg_approx_token_18": avg_payload_18 // 4,
            "fallback_count_12": fallback_count_12,
            "fallback_count_18": fallback_count_18,
            "guard_repair_count_12": repair_count_12,
            "guard_repair_count_18": repair_count_18,
        },
        "verdict": verdict,
        "scenario_comparisons": scenario_comparisons,
    }


def _determine_verdict(
    *,
    pass_rate_12: float,
    pass_rate_18: float,
    avg_score_12: float,
    avg_score_18: float,
    avg_payload_12: float,
    avg_payload_18: float,
    fallback_count_12: int,
    fallback_count_18: int,
    repair_count_12: int,
    repair_count_18: int,
) -> dict[str, Any]:
    """판정 기준에 따라 pool=18을 default로 승격할지 판단한다."""
    payload_increase_ratio = (avg_payload_18 - avg_payload_12) / max(avg_payload_12, 1)

    criteria = {
        "pass_rate_not_regressed": pass_rate_18 >= pass_rate_12,
        "critic_score_not_regressed": avg_score_18 >= avg_score_12,
        "fallback_not_increased": fallback_count_18 <= fallback_count_12,
        "repair_not_increased": repair_count_18 <= repair_count_12,
        "payload_increase_within_limit": payload_increase_ratio <= 0.60,
    }
    all_passed = all(criteria.values())

    return {
        "recommendation": "promote_pool_18_to_default" if all_passed else "keep_pool_12_as_default",
        "all_criteria_met": all_passed,
        "criteria": criteria,
        "payload_increase_ratio": payload_increase_ratio,
        "notes": (
            "모든 판정 기준 통과 — pool=18을 production default로 승격 가능"
            if all_passed
            else "하나 이상의 판정 기준 실패 — pool=12 유지 권장, 실패 항목 확인 필요"
        ),
    }
```

---

## 6. JSON 리포트 형식

실험 실행 후 `tests/evaluation/experiment/pool_size_experiment_report.json`에 저장된다.

```json
{
  "experiment_id": "P1-4-candidate-pool-size",
  "pool_sizes_tested": [12, 18],
  "total_scenarios": 4,
  "summary": {
    "pass_rate_pool_12": 1.0,
    "pass_rate_pool_18": 1.0,
    "avg_rule_critic_score_12": 88.5,
    "avg_rule_critic_score_18": 90.2,
    "avg_payload_char_12": 2840,
    "avg_payload_char_18": 4120,
    "payload_increase_ratio": 0.451,
    "avg_approx_token_12": 710,
    "avg_approx_token_18": 1030,
    "fallback_count_12": 0,
    "fallback_count_18": 0,
    "guard_repair_count_12": 1,
    "guard_repair_count_18": 1
  },
  "verdict": {
    "recommendation": "promote_pool_18_to_default",
    "all_criteria_met": true,
    "criteria": {
      "pass_rate_not_regressed": true,
      "critic_score_not_regressed": true,
      "fallback_not_increased": true,
      "repair_not_increased": true,
      "payload_increase_within_limit": true
    },
    "payload_increase_ratio": 0.451,
    "notes": "모든 판정 기준 통과 — pool=18을 production default로 승격 가능"
  },
  "scenario_comparisons": [
    {
      "scenario_id": "EXP01",
      "scenario_name": "large_pool_upper_hypertrophy",
      "pool_12": {
        "candidate_count": 12,
        "compound_count": 6,
        "isolation_count": 6,
        "rule_critic_score": 88,
        "candidate_payload_char_count": 2760,
        "approximate_token_count": 690,
        "fallback_used": false,
        "scenario_passed": true
      },
      "pool_18": {
        "candidate_count": 18,
        "compound_count": 8,
        "isolation_count": 10,
        "rule_critic_score": 90,
        "candidate_payload_char_count": 4020,
        "approximate_token_count": 1005,
        "fallback_used": false,
        "scenario_passed": true
      },
      "delta": {
        "candidate_count": 6,
        "rule_critic_score": 2,
        "guard_repair_count": 0,
        "candidate_payload_char_count": 1260,
        "approximate_token_count": 315,
        "payload_increase_ratio": 0.457,
        "diversity_potential": 4,
        "candidate_shortage_rate": 0.0,
        "pass_regression": false
      }
    }
  ]
}
```

---

## 7. Default 변경 판단 기준

pool=18을 production default(`top_n=12` → `top_n=18`)로 승격하려면 **아래 조건을 모두 충족**해야 한다.

| 기준 | 조건 | 실패 시 |
|------|------|---------|
| Pass rate | pool=18의 시나리오 pass rate ≥ pool=12 | keep pool=12 |
| Critic score | pool=18의 평균 Rule Critic score ≥ pool=12 | keep pool=12 |
| Fallback | pool=18의 fallback 발생 건수 ≤ pool=12 | keep pool=12 |
| Repair | pool=18의 guard repair 건수 ≤ pool=12 | keep pool=12 |
| Payload size | candidate payload 문자 수 증가율 ≤ 60% | 비용 재검토 |

**추가 고려사항 (실험 이후):**
- pool=18에서 DOMS/통증 필터 이후 실제 유효 후보 수가 유의미하게 증가하는지 확인
- `bodyweight_only_count` 증가 없이 `compound_count`가 증가하는지 확인
- 실제 LLM A/B test(optional, 이번 단계 범위 외)에서 루틴 품질 개선 확인

---

## 8. 리스크

| 리스크 | 영향 | 완화 방안 |
|--------|------|-----------|
| 18개 후보 fixture 작성 부담 | 실험 데이터 품질 저하 | COMPOUND+ISOLATION 각 9개씩 생성 규칙 제공 |
| mock_llm_output이 12개 기준이라 18개 이점 미반영 | Quality proxy 과소평가 | Prompt Cost Metric으로 보완, LLM test는 Phase 2로 미룸 |
| 후보 증가로 guard repair 대상 pool도 확대 | repair 패턴 변화 가능성 | repair_count를 stability metric으로 명시 추적 |
| payload 50%+ 증가로 LLM context 비용 증가 | token cost 상승 | 60% 허용 한도 설정, 초과 시 keep pool=12 |
| 기존 15개 scenario 재활용 불가 | 실험 coverage 부족 | 전용 fixtures 4개 이상 신규 작성으로 보완 |

---

## 9. Codex 구현용 최종 지시문

### 9-1. 구현 범위 요약

**변경 금지:**
- `engines/candidate_ranker.py` line 47의 `top_n: int = 12` 기본값
- `engines/routine_pipeline.py`의 `score_candidate_exercises()` 호출부
- ranking score 로직, guard/fallback 로직, Rule Critic 활성화 로직
- 기존 `tests/evaluation/fixtures/scenarios.json`
- 기존 `tests/evaluation/helpers.py`

**신규 생성:**
1. `tests/evaluation/experiment/` 디렉토리
2. `tests/evaluation/experiment/__init__.py` (빈 파일)
3. `tests/evaluation/experiment/pool_size_fixtures.json`
4. `tests/evaluation/experiment/test_candidate_pool_size.py`

### 9-2. `pool_size_fixtures.json` 작성 요구사항

각 fixture는 다음을 포함해야 한다:
- `id`, `name`, `description`, `goal`, `target_muscles`, `duration_min`
- `readiness_level`, `unavailable_equipment`, `current_pain_areas`, `doms`, `recent_exercises`
- `candidates`: **20개 이상**의 후보 (raw, ranking 전)
  - COMPOUND: 최소 10개 (다양한 target_muscle 커버)
  - ISOLATION: 최소 8개
  - equipment: BARBELL, DUMBBELL, CABLE, BODYWEIGHT, MACHINE 혼합
  - `id`, `name_kr`, `name_en`, `primary_muscle`, `secondary_muscle`, `equipment_req`, `efficiency_tier`, `movement_type`, `pain_triggers` 필드 포함
- `mock_llm_output`: **top-12 기준으로 유효**한 exercise_id만 사용 (12와 18 모두에서 유효)
- `expected`: `must_pass`, `is_fallback`, `expected_repair`, `required_muscle_coverage`, `min_compound_count`, `max_working_sets`

필수 4개 시나리오:
1. `EXP01` — 정상 상체 hypertrophy (20+ 후보, 4개 target muscle)
2. `EXP02` — 하체 장비 제한 (BARBELL 제외, shortage 가능)
3. `EXP03` — 통증 부위 존재 (pain_triggers 혼합)
4. `EXP04` — DOMS 패널티 (DOMS level 1~2 적용)

### 9-3. `test_candidate_pool_size.py` 구현 요구사항

- 위 섹션 5-3의 코드를 그대로 구현한다.
- `POOL_SIZES = [12, 18]`
- `run_single_scenario(scenario, pool_size)` 함수: `top_n=pool_size`를 `score_candidate_exercises()`에 전달
- 6개의 `test_*` 메서드 구현 (섹션 5-3 참조)
- `test_generate_json_report()`: `pool_size_experiment_report.json` 생성
- `_build_experiment_report()` 및 `_determine_verdict()` 함수 구현

### 9-4. 실행 방법

```bash
# 실험 테스트만 실행
pytest tests/evaluation/experiment/test_candidate_pool_size.py -v

# 리포트 생성 후 확인
cat tests/evaluation/experiment/pool_size_experiment_report.json

# 기존 전체 pytest와 함께 실행 (199 passed 유지 확인)
pytest --tb=short -q
```

### 9-5. 성공 기준

- 기존 pytest 199 passed 유지 (회귀 없음)
- 실험 테스트 6개 모두 pass (EXP 시나리오 기준)
- `pool_size_experiment_report.json` 생성 및 `verdict.all_criteria_met` 값 기록
- production `top_n` 기본값 변경 없음 확인

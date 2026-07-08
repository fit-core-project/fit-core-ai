# Routine Recommendation Quality Loop Result

작성일: 2026-06-30  
대상 worktree: `/mnt/d/project-fit-core/worktrees/fit-core-ai-workout-gym-db`  
브랜치: `codex/workout-gym-db-integration-2026-06-30`

## Summary

운동 백과 DB의 확장 재료를 루틴 추천 경로에 연결했다.

- 루틴 후보 조회 결과에 가동성 요구도, 관절 부하, 효과, 난이도, 체형 민감도, 부상 주의도, 대체 운동 후보를 붙인다.
- `RoutineRequest`가 `mobilityLimits`, `conditionPolicies`, `anthropometrySignals`를 받을 수 있게 열었다.
- 후보 스코어러를 `score_breakdown_version=v2`로 올리고, 허리/무릎/어깨/손목 제약, 가동성 제한, 낮은 readiness, 체형 신호, 목표 효과, 자극 대비 피로도를 점수에 반영한다.
- 프롬프트 후보 텍스트에 `jointLoad`, `mobility`, `effect`, `difficulty`, `substitutions`를 압축 노출해 LLM이 설명에 사용할 수 있게 했다.
- 안전/필터/랭킹은 Python/DB가 소유하고 LLM은 구성과 한국어 설명만 담당한다는 정책을 프롬프트에 명시했다.

## Changed Files

- `/mnt/d/project-fit-core/worktrees/fit-core-ai-workout-gym-db/engines/db_queries.py`
- `/mnt/d/project-fit-core/worktrees/fit-core-ai-workout-gym-db/engines/candidate_ranker.py`
- `/mnt/d/project-fit-core/worktrees/fit-core-ai-workout-gym-db/engines/routine_pipeline.py`
- `/mnt/d/project-fit-core/worktrees/fit-core-ai-workout-gym-db/engines/prompt_builder.py`
- `/mnt/d/project-fit-core/worktrees/fit-core-ai-workout-gym-db/engines/schemas.py`
- `/mnt/d/project-fit-core/worktrees/fit-core-ai-workout-gym-db/tests/test_prompt_regression.py`
- `/mnt/d/project-fit-core/worktrees/fit-core-ai-workout-gym-db/tests/test_contract.py`

## Deterministic Checks

검증한 시나리오:

- 요추 디스크/허리 제약이 있을 때 `lumbar_load`/`axial_load`가 높은 후보 감점
- 전문가 허가가 없는 재활/질환 조건에서 고부하 후보 추가 감점
- 발목 배굴 제한이 있을 때 `ankle_dorsiflexion_demand=high` 후보 감점
- 긴 대퇴 신호가 있을 때 `long_femur_sensitivity` 후보 감점
- 목표가 근비대일 때 `hypertrophy_effect`와 `stimulus_to_fatigue`가 좋은 후보 가점
- 프롬프트 후보 payload에 관절 부하/가동성/효과/난이도/대체 운동 후보가 노출되는지 확인
- `test_user` SQLite seed에서 가동성/체형/질환 정책이 읽히는지 확인

## Commands / Evidence

```bash
python3 -m compileall -q engines
```

결과: pass

```bash
python3 -m pytest -q -s --capture=no \
  tests/test_prompt_regression.py \
  tests/test_contract.py \
  tests/test_api_regression.py \
  tests/test_extended_db_quality.py
```

결과: `48 passed, 1 warning`

```bash
python3 -m pytest -q -s --capture=no tests
```

결과: `682 passed, 1 skipped, 1 warning`

```bash
python3 scripts/run_gemma4_quality_seed.py plan \
  --model gemma4:latest \
  --ollama-base-url http://127.0.0.1:11434 \
  --output-dir tests/evaluation/.artifacts/gemma4-quality-seed
```

결과:

- report: `/mnt/d/project-fit-core/worktrees/fit-core-ai-workout-gym-db/tests/evaluation/.artifacts/gemma4-quality-seed/gemma4-quality-seed-20260630T084149Z/gemma4-quality-seed-report.md`
- seed scenarios: 5
- live eval: not run
- blocker: `failed_ollama_unreachable`

```bash
python3 scripts/run_gemma4_evaluation_suite.py preflight
```

결과:

- output: `/mnt/d/project-fit-core/worktrees/fit-core-ai-workout-gym-db/tests/evaluation/.artifacts/gemma4-suite/preflight.json`
- readiness: failed
- primary blocker: `ollama_unreachable`
- DB readiness: `failed_connection`

## Failure Classification

현재 live Gemma 평가가 막힌 원인은 코드 실패가 아니라 환경 blocker다.

- primary category: `environment_blocker`
- reason: Windows Ollama 설치와 실행 파일은 감지되지만 WSL 기준 `http://127.0.0.1:11434`로 접근되지 않음
- next action:
  - Windows Ollama host를 WSL에서 접근 가능하게 열기
  - `OLLAMA_BASE_URL`을 WSL에서 접근 가능한 Windows host IP로 지정
  - readiness probe 재실행

## Developer Copy

이번 변경으로 루틴 추천은 운동 백과 DB의 확장 재료를 실제 추천 경로에서 사용합니다.

- DB 후보에 관절 부하/가동성/효과/난이도/체형 민감도/대체 운동 후보가 붙습니다.
- Python 스코어러가 허리·무릎·어깨·손목 제약, 발목/고관절/어깨/손목 가동성 제한, readiness, 체형 신호, 목표 효과를 deterministic하게 반영합니다.
- LLM 프롬프트에는 이미 계산된 후보 근거만 압축 제공됩니다. 안전 판단은 LLM에게 넘기지 않습니다.
- 전체 테스트는 `682 passed, 1 skipped`로 통과했습니다.
- Gemma live 품질 평가는 아직 Ollama 접근 환경 문제로 실행하지 못했습니다. 코드 실패가 아니라 WSL/Ollama 연결 문제입니다.

## Next Action

1. Windows Ollama를 WSL에서 접근 가능하게 설정한다.
2. `OLLAMA_BASE_URL`을 실제 접근 가능한 host로 지정한다.
3. `scripts/run_gemma4_quality_seed.py template` 또는 live eval을 실행해 base/tuned Gemma 품질을 비교한다.
4. 반복 실패 유형을 `DB 재료 / Python 룰 / Prompt / LLM / Fine-tuning 후보`로 분류해 다음 패치를 결정한다.

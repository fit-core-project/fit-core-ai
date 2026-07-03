# Loop Result: AI Live Eval Harness

작성일: 2026-07-03
실행자: Codex
대상: Fit-Core AI quicklog / supplement
Risk Level: L1/L2
Status: partial

## Summary

- QuickLog live/fallback 평가가 pytest에 묶여 멈추던 문제를 케이스별 timeout harness로 분리했다.
- QuickLog fallback parser는 20개 평가 시나리오 기준 20/20 pass로 개선됐다.
- QuickLog real Gemma4 경로는 `gemma4:latest` 생성이 15초 안에 응답하지 않아 timeout으로 기록됐다.
- Supplement endpoint live 평가는 FastAPI 서버가 꺼져 있어 connection refused로 기록됐다.
- Supplement / QuickLog unit regression은 102개 테스트 기준 pass로 확인했다.

## Inputs

- repo/branch/commit: `codex/workout-gym-db-integration-2026-06-30`
- scenarios:
  - `tests/evaluation/fixtures/quicklog_scenarios.json`
  - `tests/fixtures/supplement_live_eval_cases.json`
- environment:
  - Ollama API reachable at `http://127.0.0.1:11434`
  - `gemma4:latest` installed
  - FastAPI server not running at `http://127.0.0.1:8000`

## Commands

```bash
python3 -m pytest -q -s --capture=no tests/test_diet_parser.py tests/test_quicklog_engine.py
python3 -m pytest -q -s --capture=no tests/test_supplement_response_composer.py tests/test_supplement_query_understanding.py tests/test_supplement_llm_first_policy.py tests/test_supplement_kb_schema.py tests/test_supplement_engine_degraded.py tests/test_diet_parser.py tests/test_quicklog_engine.py
python3 scripts/eval_quicklog_live.py --mode fallback --timeout-sec 10 --run-id quicklog-fallback-20260703-final
python3 scripts/eval_quicklog_live.py --mode real --timeout-sec 15 --limit 1 --run-id quicklog-real-timeout-check-20260703-final
python3 scripts/eval_supplement_live.py --limit 1 --timeout-sec 5 --run-id supplement-endpoint-check-20260703-final
python3 -m compileall engines/quicklog scripts/eval_quicklog_live.py scripts/eval_supplement_live.py tests/test_diet_parser.py
git diff --check
```

## Evidence

| Evidence | Path / Output | Result |
|---|---|---|
| QuickLog unit tests | `tests/test_diet_parser.py`, `tests/test_quicklog_engine.py` | 15 passed |
| Supplement + QuickLog regression | `tests/test_supplement_*.py`, `tests/test_diet_parser.py`, `tests/test_quicklog_engine.py` | 102 passed |
| QuickLog fallback eval | `tests/evaluation/.artifacts/quicklog-live/quicklog-fallback-20260703-final/quicklog_live_eval_summary.md` | 20/20 pass |
| QuickLog real eval smoke | `tests/evaluation/.artifacts/quicklog-live/quicklog-real-timeout-check-20260703-final/quicklog_live_eval_summary.md` | 1/1 timeout |
| Supplement endpoint smoke | `tests/evaluation/.artifacts/supplement-live/supplement-endpoint-check-20260703-final/supplement_live_eval_summary.md` | connection refused |
| Compile/diff check | `compileall`, `git diff --check` | pass |

## Findings

### P0

- 없음.

### P1

- QuickLog real Gemma4 path is not currently evaluable within a short bounded loop. Even a one-case run timed out at 15 seconds.
- Supplement live endpoint cannot be evaluated until the FastAPI server is running.

### P2

- Old pytest-based QuickLog eval can still hang because it runs REAL LLM cases inline. The new script should be used for bounded local evaluation.
- QuickLog fallback now handles Korean quantity units and mixed workout/diet text better:
  - `밥 반 공기`
  - `계란 한 개`
  - `바나나 1개`
  - `벤치프레스 60kg 10회 3세트 하고 닭가슴살 200g`
  - `오늘 운동하고 계란 2개랑 바나나`
  - invalid text such as `asdfjkl` and `내일 날씨 어때`

## Decisions

- 확정:
  - QuickLog fallback 평가 루프는 `scripts/eval_quicklog_live.py --mode fallback`을 사용한다.
  - Supplement endpoint 평가는 `scripts/eval_supplement_live.py`를 사용한다.
- 보류:
  - QuickLog real Gemma4 평가는 Ollama 응답 속도/모델 설정 확인 후 다시 실행한다.
  - Supplement live 평가는 FastAPI 서버 기동 후 다시 실행한다.
- 변경 제안:
  - 기존 `tests/evaluation/test_quicklog_eval.py`는 release/eval 전용으로 두고, 평시에는 bounded script를 사용한다.

## Open Questions

- `gemma4:latest` 첫 생성이 30초 이상 무응답인 원인이 모델 로딩 시간인지, 런타임 자원 부족인지, Ollama/model 설정 문제인지 확인 필요.
- Supplement RAG full mode를 로컬에서 띄울 때 사용할 표준 env와 Chroma DB 경로 확정 필요.

## Next Action

- 완료:
  - QuickLog fallback parser 보강 및 evidence 생성.
  - Supplement endpoint 평가 harness 생성.
- 개발자 전달:
  - 아직 없음. 서버 기동/LLM 런타임 확인 후 한 번에 공유한다.
- 사용자 승인 필요:
  - 원격 push 여부.
- 다음 루프:
  - `FastAPI server on -> supplement live 5-case smoke -> QuickLog real 3-case smoke`.

## Memory Candidates

- LLM live eval은 pytest inline 호출보다 subprocess/HTTP timeout harness로 분리해야 한다.

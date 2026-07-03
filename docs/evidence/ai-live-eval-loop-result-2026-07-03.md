# Loop Result: AI Live Eval Harness

작성일: 2026-07-03
실행자: Codex
대상: Fit-Core AI quicklog / supplement
Risk Level: L1/L2
Status: pass_with_observations

## Summary

- QuickLog live/fallback 평가가 pytest에 묶여 멈추던 문제를 케이스별 timeout harness로 분리했다.
- QuickLog fallback parser는 20개 평가 시나리오 기준 20/20 pass로 개선됐다.
- QuickLog real Gemma4 경로는 warmup 후 3개 평가 시나리오 기준 3/3 pass로 확인됐다.
- Supplement endpoint live 평가는 curated JSON KB fallback full mode 기준 10개 평가 시나리오에서 10/10 pass로 확인됐다.
- Chroma index와 HuggingFace cache가 없어도 로컬 `data/documents/supplement_*.json` 문서로 근거 있는 full 응답을 만들도록 보강했다.
- Supplement / QuickLog unit regression은 103개 테스트 기준 pass로 확인했다.

## Inputs

- repo/branch/commit: `codex/workout-gym-db-integration-2026-06-30`
- scenarios:
  - `tests/evaluation/fixtures/quicklog_scenarios.json`
  - `tests/fixtures/supplement_live_eval_cases.json`
- environment:
  - Ollama API reachable at `http://127.0.0.1:11434`
  - `gemma4:latest` installed
  - FastAPI server running at `http://127.0.0.1:8000` during supplement smoke
  - `data/chroma_db/latest_index` absent, curated JSON fallback KB used

## Commands

```bash
python3 -m pytest -q -s --capture=no tests/test_diet_parser.py tests/test_quicklog_engine.py
python3 -m pytest -q -s --capture=no tests/test_supplement_response_composer.py tests/test_supplement_query_understanding.py tests/test_supplement_llm_first_policy.py tests/test_supplement_kb_schema.py tests/test_supplement_engine_degraded.py tests/test_diet_parser.py tests/test_quicklog_engine.py
python3 scripts/eval_quicklog_live.py --mode fallback --timeout-sec 10 --run-id quicklog-fallback-20260703-final
python3 scripts/eval_quicklog_live.py --mode real --timeout-sec 15 --limit 1 --run-id quicklog-real-timeout-check-20260703-final
python3 scripts/eval_quicklog_live.py --mode real --timeout-sec 60 --limit 1 --num-predict 256 --num-ctx 2048 --run-id quicklog-real-1case-60s-20260703
python3 scripts/eval_quicklog_live.py --mode real --timeout-sec 75 --limit 3 --num-predict 256 --num-ctx 2048 --run-id quicklog-real-3case-75s-20260703
python3 scripts/eval_supplement_live.py --limit 1 --timeout-sec 5 --run-id supplement-endpoint-check-20260703-final
python3 scripts/eval_supplement_live.py --limit 5 --timeout-sec 12 --run-id supplement-live-curated-json-full-v6-20260703
python3 scripts/eval_supplement_live.py --limit 10 --timeout-sec 12 --run-id supplement-live-curated-json-full-10case-20260703
python3 -m compileall engines/supplement engines/quicklog scripts/eval_supplement_live.py scripts/eval_quicklog_live.py tests/test_supplement_engine_degraded.py
git diff --check
```

## Evidence

| Evidence | Path / Output | Result |
|---|---|---|
| QuickLog unit tests | `tests/test_diet_parser.py`, `tests/test_quicklog_engine.py` | 15 passed |
| Supplement + QuickLog regression | `tests/test_supplement_*.py`, `tests/test_diet_parser.py`, `tests/test_quicklog_engine.py` | 103 passed |
| QuickLog fallback eval | `tests/evaluation/.artifacts/quicklog-live/quicklog-fallback-20260703-final/quicklog_live_eval_summary.md` | 20/20 pass |
| QuickLog real eval smoke | `tests/evaluation/.artifacts/quicklog-live/quicklog-real-3case-75s-20260703/quicklog_live_eval_summary.md` | 3/3 pass |
| Supplement endpoint strict smoke | `tests/evaluation/.artifacts/supplement-live/supplement-live-curated-json-full-v6-20260703/supplement_live_eval_summary.md` | 5/5 pass |
| Supplement endpoint expanded strict smoke | `tests/evaluation/.artifacts/supplement-live/supplement-live-curated-json-full-10case-20260703/supplement_live_eval_summary.md` | 10/10 pass |
| Compile/diff check | `compileall`, `git diff --check` | pass |

## Findings

### P0

- 없음.

### P1

- Local Gemma4 has material latency. Readiness probe needed about 18.7s before warmup, and QuickLog real 3-case smoke took about 7.5s to 9.2s per case after warmup.
- Supplement full mode currently relies on curated JSON fallback KB when Chroma index or HuggingFace local model cache is missing. This is acceptable for local demo/eval, but should be explicit in ops docs.
- Supplement LLM generation often times out and falls back to deterministic KB composition. Output quality is acceptable for the 10-case known-single smoke, but broader multi/drug/condition cases still need expansion.

### P2

- Old pytest-based QuickLog eval can still hang because it runs REAL LLM cases inline. The new script should be used for bounded local evaluation.
- Supplement strict eval now fails `degraded` responses by default. Use `--allow-degraded` only for endpoint-survival smoke.
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
  - Supplement 품질 평가는 full mode + sources required를 기본으로 한다.
  - Chroma index가 없을 때는 curated JSON KB fallback을 사용하되 `mode=full`과 source evidence를 유지한다.
- 보류:
  - Supplement 57-case 전체 라이브 평가는 다음 루프에서 카테고리별로 확장한다.
  - Gemma4 latency 최적화는 별도 성능 루프로 분리한다.
- 변경 제안:
  - 기존 `tests/evaluation/test_quicklog_eval.py`는 release/eval 전용으로 두고, 평시에는 bounded script를 사용한다.

## Open Questions

- Supplement full smoke의 다음 범위는 `multi_supplement`, `drug_interaction`, `condition_risk` 중 어디까지를 MVP 품질 기준으로 잠글지 결정 필요.
- `SUPPLEMENT_LLM_TIMEOUT_SEC=5`는 로컬 테스트에서는 유용하지만, 운영/시연 기본값으로 둘지는 별도 결정 필요.

## Next Action

- 완료:
  - QuickLog fallback parser 보강 및 evidence 생성.
  - Supplement endpoint 평가 harness 생성.
  - Supplement full mode curated JSON fallback 및 strict live smoke 통과.
  - QuickLog real Gemma4 3-case smoke 통과.
- 개발자 전달:
  - 아직 없음. 서버 기동/LLM 런타임 확인 후 한 번에 공유한다.
- 사용자 승인 필요:
  - 원격 push 여부.
- 다음 루프:
  - `supplement category expansion -> 57-case partial/full eval -> response quality fixes`.

## Memory Candidates

- LLM live eval은 pytest inline 호출보다 subprocess/HTTP timeout harness로 분리해야 한다.

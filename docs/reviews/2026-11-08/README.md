# AI 품질 리뷰 및 수정 작업 기록 (2026-11-08, Claude Code)

Claude Code 세션에서 진행한 AI 실측 분석과 1~6번 개선 작업 기록입니다.

## 파일
- `ai-quality-review-2026-11-08.md`: 수정 전 실측 리뷰 (문제점, 근거, 개선 우선순위 1~6)
- `STATUS.md`: 수정 후 재평가 진행 상태 (중단 시점 기준)
- `eval/`: 평가 스크립트와 결과
  - `routine_eval.py`, `quicklog_eval.py`: 로컬 서버(gemma4)에 직접 요청하는 실측 스크립트
  - `routine/`, `quicklog/`, `supplement_live_eval_*_2014.*`: 수정 전 기준선
  - `routine2/`, `quicklog2/`, `supplement_live_eval_*_2157.jsonl`: 수정 후 (일부만 완료)

## 수정 내용 (우선순위 1~6)
1. **custom 분할**: 분할 라벨이 매핑되지 않으면 `targetMuscles`로 후보를 찾음 (`engines/routine_pipeline.py`).
2. **식단 칼로리 정확도**
   - 음식 검색 리랭커가 부분 문자열로 엉뚱한 음식을 고르는 문제를 수정 (예: 콜라 → 쇼콜라 케이크).
   - 단위 테이블(`engines/food/food_units.py`)로 개/공기/조각을 그램으로 환산.
   - LLM 매크로 sanity check와 큐레이션 테이블 우선 적용 (`engines/quicklog/diet_parser.py`).
3. **목표 중량**
   - 우선순위: 같은 운동 기록 > 본인 기준 > 같은 근육 기록 > 빅4 기준 > LLM 힌트.
   - 장비·동작 계수로 환산 (예: 덤벨은 한 손 기준 0.4, 고립 운동 0.55).
   - 무게는 Epley 공식에 반복 횟수와 RIR을 함께 넣어 계산 (`engines/prescription/weight.py`).
4. **루틴 검증**
   - 같은 운동이 두 번 나오면 남은 후보로 교체하고, 대체할 후보가 없으면 뺌.
   - 분할의 목표 근육이 최소 한 번은 들어가도록 보정.
   - 세트 예산은 뒤쪽 운동부터 2세트까지 줄이고, 그래도 넘치면 운동을 뺌 (`engines/llm_parser.py`).
5. **통증과 fallback 문구**
   - 프로필에 등록된 통증 부위도 후보 필터에 반영.
   - fallback 사유별 안내 문구를 분리하고, fallback 루틴도 시간 예산을 지키도록 함 (`engines/fallback.py`).
6. **영양제 상담**
   - 웹 검색이 필요하다고 판단된 질문도 보유 문서를 근거로 답변.
   - 성분명은 단어 경계로 매칭 (예: 수술의 '술', 웨이트의 '웨이'가 더 이상 성분으로 잡히지 않음) (`engines/supplement/`).

## 테스트
- 새 테스트 파일:
  - `tests/test_routine_quality_fixes.py`
  - `tests/test_food_diet_calorie_accuracy.py`
  - `tests/test_supplement_answer_depth.py`
- 단위 테스트 전체: 1037 passed, 19 skipped.
- 의도적으로 바꾼 기존 기대값:
  - `tests/test_fallback.py`: 플라이 무게 72.5 → 15.0
  - `tests/test_pipeline.py`: 중량 85.0 → 87.5, 80.0 → 82.5

## 남은 일
- 식단·영양제 실측 재평가 다시 실행. 이번 실행은 GPU 경합으로 시간 초과가 나서 중단됨.
- `routine2` 결과를 기준선과 상세 비교: 무게, 근육 커버리지, 중복.
- 알려진 미수정 사항:
  - 후보 운동이 0개여도 LLM을 호출함.
  - `_doc_is_relevant_for_caution`이 문서 ID를 부분 문자열로 매칭함.

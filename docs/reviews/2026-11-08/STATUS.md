# AI 실측 재평가 상태 (2026-11-08, 중단)

- 1~6번 수정은 fit-core-ai 작업 트리에 반영됨 (미커밋). 단위 테스트 1037 passed / 19 skipped.
- eval/routine2/: 수정 후 루틴 실측 11케이스 x 2회 완료.
  - rep0 앞 7건은 Ollama 연결 실패(networkError, GPU 경합)로 fallback.
  - 나머지 15건은 success, 하드 제약 위반 0건. custom(R11) 성공.
  - 무게/근육 커버리지/중복 상세 검토는 아직 안 함.
- eval/quicklog2/: D1, D2만 완료. D3에서 타임아웃으로 중단 (GPU 경합).
- eval/supplement2.log: 전부 타임아웃. 서버가 임베딩 단계에서 멈춤.
- diet_parser 큐레이션 테이블 우선 수정은 실측 서버에 반영 안 됨.
- 남은 일: GPU 여유 있을 때 서버 재기동 → quicklog/supplement 재측정 → routine2 상세 비교 → 커밋 여부 결정.
- 기준선(수정 전): eval/routine/, eval/quicklog/, eval/supplement_live_eval_*_2014.*, ai-quality-review-2026-11-08.md

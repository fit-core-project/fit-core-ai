# Legacy fail-closed 적용 범위

2026-10-16 / D1 승인 반영 / **BE 및 AI HTTP adapter 필수 구현 작업**.
정책은 확정됐지만 이번 문서 패키지에서 코어 런타임을 수정하지 않았다. 공통 코어의 작은 조건문 수정이 아니라 HTTP 응답·조회 소유권·두 진입 경로를 함께 연결해야 하는 범위이기 때문이다.

## 현재 코드 근거

AI 서버 제공 기준 커밋 `e0d04e7952cb8e63c36e727f51abe29d17d400f7`의 상대 경로다.

| 지점 | 현재 상태 | 필요한 변경 |
|---|---|---|
| `main.py:240` api_generate_routine | 기존 HTTP는 RoutineRequest를 받아 공통 generate_smart_routine을 호출, RoutineDraftResponse 반환 | 인증/public 변환·typed 조회·legacy 사전/사후 검사·외부 오류 적용 |
| `engines/db_queries.py:1380` legacy loader | typed 결과의 profile만 꺼내 오류와 없음 정보가 유실될 수 있음 | 어댑터가 found/not_found/error를 유지 |
| `engines/routine_pipeline.py:2980` 문맥 로딩 | legacy/v4/내부 호출이 공유하는 경로 | 공통 pipeline에 legacy 전용 capability 제한을 넣지 않음 |
| `engines/routine_generation_choice.py:248` 저장 문맥 | choice facade도 legacy loader 사용 | 해당 경로를 공개 연결한다면 동일 typed 조회·실효 제약 보존 |
| `engines/routine_connection_boundary.py` | checked-v4 및 legacy 검사 helper 존재, HTTP 적용은 없음 | helper를 라우팅/오류 adapter에 연결. v4 envelope는 구 response_model과 호환되지 않음 |

checked-v4로 함수 한 줄만 바꾸면 legacy 응답 형식이 달라진다. 공통 pipeline에 legacy 검사를 넣으면 정상 v4 그룹·방법까지 차단될 수 있다. 내부 SQLite 오류는 현 schemaError 응답만으로 machine-readable 원인을 복원할 수 없으므로 D2 typed 진단 경계도 함께 필요하다. 이 때문에 이번에는 BE 연결 요구사항으로 두고 AI HTTP adapter 협업 범위를 명시한다.

## 구현 순서와 수용 기준

1. BE 자체 프로필 및 AI 내부 제약 조회의 typed 오류를 구별한다. error에서 모델·생성·fallback 0호출, D2의 HTTP503 오류를 반환한다. 정상 조회 후 행 없음과 구분한다.
2. legacy 사전 검사는 원시 FE 입력만이 아니라 저장 문맥을 합친 **실효 요청**에 적용한다. 요청에서 생략한 저장 제약도 보존한다.
3. 확인한 문맥을 생성 경계에 전달한다. FE가 보낸 성공 플래그를 신뢰하거나 생성 중 빈 문맥으로 다시 해석하지 않는다.
4. 생성 후 자동 운동방법·상세 부위·그룹·짧음 안내 등도 legacy 표현 가능성을 검사한다. 불가능하면 HTTP422 capability 오류이며 확장 필드 삭제/자동 강등은 금지한다.
5. 별도 v4 경로에서는 checked-v4·해당 envelope를 사용한다. 일반 후보 없음의 도메인 failed와 조회 오류를 섞지 않는다.
6. 외부 응답에 raw exception/SQL/stack를 노출하지 않는다. schemaError 전체 또는 한국어 문구를 조회 오류로 추정하지 않는다.

필수 테스트: connection/query 오류 각각 0생성, stored-only 제약 보존, 요청 검사 후 생긴 자동 확장의 사후 거절, 표현 가능한 legacy 정상, v4 그룹/방법 정상, 버전 불일치 및 강등 금지, 명시 재시도 시 최신 제약·숫자 ID 보존. BE04·07·FE09의 실제 HTTP 증거로 닫는다.

## 이번 검증 범위

런타임·DB 무변경으로 기존 회귀 4,119 통과 및 재생81/강제장애81 결과를 재사용한다. 이번에 전체를 새로 실행했다고 주장하지 않는다. 패키지 스키마·합성 golden·파일 해시를 새로 검사한다. HTTP adapter 구현 시 전체 회귀와 재생/장애81 차이 전수 및 위 HTTP 테스트를 별도로 수행해야 한다.

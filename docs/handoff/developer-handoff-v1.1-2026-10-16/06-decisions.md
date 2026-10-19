# 연결 정책 결정 기록

결정일: **2026-10-16**. 사용자 결정: v1의 D1~D7 권장안 모두 채택. 이전 패키지는 보존하고 v1.1에 확정 정책을 반영한다. 정책 승인과 코드 구현·실서비스 배포는 별개다.

| ID / 우선순위 | 확정 내용 | 근거·이유 | 구현·완료 확인 |
|---|---|---|---|
| D1 필수 | legacy도 조회 실패 fail-closed. 실효 요청/생성 후 응답 capability 검사. 확장값 제거·자동 강등 금지 | 조회 오류를 제약 없음으로 해석하거나 소비자 유실을 성공으로 표시하면 안 됨 | 검사 helper 존재. HTTP adapter 적용은 미구현. BE04·07, 07-scope의 사전/사후 거절 테스트 |
| D2 필수 | 외부 PROFILE_LOOKUP_UNAVAILABLE, HTTP503. 자동 생성 재시도 없음, 명시 재시도 제공. 입력오류422/후보없음 도메인 failed와 구별 | 기존 routine reason enum을 임의 확장하지 않고 원인을 기계적으로 구별 | BE 소유 및 AI 내부 조회의 typed 진단 경계 구현 필요. 문구 비교/schemaError 일괄 변환 금지. BE04·13 |
| D3 필수 | 별도 `{error:{code,messageKo,retryable,requestId}}`. capability 오류에는 requiredContractVersion/unsupportedFields 제공. 비밀·stack·SQL 미노출 | 성공 routine DTO와 전송 오류 구별, 사용자 안내 및 원인 추적 | 신규 error schema/합성 fixture 제공. 실제 HTTP 및 FE 분기는 미구현. BE01·07, FE09 |
| D4 필수 | 명시적 v4 경로와 응답 contractVersion 검사. 현재 wire `routine-core-v4-draft-v3` 유지, unsupported 거절, 자동 legacy fallback 금지 | 동일 요청의 의미·처방 단위가 버전 사이에서 손실되지 않도록 함 | 기존 초안 경로 `/v4/routines/generate` 기준, 서비스 prefix는 연결 문서에 기록. 실제 라우트/버전 음성 검증 필요 |
| D5 필수(고급 기능 연결 시) | focus/method/관리 부상 선택은 별도 버전화된 폐쇄형 allowlist adapter로 수용. 서버 정책·이력·검수 여부를 FE에서 받지 않음 | 내부 dict 노출과 사용자에 의한 신뢰 경계 우회 방지 | adapter/schema 구현 전에는 현재 public 입력만 제공, 고급 기능은 코어 지원/웹 미연결. BE11·FE10의 값 보존·상충·권한 검사 |
| D6 필수 | 승인 연결·실제 admission·원 숫자 ID 역변환까지 검증한 범위만 초기 제공. 미연결 V24ROW 저장 거절 | identity 승인과 추천 허용은 다름. N:1 이력을 임의 병합할 수 없음 | 소비자 연결 시 제공 카탈로그 manifest 작성, 생성·대체·저장·이력 왕복. 승인11 전체 활성화 아님. BE02 |
| D7 권장 | []의 저장 장비 상속 유지. 맨몸은 BODYWEIGHT 명시. 별도 장비0개 상태는 차후 계약으로 분리 | 기존 입력 의미 보존, 빈 목록의 오해 방지 | BE09·FE06: 누락/[]/BODYWEIGHT 및 저장값 해석 비교 |

## D2의 typed 원인 경계

BE가 소유하는 조회 오류는 코어 호출 전에 차단한다. AI 내부 SQLite 조회 오류도 typed 진단/예외를 adapter에 전달해야 한다. **현재 checked 코어는 failed/schemaError를 반환하며 별도 외부용 typed 오류 전달은 아직 없다.** 안내 문구 비교 또는 schemaError 전체의 PROFILE_LOOKUP_UNAVAILABLE 변환은 금지한다. 내부 진단은 외부 routine reason enum을 조용히 확장하는 방식으로 해결하지 않는다.

503의 retryable=true는 명시 재시도를 제공한다는 뜻이지 자동 반복 호출 허가가 아니다. 명시 재시도 때도 최신 프로필·제약을 다시 조회한다. capability 오류는 HTTP422/retryable=false와 요구 버전/미지원 필드를 제공한다. 일반 입력 오류는422, 정상 조회 뒤 후보 없음은 도메인 failed로 구분한다.

## 이번 승인에 포함하지 않은 것

편측 registry·케이블 세팅 입력·킥백/바벨 힙 쓰러스트 admission 활성화, 미승인 ID 연결, DB 변경은 이번 결정으로 허용되지 않는다. 해당 비활성/보류 상태는 그대로다. D5 상세 schema와 D6 제공 manifest는 개발 산출물이며 검증 없이 기능을 노출하거나 범위를 넓히지 않는다.

## 추적

근거 기록: 2026-10-16 사용자 메시지의 "06-decisions D1~D7 전부 권장안 채택". v1의 결정 제안은 원본 보존. v1.1의 통합 계약·BE/FE 요청서·검증표가 확정 정책의 구현 기준이다. 이번 패키지 제작은 협업 저장소·배포·웹 통합을 변경하지 않았다.

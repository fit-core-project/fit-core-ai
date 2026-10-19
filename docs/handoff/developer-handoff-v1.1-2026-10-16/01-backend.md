# BE 연결 요청

현재 코어 기능을 손실 없이 연결하기 위한 요청이다. 구현 방식은 BE에서 정하되 아래 결과 조건을 충족한다. **D1~D7은 2026-10-16 사용자 승인으로 확정**됐다. 기존 서비스는 호환 검사와 별도 v4 경로를 거쳐 연결하며 조용히 DTO/라우트를 교체하지 않는다.

| ID / 우선순위 | 요청·이유 | 완료 확인 |
|---|---|---|
| BE01 필수 | v4 envelope와 routine/block/prescription 전체 DTO 수용. 세트 단위·그룹·방법·상세값·안내 유실 방지 | `transport-golden.json` 전체를 Jackson→저장→조회→JSON으로 왕복. [검증 스크립트](04-verification.md) 비교 0차이. runtime generated ID 외 값을 임의 삭제하지 않음 |
| BE02 필수 | 숫자 ID 원형 유지 + 승인 연결표 기반 enrichment, stale/모호 연결 차단. catalog ID를 DB PK처럼 저장하지 않음 | stage2 numeric fixture. 74/124 각각의 이력 유지, source/target 변경·revoked·경합 승인 시 메타데이터 적용 안 됨. 125/151 identity와 admission 분리 |
| BE03 필수 | 외부 폐쇄형 입력 검증 후 내부 변환. 인증 userId·프로필·이력은 서버 공급 | `public-input-golden.json`, `test_stage2_input_boundary_v1.py`. 위조 userId/의료정책/내부 플래그 거절. DOMS20 정확 보존, 0/head/knees 거절 |
| BE04 필수 | checked-v4 연결 및 typed 프로필 조회. 저장 부상/제약과 요청 합집합. 조회 실패를 []로 정상화하지 않음. **안내 문구 문자열로 오류 분기 금지** | `test_stage2_connection_boundary_v1.py`: error에서 모델/생성/fallback 0호출. 외부는 D2의 HTTP503/PROFILE_LOOKUP_UNAVAILABLE. BE 자체 조회와 AI 내부 조회 오류 모두 typed 원인을 보존하는 adapter 구현 필요. 현 failed/schemaError 전체를 조회 오류로 매핑하지 않음 |
| BE05 필수 | BE fallback도 동일 정책·장비·부상·DOMS·구성·처방 검증. 검사 불가이면 실패 | BF01~20 모두 Java/실 DB/장애 주입으로 증거. 코어 oracle 통과를 BE 통과로 대신하지 않음 |
| BE06 필수 | v4 무손실 저장·편집·교체·대체 후보·재실행·이력 조회 | F02 timed / F03 distance / F04 interval / F05 그룹+방법의 신규 통합 테스트 통과. 역사적 실패 파일은 삭제/xfail하지 않음 |
| BE07 필수 | legacy도 조회 실패 fail-closed. 저장 문맥을 합친 요청/생성 후 응답 두 단계 표현 검사. v4→legacy 강등·필드 삭제 금지 | 표준 가능 예 + 사전/사후 확장 거절. typed 조회 실패0호출. HTTP422 capability 오류. [07 구현 범위](07-legacy-implementation-scope.md)의 음성 사례 모두 확인 |
| BE08 필수 | shortRoutineNotice를 계산 결과 그대로 전달. 확인 후 새 부위 요청은 서버 최신 상태로 재검증 | 20/50=true,25/50=false,empty/failed=false,nonempty0분 오류. 오래된/위조 옵션·false동의 거절. 나머지 제약 불변 |
| BE09 필수 | whitelist와 blacklist 분리, MACHINE/SMITH_MACHINE 분리 | MACHINE-only에서 스미스 불가, 반대 조건·중첩 거절, 요청 []→저장 목록 해석 일치. BODYWEIGHT 명시 유지 |
| BE10 필수 | 공개 18개 + primaryMuscleDetails 독립 보존, DOMS/부상 범위 구별 | side-deltoids-only []+상세값 왕복. 상세값을 front-deltoids로 바꾸지 않음. 좌우 soleus DOMS 합치지 않음 |
| BE11 필수(해당 기능 연결 시) | focus/method/관리 부상 등 아직 public 모델에 없는 사용자 선택은 별도 버전화된 폐쇄형 adapter로 연결(D5) | internal-request.schema 전체를 외부 허용하지 않음. UI→BE→코어 값 비교·상충·권한 음성 테스트. adapter/schema 구현 전에는 해당 기능을 웹에 노출하지 않음 |
| BE12 필수 | policy/catalog/bridge/source 버전 기록, 실패 사유·fallback 구분 | 각 요청 evidence에 fixture ID·버전·제약 결과·status 기록. token/개인 프로필 원문 로그 없음. 정책 자료 부재시 BF19 실패 |
| BE13 권장 | 타임아웃·재시도·응답 지연 지표를 원인별 수집 | D2 승인 정책 안에서 중복 생성/중복 저장 없이 테스트. timeout을 emptyCandidate로 바꾸지 않음 |
| BE14 권장(활성화 전 필수) | 편측 확장 필드 무손실 보관·예약 처리 | 합성 unilateral golden 왕복. 현재 registry 비활성 유지, unsupported 단위는 강등하지 않음 |

## 연결 순서와 종료 조건

1. 확정 계약과 현재 develop 대조, D6 조건을 만족하는 숫자 제공 범위를 manifest로 고정(승인11 전체를 자동 활성화하지 않음).
2. DTO·입력 adapter·checked 경계·저장/조회 구현. 전송 fixture부터 통과.
3. 생성/편집/교체/이력·fallback 전 경로를 실제 Java와 DB에서 검증.
4. FE와 사용자 선택→표시→실행→재조회, 오류 상태를 E2E 확인.

코어 승인11쌍에서 합성 transport/이력 double이 통과한 것과 실제 11쌍 모두 모델이 선택·저장된 것은 다르다. 새 범위의 생성/대체 가능 운동이 numeric DB에서 식별되지 않으면 이름 매칭으로 해결하지 말고 D6의 미지원 처리로 분리한다.

배포·DB migration·라이선스 확인·운영 토큰 설정은 이 문서에서 실행하지 않았다. 완료 증거에는 사용한 develop SHA, migration 버전, 실제 오류 응답, F02~F05 신규 통과 결과, BF20군별 결과를 남긴다.

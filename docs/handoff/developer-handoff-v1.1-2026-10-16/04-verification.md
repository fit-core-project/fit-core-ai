# Golden 및 검증 방법

모든 예시는 합성 데이터다. 코어 회귀, 전송 왕복, 실제 BE/FE 통합을 별도로 기록한다. 이전 golden/실패 증거는 덮어쓰지 않는다.

## 1. 패키지 자체 검사

패키지 폴더에서 Python 3으로 실행한다. 네트워크/DB/모델 호출 없음.

```bash
python3 -B verify_transport.py
```

파일 SHA-256, 합성 전송 golden 13개, BF01~20 색인, approved11 연결 수를 검사한다. **이 명령만 통과하면 Java/DB 검증이 됐다는 뜻은 아니다.** 3개 모델 schema와 기존 golden은 v1에서 그대로 보존했다. 원본 생성 방법은 AI 서버 저장소의 `docs/handoff/developer-handoff-v1-2026-10-16/build_assets.py`에 있다. 재생성은 새 폴더로 하며 실패를 덮지 않는다.

승인된 신규 오류 envelope는 `schemas/external-error.schema.json`, 합성 HTTP 예시는 `fixtures/external-error-golden.json`이다. `python3 -B verify_external_errors.py`로 양성/음성 검사를 실행한다. 실제 HTTP 서버가 이 응답을 낸다는 증거는 아니며 BE04·07/FE09에서 연결 후 재검증한다.

| 전송 fixture | 확인할 의미 | 연결 요청 |
|---|---|---|
| working | 기본 반복·상태·숫자 ID | BE01·02, FE01 |
| F02-timed | targetDurationSec 보존, 횟수 오표시 금지 | BE06, FE01 |
| F03-distance | targetDistanceM 보존 | BE06, FE01 |
| F04-interval | work/recovery/rounds/total 보존 | BE06, FE01 |
| F05-group-method | 두 블록 묶음 원자성·그룹 휴식·정지 방법 | BE06, FE02·03 |
| detail | public18에 없는 side-deltoids 상세값 | BE10, FE07 |
| unilateral-reserved | 양쪽 완료1세트·한쪽당 반복 | BE14, FE11; 추천 활성화 아님 |
| short-20-of-50 / exact-half | 50% 미만과 정확 50% 차이 | BE08, FE04 |
| confirmed-context-options | 서버 확인된 합성 문맥에서만 선택지 | BE08, FE05; 자동 생성 아님 |
| failed-empty / lookup-failed | 빈 실패0분, 실패와 짧은 성공 구별 | BE04, FE09 |
| fallback-network | fallback 상태·사유 유지 | BE05, FE09 |

`fixtures/public-input-golden.json`은 정상3/거절7 입력과 정상 입력의 정확 내부 변환을 포함한다. 0단계 DOMS·무릎 DOMS·가자미근 타깃·위조 userId·미지원 focus·장비 중첩·low+서킷 거절을 확인한다. public schema만 검사하지 말고 아래 코어 validator도 사용한다.

## 2. 실제 BE 왕복

13개 응답을 실제 DTO→DB 저장→조회→DTO 직렬화 경로에 통과시킨 뒤 아래 배열 형태로 내보낸다. label/id는 fixture 식별자이며 실제 requestId를 대체하는 서비스 필드가 아니다. 응답 값의 합성 ID는 테스트에서 고정해 비교한다.

```json
[{"id":"F02-timed","response":{"contractVersion":"...","routine":{},"shortRoutineNotice":{}}}]
```

위 JSON은 **형태 설명**일 뿐 유효 fixture가 아니다. 실제 입력/기대값은 `fixtures/transport-golden.json` 전체 13개다.

```bash
python3 -B verify_transport.py --actual ./be-roundtrip.json
```

키 순서 외 전체 의미값을 비교한다. 누락·추가·null 변경·세트/그룹/단위 변환·중복 fixture ID는 실패다. 원본 파일을 그냥 복사해 통과시킨 것은 BE 증거가 아니다. 실행한 Java test, 저장 스키마, transaction/조회 로그, develop SHA를 함께 제출한다. 생성/편집/교체/재실행 경로마다 확인한다. UI는 같은 fixture에서 별도 캡처 QA를 수행한다.

역사적 `tests/test_numeric_id_boundary_v1.py`의 F02~F05는 고정 BE 커밋의 필드 부재를 검사하므로 계속 실패한다. 이 파일을 수정/xfail하지 않는다. 새 Java 통합 fixture 4개가 각각 통과하면 현재 소비자 문제를 해소한 것으로 기록하고 역사 증거와 구분한다.

## 3. 코어 oracle 실행

소스 `e0d04e7` 및 같은 DB/의존성 환경을 준비한 뒤 **코어 루트**에서 실행한다. 출력 경로는 매번 새 경로로 지정한다.

```bash
python -B tests/run_stage2_portable_oracle_v1.py --out ./evidence/handoff-oracle-new
```

이 러너는 필요한 코어/DB 복사본/선택 테스트를 새 임시 폴더에서 실행한다. 기존 Python 의존성을 쓰며 빈 OS 설치나 새 배포 이미지 검증은 아니다. private 스크립트·협업 BE repo·원격 서비스 없이 네트워크 차단 상태에서 수행된다. 테스트 선택 정본은 `fixtures/bf01-20-oracle.json`이며 코어의 `tests/fixtures/stage2_fallback_oracle_v1.json`과 동일 해시다.

2026-10-16 재실행: **412 passed, 실패/skip 0, BF20군 모두 실행, 원본/임시 패키지 해시 불변**. 그룹 간 중복 테스트가 있으므로 그룹 건수 합산 금지. 원본 증거는 `docs/evidence/core-completion-v1-2026-10-16/portable-oracle/`에 보존한다. 신규 실모델 호출은 0회다.

현재 모델로 생성한 schema/golden도 재검증할 수 있다(코어의 기존 의존성이 설치된 Python 필요).

```bash
python -B verify_core_assets.py --core ../fit-core-ai
```

소스 네 파일의 해시·모델 스키마·13응답 왕복·10입력 수용/거절·정상 변환을 대조한다. 고급 입력은 아직 public 요청이 아님을 음성 fixture로 고정했다.

## 4. BE fallback BF01~20

각 행은 **필수**다. 우선 코어 oracle 결과를 기준으로 삼고 BE 구현의 같은 조건/장애 주입 결과를 대조한다. 단순 고정 추천 목록은 통과 근거가 아니다.

| ID | BE가 확인할 조건 | 완료 기준 |
|---|---|---|
| BF01 | network/timeout + 숫자 ID | 제약 검사 후 fallback, 원 ID 유지 |
| BF02 | 중복 이름·미승인 연결 | 다른 catalog 메타데이터 차용 안 됨 |
| BF03 | 승인 후 source/target 변경 | stale 연결 거절 |
| BF04 | MACHINE 가능·SMITH 불가 | 스미스 추천 안 됨 |
| BF05 | 부상 직접 부하 | core hard filter 동일, 경고만 붙여 복원 금지 |
| BF06 | 관리 중 선택 동작 제한 | 해당 제한 정확히 적용, 무조건 전면 허용/차단 아님 |
| BF07 | DOMS 없음/1/2/3 | 용량·RIR·제외 동등. '없음'은 기록 없음이며 0 입력은 거절 |
| BF08 | 가슴 DOMS3, push 대 어깨/삼두 custom | 가슴 주운동 필요 여부와 별도 타깃 의도 구분 |
| BF09 | 가동성/체형 없음·있음 | 요청/프로필 값 유실 없음, 진단으로 오인 금지 |
| BF10 | low/normal/high | 용량·강도·설명·상충 일치 |
| BF11 | 초/중/상급 | admission·방법·세트 정책, 메인 휴식 보존 |
| BF12 | 두 머신 동시 점유 그룹 | 불가능한 그룹 거절/정책상 축소, 그룹 원자성 |
| BF13 | 컨디셔닝 대 회복/쿨다운 | 고강도를 회복으로 표시하지 않음 |
| BF14 | 시간/거리/인터벌 저장 | 단위·rounds 왕복 동일, 횟수 변환 금지 |
| BF15 | auto/manual/off와 제한 | 수동 방법이 부상/DOMS/회복 우회 안 됨 |
| BF16 | 0후보·빈 루틴·시간 부족 | 빈 실패0분, 짧은 성공과 구별 |
| BF17 | catalog N:1 | 숫자 이력 ID 별도 보존 |
| BF18 | 측면 어깨·상세·그룹 | v4 무손실, legacy 유실 감지 |
| BF19 | 정책/자료 미준비·조회 실패 | 검증 불가를 성공으로 처리하지 않음 |
| BF20 | 모델 오류 뒤 fallback 검증 실패 | 조건 완화 재시도 없이 실패 유지 |

BE 완료 evidence: ruleId·fixture ID·정책/DB/bridge 버전·위반 근거·최종 상태·실행 SHA. 조회 실패/버전 정책 D1~D4는 승인된 최종안으로 테스트한다. 지금 코어 oracle이 HTTP envelope를 검증한 것으로 주장하지 않는다.

## 5. 최종 통합 인수

- 필수 BE/FE 요청 ID별 테스트와 화면 증거 누락 0.
- 미지원 기능·ID·계약 오류를 조용히 변환하는 경로 0.
- v4 생성→저장→조회→실행 및 편집/교체, 정상·fallback·failed 상태 모두 확인.
- 요청/프로필/이력 공급자의 신뢰 경계 및 개인정보 로그 검토.
- 승인 한계와 미활성 기능을 사용자에게 과장 없이 표시.
- 확정 D1~D7의 구현·음성 검증 증거 및 D6 제공 범위 manifest를 확인하고 현재 develop과 차이를 재대조.

이 패키지는 검증 도구/기준이다. 실제 BE/FE 적용·배포·웹 QA는 아직 완료되지 않았다.

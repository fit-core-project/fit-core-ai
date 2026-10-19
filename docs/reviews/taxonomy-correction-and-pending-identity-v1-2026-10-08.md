# 정확 ID 분류 보정 및 보류 연결 재검수 v1

2026-10-08. 코어 스냅샷 `core-snapshot`, 시작 커밋 `b56a171`. 사용자 승인 범위의 로컬 변경이며 협업 repo, 원천 CSV, SQLite 원본, active 계약, 연결표 상태는 수정하지 않는다. 개발자 연결 전달 보류는 유지한다.

## 1. 적용 범위

`engines/muscle_taxonomy.py`에 버전 `routine-taxonomy-exact-id-v1-2026-10-08`의 정확 ID 보정 13개를 추가했다. `engines/db_queries.py`의 기존 카탈로그 enrichment 단계에서 적용한다. 이름만 같은 미승인 숫자 운동에는 적용하지 않는다.

| 정확 ID (V24ROW-) | 런타임 보정 | 적용 조건 |
|---|---|---|
| 004721, 004722, 004723, 006126, 006135, 006146, 006155, 006164, 006176, 006187 | upper-back → forearm | ID, 영문 변형명, 검증된 원천과 카탈로그 한글명, 원천/기존 canonical 주동근, sourceRetained 상태 일치 |
| 004709, 004710, 004711 | adductor → abductors | 위와 동일 |
| 004724 | 기존 forearm 보정 유지 | 기존 provenance 검수와 forearmEmphasis 전용 admission 경로 유지. 새 보정표에 중복 등록하지 않음 |

새 보정은 `primary_muscle`, `canonical_primary_muscle`와 코어 보정 출처 2개 필드만 바꾼다. 원천 분류와 DB의 기존 taxonomy metadata는 보존하고 별도 `core_taxonomy_review_version`으로 적용 이력을 구분한다. 통증 트리거, 보조근, 장비, 동작 유형, 시간/횟수 처방, 검수·난이도·자동 추천 허용은 변경하지 않는다. 주동근 수정만으로 부상 회피 정책이나 장비 정확성이 해결된 것은 아니다.

이는 루틴 코어 enrichment의 보정이다. SQLite를 직접 읽고 이 보정층을 사용하지 않는 백과/관리자 화면까지 고쳤다고 주장하지 않는다. 그런 소비자는 원본 분류를 계속 볼 수 있으며 실제 협업 웹 연결은 이번 범위 밖이다.

외전근 조회 시 원천 adductor 버킷도 읽도록 내부 조회 범위를 보완하되, enrichment 후 canonical 타깃 필터를 유지한다. 정상 내전근 및 이미 보정된 004706~004708, 004712~004714는 보존한다. 모든 기존 테스트와 golden은 변경하지 않는다.

## 2. 분류 영향 측정

원천 DB 오류 목록 14개를 그대로 고정한 뒤, 실제 SQLite를 read-only로 조회했다. 목표 하나만 선택한 custom/헬스장/근비대/60분/보통 컨디션/부상·DOMS·장비 제한 없음 조건에서 초급·중급·상급을 각각 검사했다.

| 타깃 | 보정 후 전체 query 후보 | 문제 목록 ID의 query 포함 | 문제 목록 ID의 top 20 포함 |
|---|---:|---:|---:|
| upper-back | 7,206 | 0 (직전 10) | 0 |
| adductor | 1,313 | 0 (직전 3) | 0 |
| forearm | 61 | 11 | 0 |
| abductors | 41 | 3 | 0 |

위 수치는 각 경력에서 동일하다. 조회 후보 수는 사용자에게 추천되는 종목 수가 아니다. 새 보정 13개는 모두 `main_routine_default_allowed=0`, 초·중·상급 모두 `automatic_catalog_denied` 유지다. 기존 004724만 종전대로 초급 제외, 중·상급 admission 가능이며 forearmEmphasis 범위 등 추가 정책의 통제를 받는다. 이번 검사에서 해당 운동도 top 20에는 포함되지 않았다.

근거: 코어 `docs/evidence/taxonomy-correction-v1-2026-10-08/taxonomy-effective.json`, `taxonomy-probes.json`.

## 3. 보류 연결 재검수

BE seed는 고정 커밋 `1490c91a08b674291754ced722af8dd979cf116c`의 `V6__seed_exercise_catalog.sql`을 `git show`로 읽었다. SQLite 원값, 현재 enrichment 값, 원천 CSV의 정확 변형명·장비·주동근·패턴·release/quality gate를 대조했다. 승인을 메모리에서만 가정한 시험으로 숫자 ID 보존과 적용 결과를 확인했고 연결표 파일에는 쓰지 않았다.

`approve-recommended`는 사용자 최종 승인 전의 동일성 권고다. 추천 허용, 임상 안전성, 라이선스 또는 운영 적용 승인이 아니다.

| 숫자 ID → V24ROW- | 권고 | 근거 및 남은 문제 |
|---|---|---|
| 6 → 004721 | needs-user | Wrist Roller / 전완 굴근·신근 및 악력 / accessory 패턴. forearm 보정 완료, ISOLATION 일치. 그러나 seed MACHINE, 원천 전용 리스트 롤러, 유효 장비 BODYWEIGHT. 전용 도구 요구를 잃으므로 아직 승인 권고하지 않음. reps 외 회전·왕복·시간의 처방 단위도 별도 검수 대상. |
| 9 → 004723 | needs-user | Hand Gripper / 악력·전완 굴근 / accessory. forearm 및 ISOLATION 일치. seed MACHINE, 원천 그리퍼, 유효 BODYWEIGHT 충돌은 남음. 반복 쥐기와 정적 유지 구분을 주동근 수정으로 해결하지 않음. |
| 140 → 004722 | needs-user | Plate Pinch Hold / PLATE / 악력·엄지 집기 힘. forearm과 장비는 일치하나 seed STATIC, 유효 ISOLATION/reps여서 유지 시간을 횟수로 취급하는 문제가 남음. F02 시간 계약도 미해결. |
| 151 → 004710 | **approve-recommended** | 일반 Clamshell, 원천 밴드/맨몸·둔근중부/외회전근·고관절 외전/내전. 유효 abductors/ISOLATION/reps로 seed와 정합. 유효 BODYWEIGHT는 seed BAND/BODYWEIGHT와 공통이며 원천도 맨몸을 허용. 이 연결은 맨몸 기본형에 대한 권고이며 밴드 변형 지원을 보장하지 않음. main-default=0 유지. |
| 171 → 004703 | needs-user | Spanish Squat / 대퇴사두 / 무릎·고관절 신전. 원천 밴드/스트랩, seed BAND지만 유효 장비 BODYWEIGHT. COMPOUND↔ISOLATION 충돌도 남음. 장비 요구 누락을 먼저 보정·검증해야 함. |
| 135 → 004692 | needs-user | Bird Dog / 맨몸 / 척추 안정근·둔근·복횡근 / 안티익스텐션·회전·굴곡. seed lower-back/STATIC, 유효 gluteal/ISOLATION/reps. 정적 유지와 교대 반복의 정확 변형·주동근·단위 결정 전 연결 보류. |
| 125 → 006742 | **approve-recommended** | 정확 Barbell Hip Thrust / BARBELL / 대둔근·중둔근 / 고관절 신전·외전. 원 DB abductors/ISOLATION과 달리 기존 canonical/처방 보정으로 현재는 gluteal/COMPOUND/reps가 되어 seed와 정합. 이번에 동작 유형을 새로 바꾸지 않았으며 기존 검토의 충돌이 현재 유효값에서는 해소됐음을 확인. 원천 hold, main-default=0은 그대로 유지. |
| 98 → 003251 | ambiguous | BARBELL/quadriceps/COMPOUND와 복합 무릎·고관절 신전은 부합. 하지만 seed 일반 Back Squat에는 원천의 **어깨너비·풀 ROM** 지정이 없음. 일반 운동을 특정 자세/깊이 대표로 연결할 정책 판단이 필요하며 현재 자동 대표 선정하지 않음. 유효 main-default=1은 경력·슬롯·다른 안전 제한 전체 통과를 의미하지 않음. |

151의 나머지 후보 007275(필라테스 기본), 007277(미니밴드)는 **ambiguous 유지**, 007276(매직서클), 007278(짐볼)는 **reject 권고 유지**다. 원천 추가 장비와 자세 차이가 있으므로 기본 클램셸과 영문 이름만 같다고 연결하지 않는다. 문서의 reject도 실제 연결표 상태 변경은 아니다. 총 8개 숫자 ID에 대해 12개 candidate 쌍을 대조했다.

연결표 최종 승인 대상은 **151→004710, 125→006742의 2쌍 권고**다. 나머지는 위 사유로 보류하며, 기존 approved 9 / candidate 36 / rejected 1은 유지한다. 98→006191의 기존 거절도 그대로다.

근거: 코어 `docs/evidence/taxonomy-correction-v1-2026-10-08/pending-identity-evidence.json`. 시뮬레이션 승인은 실행 중에만 대체한 연결표이며 실제 BE/FE 저장 통합 시험은 아니다.

## 4. 검증 결과

요청한 검증을 완료했다. 숫자 ID의 기존 BE 계약 실패는 별도 유지하며 이를 전체 회귀 성공 수에 섞지 않는다.

- 새 경계 검사 및 기존 분류 검사: **57 passed, 0 failed**, 58.58초.
- 전체 회귀: **3,617 passed / 0 failed / 0 skipped**, 572.20초. 직전 3,560개 전부 + 새 경계 45개 + 기존 분류 파일의 12개를 포함한다. 기존 테스트 수정 없음.
- 재생 81건: 직전 기준과 응답 차이 0건. `routineDraftId`만 제외한 공개 응답 전체 비교. success 79 / failed 2, 하드 위반 0, 입력 불변. 실제 모델 호출이 아닌 고정 선택 재생이다.
- 강제 장애 81건: 직전 기준과 응답 차이 0건. fallback 79 / failed 2, 하드 위반 0, 입력 불변. 양 모드의 022·024는 종전과 같은 `failed + emptyCandidate`이며 성공으로 집계하지 않는다.
- 차이 전수 목록: `all-response-changes.json`의 replay/fallback `changes={}`. 요약뿐 아니라 저장한 공개 응답 162개를 직접 대조했다. 비교 기준은 `b56a171`의 최종 검증 증거이며 P1 이전의 더 오래된 출력을 기준으로 하지 않는다.
- 숫자 ID 별도 fixture: **17 passed / 기존 F02~F05 4 failed**. 시간·거리·인터벌·그룹/방법을 BE DTO가 보존하지 않는 기존 실패만 남음. 수정/skip/xfail 없음.
- 보호 해시 및 협업 저장소 상태: SQLite, 원천 CSV, active 계약/golden, 연결표, 기존 테스트 및 협업 3개 repo의 상태·diff·HEAD가 불변이다. 보호 대상 중 변경은 엔진 2개와 새 검증 파일 2개뿐이다.

### 실제 Gemma

등록된 A100의 기존 `gemma4:latest`, digest `c6eb396dbd5992bbe3f5cdb947e8bbc0ee413d7c17e2beaae69f5d569cf982eb`를 확인했다. 기존 로컬 터널은 connection refused여서, 검증 시간에만 엄격한 host-key 확인을 사용하는 임시 SSH 터널을 열었다. 원격 서비스 시작/재설정·설치·자격증명 변경은 없었다. 테스트 종료 시 이 실행이 사용한 모델을 unload하고 터널을 닫았다.

| 사례 | 입력 | 실제 결과 | 경과 시간 |
|---|---|---|---:|
| expert-current-v3-001 | 초급 / 전신 / 근비대 / 50분 / 보통 / 부상·DOMS 없음 / 기본 헬스장 장비 | success, 6종목, 예상 49분 | 28.25초 |
| expert-current-v3-002 | 중급 / 나머지 조건 동일 | success, 6종목, 예상 49분 | 19.61초 |

모델 호출 정확히 2회, 재시도 0회, 별도 warmup 생성 0회. JSON schema 유효 2건, fallback 0, 하드 위반 0, 출력·언어 품질 플래그와 보조 목적 계약 위반 모두 0이다. 기본 두 사례의 성공은 모든 입력 조합이나 임상 안전성을 입증하지 않는다. 모델 원시 추론 문장은 저장하지 않고 선택 ID/슬롯과 최종 응답을 관측했다.

최종 정합성 검사 `closeout.py`는 3,617개 통과, 기존 숫자 실패 목록 동일성, 응답의 필드 존재 여부까지 포함한 엄격한 전체 비교, 원천 보존, 실제 모델 2건 성공을 확인했다. 재생/장애의 최종 routineBlocks에서 이번 분류 목록 14개는 모두 0회 등장한다. 원래 제외된 운동을 분류 보정만으로 새로 추천하지 않았다.

재현 및 원문 증거는 코어 `docs/evidence/taxonomy-correction-v1-2026-10-08/`에 보존했다.

- `regression/pytest.xml`, `numeric/pytest.xml`, `focused/pytest.xml`: 실제 테스트 수와 실패 원문.
- `runtime-controls/`: 재생/강제 장애 162개 전체 응답, 감사 JSON, 호출 0회 명시.
- `live/`: 입력 2개, 실제 선택/최종 응답/품질 평가, 모델 식별·정리 감사.
- `final-validation.json`: 단계별 결과와 전수 차이 목록, 보호 대상 대조.
- `tests/taxonomy_correction_evidence_v1.py`: before/inspect/focused/regression/numeric/controls/after 실행기. 새 버전 출력 경로를 지정하지 않고 재실행해 기존 증거를 덮어쓰지 않는다.
- `live_runner.py`, `closeout.py`: 실제 모델 제한 실행과 최종 대조 실행기.

보류 연결의 메모리 시험 승인은 evidence 생성에만 사용했다. 실제 연결표는 **approved 9 / candidate 36 / rejected 1**, 신규 승인 0개다.

## 5. 범위 밖과 후속 판단

전용 롤러/그리퍼/핀치의 장비 및 유지 시간 처방, 버드독의 정적/동적 구분, 스패니시 스쿼트의 밴드·동작 유형, 백스쿼트의 대표 변형 선택은 **이번에 수정하지 않았다**. 새 동일성 승인 2쌍도 사용자 결정 전까지 candidate다.

원천이나 DB의 오분류를 삭제하지 않아 동일 소스로 재빌드해도 코어 보정 이력을 확인할 수 있다. 반대로 원천명/기존 분류가 달라지면 정확 ID만으로 무조건 적용하지 않고 보정을 중단한다. 이는 해당 미래 카탈로그 변경이 자동 검수된다는 뜻이 아니다.

이 작업은 분류 및 연결 검수 범위의 개선이며 AI 코어 전체 완성, 의료 안전성, 협업 앱 연결 완료를 주장하지 않는다. Claude의 후속 요약 검토 대상이며 운영 적용이나 개발자 연결 전달은 하지 않는다.

## 6. 로컬 커밋 검수

코드·테스트·보고서의 staged `diff --check`는 통과했다. 전체 staged 검사에는 의도된 숫자 ID 실패 원문 XML/log의 traceback 공백 경고 6개가 있어 exit 2이며, 실패 증거를 가공하지 않고 보존했다.

기존 전체 스캔 v7의 비밀정보 패턴으로 변경 후보를 검사했다. 첫 53개 staged 파일에서 잠재 비밀정보 일치 0건이며, 최종 보고서와 검사 기록을 포함한 재검사 결과는 `precommit-staged-secret-scan-v2.json`에 보존한다. 변경하지 않은 원천/DB는 이전 전체 검사와 이번 보호 해시를 함께 사용한다. 패턴 검사로 모든 비밀정보 부재를 절대 보장하지 않는다.

로컬 `core-snapshot`에만 커밋하며 remote 추가/push/협업 repo 반영은 하지 않는다. 사용자 결정대로 이 세션에서 구현·검증하고 Claude의 후속 요약 검토를 받는 단계다.

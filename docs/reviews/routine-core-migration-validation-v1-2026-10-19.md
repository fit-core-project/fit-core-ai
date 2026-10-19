# 루틴 코어 이식 검증 v1

## 판정

**협업 준비 브랜치로 공유 가능. 전체 테스트 통과·develop 병합·운영 배포 완료는 아님.**

- 협업 기준: `80e4cbebc6d6ae9eb79b2d58263f2e1adacb5699`.
- 이식 출처: 별도 core-snapshot `c488834c17ba4e1601175292ca8a5b31aceca8a2`.
- 대상 브랜치: `codex/routine-core-v3-final-2026-10-19`.
- 코드/카탈로그 범위와 유지한 한계: [이식 안내](../routine-core-integration-2026-10-19.md).
- 파일 근거: [자산 manifest](../evidence/core-migration-v1-2026-10-19/migration-manifest.json), [결과 집계](../evidence/core-migration-v1-2026-10-19/validation-summary.json).

## 이식 트리에서 실행한 결과

| 검사 | 결과 | 해석 |
|---|---|---|
| 기존 선택 코어 회귀 + 새 LLM 격리 테스트 | 4,128 통과, 실패/생략 0 | 기존 4,119건 + routine 전용 옵션 9건 |
| 최신 develop의 나머지 테스트 + 이식된 버전 fixture | 834 통과, 19 생략, 1 실패 | 아래 골든 불일치 포함. 전체 green 아님 |
| 고정 선택 재생 | 81건, 이전 응답과 차이 0 | success79/의도된 failed2, hard 위반0 |
| 강제 모델 장애 | 81건, 이전 응답과 차이 0 | fallback79/의도된 failed2, hard 위반0 |
| 숫자 ID 기존 BE DTO 경계 | 17 통과, 4 실패 | F02~F05를 숨기지 않고 유지 |
| v4 fallback 경계 추가 검사 | 4 통과 | timeout/network 오류, 잘못된 fallback 차단 |
| 골든 + 부위 enum 단독 재현 | 원본 12통과/1실패, 이식본 12통과/1실패 | 같은 골든 실패 재현 |

19개 생략은 식품 검색 외부 환경 opt-in 18개와 실제 영양제 HTTP smoke 1개다. 실제 Gemma 호출, 새로운 의존성 설치, BE/FE 종단 통합, 배포 검증은 이번 실행에 포함하지 않았다. 합성 재생을 실모델 성능으로 합산하지 않는다.

검사는 기존 Python 3.10.12/pytest 9.0.3 환경에서 실행했다. 테스트 플러그인으로 TCP/UDP 접속을 차단하고 카탈로그 연결은 읽기 전용으로 구성했다. 일부 초기 제한 환경 실행은 스레드/ASGI 대기 또는 구 fixture 입력 때문에 중단·실패했으며 성공 집계에 포함하지 않았다. 표는 마지막 완주 실행 결과다.

## 알려진 실패

### 1. 보존된 골든과 현재 코어의 DOMS 세트 수 불일치

`tests/test_golden_baseline.py::test_main_path_with_deterministic_matches_pre_refactor_baseline`

- 보존된 v3 골든: 가슴 DOMS1, 복합운동 각각 2세트, 예상 20분.
- 원본 c488834와 이식본: 같은 입력에서 복합운동 각각 3세트, 예상 26분.
- 이 테스트는 기존 4,119건 선택 목록 밖에 있었으며 이번 협업 전체 검사에서 드러났다. 이식으로 새로 발생한 회귀가 아니다.
- golden v1/v2/v3는 모두 보존했다. 새 출력을 기대값으로 덮어쓰기, assertion 약화, xfail 처리는 하지 않았다.
- **사용자 결정 필요:** 현재 DOMS 정책에 맞춰 새 golden 버전을 승인할지, 처방 정책을 바꿀지 별도로 정합화한다. 이번 이식 범위에서 결정하지 않는다.

### 2. 기존 BE DTO가 확장 필드를 보존하지 못함

`tests/test_numeric_id_boundary_v1.py`의 실패 4건:

- `test_extended_prescription_survives_declared_be_dto[timed]`
- `test_extended_prescription_survives_declared_be_dto[distance]`
- `test_extended_prescription_survives_declared_be_dto[interval]`
- `test_execution_group_and_method_survive_declared_be_dto`

이는 BE `1490c91a08b674291754ced722af8dd979cf116c`의 DTO 소스를 고정 fixture로 검사한다. 시간/거리/인터벌·그룹/운동방법 보존은 BE 통합 과제이며 코어 테스트 통과로 대체하지 않는다. 최신 실행 서버 BE를 호출한 결과가 아니다.

## 충돌 및 보존

- 최신 develop의 `main.py`, 식단/영양제/quicklog 구현과 관련 데이터, 배포·requirements 등 37개 보호 파일은 byte 단위 동일함을 확인했다.
- 공용 `llm_router.py`의 reasoning 기본값은 **routine에만 False**로 적용했다. 다른 엔진의 develop 기본값과 명시적 환경변수 동작을 9건으로 검증했다.
- 기존 prompt/평가/golden 파일은 보존하고 승인된 코어 입력/결과를 새 이름으로 추가했다. 부상 강도 입력이 남아 있던 telemetry fixture는 승인된 이진 입력으로 이식했으며 telemetry assertion은 유지했다.
- 측면 어깨 내부 분류와 공개18부위의 차이는 원본 코어의 enum 테스트를 함께 가져와 검사한다. 공개 부위를 임의 확장하지 않았다.
- 숫자 ID 검사는 BE 소스 5파일을 커밋·해시와 함께 고정하여 인접 저장소 의존을 제거했다.
- 개인 경로 기본값은 없애고 repository-relative/명시적 환경변수로 변경했다. 스냅샷 이력 전체, 비공개 문서, 자격증명, DB 백업은 포함하지 않았다.
- 원본 core-snapshot, 협업 BE/FE 로컬 수정 및 develop 브랜치는 변경하지 않았다.

## 재현

프로젝트 루트에서 기존 의존성을 갖춘 Python으로 실행한다. 각 출력 디렉터리는 새 경로여야 한다.

```bash
python -B scripts/verify_core_migration.py core --out .artifacts/migration-core
python -B scripts/verify_core_migration.py collaboration --out .artifacts/migration-collaboration
python -B scripts/verify_core_migration.py numeric --out .artifacts/migration-numeric
python -B scripts/verify_core_migration.py oracle --out .artifacts/migration-oracle
```

collaboration/numeric은 위 알려진 실패 때문에 종료 코드 1이 정상 관찰값이다. 이를 성공 코드로 바꾸지 않는다. 선택 목록은 `tests/fixtures/core_migration_checks_v1.json`, BF01~20 세부 계약과 portable 검증은 `tests/fixtures/stage2_fallback_oracle_v1.json` 및 전달 패키지를 참조한다.

재생81/장애81의 원본 대비 응답 비교는 이식 담당 환경에서 수행했다. 공개 집계에는 입력·응답 파일 SHA-256과 UUID 제외 비교 결과를 기록한다. 해당 전체 과거 실행 이력은 이 브랜치의 별도 자동 실행 세트로 주장하지 않는다.

## 병합 전 필요한 작업

1. 기존 골든 불일치의 정책 정합화 결정.
2. D1 legacy fail-closed와 v4 HTTP adapter 구현. 현재 legacy 경로는 조회 실패/행 없음을 구분하지 못할 수 있다.
3. BE/FE v4 통합과 F02~F05 해소, 실제 API·저장·화면 검증.
4. 약 61.4MiB SQLite의 향후 갱신 방식/LFS 검토. 합성 QA 프로필 3행 포함 및 원천 라이선스 검수 상태를 유지한다.

독립 검토는 위 한계를 명시하고 최종 집계를 공개하는 **Draft PR**에 한해 진행 가능으로 판정했다. 병합은 사용자 결정이다.

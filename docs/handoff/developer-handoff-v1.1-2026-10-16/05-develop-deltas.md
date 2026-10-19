# 협업 develop 차이 D01~D10 해소표

현재 협업 소스를 이번에 수정하거나 fetch하지 않았다. 비교 기준은 **2026-10-06 조사**다.
AI `80e4cbebc6d6ae9eb79b2d58263f2e1adacb5699`, BE `1490c91a08b674291754ced722af8dd979cf116c`, FE `763f611858b99bcaf4105973590b612a277f0ef7`.
코어 기준은 `e0d04e7`. 개발자는 착수 시 현재 develop SHA와 로컬 변경을 보존해 재대조하며 이미 해결된 항목은 증거로 닫는다.

| 차이 / 우선순위 | 남은 연결 작업 | 완료 확인 / 요청 ID |
|---|---|---|
| D01 확장 응답 유실 / 필수 | DTO·저장·조회·플레이어에 단위·그룹·방법·세부값 보존 | F02~F05 신규 실제 왕복 통과 / BE01·06, FE01~03 |
| D02 숫자 ID·DB·이력 / 필수 | 승인 연결·admission·역변환 검증된 제공 manifest 작성, 원 숫자 ID 보존, 미연결 저장 거절(D6 확정) | N:1 비병합·stale거절·생성/대체/이력 저장 / BE02 |
| D03 장비 의미 / 필수 | available/unavailable 전달·우선순위·SMITH 분리 | MACHINE-only 등 정상·장애 대조 / BE09, FE06 |
| D04 BE fallback 우회 / 필수 | BF01~20 동등 제약과 검증불가 실패 | 실제 BE 장애 주입 20군 / BE05·12 |
| D05 신규 입력 누락 / 필수(범위별) | 현재 public 입력 연결, 고급 입력은 확정 D5 정책의 별도 버전화 adapter 구현 후 | 공급자·기본값·합집합·상충 캡처 / BE03·11, FE10 |
| D06 body-map·엄격 입력 / 필수 | target18/DOMS20/pain22 분리 | soleus DOMS 보존, head/knees target·DOMS 거절 / BE03·10, FE08 |
| D07 세부부위 노출 / 필수 | 공개18+details 독립 저장·표시 | 측면 어깨를 전면으로 치환하지 않는 왕복 / BE10, FE07 |
| D08 계약·golden 차이 / 필수 | 본 연결본과 실제 wire 버전 정합, 승인 후 활성 문서 새 버전 발행 | DOMS20·reason 계층·버전 협상 테스트 / D2~D4, BE07 |
| D09 pain-trigger·확장정보 / 필수 | 숫자 seed의 검수 메타데이터 적용·조회 오류 구별 | source/target 변경·옛 slug·조회 실패 음성 대조 / BE02·04·12 |
| D10 URL·편집·프로그램 연계 / 필수 | 명시 버전 라우팅 후 생성/편집/교체/저장/재실행·장애 전환 | FE→BE→AI→DB→조회 실제 E2E / D4, BE06, FE09 |

정적 DTO fixture 통과와 배포 환경의 실제 통합은 별도다. 식단/영양제 변경은 이번 코어 이식에 포함하지 않는다. source SHA·DB schema/migration·호환 버전과 각 항목 결과를 통합 인수 기록에 남긴다. 기존 파일 덮어쓰기로 과거 계약을 없애지 않는다.

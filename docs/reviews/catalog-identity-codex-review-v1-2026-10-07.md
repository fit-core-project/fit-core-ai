# 카탈로그 동일성 Codex 1차 검수 v1

2026-10-07. **45쌍 전수 검토. 최종 승인은 사용자이며 연결표 상태는 전부 candidate, approved 0개 유지.**

권고 집계: {'needs-user': 13, 'ambiguous': 20, 'approve-recommended': 4, 'reject': 8}.

`approve-recommended`는 동일성 연결에 대한 1차 권고만 뜻한다. 운동 안전성, 추천 admission, 자료 라이선스, 운영 배포 승인이 아니다. `reject`는 현재 seed의 동일 운동 연결로 부적합한 정확 변형이고, `ambiguous`는 여러 변형/수행 방식 중 결정 불가, `needs-user`는 원천-파생 충돌/처방 의미를 먼저 정리해야 하는 경우다. needs-user는 잘못된 DB 값을 그대로 승인해 달라는 뜻이 아니다.

BE seed 커밋 `1490c91a08b674291754ced722af8dd979cf116c`, seed SHA `d54e3e68b44c72ff9478695c11bca1d867c420bc482ce12d546d849d181d02ef`. SQLite SHA `cf0d651dbb5c6e961289467a2d4e23520cafd083f24ca0a77554bdd003cebe81`. git show로 고정 seed를 읽고 실제 SQLite를 read-only 조회해 45쌍의 5개 identity 필드를 다시 대조했다. CSV 원천의 정확 이름·장비·주동근·패턴·난이도·release/quality/license gate를 함께 읽었다. 이름 일치만으로 자동 승인하지 않았다.

## 쌍별 권고

seed/DB 값은 `장비 / 주동근 / 동작유형` 순서다. 원천 패턴은 COMPOUND/ISOLATION보다 구체적인 동작 근거다. 원천의 gold도 모든 용도의 승인이나 라이선스 해제를 뜻하지 않는다.

| 숫자 ID → 카탈로그 | 정확 변형명 | seed → DB | 원천 장비·근육·패턴 | 원천 검수값 | 권고 및 이유 |
|---|---|---|---|---|---|
| 6 → V24ROW-004721 | 리스트 롤러 | MACHINE / forearm / ISOLATION → BODYWEIGHT / upper-back / ISOLATION | 리스트 롤러 / 전완 굴근/신근, 악력 / 전완/목/액세서리 | 중급 / silver / silver_release_candidate | needs-user: 원천은 리스트 롤러/전완인데 파생은 BODYWEIGHT/upper-back. seed MACHINE 분류와 실제 전용 장비 의미를 정리한 뒤 재검수. |
| 7 → V24ROW-007417 | 크로스핏 데드리프트 (RX/표준) | BARBELL, DUMBBELL / lower-back / COMPOUND → BODYWEIGHT / lower-back / COMPOUND | 크로스핏 장비/맨몸 / 햄스트링/둔근/척추기립근 / 힙힌지 | 중급 / hold / hold_manual_review | ambiguous: RX/Scaled/Beginner라는 구체 변형이 seed에 없음. 원천 장비가 크로스핏 장비/맨몸으로 포괄적이며 파생 주동근/장비/동작 충돌도 존재. 정확 세팅·스케일 기준 확인 전 연결 불가. |
| 7 → V24ROW-007418 | 크로스핏 데드리프트 (Scaled) | BARBELL, DUMBBELL / lower-back / COMPOUND → BODYWEIGHT / lower-back / COMPOUND | 크로스핏 장비/맨몸 / 햄스트링/둔근/척추기립근 / 힙힌지 | 중급 / hold / hold_manual_review | ambiguous: RX/Scaled/Beginner라는 구체 변형이 seed에 없음. 원천 장비가 크로스핏 장비/맨몸으로 포괄적이며 파생 주동근/장비/동작 충돌도 존재. 정확 세팅·스케일 기준 확인 전 연결 불가. |
| 7 → V24ROW-007419 | 크로스핏 데드리프트 (Beginner) | BARBELL, DUMBBELL / lower-back / COMPOUND → BODYWEIGHT / lower-back / COMPOUND | 크로스핏 장비/맨몸 / 햄스트링/둔근/척추기립근 / 힙힌지 | 초급 / hold / hold_manual_review | ambiguous: RX/Scaled/Beginner라는 구체 변형이 seed에 없음. 원천 장비가 크로스핏 장비/맨몸으로 포괄적이며 파생 주동근/장비/동작 충돌도 존재. 정확 세팅·스케일 기준 확인 전 연결 불가. |
| 9 → V24ROW-004723 | 그리퍼 | MACHINE / forearm / ISOLATION → BODYWEIGHT / upper-back / ISOLATION | 그리퍼 / 악력, 전완 굴근 / 전완/목/액세서리 | 초급 / silver / silver_release_candidate | needs-user: 원천은 그리퍼/전완인데 파생은 BODYWEIGHT/upper-back. seed MACHINE과 전용 악력기 분류 차이를 먼저 확정. |
| 14 → V24ROW-006036 | 케이블 베이지안 컬 | CABLE / biceps / ISOLATION → CABLE / biceps / ISOLATION | 케이블 / 상완이두근 / 팔꿈치 굴곡 | 초급~중급 / gold / gold_release_candidate | approve-recommended: 정확 종목명과 장비, 주동근, 고립 동작 유형이 일치하고 원천 패턴도 부합. 동일성 연결만 권고하며 hold/라이선스/추천 허용 해제는 아님. |
| 30 → V24ROW-000086 | 바벨 플랫 스탠다드 그립 체스트 프레스 | BARBELL / chest / COMPOUND → BARBELL / chest / COMPOUND | 바벨 / 대흉근 중부/흉골부 / 수평/대각선 밀기 | 중급 / hold / hold_manual_review | ambiguous: 기본 바벨 운동과 패턴/장비는 부합하나 복수 변형 후보가 존재. 그립·자세·ROM·대표 레코드 선택을 seed만으로 확정할 수 없음. 두 후보 동시 승인은 런타임 모호성으로 거절됨. |
| 30 → V24ROW-005067 | 바벨 플랫 일반그립 기본 체스트 프레스 | BARBELL / chest / COMPOUND → BARBELL / chest / COMPOUND | 바벨 / 대흉근 / 수평/사선 밀기 | 중급 / gold / gold_release_candidate | ambiguous: 기본 바벨 운동과 패턴/장비는 부합하나 복수 변형 후보가 존재. 그립·자세·ROM·대표 레코드 선택을 seed만으로 확정할 수 없음. 두 후보 동시 승인은 런타임 모호성으로 거절됨. |
| 47 → V24ROW-007267 | 필라테스 필라테스 푸쉬업 (기본) | BODYWEIGHT / chest / COMPOUND → BODYWEIGHT / abs / COMPOUND | 맨몸 / 대흉근/삼두/코어 / 수평 밀기 | 중급 / hold / hold_manual_review | ambiguous: 기본 필라테스 푸시업이지만 seed 기본 푸시업과 실시 범위가 같은지 불명. 원천은 가슴/삼두/코어, 파생값은 abs 단일로 충돌. 47과117 이력은 합치지 않음. |
| 47 → V24ROW-007268 | 필라테스 필라테스 푸쉬업 (매직서클) | BODYWEIGHT / chest / COMPOUND → BODYWEIGHT / abs / COMPOUND | 매직서클 / 대흉근/삼두/코어 / 수평 밀기 | 중급 / hold / hold_manual_review | reject: 원천의 매직서클/미니밴드/짐볼 변형은 seed의 기본 맨몸 푸시업과 동일 운동으로 볼 근거가 없음. BODYWEIGHT 파생값이 원천 추가 장비를 지운 상태. 대체 관계 가능성과 동일성은 별개. |
| 47 → V24ROW-007269 | 필라테스 필라테스 푸쉬업 (미니밴드) | BODYWEIGHT / chest / COMPOUND → BODYWEIGHT / abs / COMPOUND | 밴드 / 대흉근/삼두/코어 / 수평 밀기 | 중급 / hold / hold_manual_review | reject: 원천의 매직서클/미니밴드/짐볼 변형은 seed의 기본 맨몸 푸시업과 동일 운동으로 볼 근거가 없음. BODYWEIGHT 파생값이 원천 추가 장비를 지운 상태. 대체 관계 가능성과 동일성은 별개. |
| 47 → V24ROW-007270 | 필라테스 필라테스 푸쉬업 (짐볼) | BODYWEIGHT / chest / COMPOUND → BODYWEIGHT / abs / COMPOUND | 짐볼 / 대흉근/삼두/코어 / 수평 밀기 | 중급 / hold / hold_manual_review | reject: 원천의 매직서클/미니밴드/짐볼 변형은 seed의 기본 맨몸 푸시업과 동일 운동으로 볼 근거가 없음. BODYWEIGHT 파생값이 원천 추가 장비를 지운 상태. 대체 관계 가능성과 동일성은 별개. |
| 50 → V24ROW-007474 | 크로스핏 풀업 (RX/표준) | BODYWEIGHT / upper-back / COMPOUND → BODYWEIGHT / biceps / COMPOUND | 크로스핏 장비/맨몸 / 광배근/이두 / 수직 당기기 | 중급 / hold / hold_manual_review | ambiguous: RX/Scaled/Beginner라는 구체 변형이 seed에 없음. 원천 장비가 크로스핏 장비/맨몸으로 포괄적이며 파생 주동근/장비/동작 충돌도 존재. 정확 세팅·스케일 기준 확인 전 연결 불가. |
| 50 → V24ROW-007475 | 크로스핏 풀업 (Scaled) | BODYWEIGHT / upper-back / COMPOUND → BODYWEIGHT / biceps / COMPOUND | 크로스핏 장비/맨몸 / 광배근/이두 / 수직 당기기 | 중급 / hold / hold_manual_review | ambiguous: RX/Scaled/Beginner라는 구체 변형이 seed에 없음. 원천 장비가 크로스핏 장비/맨몸으로 포괄적이며 파생 주동근/장비/동작 충돌도 존재. 정확 세팅·스케일 기준 확인 전 연결 불가. |
| 50 → V24ROW-007476 | 크로스핏 풀업 (Beginner) | BODYWEIGHT / upper-back / COMPOUND → BODYWEIGHT / biceps / COMPOUND | 크로스핏 장비/맨몸 / 광배근/이두 / 수직 당기기 | 초급 / hold / hold_manual_review | ambiguous: RX/Scaled/Beginner라는 구체 변형이 seed에 없음. 원천 장비가 크로스핏 장비/맨몸으로 포괄적이며 파생 주동근/장비/동작 충돌도 존재. 정확 세팅·스케일 기준 확인 전 연결 불가. |
| 73 → V24ROW-004796 | 덤벨 슈러그 | DUMBBELL / trapezius / ISOLATION → DUMBBELL / upper-back / ISOLATION | 덤벨 / 상부승모근, 견갑거근 / 견갑 거상 | 초급~중급 / silver / silver_release_candidate | needs-user: 덤벨/견갑 거상/상부승모 원천은 seed와 부합하나 파생 upper-back은 seed trapezius와 다름. taxonomy 보정 후 동일성 재검수. |
| 74 → V24ROW-004791 | 바벨 슈러그 | BARBELL / trapezius / ISOLATION → BARBELL / upper-back / ISOLATION | 바벨 / 상부승모근, 견갑거근 / 견갑 거상 | 중급 / silver / silver_release_candidate | needs-user: 바벨/견갑 거상/상부승모 원천은 부합하나 파생 upper-back 충돌. 124와 N:1일 수 있으나 숫자 이력 ID 유지 필요. |
| 75 → V24ROW-000376 | 바벨 스탠딩 밀리터리 스탠다드 그립 프레스 | BARBELL / front-deltoids / COMPOUND → BARBELL / front-deltoids / COMPOUND | 바벨 / 전면삼각근, 측면삼각근 / 수직 밀기 | 중급 / hold / hold_manual_review | ambiguous: 기본 바벨 운동과 패턴/장비는 부합하나 복수 변형 후보가 존재. 그립·자세·ROM·대표 레코드 선택을 seed만으로 확정할 수 없음. 두 후보 동시 승인은 런타임 모호성으로 거절됨. |
| 75 → V24ROW-000379 | 바벨 스탠딩 숄더 스탠다드 그립 프레스 | BARBELL / front-deltoids / COMPOUND → BARBELL / front-deltoids / COMPOUND | 바벨 / 전면삼각근, 측면삼각근 / 수직 밀기 | 중급 / hold / hold_manual_review | ambiguous: 기본 바벨 운동과 패턴/장비는 부합하나 복수 변형 후보가 존재. 그립·자세·ROM·대표 레코드 선택을 seed만으로 확정할 수 없음. 두 후보 동시 승인은 런타임 모호성으로 거절됨. |
| 98 → V24ROW-006191 | 바벨 어깨너비 백 스쿼트 | BARBELL / quadriceps / COMPOUND → BARBELL / quadriceps / ISOLATION | 바벨 / 대퇴사두근 / 무릎주도 스쿼트 | 중급 / hold / hold_manual_review | needs-user: 백스쿼트 원천은 무릎주도 스쿼트이나 파생 ISOLATION과 seed COMPOUND가 충돌. 003251과 다른 정확 ID이며 대신 자동 연결하지 않음. |
| 99 → V24ROW-007402 | 크로스핏 프론트 스쿼트 (RX/표준) | BARBELL / quadriceps / COMPOUND → BODYWEIGHT / abs / COMPOUND | 크로스핏 장비/맨몸 / 대퇴사두/코어 / 스쿼트 | 중급 / hold / hold_manual_review | ambiguous: RX/Scaled/Beginner라는 구체 변형이 seed에 없음. 원천 장비가 크로스핏 장비/맨몸으로 포괄적이며 파생 주동근/장비/동작 충돌도 존재. 정확 세팅·스케일 기준 확인 전 연결 불가. |
| 99 → V24ROW-007403 | 크로스핏 프론트 스쿼트 (Scaled) | BARBELL / quadriceps / COMPOUND → BODYWEIGHT / abs / COMPOUND | 크로스핏 장비/맨몸 / 대퇴사두/코어 / 스쿼트 | 중급 / hold / hold_manual_review | ambiguous: RX/Scaled/Beginner라는 구체 변형이 seed에 없음. 원천 장비가 크로스핏 장비/맨몸으로 포괄적이며 파생 주동근/장비/동작 충돌도 존재. 정확 세팅·스케일 기준 확인 전 연결 불가. |
| 99 → V24ROW-007404 | 크로스핏 프론트 스쿼트 (Beginner) | BARBELL / quadriceps / COMPOUND → BODYWEIGHT / abs / COMPOUND | 크로스핏 장비/맨몸 / 대퇴사두/코어 / 스쿼트 | 초급 / hold / hold_manual_review | ambiguous: RX/Scaled/Beginner라는 구체 변형이 seed에 없음. 원천 장비가 크로스핏 장비/맨몸으로 포괄적이며 파생 주동근/장비/동작 충돌도 존재. 정확 세팅·스케일 기준 확인 전 연결 불가. |
| 115 → V24ROW-006763 | 머신 힙 어브덕션 | MACHINE / abductors / ISOLATION → MACHINE / abductors / ISOLATION | 머신 / 대둔근, 중둔근 / 고관절 신전/외전 | 초급~중급 / hold / hold_manual_review | approve-recommended: 정확 종목명과 장비, 주동근, 고립 동작 유형이 일치하고 원천 패턴도 부합. 동일성 연결만 권고하며 hold/라이선스/추천 허용 해제는 아님. |
| 116 → V24ROW-006771 | 케이블 힙 어브덕션 | CABLE / abductors / ISOLATION → CABLE / abductors / ISOLATION | 케이블 / 대둔근, 중둔근 / 고관절 신전/외전 | 초급~중급 / hold / hold_manual_review | approve-recommended: 정확 종목명과 장비, 주동근, 고립 동작 유형이 일치하고 원천 패턴도 부합. 동일성 연결만 권고하며 hold/라이선스/추천 허용 해제는 아님. |
| 117 → V24ROW-007267 | 필라테스 필라테스 푸쉬업 (기본) | BODYWEIGHT / chest / COMPOUND → BODYWEIGHT / abs / COMPOUND | 맨몸 / 대흉근/삼두/코어 / 수평 밀기 | 중급 / hold / hold_manual_review | ambiguous: 기본 필라테스 푸시업이지만 seed 기본 푸시업과 실시 범위가 같은지 불명. 원천은 가슴/삼두/코어, 파생값은 abs 단일로 충돌. 47과117 이력은 합치지 않음. |
| 117 → V24ROW-007268 | 필라테스 필라테스 푸쉬업 (매직서클) | BODYWEIGHT / chest / COMPOUND → BODYWEIGHT / abs / COMPOUND | 매직서클 / 대흉근/삼두/코어 / 수평 밀기 | 중급 / hold / hold_manual_review | reject: 원천의 매직서클/미니밴드/짐볼 변형은 seed의 기본 맨몸 푸시업과 동일 운동으로 볼 근거가 없음. BODYWEIGHT 파생값이 원천 추가 장비를 지운 상태. 대체 관계 가능성과 동일성은 별개. |
| 117 → V24ROW-007269 | 필라테스 필라테스 푸쉬업 (미니밴드) | BODYWEIGHT / chest / COMPOUND → BODYWEIGHT / abs / COMPOUND | 밴드 / 대흉근/삼두/코어 / 수평 밀기 | 중급 / hold / hold_manual_review | reject: 원천의 매직서클/미니밴드/짐볼 변형은 seed의 기본 맨몸 푸시업과 동일 운동으로 볼 근거가 없음. BODYWEIGHT 파생값이 원천 추가 장비를 지운 상태. 대체 관계 가능성과 동일성은 별개. |
| 117 → V24ROW-007270 | 필라테스 필라테스 푸쉬업 (짐볼) | BODYWEIGHT / chest / COMPOUND → BODYWEIGHT / abs / COMPOUND | 짐볼 / 대흉근/삼두/코어 / 수평 밀기 | 중급 / hold / hold_manual_review | reject: 원천의 매직서클/미니밴드/짐볼 변형은 seed의 기본 맨몸 푸시업과 동일 운동으로 볼 근거가 없음. BODYWEIGHT 파생값이 원천 추가 장비를 지운 상태. 대체 관계 가능성과 동일성은 별개. |
| 124 → V24ROW-004791 | 바벨 슈러그 | BARBELL / trapezius / ISOLATION → BARBELL / upper-back / ISOLATION | 바벨 / 상부승모근, 견갑거근 / 견갑 거상 | 중급 / silver / silver_release_candidate | needs-user: 74와 같은 카탈로그 후보. 원천 상부승모와 파생 upper-back 충돌 정리 필요. 이력 병합은 금지. |
| 125 → V24ROW-006742 | 바벨 힙 쓰러스트 | BARBELL / gluteal / COMPOUND → BARBELL / abductors / ISOLATION | 바벨 / 대둔근, 중둔근 / 고관절 신전/외전 | 중급 / hold / hold_manual_review | needs-user: 원천 대둔근/중둔근 고관절 신전과 파생 abductors/ISOLATION이 seed gluteal/COMPOUND와 충돌. 메타데이터/처방 단위 확정 필요. |
| 130 → V24ROW-006028 | 덤벨 해머 컬 | DUMBBELL / biceps / ISOLATION → DUMBBELL / upper-back / ISOLATION | 덤벨 / 상완근, 상완요골근 / 팔꿈치 굴곡 | 초급~중급 / silver / silver_release_candidate | needs-user: 원천 상완근/상완요골근 팔꿈치 굴곡이 파생 upper-back으로 축약됨. seed biceps와의 공개/세부 분류 기준 정리 필요. |
| 135 → V24ROW-004692 | 버드독 | BODYWEIGHT / lower-back / STATIC → BODYWEIGHT / gluteal / ISOLATION | 맨몸 / 척추 안정근, 둔근, 복횡근 / 안티익스텐션/회전/굴곡 | 초급 / hold / hold_manual_review | needs-user: 원천 척추 안정근/둔근/복횡근이며 seed STATIC, 파생 ISOLATION. 정적 유지/교대 반복 버드독 구분 및 시간 처방 계약 확인 필요. |
| 140 → V24ROW-004722 | 플레이트 핀치 홀드 | PLATE / forearm / STATIC → PLATE / upper-back / ISOLATION | 플레이트 / 악력, 엄지 집기 힘 / 전완/목/액세서리 | 초급-중급 / silver / silver_release_candidate | needs-user: 장비 플레이트는 일치하나 원천 악력/엄지 집기 힘을 upper-back으로 파생. STATIC vs ISOLATION도 충돌하여 시간 처방 경계 먼저 검수. |
| 144 → V24ROW-006904 | 케이블 크런치 | CABLE / abs / ISOLATION → CABLE / abs / ISOLATION | 케이블 / 복직근, 복횡근 / 코어 안정화 | 초급~중급 / gold / gold_release_candidate | approve-recommended: 정확 종목명과 장비, 주동근, 고립 동작 유형이 일치하고 원천 패턴도 부합. 동일성 연결만 권고하며 hold/라이선스/추천 허용 해제는 아님. |
| 151 → V24ROW-004710 | 클램쉘 | BAND, BODYWEIGHT / abductors / ISOLATION → BODYWEIGHT / adductor / ISOLATION | 밴드/맨몸 / 둔근중부, 외회전근 / 고관절 외전/내전 | 초급 / gold / gold_release_candidate | needs-user: 원천 밴드/맨몸·둔근중부/외회전근은 seed와 가까우나 파생 adductor는 외전 abductors와 반대 분류. 잘못된 파생값을 승계하지 않도록 보정 검토 필요. |
| 151 → V24ROW-007275 | 필라테스 클램쉘 (기본) | BAND, BODYWEIGHT / abductors / ISOLATION → BODYWEIGHT / gluteal / ISOLATION | 맨몸 / 중둔근/외회전근 / 고관절 외회전 | 초급 / hold / hold_manual_review | ambiguous: 기본/미니밴드 필라테스 변형은 seed 장비와 겹치지만 기본 클램셸 후보와의 선택 및 수행 조건이 미확정. 주동근의 파생 taxonomy 차이도 확인 필요. |
| 151 → V24ROW-007276 | 필라테스 클램쉘 (매직서클) | BAND, BODYWEIGHT / abductors / ISOLATION → BODYWEIGHT / gluteal / ISOLATION | 매직서클 / 중둔근/외회전근 / 고관절 외회전 | 초급 / hold / hold_manual_review | reject: 매직서클 또는 짐볼의 별도 세팅은 seed BAND/BODYWEIGHT 조건에 포함되지 않음. 일반 클램셸과 무조건 같은 ID로 연결하지 않음. |
| 151 → V24ROW-007277 | 필라테스 클램쉘 (미니밴드) | BAND, BODYWEIGHT / abductors / ISOLATION → BODYWEIGHT / gluteal / ISOLATION | 밴드 / 중둔근/외회전근 / 고관절 외회전 | 초급 / hold / hold_manual_review | ambiguous: 기본/미니밴드 필라테스 변형은 seed 장비와 겹치지만 기본 클램셸 후보와의 선택 및 수행 조건이 미확정. 주동근의 파생 taxonomy 차이도 확인 필요. |
| 151 → V24ROW-007278 | 필라테스 클램쉘 (짐볼) | BAND, BODYWEIGHT / abductors / ISOLATION → BODYWEIGHT / gluteal / ISOLATION | 짐볼 / 중둔근/외회전근 / 고관절 외회전 | 초급 / hold / hold_manual_review | reject: 매직서클 또는 짐볼의 별도 세팅은 seed BAND/BODYWEIGHT 조건에 포함되지 않음. 일반 클램셸과 무조건 같은 ID로 연결하지 않음. |
| 161 → V24ROW-007456 | 크로스핏 박스 점프 (RX/표준) | BOX / quadriceps / PLYOMETRIC → BODYWEIGHT / calves / COMPOUND | 크로스핏 장비/맨몸 / 둔근/대퇴사두/종아리 / 점프 | 중급 / hold / hold_manual_review | ambiguous: RX/Scaled/Beginner라는 구체 변형이 seed에 없음. 원천 장비가 크로스핏 장비/맨몸으로 포괄적이며 파생 주동근/장비/동작 충돌도 존재. 정확 세팅·스케일 기준 확인 전 연결 불가. |
| 161 → V24ROW-007457 | 크로스핏 박스 점프 (Scaled) | BOX / quadriceps / PLYOMETRIC → BODYWEIGHT / calves / COMPOUND | 크로스핏 장비/맨몸 / 둔근/대퇴사두/종아리 / 점프 | 중급 / hold / hold_manual_review | ambiguous: RX/Scaled/Beginner라는 구체 변형이 seed에 없음. 원천 장비가 크로스핏 장비/맨몸으로 포괄적이며 파생 주동근/장비/동작 충돌도 존재. 정확 세팅·스케일 기준 확인 전 연결 불가. |
| 161 → V24ROW-007458 | 크로스핏 박스 점프 (Beginner) | BOX / quadriceps / PLYOMETRIC → BODYWEIGHT / calves / COMPOUND | 크로스핏 장비/맨몸 / 둔근/대퇴사두/종아리 / 점프 | 초급 / hold / hold_manual_review | ambiguous: RX/Scaled/Beginner라는 구체 변형이 seed에 없음. 원천 장비가 크로스핏 장비/맨몸으로 포괄적이며 파생 주동근/장비/동작 충돌도 존재. 정확 세팅·스케일 기준 확인 전 연결 불가. |
| 165 → V24ROW-004712 | 프로그 펌프 | BODYWEIGHT, BAND / gluteal / ISOLATION → DUMBBELL, BODYWEIGHT / adductor / ISOLATION | 맨몸/덤벨 / 둔근대근 / 고관절 외전/내전 | 초급 / gold / gold_release_candidate | needs-user: 원천 둔근대근인데 파생 adductor. seed BAND/BODYWEIGHT와 원천 BODYWEIGHT/DUMBBELL의 추가 장비 범위도 서로 다름. |
| 171 → V24ROW-004703 | 스페니시 스쿼트 | BAND / quadriceps / COMPOUND → BODYWEIGHT / quadriceps / ISOLATION | 밴드/스트랩 / 대퇴사두근 / 무릎/고관절 신전 | 초급-중급 / gold / gold_release_candidate | needs-user: 원천 밴드/스트랩은 seed BAND와 가깝지만 파생 BODYWEIGHT로 장비 요구 누락. COMPOUND/ISOLATION 충돌 정리 필요. |

## 승인 전에 확인할 사항

- 14/115/116/144는 동일성 권고 4쌍이며, 특히 115/116의 원천 hold는 유지한다. 이 연결로 추천 허용을 자동 승격하지 않는다.
- 30의 두 벤치프레스와 75의 두 오버헤드 프레스는 대표 변형/ROM/그립을 확인해야 한다. 흔하거나 gold인 레코드라는 이유로 대표를 자동 선택하지 않는다.
- 98→006191의 ISOLATION 충돌은 별도의 백스쿼트 admission 대상 003251과 혼동하지 않는다.
- 74/124, 47/117처럼 같은 카탈로그를 가리켜도 사용자 기록의 숫자 ID는 보존한다.
- 원천 메타데이터는 DB 파생 오류를 발견하는 근거이지 임상 안전 인증이 아니다. 원천의 license gate는 전부 링크 메타데이터 한정/추가 검토 필요 상태다.
- 전문 검수/코드 변경 없이 운동 분류를 덮어씌우지 않았다. 실제 동작 영상, 그립/ROM 상세와 라이선스 원문은 이번 동일성 검수에서 추가 확보하지 않았다.

기계 판독 근거: 같은 이름의 `.json` 파일. 소스/DB/CSV 필드와 각 권고를 모두 포함한다. 문서/JSON 생성 전후 연결표 바이트 동일성을 검사했다.

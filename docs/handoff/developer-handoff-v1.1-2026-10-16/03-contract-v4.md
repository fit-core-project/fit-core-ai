# 루틴 요청·응답 통합 계약 v4 연결본

문서 버전: handoff-v1.1 / 2026-10-16. **구현된 코어 동작 + 승인된 HTTP 연결 구현 명세**.
실행 소스 `e0d04e7`, 응답 `contractVersion="routine-core-v4-draft-v3"`.
활성 v3.1 문서를 대체 발행하거나 새 URL을 배포한 문서가 아니다.

## 1. 진입 경계

새 연결의 코어 기준 함수는 `engines.routine_connection_boundary.generate_checked_v4(request, db, profile=..., recent_sets=...)`다. 외부 JSON은 `PublicRoutineGenerateRequest`로 검증하고 `to_internal_request_payload`로 변환한다. 인증된 `userId`, 프로필·최근 운동 이력은 서버가 공급한다.

기존 `main.py` HTTP 경로는 내부 `RoutineRequest`를 받는다. 폐쇄형 public 모델이나 checked 경계가 HTTP에 이미 적용됐다고 간주하지 않는다. 기존 `generate_smart_routine_v4`는 조회 오류를 구별하지 못하는 낮은 수준 wrapper이며 checked 함수의 대체가 아니다. checked 함수는 generation-choice facade가 아닌 pipeline을 직접 호출한다.

명시적 v4 경로와 응답 버전 검사, 외부 오류 형식, 별도 고급 입력 adapter 정책은 [D1~D7 결정 기록](06-decisions.md)으로 확정됐다. 이전 초안의 `/v4/routines/generate`를 연결 경로 기준으로 사용하고 서비스 공통 prefix는 배포 문서에 기록한다. 현 AI HTTP 서버에는 아직 적용되지 않았다. 내부 스키마를 그대로 외부 입력 허용목록으로 사용하지 않는다.

## 2. 현재 폐쇄형 사용자 입력

camelCase만, 추가 키 거절. `schemas/public-request.schema.json`과 실행 validator가 기준이다. JSON Schema만으로 표현되지 않는 필드 간 상충은 oracle도 실행해야 한다.

| 필드 | 허용·기본값 | 공급/보존 |
|---|---|---|
| targetSplitLabel / targetMuscles | split 또는 부위 필요, custom은 부위 필수 | 직접 어깨·삼두 선택을 push로 강제하지 않음 |
| timeAvailableMin | 정수 10~180, 필수 | 실제 요청 시간 보존 |
| readinessLevel | low/normal/high, 기본 normal | 회복 제한 적용 |
| currentPainAreas | `[{bodyPart: slug}]`, 기본 [] | 부상22, 강도 입력 없음. 프로필 부상과 합집합 |
| currentDoms | `[{bodyPart: slug, level: 1~3}]`, 기본 [] | DOMS20. 0·미지정 부위로 부상 해제 금지 |
| availableEquipment | 아래 장비 코드 배열, 기본 [] | 비어 있지 않은 요청 > 저장 장비 > []. []는 명시적 무장비가 아님 |
| unavailableEquipment | 장비 배열, 기본 [] | 내부 `equipment`는 이 블랙리스트. resolved whitelist와 겹치면 거절 |
| goal | strength/hypertrophy/endurance/fatLoss/recomposition/generalFitness | 요청 > 저장 > hypertrophy |
| experienceLevel | beginner/intermediate/advanced | 요청 > 저장 > intermediate |
| trainingStyle | gym만, 기본 gym | 독립 요가/크로스핏 루틴 엔진 아님 |
| sessionStructure | standard/superset/circuit/density, 기본 standard | 실제 그룹이 성립하지 않으면 코어 결과·설명 따름 |
| adjunctOptions | mobility_warmup/prehab_accessory/conditioning_finisher/endurance_support/cooldown_stretch, 최대 2 | conditioning+endurance 동시 선택 금지 |
| profileSignals | 아래 6개 선택형, 기본 [] | 없음은 선택 없음이지 의학적 정상 확인 아님 |
| userNote | 선택 텍스트 | 제약 해제·의료 허가로 해석하지 않음 |

장비: `BAND, BARBELL, BODYWEIGHT, CABLE, DUMBBELL, KETTLEBELL, LANDMINE, MACHINE, PLATE, SMITH_MACHINE`. MACHINE이 SMITH_MACHINE을 포함하지 않는다. 맨몸 운동 의도는 BODYWEIGHT 명시를 사용한다. D7에 따라 []의 저장값 상속은 유지하며 별도 '장비 0개' 상태는 이번 계약에 추가하지 않는다.

split: `push, pull, legs, upper, chest_back, lower, chest, back, shoulder, arm, core, neck, full_body, custom`.

신체 선택: `heel_down_deep_squat_difficult`(뒤꿈치를 붙인 깊은 앉기 어려움), `hip_hinge_difficult`(고관절 접기 어려움), `overhead_reach_difficult`(머리 위로 팔 올리기 어려움), `loaded_wrist_extension_difficult`(손목을 젖혀 체중 싣기 어려움), `self_reported_long_arms`(긴 팔 자가 인식), `self_reported_long_thighs`(긴 허벅지 자가 인식). 진단·정량 측정값이 아니다.

low+서킷/밀도형, low+컨디셔닝/지구력 보조, strength+서킷은 거절한다. goal이 저장 프로필에서 해석될 때도 다시 검사한다. 리스트 null/숫자 문자열/boolean 단계 등을 임의 기본값으로 바꾸지 않는다. 상세 오류 조건은 public validator 및 BF oracle이 기준이다.

## 3. 부위 집합과 세부값

운동 타깃/공개 `primaryMuscles` 18개:
`abductors, abs, adductor, back-deltoids, biceps, calves, chest, forearm, front-deltoids, gluteal, hamstring, lower-back, neck, obliques, quadriceps, trapezius, triceps, upper-back`.

DOMS20 = 위18 + `left-soleus, right-soleus`. 부상22 = 위20 + `head, knees`. DOMS 없음은 [], 내부는 {}. 머리·무릎을 DOMS/운동 타깃으로 만들지 않는다. 가자미근도 타깃으로 추가하지 않는다.

내부 세부값은 `primaryMuscleDetails`로 별도 보존한다. 측면 어깨만 있으면 `primaryMuscles: []`, `primaryMuscleDetails: ["side-deltoids"]`가 유효하다. 전면 어깨로 치환하지 않는다. FE는 한글 상세명 표시, BE는 무손실 저장. 미지 세부값은 임의 판단에 쓰지 않고 호환성 문제로 기록한다.

## 4. 서버 문맥과 아직 외부 입력이 아닌 것

내부 `RoutineRequest`, `UserProfileContext`, 최근 이력의 필드는 `schemas/internal-request.schema.json` 및 코어 모델을 참조한다. 이는 FE가 raw dict로 보낼 권한이 아니다.

- `specializationFocus` 18종, `methodMode=auto/manual/off`, `requestedMethod=pausedReps/tempoEccentric`는 코어에 있으나 현재 폐쇄형 public 요청에는 없다. D5의 별도 버전화된 allowlist adapter/schema 구현·검증 전 전송하지 않는다.
- `recentTrainingMethodUses`, 선호/비선호 운동 ID, 기준 중량·운동 이력은 서버 자료다. 선호가 제외 규칙을 해제하지 않는다.
- `mobilityLimits`는 profileSignals/저장 문맥에서 도출하며 추가 심각도 slider를 만들지 않는다. 체형 dict·수술/질환 정책 dict·조회 성공 플래그를 사용자 신뢰값으로 받지 않는다.
- 관리 상태 `injuryConditions`: shoulder/knee/lumbar/wrist/ankle/hip/elbow/neck, 상태 activeUncleared/managedWithRestrictions/managedNoRestrictions/recovered. 해당 부위별 동작 제한 검증은 코어 oracle을 따른다. UI 선택 사실이 의료진 허가라는 뜻이 아니다. D5 adapter는 사용자 선택만 받고 서버 정책·검수·조회 성공 플래그는 받지 않는다.
- `trainingMethod`, `advancedChestFocus`, 과거 trainingStyle, 어깨 전용 legacy 필드는 내부 호환이다. 사용자에게 현대 입력과 중복 노출하지 않는다.

저장 제약 조회는 found/not_found/error로 구별한다. error면 checked 진입점에서 모델·생성·fallback 미호출, `failed/schemaError`, 빈 블록·0분·조회 실패 안내. 이는 현재 내부 응답이다. D2 구현 목표는 typed 조회 오류를 외부 HTTP503/PROFILE_LOOKUP_UNAVAILABLE로 전달하고 명시 재시도만 제공하는 것이다. 내부 전용 진단 전달은 아직 미구현이므로 schemaError 전체나 문구로 추론하지 않는다. not_found는 성공 조회 후 행 없음 등이며 건강 상태 확인이 아니다. 요청 []로 프로필 부상을 지우지 않는다.

## 5. 응답 전체 구조

정본 기계 자료: `schemas/response-v4.schema.json`. 생성 시 serialization mode, camelCase를 사용했다. 의미 검증은 아래 규칙/코어 validator와 함께 한다. 내부 `_...` 진단 필드는 전송하지 않는다.

| 위치 | 보존 필드 |
|---|---|
| envelope | contractVersion, routine, shortRoutineNotice |
| routine | routineDraftId, generationStatus, statusReasonCode, isFallback, totalEstimatedTime, summaryTitle, rationaleSummary, routineBlocks, warnings, adjunctDecisions, personalizationSuggestions |
| block | order, exerciseId, exerciseName, movementPattern, movementType, primaryMuscles, primaryMuscleDetails, equipmentType, defaultRestSec, prescription, exerciseRationale, substitutionCandidates, reasons, boosts, penalties |
| block 실행 | sessionPhase, adjunctOption, executionGroupId/Type/Position/Size, transitionRestSec, groupRestSec, trainingMethod, unilateralPrescription |
| prescription | setIndex, setType, targetReps, targetWeightKg, targetRir, targetRestSec, targetDurationSec, targetDistanceM, intervalWorkSec, intervalRecoverySec, intervalRounds |
| substitutionCandidates[] | exerciseId, exerciseName, reason |
| boosts[] / penalties[] | constraintCode, ruleCode, profileSignalCode, score, reason |
| trainingMethod | code, labelKo, purposeKo, instructionKo, pauseSec, eccentricSec, source(auto/manual/legacy), reasonCode |
| adjunctDecisions[] | option, status(applied/suppressed), reasonCode, slotId, exerciseId |
| personalizationSuggestions[] | suggestionType, code, reasonKo, evidenceSetCount, comparisonMaxSetCount, requiresConfirmation |

`generationStatus=success/fallback/failed`; reason enum은 `none/llmTimeout/schemaError/networkError/emptyCandidate` 5개다. reasonCode가 붙는 모든 하위 객체를 이 enum 하나로 뭉치지 않는다. 실패 빈 루틴은 0분. fallback은 fallback/true이며 원인 보존. 출력 미지원·조회 오류를 정상 성공으로 바꾸지 않는다.

optional 필드의 생략/null/빈 배열은 추출 schema와 golden을 따른다. DTO 기본값 주입으로 의미를 바꾸지 않는다. 저장→조회→편집/교체→실행에서도 같은 의미를 유지한다. 단순 HTTP200이나 Pydantic 통과만으로 구성 검증까지 완료된 것은 아니다.

## 6. 처방·그룹·운동방법

| setType | 표시/동작 | 유실 방지 |
|---|---|---|
| working/warmup | 반복·중량·RIR·휴식 | null 중량을 0kg 실측으로 단정하지 않음 |
| timed | targetDurationSec 초 | 호환 targetReps가 있어도 횟수로 표시하지 않음 |
| distance | targetDistanceM 미터 | 초·횟수로 변환하지 않음 |
| interval | work/recovery 초, rounds, 총 targetDurationSec | 마지막 휴식을 임의로 더하지 않음; 코어 계산 보존 |

`targetReps`는 현 모델에서 시간형에도 양의 정수로 존재한다. 필드를 삭제하는 대신 setType별 렌더링을 한다. 그룹은 ID/type/size/position과 모든 소속 블록을 원자적으로 저장·표시한다. single/superset/circuit를 이름만 바꾸지 않는다. 그룹 휴식과 전환 휴식을 중복 적용하지 않는다. 기구 점유 가능성 검사를 우회하지 않는다.

운동방법은 정지/하강 시간과 지시·선택 출처를 함께 표시한다. 상급자라도 부상·DOMS·컨디션 제한이 우선이며 manual이 이를 해제하지 않는다. 상세값을 임의 기본값으로 바꿔 실행하지 않는다.

## 7. 짧은 루틴 및 사용자 확인

`shortRoutineNotice` 필드: isShortRoutine, requestedTimeMin, plannedTimeMin, thresholdRatio(0.5), messageKo, targetExpansionOptions.

success/fallback + 비어 있지 않은 루틴 + `2*plannedTimeMin < requestedTimeMin`만 true다. 정확히 50%면 false. failed/empty는 짧은 성공이 아니다. 0분의 nonempty 성공은 오류다. 기존 underfill 경고와 이 플래그는 별개다.

선택지: code(addQuadriceps/addHamstring/addCalves), labelKo, proposedTargetMuscles, requiresConfirmation=true, requiresRegeneration=true, availability=notEvaluated. 짧은 루틴 모두에 항상 제공되지 않는다. 현재 제한 없는 둔근 단독/근비대/gym/standard/normal 또는 high 및 서버 확인 문맥에서만 가능하다. 조건·메모가 있거나 문맥 미확인이면 빈 배열일 수 있다.

BE는 서버 보관 요청·응답·프로필의 최신 상태로 `build_confirmed_expansion_request`를 검사한다. true 동의+현재 유효 code만 인정한다. 변경 가능 값은 targetMuscles/targetSplitLabel뿐이며 장비·부상·DOMS·시간·goal 등은 그대로다. helper는 생성하지 않으므로 checked 경로로 새 생성하고 모든 제약을 다시 검사한다. 원래 루틴은 자동 덮어쓰지 않는다. FE는 옵션 존재만으로 운동 가능/분량 충족을 보장하지 않는다.

## 8. ID와 데이터 버전

숫자 ID는 `exerciseId` 문자열로 원래 값을 보존한다. V24ROW는 다른 namespace다. `fixtures/catalog-identity-bridge.json`의 approved + source/target identity 일치만 enrichment에 사용한다. candidate/rejected/ambiguous 또는 stale/경합 연결을 승인으로 간주하지 않는다. 장비·주동근 보정값을 원천값으로 되돌리지 않는다.

현재 46행 = approved11/candidate34/rejected1. 74/124는 같은 catalog를 공유해도 이력 숫자 ID는 분리한다. 98→006191 rejected, 98→003251 candidate다. 125/151은 identity 승인이지 추천 admission 승인이 아니다. V24ROW 결과를 이름이나 역방향 N:1로 숫자 ID 자동 변환하지 않는다. D6에 따라 approved 연결·실제 admission·원 숫자 ID를 보존하는 역변환까지 통합 검증한 범위만 제공한다. 그 범위 manifest는 소비자 연결 시 작성할 산출물이며 이 패키지가 승인11의 제공 가능성을 보증하지 않는다. 미연결 ID의 저장은 거절한다.

운영 MySQL을 SQLite로 교체하는 요청이 아니다. 후보·프로필·메타데이터 소비 방식과 버전/해시를 고정하고 실제 DB 표본으로 검증한다. 이 패키지는 전체 카탈로그의 최신성·저작권·배포 허가를 새로 판정하지 않는다.

## 9. 구 경로와 오류

생성 전 `check_legacy_request`, 생성 후 `check_legacy_response`로 미지원 필드를 확인한다. 현 함수는 검사 결과를 반환할 뿐 HTTP 자동 차단은 구현하지 않는다. v4 envelope를 벗겨 legacy로 보내거나 unsupported 항목을 제거하면 안 된다.

기존 반복 working/warmup·single 등 표현 가능한 경우만 legacy 후보가 된다. 그룹·방법·세부 부위·시간/거리/interval·부가 목적·개인화·편측·soleus DOMS·짧음 안내 등이 유실되면 구 경로 거절 대상이다. 요청검사 통과만으로 사후 결과가 호환된다고 간주하지 않는다.

D1~D4 확정: legacy도 조회 실패 fail-closed, 생성 전 실효 요청/생성 후 응답 capability 검사. 조회 실패는 HTTP503 + `PROFILE_LOOKUP_UNAVAILABLE`, capability 미지원은 HTTP422 + `UNSUPPORTED_CONTRACT_CAPABILITY`. 외부 오류는 `{error:{code,messageKo,retryable,requestId}}`이며 capability 오류에 requiredContractVersion/unsupportedFields를 함께 제공한다. 기계 형식은 `schemas/external-error.schema.json`이다. 이 형식은 신규 HTTP 구현 목표이며 현재 코어 statusReasonCode 5개를 확장하지 않는다. 입력오류422와 후보 없음의 도메인 failed는 조회 실패503과 구별한다. 원인별 typed 전달·오류 응답·legacy 차단은 [07 구현 범위](07-legacy-implementation-scope.md)의 통합 작업이다.

D4: 별도 v4 경로에서 `routine-core-v4-draft-v3` 응답 버전을 확인한다. 임의 stable-v4 변경이나 장애 시 v4→legacy 자동 강등은 금지한다. 응답 버전 불일치 시 실행·저장하지 않고 호환성 실패로 처리한다. D2: 자동 생성 재시도 없이 사용자의 명시 재시도를 받으며 최신 제약을 다시 조회한다. requestId는 서버 발행 상관 식별자이며 개인 식별 정보를 담지 않는다. stack/SQL/자격증명은 외부 오류에 넣지 않는다. BE fallback 상세 기준은 [검증 방법](04-verification.md)의 BF01~20을 따른다.

이번에 확정한 외부 코드 목록:

| code | HTTP | retryable | FE 처리 |
|---|---|---|---|
| PROFILE_LOOKUP_UNAVAILABLE | 503 | true | 조회 실패 안내·사용자 명시 재시도, 자동 생성 금지 |
| UNSUPPORTED_CONTRACT_CAPABILITY | 422 | false | 요구 계약 버전·미지원 필드 안내, 실행/저장/자동 강등 금지 |

일반 입력 오류422의 공통 BE 코드까지 새로 정의한 목록은 아니다. schema의 code가 문자열인 것은 다른 전송 오류와의 형식 공존을 위한 것이며, 알 수 없는 코드를 위 두 원인으로 추정하지 않는다. 미지원 코드는 일반 실패로 표시하고 자동 재시도하지 않는다. 소비자가 추가 코드를 사용할 때는 코드표와 오류 fixture를 버전화한다.

## 10. 준비됐으나 미활성인 기능

편측 `unilateralPrescription`: quantityMode(repsPerSide/durationPerSide), sideCount=2, setUnit=bothSides, sideOrder(leftThenRight/rightThenLeft), sideSwitchSec, secondsPerRep, restPlacement=afterBothSides. 한 prescription은 양쪽 완료 1세트. 3×한쪽12회를 6세트/24회로 바꾸지 않는다. 중량은 두 배 하지 않는다. 시간은 양쪽 수행+전환, 마지막 세트 뒤 휴식 없음. 그룹 안에서도 한 운동 차례를 양쪽 완료한 뒤 이동한다.

현 exact-ID registry는 비어 있으며 자동 추천에서 활성화하지 않았다. FE 렌더링의 예약/합성 QA는 가능하나 미지원 시 조용히 반복으로 강등하면 안 된다. 실제 활성화는 별도 검수 후다.

equipmentSetupChecks(케이블 lowPulley/ankleStrap/stableHandSupport), exerciseFamiliarity는 **미연결 설계**다. 현재 public 요청에 보내면 추가 키로 거절된다. 킥백 admission이나 세팅 UI를 이번 연결에 슬쩍 추가하지 않는다.

## 이전 초안

[이전 초안 원본 링크·버전별 변경](references/prior-drafts.md). 원본은 변경하지 않았고 ZIP에는 현재 통합 계약 한 개만 포함했다. D1~D7 정책은 확정, HTTP·BE/FE 구현과 통합 검증은 미완료다.

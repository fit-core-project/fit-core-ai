# FE 연결 요청

웹 사용자 경험 기준이다. 레이아웃·컴포넌트 구현은 FE에서 정하되 데이터 의미와 실패 상태를 보존한다. 코어 검증은 브라우저 QA를 대신하지 않는다.

| ID / 우선순위 | 요청·이유 | 완료 확인 |
|---|---|---|
| FE01 필수 | setType별 반복/초/미터/인터벌 표시. timed의 호환 targetReps를 횟수로 표시하지 않음 | timed30초, distance20m, interval20초운동+10초휴식×2(총50초) 합성 fixture를 모바일/데스크톱에서 확인. 저장 후 재조회 동일 |
| FE02 필수 | 슈퍼세트/서킷 묶음·순서·라운드·그룹 휴식·전환 표시 | 2블록 한 그룹이 떨어지거나 두 휴식이 중복되지 않음. 1개 누락 그룹을 일반세트 성공으로 보이지 않음. 코어가 single로 제공하면 그 결과/이유 표시 |
| FE03 필수 | 운동방법 labelKo/purposeKo/instructionKo와 pause/eccentric 시간 표시 | 같은 벤치프레스라도 정지 반복 지시가 세트 실행·재조회까지 유지. 이름만 바꾸거나 일반 반복으로 누락하지 않음 |
| FE04 필수 | 요청시간과 예상시간 함께 표시, 짧음 안내·경고 보존 | 50→20분 배너, 정확25분 C=false, failed0분은 운동 시작 버튼 비활성. 기존 부족 설명은 C=false라도 숨기지 않음 |
| FE05 필수 | 서버 targetExpansionOptions가 있을 때만 추가 부위 선택·명시 확인 후 재생성 | 옵션 []면 선택 버튼 없음. 선택 전 기존 루틴 유지, 취소 시 그대로, 재생성 실패도 표시. 제한이 바뀐 뒤 오래된 옵션 재사용 거절 안내 |
| FE06 필수 | 사용 가능 장비 whitelist를 명시 전달, 스미스/일반 머신 구별 | 머신만 체크한 캡처에서 SMITH_MACHINE 없음. UI 선택→BE→AI 값 일치. []가 무장비라는 오해 방지 |
| FE07 필수 | 부위 한글 표시 + 공개18/상세값 분리. 운동명 표시, ID는 사용자 운동명 대용으로 쓰지 않음 | side-deltoids는 측면 어깨, front-deltoids로 치환하지 않음. public배열 비어도 상세값으로 표시. head/knees/soleus는 타깃 선택 불가 |
| FE08 필수 | DOMS20 입력 1/2/3, 해당 없음은 기록 없음. 부상22는 강도 없는 별도 상태 | 좌우 가자미근 DOMS 보존, 머리·무릎 DOMS 없음. 프로필 부상은 DOMS 선택/초기화로 해제되지 않음. 저장값 재진입 일치 |
| FE09 필수 | loading/success/fallback/failed 및 외부 error 구별. HTTP503/PROFILE_LOOKUP_UNAVAILABLE에는 명시 재시도 제공, 자동 생성 재시도 없음 | 실패0분을 성공으로 보이지 않음. fallback 표시 유지. HTTP422 capability 오류는 지원 버전/필드 안내. `external-error-golden.json`으로 원인·재시도·요청 식별자 확인. 실제 HTTP 구현과 브라우저 QA는 별도 필요 |
| FE10 필수(해당 기능 연결 시) | focus/method/관리 부상 추가 입력은 D5의 adapter에 맞춰 점진 노출 | 처음부터 내부 필드 모두 노출하지 않음. 미지원 입력을 보내지 않고 비활성 이유 표시. 슈퍼세트·보조 목적 상충은 서버 검사와 일치 |
| FE11 권장(활성화 전 필수) | 편측 양쪽 완료1세트, 한쪽당 반복/시간·좌우 순서 표시 | 3세트×한쪽12회를 6세트/24회로 표시하지 않음. 중량2배 금지. 합성 QA만; registry 활성화/킥백 추천 추가하지 않음 |
| FE12 권장 | 뒤로가기·재생성 취소·입력 보존 및 중복 클릭 제어 | 상세→루틴 복귀 시 선택/스크롤/입력 유지. 브라우저 뒤로가기·요청중 취소·오류후재시도·모바일 가로넘침 QA |

## 한글 부위 표시 기준

가슴 chest / 등 upper-back / 허리 lower-back / 승모근 trapezius / 전면 어깨 front-deltoids / 후면 어깨 back-deltoids / 이두 biceps / 삼두 triceps / 전완 forearm / 복부 abs / 옆구리 obliques / 둔근 gluteal / 앞 허벅지 quadriceps / 뒤 허벅지 hamstring / 내전근 adductor / 외전근 abductors / 종아리 calves / 목 neck.
상세 side-deltoids는 측면 어깨. DOMS의 left-soleus/right-soleus는 좌/우 가자미근. 머리·무릎은 부상 입력 전용이다. 코드값 자체는 번역하지 않고 label만 한글로 표시한다.

## QA 산출물

항목별 viewport·입력 fixture·네트워크 요청/응답·화면 캡처·저장 재조회 결과를 남긴다. 사용자 원문/식별 개인정보를 넣지 않은 합성 계정으로 수행한다. 앱 전체 UI 개편이 아니라 새 코어 의미가 웹에서 사라지지 않는지 먼저 확인한다. 실제 웹 통합 QA는 아직 수행되지 않았다.

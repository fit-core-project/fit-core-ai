# Supplement AI Architecture Audit

작성일: 2026-07-03
대상: Fit-Core AI supplement chat
범위: DB / query understanding / retrieval / prompt / response composer / eval fixture
상태: 구조 진단 완료, 재설계 권장

## 결론

현재 보충제 AI는 "근거 없는 답변을 방어하는 안전장치"는 어느 정도 갖췄지만, 제품 수준의 약사 챗봇으로 보기에는 지식 기반과 정규화 구조가 너무 좁고 불안정하다.

완전 폐기보다는 아래 방식이 맞다.

1. 현재 살릴 것
   - safety wording guard
   - weak/refusal answer recovery
   - `mode=full + sources` 강제 eval
   - 57-case fixture 방향성
2. 다시 세울 것
   - supplement canonical SSOT
   - ingredient / interaction / safety rule DB schema
   - retrieval scoring policy
   - intent별 response template
   - full 57-case category eval gate

즉, 프롬프트부터 고치는 작업은 우선순위가 낮다. 먼저 DB와 정규화가 흔들리지 않아야 한다.

## 실행 근거

### 원천 데이터 규모

명령:

```bash
python3 - <<'PY'
import json
from pathlib import Path
for p in sorted(Path('data/documents').glob('supplement_*.json')):
    data=json.loads(p.read_text(encoding='utf-8'))
    print(p.name, 'rows=', len(data), 'keys=', sorted(data[0].keys()) if data else [])
PY
```

결과:

| File | Rows | 의미 |
|---|---:|---|
| `supplement_ingredient_profiles.json` | 10 | 실제 성분 profile |
| `supplement_interaction_rules.json` | 12 | 병용/상호작용 rule |
| `supplement_safety_rules.json` | 12 | 질환/상황 safety rule |

현재는 총 34개 문서다. 제품형 보충제 챗봇이라기보다는 10개 대표 성분 안전 응답기에 가깝다.

### 57-case 결정론 평가

LLM을 빈 응답으로 고정하고, 현재 DB/retrieval/response composer만으로 57개 fixture를 평가했다.

결과:

| Category | Pass |
|---|---:|
| `known_single` | 10/10 |
| `multi_supplement` | 8/10 |
| `drug_interaction` | 5/8 |
| `condition_risk` | 4/10 |
| `unknown_partial_known` | 3/7 |
| `symptom_after_intake` | 7/7 |
| `llm_naturalness` | 2/5 |
| Total | 39/57 |

해석:

- 단일 known 성분은 꽤 안정적이다.
- 질환/약물/부분 unknown/자연어 추천에서 빈 구멍이 크다.
- 이 실패는 단순 프롬프트 문제가 아니라 DB coverage와 entity normalization 문제다.

## 주요 문제

### P0. Canonical SSOT가 없다

`query_understanding.py`의 canonical과 JSON profile canonical이 서로 어긋난다.

확인 결과:

| 구분 | 값 |
|---|---|
| parser에는 있지만 profile canonical에 없음 | `caffeine`, `calcium`, `probiotics`, `protein`, `vitamin a`, `vitamin c`, `vitamin e` |
| profile canonical에는 있지만 parser 기준과 다름 | `probiotic`, `whey` |

원인:

- ingredient profile에 명시적 `canonical` field가 없다.
- 현재 `_canonical_from_ingredient()`가 첫 번째 ASCII alias를 canonical처럼 사용한다.
- 그래서 `ING_PROTEIN`은 `protein`이 아니라 `whey`로, `ING_PROBIOTIC`은 `probiotics`가 아니라 `probiotic`으로 취급될 수 있다.

영향:

- 같은 성분인데 retrieval/filter/response composer에서 다른 성분처럼 보인다.
- "프로틴", "유산균" 케이스가 운 좋게 통과하더라도 구조적으로 불안정하다.

권장:

- ingredient profile에 `canonical`을 필수 field로 추가한다.
- `aliases`는 별칭일 뿐 canonical 산출에 쓰지 않는다.
- parser alias registry도 JSON/DB에서 생성하거나, 최소한 schema test로 양방향 일치 검증한다.

### P0. 평가 fixture가 DB coverage보다 넓다

57개 fixture는 다음 성분/조건을 기대한다.

- `vitamin a`, `vitamin c`, `vitamin e`
- `theanine`, `melatonin`, `vitamin b`
- `eaa`, `glutamine`
- `nmn`, `quercetin`, `nac`, `berberine`
- `blood pressure medication`, `diabetes medication`, `aspirin`
- `digestive symptom`

하지만 현재 KB에는 이 중 상당수가 profile/rule로 존재하지 않는다.

영향:

- LLM이 그럴듯하게 말해도 source evidence가 없거나, 엉뚱한 기존 rule이 끌려온다.
- `condition_risk`, `unknown_partial_known`, `llm_naturalness`가 약하다.

권장:

- fixture가 요구하는 entity를 `covered / partial / unknown`으로 분류한다.
- covered entity는 profile/rule을 만든다.
- unknown entity는 "확인 불가 + 성분표/함량 요청" template으로 빠지게 한다.

### P0. Retrieval이 evidence matching보다 "비슷한 문서 끌어오기"에 가깝다

현재 curated JSON fallback은 Chroma/reranker 없이 BM25 + priority scoring으로 동작한다.

관찰된 실패 예:

- `혈압약 + 마그네슘` 질문에 갑상선약/항생제 마그네슘 rule이 답변 중심으로 노출됨.
- `당뇨약 + 비타민D` 질문에 종합비타민 overlap rule이 노출됨.
- `NMN` 단독 질문에 유산균/항생제 rule이 노출됨.

영향:

- 사용자는 "내 질문을 이해하지 못했다"고 느낀다.
- 약물/질환 영역에서는 잘못된 안심보다 낫더라도, 엉뚱한 주의문은 제품 신뢰를 떨어뜨린다.

권장:

- retrieval 전 단계에 intent/entity coverage decision을 둔다.
- `covered_exact`: exact profile/rule 응답
- `covered_partial`: 아는 성분만 설명하고 unknown은 분리
- `unsupported`: 근거 없음, 제품명/성분표/함량 요청
- 약물/질환은 exact pair rule이 없으면 일반 병용 답변으로 확대하지 말고 "복용약/질환명/제품명 확인 필요"로 후퇴한다.

### P1. Prompt가 제품 정책을 잠그지 못한다

현재 system prompt는 안전한 일반 원칙은 담고 있지만 intent별 출력 정책은 약하다.

현재 prompt 성격:

- "verified documents를 근거로 자연스럽게 답하라"
- "질병 치료/예방/진단 주장 금지"
- "자료 부족하면 WEB_SEARCH_REQUIRED"

부족한 점:

- known timing, interaction, condition risk, unknown product, symptom after intake의 답변 구조가 다르지 않다.
- "무엇을 반드시 말해야 하는지"가 intent별로 잠겨 있지 않다.
- source coverage가 없을 때 답변하지 말아야 하는 기준이 약하다.

권장:

- LLM prompt를 하나로 크게 두기보다 intent별 response template을 먼저 적용한다.
- LLM은 template 내 `shortExplanation` 또는 `userFriendlySummary`만 작성하게 제한한다.
- high-risk intent에서는 LLM 자유도를 낮춘다.

### P1. Response composer가 하드코딩 rule pile로 커지고 있다

`response_composer.py`는 현재 안전 문구 복구에 강점이 있지만, 특정 case direct answer가 늘어나고 있다.

문제:

- 케이스가 늘수록 코드가 knowledge base를 대체한다.
- DB를 고도화해도 composer hardcode와 충돌할 수 있다.

권장:

- direct answer는 code가 아니라 DB rule action/template으로 이동한다.
- 예:
  - `answer_template_id`
  - `required_mentions`
  - `avoid_phrases`
  - `action`: `space_apart`, `consult`, `avoid_until_confirmed`, `ask_label`

### P1. Web fallback은 현재 제품 기준으로 위험하다

현재 `DuckDuckGoSearchRun`이 optional이고, dependency가 없으면 disabled된다.

문제:

- 아무 웹 검색 결과를 약학 답변 근거로 쓰는 것은 검증/출처/재현성이 약하다.
- 의료/약학 영역에서는 검색 fallback보다 source allowlist 기반 문서화가 맞다.

권장:

- web fallback은 제품 기본값에서 끈다.
- 필요하면 `official_source_ingestion` 루프로 따로 운영한다.
- source refs를 KB row에 명시하고, LLM 답변에는 source id를 숨기더라도 내부 evidence에는 남긴다.

## 재설계 권장안

### 1단계. Canonical + Coverage 잠금

먼저 아래 문서를 active contract로 만든다.

```text
supplement_canonical_entities_v1.json
```

필수 field:

```json
{
  "canonical": "magnesium",
  "displayNameKo": "마그네슘",
  "entityType": "supplement_ingredient",
  "aliases": ["마그네슘", "magnesium", "Mg"],
  "coverageStatus": "covered",
  "riskDefault": "moderate"
}
```

규칙:

- parser, DB, eval fixture는 이 canonical만 사용한다.
- aliases는 입력 인식용이다.
- coverageStatus는 `covered / partial / unsupported`로 둔다.

### 2단계. KB Schema v2

성분 profile:

```json
{
  "canonical": "iron",
  "displayNameKo": "철분",
  "category": "mineral",
  "commonUses": [],
  "timingGuidance": [],
  "commonSideEffects": [],
  "highRiskContexts": [],
  "evidenceLevel": "general_consensus",
  "sourceRefs": []
}
```

상호작용 rule:

```json
{
  "id": "INT_IRON_CAFFEINE",
  "subject": "iron",
  "object": "caffeine",
  "objectType": "food_or_compound",
  "severity": "moderate",
  "mechanism": "...",
  "action": "space_apart",
  "userMessageTemplate": "...",
  "requiredMentions": ["철분", "커피", "흡수", "간격"],
  "sourceRefs": []
}
```

안전 rule:

```json
{
  "id": "SAFE_KIDNEY_MAGNESIUM",
  "trigger": {"type": "condition", "canonical": "kidney disease"},
  "affectedEntity": "magnesium",
  "severity": "high",
  "action": "consult_before_use",
  "userMessageTemplate": "...",
  "sourceRefs": []
}
```

### 3단계. Answer Policy Router

질문 처리 순서:

1. parse entities/intents
2. coverage classify
3. exact rule lookup
4. safety rule lookup
5. response template compose
6. LLM polishing only for low/moderate risk
7. validator
8. response with internal evidence

high-risk answer는 LLM 단독 생성 금지.

### 4단계. 57-case Eval Gate

최소 gate:

| Category | MVP Gate |
|---|---:|
| `known_single` | 10/10 |
| `multi_supplement` | 10/10 |
| `drug_interaction` | 8/8 |
| `condition_risk` | 10/10 |
| `unknown_partial_known` | 7/7 |
| `symptom_after_intake` | 7/7 |
| `llm_naturalness` | 4/5 이상 |

실패 시 제품 반영 금지.

## 다음 작업 순서

0. source map 기준 파일 확인: `docs/research/supplement_source_map_v1_2026-07-03.json`
1. `supplement_canonical_entities_v1.json` 작성
2. 기존 3개 JSON에 `canonical`, `displayNameKo`, `coverageStatus` 추가
3. parser alias registry를 canonical JSON 기준으로 생성하거나 일치 test 추가
4. 57-case fixture 기준 누락 entity 목록 보강
5. `coverage classifier` 구현
6. exact match가 없을 때 엉뚱한 rule이 답변 중심에 오지 않도록 retrieval guard 추가
7. 57-case deterministic eval을 pass 기준까지 끌어올림
8. 그 뒤에야 LLM prompt 고도화

## 현재 판단

처음부터 완전히 새로 만드는 것은 비용이 크고, 이미 만든 safety/eval guard를 버릴 이유는 없다.

다만 "현재 구조에 프롬프트만 더 붙이는 것"은 비추천이다. 지금 필요한 것은 보충제 지식 시스템의 뼈대 재정렬이다.

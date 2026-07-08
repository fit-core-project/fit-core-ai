# Food Search Baseline v1

- Evalset: 1000 queries
- Overall accuracy: 863/1000 (86.3%)

## Accuracy by Type

| Type | Pass | Total | Accuracy |
|---|---:|---:|---:|
| A | 200 | 200 | 100.0% |
| B | 184 | 200 | 92.0% |
| C | 200 | 200 | 100.0% |
| D | 166 | 200 | 83.0% |
| E | 113 | 200 | 56.5% |

## Accuracy by Major Category

| Major category | Pass | Total | Accuracy |
|---|---:|---:|---:|
| 어패류 및 기타 수산물 | 94 | 112 | 83.9% |
| 버섯류 | 50 | 70 | 71.4% |
| 곡류 | 40 | 66 | 60.6% |
| 감자 및 전분류 | 54 | 64 | 84.4% |
| 견과 및 종실류 | 59 | 62 | 95.2% |
| 두류 | 52 | 61 | 85.2% |
| 육류 | 53 | 53 | 100.0% |
| 채소류 | 43 | 43 | 100.0% |
| 과일류 | 31 | 40 | 77.5% |
| 난류 | 24 | 35 | 68.6% |
| 빵 및 과자류 | 16 | 25 | 64.0% |
| 음료 및 차류 | 23 | 23 | 100.0% |
| 밥류 | 18 | 22 | 81.8% |
| 국 및 탕류 | 19 | 20 | 95.0% |
| 기타 | 16 | 19 | 84.2% |
| 생채·무침류 | 17 | 19 | 89.5% |
| 볶음류 | 17 | 18 | 94.4% |
| 전·적 및 부침류 | 14 | 18 | 77.8% |
| 죽 및 스프류 | 16 | 17 | 94.1% |
| 찌개 및 전골류 | 15 | 17 | 88.2% |
| 해조류 | 17 | 17 | 100.0% |
| 찜류 | 14 | 16 | 87.5% |
| 당류 | 13 | 13 | 100.0% |
| 구이류 | 11 | 12 | 91.7% |
| 면 및 만두류 | 11 | 12 | 91.7% |
| 조미료류 | 12 | 12 | 100.0% |
| 나물·숙채류 | 11 | 11 | 100.0% |
| 유지류 | 11 | 11 | 100.0% |
| 차류 | 11 | 11 | 100.0% |
| 유제품류 및 빙과류 | 10 | 10 | 100.0% |
| 김치류 | 9 | 9 | 100.0% |
| 장류, 양념류 | 9 | 9 | 100.0% |
| 장아찌·절임류 | 9 | 9 | 100.0% |
| 젓갈류 | 9 | 9 | 100.0% |
| 튀김류 | 9 | 9 | 100.0% |
| 조림류 | 8 | 8 | 100.0% |
| 수·조·어·육류 | 6 | 6 | 100.0% |
| 우유류 | 6 | 6 | 100.0% |
| 곡류, 서류 제품 | 4 | 4 | 100.0% |
| 두류, 견과 및 종실류 | 1 | 1 | 100.0% |
| 채소, 해조류 | 1 | 1 | 100.0% |

## Failure Stage Distribution

| Stage | Count |
|---|---:|
| 정규화 | 77 |
| canonical | 4 |
| vector | 0 |
| reranker | 56 |
| 데이터 | 0 |
| 기대값의심 | 0 |
| 미분류 | 0 |

## Representative Failures

### 정규화

| Query | Expected | Actual | Evidence |
|---|---|---|---|
| 감자 대지 | {"must_not_force": "raw ingredient", "blocked_names": ["감자 대지 생것"]} | 감자 대지 생것 | ambiguous query resolved to a blocked raw ingredient |
| 감자 자색 | {"must_not_force": "raw ingredient", "blocked_names": ["감자 자색 생것"]} | 감자 자색 생것 | ambiguous query resolved to a blocked raw ingredient |
| 고구마 | {"must_not_force": "raw ingredient", "blocked_names": ["고구마 생것"]} | 고구마 생것 | ambiguous query resolved to a blocked raw ingredient |
| 고구마 분질(밤) 고구마 | {"must_not_force": "raw ingredient", "blocked_names": ["고구마 분질(밤) 고구마 생것"]} | 고구마 분질(밤) 고구마 생것 | ambiguous query resolved to a blocked raw ingredient |
| 돼지감자 | {"must_not_force": "raw ingredient", "blocked_names": ["돼지감자 생것"]} | 돼지감자 생것 | ambiguous query resolved to a blocked raw ingredient |

### canonical

| Query | Expected | Actual | Evidence |
|---|---|---|---|
| 볶은 콩(대두) 노란콩 | "콩(대두) 노란콩 볶은것" | None | expected was not returned by canonical lookup |
| 볶은 콩(대두) 대풍 | "콩(대두) 대풍 볶은것" | None | expected was not returned by canonical lookup |
| 화양적 한컵 | "화양적" | None | expected was not returned by canonical lookup |
| 붉은맛(큰죽합) 육 | ["붉은맛(큰죽합) 육 말린것 대표 평균", "붉은맛(큰죽합) 육 생것 대표 평균"] | None | expected was not returned by canonical lookup |

### reranker

| Query | Expected | Actual | Evidence |
|---|---|---|---|
| 삶은 국수 말린것을 | "국수 말린것을 삶은것" | 국수 말린것 | expected appears in vector top-k but reranker top-1 differs |
| 삶은 강낭콩 말린것을 | "강낭콩 말린것을 삶은것" | 강낭콩 말린것 | expected appears in vector top-k but reranker top-1 differs |
| 삶은 국수 소면 말린것을 | "국수 소면 말린것을 삶은것" | 국수 소면 말린것 | expected appears in vector top-k but reranker top-1 differs |
| 삶은 국수 우동 | "국수 우동 삶은것" | 국수 | expected appears in vector top-k but reranker top-1 differs |
| 삶은 땅콩 에 소금을 가한것 | "땅콩 삶은것에 소금을 가한것" | 소금 | expected appears in vector top-k but reranker top-1 differs |

## Evalset Quality Signal

- 기대값의심: 0 cases. These should be reviewed before using a future v2 evalset.

## Improvement Priority

| Priority | Stage | Potential XPASS | Rationale |
|---:|---|---:|---|
| 1 | 정규화 | 77 | Largest current deterministic failure bucket. |
| 2 | reranker | 56 | Largest current deterministic failure bucket. |
| 3 | canonical | 4 | Largest current deterministic failure bucket. |

# Food Search Baseline v2

- Evalset: 1000 queries
- Overall accuracy: 942/1000 (94.2%)
- v1 comparison: 863/1000 (86.3%) -> 942/1000 (94.2%)
- v2 refinement: preserve_ambiguous now passes same-base raw/basic ingredient matches.
- allowed_set expansion: E allowed sets include same-base CSV rows, excluding processed or compound terms.
- Still-fail contract: no-match, different-base, processed, and compound foods remain failures.

## Accuracy by Type

| Type | Pass | Total | Accuracy |
|---|---:|---:|---:|
| A | 200 | 200 | 100.0% |
| B | 184 | 200 | 92.0% |
| C | 200 | 200 | 100.0% |
| D | 166 | 200 | 83.0% |
| E | 192 | 200 | 96.0% |

## Accuracy by Major Category

| Major category | Pass | Total | Accuracy |
|---|---:|---:|---:|
| 어패류 및 기타 수산물 | 109 | 112 | 97.3% |
| 버섯류 | 67 | 70 | 95.7% |
| 곡류 | 48 | 66 | 72.7% |
| 감자 및 전분류 | 63 | 64 | 98.4% |
| 견과 및 종실류 | 61 | 62 | 98.4% |
| 두류 | 57 | 61 | 93.4% |
| 육류 | 53 | 53 | 100.0% |
| 채소류 | 43 | 43 | 100.0% |
| 과일류 | 40 | 40 | 100.0% |
| 난류 | 35 | 35 | 100.0% |
| 빵 및 과자류 | 16 | 25 | 64.0% |
| 음료 및 차류 | 23 | 23 | 100.0% |
| 밥류 | 18 | 22 | 81.8% |
| 국 및 탕류 | 19 | 20 | 95.0% |
| 기타 | 19 | 19 | 100.0% |
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
| 정규화 | 0 |
| canonical | 7 |
| vector | 3 |
| reranker | 48 |
| 데이터 | 0 |
| 기대값의심 | 0 |
| 미분류 | 0 |

## Representative Failures

### canonical

| Query | Expected | Actual | Evidence |
|---|---|---|---|
| 볶은 콩(대두) 노란콩 | "콩(대두) 노란콩 볶은것" | None | expected was not returned by canonical lookup |
| 볶은 콩(대두) 대풍 | "콩(대두) 대풍 볶은것" | None | expected was not returned by canonical lookup |
| 화양적 한컵 | "화양적" | None | expected was not returned by canonical lookup |
| 곤약(구약나물) 국수형 | {"allowed_base": "곤약(구약나물) 국수형", "disallow": "different_base_or_processed_or_compound_or_no_match", "blocked_terms": ["젓갈", "맛탕", "샐러드", "샌드위치", "김밥", "찌개", "전골", "볶음밥", "덮밥", "튀김", "장아찌", "절임"]} | None | ambiguous query returned no top-1 match |
| 동충하초 누에동충하초 | {"allowed_base": "동충하초 누에동충하초", "disallow": "different_base_or_processed_or_compound_or_no_match", "blocked_terms": ["젓갈", "맛탕", "샐러드", "샌드위치", "김밥", "찌개", "전골", "볶음밥", "덮밥", "튀김", "장아찌", "절임"]} | None | ambiguous query returned no top-1 match |

### vector

| Query | Expected | Actual | Evidence |
|---|---|---|---|
| 국수 우동 | {"allowed_base": "국수 우동", "disallow": "different_base_or_processed_or_compound_or_no_match", "blocked_terms": ["젓갈", "맛탕", "샐러드", "샌드위치", "김밥", "찌개", "전골", "볶음밥", "덮밥", "튀김", "장아찌", "절임"]} | 국수 | ambiguous query matched a different base ingredient |
| 국수 중국국수 | {"allowed_base": "국수 중국국수", "disallow": "different_base_or_processed_or_compound_or_no_match", "blocked_terms": ["젓갈", "맛탕", "샐러드", "샌드위치", "김밥", "찌개", "전골", "볶음밥", "덮밥", "튀김", "장아찌", "절임"]} | 국수 | ambiguous query matched a different base ingredient |
| 국수 칼국수 | {"allowed_base": "국수 칼국수", "disallow": "different_base_or_processed_or_compound_or_no_match", "blocked_terms": ["젓갈", "맛탕", "샐러드", "샌드위치", "김밥", "찌개", "전골", "볶음밥", "덮밥", "튀김", "장아찌", "절임"]} | 칼국수 | ambiguous query matched a different base ingredient |

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
| 1 | reranker | 48 | Largest current deterministic failure bucket. |
| 2 | canonical | 7 | Largest current deterministic failure bucket. |
| 3 | vector | 3 | Largest current deterministic failure bucket. |

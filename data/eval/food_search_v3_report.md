# Food Search v3 Cross-Validation Report

## Independence Check

- v3 evalset rows: 1000
- v2 source_food_id overlap: 0
- v2 query overlap: 0
- v3 unique queries: 1000
- v3 unique source_food_id: 735
- E compact single-ingredient variants used to fill after v2 exclusion: 23

## v2 vs v3 Accuracy

| Evalset | Pass | Total | Accuracy |
|---|---:|---:|---:|
| v2 current pipeline | 985 | 1000 | 98.5% |
| v3 current pipeline | 986 | 1000 | 98.6% |

## Accuracy by Type

| Type | v2 | v3 | Delta pct |
|---|---:|---:|---:|
| A | 200/200 (100.0%) | 200/200 (100.0%) | +0.0pp |
| B | 197/200 (98.5%) | 196/200 (98.0%) | -0.5pp |
| C | 200/200 (100.0%) | 200/200 (100.0%) | +0.0pp |
| D | 195/200 (97.5%) | 198/200 (99.0%) | +1.5pp |
| E | 193/200 (96.5%) | 192/200 (96.0%) | -0.5pp |

## Accuracy by Major Category

| Major category | v2 | v3 | Delta pct |
|---|---:|---:|---:|
| 감자 및 전분류 | 63/64 (98.4%) | 21/21 (100.0%) | +1.6pp |
| 견과 및 종실류 | 62/62 (100.0%) | 21/22 (95.5%) | -4.5pp |
| 곡류 | 62/66 (93.9%) | 28/28 (100.0%) | +6.1pp |
| 곡류, 서류 제품 | 4/4 (100.0%) | 0/0 (0.0%) | -100.0pp |
| 과일류 | 40/40 (100.0%) | 21/21 (100.0%) | +0.0pp |
| 구이류 | 12/12 (100.0%) | 17/17 (100.0%) | +0.0pp |
| 국 및 탕류 | 20/20 (100.0%) | 16/16 (100.0%) | +0.0pp |
| 기타 | 19/19 (100.0%) | 19/19 (100.0%) | +0.0pp |
| 김치류 | 9/9 (100.0%) | 16/16 (100.0%) | +0.0pp |
| 나물·숙채류 | 11/11 (100.0%) | 16/16 (100.0%) | +0.0pp |
| 난류 | 35/35 (100.0%) | 14/14 (100.0%) | +0.0pp |
| 당류 | 13/13 (100.0%) | 0/0 (0.0%) | -100.0pp |
| 두류 | 59/61 (96.7%) | 20/21 (95.2%) | -1.5pp |
| 두류, 견과 및 종실류 | 1/1 (100.0%) | 0/0 (0.0%) | -100.0pp |
| 면 및 만두류 | 12/12 (100.0%) | 17/17 (100.0%) | +0.0pp |
| 밥류 | 22/22 (100.0%) | 18/18 (100.0%) | +0.0pp |
| 버섯류 | 69/70 (98.6%) | 27/27 (100.0%) | +1.4pp |
| 볶음류 | 18/18 (100.0%) | 17/17 (100.0%) | +0.0pp |
| 빵 및 과자류 | 22/25 (88.0%) | 20/20 (100.0%) | +12.0pp |
| 생채·무침류 | 19/19 (100.0%) | 19/19 (100.0%) | +0.0pp |
| 수·조·어·육류 | 6/6 (100.0%) | 2/2 (100.0%) | +0.0pp |
| 어패류 및 기타 수산물 | 110/112 (98.2%) | 100/101 (99.0%) | +0.8pp |
| 우유류 | 6/6 (100.0%) | 0/0 (0.0%) | -100.0pp |
| 유제품류 및 빙과류 | 10/10 (100.0%) | 19/19 (100.0%) | +0.0pp |
| 유지류 | 11/11 (100.0%) | 19/19 (100.0%) | +0.0pp |
| 육류 | 53/53 (100.0%) | 105/110 (95.5%) | -4.5pp |
| 음료 및 차류 | 23/23 (100.0%) | 20/20 (100.0%) | +0.0pp |
| 장류, 양념류 | 9/9 (100.0%) | 3/3 (100.0%) | +0.0pp |
| 장아찌·절임류 | 9/9 (100.0%) | 14/15 (93.3%) | -6.7pp |
| 전·적 및 부침류 | 17/18 (94.4%) | 11/11 (100.0%) | +5.6pp |
| 젓갈류 | 9/9 (100.0%) | 12/12 (100.0%) | +0.0pp |
| 조림류 | 8/8 (100.0%) | 14/14 (100.0%) | +0.0pp |
| 조미료류 | 12/12 (100.0%) | 19/19 (100.0%) | +0.0pp |
| 죽 및 스프류 | 16/17 (94.1%) | 16/16 (100.0%) | +5.9pp |
| 찌개 및 전골류 | 17/17 (100.0%) | 15/15 (100.0%) | +0.0pp |
| 찜류 | 16/16 (100.0%) | 15/15 (100.0%) | +0.0pp |
| 차류 | 11/11 (100.0%) | 17/17 (100.0%) | +0.0pp |
| 채소, 해조류 | 1/1 (100.0%) | 0/0 (0.0%) | -100.0pp |
| 채소류 | 43/43 (100.0%) | 205/209 (98.1%) | -1.9pp |
| 튀김류 | 9/9 (100.0%) | 16/16 (100.0%) | +0.0pp |
| 해조류 | 17/17 (100.0%) | 37/38 (97.4%) | -2.6pp |

## Overfitting Judgment

- Judgment: 일반화됨
- Basis: v3 accuracy is within the 97~99% generalization band.
- v2 -> v3 delta: +0.1pp

## Failure Stage Distribution

| Stage | v2 residual | v3 failures | Delta |
|---|---:|---:|---:|
| canonical | 6 | 10 | +4 |
| reranker | 6 | 0 | -6 |
| vector | 3 | 4 | +1 |

## v3 Failure Reason Deltas

| Stage | Reason | v2 | v3 | Delta |
|---|---|---:|---:|---:|
| canonical | ambiguous query returned no top-1 match | 2 | 4 | +2 |
| canonical | expected was not returned by canonical lookup | 4 | 6 | +2 |
| vector | ambiguous query matched a different base ingredient | 3 | 4 | +1 |

## Representative v3 Failures

### canonical

| Query | Type | Expected kind | Expected | Actual | Reason |
|---|---|---|---|---|---|
| 구운 돼지고기 삼겹살(삼겹살) (팬) | B | exact | "돼지고기 삼겹살(삼겹살) 구운것(팬)" | 피자 불고기팬(R) | expected was not returned by canonical lookup |
| 구운 소 부산물 간 (팬) | B | exact | "소 부산물 간 구운것(팬)" | 피자 불고기팬(R) | expected was not returned by canonical lookup |
| 데친 두릅 참두릅 잎 | B | exact | "두릅 참두릅 잎 데친것" | None | expected was not returned by canonical lookup |
| 구운 오리고기 껍질 포함 (팬) | B | exact | "오리고기 껍질 포함 구운것(팬)" | 피자 불고기팬(R) | expected was not returned by canonical lookup |
| 렌즈콩(렌틸콩)수입산(인도산)빨간색말린것 | D | exact | "렌즈콩(렌틸콩) 수입산(인도산) 빨간색 말린것" | None | expected was not returned by canonical lookup |

### vector

| Query | Type | Expected kind | Expected | Actual | Reason |
|---|---|---|---|---|---|
| 밤 | E | preserve_ambiguous | {"allowed_base": "밤", "blocked_terms": ["젓갈", "맛탕", "샐러드", "샌드위치", "김밥", "찌개", "전골", "볶음밥", "덮밥", "튀김", "장아찌", "절임"], "disallow": "different_base_or_processed_or_compound_or_no_match"} | 밥 옥광 구운것 | ambiguous query matched a different base ingredient |
| 마늘 산(중) 구근 | E | preserve_ambiguous | {"allowed_base": "마늘 산(중) 구근", "blocked_terms": ["젓갈", "맛탕", "샐러드", "샌드위치", "김밥", "찌개", "전골", "볶음밥", "덮밥", "튀김", "장아찌", "절임"], "disallow": "different_base_or_processed_or_compound_or_no_match"} | 마늘 구근 다진것 | ambiguous query matched a different base ingredient |
| 호박 국수호박 | E | preserve_ambiguous | {"allowed_base": "호박 국수호박", "blocked_terms": ["젓갈", "맛탕", "샐러드", "샌드위치", "김밥", "찌개", "전골", "볶음밥", "덮밥", "튀김", "장아찌", "절임"], "disallow": "different_base_or_processed_or_compound_or_no_match"} | 국수 | ambiguous query matched a different base ingredient |
| 문어류문어육 | E | preserve_ambiguous | {"allowed_base": "문어류 문어 육", "blocked_terms": ["젓갈", "맛탕", "샐러드", "샌드위치", "김밥", "찌개", "전골", "볶음밥", "덮밥", "튀김", "장아찌", "절임"], "disallow": "different_base_or_processed_or_compound_or_no_match"} | 문어무침 | ambiguous query matched a different base ingredient |

## Conclusion

- Generalization: yes
- Ceiling signal: v3 failure distribution is similar to v2 residual by stage.
- Further improvement exists: yes; prioritize the largest v3 residual stage buckets.

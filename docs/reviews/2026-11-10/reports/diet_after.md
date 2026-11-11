# Diet stage eval — tests\evaluation\datasets\diet_eval_v1.jsonl

- llm: `cache`
- food engine: available=True thr=0.55  | cases: 266 | elapsed: 22.4s
- branches: {'llm': 250, 'llm_empty->deterministic+enrich': 16}  | item sources: {'curated': 77, 'db': 237, 'llm': 33, 'none': 2}

## Accuracy

| slice | n | scored | pass | acc | total-kcal acc | first failing stage |
|---|---|---|---|---|---|---|
| overall | 266 | 266 | 168 | 0.632 | 0.797 | S1_parse=12, S2_match=51, S3_grams=26, S4_kcal=9, pass=168 |
| dev | 129 | 129 | 82 | 0.636 | 0.806 | S1_parse=4, S2_match=26, S3_grams=14, S4_kcal=3, pass=82 |
| test | 137 | 137 | 86 | 0.628 | 0.788 | S1_parse=8, S2_match=25, S3_grams=12, S4_kcal=6, pass=86 |

## Item-level stage outcomes (required items)

- S1_parse: {'pass': 335, 'fail': 14}
- S2_match: {'pass': 282, 'fail': 53}
- S3_grams: {'pass': 228, 'fail': 29, 'n/a': 21, 'unknown': 4}
- S4_kcal: {'pass': 244, 'fail': 9}
- S1 failure causes (llm / reconcile / deterministic; case-level extras): {'llm': 9, 'deterministic': 6, 'case:extra_items:llm': 3}
- S1 failure reasons: {'fail:amount_mismatch': 5, 'fail:amount_missing': 1, 'fail:missing': 7, 'fail:merged_items': 2}
- items whose amount/unit were changed by _reconcile_amounts_with_text: 32
- S2 failure reasons: {'db:음식_017266': 2, 'db:음식_013425': 1, 'db:음식_009943': 1, 'db:음식_010030': 1, 'db:음식_010251': 1, 'llm_estimate(below_threshold)': 1, 'db:음식_016946': 1, 'db:원재_001349': 1, 'db:음식_013429': 1, 'llm_estimate(empty_macros)': 6, 'db:음식_007100': 3, 'db:원재_001728': 1, 'db:음식_011542': 3, 'db:음식_011468': 1, 'db:음식_007534': 1, 'db:음식_016255': 1, 'db:음식_010361': 1, 'db:원재_001132': 5, 'db:음식_010813': 1, 'db:음식_017012': 1, 'db:음식_013168': 2, 'db:음식_013885': 1, 'db:음식_016607': 1, 'no_macros(no_candidates_after_rerank)': 1, 'db:음식_014683': 1, 'db:음식_017516': 1, 'no_macros(empty_macros)': 1, 'db:원재_001183': 2, 'db:음식_012107': 1, 'db:음식_016949': 2, 'db:음식_012919': 1, 'db:음식_013088': 1, 'db:원재_000198': 1, 'db:음식_007798': 1, 'db:음식_016314': 1, 'db:음식_010942': 1}
- note: S4 kcal ranges are derived from the same gold rows x grams, so S4 can only fail for LLM-estimated items, rows with inconsistent macros, or kcal leaks; S4=0 does not by itself prove kcal is right beyond S2+S3.

## By tag

| tag | n | acc | total-kcal acc | first failing stage |
|---|---|---|---|---|
| flag:brand | 1 | 0.0 | 1.0 | S3_grams=1 |
| flag:invalid | 12 | 1.0 | 1.0 | pass=12 |
| flag:noncaloric | 2 | 0.5 | 0.5 | S1_parse=1, pass=1 |
| group:계란유제품 | 33 | 0.576 | 0.818 | S1_parse=3, S2_match=10, S3_grams=1, pass=19 |
| group:고기 | 22 | 0.773 | 0.909 | S2_match=1, S3_grams=4, pass=17 |
| group:과일 | 29 | 0.552 | 0.793 | S1_parse=1, S2_match=8, S3_grams=4, pass=16 |
| group:국찌개 | 23 | 0.739 | 0.87 | S1_parse=1, S2_match=4, S3_grams=1, pass=17 |
| group:면 | 22 | 0.727 | 0.864 | S2_match=1, S3_grams=5, pass=16 |
| group:밥 | 47 | 0.702 | 0.872 | S1_parse=1, S2_match=7, S3_grams=6, pass=33 |
| group:배달 | 18 | 0.778 | 0.889 | S1_parse=1, S2_match=2, S3_grams=1, pass=14 |
| group:보충제 | 8 | 0.75 | 0.875 | S2_match=1, S4_kcal=1, pass=6 |
| group:빵과자 | 31 | 0.516 | 0.742 | S1_parse=2, S2_match=10, S3_grams=2, S4_kcal=1, pass=16 |
| group:생선 | 9 | 0.667 | 0.778 | S1_parse=1, S2_match=2, pass=6 |
| group:음료 | 31 | 0.484 | 0.774 | S1_parse=7, S2_match=5, S3_grams=4, pass=15 |
| group:주류 | 14 | 0.786 | 0.929 | S2_match=2, S3_grams=1, pass=11 |
| group:채소샐러드 | 24 | 0.75 | 0.875 | S1_parse=1, S2_match=5, pass=18 |
| group:패스트푸드 | 10 | 0.2 | 0.4 | S1_parse=1, S2_match=3, S3_grams=4, pass=2 |
| group:편의점브랜드 | 19 | 0.316 | 0.421 | S2_match=6, S4_kcal=7, pass=6 |
| items:0 | 14 | 0.929 | 0.929 | S1_parse=1, pass=13 |
| items:1 | 174 | 0.632 | 0.753 | S1_parse=6, S2_match=30, S3_grams=20, S4_kcal=8, pass=110 |
| items:2 | 62 | 0.565 | 0.855 | S1_parse=2, S2_match=18, S3_grams=6, S4_kcal=1, pass=35 |
| items:3+ | 16 | 0.625 | 0.938 | S1_parse=3, S2_match=3, pass=10 |
| noise:끼니 | 31 | 0.581 | 0.774 | S1_parse=3, S2_match=7, S3_grams=3, pass=18 |
| noise:띄어쓰기 | 15 | 0.4 | 0.733 | S1_parse=3, S2_match=5, S3_grams=1, pass=6 |
| noise:부정 | 12 | 0.917 | 0.917 | S2_match=1, pass=11 |
| noise:슬랭 | 8 | 0.375 | 0.75 | S2_match=3, S3_grams=2, pass=3 |
| noise:시간 | 5 | 0.6 | 1.0 | S2_match=2, pass=3 |
| noise:없음 | 193 | 0.637 | 0.793 | S1_parse=6, S2_match=36, S3_grams=19, S4_kcal=9, pass=123 |
| noise:영어 | 7 | 0.286 | 0.714 | S1_parse=2, S2_match=2, S3_grams=1, pass=2 |
| noise:오타 | 6 | 0.833 | 0.833 | S1_parse=1, pass=5 |
| noise:이모지 | 5 | 0.8 | 0.8 | S2_match=1, pass=4 |
| noise:잡담 | 12 | 0.833 | 1.0 | S2_match=2, pass=10 |
| unit:g | 25 | 0.8 | 1.0 | S1_parse=1, S2_match=4, pass=20 |
| unit:ml | 11 | 0.636 | 0.727 | S1_parse=2, S2_match=2, pass=7 |
| unit:갑 | 1 | 0.0 | 0.0 | S2_match=1 |
| unit:개 | 87 | 0.621 | 0.828 | S1_parse=3, S2_match=19, S3_grams=8, S4_kcal=3, pass=54 |
| unit:공기 | 14 | 0.857 | 0.857 | S1_parse=1, S3_grams=1, pass=12 |
| unit:그릇 | 23 | 0.565 | 0.783 | S1_parse=1, S2_match=7, S3_grams=2, pass=13 |
| unit:마리 | 5 | 0.6 | 0.8 | S1_parse=1, S2_match=1, pass=3 |
| unit:모 | 2 | 1.0 | 1.0 | pass=2 |
| unit:반 | 20 | 0.7 | 0.75 | S1_parse=2, S2_match=2, S3_grams=1, S4_kcal=1, pass=14 |
| unit:범위 | 4 | 0.75 | 1.0 | S2_match=1, pass=3 |
| unit:병 | 5 | 1.0 | 1.0 | pass=5 |
| unit:봉지 | 3 | 0.0 | 0.0 | S2_match=1, S4_kcal=2 |
| unit:분수 | 1 | 1.0 | 1.0 | pass=1 |
| unit:송이 | 1 | 0.0 | 0.0 | S3_grams=1 |
| unit:숫자없음 | 65 | 0.462 | 0.723 | S1_parse=4, S2_match=16, S3_grams=14, S4_kcal=1, pass=30 |
| unit:스쿱 | 4 | 0.75 | 0.75 | S4_kcal=1, pass=3 |
| unit:인분 | 21 | 0.714 | 0.952 | S1_parse=2, S2_match=4, pass=15 |
| unit:잔 | 13 | 0.615 | 0.769 | S1_parse=1, S2_match=3, S3_grams=1, pass=8 |
| unit:장 | 4 | 0.25 | 0.75 | S2_match=2, S4_kcal=1, pass=1 |
| unit:조각 | 17 | 0.706 | 0.765 | S2_match=2, S3_grams=3, pass=12 |
| unit:줄 | 4 | 1.0 | 1.0 | pass=4 |
| unit:줌 | 3 | 0.667 | 1.0 | S2_match=1, pass=2 |
| unit:캔 | 11 | 0.727 | 0.909 | S1_parse=1, S2_match=1, S4_kcal=1, pass=8 |
| unit:컵 | 5 | 0.6 | 0.8 | S1_parse=1, S3_grams=1, pass=3 |
| unit:크기 | 11 | 0.364 | 0.727 | S2_match=2, S3_grams=5, pass=4 |
| unit:통 | 1 | 0.0 | 0.0 | S4_kcal=1 |
| unit:판 | 1 | 1.0 | 1.0 | pass=1 |
| unit:팩 | 6 | 0.667 | 0.667 | S2_match=2, pass=4 |
| unit:한글숫자 | 125 | 0.608 | 0.736 | S1_parse=5, S2_match=27, S3_grams=12, S4_kcal=5, pass=76 |

## Failures (first 400)

### dv1_001 [test] `S3_grams` — 점심 백반집에서 밥 한 공기랑 된장찌개 먹었어
- branch `llm`; parsed: 밥(1.0 공기), 된장찌개(1.0 )
  - rice: `밥` 1.0 공기 src=curated grams=210.0 kcal=265.8 → S1_parse=pass:literal; S2_match=pass:curated; S3_grams=pass:210g; S4_kcal=pass:265.8kcal vs [200.2, 425.0]
  - doenjang: `된장찌개` 1.0  src=db/음식_016312 (된장찌개) grams=100.0 kcal=46.0 → S1_parse=pass:no_literal_amount; S2_match=pass:db:음식_016312; S3_grams=fail:100g not in [300, 500]
  - total 311.8 vs [319.4, 683.0] (out)
### dv1_004 [dev] `S2_match` — 김볶밥 한 그릇 해치움 ㅋㅋ
- branch `llm`; parsed: 김볶밥(1.0 그릇)
  - kimchi_fried_rice: `김볶밥` 1.0 그릇 src=db/음식_017266 (김치 볶음밥 모듬) grams=210.0 kcal=273.0 → S1_parse=pass:literal; S2_match=fail:db:음식_017266
  - total 273.0 vs [410.8, 831.7] (out)
### dv1_011 [test] `S2_match` — 새우볶음밥 한그릇 뚝딱
- branch `llm`; parsed: 새우볶음밥(1.0 그릇)
  - fried_rice: `새우볶음밥` 1.0 그릇 src=db/음식_013425 (새우볶음) grams=350.0 kcal=290.5 → S1_parse=pass:literal; S2_match=fail:db:음식_013425
  - total 290.5 vs [243.4, 1118.8] (ok)
### dv1_012 [dev] `S3_grams` — 주먹밥 3개 먹었음
- branch `llm`; parsed: 주먹밥(3.0 개)
  - jumeokbap: `주먹밥` 3.0 개 src=db/음식_010056 (주먹밥) grams=630.0 kcal=850.5 → S1_parse=pass:literal; S2_match=pass:db:음식_010056; S3_grams=fail:630g not in [300.0, 450.0]
  - total 850.5 vs [348.7, 1034.6] (ok)
### dv1_013 [test] `S2_match` — 초밥 10피스
- branch `llm`; parsed: 초밥(10.0 피스)
  - sushi: `초밥` 10.0 피스 src=db/음식_009943 (초밥 한치) grams=300.0 kcal=450.0 → S1_parse=pass:literal; S2_match=fail:db:음식_009943
  - total 450.0 vs [334.8, 915.8] (ok)
### dv1_022 [dev] `S3_grams` — 짜장면 곱빼기
- branch `llm`; parsed: 짜장면(None )
  - jjajang: `짜장면` None  src=db/음식_010034 (짜장면) grams=650.0 kcal=767.0 → S1_parse=pass:no_literal_amount; S2_match=pass:db:음식_010034; S3_grams=fail:650g not in [800, 1050]
  - total 767.0 vs [758.2, 1483.4] (ok)
### dv1_025 [dev] `S3_grams` — 비빔냉면 한그릇+만두 4개
- branch `llm`; parsed: 비빔냉면(1.0 그릇), 만두(4.0 개)
  - bibim_naengmyeon: `비빔냉면` 1.0 그릇 src=db/음식_013746 (비빔냉면) grams=700.0 kcal=651.0 → S1_parse=pass:literal; S2_match=pass:db:음식_013746; S3_grams=fail:700g not in [400.0, 550.0]
  - mandu: `만두` 4.0 개 src=db/음식_015035 (만두 물만두) grams=120.0 kcal=234.0 → S1_parse=pass:literal; S2_match=pass:db:음식_015035; S3_grams=pass:120g; S4_kcal=pass:234.0kcal vs [81.4, 438.6]
  - total 885.0 vs [411.2, 1006.3] (ok)
### dv1_030 [test] `S3_grams` — pho 한 그릇
- branch `llm`; parsed: pho(1.0 그릇)
  - pho: `pho` 1.0 그릇 src=db/음식_011601 (쌀국수) grams=400.0 kcal=212.0 → S1_parse=pass:literal; S2_match=pass:db:음식_011601; S3_grams=fail:400g not in [600.0, 800.0]
  - total 212.0 vs [281.2, 568.2] (out)
### dv1_031 [dev] `S4_kcal` — 불닭볶음면 1봉
- branch `llm`; parsed: 불닭볶음면(1.0 봉)
  - buldak: `불닭볶음면` 1.0 봉 src=llm grams=None kcal=342.0 → S1_parse=pass:literal; S2_match=pass:llm_estimate(db empty_macros; sanity kept); S3_grams=n/a:llm_estimate; S4_kcal=fail:342.0kcal vs [410.8, 667.2] (llm macros, sanity kept)
  - total 342.0 vs [410.8, 667.2] (out)
### dv1_035 [dev] `S2_match` — 진짬뽕 한 봉지
- branch `llm`; parsed: 진짬뽕(1.0 봉지)
  - jin_jjamppong: `진짬뽕` 1.0 봉지 src=db/음식_010030 (짬뽕) grams=100.0 kcal=58.0 → S1_parse=pass:literal; S2_match=fail:db:음식_010030
  - total 58.0 vs [486.4, 720.0] (out)
### dv1_037 [dev] `S1_parse` — 라볶이 2인분 시켜서 친구랑 반띵
- branch `llm`; parsed: 라볶이(2.0 인분)
  - rabokki: `라볶이` 2.0 인분 src=db/음식_015305 (라볶이) grams=400.0 kcal=536.0 → S1_parse=fail:amount_mismatch(expected 1-1 인분, got 2.0 인분) [S1 cause: llm] [reconcile changed llm 100.0 g]
  - total 536.0 vs [242.5, 594.6] (ok)
### dv1_038 [dev] `S3_grams` — 진라면 매운맛 2개 끓여서 혼자 먹음
- branch `llm`; parsed: 진라면 매운맛(2.0 개)
  - ramen: `진라면 매운맛` 2.0 개 src=db/음식_015330 (라면) grams=200.0 kcal=164.0 → S1_parse=pass:literal; S2_match=pass:db:음식_015330; S3_grams=fail:200g not in [1000.0, 1200.0]
  - total 164.0 vs [733.0, 1641.8] (out)
### dv1_047 [dev] `S2_match` — 육개장 사발면 하나
- branch `llm`; parsed: 육개장 사발면(1.0 개)
  - cup_ramen: `육개장 사발면` 1.0 개 src=db/음식_010251 (육개장) grams=600.0 kcal=180.0 → S1_parse=pass:literal; S2_match=fail:db:음식_010251
  - total 180.0 vs [211.0, 648.5] (out)
### dv1_048 [test] `S2_match` — 마라탕 1인분에 꿔바로우 조금
- branch `llm`; parsed: 마라탕(1.0 인분), 꿔바로우(None )
  - malatang: `마라탕` 1.0 인분 src=db/음식_015212 (마라탕) grams=600.0 kcal=396.0 → S1_parse=pass:literal; S2_match=pass:db:음식_015212; S3_grams=pass:600g; S4_kcal=pass:396.0kcal vs [292.0, 779.4]
  - tangsuyuk: `꿔바로우` None  src=llm grams=None kcal=146.0 → S1_parse=pass:no_literal_amount; S2_match=fail:llm_estimate(db below_threshold; sanity kept)
  - total 542.0 vs [434.6, 1145.2] (ok)
### dv1_058 [dev] `S3_grams` — 점심 제육볶음 정식
- branch `llm`; parsed: 제육볶음 정식(None )
  - jeyuk: `제육볶음 정식` None  src=db/음식_010082 (제육볶음) grams=100.0 kcal=142.0 → S1_parse=pass:no_literal_amount; S2_match=pass:db:음식_010082; S3_grams=fail:100g not in [200, 300]
  - rice (opt): missing → S1_parse=skip:optional_absent
  - total 142.0 vs [198.4, 898.6] (out)
### dv1_064 [dev] `S2_match` — 치킨너겟 6조각
- branch `llm`; parsed: 치킨너겟(6.0 조각)
  - nugget: `치킨너겟` 6.0 조각 src=db/음식_016946 (닭튀김 New치킨너겟) grams=600.0 kcal=1422.0 → S1_parse=pass:literal; S2_match=fail:db:음식_016946
  - total 1422.0 vs [281.8, 443.2] (out)
### dv1_066 [test] `S2_match` — 닭발 1인분에 주먹밥 하나
- branch `llm`; parsed: 닭발(1.0 인분), 주먹밥(1.0 개)
  - dakbal: `닭발` 1.0 인분 src=db/원재_001349 (닭 부산물 닭발 육수) grams=200.0 kcal=8.0 → S1_parse=pass:literal; S2_match=fail:db:원재_001349
  - jumeokbap: `주먹밥` 1.0 개 src=db/음식_010056 (주먹밥) grams=210.0 kcal=283.5 → S1_parse=pass:literal; S2_match=pass:db:음식_010056; S3_grams=fail:210g not in [100.0, 150.0]
  - total 291.5 vs [442.7, 1000.0] (out)
### dv1_074 [dev] `S2_match` — 새우 10마리 구워먹음
- branch `llm`; parsed: 새우(10.0 마리)
  - shrimp: `새우` 10.0 마리 src=db/음식_013429 (새우 탕수) grams=1000.0 kcal=1980.0 → S1_parse=pass:literal; S2_match=fail:db:음식_013429
  - total 1980.0 vs [190.8, 403.8] (out)
### dv1_076 [test] `S1_parse` — 광어회 한 접시 둘이서
- branch `llm`; parsed: 광어회(1.0 접시)
  - flatfish_sashimi: `광어회` 1.0 접시 src=llm grams=None kcal=0 → S1_parse=fail:amount_mismatch(expected 0.5-0.5 접시, got 1.0 접시) [S1 cause: llm] [reconcile changed llm None 접시]
  - total 0.0 vs [109.8, 269.0] (out)
### dv1_078 [dev] `S2_match` — 연어회 150g이랑 맥주 한 캔
- branch `llm`; parsed: 연어회(150.0 g), 맥주(1.0 캔)
  - salmon: `연어회` 150.0 g src=llm grams=None kcal=258.0 → S1_parse=pass:literal; S2_match=fail:llm_estimate(db empty_macros; sanity kept)
  - beer: `맥주` 1.0 캔 src=curated grams=355.0 kcal=152.7 → S1_parse=pass:literal; S2_match=pass:curated; S3_grams=pass:355g; S4_kcal=pass:152.7kcal vs [122.8, 263.5]
  - total 410.7 vs [276.6, 662.3] (ok)
### dv1_080 [dev] `S2_match` — 계란 프라이 하나에 토스트 한 장
- branch `llm`; parsed: 계란 프라이(1.0 개), 토스트(1.0 장)
  - fried_egg: `계란 프라이` 1.0 개 src=db/음식_017018 (달걀후라이) grams=100.0 kcal=181.0 → S1_parse=pass:literal; S2_match=pass:db:음식_017018; S3_grams=fail:100g not in [50.0, 65.0]
  - bread: `토스트` 1.0 장 src=db/음식_007100 (토스트 허니버터토스트) grams=20.0 kcal=67.0 → S1_parse=pass:literal; S2_match=fail:db:음식_007100
  - total 248.0 vs [122.5, 289.4] (ok)
### dv1_082 [dev] `S2_match` — 저지방 우유 200ml
- branch `llm`; parsed: 저지방 우유(200.0 ml)
  - lowfat_milk: `저지방 우유` 200.0 ml src=db/원재_001728 (우유) grams=200.0 kcal=134.0 → S1_parse=pass:literal; S2_match=fail:db:원재_001728
  - total 134.0 vs [62.0, 112.7] (out)
### dv1_083 [test] `S2_match` — 그릭요거트 100g에 블루베리 한 줌
- branch `llm`; parsed: 그릭요거트(100.0 g), 블루베리(1.0 줌)
  - greek_yogurt: `그릭요거트` 100.0 g src=db/음식_011542 (아이스크림 그릭요거트) grams=100.0 kcal=135.0 → S1_parse=pass:literal; S2_match=fail:db:음식_011542
  - blueberry: `블루베리` 1.0 줌 src=db/원재_001198 (블루베리 생것) grams=40.0 kcal=19.2 → S1_parse=pass:literal; S2_match=pass:db:원재_001198; S3_grams=pass:40g; S4_kcal=pass:19.2kcal vs [6.1, 36.7] [reconcile changed llm None None]
  - total 154.2 vs [54.0, 198.8] (ok)
### dv1_084 [dev] `S2_match` — 떠먹는 요거트 하나
- branch `llm`; parsed: 요거트(1.0 개)
  - yogurt: `요거트` 1.0 개 src=db/음식_011468 (아이스크림 마시는 요거트) grams=85.0 kcal=55.2 → S1_parse=pass:literal; S2_match=fail:db:음식_011468
  - total 55.2 vs [45.4, 137.0] (ok)
### dv1_085 [dev] `S2_match` — 슬라이스 치즈 2장
- branch `llm`; parsed: 슬라이스 치즈(2.0 장)
  - cheese_slice: `슬라이스 치즈` 2.0 장 src=db/음식_007534 (크로와상 슬라이스햄치즈크로) grams=40.0 kcal=156.4 → S1_parse=pass:literal; S2_match=fail:db:음식_007534
  - total 156.4 vs [89.0, 159.0] (ok)
### dv1_086 [test] `S2_match` — 두유 한 팩
- branch `llm`; parsed: 두유(1.0 팩)
  - soymilk: `두유` 1.0 팩 src=db/음식_016255 (두유 카페라떼) grams=200.0 kcal=60.0 → S1_parse=pass:literal; S2_match=fail:db:음식_016255
  - total 60.0 vs [89.0, 165.9] (out)
### dv1_092 [test] `S2_match` — 그릭요거트200g+그래놀라30g
- branch `llm`; parsed: 그릭요거트(200.0 g), 그래놀라(30.0 g)
  - greek_yogurt: `그릭요거트` 200.0 g src=db/음식_011542 (아이스크림 그릭요거트) grams=200.0 kcal=270.0 → S1_parse=pass:literal; S2_match=fail:db:음식_011542
  - granola: `그래놀라` 30.0 g src=db/음식_010361 (요구르트(호상) 오가닉 프로틴 그릭 요거트 & 그래놀라) grams=30.0 kcal=61.5 → S1_parse=pass:literal; S2_match=fail:db:음식_010361
  - total 331.5 vs [206.9, 485.8] (ok)
### dv1_093 [dev] `S2_match` — I had 2 eggs and toast for breakfast
- branch `llm`; parsed: 계란(2.0 개), 토스트(None )
  - egg: `계란` 2.0 개 src=curated grams=100.0 kcal=138.7 → S1_parse=pass:literal; S2_match=pass:curated; S3_grams=pass:100g; S4_kcal=pass:138.7kcal vs [117.4, 203.0]
  - bread: `토스트` None  src=db/음식_007100 (토스트 허니버터토스트) grams=100.0 kcal=335.0 → S1_parse=pass:no_literal_amount; S2_match=fail:db:음식_007100
  - total 473.7 vs [183.7, 338.7] (out)
### dv1_096 [test] `S2_match` — 귤 5개 까먹음
- branch `llm`; parsed: 귤(5.0 개)
  - tangerine: `귤` 5.0 개 src=db/원재_001132 (귤 한라봉 생것) grams=400.0 kcal=200.0 → S1_parse=pass:literal; S2_match=fail:db:원재_001132
  - total 200.0 vs [117.9, 241.5] (ok)
### dv1_098 [test] `S3_grams` — 포도 한 송이
- branch `llm`; parsed: 포도(1.0 송이)
  - grape: `포도` 1.0 송이 src=db/원재_001257 (포도 거봉 생것) grams=100.0 kcal=60.0 → S1_parse=pass:literal; S2_match=pass:db:원재_001257; S3_grams=fail:100g not in [300.0, 500.0] [reconcile changed llm None None]
  - total 60.0 vs [135.4, 417.5] (out)
### dv1_099 [test] `S3_grams` — 수박 두 조각
- branch `llm`; parsed: 수박(2.0 조각)
  - watermelon: `수박` 2.0 조각 src=db/원재_001214 (수박 적육질 생것) grams=200.0 kcal=62.0 → S1_parse=pass:literal; S2_match=pass:db:원재_001214; S3_grams=fail:200g not in [400.0, 800.0] [reconcile changed llm None 두 조각]
  - total 62.0 vs [106.6, 321.8] (out)
### dv1_102 [dev] `S3_grams` — 배 반쪽
- branch `llm`; parsed: 배(0.5 쪽)
  - pear: `배` 0.5 쪽 src=db/원재_001185 (배 생것) grams=5.0 kcal=2.2 → S1_parse=pass:literal; S2_match=pass:db:원재_001185; S3_grams=fail:5g not in [175.0, 250.0] [reconcile changed llm 0.5 개]
  - total 2.2 vs [65.9, 128.8] (out)
### dv1_104 [test] `S2_match` — 샤인머스캣 20알
- branch `llm`; parsed: 샤인머스캣(20.0 알)
  - grape: `샤인머스캣` 20.0 알 src=db/음식_010813 (에이드 샤인머스캣 에이드) grams=200.0 kcal=92.0 → S1_parse=pass:literal; S2_match=fail:db:음식_010813 [reconcile changed llm None 20알]
  - total 92.0 vs [69.9, 203.0] (ok)
### dv1_105 [dev] `S2_match` — 아보카도 반 개 토스트에 올려먹음
- branch `llm`; parsed: 아보카도(0.5 개), 토스트(None )
  - avocado: `아보카도` 0.5 개 src=db/원재_001218 (아보카도 생것) grams=75.0 kcal=120.0 → S1_parse=pass:literal; S2_match=pass:db:원재_001218; S3_grams=pass:75g; S4_kcal=pass:120.0kcal vs [88.6, 154.6]
  - bread: `토스트` None  src=db/음식_007100 (토스트 허니버터토스트) grams=100.0 kcal=335.0 → S1_parse=pass:no_literal_amount; S2_match=fail:db:음식_007100
  - total 455.0 vs [154.9, 290.3] (out)
### dv1_106 [dev] `S2_match` — 사과 1개 바나나 1개 귤 2개
- branch `llm`; parsed: 사과(1.0 개), 바나나(1.0 개), 귤(2.0 개)
  - apple: `사과` 1.0 개 src=db/원재_001202 (사과 생것) grams=200.0 kcal=114.0 → S1_parse=pass:literal; S2_match=pass:db:원재_001202; S3_grams=pass:200g; S4_kcal=pass:114.0kcal vs [77.6, 193.1]
  - banana: `바나나` 1.0 개 src=db/원재_001183 (바나나 생것) grams=120.0 kcal=92.4 → S1_parse=pass:literal; S2_match=pass:db:원재_001183; S3_grams=pass:120g; S4_kcal=pass:92.4kcal vs [64.3, 132.1]
  - tangerine: `귤` 2.0 개 src=db/원재_001132 (귤 한라봉 생것) grams=160.0 kcal=80.0 → S1_parse=pass:literal; S2_match=fail:db:원재_001132
  - total 286.4 vs [186.0, 424.8] (ok)
### dv1_108 [dev] `S2_match` — 닭가슴살 샐러드 한 팩
- branch `llm`; parsed: 닭가슴살 샐러드(1.0 팩)
  - chicken_salad: `닭가슴살 샐러드` 1.0 팩 src=db/음식_017012 (닭가슴살 샐러드 마요네즈) grams=200.0 kcal=108.0 → S1_parse=pass:literal; S2_match=fail:db:음식_017012
  - total 108.0 vs [166.0, 450.5] (out)
### dv1_109 [dev] `S2_match` — 점심은 샐러드로 간단히
- branch `llm`; parsed: 샐러드(None )
  - salad: `샐러드` None  src=db/음식_013168 (샐러드 샐러드) grams=250.0 kcal=37.5 → S1_parse=pass:no_literal_amount; S2_match=fail:db:음식_013168
  - total 37.5 vs [165.1, 483.5] (out)
### dv1_119 [test] `S2_match` — 피자 먹고 싶었는데 그냥 샐러드
- branch `llm`; parsed: 샐러드(None )
  - salad: `샐러드` None  src=db/음식_013168 (샐러드 샐러드) grams=250.0 kcal=37.5 → S1_parse=pass:no_literal_amount; S2_match=fail:db:음식_013168
  - total 37.5 vs [165.1, 483.5] (out)
### dv1_121 [test] `S2_match` — lunch: chicken salad, cola zero
- branch `llm`; parsed: chicken salad(None ), cola zero(None )
  - chicken_salad: `chicken salad` None  src=llm grams=None kcal=252.0 → S1_parse=pass:no_literal_amount; S2_match=fail:llm_estimate(db empty_macros; sanity kept)
  - zero_cola: `cola zero` None  src=llm grams=None kcal=0 → S1_parse=pass:no_literal_amount; S2_match=pass:llm_estimate(db below_threshold; sanity kept); S3_grams=n/a:llm_estimate; S4_kcal=pass:0kcal vs [0.0, 10.5] (llm macros, sanity kept)
  - total 252.0 vs [166.0, 461.0] (ok)
### dv1_122 [test] `S4_kcal` — 식빵 두 장에 땅콩버터
- branch `llm`; parsed: 식빵(2.0 장), 땅콩버터(None )
  - bread: `식빵` 2.0 장 src=db/음식_011716 (식빵) grams=70.0 kcal=184.8 → S1_parse=pass:literal; S2_match=pass:db:음식_011716; S3_grams=pass:70g; S4_kcal=pass:184.8kcal vs [137.6, 266.4]
  - peanut_butter: `땅콩버터` None  src=llm grams=None kcal=444.0 → S1_parse=pass:no_literal_amount; S2_match=pass:llm_estimate(db empty_macros; sanity kept); S3_grams=n/a:llm_estimate; S4_kcal=fail:444.0kcal vs [73.3, 223.2] (llm macros, sanity kept)
  - total 628.8 vs [210.9, 489.6] (out)
### dv1_123 [test] `S2_match` — 베이글 하나 크림치즈 발라서
- branch `llm`; parsed: 베이글(1.0 개), 크림치즈(None )
  - bagel: `베이글` 1.0 개 src=db/음식_013959 (베이글) grams=100.0 kcal=306.0 → S1_parse=pass:literal; S2_match=pass:db:음식_013959; S3_grams=pass:100g; S4_kcal=pass:306.0kcal vs [208.8, 408.9]
  - cream_cheese: `크림치즈` None  src=db/음식_013885 (베이글 크림치즈 베이글) grams=100.0 kcal=294.0 → S1_parse=pass:no_literal_amount; S2_match=fail:db:음식_013885
  - total 600.0 vs [248.8, 548.7] (out)
### dv1_125 [test] `S2_match` — 도넛 두 개 먹음 ㅠㅠ
- branch `llm`; parsed: 도넛(2.0 개)
  - donut: `도넛` 2.0 개 src=db/음식_016607 (도넛 생도넛) grams=200.0 kcal=946.0 → S1_parse=pass:literal; S2_match=fail:db:음식_016607
  - total 946.0 vs [257.8, 911.4] (out)
### dv1_127 [dev] `S3_grams` — 치즈케이크 1조각이랑 아아
- branch `llm`; parsed: 치즈케이크(1.0 조각), 아아(1.0 잔)
  - cheesecake: `치즈케이크` 1.0 조각 src=db/음식_007788 (케이크 치즈케이크) grams=100.0 kcal=329.0 → S1_parse=pass:literal; S2_match=pass:db:음식_007788; S3_grams=pass:100g; S4_kcal=pass:329.0kcal vs [231.9, 543.6]
  - americano: `아아` 1.0 잔 src=db/음식_009005 (커피 아메리카노) grams=250.0 kcal=0 → S1_parse=pass:no_literal_amount; S2_match=pass:db:음식_009005; S3_grams=fail:250g not in [300, 500]
  - total 329.0 vs [231.9, 570.6] (ok)
### dv1_128 [test] `S4_kcal` — 허니버터칩 한 봉지 다 먹었다
- branch `llm`; parsed: 허니버터칩(1.0 봉지)
  - honeybutter_chip: `허니버터칩` 1.0 봉지 src=llm grams=None kcal=190.0 → S1_parse=pass:literal; S2_match=pass:llm_estimate(db empty_macros; sanity kept); S3_grams=n/a:llm_estimate; S4_kcal=fail:190.0kcal vs [286.6, 387.8] (llm macros, sanity kept)
  - total 190.0 vs [286.6, 387.8] (out)
### dv1_135 [dev] `S2_match` — 빼빼로 한 갑
- branch `llm`; parsed: 빼빼로(1.0 갑)
  - pepero: `빼빼로` 1.0 갑 src=none grams=None kcal=None → S1_parse=pass:literal; S2_match=fail:no_macros(db no_candidates_after_rerank; sanity -)
  - total 0.0 vs [223.4, 313.9] (out)
### dv1_137 [dev] `S4_kcal` — 초코파이 2개
- branch `llm`; parsed: 초코파이(2.0 개)
  - chocopie: `초코파이` 2.0 개 src=llm grams=None kcal=695.0 → S1_parse=pass:literal; S2_match=pass:llm_estimate(db empty_macros; sanity kept); S3_grams=n/a:llm_estimate; S4_kcal=fail:695.0kcal vs [289.8, 382.5] (llm macros, sanity kept)
  - total 695.0 vs [289.8, 382.5] (out)
### dv1_138 [test] `S1_parse` — 떡볶이 배달시켜서 2인분 중에 1인분 먹음
- branch `llm`; parsed: 떡볶이(500.0 g)
  - tteokbokki: `떡볶이` 500.0 g src=db/음식_016227 (떡볶이) grams=500.0 kcal=720.0 → S1_parse=fail:amount_mismatch(expected 1-1 그릇/대접/인분/접시, got 500.0 g) [S1 cause: llm]
  - total 720.0 vs [319.0, 1140.2] (ok)
### dv1_140 [test] `S4_kcal` — 프링글스 오리지널 반 통
- branch `llm`; parsed: 프링글스 오리지널(0.5 통)
  - pringles: `프링글스 오리지널` 0.5 통 src=llm grams=None kcal=0 → S1_parse=pass:literal; S2_match=pass:llm_estimate(db empty_macros; sanity kept); S3_grams=n/a:llm_estimate; S4_kcal=fail:0kcal vs [242.5, 331.7] (llm macros, sanity kept) [reconcile changed llm None 반 통]
  - total 0.0 vs [242.5, 331.7] (out)
### dv1_141 [test] `S3_grams` — 아이스 아메리카노 벤티
- branch `llm`; parsed: 아이스 아메리카노(None )
  - americano: `아이스 아메리카노` None  src=db/음식_009004 (커피 아메리카노) grams=355.0 kcal=14.2 → S1_parse=pass:no_literal_amount; S2_match=pass:db:음식_009004; S3_grams=fail:355g not in [580, 600]
  - total 14.2 vs [0.0, 31.4] (ok)
### dv1_142 [dev] `S3_grams` — 라떼 한 잔 마셨어
- branch `llm`; parsed: 라떼(1.0 잔)
  - latte: `라떼` 1.0 잔 src=db/음식_009731 (카페라떼) grams=250.0 kcal=95.0 → S1_parse=pass:literal; S2_match=pass:db:음식_009731; S3_grams=fail:250g not in [300.0, 500.0]
  - total 95.0 vs [97.6, 318.5] (out)
### dv1_145 [dev] `S2_match` — 오렌지주스 한 잔
- branch `llm`; parsed: 오렌지주스(1.0 잔)
  - oj: `오렌지주스` 1.0 잔 src=llm grams=None kcal=82.0 → S1_parse=pass:literal; S2_match=fail:llm_estimate(db empty_macros; sanity kept)
  - total 82.0 vs [52.6, 128.8] (ok)
### dv1_146 [test] `S1_parse` — 포카리 한 병 500ml
- branch `llm`; parsed: 포카리(1.0 병)
  - ion_drink: `포카리` 1.0 병 src=llm grams=None kcal=0 → S1_parse=fail:amount_mismatch(expected 500-500 ml/병, got 1.0 병) [S1 cause: llm]
  - total 0.0 vs [92.0, 162.1] (out)
### dv1_148 [test] `S2_match` — 버블티 라지 사이즈
- branch `llm`; parsed: 버블티(None 사이즈)
  - bubble_tea: `버블티` None 사이즈 src=db/음식_014683 (밀크티/버블티 밀크티) grams=100.0 kcal=52.0 → S1_parse=pass:no_literal_amount; S2_match=fail:db:음식_014683 [reconcile changed llm None 라지 사이즈]
  - total 52.0 vs [319.0, 747.5] (out)
### dv1_149 [test] `S1_parse` — 물 2리터 마셨다
- branch `llm_empty->deterministic+enrich`; parsed: 물(None )
  - water (opt): `물` None  src=db/음식_014823 (물회) grams=100.0 kcal=76.0 → S1_parse=fail:amount_missing [S1 cause: deterministic]
  - total 76.0 vs [0, 5.0] (out)
### dv1_150 [dev] `S3_grams` — 바닐라라떼 그란데
- branch `llm`; parsed: 바닐라라떼(None )
  - vanilla_latte: `바닐라라떼` None  src=db/음식_015799 (라떼 바닐라라떼 핫(HOT)) grams=355.0 kcal=156.2 → S1_parse=pass:no_literal_amount; S2_match=pass:db:음식_015799; S3_grams=fail:355g not in [460, 480]
  - total 156.2 vs [152.3, 274.3] (ok)
### dv1_151 [test] `S1_parse` — 아메리카노 2샷 추가
- branch `llm_empty->deterministic+enrich`; parsed: 
  - americano: missing → S1_parse=fail:missing [S1 cause: deterministic]
  - total 0 vs [0.0, 27.0] (ok)
### dv1_158 [dev] `S2_match` — 하이볼 한 잔이랑 감자튀김
- branch `llm`; parsed: 하이볼(1.0 잔), 감자튀김(None )
  - highball: `하이볼` 1.0 잔 src=db/음식_017516 (기타음료 논알콜 하이볼) grams=250.0 kcal=95.0 → S1_parse=pass:literal; S2_match=fail:db:음식_017516
  - fries: `감자튀김` None  src=db/음식_017952 (감자튀김) grams=120.0 kcal=374.4 → S1_parse=pass:no_literal_amount; S2_match=pass:db:음식_017952; S3_grams=pass:120g; S4_kcal=pass:374.4kcal vs [275.8, 416.8]
  - total 469.4 vs [365.3, 707.8] (ok)
### dv1_163 [test] `S2_match` — 빅맥 세트 먹음
- branch `llm`; parsed: 빅맥 세트(1.0 )
  - bigmac: `빅맥 세트` 1.0  src=none grams=None kcal=None → S1_parse=pass:no_literal_amount; S2_match=fail:no_macros(db empty_macros; sanity dropped)
  - fries (opt): missing → S1_parse=skip:optional_absent
  - cola (opt): missing → S1_parse=skip:optional_absent
  - total 0.0 vs [488.3, 1334.6] (out)
### dv1_164 [dev] `S3_grams` — 와퍼 하나
- branch `llm`; parsed: 와퍼(1.0 개)
  - whopper: `와퍼` 1.0 개 src=db/음식_002204 (햄버거 와퍼) grams=100.0 kcal=247.0 → S1_parse=pass:literal; S2_match=pass:db:음식_002204; S3_grams=fail:100g not in [250.0, 290.0]
  - total 247.0 vs [496.8, 792.9] (out)
### dv1_167 [test] `S3_grams` — 싸이버거 단품
- branch `llm`; parsed: 싸이버거(1.0 개)
  - thighburger: `싸이버거` 1.0 개 src=db/음식_014169 (버거 싸이버거) grams=100.0 kcal=258.0 → S1_parse=pass:no_literal_amount; S2_match=pass:db:음식_014169; S3_grams=fail:100g not in [220, 260]
  - total 258.0 vs [505.8, 742.9] (out)
### dv1_168 [test] `S3_grams` — 감자튀김 라지
- branch `llm`; parsed: 감자튀김(None 라지)
  - fries: `감자튀김` None 라지 src=db/음식_017952 (감자튀김) grams=120.0 kcal=374.4 → S1_parse=pass:no_literal_amount; S2_match=pass:db:음식_017952; S3_grams=fail:120g not in [140, 180]
  - total 374.4 vs [388.1, 622.8] (out)
### dv1_170 [test] `S3_grams` — 불고기버거 2개
- branch `llm`; parsed: 불고기버거(2.0 개)
  - bulgogi_burger: `불고기버거` 2.0 개 src=db/음식_002238 (햄버거 불고기버거) grams=200.0 kcal=496.0 → S1_parse=pass:literal; S2_match=pass:db:음식_002238; S3_grams=fail:200g not in [300.0, 360.0]
  - total 496.0 vs [661.9, 987.1] (out)
### dv1_172 [test] `S2_match` — 서브웨이 에그마요 30cm
- branch `llm`; parsed: 서브웨이 에그마요(30.0 cm)
  - egg_mayo_sub: `서브웨이 에그마요` 30.0 cm src=llm grams=None kcal=232.0 → S1_parse=pass:literal; S2_match=fail:llm_estimate(db empty_macros; sanity kept) [reconcile changed llm None 30cm]
  - total 232.0 vs [920.2, 1362.0] (out)
### dv1_176 [test] `S4_kcal` — 바나나맛우유 하나
- branch `llm`; parsed: 바나나맛우유(1.0 개)
  - banana_milk: `바나나맛우유` 1.0 개 src=llm grams=None kcal=130.0 → S1_parse=pass:literal; S2_match=pass:llm_estimate(db empty_macros; sanity kept); S3_grams=n/a:llm_estimate; S4_kcal=fail:130.0kcal vs [174.3, 237.3] (llm macros, sanity kept)
  - total 130.0 vs [174.3, 237.3] (out)
### dv1_177 [test] `S4_kcal` — 편의점 도시락 하나
- branch `llm`; parsed: 도시락(1.0 개)
  - dosirak: `도시락` 1.0 개 src=llm grams=None kcal=392.0 → S1_parse=pass:literal; S2_match=pass:llm_estimate(db empty_macros; sanity kept); S3_grams=n/a:llm_estimate; S4_kcal=fail:392.0kcal vs [535.0, 1105.0] (llm macros, sanity kept)
  - total 392.0 vs [535.0, 1105.0] (out)
### dv1_178 [test] `S2_match` — 삼각김밥 참치마요 2개랑 바나나우유
- branch `llm`; parsed: 삼각김밥 참치마요(2.0 개), 바나나우유(1.0 )
  - tri_tunamayo: `삼각김밥 참치마요` 2.0 개 src=db/음식_013457 (삼각김밥) grams=200.0 kcal=368.0 → S1_parse=pass:literal; S2_match=pass:db:음식_013457; S3_grams=pass:200g; S4_kcal=pass:368.0kcal vs [326.2, 496.0]
  - banana_milk: `바나나우유` 1.0  src=db/원재_001183 (바나나 생것) grams=200.0 kcal=154.0 → S1_parse=pass:no_literal_amount; S2_match=fail:db:원재_001183
  - total 522.0 vs [500.5, 733.3] (ok)
### dv1_181 [dev] `S2_match` — GS25 닭가슴살 스테이크 하나
- branch `llm`; parsed: 닭가슴살 스테이크(1.0 개)
  - chicken_breast: `닭가슴살 스테이크` 1.0 개 src=llm grams=None kcal=156.0 → S1_parse=pass:literal; S2_match=fail:llm_estimate(db empty_macros; sanity kept)
  - total 156.0 vs [90.4, 239.5] (ok)
### dv1_182 [dev] `S4_kcal` — 핫식스 한 캔
- branch `llm`; parsed: 핫식스(1.0 캔)
  - energy_drink: `핫식스` 1.0 캔 src=llm grams=None kcal=200.0 → S1_parse=pass:literal; S2_match=pass:llm_estimate(db empty_macros; sanity kept); S3_grams=n/a:llm_estimate; S4_kcal=fail:200.0kcal vs [85.0, 192.4] (llm macros, sanity kept)
  - total 200.0 vs [85.0, 192.4] (out)
### dv1_189 [test] `S4_kcal` — 게이너 1스쿱 100g
- branch `llm`; parsed: 게이너(1.0 스쿱)
  - gainer: `게이너` 1.0 스쿱 src=llm grams=None kcal=103.5 → S1_parse=pass:literal; S2_match=pass:llm_estimate(db below_threshold; sanity rescaled); S3_grams=n/a:llm_estimate; S4_kcal=fail:103.5kcal vs [319.0, 456.0] (llm macros, sanity rescaled)
  - total 103.5 vs [319.0, 456.0] (out)
### dv1_190 [test] `S2_match` — 프로틴 쉐이크 300ml
- branch `llm`; parsed: 프로틴 쉐이크(300.0 ml)
  - protein_shake: `프로틴 쉐이크` 300.0 ml src=db/음식_012107 (스무디 초코프로틴쉐이크) grams=300.0 kcal=237.0 → S1_parse=pass:literal; S2_match=fail:db:음식_012107
  - total 237.0 vs [100.8, 240.6] (ok)
### dv1_194 [test] `S3_grams` — 삼겹살 먹고 볶음밥까지 야무지게 ㅎ
- branch `llm`; parsed: 삼겹살(None ), 볶음밥(None )
  - samgyeopsal: `삼겹살` None  src=db/음식_013453 (삼겹살구이) grams=200.0 kcal=934.0 → S1_parse=pass:no_literal_amount; S2_match=pass:db:음식_013453; S3_grams=pass:200g; S4_kcal=pass:934.0kcal vs [497.2, 1032.4]
  - fried_rice: `볶음밥` None  src=db/음식_013843 (볶음밥) grams=350.0 kcal=658.0 → S1_parse=pass:no_literal_amount; S2_match=pass:db:음식_013843; S3_grams=fail:350g not in [150, 300]
  - total 1592.0 vs [616.4, 1779.9] (ok)
### dv1_196 [dev] `S2_match` — 저녁 8시쯤 김치볶음밥 먹었는데 너무 짰음
- branch `llm`; parsed: 김치볶음밥(None )
  - kimchi_fried_rice: `김치볶음밥` None  src=db/음식_017266 (김치 볶음밥 모듬) grams=350.0 kcal=455.0 → S1_parse=pass:no_literal_amount; S2_match=fail:db:음식_017266
  - total 455.0 vs [410.8, 831.7] (ok)
### dv1_197 [dev] `S2_match` — 오늘 다이어트 망함.. 떡볶이 2인분이랑 튀김 먹음
- branch `llm`; parsed: 떡볶이(2.0 인분), 튀김(1.0 식)
  - tteokbokki: `떡볶이` 2.0 인분 src=db/음식_016227 (떡볶이) grams=600.0 kcal=864.0 → S1_parse=pass:literal; S2_match=pass:db:음식_016227; S3_grams=pass:600g; S4_kcal=pass:864.0kcal vs [643.0, 2275.4]
  - squid_fritter: `튀김` 1.0 식 src=db/음식_016949 (닭튀김) grams=100.0 kcal=301.0 → S1_parse=pass:no_literal_amount; S2_match=fail:db:음식_016949
  - total 1165.0 vs [747.8, 2661.0] (ok)
### dv1_205 [test] `S2_match` — 귤 서너 개
- branch `llm`; parsed: 귤(3.0 개)
  - tangerine: `귤` 3.0 개 src=db/원재_001132 (귤 한라봉 생것) grams=240.0 kcal=120.0 → S1_parse=pass:literal; S2_match=fail:db:원재_001132
  - total 120.0 vs [68.7, 194.2] (ok)
### dv1_206 [dev] `S3_grams` — 치킨 한 마리를 셋이서 나눠먹음
- branch `llm`; parsed: 치킨(1.0 마리)
  - fried_chicken: `치킨` 1.0 마리 src=curated grams=1000.0 kcal=2470.0 → S1_parse=pass:no_literal_amount; S2_match=pass:curated; S3_grams=fail:1000g not in [270, 330] [reconcile changed llm None 마리]
  - total 2470.0 vs [595.2, 1097.6] (out)
### dv1_209 [dev] `S1_parse` — Big Mac 1개랑 coke 1캔
- branch `llm`; parsed: Big Mac(1.0 개), coke(1.0 캔)
- case S1: ['extra_items:llm']; extra: [{'food_name': 'Big Mac', 'amount': 1.0, 'unit': '개', 'effective_kcal': 464.0}]
  - bigmac: missing → S1_parse=fail:missing [S1 cause: llm]
  - cola: `coke` 1.0 캔 src=llm grams=None kcal=156.0 → S1_parse=pass:literal; S2_match=pass:llm_estimate(db empty_macros; sanity kept); S3_grams=n/a:llm_estimate; S4_kcal=pass:156.0kcal vs [85.0, 180.7] (llm macros, sanity kept)
  - total 620.0 vs [573.3, 846.0] (ok)
### dv1_210 [test] `S2_match` — 아침 7:30 그릭요거트 150g
- branch `llm`; parsed: 그릭요거트(150.0 g)
  - greek_yogurt: `그릭요거트` 150.0 g src=db/음식_011542 (아이스크림 그릭요거트) grams=150.0 kcal=202.5 → S1_parse=pass:literal; S2_match=fail:db:음식_011542
  - total 202.5 vs [74.4, 240.6] (ok)
### dv1_211 [test] `S1_parse` — 밥 한공기 김치찌개 계란말이
- branch `llm_empty->deterministic+enrich`; parsed: 밥(1.0 공기), 김치찌개 계란말이(None )
  - rice: `밥` 1.0 공기 src=curated grams=210.0 kcal=265.8 → S1_parse=pass:literal; S2_match=pass:curated; S3_grams=pass:210g; S4_kcal=pass:265.8kcal vs [200.2, 425.0]
  - kimchi_jjigae: `김치찌개 계란말이` None  src=db/음식_017233 (김치찌개) grams=100.0 kcal=61.0 → S1_parse=fail:merged_items(name '김치찌개 계란말이' also contains '계란말이') [S1 cause: deterministic]
  - egg_roll: missing → S1_parse=fail:missing [S1 cause: deterministic]
  - total 326.8 vs [453.0, 1089.0] (out)
### dv1_212 [dev] `S2_match` — 점심 순대국밥 한그릇 뚝딱 😋
- branch `llm`; parsed: 순대국밥(1.0 그릇)
  - sundaeguk: `순대국밥` 1.0 그릇 src=db/음식_012919 (순대) grams=700.0 kcal=1288.0 → S1_parse=pass:literal; S2_match=fail:db:음식_012919
  - total 1288.0 vs [205.6, 673.8] (out)
### dv1_215 [dev] `S3_grams` — 햇반 1개랑 닭가슴살 1개
- branch `llm`; parsed: 햇반(1.0 개), 닭가슴살(1.0 개)
  - rice: `햇반` 1.0 개 src=db/음식_011600 (쌀밥) grams=100.0 kcal=166.0 → S1_parse=pass:literal; S2_match=pass:db:음식_011600; S3_grams=fail:100g not in [200.0, 215.0]
  - chicken_breast: `닭가슴살` 1.0 개 src=curated grams=100.0 kcal=156.4 → S1_parse=pass:literal; S2_match=pass:curated; S3_grams=pass:100g; S4_kcal=pass:156.4kcal vs [90.4, 239.5]
  - total 322.4 vs [301.4, 637.1] (ok)
### dv1_216 [dev] `S2_match` — 소고기 미역국 한 그릇이랑 잡곡밥
- branch `llm`; parsed: 소고기 미역국(1.0 그릇), 잡곡밥(1.0 공기)
  - miyeokguk: `소고기 미역국` 1.0 그릇 src=db/음식_013088 (소고기 미역국) grams=350.0 kcal=70.0 → S1_parse=pass:literal; S2_match=fail:db:음식_013088
  - multigrain_rice: `잡곡밥` 1.0 공기 src=curated grams=210.0 kcal=311.5 → S1_parse=pass:no_literal_amount; S2_match=pass:curated; S3_grams=pass:210g; S4_kcal=pass:311.5kcal vs [200.2, 380.2]
  - total 381.5 vs [219.5, 478.7] (ok)
### dv1_217 [dev] `S2_match` — 파스타 반 접시만 먹었어
- branch `llm`; parsed: 파스타(0.5 접시)
  - pasta: `파스타` 0.5 접시 src=db/원재_000198 (파스타 마카로니 말린것) grams=175.0 kcal=665.0 → S1_parse=pass:literal; S2_match=fail:db:원재_000198 [reconcile changed llm None 반 접시]
  - total 665.0 vs [105.7, 381.2] (out)
### dv1_218 [test] `S2_match` — 케이크 한조각이랑 아메리카노 한잔
- branch `llm`; parsed: 케이크(1.0 조각), 아메리카노(1.0 잔)
  - cream_cake: `케이크` 1.0 조각 src=db/음식_007798 (케이크 초콜릿케이크) grams=100.0 kcal=420.0 → S1_parse=pass:literal; S2_match=fail:db:음식_007798
  - americano: `아메리카노` 1.0 잔 src=db/음식_009005 (커피 아메리카노) grams=355.0 kcal=0 → S1_parse=pass:literal; S2_match=pass:db:음식_009005; S3_grams=pass:355g; S4_kcal=pass:0kcal vs [0.0, 27.0]
  - total 420.0 vs [165.6, 344.8] (out)
### dv1_219 [test] `S2_match` — 편의점 바나나우유 하나랑 삼각김밥 하나
- branch `llm`; parsed: 바나나우유(1.0 개), 삼각김밥(1.0 개)
  - banana_milk: `바나나우유` 1.0 개 src=db/원재_001183 (바나나 생것) grams=200.0 kcal=154.0 → S1_parse=pass:literal; S2_match=fail:db:원재_001183
  - tri_gimbap: `삼각김밥` 1.0 개 src=db/음식_013457 (삼각김밥) grams=110.0 kcal=202.4 → S1_parse=pass:literal; S2_match=pass:db:음식_013457; S3_grams=pass:110g; S4_kcal=pass:202.4kcal vs [139.9, 250.5]
  - total 356.4 vs [314.2, 487.8] (ok)
### dv1_220 [test] `S1_parse` — 아침 계란 3개 점심 닭가슴살 샐러드 저녁 고구마 2개
- branch `llm_empty->deterministic+enrich`; parsed: 계란(3.0 개), 닭가슴살 샐러드 고구마(2.0 개)
  - egg: `계란` 3.0 개 src=curated grams=150.0 kcal=207.4 → S1_parse=pass:literal; S2_match=pass:curated; S3_grams=pass:150g; S4_kcal=pass:207.4kcal vs [178.6, 302.0]
  - chicken_salad: `닭가슴살 샐러드 고구마` 2.0 개 src=db/음식_013194 (샐러드 고구마) grams=300.0 kcal=555.0 → S1_parse=fail:merged_items(name '닭가슴살 샐러드 고구마' also contains '고구마') [S1 cause: deterministic]
  - sweet_potato: missing → S1_parse=fail:missing [S1 cause: deterministic]
  - total 762.4 vs [495.7, 1584.7] (ok)
### dv1_221 [dev] `S2_match` — 간식으로 귤 두 개랑 아몬드 조금
- branch `llm`; parsed: 귤(2.0 개), 아몬드(None )
  - tangerine: `귤` 2.0 개 src=db/원재_001132 (귤 한라봉 생것) grams=160.0 kcal=80.0 → S1_parse=pass:literal; S2_match=fail:db:원재_001132
  - almond: `아몬드` None  src=db/원재_000491 (아몬드 볶은것) grams=100.0 kcal=594.0 → S1_parse=pass:no_literal_amount; S2_match=pass:db:원재_000491; S3_grams=fail:100g not in [10, 30]
  - total 674.0 vs [91.4, 300.6] (out)
### dv1_226 [test] `S2_match` — 비빔밥 한 그릇에 된장국
- branch `llm`; parsed: 비빔밥(1.0 그릇), 된장국(1.0 )
  - bibimbap: `비빔밥` 1.0 그릇 src=db/음식_013743 (비빔밥) grams=450.0 kcal=639.0 → S1_parse=pass:literal; S2_match=pass:db:음식_013743; S3_grams=pass:450g; S4_kcal=pass:639.0kcal vs [313.2, 942.8]
  - doenjang_guk: `된장국` 1.0  src=db/음식_016314 (된장국 호박잎) grams=100.0 kcal=7.0 → S1_parse=pass:no_literal_amount; S2_match=fail:db:음식_016314
  - total 646.0 vs [340.6, 1124.9] (ok)
### dv1_228 [dev] `S1_parse` — 저녁 양념치킨 반 마리랑 콜라 1.25L 중 한 컵
- branch `llm`; parsed: 양념치킨(0.5 마리), 콜라(1.0 )
  - yangnyeom: `양념치킨` 0.5 마리 src=db/음식_016839 (닭튀김 양념 치킨) grams=500.0 kcal=1300.0 → S1_parse=pass:literal; S2_match=pass:db:음식_016839; S3_grams=pass:500g; S4_kcal=pass:1300.0kcal vs [931.0, 1523.0]
  - cola: `콜라` 1.0  src=curated grams=100.0 kcal=42.4 → S1_parse=fail:amount_mismatch(expected 1-1 잔/컵, got 1.0 -) [S1 cause: llm] [reconcile changed llm 1.25 L]
  - total 1342.4 vs [998.0, 1676.5] (ok)
### dv1_230 [dev] `S2_match` — 귤 하나 사과 하나 배 하나
- branch `llm`; parsed: 귤(1.0 개), 사과(1.0 개), 배(1.0 개)
  - tangerine: `귤` 1.0 개 src=db/원재_001132 (귤 한라봉 생것) grams=80.0 kcal=40.0 → S1_parse=pass:literal; S2_match=fail:db:원재_001132
  - apple: `사과` 1.0 개 src=db/원재_001202 (사과 생것) grams=200.0 kcal=114.0 → S1_parse=pass:literal; S2_match=pass:db:원재_001202; S3_grams=pass:200g; S4_kcal=pass:114.0kcal vs [77.6, 193.1]
  - pear: `배` 1.0 개 src=db/원재_001185 (배 생것) grams=400.0 kcal=180.0 → S1_parse=pass:literal; S2_match=pass:db:원재_001185; S3_grams=pass:400g; S4_kcal=pass:180.0kcal vs [136.8, 252.5]
  - total 334.0 vs [234.0, 497.9] (ok)
### dv1_234 [dev] `S3_grams` — 돈가스 정식 하나
- branch `llm`; parsed: 돈가스 정식(1.0 개)
  - donkatsu: `돈가스 정식` 1.0 개 src=db/음식_016425 (돈가스) grams=100.0 kcal=280.0 → S1_parse=pass:literal; S2_match=pass:db:음식_016425; S3_grams=fail:100g not in [150.0, 250.0]
  - rice (opt): missing → S1_parse=skip:optional_absent
  - total 280.0 vs [373.0, 1288.0] (out)
### dv1_236 [test] `S2_match` — 양배추 샐러드 한 그릇
- branch `llm`; parsed: 양배추 샐러드(1.0 그릇)
  - salad: `양배추 샐러드` 1.0 그릇 src=db/음식_010942 (양배추 샐러드) grams=250.0 kcal=247.5 → S1_parse=pass:literal; S2_match=fail:db:음식_010942
  - total 247.5 vs [165.1, 483.5] (ok)
### dv1_240 [test] `S3_grams` — 오늘은 칼국수 곱빼기
- branch `llm`; parsed: 칼국수(None )
  - kalguksu: `칼국수` None  src=db/음식_009724 (칼국수) grams=750.0 kcal=577.5 → S1_parse=pass:no_literal_amount; S2_match=pass:db:음식_009724; S3_grams=fail:750g not in [900, 1200]
  - total 577.5 vs [562.0, 1245.8] (ok)
### dv1_241 [dev] `S3_grams` — 후식으로 수박 한 조각
- branch `llm`; parsed: 수박(1.0 조각)
  - watermelon: `수박` 1.0 조각 src=db/원재_001214 (수박 적육질 생것) grams=100.0 kcal=31.0 → S1_parse=pass:literal; S2_match=pass:db:원재_001214; S3_grams=fail:100g not in [200.0, 400.0] [reconcile changed llm None 조각]
  - total 31.0 vs [50.8, 163.4] (out)
### dv1_242 [test] `S3_grams` — 와인 한 병 둘이서 마심
- branch `llm`; parsed: 와인(1.0 병)
  - wine: `와인` 1.0 병 src=curated grams=750.0 kcal=622.5 → S1_parse=pass:no_literal_amount; S2_match=pass:curated; S3_grams=fail:750g not in [350, 400]
  - total 622.5 vs [215.5, 401.0] (out)
### dv1_244 [test] `S3_grams` — 오트밀 한 컵에 우유
- branch `llm`; parsed: 오트밀(1.0 컵), 우유(None )
  - oatmeal: `오트밀` 1.0 컵 src=curated grams=200.0 kcal=789.8 → S1_parse=pass:literal; S2_match=pass:curated; S3_grams=fail:200g not in [40, 90]
  - milk: `우유` None  src=db/원재_001728 (우유) grams=200.0 kcal=134.0 → S1_parse=pass:no_literal_amount; S2_match=pass:db:원재_001728; S3_grams=pass:200g; S4_kcal=pass:134.0kcal vs [115.6, 192.0]
  - total 923.8 vs [235.9, 588.9] (out)
### dv1_247 [test] `S1_parse` — 아메리까노 한잔
- branch `llm`; parsed: 아메리까노(1.0 잔)
- case S1: ['extra_items:llm']; extra: [{'food_name': '아메리까노', 'amount': 1.0, 'unit': '잔', 'effective_kcal': 50.0}]
  - americano: missing → S1_parse=fail:missing [S1 cause: llm]
  - total 50.0 vs [0.0, 27.0] (out)
### dv1_253 [test] `S2_match` — 떡볶이 1인분 순대 1인분 튀김 3개
- branch `llm`; parsed: 떡볶이(1.0 인분), 순대(1.0 인분), 튀김(3.0 개)
  - tteokbokki: `떡볶이` 1.0 인분 src=db/음식_016227 (떡볶이) grams=300.0 kcal=432.0 → S1_parse=pass:literal; S2_match=pass:db:음식_016227; S3_grams=pass:300g; S4_kcal=pass:432.0kcal vs [319.0, 1140.2]
  - sundae: `순대` 1.0 인분 src=db/음식_012919 (순대) grams=200.0 kcal=368.0 → S1_parse=pass:literal; S2_match=pass:db:음식_012919; S3_grams=pass:200g; S4_kcal=pass:368.0kcal vs [326.2, 645.2]
  - squid_fritter: `튀김` 3.0 개 src=db/음식_016949 (닭튀김) grams=300.0 kcal=903.0 → S1_parse=pass:literal; S2_match=fail:db:음식_016949
  - total 1703.0 vs [739.0, 2075.9] (ok)
### dv1_254 [dev] `S1_parse` — Breakfast: banana 1, greek yogurt 150g, americano
- branch `llm`; parsed: banana(1.0 ), greek yogurt(150.0 g), americano(None )
- case S1: ['extra_items:llm']; extra: [{'food_name': 'banana', 'amount': 1.0, 'unit': None, 'effective_kcal': 116.5}, {'food_name': 'greek yogurt', 'amount': 150.0, 'unit': 'g', 'effective_kcal': 148.5}]
  - banana: missing → S1_parse=fail:missing [S1 cause: llm]
  - greek_yogurt: missing → S1_parse=fail:missing [S1 cause: llm]
  - americano: `americano` None  src=llm grams=None kcal=0 → S1_parse=pass:no_literal_amount; S2_match=fail:llm_estimate(db empty_macros; sanity kept)
  - total 265.0 vs [138.7, 399.7] (ok)

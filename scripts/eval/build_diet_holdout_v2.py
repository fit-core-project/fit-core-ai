"""Build tests/evaluation/datasets/diet_eval_holdout_v2.jsonl (second blind held-out Korean diet-log set).

Purpose: a HELD-OUT set written blind to the pipeline's outputs, to measure generalisation of a parser that was tuned on
diet_eval_v1 / diet_eval_products_v1 / diet_eval_holdout_v1. Every case is split "test", tagged set:holdout_v2.

Labels describe what is CORRECT for a real person's meal, never what the code returns. Label sources (only these):
  - data/food_db/food_db_clean.csv          (generic foods: gold_food_ids + kcal/100 of those rows)
  - data/food_db/raw/식품의약품안전처_통합식품영양성분정보(가공식품)_20261111.csv
                                             (branded/packaged products and alcohol: label values, asserted at build)
  - public common-knowledge portion sizes (stated in basis / note)
engines/*, data/curated/*, processed_product_index.csv, LLM caches, baselines and eval reports were NOT consulted.

Schema = diet_eval_holdout_v1.jsonl (same conventions as build_diet_holdout_v1.py):
  - names: acceptable food names; gold_food_ids: acceptable food_db_clean rows (+ "curated" / "llm_estimate_ok" tokens)
  - amount: literal quantity [lo, hi] (null = no number). Sharing / partial-eating: [per-person share, literal total];
    explicit ranges ("두세 개", "7~8개"): [lo, hi] of the stated range.
  - grams [lo, hi]: public-standard portion range (ml treated as g); always the amount the person actually ate
  - kcal [lo, hi]: generic = min/max DB kcal/100 x grams, widened 10% and +-5 kcal;
                   product = min/max MFDS label kcal/100 x single-unit size x quantity, widened 10%
Near-duplicate texts (bigram Jaccard >= 0.4 vs the existing sets, the scratch inputs, or an earlier v2 case) are DROPPED
before ids are assigned.

Run:  PYTHONUTF8=1 ./venv/Scripts/python scripts/eval/build_diet_holdout_v2.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "food_db" / "food_db_clean.csv"
MFDS_PATH = ROOT / "data" / "food_db" / "raw" / "식품의약품안전처_통합식품영양성분정보(가공식품)_20261111.csv"
DATASETS = ROOT / "tests" / "evaluation" / "datasets"
OUT_PATH = DATASETS / "diet_eval_holdout_v2.jsonl"
EXISTING = [DATASETS / "diet_eval_v1.jsonl", DATASETS / "diet_eval_products_v1.jsonl", DATASETS / "diet_eval_holdout_v1.jsonl"]
TEMPLATE = DATASETS / "diet_eval_holdout_v1.jsonl"

KCAL_TOL = 0.10
NEAR_JACCARD = 0.4  # cases at or above this vs existing / earlier v2 texts are dropped
KCAL_ABS = 5.0
LLM = "llm_estimate_ok"
CUR = "curated"
G_ML = {"g": (1, 1), "ml": (1, 1)}
SET_TAG = "set:holdout_v2"


# ---------------------------------------------------------------------------
# Registry.  units: grams per ONE unit (lo, hi); serving: portion when no number is given.
#   F(...)  generic food: gold = DB ids (+ optional tokens)
#   P(...)  branded product / alcohol: mfds = [(FOOD_CD, kcal_per_100, "exact FOOD_NM")], units = single-unit size
# ---------------------------------------------------------------------------
def F(group, names, gold, units=None, serving=None, est=None, est_basis=None, note=None):
    return dict(kind="generic", group=group, names=names, gold=gold, units=units or {}, serving=serving,
                est=est, est_basis=est_basis, note=note)


def P(group, names, mfds, units, serving, note, gold=(LLM, CUR)):
    return dict(kind="product", group=group, names=names, gold=list(gold), units=units, serving=serving,
                mfds=mfds, note=note)


R = {
    # ---------------- reused staples (definitions copied from build_diet_holdout_v1.py; side items only) ----------------
    "rice": F("밥", ["쌀밥", "밥", "공기밥", "공깃밥", "흰쌀밥", "흰밥", "rice"], ["음식_011600", "음식_011599", CUR],
              units={"공기": (190, 230), "그릇": (190, 230), **G_ML}, serving=(190, 230),
              note="밥 1공기 210g(식약처 1회 제공량)"),
    "egg": F("계란유제품", ["계란", "달걀", "삶은계란", "삶은 계란", "egg", "eggs"], [CUR, "원재_001702", "음식_017044", "원재_001700"],
             units={"개": (50, 60), "알": (50, 60)}, serving=(50, 60)),
    "kimchi": F("채소샐러드", ["김치", "배추김치"], ["음식_014367"], units=G_ML, serving=(30, 80)),
    "milk": F("계란유제품", ["우유", "milk", "흰우유"], ["원재_001728", "음식_010323"],
              units={"팩": (190, 200), "컵": (200, 250), "잔": (200, 250), **G_ML}, serving=(200, 250)),
    "americano": F("음료", ["아메리카노", "아아", "아이스 아메리카노", "아이스아메리카노", "커피", "americano", "iced americano"],
                   ["음식_009004", "음식_008945", "음식_008940"], units={"잔": (300, 500)}, serving=(300, 500)),
    "fries": F("패스트푸드", ["감자튀김", "프렌치프라이", "fries", "후렌치후라이"], ["음식_017952"], units={"개": (100, 130)},
               serving=(100, 130), note="미디엄 100-130g"),
    "cola": F("음료", ["콜라", "코카콜라", "coke"], [CUR, LLM], units={"캔": (250, 355), "잔": (200, 300), **G_ML},
              serving=(250, 355), est=(40, 45), est_basis="콜라 40-45kcal/100ml(라벨)"),
    "tri_gimbap": F("편의점브랜드", ["삼각김밥", "참치마요", "삼김"], ["음식_013454", "음식_013455", "음식_013456", "음식_013457"],
                    units={"개": (100, 120)}, serving=(100, 120), note="편의점 삼각김밥 100-120g"),
    "soju": P("주류", ["소주", "참이슬", "처음처럼", "진로"],
              [("P115-102020100-0015", 89, "좋은데이"), ("P115-102020100-0012", 89, "새로"), ("P115-102020100-0005", 100, "시원"),
               ("P115-102020100-0001", 127, "소주(알코올 17.8)")],
              units={"병": (360, 360), "잔": (45, 55)}, serving=(360, 360),
              note="소주 360ml 병; 16-17.8% 행 89-127kcal/100ml(알코올 에너지 포함)"),

    # ---------------- 밥 / 덮밥 ----------------
    "hayashi": F("밥", ["하이라이스", "하이라이스덮밥"], ["음식_002366", "음식_002367"],
                 units={"그릇": (400, 550), "인분": (400, 550), "접시": (400, 550)}, serving=(400, 550), note="밥 포함 1그릇 400-550g"),
    "japchae_bap": F("밥", ["잡채밥"], ["음식_010126", "음식_010125"], units={"그릇": (450, 600), "인분": (450, 600)},
                     serving=(450, 600), note="중식 잡채밥 1그릇 450-600g"),
    "jjamppong_bap": F("밥", ["짬뽕밥"], ["음식_010021", "음식_010020"], units={"그릇": (700, 900), "인분": (700, 900)},
                       serving=(700, 900), note="짬뽕밥 1그릇(국물+밥) 700-900g"),
    "unagi_don": F("밥", ["장어덮밥", "장어 덮밥", "우나기동", "덮밥"], ["음식_016723"],
                   units={"그릇": (350, 450), "인분": (350, 450)}, serving=(350, 450)),
    "nakji_bowl": F("밥", ["낙지덮밥", "낙지 덮밥", "덮밥"], ["음식_016730", "음식_017138"],
                    units={"그릇": (400, 550), "인분": (400, 550)}, serving=(400, 550)),
    "chicken_bowl": F("밥", ["닭고기덮밥", "오야꼬동", "오야코동", "덮밥"], ["음식_016729"],
                      units={"그릇": (400, 500), "인분": (400, 500)}, serving=(400, 500)),
    "chungmu": F("밥", ["충무김밥", "김밥"], ["음식_009928", "음식_009929"], units={"인분": (250, 350), "접시": (250, 350)},
                 serving=(250, 350), note="충무김밥 1인분(김밥 8개+오징어무침+석박지) 250-350g"),
    # ---------------- 죽 / 국 / 찌개 ----------------
    "abalone_juk": F("국찌개", ["전복죽", "죽"], ["음식_010090", "음식_010089"], units={"그릇": (400, 600)}, serving=(400, 600),
                     note="죽 전문점 1그릇 400-600g"),
    "hobak_juk": F("국찌개", ["호박죽", "단호박죽", "죽"], ["음식_001964", "음식_001963"], units={"그릇": (300, 450)},
                   serving=(300, 450)),
    "patjuk": F("국찌개", ["팥죽", "동지팥죽", "죽"], ["음식_006971", "음식_006970"], units={"그릇": (350, 500)}, serving=(350, 500)),
    "altang": F("국찌개", ["알탕"], ["음식_011003", "음식_011001"], units={"그릇": (500, 700), "인분": (500, 700)},
                serving=(500, 700), note="음식_011002(25kcal/100ml) 제외: 다른 알탕 행(39-46) 대비 이상치"),
    "hwangtae": F("국찌개", ["황태해장국", "황태국", "북엇국", "해장국"], ["음식_001882", "음식_007622"], units={"그릇": (600, 800)},
                  serving=(600, 800)),
    "bugeoguk": F("국찌개", ["북엇국", "북어국", "황태국"], ["음식_013782", "음식_013780", "음식_013781"],
                  units={"그릇": (350, 550), "대접": (350, 550)}, serving=(350, 550)),
    "kongnamul_guk": F("국찌개", ["콩나물국"], ["음식_007620", "음식_007621", "음식_007613"], units={"그릇": (250, 400)},
                       serving=(250, 400)),
    "sogogi_muguk": F("국찌개", ["소고기무국", "소고기 뭇국", "소고기뭇국", "무국", "뭇국"], ["음식_013090", "음식_014869", "음식_014870"],
                      units={"그릇": (300, 500), "대접": (300, 500)}, serving=(300, 500)),
    "eomuk_tang": F("국찌개", ["어묵탕", "오뎅탕", "어묵국", "오뎅"], ["음식_010903", "음식_010920", "음식_010921"],
                    units={"그릇": (300, 500), "개": (300, 500), "인분": (300, 500)}, serving=(300, 500)),
    "gochujang_jjigae": F("국찌개", ["고추장찌개", "고추장 찌개"], ["음식_017855", "음식_016354"],
                          units={"그릇": (300, 450), "인분": (300, 450)}, serving=(300, 450)),
    "jjageuli": F("국찌개", ["짜글이", "돼지고기 짜글이"], ["음식_016355"], units={"인분": (300, 400), "그릇": (300, 400)},
                  serving=(300, 400)),
    "seonji": F("국찌개", ["선지해장국", "선짓국", "해장국"], ["음식_002283", "음식_013108"],
                units={"그릇": (600, 800), "뚝배기": (600, 800)}, serving=(600, 800)),
    # ---------------- 면 ----------------
    "soba": F("면", ["메밀소바", "냉모밀", "모밀", "소바", "모밀국수"], ["음식_014945", "음식_014882"],
              units={"판": (350, 500), "그릇": (350, 500), "인분": (350, 500)}, serving=(350, 500), note="냉모밀 1판(장국 포함) 350-500g"),
    "pad_thai": F("면", ["팟타이", "pad thai", "볶음쌀국수"], ["음식_007017"],
                  units={"개": (250, 350), "접시": (250, 350), "인분": (250, 350)}, serving=(250, 350),
                  note="음식_007016(간편조리세트 231) 제외: 조리 전 세트 기준"),
    "carbonara": F("면", ["까르보나라", "까르보나라 파스타", "파스타", "스파게티"], ["음식_011781", "음식_011783", "음식_011795"],
                   units={"인분": (300, 400), "접시": (300, 400), "개": (300, 400)}, serving=(300, 400)),
    "tonkotsu": F("면", ["돈코츠라멘", "돈코츠 라멘", "라멘"], ["음식_015334", "음식_015335"], units={"그릇": (600, 800)},
                  serving=(600, 800), note="라멘 1그릇(국물 포함) 600-800g"),
    "yaki_udon": F("면", ["야끼우동", "볶음우동", "우동볶음"], ["음식_010349"], units={"접시": (300, 400), "인분": (300, 400)},
                   serving=(300, 400)),
    # ---------------- 분식 / 빵 / 디저트 ----------------
    "rose_tteok": F("빵과자", ["로제떡볶이", "로제 떡볶이", "떡볶이"], ["음식_016214", "음식_016199"], units={"인분": (300, 450)},
                    serving=(300, 450)),
    "gun_mandu": F("빵과자", ["군만두", "만두"], ["음식_015037", "음식_017607"], units={"개": (25, 35)}, serving=(150, 210)),
    "dak_kkochi": F("빵과자", ["닭꼬치", "닭꼬치구이"], ["음식_016998", "음식_017182"], units={"개": (80, 130)}, serving=(80, 130)),
    "bungeoppang": F("빵과자", ["붕어빵", "팥붕"], ["음식_006987", "음식_006979"], units={"마리": (35, 55), "개": (35, 55)},
                     serving=(35, 55), note="길거리 붕어빵 1마리 35-55g"),
    "gyeranppang": F("빵과자", ["계란빵", "달걀빵"], ["음식_017896", "음식_017893"], units={"개": (90, 130)}, serving=(90, 130)),
    "croquette": F("빵과자", ["고로케", "야채고로케", "크로켓", "croquette", "croquettes"], ["음식_007448", "음식_007451", "음식_007450"],
                   units={"개": (80, 120)}, serving=(80, 120)),
    "salt_bread": F("빵과자", ["소금빵", "시오빵"], ["음식_013039", "음식_013043"], units={"개": (60, 90)}, serving=(60, 90),
                    note="소금빵 1개 60-90g"),
    "danpat": F("빵과자", ["단팥빵", "팥빵"], ["음식_007012", "음식_006989"], units={"개": (80, 130)}, serving=(80, 130)),
    "croffle": F("빵과자", ["크로플", "플레인 크로플"], ["음식_007287", "음식_007288"], units={"개": (50, 80)}, serving=(50, 80)),
    "tiramisu": F("빵과자", ["티라미수", "티라미수 케이크", "tiramisu"], ["음식_007729"], units={"조각": (100, 130), "개": (100, 130)},
                  serving=(100, 130)),
    "castella": F("빵과자", ["카스테라", "카스텔라"], ["음식_009733", "음식_009737"], units={"조각": (50, 80), "개": (50, 80)},
                  serving=(50, 80)),
    "yakgwa": F("빵과자", ["약과", "미니약과"], ["음식_010959"], units={"개": (20, 40)}, serving=(20, 40), note="약과 1개 20-40g(미니~일반)"),
    "baekseolgi": F("빵과자", ["백설기", "떡"], ["음식_014354"], units={"덩이": (80, 120), "개": (80, 120), "조각": (80, 120)},
                    serving=(80, 120)),
    "songpyeon": F("빵과자", ["송편", "떡"], ["음식_012944", "음식_012946", "음식_012945"], units={"개": (15, 25)}, serving=(75, 125)),
    "kkuldeok": F("빵과자", ["꿀떡", "떡"], ["음식_017149"], units={"개": (10, 15)}, serving=(50, 100)),
    "patbingsu": F("빵과자", ["팥빙수", "빙수"], ["음식_007013", "음식_007014", "음식_013488"], units={"개": (400, 600), "그릇": (400, 600)},
                   serving=(400, 600), note="카페 빙수 1개(2인용 포함) 400-600g"),
    # ---------------- 고기 / 요리 ----------------
    "dakgalbi": F("고기", ["닭갈비", "춘천닭갈비", "철판닭갈비"], ["음식_016970", "음식_016971", "음식_016965"],
                  units={"인분": (250, 350)}, serving=(250, 350), note="철판 닭갈비 1인분(채소·떡 포함) 250-350g"),
    "jjimdak": F("고기", ["찜닭", "안동찜닭"], ["음식_016950", "음식_016952"], units={"마리": (900, 1300)}, serving=(400, 600),
                 note="안동찜닭 1마리(당면·감자 포함) 900-1300g"),
    "nakji_bokkeum": F("고기", ["낙지볶음"], ["음식_017133", "음식_017132"], units={"인분": (250, 350), "접시": (250, 350)},
                       serving=(250, 350)),
    "jukkumi": F("고기", ["쭈꾸미볶음", "주꾸미볶음", "쭈꾸미", "주꾸미"], ["음식_010060", "음식_010059"],
                 units={"인분": (250, 350), **G_ML}, serving=(250, 350)),
    "ojingeo_bokkeum": F("고기", ["오징어볶음"], ["음식_010601", "음식_010600"], units={"인분": (250, 350), "접시": (250, 350)},
                         serving=(250, 350)),
    "duck_roast": F("고기", ["오리로스", "오리구이", "생오리구이", "오리고기"], ["원재_001687", "원재_001686"],
                    units={"인분": (150, 200), **G_ML}, serving=(150, 200), note="오리고기 껍질 포함 구운/삶은 행; 1인분 150-200g"),
    "hangjeong": F("고기", ["항정살", "항정살구이"], ["원재_001369"], units={"인분": (150, 200), **G_ML}, serving=(150, 200),
                   note="생고기 기준"),
    "galmaegi": F("고기", ["갈매기살", "갈매기살구이"], ["음식_018022", "원재_001365"], units={"인분": (150, 200), **G_ML},
                  serving=(150, 200)),
    "pork_skin": F("고기", ["돼지껍데기", "껍데기", "돼지껍데기구이"], ["음식_016351"], units={"접시": (150, 250), "인분": (150, 250)},
                   serving=(150, 250)),
    "daechang": F("고기", ["대창", "소대창", "대창구이"], ["음식_016746"], units={"인분": (150, 200), **G_ML}, serving=(150, 200)),
    "kkanpunggi": F("배달", ["깐풍기"], ["음식_017211"], units={"접시": (250, 400), "인분": (250, 400)}, serving=(250, 400)),
    "hamburg": F("고기", ["함박스테이크", "함박"], ["음식_002364", "음식_002365"], units={"인분": (200, 300), "개": (200, 300)},
                 serving=(200, 300), note="소스 포함 함박 1인분 200-300g"),
    "mapo": F("채소샐러드", ["마파두부"], ["음식_015042", "음식_015043"], units={"접시": (150, 250)}, serving=(150, 250),
               note="덜어먹는 1접시 150-250g"),
    "haemul_jjim": F("생선", ["해물찜"], ["음식_002294", "음식_002293"], units={"인분": (350, 500)}, serving=(350, 500)),
    "agujjim": F("생선", ["아구찜", "아귀찜"], ["음식_011567", "음식_011566"], units={"인분": (350, 500)}, serving=(350, 500),
                 note="음식_011568(30kcal/100ml) 제외: g 행의 40% 이하로 이상치"),
    # ---------------- 생선 ----------------
    "jogi": F("생선", ["조기구이", "조기", "굴비"], ["음식_010069", "음식_010068"], units={"마리": (60, 120)}, serving=(60, 120),
              note="조기 1마리 가식부 60-120g; 음식_010070(오븐구이 40/100ml)·굴비구이(392/100ml) 이상치 제외"),
    "salmon_gui": F("생선", ["연어구이", "연어 스테이크", "연어"], ["음식_010693", "음식_010692"],
                    units={"토막": (100, 150), "조각": (100, 150)}, serving=(100, 150)),
    # ---------------- 과일 / 채소 / 견과 / 반찬 ----------------
    "grapefruit": F("과일", ["자몽", "grapefruit"], ["원재_001241"], units={"개": (250, 350)}, serving=(250, 350),
                    note="자몽 1개 가식부 250-350g"),
    "pomegranate": F("과일", ["석류"], ["원재_001212"], units={"개": (150, 250)}, serving=(150, 250), note="석류 1개 가식부(알) 150-250g"),
    "fig": F("과일", ["무화과"], ["원재_001179", "원재_001181"], units={"개": (40, 70)}, serving=(40, 70)),
    "dragonfruit": F("과일", ["용과"], ["원재_001230", "원재_001231"], units={"개": (250, 400)}, serving=(250, 400),
                     note="용과 1개 가식부 250-400g"),
    "lychee": F("과일", ["리치", "lychee"], ["원재_001161", "원재_001162"], units={"알": (10, 15), "개": (10, 15)}, serving=(100, 150)),
    "gotgam": F("과일", ["곶감"], ["원재_001130"], units={"개": (30, 50)}, serving=(30, 50)),
    "carrot": F("채소샐러드", ["당근", "당근스틱"], ["원재_000597", "원재_000598"], units={"개": (150, 200)}, serving=(80, 150)),
    "paprika": F("채소샐러드", ["파프리카"], ["원재_000930", "원재_000928"], units={"개": (130, 180)}, serving=(130, 180)),
    "pistachio": F("빵과자", ["피스타치오", "견과류"], ["원재_000505", "원재_000504"], units={"줌": (20, 30), **G_ML}, serving=(20, 30)),
    "dried_sp": F("빵과자", ["고구마말랭이", "말랭이"], ["원재_000381"], units={"봉": (50, 60), "봉지": (50, 60), **G_ML}, serving=(50, 60)),
    "dubu_jorim": F("채소샐러드", ["두부조림"], ["음식_016268", "음식_016267"], units=G_ML, serving=(80, 150)),
    "myeolchi": F("채소샐러드", ["멸치볶음"], ["음식_014932", "음식_014925"], units=G_ML, serving=(15, 30)),
    "jinmichae": F("채소샐러드", ["진미채", "오징어채볶음", "진미채볶음"], ["음식_010587", "음식_010585"], units=G_ML, serving=(20, 40)),
    "gamja_jorim": F("채소샐러드", ["감자조림"], ["음식_017956", "음식_017955"], units=G_ML, serving=(50, 100)),
    # ---------------- 음료 ----------------
    "cappuccino": F("음료", ["카푸치노", "cappuccino"], ["음식_008558", "음식_008559", "음식_008549"], units={"잔": (240, 400)},
                    serving=(240, 400)),
    "einspanner": F("음료", ["아인슈페너", "비엔나커피"], ["음식_008924", "음식_008921", "음식_008918"], units={"잔": (300, 400)},
                    serving=(300, 400)),
    "flat_white": F("음료", ["플랫화이트", "플랫 화이트", "flat white"], ["음식_008375", "음식_008376"], units={"잔": (240, 355)},
                    serving=(240, 355)),
    "lemonade": F("음료", ["레몬에이드", "레몬 에이드", "에이드"], ["음식_010853", "음식_010864", "음식_010862"],
                  units={"잔": (400, 600)}, serving=(400, 600)),
    "grapefruit_ade": F("음료", ["자몽에이드", "자몽 에이드", "에이드"], ["음식_010757", "음식_010770"], units={"잔": (400, 600)},
                        serving=(400, 600)),
    "yuja": F("음료", ["유자차", "유자티"], ["음식_010271", "음식_010272", "음식_010267"], units={"잔": (300, 400)}, serving=(300, 400)),
    "sikhye": F("음료", ["식혜"], ["음식_011605", "음식_011603", "음식_011608"], units={"컵": (200, 250), "잔": (200, 250)},
                serving=(200, 250), note="음식_011610(87/100g) 제외: 다른 식혜 행(40-47)의 2배"),
    "tomato_juice": F("음료", ["토마토주스", "토마토 주스"], ["음식_017648", "음식_017649", "음식_017679"],
                      units={"잔": (200, 250), "컵": (200, 250)}, serving=(200, 250),
                      note="음식_017646(병음료 37) 제외: 가당 병음료"),
    # ---------------- 프랜차이즈 (food_db 음식 행; flag:product) ----------------
    "abalone_juk_bon": F("국찌개", ["본죽 전복죽", "전복죽", "죽"], ["음식_010090", "음식_010089"], units={"그릇": (450, 600)},
                         serving=(450, 600), note="본죽 일반 1그릇 450-600g(통상 중량 추정)"),
    "mcchicken": F("패스트푸드", ["맥치킨", "맥치킨버거", "버거"], ["음식_014242"], units={"개": (160, 200)}, serving=(160, 200),
                   note="맥치킨 1개 160-200g(통상 중량 추정)"),
    "burger_1955": F("패스트푸드", ["1955버거", "1955 버거", "버거"], ["음식_014312"], units={"개": (240, 290), "세트": (240, 290)},
                     serving=(240, 290), note="1955버거 1개 240-290g(통상 중량 추정)"),
    "whopper_jr": F("패스트푸드", ["와퍼주니어", "와퍼 주니어", "와퍼쥬니어", "버거"], ["음식_002203", "음식_014140"],
                    units={"개": (160, 190)}, serving=(160, 190), note="와퍼주니어 1개 160-190g(통상 중량 추정)"),
    "zinger": F("패스트푸드", ["징거버거", "징거 버거", "징거", "버거"], ["음식_014130"], units={"개": (190, 240)}, serving=(190, 240),
                note="징거버거 1개 190-240g(통상 중량 추정)"),
    "teri_burger": F("패스트푸드", ["데리버거", "데리 버거", "버거"], ["음식_002256"], units={"개": (130, 160)}, serving=(130, 160),
                     note="데리버거 1개 130-160g(통상 중량 추정)"),
    "golden_olive": F("배달", ["황금올리브", "황올", "후라이드", "치킨"], ["음식_016776", "음식_016773"], units={"마리": (800, 1100)},
                      serving=(300, 450), note="BBQ 황금올리브 1마리 800-1100g"),
    "gochu_basasak": F("배달", ["고추바사삭", "치킨"], ["음식_016941"], units={"조각": (80, 120)}, serving=(300, 450),
                       note="뼈 포함 1조각 80-120g"),
    "grapefruit_blacktea": F("음료", ["자몽 허니 블랙티", "자몽허니블랙티", "자허블"], ["음식_010181", "음식_010180", "음식_010183"],
                             units={"잔": (355, 473)}, serving=(355, 473), note="스타벅스 Tall 355 / Grande 473ml"),
}

# ---------------- 브랜드 제품 (MFDS 가공식품 라벨; 각 행은 빌드 시 원본 CSV와 대조) ----------------
_RAMEN = lambda g: {"개": (g, g), "봉지": (g, g), "봉": (g, g)}
PR = {
    "jin_ramen": P("편의점브랜드", ["진라면", "진라면 순한맛", "라면"],
                   [("P108-003000400-0993", 417, "진라면(순한맛)"), ("P108-003000400-0243", 417, "진라면(순한맛)")],
                   _RAMEN(120), (120, 120), "진라면 순한맛 1봉 120g(3600g 업소용 행 제외)"),
    "redbull": P("음료", ["레드불", "red bull", "에너지드링크"],
                 [("P109-401040100-1274", 45, "레드불 에너지 드링크"), ("P109-401040100-1273", 44, "레드불 레드 에디션")],
                 {"캔": (250, 250), "개": (250, 250)}, (250, 250), "레드불 250ml; 115kcal/100ml 행은 1캔값 오기로 보고 제외"),
    "galaman_bae": P("음료", ["갈아만든배", "갈아만든 배"], [("P109-303030300-0046", 49, "갈아만든배"), ("P109-303030300-0047", 48, "갈아만든배")],
                     {"캔": (238, 340), "개": (238, 340)}, (238, 340), "갈아만든배 캔 238/340ml(저칼로리·퓨레 행 제외)"),
    "bilak_sikhye": P("음료", ["비락식혜", "식혜"], [("P109-600080100-0039", 34, "비락식혜")], {"캔": (238, 238), "개": (238, 238)},
                      (238, 238), "비락식혜 238ml 캔"),
    "welchs": P("음료", ["웰치스", "웰치스 딸기"], [("P109-401040100-0398", 54, "웰치스 스트로베리맛")], {"캔": (355, 355), "개": (355, 355)},
                (355, 355), "웰치스 스트로베리 355ml"),
    "seolleim": P("편의점브랜드", ["설레임", "설레임 밀크쉐이크", "아이스크림"],
                  [("P102-103010300-1206", 113, "설레임 밀크쉐이크"), ("P102-103010300-1205", 113, "설레임 밀크쉐이크")],
                  {"개": (160, 160)}, (160, 160), "설레임 밀크쉐이크 160ml"),
    "bulgaris": P("계란유제품", ["불가리스", "불가리스 사과", "요구르트"], [("P119-202040200-1578", 90, "불가리스 사과")],
                  {"병": (150, 150), "개": (150, 150)}, (150, 150), "불가리스 사과 150ml"),
    "seoul_coffee": P("계란유제품", ["서울우유 커피", "커피우유", "서울우유커피"],
                      [("P109-201020000-0183", 60, "서울우유커피"), ("P109-201020000-2929", 65, "서울우유 커피")],
                      {"팩": (200, 200), "개": (200, 200)}, (200, 200), "서울우유 커피 200ml 팩"),
    "sariguk": P("편의점브랜드", ["사리곰탕면", "사리곰탕", "라면"],
                 [("P108-003000400-0109", 432, "사리곰탕면"), ("P108-002000400-0012", 445, "사리곰탕면")],
                 _RAMEN(110), (110, 110), "사리곰탕면 1봉 110g(550g 5입 행 제외)"),
    "snack_myeon": P("편의점브랜드", ["스낵면", "라면"], [("P108-003000400-0969", 440, "스낵면"), ("P108-003000400-0134", 440, "스낵면")],
                     _RAMEN(120), (120, 120), "스낵면 1봉 120g"),
    "jin_bibim": P("편의점브랜드", ["진비빔면", "비빔면"], [("P108-009000400-0051", 401, "진비빔면")], _RAMEN(156), (156, 156),
                   "진비빔면 1봉 156g"),
    "baehongdong": P("편의점브랜드", ["배홍동비빔면", "배홍동", "비빔면"], [("P108-009000400-0207", 427, "배홍동비빔면")],
                     _RAMEN(137), (137, 137), "배홍동비빔면 1봉 137g(548g 4입 행 제외)"),
    "yangban_juk": P("편의점브랜드", ["양반 전복죽", "양반죽", "전복죽", "죽"], [("P123-205020300-0887", 57, "양반 전복죽")],
                     {"개": (287.5, 287.5), "캔": (287.5, 287.5), "그릇": (287.5, 287.5)}, (287.5, 287.5), "양반 전복죽 287.5g 용기"),
    "3min_jjajang": P("편의점브랜드", ["3분짜장", "삼분짜장", "짜장"], [("P113-206020400-8178", 108, "3분짜장"), ("P113-206020400-0003", 108, "3분짜장")],
                      {"개": (200, 200), "봉지": (200, 200)}, (200, 200), "오뚜기 3분짜장 200g(소스만)"),
    "yangparing": P("편의점브랜드", ["양파링"], [("P101-106000100-7172", 469, "양파링 오리지널"), ("P101-106000100-1211", 467, "양파링Original")],
                    {"봉지": (80, 84), "봉": (80, 84), "개": (80, 84)}, (80, 84), "양파링 80/84g 봉"),
    "tankboy": P("편의점브랜드", ["탱크보이", "탱크보이 배", "아이스크림"],
                 [("P102-301030000-1002", 75, "탱크보이 배"), ("P102-301030000-0534", 75, "탱크보이 배")],
                 {"개": (120, 120)}, (120, 120), "탱크보이 배 120ml"),
    "deowi": P("편의점브랜드", ["더위사냥", "더위사냥 커피", "아이스크림"],
               [("P102-301030000-0908", 89, "더위사냥 커피"), ("P102-301030000-0028", 89, "더위사냥actiV커피")],
               {"개": (140, 140)}, (140, 140), "더위사냥 커피 140ml(반으로 쪼개는 2입 구조)"),
    "dejawa": P("음료", ["데자와", "데자와 밀크티", "밀크티"], [("P109-102010200-0124", 40, "데자와로얄밀크티")],
                {"캔": (240, 240), "개": (240, 240)}, (240, 240), "데자와 240ml 캔(500ml 페트 행 38)"),
    "sprite": P("음료", ["스프라이트", "사이다"], [("P109-401040100-0955", 45, "스프라이트"), ("P109-401040100-0289", 48, "스프라이트")],
                {"캔": (250, 355), "개": (250, 355)}, (250, 355), "스프라이트 캔 250/355ml"),
    "gt_milk": P("계란유제품", ["맛있는우유GT", "맛있는우유", "남양 맛있는우유", "우유"], [("P119-102010100-0003", 65, "남양맛있는우유GT/ 멸균")],
                 {"팩": (200, 200), "개": (200, 200)}, (200, 200), "맛있는우유GT 200ml 팩"),
    # branded alcohol
    "daeseon": P("주류", ["대선", "대선소주", "소주"], [("P115-102020100-0010", 88, "대선159")], {"병": (360, 360), "잔": (45, 55)},
                 (360, 360), "대선 360ml 병"),
    "jamongjin": P("주류", ["순하리 자몽진", "자몽진"], [("P115-105010500-0006", 53, "순하리 자몽진 9")], {"캔": (355, 500)}, (355, 500),
                   "순하리 자몽진 9% 53kcal/100ml, 캔 355/500ml"),
    "saero_darae": P("주류", ["새로 다래", "새로다래", "새로"], [("P115-207020400-0004", 68, "새로 다래")], {"병": (360, 360)}, (360, 360),
                     "새로 다래 360ml"),
    "wolmae": P("주류", ["월매 막걸리", "월매", "막걸리"], [("P115-101010100-0011", 41, "월매 막걸리")], {"병": (750, 750)}, (750, 750),
                "월매 막걸리 750ml"),
    "dalbit_yuja": P("주류", ["달빛유자", "달빛유자 막걸리", "막걸리"], [("P115-101010100-0010", 56, "달빛유자")], {"병": (750, 750)},
                     (750, 750), "달빛유자 750ml"),
    "cheongha": P("주류", ["청하", "청주"], [("P115-103010300-0001", 132, "청주(알코올 16)")], {"병": (300, 300), "잔": (50, 70)},
                  (300, 300), "청하 300ml 병; MFDS 청주(16%) 행만 존재 -> 하한은 13% 청하 공개값 약 80kcal/100ml로 확장"),
    "goryangju": P("주류", ["고량주", "빼갈", "연태고량주"], [("P115-206020400-0001", 355, "고량주(알코올 50)")],
                   {"잔": (20, 50)}, (20, 50), "고량주 1잔 20-50ml; 연태(34%) 등 저도수 고려 하한 240kcal/100ml로 확장"),
    "bokbunja": P("주류", ["복분자주", "복분자", "복분자술"], [("P115-105010500-0001", 137, "복분자주수(알코올 15)")],
                  {"잔": (50, 100), "병": (375, 375)}, (50, 100), "복분자주 1잔 50-100ml, 병 375ml"),
}
assert not (set(R) & set(PR))
R.update(PR)
# label rows that do not span the realistic range -> explicit extra density bounds (stated in note)
EXTRA_DENS = {"cheongha": (80, 132), "goryangju": (240, 355)}


# ---------------------------------------------------------------------------
# Case DSL
# ---------------------------------------------------------------------------
def I(key, amt=None, unit=None, grams=None, optional=False, units_ok=None, note=None, low=False):
    return dict(key=key, amt=amt, unit=unit, grams=grams, optional=optional, units_ok=units_ok, note=note, low=low)


CASES: list[dict] = []


def C(text, items, noise=(), tags=(), conf="high", note=None, max_items=None, min_items=None):
    CASES.append(dict(text=text, items=items, noise=list(noise), extra_tags=list(tags), conf=conf, note=note,
                      max_items=max_items, min_items=min_items))


SHARE = "flag:share"
SIZE = "flag:size"
PROD = "flag:product"
RANGE = "flag:range"
PART = "flag:partial"
NEG = "flag:negation"

# ======================= 밥 / 덮밥 =======================
C("하이라이스 한 그릇 뚝딱함", [I("hayashi", 1, "그릇")])
C("중국집에서 잡채밥 시켜먹었음", [I("japchae_bap")])
C("잡채밥 곱빼기로 시켰더니 양이 어마어마함", [I("japchae_bap", grams=(675, 900))], tags=["unit:크기", SIZE], noise=["잡담"],
  conf="medium", note="곱빼기 = 보통(450-600g)의 1.5배")
C("짬뽕밥 먹었는데 국물까지 싹 다 마심", [I("jjamppong_bap")], noise=["잡담"])
C("장어덮밥 1인분 ㄹㅇ 존맛", [I("unagi_don", 1, "인분")], noise=["슬랭"])
C("낙지덮밥에 콩나물국 같이 나와서 둘 다 먹음", [I("nakji_bowl"), I("kongnamul_guk")])
C("오야꼬동 한그릇", [I("chicken_bowl", 1, "그릇")], noise=["띄어쓰기"], conf="medium", note="오야꼬동 = 닭고기덮밥 행")
C("통영 와서 충무김밥 1인분 포장", [I("chungmu", 1, "인분")], noise=["잡담"])
C("하이라이쓰 반 그릇만 먹었당", [I("hayashi", 0.5, "그릇")], noise=["오타"])

# ======================= 죽 / 국 / 찌개 =======================
C("아파서 호박죽 반 그릇 데워먹음", [I("hobak_juk", 0.5, "그릇")], noise=["잡담"])
C("동지라 팥죽 한그릇 먹었다", [I("patjuk", 1, "그릇")], noise=["띄어쓰기"])
C("점심 알탕+밥", [I("altang"), I("rice")], noise=["끼니"])
C("해장은 역시 황태해장국", [I("hwangtae")], noise=["잡담"])
C("아침에 북엇국에 밥 말아먹고 출근", [I("bugeoguk"), I("rice")], noise=["끼니"])
C("엄마가 끓여준 소고기뭇국 한 대접", [I("sogogi_muguk", 1, "대접")])
C("분식집 어묵탕 하나 시킴", [I("eomuk_tang", 1, "개", units_ok=["개", "그릇", "인분"])], conf="medium")
C("고추장찌개 끓여서 밥 2/3공기랑", [I("gochujang_jjigae"), I("rice", 2 / 3, "공기")], tags=["unit:분수"])
C("짜글이 1인분 완전 밥도둑이네", [I("jjageuli", 1, "인분")], noise=["잡담"])
C("선지해장국 한 뚝배기 클리어", [I("seonji", 1, "뚝배기", units_ok=["뚝배기", "그릇", "개"])], noise=["슬랭"])

# ======================= 면 =======================
C("냉모밀 한 판 먹고 옴", [I("soba", 1, "판", units_ok=["판", "그릇", "인분", "개"])])
C("팟타이 하나 포장해옴", [I("pad_thai", 1, "개", units_ok=["개", "접시", "인분"])])
C("까르보나라 파스타 1인분 먹었어요", [I("carbonara", 1, "인분")])
C("돈코츠라멘 한 그릇 국물은 반만", [I("tonkotsu", 1, "그릇", grams=(450, 650))], conf="medium",
  note="국물 절반 남김 -> 600-800g 중 450-650g")
C("야끼우동 한 접시", [I("yaki_udon", 1, "접시", units_ok=["접시", "인분", "그릇", "개"])])
C("pad thai one plate for dinner", [I("pad_thai", 1, "접시", units_ok=["접시", "plate", "개", "one"])], noise=["영어"])

# ======================= 분식 / 빵 / 디저트 =======================
C("로제떡볶이 1인분 배달시킴", [I("rose_tteok", 1, "인분")])
C("군만두 6개 에어프라이어에 구움", [I("gun_mandu", 6, "개")], conf="medium")
C("길거리에서 닭꼬치 두 개", [I("dak_kkochi", 2, "개")])
C("붕어빵 3마리 사먹음 ㅎㅎ", [I("bungeoppang", 3, "마리", units_ok=["마리", "개"])], noise=["슬랭"])
C("출근길에 계란빵 하나", [I("gyeranppang", 1, "개")], noise=["끼니"])
C("had 2 croquettes and a cappuccino", [I("croquette", 2, "개", units_ok=["개", "croquettes"]),
                                         I("cappuccino", 1, "잔", units_ok=["잔", "a", "개"])], noise=["영어"])
C("소금빵 하나랑 아아", [I("salt_bread", 1, "개"), I("americano")], noise=["슬랭"])
C("단팥빵 한 개에 우유 한 팩", [I("danpat", 1, "개"), I("milk", 1, "팩")])
C("카페에서 크로플 두 개 먹음", [I("croffle", 2, "개")], conf="medium")
C("티라미수 한 조각 ☕", [I("tiramisu", 1, "조각")], noise=["이모지"])
C("카스테라 두 조각이랑 우유", [I("castella", 2, "조각"), I("milk")])
C("약과 3개 집어먹음..", [I("yakgwa", 3, "개")], conf="medium")
C("백설기 한 덩이", [I("baekseolgi", 1, "덩이", units_ok=["덩이", "개", "조각"])])
C("고구마말랭이 60g 🍠", [I("dried_sp", 60, "g")], noise=["이모지"])

# ======================= 고기 / 요리 =======================
C("춘천닭갈비 2인분 혼자 다 먹음 ㅋㅋ", [I("dakgalbi", 2, "인분")], noise=["슬랭"])
C("안동찜닭 반마리 시켜서 혼자", [I("jjimdak", 0.5, "마리")], tags=["unit:반"], conf="medium", note="1마리 900-1300g의 절반")
C("낙지볶음 1인분에 밥 한 공기", [I("nakji_bokkeum", 1, "인분"), I("rice", 1, "공기")])
C("쭈꾸미볶음 200그램", [I("jukkumi", 200, "g")])
C("오리로스 1인분 구워먹었어", [I("duck_roast", 1, "인분")], conf="medium")
C("항정살 300g 구워 먹었다", [I("hangjeong", 300, "g")])
C("갈매기살 2인분", [I("galmaegi", 2, "인분")])
C("돼지껍데기 한 접시에 소주 반병", [I("pork_skin", 1, "접시"), I("soju", 0.5, "병")])
C("소대창 1인분 기름 장난아님", [I("daechang", 1, "인분")], noise=["잡담"], conf="medium")
C("깐풍기 한 접시", [I("kkanpunggi", 1, "접시")], conf="medium")
C("함박스테이크 정식 (밥 포함)", [I("hamburg"), I("rice", optional=True)], conf="medium", note="정식 = 함박 1인분 + 밥(선택)")
C("마파두부 한 접시 덜어먹음", [I("mapo", 1, "접시")], conf="medium")
C("해물찜 1인분 정도 먹은듯", [I("haemul_jjim", 1, "인분")], conf="medium")

# ======================= 생선 =======================
C("조기구이 두 마리 발라먹음", [I("jogi", 2, "마리")])
C("연어구이 한 토막 + 밥 반공기", [I("salmon_gui", 1, "토막"), I("rice", 0.5, "공기")], noise=["띄어쓰기"])

# ======================= 과일 / 채소 / 반찬 =======================
C("자몽 하나 까먹음", [I("grapefruit", 1, "개")])
C("석류 반 개", [I("pomegranate", 0.5, "개")])
C("무화과 4개 먹음", [I("fig", 4, "개")])
C("용과 한 개 다 먹었다", [I("dragonfruit", 1, "개")])
C("리치 열 알", [I("lychee", 10, "알", units_ok=["알", "개"])])
C("당근 1개 스틱으로 잘라서 씹어먹음", [I("carrot", 1, "개")])
C("빨간 파프리카 한 개 생으로", [I("paprika", 1, "개")])
C("집밥: 밥 한 공기, 두부조림, 멸치볶음, 김치", [I("rice", 1, "공기"), I("dubu_jorim"), I("myeolchi"), I("kimchi")], conf="medium")
C("도시락 반찬으로 진미채랑 감자조림", [I("jinmichae"), I("gamja_jorim")], conf="medium")

# ======================= 음료 =======================
C("따뜻한 카푸치노 한 잔", [I("cappuccino", 1, "잔")])
C("아인슈패너 아이스로 1잔", [I("einspanner", 1, "잔")], noise=["오타"])
C("플랫화이트 한잔 테이크아웃", [I("flat_white", 1, "잔")], noise=["띄어쓰기"])
C("레몬에이드 큰 사이즈", [I("lemonade", grams=(600, 710))], tags=["unit:크기", SIZE], conf="medium", note="대형 컵 600-710ml")
C("자몽에이드 한잔 시원하다", [I("grapefruit_ade", 1, "잔")], noise=["잡담"])
C("자몽에이드 라지 사이즈로 하나 테이크아웃", [I("grapefruit_ade", grams=(600, 710))], tags=["unit:크기", SIZE], conf="medium",
  note="라지 컵 600-710ml")
C("유자차 핫으로 한 잔", [I("yuja", 1, "잔")])
C("식혜 한 컵", [I("sikhye", 1, "컵")])
C("토마토 주스 한 잔이랑 삶은 계란 2개", [I("tomato_juice", 1, "잔"), I("egg", 2, "개")])

# ======================= 프랜차이즈 =======================
C("본죽 전복죽 한 그릇", [I("abalone_juk_bon", 1, "그릇")], tags=[PROD])
C("맥치킨 단품 하나", [I("mcchicken", 1, "개")], tags=[PROD], conf="medium")
C("1955버거 세트 먹음", [I("burger_1955"), I("fries"), I("cola")], tags=[PROD], conf="medium",
  note="세트 = 버거 + 감자튀김 M + 콜라 M", min_items=1)
C("버거킹 와퍼 쥬니어 2개", [I("whopper_jr", 2, "개")], tags=[PROD], noise=["오타"], conf="medium")
C("KFC 징거버거 하나", [I("zinger", 1, "개")], tags=[PROD], conf="medium")
C("롯데리아 데리버거 1개", [I("teri_burger", 1, "개")], tags=[PROD], conf="medium")
C("bhc 고추바사삭 2조각만", [I("gochu_basasak", 2, "조각")], tags=[PROD], conf="medium")
C("BBQ 황올 반마리 혼자 순삭", [I("golden_olive", 0.5, "마리")], tags=[PROD], noise=["슬랭"], conf="low",
  note="1마리 800-1100g의 절반")
C("스벅 자몽허니블랙티 톨", [I("grapefruit_blacktea", grams=(345, 365))], tags=[PROD, "unit:크기", SIZE], noise=["슬랭"])

# ======================= 편의점 / 브랜드 제품 =======================
C("사리곰탕면 한 봉에 계란 하나 풀어서", [I("sariguk", 1, "봉"), I("egg", 1, "개")], tags=[PROD])
C("야식으로 진비빔면 1개", [I("jin_bibim", 1, "개")], tags=[PROD], noise=["끼니"])
C("배홍동비빔면 두 개 비벼먹음", [I("baehongdong", 2, "개")], tags=[PROD])
C("양반 전복죽 하나 데워먹음", [I("yangban_juk", 1, "개")], tags=[PROD])
C("3분짜장 하나 밥에 부어먹음", [I("3min_jjajang", 1, "개", units_ok=["개", "봉지"]), I("rice")], tags=[PROD])
C("스낵면 하나 끓이고 찬밥 말아먹음", [I("snack_myeon", 1, "개"), I("rice")], tags=[PROD])
C("양파링 한 봉지 혼자 해치움", [I("yangparing", 1, "봉지")], tags=[PROD], noise=["슬랭"])
C("탱크보이 하나 쪽쪽", [I("tankboy", 1, "개")], tags=[PROD], noise=["슬랭"])
C("데자와 한 캔 마심", [I("dejawa", 1, "캔")], tags=[PROD])
C("스프라이트 한 캔 원샷", [I("sprite", 1, "캔")], tags=[PROD], noise=["슬랭"])
C("레드불 1캔 마시고 야근", [I("redbull", 1, "캔")], tags=[PROD], noise=["잡담"])
C("숙취엔 갈아만든배 한 캔", [I("galaman_bae", 1, "캔")], tags=[PROD], noise=["잡담"])
C("비락식혜 1캔", [I("bilak_sikhye", 1, "캔")], tags=[PROD])
C("웰치스 딸기 한 캔", [I("welchs", 1, "캔")], tags=[PROD])
C("설레임 밀크쉐이크 한 개 쭉쭉", [I("seolleim", 1, "개")], tags=[PROD])
C("불가리스 사과맛 한 병", [I("bulgaris", 1, "병")], tags=[PROD])
C("서울우유 커피 한 팩 쪽쪽", [I("seoul_coffee", 1, "팩")], tags=[PROD], noise=["슬랭"])
C("맛있는우유GT 한 팩", [I("gt_milk", 1, "팩")], tags=[PROD])

# ======================= 주류 =======================
C("부산 와서 대선 한 병 깠다", [I("daeseon", 1, "병")], tags=[PROD], noise=["잡담"])
C("순하리 자몽진 한 캔", [I("jamongjin", 1, "캔")], tags=[PROD], conf="medium", note="캔 355/500ml 모두 허용")
C("새로 다래 반 병", [I("saero_darae", 0.5, "병")], tags=[PROD])
C("비 오는 날 월매 한 병 땡김", [I("wolmae", 1, "병")], tags=[PROD], noise=["슬랭"])
C("달빛유자 반 병 마심", [I("dalbit_yuja", 0.5, "병")], tags=[PROD])
C("청하 한 병", [I("cheongha", 1, "병")], tags=[PROD], conf="medium")
C("중국집에서 고량주 세 잔", [I("goryangju", 3, "잔")], tags=[PROD], conf="low", note="잔 크기(20-50ml)·도수(34-50%) 불확실")
C("복분자주 두 잔", [I("bokbunja", 2, "잔")], tags=[PROD], conf="medium")

# ======================= 까다로운 파싱: 나눠먹기 / 부분 / 범위 / 부정 =======================
C("닭갈비 3인분 시켜서 둘이 먹음", [I("dakgalbi", (1.5, 3), "인분", grams=(375, 525))], tags=[SHARE], note="3인분/2명 = 1.5인분")
C("아구찜 중자 셋이서 먹음", [I("agujjim", grams=(330, 550), low=True)], tags=[SHARE], conf="low",
  note="중자 1.0-1.6kg / 3명")
C("팥빙수 하나 시켜서 둘이 나눠먹음", [I("patbingsu", (0.5, 1), "개", grams=(200, 300))], tags=[SHARE], conf="medium",
  note="빙수 1개 400-600g / 2명")
C("대선 2병 셋이서 나눠마심", [I("daeseon", (0.67, 2), "병", grams=(240, 240))], tags=[SHARE, PROD], note="720ml / 3")
C("깐풍기 작은 거 하나 둘이서 먹음", [I("kkanpunggi", grams=(150, 250))], tags=[SHARE], conf="medium", note="소자 300-500g / 2")
C("넷이서 갈매기 2인분, 껍데기 1인분 시킴", [I("galmaegi", (0.5, 2), "인분", grams=(75, 100)),
                                     I("pork_skin", (0.25, 1), "인분", grams=(38, 63))], tags=[SHARE], conf="medium",
  note="갈매기 300-400g/4, 껍데기 150-250g/4")
C("로제떡볶이 2인분 시켰는데 친구가 조금만 먹어서 내가 1인분 반 정도", [I("rose_tteok", (1.5, 2), "인분", grams=(450, 675), low=True)],
  tags=[SHARE], conf="low", note="본인 1.5인분")
C("피스타치오 한두 줌", [I("pistachio", (1, 2), "줌")], tags=[RANGE], conf="medium")
C("추석이라 송편 대여섯 개", [I("songpyeon", (5, 6), "개")], tags=[RANGE], noise=["잡담"])
C("꿀떡 7~8개 집어먹음", [I("kkuldeok", (7, 8), "개")], tags=[RANGE])
C("야채고로케 두세 개 사서 먹음", [I("croquette", (2, 3), "개")], tags=[RANGE])
C("진라면 하나 끓였는데 반은 남김", [I("jin_ramen", (0.5, 1), "개", grams=(60, 60))], tags=[PART, PROD], note="1봉 120g 중 절반")
C("단팥빵 반 개만 먹고 나머지는 내일 먹을 거", [I("danpat", 0.5, "개")], tags=[PART], noise=["잡담"])
C("소금빵 3개 사서 하나는 동생 줌", [I("salt_bread", (2, 3), "개", grams=(120, 180))], tags=[PART], note="본인 2개")
C("더위사냥 반 갈라서 반만 먹음", [I("deowi", (0.5, 1), "개", grams=(70, 70))], tags=[PART, PROD], note="140ml 중 절반")
C("월매 1.5병 마심", [I("wolmae", 1.5, "병")], tags=[PROD, "unit:분수"])
C("치킨 시키려다 참고 자몽 하나만 먹음", [I("grapefruit", 1, "개")], tags=[NEG], noise=["부정"])
C("술은 안 마시고 오징어볶음에 밥만 먹음", [I("ojingeo_bokkeum"), I("rice")], tags=[NEG], noise=["부정"])
C("라떼 말고 아인슈페너 마심 (라떼는 안 마심)", [I("einspanner")], tags=[NEG], noise=["부정"])
C("라면 끓이려다 그냥 굶음", [], tags=["flag:invalid", NEG], noise=["부정"], max_items=0)


# ---------------------------------------------------------------------------
# Build (helpers as in build_diet_holdout_v1.py)
# ---------------------------------------------------------------------------
_SYN_GROUPS = [
    {"개", "알", "ea", "pc", "피스"}, {"공기", "그릇"}, {"그릇", "인분", "접시", "대접"}, {"잔", "컵"},
    {"봉", "봉지", "개"}, {"조각", "쪽", "피스", "개"}, {"장", "쪽", "개", "slice"}, {"캔", "개"}, {"병", "개"},
    {"팩", "개"}, {"줄", "개"}, {"토막", "조각", "개"}, {"스쿱", "scoop", "숟가락"}, {"L", "l", "리터"},
    {"g", "그램", "gram", "grams"}, {"ml", "mL", "cc", "밀리리터"}, {"통", "개"}, {"갑", "곽", "개"},
    {"뚝배기", "그릇"}, {"대", "개"},
]
_HANGUL_NUM_RE = re.compile(r"(?:(?:한|두|세|네|다섯|여섯|일곱|여덟|아홉|열|서너|두세|대여섯|한두|이백)\s*(?:개|공기|그릇|잔|캔|병|조각|줄|장|마리|인분|봉지|팩|컵|스쿱|토막|송이|모|알|줌|갑|통|판|덩이|접시|쪽|봉|대|대접|뚝배기|미리|시|스쿱)(?![가-힣])|(?:한|두|세|네|다섯)\s*(?:개|공기|그릇|잔|캔|병|조각|줄|장|마리|인분|봉지|팩|컵|토막|송이|모|알|줌|갑|통|판|접시|쪽|봉|대|대접|뚝배기)|하나|둘이|셋이|넷이)")
UNIT_TAGS = {"g": "g", "ml": "ml", "cc": "ml", "L": "ml", "개": "개", "알": "개", "조각": "조각", "토막": "조각",
             "그릇": "그릇", "접시": "그릇", "대접": "그릇", "뚝배기": "그릇", "공기": "공기", "인분": "인분", "캔": "캔",
             "병": "병", "잔": "잔", "컵": "컵", "스쿱": "스쿱", "줄": "줄", "봉": "봉지", "봉지": "봉지", "팩": "팩",
             "마리": "마리", "모": "모", "송이": "송이", "줌": "줌", "갑": "갑", "통": "통", "판": "판", "대": "개", "세트": "세트",
             "덩이": "개"}


def _units_ok(unit, explicit):
    if explicit is not None:
        return list(explicit) if explicit else ([unit] if unit else [])
    if unit is None:
        return []
    out = {unit}
    for grp in _SYN_GROUPS:
        if unit in grp:
            out |= grp
    return sorted(out)


def _compact(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    return re.sub(r"[\s\W_]+", "", s)


def _bigrams(s: str) -> set:
    c = _compact(s)
    return {c[i:i + 2] for i in range(len(c) - 1)} or {c}


def _jac(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def _check_mfds():
    """Stream the raw MFDS CSV once and assert every transcribed (FOOD_CD, kcal, name) row."""
    want = {}
    for key, f in R.items():
        if f["kind"] == "product":
            for code, k, nm in f["mfds"]:
                want[code] = (key, float(k), nm)
    seen = {}
    with MFDS_PATH.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["FOOD_CD"] in want:
                seen[row["FOOD_CD"]] = row
    for code, (key, k, nm) in want.items():
        assert code in seen, f"{key}: MFDS code {code} not found"
        row = seen[code]
        assert float(row["ENERC"]) == k, f"{key}: {code} ENERC {row['ENERC']} != {k}"
        assert row["FOOD_NM"].strip() == nm, f"{key}: {code} name {row['FOOD_NM']!r} != {nm!r}"
        assert row["NUT_CON_SRTR_QUA"] in ("100g", "100ml"), f"{key}: {code} basis {row['NUT_CON_SRTR_QUA']}"
    return len(want)


def _densities(f, rows, cid):
    dens, parts = [], []
    if f["kind"] == "product":
        for code, k, nm in f["mfds"]:
            dens.append(float(k))
            parts.append(f"MFDS {code}[{nm}] {k}kcal/100")
        return dens, parts
    for gid in f["gold"]:
        if gid in (CUR, LLM):
            continue
        assert gid in rows, f"{cid}: unknown food_id {gid}"
        k = float(rows[gid]["kcal"])
        dens.append(k)
        parts.append(f"{gid}[{rows[gid]['name']}] {k:g}kcal/100{rows[gid]['basis_unit']}")
    if not dens and f["est"] is not None:
        dens += list(f["est"])
        parts.append(f"label {f['est'][0]}-{f['est'][1]}kcal/100" + (f" ({f['est_basis']})" if f["est_basis"] else ""))
    elif f["est"] is not None:
        dens += list(f["est"])
        parts.append(f"public {f['est'][0]}-{f['est'][1]}kcal/100 ({f['est_basis']})")
    return dens, parts


def _scratch_inputs() -> list[str]:
    src = (ROOT / "scripts" / "eval" / "build_diet_eval_v1.py").read_text(encoding="utf-8")
    block = src[src.index("SCRATCH_INPUTS = {"):]
    block = block[:block.index("}") + 1]
    return re.findall(r'"([^"]+)"', block)


def _category(tags: set) -> str:
    if tags & {SHARE, RANGE, PART, NEG}:
        return "tricky"
    if PROD in tags:
        return "product"
    return "generic"


def main():
    rows = {r["food_id"]: r for r in csv.DictReader(CSV_PATH.open(encoding="utf-8-sig"))}
    n_mfds = _check_mfds()
    existing, ex_rows = [], []
    for p in EXISTING:
        for line in p.open(encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                existing.append(d["text"])
                ex_rows.append(d)
    scratch = _scratch_inputs()
    assert len(scratch) == 12, scratch
    existing += scratch
    ex_compact = {_compact(t): t for t in existing}
    ex_bigrams = [(t, _bigrams(t)) for t in existing]
    template = json.loads(TEMPLATE.open(encoding="utf-8").readline())
    case_keys, item_keys = set(template), set(template["expected_items"][0])

    # existing item vocabulary (for the new-key measurement)
    ex_keys, ex_gold, ex_names = set(), {}, set()
    for d in ex_rows:
        for it in d.get("expected_items", []):
            ex_keys.add(it["key"])
            for g in it.get("gold_food_ids") or []:
                if g not in (CUR, LLM):
                    ex_gold.setdefault(g, set()).add(it["key"])
            ex_names |= {_compact(n) for n in it.get("names") or []}

    # ---- drop near duplicates BEFORE assigning ids ----
    kept, dropped, kept_bg = [], [], []
    for spec in CASES:
        text = spec["text"]
        c = _compact(text)
        bg = _bigrams(text)
        why = None
        if c in ex_compact:
            why = f"exact dup of existing {ex_compact[c]!r}"
        else:
            for t, b in ex_bigrams:
                j = _jac(bg, b)
                if j >= NEAR_JACCARD:
                    why = f"jaccard {j:.2f} vs existing {t!r}"
                    break
        if why is None:
            for t, b in kept_bg:
                j = _jac(bg, b)
                if j >= NEAR_JACCARD:
                    why = f"jaccard {j:.2f} vs earlier v2 {t!r}"
                    break
        if why:
            dropped.append((text, why))
            continue
        kept.append(spec)
        kept_bg.append((text, bg))

    out = []
    for n, spec in enumerate(kept, 1):
        cid = f"dh2_{n:03d}"
        text = spec["text"]
        items_out, tags = [], set()
        for it in spec["items"]:
            f = R[it["key"]]
            tags.add(f"group:{f['group']}")
            amt, unit = it["amt"], it["unit"]
            if amt is None:
                amount = None
                tags.add("unit:숫자없음")
            elif isinstance(amt, tuple):
                amount = [round(float(amt[0]), 4), round(float(amt[1]), 4)]
                tags.add("unit:범위")
            else:
                amount = [round(float(amt), 4), round(float(amt), 4)]
                if float(amt) < 1:
                    tags.add("unit:반" if abs(float(amt) - 0.5) < 1e-9 else "unit:분수")
                elif float(amt) != int(float(amt)):
                    tags.add("unit:분수")
            if unit:
                tags.add(f"unit:{UNIT_TAGS.get(unit, unit)}")
            # grams
            if it["grams"] is not None:
                g_lo, g_hi = it["grams"]
                g_basis = f"{g_lo}-{g_hi}g (case-specific)"
            elif amt is None:
                g_lo, g_hi = f["serving"]
                g_basis = f"1회 제공량/1개 {g_lo}-{g_hi}g"
            elif isinstance(amt, tuple):
                # explicit range ("두세 개"); sharing/partial tuples always pass grams
                assert RANGE in spec["extra_tags"], f"{cid}: tuple amount without grams must be a range case"
                a_lo, a_hi = float(amt[0]), float(amt[1])
                per = f["units"][unit]
                g_lo, g_hi = a_lo * per[0], a_hi * per[1]
                g_basis = f"{a_lo:g}-{a_hi:g}{unit} x {per[0]}-{per[1]}g"
            else:
                a = float(amt)
                if unit in ("g", "ml", "cc"):
                    g_lo, g_hi = a * 0.98, a * 1.02
                    g_basis = f"{a:g}{unit}"
                else:
                    per = f["units"][unit]
                    g_lo, g_hi = a * per[0], a * per[1]
                    g_basis = f"{a:g}{unit} x {per[0]}-{per[1]}g"
            g_lo, g_hi = round(g_lo, 1), round(g_hi, 1)
            dens, parts = _densities(f, rows, cid)
            if it["key"] in EXTRA_DENS:
                dens += list(EXTRA_DENS[it["key"]])
            assert dens, f"{cid}: no kcal basis for {it['key']}"
            if f["kind"] == "product":
                k_lo = min(dens) * g_lo / 100 * (1 - KCAL_TOL)
                k_hi = max(dens) * g_hi / 100 * (1 + KCAL_TOL)
                basis = f"{' | '.join(parts)} x {g_basis} (label +-10%)"
            else:
                k_lo = max(0.0, min(dens) * g_lo / 100 * (1 - KCAL_TOL) - KCAL_ABS)
                k_hi = max(dens) * g_hi / 100 * (1 + KCAL_TOL) + KCAL_ABS
                basis = f"{' | '.join(parts)} x {g_basis} (+-10%, +-5kcal)"
            if f.get("note"):
                basis += f"; {f['note']}"
            if it.get("note"):
                basis += f"; {it['note']}"
            item = {
                "key": it["key"],
                "names": f["names"],
                "gold_food_ids": f["gold"],
                "amount": amount,
                "units": _units_ok(unit, it["units_ok"]),
                "grams": [g_lo, g_hi],
                "kcal": [round(k_lo, 1), round(k_hi, 1)],
                "basis": basis,
                "optional": bool(it["optional"]),
                "low_discrimination": bool(it["low"] or (k_lo > 0 and k_hi / k_lo > 3.0)),
            }
            assert set(item) == item_keys, f"{cid}: item keys {set(item) ^ item_keys}"
            items_out.append(item)
        req = [i for i in items_out if not i["optional"]]
        n_req = len(req)
        tags.add("items:" + ("0" if n_req == 0 else "1" if n_req == 1 else "2" if n_req == 2 else "3+"))
        for nz in spec["noise"]:
            tags.add(f"noise:{nz}")
        if not spec["noise"]:
            tags.add("noise:없음")
        if _HANGUL_NUM_RE.search(text):
            tags.add("unit:한글숫자")
        for t in spec["extra_tags"]:
            tags.add(t if ":" in t else f"flag:{t}")
        tags.add(SET_TAG)
        assert "set:holdout" not in tags
        total_lo = round(sum(i["kcal"][0] for i in req), 1)
        total_hi = round(sum(i["kcal"][1] for i in items_out), 1)
        row = {
            "id": cid,
            "text": text,
            "split": "test",
            "tags": sorted(tags),
            "expected_items": items_out,
            "expected_item_count": {
                "min": spec["min_items"] if spec["min_items"] is not None else n_req,
                "max": spec["max_items"] if spec["max_items"] is not None else len(items_out),
            },
            "total_kcal": [total_lo, total_hi],
            "label_confidence": spec["conf"],
            "note": spec["note"],
        }
        if row["label_confidence"] == "high" and any(i["low_discrimination"] and i["kcal"][1] >= 100 for i in req):
            row["label_confidence"] = "medium"
        assert set(row) == case_keys, f"{cid}: case keys {set(row) ^ case_keys}"
        out.append(row)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- summary ----
    def is_new(key):
        if key in ex_keys:
            return False
        f = R[key]
        if f["kind"] == "generic":
            return not any(g in ex_gold for g in f["gold"] if g not in (CUR, LLM))
        return _compact(f["names"][0]) not in ex_names

    tg = Counter(t for r in out for t in r["tags"])
    cf = Counter(r["label_confidence"] for r in out)
    cat = Counter(_category(set(r["tags"])) for r in out)
    ld = sum(1 for r in out for i in r["expected_items"] if i["low_discrimination"])
    mains = [next(i["key"] for i in r["expected_items"] if not i["optional"]) for r in out if r["expected_items"]]
    items_all = [i["key"] for r in out for i in r["expected_items"]]
    new_main = sum(is_new(k) for k in mains)
    new_items = sum(is_new(k) for k in items_all)
    uniq = sorted(set(items_all))
    renamed = [(k, sorted({ek for g in R[k]["gold"] for ek in ex_gold.get(g, ())})) for k in uniq
               if k not in ex_keys and R[k]["kind"] == "generic" and not is_new(k)]
    renamed += [(k, "names[0] in existing names") for k in uniq
                if k not in ex_keys and R[k]["kind"] == "product" and not is_new(k)]
    llm_items = sum(1 for r in out for i in r["expected_items"] if LLM in i["gold_food_ids"])
    llm_generic = sum(1 for r in out for i in r["expected_items"] if LLM in i["gold_food_ids"] and R[i["key"]]["kind"] == "generic")
    print(f"MFDS rows asserted against raw CSV: {n_mfds}")
    print(f"written {len(CASES)} specs, dropped {len(dropped)} near-duplicates:")
    for t, why in dropped:
        print(f"    DROP {t!r}: {why}")
    print(f"wrote {len(out)} cases -> {OUT_PATH.relative_to(ROOT)}")
    print(f"category: {dict(cat)}  ({', '.join(f'{k} {v / len(out):.0%}' for k, v in cat.items())})")
    print(f"label_confidence: {dict(cf)}  low_discrimination items: {ld}")
    print(f"low-confidence ids: {[r['id'] for r in out if r['label_confidence'] == 'low']}")
    print(f"main-item keys new vs existing sets: {new_main}/{len(mains)} = {new_main / max(1, len(mains)):.0%}")
    print(f"all items new: {new_items}/{len(items_all)} = {new_items / max(1, len(items_all)):.0%}; unique keys {len(uniq)}")
    print(f"new-named keys that overlap existing gold/names (not counted new): {renamed}")
    print(f"items carrying llm_estimate_ok: {llm_items} (product rows, token kept per v1 convention) ; generic LLM-only: {llm_generic}")
    for t, c in sorted(tg.items()):
        print(f"  {t}: {c}")
    print("sha256:", hashlib.sha256(OUT_PATH.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()

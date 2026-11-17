"""Build tests/evaluation/datasets/diet_eval_holdout_v5.jsonl (fifth blind held-out Korean diet-log set).

Purpose: a HELD-OUT set written blind to the pipeline's outputs, to measure generalisation of a parser that was tuned on
diet_eval_v1 / diet_eval_products_v1 / diet_eval_holdout_v1..v4. Every case is split "test", tagged set:holdout_v5 plus
persona:<x>, style:<y> and domain:<z> (domain of the main = first required item).

v5 rotates the GENERATOR STYLE again (different from v4's beginner/lifter/office/elderly/teen x telegraphic/diary/typo/
mixed/voice):
  personas (even spread): pregnant (pregnant / postpartum), diabetic (counts carbs, blood sugar), nightshift (night-shift
                          worker, odd hours), vegan (vegetarian / vegan), student (college student living alone),
                          parent (logs a family meal -- only MY share)
  styles (even spread):   kakao (KakaoTalk chat: ㅇㅇ ㄱㄱ ㅋㅋ ㄹㅇ), list ("아침/ 토스트, 라떼"), oddnum (odd number / unit
                          spacing: "밥2/3공기", "1개반", "0.5인분"), honorific (polite diary -습니다), english (English only),
                          emoji (emoji-heavy)
  domains (main item):    cafeteria (hospital / office cafeteria trays), school_meal, holiday, regional,
                          sea_me (Southeast Asian / Middle Eastern foods sold in Korea), cvs_new (new convenience-store
                          items), keto (low-carb / keto), plant (plant-based); staple only for side items

Labels describe what is CORRECT for a real person's meal, never what the code returns. Label sources (only these):
  - data/food_db/food_db_clean.csv          (generic foods: gold_food_ids + kcal/100 of those rows)
  - data/food_db/raw/식품의약품안전처_통합식품영양성분정보(가공식품)_20261111.csv
                                             (branded / packaged products: label values, asserted at build)
  - public common-knowledge Korean portion sizes (stated in each registry note / basis)
engines/*, data/curated/*, processed_product_index.csv, LLM caches, baselines and eval reports were NOT consulted.

Schema = diet_eval_holdout_v4.jsonl (same conventions as build_diet_holdout_v4.py):
  - names: acceptable food names (incl. English aliases used in english-style cases, chat abbreviations like 두쫀쿠/삼김)
  - gold_food_ids: acceptable food_db_clean rows (+ "curated" / "llm_estimate_ok" tokens)
  - amount: literal quantity [lo, hi] (null = no number). Sharing / partial-eating: [per-person share, literal total];
    explicit ranges ("2~3개"): [lo, hi] of the stated range.
  - units: accepted unit tokens; [] when the text has a bare number without a unit word ("치킨케밥라이스1")
  - grams [lo, hi]: public-standard portion range (ml treated as g); always the amount the person actually ate
  - kcal [lo, hi]: generic = min/max DB kcal/100 x grams, widened 10% and +-5 kcal;
                   product = min/max MFDS label kcal/100 x single-unit size x quantity, widened 10%
Completeness (asserted at build): every food_db row whose name is IDENTICAL to a gold row's name, and every MFDS row
whose FOOD_NM is identical to a transcribed product name, is either in gold or listed in the key's `excl` dict with a
reason (the reason is copied into basis). Same-food rows with different wording (g/ml twins, reversed word order such as
"근대 된장국" / "된장국 근대", synonyms such as 호박전 / 애호박전) were added by hand; rows whose token set equals a gold
row's token set are additionally PRINTED for review.
Numbers that are NOT amounts (임신 20주, 새벽 3시, 50KCAL 곤약밥, "탄수 45g") are deliberate traps; those cases are medium.
Near-duplicate texts (bigram Jaccard >= 0.4 vs the six existing sets, the scratch inputs, or an earlier v5 case) are
DROPPED before ids are assigned.

Run:  PYTHONUTF8=1 python scripts/eval/build_diet_holdout_v5.py
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
OUT_PATH = DATASETS / "diet_eval_holdout_v5.jsonl"
EXISTING = [DATASETS / "diet_eval_v1.jsonl", DATASETS / "diet_eval_products_v1.jsonl",
            DATASETS / "diet_eval_holdout_v1.jsonl", DATASETS / "diet_eval_holdout_v2.jsonl",
            DATASETS / "diet_eval_holdout_v3.jsonl", DATASETS / "diet_eval_holdout_v4.jsonl"]
TEMPLATE = DATASETS / "diet_eval_holdout_v4.jsonl"

KCAL_TOL = 0.10
NEAR_JACCARD = 0.4  # cases at or above this vs existing / earlier v5 texts are dropped
KCAL_ABS = 5.0
LLM = "llm_estimate_ok"
CUR = "curated"
G_ML = {"g": (1, 1), "ml": (1, 1)}
SET_TAG = "set:holdout_v5"
ID_PREFIX = "dh5_"
N_RANGE = (120, 140)   # asserted size of the written set
MIN_CELL = 3           # asserted minimum cases per persona x style cell

PERSONAS = ("pregnant", "diabetic", "nightshift", "vegan", "student", "parent")
STYLES = ("kakao", "list", "oddnum", "honorific", "english", "emoji")
DOMAINS = ("staple", "cafeteria", "school_meal", "holiday", "regional", "sea_me", "cvs_new", "keto", "plant")
STYLE_NOISE = {"kakao": ["초성"], "list": ["구분자"], "oddnum": ["숫자단위"], "honorific": [], "english": ["영어"],
               "emoji": ["이모지"]}


# ---------------------------------------------------------------------------
# Registry.  units: grams per ONE unit (lo, hi); serving: portion when no number is given.
#   F(...)  generic food: gold = DB ids (+ optional tokens); excl = {id: reason} for same-name rows left out
#   P(...)  branded product: mfds = [(FOOD_CD, kcal_per_100, "exact FOOD_NM")], units = single-unit size
# ---------------------------------------------------------------------------
def F(group, domain, names, gold, units=None, serving=None, note=None, excl=None):
    assert domain in DOMAINS, domain
    return dict(kind="generic", group=group, domain=domain, names=names, gold=gold, units=units or {}, serving=serving,
                est=None, est_basis=None, note=note, excl=excl or {})


def P(group, domain, names, mfds, units, serving, note, gold=(LLM, CUR), excl=None):
    assert domain in DOMAINS, domain
    return dict(kind="product", group=group, domain=domain, names=names, gold=list(gold), units=units, serving=serving,
                mfds=mfds, note=note, excl=excl or {})


BAP = {"공기": (190, 230), "그릇": (190, 230), "bowl": (190, 230), **G_ML}
BAP_NOTE = "밥 1공기 190-230g(식약처 1회 제공량 210g)"
GUK = {"그릇": (250, 350), "bowl": (250, 350), **G_ML}
GUK_NOTE = "국 1그릇 250-350g"
NAMUL = {"접시": (60, 100), **G_ML}
NAMUL_NOTE = "나물·볶음·무침 반찬 1회 40-70g, 1접시 60-100g"
JORIM_NOTE = "조림 반찬 1회 30-60g"
KIMCHI_NOTE = "김치 반찬 1회 30-60g"

R = {
    # ---------------- reused staples (definitions copied from build_diet_holdout_v4.py; side items only) ----------------
    "rice": F("밥", "staple", ["쌀밥", "밥", "공기밥", "공깃밥", "흰쌀밥", "흰밥", "rice"], ["음식_011600", "음식_011599", CUR],
              units={"공기": (190, 230), "그릇": (190, 230), "개": (190, 230), "bowl": (190, 230), **G_ML}, serving=(190, 230),
              note="밥 1공기 210g(식약처 1회 제공량)"),
    "egg": F("계란유제품", "staple", ["계란", "달걀", "삶은계란", "삶은 계란", "egg", "eggs", "boiled eggs"],
             [CUR, "원재_001702", "음식_017044", "원재_001700"], units={"개": (50, 60), "알": (50, 60)}, serving=(50, 60)),
    "americano": F("음료", "staple", ["아메리카노", "아아", "아이스 아메리카노", "아이스아메리카노", "커피", "americano", "iced americano"],
                   ["음식_009004", "음식_008945", "음식_008940", "음식_009005", "음식_008946"], units={"잔": (300, 500)}, serving=(300, 500)),
    "cola": P("음료", "staple", ["콜라", "코카콜라", "coke"],
              [("P109-401040100-0479", 43, "코카콜라"), ("P109-401040100-2381", 44, "코카콜라"), ("P109-401040100-2382", 44, "코카콜라")],
              {"캔": (250, 355), "잔": (200, 300), "개": (250, 355)}, (250, 355),
              "콜라 1캔 250-355ml; P109-401040100-1982(코카-콜라 158/100ml)는 1캔값 오기로 보고 제외"),

    # ---------------- 병원·회사 구내식당 식판 (cafeteria) ----------------
    "miyeokjulgi": F("반찬", "cafeteria", ["미역줄기볶음", "미역줄기", "seaweed stem stir-fry"], ["음식_014769", "음식_014770"],
                     units=NAMUL, serving=(40, 70), note=NAMUL_NOTE + "; 재료 표기 행(음식_014767/014768)은 다른 이름"),
    "musaengchae": F("반찬", "cafeteria", ["무생채", "무채", "radish salad"], ["음식_014853", "음식_014854"], units=NAMUL,
                     serving=(40, 70), note=NAMUL_NOTE + "; 굴·미나리 등 부재료 행은 다른 이름"),
    "kkakdugi": F("반찬", "cafeteria", ["깍두기", "kkakdugi", "cubed radish kimchi"], ["음식_017213", "음식_017214"], units=G_ML,
                  serving=(30, 60), note=KIMCHI_NOTE),
    "yeongeun_jorim": F("반찬", "cafeteria", ["연근조림", "braised lotus root"], ["음식_010697", "음식_010698"], units=G_ML,
                        serving=(30, 60), note=JORIM_NOTE + "; g/ml 두 행 kcal 차이가 커서 범위가 넓음"),
    "ueong_jorim": F("반찬", "cafeteria", ["우엉조림", "braised burdock"], ["음식_010326", "음식_010327"], units=G_ML,
                     serving=(30, 60), note=JORIM_NOTE),
    "sigeumchi": F("반찬", "cafeteria", ["시금치나물", "시금치무침", "spinach namul"], ["음식_011730", "음식_011731"], units=NAMUL,
                   serving=(40, 70), note=NAMUL_NOTE + "; 양념별 시금치나물 행(음식_011725-011729)은 다른 이름"),
    "geundae_guk": F("국찌개", "cafeteria", ["근대 된장국", "근대국", "근대된장국"], ["음식_017565", "음식_016338"], units=GUK,
                     serving=(250, 350), note=GUK_NOTE + "; 어순만 다른 '된장국 근대' 행 포함"),
    "baechu_guk": F("국찌개", "cafeteria", ["배추된장국", "배추 된장국", "된장국 배추"], ["음식_016330", "음식_016331", "음식_014375"],
                    units=GUK, serving=(250, 350),
                    note=GUK_NOTE + "; '된장국 배추'·'배추 된장국' 행; 된장 없는 '배추국'(음식_014370)과 소고기·바지락 행은 다른 이름"),
    "susu_bap": F("밥", "cafeteria", ["수수밥"], ["음식_012939", "음식_012940"], units=BAP, serving=(190, 230), note=BAP_NOTE),
    "chajo_bap": F("밥", "cafeteria", ["차조밥", "foxtail millet rice"], ["음식_010014", "음식_010015"], units=BAP,
                   serving=(190, 230), note=BAP_NOTE),
    "kong_bap": F("밥", "cafeteria", ["콩밥", "검정콩밥", "black bean rice", "bean rice"], ["음식_007587", "음식_007586", "음식_007588"],
                  units=BAP, serving=(190, 230), note=BAP_NOTE + "; 검정콩·완두콩·강낭콩 콩밥 행 모두"),
    "gaji_namul": F("반찬", "cafeteria", ["가지나물", "가지무침", "sauteed eggplant", "eggplant namul"], ["음식_018035", "음식_018036"],
                    units=NAMUL, serving=(40, 70), note=NAMUL_NOTE),
    "neutari": F("반찬", "cafeteria", ["느타리버섯볶음", "버섯볶음", "stir-fried mushrooms", "sauteed mushrooms"],
                 ["음식_017063", "음식_017064", "음식_014023"], units=NAMUL, serving=(40, 70),
                 note=NAMUL_NOTE + "; '버섯볶음 느타리버섯' 행 포함"),
    "pyogo": F("반찬", "cafeteria", ["표고버섯볶음", "표고볶음"], ["음식_006891", "음식_014020"], units=NAMUL, serving=(40, 70),
               note=NAMUL_NOTE + "; '버섯볶음 표고버섯' 행 포함"),
    "gamja_bokkeum": F("반찬", "cafeteria", ["감자볶음", "감자채볶음", "stir-fried potato"], ["음식_017988", "음식_017989"],
                       units=NAMUL, serving=(40, 70), note=NAMUL_NOTE + "; 햄·베이컨 등 부재료 행은 다른 이름"),
    "osam": F("고기", "cafeteria", ["오삼불고기", "제육볶음 오징어", "오징어제육"], ["음식_010081"],
              units={"인분": (200, 300), "접시": (200, 300)}, serving=(200, 300), note="오삼불고기 1인분 200-300g"),
    "godeungeo_jjim": F("생선", "cafeteria", ["고등어찜"], ["음식_017865"], units={"토막": (80, 120), "인분": (100, 150)},
                        serving=(100, 150), note="고등어찜 1토막 80-120g, 1인분 100-150g"),
    "tofu_curry": F("밥", "cafeteria", ["두부카레", "두부 카레", "두부 카레라이스", "카레라이스 두부"], ["음식_009763"],
                    units={"그릇": (350, 450)}, serving=(350, 450), note="카레라이스 1그릇(밥+카레) 350-450g"),

    # ---------------- 학교 급식 (school_meal) ----------------
    "heukmi_bap": F("밥", "school_meal", ["흑미밥", "black rice"], ["음식_001862", "음식_001861"], units=BAP, serving=(190, 230),
                    note=BAP_NOTE + "; '흑미밥 찹쌀' 행 포함"),
    "gijang_bap": F("밥", "school_meal", ["기장밥", "millet rice"], ["음식_017560", "음식_017561", "음식_017559"], units=BAP,
                    serving=(190, 230), note=BAP_NOTE + "; '기장밥 찹쌀' 행 포함"),
    "curry_tteokbokki": F("빵과자", "school_meal", ["카레떡볶이", "카레 떡볶이", "떡볶이 카레"], ["음식_016197"],
                          units={"인분": (150, 250), "접시": (150, 250)}, serving=(150, 250), note="급식 떡볶이 1인분 150-250g"),
    "ganjang_tteokbokki": F("빵과자", "school_meal", ["간장떡볶이", "궁중떡볶이", "떡볶이 간장"], ["음식_016225"],
                            units={"인분": (150, 250), "접시": (150, 250)}, serving=(150, 250), note="급식 떡볶이 1인분 150-250g"),
    "goguma_donkatsu": F("고기", "school_meal", ["고구마돈가스", "고구마 돈까스", "고구마돈까스", "sweet potato pork cutlet"],
                         ["음식_016422"], units={"장": (100, 150), "개": (100, 150)}, serving=(100, 150),
                         note="급식 고구마돈가스 1장 100-150g"),
    "tomato_spaghetti": F("면", "school_meal", ["토마토 스파게티", "토마토스파게티", "스파게티", "spaghetti"],
                          ["음식_011754", "음식_011755", "음식_011753"], units={"접시": (250, 350), "인분": (250, 350)},
                          serving=(250, 350), note="급식 스파게티 1인분 250-350g; '스파게티 토마토소스'·'스파게티 토마토스파게티' 행"),
    "meatball": F("고기", "school_meal", ["미트볼", "미트볼조림", "meatballs"], ["음식_014756", "음식_014757"],
                  units={"개": (15, 25), "알": (15, 25)}, serving=(60, 100), note="미트볼 1개 15-25g, 반찬 1회 60-100g"),
    "sausage_bokkeum": F("고기", "school_meal", ["소시지볶음", "소세지볶음", "비엔나볶음", "sausage stir-fry"],
                         ["음식_012955", "음식_012956"], units={"접시": (80, 120), **G_ML}, serving=(50, 80),
                         note="볶음 반찬 1회 50-80g; 채소 표기 행(음식_012954)은 다른 이름"),
    "yubu_guk": F("국찌개", "school_meal", ["유부국", "유부 된장국"], ["음식_010311", "음식_016318"], units=GUK, serving=(250, 350),
                  note=GUK_NOTE + "; '된장국 유부' 행 포함"),
    "gamja_guk": F("국찌개", "school_meal", ["감자국", "감잣국", "potato soup"], ["음식_017996", "음식_017997"], units=GUK,
                   serving=(250, 350), note=GUK_NOTE + "; 소고기 감자국(음식_017995)은 다른 이름"),
    "curry_squid": F("밥", "school_meal", ["오징어카레", "오징어 카레", "카레라이스 오징어"], ["음식_009761"],
                     units={"그릇": (350, 450)}, serving=(350, 450), note="카레라이스 1그릇(밥+카레) 350-450g"),

    # ---------------- 명절 음식 (holiday) ----------------
    "toran_guk": F("국찌개", "holiday", ["토란국", "토란탕", "taro soup"], ["음식_007136", "음식_007137"], units=GUK,
                   serving=(250, 350), note=GUK_NOTE),
    "nokdu_jeon": F("빵과자", "holiday", ["녹두전", "빈대떡", "녹두빈대떡", "mung bean pancake"], ["음식_017095"],
                    units={"장": (100, 180), "개": (100, 180)}, serving=(100, 180), note="녹두전(빈대떡) 1장 100-180g"),
    "hobak_jeon": F("빵과자", "holiday", ["호박전", "애호박전", "zucchini jeon"], ["음식_001965", "음식_001966", "음식_010978", "음식_010979"],
                    units={"개": (15, 25), "piece": (15, 25), "접시": (100, 150)}, serving=(60, 100),
                    note="호박전 1개 15-25g; 동의어 '애호박전' 행 포함"),
    "gochu_jeon": F("빵과자", "holiday", ["고추전", "풋고추전"], ["음식_017853", "음식_017854"], units={"개": (20, 30)},
                    serving=(40, 90), note="고추전 1개 20-30g"),
    "saeu_jeon": F("빵과자", "holiday", ["새우전", "shrimp jeon"], ["음식_013423", "음식_013424"],
                   units={"개": (15, 25), "piece": (15, 25)}, serving=(45, 75), note="새우전 1개 15-25g"),
    "hwayangjeok": F("고기", "holiday", ["화양적", "꼬치전", "산적꼬치"], ["음식_001886", "음식_001887"],
                     units={"꼬치": (30, 50), "개": (30, 50)}, serving=(30, 50), note="화양적 1꼬치 30-50g"),
    "sanjeok": F("고기", "holiday", ["산적", "돼지고기 산적", "돼지산적"], ["음식_013459"], units={"개": (40, 70), "꼬치": (40, 70)},
                 serving=(40, 70), note="산적 1개 40-70g"),
    "dwaeji_galbijjim": F("고기", "holiday", ["돼지갈비찜", "braised pork ribs"], ["음식_016389", "음식_016390"],
                          units={"인분": (250, 350), "접시": (250, 350)}, serving=(250, 350),
                          note="돼지갈비찜 1인분(1접시) 250-350g; 채소·떡 표기 행은 다른 이름"),
    "songpyeon_bk": F("빵과자", "holiday", ["검정콩송편", "송편 검정콩", "서리태송편"], ["음식_012947"], units={"개": (20, 30)},
                      serving=(60, 90), note="송편 1개 20-30g"),
    "sujeonggwa": F("음료", "holiday", ["수정과", "sujeonggwa", "cinnamon punch"], ["음식_013131", "음식_013132"],
                    units={"잔": (150, 200), "컵": (150, 200), "cup": (150, 200), **G_ML}, serving=(150, 200),
                    note="수정과 1잔 150-200ml; 전통 수정과 핫/아이스 행(카페 '홍시 동동 수정과' 행은 다른 음료)"),
    "gotgam": F("과일", "holiday", ["곶감", "dried persimmon"], ["원재_001130"], units={"개": (35, 55)}, serving=(35, 55),
                note="곶감 1개 35-55g"),
    "gul_tteokguk": F("국찌개", "holiday", ["굴떡국", "떡국 굴"], ["음식_016236"], units={"그릇": (450, 650)}, serving=(450, 650),
                      note="떡국 1그릇 450-650g"),
    "torandae": F("반찬", "holiday", ["토란대나물", "토란대볶음", "토란줄기나물"], ["음식_007135", "음식_007134"], units=NAMUL,
                  serving=(40, 70), note=NAMUL_NOTE + "; '토란대나물무침' 행 포함"),

    # ---------------- 지역 음식 (regional) ----------------
    "hongeo_muchim": F("생선", "regional", ["홍어무침", "홍어회무침", "spicy skate salad"], ["음식_001959", "음식_001956"],
                       units={"접시": (200, 300)}, serving=(80, 150),
                       note="전라도 홍어무침 1접시 200-300g, 나눠 먹을 때 1인 몫 80-150g; '홍어회무침' 행 포함"),
    "kkomak_jjim": F("생선", "regional", ["꼬막찜", "삶은 꼬막", "벌교 꼬막"], ["음식_017184", "음식_017185", "음식_017183"],
                     units={"접시": (200, 300)}, serving=(100, 200), note="벌교 꼬막찜 1접시 200-300g; '꼬막찜 참꼬막' 행 포함"),
    "ganjang_gejang": F("생선", "regional", ["간장게장", "꽃게장", "게장"], ["음식_017165", "음식_017906"], units={"마리": (150, 250)},
                        serving=(150, 250), note="꽃게장 1마리 150-250g(껍데기 포함 여부 불명 -> 저확신); '게장 간장' 행 포함"),
    "jangeo_tang": F("국찌개", "regional", ["장어탕", "붕장어탕"], ["음식_010107"], units={"그릇": (500, 700)}, serving=(500, 700),
                     note="통영·여수 장어탕 1그릇 500-700g"),
    "nakji_bibimbap": F("밥", "regional", ["낙지비빔밥", "낙지 비빔밥"], ["음식_017137"], units={"그릇": (400, 500)},
                        serving=(400, 500), note="목포 낙지비빔밥 1그릇 400-500g"),
    "galchi_jorim": F("생선", "regional", ["갈치조림", "제주 갈치조림"], ["음식_018008"], units={"인분": (200, 300)},
                      serving=(200, 300), note="제주 갈치조림 1인분 200-300g"),
    "meongge_bibimbap": F("밥", "regional", ["멍게비빔밥", "멍게 비빔밥"], ["음식_013738"], units={"그릇": (300, 400)},
                          serving=(300, 400), note="통영 멍게비빔밥 1그릇 300-400g"),
    "tot_muchim": F("반찬", "regional", ["톳무침", "톳나물무침", "톳나물"], ["음식_007098"], units=NAMUL, serving=(40, 70),
                    note=NAMUL_NOTE + "; 두부 넣은 '톳나물 두부'(음식_007099)는 다른 이름"),
    "daegu_jiri": F("국찌개", "regional", ["대구지리", "대구맑은탕", "대구 지리"], ["음식_016752"],
                    units={"그릇": (500, 700), "인분": (500, 700)}, serving=(500, 700), note="거제 대구지리 1인분 500-700g"),
    "kkotge_tang": F("국찌개", "regional", ["꽃게탕"], ["음식_017159", "음식_017160"], units={"냄비": (1600, 2400), "인분": (400, 600)},
                     serving=(400, 600), note="꽃게탕 4인 냄비 1600-2400g, 1인분 400-600g"),

    # ---------------- 식물성 원재료 (plant, generic) ----------------
    "chickpea": F("콩두부", "plant", ["병아리콩", "삶은 병아리콩", "boiled chickpeas", "chickpeas"], ["원재_000454"],
                  units={"컵": (160, 170), "cup": (160, 170), **G_ML}, serving=(80, 120), note="삶은 병아리콩 1컵 약 165g"),
    "lentil": F("콩두부", "plant", ["렌틸콩", "렌즈콩", "삶은 렌틸콩", "lentils"], ["원재_000453"],
                units={"컵": (190, 200), "cup": (190, 200), **G_ML}, serving=(80, 120), note="삶은 렌틸콩 1컵 약 198g"),
}

# ---------------- 브랜드 제품 (MFDS 가공식품 라벨; 각 행은 빌드 시 원본 CSV와 대조) ----------------
PR = {
    # Southeast Asian / Middle Eastern foods sold in Korea
    "migoreng": P("면", "sea_me", ["인도미 미고렝", "미고렝", "indomie mi goreng", "mi goreng", "indomie"],
                  [("P108-003000400-0663", 475, "인도미 미고렝")], {"봉": (80, 85), "개": (80, 85), "pack": (80, 85)}, (80, 85),
                  "인도미 미고렝 1봉 80-85g(MFDS FOOD_SIZE 미기재 -> 시판 1봉 표기 85g 전후)"),
    "nasigoreng": P("밥", "sea_me", ["인생덮밥 나시고렝", "나시고렝", "나시고렝 덮밥", "nasi goreng"],
                    [("P123-201020300-5384", 191, "인생덮밥 나시고렝")], {"개": (255, 255), "팩": (255, 255)}, (255, 255),
                    "1개 255g; 업소용 '나시고렝'(P123-201020300-0733, 1670g)은 다른 이름 행"),
    "hummus": P("콩두부", "sea_me", ["맷도르 클래식 후무스", "후무스", "맷도르 후무스", "hummus"],
                [("P113-206020400-9794", 235, "맷도르 클래식 후무스")], {"통": (100, 100), "개": (100, 100)}, (30, 60),
                "1통 100g; 딥 1회 30-60g"),
    "falafel": P("콩두부", "sea_me", ["비건 팔라펠", "팔라펠", "falafel"], [("P116-703070300-0924", 285, "비건 팔라펠")],
                 {"팩": (240, 240), "개": (20, 30), "알": (20, 30)}, (80, 120), "1팩 240g; 팔라펠 1개 20-30g(통상)"),
    "tofu_padthai": P("면", "sea_me", ["두부 팟타이", "두부팟타이", "tofu pad thai"], [("P123-204020300-1604", 182, "두부 팟타이")],
                      {"개": (250, 250), "팩": (250, 250), "pack": (250, 250)}, (250, 250), "1팩 250g"),
    "tomyum_noodle": P("면", "sea_me", ["똠얌꿍 누들", "똠얌꿍누들", "똠얌 누들", "tom yum noodle"],
                       [("P108-001000200-0239", 292, "똠얌꿍 누들")], {"개": (134, 134), "봉": (134, 134)}, (134, 134), "1개 134g"),
    "kebab_dosirak": P("편의점브랜드", "sea_me", ["지중해식 치킨케밥", "지중해식 치킨케밥 도시락", "치킨케밥 도시락", "치킨케밥"],
                       [("P123-213020200-1387", 154, "지중해식 치킨케밥")], {"개": (218, 218)}, (218, 218), "도시락 1개 218g"),
    "kebab_rice": P("밥", "sea_me", ["치킨케밥라이스", "치킨케밥 라이스", "chicken kebab rice"],
                    [("P123-201020200-0166", 143, "치킨케밥라이스")], {"개": (200, 200)}, (200, 200), "1개 200g"),
    "pho_cup": P("면", "sea_me", ["CHIN-SU PHO STORY 비프플레이버쌀국수", "CHIN-SU 쌀국수", "친수 쌀국수", "쌀국수컵", "쌀국수",
                                  "chin-su pho", "pho"],
                 [("P108-010000300-0529", 358, "CHIN-SU PHO STORY 비프플레이버쌀국수")],
                 {"개": (67, 67), "컵": (67, 67), "cup": (67, 67)}, (67, 67), "1개 67g(건면+스프)"),
    "shrimp_padthai": P("면", "sea_me", ["드렁킨타이새우팟타이", "드렁킨타이 새우팟타이", "새우팟타이", "drunken thai shrimp pad thai"],
                        [("P123-204020400-0062", 165, "드렁킨타이새우팟타이")],
                        {"봉": (360, 360), "팩": (360, 360), "개": (360, 360), "pack": (360, 360)}, (360, 360), "1봉 360g"),
    # new convenience-store items
    "soft_egg": P("계란유제품", "cvs_new", ["반숙란", "편의점 반숙란", "반숙계란", "soft-boiled egg", "soft-boiled eggs"],
                  [("P118-100010700-0531", 120, "1+등급란으로 만든 반숙란"), ("P118-100010700-0252", 120, "깐반숙란"),
                   ("P118-100010700-0041", 130, "동물복지 반숙란"), ("P118-100010700-0316", 135, "동물복지 반숙란"),
                   ("P118-100010700-0231", 124, "동물복지 반숙란")],
                  {"개": (45, 55), "알": (45, 55)}, (45, 55),
                  "반숙란 1개 45-55g(MFDS 1회 제공량 50g); 70kcal/100g 표기 반숙란 행은 달걀 값과 동떨어져 미채택(다른 이름)"),
    "dubai_cookie": P("빵과자", "cvs_new", ["두바이 쫀득 쿠키", "두바이쫀득쿠키", "두쫀쿠", "dubai chewy cookie", "dubai chewy cookies"],
                      [("P101-103000100-5123", 460, "두바이 쫀득 쿠키"), ("P101-103000100-5051", 508, "두바이 쫀득 쿠키"),
                       ("P101-405000400-8754", 390, "두바이 쫀득 쿠키"), ("P101-405000400-8755", 390, "두바이 쫀득 쿠키"),
                       ("P101-103000100-5122", 291, "두바이 쫀득 쿠키"), ("P103-102020500-4068", 376, "두바이 쫀득 쿠키"),
                       ("P103-102020500-4067", 365, "두바이 쫀득 쿠키"), ("P101-103000400-2779", 468, "두바이 쫀득 쿠키"),
                       ("P101-103000400-2724", 433, "두바이 쫀득 쿠키")],
                      {"개": (40, 60)}, (40, 60), "1개 40-60g(단품 FOOD_SIZE 45-60g; 80/160g 행은 다입 포장); 제조사별 9행 모두"),
    "yonsei_bread": P("빵과자", "cvs_new", ["연세우유 우유생크림빵", "연세우유 생크림빵", "연세우유 크림빵", "생크림빵"],
                      [("P101-430000400-0719", 309, "연세우유 우유생크림빵")], {"개": (130, 130)}, (130, 130), "1개 130g"),
    "gyul_tanghulu": P("과일", "cvs_new", ["귤탕후루", "귤 탕후루", "탕후루", "tangerine tanghulu"],
                       [("P116-701070100-0163", 107, "귤탕후루")], {"개": (61, 61), "꼬치": (61, 61)}, (61, 61), "1개 61g"),
    "highball": P("음료", "cvs_new", ["엔조이 논알콜하이볼", "논알콜 하이볼", "논알콜하이볼", "하이볼", "non-alcoholic highball"],
                  [("P109-401040100-0657", 11, "엔조이 논알콜하이볼")], {"캔": (355, 355), "개": (355, 355), "can": (355, 355)},
                  (355, 355), "1캔 355ml"),
    "chamchi_samgak": P("편의점브랜드", "cvs_new", ["뉴참치마요삼각김밥", "참치마요 삼각김밥", "참치마요삼각김밥", "참치마요 삼김", "참치마요", "삼김",
                                                 "tuna mayo triangle kimbap"],
                        [("P123-203020200-0088", 173, "뉴참치마요삼각김밥"), ("P123-203020200-0087", 171, "뉴참치마요삼각김밥")],
                        {"개": (105, 115)}, (105, 115), "1개 105-115g"),
    "nucare": P("음료", "cvs_new", ["뉴케어 당플랜", "당플랜", "nucare dangplan"], [("P111-000010200-0009", 100, "뉴케어 당플랜")],
                {"팩": (200, 200), "개": (200, 200)}, (200, 200), "1팩 200ml(당뇨 환자용 균형영양식)"),
    # low-carb / keto
    "greek_plain": P("계란유제품", "keto", ["그릭요거트 플레인", "그릭요거트", "플레인 그릭요거트", "plain greek yogurt", "greek yogurt"],
                     [("P119-202040200-1573", 156, "그릭요거트 플레인"), ("P119-202040200-1511", 147, "그릭요거트 플레인")],
                     {"개": (80, 100), "컵": (80, 100), "cup": (80, 100)}, (80, 100), "1개 80-100g"),
    "bulletproof": P("음료", "keto", ["마이노멀 방탄커피", "방탄커피", "my normal bulletproof coffee", "bulletproof coffee"], [("P109-201020000-0119", 59, "마이노멀 방탄커피")],
                     {"병": (230, 230), "개": (230, 230), "bottle": (230, 230)}, (230, 230), "1병 230ml"),
    "almond_unsweet": P("음료", "keto", ["쏘굿 아몬드 언스위트", "쏘굿 아몬드", "아몬드 언스위트", "unsweetened almond milk", "almond milk"],
                        [("P109-801080100-4908", 15, "쏘굿 아몬드 언스위트"), ("P109-801080100-4525", 15, "쏘굿 아몬드 언스위트"),
                         ("P109-801080100-0848", 15, "쏘굿 아몬드 언스위트")],
                        {"잔": (200, 250), "컵": (200, 250), "cup": (200, 250)}, (200, 250), "1잔 200-250ml(MFDS 1회 제공량 200ml)"),
    "konjac_rice": P("밥", "keto", ["50KCAL 곤약밥", "50kcal 곤약밥", "곤약밥", "konjac rice"], [("P123-201020300-5710", 38, "50KCAL 곤약밥")],
                     {"개": (130, 130), "팩": (130, 130), "pack": (130, 130)}, (130, 130), "1개 130g"),
    "tofu_noodle": P("면", "keto", ["40kcal 두부면", "두부면", "tofu noodles"], [("P116-703070300-0876", 22, "40kcal 두부면")],
                     {"팩": (180, 180), "개": (180, 180)}, (180, 180), "1팩 180g"),
    "keto_cheesecake": P("빵과자", "keto", ["kohl 제로슈거키토치즈케이크플레인", "키토치즈케이크", "키토 치즈케이크", "콜 키토치즈케이크"],
                         [("P101-409000400-4968", 258, "kohl 제로슈거키토치즈케이크플레인")], {"개": (100, 100)}, (70, 100),
                         "1개 100g(MFDS 1회 제공량 70g)"),
    "konjac_ricenoodle": P("면", "keto", ["가벼운 곤약쌀국수", "곤약쌀국수", "곤약 쌀국수"], [("P123-204020300-1500", 22, "가벼운 곤약쌀국수")],
                           {"개": (320, 320), "팩": (320, 320)}, (320, 320), "1개 320g"),
    "konjac_naengmyeon": P("면", "keto", ["1am바로먹는알뜰곤약면물냉면", "곤약면 물냉면", "곤약 물냉면", "1am 곤약면 물냉면"],
                           [("P123-204020300-0004", 21, "1am바로먹는알뜰곤약면물냉면")], {"개": (280, 280), "봉": (280, 280)},
                           (280, 280), "1개 280g"),
    # plant-based products
    "oatly": P("음료", "plant", ["오틀리 오트 드링크", "오틀리", "귀리음료", "oat milk", "oatly"],
               [("P109-801080100-1068", 43, "오틀리 오트 드링크")],
               {"잔": (200, 250), "컵": (200, 250), "팩": (250, 250), "glass": (200, 250)}, (200, 250), "1잔 200-250ml",
               excl={"P109-801080100-4943": "250ml 팩 행의 110/100ml는 1팩값(110kcal/250ml)을 100ml 값으로 적은 오기로 보고 제외",
                     "P109-801080100-4942": "250ml 팩 행의 110/100ml는 1팩값(110kcal/250ml)을 100ml 값으로 적은 오기로 보고 제외"}),
    "unlimeat_kimbap": P("편의점브랜드", "plant", ["UNLIMEAT BULDAK KIMBAP 언리미트 식물성 불닭 김밥", "언리미트 불닭김밥",
                                                "언리미트 식물성 불닭 김밥", "불닭김밥", "unlimeat buldak kimbap"],
                         [("P123-203020300-0364", 157, "UNLIMEAT BULDAK KIMBAP 언리미트 식물성 불닭 김밥")],
                         {"줄": (230, 230), "개": (230, 230), "roll": (230, 230)}, (230, 230), "1줄 230g"),
    "like_tender": P("콩두부", "plant", ["식물성 지구식단 LIKE 텐더", "지구식단 텐더", "지구식단 라이크 텐더", "식물성 텐더"],
                     [("P106-001000300-0018", 270, "식물성 지구식단 LIKE 텐더")],
                     {"조각": (25, 35), "개": (25, 35), "팩": (400, 400)}, (80, 80),
                     "1팩 400g, 1회 제공량 80g; 텐더 1조각 25-35g(통상 추정)"),
    "tofu_bar": P("콩두부", "plant", ["식물성 지구식단 단백바 불두부", "단백바 불두부", "지구식단 단백바", "불두부 단백바"],
                  [("P106-005000300-0012", 227, "식물성 지구식단 단백바 불두부")], {"개": (75, 75)}, (75, 75), "1개 75g"),
    "tempeh": P("콩두부", "plant", ["검은콩템페", "템페", "검은콩 템페", "black bean tempeh", "tempeh"],
                [("P116-703070300-0069", 157, "검은콩템페")], {"팩": (100, 100), "개": (100, 100)}, (100, 100), "1팩 100g"),
    "vegemil": P("음료", "plant", ["베지밀 검은콩 저당 두유", "베지밀 저당", "베지밀 검은콩 저당", "저당 두유", "두유"],
                 [("P109-500050200-0431", 44, "베지밀 검은콩 저당 두유")], {"팩": (190, 190), "개": (190, 190)}, (190, 190),
                 "1팩 190ml"),
    "beyond_salad": P("채소샐러드", "plant", ["비욘드미트과카몰리샐러드", "비욘드미트 과카몰리 샐러드", "비욘드미트 샐러드"],
                      [("P123-226020200-0325", 132, "비욘드미트과카몰리샐러드")], {"개": (220, 220)}, (220, 220), "1개 220g"),
}
assert not (set(R) & set(PR))
R.update(PR)
EXTRA_DENS: dict = {}  # v5: no density outside DB/MFDS rows


# ---------------------------------------------------------------------------
# Case DSL
# ---------------------------------------------------------------------------
def I(key, amt=None, unit=None, grams=None, optional=False, units_ok=None, note=None, low=False, gunit=None):
    """gunit = implied unit for a bare number without a unit word ("치킨케밥라이스1"): used for grams only, units -> []."""
    assert unit is None or gunit is None
    return dict(key=key, amt=amt, unit=unit, grams=grams, optional=optional, units_ok=units_ok, note=note, low=low,
                gunit=gunit)


CASES: list[dict] = []

# Product keys -> tokens that identify the transcribed SKU in the text. A product item whose text carries none of its
# tokens is an UNBRANDED mention labelled against one representative SKU: the case is capped at medium confidence and
# tagged flag:unbranded. None = exempt (label already spans every same-food SKU row: 반숙란 5 rows, 두쫀쿠 9 rows, cola).
BRAND_TOKENS = {
    "migoreng": ["인도미", "indomie"], "nasigoreng": ["인생덮밥"], "hummus": ["맷도르"], "falafel": ["비건팔라펠", "비건 팔라펠"],
    "tofu_padthai": [], "tomyum_noodle": [], "kebab_dosirak": ["지중해식"],
    "kebab_rice": ["치킨케밥라이스", "chicken kebab rice"], "pho_cup": ["chin-su", "친수"],
    "shrimp_padthai": ["드렁킨타이", "drunken thai"], "soft_egg": None, "dubai_cookie": None, "yonsei_bread": ["연세우유"],
    "gyul_tanghulu": ["귤탕후루", "tangerine tanghulu"], "highball": ["엔조이"], "chamchi_samgak": ["뉴참치마요"],
    "nucare": ["뉴케어", "당플랜"], "greek_plain": [], "bulletproof": ["마이노멀", "my normal"], "almond_unsweet": ["쏘굿"],
    "konjac_rice": ["50kcal"], "tofu_noodle": ["40kcal"], "keto_cheesecake": ["kohl"], "konjac_ricenoodle": ["가벼운"],
    "konjac_naengmyeon": ["1am"], "oatly": ["오틀리", "oatly"], "unlimeat_kimbap": ["언리미트", "unlimeat"],
    "like_tender": ["지구식단"], "tofu_bar": ["지구식단", "불두부"], "tempeh": [], "vegemil": ["베지밀"],
    "beyond_salad": ["비욘드미트"], "cola": None,
}
UNBRANDED = "flag:unbranded"


def C(text, items, persona, style, noise=(), tags=(), conf="high", note=None, max_items=None, min_items=None):
    assert persona in PERSONAS, persona
    assert style in STYLES, style
    low_text = text.lower()
    unbranded = []
    for it in items:
        if R[it["key"]]["kind"] != "product":
            continue
        toks = BRAND_TOKENS[it["key"]]
        if toks is not None and not any(t in low_text for t in toks):
            unbranded.append(R[it["key"]]["names"][0])
    if unbranded:
        tags = list(tags) + [UNBRANDED]
        if conf == "high":
            conf = "medium"
        extra = "브랜드 미기재 -> 대표 SKU 라벨: " + ", ".join(unbranded)
        note = f"{note}; {extra}" if note else extra
    nz = list(STYLE_NOISE[style]) + [n for n in noise if n not in STYLE_NOISE[style]]
    if NEG in tags and "부정" not in nz:
        nz.append("부정")
    if CORR in tags and "정정" not in nz:
        nz.append("정정")
    CASES.append(dict(text=text, items=items, persona=persona, style=style, noise=nz, extra_tags=list(tags),
                      conf=conf, note=note, max_items=max_items, min_items=min_items))


SHARE = "flag:share"
SIZE = "flag:size"
PROD = "flag:product"
RANGE = "flag:range"
PART = "flag:partial"
NEG = "flag:negation"
CORR = "flag:correction"
TRAP = "flag:non_amount_number"
PG, DB, NS, VG, ST, PA = PERSONAS
KK, LS, ON, HD, EN, EM = STYLES

# ============================== pregnant / postpartum ==============================
C("ㅇㅇ 입덧땜에 귤탕후루 1개만 겨우 먹음 ㅠㅠ", [I("gyul_tanghulu", 1, "개")], PG, KK, tags=[PROD])
C("산후조리원 점심 미역줄기볶음 조금이랑 차조밥 한공기 ㅇㅋ", [I("miyeokjulgi"), I("chajo_bap", 1, "공기")], PG, KK)
C("임신 20주차 ㄱㄱ 오늘 간식은 연세우유 생크림빵 반개 ㅋㅋ", [I("yonsei_bread", 0.5, "개")], PG, KK, tags=[PROD, TRAP, "unit:반"],
  conf="medium", note="'20주차'는 임신 주수(양 아님)")
C("아침/ 녹두전 1장, 수정과 1잔", [I("nokdu_jeon", 1, "장"), I("sujeonggwa", 1, "잔")], PG, LS)
C("점심 - 토란국, 수수밥 반공기, 깍두기", [I("toran_guk"), I("susu_bap", 0.5, "공기"), I("kkakdugi")], PG, LS)
C("간식: 그릭요거트 플레인 1개 / 반숙란 2개", [I("greek_plain", 1, "개"), I("soft_egg", 2, "개")], PG, LS, tags=[PROD])
C("차조밥2/3공기 무생채 조금", [I("chajo_bap", 0.6667, "공기"), I("musaengchae")], PG, ON, conf="medium",
  note="2/3공기 = 0.67공기")
C("반숙란1개반 + 베지밀 저당 1팩", [I("soft_egg", 1.5, "개"), I("vegemil", 1, "팩")], PG, ON, tags=[PROD])
C("꼬막찜 1/2접시 혼자 먹음", [I("kkomak_jjim", 0.5, "접시")], PG, ON)
C("곶감 1 개 반, 수정과 200 ml", [I("gotgam", 1.5, "개"), I("sujeonggwa", 200, "ml")], PG, ON)
C("오늘은 산부인과에 입원 중이라 병원 식판으로 근대 된장국 한 그릇과 기장밥 한 공기, 시금치나물을 먹었습니다.",
  [I("geundae_guk", 1, "그릇"), I("gijang_bap", 1, "공기"), I("sigeumchi")], PG, HD)
C("출산 후 첫 명절이라 시어머님께서 해주신 돼지갈비찜을 한 접시 먹었고, 호박전도 세 개 먹었습니다.",
  [I("dwaeji_galbijjim", 1, "접시"), I("hobak_jeon", 3, "개")], PG, HD)
C("임신 중이라 날것은 피하고 있어서 저녁은 대구지리 한 그릇에 밥 반 공기를 먹었습니다.",
  [I("daegu_jiri", 1, "그릇"), I("rice", 0.5, "공기")], PG, HD)
C("모유 수유 중이라 밤중에 출출해서 오틀리 오트 드링크 한 잔과 검은콩템페 반 팩을 먹었습니다.",
  [I("oatly", 1, "잔"), I("tempeh", 0.5, "팩")], PG, HD, tags=[PROD])
C("Pregnant, 28 weeks. Lunch was taro soup and half a bowl of millet rice.",
  [I("toran_guk"), I("gijang_bap", 0.5, "그릇", units_ok=["그릇", "공기", "bowl"])], PG, EN, tags=[TRAP], conf="medium",
  note="'28 weeks'는 임신 주수")
C("Craving sweets today: one tangerine tanghulu and a Dubai chewy cookie.",
  [I("gyul_tanghulu", 1, gunit="개"), I("dubai_cookie", 1, gunit="개")], PG, EN, tags=[PROD])
C("Postpartum dinner at the care center: a bowl of black bean rice, seaweed stem stir-fry and spinach namul.",
  [I("kong_bap", 1, "그릇", units_ok=["그릇", "공기", "bowl"]), I("miyeokjulgi"), I("sigeumchi")], PG, EN)
C("Had 3 soft-boiled eggs from the convenience store and a cup of plain greek yogurt.",
  [I("soft_egg", 3, gunit="개"), I("greek_plain", 1, "컵", units_ok=["컵", "개", "cup"])], PG, EN, tags=[PROD])
C("🍞✨ 연세우유 생크림빵 1개 😋😋 입덧 끝!!", [I("yonsei_bread", 1, "개")], PG, EM, tags=[PROD])
C("🥟❌ 🦐새우전 4개 + 🌶️고추전 2개 🎉 명절 최고", [I("saeu_jeon", 4, "개"), I("gochu_jeon", 2, "개")], PG, EM, tags=[NEG])
C("🍚 콩밥 한 공기 + 🥬 무생채 + 🐟 갈치조림 1인분 🤰", [I("galchi_jorim", 1, "인분"), I("kong_bap", 1, "공기"),
                                                 I("musaengchae")], PG, EM)
C("🍜 1am 곤약면 물냉면 1개 🙆‍♀️ 임당 검사 앞두고 💪", [I("konjac_naengmyeon", 1, "개")], PG, EM, tags=[PROD])

# ============================== diabetic (carb counting) ==============================
C("ㅇㅇ 혈당땜에 밥 대신 50kcal 곤약밥 1개로 바꿈 ㄱㄱ", [I("konjac_rice", 1, "개")], DB, KK, tags=[PROD, TRAP, NEG],
  conf="medium", note="'50kcal'은 제품명(양 아님)")
C("점심 학교급식 흑미밥 반공기만 ㅇㅇ 감자국은 국물만", [I("heukmi_bap", 0.5, "공기"), I("gamja_guk", grams=(150, 250))], DB, KK,
  tags=[PART], conf="medium", note="감자국 건더기 빼고 국물 위주 150-250g")
C("당뇨식 ㅋㅋ 뉴케어 당플랜 1팩 + 반숙란 1개", [I("nucare", 1, "팩"), I("soft_egg", 1, "개")], DB, KK, tags=[PROD])
C("ㄹㅇ 홍어무침 몇 점 집어먹음 ㅠ 탄수 거의 없다길래", [I("hongeo_muchim", grams=(40, 80), low=True)], DB, KK, conf="low",
  note="'몇 점' = 40-80g 추정")
C("아침/ 베지밀 검은콩 저당 두유 1팩, 삶은 계란 1개", [I("vegemil", 1, "팩"), I("egg", 1, "개")], DB, LS, tags=[PROD])
C("점심: 콩밥 1/2공기, 고등어찜, 가지나물", [I("kong_bap", 0.5, "공기"), I("godeungeo_jjim"), I("gaji_namul")], DB, LS)
C("저녁 - 두부면 1팩(40kcal짜리), 쏘굿 아몬드 언스위트 1잔", [I("tofu_noodle", 1, "팩"), I("almond_unsweet", 1, "잔")], DB, LS,
  tags=[PROD, TRAP], conf="medium", note="'40kcal'은 제품명")
C("간식 / 곶감 반개, 수정과 반 잔", [I("gotgam", 0.5, "개"), I("sujeonggwa", 0.5, "잔")], DB, LS, tags=["unit:반"])
C("현미 대신 수수밥2/3공기(탄수 약 45g) 연근조림 30g", [I("susu_bap", 0.6667, "공기"), I("yeongeun_jorim", 30, "g")], DB, ON,
  tags=[TRAP], conf="medium", note="'탄수 약 45g'은 탄수화물량(양 아님); 2/3공기")
C("키토치즈케이크1개 + 마이노멀 방탄커피1병", [I("keto_cheesecake", 1, "개"), I("bulletproof", 1, "병")], DB, ON, tags=[PROD])
C("곤약면 물냉면 1 봉 다 먹음 + 반숙란 2 개", [I("konjac_naengmyeon", 1, "봉"), I("soft_egg", 2, "개")], DB, ON, tags=[PROD])
C("오늘 혈당이 높게 나와서 저녁은 장어탕 한 그릇만 먹고 밥은 먹지 않았습니다.", [I("jangeo_tang", 1, "그릇")], DB, HD, tags=[NEG])
C("병원 영양사 선생님 말씀대로 점심에 차조밥 반 공기, 우엉조림 조금, 미역줄기볶음 한 접시를 먹었습니다.",
  [I("chajo_bap", 0.5, "공기"), I("ueong_jorim"), I("miyeokjulgi", 1, "접시")], DB, HD)
C("명절이라 녹두전을 두 장 먹고 싶었지만 한 장만 먹었고 토란국도 반 그릇만 먹었습니다.",
  [I("nokdu_jeon", 1, "장"), I("toran_guk", 0.5, "그릇")], DB, HD, tags=[NEG, CORR], conf="medium",
  note="'두 장 먹고 싶었지만' -> 실제 1장")
C("Diabetic lunch: one 130g pack of konjac rice and some stir-fried mushrooms.",
  [I("konjac_rice", 1, "팩", units_ok=["팩", "개", "pack"]), I("neutari")], DB, EN, tags=[PROD])
C("Breakfast: unsweetened almond milk, 1 cup, plus 2 boiled eggs.",
  [I("almond_unsweet", 1, "컵", units_ok=["컵", "잔", "cup"]), I("egg", 2, gunit="개")], DB, EN, tags=[PROD])
C("Dinner: braised lotus root, half a bowl of black rice, and some cubed radish kimchi.",
  [I("yeongeun_jorim"), I("heukmi_bap", 0.5, "그릇", units_ok=["그릇", "공기", "bowl"]), I("kkakdugi")], DB, EN)
C("Snack: black bean tempeh, 100g, pan-fried.", [I("tempeh", 100, "g")], DB, EN, tags=[PROD])
C("🩸혈당 체크✅ 🍚기장밥 반공기 + 🥬시금치나물 + 🍲배추된장국 1그릇",
  [I("gijang_bap", 0.5, "공기"), I("sigeumchi"), I("baechu_guk", 1, "그릇")], DB, EM)
C("🥛 뉴케어 당플랜 2팩 😮‍💨 밥 대신", [I("nucare", 2, "팩")], DB, EM, tags=[PROD, NEG])
C("🦀 간장게장 1마리 🍚❌ 밥 없이 ㅠㅠ", [I("ganjang_gejang", 1, "마리")], DB, EM, tags=[NEG], conf="low",
  note="게장 1마리 무게·가식부 불확실")
C("🍜 가벼운 곤약쌀국수 1개 + 🥚반숙란 1개 👍", [I("konjac_ricenoodle", 1, "개"), I("soft_egg", 1, "개")], DB, EM, tags=[PROD])

# ============================== night-shift worker ==============================
C("야간 3시 ㅇㅇ 삼김 참치마요 1개 + 엔조이 논알콜하이볼 1캔 ㄱㄱ", [I("chamchi_samgak", 1, "개"), I("highball", 1, "캔")], NS, KK,
  tags=[PROD, TRAP], conf="medium", note="'3시'는 시각")
C("ㅋㅋ 새벽에 인도미 미고렝 1봉 끓여먹음 ㄹㅇ 맛도리", [I("migoreng", 1, "봉")], NS, KK, tags=[PROD])
C("ㅇㅋ 퇴근길 편의점 반숙란 2개 두쫀쿠 1개", [I("soft_egg", 2, "개"), I("dubai_cookie", 1, "개")], NS, KK, tags=[PROD])
C("나이트 끝나고 똠얌꿍 누들 1개 ㅠㅠ 맵다 ㄹㅇ", [I("tomyum_noodle", 1, "개")], NS, KK, tags=[PROD])
C("새벽 2시/ 지중해식 치킨케밥 도시락 1개, 아메리카노", [I("kebab_dosirak", 1, "개"), I("americano")], NS, LS,
  tags=[PROD, TRAP], conf="medium", note="'2시'는 시각")
C("야식: 인생덮밥 나시고렝 1개 / 콜라 X", [I("nasigoreng", 1, "개")], NS, LS, tags=[PROD, NEG])
C("휴게시간 - 오삼불고기 1인분, 흑미밥 1공기, 깍두기", [I("osam", 1, "인분"), I("heukmi_bap", 1, "공기"), I("kkakdugi")], NS, LS)
C("퇴근 후, 키토 치즈케이크 반 개, 쏘굿 아몬드 1잔", [I("keto_cheesecake", 0.5, "개"), I("almond_unsweet", 1, "잔")], NS, LS,
  tags=[PROD, "unit:반"])
C("새벽4시 쌀국수컵1개+귤탕후루1개", [I("pho_cup", 1, "개"), I("gyul_tanghulu", 1, "개")], NS, ON, tags=[PROD, TRAP],
  conf="medium", note="'4시'는 시각")
C("곤약밥130g 1개 + 오삼불고기0.5인분", [I("konjac_rice", 1, "개"), I("osam", 0.5, "인분")], NS, ON, tags=[PROD])
C("드렁킨타이 새우팟타이 반 봉 (360g짜리)", [I("shrimp_padthai", 0.5, "봉")], NS, ON, tags=[PROD, PART, "unit:반"],
  conf="medium", note="360g은 1봉 중량, 먹은 양은 반 봉 = 180g")
C("치킨케밥라이스1 + 콜라1캔", [I("kebab_rice", 1, gunit="개"), I("cola", 1, "캔")], NS, ON, tags=[PROD])
C("야간 근무를 마치고 아침 7시에 사내 식당에서 감자볶음과 유부국, 기장밥 한 공기를 먹었습니다.",
  [I("gamja_bokkeum"), I("yubu_guk"), I("gijang_bap", 1, "공기")], NS, HD, tags=[TRAP], conf="medium", note="'7시'는 시각")
C("밤 근무 중에 간식으로 연세우유 우유생크림빵 하나를 동료와 반씩 나누어 먹었습니다.",
  [I("yonsei_bread", (0.5, 1), "개", grams=(60, 70))], NS, HD, tags=[PROD, SHARE], note="1개 130g을 둘이 반씩")
C("오늘은 새벽에 출출하여 CHIN-SU 쌀국수 한 컵과 반숙란 한 개를 먹었습니다.",
  [I("pho_cup", 1, "컵"), I("soft_egg", 1, "개")], NS, HD, tags=[PROD])
C("근무 끝나고 가족들이 남겨둔 돼지갈비찜을 조금 데워 먹었습니다.", [I("dwaeji_galbijjim", grams=(100, 200), low=True)], NS, HD,
  conf="low", note="'조금' = 100-200g 추정")
C("Night shift snack at 3am: one tuna mayo triangle kimbap and a non-alcoholic highball can.",
  [I("chamchi_samgak", 1, gunit="개"), I("highball", 1, "캔", units_ok=["캔", "can"])], NS, EN, tags=[PROD, TRAP],
  conf="medium", note="'3am'은 시각")
C("4am: Indomie mi goreng, one pack, plus a soft-boiled egg.",
  [I("migoreng", 1, "봉", units_ok=["봉", "pack"]), I("soft_egg", 1, gunit="개")], NS, EN, tags=[PROD, TRAP], conf="medium",
  note="'4am'은 시각")
C("Before sleeping after my shift: My Normal bulletproof coffee, 1 bottle.", [I("bulletproof", 1, "병", units_ok=["병", "bottle"])],
  NS, EN, tags=[PROD])
C("🌙🏥 야간 끝 ➡️ 🍛 오징어카레 1그릇 😴", [I("curry_squid", 1, "그릇")], NS, EM)
C("🌃🍜 똠얌꿍누들 반개 + 🥚반숙란 1개 🔥🔥", [I("tomyum_noodle", 0.5, "개"), I("soft_egg", 1, "개")], NS, EM,
  tags=[PROD, "unit:반"])
C("☕️ 방탄커피 ❌ → 🥛 오틀리 1잔 ✅ 🌅", [I("oatly", 1, "잔")], NS, EM, tags=[PROD, NEG])

# ============================== vegetarian / vegan ==============================
C("ㅇㅇ 비건이라 언리미트 불닭김밥 1줄 ㄱㄱ", [I("unlimeat_kimbap", 1, "줄")], VG, KK, tags=[PROD])
C("점심 두부팟타이 1개 ㅋㅋ 땅콩 듬뿍 ㄹㅇ", [I("tofu_padthai", 1, "개")], VG, KK, tags=[PROD])
C("ㅇㅋ 지구식단 텐더 4조각 에어프라이어 ㄱ", [I("like_tender", 4, "조각")], VG, KK, tags=[PROD], conf="medium",
  note="텐더 1조각 25-35g 추정")
C("아침/ 오틀리 1잔, 곶감 1개", [I("oatly", 1, "잔"), I("gotgam", 1, "개")], VG, LS, tags=[PROD])
C("점심 - 후무스 50g, 팔라펠 4개, 피타 X", [I("hummus", 50, "g"), I("falafel", 4, "개")], VG, LS, tags=[PROD, NEG])
C("저녁: 가지나물, 표고버섯볶음, 수수밥 1공기", [I("gaji_namul"), I("pyogo"), I("susu_bap", 1, "공기")], VG, LS)
C("병아리콩 80g + 렌틸콩1/2컵", [I("chickpea", 80, "g"), I("lentil", 0.5, "컵")], VG, ON)
C("템페1팩반 볶아서 먹음", [I("tempeh", 1.5, "팩")], VG, ON, tags=[PROD])
C("비건팔라펠 240g 1팩의 1/3", [I("falafel", 0.3333, "팩")], VG, ON, tags=[PROD, PART], conf="medium",
  note="240g 1팩의 1/3 = 80g")
C("두부카레 1그릇반 ㅎ", [I("tofu_curry", 1.5, "그릇")], VG, ON)
C("채식을 하고 있어서 명절 음식 중에 호박전과 고추전만 골라서 각각 두 개씩 먹었습니다.",
  [I("hobak_jeon", 2, "개"), I("gochu_jeon", 2, "개")], VG, HD)
C("오늘 회사 식당에 채식 메뉴가 나와서 두부카레 한 그릇과 무생채를 먹었습니다.",
  [I("tofu_curry", 1, "그릇"), I("musaengchae")], VG, HD)
C("점심으로 비욘드미트 과카몰리 샐러드 한 개를 먹었고 오후에는 베지밀 저당 두유 한 팩을 마셨습니다.",
  [I("beyond_salad", 1, "개"), I("vegemil", 1, "팩")], VG, HD, tags=[PROD])
C("저녁에는 지구식단 단백바 불두부 한 개와 토란대나물 조금을 먹었습니다.", [I("tofu_bar", 1, "개"), I("torandae")], VG, HD,
  tags=[PROD])
C("Vegan lunch: tofu pad thai (one pack) and a glass of oat milk.",
  [I("tofu_padthai", 1, "팩", units_ok=["팩", "개", "pack"]), I("oatly", 1, "잔", units_ok=["잔", "컵", "glass"])], VG, EN,
  tags=[PROD])
C("Had hummus with carrot sticks, about 100g of hummus.", [I("hummus", 100, "g")], VG, EN, tags=[PROD])
C("Dinner: 1 cup of boiled chickpeas and some sauteed eggplant.",
  [I("chickpea", 1, "컵", units_ok=["컵", "cup"]), I("gaji_namul")], VG, EN)
C("Snack: one Unlimeat plant-based buldak kimbap roll.", [I("unlimeat_kimbap", 1, "줄", units_ok=["줄", "개", "roll"])], VG, EN,
  tags=[PROD])
C("🌱🥗 비욘드미트 과카몰리샐러드 1개 💚", [I("beyond_salad", 1, "개")], VG, EM, tags=[PROD])
C("🌱 렌틸콩 1컵 + 🍄 표고볶음 😋", [I("lentil", 1, "컵"), I("pyogo")], VG, EM)
C("🍢❌ 🥬 톳무침 + 🍚 차조밥 1공기 🌿", [I("tot_muchim"), I("chajo_bap", 1, "공기")], VG, EM, tags=[NEG])
C("🌱💪 단백바 불두부 2개 + 🥛 쏘굿 아몬드 언스위트 1잔", [I("tofu_bar", 2, "개"), I("almond_unsweet", 1, "잔")], VG, EM,
  tags=[PROD])

# ============================== college student living alone ==============================
C("ㅇㅇ 자취방에서 미고렝 2봉 ㄹㅇ 배고팠음", [I("migoreng", 2, "봉")], ST, KK, tags=[PROD])
C("ㄱㄱ 학식 오삼불고기 1인분 ㅋㅋ 밥 리필함", [I("osam", 1, "인분"), I("rice", grams=(190, 460), low=True)], ST, KK, conf="low",
  note="밥 기본 1공기 + 리필(양 불명) 190-460g")
C("두쫀쿠 2개 사옴 ㅠ 통장 텅장", [I("dubai_cookie", 2, "개")], ST, KK, tags=[PROD])
C("ㅇㅋ 공강때 편의점 반숙란 3개 까먹음 ㅋㅋ", [I("soft_egg", 3, "개")], ST, KK, tags=[PROD])
C("아침/ 그릭요거트 플레인 반 개, 아아", [I("greek_plain", 0.5, "개"), I("americano")], ST, LS, tags=[PROD, "unit:반"])
C("점심 - 학식 카레떡볶이, 흑미밥 반공기", [I("curry_tteokbokki"), I("heukmi_bap", 0.5, "공기")], ST, LS)
C("저녁: 맷도르 후무스 1통, 크래커 X", [I("hummus", 1, "통")], ST, LS, tags=[PROD, NEG])
C("야식 / 뉴참치마요삼각김밥 2개, 엔조이 하이볼(논알콜) 1캔", [I("chamchi_samgak", 2, "개"), I("highball", 1, "캔")], ST, LS, tags=[PROD])
C("참치마요삼김2개+반숙란1개", [I("chamchi_samgak", 2, "개"), I("soft_egg", 1, "개")], ST, ON, tags=[PROD])
C("나시고렝 1/2개 + 똠얌꿍누들 1개", [I("nasigoreng", 0.5, "개"), I("tomyum_noodle", 1, "개")], ST, ON, tags=[PROD])
C("두바이쫀득쿠키 1개 3/4", [I("dubai_cookie", 1.75, "개")], ST, ON, tags=[PROD], conf="medium", note="1개 3/4 = 1.75개")
C("오늘은 시험 기간이라 학교 식당에서 고구마돈가스 한 장과 감자국 한 그릇을 먹었습니다.",
  [I("goguma_donkatsu", 1, "장"), I("gamja_guk", 1, "그릇")], ST, HD)
C("교생 실습 중이라 학생들과 함께 급식으로 간장떡볶이와 흑미밥 한 공기를 먹었습니다.",
  [I("ganjang_tteokbokki"), I("heukmi_bap", 1, "공기")], ST, HD)
C("명절에 고향에 못 가서 편의점에서 반숙란 두 개와 곶감 하나로 아쉬움을 달랬습니다.",
  [I("soft_egg", 2, "개"), I("gotgam", 1, "개")], ST, HD, tags=[PROD])
C("Late-night ramen alternative: one cup of CHIN-SU pho.", [I("pho_cup", 1, "컵", units_ok=["컵", "개", "cup"])], ST, EN,
  tags=[PROD])
C("Lunch at the campus cafeteria: sweet potato pork cutlet and kkakdugi.", [I("goguma_donkatsu"), I("kkakdugi")], ST, EN)
C("Two Dubai chewy cookies and an iced americano, that was lunch.", [I("dubai_cookie", 2, gunit="개"), I("americano")], ST, EN,
  tags=[PROD])
C("Dinner: half a pack of Drunken Thai shrimp pad thai.", [I("shrimp_padthai", 0.5, "봉", units_ok=["봉", "팩", "pack"])], ST, EN,
  tags=[PROD, PART])
C("🍙 참치마요 하나 + 🥤 콜라 한 캔 💸 월말 거지", [I("chamchi_samgak", 1, "개"), I("cola", 1, "캔")], ST, EM, tags=[PROD],
  conf="medium", note="'참치마요' = 편의점 참치마요 삼각김밥")
C("🍚 곤약밥 1개 + 🥚 반숙란 2개 🏋️ 자취 다이어트", [I("konjac_rice", 1, "개"), I("soft_egg", 2, "개")], ST, EM, tags=[PROD])
C("🍡 귤탕후루 2개 🍊🍊 ㅎㅎ", [I("gyul_tanghulu", 2, "개")], ST, EM, tags=[PROD])
C("🍛 오늘 학식 메뉴 오징어 카레!! 한 그릇 뚝딱 🥒깍두기도", [I("curry_squid", 1, "그릇"), I("kkakdugi")], ST, EM)

# ============================== parent logging a family meal (my share only) ==============================
C("ㅇㅇ 애들이랑 꼬막찜 1접시 셋이 나눠먹음 ㄱㄱ", [I("kkomak_jjim", (0.3333, 1), "접시", grams=(67, 100))], PA, KK,
  tags=[SHARE], note="1접시 200-300g / 3명")
C("ㅋㅋ 남편이랑 대구지리 2인분 시켜서 반반", [I("daegu_jiri", (1, 2), "인분", grams=(500, 700))], PA, KK, tags=[SHARE],
  note="2인분 / 2명 = 1인분")
C("애 학교 급식 모니터링 갔다가 토마토 스파게티 한 접시 먹음 ㅇㅇ", [I("tomato_spaghetti", 1, "접시")], PA, KK)
C("ㄹㅇ 시댁에서 화양적 3꼬치 산적 2개 ㅠㅠ", [I("hwayangjeok", 3, "꼬치"), I("sanjeok", 2, "개")], PA, KK)
C("저녁(가족 4명)/ 꽃게탕 1냄비, 나는 1/4", [I("kkotge_tang", (0.25, 1), "냄비", grams=(400, 600))], PA, LS, tags=[SHARE],
  note="4인 냄비 1600-2400g / 4명")
C("점심: 미트볼 5개, 소시지볶음 조금, 기장밥 1공기 (애들 반찬 같이)",
  [I("meatball", 5, "개"), I("sausage_bokkeum"), I("gijang_bap", 1, "공기")], PA, LS)
C("간식 - 애가 남긴 카레떡볶이 조금, 귤탕후루 1개", [I("curry_tteokbokki", grams=(50, 100), low=True), I("gyul_tanghulu", 1, "개")],
  PA, LS, tags=[PART, PROD], conf="low", note="아이가 남긴 떡볶이 '조금' = 50-100g 추정")
C("아침 / 굴떡국 1그릇, 깍두기", [I("gul_tteokguk", 1, "그릇"), I("kkakdugi")], PA, LS)
C("빈대떡3장중 1장만 내꺼 + 식혜X", [I("nokdu_jeon", (1, 3), "장", grams=(100, 180))], PA, ON, tags=[SHARE, NEG],
  note="3장 중 본인 1장")
C("멍게비빔밥1.5그릇 (남편꺼 반 뺏음)", [I("meongge_bibimbap", 1.5, "그릇")], PA, ON)
C("새우전 10개 사서 넷이 → 내꺼 2~3개", [I("saeu_jeon", (2, 3), "개")], PA, ON, tags=[RANGE, SHARE], note="본인 2~3개")
C("갈치조림 2인분 3명이서 / 밥2/3공기", [I("galchi_jorim", (0.6667, 2), "인분", grams=(133, 200)), I("rice", 0.6667, "공기")],
  PA, ON, tags=[SHARE], conf="medium", note="2인분 / 3명; 밥 2/3공기")
C("오늘 친정에서 돼지갈비찜 4인분을 다섯 명이 나누어 먹었고, 저는 밥 한 공기와 함께 먹었습니다.",
  [I("dwaeji_galbijjim", (0.8, 4), "인분", grams=(200, 280)), I("rice", 1, "공기")], PA, HD, tags=[SHARE],
  note="4인분(1000-1400g) / 5명")
C("아이 학교 운동회에 가서 학부모 도시락으로 고구마돈가스 한 장과 흑미밥 반 공기를 먹었습니다.",
  [I("goguma_donkatsu", 1, "장"), I("heukmi_bap", 0.5, "공기")], PA, HD)
C("남편이 장어탕을 두 그릇 사 와서 저는 한 그릇의 절반만 먹었습니다.", [I("jangeo_tang", (0.5, 2), "그릇", grams=(250, 350))],
  PA, HD, tags=[PART, SHARE], note="1그릇 500-700g의 절반")
C("명절 아침에 온 가족이 모여 저는 토란국 한 그릇과 검정콩송편 세 개를 먹었습니다.",
  [I("toran_guk", 1, "그릇"), I("songpyeon_bk", 3, "개")], PA, HD)
C("Family dinner, four of us shared one plate of spicy skate salad, so I had about a quarter.",
  [I("hongeo_muchim", (0.25, 1), "접시", grams=(50, 75), units_ok=["접시", "plate"])], PA, EN, tags=[SHARE],
  note="1접시 200-300g / 4명")
C("Kids' school menu at home: I had 4 meatballs and a bowl of potato soup.",
  [I("meatball", 4, gunit="개"), I("gamja_guk", 1, "그릇", units_ok=["그릇", "bowl"])], PA, EN)
C("Holiday brunch: two zucchini jeon and one shrimp jeon, plus a cup of sujeonggwa.",
  [I("hobak_jeon", 2, gunit="개"), I("saeu_jeon", 1, gunit="개"), I("sujeonggwa", 1, "컵", units_ok=["컵", "잔", "cup"])], PA, EN)
C("👨‍👩‍👧‍👦 낙지비빔밥 2그릇 셋이서 🍽️ 나는 1/3", [I("nakji_bibimbap", (0.6667, 2), "그릇", grams=(267, 333))], PA, EM,
  tags=[SHARE], conf="medium", note="2그릇(800-1000g)의 1/3")
C("🎑🌕 검정콩송편 5개 + 🍵 수정과 반잔 😊", [I("songpyeon_bk", 5, "개"), I("sujeonggwa", 0.5, "잔")], PA, EM, tags=["unit:반"])
C("🍗❌ 애들 치킨은 안 먹고 🐟 고등어찜 1토막 + 🍚 콩밥 반공기",
  [I("godeungeo_jjim", 1, "토막"), I("kong_bap", 0.5, "공기")], PA, EM, tags=[NEG])


# ---------------------------------------------------------------------------
# Build (helpers as in build_diet_holdout_v4.py)
# ---------------------------------------------------------------------------
_SYN_GROUPS = [
    {"개", "알", "ea", "pc", "피스"}, {"공기", "그릇"}, {"그릇", "인분", "접시", "대접"}, {"잔", "컵"},
    {"봉", "봉지", "개"}, {"조각", "쪽", "피스", "개"}, {"장", "쪽", "개", "slice"}, {"캔", "개"}, {"병", "개"},
    {"팩", "개"}, {"줄", "개"}, {"토막", "조각", "개"}, {"스쿱", "scoop", "숟가락"}, {"L", "l", "리터"},
    {"g", "그램", "gram", "grams"}, {"ml", "mL", "cc", "밀리리터"}, {"통", "개"}, {"갑", "곽", "개"},
    {"뚝배기", "그릇"}, {"대", "개"}, {"꼬치", "개"},
]
_HANGUL_NUM_RE = re.compile(r"(?:(?:한|두|세|네|다섯|여섯|일곱|여덟|아홉|열|서너|두세|대여섯|한두|이백)\s*(?:개|공기|그릇|잔|캔|병|조각|줄|장|마리|인분|봉지|팩|컵|스쿱|토막|송이|모|알|줌|갑|통|판|덩이|접시|쪽|봉|대|대접|뚝배기|미리|시|스쿱)(?![가-힣])|(?:한|두|세|네|다섯)\s*(?:개|공기|그릇|잔|캔|병|조각|줄|장|마리|인분|봉지|팩|컵|토막|송이|모|알|줌|갑|통|판|접시|쪽|봉|대|대접|뚝배기)|하나|둘이|셋이|넷이)")
UNIT_TAGS = {"g": "g", "ml": "ml", "cc": "ml", "L": "ml", "개": "개", "알": "개", "조각": "조각", "토막": "조각",
             "그릇": "그릇", "접시": "그릇", "대접": "그릇", "뚝배기": "그릇", "공기": "공기", "인분": "인분", "캔": "캔",
             "병": "병", "잔": "잔", "컵": "컵", "스쿱": "스쿱", "줄": "줄", "봉": "봉지", "봉지": "봉지", "팩": "팩",
             "마리": "마리", "모": "모", "송이": "송이", "줌": "줌", "갑": "갑", "통": "통", "판": "판", "대": "개", "세트": "세트",
             "덩이": "개", "사발": "그릇", "근": "근", "꼬치": "개", "냄비": "냄비"}


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
    """Stream the raw MFDS CSV once; assert every transcribed (FOOD_CD, kcal, name) row and that every MFDS row whose
    FOOD_NM is identical to a transcribed name is either transcribed or excluded (with a reason) for that key."""
    want, names = {}, {}
    for key, f in R.items():
        if f["kind"] == "product":
            for code, k, nm in f["mfds"]:
                want[code] = (key, float(k), nm)
                names.setdefault(nm, set()).add(key)
    seen, same_name = {}, {}
    with MFDS_PATH.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["FOOD_CD"] in want:
                seen[row["FOOD_CD"]] = row
            nm = row["FOOD_NM"].strip()
            if nm in names:
                same_name.setdefault(nm, []).append((row["FOOD_CD"], row["ENERC"], row["FOOD_SIZE"]))
    for code, (key, k, nm) in want.items():
        assert code in seen, f"{key}: MFDS code {code} not found"
        row = seen[code]
        assert float(row["ENERC"]) == k, f"{key}: {code} ENERC {row['ENERC']} != {k}"
        assert row["FOOD_NM"].strip() == nm, f"{key}: {code} name {row['FOOD_NM']!r} != {nm!r}"
        assert row["NUT_CON_SRTR_QUA"] in ("100g", "100ml"), f"{key}: {code} basis {row['NUT_CON_SRTR_QUA']}"
    missing = []
    for nm, rows_ in same_name.items():
        for key in names[nm]:
            f = R[key]
            listed = {c for c, _, _ in f["mfds"]} | set(f["excl"])
            missing += [(key, nm, c, e, s) for c, e, s in rows_ if c not in listed]
    assert not missing, f"same-name MFDS rows neither transcribed nor excluded: {missing}"
    return len(want)


def _check_db_complete(rows):
    """Every food_db row whose name is identical to a gold row's name must be in gold or in excl (with a reason)."""
    by_name = {}
    for fid, r in rows.items():
        by_name.setdefault(r["name"], []).append(fid)
    missing = []
    for key, f in R.items():
        if f["kind"] != "generic":
            continue
        gold = {g for g in f["gold"] if g not in (CUR, LLM)}
        for g in gold:
            assert g in rows, f"{key}: unknown food_id {g}"
            for fid in by_name[rows[g]["name"]]:
                if fid not in gold and fid not in f["excl"]:
                    missing.append((key, fid, rows[fid]["name"], rows[fid]["kcal"]))
    assert not missing, f"same-name food_db rows neither in gold nor excluded: {missing}"


def _review_token_twins(rows) -> list:
    """PRINT-only review: food_db rows whose name token set equals a gold row's token set (reordered wording such as
    '근대 된장국' / '된장국 근대') but that are neither in gold nor excluded."""
    by_tok = {}
    for fid, r in rows.items():
        by_tok.setdefault(frozenset(r["name"].split()), []).append(fid)
    out = []
    for key, f in R.items():
        if f["kind"] != "generic":
            continue
        gold = {g for g in f["gold"] if g not in (CUR, LLM)}
        for g in gold:
            for fid in by_tok.get(frozenset(rows[g]["name"].split()), []):
                if fid not in gold and fid not in f["excl"]:
                    out.append((key, fid, rows[fid]["name"]))
    return sorted(set(out))


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
    return dens, parts


def _scratch_inputs() -> list[str]:
    src = (ROOT / "scripts" / "eval" / "build_diet_eval_v1.py").read_text(encoding="utf-8")
    block = src[src.index("SCRATCH_INPUTS = {"):]
    block = block[:block.index("}") + 1]
    return re.findall(r'"([^"]+)"', block)


def _category(tags: set) -> str:
    if tags & {SHARE, RANGE, PART, NEG, CORR}:
        return "tricky"
    if PROD in tags:
        return "product"
    return "generic"


def main():
    rows = {r["food_id"]: r for r in csv.DictReader(CSV_PATH.open(encoding="utf-8-sig"))}
    _check_db_complete(rows)
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
                    why = f"jaccard {j:.2f} vs earlier v5 {t!r}"
                    break
        if why:
            dropped.append((text, why))
            continue
        kept.append(spec)
        kept_bg.append((text, bg))

    out, meta = [], []
    for n, spec in enumerate(kept, 1):
        cid = f"{ID_PREFIX}{n:03d}"
        text = spec["text"]
        items_out, tags = [], set()
        for it in spec["items"]:
            f = R[it["key"]]
            tags.add(f"group:{f['group']}")
            amt, unit, gunit = it["amt"], it["unit"], it["gunit"]
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
            elif gunit is not None:
                tags.add("unit:단위생략")
            # grams
            gu = unit or gunit
            if it["grams"] is not None:
                g_lo, g_hi = it["grams"]
                g_basis = f"{g_lo}-{g_hi}g (case-specific)"
            elif amt is None:
                g_lo, g_hi = f["serving"]
                g_basis = f"1회 제공량/1개 {g_lo}-{g_hi}g"
            elif isinstance(amt, tuple):
                # explicit range ("서너 개"); sharing/partial tuples always pass grams
                assert RANGE in spec["extra_tags"], f"{cid}: tuple amount without grams must be a range case"
                a_lo, a_hi = float(amt[0]), float(amt[1])
                per = f["units"][gu]
                g_lo, g_hi = a_lo * per[0], a_hi * per[1]
                g_basis = f"{a_lo:g}-{a_hi:g}{gu} x {per[0]}-{per[1]}g"
            else:
                a = float(amt)
                assert gu is not None, f"{cid}: numeric amount for {it['key']} needs unit or gunit"
                if gu in ("g", "ml", "cc"):
                    g_lo, g_hi = a * 0.98, a * 1.02
                    g_basis = f"{a:g}{gu}"
                else:
                    assert gu in f["units"], f"{cid}: unit {gu!r} not in registry units of {it['key']}"
                    per = f["units"][gu]
                    g_lo, g_hi = a * per[0], a * per[1]
                    g_basis = f"{a:g}{gu} x {per[0]}-{per[1]}g" + (" (단위 생략, 암묵 단위)" if unit is None else "")
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
            for xid, why in f.get("excl", {}).items():
                basis += f"; {xid} 제외: {why}"
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
        main_key = next((i["key"] for i in req), None)
        domain = R[main_key]["domain"] if main_key else "none"
        tags |= {f"persona:{spec['persona']}", f"style:{spec['style']}", f"domain:{domain}", SET_TAG}
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
        meta.append((spec["persona"], spec["style"], domain))

    # ---- balance checks (asserted) ----
    for t, why in dropped:
        print(f"    DROP {t!r}: {why}")
    assert N_RANGE[0] <= len(out) <= N_RANGE[1], f"case count {len(out)} outside {N_RANGE}"
    cells = Counter((m[0], m[1]) for m in meta)
    thin = [(p, s, cells[(p, s)]) for p in PERSONAS for s in STYLES if cells[(p, s)] < MIN_CELL]
    assert not thin, f"persona x style cells below {MIN_CELL}: {thin}"
    assert all(r["split"] == "test" and SET_TAG in r["tags"] for r in out)
    unused = sorted(set(R) - {it["key"] for c in kept for it in c["items"]})
    assert not unused, f"registry keys never used by a kept case: {unused}"

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
    name_overlap = sorted(k for k in set(mains) if is_new(k) and _compact(R[k]["names"][0]) in ex_names)
    print(f"MFDS rows asserted against raw CSV: {n_mfds}")
    print(f"written {len(CASES)} specs, dropped {len(dropped)} near-duplicates (listed above)")
    twins = _review_token_twins(rows)
    print(f"REVIEW same-token-set food_db rows not in gold/excl (print only): {twins}")
    print(f"wrote {len(out)} cases -> {OUT_PATH.relative_to(ROOT)}")
    print(f"persona: {dict(Counter(m[0] for m in meta))}")
    print(f"style:   {dict(Counter(m[1] for m in meta))}")
    print(f"domain (main item): {dict(Counter(m[2] for m in meta))}")
    grid = Counter((m[0], m[1]) for m in meta)
    print("grid persona x style: " + "  ".join(f"{p}:" + ",".join(str(grid[(p, s)]) for s in STYLES) for p in PERSONAS))
    print(f"category: {dict(cat)}  ({', '.join(f'{k} {v / len(out):.0%}' for k, v in cat.items())})")
    print(f"label_confidence: {dict(cf)}  low_discrimination items: {ld}")
    print(f"low-confidence ids: {[r['id'] for r in out if r['label_confidence'] == 'low']}")
    print(f"main-item keys new vs existing sets: {new_main}/{len(mains)} = {new_main / max(1, len(mains)):.0%}")
    print(f"all items new: {new_items}/{len(items_all)} = {new_items / max(1, len(items_all)):.0%}; unique keys {len(uniq)}")
    print(f"new-named keys that overlap existing gold/names (not counted new): {renamed}")
    print(f"main keys counted new but whose names[0] appears in existing names: {name_overlap}")
    for t, c in sorted(tg.items()):
        if not t.startswith(("group:", "noise:")):
            print(f"  {t}: {c}")
    print("sha256:", hashlib.sha256(OUT_PATH.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()

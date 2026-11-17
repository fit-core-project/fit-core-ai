"""Build tests/evaluation/datasets/diet_eval_holdout_v7.jsonl (seventh blind held-out Korean diet-log set).

Purpose: a HELD-OUT set written blind to the pipeline's outputs, to measure generalisation of a parser that was tuned on
diet_eval_v1 / diet_eval_products_v1 / diet_eval_holdout_v1..v6. Every case is split "test", tagged set:holdout_v7 plus
persona:<x>, style:<y> and domain:<z> (domain of the main = first required item).

v7 rotates the GENERATOR again; personas and style framings differ from v4 (beginner/lifter/office/elderly/teen x
telegraphic/diary/typo/mixed/voice), v5 (pregnant/diabetic/nightshift/vegan/student/parent x kakao/list/oddnum/
honorific/english/emoji) and v6 (soldier/rider/runner/couple/freelancer/shopowner x dictation/ask/afterthought/review/
banmal/memo); v1-v3 carry no persona/style tags:
  personas (even spread): hiker (등산객: 정상 간식, 하산주, 산채·백숙집), gamer (PC방·밤샘 게이머: PC방 메뉴, 컵라면,
                          과자·음료), roadtrip (국내 여행·휴게소: 휴게소 간식, 지역 시장, 디저트), bodyprofile (바디프로필
                          준비: 생/조리 무게 계량, 치팅데이 디저트), teacher (교사: 급식, 교무실 간식, 회식), fieldworker
                          (현장·농사 근로자: 함바식당, 새참, 막걸리)
  styles (even spread):   receipt (영수증·결제내역 붙여넣기: 가격·수량 혼재), reply (친구 질문에 답하는 대화체 "점심? ~
                          먹었지"), dialect (경상·전라·제주 사투리 "묵었다", "묵었당께"), timeline (시각순 기록 "07:00 ~
                          / 12시 ~"), regret (자책·후회 "~먹어버림.. 망했다"), slang (줄임말·인터넷 은어 "ㄱㄱ", "국룰",
                          "공밥", "된찌")
  cross-style noise:      정정("~아니 ~"), 부정("~는 안 먹음"), 나눠 먹기, 묶음 상품 중 일부(12개입 중 2개), 분수(1/3모,
                          2/3공기), 모호량(쪼매, 몇 개), 가격·시각·회차 등 양이 아닌 숫자(traps)
  content mix:            합성 이름 디저트(앙버터 소금빵, 티라미수 크로플, 바스크치즈케이크, 슈크림 붕어빵, 빵또아 티라미수),
                          맛/브랜드 제품(포카칩 어니언맛, 바나나초코파이, 썬칩 하비스트 체다), 생것/조리 무게 모호성
                          (닭가슴살·고구마·감자·현미·우둔·연어), 음료(막걸리, 차, 라떼, 이온·비타민 음료), 묶음 판매 제품의
                          낱개(초코파이 情 12개입 중 1개, 카스타드, 에너지바, 맥스봉, 팔도비빔면 5개입 중 1봉, 베지밀,
                          진라면 컵), 평범한 집밥·급식·국밥
  domains (main item):    staple, home, restaurant, street, cafe, cvs, snack, fruit, school (급식), highway (휴게소),
                          diet (식단 재료), dessert (베이커리·디저트)

Labels describe what is CORRECT for a real person's meal, never what the code returns. Label sources (only these):
  - data/food_db/food_db_clean.csv          (generic foods: gold_food_ids + kcal/100 of those rows; rows trace to the
                                             MFDS/RDA/NIFS public composition tables in data/food_db/raw)
  - data/food_db/raw/식품의약품안전처_통합식품영양성분정보(가공식품)_20261111.csv
                                             (branded / packaged products: label values, asserted at build)
  - public common-knowledge Korean portion sizes (stated in each registry note / basis = the gram assumption)
engines/*, data/curated/*, processed_product_index.csv, LLM caches, eval harness internals, baselines and eval
reports/results were NOT consulted; the estimator was NOT run on this set.

Raw vs cooked (new emphasis in v7): when the text states the weighing state ("생으로 150g 재서 구웠다", "구운 거 150g")
the gold rows are the rows of THAT state only (생것 rows for raw weight, 구운것/삶은것/찐것 rows for cooked weight);
when the text gives a weight without the state, gold is the union of raw and cooked rows (flag:raw_cooked, medium/low).
Multipack singles (flag:multipack_unit): the unit size is the single piece (label pack weight / piece count), stated in
the basis.

Schema = diet_eval_holdout_v6.jsonl (same conventions as build_diet_holdout_v6.py):
  - names: acceptable food names (incl. English aliases)
  - gold_food_ids: acceptable food_db_clean rows (+ "curated" / "llm_estimate_ok" tokens)
  - amount: literal quantity [lo, hi] (null = no number). Sharing / partial-eating: [per-person share, literal total];
  - units: accepted unit tokens; [] when the text has a bare number without a unit word ("치즈볼 5", "맥스봉 2")
  - grams [lo, hi]: public-standard portion range (ml treated as g); always the amount the person actually ate
  - kcal [lo, hi]: generic = min/max DB kcal/100 x grams, widened 10% and +-5 kcal (solid foods use only their
                   per-100g gold rows when any exist; per-100ml twins stay in gold for matching only);
                   product = min/max MFDS label kcal/100 x single-unit size x quantity, widened 10%
Completeness (asserted at build): every food_db row whose name is IDENTICAL to a gold row's name, and every MFDS row
whose FOOD_NM is identical to a transcribed product name, is either in gold or listed in the key's `excl` dict with a
reason (copied into basis). Rows whose token set equals a gold row's token set are PRINTED for review.
Numbers that are NOT amounts (가격, 시각, 1교시, D-30, 3일차, 500칼로리) are deliberate traps; those cases are medium.
Independence (asserted / enforced at build):
  - near-duplicate texts (bigram Jaccard >= 0.4 vs the eight existing sets, the scratch inputs, or an earlier v7 case)
    are DROPPED before ids are assigned;
  - FOOD-COMBINATION duplicates are DROPPED too: a case whose item set matches, item for item, the item set of any case
    in the existing sets or an earlier v7 case (items match when their non-token gold_food_ids intersect or any of their
    compacted names are equal) is not kept.

Run:  PYTHONUTF8=1 python scripts/eval/build_diet_holdout_v7.py
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
OUT_PATH = DATASETS / "diet_eval_holdout_v7.jsonl"
EXISTING = [DATASETS / "diet_eval_v1.jsonl", DATASETS / "diet_eval_products_v1.jsonl",
            DATASETS / "diet_eval_holdout_v1.jsonl", DATASETS / "diet_eval_holdout_v2.jsonl",
            DATASETS / "diet_eval_holdout_v3.jsonl", DATASETS / "diet_eval_holdout_v4.jsonl",
            DATASETS / "diet_eval_holdout_v5.jsonl", DATASETS / "diet_eval_holdout_v6.jsonl"]
TEMPLATE = DATASETS / "diet_eval_holdout_v6.jsonl"

KCAL_TOL = 0.10
NEAR_JACCARD = 0.4  # cases at or above this vs existing / earlier v7 texts are dropped
KCAL_ABS = 5.0
LLM = "llm_estimate_ok"
CUR = "curated"
G_ML = {"g": (1, 1), "ml": (1, 1)}
SET_TAG = "set:holdout_v7"
ID_PREFIX = "dh7_"
N_RANGE = (130, 150)   # asserted size of the written set
MIN_CELL = 3           # asserted minimum cases per persona x style cell

PERSONAS = ("hiker", "gamer", "roadtrip", "bodyprofile", "teacher", "fieldworker")
STYLES = ("receipt", "reply", "dialect", "timeline", "regret", "slang")
DOMAINS = ("staple", "home", "restaurant", "street", "cafe", "cvs", "snack", "fruit", "school", "highway", "diet",
           "dessert")
STYLE_NOISE = {"receipt": ["영수증"], "reply": ["대화체"], "dialect": ["사투리"], "timeline": ["시각표기"],
               "regret": ["자책"], "slang": ["줄임말"]}
_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
_LATIN_RE = re.compile(r"[A-Za-z]{2,}")


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


GUK = {"그릇": (250, 350), "뚝배기": (250, 350), "bowl": (250, 350), **G_ML}
GUK_NOTE = "국·찌개 1그릇 250-350g"
NAMUL = {"접시": (60, 100), **G_ML}
NAMUL_NOTE = "나물·볶음·무침 반찬 1회 40-70g, 1접시 60-100g"
KIMCHI_NOTE = "김치 반찬 1회 30-60g"
NOODLE = {"그릇": (600, 800), "인분": (600, 800), "bowl": (600, 800), **G_ML}
NOODLE_NOTE = "국물 면 1그릇(국물 포함) 600-800g"
DRINK = {"잔": (350, 500), "cup": (350, 500), **G_ML}
DRINK_NOTE = "카페 음료 1잔(톨~그란데) 350-500ml"
TEA = {"잔": (250, 350), "cup": (250, 350), **G_ML}
TEA_NOTE = "전통차 1잔 250-350ml"


def BAG(g):
    return {"봉": (g, g), "봉지": (g, g), "개": (g, g)}


R = {
    # ---------------- reused staples (definitions copied from build_diet_holdout_v6.py) ----------------
    "rice": F("밥", "staple", ["쌀밥", "밥", "공기밥", "공깃밥", "공밥", "흰쌀밥", "흰밥", "rice"], ["음식_011600", "음식_011599", CUR],
              units={"공기": (190, 230), "그릇": (190, 230), "개": (190, 230), "bowl": (190, 230), **G_ML}, serving=(190, 230),
              note="밥 1공기 210g(식약처 1회 제공량)"),
    "egg": F("계란유제품", "staple", ["계란", "달걀", "삶은계란", "삶은 계란", "삶은 달걀", "egg", "eggs", "boiled eggs"],
             [CUR, "원재_001702", "음식_017044", "원재_001700"], units={"개": (50, 60), "알": (50, 60)}, serving=(50, 60)),
    "americano": F("음료", "staple", ["아메리카노", "아아", "아이스 아메리카노", "아이스아메리카노", "커피", "americano", "iced americano"],
                   ["음식_009004", "음식_008945", "음식_008940", "음식_009005", "음식_008946"], units={"잔": (300, 500)}, serving=(300, 500)),
    "cola": P("음료", "staple", ["콜라", "코카콜라", "coke"],
              [("P109-401040100-0479", 43, "코카콜라"), ("P109-401040100-2381", 44, "코카콜라"), ("P109-401040100-2382", 44, "코카콜라")],
              {"캔": (250, 355), "잔": (200, 300), "개": (250, 355)}, (250, 355),
              "콜라 1캔 250-355ml; P109-401040100-1982(코카-콜라 158/100ml)는 1캔값 오기로 보고 제외"),
    "milk": F("음료", "staple", ["우유", "흰우유", "milk"], ["원재_001728", "음식_010323"],
              units={"잔": (200, 250), "컵": (200, 250), "팩": (200, 200), **G_ML}, serving=(200, 250),
              note="우유 1잔/1컵 200-250ml, 1팩 200ml"),
    "banana": F("과일", "staple", ["바나나", "banana"], ["원재_001183"], units={"개": (100, 130)}, serving=(100, 130),
                note="바나나 1개(가식부) 100-130g; 말린 바나나(원재_001184)는 다른 이름"),

    # ---------------- 식당·산행 (restaurant) ----------------
    "haemul_pajeon": F("빵과자", "restaurant", ["해물파전", "seafood pancake"], ["음식_002287", "음식_007022", "음식_007023"],
                       units={"장": (300, 450), "접시": (300, 450), "조각": (40, 70), **G_ML}, serving=(100, 150),
                       note="해물파전 1장(식당) 300-450g, 1조각 40-70g, 나눠 먹을 때 1인 몫 100-150g; '파전 해물' 행 포함, 간편조리세트 행(음식_002286)은 다른 이름"),
    "dubu_kimchi": F("반찬", "restaurant", ["두부김치", "tofu kimchi"], ["음식_016278", "음식_016279"],
                     units={"접시": (300, 450), **G_ML}, serving=(150, 250),
                     note="두부김치 1접시 300-450g, 1인 몫 150-250g; 돼지고기 부재료 행(음식_016277)은 다른 이름"),
    "ori_baeksuk": F("고기", "restaurant", ["오리백숙", "백숙", "duck baeksuk"], ["음식_010668", "음식_010667"],
                     units={"인분": (400, 600), "그릇": (400, 600), "마리": (1600, 2400), **G_ML}, serving=(400, 600),
                     note="오리백숙 1인분(고기+국물) 400-600g, 1마리(3-4인) 1600-2400g"),
    "hwangtae": F("생선", "restaurant", ["황태구이", "grilled dried pollack"], ["음식_001885", "음식_001884"],
                  units={"마리": (100, 150), "인분": (100, 150), "접시": (100, 150), **G_ML}, serving=(100, 150),
                  note="황태구이 1마리·1인분(양념 포함) 100-150g; '황태구이 양념'(100ml) 행 포함"),
    "sundae_gukbap": F("국찌개", "restaurant", ["순대국밥", "sundae gukbap"], ["음식_017618"],
                       units={"그릇": (600, 800), "뚝배기": (600, 800), "bowl": (600, 800), **G_ML}, serving=(600, 800),
                       note="순대국밥 1그릇(밥·국물 포함) 600-800g; '국밥 순대국밥' 행; 순대국(음식_012916/012917)은 밥 미포함 다른 음식이라 이름·gold 모두 제외"),
    "bibimbap": F("밥", "restaurant", ["비빔밥", "전주비빔밥", "bibimbap"], ["음식_013743", "음식_013742"],
                  units={"그릇": (400, 500), "bowl": (400, 500), **G_ML}, serving=(400, 500), note="비빔밥 1그릇 400-500g"),
    "jeyuk": F("고기", "restaurant", ["제육볶음", "제육", "돼지고기볶음", "spicy stir-fried pork"], ["음식_016370", "음식_010082"],
               units={"인분": (150, 250), "접시": (150, 250), **G_ML}, serving=(150, 250),
               note="제육볶음 1인분 150-250g; '돼지고기볶음(제육볶음)' 100g 행 + 100ml 행; 채소·오징어 부재료 행은 다른 이름"),
    "doenjang": F("국찌개", "restaurant", ["된장찌개", "된찌", "doenjang jjigae"], ["음식_016312", "음식_016311"], units=GUK,
                  serving=(250, 350), note=GUK_NOTE + "; 부재료 표기 행은 다른 이름"),
    "kimchi_mari": F("면", "home", ["김치말이국수", "김치말이 국수"], ["음식_017615"], units=NOODLE, serving=(600, 800),
                     note=NOODLE_NOTE + "; '국수 김치말이국수' 행"),
    "janchi": F("면", "home", ["잔치국수", "멸치국수", "janchi guksu"], ["음식_017610", "음식_010144"], units=NOODLE,
                serving=(600, 800), note=NOODLE_NOTE + "; '국수 잔치국수'(100g) + '잔치국수'(100ml) 행"),
    "eomuk_bokkeum": F("반찬", "home", ["어묵볶음", "오뎅볶음", "stir-fried fish cake"], ["음식_010914", "음식_010913"],
                       units=NAMUL, serving=(40, 70), note=NAMUL_NOTE + "; 채소·매운 변형 행은 다른 이름"),
    "kongnamul": F("반찬", "home", ["콩나물무침", "콩나물 무침", "seasoned bean sprouts"], ["음식_007607", "음식_007606"],
                   units=NAMUL, serving=(40, 70), note=NAMUL_NOTE + "; 부재료 표기 행은 다른 이름"),
    "kkakdugi": F("반찬", "home", ["깍두기", "cubed radish kimchi"], ["음식_017213", "음식_017214"], units=G_ML,
                  serving=(30, 60), note=KIMCHI_NOTE),

    # ---------------- 급식 (school) ----------------
    "curry_rice": F("밥", "school", ["카레라이스", "카레", "카레밥", "curry rice"], ["음식_009766"],
                    units={"접시": (400, 550), "그릇": (400, 550), "인분": (400, 550), **G_ML}, serving=(400, 550),
                    note="카레라이스 1접시(밥+카레) 400-550g; 부재료 표기(100ml) 행은 다른 이름"),
    "miyeokguk": F("국찌개", "school", ["미역국", "seaweed soup"], ["음식_014788", "음식_014787"], units=GUK, serving=(250, 350),
                   note=GUK_NOTE + "; 굴·홍합·참치 등 부재료 행은 다른 이름"),
    "japchae": F("반찬", "school", ["잡채", "japchae"], ["음식_010136"], units={"접시": (100, 150), **G_ML}, serving=(60, 100),
                 note="잡채 반찬 1회 60-100g, 1접시 100-150g; 잡채밥·고추잡채 등은 다른 이름"),
    "hambak": F("고기", "school", ["함박스테이크", "함박", "hamburg steak"], ["음식_002364", "음식_002365"],
                units={"개": (120, 180), "장": (120, 180)}, serving=(120, 180), note="함박스테이크 1개(소스 포함) 120-180g"),
    "gamja_jorim": F("반찬", "school", ["감자조림", "braised potatoes"], ["음식_017956", "음식_017955"], units=NAMUL,
                     serving=(40, 70), note=NAMUL_NOTE),
    "tteokguk": F("국찌개", "school", ["떡국", "rice cake soup"], ["음식_016237"], units={"그릇": (500, 700), "bowl": (500, 700), **G_ML},
                  serving=(500, 700), note="떡국 1그릇(국물 포함) 500-700g; 굴·소고기 등 부재료(100ml) 행은 다른 이름"),

    # ---------------- 분식·PC방 (street) ----------------
    "gimbap": F("밥", "street", ["김밥", "야채김밥", "gimbap"], ["음식_017295", "음식_017294"],
                units={"줄": (200, 250), "알": (15, 25)}, serving=(200, 250), note="김밥 1줄 200-250g, 1알 15-25g"),
    "cheeseball": F("빵과자", "street", ["치즈볼", "cheese ball"], ["음식_009859", "음식_009853", "음식_009854"], units={"개": (20, 30), "알": (20, 30)},
                    serving=(80, 120), note="치즈볼 1개 20-30g; '치즈볼 치즈볼 오리지널'·'치즈볼 치즈볼 6개' 행 포함, 크림치즈·퐁듀 등 변형 행은 다른 이름"),
    "fries": F("빵과자", "street", ["감자튀김", "french fries", "프렌치프라이"], ["음식_017952", "음식_017935", "음식_017951"],
               units={"컵": (100, 150), "접시": (100, 150), "개": (100, 150)}, serving=(100, 150),
               note="감자튀김 1컵·1접시 100-150g; '감자튀김 프렌치 프라이' 행 포함"),
    "dakgangjeong": F("고기", "street", ["닭강정", "sweet crispy chicken"], ["음식_017011"],
                      units={"컵": (150, 250), "인분": (150, 250), **G_ML}, serving=(150, 250),
                      note="닭강정 1컵·1인분 150-250g; DB의 유일한 닭강정 행은 100ml 기준(182)이라 매칭만 허용하고, kcal은 MFDS 가공식품 '닭강정' 품목(소스·전용유 제외, n=401) 사분위 209-300kcal/100g 사용"),

    # ---------------- 휴게소·여행 (highway) ----------------
    "hodu": F("빵과자", "highway", ["호두과자", "walnut cake"], ["음식_006975", "음식_006977", "음식_006978"], units={"개": (13, 18), "알": (13, 18)},
              serving=(40, 90), note="호두과자 1개 13-18g; '팥호두과자(8·20개입)' 행 포함(팥소 기본), 앙버터·레몬 등 맛 변형 행은 다른 이름"),
    "chapssal": F("빵과자", "highway", ["찹쌀도넛", "찹쌀 도넛", "glutinous rice donut"], ["음식_016531", "음식_016539", "음식_016535", "음식_016540", "음식_016537"],
                  units={"개": (50, 80)}, serving=(50, 80), note="찹쌀도넛 1개 50-80g; 띄어쓰기·어순만 다른 '찹쌀 도넛'·'쫄깃한찹쌀 도넛'·'도넛 찹쌀' 행 포함, 미니·크림치즈 행은 다른 이름"),
    "hotdog": F("빵과자", "highway", ["핫도그", "hot dog", "corn dog"], ["음식_002362", "음식_002307"], units={"개": (90, 130)},
                serving=(90, 130), note="핫도그 1개 90-130g; '핫도그 핫도그' 행 포함, 맛 변형 행은 다른 이름"),
    "udon": F("면", "highway", ["우동", "휴게소 우동", "udon"], ["음식_010357"], units=NOODLE, serving=(600, 800),
              note=NOODLE_NOTE + "; '우동'(100ml) 행; 일식·해물·김치 등 수식어 행은 다른 이름"),
    "shu_bungeo": F("빵과자", "highway", ["슈크림 붕어빵", "슈크림붕어빵", "custard fish bread"], ["음식_007245"],
                    units={"개": (35, 55), "마리": (35, 55)}, serving=(35, 55), note="붕어빵 1개 35-55g; '크림빵 슈크림 붕어빵' 행"),
    "hallabong": F("과일", "fruit", ["한라봉", "hallabong"], ["원재_001132"], units={"개": (200, 300)}, serving=(200, 300),
                   note="한라봉 1개(가식부) 200-300g; '귤 한라봉 생것' 행"),
    "corn": F("곡류", "highway", ["찐옥수수", "옥수수", "찰옥수수", "steamed corn"], ["음식_010574", "원재_000245", "원재_000237"],
              units={"개": (120, 180), "자루": (120, 180)}, serving=(120, 180),
              note="찐옥수수 1자루(가식부) 120-180g; '옥수수 찐옥수수' + 찰·단옥수수 찐것 행"),
    "apple_pie": F("빵과자", "dessert", ["애플파이", "apple pie"], ["음식_007035"], units={"개": (100, 150), "조각": (100, 150)},
                   serving=(100, 150), note="애플파이 1개·1조각 100-150g; '클래식애플파이' 행"),
    "tiramisu_croffle": F("빵과자", "dessert", ["티라미수 크로플", "티라미수크로플", "tiramisu croffle"], ["음식_007295"],
                          units={"개": (90, 140)}, serving=(90, 140), note="크림 올린 크로플 1개 90-140g; '크로플 티라미수 크로플' 행"),
    "injeolmi_bingsu": F("빵과자", "dessert", ["인절미빙수", "인절미 빙수", "injeolmi bingsu"], ["음식_013527"],
                         units={"그릇": (400, 600), "인분": (400, 600), **G_ML}, serving=(400, 600),
                         note="빙수 1그릇 400-600g; '빙수 인절미 빙수' 행"),

    # ---------------- 과일·채소 (fruit) ----------------
    "apple": F("과일", "fruit", ["사과", "부사", "apple"], ["원재_001202", "원재_001203"],
               units={"개": (200, 280), "쪽": (25, 45), "조각": (25, 45)}, serving=(200, 280),
               note="사과 1개(가식부) 200-280g, 1쪽 25-45g; '사과 생것'·'사과 부사 생것' 행"),
    "cucumber": F("채소", "fruit", ["오이", "생오이", "cucumber"], ["원재_000836", "원재_000839"], units={"개": (150, 250)},
                  serving=(150, 250), note="오이 1개 150-250g(생것); 개량종·취청 생것 행"),
    "watermelon": F("과일", "fruit", ["수박", "watermelon"], ["원재_001214"], units={"조각": (200, 350), "쪽": (200, 350)},
                    serving=(200, 350), note="수박 1조각·1쪽(가식부) 200-350g; '수박 적육질 생것' 행"),

    # ---------------- 식단 재료: 생것 vs 조리 (diet) ----------------
    "chicken_raw": F("고기", "diet", ["닭가슴살", "생닭가슴살", "raw chicken breast"], ["원재_001313"], units=G_ML,
                     serving=(100, 150), note="생것 무게 기준 -> '닭고기 가슴(껍질 제거) 생것' 행만"),
    "chicken_cooked": F("고기", "diet", ["닭가슴살", "구운 닭가슴살", "삶은 닭가슴살", "cooked chicken breast"],
                        ["원재_001315", "원재_001314"], units=G_ML, serving=(100, 150),
                        note="조리 후 무게 기준 -> 가슴(껍질 제거) 구운것(팬)·삶은것 행"),
    "chicken_any": F("고기", "diet", ["닭가슴살", "chicken breast"], ["원재_001313", "원재_001314", "원재_001315"], units=G_ML,
                     serving=(100, 150), note="계량 상태 미기재 -> 가슴(껍질 제거) 생것·삶은것·구운것 행 모두"),
    "goguma_steamed": F("곡류", "diet", ["고구마", "찐고구마", "찐 고구마", "steamed sweet potato"],
                        ["원재_000336", "원재_000344", "원재_000347"], units={"개": (150, 250), **G_ML}, serving=(150, 250),
                        note="찐고구마 중 1개 150-250g; 고구마·밤고구마·호박고구마 찐것 행"),
    "goguma_raw": F("곡류", "diet", ["고구마", "생고구마", "raw sweet potato"], ["원재_000334", "원재_000343", "원재_000346"],
                    units=G_ML, serving=(150, 250), note="생것 무게 기준 -> 고구마·밤고구마·호박고구마 생것 행"),
    "brown_rice_raw": F("곡류", "diet", ["현미", "현미쌀", "brown rice (raw)"], ["원재_000074"], units=G_ML, serving=(60, 90),
                        note="조리 전 생쌀 무게 기준 -> '멥쌀 현미 생것' 행"),
    "hyunmi_bap": F("밥", "diet", ["현미밥", "brown rice"], ["음식_001990", "음식_001989", "원재_000152"],
                    units={"공기": (190, 230), "그릇": (190, 230), **G_ML}, serving=(190, 230),
                    note="밥 1공기 190-230g; '멥쌀밥 현미'(RDA 현미밥) 행 포함; '현미밥 쌀'·'순 현미밥' 행은 다른 이름"),
    "salmon_raw": F("생선", "diet", ["연어", "생연어", "연어회", "raw salmon"],
                    ["원재_019427", "원재_019424", "원재_019425", "원재_019426"], units=G_ML, serving=(80, 120),
                    note="생것 무게 기준 -> '연어 육 생것 수입' 행 4개(6월·노르웨이·뉴질랜드·영국, 179-273kcal)",
                    excl={"원재_019428": "양양 8월 자연산(회귀 연어, 106kcal) - 횟감용 수입 양식 연어와 다른 원산지",
                          "원재_019416": "대서양연어 대표 평균(120kcal) - 수입 양식 연어 육 4행(179-273)보다 크게 낮은 별도 대표값이라 범위에서 제외"}),
    "salmon_grilled": F("생선", "diet", ["연어구이", "구운 연어", "연어 스테이크", "grilled salmon"],
                        ["음식_010693", "음식_010692", "원재_019420", "원재_019430"],
                        units={"토막": (120, 180), "조각": (120, 180), **G_ML}, serving=(120, 180),
                        note="연어구이 1토막 120-180g; '연어 양식,육 구운것 수입'·'연어 육 조미한것을 구운것' 원재료 행 포함"),
    "potato_steamed": F("곡류", "diet", ["찐감자", "감자", "삶은 감자", "steamed potato"],
                        ["원재_000320", "원재_000324", "원재_000319", "원재_000323"], units={"개": (130, 200), **G_ML},
                        serving=(130, 200), note="찐감자 중 1개 130-200g; 대지·수미 찐것·삶은것 행"),
    "potato_raw": F("곡류", "diet", ["감자", "생감자", "raw potato"], ["원재_000318", "원재_000322"], units=G_ML,
                    serving=(130, 200), note="생것 무게 기준 -> 대지·수미 생것 행"),
    "broccoli": F("채소", "diet", ["브로콜리", "데친 브로콜리", "broccoli"], ["원재_000714", "원재_000713"], units=G_ML,
                  serving=(50, 100), note="데친·삶은 브로콜리 행; 반찬 1회 50-100g"),
    "danhobak": F("채소", "diet", ["단호박", "찐 단호박", "kabocha"], ["원재_000955"], units={"개": (600, 900), **G_ML},
                  serving=(100, 150), note="찐 단호박 1회(1/4통) 100-150g; '호박 단호박 찐것' 행"),
    "egg_white": F("계란유제품", "diet", ["달걀 흰자", "계란 흰자", "흰자", "egg white"], ["원재_001704", "원재_001703"],
                   units={"개": (30, 35), "알": (30, 35)}, serving=(30, 35), note="달걀 1개분 흰자 30-35g; 난백 삶은것·생것 행"),
    "tofu": F("콩", "diet", ["두부", "tofu"], ["원재_019892"], units={"모": (300, 420), **G_ML}, serving=(100, 150),
              note="두부 1모 300-420g"),
    "udun": F("고기", "diet", ["소고기 우둔", "우둔살", "우둔", "beef round"], ["원재_001522", "원재_001612", "원재_001492", "원재_001582"],
              units=G_ML, serving=(100, 150), note="생것 무게 기준 -> 한우 1·2등급 우둔/우둔살 생것 행(등급 미기재)"),

    # ---------------- 치팅·디저트 (dessert) ----------------
    "maratang": F("면", "restaurant", ["마라탕", "malatang"], ["음식_015212", "음식_015211"],
                  units={"그릇": (600, 900), "bowl": (600, 900), **G_ML}, serving=(600, 900),
                  note="마라탕 1그릇(국물 포함) 600-900g; 100g 기준 '마라탕'·간편조리세트 행"),
    "rose_tteok": F("빵과자", "street", ["로제떡볶이", "로제 떡볶이", "rose tteokbokki"], ["음식_016214"],
                    units={"인분": (250, 350), "접시": (250, 350)}, serving=(250, 350),
                    note="떡볶이 1인분 250-350g; '떡볶이 로제떡볶이' 행, 치즈로제(음식_016199)는 다른 이름"),
    "salt_bread": F("빵과자", "dessert", ["소금빵", "salt bread"], ["음식_013043", "음식_013039", "음식_014003"],
                    units={"개": (60, 80)}, serving=(60, 80), note="소금빵 1개 60-80g; '소금빵 소금빵'·'오리지널 소금빵'·'버터빵 소금빵' 행"),
    "ang_salt": F("빵과자", "dessert", ["앙버터 소금빵", "앙버터소금빵", "anbutter salt bread"], ["음식_013042"],
                  units={"개": (90, 130)}, serving=(90, 130), note="앙버터 소금빵 1개(팥·버터 포함) 90-130g; (L) 행은 다른 이름"),
    "basque": F("빵과자", "dessert", ["바스크치즈케이크", "바스크 치즈케이크", "basque cheesecake"], ["음식_008077", "음식_008078"],
                units={"조각": (90, 130), "개": (90, 130)}, serving=(90, 130), note="조각 케이크 1조각 90-130g"),
    "matcha_macaron": F("빵과자", "dessert", ["녹차마카롱", "녹차 마카롱", "말차 마카롱", "matcha macaron"], ["음식_015204", "음식_015206"],
                        units={"개": (15, 25)}, serving=(15, 25), note="마카롱 1개 15-25g; '마카롱 녹차마카롱'·'마카롱 녹차 마카롱' 행"),
    "levain": F("빵과자", "dessert", ["르뱅쿠키", "르뱅 쿠키", "levain cookie"], ["음식_013646"], units={"개": (70, 110)},
                serving=(70, 110), note="르뱅쿠키 1개 70-110g; '클래식 르뱅쿠키' 행"),
    "roll_cake": F("빵과자", "dessert", ["롤케이크", "롤 케이크", "roll cake"], ["음식_008131"], units={"조각": (50, 80), "개": (50, 80)},
                   serving=(50, 80), note="롤케이크 1조각 50-80g"),
    "danpat": F("빵과자", "dessert", ["단팥빵", "red bean bread"], ["음식_007012"], units={"개": (80, 120)}, serving=(80, 120),
                note="단팥빵 1개 80-120g; '팥빵 단팥빵' 행"),

    # ---------------- 음료 (cafe) ----------------
    "daechu_tea": F("음료", "cafe", ["대추차", "jujube tea"], ["음식_016742", "음식_016740", "음식_016741", "음식_016743", "음식_016744", "음식_016745"], units=TEA,
                    serving=(250, 350), note=TEA_NOTE + "; '대추차 대추차 핫/아이스' 행들(온도 미기재)"),
    "ginger_tea": F("음료", "cafe", ["생강차", "ginger tea"], ["음식_013136", "음식_013134", "음식_013135"], units=TEA, serving=(250, 350),
                    note=TEA_NOTE + "; 핫·아이스 행(온도 미기재); 유자·레몬 변형 행은 다른 이름"),
    "milk_tea": F("음료", "cafe", ["밀크티", "milk tea"], ["음식_014683", "음식_015838", "음식_015837"], units=DRINK, serving=(350, 500),
                  note=DRINK_NOTE + "; '밀크티/버블티 밀크티'·'라떼 밀크티'·'라떼 밀크티 라떼' 행; 흑당·타로 등 변형은 다른 이름"),
    "strawberry_latte": F("음료", "cafe", ["딸기라떼", "딸기 라떼", "strawberry latte"], ["음식_015956", "음식_015978", "음식_015979"], units=DRINK,
                          serving=(350, 500), note=DRINK_NOTE + "; '라떼 딸기라떼'·'라떼 딸기 라떼' 아이스 행"),
    "heukimja_latte": F("음료", "cafe", ["흑임자라떼", "흑임자 라떼", "black sesame latte"], ["음식_015363", "음식_015364"],
                        units=DRINK, serving=(350, 500), note=DRINK_NOTE),
    "espresso": F("음료", "cafe", ["에스프레소", "espresso"], ["음식_008892", "음식_008893", "음식_008870", "음식_008868"],
                  units={"잔": (30, 40), "샷": (30, 40), **G_ML}, serving=(30, 40), note="에스프레소 1샷 30-40ml; '커피 에스프레소'·핫·핫(싱글) 행"),
}

# ---------------- 브랜드 제품 (MFDS 가공식품 라벨; 각 행은 빌드 시 원본 CSV와 대조) ----------------
PR = {
    "makgeolli": P("음료", "restaurant", ["막걸리", "makgeolli"],
                   [("P115-101010100-0006", 63, "막걸리"), ("P115-101010100-0007", 56, "막걸리"), ("P115-101010100-0008", 63, "막걸리"),
                    ("P115-101010100-0001", 54, "막걸리(알코올 6)"), ("P115-101010100-0013", 43, "장수생막걸리"),
                    ("P115-101010100-0011", 41, "월매 막걸리"), ("P115-101010100-0012", 42, "월매 쌀막걸리")],
                   {"병": (750, 750), "잔": (150, 250), "사발": (200, 300)}, (200, 300),
                   "막걸리 1병 750ml, 1잔 150-250ml, 1사발 200-300ml; 제조사 미기재 '막걸리' 4행 + 대표 생막걸리(장수·월매) 범위 41-63kcal/100ml; 홍삼 등 맛 변형은 다른 이름"),
    "energybar": P("빵과자", "snack", ["오리온 에너지바", "에너지바", "energy bar"],
                   [("P101-106000100-7091", 470, "에너지바"), ("P101-106000100-6354", 470, "에너지바"),
                    ("P101-106000100-1231", 470, "에너지바")], {"개": (40, 40), "봉": (40, 40)}, (40, 40),
                   "1개 40g; 600g 행은 15개입 멀티팩(낱개 40g)"),
    "lipton_peach": P("음료", "cvs", ["립톤 아이스티 복숭아", "립톤아이스티복숭아", "립톤 복숭아"],
                      [("P109-102010200-0155", 30, "립톤아이스티복숭아")], {"병": (500, 500), "개": (500, 500)}, (500, 500),
                      "1병 500ml"),
    "yeonyang": P("빵과자", "snack", ["연양갱", "양갱"], [("P101-204000200-0081", 291, "연양갱"), ("P101-204000200-0139", 291, "해태연양갱"),
                   ("P101-204000200-0497", 282, "해태연양갱"), ("P101-204000200-0465", 290, "해태연양갱")], {"개": (55, 55)}, (55, 55),
                  "1개 55g(5개입 묶음 판매의 낱개; 해태연양갱 100g 행은 kcal/100만 반영)"),
    "jayu": P("빵과자", "snack", ["자유시간", "자유시간 초코바"], [("P103-101020500-0733", 478, "자유시간")], {"개": (36, 36)},
              (36, 36), "1개 36g; 로얄밀크티·Mini 등 변형은 다른 이름"),
    "jin_cup": P("면", "cvs", ["진라면 컵 순한맛", "진라면컵 순한맛", "진라면 컵", "진라면컵"],
                 [("P108-003000400-0245", 431, "진라면Cup순한맛"), ("P108-003000400-0995", 431, "진라면Cup순한맛")],
                 {"개": (65, 65), "컵": (65, 65)}, (65, 65), "작은 컵 1개 65g(6개입 묶음의 낱개)"),
    "choco_pie": P("빵과자", "snack", ["오리온 초코파이", "초코파이 정", "초코파이", "choco pie"],
                   [("P103-103020500-0053", 438, "초코파이정"), ("P101-412000400-1294", 438, "초코파이 정")],
                   {"개": (39, 39)}, (39, 39), "1개 39g(12개입 468g / 30개입 1170g의 낱개)"),
    "banana_choco_pie": P("빵과자", "snack", ["바나나 초코파이", "바나나초코파이", "초코파이 바나나"],
                          [("P103-103020500-0022", 465, "바나나초코파이정"), ("P103-103020500-0119", 465, "바나나초코파이정")],
                          {"개": (37, 37)}, (37, 37), "1개 37g(12개입 444g의 낱개)"),
    "castard": P("빵과자", "snack", ["카스타드", "castard"],
                 [("P101-103000400-1512", 443, "카스타드"), ("P101-103000400-1511", 457, "카스타드"),
                  ("P101-103000400-2740", 443, "카스타드"), ("P101-106000100-6376", 457, "카스타드"),
                  ("P101-106000100-7111", 457, "카스타드")], {"개": (23, 23)}, (23, 23),
                 "1개 23g(12개입 276g의 낱개; 552g·828g 행은 24·36개입); 오리온·롯데 동명 5행 범위"),
    "candybar": P("빵과자", "snack", ["빙그레 캔디바", "캔디바"],
                  [("P102-104010400-0641", 100, "빙그레 캔디바"), ("P102-104010400-0101", 100, "빙그레 캔디바")],
                  {"개": (75, 75)}, (75, 75), "1개 75ml; 메론소다 등 변형은 다른 이름"),
    "watermelon_bar": P("빵과자", "snack", ["잘익은수박바", "수박바"],
                        [("P102-301030000-0918", 120, "잘익은수박바"), ("P102-301030000-0253", 120, "잘익은수박바")],
                        {"개": (75, 80)}, (75, 80), "1개 75g/80ml(동명 2행)"),
    "hodumaru": P("빵과자", "snack", ["해태 호두마루", "호두마루 바", "호두마루"],
                  [("P102-103010300-1273", 164, "해태아이스 호두마루"), ("P102-103010300-1124", 164, "해태아이스 호두마루"),
                   ("P102-103010300-0557", 164, "해태아이스 호두마루"), ("P102-103010300-0617", 164, "해태아이스 호두마루")],
                  {"개": (70, 70)}, (70, 70), "바 1개 70ml; 홈타입 호두마루(540-660ml)는 다른 이름"),
    "ssamanco": P("빵과자", "snack", ["참붕어싸만코", "붕어싸만코", "싸만코"],
                  [("P102-101010500-0152", 140, "참붕어싸만코"), ("P102-101010500-0053", 140, "참붕어싸만코")],
                  {"개": (150, 150)}, (150, 150), "1개 150ml"),
    "bbangttoa_tiramisu": P("빵과자", "snack", ["빵또아 티라미수", "빵또아"], [("P102-103010300-0808", 116, "빵또아 티라미수")],
                            {"개": (180, 180)}, (180, 180), "1개 180ml"),
    "sunchip_cheddar": P("빵과자", "snack", ["썬칩 하비스트 체다", "썬칩", "sunchips harvest cheddar"],
                         [("P101-104000100-2998", 482, "썬칩 하비스트 체다 통곡물 스낵 (42.5g)"),
                          ("P101-104000100-2997", 497, "썬칩 하비스트 체다 통곡물 스낵 (184.2g)"),
                          ("P101-104000100-1420", 497, "썬칩 하비스트 체다 통곡물 스낵")],
                         {"봉": (42.5, 42.5), "봉지": (42.5, 42.5), "개": (42.5, 42.5)}, (42.5, 42.5),
                         "작은 봉지 1봉 42.5g; 184.2g·무표기 행은 kcal/100만 반영"),
    "pokachip_onion": P("빵과자", "snack", ["포카칩 어니언맛", "포카칩 어니언", "포카칩어니언맛"],
                        [("P101-106000100-1874", 570, "포카칩어니언맛")], BAG(66), (66, 66), "1봉 66g"),
    "jetty_drink": P("음료", "cvs", ["제티 쵸코렛맛 드링크", "제티 드링크", "제티"], [("P109-801080100-1294", 49, "제티쵸코렛맛드링크")],
                     {"개": (175, 175), "병": (175, 175)}, (175, 175), "1병 175ml"),
    "ssakssak": P("음료", "cvs", ["쌕쌕 오렌지", "쌕쌕오렌지", "쌕쌕"],
                  [("P109-303030300-0622", 38, "쌕쌕오렌지"), ("P109-303030300-0621", 36, "쌕쌕오렌지")],
                  {"캔": (238, 238), "개": (238, 238)}, (238, 238), "1캔 238ml"),
    "vegemil_b": P("음료", "cvs", ["베지밀 B", "베지밀비", "베지밀", "달콤한 베지밀비"], [("P109-500050200-0024", 63, "달콤한 베지밀비")],
                   {"팩": (190, 190), "개": (190, 190)}, (190, 190), "1팩 190ml(16·24팩 묶음의 낱개)"),
    "vitaminwater": P("음료", "cvs", ["비타민워터 파워씨", "글라소 비타민워터", "비타민워터"],
                      [("P109-801080100-0185", 18, "글라소비타민워터power-c"), ("P109-801080100-5346", 18, "글라소 비타민워터 power-c")],
                      {"병": (500, 500), "개": (500, 500)}, (500, 500), "1병 500ml(power-c)"),
    "paldo_bibim": P("면", "cvs", ["팔도비빔면", "팔도 비빔면", "비빔면"],
                     [("P108-009000400-0216", 404, "팔도비빔면"), ("P108-009000400-0201", 404, "팔도비빔면"),
                      ("P108-009000400-0200", 404, "팔도비빔면"), ("P108-009000400-0084", 408, "팔도비빔면")],
                     BAG(130), (130, 130), "1봉 130g(650g 행은 5개입 묶음)"),
    "maxbong": P("고기", "cvs", ["맥스봉 오리지널", "맥스봉", "maxbong"], [("P120-100010500-0033", 186, "맥스봉오리지널"), ("P120-100010500-0031", 171, "맥스봉"),
                  ("P120-100010500-0032", 148, "맥스봉 오리지널(어린이기호식품)"), ("P120-100010500-0096", 143, "맥스봉 더오리지널")],
                 {"개": (35, 35)}, (35, 35), "1개 35g(525g·560g 묶음 = 15·16개의 낱개); 오리지널 계열 4행 143-186kcal/100g"),
    "cream_bread": P("빵과자", "dessert", ["삼립 정통크림빵", "정통크림빵", "크림빵"],
                     [("P101-430000400-0966", 383, "정통크림빵"), ("P101-430000400-0885", 375, "정통크림빵")],
                     {"개": (75, 75)}, (75, 75), "1개 75g(600g 행은 8개입 묶음)"),
    "whitegold": P("음료", "cvs", ["맥심 화이트골드", "화이트골드 커피믹스", "화이트골드", "커피믹스"],
                   [("P109-202020000-0636", 417, "Maxim 화이트골드 커피믹스"), ("P109-202020000-0025", 427, "Maxim 화이트골드 커피믹스")],
                   {"봉": (11.7, 11.7), "개": (11.7, 11.7), "잔": (11.7, 11.7)}, (11.7, 11.7),
                   "스틱 1봉 11.7g(대용량 박스의 낱개; 1잔 = 1봉 타서)"),
}
assert not set(R) & set(PR)
R.update(PR)
EXTRA_DENS: dict = {}  # v7: no density outside DB/MFDS rows
# Label audit: solid foods whose only food_db rows are per-100ml take kcal/100g from MFDS processed-food labels instead
# (the ml rows stay in gold_food_ids for matching only).  Values = interquartile range of the MFDS rows named for the dish.
EXTRA_DENS_G = {"dakgangjeong": (209.0, 300.0, "MFDS 가공식품 FOOD_NM에 '닭강정' 포함 행(소스·전용유 제외, n=401) 사분위 209-300kcal/100g")}

MULTIPACK = {"energybar", "yeonyang", "jin_cup", "choco_pie", "banana_choco_pie", "castard", "vegemil_b", "paldo_bibim",
             "maxbong", "cream_bread", "whitegold", "candybar"}
COMPOUND = {"ang_salt", "tiramisu_croffle", "basque", "shu_bungeo", "matcha_macaron", "bbangttoa_tiramisu",
            "injeolmi_bingsu", "heukimja_latte", "strawberry_latte", "banana_choco_pie", "apple_pie", "levain",
            "roll_cake", "hodu", "chapssal"}
RAW_COOKED_KEYS = {"chicken_raw", "chicken_cooked", "chicken_any", "goguma_raw", "brown_rice_raw", "salmon_raw",
                   "potato_raw", "udun"}


# ---------------------------------------------------------------------------
# Case DSL
# ---------------------------------------------------------------------------
def I(key, amt=None, unit=None, grams=None, optional=False, units_ok=None, note=None, low=False, gunit=None):
    """gunit = implied unit for a bare number without a unit word ("치즈볼 5"): used for grams only, units -> []."""
    assert unit is None or gunit is None
    return dict(key=key, amt=amt, unit=unit, grams=grams, optional=optional, units_ok=units_ok, note=note, low=low,
                gunit=gunit)


CASES: list[dict] = []

# Product keys -> tokens that identify the transcribed SKU in the text. A product item whose text carries none of its
# tokens is an UNBRANDED mention labelled against one representative SKU: the case is capped at medium confidence and
# tagged flag:unbranded. None = exempt (label already spans every same-name SKU row: 막걸리, cola).
BRAND_TOKENS = {
    "cola": None, "makgeolli": None, "energybar": ["에너지바"], "lipton_peach": ["립톤"], "yeonyang": ["연양갱"],
    "jayu": ["자유시간"], "jin_cup": ["진라면"], "choco_pie": ["초코파이 정", "초코파이정", "오리온 초코파이", "情"],
    "banana_choco_pie": ["바나나 초코파이", "바나나초코파이"], "castard": ["카스타드"], "candybar": ["캔디바"],
    "watermelon_bar": ["수박바"], "hodumaru": ["호두마루"], "ssamanco": ["싸만코"], "bbangttoa_tiramisu": ["빵또아"],
    "sunchip_cheddar": ["썬칩"], "pokachip_onion": ["포카칩"], "jetty_drink": ["제티"], "ssakssak": ["쌕쌕"],
    "vegemil_b": ["베지밀"], "vitaminwater": ["비타민워터"], "paldo_bibim": ["팔도"], "maxbong": ["맥스봉"],
    "cream_bread": ["정통크림빵"], "whitegold": ["화이트골드"],
}
UNBRANDED = "flag:unbranded"

SHARE = "flag:share"
SIZE = "flag:size"
PROD = "flag:product"
RANGE = "flag:range"
PART = "flag:partial"
NEG = "flag:negation"
CORR = "flag:correction"
TRAP = "flag:non_amount_number"
LEFT = "flag:leftover"
RAWC = "flag:raw_cooked"
MULTI = "flag:multipack_unit"
CMPD = "flag:compound_name"
VAGUE = "모호량"


def C(text, items, persona, style, noise=(), tags=(), conf="high", note=None, max_items=None, min_items=None):
    assert persona in PERSONAS, persona
    assert style in STYLES, style
    low_text = text.lower()
    tags = list(tags)
    keys = {it["key"] for it in items}
    if any(R[k]["kind"] == "product" for k in keys) and PROD not in tags:
        tags.append(PROD)
    if keys & MULTIPACK:
        tags.append(MULTI)
    if keys & COMPOUND:
        tags.append(CMPD)
    if keys & RAW_COOKED_KEYS and RAWC not in tags:
        tags.append(RAWC)
    unbranded = []
    for it in items:
        if R[it["key"]]["kind"] != "product":
            continue
        toks = BRAND_TOKENS[it["key"]]
        if toks is not None and not any(t.lower() in low_text for t in toks):
            unbranded.append(R[it["key"]]["names"][0])
    if unbranded:
        tags.append(UNBRANDED)
        if conf == "high":
            conf = "medium"
        extra = "브랜드 미기재 -> 대표 SKU 라벨: " + ", ".join(unbranded)
        note = f"{note}; {extra}" if note else extra
    if TRAP in tags and conf == "high":
        conf = "medium"
    nz = list(STYLE_NOISE[style]) + [n for n in noise if n not in STYLE_NOISE[style]]
    if _EMOJI_RE.search(text) and "이모지" not in nz:
        nz.append("이모지")
    if _LATIN_RE.search(text) and "영어혼용" not in nz:
        nz.append("영어혼용")
    if NEG in tags and "부정" not in nz:
        nz.append("부정")
    if CORR in tags and "정정" not in nz:
        nz.append("정정")
    CASES.append(dict(text=text, items=items, persona=persona, style=style, noise=nz, extra_tags=tags,
                      conf=conf, note=note, max_items=max_items, min_items=min_items))


HK, GM, TR, BP, TC, FW = PERSONAS
RC, RP, DL, TL, RG, SL = STYLES

# ============================== hiker (등산객) ==============================
C("[산행 매점 영수증] 오리온 에너지바 2개 3,000원 / 립톤 아이스티 복숭아 1병 2,200원 → 정상에서 다 먹음",
  [I("energybar", 2, "개"), I("lipton_peach", 1, "병")], HK, RC, tags=[TRAP], note="가격은 양이 아님")
C("등산로 입구 분식 결제내역: 김밥 1줄 3,500원, 막걸리 1병 4,000원 (막걸리는 셋이 나눠 마심)",
  [I("gimbap", 1, "줄"), I("makgeolli", (1 / 3, 1), "병", grams=(250, 250))], HK, RC, tags=[TRAP, SHARE],
  note="막걸리 1병 750ml을 셋이 나눔 -> 250ml")
C("하산주 계산서) 오리백숙 1마리 65,000원 4명 / 막걸리 2병 → 나는 백숙 1인분이랑 막걸리 두 사발",
  [I("ori_baeksuk", 1, "인분"), I("makgeolli", 2, "사발")], HK, RC, tags=[TRAP], note="1마리·4명·2병은 테이블 전체 수량")
C("산 아래 편의점 영수증: 연양갱 1 1,500 / 자유시간 1 1,200 / 사과 1 2,000 — 능선에서 간식으로",
  [I("yeonyang", 1, gunit="개"), I("jayu", 1, gunit="개"), I("apple", 1, gunit="개")], HK, RC, tags=[TRAP])
C("ㅇㅇ 정상 찍고 내려와서? 황태구이 정식 먹었지. 황태구이 1인분에 공깃밥 하나",
  [I("hwangtae", 1, "인분"), I("rice", 1, gunit="개")], HK, RP)
C("점심? 산에서 오이 하나랑 찐옥수수 한 자루 먹은 게 다야", [I("cucumber", 1, gunit="개"), I("corn", 1, "자루")], HK, RP)
C("응 막걸리는 안 마셨어ㅋㅋ 대신 두부김치 반 접시랑 대추차 한 잔", [I("dubu_kimchi", 0.5, "접시"), I("daechu_tea", 1, "잔")], HK, RP,
  tags=[NEG])
C("뭐 먹었냐고? 대피소에서 진라면 컵 순한맛 하나 끓여먹고 오리온 초코파이 하나",
  [I("jin_cup", 1, gunit="개"), I("choco_pie", 1, gunit="개")], HK, RP)
C("산 타고 내리와가 해물파전 한 장 시키가 서이서 노나 묵었다 아이가",
  [I("haemul_pajeon", (1 / 3, 1), "장", grams=(100, 150))], HK, DL, tags=[SHARE], conf="medium",
  note="파전 1장 300-450g을 셋이 나눔 -> 100-150g")
C("정상서 사과 반쪽허고 연양갱 한 개 묵었당께", [I("apple", 0.5, gunit="개"), I("yeonyang", 1, "개")], HK, DL,
  note="'반쪽' = 사과 1/2개")
C("등산 끝나고 오리백숙 한 마리 너이서 묵고 국물에 밥꺼정 쪼매 말아 묵었심더",
  [I("ori_baeksuk", (0.25, 1), "마리", grams=(400, 600)), I("rice", grams=(70, 120), note="'쪼매' 소량(1/3-1/2공기) 추정")],
  HK, DL, noise=[VAGUE], tags=[SHARE], conf="medium", note="백숙 1마리 1600-2400g을 넷이 나눔 -> 400-600g")
C("아따 산에서 김밥 한 줄 하고 오이 하나 깨물어 묵었지라", [I("gimbap", 1, "줄"), I("cucumber", 1, gunit="개")], HK, DL)
C("06:30 바나나 1개 / 09:00 오리온 에너지바 1개 / 12:40 김밥 반 줄",
  [I("banana", 1, "개"), I("energybar", 1, "개"), I("gimbap", 0.5, "줄")], HK, TL, tags=[TRAP], note="시각은 양이 아님")
C("8시 출발 전 찐감자 2개, 11시 정상에서 오이 1개, 3시 하산 후 황태구이 1인분",
  [I("potato_steamed", 2, "개"), I("cucumber", 1, "개"), I("hwangtae", 1, "인분")], HK, TL, tags=[TRAP])
C("아침 7시: 삶은 계란 2개 / 오후 1시: 진라면 컵 하나 + 자유시간 1개",
  [I("egg", 2, "개"), I("jin_cup", 1, gunit="개"), I("jayu", 1, "개")], HK, TL, tags=[TRAP])
C("05시 기상 → 생강차 한 잔 / 10시 쉼터에서 한라봉 1개 / 14시 두부김치 1접시 혼자",
  [I("ginger_tea", 1, "잔"), I("hallabong", 1, "개"), I("dubu_kimchi", 1, "접시")], HK, TL, tags=[TRAP])
C("아 등산 가서 칼로리 다 태웠는데 하산주로 막걸리 한 병 혼자 다 마시고 황태구이까지 먹어버림.. 망했다",
  [I("makgeolli", 1, "병"), I("hwangtae")], HK, RG)
C("운동하러 산 갔다가 해물파전 두 조각이랑 백숙 1인분까지 먹어버렸어 ㅠ",
  [I("haemul_pajeon", 2, "조각"), I("ori_baeksuk", 1, "인분")], HK, RG)
C("산 정상에서 초코파이 정 세 개나 까먹음... 하나만 먹으려 했는데", [I("choco_pie", 3, "개")], HK, RG)
C("하산하고 배고파서 찹쌀도넛 2개에 대추차까지 마셔버림, 오늘 등산 의미 없음",
  [I("chapssal", 2, "개"), I("daechu_tea")], HK, RG)
C("산린이 첫 등산 ㅎㅎ 정상 컵라면 국룰이라 진라면컵 순한맛 ㄱㄱ + 연양갱 반개",
  [I("jin_cup"), I("yeonyang", 0.5, "개")], HK, SL, note="컵라면 수량 미기재 -> 1개")
C("하산 후 국밥 ㄱ 순대국밥 한 뚝 클리어 + 깍두기 쫌", [I("sundae_gukbap", 1, gunit="뚝배기"), I("kkakdugi")], HK, SL,
  note="'한 뚝' = 1뚝배기")
C("백패킹 저녁 ㅋㅋ 오리백숙은 무슨, 찐옥수수 2개 + 막걸리 반병 끝", [I("corn", 2, "개"), I("makgeolli", 0.5, "병")], HK, SL,
  tags=[NEG])
C("능선 쉼터에서 자유시간 1개 + 사과 1쪽 ㅇㅈ?", [I("jayu", 1, "개"), I("apple", 1, "쪽")], HK, SL)

# ============================== gamer (PC방 게이머) ==============================
C("PC방 주문내역 / 치즈볼 5개 4,500 / 제티 쵸코 드링크 1 1,800 / 남김없이 먹음",
  [I("cheeseball", 5, "개"), I("jetty_drink", 1, gunit="개")], GM, RC, tags=[TRAP])
C("[PC방 결제] 닭강정 컵 1개 5,500원 + 코카콜라 1캔 2,000원", [I("dakgangjeong", 1, "컵"), I("cola", 1, "캔")], GM, RC,
  tags=[TRAP])
C("주문: 감자튀김 1 / 진라면 컵 1 / 빙그레 캔디바 1 — 총 9,800원",
  [I("fries", 1, gunit="컵"), I("jin_cup", 1, gunit="개"), I("candybar", 1, gunit="개")], GM, RC, tags=[TRAP])
C("PC방 영수증 ▶ 팔도비빔면 1 (3,500) ▶ 삶은계란 추가 1 (1,000)", [I("paldo_bibim", 1, gunit="봉"), I("egg", 1, gunit="개")],
  GM, RC, tags=[TRAP])
C("ㅇㅇ 롤 다섯 판 하면서 포카칩 어니언 한 봉 다 먹었지 뭐", [I("pokachip_onion", 1, "봉")], GM, RP, tags=[TRAP],
  note="'다섯 판'은 게임 횟수")
C("저녁? 그냥 게임하면서 맥스봉 2개랑 바나나맛우유... 아 아니 제티 드링크 마셨다", [I("maxbong", 2, "개"), I("jetty_drink")], GM, RP,
  tags=[CORR], note="바나나맛우유는 정정으로 취소")
C("먹긴 먹었지, 감자튀김 반 컵이랑 치즈볼 3개", [I("fries", 0.5, "컵"), I("cheeseball", 3, "개")], GM, RP)
C("아직 안 먹었냐고? 방금 썬칩 하비스트 체다 작은 봉지 하나 뜯었어", [I("sunchip_cheddar", 1, "봉지")], GM, RP)
C("피시방서 비빔면 하나 비벼 묵고 캔디바 하나 빨았다 아이가", [I("paldo_bibim", 1, gunit="봉"), I("candybar", 1, gunit="개")],
  GM, DL)
C("겜하다가 닭강정 한 컵에 빙그레 캔디바까지 묵어삐맀다", [I("dakgangjeong", 1, "컵"), I("candybar")], GM, DL)
C("밤새 겜 했당께 포카칩 어니언 반 봉이랑 콜라 한 캔 묵었제", [I("pokachip_onion", 0.5, "봉"), I("cola", 1, "캔")], GM, DL)
C("아침부터 피방 와가 진라면 컵 하나에 삶은 달걀 두 개 묵었심더", [I("jin_cup", 1, gunit="개"), I("egg", 2, "개")], GM, DL)
C("21:00 치즈볼 4개 / 23:30 진라면 컵 1개 / 02:00 캔디바 1개",
  [I("cheeseball", 4, "개"), I("jin_cup", 1, "개"), I("candybar", 1, "개")], GM, TL, tags=[TRAP])
C("오후 3시 맥스봉 1개, 저녁 8시 팔도비빔면 1봉, 새벽 1시 썬칩 하비스트 체다 반 봉",
  [I("maxbong", 1, "개"), I("paldo_bibim", 1, "봉"), I("sunchip_cheddar", 0.5, "봉")], GM, TL, tags=[TRAP])
C("13:10 닭강정 컵 반 / 18:40 카레라이스 한 접시 / 22:00 잘익은수박바 2개",
  [I("dakgangjeong", 0.5, "컵"), I("curry_rice", 1, "접시"), I("watermelon_bar", 2, "개")], GM, TL, tags=[TRAP])
C("10시 기상 바나나맛.. 아니 제티 드링크 1병 / 16시 포카칩 어니언 1봉", [I("jetty_drink", 1, "병"), I("pokachip_onion", 1, "봉")],
  GM, TL, tags=[TRAP, CORR])
C("또 밤샘 겜.. 새벽에 진라면 컵 두 개에 맥스봉 세 개 먹음. 몸 망가지는 소리 들린다",
  [I("jin_cup", 2, "개"), I("maxbong", 3, "개")], GM, RG)
C("게임 지고 화나서 닭강정 한 컵이랑 치즈볼 6개 먹어버림 ㅠ", [I("dakgangjeong", 1, "컵"), I("cheeseball", 6, "개")], GM, RG)
C("PC방에서 감자튀김 한 컵 시키고 콜라 두 캔 마심.. 다이어트 3일차인데", [I("fries", 1, "컵"), I("cola", 2, "캔")], GM, RG,
  tags=[TRAP])
C("자제하려 했는데 포카칩 어니언 한 봉에 참붕어싸만코까지 먹음", [I("pokachip_onion", 1, "봉"), I("ssamanco")], GM, RG)
C("피방 국룰 ㅋㅋ 팔도비빔면 1봉 + 치즈볼 4알", [I("paldo_bibim", 1, "봉"), I("cheeseball", 4, "알")], GM, SL)
C("겜하면서 썬칩 하비스트 체다 1봉 순삭 ㄹㅇ + 제티 드링크 1개", [I("sunchip_cheddar", 1, "봉"), I("jetty_drink", 1, "개")], GM, SL)
C("새벽 야식 ㄱㄱ 진라면컵 순한맛 1 + 감자튀김 반컵", [I("jin_cup", 1, gunit="개"), I("fries", 0.5, "컵")], GM, SL)
C("점심 ㄱㄴ? 피방 카레라이스 곱빼기 + 제티 드링크 1개 ㄱ",
  [I("curry_rice", grams=(600, 825), note="곱빼기 = 1접시의 1.5배"), I("jetty_drink", 1, "개")], GM, SL,
  tags=[SIZE], conf="medium")

# ============================== roadtrip (국내 여행·휴게소) ==============================
C("휴게소 결제 내역 :: 호두과자 10개입 1봉 4,000원 / 핫도그 1 3,500원 → 호두과자는 반만 먹고 핫도그는 다",
  [I("hodu", (5, 10), "개", grams=(65, 90)), I("hotdog", 1, gunit="개")], TR, RC, tags=[TRAP, PART],
  note="호두과자 10개 중 절반 = 5개")
C("[영수증] 우동 1 6,500 / 찹쌀도넛 2 3,000 / 덕평휴게소", [I("udon", 1, gunit="그릇"), I("chapssal", 2, gunit="개")], TR, RC,
  tags=[TRAP])
C("카드승인 내역 휴게소 소떡소떡? 아니 핫도그 1개 4,000원, 아이스 아메리카노 1잔 4,500원",
  [I("hotdog", 1, "개"), I("americano", 1, "잔")], TR, RC, tags=[TRAP, CORR], note="소떡소떡은 정정으로 취소")
C("제주 동문시장 계산: 한라봉 3개 10,000원 / 대추차 1잔 5,000원 → 숙소에서 한라봉 1개랑 대추차 마심",
  [I("hallabong", (1, 3), "개", grams=(200, 300)), I("daechu_tea", 1, "잔")], TR, RC, tags=[TRAP, PART])
C("응 지금 휴게소야. 우동 한 그릇 먹고 호두과자 몇 개 집어먹음",
  [I("udon", 1, "그릇"), I("hodu", grams=(40, 90), low=True, note="'몇 개' 3-5개 추정")], TR, RP, noise=[VAGUE], conf="low",
  note="호두과자 개수 모호")
C("아까? 강릉 가는 길에 슈크림 붕어빵 3마리 먹었지", [I("shu_bungeo", 3, "마리")], TR, RP)
C("점심은 순대국밥 한 그릇이었고 디저트는 없었어, 그냥 생강차", [I("sundae_gukbap", 1, "그릇"), I("ginger_tea")], TR, RP,
  tags=[NEG])
C("ㅇㅇ 전주 왔으니까 비빔밥 한 그릇에 콩나물무침 더 달래서 먹었지, 막걸리는 운전해서 패스",
  [I("bibimbap", 1, "그릇"), I("kongnamul", grams=(70, 140), note="'더 달래서' = 반찬 1회분 + 리필 1회분")], TR, RP,
  tags=[NEG])
C("휴게소서 핫도그 하나 사가 설탕 뿌리가 묵고 찹쌀도넛도 한 개 묵었다", [I("hotdog", 1, gunit="개"), I("chapssal", 1, "개")],
  TR, DL)
C("제주 와서 한라봉 두 개 까묵고 귤은 안 묵었수다, 비타민워터 한 병은 마셨고",
  [I("hallabong", 2, "개"), I("vitaminwater", 1, "병")], TR, DL, tags=[NEG])
C("여수 가는 길 휴게소서 우동 한 그릇 후루룩 했지라, 김밥 반 줄도", [I("udon", 1, "그릇"), I("gimbap", 0.5, "줄")], TR, DL)
C("안동서 찜닭 묵을라 캤는데 아이다, 그냥 순대국밥 한 그릇 비웠심더", [I("sundae_gukbap", 1, "그릇")], TR, DL, tags=[CORR],
  note="찜닭은 취소")
C("07:00 휴게소 아메리카노 1잔 → 10:30 호두과자 6개 → 13:00 전주 비빔밥 1그릇",
  [I("americano", 1, "잔"), I("hodu", 6, "개"), I("bibimbap", 1, "그릇")], TR, TL, tags=[TRAP])
C("11시 핫도그 1개 / 15시 슈크림 붕어빵 2개 / 19시 해물파전 2조각",
  [I("hotdog", 1, "개"), I("shu_bungeo", 2, "개"), I("haemul_pajeon", 2, "조각")], TR, TL, tags=[TRAP])
C("아침 8시 찐옥수수 1개, 점심 1시 김치말이국수 1그릇", [I("corn", 1, "개"), I("kimchi_mari", 1, "그릇")], TR, TL, tags=[TRAP])
C("09:20 글라소 비타민워터 1병 / 12:10 우동 반 그릇 / 16:00 찹쌀도넛 1개",
  [I("vitaminwater", 1, "병"), I("udon", 0.5, "그릇"), I("chapssal", 1, "개")], TR, TL, tags=[TRAP])
C("여행 오면 다이어트는 끝이지.. 휴게소 호두과자 한 봉지(10개) 혼자 다 먹고 빵또아 티라미수까지",
  [I("hodu", 10, "개"), I("bbangttoa_tiramisu")], TR, RG)
C("속초 와서 닭강정 한 컵이랑 핫도그까지 먹어버렸다 ㅠㅠ", [I("dakgangjeong", 1, "컵"), I("hotdog")], TR, RG)
C("휴게소마다 멈춰서 애플파이 1개, 슈크림 붕어빵 2개.. 미쳤나봐", [I("apple_pie", 1, "개"), I("shu_bungeo", 2, "개")], TR, RG)
C("제주 3일째, 흑돼지는 참았는데 한라봉 2개랑 인절미빙수 반 그릇 먹음",
  [I("hallabong", 2, "개"), I("injeolmi_bingsu", 0.5, "그릇")], TR, RG, tags=[TRAP, NEG], note="'3일째'는 양이 아님")
C("휴게소 털이 ㄱㄱ 감튀? 노노 슈크림 붕어빵 4마리 + 밀크티 한 잔", [I("shu_bungeo", 4, "마리"), I("milk_tea", 1, "잔")], TR, SL,
  tags=[NEG])
C("강릉 커피거리 ㄱㄱ 에스프레소 1샷 + 티라미수 크로플 1개 ㅇㅈ", [I("espresso", 1, "샷"), I("tiramisu_croffle", 1, "개")], TR,
  SL)
C("여행 마지막날 ㅋㅋ 김밥 1줄 + 쌕쌕 오렌지 1캔 휴게소 국룰", [I("gimbap", 1, "줄"), I("ssakssak", 1, "캔")], TR, SL)
C("부산 왔으면 국밥이지 ㄹㅇ 순대국밥 한 그릇 + 공밥 반", [I("sundae_gukbap", 1, "그릇"), I("rice", 0.5, gunit="공기")], TR, SL)

# ============================== bodyprofile (바디프로필 준비) ==============================
C("샐러드가게 영수증: 연어 포케... 아니 구운 연어 1토막 추가 3,900 / 현미밥 1공기 2,000",
  [I("salmon_grilled", 1, "토막"), I("hyunmi_bap", 1, "공기")], BP, RC, tags=[TRAP, CORR], note="연어 포케는 정정으로 취소")
C("마트 결제 — 닭가슴살(생) 1kg 12,900원 / 오늘 그중 200g 구워서 깍두기 조금이랑 먹음",
  [I("chicken_raw", (200, 1000), "g", grams=(196, 204)), I("kkakdugi")], BP, RC, tags=[TRAP, PART],
  note="생것 1kg 중 200g(생것 무게)을 구워 먹음")
C("치팅데이 영수증 ☞ 마라탕 1그릇 13,500 / 로제떡볶이 1인분 14,000 (둘이서 나눠 먹음)",
  [I("maratang", (0.5, 1), "그릇", grams=(300, 450)), I("rose_tteok", (0.5, 1), "인분", grams=(125, 175))], BP, RC,
  tags=[TRAP, SHARE], note="둘이 반씩")
C("베이커리 결제: 소금빵 2 / 바스크치즈케이크 1조각 / 치팅이니까 다 먹음",
  [I("salt_bread", 2, gunit="개"), I("basque", 1, "조각")], BP, RC)
C("식단? 오늘은 현미 생쌀 70g으로 밥 지어서 먹고 닭가슴살 구운 거 150g",
  [I("brown_rice_raw", 70, "g"), I("chicken_cooked", 150, "g")], BP, RP, note="현미는 생쌀 무게, 닭가슴살은 조리 후 무게")
C("응 아침은 계란 흰자 4개랑 고구마 찐 거 한 개", [I("egg_white", 4, "개"), I("goguma_steamed", 1, "개")], BP, RP)
C("점심 뭐 먹었냐면 소고기 우둔 생으로 150g 재서 구웠고 브로콜리 데친 거 100g",
  [I("udun", 150, "g"), I("broccoli", 100, "g")], BP, RP, note="우둔은 생것 무게")
C("치팅? 했지ㅋㅋ 앙버터 소금빵 하나랑 딸기라떼 한 잔", [I("ang_salt", 1, gunit="개"), I("strawberry_latte", 1, "잔")], BP, RP)
C("닭가슴살 200그램에 오이 하나 묵었다, 생으로 잰 건지 구운 거 잰 건지는 기억 안 난다 아이가",
  [I("chicken_any", 200, "g", low=True), I("cucumber", 1, gunit="개")], BP, DL, conf="low", note="계량 상태 미상 -> 생것·조리 행 모두(106-164kcal/100g)")
C("고구마 생거로 150그램 달아가 쪄 묵고 오이 반 개 아삭아삭 무따", [I("goguma_raw", 150, "g"), I("cucumber", 0.5, "개")], BP, DL,
  note="고구마는 생것 무게 150g을 찜")
C("두부 반 모하고 단호박 찐 거 쪼매 묵었당께", [I("tofu", 0.5, "모"), I("danhobak", grams=(50, 100), note="'쪼매' = 1회분(100-150g)보다 적게")], BP, DL,
  noise=[VAGUE], conf="medium", note="단호박 '쪼매' -> 50-100g")
C("치팅데이라꼬 마라탕 한 그릇 다 비아뿟다", [I("maratang", 1, "그릇")], BP, DL)
C("07:00 오트밀... 말고 현미밥 반 공기 / 12:00 연어구이 1토막 / 19:00 찐 단호박 100g",
  [I("hyunmi_bap", 0.5, "공기"), I("salmon_grilled", 1, "토막"), I("danhobak", 100, "g")], BP, TL, tags=[TRAP, CORR],
  note="오트밀은 정정으로 취소")
C("6시 공복 아메리카노 / 9시 달걀 흰자 3개 + 감자 생것 기준 200g 쪄서 / 13시 닭가슴살 구운 거 120g",
  [I("americano"), I("egg_white", 3, "개"), I("potato_raw", 200, "g"), I("chicken_cooked", 120, "g")], BP, TL, tags=[TRAP],
  note="감자는 생것 무게, 닭가슴살은 조리 후 무게")
C("10:00 생연어 100g / 15:00 그릭요거트 말고 두부 1/3모 / 20:00 찐고구마 1개",
  [I("salmon_raw", 100, "g"), I("tofu", 1 / 3, "모"), I("goguma_steamed", 1, "개")], BP, TL, tags=[TRAP, CORR],
  note="그릭요거트는 정정으로 취소")
C("11시 바스크치즈케이크 한 조각, 4시 녹차 마카롱 2개 — 치팅 타임",
  [I("basque", 1, "조각"), I("matcha_macaron", 2, "개")], BP, TL, tags=[TRAP])
C("바프 D-30인데 소금빵 3개에 밀크티까지 먹어버림.. 나 진짜 왜 이러냐", [I("salt_bread", 3, "개"), I("milk_tea")], BP, RG, tags=[TRAP],
  note="'D-30'은 양이 아님")
C("식단 지키다가 로제떡볶이 1인분에 치즈볼 3개 추가해서 혼자 다 먹었다 ㅠ 내일 공복 유산소 각",
  [I("rose_tteok", 1, "인분"), I("cheeseball", 3, "개")], BP, RG)
C("참다가 터짐.. 녹차 마카롱 3개에 앙버터 소금빵까지", [I("matcha_macaron", 3, "개"), I("ang_salt")], BP, RG)
C("닭가슴살만 먹다가 결국 르뱅쿠키 2개 먹음, 하나에 500칼로리 넘는다던데", [I("levain", 2, "개")], BP, RG, tags=[TRAP],
  note="'500칼로리'는 양이 아님")
C("오운완 후 단백질 보충 ㄱ 계란 흰자 5개 + 바나나 1개", [I("egg_white", 5, "개"), I("banana", 1, "개")], BP, SL)
C("치팅데이 ㄱㄱ 마라탕 반그릇 + 딸기라떼 1잔 ㅎㅎ", [I("maratang", 0.5, "그릇"), I("strawberry_latte", 1, "잔")], BP, SL)
C("식단 ㅇㅈ? 현미밥 1공기 + 소고기 우둔(생) 120g 구움 + 데친 브로콜리 70g",
  [I("hyunmi_bap", 1, "공기"), I("udun", 120, "g"), I("broccoli", 70, "g")], BP, SL)
C("바프 끝 ㅋㅋ 바스크치즈케이크 2조각 + 흑임자라떼", [I("basque", 2, "조각"), I("heukimja_latte")], BP, SL)

# ============================== teacher (교사) ==============================
C("교사 회식 영수증) 해물파전 2장 36,000 / 막걸리 4병 / 나는 파전 3조각 + 두부김치 조금 + 막걸리 2잔",
  [I("haemul_pajeon", 3, "조각"), I("dubu_kimchi", grams=(80, 150), note="'조금' = 1인 몫보다 적게 80-150g"),
   I("makgeolli", 2, "잔")], TC, RC, noise=[VAGUE], tags=[TRAP], note="2장·4병은 테이블 전체 수량")
C("학교 앞 편의점 결제: 정통크림빵 1개 1,500원 / 달콤한 베지밀비 1팩 1,000원",
  [I("cream_bread", 1, "개"), I("vegemil_b", 1, "팩")], TC, RC, tags=[TRAP])
C("교무실 간식 주문서: 카스타드 1박스(12개) → 내가 먹은 건 2개",
  [I("castard", (2, 12), "개", grams=(46, 46))], TC, RC, tags=[PART], note="12개입 중 2개 x 23g")
C("체험학습 휴게소 영수증 — 호두과자 6개 3,000 / 맥심 화이트골드... 아니 아메리카노 1잔",
  [I("hodu", 6, "개"), I("americano", 1, "잔")], TC, RC, tags=[TRAP, CORR], note="화이트골드는 정정으로 취소")
C("점심 급식? 카레라이스 한 접시에 깍두기 조금, 우유는 반만 마셨어",
  [I("curry_rice", 1, "접시"), I("kkakdugi"), I("milk", 0.5, gunit="팩")], TC, RP, note="급식 우유 1팩 200ml의 반")
C("ㅇㅇ 오늘 급식 미역국 한 그릇이랑 잡채 조금, 밥은 반 공기",
  [I("miyeokguk", 1, "그릇"), I("japchae"), I("rice", 0.5, "공기")], TC, RP)
C("교무실에서? 오리온 초코파이 하나랑 맥심 화이트골드 한 봉 타 마셨지",
  [I("choco_pie", 1, gunit="개"), I("whitegold", 1, "봉")], TC, RP)
C("야자 감독하면서 바나나 초코파이 2개랑 롤케이크 한 조각 먹었어 ㅎ",
  [I("banana_choco_pie", 2, "개"), I("roll_cake", 1, "조각")], TC, RP)
C("급식으로 함박스테이크 하나에 감자조림 쪼매 묵었다", [I("hambak", 1, gunit="개"), I("gamja_jorim")], TC, DL, noise=[VAGUE])
C("교무실서 카스타드 두 개 하고 화이트골드 커피 한 잔 타 묵었데이", [I("castard", 2, "개"), I("whitegold", 1, "잔")], TC, DL)
C("오늘 급식 떡국 나왔는디 한 그릇 싹 비우고 잡채도 쪼매 묵었당께", [I("tteokguk", 1, "그릇"), I("japchae")], TC, DL,
  noise=[VAGUE])
C("애들이랑 운동회 끝나고 수박 두 조각이랑 해태 호두마루 하나 묵었제",
  [I("watermelon", 2, "조각"), I("hodumaru", 1, gunit="개")], TC, DL)
C("08:10 화이트골드 1봉 / 12:20 급식 카레라이스 1접시 + 깍두기 / 15:30 초코파이 정 1개",
  [I("whitegold", 1, "봉"), I("curry_rice", 1, "접시"), I("kkakdugi"), I("choco_pie", 1, "개")], TC, TL, tags=[TRAP])
C("1교시 전 베지밀 1팩, 점심 미역국 1그릇 + 함박스테이크 1개, 퇴근길 단팥빵 1개",
  [I("vegemil_b", 1, "팩"), I("miyeokguk", 1, "그릇"), I("hambak", 1, "개"), I("danpat", 1, "개")], TC, TL, tags=[TRAP],
  note="'1교시'는 양이 아님")
C("7시 반 집에서 사과 반 개 / 12시 반 급식 잡채 한 접시 + 밥 2/3공기",
  [I("apple", 0.5, "개"), I("japchae", 1, "접시"), I("rice", 2 / 3, "공기")], TC, TL, tags=[TRAP])
C("오전 10시 카스타드 1개 / 오후 4시 바나나 초코파이 1개 / 저녁 9시 정통크림빵 반 개",
  [I("castard", 1, "개"), I("banana_choco_pie", 1, "개"), I("cream_bread", 0.5, "개")], TC, TL, tags=[TRAP])
C("애들 시험 채점하다 스트레스로 초코파이 정 4개에 베지밀 한 팩까지..", [I("choco_pie", 4, "개"), I("vegemil_b", 1, "팩")],
  TC, RG)
C("급식 떡국에 깍두기 잔뜩 먹고 떡국 한 그릇 더 받음, 총 두 그릇 ㅠ", [I("tteokguk", 2, "그릇"), I("kkakdugi", grams=(60, 120), note="'잔뜩' = 김치 반찬 2회분가량")], TC, RG,
  note="떡국은 총 2그릇")
C("퇴근길에 정통크림빵 2개 사서 차에서 다 먹었다.. 저녁도 먹었는데", [I("cream_bread", 2, "개")], TC, RG)
C("학부모 상담 끝나고 단팥빵 하나에 바나나 초코파이 하나 더.. 왜 그랬지",
  [I("danpat", 1, gunit="개"), I("banana_choco_pie", 1, gunit="개")], TC, RG)
C("급식 ㅇㅈ 함박 1 + 미역국 1그릇 + 깍두기 쬐끔", [I("hambak", 1, gunit="개"), I("miyeokguk", 1, "그릇"), I("kkakdugi")],
  TC, SL)
C("교무실 커피 국룰 화이트골드 2봉 ㄱ", [I("whitegold", 2, "봉")], TC, SL)
C("체육대회 끝 수박 3조각 + 쌕쌕 오렌지 1캔 클리어", [I("watermelon", 3, "조각"), I("ssakssak", 1, "캔")], TC, SL)
C("점심 급식 카레 반 접시 + 잡채 ㅇㅋ", [I("curry_rice", 0.5, "접시"), I("japchae")], TC, SL)

# ============================== fieldworker (현장·농사) ==============================
C("함바식당 장부: 제육볶음 정식 1 (8,000) — 제육 1인분 + 공기밥 2 + 깍두기 리필",
  [I("jeyuk", 1, "인분"), I("rice", 2, gunit="공기"), I("kkakdugi", grams=(60, 120), note="'리필' = 김치 반찬 2회분")],
  FW, RC, tags=[TRAP])
C("새참 영수증 / 잔치국수 2그릇 / 막걸리 1병 / 나는 국수 1그릇이랑 막걸리 반병",
  [I("janchi", 1, "그릇"), I("makgeolli", 0.5, "병")], FW, RC, tags=[TRAP], note="2그릇·1병은 전체 주문 수량")
C("철물점 옆 편의점 결제: 맥스봉 2 + 정통크림빵 1 + 베지밀 1 = 6,300원",
  [I("maxbong", 2, gunit="개"), I("cream_bread", 1, gunit="개"), I("vegemil_b", 1, gunit="팩")], FW, RC, tags=[TRAP])
C("현장 식당 계산서 → 된장찌개 1 / 어묵볶음 반찬 / 공기밥 1 / 7,000원",
  [I("doenjang", 1, gunit="그릇"), I("eomuk_bokkeum"), I("rice", 1, gunit="공기")], FW, RC, tags=[TRAP])
C("밥? 현장에서 순대국밥 한 그릇 때렸지, 새참으론 찐감자 2개", [I("sundae_gukbap", 1, "그릇"), I("potato_steamed", 2, "개")],
  FW, RP)
C("어 오늘은 비 와서 쉬었어. 집에서 김치말이국수 한 그릇 해먹었지", [I("kimchi_mari", 1, "그릇")], FW, RP)
C("점심 함바에서 제육볶음이랑 콩나물무침, 밥 한 공기", [I("jeyuk"), I("kongnamul"), I("rice", 1, "공기")], FW, RP)
C("새참? 막걸리 한 잔이랑 수박 한 조각", [I("makgeolli", 1, "잔"), I("watermelon", 1, "조각")], FW, RP)
C("새참으로 잔치국수 한 그릇 후루룩 하고 김밥 두 알 집어 묵었심더", [I("janchi", 1, "그릇"), I("gimbap", 2, "알")], FW, DL)
C("고추 따다가 참으로 단팥빵 하나 묵고 막걸리 한 사발 걸쳤다 아이가",
  [I("danpat", 1, gunit="개"), I("makgeolli", 1, "사발")], FW, DL)
C("함바서 된장찌개에 제육 쪼매, 밥은 고봉으로 묵었지라",
  [I("doenjang"), I("jeyuk", grams=(75, 125), note="'쪼매' = 1인분의 절반가량"),
   I("rice", grams=(300, 400), note="고봉밥 300-400g")], FW, DL, noise=[VAGUE], conf="medium")
C("땡볕에 일하고 수박 반 통... 아니 두 쪽에 찐감자 한 개 묵었당께",
  [I("watermelon", 2, "쪽"), I("potato_steamed", 1, "개")], FW, DL, tags=[CORR], note="수박 반 통은 정정으로 취소")
C("05:30 화이트골드 2봉 / 10:00 새참 정통크림빵 1개 / 12:00 함바 제육볶음 1인분 + 밥 1공기",
  [I("whitegold", 2, "봉"), I("cream_bread", 1, "개"), I("jeyuk", 1, "인분"), I("rice", 1, "공기")], FW, TL, tags=[TRAP])
C("6시 바나나 1개, 10시 막걸리 1잔, 정오 된장찌개 1그릇+공기밥 1, 3시 수박 1조각",
  [I("banana", 1, "개"), I("makgeolli", 1, "잔"), I("doenjang", 1, "그릇"), I("rice", 1, gunit="공기"),
   I("watermelon", 1, "조각")], FW, TL, tags=[TRAP])
C("새벽 4시 반 진라면 컵 1 / 9시 맥스봉 1 / 13시 김치말이국수 1그릇",
  [I("jin_cup", 1, gunit="개"), I("maxbong", 1, gunit="개"), I("kimchi_mari", 1, "그릇")], FW, TL, tags=[TRAP])
C("7시 순대국밥 반 그릇 / 15시 찐옥수수 2개 / 21시 소주... 말고 쌕쌕 오렌지 1캔",
  [I("sundae_gukbap", 0.5, "그릇"), I("corn", 2, "개"), I("ssakssak", 1, "캔")], FW, TL, tags=[TRAP, CORR],
  note="소주는 정정으로 취소")
C("일 끝나고 술은 안 먹기로 했는데 막걸리 두 병에 두부김치 반 접시.. 내일 또 일 나가야 하는데",
  [I("makgeolli", 2, "병"), I("dubu_kimchi", 0.5, "접시")], FW, RG)
C("힘들어서 당 땡겨 연양갱 2개랑 정통크림빵 1개 연속으로 먹음", [I("yeonyang", 2, "개"), I("cream_bread", 1, "개")], FW, RG)
C("저녁에 제육볶음 2인분 혼자 다 먹음.. 배 터질 듯", [I("jeyuk", 2, "인분")], FW, RG)
C("새참 때 진라면 컵 두 개 먹고 김밥까지 한 줄.. 과식했다", [I("jin_cup", 2, "개"), I("gimbap", 1, "줄")], FW, RG)
C("함바 ㄱㄱ 제육 1인분 + 미역국 + 공밥 2", [I("jeyuk", 1, "인분"), I("miyeokguk"), I("rice", 2, gunit="공기")], FW, SL)
C("새참 ㅇㅈ 막걸리 1사발 + 김치말이국수 반그릇", [I("makgeolli", 1, "사발"), I("kimchi_mari", 0.5, "그릇")], FW, SL)
C("점심 국밥 ㄱ 순대국밥 특 1그릇 + 콩나물무침",
  [I("sundae_gukbap", 1, "그릇", grams=(750, 1000), note="'특' = 보통보다 큰 그릇 750-1000g"), I("kongnamul")], FW, SL,
  tags=[SIZE], conf="medium")
C("쉬는 시간 화이트골드 1봉 + 연양갱 1개 ㄱ", [I("whitegold", 1, "봉"), I("yeonyang", 1, "개")], FW, SL)

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
    ids = [g for g in f["gold"] if g not in (CUR, LLM)]
    for gid in ids:
        assert gid in rows, f"{cid}: unknown food_id {gid}"
    # Label-audit rule (from v6): grams are grams, so a SOLID food with any per-100g gold row takes its kcal from the
    # per-100g rows only (labels follow public per-100g data).  Unioning per-100ml twins with a different value
    # (김치볶음 44/ml vs 106/g, 쌀밥 120/ml vs 166/g, 들깨칼국수 113/ml vs 76/g) made ranges 3-8x wide.  The ml rows
    # stay in gold_food_ids (matching still accepts them).  Drinks (group 음료) keep every row: ml ~ g for liquids.
    ml_skip = []
    if f["group"] != "음료" and any(rows[g]["basis_unit"] == "g" for g in ids):
        ml_skip = [g for g in ids if rows[g]["basis_unit"] != "g"]
    for gid in ids:
        if gid in ml_skip:
            continue
        k = float(rows[gid]["kcal"])
        dens.append(k)
        parts.append(f"{gid}[{rows[gid]['name']}] {k:g}kcal/100{rows[gid]['basis_unit']}")
    if ml_skip:
        parts[-1] += " (kcal은 100g 기준 행만; 100ml 행 " + ",".join(ml_skip) + "은 매칭만 허용)"
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

    # ---- food-combination signatures (existing sets) ----
    # Label audit: items match on ANY shared compacted name (names[0]-only matching let product items and renamed
    # generics such as 초코파이 / 순대국밥 single-item cases slip past v1-v6).
    def _ex_item_sig(it):
        gold = frozenset(g for g in (it.get("gold_food_ids") or []) if g not in (CUR, LLM))
        names = it.get("names") or [it["key"]]
        return gold, frozenset(_compact(n) for n in names)

    def _spec_item_sig(key):
        f = R[key]
        gold = frozenset(g for g in f["gold"] if g not in (CUR, LLM)) if f["kind"] == "generic" else frozenset()
        return gold, frozenset(_compact(n) for n in f["names"])

    def _item_match(a, b):
        return bool(a[0] & b[0]) or bool(a[1] & b[1])

    def _combo_match(sa, sb):
        if len(sa) != len(sb):
            return False
        return all(any(_item_match(x, y) for y in sb) for x in sa) and all(any(_item_match(y, x) for x in sa) for y in sb)

    ex_combos = [(d["text"], [_ex_item_sig(it) for it in d.get("expected_items", [])]) for d in ex_rows
                 if d.get("expected_items")]

    # ---- drop near duplicates (text) and repeated food combinations BEFORE assigning ids ----
    kept, dropped, kept_bg, kept_combo = [], [], [], []
    for spec in CASES:
        text = spec["text"]
        c = _compact(text)
        bg = _bigrams(text)
        combo = [_spec_item_sig(it["key"]) for it in spec["items"]]
        why = None
        for t, sb in ex_combos:
            if _combo_match(combo, sb):
                why = f"food combination repeats existing {t!r}"
                break
        if why is None:
            for t, sb in kept_combo:
                if _combo_match(combo, sb):
                    why = f"food combination repeats earlier v7 {t!r}"
                    break
        if why is not None:
            dropped.append((text, why))
            continue
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
                    why = f"jaccard {j:.2f} vs earlier v7 {t!r}"
                    break
        if why:
            dropped.append((text, why))
            continue
        kept.append(spec)
        kept_bg.append((text, bg))
        kept_combo.append((text, combo))

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
            if it["key"] in EXTRA_DENS_G:
                lo_, hi_, why_ = EXTRA_DENS_G[it["key"]]
                dens = [lo_, hi_]
                parts = [why_ + " (DB 100ml 행 " + ",".join(g for g in f["gold"] if g not in (CUR, LLM)) + "은 매칭만 허용)"]
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
                "low_discrimination": bool(it["low"] or it["key"] in EXTRA_DENS_G or (k_lo > 0 and k_hi / k_lo > 3.0)),
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

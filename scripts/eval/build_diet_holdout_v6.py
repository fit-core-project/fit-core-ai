"""Build tests/evaluation/datasets/diet_eval_holdout_v6.jsonl (sixth blind held-out Korean diet-log set).

Purpose: a HELD-OUT set written blind to the pipeline's outputs, to measure generalisation of a parser that was tuned on
diet_eval_v1 / diet_eval_products_v1 / diet_eval_holdout_v1..v5. Every case is split "test", tagged set:holdout_v6 plus
persona:<x>, style:<y> and domain:<z> (domain of the main = first required item).

v6 rotates the GENERATOR again; personas and style framings differ from v4 (beginner/lifter/office/elderly/teen x
telegraphic/diary/typo/mixed/voice) and v5 (pregnant/diabetic/nightshift/vegan/student/parent x kakao/list/oddnum/
honorific/english/emoji):
  personas (even spread): soldier (군인: 짬밥 식판, PX, 휴가·외출), rider (배달 라이더: 편의점 끼니, 신호 대기 간식),
                          runner (러너: 러닝 전후 보충, 카보로딩), couple (데이트: 나눠 먹기·반반), freelancer (재택: 남은
                          음식, 홈카페, 배달), shopowner (자영업 사장님: 가게에서 대충 끼니, 시장·반찬가게)
  styles (even spread):   dictation (STT 받아쓰기: 어/음/그 뭐냐 필러, 무구두점, 한글 숫자), ask (질문형 "~먹었는데 몇
                          칼로리야?"), afterthought (덧붙임·정정 "아 그리고", "아 아니다 반만"), review (SNS 캡션 #해시태그
                          + 이모지), banmal (평서문 일기 "~먹었다."), memo (전보체 메모 "점심) 해물덮밥1, 총각김치2")
  cross-style noise:      오타(먹엇음, 꿀꽈베기, 휘낭시애), 영어혼용(Cheetos, mango smoothie), 이모지, 분수(2/3공기, 1/2개,
                          삼분의 일), 모호량(조금, 한 입, 몇 개, 한 주먹), 정정·부정, 나눠 먹기·남은 음식
  domains (main item):    home (집밥 반찬·찌개), restaurant, delivery, street (분식·길거리), cafe, cvs (편의점 면·음료),
                          snack (과자·아이스크림), px (군 PX), fruit; staple only for side items

Labels describe what is CORRECT for a real person's meal, never what the code returns. Label sources (only these):
  - data/food_db/food_db_clean.csv          (generic foods: gold_food_ids + kcal/100 of those rows; rows trace to the
                                             MFDS/RDA/NIFS public composition tables in data/food_db/raw)
  - data/food_db/raw/식품의약품안전처_통합식품영양성분정보(가공식품)_20261111.csv
                                             (branded / packaged products: label values, asserted at build)
  - public common-knowledge Korean portion sizes (stated in each registry note / basis = the gram assumption)
engines/*, data/curated/*, processed_product_index.csv, LLM caches, baselines and eval reports were NOT consulted; the
estimator was NOT run on this set.

Schema = diet_eval_holdout_v5.jsonl (same conventions as build_diet_holdout_v5.py):
  - names: acceptable food names (incl. English aliases used in mixed-language cases)
  - gold_food_ids: acceptable food_db_clean rows (+ "curated" / "llm_estimate_ok" tokens)
  - amount: literal quantity [lo, hi] (null = no number). Sharing / partial-eating: [per-person share, literal total];
  - units: accepted unit tokens; [] when the text has a bare number without a unit word ("짜슐랭1")
  - grams [lo, hi]: public-standard portion range (ml treated as g); always the amount the person actually ate
  - kcal [lo, hi]: generic = min/max DB kcal/100 x grams, widened 10% and +-5 kcal (label audit: solid foods use only
                   their per-100g gold rows when any exist; per-100ml twins stay in gold for matching only);
                   product = min/max MFDS label kcal/100 x single-unit size x quantity, widened 10%
Completeness (asserted at build): every food_db row whose name is IDENTICAL to a gold row's name, and every MFDS row
whose FOOD_NM is identical to a transcribed product name, is either in gold or listed in the key's `excl` dict with a
reason (copied into basis). Rows whose token set equals a gold row's token set are PRINTED for review.
Numbers that are NOT amounts (10K, 30km, 오후 4시, 톨사이즈) are deliberate traps; those cases are medium.
Independence (asserted / enforced at build):
  - near-duplicate texts (bigram Jaccard >= 0.4 vs the seven existing sets, the scratch inputs, or an earlier v6 case)
    are DROPPED before ids are assigned;
  - FOOD-COMBINATION duplicates are DROPPED too: a case whose item set matches, item for item, the item set of any case
    in the existing sets or an earlier v6 case (items match when their non-token gold_food_ids intersect or their
    compacted names[0] are equal) is not kept -- so no single-food case repeats an earlier single-food case either.

Run:  PYTHONUTF8=1 python scripts/eval/build_diet_holdout_v6.py
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
OUT_PATH = DATASETS / "diet_eval_holdout_v6.jsonl"
EXISTING = [DATASETS / "diet_eval_v1.jsonl", DATASETS / "diet_eval_products_v1.jsonl",
            DATASETS / "diet_eval_holdout_v1.jsonl", DATASETS / "diet_eval_holdout_v2.jsonl",
            DATASETS / "diet_eval_holdout_v3.jsonl", DATASETS / "diet_eval_holdout_v4.jsonl",
            DATASETS / "diet_eval_holdout_v5.jsonl"]
TEMPLATE = DATASETS / "diet_eval_holdout_v5.jsonl"

KCAL_TOL = 0.10
NEAR_JACCARD = 0.4  # cases at or above this vs existing / earlier v6 texts are dropped
KCAL_ABS = 5.0
LLM = "llm_estimate_ok"
CUR = "curated"
G_ML = {"g": (1, 1), "ml": (1, 1)}
SET_TAG = "set:holdout_v6"
ID_PREFIX = "dh6_"
N_RANGE = (130, 150)   # asserted size of the written set
MIN_CELL = 3           # asserted minimum cases per persona x style cell

PERSONAS = ("soldier", "rider", "runner", "couple", "freelancer", "shopowner")
STYLES = ("dictation", "ask", "afterthought", "review", "banmal", "memo")
DOMAINS = ("staple", "home", "restaurant", "delivery", "street", "cafe", "cvs", "snack", "px", "fruit")
STYLE_NOISE = {"dictation": ["필러"], "ask": ["질문형"], "afterthought": ["덧붙임"], "review": ["해시태그"], "banmal": [],
               "memo": ["전보체"]}
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


BAP_NOTE = "밥 1공기 190-230g(식약처 1회 제공량 210g)"
GUK = {"그릇": (250, 350), "뚝배기": (250, 350), "bowl": (250, 350), **G_ML}
GUK_NOTE = "국·찌개 1그릇 250-350g"
NAMUL = {"접시": (60, 100), **G_ML}
NAMUL_NOTE = "나물·볶음·무침 반찬 1회 40-70g, 1접시 60-100g"
DRY_NOTE = "마른반찬·장아찌류 1회 30-50g"
KIMCHI_NOTE = "김치 반찬 1회 30-60g"
DEOP = {"그릇": (400, 500), "개": (400, 500), "인분": (400, 500), "bowl": (400, 500), **G_ML}
DEOP_NOTE = "덮밥 1그릇(밥+토핑) 400-500g"
JEONGOL = {"인분": (300, 450), "그릇": (300, 450), "냄비": (900, 1400), **G_ML}
JEONGOL_NOTE = "전골 1인분(국물 포함) 300-450g, 1냄비(2-3인) 900-1400g"
NOODLE = {"그릇": (600, 800), "인분": (600, 800), "bowl": (600, 800), **G_ML}
NOODLE_NOTE = "국물 면 1그릇(국물 포함) 600-800g"
DRINK = {"잔": (350, 500), "cup": (350, 500), **G_ML}
DRINK_NOTE = "카페 음료 1잔(톨~그란데) 350-500ml"
PASTA = {"접시": (300, 400), "인분": (300, 400), **G_ML}
PASTA_NOTE = "파스타 1인분 300-400g"


def BAG(g):
    return {"봉": (g, g), "봉지": (g, g), "개": (g, g)}


R = {
    # ---------------- reused staples (definitions copied from build_diet_holdout_v5.py; side items only) ----------------
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
    # new staples (v6)
    "milk": F("음료", "staple", ["우유", "흰우유", "milk"], ["원재_001728", "음식_010323"],
              units={"잔": (200, 250), "컵": (200, 250), "팩": (200, 200), **G_ML}, serving=(200, 250),
              note="우유 1잔/1컵 200-250ml, 1팩 200ml"),
    "banana": F("과일", "staple", ["바나나", "banana"], ["원재_001183"], units={"개": (100, 130)}, serving=(100, 130),
                note="바나나 1개(가식부) 100-130g; 말린 바나나(원재_001184)는 다른 이름"),

    # ---------------- 집밥 반찬·찌개 (home) ----------------
    "kimchi_bokkeum": F("반찬", "home", ["김치볶음", "볶음김치", "stir-fried kimchi"], ["음식_017252", "음식_017253"], units=NAMUL,
                        serving=(40, 70), note=NAMUL_NOTE),
    "ojingeochae": F("반찬", "home", ["오징어채볶음", "오징어채", "squid strip stir-fry"], ["음식_010584", "음식_010585"],
                     units=NAMUL, serving=(30, 50), note=DRY_NOTE + "; 조림·무침 행은 다른 이름"),
    "mumallaengi": F("반찬", "home", ["무말랭이무침", "무말랭이", "무말랭이 무침"], ["음식_014858", "음식_014859"], units=NAMUL,
                     serving=(30, 50), note=DRY_NOTE + "; 오징어 부재료·장아찌 행은 다른 이름"),
    "kkongchi_jjigae": F("국찌개", "home", ["꽁치김치찌개", "꽁치찌개", "김치꽁치찌개"], ["음식_017168", "음식_017169"], units=GUK,
                         serving=(250, 350), note=GUK_NOTE + "; '꽁치찌개 김치' 행 포함"),
    "kimchi_jjim": F("국찌개", "home", ["김치찜", "돼지고기 김치찜", "돼지 김치찜"], ["음식_017217", "음식_017216", "음식_017219"],
                     units={"인분": (250, 350), "접시": (250, 350), "그릇": (250, 350), **G_ML}, serving=(250, 350),
                     note="김치찜 1인분·1그릇 250-350g; 100g 기준 행은 '간편조리세트 돼지고기 김치찜'(음식_017219)뿐이라 포함, 묵은지찜(음식_017218)은 다른 이름"),
    "aehobak": F("반찬", "home", ["애호박볶음", "호박볶음", "zucchini stir-fry"], ["음식_010982"], units=NAMUL, serving=(40, 70),
                 note=NAMUL_NOTE),
    "pa_kimchi": F("반찬", "home", ["파김치", "green onion kimchi"], ["음식_007095"], units=G_ML, serving=(30, 60), note=KIMCHI_NOTE),
    "yeolmu_kimchi": F("반찬", "home", ["열무김치", "young radish kimchi"], ["음식_010684"], units=G_ML, serving=(30, 60),
                       note=KIMCHI_NOTE),
    "chonggak": F("반찬", "home", ["총각김치", "알타리김치", "ponytail radish kimchi"], ["음식_009932"],
                  units={"개": (25, 40), **G_ML}, serving=(30, 60), note=KIMCHI_NOTE + "; 총각김치 무 1개 25-40g"),
    "dotori_muk": F("반찬", "home", ["도토리묵무침", "도토리묵", "acorn jelly salad"], ["음식_016427", "음식_016426"],
                    units={"접시": (150, 250), "팩": (200, 300), **G_ML}, serving=(60, 120),
                    note="묵무침 반찬 1회 60-120g, 1접시 150-250g, 반찬가게 1팩 200-300g"),
    "nurungji": F("밥", "home", ["누룽지", "끓인 누룽지", "누룽지탕", "boiled scorched rice"], ["음식_017066", "음식_017067"],
                  units={"그릇": (250, 350), "bowl": (250, 350), **G_ML}, serving=(250, 350), note="끓인 누룽지 1그릇 250-350g"),
    "buchu_muchim": F("반찬", "home", ["부추무침", "부추겉절이", "chive salad"], ["음식_013796", "음식_013797"], units=NAMUL,
                      serving=(30, 60), note=NAMUL_NOTE + "; 양파·채소 부재료 행은 다른 이름"),
    "parae": F("반찬", "home", ["파래무침", "green laver salad"], ["음식_007093", "음식_007094"], units=NAMUL, serving=(30, 60),
               note=NAMUL_NOTE),
    "gochujang_bulgogi": F("고기", "home", ["고추장불고기", "고추장 불고기", "고추장 돼지불고기", "spicy pork bulgogi"], ["음식_016343", "음식_016345"],
                           units={"인분": (150, 250), "접시": (150, 250), **G_ML}, serving=(150, 250),
                           note="고추장불고기 1인분 150-250g; '돼지불고기 고추장'(100ml) 행 + 100g 기준 '대파고추장불고기' 행(음식_016345); 간장·버섯·기사식당 행은 다른 이름"),
    "ggwari_myeolchi": F("반찬", "home", ["꽈리고추멸치볶음", "멸치꽈리고추볶음", "꽈리멸치볶음"],
                         ["음식_017155", "음식_017154", "음식_014927", "음식_014928", "음식_010152", "음식_010153"],
                         units=NAMUL, serving=(30, 50), note=DRY_NOTE + "; '꽈리고추볶음 멸치/잔멸치'·'멸치볶음 꽈리고추'·'잔멸치볶음 꽈리고추' 행"),
    "cheongpomuk": F("반찬", "home", ["청포묵무침", "청포묵 무침", "mung bean jelly salad"], ["음식_009954", "음식_009956"],
                     units={"접시": (150, 250), **G_ML}, serving=(60, 120), note="묵무침 반찬 1회 60-120g; 청포묵국(음식_009955)은 다른 음식"),
    "oi_muchim": F("반찬", "home", ["오이무침", "cucumber salad"], ["음식_010631"], units=NAMUL, serving=(40, 70),
                   note=NAMUL_NOTE + "; 양파·부추·달래 부재료 행은 다른 이름"),
    "chamnamul": F("반찬", "home", ["참나물무침", "참나물", "참나물 무침"], ["음식_009999", "음식_010000"], units=NAMUL,
                   serving=(40, 70), note=NAMUL_NOTE + "; 겉절이·된장 무침 행은 다른 이름"),
    "dureup": F("반찬", "home", ["두릅무침", "두릅", "데친 두릅"], ["음식_016291", "음식_016292"], units=NAMUL, serving=(40, 80),
                note=NAMUL_NOTE + "; '두릅무침 초고추장' 행 포함"),

    # ---------------- 식당 (restaurant) ----------------
    "haemul_tang": F("국찌개", "restaurant", ["해물탕", "seafood stew"], ["음식_002289"],
                     units={"인분": (400, 550), "그릇": (400, 550), "냄비": (1200, 1800), **G_ML}, serving=(400, 550),
                     note="해물탕 1인분 400-550g(국물 포함), 중자(2-3인) 900-1300g; 간편조리세트 행(음식_002288)은 다른 이름"),
    "yeonpo_tang": F("국찌개", "restaurant", ["연포탕", "낙지연포탕", "octopus soup"], ["음식_010688"],
                     units={"인분": (400, 550), "그릇": (400, 550), **G_ML}, serving=(400, 550), note="연포탕 1인분 400-550g(국물 포함)"),
    "dolsot": F("밥", "restaurant", ["돌솥비빔밥", "돌솥 비빔밥", "dolsot bibimbap"], ["음식_013740", "음식_016414"],
                units={"그릇": (400, 500), "개": (400, 500), **G_ML}, serving=(400, 500),
                note="돌솥비빔밥 1그릇 400-500g; '비빔밥 돌솥' 행 포함, 양념장 행(음식_016413)은 다른 이름"),
    "momil": F("면", "restaurant", ["모밀국수", "판모밀", "냉모밀", "메밀국수", "soba"], ["음식_014882", "음식_014945"],
               units={"판": (250, 350), "그릇": (300, 450), "인분": (300, 450), **G_ML}, serving=(300, 450),
               note="모밀국수 1인분(쯔유 포함) 300-450g, 판모밀 1판 250-350g; 동의어 '메밀소바' 행 포함"),
    "memil_jeon": F("빵과자", "restaurant", ["메밀전", "메밀부침", "buckwheat pancake"], ["음식_014943"],
                    units={"장": (100, 150), "개": (100, 150)}, serving=(100, 150), note="메밀전 1장 100-150g"),
    "minari_jeon": F("빵과자", "restaurant", ["미나리전", "water parsley pancake"], ["음식_014810"],
                     units={"장": (100, 160), "개": (100, 160)}, serving=(100, 160), note="미나리전 1장 100-160g"),
    "haemul_deopbap": F("밥", "restaurant", ["해물덮밥", "seafood rice bowl"], ["음식_016721", "음식_002303"], units=DEOP, serving=(400, 500),
                        note=DEOP_NOTE + "; '덮밥 해물'·'해물 덮밥' 행"),
    "chamchi_deopbap": F("밥", "restaurant", ["참치덮밥", "tuna rice bowl"], ["음식_016722"], units=DEOP, serving=(400, 500),
                         note=DEOP_NOTE + "; '덮밥 참치' 행"),
    "bulgogi_deopbap": F("밥", "restaurant", ["불고기덮밥", "불고기 덮밥", "bulgogi rice bowl"], ["음식_016727"], units=DEOP,
                         serving=(400, 500), note=DEOP_NOTE + "; '덮밥 불고기' 행"),
    "yusanseul_deopbap": F("밥", "restaurant", ["유산슬덮밥", "유산슬밥"], ["음식_016724"], units=DEOP, serving=(400, 500),
                           note=DEOP_NOTE + "; '덮밥 유산슬' 행"),
    "japtangbap": F("밥", "restaurant", ["잡탕밥"], ["음식_010123", "음식_010124"],
                    units={"그릇": (450, 600), "인분": (450, 600), **G_ML}, serving=(450, 600), note="중식 덮밥 1그릇 450-600g"),
    "gochu_japchae": F("고기", "restaurant", ["고추잡채", "pepper japchae"], ["음식_017860", "음식_010134"],
                       units={"접시": (250, 350), "인분": (200, 300), **G_ML}, serving=(150, 250),
                       note="고추잡채 1접시 250-350g, 1인분 200-300g, 나눠 먹을 때 1인 몫 150-250g; 100g 기준 '잡채 고추'(음식_010134) 행 포함; 꽃빵키트(음식_017859)는 다른 이름"),
    "yusanseul": F("고기", "restaurant", ["유산슬"], ["음식_010308", "음식_010309"],
                   units={"접시": (300, 400), **G_ML}, serving=(150, 250), note="중식 요리 1접시 300-400g, 1인 몫 150-250g"),
    "nanjawanseu": F("고기", "restaurant", ["난자완스"], ["음식_017122", "음식_017123"], units={"접시": (300, 400), **G_ML},
                     serving=(150, 250), note="중식 요리 1접시 300-400g, 1인 몫 150-250g"),
    "ulmyeon": F("면", "restaurant", ["울면"], ["음식_010320", "음식_010321"], units=NOODLE, serving=(600, 800), note=NOODLE_NOTE),
    "beoseot_jeongol": F("국찌개", "restaurant", ["버섯전골", "mushroom hot pot"], ["음식_014015", "음식_014016"], units=JEONGOL,
                         serving=(300, 450), note=JEONGOL_NOTE + "; 소고기·들깨·간편조리세트 행은 다른 이름"),
    "kimchi_jeongol": F("국찌개", "restaurant", ["김치전골", "kimchi hot pot"], ["음식_017235", "음식_017236"], units=JEONGOL,
                        serving=(300, 450), note=JEONGOL_NOTE),
    "bulnak": F("국찌개", "restaurant", ["불낙전골", "불고기낙지전골"], ["음식_013765", "음식_013763", "음식_013764"], units=JEONGOL,
                serving=(300, 450), note=JEONGOL_NOTE + "; '불낙전골 해물' 행 포함, 100g 기준 행은 간편조리세트(음식_013764)뿐이라 포함"),
    "bajirak_kalguksu": F("면", "restaurant", ["바지락칼국수", "바지락 칼국수", "clam kalguksu"], ["음식_009713", "음식_009714"],
                          units=NOODLE, serving=(600, 800),
                          note=NOODLE_NOTE + "; '칼국수 바지락' 행; 간편조리세트 세숫대야 바지락칼국수(음식_009721)는 다른 이름"),
    "deulkkae_kalguksu": F("면", "restaurant", ["들깨칼국수", "들깨 칼국수", "perilla kalguksu"], ["음식_009715", "음식_009716"],
                           units=NOODLE, serving=(600, 800), note=NOODLE_NOTE + "; '칼국수 들깨' 행"),
    "gajami": F("생선", "restaurant", ["가자미구이", "grilled flounder"], ["음식_018044", "음식_018045"],
                units={"마리": (120, 200), "토막": (60, 100)}, serving=(120, 200), note="가자미구이 1마리(가식부) 120-200g"),
    "cheese_katsu": F("고기", "restaurant", ["치즈돈가스", "치즈돈까스", "치즈카츠", "cheese katsu"], ["음식_016418"],
                      units={"장": (200, 280), "개": (200, 280), "인분": (200, 280)}, serving=(200, 280),
                      note="치즈돈가스 1장 200-280g; '돈가스 치즈' 행"),
    "cream_pasta": F("면", "restaurant", ["크림파스타", "크림 파스타", "크림스파게티", "크림 스파게티", "cream pasta"],
                     ["음식_011757", "음식_011758"], units=PASTA, serving=(300, 400),
                     note=PASTA_NOTE + "; '스파게티 크림스파게티/크림소스' 행; 해물 크림 행은 다른 이름"),
    "toowoomba": F("면", "restaurant", ["투움바파스타", "투움바 파스타", "투움바 스파게티", "toowoomba pasta"], ["음식_011752"],
                   units=PASTA, serving=(300, 400), note=PASTA_NOTE + "; '스파게티 투움바 스파게티' 행"),

    # ---------------- 배달 (delivery) ----------------
    "fried_chicken": F("고기", "delivery", ["후라이드치킨", "후라이드 치킨", "프라이드치킨", "후라이드", "fried chicken"],
                       ["음식_016759"], units={"마리": (650, 900), "조각": (50, 90), "pc": (50, 90)}, serving=(250, 400),
                       note="후라이드 치킨 1마리 가식부 650-900g, 1조각 50-90g; '닭튀김 후라이드 치킨' 행"),
    "pepperoni": F("빵과자", "delivery", ["페퍼로니피자", "페퍼로니 피자", "pepperoni pizza"], ["음식_003103", "음식_003101"],
                   units={"조각": (100, 150), "slice": (100, 150), "판": (800, 1200)}, serving=(100, 150),
                   note="L 피자 1조각 100-150g, 1판(8조각) 800-1200g; '피자 페퍼로니피자(R)/오리지널(L)' 행"),
    "nakgopsae": F("국찌개", "delivery", ["낙곱새", "낙지곱창새우전골"], ["음식_017127"], units={"인분": (300, 450), **G_ML},
                   serving=(300, 450), note="낙곱새 1인분 300-450g (간편조리세트 행이 DB의 유일한 낙곱새 행)"),
    "golbaengi": F("반찬", "delivery", ["골뱅이무침", "spicy sea snail salad"], ["음식_017839", "음식_017838"],
                   units={"접시": (300, 450), **G_ML}, serving=(100, 200),
                   note="골뱅이무침 1접시 300-450g, 1인 몫 100-200g; 소면·채소 부재료 행은 다른 이름"),

    # ---------------- 분식·길거리 (street) ----------------
    "cheese_tteok": F("빵과자", "street", ["치즈떡볶이", "치즈 떡볶이", "cheese tteokbokki"], ["음식_016202", "음식_016200"],
                      units={"인분": (250, 350), "접시": (250, 350), "컵": (150, 200)}, serving=(250, 350),
                      note="분식 떡볶이 1인분 250-350g, 컵떡볶이 150-200g; '떡볶이 치즈/치즈떡볶이' 행"),
    "baek_sundae": F("고기", "street", ["백순대", "백순대볶음"], ["음식_012909"], units={"인분": (300, 400), "접시": (300, 400)},
                     serving=(300, 400), note="백순대볶음 1인분 300-400g; 간편조리세트 신림동 백순대볶음(음식_012912)은 다른 이름"),
    "mul_mandu": F("빵과자", "street", ["물만두", "boiled dumplings"], ["음식_015035"],
                   units={"개": (10, 15), "알": (10, 15), "접시": (150, 220)}, serving=(150, 220),
                   note="물만두 1개 10-15g, 1접시(약 15개) 150-220g"),
    "kimchi_mandu": F("빵과자", "street", ["김치만두", "kimchi dumpling"], ["음식_015036"], units={"개": (30, 45), "알": (30, 45)},
                      serving=(150, 225), note="김치만두 1개 30-45g"),
    "waffle": F("빵과자", "street", ["와플", "플레인 와플", "waffle"], ["음식_010398", "음식_010402", "음식_010404"],
                units={"개": (80, 120)}, serving=(80, 120), note="길거리 와플 1개 80-120g; 크림·과일 토핑 와플 행은 다른 이름"),
    "goguma_gui": F("과일", "street", ["군고구마", "구운 고구마", "roasted sweet potato"], ["원재_000337", "원재_000345", "원재_000348"],
                    units={"개": (150, 250)}, serving=(150, 250), note="군고구마 중 1개 150-250g"),

    # ---------------- 과일 (fruit) ----------------
    "dangam": F("과일", "fruit", ["단감", "persimmon"], ["원재_001125", "원재_000036", "원재_000037", "원재_000038", "원재_000039"],
                units={"개": (150, 200), "조각": (20, 35)}, serving=(150, 200), note="단감 1개(가식부) 150-200g; 품종별 단감 행 포함"),
    "geobong": F("과일", "fruit", ["거봉", "거봉포도", "kyoho grape"], ["원재_001257", "원재_001258"],
                 units={"송이": (300, 450), "알": (10, 14), "개": (10, 14)}, serving=(150, 250), note="거봉 1알 10-14g, 1송이 300-450g"),
    "cheondo": F("과일", "fruit", ["천도복숭아", "천도", "nectarine"], ["원재_001194"], units={"개": (100, 150)}, serving=(100, 150),
                 note="천도복숭아 1개(가식부) 100-150g"),
    "redhyang": F("과일", "fruit", ["레드향", "red hyang"], ["원재_001143"], units={"개": (180, 250)}, serving=(180, 250),
                  note="레드향 1개(가식부) 180-250g"),

    # ---------------- 카페 (cafe) ----------------
    "asyachu": F("음료", "cafe", ["아샷추", "아이스티 샷추가", "아이스티에 샷 추가"], ["음식_011048", "음식_011046", "음식_011032", "음식_011036"],
                 units=DRINK, serving=(350, 500), note=DRINK_NOTE + "; 제로슈가 아샷추 행(음식_011017/011092)은 다른 이름"),
    "goguma_latte": F("음료", "cafe", ["고구마라떼", "고구마 라떼", "sweet potato latte"], ["음식_015652", "음식_015653", "음식_015679"],
                      units=DRINK, serving=(350, 500), note=DRINK_NOTE + "; 흑당·자색·치즈 등 변형 행은 다른 이름"),
    "toffeenut": F("음료", "cafe", ["토피넛라떼", "토피넛 라떼", "toffee nut latte"], ["음식_008403", "음식_008404", "음식_008405"],
                   units=DRINK, serving=(350, 500), note=DRINK_NOTE + "; 샷 추가·에스프레소 변형 행은 다른 이름"),
    "mango_smoothie": F("음료", "cafe", ["망고스무디", "망고 스무디", "mango smoothie"], ["음식_012596", "음식_012309", "음식_012310"],
                        units=DRINK, serving=(350, 500), note=DRINK_NOTE + "; '애플망고 스무디' 행 포함, 저당·요거트 변형은 다른 이름"),
    "yogurt_smoothie": F("음료", "cafe", ["요거트스무디", "요거트 스무디", "플레인 요거트 스무디", "yogurt smoothie"],
                         ["음식_011912", "음식_011913", "음식_011921", "음식_011922", "음식_011914"], units=DRINK,
                         serving=(350, 500), note=DRINK_NOTE + "; '플레인요거트 스무디'·'스무디 플레인요거트' 행"),
    "cheongpodo_ade": F("음료", "cafe", ["청포도에이드", "청포도 에이드", "green grape ade"], ["음식_010732", "음식_010733"],
                        units=DRINK, serving=(350, 500), note=DRINK_NOTE + "; 사이즈 표기 행은 다른 이름"),
    "madeleine": F("빵과자", "cafe", ["마들렌", "madeleine"], ["음식_014991"], units={"개": (25, 40)}, serving=(25, 40),
                   note="마들렌 1개 25-40g; 맛 변형 행은 다른 이름"),
    "financier": F("빵과자", "cafe", ["휘낭시에", "피낭시에", "financier"], ["음식_014964", "음식_014963", "음식_014951"],
                   units={"개": (30, 45)}, serving=(30, 45), note="휘낭시에 1개 30-45g; 클래식·플레인 행"),
    "brownie": F("빵과자", "cafe", ["브라우니", "brownie"], ["음식_008053"], units={"조각": (60, 90), "개": (60, 90)}, serving=(60, 90),
                 note="브라우니 1조각 60-90g"),
    "carrot_cake": F("빵과자", "cafe", ["당근케이크", "당근 케이크", "carrot cake"], ["음식_008215"],
                     units={"조각": (100, 140), "개": (100, 140)}, serving=(100, 140), note="조각 케이크 1조각 100-140g; '(조각)' 행"),
    "redvelvet": F("빵과자", "cafe", ["레드벨벳 케이크", "레드벨벳", "red velvet cake"], ["음식_008144", "음식_008145"],
                   units={"조각": (100, 130), "개": (100, 130)}, serving=(100, 130), note="조각 케이크 1조각 100-130g"),
}

# ---------------- 브랜드 제품 (MFDS 가공식품 라벨; 각 행은 빌드 시 원본 CSV와 대조) ----------------
PR = {
    # PX
    "schnell": P("고기", "px", ["삼양슈넬치킨", "슈넬치킨", "schnell chicken"], [("P117-502050200-0837", 290, "삼양슈넬치킨")],
                 BAG(220), (110, 220), "1봉 220g"),
    "geonppang": P("빵과자", "px", ["건빵", "hardtack"],
                   [("P101-103000400-0067", 417, "건빵"), ("P101-106000100-0098", 419, "건빵"), ("P101-106000100-4559", 414, "건빵"),
                    ("P101-106000100-7625", 413, "건빵")], {}, (20, 40), "건빵 한 주먹 20-40g(통상)",
                   excl={"P101-106000100-0099": "209kcal/100g는 동명 건빵 4행(413-419)의 절반으로 표기 오류 추정"}),
    "byeolsatang": P("빵과자", "px", ["별사탕", "star candy"],
                     [("P101-203000200-2086", 398, "별사탕"), ("P101-203000200-2085", 392, "별사탕"),
                      ("P101-203000200-2084", 349, "별사탕"), ("P101-203000200-2087", 400, "별사탕"),
                      ("P101-203000200-0438", 400, "별사탕")], {}, (3, 10), "건빵 봉지 속 별사탕 몇 개 3-10g(통상)"),
    "matdasi": P("반찬", "px", ["맛다시", "맛다시프레시"], [("P112-013001300-0515", 219, "맛다시프레시")],
                 {"숟갈": (10, 20), "숟가락": (10, 20), "개": (45, 45)}, (10, 20), "1개 45g; 밥에 비비는 1숟갈 10-20g"),
    # convenience-store noodles / meals
    "shrimp_cup": P("면", "cvs", ["새우탕컵", "새우탕 컵", "새우탕 작은컵", "새우탕"],
                    [("P108-003000400-0119", 433, "새우탕컵"), ("P108-003000400-0966", 433, "새우탕컵"),
                     ("P108-003000400-1007", 433, "새우탕컵"), ("P108-003000400-0858", 433, "새우탕컵")],
                    {"개": (67, 67), "컵": (67, 67)}, (67, 67), "작은컵 1개 67g(텍스트가 '작은컵/작은 거'); 115g 동명 행은 kcal/100만 반영"),
    "kimchi_keunsabal": P("면", "cvs", ["김치큰사발면", "김치큰사발", "김치 큰사발"],
                          [("P108-003000400-0042", 446, "김치큰사발면"), ("P108-003000400-0912", 446, "김치큰사발면")],
                          {"개": (112, 112), "컵": (112, 112)}, (112, 112), "1개 112g"),
    "jjaseullaeng": P("면", "cvs", ["짜슐랭", "jjaseullaeng"],
                      [("P108-004000400-0097", 414, "짜슐랭"), ("P108-004000400-0038", 414, "짜슐랭")], BAG(145), (145, 145), "1봉 145g"),
    "omori": P("면", "cvs", ["오모리 김치찌개라면", "오모리 김치찌개 라면", "오모리라면"], [("P108-003000400-0186", 327, "오모리 김치찌개라면")],
               {"개": (150, 150), "컵": (150, 150)}, (150, 150), "1개 150g"),
    "toomba": P("면", "cvs", ["신라면툼바", "신라면 툼바", "툼바"],
                [("P108-003000400-0977", 431, "신라면툼바"), ("P108-003000400-0978", 431, "신라면툼바"),
                 ("P108-003000400-0493", 431, "신라면툼바"), ("P108-003000400-0494", 431, "신라면툼바")], BAG(137), (137, 137),
                "1봉 137g; 큰사발면(신라면툼바큰사발면)은 다른 이름"),
    "yeolmu_bibim": P("면", "cvs", ["열무비빔면", "삼양 열무비빔면"], [("P108-009000400-0034", 408, "열무비빔면")], BAG(130), (130, 130),
                      "1봉 130g"),
    "hack_buldak": P("면", "cvs", ["핵불닭볶음면", "핵불닭"],
                     [("P108-009000400-0094", 396, "핵불닭볶음면"), ("P108-009000400-0218", 396, "핵불닭볶음면")], BAG(140), (140, 140),
                     "1봉 140g"),
    # drinks
    "maeil_coffee": P("음료", "cvs", ["매일우유 커피맛", "매일 커피우유", "커피우유"],
                      [("P119-204020400-0902", 70, "매일우유 커피맛"), ("P119-204020400-0177", 70, "매일우유 커피맛")],
                      {"개": (200, 200), "팩": (200, 200)}, (200, 200), "1팩 200ml"),
    "georgia_latte": P("음료", "cvs", ["조지아 고티카 빈티지 라떼", "고티카 빈티지 라떼", "조지아 라떼"],
                       [("P109-201020000-0309", 44, "조지아 고티카 빈티지 라떼")], {"캔": (270, 270), "개": (270, 270)}, (270, 270),
                       "1캔 270ml"),
    "mountain_dew": P("음료", "snack", ["마운틴듀", "마운틴 듀", "mountain dew"],
                      [("P109-401040100-0117", 48, "마운틴듀"), ("P109-401040100-2310", 48, "마운틴듀")],
                      {"캔": (250, 355), "개": (250, 355)}, (250, 355), "1캔 250-355ml"),
    "dr_pepper": P("음료", "snack", ["닥터페퍼", "닥터 페퍼", "dr pepper"], [("P109-401040100-0071", 42, "닥터페퍼")],
                   {"캔": (355, 355), "개": (355, 355)}, (355, 355), "1캔 355ml"),
    "pepsi": P("음료", "snack", ["펩시", "펩시콜라", "pepsi"],
               [("P109-401040100-1224", 45, "펩시"), ("P109-401040100-0551", 45, "펩시"), ("P109-401040100-2414", 45, "펩시"),
                ("P109-401040100-2415", 45, "펩시")], {"캔": (245, 355), "개": (245, 355)}, (245, 355), "1캔 245-355ml"),
    "sunnyten": P("음료", "snack", ["써니텐 포도", "써니텐 포도향", "써니텐"],
                  [("P109-401040100-0848", 51, "써니텐 포도향"), ("P109-401040100-0309", 52, "써니텐포도향")],
                  {"캔": (250, 250), "개": (250, 250)}, (250, 250), "1캔 250ml"),
    # snacks
    "kkulkkwa": P("빵과자", "snack", ["꿀꽈배기", "꿀꽈베기"],
                  [("P101-106000100-0302", 500, "꿀꽈배기"), ("P101-106000100-6281", 500, "꿀꽈배기"),
                   ("P101-104000100-1900", 500, "꿀꽈배기"), ("P101-106000100-7143", 500, "꿀꽈배기"),
                   ("P101-104000100-2343", 500, "꿀꽈배기")], BAG(90), (90, 90), "1봉 90g",
                  excl={"P101-106000100-6426": "중국 제조 동명 타사 제품(500g 대용량), 농심 꿀꽈배기 아님"}),
    "jagalchi": P("빵과자", "snack", ["자갈치", "자갈치 과자", "자갈치문어맛"],
                  [("P101-106000100-1563", 522, "자갈치문어맛"), ("P101-106000100-7192", 511, "자갈치문어맛")], BAG(90), (90, 90),
                  "1봉 90g; 같은 이름 어묵·생선 제품은 다른 이름"),
    "swingchip": P("빵과자", "snack", ["스윙칩 간장치킨맛", "스윙칩간장치킨맛", "스윙칩"], [("P101-106000100-1086", 603, "스윙칩간장치킨맛")],
                   BAG(60), (60, 60), "1봉 60g"),
    "noonul": P("빵과자", "snack", ["눈을감자", "눈을 감자"],
                [("P101-106000100-7821", 496, "눈을감자 아메리칸치즈맛"), ("P101-106000100-3869", 512, "눈을감자 칠리치즈 프라이즈맛"),
                 ("P101-106000100-0362", 512, "눈을감자 페퍼솔트맛"), ("P101-106000100-0361", 490, "눈을감자오지치즈후라이맛")],
                BAG(56), (56, 56), "1봉 56g; 맛 미기재 -> 56g 4종 라벨 범위"),
    "cheetos": P("빵과자", "snack", ["치토스", "cheetos"],
                 [("P101-106000100-1743", 539, "치토스 스모키바베큐맛"), ("P101-104000100-0996", 531, "치토스매콤달콤한맛")],
                 BAG(64), (64, 64), "1봉 64g; 맛 미기재 -> 64g 2종 라벨 범위"),
    "potato_meoktae": P("빵과자", "snack", ["포테토칩 먹태청양마요맛", "포테토칩 먹태청양마요", "먹태칩", "포테토칩"],
                        [("P101-106000100-2915", 560, "포테토칩 먹태청양마요맛")], BAG(50), (50, 50), "1봉 50g"),
    "ogamja": P("빵과자", "snack", ["오감자 어니언맛", "오감자", "ogamja"], [("P101-106000100-3178", 562, "오감자 어니언맛")],
                BAG(50), (50, 50), "1봉 50g; 맛 미기재 시 대표 SKU"),
    "ace": P("빵과자", "snack", ["에이스", "에이스 크래커", "ace crackers"],
             [("P101-103000100-5211", 540, "에이스"), ("P101-103000100-5210", 540, "에이스"), ("P101-103000100-5319", 533, "에이스"),
              ("P101-106000100-5347", 542, "에이스"), ("P101-103000400-1024", 540, "에이스")],
             {"곽": (121, 121), "갑": (121, 121), "봉": (121, 121)}, (20, 40), "1곽 121g; 몇 개 집어먹기 20-40g(통상)",
             excl={"P119-401040100-0186": "동명 발효유 음료(75kcal/100ml), 크래커 아님",
                   "P119-401040100-0185": "동명 발효유 음료(64kcal/100ml), 크래커 아님"}),
    "haribo": P("빵과자", "snack", ["하리보 골드베렌", "하리보", "haribo"],
                [("P101-205000200-4551", 336, "하리보 골드베렌"), ("P101-205000200-4553", 333, "하리보 골드베렌(1000G)"),
                 ("P101-205000200-4949", 335, "하리보 골드베렌 여름한정판 80g")],
                {"봉": (80, 100), "봉지": (80, 100)}, (80, 100), "시판 1봉 80-100g"),
    "ferrero": P("빵과자", "snack", ["페레로 로쉐", "페레로로쉐", "ferrero rocher"],
                 [("P103-102020500-2426", 593, "페레로 로쉐"), ("P103-102020500-4225", 593, "페레로 로쉐"),
                  ("P103-101020500-1613", 595, "페레로 로쉐"), ("P103-101020500-1612", 595, "페레로 로쉐"),
                  ("P103-102020500-3348", 593, "페레로 로쉐")], {"알": (12, 13), "개": (12, 13)}, (12, 13),
                 "1알 12.5g(포장 표기 통상값)"),
    "snickers": P("빵과자", "snack", ["스니커즈", "snickers"],
                  [("P103-102020500-4290", 490, "스니커즈 땅콩 (10249601-51g)"), ("P103-102020500-4703", 490, "스니커즈 땅콩 (60021020-51g)")],
                  {"개": (51, 51)}, (51, 51), "1개 51g"),
    "okdongja": P("빵과자", "snack", ["옥동자", "옥동자 아이스크림"],
                  [("P102-103010300-0324", 243, "옥동자"), ("P102-103010300-0325", 243, "옥동자"), ("P102-103010300-1211", 243, "옥동자"),
                   ("P102-103010300-1404", 257, "옥동자")], {"개": (70, 70)}, (70, 70), "1개 70ml(420ml 멀티팩 행은 kcal/100만 반영)"),
    "chaltteok": P("빵과자", "snack", ["찰떡아이스", "찰떡 아이스"],
                   [("P102-103010300-0387", 189, "찰떡아이스"), ("P102-103010300-1217", 189, "찰떡아이스")],
                   {"개": (90, 90), "봉": (90, 90)}, (90, 90), "1봉 90ml"),
    "ppangppare": P("빵과자", "snack", ["빵빠레 초코바", "빵빠레 초코", "빵빠레"], [("P102-103010300-0585", 160, "빵빠레 초코바")],
                    {"개": (75, 75)}, (75, 75), "1개 75ml"),
    "haagen": P("빵과자", "snack", ["하겐다즈 미니컵 스트로베리", "하겐다즈 미니컵 딸기", "하겐다즈 미니컵", "하겐다즈"],
                [("P102-101010100-0674", 220, "하겐다즈 미니컵 스트로베리")], {"개": (100, 100), "컵": (100, 100)}, (100, 100),
                "미니컵 1개 100g"),
}
assert not set(R) & set(PR)
R.update(PR)
EXTRA_DENS: dict = {}  # v6: no density outside DB/MFDS rows


# ---------------------------------------------------------------------------
# Case DSL
# ---------------------------------------------------------------------------
def I(key, amt=None, unit=None, grams=None, optional=False, units_ok=None, note=None, low=False, gunit=None):
    """gunit = implied unit for a bare number without a unit word ("짜슐랭1"): used for grams only, units -> []."""
    assert unit is None or gunit is None
    return dict(key=key, amt=amt, unit=unit, grams=grams, optional=optional, units_ok=units_ok, note=note, low=low,
                gunit=gunit)


CASES: list[dict] = []

# Product keys -> tokens that identify the transcribed SKU in the text. A product item whose text carries none of its
# tokens is an UNBRANDED mention labelled against one representative SKU: the case is capped at medium confidence and
# tagged flag:unbranded. None = exempt (label already spans every same-food SKU row: 건빵, 별사탕, cola).
BRAND_TOKENS = {
    "cola": None, "schnell": ["슈넬"], "geonppang": None, "byeolsatang": None, "matdasi": ["맛다시"],
    "shrimp_cup": ["새우탕"], "kimchi_keunsabal": ["김치큰사발", "김치 큰사발"], "jjaseullaeng": ["짜슐랭"], "omori": ["오모리"],
    "toomba": ["툼바"], "yeolmu_bibim": ["열무비빔면"], "hack_buldak": ["핵불닭"], "maeil_coffee": ["매일우유"],
    "georgia_latte": ["고티카"], "mountain_dew": ["마운틴듀", "마운틴 듀", "mountain dew"],
    "dr_pepper": ["닥터페퍼", "닥터 페퍼", "dr pepper"], "pepsi": ["펩시", "pepsi"], "sunnyten": ["써니텐"],
    "kkulkkwa": ["꿀꽈배기", "꿀꽈베기"], "jagalchi": ["자갈치"], "swingchip": ["스윙칩"], "noonul": ["눈을감자", "눈을 감자"],
    "cheetos": ["치토스", "cheetos"], "potato_meoktae": ["먹태"], "ogamja": ["오감자"], "ace": ["에이스"],
    "haribo": ["하리보", "haribo"], "ferrero": ["페레로", "ferrero"], "snickers": ["스니커즈", "snickers"],
    "okdongja": ["옥동자"], "chaltteok": ["찰떡아이스"], "ppangppare": ["빵빠레"], "haagen": ["하겐다즈"],
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
    if _EMOJI_RE.search(text) and "이모지" not in nz:
        nz.append("이모지")
    if _LATIN_RE.search(text) and "영어혼용" not in nz:
        nz.append("영어혼용")
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
LEFT = "flag:leftover"
TYPO = "오타"
SD, RD, RN, CP, FL, SO = PERSONAS
DC, AK, AT, RV, BM, MM = STYLES

# ============================== soldier (군인) ==============================
C("어 오늘 PX에서 그 뭐냐 슈넬치킨 반 봉지랑 음 맛다시 한 숟갈 밥에 비벼 먹음",
  [I("schnell", 0.5, "봉지"), I("matdasi", 1, "숟갈"), I("rice")], SD, DC, tags=[PROD], conf="medium",
  note="밥 양 미기재 -> 1공기 190-230g")
C("음 점심 짬밥 김치볶음 조금 하고 어 콩나물국 아니 그 꽁치김치찌개 한 그릇 먹었습니다",
  [I("kimchi_bokkeum"), I("kkongchi_jjigae", 1, "그릇")], SD, DC, tags=[CORR], note="콩나물국은 정정으로 취소")
C("그 야간 근무 끝나고 어 건빵 한 주먹하고 별사탕 몇 개 먹었어",
  [I("geonppang", grams=(20, 40), low=True), I("byeolsatang", grams=(3, 10), low=True)], SD, DC, tags=[PROD], conf="low",
  note="'한 주먹' 20-40g, '몇 개' 3-10g 추정; 두 양 모두 모호")
C("어 그 김치큰사발 하나 음 뽀글이로 먹었고 그 국물까지 다", [I("kimchi_keunsabal", 1, gunit="개")], SD, DC, tags=[PROD],
  note="'하나' = 1개(단위어 없음)")
C("PX 핵불닭볶음면 1개 먹었는데 몇 칼로리임?", [I("hack_buldak", 1, "개")], SD, AK, tags=[PROD])
C("휴가 나와서 엄마가 해준 김치찜 한 접시랑 밥 두 공기 먹었는데 칼로리 얼마나 돼?",
  [I("kimchi_jjim", 1, "접시"), I("rice", 2, "공기")], SD, AK)
C("PX에서 찰떡아이스 2개랑 옥동자 1개 먹었는데 이거 몇 칼로리야?", [I("chaltteok", 2, "개"), I("okdongja", 1, "개")], SD, AK,
  tags=[PROD])
C("외출 나가서 페퍼로니 피자 L 세 조각이랑 닥터페퍼 한 캔 먹었는데 계산 좀",
  [I("pepperoni", 3, "조각"), I("dr_pepper", 1, "캔")], SD, AK, tags=[PROD, SIZE])
C("당직 서면서 레드향 두 개 까먹었는데 칼로리 얼마나 돼?", [I("redhyang", 2, "개")], SD, AK)
C("점심 짬밥 고추장 불고기 1인분. 아 그리고 오이무침도 좀 먹음", [I("gochujang_bulgogi", 1, "인분"), I("oi_muchim")], SD, AT,
  noise=[TYPO])
C("슈넬치킨 한 봉 다 먹음... 아 아니다 동기랑 반 나눠먹었으니까 반",
  [I("schnell", (0.5, 1), "봉", grams=(110, 110))], SD, AT, tags=[PROD, SHARE, CORR], conf="medium",
  note="1봉 220g을 둘이 나눔 -> 110g")
C("저녁 김치전골 1인분 먹고 나서 아 맞다 PX 꿀꽈배기 한 봉도 먹었네", [I("kimchi_jeongol", 1, "인분"), I("kkulkkwa", 1, "봉")],
  SD, AT, tags=[PROD])
C("새우탕 작은컵 하나 먹음. 아 그리고 거기에 밥 반공기 말아먹음", [I("shrimp_cup", 1, "컵", units_ok=["컵", "개"]),
                                                    I("rice", 0.5, "공기")], SD, AT, tags=[PROD, "unit:반"])
C("#휴가첫끼 돌솥비빔밥 한 그릇 + 메밀전 1장 진짜 살 것 같다 🙏", [I("dolsot", 1, "그릇"), I("memil_jeon", 1, "장")], SD, RV)
C("#PX털이 🛒 스윙칩 간장치킨맛 1봉 + 펩시 1캔 🔥", [I("swingchip", 1, "봉"), I("pepsi", 1, "캔")], SD, RV, tags=[PROD])
C("#불침번 끝 🌙 신라면툼바 하나 끓여먹음 존맛탱", [I("toomba", 1, gunit="개")], SD, RV, tags=[PROD])
C("#외박 #치킨 🍗 후라이드 한 마리 둘이서 나눠먹고 펩시 한 캔 소대장님 몰래ㅋㅋ",
  [I("fried_chicken", (0.5, 1), "마리", grams=(325, 450)), I("pepsi", 1, "캔")], SD, RV, tags=[SHARE, PROD], conf="medium",
  note="치킨 1마리 가식부 650-900g의 1/2")
C("오늘 저녁 배식은 해물덮밥이었고 나는 한 그릇 반 먹었다.", [I("haemul_deopbap", 1.5, "그릇")], SD, BM)
C("훈련 끝나고 PX에서 마운틴듀 한 캔이랑 자갈치 한 봉지 먹었다.", [I("mountain_dew", 1, "캔"), I("jagalchi", 1, "봉지")], SD, BM,
  tags=[PROD])
C("아침에 누룽지 한 그릇이랑 파김치 조금 먹었다.", [I("nurungji", 1, "그릇"), I("pa_kimchi")], SD, BM)
C("주말 면회 때 여자친구가 싸 온 거봉 반 송이랑 마들렌 두 개를 먹었다.", [I("geobong", 0.5, "송이"), I("madeleine", 2, "개")], SD, BM,
  tags=["unit:반"])
C("점심) 짬밥 버섯전골1인분, 총각김치2개", [I("beoseot_jeongol", 1, "인분"), I("chonggak", 2, "개")], SD, MM)
C("야식 짜슐랭1 맛다시x", [I("jjaseullaeng", 1, gunit="개")], SD, MM, tags=[PROD, NEG], note="맛다시는 안 먹음(x)")
C("간식 건빵+별사탕 조금, 닥터페퍼1캔", [I("geonppang", grams=(20, 40)), I("byeolsatang", grams=(3, 10)), I("dr_pepper", 1, "캔")],
  SD, MM, tags=[PROD], conf="low", note="'조금' = 건빵 20-40g, 별사탕 3-10g 추정; 양 모호")
C("휴가복귀 전 바지락칼국수1 + 김치만두3개", [I("bajirak_kalguksu", 1, gunit="그릇"), I("kimchi_mandu", 3, "개")], SD, MM)

# ============================== rider (배달 라이더) ==============================
C("어 신호 대기 중에 그 음 스니커즈 하나 먹었고 그 마운틴듀 말고 펩시 한 캔",
  [I("snickers", 1, gunit="개"), I("pepsi", 1, "캔")], RD, DC, tags=[PROD, CORR], note="마운틴듀는 정정으로 취소")
C("음 콜 끊겨서 편의점에서 열무비빔면 하나 그리고 어 삶은 계란 두 개",
  [I("yeolmu_bibim", 1, gunit="개"), I("egg", 2, "개")], RD, DC, tags=[PROD])
C("그 배달 끝나고 어 집 와서 어제 남은 페퍼로니 피자 두 조각 데워 먹음", [I("pepperoni", 2, "조각")], RD, DC, tags=[LEFT])
C("어 그 오모리 김치찌개라면 컵으로 하나 먹었고 음 밥은 안 말았어", [I("omori", 1, "컵", units_ok=["컵", "개"])], RD, DC,
  tags=[PROD, NEG])
C("어 콜 기다리다가 음 하리보 골드베렌 한 봉이랑 그 마운틴듀 반 캔", [I("haribo", 1, "봉"), I("mountain_dew", 0.5, "캔")], RD, DC,
  tags=[PROD])
C("배달하다가 조지아 고티카 빈티지 라떼 한 캔 마셨는데 이거 칼로리 높아?", [I("georgia_latte", 1, "캔")], RD, AK, tags=[PROD])
C("라이더 점심으로 김치큰사발 하나랑 공깃밥 반 공기 말아 먹으면 몇 kcal야?",
  [I("kimchi_keunsabal", 1, gunit="개"), I("rice", 0.5, "공기")], RD, AK, tags=[PROD])
C("비 와서 쉬는 중에 치즈떡볶이 1인분이랑 물만두 반 접시 먹었는데 칼로리 계산해죠",
  [I("cheese_tteok", 1, "인분"), I("mul_mandu", 0.5, "접시")], RD, AK, noise=[TYPO])
C("새벽에 핵불닭 반 개랑 삶은 계란 하나 먹었는데 몇 칼로리야?", [I("hack_buldak", 0.5, "개"), I("egg", 1, gunit="개")],
  RD, AK, tags=[PROD, PART])
C("점심 가자미구이 정식 먹음 가자미 한 마리랑 밥 한 공기. 아 그리고 반찬으로 파래무침도",
  [I("gajami", 1, "마리"), I("rice", 1, "공기"), I("parae")], RD, AT)
C("편의점 찰떡아이스 하나 먹음 *정정: 두 개, 그리고 써니텐 포도 한 캔", [I("chaltteok", 2, "개"), I("sunnyten", 1, "캔")], RD, AT,
  tags=[PROD, CORR])
C("써니텐 포도 한 캔 마심. 아 그리고 눈을감자도 한 봉", [I("sunnyten", 1, "캔"), I("noonul", 1, "봉")], RD, AT, tags=[PROD])
C("저녁은 낙곱새 1인분이랑 공기밥 하나. 아 볶음밥은 안 먹었고",
  [I("nakgopsae", 1, "인분"), I("rice", 1, gunit="개")], RD, AT, tags=[NEG])
C("#라이더일상 오늘 콜 폭발🔥 점심은 불고기덮밥 한 그릇 후딱", [I("bulgogi_deopbap", 1, "그릇")], RD, RV)
C("#편의점털이 Cheetos 1봉 + Dr Pepper 1캔 이 조합 미쳤다🤤", [I("cheetos", 1, "봉"), I("dr_pepper", 1, "캔")], RD, RV, tags=[PROD])
C("#야간배달 끝 🌃 백순대 1인분 + 콜라 한 캔 포장해서", [I("baek_sundae", 1, "인분"), I("cola", 1, "캔")], RD, RV, tags=[PROD])
C("#비오는날 바지락칼국수 아니고 들깨칼국수 한 그릇 뜨끈 😌", [I("deulkkae_kalguksu", 1, "그릇")], RD, RV, tags=[NEG])
C("배달 대기하면서 편의점 매일우유 커피맛 하나 마셨다.", [I("maeil_coffee", 1, gunit="개")], RD, BM, tags=[PROD])
C("오늘 점심은 기사식당에서 고추장불고기 1인분에 밥 한 공기 먹었다.", [I("gochujang_bulgogi", 1, "인분"), I("rice", 1, "공기")], RD, BM)
C("집에 와서 군고구마 두 개랑 열무김치 조금 먹었다.", [I("goguma_gui", 2, "개"), I("yeolmu_kimchi")], RD, BM)
C("퇴근길에 옥동자 하나 먹고 집에 와서 꽁치김치찌개에 밥 반 공기 먹었다.",
  [I("okdongja", 1, gunit="개"), I("kkongchi_jjigae"), I("rice", 0.5, "공기")], RD, BM, tags=[PROD])
C("점심: 해물탕 1인분 / 밥1공기", [I("haemul_tang", 1, "인분"), I("rice", 1, "공기")], RD, MM)
C("간식 포테토칩 먹태청양마요 1봉, 콜라x 물만", [I("potato_meoktae", 1, "봉")], RD, MM, tags=[PROD, NEG])
C("아침 X / 점심 김치만두5 / 저녁 울면1 먹엇음", [I("kimchi_mandu", 5, gunit="개"), I("ulmyeon", 1, gunit="그릇")], RD, MM,
  noise=[TYPO])
C("야식 하겐다즈 미니컵 딸기1", [I("haagen", 1, gunit="개")], RD, MM, tags=[PROD])

# ============================== runner (러너) ==============================
C("어 십 킬로 뛰고 와서 음 바나나 두 개랑 그 요거트스무디 한 잔 마셨어", [I("banana", 2, "개"), I("yogurt_smoothie", 1, "잔")],
  RN, DC, tags=[TRAP], conf="medium", note="'십 킬로'는 러닝 거리(양 아님)")
C("음 하프 뛰기 전날이라 그 탄수 로딩으로 크림파스타 한 접시 그리고 어 마들렌 두 개",
  [I("cream_pasta", 1, "접시"), I("madeleine", 2, "개")], RN, DC)
C("그 대회 끝나고 어 뒷풀이로 후라이드 치킨 음 세 조각이랑 치즈떡볶이 조금 먹었어",
  [I("fried_chicken", 3, "조각"), I("cheese_tteok", grams=(80, 150))], RN, DC, conf="medium", note="떡볶이 '조금' 80-150g")
C("어 아침 조깅 전에 그 단감 하나랑 음 아메리카노 반 잔 마시고 뛰었어", [I("dangam", 1, gunit="개"), I("americano", 0.5, "잔")],
  RN, DC, tags=["unit:반"])
C("풀코스 완주하고 돌솥비빔밥 한 그릇이랑 레드향 하나 먹었는데 회복식으로 몇 칼로리 들어간 거야?",
  [I("dolsot", 1, "그릇"), I("redhyang", 1, gunit="개")], RN, AK)
C("러닝 끝나고 고구마라떼 톨사이즈 한 잔 마셨는데 칼로리 얼마나 돼?", [I("goguma_latte", 1, "잔", grams=(340, 370))], RN, AK,
  tags=[SIZE], note="톨 355ml")
C("LSD 30km 뛰고 참치덮밥 곱빼기 먹었는데 이거 몇 칼로리야?", [I("chamchi_deopbap", grams=(600, 750))], RN, AK,
  tags=[TRAP, SIZE], conf="medium", note="'30km'는 거리; 곱빼기 = 1.5그릇 600-750g")
C("인터벌 끝나고 천도복숭아 2개랑 우유 한 팩 먹었는데 칼로리 얼마?", [I("cheondo", 2, "개"), I("milk", 1, "팩")], RN, AK)
C("회복식으로 연포탕 1인분 먹었음. 아 공기밥도 반 공기 추가", [I("yeonpo_tang", 1, "인분"), I("rice", 0.5, "공기")], RN, AT)
C("아침 protein 보충으로 삶은 계란 3개랑 우유 한 컵. 아 계란은 2개였네", [I("egg", 2, "개"), I("milk", 1, "컵")], RN, AT,
  tags=[CORR])
C("저녁 바지락칼국수 한 그릇. 아 그리고 겉절이 대신 부추무침 조금", [I("bajirak_kalguksu", 1, "그릇"), I("buchu_muchim")], RN, AT)
C("레이스 전 에너지 보충 브라우니 한 조각. 아 그리고 아샷추 한 잔도", [I("brownie", 1, "조각"), I("asyachu", 1, "잔")], RN, AT)
C("#러닝크루 #뒷풀이 🍕 페퍼로니 피자 2조각 + 펩시 반 캔 🏃‍♀️", [I("pepperoni", 2, "조각"), I("pepsi", 0.5, "캔")], RN, RV,
  tags=[PROD])
C("#오운완 🏃 10K 완주 기념 mango smoothie 한 잔 🥭", [I("mango_smoothie", 1, "잔")], RN, RV, tags=[TRAP], conf="medium",
  note="'10K'는 러닝 거리")
C("#carbloading 🍝 toowoomba pasta 1인분 싹싹 비움", [I("toowoomba", 1, "인분")], RN, RV)
C("#회복식 🥣 누룽지 한 그릇 + 열무김치 조금", [I("nurungji", 1, "그릇"), I("yeolmu_kimchi")], RN, RV)
C("오늘 아침은 군고구마 하나랑 아메리카노 한 잔으로 때웠다.", [I("goguma_gui", 1, gunit="개"), I("americano", 1, "잔")], RN, BM)
C("점심에는 메밀전 한 장이랑 모밀국수 한 그릇을 먹었다.", [I("memil_jeon", 1, "장"), I("momil", 1, "그릇")], RN, BM)
C("러닝 후에 청포도에이드 한 잔을 마셨다.", [I("cheongpodo_ade", 1, "잔")], RN, BM)
C("저녁은 닭가슴살 대신 가자미구이 반 마리를 먹었다.", [I("gajami", 0.5, "마리")], RN, BM, tags=[NEG, "unit:반"])
C("운동후 바나나1 + 휘낭시애1", [I("banana", 1, gunit="개"), I("financier", 1, gunit="개")], RN, MM, noise=[TYPO])
C("점심 김치전골1인분, 밥 2/3공기", [I("kimchi_jeongol", 1, "인분"), I("rice", 0.6667, "공기")], RN, MM, note="2/3공기 = 0.67공기")
C("대회날 아침: 거봉 10알, 아메리카노", [I("geobong", 10, "알"), I("americano")], RN, MM)
C("보충 요거트스무디1잔 / 삶은계란1", [I("yogurt_smoothie", 1, "잔"), I("egg", 1, gunit="개")], RN, MM)

# ============================== couple (데이트) ==============================
C("어 오늘 데이트하면서 음 당근케이크 한 조각을 둘이 나눠 먹었고 그 토피넛라떼 한 잔 마셨어",
  [I("carrot_cake", (0.5, 1), "조각", grams=(50, 70)), I("toffeenut", 1, "잔")], CP, DC, tags=[SHARE], conf="medium",
  note="케이크 1조각 100-140g의 1/2")
C("음 남자친구랑 그 해물탕 중자 하나 시켜서 반씩 먹었어", [I("haemul_tang", grams=(450, 650))], CP, DC, tags=[SHARE, SIZE],
  conf="medium", note="해물탕 중자(2-3인) 900-1300g의 1/2")
C("그 영화 보면서 어 팝콘 말고 그 오감자 한 봉지 같이 먹었어 나는 한 삼분의 일",
  [I("ogamja", (0.3333, 1), "봉지", grams=(13, 22))], CP, DC, tags=[PROD, SHARE, NEG], conf="medium",
  note="1봉 50g의 약 1/3 ('한 삼분의 일' -> 13-22g)")
C("어 여자친구가 만들어준 고추잡채 한 접시 음 나는 반 정도 먹었어",
  [I("gochu_japchae", (0.5, 1), "접시", grams=(125, 175))], CP, DC, tags=[SHARE], conf="medium", note="1접시 250-350g의 1/2")
C("커플 세트로 치즈돈가스 하나랑 크림파스타 하나 시켜서 반반 나눠 먹었는데 나 몇 칼로리 먹은 거야?",
  [I("cheese_katsu", (0.5, 1), gunit="개", grams=(100, 140)), I("cream_pasta", (0.5, 1), gunit="개", grams=(150, 200))],
  CP, AK, tags=[SHARE], conf="medium", note="각각 1인분의 1/2")
C("여친이랑 투움바파스타 하나랑 레드벨벳 한 조각 나눠 먹었으면 내 몫은 몇 칼로리야?",
  [I("toowoomba", (0.5, 1), gunit="개", grams=(150, 200)), I("redvelvet", (0.5, 1), "조각", grams=(50, 65))], CP, AK,
  tags=[SHARE], conf="medium", note="각각 1/2")
C("남친이 사온 Ferrero 로쉐 3알 먹었는데 칼로리 높아?", [I("ferrero", 3, "알")], CP, AK, tags=[PROD])
C("카페에서 휘낭시에 2개랑 아샷추 한 잔 먹었는데 계산해줄래?", [I("financier", 2, "개"), I("asyachu", 1, "잔")], CP, AK)
C("점심 둘이서 버섯전골 2인분 먹음. 아 볶음밥은 안 먹음", [I("beoseot_jeongol", (1, 2), "인분", grams=(300, 450))], CP, AT,
  tags=[SHARE, NEG], conf="medium", note="2인분을 둘이 -> 1인분")
C("청포도에이드 한 잔이랑 마들렌 하나 먹음. 아 근데 에이드는 여친이 반 뺏어먹음",
  [I("cheongpodo_ade", (0.5, 1), "잔", grams=(175, 250)), I("madeleine", 1, gunit="개")], CP, AT, tags=[SHARE, PART],
  conf="medium", note="에이드 1잔 350-500ml의 1/2")
C("저녁 데이트 난자완스 시켜서 내가 1/3 먹음. 아 그리고 울면도 한 그릇",
  [I("nanjawanseu", (0.3333, 1), gunit="접시", grams=(100, 135)), I("ulmyeon", 1, "그릇")], CP, AT, tags=[SHARE], conf="medium",
  note="난자완스 1접시 300-400g의 1/3")
C("쿠키 대신 브라우니 한 입만 먹음, 아 그리고 커피는 아메리카노", [I("brownie", grams=(10, 25)), I("americano")], CP, AT,
  tags=[NEG], conf="medium", note="'한 입' 10-25g")
C("#데이트 #성수카페 🍰 레드벨벳 케이크 한 조각 + 고구마라떼 ☕️ 행복💕", [I("redvelvet", 1, "조각"), I("goguma_latte")], CP, RV)
C("#100일 🎉 돌솥비빔밥이랑 미나리전 하나 시켜서 나눠먹음 🥰 내 몫 반",
  [I("dolsot", (0.5, 1), gunit="개", grams=(200, 250)), I("minari_jeon", (0.5, 1), gunit="개", grams=(50, 80))],
  CP, RV, tags=[SHARE], conf="medium", note="각각 1/2")
C("#한강데이트 🌉 치토스 한 봉지 + 마운틴 듀 1캔 혼자 다 먹음ㅋㅋ", [I("cheetos", 1, "봉지"), I("mountain_dew", 1, "캔")], CP, RV,
  tags=[PROD], noise=[TYPO])
C("#집데이트 🍳 둘이 김치찜 해먹음 나는 한 그릇 + 오이무침 조금 😋", [I("kimchi_jjim", 1, "그릇"), I("oi_muchim")], CP, RV)
C("여자친구와 함께 들깨칼국수 두 그릇을 시켰고 나는 한 그릇을 다 먹었다. 그리고 메밀전 한 장을 나눠 먹었다.",
  [I("deulkkae_kalguksu", 1, "그릇"), I("memil_jeon", (0.5, 1), "장", grams=(50, 75))], CP, BM, tags=[SHARE], conf="medium",
  note="칼국수는 내 몫 1그릇; 메밀전 1장의 1/2")
C("남자친구가 사준 하겐다즈 미니컵 하나랑 페레로 로쉐 두 알을 먹었다.",
  [I("haagen", 1, gunit="개"), I("ferrero", 2, "알")], CP, BM, tags=[PROD])
C("데이트 끝나고 편의점에서 빵빠레 초코바 하나씩 사 먹었다.", [I("ppangppare", 1, gunit="개")], CP, BM, tags=[PROD],
  note="'하나씩' = 내 몫 1개")
C("둘이서 골뱅이무침 한 접시를 안주로 반씩 나눠 먹었다.", [I("golbaengi", (0.5, 1), "접시", grams=(150, 225))], CP, BM,
  tags=[SHARE], conf="medium", note="1접시 300-450g의 1/2")
C("데이트) 바지락칼국수1 + 물만두 반접시", [I("bajirak_kalguksu", 1, gunit="그릇"), I("mul_mandu", 0.5, "접시")], CP, MM)
C("카페: 당근케이크1조각(혼자), 아메리카노", [I("carrot_cake", 1, "조각"), I("americano")], CP, MM)
C("디저트 휘낭시에 1/2개, 토피넛라떼 1잔", [I("financier", 0.5, "개"), I("toffeenut", 1, "잔")], CP, MM)
C("야식 둘이 낙곱새 2인분 → 나 1인분", [I("nakgopsae", (1, 2), "인분", grams=(300, 450))], CP, MM, tags=[SHARE], conf="medium")
C("데이트 간식: waffle 1 반띵, 청포도에이드1잔(혼자)",
  [I("waffle", (0.5, 1), gunit="개", grams=(40, 60)), I("cheongpodo_ade", 1, "잔")], CP, MM, tags=[SHARE], conf="medium",
  note="와플 1개 80-120g의 1/2")

# ============================== freelancer (재택) ==============================
C("어 마감 끝나고 음 배달로 유산슬덮밥 시켜서 한 그릇 다 먹었어", [I("yusanseul_deopbap", 1, "그릇")], FL, DC)
C("음 그 오전에 작업하면서 고구마라떼 아니 토피넛 라떼 한 잔 마셨고 어 마들렌 하나",
  [I("toffeenut", 1, "잔"), I("madeleine", 1, gunit="개")], FL, DC, tags=[CORR], note="고구마라떼는 정정으로 취소")
C("어 어제 시켜 먹고 남은 고추장불고기 한 반 인분 정도 데워 먹었어", [I("gochujang_bulgogi", 0.5, "인분")], FL, DC,
  tags=[LEFT, "unit:반"])
C("그 점심으로 어 냉장고에 있던 애호박볶음이랑 음 김치볶음 조금에 밥 한 공기",
  [I("aehobak"), I("kimchi_bokkeum"), I("rice", 1, "공기")], FL, DC, tags=[LEFT])
C("재택하면서 에이스 반 곽 먹었는데 이거 몇 칼로리야?", [I("ace", 0.5, "곽")], FL, AK, tags=[PROD, "unit:반"])
C("점심에 잡탕밥 한 그릇 먹고 저녁에 군만두 아니 김치만두 4개 먹었는데 하루 총 몇 kcal야?",
  [I("japtangbap", 1, "그릇"), I("kimchi_mandu", 4, "개")], FL, AK, tags=[CORR], note="군만두는 정정으로 취소")
C("작업하면서 꿀꽈베기 한 봉지 반 먹었는데 칼로리 얼마나 돼?", [I("kkulkkwa", 1.5, "봉지")], FL, AK, tags=[PROD], noise=[TYPO])
C("아침 겸 점심으로 브런치 카페에서 yogurt smoothie 한 잔이랑 carrot cake 한 조각 먹었는데 칼로리 계산해줘",
  [I("yogurt_smoothie", 1, "잔"), I("carrot_cake", 1, "조각")], FL, AK)
C("저녁 고추잡채 1인분에 밥 반공기. 아 꽃빵은 안 먹었어", [I("gochu_japchae", 1, "인분"), I("rice", 0.5, "공기")], FL, AT,
  tags=[NEG])
C("americano 2잔 마심. 아 그리고 오후에 아샷추 한 잔 더", [I("americano", 2, "잔"), I("asyachu", 1, "잔")], FL, AT)
C("점심 김치찜 남은 거 한 그릇 먹음. 아 그리고 두릅무침도 조금", [I("kimchi_jjim", 1, "그릇"), I("dureup")], FL, AT, tags=[LEFT])
C("간식 스니커즈 한 개. 아 정정 반 개만", [I("snickers", 0.5, "개")], FL, AT, tags=[PROD, CORR])
C("#재택러 #점심 🍲 버섯전골 1인분 + 참나물무침 조금 🌿", [I("beoseot_jeongol", 1, "인분"), I("chamnamul")], FL, RV)
C("#마감끝 🎉 투움바파스타 + 청포도에이드 셀프 보상🥂", [I("toowoomba"), I("cheongpodo_ade")], FL, RV,
  note="양 미기재 -> 각 1인분/1잔")
C("#홈카페 ☕ 토피넛라떼 직접 만들어봄 + 브라우니 1조각", [I("toffeenut"), I("brownie", 1, "조각")], FL, RV)
C("#야식 🌙 물만두 15개 + 하이볼은 참음ㅠ", [I("mul_mandu", 15, "개")], FL, RV, tags=[NEG])
C("오늘은 점심에 불낙전골 1인분을 먹고 밥은 반 공기만 먹었다.", [I("bulnak", 1, "인분"), I("rice", 0.5, "공기")], FL, BM)
C("오후에 출출해서 레드향 하나랑 우유 한 잔을 먹었다.", [I("redhyang", 1, gunit="개"), I("milk", 1, "잔")], FL, BM)
C("저녁에는 크림파스타를 만들어서 한 접시 먹엇다.", [I("cream_pasta", 1, "접시")], FL, BM, noise=[TYPO])
C("자기 전에 찰떡아이스 하나를 먹었다.", [I("chaltteok", 1, gunit="개")], FL, BM, tags=[PROD])
C("점심은 어제 남은 연포탕을 데워서 한 그릇 먹었다.", [I("yeonpo_tang", 1, "그릇")], FL, BM, tags=[LEFT])
C("아점: 누룽지1그릇, 무말랭이 약간, 삶은계란1", [I("nurungji", 1, "그릇"), I("mumallaengi"), I("egg", 1, gunit="개")], FL, MM)
C("간식) 하리보 반봉", [I("haribo", 0.5, "봉")], FL, MM, tags=[PROD, "unit:반"])
C("저녁 메밀전1, 막걸리x", [I("memil_jeon", 1, gunit="장")], FL, MM, tags=[NEG])
C("야식 오징어채볶음 조금 + 밥 1/2공기", [I("ojingeochae"), I("rice", 0.5, "공기")], FL, MM)

# ============================== shopowner (자영업 사장님) ==============================
C("어 가게 문 열기 전에 음 그 시장에서 산 꽈리고추멸치볶음이랑 콩자반 말고 그냥 그거랑 밥 한 공기",
  [I("ggwari_myeolchi"), I("rice", 1, "공기")], SO, DC, tags=[NEG])
C("음 손님 없을 때 어 컵라면 새우탕 작은 거 하나 먹었어", [I("shrimp_cup", 1, gunit="개")], SO, DC, tags=[PROD])
C("그 점심 배달로 음 백순대 한 이인분 시켜서 어 알바생이랑 반 나눠 먹고 써니텐 한 캔",
  [I("baek_sundae", (1, 2), "인분", grams=(300, 400)), I("sunnyten", 1, "캔")], SO, DC, tags=[SHARE, PROD], conf="medium",
  note="백순대 2인분을 둘이 -> 1인분")
C("어 마감하고 그 포장마차에서 와플 하나 먹었어 음 생크림 들어간 거 말고 그냥", [I("waffle", 1, gunit="개")], SO, DC,
  tags=[NEG])
C("가게에서 바빠서 짜슐랭 하나에 총각김치 몇 개 먹었는데 몇 칼로리야?",
  [I("jjaseullaeng", 1, gunit="개"), I("chonggak", grams=(60, 130))], SO, AK, tags=[PROD], conf="medium",
  note="총각김치 '몇 개' = 2-4개(통상 3개) x 25-40g -> 60-130g")
C("반찬가게에서 도토리묵무침 한 팩 사서 반 먹었는데 칼로리 얼마나 될까?",
  [I("dotori_muk", (0.5, 1), "팩", grams=(100, 150))], SO, AK, tags=[PART], conf="medium", note="1팩 200-300g의 1/2")
C("시장 칼국수집에서 들깨칼국수 한 그릇 먹고 김치만두 2개 먹으면 칼로리 얼마야?",
  [I("deulkkae_kalguksu", 1, "그릇"), I("kimchi_mandu", 2, "개")], SO, AK)
C("단골이 준 단감 2개랑 거봉 반 송이 먹었는데 칼로리?", [I("dangam", 2, "개"), I("geobong", 0.5, "송이")], SO, AK)
C("점심 유산슬 조금이랑 밥 한 공기. 아 짬뽕 국물도 몇 숟갈 먹었는데 그건 빼줘",
  [I("yusanseul", grams=(80, 150)), I("rice", 1, "공기")], SO, AT, tags=[NEG], conf="medium", note="'조금' 80-150g")
C("옆 가게 사장님이랑 해물파전 아니고 미나리전 한 장 나눠 먹음",
  [I("minari_jeon", (0.5, 1), "장", grams=(50, 80))], SO, AT, tags=[SHARE, NEG], conf="medium", note="1장의 1/2")
C("매일우유 커피맛 하나 마심. 아 그리고 마들렌 하나도",
  [I("maeil_coffee", 1, gunit="개"), I("madeleine", 1, gunit="개")], SO, AT, tags=[PROD])
C("저녁은 김치전골. 아 가게 끝나고 혼자라 1인분만 했음", [I("kimchi_jeongol", 1, "인분")], SO, AT)
C("#자영업자 #점심 🍱 오늘은 가자미구이 한 마리 + 청포묵무침 🌿 맛있당", [I("gajami", 1, "마리"), I("cheongpomuk")], SO, RV)
C("#장사끝 🏪 혼술 안주로 골뱅이무침 반 접시 + 눈을감자 1봉 😋", [I("golbaengi", 0.5, "접시"), I("noonul", 1, "봉")], SO, RV,
  tags=[PROD])
C("#가게일상 😅 바빠서 점심 거르고 오후 4시에 잡탕밥 한 그릇 🔥", [I("japtangbap", 1, "그릇")], SO, RV, tags=[TRAP],
  conf="medium", note="'4시'는 시각")
C("#시장투어 🧆 물만두 한 접시 + 군고구마 1개 🍠", [I("mul_mandu", 1, "접시"), I("goguma_gui", 1, "개")], SO, RV)
C("가게 문 닫고 집에 와서 콩나물국 대신 꽁치김치찌개 한 그릇을 먹었다.", [I("kkongchi_jjigae", 1, "그릇")], SO, BM, tags=[NEG])
C("점심은 시장에서 파는 치즈떡볶이 한 컵이었다.", [I("cheese_tteok", 1, "컵")], SO, BM)
C("손님이 사 온 옥동자 두 개를 먹었다.", [I("okdongja", 2, "개")], SO, BM, tags=[PROD])
C("저녁으로 모밀국수 한 그릇과 김치만두 세 개를 먹었따.", [I("momil", 1, "그릇"), I("kimchi_mandu", 3, "개")], SO, BM, noise=[TYPO])
C("아침 x / 점심 해물덮밥 반그릇 / 간식 자갈치 반봉", [I("haemul_deopbap", 0.5, "그릇"), I("jagalchi", 0.5, "봉")], SO, MM,
  tags=[PROD, "unit:반"])
C("점심 돌솥비빔밥1 (계란x) + 파김치", [I("dolsot", 1, gunit="그릇"), I("pa_kimchi")], SO, MM, tags=[NEG])
C("간식 빵빠레 초코바 2, 매일우유 커피맛1", [I("ppangppare", 2, gunit="개"), I("maeil_coffee", 1, gunit="개")], SO, MM, tags=[PROD])
C("저녁 버섯전골 남은거 1그릇 + 파래무침", [I("beoseot_jeongol", 1, "그릇"), I("parae")], SO, MM, tags=[LEFT])


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
    # Label-audit patch (v6): grams are grams, so a SOLID food with any per-100g gold row takes its kcal from the
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
    def _ex_item_sig(it):
        gold = frozenset(g for g in (it.get("gold_food_ids") or []) if g not in (CUR, LLM))
        names = it.get("names") or [it["key"]]
        return gold, _compact(names[0])

    def _spec_item_sig(key):
        f = R[key]
        gold = frozenset(g for g in f["gold"] if g not in (CUR, LLM)) if f["kind"] == "generic" else frozenset()
        return gold, _compact(f["names"][0])

    def _item_match(a, b):
        return bool(a[0] & b[0]) or a[1] == b[1]

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
                    why = f"food combination repeats earlier v6 {t!r}"
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
                    why = f"jaccard {j:.2f} vs earlier v6 {t!r}"
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

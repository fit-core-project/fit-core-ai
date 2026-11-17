"""Build tests/evaluation/datasets/diet_eval_holdout_v4.jsonl (fourth blind held-out Korean diet-log set).

Purpose: a HELD-OUT set written blind to the pipeline's outputs, to measure generalisation of a parser that was tuned on
diet_eval_v1 / diet_eval_products_v1 / diet_eval_holdout_v1 / diet_eval_holdout_v2 / diet_eval_holdout_v3. Every case is
split "test", tagged set:holdout_v4 plus persona:<x> and style:<y> (and domain:<z> of the main item).

v4 deliberately changes the GENERATOR STYLE so the set does not look like v1-v3:
  personas (even spread): beginner (diet beginner), lifter (bodybuilder, big portions / protein foods),
                          office (busy office worker: convenience store, delivery, coffee, 회식),
                          elderly (traditional / home food, porridge, rice cakes, plain wording),
                          teen (snacks, academy / school food, slang)
  styles (even spread):   telegraphic ("현미죽1 단호박찜 조금"), diary (long run-on), typo (jamo / spelling errors),
                          mixed (English-mixed or romanized Korean), voice (transcript: no punctuation, fillers 어/음/그)
  domains (main item):    regional, temple_home (home / temple side dishes), porridge, rice_cake, health,
                          bakery_cafe, asian (Asian foreign), franchise (franchise / convenience), protein, alcohol

Labels describe what is CORRECT for a real person's meal, never what the code returns. Label sources (only these):
  - data/food_db/food_db_clean.csv          (generic foods + franchise menu rows: gold_food_ids + kcal/100 of those rows)
  - data/food_db/raw/식품의약품안전처_통합식품영양성분정보(가공식품)_20261111.csv
                                             (branded/packaged products and alcohol: label values, asserted at build)
  - public common-knowledge Korean portion sizes (stated in each registry note / basis)
engines/*, data/curated/*, processed_product_index.csv, LLM caches, baselines and eval reports were NOT consulted.

Schema = diet_eval_holdout_v3.jsonl (same conventions as build_diet_holdout_v3.py):
  - names: acceptable food names (incl. romanized / English aliases used in mixed-style cases)
  - gold_food_ids: acceptable food_db_clean rows (+ "curated" / "llm_estimate_ok" tokens)
  - amount: literal quantity [lo, hi] (null = no number). Sharing / partial-eating: [per-person share, literal total];
    explicit ranges ("서너 개"): [lo, hi] of the stated range.
  - units: accepted unit tokens; [] when the text has a bare number without a unit word ("현미죽1") -- grams then use
    the implied unit stated in basis
  - grams [lo, hi]: public-standard portion range (ml treated as g); always the amount the person actually ate
  - kcal [lo, hi]: generic = min/max DB kcal/100 x grams, widened 10% and +-5 kcal;
                   product = min/max MFDS label kcal/100 x single-unit size x quantity, widened 10%
Completeness (asserted at build): every food_db row whose name is IDENTICAL to a gold row's name, and every MFDS row
whose FOOD_NM is identical to a transcribed product name, is either in gold or listed in the key's `excl` dict with a
reason (the reason is copied into basis).
Near-duplicate texts (bigram Jaccard >= 0.4 vs the five existing sets, the scratch inputs, or an earlier v4 case) are
DROPPED before ids are assigned.

Run:  PYTHONUTF8=1 python scripts/eval/build_diet_holdout_v4.py
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
OUT_PATH = DATASETS / "diet_eval_holdout_v4.jsonl"
EXISTING = [DATASETS / "diet_eval_v1.jsonl", DATASETS / "diet_eval_products_v1.jsonl",
            DATASETS / "diet_eval_holdout_v1.jsonl", DATASETS / "diet_eval_holdout_v2.jsonl",
            DATASETS / "diet_eval_holdout_v3.jsonl"]
TEMPLATE = DATASETS / "diet_eval_holdout_v3.jsonl"

KCAL_TOL = 0.10
NEAR_JACCARD = 0.4  # cases at or above this vs existing / earlier v4 texts are dropped
KCAL_ABS = 5.0
LLM = "llm_estimate_ok"
CUR = "curated"
G_ML = {"g": (1, 1), "ml": (1, 1)}
SET_TAG = "set:holdout_v4"
ID_PREFIX = "dh4_"

PERSONAS = ("beginner", "lifter", "office", "elderly", "teen")
STYLES = ("telegraphic", "diary", "typo", "mixed", "voice")
DOMAINS = ("staple", "regional", "temple_home", "porridge", "rice_cake", "health", "bakery_cafe", "asian", "franchise",
           "protein", "alcohol")


# ---------------------------------------------------------------------------
# Registry.  units: grams per ONE unit (lo, hi); serving: portion when no number is given.
#   F(...)  generic food / franchise menu: gold = DB ids (+ optional tokens); excl = {id: reason} for same-name rows left out
#   P(...)  branded product / alcohol: mfds = [(FOOD_CD, kcal_per_100, "exact FOOD_NM")], units = single-unit size
# ---------------------------------------------------------------------------
def F(group, domain, names, gold, units=None, serving=None, note=None, excl=None):
    assert domain in DOMAINS, domain
    return dict(kind="generic", group=group, domain=domain, names=names, gold=gold, units=units or {}, serving=serving,
                est=None, est_basis=None, note=note, excl=excl or {})


def P(group, domain, names, mfds, units, serving, note, gold=(LLM, CUR), excl=None):
    assert domain in DOMAINS, domain
    return dict(kind="product", group=group, domain=domain, names=names, gold=list(gold), units=units, serving=serving,
                mfds=mfds, note=note, excl=excl or {})


JUK = {"그릇": (300, 450), "bowl": (300, 450), **G_ML}
JUK_NOTE = "죽 1그릇 300-450g(가정식~죽 전문점 보통)"
NAMUL = {"접시": (60, 100), **G_ML}
NAMUL_NOTE = "나물·무침 반찬 1회 40-70g, 1접시 60-100g"
GUK = {"그릇": (250, 350), **G_ML}
GUK_NOTE = "국 1그릇 250-350g"

R = {
    # ---------------- reused staples (definitions copied from build_diet_holdout_v3.py; side items only) ----------------
    "rice": F("밥", "staple", ["쌀밥", "밥", "공기밥", "공깃밥", "흰쌀밥", "흰밥", "rice"], ["음식_011600", "음식_011599", CUR],
              units={"공기": (190, 230), "그릇": (190, 230), "개": (190, 230), "bowl": (190, 230), **G_ML}, serving=(190, 230),
              note="밥 1공기 210g(식약처 1회 제공량)"),
    "egg": F("계란유제품", "staple", ["계란", "달걀", "삶은계란", "삶은 계란", "egg", "eggs"], [CUR, "원재_001702", "음식_017044", "원재_001700"],
             units={"개": (50, 60), "알": (50, 60)}, serving=(50, 60)),
    "americano": F("음료", "staple", ["아메리카노", "아아", "아이스 아메리카노", "아이스아메리카노", "커피", "americano", "iced americano"],
                   ["음식_009004", "음식_008945", "음식_008940", "음식_009005", "음식_008946"], units={"잔": (300, 500)}, serving=(300, 500)),
    "fries": F("패스트푸드", "staple", ["감자튀김", "프렌치프라이", "fries", "후렌치후라이"], ["음식_017952", "음식_017951"], units={"개": (100, 130)},
               serving=(100, 130), note="미디엄 100-130g"),
    "cola": P("음료", "staple", ["콜라", "코카콜라", "coke"],
              [("P109-401040100-0479", 43, "코카콜라"), ("P109-401040100-2381", 44, "코카콜라"), ("P109-401040100-2382", 44, "코카콜라")],
              {"캔": (250, 355), "잔": (200, 300), "개": (250, 355)}, (250, 355),
              "세트 콜라 M 250-355ml; P109-401040100-1982(코카-콜라 158/100ml)는 1캔값 오기로 보고 제외"),

    # ---------------- 죽 (porridge) ----------------
    "heugimja_juk": F("국찌개", "porridge", ["흑임자죽", "검은깨죽", "heugimja-juk", "heugimjajuk", "black sesame porridge"],
                      ["음식_001859"], units=JUK, serving=(300, 450), note=JUK_NOTE),
    "deulkkae_juk": F("국찌개", "porridge", ["들깨죽", "들깨쥭"], ["음식_016252"], units=JUK, serving=(300, 450), note=JUK_NOTE),
    "hyeonmi_juk": F("국찌개", "porridge", ["현미죽", "hyeonmi juk", "brown rice porridge"], ["음식_001987"], units=JUK,
                     serving=(300, 450), note=JUK_NOTE),
    "baekhap_juk": F("국찌개", "porridge", ["백합죽", "조개죽"], ["음식_014352"], units=JUK, serving=(300, 450), note=JUK_NOTE),
    "gul_juk": F("국찌개", "porridge", ["굴죽", "gul-juk", "oyster porridge"], ["음식_017594"], units=JUK, serving=(300, 450),
                 note=JUK_NOTE),
    "yulmu_juk": F("국찌개", "porridge", ["율무죽"], ["음식_010241"], units=JUK, serving=(300, 450), note=JUK_NOTE),
    "mieum": F("국찌개", "porridge", ["미음", "쌀미음"], ["음식_014762", "음식_014760", "음식_014761"],
               units={"그릇": (200, 300), **G_ML}, serving=(200, 300), note="환자식 미음 1그릇 200-300g; 현미/칠분도미 미음 행 포함"),
    "eojuk": F("국찌개", "regional", ["어죽", "생선국수죽"], ["음식_010899"], units={"그릇": (500, 700), **G_ML}, serving=(500, 700),
               note="충청 금강변 어죽 1그릇 500-700g"),
    # ---------------- 떡 / 전통과자 (rice cake) ----------------
    "susu_bukkumi": F("빵과자", "regional", ["수수부꾸미", "부꾸미"], ["음식_012938"], units={"개": (50, 80), "장": (50, 80)},
                      serving=(100, 160), note="강원 수수부꾸미 1개 50-80g"),
    "susu_pattteok": F("빵과자", "rice_cake", ["수수팥떡", "수수팥경단", "수수경단"], ["음식_012937", "음식_012941"], units={"개": (20, 35)},
                       serving=(60, 105), note="수수팥떡(경단형) 1개 20-35g; 수수경단 행 포함"),
    "ssuk_tteok": F("빵과자", "rice_cake", ["쑥떡", "ssuk-tteok", "mugwort rice cake"], ["음식_011576"],
                    units={"개": (30, 50), "pieces": (30, 50), "조각": (30, 50)}, serving=(60, 100), note="쑥떡 1개(1조각) 30-50g"),
    "gyeongdan": F("빵과자", "rice_cake", ["경단", "콩고물 경단", "깨경단"], ["음식_017899", "음식_017901"], units={"개": (15, 25)},
                   serving=(60, 100), note="경단 1개 15-25g; 카스텔라 경단(음식_017900)은 다른 고물이라 제외"),
    "mosit_songpyeon": F("빵과자", "regional", ["모싯잎송편", "모시잎송편", "모시송편", "mosit-ip songpyeon", "songpyeon"],
                         ["음식_014879", "음식_014880", "음식_014881"], units={"개": (35, 50)}, serving=(105, 150),
                         note="영광 모싯잎송편 1개 35-50g(일반 송편보다 큼)"),
    "maejakgwa": F("빵과자", "rice_cake", ["매작과", "타래과"], ["음식_015004"], units={"개": (8, 15)}, serving=(24, 45),
                   note="매작과 1개 8-15g"),
    "yugwa": F("빵과자", "rice_cake", ["유과"], ["음식_010314"], units={"개": (5, 15)}, serving=(15, 45), note="유과 1개 5-15g"),
    "bingtteok": F("빵과자", "regional", ["빙떡", "메밀전병", "제주 빙떡"], ["음식_013627"], units={"개": (60, 90)}, serving=(60, 90),
                   note="제주 빙떡 1개 60-90g"),
    # ---------------- 사찰·가정 반찬 / 국 / 밥 (temple / home) ----------------
    "siraegi_guk": F("국찌개", "temple_home", ["시래기 된장국", "시래기국", "시래기된장국"], ["음식_011724", "음식_016326", "음식_016327"],
                     units=GUK, serving=(250, 350), note=GUK_NOTE + "; '된장국 시래기' 행 포함"),
    "auk_guk": F("국찌개", "temple_home", ["아욱국", "아욱 된장국"], ["음식_011565", "음식_016323", "음식_016324"], units=GUK,
                 serving=(250, 350), note=GUK_NOTE + "; 건새우 아욱국(음식_011564)은 다른 이름 행"),
    "gosari": F("채소샐러드", "temple_home", ["고사리나물", "고사리", "고사리 namul", "gosari namul"], ["음식_017862", "음식_017861"],
                units=NAMUL, serving=(40, 70), note=NAMUL_NOTE),
    "bireum": F("채소샐러드", "temple_home", ["비름나물"], ["음식_013755", "음식_013751", "음식_013752", "음식_013753", "음식_013754"],
                units=NAMUL, serving=(40, 70), note=NAMUL_NOTE + "; 양념별 비름나물 행 포함"),
    "meowi": F("채소샐러드", "temple_home", ["머위나물", "머위"], ["음식_015003", "음식_015002", "음식_015001"], units=NAMUL,
               serving=(40, 70), note=NAMUL_NOTE),
    "chwinamul": F("채소샐러드", "temple_home", ["취나물", "취나물무침"], ["음식_009922", "음식_009923", "음식_009920", "음식_009921"],
                   units=NAMUL, serving=(40, 70), note=NAMUL_NOTE),
    "hobak_ogari": F("채소샐러드", "temple_home", ["호박오가리나물", "호박고지나물", "호박오가리나물무침"], ["음식_001968"], units=NAMUL,
                     serving=(40, 70), note=NAMUL_NOTE),
    "gogumajulgi": F("채소샐러드", "temple_home", ["고구마줄기나물", "고구마순나물"],
                     ["음식_017883", "음식_017884", "음식_017881", "음식_017882"], units=NAMUL, serving=(40, 70), note=NAMUL_NOTE),
    "deodeok_gui": F("채소샐러드", "temple_home", ["더덕구이", "더덕"], ["음식_016735", "음식_016736"], units={"접시": (80, 120), **G_ML},
                     serving=(50, 100), note="더덕구이 반찬 50-100g, 1접시 80-120g"),
    "godeulppaegi": F("채소샐러드", "temple_home", ["고들빼기김치", "고들빼기"], ["음식_017877"], units=G_ML, serving=(30, 60),
                      note="김치 반찬 1회 30-60g"),
    "nabak_kimchi": F("채소샐러드", "temple_home", ["나박김치"], ["음식_017141"], units={"그릇": (150, 250), **G_ML}, serving=(100, 200),
                      note="국물김치 1회 100-200g"),
    "dongchimi": F("채소샐러드", "temple_home", ["동치미", "동치미 국물"], ["음식_016412"], units={"그릇": (200, 300), **G_ML},
                   serving=(150, 250), note="동치미(국물 포함) 1회 150-250g"),
    "baek_kimchi": F("채소샐러드", "temple_home", ["백김치"], ["음식_014355"], units=G_ML, serving=(40, 80), note="김치 반찬 1회 40-80g"),
    "gat_kimchi": F("채소샐러드", "temple_home", ["갓김치"], ["음식_017932"], units=G_ML, serving=(30, 60), note="김치 반찬 1회 30-60g"),
    "gajami_sikhae": F("생선", "regional", ["가자미식해", "가자미식혜"], ["음식_018043"], units=G_ML, serving=(30, 60),
                       note="함경·강원 가자미식해 반찬 1회 30-60g"),
    "baechujeon": F("빵과자", "regional", ["배추전", "배추적", "배추찌짐"], ["음식_014356", "음식_014357"],
                    units={"장": (80, 150), "개": (80, 150)}, serving=(80, 150), note="경상도 배추전 1장 80-150g"),
    "ogok_bap": F("밥", "temple_home", ["오곡밥", "ogokbap", "five-grain rice"], ["음식_010673", "음식_010674"],
                  units={"공기": (190, 230), "그릇": (190, 230), "gonggi": (190, 230), **G_ML}, serving=(190, 230), note="밥 1공기 190-230g"),
    "yeonip_bap": F("밥", "temple_home", ["연잎밥"], ["음식_010689"], units={"개": (200, 300)}, serving=(200, 300),
                    note="연잎밥 1개 200-300g"),
    "yeongyang_dolsot": F("밥", "temple_home", ["영양돌솥밥", "돌솥밥"], ["음식_010676", "음식_010677"], units={"그릇": (300, 400)},
                          serving=(300, 400), note="영양돌솥밥 1그릇 300-400g"),
    "gwiri_bap": F("밥", "health", ["귀리밥", "gwiri-bap", "oat rice"], ["음식_017592"],
                   units={"공기": (190, 230), "그릇": (190, 230), "bowl": (190, 230), **G_ML}, serving=(190, 230),
                   note="밥 1공기 190-230g; 원재 귀리밥(도정) 행은 다른 이름"),
    "jogi_jjim": F("생선", "temple_home", ["조기찜", "참조기찜"], ["음식_010064", "음식_010065"], units={"마리": (80, 120)},
                   serving=(80, 120), note="조기 1마리 가식부 80-120g"),
    "momguk": F("국찌개", "regional", ["몸국", "모자반국"], ["음식_014877"], units={"그릇": (500, 700)}, serving=(500, 700),
                note="제주 몸국 1그릇 500-700g"),
    "olgaengi_guksu": F("면", "regional", ["올갱이국수", "다슬기국수"], ["음식_010565"], units={"그릇": (600, 800)}, serving=(600, 800),
                        note="충북 올갱이국수 1그릇 600-800g"),
    "gegukji": F("국찌개", "regional", ["게국지", "게국지찌개"], ["음식_017909"], units={"인분": (400, 600), "그릇": (400, 600)},
                 serving=(400, 600), note="서산 게국지 1인분 400-600g"),
    "ojingeo_sundae": F("고기", "regional", ["오징어순대"], ["음식_010595", "음식_010596"], units={"접시": (200, 300), "마리": (200, 300)},
                        serving=(200, 300), note="속초 오징어순대 1접시(1마리) 200-300g"),
    "chodang_sundubu": F("국찌개", "regional", ["초당순두부", "강릉 초당순두부", "순두부"], ["음식_009952"], units={"그릇": (300, 400)},
                         serving=(300, 400), note="초당순두부(백순두부) 1그릇 300-400g"),
    # ---------------- 건강식 (health) ----------------
    "danhobak_jjim": F("채소샐러드", "health", ["단호박찜", "찐 단호박"], ["음식_017046"],
                       units={"개": (500, 800), "통": (500, 800), "조각": (80, 120), **G_ML}, serving=(100, 200),
                       note="단호박 1통 500-800g, 1조각 80-120g"),
    "danhobak_salad": F("채소샐러드", "health", ["단호박 샐러드", "단호박샐러드", "pumpkin salad"], ["음식_017048"],
                        units={"접시": (100, 150), **G_ML}, serving=(80, 150), note="샐러드 1접시 100-150g"),
    "saessak_salad": F("채소샐러드", "health", ["새싹 샐러드", "새싹샐러드", "saessak salad", "sprout salad"], ["음식_013430"],
                       units={"접시": (100, 150), **G_ML}, serving=(80, 150), note="샐러드 1접시 100-150g"),
    "yangbaechu_salad": F("채소샐러드", "health", ["양배추 샐러드", "양배추샐러드", "코울슬로"], ["음식_010942"],
                          units={"접시": (100, 150), **G_ML}, serving=(80, 150), note="샐러드 1접시 100-150g"),
    "broccoli_dubu": F("채소샐러드", "health", ["브로콜리 두부무침", "브로콜리두부무침"], ["음식_013760"], units=NAMUL, serving=(50, 100),
                       note=NAMUL_NOTE),
    "gonyak_jorim": F("채소샐러드", "health", ["곤약조림", "곤약 조림"], ["음식_017840"], units=G_ML, serving=(50, 100),
                      note="조림 반찬 1회 50-100g"),
    "yeondubu_jjim": F("채소샐러드", "health", ["연두부찜", "연두부"], ["음식_010694"], units={"그릇": (150, 250), **G_ML},
                       serving=(150, 250), note="연두부 1팩(150-250g) 찜 1그릇"),
    "oi_naengguk": F("국찌개", "health", ["오이냉국", "oi naengguk"], ["음식_010634", "음식_010635"], units=GUK, serving=(250, 350),
                     note=GUK_NOTE + "; 미역 넣은 오이냉국(음식_010633)은 다른 이름 행"),
    "miyeok_naengguk": F("국찌개", "health", ["미역냉국"], ["음식_014773"], units=GUK, serving=(200, 300), note=GUK_NOTE),
    "sukju": F("채소샐러드", "health", ["숙주나물", "숙주", "sukju namul"], ["음식_012926", "음식_012927"], units=NAMUL, serving=(40, 70),
               note=NAMUL_NOTE),
    "fruit_salad": F("과일", "health", ["과일샐러드", "과일 샐러드", "fruit salad"], ["음식_017625"],
                     units={"컵": (150, 250), "접시": (150, 250), **G_ML}, serving=(150, 250), note="과일샐러드 1컵 150-250g"),
    "kombucha": F("음료", "health", ["콤부차", "kombucha"], ["음식_007629", "음식_007630"],
                  units={"병": (300, 500), "잔": (300, 500), "bottle": (300, 500)}, serving=(300, 500),
                  note="플레인 콤부차 1병/1잔 300-500ml"),
    "dubu_poke": F("채소샐러드", "health", ["두부포케 샐러드", "두부포케", "두부 포케", "tofu poke salad", "tofu poke"], ["음식_013181"],
                   units={"개": (300, 400), "그릇": (300, 400)}, serving=(300, 400), note="포케 1볼 300-400g"),
    # ---------------- 단백질 (protein) ----------------
    "so_ansim": F("고기", "protein", ["소안심스테이크", "안심스테이크", "안심 스테이크", "tenderloin steak"], ["음식_012950"],
                  units={"인분": (150, 250), "덩이": (150, 250), **G_ML}, serving=(150, 250), note="스테이크 1인분 150-250g"),
    "chilmyeonjo": F("고기", "protein", ["칠면조구이", "칠면조", "칠면조 가슴살", "turkey breast", "turkey"], ["음식_009767"],
                     units={"인분": (150, 200), **G_ML}, serving=(150, 200)),
    "dwaeji_steak": F("고기", "protein", ["돼지고기 스테이크", "돼지 스테이크", "포크스테이크"], ["음식_016380"],
                      units={"인분": (200, 300), "덩이": (200, 300), **G_ML}, serving=(200, 300), note="포크스테이크 1인분 200-300g"),
    "saetae_jjim": F("고기", "protein", ["사태찜", "소사태찜", "saetae-jjim"], ["음식_013461"],
                     units={"인분": (250, 350), "serving": (250, 350)}, serving=(250, 350), note="사태찜 1인분 250-350g"),
    "omelet": F("계란유제품", "protein", ["오믈렛", "omelette", "omelet"], ["음식_010657"], units={"개": (150, 250)}, serving=(150, 250),
                note="계란 3개 오믈렛 1개 150-250g"),
    "sutran": F("계란유제품", "protein", ["수란", "poached egg", "poached eggs"], ["음식_012943"], units={"개": (50, 60), "알": (50, 60)},
                serving=(50, 60), note="수란 1개 50-60g"),
    "dalgyal_salad": F("계란유제품", "protein", ["달걀 샐러드", "에그샐러드", "egg salad"], ["음식_017045"],
                       units={"접시": (100, 150), **G_ML}, serving=(100, 150), note="샐러드 1접시 100-150g"),
    "mechurial": F("계란유제품", "protein", ["메추리알장조림", "메추리알 장조림"], ["음식_014940"],
                   units={"알": (10, 14), "개": (10, 14), **G_ML}, serving=(60, 100),
                   note="메추리알장조림 1알 10-14g(양념 포함); 꽈리고추 행(음식_014939)은 다른 이름"),
    "so_shabu": F("국찌개", "protein", ["소고기 샤브샤브", "샤브샤브", "소고기샤브샤브"], ["음식_013084", "음식_013067"],
                  units={"인분": (400, 600)}, serving=(400, 600), note="샤브샤브 1인분(고기+채소+국물) 400-600g"),
    "chicken_teriyaki": F("고기", "protein", ["치킨데리야끼", "치킨 데리야끼", "데리야끼 치킨"], ["음식_009768", "음식_009769"],
                          units={"인분": (200, 300), "접시": (200, 300)}, serving=(200, 300), note="1인분 200-300g"),
    # ---------------- 베이커리 / 카페 (bakery_cafe) ----------------
    "scone": F("빵과자", "bakery_cafe", ["스콘", "플레인 스콘", "scone"], ["음식_011816", "음식_011819"], units={"개": (80, 120)},
               serving=(80, 120), note="카페 스콘 1개 80-120g; 플레인/클래식 스콘 행"),
    "egg_tart": F("빵과자", "bakery_cafe", ["에그타르트", "에그 타르트", "egg tart"], ["음식_007161", "음식_007162", "음식_007159"],
                  units={"개": (50, 75)}, serving=(50, 75), note="에그타르트 1개 50-75g"),
    "canele": F("빵과자", "bakery_cafe", ["까눌레", "카눌레", "canele", "canelé", "canelés"], ["음식_017556", "음식_014996"],
                units={"개": (40, 60)}, serving=(40, 60), note="까눌레 1개 40-60g"),
    "muffin": F("빵과자", "bakery_cafe", ["머핀", "muffin"], ["음식_014998", "음식_014990"], units={"개": (90, 130)}, serving=(90, 130),
                note="카페 머핀 1개 90-130g"),
    "soy_latte": F("음료", "bakery_cafe", ["두유 카페라떼", "두유라떼", "두유 라떼", "soy latte"], ["음식_016255", "음식_016254"],
                   units={"잔": (355, 473), "컵": (355, 473)}, serving=(355, 473), note="카페 톨~그란데 355-473ml"),
    "ice_mocha": F("음료", "bakery_cafe", ["아이스 카페모카", "카페모카", "iced mocha"], ["음식_011560", "음식_011557"],
                   units={"잔": (355, 473), "컵": (355, 473)}, serving=(355, 473), note="카페 톨~그란데 355-473ml"),
    # ---------------- 아시아 외국음식 (asian) ----------------
    "shio_ramen": F("면", "asian", ["시오라멘", "시오 라멘", "shio ramen", "라멘"], ["음식_015331"], units={"그릇": (600, 800)},
                    serving=(600, 800), note="라멘 1그릇(국물 포함) 600-800g"),
    "shoyu_ramen": F("면", "asian", ["소유라멘", "쇼유라멘", "shoyu ramen", "라멘"], ["음식_015332"], units={"그릇": (600, 800)},
                     serving=(600, 800), note="라멘 1그릇(국물 포함) 600-800g"),
    "miso_ramen": F("면", "asian", ["미소라멘", "미소 라멘", "miso ramen", "라멘"], ["음식_015333"], units={"그릇": (600, 800)},
                    serving=(600, 800), note="라멘 1그릇(국물 포함) 600-800g"),
    "jungsik_udon": F("면", "asian", ["중식우동", "중국집 우동"], ["음식_010046", "음식_010351"], units={"그릇": (700, 900)},
                      serving=(700, 900), note="중식 우동 1그릇 700-900g; '우동 중식' 행 포함"),
    "yangjangpi": F("배달", "asian", ["양장피"], ["음식_010932", "음식_010933"], units={"접시": (400, 600)}, serving=(150, 250),
                    note="양장피 1접시 400-600g, 여럿이 나눌 때 1인 몫 100-250g"),
    "kkanpung_saeu": F("배달", "asian", ["깐풍새우"], ["음식_017210"], units={"접시": (300, 450)}, serving=(300, 450),
                       note="깐풍새우 1접시 300-450g"),
    "california": F("빵과자", "asian", ["캘리포니아롤", "캘리포니아 롤"], ["음식_009707"], units={"줄": (200, 260), "개": (25, 32)},
                    serving=(200, 260), note="캘리포니아롤 1줄(8피스) 200-260g"),
    "shrimp_roll": F("빵과자", "asian", ["새우튀김롤"], ["음식_013409"], units={"줄": (220, 280)}, serving=(220, 280),
                     note="새우튀김롤 1줄 220-280g"),
    "jangeo_chobap": F("빵과자", "asian", ["장어초밥", "장어 초밥"], ["음식_010114"], units={"개": (30, 40)}, serving=(30, 40),
                       note="초밥 1피스 30-40g"),
    "banhmi": F("빵과자", "asian", ["반미", "반미 샌드위치", "banh mi"], ["음식_013198", "음식_013219", "음식_013305", "음식_013315"],
                units={"개": (200, 280)}, serving=(200, 280), note="반미 1개 200-280g; 속재료별 반미 샌드위치 행"),
    # ---------------- 프랜차이즈 / 길거리 (franchise; food_db 음식 행) ----------------
    "tandoori_thigh": F("패스트푸드", "franchise", ["탄두리싸이버거", "탄두리싸이 버거", "탄두리 싸이버거", "tandoori thigh burger", "버거"],
                        ["음식_014079"], units={"개": (230, 270)}, serving=(230, 270), note="맘스터치 싸이버거류 1개 230-270g(통상 중량 추정)"),
    "bul_thigh": F("패스트푸드", "franchise", ["불싸이버거", "불싸이 버거", "버거"], ["음식_014209"], units={"개": (230, 270)},
                   serving=(230, 270), note="맘스터치 싸이버거류 1개 230-270g(통상 중량 추정)"),
    "volcano_chicken": F("배달", "franchise", ["볼케이노 치킨", "볼케이노치킨", "치킨"], ["음식_016871"],
                         units={"조각": (40, 60), "마리": (600, 800)}, serving=(250, 350), note="치킨 1조각 40-60g(통상 중량 추정)"),
    "cheese_ball": F("빵과자", "franchise", ["치즈볼", "치즈 볼"], ["음식_009859", "음식_009853"], units={"개": (15, 25), "알": (15, 25)},
                     serving=(75, 125), note="치킨집 치즈볼 1개 15-25g"),
    "conchi": F("빵과자", "franchise", ["콘치즈구이", "콘치즈", "콘치즈 구이"], ["음식_007639", "음식_007640"],
                units={"접시": (150, 250)}, serving=(150, 250), note="콘치즈 1접시(철판) 150-250g"),
    "sotteok": F("빵과자", "franchise", ["소떡소떡", "소떡", "sotteok sotteok"], ["음식_016238", "음식_016241", "음식_016239"],
                 units={"개": (100, 150), "stick": (100, 150), "꼬치": (100, 150)}, serving=(100, 150),
                 note="휴게소 소떡소떡 1꼬치 100-150g"),
    "pizza_bread": F("빵과자", "bakery_cafe", ["피자빵", "pizza bread"], ["음식_002387"], units={"개": (100, 150)}, serving=(100, 150),
                     note="피자빵 1개 100-150g"),
    "sausage_bread": F("빵과자", "bakery_cafe", ["소시지빵", "소세지빵"], ["음식_012972", "음식_012973"], units={"개": (100, 150)},
                       serving=(100, 150), note="소시지빵 1개 100-150g"),
    "pretzel": F("빵과자", "bakery_cafe", ["프레즐", "프레첼", "pretzel"], ["음식_006883", "음식_006836"], units={"개": (80, 130)},
                 serving=(80, 130), note="프레즐 1개 80-130g"),
    "gamjabbang": F("빵과자", "regional", ["감자빵", "춘천 감자빵"], ["음식_017969", "음식_017967", "음식_017968"], units={"개": (90, 120)},
                    serving=(90, 120), note="춘천 감자빵 1개 90-120g"),
    "oven_spaghetti": F("면", "franchise", ["오븐스파게티", "오븐 스파게티", "oven spaghetti"], ["음식_010647", "음식_010644"],
                        units={"개": (300, 400)}, serving=(300, 400), note="피자집 오븐스파게티 1개 300-400g"),
}

# ---------------- 브랜드 제품 (MFDS 가공식품 라벨; 각 행은 빌드 시 원본 CSV와 대조) ----------------
PR = {
    # protein products
    "pinksalt": P("고기", "protein", ["랭킹닭컴 고독한 닭가슴살 핑크솔트", "닭가슴살 핑크솔트", "고독한 닭가슴살", "닭가슴살"],
                  [("P117-100010100-0533", 98, "랭킹닭컴 고독한 닭가슴살 핑크솔트"),
                   ("P117-100010100-1205", 98, "랭킹닭컴 고독한 닭가슴살 핑크솔트")],
                  {"팩": (100, 100), "개": (100, 100)}, (100, 100), "1팩 100g"),
    "dakball_spicy": P("고기", "protein", ["맛있닭 닭가슴살 볼 스파이시", "닭가슴살볼 스파이시", "맛있닭 닭가슴살볼", "닭가슴살볼"],
                       [("P117-502050200-0063", 165, "맛있닭 닭가슴살 볼 스파이시")], {"팩": (100, 100), "개": (100, 100)}, (100, 100),
                       "1팩 100g"),
    "easyprotein_bar": P("빵과자", "protein", ["이지프로틴 고단백질바", "이지프로틴 바", "고단백질바", "프로틴바"],
                         [("P101-106000100-2454", 325, "이지프로틴 고단백질바")], {"개": (50, 60)}, (50, 60),
                         "1개 50-60g(MFDS 행 FOOD_SIZE 60g)"),
    "selex_nutsbar": P("빵과자", "protein", ["셀렉스 프로틴 너츠바", "셀렉스 너츠바", "selex protein nuts bar", "프로틴바"],
                       [("P101-106000100-4420", 490, "셀렉스 프로틴 너츠바"), ("P101-106000100-3585", 490, "셀렉스 프로틴 너츠바")],
                       {"개": (50, 50)}, (50, 50), "1개 50g(300g 행은 6입)"),
    "himmune_bar": P("빵과자", "protein", ["하이뮨 액티브 프로틴 바 너티초코", "하이뮨 액티브 프로틴바", "하이뮨 프로틴바", "프로틴바"],
                     [("P101-106000100-8097", 500, "하이뮨 액티브 프로틴 바 너티초코")], {"개": (50, 50)}, (50, 50), "1개 50g"),
    "mitri_bar": P("빵과자", "protein", ["미트리 저당 프로틴바 카라멜 초코", "미트리 저당 프로틴바", "mitri low sugar protein bar", "프로틴바"],
                   [("P103-102020500-4850", 414, "미트리 저당 프로틴바 카라멜 초코")], {"개": (40, 40)}, (40, 40), "1개 40g"),
    "dakjuice": P("음료", "protein", ["REAL 닭주스 오리지널 닭가슴살 쉐이크", "닭주스", "REAL 닭주스", "real dak juice", "닭가슴살 쉐이크"],
                  [("P123-231020300-0010", 88, "REAL 닭주스 오리지널 닭가슴살 쉐이크"),
                   ("P123-231020300-0011", 78, "REAL 닭주스 오리지널 닭가슴살 쉐이크")],
                  {"개": (200, 200), "팩": (200, 200)}, (200, 200), "1팩 200g"),
    "calobye": P("음료", "protein", ["CALOBYE퍼펙트파워쉐이크초코맛", "칼로바이 퍼펙트파워쉐이크 초코", "CALOBYE 초코 쉐이크", "단백질 쉐이크"],
                 [("P119-204020400-0011", 62, "CALOBYE퍼펙트파워쉐이크초코맛")], {"팩": (250, 250), "개": (250, 250)}, (250, 250),
                 "1팩 250ml"),
    "easyprotein_choco": P("음료", "protein", ["이지프로틴 저당초코", "이지프로틴 초코", "단백질 음료"],
                           [("P119-204020400-0546", 44, "이지프로틴 저당초코")], {"팩": (250, 250), "개": (250, 250)}, (250, 250),
                           "1팩 250ml"),
    # convenience-store meals
    "mitdak_dosirak": P("편의점브랜드", "franchise", ["맛있닭 다이어트 도시락 카레곤약볶음밥 & 닭가슴살볼 오리지널", "맛있닭 다이어트 도시락",
                                                     "맛있닭 도시락", "다이어트 도시락"],
                        [("P110-500050000-0397", 141, "맛있닭 다이어트 도시락 카레곤약볶음밥 & 닭가슴살볼 오리지널")],
                        {"개": (210, 210)}, (210, 210), "1개 210g(텍스트에 맛 미기재 -> 카레곤약볶음밥 행)"),
    "rk_oat_rice": P("편의점브랜드", "franchise", ["랭킹닭컴 고독한 계란 귀리볶음밥 갈비맛", "고독한 계란 귀리볶음밥", "귀리볶음밥 갈비맛",
                                                  "랭킹닭컴 귀리볶음밥"],
                     [("P123-201020300-5774", 175, "랭킹닭컴 고독한 계란 귀리볶음밥 갈비맛")], {"개": (200, 200), "팩": (200, 200)},
                     (200, 200), "1팩 200g"),
    # snacks / ice cream / drinks
    "bints": P("편의점브랜드", "franchise", ["빈츠", "bints"],
               [("P103-102020500-4802", 529, "빈츠"), ("P103-102020500-4129", 563, "빈츠"), ("P103-102020500-0155", 563, "빈츠")],
               {"갑": (76, 102), "box": (76, 102)}, (76, 102), "빈츠 1갑 76-102g(204g 행은 대용량)"),
    "babamba": P("편의점브랜드", "franchise", ["바밤바", "아이스크림"],
                 [("P102-104010400-0576", 143, "바밤바"), ("P102-104010400-0483", 143, "바밤바"), ("P102-301030000-0111", 151, "바밤바"),
                  ("P102-104010400-0321", 143, "바밤바")], {"개": (63, 70)}, (63, 70), "바밤바 1개 70ml(63g)"),
    "supercone": P("편의점브랜드", "franchise", ["빙그레슈퍼콘바닐라", "슈퍼콘 바닐라", "빙그레 슈퍼콘", "supercone vanilla", "아이스크림"],
                   [("P102-103010100-0001", 160, "빙그레슈퍼콘바닐라")], {"개": (150, 150)}, (150, 150), "슈퍼콘 1개 150ml"),
    "yomamte": P("편의점브랜드", "health", ["요맘때 플레인 바", "요맘때", "yomamte plain bar", "아이스크림"],
                 [("P102-104010400-0385", 136, "요맘때 플레인 바"), ("P102-104010400-0183", 136, "요맘때 플레인 바")],
                 {"개": (70, 70)}, (70, 70), "요맘때 바 1개 70ml"),
    "polapo_coffee": P("편의점브랜드", "franchise", ["폴라포 커피 제로슈거", "폴라포 커피", "폴라포 coffee zero", "폴라포"],
                       [("P102-301030000-0468", 58, "폴라포 커피 제로슈거")], {"개": (120, 120)}, (120, 120), "폴라포 1개 120ml"),
    "coolpis": P("음료", "franchise", ["쿨피스 바나나", "쿨피스", "coolpis banana", "coolpis"], [("P109-601060100-0306", 32, "쿨피스 바나나")],
                 {"개": (300, 300), "병": (300, 300)}, (300, 300), "쿨피스 바나나 300ml"),
    "oronamin": P("음료", "franchise", ["오로나민C", "오로나민씨", "오로나민"],
                  [("P109-401040100-2482", 58, "오로나민C"), ("P109-401040100-0370", 58, "오로나민C")],
                  {"개": (120, 120), "병": (120, 120)}, (120, 120), "오로나민C 1병 120ml"),
    "solui_nun": P("음료", "franchise", ["솔의눈", "솔의 눈"], [("P109-801080100-5421", 31, "솔의눈"), ("P109-801080100-0772", 31, "솔의눈")],
                   {"캔": (240, 240), "개": (240, 240), "can": (240, 240), "a": (240, 240)}, (240, 240), "솔의눈 1캔 240ml"),
    "selex_dangsolve": P("음료", "health", ["셀렉스 프로틴 당솔브 음료 오곡맛", "셀렉스 당솔브", "셀렉스 프로틴 당솔브", "당솔브"],
                         [("P109-801080100-6016", 79, "셀렉스 프로틴 당솔브 음료 오곡맛")],
                         {"병": (190, 190), "개": (190, 190), "팩": (190, 190)}, (190, 190), "1개 190ml(맛 미기재 -> 오곡맛 행)"),
    # alcohol
    "joeunday": P("주류", "alcohol", ["좋은데이", "좋은데이 소주", "소주"],
                  [("P115-102020100-0015", 89, "좋은데이"), ("P115-102020100-0014", 89, "좋은데이"), ("P115-102020100-0016", 89, "좋은데이"),
                   ("P115-102020100-0013", 89, "좋은데이")], {"병": (360, 360), "잔": (45, 55)}, (360, 360),
                  "소주 1병 360ml, 소주잔 45-55ml"),
    "siwon": P("주류", "alcohol", ["시원", "시원 소주", "C1 소주", "소주"], [("P115-102020100-0005", 100, "시원")],
               {"병": (360, 360), "잔": (45, 55)}, (360, 360), "소주 1병 360ml"),
    "kangali": P("주류", "alcohol", ["강알리", "강알리 소주", "소주"], [("P115-102020100-0002", 89, "강알리")],
                 {"병": (360, 360), "잔": (45, 55)}, (360, 360), "소주 1병 360ml"),
    "zero_lager": P("주류", "alcohol", ["제로슈거라거", "제로슈거 라거", "zero sugar lager", "맥주"], [("P115-104010400-0006", 25, "제로슈거라거")],
                    {"캔": (500, 500), "개": (500, 500)}, (500, 500), "500ml 캔"),
    "zeromac": P("주류", "alcohol", ["제로맥", "맥주"], [("P115-104010400-0005", 25, "제로맥")], {"캔": (500, 500), "개": (500, 500)},
                 (500, 500), "500ml 캔"),
    "hongsam_makgeolli": P("주류", "alcohol", ["홍삼막걸리", "홍삼 막걸리", "장홍삼장수막걸리", "막걸리"],
                           [("P115-101010100-0015", 40, "홍삼막걸리"), ("P115-101010100-0014", 40, "장홍삼장수막걸리")],
                           {"병": (750, 750), "사발": (250, 300), "잔": (150, 250), "그릇": (250, 300)}, (250, 300),
                           "막걸리 1병 750ml, 1사발 250-300ml"),
}
assert not (set(R) & set(PR))
R.update(PR)
EXTRA_DENS: dict = {}  # v4: no density outside DB/MFDS rows


# ---------------------------------------------------------------------------
# Case DSL
# ---------------------------------------------------------------------------
def I(key, amt=None, unit=None, grams=None, optional=False, units_ok=None, note=None, low=False, gunit=None):
    """gunit = implied unit for a bare number without a unit word ("현미죽1"): used for grams only, units -> []."""
    assert unit is None or gunit is None
    return dict(key=key, amt=amt, unit=unit, grams=grams, optional=optional, units_ok=units_ok, note=note, low=low,
                gunit=gunit)


CASES: list[dict] = []


def C(text, items, persona, style, noise=(), tags=(), conf="high", note=None, max_items=None, min_items=None):
    assert persona in PERSONAS, persona
    assert style in STYLES, style
    CASES.append(dict(text=text, items=items, persona=persona, style=style, noise=list(noise), extra_tags=list(tags),
                      conf=conf, note=note, max_items=max_items, min_items=min_items))


SHARE = "flag:share"
SIZE = "flag:size"
PROD = "flag:product"
RANGE = "flag:range"
PART = "flag:partial"
NEG = "flag:negation"
CORR = "flag:correction"
B, L, O, E, T = PERSONAS
TEL, DIA, TYP, MIX, VOI = STYLES

# ============================== beginner dieter ==============================
C("현미죽1 단호박찜 조금", [I("hyeonmi_juk", 1, gunit="그릇"), I("danhobak_jjim")], B, TEL, conf="medium")
C("아점: 귀리밥 반공기 + 숙주나물", [I("gwiri_bap", 0.5, "공기"), I("sukju")], B, TEL)
C("요맘때 바1", [I("yomamte", 1, gunit="개")], B, TEL, tags=[PROD])
C("콤부차 1병, 두부포케 1", [I("kombucha", 1, "병"), I("dubu_poke", 1, gunit="개")], B, TEL)
C("새싹샐러드 1접시 / 수란2", [I("saessak_salad", 1, "접시"), I("sutran", 2, gunit="개")], B, TEL)
C("저녁 맛있닭도시락1 끝", [I("mitdak_dosirak", 1, gunit="개")], B, TEL, tags=[PROD], conf="medium")

C("다이어트 3일차인데 아침은 오이냉국 한 그릇이랑 밥 반 공기만 먹었고 점심엔 회사 앞에서 두부포케 샐러드 하나 먹었는데 생각보다 "
  "배불러서 저녁은 건너뛸까 고민중이다",
  [I("oi_naengguk", 1, "그릇"), I("rice", 0.5, "공기"), I("dubu_poke", 1, "개")], B, DIA)
C("오늘은 진짜 착하게 먹었다 아침에 이지프로틴 저당초코 한 팩 마시고 점심은 곤약조림이랑 귀리밥 한 공기 저녁은 단호박 샐러드 조금 "
  "먹고 끝",
  [I("easyprotein_choco", 1, "팩"), I("gonyak_jorim"), I("gwiri_bap", 1, "공기"), I("danhobak_salad")], B, DIA, tags=[PROD])
C("친구들이랑 브런치 갔는데 나는 참으려고 새싹 샐러드만 시켰고 친구들 와플은 한 입도 안 먹었다 대신 아메리카노 한 잔 마심 뿌듯",
  [I("saessak_salad"), I("americano", 1, "잔")], B, DIA, tags=[NEG], noise=["부정"])
C("운동 시작하고 처음으로 저녁을 가볍게 먹어봤는데 브로콜리 두부무침 한 접시랑 미역냉국 한 그릇 먹으니까 의외로 괜찮았고 "
  "디저트로 과일샐러드 반 컵 정도 먹었다",
  [I("broccoli_dubu", 1, "접시"), I("miyeok_naengguk", 1, "그릇"), I("fruit_salad", 0.5, "컵")], B, DIA)
C("주말이라 엄마가 영양돌솥밥 해주셨는데 다이어트 중이라 반만 먹고 반은 남겨뒀다 반찬은 백김치 조금",
  [I("yeongyang_dolsot", (0.5, 1), "그릇", grams=(150, 200)), I("baek_kimchi")], B, DIA, tags=[PART], note="1그릇 300-400g의 절반")
C("편의점에서 한참 고민하다가 결국 요맘때 플레인 바 하나 집었는데 아이스크림 치고 칼로리 낮다길래 샀음 쿨피스는 참았다",
  [I("yomamte", 1, "개")], B, DIA, tags=[PROD, NEG], noise=["부정"])

C("연두부찜 한그룻 먹엇슴", [I("yeondubu_jjim", 1, "그릇")], B, TYP, noise=["오타"])
C("단호박쩜 반개 먹엇어여", [I("danhobak_jjim", 0.5, "개")], B, TYP, noise=["오타"], conf="medium",
  note="단호박 1통 500-800g의 절반")
C("곤약죠림 이랑 현미쥭 한 그릇", [I("gonyak_jorim"), I("hyeonmi_juk", 1, "그릇")], B, TYP, noise=["오타"], conf="medium")
C("양배추셀러드 반 졉시 + 숙쥬나물", [I("yangbaechu_salad", 0.5, "접시"), I("sukju")], B, TYP, noise=["오타"])
C("점심 맛잇닭 다이어트 도시락 하나 먹엇당", [I("mitdak_dosirak", 1, "개")], B, TYP, tags=[PROD], noise=["오타"], conf="medium")
C("초당순두부 한 그룻 + 갓김치 쪼금", [I("chodang_sundubu", 1, "그릇"), I("gat_kimchi")], B, TYP, noise=["오타"])

C("breakfast: hyeonmi juk 1 bowl", [I("hyeonmi_juk", 1, "그릇", units_ok=["그릇", "bowl"])], B, MIX, noise=["영어"])
C("lunch로 tofu poke salad 하나 먹음 so healthy", [I("dubu_poke", 1, "개")], B, MIX, noise=["영어"])
C("dinner = 단호박 salad 1 접시 + kombucha 1 bottle",
  [I("danhobak_salad", 1, "접시"), I("kombucha", 1, "병", units_ok=["병", "bottle"])], B, MIX, noise=["영어"])
C("had saessak salad and 2 poached eggs", [I("saessak_salad"), I("sutran", 2, "개", units_ok=["개", "eggs"])], B, MIX,
  noise=["영어"], conf="medium")
C("diet day 5, ate gwiri-bap half bowl w/ sukju namul",
  [I("gwiri_bap", 0.5, "공기", units_ok=["공기", "그릇", "bowl"]), I("sukju")], B, MIX, noise=["영어"], conf="medium")
C("snack: yomamte plain bar 1개", [I("yomamte", 1, "개")], B, MIX, tags=[PROD], noise=["영어"])

C("어 오늘 아침에는 음 오이냉국 그 한 그릇 먹었고 어 그리고 점심은 그 초당순두부 먹었어요",
  [I("oi_naengguk", 1, "그릇"), I("chodang_sundubu")], B, VOI, noise=["음성"])
C("음 그러니까 저녁은 그 맛있닭 도시락 아 아니 두 개 아니고 한 개요 한 개", [I("mitdak_dosirak", 1, "개")], B, VOI,
  tags=[PROD, CORR], noise=["음성", "정정"], conf="medium")
C("어 브로콜리 두부무침이랑 음 미역냉국 조금 그리고 어 콤부차 한 병",
  [I("broccoli_dubu"), I("miyeok_naengguk"), I("kombucha", 1, "병")], B, VOI, noise=["음성"], conf="medium")
C("그 뭐지 연두부찜 반 그릇 정도 먹고 어 나머지는 남겼어요", [I("yeondubu_jjim", (0.5, 1), "그릇", grams=(75, 125))], B, VOI,
  tags=[PART], noise=["음성"])
C("어 이지프로틴 저당초코 한 팩 마시고 음 그게 다예요", [I("easyprotein_choco", 1, "팩")], B, VOI, tags=[PROD], noise=["음성"])
C("음 과일샐러드 한 컵이랑 어 단호박찜 두 조각", [I("fruit_salad", 1, "컵"), I("danhobak_jjim", 2, "조각")], B, VOI, noise=["음성"])

# ============================== heavy lifter / bodybuilder ==============================
C("닭가슴살 핑크솔트2 귀리밥1", [I("pinksalt", 2, gunit="팩"), I("gwiri_bap", 1, gunit="공기")], L, TEL, tags=[PROD])
C("안심스테이크 300g + 밥2공기", [I("so_ansim", 300, "g"), I("rice", 2, "공기")], L, TEL)
C("칠면조 200g, 계란6", [I("chilmyeonjo", 200, "g"), I("egg", 6, gunit="개")], L, TEL)
C("이지프로틴바 2개 닭주스1", [I("easyprotein_bar", 2, "개"), I("dakjuice", 1, gunit="팩")], L, TEL, tags=[PROD])
C("사태찜 2인분 혼자 클리어", [I("saetae_jjim", 2, "인분")], L, TEL, noise=["슬랭"])
C("오믈렛2 메추리알장조림 15알", [I("omelet", 2, gunit="개"), I("mechurial", 15, "알")], L, TEL)

C("하체날이라 아침부터 수란 네 개 먹고 점심엔 돼지고기 스테이크 1인분에 밥 두 공기 저녁에는 소고기 샤브샤브 2인분 혼자 다 "
  "먹었는데 그래도 단백질이 모자란 느낌이라 자기 전에 REAL 닭주스 하나 더 마심",
  [I("sutran", 4, "개"), I("dwaeji_steak", 1, "인분"), I("rice", 2, "공기"), I("so_shabu", 2, "인분"), I("dakjuice", 1, "개")],
  L, DIA, tags=[PROD])
C("벌크 중이라 점심에 치킨데리야끼 2인분이랑 공깃밥 두 개 먹고 운동 끝나고 셀렉스 프로틴 너츠바 하나에 CALOBYE 초코 쉐이크 "
  "한 팩까지 먹었다",
  [I("chicken_teriyaki", 2, "인분"), I("rice", 2, "개", units_ok=["개", "공기"]), I("selex_nutsbar", 1, "개"),
   I("calobye", 1, "팩")], L, DIA, tags=[PROD])
C("대회 준비 중이라 식단 철저하게 하는 중인데 오늘은 칠면조구이 150g씩 세 번 먹었고 사이사이 달걀 샐러드 한 접시랑 브로콜리 "
  "두부무침 조금 먹었음",
  [I("chilmyeonjo", 450, "g", note="150g x 3회"), I("dalgyal_salad", 1, "접시"), I("broccoli_dubu")], L, DIA, conf="medium",
  note="칠면조 150g x 3 = 450g")
C("오늘 등 운동하고 헬스장 앞 고깃집에서 돼지고기 스테이크 두 덩이 먹었는데 한 덩이에 250g 정도라고 했고 거기에 밥은 한 공기만 "
  "먹고 대신 수란 두 개 추가했다",
  [I("dwaeji_steak", 2, "덩이", grams=(450, 550), units_ok=["덩이", "개"]), I("rice", 1, "공기"), I("sutran", 2, "개")], L, DIA,
  note="250g 정도 x 2덩이")
C("치팅데이라 맛있닭 닭가슴살볼 스파이시 두 팩을 그냥 간식처럼 까먹고 저녁엔 소안심스테이크 1인분 먹고 하이뮨 액티브 프로틴바 "
  "너티초코 하나로 마무리",
  [I("dakball_spicy", 2, "팩"), I("so_ansim", 1, "인분"), I("himmune_bar", 1, "개")], L, DIA, tags=[PROD])
C("벌크 시즌에 엄마가 사태찜을 한 솥 해놓으셔서 오늘 세 번에 나눠서 먹었는데 다 합치면 한 2인분은 될 듯하고 밥은 매번 한 "
  "공기씩 총 세 공기 먹었다",
  [I("saetae_jjim", 2, "인분"), I("rice", 3, "공기")], L, DIA, conf="medium", note="본인 추정 2인분")

C("닭가슴쌀 핑크솔트 3팩 먹움", [I("pinksalt", 3, "팩")], L, TYP, tags=[PROD], noise=["오타"])
C("칠면죠구이 250그람", [I("chilmyeonjo", 250, "g", units_ok=["g", "그람", "그램"])], L, TYP, noise=["오타"])
C("오물렛 계란4개짜리 하나", [I("omelet", 1, "개", grams=(220, 320))], L, TYP, noise=["오타"], conf="medium",
  note="계란 4개 오믈렛 220-320g")
C("올갱이국쑤 곱배기 + 공기밥 2", [I("olgaengi_guksu", grams=(900, 1200)), I("rice", 2, gunit="공기")], L, TYP,
  tags=[SIZE, "unit:크기"], noise=["오타"], conf="medium", note="곱빼기 = 보통 1.5배")
C("이지프로틴 고단벡질바 2개 해치움", [I("easyprotein_bar", 2, "개")], L, TYP, tags=[PROD], noise=["오타"])
C("소안심스태이크 반근", [I("so_ansim", 0.5, "근", grams=(290, 310), units_ok=["근"])], L, TYP, noise=["오타"], conf="medium",
  note="고기 한 근 600g의 반 = 300g")

C("post-workout: REAL dak juice 1 + selex protein nuts bar 1",
  [I("dakjuice", 1, gunit="팩"), I("selex_nutsbar", 1, gunit="개")], L, MIX, tags=[PROD], noise=["영어"])
C("bulking meal: tenderloin steak 400g, rice 2 bowls",
  [I("so_ansim", 400, "g"), I("rice", 2, "공기", units_ok=["공기", "bowl", "bowls"])], L, MIX, noise=["영어"])
C("turkey breast 300g이랑 egg salad 한 접시", [I("chilmyeonjo", 300, "g"), I("dalgyal_salad", 1, "접시")], L, MIX, noise=["영어"])
C("ate 2 omelettes (3 eggs each) after leg day", [I("omelet", 2, "개", units_ok=["개", "omelettes"])], L, MIX, noise=["영어"])
C("dinner: saetae-jjim 1 serving + 공기밥", [I("saetae_jjim", 1, "인분", units_ok=["인분", "serving"]), I("rice")], L, MIX,
  noise=["영어"])
C("Mitri low sugar protein bar 2개 before gym", [I("mitri_bar", 2, "개")], L, MIX, tags=[PROD], noise=["영어"])

C("어 운동 끝나고 음 닭가슴살 그 핑크솔트 한 팩이랑 어 이지프로틴 고단백질바 하나 먹었어요",
  [I("pinksalt", 1, "팩"), I("easyprotein_bar", 1, "개")], L, VOI, tags=[PROD], noise=["음성"])
C("음 점심은 그 소고기 샤브샤브 어 2인분 시켜서 형이랑 나눠 먹었고요", [I("so_shabu", (1, 2), "인분", grams=(400, 600))], L, VOI,
  tags=[SHARE], noise=["음성"], note="2인분/2명 = 1인분")
C("어 저녁에 치킨데리야끼 음 한 접시 그리고 밥 어 두 공기 아 세 공기였나 세 공기요",
  [I("chicken_teriyaki", 1, "접시"), I("rice", 3, "공기")], L, VOI, tags=[CORR], noise=["음성", "정정"])
C("그 뭐냐 메추리알장조림 한 스무 알 정도 집어먹고 어 수란 세 개", [I("mechurial", 20, "알"), I("sutran", 3, "개")], L, VOI,
  noise=["음성"], conf="medium", note="'한 스무 알 정도' = 약 20알")
C("음 맛있닭 닭가슴살볼 스파이시 그거 한 팩 반 먹었어요", [I("dakball_spicy", 1.5, "팩")], L, VOI, tags=[PROD], noise=["음성"])
C("어 CALOBYE 초코 쉐이크 두 팩 마시고 어 하이뮨 프로틴바 하나", [I("calobye", 2, "팩"), I("himmune_bar", 1, "개")], L, VOI,
  tags=[PROD], noise=["음성"], conf="medium")

# ============================== busy office worker ==============================
C("점심 시오라멘, 저녁 굶음", [I("shio_ramen")], O, TEL, tags=[NEG], noise=["부정"])
C("아아x, 두유라떼1 + 스콘1", [I("soy_latte", 1, gunit="잔"), I("scone", 1, gunit="개")], O, TEL, tags=[NEG], noise=["부정"])
C("편의점 귀리볶음밥 갈비맛1", [I("rk_oat_rice", 1, gunit="개")], O, TEL, tags=[PROD], conf="medium")
C("회식: 좋은데이 1병, 양장피, 깐풍새우", [I("joeunday", 1, "병"), I("yangjangpi", grams=(100, 250)),
                                   I("kkanpung_saeu", grams=(100, 225))], O, TEL, tags=[PROD, SHARE], conf="low",
  note="회식 공유 안주: 1인 몫 양장피 100-250g, 깐풍새우 100-225g 추정")
C("캘리포니아롤 1줄 + 장어초밥 4p", [I("california", 1, "줄"), I("jangeo_chobap", 4, "개", units_ok=["개", "p", "피스"])], O, TEL)
C("아이스 카페모카 벤티", [I("ice_mocha", grams=(580, 600))], O, TEL, tags=[SIZE, "unit:크기"], conf="medium", note="벤티 591ml")

C("야근하다가 배고파서 회사 근처 일본 라멘집 가서 미소라멘 한 그릇 먹고 들어왔는데 국물까지 다 마셔버려서 내일 얼굴 부을 듯 "
  "그리고 편의점에서 폴라포 커피 제로슈거 하나 사서 먹으면서 다시 일함",
  [I("miso_ramen", 1, "그릇"), I("polapo_coffee", 1, "개")], O, DIA, tags=[PROD])
C("오늘 점심은 팀장님이 쏘신다고 해서 중국집 갔는데 중식우동 한 그릇 먹었고 다같이 깐풍새우 하나 시켜서 다섯 명이 나눠먹었다",
  [I("jungsik_udon", 1, "그릇"), I("kkanpung_saeu", (0.2, 1), "접시", grams=(60, 90))], O, DIA, tags=[SHARE],
  note="깐풍새우 1접시 300-450g / 5명")
C("아침에 지하철 타기 전에 카페에서 두유 카페라떼 큰 거 하나 사고 사무실 와서 까눌레 두 개를 먹었는데 점심때까지 배가 안 고팠다",
  [I("soy_latte", 1, "잔", grams=(473, 591)), I("canele", 2, "개")], O, DIA, tags=[SIZE, "unit:크기"], conf="medium",
  note="큰 거 = 그란데~벤티 473-591ml")
C("외근 나갔다가 시간이 애매해서 베트남 반미 샌드위치 하나 테이크아웃해서 차에서 먹고 오후엔 탕비실 솔의눈 한 캔 마셨다",
  [I("banhmi", 1, "개"), I("solui_nun", 1, "캔")], O, DIA)
C("금요일이라 팀 사람들이랑 이자카야 가서 콘치즈구이랑 새우튀김롤 시키고 나는 제로슈거라거 두 캔 마셨는데 안주는 셋이서 "
  "나눠 먹었음",
  [I("conchi", (0.33, 1), "접시", grams=(50, 83)), I("shrimp_roll", (0.33, 1), "줄", grams=(73, 93)),
   I("zero_lager", 2, "캔")], O, DIA, tags=[SHARE, PROD], conf="medium", note="안주 1접시/1줄을 3명이 나눔")
C("재택하는 날이라 점심을 대충 랭킹닭컴 계란 귀리볶음밥 하나 데워 먹고 오후 세 시쯤 당 떨어져서 머핀 하나랑 아이스 카페모카 "
  "한 잔 먹었다",
  [I("rk_oat_rice", 1, "개"), I("muffin", 1, "개"), I("ice_mocha", 1, "잔")], O, DIA, tags=[PROD])

C("점심 소유라맨 한그릇 국물 쪼금", [I("shoyu_ramen", 1, "그릇")], O, TYP, noise=["오타"])
C("캘리포니아롤 한줄 머겄다", [I("california", 1, "줄")], O, TYP, noise=["오타"])
C("스콘 두개랑 아메리까노", [I("scone", 2, "개"), I("americano")], O, TYP, noise=["오타"])
C("두유라떼 한잔 머핀 하나 먹어씀", [I("soy_latte", 1, "잔"), I("muffin", 1, "개")], O, TYP, noise=["오타"])
C("서산 출장와서 게국지 일인붕 먹엇음", [I("gegukji", 1, "인분", units_ok=["인분", "일인분"])], O, TYP, noise=["오타"],
  conf="medium")
C("회식 2차에서 강알리 반병 마셔씀", [I("kangali", 0.5, "병")], O, TYP, tags=[PROD, "unit:반"], noise=["오타"])

C("lunch: shio ramen 1 bowl, no rice", [I("shio_ramen", 1, "그릇", units_ok=["그릇", "bowl"])], O, MIX, tags=[NEG],
  noise=["영어", "부정"])
C("grabbed a banh mi and iced mocha for lunch", [I("banhmi", 1, "개", units_ok=["개", "a"]), I("ice_mocha")], O, MIX,
  noise=["영어"], conf="medium")
C("after work: zero sugar lager 2 cans w/ 콘치즈", [I("zero_lager", 2, "캔", units_ok=["캔", "cans"]), I("conchi")], O, MIX,
  tags=[PROD], noise=["영어"], conf="medium")
C("2 canelés + 1 soy latte at the café", [I("canele", 2, gunit="개"), I("soy_latte", 1, gunit="잔")], O, MIX, noise=["영어"])
C("miso ramen 곱빼기 시킴", [I("miso_ramen", grams=(900, 1200))], O, MIX, tags=[SIZE, "unit:크기"], noise=["영어"],
  conf="medium", note="곱빼기 = 보통 1.5배")
C("snack: 폴라포 coffee zero 1개", [I("polapo_coffee", 1, "개")], O, MIX, tags=[PROD], noise=["영어"])

C("어 점심은 음 그 회사 앞에서 소유라멘 먹었고 어 저녁은 아직이요", [I("shoyu_ramen")], O, VOI, noise=["음성"])
C("그 아침에 어 스콘 하나랑 음 두유 라떼 한 잔 아 아니다 스콘 두 개", [I("scone", 2, "개"), I("soy_latte", 1, "잔")], O, VOI,
  tags=[CORR], noise=["음성", "정정"])
C("음 회식에서 그 시원 소주 한 병 정도 마시고 어 양장피 좀 먹은 것 같아요",
  [I("siwon", 1, "병"), I("yangjangpi", grams=(100, 250), low=True)], O, VOI, tags=[PROD], noise=["음성"], conf="low",
  note="양장피 1인 몫 100-250g 추정")
C("어 랭킹닭컴 그 귀리볶음밥 갈비맛 하나 전자레인지 돌려 먹었어요", [I("rk_oat_rice", 1, "개")], O, VOI, tags=[PROD], noise=["음성"])
C("음 그 머핀 반 개 먹고 반 개는 옆자리 줬어요", [I("muffin", (0.5, 1), "개", grams=(45, 65))], O, VOI, tags=[PART],
  noise=["음성"])
C("어 퇴근하고 제로맥 한 캔 그리고 어 캘리포니아롤 반 줄", [I("zeromac", 1, "캔"), I("california", 0.5, "줄")], O, VOI,
  tags=[PROD], noise=["음성"])

# ============================== elderly ==============================
C("아침 흑임자죽 한그릇", [I("heugimja_juk", 1, "그릇")], E, TEL)
C("점심 오곡밥, 고사리나물, 시래기국", [I("ogok_bap"), I("gosari"), I("siraegi_guk")], E, TEL)
C("간식 수수부꾸미 2", [I("susu_bukkumi", 2, gunit="개")], E, TEL)
C("저녁 어죽 반그릇", [I("eojuk", 0.5, "그릇")], E, TEL)
C("쑥떡3 솔의눈1", [I("ssuk_tteok", 3, gunit="개"), I("solui_nun", 1, gunit="캔")], E, TEL)
C("아욱국 밥 한공기 조기찜 한마리", [I("auk_guk"), I("rice", 1, "공기"), I("jogi_jjim", 1, "마리")], E, TEL)

C("아침에 일어나서 영감이랑 들깨죽 한 그릇씩 먹고 점심에는 경로당에서 할멈들이 모싯잎송편 쪄와서 세 개 얻어먹었다 저녁은 입맛이 "
  "없어서 동치미 국물에 밥 반 공기 말아 먹었다",
  [I("deulkkae_juk", 1, "그릇"), I("mosit_songpyeon", 3, "개"), I("dongchimi"), I("rice", 0.5, "공기")], E, DIA)
C("손주들 왔다 가서 남은 연잎밥 하나 데워 먹고 반찬은 더덕구이 조금이랑 고들빼기김치 몇 점 먹었다 요새는 많이 못 먹는다",
  [I("yeonip_bap", 1, "개"), I("deodeok_gui"), I("godeulppaegi")], E, DIA)
C("절에 다녀왔는데 점심 공양으로 비름나물이랑 머위나물 취나물 고사리 이렇게 나물 반찬에 밥 한 공기 먹고 왔다 맛이 담백하니 좋았다",
  [I("bireum"), I("meowi"), I("chwinamul"), I("gosari"), I("rice", 1, "공기")], E, DIA, conf="medium",
  note="사찰 공양 나물 각 40-70g")
C("정월 대보름이라 오곡밥 한 공기에 호박오가리나물이랑 고구마줄기나물 해서 먹었다 부럼은 이가 안 좋아서 못 깨물었다",
  [I("ogok_bap", 1, "공기"), I("hobak_ogari"), I("gogumajulgi")], E, DIA, tags=[NEG], noise=["부정"])
C("제주 사는 딸네 집에 왔는데 사돈이 몸국을 끓여주셔서 한 그릇 비우고 오후엔 시장에서 빙떡 두 개 사서 하나는 내가 먹고 하나는 "
  "딸 줬다",
  [I("momguk", 1, "그릇"), I("bingtteok", (1, 2), "개", grams=(60, 90))], E, DIA, tags=[PART], note="빙떡 본인 1개")
C("속이 안 좋아서 하루 종일 미음만 먹었다 아침 점심 저녁 한 그릇씩 해서 세 그릇", [I("mieum", 3, "그릇")], E, DIA)

C("들깨쥭 한 그릇 먹엇다", [I("deulkkae_juk", 1, "그릇")], E, TYP, noise=["오타"])
C("모시잎송편 다섯개", [I("mosit_songpyeon", 5, "개")], E, TYP, noise=["오타", "띄어쓰기"])
C("백합쥭 반 그륵", [I("baekhap_juk", 0.5, "그릇")], E, TYP, tags=["unit:반"], noise=["오타"])
C("수수팟떡 네 개 먹음", [I("susu_pattteok", 4, "개")], E, TYP, noise=["오타"])
C("시래기 된쟝국에 밥 말아서", [I("siraegi_guk"), I("rice")], E, TYP, noise=["오타"])
C("가자미식혜 쪼끔이랑 배추젼 한 장", [I("gajami_sikhae"), I("baechujeon", 1, "장")], E, TYP, noise=["오타"], conf="medium")

C("grandma had 2 bowls of mieum (rice gruel) bc 속이 안좋아서", [I("mieum", 2, "그릇", units_ok=["그릇", "bowls"])], E, MIX,
  noise=["영어"])
C("grandpa ate 3 songpyeon (mosit-ip) today", [I("mosit_songpyeon", 3, gunit="개")], E, MIX, noise=["영어"], conf="medium")
C("lunch는 gul-juk 한 그릇, 나박김치 조금", [I("gul_juk", 1, "그릇"), I("nabak_kimchi")], E, MIX, noise=["영어"])
C("ssuk-tteok 2 pieces with 솔의눈 a can",
  [I("ssuk_tteok", 2, "개", units_ok=["개", "pieces"]), I("solui_nun", 1, "캔", units_ok=["캔", "can", "a"])], E, MIX,
  noise=["영어"])
C("dinner: ogokbap 1 gonggi + gosari namul", [I("ogok_bap", 1, "공기", units_ok=["공기", "gonggi"]), I("gosari")], E, MIX,
  noise=["영어"])
C("mom drank 셀렉스 protein 당솔브 1병 today", [I("selex_dangsolve", 1, "병")], E, MIX, tags=[PROD], noise=["영어"],
  conf="medium")

C("어 아침에 그 흑임자죽 음 반 그릇 먹고 어 그리고 오로나민씨 하나", [I("heugimja_juk", 0.5, "그릇"), I("oronamin", 1, "개")], E, VOI,
  tags=[PROD], noise=["음성"])
C("음 점심에 그 몸국 어 한 그릇 먹었지 그리고 어 뭐더라 빙떡 하나", [I("momguk", 1, "그릇"), I("bingtteok", 1, "개")], E, VOI,
  noise=["음성"])
C("어 경단 그 콩고물 묻은 거 음 다섯 개쯤 먹었어", [I("gyeongdan", 5, "개")], E, VOI, noise=["음성"], conf="medium")
C("그 뭐야 율무죽 한 그릇 아 반 그릇 반 그릇만 먹었어", [I("yulmu_juk", 0.5, "그릇")], E, VOI, tags=[CORR, "unit:반"],
  noise=["음성", "정정"])
C("어 매작과 서너 개 하고 음 유과 두 개", [I("maejakgwa", (3, 4), "개"), I("yugwa", 2, "개")], E, VOI, tags=[RANGE],
  noise=["음성"])
C("음 아들이 홍삼막걸리 사 와서 어 한 사발 마셨어", [I("hongsam_makgeolli", 1, "사발", units_ok=["사발", "잔", "그릇"])], E, VOI,
  tags=[PROD], noise=["음성"])

# ============================== teenager ==============================
C("학원끝 소떡소떡1 바밤바1", [I("sotteok", 1, gunit="개"), I("babamba", 1, gunit="개")], T, TEL, tags=[PROD])
C("피자빵2 쿨피스1", [I("pizza_bread", 2, gunit="개"), I("coolpis", 1, gunit="개")], T, TEL, tags=[PROD])
C("치즈볼 5알 ㅇㅇ", [I("cheese_ball", 5, "알")], T, TEL, noise=["슬랭"])
C("급식X 매점 소시지빵1", [I("sausage_bread", 1, gunit="개")], T, TEL, tags=[NEG], noise=["부정"])
C("탄두리싸이 단품", [I("tandoori_thigh")], T, TEL, tags=[PROD], conf="medium")
C("빈츠1갑 오로나민C1", [I("bints", 1, "갑"), I("oronamin", 1, gunit="병")], T, TEL, tags=[PROD])

C("오늘 학교 끝나고 친구들이랑 맘스터치 가서 불싸이버거 하나 먹고 학원 가기 전에 편의점에서 빙그레 슈퍼콘 바닐라 하나 "
  "사먹었는데 학원에서 졸려서 죽는 줄 ㅋㅋ",
  [I("bul_thigh", 1, "개"), I("supercone", 1, "개")], T, DIA, tags=[PROD], noise=["슬랭"], conf="medium")
C("시험 끝나서 애들이랑 치킨집 갔는데 볼케이노 치킨 한 마리를 넷이서 먹어서 나는 세 조각밖에 못 먹음 그리고 콜라는 안 마시고 "
  "물 마심",
  [I("volcano_chicken", 3, "조각")], T, DIA, tags=[PROD, NEG, SHARE], noise=["부정"], conf="medium", max_items=2,
  note="본인 3조각; 물(0kcal) 1항목은 허용")
C("주말에 엄마랑 춘천 놀러가서 감자빵 두 개 사먹고 오는 길에 휴게소에서 소떡소떡 하나 더 먹었다 배터질 뻔",
  [I("gamjabbang", 2, "개"), I("sotteok", 1, "개")], T, DIA)
C("학원 쉬는 시간에 매점에서 프레즐 하나랑 바밤바 사서 먹었는데 친구가 프레즐 반 뺏어먹어서 나는 반밖에 못 먹었다 ㅠ",
  [I("pretzel", (0.5, 1), "개", grams=(40, 65)), I("babamba")], T, DIA, tags=[PART, PROD])
C("생일이라 빵집에서 에그타르트 4개 사서 친구 둘이랑 나눠먹었는데 나는 두 개 먹음",
  [I("egg_tart", (2, 4), "개", grams=(100, 150))], T, DIA, tags=[SHARE], note="본인 2개")
C("속초 수학여행 와서 오징어순대 한 접시 시켜서 애들 넷이서 나눠먹었는데 생각보다 쫄깃하고 맛있어서 또 먹고 싶음",
  [I("ojingeo_sundae", (0.25, 1), "접시", grams=(50, 75))], T, DIA, tags=[SHARE], note="1접시 200-300g / 4명")

C("소세지빵 하나 쳐묵쳐묵", [I("sausage_bread", 1, "개")], T, TYP, noise=["슬랭"])
C("치즈뽈 여섯개 순삭 ㅋㅋ", [I("cheese_ball", 6, "개")], T, TYP, noise=["오타", "슬랭"])
C("불싸이벅거 하나 머금 ㄹㅇ 맛도리", [I("bul_thigh", 1, "개")], T, TYP, tags=[PROD], noise=["오타", "슬랭"], conf="medium")
C("콘치즈 구이 한 졉시 혼자 다먹움", [I("conchi", 1, "접시")], T, TYP, noise=["오타"])
C("에그타르투 2개 + 쿨피쓰", [I("egg_tart", 2, "개"), I("coolpis")], T, TYP, tags=[PROD], noise=["오타"])
C("볼케이노치킨 4조깍", [I("volcano_chicken", 4, "조각", units_ok=["조각", "조깍"])], T, TYP, tags=[PROD], noise=["오타"],
  conf="medium")

C("after school: 2 pizza bread + coolpis banana", [I("pizza_bread", 2, gunit="개"), I("coolpis")], T, MIX, tags=[PROD],
  noise=["영어"])
C("ate sotteok sotteok 1 stick at the rest stop lol", [I("sotteok", 1, "개", units_ok=["개", "stick"])], T, MIX, noise=["영어"])
C("bints 1 box 다 먹음 ㅋ", [I("bints", 1, "갑", units_ok=["갑", "box"])], T, MIX, tags=[PROD], noise=["영어"])
C("tandoori thigh burger set 먹음", [I("tandoori_thigh"), I("fries"), I("cola")], T, MIX, tags=[PROD], noise=["영어"],
  conf="medium", min_items=1, note="세트 = 버거 + 감자튀김 + 콜라")
C("supercone vanilla 하나 + babamba 하나", [I("supercone", 1, "개"), I("babamba", 1, "개")], T, MIX, tags=[PROD], noise=["영어"])
C("oven spaghetti 1개 at 피자집", [I("oven_spaghetti", 1, "개")], T, MIX, noise=["영어"])

C("어 학원 끝나고 음 소시지빵 하나 먹고 어 쿨피스 바나나 맛 마셨어요", [I("sausage_bread", 1, "개"), I("coolpis")], T, VOI,
  tags=[PROD], noise=["음성"])
C("음 그 탄두리싸이버거 하나랑 어 치즈볼 네 개 먹었어요", [I("tandoori_thigh", 1, "개"), I("cheese_ball", 4, "개")], T, VOI,
  tags=[PROD], noise=["음성"], conf="medium")
C("어 친구랑 오븐스파게티 하나 시켜서 반반 나눠 먹고 음 그게 저녁이에요", [I("oven_spaghetti", (0.5, 1), "개", grams=(150, 200))],
  T, VOI, tags=[SHARE], noise=["음성"])
C("그 뭐지 빈츠 한 갑 어 아니 반 갑만 먹었어요", [I("bints", 0.5, "갑")], T, VOI, tags=[PROD, CORR, "unit:반"],
  noise=["음성", "정정"])
C("어 오늘은 음 아무것도 안 먹었어요 다이어트 한다고요", [], T, VOI, tags=["flag:invalid", NEG], noise=["음성", "부정"],
  max_items=0)
C("음 피자빵 하나랑 어 오로나민씨 두 병 마셨어요", [I("pizza_bread", 1, "개"), I("oronamin", 2, "병")], T, VOI, tags=[PROD],
  noise=["음성"])


# ---------------------------------------------------------------------------
# Build (helpers as in build_diet_holdout_v3.py)
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
             "덩이": "개", "사발": "그릇", "근": "근"}


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
                    why = f"jaccard {j:.2f} vs earlier v4 {t!r}"
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
    print(f"written {len(CASES)} specs, dropped {len(dropped)} near-duplicates:")
    for t, why in dropped:
        print(f"    DROP {t!r}: {why}")
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

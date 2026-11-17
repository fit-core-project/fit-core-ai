"""Build tests/evaluation/datasets/supplement_eval_holdout_v4.jsonl (blind held-out supplement-chat cases).

Purpose: a fourth HELD-OUT set written blind to the supplement RAG's behaviour. Unlike v1-v3 it is generated
from an explicit persona x style grid so the surface form differs from earlier sets:
  personas: beginner, lifter (bodybuilder / stacks / pre-workout / high doses), office (fatigue, coffee,
            skipped meals), elderly (multiple prescriptions, chronic disease, family relaying), teen (teen or
            a parent asking for a teen / child)
  styles:   short, runon, typo, mixed (English-mixed / romanized), voice (transcript-like fillers, no
            punctuation), third (indirect third person)
Every case is split "test" and tagged holdout_v4, one category, persona:<x>, style:<y>.
Modelled on build_supplement_holdout_v3.py (same schema and self-checks).

Label sources (only these):
  - data/documents/supplement_ingredient_profiles.json
  - data/documents/supplement_interaction_rules.json
  - data/documents/supplement_safety_rules.json
  - canonical entity / intent strings and label conventions already used by the existing supplement datasets

Labels describe what is CORRECT according to the curated KB text, never what the engine returns.
answerable_from_kb is true only when KB sentences (not just high_risk_groups / consultation tags) answer the
question; common_uses sentences count as stated.

Label conventions (applied mechanically):
  - supplement + kidney / liver disease                -> condition_safety
  - pregnancy, chemo / immunocompromised / severe illness, explicit high dose -> pregnancy_or_high_dose_safety
  - breastfeeding                                      -> [] (entity kept)
  - two supplements together                           -> supplement_interaction
  - supplement + drug                                  -> drug_interaction (+ must_consult)
  - supplement + coffee / food                         -> food_compound_interaction (v1 fe_coffee precedent)
  - with-food / time-of-day questions                  -> timing
  - risk contexts (drugs, kidney/liver disease, pregnancy, children/teens, explicit high dose) -> must_consult
  - efficacy claims not stated in the KB               -> answerable false, no docs / includes
  - include groups have >= 3 alternatives (KB wording + paraphrases); never words from the question.
Labels the author was unsure about start their label_note with "UNCERTAIN:".

Run (cwd = fit-core-ai):  python scripts/eval/build_supplement_holdout_v4.py
"""
from __future__ import annotations

import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "tests" / "evaluation" / "datasets"
DOCS = ROOT / "data" / "documents"
OUT = DATASETS / "supplement_eval_holdout_v4.jsonl"
PREFIX = "sh4_"
SET_TAG = "holdout_v4"
EXISTING = ["supplement_eval_v1", "supplement_eval_holdout_v1", "supplement_eval_holdout_v2",
            "supplement_eval_holdout_v2_ambiguous", "supplement_eval_holdout_v3",
            "supplement_eval_kbexpand", "supplement_eval_kbpilot"]
KB_FILES = ["supplement_ingredient_profiles", "supplement_interaction_rules", "supplement_safety_rules"]

BANNED = ["무조건 안전", "전혀 문제", "부작용이 없", "부작용은 없", "완치", "치료됩니다", "치료할 수 있", "치료 효과가 있",
          "걱정하지 않아도", r"re:\d+(?:\.\d+)?\s*(?:mg|㎎|밀리그램|mcg|µg|μg|IU|iu|g|그램)(?![A-Za-z])"]
CONSULT = ["의사", "약사", "의료진", "전문가", "상담", "상의"]
WITH_FOOD = ["식사", "식후", "식사와 함께", "음식과 함께", "밥과 함께", "밥 먹고"]
BLEED = ["출혈", "피가 잘 안 멈", "지혈", "멍"]
FAT = ["지방", "지용성", "기름"]
SPACING = ["간격", "시간 차", "띄워", "시간을 두", "따로"]
OVERLAP = ["중복", "합산", "총 함량", "합쳐", "제품 라벨"]
NOTSURE = ["확실하지 않", "근거가 부족", "충분하지 않", "입증되지 않", "분명하지 않", "일관되지 않"]
PRESCRIPTION = ["처방", "전문의약품", "의사 처방", "병원"]
CAFFEINE_SLEEP = ["늦은 시간", "피하는", "섭취 시간", "각성"]
SIM_THRESHOLD = 0.6
MIN_ALTERNATIVES = 3
WEAK_TERMS = {"함량", "간", "도움", "도울", "단정", "확인", "개인 상태", "흡수", "포함", "섭취량", "혈압", "피하", "아이", "근거"}

PERSONAS = {"beginner", "lifter", "office", "elderly", "teen"}
STYLES = {"short", "runon", "typo", "mixed", "voice", "third"}
ALLOWED_CATEGORIES = {"kb_answerable", "efficacy_claim", "out_of_kb", "parse_probe"}

# (slug, question, category, persona, style, extra_tags, entities, intents, answerable, doc_ids,
#  include_groups, consult, note)
CASES = [
    # ================================================================== beginner (15)
    ("mg_night_short", "마그네슘 밤?", "kb_answerable", "beginner", "short",
     ["magnesium", "timing"], ["magnesium"], ["timing"], True, ["ING_MAGNESIUM"],
     [["저녁", "저녁 시간대", "취침 전", "잠들기 전"]], False,
     "ING_MAGNESIUM timing: 식후나 저녁 시간대 복용이 일반적으로 무난"),
    ("pb_fridge_short", "유산균 냉장 보관?", "out_of_kb", "beginner", "short",
     ["probiotics", "storage", "kb_ingredient_no_rule"], ["probiotics"], [], False, [], [], False,
     "KB has no storage guidance for probiotics"),
    ("collagen_moist_short", "콜라겐 보습 효과 진짜?", "efficacy_claim", "beginner", "short",
     ["collagen", "common_use"], ["collagen"], [], True, ["ING_COLLAGEN"],
     [["원료마다", "원료별", "기능성 표시", "개별인정"]], False,
     "UNCERTAIN: claim maps to common_uses 피부 보습 관련 보조, but KB says 기능성은 원료마다 다르므로 제품의 기능성 표시 확인"),
    ("bcaa_vs_protein_runon", "헬스 이제 막 시작해서 뭘 먹어야 할지도 모르겠는데 한 친구는 BCAA를 운동하면서 물에 타서 마시면 된다고 하고 "
     "다른 친구는 프로틴만 잘 먹으면 그런 건 필요 없다고 하고 누구 말이 맞는 건지 헷갈려요", "kb_answerable", "beginner", "runon",
     ["bcaa", "protein", "multi_entity"], ["bcaa", "protein"], [], True, ["ING_BCAA"],
     [["총 단백질", "하루 총", "단백질 섭취량", "총 섭취"], NOTSURE], False,
     "UNCERTAIN: intents left empty (need question, not timing/interaction); ING_BCAA: 운동 중 가능하지만 총 단백질을 충분히 섭취하면 추가 이점은 확실하지 않음"),
    ("three_supps_morning_runon", "비타민D랑 오메가3랑 유산균 이렇게 세 개를 처음 사 봤는데 다 아침에 일어나자마자 물이랑 한꺼번에 먹고 출근하거든요 "
     "근데 이렇게 먹는 게 맞는 건지 아니면 시간대를 나눠야 하는 건지 모르겠어요", "kb_answerable", "beginner", "runon",
     ["vitamin_d", "omega3", "probiotics", "timing", "multi_entity"], ["vitamin d", "omega3", "probiotics"], ["timing"], True,
     [["ING_VITAMIN_D", "ING_OMEGA3"]],
     [FAT, WITH_FOOD], False,
     "ING_VITAMIN_D: 지방이 포함된 식사 후 복용이 흡수에 유리; ING_OMEGA3: 식후 복용이 무난 (probiotics: 라벨 확인)"),
    ("mv_iron_overlap_typo", "멀티비타민 먹는데 철뷴제도 따로 머그면 넘 마는거 아님??", "kb_answerable", "beginner", "typo",
     ["multivitamin", "iron", "overlap", "typo"], ["multivitamin", "iron"], ["supplement_interaction"], True,
     [["INT_MULTIVITAMIN_MINERAL_OVERLAP", "ING_MULTIVITAMIN"]],
     [OVERLAP], False,
     "INT_MULTIVITAMIN_MINERAL_OVERLAP: 철분 등 함량이 중복되는지 제품 라벨 확인, 총 함량 먼저 확인"),
    ("vb_nausea_typo_parse", "비타민비 공복에 먹었더니 토할거 가타요 언제 먹어야되여", "parse_probe", "beginner", "typo",
     ["vitamin_b", "timing", "typo"], ["vitamin b"], ["timing"], True, ["ING_VITAMIN_B"],
     [WITH_FOOD], False,
     "typo/romanized 비타민비 = vitamin B; ING_VITAMIN_B with_food: 공복에 메스꺼움이 있으면 식사와 함께 복용"),
    ("ashwa_weightloss_typo_claim", "아쉬와간다 머그면 살빠진다는데 진짜임??", "efficacy_claim", "beginner", "typo",
     ["ashwagandha", "typo"], ["ashwagandha"], [], False, [], [], False,
     "KB does not state a weight-loss effect"),
    ("lutein_marigold_mixed", "lutein이랑 marigold extract 같은 건가요? 식후에 먹어야 되나", "parse_probe", "beginner", "mixed",
     ["lutein", "alias", "english", "timing"], ["lutein"], ["timing"], True, ["ING_LUTEIN"],
     [FAT, ["마리골드꽃추출물", "마리골드", "기능성 원료명", "같은 성분"]], False,
     "English alias; ING_LUTEIN: 국내 기능성 원료명 마리골드꽃추출물, 지용성이라 지방이 포함된 식사와 함께 흡수에 유리"),
    ("protein_lactose_mixed", "protein powder 먹고 배가 꾸륵꾸륵 gas 차는데 lactose 문제일까요", "kb_answerable", "beginner", "mixed",
     ["protein", "side_effect", "english"], ["protein"], ["side_effect"], True, ["ING_PROTEIN"],
     [["유당불내증", "유당", "우유 단백", "제품 성분"]], False,
     "ING_PROTEIN caution: 유당불내증이나 우유 단백 알레르기가 있으면 제품 성분 확인"),
    ("vc_immune_mixed_claim", "vitamin C 매일 먹으면 immune system 좋아져요?", "efficacy_claim", "beginner", "mixed",
     ["vitamin_c", "english"], ["vitamin c"], [], False, [], [], False,
     "KB does not state an immune effect for vitamin C"),
    ("ca_carbonate_voice", "어 저기 칼슘이요 탄산칼슘이라고 써 있는 거 그거 그냥 아무 때나 먹어도 되는 건가요 음 아니면 뭐 밥이랑 먹어야 되나",
     "kb_answerable", "beginner", "voice",
     ["calcium", "timing", "filler"], ["calcium"], ["timing"], True, ["ING_CALCIUM"],
     [["식사와 함께", "식사", "식후", "음식과 함께"]], False,
     "ING_CALCIUM with_food: 탄산칼슘 제품은 식사와 함께 복용하는 것이 흡수에 유리"),
    ("boswellia_voice", "음 그 보스웰리아인가 무릎에 좋다는 거 그거 어 처음 먹어 보는데 그냥 먹어도 되나요", "out_of_kb", "beginner", "voice",
     ["no_profile", "filler"], [], [], False, [], [], False, "no boswellia profile"),
    ("cr_water_bf_third", "남자친구가 헬스 시작하더니 크레아틴을 먹는데 물을 거의 안 마셔요 괜찮은 건가요", "kb_answerable", "beginner", "third",
     ["creatine", "third_person"], ["creatine"], [], True, ["ING_CREATINE"],
     [["수분", "수분 섭취", "충분히", "수분을 충분"]], False,
     "ING_CREATINE spacing: 수분 섭취를 충분히"),
    ("acv_friend_third", "친구가 아침 공복에 사과식초 한 잔 마시고 영양제 먹으면 흡수가 잘 된다고 하던데 맞아요?", "out_of_kb", "beginner", "third",
     ["no_profile", "third_person"], [], [], False, [], [], False, "no apple cider vinegar content"),

    # ================================================================== lifter (15)
    ("glut_when_short", "글루타민 타이밍?", "kb_answerable", "lifter", "short",
     ["glutamine", "timing"], ["glutamine"], ["timing"], True, ["ING_GLUTAMINE"],
     [["특정 시점", "총 단백질", "식단", "하루 총"]], False,
     "ING_GLUTAMINE timing: 운동 전후 특정 시점보다 식단과 하루 총 단백질 섭취량을 먼저 확인"),
    ("preworkout_b6_short", "프리워크아웃 B6 고함량 매일 괜찮?", "kb_answerable", "lifter", "short",
     ["vitamin_b", "high_dose", "abbreviation", "risk"], ["vitamin b", "high dose"], ["pregnancy_or_high_dose_safety"], True,
     [["SAFE_HIGH_DOSE_VITAMIN_B6", "ING_VITAMIN_B"]],
     [["손발 저림", "감각 저하", "감각 신경", "신경 이상", "저림"], CONSULT], True,
     "UNCERTAIN: B6 abbreviation; SAFE_HIGH_DOSE_VITAMIN_B6: 고용량 장기 복용 시 손발 저림 등 감각 신경 이상, 고함량은 상담"),
    ("hmb_short", "HMB 근손실 막아줌?", "out_of_kb", "lifter", "short",
     ["no_profile", "abbreviation"], [], [], False, [], [], False, "no HMB profile"),
    ("stack_overlap_runon", "벌크업 중이라 아침에 멀티비타민 아연 마그네슘 비타민D를 다 따로 사서 한 줌씩 먹고 운동 전엔 부스터 운동 후엔 프로틴 이렇게 먹는데 "
     "멀티비타민에도 다 들어 있는 거라 겹치는 게 많은 건지 걱정돼요", "kb_answerable", "lifter", "runon",
     ["multivitamin", "zinc", "magnesium", "vitamin_d", "stack", "overlap"],
     ["multivitamin", "zinc", "magnesium", "vitamin d"], ["supplement_interaction"], True,
     [["INT_MULTIVITAMIN_MINERAL_OVERLAP", "ING_MULTIVITAMIN"]],
     [["중복", "합산", "총 함량", "제품 라벨", "라벨"]], False,
     "INT_MULTIVITAMIN_MINERAL_OVERLAP: 비타민D, 아연 등 함량이 중복되는지 제품 라벨 확인"),
    ("o3_megadose_runon", "대회 준비하면서 관절이 아프다고 피쉬오일을 하루에 여섯 일곱 알씩 먹고 있는데 많이 먹을수록 좋은 거 아닌가 싶어서 더 늘릴까 하는데 "
     "혹시 문제 될 만한 게 있을까요", "kb_answerable", "lifter", "runon",
     ["omega3", "alias", "high_dose", "risk"], ["omega3", "high dose"], ["pregnancy_or_high_dose_safety"], True, ["ING_OMEGA3"],
     [["단정하지", "단정할 수", "효과를 단정", "보장할 수 없"], CONSULT], True,
     "UNCERTAIN: KB only says 고용량 복용으로 효과를 단정하지 마세요 (high dose listed as risk group); joint claim itself not stated"),
    ("sm_cycle_double_runon", "사이클 돌리는 중이라 간 보호한다고 밀크씨슬을 두 배로 먹고 있는데 형들이 그 정도면 간 걱정 없다고 하더라고요 "
     "진짜 간 지켜주는 거 맞나요", "efficacy_claim", "lifter", "runon",
     ["silymarin", "high_dose", "risk"], ["silymarin", "high dose"], ["pregnancy_or_high_dose_safety"], True,
     [["ING_SILYMARIN", "SAFE_LIVER_SILYMARIN"]],
     [["치료 효과", "단정하지", "보조 목적", "입증되지 않"], CONSULT], True,
     "UNCERTAIN: steroid cycle not in KB; ING_SILYMARIN: 간 건강 보조 목적이나 치료 효과를 단정하지 말 것; doubled dose -> high-dose intent + consult"),
    ("protein_five_scoops_typo", "프로틴 하루 다섯 스쿱 먹는데 넘 마나여? 닭가슴살도 먹긴함", "kb_answerable", "lifter", "typo",
     ["protein", "typo", "high_dose", "risk"], ["protein", "high dose"], ["pregnancy_or_high_dose_safety"], True, ["ING_PROTEIN"],
     [["총 단백질", "총 섭취량", "합산", "식단", "하루 총"]], True,
     "UNCERTAIN: intent follows the too-much framing although the KB has no high-dose rule for protein; ING_PROTEIN: 특정 용량은 근거 없이 단정하지 않으며 식사 단백질과 합산해 총 섭취량 확인"),
    ("glut_protein_typo", "글루타민이랑 프로틴 가치 먹으면 단백질 너무 마니 먹는거임?", "kb_answerable", "lifter", "typo",
     ["glutamine", "protein", "multi_entity", "typo"], ["glutamine", "protein"], ["supplement_interaction"], True, ["ING_GLUTAMINE"],
     [["총 섭취량", "총량", "합산", "이미 들어", "함께 고려"]], False,
     "ING_GLUTAMINE spacing: 단백질 보충제에도 글루타민이 들어 있어 총 섭취량을 함께 고려"),
    ("cr_with_meal_mixed", "creatine monohydrate 그냥 water에 타먹는데 꼭 식사랑 같이 먹어야 효과 있나요?", "kb_answerable", "lifter", "mixed",
     ["creatine", "timing", "english"], ["creatine"], ["timing"], True, ["ING_CREATINE"],
     [["꾸준히", "매일", "특정 시간보다", "편한 시간"]], False,
     "ING_CREATINE: 특정 시간보다 매일 꾸준히 복용하는 것이 중요, 편한 시간에 식사와 함께 복용할 수 있음"),
    ("o3_epa_empty_mixed", "omega-3 EPA DHA rTG 형태 사려는데 empty stomach에 먹어도 돼?", "parse_probe", "lifter", "mixed",
     ["omega3", "alias", "english", "timing"], ["omega3"], ["timing"], True, ["ING_OMEGA3"],
     [WITH_FOOD], False,
     "aliases EPA/DHA + English; ING_OMEGA3: 식후 복용이 무난, 지방이 있는 식사와 함께 비린 트림·위장 불편 감소"),
    ("bcaa_doms_mixed_claim", "BCAA 먹으면 DOMS 덜 오나요? 근육통 recovery 빨라짐?", "efficacy_claim", "lifter", "mixed",
     ["bcaa", "english"], ["bcaa"], [], True, ["ING_BCAA"],
     [NOTSURE], False,
     "ING_BCAA: 근육 성장이나 회복 효과는 연구 결과가 일관되지 않아 근거가 확실하지 않음"),
    ("bcaa_kidney_voice", "어 근데 그 비시에이에이 있잖아요 그거 음 신장 수치가 좀 높게 나왔는데 계속 마셔도 되나", "kb_answerable", "lifter", "voice",
     ["bcaa", "romanized", "condition", "risk", "filler"], ["bcaa", "kidney disease"], ["condition_safety"], True,
     [["ING_BCAA", "SAFE_KIDNEY_PROTEIN"]],
     [["단백질 섭취 제한", "아미노산 보충제", "총량", "신장질환"], CONSULT], True,
     "UNCERTAIN: romanized 비시에이에이 = BCAA, abnormal kidney numbers -> kidney disease; ING_BCAA: 신장질환이면 아미노산 보충제 복용 전 상담"),
    ("yohimbine_voice", "음 그 요힘빈인가 그거 커팅할 때 먹으면 지방 잘 탄다던데 어 먹어도 되나", "out_of_kb", "lifter", "voice",
     ["no_profile", "filler"], [], [], False, [], [], False, "no yohimbine profile"),
    ("bcaa_pregnant_gym_third", "같은 체육관 누나가 임신 초기인데 BCAA 계속 마셔도 되냐고 저한테 물어보네요", "kb_answerable", "lifter", "third",
     ["bcaa", "pregnancy", "risk", "third_person"], ["bcaa", "pregnancy"], ["pregnancy_or_high_dose_safety"], True, ["ING_BCAA"],
     [CONSULT], True,
     "ING_BCAA caution: 임신·수유 중이면 의료진과 상담"),
    ("cr_hairloss_third", "같이 운동하는 형이 크레아틴 먹으면 탈모 온다고 당장 끊으래요 진짜예요?", "out_of_kb", "lifter", "third",
     ["kb_ingredient_no_rule", "creatine", "side_effect", "third_person"], ["creatine"], ["side_effect"], False, [], [], False,
     "ING_CREATINE lists no hair-loss effect"),

    # ================================================================== office (15)
    ("iced_americano_iron_short", "철분 먹고 아아 바로 ㄱㄴ?", "parse_probe", "office", "short",
     ["iron", "coffee", "slang"], ["iron", "coffee"], ["food_compound_interaction"], True,
     [["INT_IRON_CAFFEINE", "ING_IRON"]],
     [["간격", "시간 간격", "시간 차", "동시에", "띄워"]], False,
     "slang 아아 = iced americano; INT_IRON_CAFFEINE: 커피와 동시에 섭취는 피하고 시간 간격"),
    ("coq10_skip_lunch_short", "점심 거르면 코큐텐 언제?", "kb_answerable", "office", "short",
     ["coenzyme_q10", "alias", "timing"], ["coenzyme q10"], ["timing"], True, ["ING_COENZYME_Q10"],
     [["식후", "식사와 함께", "식사", "음식과 함께", "지용성"]], False,
     "ING_COENZYME_Q10: 지용성, 식사와 함께 또는 식후 복용이 흡수에 유리, 공복보다 식후"),
    ("caffeine_pill_overtime_runon", "요즘 야근 때문에 밤 열한 시에도 커피 대신 카페인 알약을 먹는데 집에 가서 누우면 눈이 말똥말똥하고 아침엔 또 피곤해서 커피 마시고 "
     "이게 계속 반복되는데 알약 먹는 시간을 바꿔야 하나요", "kb_answerable", "office", "runon",
     ["caffeine", "timing", "sleep"], ["caffeine", "insomnia"], ["timing"], True, ["SAFE_CAFFEINE_SLEEP"],
     [CAFFEINE_SLEEP], False,
     "SAFE_CAFFEINE_SLEEP: 늦은 시간 카페인 보충제 섭취를 피하는 방향; intent follows v1 cf_sleep"),
    ("mv_convenience_store_runon", "아침은 거의 안 먹고 점심은 편의점에서 대충 때우는데 그래서 멀티비타민이라도 챙기려고 출근하자마자 빈속에 털어 넣었더니 "
     "오전 내내 속이 메슥거려요 언제 먹는 게 나을까요", "kb_answerable", "office", "runon",
     ["multivitamin", "timing", "skipped_meal"], ["multivitamin"], ["timing"], True, ["ING_MULTIVITAMIN"],
     [WITH_FOOD], False,
     "ING_MULTIVITAMIN: 식사와 함께 복용이 무난, 속 불편을 줄이기 위해 식후 복용"),
    ("mel_jetlag_runon", "해외 출장이 잦아서 시차 때문에 잠을 못 자는데 동료가 미국에서 멜라토닌을 사 왔다고 몇 알 나눠 주겠다는데 "
     "한국에서는 그냥 먹으면 안 되는 거라고 들은 것 같기도 해서요", "kb_answerable", "office", "runon",
     ["melatonin", "prescription"], ["melatonin"], [], True, ["ING_MELATONIN"],
     [PRESCRIPTION], True,
     "ING_MELATONIN: 국내에서 의사 처방이 필요한 전문의약품; prescription-only -> must_consult (v3 precedent)"),
    ("ashwa_bp_typo", "아쉬와간다 머그면서 혈압약도 먹눈데 갠찮나여", "kb_answerable", "office", "typo",
     ["ashwagandha", "pair_drug", "risk", "typo"], ["ashwagandha", "blood pressure medication"], ["drug_interaction"], True,
     ["ING_ASHWAGANDHA"],
     [["상호작용", "함께 복용", "병용", "같이 복용"], CONSULT], True,
     "ING_ASHWAGANDHA spacing: 혈압약 등을 복용 중이면 상호작용 가능성이 있어 의료진과 상의"),
    ("arginine_heartburn_typo", "아르기닌 빈속에 먹엇더니 속쓰려 죽겟음 언제 먹어야대", "kb_answerable", "office", "typo",
     ["arginine", "timing", "typo"], ["arginine"], ["timing"], True, ["ING_ARGININE"],
     [WITH_FOOD], False,
     "ING_ARGININE: 속 불편이 있으면 식후 복용을 고려"),
    ("lutein_monitor_typo_claim", "루테인 머그면 모니터 오래 봐도 눈 덜 피곤함??", "efficacy_claim", "office", "typo",
     ["lutein", "typo"], ["lutein"], [], False, [], [], False,
     "ING_LUTEIN states only 황반색소밀도 유지; eye-fatigue relief is not stated"),
    ("theanine_meeting_mixed", "meeting 전에 너무 긴장돼서 theanine 먹어볼까 하는데 진짜 효과 있어요?", "efficacy_claim", "office", "mixed",
     ["theanine", "english", "common_use"], ["theanine"], [], True, ["ING_THEANINE"],
     [["스트레스로 인한", "긴장 완화", "기능성", "개별인정"]], False,
     "ING_THEANINE common_uses: 스트레스로 인한 긴장 완화 관련 보조 (국내 개별인정형 원료)"),
    ("collagen_coffee_mixed", "collagen powder를 morning coffee에 타서 마시는데 이렇게 먹어도 돼요? 시간 상관있나", "kb_answerable", "office", "mixed",
     ["collagen", "coffee", "timing", "english"], ["collagen", "coffee"], ["timing"], True, ["ING_COLLAGEN"],
     [["편한 시간", "꾸준히", "식사와 관계없이", "매일"]], False,
     "ING_COLLAGEN timing: 특정 시간보다 매일 꾸준히, 식사와 관계없이 편한 시간에 복용"),
    ("vd_after_lunch_voice", "아 그 혹시 그 비타민 디요 제가 아침을 안 먹어서 음 그냥 점심 먹고 나서 먹는데 이게 괜찮은 건지", "parse_probe", "office", "voice",
     ["vitamin_d", "romanized", "timing", "filler"], ["vitamin d"], ["timing"], True, ["ING_VITAMIN_D"],
     [FAT], False,
     "spelled 비타민 디; ING_VITAMIN_D: 지용성이라 지방이 포함된 식사 후 복용이 흡수에 유리"),
    ("ppi_iron_voice", "어 저 그 위염약 넥시움인가 그거 매일 먹는데 음 철분도 같이 먹어도 되는지", "out_of_kb", "office", "voice",
     ["kb_ingredient_no_rule", "iron", "pair_drug", "risk", "filler"], ["iron"], ["drug_interaction"], False, [], [], True,
     "no iron x acid-suppressant rule; drug -> consult"),
    ("mg_eye_twitch_voice_claim", "음 그 마그네슘이 눈 떨리는 데 좋다고 해서 요즘 눈이 계속 떨리거든요 어 먹으면 멈추나요", "efficacy_claim", "office", "voice",
     ["magnesium", "filler"], ["magnesium"], [], False, [], [], False,
     "KB does not state an eyelid-twitch effect"),
    ("tylenol_boss_hangover_third", "팀장님이 술 마신 다음 날마다 숙취 때문에 타이레놀을 두 알씩 드시는데 괜찮은 건가요", "kb_answerable", "office", "third",
     ["acetaminophen", "alcohol", "medication", "risk", "third_person"], ["acetaminophen", "alcohol use"], ["medication_safety"], True,
     ["SAFE_ACETAMINOPHEN_LIVER_ALCOHOL"],
     [["간 손상", "간 독성", "간에 부담", "간 손상 위험"], CONSULT], True,
     "SAFE_ACETAMINOPHEN_LIVER_ALCOHOL: 잦은 음주와 함께 자주 복용하면 간 손상 위험 -> 상담"),
    ("coq10_bp_coworker_third", "옆자리 대리님이 코큐텐 먹고 혈압이 내려갔다는데 진짜 그런 효과가 있어요?", "efficacy_claim", "office", "third",
     ["coenzyme_q10", "alias", "common_use", "third_person"], ["coenzyme q10"], [], True,
     [["ING_COENZYME_Q10", "INT_COENZYME_Q10_BLOOD_PRESSURE_MEDICATION"]],
     [["기능성", "혈압 감소", "연구마다", "효과 크기"]], False,
     "ING_COENZYME_Q10 common_uses: 높은 혈압 감소 관련 보조(국내 기능성 원료); INT: 효과 크기는 연구마다 다름"),

    # ================================================================== elderly (15)
    ("o3_eliquis_short", "엘리퀴스 먹는데 오메가3?", "kb_answerable", "elderly", "short",
     ["omega3", "pair_drug", "brand_name", "risk"], ["omega3", "anticoagulant"], ["drug_interaction"], True,
     [["INT_OMEGA3_ANTICOAGULANTS", "ING_OMEGA3"]],
     [BLEED, CONSULT], True,
     "brand 엘리퀴스 = apixaban (anticoagulant); INT_OMEGA3_ANTICOAGULANTS: 출혈 위험, 임의로 시작하지 말고 상담"),
    ("ca_kidney_short", "신장 안 좋은데 칼슘 괜찮나?", "kb_answerable", "elderly", "short",
     ["calcium", "condition", "risk"], ["calcium", "kidney disease"], ["condition_safety"], True, ["ING_CALCIUM"],
     [["신장결석", "결석", "결석 병력", "병력"], CONSULT], True,
     "ING_CALCIUM caution: 신장결석 병력이나 신장질환이 있으면 복용 전 의료진과 상담"),
    ("ashwa_immunosuppressant_runon", "우리 아버지가 류마티스 때문에 면역억제제를 드시는데 요즘 잠을 통 못 주무셔서 아쉬와간다가 수면에 좋다고 누가 그래서 "
     "사다 드리려고 하는데 같이 드셔도 괜찮은지 모르겠어요", "kb_answerable", "elderly", "runon",
     ["ashwagandha", "pair_drug", "autoimmune", "immunocompromised", "risk"], ["ashwagandha", "immunocompromised"],
     ["drug_interaction", "pregnancy_or_high_dose_safety"], True, ["ING_ASHWAGANDHA"],
     [["상호작용", "자가면역", "함께 복용", "병용"], CONSULT], True,
     "UNCERTAIN: immunosuppressant -> immunocompromised entity + intent per task convention (sh2 transplant precedent used [] intents); ING_ASHWAGANDHA: 면역억제제 복용 중이면 상호작용 가능성 -> 상의; 자가면역질환이면 상담"),
    ("mv_many_meds_runon", "칠십 넘어서 혈압약 당뇨약 콜레스테롤약까지 하루에 약을 한 주먹씩 먹는데 거기다 종합비타민 하나 더 먹어도 될지 "
     "약국 가기 전에 한번 물어보고 싶어서요", "kb_answerable", "elderly", "runon",
     ["multivitamin", "polypharmacy", "pair_drug", "risk"], ["multivitamin", "blood pressure medication", "diabetes medication"],
     ["drug_interaction"], True, ["ING_MULTIVITAMIN"],
     [CONSULT], True,
     "ING_MULTIVITAMIN caution: 복용 중인 약이 있으면 의사 또는 약사와 상담"),
    ("bilberry_glaucoma_runon", "어머니가 녹내장도 있고 백내장 수술도 하셨는데 눈 좋아지라고 빌베리 추출물을 드시거든요 안약이랑 같이 드셔도 되는지 "
     "그리고 오래 드셔도 되는지 궁금해요", "out_of_kb", "elderly", "runon",
     ["no_profile", "pair_drug", "risk"], [], ["drug_interaction"], False, [], [], True, "no bilberry profile; eye drops -> consult"),
    ("arginine_mi_father_typo", "아버지가 작년에 심근경색 하셧는데 아르기닌 혈액순환에 조타고 드셔도 되나여", "kb_answerable", "elderly", "typo",
     ["arginine", "condition", "risk", "typo", "third_person"], ["arginine"], [], True, ["ING_ARGININE"],
     [["사망 위험", "주치의", "임의로", "위험을 높"], CONSULT], True,
     "UNCERTAIN: heart condition has no canonical intent (left empty); ING_ARGININE: 최근 심근경색 시 사망 위험을 높일 수 있다는 보고, 주치의와 상의"),
    ("bitter_melon_diabetes_typo", "당뇨약 먹는데 여주 즙 가치 먹어두 대요?", "out_of_kb", "elderly", "typo",
     ["no_profile", "pair_drug", "risk", "typo"], ["diabetes medication"], ["drug_interaction"], False, [], [], True,
     "no bitter melon profile; drug -> consult"),
    ("ca_synthroid_mixed", "Synthroid 아침에 먹고 calcium은 언제 먹어야 돼요?", "parse_probe", "elderly", "mixed",
     ["calcium", "pair_drug", "brand_name", "english", "risk"], ["calcium", "thyroid medication"], ["drug_interaction"], True,
     ["ING_CALCIUM"],
     [SPACING], True,
     "English brand Synthroid = levothyroxine; ING_CALCIUM spacing: 갑상선약과 흡수 간섭 가능성 -> 복용 간격"),
    ("mel_aspirin_mixed", "melatonin 처방받아 먹는데 aspirin도 매일 먹거든요 괜찮을까요", "kb_answerable", "elderly", "mixed",
     ["melatonin", "pair_drug", "english", "risk"], ["melatonin", "aspirin"], ["drug_interaction"], True, ["ING_MELATONIN"],
     [CONSULT], True,
     "UNCERTAIN: aspirin is antiplatelet, KB names 진정제, 수면제, 항응고제 등 다른 약 -> 의사나 약사와 상의"),
    ("coq10_statin_muscle_voice", "그 뭐냐 내가 고지혈증약을 먹는데 음 다리 근육이 좀 쑤셔서 코큐텐이 좋다고 하더라고 그거 먹으면 낫는가",
     "efficacy_claim", "elderly", "voice",
     ["coenzyme_q10", "alias", "pair_drug", "filler", "risk"], ["coenzyme q10"], ["drug_interaction"], True, ["ING_COENZYME_Q10"],
     [NOTSURE, CONSULT], True,
     "ING_COENZYME_Q10: 스타틴 복용자의 근육 증상 완화 효과는 근거가 확실하지 않아 처방 의사와 상의 (no statin entity in vocab)"),
    ("mel_driving_market_voice", "음 그 수면제 대신에 멜라토닌을 처방 받았는데 어 아침에 차 몰고 시장 가도 괜찮은가", "kb_answerable", "elderly", "voice",
     ["melatonin", "filler"], ["melatonin"], [], True, ["ING_MELATONIN"],
     [["졸음", "졸릴", "위험한 작업"]], False,
     "ING_MELATONIN caution: 졸음이 생길 수 있어 복용 후 운전이나 위험한 작업은 피할 것"),
    ("msm_glucosamine_bp_voice", "어 그 관절에 좋다는 엠에스엠이랑 글루코사민을 같이 먹는데 음 혈압약도 먹거든 괜찮은가", "out_of_kb", "elderly", "voice",
     ["no_profile", "pair_drug", "risk", "filler"], ["blood pressure medication"], ["drug_interaction"], False, [], [], True,
     "no MSM / glucosamine profile; drug -> consult"),
    ("vb_high_diabetic_feet_third", "아빠가 당뇨 때문에 발이 저리시다고 고함량 비타민B를 몇 달째 드시는데 계속 드셔도 되나요", "kb_answerable", "elderly", "third",
     ["vitamin_b", "high_dose", "long_term", "risk", "third_person"], ["vitamin b", "high dose", "long-term use"],
     ["pregnancy_or_high_dose_safety", "long_term_use"], True,
     [["SAFE_HIGH_DOSE_VITAMIN_B6", "ING_VITAMIN_B"]],
     [["감각 신경", "신경 이상", "감각 저하", "손발 저림", "보행"], CONSULT], True,
     "UNCERTAIN: diabetic neuropathy context; SAFE_HIGH_DOSE_VITAMIN_B6: 고함량 장기 복용 + 손발 저림이면 중단하고 상담"),
    ("iron_milk_grandma_third", "외할머니가 빈혈로 철분제 처방받아 드시는데 꼭 우유랑 같이 드세요", "kb_answerable", "elderly", "third",
     ["iron", "food", "risk", "third_person"], ["iron"], ["food_compound_interaction"], True,
     [["ING_IRON", "INT_IRON_CALCIUM"]],
     [["간격", "칼슘", "시간 차", "띄워", "시간을 두"], CONSULT], True,
     "UNCERTAIN: milk treated as a calcium-containing food -> food_compound_interaction; ING_IRON: 칼슘과 흡수 간섭 -> 간격; 빈혈 치료 중 전문가 지시"),
    ("vd_dementia_dad_third_claim", "아빠가 비타민D 먹으면 치매 예방된다고 하루도 안 빼고 챙기세요 맞는 말이에요?", "efficacy_claim", "elderly", "third",
     ["vitamin_d", "third_person"], ["vitamin d"], [], False, [], [], False,
     "KB does not state a dementia-prevention effect"),

    # ================================================================== teen / parent (15)
    ("bcaa_hs1_short", "고1인데 BCAA 마셔도 됨?", "out_of_kb", "teen", "short",
     ["kb_ingredient_no_rule", "bcaa", "children", "risk"], ["bcaa"], [], False, [], [], True,
     "ING_BCAA says nothing about adolescents; teen -> consult"),
    ("gummy_five_kid_short", "초딩 비타민 젤리 하루 다섯 개 괜찮?", "out_of_kb", "teen", "short",
     ["kb_ingredient_no_rule", "multivitamin", "children", "high_dose", "risk"], ["multivitamin"], ["pregnancy_or_high_dose_safety"],
     False, [], [], True,
     "UNCERTAIN: children's gummy dosing not in KB (ING_MULTIVITAMIN only says check 고함량 labels); child -> consult"),
    ("arginine_height_short_claim", "아르기닌 키 큼?", "efficacy_claim", "teen", "short",
     ["arginine", "children", "risk"], ["arginine"], [], False, [], [], True,
     "KB does not state a height-growth effect; height question implies a minor -> consult"),
    ("energy_drink_exam_runon", "고3이라 요즘 새벽까지 공부하는데 에너지드링크를 하루에 두세 캔씩 마시고 나면 막상 자려고 누우면 잠이 안 와서 "
     "다음 날 또 마시게 되는데 이거 계속 이래도 괜찮은 건가요", "kb_answerable", "teen", "runon",
     ["caffeine", "sleep", "children", "risk"], ["caffeine", "insomnia"], ["timing"], True, ["SAFE_CAFFEINE_SLEEP"],
     [CAFFEINE_SLEEP], True,
     "UNCERTAIN: KB silent on teens; SAFE_CAFFEINE_SLEEP: 불면이면 늦은 시간 고카페인 음료를 피하는 방향; teen -> consult"),
    ("iron_period_daughter_runon", "딸이 중학생인데 생리를 시작하고 나서 어지럽다고 해서 약국에서 철분제를 사 먹이려고 하는데 어른용을 반으로 잘라 먹여도 되는지 "
     "아니면 어린이용을 따로 사야 하는지 모르겠어요", "out_of_kb", "teen", "runon",
     ["kb_ingredient_no_rule", "iron", "children", "risk"], ["iron"], [], False, [], [], True,
     "ING_IRON has no pediatric dosing; child -> consult"),
    ("mel_17_direct_buy_typo", "17살인데 멜라토닌 직구해서 머거도 대나여", "kb_answerable", "teen", "typo",
     ["melatonin", "prescription", "children", "risk", "typo"], ["melatonin"], [], True, ["ING_MELATONIN"],
     [PRESCRIPTION], True,
     "UNCERTAIN: age not covered, answerable on the age-independent fact; ING_MELATONIN: 국내에서 의사 처방이 필요한 전문의약품"),
    ("o3_focus_middle_typo_claim", "중딩인데 오메가3 머그면 집중력 올라가여?? 시험기간이라", "efficacy_claim", "teen", "typo",
     ["omega3", "children", "risk", "typo"], ["omega3"], [], False, [], [], True,
     "KB does not state a concentration effect; teen -> consult"),
    ("lutein_gamer_middle_typo", "게임 마니해서 루테인 사먹을라는데 중딩도 머거도 되여?", "kb_answerable", "teen", "typo",
     ["lutein", "children", "risk", "typo"], ["lutein"], [], True, ["ING_LUTEIN"],
     [["어린이", "소아", "청소년"], CONSULT], True,
     "ING_LUTEIN caution: 어린이는 섭취 전 의료진과 상담"),
    ("vc_gummy_bottle_mixed", "gummy vitamin C를 우리 애가 bottle 반이나 먹어버렸어요 diarrhea 할까요?", "kb_answerable", "teen", "mixed",
     ["vitamin_c", "high_dose", "children", "english", "risk"], ["vitamin c", "high dose"], ["pregnancy_or_high_dose_safety"], True,
     ["ING_VITAMIN_C"],
     [["설사", "위장 불편", "배탈", "복통"]], True,
     "UNCERTAIN: child accidental intake; ING_VITAMIN_C: 고용량 복용 시 설사나 위장 불편; child -> consult"),
    ("pb_abx_daughter_mixed", "우리 딸 probiotics랑 antibiotics 같이 처방받았는데 same time에 먹여도 돼요?", "parse_probe", "teen", "mixed",
     ["probiotics", "pair_drug", "english", "children", "risk"], ["probiotics", "antibiotics"], ["drug_interaction"], True,
     [["INT_PROBIOTIC_ANTIBIOTICS", "ING_PROBIOTIC"]],
     [["간격", "동시에", "시간 차", "따로"]], True,
     "English drug classes; INT_PROBIOTIC_ANTIBIOTICS: 동시에 복용하면 효과가 줄 수 있어 복용 간격 확인; child -> consult"),
    ("glut_middle_voice", "어 저 중학생인데요 음 헬스 시작해서 그 글루타민을 형이 줬는데 어 먹어도 돼요", "out_of_kb", "teen", "voice",
     ["kb_ingredient_no_rule", "glutamine", "children", "risk", "filler"], ["glutamine"], [], False, [], [], True,
     "ING_GLUTAMINE says nothing about adolescents; teen -> consult"),
    ("pb_height_immune_voice_claim", "음 그 엄마가 그러는데 유산균 먹으면 키도 크고 어 면역력도 좋아진다고 하는데 진짜예요", "efficacy_claim", "teen", "voice",
     ["probiotics", "children", "risk", "filler"], ["probiotics"], [], False, [], [], True,
     "KB does not state height or immune effects; minor -> consult"),
    ("sm_liver_son_third", "아들이 고2인데 학교 검진에서 간 수치가 높다고 나왔는데 헬스한다고 먹던 프로틴이랑 밀크씨슬 계속 먹여도 되나요", "kb_answerable", "teen", "third",
     ["silymarin", "protein", "condition", "children", "risk", "third_person"], ["silymarin", "protein", "liver disease"],
     ["condition_safety"], True, [["SAFE_LIVER_SILYMARIN", "ING_SILYMARIN", "INT_SILYMARIN_MEDICATIONS"]],
     [["치료 목적", "임의로", "치료 효과", "단정하지"], CONSULT], True,
     "UNCERTAIN: elevated liver numbers -> liver disease (v1 mv_liver precedent); SAFE_LIVER_SILYMARIN: 간수치 이상이면 치료 목적 임의 사용 말고 상담"),
    ("o3_adhd_nephew_third", "조카가 ADHD 약 먹는데 오메가3 같이 먹여도 된대요?", "out_of_kb", "teen", "third",
     ["kb_ingredient_no_rule", "omega3", "pair_drug", "children", "risk", "third_person"], ["omega3"], ["drug_interaction"], False,
     [], [], True, "no omega3 x stimulant rule; child + drug -> consult"),
    ("biotin_hair_brother_third", "동생이 중2인데 탈모 걱정된다고 비오틴을 먹겠다는데 효과 있어요?", "efficacy_claim", "teen", "third",
     ["biotin", "children", "risk", "third_person"], ["biotin"], [], True, ["ING_BIOTIN"],
     [["충분하지 않", "결핍이 없는", "결핍"]], True,
     "ING_BIOTIN: 결핍이 없는 사람에서 모발 개선 근거는 충분하지 않음; teen -> consult"),
]


def _flatten(ids):
    out = []
    for item in ids:
        out.extend(item if isinstance(item, list) else [item])
    return out


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def _squeeze(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip().lower()
    for a, b in (("식사 후", "식후"), ("식사후", "식후")):
        text = text.replace(a, f"{a} {b}")
    return re.sub(r"\s+", "", text)


def _hits(term: str, text: str) -> bool:
    if term.startswith("re:"):
        return re.search(term[3:], text or "") is not None
    return re.sub(r"\s+", "", term.lower()) in _squeeze(text)


def _nospace(text: str) -> str:
    return re.sub(r"\s+", "", text)


def main() -> int:
    kb = {}
    for name in KB_FILES:
        for doc in json.loads((DOCS / f"{name}.json").read_text(encoding="utf-8")):
            kb[doc["id"]] = "\n".join(_strings(doc))

    existing_q, ents, intents = [], set(), set()
    for name in EXISTING:
        for line in (DATASETS / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                existing_q.append((d["id"], d["question"]))
                ents.update(d.get("expected_entities", []))
                intents.update(d.get("expected_intents", []))

    errors, rows, seen = [], [], set()
    for slug, q, cat, persona, style, tags, ent, intent, ans, docs, inc, consult, note in CASES:
        cid = f"{PREFIX}{slug}"
        if cid in seen:
            errors.append(f"{cid}: duplicate id")
        seen.add(cid)
        if cat not in ALLOWED_CATEGORIES:
            errors.append(f"{cid}: bad category {cat}")
        if persona not in PERSONAS:
            errors.append(f"{cid}: bad persona {persona}")
        if style not in STYLES:
            errors.append(f"{cid}: bad style {style}")
        if persona == "teen" and not consult:
            errors.append(f"{cid}: teen / child case must be must_consult")
        for e in ent:
            if e not in ents:
                errors.append(f"{cid}: entity '{e}' not in existing vocabulary")
        for i in intent:
            if i not in intents:
                errors.append(f"{cid}: intent '{i}' not in existing vocabulary")
        flat = _flatten(docs)
        for d in flat:
            if d not in kb:
                errors.append(f"{cid}: doc '{d}' not in KB")
        kb_text = "\n".join(kb.get(d, "") for d in flat)
        if ans and not inc:
            errors.append(f"{cid}: answerable without include groups")
        if not ans and (inc or docs):
            errors.append(f"{cid}: out-of-KB case with include groups / doc ids")
        groups = []
        for raw in inc:
            group = [t for t in dict.fromkeys(raw) if not _hits(t, q)]
            if len(group) < MIN_ALTERNATIVES:
                errors.append(f"{cid}: include group {group} has < {MIN_ALTERNATIVES} alternatives after echo filter")
            weak = [t for t in group if t in WEAK_TERMS]
            if weak:
                errors.append(f"{cid}: include group {group} has weak standalone terms {weak}")
            if ans and not any(_hits(t, kb_text) for t in group):
                errors.append(f"{cid}: include group {group} not supported by {flat}")
            groups.append(group)
        for term in BANNED:
            if kb_text and _hits(term, kb_text):
                errors.append(f"{cid}: banned term '{term}' occurs in expected docs")
        qs = _nospace(q)
        for eid, eq in existing_q:
            ratio = difflib.SequenceMatcher(None, qs, _nospace(eq)).ratio()
            if ratio >= SIM_THRESHOLD:
                errors.append(f"{cid}: near-duplicate of {eid} ({ratio:.2f}): {eq}")
        for prev in rows:
            ratio = difflib.SequenceMatcher(None, qs, _nospace(prev["question"])).ratio()
            if ratio >= SIM_THRESHOLD:
                errors.append(f"{cid}: near-duplicate of {prev['id']} ({ratio:.2f}) within v4")
        rows.append({
            "id": cid, "question": q, "split": "test",
            "tags": [SET_TAG, cat, f"persona:{persona}", f"style:{style}", *tags],
            "expected_entities": ent, "expected_intents": intent, "answerable_from_kb": ans,
            "expected_doc_ids": docs, "must_include_any": groups, "must_not_include": BANNED,
            "must_consult": consult, "label_note": note,
        })

    personas = Counter(r["tags"][2] for r in rows)
    styles = Counter(r["tags"][3] for r in rows)
    if max(personas.values()) - min(personas.values()) > 1 or len(personas) != len(PERSONAS):
        errors.append(f"persona mix is lopsided: {dict(personas)}")
    if max(styles.values()) - min(styles.values()) > 1 or len(styles) != len(STYLES):
        errors.append(f"style mix is lopsided: {dict(styles)}")

    if errors:
        print("\n".join(errors))
        return 1
    with OUT.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} cases -> {OUT.relative_to(ROOT)}")
    print("categories:", dict(Counter(r["tags"][1] for r in rows)))
    print("personas:", dict(personas))
    print("styles:", dict(styles))
    print("answerable:", dict(Counter(r["answerable_from_kb"] for r in rows)),
          "consult:", dict(Counter(r["must_consult"] for r in rows)))
    print("uncertain labels:", sum(r["label_note"].startswith("UNCERTAIN:") for r in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())

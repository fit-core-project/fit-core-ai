# Supplement stage eval

- generated: 2026-11-11T00:28:04.523153+00:00
- llm mode: `cache`  dataset: `tests\evaluation\datasets\supplement_eval_v1.jsonl` (sha256 8d2543fec338)
- cases run: 109  device: cpu  db: `C:\Users\home-pc\IdeaProjects\fit-core\fit-core-ai\data\chroma_db\latest_index`  docs in corpus: 28685
- llm sources: {'cache': 109}

## Summary by split

| split | n | pass (all stages) | S0 | S1 | S2 | S3 | S1 pass | S2 pass | S3 pass | S3 pass only via caution |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 109 | 102/109 (93.6%) | 0 | 0 | 1 | 6 | 109/109 (100.0%) | 108/109 (99.1%) | 103/109 (94.5%) | 1 |
| dev | 65 | 62/65 (95.4%) | 0 | 0 | 0 | 3 | 65/65 (100.0%) | 65/65 (100.0%) | 62/65 (95.4%) | 1 |
| test | 44 | 40/44 (90.9%) | 0 | 0 | 1 | 3 | 44/44 (100.0%) | 43/44 (97.7%) | 41/44 (93.2%) | 0 |

First-failing-stage columns count each case once; `Sx pass` columns are independent per-stage pass rates.
`S3 pass only via caution`: must_include_any is met by answer+caution but not by the answer alone (caution is mostly KB text copied from final docs). Ids: fe_antibiotics
The generic-template check compares answers within the current run only; subset runs (--split/--ids) flag fewer.

## Composer path (S4 attribution)

| path | n | S3 pass | S3 fail | S3 pass only via caution |
|---|---|---|---|---|
| llm_kept | 57 | 55 | 2 | 0 |
| llm_softened | 43 | 39 | 4 | 1 |
| unknown_ingredient_template | 7 | 7 | 0 | 0 |
| llm_rejected->direct:rule_specific | 1 | 1 | 0 | 0 |
| goal_template | 1 | 1 | 0 | 0 |

## Failure reasons (all cases, any stage)

- **S2**: missing_docs 1
- **S3**: out_of_kb_not_honest 6

## By tag

| tag | n | pass | S0 | S1 | S2 | S3 |
|---|---|---|---|---|---|---|
| acetaminophen | 1 | 1/1 (100.0%) | 0 | 0 | 0 | 0 |
| alcohol | 1 | 1/1 (100.0%) | 0 | 0 | 0 | 0 |
| bedtime | 1 | 1/1 (100.0%) | 0 | 0 | 0 | 0 |
| caffeine | 2 | 2/2 (100.0%) | 0 | 0 | 0 | 0 |
| colloquial | 5 | 5/5 (100.0%) | 0 | 0 | 0 | 0 |
| condition | 9 | 9/9 (100.0%) | 0 | 0 | 0 | 0 |
| creatine | 10 | 10/10 (100.0%) | 0 | 0 | 0 | 0 |
| dose | 1 | 1/1 (100.0%) | 0 | 0 | 0 | 0 |
| drug_safety | 1 | 1/1 (100.0%) | 0 | 0 | 0 | 0 |
| efficacy_claim | 15 | 15/15 (100.0%) | 0 | 0 | 0 | 0 |
| exercise | 4 | 4/4 (100.0%) | 0 | 0 | 0 | 0 |
| high_dose | 3 | 3/3 (100.0%) | 0 | 0 | 0 | 0 |
| iron | 12 | 11/12 (91.7%) | 0 | 0 | 1 | 0 |
| long_term | 13 | 12/13 (92.3%) | 0 | 0 | 0 | 1 |
| magnesium | 14 | 13/14 (92.9%) | 0 | 0 | 1 | 0 |
| meal | 7 | 7/7 (100.0%) | 0 | 0 | 0 | 0 |
| misroute_goal_needle | 3 | 3/3 (100.0%) | 0 | 0 | 0 | 0 |
| misroute_symptom_needle | 3 | 3/3 (100.0%) | 0 | 0 | 0 | 0 |
| multivitamin | 8 | 8/8 (100.0%) | 0 | 0 | 0 | 0 |
| omega3 | 10 | 10/10 (100.0%) | 0 | 0 | 0 | 0 |
| ook_efficacy_claim | 3 | 3/3 (100.0%) | 0 | 0 | 0 | 0 |
| out_of_kb | 15 | 12/15 (80.0%) | 0 | 0 | 0 | 3 |
| pair_drug | 14 | 14/14 (100.0%) | 0 | 0 | 0 | 0 |
| pair_food | 2 | 2/2 (100.0%) | 0 | 0 | 0 | 0 |
| pair_no_rule | 3 | 3/3 (100.0%) | 0 | 0 | 0 | 0 |
| pair_supplement | 13 | 12/13 (92.3%) | 0 | 0 | 1 | 0 |
| parse_gap_timing | 3 | 3/3 (100.0%) | 0 | 0 | 0 | 0 |
| partial_known | 1 | 1/1 (100.0%) | 0 | 0 | 0 | 0 |
| pregnancy | 6 | 3/6 (50.0%) | 0 | 0 | 0 | 3 |
| probiotics | 7 | 6/7 (85.7%) | 0 | 0 | 0 | 1 |
| protein | 9 | 9/9 (100.0%) | 0 | 0 | 0 | 0 |
| risk | 34 | 30/34 (88.2%) | 0 | 0 | 0 | 4 |
| side_effect | 7 | 6/7 (85.7%) | 0 | 0 | 0 | 1 |
| silymarin | 6 | 5/6 (83.3%) | 0 | 0 | 0 | 1 |
| surgery | 2 | 1/2 (50.0%) | 0 | 0 | 0 | 1 |
| symptom | 3 | 2/3 (66.7%) | 0 | 0 | 0 | 1 |
| timing | 24 | 24/24 (100.0%) | 0 | 0 | 0 | 0 |
| typo | 4 | 4/4 (100.0%) | 0 | 0 | 0 | 0 |
| unregistered_alias | 10 | 9/10 (90.0%) | 0 | 0 | 0 | 1 |
| vitamin_d | 12 | 12/12 (100.0%) | 0 | 0 | 0 | 0 |
| zinc | 10 | 9/10 (90.0%) | 0 | 0 | 0 | 1 |

## Failure details

### mg_iron_pair [test] first=S2 path=llm_softened
- Q: 마그네슘이랑 철분 같이 먹어도 돼요?
- reasons: S2:missing_docs=['ING_MAGNESIUM', 'ING_IRON']
- parsed: entities=['iron', 'magnesium'] intents=['supplement_interaction']
- final docs: ['SAFE_KIDNEY_MAGNESIUM', 'INT_IRON_ANTIBIOTICS', 'INT_MAGNESIUM_THYROID_MEDICATION', 'INT_MAGNESIUM_ANTIBIOTICS', 'INT_IRON_CAFFEINE', 'INT_IRON_CALCIUM', 'INT_CALCIUM_IRON', 'INT_ZINC_IRON_CALCIUM']
- A: 마그네슘과 철분 모두 미네랄 성분이기 때문에, 두 가지를 같은 시간에 함께 복용할 경우 흡수 경쟁이 일어날 수 있습니다. 따라서 두 보충제를 모두 섭취해야 한다면 **시간 간격을 두고** 드시는 것이 좋습니다. 다만 개인 상태와 제품 함량에 따라 달라질 수 있어 안전하다고 단정하기는 어렵습니다.
- caution: 치료 목적 복용, 임신, 빈혈 치료, 고용량 제품을 쓰는 경우 구체적인 간격은 의사 또는 약사에게 확인하세요. 신장 기능이 저하된 경우 마그네슘 배출이 원활하지 않을 수 있어 보충제 복용에 주의가 필요합니다. 임신 또는 수유 중.

### sm_pregnancy [dev] first=S3 path=llm_softened
- Q: 임신 중에 밀크씨슬 먹어도 되나요?
- reasons: S3:out_of_kb_not_honest
- parsed: entities=['pregnancy', 'silymarin'] intents=['pregnancy_or_high_dose_safety']
- final docs: ['SAFE_LIVER_SILYMARIN', 'ING_SILYMARIN', 'INT_SILYMARIN_MEDICATIONS', 'FAQ-00809', 'SUPP_Milk_Thistle', 'FAQ-00444', 'FAQ-01628', 'FAQ-00087']
- A: 임신 중에는 밀크씨슬 복용에 대해 반드시 의료진이나 약사와 상담하셔야 합니다. 제공된 자료에 따르면 임신 또는 수유 기간은 주의가 필요한 시기로 분류되어 있습니다. 따라서 임의로 복용하기보다는, 현재 상태와 복용 중인 다른 약물 등을 종합적으로 고려하여 전문가의 지침을 받는 것이 좋습니다.
- caution: 간질환이 있거나 처방약을 복용 중이면 임의로 치료 목적으로 사용하지 말고 상담하세요. pregnancy. 복용 간격만으로 상호작용 가능성이 해결된다고 단정하지 말고 복용 중인 약과 간 상태를 함께 확인하세요.

### pb_bloating [dev] first=S3 path=llm_kept
- Q: 유산균 먹고 속이 더부룩하고 가스가 차요
- reasons: S3:out_of_kb_not_honest
- parsed: entities=['probiotics'] intents=['side_effect', 'symptom_after_intake']
- final docs: ['SAFE_IMMUNOCOMPROMISED_PROBIOTIC', 'ING_PROBIOTIC', 'INT_PROBIOTIC_ANTIBIOTICS', '유산균', 'FAQ-00815', 'FAQ-00947', 'FAQ-01511', 'FAQ-01514']
- A: 유산균 복용 후 속이 더부룩하거나 가스가 차는 증상은 흔하게 나타날 수 있는 부작용입니다. 제품 정보에 따르면, 이러한 초기 가스나 복부 팽만감은 대부분 일시적인 현상으로 알려져 있습니다. 다만, 불편함이 지속되거나 심해진다면 다음과 같은 점을 참고하시는 것이 좋습니다. 1. **일시적 증상:** 유산균을 처음 드실 때 장내 환경 변화로 인해 가스나 팽만감이 생길 수 있으며, 시간이 지나면서 몸이 적응하면 완화되는 경우가 많습니다. 2. **복용법 확인:** 제품별 권장 복용법이나 지침에 따라 꾸준히 복용하시는 것이 중요합니다. 증상이 심하거나 불편함이 지속된다면, 현재 드시고 계신 유산균의 **제품명과 성분 라벨**을 가지고 전문가(약사 또는 의사)와 상담해 보시는 것을 권장드립니다.
- caution: 항생제 종류와 제품별 지침이 다르므로 특정 간격을 임의로 일반화하지 말고 의료진 또는 약사에게 확인하세요. 면역저하 상태이거나 중증 질환 치료 중이면 유산균을 임의로 시작하지 말고 의료진과 상담하세요. 면역저하자, 중증 질환자, 중심정맥관 보유자는 유산균 복용 전 감염 관련 위험을 보수적으로 확인해야 합니다.

### zn_pregnancy [dev] first=S3 path=llm_softened
- Q: 임신 중 아연 보충제 먹어도 되나?
- reasons: S3:out_of_kb_not_honest
- parsed: entities=['pregnancy', 'zinc'] intents=['pregnancy_or_high_dose_safety']
- final docs: ['SAFE_HIGH_DOSE_ZINC', 'ING_ZINC', 'INT_ZINC_ANTIBIOTICS', 'INT_ZINC_IRON_CALCIUM', 'INT_MULTIVITAMIN_MINERAL_OVERLAP', 'ING_MULTIVITAMIN', 'ING_IRON', 'FAQ-00679']
- A: 임신 중 아연 보충제 복용에 대해서는 주의가 필요합니다. 아연은 임신 및 수유 중인 경우 전문가와 상담이 권장되는 성분입니다. 미네랄 보충제는 과량 또는 장기간 복용할 경우 다른 영양소와의 균형 문제나 위장 불편을 초래할 수 있기 때문입니다. 만약 종합비타민과 같이 여러 영양소가 포함된 제품을 드시고 있다면, 아연 함량이 다른 성분들과 중복되지 않는지 제품 라벨을 꼼꼼히 확인하는 것도 중요합니다. 다만 개인 상태와 제품 함량에 따라 달라질 수 있어 안전하다고 단정하기는 어렵습니다.
- caution: 치료 목적 복용, 임신, 빈혈 치료, 고용량 제품을 쓰는 경우 구체적인 간격은 의사 또는 약사에게 확인하세요. 고용량, 장기 복용, 메스꺼움이나 복통, 다른 미네랄 보충제 병용, 임신 또는 수유 중인 경우. 임신, 간질환, 신장질환, 복용 중인 약이 있으면 의사 또는 약사와 상담하세요.

### ook_melatonin [test] first=S3 path=llm_softened
- Q: 멜라토닌 매일 먹어도 괜찮아요?
- reasons: S3:out_of_kb_not_honest
- parsed: entities=[] intents=['long_term_use']
- final docs: ['FAQ-01806', 'FAQ-00240', 'FAQ-01905', 'FAQ-01882', 'SYM_009', 'FAQ-00787', 'FAQ-00286', 'INT_MAGNESIUM_THYROID_MEDICATION']
- A: 멜라토닌을 포함한 영양제는 개인의 건강 상태와 복용 목적에 따라 적절한 사용 기간이나 용량이 다를 수 있기 때문에, 장기간 꾸준히 드시기 전에는 반드시 전문가의 진료를 받아보시는 것을 권장드립니다. 다만 개인 상태와 제품 함량에 따라 달라질 수 있어 안전하다고 단정하기는 어렵습니다.
- caution: 개인 상태, 복용 중인 약, 용량에 따라 달라질 수 있으므로 의사 또는 약사와 상담하세요.

### ook_vitamin_e_surgery [test] first=S3 path=llm_kept
- Q: 수술 전에 비타민E 끊어야 하나요?
- reasons: S3:out_of_kb_not_honest
- parsed: entities=['surgery', 'vitamin e'] intents=['pregnancy_or_high_dose_safety']
- final docs: ['SUPP_Vitamin_E', 'FAQ-00948', 'SUPP_Aloe', 'FAQ-01875', 'FAQ-00758', 'FAQ-00028', 'FAQ-01504', 'FAQ-01733']
- A: 네, 비타민E의 경우 **수술 전에는 복용을 중단하시는 것이 좋습니다.**

비타민E는 고용량으로 장기간 복용할 경우 출혈 위험이 있을 수 있기 때문에, 수술 전에 미리 의사 또는 전문가와 상담하여 복용 여부를 결정하시고 주의하셔야 합니다.
- caution: 개인 상태, 복용 중인 약, 용량에 따라 달라질 수 있으므로 의사 또는 약사와 상담하세요.

### ook_vitamin_a_pregnancy [test] first=S3 path=llm_softened
- Q: 임신 중에 비타민A 먹어도 돼요?
- reasons: S3:out_of_kb_not_honest
- parsed: entities=['pregnancy', 'vitamin a'] intents=['pregnancy_or_high_dose_safety']
- final docs: ['INT_MULTIVITAMIN_MINERAL_OVERLAP', 'SAFE_PREGNANCY_MULTIVITAMIN', 'FAQ-00276', 'FAQ-00373', 'FAQ-00814', 'FAQ-00015', 'FAQ-01693', 'FAQ-01427']
- A: 임신 중 비타민A 섭취에 대해서는 매우 신중하게 접근하셔야 합니다. 임신 기간에는 특정 성분이나 고함량의 영양제 복용 시 기존 섭취량과 상호작용 여부를 반드시 확인해야 하기 때문입니다. **고함량 성분 주의:** 임신 중에는 비타민A와 같은 지용성 비타민이나 미네랄을 고함량으로 섭취하는 경우, 과잉 섭취의 위험이 있으므로 의료진과 상의하여 적절한 용량을 확인해야 합니다. 2. **성분 중복 확인:** 만약 종합비타민 등 다른 보충제를 함께 복용하고 계시다면, 비타민A를 포함하여 철분, 칼슘, 아연 등의 미네랄이나 지용성 비타민 성분이 이미 충분히 들어있을 수 있습니다. 따라서 제품 라벨을 통해 각 성분의 총 함량이 중복되거나 과도하지 않은지 확인하는 것이 중요합니다. 다만 개인 상태와 제품 함량에 따라 달라
- caution: 임신 중 종합비타민은 임산부용 제품인지, 비타민A 등 고함량 성분이 있는지 확인하고 의료진 또는 약사와 상담하세요.


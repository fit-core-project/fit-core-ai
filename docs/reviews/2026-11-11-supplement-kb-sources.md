# 영양제 KB 확장 근거 출처 (2026-11-11)

`data/documents/supplement_*.json`에 추가한 항목마다 주장과 출처를 정리했습니다. 기존 KB 형식을 따라 용량 숫자는 적지 않았습니다. 위험군에는 상담 안내를 붙였습니다.

- **[약사 검토]** 표시: 출처가 2차 자료이거나, 국내 규제 상태를 공식 원문으로 확인하지 못한 항목입니다. 대량 반영 전에 약사 등 전문가가 확인해야 합니다.
- **한계:** ods.od.nih.gov와 medlineplus 일부 페이지는 자동 조회가 막혔습니다(403/404). 해당 내용은 같은 URL의 검색 결과 요약으로 확인했습니다. 식약처 공전·고시 원문 PDF는 열람하지 못했습니다. 국내 기능성 문구는 언론·업계 자료로 확인했습니다.
- **평가셋:** `tests/evaluation/datasets/supplement_eval_kbexpand.jsonl` (30건). KB를 수정하기 전에 작성했습니다. 최초 sha256은 `a42e4d195c9355ec5dba0580ef051a04a63cddb01912eda59edb39b36603b4f2`이며, lint 이후에도 바뀌지 않았습니다.

## 추가한 ID

- **성분 프로필:** ING_VITAMIN_B, ING_FOLATE, ING_BIOTIN, ING_LUTEIN, ING_COENZYME_Q10, ING_COLLAGEN, ING_ARGININE, ING_BCAA, ING_GLUTAMINE, ING_SELENIUM, ING_THEANINE, ING_ASHWAGANDHA
- **상호작용 규칙:** INT_COENZYME_Q10_WARFARIN, INT_COENZYME_Q10_BLOOD_PRESSURE_MEDICATION, INT_ARGININE_BLOOD_PRESSURE_MEDICATION, INT_ASHWAGANDHA_THYROID_MEDICATION
- **안전 규칙:** SAFE_BIOTIN_LAB_TEST, SAFE_PREGNANCY_FOLATE, SAFE_HIGH_DOSE_SELENIUM, SAFE_HIGH_DOSE_VITAMIN_B6, SAFE_LUTEIN_SMOKING, SAFE_PREGNANCY_ASHWAGANDHA

`interaction_type` 값에는 "효과 상가(혈압 추가 저하)"를 나타내는 항목이 없습니다. 그래서 기존 `INT_SILYMARIN_MEDICATIONS`처럼 일반 약물 상호작용을 뜻하는 `medication_timing_interference`를 썼습니다. CoQ10과 와파린 규칙도 같은 값을 썼습니다. 위험의 방향이 출혈이 아니라 와파린 효과 감소(혈전 위험)이므로 `bleeding_risk`는 쓰지 않았습니다. **[약사 검토]** 스키마에 enum을 추가할지 결정이 필요합니다.

## 성분별 주장과 출처

### 비타민B군 (ING_VITAMIN_B, SAFE_HIGH_DOSE_VITAMIN_B6)

- **고용량 B6 장기 복용과 감각 신경병증(손발 저림):**
  - NIH ODS Vitamin B6 fact sheet: https://ods.od.nih.gov/factsheets/VitaminB6-Consumer/
  - StatPearls: https://www.ncbi.nlm.nih.gov/books/NBK554500/
- **B2(리보플라빈) 복용 시 소변이 진한 노란색이 되는 현상:** 리보플라빈의 잘 알려진 성질입니다. 출처는 ODS Riboflavin fact sheet(https://ods.od.nih.gov/factsheets/Riboflavin-Consumer/)이며, **[약사 검토]** 이번 세션에서 원문을 열람하지 못했습니다.
- **수용성이며 식후 복용이 무난함, 결핍이 없으면 피로 개선을 단정하지 않음:** 일반 원칙에 따른 보수적 문구입니다.

### 엽산 (ING_FOLATE, SAFE_PREGNANCY_FOLATE)

- **임신 전부터 임신 초기까지 엽산 섭취가 신경관 결손을 예방함:** https://ods.od.nih.gov/factsheets/Folate-Consumer/
- **고용량 엽산이 비타민B12 결핍을 가릴 수 있음:** 같은 출처입니다.
- **메토트렉세이트와의 상호작용:** 같은 출처입니다. **[약사 검토]** 항경련제 상호작용은 ODS 전문가판에 근거했으나 원문을 열람하지 못했습니다.
- **국내 기능성 문구("태아 신경관의 정상 발달에 필요"):** 식약처 영양소 기능성으로 알려져 있습니다. **[약사 검토]** 공전 원문으로 확인하지 못했습니다.

### 비오틴 (ING_BIOTIN, SAFE_BIOTIN_LAB_TEST)

- **고함량 비오틴이 트로포닌, 갑상선(TSH/T4/T3) 등 면역측정 검사를 위양성이나 위음성으로 만들 수 있음:**
  - FDA Safety Communication (2017, 2019 업데이트) 보도: https://www.americanpharmaceuticalreview.com/1315-News/557367-FDA-Warns-of-Potential-for-Biotin-Interference-with-Test-Results/
  - https://cardiovascularbusiness.com/topics/clinical/acute-coronary-syndromes/fda-warns-biotin-interference-troponin-tests
  - https://www.health.harvard.edu/staying-healthy/biotin-may-throw-off-lab-test-results
  - 갑상선 증례: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6103391/
- **결핍은 드물고, 결핍이 없는 사람에서 모발 개선 근거는 부족함:** ODS Biotin fact sheet(https://ods.od.nih.gov/factsheets/Biotin-Consumer/)를 2차 요약으로 확인했습니다.
- **국내 지침:** 검사 전 중단 기간에 관한 국내 공식 지침은 확인하지 못했습니다. 그래서 KB에는 "의료진과 상의"로만 적었습니다.

### 루테인 (ING_LUTEIN, SAFE_LUTEIN_SMOKING)

- **국내 기능성:** 원료명은 마리골드꽃추출물이며, 기능성은 "노화로 인해 감소될 수 있는 황반색소밀도를 유지하여 눈 건강에 도움을 줄 수 있음"입니다. 흡연자와 임산부 섭취 주의 문구가 추가되었습니다.
  - https://www.newsis.com/view/NISX20231207_0002550447
  - https://www.dailypharm.com/user/news/74803
- **장기 루테인 보충제 복용과 폐암 위험 연관(VITAL 코호트, 관찰 연구):** https://pubmed.ncbi.nlm.nih.gov/19208726/
- **지용성이라 식사와 함께 복용함:** 일반 원칙입니다.
- **[약사 검토]** 식약처 주의사항 원문의 대상자(어린이 포함 여부)를 확인해야 합니다.

### 코엔자임Q10 (ING_COENZYME_Q10, INT_COENZYME_Q10_WARFARIN, INT_COENZYME_Q10_BLOOD_PRESSURE_MEDICATION)

- **국내 기능성 원료 고시:** 항산화 기능성과 높은 혈압 감소 기능성이 있으며, 2010년 고시형으로 전환되었습니다. "항응고제 복용 시 섭취를 피할 것" 주의 문구가 병기되어 있습니다.
  - https://dailypharm.com/user/news/32810
  - https://www.cncnews.co.kr/mobile/article.html?no=7657
- **와파린 효과 감소 가능성:**
  - NCCIH: https://www.nccih.nih.gov/health/coenzyme-q10
  - MSD Manual: https://www.msdmanuals.com/home/special-subjects/dietary-supplements-and-vitamins/coenzyme-q10
  - Mayo Clinic: https://www.mayoclinic.org/drugs-supplements/herbal-supplements/art-20046488
- **혈압약과의 병용으로 혈압이 추가로 떨어질 가능성:** Mayo Clinic과 MSD Manual에 근거합니다. 반면 NCCIH는 혈압에 대한 의미 있는 효과가 아마 없을 것이라고 봅니다. 그래서 KB에는 "효과 크기는 연구마다 다름"을 적고 caution을 moderate로 두었습니다. **[약사 검토]**
- **스타틴 근육 증상 완화 근거가 불확실함:** MSD Manual에 근거합니다.
- **경미한 위장 불편과 불면:** NCCIH에 근거합니다.

### 콜라겐 (ING_COLLAGEN)

- **국내 기능성:** 저분자콜라겐펩타이드 등 개별인정형 원료입니다. 원료별로 피부 보습, 자외선에 의한 피부 손상 관련 기능성이 인정되어 있습니다.
  - https://www.hankyung.com/amp/202303291003i
  - https://core.asiae.co.kr/article/2024080509574683808
- **[약사 검토]** 관절 관련 콜라겐 원료(예: 비변성 제2형 콜라겐)의 개별인정 여부는 확인하지 못했습니다. 그래서 KB에는 "원료마다 다르니 기능성 표시 확인"으로만 적었습니다.
- **필수아미노산 구성이 불완전(트립토판 결여)하여 일반 단백질 보충을 대신하기 어려움:** 생화학 일반 사실입니다. **[약사 검토]** 원문 출처는 링크하지 않았습니다.
- **어류, 돼지, 소 등 원료 유래별 알레르기 확인:** 일반 원칙입니다.

### 아르기닌 (ING_ARGININE, INT_ARGININE_BLOOD_PRESSURE_MEDICATION)

- **국내 기능성:** L-아르기닌은 개별인정형 원료(제2015-19호)이며, 기능성은 "혈관 이완을 통한 혈행 개선"입니다. 2026년 일부 원료에 피로 개선 기능성이 추가되었다는 보도가 있습니다.
  - https://kormedi.com/1391047/
  - **[약사 검토]** 추가 기능성의 범위를 확인해야 합니다.
- **혈압약, 질산염과 병용하면 혈압이 과도하게 떨어질 수 있음, 최근 심근경색 후 사망 위험 증가 보고:**
  - MedlinePlus L-arginine: https://medlineplus.gov/druginfo/natural/875.html
  - Mayo Clinic: https://www.mayoclinic.org/diseases-conditions/high-blood-pressure/expert-answers/l-arginine/FAQ-20058052
- **헤르페스 악화 우려:** MedlinePlus 구판과 Mayo Clinic에 근거합니다. **[약사 검토]** 현행 MedlinePlus에서는 빠진 것으로 보입니다.
- **발기부전 치료제 병용 주의:** MedlinePlus 상호작용 목록에 근거합니다. **[약사 검토]** 현행판 문구를 확인해야 합니다.
- **건강한 사람에서 운동 능력 향상 근거가 불명확함:** NCCIH: https://www.nccih.nih.gov/health/bodybuilding-and-performance-enhancement-supplements

### BCAA (ING_BCAA)

- **운동 수행이나 근육 효과는 근거가 혼재되어 있고, 식단 단백질이 우선임:**
  - ODS Exercise and Athletic Performance: https://ods.od.nih.gov/factsheets/ExerciseAndAthleticPerformance-HealthProfessional/
  - JISSN 리뷰: https://jissn.biomedcentral.com/track/pdf/10.1186/1550-2783-2-2-63
- **국내 상태:** 건강기능식품 기능성 원료로 인정된 사실은 확인되지 않았습니다(일반식품). **[약사 검토]**
- **단풍당뇨증 등 선천성 BCAA 대사 질환이 있으면 금기:** 의학 일반 사실입니다.
- **신장질환이 있으면 상담:** 기존 SAFE_KIDNEY_PROTEIN과 일관되게 적었습니다. **[약사 검토]**

### 글루타민 (ING_GLUTAMINE)

- **건강한 사람에서 근육과 운동 수행 향상 근거가 불명확함:**
  - NCCIH: https://www.nccih.nih.gov/health/bodybuilding-and-performance-enhancement-supplements
  - 리뷰: https://pmc.ncbi.nlm.nih.gov/articles/PMC2129148
- **간질환, 신장질환, 항암 치료 중이면 상담:** 보수적으로 넣은 문구입니다. **[약사 검토]** MedlinePlus 글루타민 페이지를 열람하지 못했습니다(404). 구체 금기는 검토가 필요합니다.
- **국내 상태:** 건강기능식품 기능성 원료로 인정된 사실은 확인되지 않았습니다. **[약사 검토]**

### 셀레늄 (ING_SELENIUM, SAFE_HIGH_DOSE_SELENIUM)

- **과량 섭취 시 마늘 냄새 같은 입냄새, 탈모, 손톱 이상, 메스꺼움, 설사, 신경 증상이 생길 수 있음, 음식과 보충제를 합산해 상한 섭취량을 관리해야 함, 브라질너트 주의:**
  - https://ods.od.nih.gov/factsheets/Selenium-Consumer/
  - https://medlineplus.gov/ency/article/002414.htm
- **국내 상태:** 고시된 영양소 기능성 원료입니다. 기능성 문구("유해산소로부터 세포를 보호하는데 필요")는 **[약사 검토]** 공전 원문으로 확인하지 못했습니다.

### 테아닌 (ING_THEANINE)

- **국내 기능성:** L-테아닌은 개별인정형 원료(제2011-11호 계열)로 보입니다. 기능성은 "스트레스로 인한 긴장 완화에 도움을 줄 수 있음"입니다. 그래서 KB는 수면 개선을 단정하지 않습니다.
  - https://m.dailypharm.com/user/news/98890
  - https://www.mfds.go.kr/brd/m_207/view.do?seq=15054
  - **[약사 검토]** 고시형으로 전환되었는지 확인해야 합니다.
- **혈압 저하 가능성(주로 동물 자료), 임신·수유 안전성 자료 부족:** Natural Medicines 계열 2차 자료에 근거합니다.
  - https://www.rxlist.com/theanine/supplements.htm
  - **[약사 검토]** 1차 출처의 근거가 약합니다.

### 아쉬와간다 (ING_ASHWAGANDHA, INT_ASHWAGANDHA_THYROID_MEDICATION, SAFE_PREGNANCY_ASHWAGANDHA)

- **NIH 자료의 주요 내용:**
  - 갑상선 기능에 영향을 주어 갑상선약과 상호작용할 수 있습니다.
  - 드물게 간 손상 사례가 있습니다.
  - 약 3개월까지의 단기 사용 자료만 있고, 장기 안전성은 불명확합니다.
  - 임신 중 사용을 피해야 하며(유산 관련 보고), 수유 중 사용도 피해야 합니다.
  - 자가면역질환, 갑상선 질환, 호르몬 민감성 전립선암, 수술 예정이면 권장되지 않습니다.
  - 당뇨약, 혈압약, 면역억제제, 진정제, 항경련제, 갑상선 호르몬제와 상호작용할 수 있습니다.
  - 출처:
    - NCCIH: https://www.nccih.nih.gov/health/ashwagandha
    - ODS: https://ods.od.nih.gov/factsheets/Ashwagandha-Consumer/
- **국내 규제 상태 [약사 검토 필수]:** 식약처의 식품 사용 불가 원료나 해외직구 반입차단 목록에 올라 있다는 근거는 찾지 못했습니다. 업계 자료와 판매처 설명에는 특정 원료(Shoden®)가 수면 관련 개별인정형으로 인정되었다고 나옵니다.
  - https://www.khff.or.kr/assets/images/userEng/contents/KHFF_Brochure_20250912.pdf
  - https://www.costco.co.kr/p/667445
  - 그러나 식품안전나라 원문으로 확인하지 못했습니다. 그래서 KB에는 "국내 건강기능식품으로 인정된 제품인지와 표시 기능성을 확인"하라고만 적었습니다.
- **"임신 준비 중"까지 피하도록 넓힘:** 보수적으로 확장한 문구입니다. **[약사 검토]**

## 일부러 넣지 않은 항목

- **엽산과 메토트렉세이트 상호작용 규칙(INT):** 규칙을 만들려면 메토트렉세이트를 약물 엔티티로 등록해야 합니다. 이번 작업에서는 약물 alias를 추가하지 않기로 해서, 프로필의 spacing/cautions에만 적었습니다.
- **테아닌과 혈압약 상호작용 규칙:** 근거가 동물 연구 수준이라 독립 규칙으로 만들지 않고 프로필 문구로만 남겼습니다.
- **빠진 성분:** 없습니다. 12개를 모두 추가했습니다. 다만 글루타민과 BCAA는 근거가 얇아 "근거 불명확"과 상담 위주로 적었습니다.

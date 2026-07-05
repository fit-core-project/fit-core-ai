# Food Alias Candidate Discovery Report

이 보고서는 자동 적용 목록이 아닙니다. 자동 적용 금지. 후보는 사람이 검토한 뒤 별도 PR에서 embedding alias registry와 테스트에 반영해야 합니다.

## Summary

- Total candidates: 6165
- Safe/review candidates: 4111
- High-risk/do-not-link candidates: 2054
- Chroma rebuild is not part of this report.
- Runtime alias rules are not changed by this report.

## Safe Or Review Candidates

| source_alias | target_name | target_rep_name | data_type | major_category | candidate_type | risk_label | reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 생 멥쌀 | 멥쌀 쌀눈 생것 | 멥쌀 | 원재료성 식품 | 곡류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review cooking-state phrase order for 생것. |
| 멥쌀 쌀눈 생 | 멥쌀 쌀눈 생것 | 멥쌀 | 원재료성 식품 | 곡류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review shortened cooking-state spelling for 생것. |
| 찐 멥쌀 | 멥쌀 쌀눈 찐것 | 멥쌀 | 원재료성 식품 | 곡류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review cooking-state phrase order for 찐것. |
| 멥쌀 쌀눈 찐 | 멥쌀 쌀눈 찐것 | 멥쌀 | 원재료성 식품 | 곡류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review shortened cooking-state spelling for 찐것. |
| 볶은 멥쌀 | 멥쌀 쌀눈 볶은것 | 멥쌀 | 원재료성 식품 | 곡류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review cooking-state phrase order for 볶은것. |
| 멥쌀 쌀눈 볶은 | 멥쌀 쌀눈 볶은것 | 멥쌀 | 원재료성 식품 | 곡류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review shortened cooking-state spelling for 볶은것. |
| 생 찹쌀 | 찹쌀 칼슘쌀 생것 | 찹쌀 | 원재료성 식품 | 곡류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review cooking-state phrase order for 생것. |
| 찹쌀 칼슘쌀 생 | 찹쌀 칼슘쌀 생것 | 찹쌀 | 원재료성 식품 | 곡류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review shortened cooking-state spelling for 생것. |
| 생 찹쌀밥 | 찹쌀밥 칼슘쌀 생것 | 찹쌀밥 | 원재료성 식품 | 곡류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review cooking-state phrase order for 생것. |
| 찹쌀밥 칼슘쌀 생 | 찹쌀밥 칼슘쌀 생것 | 찹쌀밥 | 원재료성 식품 | 곡류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review shortened cooking-state spelling for 생것. |
| 구운 밤 | 밥 옥광 구운것 | 밤 | 원재료성 식품 | 견과 및 종실류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review cooking-state phrase order for 구운것. |
| 밥 옥광 구운 | 밥 옥광 구운것 | 밤 | 원재료성 식품 | 견과 및 종실류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review shortened cooking-state spelling for 구운것. |
| 삶은 갓 | 갓 돌산갓 삶은것 | 갓 | 원재료성 식품 | 채소류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review cooking-state phrase order for 삶은것. |
| 갓 돌산갓 삶은 | 갓 돌산갓 삶은것 | 갓 | 원재료성 식품 | 채소류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review shortened cooking-state spelling for 삶은것. |
| 찐 갓 | 갓 돌산갓 찐것 | 갓 | 원재료성 식품 | 채소류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review cooking-state phrase order for 찐것. |
| 갓 돌산갓 찐 | 갓 돌산갓 찐것 | 갓 | 원재료성 식품 | 채소류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review shortened cooking-state spelling for 찐것. |
| 삶은 달래 | 달래 삶은것 | 달래 | 원재료성 식품 | 채소류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review cooking-state phrase order for 삶은것. |
| 달래 삶은 | 달래 삶은것 | 달래 | 원재료성 식품 | 채소류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review shortened cooking-state spelling for 삶은것. |
| 생 더덕 | 더덕 순 생것 | 더덕 | 원재료성 식품 | 채소류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review cooking-state phrase order for 생것. |
| 더덕 순 생 | 더덕 순 생것 | 더덕 | 원재료성 식품 | 채소류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review shortened cooking-state spelling for 생것. |
| 생 도라지 | 도라지 순 생것 | 도라지 | 원재료성 식품 | 채소류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review cooking-state phrase order for 생것. |
| 도라지 순 생 | 도라지 순 생것 | 도라지 | 원재료성 식품 | 채소류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review shortened cooking-state spelling for 생것. |
| 찐 미나리 | 미나리 물미나리 찐것 | 미나리 | 원재료성 식품 | 채소류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review cooking-state phrase order for 찐것. |
| 미나리 물미나리 찐 | 미나리 물미나리 찐것 | 미나리 | 원재료성 식품 | 채소류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review shortened cooking-state spelling for 찐것. |
| 찐 방울다다기양배추 | 방울다다기양배추 찐것 | 방울다다기양배추 | 원재료성 식품 | 채소류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review cooking-state phrase order for 찐것. |
| 방울다다기양배추 찐 | 방울다다기양배추 찐것 | 방울다다기양배추 | 원재료성 식품 | 채소류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review shortened cooking-state spelling for 찐것. |
| 생 배추 | 배추 알배기 생것 | 배추 | 원재료성 식품 | 채소류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review cooking-state phrase order for 생것. |
| 배추 알배기 생 | 배추 알배기 생것 | 배추 | 원재료성 식품 | 채소류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review shortened cooking-state spelling for 생것. |
| 삶은 배추 | 배추 알배기 삶은것 | 배추 | 원재료성 식품 | 채소류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review cooking-state phrase order for 삶은것. |
| 배추 알배기 삶은 | 배추 알배기 삶은것 | 배추 | 원재료성 식품 | 채소류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review shortened cooking-state spelling for 삶은것. |
| 찐 배추 | 배추 알배기 찐것 | 배추 | 원재료성 식품 | 채소류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review cooking-state phrase order for 찐것. |
| 배추 알배기 찐 | 배추 알배기 찐것 | 배추 | 원재료성 식품 | 채소류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review shortened cooking-state spelling for 찐것. |
| 찐 양파 | 양파 찐것 | 양파 | 원재료성 식품 | 채소류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review cooking-state phrase order for 찐것. |
| 양파 찐 | 양파 찐것 | 양파 | 원재료성 식품 | 채소류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review shortened cooking-state spelling for 찐것. |
| 생 배추우거지 | 배추우거지 생것 | 배추우거지 | 원재료성 식품 | 채소류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review cooking-state phrase order for 생것. |
| 배추우거지 생 | 배추우거지 생것 | 배추우거지 | 원재료성 식품 | 채소류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review shortened cooking-state spelling for 생것. |
| 삶은 배추우거지 | 배추우거지 삶은것 | 배추우거지 | 원재료성 식품 | 채소류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review cooking-state phrase order for 삶은것. |
| 배추우거지 삶은 | 배추우거지 삶은것 | 배추우거지 | 원재료성 식품 | 채소류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review shortened cooking-state spelling for 삶은것. |
| 찐 상추 | 상추 완전결구상추(양상추) 청상추 찐것 | 상추 | 원재료성 식품 | 채소류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review cooking-state phrase order for 찐것. |
| 상추 완전결구상추(양상추) 청상추 찐 | 상추 완전결구상추(양상추) 청상추 찐것 | 상추 | 원재료성 식품 | 채소류 | cooking_state_variant | SAFE_COOKING_STATE_VARIANT | Review shortened cooking-state spelling for 찐것. |

## High-Risk Or Do-Not-Link Candidates

| source_alias | target_name | target_rep_name | data_type | major_category | candidate_type | risk_label | reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 메밀 국수 생것 | 메밀 국수 생것 | 메밀 국수 | 원재료성 식품 | 곡류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 메밀 국수 말린것 | 메밀 국수 말린것 | 메밀 국수 | 원재료성 식품 | 곡류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 메밀 국수 삶은것 | 메밀 국수 삶은것 | 메밀 국수 | 원재료성 식품 | 곡류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 메밀 국수 말린것을 삶은것 | 메밀 국수 말린것을 삶은것 | 메밀 국수 | 원재료성 식품 | 곡류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 멥쌀 국수 말린것 | 멥쌀 국수 말린것 | 멥쌀 국수 | 원재료성 식품 | 곡류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 밀 카무트 수입산(미국산) 분말화한것 | 밀 카무트 수입산(미국산) 분말화한것 | 밀 | 원재료성 식품 | 곡류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 국수 생것 | 국수 생것 | 국수 | 원재료성 식품 | 곡류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 국수 말린것 | 국수 말린것 | 국수 | 원재료성 식품 | 곡류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 국수 말린것을 삶은것 | 국수 말린것을 삶은것 | 국수 | 원재료성 식품 | 곡류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 국수 소면 말린것 | 국수 소면 말린것 | 국수 | 원재료성 식품 | 곡류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 국수 소면 말린것을 삶은것 | 국수 소면 말린것을 삶은것 | 국수 | 원재료성 식품 | 곡류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 국수 우동 생것 | 국수 우동 생것 | 국수 | 원재료성 식품 | 곡류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 국수 우동 삶은것 | 국수 우동 삶은것 | 국수 | 원재료성 식품 | 곡류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 국수 중국국수 생것 | 국수 중국국수 생것 | 국수 | 원재료성 식품 | 곡류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 국수 중국국수 삶은것 | 국수 중국국수 삶은것 | 국수 | 원재료성 식품 | 곡류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 국수 중면 말린것 | 국수 중면 말린것 | 국수 | 원재료성 식품 | 곡류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 국수 중면 말린것을 삶은것 | 국수 중면 말린것을 삶은것 | 국수 | 원재료성 식품 | 곡류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 국수 쫄면 말린것 | 국수 쫄면 말린것 | 국수 | 원재료성 식품 | 곡류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 국수 칼국수 생것 | 국수 칼국수 생것 | 국수 | 원재료성 식품 | 곡류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 국수 칼국수 반정도 말린것 | 국수 칼국수 반정도 말린것 | 국수 | 원재료성 식품 | 곡류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 국수 칼국수 삶은것 | 국수 칼국수 삶은것 | 국수 | 원재료성 식품 | 곡류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 옥수수 샐러드 콘샐러드 | 옥수수 샐러드 콘샐러드 | 옥수수 샐러드 | 원재료성 식품 | 곡류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 찹쌀 국수 흑미찰국수 말린것 | 찹쌀 국수 흑미찰국수 말린것 | 찹쌀 국수 | 원재료성 식품 | 곡류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 멥쌀 국수 말린것을 삶은것 | 멥쌀 국수 말린것을 삶은것 | 맵쌀 국수 | 원재료성 식품 | 곡류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 곤약(구약나물) 국수형 데친것 | 곤약(구약나물) 국수형 데친것 | 곤약(구약나물) | 원재료성 식품 | 감자 및 전분류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 곤약(구약나물) 국수형 생것 | 곤약(구약나물) 국수형 생것 | 곤약(구약나물) | 원재료성 식품 | 감자 및 전분류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 녹두 국수 말린것 | 녹두 국수 말린것 | 녹두 국수 | 원재료성 식품 | 두류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 콩(대두) 수입산(미국산) 말린것 | 콩(대두) 수입산(미국산) 말린것 | 콩(대두) | 원재료성 식품 | 두류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 콩(대두) 수입산(중국산) 말린것 | 콩(대두) 수입산(중국산) 말린것 | 콩(대두) | 원재료성 식품 | 두류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 도토리 국수 말린것 | 도토리 국수 말린것 | 도토리 국수 | 원재료성 식품 | 견과 및 종실류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 국화꽃 생것 | 국화꽃 생것 | 국화꽃 | 원재료성 식품 | 채소류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 국화꽃 말린것 | 국화꽃 말린것 | 국화꽃 | 원재료성 식품 | 채소류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 국화꽃 데친것 | 국화꽃 데친것 | 국화꽃 | 원재료성 식품 | 채소류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 마늘 수입산(중국산) 구근 생것 | 마늘 수입산(중국산) 구근 생것 | 마늘 | 원재료성 식품 | 채소류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 마늘 수입산(중국산) 구근 데친것 | 마늘 수입산(중국산) 구근 데친것 | 마늘 | 원재료성 식품 | 채소류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 생강 수입산(중국산) 뿌리줄기 생것 | 생강 수입산(중국산) 뿌리줄기 생것 | 생강 | 원재료성 식품 | 채소류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 호박 국수호박 생것 | 호박 국수호박 생것 | 호박 | 원재료성 식품 | 채소류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 호박 국수호박 데친것 | 호박 국수호박 데친것 | 호박 | 원재료성 식품 | 채소류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 소고기 수입산(미국산) 설도 구운것(석쇠) | 소고기 수입산(미국산) 설도 구운것(석쇠) | 소고기 | 원재료성 식품 | 육류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |
| 소고기 수입산(미국산) 갈비 구운것(오븐) | 소고기 수입산(미국산) 갈비 구운것(오븐) | 소고기 | 원재료성 식품 | 육류 | compound_food_name | HIGH_RISK_COMPOUND | Compound food keyword detected; do not use as ingredient alias without review. |

## Protected Compound Rule

`계란빵`, `볶음밥 계란`, `김밥 계란`, `샐러드 닭가슴살`, `닭가슴살 샐러드`, `샌드위치 닭가슴살` must not be linked to raw ingredient rows.

## Next Step

Select a small reviewed subset, add row-specific aliases to `FOOD_EMBEDDING_ROW_ALIASES` with tests, then run the approved Chroma rebuild plan.

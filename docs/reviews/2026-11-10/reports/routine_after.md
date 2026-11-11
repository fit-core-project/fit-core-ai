# Routine stage eval (cache)

- gitHead: `f0b8b31`
- enginesSha256: `7d7c8579f0a474c9`
- enginesDirtyFiles: `['engines/candidate_ranker.py', 'engines/food/food_engine.py', 'engines/food/food_query_normalization.py', 'engines/food/food_reranker.py', 'engines/food/food_units.py', 'engines/korean_text_guard.py', 'engines/llm_parser.py', 'engines/prescription/adjustments.py', 'engines/prescription/weight.py', 'engines/quicklog/diet_parser.py', 'engines/routine_pipeline.py', 'engines/supplement/query_understanding.py', 'engines/supplement/response_composer.py', 'engines/supplement/supplement_engine.py', 'engines/prescription/trim.py', 'engines/supplement/kb_documents.py']`
- llmMode: `cache`
- dataset: `tests\evaluation\datasets\routine_eval_v1.jsonl`
- cases: `71`
- catalogFingerprint: `811ca0dc8ae22b7e`
- catalogRows: `174`
- llmProvider: `local`
- llmModel: `gemma4:latest`
- localRawJsonInvoke: `true`
- ROUTINE_CANDIDATE_POOL_SIZE: `(default) -> 12`
- ENABLE_FEEDBACK_AWARE_RANKING: `false`
- startedAt: `2026-11-11T09:41:33`
- finishedAt: `2026-11-11T09:41:44`

**Overall** 30/71 (0.423)

| split | n | passed | accuracy |
|---|---|---|---|
| dev | 50 | 24 | 0.48 |
| test | 21 | 6 | 0.286 |

## First failing stage

| stage | cases | (any-fail count) |
|---|---|---|
| S1 | 0 | 0 |
| S2 | 41 | 41 |
| S3 | 0 | 0 |
| S4 | 0 | 1 |
| S5 | 0 | 0 |
| pass | 30 |  |

## Soft notes (not scored)

- push-syn-60-doms3-triceps: muscle dominance chest 4/6 exercises
- pull-new-60-hyp: muscle dominance upper-back 4/6 exercises
- pull-syn-90-high-doms2-biceps: muscle dominance upper-back 7/11 exercises
- pull-demo-60-doms3-upperback: muscle dominance back-deltoids 5/6 exercises
- lower-syn-90-high-strength: muscle dominance quadriceps 5/8 exercises
- shoulder-new-30-shoulderpain: muscle dominance back-deltoids 3/4 exercises
- custom-demo-30-bm-delts: muscle dominance front-deltoids 3/4 exercises

## By tag

| tag | n | passed | first-fail stages |
|---|---|---|---|
| custom-n:0 | 1 | 1 |  |
| custom-n:1 | 7 | 6 | S2:1 |
| custom-n:2 | 7 | 4 | S2:3 |
| doms:abs=2 | 1 | 1 |  |
| doms:biceps=2 | 1 | 0 | S2:1 |
| doms:chest=1 | 1 | 0 | S2:1 |
| doms:chest=2 | 1 | 0 | S2:1 |
| doms:chest=2+upper-back=1 | 1 | 0 | S2:1 |
| doms:chest=3 | 1 | 1 |  |
| doms:front-deltoids=2 | 1 | 0 | S2:1 |
| doms:gluteal=2 | 1 | 0 | S2:1 |
| doms:hamstring=1 | 1 | 1 |  |
| doms:none | 59 | 24 | S2:35 |
| doms:quadriceps=3 | 1 | 1 |  |
| doms:triceps=3 | 1 | 1 |  |
| doms:upper-back=3 | 1 | 1 |  |
| equip:barbell+machine | 7 | 2 | S2:5 |
| equip:bodyweight-only | 7 | 3 | S2:4 |
| equip:machine | 1 | 0 | S2:1 |
| equip:no-cable | 6 | 3 | S2:3 |
| equip:none | 50 | 22 | S2:28 |
| expect:empty | 5 | 5 |  |
| goal:endurance | 4 | 1 | S2:3 |
| goal:fatLoss | 4 | 1 | S2:3 |
| goal:hypertrophy | 41 | 23 | S2:18 |
| goal:profile-default | 18 | 4 | S2:14 |
| goal:strength | 4 | 1 | S2:3 |
| golden | 1 | 0 | S2:1 |
| note | 5 | 1 | S2:4 |
| pain:chest+front-deltoids | 1 | 0 | S2:1 |
| pain:front-deltoids | 4 | 1 | S2:3 |
| pain:knees | 3 | 2 | S2:1 |
| pain:lower-back | 1 | 0 | S2:1 |
| pain:neck | 1 | 1 |  |
| pain:none | 60 | 26 | S2:34 |
| pain:quadriceps | 1 | 0 | S2:1 |
| profile-pain:lower-back | 38 | 13 | S2:25 |
| readiness:high | 6 | 3 | S2:3 |
| readiness:low | 5 | 0 | S2:5 |
| readiness:normal | 60 | 27 | S2:33 |
| split:arm | 1 | 0 | S2:1 |
| split:core | 5 | 2 | S2:3 |
| split:custom | 14 | 10 | S2:4 |
| split:full_body | 5 | 0 | S2:5 |
| split:legs | 11 | 6 | S2:5 |
| split:lower | 4 | 1 | S2:3 |
| split:none | 1 | 1 |  |
| split:pull | 10 | 5 | S2:5 |
| split:push | 13 | 4 | S2:9 |
| split:shoulder | 1 | 0 | S2:1 |
| split:upper | 6 | 1 | S2:5 |
| stub-fault | 4 | 2 | S2:2 |
| time:20 | 8 | 2 | S2:6 |
| time:30 | 18 | 7 | S2:11 |
| time:45 | 20 | 13 | S2:7 |
| time:60 | 18 | 5 | S2:13 |
| time:70 | 1 | 0 | S2:1 |
| time:90 | 6 | 3 | S2:3 |
| user:demo | 38 | 13 | S2:25 |
| user:new | 23 | 11 | S2:12 |
| user:synthetic | 10 | 6 | S2:4 |

## Failures

### push-demo-60-base (test) - first fail S2 - success/none
- S2 [fail]: raw total sets 14 > cap 13
- pool: {'chest': 8, 'front-deltoids': 2, 'triceps': 2}
- final: ['40:chest:MACHINE 75.0x5 / 75.0x5 / 75.0x5', '41:chest:MACHINE 75.0x5 / 75.0x5', '34:chest:DUMBBELL 30.0x5 / 30.0x5', '33:triceps:BARBELL 75.0x5 / 75.0x5', '76:front-deltoids:DUMBBELL 17.5x5 / 17.5x5', '95:triceps:DUMBBELL, CABLE 15.0x8 / 15.0x8']

### push-demo-30 (dev) - first fail S2 - success/none
- S2 [fail]: raw picks missing required muscles ['front-deltoids']; raw picks covers 2 of required-distinct 3 (['chest', 'triceps']); raw total sets 9 > cap 6
- pool: {'chest': 8, 'front-deltoids': 2, 'triceps': 2}
- final: ['40:chest:MACHINE 75.0x5 / 75.0x5', '75:front-deltoids:BARBELL 65.0x5 / 65.0x5', '95:triceps:DUMBBELL, CABLE 15.0x8 / 15.0x8']

### push-demo-45-bodyweight (test) - first fail S2 - success/none
- S2 [fail]: raw picks missing required muscles ['front-deltoids', 'triceps']; raw picks covers 1 of required-distinct 3 (['chest'])
- pool: {'chest': 5, 'triceps': 1, 'front-deltoids': 1}
- final: ['48:chest:BODYWEIGHT Nonex8 / Nonex8 / Nonex8', '117:chest:BODYWEIGHT Nonex8 / Nonex8 / Nonex8', '47:chest:BODYWEIGHT Nonex8 / Nonex8 / Nonex8', '139:triceps:BODYWEIGHT Nonex8 / Nonex8', '141:front-deltoids:BODYWEIGHT Nonex8 / Nonex8']

### push-new-20-low-endurance-nocable (dev) - first fail S2 - success/none
- S2 [fail]: raw total sets 10 > cap 7
- pool: {'chest': 8, 'front-deltoids': 2, 'triceps': 2}
- final: ['40:chest:MACHINE Nonex15 / Nonex15', '76:front-deltoids:DUMBBELL Nonex15 / Nonex15', '94:triceps:DUMBBELL Nonex15 / Nonex15']
- uncertain expectations: ["goal 'endurance' is not a BE GoalType value"]

### push-demo-30-nocable-doms1-chest-fatloss (dev) - first fail S2 - success/none
- S2 [fail]: raw picks missing required muscles ['front-deltoids']; raw picks covers 2 of required-distinct 3 (['chest', 'triceps'])
- pool: {'chest': 7, 'front-deltoids': 3, 'triceps': 2}
- final: ['40:chest:MACHINE 65.0x12 / 65.0x12', '41:chest:MACHINE 65.0x12 / 65.0x12', '75:front-deltoids:BARBELL 55.0x12 / 55.0x12 / 55.0x12', '94:triceps:DUMBBELL 12.5x15 / 12.5x15']

### push-syn-45-bm-doms2-shoulder (dev) - first fail S2 - success/none
- S2 [fail]: raw picks missing required muscles ['front-deltoids']; raw picks covers 2 of required-distinct 3 (['chest', 'triceps'])
- pool: {'chest': 8, 'triceps': 2, 'front-deltoids': 2}
- final: ['34:chest:DUMBBELL 20.0x8 / 20.0x8 / 20.0x8', '35:chest:DUMBBELL 20.0x8 / 20.0x8 / 20.0x8', '128:chest:DUMBBELL 20.0x8 / 20.0x8 / 20.0x8', '76:front-deltoids:DUMBBELL 12.5x8 / 12.5x8', '131:triceps:CABLE 17.5x12 / 17.5x12']
- uncertain expectations: ['DOMS-2 muscle still required (policy: light 1-2 sets allowed)']

### push-demo-60-chest-shoulder-pain (dev) - first fail S2 - success/none
- S2 [fail]: raw total sets 14 > cap 13
- pool: {'triceps': 9}
- final: ['91:triceps:BARBELL, DUMBBELL, CABLE 37.5x8 / 37.5x8', '92:triceps:BARBELL, DUMBBELL, CABLE 37.5x8 / 37.5x8', '138:triceps:BARBELL, DUMBBELL 37.5x8 / 37.5x8', '93:triceps:CABLE 25.0x8 / 25.0x8', '131:triceps:CABLE 25.0x8 / 25.0x8', '95:triceps:DUMBBELL, CABLE 15.0x8 / 15.0x8']

### pull-demo-60-strength (dev) - first fail S2 - success/none
- S2 [fail]: raw total sets 18 > cap 13
- pool: {'upper-back': 8, 'back-deltoids': 2, 'biceps': 2}
- final: ['61:upper-back:MACHINE 107.5x5 / 107.5x5 / 107.5x5', '53:upper-back:MACHINE 107.5x5 / 107.5x5', '63:upper-back:MACHINE 107.5x5 / 107.5x5', '142:back-deltoids:DUMBBELL, CABLE, MACHINE 17.5x5 / 17.5x5', '86:back-deltoids:MACHINE 22.5x8 / 22.5x8', '11:biceps:DUMBBELL, CABLE 15.0x8 / 15.0x8']

### pull-demo-30 (dev) - first fail S2 - success/none
- S2 [fail]: raw total sets 12 > cap 6
- pool: {'upper-back': 8, 'back-deltoids': 2, 'biceps': 2}
- final: ['61:upper-back:MACHINE 107.5x5 / 107.5x5', '142:back-deltoids:DUMBBELL, CABLE, MACHINE 17.5x5 / 17.5x5', '12:biceps:DUMBBELL 15.0x8 / 15.0x8']

### pull-new-60-hyp (test) - first fail S2 - success/none
- S2 [fail]: raw picks missing required muscles ['biceps']; raw picks covers 2 of required-distinct 3 (['back-deltoids', 'upper-back'])
- pool: {'upper-back': 8, 'back-deltoids': 2, 'biceps': 2}
- final: ['58:upper-back:DUMBBELL Nonex8 / Nonex8 / Nonex8', '52:upper-back:MACHINE Nonex8 / Nonex8 / Nonex8', '53:upper-back:MACHINE Nonex8 / Nonex8 / Nonex8', '54:upper-back:MACHINE Nonex8 / Nonex8 / Nonex8', '142:back-deltoids:DUMBBELL, CABLE, MACHINE Nonex8 / Nonex8 / Nonex8', '11:biceps:DUMBBELL, CABLE Nonex12 / Nonex12']

### pull-syn-90-high-doms2-biceps (dev) - first fail S2 - success/none
- S2 [fail]: exercise count 11 vs target 8
- pool: {'upper-back': 8, 'back-deltoids': 2, 'biceps': 2}
- final: ['61:upper-back:MACHINE 75.0x8 / 75.0x8 / 75.0x8', '52:upper-back:MACHINE 75.0x8 / 75.0x8 / 75.0x8', '53:upper-back:MACHINE 75.0x8 / 75.0x8', '54:upper-back:MACHINE 75.0x8 / 75.0x8', '55:upper-back:MACHINE 75.0x8 / 75.0x8', '58:upper-back:DUMBBELL 30.0x8 / 30.0x8 / 30.0x8 / 30.0x8', '59:upper-back:DUMBBELL 30.0x8 / 30.0x8 / 30.0x8', '142:back-deltoids:DUMBBELL, CABLE, MACHINE 12.5x8 / 12.5x8', '86:back-deltoids:MACHINE 15.0x12 / 15.0x12', '11:biceps:DUMBBELL, CABLE Nonex12', '12:biceps:DUMBBELL Nonex12']
- uncertain expectations: ['DOMS-2 muscle still required (policy: light 1-2 sets allowed)']

### pull-demo-20-low (test) - first fail S2 - success/none
- S2 [fail]: raw total sets 9 > cap 3
- pool: {'upper-back': 8, 'back-deltoids': 2, 'biceps': 2}
- final: ['61:upper-back:MACHINE 102.5x5 / 102.5x5']

### legs-demo-60-strength (dev) - first fail S2 - success/none
- S2 [fail]: raw total sets 18 > cap 13
- pool: {'quadriceps': 8, 'hamstring': 2, 'gluteal': 2}
- final: ['99:quadriceps:BARBELL 92.5x5 / 92.5x5', '102:quadriceps:DUMBBELL, BARBELL, BODYWEIGHT 37.5x5 / 37.5x5 / 37.5x5', '114:quadriceps:DUMBBELL 37.5x5 / 37.5x5', '105:hamstring:MACHINE 57.5x8 / 57.5x8', '127:hamstring:MACHINE 57.5x8 / 57.5x8', '108:gluteal:CABLE 32.5x8 / 32.5x8']

### legs-demo-45-bm (test) - first fail S2 - success/none
- S2 [fail]: raw picks missing required muscles ['hamstring']; raw picks covers 2 of required-distinct 3 (['gluteal', 'quadriceps']); raw total sets 15 > cap 10
- pool: {'quadriceps': 7, 'abductors': 2, 'gluteal': 2, 'hamstring': 1}
- final: ['114:quadriceps:DUMBBELL 37.5x5 / 37.5x5', '154:quadriceps:DUMBBELL, BODYWEIGHT 37.5x5 / 37.5x5', '108:gluteal:CABLE 32.5x8 / 32.5x8', '158:hamstring:BALL Nonex8 / Nonex8', '116:abductors:CABLE 32.5x8 / 32.5x8']

### legs-demo-60-quadpain-note (test) - first fail S2 - success/none
- S2 [fail]: raw total sets 17 > cap 13
- pool: {'hamstring': 4, 'abductors': 2, 'gluteal': 2, 'adductor': 1, 'calves': 3}
- final: ['105:hamstring:MACHINE 57.5x8 / 57.5x8 / 57.5x8', '108:gluteal:CABLE 32.5x8 / 32.5x8', '113:adductor:MACHINE 50.0x8 / 50.0x8', '115:abductors:MACHINE 50.0x8 / 50.0x8', '112:calves:MACHINE 50.0x8 / 50.0x8', '116:abductors:CABLE 32.5x8 / 32.5x8']

### legs-new-20-low-endurance (dev) - first fail S2 - success/none
- S2 [fail]: exercise count 1 vs target 4
- pool: {'quadriceps': 8, 'gluteal': 2, 'hamstring': 2}
- final: ['154:quadriceps:DUMBBELL, BODYWEIGHT Nonex15 / Nonex15']
- uncertain expectations: ["goal 'endurance' is not a BE GoalType value"]

### core-demo-20-low (test) - first fail S2 - success/none
- S2 [fail]: raw total sets 8 > cap 3
- pool: {'abs': 2, 'obliques': 1}
- final: ['21:abs:MACHINE Nonex8 / Nonex8']

### core-new-30-bodyweight (dev) - first fail S2 - success/none
- S2 [fail]: raw total sets 11 > cap 10
- pool: {'abs': 7, 'lower-back': 2, 'obliques': 3}
- final: ['24:abs:BODYWEIGHT Nonex12 / Nonex12 / Nonex12', '147:obliques:BODYWEIGHT Nonex12 / Nonex12 / Nonex12', '146:obliques:BODYWEIGHT Nonex12 / Nonex12', '135:lower-back:BODYWEIGHT Nonex30 / Nonex30']

### core-new-20-lowbackpain (test) - first fail S2 - success/none
- S2 [fail]: raw total sets 8 > cap 7
- pool: {'abs': 2, 'obliques': 1}
- final: ['21:abs:MACHINE Nonex15 / Nonex15 / Nonex15', '23:abs:MACHINE Nonex15 / Nonex15', '147:obliques:BODYWEIGHT Nonex15 / Nonex15']

### upper-demo-60 (dev) - first fail S2 - success/none
- S2 [fail]: raw picks missing required muscles ['biceps']; raw picks covers 5 of required-distinct 6 (['back-deltoids', 'chest', 'front-deltoids', 'triceps', 'upper-back']); raw total sets 17 > cap 13
- pool: {'chest': 2, 'upper-back': 2, 'front-deltoids': 2, 'back-deltoids': 2, 'triceps': 2, 'biceps': 2}
- final: ['61:upper-back:MACHINE 107.5x5 / 107.5x5', '40:chest:MACHINE 75.0x5 / 75.0x5 / 75.0x5', '75:front-deltoids:BARBELL 65.0x5 / 65.0x5', '142:back-deltoids:DUMBBELL, CABLE, MACHINE 17.5x5 / 17.5x5', '11:biceps:DUMBBELL, CABLE 15.0x8 / 15.0x8', '95:triceps:DUMBBELL, CABLE 15.0x8 / 15.0x8']

### upper-demo-90-hyp (dev) - first fail S2 - success/none
- S2 [fail]: raw picks missing required muscles ['biceps']; raw picks covers 5 of required-distinct 6 (['back-deltoids', 'chest', 'front-deltoids', 'triceps', 'upper-back'])
- pool: {'chest': 2, 'upper-back': 2, 'front-deltoids': 2, 'back-deltoids': 2, 'triceps': 2, 'biceps': 2}
- final: ['61:upper-back:MACHINE 100.0x8 / 100.0x8 / 100.0x8 / 100.0x8', '53:upper-back:MACHINE 100.0x8 / 100.0x8 / 100.0x8', '40:chest:MACHINE 70.0x8 / 70.0x8 / 70.0x8 / 70.0x8', '41:chest:MACHINE 70.0x8 / 70.0x8 / 70.0x8 / 70.0x8', '33:triceps:BARBELL 70.0x8 / 70.0x8', '76:front-deltoids:DUMBBELL 17.5x8 / 17.5x8 / 17.5x8', '142:back-deltoids:DUMBBELL, CABLE, MACHINE 17.5x8 / 17.5x8 / 17.5x8', '11:biceps:DUMBBELL, CABLE 15.0x12 / 15.0x12']

### upper-syn-30-bm (test) - first fail S2 - success/none
- S2 [fail]: raw total sets 11 > cap 10
- pool: {'upper-back': 2, 'chest': 2, 'front-deltoids': 2, 'back-deltoids': 2, 'biceps': 2, 'triceps': 2}
- final: ['58:upper-back:DUMBBELL 30.0x8 / 30.0x8 / 30.0x8', '34:chest:DUMBBELL 20.0x8 / 20.0x8', '76:front-deltoids:DUMBBELL 12.5x8 / 12.5x8', '88:back-deltoids:CABLE 10.0x12 / 10.0x12']
- uncertain expectations: ['6 coverage muscles but budget 4 exercises: only minDistinct=4 required']

### upper-demo-60-shoulderpain-doms2-chest (dev) - first fail S2 - success/none
- S2 [fail]: raw picks missing required muscles ['biceps']; raw picks covers 3 of required-distinct 4 (['back-deltoids', 'triceps', 'upper-back'])
- pool: {'upper-back': 6, 'back-deltoids': 2, 'triceps': 2, 'biceps': 2}
- final: ['61:upper-back:MACHINE 100.0x8 / 100.0x8 / 100.0x8 / 100.0x8', '53:upper-back:MACHINE 100.0x8 / 100.0x8 / 100.0x8', '63:upper-back:MACHINE 100.0x8 / 100.0x8 / 100.0x8', '142:back-deltoids:DUMBBELL, CABLE, MACHINE 17.5x8 / 17.5x8 / 17.5x8', '131:triceps:CABLE 22.5x12 / 22.5x12', '13:biceps:BARBELL, DUMBBELL, CABLE, MACHINE 15.0x12 / 15.0x12']

### upper-new-60-bodyweight (test) - first fail S2 - success/none
- S2 [fail]: raw picks missing required muscles ['triceps']; raw picks covers 3 of required-distinct 4 (['chest', 'front-deltoids', 'upper-back'])
- pool: {'upper-back': 3, 'chest': 5, 'triceps': 1, 'front-deltoids': 1}
- final: ['50:upper-back:BODYWEIGHT Nonex15 / Nonex15 / Nonex15 / Nonex15', '48:chest:BODYWEIGHT Nonex15 / Nonex15 / Nonex15 / Nonex15', '65:upper-back:BODYWEIGHT Nonex15 / Nonex15 / Nonex15 / Nonex15', '117:chest:BODYWEIGHT Nonex15 / Nonex15 / Nonex15', '139:triceps:BODYWEIGHT Nonex15 / Nonex15 / Nonex15', '141:front-deltoids:BODYWEIGHT Nonex15 / Nonex15 / Nonex15']
- uncertain expectations: ["goal 'endurance' is not a BE GoalType value"]

### lower-demo-45 (dev) - first fail S2 - success/none
- S2 [fail]: raw total sets 15 > cap 10
- pool: {'quadriceps': 8, 'hamstring': 2, 'gluteal': 2}
- final: ['99:quadriceps:BARBELL 92.5x5 / 92.5x5', '102:quadriceps:DUMBBELL, BARBELL, BODYWEIGHT 37.5x5 / 37.5x5', '105:hamstring:MACHINE 57.5x8 / 57.5x8', '110:gluteal:MACHINE 50.0x8 / 50.0x8', '108:gluteal:CABLE 32.5x8 / 32.5x8']

### lower-new-60-bm-fatloss (dev) - first fail S2 - success/none
- S2 [fail]: raw picks missing required muscles ['gluteal']; raw picks covers 2 of required-distinct 3 (['hamstring', 'quadriceps'])
- pool: {'quadriceps': 6, 'hamstring': 2, 'abductors': 2, 'gluteal': 2}
- final: ['114:quadriceps:DUMBBELL Nonex12 / Nonex12 / Nonex12 / Nonex12', '154:quadriceps:DUMBBELL, BODYWEIGHT Nonex12 / Nonex12 / Nonex12 / Nonex12', '159:hamstring:DUMBBELL, BODYWEIGHT Nonex12 / Nonex12 / Nonex12', '174:quadriceps:BODYWEIGHT, DUMBBELL Nonex12 / Nonex12 / Nonex12', '152:abductors:DUMBBELL, BODYWEIGHT Nonex12 / Nonex12 / Nonex12', '108:gluteal:CABLE Nonex15 / Nonex15 / Nonex15']

### lower-demo-30-low-doms2-glute (dev) - first fail S2 - success/none
- S2 [fail]: raw total sets 11 > cap 10
- pool: {'quadriceps': 8, 'hamstring': 2, 'gluteal': 2}
- final: ['99:quadriceps:BARBELL 82.5x8 / 82.5x8', '102:quadriceps:DUMBBELL, BARBELL, BODYWEIGHT 32.5x8 / 32.5x8', '105:hamstring:MACHINE 52.5x12 / 52.5x12', '108:gluteal:CABLE 30.0x12 / 30.0x12']
- uncertain expectations: ['DOMS-2 muscle still required (policy: light 1-2 sets allowed)']

### full-demo-60 (dev) - first fail S2 - success/none
- S2 [fail]: raw total sets 18 > cap 13
- pool: {'quadriceps': 3, 'chest': 4, 'upper-back': 1, 'front-deltoids': 1, 'hamstring': 1, 'gluteal': 1, 'abs': 1}
- final: ['61:upper-back:MACHINE 107.5x5 / 107.5x5', '99:quadriceps:BARBELL 92.5x5 / 92.5x5 / 92.5x5', '40:chest:MACHINE 75.0x5 / 75.0x5', '75:front-deltoids:BARBELL 65.0x5 / 65.0x5', '105:hamstring:MACHINE 57.5x8 / 57.5x8', '108:gluteal:CABLE 32.5x8 / 32.5x8']
- uncertain expectations: ['7 coverage muscles but budget 6 exercises: only minDistinct=6 required']

### full-new-45-hyp (dev) - first fail S2 - success/none
- S2 [fail]: raw picks covers 4 of required-distinct 5 (['abs', 'chest', 'quadriceps', 'upper-back'])
- pool: {'quadriceps': 3, 'gluteal': 1, 'chest': 2, 'upper-back': 3, 'hamstring': 1, 'front-deltoids': 1, 'abs': 1}
- final: ['41:chest:MACHINE Nonex8 / Nonex8 / Nonex8', '102:quadriceps:DUMBBELL, BARBELL, BODYWEIGHT Nonex8 / Nonex8 / Nonex8', '58:upper-back:DUMBBELL Nonex8 / Nonex8 / Nonex8', '75:front-deltoids:BARBELL Nonex8 / Nonex8 / Nonex8', '21:abs:MACHINE Nonex12 / Nonex12']
- uncertain expectations: ['7 coverage muscles but budget 5 exercises: only minDistinct=5 required']

### full-syn-90-hyp (test) - first fail S2 - success/none
- S2 [fail]: raw picks missing required muscles ['hamstring']; raw picks covers 6 of required-distinct 7 (['abs', 'chest', 'front-deltoids', 'gluteal', 'quadriceps', 'upper-back'])
- pool: {'quadriceps': 3, 'gluteal': 1, 'chest': 2, 'upper-back': 3, 'hamstring': 1, 'front-deltoids': 1, 'abs': 1}
- final: ['157:hamstring:MACHINE, BODYWEIGHT 72.5x8 / 72.5x8 / 72.5x8', '107:gluteal:BARBELL, DUMBBELL, MACHINE 57.5x8 / 57.5x8 / 57.5x8 / 57.5x8', '99:quadriceps:BARBELL 57.5x8 / 57.5x8 / 57.5x8', '41:chest:MACHINE 52.5x8 / 52.5x8 / 52.5x8 / 52.5x8', '58:upper-back:DUMBBELL 30.0x8 / 30.0x8 / 30.0x8', '102:quadriceps:DUMBBELL, BARBELL, BODYWEIGHT 22.5x8 / 22.5x8 / 22.5x8 / 22.5x8', '75:front-deltoids:BARBELL 45.0x8 / 45.0x8', '21:abs:MACHINE Nonex12 / Nonex12']

### full-demo-30-bodyweight (dev) - first fail S2 - success/none
- S2 [fail]: raw picks missing required muscles ['front-deltoids']; raw picks covers 3 of required-distinct 4 (['chest', 'quadriceps', 'upper-back']); raw total sets 12 > cap 10
- pool: {'upper-back': 3, 'chest': 5, 'quadriceps': 2, 'front-deltoids': 1}
- final: ['50:upper-back:BODYWEIGHT Nonex8 / Nonex8 / Nonex8', '48:chest:BODYWEIGHT Nonex8 / Nonex8', '118:quadriceps:BODYWEIGHT Nonex8 / Nonex8', '141:front-deltoids:BODYWEIGHT Nonex8 / Nonex8']

### full-new-60-nocable-kneepain (test) - first fail S2 - success/none
- S2 [fail]: raw picks missing required muscles ['front-deltoids', 'hamstring']; raw picks covers 4 of required-distinct 6 (['abs', 'chest', 'gluteal', 'upper-back'])
- pool: {'gluteal': 1, 'chest': 6, 'upper-back': 2, 'hamstring': 1, 'front-deltoids': 1, 'abs': 1}
- final: ['40:chest:MACHINE Nonex8 / Nonex8 / Nonex8 / Nonex8', '61:upper-back:MACHINE Nonex8 / Nonex8 / Nonex8 / Nonex8', '107:gluteal:BARBELL, DUMBBELL, MACHINE Nonex8 / Nonex8 / Nonex8', '157:hamstring:MACHINE, BODYWEIGHT Nonex8 / Nonex8', '75:front-deltoids:BARBELL Nonex8 / Nonex8', '21:abs:MACHINE Nonex12 / Nonex12']

### arm-demo-45 (dev) - first fail S2 - success/none
- S2 [fail]: raw picks missing required muscles ['forearm']; raw picks covers 2 of required-distinct 3 (['biceps', 'triceps'])
- pool: {'biceps': 5, 'forearm': 2, 'triceps': 5}
- final: ['33:triceps:BARBELL 70.0x8 / 70.0x8 / 70.0x8', '131:triceps:CABLE 22.5x12 / 22.5x12 / 22.5x12', '14:biceps:CABLE 15.0x12 / 15.0x12 / 15.0x12', '12:biceps:DUMBBELL 15.0x12 / 15.0x12 / 15.0x12', '1:forearm:BARBELL, DUMBBELL, CABLE 12.5x12 / 12.5x12 / 12.5x12']

### shoulder-new-30-shoulderpain (dev) - first fail S2 - success/none
- S2 [fail]: raw total sets 12 > cap 10
- pool: {'back-deltoids': 8, 'trapezius': 4}
- final: ['142:back-deltoids:DUMBBELL, CABLE, MACHINE Nonex8 / Nonex8 / Nonex8', '86:back-deltoids:MACHINE Nonex12 / Nonex12 / Nonex12', '121:back-deltoids:CABLE Nonex12 / Nonex12', '124:trapezius:BARBELL Nonex12 / Nonex12']

### custom-demo-45-bi-tri (dev) - first fail S2 - success/none
- S2 [fail]: raw total sets 13 > cap 10
- pool: {'biceps': 5, 'triceps': 7}
- final: ['33:triceps:BARBELL 75.0x5 / 75.0x5', '92:triceps:BARBELL, DUMBBELL, CABLE 37.5x8 / 37.5x8', '131:triceps:CABLE 25.0x8 / 25.0x8', '12:biceps:DUMBBELL 15.0x8 / 15.0x8', '11:biceps:DUMBBELL, CABLE 15.0x8 / 15.0x8']

### custom-demo-20-biceps (test) - first fail S2 - success/none
- S2 [fail]: raw total sets 8 > cap 6
- pool: {'biceps': 10}
- final: ['14:biceps:CABLE 15.0x12 / 15.0x12', '12:biceps:DUMBBELL 15.0x12 / 15.0x12', '130:biceps:DUMBBELL 15.0x12 / 15.0x12']

### custom-demo-30-bm-delts (dev) - first fail S2 - success/none
- S2 [fail]: raw total sets 11 > cap 10
- pool: {'front-deltoids': 6, 'back-deltoids': 6}
- final: ['90:front-deltoids:CABLE 27.5x8 / 27.5x8', '76:front-deltoids:DUMBBELL 17.5x8 / 17.5x8 / 17.5x8', '78:front-deltoids:DUMBBELL 17.5x8 / 17.5x8', '88:back-deltoids:CABLE 12.5x12 / 12.5x12']

### custom-new-30-high-abs-obliques (test) - first fail S2 - success/none
- S2 [fail]: raw total sets 11 > cap 10
- pool: {'abs': 10, 'obliques': 2}
- final: ['21:abs:MACHINE Nonex12 / Nonex12 / Nonex12', '144:abs:CABLE Nonex12 / Nonex12 / Nonex12', '29:obliques:CABLE Nonex12 / Nonex12', '146:obliques:BODYWEIGHT Nonex12 / Nonex12']

### fault-demo-60-push-offcandidate (dev) - first fail S2 - success/none
- S2 [fail]: raw total sets 14 > cap 13
- pool: {'chest': 8, 'front-deltoids': 2, 'triceps': 2}
- final: ['40:chest:MACHINE 75.0x5 / 75.0x5 / 75.0x5', '41:chest:MACHINE 75.0x5 / 75.0x5', '34:chest:DUMBBELL 30.0x5 / 30.0x5', '33:triceps:BARBELL 75.0x5 / 75.0x5', '76:front-deltoids:DUMBBELL 17.5x5 / 17.5x5', '95:triceps:DUMBBELL, CABLE 15.0x8 / 15.0x8']

### fault-demo-30-legs-badjson (dev) - first fail S2 - success/none
- S2 [fail]: raw total sets 12 > cap 10
- pool: {'quadriceps': 8, 'hamstring': 2, 'gluteal': 2}
- final: ['99:quadriceps:BARBELL 85.0x8 / 85.0x8', '102:quadriceps:DUMBBELL, BARBELL, BODYWEIGHT 35.0x8 / 35.0x8 / 35.0x8', '105:hamstring:MACHINE 52.5x12 / 52.5x12', '108:gluteal:CABLE 30.0x12 / 30.0x12']

### golden-ai-adapter-request (test) - first fail S2 - success/none
- S2 [fail]: exercise count 4 vs target 7
- S4 [fail]: final time 31 < 50% of 70
- pool: {'triceps': 9}
- final: ['91:triceps:BARBELL, DUMBBELL, CABLE Nonex12 / Nonex12', '93:triceps:CABLE Nonex12 / Nonex12', '96:triceps:CABLE Nonex12 / Nonex12', '95:triceps:DUMBBELL, CABLE Nonex12 / Nonex12', '131:triceps:CABLE Nonex12 / Nonex12', '92:triceps:BARBELL, DUMBBELL, CABLE Nonex12 / Nonex12']

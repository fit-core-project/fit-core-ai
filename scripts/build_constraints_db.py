import pandas as pd
import numpy as np
import os
import sqlite3
from datetime import datetime
import sys

print("=== Phase 1. Starting build_constraints_db.py ===")

# Paths
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
fit_core_ai_dir = str(SCRIPT_DIR.parent)
db_path = os.path.join(fit_core_ai_dir, "fit_core.sqlite")
BASE_DIR = str(SCRIPT_DIR.parent.parent.parent)
core_csv_path = os.path.join(BASE_DIR, "v25_canonical_filter_core.csv")

if fit_core_ai_dir not in sys.path:
    sys.path.append(fit_core_ai_dir)

try:
    from database import engine
    use_sqlalchemy = True
    print("[INFO] Successfully imported database engine from database.py.")
except Exception as e:
    print(f"[WARNING] Failed to import database engine ({e}). Will use sqlite3 directly.")
    use_sqlalchemy = False

if not os.path.exists(core_csv_path):
    print(f"Error: Core CSV file not found at {core_csv_path}")
    sys.exit(1)

# Read core data
df_core = pd.read_csv(core_csv_path, encoding='utf-8')
print(f"Loaded Core data size: {len(df_core)}")

# Current time
current_time_str = datetime.now().isoformat()

# Rule helpers
def clean_str(val):
    if pd.isna(val):
        return ""
    return str(val).lower().strip()


def row_text(row):
    return " ".join(
        clean_str(row.get(field, ""))
        for field in ["name", "pattern", "english"]
    )


def contains_any(text, keywords):
    return any(keyword in text for keyword in keywords)


def is_overhead_press_family(row):
    text = row_text(row)
    return contains_any(text, [
        "오버헤드 프레스",
        "overhead press",
        "밀리터리 프레스",
        "military press",
        "숄더 프레스",
        "shoulder press",
    ])


def is_hinge_family(row):
    text = row_text(row)
    return contains_any(text, [
        "데드리프트",
        "deadlift",
        "루마니안",
        "romanian",
        "굿모닝",
        "good morning",
        "goodmorning",
    ])

# ==========================================================
# 1. exercise_mobility_requirement (가동성 요구량)
# ==========================================================
print("Building exercise_mobility_requirement...")
mobility_rows = []

for idx, row in df_core.iterrows():
    eid = row['v24_row_uid']
    name = clean_str(row['name'])
    pattern = clean_str(row['pattern'])
    equipment = clean_str(row['equipment'])
    subpart = clean_str(row['subpart'])
    text = row_text(row)
    
    # ankle_dorsiflexion_demand
    ankle = "low"
    if contains_any(text, ["스쿼트", "squat", "런지", "lunge"]):
        if any(k in equipment for k in ["바벨", "덤벨", "맨몸", "케틀벨"]):
            ankle = "high"
        else:
            ankle = "medium"
    elif contains_any(text, ["레그 프레스", "레그프레스", "leg press"]):
        ankle = "medium"
    elif contains_any(text, ["카프레이즈", "calf raise"]):
        ankle = "high"
        
    # hip_flexion_demand
    hip = "low"
    if is_hinge_family(row):
        hip = "high"
    elif contains_any(text, ["스쿼트", "squat", "레그 프레스", "레그프레스", "lunge", "런지"]):
        hip = "medium"
        
    # shoulder_flexion_demand
    shoulder = "low"
    if is_overhead_press_family(row) or any(k in text for k in ["풀업", "pullup", "랫풀", "lat pull"]):
        shoulder = "high"
    elif contains_any(text, ["벤치 프레스", "벤치프레스", "bench press", "푸쉬업", "pushup", "딥스", "dips", "레이즈", "raise"]):
        shoulder = "medium"
        
    # wrist_extension_demand
    wrist = "low"
    if contains_any(text, ["푸쉬업", "pushup", "플랭크", "plank", "벤치 프레스", "bench press"]) and any(k in equipment for k in ["맨몸", "바벨"]):
        wrist = "high"
    elif is_overhead_press_family(row) or any(k in text for k in ["바벨 컬", "bicep curl"]):
        wrist = "medium"
        
    # overhead_position_required
    overhead_req = 0
    if is_overhead_press_family(row) or any(k in text for k in ["풀업", "pullup", "랫풀", "lat pull"]):
        overhead_req = 1
        
    mobility_rows.append({
        "exercise_id": eid,
        "ankle_dorsiflexion_demand": ankle,
        "hip_flexion_demand": hip,
        "shoulder_flexion_demand": shoulder,
        "wrist_extension_demand": wrist,
        "overhead_position_required": overhead_req,
        "source_type": "generated",
        "confidence": "medium",
        "review_required": 1,
        "updated_at": current_time_str
    })

df_mobility = pd.DataFrame(mobility_rows)

# ==========================================================
# 2. exercise_joint_load_profile (관절 부하 프로필)
# ==========================================================
print("Building exercise_joint_load_profile...")
load_rows = []

for idx, row in df_core.iterrows():
    eid = row['v24_row_uid']
    name = clean_str(row['name'])
    pattern = clean_str(row['pattern'])
    equipment = clean_str(row['equipment'])
    
    # lumbar_load
    english = clean_str(row.get('english', ''))
    lumbar = "low"
    has_lumbar_pattern = is_hinge_family(row) or any(
        any(k in field for k in ["데드리프트", "deadlift", "바벨로우", "바벨 로우", "bent over row", "티바", "t-bar", "스쿼트", "squat"])
        for field in [name, pattern, english]
    )
    if has_lumbar_pattern:
        if "바벨" in equipment or "프리웨이트" in equipment or "barbell" in equipment or "free weight" in equipment:
            lumbar = "high"
        else:
            lumbar = "medium"
            
    # axial_load
    axial = "low"
    has_axial_pattern = is_overhead_press_family(row) or any(
        any(k in field for k in ["스쿼트", "squat", "오버헤드 프레스", "밀리터리 프레스", "overhead press"])
        for field in [name, pattern, english]
    )
    if has_axial_pattern:
        if "바벨" in equipment or "barbell" in equipment:
            axial = "high"
        elif "덤벨" in equipment or "스미스" in equipment or "dumbbell" in equipment or "smith" in equipment:
            axial = "medium"
            
    # knee_shear_load
    knee = "low"
    has_knee_high = any(
        any(k in field for k in ["레그 익스텐션", "leg extension", "런지", "lunge", "피스톨"])
        for field in [name, pattern, english]
    )
    has_knee_med = any(
        any(k in field for k in ["스쿼트", "squat", "레그 프레스", "leg press"])
        for field in [name, pattern, english]
    )
    if has_knee_high:
        knee = "high"
    elif has_knee_med:
        knee = "medium"
        
    # shoulder_impingement_risk
    shoulder_imp = "low"
    has_shoulder_high = any(
        any(k in field for k in ["비하인드 넥", "behind neck", "업라이트 로우", "upright row", "딥스", "dips"])
        for field in [name, pattern, english]
    )
    has_shoulder_med = any(
        any(k in field for k in ["벤치 프레스", "bench press", "벤치프레스", "숄더 프레스", "shoulder press", "사이드 레터럴", "lateral raise"])
        for field in [name, pattern, english]
    )
    if has_shoulder_high:
        shoulder_imp = "high"
    elif has_shoulder_med:
        shoulder_imp = "medium"
        
    # wrist_stress
    wrist_str = "low"
    has_wrist_pattern = any(
        any(k in field for k in ["벤치 프레스", "bench press", "벤치프레스", "바벨 컬", "바벨컬", "bicep curl", "푸쉬업", "pushup", "푸시업"])
        for field in [name, pattern, english]
    )
    if has_wrist_pattern:
        if "바벨" in equipment or "barbell" in equipment:
            wrist_str = "high"
        elif "덤벨" in equipment or "dumbbell" in equipment:
            wrist_str = "medium"
            
    load_rows.append({
        "exercise_id": eid,
        "lumbar_load": lumbar,
        "axial_load": axial,
        "knee_shear_load": knee,
        "shoulder_impingement_risk": shoulder_imp,
        "wrist_stress": wrist_str,
        "source_type": "generated",
        "confidence": "medium",
        "review_required": 1,
        "updated_at": current_time_str
    })

df_load = pd.DataFrame(load_rows)

# ==========================================================
# 3. exercise_effect_profile (운동 효과 프로필)
# ==========================================================
print("Building exercise_effect_profile...")
effect_rows = []

for idx, row in df_core.iterrows():
    eid = row['v24_row_uid']
    release_tier = clean_str(row['v24_release_tier_norm'])
    quality_score = row['v24_preprocess_quality_score'] or 0
    pattern = clean_str(row['pattern'])
    equipment = clean_str(row['equipment'])
    text = row_text(row)
    
    # Default based on release tier
    eff_level = "low"
    if release_tier == "gold":
        eff_level = "high"
    elif release_tier == "silver":
        eff_level = "medium"
    elif release_tier == "bronze":
        eff_level = "low"
    else:
        eff_level = "unknown"
        
    # Multi-joint check
    is_compound = (
        is_hinge_family(row)
        or is_overhead_press_family(row)
        or any(k in text for k in ["스쿼트", "프레스", "로우", "풀업", "벤치", "squat", "press", "row", "pullup", "bench"])
    )
    
    hypertrophy = eff_level
    if eff_level == "high" and is_compound:
        hypertrophy = "elite"
        
    strength = eff_level
    if is_compound and "바벨" in equipment:
        strength = "elite" if release_tier == "gold" else "high"
        
    power = "low"
    if any(k in text for k in ["역도", "클린", "스내치", "점프", "클랩", "clean", "snatch", "jump", "power"]):
        power = "high"
        
    rehab = "none"
    if any(k in pattern for k in ["모빌리티", "스트레칭", "회전근개", "재활", "y-raise", "y레이즈", "w-raise", "슬라이드", "폼롤러"]):
        rehab = "high"
        
    loadability = "low"
    if "바벨" in equipment or "머신" in equipment:
        loadability = "high"
    elif "덤벨" in equipment:
        loadability = "medium"
        
    prog_ceiling = "low"
    if is_compound and ("바벨" in equipment or "머신" in equipment):
        prog_ceiling = "high"
        
    mobility_eff = "low"
    if any(k in text for k in ["스트레칭", "모빌리티", "요가", "필라테스", "foam", "stretching"]):
        mobility_eff = "high"
        
    uniqueness = "common"
    if is_hinge_family(row) or is_overhead_press_family(row) or any(k in text for k in ["스쿼트", "벤치 프레스", "풀업"]):
        uniqueness = "hard_to_replace"
        
    stim_to_fatigue = "good"
    if is_compound and "바벨" in equipment:
        stim_to_fatigue = "fair" # High fatigue
    elif "머신" in equipment:
        stim_to_fatigue = "excellent"
        
    effect_rows.append({
        "exercise_id": eid,
        "hypertrophy_effect": hypertrophy,
        "strength_effect": strength,
        "power_effect": power,
        "rehab_utility": rehab,
        "loadability": loadability,
        "progression_ceiling": prog_ceiling,
        "mobility_effect": mobility_eff,
        "uniqueness": uniqueness,
        "stimulus_to_fatigue": stim_to_fatigue,
        "source_type": "generated",
        "confidence": "medium",
        "review_required": 1,
        "updated_at": current_time_str
    })

df_effect = pd.DataFrame(effect_rows)

# ==========================================================
# 4. exercise_difficulty_profile (상세 난이도 프로필)
# ==========================================================
print("Building exercise_difficulty_profile...")
diff_rows = []

for idx, row in df_core.iterrows():
    eid = row['v24_row_uid']
    equipment = clean_str(row['equipment'])
    pattern = clean_str(row['pattern'])
    difficulty_text = clean_str(row['difficulty'])
    text = row_text(row)
    
    # technical_difficulty
    tech = "low"
    if difficulty_text == "고급":
        tech = "high"
    elif difficulty_text == "중급":
        tech = "medium"
    
    # Override by equipment complexity
    if "바벨" in equipment and (is_hinge_family(row) or any(k in text for k in ["스쿼트", "역도"])):
        tech = "high"
        
    # balance_requirement
    bal = "low"
    if "덤벨" in equipment:
        bal = "medium"
    if any(k in text for k in ["한 다리", "싱글", "원 레그", "피스톨", "편측", "single leg", "unilateral"]):
        bal = "high"
        
    # failure_penalty
    penalty = "low"
    if "바벨" in equipment and (is_overhead_press_family(row) or any(k in text for k in ["벤치 프레스", "스쿼트"])):
        penalty = "high" # 깔림 위험
    elif "덤벨" in equipment and (is_overhead_press_family(row) or any(k in text for k in ["스쿼트", "벤치 프레스"])):
        penalty = "medium"
        
    diff_rows.append({
        "exercise_id": eid,
        "technical_difficulty": tech,
        "balance_requirement": bal,
        "failure_penalty": penalty,
        "source_type": "generated",
        "confidence": "medium",
        "review_required": 1,
        "updated_at": current_time_str
    })

df_diff = pd.DataFrame(diff_rows)

# ==========================================================
# 5. exercise_anthropometry_sensitivity_profile (체형/세팅 민감도 프로필)
# ==========================================================
print("Building exercise_anthropometry_sensitivity_profile...")
anthro_rows = []

for idx, row in df_core.iterrows():
    eid = row['v24_row_uid']
    pattern = clean_str(row['pattern'])
    equipment = clean_str(row['equipment'])
    text = row_text(row)
    
    limb = "low"
    femur = "low"
    arm = "low"
    torso = "low"
    adjust = "unknown"
    setup = 0
    
    if is_hinge_family(row) or any(k in text for k in ["스쿼트", "벤치 프레스", "squat", "bench press"]):
        limb = "high"
        
    if any(k in pattern for k in ["스쿼트", "squat", "런지", "lunge"]):
        if "바벨" in equipment or "프리웨이트" in equipment:
            femur = "high"
            torso = "high"
            
    if is_hinge_family(row) or any(k in text for k in ["벤치 프레스", "bench press"]):
        if "바벨" in equipment:
            arm = "high"
            
    if "머신" in equipment or "시티드" in pattern:
        adjust = "good"
        setup = 1
        
    anthro_rows.append({
        "exercise_id": eid,
        "limb_length_sensitivity": limb,
        "long_femur_sensitivity": femur,
        "long_arm_sensitivity": arm,
        "torso_angle_demand": torso,
        "machine_adjustability": adjust,
        "setup_modification_available": setup,
        "source_type": "generated",
        "confidence": "medium",
        "review_required": 1,
        "updated_at": current_time_str
    })

df_anthro = pd.DataFrame(anthro_rows)

# ==========================================================
# 6. exercise_condition_policy (수술/질환 정책 테이블 - Sparse Many-to-Many)
# ==========================================================
print("Building exercise_condition_policy...")
policy_rows = []

# 가이드라인에 기초한 수술/질환 정책 매핑 규칙 생성 (재활 1단계 예시)
# 1) 디스크 환자 (lumbar_herniation)
# 2) 무릎 십자인대 수술 후 (post_acl_reconstruction)
for idx, row in df_core.iterrows():
    eid = row['v24_row_uid']
    name = clean_str(row['name'])
    pattern = clean_str(row['pattern'])
    equipment = clean_str(row['equipment'])
    
    clean_name = name.replace(" ", "")
    clean_pattern = pattern.replace(" ", "")
    
    # 디스크
    is_lumbar_heavy = (
        any(k in clean_name or k in clean_pattern for k in ["데드리프트", "deadlift", "바벨로우", "bentoverrow", "백스쿼트", "backsquat", "굿모닝", "goodmorning"]) and
        any(k in equipment for k in ["바벨", "덤벨", "프리웨이트"])
    )
    
    if is_lumbar_heavy:
        policy_rows.append({
            "condition_code": "lumbar_herniation",
            "condition_label_ko": "요추 디스크 질환",
            "exercise_id": eid,
            "phase": "rehab",
            "policy": "avoid",
            "modification": "상체 숙임 프리웨이트 금지, 등받이 지지 머신 대체 권장",
            "replacement_exercise_id": "V24ROW-008284", # 디스크 회피 기본 대체 (매트 쏘우 등 안전 하체)
            "requires_professional_clearance": 1,
            "source_type": "generated",
            "confidence": "medium",
            "review_required": 1,
            "updated_at": current_time_str
        })
    elif any(k in clean_name or k in clean_pattern for k in ["레그프레스", "legpress", "시티드인클라인", "스미스머신스쿼트"]):
        policy_rows.append({
            "condition_code": "lumbar_herniation",
            "condition_label_ko": "요추 디스크 질환",
            "exercise_id": eid,
            "phase": "rehab",
            "policy": "caution",
            "modification": "요추 말림 주의, 가동범위 제한 수행",
            "replacement_exercise_id": None,
            "requires_professional_clearance": 0,
            "source_type": "generated",
            "confidence": "medium",
            "review_required": 1,
            "updated_at": current_time_str
        })
        
    # 십자인대 무릎 수술
    if any(k in clean_name or k in clean_pattern for k in ["레그익스텐션", "legextension", "런지", "lunge"]):
        policy_rows.append({
            "condition_code": "post_acl_reconstruction",
            "condition_label_ko": "전방십자인대 재건술 후",
            "exercise_id": eid,
            "phase": "rehab",
            "policy": "avoid",
            "modification": "초기 재활 단계에서는 무릎의 개방사슬 신전(Open Chain Extension) 피하기",
            "replacement_exercise_id": "V24ROW-008284",
            "requires_professional_clearance": 1,
            "source_type": "generated",
            "confidence": "medium",
            "review_required": 1,
            "updated_at": current_time_str
        })
    elif any(k in clean_name or k in clean_pattern for k in ["레그컬", "legcurl"]):
        policy_rows.append({
            "condition_code": "post_acl_reconstruction",
            "condition_label_ko": "전방십자인대 재건술 후",
            "exercise_id": eid,
            "phase": "rehab",
            "policy": "prefer_rehab",
            "modification": "햄스트링 수축을 통한 슬관절 안정화 운동 추천",
            "replacement_exercise_id": None,
            "requires_professional_clearance": 0,
            "source_type": "generated",
            "confidence": "medium",
            "review_required": 1,
            "updated_at": current_time_str
        })

df_policy = pd.DataFrame(policy_rows)

# ==========================================================
# 7. exercise_regression_progression_map (대체/회귀 맵 테이블 - Sparse Relation)
# ==========================================================
print("Building exercise_regression_progression_map...")
map_rows = []

# 데이터 매핑의 신뢰도를 보장하기 위해 핵심 정보사전 구축
core_dict = {}
for idx, row in df_core.iterrows():
    eid = row['v24_row_uid']
    name_str = clean_str(row['name'])
    pattern_str = clean_str(row['pattern'])
    equipment_str = clean_str(row['equipment'])
    primary = clean_str(row['primary_muscle'])
    
    # 1:1로 앞에서 빌드한 lumbar_load 및 axial_load 매핑 조회용 규칙 사전 생성
    english_str = clean_str(row.get('english', ''))
    lumbar = "low"
    has_lumbar = is_hinge_family(row) or any(
        any(k in field for k in ["데드리프트", "deadlift", "바벨로우", "바벨 로우", "bent over row", "티바", "t-bar", "스쿼트", "squat"])
        for field in [name_str, pattern_str, english_str]
    )
    if has_lumbar:
        if "바벨" in equipment_str or "프리웨이트" in equipment_str or "barbell" in equipment_str or "free weight" in equipment_str:
            lumbar = "high"
        else:
            lumbar = "medium"
            
    axial = "low"
    has_axial = is_overhead_press_family(row) or any(
        any(k in field for k in ["스쿼트", "squat", "오버헤드 프레스", "밀리터리 프레스", "overhead press"])
        for field in [name_str, pattern_str, english_str]
    )
    if has_axial:
        if "바벨" in equipment_str or "barbell" in equipment_str:
            axial = "high"
        elif "덤벨" in equipment_str or "스미스" in equipment_str or "dumbbell" in equipment_str or "smith" in equipment_str:
            axial = "medium"
            
    # pain_risk
    pain_risk = clean_str(row.get('v24_pain_risk_norm', 'low'))
    english_str = clean_str(row.get('english', ''))
    quality = row.get('v24_preprocess_quality_score') or 0
    
    core_dict[eid] = {
        "uid": eid,
        "name": name_str,
        "english": english_str,
        "pattern": pattern_str,
        "equipment": equipment_str,
        "primary_muscle": primary,
        "lumbar_load": lumbar,
        "axial_load": axial,
        "pain_risk": pain_risk,
        "quality_score": quality
    }

# 대표 운동들의 ID 리스트 추출 (한글명/영문명/장비명칭 매핑 보정)
back_squat_ids = [
    eid for eid, val in core_dict.items()
    if (any(k in val['name'] for k in ["백스쿼트", "백 스쿼트"]) or any(k in val['english'] for k in ["back squat"]))
    and ("바벨" in val['equipment'] or "barbell" in val['equipment'])
]
leg_press_ids = [
    eid for eid, val in core_dict.items()
    if any(k in val['name'] for k in ["레그 프레스", "레그프레스"]) or any(k in val['english'] for k in ["leg press"])
]
deadlift_ids = [
    eid for eid, val in core_dict.items()
    if (any(k in val['name'] for k in ["데드리프트", "컨벤셔널"]) or any(k in val['english'] for k in ["deadlift", "conventional"]))
    and ("바벨" in val['equipment'] or "barbell" in val['equipment'])
    and not any(k in val['name'] or k in val['english'] for k in ["루마니안", "romanian", "스티프", "stiff", "덤벨", "dumbbell"])
]
romanian_ids = [
    eid for eid, val in core_dict.items()
    if any(k in val['name'] for k in ["루마니안", "로마니안"]) or any(k in val['english'] for k in ["romanian"])
]
db_deadlift_ids = [
    eid for eid, val in core_dict.items()
    if (any(k in val['name'] for k in ["데드리프트"]) or any(k in val['english'] for k in ["deadlift"]))
    and ("덤벨" in val['equipment'] or "dumbbell" in val['equipment'])
]
bench_press_ids = [
    eid for eid, val in core_dict.items()
    if (any(k in val['name'] for k in ["벤치 프레스", "벤치프레스"]) or any(k in val['english'] for k in ["bench press"]))
    and ("바벨" in val['equipment'] or "barbell" in val['equipment'])
    and not any(k in val['name'] or k in val['english'] for k in ["덤벨", "dumbbell", "인클라인", "incline", "디클라인", "decline"])
]
db_bench_ids = [
    eid for eid, val in core_dict.items()
    if (any(k in val['name'] for k in ["덤벨 벤치", "덤벨 프레스", "덤벨 플랫 벤치 프레스"]) or any(k in val['english'] for k in ["dumbbell bench", "dumbbell press"]))
    and not any(k in val['name'] or k in val['english'] for k in ["인클라인", "incline", "디클라인", "decline"])
]

# 주동근의 대표 토큰(첫 번째 근육)을 추출하는 헬퍼 함수
def get_primary_token(muscle_str):
    if not muscle_str:
        return ""
    return muscle_str.split(',')[0].strip()

# 1) 백스쿼트 -> 레그프레스 (발목 가동성 회피 대체 - mobilityAlternative)
for bs_id in back_squat_ids[:20]:
    bs = core_dict[bs_id]
    count = 0
    for lp_id in leg_press_ids:
        lp = core_dict[lp_id]
        if bs_id == lp_id: continue
        # 대표 주동근 일치 및 품질 제한 필터 (mobilityAlternative는 pain_risk=high 허용)
        if get_primary_token(bs['primary_muscle']) == get_primary_token(lp['primary_muscle']) and lp['quality_score'] >= 75:
            map_rows.append({
                "source_exercise_id": bs_id,
                "target_exercise_id": lp_id,
                "relation_type": "mobilityAlternative",
                "constraint_code": "limited_ankle_dorsiflexion",
                "reason": "발목 배굴 요구도가 낮고 하체 자극은 유사하게 유지됨",
                "source_type": "generated",
                "confidence": "medium",
                "review_required": 1,
                "updated_at": current_time_str
            })
            count += 1
            if count >= 3: break

# 2) 바벨 데드리프트 -> 덤벨 데드리프트 (장비 대체 - equipmentAlternative)
# equipmentAlternative: 주동근 일치 필수 (pain_risk=high 허용)
for dl_id in deadlift_ids[:20]:
    dl = core_dict[dl_id]
    count = 0
    for dbd_id in db_deadlift_ids:
        dbd = core_dict[dbd_id]
        if dl_id == dbd_id: continue
        if get_primary_token(dl['primary_muscle']) == get_primary_token(dbd['primary_muscle']) and dbd['quality_score'] >= 75:
            map_rows.append({
                "source_exercise_id": dl_id,
                "target_exercise_id": dbd_id,
                "relation_type": "equipmentAlternative",
                "constraint_code": "limited_wrist_extension",
                "reason": "바벨 대신 덤벨을 사용하여 손목 각도를 유연하게 유지하고 척추 중립을 잡기 수월함 (주동근 일치)",
                "source_type": "generated",
                "confidence": "medium",
                "review_required": 1,
                "updated_at": current_time_str
            })
            count += 1
            if count >= 2: break

# 3) 바벨 데드리프트 -> 루마니안 데드리프트 (변형 - variation)
# variation: 주동근 일치 및 품질 (pain_risk=high 허용)
# 요추 부하가 존재하여 요추 통증 조건(lumbar_pain_recent)은 배제하고 가동성 제한(limited_hip_flexion)으로 매핑
for dl_id in deadlift_ids[:20]:
    dl = core_dict[dl_id]
    count = 0
    for r_id in romanian_ids:
        r = core_dict[r_id]
        if dl_id == r_id: continue
        if get_primary_token(dl['primary_muscle']) == get_primary_token(r['primary_muscle']) and r['quality_score'] >= 75:
            map_rows.append({
                "source_exercise_id": dl_id,
                "target_exercise_id": r_id,
                "relation_type": "variation",
                "constraint_code": "limited_hip_flexion",
                "reason": "데드리프트 동작의 변형(variation)으로, 힙힌지 및 햄스트링 자극을 강조함 (요추 디스크/통증 추천에서 제외)",
                "source_type": "generated",
                "confidence": "medium",
                "review_required": 1,
                "updated_at": current_time_str
            })
            count += 1
            if count >= 2: break

# 4) 바벨 데드리프트 -> 요추 안전 회피 대체 세분화
# - Hip Thrust -> safetyAlternative (둔근/햄스트링 커버, pain_risk_level != 'high' 보장)
# - Leg Curl -> regression (hamstring isolation, pain_risk_level != 'high' 보장)
# - Back Extension -> rehabAlternative (요추 통증 우회이나 high-risk이므로 임상 검토 rehabAlternative로 격리)
safe_candidates = [
    eid for eid, val in core_dict.items()
    if val['lumbar_load'] == "low"
    and val['axial_load'] == "low"
    and val['quality_score'] >= 75
    and (any(k in val['name'] for k in ["힙 쓰러스트", "hip thrust", "레그 컬", "leg curl", "하이퍼 익스텐션", "hyper extension", "백 익스텐션"])
         or any(k in val['english'] for k in ["hip thrust", "leg curl", "hyper extension", "back extension"]))
]

for dl_id in deadlift_ids[:20]:
    dl = core_dict[dl_id]
    safety_count = 0
    regression_count = 0
    rehab_count = 0
    for safe_id in safe_candidates:
        if dl_id == safe_id: continue
        val = core_dict[safe_id]
        
        # 4-A) Hip Thrust -> safetyAlternative (pain_risk != 'high' 필수)
        if (any(k in val['name'] for k in ["힙 쓰러스트", "hip thrust"]) or any(k in val['english'] for k in ["hip thrust"])) and val['pain_risk'] != 'high':
            if safety_count < 2:
                map_rows.append({
                    "source_exercise_id": dl_id,
                    "target_exercise_id": safe_id,
                    "relation_type": "safetyAlternative",
                    "constraint_code": "lumbar_pain_recent",
                    "reason": "hip-extension safetyAlternative: 요추 수직 압박 부하 없이 대둔근(gluteal) 및 햄스트링 후면 사슬을 활성화하는 안전 대체",
                    "source_type": "generated",
                    "confidence": "medium",
                    "review_required": 1,
                    "updated_at": current_time_str
                })
                safety_count += 1
            
        # 4-B) Leg Curl -> regression (pain_risk != 'high' 필수)
        elif (any(k in val['name'] for k in ["레그 컬", "leg curl"]) or any(k in val['english'] for k in ["leg curl"])) and val['pain_risk'] != 'high':
            if regression_count < 2:
                map_rows.append({
                    "source_exercise_id": dl_id,
                    "target_exercise_id": safe_id,
                    "relation_type": "regression",
                    "constraint_code": "lumbar_pain_recent",
                    "reason": "hamstring isolation regression: 척추 기립근 개입을 배제하고 대퇴이두근만을 고립 수축하는 회귀 단계 운동",
                    "source_type": "generated",
                    "confidence": "medium",
                    "review_required": 1,
                    "updated_at": current_time_str
                })
                regression_count += 1
            
        # 4-C) Back/Hyper Extension -> rehabAlternative (요추 통증 우회이나 pain_risk_level=high 가능성으로 일반 대체 노출 차단)
        elif any(k in val['name'] for k in ["하이퍼 익스텐션", "hyper extension", "백 익스텐션"]) or any(k in val['english'] for k in ["hyper extension", "back extension"]):
            if rehab_count < 2:
                map_rows.append({
                    "source_exercise_id": dl_id,
                    "target_exercise_id": safe_id,
                    "relation_type": "rehabAlternative",
                    "constraint_code": "lumbar_pain_recent",
                    "reason": "rehabAlternative: 요추 수직 압박 부하는 없으나 척추 기립근 자체의 강한 수축을 동반하므로 전문 재활 지도가 필요한 대체 동작",
                    "source_type": "generated",
                    "confidence": "medium",
                    "review_required": 1,
                    "updated_at": current_time_str
                })
                rehab_count += 1
                
        if safety_count >= 2 and regression_count >= 2 and rehab_count >= 2:
            break


# 5) 바벨 벤치프레스 -> 덤벨 벤치프레스 (손목 회피 및 장비 대체)
for bp_id in bench_press_ids[:20]:
    bp = core_dict[bp_id]
    count = 0
    for dbp_id in db_bench_ids:
        dbp = core_dict[dbp_id]
        if bp_id == dbp_id: continue
        
        is_press = any(k in dbp['name'] or k in dbp['pattern'] for k in ["프레스", "press"])
        is_pullover_kickback = any(k in dbp['name'] for k in ["풀오버", "pullover", "킥백", "kickback", "삼두", "triceps"])
        
        # 대표 주동근 일칭 및 프레스 계열, 킥백 제외, 품질 조건
        if (get_primary_token(bp['primary_muscle']) == get_primary_token(dbp['primary_muscle']) 
            and is_press 
            and not is_pullover_kickback 
            and dbp['quality_score'] >= 75):
            
            # painAlternative 적재 시에는 pain_risk != 'high' 일 때만 적재
            if dbp['pain_risk'] != 'high':
                map_rows.append({
                    "source_exercise_id": bp_id,
                    "target_exercise_id": dbp_id,
                    "relation_type": "painAlternative",
                    "constraint_code": "limited_wrist_extension",
                    "reason": "손목 그립 각도를 조절할 수 있어 손목 꺾임 부하가 감소됨",
                    "source_type": "generated",
                    "confidence": "medium",
                    "review_required": 1,
                    "updated_at": current_time_str
                })
            
            # equipmentAlternative 적재 시에는 pain_risk=high도 허용하여 장비 대체 구색 확보
            map_rows.append({
                "source_exercise_id": bp_id,
                "target_exercise_id": dbp_id,
                "relation_type": "equipmentAlternative",
                "constraint_code": "limited_wrist_extension",
                "reason": "바벨 대신 덤벨을 사용하는 가슴 볼륨 보존 장비 대체 (주동근 일치)",
                "source_type": "generated",
                "confidence": "medium",
                "review_required": 1,
                "updated_at": current_time_str
            })
            count += 1
            if count >= 3: break

df_map = pd.DataFrame(map_rows)

# ==========================================================
# SQLite DB에 테이블 생성 및 적재 진행
# ==========================================================
print(f"Connecting to SQLite database: {db_path}")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Enable FK support
cursor.execute("PRAGMA foreign_keys = ON;")

# SQLite에서 FK 대상이 되려면 parent key가 PK 혹은 UNIQUE 인덱스를 가지고 있어야 하므로 선제 생성
cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_exercise_tier_id ON exercise_tier(id);")

# DDL Create Statements
ddl_statements = [
    # 1. exercise_mobility_requirement
    """
    CREATE TABLE IF NOT EXISTS exercise_mobility_requirement (
        exercise_id TEXT PRIMARY KEY,
        ankle_dorsiflexion_demand TEXT,
        hip_flexion_demand TEXT,
        shoulder_flexion_demand TEXT,
        wrist_extension_demand TEXT,
        overhead_position_required INTEGER,
        source_type TEXT,
        confidence TEXT,
        review_required INTEGER,
        updated_at TEXT,
        FOREIGN KEY (exercise_id) REFERENCES exercise_tier(id) ON DELETE CASCADE
    );
    """,
    # 2. exercise_joint_load_profile
    """
    CREATE TABLE IF NOT EXISTS exercise_joint_load_profile (
        exercise_id TEXT PRIMARY KEY,
        lumbar_load TEXT,
        axial_load TEXT,
        knee_shear_load TEXT,
        shoulder_impingement_risk TEXT,
        wrist_stress TEXT,
        source_type TEXT,
        confidence TEXT,
        review_required INTEGER,
        updated_at TEXT,
        FOREIGN KEY (exercise_id) REFERENCES exercise_tier(id) ON DELETE CASCADE
    );
    """,
    # 3. exercise_effect_profile
    """
    CREATE TABLE IF NOT EXISTS exercise_effect_profile (
        exercise_id TEXT PRIMARY KEY,
        hypertrophy_effect TEXT,
        strength_effect TEXT,
        power_effect TEXT,
        rehab_utility TEXT,
        loadability TEXT,
        progression_ceiling TEXT,
        mobility_effect TEXT,
        uniqueness TEXT,
        stimulus_to_fatigue TEXT,
        source_type TEXT,
        confidence TEXT,
        review_required INTEGER,
        updated_at TEXT,
        FOREIGN KEY (exercise_id) REFERENCES exercise_tier(id) ON DELETE CASCADE
    );
    """,
    # 4. exercise_difficulty_profile
    """
    CREATE TABLE IF NOT EXISTS exercise_difficulty_profile (
        exercise_id TEXT PRIMARY KEY,
        technical_difficulty TEXT,
        balance_requirement TEXT,
        failure_penalty TEXT,
        source_type TEXT,
        confidence TEXT,
        review_required INTEGER,
        updated_at TEXT,
        FOREIGN KEY (exercise_id) REFERENCES exercise_tier(id) ON DELETE CASCADE
    );
    """,
    # 5. exercise_anthropometry_sensitivity_profile
    """
    CREATE TABLE IF NOT EXISTS exercise_anthropometry_sensitivity_profile (
        exercise_id TEXT PRIMARY KEY,
        limb_length_sensitivity TEXT,
        long_femur_sensitivity TEXT,
        long_arm_sensitivity TEXT,
        torso_angle_demand TEXT,
        machine_adjustability TEXT,
        setup_modification_available INTEGER,
        source_type TEXT,
        confidence TEXT,
        review_required INTEGER,
        updated_at TEXT,
        FOREIGN KEY (exercise_id) REFERENCES exercise_tier(id) ON DELETE CASCADE
    );
    """,
    # 6. exercise_condition_policy
    """
    CREATE TABLE IF NOT EXISTS exercise_condition_policy (
        condition_code TEXT,
        condition_label_ko TEXT,
        exercise_id TEXT,
        phase TEXT,
        policy TEXT,
        modification TEXT,
        replacement_exercise_id TEXT,
        requires_professional_clearance INTEGER,
        source_type TEXT,
        confidence TEXT,
        review_required INTEGER,
        updated_at TEXT,
        PRIMARY KEY (condition_code, exercise_id, phase),
        FOREIGN KEY (exercise_id) REFERENCES exercise_tier(id) ON DELETE CASCADE,
        FOREIGN KEY (replacement_exercise_id) REFERENCES exercise_tier(id) ON DELETE SET NULL
    );
    """,
    # 7. exercise_regression_progression_map
    """
    CREATE TABLE IF NOT EXISTS exercise_regression_progression_map (
        source_exercise_id TEXT,
        target_exercise_id TEXT,
        relation_type TEXT CHECK(relation_type IN ('regression', 'progression', 'equipmentAlternative', 'painAlternative', 'mobilityAlternative', 'rehabAlternative', 'safetyAlternative', 'variation')),
        constraint_code TEXT,
        reason TEXT,
        source_type TEXT,
        confidence TEXT,
        review_required INTEGER,
        updated_at TEXT,
        PRIMARY KEY (source_exercise_id, target_exercise_id, relation_type, constraint_code),
        FOREIGN KEY (source_exercise_id) REFERENCES exercise_tier(id) ON DELETE CASCADE,
        FOREIGN KEY (target_exercise_id) REFERENCES exercise_tier(id) ON DELETE CASCADE
    );
    """,
    # 8. user_mobility_profile
    """
    CREATE TABLE IF NOT EXISTS user_mobility_profile (
        user_id TEXT PRIMARY KEY NOT NULL,
        ankle_dorsiflexion_level TEXT,
        hip_flexion_level TEXT,
        shoulder_flexion_level TEXT,
        wrist_extension_level TEXT
    );
    """,
    # 9. user_anthropometry_profile
    """
    CREATE TABLE IF NOT EXISTS user_anthropometry_profile (
        user_id TEXT PRIMARY KEY NOT NULL,
        height_cm REAL,
        arm_span_cm REAL,
        inseam_cm REAL,
        femur_length_level TEXT,
        arm_length_level TEXT,
        torso_length_level TEXT
    );
    """,
    # 10. user_condition_profile
    """
    CREATE TABLE IF NOT EXISTS user_condition_profile (
        condition_instance_id TEXT PRIMARY KEY NOT NULL,
        user_id TEXT NOT NULL,
        condition_code TEXT NOT NULL,
        body_part TEXT,
        side TEXT,
        phase TEXT,
        professional_clearance INTEGER,
        note TEXT
    );
    """
]

print("Executing DDL Statements...")
# Drop existing tables to ensure clean schema update
tables_to_drop = [
    "exercise_mobility_requirement",
    "exercise_joint_load_profile",
    "exercise_effect_profile",
    "exercise_difficulty_profile",
    "exercise_anthropometry_sensitivity_profile",
    "exercise_condition_policy",
    "exercise_regression_progression_map",
    "user_mobility_profile",
    "user_anthropometry_profile",
    "user_condition_profile"
]
for table in tables_to_drop:
    cursor.execute(f"DROP TABLE IF EXISTS {table};")

for ddl in ddl_statements:
    cursor.execute(ddl)
conn.commit()

# Save dataframes to database using to_sql (if_exists='replace' requires attention to primary keys)
# To preserve PRIMARY KEY definitions, we truncate/delete existing rows and insert them.
dfs_to_load = [
    ("exercise_mobility_requirement", df_mobility),
    ("exercise_joint_load_profile", df_load),
    ("exercise_effect_profile", df_effect),
    ("exercise_difficulty_profile", df_diff),
    ("exercise_anthropometry_sensitivity_profile", df_anthro),
    ("exercise_condition_policy", df_policy),
    ("exercise_regression_progression_map", df_map)
]

for table_name, df in dfs_to_load:
    print(f"Loading {len(df)} rows into {table_name}...")
    cursor.execute(f"DELETE FROM {table_name};")
    df.to_sql(table_name, conn, if_exists='append', index=False)
    conn.commit()

# Clean up any NULL condition_instance_id rows from user_condition_profile
cursor.execute("DELETE FROM user_condition_profile WHERE condition_instance_id IS NULL;")
conn.commit()

# Insert test user data
print("Inserting test_user data for QA...")
cursor.execute("DELETE FROM user_mobility_profile WHERE user_id = 'test_user';")
cursor.execute("""
    INSERT INTO user_mobility_profile (user_id, ankle_dorsiflexion_level, hip_flexion_level, shoulder_flexion_level, wrist_extension_level)
    VALUES ('test_user', 'limited', 'normal', 'normal', 'normal');
""")

cursor.execute("DELETE FROM user_anthropometry_profile WHERE user_id = 'test_user';")
cursor.execute("""
    INSERT INTO user_anthropometry_profile (user_id, height_cm, arm_span_cm, inseam_cm, femur_length_level, arm_length_level, torso_length_level)
    VALUES ('test_user', 175.0, 178.0, 82.0, 'long', 'average', 'average');
""")

cursor.execute("DELETE FROM user_condition_profile WHERE user_id = 'test_user';")
cursor.execute("""
    INSERT INTO user_condition_profile (condition_instance_id, user_id, condition_code, body_part, side, phase, professional_clearance, note)
    VALUES ('inst_001', 'test_user', 'lumbar_herniation', 'lower_back', 'bilateral', 'rehab', 0, '요추 4-5번 허리디스크 초기 재활 단계');
""")
conn.commit()

# Validation asserts for relation map count in fit_core.sqlite
print("Validating relation map counts in database...")
cursor.execute("SELECT relation_type, COUNT(*) FROM exercise_regression_progression_map GROUP BY relation_type;")
counts = dict(cursor.fetchall())
print("Relation type counts:", counts)

# Assertions
assert counts.get("mobilityAlternative", 0) > 0, "Error: mobilityAlternative count is 0!"
assert counts.get("regression", 0) > 0, "Error: regression count is 0!"
assert counts.get("equipmentAlternative", 0) > 0, "Error: equipmentAlternative count is 0!"
assert counts.get("safetyAlternative", 0) > 0, "Error: safetyAlternative count is 0!"
assert counts.get("variation", 0) > 0, "Error: variation count is 0!"
assert counts.get("rehabAlternative", 0) > 0, "Error: rehabAlternative count is 0!"

# Verify that no high-risk target exercises are present for painAlternative, safetyAlternative, regression
cursor.execute("""
    SELECT COUNT(*)
    FROM exercise_regression_progression_map m
    JOIN exercise_tier t ON t.id = m.target_exercise_id
    WHERE m.relation_type IN ('painAlternative', 'safetyAlternative', 'regression')
      AND t.pain_risk_level = 'high';
""")
high_risk_targets_count = cursor.fetchone()[0]
print(f"High risk targets count for pain/safety/regression: {high_risk_targets_count}")
assert high_risk_targets_count == 0, f"Error: Found {high_risk_targets_count} high-risk targets in pain/safety/regression!"

print("[SUCCESS] All database counts and safety rules validated successfully!")
print("[SUCCESS] All 7 tables created and populated successfully in fit_core.sqlite!")
conn.close()
print("=== Phase 1 Complete ===")

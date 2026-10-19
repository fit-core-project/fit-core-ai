import pandas as pd
import numpy as np
import os
import re
import sqlite3
from datetime import datetime
import sys

print("=== Phase 1. Starting build_constraints_db.py ===")

# Paths
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
fit_core_ai_dir = str(SCRIPT_DIR.parent)
db_path = os.path.join(fit_core_ai_dir, "fit_core.sqlite")

core_csv_candidates = [
    SCRIPT_DIR.parent / "data" / "provenance" / "v25_canonical_filter_core.csv",
    SCRIPT_DIR.parent / "v25_canonical_filter_core.csv",
    SCRIPT_DIR.parent.parent / "v25_canonical_filter_core.csv",
    SCRIPT_DIR.parent.parent.parent / "v25_canonical_filter_core.csv",
]
core_csv_path = next((str(path) for path in core_csv_candidates if path.exists()), None)

if fit_core_ai_dir not in sys.path:
    sys.path.append(fit_core_ai_dir)

from engines.equipment_policy import equipment_option_profiles
from engines.muscle_taxonomy import classify_exercise_muscle_taxonomy
from engines.prescription.metadata import classify_exercise_prescription_metadata
from engines.specialization_policy import (
    SPECIALIZATION_FOCUS_TARGETS,
    TRAINING_METHOD_GOALS,
    candidate_matches_specialization_focus,
    candidate_matches_training_method,
)

try:
    from database import engine
    use_sqlalchemy = True
    print("[INFO] Successfully imported database engine from database.py.")
except Exception as e:
    print(f"[WARNING] Failed to import database engine ({e}). Will use sqlite3 directly.")
    use_sqlalchemy = False

if not core_csv_path or not os.path.exists(core_csv_path):
    print("Error: Core CSV file not found. Tried:")
    for candidate in core_csv_candidates:
        print(f" - {candidate}")
    sys.exit(1)

# Read core data
df_core = pd.read_csv(core_csv_path, encoding='utf-8')
print(f"Loaded Core data size: {len(df_core)}")
source_equipment_by_id = {}
for _, source_row in df_core.iterrows():
    source_equipment = source_row.get("equipment")
    if pd.notna(source_equipment) and str(source_equipment).strip():
        source_equipment_by_id[str(source_row["v24_row_uid"])] = str(source_equipment).strip()

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


def compact_text(value):
    return clean_str(value).replace(" ", "").replace("_", "").replace("-", "").replace("/", "")


def build_display_name_ko(value):
    text = "" if pd.isna(value) else str(value).strip()
    text = re.sub(r"\s*\([^)]*\)", "", text)
    replacements = [
        ("리버스아님", ""),
        ("시니어 ", ""),
        ("Low-Impact ", ""),
        ("low-impact ", ""),
        ("푸쉬업", "푸시업"),
        ("프레스/스쿼트", "프레스"),
        ("머신 머신", "머신"),
        ("링 링", "링"),
        ("밴드 밴드", "밴드"),
        ("벤치 벤치", "벤치"),
        ("딥 딥스", "딥스"),
    ]
    for raw, replacement in replacements:
        text = text.replace(raw, replacement)
    words = [word for word in re.split(r"\s+", text) if word]
    collapsed = []
    for word in words:
        if not collapsed or collapsed[-1] != word:
            collapsed.append(word)
    return " ".join(collapsed).strip() or str(value).strip()


BASE_EXERCISE_PATTERNS = [
    ("루마니안 데드리프트", ["루마니안데드리프트", "romaniandeadlift", "rdl"]),
    ("데드리프트", ["데드리프트", "deadlift"]),
    ("불가리안 스플릿 스쿼트", ["불가리안스플릿스쿼트", "bulgariansplitsquat"]),
    ("스플릿 스쿼트", ["스플릿스쿼트", "splitsquat"]),
    # The source catalog appends "/스쿼트" to leg-press rows. Match the actual
    # equipment movement before the generic squat family.
    ("레그 프레스", ["레그프레스", "legpress"]),
    ("백 스쿼트", ["백스쿼트", "backsquat"]),
    ("핵 스쿼트", ["핵스쿼트", "hacksquat"]),
    ("브이 스쿼트", ["브이스쿼트", "vsquat"]),
    ("오버헤드 스쿼트", ["오버헤드스쿼트", "overheadsquat"]),
    ("스쿼트", ["스쿼트", "squat"]),
    ("리버스 런지", ["리버스런지", "reverselunge"]),
    ("포워드 런지", ["포워드런지", "forwardlunge"]),
    ("사이드 런지", ["사이드런지", "sidelunge"]),
    ("런지", ["런지", "lunge"]),
    ("스텝업", ["스텝업", "stepup"]),
    ("스텝다운", ["스텝다운", "stepdown"]),
    ("레그 익스텐션", ["레그익스텐션", "legextension"]),
    ("레그 컬", ["레그컬", "legcurl"]),
    ("힙 쓰러스트", ["힙쓰러스트", "hipthrust"]),
    ("글루트 브리지", ["글루트브리지", "glutebridge"]),
    ("힙 어브덕션", ["힙어브덕션", "hipabduction"]),
    ("힙 어덕션", ["힙어덕션", "hipadduction"]),
    ("카프 레이즈", ["카프레이즈", "calfraise"]),
    ("티비알리스 레이즈", ["티비알리스", "tibialis"]),
    ("벤치프레스", ["벤치프레스", "benchpress"]),
    ("체스트 프레스", ["체스트프레스", "chestpress"]),
    ("숄더 프레스", ["숄더프레스", "shoulderpress"]),
    ("오버헤드 프레스", ["오버헤드프레스", "overheadpress", "밀리터리프레스", "militarypress"]),
    ("JM 프레스", ["jm프레스", "jmpress"]),
    ("랫풀다운", ["랫풀다운", "latpulldown"]),
    ("어시스트 풀업", ["어시스트풀업", "보조어시스트풀업", "assistedpullup"]),
    ("풀업", ["풀업", "pullup", "pull-up", "친업", "chinup", "chin-up"]),
    ("하이 풀다운", ["하이케이블풀다운", "highcablepulldown"]),
    ("페이스 풀", ["페이스풀", "facepull"]),
    ("레터럴 레이즈", ["레터럴레이즈", "lateralraise"]),
    ("사이드라잉 외회전", ["사이드라잉외회전", "sidelyingexternalrotation"]),
    ("엎드린 T 레이즈", ["프론t레이즈", "pronetraise"]),
    ("프론트 레이즈", ["프론트레이즈", "frontraise"]),
    ("리버스 펙덱 플라이", ["리버스펙덱", "reversepecdeck"]),
    ("펙덱 플라이", ["펙덱", "pecdeck"]),
    ("리어 델트 플라이", ["리어델트플라이", "reardeltfly"]),
    ("플라이", ["플라이", "fly", "크로스오버", "crossover"]),
    ("시티드 로우", ["시티드로우", "seatedrow"]),
    ("케이블 로우", ["케이블로우", "cablerow"]),
    ("덤벨 로우", ["덤벨로우", "dumbbellrow"]),
    ("바벨 로우", ["바벨로우", "barbellrow"]),
    ("로우", ["로우", "row"]),
    ("트라이셉스 익스텐션", ["트라이셉스익스텐션", "tricepsextension"]),
    ("푸시다운", ["푸쉬다운", "푸시다운", "pushdown"]),
    ("킥백", ["킥백", "kickback"]),
    ("리스트 컬", ["리스트컬", "wristcurl", "손목컬"]),
    ("슈러그", ["슈러그", "shrug"]),
    ("컬", ["컬", "curl"]),
    ("딥스", ["딥스", "dip"]),
    ("푸시업", ["푸쉬업", "푸시업", "pushup"]),
    ("플랭크", ["플랭크", "plank"]),
    ("크런치", ["크런치", "crunch"]),
    ("백 익스텐션", ["백익스텐션", "backextension"]),
]


DEFAULT_VARIATION_PATTERNS = [
    "싱글레그", "singleleg", "단일다리", "교대", "발끝", "가동범위",
    "상체세움", "상체 세움", "전경사", "박스터치", "박스 터치", "템포",
    "정지", "내로우", "narrow", "와이드", "wide", "보폭", "고정식",
    "힐엘리베이티드", "한발앞", "싱글핸들", "싱글 핸들", "싱글암",
    "singlearm", "팔꿈치패드", "팔꿈치 패드", "발가까이", "발 가까이",
    "발멀리", "발 멀리", "느린", "이완", "1.5", "반복", "클로즈그립",
    "클로즈 그립", "와이드그립", "와이드 그립", "하이투로우", "하이 투 로우",
    "플러스", "플로어", "floor", "스플릿스탠스", "splitstance",
    "싱글사이드", "single-side", "정면싱글",
]


def derive_base_exercise_name_kr(name, english):
    compact = compact_text(f"{name} {english}")
    # Direction words in rotational cable lifts contain "low/로우" but are not
    # rowing movements. Preserve the detailed catalog name instead of reducing
    # them to the generic "케이블 로우" family.
    if (
        any(token in compact for token in ["로우투하이", "하이투로우", "lowtohigh", "hightolow"])
        and any(token in compact for token in ["리프트", "lift", "촙", "chop"])
    ):
        return build_display_name_ko(name)
    # Equipment-based variants such as "Leg Press Calf Raise" should keep the
    # actual movement family, not the machine/setup name.
    if any(compact_text(pattern) in compact for pattern in ["카프레이즈", "calfraise"]):
        return "카프 레이즈"
    if any(compact_text(pattern) in compact for pattern in ["티비알리스", "tibialis"]):
        return "티비알리스 레이즈"
    for base_name, patterns in BASE_EXERCISE_PATTERNS:
        if any(compact_text(pattern) in compact for pattern in patterns):
            return base_name
    return build_display_name_ko(name)


def derive_equipment_prefix(name, equipment):
    name_source = compact_text(name)
    equipment_source = compact_text(equipment)
    source = name_source or equipment_source
    if "스미스머신" in source or "smithmachine" in source:
        return "스미스 머신"
    if "레그프레스" in source or "legpress" in source:
        return ""
    if "핵스쿼트" in source or "hacksquat" in source:
        return ""
    if "브이스쿼트" in source or "vsquat" in source:
        return ""

    if "머신" in name_source or "machine" in name_source:
        return "머신"
    if "케이블" in name_source or "cable" in name_source:
        return "케이블"
    if "바벨" in name_source or "barbell" in name_source:
        return "바벨"
    if "덤벨" in name_source or "dumbbell" in name_source:
        return "덤벨"
    if "밴드" in name_source or "band" in name_source:
        return "밴드"
    if "케틀벨" in name_source or "kettlebell" in name_source:
        return "케틀벨"
    if "맨몸" in name_source or "bodyweight" in name_source:
        return "맨몸"

    # When the catalog says "bodyweight/dumbbell" but the exercise name does not
    # commit to one tool, keep the display name neutral. The actual equipment list
    # still remains available for scoring and UI details.
    has_mixed_bodyweight_option = (
        ("맨몸" in equipment_source or "bodyweight" in equipment_source)
        and any(token in equipment_source for token in ["덤벨", "dumbbell", "밴드", "band", "케틀벨", "kettlebell"])
    )
    if has_mixed_bodyweight_option:
        return ""

    if "머신" in equipment_source or "machine" in equipment_source:
        return "머신"
    if "케이블" in equipment_source or "cable" in equipment_source:
        return "케이블"
    if "바벨" in equipment_source or "barbell" in equipment_source:
        return "바벨"
    if "덤벨" in equipment_source or "dumbbell" in equipment_source:
        return "덤벨"
    if "밴드" in equipment_source or "band" in equipment_source:
        return "밴드"
    if "케틀벨" in equipment_source or "kettlebell" in equipment_source:
        return "케틀벨"
    return ""


def derive_user_facing_display_name_kr(name, english, equipment):
    base_name = derive_base_exercise_name_kr(name, english)
    compact = compact_text(f"{name} {english}")
    if base_name == "카프 레이즈" and any(compact_text(pattern) in compact for pattern in ["레그프레스", "legpress"]):
        return "레그 프레스 카프 레이즈"
    display_base_name = base_name
    if base_name in {"벤치프레스", "체스트 프레스"}:
        for patterns, label in (
            (("인클라인", "incline"), f"인클라인 {base_name}"),
            (("디클라인", "decline"), f"디클라인 {base_name}"),
            (("플랫", "flat"), f"플랫 {base_name}"),
        ):
            if any(compact_text(pattern) in compact for pattern in patterns):
                display_base_name = label
                break
    elif base_name == "리스트 컬":
        display_base_name = "리스트 컬"
    elif base_name == "컬":
        for patterns, label in (
            (("크로스바디해머", "crossbodyhammer"), "크로스바디 해머 컬"),
            (("해머", "hammer"), "해머 컬"),
            (("프리처", "preacher"), "프리처 컬"),
            (("드래그", "drag"), "드래그 컬"),
            (("스파이더", "spider"), "스파이더 컬"),
        ):
            if any(compact_text(pattern) in compact for pattern in patterns):
                display_base_name = label
                break
    elif base_name == "트라이셉스 익스텐션":
        for patterns, label in (
            (("오버헤드", "overhead"), "오버헤드 트라이셉스 익스텐션"),
            (("라잉", "lying"), "라잉 트라이셉스 익스텐션"),
        ):
            if any(compact_text(pattern) in compact for pattern in patterns):
                display_base_name = label
                break
    if base_name == "스쿼트":
        for patterns, label in (
            (("와이드", "wide"), "와이드 스쿼트"),
            (("스모", "sumo"), "스모 스쿼트"),
            (("고블릿", "goblet"), "고블릿 스쿼트"),
        ):
            if any(compact_text(pattern) in compact for pattern in patterns):
                display_base_name = label
                break
    prefix = derive_equipment_prefix(name, equipment)
    if prefix and not compact_text(display_base_name).startswith(compact_text(prefix)):
        return f"{prefix} {display_base_name}".strip()
    return display_base_name


def derive_variation_name_kr(name, english):
    display_name = build_display_name_ko(name)
    base_name = derive_base_exercise_name_kr(name, english)
    variation = display_name.replace(base_name, "").strip()
    variation = re.sub(r"\s+", " ", variation).strip()
    return variation


def is_default_variation_name(name, english):
    compact = compact_text(f"{name} {english}")
    return any(compact_text(pattern) in compact for pattern in DEFAULT_VARIATION_PATTERNS)


def corrected_primary_muscle_slug(name, english, current_slug):
    compact = compact_text(f"{name} {english}")
    current_slug = str(current_slug or "").strip()
    if any(token in compact for token in ["리스트컬", "wristcurl", "손목컬"]):
        return "forearm"
    if any(token in compact for token in ["슈러그", "shrug"]):
        return "trapezius"
    if any(token in compact for token in ["브레이싱", "bracing", "코어안정", "corestability"]):
        return "abs"
    if any(token in compact for token in ["티비알리스", "tibialis"]):
        return "calves"
    if any(token in compact for token in ["힙쓰러스트", "hipthrust", "글루트브리지", "glutebridge"]):
        return "gluteal"
    if "오버헤드스쿼트" in compact or "overheadsquat" in compact:
        return "quadriceps"
    if any(token in compact for token in ["플라이", "fly", "크로스오버", "crossover", "펙덱", "pecdeck"]):
        if any(token in compact for token in ["리버스", "리어", "후면", "reverse", "rear"]):
            return "back-deltoids"
        return "chest"
    if "레그컬" in compact or "legcurl" in compact:
        return "hamstring"
    if "카프레이즈" in compact or "calfraise" in compact:
        return "calves"
    if (
        "해머컬" in compact
        or "바이셉스컬" in compact
        or "bicepscurl" in compact
        or "hammercurl" in compact
        or (
            ("컬" in compact or "curl" in compact)
            and not any(token in compact for token in ["펠빅컬", "pelviccurl", "레그컬", "legcurl"])
        )
    ):
        return "biceps"
    if "플랭크" in compact or "plank" in compact or "크런치" in compact or "crunch" in compact:
        return "abs"
    if (
        ("스쿼트" in compact or "squat" in compact)
        and (current_slug == "hamstring" or "햄스트링" in current_slug)
    ):
        return "quadriceps"
    if "프로그펌프" in compact or "frogpump" in compact:
        return "gluteal"
    if "어브덕션" in compact or "abduction" in compact:
        return "abductors"
    if "어덕션" in compact or "adduction" in compact:
        return "adductor"
    if "로우" in compact or "row" in compact:
        if (
            current_slug == "front-deltoids"
            or "후면삼각" in current_slug
            or "rear" in current_slug
            or "back-deltoids" in current_slug
        ):
            return "back-deltoids" if "팔꿈치벌림" in compact else "upper-back"
        if any(token in current_slug for token in ["능형", "승모", "광배", "upper-back"]):
            return "upper-back"
    return ""


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
# 6. exercise_recommendation_metadata (추천 보편성/스타일 메타)
# ==========================================================
print("Building exercise_recommendation_metadata...")
recommendation_rows = []
# The 24k source catalog deliberately preserves micro-variations for search and
# substitution. Automatic routines use a separate system-curated representative
# layer:
# familiar public concepts with one stable source row per useful equipment form.
# A row not listed here remains available to the encyclopedia and explicit
# substitution flows, but cannot surprise users in a default gym routine.
# Human familiarity approval is tracked separately through review_required.
curated_default_exercise_ids = {
    # Lower body main and posterior chain
    "V24ROW-003053",  # leg press, centred bilateral stance
    "V24ROW-003477",  # hack squat, shoulder-width full range
    "V24ROW-003251",  # barbell back squat, shoulder-width full range
    "V24ROW-003367",  # Smith-machine back squat, shoulder-width full range
    "V24ROW-003110",  # dumbbell goblet squat, shoulder-width full range
    "V24ROW-023880",  # bodyweight hip thrust, release-cleared basic variation
    "V24ROW-023882",  # dumbbell hip thrust, release-cleared loaded variation
    "V24ROW-002758",  # bilateral neutral machine leg extension
    "V24ROW-003496",  # lying machine leg curl
    "V24ROW-003502",  # seated machine leg curl
    "V24ROW-003682",  # seated machine hip abduction, release-cleared representative
    "V24ROW-006763",  # machine hip abduction
    "V24ROW-004713",  # machine hip adduction
    "V24ROW-006846",  # standing machine calf raise
    "V24ROW-006847",  # seated machine calf raise
    "V24ROW-006860",  # standing dumbbell calf raise
    "V24ROW-006867",  # standing bodyweight calf raise
    # Chest and shoulder
    "V24ROW-005067",  # flat barbell chest press
    "V24ROW-005079",  # incline barbell chest press
    "V24ROW-005115",  # flat dumbbell chest press
    "V24ROW-005131",  # incline dumbbell chest press
    "V24ROW-005091",  # decline barbell chest press
    "V24ROW-005147",  # decline dumbbell chest press
    "V24ROW-005243",  # flat machine chest press
    "V24ROW-005259",  # incline machine chest press
    "V24ROW-005179",  # flat Smith-machine chest press
    "V24ROW-005195",  # incline Smith-machine chest press
    "V24ROW-005303",  # flat cable fly
    "V24ROW-000227",  # machine pec-deck fly
    "V24ROW-000478",  # dumbbell front raise
    "V24ROW-000666",  # standing dumbbell lateral raise
    "V24ROW-000702",  # seated machine lateral raise
    # Back and rear shoulder
    "V24ROW-005645",  # neutral-grip machine lat pulldown
    "V24ROW-005709",  # standard assisted pull-up
    "V24ROW-001379",  # neutral dumbbell row
    "V24ROW-005499",  # neutral cable row
    "V24ROW-005539",  # neutral machine row
    "V24ROW-000920",  # machine reverse pec deck
    "V24ROW-001016",  # standing rope face pull
    "V24ROW-002158",  # standing dumbbell shrug
    "V24ROW-002164",  # standing machine shrug
    # Direct arm work
    "V24ROW-006004",  # barbell preacher curl
    "V24ROW-006022",  # dumbbell preacher curl
    "V24ROW-006028",  # dumbbell hammer curl
    "V24ROW-006032",  # cable preacher curl
    "V24ROW-006042",  # machine preacher curl
    "V24ROW-006048",  # machine hammer curl
    "V24ROW-006120",  # dumbbell wrist curl
    "V24ROW-006129",  # barbell wrist curl
    "V24ROW-006140",  # cable wrist curl
    "V24ROW-006181",  # machine wrist curl
    "V24ROW-002614",  # dumbbell kickback, shoulder-flexion-neutral accessory
    "V24ROW-002639",  # lying dumbbell triceps extension, non-overhead accessory
    "V24ROW-006076",  # dumbbell overhead triceps extension
    "V24ROW-006083",  # cable rope pushdown
    "V24ROW-006093",  # machine rope pushdown
    # Core
    "V24ROW-006891",  # bodyweight crunch
    "V24ROW-006892",  # bodyweight reverse crunch
    "V24ROW-006895",  # bodyweight plank
    "V24ROW-006898",  # bodyweight dead bug
    "V24ROW-006899",  # bodyweight bird dog
    "V24ROW-006904",  # cable crunch
}

reviewed_beginner_excluded_exercise_ids = {
    "V24ROW-003251",
    "V24ROW-005067",
    "V24ROW-005079",
}

mainstream_patterns = [
    "벤치프레스", "벤치 프레스", "benchpress", "bench press",
    "체스트프레스", "체스트 프레스", "chestpress", "chest press",
    "숄더프레스", "숄더 프레스", "shoulderpress", "shoulder press",
    "오버헤드프레스", "오버헤드 프레스", "overheadpress", "overhead press",
    "랫풀다운", "랫 풀다운", "latpulldown", "lat pull",
    "시티드로우", "시티드 로우", "seatedrow", "seated row",
    "케이블로우", "케이블 로우", "cablerow", "cable row",
    "덤벨로우", "덤벨 로우", "dumbbellrow", "dumbbell row",
    "바벨로우", "바벨 로우", "barbellrow", "barbell row",
    "풀업", "pullup", "pull-up", "친업", "chinup", "chin-up",
    "어시스트풀업", "어시스트 풀업", "assistedpullup", "assisted pull-up",
    "레그프레스", "레그 프레스", "legpress", "leg press",
    "레그익스텐션", "레그 익스텐션", "legextension", "leg extension",
    "레그컬", "레그 컬", "legcurl", "leg curl",
    "데드리프트", "deadlift", "루마니안데드리프트", "루마니안 데드리프트",
    "romaniandeadlift", "romanian deadlift", "rdl",
    "스쿼트", "squat", "런지", "lunge",
    "힙쓰러스트", "힙 쓰러스트", "hipthrust", "hip thrust",
    "글루트브리지", "글루트 브리지", "glutebridge", "glute bridge",
    "힙어브덕션", "힙 어브덕션", "hipabduction", "hip abduction",
    "힙어덕션", "힙 어덕션", "hipadduction", "hip adduction",
    "레터럴레이즈", "레터럴 레이즈", "lateralraise", "lateral raise",
    "프론트레이즈", "프론트 레이즈", "frontraise", "front raise",
    "푸쉬다운", "푸시다운", "pushdown",
    "트라이셉스익스텐션", "트라이셉스 익스텐션",
    "tricepsextension", "triceps extension",
    "킥백", "kickback",
    "바벨컬", "바벨 컬", "barbellcurl", "barbell curl",
    "덤벨컬", "덤벨 컬", "dumbbellcurl", "dumbbell curl",
    "케이블컬", "케이블 컬", "cablecurl", "cable curl",
    "머신컬", "머신 컬", "machinecurl", "machine curl",
    "해머컬", "해머 컬", "hammercurl", "hammer curl",
    "프리처컬", "프리처 컬", "preachercurl", "preacher curl",
    "드래그컬", "드래그 컬", "dragcurl", "drag curl",
    "바이셉스컬", "바이셉스 컬", "bicepscurl", "biceps curl",
    "페이스풀", "페이스 풀", "facepull", "face pull",
    "펙덱", "pecdeck", "pec deck", "플라이", "fly",
    "리어델트플라이", "리어 델트 플라이", "reardeltfly", "rear delt fly",
    "리버스펙덱", "리버스 펙덱", "reversepecdeck", "reverse pec deck",
    "카프레이즈", "카프 레이즈", "calfraise", "calf raise",
    "플랭크", "plank", "크런치", "crunch",
    "백익스텐션", "백 익스텐션", "backextension", "back extension",
]
uncommon_patterns = [
    "링", "ring", "trx", "스내치", "snatch", "클린", "clean", "저크", "jerk",
    "스러스터", "thruster", "월볼", "wallball", "키핑", "kipping",
    "피터슨", "peterson", "시시", "sissy", "제퍼슨", "jefferson",
    "코사크", "cossack", "에어플레인", "airplane", "아처", "archer",
    "싱글레그", "singleleg", "단일다리", "교대", "가동범위", "발끝",
    "전경사", "상체세움", "상체 세움", "박스터치", "박스 터치",
    "워리어", "warrior", "미니볼", "mini ball", "소도구",
    "펠빅컬", "pelvic curl", "정렬", "alignment", "요가", "yoga",
    "필라테스", "pilates",
    "템포", "정지", "내로우", "narrow",
    "역도 드릴", "olympic weightlifting drill",
    "스캡", "scapular", "백팩", "backpack", "홈트", "home training",
    "소파", "sofa", "등산", "하이킹", "hiking", "테니스", "tennis",
    "보강", "conditioning drill",
]
crossfit_patterns = [
    "crossfit", "크로스핏", "wod", "amrap", "emom", "스내치", "snatch",
    "클린", "clean", "저크", "jerk", "스러스터", "thruster", "월볼", "wallball",
    "키핑", "kipping", "버피", "burpee",
]
athletic_patterns = [
    "점프", "jump", "착지", "landing", "스프린트", "sprint", "스킵", "skip",
    "파워", "power", "보강", "agility", "plyo", "플라이오",
]
mobility_patterns = [
    "모빌리티", "mobility", "가동성", "range of motion", "가동범위",
    "정렬", "alignment", "스트레칭", "stretch", "폼롤러", "foam",
]
yoga_pilates_patterns = [
    "요가", "yoga", "필라테스", "pilates", "워리어", "warrior",
    "미니볼", "mini ball", "소도구", "펠빅컬", "pelvic curl",
]
rehab_patterns = [
    "재활", "프리햅", "prehab", "rehab", "회전근개", "rotator", "시니어", "senior",
    "low-impact", "lowimpact",
]
cardio_patterns = [
    "러닝머신", "걷기", "워킹", "트레드밀", "treadmill", "walking", "walk",
    "러닝", "running", "유산소", "cardio",
]

for idx, row in df_core.iterrows():
    eid = row['v24_row_uid']
    name = clean_str(row['name'])
    english = clean_str(row.get('english', ''))
    pattern = clean_str(row['pattern'])
    equipment = clean_str(row['equipment'])
    difficulty = clean_str(row['difficulty'])
    release_tier = clean_str(row['v24_release_tier_norm'])
    quality_score = int(row.get('v24_preprocess_quality_score') or 0)
    text = " ".join([name, english, pattern, equipment])
    compact = compact_text(text)

    def text_has(patterns):
        return any(compact_text(item) in compact for item in patterns)

    is_mainstream = text_has(mainstream_patterns) or eid in curated_default_exercise_ids
    is_uncommon = text_has(uncommon_patterns) and eid not in curated_default_exercise_ids
    is_crossfit = text_has(crossfit_patterns)
    is_cardio = text_has(cardio_patterns)
    is_athletic = text_has(athletic_patterns) or is_cardio
    is_mobility = text_has(mobility_patterns)
    is_yoga_pilates = text_has(yoga_pilates_patterns)
    is_rehab = text_has(rehab_patterns)
    is_home_specific = text_has([
        "홈트", "home training", "백팩", "backpack", "소파", "sofa",
    ])
    is_accessory_only_default_excluded = text_has([
        "티비알리스", "tibialis", "브레이싱", "bracing",
    ])
    is_calisthenics = any(k in equipment for k in ["링", "맨몸", "평행봉"]) or text_has(["ring", "bodyweight", "calisthenics", "parallel bar"])
    is_loadable = any(k in equipment for k in ["머신", "케이블", "바벨", "덤벨", "스미스", "해머", "플레이트", "랜드마인"])
    is_home_friendly = any(k in equipment for k in ["맨몸", "덤벨", "밴드", "케틀벨"])

    mainstream_score = 45
    if is_mainstream:
        mainstream_score += 28
    if is_loadable:
        mainstream_score += 10
    if release_tier == "gold":
        mainstream_score += 8
    elif release_tier == "silver":
        mainstream_score += 4
    elif release_tier in {"hold", "bronze"}:
        mainstream_score -= 8
    if quality_score >= 90:
        mainstream_score += 4
    elif quality_score < 75:
        mainstream_score -= 10
    if difficulty in {"초급", "입문"}:
        mainstream_score += 8
    elif difficulty == "고급":
        mainstream_score -= 12
    if is_uncommon:
        mainstream_score -= 35
    if is_crossfit:
        mainstream_score -= 18
    if is_athletic and not is_mainstream:
        mainstream_score -= 12
    if is_rehab:
        mainstream_score -= 18
    if is_cardio:
        mainstream_score -= 20
    mainstream_score = max(0, min(100, mainstream_score))
    if is_uncommon:
        mainstream_score = min(mainstream_score, 39)

    if mainstream_score >= 80:
        familiarity_tier = "common"
    elif mainstream_score >= 60:
        familiarity_tier = "standard"
    elif mainstream_score >= 40:
        familiarity_tier = "specialized"
    else:
        familiarity_tier = "obscure"

    if familiarity_tier in {"common", "standard"} and not is_uncommon:
        novelty_level = "low"
    elif familiarity_tier == "specialized":
        novelty_level = "medium"
    else:
        novelty_level = "high"

    base_exercise_name_kr = derive_base_exercise_name_kr(name, english)
    display_name_ko = build_display_name_ko(row['name'])
    user_facing_display_name_kr = derive_user_facing_display_name_kr(name, english, equipment)
    variation_name_kr = derive_variation_name_kr(name, english)
    is_default_variation = is_default_variation_name(name, english)

    style_tags = set()
    is_style_specific_non_gym = (
        is_rehab
        or is_mobility
        or is_yoga_pilates
        or is_crossfit
        or is_cardio
        or is_home_specific
    )
    if (is_mainstream and not is_style_specific_non_gym) or (is_loadable and not is_style_specific_non_gym):
        style_tags.add("gym")
        style_tags.add("bodybuilding")
    if is_crossfit:
        style_tags.add("crossfit")
    if is_athletic:
        style_tags.add("athletic")
    if is_mobility:
        style_tags.add("mobility")
    if is_yoga_pilates:
        style_tags.add("yoga_pilates")
    if is_rehab:
        style_tags.add("rehab_prehab")
    if is_cardio:
        style_tags.add("cardio")
    if is_home_friendly:
        style_tags.add("home")
    if is_home_specific:
        style_tags.add("home_specific")
    if is_calisthenics and not is_mainstream:
        style_tags.add("calisthenics")
    if not style_tags:
        style_tags.add("general")

    user_facing_priority_score = mainstream_score
    if is_rehab:
        user_facing_priority_score -= 8
    if is_cardio:
        user_facing_priority_score -= 12
    if release_tier == "hold":
        user_facing_priority_score -= 12
    user_facing_priority_score = max(0, min(100, user_facing_priority_score))

    main_routine_default_allowed = int(
        eid in curated_default_exercise_ids
        and is_mainstream
        and familiarity_tier in {"common", "standard"}
        and "gym" in style_tags
        and not is_uncommon
        and not is_crossfit
        and not is_cardio
        and not is_rehab
        and not is_home_specific
        and not is_accessory_only_default_excluded
        and (not is_default_variation or eid in curated_default_exercise_ids)
        and release_tier != "hold"
    )
    beginner_default_allowed = int(
        main_routine_default_allowed
        and eid not in reviewed_beginner_excluded_exercise_ids
        and difficulty not in {"고급"}
        and not is_uncommon
    )

    recommendation_rows.append({
        "exercise_id": eid,
        "display_name_ko": display_name_ko,
        "base_exercise_name_kr": base_exercise_name_kr,
        "variation_name_kr": variation_name_kr,
        "user_facing_display_name_kr": user_facing_display_name_kr,
        "corrected_primary_muscle": corrected_primary_muscle_slug(row['name'], row.get('english', ''), row.get('primary_muscle', '')),
        "mainstream_gym_score": mainstream_score,
        "exercise_familiarity_tier": familiarity_tier,
        "commonness_level": familiarity_tier,
        "novelty_level": novelty_level,
        "training_style_tags": ",".join(sorted(style_tags)),
        "user_facing_priority_score": user_facing_priority_score,
        "beginner_default_allowed": beginner_default_allowed,
        "main_routine_default_allowed": main_routine_default_allowed,
        "source_type": "generated",
        "confidence": "medium",
        "review_required": 1,
        "updated_at": current_time_str
    })

df_recommendation = pd.DataFrame(recommendation_rows)

# ==========================================================
# 7. exercise_condition_policy (수술/질환 정책 테이블 - Sparse Many-to-Many)
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
# 8. exercise_domain_policy_evidence
#    운동 도메인 판단 근거/검수 큐 테이블
# ==========================================================
print("Building exercise_domain_policy_evidence...")
domain_policy_evidence_rows = [
    {
        "policy_id": "lower-back-pain-heavy-barbell-screen-v1",
        "condition_code": "lower_back_pain",
        "body_part": "lower-back",
        "exercise_id": None,
        "target_pattern": "heavy_barbell_lift",
        "policy_action": "caution",
        "policy_reason_ko": "허리 통증 문맥에서는 백스쿼트/밀리터리 프레스처럼 척추 수직 압박이 큰 운동과 데드리프트/바벨 로우처럼 요추 전단 및 기립근 모멘트가 큰 고중량 바벨 리프트를 함께 스크리닝한다.",
        "source_title": "NHS inform - Exercises for back pain",
        "source_url": "https://www.nhsinform.scot/illnesses-and-conditions/muscle-bone-and-joints/neck-and-back-problems-and-conditions/exercises-for-back-pain/",
        "source_summary_ko": "허리 통증 운동은 통증 반응을 관찰하며 조절해야 하고, 통증이 크게 증가하거나 지속되면 중단/상담이 필요하다는 보수적 기준을 제공한다.",
        "evidence_grade": "official",
        "review_status": "needsExpertReview",
        "expert_review_required": 1,
        "expert_question_ko": "요추 통증 사용자의 일반 루틴에서 스쿼트/데드리프트/바벨 로우 계열을 hard filter가 아니라 penalty+caution으로 둘 수 있는 조건은 어디까지인가?",
        "priority": 10,
        "updated_at": current_time_str
    },
    {
        "policy_id": "lower-back-pain-supported-leg-prefer-v1",
        "condition_code": "lower_back_pain",
        "body_part": "lower-back",
        "exercise_id": None,
        "target_pattern": "supported_leg_alternative",
        "policy_action": "prefer",
        "policy_reason_ko": "허리 부담을 낮춰야 하는 하체 루틴에서는 등받이/기구 지지 기반의 레그 프레스, 낮은 스텝업, 브릿지, 레그 컬 계열을 우선 후보로 올린다.",
        "source_title": "NASM - Squat Alternatives: Bad Knees, Back or Hips?",
        "source_url": "https://www.nasm.org/resource-center/blog/squat-alternatives",
        "source_summary_ko": "등/무릎/고관절 이슈가 있는 사용자에게 스쿼트 대체로 머신 레그 프레스, 낮은 스텝업, 플로어 브리지, 레그 컬 등을 제시한다.",
        "evidence_grade": "professional",
        "review_status": "needsExpertReview",
        "expert_review_required": 1,
        "expert_question_ko": "요추 통증 legs 루틴에서 레그프레스/레그컬/힙쓰러스트 계열을 기본 우선 후보로 두는 정책이 적절한가?",
        "priority": 20,
        "updated_at": current_time_str
    },
    {
        "policy_id": "front-deltoid-pain-push-caution-v1",
        "condition_code": "front_deltoid_pain",
        "body_part": "front-deltoids",
        "exercise_id": None,
        "target_pattern": "push_pressing",
        "policy_action": "caution",
        "policy_reason_ko": "어깨 전면 통증이 있는 push 루틴에서는 프레스 운동을 무조건 안전하다고 표현하지 않고, 서버 필터를 통과한 후보라도 통증 반응에 따라 중단/교체할 수 있다는 주의 문구를 붙인다.",
        "source_title": "The Prehab Guys - Bench Press Without Shoulder Pain",
        "source_url": "https://landing.theprehabguys.com/bench-press-without-shoulder-pain/",
        "source_summary_ko": "벤치프레스 중 어깨 통증은 자세/가동범위/견갑 안정성 등 조절 요소가 있으며, 통증이 있으면 수정과 주의가 필요하다는 코칭 관점을 제공한다.",
        "evidence_grade": "professional",
        "review_status": "needsExpertReview",
        "expert_review_required": 1,
        "expert_question_ko": "어깨 전면 통증 push 루틴에서 안전 후보가 부족하면 운동 수를 줄이고 경고를 띄우는 정책이 맞는가?",
        "priority": 30,
        "updated_at": current_time_str
    },
    {
        "policy_id": "beginner-lower-tibialis-accessory-v1",
        "condition_code": "beginner_lower_routine",
        "body_part": "lower-leg",
        "exercise_id": None,
        "target_pattern": "tibialis_raise",
        "policy_action": "accessoryOnly",
        "policy_reason_ko": "티비알리스 레이즈는 하체 메인 리프트라기보다 정강이/발목 보강용 액세서리 성격이 강하므로 초보자 하체 기본 루틴에서는 메인 운동보다 보조 또는 재활/프리햅 후보로 둔다.",
        "source_title": "Hinge Health - Tibialis Raises",
        "source_url": "https://www.hingehealth.com/resources/articles/tibialis-raises/",
        "source_summary_ko": "티비알리스 레이즈는 정강이 앞쪽 근육과 발목 움직임을 보강하는 운동으로 설명되며, 하체 대근육 메인 볼륨 운동과는 성격이 다르다.",
        "evidence_grade": "professional",
        "review_status": "needsExpertReview",
        "expert_review_required": 1,
        "expert_question_ko": "초보자 legs 루틴에서 머신 티비알리스 레이즈를 기본 후보가 아니라 액세서리/프리햅 후보로 낮추는 정책이 적절한가?",
        "priority": 40,
        "updated_at": current_time_str
    },
]
df_domain_policy_evidence = pd.DataFrame(domain_policy_evidence_rows)

# ==========================================================
# SQLite DB에 테이블 생성 및 적재 진행
# ==========================================================
print(f"Connecting to SQLite database: {db_path}")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

equipment_option_rows = []
for exercise_id, equipment_req, name_kr, name_en, detail_subpart in cursor.execute(
    """
    SELECT id, equipment_req, name_kr, name_en, detail_subpart
    FROM exercise_tier
    ORDER BY id
    """
).fetchall():
    equipment_option_rows.extend(
        equipment_option_profiles(
            str(exercise_id),
            equipment_req,
            name_kr=name_kr,
            name_en=name_en,
            detail_subpart=detail_subpart,
            source_equipment=source_equipment_by_id.get(str(exercise_id)),
        )
    )
df_equipment_options = pd.DataFrame(equipment_option_rows)

recommendation_primary_by_id = {
    str(row["exercise_id"]): str(row.get("corrected_primary_muscle") or "").strip()
    for row in recommendation_rows
}
recommendation_by_id = {
    str(row["exercise_id"]): row
    for row in recommendation_rows
}
muscle_taxonomy_rows = []
for row in cursor.execute("""
    SELECT id, name_kr, name_en, primary_muscle, secondary_muscle,
           pain_triggers, detail_subpart
    FROM exercise_tier
    ORDER BY id
""").fetchall():
    (
        exercise_id,
        name_kr,
        name_en,
        source_primary,
        source_secondary,
        source_pain,
        detail_subpart,
    ) = row
    decision = classify_exercise_muscle_taxonomy(
        name_kr=name_kr,
        name_en=name_en,
        source_primary_muscle=source_primary,
        source_secondary_muscles=source_secondary,
        source_pain_triggers=source_pain,
        detail_subpart=detail_subpart,
    ).to_dict()
    existing_correction = recommendation_primary_by_id.get(str(exercise_id), "")
    if (
        decision["classification_rule_code"] == "source_primary_retained"
        and existing_correction
        and existing_correction != decision["source_primary_muscle"]
    ):
        source_primary = decision["source_primary_muscle"]
        decision["canonical_primary_muscle"] = existing_correction
        decision["canonical_secondary_muscles"] = ", ".join(
            existing_correction if token.strip() == source_primary else token.strip()
            for token in decision["source_secondary_muscles"].split(",")
            if token.strip()
        )
        decision["canonical_pain_triggers"] = ", ".join(
            existing_correction if token.strip() == source_primary else token.strip()
            for token in decision["source_pain_triggers"].split(",")
            if token.strip()
        )
        decision["classification_rule_code"] = "existing_deterministic_correction"
        decision["source_type"] = "deterministicExistingCorrectionRule"
        decision["confidence"] = "high"
        decision["review_status"] = "autoClassified"
        decision["review_required"] = 0
    muscle_taxonomy_rows.append({"exercise_id": str(exercise_id), **decision})
df_muscle_taxonomy = pd.DataFrame(muscle_taxonomy_rows)

prescription_metadata_rows = []
for exercise_id, name_kr, name_en, movement_type in cursor.execute("""
    SELECT id, name_kr, name_en, movement_type
    FROM exercise_tier
    ORDER BY id
""").fetchall():
    decision = classify_exercise_prescription_metadata(
        name_kr=name_kr,
        name_en=name_en,
        source_movement_type=movement_type,
    ).to_dict()
    prescription_metadata_rows.append({"exercise_id": str(exercise_id), **decision})
df_prescription_metadata = pd.DataFrame(prescription_metadata_rows)

taxonomy_by_id = {
    str(row["exercise_id"]): row
    for row in muscle_taxonomy_rows
}
prescription_by_id = {
    str(row["exercise_id"]): row
    for row in prescription_metadata_rows
}
specialization_focus_rows = []
training_method_compatibility_rows = []
for row in cursor.execute("""
    SELECT id, name_kr, name_en, primary_muscle, secondary_muscle,
           equipment_req, movement_type, detail_subpart
    FROM exercise_tier
    ORDER BY id
""").fetchall():
    (
        exercise_id,
        name_kr,
        name_en,
        source_primary,
        secondary_muscle,
        equipment_req,
        movement_type,
        detail_subpart,
    ) = row
    exercise_id = str(exercise_id)
    taxonomy = taxonomy_by_id.get(exercise_id, {})
    prescription = prescription_by_id.get(exercise_id, {})
    candidate = {
        "id": exercise_id,
        "name_kr": name_kr,
        "name_en": name_en,
        "primary_muscle": (
            taxonomy.get("canonical_primary_muscle")
            or source_primary
        ),
        "secondary_muscle": secondary_muscle,
        "equipment_req": equipment_req,
        "movement_type": movement_type,
        "canonical_movement_type": prescription.get("canonical_movement_type"),
        "detail_subpart": detail_subpart,
    }
    candidate.update(recommendation_by_id.get(exercise_id, {}))
    candidate.update(prescription)
    for focus_code in SPECIALIZATION_FOCUS_TARGETS:
        if candidate_matches_specialization_focus(candidate, focus_code):
            specialization_focus_rows.append({
                "exercise_id": exercise_id,
                "focus_code": focus_code,
                "applicability": "direct",
                "source_type": "deterministicCatalogRule",
                "confidence": "high",
                "review_required": 0,
                "version": "specialization-focus-map-v1",
            })
    for method_code in TRAINING_METHOD_GOALS:
        if candidate_matches_training_method(candidate, method_code):
            training_method_compatibility_rows.append({
                "exercise_id": exercise_id,
                "method_code": method_code,
                "compatibility": "allowed",
                "source_type": "deterministicCuratedGymCompatibilityRule",
                "confidence": "high",
                "review_required": 0,
                "version": "training-method-compatibility-v2",
            })

df_specialization_focus = pd.DataFrame(specialization_focus_rows)
df_training_method_compatibility = pd.DataFrame(
    training_method_compatibility_rows
)

# Enable FK support
cursor.execute("PRAGMA foreign_keys = ON;")

# SQLite에서 FK 대상이 되려면 parent key가 PK 혹은 UNIQUE 인덱스를 가지고 있어야 하므로 선제 생성
cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_exercise_tier_id ON exercise_tier(id);")

# DDL Create Statements
ddl_statements = [
    # 0. exercise_equipment_option_v2
    """
    CREATE TABLE IF NOT EXISTS exercise_equipment_option_v2 (
        exercise_id TEXT NOT NULL,
        equipment_code TEXT NOT NULL CHECK(equipment_code IN ('BAND', 'BODYWEIGHT', 'KETTLEBELL', 'DUMBBELL', 'MACHINE', 'BARBELL', 'CABLE', 'LANDMINE', 'SMITH_MACHINE', 'PLATE', 'UNKNOWN')),
        equipment_category TEXT NOT NULL CHECK(equipment_category IN ('bodyweight', 'resistance_band', 'free_weight', 'cable_machine', 'machine', 'unknown')),
        station_class TEXT NOT NULL CHECK(station_class IN ('open_floor', 'free_weight_area', 'barbell_station', 'cable_station', 'landmine_station', 'generic_machine', 'smith_station', 'unknown')),
        portability TEXT NOT NULL CHECK(portability IN ('portable', 'semi_portable', 'fixed', 'unknown')),
        setup_transition_class TEXT NOT NULL CHECK(setup_transition_class IN ('quick', 'moderate', 'slow', 'unknown')),
        source_type TEXT NOT NULL,
        confidence TEXT NOT NULL CHECK(confidence IN ('low', 'medium', 'high')),
        review_required INTEGER NOT NULL CHECK(review_required IN (0, 1)),
        version TEXT NOT NULL,
        PRIMARY KEY (exercise_id, equipment_code),
        FOREIGN KEY (exercise_id) REFERENCES exercise_tier(id) ON DELETE CASCADE
    );
    """,
    # 0a. exercise_muscle_taxonomy_v2
    """
    CREATE TABLE IF NOT EXISTS exercise_muscle_taxonomy_v2 (
        exercise_id TEXT PRIMARY KEY NOT NULL,
        source_primary_muscle TEXT NOT NULL,
        canonical_primary_muscle TEXT NOT NULL,
        source_secondary_muscles TEXT NOT NULL,
        canonical_secondary_muscles TEXT NOT NULL,
        source_pain_triggers TEXT NOT NULL,
        canonical_pain_triggers TEXT NOT NULL,
        classification_rule_code TEXT NOT NULL,
        taxonomy_version TEXT NOT NULL,
        source_type TEXT NOT NULL,
        confidence TEXT NOT NULL CHECK(confidence IN ('low', 'medium', 'high')),
        review_status TEXT NOT NULL CHECK(review_status IN ('sourceRetained', 'autoClassified', 'reviewRequired')),
        review_required INTEGER NOT NULL CHECK(review_required IN (0, 1)),
        FOREIGN KEY (exercise_id) REFERENCES exercise_tier(id) ON DELETE CASCADE
    );
    """,
    # 0b. exercise_prescription_metadata_v2
    """
    CREATE TABLE IF NOT EXISTS exercise_prescription_metadata_v2 (
        exercise_id TEXT PRIMARY KEY NOT NULL,
        source_movement_type TEXT NOT NULL,
        canonical_movement_type TEXT NOT NULL,
        prescription_mode TEXT NOT NULL CHECK(prescription_mode IN ('reps', 'duration', 'interval', 'distance')),
        preferred_session_phase TEXT NOT NULL,
        allowed_session_phases TEXT NOT NULL,
        default_duration_sec INTEGER,
        default_distance_m INTEGER,
        default_interval_work_sec INTEGER,
        default_interval_recovery_sec INTEGER,
        default_interval_rounds INTEGER,
        default_rest_sec INTEGER,
        classification_rule_code TEXT NOT NULL,
        metadata_version TEXT NOT NULL,
        source_type TEXT NOT NULL,
        confidence TEXT NOT NULL CHECK(confidence IN ('low', 'medium', 'high')),
        review_status TEXT NOT NULL CHECK(review_status IN ('sourceDerived', 'autoClassified', 'reviewRequired')),
        review_required INTEGER NOT NULL CHECK(review_required IN (0, 1)),
        FOREIGN KEY (exercise_id) REFERENCES exercise_tier(id) ON DELETE CASCADE
    );
    """,
    # 0c. exercise_specialization_focus_map_v1
    """
    CREATE TABLE IF NOT EXISTS exercise_specialization_focus_map_v1 (
        exercise_id TEXT NOT NULL,
        focus_code TEXT NOT NULL,
        applicability TEXT NOT NULL CHECK(applicability IN ('direct')),
        source_type TEXT NOT NULL,
        confidence TEXT NOT NULL CHECK(confidence IN ('low', 'medium', 'high')),
        review_required INTEGER NOT NULL CHECK(review_required IN (0, 1)),
        version TEXT NOT NULL,
        PRIMARY KEY (exercise_id, focus_code),
        FOREIGN KEY (exercise_id) REFERENCES exercise_tier(id) ON DELETE CASCADE
    );
    """,
    # 0d. exercise_training_method_compatibility_v1
    """
    CREATE TABLE IF NOT EXISTS exercise_training_method_compatibility_v1 (
        exercise_id TEXT NOT NULL,
        method_code TEXT NOT NULL CHECK(method_code IN ('pausedReps', 'tempoEccentric')),
        compatibility TEXT NOT NULL CHECK(compatibility IN ('allowed')),
        source_type TEXT NOT NULL,
        confidence TEXT NOT NULL CHECK(confidence IN ('low', 'medium', 'high')),
        review_required INTEGER NOT NULL CHECK(review_required IN (0, 1)),
        version TEXT NOT NULL,
        PRIMARY KEY (exercise_id, method_code),
        FOREIGN KEY (exercise_id) REFERENCES exercise_tier(id) ON DELETE CASCADE
    );
    """,
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
    # 6. exercise_recommendation_metadata
    """
    CREATE TABLE IF NOT EXISTS exercise_recommendation_metadata (
        exercise_id TEXT PRIMARY KEY,
        display_name_ko TEXT,
        base_exercise_name_kr TEXT,
        variation_name_kr TEXT,
        user_facing_display_name_kr TEXT,
        corrected_primary_muscle TEXT,
        mainstream_gym_score INTEGER,
        exercise_familiarity_tier TEXT CHECK(exercise_familiarity_tier IN ('common', 'standard', 'specialized', 'obscure')),
        commonness_level TEXT CHECK(commonness_level IN ('common', 'standard', 'specialized', 'obscure')),
        novelty_level TEXT CHECK(novelty_level IN ('low', 'medium', 'high')),
        training_style_tags TEXT,
        user_facing_priority_score INTEGER,
        beginner_default_allowed INTEGER,
        main_routine_default_allowed INTEGER,
        source_type TEXT,
        confidence TEXT,
        review_required INTEGER,
        updated_at TEXT,
        FOREIGN KEY (exercise_id) REFERENCES exercise_tier(id) ON DELETE CASCADE
    );
    """,
    # 7. exercise_condition_policy
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
    # 8. exercise_domain_policy_evidence
    """
    CREATE TABLE IF NOT EXISTS exercise_domain_policy_evidence (
        policy_id TEXT PRIMARY KEY NOT NULL,
        condition_code TEXT NOT NULL,
        body_part TEXT,
        exercise_id TEXT,
        target_pattern TEXT,
        policy_action TEXT NOT NULL CHECK(policy_action IN ('avoid', 'caution', 'penalty', 'prefer', 'boost', 'accessoryOnly', 'reviewOnly')),
        policy_reason_ko TEXT NOT NULL,
        source_title TEXT NOT NULL,
        source_url TEXT NOT NULL,
        source_summary_ko TEXT NOT NULL,
        evidence_grade TEXT NOT NULL CHECK(evidence_grade IN ('official', 'professional', 'clinicalReview', 'domainExpert', 'internalHeuristic')),
        review_status TEXT NOT NULL CHECK(review_status IN ('draft', 'needsExpertReview', 'approved', 'rejected', 'superseded')),
        expert_review_required INTEGER NOT NULL DEFAULT 1,
        expert_question_ko TEXT,
        priority INTEGER,
        updated_at TEXT,
        FOREIGN KEY (exercise_id) REFERENCES exercise_tier(id) ON DELETE CASCADE
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
    "exercise_equipment_option_v2",
    "exercise_muscle_taxonomy_v2",
    "exercise_prescription_metadata_v2",
    "exercise_specialization_focus_map_v1",
    "exercise_training_method_compatibility_v1",
    "exercise_mobility_requirement",
    "exercise_joint_load_profile",
    "exercise_effect_profile",
    "exercise_difficulty_profile",
    "exercise_anthropometry_sensitivity_profile",
    "exercise_recommendation_metadata",
    "exercise_condition_policy",
    "exercise_regression_progression_map",
    "exercise_domain_policy_evidence",
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
    ("exercise_equipment_option_v2", df_equipment_options),
    ("exercise_muscle_taxonomy_v2", df_muscle_taxonomy),
    ("exercise_prescription_metadata_v2", df_prescription_metadata),
    ("exercise_specialization_focus_map_v1", df_specialization_focus),
    (
        "exercise_training_method_compatibility_v1",
        df_training_method_compatibility,
    ),
    ("exercise_mobility_requirement", df_mobility),
    ("exercise_joint_load_profile", df_load),
    ("exercise_effect_profile", df_effect),
    ("exercise_difficulty_profile", df_diff),
    ("exercise_anthropometry_sensitivity_profile", df_anthro),
    ("exercise_recommendation_metadata", df_recommendation),
    ("exercise_condition_policy", df_policy),
    ("exercise_regression_progression_map", df_map),
    ("exercise_domain_policy_evidence", df_domain_policy_evidence)
]

for table_name, df in dfs_to_load:
    print(f"Loading {len(df)} rows into {table_name}...")
    cursor.execute(f"DELETE FROM {table_name};")
    df.to_sql(table_name, conn, if_exists='append', index=False)
    conn.commit()

equipment_coverage = cursor.execute("""
    SELECT COUNT(DISTINCT exercise_id)
    FROM exercise_equipment_option_v2
""").fetchone()[0]
exercise_count = cursor.execute("SELECT COUNT(*) FROM exercise_tier").fetchone()[0]
assert equipment_coverage == exercise_count, (
    f"Equipment option coverage mismatch: {equipment_coverage}/{exercise_count}"
)
taxonomy_coverage = cursor.execute("""
    SELECT COUNT(DISTINCT exercise_id)
    FROM exercise_muscle_taxonomy_v2
""").fetchone()[0]
assert taxonomy_coverage == exercise_count, (
    f"Muscle taxonomy coverage mismatch: {taxonomy_coverage}/{exercise_count}"
)
prescription_coverage = cursor.execute("""
    SELECT COUNT(DISTINCT exercise_id)
    FROM exercise_prescription_metadata_v2
""").fetchone()[0]
assert prescription_coverage == exercise_count, (
    f"Prescription metadata coverage mismatch: {prescription_coverage}/{exercise_count}"
)
legacy_phase_count = cursor.execute("""
    SELECT COUNT(*)
    FROM exercise_prescription_metadata_v2
    WHERE preferred_session_phase = 'finisher_or_recovery'
       OR ',' || allowed_session_phases || ',' LIKE '%,finisher_or_recovery,%'
""").fetchone()[0]
assert legacy_phase_count == 0, (
    f"Prescription metadata still emits merged finisher/recovery phase: {legacy_phase_count}"
)
assert cursor.execute("""
    SELECT COUNT(*)
    FROM exercise_muscle_taxonomy_v2
    WHERE canonical_primary_muscle = 'side-deltoids'
""").fetchone()[0] > 0, "Canonical side-deltoid taxonomy is empty"
smith_semantic_violations = cursor.execute("""
    SELECT COUNT(*)
    FROM exercise_tier et
    JOIN exercise_equipment_option_v2 eo ON eo.exercise_id = et.id
    WHERE UPPER(COALESCE(et.equipment_req, '')) LIKE '%SMITH_MACHINE%'
      AND eo.equipment_code <> 'SMITH_MACHINE'
""").fetchone()[0]
assert smith_semantic_violations == 0, (
    f"Smith rows expose non-Smith alternatives: {smith_semantic_violations}"
)

# Candidate v2 tables are rebuilt side-by-side with the active runtime tables.
# Their execution rules remain disabled until the evidence/promotion gate passes.
from scripts.build_closed_input_constraint_catalog_v2 import install_closed_input_constraint_catalog_v2

closed_input_v2_report = install_closed_input_constraint_catalog_v2(conn)
assert closed_input_v2_report["coveragePass"], "Closed-input v2 profiles do not cover the full exercise catalog"
assert closed_input_v2_report["manifestCoveragePass"], "Closed-input v2 manifest does not cover every profile signal"
assert closed_input_v2_report["unsupportedPoliciesDelegatedToLlm"] == 0
assert closed_input_v2_report["runtimeEnabledCandidateRules"] == 0
print("Closed-input constraint catalog report:", closed_input_v2_report)

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
print("[SUCCESS] All generated constraint/policy tables created and populated successfully in fit_core.sqlite!")
conn.close()
print("=== Phase 1 Complete ===")

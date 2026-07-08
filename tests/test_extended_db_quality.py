import sqlite3
import os
import pytest

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "fit_core.sqlite")

def get_db_conn():
    if not os.path.exists(DB_PATH):
        pytest.fail(f"SQLite DB not found at {DB_PATH}. Run build_constraints_db.py first.")
    conn = sqlite3.connect(DB_PATH)
    return conn

def test_orphan_foreign_keys():
    """Verify that there are no orphan foreign keys in the sqlite database."""
    conn = get_db_conn()
    cursor = conn.cursor()
    
    # Pragma foreign_key_check returns a row for every foreign key violation.
    # Format of each row: [table_name, rowid, parent_table_name, fkid]
    cursor.execute("PRAGMA foreign_key_check;")
    violations = cursor.fetchall()
    conn.close()
    
    assert len(violations) == 0, f"Foreign Key violations found: {violations}"

def test_relation_types_counts():
    """Verify that all expected relation types have at least one entry in the map."""
    conn = get_db_conn()
    cursor = conn.cursor()
    
    cursor.execute("SELECT relation_type, COUNT(*) FROM exercise_regression_progression_map GROUP BY relation_type;")
    counts = dict(cursor.fetchall())
    conn.close()
    
    expected_relations = [
        "mobilityAlternative",
        "regression",
        "equipmentAlternative",
        "safetyAlternative",
        "variation",
        "rehabAlternative"
    ]
    for rel in expected_relations:
        assert counts.get(rel, 0) > 0, f"Relation type '{rel}' is missing or has 0 entries in DB!"

def test_no_high_risk_targets_in_pain_safety_regression():
    """Verify that t.pain_risk_level != 'high' for painAlternative, safetyAlternative, and regression."""
    conn = get_db_conn()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT m.source_exercise_id, m.target_exercise_id, m.relation_type, t.name_kr, t.pain_risk_level
        FROM exercise_regression_progression_map m
        JOIN exercise_tier t ON t.id = m.target_exercise_id
        WHERE m.relation_type IN ('painAlternative', 'safetyAlternative', 'regression')
          AND t.pain_risk_level = 'high';
    """)
    high_risk_violations = cursor.fetchall()
    conn.close()
    
    assert len(high_risk_violations) == 0, f"Found high-risk target exercises in pain/safety/regression maps: {high_risk_violations}"

def test_bench_press_alternatives_integrity():
    """Verify that Bench Press alternatives do not contain kickbacks or pullovers."""
    conn = get_db_conn()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT m.source_exercise_id, m.target_exercise_id, t_src.name_kr, t_tgt.name_kr
        FROM exercise_regression_progression_map m
        JOIN exercise_tier t_src ON t_src.id = m.source_exercise_id
        JOIN exercise_tier t_tgt ON t_tgt.id = m.target_exercise_id
        WHERE (t_src.name_kr LIKE '%벤치 프레스%' OR t_src.name_kr LIKE '%벤치프레스%')
          AND (t_tgt.name_kr LIKE '%킥백%' OR t_tgt.name_kr LIKE '%kickback%' OR t_tgt.name_kr LIKE '%풀오버%' OR t_tgt.name_kr LIKE '%pullover%');
    """)
    violations = cursor.fetchall()
    conn.close()
    
    assert len(violations) == 0, f"Bench press alternative violates safety rule: {violations}"

def test_lumbar_herniation_policy_counts():
    """Verify that lumbar_herniation policies are generated and have minimum required entries."""
    conn = get_db_conn()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT policy, COUNT(*) 
        FROM exercise_condition_policy 
        WHERE condition_code = 'lumbar_herniation' 
        GROUP BY policy;
    """)
    policies = dict(cursor.fetchall())
    conn.close()
    
    assert policies.get("avoid", 0) > 0, "No 'avoid' policies found for lumbar_herniation"
    assert policies.get("caution", 0) > 0, "No 'caution' policies found for lumbar_herniation"

def test_condition_instance_id_not_null():
    """Verify that there are no NULL condition_instance_id values in user_condition_profile."""
    conn = get_db_conn()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM user_condition_profile WHERE condition_instance_id IS NULL;")
    null_count = cursor.fetchone()[0]
    conn.close()
    
    assert null_count == 0, f"Found {null_count} NULL condition_instance_id values in user_condition_profile!"

def test_representative_exercises_load_profiles():
    """Verify that representative exercises have correct joint load classifications locked in the DB."""
    conn = get_db_conn()
    cursor = conn.cursor()
    
    # 1. Barbell Conventional Deadlift (V24ROW-006680) -> lumbar_load = high, axial_load = low (barbell held in hands, not on shoulders)
    cursor.execute("SELECT lumbar_load, axial_load FROM exercise_joint_load_profile WHERE exercise_id = 'V24ROW-006680';")
    dl = cursor.fetchone()
    assert dl is not None, "Conventional Deadlift not found in exercise_joint_load_profile!"
    assert dl[0] == 'high', f"Conventional Deadlift lumbar_load should be 'high', got '{dl[0]}'"
    assert dl[1] == 'low', f"Conventional Deadlift axial_load should be 'low', got '{dl[1]}'"
    
    # 2. Barbell Back Squat -> lumbar_load = high, axial_load = high (barbell on shoulders creates vertical spinal compression)
    cursor.execute("""
        SELECT jl.lumbar_load, jl.axial_load, t.name_kr 
        FROM exercise_joint_load_profile jl
        JOIN exercise_tier t ON t.id = jl.exercise_id
        WHERE (t.name_kr LIKE '%백스쿼트%' OR t.name_kr LIKE '%백 스쿼트%' OR t.name_en LIKE '%back squat%')
          AND (t.equipment_req LIKE '%바벨%' OR t.equipment_req LIKE '%barbell%')
        LIMIT 1;
    """)
    squat = cursor.fetchone()
    assert squat is not None, "Barbell Back Squat not found in DB!"
    assert squat[0] == 'high', f"Barbell Back Squat '{squat[2]}' lumbar_load should be 'high', got '{squat[0]}'"
    assert squat[1] == 'high', f"Barbell Back Squat '{squat[2]}' axial_load should be 'high', got '{squat[1]}'"
    
    # 3. Dumbbell Lying Leg Curl (V24ROW-003494) -> lumbar_load = low (lying hamstring isolation baseline)
    cursor.execute("SELECT lumbar_load, axial_load FROM exercise_joint_load_profile WHERE exercise_id = 'V24ROW-003494';")
    curl = cursor.fetchone()
    assert curl is not None, "Leg Curl not found in exercise_joint_load_profile!"
    assert curl[0] == 'low', f"Lying Leg Curl lumbar_load should be 'low', got '{curl[0]}'"
    
    conn.close()

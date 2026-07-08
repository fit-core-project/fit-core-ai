import pytest
from fastapi.testclient import TestClient

def test_main_import():
    """
    1. Verify that main.py can be imported without any syntax or import errors,
    ensuring all typing imports (including Optional) are correct.
    """
    try:
        from main import app
        assert app is not None
    except Exception as e:
        pytest.fail(f"Failed to import main.app: {e}")

def test_details_batch_api():
    """
    2. Verify that /api/ai/exercises/details?ids=... batch query executes successfully,
    returns a 200 status, returns the exact number of requested IDs,
    and includes the physical, load, difficulty, and anthropometry metrics in the fields.
    """
    from main import app
    client = TestClient(app)
    
    # We query two known IDs, but first let's get any valid IDs from the database to be safe,
    # or use standard ones that we know exist (e.g., Conventional Deadlift: V24ROW-006680 and another)
    test_ids = "V24ROW-006680,V24ROW-003494"
    
    response = client.get(f"/api/ai/exercises/details?ids={test_ids}")
    assert response.status_code == 200, f"Details batch API failed: {response.text}"
    
    data = response.json()
    assert isinstance(data, list), "Response must be a list"
    assert len(data) == 2, f"Expected 2 exercises, got {len(data)}"
    
    # Verify required fields
    required_fields = [
        "id", "name_kr", "name_en", "primary_muscle",
        "ankle_dorsiflexion_demand", "hip_flexion_demand", "shoulder_flexion_demand", "wrist_extension_demand",
        "overhead_position_required", "lumbar_load", "axial_load", "knee_shear_load",
        "shoulder_impingement_risk", "wrist_stress", "hypertrophy_effect", "strength_effect",
        "stimulus_to_fatigue", "technical_difficulty", "balance_requirement", "failure_penalty",
        "limb_length_sensitivity", "long_femur_sensitivity", "long_arm_sensitivity"
    ]
    
    for exercise in data:
        for field in required_fields:
            assert field in exercise, f"Missing required field '{field}' in exercise details response"

def test_substitution_map_relation_type_integrity():
    """
    3. Verify that the substitution map returns valid relation types and that they
    can be correctly categorized into the four frontend groupings:
    - substitution (regression, safetyAlternative, painAlternative)
    - equipment (equipmentAlternative)
    - rehab (rehabAlternative)
    - variation (variation)
    """
    from main import app
    client = TestClient(app)
    
    response = client.get("/api/ai/exercises/substitution-map")
    assert response.status_code == 200, f"Substitution map API failed: {response.text}"
    
    data = response.json()
    assert isinstance(data, list), "Response must be a list"
    assert len(data) > 0, "Substitution map should not be empty"
    
    # Define valid relation types
    substitution_types = {"regression", "safetyAlternative", "painAlternative"}
    equipment_types = {"equipmentAlternative"}
    rehab_types = {"rehabAlternative"}
    variation_types = {"variation"}
    mobility_types = {"mobilityAlternative"}
    
    all_valid_types = substitution_types | equipment_types | rehab_types | variation_types | mobility_types
    
    found_types = set()
    for item in data:
        rel_type = item.get("relation_type")
        assert rel_type in all_valid_types, f"Invalid relation_type '{rel_type}' found in substitution map"
        found_types.add(rel_type)
        
        # Ensure name_kr, reason etc. are included as optional/provided fields
        assert "source_exercise_id" in item
        assert "target_exercise_id" in item
        assert "relation_type" in item
        assert "constraint_code" in item
        assert "reason" in item
        assert "source_name_kr" in item
        assert "target_name_kr" in item
        
    # Check that we actually have a variety of relation types represented in the database
    # at least one for substitution, one for equipment, one for rehab, and one for variation
    has_sub = any(t in found_types for t in substitution_types)
    has_equip = any(t in found_types for t in equipment_types)
    has_rehab = any(t in found_types for t in rehab_types)
    has_var = any(t in found_types for t in variation_types)
    
    assert has_sub, "Missing substitution relation types (regression, safetyAlternative, painAlternative)"
    assert has_equip, "Missing equipmentAlternative relation types"
    assert has_rehab, "Missing rehabAlternative relation types"
    assert has_var, "Missing variation relation types"

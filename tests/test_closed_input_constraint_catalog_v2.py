import sqlite3
from pathlib import Path

from scripts.build_closed_input_constraint_catalog_v2 import (
    install_closed_input_constraint_catalog_v2,
)


def build_test_db():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE exercise_tier (
            id TEXT PRIMARY KEY,
            name_kr TEXT,
            name_en TEXT,
            equipment_req TEXT
        );
        CREATE TABLE exercise_mobility_requirement (
            exercise_id TEXT PRIMARY KEY,
            ankle_dorsiflexion_demand TEXT,
            hip_flexion_demand TEXT,
            shoulder_flexion_demand TEXT,
            wrist_extension_demand TEXT,
            overhead_position_required INTEGER
        );
        CREATE TABLE exercise_anthropometry_sensitivity_profile (
            exercise_id TEXT PRIMARY KEY,
            limb_length_sensitivity TEXT,
            long_femur_sensitivity TEXT,
            long_arm_sensitivity TEXT,
            torso_angle_demand TEXT
        );
        """
    )
    exercises = [
        ("overhead", "바벨 오버헤드 프레스", "Barbell Overhead Press", "BARBELL", "low", "low", "high", "medium", 0),
        ("lateral", "덤벨 레터럴 레이즈", "Dumbbell Lateral Raise", "DUMBBELL", "low", "low", "medium", "low", 0),
        ("incline", "덤벨 인클라인 체스트 프레스", "Dumbbell Incline Chest Press", "DUMBBELL", "low", "low", "low", "low", 0),
        ("machine", "머신 어깨너비 풀 백 스쿼트", "Machine Full Back Squat", "MACHINE", "high", "medium", "low", "low", 0),
    ]
    for item in exercises:
        eid, name_kr, name_en, equipment, ankle, hip, shoulder, wrist, overhead = item
        conn.execute("INSERT INTO exercise_tier VALUES (?, ?, ?, ?)", (eid, name_kr, name_en, equipment))
        conn.execute(
            "INSERT INTO exercise_mobility_requirement VALUES (?, ?, ?, ?, ?, ?)",
            (eid, ankle, hip, shoulder, wrist, overhead),
        )
        conn.execute(
            "INSERT INTO exercise_anthropometry_sensitivity_profile VALUES (?, ?, ?, ?, ?)",
            (eid, "high" if eid == "machine" else "low", "low", "low", "low"),
        )
    conn.commit()
    return conn


def test_profile_input_is_closed_to_four_binary_movement_signals():
    conn = build_test_db()
    report = install_closed_input_constraint_catalog_v2(conn)

    assert report["profileSelectableMovementSignals"] == [
        "heel_down_deep_squat_difficult",
        "hip_hinge_difficult",
        "loaded_wrist_extension_difficult",
        "overhead_reach_difficult",
    ]


def test_self_reported_body_proportion_signals_are_binary_advisory_and_default_off():
    conn = build_test_db()
    report = install_closed_input_constraint_catalog_v2(conn)

    assert report["profileSelectableBodyProportionSignals"] == [
        "self_reported_long_arms",
        "self_reported_long_thighs",
    ]
    assert report["profileSelectableSignalCount"] == 6
    assert report["defaultSelectedProfileSignalCount"] == 0
    assert conn.execute(
        """
        SELECT COUNT(*) FROM user_constraint_option_catalog_v2
        WHERE category = 'selfReportedAnthropometry'
          AND user_selectable = 1
          AND safety_authority = 'advisoryOnly'
          AND default_selected = 0
        """
    ).fetchone()[0] == 2
    assert conn.execute(
        """
        SELECT COUNT(*) FROM constraint_execution_rule_v2
        WHERE signal_code LIKE 'self_reported_%'
          AND action = 'hardFilter'
        """
    ).fetchone()[0] == 0


def test_unsupported_safety_input_is_never_delegated_to_llm():
    conn = build_test_db()
    report = install_closed_input_constraint_catalog_v2(conn)

    assert report["unsupportedPoliciesDelegatedToLlm"] == 0
    assert conn.execute(
        "SELECT action FROM unsupported_input_policy_v2 WHERE input_class = 'unmodeled_medical_or_neurologic_condition'"
    ).fetchone()[0] == "failClosed"


def test_v2_profiles_cover_every_exercise_but_remain_runtime_disabled():
    conn = build_test_db()
    report = install_closed_input_constraint_catalog_v2(conn)

    assert report["coveragePass"] is True
    assert report["manifestCoveragePass"] is True
    assert report["profileInputManifestCount"] == 6
    assert report["runtimeEnabledCandidateRules"] == 0
    assert conn.execute("SELECT COUNT(*) FROM exercise_movement_requirement_profile_v2").fetchone()[0] == 4
    assert conn.execute("SELECT COUNT(*) FROM exercise_setup_fit_profile_v2").fetchone()[0] == 4
    assert conn.execute(
        "SELECT COUNT(*) FROM exercise_managed_injury_movement_profile_v1"
    ).fetchone()[0] == 4
    assert report["managedInjuryMovementCoveragePass"] is True


def test_shoulder_plane_and_incline_are_not_collapsed_into_one_flexion_label():
    conn = build_test_db()
    install_closed_input_constraint_catalog_v2(conn)

    overhead = conn.execute(
        "SELECT shoulder_elevation_plane, overhead_position_required, bench_angle_band FROM exercise_movement_requirement_profile_v2 WHERE exercise_id='overhead'"
    ).fetchone()
    lateral = conn.execute(
        "SELECT shoulder_elevation_plane, overhead_position_required FROM exercise_movement_requirement_profile_v2 WHERE exercise_id='lateral'"
    ).fetchone()
    incline = conn.execute(
        "SELECT shoulder_elevation_plane, bench_angle_band FROM exercise_movement_requirement_profile_v2 WHERE exercise_id='incline'"
    ).fetchone()

    assert tuple(overhead) == ("overhead", 1, "overhead")
    assert tuple(lateral) == ("frontal", 0)
    assert tuple(incline) == ("horizontal", "inclineUnspecified")


def test_full_back_squat_name_is_classified_as_deep_squat_required():
    conn = build_test_db()
    install_closed_input_constraint_catalog_v2(conn)

    row = conn.execute(
        """
        SELECT deep_squat_position_required
        FROM exercise_movement_requirement_profile_v2
        WHERE exercise_id='machine'
        """
    ).fetchone()
    rule = conn.execute(
        """
        SELECT action, score_delta
        FROM constraint_execution_rule_v2
        WHERE signal_code='heel_down_deep_squat_difficult'
          AND candidate_field='deep_squat_position_required'
        """
    ).fetchone()

    assert row[0] == 1
    assert tuple(rule) == ("hardFilter", None)


def test_machine_word_does_not_claim_unverified_adjustability():
    conn = build_test_db()
    install_closed_input_constraint_catalog_v2(conn)

    row = conn.execute(
        """
        SELECT path_type, machine_model_dependency, seat_adjustability,
               start_position_adjustability, pad_axis_adjustability,
               handle_reach_adjustability
        FROM exercise_setup_fit_profile_v2
        WHERE exercise_id='machine'
        """
    ).fetchone()

    assert tuple(row) == ("machineDependent", 1, "unknown", "unknown", "unknown", "unknown")


def test_unreviewed_long_lower_leg_sensitivity_stays_unknown():
    conn = build_test_db()
    install_closed_input_constraint_catalog_v2(conn)

    assert conn.execute(
        """
        SELECT COUNT(*) FROM exercise_setup_fit_profile_v2
        WHERE long_tibia_sensitivity != 'unknown'
        """
    ).fetchone()[0] == 0


def test_repository_db_has_full_v2_coverage_and_no_active_candidate_rules():
    db_path = Path(__file__).resolve().parents[1] / "fit_core.sqlite"
    conn = sqlite3.connect(db_path)
    exercise_count = conn.execute("SELECT COUNT(*) FROM exercise_tier").fetchone()[0]

    assert conn.execute(
        "SELECT COUNT(*) FROM exercise_movement_requirement_profile_v2"
    ).fetchone()[0] == exercise_count
    assert conn.execute(
        "SELECT COUNT(*) FROM exercise_setup_fit_profile_v2"
    ).fetchone()[0] == exercise_count
    assert conn.execute(
        "SELECT COUNT(*) FROM exercise_managed_injury_movement_profile_v1"
    ).fetchone()[0] == exercise_count
    assert conn.execute(
        "SELECT COUNT(*) FROM constraint_execution_rule_v2 WHERE runtime_enabled != 0"
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM unsupported_input_policy_v2 WHERE llm_may_decide != 0"
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM user_profile_input_manifest_v2"
    ).fetchone()[0] == 6

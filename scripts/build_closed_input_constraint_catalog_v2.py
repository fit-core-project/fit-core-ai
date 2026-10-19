"""Build the closed-input constraint catalog on the compatible v2 DB tables.

The physical table names remain v2 for compatibility. The versioned rows follow
the current public profile-input contract and exclude unsupported no-op inputs.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engines.profile_constraint_contract import (
    PROFILE_INPUT_CONTRACT_VERSION,
    PROFILE_SIGNAL_DEFINITIONS,
)
from engines.injury_movement_metadata import (
    PROFILE_FIELDS as INJURY_MOVEMENT_PROFILE_FIELDS,
    derive_injury_movement_metadata,
)


CONSTRAINT_CATALOG_VERSION = "routine-constraint-db-v3-2026-07-22"
VERSION = CONSTRAINT_CATALOG_VERSION


USER_CONSTRAINT_OPTIONS = (
    *PROFILE_SIGNAL_DEFINITIONS,
    {
        "code": "long_arm_span_relative_to_height",
        "category": "derivedAnthropometry",
        "label_ko": "키 대비 긴 팔 길이 신호",
        "help_text_ko": "사용자 자가판정이 아니라 검증된 신체 측정값에서만 내부 파생합니다.",
        "input_stage": "internalDerived",
        "user_selectable": 0,
        "safety_authority": "advisoryOnly",
        "legacy_internal_code": "long_arm",
        "context_requirement": "height_and_arm_span_measurements",
    },
    {
        "code": "long_leg_relative_to_torso",
        "category": "derivedAnthropometry",
        "label_ko": "몸통 대비 긴 다리 신호",
        "help_text_ko": "키나 인심만으로 대퇴·경골 비율을 추측하지 않고, 충분한 측정값이 있을 때만 파생합니다.",
        "input_stage": "internalDerived",
        "user_selectable": 0,
        "safety_authority": "advisoryOnly",
        "legacy_internal_code": "long_leg",
        "context_requirement": "height_seated_height_and_leg_measurements",
    },
    {
        "code": "long_femur_relative_to_tibia",
        "category": "derivedAnthropometry",
        "label_ko": "경골 대비 긴 대퇴 신호",
        "help_text_ko": "자가 선택을 허용하지 않으며 분절 길이 측정이 없으면 unknown으로 유지합니다.",
        "input_stage": "internalDerived",
        "user_selectable": 0,
        "safety_authority": "advisoryOnly",
        "legacy_internal_code": "long_femur",
        "context_requirement": "femur_and_tibia_measurements",
    },
    {
        "code": "machine_start_position_unreachable",
        "category": "exerciseFitFeedback",
        "label_ko": "기구 시작 위치에 편하게 닿지 않음",
        "help_text_ko": "특정 운동·기구 사용 후에만 기록하며 전체 머신 운동으로 일반화하지 않습니다.",
        "input_stage": "postExerciseFeedback",
        "user_selectable": 1,
        "safety_authority": "deterministicConstraint",
        "legacy_internal_code": None,
        "context_requirement": "exercise_id_and_machine_identity",
    },
    {
        "code": "machine_range_uncomfortable",
        "category": "exerciseFitFeedback",
        "label_ko": "기구의 시작 또는 끝 가동범위가 맞지 않음",
        "help_text_ko": "특정 기구의 궤도·가동범위 문제로만 저장합니다.",
        "input_stage": "postExerciseFeedback",
        "user_selectable": 1,
        "safety_authority": "deterministicConstraint",
        "legacy_internal_code": None,
        "context_requirement": "exercise_id_and_machine_identity",
    },
    {
        "code": "machine_pad_axis_mismatch",
        "category": "exerciseFitFeedback",
        "label_ko": "패드나 회전축 위치가 몸에 맞지 않음",
        "help_text_ko": "좌석 조절 후에도 관절축과 기구축이 맞지 않는 특정 기구 피드백입니다.",
        "input_stage": "postExerciseFeedback",
        "user_selectable": 1,
        "safety_authority": "deterministicConstraint",
        "legacy_internal_code": None,
        "context_requirement": "exercise_id_and_machine_identity",
    },
)


UNSUPPORTED_INPUT_POLICIES = (
    {
        "input_class": "unknown_enum_value",
        "action": "reject",
        "llm_may_decide": 0,
        "user_message_ko": "지원하는 선택지에서 다시 선택해 주세요.",
    },
    {
        "input_class": "unmodeled_medical_or_neurologic_condition",
        "action": "failClosed",
        "llm_may_decide": 0,
        "user_message_ko": "현재 서비스가 안전하게 반영할 수 없는 의료 조건입니다. 전문적인 확인 없이 맞춤 루틴을 생성하지 않습니다.",
    },
    {
        "input_class": "free_text_safety_claim",
        "action": "ignoreForSafety",
        "llm_may_decide": 0,
        "user_message_ko": "자유 입력 내용은 안전 필터의 근거로 사용하지 않습니다. 지원하는 부상·동작 제한 항목을 선택해 주세요.",
    },
    {
        "input_class": "unmodeled_non_safety_preference",
        "action": "storeAsPreference",
        "llm_may_decide": 0,
        "user_message_ko": "선호 기록으로만 보관하며 안전 또는 적합성 판정에는 사용하지 않습니다.",
    },
)


CONSTRAINT_RULE_CANDIDATES = (
    ("overhead_reach_difficult", 10, "overhead_position_required", "eq", "1", "hardFilter", None, "머리 위 자세가 필수인 운동은 수행 가능 범위를 벗어납니다."),
    ("overhead_reach_difficult", 20, "shoulder_elevation_demand", "eq", "high", "hardFilter", None, "어깨 거상 요구도가 높은 운동은 후보에서 제외합니다."),
    ("overhead_reach_difficult", 30, "shoulder_elevation_demand", "eq", "medium", "penalty", -8, "중간 수준의 어깨 거상 요구도는 대체 후보보다 우선순위를 낮춥니다."),
    ("heel_down_deep_squat_difficult", 10, "deep_squat_position_required", "eq", "1", "hardFilter", None, "발뒤꿈치를 붙인 깊은 스쿼트 자세가 필수인 운동은 수행 가능 범위를 벗어납니다."),
    ("heel_down_deep_squat_difficult", 20, "ankle_dorsiflexion_demand", "eq", "high", "penalty", -20, "발목 배굴 요구도가 높은 운동은 강하게 감점합니다."),
    ("heel_down_deep_squat_difficult", 30, "ankle_dorsiflexion_demand", "eq", "medium", "penalty", -8, "중간 수준의 발목 배굴 요구도는 보수적으로 감점합니다."),
    ("hip_hinge_difficult", 10, "hip_flexion_demand", "eq", "high", "penalty", -20, "고관절 굴곡 요구도가 높은 운동은 강하게 감점합니다."),
    ("hip_hinge_difficult", 20, "hip_flexion_demand", "eq", "medium", "penalty", -8, "중간 수준의 고관절 굴곡 요구도는 보수적으로 감점합니다."),
    ("loaded_wrist_extension_difficult", 10, "wrist_loaded_extension_demand", "eq", "high", "hardFilter", None, "부하가 실린 손목 신전 자세가 필수인 운동은 후보에서 제외합니다."),
    ("loaded_wrist_extension_difficult", 20, "wrist_loaded_extension_demand", "eq", "medium", "penalty", -8, "중간 수준의 손목 신전 요구도는 대체 후보보다 우선순위를 낮춥니다."),
    ("self_reported_long_arms", 10, "long_arm_sensitivity", "eq", "high", "penalty", -4, "자가 보고 팔 길이 신호는 검수된 체형 민감도가 높은 후보를 약하게 감점합니다."),
    ("self_reported_long_thighs", 10, "long_femur_sensitivity", "eq", "high", "penalty", -4, "자가 보고 허벅지 길이 신호는 검수된 체형 민감도가 높은 후보를 약하게 감점합니다."),
)


V2_TABLES = (
    "exercise_managed_injury_movement_profile_v1",
    "constraint_execution_rule_v2",
    "unsupported_input_policy_v2",
    "user_profile_input_manifest_v2",
    "user_constraint_option_catalog_v2",
    "exercise_setup_fit_profile_v2",
    "exercise_movement_requirement_profile_v2",
)


def _compact(*values: Any) -> str:
    return "".join(str(value or "").lower().replace(" ", "").replace("_", "").replace("-", "") for value in values)


def _contains(text: str, tokens: Iterable[str]) -> bool:
    return any(_compact(token) in text for token in tokens)


def _movement_row(row: sqlite3.Row) -> tuple[Any, ...]:
    text = _compact(row["name_kr"], row["name_en"])
    current_ankle = row["ankle_dorsiflexion_demand"] or "unknown"
    current_hip = row["hip_flexion_demand"] or "unknown"
    current_shoulder = row["shoulder_flexion_demand"] or "unknown"
    current_wrist = row["wrist_extension_demand"] or "unknown"
    overhead = int(
        row["overhead_position_required"]
        or _contains(text, ("오버헤드", "overhead"))
    )

    is_squat_family = _contains(text, ("스쿼트", "squat"))
    deep_squat_required = int(
        _contains(text, ("atg", "딥스쿼트", "deep squat", "풀뎁스", "full depth"))
        or (
            is_squat_family
            and _contains(text, ("풀", "full", "딥", "deep", "atg"))
        )
    )
    knee_depth = "unknown"
    if deep_squat_required:
        knee_depth = "high"
    elif _contains(text, ("스쿼트", "squat", "런지", "lunge", "레그프레스", "legpress")):
        knee_depth = "medium"
    elif _contains(text, ("레그컬", "legcurl", "레그익스텐션", "legextension", "카프레이즈", "calfraise")):
        knee_depth = "low"

    shoulder_plane = "unknown"
    if overhead:
        shoulder_plane = "overhead"
    elif _contains(text, ("레터럴레이즈", "lateralraise", "사이드레이즈", "sideraise")):
        shoulder_plane = "frontal"
    elif _contains(text, ("프론트레이즈", "frontraise")):
        shoulder_plane = "sagittal"
    elif _contains(text, ("체스트프레스", "chestpress", "벤치프레스", "benchpress", "플라이", "fly", "푸시업", "pushup")):
        shoulder_plane = "horizontal"

    if shoulder_plane == "frontal" and current_shoulder == "medium":
        # v1 grouped every raise under flexion. Preserve the demand as reviewable
        # shoulder elevation without continuing the flexion-plane mislabel.
        shoulder_elevation = "medium"
    else:
        shoulder_elevation = current_shoulder

    bench_angle = "notApplicable"
    if overhead:
        bench_angle = "overhead"
    elif _contains(text, ("인클라인", "incline")):
        bench_angle = "inclineUnspecified"
    elif _contains(text, ("디클라인", "decline")):
        bench_angle = "decline"
    elif _contains(text, ("플랫", "flat")):
        bench_angle = "flat"

    return (
        row["id"], current_ankle, current_hip, knee_depth,
        shoulder_elevation, shoulder_plane, current_wrist, overhead,
        deep_squat_required, bench_angle,
        "derivedFromV25AndV1", "medium", 1, VERSION,
    )


def _fit_row(row: sqlite3.Row) -> tuple[Any, ...]:
    text = _compact(row["name_kr"], row["name_en"])
    equipment = _compact(row["equipment_req"])

    path_type = "unknown"
    if _contains(equipment, ("smithmachine", "스미스머신")):
        path_type = "guided"
    elif _contains(equipment, ("machine", "머신")):
        path_type = "machineDependent"
    elif _contains(equipment, ("cable", "케이블")):
        path_type = "semiFree"
    elif _contains(equipment, ("barbell", "바벨", "dumbbell", "덤벨", "kettlebell", "케틀벨", "bodyweight", "맨몸")):
        path_type = "free"

    coupling = "unknown"
    if _contains(text, ("싱글암", "singlearm", "싱글레그", "singleleg", "교대", "alternating")):
        coupling = "independent"
    elif _contains(text, ("양측", "bilateral")) or _contains(equipment, ("barbell", "바벨", "smithmachine", "스미스머신")):
        coupling = "coupled"
    elif _contains(equipment, ("dumbbell", "덤벨", "cable", "케이블")):
        coupling = "independentCapable"

    grip_freedom = "unknown"
    if _contains(equipment, ("dumbbell", "덤벨", "cable", "케이블", "kettlebell", "케틀벨")):
        grip_freedom = "high"
    elif _contains(equipment, ("barbell", "바벨", "smithmachine", "스미스머신")):
        grip_freedom = "low"

    machine_dependency = int(_contains(equipment, ("machine", "머신", "smithmachine", "스미스머신")))

    return (
        row["id"], path_type, coupling, grip_freedom,
        "unknown", "unknown", "unknown", "unknown",
        machine_dependency,
        row["limb_length_sensitivity"] or "unknown",
        row["long_femur_sensitivity"] or "unknown",
        "unknown",
        row["long_arm_sensitivity"] or "unknown",
        row["torso_angle_demand"] or "unknown",
        "derivedFromV25AndV1", "low", 1, VERSION,
    )


def _injury_movement_row(row: sqlite3.Row) -> tuple[Any, ...]:
    metadata = derive_injury_movement_metadata(dict(row))
    return (
        row["id"],
        *(metadata[field] for field in INJURY_MOVEMENT_PROFILE_FIELDS),
        metadata["source_type"],
        metadata["confidence"],
        metadata["review_required"],
        metadata["version"],
    )


def install_closed_input_constraint_catalog_v2(conn: sqlite3.Connection) -> dict[str, Any]:
    """Populate compatible v2 physical tables with current versioned rows."""
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    for table in V2_TABLES:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")

    cursor.executescript(
        """
        CREATE TABLE user_constraint_option_catalog_v2 (
            code TEXT PRIMARY KEY NOT NULL,
            category TEXT NOT NULL CHECK(category IN ('functionalMovement', 'selfReportedAnthropometry', 'derivedAnthropometry', 'exerciseFitFeedback')),
            label_ko TEXT NOT NULL,
            help_text_ko TEXT NOT NULL,
            input_stage TEXT NOT NULL CHECK(input_stage IN ('profileOptional', 'internalDerived', 'postExerciseFeedback')),
            user_selectable INTEGER NOT NULL CHECK(user_selectable IN (0, 1)),
            default_selected INTEGER NOT NULL DEFAULT 0 CHECK(default_selected IN (0, 1)),
            safety_authority TEXT NOT NULL CHECK(safety_authority IN ('deterministicConstraint', 'advisoryOnly')),
            legacy_internal_code TEXT,
            context_requirement TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0, 1)),
            version TEXT NOT NULL
        );

        CREATE TABLE unsupported_input_policy_v2 (
            input_class TEXT PRIMARY KEY NOT NULL,
            action TEXT NOT NULL CHECK(action IN ('reject', 'failClosed', 'ignoreForSafety', 'storeAsPreference')),
            llm_may_decide INTEGER NOT NULL CHECK(llm_may_decide IN (0, 1)),
            user_message_ko TEXT NOT NULL,
            version TEXT NOT NULL
        );

        CREATE TABLE user_profile_input_manifest_v2 (
            signal_code TEXT PRIMARY KEY NOT NULL,
            ui_section TEXT NOT NULL CHECK(ui_section IN ('movementChecks', 'bodyProportionChecks')),
            display_order INTEGER NOT NULL UNIQUE,
            question_ko TEXT NOT NULL,
            selection_type TEXT NOT NULL DEFAULT 'checkbox' CHECK(selection_type = 'checkbox'),
            default_selected INTEGER NOT NULL DEFAULT 0 CHECK(default_selected IN (0, 1)),
            unchecked_semantics TEXT NOT NULL CHECK(unchecked_semantics IN ('noKnownDifficultyReported', 'notSelfReportedAsNotable')),
            pain_during_task_action TEXT NOT NULL CHECK(pain_during_task_action IN ('routeToRegisteredInjury', 'notApplicable')),
            version TEXT NOT NULL,
            FOREIGN KEY(signal_code) REFERENCES user_constraint_option_catalog_v2(code) ON DELETE CASCADE
        );

        CREATE TABLE constraint_execution_rule_v2 (
            signal_code TEXT NOT NULL,
            rule_order INTEGER NOT NULL,
            candidate_field TEXT NOT NULL,
            operator TEXT NOT NULL CHECK(operator IN ('eq', 'in', 'gte')),
            compare_value TEXT NOT NULL,
            action TEXT NOT NULL CHECK(action IN ('hardFilter', 'penalty', 'boost', 'setupPreference')),
            score_delta INTEGER,
            reason_ko TEXT NOT NULL,
            runtime_enabled INTEGER NOT NULL DEFAULT 0 CHECK(runtime_enabled IN (0, 1)),
            review_status TEXT NOT NULL DEFAULT 'workingDraft' CHECK(review_status IN ('workingDraft', 'approved', 'rejected', 'superseded')),
            version TEXT NOT NULL,
            PRIMARY KEY(signal_code, rule_order),
            FOREIGN KEY(signal_code) REFERENCES user_constraint_option_catalog_v2(code) ON DELETE CASCADE
        );

        CREATE TABLE exercise_movement_requirement_profile_v2 (
            exercise_id TEXT PRIMARY KEY NOT NULL,
            ankle_dorsiflexion_demand TEXT NOT NULL,
            hip_flexion_demand TEXT NOT NULL,
            knee_flexion_depth_demand TEXT NOT NULL,
            shoulder_elevation_demand TEXT NOT NULL,
            shoulder_elevation_plane TEXT NOT NULL CHECK(shoulder_elevation_plane IN ('overhead', 'frontal', 'sagittal', 'horizontal', 'mixed', 'notApplicable', 'unknown')),
            wrist_loaded_extension_demand TEXT NOT NULL,
            overhead_position_required INTEGER NOT NULL CHECK(overhead_position_required IN (0, 1)),
            deep_squat_position_required INTEGER NOT NULL CHECK(deep_squat_position_required IN (0, 1)),
            bench_angle_band TEXT NOT NULL CHECK(bench_angle_band IN ('flat', 'inclineUnspecified', 'decline', 'overhead', 'notApplicable', 'unknown')),
            source_type TEXT NOT NULL,
            confidence TEXT NOT NULL CHECK(confidence IN ('low', 'medium', 'high')),
            review_required INTEGER NOT NULL CHECK(review_required IN (0, 1)),
            version TEXT NOT NULL,
            FOREIGN KEY(exercise_id) REFERENCES exercise_tier(id) ON DELETE CASCADE
        );

        CREATE TABLE exercise_setup_fit_profile_v2 (
            exercise_id TEXT PRIMARY KEY NOT NULL,
            path_type TEXT NOT NULL CHECK(path_type IN ('free', 'semiFree', 'guided', 'machineDependent', 'unknown')),
            bilateral_coupling TEXT NOT NULL CHECK(bilateral_coupling IN ('independent', 'independentCapable', 'coupled', 'notApplicable', 'unknown')),
            grip_freedom TEXT NOT NULL CHECK(grip_freedom IN ('low', 'medium', 'high', 'notApplicable', 'unknown')),
            seat_adjustability TEXT NOT NULL CHECK(seat_adjustability IN ('none', 'limited', 'good', 'unknown')),
            start_position_adjustability TEXT NOT NULL CHECK(start_position_adjustability IN ('none', 'limited', 'good', 'unknown')),
            pad_axis_adjustability TEXT NOT NULL CHECK(pad_axis_adjustability IN ('none', 'limited', 'good', 'unknown')),
            handle_reach_adjustability TEXT NOT NULL CHECK(handle_reach_adjustability IN ('none', 'limited', 'good', 'unknown')),
            machine_model_dependency INTEGER NOT NULL CHECK(machine_model_dependency IN (0, 1)),
            limb_length_sensitivity TEXT NOT NULL,
            long_femur_sensitivity TEXT NOT NULL,
            long_tibia_sensitivity TEXT NOT NULL,
            long_arm_sensitivity TEXT NOT NULL,
            torso_angle_demand TEXT NOT NULL,
            source_type TEXT NOT NULL,
            confidence TEXT NOT NULL CHECK(confidence IN ('low', 'medium', 'high')),
            review_required INTEGER NOT NULL CHECK(review_required IN (0, 1)),
            version TEXT NOT NULL,
            FOREIGN KEY(exercise_id) REFERENCES exercise_tier(id) ON DELETE CASCADE
        );

        CREATE TABLE exercise_managed_injury_movement_profile_v1 (
            exercise_id TEXT PRIMARY KEY NOT NULL,
            ankle_region_load_level TEXT NOT NULL CHECK(ankle_region_load_level IN ('low', 'medium', 'high')),
            deep_dorsiflexion_demand TEXT NOT NULL CHECK(deep_dorsiflexion_demand IN ('low', 'medium', 'high')),
            plantar_flexion_load TEXT NOT NULL CHECK(plantar_flexion_load IN ('low', 'medium', 'high')),
            single_leg_stability_demand TEXT NOT NULL CHECK(single_leg_stability_demand IN ('low', 'medium', 'high')),
            impact_demand TEXT NOT NULL CHECK(impact_demand IN ('low', 'medium', 'high')),
            hip_region_load_level TEXT NOT NULL CHECK(hip_region_load_level IN ('low', 'medium', 'high')),
            deep_hip_flexion_demand TEXT NOT NULL CHECK(deep_hip_flexion_demand IN ('low', 'medium', 'high')),
            hip_hinge_load TEXT NOT NULL CHECK(hip_hinge_load IN ('low', 'medium', 'high')),
            hip_abduction_load TEXT NOT NULL CHECK(hip_abduction_load IN ('low', 'medium', 'high')),
            hip_adduction_load TEXT NOT NULL CHECK(hip_adduction_load IN ('low', 'medium', 'high')),
            hip_rotation_demand TEXT NOT NULL CHECK(hip_rotation_demand IN ('low', 'medium', 'high')),
            elbow_region_load_level TEXT NOT NULL CHECK(elbow_region_load_level IN ('low', 'medium', 'high')),
            elbow_flexion_load TEXT NOT NULL CHECK(elbow_flexion_load IN ('low', 'medium', 'high')),
            elbow_extension_load TEXT NOT NULL CHECK(elbow_extension_load IN ('low', 'medium', 'high')),
            loaded_grip_demand TEXT NOT NULL CHECK(loaded_grip_demand IN ('low', 'medium', 'high')),
            forearm_rotation_load TEXT NOT NULL CHECK(forearm_rotation_load IN ('low', 'medium', 'high')),
            upper_limb_weight_bearing_load TEXT NOT NULL CHECK(upper_limb_weight_bearing_load IN ('low', 'medium', 'high')),
            neck_region_load_level TEXT NOT NULL CHECK(neck_region_load_level IN ('low', 'medium', 'high')),
            direct_neck_load TEXT NOT NULL CHECK(direct_neck_load IN ('low', 'medium', 'high')),
            overhead_axial_load TEXT NOT NULL CHECK(overhead_axial_load IN ('low', 'medium', 'high')),
            sustained_neck_position_demand TEXT NOT NULL CHECK(sustained_neck_position_demand IN ('low', 'medium', 'high')),
            ballistic_impact_demand TEXT NOT NULL CHECK(ballistic_impact_demand IN ('low', 'medium', 'high')),
            source_type TEXT NOT NULL,
            confidence TEXT NOT NULL CHECK(confidence IN ('low', 'medium', 'high')),
            review_required INTEGER NOT NULL CHECK(review_required IN (0, 1)),
            version TEXT NOT NULL,
            FOREIGN KEY(exercise_id) REFERENCES exercise_tier(id) ON DELETE CASCADE
        );
        """
    )

    cursor.executemany(
        """
        INSERT INTO user_constraint_option_catalog_v2 (
            code, category, label_ko, help_text_ko, input_stage,
            user_selectable, default_selected, safety_authority, legacy_internal_code,
            context_requirement, active, version
        ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, 1, ?)
        """,
        [
            (
                item["code"], item["category"], item["label_ko"], item["help_text_ko"],
                item["input_stage"], item["user_selectable"], item["safety_authority"],
                item["legacy_internal_code"], item["context_requirement"], VERSION,
            )
            for item in USER_CONSTRAINT_OPTIONS
        ],
    )
    cursor.executemany(
        "INSERT INTO unsupported_input_policy_v2 VALUES (?, ?, ?, ?, ?)",
        [
            (item["input_class"], item["action"], item["llm_may_decide"], item["user_message_ko"], VERSION)
            for item in UNSUPPORTED_INPUT_POLICIES
        ],
    )
    cursor.executemany(
        """
        INSERT INTO user_profile_input_manifest_v2 (
            signal_code, ui_section, display_order, question_ko,
            selection_type, default_selected, unchecked_semantics,
            pain_during_task_action, version
        ) VALUES (?, ?, ?, ?, 'checkbox', ?, ?, ?, ?)
        """,
        [
            (
                item["code"], item["ui_section"], item["display_order"],
                item["question_ko"], item["default_selected"],
                item["unchecked_semantics"], item["pain_during_task_action"],
                PROFILE_INPUT_CONTRACT_VERSION,
            )
            for item in PROFILE_SIGNAL_DEFINITIONS
        ],
    )
    cursor.executemany(
        """
        INSERT INTO constraint_execution_rule_v2 (
            signal_code, rule_order, candidate_field, operator, compare_value,
            action, score_delta, reason_ko, runtime_enabled, review_status, version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 'workingDraft', ?)
        """,
        [(*item, VERSION) for item in CONSTRAINT_RULE_CANDIDATES],
    )

    exercise_columns = {
        row[1] for row in cursor.execute("PRAGMA table_info(exercise_tier)")
    }
    has_taxonomy = bool(cursor.execute(
        """
        SELECT 1 FROM sqlite_master
        WHERE type = 'table' AND name = 'exercise_muscle_taxonomy_v2'
        """
    ).fetchone())
    optional_exercise_select = {
        column: (
            f"et.{column}" if column in exercise_columns else f"NULL AS {column}"
        )
        for column in ("primary_muscle", "secondary_muscle", "movement_type")
    }
    taxonomy_select = (
        """
            mt.canonical_primary_muscle,
            mt.canonical_secondary_muscles,
        """
        if has_taxonomy
        else
        """
            NULL AS canonical_primary_muscle,
            NULL AS canonical_secondary_muscles,
        """
    )
    taxonomy_join = (
        "LEFT JOIN exercise_muscle_taxonomy_v2 mt ON mt.exercise_id = et.id"
        if has_taxonomy
        else ""
    )
    rows = cursor.execute(
        f"""
        SELECT
            et.id, et.name_kr, et.name_en, et.equipment_req,
            {optional_exercise_select["primary_muscle"]},
            {optional_exercise_select["secondary_muscle"]},
            {optional_exercise_select["movement_type"]},
            {taxonomy_select}
            mr.ankle_dorsiflexion_demand,
            mr.hip_flexion_demand,
            mr.shoulder_flexion_demand,
            mr.wrist_extension_demand,
            mr.overhead_position_required,
            ap.limb_length_sensitivity,
            ap.long_femur_sensitivity,
            ap.long_arm_sensitivity,
            ap.torso_angle_demand
        FROM exercise_tier et
        JOIN exercise_mobility_requirement mr ON mr.exercise_id = et.id
        JOIN exercise_anthropometry_sensitivity_profile ap ON ap.exercise_id = et.id
        {taxonomy_join}
        ORDER BY et.id
        """
    ).fetchall()

    cursor.executemany(
        """
        INSERT INTO exercise_movement_requirement_profile_v2 VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        [_movement_row(row) for row in rows],
    )
    cursor.executemany(
        """
        INSERT INTO exercise_setup_fit_profile_v2 VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        [_fit_row(row) for row in rows],
    )
    cursor.executemany(
        """
        INSERT INTO exercise_managed_injury_movement_profile_v1 VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?
        )
        """,
        [_injury_movement_row(row) for row in rows],
    )
    conn.commit()
    return audit_closed_input_constraint_catalog_v2(conn)


def audit_closed_input_constraint_catalog_v2(conn: sqlite3.Connection) -> dict[str, Any]:
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    exercise_count = cursor.execute("SELECT COUNT(*) FROM exercise_tier").fetchone()[0]
    movement_count = cursor.execute("SELECT COUNT(*) FROM exercise_movement_requirement_profile_v2").fetchone()[0]
    fit_count = cursor.execute("SELECT COUNT(*) FROM exercise_setup_fit_profile_v2").fetchone()[0]
    injury_movement_count = cursor.execute(
        "SELECT COUNT(*) FROM exercise_managed_injury_movement_profile_v1"
    ).fetchone()[0]
    manifest_count = cursor.execute("SELECT COUNT(*) FROM user_profile_input_manifest_v2").fetchone()[0]
    manifest_version = cursor.execute(
        "SELECT MIN(version) FROM user_profile_input_manifest_v2"
    ).fetchone()[0]
    selectable = [
        row[0]
        for row in cursor.execute(
            """
            SELECT code FROM user_constraint_option_catalog_v2
            WHERE user_selectable = 1 AND input_stage = 'profileOptional'
            ORDER BY code
            """
        )
    ]
    movement_selectable = [
        row[0]
        for row in cursor.execute(
            """
            SELECT code FROM user_constraint_option_catalog_v2
            WHERE user_selectable = 1
              AND input_stage = 'profileOptional'
              AND category = 'functionalMovement'
            ORDER BY code
            """
        )
    ]
    body_proportion_selectable = [
        row[0]
        for row in cursor.execute(
            """
            SELECT code FROM user_constraint_option_catalog_v2
            WHERE user_selectable = 1
              AND input_stage = 'profileOptional'
              AND category = 'selfReportedAnthropometry'
            ORDER BY code
            """
        )
    ]
    default_selected_count = cursor.execute(
        """
        SELECT COUNT(*) FROM user_constraint_option_catalog_v2
        WHERE user_selectable = 1
          AND input_stage = 'profileOptional'
          AND default_selected != 0
        """
    ).fetchone()[0]
    fit_unknown = dict(
        cursor.execute(
            """
            SELECT
                SUM(seat_adjustability = 'unknown') AS seat_adjustability,
                SUM(start_position_adjustability = 'unknown') AS start_position_adjustability,
                SUM(pad_axis_adjustability = 'unknown') AS pad_axis_adjustability,
                SUM(handle_reach_adjustability = 'unknown') AS handle_reach_adjustability
            FROM exercise_setup_fit_profile_v2
            """
        ).fetchone()
    )
    llm_authority_count = cursor.execute(
        "SELECT COUNT(*) FROM unsupported_input_policy_v2 WHERE llm_may_decide != 0"
    ).fetchone()[0]
    enabled_rule_count = cursor.execute(
        "SELECT COUNT(*) FROM constraint_execution_rule_v2 WHERE runtime_enabled != 0"
    ).fetchone()[0]
    shoulder_plane_distribution = {
        row[0]: row[1]
        for row in cursor.execute(
            """
            SELECT shoulder_elevation_plane, COUNT(*)
            FROM exercise_movement_requirement_profile_v2
            GROUP BY shoulder_elevation_plane
            ORDER BY shoulder_elevation_plane
            """
        )
    }
    bench_angle_distribution = {
        row[0]: row[1]
        for row in cursor.execute(
            """
            SELECT bench_angle_band, COUNT(*)
            FROM exercise_movement_requirement_profile_v2
            GROUP BY bench_angle_band
            ORDER BY bench_angle_band
            """
        )
    }
    path_type_distribution = {
        row[0]: row[1]
        for row in cursor.execute(
            """
            SELECT path_type, COUNT(*)
            FROM exercise_setup_fit_profile_v2
            GROUP BY path_type
            ORDER BY path_type
            """
        )
    }
    v1_incline_low_count = cursor.execute(
        """
        SELECT COUNT(*)
        FROM exercise_tier et
        JOIN exercise_mobility_requirement mr ON mr.exercise_id = et.id
        WHERE (
            lower(COALESCE(et.name_kr, '')) LIKE '%인클라인%'
            OR lower(COALESCE(et.name_en, '')) LIKE '%incline%'
        )
          AND mr.shoulder_flexion_demand = 'low'
        """
    ).fetchone()[0]
    v1_lateral_medium_flexion_count = cursor.execute(
        """
        SELECT COUNT(*)
        FROM exercise_tier et
        JOIN exercise_mobility_requirement mr ON mr.exercise_id = et.id
        WHERE (
            replace(lower(COALESCE(et.name_kr, '')), ' ', '') LIKE '%레터럴레이즈%'
            OR replace(lower(COALESCE(et.name_en, '')), ' ', '') LIKE '%lateralraise%'
        )
          AND mr.shoulder_flexion_demand = 'medium'
        """
    ).fetchone()[0]
    anthropometry_columns = {
        row[1]
        for row in cursor.execute(
            "PRAGMA table_info(exercise_anthropometry_sensitivity_profile)"
        )
    }
    v1_machine_good_adjustability_count = None
    if "machine_adjustability" in anthropometry_columns:
        v1_machine_good_adjustability_count = cursor.execute(
            """
            SELECT COUNT(*)
            FROM exercise_anthropometry_sensitivity_profile
            WHERE machine_adjustability = 'good'
            """
        ).fetchone()[0]
    return {
        "version": VERSION,
        "exerciseCount": exercise_count,
        "movementProfileCount": movement_count,
        "setupFitProfileCount": fit_count,
        "managedInjuryMovementProfileCount": injury_movement_count,
        "profileSelectableSignals": selectable,
        "profileSelectableSignalCount": len(selectable),
        "profileInputManifestCount": manifest_count,
        "profileInputContractVersion": manifest_version,
        "profileSelectableMovementSignals": movement_selectable,
        "profileSelectableBodyProportionSignals": body_proportion_selectable,
        "defaultSelectedProfileSignalCount": default_selected_count,
        "unknownAdjustabilityCounts": fit_unknown,
        "unsupportedPoliciesDelegatedToLlm": llm_authority_count,
        "runtimeEnabledCandidateRules": enabled_rule_count,
        "shoulderPlaneDistribution": shoulder_plane_distribution,
        "benchAngleDistribution": bench_angle_distribution,
        "pathTypeDistribution": path_type_distribution,
        "v1KnownGapCounts": {
            "inclineClassifiedLowShoulderDemand": v1_incline_low_count,
            "lateralRaiseClassifiedAsMediumFlexion": v1_lateral_medium_flexion_count,
            "genericMachineMarkedGoodAdjustability": v1_machine_good_adjustability_count,
        },
        "coveragePass": (
            exercise_count
            == movement_count
            == fit_count
            == injury_movement_count
        ),
        "managedInjuryMovementCoveragePass": (
            injury_movement_count == exercise_count
        ),
        "manifestCoveragePass": manifest_count == len(selectable),
    }


def write_markdown_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Closed Input Constraint DB v2 Audit",
        "",
        f"- Version: `{report['version']}`",
        f"- Exercise rows: `{report['exerciseCount']}`",
        f"- Movement profile rows: `{report['movementProfileCount']}`",
        f"- Setup/fit profile rows: `{report['setupFitProfileCount']}`",
        f"- Managed-injury movement profile rows: `{report['managedInjuryMovementProfileCount']}`",
        f"- Full-row coverage: `{'PASS' if report['coveragePass'] else 'FAIL'}`",
        f"- Profile input manifest coverage: `{'PASS' if report['manifestCoveragePass'] else 'FAIL'}`",
        f"- Profile input contract: `{report['profileInputContractVersion']}`",
        f"- Runtime-enabled candidate rules: `{report['runtimeEnabledCandidateRules']}`",
        f"- Unsupported policies delegated to LLM: `{report['unsupportedPoliciesDelegatedToLlm']}`",
        "",
        "## Profile Selectable Movement Signals",
        "",
    ]
    lines.extend(f"- `{code}`" for code in report["profileSelectableMovementSignals"])
    lines.extend([
        "",
        "## Profile Selectable Self-Reported Body-Proportion Signals",
        "",
    ])
    lines.extend(f"- `{code}`" for code in report["profileSelectableBodyProportionSignals"])
    lines.extend([
        "",
        f"Profile signals selected by default: `{report['defaultSelectedProfileSignalCount']}`",
        "",
        "Unchecked means that the user did not report the trait as notably present. It is not an objective finding that the user's proportions are average.",
    ])
    lines.extend([
        "",
        "## Deliberately Unknown Setup Fields",
        "",
    ])
    lines.extend(
        f"- `{field}`: `{count}`"
        for field, count in report["unknownAdjustabilityCounts"].items()
    )
    lines.extend([
        "",
        "## Candidate v2 Distributions",
        "",
        f"- Shoulder elevation plane: `{json.dumps(report['shoulderPlaneDistribution'], ensure_ascii=False)}`",
        f"- Bench angle band: `{json.dumps(report['benchAngleDistribution'], ensure_ascii=False)}`",
        f"- Path type: `{json.dumps(report['pathTypeDistribution'], ensure_ascii=False)}`",
        "",
        "## Quantified v1 Review Queue",
        "",
    ])
    lines.extend(
        f"- `{field}`: `{'not available' if count is None else count}`"
        for field, count in report["v1KnownGapCounts"].items()
    )
    lines.extend([
        "",
        "These unknown values are an evidence queue, not missing-value defaults. A machine name alone is not evidence that its seat, start position, pad axis, or handle reach is adjustable for every model and user.",
        "",
        "The v2 rules remain disabled until movement-family evidence, reviewed exceptions, and differential scenario tests pass the promotion gate.",
    ])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path(__file__).resolve().parents[1] / "fit_core.sqlite")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        report = install_closed_input_constraint_catalog_v2(conn)
    finally:
        conn.close()
    if args.report:
        write_markdown_report(report, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

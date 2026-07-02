from pathlib import Path

from scripts import run_candidate_quality_seed as candidate_seed
from scripts import run_gemma4_quality_seed as runner


V3_SEED = Path("tests/fixtures/gemma4_routine_quality_eval_seed_v3.jsonl")


def test_gemma4_quality_seed_v3_loads_and_summarizes():
    rows = runner.load_seed(V3_SEED)
    summary = runner.summarize_seed(rows)

    assert summary["scenario_count"] == 18
    assert summary["scenario_ids"][-6:] == [
        "knee-pain-limited-ankle-beginner-001",
        "wrist-limited-push-001",
        "low-readiness-home-short-fullbody-001",
        "shoulder-surgery-home-push-001",
        "upper-back-doms-lower-back-pull-short-001",
        "advanced-high-readiness-effect-safe-001",
    ]
    assert summary["hard_constraint_count"] >= 35
    assert summary["failure_pattern_count"] >= 45
    assert summary["rubric_item_count"] >= 45


def test_v3_template_is_report_safe():
    template = runner.build_results_template(
        seed_path=V3_SEED,
        output_dir=Path("tests/evaluation/.artifacts/gemma4-quality-seed/test-v3"),
        run_id="v3",
    )

    assert len(template["scenarios"]) == 18
    runner.assert_no_forbidden_tokens(template)


def test_candidate_tags_cover_v3_knee_ankle_beginner_case():
    row = {
        "scenarioId": "knee-pain-limited-ankle-beginner-001",
        "preferredPatterns": [
            "low_knee_shear_option",
            "machine_or_stable_option",
            "ankle_mobility_reason",
            "technique_risk_reason",
        ],
    }
    candidates = [
        {
            "id": "machine_leg_press",
            "name_kr": "머신 레그 프레스",
            "primary_muscle": "quadriceps",
            "equipment_req": "MACHINE",
            "knee_shear_load": "low",
            "ankle_dorsiflexion_demand": "low",
            "score_reasons": [
                "limited_ankle_dorsiflexion lower demand",
                "beginner low technique risk candidate",
                "knee_shear_load low",
            ],
        }
    ]

    tags = candidate_seed.derive_candidate_quality_tags(row, candidates)

    assert "low_knee_shear_option" in tags
    assert "machine_or_stable_option" in tags
    assert "ankle_mobility_reason" in tags
    assert "technique_risk_reason" in tags


def test_candidate_tags_cover_v3_wrist_push_case():
    row = {
        "scenarioId": "wrist-limited-push-001",
        "preferredPatterns": [
            "wrist_context_reason",
            "neutral_grip_press",
            "machine_chest_press",
            "dumbbell_when_available",
        ],
    }
    candidates = [
        {
            "id": "machine_chest_press_neutral",
            "name_kr": "머신 체스트 프레스 뉴트럴 그립",
            "primary_muscle": "chest",
            "equipment_req": "MACHINE, DUMBBELL",
            "wrist_extension_demand": "low",
            "wrist_stress": "low",
            "score_reasons": ["wrist extension context", "neutral grip option"],
        }
    ]

    tags = candidate_seed.derive_candidate_quality_tags(row, candidates)

    assert "wrist_context_reason" in tags
    assert "neutral_grip_press" in tags
    assert "machine_chest_press" in tags
    assert "dumbbell_when_available" in tags

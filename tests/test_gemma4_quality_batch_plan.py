from pathlib import Path

from scripts import plan_gemma4_quality_batches as batch_plan
from scripts import run_gemma4_quality_seed as seed_runner


V3_SEED = Path("tests/fixtures/gemma4_routine_quality_eval_seed_v3.jsonl")


def test_default_batch_plan_covers_v3_seed_once():
    seed_rows = seed_runner.load_seed(V3_SEED)
    plan = batch_plan.build_batch_plan(
        seed_path=V3_SEED,
        output_dir=Path("tests/evaluation/.artifacts/gemma4-quality-batches/unit"),
        run_prefix="unit",
        live_output_dir=Path("tests/evaluation/.artifacts/live"),
        score_output_dir=Path("tests/evaluation/.artifacts/score"),
        max_elapsed_ms=120000,
        scenario_timeout_sec=240,
        cooldown_sec=1,
        resume=True,
    )

    seed_ids = [row["scenarioId"] for row in seed_rows]
    planned_ids = [
        scenario_id
        for batch in plan["batches"]
        for scenario_id in batch["scenarioIds"]
    ]

    assert planned_ids != []
    assert sorted(planned_ids) == sorted(seed_ids)
    assert len(planned_ids) == len(set(planned_ids))
    assert plan["selectedScenarioCount"] == 18


def test_batch_plan_commands_include_latency_gate_and_selected_scenarios():
    plan = batch_plan.build_batch_plan(
        seed_path=V3_SEED,
        output_dir=Path("tests/evaluation/.artifacts/gemma4-quality-batches/unit"),
        run_prefix="unit",
        live_output_dir=Path("tests/evaluation/.artifacts/live"),
        score_output_dir=Path("tests/evaluation/.artifacts/score"),
        max_elapsed_ms=90000,
        scenario_timeout_sec=180,
        cooldown_sec=0.5,
        resume=False,
        selected_batch_ids=["time-equipment"],
    )

    assert plan["selectedBatchCount"] == 1
    batch = plan["batches"][0]
    assert batch["batchId"] == "time-equipment"
    assert "short-time-push-001" in batch["scenarioIds"]
    assert "--scenario-id" in batch["liveCommand"]
    assert "--max-elapsed-ms" in batch["scoreCommand"]
    assert "90000" in batch["scoreCommand"]
    assert "short-time-push-001" in batch["scoreShell"]


def test_batch_plan_markdown_is_report_safe():
    plan = batch_plan.build_batch_plan(
        seed_path=V3_SEED,
        output_dir=Path("tests/evaluation/.artifacts/gemma4-quality-batches/unit"),
        run_prefix="unit",
        live_output_dir=Path("tests/evaluation/.artifacts/live"),
        score_output_dir=Path("tests/evaluation/.artifacts/score"),
        max_elapsed_ms=120000,
        scenario_timeout_sec=240,
        cooldown_sec=1,
        resume=True,
        selected_batch_ids=["safety-core"],
    )
    markdown = batch_plan.render_markdown(plan)

    assert "Gemma4 Routine Quality Batch Plan" in markdown
    assert "safety-core" in markdown
    assert "run_gemma4_quality_seed_live.py" in markdown
    assert "score-file" in markdown


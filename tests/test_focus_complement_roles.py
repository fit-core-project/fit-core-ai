"""Requested direct focus must survive optional-slot omission, not fill time."""
import pytest

from engines.fallback import _blueprint_fallback_candidates
from engines.routine_blueprint import (
    blueprint_invariant_errors, build_blueprint_candidate_bundles, build_routine_blueprint,
)
from engines.routine_selection import adapt_slot_selection_output
from engines.schemas import LLMRoutineSelectionOutput


def blueprint(focus="calvesEmphasis", **overrides):
    args = dict(target_split_label="legs" if focus == "calvesEmphasis" else "pull",
                target_muscles=[], goal="hypertrophy", experience_level="advanced",
                readiness_level="normal", time_available_min=60, method_mode="off",
                specialization_focus=focus)
    args.update(overrides)
    return build_routine_blueprint(**args)


def row(eid, name, primary, *, equipment="MACHINE", movement="ISOLATION", **extra):
    return dict(id=eid, name_kr=name, primary_muscle=primary, equipment_req=equipment,
                movement_type=movement, main_routine_default_allowed=1, score=100, **extra)


def candidates(focus):
    if focus == "calvesEmphasis":
        return [row("squat", "핵 스쿼트", "quadriceps", movement="COMPOUND"),
                row("curl", "머신 레그 컬", "hamstring"),
                row("standing", "머신 카프 레이즈", "calves", name_en="Machine Standing Calf Raise"),
                row("seated", "머신 카프 레이즈", "calves", name_en="Machine Seated Calf Raise"),
                row("adduction", "머신 힙 어덕션", "adductor"),
                row("thrust", "덤벨 힙 쓰러스트", "gluteal", movement="COMPOUND", equipment="DUMBBELL")]
    return [row("row", "덤벨 로우", "upper-back", movement="COMPOUND", equipment="DUMBBELL"),
            row("pulldown", "랫풀다운", "upper-back", movement="COMPOUND"),
            row("shrug", "머신 슈러그", "trapezius"),
            row("db_shrug", "덤벨 슈러그", "trapezius", equipment="DUMBBELL"),
            row("curl", "머신 프리처 컬", "biceps"),
            row("wrist", "머신 리스트 컬", "forearm")]


def omitted_optional_selection(bp, rows):
    bundles = build_blueprint_candidate_bundles(bp, rows)
    selections = [{"slotId": b["slotId"], "exerciseId": b["candidates"][0]["exerciseId"]}
                  for b in bundles if b["requirement"] == "required_when_safe" and b["candidates"]]
    result = adapt_slot_selection_output(
        LLMRoutineSelectionOutput.model_validate({"selections": selections}),
        blueprint=bp, slot_candidate_bundles=bundles, ranked_candidates=rows)
    return result, bundles


@pytest.mark.parametrize("focus,slot_id,expected", [
    ("calvesEmphasis", "calves_accessory", {"standing", "seated"}),
    ("trapeziusEmphasis", "specialization_complement", {"shrug", "db_shrug"}),
])
def test_focus_complement_is_requested_and_repaired_when_model_omits_it(focus, slot_id, expected):
    bp = blueprint(focus)
    slot = next(s for s in bp.slots if s.slot_id == slot_id)
    assert slot.requirement == "preferred_when_requested_and_safe"
    result, bundles = omitted_optional_selection(bp, candidates(focus))
    assert expected <= {p.exercise_id for p in result.exercises}
    assert result._selection_diagnostics["requested_slot_repair_count"] >= 1
    assert slot_id in result._selection_slot_by_exercise_id.values()
    assert len(result.exercises) <= bp.target_exercise_count
    assert blueprint_invariant_errors(bp) == []
    assert any(b["slotId"] == slot_id and b["candidates"] for b in bundles)


@pytest.mark.parametrize("focus", ["calvesEmphasis", "trapeziusEmphasis"])
def test_missing_complement_candidate_never_duplicates_or_invents(focus):
    rows = [r for r in candidates(focus) if r["id"] not in {"seated", "db_shrug"}]
    bp = blueprint(focus)
    result, _ = omitted_optional_selection(bp, rows)
    ids = [p.exercise_id for p in result.exercises]
    assert len(ids) == len(set(ids))
    assert set(ids) <= {r["id"] for r in rows}
    assert len(result.exercises) <= bp.target_exercise_count


@pytest.mark.parametrize("focus,slot_id", [("calvesEmphasis", "calves_accessory"), ("trapeziusEmphasis", "specialization_complement")])
@pytest.mark.parametrize("overrides", [
    {"readiness_level": "low"}, {"time_available_min": 45}, {"goal": "strength"},
    {"session_structure": "superset"}, {"pain_areas": [{"bodyPart": "knees"}]},
    {"injury_conditions": [{"areaCode": "shoulder", "status": "managedWithRestrictions"}]},
])
def test_restricted_or_compact_sessions_do_not_promote_complement(focus, slot_id, overrides):
    bp = blueprint(focus, **overrides)
    assert all(s.requirement != "preferred_when_requested_and_safe" for s in bp.slots if s.slot_id == slot_id)


@pytest.mark.parametrize("focus,part", [("calvesEmphasis", "left-soleus"), ("trapeziusEmphasis", "upper-back")])
@pytest.mark.parametrize("level", [1, 2, 3])
def test_focus_doms_never_opens_extra_requested_role(focus, part, level):
    bp = blueprint(focus, doms_data={part: level})
    assert not any(s.requirement == "preferred_when_requested_and_safe" for s in bp.slots)


@pytest.mark.parametrize("focus", ["calvesEmphasis", "trapeziusEmphasis"])
def test_requested_adjunct_and_required_main_keep_capacity(focus):
    bp = blueprint(focus, adjunct_options=["cooldown_stretch"])
    assert blueprint_invariant_errors(bp) == []
    assert bp.target_exercise_count == 6
    assert "cooldown_stretch" in bp.planned_adjunct_options
    assert len(bp.slots) <= bp.target_exercise_count
    baseline = blueprint(focus, readiness_level="low")
    required = {s.slot_id for s in baseline.slots if s.requirement == "required_when_safe"}
    assert required <= {s.slot_id for s in bp.slots}


@pytest.mark.parametrize("focus,expected", [("calvesEmphasis", {"standing", "seated"}), ("trapeziusEmphasis", {"shrug", "db_shrug"})])
def test_fallback_uses_same_requested_focus_roles(focus, expected):
    bp = blueprint(focus)
    result = _blueprint_fallback_candidates(bp, candidates(focus))
    assert expected <= {r["id"] for r in result}
    assert len(result) <= bp.target_exercise_count


def test_plain_and_shorter_session_keep_original_optional_calf():
    for bp in (blueprint(specialization_focus=None), blueprint(time_available_min=45)):
        assert next(s for s in bp.slots if s.slot_id == "calves_accessory").requirement == "optional"


def test_same_calf_position_with_other_equipment_is_not_complement():
    rows = [r for r in candidates("calvesEmphasis") if r["id"] != "seated"]
    rows.append(row("duplicate", "덤벨 카프 레이즈", "calves", equipment="DUMBBELL", name_en="Dumbbell Standing Calf Raise"))
    result, _ = omitted_optional_selection(blueprint(), rows)
    assert not {"standing", "duplicate"} <= {p.exercise_id for p in result.exercises}


def test_trapezius_complement_stays_adjacent_before_other_accessories():
    bp = blueprint("trapeziusEmphasis")
    result, _ = omitted_optional_selection(bp, candidates("trapeziusEmphasis"))
    ids = [p.exercise_id for p in result.exercises]
    assert ids.index("db_shrug") == ids.index("shrug") + 1
    assert ids.index("db_shrug") < ids.index("curl")

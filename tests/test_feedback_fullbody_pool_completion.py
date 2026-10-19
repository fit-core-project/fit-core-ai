from engines.candidate_ranker import _ensure_full_body_vertical_press_candidate


def candidate(exercise_id, name, primary="front-deltoids", **extra):
    return {
        "id": exercise_id,
        "name_kr": name,
        "name_en": name,
        "primary_muscle": primary,
        "main_routine_default_allowed": 1,
        "mainstream_gym_score": 90,
        "exercise_familiarity_tier": "common",
        "novelty_level": "low",
        **extra,
    }


def preserve(selected, candidates, *, split="full_body", targets=None, top_n=3, **kwargs):
    return _ensure_full_body_vertical_press_candidate(
        selected=selected,
        sorted_candidates=candidates,
        target_muscles=targets or ["chest", "front-deltoids", "quadriceps"],
        target_split_label=split,
        top_n=top_n,
        specialization_focus=kwargs.pop("specialization_focus", None),
        adjunct_options=kwargs.pop("adjunct_options", None),
        **kwargs,
    )


def ids(result):
    return [item["id"] for item in result]


def test_full_body_preserves_qualified_vertical_press_over_front_raise():
    front_raise = candidate("raise", "Dumbbell Front Raise")
    press = candidate("press", "Dumbbell Shoulder Press", movement_pattern="vertical_press")

    result = preserve([front_raise, candidate("squat", "Hack Squat", "quadriceps"), candidate("row", "Machine Row", "upper-back")], [press, front_raise])

    assert "press" in ids(result)
    assert "raise" not in ids(result)
    assert len(result) == 3


def test_no_qualified_press_or_safe_same_primary_donor_preserves_pool():
    raise_candidate = candidate("raise", "Dumbbell Front Raise")
    unreviewed_press = candidate(
        "press", "Dumbbell Shoulder Press", movement_pattern="vertical_press",
        main_routine_default_allowed=0,
    )
    result = preserve([raise_candidate], [unreviewed_press, raise_candidate])
    assert result == [raise_candidate]

    result = preserve(
        [candidate("row", "Machine Row", "upper-back"), candidate("fly", "Machine Chest Fly", "chest")],
        [candidate("press", "Dumbbell Shoulder Press", movement_pattern="vertical_press"), candidate("fly", "Machine Chest Fly", "chest")],
        top_n=2,
    )
    assert ids(result) == ["row", "fly"]


def test_other_split_is_unchanged():
    raise_candidate = candidate("raise", "Dumbbell Front Raise")
    press = candidate("press", "Dumbbell Shoulder Press", movement_pattern="vertical_press")
    selected = [raise_candidate]
    assert preserve(selected, [press, raise_candidate], split="push", top_n=1) == selected


def test_denied_main_routine_press_is_not_unlocked():
    raise_candidate = candidate("raise", "Dumbbell Front Raise")
    press = candidate("press", "Dumbbell Shoulder Press", movement_pattern="vertical_press", main_routine_default_allowed=0)
    assert preserve([raise_candidate], [press, raise_candidate], top_n=1) == [raise_candidate]


def test_requested_adjunct_front_raise_is_protected():
    adjunct_raise = candidate("raise", "Mobility Front Raise", training_style_tags="mobility")
    press = candidate("press", "Dumbbell Shoulder Press", movement_pattern="vertical_press")
    result = preserve([adjunct_raise], [press, adjunct_raise], top_n=1, adjunct_options=["mobility_warmup"])
    assert ids(result) == ["raise"]


def test_focused_mandatory_front_raise_is_protected():
    focused_raise = candidate("raise", "Focused Front Raise", specialization_focus_codes="upperChest")
    press = candidate("press", "Dumbbell Shoulder Press", movement_pattern="vertical_press")
    result = preserve([focused_raise], [press, focused_raise], top_n=1, specialization_focus="upperChest")
    assert ids(result) == ["raise"]


def test_sole_other_primary_coverage_is_not_displaced():
    row = candidate("row", "Machine Row", "upper-back")
    raise_candidate = candidate("raise", "Dumbbell Front Raise")
    press = candidate("press", "Dumbbell Shoulder Press", movement_pattern="vertical_press")
    result = preserve([row, raise_candidate], [press, row, raise_candidate], top_n=2)
    assert ids(result) == ["row", "press"]


def test_result_is_bounded_unique_and_idempotent():
    raise_candidate = candidate("raise", "Dumbbell Front Raise")
    press = candidate("press", "Dumbbell Shoulder Press", movement_pattern="vertical_press")
    selected = [raise_candidate, candidate("row", "Machine Row", "upper-back")]
    result = preserve(selected, [press, *selected], top_n=2)
    again = preserve(result, [press, *selected], top_n=2)
    assert len(result) == 2
    assert len({item["id"] for item in result}) == 2
    assert ids(again) == ids(result)

from engines.candidate_ranker import _ensure_specialization_focus_candidate


def _candidate(
    exercise_id: str,
    name: str,
    *,
    primary: str,
    score: int,
    movement_type: str = "ISOLATION",
    hip_hinge_load: str = "low",
) -> dict:
    return {
        "id": exercise_id,
        "name_kr": name,
        "name_en": name,
        "primary_muscle": primary,
        "movement_type": movement_type,
        "hip_hinge_load": hip_hinge_load,
        "score": score,
        "efficiency_tier": 3,
        "main_routine_default_allowed": 1,
        "mainstream_gym_score": 90,
        "exercise_familiarity_tier": "common",
        "novelty_level": "low",
    }


def _ids(candidates: list[dict]) -> list[str]:
    return [candidate["id"] for candidate in candidates]


def test_hamstring_focus_keeps_admitted_direct_hinge_without_losing_sole_other_target():
    squat = _candidate("squat", "Barbell Squat", primary="quadriceps", score=120, movement_type="COMPOUND")
    leg_curl = _candidate("leg-curl", "Machine Leg Curl", primary="hamstring", score=110)
    rdl = _candidate(
        "rdl",
        "Barbell Romanian Deadlift",
        primary="hamstring",
        score=100,
        movement_type="COMPOUND",
        hip_hinge_load="high",
    )

    result = _ensure_specialization_focus_candidate(
        selected=[squat, leg_curl],
        sorted_candidates=[squat, leg_curl, rdl],
        specialization_focus="hamstringEmphasis",
        adjunct_options=[],
        target_muscles=["quadriceps", "hamstring"],
        top_n=2,
    )

    assert set(_ids(result)) == {"squat", "leg-curl"}


def test_hamstring_focus_replaces_redundant_focus_candidate_with_admitted_direct_hinge():
    leg_curl = _candidate("leg-curl", "Machine Leg Curl", primary="hamstring", score=120)
    seated_curl = _candidate("seated-curl", "Seated Leg Curl", primary="hamstring", score=115)
    rdl = _candidate(
        "rdl",
        "Barbell Romanian Deadlift",
        primary="hamstring",
        score=100,
        movement_type="COMPOUND",
        hip_hinge_load="high",
    )

    result = _ensure_specialization_focus_candidate(
        selected=[leg_curl, seated_curl],
        sorted_candidates=[leg_curl, seated_curl, rdl],
        specialization_focus="hamstringEmphasis",
        adjunct_options=[],
        target_muscles=["hamstring"],
        top_n=2,
    )

    assert "rdl" in _ids(result)
    assert "leg-curl" in _ids(result)


def test_biceps_focus_reserves_three_distinct_curl_patterns_and_preserves_sole_triceps():
    preacher_machine = _candidate("preacher-machine", "Machine Preacher Curl", primary="biceps", score=120)
    preacher_barbell = _candidate("preacher-barbell", "Barbell Preacher Curl", primary="biceps", score=118)
    preacher_dumbbell = _candidate("preacher-dumbbell", "Dumbbell Preacher Curl", primary="biceps", score=116)
    hammer = _candidate("hammer", "Dumbbell Hammer Curl", primary="biceps", score=110)
    incline = _candidate("incline", "Cable Incline Curl", primary="biceps", score=105)
    pushdown = _candidate("pushdown", "Cable Pushdown", primary="triceps", score=100)

    result = _ensure_specialization_focus_candidate(
        selected=[preacher_machine, preacher_barbell, preacher_dumbbell, hammer, pushdown],
        sorted_candidates=[preacher_machine, preacher_barbell, preacher_dumbbell, hammer, incline, pushdown],
        specialization_focus="bicepsEmphasis",
        adjunct_options=[],
        target_muscles=["biceps", "triceps"],
        top_n=5,
    )

    assert {"preacher-machine", "hammer", "incline", "pushdown"} <= set(_ids(result))
    assert len(result) == 5


def test_forearm_focus_replaces_equipment_clone_with_admitted_reverse_wrist_pattern():
    wrist_machine = _candidate("wrist-machine", "Machine Wrist Curl", primary="forearm", score=120)
    wrist_barbell = _candidate("wrist-barbell", "Barbell Wrist Curl", primary="forearm", score=118)
    reverse_wrist = _candidate("reverse-wrist", "Cable Reverse Wrist Curl", primary="forearm", score=105)
    preacher = _candidate("preacher", "Machine Preacher Curl", primary="biceps", score=110)
    pushdown = _candidate("pushdown", "Cable Pushdown", primary="triceps", score=100)

    result = _ensure_specialization_focus_candidate(
        selected=[wrist_machine, wrist_barbell, preacher, pushdown],
        sorted_candidates=[wrist_machine, wrist_barbell, preacher, reverse_wrist, pushdown],
        specialization_focus="forearmEmphasis",
        adjunct_options=[],
        target_muscles=["biceps", "triceps", "forearm"],
        top_n=4,
    )

    assert {"wrist-machine", "reverse-wrist", "preacher", "pushdown"} == set(_ids(result))


def test_nonfocus_pool_is_unchanged():
    selected = [
        _candidate("press", "Machine Chest Press", primary="chest", score=120, movement_type="COMPOUND"),
        _candidate("fly", "Cable Fly", primary="chest", score=100),
    ]

    result = _ensure_specialization_focus_candidate(
        selected=selected,
        sorted_candidates=selected,
        specialization_focus=None,
        adjunct_options=[],
        target_muscles=["chest"],
        top_n=2,
    )

    assert result == selected

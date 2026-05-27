from unittest.mock import MagicMock

from engines.db_queries import get_candidate_exercises


class _Row:
    def __init__(self, mapping):
        self._mapping = mapping


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


def test_blocked_equipment_adds_safe_mapped_substitute_candidate():
    db = MagicMock()
    db.execute.side_effect = [
        _Result([
            _Row({
                "id": 1,
                "name_kr": "바벨 벤치프레스",
                "name_en": "Barbell Bench Press",
                "primary_muscle": "chest",
                "secondary_muscle": "triceps",
                "equipment_req": "BARBELL",
                "difficulty_tier": 3,
                "efficiency_tier": 1,
                "pain_triggers": None,
                "movement_type": "COMPOUND",
                "substitute_exercise_ids": "2",
            }),
        ]),
        _Result([
            _Row({
                "id": 2,
                "name_kr": "덤벨 벤치프레스",
                "name_en": "Dumbbell Bench Press",
                "primary_muscle": "chest",
                "secondary_muscle": "triceps",
                "equipment_req": "DUMBBELL",
                "difficulty_tier": 3,
                "efficiency_tier": 2,
                "pain_triggers": None,
                "movement_type": "COMPOUND",
                "substitute_exercise_ids": None,
            }),
        ]),
    ]

    candidates = get_candidate_exercises(
        db,
        target_muscles=["chest"],
        unavailable_equipment=["BARBELL"],
        pain_areas=[],
    )

    assert [candidate["id"] for candidate in candidates] == [2]
    assert "mapped substitute" in candidates[0]["score_reasons"][0]

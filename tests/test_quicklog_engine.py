import json

from engines.quicklog import nlp_engine
from engines.quicklog.nlp_engine import ParsedDailyLog, parse_natural_language_log


class _StructuredSuccess:
    def invoke(self, payload):
        return ParsedDailyLog(
            diet_logs=[],
            workout_logs=[],
            overall_summary="parsed",
        )


class _StructuredFailure:
    def invoke(self, payload):
        raise RuntimeError("raw user text should not leak")


class _LlmSuccess:
    def with_structured_output(self, schema):
        return _StructuredSuccess()


class _LlmFailure:
    def with_structured_output(self, schema):
        return _StructuredFailure()


def test_quicklog_parse_success_mock(monkeypatch):
    monkeypatch.setattr(nlp_engine, "get_llm", lambda *args, **kwargs: _LlmSuccess())

    payload = json.loads(parse_natural_language_log("bench 80kg 3 sets 8 reps"))

    assert ParsedDailyLog.model_validate(payload).overall_summary == "parsed"


def test_quicklog_llm_failure_returns_fallback_json(monkeypatch, capsys):
    monkeypatch.setattr(nlp_engine, "get_llm", lambda *args, **kwargs: _LlmFailure())

    payload = json.loads(parse_natural_language_log("bench 80kg 3 sets 8 reps"))

    parsed = ParsedDailyLog.model_validate(payload)
    assert parsed.overall_summary
    assert parsed.workout_logs[0].weight_kg == 80
    assert "raw user text should not leak" not in capsys.readouterr().out


def test_quicklog_deterministic_korean_workout_and_diet(monkeypatch):
    monkeypatch.setattr(nlp_engine, "get_llm", lambda *args, **kwargs: _LlmFailure())

    payload = json.loads(parse_natural_language_log("오늘 벤치프레스 60kg 10회 3세트 하고 닭가슴살 200g 먹었어."))

    parsed = ParsedDailyLog.model_validate(payload)
    assert parsed.workout_logs[0].exercise_name == "벤치프레스"
    assert parsed.workout_logs[0].weight_kg == 60
    assert parsed.workout_logs[0].reps == 10
    assert parsed.workout_logs[0].sets == 3
    assert parsed.diet_logs[0].food_name == "닭가슴살"
    assert parsed.diet_logs[0].amount == 200
    assert parsed.diet_logs[0].unit == "g"
    assert parsed.overall_summary != nlp_engine._FALLBACK_SUMMARY


def test_quicklog_deterministic_common_patterns(monkeypatch):
    monkeypatch.setattr(nlp_engine, "get_llm", lambda *args, **kwargs: _LlmFailure())

    squat = ParsedDailyLog.model_validate(json.loads(parse_natural_language_log("스쿼트 80kg 5회 5세트")))
    pushup = ParsedDailyLog.model_validate(json.loads(parse_natural_language_log("푸쉬업 20회 3세트")))
    egg = ParsedDailyLog.model_validate(json.loads(parse_natural_language_log("계란 2개 먹음")))

    assert squat.workout_logs[0].exercise_name == "스쿼트"
    assert squat.workout_logs[0].weight_kg == 80
    assert squat.workout_logs[0].reps == 5
    assert squat.workout_logs[0].sets == 5
    assert pushup.workout_logs[0].exercise_name == "푸쉬업"
    assert pushup.workout_logs[0].weight_kg is None
    assert pushup.workout_logs[0].reps == 20
    assert pushup.workout_logs[0].sets == 3
    assert egg.diet_logs[0].food_name == "계란"
    assert egg.diet_logs[0].amount == 2
    assert egg.diet_logs[0].unit == "개"


def test_quicklog_empty_text_returns_fallback_json():
    payload = json.loads(parse_natural_language_log(""))

    parsed = ParsedDailyLog.model_validate(payload)
    assert parsed.diet_logs == []
    assert parsed.workout_logs == []
    assert parsed.overall_summary


def test_old_quicklog_import_shim_still_works():
    from engines.nlp_engine import ParsedDailyLog as ShimParsedDailyLog

    assert ShimParsedDailyLog is ParsedDailyLog

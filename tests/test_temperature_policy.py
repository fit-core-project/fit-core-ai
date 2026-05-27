from engines.temperature_policy import resolve_generation_temperature


def test_low_readiness_returns_low_temp_when_enabled():
    assert resolve_generation_temperature("low", _override_enabled=True) == 0.0


def test_normal_readiness_returns_current_default_when_enabled():
    assert resolve_generation_temperature("normal", _override_enabled=True) == 0.0


def test_high_readiness_returns_limited_variety_temp_when_enabled():
    assert resolve_generation_temperature("high", _override_enabled=True) == 0.2


def test_missing_readiness_returns_default():
    assert resolve_generation_temperature(None, _override_enabled=True) == 0.0


def test_invalid_readiness_returns_default():
    assert resolve_generation_temperature("extreme", _override_enabled=True) == 0.0


def test_empty_string_readiness_returns_normal_map():
    assert resolve_generation_temperature("", _override_enabled=True) == 0.0


def test_flag_off_ignores_low_readiness():
    assert resolve_generation_temperature("low", _override_enabled=False) == 0.0


def test_flag_off_ignores_high_readiness():
    assert resolve_generation_temperature("high", _override_enabled=False) == 0.0


def test_flag_off_ignores_normal_readiness():
    assert resolve_generation_temperature("normal", _override_enabled=False) == 0.0


def test_custom_default_temperature_respected_for_missing_readiness():
    assert resolve_generation_temperature(None, default_temperature=0.5, _override_enabled=True) == 0.5


def test_case_insensitive_readiness():
    assert resolve_generation_temperature("HIGH", _override_enabled=True) == 0.2

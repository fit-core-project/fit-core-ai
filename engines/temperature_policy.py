"""Readiness-based LLM temperature policy for routine generation."""
from __future__ import annotations

import os
from typing import Optional

_DYNAMIC_ENABLED: bool = os.getenv("ENABLE_DYNAMIC_TEMPERATURE", "false").lower() == "true"

_READINESS_TEMPERATURE_MAP: dict[str, float] = {
    "low": 0.0,
    "normal": 0.0,
    "high": 0.2,
}

_DEFAULT_TEMPERATURE: float = 0.0


def is_dynamic_temperature_enabled() -> bool:
    return _DYNAMIC_ENABLED


def resolve_generation_temperature(
    readiness_level: Optional[str],
    default_temperature: float = _DEFAULT_TEMPERATURE,
    *,
    _override_enabled: Optional[bool] = None,
) -> float:
    """Return generation temperature for the current readiness level.

    Dynamic temperature is feature-flagged off by default, preserving the
    existing deterministic routine generation behavior.
    """
    enabled = _DYNAMIC_ENABLED if _override_enabled is None else _override_enabled
    if not enabled:
        return default_temperature

    if readiness_level is None:
        return default_temperature

    normalized = readiness_level.strip().lower()
    if not normalized:
        normalized = "normal"
    return _READINESS_TEMPERATURE_MAP.get(normalized, default_temperature)

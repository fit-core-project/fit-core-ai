"""Candidate pool size policy for routine generation."""
from __future__ import annotations

import os
from typing import Optional

ENV_CANDIDATE_POOL_SIZE = "ROUTINE_CANDIDATE_POOL_SIZE"
DEFAULT_CANDIDATE_POOL_SIZE = 12
MIN_CANDIDATE_POOL_SIZE = 1
MAX_CANDIDATE_POOL_SIZE = 18


def resolve_candidate_pool_size(
    value: Optional[str],
    default: int = DEFAULT_CANDIDATE_POOL_SIZE,
) -> int:
    """Return a safe candidate pool size.

    Invalid or out-of-range values fall back to the production default instead
    of being clamped, keeping rollout behavior explicit and predictable.
    """
    if value is None:
        return default

    normalized = value.strip()
    if not normalized:
        return default

    try:
        parsed = int(normalized)
    except ValueError:
        return default

    if parsed < MIN_CANDIDATE_POOL_SIZE or parsed > MAX_CANDIDATE_POOL_SIZE:
        return default
    return parsed


def get_candidate_pool_size() -> int:
    return resolve_candidate_pool_size(os.getenv(ENV_CANDIDATE_POOL_SIZE))

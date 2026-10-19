from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_PATHS = (
    Path("scripts/routine_eval_provenance.py"),
    Path("scripts/build_constraints_db.py"),
    Path("scripts/build_closed_input_constraint_catalog_v2.py"),
    Path("database.py"),
    Path("data/provenance/v25_canonical_filter_core.csv"),
    Path("engines/prompt_builder.py"),
    Path("engines/candidate_ranker.py"),
    Path("engines/db_queries.py"),
    Path("engines/equipment_policy.py"),
    Path("engines/experience_policy.py"),
    Path("engines/muscle_mapping.py"),
    Path("engines/muscle_taxonomy.py"),
    Path("engines/option_resolver.py"),
    Path("engines/profile_constraint_contract.py"),
    Path("engines/routine_blueprint.py"),
    Path("engines/registry.py"),
    Path("engines/routine_pipeline.py"),
    Path("engines/routine_selection.py"),
    Path("engines/llm_parser.py"),
    Path("engines/llm_router.py"),
    Path("engines/korean_text_guard.py"),
    Path("engines/fallback.py"),
    Path("engines/prescription/adjuncts.py"),
    Path("engines/prescription/adjustments.py"),
    Path("engines/prescription/estimator.py"),
    Path("engines/prescription/experience.py"),
    Path("engines/prescription/metadata.py"),
    Path("engines/prescription/params.py"),
    Path("engines/prescription/targets.py"),
    Path("engines/prescription/underfill.py"),
    Path("engines/rule_critic.py"),
    Path("engines/routine_telemetry.py"),
    Path("engines/schemas.py"),
    Path("fit_core.sqlite"),
)


def _git_output(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


@lru_cache(maxsize=8)
def git_snapshot(repo_root: Path = PROJECT_ROOT) -> dict[str, object]:
    status = _git_output(repo_root, "status", "--porcelain", "--untracked-files=normal")
    dirty_paths = [line for line in (status or "").splitlines() if line.strip()]
    return {
        "branch": _git_output(repo_root, "branch", "--show-current") or "unknown",
        "commit": _git_output(repo_root, "rev-parse", "HEAD") or "unknown",
        "dirty": bool(dirty_paths),
        "dirtyPathCount": len(dirty_paths),
    }


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


@lru_cache(maxsize=64)
def file_fingerprint(path: Path, repo_root: Path = PROJECT_ROOT) -> dict[str, object]:
    resolved = path if path.is_absolute() else repo_root / path
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": _display_path(resolved, repo_root),
        "sha256": digest.hexdigest(),
        "sizeBytes": resolved.stat().st_size,
    }


def build_provenance(
    *,
    runner: str,
    input_paths: Iterable[Path] = (),
    asset_paths: Iterable[Path] = DEFAULT_ASSET_PATHS,
    model: str | None = None,
    repo_root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    fingerprints: dict[str, dict[str, object]] = {}
    seen: set[Path] = set()
    # The runner is part of the evidence oracle and must be reproducible too.
    runner_path = Path(str(runner).strip().split(maxsplit=1)[0])
    for raw_path in [runner_path, *input_paths, *asset_paths]:
        resolved = raw_path if raw_path.is_absolute() else repo_root / raw_path
        resolved = resolved.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        fingerprint = file_fingerprint(resolved, repo_root)
        fingerprints[str(fingerprint["path"])] = fingerprint

    return {
        "schemaVersion": "routine-eval-provenance-v2",
        "capturedAtUtc": datetime.now(timezone.utc).isoformat(),
        "runner": runner,
        "model": model or os.getenv("LOCAL_LLM_MODEL") or "not_applicable",
        "pythonVersion": platform.python_version(),
        "git": git_snapshot(repo_root),
        "files": fingerprints,
    }

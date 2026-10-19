"""Build a credential-free temporary core fixture package and execute BF01-20.

Usage: python -B tests/run_stage2_portable_oracle_v1.py --out <new-evidence-dir>
Uses the current interpreter's installed dependencies; no dependency installation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

CORE = Path(__file__).resolve().parents[1]
MANIFEST = Path("tests/fixtures/stage2_fallback_oracle_v1.json")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def matches(selector, case):
    parts = selector.split("::")
    prefix = parts[0][:-3].replace("/", ".")
    label = case.attrib["classname"] + "." + case.attrib["name"].split("[")[0]
    if len(parts) > 1:
        prefix += "." + ".".join(parts[1:])
    return label == prefix or label.startswith(prefix + ".")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((CORE / MANIFEST).read_text())
    assert [g["id"] for g in manifest["groups"]] == [f"BF{i:02}" for i in range(1, 21)]
    all_selectors = sorted(set(manifest["supplemental"] + [s for g in manifest["groups"] for s in g["selectors"]]))
    selectors = [s for s in all_selectors if not any(s.startswith(other + "::") for other in all_selectors)]
    selected_files = {s.split("::")[0] for s in selectors}
    paths = {Path(p) for p in selected_files} | {MANIFEST, Path("database.py"), Path("fit_core.sqlite"),
        Path("tests/conftest.py"), Path("tests/stage2_offline_guard.py"),
        Path("docs/evidence/glute-coverage-ab-v1-2026-10-10/after/responses.json")}
    for folder, suffixes in (("engines", {".py", ".json"}), ("models", {".py"}),
                             ("data", {".csv", ".json", ".jsonl", ".md"}),
                             ("tests/fixtures", {".json", ".jsonl"})):
        paths.update(p.relative_to(CORE) for p in (CORE / folder).rglob("*")
                     if p.is_file() and p.suffix in suffixes and "__pycache__" not in p.parts)
    hashes = {str(p): digest(CORE / p) for p in sorted(paths)}
    write(out / "package-hashes.json", hashes)
    write(out / "manifest.json", manifest)
    with tempfile.TemporaryDirectory(prefix="fitcore-stage2-portable-") as temp:
        package = Path(temp)
        for path in sorted(paths):
            source, target = CORE / path, package / path
            assert not source.is_symlink()
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        assert {str(p): digest(package / p) for p in sorted(paths)} == hashes
        env = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "SYSTEMROOT") if key in os.environ}
        env.update(PYTHONDONTWRITEBYTECODE="1", PYTHON_DOTENV_DISABLED="1", LLM_PROVIDER="local",
                   PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONPATH=f"{package}:{package / 'tests'}",
                   DATABASE_URL=f"sqlite:///file:{package / 'fit_core.sqlite'}?mode=ro&uri=true",
                   FITCORE_SQLITE_DB_PATH=str(package / "fit_core.sqlite"))
        proof_code = (
            "import json,sys; from pathlib import Path; from engines import db_queries; "
            "from engines.routine_response_v4 import CONTRACT_VERSION; "
            "assert Path(db_queries.__file__).resolve().is_relative_to(Path.cwd()); "
            f"assert CONTRACT_VERSION == {manifest['contractVersion']!r}; "
            "print(json.dumps({'module':db_queries.__file__,'cwd':str(Path.cwd()),'contract':CONTRACT_VERSION}))"
        )
        proof = subprocess.run([sys.executable, "-B", "-c", proof_code], cwd=package, env=env,
                               text=True, capture_output=True, timeout=120)
        (out / "import-proof.txt").write_text(proof.stdout + proof.stderr)
        assert proof.returncode == 0, "Portable import failed; see import-proof.txt"
        command = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                   "-p", "stage2_offline_guard", f"--junitxml={out / 'pytest.xml'}", *selectors]
        write(out / "command.json", command)
        with (out / "runner.txt").open("x") as log:
            result = subprocess.run(command, cwd=package, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900)
        cases = list(ET.parse(out / "pytest.xml").iter("testcase"))
        failed = [c for c in cases if c.find("failure") is not None or c.find("error") is not None]
        skipped = [c for c in cases if c.find("skipped") is not None]
        groups = []
        for group in manifest["groups"]:
            matched = [c for c in cases if any(matches(s, c) for s in group["selectors"])]
            uncovered = [s for s in group["selectors"] if not any(matches(s, c) for c in matched)]
            groups.append({"id": group["id"], "testCount": len(matched), "unexecutedSelectors": uncovered,
                           "passed": bool(matched) and not uncovered and all(c not in failed + skipped for c in matched)})
        unchanged = hashes == {str(p): digest(package / p) for p in sorted(paths)}
        source_unchanged = hashes == {str(p): digest(CORE / p) for p in sorted(paths)}
        summary = {"exitCode": result.returncode, "testCount": len(cases), "passed": len(cases)-len(failed)-len(skipped),
                   "failed": [c.attrib for c in failed], "skipped": len(skipped), "groups": groups,
                   "packageUnchanged": unchanged, "sourceUnchanged": source_unchanged,
                   "network": "blocked by pytest plugin", "beIntegrationTested": False,
                   "dependencyEnvironment": "existing interpreter; fresh dependency installation not tested"}
        write(out / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        assert result.returncode == 0 and unchanged and source_unchanged and all(g["passed"] for g in groups)


if __name__ == "__main__":
    main()

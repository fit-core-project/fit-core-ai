"""Run the pinned migration checks offline using installed dependencies.

This does not run live models or prove BE/FE integration. Known failures remain
real pytest failures; the process returns pytest's exit code without suppression.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests/fixtures/core_migration_checks_v1.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("core", "collaboration", "numeric", "oracle"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    selectors = manifest["suites"][args.suite]
    for selector in selectors:
        path = (ROOT / selector.split("::")[0]).resolve()
        if not path.is_relative_to(ROOT / "tests") or not path.is_file():
            parser.error(f"Invalid test selector: {selector}")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = {k: os.environ[k] for k in ("PATH", "LANG", "LC_ALL", "HOME", "SYSTEMROOT") if k in os.environ}
    env.update(
        PYTHONDONTWRITEBYTECODE="1", PYTHON_DOTENV_DISABLED="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", LLM_PROVIDER="local",
        SUPPLEMENT_RAG_PRELOAD="false", AI_PRELOAD_MODELS="false",
        PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "tests"))),
        DATABASE_URL=f"sqlite:///file:{ROOT / 'fit_core.sqlite'}?mode=ro&uri=true",
        FITCORE_SQLITE_DB_PATH=str(ROOT / "fit_core.sqlite"),
    )
    command = [sys.executable, "-B", "-m", "pytest", "-v", "-s",
        "-p", "no:cacheprovider", "-p", "stage2_offline_guard",
        "-o", "faulthandler_timeout=60", f"--junitxml={out / 'pytest.xml'}", *selectors]
    with (out / "output.txt").open("x", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
            stderr=subprocess.STDOUT, timeout=1800, check=False)
    cases = list(ET.parse(out / "pytest.xml").iter("testcase"))
    failed = [c.attrib for c in cases if c.find("failure") is not None or c.find("error") is not None]
    skipped = [c.attrib for c in cases if c.find("skipped") is not None]
    summary = {"suite": args.suite, "exitCode": result.returncode, "count": len(cases),
        "passed": len(cases) - len(failed) - len(skipped), "failed": failed, "skipped": skipped,
        "network": "blocked by pytest plugin", "catalog": "read-only connection",
        "liveModelTested": False, "beIntegrationTested": False}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

"""Check the package or BE-exported full JSON values without private dependencies."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def index(rows):
    result = {row["id"]: row["response"] for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate fixture IDs")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual", type=Path, help="BE round-trip array of {id,response}")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = read(root / "package-manifest.json")
    files = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if any(path.is_symlink() for path in root.rglob("*")):
        raise SystemExit("Symlinks are not allowed in the handoff package")
    if files != set(manifest["sha256"]) | {"package-manifest.json"}:
        raise SystemExit("Package file list mismatch")
    for path, expected in manifest["sha256"].items():
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise SystemExit("Unsafe manifest path")
        actual = hashlib.sha256((root / path).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit("Package hash mismatch: " + path)
    expected = index(read(root / "fixtures/transport-golden.json"))
    if args.actual:
        actual = index(read(args.actual))
        differences = sorted(key for key in set(expected) | set(actual) if expected.get(key) != actual.get(key))
        if differences:
            raise SystemExit("Round-trip mismatch: " + ", ".join(differences))
    assert len(expected) == 13
    assert expected["short-20-of-50"]["shortRoutineNotice"]["isShortRoutine"] is True
    assert expected["exact-half"]["shortRoutineNotice"]["isShortRoutine"] is False
    assert expected["failed-empty"]["routine"]["totalEstimatedTime"] == 0
    detail = expected["detail"]["routine"]["routineBlocks"][0]
    assert detail["primaryMuscles"] == [] and detail["primaryMuscleDetails"] == ["side-deltoids"]
    bridge = read(root / "fixtures/catalog-identity-bridge.json")
    assert sum(edge["status"] == "approved" for edge in bridge["edges"]) == 11
    assert [g["id"] for g in read(root / "fixtures/bf01-20-oracle.json")["groups"]] == [f"BF{i:02}" for i in range(1, 21)]
    print(json.dumps({"packageIntegrity": "passed", "transportCases": len(expected),
        "beRoundTrip": "passed for supplied JSON; verify its Java/DB provenance separately" if args.actual else "not tested",
        "clinicalSafety": "not assessed"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

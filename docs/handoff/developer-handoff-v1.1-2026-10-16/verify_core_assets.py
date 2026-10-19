"""Validate packaged model artifacts against a supplied core without model calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", required=True, type=Path)
    args = parser.parse_args()
    root, core = Path(__file__).resolve().parent, args.core.resolve()
    for path, expected in read(root / "asset-provenance.json")["sourceHashes"].items():
        assert hashlib.sha256((core / path).read_bytes()).hexdigest() == expected, path
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(core))
    from engines.public_contract import PublicRoutineGenerateRequest
    from engines.routine_response_v4 import RoutineResponseV4
    from engines.schemas import RoutineRequest
    for name, model, mode in (
        ("public-request", PublicRoutineGenerateRequest, "validation"),
        ("internal-request", RoutineRequest, "validation"),
        ("response-v4", RoutineResponseV4, "serialization"),
    ):
        assert model.model_json_schema(by_alias=True, mode=mode) == read(root / "schemas" / (name + ".schema.json"))
    responses = read(root / "fixtures/transport-golden.json")
    for row in responses:
        value = row["response"]
        assert RoutineResponseV4.model_validate(value).model_dump(mode="json", by_alias=True) == value, row["id"]
    requests = read(root / "fixtures/public-input-golden.json")
    for row in requests:
        try:
            value = PublicRoutineGenerateRequest.model_validate(row["request"])
        except ValueError:
            assert not row["accepted"], row["id"]
        else:
            assert row["accepted"], row["id"]
            payload = value.to_internal_request_payload(user_id="synthetic-server-user")
            assert payload == row["expectedInternalPayload"], row["id"]
            RoutineRequest.model_validate(payload)
    print(json.dumps({"sourceSchemaChecks": 3, "responseRoundTrips": len(responses),
                      "inputChecks": len(requests), "passed": True, "liveCalls": 0}))


if __name__ == "__main__":
    main()

"""Check approved error examples and negative mutations, without an HTTP server."""
from copy import deepcopy
import json
from pathlib import Path


def validate(row):
    response = row["response"]
    assert set(response) == {"error"}
    error = response["error"]
    required = {"code", "messageKo", "retryable", "requestId"}
    optional = {"requiredContractVersion", "unsupportedFields"}
    assert required <= set(error) <= required | optional
    assert all(isinstance(error[key], str) and error[key] for key in required - {"retryable"})
    assert type(error["retryable"]) is bool
    if "requiredContractVersion" in error:
        assert error["requiredContractVersion"] == "routine-core-v4-draft-v3"
    if "unsupportedFields" in error:
        fields = error["unsupportedFields"]
        assert isinstance(fields, list) and fields
        assert all(isinstance(field, str) and field for field in fields)
        assert len(fields) == len(set(fields))
    if error["code"] == "PROFILE_LOOKUP_UNAVAILABLE":
        assert row["httpStatus"] == 503 and error["retryable"] is True
    elif error["code"] == "UNSUPPORTED_CONTRACT_CAPABILITY":
        assert row["httpStatus"] == 422 and error["retryable"] is False
        assert optional <= set(error)
    else:
        raise AssertionError("This verifier covers only the two approved codes")


def main():
    root = Path(__file__).resolve().parent
    rows = json.loads((root / "fixtures/external-error-golden.json").read_text(encoding="utf-8"))
    assert len(rows) == 2
    for row in rows:
        validate(row)
    mutations = []
    for index, key, value in [(0, "retryable", False), (0, "retryable", "true"),
                              (1, "unsupportedFields", []), (1, "requiredContractVersion", "v4"),
                              (0, "requestId", ""), (0, "stack", "synthetic-debug")]:
        row = deepcopy(rows[index])
        row["response"]["error"][key] = value
        mutations.append(row)
    row = deepcopy(rows[0])
    row["httpStatus"] = 200
    mutations.append(row)
    row = deepcopy(rows[1])
    del row["response"]["error"]["unsupportedFields"]
    mutations.append(row)
    for row in mutations:
        try:
            validate(row)
        except AssertionError:
            continue
        raise AssertionError("Invalid error fixture accepted")
    print(json.dumps({"syntheticErrorCases": len(rows), "negativeCasesRejected": len(mutations),
                      "actualHttpTested": False, "passed": True}))


if __name__ == "__main__":
    main()

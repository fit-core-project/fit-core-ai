"""Request-scoped decoding constraints; final selection validation stays server-side."""
from typing import Any, Mapping, Sequence


def build_selection_decoding_schema(bundles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    branches = []
    for bundle in bundles:
        slot_id = bundle["slotId"]
        # Flex is server-only and is deliberately absent from the model prompt.
        if slot_id == "flex":
            continue
        ids = sorted({candidate["exerciseId"] for candidate in bundle["candidates"]})
        if ids:
            branches.append({
                "type": "object",
                "properties": {
                    "slotId": {"const": slot_id},
                    "exerciseId": {"type": "string", "enum": ids},
                },
                "required": ["slotId", "exerciseId"],
                "additionalProperties": False,
            })
    return {
        "type": "object",
        "properties": {
            "selections": {
                "type": "array",
                "maxItems": len(branches),
                "items": {"anyOf": branches} if branches else {},
            },
        },
        "required": ["selections"],
        "additionalProperties": False,
    }

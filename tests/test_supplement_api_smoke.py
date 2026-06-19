from __future__ import annotations

import json
import os
import urllib.request

import pytest


pytestmark = [
    pytest.mark.smoke,
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_SUPPLEMENT_SMOKE", "").strip().lower() not in {"1", "true", "yes", "on"},
        reason="Set RUN_SUPPLEMENT_SMOKE=true to run against a live AI server.",
    ),
]


def _post_json(base_url: str, path: str, payload: dict) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def test_live_supplement_chat_smoke_contract():
    base_url = os.getenv("SUPPLEMENT_SMOKE_BASE_URL", "http://100.114.8.118:8000")
    questions = [
        "종합비타민, 오메가3, 실리마린을 한번에 섭취해도 괜찮아?",
        "유산균 항생제랑 같이 먹어도 돼?",
    ]

    for question in questions:
        payload = _post_json(base_url, "/api/ai/supplement-chat", {"question": question})

        assert payload["mode"] == "full"
        assert isinstance(payload["answer"], str) and payload["answer"].strip()
        assert isinstance(payload.get("sources"), list) and payload["sources"]

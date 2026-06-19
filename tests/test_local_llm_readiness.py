from __future__ import annotations

import json
import socket
import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest

from scripts import check_local_llm_readiness as readiness


def test_parse_tags_detects_gemma4_latest():
    payload = {"models": [{"name": "llama3"}, {"name": "gemma4:latest"}]}

    assert "gemma4:latest" in readiness.parse_ollama_tags(payload)


def test_model_missing_reports_failed_model_missing():
    def fake_request(method, url, *, timeout_sec, payload=None):
        return {"models": [{"name": "llama3"}]}, 12

    report = readiness.build_readiness_report(
        base_url="http://127.0.0.1:11434",
        model="gemma4:latest",
        timeout_sec=30,
        request_json=fake_request,
    )

    assert report["ollama_reachable"] is True
    assert report["model_installed"] is False
    assert report["readiness_status"] == "failed_model_missing"
    assert report["error_category"] == "model_missing"


def test_unreachable_reports_failed_ollama_unreachable():
    def fake_request(method, url, *, timeout_sec, payload=None):
        raise urllib.error.URLError("connection refused")

    report = readiness.build_readiness_report(
        base_url="http://127.0.0.1:11434",
        model="gemma4:latest",
        timeout_sec=30,
        request_json=fake_request,
    )

    assert report["ollama_reachable"] is False
    assert report["readiness_status"] == "failed_ollama_unreachable"
    assert report["error_category"] == "ollama_unreachable"


def test_probe_timeout_reports_failed_probe_timeout():
    def fake_request(method, url, *, timeout_sec, payload=None):
        if method == "GET":
            return {"models": [{"name": "gemma4:latest"}]}, 10
        raise socket.timeout("timed out")

    report = readiness.build_readiness_report(
        base_url="http://127.0.0.1:11434",
        model="gemma4:latest",
        timeout_sec=1,
        request_json=fake_request,
    )

    assert report["readiness_status"] == "failed_probe_timeout"
    assert report["error_category"] == "timeout"


def test_malformed_probe_json_reports_failed_json_parse():
    def fake_request(method, url, *, timeout_sec, payload=None):
        if method == "GET":
            return {"models": [{"name": "gemma4:latest"}]}, 10
        return {"response": "not json"}, 25

    report = readiness.build_readiness_report(
        base_url="http://127.0.0.1:11434",
        model="gemma4:latest",
        timeout_sec=30,
        request_json=fake_request,
    )

    assert report["probe_success"] is True
    assert report["json_parse_success"] is False
    assert report["readiness_status"] == "failed_json_parse"
    assert report["error_category"] == "json_parse_error"


def test_sanitize_base_url_hides_path_query_and_credentials():
    sanitized = readiness.sanitize_base_url_for_report(
        "http://user:pass@127.0.0.1:11434/private/path?token=secret"
    )

    assert sanitized == {
        "base_url": "http://127.0.0.1:11434",
        "base_url_host": "127.0.0.1",
    }
    assert "user" not in json.dumps(sanitized)
    assert "secret" not in json.dumps(sanitized)


def test_strict_mode_returns_nonzero_on_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        readiness,
        "build_readiness_report",
        lambda **_kwargs: {
            "readiness_status": "failed_model_missing",
            "error_category": "model_missing",
        },
    )
    monkeypatch.setattr(sys, "argv", ["check_local_llm_readiness.py", "--strict"])

    with pytest.raises(SystemExit) as exc:
        readiness.main()

    assert exc.value.code == 1
    assert "failed_model_missing" in capsys.readouterr().out


def test_non_strict_prints_failed_readiness_report(monkeypatch, capsys):
    monkeypatch.setattr(
        readiness,
        "build_readiness_report",
        lambda **_kwargs: {
            "readiness_status": "failed_model_missing",
            "error_category": "model_missing",
        },
    )
    monkeypatch.setattr(sys, "argv", ["check_local_llm_readiness.py"])

    readiness.main()

    assert "failed_model_missing" in capsys.readouterr().out


def test_output_file_contains_no_raw_prompt_or_raw_response(tmp_path):
    def fake_request(method, url, *, timeout_sec, payload=None):
        if method == "GET":
            return {"models": [{"name": "gemma4:latest"}]}, 10
        assert readiness.PROBE_PROMPT in payload["prompt"]
        return {"response": "{\"ok\": true, \"secret_raw_response\": \"must-not-leak\"}"}, 20

    report = readiness.build_readiness_report(
        base_url="http://user:pass@127.0.0.1:11434/raw?token=secret",
        model="gemma4:latest",
        timeout_sec=30,
        request_json=fake_request,
    )
    output = tmp_path / "readiness.json"
    readiness._write_json_if_requested(str(output), report)
    text = output.read_text(encoding="utf-8")

    assert report["readiness_status"] == "pass"
    assert readiness.PROBE_PROMPT not in text
    assert "must-not-leak" not in text
    assert "user:pass" not in text
    assert "token=secret" not in text


def test_default_model_and_base_url_env_behavior(monkeypatch):
    seen = {}

    def fake_build(**kwargs):
        seen.update(kwargs)
        return {"readiness_status": "pass", "error_category": None}

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "gemma4:latest")
    monkeypatch.setattr(readiness, "build_readiness_report", fake_build)
    monkeypatch.setattr(sys, "argv", ["check_local_llm_readiness.py", "--no-probe"])

    readiness.main()

    assert seen["base_url"] == "http://127.0.0.1:11434"
    assert seen["model"] == "gemma4:latest"
    assert seen["probe"] is False


@pytest.mark.slow
@pytest.mark.integration
def test_script_help_works():
    result = subprocess.run(
        [sys.executable, "scripts/check_local_llm_readiness.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--base-url" in result.stdout
    assert "--strict" in result.stdout

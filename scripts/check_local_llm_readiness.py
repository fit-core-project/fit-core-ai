from __future__ import annotations

import argparse
import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable


DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "gemma4:latest"
PROBE_PROMPT = "Return exactly this JSON object and no extra text: {\"ok\": true}"
DEFAULT_WINDOWS_OLLAMA_DATA_DIR = ""
DEFAULT_WINDOWS_OLLAMA_PROGRAM_DIR = ""


def sanitize_base_url_for_report(url: str) -> dict[str, str | None]:
    parsed = urllib.parse.urlparse(url.strip() or DEFAULT_BASE_URL)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "unknown"
    port = parsed.port
    netloc = f"{host}:{port}" if port is not None else host
    return {
        "base_url": f"{scheme}://{netloc}",
        "base_url_host": host,
    }


def parse_ollama_tags(payload: dict[str, Any]) -> list[str]:
    models = payload.get("models")
    if not isinstance(models, list):
        return []
    names: list[str] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("model")
        if name:
            names.append(str(name))
    return sorted(names)


def detect_windows_ollama_install(
    *,
    data_dir: str | None = None,
    program_dir: str | None = None,
) -> dict[str, Any]:
    """Detect a Windows Ollama install visible from WSL without invoking Windows processes."""
    data_value = data_dir or os.getenv("WINDOWS_OLLAMA_DATA_DIR", DEFAULT_WINDOWS_OLLAMA_DATA_DIR)
    program_value = program_dir or os.getenv("WINDOWS_OLLAMA_PROGRAM_DIR", DEFAULT_WINDOWS_OLLAMA_PROGRAM_DIR)
    data_path = Path(data_value).expanduser() if data_value else None
    program_path = Path(program_value).expanduser() if program_value else None
    data_exists = bool(data_path and data_path.exists())
    program_exists = bool(program_path and program_path.exists())
    executable_exists = bool(program_path and any(
        (program_path / name).exists() for name in ("ollama.exe", "ollama app.exe")))

    return {
        "windows_install_detected": data_exists or executable_exists,
        "windows_data_dir_detected": data_exists,
        "windows_program_dir_detected": program_exists,
        "windows_executable_detected": executable_exists,
        "windows_pid_file_detected": bool(data_path and (data_path / "ollama.pid").exists()),
        "windows_server_log_detected": bool(data_path and (data_path / "server.log").exists()),
        "windows_data_dir": str(data_path) if data_exists else None,
        "windows_program_dir": str(program_path) if program_exists else None,
    }


def classify_local_llm_error(exc: Exception) -> str:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return "timeout"
    if isinstance(exc, urllib.error.URLError):
        reason = getattr(exc, "reason", None)
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return "timeout"
        return "network_error"
    if isinstance(exc, urllib.error.HTTPError):
        return "http_error"
    if isinstance(exc, json.JSONDecodeError):
        return "json_parse_error"
    return "unknown"


def _request_json(
    method: str,
    url: str,
    *,
    timeout_sec: float,
    payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    body = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_sec) as response:
        raw = response.read()
    latency_ms = round((time.perf_counter() - started) * 1000)
    return json.loads(raw.decode("utf-8")), latency_ms


def _readiness_status(
    *,
    ollama_reachable: bool,
    model_installed: bool,
    warmup_enabled: bool,
    warmup_success: bool,
    probe_enabled: bool,
    probe_success: bool,
    json_parse_success: bool,
    error_category: str | None,
) -> str:
    if not ollama_reachable:
        return "failed_ollama_unreachable"
    if not model_installed:
        return "failed_model_missing"
    if warmup_enabled and error_category == "warmup_timeout":
        return "failed_warmup_timeout"
    if warmup_enabled and not warmup_success:
        return "failed_warmup_error"
    if not probe_enabled:
        return "pass"
    if error_category == "timeout":
        return "failed_probe_timeout"
    if not probe_success:
        return "failed_probe_error"
    if not json_parse_success:
        return "failed_json_parse"
    return "pass"


def build_readiness_report(
    *,
    base_url: str,
    model: str,
    timeout_sec: float,
    probe: bool = True,
    warmup: bool = False,
    warmup_timeout_sec: float | None = None,
    warmup_num_predict: int = 1,
    request_json: Callable[..., tuple[dict[str, Any], int]] = _request_json,
) -> dict[str, Any]:
    base = base_url.rstrip("/")
    sanitized = sanitize_base_url_for_report(base_url)
    windows_install = detect_windows_ollama_install()
    report: dict[str, Any] = {
        "ollama_reachable": False,
        "model_installed": False,
        "model": model,
        "base_url_host": sanitized["base_url_host"],
        **windows_install,
        "tags_latency_ms": None,
        "warmup_enabled": bool(warmup),
        "warmup_success": False,
        "warmup_latency_ms": None,
        "warmup_timeout_sec": warmup_timeout_sec if warmup_timeout_sec is not None else timeout_sec,
        "warmup_num_predict": warmup_num_predict,
        "probe_enabled": bool(probe),
        "probe_success": False,
        "probe_latency_ms": None,
        "json_parse_success": False,
        "readiness_status": "failed_ollama_unreachable",
        "error_category": None,
        "next_action": None,
    }

    try:
        tags_payload, tags_latency_ms = request_json(
            "GET",
            f"{base}/api/tags",
            timeout_sec=timeout_sec,
        )
        report["ollama_reachable"] = True
        report["tags_latency_ms"] = tags_latency_ms
    except Exception as exc:
        category = classify_local_llm_error(exc)
        report["error_category"] = "timeout" if category == "timeout" else "ollama_unreachable"
        report["readiness_status"] = "failed_ollama_unreachable"
        report["next_action"] = (
            "windows_ollama_detected_but_http_unreachable"
            if report["windows_install_detected"]
            else "install_or_start_ollama"
        )
        return report

    installed_models = parse_ollama_tags(tags_payload)
    report["model_installed"] = model in installed_models
    if not report["model_installed"]:
        report["error_category"] = "model_missing"
        report["readiness_status"] = "failed_model_missing"
        report["next_action"] = "pull_model_or_update_LOCAL_LLM_MODEL"
        return report

    if warmup:
        try:
            warmup_payload, warmup_latency_ms = request_json(
                "POST",
                f"{base}/api/generate",
                timeout_sec=warmup_timeout_sec if warmup_timeout_sec is not None else timeout_sec,
                payload={
                    "model": model,
                    "prompt": PROBE_PROMPT,
                    "format": "json",
                    "think": False,
                    "stream": False,
                    "options": {
                        "temperature": 0,
                        "num_predict": max(1, int(warmup_num_predict or 1)),
                        "seed": 42,
                    },
                },
            )
            report["warmup_latency_ms"] = warmup_latency_ms
            report["warmup_success"] = isinstance(warmup_payload.get("response"), str)
        except Exception as exc:
            category = classify_local_llm_error(exc)
            report["error_category"] = "warmup_timeout" if category == "timeout" else "warmup_error"
            report["readiness_status"] = _readiness_status(
                ollama_reachable=bool(report["ollama_reachable"]),
                model_installed=bool(report["model_installed"]),
                warmup_enabled=bool(report["warmup_enabled"]),
                warmup_success=bool(report["warmup_success"]),
                probe_enabled=bool(report["probe_enabled"]),
                probe_success=bool(report["probe_success"]),
                json_parse_success=bool(report["json_parse_success"]),
                error_category=report["error_category"],
            )
            report["next_action"] = "restart_or_prewarm_ollama_then_retry"
            return report

    if not probe:
        report["probe_success"] = True
        report["json_parse_success"] = True
        report["readiness_status"] = "pass"
        report["next_action"] = "run_probe_or_routine_smoke"
        return report

    try:
        probe_payload, probe_latency_ms = request_json(
            "POST",
            f"{base}/api/generate",
            timeout_sec=timeout_sec,
            payload={
                "model": model,
                "prompt": PROBE_PROMPT,
                "format": "json",
                "think": False,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 32,
                    "seed": 42,
                },
            },
        )
        report["probe_latency_ms"] = probe_latency_ms
        response_text = probe_payload.get("response")
        report["probe_success"] = isinstance(response_text, str)
        if report["probe_success"]:
            json.loads(response_text)
            report["json_parse_success"] = True
    except json.JSONDecodeError:
        report["error_category"] = "json_parse_error"
    except Exception as exc:
        category = classify_local_llm_error(exc)
        report["error_category"] = category if category != "unknown" else "probe_error"

    report["readiness_status"] = _readiness_status(
        ollama_reachable=bool(report["ollama_reachable"]),
        model_installed=bool(report["model_installed"]),
        warmup_enabled=bool(report["warmup_enabled"]),
        warmup_success=bool(report["warmup_success"]),
        probe_enabled=bool(report["probe_enabled"]),
        probe_success=bool(report["probe_success"]),
        json_parse_success=bool(report["json_parse_success"]),
        error_category=report["error_category"],
    )
    if report["readiness_status"] == "pass":
        report["error_category"] = None
        report["next_action"] = "ready_for_routine_quality_eval"
    elif report["error_category"] is None:
        report["error_category"] = "probe_error"
    if report["readiness_status"] != "pass" and report["next_action"] is None:
        report["next_action"] = "inspect_ollama_model_json_probe"
    return report


def _write_json_if_requested(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check local Ollama/Gemma4 readiness.")
    parser.add_argument("--base-url", default=os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.getenv("LOCAL_LLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--timeout-sec", type=float, default=30)
    parser.add_argument("--probe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--warmup-timeout-sec", type=float)
    parser.add_argument("--warmup-num-predict", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    report = build_readiness_report(
        base_url=args.base_url,
        model=args.model,
        timeout_sec=args.timeout_sec,
        probe=args.probe,
        warmup=args.warmup,
        warmup_timeout_sec=args.warmup_timeout_sec,
        warmup_num_predict=args.warmup_num_predict,
    )
    _write_json_if_requested(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.strict and report["readiness_status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

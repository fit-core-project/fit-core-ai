from fastapi.testclient import TestClient

import main
from engines.supplement.supplement_engine import SupplementRAGEngine


def test_quicklog_endpoint_returns_200_on_empty_text():
    client = TestClient(main.app)

    response = client.post("/api/ai/parse-log", json={"text": ""})

    assert response.status_code == 200
    payload = response.json()
    assert "diet_logs" in payload
    assert "workout_logs" in payload
    assert "overall_summary" in payload


def test_health_exposes_sanitized_provider_state(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("ALLOW_LOCAL_LLM_IN_PRODUCTION", "true")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "gemma4:latest")
    client = TestClient(main.app)

    response = client.get("/api/ai/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["appEnv"] == "production"
    assert payload["llmProviderConfigured"] == "local"
    assert payload["llmProviderEffective"] == "local"
    assert payload["allowLocalLlmInProduction"] is True
    assert payload["localLlmModel"] == "gemma4:latest"
    assert "GOOGLE_API_KEY" not in payload


def test_supplement_endpoint_returns_degraded_200(monkeypatch):
    monkeypatch.setattr(main, "supplement_rag", SupplementRAGEngine.degraded("test"))
    client = TestClient(main.app)

    response = client.post("/api/ai/supplement-chat", json={"question": "마그네슘 먹어도 돼?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert payload["sources"] == []
    assert payload["mode"] == "degraded"


def test_stt_unavailable_returns_200_without_crash(monkeypatch):
    monkeypatch.setattr(main.importlib.util, "find_spec", lambda name: None if name == "faster_whisper" else object())
    client = TestClient(main.app)

    with open(__file__, "rb") as handle:
        response = client.post(
            "/api/ai/stt",
            files={"audio_file": ("demo.webm", handle, "audio/webm")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == ""
    assert payload["status"] == "unavailable"


def test_stt_full_path_returns_transcript_when_dependencies_available(monkeypatch):
    class FakeSegment:
        text = "테스트 음성 기록"

    class FakeWhisperModel:
        def transcribe(self, path, beam_size, language):
            return [FakeSegment()], object()

    monkeypatch.setattr(main.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(main.shutil, "which", lambda name: "ffmpeg")
    monkeypatch.setattr(main, "whisper_model", FakeWhisperModel())
    client = TestClient(main.app)

    with open(__file__, "rb") as handle:
        response = client.post(
            "/api/ai/stt",
            files={"audio_file": ("demo.webm", handle, "audio/webm")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "테스트 음성 기록"
    assert payload["status"] == "success"

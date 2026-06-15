from __future__ import annotations

import sys
import types

import pytest

from engines import llm_router


class _DummyOllama:
    def __init__(self, *, model, base_url, temperature, format, **options):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.format = format
        self.options = options

    def invoke(self, input, config=None, **kwargs):
        return input

    def __or__(self, other):
        return other


class _DummyGemini:
    def __init__(self, *, model, temperature):
        self.model = model
        self.temperature = temperature


@pytest.fixture
def dummy_provider_modules(monkeypatch):
    ollama = types.ModuleType("langchain_ollama")
    ollama.ChatOllama = _DummyOllama
    gemini = types.ModuleType("langchain_google_genai")
    gemini.ChatGoogleGenerativeAI = _DummyGemini
    monkeypatch.setitem(sys.modules, "langchain_ollama", ollama)
    monkeypatch.setitem(sys.modules, "langchain_google_genai", gemini)


def _clear_provider_env(monkeypatch):
    for key in (
        "APP_ENV",
        "LLM_PROVIDER",
        "ALLOW_LOCAL_LLM_IN_PRODUCTION",
        "LOCAL_LLM_MODEL",
        "LOCAL_LLM_NUM_PREDICT",
        "LOCAL_LLM_NUM_CTX",
        "OLLAMA_BASE_URL",
        "GOOGLE_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def test_non_production_local_provider_is_allowed(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("LLM_PROVIDER", "local")

    config = llm_router.resolve_llm_provider()

    assert config.requested_provider == "local"
    assert config.effective_provider == "local"
    assert config.blocked_reason is None


def test_production_local_without_allow_flag_is_blocked_to_gemini(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("LLM_PROVIDER", "local")

    config = llm_router.resolve_llm_provider()

    assert config.requested_provider == "local"
    assert config.effective_provider == "gemini"
    assert config.blocked_reason == "production_local_not_allowed"


@pytest.mark.parametrize("value", ["true", "1", "yes", "on", " TRUE "])
def test_production_local_allow_flag_true_variants_enable_local(monkeypatch, value):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("ALLOW_LOCAL_LLM_IN_PRODUCTION", value)

    config = llm_router.resolve_llm_provider()

    assert config.effective_provider == "local"
    assert config.local_allowed_in_production is True
    assert config.blocked_reason is None


@pytest.mark.parametrize("value", ["", "false", "0", "no", "off", "maybe"])
def test_production_local_allow_flag_false_or_invalid_values_block(monkeypatch, value):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("ALLOW_LOCAL_LLM_IN_PRODUCTION", value)

    config = llm_router.resolve_llm_provider()

    assert config.effective_provider == "gemini"
    assert config.local_allowed_in_production is False
    assert config.blocked_reason == "production_local_not_allowed"


def test_default_provider_uses_local_ollama(monkeypatch):
    _clear_provider_env(monkeypatch)

    config = llm_router.resolve_llm_provider()

    assert config.requested_provider == "local"
    assert config.effective_provider == "local"
    assert config.model_name == "gemma4:latest"


def test_blocked_production_log_has_no_raw_env_or_api_key(monkeypatch, capsys, dummy_provider_modules):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("GOOGLE_API_KEY", "secret-google-key")

    llm = llm_router.get_llm("routine")
    captured = capsys.readouterr().out

    assert isinstance(llm, _DummyGemini)
    assert "local provider blocked in production" in captured
    assert "secret-google-key" not in captured
    assert "GOOGLE_API_KEY" not in captured


def test_allowed_local_log_uses_sanitized_model_name(monkeypatch, capsys, dummy_provider_modules):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("ALLOW_LOCAL_LLM_IN_PRODUCTION", "true")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "gemma4:latest ; raw-token")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://user:pass@127.0.0.1:11434")

    llm = llm_router.get_llm("routine", temperature=0.1)
    captured = capsys.readouterr().out

    assert llm.model == "gemma4:latestraw-token"
    assert "production local provider enabled via explicit opt-in" in captured
    assert "model=gemma4:latestraw-token" in captured
    assert "gemma4:latest ; raw-token" not in captured
    assert "user:pass" not in captured


def test_local_num_predict_unset_preserves_existing_ollama_options(monkeypatch, dummy_provider_modules):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "local")

    llm = llm_router.get_llm("routine")

    assert llm.options == {}


def test_local_num_predict_and_num_ctx_valid_env_are_passed(monkeypatch, dummy_provider_modules):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_LLM_NUM_PREDICT", "4096")
    monkeypatch.setenv("LOCAL_LLM_NUM_CTX", "8192")

    llm = llm_router.get_llm("routine")

    assert llm.options["num_predict"] == 4096
    assert llm.options["num_ctx"] == 8192


def test_invalid_local_generation_budget_env_is_ignored(monkeypatch, dummy_provider_modules):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_LLM_NUM_PREDICT", "invalid")
    monkeypatch.setenv("LOCAL_LLM_NUM_CTX", "0")

    llm = llm_router.get_llm("routine")

    assert llm.options == {}


def test_gemini_path_ignores_local_generation_budget(monkeypatch, dummy_provider_modules):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LOCAL_LLM_NUM_PREDICT", "4096")
    monkeypatch.setenv("LOCAL_LLM_NUM_CTX", "8192")

    llm = llm_router.get_llm("routine")

    assert isinstance(llm, _DummyGemini)
    assert not hasattr(llm, "options")

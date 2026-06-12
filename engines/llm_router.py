"""LLM provider factory and provider-neutral error mapping."""

import asyncio
import os
import json
import re
from dataclasses import dataclass
from typing import Literal

import httpx
from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import Runnable
from pydantic import ValidationError

load_dotenv()

StatusReasonCode = Literal["none", "llmTimeout", "schemaError", "networkError", "emptyCandidate"]


_MARKDOWN_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```")
_TRUE_VALUES = {"true", "1", "yes", "on"}


@dataclass(frozen=True)
class LlmProviderConfig:
    requested_provider: str
    effective_provider: str
    local_allowed_in_production: bool
    model_name: str
    is_production: bool
    blocked_reason: str | None = None


def _parse_json_content(content: str) -> dict:
    cleaned = _MARKDOWN_FENCE_RE.sub("", content).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        first = cleaned.find("{")
        last = cleaned.rfind("}")
        if first != -1 and last != -1 and first < last:
            return json.loads(cleaned[first : last + 1])
        raise


def _env_flag_enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in _TRUE_VALUES


def _sanitize_model_name(value: str | None, default: str = "gemma4") -> str:
    raw = str(value or "").strip() or default
    sanitized = re.sub(r"[^A-Za-z0-9._:-]", "", raw)
    return sanitized or default


def _parse_positive_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw.strip())
    except ValueError:
        return None
    return value if value > 0 else None


def _local_ollama_options_from_env() -> dict[str, int]:
    options: dict[str, int] = {}
    num_predict = _parse_positive_int_env("LOCAL_LLM_NUM_PREDICT")
    num_ctx = _parse_positive_int_env("LOCAL_LLM_NUM_CTX")
    if num_predict is not None:
        options["num_predict"] = num_predict
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    return options


def resolve_llm_provider() -> LlmProviderConfig:
    requested_provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower() or "gemini"
    app_env = os.getenv("APP_ENV", "").strip().lower()
    is_production = app_env == "production"
    local_allowed = _env_flag_enabled(os.getenv("ALLOW_LOCAL_LLM_IN_PRODUCTION"))
    model_name = _sanitize_model_name(os.getenv("LOCAL_LLM_MODEL"), "gemma4")

    if is_production and requested_provider == "local" and not local_allowed:
        return LlmProviderConfig(
            requested_provider=requested_provider,
            effective_provider="gemini",
            local_allowed_in_production=False,
            model_name=model_name,
            is_production=True,
            blocked_reason="production_local_not_allowed",
        )

    return LlmProviderConfig(
        requested_provider=requested_provider,
        effective_provider=requested_provider,
        local_allowed_in_production=local_allowed,
        model_name=model_name,
        is_production=is_production,
    )


class LocalJsonChatModel(Runnable):
    """Runnable wrapper adding Pydantic structured output to ChatOllama."""

    def __init__(self, llm):
        self._llm = llm
        self.model = llm.model
        self.format = llm.format
        self.temperature = llm.temperature
        self.base_url = llm.base_url

    def invoke(self, input, config=None, **kwargs):
        return self._llm.invoke(input, config=config, **kwargs)

    def with_structured_output(self, schema, **_kwargs):
        from langchain_core.runnables import RunnableLambda

        def parse_message(message):
            content = message.content if hasattr(message, "content") else str(message)
            try:
                data = _parse_json_content(content)
                if hasattr(schema, "model_validate"):
                    return schema.model_validate(data)
                return data
            except Exception as exc:
                from langchain_core.exceptions import OutputParserException

                raise OutputParserException(str(exc), llm_output=content) from exc

        return self | RunnableLambda(parse_message)

    def __or__(self, other):
        return self._llm | other

    def __ror__(self, other):
        return other | self._llm

    def __getattr__(self, name):
        return getattr(self._llm, name)


def get_llm(engine_type: str, temperature: float = 0):
    """Return the configured LangChain chat model.

    LLM_PROVIDER=local switches generation to Ollama. Local mode always uses
    Ollama JSON mode because routine generation depends on strict JSON output.
    In production, local is overridden to gemini unless
    ALLOW_LOCAL_LLM_IN_PRODUCTION is explicitly enabled.
    """
    config = resolve_llm_provider()
    provider = config.effective_provider

    app_env = os.getenv("APP_ENV", "").strip().lower()
    if config.blocked_reason == "production_local_not_allowed":
        print(
            "[LLM Router] WARNING: local provider blocked in production; "
            "set ALLOW_LOCAL_LLM_IN_PRODUCTION=true to enable"
        )
    elif app_env == "production" and config.requested_provider == "local" and config.local_allowed_in_production:
        print("[LLM Router] production local provider enabled via explicit opt-in")

    if provider == "local":
        from langchain_ollama import ChatOllama

        model_name = config.model_name
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip() or "http://localhost:11434"
        print(f"[LLM Router] local -> ChatOllama(model={model_name}, format=json, engine={engine_type})")
        llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            format="json",
            **_local_ollama_options_from_env(),
        )
        return LocalJsonChatModel(llm)

    from langchain_google_genai import ChatGoogleGenerativeAI

    print("[LLM Router] gemini -> ChatGoogleGenerativeAI(gemini-2.5-flash)")
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temperature)


def map_llm_error(exc: Exception) -> StatusReasonCode:
    """Map provider-specific exceptions to public routine status codes."""
    import asyncio

    import httpx
    from langchain_core.exceptions import OutputParserException
    from pydantic import ValidationError

    exc_type_name = type(exc).__name__.lower()
    exc_module = (type(exc).__module__ or "").lower()
    exc_msg = str(exc).lower()

    if isinstance(exc, (asyncio.TimeoutError, httpx.TimeoutException, httpx.ReadTimeout)):
        return "llmTimeout"
    if "timeout" in exc_type_name or "deadline" in exc_type_name:
        return "llmTimeout"
    if "timeout" in exc_msg or "deadline exceeded" in exc_msg:
        return "llmTimeout"

    if "google" in exc_module:
        if "deadline" in exc_type_name or "deadline" in exc_msg:
            return "llmTimeout"
        return "networkError"

    if isinstance(exc, (httpx.ConnectError, httpx.NetworkError, ConnectionError, ConnectionRefusedError)):
        return "networkError"
    if "ollama" in exc_module or "connect" in exc_type_name:
        return "networkError"
    if any(k in exc_msg for k in ("connection refused", "connect error", "failed to connect")):
        return "networkError"

    if isinstance(exc, (OutputParserException, ValidationError)):
        return "schemaError"
    if "parse" in exc_type_name or "validation" in exc_type_name:
        return "schemaError"

    return "networkError"


def classify_llm_error(exc: Exception) -> str:
    """Return a structured, non-sensitive LLM error type for telemetry."""
    try:
        if isinstance(exc, (asyncio.TimeoutError, httpx.TimeoutException)):
            return "timeout"

        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
            return "quota_exhausted"

        exc_msg = str(exc).lower()
        compact_msg = exc_msg.replace("_", "").replace(" ", "")
        if "resourceexhausted" in compact_msg or "quota" in exc_msg or "429" in exc_msg:
            return "quota_exhausted"

        if isinstance(exc, OutputParserException):
            return "schema_parse_error"
        if isinstance(exc, ValidationError):
            return "validation_error"

        network_types = (
            httpx.NetworkError,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
        )
        if isinstance(exc, network_types):
            return "network_error"

        return "unknown"
    except Exception:
        return "unknown"

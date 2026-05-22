"""LLM provider factory and provider-neutral error mapping."""

import os
import json
import re
from typing import Literal

from dotenv import load_dotenv
from langchain_core.runnables import Runnable

load_dotenv()

StatusReasonCode = Literal["none", "llmTimeout", "schemaError", "networkError", "emptyCandidate"]


_MARKDOWN_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```")


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
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

    if provider == "local":
        from langchain_ollama import ChatOllama

        model_name = os.getenv("LOCAL_LLM_MODEL", "gemma4").strip() or "gemma4"
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip() or "http://localhost:11434"
        print(f"[LLM Router] local -> ChatOllama(model={model_name}, format=json, engine={engine_type})")
        llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            format="json",
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

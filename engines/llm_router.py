"""
llm_router.py — Provider Router + Status Reason Mapper

.env의 LLM_PROVIDER 값("gemini" | "local")에 따라 LLM 인스턴스를 반환하고,
provider별 예외를 공통 StatusReasonCode로 정규화한다.
"""
import os
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

# routine_engine.py가 여기서 import하므로 단일 정의 지점
StatusReasonCode = Literal["none", "llmTimeout", "schemaError", "networkError"]

_LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").lower()

# local 모드 engine → Ollama 모델 매핑
_LOCAL_MODEL_MAP: dict[str, str] = {
    "nlp":        "gemma4-e4b",
    "supplement": "gemma4-31b",
    "routine":    "gemma4-26b-moe",
}


def get_llm(engine_type: str, temperature: float = 0):
    """
    LLM_PROVIDER 환경 변수에 따라 적합한 LLM 인스턴스를 반환한다.

    Args:
        engine_type: "routine" | "nlp" | "supplement"
        temperature: 생성 다양성 (기본 0 — 결정론적)

    Returns:
        LangChain BaseChatModel (with_structured_output 지원)
    """
    if _LLM_PROVIDER == "local":
        from langchain_ollama import ChatOllama

        model_name = _LOCAL_MODEL_MAP.get(engine_type, "gemma4-26b-moe")
        print(f"[LLM Router] local → ChatOllama(model={model_name})")
        return ChatOllama(model=model_name, temperature=temperature)

    # 기본값: gemini
    from langchain_google_genai import ChatGoogleGenerativeAI

    print(f"[LLM Router] gemini → ChatGoogleGenerativeAI(gemini-2.5-flash)")
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temperature)


# ─────────────────────────────────────────────────────────────────────────────
# Status Reason Mapper
# ─────────────────────────────────────────────────────────────────────────────

def map_llm_error(exc: Exception) -> StatusReasonCode:
    """
    Provider별 예외를 공통 StatusReasonCode로 매핑한다.

    분류 우선순위:
    1. Timeout 계열          → "llmTimeout"
    2. 연결/네트워크 계열     → "networkError"
    3. 파싱/스키마 계열       → "schemaError"
    4. 나머지 알 수 없는 에러 → "networkError"
    """
    import asyncio

    import httpx
    from langchain_core.exceptions import OutputParserException
    from pydantic import ValidationError

    exc_type_name: str = type(exc).__name__.lower()
    exc_module: str = (type(exc).__module__ or "").lower()
    exc_msg: str = str(exc).lower()

    # ── 1. Timeout ────────────────────────────────────────────────────
    if isinstance(exc, (asyncio.TimeoutError, httpx.TimeoutException, httpx.ReadTimeout)):
        return "llmTimeout"
    if "timeout" in exc_type_name or "deadline" in exc_type_name:
        return "llmTimeout"
    if "timeout" in exc_msg or "deadline exceeded" in exc_msg:
        return "llmTimeout"

    # ── 2. Google API 에러 ────────────────────────────────────────────
    if "google" in exc_module:
        # google.api_core.exceptions.DeadlineExceeded
        if "deadline" in exc_type_name or "deadline" in exc_msg:
            return "llmTimeout"
        # ResourceExhausted (429), ServiceUnavailable (503) 등
        return "networkError"

    # ── 3. Ollama / httpx 연결 에러 ───────────────────────────────────
    if isinstance(exc, (httpx.ConnectError, httpx.NetworkError, ConnectionError, ConnectionRefusedError)):
        return "networkError"
    if "ollama" in exc_module or "connect" in exc_type_name:
        return "networkError"
    if any(k in exc_msg for k in ("connection refused", "connect error", "failed to connect")):
        return "networkError"

    # ── 4. LangChain 파싱 / Pydantic 스키마 에러 ─────────────────────
    if isinstance(exc, (OutputParserException, ValidationError)):
        return "schemaError"
    if "parse" in exc_type_name or "validation" in exc_type_name:
        return "schemaError"

    # ── 5. 그 외 ─────────────────────────────────────────────────────
    return "networkError"

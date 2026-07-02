import io
import importlib.util
import json
import logging
import os
import shutil
import tempfile
import urllib.request
from contextlib import asynccontextmanager, redirect_stderr

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette.responses import JSONResponse

from database import get_db
from dev_logs import dev_log_buffer, install_stdout_capture
from engines.db_queries import (
    get_exercise_details_from_sqlite,
    get_exercise_substitution_map_from_sqlite,
    get_recent_sets,
    get_user_profile_context,
)
from engines.log_redaction import sanitize_exception_for_log, summarize_text_for_log
from engines.llm_router import resolve_llm_provider
from engines.quicklog.nlp_engine import parse_natural_language_log
from engines.quicklog.diet_parser import parse_diet_log
from engines.routine_pipeline import generate_smart_routine
from engines.schemas import RoutineDraftResponse, RoutineFeedbackRequest, RoutineFeedbackResponse, RoutineRequest
from engines.supplement.supplement_engine import SupplementRAGEngine
from models.routine_feedback import RoutineFeedback
from scripts.check_local_llm_readiness import build_readiness_report

load_dotenv()
install_stdout_capture()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

whisper_model = None
supplement_rag = None


def get_supplement_engine() -> SupplementRAGEngine:
    global supplement_rag
    if supplement_rag is None:
        supplement_rag = SupplementRAGEngine(db_path=os.environ.get("SUPPLEMENT_RAG_DB_PATH", "./data/chroma_db"))
    return supplement_rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_model, supplement_rag

    logger.info("\n" + "=" * 40)
    logger.info("[server startup] loading optional AI engines")

    if whisper_model is None and os.environ.get("WHISPER_PRELOAD", "false").strip().lower() == "true":
        logger.info("[STT] loading Whisper model")
        from faster_whisper import WhisperModel

        whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
        logger.info("[STT] Whisper model loaded")
    elif os.environ.get("WHISPER_PRELOAD", "false").strip().lower() != "true":
        logger.info("[STT] startup preload skipped")

    if supplement_rag is None:
        logger.info("[Supplement RAG] startup load")
        try:
            with redirect_stderr(io.StringIO()):
                supplement_rag = SupplementRAGEngine(db_path=os.environ.get("SUPPLEMENT_RAG_DB_PATH", "./data/chroma_db"))
            if getattr(supplement_rag, "ready", False):
                logger.info("[Supplement RAG] full mode ready")
            else:
                logger.warning(f"[Supplement RAG] degraded mode {getattr(supplement_rag, 'degraded_reason', 'unavailable')}")
        except Exception as exc:
            logger.warning(f"[Supplement RAG] startup fallback {sanitize_exception_for_log(exc)}")
            supplement_rag = SupplementRAGEngine(db_path=os.environ.get("SUPPLEMENT_RAG_DB_PATH", "./data/chroma_db"))

    logger.info("[server startup] ready")
    logger.info("=" * 40 + "\n")

    yield

    whisper_model = None
    supplement_rag = None
    logger.info("\n[server shutdown] optional AI engines cleared")


app = FastAPI(title="Fit-Core AI Server", lifespan=lifespan)

_default_origins = "http://localhost:3000,http://localhost:3001"
_cors_origins = [origin.strip() for origin in os.environ.get("AI_CORS_ALLOWED_ORIGINS", _default_origins).split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/dev/logs")
def api_dev_logs(limit: int = 120):
    if os.getenv("APP_ENV", "").strip().lower() == "production":
        raise HTTPException(status_code=404, detail="Not found")
    return dev_log_buffer.tail(limit)


@app.get("/api/health")
@app.get("/api/ai/health")
@app.get("/health")
def api_health():
    rag_ready = bool(getattr(supplement_rag, "ready", False)) if supplement_rag is not None else False
    provider_config = resolve_llm_provider()

    llm_status = "down"
    try:
        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        req = urllib.request.Request(f"{ollama_url}/api/version")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            if resp.status == 200:
                llm_status = "up"
    except Exception:
        llm_status = "down"

    return {
        "ai": "up",
        "llm": llm_status,
        "appEnv": os.environ.get("APP_ENV", "local"),
        "llmProvider": provider_config.requested_provider,
        "llmProviderConfigured": provider_config.requested_provider,
        "llmProviderEffective": provider_config.effective_provider,
        "allowLocalLlmInProduction": provider_config.local_allowed_in_production,
        "localLlmModel": provider_config.model_name,
        "ollamaBaseUrlConfigured": bool(os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")),
        "supplementRag": "ready" if rag_ready else "degraded_or_not_loaded",
        "whisperPreload": os.environ.get("WHISPER_PRELOAD", "false").strip().lower() == "true",
    }


@app.get("/api/ai/local-llm/readiness")
def api_local_llm_readiness(probe: bool = True, timeout_sec: float = 5.0):
    """Return a sanitized Ollama/local Gemma readiness report for demo and ops screens."""
    safe_timeout = max(1.0, min(float(timeout_sec or 5.0), 30.0))
    return build_readiness_report(
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        model=os.environ.get("LOCAL_LLM_MODEL", "gemma4:latest"),
        timeout_sec=safe_timeout,
        probe=probe,
    )


@app.get("/api/ai/exercises/details")
def api_get_exercise_details(ids: str = ""):
    """로컬 운동 백과 상세 메타데이터 조회.

    루틴 생성용 운영 DB가 아니라, 운동 DB 고도화 산출물인 `fit_core.sqlite`를
    읽어 프론트 운동 백과 화면에 가동성/부하/효과/난이도/체형 민감도 필드를 제공한다.
    """
    requested_ids = [item.strip() for item in ids.split(",") if item.strip()]
    try:
        return get_exercise_details_from_sqlite(requested_ids)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        print("[Exercise Details Error]", sanitize_exception_for_log(exc))
        raise HTTPException(status_code=500, detail="Failed to load exercise details") from exc


@app.get("/api/ai/exercises/substitution-map")
def api_get_substitution_map(
    source_exercise_id: str | None = None,
    relation_type: str | None = None,
    constraint_code: str | None = None,
    limit: int = 5000,
):
    """운동 대체/회귀/변형 관계 맵 조회."""
    try:
        return get_exercise_substitution_map_from_sqlite(
            source_exercise_id=source_exercise_id,
            relation_type=relation_type,
            constraint_code=constraint_code,
            limit=limit,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        print("[Substitution Map Error]", sanitize_exception_for_log(exc))
        raise HTTPException(status_code=500, detail="Failed to load substitution map") from exc


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    def sanitize_error(error: dict) -> dict:
        sanitized = dict(error)
        loc = sanitized.get("loc") or []
        sensitive_fields = {
            "body",
            "request_body",
            "text",
            "question",
            "transcript",
            "user_note",
            "userNote",
        }
        if any(part in sensitive_fields for part in loc):
            sanitized["input"] = "[REDACTED]"
        if "ctx" in sanitized:
            sanitized["ctx"] = {key: str(value) for key, value in sanitized["ctx"].items()}
        return sanitized

    sanitized_errors = [sanitize_error(error) for error in exc.errors()]
    logger.warning(f"[validation error] {sanitized_errors}")
    return JSONResponse(status_code=422, content={"detail": sanitized_errors})


@app.post("/api/ai/generate-routine", response_model=RoutineDraftResponse, response_model_by_alias=True)
def api_generate_routine(req: RoutineRequest, db: Session = Depends(get_db)):
    try:
        logger.info("[routine generation request]")
        profile = get_user_profile_context(db, req.user_id)
        recent_sets = get_recent_sets(db, req.user_id)
        return generate_smart_routine(req, db, profile=profile, recent_sets=recent_sets)
    except Exception as exc:
        logger.error(f"[Routine Error] {sanitize_exception_for_log(exc)}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    "/api/ai/routine-feedback",
    response_model=RoutineFeedbackResponse,
    status_code=201,
)
def api_routine_feedback(req: RoutineFeedbackRequest, db: Session = Depends(get_db)):
    user_note = req.user_note.strip() if req.user_note else None
    if user_note == "":
        user_note = None
    feedback = RoutineFeedback(
        user_id=req.user_id,
        routine_draft_id=req.routine_draft_id,
        rating=req.rating,
        completed=req.completed,
        accepted_without_edits=req.accepted_without_edits,
        skipped_exercise_ids=req.skipped_exercises,
        edited_exercises=[item.model_dump() for item in req.edited_exercises],
        user_note=user_note,
        source="api",
    )
    try:
        db.add(feedback)
        db.commit()
        db.refresh(feedback)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to store routine feedback") from exc

    return RoutineFeedbackResponse(
        feedback_id=feedback.id,
        stored_at=feedback.created_at.isoformat(),
    )


class LogRequest(BaseModel):
    text: str


@app.post("/api/ai/parse-log")
def api_parse_log(req: LogRequest):
    try:
        logger.info(f"[quicklog parse request] {summarize_text_for_log(req.text)}")
        result_json_str = parse_natural_language_log(req.text)
        return json.loads(result_json_str)
    except Exception as exc:
        logger.error(f"[NLP Error] {sanitize_exception_for_log(exc)}")
        fallback = parse_natural_language_log("")
        return json.loads(fallback)


@app.post("/api/ai/parse-diet")
def api_parse_diet(req: LogRequest):
    try:
        logger.info(f"[diet parse request] {summarize_text_for_log(req.text)}")
        result_json_str = parse_diet_log(req.text)
        return json.loads(result_json_str)
    except Exception as exc:
        logger.error(f"[Diet NLP Error] {sanitize_exception_for_log(exc)}")
        fallback = parse_diet_log("")
        return json.loads(fallback)


@app.post("/api/ai/stt")
async def api_speech_to_text(audio_file: UploadFile = File(...)):
    temp_file_path = ""
    try:
        logger.info(f"[STT request] {summarize_text_for_log(audio_file.filename or '')}")
        if importlib.util.find_spec("faster_whisper") is None or shutil.which("ffmpeg") is None:
            return {
                "text": "",
                "status": "unavailable",
                "message": "음성 인식 기능은 현재 데모 환경에서 비활성화되어 있습니다. 텍스트 입력을 사용해 주세요.",
            }

        audio_bytes = await audio_file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name

        global whisper_model
        if whisper_model is None:
            logger.info("[STT] lazy loading Whisper model")
            from faster_whisper import WhisperModel

            whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
            logger.info("[STT] Whisper model loaded")
        segments, info = whisper_model.transcribe(temp_file_path, beam_size=5, language="ko")
        transcript = " ".join([segment.text for segment in segments])
        logger.info(f"[STT result] {summarize_text_for_log(transcript)}")
        return {"text": transcript, "status": "success"}
    except Exception as exc:
        logger.warning(f"[STT Error] {sanitize_exception_for_log(exc)}")
        return {
            "text": "",
            "status": "unavailable",
            "message": "음성 인식 기능은 현재 데모 환경에서 사용할 수 없습니다. 텍스트 입력을 사용해 주세요.",
        }
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


class SupplementChatRequest(BaseModel):
    question: str


@app.post("/api/ai/supplement-chat")
def api_supplement_chat(req: SupplementChatRequest):
    try:
        logger.info(f"[supplement question] {summarize_text_for_log(req.question)}")
        return get_supplement_engine().answer_question(req.question)
    except Exception as exc:
        logger.error(f"[Supplement Error] {sanitize_exception_for_log(exc)}")
        return SupplementRAGEngine.degraded("endpoint_runtime_error").answer_question(req.question)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

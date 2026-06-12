import io
import importlib.util
import json
import os
import shutil
import tempfile
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
from engines.db_queries import get_recent_sets, get_user_profile_context
from engines.log_redaction import sanitize_exception_for_log, summarize_text_for_log
from engines.llm_router import resolve_llm_provider
from engines.quicklog.nlp_engine import parse_natural_language_log
from engines.routine_pipeline import generate_smart_routine
from engines.schemas import RoutineDraftResponse, RoutineFeedbackRequest, RoutineFeedbackResponse, RoutineRequest
from engines.supplement.supplement_engine import SupplementRAGEngine
from models.routine_feedback import RoutineFeedback

load_dotenv()
install_stdout_capture()

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

    print("\n" + "=" * 40)
    print("[server startup] loading optional AI engines")

    if whisper_model is None and os.environ.get("WHISPER_PRELOAD", "false").strip().lower() == "true":
        print("[STT] loading Whisper model")
        from faster_whisper import WhisperModel

        whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
        print("[STT] Whisper model loaded")
    elif os.environ.get("WHISPER_PRELOAD", "false").strip().lower() != "true":
        print("[STT] startup preload skipped")

    if supplement_rag is None:
        print("[Supplement RAG] startup load")
        try:
            with redirect_stderr(io.StringIO()):
                supplement_rag = SupplementRAGEngine(db_path=os.environ.get("SUPPLEMENT_RAG_DB_PATH", "./data/chroma_db"))
            if getattr(supplement_rag, "ready", False):
                print("[Supplement RAG] full mode ready")
            else:
                print("[Supplement RAG] degraded mode", getattr(supplement_rag, "degraded_reason", "unavailable"))
        except Exception as exc:
            print("[Supplement RAG] startup fallback", sanitize_exception_for_log(exc))
            supplement_rag = SupplementRAGEngine(db_path=os.environ.get("SUPPLEMENT_RAG_DB_PATH", "./data/chroma_db"))

    print("[server startup] ready")
    print("=" * 40 + "\n")

    yield

    whisper_model = None
    supplement_rag = None
    print("\n[server shutdown] optional AI engines cleared")


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
    return {
        "ai": "up",
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
    print("[validation error]", sanitized_errors)
    return JSONResponse(status_code=422, content={"detail": sanitized_errors})


@app.post("/api/ai/generate-routine", response_model=RoutineDraftResponse, response_model_by_alias=True)
def api_generate_routine(req: RoutineRequest, db: Session = Depends(get_db)):
    try:
        print("[routine generation request]")
        profile = get_user_profile_context(db, req.user_id)
        recent_sets = get_recent_sets(db, req.user_id)
        return generate_smart_routine(req, db, profile=profile, recent_sets=recent_sets)
    except Exception as exc:
        print("[Routine Error]", sanitize_exception_for_log(exc))
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
        print("[quicklog parse request]", summarize_text_for_log(req.text))
        result_json_str = parse_natural_language_log(req.text)
        return json.loads(result_json_str)
    except Exception as exc:
        print("[NLP Error]", sanitize_exception_for_log(exc))
        fallback = parse_natural_language_log("")
        return json.loads(fallback)


@app.post("/api/ai/stt")
async def api_speech_to_text(audio_file: UploadFile = File(...)):
    temp_file_path = ""
    try:
        print("[STT request]", summarize_text_for_log(audio_file.filename or ""))
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
            print("[STT] lazy loading Whisper model")
            from faster_whisper import WhisperModel

            whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
            print("[STT] Whisper model loaded")
        segments, info = whisper_model.transcribe(temp_file_path, beam_size=5, language="ko")
        transcript = " ".join([segment.text for segment in segments])
        print("[STT result]", summarize_text_for_log(transcript))
        return {"text": transcript, "status": "success"}
    except Exception as exc:
        print("[STT Error]", sanitize_exception_for_log(exc))
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
        print("[supplement question]", summarize_text_for_log(req.question))
        return get_supplement_engine().answer_question(req.question)
    except Exception as exc:
        print("[Supplement Error]", sanitize_exception_for_log(exc))
        return SupplementRAGEngine.degraded("endpoint_runtime_error").answer_question(req.question)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

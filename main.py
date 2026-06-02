import uvicorn
import json
import io
import tempfile
import os
from contextlib import asynccontextmanager, redirect_stderr

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Depends
from sqlalchemy.orm import Session
from database import get_db
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import JSONResponse
from dev_logs import dev_log_buffer, install_stdout_capture

from models.routine_feedback import RoutineFeedback
from engines.db_queries import get_recent_sets, get_user_profile_context
from engines.log_redaction import sanitize_exception_for_log, summarize_text_for_log
from engines.routine_pipeline import generate_smart_routine
from engines.schemas import RoutineDraftResponse, RoutineFeedbackRequest, RoutineFeedbackResponse, RoutineRequest
from engines.nlp_engine import parse_natural_language_log

install_stdout_capture()

# ==========================================================
# 🚀 전역 변수 설정 (함수 밖에서는 'None'으로 이름만 선언)
# ==========================================================
whisper_model = None
supplement_rag = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    서버의 시작과 종료 시 딱 1번만 실행되는 수명 주기 관리 함수
    """
    global whisper_model, supplement_rag

    print("\n" + "="*40)
    print("🚀 [서버 시작] AI 모델 로딩을 시작합니다 (딱 1회만 로드됨)")

    # 1. Whisper 모델 로드 (WHISPER_PRELOAD=true 일 때만 시작 시 로드, 기본 false)
    if whisper_model is None and os.environ.get("WHISPER_PRELOAD", "false").strip().lower() == "true":
        print("🤖 1. Whisper AI(STT) 모델 로딩 중...")
        from faster_whisper import WhisperModel
        whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
        print("✅ Whisper 모델 로딩 완료!")
    elif os.environ.get("WHISPER_PRELOAD", "false").strip().lower() != "true":
        print("⏭️ 1. Whisper 모델 시작 로드 스킵 (WHISPER_PRELOAD=false)")

    # 2. 영양제 RAG 엔진 로드
    if supplement_rag is None:
        print("💊 2. 영양제 RAG 엔진 로딩 중...")
        try:
            with redirect_stderr(io.StringIO()):
                from engines.supplement_engine import SupplementRAGEngine

            # 💡 Tip: DB 경로가 'latest_index'를 포함하고 있는지 확인하세요.
                supplement_rag = SupplementRAGEngine(db_path="./data/chroma_db")
            print("✅ 영양제 RAG 엔진 로딩 완료!")
        except Exception as e:
            print("⚠️ RAG 엔진 로드 실패:", sanitize_exception_for_log(e))
            supplement_rag = None

    print("✅ 모든 모델 로드 완료! 서버가 준비되었습니다.")
    print("="*40 + "\n")

    yield  # 서버 가동 시작

    # 서버 종료 시 정리
    whisper_model = None
    supplement_rag = None
    print("\n👋 [서버 종료] AI 모델 메모리가 정리되었습니다.")

# FastAPI 앱 객체 생성 및 lifespan 연결
app = FastAPI(title="Fit-Core AI Server", lifespan=lifespan)

# --- CORS 설정 (AI_CORS_ALLOWED_ORIGINS env, 기본값: localhost:3000,3001) ---
_default_origins = "http://localhost:3000,http://localhost:3001"
_cors_origins = [o.strip() for o in os.environ.get("AI_CORS_ALLOWED_ORIGINS", _default_origins).split(",") if o.strip()]
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

# ==========================================================
# 🚨 [에러 핸들러] 422 검증 에러 발생 시 상세 로깅
# ==========================================================
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
            sanitized["ctx"] = {
                key: str(value)
                for key, value in sanitized["ctx"].items()
            }
        return sanitized

    sanitized_errors = [sanitize_error(error) for error in exc.errors()]
    print("\n" + "="*50)
    print("🚨 [데이터 검증 에러 (422)] 프론트엔드 데이터 입구컷 발생!")
    print(f"상세 원인: {sanitized_errors}")
    print("="*50 + "\n")
    return JSONResponse(status_code=422, content={"detail": sanitized_errors})


# ==========================================================
# 🚀 API 1: 맞춤형 AI 루틴 생성기
# ==========================================================
@app.post("/api/ai/generate-routine", response_model=RoutineDraftResponse, response_model_by_alias=True)
def api_generate_routine(req: RoutineRequest, db: Session = Depends(get_db)):
    try:
        print("\n✅ [루틴 생성 요청 수신]")
        profile = get_user_profile_context(db, req.user_id)
        recent_sets = get_recent_sets(db, req.user_id)
        return generate_smart_routine(req, db, profile=profile, recent_sets=recent_sets)

    except Exception as e:
        print("[Routine Error]", sanitize_exception_for_log(e))
        raise HTTPException(status_code=500, detail=str(e))


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


# ==========================================================
# 🚀 API 2: 자연어 퀵 로그 분석 (텍스트 -> 식단/운동 추출)
# ==========================================================
class LogRequest(BaseModel):
    text: str

@app.post("/api/ai/parse-log")
def api_parse_log(req: LogRequest):
    try:
        print("\n✅ [NLP 파싱 요청 수신] 텍스트:", summarize_text_for_log(req.text))
        result_json_str = parse_natural_language_log(req.text)
        return json.loads(result_json_str)

    except Exception as e:
        print("[NLP Error]", sanitize_exception_for_log(e))
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================================
# 🎙️ API 3: 오디오 파일 STT 변환 (음성 -> 텍스트)
# ==========================================================
@app.post("/api/ai/stt")
async def api_speech_to_text(audio_file: UploadFile = File(...)):
    temp_file_path = ""
    try:
        print(f"\n🎙️ [음성 인식 요청 수신] 파일명: {audio_file.filename}")
        audio_bytes = await audio_file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name

        # whisper_model이 None이면 요청 시점에 lazy 로드
        global whisper_model
        if whisper_model is None:
            print("🤖 [STT] Whisper 모델 lazy 로딩 중...")
            from faster_whisper import WhisperModel
            whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
            print("✅ [STT] Whisper 모델 로딩 완료!")
        segments, info = whisper_model.transcribe(
            temp_file_path,
            beam_size=5,
            language="ko"
        )

        transcript = " ".join([segment.text for segment in segments])
        print("🗣️ [Whisper 인식 결과]:", summarize_text_for_log(transcript))

        return {"text": transcript}

    except Exception as e:
        print("[STT Error]", sanitize_exception_for_log(e))
        raise HTTPException(status_code=500, detail="음성 인식 실패")

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# ==========================================================
# 💊 API 4: 영양제 및 약물 전문 AI 챗봇
# ==========================================================
class SupplementChatRequest(BaseModel):
    question: str

@app.post("/api/ai/supplement-chat")
def api_supplement_chat(req: SupplementChatRequest):
    # lifespan에서 로드된 supplement_rag 사용
    if not supplement_rag:
        raise HTTPException(
            status_code=503,
            detail="영양제 챗봇 엔진이 준비되지 않았습니다. DB 경로를 확인하세요."
        )

    try:
        print("\n💬 [영양제 질문 수신]:", summarize_text_for_log(req.question))
        result = supplement_rag.answer_question(req.question)
        return result

    except Exception as e:
        print("❌ [챗봇 에러]:", sanitize_exception_for_log(e))
        raise HTTPException(status_code=500, detail="답변 생성 중 오류가 발생했습니다.")


if __name__ == "__main__":
    # 문자열 형태로 전달해야 리로더가 정상적으로 작동하며 중복 로드를 피하기 쉬움
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

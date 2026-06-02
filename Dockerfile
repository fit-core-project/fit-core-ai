# =============================================================
# fit-core-ai  —  Python AI Server (slim / routine-only)
# =============================================================
# Runtime environment variables (values must be set externally):
#   DATABASE_URL            — SQLAlchemy connection string (MySQL/Postgres)
#   GOOGLE_API_KEY          — Gemini API key for LLM calls
#   LLM_PROVIDER=gemini     — LLM backend selector
#   APP_ENV=production      — disables /api/dev/logs endpoint
#   AI_CORS_ALLOWED_ORIGINS — comma-separated allowed origins (default: localhost:3000,3001)
#   WHISPER_PRELOAD         — set to "true" to preload Whisper at boot (default: false)
# =============================================================

FROM python:3.11-slim

# ── Unicode crash prevention ──────────────────────────────────
ENV PYTHONUTF8=1
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8

# ── Whisper 시작 로드 비활성화 (slim에는 faster-whisper 미포함) ──
ENV WHISPER_PRELOAD=false

# ── Non-root user ─────────────────────────────────────────────
RUN groupadd --gid 1001 appgroup \
 && useradd --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# ── System deps (kiwipiepy 빌드용 — ffmpeg 제외, slim에 STT 미포함) ──
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
 && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────
COPY requirements-slim.txt requirements-slim.txt
RUN pip install --no-cache-dir -r requirements-slim.txt

# ── Application source ────────────────────────────────────────
COPY --chown=appuser:appgroup . .

USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

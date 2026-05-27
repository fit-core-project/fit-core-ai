# Fit Core AI

FastAPI routine engine for Fit Core.

## Requirements

- Python 3.11+
- pip
- Optional: Ollama for local LLM mode
- Optional: Google Gemini API key for Gemini mode

## Environment

Create `.env` in this directory if one does not already exist.

Gemini mode:

```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your-google-api-key
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001
```

Local Ollama mode:

```env
LLM_PROVIDER=local
LOCAL_LLM_MODEL=gemma4:latest
OLLAMA_BASE_URL=http://localhost:11434
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001
```

Production guard:

```env
APP_ENV=production
```

When `APP_ENV=production`, `LLM_PROVIDER=local` is overridden to Gemini by `engines/llm_router.py`.

## Install

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements/dev.txt
```

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements/dev.txt
```

`requirements.txt` installs runtime dependencies. `requirements/dev.txt` adds test/evaluation tooling.

## Build / Verification

Python services do not have a compile build step. Use tests as the build gate:

```bash
pytest
```

Optional smoke run:

```bash
python scripts/smoke_routine.py
```

## Run Locally

With the virtual environment activated:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Or:

```bash
python main.py
```

Default URL:

```text
http://localhost:8000
```

FastAPI docs:

```text
http://localhost:8000/docs
```

## Local Ollama Setup

If using `LLM_PROVIDER=local`, start Ollama first and make sure the configured model exists:

```bash
ollama pull gemma4:latest
ollama serve
```

Then start the AI server with:

```bash
LLM_PROVIDER=local LOCAL_LLM_MODEL=gemma4:latest uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

PowerShell equivalent:

```powershell
$env:LLM_PROVIDER="local"
$env:LOCAL_LLM_MODEL="gemma4:latest"
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## Project Notes

- API entry point: `main.py`
- Routine pipeline: `engines/routine_pipeline.py`
- LLM routing: `engines/llm_router.py`
- Public request/response schemas: `engines/schemas.py`
- Exercise tier source data: `scripts/exercise_tier.xlsx` and generated CSV/data exports
- Tests: `tests/`

## Common Issues

- If imports fail, confirm the virtual environment is activated and dependencies were installed from both `requirements.txt` and `requirements/dev.txt`.
- If Gemini calls fail, confirm `GOOGLE_API_KEY` is set.
- If local LLM calls fail, confirm Ollama is running and `LOCAL_LLM_MODEL` is installed.
- The server loads Whisper and supplement RAG resources on startup; first startup can take longer than normal.

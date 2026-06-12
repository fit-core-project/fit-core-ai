# AI Demo Feature Modes

This document defines the demo behavior for quicklog, supplement chat, and STT.
The policy is full path first, fallback second. Fallbacks exist to prevent 500
responses during demos, not to disable the real feature.

## Quicklog

Full mode:

- Endpoint: `POST /api/ai/parse-log`
- Input: JSON body with `text`
- Path: `parse_natural_language_log()` calls `get_llm("nlp")`, applies
  `with_structured_output(ParsedDailyLog)`, and returns schema-compatible JSON.
- Expected response fields: `diet_logs`, `workout_logs`, `overall_summary`

Fallback mode:

- Used only when text is empty, the provider fails, structured parsing fails, or
  schema validation fails.
- Returns a `ParsedDailyLog`-compatible JSON response.
- Logs only redacted text metadata and sanitized exception categories.

## Supplement Chat

Full mode:

- Endpoint: `POST /api/ai/supplement-chat`
- `SupplementRAGEngine` attempts full RAG initialization first.
- Full mode requires:
  - Chroma index under `data/chroma_db/latest_index` or `doc_index_*`
  - `chromadb` / `langchain-chroma` support
  - embedding model dependencies
  - `sentence-transformers` reranker dependencies
  - Kiwi and BM25 dependencies
  - enough memory to load embedding and reranker models
- When ready, `answer_question()` calls `_answer_full()` and returns
  `mode="full"`.

Degraded mode:

- Used only when dependencies, index files, model loading, or runtime RAG calls
  fail.
- Returns `answer`, `sources`, `mode="degraded"`, and `degraded_reason`.
- Does not raise HTTP 500 for missing local RAG assets.

Current Docker note:

- The demo `Dockerfile` installs `requirements-slim.txt`.
- `requirements-slim.txt` intentionally excludes heavy RAG packages such as
  `chromadb`, `sentence-transformers`, and `faster-whisper`.
- Therefore the current slim container can expose the supplement endpoint safely,
  but full supplement RAG requires a fuller image or additional dependency layer.

## STT

Full mode:

- Endpoint: `POST /api/ai/stt`
- Multipart field: `audio_file`
- If `faster_whisper` and `ffmpeg` are available, the endpoint lazy-loads
  `WhisperModel("small", device="cpu", compute_type="int8")`, transcribes the
  uploaded file, and returns `text` with `status="success"`.

Unavailable fallback:

- Used only when STT dependencies are missing, ffmpeg is missing, model loading
  fails, or transcription fails.
- Returns HTTP 200 with `text=""`, `status="unavailable"`, and a user-facing
  message recommending text input.
- The backend `/api/ai/stt` route proxies multipart requests to the AI endpoint
  first and only uses its own unavailable response if the AI call itself fails.

Current Docker note:

- The current slim container does not install `ffmpeg` or `faster-whisper`.
- STT full mode requires both to be present, plus enough local memory and model
  cache/download access.

## Mode Checks

- Quicklog: inspect AI logs for `local -> ChatOllama(..., engine=nlp)` followed
  by either normal response or `LLM quicklog parsing fallback`.
- Supplement: response `mode` is `full` or `degraded`.
- STT: response `status` is `success` or `unavailable`.

## Privacy

Do not store raw prompts, raw LLM output, uploaded audio content, raw user text,
API keys, or full environment dumps. Logs should contain only redacted summaries,
aggregate counts, safe mode fields, and sanitized exception categories.

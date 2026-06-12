# Engine Module Layout

This layout keeps demo-facing AI features stable while avoiding a large routine refactor before the interview demo.

## Current Packages

- `engines/quicklog/`: natural-language daily log parsing. The parser uses the configured LLM when available and returns a schema-compatible fallback JSON when parsing fails.
- `engines/supplement/`: supplement answer engine. Full RAG mode is used when Chroma, embeddings, reranker, BM25, and local index files are available. If any dependency or index is missing, the engine enters degraded mode and returns safe general guidance instead of raising.
- `engines/`: existing routine generation modules remain in place for compatibility.
- `engines/prescription/`: existing routine prescription helpers remain in place.

## Compatibility Shims

These old imports remain valid:

```python
from engines.nlp_engine import parse_natural_language_log
from engines.supplement_engine import SupplementRAGEngine
```

They re-export the new package implementations:

```python
from engines.quicklog.nlp_engine import parse_natural_language_log
from engines.supplement.supplement_engine import SupplementRAGEngine
```

## Demo Degraded Mode Policy

Quicklog and supplement endpoints must not fail with HTTP 500 just because an LLM, vector index, web search tool, or optional heavy dependency is unavailable.

- Quicklog fallback returns `diet_logs`, `workout_logs`, and `overall_summary`.
- Supplement fallback returns `answer`, `sources`, `mode=degraded`, and `degraded_reason`.
- Raw prompts, raw model output, user text, API keys, and full exception messages must not be logged.

## P4 Follow-up

Move routine modules into `engines/routine/` after the demo, with a dedicated import migration and focused routine regression suite. This was intentionally deferred to avoid destabilizing routine generation, Gemma4 provider policy, feedback ranking, and Rule Critic behavior.

"""Stage-instrumented evaluation for the supplement-chat RAG.

For every case in tests/evaluation/datasets/supplement_eval_v1.jsonl the harness runs the real
in-process engine (SupplementRAGEngine.answer_question) and classifies the FIRST failing stage:

  S0 runtime   non-"full" mode (degraded / runtime error), LLM cache miss, or LLM timeout/error in live/cache mode
  S1 parse     parse_supplement_query entities / intents vs expected
  S2 retrieval expected_doc_ids must all be in the final docs handed to the composer
  S3 content   must_include_any / must_not_include / consult / out-of-KB honesty / generic template
  S4 composer  (attribution, not a gate) which composer path produced the answer

The generic-template check (S3) compares answers across the cases of the current run, so subset runs
(--split / --ids) flag fewer generic answers than a full run.

Engine files are not modified: the harness wraps module attributes at runtime.

LLM modes:
  --llm none   LLM returns "WEB_SEARCH_REQUIRED" -> deterministic composer paths only (no GPU/Ollama)
  --llm live   real LLM, raw outputs cached by prompt hash in tests/evaluation/llm_cache/supplement_llm_cache.json
  --llm cache  replay the cache only; misses are bucketed as S0 (llm_cache_miss)

Examples (cwd = fit-core-ai):
  ./venv/Scripts/python scripts/eval/supplement_stage_eval.py --lint
  ./venv/Scripts/python scripts/eval/supplement_stage_eval.py --llm none
  ./venv/Scripts/python scripts/eval/supplement_stage_eval.py --llm none --ids zn_fasting,fe_coffee
  ./venv/Scripts/python scripts/eval/supplement_stage_eval.py --llm live --llm-timeout 180
  ./venv/Scripts/python scripts/eval/supplement_stage_eval.py --llm cache
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATASET = PROJECT_ROOT / "tests" / "evaluation" / "datasets" / "supplement_eval_v1.jsonl"
ARTIFACT_DIR = PROJECT_ROOT / "tests" / "evaluation" / ".artifacts"
# Committed so any machine can replay the measured LLM outputs (--llm cache).
DEFAULT_CACHE = ARTIFACT_DIR.parent / "llm_cache" / "supplement_llm_cache.json"
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"

# Non-confirmation phrases. Deliberately excludes 제품명/성분표/함량: _enhance_answer appends
# "...제품명과 함량을 알려주면..." to many fallback answers, which would let templates pass.
HONESTY_MARKERS = (
    "확인되지 않",
    "확인할 수 없",
    "확인하기 어렵",
    "근거 문서에서 확인",
    "근거가 부족",
    "근거가 없",
    "근거가 충분하지",
    "자료가 없",
    "정보가 없",
    "정보가 부족",
    "알려진 근거",
    # Claim-refusal phrasing (2026-11-10 review: honest answers such as "불면증을 치료한다고
    # 단정하여 말씀드릴 수 없습니다" were scored as dishonest).
    "단정하여 말씀드릴 수 없",
    "단정적으로 말씀드리기",
    "말씀드리기 어렵",
    "명확히 말씀드리기",
    "자료에는",
    "자료만으로는",
    "정보는 없",
)
# Composer boilerplate appended to many answers; removed before the honesty check so it
# cannot make an invented answer look honest.
_COMPOSER_HEDGES = ("다만 개인 상태와 제품 함량에 따라 달라질 수 있어 안전하다고 단정하기는 어렵습니다.",)
# A banned claim quoted inside a refusal is not the claim ("'완치'한다고 단정하기 어렵습니다").
_NEGATION_MARKERS = ("단정하기 어렵", "단정할 수 없", "단정하여 말씀드릴 수 없", "단정적으로 말씀드리기 어렵",
                     "아닙니다", "않습니다", "어렵습니다", "없습니다", "말씀드리기 어렵")
# Same advice, different wording.
_SYNONYMS = (("식사 후", "식후"), ("식사후", "식후"))
# Composer templates that legitimately state non-confirmation.
_HONEST_TEMPLATE_ALLOWLIST = ("확보된 근거 문서에서 확인되지 않는", "확인되지 않은 제품이나 성분이 함께")
_CONSULT_WHO = re.compile(r"(의사|약사|의료진|전문가)")
_CONSULT_ACT = re.compile(r"(상담|확인|지시|알리)")
_SYMPTOM_TEMPLATE_PREFIX = "먹고 속이 안 좋다면"
_CANONICAL_ALIASES = {"probiotic": "probiotics", "whey protein": "protein", "fish oil": "omega3"}


# --------------------------------------------------------------------------------------------
# dataset / KB helpers (no heavy imports)
# --------------------------------------------------------------------------------------------

def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}")
    return cases


def load_kb() -> dict[str, dict[str, Any]]:
    from engines.supplement.kb_documents import load_curated_kb_records

    kb: dict[str, dict[str, Any]] = {}
    for text, meta, source in load_curated_kb_records(DOCUMENTS_DIR):
        kb[str(meta["id"])] = {"text": text, "meta": meta, "source": source}
    return kb


def _flatten_doc_ids(expected: list[Any]) -> list[str]:
    out: list[str] = []
    for item in expected or []:
        out.extend(item if isinstance(item, list) else [item])
    return out


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _with_synonyms(text: str) -> str:
    out = _norm_text(text)
    for a, b in _SYNONYMS:
        out = out.replace(a, f"{a} {b}")
    return out


def _term_hits(term: str, text: str) -> bool:
    if term.startswith("re:"):
        return re.search(term[3:], text or "") is not None
    return term.lower() in _with_synonyms(text)


def _banned_hits(term: str, text: str) -> bool:
    """Like _term_hits, but a plain term inside a refusing sentence does not count."""
    if term.startswith("re:"):
        return _term_hits(term, text)
    needle = term.lower()
    for sentence in re.split(r"(?<=[.!?])\s+|\n", text or ""):
        norm = _norm_text(sentence)
        if needle in norm and not any(marker in norm for marker in _NEGATION_MARKERS):
            return True
    return False


def _canon(value: str) -> str:
    value = (value or "").strip().lower()
    return _CANONICAL_ALIASES.get(value, value)


def _profile_sentences(kb: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """canonical -> KB bullet sentences (timing/spacing/cautions) of that ingredient profile."""
    out: dict[str, list[str]] = {}
    for doc_id, item in kb.items():
        if item["source"] != "ingredient_profile":
            continue
        canonical = _canon(item["meta"].get("canonical") or "")
        sentences = []
        for line in item["text"].splitlines():
            stripped = line.strip()
            if not stripped.startswith("- "):
                continue
            body = stripped.removeprefix("- ")
            body = re.sub(r"^(General|With food|Before bed|Exercise related):\s*", "", body)
            if len(body) >= 15:
                sentences.append(body.rstrip("."))
        out[canonical] = sentences
    return out


def lint(cases: list[dict[str, Any]], kb: dict[str, dict[str, Any]]) -> tuple[list[str], list[str]]:
    from engines.supplement.query_understanding import ALIAS_REGISTRY, IntentType

    errors: list[str] = []
    warnings: list[str] = []
    known_canonicals = {entry.canonical for entry in ALIAS_REGISTRY}
    intent_values = {intent.value for intent in IntentType}
    required = ("id", "question", "split", "tags", "expected_entities", "expected_intents", "answerable_from_kb",
                "expected_doc_ids", "must_include_any", "must_not_include")
    seen: set[str] = set()
    for case in cases:
        cid = case.get("id", "<no id>")
        for key in required:
            if key not in case:
                errors.append(f"{cid}: missing field {key}")
        if cid in seen:
            errors.append(f"{cid}: duplicate id")
        seen.add(cid)
        if case.get("split") not in {"dev", "test"}:
            errors.append(f"{cid}: bad split {case.get('split')}")
        for ent in case.get("expected_entities", []):
            if ent not in known_canonicals:
                errors.append(f"{cid}: expected entity '{ent}' is not a query_understanding canonical")
        for intent in [*case.get("expected_intents", []), *case.get("must_not_intents", [])]:
            if intent not in intent_values:
                errors.append(f"{cid}: unknown intent '{intent}'")
        doc_ids = _flatten_doc_ids(case.get("expected_doc_ids", []))
        for doc_id in doc_ids:
            if doc_id not in kb:
                errors.append(f"{cid}: expected doc id '{doc_id}' not in curated KB")
        kb_text = "\n".join(kb[d]["text"] for d in doc_ids if d in kb)
        question = case.get("question", "")
        groups = case.get("must_include_any", [])
        if case.get("answerable_from_kb") and not groups:
            errors.append(f"{cid}: answerable case without must_include_any")
        for group in groups:
            if not group:
                errors.append(f"{cid}: empty include group")
                continue
            if case.get("answerable_from_kb") and not any(_term_hits(t, kb_text) for t in group):
                errors.append(f"{cid}: include group {group} not supported by KB text of {doc_ids}")
            echoed = [t for t in group if _term_hits(t, question)]
            if echoed:
                errors.append(f"{cid}: include group {group} satisfiable by question echo ({echoed})")
        for term in case.get("must_not_include", []):
            if kb_text and _term_hits(term, kb_text):
                errors.append(f"{cid}: must_not term '{term}' occurs in its own expected KB docs")
        if not case.get("answerable_from_kb") and case.get("must_include_any"):
            warnings.append(f"{cid}: out-of-KB case also has must_include_any (checked in addition to honesty)")

    # Honesty markers must not be satisfiable by the composer's non-honest templates.
    import engines.supplement.response_composer as rc

    tree = ast.parse(Path(rc.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) > 40:
            text = node.value
            if any(marker in text for marker in HONESTY_MARKERS) and not any(a in text for a in _HONEST_TEMPLATE_ALLOWLIST):
                errors.append(f"composer template satisfies honesty markers: {text[:60]}...")
    return errors, warnings


# --------------------------------------------------------------------------------------------
# engine instrumentation
# --------------------------------------------------------------------------------------------

def _set_env(llm_mode: str, llm_timeout: float | None) -> None:
    # Must happen before engine modules import (load_dotenv() does not override existing vars).
    os.environ["AI_EMBEDDING_DEVICE"] = "cpu"  # same device for live/cache/none so prompt hashes match
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["SUPPLEMENT_ENABLE_LLM_QUERY_OPTIMIZER"] = "false"
    os.environ["SUPPLEMENT_ENABLE_WEB_FALLBACK"] = "false"
    os.environ["APP_ENV"] = "local"
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if llm_timeout:
        os.environ["SUPPLEMENT_LLM_TIMEOUT_SEC"] = str(llm_timeout)


def _resolve_db_path() -> Path:
    raw = os.environ.get("SUPPLEMENT_RAG_DB_PATH") or "./data/chroma_db"
    path = Path(raw)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if not ((path / "latest_index").exists() or list(path.glob("doc_index_*"))):
        raise SystemExit(f"refusing to start: no latest_index/doc_index_* under {path} (would create a stray Chroma DB)")
    return path


class LLMCache:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.data: dict[str, Any] = {"entries": {}, "case_index": {}}
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
            self.data.setdefault("entries", {})
            self.data.setdefault("case_index", {})

    def get(self, key: str) -> dict[str, Any] | None:
        return self.data["entries"].get(key)

    def put(self, key: str, entry: dict[str, Any]) -> None:
        with self.lock:
            self.data["entries"][key] = entry
            self._save()

    def index_case(self, case_id: str, key: str) -> None:
        with self.lock:
            self.data["case_index"][case_id] = key

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def save(self) -> None:
        with self.lock:
            self._save()


class LLMProxy:
    """Stands in for engine.llm inside `local_prompt | llm | StrOutputParser()`."""

    def __init__(self, mode: str, real_llm: Any, cache: LLMCache | None):
        self.mode = mode
        self.real_llm = real_llm
        self.cache = cache
        self.current: dict[str, Any] = {}
        model = getattr(real_llm, "model", None) or getattr(real_llm, "model_name", None) or type(real_llm).__name__
        self.model_id = f"{model}|t={getattr(real_llm, 'temperature', None)}"

    def prompt_key(self, prompt_value: Any) -> str:
        messages = prompt_value.to_messages() if hasattr(prompt_value, "to_messages") else [prompt_value]
        payload = [[getattr(m, "type", "str"), getattr(m, "content", str(m))] for m in messages]
        raw = json.dumps({"model": self.model_id, "messages": payload}, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def __call__(self, prompt_value: Any) -> Any:
        from langchain_core.messages import AIMessage

        rec = self.current  # bound at entry: a timed-out worker must not write into the next case
        key = self.prompt_key(prompt_value)
        rec["llm_prompt_hash"] = key
        if self.cache is not None and rec.get("case_id"):
            self.cache.index_case(rec["case_id"], key)
        if self.mode == "none":
            rec["llm_source"] = "stub"
            return AIMessage(content="WEB_SEARCH_REQUIRED")
        hit = self.cache.get(key) if self.cache else None
        if self.mode == "cache":
            if hit is None:
                rec["llm_source"] = "cache_miss"
                return AIMessage(content="")
            rec["llm_source"] = "cache"
            rec["llm_raw"] = hit["output"]
            return AIMessage(content=hit["output"])
        # live
        if hit is not None:
            rec["llm_source"] = "cache"
            rec["llm_raw"] = hit["output"]
            return AIMessage(content=hit["output"])
        started = time.perf_counter()
        result = self.real_llm.invoke(prompt_value)
        content = result.content if hasattr(result, "content") else str(result)
        rec["llm_source"] = "live"
        rec["llm_raw"] = content
        rec["llm_latency_ms"] = round((time.perf_counter() - started) * 1000)
        if self.cache is not None:
            self.cache.put(key, {
                "output": content,
                "model": self.model_id,
                "case_id": rec.get("case_id"),
                "question": rec.get("question"),
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
        return AIMessage(content=content)


class _NoWebSearch:
    def run(self, *_args: Any, **_kwargs: Any) -> str:
        raise RuntimeError("web search disabled in supplement_stage_eval")


class _NoQueryOptimizer:
    def invoke(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("query optimizer disabled in supplement_stage_eval")


class Harness:
    def __init__(self, llm_mode: str, cache_path: Path):
        self.llm_mode = llm_mode
        self.current: dict[str, Any] = {}
        self.cache = LLMCache(cache_path) if llm_mode in {"live", "cache"} else None
        if llm_mode == "cache" and not cache_path.exists():
            raise SystemExit(f"--llm cache: cache file not found: {cache_path}")
        self.db_path = _resolve_db_path()

        import engines.supplement.response_composer as rc
        import engines.supplement.supplement_engine as se
        from langchain_core.runnables import RunnableLambda

        self.rc = rc
        self.se = se
        engine = se.SupplementRAGEngine(db_path=str(self.db_path))
        if not engine.ready:
            reason = engine.degraded_reason
            try:
                engine._init_full_rag(str(self.db_path))
            except Exception:
                traceback.print_exc()
            raise SystemExit(f"engine not ready (degraded_reason={reason}); aborting")
        self.engine = engine
        self.proxy = LLMProxy(llm_mode, engine.llm, self.cache)
        engine.llm = RunnableLambda(self.proxy)
        engine.query_optimizer = _NoQueryOptimizer()
        engine.web_search = _NoWebSearch()
        self._install_patches()

    # ---- patches -------------------------------------------------------------------------
    def _install_patches(self) -> None:
        rc, se, harness = self.rc, self.se, self

        orig_compose = se.compose_supplement_response

        def compose_wrapper(*args: Any, **kwargs: Any) -> Any:
            rec = harness.current
            rec["parsed_query"] = kwargs.get("parsed_query")
            rec["final_docs"] = list(kwargs.get("selected_docs") or [])
            rec["generated_answer"] = kwargs.get("generated_answer")
            rec["fallback_calls"] = []
            rec["enhance_direct_calls"] = []
            rec["soften_events"] = []
            result = orig_compose(*args, **kwargs)
            rec["composed"] = result
            return result

        se.compose_supplement_response = compose_wrapper

        orig_priority = self.engine._priority_kb_docs

        def priority_wrapper(parsed_query: Any) -> list[Any]:
            docs = orig_priority(parsed_query)
            harness.current["priority_doc_ids"] = [str((d.metadata or {}).get("id")) for d in docs]
            return docs

        self.engine._priority_kb_docs = priority_wrapper

        orig_fallback = rc._fallback_answer_from_docs
        orig_direct = rc._direct_answer_from_query_and_docs
        orig_grounded = rc._grounded_answer_from_entity_docs
        orig_soften = rc._soften_unsafe_answer

        def fallback_wrapper(*args: Any, **kwargs: Any) -> str:
            rec = harness.current
            frame: dict[str, Any] = {"direct": None, "grounded": None}
            rec["_fallback_frame"] = frame
            try:
                result = orig_fallback(*args, **kwargs)
            finally:
                rec.pop("_fallback_frame", None)
            frame["result"] = result
            rec.setdefault("fallback_calls", []).append(frame)
            return result

        def direct_wrapper(*args: Any, **kwargs: Any) -> str:
            result = orig_direct(*args, **kwargs)
            frame = harness.current.get("_fallback_frame")
            if frame is not None and frame["direct"] is None:
                frame["direct"] = result
            elif frame is None:
                harness.current.setdefault("enhance_direct_calls", []).append(result)
            return result

        def grounded_wrapper(*args: Any, **kwargs: Any) -> str:
            result = orig_grounded(*args, **kwargs)
            frame = harness.current.get("_fallback_frame")
            if frame is not None and frame["grounded"] is None:
                frame["grounded"] = result
            return result

        def soften_wrapper(answer: str) -> str:
            result = orig_soften(answer)
            if (result or "").strip() != (answer or "").strip():
                harness.current.setdefault("soften_events", []).append({"before": answer, "after": result})
            return result

        rc._fallback_answer_from_docs = fallback_wrapper
        rc._direct_answer_from_query_and_docs = direct_wrapper
        rc._grounded_answer_from_entity_docs = grounded_wrapper
        rc._soften_unsafe_answer = soften_wrapper

    # ---- run -----------------------------------------------------------------------------
    def run_case(self, case: dict[str, Any]) -> dict[str, Any]:
        rec: dict[str, Any] = {"case_id": case["id"], "question": case["question"]}
        self.current = rec
        self.proxy.current = rec
        started = time.perf_counter()
        try:
            rec["result"] = self.engine.answer_question(case["question"])
        except Exception as exc:  # answer_question already swallows; this is a harness bug guard
            rec["result"] = {"mode": "harness_error", "answer": "", "error": repr(exc)}
        rec["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
        return rec

    def composer_path(self, rec: dict[str, Any]) -> str:
        rc = self.rc
        result = rec.get("result") or {}
        if result.get("mode") != "full":
            return "degraded"
        composed = rec.get("composed")
        if composed is None:
            return "unknown"
        if not composed.used_kb_fallback:
            return "llm_softened" if rec.get("soften_events") else "llm_kept"
        calls = rec.get("fallback_calls") or []
        if not calls:
            return "unknown"
        first = calls[0]
        direct, grounded, final = first.get("direct"), first.get("grounded"), first.get("result")
        prefix = "llm_rejected->" if (rec.get("generated_answer") or "").strip() else ""
        if direct:
            if direct == rc._GOAL_RECOMMENDATION_ANSWER:
                return prefix + "direct:goal_template"
            if direct.startswith(_SYMPTOM_TEMPLATE_PREFIX):
                return prefix + "direct:symptom_template"
            return prefix + "direct:rule_specific"
        if grounded:
            return prefix + "grounded"
        if final == rc._UNKNOWN_INGREDIENT_ANSWER:
            return prefix + "unknown_ingredient_template"
        if final == rc._GOAL_RECOMMENDATION_ANSWER:
            return prefix + "goal_template"
        return prefix + "generic_template"


# --------------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------------

def _doc_id(doc: Any) -> str:
    return str(((getattr(doc, "metadata", None) or {}).get("id")) or "")


def score_case(case: dict[str, Any], rec: dict[str, Any], harness: Harness, kb: dict[str, dict[str, Any]],
               profile_sentences: dict[str, list[str]]) -> dict[str, Any]:
    result = rec.get("result") or {}
    answer = str(result.get("answer") or "")
    caution = str(result.get("caution") or "")
    combined = f"{answer} {caution}".strip()
    out: dict[str, Any] = {
        "id": case["id"],
        "split": case["split"],
        "tags": case.get("tags", []),
        "question": case["question"],
        "answerable_from_kb": case.get("answerable_from_kb"),
        "mode": result.get("mode"),
        "degraded_reason": result.get("degraded_reason"),
        "answer": answer,
        "caution": caution,
        "elapsed_ms": rec.get("elapsed_ms"),
        "llm_source": rec.get("llm_source"),
        "llm_prompt_hash": rec.get("llm_prompt_hash"),
        "llm_raw": rec.get("llm_raw"),
        "debug_counts": {k: v for k, v in (result.get("debugCounts") or {}).items()
                         if k in {"parsedIntents", "priorityKbDocs", "finalDocs", "llmAnswerFallbackReason", "answerRecoveredFromKbDocs"}},
    }

    # S0
    s0_reasons = []
    if result.get("mode") != "full":
        s0_reasons.append(f"mode={result.get('mode')} reason={result.get('degraded_reason')}")
    if rec.get("llm_source") == "cache_miss":
        s0_reasons.append("llm_cache_miss")
    elif harness.llm_mode in {"live", "cache"} and result.get("mode") == "full" and (
        rec.get("llm_source") not in {"live", "cache"}
        or (result.get("debugCounts") or {}).get("llmAnswerFallbackReason") == "generation_error"
    ):
        # LLM timed out / errored -> composer fell back deterministically; never score that as LLM content.
        s0_reasons.append("llm_timeout_or_error")
    out["s0"] = {"pass": not s0_reasons, "reasons": s0_reasons}

    # S1 (from the parsed query the engine actually used)
    parsed = rec.get("parsed_query")
    if parsed is None:
        from engines.supplement.query_understanding import parse_supplement_query

        parsed = parse_supplement_query(case["question"])
    parsed_entities = sorted({e.canonical for e in parsed.entities})
    parsed_supplements = {e.canonical for e in parsed.entities if e.type.value == "supplement_ingredient"}
    parsed_intents = sorted({i.value for i in parsed.intents})
    expected_entities = set(case.get("expected_entities", []))
    missing_entities = sorted(expected_entities - set(parsed_entities))
    unexpected_supplements = sorted(parsed_supplements - expected_entities)
    missing_intents = sorted(set(case.get("expected_intents", [])) - set(parsed_intents))
    forbidden_intents = sorted(set(case.get("must_not_intents", [])) & set(parsed_intents))
    s1_reasons = []
    if missing_entities:
        s1_reasons.append(f"missing_entities={missing_entities}")
    if unexpected_supplements:
        s1_reasons.append(f"unexpected_supplements={unexpected_supplements}")
    if missing_intents:
        s1_reasons.append(f"missing_intents={missing_intents}")
    if forbidden_intents:
        s1_reasons.append(f"forbidden_intents={forbidden_intents}")
    out["s1"] = {"pass": not s1_reasons, "reasons": s1_reasons, "entities": parsed_entities, "intents": parsed_intents}

    # S2
    final_docs = rec.get("final_docs") or []
    final_ids = [_doc_id(d) for d in final_docs]
    missing_docs = []
    for item in case.get("expected_doc_ids", []):
        options = item if isinstance(item, list) else [item]
        if not any(option in final_ids for option in options):
            missing_docs.append(item)
    stale = []
    for doc in final_docs:
        doc_id = _doc_id(doc)
        if doc_id in kb and (getattr(doc, "page_content", "") or "").strip() != kb[doc_id]["text"].strip():
            stale.append(doc_id)
    priority_ids = rec.get("priority_doc_ids") or []
    out["s2"] = {
        "pass": not missing_docs and rec.get("final_docs") is not None,
        "reasons": ([f"missing_docs={missing_docs}"] if missing_docs else []) + ([] if rec.get("final_docs") is not None else ["composer_not_reached"]),
        "final_doc_ids": final_ids,
        "priority_doc_ids": priority_ids,
        "missing_from_priority": [d for d in _flatten_doc_ids(case.get("expected_doc_ids", [])) if d not in priority_ids],
        "stale_curated_copies": stale,
    }

    # S3
    s3_reasons = []
    missing_groups = [g for g in case.get("must_include_any", []) if not any(_term_hits(t, combined) for t in g)]
    answer_only_missing = [g for g in case.get("must_include_any", []) if not any(_term_hits(t, answer) for t in g)]
    if missing_groups:
        s3_reasons.append(f"missing_include={missing_groups}")
    banned = [t for t in case.get("must_not_include", []) if _banned_hits(t, combined)]
    if banned:
        s3_reasons.append(f"banned_present={banned}")
    consult_ok = bool(_CONSULT_WHO.search(combined) and _CONSULT_ACT.search(combined))
    if case.get("must_consult") and not consult_ok:
        s3_reasons.append("consult_missing")
    honesty_ok = None
    borrowed: list[str] = []
    if not case.get("answerable_from_kb"):
        honesty_text = combined
        for hedge in _COMPOSER_HEDGES:
            honesty_text = honesty_text.replace(hedge, " ")
        honesty_ok = any(marker in honesty_text for marker in HONESTY_MARKERS)
        if not honesty_ok:
            s3_reasons.append("out_of_kb_not_honest")
        allowed = {_canon(e) for e in case.get("expected_entities", [])}
        norm_answer = _norm_text(answer)
        for canonical, sentences in profile_sentences.items():
            if canonical in allowed:
                continue
            for sentence in sentences:
                if _norm_text(sentence) in norm_answer:
                    borrowed.append(f"{canonical}: {sentence[:50]}")
        if borrowed:
            s3_reasons.append("borrowed_unrelated_kb_text")
    out["s3"] = {
        "pass": None,  # finalized after the generic-template pass
        "reasons": s3_reasons,
        "answer_only_missing_include": answer_only_missing,
        "consult_present": consult_ok,
        "honesty_present": honesty_ok,
        "borrowed_kb_text": borrowed,
    }
    out["s4_composer_path"] = harness.composer_path(rec)
    out["soften_events"] = len(rec.get("soften_events") or [])
    return out


_TEMPLATE_PATHS = {"direct:goal_template", "direct:symptom_template", "goal_template", "generic_template"}


def _supplement_key(case: dict[str, Any]) -> frozenset:
    from engines.supplement.query_understanding import ALIAS_REGISTRY, EntityType

    supplements = {e.canonical for e in ALIAS_REGISTRY if e.entity_type == EntityType.SUPPLEMENT_INGREDIENT}
    return frozenset(e for e in case.get("expected_entities", []) if e in supplements)


def finalize(scored: list[dict[str, Any]], cases_by_id: dict[str, dict[str, Any]]) -> None:
    # "Generic" = the same answer text is returned for questions about different supplement ingredients.
    groups: dict[str, set[frozenset]] = defaultdict(set)
    question_counts: Counter = Counter()
    for item in scored:
        key = re.sub(r"[\s.,!?]", "", item["answer"])
        if key:
            groups[key].add(_supplement_key(cases_by_id[item["id"]]))
            question_counts[key] += 1
    for item in scored:
        case = cases_by_id[item["id"]]
        key = re.sub(r"[\s.,!?]", "", item["answer"])
        path = item["s4_composer_path"].removeprefix("llm_rejected->")
        cross_entity = len(groups.get(key, set())) >= 2
        generic = bool(case.get("answerable_from_kb")) and (cross_entity or path in _TEMPLATE_PATHS)
        item["s3"]["generic_template"] = generic
        item["s3"]["identical_answer_across_entities"] = cross_entity
        item["s3"]["identical_answer_case_count"] = question_counts.get(key, 0)
        if generic:
            item["s3"]["reasons"].append("generic_template_answer")
        item["s3"]["pass"] = not item["s3"]["reasons"]
        first = None
        for stage in ("s0", "s1", "s2", "s3"):
            if not item[stage]["pass"]:
                first = stage.upper()
                break
        item["first_failing_stage"] = first or "PASS"


# --------------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------------

def _rate(n: int, d: int) -> str:
    return f"{n}/{d} ({(100.0 * n / d) if d else 0:.1f}%)"


def build_report(scored: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    by_split: dict[str, Any] = {}
    for split in ("all", "dev", "test"):
        items = [s for s in scored if split == "all" or s["split"] == split]
        if not items:
            continue
        by_split[split] = {
            "n": len(items),
            "pass": sum(1 for s in items if s["first_failing_stage"] == "PASS"),
            "first_failing_stage": dict(Counter(s["first_failing_stage"] for s in items)),
            "stage_pass_independent": {st: sum(1 for s in items if s[st]["pass"]) for st in ("s0", "s1", "s2", "s3")},
            "composer_paths": dict(Counter(s["s4_composer_path"] for s in items)),
            "s3_pass_but_answer_only_include_fails": [
                s["id"] for s in items if s["s3"]["pass"] and s["s3"]["answer_only_missing_include"]
            ],
        }
    by_tag: dict[str, Any] = {}
    tags = sorted({t for s in scored for t in s["tags"]})
    for tag in tags:
        items = [s for s in scored if tag in s["tags"]]
        by_tag[tag] = {
            "n": len(items),
            "pass": sum(1 for s in items if s["first_failing_stage"] == "PASS"),
            "first_failing_stage": dict(Counter(s["first_failing_stage"] for s in items)),
        }
    path_outcomes: dict[str, Any] = {}
    for s in scored:
        entry = path_outcomes.setdefault(s["s4_composer_path"], {"n": 0, "s3_pass": 0, "s3_fail": 0, "s3_pass_caution_carried": 0})
        entry["n"] += 1
        entry["s3_pass" if s["s3"]["pass"] else "s3_fail"] += 1
        if s["s3"]["pass"] and s["s3"]["answer_only_missing_include"]:
            entry["s3_pass_caution_carried"] += 1
    reason_counts: dict[str, Counter] = {st: Counter() for st in ("s0", "s1", "s2", "s3")}
    for s in scored:
        for st in reason_counts:
            for reason in s[st]["reasons"]:
                reason_counts[st][reason.split("=")[0]] += 1
    return {
        "meta": meta,
        "summary": by_split,
        "by_tag": by_tag,
        "composer_path_outcomes": path_outcomes,
        "reason_counts": {k: dict(v) for k, v in reason_counts.items()},
        "stale_curated_copy_cases": [s["id"] for s in scored if s["s2"]["stale_curated_copies"]],
        "cases": scored,
    }


def render_md(report: dict[str, Any]) -> str:
    meta = report["meta"]
    lines = [
        "# Supplement stage eval",
        "",
        f"- generated: {meta['generated_at']}",
        f"- llm mode: `{meta['llm_mode']}`  dataset: `{meta['dataset']}` (sha256 {meta['dataset_sha256'][:12]})",
        f"- cases run: {meta['n_cases']}  device: {meta.get('device')}  db: `{meta.get('db_dir')}`  docs in corpus: {meta.get('corpus_docs')}",
        f"- llm sources: {meta.get('llm_sources')}",
        "",
        "## Summary by split",
        "",
        "| split | n | pass (all stages) | S0 | S1 | S2 | S3 | S1 pass | S2 pass | S3 pass | S3 pass only via caution |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for split, s in report["summary"].items():
        ff = s["first_failing_stage"]
        ind = s["stage_pass_independent"]
        lines.append(
            f"| {split} | {s['n']} | {_rate(s['pass'], s['n'])} | {ff.get('S0', 0)} | {ff.get('S1', 0)} | {ff.get('S2', 0)} | {ff.get('S3', 0)} "
            f"| {_rate(ind['s1'], s['n'])} | {_rate(ind['s2'], s['n'])} | {_rate(ind['s3'], s['n'])} "
            f"| {len(s['s3_pass_but_answer_only_include_fails'])} |"
        )
    lines += [
        "",
        "First-failing-stage columns count each case once; `Sx pass` columns are independent per-stage pass rates.",
        "`S3 pass only via caution`: must_include_any is met by answer+caution but not by the answer alone "
        "(caution is mostly KB text copied from final docs). Ids: "
        + ", ".join(report["summary"].get("all", {}).get("s3_pass_but_answer_only_include_fails", [])),
        "The generic-template check compares answers within the current run only; subset runs (--split/--ids) flag fewer.",
        "",
    ]
    lines += ["## Composer path (S4 attribution)", "", "| path | n | S3 pass | S3 fail | S3 pass only via caution |", "|---|---|---|---|---|"]
    for path, v in sorted(report["composer_path_outcomes"].items(), key=lambda kv: -kv[1]["n"]):
        lines.append(f"| {path} | {v['n']} | {v['s3_pass']} | {v['s3_fail']} | {v['s3_pass_caution_carried']} |")
    lines += ["", "## Failure reasons (all cases, any stage)", ""]
    for stage, counts in report["reason_counts"].items():
        if counts:
            lines.append(f"- **{stage.upper()}**: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
    if report["stale_curated_copy_cases"]:
        lines.append(f"- stale curated doc copies reached final docs in: {', '.join(report['stale_curated_copy_cases'])}")
    lines += ["", "## By tag", "", "| tag | n | pass | S0 | S1 | S2 | S3 |", "|---|---|---|---|---|---|---|"]
    for tag, v in report["by_tag"].items():
        ff = v["first_failing_stage"]
        lines.append(f"| {tag} | {v['n']} | {_rate(v['pass'], v['n'])} | {ff.get('S0', 0)} | {ff.get('S1', 0)} | {ff.get('S2', 0)} | {ff.get('S3', 0)} |")
    lines += ["", "## Failure details", ""]
    for s in report["cases"]:
        if s["first_failing_stage"] == "PASS":
            continue
        reasons = []
        for st in ("s0", "s1", "s2", "s3"):
            reasons.extend(f"{st.upper()}:{r}" for r in s[st]["reasons"])
        lines.append(f"### {s['id']} [{s['split']}] first={s['first_failing_stage']} path={s['s4_composer_path']}")
        lines.append(f"- Q: {s['question']}")
        lines.append(f"- reasons: {'; '.join(reasons)}")
        lines.append(f"- parsed: entities={s['s1']['entities']} intents={s['s1']['intents']}")
        lines.append(f"- final docs: {s['s2']['final_doc_ids']}")
        lines.append(f"- A: {s['answer'][:400]}")
        if s["caution"]:
            lines.append(f"- caution: {s['caution'][:300]}")
        lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--llm", choices=["none", "live", "cache"], default="none")
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--split", choices=["all", "dev", "test"], default="all")
    parser.add_argument("--ids", default="", help="comma-separated case ids")
    parser.add_argument("--tags", default="", help="comma-separated tags (any match)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-prefix", default="supplement_stage_report", help="artifact basename (in tests/evaluation/.artifacts)")
    parser.add_argument("--llm-timeout", type=float, default=None, help="SUPPLEMENT_LLM_TIMEOUT_SEC override for live runs")
    parser.add_argument("--lint", action="store_true", help="validate the dataset against the KB and exit")
    parser.add_argument("--skip-lint", action="store_true")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    _set_env(args.llm, args.llm_timeout)

    cases = load_cases(args.dataset)
    kb = load_kb()
    if args.lint or not args.skip_lint:
        errors, warnings = lint(cases, kb)
        for w in warnings:
            print(f"[lint warn] {w}")
        for e in errors:
            print(f"[lint error] {e}")
        print(f"[lint] {len(cases)} cases, {len(errors)} errors, {len(warnings)} warnings")
        if args.lint:
            return 1 if errors else 0
        if errors:
            print("[lint] fix errors or pass --skip-lint")
            return 1

    selected = cases
    if args.split != "all":
        selected = [c for c in selected if c["split"] == args.split]
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        selected = [c for c in selected if c["id"] in wanted]
    if args.tags:
        wanted_tags = {x.strip() for x in args.tags.split(",") if x.strip()}
        selected = [c for c in selected if wanted_tags & set(c.get("tags", []))]
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        print("no cases selected")
        return 1

    harness = Harness(args.llm, args.cache_path)
    profile_sentences = _profile_sentences(kb)
    cases_by_id = {c["id"]: c for c in cases}
    scored = []
    for idx, case in enumerate(selected, 1):
        rec = harness.run_case(case)
        item = score_case(case, rec, harness, kb, profile_sentences)
        scored.append(item)
        print(f"[{idx}/{len(selected)}] {case['id']} mode={item['mode']} path={item['s4_composer_path']} {item['elapsed_ms']}ms", flush=True)
    finalize(scored, cases_by_id)
    if harness.cache is not None:
        harness.cache.save()

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "llm_mode": args.llm,
        "dataset": str(args.dataset.relative_to(PROJECT_ROOT) if args.dataset.is_relative_to(PROJECT_ROOT) else args.dataset),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "n_cases": len(scored),
        "filters": {"split": args.split, "ids": args.ids, "tags": args.tags, "limit": args.limit},
        "device": getattr(harness.engine, "device", None),
        "db_dir": str(getattr(harness.engine, "db_dir", "")),
        "corpus_docs": len(getattr(harness.engine, "original_docs", []) or []),
        "llm_model": harness.proxy.model_id,
        "llm_sources": dict(Counter(s["llm_source"] for s in scored)),
        "cache_path": str(args.cache_path) if harness.cache else None,
    }
    report = build_report(scored, meta)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = ARTIFACT_DIR / f"{args.out_prefix}.json"
    md_path = ARTIFACT_DIR / f"{args.out_prefix}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    md_path.write_text(render_md(report), encoding="utf-8")
    s = report["summary"]["all"]
    print(f"\nPASS {_rate(s['pass'], s['n'])}  first-fail={s['first_failing_stage']}  paths={s['composer_paths']}")
    print(f"report: {json_path}\n        {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

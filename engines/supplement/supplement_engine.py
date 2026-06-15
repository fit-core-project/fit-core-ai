from __future__ import annotations

import json
import time
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from engines.llm_router import get_llm
from engines.log_redaction import sanitize_exception_for_log

load_dotenv()

os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    str(Path(__file__).resolve().parents[2] / ".numba_cache"),
)


_DEGRADED_ANSWER = (
    "현재 영양제 지식베이스가 준비되지 않아 일반적인 안전 안내만 제공할 수 있습니다. "
    "복용 중인 약, 질환, 임신 여부가 있거나 이상 증상이 있다면 영양제 복용 전 약사나 의사와 상담하세요. "
    "제품 라벨의 1일 섭취량을 넘기지 말고, 여러 제품을 함께 복용할 때는 중복 성분을 확인하세요."
)
_DEFAULT_CAUTION = (
    "복용 중인 약, 질환, 임신/수유 여부가 있거나 고용량 복용을 고려 중이라면 "
    "의사 또는 약사와 상담하세요."
)
_TRUE_VALUES = {"true", "1", "yes", "y", "on"}
_KEYWORD_MAP = {
    "크레아틴": ["크레아틴", "creatine", "복용", "섭취", "타이밍", "운동 전", "운동 후", "로딩", "용량"],
    "creatine": ["크레아틴", "creatine", "복용", "섭취", "타이밍", "운동 전", "운동 후", "로딩", "용량"],
    "카페인": ["카페인", "caffeine", "부스터", "프리워크아웃", "저녁", "수면", "심박", "불면"],
    "caffeine": ["카페인", "caffeine", "부스터", "프리워크아웃", "저녁", "수면", "심박", "불면"],
    "단백질": ["단백질", "프로틴", "protein", "whey", "섭취량", "운동 후"],
    "프로틴": ["단백질", "프로틴", "protein", "whey", "섭취량", "운동 후"],
    "protein": ["단백질", "프로틴", "protein", "whey", "섭취량", "운동 후"],
    "오메가3": ["오메가3", "omega-3", "EPA", "DHA"],
    "omega": ["오메가3", "omega-3", "EPA", "DHA"],
    "비타민d": ["비타민D", "vitamin D"],
    "vitamin d": ["비타민D", "vitamin D"],
}


_KEYWORD_MAP.update({
    "마그네슘": ["마그네슘", "magnesium", "복용 시간", "식후", "저녁", "신장질환", "항생제", "갑상선약", "간격"],
    "magnesium": ["마그네슘", "magnesium", "복용 시간", "식후", "저녁", "kidney disease", "antibiotics", "thyroid medication"],
    "아연": ["아연", "zinc", "공복", "식후", "속쓰림", "항생제", "철분", "구리", "간격"],
    "zinc": ["아연", "zinc", "empty stomach", "with food", "nausea", "antibiotics", "iron", "copper"],
    "철분": ["철분", "iron", "공복", "비타민C", "칼슘", "카페인", "커피", "갑상선약", "간격"],
    "iron": ["철분", "iron", "vitamin c", "calcium", "caffeine", "coffee", "thyroid medication", "spacing"],
    "비타민b": ["비타민B", "비타민B군", "vitamin b", "b-complex", "식후", "아침", "소변색"],
    "비타민b군": ["비타민B", "비타민B군", "vitamin b", "b-complex", "식후", "아침"],
    "vitamin b": ["비타민B", "비타민B군", "vitamin b", "b-complex", "with food", "morning"],
    "b-complex": ["비타민B", "비타민B군", "vitamin b", "b-complex", "with food", "morning"],
    "비타민d": ["비타민D", "vitamin d", "지용성", "식후", "지방", "칼슘", "고용량", "임신"],
    "vitamin d": ["비타민D", "vitamin d", "fat soluble", "with meal", "calcium", "high dose", "pregnancy"],
    "오메가3": ["오메가3", "omega-3", "fish oil", "EPA", "DHA", "식후", "출혈", "항응고제", "와파린"],
    "omega": ["오메가3", "omega-3", "fish oil", "EPA", "DHA", "with meal", "bleeding", "warfarin"],
    "fish oil": ["오메가3", "omega-3", "fish oil", "EPA", "DHA", "with meal", "bleeding", "warfarin"],
    "크레아틴": ["크레아틴", "creatine", "운동 전", "운동 후", "꾸준히", "수분", "로딩", "근력"],
    "creatine": ["크레아틴", "creatine", "pre workout", "post workout", "daily", "hydration", "loading", "strength"],
    "칼슘": ["칼슘", "calcium", "식후", "마그네슘", "철분", "갑상선약", "간격", "결석"],
    "calcium": ["칼슘", "calcium", "with food", "magnesium", "iron", "thyroid medication", "spacing", "kidney stone"],
    "유산균": ["유산균", "probiotics", "프로바이오틱스", "공복", "식전", "식후", "항생제", "간격"],
    "프로바이오틱스": ["유산균", "probiotics", "프로바이오틱스", "공복", "식전", "식후", "항생제", "간격"],
    "probiotics": ["유산균", "probiotics", "프로바이오틱스", "empty stomach", "with food", "antibiotics", "spacing"],
    "밀크씨슬": ["밀크씨슬", "milk thistle", "실리마린", "silymarin", "간", "간질환", "약물상호작용"],
    "milk thistle": ["밀크씨슬", "milk thistle", "실리마린", "silymarin", "liver", "interaction"],
    "루테인": ["루테인", "lutein", "지용성", "식후", "눈 건강", "카로티노이드"],
    "lutein": ["루테인", "lutein", "fat soluble", "with meal", "eye health", "carotenoid"],
    "프로틴": ["프로틴", "단백질 보충제", "protein", "whey protein", "운동 전", "운동 후", "총 단백질"],
    "단백질 보충제": ["프로틴", "단백질 보충제", "protein", "whey protein", "운동 전", "운동 후", "총 단백질"],
    "protein": ["프로틴", "단백질 보충제", "protein", "whey protein", "pre workout", "post workout", "daily protein"],
    "whey protein": ["프로틴", "단백질 보충제", "protein", "whey protein", "pre workout", "post workout", "daily protein"],
    "카페인": ["카페인", "caffeine", "부스터", "프리워크아웃", "운동 전", "수면", "두근거림", "불면", "혈압"],
    "caffeine": ["카페인", "caffeine", "pre workout", "stimulant", "sleep", "palpitation", "insomnia", "blood pressure"],
})


def _normalize_answer_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Promote JSON-string answer/caution fields without changing fallback shape."""
    answer = payload.get("answer")
    if not isinstance(answer, str):
        return payload

    try:
        parsed = json.loads(answer)
    except (TypeError, json.JSONDecodeError):
        return payload

    if not isinstance(parsed, dict):
        return payload

    normalized = dict(payload)
    parsed_answer = parsed.get("answer")
    parsed_caution = parsed.get("caution")

    if isinstance(parsed_answer, str) and parsed_answer.strip():
        normalized["answer"] = parsed_answer
    if isinstance(parsed_caution, str) and parsed_caution.strip():
        normalized["caution"] = parsed_caution

    return normalized


def _ensure_full_answer_caution(payload: Dict[str, Any]) -> Dict[str, Any]:
    if payload.get("mode") != "full" or payload.get("caution"):
        return payload
    normalized = dict(payload)
    normalized["caution"] = _DEFAULT_CAUTION
    return normalized


def _env_flag_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUE_VALUES


class SupplementRAGEngine:
    def __init__(self, db_path: str = "./data/chroma_db"):
        self.ready = False
        self.degraded_reason: str | None = None
        self.db_path = db_path
        try:
            self._init_full_rag(db_path)
            self.ready = True
        except ImportError as exc:
            self.degraded_reason = "missing_dependency"
            print("[Supplement RAG degraded]", sanitize_exception_for_log(exc))
        except Exception as exc:
            self.degraded_reason = "rag_init_failed"
            print("[Supplement RAG degraded]", sanitize_exception_for_log(exc))

    @classmethod
    def degraded(cls, reason: str = "unavailable") -> "SupplementRAGEngine":
        engine = cls.__new__(cls)
        engine.ready = False
        engine.degraded_reason = reason
        engine.db_path = None
        return engine

    def _init_full_rag(self, db_path: str) -> None:
        self.db_dir = self._find_latest_db(db_path)
        if not self.db_dir:
            raise RuntimeError("chroma_db_missing")

        from langchain_core.documents import Document
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_community.tools import DuckDuckGoSearchRun
        from langchain_community.vectorstores import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        from kiwipiepy import Kiwi
        from rank_bm25 import BM25Okapi
        from sentence_transformers import CrossEncoder
        import numpy as np

        self.np = np
        try:
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            self.device = "cpu"

        self.Document = Document
        self.StrOutputParser = StrOutputParser
        self.llm = get_llm("supplement", temperature=0.1)
        self.query_optimizer = get_llm("supplement", temperature=0.0)
        self.web_search = DuckDuckGoSearchRun()

        self.embeddings = HuggingFaceEmbeddings(
            model_name="dragonkue/BGE-m3-ko",
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vector_store = Chroma(persist_directory=str(self.db_dir), embedding_function=self.embeddings)
        self.kiwi = Kiwi(num_workers=-1)
        self.target_tags = ("N", "V", "X", "S", "VA")
        all_db_data = self.vector_store.get()
        self.original_docs = [
            Document(page_content=txt, metadata=meta)
            for txt, meta in zip(all_db_data.get("documents", []), all_db_data.get("metadatas", []))
        ]
        if not self.original_docs:
            raise RuntimeError("chroma_db_empty")

        clean_texts = [(doc.page_content or "").strip() for doc in self.original_docs]
        tokenized_corpus = [self._tokenize_kiwi(text) for text in clean_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=self.device, max_length=512)

        self.local_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a cautious supplement information assistant.
Use only the verified documents below. If the documents are not enough, output WEB_SEARCH_REQUIRED.
Give practical, general timing guidance when the documents support it.
Do not diagnose, treat, or guarantee effects. Keep drug, disease, pregnancy, lactation, kidney, liver, bleeding, and high-dose risks conservative.
Return a plain text answer, not a JSON object or markdown code block.

[Verified documents]
{context}""",
                ),
                ("human", "{question}"),
            ]
        )
        self.web_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Use the web search summary below to provide cautious, general supplement guidance.
Do not diagnose. Include a recommendation to consult a pharmacist or physician when relevant.

[Web search summary]
{context}""",
                ),
                ("human", "{question}"),
            ]
        )

    def _find_latest_db(self, base_path: str) -> Path | None:
        base_dir = Path(base_path)
        fixed_path = base_dir / "latest_index"
        if fixed_path.exists():
            return fixed_path
        if not base_dir.exists():
            return None
        db_folders = sorted(list(base_dir.glob("doc_index_*")), reverse=True)
        return db_folders[0] if db_folders else None

    def _tokenize_kiwi(self, text: str) -> list[str]:
        if not text:
            return []
        return [token.form for token in self.kiwi.tokenize(text) if token.tag.startswith(self.target_tags)]

    def _deterministic_query(self, query: str) -> str:
        normalized = (query or "").strip()
        lowered = normalized.lower()
        keywords: list[str] = []
        for needle, mapped_keywords in _KEYWORD_MAP.items():
            if needle in lowered or needle in normalized:
                keywords.extend(mapped_keywords)
        deduped = list(dict.fromkeys([normalized, *keywords]))
        return " ".join(item for item in deduped if item)

    def _expand_query(self, query: str) -> str:
        if not _env_flag_enabled("SUPPLEMENT_ENABLE_LLM_QUERY_OPTIMIZER"):
            return self._deterministic_query(query)
        prompt = f"Extract supplement names, diseases, symptoms, and interaction keywords only. Question: {query}"
        try:
            response = self.query_optimizer.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            return content.strip() or query
        except Exception:
            return query

    def _doc_key(self, doc: Any) -> Any:
        meta = doc.metadata or {}
        return meta.get("id") or meta.get("_source_file") or hash(doc.page_content)

    def _supplement_timing_docs_for_query(self, query: str) -> list[Any]:
        query_text = (query or "").lower()
        expanded_query = self._deterministic_query(query).lower()
        matched_docs: list[Any] = []

        for doc in getattr(self, "original_docs", []):
            meta = doc.metadata or {}
            if meta.get("source") != "supp_timing":
                continue
            content = (doc.page_content or "").lower()
            if any(term in content for term in (query_text, expanded_query) if term):
                matched_docs.append(doc)
                continue
            for keyword in expanded_query.split():
                if len(keyword) >= 3 and keyword in content:
                    matched_docs.append(doc)
                    break

        return matched_docs

    def _answer_degraded(self, reason: str | None = None) -> Dict[str, Any]:
        return {
            "answer": _DEGRADED_ANSWER,
            "sources": [],
            "mode": "degraded",
            "degraded_reason": reason or self.degraded_reason or "unavailable",
        }

    def answer_question(self, question: str) -> Dict[str, Any]:
        if not (question or "").strip():
            return self._answer_degraded("empty_question")
        if not self.ready:
            return self._answer_degraded()
        try:
            return self._answer_full(question)
        except Exception as exc:
            print("[Supplement RAG runtime fallback]", sanitize_exception_for_log(exc))
            return self._answer_degraded("runtime_error")

    def _answer_full(self, question: str) -> Dict[str, Any]:
        start_time = time.perf_counter()
        timing_ms: dict[str, int] = {}
        counts: dict[str, Any] = {
            "retrievalTopK": 30,
            "rerankTopK": 15,
            "finalTopK": 8,
            "device": self.device,
            "dbPath": str(self.db_dir),
            "queryOptimizerEnabled": _env_flag_enabled("SUPPLEMENT_ENABLE_LLM_QUERY_OPTIMIZER"),
            "queryOptimizerUsed": False,
            "webSearchEnabled": _env_flag_enabled("SUPPLEMENT_ENABLE_WEB_FALLBACK"),
        }

        def mark(name: str, started_at: float) -> None:
            timing_ms[name] = round((time.perf_counter() - started_at) * 1000)

        stage_start = time.perf_counter()
        normalized_question = (question or "").strip()
        mark("inputNormalization", stage_start)

        stage_start = time.perf_counter()
        search_query = self._expand_query(question)
        mark("queryOptimization", stage_start)
        counts["queryOptimizerUsed"] = counts["queryOptimizerEnabled"]
        counts["searchQueryCharCount"] = len(search_query or "")

        k = 30
        stage_start = time.perf_counter()
        query_embedding = self.embeddings.embed_query(search_query)
        mark("embedding", stage_start)
        counts["embeddingDim"] = len(query_embedding) if hasattr(query_embedding, "__len__") else None

        stage_start = time.perf_counter()
        vector_docs = self.vector_store.similarity_search_by_vector(query_embedding, k=k)
        mark("chromaRetrieval", stage_start)
        counts["vectorDocs"] = len(vector_docs)

        stage_start = time.perf_counter()
        q_tokens = self._tokenize_kiwi(search_query)
        bm25_docs = []
        if q_tokens:
            scores = self.bm25.get_scores(q_tokens)
            top_idx = self.np.argsort(scores)[::-1][:k]
            bm25_docs = [self.original_docs[i] for i in top_idx]
        mark("bm25Retrieval", stage_start)
        counts["bm25Docs"] = len(bm25_docs)
        counts["queryTokenCount"] = len(q_tokens)

        stage_start = time.perf_counter()
        rrf_k = 60
        fused_scores: dict[Any, float] = {}
        doc_map: dict[Any, Any] = {}

        def add_to_rrf(docs: list[Any], weight: float) -> None:
            for rank, doc in enumerate(docs):
                key = self._doc_key(doc)
                if key not in fused_scores:
                    fused_scores[key] = 0.0
                    doc_map[key] = doc
                fused_scores[key] += weight * (1.0 / (rank + rrf_k))

        add_to_rrf(bm25_docs, 0.6)
        add_to_rrf(vector_docs, 0.4)
        ranked_keys = sorted(fused_scores, key=lambda key: fused_scores[key], reverse=True)[:15]
        candidate_docs = [doc_map[key] for key in ranked_keys]
        timing_docs = self._supplement_timing_docs_for_query(question)
        if timing_docs:
            seen_candidate_keys = {self._doc_key(doc) for doc in candidate_docs}
            promoted_docs = [doc for doc in timing_docs if self._doc_key(doc) not in seen_candidate_keys]
            candidate_docs = promoted_docs + candidate_docs
        mark("fusion", stage_start)
        counts["candidateDocs"] = len(candidate_docs)
        counts["fusedDocs"] = len(fused_scores)
        counts["timingDocsPromoted"] = len(timing_docs)
        if not candidate_docs:
            return self._answer_degraded("no_documents")

        stage_start = time.perf_counter()
        pairs = [[question, doc.page_content] for doc in candidate_docs]
        rerank_scores = self.reranker.predict(pairs)
        reranked_results = sorted(zip(rerank_scores, candidate_docs), key=lambda item: item[0], reverse=True)
        final_docs = [doc for _score, doc in reranked_results[:8]]
        if timing_docs:
            timing_keys = {self._doc_key(doc) for doc in timing_docs}
            final_docs = timing_docs + [doc for doc in final_docs if self._doc_key(doc) not in timing_keys]
            final_docs = final_docs[:8]
        mark("rerank", stage_start)
        counts["rerankPairs"] = len(pairs)
        counts["finalDocs"] = len(final_docs)

        stage_start = time.perf_counter()
        context_text = "\n\n---\n\n".join([doc.page_content for doc in final_docs])
        mark("promptBuild", stage_start)
        counts["contextCharCount"] = len(context_text)

        stage_start = time.perf_counter()
        local_chain = self.local_prompt | self.llm | self.StrOutputParser()
        answer = local_chain.invoke({"context": context_text, "question": question})
        mark("llmGeneration", stage_start)
        counts["answerCharCount"] = len(answer or "")

        web_search_used = False
        timing_ms["webSearch"] = 0
        timing_ms["webLlmGeneration"] = 0
        if "WEB_SEARCH_REQUIRED" in answer.strip().upper() and counts["webSearchEnabled"] and not final_docs:
            try:
                web_search_used = True
                web_query = f"{search_query} dosage efficacy side effects interaction"
                stage_start = time.perf_counter()
                web_result_text = self.web_search.run(web_query)
                mark("webSearch", stage_start)
                web_chain = self.web_prompt | self.llm | self.StrOutputParser()
                stage_start = time.perf_counter()
                answer = web_chain.invoke({"context": web_result_text, "question": question})
                mark("webLlmGeneration", stage_start)
                counts["webResultCharCount"] = len(web_result_text or "")
                counts["answerCharCount"] = len(answer or "")
                sources: list[Any] = ["DuckDuckGo web search summary"]
            except Exception as exc:
                print("[Supplement web fallback unavailable]", sanitize_exception_for_log(exc))
                return self._answer_degraded("web_search_failed")
        else:
            stage_start = time.perf_counter()
            if "WEB_SEARCH_REQUIRED" in answer.strip().upper():
                answer = (
                    "확보된 로컬 근거 문서만으로 답변합니다. 개인의 질환, 복용 중인 약, 증상에 따라 달라질 수 있으므로 "
                    "보충제 복용 전에는 의사 또는 약사와 상담하세요."
                )
            sources = []
            seen_keys = set()
            for doc in final_docs:
                meta = doc.metadata or {}
                source_key = f"{meta.get('_source_file')}_{meta.get('id')}"
                if source_key in seen_keys:
                    continue
                sources.append(
                    {
                        "file": meta.get("_source_file", "unknown"),
                        "id": str(meta.get("id", "N/A")),
                        "type": str(meta.get("source", "DOC")),
                    }
                )
                seen_keys.add(source_key)
            mark("sourceFormatting", stage_start)

        counts["webSearchUsed"] = web_search_used
        counts["sourcesCount"] = len(sources)
        counts["questionCharCount"] = len(normalized_question)
        if "sourceFormatting" not in timing_ms:
            timing_ms["sourceFormatting"] = 0

        result = _ensure_full_answer_caution(
            _normalize_answer_payload({"answer": answer, "sources": sources, "mode": "full"})
        )
        timing_ms["total"] = round((time.perf_counter() - start_time) * 1000)
        print(
            "[Supplement RAG completed] elapsed_sec={:.2f} web_search_used={} sources_count={}".format(
                timing_ms["total"] / 1000,
                web_search_used,
                len(sources),
            )
        )
        if os.getenv("APP_ENV", "").strip().lower() == "local":
            result["debugTimingMs"] = timing_ms
            result["debugCounts"] = counts
        return result

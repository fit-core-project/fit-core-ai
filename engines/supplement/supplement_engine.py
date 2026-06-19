from __future__ import annotations

import json
import time
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from engines.llm_router import get_llm
from engines.log_redaction import sanitize_exception_for_log
from engines.supplement.query_understanding import EntityType, IntentType, ParsedSupplementQuery, parse_supplement_query
from engines.supplement.response_composer import compose_supplement_response

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
_TRUE_VALUES = {"true", "1", "yes", "y", "on"}
_KEYWORD_MAP = {
    "마그네슘": [
        "마그네슘",
        "magnesium",
        "Magnesium",
        "Mg",
        "마그네슘 복용 시간",
        "마그네슘 복용 타이밍",
        "마그네슘 식후",
        "마그네슘 저녁",
        "마그네슘 신장질환",
        "복용 시간",
        "복용 타이밍",
        "저녁",
        "식후",
        "수면",
        "근육",
        "설사",
        "신장질환",
        "간격",
    ],
    "magnesium": [
        "마그네슘",
        "magnesium",
        "Magnesium",
        "Mg",
        "마그네슘 복용 시간",
        "마그네슘 복용 타이밍",
        "마그네슘 식후",
        "마그네슘 저녁",
        "마그네슘 신장질환",
        "복용 시간",
        "복용 타이밍",
        "저녁",
        "식후",
        "수면",
        "근육",
        "설사",
        "신장질환",
        "간격",
    ],
    "크레아틴": ["크레아틴", "creatine", "복용", "섭취", "타이밍", "운동 전", "운동 후", "로딩", "용량"],
    "creatine": ["크레아틴", "creatine", "복용", "섭취", "타이밍", "운동 전", "운동 후", "로딩", "용량"],
    "카페인": ["카페인", "caffeine", "부스터", "프리워크아웃", "저녁", "수면", "심박", "불면"],
    "caffeine": ["카페인", "caffeine", "부스터", "프리워크아웃", "저녁", "수면", "심박", "불면"],
    "단백질": ["단백질", "프로틴", "protein", "whey", "섭취량", "운동 후"],
    "프로틴": ["단백질", "프로틴", "protein", "whey", "섭취량", "운동 후"],
    "protein": ["단백질", "프로틴", "protein", "whey", "섭취량", "운동 후"],
    "웨이": ["단백질", "프로틴", "웨이", "whey", "protein powder", "신장질환", "총 단백질", "유당불내증"],
    "protein powder": ["단백질", "프로틴", "whey", "protein powder", "kidney disease", "total protein"],
    "오메가3": ["오메가3", "omega-3", "EPA", "DHA"],
    "오메가-3": ["오메가3", "오메가-3", "omega3", "omega-3", "EPA", "DHA"],
    "omega3": ["오메가3", "오메가-3", "omega3", "omega-3", "EPA", "DHA"],
    "omega": ["오메가3", "omega-3", "EPA", "DHA"],
    "비타민d": ["비타민D", "vitamin D"],
    "비타민 d": ["비타민D", "비타민 D", "vitamin D"],
    "vitamin d": ["비타민D", "vitamin D"],
    "철분": ["철분", "iron", "Fe", "커피", "카페인", "칼슘", "흡수 방해", "공복", "비타민C", "복용 간격", "변비", "갑상선약"],
    "iron": ["철분", "iron", "Fe", "커피", "카페인", "칼슘", "흡수 방해", "공복", "비타민C", "복용 간격", "변비", "갑상선약"],
    "종합비타민": ["종합비타민", "멀티비타민", "multivitamin", "비타민A", "비타민D", "철분", "아연", "성분 중복", "고함량"],
    "멀티비타민": ["종합비타민", "멀티비타민", "multivitamin", "비타민A", "비타민D", "철분", "아연", "성분 중복", "고함량"],
    "multivitamin": ["종합비타민", "멀티비타민", "multivitamin", "vitamin A", "vitamin D", "iron", "zinc", "nutrient overlap"],
    "실리마린": ["실리마린", "밀크씨슬", "silymarin", "milk thistle", "간질환", "간수치", "처방약", "약물 대사"],
    "밀크씨슬": ["실리마린", "밀크씨슬", "silymarin", "milk thistle", "간질환", "간수치", "처방약", "약물 대사"],
    "silymarin": ["실리마린", "밀크씨슬", "silymarin", "milk thistle", "liver disease", "medication"],
    "유산균": ["유산균", "프로바이오틱스", "probiotic", "probiotics", "항생제", "복용 간격", "면역저하"],
    "프로바이오틱스": ["유산균", "프로바이오틱스", "probiotic", "probiotics", "항생제", "복용 간격", "면역저하"],
    "probiotic": ["유산균", "프로바이오틱스", "probiotic", "probiotics", "antibiotics", "spacing", "immunocompromised"],
    "아연": ["아연", "zinc", "철분", "칼슘", "항생제", "흡수 간섭", "구리 결핍", "위장 불편"],
    "zinc": ["아연", "zinc", "iron", "calcium", "antibiotics", "absorption interference", "high dose"],
}
_TIMING_INTENT_NEEDLES = ("언제", "먹는 시간", "복용 시간", "복용 타이밍", "타이밍", "식전", "식후", "공복", "자기 전", "저녁", "간격")
_TIMING_INTENT_KEYWORDS = ["복용", "섭취", "복용 시간", "복용 타이밍", "식전", "식후", "공복", "자기 전", "저녁", "간격"]
_ENTITY_EQUIVALENTS = {
    "anticoagulant": {"anticoagulant", "warfarin"},
    "warfarin": {"warfarin", "anticoagulant"},
    "caffeine": {"caffeine", "coffee"},
    "coffee": {"coffee", "caffeine"},
    "alcohol": {"alcohol", "alcohol use"},
    "alcohol use": {"alcohol use", "alcohol"},
    "vitamin d": {"vitamin d", "vitamind"},
    "omega3": {"omega3", "omega-3", "omega", "fish oil"},
    "thyroid medication": {"thyroid medication", "levothyroxine"},
    "levothyroxine": {"levothyroxine", "thyroid medication"},
    "probiotics": {"probiotics", "probiotic"},
    "probiotic": {"probiotics", "probiotic"},
    "silymarin": {"silymarin", "milk thistle"},
    "milk thistle": {"silymarin", "milk thistle"},
    "multivitamin": {"multivitamin", "multi vitamin"},
    "protein": {"protein", "whey protein", "protein powder"},
    "whey protein": {"protein", "whey protein", "protein powder"},
    "nutrient_overlap": {"nutrient_overlap", "iron", "calcium", "zinc", "vitamin d", "vitamin a"},
}
_EMBEDDING_MODEL_ID = "dragonkue/BGE-m3-ko"
_RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"


def _cached_hf_model_path(model_id: str, required_file: str) -> str:
    """Resolve a local HuggingFace snapshot path to avoid startup network probes."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return model_id

    cached_file = try_to_load_from_cache(model_id, required_file)
    if not cached_file or not isinstance(cached_file, str):
        raise RuntimeError(f"huggingface_model_cache_missing:{model_id}:{required_file}")
    return str(Path(cached_file).parent)


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
    if not isinstance(parsed_answer, str) or not parsed_answer.strip():
        safety_check = parsed.get("safety_check")
        if isinstance(safety_check, dict):
            answer_parts = [
                safety_check.get("summary"),
                safety_check.get("recommendation"),
            ]
            parsed_answer = " ".join(
                part.strip()
                for part in answer_parts
                if isinstance(part, str) and part.strip()
            )
            parsed_caution = parsed_caution or safety_check.get("caution") or safety_check.get("risk")
    if not isinstance(parsed_answer, str) or not parsed_answer.strip():
        flattened_parts = []
        for key, value in parsed.items():
            if key in {"caution", "warning"}:
                continue
            if isinstance(value, str) and value.strip():
                flattened_parts.append(f"{key}: {value}".strip())
            elif isinstance(value, (dict, list)):
                try:
                    flattened_parts.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
                except TypeError:
                    flattened_parts.append(f"{key}: {value}")
            elif value is not None:
                flattened_parts.append(f"{key}: {value}")
        parsed_answer = " ".join(part for part in flattened_parts if part).strip()
        parsed_caution = parsed_caution or parsed.get("warning")
        if parsed_answer and not parsed_caution:
            parsed_caution = "복용 중인 약, 질환, 음주 등 위험 요인이 있으면 의사 또는 약사와 상담하세요."

    if isinstance(parsed_answer, str) and parsed_answer.strip():
        normalized["answer"] = parsed_answer
    if isinstance(parsed_caution, str) and parsed_caution.strip():
        normalized["caution"] = parsed_caution

    return normalized


def _env_flag_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUE_VALUES


def _is_unusable_generated_answer(answer: Any) -> bool:
    if not isinstance(answer, str):
        return True
    stripped = answer.strip()
    if not stripped or stripped in {"{}", "[]", "null"}:
        return True
    try:
        parsed = json.loads(stripped)
    except (TypeError, json.JSONDecodeError):
        return False
    if isinstance(parsed, dict):
        parsed_answer = parsed.get("answer")
        return not (isinstance(parsed_answer, str) and parsed_answer.strip())
    return True


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

        embedding_model_path = _cached_hf_model_path(_EMBEDDING_MODEL_ID, "modules.json")
        reranker_model_path = _cached_hf_model_path(_RERANKER_MODEL_ID, "config.json")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            model_kwargs={"device": self.device, "local_files_only": True},
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
        self.reranker = CrossEncoder(reranker_model_path, device=self.device, max_length=512, local_files_only=True)

        self.local_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a cautious supplement information assistant.
Use only the verified documents below. If the documents are not enough, output WEB_SEARCH_REQUIRED.

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
        if any(needle in normalized for needle in _TIMING_INTENT_NEEDLES):
            keywords.extend(_TIMING_INTENT_KEYWORDS)
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

    def _expanded_canonicals(self, canonical: str) -> set[str]:
        normalized = (canonical or "").strip().lower()
        if not normalized:
            return set()
        return {normalized, *_ENTITY_EQUIVALENTS.get(normalized, set())}

    def _parsed_entities_by_type(self, parsed_query: ParsedSupplementQuery, entity_type: EntityType) -> set[str]:
        values: set[str] = set()
        for entity in parsed_query.entities:
            if entity.type != entity_type:
                continue
            values.update(self._expanded_canonicals(entity.canonical))
        return values

    def _all_parsed_entity_values(self, parsed_query: ParsedSupplementQuery) -> set[str]:
        values: set[str] = set()
        for entity in parsed_query.entities:
            values.update(self._expanded_canonicals(entity.canonical))
        return values

    def _metadata_values(self, value: Any) -> set[str]:
        if value is None:
            return set()
        if isinstance(value, str):
            parts = [part.strip().lower() for part in value.split(",")]
            return {part for part in parts if part}
        if isinstance(value, list):
            values: set[str] = set()
            for item in value:
                values.update(self._metadata_values(item))
            return values
        return {str(value).strip().lower()}

    def _doc_source_type(self, doc: Any) -> str:
        meta = doc.metadata or {}
        return str(meta.get("source") or meta.get("type") or "").strip()

    def _doc_text_contains_any(self, doc: Any, values: set[str]) -> bool:
        text = (doc.page_content or "").lower()
        return any(value and value in text for value in values)

    def _priority_kb_docs(self, parsed_query: ParsedSupplementQuery) -> list[Any]:
        intents = set(parsed_query.intents)
        supplement_entities = self._parsed_entities_by_type(parsed_query, EntityType.SUPPLEMENT_INGREDIENT)
        drug_entities = self._parsed_entities_by_type(parsed_query, EntityType.DRUG_OR_DRUG_CLASS)
        food_entities = self._parsed_entities_by_type(parsed_query, EntityType.FOOD_OR_COMPOUND)
        condition_entities = self._parsed_entities_by_type(parsed_query, EntityType.CONDITION)
        risk_entities = self._parsed_entities_by_type(parsed_query, EntityType.RISK_CONTEXT)
        all_entities = self._all_parsed_entity_values(parsed_query)

        scored_docs: list[tuple[float, Any]] = []
        for doc in getattr(self, "original_docs", []):
            meta = doc.metadata or {}
            source = self._doc_source_type(doc)
            score = 0.0

            if source == "interaction_rule":
                entity_a = self._expanded_canonicals(str(meta.get("entity_a_canonical") or ""))
                entity_b = self._expanded_canonicals(str(meta.get("entity_b_canonical") or ""))
                a_match = bool(entity_a & all_entities)
                b_match = bool(entity_b & all_entities)
                supplement_match = bool((entity_a | entity_b) & supplement_entities)
                risk_match = bool((entity_a | entity_b) & (drug_entities | food_entities | risk_entities))
                if a_match and b_match:
                    score += 120.0
                elif supplement_match and risk_match:
                    score += 90.0
                elif supplement_match:
                    score += 35.0
                if score > 0 and intents & {IntentType.DRUG_INTERACTION, IntentType.FOOD_COMPOUND_INTERACTION, IntentType.SUPPLEMENT_INTERACTION}:
                    score += 30.0
                if score > 0 and str(meta.get("caution_level") or "").lower() in {"high", "critical"}:
                    score += 5.0
                if meta.get("id") == "INT_MULTIVITAMIN_MINERAL_OVERLAP" and "multivitamin" in supplement_entities:
                    score += 95.0
                if meta.get("id") == "INT_ZINC_IRON_CALCIUM" and "zinc" in supplement_entities and {"iron", "calcium"} & supplement_entities:
                    score += 110.0

            elif source == "safety_rule":
                affected = set()
                for value in self._metadata_values(meta.get("affected_entities")):
                    affected.update(self._expanded_canonicals(value))
                affected_match = bool(affected & (supplement_entities | drug_entities | food_entities | all_entities))
                trigger_match = self._doc_text_contains_any(doc, condition_entities | risk_entities | drug_entities | food_entities)
                if affected_match and trigger_match:
                    score += 125.0
                elif affected_match:
                    score += 70.0
                if score > 0 and intents & {
                    IntentType.CONDITION_SAFETY,
                    IntentType.PREGNANCY_OR_HIGH_DOSE_SAFETY,
                    IntentType.MEDICATION_SAFETY,
                    IntentType.DRUG_INTERACTION,
                }:
                    score += 35.0
                if score > 0 and str(meta.get("caution_level") or "").lower() in {"high", "critical"}:
                    score += 8.0

            elif source == "ingredient_profile":
                canonical = self._expanded_canonicals(str(meta.get("canonical") or ""))
                if canonical & supplement_entities:
                    score += 60.0
                    if IntentType.TIMING in intents:
                        score += 35.0

            if score > 0:
                scored_docs.append((score, doc))

        scored_docs.sort(key=lambda item: item[0], reverse=True)
        return [doc for _score, doc in scored_docs[:8]]

    def _answer_from_timing_doc(self, doc: Any) -> Dict[str, str] | None:
        lines = (doc.page_content or "").splitlines()
        timing = ""
        spacing = ""
        cautions: list[str] = []
        in_cautions = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("Timing:"):
                timing = stripped.removeprefix("Timing:").strip()
                in_cautions = False
            elif stripped.startswith("Spacing:"):
                spacing = stripped.removeprefix("Spacing:").strip()
                in_cautions = False
            elif stripped == "Cautions:":
                in_cautions = True
            elif in_cautions and stripped.startswith("- "):
                cautions.append(stripped.removeprefix("- ").strip())

        answer = " ".join(part for part in [timing, spacing] if part).strip()
        if not answer:
            return None
        caution = " ".join(cautions).strip()
        return {"answer": answer, "caution": caution} if caution else {"answer": answer}

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
        parsed_query = parse_supplement_query(normalized_question)
        mark("inputNormalization", stage_start)
        counts["parsedEntityCount"] = len(parsed_query.entities)
        counts["parsedIntents"] = [intent.value for intent in parsed_query.intents]

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

        priority_kb_docs = self._priority_kb_docs(parsed_query)
        counts["priorityKbDocs"] = len(priority_kb_docs)

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
        add_to_rrf(priority_kb_docs, 1.2)
        ranked_keys = sorted(fused_scores, key=lambda key: fused_scores[key], reverse=True)[:15]
        candidate_docs = [doc_map[key] for key in ranked_keys]
        mark("fusion", stage_start)
        counts["candidateDocs"] = len(candidate_docs)
        counts["fusedDocs"] = len(fused_scores)
        if not candidate_docs:
            return self._answer_degraded("no_documents")

        stage_start = time.perf_counter()
        pairs = [[question, doc.page_content] for doc in candidate_docs]
        rerank_scores = self.reranker.predict(pairs)
        reranked_results = sorted(zip(rerank_scores, candidate_docs), key=lambda item: item[0], reverse=True)
        reranked_docs = [doc for _score, doc in reranked_results]
        final_docs = []
        seen_final_keys = set()
        for doc in [*priority_kb_docs, *reranked_docs]:
            key = self._doc_key(doc)
            if key in seen_final_keys:
                continue
            final_docs.append(doc)
            seen_final_keys.add(key)
            if len(final_docs) >= 8:
                break
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
                        "type": str(meta.get("source") or meta.get("type") or "DOC"),
                    }
                )
                seen_keys.add(source_key)
            mark("sourceFormatting", stage_start)

        counts["webSearchUsed"] = web_search_used
        counts["sourcesCount"] = len(sources)
        counts["questionCharCount"] = len(normalized_question)
        if "sourceFormatting" not in timing_ms:
            timing_ms["sourceFormatting"] = 0

        composed = compose_supplement_response(
            question=normalized_question,
            parsed_query=parsed_query,
            generated_answer=answer,
            generated_caution=None,
            selected_docs=final_docs,
        )
        payload = {"answer": composed.answer, "sources": sources, "mode": "full"}
        if composed.caution:
            payload["caution"] = composed.caution
        if composed.used_kb_fallback:
            counts["answerRecoveredFromKbDocs"] = True
        result = _normalize_answer_payload(payload)
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

"""식품 DB 벡터 검색 엔진 (ChromaDB + BGE-m3-ko).

배치 위치: engines/food/food_engine.py

- 오프라인 스크립트(scripts/build_food_db.py)가 만든 인덱스를 로드
- search(food_name, amount, unit) → DB 기반 매크로 dict 또는 None
- 인덱스/모델/의존성이 없으면 available=False 로 폴백 (예외를 밖으로 던지지 않음)

무거운 의존성(langchain/chroma/torch)은 전부 함수 내부에서 지역 import 한다.
=> diet_parser 가 이 모듈을 import 해도 로딩이 즉시 발생하지 않고, 환경이 깨져도
   import 자체가 실패하지 않는다.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

from engines.food.food_canonical_index import load_food_canonical_index
from engines.food.food_query_normalization import analyze_food_query_for_search
from engines.food.food_query_normalization import build_food_search_queries
from engines.food.food_units import (
    FOOD_UNIT_GRAMS,
    GRAM_UNITS,
    UNIT_DEFAULT_GRAMS,
    preferred_basis_unit,
    resolve_portion_grams,
    unit_to_grams,
)
from engines.food.food_reranker import (
    FoodSearchCandidate,
    build_food_candidate_dedupe_key,
    get_food_candidate_name,
    rerank_food_candidates,
)

logger = logging.getLogger(__name__)

# engines/food/food_engine.py → parents[2] = 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── build 스크립트와 공유하는 설정 (반드시 일치해야 함) ──────────────────────
EMBEDDING_MODEL_ID = os.getenv("FOOD_EMBEDDING_MODEL", "dragonkue/BGE-m3-ko")
FOOD_DB_DIR = PROJECT_ROOT / "data" / "food_chroma_db"
FOOD_COLLECTION_NAME = "food_db"
FOOD_CSV_PATH = PROJECT_ROOT / "data" / "food_db" / "food_db_clean.csv"

# 코사인 유사도 임계값. 이 값 미만이면 매칭 실패로 간주(오탐 방지). 캘리브레이션 필요.
SIMILARITY_THRESHOLD = float(os.getenv("FOOD_SIM_THRESHOLD", "0.55"))
# 검색 후보 수
FOOD_SEARCH_FINAL_K = int(os.getenv("FOOD_SEARCH_FINAL_K", os.getenv("FOOD_TOP_K", "3")))
TOP_K = FOOD_SEARCH_FINAL_K
FOOD_SEARCH_FETCH_K = max(
    int(os.getenv("FOOD_SEARCH_FETCH_K", "10")),
    FOOD_SEARCH_FINAL_K,
)

# ── 단위 → 그램 변환: engines/food/food_units.py 가 단일 출처 ──────────────
_GRAM_UNITS = GRAM_UNITS
_UNIT_DEFAULT_GRAMS = UNIT_DEFAULT_GRAMS
_FOOD_UNIT_GRAMS = FOOD_UNIT_GRAMS


def resolve_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def build_embeddings():
    """build 스크립트/엔진이 공유하는 임베딩 로더. 정규화 ON.

    - FOOD_EMBEDDING_PATH 로 로컬 경로를 직접 지정 가능
    - FOOD_EMBEDDING_LOCAL_ONLY=0 으로 하면 네트워크 다운로드 허용 (기본 오프라인)
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except Exception:  # 구버전 폴백
        from langchain_community.embeddings import HuggingFaceEmbeddings

    model_ref = os.getenv("FOOD_EMBEDDING_PATH", EMBEDDING_MODEL_ID)
    local_only = os.getenv("FOOD_EMBEDDING_LOCAL_ONLY", "1").strip().lower() in {
        "1", "true", "yes", "on",
    }
    return HuggingFaceEmbeddings(
        model_name=model_ref,
        model_kwargs={"device": resolve_device(), "local_files_only": local_only},
        encode_kwargs={"normalize_embeddings": True},
    )


class FoodSearchEngine:
    """식품 DB 검색 싱글톤. 인스턴스는 항상 생성되며, 사용 불가 시 available=False."""

    def __init__(self):
        self.available = False
        self._vector_store = None
        self._canonical_index = None
        self._lock = threading.Lock()  # 임베딩/검색 직렬화 (모델 스레드안전 보장 X)
        self._load()

    def _load(self):
        # 인덱스 디렉토리 확인 (없거나 비어있으면 폴백)
        if not FOOD_DB_DIR.exists() or not any(FOOD_DB_DIR.iterdir()):
            logger.warning("[food_engine] 인덱스 없음 → 폴백 모드: %s", FOOD_DB_DIR)
            return
        try:
            try:
                from langchain_chroma import Chroma
            except Exception:
                from langchain_community.vectorstores import Chroma

            embeddings = build_embeddings()
            self._vector_store = Chroma(
                collection_name=FOOD_COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=str(FOOD_DB_DIR),
                collection_metadata={"hnsw:space": "cosine"},
            )

            # 헬스체크: 컬렉션에 문서가 실제로 있는지
            try:
                cnt = self._vector_store._collection.count()
            except Exception:
                cnt = None
            if cnt == 0:
                logger.warning("[food_engine] 인덱스가 비어있음 → 폴백 모드")
                self._vector_store = None
                return

            self.available = True
            self._load_canonical_index()
            logger.info(
                "[food_engine] 로드 완료 (docs=%s, model=%s, device=%s, thr=%.2f)",
                cnt, EMBEDDING_MODEL_ID, resolve_device(), SIMILARITY_THRESHOLD,
            )
        except Exception as exc:
            logger.warning("[food_engine] 로드 실패 → 폴백 모드: %s", exc)
            self._vector_store = None
            self.available = False

    def _load_canonical_index(self):
        try:
            self._canonical_index = load_food_canonical_index(FOOD_CSV_PATH)
        except Exception as exc:
            logger.warning("[food_engine] canonical index 로드 실패: %s", exc)
            self._canonical_index = None

    # ── 단위 → 그램 ─────────────────────────────────────────────
    @staticmethod
    def _unit_to_grams(food_name, unit):
        return unit_to_grams(food_name, unit)

    def _resolve_grams(self, food_name, amount, unit):
        """섭취량을 g(ml≈g) 로 환산. amount 가 없으면 None (basis 기준값 사용).

        개/조각/캔/병/그릇/인분/봉지/팩/컵/장/모 등 개수 단위는 식품별 기본값,
        없으면 단위별 일반 기본값으로 환산한다 (더 이상 100g 고정 폴백 없음).
        단위 없는 숫자("계란 2")는 셀 수 있는 식품이면 개수로 본다.
        """
        return resolve_portion_grams(food_name, amount, unit)

    @staticmethod
    def _prefer_basis_unit(candidates, unit):
        """같은 이름의 g/ml 중복 행 중 단위 성격에 맞는 행을 선택."""
        if not candidates:
            return candidates
        top_name = get_food_candidate_name(candidates[0].document)
        want = preferred_basis_unit(unit)
        for index, candidate in enumerate(candidates):
            if get_food_candidate_name(candidate.document) != top_name:
                break
            meta = candidate.document.metadata or {}
            if (meta.get("basis_unit") or "g") == want:
                if index:
                    return [candidate] + candidates[:index] + candidates[index + 1:]
                return candidates
        return candidates

    # ── 메인 검색 ───────────────────────────────────────────────
    @staticmethod
    def _dedupe_result_key(doc):
        return build_food_candidate_dedupe_key(doc)

    @staticmethod
    def _collect_candidates(result_groups, search_queries):
        seen = set()
        candidates = []
        for source_query_index, results in enumerate(result_groups):
            source_query = search_queries[source_query_index]
            for original_rank, (doc, distance) in enumerate(results):
                dedupe_key = build_food_candidate_dedupe_key(doc)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                candidates.append(
                    FoodSearchCandidate(
                        document=doc,
                        vector_score=float(distance),
                        source_query=source_query,
                        source_query_index=source_query_index,
                        original_rank=original_rank,
                        dedupe_key=dedupe_key,
                    )
                )
        return candidates

    @staticmethod
    def _dedupe_candidates(candidates):
        seen = set()
        deduped = []
        for candidate in candidates:
            if candidate.dedupe_key in seen:
                continue
            seen.add(candidate.dedupe_key)
            deduped.append(candidate)
        return deduped

    @staticmethod
    def _canonical_lookup_queries(analysis):
        if not analysis.cleaned_query:
            return []
        if analysis.protected:
            return [analysis.cleaned_query]
        queries = []
        for query in (analysis.normalized_query, analysis.cleaned_query):
            if query and query not in queries:
                queries.append(query)
        return queries

    def _lookup_canonical_candidates(self, analysis):
        index = getattr(self, "_canonical_index", None)
        if index is None:
            return []
        candidates = []
        for lookup_index, query in enumerate(self._canonical_lookup_queries(analysis)):
            for original_rank, record in enumerate(index.lookup(query, limit=TOP_K)):
                doc = record.to_document()
                dedupe_key = build_food_candidate_dedupe_key(doc)
                candidates.append(
                    FoodSearchCandidate(
                        document=doc,
                        vector_score=0.0,
                        source_query=query,
                        source_query_index=-1 + lookup_index,
                        original_rank=original_rank,
                        dedupe_key=dedupe_key,
                    )
                )
        return self._dedupe_candidates(candidates)

    @classmethod
    def _dedupe_results(cls, result_groups):
        candidates = cls._collect_candidates(
            result_groups,
            [str(index) for index, _ in enumerate(result_groups)],
        )
        return [
            (candidate.document, candidate.vector_score)
            for candidate in candidates[:TOP_K]
        ]

    def search(self, food_name: str, amount=None, unit=None) -> Optional[dict]:
        if not self.available or self._vector_store is None:
            return None
        analysis = analyze_food_query_for_search(food_name)
        search_queries = build_food_search_queries(food_name)
        if not search_queries:
            return None

        try:
            with self._lock:
                result_groups = [
                    self._vector_store.similarity_search_with_score(query, k=FOOD_SEARCH_FETCH_K)
                    for query in search_queries
                ]
        except Exception as exc:
            logger.warning("[food_engine] 검색 실패(%s): %s", search_queries, exc)
            return None
        canonical_candidates = self._lookup_canonical_candidates(analysis)
        vector_candidates = self._collect_candidates(result_groups, search_queries)
        candidates = self._dedupe_candidates(canonical_candidates + vector_candidates)
        candidates = rerank_food_candidates(analysis, candidates, final_k=FOOD_SEARCH_FINAL_K)
        if not candidates:
            return None
        candidates = self._prefer_basis_unit(candidates, unit)

        # cosine space: score=거리(작을수록 유사), 코사인 유사도 = 1 - 거리
        doc = candidates[0].document
        distance = candidates[0].vector_score
        similarity = 1.0 - float(distance)
        if similarity < SIMILARITY_THRESHOLD:
            logger.debug("[food_engine] 임계값 미달: '%s' sim=%.3f", search_queries, similarity)
            return None

        meta = doc.metadata or {}
        protein = meta.get("protein_g")
        carbs = meta.get("carb_g")   # CSV 컬럼명은 carb_g (출력은 carbs_g)
        fat = meta.get("fat_g")
        # 핵심 매크로가 전부 비어있으면 무의미한 매칭 → 폴백
        if protein is None and carbs is None and fat is None:
            return None

        basis_value = meta.get("basis_value") or 100.0
        if not basis_value:
            basis_value = 100.0
        basis_unit = meta.get("basis_unit") or "g"

        grams = self._resolve_grams(food_name, amount, unit)
        if grams is None:
            # 수량 없음 → basis(예:100g) 기준값 그대로, 수량은 basis 로 명시
            factor = 1.0
            resolved_amount = float(basis_value)
            resolved_unit = basis_unit
            resolved_grams = float(basis_value)
        else:
            factor = grams / float(basis_value)
            resolved_amount = float(amount)
            resolved_unit = unit
            resolved_grams = float(grams)

        def _scale(v):
            return round(v * factor, 1) if v is not None else None

        return {
            "protein_g": _scale(protein),
            "carbs_g": _scale(carbs),
            "fat_g": _scale(fat),
            "kcal": _scale(meta.get("kcal")),
            # 다영양소 (모델/프론트 확장 시 그대로 사용)
            "sugar_g": _scale(meta.get("sugar_g")),
            "fiber_g": _scale(meta.get("fiber_g")),
            "sodium_mg": _scale(meta.get("sodium_mg")),
            "source": "db",
            "matched_name": meta.get("name") or meta.get("rep_name"),
            "similarity": round(similarity, 3),
            "basis_value": float(basis_value),
            "basis_unit": basis_unit,
            "resolved_amount": resolved_amount,
            "resolved_unit": resolved_unit,
            "resolved_grams": round(resolved_grams, 1),
        }


# ── 프로세스 전역 싱글톤 ─────────────────────────────────────────
_engine_instance: Optional[FoodSearchEngine] = None
_engine_lock = threading.Lock()


def get_food_engine() -> FoodSearchEngine:
    """항상 인스턴스를 반환. 사용 불가 시 .available == False (search 는 None 반환)."""
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = FoodSearchEngine()
    return _engine_instance

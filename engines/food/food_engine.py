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

from engines.food.food_query_normalization import analyze_food_query_for_search
from engines.food.food_query_normalization import build_food_search_queries
from engines.food.food_reranker import (
    FoodSearchCandidate,
    build_food_candidate_dedupe_key,
    rerank_food_candidates,
)

logger = logging.getLogger(__name__)

# engines/food/food_engine.py → parents[2] = 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── build 스크립트와 공유하는 설정 (반드시 일치해야 함) ──────────────────────
EMBEDDING_MODEL_ID = os.getenv("FOOD_EMBEDDING_MODEL", "dragonkue/BGE-m3-ko")
FOOD_DB_DIR = PROJECT_ROOT / "data" / "food_chroma_db"
FOOD_COLLECTION_NAME = "food_db"

# 코사인 유사도 임계값. 이 값 미만이면 매칭 실패로 간주(오탐 방지). 캘리브레이션 필요.
SIMILARITY_THRESHOLD = float(os.getenv("FOOD_SIM_THRESHOLD", "0.55"))
# 검색 후보 수
TOP_K = int(os.getenv("FOOD_TOP_K", "3"))

# ── 단위 → 그램 변환 테이블 (diet_parser 와 의미상 동일. 추후 단일 출처로 통합 권장) ──
_GRAM_UNITS = {"g", "gram", "grams", "그램", "ml", "밀리리터", "cc"}
_UNIT_DEFAULT_GRAMS: dict[str, float] = {
    "공기": 210.0, "그릇": 400.0, "인분": 200.0,
    "스쿱": 30.0, "모": 300.0, "컵": 200.0,
    "장": 20.0, "쪽": 10.0, "봉지": 100.0, "팩": 100.0,
}
_FOOD_UNIT_GRAMS: dict[str, dict[str, float]] = {
    "계란": {"개": 50.0}, "달걀": {"개": 50.0}, "삶은 달걀": {"개": 50.0},
    "바나나": {"개": 120.0}, "사과": {"개": 200.0}, "고구마": {"개": 150.0},
    "감자": {"개": 130.0}, "오렌지": {"개": 180.0},
}


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
            logger.info(
                "[food_engine] 로드 완료 (docs=%s, model=%s, device=%s, thr=%.2f)",
                cnt, EMBEDDING_MODEL_ID, resolve_device(), SIMILARITY_THRESHOLD,
            )
        except Exception as exc:
            logger.warning("[food_engine] 로드 실패 → 폴백 모드: %s", exc)
            self._vector_store = None
            self.available = False

    # ── 단위 → 그램 ─────────────────────────────────────────────
    @staticmethod
    def _unit_to_grams(food_name, unit):
        if unit is None:
            return None
        u = unit.lower().strip()
        overrides = _FOOD_UNIT_GRAMS.get((food_name or "").strip(), {})
        if u in overrides:
            return overrides[u]
        return _UNIT_DEFAULT_GRAMS.get(u)

    def _resolve_grams(self, food_name, amount, unit):
        """섭취량을 basis(보통 100g) 기준 수치로 환산. 불가 시 None."""
        if amount is None:
            return None
        u = (unit or "").lower().strip()
        # g/ml 계열이거나 단위 미지정 → amount 를 그대로 사용
        if u in _GRAM_UNITS or u == "":
            return float(amount)
        per_unit = self._unit_to_grams(food_name, unit)
        if per_unit is None:
            return None
        return float(amount) * per_unit

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
                    self._vector_store.similarity_search_with_score(query, k=TOP_K)
                    for query in search_queries
                ]
        except Exception as exc:
            logger.warning("[food_engine] 검색 실패(%s): %s", search_queries, exc)
            return None
        candidates = self._collect_candidates(result_groups, search_queries)
        candidates = rerank_food_candidates(analysis, candidates, final_k=TOP_K)
        if not candidates:
            return None

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
            # 수량 환산 불가 → basis(예:100g) 기준값 그대로, 수량은 basis 로 명시
            factor = 1.0
            resolved_amount = float(basis_value)
            resolved_unit = basis_unit
        else:
            factor = grams / float(basis_value)
            resolved_amount = float(amount)
            resolved_unit = unit

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

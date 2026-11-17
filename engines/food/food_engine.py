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
import functools
import csv
import re
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
    category_portion_grams,
    preferred_basis_unit,
    resolve_portion_grams,
    serving_grams,
    portion_confidence,
    unit_to_grams,
)
from engines.food.food_reranker import (
    FoodSearchCandidate,
    build_food_candidate_dedupe_key,
    get_food_candidate_name,
    rerank_food_candidates,
    has_lexical_relation,
)

logger = logging.getLogger(__name__)

# engines/food/food_engine.py → parents[2] = 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── build 스크립트와 공유하는 설정 (반드시 일치해야 함) ──────────────────────
EMBEDDING_MODEL_ID = os.getenv("FOOD_EMBEDDING_MODEL", "dragonkue/BGE-m3-ko")
FOOD_DB_DIR = PROJECT_ROOT / "data" / "food_chroma_db"
FOOD_COLLECTION_NAME = "food_db"
FOOD_CSV_PATH = PROJECT_ROOT / "data" / "food_db" / "food_db_clean.csv"
# Below this similarity a top hit must also share wording with the query.
LEXICAL_GUARD_SIMILARITY = 0.9
LITERAL_HIT_SIMILARITY = 0.45
_DISH_SERVING_G = 300.0
_REORDERED_SOURCE_INDEX = -10
_LEXICAL_SOURCE_INDEX = 99  # lexical-index rows rank after every vector / canonical hit on ties


def _compact_name(candidate) -> str:
    return "".join(str((candidate.document.metadata or {}).get("name") or "").split())
_DRIED_WORDS_RE = re.compile(r"말린|말랭이|건조|마른|육포|dried|dry|건면")


def _protein_dense(meta: dict) -> bool:
    """Grilled meat / fish / cutlet rows (장어구이, 대창, 오겹살): a 1인분 is the ~200 g meat portion."""
    try:
        kcal = float(meta.get("kcal") or 0)
        protein = float(meta.get("protein_g") or 0)
        fat = float(meta.get("fat_g") or 0)
    except (TypeError, ValueError):
        return False
    return kcal > 0 and (protein * 4 + fat * 9) / kcal > 0.6

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
    # AI_EMBEDDING_DEVICE=cpu keeps embeddings off a small GPU that the local LLM (Ollama) needs;
    # on 8GB cards sharing it made gemma4 swap out and requests hang for minutes.
    forced = (os.getenv("AI_EMBEDDING_DEVICE") or "").strip().lower()
    if forced in {"cpu", "cuda", "mps"}:
        return forced
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



# Same dish, very different densities: "식혜" rows are 40-47 kcal/100 except one at 87, "샐러드"
# rows ~125-145 except a bare 15. When the matched dish row is far from the median of the rows
# that name the same food (same rep_name, query in the name), use the median row instead.
_MEDIAN_MIN_ROWS = 3
_MEDIAN_MAX_DEVIATION = 0.4


@functools.lru_cache(maxsize=1)
def _dish_rows_by_rep() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    try:
        with FOOD_CSV_PATH.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("data_type") != "음식" or not row.get("kcal"):
                    continue
                out.setdefault(row.get("rep_name") or "", []).append(row)
    except OSError:
        return {}
    return out


_CATEGORY_MIN_ROWS = 5
_DRINK_SUFFIX_RE = re.compile(r"(라떼|커피|차|티|에이드|주스|스무디|쉐이크|음료|우유|하이볼|맥주|소주|와인|막걸리)$")


@functools.lru_cache(maxsize=1)
def _dish_rows_by_suffix() -> dict[str, list[dict]]:
    """Dish rows grouped by the last 2-4 syllables of their name words (볶음, 버거, 치킨, 하이볼)."""
    out: dict[str, list[dict]] = {}
    words_of: list[tuple[dict, set[str]]] = []
    vocabulary: set[str] = set()
    for rows in _dish_rows_by_rep().values():
        for row in rows:
            words = {w for w in re.sub(r"\([^)]*\)", " ", row.get("name") or "").split() if len(w) >= 2}
            words_of.append((row, words))
            vocabulary |= words
    for row, words in words_of:
        for word in words:
            for size in (2, 3, 4):
                suffix = word[-size:]
                # only head nouns that are food words themselves (볶음, 치킨, 하이볼), not "이볼"
                if len(word) >= size and suffix in vocabulary:
                    out.setdefault(suffix, []).append(row)
    return out


def category_nutrition(food_name: str, amount=None, unit=None) -> dict | None:
    """Deterministic fallback when no DB row names the food: the median kcal/macros per 100 of the
    dish rows whose representative name is the query's head noun (longest rep_name the name ends
    with, at least two syllables and three rows), times the deterministic portion grams."""
    compact = "".join((food_name or "").split())
    if len(compact) < 3:
        return None
    by_suffix = _dish_rows_by_suffix()
    best = ""
    for size in (4, 3, 2):
        suffix = compact[-size:]
        if len(compact) > size and len(by_suffix.get(suffix, ())) >= _CATEGORY_MIN_ROWS:
            best = suffix
            break
    rows = by_suffix.get(best) if best else None
    if not rows:
        return None
    # one basis only: g rows, or ml rows for drinks (a soup's ml row and a dish's g row differ)
    basis = "ml" if _DRINK_SUFFIX_RE.search(compact) else "g"
    rows = [r for r in rows if (r.get("basis_unit") or "g") == basis]
    if len(rows) < _CATEGORY_MIN_ROWS:
        return None

    def med(key):
        vals = sorted(float(r[key]) for r in rows if r.get(key) not in (None, ""))
        return vals[len(vals) // 2] if vals else None
    kcal, protein, carbs, fat = med("kcal"), med("protein_g"), med("carb_g"), med("fat_g")
    if kcal is None:
        return None
    grams = resolve_portion_grams(food_name, amount, unit)
    out_amount, out_unit = amount, unit
    if grams is None:
        grams = serving_grams(food_name) or serving_grams(best) or 200.0
        out_amount, out_unit = 1.0, "인분"
    factor = grams / 100.0
    r1 = lambda v: round(v * factor, 1) if v is not None else None  # noqa: E731
    logger.info("[food_engine] category fallback '%s' (%s rows, basis %s, median %s kcal/100)",
                best, len(rows), basis, kcal)
    return {"kcal": r1(kcal), "protein_g": r1(protein), "carbs_g": r1(carbs), "fat_g": r1(fat),
            "amount": out_amount, "unit": out_unit, "category": best, "category_rows": len(rows),
            "basis": basis, "grams": grams}


def _median_density_row(meta_top: dict, query: str) -> dict | None:
    if (meta_top.get("data_type") or "") != "음식":
        return None
    rep = meta_top.get("rep_name") or ""
    q = "".join((query or "").split())
    if not rep or len(q) < 2:
        return None
    basis = meta_top.get("basis_unit") or "g"
    # same basis only: a dish's g row and ml row are measured differently (soup with broth)
    rows = [r for r in _dish_rows_by_rep().get(rep, [])
            if q in "".join((r.get("name") or "").split()) and (r.get("basis_unit") or "g") == basis]
    if len(rows) < _MEDIAN_MIN_ROWS:
        return None
    try:
        top_kcal = float(meta_top.get("kcal"))
        kcals = sorted(float(r["kcal"]) for r in rows)
    except (TypeError, ValueError):
        return None
    median = kcals[len(kcals) // 2]
    if top_kcal < 10:
        return None  # black coffee / plain tea rows are legitimately ~0 kcal; flavoured variants are not them
    if median <= 0 or abs(top_kcal - median) / median <= _MEDIAN_MAX_DEVIATION:
        return None
    best = min(rows, key=lambda r: abs(float(r["kcal"]) - median))

    def _num(v):
        try:
            return float(v) if v not in (None, "") else None
        except ValueError:
            return None
    logger.info("[food_engine] outlier row replaced by median-density row (%s -> %s)", top_kcal, best["kcal"])
    return {**meta_top, "name": best.get("name"), "kcal": _num(best.get("kcal")),
            "protein_g": _num(best.get("protein_g")), "carb_g": _num(best.get("carb_g")),
            "fat_g": _num(best.get("fat_g")), "sugar_g": _num(best.get("sugar_g")),
            "fiber_g": _num(best.get("fiber_g")), "sodium_mg": _num(best.get("sodium_mg")),
            "basis_value": _num(best.get("basis_value")) or 100.0, "basis_unit": best.get("basis_unit") or basis,
            "food_id": best.get("food_id")}

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
        reordered = index.lookup_reordered(analysis.normalized_query) if hasattr(index, "lookup_reordered") else None
        if reordered is not None:
            # "고구마돈가스" = DB row "돈가스 고구마" (grams still follow the user's name)
            doc = reordered.to_document()
            candidates.append(FoodSearchCandidate(
                document=doc, vector_score=0.0, source_query=analysis.normalized_query,
                source_query_index=_REORDERED_SOURCE_INDEX, original_rank=0, dedupe_key=build_food_candidate_dedupe_key(doc)))
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

    def _lookup_lexical_candidates(self, analysis):
        """Rows that name the query only as a parenthetical alias or a raw variety word, which the
        embedding search can miss. Appended last: a row already in the pool keeps its own rank.

        Each row carries its real embedding distance to the query (a vector search restricted to
        those rows), so a lexical hit still has to pass the similarity and lexical guards; when
        that distance is unavailable the rows are not offered at all."""
        index = getattr(self, "_canonical_index", None)
        store = getattr(self, "_vector_store", None)
        if index is None or store is None or not hasattr(index, "lookup_lexical") or analysis.protected:
            return []
        records = index.lookup_lexical(analysis.normalized_query, limit=TOP_K)
        ids = [r.metadata.get("food_id") for r in records if r.metadata.get("food_id")]
        if not ids:
            return []
        try:
            with self._lock:
                scored = store.similarity_search_with_score(
                    analysis.normalized_query, k=len(ids), filter={"food_id": {"$in": ids}})
        except Exception as exc:
            logger.debug("[food_engine] lexical rescoring failed: %s", exc)
            return []
        candidates = []
        for original_rank, (doc, distance) in enumerate(scored):
            candidates.append(FoodSearchCandidate(
                document=doc, vector_score=float(distance), source_query=analysis.normalized_query,
                source_query_index=_LEXICAL_SOURCE_INDEX, original_rank=original_rank,
                dedupe_key=build_food_candidate_dedupe_key(doc)))
        return candidates

    def _cooked_sibling(self, meta: dict, food_name: str) -> dict | None:
        """A dried pulse / pasta row ("병아리콩 ... 말린것") for a food the user ate cooked: its
        "말린것을 삶은 것" sibling row, unless the user said it was dried."""
        name = meta.get("name") or ""
        index = getattr(self, "_canonical_index", None)
        if (index is None or not hasattr(index, "lookup_name") or not name.endswith("말린것")
                or _DRIED_WORDS_RE.search(food_name or "")):
            return None
        siblings = [index.lookup_name(name + suffix) for suffix in ("을 삶은 것", "을 삶은것")]
        # else the cooked row of a sibling variety (렌즈콩 … 갈색 말린것 -> 렌즈콩 … 빨간색 말린것을
        # 삶은 것): same first word and word count, one word different
        words = name.split()
        for record in index.lookup(meta.get("rep_name"), limit=200):
            m = re.match(r"^(.*말린것)을\s*삶은\s*것$", record.metadata.get("name") or "")
            other = m.group(1).split() if m else []
            if (len(other) == len(words) and other[0] == words[0]
                    and sum(a != b for a, b in zip(other, words)) <= 1):
                siblings.append(record)
        sibling = next((s for s in siblings if s is not None), None)
        if sibling is None:
            return None
        logger.info("[food_engine] dried row replaced by its cooked sibling")
        return {**meta, **sibling.metadata}

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

    def search(self, food_name: str, amount=None, unit=None, small_portion: bool = False) -> Optional[dict]:
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
        candidates = self._dedupe_candidates(
            canonical_candidates + vector_candidates + self._lookup_lexical_candidates(analysis))
        candidates = rerank_food_candidates(analysis, candidates, final_k=FOOD_SEARCH_FINAL_K)
        reordered = [c for c in canonical_candidates if c.source_query_index == _REORDERED_SOURCE_INDEX]
        if reordered and not any(_compact_name(c) == "".join(analysis.normalized_query.split()) for c in candidates):
            # the exact dish in MFDS word order beats the generic head row (돈가스 for 고구마돈가스)
            candidates = reordered + [c for c in candidates if c.dedupe_key != reordered[0].dedupe_key]
        if not candidates:
            return None
        candidates = self._prefer_basis_unit(candidates, unit)

        # cosine space: score=거리(작을수록 유사), 코사인 유사도 = 1 - 거리
        doc = candidates[0].document
        distance = candidates[0].vector_score
        similarity = 1.0 - float(distance)
        meta_first = doc.metadata or {}
        query_compact = "".join((analysis.normalized_query or "").split())
        raw_name = meta_first.get("name") or ""
        paren_words = [w for part in re.findall(r"\(([^()]*)\)", raw_name) for w in re.split(r"[\s/,·]+", part)]
        head_words = [w for w in re.sub(r"\([^()]*\)", " ", raw_name).split()
                      if w not in {"생것", "구운것", "삶은것", "찐것", "볶은것", "튀긴것", "말린것", "데친것"}]
        # the queried name IS the row's food ("항정살" = "돼지고기 앞다리(항정살) 생것", "x" as the
        # head word): a weak embedding score must not hide it. A modifier word ("프로틴" in
        # "프로틴 돌체 라떼") is not that food.
        literal_hit = len(query_compact) >= 2 and (
            query_compact in paren_words or (head_words and head_words[-1] == query_compact))
        if similarity < SIMILARITY_THRESHOLD and not (literal_hit and similarity >= LITERAL_HIT_SIMILARITY):
            logger.debug("[food_engine] 임계값 미달: '%s' sim=%.3f", search_queries, similarity)
            return None
        meta_top = doc.metadata or {}
        if similarity < LEXICAL_GUARD_SIMILARITY and not has_lexical_relation(
            analysis.normalized_query, meta_top.get("name") or "", meta_top.get("rep_name") or ""
        ):
            # Embedding neighbour only (맥주 -> 맥치킨 버거, 소주 -> 소금): not this food. Let the
            # LLM estimate stand instead of logging another food's macros.
            logger.debug("[food_engine] 어휘 불일치: '%s' -> '%s' sim=%.3f", food_name, meta_top.get("name"), similarity)
            return None

        meta = _median_density_row(meta_top, analysis.normalized_query) or (doc.metadata or {})
        meta = self._cooked_sibling(meta, food_name) or meta
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
        if (grams is not None and (unit or "").strip() == "인분" and (meta.get("data_type") or "") == "음식"
                and portion_confidence(food_name, amount, unit) == "default" and not _protein_dense(meta)):
            # one serving of a prepared dish with no known serving (분짜, 쫄볶이 1인분): a meal portion,
            # not the 200 g raw-meat 1인분
            grams = float(amount) * _DISH_SERVING_G
        serving = serving_grams(food_name) if amount is None else None
        # the matched row's dish category refines blind table portions (공기 of any 밥류 row,
        # 접시 of a side dish, 전 pieces, 찜 home servings, banchan with no amount)
        category_grams = category_portion_grams(
            meta, food_name, amount, unit, grams if amount is not None else serving,
            basis_fallback=(grams is None and serving is None), small_portion=small_portion)
        if category_grams is not None:
            if amount is None:
                serving = category_grams
            else:
                grams = category_grams
        if grams is None and serving is not None:
            # 수량 미기재 → 1인분 기준 ("된장찌개 먹었어" 는 100g 이 아니라 한 그릇)
            factor = serving / float(basis_value)
            resolved_amount = 1.0
            resolved_unit = "인분"
            resolved_grams = serving
        elif grams is None:
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

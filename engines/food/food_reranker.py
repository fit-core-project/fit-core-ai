"""Deterministic food search reranking helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from engines.food.food_query_normalization import FoodQueryAnalysis


_COMPOUND_DISH_KEYWORDS = (
    "샌드위치",
    "볶음밥",
    "샐러드",
    "김밥",
    "찌개",
    "튀김",
    "조림",
    "덮밥",
    "빵",
    "국",
)

_COOKING_STATES = ("생것", "삶은것", "구운것", "후라이")


@dataclass(frozen=True)
class FoodSearchCandidate:
    document: Any
    vector_score: float
    source_query: str
    source_query_index: int
    original_rank: int
    dedupe_key: tuple


def get_food_candidate_metadata(document) -> dict:
    return document.metadata or {}


def get_food_candidate_name(document) -> str:
    meta = get_food_candidate_metadata(document)
    return str(meta.get("name") or "").strip()


def get_food_candidate_rep_name(document) -> str:
    meta = get_food_candidate_metadata(document)
    return str(meta.get("rep_name") or "").strip()


def get_food_candidate_data_type(document) -> str:
    meta = get_food_candidate_metadata(document)
    return str(meta.get("data_type") or "").strip()


def get_food_candidate_text(document) -> str:
    meta = get_food_candidate_metadata(document)
    parts = [
        str(meta.get("name") or ""),
        str(meta.get("rep_name") or ""),
        str(getattr(document, "page_content", "") or ""),
    ]
    return " ".join(part.strip() for part in parts if part).strip()


def build_food_candidate_dedupe_key(document) -> tuple:
    meta = get_food_candidate_metadata(document)
    for field in ("food_id", "id"):
        value = meta.get(field)
        if value:
            return (field, value)
    name = meta.get("name") or meta.get("rep_name")
    if name:
        return (
            "name",
            name,
            meta.get("data_type"),
            meta.get("major_category"),
        )
    page_content = getattr(document, "page_content", None)
    if page_content:
        return ("page_content", page_content)
    return ("metadata", tuple(sorted(meta.items())))


def _contains_exact_food_text(needle: str, haystack: str) -> bool:
    if not needle or not haystack:
        return False
    return needle == haystack or needle in haystack or haystack in needle


def _compact_food_text(value: str) -> str:
    return "".join(value.split())


def _canonical_match_score(analysis: FoodQueryAnalysis, candidate: FoodSearchCandidate) -> int:
    name = get_food_candidate_name(candidate.document)
    rep_name = get_food_candidate_rep_name(candidate.document)
    normalized = analysis.normalized_query
    if normalized and normalized == name:
        return 4
    # compact-exact: treat whitespace-only differences as exact match.
    # "해물스파게티"(user) vs "해물 스파게티"(name) -> compact equal -> score 4,
    # so the correct dish outranks a shorter substring candidate ("스파게티", score 2).
    # This is additive: existing substring logic (incl. name reverse for 흰쌀밥) is kept.
    if normalized and name and _compact_food_text(normalized) == _compact_food_text(name):
        return 4
    if normalized and normalized == rep_name:
        return 3
    if normalized and rep_name and _compact_food_text(normalized) == _compact_food_text(rep_name):
        return 3
    # name matches allow both directions: name is a specific food name, so
    # "쌀밥" ⊂ query "흰쌀밥" is a valid canonical equivalence.
    if _contains_exact_food_text(normalized, name):
        return 2
    # rep_name matches are forward-only: rep_name is a shared group root
    # ("고구마" covers 고구마 생것/찐것/구운것...), so rep_name ⊂ query
    # ("고구마" ⊂ "고구마 조림") is a false positive that lets a raw-ingredient
    # row outrank the correct dish. Stage-level diagnosis 2026-07-07.
    if normalized and rep_name and normalized in rep_name:
        return 2
    return 0


def _protected_dish_score(analysis: FoodQueryAnalysis, candidate: FoodSearchCandidate) -> int:
    if not analysis.protected or analysis.intent != "dish":
        return 0
    cleaned = analysis.cleaned_query
    name = get_food_candidate_name(candidate.document)
    rep_name = get_food_candidate_rep_name(candidate.document)
    if cleaned and (cleaned == name or cleaned == rep_name):
        return 4
    if cleaned and (cleaned in name or cleaned in rep_name):
        return 3
    return 0


def _cooking_state_score(analysis: FoodQueryAnalysis, candidate: FoodSearchCandidate) -> int:
    if not analysis.cooking_state:
        return 0
    text = get_food_candidate_text(candidate.document)
    if analysis.cooking_state in text:
        return 2
    mismatched = any(
        state != analysis.cooking_state and state in text
        for state in _COOKING_STATES
    )
    return -1 if mismatched else 0


def _compound_score(analysis: FoodQueryAnalysis, candidate: FoodSearchCandidate) -> int:
    if analysis.protected and analysis.intent == "dish":
        return 0
    if analysis.intent != "ingredient":
        return 0
    text = get_food_candidate_text(candidate.document)
    query_text = analysis.cleaned_query
    candidate_is_compound = any(keyword in text for keyword in _COMPOUND_DISH_KEYWORDS)
    query_is_compound = any(keyword in query_text for keyword in _COMPOUND_DISH_KEYWORDS)
    return -1 if candidate_is_compound and not query_is_compound else 0


def _data_type_score(analysis: FoodQueryAnalysis, candidate: FoodSearchCandidate) -> int:
    data_type = get_food_candidate_data_type(candidate.document)
    if analysis.intent == "ingredient" and data_type == "원재료성 식품":
        return 1
    if analysis.intent == "dish" and data_type == "음식":
        return 1
    return 0


def _token_overlap_score(analysis: FoodQueryAnalysis, candidate: FoodSearchCandidate) -> int:
    tokens = {token for token in analysis.normalized_query.split() if token}
    if not tokens:
        return 0
    text = get_food_candidate_text(candidate.document)
    return min(sum(1 for token in tokens if token in text), 2)


def _non_cooking_token_overlap_score(analysis: FoodQueryAnalysis, candidate: FoodSearchCandidate) -> int:
    tokens = {token for token in analysis.normalized_query.split() if token}
    if analysis.cooking_state:
        tokens = {
            token
            for token in tokens
            if token not in analysis.cooking_state and analysis.cooking_state not in token
        }
    if not tokens:
        return 0
    text = get_food_candidate_text(candidate.document)
    return min(sum(1 for token in tokens if token in text), 2)


def _rerank_key(analysis: FoodQueryAnalysis, candidate: FoodSearchCandidate) -> tuple:
    canonical = _canonical_match_score(analysis, candidate)
    cooking = _cooking_state_score(analysis, candidate)
    token_overlap = _token_overlap_score(analysis, candidate)
    # G1 fix: when the user specifies a cooking state and a candidate matches it,
    # promote that candidate above shorter substring matches (canonical=2). The
    # token guard prevents unrelated foods with only the same cooking state from
    # receiving the boost.
    if (
        analysis.cooking_state
        and cooking > 0
        and token_overlap > 0
        and _non_cooking_token_overlap_score(analysis, candidate) > 0
    ):
        canonical = max(canonical, 3)
    return (
        _protected_dish_score(analysis, candidate),
        canonical,
        cooking,
        _compound_score(analysis, candidate),
        _data_type_score(analysis, candidate),
        token_overlap,
        -candidate.source_query_index,
        -candidate.original_rank,
    )


def rerank_food_candidates(
    analysis: FoodQueryAnalysis,
    candidates: list[FoodSearchCandidate],
    final_k: int,
) -> list[FoodSearchCandidate]:
    reranked = sorted(candidates, key=lambda candidate: _rerank_key(analysis, candidate), reverse=True)
    return reranked[:final_k]

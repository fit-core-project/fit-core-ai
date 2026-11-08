"""Deterministic food search reranking helpers."""
from __future__ import annotations

import re
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


_PAREN_RE = re.compile(r"\(([^()]*)\)")
_TOKEN_SPLIT_RE = re.compile(r"[\s()\[\]/,&·]+")


def _food_tokens(value: str) -> list[str]:
    return [token for token in _TOKEN_SPLIT_RE.split(value or "") if token]


def _forward_match_kind(query: str, name: str) -> str:
    """How `query` appears inside a longer candidate `name`.

    - "prefix": name starts with the query as whole token(s) ("라면" -> "라면 치즈")
    - "paren": a whole token inside a parenthetical ("커피(아메리카노 2샷)")
    - "token": the query is a whole non-leading token ("치킨" -> "햄버거 치킨")
    - "infix": the query only appears inside a longer word
      ("콜라" -> "케이크 쇼콜라 케이크", "콜라" -> "트로피컬 콜라다")
    - "": no forward containment
    """
    if not query or not name or query not in name:
        return ""
    query_tokens = _food_tokens(query)
    name_tokens = _food_tokens(name)
    if not query_tokens or not name_tokens:
        return "infix"
    width = len(query_tokens)
    if name_tokens[:width] == query_tokens:
        return "prefix"
    # A whole token inside a parenthetical descriptor names the food itself:
    # "커피 원두 커피(아메리카노 2샷) 용액" for "아메리카노".
    for inner in _PAREN_RE.findall(name):
        inner_tokens = _food_tokens(inner)
        for start in range(0, len(inner_tokens) - width + 1):
            if inner_tokens[start:start + width] == query_tokens:
                return "paren"
    for start in range(1, len(name_tokens) - width + 1):
        if name_tokens[start:start + width] == query_tokens:
            return "token"
    return "infix"


# rep_name roots whose "<root> <variant>" rows name the variant itself
# (커피 아메리카노 == 아메리카노). Composite dishes (햄버거, 피자, 케이크, ...)
# are deliberately absent: "햄버거 치킨" is a burger, not chicken.
_CATEGORY_REP_NAMES = frozenset({
    "커피", "차", "기타차", "녹차", "홍차", "주스", "과일주스", "에이드",
    "탄산음료", "음료", "우유", "가공유", "두유", "발효유", "요구르트",
    "맥주", "소주", "와인", "막걸리",
})


def _is_category_variant(query: str, name: str, rep_name: str) -> bool:
    if not rep_name or rep_name not in _CATEGORY_REP_NAMES:
        return False
    name_tokens = _food_tokens(name)
    query_tokens = _food_tokens(query)
    rep_tokens = _food_tokens(rep_name)
    return (
        name_tokens[:len(rep_tokens)] == rep_tokens
        and name_tokens[len(rep_tokens):len(rep_tokens) + len(query_tokens)] == query_tokens
    )


def is_implausible_food_candidate(analysis: FoodQueryAnalysis, candidate: FoodSearchCandidate) -> bool:
    """True when the candidate only matches the query as a substring of an
    unrelated longer dish name (콜라 -> 쇼콜라 케이크, 치킨 -> 햄버거 치킨).

    Protected queries keep their legacy substring behaviour (닭가슴살 ->
    샐러드 닭가슴살 is an intentional golden mapping). Candidates without any
    lexical overlap are left to the vector similarity threshold.
    """
    if analysis.protected:
        return False
    if _canonical_match_score(analysis, candidate) > 0:
        return False
    queries = {q for q in (analysis.normalized_query, analysis.cleaned_query) if q}
    names = [
        get_food_candidate_name(candidate.document),
        get_food_candidate_rep_name(candidate.document),
    ]
    for query in queries:
        for name in names:
            if _forward_match_kind(query, name) in {"token", "infix"}:
                return True
    return False


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
    if analysis.protected:
        # Protected queries keep the legacy bidirectional substring rule
        # (닭가슴살 -> 샐러드 닭가슴살 is an intentional golden mapping).
        if _contains_exact_food_text(normalized, name):
            return 2
        if normalized and rep_name and normalized in rep_name:
            return 2
        return 0
    # Reverse direction (name ⊂ query): "쌀밥" ⊂ "흰쌀밥" (head-noun suffix) and
    # "국수 말린것" ⊂ "삶은 국수 말린것을" (starts at a token boundary) are valid
    # canonical equivalences.
    if normalized and name and len(name) >= 2 and normalized != name and name in normalized:
        position = normalized.find(name)
        if (
            _compact_food_text(normalized).endswith(_compact_food_text(name))
            or position == 0
            or normalized[position - 1].isspace()
        ):
            return 2
    # Forward direction (query ⊂ name) only counts as a leading token match
    # ("라면" -> "라면 치즈"). A query buried inside a longer word
    # ("콜라" -> "쇼콜라 케이크") or trailing a different dish root
    # ("치킨" -> "햄버거 치킨") is not a canonical match.
    name_kind = _forward_match_kind(normalized, name)
    if name_kind == "prefix":
        return 2
    # "커피 아메리카노": rep_name is a plain beverage/category root and the query
    # is the variant itself. Unlike "햄버거 치킨" (치킨 is a burger filling), the
    # product IS the queried food.
    if name_kind == "token" and _is_category_variant(normalized, name, rep_name):
        return 2
    # rep_name matches are forward-only: rep_name is a shared group root
    # ("고구마" covers 고구마 생것/찐것/구운것...), so rep_name ⊂ query
    # ("고구마" ⊂ "고구마 조림") is a false positive that lets a raw-ingredient
    # row outrank the correct dish. Stage-level diagnosis 2026-07-07.
    if _forward_match_kind(normalized, rep_name) == "prefix":
        return 2
    if name_kind == "paren":
        return 1
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
    plausible = [
        candidate for candidate in candidates
        if not is_implausible_food_candidate(analysis, candidate)
    ]
    if len(plausible) < len(candidates) and not any(
        _canonical_match_score(analysis, candidate) > 0
        or _token_overlap_score(analysis, candidate) > 0
        for candidate in plausible
    ):
        # Every lexical hit was a false substring match (콜라 -> 쇼콜라 케이크) and
        # what is left has no lexical relation to the query (치킨 -> 치커리):
        # report no DB hit instead of a wrong one.
        return []
    reranked = sorted(plausible, key=lambda candidate: _rerank_key(analysis, candidate), reverse=True)
    return reranked[:final_k]

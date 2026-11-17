"""Deterministic food search reranking helpers."""
from __future__ import annotations

import csv
import difflib
import functools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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


# Composite dishes: "<rep> <variant>" rows are that dish, so a variant token only names
# the query when the query is itself that dish ("불고기버거" -> 햄버거 소불고기버거) — not
# "치킨" -> 햄버거 치킨 or "콜라" -> 케이크 쇼콜라 케이크.
_COMPOSITE_DISH_REPS = frozenset({
    "햄버거", "버거", "피자", "피자빵", "케이크", "페이스트리", "크로플", "샐러드", "샌드위치",
    "김밥", "토스트", "도시락", "덮밥", "볶음밥", "비빔밥", "크로와상", "머핀", "와플", "롤",
    "스무디", "에이드", "빵", "쿠키", "비스킷/쿠키/크래커",
})
# "<query><suffix>" is still the query food, cooked a certain way (삼겹살 -> 삼겹살구이).
_COOKING_SUFFIXES = ("구이", "볶음", "찜", "튀김", "무침", "조림", "전", "수육", "회", "말이")


def _composite_heads(rep_name: str) -> tuple[str, ...]:
    heads = {rep_name}
    if len(rep_name) >= 3:
        heads.add(rep_name[-2:])  # 햄버거 -> 버거
    return tuple(heads)


def _token_match_is_plausible(query: str, token: str, rep_name: str, data_type: str) -> bool:
    """Korean compounds put the head last: 멥쌀밥 is a kind of 밥, 치킨치즈마요버거 is not 치킨."""
    composite = rep_name in _COMPOSITE_DISH_REPS
    names_composite = composite and query.endswith(_composite_heads(rep_name))
    if token == query:
        if data_type.startswith("원재료") or rep_name in _CATEGORY_REP_NAMES:
            return True  # 돼지고기 삼겹살, 토마토 방울토마토, 커피 바닐라라떼
        return names_composite if composite else True  # 감자 찐감자, 닭튀김 너겟
    if token.endswith(query):
        return names_composite if composite else True  # 멥쌀밥 for 밥
    if token.startswith(query) and token[len(query):] in _COOKING_SUFFIXES:
        return not composite
    return False


# A drink row for a food query (티라미수 -> 커피 티라미수 라떼, 자몽 -> 에이드 자몽 에이드) or a
# dessert row for a drink query (초코라떼 -> 마카롱 초코라떼 마카롱): the flavour name matches,
# the food category does not.
_DRINK_WORDS = ("커피", "라떼", "에이드", "스무디", "쉐이크", "주스", "음료", "프라푸치노", "아이스티",
                "차", "티", "밀크티", "콜드브루", "버블티", "우유", "요구르트", "드링크", "소다",
                "카푸치노", "플랫화이트", "에스프레소", "마끼아또", "마키아토", "모카", "아메리카노",
                "아인슈페너", "돌체", "샷", "아샷추", "주스")
_CONDIMENT_WORDS = ("소스", "드레싱", "양념장", "시럽", "분말", "가루", "육수")
_DESSERT_REPS = ("마카롱", "케이크", "쿠키", "빵", "크로플", "와플", "도넛", "아이스크림", "빙수", "타르트", "푸딩")


def _is_drink_text(text: str) -> bool:
    compact = _compact_food_text(text or "")
    return any(compact.endswith(w) or w in compact for w in _DRINK_WORDS if len(w) >= 2) or compact.endswith(("차", "티"))


def _category_mismatch(analysis: FoodQueryAnalysis, candidate: FoodSearchCandidate) -> bool:
    query = analysis.normalized_query or ""
    name = _compact_food_text(get_food_candidate_name(candidate.document) or "")
    # a sauce / dressing row for a dish query ("까르보나라 파스타" -> 까르보나라소스)
    if any(w in name for w in _CONDIMENT_WORDS) and not any(w in query for w in _CONDIMENT_WORDS):
        return True
    rep = _compact_food_text(get_food_candidate_rep_name(candidate.document) or "")
    if not query or not rep or _compact_food_text(query) == rep:
        return False
    rep_is_drink = rep in {_compact_food_text(w) for w in _DRINK_WORDS if len(w) >= 2}
    if rep_is_drink and not _is_drink_text(query):
        return True
    if rep in _DESSERT_REPS and _is_drink_text(query) and not any(d in query for d in _DESSERT_REPS):
        return True
    return False


def is_implausible_food_candidate(analysis: FoodQueryAnalysis, candidate: FoodSearchCandidate) -> bool:
    """True when the query only appears in the candidate as a modifier or substring of a
    different food (콜라 -> 쇼콜라 케이크, 치킨 -> 햄버거 치킨치즈마요버거), not as its head noun.

    Protected queries keep their legacy substring behaviour (닭가슴살 -> 샐러드 닭가슴살 is an
    intentional golden mapping). Candidates without lexical overlap are left to the vector
    similarity threshold.
    """
    if analysis.protected:
        return False
    if not _base_implausible(analysis, candidate):
        return False
    if _names_composite_variant(analysis, candidate) or _query_is_name_tail(analysis, candidate):
        return False
    return True


# Size / temperature markers that trail a DB row name without naming a different food
# ("커피 아메리카노 아이스 L", "피자 불고기 피자 R").
_SIZE_MARKER_RE = re.compile(
    r"^(?:핫|아이스|HOT|ICED|hot|iced|R|L|M|S|XL|EX|J|\d+(?:\.\d+)?(?:g|ml|개|조각|인분|리터|L)?|\d+인)$"
)
_NAME_WORD_SPLIT_RE = re.compile(r"[\s/,&·+]+")
_FOOD_DB_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "food_db" / "food_db_clean.csv"


def _name_words(name: str) -> list[str]:
    """Row-name words without parentheticals, cooking-state words and size markers."""
    stripped = _PAREN_RE.sub(" ", name or "")
    state_words = set(_STATE_WORDS) | {"후라이"}
    return [
        w for w in _NAME_WORD_SPLIT_RE.split(stripped)
        if w and w not in state_words and not _SIZE_MARKER_RE.match(w)
    ]


@dataclass(frozen=True)
class _FoodNameVocab:
    # last word of every non-composite (rep not a composite dish) prepared-food row
    standalone_heads: frozenset
    # every rep_name, compacted
    rep_names: frozenset
    # rep_names of raw-ingredient rows (달걀, 감자, 딸기): heads of descriptive compounds only
    raw_rep_names: frozenset = frozenset()


@functools.lru_cache(maxsize=1)
def _food_name_vocab() -> Optional[_FoodNameVocab]:
    """Vocabulary of the food DB used to tell a standalone food from a dish variant name."""
    if not _FOOD_DB_CSV_PATH.exists():
        return None
    heads, reps, raw_reps = set(), set(), set()
    with _FOOD_DB_CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rep = _compact_food_text(_PAREN_RE.sub("", row.get("rep_name") or ""))
            if (row.get("data_type") or "").startswith("원재료"):
                raw_reps.add(rep)
                continue
            reps.add(rep)
            words = _name_words(row.get("name") or "")
            if words and rep not in _COMPOSITE_DISH_REPS:
                heads.add(words[-1])
    return _FoodNameVocab(standalone_heads=frozenset(heads), rep_names=frozenset(reps),
                          raw_rep_names=frozenset(raw_reps))


# English loanword heads of descriptive compounds, as the DB spells that food
_LOANWORD_HEADS = {"에그": "달걀", "밀크": "우유"}


def _names_composite_variant(analysis: FoodQueryAnalysis, candidate: FoodSearchCandidate) -> bool:
    """A one-word query that is a whole variant word of a composite-dish row and never a food on
    its own: "브라우니" -> 케이크 브라우니, "휘낭시에" -> 머핀 휘낭시에. "치킨" -> 햄버거 치킨 stays
    implausible because 치킨 heads standalone rows (닭튀김 ... 치킨)."""
    vocab = _food_name_vocab()
    query = analysis.normalized_query or ""
    if vocab is None or len(_food_tokens(query)) != 1:
        return False
    q = _compact_food_text(query)
    words = _name_words(get_food_candidate_name(candidate.document))
    if q not in words[1:] or q in vocab.standalone_heads or q in vocab.rep_names:
        return False
    # a descriptive compound whose head noun is a food on its own (딸기우유, 감자베이컨,
    # 생딸기요거트): it names that food, not a dessert / pizza variant
    known = (vocab.standalone_heads, vocab.rep_names, vocab.raw_rep_names)
    for i in range(1, len(q) - 1):
        suffix = q[i:]
        suffix = _LOANWORD_HEADS.get(suffix, suffix)  # 햄에그: 에그 is 달걀
        if any(suffix in words_set for words_set in known):
            return False
    # a drink word (쉐이크) is not a variant of a smoothie / café row
    if _is_drink_text(q):
        return False
    # "피자 감자베이컨 피자", "크로플 누텔라 크로플": a word followed by the row's category word
    # again is that category's topping / flavour, not the item itself (케이크 브라우니)
    return words[0] not in words[words.index(q) + 1:]


def _query_is_name_tail(analysis: FoodQueryAnalysis, candidate: FoodSearchCandidate) -> bool:
    """A multi-word query that is the tail of the row name names that row:
    "까르보나라 파스타" -> 스파게티 까르보나라 파스타, "토마토 주스" -> 과ㆍ채주스 토마토 주스."""
    query_tokens = _food_tokens(analysis.normalized_query or "")
    if len(query_tokens) < 2:
        return False
    words = _name_words(get_food_candidate_name(candidate.document))
    return words[-len(query_tokens):] == query_tokens


def _base_implausible(analysis: FoodQueryAnalysis, candidate: FoodSearchCandidate) -> bool:
    if _names_only_the_modifier(analysis.normalized_query, candidate):
        return True
    if _canonical_match_score(analysis, candidate) > 0:
        return False
    queries = {q for q in (analysis.normalized_query, analysis.cleaned_query) if q}
    name = get_food_candidate_name(candidate.document)
    rep_name = get_food_candidate_rep_name(candidate.document)
    data_type = get_food_candidate_data_type(candidate.document) or ""
    for query in queries:
        kind = _forward_match_kind(query, name)
        if kind in {"", "prefix", "paren"}:
            continue
        tokens = [t for t in _food_tokens(name) if query in t]
        if not any(_token_match_is_plausible(query, t, rep_name, data_type) for t in tokens):
            return True
    return False


# Dish heads that make "<food><head>" a different dish from the food itself: 두부면 is noodles,
# not tofu; 두부포케 a poke bowl. Cooking suffixes (구이, 볶음, 튀김 ...) are not here: 삼겹살구이
# is still 삼겹살.
_DISH_HEADS = frozenset({
    "면", "국", "탕", "밥", "죽", "국수", "김밥", "빵", "떡", "찌개", "전골", "샐러드", "포케", "라떼",
    "주스", "스무디", "국밥", "덮밥", "볶음밥", "버거", "피자", "케이크", "칩", "쿠키", "우유", "수프",
    "스프", "짜장", "짬뽕", "파스타", "스파게티", "만두", "냉면", "라면", "쌈",
})
_STATE_WORDS = ("생것", "구운것", "삶은것", "찐것", "볶은것", "튀긴것", "말린것", "데친것")


def _names_only_the_modifier(query: str, candidate: FoodSearchCandidate) -> bool:
    """The row is the plain food that is only the modifier of the queried dish (두부면 -> 두부)."""
    q = _compact_food_text(query or "")
    name_words = [w for w in _PAREN_RE.sub(" ", get_food_candidate_name(candidate.document) or "").split()
                  if w not in _STATE_WORDS]
    name = "".join(name_words)
    if len(name) < 2 or len(q) <= len(name) or not q.startswith(name):
        return False
    return q[len(name):] in _DISH_HEADS


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
        end = position + len(name)
        # A name at the start only counts when it is a whole word ("국수 말린것" in "삶은 국수
        # 말린것을"); "순대" at the start of the single word "순대국밥" is a modifier, not the dish.
        starts_whole_word = position == 0 and (end == len(normalized) or normalized[end].isspace())
        if (
            _compact_food_text(normalized).endswith(_compact_food_text(name))
            or starts_whole_word
            or (position > 0 and normalized[position - 1].isspace())
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


# Processed forms a bare food name should not resolve to (귤 -> 귤 착즙, 사과 -> 사과 잼).
_PROCESSED_MARKERS = ("착즙", "주스", "에이드", "시럽", "잼", "통조림", "분말", "말린것", "칩", "청", "농축", "건조")


def _plainness_score(analysis: FoodQueryAnalysis, candidate: FoodSearchCandidate) -> int:
    """For a bare generic query, prefer the plain row over named variants.

    "귤" should land on 귤 생것, not 귤 착즙; "샐러드" on 샐러드 채소, not a branded salad.
    Only applies when the query names no variant itself (single token, no cooking state).
    """
    query = analysis.normalized_query
    if not query or analysis.cooking_state or len(_food_tokens(query)) != 1:
        return 0
    name = get_food_candidate_name(candidate.document)
    tokens = _food_tokens(name)
    if _forward_match_kind(query, name) == "prefix":
        extra = tokens[1:]
    elif query in tokens[1:] and get_food_candidate_data_type(candidate.document) == "원재료성 식품":
        # raw-ingredient rows name the variety second: "옥수수 찰옥수수 말린것", "호박 단호박 생것"
        extra = tokens[tokens.index(query) + 1:]
    else:
        return 0
    if any(marker in token for token in extra for marker in _PROCESSED_MARKERS) and not any(
        marker in query for marker in _PROCESSED_MARKERS
    ):
        return -3
    plain_state = [t for t in extra if t not in _COOKING_STATES]
    return -min(len(plain_state), 2)


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
    if _category_mismatch(analysis, candidate):
        # a drink row for a food query (or the reverse) only wins when nothing else matches
        canonical -= 5
    return (
        _protected_dish_score(analysis, candidate),
        canonical,
        cooking,
        _plainness_score(analysis, candidate),
        _compound_score(analysis, candidate),
        _data_type_score(analysis, candidate),
        token_overlap,
        -candidate.source_query_index,
        -candidate.original_rank,
    )


_MODIFIER_PREFIXES = ("한우", "돼지", "국산", "수입", "수제", "소")


_HANGUL_BASE = 0xAC00
_HANGUL_COUNT = 11172


def has_lexical_relation(query: str, name: str, rep_name: str = "") -> bool:
    """`_direct_lexical_relation`, plus the 계란/달걀 synonym and one-vowel spelling variants
    (배추젼 ~ 배추전, 현미쥭 ~ 현미죽, 멘보사 ~ 멘보샤, 깐쇼새오 ~ 깐쇼새우)."""
    if _direct_lexical_relation(query, name, rep_name):
        return True
    q = _compact_food_text(query or "").replace("계란", "달걀")
    if _direct_lexical_relation(q, (name or "").replace("계란", "달걀"), rep_name):
        return True
    if len(q) < 3:
        return False
    for word in [w for w in _food_tokens(name or "") if len(w) >= 3] + [_compact_food_text(name or "")]:
        if _vowel_spelling_variant(q, word):
            return True
    return False


def _vowel_spelling_variant(a: str, b: str) -> bool:
    """Same syllables except one vowel (배추젼 ~ 배추전, 현미쥭 ~ 현미죽, 깐쇼새오 ~ 깐쇼새우).
    A changed initial or final consonant changes the food (녹두죽 / 녹두묵, 미역죽 / 미역국,
    칼국수 / 쌀국수, 새우깡 / 새우탕), so only a vowel may differ."""
    if len(a) != len(b) or a == b:
        return False
    diffs = 0
    for x, y in zip(a, b):
        if x == y:
            continue
        ox, oy = ord(x) - _HANGUL_BASE, ord(y) - _HANGUL_BASE
        if not (0 <= ox < _HANGUL_COUNT and 0 <= oy < _HANGUL_COUNT):
            return False
        same_initial = ox // 588 == oy // 588
        same_final = ox % 28 == oy % 28
        if not (same_initial and same_final):
            return False
        diffs += 1
    return diffs == 1


def _direct_lexical_relation(query: str, name: str, rep_name: str = "") -> bool:
    """Whether a DB row names the queried food at all, ignoring spaces.

    True when one contains the other ("양념치킨" ~ "닭튀김 양념 치킨", "저지방 우유" ~ "우유") or
    every query word appears in the name. "맥주" vs "버거 맥치킨 버거" or "소주" vs "소금" are
    only embedding neighbours and fail this check.
    """
    qc, nc, rc = _compact_food_text(query or ""), _compact_food_text(name or ""), _compact_food_text(rep_name or "")
    if not qc or not nc:
        return False
    if qc in nc or nc in qc or (len(rc) >= 2 and rc in qc):
        return True
    words = [w for w in _food_tokens(query) if len(w) >= 2]
    if words and all(w in nc for w in words):
        return True
    # Spelling variants of the same dish: 쭈꾸미볶음 ~ 주꾸미볶음, 아구찜 ~ 아귀찜, 오리로스 ~
    # 오리고기로스, 항정살 ~ 앞다리(항정살). Short words stay strict (소주 vs 소금, 맥주 vs 맥치킨).
    for prefix in _MODIFIER_PREFIXES:
        # "소대창" ~ 대창구이, "돼지막창" ~ 막창구이: an origin word in front of the dish name
        head = qc[len(prefix):]
        if qc.startswith(prefix) and len(head) >= 2 and head in nc:
            return True
    if len(qc) >= 3:
        name_words = [w for w in _food_tokens(name or "") if len(w) >= 2] + [nc]
        for w in name_words:
            if w == qc or (len(w) >= 3 and (qc in w or w in qc)):
                return True
            ratio = difflib.SequenceMatcher(None, qc, w).ratio()
            if ratio >= 0.8 or (ratio >= 0.66 and len(w) == len(qc) and qc[-1] == w[-1] and qc[0] == w[0]):
                return True
    return False


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
        or has_lexical_relation(analysis.normalized_query, get_food_candidate_name(candidate.document),
                                get_food_candidate_rep_name(candidate.document))
        for candidate in plausible
    ):
        # Every lexical hit was a false substring match (콜라 -> 쇼콜라 케이크) and
        # what is left has no lexical relation to the query (치킨 -> 치커리):
        # report no DB hit instead of a wrong one.
        return []
    reranked = sorted(plausible, key=lambda candidate: _rerank_key(analysis, candidate), reverse=True)
    return reranked[:final_k]

"""Branded / packaged products with label nutrition (data/curated/curated_products.csv).

The public food DB has generic dishes per 100 g, not "신라면 1봉" or "바나나맛우유 1개". These
rows carry per-unit label values (kcal is explicit, so items whose energy the macros do not
explain stay correct). Lookup is by exact name/alias, then by the longest alias contained in
the logged name ("편의점 바나나우유" -> 바나나맛우유).
"""
from __future__ import annotations

import csv
import re
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .food_units import GRAM_UNITS, food_unit_overrides

logger = logging.getLogger(__name__)

PRODUCTS_CSV = Path(__file__).resolve().parents[2] / "data" / "curated" / "curated_products.csv"
# Generated from the MFDS processed-food dataset by scripts/build_product_index.py (not committed).
PROCESSED_INDEX_CSV = Path(__file__).resolve().parents[2] / "data" / "food_db" / "processed_product_index.csv"
_CONTAINER_UNITS = {"봉지", "봉", "캔", "병", "팩", "통", "갑", "컵", "줄", "사발"}
_PIECE_UNITS = {"", "개", "인분", "판"}
# Plain-variant markers: "몽쉘" alone usually means 몽쉘 오리지널.
_PLAIN_MARKERS = ("오리지널", "original", "클래식", "classic", "플레인", "plain", "고소한맛", "기본", "순한맛")
_MAX_BRAND_PREFIX = 4
_BULK_PACK_G = 2000.0
_PLACEHOLDER_SERVING_G = 100.0
_DRINK_CATEGORIES = ("음료류", "유가공품류", "주류")
_GLASS_UNITS = {"잔", "컵"}
_MAX_VARIANT_SUFFIX = 6
_PACKAGED_CATEGORIES = {
    "과자류·빵류 또는 떡류", "음료류", "코코아가공품류 또는 초콜릿류", "면류", "유가공품류",
    "빙과류", "특수영양식품", "당류", "주류", "잼류",
}
# Units that mean "one package of this product".
_PACKAGE_UNITS = {"", "개", "봉지", "봉", "병", "캔", "갑", "팩", "통", "컵", "잔", "인분", "판", "세트"}


@dataclass(frozen=True)
class Product:
    name: str
    unit_grams: float
    kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    confidence: str


def _compact(value: str) -> str:
    return "".join((value or "").lower().split())


@lru_cache(maxsize=1)
def _index() -> dict[str, Product]:
    index: dict[str, Product] = {}
    if not PRODUCTS_CSV.exists():
        logger.warning("[curated_products] missing %s", PRODUCTS_CSV)
        return index
    with PRODUCTS_CSV.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            product = Product(
                name=row["name"],
                unit_grams=float(row["unit_grams"]),
                kcal=float(row["kcal"]),
                protein_g=float(row["protein_g"]),
                carbs_g=float(row["carbs_g"]),
                fat_g=float(row["fat_g"]),
                confidence=row.get("confidence") or "",
            )
            for key in [row["name"], *(row.get("aliases") or "").split("|")]:
                if key.strip():
                    index[_compact(key)] = product
    return index


@dataclass(frozen=True)
class IndexedProduct:
    key: str
    name: str
    kcal_100: float
    protein_100: float
    carbs_100: float
    fat_100: float
    unit_g: float
    package_sizes: tuple[float, ...]
    n_rows: int
    maker_root: str = ""
    category_lv3: str = ""
    serving_g: float = 0.0


def _norm_key(value: str) -> str:
    import re
    value = re.sub(r"[\(\[\{（【].*?[\)\]\}）】]", " ", value or "")
    return re.sub(r"[^0-9A-Za-z가-힣]", "", value).lower()


@lru_cache(maxsize=1)
def _processed_index() -> dict[str, IndexedProduct]:
    index: dict[str, IndexedProduct] = {}
    # FOOD_PROCESSED_INDEX=off: behave as if the generated index were absent (CI, A/B checks).
    if os.getenv("FOOD_PROCESSED_INDEX", "").strip().lower() in {"0", "off", "false"}:
        return index
    if not PROCESSED_INDEX_CSV.exists():
        return index
    with PROCESSED_INDEX_CSV.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                index[row["key"]] = IndexedProduct(
                    key=row["key"], name=row["name"], kcal_100=float(row["kcal_100"]),
                    protein_100=float(row["protein_100"]), carbs_100=float(row["carbs_100"]),
                    fat_100=float(row["fat_100"]), unit_g=float(row["unit_g"]),
                    package_sizes=tuple(float(x) for x in row["package_sizes"].split("|") if x),
                    n_rows=int(row["n_rows"]), maker_root=row.get("maker_root") or "",
                    category_lv3=row.get("category_lv3") or "",
                    serving_g=float(row.get("serving_g") or 0.0),
                )
            except (KeyError, ValueError):
                continue

    logger.info("[curated_products] processed index: %s products", len(index))
    return index


GENERIC_FOOD_CSV = Path(__file__).resolve().parents[2] / "data" / "food_db" / "food_db_clean.csv"


@lru_cache(maxsize=1)
def _generic_food_keys() -> frozenset[str]:
    """Names of generic foods/dishes (라면, 바나나, 된장찌개 …): these stay on the generic DB /
    one-serving path. The processed index also holds products literally named "라면" (multi-packs)
    or "바나나" (a snack), which would otherwise hijack everyday words."""
    keys: set[str] = set()
    if GENERIC_FOOD_CSV.exists():
        with GENERIC_FOOD_CSV.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                for col in ("rep_name", "name"):
                    key = _norm_key(row.get(col) or "")
                    if key:
                        keys.add(key)
                # words of the representative (generic) name: "튀김", "커피". Not every word of
                # `name`: the dish DB also lists franchise items whose words are brands ("몽쉘").
                rep = row.get("rep_name") or ""
                keys.update(w for w in (_norm_key(t) for t in rep.split()) if len(w) >= 2)
                # raw-ingredient rows name varieties, never brands ("귤 한라봉 생것" -> 한라봉)
                # dish variants named "<variant><category>" in the dish DB ("콤비네이션피자",
                # "고구마피자", "물냉면") are menu dishes, not packaged products
                if not (row.get("data_type") or "").startswith("원재료"):
                    rep_words = [w for w in (_norm_key(t) for t in rep.split()) if len(w) >= 2]
                    dish_name = re.sub(r"\([^)]*\)", "", row.get("name") or "")
                    for token in dish_name.split():
                        word = _norm_key(token)
                        if any(word != rw and word.endswith(rw) for rw in rep_words):
                            keys.add(word)
                if (row.get("data_type") or "").startswith("원재료"):
                    keys.update(w for w in (_norm_key(t) for t in (row.get("name") or "").split()) if len(w) >= 2)
    from .food_query_normalization import FOOD_SEARCH_ALIAS_RULES
    from .food_units import FOOD_UNIT_GRAMS, SERVING_GRAMS_BY_SUFFIX
    keys.update(_norm_key(k) for k in FOOD_UNIT_GRAMS)
    keys.update(_norm_key(k) for k in SERVING_GRAMS_BY_SUFFIX)
    keys.update(_norm_key(k) for k in FOOD_SEARCH_ALIAS_RULES)
    return frozenset(keys)


def _is_generic_query(food_name: str) -> bool:
    """"삶은계란", "단백질 쉐이크", "블랙 커피": made only of generic food words -> generic DB path."""
    generic = _generic_food_keys()
    key = _norm_key(food_name)
    if key in generic:
        return True
    words = [_norm_key(t) for t in (food_name or "").split()]
    words = [w for w in words if w]
    return len(words) > 1 and all(w in generic for w in words)


def _is_packaged(product: IndexedProduct) -> bool:
    """Snacks, drinks, ramen, dairy, ice cream … (not ready meals / meat / seafood dishes)."""
    return not product.category_lv3 or product.category_lv3 in _PACKAGED_CATEGORIES


@lru_cache(maxsize=1)
def _dish_prefix_keys() -> tuple[str, ...]:
    """Generic dish-variant keys (ending with a dish category word), for prefix tests."""
    return tuple(k for k in _generic_food_keys() if len(k) >= 4)


def _is_dish_phrase(words: list[str]) -> bool:
    """"까르보나라 파스타", "크림 우동": a menu dish named in words, ending with a dish category."""
    return _norm_key(words[-1]) in _generic_food_keys()


def _names_a_brand(food_name: str) -> bool:
    """"비비고 왕교자", "동원참치 살코기": several words led by a non-generic (brand) word."""
    words = [w for w in (food_name or "").split() if w]
    if len(words) < 2:
        return False
    first = _norm_key(words[0])
    generic = _generic_food_keys()
    if first in generic:
        return False
    # "까르보나라 파스타": the first word is the start of a menu dish name (까르보나라스파게티),
    # not a brand
    return not (len(first) >= 3 and any(k.startswith(first) and len(k) > len(first) for k in _dish_prefix_keys()))


_FLAVOUR_WORDS = {
    "딸기": "스트로베리", "포도": "그레이프", "청포도": "화이트그레이프", "사과": "애플", "복숭아": "피치",
    "오렌지": "오렌지", "망고": "망고", "레몬": "레몬", "초코": "초콜릿", "바닐라": "바닐라",
}


def _flavour_variants(key: str) -> list[str]:
    out: list[str] = []
    base = key[:-1] if key.endswith(("맛", "향")) else key
    out.append(base)
    out.append(base + "맛")
    for ko, en in _FLAVOUR_WORDS.items():
        if base.endswith(ko):
            stem = base[: -len(ko)]
            out += [stem + en, stem + en + "맛"]
    return [v for v in dict.fromkeys(out) if v and v != key]


def find_processed_product(food_name: Optional[str], allow_single_line: bool = True) -> Optional[IndexedProduct]:
    """Exact normalised name, else the plain variant of a product family ("몽쉘" -> 몽쉘 오리지널).

    Only for brand / product names: a generic food word never resolves here.
    """
    key = _norm_key(food_name or "")
    if len(key) < 2 or _is_generic_query(food_name or ""):
        return None
    index = _processed_index()
    if key not in index:
        # flavour spelling: "불가리스 사과맛" = 불가리스 사과, "웰치스 딸기" = 웰치스 스트로베리맛
        for variant in _flavour_variants(key):
            if variant in index and (_is_packaged(index[variant]) or _names_a_brand(food_name or "")):
                return index[variant]
    if key in index:
        hit = index[key]
        words = (food_name or "").split()
        if _names_a_brand(food_name or "") or (_is_packaged(hit) and (len(words) < 2 or not _is_dish_phrase(words))):
            return hit
        # ready meal / meat / seafood row with a dish name ("제육 덮밥", "목살"): generic dish path
        return None
    for marker in _PLAIN_MARKERS:
        if key + marker in index:
            return index[key + marker]  # "스팸" -> 스팸 클래식, "몽쉘" -> 몽쉘 오리지널
    if not allow_single_line:
        return None  # strict mode: exact names only, no product-line guessing
    family = [p for k, p in index.items()
              if k.startswith(key) and len(k) - len(key) <= _MAX_VARIANT_SUFFIX and _is_packaged(p)]
    # The user may drop the brand prefix of a product line ("빨계떡" -> 틈새라면 빨계떡,
    # "왕뚜껑" -> 팔도 김치왕뚜껑): packaged names that END with the query.
    family += [p for k, p in index.items()
               if k.endswith(key) and k != key and len(k) - len(key) <= _MAX_BRAND_PREFIX
               and _is_packaged(p) and p not in family]
    # A product line ("몽쉘 …", "포카칩 …") has several variants from one company. A one-off
    # longer name ("삶은계란모양구미") or a word many makers use is not the product the user named.
    if (allow_single_line and len(family) == 1 and len(key) >= 3 and family[0].key.startswith(key)
            and len((food_name or "").split()) == 1):
        # a one-product line named by its brand ("데자와" -> 데자와 로얄밀크티)
        return family[0]
    if len(family) < 2:
        return None
    # Company root, first two syllables: 롯데제과 / 롯데웰푸드 (renamed) are one maker.
    makers = [p.maker_root[:2] for p in family if p.maker_root]
    if makers:
        top = max(sorted(set(makers)), key=makers.count)  # sorted: ties must not depend on the hash seed
        if makers.count(top) / len(makers) < 0.5:
            return None
        family = [p for p in family if p.maker_root[:2] == top] or family
    return max(family, key=lambda p: (p.n_rows, -len(p.key)))


def _single_serve_package(product: IndexedProduct) -> Optional[float]:
    """The package one person eats as "하나": the smallest package holding at least ~one label
    serving (아침햇살 50|180 ml, 200 ml serving -> 180; 포스틱 84|156 g, 30 g serving -> 84).
    A 50 ml sample or a multi-pack is not "하나"."""
    serving = product.serving_g
    # 0 and an exact 100 g are placeholders in the MFDS label data, not a real label serving
    # (누가바 serving "100" picked its 160 g multi-pack over the 63 g bar)
    if serving <= 0 or serving == _PLACEHOLDER_SERVING_G or not product.package_sizes:
        return None
    fits = [s for s in product.package_sizes if 0.8 * serving <= s <= 4.0 * serving]
    return min(fits) if fits else None


def _indexed_grams(product: IndexedProduct, food_name: str, qty: float, unit: str) -> Optional[float]:
    if unit in GRAM_UNITS:
        return qty
    if unit in _CONTAINER_UNITS:
        sizes = product.package_sizes or (product.unit_g,)
        if unit == "캔":
            # Korean cans are 355 / 250 ml; a same-named 70 g candy or 1.5 L bottle is not "한 캔".
            drink_default = 250.0 if product.category_lv3 == "음료류" else product.unit_g
            size = next((s for s in (355.0, 250.0) if s in sizes), None) or min(
                (s for s in sizes if 150 <= s <= 500), key=lambda s: abs(s - 300), default=drink_default)
        elif unit == "병":
            size = 500.0 if 500.0 in sizes else min(
                (s for s in sizes if 100 <= s <= 600), key=lambda s: abs(s - 500), default=product.unit_g)
        else:
            size = min(sizes)
            # a ramen 4/5-pack (440-600 g) is not "한 봉": cap by the bag/cup reference serving
            if product.category_lv3 == "면류" and size > 2.5 * product.unit_g:
                size = product.unit_g
        return qty * size
    if unit in _GLASS_UNITS and product.category_lv3 in _DRINK_CATEGORIES:
        # "쏘굿 아몬드 언스위트 1잔": one serving of the bottled drink (its single-serve package)
        return qty * (_single_serve_package(product) or product.unit_g)
    if unit in _PIECE_UNITS:
        # 교자 / 만두 / 초밥 have a known piece weight smaller than the package; "라면 1개 = 600 g"
        # is the cooked generic dish and must not scale a dry-weight product row.
        piece = food_unit_overrides(food_name).get("개")
        # drinks are never counted in pieces: "덴마크 요구르트 딸기 하나" is a bottle, not a strawberry
        is_drink = product.category_lv3 in ("음료류", "유가공품류", "주류")
        use_piece = piece is not None and piece < product.unit_g and not is_drink
        if use_piece:
            return qty * piece
        # "포스틱 하나" is the bag; "칙촉 3개" counts the cookies inside the box
        package = _single_serve_package(product) if qty <= 1 else None
        return qty * (package or product.unit_g)
    return None


def find_product(food_name: Optional[str]) -> Optional[Product]:
    key = _compact(food_name or "")
    if not key:
        return None
    index = _index()
    if key in index:
        return index[key]
    best = ""
    for alias in index:
        # Korean names are head-final: "편의점바나나우유" is 바나나우유, "신라면컵" is not 신라면.
        if len(alias) >= 3 and key.endswith(alias) and len(alias) > len(best):
            best = alias
    return index.get(best) if best else None


def product_nutrition(food_name: Optional[str], amount, unit: Optional[str],
                      allow_single_line: bool = True) -> Optional[dict]:
    """Label nutrition for the eaten amount, or None when the name is not a curated product."""
    u = (unit or "").strip().lower()
    try:
        qty = float(amount) if amount is not None else 1.0
    except (TypeError, ValueError):
        qty = 1.0
    product = find_product(food_name)
    if product is None:
        indexed = find_processed_product(food_name, allow_single_line=allow_single_line)
        if indexed is None:
            return None
        if indexed.unit_g >= _BULK_PACK_G and u not in GRAM_UNITS:
            return None  # a 5 kg catering pack is not "하나"; let the generic DB row answer
        grams = _indexed_grams(indexed, food_name or "", qty, u)
        if grams is None:
            return None
        f = grams / 100.0
        return {
            "protein_g": round(indexed.protein_100 * f, 1),
            "carbs_g": round(indexed.carbs_100 * f, 1),
            "fat_g": round(indexed.fat_100 * f, 1),
            "kcal": round(indexed.kcal_100 * f, 1),
            "matched_product": indexed.name,
        }
    if u in GRAM_UNITS:
        factor = qty / product.unit_grams
    elif u in _PACKAGE_UNITS:
        factor = qty
    else:
        return None  # an unexpected unit (e.g. "조각" of a drink): leave it to the DB path
    return {
        "protein_g": round(product.protein_g * factor, 1),
        "carbs_g": round(product.carbs_g * factor, 1),
        "fat_g": round(product.fat_g * factor, 1),
        "kcal": round(product.kcal * factor, 1),
        "matched_product": product.name,
    }


_FAMILY_MIN_ROWS = 3
_FAMILY_MAX_UNIT_G = 1000.0


@lru_cache(maxsize=512)
def _family_rows(key: str) -> tuple[IndexedProduct, ...]:
    """Processed-index rows that name this food: the exact key, else the product line (keys
    starting / ending with it), else every key containing it (at least three rows)."""
    index = _processed_index()
    if not index or len(key) < 2:
        return ()
    if key in index:
        return (index[key],)
    line = [p for k, p in index.items()
            if (k.startswith(key) and len(k) - len(key) <= _MAX_VARIANT_SUFFIX)
            or (k.endswith(key) and len(k) - len(key) <= _MAX_BRAND_PREFIX)]
    if len(line) >= 2:
        return tuple(line)
    if len(key) >= 3:
        contained = [p for k, p in index.items() if key in k]
        if len(contained) >= _FAMILY_MIN_ROWS:
            return tuple(contained)
    return ()


def _median(values: list[float]) -> float:
    values = sorted(values)
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2.0


def product_family_density(food_name: Optional[str]) -> Optional[dict]:
    """Median per-100 g label values (and median unit weight) of the processed-index rows that
    name `food_name` (exact key, product line, or keys containing it). Used only when the LLM's
    own estimate is implausible; None when the index is absent or nothing names the food."""
    key = _norm_key(food_name or "")
    if key:
        rows = _family_rows(key)
        if not rows:
            return None
        kcals = [p.kcal_100 for p in rows]
        median_kcal = _median(kcals)
        # interquartile window: drop powders / dried mixes far from the line's typical density
        kept = [p for p in rows if 0.5 * median_kcal <= p.kcal_100 <= 2.0 * median_kcal] or list(rows)
        units = [p.unit_g for p in kept if 0 < p.unit_g <= _FAMILY_MAX_UNIT_G]
        return {
            "kcal_100": _median([p.kcal_100 for p in kept]),
            "protein_100": _median([p.protein_100 for p in kept]),
            "carbs_100": _median([p.carbs_100 for p in kept]),
            "fat_100": _median([p.fat_100 for p in kept]),
            "unit_g": _median(units) if units else None,
            "rows": len(kept),
            "key": key,
        }
    return None

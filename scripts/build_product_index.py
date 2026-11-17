"""Build the packaged-product nutrition index from the MFDS processed-food dataset.

Input : data/food_db/raw/식품의약품안전처_통합식품영양성분정보(가공식품)_*.csv
        (data.go.kr pk 15100066; fetched with the dataset page's own JSON endpoint)
Output: data/food_db/processed_product_index.csv (generated, not committed, like food_db_clean.csv)

One row per normalised product name: per-package kcal and macros computed from the label basis
(per 100 g / 100 ml) x package weight (FOOD_SIZE). When several manufacturers list the same name,
the median-kcal row is kept and the count is recorded.

  python scripts/build_product_index.py
"""
from __future__ import annotations

import csv
import glob
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_GLOB = str(ROOT / "data" / "food_db" / "raw" / "식품의약품안전처_통합식품영양성분정보(가공식품)_*.csv")
OUT = ROOT / "data" / "food_db" / "processed_product_index.csv"

_SIZE_RE = re.compile(r"^\s*([\d.]+)\s*(kg|g|ml|mL|l|L)\b")
_BRACKETS_RE = re.compile(r"[\(\[\{（【].*?[\)\]\}）】]")


def normalise_name(name: str) -> str:
    name = _BRACKETS_RE.sub(" ", name or "")
    name = re.sub(r"[^0-9A-Za-z가-힣]", "", name)
    return name.lower()


def grams(size: str) -> float | None:
    m = _SIZE_RE.match(size or "")
    if not m:
        return None
    value, unit = float(m.group(1)), m.group(2).lower()
    if unit in ("kg", "l"):
        value *= 1000
    return value if 0 < value <= 5000 else None


def basis_grams(basis: str) -> float | None:
    return grams(basis)


def maker_root(maker: str) -> str:
    """'(주)오리온제4청주공장' / '㈜오리온 제4청주공장' -> '오리온' (company, not plant)."""
    m = re.sub(r"\(주\)|㈜|주식회사|\(유\)|농업회사법인|영농조합법인", " ", maker or "")
    m = re.split(r"\s|제\d|공장|\(", m.strip())[0] if m.strip() else ""
    return re.sub(r"[^0-9A-Za-z가-힣]", "", m).lower()


def num(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


_SERVING_RE = re.compile(r"^\s*([\d.]+)\s*(g|ml|mL)(?![A-Za-z])")


_RAMEN_SERVING_RE = {
    "bag": re.compile(r"유탕면\(봉지\)\s*([\d.]+)\s*g"),
    "cup": re.compile(r"유탕면\(용기\)\s*([\d.]+)\s*g"),
}
_CUP_WORDS = ("컵", "사발", "용기", "큰컵", "볼")


def serving_grams(text: str, name: str = "") -> float | None:
    """'30g', '200ml', '250ml(g)' -> grams. The noodle category lists several references in one
    text ('… 유탕면(봉지)120g, 유탕면(용기)80g'): pick bag or cup by the product name."""
    if not text:
        return None
    if "," in text:
        kind = "cup" if any(w in (name or "") for w in _CUP_WORDS) else "bag"
        m = _RAMEN_SERVING_RE[kind].search(text)
        return float(m.group(1)) if m else None
    m = _SERVING_RE.match(text)
    return float(m.group(1)) if m else None


def main() -> int:
    files = sorted(glob.glob(RAW_GLOB))
    if not files:
        print("no raw processed-food CSV found:", RAW_GLOB)
        return 1
    src = files[-1]
    groups: dict[str, list[dict]] = defaultdict(list)
    seen = kept = dropped = 0
    with open(src, encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            seen += 1
            basis = grams(row.get("NUT_CON_SRTR_QUA", ""))
            kcal = num(row.get("ENERC"))
            key = normalise_name(row.get("FOOD_NM", ""))
            if not basis or kcal is None or len(key) < 2:
                continue
            per100 = 100.0 / basis
            kcal100 = kcal * per100
            is_drink = "음료" in (row.get("FOOD_LV3_NM") or "")
            # Entry errors seen in the data: per-package energy typed into the per-100ml field
            # (칠성사이다 215, 코카-콜라 158 kcal/100ml). Real drinks stay well under 120.
            if kcal100 > 900 or (is_drink and kcal100 > 120):
                dropped += 1
                continue
            groups[key].append({
                "name": (row.get("FOOD_NM") or "").strip(),
                "maker": (row.get("MFR_NM") or "").strip(),
                "category": (row.get("FOOD_LV4_NM") or "").strip(),
                "kcal_100": kcal100,
                "protein_100": (num(row.get("PROT")) or 0) * per100,
                "carbs_100": (num(row.get("CHOCDF")) or 0) * per100,
                "fat_100": (num(row.get("FATCE")) or 0) * per100,
                "package_g": grams(row.get("FOOD_SIZE", "")),
                "serving_g": serving_grams(row.get("SERV_SIZE", ""), row.get("FOOD_NM", "")),
                "category_lv3": (row.get("FOOD_LV3_NM") or "").strip(),
                "food_cd": row.get("FOOD_CD", ""),
            })
            kept += 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ["key", "name", "maker", "maker_root", "category", "category_lv3", "kcal_100", "protein_100", "carbs_100", "fat_100",
              "unit_g", "package_sizes", "serving_g", "food_cd", "n_rows"]
    written = 0
    with open(OUT, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, rows in groups.items():
            med = {f: statistics.median(r[f] for r in rows) for f in ("kcal_100", "protein_100", "carbs_100", "fat_100")}
            rep = min(rows, key=lambda r: abs(r["kcal_100"] - med["kcal_100"]))
            sizes = sorted({r["package_g"] for r in rows if r["package_g"]})
            servings = sorted(r["serving_g"] for r in rows if r["serving_g"])
            serving = servings[len(servings) // 2] if servings else None
            unit = sizes[0] if sizes else serving
            # FOOD_SIZE is often a multipack / box total (안성탕면 625 g = 5 bags, 오예스 360 g box);
            # the smallest listed package is the single unit unless even that is a box of servings.
            # 2.5x: a 4-bag ramen pack (440 g vs 110 g reference) or a 12-piece box is not one unit.
            if unit and serving and unit > 2.5 * serving:
                unit = serving
            if not unit:
                continue
            writer.writerow({
                "key": key, "name": rep["name"], "maker": rep["maker"], "maker_root": maker_root(rep["maker"]),
                "category": rep["category"], "category_lv3": rep["category_lv3"],
                **{f: round(v, 2) for f, v in med.items()},
                "unit_g": unit, "package_sizes": "|".join(f"{s:g}" for s in sizes),
                "serving_g": serving or "", "food_cd": rep["food_cd"], "n_rows": len(rows),
            })
            written += 1
    print(f"source={Path(src).name} rows={seen} usable={kept} dropped_entry_errors={dropped} products={written} -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

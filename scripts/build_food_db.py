#!/usr/bin/env python3
"""food_db_clean.csv → ChromaDB 인덱스 오프라인 빌더.

배치 위치: scripts/build_food_db.py

사용법:
    python scripts/build_food_db.py               # 인덱스가 없을 때만 빌드
    python scripts/build_food_db.py --rebuild     # 기존 인덱스 삭제 후 재빌드
    python scripts/build_food_db.py --limit 500   # 상위 N개만 (빠른 스모크 테스트)

특징:
- 서버 시작과 무관한 1회성 배치 작업 (서버 부팅 시 빌드 금지 요구사항 준수)
- 임베딩 모델 / 컬렉션 설정을 engines.food.food_engine 에서 import 해서
  빌드-검색 간 설정 불일치를 원천 차단
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

# scripts/ 하위에서 실행되므로 프로젝트 루트를 import path 에 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 엔진과 '똑같은' 상수/임베딩 로더를 공유 (모델·컬렉션·경로 드리프트 방지)
from engines.food.food_engine import (  # noqa: E402
    EMBEDDING_MODEL_ID,
    FOOD_COLLECTION_NAME,
    FOOD_DB_DIR,
    build_embeddings,
    resolve_device,
)
from engines.food.food_embedding_text import build_food_embedding_text  # noqa: E402

CSV_PATH = PROJECT_ROOT / "data" / "food_db" / "food_db_clean.csv"
DEFAULT_BATCH_SIZE = 256

# CSV → 메타데이터로 보존할 숫자 컬럼 (검색 후 스케일링에 사용)
_NUMERIC_FIELDS = (
    "basis_value", "kcal", "carb_g", "protein_g", "fat_g",
    "sugar_g", "fiber_g", "sodium_mg", "satfat_g",
)
# CSV → 메타데이터로 보존할 문자열 컬럼
_TEXT_FIELDS = (
    "food_id", "name", "rep_name", "basis_unit",
    "major_category", "data_type",
)


def _to_float(value):
    """빈 문자열/None/파싱불가 → None.

    ChromaDB 메타데이터는 None 을 허용하지 않으므로, None 인 키는 호출측에서 제외한다.
    """
    if value is None:
        return None
    s = str(value).strip().replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_rows(csv_path: Path, limit: int | None = None):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    rows = []
    # UTF-8 with BOM 처리
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def build_documents(rows):
    """(texts, metadatas, ids, skipped) 반환.

    임베딩 대상 텍스트는 build_food_embedding_text(row)로 생성한다.
    영양 수치는 전부 metadata 에 실어서 검색 후 별도 조회 없이 바로 사용.
    """
    texts, metadatas, ids = [], [], []
    seen_ids = set()
    skipped = 0

    for row in rows:
        embed_text = build_food_embedding_text(row).strip()
        food_id = (row.get("food_id") or "").strip()
        if not embed_text or not food_id:
            skipped += 1
            continue

        # id 중복 방지 (혹시 CSV 에 중복 food_id 가 있으면 suffix)
        uid = food_id
        if uid in seen_ids:
            uid = f"{food_id}__{len(ids)}"
        seen_ids.add(uid)

        meta: dict = {}
        for field in _TEXT_FIELDS:
            val = (row.get(field) or "").strip()
            if val:
                meta[field] = val
        for field in _NUMERIC_FIELDS:
            fval = _to_float(row.get(field))
            if fval is not None:
                meta[field] = fval

        # basis 기본값 보정 (검색측 0-division 방지)
        meta.setdefault("basis_value", 100.0)
        meta.setdefault("basis_unit", "g")

        texts.append(embed_text)
        metadatas.append(meta)
        ids.append(uid)

    return texts, metadatas, ids, skipped


def _open_vector_store(embeddings):
    try:
        from langchain_chroma import Chroma
    except Exception:
        from langchain_community.vectorstores import Chroma
    return Chroma(
        collection_name=FOOD_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(FOOD_DB_DIR),
        # 정규화된 BGE 임베딩 → 코사인 공간 (검색측 sim 계산과 반드시 일치)
        collection_metadata={"hnsw:space": "cosine"},
    )


def main():
    parser = argparse.ArgumentParser(description="Build food ChromaDB index")
    parser.add_argument("--rebuild", action="store_true",
                        help="기존 인덱스 삭제 후 재빌드")
    parser.add_argument("--limit", type=int, default=None,
                        help="상위 N개만 처리 (스모크 테스트)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    print(f"[1/5] CSV 로드: {CSV_PATH}")
    rows = load_rows(CSV_PATH, limit=args.limit)
    print(f"      → {len(rows):,} rows")

    print("[2/5] 문서 구성")
    texts, metadatas, ids, skipped = build_documents(rows)
    print(f"      → 유효 {len(texts):,}건 / 스킵 {skipped:,}건")
    if not texts:
        print("!! 임베딩할 문서가 없습니다. CSV 컬럼(embed_text/food_id)을 확인하세요.")
        sys.exit(1)

    # 인덱스 디렉토리 준비
    if FOOD_DB_DIR.exists() and any(FOOD_DB_DIR.iterdir()):
        if args.rebuild:
            print(f"[3/5] 기존 인덱스 삭제: {FOOD_DB_DIR}")
            shutil.rmtree(FOOD_DB_DIR)
        else:
            print(f"!! 인덱스가 이미 존재합니다: {FOOD_DB_DIR}")
            print("   재빌드하려면 --rebuild 옵션을 사용하세요.")
            sys.exit(1)
    FOOD_DB_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[4/5] 임베딩 모델 로드: {EMBEDDING_MODEL_ID} (device={resolve_device()})")
    embeddings = build_embeddings()  # 엔진과 동일 로더 (normalize=True)
    vector_store = _open_vector_store(embeddings)

    print(f"[5/5] 인덱싱 시작 (batch={args.batch_size})")
    total = len(texts)
    start = time.time()

    use_tqdm = False
    try:
        from tqdm import tqdm
        batches = tqdm(range(0, total, args.batch_size), unit="batch")
        use_tqdm = True
    except Exception:
        batches = range(0, total, args.batch_size)

    for i in batches:
        j = min(i + args.batch_size, total)
        vector_store.add_texts(
            texts=texts[i:j],
            metadatas=metadatas[i:j],
            ids=ids[i:j],
        )
        if not use_tqdm:
            pct = j / total * 100
            print(f"      {j:,}/{total:,} ({pct:5.1f}%)", flush=True)

    elapsed = time.time() - start
    try:
        count = vector_store._collection.count()
    except Exception:
        count = "unknown"

    print("-" * 48)
    print(f"완료: {count} documents  |  {elapsed:.1f}s")
    print(f"저장 위치: {FOOD_DB_DIR}")
    print(
        "스모크 테스트:\n"
        "  python -c \"from engines.food.food_engine import get_food_engine; "
        "print(get_food_engine().search('닭가슴살', 200, 'g'))\""
    )


if __name__ == "__main__":
    main()

# Food Chroma Rebuild Operational Plan

This runbook describes how to rebuild the food Chroma index after the
embedding text enrichment generator has been connected to the build pipeline.

This document is a plan only. Do not run the actual rebuild until an approved
operational rebuild window exists.

## Purpose

The runtime food search layer already includes query normalization, query
analysis, multi-query retrieval, deterministic reranking, runtime canonical
exact lookup, and the `fetch_k` / `final_k` split. The Chroma rebuild is a
separate operational step that updates vector embeddings so enriched generated
text can improve embedding recall.

The rebuild uses `build_food_embedding_text(row)` through
`scripts/build_food_db.py`. It must not change API response shape, nutrition
metadata, or runtime search behavior.

## Preconditions

- Current commit includes `55bd7f5` or later.
- `git status` is clean.
- Full pytest is green.
- Food test subset is green.
- `scripts/build_food_db.py` imports and uses
  `engines.food.food_embedding_text.build_food_embedding_text`.
- Existing Chroma data may still have been built from the old `embed_text` or
  `name` text until this rebuild is explicitly executed.
- Rebuild has explicit operational approval.

Recommended verification before the approved rebuild window:

```powershell
git status
git rev-parse --short HEAD
python -m pytest tests/test_food_embedding_build_pipeline.py -q
python -m pytest tests/test_food_embedding_text.py -q
python -m pytest tests -q -k "food_query or food_search or food_multi_query or food_reranker or food_canonical_index or food_fetch or food_embedding_text or food_embedding_build"
python -m pytest -q
```

## Target Files And Paths

- Input CSV: `data/food_db/food_db_clean.csv`
- Build script: `scripts/build_food_db.py`
- Embedding text generator: `engines/food/food_embedding_text.py`
- Chroma persist path: `data/food_chroma_db`
- Chroma collection name: `food_db`
- Embedding model default: `dragonkue/BGE-m3-ko`
- Embedding model override: `FOOD_EMBEDDING_MODEL`
- Local model path override: `FOOD_EMBEDDING_PATH`
- Local-only model loading flag: `FOOD_EMBEDDING_LOCAL_ONLY`

## Rebuild Before Checklist

- Confirm `git status` is clean.
- Confirm the deployed commit includes `55bd7f5` or later.
- Confirm full pytest and the food subset pass.
- Confirm disk space is enough for a full timestamped backup and a new index.
- Confirm `data/food_chroma_db` exists and identify its current size.
- Create a timestamped backup before any rebuild command.
- Prepare rollback commands before touching the current persist directory.
- Confirm whether the AI server must be stopped during rebuild.
- Confirm service restart procedure after rebuild.
- Confirm model cache availability for the embedding model.
- Confirm the validation smoke owner and acceptance criteria.

Never delete or overwrite `data/food_chroma_db` without a verified backup.

## Backup Procedure

Copy backup is preferred over move backup when disk space allows. It keeps the
current index available until the rebuild command starts. If disk space is
limited, use an approved maintenance window and a move-based backup.

PowerShell example to run during the approved rebuild window:

```powershell
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$src = "data\food_chroma_db"
$backup = "data\food_chroma_db.backup.$stamp"
Copy-Item -Recurse -Force $src $backup
Write-Host "Backup created: $backup"
```

Bash example to run during the approved rebuild window:

```bash
stamp="$(date +%Y%m%d-%H%M%S)"
src="data/food_chroma_db"
backup="data/food_chroma_db.backup.${stamp}"
cp -a "$src" "$backup"
echo "Backup created: $backup"
```

After backup, verify the backup directory exists and is non-empty before
running any rebuild command.

## Rebuild Command

The actual rebuild command must be run only during an approved rebuild window.
This PR does not execute it.

PowerShell:

```powershell
python scripts\build_food_db.py --rebuild
```

Bash:

```bash
python scripts/build_food_db.py --rebuild
```

Optional smoke-size build for a non-production scratch environment only:

```powershell
python scripts\build_food_db.py --rebuild --limit 500
```

Do not use `--limit` for production because it builds an incomplete index.

Environment notes:

- Default local-only loading is controlled by `FOOD_EMBEDDING_LOCAL_ONLY`.
- If the model is not cached and network access is approved, set the required
  environment explicitly according to the deployment environment.
- Keep the same embedding model as runtime search unless a separate model
  migration has been approved.

## Validation Smoke

After rebuild and service restart, run the opt-in Chroma smoke tests:

```powershell
$env:RUN_FOOD_SEARCH_SMOKE='1'
python -m pytest tests\test_food_search_golden_queries.py -q -m "chroma and smoke"
```

Bash:

```bash
RUN_FOOD_SEARCH_SMOKE=1 python -m pytest tests/test_food_search_golden_queries.py -q -m "chroma and smoke"
```

Manual validation should also check these queries through the current
`FoodSearchEngine.search()` path:

| Query | Expected behavior |
| --- | --- |
| `계란` | top result is `달걀 생것` or canonical raw egg |
| `삶은 계란` | top result is `달걀 삶은것` |
| `계란후라이` | top result is `달걀후라이` |
| `계란빵` | preserved as `계란빵`, not raw egg |
| `닭가슴살 생것` | top result is `닭고기 가슴(껍질 제거) 생것` |
| `삶은 닭가슴살` | top result is `닭고기 가슴(껍질 제거) 삶은것` |
| `구운 닭가슴살` | top result is `닭고기 가슴(껍질 제거) 구운것(팬)` |
| `샐러드 닭가슴살` | preserved as `샐러드 닭가슴살`, not raw chicken breast |
| `달걀 삶은것` | top result is `달걀 삶은것` |
| `닭 가슴살 생것` | top result is `닭고기 가슴(껍질 제거) 생것` |
| `닭가슴살 100g` | remains protected/ambiguous, not forced to raw chicken |
| `볶음밥 계란` | protected dish query remains dish-oriented |
| `김밥 계란` | protected dish query remains dish-oriented |
| `샌드위치 닭가슴살` | protected dish query remains dish-oriented |

Record query, normalized query, final top result, and any unexpected protected
query regression.

## Rollback Procedure

If validation fails:

1. Stop or isolate the AI server if it is reading the new index.
2. Preserve the failed new `data/food_chroma_db` directory for inspection if
   disk space allows.
3. Restore the timestamped backup to `data/food_chroma_db`.
4. Restart the AI server if required.
5. Re-run the validation smoke.
6. Attach rebuild logs, failed smoke output, and the backup path to the incident
   or follow-up issue.

PowerShell rollback example:

```powershell
$failed = "data\food_chroma_db.failed.$(Get-Date -Format 'yyyyMMdd-HHmmss')"
Move-Item "data\food_chroma_db" $failed
Copy-Item -Recurse -Force "data\food_chroma_db.backup.YYYYMMDD-HHMMSS" "data\food_chroma_db"
```

Bash rollback example:

```bash
failed="data/food_chroma_db.failed.$(date +%Y%m%d-%H%M%S)"
mv data/food_chroma_db "$failed"
cp -a data/food_chroma_db.backup.YYYYMMDD-HHMMSS data/food_chroma_db
```

Replace `YYYYMMDD-HHMMSS` with the actual verified backup timestamp.

## Deployment Sequence

1. Deploy code that includes the enriched text generator and build pipeline
   connection.
2. Confirm tests and smoke pass before the rebuild window.
3. During the approved window, create a timestamped backup.
4. Run the rebuild command.
5. Restart or reload the AI server if needed.
6. Run Chroma smoke tests and manual golden query checks.
7. Monitor error logs, latency, and food search quality.
8. Keep the backup through the rollback window.
9. Remove old backups only after operational approval.

## Stop Criteria

Stop the rollout and use rollback if any of these occur:

- Full pytest fails before rebuild.
- Food subset or Chroma smoke fails.
- Protected query regression appears, especially `계란빵`, `볶음밥 계란`,
  `김밥 계란`, `샐러드 닭가슴살`, or `샌드위치 닭가슴살`.
- Chroma fails to load after rebuild.
- Embedding model or cache loading fails.
- Latency exceeds the accepted operational threshold.
- API response contract or source metadata shape changes unexpectedly.
- Nutrition/business metadata is missing or type-shifted.

## What This Plan Does Not Do

- It does not execute the rebuild.
- It does not modify `food_db_clean.csv`.
- It does not modify the Chroma persist directory.
- It does not change the embedding model.
- It does not change runtime search behavior.
- It does not add broad aliases, fuzzy matching, or BM25.

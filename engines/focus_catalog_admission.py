"""Identity-bound product admission, separate from catalog and medical review."""
from __future__ import annotations

import csv
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path

REVIEW_PATH = Path(__file__).resolve().parents[1] / 'data/provenance/focus-basic-admission-review-v2-2026-10-01.json'
REVIEW_SHA256 = '21d2df5dd373ab2f1af0724d0be592ec8d21f99bf868d5c12f6b68f31d75c05f'
SOURCE_SHA256 = '22e7b98199b3ee8b1d4e880e88fe76a5ad7a8dc14430c7f94ed0143f4f234ff2'
REVIEW_VERSION = 'focus-basic-admission-v2-2026-10-01'
REVIEWED_IDS = frozenset({'V24ROW-006682', 'V24ROW-006024', 'V24ROW-006121',
                         'V24ROW-005299', 'V24ROW-005319'})


@lru_cache(maxsize=4)
def _load(review_path: str, review_stat: tuple, source_path: str, source_stat: tuple) -> dict:
    try:
        raw = Path(review_path).read_bytes()
        source = Path(source_path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != REVIEW_SHA256 or hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
            return {}
        payload = json.loads(raw)
        if (payload.get('schemaVersion') != 'routine-focus-admission-v1'
                or payload.get('decisionVersion') != REVIEW_VERSION
                or payload.get('sourceCatalogSha256') != SOURCE_SHA256):
            return {}
        identities = {r['v24_row_uid']: r for r in csv.DictReader(io.StringIO(source.decode('utf-8-sig')))
                      if r.get('v24_row_uid') in REVIEWED_IDS}
        reviews = {}
        for review in payload['reviews']:
            key = review['catalogExerciseId']
            if key in reviews or key not in identities or any(
                    identities[key].get(k) != v for k, v in review['sourceIdentity'].items()):
                return {}
            reviews[key] = review
        return reviews if reviews.keys() == REVIEWED_IDS else {}
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, csv.Error):
        return {}


def admission_patch(exercise_id: str, catalog_identity: dict, effective: dict, source_path: Path) -> dict:
    if exercise_id not in REVIEWED_IDS:
        return {}
    try:
        review_stat, source_stat = REVIEW_PATH.stat(), source_path.stat()
        reviews = _load(str(REVIEW_PATH), (review_stat.st_mtime_ns, review_stat.st_size),
                        str(source_path), (source_stat.st_mtime_ns, source_stat.st_size))
    except OSError:
        return {}
    review = reviews.get(exercise_id)
    if (not review or catalog_identity != review['catalogIdentity']
            or any(effective.get(k) != v for k, v in review['effectiveIdentity'].items())
            or effective.get('beginner_default_allowed') != 0
            or effective.get('main_routine_default_allowed') != 0):
        return {}
    # Only admission and a searchable title change. Risk, clinical restrictions,
    # quality/licence gates and popularity scores are deliberately untouched.
    patch = {
        'catalog_original_main_routine_default_allowed': 0,
        'main_routine_default_allowed': 1,
        'core_main_role_reviewed': 1,
        'core_admission_review_version': REVIEW_VERSION,
        'core_admission_source_sha256': SOURCE_SHA256,
        'name_kr': review['displayNameKo'],
        'user_facing_display_name_kr': review['displayNameKo'],
    }
    if review['scope'] == 'advanced_lower_chest_focus':
        patch.update(
            core_admission_scope='lowerChest',
            core_chest_focus_reviewed=1,
            specialization_focus_codes='lowerChest',
        )
    return patch

"""Purpose-only exceptions backed by the checked-in exact-ID review catalog."""
from functools import lru_cache
import json
import logging
from pathlib import Path

from .option_resolver import ADJUNCT_OPTION_PRIORITY, canonical_adjunct_options, LEGACY_STYLE_TO_ADJUNCT_OPTION

CATALOG_PATH = Path(__file__).resolve().parent / 'data/routine_adjunct_catalog_v1.json'
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def reviewed_purpose_index() -> dict[str, frozenset[str]]:
    try:
        data = json.loads(CATALOG_PATH.read_text(encoding='utf-8'))
        if not isinstance(data, dict) or (
            data.get('catalogVersion') != 'routine-adjunct-catalog-v1'
            or data.get('reviewStatus') != 'core-reviewed'
            or data.get('scope') != 'gym-routine-adjunct-only'
            or not isinstance(data.get('options'), dict)
        ):
            raise ValueError('invalid review catalog header')
        index: dict[str, set[str]] = {}
        for option, entries in data['options'].items():
            if option not in ADJUNCT_OPTION_PRIORITY or not isinstance(entries, list):
                raise ValueError('invalid review catalog option')
            for entry in entries:
                eid = entry.get('exerciseId') if isinstance(entry, dict) else None
                if not isinstance(eid, str) or len(eid) != 13 or not eid.startswith('V24ROW-') or not eid[7:].isdigit():
                    raise ValueError('invalid reviewed exercise ID')
                index.setdefault(entry['exerciseId'].strip().lower(), set()).add(option)
        return {eid: frozenset(options) for eid, options in index.items()}
    except (OSError, ValueError, TypeError):
        logger.warning('reviewed adjunct admission catalog unavailable; exceptions disabled')
        return {}


def admission_purposes(options=(), training_style=None) -> tuple[str, ...]:
    requested = canonical_adjunct_options(options)
    # Keep the existing low-level yoga style mapped to its reviewed purpose.
    if not requested and str(training_style or '').strip().lower() == 'yoga_pilates':
        return (LEGACY_STYLE_TO_ADJUNCT_OPTION['yoga_pilates'],)
    return requested


def reviewed_for_purpose(candidate: dict, options) -> bool:
    eid = str(candidate.get('id') or '').strip().lower()
    catalog_id = str(candidate.get('catalog_exercise_id') or eid).strip().lower()
    if not eid or catalog_id != eid:
        return False
    return bool(reviewed_purpose_index().get(eid, frozenset())
                & set(canonical_adjunct_options(options)))

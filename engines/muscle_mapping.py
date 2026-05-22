"""Muscle target passthrough helpers."""
from typing import Dict, List, Tuple

from .registry import DOMS_LEVEL_MAP, SPLIT_LABEL_TO_MUSCLES
from .schemas import DomEntry


def split_label_to_muscles(label: str) -> List[str]:
    """targetSplitLabel("push" 등)을 spreadsheet slug 리스트로 변환한다."""
    return SPLIT_LABEL_TO_MUSCLES.get(label.lower(), [])


def get_mapped_targets(raw_muscles: List[str]) -> Tuple[List[str], List[str]]:
    """
    FE에서 넘어온 target_muscles slug를 값 변경 없이 보존한다.
    반환: (ai_targets, db_enums) - shim 호환을 위해 같은 리스트를 두 번 반환한다.
    """
    seen: set[str] = set()
    targets: List[str] = []
    for raw in raw_muscles:
        normalized = raw.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        targets.append(normalized)
    return targets, targets


def map_doms_to_db(doms: List[DomEntry]) -> Dict[str, int]:
    """DomEntry 배열을 값 변경 없이 {body_part_slug: level_int} 딕셔너리로 변환한다."""
    result: Dict[str, int] = {}
    for entry in doms:
        level_int = DOMS_LEVEL_MAP.get(entry.level.lower(), 1)
        normalized = entry.body_part.strip()
        if normalized:
            result[normalized] = max(result.get(normalized, 0), level_int)
    return result

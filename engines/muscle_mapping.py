"""Muscle target passthrough helpers."""
from typing import Dict, List, Tuple

from .registry import DOMS_LEVEL_MAP, SPLIT_LABEL_TO_MUSCLES
from .schemas import DomEntry


def split_label_to_muscles(label: str) -> List[str]:
    """targetSplitLabel("push" 등)을 spreadsheet slug 리스트로 변환한다."""
    return SPLIT_LABEL_TO_MUSCLES.get(label.lower(), [])


def resolve_request_target_muscles(
    target_split_label: str | None,
    target_muscles: List[str] | None,
) -> List[str]:
    """Resolve targets while preserving the user's explicit muscle choices."""
    _, explicit_targets = get_mapped_targets(target_muscles or [])
    if explicit_targets:
        return explicit_targets
    label = str(target_split_label or "").strip().lower()
    return split_label_to_muscles(label) if label else []


def infer_target_split_label(target_muscles: List[str] | None) -> str:
    """Infer the narrowest supported split from explicit target muscles."""
    _, normalized_targets = get_mapped_targets(target_muscles or [])
    targets = set(normalized_targets)
    if not targets:
        return "custom"
    if targets == {"chest"}:
        return "chest"
    if targets <= {"abs", "obliques", "lower-back"} and targets & {"abs", "obliques"}:
        return "core"
    if targets <= {"front-deltoids", "side-deltoids", "back-deltoids", "trapezius"}:
        return "shoulder"
    if targets <= {"biceps", "triceps", "forearm"}:
        return "arm"
    if targets <= {"upper-back", "trapezius", "lower-back", "back-deltoids"} and "upper-back" in targets:
        return "back"
    if "chest" in targets and "upper-back" in targets:
        return "chest_back"
    if "chest" in targets and targets & {"triceps", "front-deltoids", "side-deltoids"}:
        return "push"
    if "upper-back" in targets and targets & {"biceps", "trapezius", "back-deltoids"}:
        return "pull"
    if {"quadriceps", "hamstring"} <= targets or (
        "quadriceps" in targets and "gluteal" in targets
    ):
        return "legs"
    return "custom"


def resolve_request_target_split_label(
    target_split_label: str | None,
    target_muscles: List[str] | None,
) -> str:
    """Reconcile a preset label with an explicitly edited muscle selection.

    FE presets send both fields. Once the user removes a preset muscle, the
    explicit selection becomes authoritative and the effective split is
    narrowed instead of silently restoring the entire preset.
    """
    label = str(target_split_label or "").strip().lower()
    _, explicit_targets = get_mapped_targets(target_muscles or [])
    if not explicit_targets:
        return label or "custom"

    targets = set(explicit_targets)
    preset_targets = set(split_label_to_muscles(label))
    if preset_targets and targets <= preset_targets:
        required_anchors = {
            "push": ({"chest"}, {"triceps", "front-deltoids", "side-deltoids"}),
            "pull": ({"upper-back"}, {"biceps", "trapezius", "back-deltoids"}),
            "legs": ({"quadriceps"}, {"hamstring", "gluteal"}),
            "lower": ({"quadriceps"}, {"hamstring", "gluteal"}),
            "upper": ({"chest"}, {"upper-back"}),
            "chest_back": ({"chest"}, {"upper-back"}),
            "full_body": ({"chest", "upper-back"}, {"quadriceps", "hamstring", "gluteal"}),
        }
        anchor_groups = required_anchors.get(label)
        if anchor_groups and all(targets & group for group in anchor_groups):
            return label
        if label in {"chest", "back", "shoulder", "arm", "core", "neck"}:
            return infer_target_split_label(explicit_targets)

    return infer_target_split_label(explicit_targets)


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
    """Map legacy DOMS labels to active 1..3 levels; this is not injury data."""
    result: Dict[str, int] = {}
    for entry in doms:
        level_int = DOMS_LEVEL_MAP.get(entry.level.lower(), 1)
        normalized = entry.body_part.strip()
        if normalized:
            result[normalized] = max(result.get(normalized, 0), level_int)
    return result

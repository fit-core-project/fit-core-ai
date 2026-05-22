# 주의: 이 파일은 BE의 MuscleMapper.java와 수동 동기화 중.
# 향후 공통 YAML/JSON config로 이관 예정. 변경 시 BE 코드와 동기화 필수.
from dataclasses import dataclass
from typing import Dict, List, Set

DOMS_LEVEL_MAP: Dict[str, int] = {
    "mild": 1,
    "moderate": 2,
    "severe": 3,
}

# scripts/exercise_tier.xlsx primary_muscle 기준 slug SSOT
MUSCLE_SLUGS: Set[str] = {
    "abductors",
    "abs",
    "adductor",
    "back-deltoids",
    "biceps",
    "calves",
    "chest",
    "forearm",
    "front-deltoids",
    "gluteal",
    "hamstring",
    "lower-back",
    "neck",
    "obliques",
    "quadriceps",
    "trapezius",
    "triceps",
    "upper-back",
}

LARGE_MUSCLE_SLUGS: Set[str] = {
    "chest",
    "upper-back",
    "lower-back",
    "quadriceps",
    "hamstring",
    "gluteal",
}

ACCESSORY_MUSCLE_SLUGS: Set[str] = {
    "abductors",
    "abs",
    "adductor",
    "back-deltoids",
    "biceps",
    "calves",
    "forearm",
    "front-deltoids",
    "neck",
    "obliques",
    "trapezius",
    "triceps",
}

ACCESSORY_PRIMARY_SPLITS: Set[str] = {"arm", "core", "shoulder", "neck"}

LOADED_EQUIPMENT_TOKENS: Set[str] = {
    "BARBELL",
    "DUMBBELL",
    "MACHINE",
    "CABLE",
    "SMITH_MACHINE",
    "KETTLEBELL",
    "PLATE",
    "LANDMINE",
}

SPLIT_LABEL_TO_MUSCLES: Dict[str, List[str]] = {
    "push":      ["chest", "front-deltoids", "triceps"],
    "pull":      ["upper-back", "trapezius", "biceps", "forearm", "back-deltoids"],
    "legs":      ["quadriceps", "hamstring", "gluteal", "calves", "adductor", "abductors"],
    "upper":     ["chest", "upper-back", "trapezius", "front-deltoids", "back-deltoids", "biceps", "triceps"],
    "lower":     ["quadriceps", "hamstring", "gluteal", "calves", "adductor", "abductors"],
    "chest":     ["chest"],
    "back":      ["upper-back", "trapezius", "lower-back"],
    "shoulder":  ["front-deltoids", "back-deltoids", "trapezius"],
    "arm":       ["biceps", "triceps", "forearm"],
    "core":      ["abs", "lower-back", "obliques"],
    "neck":      ["neck"],
    "full_body": ["chest", "upper-back", "front-deltoids", "quadriceps", "hamstring", "gluteal", "abs"],
}


@dataclass
class MuscleMapping:
    ai_targets: List[str]
    db_enums: List[str]


# FE부터 exercise_tier 조회까지 같은 slug를 직렬 사용한다. 별칭 확장 금지.
MUSCLE_REGISTRY: Dict[str, MuscleMapping] = {
    slug: MuscleMapping([slug], [slug])
    for slug in MUSCLE_SLUGS
}

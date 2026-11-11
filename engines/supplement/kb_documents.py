"""Curated supplement KB (profiles / interaction rules / safety rules) as retrieval documents.

The JSON files under data/documents are the source of truth for these three small sets.
The Chroma index is rebuilt rarely, so the engine loads them straight from JSON at startup
and replaces whatever (possibly stale) copies the index holds. scripts/build_embedding_db.py
uses the same formatters so indexed and in-memory text stay identical.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

CURATED_KB_FILES = {
    "ingredient_profile": "supplement_ingredient_profiles.json",
    "interaction_rule": "supplement_interaction_rules.json",
    "safety_rule": "supplement_safety_rules.json",
}


def _norm(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, str):
        s = " ".join(x.split()).strip()
        return s if s else None
    if isinstance(x, (int, float, bool)):
        return str(x)
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def _bullet(items: Any, indent: str = "  - ") -> str:
    if not isinstance(items, list) or not items:
        return ""
    cleaned = [v for v in [_norm(v) for v in items] if v]
    return "\n".join(f"{indent}{v}" for v in cleaned) if cleaned else ""


def _load(payload: Union[str, Dict[str, Any], List[Any]]) -> Dict[str, Any]:
    data = json.loads(payload) if isinstance(payload, str) else payload
    if isinstance(data, list):
        data = data[0] if data else {}
    return data or {}


def format_interaction_rule_text(payload: Union[str, Dict[str, Any], List[Any]]) -> str:
    data = _load(payload)
    entity_a = data.get("entity_a") or {}
    entity_b = data.get("entity_b") or {}

    lines = ["[Interaction rule]"]
    if _norm(data.get("id")):
        lines.append(f"ID: {_norm(data.get('id'))}")
    for label, entity in (("Entity A", entity_a), ("Entity B", entity_b)):
        if isinstance(entity, dict):
            text = " / ".join([v for v in [_norm(entity.get("label")), _norm(entity.get("canonical")), _norm(entity.get("type"))] if v])
            if text:
                lines.append(f"{label}: {text}")
    for key, label in (
        ("interaction_type", "Interaction type"),
        ("mechanism", "Mechanism"),
        ("recommendation", "Recommendation"),
        ("spacing_guidance", "Spacing guidance"),
        ("caution_level", "Caution level"),
    ):
        if _norm(data.get(key)):
            lines.append(f"{label}: {_norm(data.get(key))}")
    if data.get("consultation_required_when"):
        lines.append("Consultation required when:\n" + _bullet(data.get("consultation_required_when")))
    return "\n".join(lines).strip()


def format_safety_rule_text(payload: Union[str, Dict[str, Any], List[Any]]) -> str:
    data = _load(payload)
    lines = ["[Safety rule]"]
    if _norm(data.get("id")):
        lines.append(f"ID: {_norm(data.get('id'))}")
    if data.get("trigger_conditions"):
        lines.append("Trigger conditions:\n" + _bullet(data.get("trigger_conditions")))
    if data.get("affected_entities"):
        lines.append("Affected entities:\n" + _bullet(data.get("affected_entities")))
    for key, label in (
        ("risk", "Risk"),
        ("recommendation", "Recommendation"),
        ("when_to_consult", "When to consult"),
        ("caution_level", "Caution level"),
        ("evidence_summary", "Evidence summary"),
    ):
        if _norm(data.get(key)):
            lines.append(f"{label}: {_norm(data.get(key))}")
    if data.get("source_refs"):
        lines.append("Source refs:\n" + _bullet(data.get("source_refs")))
    return "\n".join(lines).strip()


def format_ingredient_profile_text(payload: Union[str, Dict[str, Any], List[Any]]) -> str:
    data = _load(payload)
    timing = data.get("timing") or {}

    lines = ["[Ingredient profile]"]
    for key, label in (("id", "ID"), ("name", "Name"), ("category", "Category")):
        if _norm(data.get(key)):
            lines.append(f"{label}: {_norm(data.get(key))}")
    if data.get("aliases"):
        lines.append("Aliases:\n" + _bullet(data.get("aliases")))
    if data.get("common_uses"):
        lines.append("Common uses:\n" + _bullet(data.get("common_uses")))
    if isinstance(timing, dict):
        timing_lines = []
        for key, label in (
            ("general", "General"),
            ("with_food", "With food"),
            ("before_bed", "Before bed"),
            ("exercise_related", "Exercise related"),
        ):
            value = _norm(timing.get(key))
            if value:
                timing_lines.append(f"  - {label}: {value}")
        if timing_lines:
            lines.append("Timing:\n" + "\n".join(timing_lines))
    for key, label in (
        ("spacing", "Spacing"),
        ("cautions", "Cautions"),
        ("high_risk_groups", "High risk groups"),
        ("related_interactions", "Related interactions"),
    ):
        if data.get(key):
            lines.append(f"{label}:\n" + _bullet(data.get(key)))
    return "\n".join(lines).strip()


def ingredient_profile_metadata(record: Dict[str, Any], source_file: str) -> Dict[str, Any]:
    return {
        "id": record.get("id"),
        "source": "ingredient_profile",
        "_source_file": record.get("_source_file") or source_file,
        "type": record.get("type"),
        "name": record.get("name"),
        "category": record.get("category"),
        "canonical": (record.get("id") or "").removeprefix("ING_").lower().replace("_", " "),
    }


def interaction_rule_metadata(record: Dict[str, Any], source_file: str) -> Dict[str, Any]:
    entity_a = record.get("entity_a") or {}
    entity_b = record.get("entity_b") or {}
    return {
        "id": record.get("id"),
        "source": "interaction_rule",
        "_source_file": record.get("_source_file") or source_file,
        "type": record.get("type"),
        "entity_a_canonical": entity_a.get("canonical"),
        "entity_b_canonical": entity_b.get("canonical"),
        "interaction_type": record.get("interaction_type"),
        "caution_level": record.get("caution_level"),
    }


def safety_rule_metadata(record: Dict[str, Any], source_file: str) -> Dict[str, Any]:
    affected = record.get("affected_entities") or []
    return {
        "id": record.get("id"),
        "source": "safety_rule",
        "_source_file": record.get("_source_file") or source_file,
        "type": record.get("type"),
        "affected_entities": ", ".join(affected) if isinstance(affected, list) else affected,
        "caution_level": record.get("caution_level"),
    }


_BUILDERS = {
    "ingredient_profile": (format_ingredient_profile_text, ingredient_profile_metadata),
    "interaction_rule": (format_interaction_rule_text, interaction_rule_metadata),
    "safety_rule": (format_safety_rule_text, safety_rule_metadata),
}


def load_curated_kb_records(documents_dir: Path) -> List[tuple[str, Dict[str, Any], str]]:
    """Return (page_content, metadata, source) for every curated record; missing files are skipped."""
    out: List[tuple[str, Dict[str, Any], str]] = []
    for source, filename in CURATED_KB_FILES.items():
        path = Path(documents_dir) / filename
        if not path.exists():
            logger.warning("[supplement kb] curated file missing: %s", path)
            continue
        records = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(records, dict):
            records = next((v for v in records.values() if isinstance(v, list)), [])
        fmt, meta = _BUILDERS[source]
        for record in records:
            if isinstance(record, dict) and record.get("id"):
                out.append((fmt(record), meta(record, filename), source))
    return out


def merge_curated_kb_documents(index_docs: List[Any], documents_dir: Path, document_cls: Any) -> List[Any]:
    """Replace the index's copies of curated KB docs with the current JSON versions."""
    records = load_curated_kb_records(documents_dir)
    if not records:
        return index_docs
    curated_sources = {source for _text, _meta, source in records}
    kept = [
        doc for doc in index_docs
        if ((getattr(doc, "metadata", {}) or {}).get("source")) not in curated_sources
    ]
    curated = [document_cls(page_content=text, metadata=meta) for text, meta, _source in records]
    logger.info(
        "[supplement kb] curated docs from JSON: %s (replaced %s indexed copies)",
        len(curated), len(index_docs) - len(kept),
    )
    return [*kept, *curated]

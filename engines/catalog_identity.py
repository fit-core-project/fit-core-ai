"""Reviewed numeric-to-catalog identity edges; names are evidence, not keys."""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
BRIDGE_PATH = Path(__file__).parent / 'data/catalog_identity_bridge_v1.json'
IDENTITY_FIELDS = ('name_kr', 'name_en', 'primary_muscle', 'equipment_req', 'movement_type')
STATUSES = frozenset({'approved', 'candidate', 'rejected', 'ambiguous'})


def identity_snapshot(row):
    return {key: str(row.get(key) or '').strip() for key in IDENTITY_FIELDS}


def load_identity_edges(path=None):
    reason = 'unreadable_document'
    try:
        document = json.loads((path or BRIDGE_PATH).read_text(encoding='utf-8'))
        reason = 'invalid_document_shape'
        if not isinstance(document, dict):
            raise ValueError
        if document['version'] != 'catalog-identity-bridge-v1-draft':
            reason = 'unsupported_version'
            raise ValueError
        if document['sourceNamespace'] != 'fit-core-backend.exercise_tier':
            reason = 'unsupported_namespace'
            raise ValueError
        edges = document['edges']
        if not isinstance(edges, list):
            reason = 'invalid_edges'
            raise ValueError
        pairs = set()
        for edge in edges:
            reason = 'invalid_edge'
            if not isinstance(edge, dict):
                raise ValueError
            pair = (edge['sourceId'], edge['catalogId'])
            if (not isinstance(pair[0], str) or not pair[0].isdigit()
                    or not isinstance(pair[1], str) or not pair[1].startswith('V24ROW-')
                    or not isinstance(edge['status'], str)
                    or edge['status'] not in STATUSES or pair in pairs):
                raise ValueError
            pairs.add(pair)
            reason = 'invalid_identity_snapshot'
            for key in ('sourceIdentity', 'catalogIdentity'):
                if (not isinstance(edge[key], dict) or set(edge[key]) != set(IDENTITY_FIELDS)
                        or not all(isinstance(v, str) for v in edge[key].values())):
                    raise ValueError
            if edge['status'] == 'approved':
                reason = 'missing_approval_evidence'
                review = edge.get('review')
                if not isinstance(review, dict) or not all(
                    isinstance(review.get(k), str) and review[k].strip()
                    for k in ('reviewedBy', 'reviewedAt', 'rationale')
                ):
                    raise ValueError
        return edges
    except (OSError, ValueError, TypeError, KeyError):
        logger.warning('Catalog identity bridge unavailable: reason=%s; numeric enrichment disabled', reason)
        return []


def resolve_catalog_identity(candidate, by_id, edges):
    source_id = str(candidate.get('id') or '').strip()
    if source_id.startswith('V24ROW-'):
        return by_id.get(source_id.lower())
    if not source_id.isdigit():
        return None
    # 1:N can be recorded, but competing approvals require a future selector.
    approved = [e for e in edges if e['sourceId'] == source_id and e['status'] == 'approved']
    if len(approved) != 1:
        return None
    edge = approved[0]
    if identity_snapshot(candidate) != edge['sourceIdentity']:
        return None
    row = by_id.get(edge['catalogId'].lower())
    if row is None:
        return None
    target = {key: row.get('catalog_' + key) for key in IDENTITY_FIELDS}
    if identity_snapshot(target) != edge['catalogIdentity']:
        return None
    return row

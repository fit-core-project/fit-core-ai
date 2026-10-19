"""Bridge failure is observable without exposing registry content or paths."""
import json
import logging

import pytest

from engines import catalog_identity as bridge


def document():
    return json.loads(bridge.BRIDGE_PATH.read_text(encoding='utf-8'))


@pytest.mark.parametrize('damage,reason', [
    ('missing', 'unreadable_document'), ('json', 'unreadable_document'),
    ('encoding', 'unreadable_document'), ('shape', 'invalid_document_shape'),
    ('missing_key', 'invalid_document_shape'), ('version', 'unsupported_version'),
    ('namespace', 'unsupported_namespace'), ('edges', 'invalid_edges'),
    ('edge', 'invalid_edge'), ('duplicate', 'invalid_edge'), ('status', 'invalid_edge'),
    ('snapshot', 'invalid_identity_snapshot'), ('snapshot_list', 'invalid_identity_snapshot'),
    ('review', 'missing_approval_evidence'),
])
def test_failure_warns_once_and_fails_closed(tmp_path, caplog, damage, reason):
    data = document()
    if damage == 'shape': data = []
    elif damage == 'missing_key': data.pop('version')
    elif damage == 'version': data['version'] = 'secret-marker'
    elif damage == 'namespace': data['sourceNamespace'] = 'secret-marker'
    elif damage == 'edges': data['edges'] = 'secret-marker'
    elif damage == 'edge': data['edges'] = ['secret-marker']
    elif damage == 'duplicate': data['edges'].append(data['edges'][0])
    elif damage == 'status': data['edges'][0]['status'] = []
    elif damage == 'snapshot': data['edges'][0]['sourceIdentity'] = 'secret-marker'
    elif damage == 'snapshot_list': data['edges'][0]['sourceIdentity'] = list(bridge.IDENTITY_FIELDS)
    elif damage == 'review': data['edges'][0].update(status='approved', review=None)
    path = tmp_path/'secret-marker.json'
    if damage != 'missing':
        path.write_bytes(b'\xff' if damage == 'encoding' else (
            b'{secret-marker' if damage == 'json' else json.dumps(data).encode()))
    with caplog.at_level(logging.WARNING, logger=bridge.__name__):
        assert bridge.load_identity_edges(path) == []
    records = [r for r in caplog.records if r.name == bridge.__name__]
    assert len(records) == 1
    assert f'reason={reason};' in records[0].getMessage()
    assert 'secret-marker' not in caplog.text
    assert str(tmp_path) not in caplog.text
    assert not records[0].exc_info


@pytest.mark.parametrize('empty', [False, True])
def test_valid_registry_does_not_warn(tmp_path, caplog, empty):
    data = document()
    if empty: data['edges'] = []
    path = tmp_path/'valid.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    with caplog.at_level(logging.WARNING, logger=bridge.__name__):
        assert bridge.load_identity_edges(path) == data['edges']
    assert not [r for r in caplog.records if r.name == bridge.__name__]

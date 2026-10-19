"""Read-only equipment/prescription audit; no identity or admission approvals."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from unittest.mock import patch

CORE = Path(__file__).resolve().parents[1]
ROOT = CORE.parents[1]
PAIRS = {'6': 'V24ROW-004721', '9': 'V24ROW-004723',
         '140': 'V24ROW-004722', '171': 'V24ROW-004703', '135': 'V24ROW-004692'}
EQUIPMENT_CASES = {
    'bodyweight_only': (['BODYWEIGHT'], []),
    'band_available': (['BODYWEIGHT', 'BAND'], []),
    'plate_available': (['BODYWEIGHT', 'PLATE'], []),
    'equipment_blocked': (['BODYWEIGHT'], ['BAND', 'PLATE', 'MACHINE']),
}
LEVELS = ('beginner', 'intermediate', 'advanced')


def connect():
    conn = sqlite3.connect((CORE/'fit_core.sqlite').as_uri()+'?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def preserved_state():
    files = [CORE/'fit_core.sqlite', CORE/'data/provenance/v25_canonical_filter_core.csv']
    for directory in (CORE/'engines', ROOT/'docs/contracts', ROOT/'docs/ops/golden-examples'):
        files.extend(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in str(p))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}
    repos = {}
    for name in ('fit-core-ai', 'fit-core-backend', 'fit-core-frontend'):
        repos[name] = {}
        for label, args in {'head': ['rev-parse', 'HEAD'], 'status': ['status', '--porcelain=v1'],
                            'diff': ['diff', 'HEAD', '--binary']}.items():
            payload = subprocess.check_output(['git', '--no-optional-locks', '-C', str(ROOT/name), *args])
            repos[name][label+'Sha256'] = hashlib.sha256(payload).hexdigest()
    return {'hashes': hashes, 'collaborationStateDigests': repos}


def collect(dbq):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from engines.candidate_ranker import score_candidate_exercises
    from engines.equipment_policy import select_equipment_option
    from engines.experience_policy import automatic_recommendation_eligibility
    from tests.test_numeric_id_boundary_v1 import seed_rows, BE_SHA

    seeds = {str(r['id']): r for r in seed_rows()}
    with connect() as conn:
        raw = {r['id']: dict(r) for r in conn.execute('SELECT * FROM exercise_tier') if r['id'] in PAIRS.values()}
    source = dbq._load_source_equipment_by_id()
    manifest = json.loads((CORE/'engines/data/catalog_identity_bridge_v1.json').read_text())
    rows = []
    for sid, cid in PAIRS.items():
        edge, = [e for e in manifest['edges'] if e['sourceId'] == sid and e['catalogId'] == cid]
        assert edge['status'] == 'candidate'
        native, = dbq._enrich_candidates_with_local_catalog([copy.deepcopy(raw[cid])])
        numeric, = dbq._enrich_candidates_with_local_catalog([copy.deepcopy(seeds[sid])])
        # Counterfactual probe only: no persisted edge is approved or written.
        with patch.object(dbq, 'load_identity_edges', return_value=[{**edge, 'status': 'approved'}]):
            hypothetical, = dbq._enrich_candidates_with_local_catalog([copy.deepcopy(seeds[sid])])
        paths = {'native_catalog': native, 'current_numeric_seed': numeric,
                 'hypothetical_approved_numeric': hypothetical}
        probes = []
        for path, candidate in paths.items():
            for equipment_case, (available, blocked) in EQUIPMENT_CASES.items():
                for level in LEVELS:
                    selected = select_equipment_option(candidate.get('equipment_req'), blocked, available)
                    admission = automatic_recommendation_eligibility(candidate, level)
                    ranked = score_candidate_exercises(
                        [copy.deepcopy(candidate)] if selected is not None else [],
                        [candidate['primary_muscle']], available_equipment=available,
                        blocked_equipment=blocked, experience_level=level,
                        goal='hypertrophy', readiness_level='normal', training_style='gym',
                        target_split_label='custom', time_available_min=60,
                    )
                    probes.append({'path': path, 'equipmentCase': equipment_case, 'experience': level,
                                   'selectedEquipment': selected, 'admission': admission.__dict__,
                                   'isolatedRankedIds': [r['id'] for r in ranked]})
        rows.append({'sourceId': sid, 'catalogId': cid, 'sourceEquipment': source[cid],
                     'rawCatalog': raw[cid], 'seed': seeds[sid], 'paths': paths, 'probes': probes})
    queries = []
    engine = create_engine('sqlite://', creator=connect)
    try:
        with Session(engine) as db:
            for target in ('forearm', 'quadriceps', 'gluteal'):
                for name, (available, blocked) in EQUIPMENT_CASES.items():
                    candidates = dbq.get_candidate_exercises(db, [target], blocked, [], available)
                    queries.append({'target': target, 'equipmentCase': name,
                                    'queryCount': len(candidates),
                                    'suspectIds': [r['id'] for r in candidates if r['id'] in PAIRS.values()]})
    finally:
        engine.dispose()
    return {'backendSeedCommit': BE_SHA, 'rows': rows, 'queries': queries,
            'probeCount': sum(len(r['probes']) for r in rows),
            'scope': 'Candidate query, equipment selector, admission and isolated ranker; not final routine or live BE/LLM.',
            'hypotheticalApprovalPersisted': False}


def main():
    sys.path.insert(0, str(CORE))
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', PYTHON_DOTENV_DISABLED='1',
                      DATABASE_URL=f'sqlite:///file:{CORE/"fit_core.sqlite"}?mode=ro&uri=true',
                      FITCORE_SQLITE_DB_PATH=str(CORE/'fit_core.sqlite'))
    out = CORE/'docs/evidence/pending-equipment-audit-v1-2026-10-08'
    out.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        with (out/name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)

    before = preserved_state()
    write('before.json', before)
    from engines import db_queries as dbq
    with patch.object(dbq, '_connect_local_catalog_db', connect):
        dbq.clear_local_catalog_enrichment_cache()
        try:
            data = collect(dbq)
        finally:
            dbq.clear_local_catalog_enrichment_cache()
    write('audit.json', data)
    command = [sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
               'tests/test_pending_equipment_containment_v1.py',
               'tests/test_taxonomy_exact_id_correction_v1.py',
               'tests/test_muscle_taxonomy.py', f'--junitxml={out/"pytest.xml"}']
    write('command.json', command)
    with (out/'pytest.log').open('x') as stream:
        result = subprocess.run(command, cwd=CORE, stdout=stream, stderr=subprocess.STDOUT, timeout=600)
    write('exit.json', {'exitCode': result.returncode})
    after = preserved_state()
    write('after.json', after)
    assert before == after, 'Protected source or collaboration state changed; inspect digests.'
    print(json.dumps({'probes': data['probeCount'], 'queries': len(data['queries']),
                      'protectedStateUnchanged': before == after, 'pytestExit': result.returncode}))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()

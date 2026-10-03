"""Rebuild the minimal run outputs that the analysis scripts read from predictions/.

The analysis scripts in analysis/fse_revision_20260923/ read run outputs at their original paths
(outdir/..., data/..., icl/...). For every table predictions/<path>.csv.gz, this script writes a
JSON-lines file at <path>.jsonl that holds the fields those scripts use: idx, target, arm, the raw
answer, the retrieved ids (candidate_ids for fixed-candidate and CWE/CVE runs; the demonstrations
as `similar` for retrieved demonstrations), and the fields of `extra` (job, row and pair ids; the
SHA-256 of a mode B evidence block as `evidence_sha256`; for three mode B runs, the code-free
retrieval records as `query_chunks`). Every predictions/<path>.gz is
decompressed to <path> (code-free auxiliary records: jobs, targets, retrieval lists, configs).
Prompts are not restored; analyses that inspect the full prompt text need the full outputs.

Run from the repository root: python3 tools/expand_predictions.py
"""
import csv
import gzip
import json
import shutil
from pathlib import Path

csv.field_size_limit(1 << 30)  # rows that keep the code-free query_chunks of mode B
ROOT = Path('predictions')
for src in sorted(ROOT.rglob('*.gz')):
    rel = str(src.relative_to(ROOT))
    table = rel.endswith('.csv.gz')
    dest = Path(rel[:-len('.csv.gz')] + '.jsonl' if table else rel[:-len('.gz')])
    if dest.exists():
        print('exists, skipped:', dest)
        continue
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not table:
        with gzip.open(src, 'rb') as fh, open(dest, 'wb') as out:
            shutil.copyfileobj(fh, out)
        print('wrote', dest)
        continue
    with gzip.open(src, 'rt', encoding='utf-8', newline='') as fh, open(dest, 'w', encoding='utf-8') as out:
        for r in csv.DictReader(fh):
            extra = json.loads(r['extra']) if r.get('extra') else {}
            idx = int(r['idx']) if r['idx'].isdigit() else r['idx']
            rec = {'idx': idx, 'query_idx': idx, 'target': int(r['target']) if r['target'] else None,
                   'query_target': int(r['target']) if r['target'] else None, 'response': r['response']}
            if r['arm']:
                rec['arm'] = r['arm']
            ids = json.loads(r['retrieved_ids']) if r['retrieved_ids'] else None
            if ids is not None:
                if extra.pop('ids', None) == 'candidate_ids':
                    rec['candidate_ids'] = ids
                else:
                    rec['similar'] = [{'idx': i} for i in ids]
                    rec['num_examples_used'] = len(ids)
            rec.update(extra)
            out.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print('wrote', dest)

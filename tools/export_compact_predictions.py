"""Export compact prediction tables and code-free auxiliary records from the full run outputs.

The full outputs repeat the complete prompt in every row (about 11 GB in total) and contain the
code of PrimeVul and MegaVul. This script writes what the analysis scripts need without that code.

--list FILE   one run output per line. For each run, a table (.csv.gz) with one row per function:
              the function index, its label, the arm, the ids of the retrieved items where the run
              records them, the YES/NO prediction (same parser as the analyses), the raw model
              answer, and `extra`, a JSON object with the job, row and pair ids where the run has
              them and, for prompts with a chunk evidence block (mode B), the SHA-256 of that block
              (as cut by rq2_zero_shot_subsets.evidence), which identifies identical evidence
              without the code.
--query-chunks FILE
              runs (a subset of --list) whose rows also keep, in `extra`, the retrieval records of
              mode B (`query_chunks`) without code: scores, ranks, labels, and line numbers; every
              field that holds code or a function signature is dropped (strip_code).
--slim FILE   one auxiliary file per line (jobs, targets, retrieval lists, configs). Each record of a
              JSON-lines file keeps only the id fields in KEEP; a JSON config loses every key that
              names an endpoint. Written as <path>.gz.

Request metadata (endpoints, usage) is dropped. Run from the root of the full working tree:
  python3 tools/export_compact_predictions.py --list compact_list.txt --slim slim_list.txt --out <release>/predictions
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, '.')
from evaluate_prompt_outputs import parse_prediction  # noqa: E402  std prompts
sys.path.insert(0, 'experiments/knowledge_rag')
import experiment as fc_exp  # noqa: E402  fixed-candidate and CWE/CVE prompts

FIELDS = ['idx', 'target', 'arm', 'retrieved_ids', 'prediction', 'response', 'extra']
EXTRA = ('job_id', 'row_id', 'pair_id')
KEEP = ('row_id', 'idx', 'query_idx', 'pair_id', 'target', 'query_target', 'cluster', 'arm', 'job_id',
        'candidate_ids', 'evidence_pair', 'num_examples_used', 'similar')


def evidence(msg):  # as in analysis/fse_revision_20260923/rq2_zero_shot_subsets.py
    f = msg.find('Focus Regions:')
    if f == -1:
        return None
    j = msg.find('\n\nRetrieved ', f)
    return msg[j:] if j != -1 else msg[f:]


CODE_KEYS = ('code', 'func', 'signature', 'slice')
KEEP_FUNC = ('func_hash', 'function_target', 'function_id', 'query_func_id')


def strip_code(x):
    """Drop every field that holds code; keep numbers, ids, and labels."""
    if isinstance(x, dict):
        return {k: strip_code(v) for k, v in x.items()
                if k in KEEP_FUNC or not any(c in k.lower() for c in CODE_KEYS)
                and not (isinstance(v, str) and ('\n' in v or len(v) > 80))}
    if isinstance(x, list):
        return [strip_code(v) for v in x]
    return x


def rows(path, keep_chunks=False):
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        idx = r.get('query_idx', r.get('idx'))
        target = r.get('query_target', r.get('target'))
        extra = {k: r[k] for k in EXTRA if r.get(k) is not None}
        if r.get('candidate_ids') is not None:
            retrieved, fc = r['candidate_ids'], True
            extra['ids'] = 'candidate_ids'
        elif r.get('similar'):
            used = r.get('num_examples_used', len(r['similar']))
            retrieved, fc = [s.get('idx') for s in r['similar'][:used] if isinstance(s, dict)], False
        else:
            retrieved, fc = None, 'knowledge_rag' in path or '/e_arm_' in path
        msgs = r.get('messages')
        if isinstance(msgs, list) and msgs and not fc:
            ev = evidence(msgs[-1]['content'])
            if ev is not None:
                extra['evidence_sha256'] = hashlib.sha256(ev.encode()).hexdigest()
        if keep_chunks and r.get('query_chunks') is not None:
            extra['query_chunks'] = strip_code(r['query_chunks'])
        response = r.get('response')
        pred = fc_exp.prediction(response) if fc else parse_prediction(response)
        yield {'idx': idx, 'target': target, 'arm': r.get('arm', ''),
               'retrieved_ids': json.dumps(retrieved) if retrieved is not None else '',
               'prediction': '' if pred is None else pred, 'response': response if response is not None else '',
               'extra': json.dumps(extra, sort_keys=True) if extra else ''}


def slim_record(r):
    out = {k: r[k] for k in KEEP if k in r}
    if isinstance(out.get('similar'), list):
        out['similar'] = [{'idx': s.get('idx')} for s in out['similar'] if isinstance(s, dict)]
    return out


def drop_endpoints(x):
    if isinstance(x, dict):
        return {k: drop_endpoints(v) for k, v in x.items() if 'url' not in k.lower() and 'endpoint' not in k.lower()}
    if isinstance(x, list):
        return [drop_endpoints(v) for v in x]
    return x


def gz_text(dest):
    return io.TextIOWrapper(gzip.GzipFile(dest, 'wb', mtime=0), encoding='utf-8', newline='')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', help='file with one run output path per line')
    ap.add_argument('--query-chunks', help='runs whose rows keep the code-free query_chunks')
    ap.add_argument('--slim', help='file with one auxiliary file path per line')
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()
    keep = {l.strip() for l in open(a.query_chunks) if l.strip()} if a.query_chunks else set()
    for path in [l.strip() for l in open(a.list) if l.strip()] if a.list else []:
        dest = a.out / (path.rsplit('.', 1)[0] + '.csv.gz')
        dest.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with gz_text(dest) as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            for row in rows(path, path in keep):
                w.writerow(row); n += 1
        print(f'{n:5d} rows  {dest}')
    for path in [l.strip() for l in open(a.slim) if l.strip()] if a.slim else []:
        dest = a.out / (path + '.gz')
        dest.parent.mkdir(parents=True, exist_ok=True)
        with gz_text(dest) as fh:
            if path.endswith('.jsonl'):
                for line in open(path, encoding='utf-8'):
                    fh.write(json.dumps(slim_record(json.loads(line)), ensure_ascii=False) + '\n')
            else:
                fh.write(json.dumps(drop_endpoints(json.load(open(path))), indent=1, ensure_ascii=False) + '\n')
        print(f'slim   {dest}')


if __name__ == '__main__':
    main()

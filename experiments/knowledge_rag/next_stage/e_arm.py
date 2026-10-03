import os
#!/usr/bin/env python3
"""CWE/CVE-entry retrieval arms (usage mode E) on top of a complete v7 knowledge run.

Arms (retrieval fixed in advance by build_e_retrieval.py, identical for every model):
  cwe_code        X2  top-3 CWE entries retrieved with the function code as query
  cwe_desc        X3  top-3 CWE entries retrieved with the LLM-generated description as query
  cve_desc        X4  top-3 CVE descriptions (training split) retrieved with that description
  cwe_desc_short  X6  the same entries as cwe_desc, shown as ID + name + first sentence
Each prompt is the run's own no_retrieval prompt with only the evidence slot filled, so the
arms differ from no_retrieval, code_pair and knowledge only in the retrieved content.
Mirrors extra_arm.py (same validation, freezing, runner, model-name check and statistics);
extra_arm.py itself is not modified. No truncation: over-budget prompts are errors.
"""
import argparse, fcntl, hashlib, json, sys
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiment as exp
from offline_review import comparisons

ARMS = {'cwe_code': ('retrieval_cwe_code.jsonl', 'cwe_entries.jsonl', 'prompt_full'),
        'cwe_desc': ('retrieval_cwe_desc.jsonl', 'cwe_entries.jsonl', 'prompt_full'),
        'cve_desc': ('retrieval_cve_desc.jsonl', 'cve_entries.jsonl', 'prompt_full'),
        'cwe_desc_short': ('retrieval_cwe_desc.jsonl', 'cwe_entries.jsonl', 'prompt_short')}
TOP_K = 3
# Name the gateway reports for a pinned alias; every other model must return its own name.
RETURNED = {'deepseek-v4-pro-0813': 'deepseek-v4-pro-ga-260813'}
MARKER = 'Historical evidence (may be irrelevant; check the target independently):\n(none)\n\n'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_once(p, rows):
    if p.exists():
        if exp.read(p) != rows:
            raise ValueError('Frozen jobs changed')
    else:
        exp.write(p, rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--predictions', type=Path)
    p.add_argument('--retrieval-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arm', choices=sorted(ARMS), required=True)
    p.add_argument('--stage', choices=['prepare', 'run', 'score'], default='prepare')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--max-calls', type=int, default=870)
    a = p.parse_args()
    if a.workers < 1 or a.max_calls < 1:
        p.error('positive workers/max-calls required')
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(a)


def execute(a):
    src = a.source
    pred = a.predictions or src / 'predictions.jsonl'
    original_jobs = exp.read(src / 'detection_jobs.jsonl')
    original = exp.read(pred)
    if len(original) != len(original_jobs) or len({r['job_id'] for r in original}) != len(original):
        raise ValueError('Use complete original/recovered predictions')
    if {r['job_id'] for r in original} != {r['job_id'] for r in original_jobs}:
        raise ValueError('Original job IDs differ')
    if len({exp.digest(r['config']) for r in original}) != 1:
        raise ValueError('Mixed original configs')
    oj = {j['job_id']: j for j in original_jobs}
    for r in original:
        if r['messages'] != oj[r['job_id']]['messages'] or exp.prediction(r.get('response')) is None:
            raise ValueError('Invalid original prediction')
    cfg = original[0]['config']
    targets = exp.read(src / 'targets.jsonl')
    no = {r['row_id']: r for r in original_jobs if r['arm'] == 'no_retrieval'}

    rfile, efile, field = ARMS[a.arm]
    rd = a.retrieval_dir
    retrieved = {r['row_id']: r['ids'][:TOP_K] for r in exp.read(rd / rfile)}
    entries = {e['id']: e[field] for e in exp.read(rd / efile)}
    if set(retrieved) != {t['row_id'] for t in targets}:
        raise ValueError('Retrieval rows differ from target rows')
    manifest = json.loads((rd / 'manifest.json').read_text())
    for name in (rfile, efile):
        if manifest['outputs_sha256'].get(name) != sha(rd / name):
            raise ValueError('Retrieval file differs from its manifest: ' + name)
    frozen = {'arm': a.arm, 'config': cfg, 'top_k': TOP_K, 'entry_field': field,
              'files': {n: sha(src / n) for n in ['targets.jsonl', 'detection_jobs.jsonl']},
              'retrieval_files': {n: sha(rd / n) for n in (rfile, efile, 'manifest.json')},
              'predictions_sha256': sha(pred), 'script_sha256': sha(Path(__file__)),
              'stats_sha256': sha(Path(__file__).with_name('offline_review.py'))}
    fp = a.out / 'config.json'
    if fp.exists() and json.loads(fp.read_text()) != frozen:
        raise ValueError('Configuration changed; use new output directory')

    jobs, contexts = [], []
    for t in targets:
        ids = retrieved[t['row_id']]
        context = '\n\n'.join(entries[i] for i in ids)
        messages = [dict(m) for m in no[t['row_id']]['messages']]
        if not messages[1]['content'].startswith(MARKER):
            raise ValueError('Unknown base prompt')
        messages[1]['content'] = MARKER.replace('(none)', context) + messages[1]['content'][len(MARKER):]
        j = {k: v for k, v in t.items() if k != 'func'}
        j.update(arm=a.arm, messages=messages, candidate_ids=ids, job_id=exp.digest([t['row_id'], a.arm, messages]))
        jobs.append(j)
        contexts.append({'row_id': t['row_id'], 'entry_ids': ids, 'evidence_bytes': len(context.encode())})
    fp.write_text(json.dumps(frozen, indent=2))
    write_once(a.out / 'jobs.jsonl', jobs)
    write_once(a.out / 'evidence_manifest.jsonl', contexts)
    from context_budget import audit_requests
    audit_requests(jobs, cfg['model'], cfg['max_tokens'], a.out / 'context_audit.json', cfg['prompt_strategy'], cfg['temperature'])
    print(f'Prepared {len(jobs)} {a.arm} requests.', flush=True)
    output = a.out / 'predictions.jsonl'
    if a.stage == 'prepare':
        return
    if a.stage == 'run':
        # Route each model to its ORIGINAL gateway.
        if os.environ.get('SLICERAG_UNIAPI_HOST', '\0') in cfg['base_url']:  # host of the OpenAI-compatible endpoint used for GLM/DeepSeek
            if cfg['model'].startswith('deepseek'):
                from run_deepseek_endpoint import configure
            else:
                from run_glm_endpoint import configure
            configure()
        from concurrent_runner import inference_config, run
        api = exp.load_client()
        args = NS(model=cfg['model'], temperature=cfg['temperature'], max_output=cfg['max_tokens'],
                  prompt_strategy=cfg['prompt_strategy'], seed=cfg['seed'])
        if inference_config(api, args) != cfg:
            raise ValueError('Gateway/client/config differs from original; no calls made')
        expected = RETURNED.get(cfg['model'], cfg['model'])
        loader = exp.load_client

        def checked_client():
            c = loader()
            call = c.get_openai_chat

            def checked(*args, **kwargs):
                result = call(*args, **kwargs)
                if c.last_response_metadata.get('returned_model') != expected:
                    raise ValueError('Returned model differs from requested model')
                return result
            c.get_openai_chat = checked
            return c
        exp.load_client = checked_client
        run(NS(**vars(args), jobs=a.out / 'jobs.jsonl', output=output, workers=a.workers,
               read_timeout=600, max_attempts=3, max_calls=a.max_calls))
    rs = exp.read(output)
    if len(rs) != len(jobs):
        print('Partial run saved. Repeat run without --max-calls to continue.')
        return
    stats = comparisons(original_jobs + jobs, original + rs)
    (a.out / 'paired_comparisons.json').write_text(json.dumps(stats, indent=2))
    states = {r['job_id']: exp.prediction(r['response']) for r in rs}
    tp = sum(states[j['job_id']] == 1 and j['target'] == 1 for j in jobs)
    fpn = sum(states[j['job_id']] == 1 and j['target'] == 0 for j in jobs)
    fn = sum(states[j['job_id']] == 0 and j['target'] == 1 for j in jobs)
    tn = len(jobs) - tp - fpn - fn
    pairs = {}
    for j in jobs:
        pairs.setdefault(j['pair_id'], {})[j['target']] = states[j['job_id']]
    pc = sum(v == {0: 0, 1: 1} for v in pairs.values()) / len(pairs)
    (a.out / 'arm_metrics.json').write_text(json.dumps(
        {'arm': a.arm, 'calls': len(jobs), 'P-C': pc, 'pairs': len(pairs), 'TP': tp, 'FP': fpn, 'FN': fn, 'TN': tn,
         'F1': 2 * tp / (2 * tp + fpn + fn) if 2 * tp + fpn + fn else 0}, indent=2))
    print('Complete: arm_metrics.json and paired_comparisons.json')


if __name__ == '__main__':
    main()

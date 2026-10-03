"""Second dataset, modes B and E: change in P-C and net P-C (P-C minus P-R) within the MegaVul Java pairs
whose two members receive the same retrieved items and within the others (offline; no API calls).

B is compared with the Java two-shot prompt, E (CWE entries with code or description query, CVE
descriptions) with the Java no-retrieval prompt, as in java_figure_data.py. Same items: B, the
inserted evidence (hashes of data/b_keys/megavul_java_b_keys.jsonl.gz); E, the set of the top-3
entries. Intervals: commit-cluster bootstrap of java_replication.py (3000 draws, seed 20260920),
not corrected for multiplicity.

Run from the repository root: python3 analysis/fse_revision_20260923/java_subsets.py
"""
import gzip
import json
from pathlib import Path

import numpy as np

SRC = Path('analysis/fse_revision_20260923/java_replication.py').read_text()
exec(compile(SRC[:SRC.index('demo = {')], 'java_replication', 'exec'))  # loaders, PAIRS, W, counts, rates

B = {r['idx']: r['b_key_sha256'] for r in map(json.loads, gzip.open('data/b_keys/megavul_java_b_keys.jsonl.gz', 'rt'))}


def e_preds(model, arm):  # as in java_figure_data.py
    d = f'outdir/java_e_arm_{model}_{arm}'
    jobs = {j['job_id']: str(j['idx']) for j in map(json.loads, open(f'{d}/jobs.jsonl'))}
    return {jobs[r['job_id']]: fc_exp.prediction(r.get('response')) for r in map(json.loads, open(f'{d}/predictions.jsonl'))}


def e_keys(model, arm):
    return {str(r['idx']): frozenset(r['candidate_ids'][:3]) for r in map(json.loads, open(f'outdir/java_e_arm_{model}_{arm}/predictions.jsonl'))}


def change(new, ref, keep):
    sub = lambda m: {x: y for x, y in m.items() if x in keep and new.get(x) is not None and ref.get(x) is not None}
    a, b = counts(sub(new)), counts(sub(ref))
    out = {'pairs': int(a[:, 4].sum())}
    for name, f in (('P-C', lambda r: r['P-C']), ('net P-C', lambda r: r['P-C'] - r['P-R'])):
        d = f(rates(W @ a)) - f(rates(W @ b))
        out[name] = {'before': round(100 * float(f(rates(b.sum(0)))), 1), 'after': round(100 * float(f(rates(a.sum(0)))), 1),
                     'diff': round(100 * float(f(rates(a.sum(0))) - f(rates(b.sum(0)))), 1),
                     'ci95': [round(100 * float(q), 1) for q in np.quantile(d, [.025, .975])]}
        out[name]['significant'] = out[name]['ci95'][0] > 0 or out[name]['ci95'][1] < 0
    return out


OUT = {}
for model in MODELS:
    std = {k: std_preds(f'{D}/{model}_{k}.jsonl') for k in ('two_shot_java', 'B_chunks_diff')}
    nr = fc_preds(f'outdir/java_fc_{model}/predictions.jsonl', 'no_retrieval')
    runs = [('B', std['B_chunks_diff'], std['two_shot_java'], B)]
    runs += [(f'E {arm}', e_preds(model, arm), nr, e_keys(model, arm)) for arm in ('cwe_code', 'cwe_desc', 'cve_desc')]
    for name, new, ref, key in runs:
        same = {x for v, f, _ in PAIRS if key.get(v) is not None and key.get(v) == key.get(f) for x in (v, f)}
        other = {x for v, f, _ in PAIRS for x in (v, f)} - same
        OUT[f'{model} | {name}'] = {'same items': change(new, ref, same), 'different items': change(new, ref, other)}
json.dump(OUT, open('analysis/fse_revision_20260923/java_subsets.json', 'w'), indent=1)
for k, v in OUT.items():
    for s, r in v.items():
        print(f"{k:22s} {s:16s} n={r['pairs']:3d}  net {r['net P-C']['before']:5.1f} -> {r['net P-C']['after']:5.1f} {r['net P-C']['ci95']} {'*' if r['net P-C']['significant'] else ''}")

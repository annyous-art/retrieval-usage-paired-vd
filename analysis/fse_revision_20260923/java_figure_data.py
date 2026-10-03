"""Data for the Java figure (offline): the four comparisons of Table tab:java on the 620 MegaVul Java
pairs, with P-C, P-B, net P-C and the YES rate before and after, the change, and its commit-cluster bootstrap interval
(3000 draws, seed 20260920). Modes A and B are compared with the Java two-shot prompt (two_shot_java), as in
the table; java_replication.py compares A with the C two-shot prompt instead. Mode E (CWE entries,
code query; outdir/java_e_arm_*_cwe_code) is compared with the Java no-retrieval prompt.

Run from the repository root: python3 analysis/fse_revision_20260923/java_figure_data.py
"""
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path('analysis/fse_revision_20260923/java_replication.py').read_text()
exec(compile(SRC[:SRC.index('demo = {')], 'java_replication', 'exec'))  # loaders, PAIRS, W

ROWS = [('A', 'Demonstrations vs. Java two-shot', 'A', 'two_shot_java'),
        ('B', 'Chunks vs. Java two-shot', 'B', 'two_shot_java'),
        ('C', 'Code pair vs. no retrieval', 'code_pair', 'no_retrieval'),
        ('D', 'Knowledge vs. no retrieval', 'knowledge', 'no_retrieval'),
        ('D', 'Knowledge vs. code pair', 'knowledge', 'code_pair'),
        ('E', 'CWE entries vs. no retrieval', 'E', 'no_retrieval'),
        ('E', 'CWE entries, description query vs. no retrieval', 'E_cwe_desc', 'no_retrieval'),
        ('E', 'CVE descriptions vs. no retrieval', 'E_cve_desc', 'no_retrieval')]


def e_preds(model, arm='cwe_code'):
    d = f'outdir/java_e_arm_{model}_{arm}'
    jobs = {j['job_id']: str(j['idx']) for j in map(json.loads, open(f'{d}/jobs.jsonl'))}
    return {jobs[r['job_id']]: fc_exp.prediction(r.get('response')) for r in map(json.loads, open(f'{d}/predictions.jsonl'))}


def change(A, B):
    keep = {x for v, f, _ in PAIRS for x in (v, f) if A.get(x) is not None and B.get(x) is not None}
    a = counts({k: v for k, v in A.items() if k in keep}); b = counts({k: v for k, v in B.items() if k in keep})
    # columns of a count table: P-C, P-V, P-B, P-R, pairs; net P-C = P-C - P-R; a pair in P-V has two YES
    # answers and a pair in P-C or P-R one, so the YES rate is P-V + (P-C + P-R) / 2
    STATS = {'P-C': lambda t: t[:, 0], 'P-B': lambda t: t[:, 2], 'net': lambda t: t[:, 0] - t[:, 3],
             'YES': lambda t: t[:, 1] + (t[:, 0] + t[:, 3]) / 2}
    out = {'pairs': int(a[:, 4].sum())}
    for name, f in STATS.items():
        rate = lambda t: (W @ f(t)) / (W @ t[:, 4]); pt = lambda t: f(t).sum() / t[:, 4].sum()  # noqa: E731
        d = rate(a) - rate(b)
        out[name] = [round(100 * pt(b), 1), round(100 * pt(a), 1),
                     [round(100 * float(q), 1) for q in np.quantile(d, [.025, .975])], round(100 * (pt(a) - pt(b)), 1)]
    return out


rows = []
for model in MODELS:
    M = {'A': std_preds(f'{D}/{model}_A_top2.jsonl'), 'two_shot_java': std_preds(f'{D}/{model}_two_shot_java.jsonl'),
         'B': std_preds(f'{D}/{model}_B_chunks_diff.jsonl'), 'E': e_preds(model),
         'E_cwe_desc': e_preds(model, 'cwe_desc'), 'E_cve_desc': e_preds(model, 'cve_desc')}
    M.update({arm: fc_preds(f'outdir/java_fc_{model}/predictions.jsonl', arm) for arm in ('no_retrieval', 'code_pair', 'knowledge')})
    for mode, label, a, b in ROWS:
        rows.append({'mode': mode, 'comparison': label, 'model': model, **change(M[a], M[b])})
        print(model, label, rows[-1])
Path(__file__).with_suffix('.json').write_text(json.dumps(rows, indent=1))

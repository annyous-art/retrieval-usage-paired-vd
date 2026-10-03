"""GLM-5.1 prompt baselines and std-setting retrieval variants on the 431 resolvable pairs.

Rows of the supporting table in RQ1 (all-vulnerable constant, three prompt baselines, retrieved
demonstrations, three chunk variants, patch contrast) and the 21 pairwise F1 / P-C differences
among the seven non-constant methods other than patch contrast, with the commit-cluster
bootstrap and Holm correction per metric. Replaces table2_pairwise_holm.py, which counted the
four pairs whose idx occurs twice (435 pairs).

Run from the repository root: python3 analysis/fse_revision_20260923/glm_std_prompts.py
"""
import collections
import itertools
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from evaluate_prompt_outputs import parse_prediction  # noqa: E402

RAW = [json.loads(l) for l in open('data/primevul_test_paired_labeled.jsonl')]
dup = collections.Counter(str(r['idx']) for r in RAW)
PAIRS = []
for i in range(0, len(RAW), 2):
    a, b = RAW[i], RAW[i + 1]
    if dup[str(a['idx'])] > 1 or dup[str(b['idx'])] > 1:
        continue
    v, f = (a, b) if int(a['target']) == 1 else (b, a)
    PAIRS.append((str(v['idx']), str(f['idx']), f"{v['project']}:{v['commit_id']}"))
clusters = sorted({c for *_, c in PAIRS})
cid = {c: i for i, c in enumerate(clusters)}
W = np.random.default_rng(20260920).multinomial(len(clusters), np.ones(len(clusters)) / len(clusters), size=3000)


def jsonl_preds(path):
    m = {}
    for l in open(path):
        r = json.loads(l)
        k = str(r.get('query_idx', r.get('idx')))
        p = parse_prediction(r.get('response'))
        if p is not None or k not in m:
            m[k] = p
    return m


df = pd.read_csv('outdir/analysis/per_sample_feature_table.csv')
col = lambda c: {str(i): (int(p) if p in (0, 1) else None) for i, p in zip(df.idx, pd.to_numeric(df[c], errors='coerce'))}
# PrimeVul two-shot, retrieved demonstrations and BC use the same files as rq1_rq3_modes.py.
METHODS = {
    'All-vulnerable constant': {x: 1 for v, f, _ in PAIRS for x in (v, f)},
    'Zero-shot': col('zero_shot_pred'),
    'YES-only few-shot': col('yes_only_pred'),
    'PrimeVul two-shot': jsonl_preds('outdir/glm-5.1_std_cls_logprobsFalse_fewshotegTrue_baseline_870.jsonl'),
    'Retrieved demonstrations': jsonl_preds('outdir/icl_fixed_baseline_compatible_test870/glm-5.1_std_cls_fewshotegTrue_top2_baseline_compatible.jsonl'),
    'Chunks + PrimeVul demonstrations': col('old_rag_pred'),
    'Positive-weighted chunks': col('positive_weighted_rag_pred'),
    'Balanced grouped no-label (BC)': jsonl_preds('outdir/prompt_baseline_compatible_glm51_test870/glm-5.1_std_cls_grouped_no_label_baseline_compatible_fewshotegTrue_cls0.jsonl'),
    'Patch contrast': jsonl_preds('outdir/prompt_patch_contrast_vuln_margin_glm51_test870/glm-5.1_std_cls_patch_contrast_fewshotegTrue.jsonl'),
}
MODE = {'Retrieved demonstrations': 'A', 'Chunks + PrimeVul demonstrations': 'B', 'Positive-weighted chunks': 'B',
        'Balanced grouped no-label (BC)': 'B', 'Patch contrast': 'C'}


def counts(m, keep=None):
    arr = np.zeros((len(clusters), 8))  # P-C P-V P-B P-R tp fp correct n
    for v, f, c in PAIRS:
        a, b = m.get(v), m.get(f)
        if a is None or b is None or (keep is not None and (v, f) not in keep):
            continue
        k = {(1, 0): 0, (1, 1): 1, (0, 0): 2, (0, 1): 3}[(a, b)]
        arr[cid[c], k] += 1; arr[cid[c], 4] += a; arr[cid[c], 5] += b
        arr[cid[c], 6] += a + (1 - b); arr[cid[c], 7] += 1
    return arr


def rates(t):
    n, tp, fp = t[..., 7], t[..., 4], t[..., 5]
    safe = np.where(n > 0, n, 1)
    den = tp + fp + n  # 2tp + fp + fn with fn = n - tp
    f1 = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)
    return {'Acc.': t[..., 6] / (2 * safe), 'F1': f1, 'YES': (tp + fp) / (2 * safe),
            'P-C': t[..., 0] / safe, 'P-V': t[..., 1] / safe, 'P-B': t[..., 2] / safe, 'P-R': t[..., 3] / safe}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i])); adj[i] = run
    return adj


OUT = {'pairs': len(PAIRS), 'rows': {}, 'pairwise': []}
for name, m in METHODS.items():
    t = counts(m).sum(0)
    OUT['rows'][name] = {'mode': MODE.get(name, '--'), 'pairs': int(t[7]), **{k: round(float(v), 4) for k, v in rates(t).items()}}

tested = [n for n in METHODS if n not in ('All-vulnerable constant', 'Patch contrast')]
for a, b in itertools.combinations(tested, 2):
    keep = {(v, f) for v, f, _ in PAIRS if all(M.get(x) is not None for M in (METHODS[a], METHODS[b]) for x in (v, f))}
    ca, cb = counts(METHODS[a], keep), counts(METHODS[b], keep)
    for met in ('F1', 'P-C'):
        point = rates(ca.sum(0))[met] - rates(cb.sum(0))[met]
        d = rates(W @ ca)[met] - rates(W @ cb)[met]
        OUT['pairwise'].append({'a': a, 'b': b, 'metric': met, 'pairs': len(keep), 'diff_points': round(100 * float(point), 2),
                                'ci95': [round(100 * float(q), 2) for q in np.quantile(d, [.025, .975])],
                                'p': round(float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))), 4)})
for met in ('F1', 'P-C'):
    sub = [r for r in OUT['pairwise'] if r['metric'] == met]
    for r, adj in zip(sub, holm([r['p'] for r in sub])):
        r['p_holm'] = round(adj, 4)

json.dump(OUT, open('analysis/fse_revision_20260923/glm_std_prompts.json', 'w'), indent=1)
for n, r in OUT['rows'].items():
    print(f"{r['mode']:2s} {n:32s} n={r['pairs']} " + ' '.join(f"{k} {r[k]:.3f}" for k in ('Acc.', 'F1', 'YES', 'P-C', 'P-V', 'P-B', 'P-R')))
for met in ('F1', 'P-C'):
    sub = sorted([r for r in OUT['pairwise'] if r['metric'] == met], key=lambda r: r['p'])
    print(f"{met}: {sum(r['p_holm'] < .05 for r in sub)} of {len(sub)} significant after Holm")
    for r in sub:
        print(f"  {r['a']:30s} - {r['b']:30s} {r['diff_points']:+6.2f} {r['ci95']} p_holm {r['p_holm']}")

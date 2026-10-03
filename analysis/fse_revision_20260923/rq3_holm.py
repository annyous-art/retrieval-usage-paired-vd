"""Holm correction for the RQ3 presentation comparisons (offline).

Family: the change in P-C and in net P-C (P-C minus P-R) for all 16 presentation comparisons
(8 order reversals in rq3_order.py, 4 regroupings and 4 entry shortenings in rq3_figure_data.py),
32 tests in total, as in Figure fig:rq3. Two-sided p-values come from the commit-cluster
bootstrap of rq2_zero_shot_subsets.py (3000 draws, seed 20260920). 
Run from the repository root: python3 analysis/fse_revision_20260923/rq3_holm.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'analysis/fse_revision_20260923')
import rq3_figure_data as fd  # noqa: E402
import rq2_zero_shot_subsets as z  # noqa: E402
import rq3_order as r  # noqa: E402

FAMILY = 32


def pvalues(new, old):
    keep = [all(p.get(x) is not None for p in (new, old) for x in (v, f)) for v, f, _ in z.PAIRS]
    arr = {}
    for name, pred in (('new', new), ('old', old)):
        a = np.zeros((len(z.clusters), 3))
        for (v, f, c), k in zip(z.PAIRS, keep):
            if k:
                o = (pred[v], pred[f])
                a[z.cid[c], 0] += o == (1, 0)
                a[z.cid[c], 1] += o == (0, 1)
                a[z.cid[c], 2] += 1
        arr[name] = a
    rate = lambda a, i: (z.W @ a[:, i]) / (z.W @ a[:, 2])
    dpc = rate(arr['new'], 0) - rate(arr['old'], 0)
    dnet = (rate(arr['new'], 0) - rate(arr['new'], 1)) - (rate(arr['old'], 0) - rate(arr['old'], 1))
    p = lambda d: min(1.0, 2 * min(float((d <= 0).mean()), float((d >= 0).mean())))
    return p(dpc), p(dnet)


def main():
    tests = []
    for mode, change, model, new, old in list(r.COMPARISONS) + fd.NEW:
        ppc, pnet = pvalues(new(), old())
        tests += [(f'{mode} | {change} | {model} | P-C', ppc), (f'{mode} | {change} | {model} | net P-C', pnet)]
    tests.sort(key=lambda t: t[1])
    out, alive = [], True
    for i, (name, p) in enumerate(tests):
        threshold = 0.05 / (FAMILY - i)
        alive = alive and p < threshold
        adj = min(1.0, max([(FAMILY - j) * tests[j][1] for j in range(i + 1)]))
        out.append({'test': name, 'p': round(p, 4), 'p_holm': round(adj, 4), 'holm_threshold': round(threshold, 4), 'significant_after_holm': alive})
        print(f'{name:28s} p={p:.4f} threshold={threshold:.4f} {"significant" if alive else "not significant"}')
    Path(__file__).with_suffix('.json').write_text(json.dumps({'family_size': FAMILY, 'tests': out}, indent=1))


if __name__ == '__main__':
    main()

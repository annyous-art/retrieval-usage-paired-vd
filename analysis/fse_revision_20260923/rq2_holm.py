"""Holm correction for the RQ2 subset comparisons (offline).

Families: the change in net P-C (P-C minus P-R) relative to zero-shot for the 20 comparisons
(10 configurations x 2 models) within each subset of rq2_zero_shot_subsets.py, the pairs whose two
versions receive the same items and those that receive different items. Two-sided p-values come
from the same commit-cluster bootstrap (3000 draws, seed 20260920).

Run from the repository root: python3 analysis/fse_revision_20260923/rq2_holm.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, 'analysis/fse_revision_20260923')
sys.path.insert(0, 'experiments/knowledge_rag')
import rq2_zero_shot_subsets as m  # noqa: E402

SKIP = ('Retrieved chunks only', 'Changed chunks')  # not part of the RQ2 figure


def pvalue(pred, ctl, mask):
    mask = [k and all(p.get(x) is not None for p in (pred, ctl) for x in (v, f)) for k, (v, f, _) in zip(mask, m.PAIRS)]
    a, b = m.counts(pred, mask), m.counts(ctl, mask)
    net = (m.W @ a[:, 0] - m.W @ a[:, 1]) / (m.W @ a[:, 2]) - (m.W @ b[:, 0] - m.W @ b[:, 1]) / (m.W @ b[:, 2])
    return min(1.0, 2 * min(float((net <= 0).mean()), float((net >= 0).mean())))


def holm(tests):
    tests = sorted(tests, key=lambda t: t[1])
    n, alive, out, adj = len(tests), True, [], 0.0
    for i, (name, p) in enumerate(tests):
        alive = alive and p < 0.05 / (n - i)
        adj = min(1.0, max(adj, (n - i) * p))
        out.append({'test': name, 'p': round(p, 4), 'p_holm': round(adj, 4), 'significant_after_holm': alive})
    return out


def main():
    fam = {'same': [], 'different': []}
    for (mode, inst), runs in m.INSTANCES.items():
        if inst in SKIP:
            continue
        for model, load in runs.items():
            pred, key = load()
            same = [key.get(v) is not None and key.get(v) == key.get(f) for v, f, _ in m.PAIRS]
            fam['same'].append((f'{model} | {mode} | {inst}', pvalue(pred, m.ZERO[model], same)))
            fam['different'].append((f'{model} | {mode} | {inst}', pvalue(pred, m.ZERO[model], [not s for s in same])))
    out = {k: holm(v) for k, v in fam.items()}
    for k, rows in out.items():
        print('==', k, len(rows))
        for r in rows[:4]:
            print(f"  {r['test']:50s} p={r['p']:.4f} p_holm={r['p_holm']:.4f} {'significant' if r['significant_after_holm'] else ''}")
    Path(__file__).with_suffix('.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()

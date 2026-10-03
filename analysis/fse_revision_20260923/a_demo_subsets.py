"""Mode A within subsets: pairs whose two members receive the same two demonstrations vs. different ones.

The demonstrations are read from the A prompt records themselves ('similar'[:2], the two
functions shown); the A prompts are the same for every model. For each model the change from the
author two-shot prompt to A is computed on the same pairs, overall and within each subset, with the
commit-cluster bootstrap (3000 draws, seed 20260920). The base prompt's P-C per subset shows how
easy the subset is without retrieval. Offline; no API calls.

Run from the repository root: python3 analysis/fse_revision_20260923/a_demo_subsets.py
"""
import collections
import json
import sys

import numpy as np

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

A_SRC = 'outdir/icl_fixed_baseline_compatible_test870/{}_std_cls_fewshotegTrue_top2_baseline_compatible.jsonl'
FILES = {  # same files as rq1_rq3_modes.py
    'gpt-5.5': ('data/gpt-5.5_std_cls_logprobsFalse_fewshotegTrue1_baseline_870.jsonl',
                'outdir/replay_a_icl_top2/gpt-5.5_std_cls_fewshotegTrue_ICL_870.jsonl'),
    'glm-5.1': ('outdir/glm-5.1_std_cls_logprobsFalse_fewshotegTrue_baseline_870.jsonl', A_SRC.format('glm-5.1')),
    'claude-opus-4-7': ('outdir/replay_fixed_fewshot/claude-opus-4-7_std_cls_logprobsFalse_fewshotegTrue_baseline_870.jsonl',
                        A_SRC.format('claude-opus-4-7')),
    'deepseek-v4-pro-0813': ('outdir/replay_fixed_fewshot/deepseek-v4-pro-0813_std_cls_logprobsFalse_fewshotegTrue_baseline_870.jsonl',
                             'outdir/replay_a_icl_top2/deepseek-v4-pro-0813_std_cls_fewshotegTrue_ICL_870.jsonl'),
}


def preds(path):
    m = {}
    for l in open(path):
        r = json.loads(l)
        k = str(r.get('query_idx', r.get('idx')))
        p = parse_prediction(r.get('response'))
        if p is not None or k not in m:
            m[k] = p
    return m


# Demonstrations actually shown, from the GLM-5.1 A records (the source of every model's A prompts).
DEMO = {}
for l in open(A_SRC.format('glm-5.1')):
    r = json.loads(l)
    DEMO[str(r['idx'])] = [str(s['idx']) for s in (r.get('similar') or [])[:r.get('num_examples_used', 2)]]
SAME = {(v, f): set(DEMO[v]) == set(DEMO[f]) for v, f, _ in PAIRS}
SUBSETS = {'all': lambda p: True, 'same demos': lambda p: SAME[p], 'different demos': lambda p: not SAME[p]}


def counts(m, keep):
    arr = np.zeros((len(clusters), 5))  # P-C P-V P-B P-R n
    for v, f, c in PAIRS:
        if (v, f) not in keep:
            continue
        k = {(1, 0): 0, (1, 1): 1, (0, 0): 2, (0, 1): 3}[(m[v], m[f])]
        arr[cid[c], k] += 1; arr[cid[c], 4] += 1
    return arr


def rates(t):
    n = np.where(t[..., 4] > 0, t[..., 4], 1)
    return {k: t[..., i] / n for i, k in enumerate(('P-C', 'P-V', 'P-B', 'P-R'))}


OUT = {'pairs': len(PAIRS), 'same_demo_pairs': sum(SAME.values()), 'models': {}}
for model, (base_path, a_path) in FILES.items():
    B, A = preds(base_path), preds(a_path)
    answered = {(v, f) for v, f, _ in PAIRS if all(M.get(x) is not None for M in (A, B) for x in (v, f))}
    res = {}
    for name, sel in SUBSETS.items():
        keep = {p for p in answered if sel(p)}
        ca, cb = counts(A, keep), counts(B, keep)
        ra, rb = rates(ca.sum(0)), rates(cb.sum(0))
        row = {'pairs': len(keep), 'base_P-C': round(float(rb['P-C']), 4), 'A_P-C': round(float(ra['P-C']), 4)}
        for met in ('P-C', 'P-R', 'P-B'):
            d = rates(W @ ca)[met] - rates(W @ cb)[met]
            row[met] = {'diff_points': round(100 * float(ra[met] - rb[met]), 2),
                        'ci95': [round(100 * float(q), 2) for q in np.quantile(d, [.025, .975])],
                        'p': round(float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))), 4)}
        res[name] = row
    OUT['models'][model] = res

json.dump(OUT, open('analysis/fse_revision_20260923/a_demo_subsets.json', 'w'), indent=1)
print(f"pairs {OUT['pairs']}, same demonstrations {OUT['same_demo_pairs']}")
for model, res in OUT['models'].items():
    print('==', model)
    for name, r in res.items():
        print(f"  {name:16s} n={r['pairs']:3d} base P-C {r['base_P-C']:.3f} A P-C {r['A_P-C']:.3f} | "
              + ' '.join(f"d{m} {r[m]['diff_points']:+5.1f} [{r[m]['ci95'][0]:+.1f},{r[m]['ci95'][1]:+.1f}]" for m in ('P-C', 'P-R', 'P-B')))

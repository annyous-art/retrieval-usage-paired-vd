"""Run-to-run variation of the one significant comparison: GLM-5.1, retrieved demonstrations (A, two)
vs. the PrimeVul two-shot prompt, each sent twice with identical prompts (run 1: the runs of Table 3;
run 2: experiments/run_glm_repeat.sh). Reports, on the 431 resolvable pairs, per run: P-C, P-B, and the
A - two-shot difference with the commit-cluster bootstrap (3000 draws, seed 20260920); the share of
functions whose label changes between the runs; and the change of P-C between runs of the same prompt.
Section 7 (Threats to Validity) reports the share of changed answers (13% and 21%), the change of P-C
between the runs (4.2 and 4.9 points), and the A - two-shot difference of each run (4.4 and 5.1 points).
Offline; no API calls.

Run from the repository root: python3 analysis/fse_revision_20260923/glm_repeat.py
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
RUNS = {'two_shot': ('outdir/glm-5.1_std_cls_logprobsFalse_fewshotegTrue_baseline_870.jsonl', 'outdir/glm_repeat_run2/glm-5.1_two_shot.jsonl'),
        'A': ('outdir/icl_fixed_baseline_compatible_test870/glm-5.1_std_cls_fewshotegTrue_top2_baseline_compatible.jsonl', 'outdir/glm_repeat_run2/glm-5.1_A_top2.jsonl')}


def preds(path):
    return {str(r.get('query_idx', r.get('idx'))): parse_prediction(r.get('response')) for r in map(json.loads, open(path))}


def counts(m):
    arr = np.zeros((len(clusters), 5))
    for v, f, c in PAIRS:
        if m.get(v) is None or m.get(f) is None:
            continue
        arr[cid[c], {(1, 0): 0, (1, 1): 1, (0, 0): 2, (0, 1): 3}[(m[v], m[f])]] += 1; arr[cid[c], 4] += 1
    return arr


def rates(t):
    n = np.where(t[..., 4] > 0, t[..., 4], 1)
    return {k: t[..., i] / n for i, k in enumerate(('P-C', 'P-V', 'P-B', 'P-R'))}


def diff(ma, mb):
    ca, cb = counts(ma), counts(mb)
    out = {}
    for met in ('P-C', 'P-B'):
        d = rates(W @ ca)[met] - rates(W @ cb)[met]
        out[met] = {'points': round(100 * float(rates(ca.sum(0))[met] - rates(cb.sum(0))[met]), 2),
                    'ci95': [round(100 * float(q), 2) for q in np.quantile(d, [.025, .975])],
                    'p': round(float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))), 4)}
    return out


M = {name: [preds(p) for p in paths] for name, paths in RUNS.items()}
fx = [x for v, f, _ in PAIRS for x in (v, f)]
OUT = {'pairs': len(PAIRS), 'runs': {}, 'between_runs': {}}
for r in (0, 1):
    t = {name: rates(counts(M[name][r]).sum(0)) for name in M}
    OUT['runs'][f'run{r + 1}'] = {'two_shot': {k: round(float(v), 4) for k, v in t['two_shot'].items()},
                                  'A': {k: round(float(v), 4) for k, v in t['A'].items()},
                                  'A - two_shot': diff(M['A'][r], M['two_shot'][r])}
for name in M:
    a, b = M[name]
    both = [x for x in fx if a.get(x) is not None and b.get(x) is not None]
    OUT['between_runs'][name] = {'functions': len(both), 'label_changed': round(sum(a[x] != b[x] for x in both) / len(both), 4),
                                 'run2 - run1': diff(b, a)}
json.dump(OUT, open('analysis/fse_revision_20260923/glm_repeat.json', 'w'), indent=1)
for r, x in OUT['runs'].items():
    d = x['A - two_shot']
    print(f"{r}: two-shot P-C {x['two_shot']['P-C']:.3f} P-B {x['two_shot']['P-B']:.3f} | A P-C {x['A']['P-C']:.3f} P-B {x['A']['P-B']:.3f} | "
          f"dP-C {d['P-C']['points']:+.1f} {d['P-C']['ci95']} p {d['P-C']['p']}  dP-B {d['P-B']['points']:+.1f} {d['P-B']['ci95']}")
for name, x in OUT['between_runs'].items():
    d = x['run2 - run1']
    print(f"{name}: labels changed {100 * x['label_changed']:.1f}% of {x['functions']}; run2-run1 dP-C {d['P-C']['points']:+.1f} {d['P-C']['ci95']} dP-B {d['P-B']['points']:+.1f}")

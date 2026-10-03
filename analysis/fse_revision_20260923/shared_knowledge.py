"""Mode D with a shared knowledge source: GLM-5.1 and DeepSeek-V4-Pro detect with GPT-5.5's or Claude Opus 4.7's knowledge.

In the fixed-candidate setting each model normally detects with the knowledge it extracted
itself. experiments/knowledge_rag/next_stage/shared_knowledge_arm.py sends the two knowledge
arms (all fields; without the fixing condition) of GPT-5.5 or of Claude Opus 4.7 to GLM-5.1 and
DeepSeek-V4-Pro, with their own inference configuration; all other prompts are identical across
models. For each of the two models and each knowledge source (GPT-5.5 or Claude Opus 4.7) this
compares, on the 431 resolvable pairs and with the commit-cluster bootstrap (3000 draws, seed
20260920):
  shared knowledge - complete code pair      (does knowledge still trail the code pair?)
  shared knowledge - own knowledge           (does the knowledge source matter?)
and the same two for the arms without the fixing condition. Holm correction within each model.
Offline; no API calls.

Run from the repository root: python3 analysis/fse_revision_20260923/shared_knowledge.py
"""
import collections
import json
import sys

import numpy as np

sys.path.insert(0, 'experiments/knowledge_rag')
import experiment as fc_exp  # noqa: E402  (the fixed-candidate framework's own parser)

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

MODELS = {'glm-5.1': 'outdir/knowledge_rag_glm-5.1/predictions.jsonl',
          'deepseek-v4-pro-0813': 'outdir/knowledge_rag_deepseek-v4-pro-0813/predictions.jsonl'}
SOURCES = ('gpt55', 'claude47')  # models whose extracted knowledge is shared
CMP = [(f'{arm}_{s}', ref) for s in SOURCES for arm, ref in (('knowledge', 'code_pair'), ('knowledge', 'knowledge'),
                                                           ('knowledge_no_fix', 'code_pair'), ('knowledge_no_fix', 'knowledge_no_fix'))]
CMP = [(a, 'knowledge' if r == 'knowledge' else r) for a, r in CMP]


def preds(path, arm):
    m = {}
    for l in open(path):
        r = json.loads(l)
        if r.get('arm') == arm:
            m[str(r['idx'])] = fc_exp.prediction(r.get('response'))
    return m


def counts(m):
    arr = np.zeros((len(clusters), 5))  # P-C P-V P-B P-R n
    for v, f, c in PAIRS:
        a, b = m.get(v), m.get(f)
        if a is None or b is None:
            continue
        arr[cid[c], {(1, 0): 0, (1, 1): 1, (0, 0): 2, (0, 1): 3}[(a, b)]] += 1; arr[cid[c], 4] += 1
    return arr


def rates(t):
    n = np.where(t[..., 4] > 0, t[..., 4], 1)
    return {k: t[..., i] / n for i, k in enumerate(('P-C', 'P-V', 'P-B', 'P-R'))}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i])); adj[i] = run
    return adj


OUT = {'pairs': len(PAIRS), 'models': {}}
for model, own in MODELS.items():
    shared = {'gpt55': f'outdir/shared_knowledge_{model}/predictions.jsonl',
              'claude47': f'outdir/shared_knowledge_claude47_{model}/predictions.jsonl'}
    M = {arm: preds(own, arm) for arm in ('no_retrieval', 'code_pair', 'knowledge', 'knowledge_no_fix')}
    M.update({f'{arm}_{s}': preds(shared[s], f'{arm}_{s}') for s in SOURCES for arm in ('knowledge', 'knowledge_no_fix')})
    res = {'methods': {}, 'comparisons': {}}
    for arm, m in M.items():
        t = counts(m).sum(0)
        res['methods'][arm] = {'pairs': int(t[4]), **{k: round(float(v), 4) for k, v in rates(t).items()}}
    for a, b in CMP:
        both = {x for v, f, _ in PAIRS for x in (v, f) if M[a].get(x) is not None and M[b].get(x) is not None}
        ca = counts({k: v for k, v in M[a].items() if k in both}); cb = counts({k: v for k, v in M[b].items() if k in both})
        row = {}
        for met in ('P-C', 'P-B'):
            d = rates(W @ ca)[met] - rates(W @ cb)[met]
            row[met] = {'diff_points': round(100 * float(rates(ca.sum(0))[met] - rates(cb.sum(0))[met]), 2),
                        'ci95': [round(100 * float(q), 2) for q in np.quantile(d, [.025, .975])],
                        'p': round(float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))), 4)}
        res['comparisons'][f'{a} - {b}'] = row
    keys = list(res['comparisons'])
    for k, adj in zip(keys, holm([res['comparisons'][k]['P-C']['p'] for k in keys])):
        res['comparisons'][k]['P-C']['p_holm'] = round(adj, 4)
    OUT['models'][model] = res

json.dump(OUT, open('analysis/fse_revision_20260923/shared_knowledge.json', 'w'), indent=1)
for model, res in OUT['models'].items():
    print('==', model)
    for arm, r in res['methods'].items():
        print(f"  {arm:24s} n={r['pairs']} P-C {r['P-C']:.3f} P-V {r['P-V']:.3f} P-B {r['P-B']:.3f} P-R {r['P-R']:.3f}")
    for k, c in res['comparisons'].items():
        print(f"  {k:42s} dP-C {c['P-C']['diff_points']:+.1f} {c['P-C']['ci95']} p_holm {c['P-C']['p_holm']}  dP-B {c['P-B']['diff_points']:+.1f} {c['P-B']['ci95']}")

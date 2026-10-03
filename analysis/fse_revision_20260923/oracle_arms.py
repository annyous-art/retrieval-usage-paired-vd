"""Control experiment of RQ2 (Figure 3b): the pair's own fix (oracle) and wrong-evidence arms of the fixed-candidate setting.

For each model with a complete run: pair outcomes on the 431 resolvable pairs for the no-retrieval
reference, the retrieved candidate (code pair C, extracted knowledge D), and the three arms of
experiments/knowledge_rag/next_stage/oracle_arm.py: the target pair's own code pair, knowledge
extracted from the target pair's own fix, and the code pair of another test pair. Differences use
the pair-preserving commit-cluster bootstrap (3000 draws, seed 20260920) with Holm correction over
the listed P-C comparisons of each model, as in rq1_rq3_modes.py. Predictions are parsed with the
fixed-candidate framework's own parser. Percentages are computed from counts and rounded once.

Run from the repository root: python3 analysis/fse_revision_20260923/oracle_arms.py
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, 'experiments/knowledge_rag')
import experiment as fc_exp  # noqa: E402

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

MODELS = {  # model: (predictions of the original fixed-candidate run, directory of the oracle arms)
    'gpt-5.5': ('outdir/knowledge_rag_gpt-5.5/predictions.jsonl', 'outdir/oracle_fc_gpt-5.5/{}/'),
    'glm-5.1': ('outdir/knowledge_rag_glm-5.1/predictions.jsonl', 'outdir/oracle_fc_glm-5.1/{}/'),
}
ARMS = ['oracle_code_pair', 'oracle_knowledge', 'wrong_code_pair']
CMP = [('oracle_code_pair', 'no_retrieval'), ('oracle_knowledge', 'no_retrieval'), ('wrong_code_pair', 'no_retrieval'),
       ('oracle_code_pair', 'code_pair'), ('oracle_knowledge', 'knowledge'), ('wrong_code_pair', 'code_pair'),
       ('oracle_knowledge', 'oracle_code_pair')]
ORDER = ('no_retrieval', 'code_pair', 'knowledge', 'wrong_code_pair', 'oracle_code_pair', 'oracle_knowledge')
LABEL = {'no_retrieval': 'No retrieval (reference)', 'code_pair': 'Retrieved candidate, code pair (C)',
         'knowledge': 'Retrieved candidate, extracted knowledge (D)', 'wrong_code_pair': 'Code pair of another test pair',
         'oracle_code_pair': "Target pair's own code pair (oracle)", 'oracle_knowledge': "Knowledge of the target pair's own fix (oracle)"}


def source_preds(path, arm):
    m = {}
    for l in open(path):
        r = json.loads(l)
        if r.get('arm') == arm:
            m[str(r['idx'])] = fc_exp.prediction(r.get('response'))
    return m


def oracle_preds(d):
    jobs = {j['job_id']: j for j in map(json.loads, open(d + 'jobs.jsonl'))}
    cfg = json.load(open(d + 'config.json'))
    m, evid = {}, {}
    for l in open(d + 'predictions.jsonl'):
        r = json.loads(l)
        j = jobs[r['job_id']]
        m[str(j['idx'])] = fc_exp.prediction(r.get('response'))
        evid[j['pair_id']] = j['evidence_pair']
    return m, {'jobs': len(jobs), 'answers': len(m), 'config_model': cfg['config']['model'],
               'derangement_shift': cfg.get('derangement_shift'),
               'evidence_from_own_pair': sum(p == e for p, e in evid.items()), 'pairs': len(evid)}


def counts(m):
    arr = np.zeros((len(clusters), 7))
    for v, f, c in PAIRS:
        a, b = m.get(v), m.get(f)
        if a is None or b is None:
            continue
        k = {(1, 0): 0, (1, 1): 1, (0, 0): 2, (0, 1): 3}[(a, b)]
        arr[cid[c], k] += 1; arr[cid[c], 4] += a; arr[cid[c], 5] += b; arr[cid[c], 6] += 1
    return arr


def rates(t):
    n = np.where(t[..., 6] > 0, t[..., 6], 1)
    return {'P-C': t[..., 0] / n, 'P-V': t[..., 1] / n, 'P-B': t[..., 2] / n, 'P-R': t[..., 3] / n,
            'YES': (t[..., 4] + t[..., 5]) / (2 * n)}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i])); adj[i] = run
    return adj


def compare(A, B):
    both = {x for v, f, _ in PAIRS for x in (v, f) if A.get(x) is not None and B.get(x) is not None}
    ca = counts({k: v for k, v in A.items() if k in both}); cb = counts({k: v for k, v in B.items() if k in both})
    out = {'pairs': int(ca[:, 6].sum())}
    for met in ('P-C', 'P-V', 'P-B', 'P-R'):
        point = rates(ca.sum(0))[met] - rates(cb.sum(0))[met]
        d = rates(W @ ca)[met] - rates(W @ cb)[met]
        out[met] = {'diff_points': round(100 * float(point), 2),
                    'ci95': [round(100 * float(q), 2) for q in np.quantile(d, [.025, .975])],
                    'p': round(float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))), 4)}
    return out


OUT = {'pairs': len(PAIRS), 'models': {}}
for model, (source, oracle_dir) in MODELS.items():
    done = {arm: os.path.exists(oracle_dir.format(arm) + 'arm_metrics.json') for arm in ARMS}
    if not all(done.values()):
        print(f'== {model}: incomplete arms {[a for a, d in done.items() if not d]}; skipped')
        continue
    M = {arm: source_preds(source, arm) for arm in ('no_retrieval', 'code_pair', 'knowledge')}
    checks = {}
    for arm in ARMS:
        M[arm], checks[arm] = oracle_preds(oracle_dir.format(arm))
    res = {'checks': checks, 'methods': {}}
    for name, m in M.items():
        t = counts(m).sum(0)
        res['methods'][name] = {'answered_pairs': int(t[6]),
                                'unknown_functions': sum(m.get(x) is None for v, f, _ in PAIRS for x in (v, f)),
                                'counts': {k: int(t[i]) for i, k in enumerate(('P-C', 'P-V', 'P-B', 'P-R'))},
                                **{k: 100 * float(v) for k, v in rates(t).items()}}  # unrounded; round once when printing
    stats = {f'{a} - {b}': compare(M[a], M[b]) for a, b in CMP}
    for k, adj in zip(stats, holm([s['P-C']['p'] for s in stats.values()])):
        stats[k]['P-C']['p_holm'] = round(adj, 4)
    res['comparisons'] = stats
    OUT['models'][model] = res
    print(f'== {model}', json.dumps({a: (c['answers'], c['evidence_from_own_pair'], c['derangement_shift']) for a, c in checks.items()}))
    for name in ORDER:
        r = res['methods'][name]
        print(f"{LABEL[name]} & {r['P-C']:.1f} & {r['P-V']:.1f} & {r['P-B']:.1f} & {r['P-R']:.1f} \\\\   "
              f"(pairs {r['answered_pairs']}, unknown {r['unknown_functions']}, YES {r['YES']:.1f})")
    for k, s in stats.items():
        print(f"{k:38s} dP-C {s['P-C']['diff_points']:+.2f} {s['P-C']['ci95']} p_holm {s['P-C']['p_holm']}  dP-B {s['P-B']['diff_points']:+.2f}")
json.dump(OUT, open('analysis/fse_revision_20260923/oracle_arms.json', 'w'), indent=1)

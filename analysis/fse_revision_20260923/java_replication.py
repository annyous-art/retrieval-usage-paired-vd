"""Second dataset: modes A and D on the MegaVul Java paired split (data/second_dataset/, built by
data/second_dataset/build/build_megavul_java_pairs.py). Same prompts, parser, metrics and statistics as
for PrimeVul: std setting (PrimeVul two-shot vs. two retrieved demonstrations) and fixed-candidate setting
(no retrieval, complete code pair, extracted knowledge with and without the fixing condition). Pair
outcomes, YES rate, the share of pairs whose members receive the same retrieved items (same two
demonstrations; same historical candidate), and differences with the commit-cluster bootstrap (3000 draws,
seed 20260920), Holm within each model's std and FC families. For mode A, the P-C change is also split by
whether the two members are shown the same two demonstrations. Offline; no API calls.

Run from the repository root: python3 analysis/fse_revision_20260923/java_replication.py
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'experiments/knowledge_rag')
from evaluate_prompt_outputs import parse_prediction  # noqa: E402
import experiment as fc_exp  # noqa: E402

RAW = [json.loads(l) for l in open('data/second_dataset/megavul_java_test_paired_labeled.jsonl')]
PAIRS = []
for i in range(0, len(RAW), 2):
    a, b = RAW[i], RAW[i + 1]
    v, f = (a, b) if int(a['target']) == 1 else (b, a)
    PAIRS.append((str(v['idx']), str(f['idx']), f"{v['project']}:{v['commit_id']}"))
clusters = sorted({c for *_, c in PAIRS})
cid = {c: i for i, c in enumerate(clusters)}
W = np.random.default_rng(20260920).multinomial(len(clusters), np.ones(len(clusters)) / len(clusters), size=3000)
D = 'outdir/java_std'
MODELS = ('gpt-5.5', 'glm-5.1')
STD = [('A', 'two_shot')]
FC = [('code_pair', 'no_retrieval'), ('knowledge', 'no_retrieval'), ('knowledge_no_fix', 'no_retrieval'), ('knowledge', 'code_pair')]


def std_preds(path):
    m = {}
    for l in open(path):
        r = json.loads(l)
        m[str(r.get('query_idx', r.get('idx')))] = parse_prediction(r.get('response'))
    return m


def fc_preds(path, arm):
    return {str(r['idx']): fc_exp.prediction(r.get('response')) for r in map(json.loads, open(path)) if r.get('arm') == arm}


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


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i])); adj[i] = run
    return adj


def compare(M, a, b):
    both = {x for v, f, _ in PAIRS for x in (v, f) if M[a].get(x) is not None and M[b].get(x) is not None}
    ca = counts({k: v for k, v in M[a].items() if k in both}); cb = counts({k: v for k, v in M[b].items() if k in both})
    row = {}
    for met in ('P-C', 'P-B'):
        d = rates(W @ ca)[met] - rates(W @ cb)[met]
        row[met] = {'diff_points': round(100 * float(rates(ca.sum(0))[met] - rates(cb.sum(0))[met]), 2),
                    'ci95': [round(100 * float(q), 2) for q in np.quantile(d, [.025, .975])],
                    'p': round(float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))), 4)}
    return row


demo = {str(r['idx']): frozenset(str(s['idx']) for s in r['similar'][:2]) for r in map(json.loads, open(f'{D}/prompts_A_top2.jsonl'))}
cand = {str(r['idx']): tuple(r['candidate_ids']) for r in map(json.loads, open('outdir/java_fc_glm-5.1/targets.jsonl'))}
OUT = {'pairs': len(PAIRS), 'clusters': len(clusters),
       'same_items': {'A (two demonstrations)': round(sum(demo[v] == demo[f] for v, f, _ in PAIRS) / len(PAIRS), 4),
                      'FC candidate (C, D)': round(sum(cand[v] == cand[f] for v, f, _ in PAIRS) / len(PAIRS), 4)},
       'models': {}}
for model in MODELS:
    files = {'two_shot': f'{D}/{model}_two_shot.jsonl', 'A': f'{D}/{model}_A_top2.jsonl'}
    fc = f'outdir/java_fc_{model}/predictions.jsonl'
    if not all(os.path.exists(p) for p in files.values()) or not os.path.exists(fc):
        continue
    M = {k: std_preds(p) for k, p in files.items()}
    M.update({arm: fc_preds(fc, arm) for arm in ('no_retrieval', 'code_pair', 'knowledge', 'knowledge_no_fix')})
    res = {'methods': {}, 'comparisons': {}}
    for name, m in M.items():
        t = counts(m).sum(0)
        yes = [m[x] for v, f, _ in PAIRS for x in (v, f) if m.get(x) is not None]
        res['methods'][name] = {'pairs': int(t[4]), **{k: round(float(x), 4) for k, x in rates(t).items()},
                                'YES': round(sum(yes) / len(yes), 4)}
    for fam in (STD, FC):
        keys = [f'{a} - {b}' for a, b in fam]
        for (a, b), k in zip(fam, keys):
            res['comparisons'][k] = compare(M, a, b)
        for k, adj in zip(keys, holm([res['comparisons'][k]['P-C']['p'] for k in keys])):
            res['comparisons'][k]['P-C']['p_holm'] = round(adj, 4)
    res['A_by_demonstrations'] = {}
    for name, same in (('same demonstrations', True), ('different demonstrations', False)):
        keep = {x for v, f, _ in PAIRS if (demo[v] == demo[f]) == same for x in (v, f)}
        sub = {k: {x: y for x, y in M[k].items() if x in keep} for k in ('A', 'two_shot')}
        added = sum((sub['A'].get(v), sub['A'].get(f)) == (1, 0) for v, f, _ in PAIRS) - sum((sub['two_shot'].get(v), sub['two_shot'].get(f)) == (1, 0) for v, f, _ in PAIRS)
        res['A_by_demonstrations'][name] = {'pairs': sum((demo[v] == demo[f]) == same for v, f, _ in PAIRS), 'added_separated_pairs': added, **compare(sub, 'A', 'two_shot')}
    OUT['models'][model] = res
json.dump(OUT, open('analysis/fse_revision_20260923/java_replication.json', 'w'), indent=1)
print('pairs', OUT['pairs'], 'clusters', OUT['clusters'], 'same items', OUT['same_items'])
for model, res in OUT['models'].items():
    print('==', model)
    for n, r in res['methods'].items():
        print(f"  {n:18s} n={r['pairs']} P-C {r['P-C']:.3f} P-V {r['P-V']:.3f} P-B {r['P-B']:.3f} P-R {r['P-R']:.3f} YES {r['YES']:.3f}")
    for k, c in res['comparisons'].items():
        print(f"  {k:32s} dP-C {c['P-C']['diff_points']:+.1f} {c['P-C']['ci95']} p_holm {c['P-C']['p_holm']}  dP-B {c['P-B']['diff_points']:+.1f} {c['P-B']['ci95']}")
    for k, c in res['A_by_demonstrations'].items():
        print(f"  A vs two-shot, {k:24s} pairs {c['pairs']} added {c['added_separated_pairs']:+d} dP-C {c['P-C']['diff_points']:+.1f} {c['P-C']['ci95']}")

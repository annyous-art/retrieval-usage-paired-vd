"""Offline localization analyses for the focus region (no API calls, no model).

In deployment the detector sees one function and no patch, so the only way retrieval
can separate a vulnerable function from its fixed version is if the focus region it
picks lands on the changed lines. This script measures how often that happens.

 A. Retrieval-selected focus vs. a random chunk: the random hit probability of a
    function is (#chunks containing a changed line) / (#chunks), with chunks rebuilt
    exactly as the pipeline builds them (six valid lines, stride two).
 B. Coverage at k: top-1 and top-2 from the stored ranking (records keep two focus
    candidates), and the exact random expectation for larger k.
 C. Pair outcomes when the focus hits vs. misses the patch, with differences from a
    pair-preserving commit-cluster bootstrap (3000 draws, seed 20260920).

Run from the repository root: python3 analysis/fse_revision_20260923/focus_localization.py
"""
import collections
import json
import sys
from math import comb

import numpy as np

sys.path.insert(0, '.')
from build_query_chunks_sim_rank import build_chunks  # noqa: E402
from evaluate_prompt_outputs import parse_prediction  # noqa: E402

RAW = [json.loads(l) for l in open('data/primevul_test_paired_labeled.jsonl')]
BY_IDX = {}
for r in RAW:
    BY_IDX.setdefault(str(r['idx']), r)
DUP = collections.Counter(str(r['idx']) for r in RAW)
PAIRS = []  # (vulnerable idx, fixed idx, commit cluster)
for i in range(0, len(RAW), 2):
    a, b = RAW[i], RAW[i + 1]
    if DUP[str(a['idx'])] > 1 or DUP[str(b['idx'])] > 1:
        continue
    v, f = (a, b) if int(a['target']) == 1 else (b, a)
    PAIRS.append((str(v['idx']), str(f['idx']), f"{v['project']}:{v['commit_id']}"))

METHODS = {
    'positive_weighted': 'outdir/prompt_positive_weighted_rag_glm51_test870/glm-5.1_std_cls_inline_function_label_rag_fewshotegTrue_cls0.jsonl',
    'balanced_bc_grouped_no_label': 'outdir/prompt_baseline_compatible_glm51_test870/glm-5.1_std_cls_grouped_no_label_baseline_compatible_fewshotegTrue_cls0.jsonl',
    'patch_contrast': 'outdir/prompt_patch_contrast_vuln_margin_glm51_test870/glm-5.1_std_cls_patch_contrast_fewshotegTrue.jsonl',
}
OUT = {'pairs': len(PAIRS), 'clusters': len({c for *_, c in PAIRS})}

# ---------- chunk inventory per function ----------
CHUNKS = {}
for idx in {x for v, f, _ in PAIRS for x in (v, f)}:
    ch, _, _ = build_chunks(BY_IDX[idx], 6, 2, 0)
    CHUNKS[idx] = ch
N = {k: len(v) for k, v in CHUNKS.items()}
M = {k: sum(c['chunk_label'] for c in v) for k, v in CHUNKS.items()}


def p_random(idx, k=1):
    """Probability that k distinct random chunks include at least one changed-line chunk."""
    n, m = N[idx], M[idx]
    if n == 0 or m == 0:
        return 0.0
    k = min(k, n)
    return 1 - comb(n - m, k) / comb(n, k)


def ranked(rec):
    qs = sorted(rec['query_chunks'], key=lambda q: q.get('query_rank', 0))
    return [q.get('query_chunk', q) for q in qs]


def mean(xs):
    return round(float(np.mean(xs)), 4)


funcs = [x for v, f, _ in PAIRS for x in (v, f)]
OUT['inventory'] = {
    'functions': len(funcs),
    'with_changed_lines': sum(M[x] > 0 for x in funcs),
    'vulnerable_with_changed_lines': sum(M[v] > 0 for v, _, _ in PAIRS),
    'fixed_with_changed_lines': sum(M[f] > 0 for _, f, _ in PAIRS),
    'chunks_per_function_median': float(np.median([N[x] for x in funcs])),
    'changed_chunk_share_median': round(float(np.median([M[x] / N[x] for x in funcs if N[x]])), 4),
}
LEN_EDGES = np.quantile([N[x] for x in funcs], [1 / 3, 2 / 3])


def length_bin(idx):
    n = N[idx]
    return 'short' if n <= LEN_EDGES[0] else ('medium' if n <= LEN_EDGES[1] else 'long')


OUT['length_bins_chunks'] = [float(e) for e in LEN_EDGES]

# ---------- A + B ----------
rng = np.random.default_rng(20260920)
clusters = sorted({c for *_, c in PAIRS})
cid = {c: i for i, c in enumerate(clusters)}
W = rng.multinomial(len(clusters), np.ones(len(clusters)) / len(clusters), size=3000)


def cluster_mean_ci(values, cl):
    """Mean over units and a 95% interval from resampling commit clusters."""
    values = np.asarray(values, float)
    s = np.zeros(len(clusters)); n = np.zeros(len(clusters))
    for x, c in zip(values, cl):
        s[cid[c]] += x; n[cid[c]] += 1
    boot = (W @ s) / (W @ n)
    return [round(values.mean(), 4), [round(q, 4) for q in np.quantile(boot, [.025, .975])]]


LOC, PAIROUT = {}, {}
for tag, path in METHODS.items():
    recs = {str(r.get('query_idx', r.get('idx'))): r for r in (json.loads(l) for l in open(path))}
    # sanity check: the stored focus is one of the rebuilt chunks (by its code; by its line numbers when
    # the records are the code-free ones of the replication package)
    key = lambda c: c['code_clean'] if 'code_clean' in ranked(recs[funcs[0]])[0] else tuple(c['chunk_line_ids'])
    miss = sum(key(ranked(recs[x])[0]) not in {key(c) for c in CHUNKS[x]} for x in funcs)
    hit1 = {x: int(ranked(recs[x])[0]['chunk_label']) == 1 for x in funcs}
    hit2 = {x: any(int(q['chunk_label']) == 1 for q in ranked(recs[x])[:2]) for x in funcs}
    fcl = [c for v, f, c in PAIRS for _ in (v, f)]
    pair_hit = [hit1[v] or hit1[f] for v, f, _ in PAIRS]
    pair_rand = [1 - (1 - p_random(v)) * (1 - p_random(f)) for v, f, _ in PAIRS]
    pcl = [c for *_, c in PAIRS]
    loc = {
        'focus_not_in_rebuilt_chunks': miss,
        'function_top1_hit': cluster_mean_ci([hit1[x] for x in funcs], fcl),
        'function_top1_random': cluster_mean_ci([p_random(x) for x in funcs], fcl),
        'function_top1_hit_minus_random': cluster_mean_ci([hit1[x] - p_random(x) for x in funcs], fcl),
        'vulnerable_top1_hit': mean([hit1[v] for v, _, _ in PAIRS]),
        'vulnerable_top1_random': mean([p_random(v) for v, _, _ in PAIRS]),
        'fixed_top1_hit': mean([hit1[f] for _, f, _ in PAIRS]),
        'fixed_top1_random': mean([p_random(f) for _, f, _ in PAIRS]),
        'pair_either_top1_hit': cluster_mean_ci(pair_hit, pcl),
        'pair_either_top1_random_independent': cluster_mean_ci(pair_rand, pcl),
        'coverage_at_k': {
            'retrieval_top1': mean(list(hit1.values())),
            'retrieval_top2': mean(list(hit2.values())),
            **{f'random_top{k}': mean([p_random(x, k) for x in funcs]) for k in (1, 2, 3, 5, 10, 20)},
        },
        'by_length': {
            b: {'functions': sum(length_bin(x) == b for x in funcs),
                'top1_hit': mean([hit1[x] for x in funcs if length_bin(x) == b]),
                'random': mean([p_random(x) for x in funcs if length_bin(x) == b])}
            for b in ('short', 'medium', 'long')
        },
    }
    LOC[tag] = loc

    # ---------- C ----------
    rows = []
    for v, f, c in PAIRS:
        pv, pf = parse_prediction(recs[v].get('response')), parse_prediction(recs[f].get('response'))
        o = {(1, 0): 'P-C', (1, 1): 'P-V', (0, 0): 'P-B', (0, 1): 'P-R'}.get((pv, pf), 'unknown')
        rows.append((hit1[v] or hit1[f], o, c))
    res = {}
    for name in ('P-C', 'P-V', 'P-B', 'P-R'):
        s = np.zeros((2, len(clusters))); n = np.zeros((2, len(clusters)))
        for h, o, c in rows:
            s[int(h), cid[c]] += o == name; n[int(h), cid[c]] += 1
        r_hit, r_miss = s[1].sum() / n[1].sum(), s[0].sum() / n[0].sum()
        bh, bm = (W @ s[1]) / (W @ n[1]), (W @ s[0]) / (W @ n[0])
        res[name] = {'hit': round(r_hit, 4), 'miss': round(r_miss, 4),
                     'hit_minus_miss': round(r_hit - r_miss, 4),
                     'ci95': [round(q, 4) for q in np.quantile(bh - bm, [.025, .975])]}
    res['pairs_hit'] = sum(h for h, _, _ in rows)
    res['pairs_miss'] = sum(not h for h, _, _ in rows)
    # Short functions are hit far more often; check P-C hit vs. miss within length bins
    # (bin of the vulnerable member).
    res['P-C_by_length'] = {}
    for b in ('short', 'medium', 'long'):
        sub = [(h, o) for (h, o, _), (v, _, _) in zip(rows, PAIRS) if length_bin(v) == b]
        hit = [o == 'P-C' for h, o in sub if h]
        mis = [o == 'P-C' for h, o in sub if not h]
        res['P-C_by_length'][b] = {'hit': [sum(hit), len(hit)], 'miss': [sum(mis), len(mis)]}
    # Length-standardised difference: bin-wise differences weighted by pairs per bin.
    num = den = 0
    for b, x in res['P-C_by_length'].items():
        if x['hit'][1] and x['miss'][1]:
            w = x['hit'][1] + x['miss'][1]
            num += w * (x['hit'][0] / x['hit'][1] - x['miss'][0] / x['miss'][1]); den += w
    res['P-C_hit_minus_miss_length_adjusted'] = round(num / den, 4)
    PAIROUT[tag] = res

OUT['A_B_localization'] = LOC
OUT['C_pair_outcomes_by_focus_hit'] = PAIROUT
json.dump(OUT, open('analysis/fse_revision_20260923/focus_localization.json', 'w'), indent=1)
print(json.dumps(OUT, indent=1))

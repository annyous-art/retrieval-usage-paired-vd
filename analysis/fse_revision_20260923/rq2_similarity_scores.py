"""RQ2: retrieval similarity scores of the two members of a pair (offline, no API calls).

For every usage mode whose records store retrieval scores, we compare the scores the
vulnerable and the fixed member receive:
  * |delta| of the top-1 score and of the mean top-k score within a pair, against a
    between-pair baseline (the same statistic for two unrelated functions: the
    vulnerable members of two different pairs, matched by a fixed random derangement);
  * for scores that are meant to lean toward the vulnerable side (BC: best vulnerable
    evidence minus best safe/fixed evidence; patch-contrast: similarity to the
    historical vulnerable side minus the fixed side), how often the vulnerable member
    scores higher than the fixed member. 50% means the score does not order the pair.
Usage modes without stored scores (function-level retrieval, Vul-RAG) are reported by
ranking agreement in vulrag_evidence_identity.py and evidence_mechanism.py.

Run from the repository root: python3 analysis/fse_revision_20260923/rq2_similarity_scores.py
"""
import collections
import json

import numpy as np

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
rng = np.random.default_rng(20260920)
W = rng.multinomial(len(clusters), np.ones(len(clusters)) / len(clusters), size=3000)
perm = rng.permutation(len(PAIRS))
while np.any(perm == np.arange(len(PAIRS))):          # derangement: never pair a function with itself
    perm = rng.permutation(len(PAIRS))

RUNS = {
    'B positive-weighted': ('chunk', 'outdir/prompt_positive_weighted_rag_glm51_test870/glm-5.1_std_cls_inline_function_label_rag_fewshotegTrue_cls0.jsonl'),
    'B balanced BC': ('chunk', 'outdir/prompt_baseline_compatible_glm51_test870/glm-5.1_std_cls_grouped_no_label_baseline_compatible_fewshotegTrue_cls0.jsonl'),
    'C patch-contrast': ('patch', 'outdir/prompt_patch_contrast_vuln_margin_glm51_test870/glm-5.1_std_cls_patch_contrast_fewshotegTrue.jsonl'),
}


def focus(rec):
    return sorted(rec['query_chunks'], key=lambda q: q.get('query_rank', 0))[0]


def scores(kind, rec):
    q = focus(rec)
    if kind == 'chunk':
        m = sorted(q.get('index_matches') or [], key=lambda x: x.get('index_rank', 999))
        s = [x['faiss_score'] for x in m]
        return {'top1': s[0] if s else np.nan, 'meank': float(np.mean(s)) if s else np.nan,
                'lean': q.get('pos_neg_margin', np.nan)}
    m = sorted(q.get('patch_pair_matches') or [], key=lambda x: x.get('patch_pair_rank', 999))
    s = [x['pair_relevance'] for x in m]
    return {'top1': s[0] if s else np.nan, 'meank': float(np.mean(s)) if s else np.nan,
            'lean': q.get('top_contrast_margin', np.nan)}


def ci_mean(x, cl):
    x = np.asarray(x, float); s = np.zeros(len(clusters)); n = np.zeros(len(clusters))
    for val, c in zip(x, cl):
        s[cid[c]] += val; n[cid[c]] += 1
    b = (W @ s) / (W @ n)
    return [round(float(x.mean()), 4), [round(float(q), 4) for q in np.quantile(b, [.025, .975])]]


OUT = {'pairs': len(PAIRS), 'runs': {}}
for name, (kind, path) in RUNS.items():
    recs = {str(r.get('query_idx', r.get('idx'))): r for r in (json.loads(l) for l in open(path))}
    S = {x: scores(kind, recs[x]) for v, f, _ in PAIRS for x in (v, f)}
    res = {}
    res['functions_without_retrieved_items'] = int(sum(np.isnan(S[x]['top1']) for v, f, _ in PAIRS for x in (v, f)))
    for key in ('top1', 'meank'):
        within_all = np.array([abs(S[v][key] - S[f][key]) for v, f, _ in PAIRS])
        keep = ~np.isnan(within_all)
        within = within_all[keep]
        cl = [c for (_, _, c), k in zip(PAIRS, keep) if k]
        between = np.array([abs(S[PAIRS[i][0]][key] - S[PAIRS[j][0]][key]) for i, j in enumerate(perm)])
        between = between[~np.isnan(between)]
        res[key] = {
            'pairs_used': int(keep.sum()),
            'within_pair_abs_diff_median': round(float(np.median(within)), 5),
            'within_pair_abs_diff_mean': ci_mean(within, cl),
            'within_pair_exactly_equal': round(float(np.mean(within == 0)), 4),
            'between_pair_abs_diff_median': round(float(np.median(between)), 5),
            'between_pair_abs_diff_mean': round(float(between.mean()), 5),
            'within_below_between_median': round(float(np.mean(within < np.median(between))), 4),
        }
    if name == 'B positive-weighted':
        res['lean_score'] = 'not applicable: the positive-only index returns no safe/fixed evidence, so the margin equals the top-1 score'
        OUT['runs'][name] = res
        continue
    lv = np.array([S[v]['lean'] for v, _, _ in PAIRS]); lf = np.array([S[f]['lean'] for _, f, _ in PAIRS])
    ties = lv == lf
    res['lean_score'] = {
        'definition': 'best vulnerable-side minus best safe/fixed-side score' if kind == 'chunk' else 'similarity to historical vulnerable side minus fixed side',
        'median_value': round(float(np.median(np.concatenate([lv, lf]))), 5),
        'pairs_tied': int(ties.sum()),
        'vulnerable_higher_among_untied': round(float(np.mean((lv > lf)[~ties])), 4) if (~ties).any() else None,
        'vulnerable_higher_ci_untied': ci_mean((lv > lf)[~ties].astype(float), [c for (_, _, c), t in zip(PAIRS, ties) if not t]) if (~ties).any() else None,
        'median_abs_diff': round(float(np.median(np.abs(lv - lf))), 5),
    }
    OUT['runs'][name] = res

json.dump(OUT, open('analysis/fse_revision_20260923/rq2_similarity_scores.json', 'w'), indent=1)
print(json.dumps(OUT, indent=1))

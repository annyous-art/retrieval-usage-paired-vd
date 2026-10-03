"""Vul-RAG (GPT-5.5, end to end) against the other GPT-5.5 usage modes on the same pairs.

Pair outcomes on the 431 resolvable pairs; differences in P-C, P-V, P-B and F1 with a
pair-preserving commit-cluster bootstrap (3000 draws, seed 20260920), as elsewhere in
the paper. Also compares P-C within Vul-RAG for pairs whose two members receive the
same vs. different top-3 knowledge, and reports the GLM-5.1 runs that reuse GPT-5.5's knowledge
base and queries (with the pair's retrieved knowledge, and with the knowledge of another pair).

Run from the repository root:
python3 analysis/fse_revision_20260923/vulrag_comparison.py [vulrag_run_dir/]
"""
import collections
import glob
import json
import sys

import numpy as np

sys.path.insert(0, '.')
from evaluate_prompt_outputs import parse_prediction  # noqa: E402

D = sys.argv[1] if len(sys.argv) > 1 else 'outdir/vulrag_gpt-5.5/'
RAW = [json.loads(l) for l in open('data/primevul_test_paired_labeled.jsonl')]
dup = collections.Counter(str(r['idx']) for r in RAW)
PAIRS = []
for i in range(0, len(RAW), 2):
    a, b = RAW[i], RAW[i + 1]
    if dup[str(a['idx'])] > 1 or dup[str(b['idx'])] > 1:
        continue
    v, f = (a, b) if int(a['target']) == 1 else (b, a)
    PAIRS.append((str(v['idx']), str(f['idx']), f"{v['project']}:{v['commit_id']}"))

T = json.load(open(D + 'targets.json'))
row2idx = {str(t['row_id']): str(t['idx']) for t in T}
VR = {}
for p in glob.glob(D + 'prediction_items/*.json'):
    x = json.load(open(p))
    VR[row2idx[str(x['row_id'])]] = int(x['prediction'])
RET = {row2idx[str(x['row_id'])]: x['knowledge_ids'] for x in json.load(open(D + 'retrieved.json'))}


def from_prompts(path):
    m = {}
    for l in open(path):
        r = json.loads(l)
        m.setdefault(str(r.get('query_idx', r.get('idx'))), parse_prediction(r.get('response')))
    return m


KNOW = collections.defaultdict(dict)
for l in open('outdir/knowledge_rag_gpt-5.5/predictions.jsonl'):
    r = json.loads(l)
    p = parse_prediction(r.get('response'))
    if p is not None:
        KNOW[r['arm']][str(r['idx'])] = p

G = 'outdir/prompt_gpt55_representative_ablation_test870/gpt-5.5_std_cls_'
METHODS = {
    'Vul-RAG (E, end to end)': VR,
    'no retrieval (knowledge-exp prompt)': KNOW['no_retrieval'],
    'fixed few-shot (prompt baseline)': from_prompts('data/gpt-5.5_std_cls_logprobsFalse_fewshotegTrue1_baseline_870.jsonl'),
    'chunk RAG positive-only (B)': from_prompts('data/gpt-5.5_std_cls_fewshotegFalse1_RAG_870.jsonl'),
    'grouped no-label RAG (B)': from_prompts(G + 'grouped_no_label_rag_fewshotegFalse_cls0.jsonl'),
    'BC grouped no-label (B)': from_prompts(G + 'grouped_no_label_baseline_compatible_fewshotegTrue_cls0.jsonl'),
    'code pair, fixed candidate (C)': KNOW['code_pair'],
    'extracted knowledge, fixed candidate (E)': KNOW['knowledge'],
}
clusters = sorted({c for *_, c in PAIRS})
cid = {c: i for i, c in enumerate(clusters)}
W = np.random.default_rng(20260920).multinomial(len(clusters), np.ones(len(clusters)) / len(clusters), size=3000)


def counts(m, sel):
    """Per-cluster counts: P-C, P-V, P-B, P-R, TP, FP, n_pairs."""
    arr = np.zeros((len(clusters), 7))
    for (v, f, c), s in zip(PAIRS, sel):
        if not s or m.get(v) is None or m.get(f) is None:
            continue
        a, b = m[v], m[f]
        k = {(1, 0): 0, (1, 1): 1, (0, 0): 2, (0, 1): 3}[(a, b)]
        arr[cid[c], k] += 1; arr[cid[c], 4] += a; arr[cid[c], 5] += b; arr[cid[c], 6] += 1
    return arr


def rates(tot):
    n = tot[..., 6]
    tp, fp = tot[..., 4], tot[..., 5]
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = tp / n
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return {'P-C': tot[..., 0] / n, 'P-V': tot[..., 1] / n, 'P-B': tot[..., 2] / n, 'P-R': tot[..., 3] / n,
            'YES': (tp + fp) / (2 * n), 'F1': f1}


common = [all(m.get(v) is not None and m.get(f) is not None for m in METHODS.values()) for v, f, _ in PAIRS]
OUT = {'common_pairs': int(sum(common)), 'methods': {}, 'vulrag_minus': {}}
C = {k: counts(m, common) for k, m in METHODS.items()}
for k, arr in C.items():
    OUT['methods'][k] = {m: round(float(x), 4) for m, x in rates(arr.sum(0)).items()}
base = C['Vul-RAG (E, end to end)']
for k, arr in C.items():
    if k.startswith('Vul-RAG'):
        continue
    point = {m: rates(base.sum(0))[m] - rates(arr.sum(0))[m] for m in ('P-C', 'P-V', 'P-B', 'F1')}
    bs = {m: rates(W @ base)[m] - rates(W @ arr)[m] for m in ('P-C', 'P-V', 'P-B', 'F1')}
    OUT['vulrag_minus'][k] = {m: [round(100 * float(point[m]), 2), [round(100 * float(q), 2) for q in np.quantile(bs[m], [.025, .975])]] for m in point}

same = [set(RET[v][:3]) == set(RET[f][:3]) for v, f, _ in PAIRS]
a = counts(VR, same); b = counts(VR, [not s for s in same])
d = rates(W @ a)['P-C'] - rates(W @ b)['P-C']
OUT['vulrag_pc_same_vs_different_knowledge'] = {
    'same_pairs': int(a[:, 6].sum()), 'different_pairs': int(b[:, 6].sum()),
    'P-C_same': round(float(rates(a.sum(0))['P-C']), 4), 'P-C_different': round(float(rates(b.sum(0))['P-C']), 4),
    'diff_points': round(100 * float(rates(a.sum(0))['P-C'] - rates(b.sum(0))['P-C']), 2),
    'ci95': [round(100 * float(q), 2) for q in np.quantile(d, [.025, .975])]}
m = json.load(open(D + 'metrics.json'))
GLM = {'retrieved knowledge': 'outdir/vulrag_glm-5.1_gpt55_knowledge/',
       'knowledge of another pair': 'outdir/vulrag_glm-5.1_other_pair_knowledge/'}
ZERO = from_prompts('outdir/baseline_fewshot_sensitivity_glm51_test870/glm-5.1_std_cls_logprobsFalse_fewshotegFalse_none.jsonl')
OUT['glm-5.1'] = {}
for k, d in GLM.items():
    rows = json.load(open(d + 'predictions.json'))
    pred = {str(r['idx']): int(r['prediction']) for r in rows}
    ret = {x['row_id']: set(x['knowledge_ids'][:3]) for x in json.load(open(d + 'retrieved.json'))}
    own = {str(r['idx']): ret[r['row_id']] for r in rows}
    arr = counts(pred, [True] * len(PAIRS))
    OUT['glm-5.1'][k] = {'pairs': int(arr[:, 6].sum()), **{m: round(float(x), 4) for m, x in rates(arr.sum(0)).items()}}
    # P-C and P-R against GLM-5.1 zero-shot on the pairs whose two members receive the same top-3 knowledge,
    # by GPT-5.5's retrieval (the 55 pairs of the RQ2 figure) and by this run's own retrieval
    for sub, sel in (('same_items_gpt55_retrieval', same), ('same_items_this_run', [own[v] == own[f] for v, f, _ in PAIRS])):
        r, z = rates(counts(pred, sel).sum(0)), rates(counts(ZERO, sel).sum(0))
        OUT['glm-5.1'][k][sub] = {'pairs': int(sum(sel)), 'P-C': round(float(r['P-C']), 4), 'P-R': round(float(r['P-R']), 4),
                                  'zero_shot_P-C': round(float(z['P-C']), 4), 'zero_shot_P-R': round(float(z['P-R']), 4)}
OUT['vulrag_exhausted_default_no'] = {'functions': m.get('exhausted_no_match'), 'of': m.get('planned')}
json.dump(OUT, open('analysis/fse_revision_20260923/vulrag_comparison.json', 'w'), indent=1)
print(json.dumps(OUT, indent=1))

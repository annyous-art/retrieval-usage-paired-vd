"""Pair outcomes of the configurations of Tables 3 and 4, and the RQ3 comparisons (offline; reads recorded predictions).

For every model and configuration: P-C, P-V, P-B, P-R, precision, recall, F1 and YES rate on the
431 resolvable pairs. Comparisons are made only within a prompt setting (the two prompts of Section 3.3):
  std  : A (retrieved demonstrations) and B (BC grouped no-label) against the author two-shot
         prompt; RQ3 placement: grouped top-3 against inline chunk-label (same three items).
  FC   : code pair, extracted knowledge (with / without the fix field), and the CWE/CVE arms
         against no retrieval; RQ3: short against full CWE entries.
Differences use the pair-preserving commit-cluster bootstrap (3000 draws, seed 20260920) with
Holm correction within each model's std family and FC family. A method whose file is missing or
incomplete is reported with its coverage and left out of the comparisons.

Run from the repository root: python3 analysis/fse_revision_20260923/rq1_rq3_modes.py
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, '.')
from evaluate_prompt_outputs import parse_prediction  # noqa: E402
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

FC = {'gpt-5.5': 'outdir/knowledge_rag_gpt-5.5/predictions.jsonl',
      'glm-5.1': 'outdir/knowledge_rag_glm-5.1/predictions.jsonl',
      'claude-opus-4-7': 'outdir/knowledge_rag_claude-opus-4-7/predictions_recovered.jsonl',
      'deepseek-v4-pro-0813': 'outdir/knowledge_rag_deepseek-v4-pro-0813/predictions.jsonl'}
ICL = 'outdir/icl_fixed_baseline_compatible_test870/{}_std_cls_fewshotegTrue_top2_baseline_compatible.jsonl'
STD = {
    'gpt-5.5': {'base': 'data/gpt-5.5_std_cls_logprobsFalse_fewshotegTrue1_baseline_870.jsonl',
                'A': 'outdir/replay_a_icl_top2/gpt-5.5_std_cls_fewshotegTrue_ICL_870.jsonl',
                'B_bc': 'outdir/prompt_gpt55_representative_ablation_test870/gpt-5.5_std_cls_grouped_no_label_baseline_compatible_fewshotegTrue_cls0.jsonl',
                'B_inline_chunk': 'outdir/grouped_top3/gpt-5.5_std_cls_inline_chunk_label_fewshotegFalse.jsonl',
                'B_grouped_top3': 'outdir/grouped_top3/gpt-5.5_std_cls_grouped_top3_chunk_label_fewshotegFalse.jsonl'},
    'glm-5.1': {'base': 'outdir/glm-5.1_std_cls_logprobsFalse_fewshotegTrue_baseline_870.jsonl',
                'A': ICL.format('glm-5.1'),
                'B_bc': 'outdir/prompt_baseline_compatible_glm51_test870/glm-5.1_std_cls_grouped_no_label_baseline_compatible_fewshotegTrue_cls0.jsonl',
                'B_inline_chunk': 'outdir/prompt_ablation_glm51_test870/glm-5.1_std_cls_inline_chunk_label_fewshotegFalse_cls0.jsonl',
                'B_grouped_top3': 'outdir/grouped_top3/glm-5.1_std_cls_grouped_top3_chunk_label_fewshotegFalse.jsonl'},
    'claude-opus-4-7': {'base': 'outdir/replay_fixed_fewshot/claude-opus-4-7_std_cls_logprobsFalse_fewshotegTrue_baseline_870.jsonl',
                        'A': ICL.format('claude-opus-4-7'),
                        'B_bc': 'outdir/replay_b_bc_grouped_no_label/claude-opus-4-7_std_cls_grouped_no_label_baseline_compatible_fewshotegTrue_cls0.jsonl'},
    'deepseek-v4-pro-0813': {'base': 'outdir/replay_fixed_fewshot/deepseek-v4-pro-0813_std_cls_logprobsFalse_fewshotegTrue_baseline_870.jsonl',
                             'A': 'outdir/replay_a_icl_top2/deepseek-v4-pro-0813_std_cls_fewshotegTrue_ICL_870.jsonl',
                             'B_bc': 'outdir/replay_b_bc_grouped_no_label/deepseek-v4-pro-0813_std_cls_grouped_no_label_baseline_compatible_fewshotegTrue_cls0.jsonl'},
}
E_ARMS = ['cwe_code', 'cwe_desc', 'cve_desc', 'cwe_desc_short']
STD_CMP = [('A', 'base'), ('B_bc', 'base')]
FC_CMP = [(x, 'no_retrieval') for x in ('code_pair', 'knowledge', 'knowledge_no_fix', 'cwe_code', 'cwe_desc', 'cve_desc')]
CD_CMP = [('knowledge', 'code_pair'), ('knowledge_no_fix', 'code_pair')]  # C vs. D on the same candidate
RQ3_CMP = [('B_grouped_top3', 'B_inline_chunk'), ('cwe_desc_short', 'cwe_desc')]
# Claude Opus 4.7: the first endpoint answered the short-entry arm (29 Sep) with hidden output tokens on 726 of 870
# calls against 29 in the full-entry arm (27 Sep), so both arms were resent unchanged through a second endpoint
# (e_arm_resent_sources.py + replay_saved_prompts.py) and the entry-length comparison uses that pair.
RESENT_E = {'claude-opus-4-7': 'outdir/e_arm_claude-opus-4-7_resent/{}.jsonl'}
# Five resent requests return empty content on every retry; that comparison uses the pairs both arms answered.
PARTIAL_OK = {'cwe_desc_resent', 'cwe_desc_short_resent'}
RQ3_CMP_BY_MODEL = {'claude-opus-4-7': [('B_grouped_top3', 'B_inline_chunk'), ('cwe_desc_short_resent', 'cwe_desc_resent')]}


def std_preds(path):
    if not os.path.exists(path):
        return None
    m = {}
    for l in open(path):
        r = json.loads(l)
        k = str(r.get('query_idx', r.get('idx')))
        p = parse_prediction(r.get('response'))
        if p is not None or k not in m:
            m[k] = p
    return m


def fc_preds(path, arm=None):
    if not os.path.exists(path):
        return None
    m = {}
    for l in open(path):
        r = json.loads(l)
        if arm and r.get('arm') != arm:
            continue
        m[str(r['idx'])] = fc_exp.prediction(r.get('response'))
    return m


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
    n = t[..., 6]; tp, fp = t[..., 4], t[..., 5]
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, n, out=np.zeros_like(tp), where=n > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    safe = np.where(n > 0, n, 1)
    return {'P-C': t[..., 0] / safe, 'P-V': t[..., 1] / safe, 'P-B': t[..., 2] / safe, 'P-R': t[..., 3] / safe,
            'precision': prec, 'recall': rec, 'F1': f1, 'YES': (tp + fp) / (2 * safe)}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i])); adj[i] = run
    return adj


def compare(A, B):
    """A minus B on pairs both methods answered."""
    both = {x for v, f, _ in PAIRS for x in (v, f) if A.get(x) is not None and B.get(x) is not None}
    ca = counts({k: v for k, v in A.items() if k in both}); cb = counts({k: v for k, v in B.items() if k in both})
    out = {'pairs': int(ca[:, 6].sum())}
    for met in ('P-C', 'P-V', 'P-B', 'F1', 'YES'):
        point = rates(ca.sum(0))[met] - rates(cb.sum(0))[met]
        d = rates(W @ ca)[met] - rates(W @ cb)[met]
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        out[met] = {'diff_points': round(100 * float(point), 2), 'ci95': [round(100 * float(q), 2) for q in np.quantile(d, [.025, .975])], 'p': round(float(p), 4)}
    return out


OUT = {'pairs': len(PAIRS), 'models': {}}
for model in STD:
    M = {}
    for name, path in STD[model].items():
        M[name] = std_preds(path)
    for arm in ('no_retrieval', 'code_pair', 'knowledge', 'knowledge_no_fix'):
        M[arm] = fc_preds(FC[model], arm)
    for arm in E_ARMS:
        M[arm] = fc_preds(f'outdir/e_arm_{model}_{arm}/predictions.jsonl')
    if model in RESENT_E:
        for arm in ('cwe_desc', 'cwe_desc_short'):
            M[arm + '_resent'] = fc_preds(RESENT_E[model].format(arm))
    # Mode D with another model's knowledge (shared_knowledge_arm.py); reported, not tested here
    for tag, d in (('gpt55', f'outdir/shared_knowledge_{model}'), ('claude47', f'outdir/shared_knowledge_claude47_{model}')):
        if os.path.exists(d + '/predictions.jsonl'):
            M[f'knowledge_{tag}'] = fc_preds(d + '/predictions.jsonl', f'knowledge_{tag}')
    res = {'methods': {}, 'std_vs_base': {}, 'fc_vs_no_retrieval': {}, 'c_vs_d': {}, 'rq3': {}}
    complete = {}
    for name, m in M.items():
        if m is None:
            continue
        answered = sum(m.get(x) is not None for v, f, _ in PAIRS for x in (v, f))
        complete[name] = answered == 2 * len(PAIRS)
        t = counts(m).sum(0)
        res['methods'][name] = {'answered_functions': answered, 'complete': complete[name],
                                **({k: round(float(v), 6) for k, v in rates(t).items()} if t[6] else {})}
    for fam, cmps in (('std_vs_base', STD_CMP), ('fc_vs_no_retrieval', FC_CMP), ('c_vs_d', CD_CMP), ('rq3', RQ3_CMP_BY_MODEL.get(model, RQ3_CMP))):
        done = [(a, b) for a, b in cmps if all(complete.get(x) or (x in PARTIAL_OK and x in complete) for x in (a, b))]
        stats = {f'{a} - {b}': compare(M[a], M[b]) for a, b in done}
        for met in ('P-C', 'F1'):
            keys = list(stats)
            for k, adj in zip(keys, holm([stats[k][met]['p'] for k in keys])):
                stats[k][met]['p_holm'] = round(adj, 4)
        res[fam] = stats
        res[fam + '_pending'] = [f'{a} - {b}' for a, b in cmps if (a, b) not in done]
    OUT['models'][model] = res

json.dump(OUT, open('analysis/fse_revision_20260923/rq1_rq3_modes.json', 'w'), indent=1)
for model, res in OUT['models'].items():
    print(f'== {model}')
    for name, r in res['methods'].items():
        if 'P-C' in r:
            print(f"  {name:18s} {'' if r['complete'] else '(partial ' + str(r['answered_functions']) + ')':14s} P-C {r['P-C']:.3f} P-V {r['P-V']:.3f} P-B {r['P-B']:.3f} F1 {r['F1']:.3f} YES {r['YES']:.3f}")
    for fam in ('std_vs_base', 'fc_vs_no_retrieval', 'c_vs_d', 'rq3'):
        for k, s in res[fam].items():
            print(f"  {fam}: {k:34s} dP-C {s['P-C']['diff_points']:+.1f} {s['P-C']['ci95']} p_holm {s['P-C']['p_holm']}  dP-B {s['P-B']['diff_points']:+.1f}")

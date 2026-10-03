"""Figure 3a: every retrieval configuration against zero-shot, within the subsets whose two members
receive identical or different retrieved items (offline; no API calls).

For each configuration and model, the change in P-C and in net separation (P-C minus P-R) relative to
the zero-shot prompt (no demonstrations, no retrieval) on the same pairs, overall and within the
identical/different subsets. Intervals: the pair-preserving commit-cluster bootstrap of
subset_net_separation.py (3000 draws, seed 20260920), not corrected for multiplicity.

Subsets (as in Table 2): mode A, the set of demonstrations; mode B, the inserted
evidence block; C and D, the retrieved candidate; E, the set of the top-3 entries; Vul-RAG, the set
of the three knowledge items. Vul-RAG with GLM-5.1 uses GPT-5.5's knowledge base and queries.

Run from the repository root: python3 analysis/fse_revision_20260923/rq2_zero_shot_subsets.py
Writes rq2_zero_shot_subsets.json next to this script and the LaTeX rows to stdout.
"""
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'experiments/knowledge_rag')
from evaluate_prompt_outputs import parse_prediction  # noqa: E402
import experiment as exp  # noqa: E402

RAW = [json.loads(l) for l in open('data/primevul_test_paired_labeled.jsonl')]
DUP = collections.Counter(str(r['idx']) for r in RAW)
PAIRS = []
for i in range(0, len(RAW), 2):
    a, b = RAW[i], RAW[i + 1]
    if DUP[str(a['idx'])] > 1 or DUP[str(b['idx'])] > 1:
        continue
    v, f = (a, b) if int(a['target']) == 1 else (b, a)
    PAIRS.append((str(v['idx']), str(f['idx']), f"{v['project']}:{v['commit_id']}"))
clusters = sorted({c for *_, c in PAIRS})
cid = {c: i for i, c in enumerate(clusters)}
W = np.random.default_rng(20260920).multinomial(len(clusters), np.ones(len(clusters)) / len(clusters), size=3000)


def jsonl(path):
    return [json.loads(l) for l in open(path)]


# ---- loaders: each returns ({idx: prediction}, {idx: retrieved-item key}) ----

def std_cls(path, key_path=None, kind='demo'):
    recs = {str(r.get('query_idx', r.get('idx'))): r for r in jsonl(path)}
    src = {str(r.get('query_idx', r.get('idx'))): r for r in jsonl(key_path)} if key_path else recs
    pred = {k: parse_prediction(r.get('response')) for k, r in recs.items()}
    if kind == 'demo':
        key = {k: tuple(sorted(str(s.get('idx')) for s in (r.get('similar') or [])[:r.get('num_examples_used', 2)]
                               if isinstance(s, dict))) for k, r in src.items()}
    elif kind == 'chunk':
        key = {k: evidence_key(r) for k, r in src.items()}
    else:
        key = None
    return pred, key


def evidence(msg):  # as in subset_net_separation.py
    f = msg.find('Focus Regions:')
    if f == -1:
        return None
    m = re.search(r'\n\nRetrieved ', msg[f:])
    return msg[f + m.start():] if m else msg[f:]


def evidence_key(r):
    """SHA-256 of the evidence block; read from `evidence_sha256` when the prompt is not stored."""
    if r.get('messages'):
        ev = evidence(r['messages'][-1]['content'])
        return None if ev is None else hashlib.sha256(ev.encode()).hexdigest()
    return r.get('evidence_sha256')


def slot(run, arm):  # knowledge_rag v7 runs: C = code_pair, D = knowledge
    rows = [r for r in jsonl(f'outdir/{run}/predictions.jsonl') if r['arm'] == arm]
    return ({str(r['idx']): exp.prediction(r.get('response')) for r in rows},
            {str(r['idx']): tuple(sorted(r['candidate_ids'])) for r in rows})


def e_arm(run):
    jobs = {j['job_id']: j for j in jsonl(f'outdir/{run}/jobs.jsonl')}
    rows = jsonl(f'outdir/{run}/predictions.jsonl')
    return ({str(jobs[r['job_id']]['idx']): exp.prediction(r.get('response')) for r in rows},
            {str(j['idx']): tuple(sorted(j['candidate_ids'])) for j in jobs.values()})


def vulrag(run):
    ret = {r['row_id']: tuple(sorted(r['knowledge_ids'])) for r in json.load(open('outdir/vulrag_gpt-5.5/retrieved.json'))}
    rows = json.load(open(f'outdir/{run}/predictions.json'))
    return {str(r['idx']): r['prediction'] for r in rows}, {str(r['idx']): ret[r['row_id']] for r in rows}


# ---- statistics ----

def counts(pred, mask):
    arr = np.zeros((len(clusters), 3))  # P-C, P-R, pairs
    for (v, f, c), keep in zip(PAIRS, mask):
        if keep:
            a, b = pred.get(v), pred.get(f)
            arr[cid[c], 0] += (a, b) == (1, 0)
            arr[cid[c], 1] += (a, b) == (0, 1)
            arr[cid[c], 2] += 1
    return arr


def change(ret, ctl, mask):
    mask = [m and all(p.get(x) is not None for p in (ret, ctl) for x in (v, f)) for m, (v, f, _) in zip(mask, PAIRS)]
    a, b = counts(ret, mask), counts(ctl, mask)
    ra, rb = a.sum(0), b.sum(0)
    if ra[2] == 0:
        return None
    dpc = (W @ a[:, 0]) / (W @ a[:, 2]) - (W @ b[:, 0]) / (W @ b[:, 2])
    net = (W @ a[:, 0] - W @ a[:, 1]) / (W @ a[:, 2]) - (W @ b[:, 0] - W @ b[:, 1]) / (W @ b[:, 2])
    pt = lambda i: 100 * (ra[i] / ra[2] - rb[i] / rb[2])
    ci = lambda x: [round(100 * float(q), 1) for q in np.quantile(x, [.025, .975])]
    return {'pairs': int(ra[2]), 'P-C': [round(100 * rb[0] / rb[2], 1), round(100 * ra[0] / ra[2], 1)],
            'P-R': [round(100 * rb[1] / rb[2], 1), round(100 * ra[1] / ra[2], 1)],
            'dP-C': [round(pt(0), 1), ci(dpc)], 'dNet': [round(pt(0) - pt(1), 1), ci(net)]}


ZERO = {'GPT-5.5': std_cls('outdir/replay_prompt_baselines_gpt55/gpt-5.5_std_cls_logprobsFalse_fewshotegFalse_none.jsonl', kind=None)[0],
        'GLM-5.1': std_cls('outdir/baseline_fewshot_sensitivity_glm51_test870/glm-5.1_std_cls_logprobsFalse_fewshotegFalse_none.jsonl', kind=None)[0]}
A2_GLM = 'outdir/icl_fixed_baseline_compatible_test870/glm-5.1_std_cls_fewshotegTrue_top2_baseline_compatible.jsonl'
GP = 'outdir/prompt_gpt55_representative_ablation_test870/gpt-5.5_std_cls_'
GL = 'outdir/prompt_ablation_glm51_test870/glm-5.1_std_cls_'
INSTANCES = {  # (mode, instance) -> {model: loader}
    ('A', 'Two retrieved demonstrations'): {
        'GPT-5.5': lambda: std_cls('outdir/replay_a_icl_top2/gpt-5.5_std_cls_fewshotegTrue_ICL_870.jsonl', A2_GLM),
        'GLM-5.1': lambda: std_cls(A2_GLM)},
    ('A', 'One retrieved demonstration'): {
        'GPT-5.5': lambda: std_cls('outdir/a_k_sensitivity/gpt-5.5_std_cls_fewshotegTrue_top1_baseline_compatible.jsonl', 'outdir/a_k_sensitivity/prompts_top1.jsonl'),
        'GLM-5.1': lambda: std_cls('outdir/a_k_sensitivity/glm-5.1_std_cls_fewshotegTrue_top1_baseline_compatible.jsonl', 'outdir/a_k_sensitivity/prompts_top1.jsonl')},
    ('A', 'Four retrieved demonstrations'): {
        'GPT-5.5': lambda: std_cls('outdir/a_k_sensitivity/gpt-5.5_std_cls_fewshotegTrue_top4_baseline_compatible.jsonl', 'outdir/a_k_sensitivity/prompts_top4.jsonl'),
        'GLM-5.1': lambda: std_cls('outdir/a_k_sensitivity/glm-5.1_std_cls_fewshotegTrue_top4_baseline_compatible.jsonl', 'outdir/a_k_sensitivity/prompts_top4.jsonl')},
    ('B', 'Two-shot with retrieved chunks'): {
        'GPT-5.5': lambda: std_cls(GP + 'grouped_no_label_baseline_compatible_fewshotegTrue_cls0.jsonl', kind='chunk'),
        'GLM-5.1': lambda: std_cls('outdir/prompt_baseline_compatible_glm51_test870/glm-5.1_std_cls_grouped_no_label_baseline_compatible_fewshotegTrue_cls0.jsonl', kind='chunk')},
    ('B', 'Retrieved chunks only'): {
        'GPT-5.5': lambda: std_cls(GP + 'grouped_no_label_rag_fewshotegFalse_cls0.jsonl', kind='chunk'),
        'GLM-5.1': lambda: std_cls(GL + 'grouped_no_label_fewshotegFalse_cls0.jsonl', kind='chunk')},
    ('B', 'Changed chunks'): {
        'GPT-5.5': lambda: std_cls('data/gpt-5.5_std_cls_fewshotegFalse1_RAG_870.jsonl', kind='chunk'),
        'GLM-5.1': lambda: std_cls('outdir/replay_rag_positive_only/glm-5.1_std_cls_fewshotegFalse1_RAG_870.jsonl', kind='chunk')},
    ('C', 'Complete code pair'): {
        'GPT-5.5': lambda: slot('knowledge_rag_gpt-5.5', 'code_pair'),
        'GLM-5.1': lambda: slot('knowledge_rag_glm-5.1', 'code_pair')},
    ('D', 'Extracted knowledge'): {
        'GPT-5.5': lambda: slot('knowledge_rag_gpt-5.5', 'knowledge'),
        'GLM-5.1': lambda: slot('knowledge_rag_glm-5.1', 'knowledge')},
    ('D', 'Vul-RAG'): {
        'GPT-5.5': lambda: vulrag('vulrag_gpt-5.5'),
        'GLM-5.1': lambda: vulrag('vulrag_glm-5.1_gpt55_knowledge')},
    ('E', 'CWE entries, code query'): {
        'GPT-5.5': lambda: e_arm('e_arm_gpt-5.5_cwe_code'), 'GLM-5.1': lambda: e_arm('e_arm_glm-5.1_cwe_code')},
    ('E', 'CWE entries, description query'): {
        'GPT-5.5': lambda: e_arm('e_arm_gpt-5.5_cwe_desc'), 'GLM-5.1': lambda: e_arm('e_arm_glm-5.1_cwe_desc')},
    ('E', 'CVE descriptions'): {
        'GPT-5.5': lambda: e_arm('e_arm_gpt-5.5_cve_desc'), 'GLM-5.1': lambda: e_arm('e_arm_glm-5.1_cve_desc')},
}

def main():
    OUT = {}
    for (mode, inst), runs in INSTANCES.items():
        for model, load in runs.items():
            pred, key = load()
            same = [key.get(v) is not None and key.get(v) == key.get(f) for v, f, _ in PAIRS]
            res = {'all': change(pred, ZERO[model], [True] * len(PAIRS)),
                   'identical': change(pred, ZERO[model], same),
                   'different': change(pred, ZERO[model], [not s for s in same])}
            OUT[f'{model} | {mode} | {inst}'] = res
            print(f'{model:8s} {mode} {inst:32s}', *(f"{k}: n={r['pairs']} dPC={r['dP-C'][0]:+.1f}{r['dP-C'][1]} dNet={r['dNet'][0]:+.1f}{r['dNet'][1]}"
                                                    for k, r in res.items() if r), sep='\n   ')
    Path(__file__).with_suffix('.json').write_text(json.dumps(OUT, indent=1))


if __name__ == '__main__':
    main()

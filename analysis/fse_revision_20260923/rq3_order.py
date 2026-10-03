"""RQ3: the same retrieved items in reversed order, GLM-5.1 and GPT-5.5 (offline analysis).

Each comparison: reordered arm minus original order on the same pairs, for P-C, P-R, P-V, P-B and
net P-C (P-C minus P-R), with the commit-cluster bootstrap of rq2_zero_shot_subsets.py.
A is compared with the original A prompts replayed through the same gateway, C, D and E with the
source runs, which used the same gateway as the reordered arms.

Run from the repository root: python3 analysis/fse_revision_20260923/rq3_order.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'analysis/fse_revision_20260923')
import rq2_zero_shot_subsets as z  # noqa: E402

GLM, GPT = 'outdir/order_glm-5.1/', 'outdir/order_gpt-5.5/'


def order_arm(arm, O=GLM):
    jobs = {j['job_id']: j for j in z.jsonl(O + arm + '/jobs.jsonl')}
    return {str(jobs[r['job_id']]['idx']): z.exp.prediction(r.get('response')) for r in z.jsonl(O + arm + '/predictions.jsonl')}


COMPARISONS = [  # (mode, change, model, reordered loader, original loader); A's original is replayed on the same gateway
    ('A', 'Demonstrations swapped', 'GLM-5.1',
     lambda: z.std_cls(GLM + 'a_demos_swapped/glm-5.1_std_cls_fewshotegTrue_top2_demos_swapped.jsonl', kind=None)[0],
     lambda: z.std_cls(GLM + 'a_original/glm-5.1_std_cls_fewshotegTrue_top2_original.jsonl', kind=None)[0]),
    ('A', 'Demonstrations swapped', 'GPT-5.5',
     lambda: z.std_cls(GPT + 'a_demos_swapped/gpt-5.5_std_cls_fewshotegTrue_top2_demos_swapped.jsonl', kind=None)[0],
     lambda: z.std_cls(GPT + 'a_original/gpt-5.5_std_cls_fewshotegTrue_top2_original.jsonl', kind=None)[0]),
    ('C', 'Repaired code before vulnerable code', 'GLM-5.1', lambda: order_arm('c_fixed_first'),
     lambda: z.slot('knowledge_rag_glm-5.1', 'code_pair')[0]),
    ('C', 'Repaired code before vulnerable code', 'GPT-5.5', lambda: order_arm('c_fixed_first', GPT),
     lambda: z.slot('knowledge_rag_gpt-5.5', 'code_pair')[0]),
    ('D', 'Knowledge fields reversed', 'GLM-5.1', lambda: order_arm('d_fields_rev'),
     lambda: z.slot('knowledge_rag_glm-5.1', 'knowledge')[0]),
    ('D', 'Knowledge fields reversed', 'GPT-5.5', lambda: order_arm('d_fields_rev', GPT),
     lambda: z.slot('knowledge_rag_gpt-5.5', 'knowledge')[0]),
    ('E', 'CWE entries reversed (description query)', 'GLM-5.1', lambda: order_arm('e_cwe_desc_rev'),
     lambda: z.e_arm('e_arm_glm-5.1_cwe_desc')[0]),
    ('E', 'CWE entries reversed (description query)', 'GPT-5.5', lambda: order_arm('e_cwe_desc_rev', GPT),
     lambda: z.e_arm('e_arm_gpt-5.5_cwe_desc')[0]),
]
OUTCOMES = {'P-C': (1, 0), 'P-R': (0, 1), 'P-V': (1, 1), 'P-B': (0, 0)}


def compare(new, old):
    keep = [all(p.get(x) is not None for p in (new, old) for x in (v, f)) for v, f, _ in z.PAIRS]
    arr = {}
    for name, pred in (('new', new), ('old', old)):
        a = np.zeros((len(z.clusters), 5))
        for (v, f, c), k in zip(z.PAIRS, keep):
            if k:
                o = (pred[v], pred[f])
                for i, t in enumerate(OUTCOMES.values()):
                    a[z.cid[c], i] += o == t
                a[z.cid[c], 4] += 1
        arr[name] = a
    res = {'pairs': int(arr['new'][:, 4].sum())}
    rate = lambda a, i: (z.W @ a[:, i]) / (z.W @ a[:, 4])
    for i, name in enumerate(OUTCOMES):
        n, o = arr['new'].sum(0), arr['old'].sum(0)
        d = rate(arr['new'], i) - rate(arr['old'], i)
        res[name] = [round(100 * o[i] / o[4], 1), round(100 * n[i] / n[4], 1),
                     [round(100 * float(q), 1) for q in np.quantile(d, [.025, .975])]]
    net = (rate(arr['new'], 0) - rate(arr['new'], 1)) - (rate(arr['old'], 0) - rate(arr['old'], 1))
    pt = (res['P-C'][1] - res['P-R'][1]) - (res['P-C'][0] - res['P-R'][0])
    res['net'] = [round(pt, 1), [round(100 * float(q), 1) for q in np.quantile(net, [.025, .975])]]
    return res


def main():
    out = {}
    for mode, change, model, new, old in COMPARISONS:
        try:
            r = compare(new(), old())
        except FileNotFoundError as e:
            print(mode, 'missing:', e.filename)
            continue
        out[f'{mode} | {change} | {model}'] = r
        print(f"{mode} {change} {model} (n={r['pairs']})")
        for k in OUTCOMES:
            o, n, ci = r[k]
            print(f"   {k}: {o:5.1f} -> {n:5.1f}  d={n - o:+.1f} {ci}")
        print(f"   net: d={r['net'][0]:+.1f} {r['net'][1]}")
    Path(__file__).with_suffix('.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()

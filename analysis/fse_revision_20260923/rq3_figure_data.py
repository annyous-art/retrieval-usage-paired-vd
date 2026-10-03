"""Data for the RQ3 figure (offline): for every presentation change and model, the change in the four
pair outcomes, net P-C (P-C minus P-R), and the YES rate against the original presentation, on the pairs both prompts
answered, with the pair-preserving commit-cluster bootstrap (3000 draws, seed 20260920).
Order reversals as in rq3_order.py; placement and entry length as in rq1_rq3_modes.py (Table tab:rq3-new).

Run from the repository root: python3 analysis/fse_revision_20260923/rq3_figure_data.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'analysis/fse_revision_20260923')
import rq3_order as r  # noqa: E402
import rq2_zero_shot_subsets as z  # noqa: E402

G = 'outdir/grouped_top3/{}_std_cls_{}_chunk_label_fewshotegFalse.jsonl'
std = lambda p: z.std_cls(p, kind=None)[0]
e = lambda run: z.e_arm(run)[0]


def uni(arm):  # Claude's entry-length pair, resent through a second endpoint (rq1_rq3_modes.py)
    return {str(x['idx']): z.exp.prediction(x.get('response')) for x in z.jsonl(f'outdir/e_arm_claude-opus-4-7_resent/{arm}.jsonl')}


NEW = [  # (mode, change, model, new loader, original loader)
    ('B', 'Chunks regrouped', 'GPT-5.5', lambda: std(G.format('gpt-5.5', 'grouped_top3')), lambda: std(G.format('gpt-5.5', 'inline'))),
    ('B', 'Chunks regrouped', 'GLM-5.1', lambda: std(G.format('glm-5.1', 'grouped_top3')),
     lambda: std('outdir/prompt_ablation_glm51_test870/glm-5.1_std_cls_inline_chunk_label_fewshotegFalse_cls0.jsonl')),
    ('B', 'Chunks regrouped', 'Claude Opus 4.7', lambda: std(G.format('claude-opus-4-7', 'grouped_top3')), lambda: std(G.format('claude-opus-4-7', 'inline'))),
    ('B', 'Chunks regrouped', 'DeepSeek-V4-Pro', lambda: std(G.format('deepseek-v4-pro-0813', 'grouped_top3')), lambda: std(G.format('deepseek-v4-pro-0813', 'inline'))),
    ('E', 'Entries shortened', 'GPT-5.5', lambda: e('e_arm_gpt-5.5_cwe_desc_short'), lambda: e('e_arm_gpt-5.5_cwe_desc')),
    ('E', 'Entries shortened', 'GLM-5.1', lambda: e('e_arm_glm-5.1_cwe_desc_short'), lambda: e('e_arm_glm-5.1_cwe_desc')),
    ('E', 'Entries shortened', 'Claude Opus 4.7', lambda: uni('cwe_desc_short'), lambda: uni('cwe_desc')),
    ('E', 'Entries shortened', 'DeepSeek-V4-Pro', lambda: e('e_arm_deepseek-v4-pro-0813_cwe_desc_short'), lambda: e('e_arm_deepseek-v4-pro-0813_cwe_desc')),
]
OUTC = {(1, 0): 0, (0, 1): 1, (1, 1): 2, (0, 0): 3}  # P-C, P-R, P-V, P-B


def counts(pred, keep):
    a = np.zeros((len(z.clusters), 5))
    for (v, f, c), k in zip(z.PAIRS, keep):
        if k:
            a[z.cid[c], OUTC[(pred[v], pred[f])]] += 1
            a[z.cid[c], 4] += 1
    return a


def change(new, old):
    keep = [all(p.get(x) is not None for p in (new, old) for x in (v, f)) for v, f, _ in z.PAIRS]
    a, b = counts(new, keep), counts(old, keep)
    rate = lambda t, i: (z.W @ t[:, i]) / (z.W @ t[:, 4])
    pt = lambda t, i: t[:, i].sum() / t[:, 4].sum()
    ci = lambda d: [round(100 * float(q), 1) for q in np.quantile(d, [.025, .975])]
    out = {'pairs': int(a[:, 4].sum())}
    for name, i in (('P-C', 0), ('P-R', 1), ('P-V', 2), ('P-B', 3)):
        out[name] = [round(100 * pt(b, i), 1), round(100 * pt(a, i), 1), ci(rate(a, i) - rate(b, i)), round(100 * (pt(a, i) - pt(b, i)), 1)]
    net_b, net_a = pt(b, 0) - pt(b, 1), pt(a, 0) - pt(a, 1)
    out['net'] = [round(100 * net_b, 1), round(100 * net_a, 1),
                  ci((rate(a, 0) - rate(a, 1)) - (rate(b, 0) - rate(b, 1))), round(100 * (net_a - net_b), 1)]
    # a pair in P-V has two YES answers and a pair in P-C or P-R one, so the YES rate is P-V + (P-C + P-R) / 2
    yes = lambda f, t: f(t, 2) + (f(t, 0) + f(t, 1)) / 2  # noqa: E731
    out['YES'] = [round(100 * yes(pt, b), 1), round(100 * yes(pt, a), 1), ci(yes(rate, a) - yes(rate, b)),
                  round(100 * (yes(pt, a) - yes(pt, b)), 1)]
    return out


def main():
    rows = []
    for mode, ch, model, new, old in [(m, c, mo, n, o) for m, c, mo, n, o in r.COMPARISONS] + NEW:
        res = change(new(), old())
        rows.append({'mode': mode, 'change': ch, 'model': model, **res})
        print(f"{mode} {ch:42s} {model:16s} n={res['pairs']} dP-B {res['P-B'][1]-res['P-B'][0]:+5.1f}{res['P-B'][2]} "
              f"dP-C {res['P-C'][1]-res['P-C'][0]:+5.1f}{res['P-C'][2]} dNet {res['net'][1]-res['net'][0]:+5.1f}{res['net'][2]}")
    Path(__file__).with_suffix('.json').write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()

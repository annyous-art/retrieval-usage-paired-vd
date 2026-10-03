"""Figure 3 (RQ2): (a) change in net P-C relative to zero-shot on the pairs whose two members receive
the same / different retrieved items; (b) P-C with different content in the evidence slot.
Data: rq2_zero_shot_subsets.json, oracle_arms.json, gpt_std_prompts.json, glm_std_prompts.json.
Writes figures/rq2_with_diff.{pdf,png} and the plotted values to figures/rq2_with_diff_values.json.

Run from the repository root: python3 figures/plot_rq2.py
"""
import json
import sys

sys.path.insert(0, 'figures')
from style import (A, INK, MODELS, MUTED, WIDTH, bands, estimate, frame, left, legend, plt,  # noqa: E402
                   row_labels, spanner)

PAIR = ('GPT-5.5', 'GLM-5.1')
Z = json.load(open(A + 'rq2_zero_shot_subsets.json'))
# mode, label, key in rq2_zero_shot_subsets.json
ROWS = [('A', 'One demonstration', 'A | One retrieved demonstration'),
        ('A', 'Two demonstrations', 'A | Two retrieved demonstrations'),
        ('A', 'Four demonstrations', 'A | Four retrieved demonstrations'),
        ('B', 'Code chunks', 'B | Two-shot with retrieved chunks'),
        ('C', 'Code pair', 'C | Complete code pair'),
        ('D', 'Knowledge', 'D | Extracted knowledge'),
        ('D', 'Vul-RAG', 'D | Vul-RAG'),
        ('E', 'CWE entries, code query', 'E | CWE entries, code query'),
        ('E', 'CWE entries, description query', 'E | CWE entries, description query'),
        ('E', 'CVE descriptions, description query', 'E | CVE descriptions')]


def change(model, inst, subset):
    s = Z[f'{model} | {inst}'][subset]
    d, (lo, hi) = s['dNet']
    n = s['pairs']  # net P-C from the pair counts, so that rounding the two rates does not add up
    before, after = (round(100 * (round(pc * n / 100) - round(pr * n / 100)) / n, 1) for pc, pr in zip(s['P-C'], s['P-R']))
    return d, lo, hi, before, after


# mode, label, same: (GPT, GLM), different: (GPT, GLM); each (dnet, lo, hi, net before, net after)
rows = [(mode, lab, tuple(change(m, inst, 'identical') for m in PAIR),
         tuple(change(m, inst, 'different') for m in PAIR)) for mode, lab, inst in ROWS]
O = json.load(open(A + 'oracle_arms.json'))['models']
zero = {m: 100 * json.load(open(A + f))['rows']['Zero-shot']['P-C']
        for m, f in (('gpt-5.5', 'gpt_std_prompts.json'), ('glm-5.1', 'glm_std_prompts.json'))}
pc = lambda arm: tuple(round(O[m]['methods'][arm]['P-C'], 1) for m in ('gpt-5.5', 'glm-5.1'))  # noqa: E731
# mode of the retrieved content, label, P-C of GPT-5.5, P-C of GLM-5.1
slot = [('', 'Zero-shot, no evidence', round(zero['gpt-5.5'], 1), round(zero['glm-5.1'], 1)),
        ('C', 'Retrieved pair, code', *pc('code_pair')),
        ('D', 'Retrieved pair, knowledge', *pc('knowledge')),
        ('', 'Unrelated pair, code', *pc('wrong_code_pair')),
        ('', 'Own pair, code', *pc('oracle_code_pair')),
        ('', 'Own pair, knowledge', *pc('oracle_knowledge'))]
json.dump({'rows': rows, 'slot': slot}, open('figures/rq2_with_diff_values.json', 'w'), indent=1)

LABEL_W = 146
ys, modes = list(range(len(rows))), [r[0] for r in rows]
fig = plt.figure(figsize=(WIDTH, 4.15))
gs = fig.add_gridspec(2, 4, height_ratios=[10.4, 6.6], hspace=0.4, wspace=0.1,
                      left=left(LABEL_W), right=0.995, top=0.84, bottom=0.12)
axs = [fig.add_subplot(gs[0, i]) for i in range(4)]
for ax, (k, j) in zip(axs, [(k, j) for k in (2, 3) for j in (0, 1)]):
    bands(ax, ys, modes)
    for y, r in zip(ys, rows):
        estimate(ax, y, *r[k][j][:3], PAIR[j])
    ax.axvline(0, color=INK, lw=0.8, zorder=2)
    ax.set_xlim(-33, 33)
    ax.set_xticks([-20, 0, 20])
    ax.set_ylim(ys[-1] + 0.5, ys[0] - 0.5)
    ax.set_title(PAIR[j], pad=3, color=MODELS[PAIR[j]], fontweight='bold', fontsize=8.5)
    frame(ax)
row_labels(axs[0], ys, modes, [r[1] for r in rows], LABEL_W)
spanner(fig, axs[0], axs[1], 'Pairs with the same items', 0.9)
spanner(fig, axs[2], axs[3], 'Pairs with different items', 0.9)
x0 = left(LABEL_W) - LABEL_W / 72 / WIDTH
fig.text(x0, 0.965, '(a) Change in net P-C relative to zero-shot, in percentage points', fontsize=9,
         fontweight='bold', color=INK)

bx = fig.add_subplot(gs[1, :])
H = 0.4
for y, (_, _, g, z) in enumerate(slot):
    for v, m, dy in ((g, PAIR[0], -H / 2), (z, PAIR[1], H / 2)):
        bx.barh(y + dy, v, height=H, color=MODELS[m], lw=0, zorder=2)
        inside = v > 50
        bx.text(v - 1.2 if inside else v + 1.2, y + dy, f'{v:.0f}', fontsize=6.8, va='center_baseline',
                ha='right' if inside else 'left', color='white' if inside else INK,
                fontweight='bold' if inside else 'normal', zorder=5)
bx.axhline(3.5, color=MUTED, lw=0.6, ls=(0, (3, 2)))
bx.set_xlim(0, 104)
bx.set_xticks(range(0, 101, 25))
bx.set_ylim(len(slot) - 0.4, -0.6)
frame(bx)
row_labels(bx, list(range(len(slot))), [s[0] for s in slot], [s[1] for s in slot], LABEL_W)
fig.text(x0, bx.get_position().y1 + 0.012, '(b) P-C when both member functions receive the same content, in percent of pairs',
         fontsize=9, fontweight='bold', color=INK)
legend(fig, PAIR, per_model=True, prefix='(a) ')
for ext in ('pdf', 'png'):
    fig.savefig(f'figures/rq2_with_diff.{ext}', dpi=300, bbox_inches='tight')
print('wrote figures/rq2_with_diff.pdf')

"""Figure 5 (RQ3): change in the YES rate and in net P-C when the same retrieved items are presented
differently. Data: analysis/fse_revision_20260923/rq3_figure_data.json (commit-cluster bootstrap).
Run from the repository root."""
import json
import sys

sys.path.insert(0, 'figures')
from style import (A, INK, MODELS, WIDTH, bands, estimate, frame, header, left, legend, plt,  # noqa: E402
                   row_labels, spanner)

SHORT = {'GPT-5.5': 'GPT-5.5', 'GLM-5.1': 'GLM-5.1', 'Claude Opus 4.7': 'Claude', 'DeepSeek-V4-Pro': 'DeepSeek'}
# heading of a group of rows, then (mode, change in rq3_figure_data.json, label)
GROUPS = [('Order reversed',
           [('A', 'Demonstrations swapped', 'Demonstrations swapped'),
            ('C', 'Repaired code before vulnerable code', 'Fixed code first'),
            ('D', 'Knowledge fields reversed', 'Knowledge fields reversed'),
            ('E', 'CWE entries reversed (description query)', 'CWE entries reversed')]),
          ('Placement or length changed',
           [('B', 'Chunks regrouped', 'Chunks regrouped'),
            ('E', 'Entries shortened', 'CWE entries shortened')])]
data = json.load(open(A + 'rq3_figure_data.json'))
get = {(d['mode'], d['change'], d['model']): d for d in data}

heads, rows, ys, y = [], [], [], 0.0
for title, members in GROUPS:
    heads.append((y, title))
    y += 1.0
    for r in members:
        rows.append(r)
        ys.append(y)
        y += 1.0
    y += 0.45

LABEL_W = 110
XLIM, TICKS = (-20, 14), [-10, 0, 10]  # the same axis in all panels
fig = plt.figure(figsize=(WIDTH, 2.7))
gs = fig.add_gridspec(1, 8, wspace=0.14, left=left(LABEL_W), right=0.995, top=0.79, bottom=0.25)
axs = [fig.add_subplot(gs[0, i]) for i in range(8)]
for ax, (key, m) in zip(axs, [(k, m) for k in ('YES', 'net') for m in MODELS]):
    bands(ax, ys, [r[0] for r in rows])
    for yy, (mode, change, _) in zip(ys, rows):
        if (mode, change, m) in get:  # the order variants were run with the two main models only
            before, after, (lo, hi), diff = get[(mode, change, m)][key]
            estimate(ax, yy, diff, lo, hi, m)
    ax.axvline(0, color=INK, lw=0.8, zorder=2)
    ax.set_xlim(*XLIM)
    ax.set_xticks(TICKS)
    ax.tick_params(axis='x', labelsize=7.2)
    ax.set_ylim(ys[-1] + 0.6, heads[0][0] - 0.5)
    ax.set_title(SHORT[m], pad=3, color=MODELS[m], fontweight='bold', fontsize=7.4)
    frame(ax)
row_labels(axs[0], ys, [r[0] for r in rows], [r[2] for r in rows], LABEL_W)
for yy, title in heads:
    header(axs[0], yy, title, LABEL_W)
spanner(fig, axs[0], axs[3], '(a) Change in YES rate', 0.87)
spanner(fig, axs[4], axs[7], '(b) Change in net P-C', 0.87)
fig.text((axs[0].get_position().x0 + axs[7].get_position().x1) / 2, 0.112, 'percentage points', ha='center', color=INK)
legend(fig, MODELS)
for ext in ('pdf', 'png'):
    fig.savefig(f'figures/rq3.{ext}', dpi=300, bbox_inches='tight')
print('wrote figures/rq3.pdf')

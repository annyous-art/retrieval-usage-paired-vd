"""Figure 4 (MegaVul Java pairs): change in net P-C and in the YES rate on the 620 pairs.
Data: analysis/fse_revision_20260923/java_figure_data.json (commit-cluster bootstrap).
Run from the repository root."""
import json
import sys

sys.path.insert(0, 'figures')
from style import (A, INK, MODELS, WIDTH, bands, estimate, frame, header, left, legend, plt,  # noqa: E402
                   row_labels, spanner)

NAMES = {'gpt-5.5': 'GPT-5.5', 'glm-5.1': 'GLM-5.1'}
# heading of a group of rows, then (mode, label, comparison in java_figure_data.json)
GROUPS = [('Relative to the two-shot prompt',
           [('A', 'Two demonstrations', 'Demonstrations vs. Java two-shot'),
            ('B', 'Code chunks', 'Chunks vs. Java two-shot')]),
          ('Relative to the empty evidence slot',
           [('C', 'Code pair', 'Code pair vs. no retrieval'),
            ('D', 'Knowledge', 'Knowledge vs. no retrieval'),
            ('E', 'CWE entries, code query', 'CWE entries vs. no retrieval'),
            ('E', 'CWE entries, description query', 'CWE entries, description query vs. no retrieval'),
            ('E', 'CVE descriptions, description query', 'CVE descriptions vs. no retrieval')]),
          ('Relative to the code pair of C',
           [('D', 'Knowledge', 'Knowledge vs. code pair')])]
data = json.load(open(A + 'java_figure_data.json'))
get = {(d['comparison'], d['model']): d for d in data}
assert {c for _, members in GROUPS for _, _, c in members} == {d['comparison'] for d in data}

heads, rows, ys, y = [], [], [], 0.0
for title, members in GROUPS:
    heads.append((y, title))
    y += 1.0
    for r in members:
        rows.append(r)
        ys.append(y)
        y += 1.0
    y += 0.45

LABEL_W = 146
XLIM = {'net': (-10.5, 10.5), 'YES': (-22.5, 9.5)}  # the same length per percentage point in all panels
TICKS = {'net': [-5, 0, 5], 'YES': [-20, -10, 0]}
fig = plt.figure(figsize=(WIDTH, 3.05))
gs = fig.add_gridspec(1, 4, width_ratios=[b - a for a, b in XLIM.values() for _ in NAMES], wspace=0.1,
                      left=left(LABEL_W), right=0.995, top=0.8, bottom=0.215)
axs = [fig.add_subplot(gs[0, i]) for i in range(4)]
for ax, (key, m) in zip(axs, [(k, m) for k in XLIM for m in NAMES]):
    bands(ax, ys, [r[0] for r in rows])
    for yy, (_, _, comp) in zip(ys, rows):
        before, after, (lo, hi), diff = get[(comp, m)][key]
        estimate(ax, yy, diff, lo, hi, NAMES[m])
    ax.axvline(0, color=INK, lw=0.8, zorder=2)
    ax.set_xlim(*XLIM[key])
    ax.set_xticks(TICKS[key])
    ax.set_ylim(ys[-1] + 0.6, heads[0][0] - 0.5)
    ax.set_title(NAMES[m], pad=3, color=MODELS[NAMES[m]], fontweight='bold', fontsize=8.5)
    frame(ax)
row_labels(axs[0], ys, [r[0] for r in rows], [r[1] for r in rows], LABEL_W)
for yy, title in heads:
    header(axs[0], yy, title, LABEL_W)
spanner(fig, axs[0], axs[1], '(a) Change in net P-C', 0.87)
spanner(fig, axs[2], axs[3], '(b) Change in YES rate', 0.87)
fig.text((axs[0].get_position().x0 + axs[3].get_position().x1) / 2, 0.092, 'percentage points', ha='center', color=INK)
legend(fig, NAMES.values())
for ext in ('pdf', 'png'):
    fig.savefig(f'figures/java.{ext}', dpi=300, bbox_inches='tight')
print('wrote figures/java.pdf')

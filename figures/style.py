"""Shared style of the paper's figures: Linux Libertine (from a TeX Live installation, if present),
the model colours and markers, and the layout used by all three figures: one estimate per row and
panel, grey and hollow when its 95% interval includes zero, in the model's colour and filled when not."""
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

fonts = glob.glob('/usr/local/texlive/*/texmf-dist/fonts/opentype/public/libertine/LinLibertine_*.otf') + \
    glob.glob('/usr/share/texlive/texmf-dist/fonts/opentype/public/libertine/LinLibertine_*.otf')
for f in fonts:
    fm.fontManager.addfont(f)
plt.rcParams.update({'font.family': 'Linux Libertine O' if fonts else 'serif', 'font.size': 8.5,
                     'axes.linewidth': 0.6, 'pdf.fonttype': 42})
INK, MUTED, GRID, NS, BAND = '#1f1f1e', '#6b6a64', '#dddcd6', '#a3a29c', '#f3f2ee'
MODELS = {'GPT-5.5': '#2a78d6', 'GLM-5.1': '#eb6834', 'Claude Opus 4.7': '#1baf7a', 'DeepSeek-V4-Pro': '#8b5cf6'}
MARKERS = {'GPT-5.5': ('o', 4.8), 'GLM-5.1': ('s', 4.3), 'Claude Opus 4.7': ('^', 5.2), 'DeepSeek-V4-Pro': ('D', 4.0)}
A = 'analysis/fse_revision_20260923/'
WIDTH = 5.5  # figure width in inches (text width of the paper)


def left(label_w):
    """Left edge of the first panel, as a fraction of the figure width, for a label column of `label_w` points."""
    return (label_w + 2) / 72 / WIDTH


def estimate(ax, y, d, lo, hi, model):
    """Point estimate with its 95% interval."""
    marker, ms = MARKERS[model]
    if lo > 0 or hi < 0:
        ax.plot([lo, hi], [y, y], color=MODELS[model], lw=1.6, solid_capstyle='butt', zorder=3)
        ax.plot([d], [y], marker, ms=ms, color=MODELS[model], zorder=4)
    else:
        ax.plot([lo, hi], [y, y], color=NS, lw=1.1, solid_capstyle='butt', zorder=3)
        ax.plot([d], [y], marker, ms=ms - 0.8, mfc='white', mec=NS, mew=1.0, zorder=4)


def frame(ax):
    """Light vertical grid, no box, no y ticks."""
    ax.grid(axis='x', color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis='x', length=2.5, colors=INK)
    ax.set_yticks([])


def row_labels(ax, ys, modes, names, width):
    """Left-aligned row labels in two columns (mode letter once per group, configuration), `width` points wide."""
    for i, (y, m, n) in enumerate(zip(ys, modes, names)):
        if m and (i == 0 or m != modes[i - 1] or y - ys[i - 1] > 1.01):
            ax.annotate(m, (0, y), xycoords=('axes fraction', 'data'), xytext=(-width, 0), textcoords='offset points',
                        ha='left', va='center', color=INK, fontweight='bold')
        ax.annotate(n, (0, y), xycoords=('axes fraction', 'data'), xytext=(-width + (12 if m is not None else 0), 0),
                    textcoords='offset points', ha='left', va='center', color=INK)


def header(ax, y, text, width):
    """Italic heading of a group of rows, in the label column."""
    ax.annotate(text, (0, y), xycoords=('axes fraction', 'data'), xytext=(-width, 0), textcoords='offset points',
                ha='left', va='center', color=MUTED, fontstyle='italic')


def bands(ax, ys, modes):
    """Shade every other mode, so that a row can be followed across the panels."""
    start, shade = 0, True
    for k in range(1, len(modes) + 1):
        if k == len(modes) or modes[k] != modes[start] or ys[k] - ys[k - 1] > 1.01:
            if shade:
                ax.axhspan(ys[start] - 0.5, ys[k - 1] + 0.5, color=BAND, lw=0, zorder=0)
            start, shade = k, not shade


def spanner(fig, axl, axr, text, y):
    """Heading over the panels from `axl` to `axr`, with a rule below it."""
    p0, p1 = axl.get_position(), axr.get_position()
    fig.text((p0.x0 + p1.x1) / 2, y, text, ha='center', va='bottom', fontweight='bold', color=INK)
    fig.add_artist(Line2D([p0.x0 + 0.008, p1.x1 - 0.008], [y - 0.006] * 2, color=INK, lw=0.5))


def legend(fig, models, per_model=False, prefix=''):
    """Legend below the panels: the grey hollow marker, then either one entry per model (when the figure
    also needs the model colours) or the filled markers of all models as one entry."""
    grey = Line2D([], [], color=NS, marker='o', ms=4.0, mfc='white', mec=NS, mew=1.0, lw=1.1)
    if per_model:
        hs = [grey] + [Line2D([], [], color=MODELS[m], marker=MARKERS[m][0], ms=MARKERS[m][1], lw=1.6) for m in models]
        labels = [prefix + 'interval includes zero'] + list(models)
    else:
        hs = [grey, tuple(Line2D([], [], ls='', color=MODELS[m], marker=MARKERS[m][0], ms=MARKERS[m][1]) for m in models)]
        labels = [prefix + 'interval includes zero', prefix + 'interval excludes zero']
    fig.legend(hs, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=len(hs), fontsize=7.8, frameon=False,
               handletextpad=0.6, columnspacing=1.8, handlelength=1.6 if per_model else max(1.8, 1.15 * len(models)),
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)})

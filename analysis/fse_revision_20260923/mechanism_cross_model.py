"""Replicate the evidence/focus mechanism on GPT-5.5 next to GLM-5.1 (offline).

For each chunk-RAG run: identical-evidence rate, pair outcomes when the evidence is
identical vs. different, and when the focus hits vs. misses the changed lines.
Differences use a pair-preserving commit-cluster bootstrap (3000 draws, seed 20260920);
the length-adjusted difference weights within-length-bin differences by pairs per bin.
target_focus runs show the focus region but no retrieved evidence, so there the
"evidence" is the focus region itself.

Run from the repository root: python3 analysis/fse_revision_20260923/mechanism_cross_model.py
"""
import collections
import json
import re
import sys

import numpy as np

sys.path.insert(0, '.')
from build_query_chunks_sim_rank import build_chunks  # noqa: E402
from evaluate_prompt_outputs import parse_prediction  # noqa: E402

RAW = [json.loads(l) for l in open('data/primevul_test_paired_labeled.jsonl')]
BY_IDX = {}
for r in RAW:
    BY_IDX.setdefault(str(r['idx']), r)
DUP = collections.Counter(str(r['idx']) for r in RAW)
PAIRS = []
for i in range(0, len(RAW), 2):
    a, b = RAW[i], RAW[i + 1]
    if DUP[str(a['idx'])] > 1 or DUP[str(b['idx'])] > 1:
        continue
    v, f = (a, b) if int(a['target']) == 1 else (b, a)
    PAIRS.append((str(v['idx']), str(f['idx']), f"{v['project']}:{v['commit_id']}"))
NCH = {x: len(build_chunks(BY_IDX[x], 6, 2, 0)[0]) for v, f, _ in PAIRS for x in (v, f)}
EDGES = np.quantile(list(NCH.values()), [1 / 3, 2 / 3])
BIN = {x: 'short' if n <= EDGES[0] else ('medium' if n <= EDGES[1] else 'long') for x, n in NCH.items()}

clusters = sorted({c for *_, c in PAIRS})
cid = {c: i for i, c in enumerate(clusters)}
W = np.random.default_rng(20260920).multinomial(len(clusters), np.ones(len(clusters)) / len(clusters), size=3000)

G = 'outdir/prompt_ablation_glm51_test870/glm-5.1_std_cls_'
P = 'outdir/prompt_gpt55_representative_ablation_test870/gpt-5.5_std_cls_'
RUNS = [
    ('BC grouped no-label (+author two-shot)', 'glm-5.1', 'outdir/prompt_baseline_compatible_glm51_test870/glm-5.1_std_cls_grouped_no_label_baseline_compatible_fewshotegTrue_cls0.jsonl'),
    ('BC grouped no-label (+author two-shot)', 'gpt-5.5', P + 'grouped_no_label_baseline_compatible_fewshotegTrue_cls0.jsonl'),
    ('grouped no-label', 'glm-5.1', G + 'grouped_no_label_fewshotegFalse_cls0.jsonl'),
    ('grouped no-label', 'gpt-5.5', P + 'grouped_no_label_rag_fewshotegFalse_cls0.jsonl'),
    ('focus only, no evidence', 'glm-5.1', G + 'target_focus_fewshotegFalse_cls0.jsonl'),
    ('focus only, no evidence', 'gpt-5.5', P + 'target_focus_rag_fewshotegFalse_cls0.jsonl'),
    ('inline chunk label', 'glm-5.1', G + 'inline_chunk_label_fewshotegFalse_cls0.jsonl'),
    # GLM-5.1 on GPT-5.5's saved positive-only chunk-RAG prompts (replay_saved_prompts.py, 2026-09-27)
    ('GLM-5.1 RAG (GPT-5.5 prompts replayed)', 'glm-5.1', 'outdir/replay_rag_positive_only/glm-5.1_std_cls_fewshotegFalse1_RAG_870.jsonl'),
    ('GPT-5.5 RAG (Table main-results)', 'gpt-5.5', 'data/gpt-5.5_std_cls_fewshotegFalse1_RAG_870.jsonl'),
]


def evidence(msg):
    """Evidence block after the focus regions; for focus-only prompts, the focus block."""
    f = msg.find('Focus Regions:')
    if f == -1:
        return None
    m = re.search(r'\n\nRetrieved ', msg[f:])
    return msg[f + m.start():] if m else msg[f:]


def focus(rec):
    q = sorted(rec['query_chunks'], key=lambda q: q.get('query_rank', 0))[0]
    q = q.get('query_chunk', q)
    return q['code_clean'], int(q.get('chunk_label', 0)) == 1


def contrast(rows, key, name):
    """Rate of outcome `name` when key is True vs. False, bootstrap CI of the difference,
    and the length-adjusted difference."""
    s = np.zeros((2, len(clusters))); n = np.zeros((2, len(clusters)))
    for r in rows:
        s[int(r[key]), cid[r['c']]] += r['o'] == name; n[int(r[key]), cid[r['c']]] += 1
    a, b = s[1].sum() / n[1].sum(), s[0].sum() / n[0].sum()
    boot = (W @ s[1]) / (W @ n[1]) - (W @ s[0]) / (W @ n[0])
    num = den = 0.0
    for bin_ in ('short', 'medium', 'long'):
        t = [r for r in rows if r['bin'] == bin_ and r[key]]
        u = [r for r in rows if r['bin'] == bin_ and not r[key]]
        if t and u:
            w = len(t) + len(u)
            num += w * (np.mean([r['o'] == name for r in t]) - np.mean([r['o'] == name for r in u])); den += w
    return {'true': round(float(a), 4), 'false': round(float(b), 4), 'n_true': int(n[1].sum()), 'n_false': int(n[0].sum()),
            'diff': round(float(a - b), 4), 'ci95': [round(float(q), 4) for q in np.quantile(boot, [.025, .975])],
            'diff_length_adjusted': round(float(num / den), 4) if den else None}


OUT = {'pairs': len(PAIRS), 'runs': []}
EVID = {}
for config, model, path in RUNS:
    recs = {str(r.get('query_idx', r.get('idx'))): r for r in (json.loads(l) for l in open(path))}
    second = sum('\nRegion 2:' in recs[x]['messages'][-1]['content'] for v, f, _ in PAIRS for x in (v, f))
    rows = []
    for v, f, c in PAIRS:
        ev, ef = evidence(recs[v]['messages'][-1]['content']), evidence(recs[f]['messages'][-1]['content'])
        (_, hv), (_, hf) = focus(recs[v]), focus(recs[f])
        pv, pf = parse_prediction(recs[v].get('response')), parse_prediction(recs[f].get('response'))
        o = {(1, 0): 'P-C', (1, 1): 'P-V', (0, 0): 'P-B', (0, 1): 'P-R'}.get((pv, pf), 'unknown')
        rows.append({'same': ev == ef, 'hit': hv or hf, 'o': o, 'c': c, 'bin': BIN[v]})
        EVID[(config, model, v)] = ev; EVID[(config, model, f)] = ef
    OUT['runs'].append({
        'config': config, 'model': model, 'file': path, 'prompts_with_second_region': second,
        'unknown_pairs': sum(r['o'] == 'unknown' for r in rows),
        'outcomes': dict(collections.Counter(r['o'] for r in rows)),
        'identical_evidence': round(float(np.mean([r['same'] for r in rows])), 4),
        'focus_hit_pairs': sum(r['hit'] for r in rows),
        'PC_by_identical_evidence': contrast(rows, 'same', 'P-C'),
        'PC_by_focus_hit': contrast(rows, 'hit', 'P-C'),
        'PV_by_focus_hit': contrast(rows, 'hit', 'P-V'),
    })

# ---------- difficulty control ----------
# Are pairs whose evidence differs (or whose focus hits the patch) simply easier? Compare
# P-C of no-retrieval runs on the same pair subsets. Subsets come from the shared BC focus
# (identical for the three BC-derived configurations and both models) and from the
# GPT-5.5 RAG run of the main table.
import csv  # noqa: E402

BASE = collections.defaultdict(dict)
for r in csv.DictReader(open('outdir/analysis/per_sample_feature_table.csv')):
    for col, name in [('zero_shot_pred', 'glm zero-shot'), ('yes_only_pred', 'glm YES-only'),
                      ('author_no_yes_pred', 'glm author two-shot')]:
        try:
            BASE[name].setdefault(str(r['idx']), int(float(r[col])))
        except (TypeError, ValueError):
            pass
for l in open('data/gpt-5.5_std_cls_logprobsFalse_fewshotegTrue1_baseline_870.jsonl'):
    r = json.loads(l)
    p = parse_prediction(r.get('response'))
    if p is not None:
        BASE['gpt fixed few-shot'].setdefault(str(r.get('query_idx', r.get('idx'))), p)


def subset_flags(path):
    recs = {str(r.get('query_idx', r.get('idx'))): r for r in (json.loads(l) for l in open(path))}
    out = []
    for v, f, c in PAIRS:
        ev, ef = evidence(recs[v]['messages'][-1]['content']), evidence(recs[f]['messages'][-1]['content'])
        out.append({'same': ev == ef, 'hit': focus(recs[v])[1] or focus(recs[f])[1], 'c': c, 'bin': BIN[v]})
    return out


OUT['difficulty_control'] = {}
for sub_name, path in [('BC focus', RUNS[0][2]), ('GPT-5.5 RAG focus', RUNS[-1][2])]:
    flags = subset_flags(path)
    for base, pred in BASE.items():
        rows = []
        for fl, (v, f, c) in zip(flags, PAIRS):
            o = {(1, 0): 'P-C', (1, 1): 'P-V', (0, 0): 'P-B', (0, 1): 'P-R'}.get((pred.get(v), pred.get(f)), 'unknown')
            rows.append({**fl, 'o': o})
        OUT['difficulty_control'][f'{sub_name} | {base}'] = {
            'unknown_pairs': sum(r['o'] == 'unknown' for r in rows),
            'PC_by_identical_evidence': contrast(rows, 'same', 'P-C'),
            'PC_by_focus_hit': contrast(rows, 'hit', 'P-C')}
for k, r in OUT['difficulty_control'].items():
    e, h = r['PC_by_identical_evidence'], r['PC_by_focus_hit']
    print(f"CONTROL {k:45s} unk={r['unknown_pairs']} PC|same={e['true']:.3f} PC|diff={e['false']:.3f} d={e['diff']:+.3f} {e['ci95']} | PC|hit={h['true']:.3f} PC|miss={h['false']:.3f} d={h['diff']:+.3f} {h['ci95']} adj={h['diff_length_adjusted']}")

# ---------- within-subset effect of the retrieved evidence ----------
# Same pairs, two prompts that differ only in the retrieved evidence (or, for the
# main-table rows, retrieval prompt vs. its no-retrieval baseline). Subsets are defined
# by the retrieval run. Delta P-C = retrieval run minus control run.
def preds(path):
    out = {}
    for l in open(path):
        r = json.loads(l)
        out.setdefault(str(r.get('query_idx', r.get('idx'))), parse_prediction(r.get('response')))
    return out


def pc_vec(pred):
    return np.array([pred.get(v) == 1 and pred.get(f) == 0 for v, f, _ in PAIRS], float)


def delta(a, b, mask):
    s = np.zeros((2, len(clusters))); n = np.zeros(len(clusters))
    for i, (_, _, c) in enumerate(PAIRS):
        if mask[i]:
            s[0, cid[c]] += a[i]; s[1, cid[c]] += b[i]; n[cid[c]] += 1
    d = (W @ s[0] - W @ s[1]) / (W @ n)
    return {'pairs': int(mask.sum()), 'retrieval': round(float(a[mask].mean()), 4), 'control': round(float(b[mask].mean()), 4),
            'delta': round(float(a[mask].mean() - b[mask].mean()), 4),
            'ci95': [round(float(q), 4) for q in np.quantile(d, [.025, .975])]}


COMPARE = [
    ('glm-5.1: grouped no-label vs focus only', RUNS[2][2], preds(RUNS[4][2])),
    ('gpt-5.5: grouped no-label vs focus only', RUNS[3][2], preds(RUNS[5][2])),
    ('glm-5.1: BC grouped no-label vs author two-shot', RUNS[0][2], BASE['glm author two-shot']),
    ('gpt-5.5: BC grouped no-label vs fixed few-shot', RUNS[1][2], BASE['gpt fixed few-shot']),
    ('gpt-5.5: RAG (main table) vs fixed few-shot', RUNS[-1][2], BASE['gpt fixed few-shot']),
    ('glm-5.1: RAG (GPT-5.5 prompts replayed) vs author two-shot', RUNS[-2][2], BASE['glm author two-shot']),
]
OUT['evidence_effect_within_subsets'] = {}
for name, path, ctrl in COMPARE:
    flags = subset_flags(path)
    a, b = pc_vec(preds(path)), pc_vec(ctrl)
    same = np.array([x['same'] for x in flags]); hit = np.array([x['hit'] for x in flags])
    res = {k: delta(a, b, m) for k, m in [('all', np.ones(len(PAIRS), bool)), ('identical_evidence', same),
                                           ('different_evidence', ~same), ('focus_hit', hit), ('focus_miss', ~hit)]}
    OUT['evidence_effect_within_subsets'][name] = res
    print('DELTA', name)
    for k, v in res.items():
        print(f"      {k:20s} n={v['pairs']:3d} retrieval={v['retrieval']:.3f} control={v['control']:.3f} dPC={v['delta']:+.3f} {v['ci95']}")

# the two models received the same evidence wherever the configuration is shared
OUT['same_evidence_across_models'] = {}
for config in {c for c, m, _ in RUNS}:
    if {(config, 'glm-5.1'), (config, 'gpt-5.5')} <= {(c, m) for c, m, _ in RUNS}:
        xs = [x for v, f, _ in PAIRS for x in (v, f)]
        OUT['same_evidence_across_models'][config] = round(float(np.mean([EVID[(config, 'glm-5.1', x)] == EVID[(config, 'gpt-5.5', x)] for x in xs])), 4)

json.dump(OUT, open('analysis/fse_revision_20260923/mechanism_cross_model.json', 'w'), indent=1)
for r in OUT['runs']:
    e, h = r['PC_by_identical_evidence'], r['PC_by_focus_hit']
    print(f"{r['config'][:38]:38s} {r['model']:8s} unk={r['unknown_pairs']:2d} reg2={r['prompts_with_second_region']} "
          f"same={r['identical_evidence']:.3f} PC|same={e['true']:.3f} PC|diff={e['false']:.3f} d={e['diff']:+.3f} {e['ci95']} adj={e['diff_length_adjusted']} | "
          f"hit={r['focus_hit_pairs']} PC|hit={h['true']:.3f} PC|miss={h['false']:.3f} d={h['diff']:+.3f} {h['ci95']} adj={h['diff_length_adjusted']} "
          f"PV|hit={r['PV_by_focus_hit']['true']:.3f} PV|miss={r['PV_by_focus_hit']['false']:.3f}")
print('same evidence across models:', OUT['same_evidence_across_models'])

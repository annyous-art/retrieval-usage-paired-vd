"""GPT-5.5 prompt baselines (the zero-shot row of Table 3) and std-setting variants on the 431 resolvable pairs.

The same prompts as the GLM-5.1 rows of glm_std_prompts.py (zero-shot and YES-only replayed from the
saved GLM-5.1 prompts with replay_saved_prompts.py; PrimeVul two-shot, retrieved demonstrations and
BC from the earlier GPT-5.5 runs). Methods GPT-5.5 has not run are left out. Offline.

Run from the repository root: python3 analysis/fse_revision_20260923/gpt_std_prompts.py
"""
import collections
import json
import sys

sys.path.insert(0, '.')
from evaluate_prompt_outputs import parse_prediction  # noqa: E402

RAW = [json.loads(l) for l in open('data/primevul_test_paired_labeled.jsonl')]
dup = collections.Counter(str(r['idx']) for r in RAW)
PAIRS = []
for i in range(0, len(RAW), 2):
    a, b = RAW[i], RAW[i + 1]
    if dup[str(a['idx'])] > 1 or dup[str(b['idx'])] > 1:
        continue
    v, f = (a, b) if int(a['target']) == 1 else (b, a)
    PAIRS.append((str(v['idx']), str(f['idx'])))


def jsonl_preds(path):
    m = {}
    for l in open(path):
        r = json.loads(l)
        m.setdefault(str(r.get('query_idx', r.get('idx'))), parse_prediction(r.get('response')))
    return m


T = 'outdir/replay_prompt_baselines_gpt55/gpt-5.5_std_cls_logprobsFalse_'
METHODS = {
    'All-vulnerable constant': {x: 1 for v, f in PAIRS for x in (v, f)},
    'Zero-shot': jsonl_preds(T + 'fewshotegFalse_none.jsonl'),
    'YES-only few-shot': jsonl_preds(T + 'fewshotegTrue_yes_only.jsonl'),
    'PrimeVul two-shot': jsonl_preds('data/gpt-5.5_std_cls_logprobsFalse_fewshotegTrue1_baseline_870.jsonl'),
    'Retrieved demonstrations': jsonl_preds('outdir/replay_a_icl_top2/gpt-5.5_std_cls_fewshotegTrue_ICL_870.jsonl'),
    'Balanced grouped no-label (BC)': jsonl_preds('outdir/prompt_gpt55_representative_ablation_test870/gpt-5.5_std_cls_grouped_no_label_baseline_compatible_fewshotegTrue_cls0.jsonl'),
}


def row(m):
    c = collections.Counter(); tp = fp = fn = tn = 0
    for v, f in PAIRS:
        a, b = m.get(v), m.get(f)
        if a is None or b is None:
            continue
        c[{(1, 0): 'P-C', (1, 1): 'P-V', (0, 0): 'P-B', (0, 1): 'P-R'}[(a, b)]] += 1
        tp += a == 1; fn += a == 0; fp += b == 1; tn += b == 0
    n = sum(c.values())
    return {'pairs': n, 'Acc.': round((tp + tn) / (2 * n), 6), 'F1': round(2 * tp / (2 * tp + fp + fn), 6) if tp else 0.0,
            'YES': round((tp + fp) / (2 * n), 6), **{k: round(c[k] / n, 6) for k in ('P-C', 'P-V', 'P-B', 'P-R')}}


OUT = {'pairs': len(PAIRS), 'rows': {name: row(m) for name, m in METHODS.items()}}
json.dump(OUT, open('analysis/fse_revision_20260923/gpt_std_prompts.json', 'w'), indent=1)
for name, r in OUT['rows'].items():
    print(f"{name:32s} n={r['pairs']} " + ' '.join(f"{k} {r[k]:.3f}" for k in ('Acc.', 'F1', 'YES', 'P-C', 'P-V', 'P-B', 'P-R')))

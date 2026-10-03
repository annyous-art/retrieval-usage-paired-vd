"""Supplementary table of an earlier draft: presentation variants with identical retrieval, GLM-5.1 and GPT-5.5 side by side.

The 14 balanced chunk variants retrieve the same items per target and differ only in how the
items are presented; both models receive byte-identical prompts (GPT-5.5's missing variants were
replayed from the saved GLM-5.1 prompts with replay_saved_prompts.py). For each variant and model:
F1 and YES rate on the functions, and P-C on the pairs, of the 431 resolvable pairs, as in
ablation_cluster_ci.py. The PrimeVul two-shot prompt is listed for reference. Offline.

Run from the repository root: python3 analysis/fse_revision_20260923/presentation_variants.py
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

GA, GC, GB = ('outdir/prompt_ablation_glm51_test870/glm-5.1_std_cls_%s_fewshotegFalse_cls0.jsonl',
              'outdir/prompt_fewshot_chunks_glm51_test870/glm-5.1_std_cls_%s_fewshotegTrue_cls0.jsonl',
              'outdir/prompt_baseline_compatible_glm51_test870/glm-5.1_std_cls_%s_baseline_compatible_fewshotegTrue_cls0.jsonl')
R = 'outdir/replay_table8_gpt55/'
PA, PC_, PB = R + 'prompt_ablation/gpt-5.5_std_cls_%s_fewshotegFalse_cls0.jsonl', R + 'prompt_fewshot_chunks/gpt-5.5_std_cls_%s_fewshotegTrue_cls0.jsonl', \
    R + 'prompt_baseline_compatible/gpt-5.5_std_cls_%s_baseline_compatible_fewshotegTrue_cls0.jsonl'
REP = 'outdir/prompt_gpt55_representative_ablation_test870/gpt-5.5_std_cls_'
ROWS = [  # (family, evidence, GLM-5.1 file, GPT-5.5 file)
    ('Prompt', 'PrimeVul two-shot', 'outdir/glm-5.1_std_cls_logprobsFalse_fewshotegTrue_baseline_870.jsonl',
     'data/gpt-5.5_std_cls_logprobsFalse_fewshotegTrue1_baseline_870.jsonl'),
    ('Evidence only', 'Target focus', GA % 'target_focus', REP + 'target_focus_rag_fewshotegFalse_cls0.jsonl'),
    ('Evidence only', 'No-label', GA % 'no_label', PA % 'no_label'),
    ('Evidence only', 'Grouped no-label', GA % 'grouped_no_label', REP + 'grouped_no_label_rag_fewshotegFalse_cls0.jsonl'),
    ('Evidence only', 'Inline func.\\ label', GA % 'inline_function_label', PA % 'inline_function_label'),
    ('Evidence only', 'Inline chunk label', GA % 'inline_chunk_label', 'outdir/grouped_top3/gpt-5.5_std_cls_inline_chunk_label_fewshotegFalse.jsonl'),
    ('Evidence only', 'Grouped chunk label', GA % 'grouped_chunk_label', PA % 'grouped_chunk_label'),
    ('With demos', 'Target focus', GC % 'target_focus', PC_ % 'target_focus'),
    ('With demos', 'Grouped no-label', GC % 'grouped_no_label', PC_ % 'grouped_no_label'),
    ('With demos', 'Grouped chunk label', GC % 'grouped_chunk_label', PC_ % 'grouped_chunk_label'),
    ('With demos', 'Inline func.\\ label', GC % 'inline_function_label', PC_ % 'inline_function_label'),
    ('With demos', 'Inline chunk label', GC % 'inline_chunk_label', PC_ % 'inline_chunk_label'),
    ('BC', 'Target focus', GB % 'target_focus', PB % 'target_focus'),
    ('BC', 'Grouped no-label', GB % 'grouped_no_label', REP + 'grouped_no_label_baseline_compatible_fewshotegTrue_cls0.jsonl'),
    ('BC', 'Inline func.\\ label', GB % 'inline_function_label', PB % 'inline_function_label'),
]


def metrics(path):
    pred = {}
    for l in open(path):
        r = json.loads(l)
        pred.setdefault(str(r.get('query_idx', r.get('idx'))), parse_prediction(r.get('response')))
    tp = fp = fn = yes = known = pc = pairs = 0
    for v, f in PAIRS:
        a, b = pred.get(v), pred.get(f)
        for x in (a, b):
            if x is not None:
                known += 1; yes += x == 1
        if a is not None and b is not None:
            pairs += 1; pc += (a, b) == (1, 0); tp += a == 1; fn += a == 0; fp += b == 1
    return {'F1': round(2 * tp / (2 * tp + fp + fn), 6), 'YES': round(yes / known, 6), 'P-C': round(pc / pairs, 6), 'pairs': pairs}


OUT = {'pairs': len(PAIRS), 'rows': []}
for fam, ev, g, p in ROWS:
    OUT['rows'].append({'family': fam, 'evidence': ev, 'GLM-5.1': metrics(g), 'GPT-5.5': metrics(p)})
variants = [r for r in OUT['rows'] if r['family'] != 'Prompt']
for m in ('GLM-5.1', 'GPT-5.5'):
    OUT[f'range_{m}'] = {k: [min(r[m][k] for r in variants), max(r[m][k] for r in variants)] for k in ('P-C', 'YES', 'F1')}
json.dump(OUT, open('analysis/fse_revision_20260923/presentation_variants.json', 'w'), indent=1)
for r in OUT['rows']:
    g, p = r['GLM-5.1'], r['GPT-5.5']
    print(f"{r['family']:14s} {r['evidence']:22s} GLM F1 {g['F1']:.3f} YES {g['YES']:.3f} P-C {g['P-C']:.3f} | GPT F1 {p['F1']:.3f} YES {p['YES']:.3f} P-C {p['P-C']:.3f}")
for m in ('GLM-5.1', 'GPT-5.5'):
    print(m, 'over the 14 variants:', OUT[f'range_{m}'])

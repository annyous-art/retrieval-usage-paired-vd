"""Code-free and context controls (Table 5) on the 431 resolvable pairs; all four models ran every control.
The controls are a separate batch of runs, so `Code + PrimeVul two-shot` is a second run of the
two-shot prompt of Tables 3 and 4.

Reads the parsed predictions of the bias-control runs and reports, per variant and model, the
YES rate, accuracy, F1, and P-C on the functions of the 431 resolvable pairs (the pair set used
by every other table). `--all870` reproduces the earlier numbers on all 870 functions / 435 pairs.
Offline; no API calls.

Run from the repository root: python3 analysis/fse_revision_20260923/bias_controls.py [--all870]
"""
import collections
import csv
import json
import os
import sys

sys.path.insert(0, '.')
from evaluate_prompt_outputs import parse_prediction  # noqa: E402

E = 'outdir/empirical_study/'
G, GX, GM = E + 'bias_controlled_glm51_test870/glm-5.1_', E + 'bias_controlled_glm51_test870_extra/glm-5.1_', E + 'bias_controlled_glm51_test870_code_metadata/glm-5.1_'
P, C = E + 'bias_controlled_gpt55_test870/gpt-5.5_', E + 'bias_controlled_claude47_test870/claude-opus-4-7_'
D = E + 'bias_controlled_deepseek_test870/deepseek-v4-pro-0813_'
ROWS = [  # (variant, {model: predictions file})
    ('No-code prior', {'GLM-5.1': G + 'no_code_prior', 'GPT-5.5': P + 'no_code_prior', 'Claude Opus 4.7': C + 'no_code_prior', 'DeepSeek-V4-Pro': D + 'no_code_prior'}),
    ('No-code + PrimeVul two-shot', {'GLM-5.1': GX + 'no_code_author_fewshot', 'GPT-5.5': P + 'no_code_author_fewshot', 'Claude Opus 4.7': C + 'no_code_author_fewshot', 'DeepSeek-V4-Pro': D + 'no_code_author_fewshot'}),
    ('No-code + flipped two-shot', {'GLM-5.1': GX + 'no_code_flipped_fewshot', 'GPT-5.5': P + 'no_code_flipped_fewshot', 'Claude Opus 4.7': C + 'no_code_flipped_fewshot', 'DeepSeek-V4-Pro': D + 'no_code_flipped_fewshot'}),
    ('Skeleton only', {'GLM-5.1': G + 'skeleton_only', 'GPT-5.5': P + 'skeleton_only', 'Claude Opus 4.7': C + 'skeleton_only', 'DeepSeek-V4-Pro': D + 'skeleton_only'}),
    ('Metadata without CWE', {'GLM-5.1': GX + 'metadata_no_cwe', 'GPT-5.5': P + 'metadata_no_cwe', 'Claude Opus 4.7': C + 'metadata_no_cwe', 'DeepSeek-V4-Pro': D + 'metadata_no_cwe'}),
    ('Metadata with CWE', {'GLM-5.1': G + 'metadata_only', 'GPT-5.5': P + 'metadata_only', 'Claude Opus 4.7': C + 'metadata_only', 'DeepSeek-V4-Pro': D + 'metadata_only'}),
    ('CWE only', {'GLM-5.1': GX + 'cwe_only', 'GPT-5.5': P + 'cwe_only', 'Claude Opus 4.7': C + 'cwe_only', 'DeepSeek-V4-Pro': D + 'cwe_only'}),
    ('Code + metadata without CWE', {'GLM-5.1': GM + 'code_metadata_no_cwe', 'GPT-5.5': P + 'code_metadata_no_cwe', 'Claude Opus 4.7': C + 'code_metadata_no_cwe', 'DeepSeek-V4-Pro': D + 'code_metadata_no_cwe'}),
    ('Code + metadata with CWE', {'GLM-5.1': GM + 'code_metadata', 'GPT-5.5': P + 'code_metadata', 'Claude Opus 4.7': C + 'code_metadata', 'DeepSeek-V4-Pro': D + 'code_metadata'}),
    ('Code + PrimeVul two-shot', {'GLM-5.1': G + 'code_author_fewshot', 'GPT-5.5': P + 'code_author_fewshot', 'Claude Opus 4.7': C + 'code_author_fewshot', 'DeepSeek-V4-Pro': D + 'code_author_fewshot'}),
    ('Code + flipped two-shot', {'GLM-5.1': G + 'code_author_flipped_fewshot', 'GPT-5.5': P + 'code_author_flipped_fewshot', 'Claude Opus 4.7': C + 'code_author_flipped_fewshot', 'DeepSeek-V4-Pro': D + 'code_author_flipped_fewshot'}),
]

all870 = '--all870' in sys.argv
RAW = [json.loads(l) for l in open('data/primevul_test_paired_labeled.jsonl')]
dup = collections.Counter(str(r['idx']) for r in RAW)
PAIRS = []
for i in range(0, len(RAW), 2):
    a, b = RAW[i], RAW[i + 1]
    if not all870 and (dup[str(a['idx'])] > 1 or dup[str(b['idx'])] > 1):
        continue
    v, f = (a, b) if int(a['target']) == 1 else (b, a)
    PAIRS.append((str(v['idx']), str(f['idx'])))


def metrics(path):
    pred = {}
    if os.path.exists(path + '_predictions.csv'):
        for r in csv.DictReader(open(path + '_predictions.csv', encoding='utf-8')):
            pred.setdefault(str(r['idx']), int(float(r['prediction'])) if r['prediction'] in ('0', '1', '0.0', '1.0') else None)
    else:  # replayed runs keep only the raw answers
        for line in open(path + '.jsonl', encoding='utf-8'):
            r = json.loads(line)
            pred.setdefault(str(r.get('query_idx', r.get('idx'))), parse_prediction(r.get('response')))
    tp = fp = tn = fn = pc = n_pairs = 0
    for v, f in PAIRS:
        a, b = pred.get(v), pred.get(f)
        if a is not None: tp += a == 1; fn += a == 0
        if b is not None: fp += b == 1; tn += b == 0
        if a is not None and b is not None:
            n_pairs += 1; pc += (a, b) == (1, 0)
    n = tp + fp + tn + fn
    return {'YES': round((tp + fp) / n, 3), 'Acc.': round((tp + tn) / n, 3),
            'F1': round(2 * tp / (2 * tp + fp + fn), 3) if tp else 0.0, 'P-C': round(pc / n_pairs, 3),
            'functions': n, 'pairs': n_pairs}


OUT = {'pairs': len(PAIRS), 'basis': '870 functions / 435 pairs' if all870 else '431 resolvable pairs', 'rows': {}}
for variant, files in ROWS:
    OUT['rows'][variant] = {m: metrics(p) for m, p in files.items()}
if not all870:
    json.dump(OUT, open('analysis/fse_revision_20260923/bias_controls.json', 'w'), indent=1)
print(OUT['basis'])
for variant, res in OUT['rows'].items():
    print(f'{variant:30s}', ' | '.join(f"{m}: YES {r['YES']:.3f} Acc {r['Acc.']:.3f} F1 {r['F1']:.3f} P-C {r['P-C']:.3f}" for m, r in res.items()))

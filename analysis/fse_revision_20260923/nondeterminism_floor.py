"""Nondeterminism floor for P-C: pair outcomes on pairs whose two members send identical prompts.

In a code-free control, both members of most pairs receive byte-identical prompts, so any
difference between their answers comes from the model, not from the input. This counts those
pairs among the 431 resolvable pairs and how many of them fall into each pair outcome, for the
metadata-without-CWE control of each of the four models (the P-C range in Threats to Validity).
Offline; reads the recorded prompts and answers of the bias-control runs.

Run from the repository root: python3 analysis/fse_revision_20260923/nondeterminism_floor.py [RUN.jsonl ...]
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

RUNS = sys.argv[1:] or ['outdir/empirical_study/bias_controlled_glm51_test870_extra/glm-5.1_metadata_no_cwe.jsonl',
                        'outdir/empirical_study/bias_controlled_gpt55_test870/gpt-5.5_metadata_no_cwe.jsonl',
                        'outdir/empirical_study/bias_controlled_claude47_test870/claude-opus-4-7_metadata_no_cwe.jsonl',
                        'outdir/empirical_study/bias_controlled_deepseek_test870/deepseek-v4-pro-0813_metadata_no_cwe.jsonl']


def floor(path):
    run = {}
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        run.setdefault(str(r['idx']), r)
    same = [(v, f) for v, f in PAIRS if v in run and f in run and run[v]['messages'] == run[f]['messages']]
    out = collections.Counter()
    for v, f in same:
        a, b = parse_prediction(run[v]['response']), parse_prediction(run[f]['response'])
        out[{(1, 0): 'P-C', (1, 1): 'P-V', (0, 0): 'P-B', (0, 1): 'P-R'}.get((a, b), 'unknown')] += 1
    return {'run': path, 'pairs': len(PAIRS), 'identical_prompt_pairs': len(same), 'outcomes': dict(out),
            'P-C_rate_on_identical': round(out['P-C'] / len(same), 4) if same else None,
            'P-R_rate_on_identical': round(out['P-R'] / len(same), 4) if same else None}


res = [floor(p) for p in RUNS]
json.dump(res, open('analysis/fse_revision_20260923/nondeterminism_floor.json', 'w'), indent=1)
for r in res:
    print(r)

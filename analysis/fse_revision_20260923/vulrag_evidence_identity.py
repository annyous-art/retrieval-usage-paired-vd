"""(2) Does Vul-RAG give the vulnerable and the fixed member the same knowledge? (offline)

Reads the Vul-RAG GPT-5.5 run: retrieved.json (top-3 knowledge per function and the
three channel rankings: LLM purpose text, LLM function text, code), queries.json (the
LLM-generated descriptions used as queries) and prediction_items/ (detect results, may
still be incomplete). Pairs touching the two duplicated test indices are excluded.

Run from the repository root: python3 analysis/fse_revision_20260923/vulrag_evidence_identity.py [run_dir/]
"""
import collections
import difflib
import glob
import json
import statistics
import sys

D = sys.argv[1] if len(sys.argv) > 1 else 'outdir/vulrag_gpt-5.5/'
R = {str(x['row_id']): x for x in json.load(open(D + 'retrieved.json'))}
Q = {str(x['row_id']): x for x in json.load(open(D + 'queries.json'))}
T = json.load(open(D + 'targets.json'))
PRED = {}
for p in glob.glob(D + 'prediction_items/*.json'):
    x = json.load(open(p))
    PRED[str(x['row_id'])] = int(x['prediction'])

dup = collections.Counter(str(x['idx']) for x in T)
by_pair = collections.defaultdict(dict)
for x in T:
    by_pair[x['pair_id']][int(x['target'])] = str(x['row_id'])
PAIRS = [(d[1], d[0]) for d in by_pair.values()
         if 1 in d and 0 in d and dup[next(str(t['idx']) for t in T if str(t['row_id']) == d[1])] == 1
         and dup[next(str(t['idx']) for t in T if str(t['row_id']) == d[0])] == 1]
n = len(PAIRS)
ids = lambda row, k=3: R[row]['knowledge_ids'][:k]
OUT = {'pairs': n, 'knowledge': {
    'same_ordered_top3': sum(ids(v) == ids(f) for v, f in PAIRS) / n,
    'same_set_top3': sum(set(ids(v)) == set(ids(f)) for v, f in PAIRS) / n,
    'same_top1': sum(ids(v, 1) == ids(f, 1) for v, f in PAIRS) / n,
    'any_overlap_top3': sum(bool(set(ids(v)) & set(ids(f))) for v, f in PAIRS) / n,
}}
OUT['channels'] = {}
for ci, name in enumerate(['purpose_text', 'function_text', 'code']):
    L = lambda row: R[row]['trace']['channel_cve_rankings'][ci]
    OUT['channels'][name] = {
        'same_top1': sum(L(v)[:1] == L(f)[:1] for v, f in PAIRS) / n,
        'same_top3': sum(L(v)[:3] == L(f)[:3] for v, f in PAIRS) / n,
        'mean_jaccard_top20': sum(len(set(L(v)) & set(L(f))) / len(set(L(v)) | set(L(f))) for v, f in PAIRS) / n,
    }
OUT['query_text'] = {
    'purpose_identical': sum(Q[v]['purpose'] == Q[f]['purpose'] for v, f in PAIRS) / n,
    'function_identical': sum(Q[v]['function'] == Q[f]['function'] for v, f in PAIRS) / n,
    'purpose_similarity_median': statistics.median(difflib.SequenceMatcher(None, Q[v]['purpose'], Q[f]['purpose']).ratio() for v, f in PAIRS),
    'function_similarity_median': statistics.median(difflib.SequenceMatcher(None, Q[v]['function'], Q[f]['function']).ratio() for v, f in PAIRS),
}

done = [(v, f) for v, f in PAIRS if v in PRED and f in PRED]
lab = lambda v, f: {(1, 0): 'P-C', (1, 1): 'P-V', (0, 0): 'P-B', (0, 1): 'P-R'}[(PRED[v], PRED[f])]


def outcomes(sub):
    c = collections.Counter(lab(v, f) for v, f in sub)
    return {'pairs': len(sub), **{k: round(c[k] / len(sub), 4) for k in ('P-C', 'P-V', 'P-B', 'P-R')}} if sub else None


OUT['detect'] = {
    'pairs_with_both_predictions': len(done),
    'yes_rate': sum(PRED[x] for p in done for x in p) / (2 * len(done)) if done else None,
    'all': outcomes(done),
    'same_top3_set': outcomes([p for p in done if set(ids(p[0])) == set(ids(p[1]))]),
    'different_top3_set': outcomes([p for p in done if set(ids(p[0])) != set(ids(p[1]))]),
    'same_top1': outcomes([p for p in done if ids(p[0], 1) == ids(p[1], 1)]),
    'different_top1': outcomes([p for p in done if ids(p[0], 1) != ids(p[1], 1)]),
}
json.dump(OUT, open('analysis/fse_revision_20260923/vulrag_evidence_identity.json', 'w'), indent=1)
print(json.dumps(OUT, indent=1))

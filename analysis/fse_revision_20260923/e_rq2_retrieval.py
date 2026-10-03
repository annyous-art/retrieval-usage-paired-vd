"""RQ2 for the CWE/CVE-entry usage mode (E): intermediate results of the two members of a pair.

Offline, no API calls. Reads the retrieval results written by build_e_retrieval.py and,
for each variant (X2 cwe_code, X3 cwe_desc, X4 cve_desc), reports on the 431 resolvable
pairs:
  * identical retrieval: same top-1 entry, same top-3 set, same top-3 order;
  * similarity scores: |delta| of the top-1 score and of the mean top-3 score within a
    pair, against a between-pair baseline (the vulnerable members of two different pairs,
    matched by a fixed random derangement), as in rq2_similarity_scores.py;
  * CWE hit: whether the top-1 / any top-3 entry carries the pair's labelled CWE (for CVE
    entries: the CWE recorded for that CVE), against the same derangement baseline, with two
    label sources: the CWE field of PrimeVul, and the current NVD mapping of the pair's CVE
    (fetch_nvd_cwe.py; weaknesses from nvd@nist.gov, otherwise from the CNA; NVD-CWE-Other
    and NVD-CWE-noinfo dropped). The labels are used only here, never in retrieval or prompts.
Rates carry 95% intervals from the commit-cluster bootstrap (3000 draws, seed 20260920).

Run from the repository root: python3 analysis/fse_revision_20260923/e_rq2_retrieval.py
"""
import ast
import collections
import json

import numpy as np

E = 'data/e_retrieval/'
TGT = 'outdir/knowledge_rag_gpt-5.5/targets.jsonl'
targets = [json.loads(l) for l in open(TGT)]
raw = [json.loads(l) for l in open('data/primevul_test_paired_labeled.jsonl')]
dup = collections.Counter(str(r['idx']) for r in raw)
label_cwe = {}
for r in raw:
    try:
        label_cwe[str(r['idx'])] = set(ast.literal_eval(r['cwe'])) if isinstance(r['cwe'], str) else set(r['cwe'])
    except (ValueError, SyntaxError):
        label_cwe[str(r['idx'])] = {r['cwe']}

by_pair = collections.defaultdict(dict)
for t in targets:
    by_pair[t['pair_id']][int(t['target'])] = t
PAIRS = []
for pid, m in sorted(by_pair.items()):
    v, f = m[1], m[0]
    if dup[str(v['idx'])] > 1 or dup[str(f['idx'])] > 1:
        continue
    PAIRS.append((v['row_id'], f['row_id'], str(v['cluster']), label_cwe[str(v['idx'])] | label_cwe[str(f['idx'])]))
clusters = sorted({c for _, _, c, _ in PAIRS})
cid = {c: i for i, c in enumerate(clusters)}
rng = np.random.default_rng(20260920)
W = rng.multinomial(len(clusters), np.ones(len(clusters)) / len(clusters), size=3000)
perm = rng.permutation(len(PAIRS))
while np.any(perm == np.arange(len(PAIRS))):
    perm = rng.permutation(len(PAIRS))

cve_cwes_primevul = {json.loads(l)['id']: set(json.loads(l)['cwes']) for l in open(E + 'cve_entries.jsonl')}


def nvd_labels(rec):
    by = [set(c for c in w['cwes'] if c.startswith('CWE-')) for w in rec['weaknesses'] if w['source'] == 'nvd@nist.gov']
    lab = set().union(*by) if by else set()
    if not lab:
        lab = set(c for w in rec['weaknesses'] for c in w['cwes'] if c.startswith('CWE-'))
    return lab


NVD = {r['cve']: nvd_labels(r) for r in map(json.loads, open(E + 'nvd_cwe.jsonl'))}
cve_of = {str(r['idx']): r.get('cve') for r in raw}
LABELS = {'primevul': [lab for _, _, _, lab in PAIRS]}
LABELS['nvd'] = []
for pid, m in sorted(by_pair.items()):
    v, f = m[1], m[0]
    if dup[str(v['idx'])] > 1 or dup[str(f['idx'])] > 1:
        continue
    LABELS['nvd'].append(NVD.get(cve_of[str(v['idx'])], set()) | NVD.get(cve_of[str(f['idx'])], set()))
CVE_CWES = {'primevul': cve_cwes_primevul, 'nvd': {c: NVD.get(c, set()) for c in cve_cwes_primevul}}


def ci(x):
    x = np.asarray(x, float)
    s = np.zeros(len(clusters)); n = np.zeros(len(clusters))
    for val, (_, _, c, _) in zip(x, PAIRS):
        s[cid[c]] += val; n[cid[c]] += 1
    b = (W @ s) / (W @ n)
    return [round(float(x.mean()), 4), [round(float(q), 4) for q in np.quantile(b, [.025, .975])]]


def entry_cwes(variant, eid, source):
    return CVE_CWES[source].get(eid, set()) if variant == 'cve_desc' else {eid}


OUT = {'pairs': len(PAIRS), 'clusters': len(clusters), 'variants': {}}
for variant in ('cwe_code', 'cwe_desc', 'cve_desc'):
    R = {json.loads(l)['row_id']: json.loads(l) for l in open(E + f'retrieval_{variant}.jsonl')}
    res = {
        'same_top1': ci([R[v]['ids'][0] == R[f]['ids'][0] for v, f, _, _ in PAIRS]),
        'same_top3_set': ci([set(R[v]['ids'][:3]) == set(R[f]['ids'][:3]) for v, f, _, _ in PAIRS]),
        'same_top3_order': ci([R[v]['ids'][:3] == R[f]['ids'][:3] for v, f, _, _ in PAIRS]),
    }
    for key, fn in (('top1', lambda r: r['scores'][0]), ('mean_top3', lambda r: float(np.mean(r['scores'][:3])))):
        within = np.array([abs(fn(R[v]) - fn(R[f])) for v, f, _, _ in PAIRS])
        between = np.array([abs(fn(R[PAIRS[i][0]]) - fn(R[PAIRS[j][0]])) for i, j in enumerate(perm)])
        res['score_' + key] = {
            'within_pair_abs_diff_median': round(float(np.median(within)), 5),
            'within_pair_exactly_equal': round(float(np.mean(within == 0)), 4),
            'between_pair_abs_diff_median': round(float(np.median(between)), 5),
            'within_below_between_median': ci(within < np.median(between)),
        }

    for source, labs in LABELS.items():
        def hit(row, labels, k):
            return bool(set().union(*(entry_cwes(variant, e, source) for e in R[row]['ids'][:k])) & labels)
        out = {'pairs_with_label': int(sum(bool(x) for x in labs))}
        for k in (1, 3):
            own = [hit(v, lab, k) or hit(f, lab, k) for (v, f, _, _), lab in zip(PAIRS, labs)]
            base = [hit(PAIRS[j][0], labs[i], k) or hit(PAIRS[j][1], labs[i], k) for i, j in enumerate(perm)]
            out[f'top{k}_either_member'] = {'own_pair': ci(own), 'other_pair_baseline': ci(base)}
            out[f'top{k}_vulnerable_vs_fixed'] = {
                'vulnerable': ci([hit(v, lab, k) for (v, _, _, _), lab in zip(PAIRS, labs)]),
                'fixed': ci([hit(f, lab, k) for (_, f, _, _), lab in zip(PAIRS, labs)])}
        res['cwe_hit_' + source] = out
    OUT['variants'][variant] = res

json.dump(OUT, open('analysis/fse_revision_20260923/e_rq2_retrieval.json', 'w'), indent=1)
print(json.dumps(OUT, indent=1))

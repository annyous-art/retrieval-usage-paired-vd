"""Table 2 rows of the configurations that the other retrieval scripts leave without scores or content match:
function-level retrieved demonstrations (A), the fixed-candidate selection (C, D), and Vul-RAG (D).

Offline, no API calls. Same pairs, derangement baseline, and commit-cluster bootstrap as
e_rq2_retrieval.py (targets of the knowledge_rag run, seed 20260920, 3000 draws).

Score gap: |delta| of the top-1 retrieval score within a pair against the same statistic for the
vulnerable members of two different pairs; "below" = share of pairs whose within-pair gap is below
the between-pair median.
  * A: cosine similarity of SFR-Embedding-Code-400M_R embeddings (CLS pooling, normalized), recomputed
    with the settings of data/second_dataset/build/retrieve_similar_java.py; the recomputed top-2 must
    reproduce the stored demonstrations (checked below).
  * Fixed candidate: IDF-weighted identifier overlap / sqrt(candidate identifier count), recomputed
    with the code of experiments/knowledge_rag/experiment.py; the recomputed top-1 must reproduce the
    stored candidate (checked below).
  * Vul-RAG: the repository stores channel rankings, not raw scores; the score is the fused rank sum
    of the selected top-1 CVE (zero-based ranks, missing rank = list length), as in its trace.
Match: a shown item carries the pair's weakness (current NVD mapping of the pair's CVE, as in
e_rq2_retrieval.py), for the pair's own items vs. the items of another pair (derangement). Shown items:
the two demonstrations (A), the one candidate pair (fixed candidate), the three knowledge items (Vul-RAG),
the top-1 CVE of each Vul-RAG channel (purpose text, function behavior text, code). An item's weakness is
the NVD mapping of its CVE. Vul-RAG is read from the run reported in the paper (vulrag_gpt-5.5).

Usage (repository root): python3 analysis/fse_revision_20260923/rq2_missing_rows.py [EMB_DIR]
EMB_DIR holds test_emb.npy and train_emb.npy (rows in the order of the icl top-5 file and of
data/primevul_train_paired.jsonl). With EMB_DIR, the script recomputes the top-1 scores of A and of
the fixed candidate from the embeddings and the candidate pool, checks them against the stored
retrieval, and saves the scores with the CVEs of the retrieved items to data/retrieval_scores.json.
Without EMB_DIR, it reads that file instead (the replication package ships it, because the
embeddings and the candidate pool hold PrimeVul code), and the recomputation checks are skipped.
"""
import ast
import collections
import gzip
import json
import math
import os
import re
import sys

import numpy as np

EMB = sys.argv[1].rstrip('/') + '/' if len(sys.argv) > 1 else None
E = 'data/e_retrieval/'
KR = 'outdir/knowledge_rag_gpt-5.5/'
VR = 'outdir/vulrag_gpt-5.5/'
ICL = 'icl/primevul_test_ccpp_with_similar_top5.jsonl'

targets = [json.loads(l) for l in open(KR + 'targets.jsonl')]
raw = [json.loads(l) for l in open('data/primevul_test_paired_labeled.jsonl')]
dup = collections.Counter(str(r['idx']) for r in raw)
cve_of = {str(r['idx']): r.get('cve') for r in raw}


def nvd_labels(rec):
    by = [set(c for c in w['cwes'] if c.startswith('CWE-')) for w in rec['weaknesses'] if w['source'] == 'nvd@nist.gov']
    lab = set().union(*by) if by else set()
    if not lab:
        lab = set(c for w in rec['weaknesses'] for c in w['cwes'] if c.startswith('CWE-'))
    return lab


NVD = {r['cve']: nvd_labels(r) for r in map(json.loads, open(E + 'nvd_cwe.jsonl'))}

by_pair = collections.defaultdict(dict)
for t in targets:
    by_pair[t['pair_id']][int(t['target'])] = t
PAIRS, LAB = [], []           # (vulnerable idx, fixed idx, cluster)
for pid, m in sorted(by_pair.items()):
    v, f = m[1], m[0]
    if dup[str(v['idx'])] > 1 or dup[str(f['idx'])] > 1:
        continue
    PAIRS.append((str(v['idx']), str(f['idx']), str(v['cluster'])))
    LAB.append(NVD.get(cve_of[str(v['idx'])], set()) | NVD.get(cve_of[str(f['idx'])], set()))
clusters = sorted({c for *_, c in PAIRS})
cid = {c: i for i, c in enumerate(clusters)}
rng = np.random.default_rng(20260920)
W = rng.multinomial(len(clusters), np.ones(len(clusters)) / len(clusters), size=3000)
perm = rng.permutation(len(PAIRS))
while np.any(perm == np.arange(len(PAIRS))):
    perm = rng.permutation(len(PAIRS))


def ci(x):
    x = np.asarray(x, float)
    s = np.zeros(len(clusters)); n = np.zeros(len(clusters))
    for val, (_, _, c) in zip(x, PAIRS):
        s[cid[c]] += val; n[cid[c]] += 1
    b = (W @ s) / (W @ n)
    return [round(float(x.mean()), 4), [round(float(q), 4) for q in np.quantile(b, [.025, .975])]]


def summarize(score, items, item_cwes, higher_is_better=True):
    """score: idx -> top-1 score; items: idx -> list of shown item ids; item_cwes: id -> set of CWEs."""
    res = {}
    if score is not None:
        within = np.array([abs(score[v] - score[f]) for v, f, _ in PAIRS])
        between = np.array([abs(score[PAIRS[i][0]] - score[PAIRS[j][0]]) for i, j in enumerate(perm)])
        res['score'] = {
            'within_pair_abs_diff_median': round(float(np.median(within)), 5),
            'within_pair_exactly_equal': round(float(np.mean(within == 0)), 4),
            'between_pair_abs_diff_median': round(float(np.median(between)), 5),
            'within_below_between_median': ci(within < np.median(between)),
        }

    def hit(x, lab):
        return bool(set().union(*(item_cwes.get(i, set()) for i in items[x])) & lab) if items[x] else False
    own = [hit(v, lab) or hit(f, lab) for (v, f, _), lab in zip(PAIRS, LAB)]
    base = [hit(PAIRS[j][0], LAB[i]) or hit(PAIRS[j][1], LAB[i]) for i, j in enumerate(perm)]
    res['match_nvd'] = {'pairs_with_label': int(sum(bool(x) for x in LAB)),
                        'own_pair': ci(own), 'other_pair_baseline': ci(base),
                        'vulnerable': ci([hit(v, lab) for (v, _, _), lab in zip(PAIRS, LAB)]),
                        'fixed': ci([hit(f, lab) for (_, f, _), lab in zip(PAIRS, LAB)])}
    res['same_items_set'] = ci([set(items[v]) == set(items[f]) for v, f, _ in PAIRS])
    return res


OUT = {'pairs': len(PAIRS), 'clusters': len(clusters), 'instances': {}}

# ---- A: function-level retrieved demonstrations; fixed candidate (IDF-weighted identifier overlap)
icl = [json.loads(l) for l in open(ICL)]
a_items = {str(r['idx']): [str(s['idx']) for s in r['similar'][:2]] for r in icl}
fc_items = {str(t['idx']): [t['candidate_ids'][0]] for t in targets}
if EMB:
    train = [json.loads(l) for l in open('data/primevul_train_paired.jsonl')]
    Q = np.load(EMB + 'test_emb.npy'); T = np.load(EMB + 'train_emb.npy')
    S = Q @ T.T
    a_score, a_mismatch = {}, 0
    for i, r in enumerate(icl):
        top = np.argsort(-S[i], kind='stable')[:2]
        if [str(train[j]['idx']) for j in top] != a_items[str(r['idx'])]:
            a_mismatch += 1
        a_score[str(r['idx'])] = float(S[i, top[0]])
    # recomputed as in experiment.py
    pool = [json.loads(l) for l in open(KR + 'candidate_pool.jsonl')]
    tokens = lambda code: set(re.findall(r'[A-Za-z_][A-Za-z_0-9]*', code))
    vocab = [tokens(p['members'][0]['func']) for p in pool]
    df = collections.Counter(t for ts in vocab for t in ts)
    idf = {t: math.log((len(vocab) + 1) / (n + 1)) + 1 for t, n in df.items()}
    fc_score, fc_mismatch = {}, 0
    for t in targets:
        q = tokens(t['func'])
        sc = [sum(idf[x] for x in q & ts) / max(1, math.sqrt(len(ts))) for ts in vocab]
        best = min(range(len(pool)), key=lambda j: (-sc[j], pool[j]['pair_id']))
        if pool[best]['pair_id'] != t['candidate_ids'][0]:
            fc_mismatch += 1
        fc_score[str(t['idx'])] = sc[best]
    shown = {i for v in a_items.values() for i in v}
    SCORES = {'A_top1': a_score, 'A_item_cve': {str(r['idx']): r.get('cve') for r in train if str(r['idx']) in shown},
              'FC_top1': fc_score, 'FC_pool_cves': {p['pair_id']: [m.get('cve') for m in p['members']] for p in pool}}
    json.dump(SCORES, open('data/retrieval_scores.json', 'w'), indent=0, sort_keys=True)
else:
    SCORES = json.load(open('data/retrieval_scores.json'))
    a_score, fc_score, a_mismatch, fc_mismatch = SCORES['A_top1'], SCORES['FC_top1'], None, None
train_cwes = {i: NVD.get(c, set()) for i, c in SCORES['A_item_cve'].items()}
pool_cwes = {p: set().union(*(NVD.get(c, set()) for c in cves)) for p, cves in SCORES['FC_pool_cves'].items()}
OUT['instances']['A function-level top-2'] = dict(summarize(a_score, a_items, train_cwes),
                                                 recomputed_top2_mismatch=a_mismatch)
OUT['instances']['Fixed candidate'] = dict(summarize(fc_score, fc_items, pool_cwes),
                                           recomputed_top1_mismatch=fc_mismatch)

# ---- Vul-RAG: fused rank sum of the selected top-1 CVE; knowledge items and channel top-1 CVEs
vt = {str(x['row_id']): str(x['idx']) for x in json.load(open(VR + 'targets.json'))}
cve_cwes = collections.defaultdict(set)
for c in NVD:
    cve_cwes[c] |= NVD[c]
v_score, v_items, ch_items = {}, {}, collections.defaultdict(dict)
for x in json.load(open(VR + 'retrieved.json')):
    idx = vt[str(x['row_id'])]
    tr = ast.literal_eval(x['trace']) if isinstance(x['trace'], str) else x['trace']
    rankings, sel = tr['channel_cve_rankings'], tr['selected_cves']
    def fused(c):
        return sum(rk.index(c) if c in rk else len(rk) for rk in rankings)
    v_score[idx] = float(fused(sel[0]))
    v_items[idx] = list(x['knowledge_ids'][:3])
    for n, rk in enumerate(rankings):
        ch_items[n][idx] = rk[:1]
KB = json.load(open(VR + 'knowledge.json')) if os.path.exists(VR + 'knowledge.json') else json.load(gzip.open(VR + 'knowledge.json.gz', 'rt'))
know_cwes = {k['id']: NVD.get(k.get('CVE_id'), set()) for k in KB}
OUT['instances']['Vul-RAG top-3 knowledge'] = summarize(v_score, v_items, know_cwes, higher_is_better=False)
OUT['instances']['Vul-RAG top-3 knowledge']['score_definition'] = 'fused rank sum of the selected top-1 CVE (lower = better)'
for n, name in enumerate(['purpose channel', 'behavior channel', 'code channel']):
    OUT['instances'][f'Vul-RAG {name} top-1'] = summarize(None, ch_items[n], cve_cwes)

json.dump(OUT, open('analysis/fse_revision_20260923/rq2_missing_rows.json', 'w'), indent=1)
print(json.dumps(OUT, indent=1))

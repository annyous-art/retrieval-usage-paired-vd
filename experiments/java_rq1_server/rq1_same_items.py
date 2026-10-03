"""RQ1 on the Java pairs: how often the two versions of a pair receive the same retrieved items,
for mode B (chunks) and mode E with code queries (CWE entries). No model calls.

B: the evidence block of the main B prompt, i.e. the focus region (top-ranked query chunk) and the
first three vulnerable and three fixed index chunks by rank, as in
run_prompting_sliced_rag.format_baseline_compatible_context(max_query_chunks=1, max_index_matches=3,
evidence_mode='grouped_no_label'). E: the set of the top-3 CWE entries.

Usage:
  python3 rq1_same_items.py --pairs TEST.jsonl --b-results query_results.jsonl --e-results e_cwe_code.jsonl --out summary.json
  python3 rq1_same_items.py --check-prompts PRIMEVUL_B_PROMPTS.jsonl --pairs PRIMEVUL_TEST.jsonl   (validation)
"""
import argparse
import collections
import json


def b_key(query_chunks, k=3):
    qs = sorted((q for q in query_chunks if isinstance(q, dict)), key=lambda q: q.get('query_rank', 0))
    if not qs:
        return None
    q = qs[0]
    ms = sorted((m for m in q.get('index_matches') or [] if isinstance(m, dict)), key=lambda m: m.get('index_rank', 999))
    pol = lambda m: m.get('evidence_polarity')
    pos = tuple(m.get('index_code_clean', '') for m in ms if str(pol(m)) == '1')[:k]
    neg = tuple(m.get('index_code_clean', '') for m in ms if str(pol(m)) == '0')[:k]
    return (q.get('query_code_clean', ''), pos, neg)


def pairs_of(path):
    rows = [json.loads(l) for l in open(path)]
    dup = collections.Counter(str(r['idx']) for r in rows)
    by = collections.defaultdict(dict)
    for r in rows:
        if dup[str(r['idx'])] == 1:
            by[(r.get('project'), r.get('commit_id'), r.get('func_name') or r.get('file_name'), r.get('pair_id'))].setdefault(int(r['target']), str(r['idx']))
    return rows, [(v[1], v[0]) for v in by.values() if 1 in v and 0 in v]


def adjacent_pairs(path):
    rows = [json.loads(l) for l in open(path)]
    dup = collections.Counter(str(r['idx']) for r in rows)
    out = []
    for i in range(0, len(rows) - 1, 2):
        a, b = rows[i], rows[i + 1]
        if dup[str(a['idx'])] > 1 or dup[str(b['idx'])] > 1 or {int(a['target']), int(b['target'])} != {0, 1}:
            continue
        v, f = (a, b) if int(a['target']) == 1 else (b, a)
        out.append((str(v['idx']), str(f['idx'])))
    return out


def rate(keys, pairs):
    ok = [(v, f) for v, f in pairs if keys.get(v) is not None and keys.get(f) is not None]
    same = sum(keys[v] == keys[f] for v, f in ok)
    return {'pairs': len(ok), 'same': same, 'same_pct': round(100 * same / len(ok), 1) if ok else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', required=True, help='paired labeled test JSONL (vulnerable/fixed rows adjacent)')
    ap.add_argument('--b-results', help='query_results.jsonl of build_query_chunks_sim_rank.py')
    ap.add_argument('--e-results', help='e_cwe_code.jsonl of e_cwe_code_java.py')
    ap.add_argument('--check-prompts', help='saved B prompt JSONL with query_chunks (validation on PrimeVul)')
    ap.add_argument('--out')
    a = ap.parse_args()
    pairs = adjacent_pairs(a.pairs)
    res = {'pairs_total': len(pairs)}
    if a.check_prompts:
        keys = {str(r.get('query_idx', r.get('idx'))): b_key(r['query_chunks']) for r in map(json.loads, open(a.check_prompts))}
        res['B_check'] = rate(keys, pairs)
    if a.b_results:
        keys = {}
        for r in map(json.loads, open(a.b_results)):
            qid = str(r.get('query_idx', r.get('query_func_id', r.get('idx'))))
            keys[qid] = b_key(r.get('query_chunks') or r.get('chunks') or [])
        res['B'] = rate(keys, pairs)
    if a.e_results:
        keys = {str(r['idx']): tuple(sorted(r['ids'][:3])) for r in map(json.loads, open(a.e_results))}
        res['E_cwe_code'] = rate(keys, pairs)
    print(json.dumps(res, indent=1))
    if a.out:
        open(a.out, 'w').write(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

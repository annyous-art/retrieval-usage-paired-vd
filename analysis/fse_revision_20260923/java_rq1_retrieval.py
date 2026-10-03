"""RQ1 on the MegaVul Java pairs, and the focus-region comparison with PrimeVul (offline; no API calls).

Same items: the share of the 620 Java pairs whose two members receive the same retrieved items,
as in Table tab:intermediate: A, the set of the two demonstrations; C and D, the retrieved
candidate; E, the set of the top-3 entries (CWE entries with code or description query, CVE
descriptions); B, the inserted evidence (focus region plus the first three vulnerable and three
fixed index chunks), compared through the hashes of data/b_keys/ (tools/export_b_keys.py).

Focus region (mode B): the share of all pairs whose two members receive the same B items (for
PrimeVul, the Same column of mode B in Table tab:intermediate), the share of functions whose focus region contains a changed line (line
labels of the paired files), the pairs in which neither member's focus region does, and among
those the pairs whose two members receive the same B items. PrimeVul is read from the B prompts of
the paper (431 resolvable pairs), MegaVul from the Java B run (620 pairs, labels computed as for
PrimeVul by experiments/java_b_relabel/relabel_java_diff.py). Also the median length (lines) of
the vulnerable functions.

Run from the repository root: python3 analysis/fse_revision_20260923/java_rq1_retrieval.py
"""
import collections
import gzip
import json
import statistics

JAVA = 'data/second_dataset/megavul_java_test_paired_labeled_diff.jsonl'
PRIME = 'data/primevul_test_paired_labeled.jsonl'


def pairs(path):
    rows = [json.loads(l) for l in open(path)]
    dup = collections.Counter(str(r['idx']) for r in rows)
    out = []
    for i in range(0, len(rows) - 1, 2):
        a, b = rows[i], rows[i + 1]
        if dup[str(a['idx'])] > 1 or dup[str(b['idx'])] > 1:
            continue
        v, f = (a, b) if int(a['target']) == 1 else (b, a)
        out.append((v, f))
    return out


def same(keys, P):
    ok = [(v, f) for v, f in P if keys.get(v) is not None and keys.get(f) is not None]
    return {'pairs': len(ok), 'same': sum(keys[v] == keys[f] for v, f in ok),
            'same_pct': round(100 * sum(keys[v] == keys[f] for v, f in ok) / len(ok), 1)}


def b_table(path):
    return {r['idx']: r for r in map(json.loads, gzip.open(path, 'rt'))}


def focus(P, B):
    def hit(r):
        changed = {i + 1 for i, x in enumerate(r['labels']) if int(x) == 1}
        return bool(set(B[str(r['idx'])]['focus_line_ids']) & changed)
    fn = [hit(r) for v, f in P for r in (v, f)]
    neither = [(v, f) for v, f in P if not hit(v) and not hit(f)]
    same_b = sum(B[str(v['idx'])]['b_key_sha256'] == B[str(f['idx'])]['b_key_sha256'] for v, f in neither)
    same_all = sum(B[str(v['idx'])]['b_key_sha256'] == B[str(f['idx'])]['b_key_sha256'] for v, f in P)
    return {'same_B_items_all_pairs': same_all, 'same_B_items_all_pairs_pct': round(100 * same_all / len(P), 1),
            'functions': len(fn), 'function_focus_hit_pct': round(100 * sum(fn) / len(fn), 1),
            'pairs': len(P), 'neither_focus_hits': len(neither), 'neither_pct': round(100 * len(neither) / len(P), 1),
            'same_B_items_among_neither': same_b, 'same_B_items_among_neither_pct': round(100 * same_b / len(neither), 1),
            'vulnerable_function_median_lines': statistics.median(len(v['func'].splitlines()) for v, _ in P)}


JP = pairs(JAVA)
ids = [(str(v['idx']), str(f['idx'])) for v, f in JP]
jl = lambda p: [json.loads(l) for l in open(p)]
A = {str(r['idx']): frozenset(str(s['idx']) for s in r['similar'][:2]) for r in jl('outdir/java_std/gpt-5.5_A_top2.jsonl')}
FC = {str(r['idx']): tuple(r['candidate_ids']) for r in jl('outdir/java_fc_glm-5.1/predictions.jsonl') if r['arm'] == 'no_retrieval'}
E = {arm: {str(r['idx']): frozenset(r['candidate_ids'][:3]) for r in jl(f'outdir/java_e_arm_gpt-5.5_{arm}/predictions.jsonl')}
     for arm in ('cwe_code', 'cwe_desc', 'cve_desc')}
JB = b_table('data/b_keys/megavul_java_b_keys.jsonl.gz')
OUT = {'same_items': {
    'A two demonstrations': same(A, ids),
    'B retrieved chunks': same({k: r['b_key_sha256'] for k, r in JB.items()}, ids),
    'C, D candidate': same(FC, ids),
    'E CWE entries, code query': same(E['cwe_code'], ids),
    'E CWE entries, description query': same(E['cwe_desc'], ids),
    'E CVE descriptions': same(E['cve_desc'], ids)},
    'focus': {'PrimeVul': focus(pairs(PRIME), b_table('data/b_keys/primevul_b_keys.jsonl.gz')),
              'MegaVul Java': focus(JP, JB)}}
json.dump(OUT, open('analysis/fse_revision_20260923/java_rq1_retrieval.json', 'w'), indent=1)
print(json.dumps(OUT, indent=1))

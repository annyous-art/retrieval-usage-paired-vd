"""Match rates of Table 2 for the pair's own vs. another pair's items, and for the
vulnerable vs. the fixed member separately (offline; reads the JSON results of the RQ2 scripts).

Chunk configurations (focus_localization.py): the focus region contains a changed line, per function.
Other configurations (rq2_missing_rows.py, e_rq2_retrieval.py): a shown item carries the pair's NVD weakness.
All rates in those files are shares of 431 pairs (or 431 functions per member, 862 functions in all),
stored rounded to four decimals; each is converted back to its integer count, checked, and printed
as a percentage rounded once.

Run from the repository root: python3 analysis/fse_revision_20260923/rq2_match_by_member.py
"""
import json

A = 'analysis/fse_revision_20260923/'
N = 431


def count(rate, n):
    k = round(rate * n)
    if abs(k / n - rate) > 5e-5:
        raise ValueError(f'{rate} is not a count out of {n}')
    return k


def pct(rate, n=N):
    return round(100 * count(rate, n) / n, 1)


foc = json.load(open(A + 'focus_localization.json'))['A_B_localization']
mis = json.load(open(A + 'rq2_missing_rows.json'))['instances']
e = json.load(open(A + 'e_rq2_retrieval.json'))['variants']
rows = {}
for name, key in (('B positive-weighted', 'positive_weighted'), ('B balanced grouped BC', 'balanced_bc_grouped_no_label'),
                  ('C patch contrast', 'patch_contrast')):
    d = foc[key]
    rows[name] = {'kind': 'focus contains a changed line', 'all_functions': pct(d['function_top1_hit'][0], 2 * N),
                  'random_region_expected': round(100 * d['function_top1_random'][0], 1),  # an expectation, not a count
                  # per pair, as for the other configurations: either member's focus region contains a changed line
                  'either_member': pct(d['pair_either_top1_hit'][0]),
                  'either_member_random_expected': round(100 * d['pair_either_top1_random_independent'][0], 1),
                  'vulnerable': pct(d['vulnerable_top1_hit']), 'fixed': pct(d['fixed_top1_hit'])}
for name, key in (('A function-level top-2', 'A function-level top-2'), ('C/D fixed candidate', 'Fixed candidate'),
                  ('D Vul-RAG top-3 knowledge', 'Vul-RAG top-3 knowledge')):
    d = mis[key]['match_nvd']
    rows[name] = {'kind': 'shown item carries the NVD weakness', 'own_pair': pct(d['own_pair'][0]),
                  'other_pair': pct(d['other_pair_baseline'][0]), 'vulnerable': pct(d['vulnerable'][0]), 'fixed': pct(d['fixed'][0])}
for name, key in (('E CWE entries, code', 'cwe_code'), ('E CWE entries, description', 'cwe_desc'), ('E CVE descriptions', 'cve_desc')):
    d = e[key]['cwe_hit_nvd']
    rows[name] = {'kind': 'top-3 entry carries the NVD weakness', 'own_pair': pct(d['top3_either_member']['own_pair'][0]),
                  'other_pair': pct(d['top3_either_member']['other_pair_baseline'][0]),
                  'vulnerable': pct(d['top3_vulnerable_vs_fixed']['vulnerable'][0]), 'fixed': pct(d['top3_vulnerable_vs_fixed']['fixed'][0])}
json.dump({'pairs': N, 'rows': rows}, open(A + 'rq2_match_by_member.json', 'w'), indent=1)
for name, r in rows.items():
    main = f"{r['all_functions']}" if 'all_functions' in r else f"{r['own_pair']} vs. {r['other_pair']}"
    print(f"{name:30s} match {main:14s} vulnerable / fixed {r['vulnerable']} / {r['fixed']}")

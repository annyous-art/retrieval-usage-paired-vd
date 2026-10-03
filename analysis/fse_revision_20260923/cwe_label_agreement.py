"""Agreement of PrimeVul's CWE field with NVD's mapping of the same CVE (Section 4.1, Datasets).

For each of the 870 test functions: does PrimeVul's CWE field share a CWE with NVD's own assessment
(source nvd@nist.gov), and with all assessments that the current NVD record lists (NVD and CNA)?
NVD records: data/e_retrieval/nvd_cwe.jsonl (fetch_nvd_cwe.py). Offline; no API calls.

Run from the repository root: python3 analysis/fse_revision_20260923/cwe_label_agreement.py
"""
import ast
import json

NVD = {r['cve']: r['weaknesses'] for r in map(json.loads, open('data/e_retrieval/nvd_cwe.jsonl'))}
rows = [json.loads(l) for l in open('data/primevul_test_paired_labeled.jsonl')]


def cwes(ws, nist_only):
    return {c for w in ws if not nist_only or w['source'] == 'nvd@nist.gov' for c in w['cwes'] if c.startswith('CWE-')}


def own(r):
    try:
        x = ast.literal_eval(r['cwe']) if isinstance(r['cwe'], str) else r['cwe']
    except (ValueError, SyntaxError):
        x = [r['cwe']]
    return set(x)


OUT = {'functions': len(rows)}
for name, nist in (('nvd_own_assessment', True), ('all_assessments', False)):
    shared = sum(bool(own(r) & cwes(NVD.get(r.get('cve'), []), nist)) for r in rows)
    OUT[name] = {'shared': shared, 'rate': round(shared / len(rows), 4)}
json.dump(OUT, open('analysis/fse_revision_20260923/cwe_label_agreement.json', 'w'), indent=1)
print(OUT)

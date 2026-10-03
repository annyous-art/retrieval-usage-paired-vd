"""English NVD descriptions of the CVEs of the MegaVul Java split (corpus of mode E's CVE variant).
Uses fetch() of fetch_nvd_cwe.py; the API key is read from NVD_API_KEY and is not written anywhere.
Resumable. Run from the repository root:
  NVD_API_KEY=... python3 experiments/java_rq1_server/fetch_nvd_desc_java.py --out data/second_dataset/java_nvd_cve.jsonl
"""
import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fetch_nvd_cwe import fetch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default=str(ROOT / 'data/second_dataset/megavul_java_train_paired_labeled.jsonl'))
    ap.add_argument('--test', default=str(ROOT / 'data/second_dataset/megavul_java_test_paired_labeled.jsonl'))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    key = os.environ.get('NVD_API_KEY', '')
    cves = sorted({json.loads(l).get('cve') for p in (a.train, a.test) for l in open(p)} - {None, ''})
    done = {json.loads(l)['cve'] for l in open(a.out)} if os.path.exists(a.out) else set()
    todo = [c for c in cves if c not in done]
    print(f'{len(cves)} CVEs, {len(done)} cached, {len(todo)} to fetch (key set: {bool(key)})', flush=True)
    with open(a.out, 'a') as f:
        for i, c in enumerate(todo):
            d = fetch(c, key)
            rec = {'cve': c, 'fetched_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                   'found': bool(d and d.get('vulnerabilities')), 'description': '', 'weaknesses': []}
            if rec['found']:
                cv = d['vulnerabilities'][0]['cve']
                rec['description'] = next((x['value'] for x in cv.get('descriptions', []) if x.get('lang') == 'en'), '')
                for w in cv.get('weaknesses', []):
                    rec['weaknesses'].append({'source': w.get('source'), 'type': w.get('type'),
                                              'cwes': [x['value'] for x in w.get('description', []) if x.get('lang') == 'en']})
            f.write(json.dumps(rec) + '\n')
            f.flush()
            if (i + 1) % 100 == 0:
                print(f'{i + 1}/{len(todo)}', flush=True)
            time.sleep(0.7 if key else 6.5)
    print('done', flush=True)


if __name__ == '__main__':
    main()

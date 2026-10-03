"""Fetch the current NVD CWE mapping for every CVE of the PrimeVul paired train and test splits.

Used only to check labels in analyses (e.g. whether a retrieved CWE entry matches the pair's
weakness); never used to build retrieval queries, corpora or prompts. The API key is read from
the NVD_API_KEY environment variable and is not written anywhere. Resumable: CVEs already in
the output file are skipped.

Run from the repository root:
NVD_API_KEY=... python3 fetch_nvd_cwe.py --out data/e_retrieval/nvd_cwe.jsonl
"""
import argparse
import datetime
import http.client
import json
import os
import time
import urllib.error
import urllib.request

API = 'https://services.nvd.nist.gov/rest/json/cves/2.0?cveId='


def fetch(cve, key):
    req = urllib.request.Request(API + cve, headers={'apiKey': key} if key else {})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(6 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, http.client.HTTPException, ConnectionError, json.JSONDecodeError):
            time.sleep(6 * (attempt + 1))
    raise RuntimeError('NVD request failed repeatedly: ' + cve)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train', default='data/primevul_train_paired_labeled.jsonl')
    ap.add_argument('--test', default='data/primevul_test_paired_labeled.jsonl')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    key = os.environ.get('NVD_API_KEY', '')
    cves = sorted({json.loads(l).get('cve') for p in (a.test, a.train) for l in open(p)} - {None, ''})
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)['cve'] for l in open(a.out)}
    todo = [c for c in cves if c not in done]
    print(f'{len(cves)} CVEs, {len(done)} cached, {len(todo)} to fetch', flush=True)
    delay = 0.7 if key else 6.5
    with open(a.out, 'a') as f:
        for i, c in enumerate(todo):
            d = fetch(c, key)
            rec = {'cve': c, 'fetched_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                   'found': bool(d and d.get('vulnerabilities')), 'weaknesses': []}
            if rec['found']:
                for w in d['vulnerabilities'][0]['cve'].get('weaknesses', []):
                    rec['weaknesses'].append({'source': w.get('source'), 'type': w.get('type'),
                                              'cwes': [x['value'] for x in w.get('description', []) if x.get('lang') == 'en']})
            f.write(json.dumps(rec) + '\n')
            f.flush()
            if (i + 1) % 200 == 0:
                print(f'{i + 1}/{len(todo)}', flush=True)
            time.sleep(delay)
    print('done', flush=True)


if __name__ == '__main__':
    main()

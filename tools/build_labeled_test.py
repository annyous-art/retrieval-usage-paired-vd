"""Rebuild data/primevul_test_paired_labeled.jsonl from the official PrimeVul file and our labels.

The PrimeVul paired test split is not redistributed here. Download the paired test file
`primevul_test_paired.jsonl` released by the PrimeVul authors and place it in data/. This script
checks that file against the SHA-256 of the version we used, adds the diff-derived line labels
from data/primevul_test_paired_labels.jsonl (one list per function, 1 = line changed by the
patch), and checks that the result is byte-identical to the labeled file used in the paper,
whose hash the run scripts also verify.

Run from the repository root: python3 tools/build_labeled_test.py [path/to/primevul_test_paired.jsonl]
"""
import hashlib
import json
import sys
from pathlib import Path

OFFICIAL_SHA256 = '384758c25142b640f1768bb0d334c43f315c580022ae94973c66e77128fb3821'
LABELED_SHA256 = '878134e69d6c3d6eef57abe40ac62ac81bbe718e2f27a33fab21c4e7548bb4c8'
src = Path(sys.argv[1] if len(sys.argv) > 1 else 'data/primevul_test_paired.jsonl')
dest = Path('data/primevul_test_paired_labeled.jsonl')

raw = src.read_bytes()
if hashlib.sha256(raw).hexdigest() != OFFICIAL_SHA256:
    raise SystemExit(f'{src} does not match the PrimeVul paired test file used in the paper (SHA-256 {OFFICIAL_SHA256}).')
labels = [json.loads(l) for l in open('data/primevul_test_paired_labels.jsonl', encoding='utf-8')]
records = [json.loads(l) for l in raw.decode('utf-8').splitlines() if l]
if len(records) != len(labels):
    raise SystemExit('Row count differs from the label file.')
lines = []
for r, lab in zip(records, labels):
    if (r['idx'], r['func_hash']) != (lab['idx'], lab['func_hash']):
        raise SystemExit(f"Row mismatch at idx {r['idx']}.")
    lines.append(json.dumps({**r, 'labels': lab['labels']}, ensure_ascii=False))
out = ('\n'.join(lines) + '\n').encode('utf-8')
if hashlib.sha256(out).hexdigest() != LABELED_SHA256:
    raise SystemExit('Rebuilt file does not match the labeled file used in the paper.')
dest.write_bytes(out)
print(f'wrote {dest} ({len(lines)} functions, SHA-256 verified)')

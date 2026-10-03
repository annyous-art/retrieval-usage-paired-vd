"""Rebuild the MegaVul Java paired split (second dataset) from MegaVul's own release and our labels.

MegaVul's data is not redistributed here. Download the Java part of the MegaVul release 2024-04
(`megavul.json`, linked from https://github.com/Icyrockton/MegaVul) and pass its path. This script
  1. checks the file against the SHA-256 of the version we used,
  2. builds the paired split with data/second_dataset/build/build_megavul_java_pairs.py (chronological
     split at July 2022, deduplication; 1,564 training and 620 test pairs) and checks both files,
  3. replaces the line labels with ours from data/second_dataset/megavul_java_paired_labels_diff.jsonl.gz,
     computed as for PrimeVul by experiments/java_b_relabel/relabel_java_diff.py (diff -u line numbers
     and mark_pair_diffs.compute_labels), and checks the result against the files used in the paper.

Run from the repository root: python3 tools/build_megavul_java.py path/to/megavul.json
"""
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

MEGAVUL_SHA256 = '32bfb057d60c25146718e90e083c210cc826e0ee9fc43b1fdd9a1d5957d62a52'
SHA256 = {
    'megavul_java_train_paired_labeled.jsonl': 'f7327e4611705af01d7be5f5ca77a004e42242d425e00566046a3b2bf22a0b83',
    'megavul_java_test_paired_labeled.jsonl': '85407f3ebd421473c5a1d788363bfceb19265274004516f25728d61e61e7adac',
    'megavul_java_train_paired_labeled_diff.jsonl': '726bd9fe075c43d50c153fcc35cc3693500a5705bbeca6798b949bab5b9660dc',
    'megavul_java_test_paired_labeled_diff.jsonl': '04640485064e475ba090f6f0803970c4724423576e024980bac660684c3ab413',
}
D = Path('data/second_dataset')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

src = sys.argv[1] if len(sys.argv) > 1 else 'data/second_dataset/megavul_java/megavul.json'
if sha(src) != MEGAVUL_SHA256:
    raise SystemExit(f'{src} does not match the MegaVul Java release used in the paper (SHA-256 {MEGAVUL_SHA256}).')
stats = D / 'megavul_java_split_stats.json'
kept = stats.read_bytes()
subprocess.run([sys.executable, 'data/second_dataset/build/build_megavul_java_pairs.py', src, str(D)], check=True)
stats.write_bytes(kept)  # the build rewrites it with the same counts
labels = {(r['split'], r['idx'], r['func_hash']): r['labels'] for r in map(json.loads, gzip.open(D / 'megavul_java_paired_labels_diff.jsonl.gz', 'rt'))}
for split in ('train', 'test'):
    base = D / f'megavul_java_{split}_paired_labeled.jsonl'
    if sha(base) != SHA256[base.name]:
        raise SystemExit(f'{base} does not match the file used in the paper.')
    out = D / f'megavul_java_{split}_paired_labeled_diff.jsonl'
    with open(out, 'w', encoding='utf-8') as fh:
        for r in map(json.loads, open(base, encoding='utf-8')):
            r['labels'] = labels[(split, r['idx'], r['func_hash'])]
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    if sha(out) != SHA256[out.name]:
        raise SystemExit(f'{out} does not match the file used in the paper.')
    print(f'wrote {base} and {out} (SHA-256 verified)')

#!/usr/bin/env python3
"""Export the frozen requests of two mode-E arms (full and short CWE entries) for a replay through a second endpoint.

The first runs of Claude Opus 4.7 differ between dates: the short-entry run (29 Sep) reports hidden
output tokens on 726 of 870 calls and about 1.9 times the input tokens per prompt character of the
full-entry run (27 Sep, 29 calls with hidden tokens). To compare the two entry lengths under one
gateway, both arms are resent, unchanged, through replay_saved_prompts.py on the second endpoint (Anthropic route,
thinking field omitted). This script only writes one source file per arm: every job of
outdir/e_arm_<model>_<arm>/jobs.jsonl in its original order, with its messages as frozen.

Usage (repository root):
  python3 experiments/knowledge_rag/next_stage/e_arm_resent_sources.py --model claude-opus-4-7
  then, per arm, replay_saved_prompts.py --model claude-opus-4-7 --source <out>/source_<arm>.jsonl --out <out>/<arm>.jsonl
"""
import argparse, hashlib, json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='claude-opus-4-7')
    p.add_argument('--arms', nargs='+', default=['cwe_desc', 'cwe_desc_short'])
    p.add_argument('--out', type=Path, default=None)
    a = p.parse_args()
    out = a.out or Path(f'outdir/e_arm_{a.model}_resent')
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for arm in a.arms:
        src = Path(f'outdir/e_arm_{a.model}_{arm}/jobs.jsonl')
        rows = [json.loads(l) for l in open(src)]
        dst = out / f'source_{arm}.jsonl'
        text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
        if dst.exists() and dst.read_text() != text:
            raise SystemExit(f'{dst} exists with different content; use a new --out')
        dst.write_text(text)
        manifest[arm] = {'jobs': str(src), 'jobs_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
                         'rows': len(rows), 'source_sha256': hashlib.sha256(dst.read_bytes()).hexdigest()}
        print(arm, len(rows), 'rows ->', dst)
    (out / 'sources_manifest.json').write_text(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()

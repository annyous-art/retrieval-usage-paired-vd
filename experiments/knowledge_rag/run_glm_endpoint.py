#!/usr/bin/env python3
"""GLM-5.1 entrypoint of the evidence-slot experiment (modes C and D) on an OpenAI-compatible endpoint.

The endpoint is read from SLICERAG_UNIAPI_BASE_URL and the key from SLICERAG_UNIAPI_KEY (or from
experiments/knowledge_rag/glm_endpoint.key, which is never committed). k=1, std prompts, adjacent test
pairs, 4096 output tokens for extraction and detection, as in the paper's run (outdir/knowledge_rag_glm-5.1).
next_stage/*.py call configure() when a source run used this endpoint.

Usage (from experiments/knowledge_rag): python3 run_glm_endpoint.py {pilot|full} STAGE [WORKERS]
"""
import argparse
import os
from pathlib import Path

BASE_URL = os.environ.get('SLICERAG_UNIAPI_BASE_URL', '')


def configure():
    key_path = Path(__file__).with_name('glm_endpoint.key')
    key = os.environ.get('SLICERAG_UNIAPI_KEY') or (key_path.read_text().strip() if key_path.exists() else '')
    if not key:
        raise SystemExit('Set SLICERAG_UNIAPI_KEY or create experiments/knowledge_rag/glm_endpoint.key')
    if not BASE_URL:
        raise SystemExit('Set SLICERAG_UNIAPI_BASE_URL')
    os.environ['SLICERAG_BASE_URL'] = BASE_URL
    os.environ['SLICERAG_API_KEY'] = key
    return key


if __name__ == '__main__':
    import sys
    p = argparse.ArgumentParser()
    p.add_argument('size', choices=['pilot', 'full'])
    p.add_argument('stage', choices=['prepare', 'extract', 'assemble', 'detect', 'score', 'all'])
    p.add_argument('workers', type=int, nargs='?', default=2)
    a = p.parse_args()
    if a.workers < 1:
        p.error('workers must be positive')
    configure()
    root = Path(__file__).resolve().parents[2]
    out = root / 'outdir' / ('knowledge_rag_glm-5.1' if a.size == 'full' else 'knowledge_rag_glm-5.1_pilot')
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('This output directory is already running')
        sys.argv = [sys.argv[0], '--model', 'glm-5.1', '--stage', a.stage,
                    '--pairs', '0' if a.size == 'full' else '10', '--k', '1',
                    '--prompt-strategy', 'std', '--test-pair-policy', 'adjacent',
                    '--out', str(out), '--workers', str(a.workers),
                    '--extract-max-output', '4096', '--max-output', '4096',
                    '--read-timeout', os.environ.get('SLICERAG_READ_TIMEOUT', '600'),
                    '--max-attempts', os.environ.get('SLICERAG_MAX_ATTEMPTS', '3')]
        from run_server import main
        main()

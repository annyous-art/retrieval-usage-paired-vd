#!/usr/bin/env python3
"""UniAPI DeepSeek-V4-Pro-0813 entrypoint for the v7 knowledge experiment.

Same arguments as the glm-5.1 full run (experiments/knowledge_rag/run_glm_endpoint.py):
k=1, std prompts, adjacent test pairs, 4096 output tokens for extraction and detection,
thinking disabled. Runs from this isolated copy so the frozen source hashes of the
original experiment directory stay unchanged. The key is read from SLICERAG_UNIAPI_KEY or
from experiments/knowledge_rag/glm_endpoint.key (same UniAPI account); it is never copied.
"""
import argparse
import os
import sys
from pathlib import Path

BASE_URL = os.environ.get('SLICERAG_UNIAPI_BASE_URL', '')  # OpenAI-compatible endpoint, set in the environment
MODEL = 'deepseek-v4-pro-0813'
HERE = Path(__file__).resolve().parent


def configure():
    key_path = HERE.parent / 'knowledge_rag' / 'glm_endpoint.key'
    key = os.environ.get('SLICERAG_UNIAPI_KEY') or (key_path.read_text().strip() if key_path.exists() else '')
    if not key:
        raise SystemExit('Set SLICERAG_UNIAPI_KEY or keep experiments/knowledge_rag/glm_endpoint.key')
    if not BASE_URL:
        raise SystemExit('Set SLICERAG_UNIAPI_BASE_URL')
    os.environ['SLICERAG_BASE_URL'] = BASE_URL
    os.environ['SLICERAG_API_KEY'] = key
    return key


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('size', choices=['pilot', 'full'])
    p.add_argument('stage', choices=['prepare', 'extract', 'assemble', 'detect', 'score', 'all'])
    p.add_argument('workers', type=int, nargs='?', default=2)
    a = p.parse_args()
    if a.workers < 1:
        p.error('workers must be positive')
    configure()
    sys.path.insert(0, str(HERE))
    out = HERE.parents[1] / 'outdir' / f'knowledge_rag_{MODEL}_{a.size}_uniapi_v7'
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('This output directory is already running')
        sys.argv = [sys.argv[0], '--model', MODEL, '--stage', a.stage,
                    '--pairs', '0' if a.size == 'full' else '10', '--k', '1',
                    '--prompt-strategy', 'std', '--test-pair-policy', 'adjacent',
                    '--out', str(out), '--workers', str(a.workers),
                    '--extract-max-output', '4096', '--max-output', '4096',
                    '--read-timeout', os.environ.get('SLICERAG_READ_TIMEOUT', '600'),
                    '--max-attempts', os.environ.get('SLICERAG_MAX_ATTEMPTS', '3')]
        from run_server import main
        main()

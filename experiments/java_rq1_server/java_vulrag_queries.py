"""Vul-RAG query descriptions for the MegaVul Java test functions (needed for mode E with
description queries), generated exactly as the PrimeVul descriptions: the query stage of
experiments/vulrag_full/run.py, i.e. Vul-RAG's purpose and function prompts
(generate_extraction_prompt_for_vulrag), GPT-5.5 through the endpoint of the PrimeVul descriptions, temperature 0.01, cached and
resumable per call.

Run from the repository root:
  python3 experiments/java_rq1_server/java_vulrag_queries.py --out outdir/java_vulrag_queries_gpt-5.5 --workers 4
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/vulrag_full'))
import run as vr  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--test', type=Path, default=ROOT / 'data/second_dataset/megavul_java_test_paired_labeled.jsonl')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', default='gpt-5.5')
    p.add_argument('--provider', default='newapi')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--max-output', type=int, default=16384)
    p.add_argument('--temperature', type=float, default=.01)
    p.add_argument('--read-timeout', type=int, default=900)
    p.add_argument('--max-attempts', type=int, default=3)
    p.add_argument('--max-calls', type=int, default=25000)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    targets = [{'row_id': str(r['idx']), 'func': r['func']} for r in map(json.loads, open(a.test))]
    manifest = {'protocol': vr.protocol(a), 'dataset': str(a.test), 'targets': len(targets)}
    mp = a.out / 'manifest.json'
    if mp.exists() and vr.load(mp)['protocol'] != manifest['protocol']:
        raise SystemExit('Protocol changed: use a new output directory')
    vr.atomic(mp, manifest)
    calls = vr.Calls(a, manifest)

    def work(t):
        out = a.out / 'query_items' / f"{t['row_id']}.json"
        if out.exists():
            return
        pp, fp = vr.generate_extraction_prompt_for_vulrag(t['func'])
        base = f"query/{t['row_id']}"
        purpose = calls.call(base + '/purpose', vr.message(pp), lambda x: vr.prefix(x, 'Function purpose:'))['content']
        function = calls.call(base + '/function', vr.message(fp), lambda x: vr.prefix(x, 'The functions of the code snippet are:'))['content']
        vr.atomic(out, {'row_id': t['row_id'], 'purpose': vr.prefix(purpose, 'Function purpose:'),
                        'function': vr.prefix(function, 'The functions of the code snippet are:')})

    vr.parallel(a, targets, work, 'query')
    vr.atomic(a.out / 'queries.json', [vr.load(a.out / 'query_items' / f"{t['row_id']}.json") for t in targets])
    print('Wrote', a.out / 'queries.json')


if __name__ == '__main__':
    main()

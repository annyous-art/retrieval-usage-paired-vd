#!/usr/bin/env python3
"""Vul-RAG detection with another pair's knowledge (negative control for the detection stage).

Takes a prepared Vul-RAG directory whose retrieval is complete (here the GLM-5.1 run that imports
GPT-5.5's knowledge base and query descriptions) and changes one thing: each target function gets
the knowledge items retrieved for the member with the same label of ANOTHER test pair, under a fixed
derangement of the pairs (seed 20260915; never a pair with the same CVE or the same commit). Prompts,
model, decoding, the two questions per item (cause, solution), early return, at most three items and
the default NO are run.py's own. The members of a pair therefore still receive different items when
the source pair's members did, but the items no longer come from retrieval for this target.

Stages:
  prepare  copy the frozen inputs, write the swapped retrieved.json, the offline context audit, the
           retrieval manifest that run.detect checks, and swap_config.json (mapping, hashes)
  detect   run.detect with the prepared directory's protocol (resumable; --max-calls for a pilot)
  score    run.score

Usage (repository root, Vul-RAG environment):
  python experiments/vulrag_full/detect_swapped_knowledge.py --source outdir/vulrag_glm-5.1_gpt55_knowledge \
      --out outdir/vulrag_glm-5.1_other_pair_knowledge --stage prepare
  ... --stage detect --workers 12 [--max-calls 12]
"""
import argparse, fcntl, hashlib, json, random, shutil, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402
from prompts import generate_detect_vul_prompt_with_response_in_HTML, generate_detect_sol_prompt_with_response_in_HTML  # noqa: E402

SEED = 20260915
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
FROZEN = ('manifest.json', 'targets.json', 'training.json', 'knowledge.json', 'queries.json', 'imported_from.json')


def derangement(targets):
    raw = {str(r['idx']): r for r in map(json.loads, open(HERE.parents[1] / 'data/primevul_test_paired_labeled.jsonl'))}
    pairs = {}
    for t in targets:
        pairs.setdefault(t['pair_id'], t)
    info = {p: (raw[str(t['idx'])].get('cve'), t['cluster']) for p, t in pairs.items()}
    order = sorted(pairs, key=lambda x: int(x[1:])); random.Random(SEED).shuffle(order)
    n = len(order)
    shift = next(k for k in range(1, n) if all(
        info[order[i]][0] != info[order[(i + k) % n]][0] and info[order[i]][1] != info[order[(i + k) % n]][1] for i in range(n)))
    return {order[i]: order[(i + shift) % n] for i in range(n)}, shift


def prepare(a, args):
    a.out.mkdir(parents=True, exist_ok=True)
    for name in FROZEN:
        dst = a.out / name
        if dst.exists() and sha(dst) != sha(a.source / name):
            raise SystemExit(f'{dst} differs from the source; use a new --out')
        shutil.copyfile(a.source / name, dst)
    targets = run.load(a.out / 'targets.json')
    src_rows = {r['row_id']: r for r in run.load(a.source / 'retrieved.json')}
    by_pair = {(t['pair_id'], t['target']): t['row_id'] for t in targets}
    mapping, shift = derangement(targets)
    rows, requests = [], []
    for t in targets:
        src_row = by_pair[(mapping[t['pair_id']], t['target'])]
        s = src_rows[src_row]
        rows.append({'row_id': t['row_id'], 'knowledge_ids': s['knowledge_ids'], 'evidence': s['evidence'],
                     'trace': {'swapped_from_row': src_row, 'swapped_from_pair': mapping[t['pair_id']]}})
        for e in s['evidence']:
            for fn in (generate_detect_vul_prompt_with_response_in_HTML, generate_detect_sol_prompt_with_response_in_HTML):
                requests.append({'row_id': t['row_id'], 'messages': run.message(fn(t['func'], e))})
    run.audit_requests(requests, args.model, args.max_output, a.out / 'detection_context_audit.json', temperature=args.temperature)
    retrieved = a.out / 'retrieved.json'
    if retrieved.exists() and run.load(retrieved) != rows:
        raise SystemExit('Swapped retrieval changed; use a new --out')
    run.atomic(retrieved, rows)
    run.atomic(a.out / 'retrieval_manifest.json', {n + '_sha256': sha(a.out / (n + '.json')) for n in ('knowledge', 'queries', 'retrieved', 'targets')})
    cfg = {'source': str(a.source), 'source_retrieved_sha256': sha(a.source / 'retrieved.json'), 'seed': SEED, 'shift': shift,
           'mapping': mapping, 'retrieved_sha256': sha(retrieved), 'script_sha256': sha(__file__)}
    cp = a.out / 'swap_config.json'
    if cp.exists() and run.load(cp) != cfg:
        raise SystemExit('swap_config differs (script or inputs changed); use a new --out')
    run.atomic(cp, cfg)
    audit = run.load(a.out / 'detection_context_audit.json')
    print(f'Prepared {len(rows)} targets, shift {shift}; detection requests {audit["request_count"]}, '
          f'needs_capacity_review {audit["needs_capacity_review"]}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--stage', choices=['prepare', 'detect', 'score'], required=True)
    p.add_argument('--model', default='glm-5.1'); p.add_argument('--provider', default='uniapi')
    p.add_argument('--workers', type=int, default=4); p.add_argument('--max-calls', type=int, default=25000)
    a = p.parse_args()
    args = argparse.Namespace(model=a.model, provider=a.provider, out=a.out, workers=a.workers, max_output=16384,
                              temperature=0.01, read_timeout=900, max_attempts=3, max_calls=a.max_calls)
    manifest = run.load(a.source / 'manifest.json')
    if manifest['protocol'] != run.protocol(args):
        raise SystemExit('Prepared protocol differs from this invocation; no calls made')
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.stage == 'prepare':
            prepare(a, args)
        elif a.stage == 'detect':
            cfg = run.load(a.out / 'swap_config.json')
            if cfg['retrieved_sha256'] != sha(a.out / 'retrieved.json'):
                raise SystemExit('Swapped retrieval changed; no calls made')
            run.detect(args, run.load(a.out / 'manifest.json'))
        else:
            run.score(args)


if __name__ == '__main__':
    main()

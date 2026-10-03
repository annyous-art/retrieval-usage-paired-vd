#!/usr/bin/env python3
"""Oracle and wrong-evidence arms in the fixed-candidate (FC) setting (reviewer M1).

Each arm fills the FC evidence slot of the original no-retrieval prompt (same instruction, same
target) with different evidence; nothing else in the prompt changes:
  oracle_code_pair   the target pair's OWN vulnerable and fixed function, in the code_pair format
                     (an upper bound: "retrieval" that returns exactly the target's fix; the evidence
                     contains the target itself)
  oracle_knowledge   knowledge extracted, with the original extraction prompt and model, from the
                     target pair's OWN fix, in the knowledge format (functional semantics, root
                     cause, repair condition)
  wrong_code_pair    the vulnerable and fixed function of ANOTHER test pair (fixed derangement,
                     seed 20260915; never a pair with the same CVE or the same commit), in the
                     code_pair format
Routing, client, model, decoding, and output limits are taken from the original run and checked
before any call (as in extra_arm.py); the returned model name must equal the requested one.
Resumable: re-running a stage skips completed calls.

Usage (from experiments/knowledge_rag):
  python3 next_stage/oracle_arm.py --source ../../outdir/knowledge_rag_gpt-5.5 \
      --out ../../outdir/oracle_gpt-5.5 --arm oracle_code_pair --stage run
  --arm oracle_knowledge runs the extraction first (stage extract), then detection.
"""
import argparse, fcntl, hashlib, json, os, random, sys
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiment as exp

MARKER = 'Historical evidence (may be irrelevant; check the target independently):\n(none)\n\n'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_once(p, rows):
    if p.exists():
        if exp.read(p) != rows:
            raise ValueError('Frozen jobs changed; use a new output directory')
    else:
        exp.write(p, rows)


def client_and_check(cfg):
    if os.environ.get('SLICERAG_UNIAPI_HOST', '\0') in cfg['base_url']:  # host of the OpenAI-compatible endpoint used for GLM/DeepSeek
        from run_glm_endpoint import configure
        configure()
    from concurrent_runner import inference_config
    api = exp.load_client()
    args = NS(model=cfg['model'], temperature=cfg['temperature'], max_output=cfg['max_tokens'],
              prompt_strategy=cfg['prompt_strategy'], seed=cfg['seed'])
    if inference_config(api, args) != cfg:
        raise ValueError('Gateway/client/config differs from the original run; no calls made')
    loader = exp.load_client

    def checked_client():
        c = loader(); call = c.get_openai_chat

        def checked(*a, **k):
            result = call(*a, **k)
            if c.last_response_metadata.get('returned_model') != cfg['model']:
                raise ValueError('Returned model differs from requested model')
            return result
        c.get_openai_chat = checked
        return c
    exp.load_client = checked_client
    return args


def run_jobs(cfg, jobs_path, output, workers, max_calls):
    from concurrent_runner import run
    args = client_and_check(cfg)
    run(NS(**vars(args), jobs=jobs_path, output=output, workers=workers, read_timeout=600,
           max_attempts=3, max_calls=max_calls))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arm', choices=['oracle_code_pair', 'oracle_knowledge', 'wrong_code_pair'], required=True)
    p.add_argument('--stage', choices=['prepare', 'run'], default='prepare')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--max-calls', type=int, default=100000)
    p.add_argument('--pairs', type=int, default=0, help='pilot: only the first N test pairs')
    a = p.parse_args()
    out = a.out / a.arm
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(a, out)


def execute(a, out):
    src = a.source
    original = exp.read(src / 'predictions.jsonl')
    cfg = original[0]['config']
    if len({exp.digest(r['config']) for r in original}) != 1:
        raise ValueError('Mixed original configs')
    no = {r['row_id']: r for r in exp.read(src / 'detection_jobs.jsonl') if r['arm'] == 'no_retrieval'}
    targets = exp.read(src / 'targets.jsonl')
    by_pair = {}
    for t in targets:
        by_pair.setdefault(t['pair_id'], {})[int(t['target'])] = t
    pair_ids = sorted(by_pair, key=lambda x: int(x[1:]))
    if a.pairs:
        pair_ids = pair_ids[:a.pairs]
    keep = set(pair_ids)
    raw = {str(r['idx']): r for r in exp.read(Path(__file__).resolve().parents[3] / 'data/primevul_test_paired_labeled.jsonl')}
    members = {pid: {'members': [dict(by_pair[pid][1]), dict(by_pair[pid][0])]} for pid in by_pair}   # [vulnerable, fixed]

    frozen = {'arm': a.arm, 'config': cfg, 'pairs': len(pair_ids), 'seed': 20260915,
              'files': {n: sha(src / n) for n in ['targets.jsonl', 'detection_jobs.jsonl']},
              'script_sha256': sha(__file__)}
    context = {}
    if a.arm in ('oracle_code_pair', 'wrong_code_pair'):
        if a.arm == 'oracle_code_pair':
            source_of = {pid: pid for pid in by_pair}
        else:
            order = sorted(by_pair, key=lambda x: int(x[1:])); random.Random(20260915).shuffle(order)
            info = {pid: (raw[str(by_pair[pid][1]['idx'])].get('cve'), raw[str(by_pair[pid][1]['idx'])].get('commit_id')) for pid in order}
            shift = next(k for k in range(1, len(order)) if all(
                info[order[i]][0] != info[order[(i + k) % len(order)]][0] and info[order[i]][1] != info[order[(i + k) % len(order)]][1]
                for i in range(len(order))))
            source_of = {order[i]: order[(i + shift) % len(order)] for i in range(len(order))}
            frozen['derangement_shift'] = shift
        for pid in pair_ids:
            v, f = members[source_of[pid]]['members']
            context[pid] = ('Historical vulnerable code:\n' + v['func'] + '\nHistorical repaired code:\n' + f['func'], source_of[pid])
    else:
        ejobs = [{'candidate_id': pid, 'messages': exp.extraction_messages(members[pid])} for pid in pair_ids]
        ejobs = [dict(j, job_id=exp.digest([j['candidate_id'], j['messages']])) for j in ejobs]
        write_once(out / 'extraction_jobs.jsonl', ejobs)
        kpath = out / 'knowledge_responses.jsonl'
        done = {r['job_id'] for r in exp.read(kpath)} if kpath.exists() else set()
        if a.stage == 'run' and len(done) < len(ejobs):
            run_jobs(cfg, out / 'extraction_jobs.jsonl', kpath, a.workers, a.max_calls)
        if not kpath.exists() or len({r['job_id'] for r in exp.read(kpath)}) < len(ejobs):
            print('Extraction incomplete; run --stage run again.'); return
        ej = {j['job_id']: j['candidate_id'] for j in ejobs}
        know = {}
        for r in exp.read(kpath):
            know[ej[r['job_id']]] = exp.parse_knowledge(r['response'])
        for pid in pair_ids:
            k = know[pid]
            context[pid] = ('Functional semantics: ' + k['functional_semantics'] + '\nRoot cause: ' + k['root_cause'] +
                            '\nRepair condition: ' + k['fix_condition'], pid)

    fp = out / 'config.json'
    if fp.exists() and json.loads(fp.read_text()) != frozen:
        raise ValueError('Configuration changed; use a new output directory')
    fp.write_text(json.dumps(frozen, indent=2))
    jobs = []
    for pid in pair_ids:
        for tgt in (1, 0):
            t = by_pair[pid][tgt]
            msgs = [dict(m) for m in no[t['row_id']]['messages']]
            if not msgs[1]['content'].startswith(MARKER):
                raise ValueError('Unknown base prompt')
            ctx, from_pair = context[pid]
            msgs[1]['content'] = MARKER.replace('(none)', ctx) + msgs[1]['content'][len(MARKER):]
            j = {k: v for k, v in t.items() if k != 'func'}
            j.update(arm=a.arm, evidence_pair=from_pair, messages=msgs, evidence_bytes=len(ctx.encode()),
                     job_id=exp.digest([t['row_id'], a.arm, msgs]))
            jobs.append(j)
    write_once(out / 'jobs.jsonl', jobs)
    print(f'Prepared {len(jobs)} {a.arm} detection requests ({len(pair_ids)} pairs).', flush=True)
    if a.stage == 'prepare':
        return
    output = out / 'predictions.jsonl'
    run_jobs(cfg, out / 'jobs.jsonl', output, a.workers, a.max_calls)
    rs = exp.read(output)
    if len(rs) < len(jobs):
        print('Partial run saved; repeat to continue.'); return
    states = {r['job_id']: exp.prediction(r['response']) for r in rs}
    pairs = {}
    for j in jobs:
        pairs.setdefault(j['pair_id'], {})[j['target']] = states[j['job_id']]
    cnt = {'P-C': 0, 'P-V': 0, 'P-B': 0, 'P-R': 0, 'unknown': 0}
    for v in pairs.values():
        key = {(1, 0): 'P-C', (1, 1): 'P-V', (0, 0): 'P-B', (0, 1): 'P-R'}.get((v.get(1), v.get(0)), 'unknown')
        cnt[key] += 1
    tokens = sum((r.get('usage') or {}).get('input_tokens', (r.get('usage') or {}).get('prompt_tokens', 0)) or 0 for r in rs)
    out_tokens = sum((r.get('usage') or {}).get('output_tokens', (r.get('usage') or {}).get('completion_tokens', 0)) or 0 for r in rs)
    (out / 'arm_metrics.json').write_text(json.dumps({'arm': a.arm, 'pairs': len(pairs), 'counts': cnt,
        'rates': {k: round(v / len(pairs), 4) for k, v in cnt.items()}, 'input_tokens': tokens, 'output_tokens': out_tokens}, indent=2))
    print(json.dumps(cnt), 'input tokens', tokens, 'output tokens', out_tokens)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Presentation arms for RQ3: the same retrieved items in reversed order (content unchanged).

Arms (each rebuilt from the saved prompts of an existing run; only the order changes):
  c_fixed_first    C, evidence slot: "Historical repaired code" before "Historical vulnerable code"
  d_fields_rev     D, evidence slot: Repair condition, Root cause, Functional semantics (reversed)
  e_cwe_desc_rev   E, CWE entries retrieved with the description query, top-3 in reversed order
  a_demos_swapped  A, the two retrieved demonstrations (user + assistant turns) swapped; written
                   as a prompt file for replay_saved_prompts.py (the client of the original A runs)
For every prompt the original order is rebuilt from the run's own materials and checked against the
saved prompt before the reordered prompt is written, so the arm differs only in the order.
C/D/E use the routing, client, decoding and output limits of the source run, checked before any
call (as in oracle_arm.py). Resumable.

Usage (from experiments/knowledge_rag):
  python3 next_stage/order_arm.py --arm c_fixed_first --source ../../outdir/knowledge_rag_glm-5.1 \
      --out ../../outdir/order_glm-5.1 --stage prepare|run
  python3 next_stage/order_arm.py --arm e_cwe_desc_rev --source ../../outdir/e_arm_glm-5.1_cwe_desc \
      --out ../../outdir/order_glm-5.1 --stage prepare|run
  python3 next_stage/order_arm.py --arm a_demos_swapped --source ../../outdir/icl_fixed_baseline_compatible_test870/glm-5.1_std_cls_fewshotegTrue_top2_baseline_compatible.jsonl \
      --out ../../outdir/order_glm-5.1
"""
import argparse
import fcntl
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiment as exp  # noqa: E402
from oracle_arm import run_jobs, sha, write_once  # noqa: E402

H = 'Historical evidence (may be irrelevant; check the target independently):\n'
ROOT = Path(__file__).resolve().parents[3]


def split_slot(content):
    """(evidence, rest) of an evidence-slot user message."""
    if not content.startswith(H):
        raise ValueError('Unknown base prompt')
    body = content[len(H):]
    i = body.index('\n\nPlease analyze the following code:')
    return body[:i], body[i:]


def reorder_c(ev, row, cand):
    v, f = cand[exp_list(row['candidate_ids'])[0]]['members']
    if int(v['target']) != 1:
        v, f = f, v
    original = 'Historical vulnerable code:\n' + v['func'] + '\nHistorical repaired code:\n' + f['func']
    if ev != original:
        raise ValueError('Code-pair evidence does not rebuild for row ' + str(row['row_id']))
    return 'Historical repaired code:\n' + f['func'] + '\nHistorical vulnerable code:\n' + v['func']


FIELDS = ('Functional semantics: ', 'Root cause: ', 'Repair condition: ')


def reorder_d(ev, row, cand):
    parts, rest = [], ev
    for k, name in enumerate(FIELDS):
        if not rest.startswith(name):
            raise ValueError('Knowledge evidence does not parse for row ' + str(row['row_id']))
        nxt = rest.find('\n' + FIELDS[k + 1]) if k + 1 < len(FIELDS) else len(rest)
        parts.append(rest[:nxt])
        rest = rest[nxt + 1:]
    if '\n'.join(parts) != ev:
        raise ValueError('Knowledge evidence does not rebuild for row ' + str(row['row_id']))
    return '\n'.join(reversed(parts))


def exp_list(x):
    import ast
    return ast.literal_eval(x) if isinstance(x, str) else x


def slot_arm(a, out):
    src = a.source
    if a.arm == 'e_cwe_desc_rev':
        rows = exp.read(src / 'jobs.jsonl')
        cfg = json.loads((src / 'config.json').read_text())['config']
        entries = {e['id']: e['prompt_full'] for e in exp.read(ROOT / 'data/e_retrieval/cwe_entries.jsonl')}
    else:
        arm = {'c_fixed_first': 'code_pair', 'd_fields_rev': 'knowledge'}[a.arm]
        preds = exp.read(src / 'predictions.jsonl')
        cfg = preds[0]['config']
        if len({exp.digest(r['config']) for r in preds}) != 1:
            raise ValueError('Mixed source configs')
        rows = [r for r in exp.read(src / 'detection_jobs.jsonl') if r['arm'] == arm]
        cand = {c['pair_id']: c for c in exp.read(src / 'candidates.jsonl')}
    if a.pairs:
        keep = sorted({r['pair_id'] for r in rows}, key=lambda x: int(x[1:]))[:a.pairs]
        rows = [r for r in rows if r['pair_id'] in keep]
    jobs = []
    for r in rows:
        msgs = [dict(m) for m in r['messages']]
        ev, rest = split_slot(msgs[1]['content'])
        if a.arm == 'e_cwe_desc_rev':
            ids = exp_list(r['candidate_ids'])
            if ev != '\n\n'.join(entries[i] for i in ids):
                raise ValueError('CWE evidence does not rebuild for row ' + str(r['row_id']))
            new = '\n\n'.join(entries[i] for i in reversed(ids))
        elif a.arm == 'c_fixed_first':
            new = reorder_c(ev, r, cand)
        else:
            new = reorder_d(ev, r, cand)
        if sorted(new) != sorted(ev):
            raise ValueError('Reordering changed the content of row ' + str(r['row_id']))
        msgs[1]['content'] = H + new + rest
        j = {k: v for k, v in r.items() if k not in ('messages', 'job_id', 'config', 'response', 'usage')}
        j.update(arm=a.arm, messages=msgs, job_id=exp.digest([r['row_id'], a.arm, msgs]))
        jobs.append(j)
    frozen = {'arm': a.arm, 'config': cfg, 'source': str(src), 'rows': len(jobs), 'script_sha256': sha(__file__)}
    fp = out / 'config.json'
    if fp.exists() and json.loads(fp.read_text()) != frozen:
        raise ValueError('Configuration changed; use a new output directory')
    fp.write_text(json.dumps(frozen, indent=2))
    write_once(out / 'jobs.jsonl', jobs)
    tokens = sum(len(m['content']) for j in jobs for m in j['messages'])
    print(f'Prepared {len(jobs)} {a.arm} requests ({tokens / 1e6:.2f}M characters).', flush=True)
    if a.stage == 'run':
        run_jobs(cfg, out / 'jobs.jsonl', out / 'predictions.jsonl', a.workers, a.max_calls)
        done = exp.read(out / 'predictions.jsonl')
        print(f'{len(done)}/{len(jobs)} answered; repeat the command to continue.' if len(done) < len(jobs) else 'Complete.')


def demo_arm(a, out):
    rows = [json.loads(l) for l in open(a.source)]
    if a.pairs:
        rows = rows[:2 * a.pairs]
    out_rows = []
    for r in rows:
        m = r['messages']
        if [x['role'] for x in m] != ['system', 'user', 'assistant', 'user', 'assistant', 'user']:
            raise ValueError('Expected two demonstrations for idx ' + str(r['idx']))
        sims = [s for s in r['similar'][:2]]
        for k, s in enumerate(sims):
            if not m[1 + 2 * k]['content'].startswith('Please analyze the following code:\n```\n' + s['func']):
                raise ValueError('Demonstration does not match similar[%d] for idx %s' % (k, r['idx']))
        new = dict(r, messages=[m[0], m[3], m[4], m[1], m[2], m[5]], presentation='a_demos_swapped')
        new['similar'] = [sims[1], sims[0]] + r['similar'][2:]
        out_rows.append(new)
    path = out / 'prompts_a_demos_swapped.jsonl'
    exp.write(path, out_rows) if not path.exists() else write_once(path, out_rows)
    print(f'Wrote {len(out_rows)} swapped-demonstration prompts to {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arm', required=True, choices=['c_fixed_first', 'd_fields_rev', 'e_cwe_desc_rev', 'a_demos_swapped'])
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--stage', choices=['prepare', 'run'], default='prepare')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--max-calls', type=int, default=100000)
    p.add_argument('--pairs', type=int, default=0, help='pilot: only the first N pairs')
    a = p.parse_args()
    out = a.out / a.arm
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (demo_arm if a.arm == 'a_demos_swapped' else slot_arm)(a, out)


if __name__ == '__main__':
    main()

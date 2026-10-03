#!/usr/bin/env python3
"""Send the exact saved prompts of an existing std_cls run to another model (resumable).

Uses model_api_clients.get_openai_chat, the client of every earlier std_cls run
(run_prompting_sliced_rag.py, icl/run_prompting_k_icl.py), with their defaults:
std_cls, temperature 0, 1024 output tokens, seed 12345. Every source row's `messages`
are sent as saved; all other source fields are kept except the previous model's
response, usage and reasoning. Rows already answered in --out are skipped; failed rows are
recorded in <out>.errors.jsonl and retried on the next invocation.

Example (B, BC grouped no-label, prompts identical for GPT-5.5 and GLM-5.1):
python3 replay_saved_prompts.py --model claude-opus-4-7 \
  --source outdir/prompt_gpt55_representative_ablation_test870/gpt-5.5_std_cls_grouped_no_label_baseline_compatible_fewshotegTrue_cls0.jsonl \
  --out outdir/replay_b_bc_grouped_no_label/claude-opus-4-7_std_cls_grouped_no_label_baseline_compatible_fewshotegTrue_cls0.jsonl
"""
import argparse
import concurrent.futures
import hashlib
import json
import threading
from pathlib import Path

import model_api_clients
from model_api_clients import get_openai_chat, normalize_usage

MODELS = ['claude-opus-4-7', 'gemini-3.1-pro-preview', 'glm-5.1', 'gpt-5.5']


def client_endpoint():
    """Base URL the client will call (no credentials), for the log and the frozen config."""
    for name in ('OPENAI_BASE_URL', 'ANTHROPIC_BASE_URL'):
        if hasattr(model_api_clients, name):
            return {'openai': getattr(model_api_clients, 'OPENAI_BASE_URL', None),
                    'anthropic': getattr(model_api_clients, 'ANTHROPIC_BASE_URL', None)}
    c = getattr(model_api_clients, 'client', None)
    a = getattr(model_api_clients, 'claude_client', None)
    return {'openai': str(getattr(c, 'base_url', '')) or None, 'anthropic': str(getattr(a, 'base_url', '')) or None}

DROP = {'response', 'usage', 'reasoning', 'prompt_tokens', 'completion_tokens', 'reasoning_tokens', 'total_tokens'}
# The client returns these strings instead of raising; they are failures, retried on the next run.
FAILED = {None, '', 'ERROR', '[ERROR]', 'EMPTY', '[EMPTY]'}


def failed(response):
    return response is None or (isinstance(response, str) and response.strip() in FAILED)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, choices=MODELS)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--max-calls', type=int, default=0)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(a.source)]
    frozen = {'model': a.model, 'source': str(a.source), 'source_sha256': hashlib.sha256(a.source.read_bytes()).hexdigest(),
              'rows': len(rows), 'prompt_strategy': 'std_cls', 'temperature': 0.0, 'max_gen_length': 1024, 'seed': 12345,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'client_file': model_api_clients.__file__,
              'client_sha256': hashlib.sha256(Path(model_api_clients.__file__).read_bytes()).hexdigest(),
              'client_endpoint': client_endpoint()}
    print('client:', frozen['client_file'], frozen['client_endpoint'], flush=True)
    meta = a.out.with_suffix('.config.json')
    if meta.exists() and json.loads(meta.read_text()) != frozen:
        raise SystemExit('Source, model, script or client changed; use a new output file')
    meta.write_text(json.dumps(frozen, indent=1))
    done = set()
    if a.out.exists():
        done = {r['source_line'] for r in map(json.loads, open(a.out)) if not failed(r.get('response'))}
    todo = [i for i in range(len(rows)) if i not in done]
    if a.max_calls:
        todo = todo[:a.max_calls]
    print(f'{len(rows)} rows, {len(done)} done, {len(todo)} to call', flush=True)
    lock = threading.Lock()

    def call(i):
        response, usage, messages, reasoning = get_openai_chat({'messages': rows[i]['messages']}, a.model, 'std_cls', 0.0, 1024, 12345)
        if failed(response):
            raise RuntimeError('client returned ' + repr(response))
        rec = {k: v for k, v in rows[i].items() if k not in DROP}
        rec.update(source_line=i, model=a.model, messages=messages, response=response,
                   usage=normalize_usage(usage, a.model), reasoning=reasoning,
                   prompt_identical_to_source=messages == rows[i]['messages'])
        return rec

    ok = err = 0
    with open(a.out, 'a') as out, open(str(a.out) + '.errors.jsonl', 'a') as errs:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
            futs = {ex.submit(call, i): i for i in todo}
            for f in concurrent.futures.as_completed(futs):
                i = futs[f]
                try:
                    rec = f.result()
                    with lock:
                        out.write(json.dumps(rec, ensure_ascii=False) + '\n'); out.flush()
                    ok += 1
                except Exception as e:  # noqa: BLE001 - recorded and retried on the next run
                    with lock:
                        errs.write(json.dumps({'source_line': i, 'error': type(e).__name__, 'message': str(e)[:500]}) + '\n'); errs.flush()
                    err += 1
                if (ok + err) % 50 == 0:
                    print(f'{ok} ok, {err} failed', flush=True)
    answered = {r['source_line'] for r in map(json.loads, open(a.out)) if not failed(r.get('response'))}
    print(f'finished this pass: {ok} ok, {err} failed; {len(answered)}/{len(rows)} rows answered', flush=True)


if __name__ == '__main__':
    main()

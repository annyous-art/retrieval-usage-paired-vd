#!/usr/bin/env python3
"""Replay the exact saved prompts of an existing std_cls run on deepseek-v4-pro-0813.

Used for the two usage modes that were not run through the knowledge framework:
  A (function-level retrieved demonstrations, baseline-compatible; the two training functions with
    the highest SFR-Embedding-Code-400M_R cosine similarity, as in Hannan et al.): source =
    outdir/icl_fixed_baseline_compatible_test870/glm-5.1_std_cls_fewshotegTrue_top2_baseline_compatible.jsonl
    (identical to the author two-shot prompt except for the two retrieved demonstrations)
  Fixed few-shot baseline: source = data/gpt-5.5_std_cls_logprobsFalse_fewshotegTrue1_baseline_870.jsonl
  B (chunk retrieval, BC grouped no-label; prompts identical for GPT-5.5 and GLM-5.1):
    source = outdir/prompt_gpt55_representative_ablation_test870/gpt-5.5_std_cls_grouped_no_label_baseline_compatible_fewshotegTrue_cls0.jsonl
Every source row's `messages` are sent verbatim (no truncation; an over-budget prompt is an
error). Decoding matches the std_cls runs: temperature 0, 1024 output tokens, thinking
disabled. Resumable: rows already answered in --out are skipped; failures go to
<out>.errors.jsonl and are retried on the next invocation.

python3 experiments/knowledge_rag_deepseek/replay_std_cls.py --source SRC --out OUT [--workers 4] [--max-calls N]
"""
import argparse
import concurrent.futures
import hashlib
import json
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_deepseek_endpoint import MODEL, configure  # noqa: E402

DROP = {'response', 'usage', 'reasoning', 'prompt_tokens', 'completion_tokens', 'reasoning_tokens', 'total_tokens'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--max-calls', type=int, default=0)
    a = ap.parse_args()
    configure()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    from legacy_api import load_client
    rows = [json.loads(l) for l in open(a.source)]
    src_sha = hashlib.sha256(a.source.read_bytes()).hexdigest()
    meta = a.out.with_suffix('.config.json')
    frozen = {'model': MODEL, 'source': str(a.source), 'source_sha256': src_sha, 'rows': len(rows),
              'temperature': 0, 'max_output': 1024, 'prompt_strategy': 'std', 'thinking': 'disabled',
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'gateway_client_sha256': hashlib.sha256((HERE / 'gateway_client.py').read_bytes()).hexdigest()}
    if meta.exists() and json.loads(meta.read_text()) != frozen:
        raise SystemExit('Source, script or client changed; use a new output file')
    meta.write_text(json.dumps(frozen, indent=1))
    done = {json.loads(l)['source_line'] for l in open(a.out)} if a.out.exists() else set()
    todo = [i for i in range(len(rows)) if i not in done]
    if a.max_calls:
        todo = todo[:a.max_calls]
    print(f'{len(rows)} rows, {len(done)} done, {len(todo)} to call', flush=True)
    lock = threading.Lock()
    local = threading.local()

    def call(i):
        if not hasattr(local, 'client'):
            local.client = load_client()
        c = local.client
        content, usage, _, reasoning = c.get_openai_chat({'messages': rows[i]['messages']}, MODEL, 'std', 0, 1024, 12345)
        rec = {k: v for k, v in rows[i].items() if k not in DROP}
        rec.update(source_line=i, model=MODEL, response=content, usage=usage, reasoning=reasoning,
                   provider_metadata=c.last_response_metadata)
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
                    if getattr(e, 'fatal', False):
                        ex.shutdown(cancel_futures=True)
                        raise SystemExit('Fatal gateway error (auth or returned model); stopped: ' + str(e)[:300])
                if (ok + err) % 50 == 0:
                    print(f'{ok} ok, {err} failed', flush=True)
    total = len({json.loads(l)['source_line'] for l in open(a.out)}) if a.out.exists() else 0
    print(f'finished this pass: {ok} ok, {err} failed; {total}/{len(rows)} rows answered', flush=True)


if __name__ == '__main__':
    main()

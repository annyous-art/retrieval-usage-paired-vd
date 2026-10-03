#!/usr/bin/env python3
"""Knowledge arms with a shared knowledge source: another model detects with GPT-5.5's (or Claude Opus 4.7's) knowledge.

In the fixed-candidate setting each model extracts its own knowledge from the same historical
candidate, so a difference between models in mode D can come from the extracted knowledge or
from its use. This script sends the two knowledge arms (all three fields; without the fixing
condition) of GPT-5.5 or of Claude Opus 4.7 (--source-tag) to another model. It first checks that
the model's own no-retrieval and code-pair prompts are identical to those of the knowledge source,
so the prompts differ from the model's own knowledge arms only in the knowledge text. The model's original gateway and inference configuration are reused
(no calls if they differ). Mirrors e_arm.py (validation, freezing, runner, model-name check).
No truncation: over-budget prompts are errors.
"""
import argparse, fcntl, hashlib, json, os, sys
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiment as exp

SOURCES = ('gpt55', 'claude47')  # tag of the model whose extracted knowledge is shared
SAME = ('no_retrieval', 'code_pair')  # arms whose prompts must be identical across models
# Name the gateway reports for a pinned alias; every other model must return its own name.
RETURNED = {'deepseek-v4-pro-0813': 'deepseek-v4-pro-ga-260813'}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_once(p, rows):
    if p.exists():
        if exp.read(p) != rows:
            raise ValueError('Frozen jobs changed')
    else:
        exp.write(p, rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', type=Path, required=True, help="the model's own complete v7 run")
    p.add_argument('--predictions', type=Path)
    p.add_argument('--knowledge-source', type=Path, required=True, help="v7 run of the model whose knowledge is shared")
    p.add_argument('--source-tag', choices=SOURCES, default='gpt55')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--stage', choices=['prepare', 'run'], default='prepare')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--max-calls', type=int, default=1740)
    a = p.parse_args()
    if a.workers < 1 or a.max_calls < 1:
        p.error('positive workers/max-calls required')
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(a)


def execute(a):
    src = a.source
    pred = a.predictions or src / 'predictions.jsonl'
    original_jobs = exp.read(src / 'detection_jobs.jsonl')
    original = exp.read(pred)
    if len(original) != len(original_jobs) or len({r['job_id'] for r in original}) != len(original):
        raise ValueError('Use complete original/recovered predictions')
    if {r['job_id'] for r in original} != {r['job_id'] for r in original_jobs}:
        raise ValueError('Original job IDs differ')
    if len({exp.digest(r['config']) for r in original}) != 1:
        raise ValueError('Mixed original configs')
    oj = {j['job_id']: j for j in original_jobs}
    for r in original:
        if r['messages'] != oj[r['job_id']]['messages'] or exp.prediction(r.get('response')) is None:
            raise ValueError('Invalid original prediction')
    cfg = original[0]['config']
    ARMS = {'knowledge': f'knowledge_{a.source_tag}', 'knowledge_no_fix': f'knowledge_no_fix_{a.source_tag}'}

    ks_jobs = exp.read(a.knowledge_source / 'detection_jobs.jsonl')
    own = {(j['row_id'], j['arm']): j for j in original_jobs}
    ks = {(j['row_id'], j['arm']): j for j in ks_jobs}
    for key, j in ks.items():
        if key[1] in SAME and (key not in own or own[key]['messages'] != j['messages']):
            raise ValueError(f'Prompt template differs from the knowledge source at {key}')
    jobs = []
    for j in ks_jobs:
        if j['arm'] not in ARMS:
            continue
        if own[(j['row_id'], j['arm'])]['candidate_ids'] != j['candidate_ids']:
            raise ValueError('Candidate differs from the knowledge source')
        arm = ARMS[j['arm']]
        n = {k: v for k, v in j.items() if k not in ('job_id', 'arm')}
        n.update(arm=arm, job_id=exp.digest([j['row_id'], arm, j['messages']]))
        jobs.append(n)

    frozen = {'config': cfg, 'arms': ARMS, 'files': {n: sha(src / n) for n in ['targets.jsonl', 'detection_jobs.jsonl']},
              'knowledge_source_jobs_sha256': sha(a.knowledge_source / 'detection_jobs.jsonl'),
              'predictions_sha256': sha(pred), 'script_sha256': sha(Path(__file__))}
    fp = a.out / 'config.json'
    if fp.exists() and json.loads(fp.read_text()) != frozen:
        raise ValueError('Configuration changed; use new output directory')
    fp.write_text(json.dumps(frozen, indent=2))
    write_once(a.out / 'jobs.jsonl', jobs)
    from context_budget import audit_requests
    audit_requests(jobs, cfg['model'], cfg['max_tokens'], a.out / 'context_audit.json', cfg['prompt_strategy'], cfg['temperature'])
    print(f'Prepared {len(jobs)} requests ({", ".join(ARMS.values())}) for {cfg["model"]}.', flush=True)
    if a.stage == 'prepare':
        return

    # Route the model to its ORIGINAL gateway.
    if os.environ.get('SLICERAG_UNIAPI_HOST', '\0') in cfg['base_url']:  # host of the OpenAI-compatible endpoint used for GLM/DeepSeek
        if cfg['model'].startswith('deepseek'):
            from run_deepseek_endpoint import configure
        else:
            from run_glm_endpoint import configure
        configure()
    from concurrent_runner import inference_config, run
    api = exp.load_client()
    args = NS(model=cfg['model'], temperature=cfg['temperature'], max_output=cfg['max_tokens'],
              prompt_strategy=cfg['prompt_strategy'], seed=cfg['seed'])
    if inference_config(api, args) != cfg:
        raise ValueError('Gateway/client/config differs from original; no calls made')
    expected = RETURNED.get(cfg['model'], cfg['model'])
    loader = exp.load_client

    def checked_client():
        c = loader()
        call = c.get_openai_chat

        def checked(*args, **kwargs):
            result = call(*args, **kwargs)
            if c.last_response_metadata.get('returned_model') != expected:
                raise ValueError('Returned model differs from requested model')
            return result
        c.get_openai_chat = checked
        return c
    exp.load_client = checked_client
    output = a.out / 'predictions.jsonl'
    run(NS(**vars(args), jobs=a.out / 'jobs.jsonl', output=output, workers=a.workers,
           read_timeout=600, max_attempts=3, max_calls=a.max_calls))
    rs = exp.read(output)
    print(f'{len(rs)}/{len(jobs)} answered.' + ('' if len(rs) == len(jobs) else ' Repeat the run to continue.'))


if __name__ == '__main__':
    main()

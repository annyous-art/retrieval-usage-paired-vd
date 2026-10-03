#!/usr/bin/env python3
"""Server entrypoint. all/extract/detect make real calls via old_baseline."""
import argparse
import hashlib
import json
import datetime
from pathlib import Path
from types import SimpleNamespace as NS
import experiment as exp
from legacy_api import baseline_dir, gateway_url
from gateway_client import ROUTES, request_body

ROOT = Path(__file__).resolve().parents[2]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['prepare','extract','assemble','detect','score','all'], default='prepare')
    p.add_argument('--train', type=Path, default=ROOT/'data/primevul_train_paired_labeled.jsonl')
    p.add_argument('--test', type=Path, default=ROOT/'data/primevul_test_paired_labeled.jsonl')
    p.add_argument('--out', type=Path, default=ROOT/'outdir/knowledge_rag_pilot')
    p.add_argument('--pairs', type=int, default=10, help='0 = all pairs, 10 = pipeline smoke run')
    p.add_argument('--k', type=exp.positive, default=1)
    p.add_argument('--test-pair-policy', choices=['adjacent','strict'], default='adjacent')
    p.add_argument('--seed', type=int, default=20260915)
    p.add_argument('--model', default='glm-5.1')
    p.add_argument('--temperature', type=float, default=0)
    p.add_argument('--prompt-strategy', choices=['std','cot'], default='std')
    p.add_argument('--extract-max-output', type=exp.positive, default=4096)
    p.add_argument('--max-output', type=exp.positive, default=4096)
    p.add_argument('--workers', type=exp.positive, default=4)
    p.add_argument('--read-timeout', type=exp.positive, default=600)
    p.add_argument('--max-attempts', type=exp.positive, default=3)
    p.add_argument('--reuse-extractions-from', type=Path)
    a = p.parse_args()
    if a.pairs < 0:
        p.error('--pairs must be nonnegative')
    d = a.out.resolve()
    a.out = d
    # Freeze configuration and source files; same directory cannot silently mix runs.
    operational={'stage','workers','read_timeout','max_attempts','reuse_extractions_from'}
    cfg = {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k not in operational}
    cfg['source_hashes'] = {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in {
        'experiment.py': Path(exp.__file__), 'legacy_api.py': Path(__file__).with_name('legacy_api.py'),
        'run_server.py':Path(__file__),
        'gateway_client.py':Path(__file__).with_name('gateway_client.py'),
        'concurrent_runner.py':Path(__file__).with_name('concurrent_runner.py'),
        'context_budget.py':Path(__file__).with_name('context_budget.py'),
        'context_budgets.json':Path(__file__).with_name('context_budgets.json'),
        'baseline/utils.py':baseline_dir()/'utils.py',
        'baseline/model_api_clients.py':baseline_dir()/'model_api_clients.py'}.items()}
    from context_budget import policy, encoders
    encoders()  # Validate offline vocabularies and pinned tokenizer before API work.
    cfg['context_policy'] = policy(a.model)
    cfg['baseline_directory'] = str(baseline_dir())
    cfg['gateway_base_url'] = gateway_url() or 'legacy-client-default'
    if gateway_url():
        if a.model not in ROUTES:
            p.error('Unsupported gateway model: '+a.model)
        cfg['api_route'] = ROUTES[a.model]
        cfg['effective_request_parameters'] = {
            stage:{k:v for k,v in request_body([],a.model,a.prompt_strategy,a.temperature,limit).items()
                   if k not in ('input','messages','system')}
            for stage,limit in [('extract',a.extract_max_output),('detect',a.max_output)]}
    cfg['data_hashes'] = {k:hashlib.sha256(path.read_bytes()).hexdigest() for k,path in [('train',a.train),('test',a.test)]}
    d.mkdir(parents=True, exist_ok=True)
    config_path = d/'server_config.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != cfg:
            raise ValueError('Configuration/data/source changed. Use a new --out directory.')
    else:
        config_path.write_text(json.dumps(cfg, indent=2))
    with (d/'execution_history.jsonl').open('a') as f:
        f.write(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                           **{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k in operational}})+'\n')
    stages = ['prepare','extract','assemble','detect','score'] if a.stage == 'all' else [a.stage]
    for stage in stages:
        print(f'=== {stage} ===', flush=True)
        if stage == 'prepare':
            if not (d/'manifest.json').exists():
                exp.prepare(a)
            else:
                print('Using existing manifest', flush=True)
            exp.preflight(a)
        elif stage in ('extract','detect'):
            if stage=='extract':
                exp.preflight(a)
            if a.reuse_extractions_from:
                from concurrent_runner import import_extractions
                import_extractions(a.reuse_extractions_from,d,NS(**{**vars(a),'max_output':a.extract_max_output if stage=='extract' else a.max_output}),stage)
            jobs = d/('extraction_jobs.jsonl' if stage == 'extract' else 'detection_jobs.jsonl')
            output = d/('knowledge_responses.jsonl' if stage == 'extract' else 'predictions.jsonl')
            exp.run(NS(jobs=jobs, output=output, model=a.model, temperature=a.temperature,
                       prompt_strategy=a.prompt_strategy, seed=a.seed,
                       max_output=a.extract_max_output if stage == 'extract' else a.max_output,
                       workers=a.workers,read_timeout=a.read_timeout,max_attempts=a.max_attempts,
                       max_calls=len(exp.read(jobs))))
        elif stage == 'assemble':
            if not (d/'detection_jobs.jsonl').exists():
                exp.assemble(NS(directory=d, knowledge=d/'knowledge_responses.jsonl', model=a.model, max_output=a.max_output, prompt_strategy=a.prompt_strategy, temperature=a.temperature))
        else:
            exp.score(NS(jobs=d/'detection_jobs.jsonl', responses=d/'predictions.jsonl',
                         output=d/'metrics.json', bootstrap=2000, seed=a.seed))

if __name__ == '__main__':
    main()

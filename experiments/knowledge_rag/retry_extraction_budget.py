#!/usr/bin/env python3
"""Explicit two-budget extraction recovery. Original experiment stays read-only."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import experiment as exp
from concurrent_runner import run
from context_budget import audit_requests
from gateway_client import request_body
from legacy_api import gateway_url, baseline_dir

HERE=Path(__file__).resolve().parent
SOURCE_FILES=('server_config.json','extraction_jobs.jsonl','knowledge_responses.jsonl','candidates.jsonl','targets.jsonl','manifest.json')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_records(records,jobs,cfg,budget):
    expected={j['job_id']:j for j in jobs};seen={}
    if len(expected)!=len(jobs):raise ValueError('Duplicate job IDs')
    for r in records:
        jid=r['job_id']
        if jid not in expected or jid in seen:raise ValueError('Unexpected/duplicate response ID')
        if r['messages']!=expected[jid]['messages'] or r.get('sent_messages')!=r['messages']:
            raise ValueError('Saved prompt mismatch')
        expected_config={'model':cfg['model'],'base_url':cfg['gateway_base_url'],'temperature':cfg['temperature'],
                         'prompt_strategy':cfg['prompt_strategy'],'seed':cfg['seed'],'max_tokens':budget}
        if any(r['config'].get(k)!=v for k,v in expected_config.items()):raise ValueError('Saved inference configuration mismatch')
        if r.get('provider_metadata',{}).get('generation_truncated'):raise ValueError('Truncated extraction cannot be reused')
        exp.parse_knowledge(r['response'])
        seen[jid]=r
    return seen


def assemble_merged(source_records,retry_records,jobs,cfg,budget):
    original=validate_records(source_records,jobs,cfg,cfg['extract_max_output'])
    retry=validate_records(retry_records,jobs,cfg,budget)
    if set(original)&set(retry):raise ValueError('Retry overlaps successful original jobs')
    merged={**original,**retry}
    if set(merged)!={j['job_id'] for j in jobs}:raise ValueError('Extraction incomplete; resume extract before assemble')
    return [merged[j['job_id']] for j in jobs]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stage',choices=['prepare','extract','assemble','detect','score','all'],default='prepare')
    p.add_argument('--extract-max-output',type=exp.positive,default=16384)
    p.add_argument('--workers',type=exp.positive,default=2)
    p.add_argument('--read-timeout',type=exp.positive,default=600)
    p.add_argument('--max-attempts',type=exp.positive,default=3)
    a=p.parse_args();source=a.source.resolve();out=a.out.resolve()
    if source==out or source in out.parents:raise ValueError('Use a separate output directory outside source')
    out.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        source_lock=source/'.run.lock'
        if source_lock.exists():
            original_lock=stack.enter_context(source_lock.open('r'))
            try:fcntl.flock(original_lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
            except BlockingIOError:raise ValueError('Original experiment is still running; wait before recovery') from None
        lock=stack.enter_context((out/'.recovery.lock').open('a'))
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Recovery is already running') from None
        execute(a,source,out)


def execute(a,source,out):
    cfg=json.loads((source/'server_config.json').read_text())
    if a.extract_max_output<=cfg['extract_max_output']:raise ValueError('Retry output budget must increase')
    if gateway_url()!=cfg['gateway_base_url']:raise ValueError('Gateway changed; this recovery changes output budget only')
    # Refuse silently changed routing, prompts, tokenizer policy, or experiment code.
    for name in ('experiment.py','gateway_client.py','legacy_api.py','concurrent_runner.py','context_budget.py','context_budgets.json'):
        if sha(HERE/name)!=cfg['source_hashes'][name]:raise ValueError('Source code/policy changed: '+name)
    if sha(baseline_dir()/'utils.py')!=cfg['source_hashes']['baseline/utils.py']:raise ValueError('Baseline prompt template changed')
    body=request_body([],cfg['model'],cfg['prompt_strategy'],cfg['temperature'],cfg['extract_max_output'])
    params={k:v for k,v in body.items() if k not in ('messages','system','input')}
    if params!=cfg['effective_request_parameters']['extract']:raise ValueError('Extraction request parameters changed')
    source_hashes={name:sha(source/name) for name in SOURCE_FILES}
    policy={'source':str(source),'source_sha256':source_hashes,'retry_script_sha256':sha(__file__),
            'original_extract_max_output':cfg['extract_max_output'],'retry_extract_max_output':a.extract_max_output,
            'classification_max_output_unchanged':cfg['max_output'],
            'rule':'Reuse valid original extractions; retry every remaining extraction with larger output budget; no input changes or truncation.'}
    policy_path=out/'recovery_config.json'
    if policy_path.exists():
        if json.loads(policy_path.read_text())!=policy:raise ValueError('Source/budget/recovery script changed; use a new recovery directory')
    else:
        if any(f.name!='.recovery.lock' for f in out.iterdir()):raise ValueError('Output directory is not empty and has no recovery config')
        policy_path.write_text(json.dumps(policy,indent=2))
    jobs=exp.read(source/'extraction_jobs.jsonl');original_records=exp.read(source/'knowledge_responses.jsonl')
    original=validate_records(original_records,jobs,cfg,cfg['extract_max_output'])
    pending=[j for j in jobs if j['job_id'] not in original]
    job_path=out/'retry_jobs.jsonl'
    if job_path.exists():
        if exp.read(job_path)!=pending:raise ValueError('Retry job manifest changed')
    else:exp.write(job_path,pending)
    for name in ('extraction_jobs.jsonl','candidates.jsonl','targets.jsonl','manifest.json'):
        target=out/name
        if not target.exists():target.write_bytes((source/name).read_bytes())
        if sha(target)!=source_hashes[name]:raise ValueError('Copied manifest changed: '+name)
    args=dict(model=cfg['model'],temperature=cfg['temperature'],prompt_strategy=cfg['prompt_strategy'],seed=cfg['seed'],
              workers=a.workers,read_timeout=a.read_timeout,max_attempts=a.max_attempts)
    retry_file=out/'retry_responses.jsonl'
    completed=exp.read(retry_file) if retry_file.exists() else []
    validate_records(completed,pending,cfg,a.extract_max_output)
    print(f'Original valid={len(original)} retry planned={len(pending)} retry cached={len(completed)} still pending={len(pending)-len(completed)} output={a.extract_max_output}',flush=True)
    stages=['prepare','extract','assemble','detect','score'] if a.stage=='all' else [a.stage]
    for stage in stages:
        print('=== '+stage+' ===',flush=True)
        if stage in ('prepare','extract'):
            audit_requests(pending,cfg['model'],a.extract_max_output,out/'retry_context_audit.json',cfg['prompt_strategy'],cfg['temperature'])
            if stage=='extract':run(NS(**args,jobs=job_path,output=retry_file,max_output=a.extract_max_output,max_calls=max(1,len(pending))))
        elif stage=='assemble':
            merged=assemble_merged(original_records,exp.read(retry_file) if retry_file.exists() else [],jobs,cfg,a.extract_max_output)
            merged_file=out/'knowledge_responses.jsonl'
            if not merged_file.exists():exp.write(merged_file,merged)
            elif exp.read(merged_file)!=merged:raise ValueError('Merged extraction records changed')
            if not (out/'detection_jobs.jsonl').exists():
                exp.assemble(NS(directory=out,knowledge=merged_file,model=cfg['model'],max_output=cfg['max_output'],prompt_strategy=cfg['prompt_strategy'],temperature=cfg['temperature']))
            (out/'extraction_recovery_report.json').write_text(json.dumps({'original_success':len(original),'retry_success':len(merged)-len(original),
                'total':len(merged),'extraction_output_budgets':[cfg['extract_max_output'],a.extract_max_output],
                'note':'Two-budget recovery, not a uniform-output-budget extraction experiment. Raw per-record configurations retained.'},indent=2))
        elif stage=='detect':
            run(NS(**args,jobs=out/'detection_jobs.jsonl',output=out/'predictions.jsonl',max_output=cfg['max_output'],max_calls=len(exp.read(out/'detection_jobs.jsonl'))))
        else:exp.score(NS(jobs=out/'detection_jobs.jsonl',responses=out/'predictions.jsonl',output=out/'metrics.json',bootstrap=2000,seed=cfg['seed']))

if __name__=='__main__':main()

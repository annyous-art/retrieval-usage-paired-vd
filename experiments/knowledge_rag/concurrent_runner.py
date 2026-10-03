"""Bounded independent requests; only the coordinator writes checkpoint files."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import hashlib
import json
import os
import threading
import time


def inference_config(api, a):
    from context_budget import POLICY_PATH
    return {'context_policy_sha256':hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest(),
            'model':a.model, 'api_client_sha256':hashlib.sha256(Path(api.__file__).read_bytes()).hexdigest(),
            'base_url':str(api.client.base_url), 'temperature':a.temperature,
            'max_tokens':a.max_output, 'prompt_strategy':a.prompt_strategy, 'seed':a.seed}


def run(a):
    import experiment as exp
    jobs = exp.read(a.jobs)
    api = exp.load_client()
    config = inference_config(api,a)
    expected = {j['job_id']:j for j in jobs}
    if len(expected) != len(jobs):
        raise ValueError('Duplicate planned job ids')
    results = exp.read(a.output) if Path(a.output).exists() else []
    for r in results:
        if r.get('config') != config or r['job_id'] not in expected or r.get('messages') != expected[r['job_id']]['messages']:
            raise ValueError('Resume config/prompt mismatch; use a new directory or verified extraction import')
    done = {r['job_id'] for r in results}
    if len(done) != len(results):
        raise ValueError('Duplicate results')
    pending = [j for j in jobs if j['job_id'] not in done][:a.max_calls]
    workers = getattr(a,'workers',1)
    limit = max(4096,a.max_output) if a.model == 'gpt-5.5' else a.max_output
    from context_budget import audit_requests
    audit_requests(jobs,a.model,a.max_output,str(a.output)+'.context_audit.json',a.prompt_strategy,a.temperature)
    for j in pending:
        if api.truncate_tokens_from_messages(j['messages'],a.model,limit) != j['messages']:
            raise ValueError('Context check would truncate '+j['job_id'])
    print(f'Queued={len(pending)} cached={len(done)} workers={workers} read_timeout={getattr(a,"read_timeout",600)}s',flush=True)
    local = threading.local()
    def task(j):
        if not hasattr(local,'api'):
            local.api = api if workers == 1 else exp.load_client()
            local.api.read_timeout = getattr(a,'read_timeout',600)
            local.api.max_attempts = getattr(a,'max_attempts',3)
        client = local.api
        started = time.monotonic()
        record = dict(j,config=config)
        try:
            content,usage,sent,reasoning = client.get_openai_chat({'messages':j['messages']},a.model,a.prompt_strategy,a.temperature,a.max_output,a.seed)
            record.update(response=content,usage=usage,sent_messages=sent,reasoning=reasoning)
            if hasattr(client,'last_response_metadata'):
                record['provider_metadata'] = client.last_response_metadata
            if not isinstance(content,str) or content.strip() in ('','[ERROR]','[EMPTY]'):
                raise ValueError('API returned no usable response')
            if sent != j['messages']:
                raise ValueError('API changed planned messages')
            truncated=record.get('provider_metadata',{}).get('generation_truncated',False)
            if truncated and ('candidate_id' in j or not exp.complete_final_answer(content)):
                raise ValueError('Output token limit reached without an acceptable complete final classification; raw response saved')
            if truncated:
                record['accepted_complete_label_despite_token_limit']=True
            if 'candidate_id' in j:
                _,record['knowledge_parse_repairs'] = exp.parse_knowledge(content,with_repairs=True)
            elif exp.prediction(content) is None:
                raise ValueError('Unparseable detection response')
            if 'row_id' in j:
                record.update(idx=j['idx'],query_idx=j['idx'],query_target=j['target'],sample_key=j['cluster'])
            record['elapsed_seconds'] = round(time.monotonic()-started,3)
            return True,record,False
        except Exception as exc:
            record.update(error=str(exc),error_type=type(exc).__name__,attempts=getattr(exc,'attempts',[]),
                          elapsed_seconds=round(time.monotonic()-started,3))
            if hasattr(exc,'response_data'):
                record['provider_response']=exc.response_data
            return False,record,getattr(exc,'fatal',False)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    started = time.monotonic()
    failures,new_success = [],[]
    queue = iter(pending)
    fatal = False
    with open(a.output,'a') as output, open(str(a.output)+'.errors.jsonl','a') as errors, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(task,j):j for j in [next(queue,None) for _ in range(min(workers,len(pending)))] if j is not None}
        while futures:
            finished,_ = wait(futures,timeout=30,return_when=FIRST_COMPLETED)
            if not finished:
                print(f'Waiting: active={len(futures)} completed={len(done)}/{len(jobs)} failed={len(failures)} stage_elapsed={time.monotonic()-started:.0f}s',flush=True)
            for future in finished:
                j = futures.pop(future)
                ok,record,is_fatal = future.result()
                sink = output if ok else errors
                sink.write(json.dumps(record,ensure_ascii=False)+'\n');sink.flush();os.fsync(sink.fileno())
                if ok:
                    done.add(j['job_id']);new_success.append(record['elapsed_seconds'])
                    print(f'Completed {len(done)}/{len(jobs)} {j["job_id"][:12]} elapsed={record["elapsed_seconds"]:.1f}s',flush=True)
                else:
                    failures.append(j['job_id']);fatal |= is_fatal
                    print(f'Failed {j["job_id"][:12]} elapsed={record["elapsed_seconds"]:.1f}s {record["error"][:240]}',flush=True)
                if not fatal:
                    nxt = next(queue,None)
                    if nxt is not None:
                        futures[pool.submit(task,nxt)] = nxt
    summary = {'planned':len(jobs),'cached_before':len(results),'successful_this_run':len(new_success),
               'completed':len(done),'failed_job_ids':failures,'workers':workers,
               'stage_elapsed_seconds':round(time.monotonic()-started,3),'successful_request_seconds':new_success}
    Path(str(a.output)+'.run_summary.json').write_text(json.dumps(summary,indent=2))
    if failures:
        raise ValueError(f'{len(failures)} jobs failed; successful results saved. Resume to retry only missing jobs. See {a.output}.errors.jsonl')


def import_extractions(source, destination, a, stage='extract'):
    """Import exact-matching cached results; original configs and raw outputs retained."""
    import experiment as exp
    source,destination = Path(source).resolve(),Path(destination).resolve()
    if source == destination:
        raise ValueError('Import source must differ from output')
    old = json.loads((source/'server_config.json').read_text())
    new = json.loads((destination/'server_config.json').read_text())
    budget = 'extract_max_output' if stage=='extract' else 'max_output'
    for field in ('data_hashes','model','temperature','prompt_strategy','seed',budget,'gateway_base_url','api_route'):
        if old.get(field) != new.get(field):
            raise ValueError('Cache import mismatch: '+field)
    if old['effective_request_parameters'][stage] != new['effective_request_parameters'][stage]:
        raise ValueError('Effective request parameters changed')
    name = 'knowledge_responses.jsonl' if stage=='extract' else 'predictions.jsonl'
    jobs_name = 'extraction_jobs.jsonl' if stage=='extract' else 'detection_jobs.jsonl'
    expected = {j['job_id']:j for j in exp.read(destination/jobs_name)}
    output = destination/name
    current = exp.read(output) if output.exists() else []
    seen = {r['job_id'] for r in current}
    config = inference_config(exp.load_client(),a)
    accepted=[]
    valid_source_ids=set()
    source_files=[(source/name,False),(source/(name+'.errors.jsonl'),True)]
    for source_file,from_error in source_files:
        if not source_file.exists(): continue
        source_hash=hashlib.sha256(source_file.read_bytes()).hexdigest()
        for r in exp.read(source_file):
            jid=r['job_id']
            if jid in valid_source_ids:
                if not from_error: raise ValueError('Duplicate source result')
                continue
            if jid not in expected: raise ValueError('Source job is not in the frozen job set')
            if r['messages'] != expected[jid]['messages']:
                raise ValueError('Source prompt differs from the new prompt')
            if from_error and not isinstance(r.get('response'),str): continue
            if r.get('sent_messages') != r['messages']:
                raise ValueError('Source sent messages differ from planned messages')
            for key in ('model','base_url','temperature','max_tokens','prompt_strategy','seed'):
                if r['config'].get(key) != config.get(key): raise ValueError('Source record config mismatch: '+key)
            repairs=[]
            try:
                if r.get('provider_metadata',{}).get('generation_truncated') and (stage=='extract' or not exp.complete_final_answer(r['response'])):
                    raise ValueError('Truncated cached response is not acceptable')
                if stage=='extract':
                    _,repairs=exp.parse_knowledge(r['response'],with_repairs=True)
                elif exp.prediction(r['response']) is None:
                    raise ValueError('Unparseable cached detection')
            except (ValueError,TypeError):
                if from_error: continue
                raise
            valid_source_ids.add(jid)
            if jid not in seen:
                copy={k:v for k,v in r.items() if k not in ('error','error_type','attempts')}
                copy.update(config=config,import_provenance={'directory':str(source),'original_config':r['config'],
                            'source_file':name+('.errors.jsonl' if from_error else ''),
                            'source_file_sha256':source_hash,'recovered_from_error_log':from_error})
                if stage=='extract': copy['knowledge_parse_repairs']=repairs
                accepted.append(copy);seen.add(jid)
    if accepted:
        with output.open('a') as f:
            for r in accepted:
                f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
    print(f'Imported {len(accepted)} verified {stage} responses; source unchanged',flush=True)

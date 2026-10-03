#!/usr/bin/env python3
"""Fixed-candidate chunk representation or shuffled existing knowledge. No truncation."""
import argparse, fcntl, hashlib, json, os, random, sys
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import experiment as exp
from offline_review import comparisons

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write_once(p,rows):
    if p.exists():
        if exp.read(p)!=rows:raise ValueError('Frozen jobs changed')
    else:exp.write(p,rows)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--predictions',type=Path)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--arm',choices=['slice_fixed','shuffled_knowledge'],required=True)
    p.add_argument('--chunk-metadata',type=Path);p.add_argument('--stage',choices=['prepare','run','score'],default='prepare')
    p.add_argument('--workers',type=int,default=2);p.add_argument('--max-calls',type=int,default=870)
    a=p.parse_args()
    if a.workers<1 or a.max_calls<1:p.error('positive workers/max-calls required')
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);execute(a)

def execute(a):
    src=a.source;pred=a.predictions or src/'predictions.jsonl'
    original_jobs=exp.read(src/'detection_jobs.jsonl');original=exp.read(pred)
    # Complete prior results are required before adding a new arm.
    if len(original)!=len(original_jobs) or len({r['job_id'] for r in original})!=len(original):raise ValueError('Use complete original/recovered predictions')
    if {r['job_id'] for r in original}!={r['job_id'] for r in original_jobs}:raise ValueError('Original job IDs differ')
    configs={exp.digest(r['config']) for r in original}
    if len(configs)!=1:raise ValueError('Mixed original configs')
    oj={j['job_id']:j for j in original_jobs}
    for r in original:
        if r['messages']!=oj[r['job_id']]['messages'] or exp.prediction(r.get('response')) is None:raise ValueError('Invalid original prediction')
    cfg=original[0]['config'];targets=exp.read(src/'targets.jsonl');cand={r['pair_id']:r for r in exp.read(src/'candidates.jsonl')}
    no={r['row_id']:r for r in original_jobs if r['arm']=='no_retrieval'}
    ej={r['job_id']:r['candidate_id'] for r in exp.read(src/'extraction_jobs.jsonl')}
    know={ej[r['job_id']]:exp.parse_knowledge(r['response']) for r in exp.read(src/'knowledge_responses.jsonl')}
    frozen={'arm':a.arm,'config':cfg,'files':{n:sha(src/n) for n in ['targets.jsonl','candidates.jsonl','detection_jobs.jsonl','knowledge_responses.jsonl']},'predictions_sha256':sha(pred),'script_sha256':sha(Path(__file__)),'stats_sha256':sha(Path(__file__).with_name('offline_review.py')),'seed':20260915}
    chunks={};mapping={}
    if a.arm=='slice_fixed':
        if not a.chunk_metadata:raise ValueError('--chunk-metadata required')
        frozen['chunk_metadata_sha256']=sha(a.chunk_metadata)
        wanted={exp.digest(m['func']) for c in cand.values() for m in c['members']}
        # Match exact historical code and project/commit/idx, never target labels.
        with a.chunk_metadata.open() as metadata:
            for line in metadata:
                c=json.loads(line);h=exp.digest(c.get('func',''))
                if h not in wanted:continue
                key=(h,str(c.get('function_id',c.get('file_id'))),str(c.get('project')),str(c.get('commit_id')))
                rank=(-float(c['score_combined']),int(c['chunk_seq']),int(c['chunk_id']))
                if key not in chunks or rank<chunks[key][0]:chunks[key]=(rank,c)
        for cid,c in cand.items():
            texts=[];ids=[]
            for role,m in zip(['vulnerable','repaired'],c['members']):
                key=(exp.digest(m['func']),str(m['idx']),str(m.get('project')),str(m.get('commit_id')))
                if key not in chunks:raise ValueError('Missing exact historical chunk: '+cid+' '+role)
                ch=chunks[key][1];lo=int(ch['start_line']);hi=int(ch['end_line'])
                if ch['code_raw']!='\n'.join(m['func'].splitlines()[lo-1:hi]):raise ValueError('Chunk differs from source lines')
                texts.append('Historical '+role+' code slice:\n'+ch['code_raw']);ids.append(ch['chunk_id'])
            mapping[cid]={'context':'\n'.join(texts),'chunk_ids':ids}
    else:
        # Permute assignments, retaining exactly the same frequency of historical candidates.
        order=list(range(len(targets)));random.Random(20260915).shuffle(order)
        shift=next((k for k in range(1,len(order)) if all(not set(targets[order[i]]['candidate_ids']) & set(targets[order[(i+k)%len(order)]]['candidate_ids']) for i in range(len(order)))),None)
        if shift is None:raise ValueError('No disjoint cyclic permutation; do not silently change control design')
        mapping={targets[order[i]]['row_id']:targets[order[(i+shift)%len(order)]]['candidate_ids'] for i in range(len(order))}
        frozen['control']='frequency-preserving permutation among existing selected candidates; not random retrieval from full training pool; not per-target length matched'
    fp=a.out/'config.json'
    if fp.exists() and json.loads(fp.read_text())!=frozen:raise ValueError('Configuration changed; use new output directory')
    contexts=[];jobs=[]
    for t in targets:
        ids=t['candidate_ids'] if a.arm=='slice_fixed' else mapping[t['row_id']]
        if a.arm=='slice_fixed':context='\n\n'.join(mapping[c]['context'] for c in ids)
        else:context='\n\n'.join('Functional semantics: '+know[c]['functional_semantics']+'\nRoot cause: '+know[c]['root_cause']+'\nRepair condition: '+know[c]['fix_condition'] for c in ids)
        messages=[dict(m) for m in no[t['row_id']]['messages']]
        marker='Historical evidence (may be irrelevant; check the target independently):\n(none)\n\n'
        if not messages[1]['content'].startswith(marker):raise ValueError('Unknown base prompt')
        messages[1]['content']=marker.replace('(none)',context)+messages[1]['content'][len(marker):]
        j={k:v for k,v in t.items() if k!='func'}
        j.update(arm=a.arm,messages=messages,candidate_ids=ids,job_id=exp.digest([t['row_id'],a.arm,messages]))
        jobs.append(j);contexts.append({'row_id':t['row_id'],'candidate_ids':ids,'evidence_bytes':len(context.encode()),'chunk_ids':[mapping[c]['chunk_ids'] for c in ids] if a.arm=='slice_fixed' else None})
    fp.write_text(json.dumps(frozen,indent=2));write_once(a.out/'jobs.jsonl',jobs);write_once(a.out/'evidence_manifest.jsonl',contexts)
    from context_budget import audit_requests
    audit_requests(jobs,cfg['model'],cfg['max_tokens'],a.out/'context_audit.json',cfg['prompt_strategy'],cfg['temperature'])
    print(f'Prepared {len(jobs)} {a.arm} requests; no extraction needed.',flush=True)
    output=a.out/'predictions.jsonl'
    if a.stage=='prepare':return
    if a.stage=='run':
        # Route each model to its ORIGINAL gateway, including GLM UniAPI.
        if os.environ.get('SLICERAG_UNIAPI_HOST', '\0') in cfg['base_url']:  # host of the OpenAI-compatible endpoint used for GLM/DeepSeek
            from run_glm_endpoint import configure
            configure()
        from concurrent_runner import inference_config,run
        api=exp.load_client();args=NS(model=cfg['model'],temperature=cfg['temperature'],max_output=cfg['max_tokens'],prompt_strategy=cfg['prompt_strategy'],seed=cfg['seed'])
        if inference_config(api,args)!=cfg:raise ValueError('Gateway/client/config differs from original; no calls made')
        # Reject mismatched model names before the existing runner caches a success.
        loader=exp.load_client
        def checked_client():
            c=loader();call=c.get_openai_chat
            def checked(*args,**kwargs):
                result=call(*args,**kwargs)
                if c.last_response_metadata.get('returned_model')!=cfg['model']:raise ValueError('Returned model differs from requested model')
                return result
            c.get_openai_chat=checked;return c
        exp.load_client=checked_client
        run(NS(**vars(args),jobs=a.out/'jobs.jsonl',output=output,workers=a.workers,read_timeout=600,max_attempts=3,max_calls=a.max_calls))
    rs=exp.read(output)
    if len(rs)!=len(jobs):print('Partial run saved. Repeat run without --max-calls to continue.');return
    # Generic comparisons support the fifth arm without modifying frozen original code.
    stats=comparisons(original_jobs+jobs,original+rs)
    (a.out/'paired_comparisons.json').write_text(json.dumps(stats,indent=2))
    states={r['job_id']:exp.prediction(r['response']) for r in rs}
    tp=sum(states[j['job_id']]==1 and j['target']==1 for j in jobs);fpn=sum(states[j['job_id']]==1 and j['target']==0 for j in jobs)
    fn=sum(states[j['job_id']]==0 and j['target']==1 for j in jobs);tn=len(jobs)-tp-fpn-fn
    pairs={}
    for j in jobs:pairs.setdefault(j['pair_id'],{})[j['target']]=states[j['job_id']]
    pc=sum(v=={0:0,1:1} for v in pairs.values())/len(pairs)
    (a.out/'arm_metrics.json').write_text(json.dumps({'arm':a.arm,'calls':len(jobs),'P-C':pc,'pairs':len(pairs),'TP':tp,'FP':fpn,'FN':fn,'TN':tn,'F1':2*tp/(2*tp+fpn+fn) if 2*tp+fpn+fn else 0},indent=2))
    print('Complete: arm_metrics.json and paired_comparisons.json')
if __name__=='__main__':main()

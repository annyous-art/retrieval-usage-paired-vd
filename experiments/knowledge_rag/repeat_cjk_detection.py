#!/usr/bin/env python3
"""Re-request the detection answers whose text contains CJK characters.

Same procedure as repeat_selected_detection.py (frozen source snapshot, identical prompts and
inference configuration, extracted knowledge unchanged, source never overwritten); only the
selection differs: the original answer contains CJK characters (a Chinese preamble before the
YES/NO answer). The merged predictions replace only the selected answers.
"""
import argparse, collections, contextlib, fcntl, hashlib, io, json, os, re, shutil
from pathlib import Path
from types import SimpleNamespace as NS
import experiment as exp
from concurrent_runner import run, inference_config

FILES=('detection_jobs.jsonl','predictions.jsonl','knowledge_responses.jsonl','extraction_jobs.jsonl','targets.jsonl','metrics.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,p)
def validate(jobs,rows,cfg=None):
    expected={j['job_id']:j for j in jobs}
    if len(expected)!=len(jobs) or len({r['job_id'] for r in rows})!=len(rows):raise ValueError('Duplicate IDs')
    for r in rows:
        if r['job_id'] not in expected or r['messages']!=expected[r['job_id']]['messages'] or r.get('sent_messages')!=r['messages']:raise ValueError('Unexpected ID or changed prompt')
        if cfg is not None and r['config']!=cfg:raise ValueError('Inference configuration mismatch')
        if exp.prediction(r.get('response')) is None:raise ValueError('Unparseable successful record')
    return expected

def prepare(a):
    src=a.source.resolve();out=a.out.resolve()
    if src==out or src in out.parents or out in src.parents:raise ValueError('Use separate sibling output directory')
    jobs=exp.read(src/'detection_jobs.jsonl');old=exp.read(src/'predictions.jsonl');validate(jobs,old)
    if len(old)!=len(jobs) or len({exp.digest(r['config']) for r in old})!=1:raise ValueError('Source must be complete and have one configuration')
    obs={r['job_id']:r for r in old};cfg=old[0]['config'];selected=[j for j in jobs if re.search(r'[\u4e00-\u9fff]',obs[j['job_id']].get('response') or '')]
    if not selected:raise ValueError('No answer contains CJK characters')
    hashes={f:sha(src/f) for f in FILES}
    here=Path(__file__).resolve().parent
    frozen={'source':str(src),'source_sha256':hashes,'config':cfg,'selection':'original answer contains CJK characters','selected_ids':[j['job_id'] for j in selected],'total':len(jobs),'selected':len(selected),'code_sha256':{f:sha(here/f) for f in ['repeat_cjk_detection.py','concurrent_runner.py','experiment.py']},'design':'Repeat selected detection requests; keep original knowledge and all new returned names; one successful parseable response per selected ID; failed calls can be resumed.'}
    fp=out/'repeat_manifest.json'
    if fp.exists() and json.loads(fp.read_text())!=frozen:raise ValueError('Frozen source/code changed; use a new output directory')
    snap=out/'source_snapshot';snap.mkdir(exist_ok=True)
    for name,h in hashes.items():
        dest=snap/name
        if not dest.exists():shutil.copy2(src/name,dest)
        if sha(dest)!=h:raise ValueError('Snapshot hash mismatch: '+name)
    if (out/'repeat_jobs.jsonl').exists():
        if exp.read(out/'repeat_jobs.jsonl')!=selected:raise ValueError('Selected jobs changed')
    else:exp.write(out/'repeat_jobs.jsonl',selected)
    save(fp,frozen)
    print(f'Source={len(jobs)} selected={len(selected)} original files unchanged. Snapshot={snap}',flush=True)
    return frozen

def compare(a,frozen):
    snap=a.out/'source_snapshot';jobs=exp.read(snap/'detection_jobs.jsonl');old=exp.read(snap/'predictions.jsonl')
    selected=exp.read(a.out/'repeat_jobs.jsonl');file=a.out/'repeat_responses.jsonl';new=exp.read(file) if file.exists() else []
    validate(selected,new,frozen['config']);original={r['job_id']:r for r in old};repeated={r['job_id']:r for r in new}
    names=collections.Counter();transitions=collections.Counter();arms=collections.defaultdict(collections.Counter);same_text=0;changed=0;details=[]
    for r in new:
        before=original[r['job_id']];name=r.get('provider_metadata',{}).get('returned_model','missing');names[name]+=1
        y0,y1=exp.prediction(before['response']),exp.prediction(r['response']);transition=f'{y0}->{y1}';transitions[transition]+=1;changed+=y0!=y1;same_text+=before['response']==r['response']
        arms[r['arm']]['completed']+=1;arms[r['arm']]['label_changed']+=y0!=y1
        details.append({'job_id':r['job_id'],'arm':r['arm'],'row_id':r['row_id'],'before_model':before.get('provider_metadata',{}).get('returned_model'),'after_model':name,'before_label':y0,'after_label':y1,'same_text':before['response']==r['response']})
    complete=len(new)==len(selected)
    report={'selected':len(selected),'completed':len(new),'remaining':len(selected)-len(new),'complete':complete,'returned_models_after_repeat':dict(names),'label_transitions_completed_only':dict(transitions),'label_changed':changed,'exact_response_text_same':same_text,'by_arm_completed_only':dict(arms),'original_predictions_modified':False,'knowledge_reextracted':False,'warning':'Selection used the language of the original answer. Only the selected answers are replaced in the merged file.'}
    save(a.out/'comparison.json',report);save(a.out/'comparison_per_job.json',details)
    if complete:
        merged=[repeated.get(j['job_id'],original[j['job_id']]) for j in jobs];path=a.out/'predictions_repeat_merged.jsonl'
        if path.exists():
            if exp.read(path)!=merged:raise ValueError('Merged output differs')
        else:exp.write(path,merged)
        with contextlib.redirect_stdout(io.StringIO()):exp.score(NS(jobs=snap/'detection_jobs.jsonl',responses=path,output=a.out/'metrics_repeat_merged.json',bootstrap=2000,seed=frozen['config']['seed']))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stage',choices=['prepare','run','compare'],default='prepare');p.add_argument('--returned-model',default='gemini-3.1-pro-low');p.add_argument('--workers',type=int,default=2);p.add_argument('--read-timeout',type=int,default=600);p.add_argument('--max-attempts',type=int,default=3)
    a=p.parse_args()
    if min(a.workers,a.read_timeout,a.max_attempts)<1:p.error('Limits must be positive')
    if a.source.resolve()==a.out.resolve() or a.source.resolve() in a.out.resolve().parents or a.out.resolve() in a.source.resolve().parents:p.error('Use an independent output directory')
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'.run.lock').open('a') as lock,(a.source/'.run.lock').open('a') as sl:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(sl,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Source or output directory is running; wait for it to stop')
        frozen=prepare(a)
        if a.stage=='prepare':return
        if a.stage=='compare':compare(a,frozen);return
        cfg=frozen['config'];args=NS(model=cfg['model'],temperature=cfg['temperature'],max_output=cfg['max_tokens'],prompt_strategy=cfg['prompt_strategy'],seed=cfg['seed'])
        api=exp.load_client()
        if inference_config(api,args)!=cfg:raise ValueError('Gateway/client/config differs from original; no API calls made')
        try:run(NS(**vars(args),jobs=a.out/'repeat_jobs.jsonl',output=a.out/'repeat_responses.jsonl',max_calls=frozen['selected'],workers=a.workers,read_timeout=a.read_timeout,max_attempts=a.max_attempts))
        finally:compare(a,frozen)
if __name__=='__main__':main()

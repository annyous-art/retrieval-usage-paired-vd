#!/usr/bin/env python3
"""Resumable full-method adaptation of official Vul-RAG repository. No silent truncation."""
import importlib.metadata, platform
import argparse, collections, concurrent.futures, difflib, fcntl, hashlib, json, os, re, sys, threading, time
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE/'support'))
from prompts import generate_extract_prompt, generate_extraction_prompt_for_vulrag, generate_detect_vul_prompt_with_response_in_HTML, generate_detect_sol_prompt_with_response_in_HTML
from data_utils import read, pairs, known, digest
from context_budget import audit_requests, estimate
from gateway_client import GatewayClient

# OpenAI-compatible endpoints are read from the environment; no host is hard-coded.
PROVIDERS={'uniapi':os.environ.get('SLICERAG_UNIAPI_BASE_URL',''),'newapi':os.environ.get('SLICERAG_NEWAPI_BASE_URL','')}
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,path)
def load(path): return json.loads(Path(path).read_text())
def message(s): return [{'role':'user','content':s}]
def runtime_versions():
    result={'python':platform.python_version()}
    for name in ('numpy','spacy','en-core-web-sm','rank-bm25','tiktoken','requests'):
        try: result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result[name]='not-installed'
    return result
def protocol(a):
    files=[HERE/'run.py',HERE/'prompts.py',HERE/'retrieval.py',HERE/'data_utils.py',HERE/'requirements.txt',HERE/'vendor/provenance.json',HERE/'support/gateway_client.py',HERE/'support/context_budget.py',HERE/'support/context_budgets.json']
    return {'model':a.model,'provider':a.provider,'base_url':PROVIDERS[a.provider],'temperature':a.temperature,'output_tokens':a.max_output,'prompt_strategy':'std','retrieval_top_k':20,'max_knowledge':3,'early_return':True,'upstream_commit':load(HERE/'vendor/provenance.json')['commit'],'code_sha256':{p.name:sha(p) for p in files},'profile':'official-repository-README-adapted-v1','runtime':runtime_versions()}
def modified(v,f):
    changes=list(difflib.ndiff(v.splitlines(),f.splitlines()))
    return {'added':[l[2:] for l in changes if l.startswith('+ ')],'deleted':[l[2:] for l in changes if l.startswith('- ')]}
def train_prompts(p):
    v,f=p['members'];return generate_extract_prompt(v['cve'],v.get('cve_desc') or 'CVE description unavailable in the supplied training data.',modified(v['func'],f['func']),v['func'],f['func'])
def prepare(a):
    source=a.source;train=read(a.train);test=read(a.test)
    if len(test)%2: raise ValueError('Odd test row count')
    original=load(source/'manifest.json')
    if original['train_sha256']!=digest(train) or original['test_sha256']!=digest(test): raise ValueError('Source manifest/data mismatch')
    pool=read(source/'candidate_pool.jsonl');targets=read(source/'targets.jsonl')
    if len(targets)!=len(test) or {r['row_id'] for r in targets}!=set(range(1,len(test)+1)): raise ValueError('Source must be full benchmark, one record per row occurrence')
    for t in targets:
        raw=test[t['row_id']-1]
        if t['func']!=raw['func'] or t['idx']!=raw['idx'] or t['target']!=int(raw['target']): raise ValueError('Target row mismatch')
    for i in range(0,len(test),2):
        if {int(r['target']) for r in test[i:i+2]}!={0,1}: raise ValueError('Invalid adjacent test pair')
    tc={r.get('cve') for r in test if known(r.get('cve'))};commits={(r.get('project'),r.get('commit_id')) for r in test};codes={digest(''.join(r['func'].split())) for r in test}
    eligible=[p for p in pairs(train)[0] if not any(r.get('cve') in tc or (r.get('project'),r.get('commit_id')) in commits or digest(''.join(r['func'].split())) in codes for r in p['members'])]
    if digest(pool)!=digest(eligible): raise ValueError('Candidate pool differs from full train/test leakage filtering')
    if not pool: raise ValueError('Empty training pool')
    settings=protocol(a);fingerprint=digest([settings,pool,targets])
    manifest={'protocol':settings,'fingerprint':fingerprint,'source_manifest_sha256':sha(source/'manifest.json'),'train_file_sha256':sha(a.train),'test_file_sha256':sha(a.test),'pool_sha256':digest(pool),'targets_sha256':digest(targets),'training_pairs':len(pool),'test_rows':len(targets),'test_pairs':len(test)//2,'missing_training_cve_description':sum(not known(p['members'][0].get('cve_desc')) for p in pool),'logical_call_upper_bounds_no_retries':{'kb':4*len(pool),'query':2*len(targets),'detect':6*len(targets)},'annotation_required':False,'no_api_calls_during_prepare':True}
    if (a.out/'manifest.json').exists() and load(a.out/'manifest.json')['fingerprint']!=fingerprint: raise ValueError('Changed protocol/input: use a new output directory')
    atomic(a.out/'training.json',pool);atomic(a.out/'targets.json',targets);atomic(a.out/'manifest.json',manifest)
    jobs=[]
    for p in pool:
        prompts=train_prompts(p)
        for phase,s in zip(('purpose','function','analysis'),prompts[:3]): jobs.append({'stage':'kb_'+phase,'item':p['pair_id'],'messages':message(s)})
    for t in targets:
        for phase,s in zip(('purpose','function'),generate_extraction_prompt_for_vulrag(t['func'])): jobs.append({'stage':'query_'+phase,'item':t['row_id'],'messages':message(s)})
    audit_requests(jobs,a.model,a.max_output,a.out/'preflight.json',temperature=a.temperature,extra={'pending':'Knowledge-organization and detector requests depend on generated content; checked in full immediately before each API call. This is an estimate, not provider capacity proof.'})
    atomic(a.out/'prepare_complete.json',{'fingerprint':fingerprint,'preflight_sha256':sha(a.out/'preflight.json')})
    print(json.dumps(manifest,indent=2),flush=True)

def nonempty(text):
    if not isinstance(text,str) or not text.strip(): raise ValueError('Empty output')
    return text.strip()
def prefix(text,tag):
    return nonempty(text.split(tag)[1].strip() if tag in text else text.strip())
def knowledge_json(text):
    s=re.sub(r'^```(?:json)?\s*|\s*```$','',text.strip(),flags=re.I)
    obj=json.loads(s)
    if not isinstance(obj,dict): raise ValueError('Knowledge must be an object')
    b=obj.get('vulnerability_behavior',{})
    for k in ('vulnerability_cause_description','trigger_condition','specific_code_behavior_causing_vulnerability'): nonempty(b.get(k))
    nonempty(obj.get('solution',b.get('solution')))
    return obj

def label(text):
    # Stricter than upstream substring test; ambiguous/unparseable response stays pending.
    clean=re.sub(r'<think>.*?</think>','',text,flags=re.S|re.I)
    tags=re.findall(r'<result>\s*(.*?)\s*</result>',clean,flags=re.S|re.I)
    if not tags or tags[-1].strip().upper() not in ('YES','NO'): raise ValueError('Expected final <result> YES/NO </result>')
    return int(tags[-1].strip().upper()=='YES')
def decision(v,s):
    if v==1 and s==0: return 1
    if s==1: return 0 # upstream README --early_return
    return None

def get_key(a):
    if a.provider=='uniapi':
        key=os.environ.get('SLICERAG_UNIAPI_KEY','');p=HERE.parent/'knowledge_rag/glm_endpoint.key'
        if not key and p.exists(): key=p.read_text().strip()
    else:
        key=os.environ.get('SLICERAG_API_KEY','')
        if not key:
            # Existing config only; never copy its secret into this package.
            import importlib.util
            p=HERE.parent/'knowledge_rag/gateway_config_local.py'
            spec=importlib.util.spec_from_file_location('local_gateway',p);cfg=importlib.util.module_from_spec(spec);spec.loader.exec_module(cfg)
            # Names checked against installed legacy_api loader.
            key=getattr(cfg,'API_KEY','') if getattr(cfg,'BASE_URL','').rstrip('/')==PROVIDERS[a.provider] else ''
    if not key: raise ValueError('Missing API key for '+a.provider+'; see README_ZH.md')
    return key

class Calls:
    def __init__(self,a,manifest,client_factory=None):
        self.a=a;self.manifest=manifest;self.key=None;self.factory=client_factory;self.lock=threading.Lock();self.count=0
    def call(self,logical_id,messages,validator=nonempty):
        identity={'logical_id':logical_id,'messages':messages,'protocol':self.manifest['protocol']}
        cid=digest(identity);p=self.a.out/'calls'/f'{cid}.json'
        if p.exists():
            record=load(p)
            if record['identity']!=identity: raise ValueError('Cache identity mismatch')
            validator(record['content']);return record
        audit=estimate(messages,self.a.model,self.a.max_output,temperature=self.a.temperature)
        if audit['status']!='within_estimated_budget':
            atomic(self.a.out/'capacity_errors'/f'{cid}.json',{'logical_id':logical_id,'estimate':audit})
            raise ValueError('Context estimate needs review; no truncation or call')
        with self.lock:
            if self.count>=self.a.max_calls: raise ValueError('Invocation max-calls reached; resume same command')
            self.count+=1
            if not self.factory and self.key is None: self.key=get_key(self.a)
        client=self.factory() if self.factory else GatewayClient(PROVIDERS[self.a.provider],self.key)
        client.read_timeout=self.a.read_timeout;client.max_attempts=self.a.max_attempts;client.retry_base=10
        record={'identity':identity,'call_id':cid,'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
        try:
            content,usage,sent,reasoning=client.get_openai_chat({'messages':messages},self.a.model,'std',self.a.temperature,self.a.max_output,20260921)
            meta=client.last_response_metadata
            record.update(content=content,usage=usage,response_metadata=meta)
            if sent!=messages: raise ValueError('Client changed full prompt')
            if meta.get('generation_truncated'): raise ValueError('Output budget exhausted; incomplete output rejected')
            if meta.get('returned_model')!=self.a.model: raise ValueError('Returned model identifier mismatch; inspect raw response before accepting')
            validator(content)
        except Exception as e:
            record.update(error=str(e),raw_response=getattr(e,'response_data',None),attempts=getattr(e,'attempts',None))
            atomic(self.a.out/'failures'/f'{cid}_{time.time_ns()}.json',record)
            raise
        atomic(p,record);return record

def parallel(a,items,worker,stage):
    failures=[];done=0;start=time.monotonic();stop=threading.Event()
    def guarded(item):
        if stop.is_set():raise ValueError('Not attempted after a fatal API error; resume later')
        try:return worker(item)
        except Exception as e:
            if getattr(e,'fatal',False):stop.set()
            raise
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as executor:
        pending={executor.submit(guarded,item):item for item in items}
        while pending:
            finished,_=concurrent.futures.wait(pending,timeout=30,return_when=concurrent.futures.FIRST_COMPLETED)
            if not finished: print(f'Waiting stage={stage} done={done}/{len(items)} failed={len(failures)} elapsed={time.monotonic()-start:.0f}s',flush=True)
            for future in finished:
                item=pending.pop(future);ident=item.get('row_id',item.get('pair_id'))
                try: future.result();done+=1;print(f'Completed {stage} {done}/{len(items)} id={ident}',flush=True)
                except Exception as e:
                    failures.append({'id':ident,'error':str(e)});print(f'Failed {stage} id={ident} {e}',flush=True)
    atomic(a.out/f'{stage}_status.json',{'planned':len(items),'completed':done,'failures':failures})
    if failures: raise ValueError(f'{stage}: {len(failures)} pending; successful per-call caches saved. Resume identical command.')

def build_kb(a,manifest,probe=False):
    calls=Calls(a,manifest);pool=load(a.out/'training.json')
    if probe:
        report=load(a.out/'preflight.json')
        worst=max((r for r in report['requests'] if r['stage']=='kb_analysis'),key=lambda r:r['estimated_total_tokens'])['item']
        pool=[p for p in pool if p['pair_id']==worst]
        print('Capacity probe reuses official KB calls for '+worst+'; no separate pilot dataset',flush=True)
    def work(p):
        out=a.out/'kb_items'/f"{p['pair_id']}.json"
        if out.exists(): return
        pp,fp,ap,kp=train_prompts(p);base='kb/'+p['pair_id']
        purpose=calls.call(base+'/purpose',message(pp),lambda x:prefix(x,'Function purpose:'))['content'];func=calls.call(base+'/function',message(fp),lambda x:prefix(x,'The functions of the code snippet are:'))['content']
        analysis=calls.call(base+'/analysis',message(ap))['content']
        knowledge=calls.call(base+'/knowledge',message(ap)+[{'role':'assistant','content':analysis},{'role':'user','content':kp}],knowledge_json)['content']
        obj=knowledge_json(knowledge);b=obj['vulnerability_behavior'];v,f=p['members']
        atomic(out,{'id':p['pair_id'],'CVE_id':v['cve'],'code_before_change':v['func'],'code_after_change':f['func'],'modified_lines':modified(v['func'],f['func']),'purpose':prefix(purpose,'Function purpose:'),'function':prefix(func,'The functions of the code snippet are:'),'analysis':analysis,**b,'solution':obj.get('solution',b.get('solution'))})
    parallel(a,pool,work,'capacity_probe' if probe else 'kb')
    if probe:atomic(a.out/'capacity_probe_complete.json',{'fingerprint':manifest['fingerprint'],'pair_id':pool[0]['pair_id'],'note':'Four official extraction requests completed; later dynamically generated requests remain checked individually.'})
    else:atomic(a.out/'knowledge.json',[load(a.out/'kb_items'/f"{p['pair_id']}.json") for p in pool])

def queries(a,manifest):
    calls=Calls(a,manifest);targets=load(a.out/'targets.json')
    def work(t):
        out=a.out/'query_items'/f"{t['row_id']}.json"
        if out.exists():return
        pp,fp=generate_extraction_prompt_for_vulrag(t['func']);base=f"query/{t['row_id']}"
        purpose=calls.call(base+'/purpose',message(pp),lambda x:prefix(x,'Function purpose:'))['content'];function=calls.call(base+'/function',message(fp),lambda x:prefix(x,'The functions of the code snippet are:'))['content']
        atomic(out,{'row_id':t['row_id'],'purpose':prefix(purpose,'Function purpose:'),'function':prefix(function,'The functions of the code snippet are:')})
    parallel(a,targets,work,'query');atomic(a.out/'queries.json',[load(a.out/'query_items'/f"{t['row_id']}.json") for t in targets])

def evidence(k):
    return {'cve_id':k['CVE_id'],'vulnerability_behavior':{f:k[f] for f in ('vulnerability_cause_description','trigger_condition','specific_code_behavior_causing_vulnerability')},'solution_behavior':k['solution']}
def retrieve(a):
    from retrieval import Retriever
    kb=load(a.out/'knowledge.json');qs={x['row_id']:x for x in load(a.out/'queries.json')};targets=load(a.out/'targets.json')
    if len(kb)!=len(load(a.out/'training.json')) or len(qs)!=len(targets):raise ValueError('Incomplete knowledge or query export')
    index=Retriever(kb);rows=[];requests=[]
    for n,t in enumerate(targets,1):
        q=qs[t['row_id']];selected,trace=index.search(t['func'],q['purpose'],q['function'])
        if not selected:raise ValueError('Empty retrieval; target cannot silently become NO')
        rows.append({'row_id':t['row_id'],'knowledge_ids':[k['id'] for k in selected],'evidence':[evidence(k) for k in selected],'trace':trace})
        for k in selected:
            for phase,fn in [('cause',generate_detect_vul_prompt_with_response_in_HTML),('solution',generate_detect_sol_prompt_with_response_in_HTML)]:requests.append({'stage':phase,'row_id':t['row_id'],'candidate_id':k['id'],'messages':message(fn(t['func'],evidence(k)))})
        if n%50==0: print(f'Retrieved {n}/{len(targets)}',flush=True)
    audit_requests(requests,a.model,a.max_output,a.out/'detection_context_audit.json',temperature=a.temperature)
    atomic(a.out/'retrieved.json',rows)
    atomic(a.out/'retrieval_manifest.json',{'knowledge_sha256':sha(a.out/'knowledge.json'),'queries_sha256':sha(a.out/'queries.json'),'retrieved_sha256':sha(a.out/'retrieved.json'),'targets_sha256':sha(a.out/'targets.json')})

def detect_one(t,retrieved,calls):
    trace=[]
    if not retrieved['evidence']:raise ValueError('Empty retrieval')
    final=-1
    for k,e in zip(retrieved['knowledge_ids'],retrieved['evidence']):
        base=f"detect/{t['row_id']}/{k}"
        v=calls.call(base+'/cause',message(generate_detect_vul_prompt_with_response_in_HTML(t['func'],e)),label)
        s=calls.call(base+'/solution',message(generate_detect_sol_prompt_with_response_in_HTML(t['func'],e)),label)
        vl,sl=label(v['content']),label(s['content']);trace.append({'knowledge_id':k,'cause':vl,'solution':sl,'call_ids':[v['call_id'],s['call_id']]})
        decided=decision(vl,sl)
        if decided is not None:final=decided;break
    # Upstream evaluator treats exhausted -1 as non-vulnerable. API errors never reach here.
    return {'row_id':t['row_id'],'pair_id':t['pair_id'],'idx':t['idx'],'target':t['target'],'cluster':t['cluster'],'raw_final_result':final,'prediction':int(final==1),'decision_status':'exhausted_no_match' if final==-1 else 'early_decision','trace':trace}

def detect(a,manifest):
    for key,name in [('knowledge','knowledge'),('queries','queries'),('retrieved','retrieved'),('targets','targets')]:
        if load(a.out/'retrieval_manifest.json')[key+'_sha256']!=sha(a.out/(name+'.json')): raise ValueError('Retrieval snapshot changed')
    calls=Calls(a,manifest);rs={r['row_id']:r for r in load(a.out/'retrieved.json')};targets=load(a.out/'targets.json')
    def work(t):
        p=a.out/'prediction_items'/f"{t['row_id']}.json"
        if not p.exists():atomic(p,detect_one(t,rs[t['row_id']],calls))
    parallel(a,targets,work,'detect');score(a)

def score(a):
    targets=load(a.out/'targets.json');results={}
    for t in targets:
        p=a.out/'prediction_items'/f"{t['row_id']}.json"
        if p.exists():results[t['row_id']]=load(p)
    tp=fp=tn=fn=0;outcomes=collections.Counter();groups=collections.defaultdict(list)
    for t in targets:
        groups[t['pair_id']].append(t)
        r=results.get(t['row_id'])
        if r:
            y,p=t['target'],r['prediction'];tp+=y==1 and p==1;fp+=y==0 and p==1;tn+=y==0 and p==0;fn+=y==1 and p==0
    for members in groups.values():
        v,f=sorted(members,key=lambda t:-t['target']);vr,fr=results.get(v['row_id']),results.get(f['row_id'])
        outcomes['unknown' if vr is None or fr is None else {(1,0):'P-C',(1,1):'P-V',(0,0):'P-B',(0,1):'P-R'}[vr['prediction'],fr['prediction']]]+=1
    n=len(results);metrics={'planned':len(targets),'completed':n,'complete':n==len(targets),'confusion_known':dict(TP=tp,FP=fp,TN=tn,FN=fn),'F1_known_only':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,'YES_rate_known':(tp+fp)/n if n else None,'pair_outcomes':dict(outcomes),'P-C_all_planned':outcomes['P-C']/len(groups),'exhausted_no_match':sum(r['raw_final_result']==-1 for r in results.values()),'note':'Positive-class F1 for our paired benchmark; differs from upstream averaged-precision/recall F1. Exhausted -1 mapped to NO as in official evaluator; failed API/parser jobs remain unknown. Incomplete known-only rates are not comparable.'}
    atomic(a.out/'metrics.json',metrics);atomic(a.out/'predictions.json',[results[i] for i in sorted(results)]);print(json.dumps(metrics,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','capacity-probe','kb','query','retrieve','detect','score','all'])
    p.add_argument('--model',default='gpt-5.5',choices=['glm-5.1','gpt-5.5','claude-opus-4-7']);p.add_argument('--provider',choices=PROVIDERS,default='newapi')
    p.add_argument('--source',type=Path,default=Path('outdir/knowledge_rag_glm-5.1'));p.add_argument('--train',type=Path,default=Path('data/primevul_train_paired_labeled.jsonl'));p.add_argument('--test',type=Path,default=Path('data/primevul_test_paired_labeled.jsonl'));p.add_argument('--out',type=Path,default=Path('outdir/vulrag_gpt-5.5_first_run'))
    p.add_argument('--workers',type=int,default=2);p.add_argument('--max-output',type=int,default=16384);p.add_argument('--temperature',type=float,default=.01);p.add_argument('--read-timeout',type=int,default=900);p.add_argument('--max-attempts',type=int,default=3);p.add_argument('--max-calls',type=int,default=25000)
    a=p.parse_args()
    if min(a.workers,a.max_output,a.max_calls,a.read_timeout,a.max_attempts)<1:p.error('Numeric limits must be positive')
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'.run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('This output directory is already running')
        stages=['prepare','capacity-probe','kb','query','retrieve','detect','score'] if a.stage=='all' else [a.stage]
        for stage in stages:
            print('=== '+stage+' ===',flush=True)
            if stage=='prepare':prepare(a);continue
            manifest=load(a.out/'manifest.json')
            if manifest['protocol']!=protocol(a):raise ValueError('Protocol/code changed: use new output or restore original configuration')
            if digest(load(a.out/'training.json'))!=manifest['pool_sha256'] or digest(load(a.out/'targets.json'))!=manifest['targets_sha256']:raise ValueError('Frozen input was modified')
            if stage not in ('score',):
                ready=load(a.out/'prepare_complete.json')
                if ready['fingerprint']!=manifest['fingerprint'] or ready['preflight_sha256']!=sha(a.out/'preflight.json') or load(a.out/'preflight.json')['needs_capacity_review']:
                    raise ValueError('Offline prepare must pass before paid stages')
            if stage=='capacity-probe':build_kb(a,manifest,probe=True)
            elif stage=='kb':
                if load(a.out/'capacity_probe_complete.json')['fingerprint']!=manifest['fingerprint']:raise ValueError('Complete capacity-probe before bulk KB extraction')
                build_kb(a,manifest)
            elif stage=='query':queries(a,manifest)
            elif stage=='retrieve':retrieve(a)
            elif stage=='detect':detect(a,manifest)
            elif stage=='score':score(a)
if __name__=='__main__':main()

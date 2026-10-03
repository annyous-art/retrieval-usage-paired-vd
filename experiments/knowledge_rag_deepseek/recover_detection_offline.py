#!/usr/bin/env python3
"""Recover explicit final labels offline; preserve originals and existing experiment files."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace as NS
import experiment as exp


def label(line):
    s=line.strip()
    for wrapper in ('**','__','`'):
        if s.startswith(wrapper) and s.endswith(wrapper) and len(s)>2*len(wrapper):
            s=s[len(wrapper):-len(wrapper)].strip()
    m=re.fullmatch(r'(?:\(([12])\)\s*)?(YES|NO)(?::\s*(A security vulnerability detected\.|No security vulnerability\.))?',s,re.I)
    if not m:return None
    n,answer,description=m.groups()
    result=int(answer.upper()=='YES')
    if n and int(n)!=(1 if result else 2):return None
    if description and description.lower()!=('a security vulnerability detected.' if result else 'no security vulnerability.'):return None
    return result


def final_label(text):
    if not isinstance(text,str):return None
    lines=[s.strip() for s in text.splitlines() if s.strip()]
    if not lines:return None
    result=label(lines[-1])
    if result is None:return None
    for line in lines[:-1]:
        prior=label(line)
        if prior is None and re.fullmatch(r'\(?[12]\)?',line):prior=exp.prediction(line)
        if prior is not None and prior!=result:return None
    return result


def recover(directory, out):
    d=Path(directory);out=Path(out)
    if out.exists():raise ValueError('Use a new --out directory; originals are never overwritten')
    jobs=exp.read(d/'detection_jobs.jsonl');expected={j['job_id']:j for j in jobs}
    if len(expected)!=len(jobs):raise ValueError('Duplicate planned IDs')
    success=exp.read(d/'predictions.jsonl');errors=exp.read(d/'predictions.jsonl.errors.jsonl')
    configs={exp.digest(r['config']) for r in success}
    if len(configs)!=1:raise ValueError('Expected one successful inference configuration')
    accepted={};recovered=[];unresolved=[]
    for is_error,records in [(False,success),(True,errors)]:
        for r in records:
            jid=r['job_id']
            if jid not in expected or r['messages']!=expected[jid]['messages']:raise ValueError('Job/prompt mismatch')
            if jid in accepted:
                if not is_error:raise ValueError('Duplicate success')
                continue
            if exp.digest(r['config']) not in configs:raise ValueError('Inference configuration mismatch')
            if r.get('sent_messages')!=r['messages']:
                if is_error:continue
                raise ValueError('Sent prompt mismatch')
            raw=r.get('response');old=exp.prediction(raw);new=final_label(raw)
            if not is_error:
                if old is None:raise ValueError('Previously successful record is unparseable')
                if new is not None and old!=new:raise ValueError('New rule disagrees with prior prediction')
                accepted[jid]=r
                continue
            if r.get('error')!='Unparseable detection response' or r.get('provider_metadata',{}).get('generation_truncated') or new is None:
                continue
            recovered.append(jid)
            copy={k:v for k,v in r.items() if k not in ('error','error_type','attempts')}
            copy.update(response='YES' if new else 'NO',original_response=raw,
                        recovery={'method':'explicit final line, optional Markdown wrapper/number; reject conflicting standalone labels',
                                  'original_error':r['error'],'source_file':'predictions.jsonl.errors.jsonl'})
            accepted[jid]=copy
    unresolved=sorted(set(expected)-set(accepted))
    out.mkdir(parents=True)
    output=out/'predictions_recovered.jsonl'
    exp.write(output,[accepted[j['job_id']] for j in jobs if j['job_id'] in accepted])
    report={'planned':len(jobs),'original_success':len(success),'recovered':len(recovered),
            'recovered_job_ids':recovered,'remaining':len(unresolved),'remaining_job_ids':unresolved,
            'api_calls':0,'original_files_modified':False,
            'source_sha256':{name:hashlib.sha256((d/name).read_bytes()).hexdigest() for name in ['detection_jobs.jsonl','predictions.jsonl','predictions.jsonl.errors.jsonl']}}
    (out/'recovery_report.json').write_text(json.dumps(report,indent=2))
    exp.score(NS(jobs=d/'detection_jobs.jsonl',responses=output,output=out/'metrics_recovered.json',bootstrap=2000,seed=20260915))
    print(json.dumps({k:v for k,v in report.items() if k not in ('recovered_job_ids','remaining_job_ids','source_sha256')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();recover(a.directory,a.out)

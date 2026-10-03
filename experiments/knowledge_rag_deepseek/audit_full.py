#!/usr/bin/env python3
"""Offline full corpus/request audit. Never loads API keys or calls a model."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import experiment as exp
from context_budget import audit_requests, text_counts, policy_document


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',type=Path,required=True)
    p.add_argument('--test',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True,help='New output directory')
    p.add_argument('--models',nargs='+',default=list(policy_document()['models']))
    a=p.parse_args()
    if a.out.exists():
        p.error('Use a new --out directory to preserve prior audits')
    exp.prepare(NS(train=a.train,test=a.test,out=a.out,pairs=0,k=1,seed=20260915,test_pair_policy='adjacent'))
    pool=exp.read(a.out/'candidate_pool.jsonl')
    targets=exp.read(a.out/'targets.jsonl')
    selected={j['candidate_id'] for j in exp.read(a.out/'extraction_jobs.jsonl')}
    # Inventory every raw target and every eligible candidate member, selected or not.
    inventory=[]
    for row in targets:
        inventory.append(dict(kind='target',row_id=row['row_id'],bytes=len(row['func'].encode()),tokens=text_counts(row['func'])))
    for pair in pool:
        for row in pair['members']:
            inventory.append(dict(kind='candidate',candidate_id=pair['pair_id'],row_id=row['row_id'],selected=pair['pair_id'] in selected,
                                  bytes=len(row['func'].encode()),tokens=text_counts(row['func'])))
    exp.write(a.out/'code_inventory.jsonl',inventory)
    candidates={j['pair_id']:j for j in pool}
    base=[]
    for target in targets:
        for arm,context in [('no_retrieval','(none)'),('code_pair',exp.code_evidence(target,candidates))]:
            base.append(dict(row_id=target['row_id'],arm=arm,candidate_ids=target['candidate_ids'],
                             evidence_bytes=len(context.encode()),target_bytes=len(target['func'].encode()),
                             messages=exp.target_messages(target['func'],context)))
    base.extend(dict(candidate_id=pair['pair_id'],stage='extract_selected' if pair['pair_id'] in selected else 'extract_unselected',
                     messages=exp.extraction_messages(pair)) for pair in pool)
    summary={'data_hashes':{k:hashlib.sha256(path.read_bytes()).hexdigest() for k,path in [('train',a.train),('test',a.test)]},
             'targets':len(targets),'eligible_candidate_pairs':len(pool),'selected_candidate_pairs':len(selected),
             'api_calls':0,'pending_arms':['knowledge','knowledge_no_fix'],'models':{}}
    for model in a.models:
        print('Scanning '+model,flush=True)
        r=audit_requests(base,model,4096,a.out/(model+'.json'),enforce=False)
        summary['models'][model]=r['groups']
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()

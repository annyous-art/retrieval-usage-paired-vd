#!/usr/bin/env python3
"""Offline paired comparisons and blinded review packets. No API calls."""
import argparse, collections, hashlib, itertools, json, random, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import experiment as exp

def comparisons(jobs, responses, seed=20260915, draws=2000):
    obs={r['job_id']:r for r in responses}
    if len(obs)!=len(responses) or set(obs)!={j['job_id'] for j in jobs}:raise ValueError('Need complete unique predictions')
    groups=collections.defaultdict(lambda:collections.defaultdict(dict))
    for j in jobs:
        r=obs[j['job_id']]
        if r['messages']!=j['messages']:raise ValueError('Prompt mismatch')
        pred=exp.prediction(r.get('response'))
        if pred is None:raise ValueError('Unclassified response')
        groups[j['arm']][j['pair_id']][j['target']]=(pred,j['cluster'])
    states={}
    for arm,pairs in groups.items():
        states[arm]={}
        for pid,x in pairs.items():
            if set(x)!={0,1} or x[0][1]!=x[1][1]:raise ValueError('Invalid pair/cluster')
            states[arm][pid]=(int(x[0][0]==0 and x[1][0]==1),x[0][1])
    result={}
    for left,right in itertools.combinations(sorted(states),2):
        if set(states[left])!=set(states[right]):raise ValueError('Pair mismatch')
        blocks=collections.defaultdict(list)
        for pid,(v,c) in states[left].items():blocks[c].append(v-states[right][pid][0])
        bs=list(blocks.values());rng=random.Random(seed);samples=[]
        for _ in range(draws):
            vs=[v for b in rng.choices(bs,k=len(bs)) for v in b];samples.append(sum(vs)/len(vs))
        samples.sort();values=[v for b in bs for v in b]
        result[left+' minus '+right]={'delta_pc':sum(values)/len(values),'cluster_bootstrap_95ci':[samples[int(.025*draws)],samples[int(.975*draws)]],'pairs':len(values),'clusters':len(bs)}
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--predictions',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--sample',type=int,default=50);a=p.parse_args()
    if a.out.exists():p.error('Use a new output directory')
    jobs=exp.read(a.source/'detection_jobs.jsonl');pred=a.predictions or a.source/'predictions.jsonl'
    stats=comparisons(jobs,exp.read(pred))
    targets=exp.read(a.source/'targets.jsonl');candidates={r['pair_id']:r for r in exp.read(a.source/'candidates.jsonl')}
    ej={r['job_id']:r['candidate_id'] for r in exp.read(a.source/'extraction_jobs.jsonl')}
    knowledge={ej[r['job_id']]:exp.parse_knowledge(r['response']) for r in exp.read(a.source/'knowledge_responses.jsonl')}
    rng=random.Random(20260915);rows=rng.sample(sorted(targets,key=lambda r:r['row_id']),min(a.sample,len(targets)))
    a.out.mkdir(parents=True);packet=[];key=[]
    for i,t in enumerate(rows):
        case=f'case_{i+1:03d}'
        packet.append({'case_id':case,'target_code':t['func'],'historical_evidence':[{'vulnerable_code':candidates[c]['members'][0]['func'],'repaired_code':candidates[c]['members'][1]['func'],'knowledge':knowledge[c]} for c in t['candidate_ids']], 'review':{'root_cause_supported':'','repair_condition_supported':'','critical_condition_omitted':'','unsupported_claim':'','repair_info_in_root_cause':'','mechanism_relevance':'','evidence_and_comments':''}})
        key.append({'case_id':case,'row_id':t['row_id'],'candidate_ids':t['candidate_ids']})
    exp.write(a.out/'blind_review.jsonl',packet);exp.write(a.out/'private_mapping_do_not_give_reviewers.jsonl',key)
    (a.out/'statistics.json').write_text(json.dumps(stats,indent=2))
    (a.out/'README.txt').write_text('No API calls. Reviewers receive blind_review.jsonl only, not model predictions or private mapping. Target labels are omitted. Historical vulnerable/repaired roles are necessary for auditing extraction correctness. Use independent reviewers, yes/no/uncertain and supporting code evidence. CIs are exploratory and unadjusted for multiple comparisons. Review sample selected independently of predictions; do not tune on the held-out test set.\n')
    print('Offline comparisons and blinded packets written:',a.out)
if __name__=='__main__':main()

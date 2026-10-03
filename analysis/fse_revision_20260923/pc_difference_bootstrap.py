import json, sys, csv, importlib.util, numpy as np, pandas as pd
sys.path.insert(0,'.'); sys.path.insert(0,'experiments/paper_reliability')
import reliability as R
from evaluate_prompt_outputs import parse_prediction
raw=[json.loads(l) for l in open('data/primevul_test_paired_labeled.jsonl')]
df=pd.read_csv('outdir/analysis/per_sample_feature_table.csv')
y=np.array([int(r['target']) for r in raw])
def from_file(path):
    recs=[]
    if path.endswith('.csv'):
        for r in csv.DictReader(open(path)): recs.append((str(r['idx']), R_int(r['prediction'])))
    else:
        for l in open(path):
            r=json.loads(l); recs.append((str(r.get('query_idx',r.get('idx'))), parse_prediction(r.get('response'))))
    if len(recs)==len(raw) and all(a==str(b['idx']) for (a,_),b in zip(recs,raw)):
        return np.array([p if p is not None else -1 for _,p in recs]),'position'
    m={}
    for i,p in recs: m.setdefault(i,p)
    return np.array([m.get(str(r['idx'])) if m.get(str(r['idx'])) is not None else -1 for r in raw]),'idx'
def R_int(v):
    try: return int(float(v))
    except: return None
src={
 'yes_only':df['yes_only_pred'].to_numpy(),'zero_shot':df['zero_shot_pred'].to_numpy(),
}
for name,path in [('glm_funcicl','outdir/icl_fixed_baseline_compatible_test870/eval/glm51_icl_fixed_predictions.csv'),
 ('gpt_fixed','data/gpt-5.5_std_cls_logprobsFalse_fewshotegTrue1_baseline_870.jsonl'),
 ('gpt_rag','data/gpt-5.5_std_cls_fewshotegFalse1_RAG_870.jsonl')]:
    p,how=from_file(path); src[name]=p; print(name,how,'unknown',int((p<0).sum()))
gr=R.groups(raw,'commit')[::2]
vec={k:R.pair_vectors(y,np.asarray(v,dtype=float)) for k,v in src.items()}
for k,(v,m) in vec.items(): print(k,'P-C',round(v[:,4].sum()/len(v),4),'F1',round(float(R.metrics(v.sum(0)[None])[0][0]),4))
def cmp(a,b):
    lv,lm=vec[a];rv,rm=vec[b];mask=lm&rm
    labels=sorted(set(gr[mask]));l=np.array([lv[mask&(gr==g)].sum(0) for g in labels]);r=np.array([rv[mask&(gr==g)].sum(0) for g in labels])
    w=np.random.default_rng(20260920).multinomial(len(labels),np.ones(len(labels))/len(labels),size=3000)
    lf,lp=R.metrics(w@l);rf,rp=R.metrics(w@r);pf,pc=R.metrics(l.sum(0));qf,qc=R.metrics(r.sum(0))
    print(f'{a} - {b}: pairs={mask.sum()} clusters={len(labels)} dF1={100*(pf-qf):+.2f} {np.round(100*np.quantile(lf-rf,[.025,.975]),2)} dPC={100*(pc-qc):+.2f} {np.round(100*np.quantile(lp-rp,[.025,.975]),2)}')
cmp('yes_only','zero_shot')
cmp('glm_funcicl','yes_only'); cmp('glm_funcicl','zero_shot'); cmp('gpt_rag','gpt_fixed')

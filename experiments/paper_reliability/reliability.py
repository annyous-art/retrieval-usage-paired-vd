#!/usr/bin/env python3
"""Offline benchmark statistics and grouped nested-CV routers. No API calls."""
import argparse, collections, hashlib, importlib.util, itertools, json, re
from pathlib import Path
import numpy as np
import pandas as pd


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def load(root,features=None):
    data=root/'data/primevul_test_paired_labeled.jsonl';table=features or root/'outdir/analysis/per_sample_feature_table.csv'
    raw=[json.loads(l) for l in data.open()];df=pd.read_csv(table)
    if len(df)!=len(raw) or len(raw)%2:raise ValueError('Benchmark/table row counts differ')
    for i,(r,(_,f)) in enumerate(zip(raw,df.iterrows())):
        if str(r['idx'])!=str(f['idx']) or int(r['target'])!=int(f['target']) or str(r['project'])!=str(f['project']) or str(r['commit_id'])!=str(f['commit_id']):
            raise ValueError(f'Row alignment differs at {i}; no implicit idx-only join permitted')
    for i in range(0,len(raw),2):
        if {int(raw[i]['target']),int(raw[i+1]['target'])}!={0,1}:raise ValueError('Invalid adjacent pair')
    return raw,df,{'test':str(data),'test_sha256':sha(data),'features':str(table),'features_sha256':sha(table),'rows':len(raw),'pairs':len(raw)//2,'duplicate_idx':{str(k):v for k,v in collections.Counter(r['idx'] for r in raw).items() if v>1},'alignment':'position + idx + target + project + commit verified; duplicates not collapsed'}

def groups(raw,mode):
    n=len(raw)//2;parents=list(range(n))
    def find(a):
        while parents[a]!=a:parents[a]=parents[parents[a]];a=parents[a]
        return a
    def join(a,b):parents[find(a)]=find(b)
    seen={}
    for i,r in enumerate(raw):
        keys=[]
        # Always keep duplicated indices and identical code with their pairs.
        keys += [('idx',str(r['idx'])),('code',hashlib.sha256(r['func'].encode()).hexdigest())]
        if mode=='commit' and r.get('commit_id'):keys.append(('commit',str(r.get('project')),str(r['commit_id'])))
        if mode=='project' and r.get('project'):keys.append(('project',str(r['project'])))
        for key in keys:
            if key in seen:join(i//2,seen[key])
            else:seen[key]=i//2
    return np.array([find(i//2) for i in range(len(raw))])

def pair_vectors(y,p):
    vals=[];valid=[]
    for i in range(0,len(y),2):
        yp=y[i:i+2];pp=p[i:i+2];ok=np.isin(pp,[0,1]).all();valid.append(ok)
        vals.append([sum((yp==1)&(pp==1)),sum((yp==0)&(pp==1)),sum((yp==1)&(pp==0)),sum((yp==0)&(pp==0)),int(ok and np.array_equal(yp,pp)),1])
    return np.asarray(vals,dtype=float),np.array(valid)

def metrics(v):
    tp,fp,fn,tn,pc,n=np.asarray(v).T
    den=2*tp+fp+fn
    return np.divide(2*tp,den,out=np.zeros_like(den,dtype=float),where=den!=0),pc/n

def stats(raw,df,draws,seed):
    y=df.target.to_numpy();cols=[c for c in df if c.endswith('_pred')];vectors={};summary={}
    for col in cols:
        p=pd.to_numeric(df[col],errors='coerce').to_numpy();v,mask=pair_vectors(y,p);vectors[col]=(v,mask)
        known=np.isin(p,[0,1]);tp=int(sum((y==1)&(p==1)));fp=int(sum((y==0)&(p==1)));fn=int(sum((y==1)&(p==0)));tn=int(sum((y==0)&(p==0)))
        summary[col]={'known_rows':int(known.sum()),'unknown_rows':int((~known).sum()),'known_pairs':int(mask.sum()),'unknown_pairs':int((~mask).sum()),'TP':tp,'FP':fp,'FN':fn,'TN':tn,'F1_known_only':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,'PC_all_planned':float(v[:,4].sum()/len(v))}
    results={}
    for mode in ['pair','commit','project']:
        gr=groups(raw,mode)[::2];res={}
        for left,right in itertools.combinations(cols,2):
            lv,lm=vectors[left];rv,rm=vectors[right];mask=lm&rm
            if not mask.any():continue
            labels=sorted(set(gr[mask]));l=np.array([lv[mask & (gr==g)].sum(0) for g in labels]);r=np.array([rv[mask & (gr==g)].sum(0) for g in labels])
            # Exactly the same resampled clusters and denominators for both methods.
            weights=np.random.default_rng(seed).multinomial(len(labels),np.ones(len(labels))/len(labels),size=draws)
            lf,lp=metrics(weights@l);rf,rp=metrics(weights@r);pf,pc=metrics(l.sum(0));qf,qc=metrics(r.sum(0))
            res[left+' minus '+right]={'rows':int(mask.sum()*2),'pairs':int(mask.sum()),'excluded_pair_ids':np.flatnonzero(~mask).tolist(),'clusters':len(labels),'delta_F1':float(pf-qf),'delta_PC':float(pc-qc),'F1_95CI':np.quantile(lf-rf,[.025,.975]).tolist(),'PC_95CI':np.quantile(lp-rp,[.025,.975]).tolist()}
        results[mode]=res
    return {'methods':summary,'comparisons':results,'note':'Exploratory unadjusted intervals. Comparisons use explicitly reported common COMPLETE PAIRS. All benchmark rows remain in coverage summaries. Pair grouping additionally links duplicate idx/exact code; commit/project groups use transitive closure, keeping all pair members together.'}

def router(root,raw,df,a,out):
    import sklearn
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import f1_score
    source=a.router_source or root/'train_router.py'
    spec=importlib.util.spec_from_file_location('original_router',source);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    def make_model(seed):
        model=base.build_model(a.model,num,cat,seed)
        if a.model=='random_forest':model.set_params(model__n_jobs=a.jobs)
        return model
    # Preserve original cleaning/model definitions; verify no rows were silently discarded.
    clean=base.coerce_dataframe(a.features or root/'outdir/analysis/per_sample_feature_table.csv').reset_index(drop=True)
    if len(clean)!=len(df):raise ValueError('Router coercion dropped rows')
    features=base.select_features(clean,base.DEFAULT_FEATURES,True)
    removed=[]
    if a.feature_policy=='no-target-cwe':
        removed=[c for c in features if 'same_cwe' in c];features=[c for c in features if c not in removed]
    num,cat=base.split_feature_types(clean,features);y=clean.target.astype(int).to_numpy();g=groups(raw,a.group)
    if len(set(g))<a.folds:raise ValueError('Too few independent groups')
    outer=StratifiedGroupKFold(n_splits=a.folds,shuffle=True,random_state=a.seed);prob=np.full(len(y),np.nan);foldid=np.zeros(len(y),int);pred=np.full(len(y),-1);details=[]
    for fold,(tr,te) in enumerate(outer.split(clean,y,g),1):
        assert not set(g[tr])&set(g[te])
        threshold=.5;inner_trace=[]
        if a.threshold=='inner-f1':
            inner=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=a.seed+fold);ip=np.full(len(tr),np.nan)
            for it,iv in inner.split(clean.iloc[tr],y[tr],g[tr]):
                assert not set(g[tr[it]])&set(g[tr[iv]])
                m=make_model(a.seed+fold);m.fit(clean.iloc[tr[it]][features],y[tr[it]]);ip[iv]=m.predict_proba(clean.iloc[tr[iv]][features])[:,1]
                inner_trace.append({'train_rows':tr[it].tolist(),'validation_rows':tr[iv].tolist()})
            assert np.isfinite(ip).all()
            candidates=np.linspace(.05,.95,91);scores=[f1_score(y[tr],ip>=t,zero_division=0) for t in candidates]
            best=max(scores);threshold=float(min([t for t,s in zip(candidates,scores) if s==best],key=lambda t:(abs(t-.5),t)))
        m=make_model(a.seed+fold);m.fit(clean.iloc[tr][features],y[tr]);prob[te]=m.predict_proba(clean.iloc[te][features])[:,1];pred[te]=(prob[te]>=threshold).astype(int);foldid[te]=fold
        details.append({'fold':fold,'threshold':threshold,'train_rows':tr.tolist(),'test_rows':te.tolist(),'inner_folds':inner_trace,'train_groups':len(set(g[tr])),'test_groups':len(set(g[te]))})
        print(f'fold={fold}/{a.folds} threshold={threshold:.2f} test_rows={len(te)}',flush=True)
    assert np.isfinite(prob).all() and (foldid>0).all()
    vec,mask=pair_vectors(y,pred);f,pc=metrics(vec.sum(0))
    fixed=(prob>=.5).astype(int);fv,_=pair_vectors(y,fixed);ff,fp=metrics(fv.sum(0))
    yesv,_=pair_vectors(y,np.ones_like(y));yf,yp=metrics(yesv.sum(0))
    dump(out/'router_metrics.json',{'model':a.model,'group':a.group,'threshold_policy':a.threshold,'feature_policy':a.feature_policy,'features':features,'removed_extra_features':removed,'rows':len(y),'pairs':len(y)//2,'groups':len(set(g)),'F1':float(f),'PC':float(pc),'yes_rate':float(pred.mean()),'confusion':vec.sum(0)[:4].astype(int).tolist(),'confusion_order':['TP','FP','FN','TN'],'fixed_0_5_diagnostic':{'F1':float(ff),'PC':float(fp),'yes_rate':float(fixed.mean()),'confusion':fv.sum(0)[:4].astype(int).tolist()},'constant_yes_baseline':{'F1':float(yf),'PC':float(yp),'yes_rate':1.0},'sklearn_version':sklearn.__version__,'jobs':a.jobs,'note':'Diagnostic grouped OOF on existing benchmark, NOT independent test validation. Models/hyperparameters are inherited; thresholds use inner grouped OOF or fixed 0.5. All imputing/scaling/encoding fit within each training fold. Bootstrap comparisons condition on these saved OOF predictions and do not retrain models in each resample. Removing target-CWE fields does not independently audit upstream historical retrieval for target-label use.'})
    dump(out/'folds.json',details)
    with (out/'oof_predictions.jsonl').open('w') as f:
        for i in range(len(y)):f.write(json.dumps({'row_id':i,'pair_id':i//2,'group':int(g[i]),'idx':int(raw[i]['idx']),'target':int(y[i]),'fold':int(foldid[i]),'prediction':int(pred[i]),'fixed_0_5_prediction':int(fixed[i]),'probability':float(prob[i])})+'\n')
    clean[a.model+'_grouped_router_pred']=pred
    clean['constant_yes_pred']=1
    dump(out/'router_comparisons.json',stats(raw,clean,a.draws,a.seed))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--features',type=Path,help='Explicit feature table; default ROOT/outdir/analysis/per_sample_feature_table.csv');p.add_argument('--router-source',type=Path,help='Optional frozen original train_router.py');p.add_argument('--jobs',type=int,default=-1,help='Random forest CPU workers');p.add_argument('--out',type=Path,required=True);p.add_argument('--task',choices=['stats','router'],required=True);p.add_argument('--draws',type=int,default=3000);p.add_argument('--seed',type=int,default=20260920);p.add_argument('--model',choices=['logistic','random_forest'],default='logistic');p.add_argument('--group',choices=['commit','project'],default='commit');p.add_argument('--threshold',choices=['fixed','inner-f1'],default='inner-f1');p.add_argument('--feature-policy',choices=['original-realistic','no-target-cwe'],default='original-realistic');p.add_argument('--folds',type=int,default=5);a=p.parse_args()
    if a.out.exists():p.error('Use new --out directory; preserve previous outputs')
    if a.draws<100 or a.folds<2 or a.jobs==0:p.error('draws>=100, folds>=2, jobs!=0 required')
    raw,df,manifest=load(a.root,a.features);a.out.mkdir(parents=True)
    manifest.update(task=a.task,seed=a.seed,draws=a.draws,script_sha256=sha(Path(__file__)),router_source_sha256=sha(a.router_source or a.root/'train_router.py'),scope='Feature-table methods only; does not replace all manuscript tables or API logs',versions={'numpy':np.__version__,'pandas':pd.__version__})
    dump(a.out/'manifest.json',manifest)
    if a.task=='stats':
        result=stats(raw,df,a.draws,a.seed);dump(a.out/'statistics.json',result)
        manuscript=a.root/'FSE/sections/04_results.tex';check={}
        mapping={'Zero-shot':'zero_shot_pred','YES-only few-shot':'yes_only_pred','Author two-shot':'author_no_yes_pred','RAG + fixed few-shot':'old_rag_pred','Positive-weighted RAG':'positive_weighted_rag_pred'}
        if manuscript.exists():
            for line in manuscript.read_text().splitlines():
                cells=[c.strip() for c in line.split('&')]
                if len(cells)==11 and cells[0] in mapping and cells[1]=='870':
                    counts=[int(re.sub(r'[^0-9]','',c)) for c in cells[-4:]]
                    actual=result['methods'][mapping[cells[0]]]
                    observed=[actual[k] for k in ['TP','FP','TN','FN']]
                    check[cells[0]]={'column':mapping[cells[0]],'paper_counts_TP_FP_TN_FN':counts,'feature_table_counts':observed,'match':counts==observed,'known_rows':actual['known_rows'],'note':'This audits a candidate source; a mismatch requires locating the exact original run. Do not replace manuscript numbers automatically.'}
        dump(a.out/'paper_source_check.json',{'manuscript':str(manuscript),'rows':check,'unmapped_tables':'Not audited by this tool'})
        manual=a.root/'outdir/empirical_study/error_taxonomy/manual_samples'
        dump(a.out/'manual_annotation_inventory.json',{'directory':str(manual),'files':[str(f.relative_to(a.root)) for f in sorted(manual.glob('*')) if f.is_file()], 'note':'Inventory only. Cannot compute agreement from marginal counts 31/51 and 35/51; requires per-case independent labels from both reviewers.'})
    else:router(a.root,raw,df,a,a.out)
    print('Complete:',a.out)
if __name__=='__main__':main()

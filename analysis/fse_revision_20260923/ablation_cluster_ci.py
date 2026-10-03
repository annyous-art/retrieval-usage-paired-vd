import json, sys, csv, numpy as np
sys.path.insert(0,'.'); sys.path.insert(0,'experiments/paper_reliability')
import reliability as R
from evaluate_prompt_outputs import parse_prediction
raw=[json.loads(l) for l in open('data/primevul_test_paired_labeled.jsonl')]
y=np.array([int(r['target']) for r in raw]); gr=R.groups(raw,'commit')[::2]
# Keep the 431 resolvable pairs: drop pairs touching a test idx that occurs twice.
from collections import Counter
_dup=Counter(str(r['idx']) for r in raw)
keep=np.array([_dup[str(raw[i]['idx'])]==1 and _dup[str(raw[i+1]['idx'])]==1 for i in range(0,len(raw),2)])
keepf=np.repeat(keep,2)
def load(p):
    m={}
    if p.endswith('.csv'):
        for r in csv.DictReader(open(p)): m.setdefault(str(r['idx']),int(float(r['prediction'])) if r['prediction'] not in ('','None') else None)
    else:
        for l in open(p):
            r=json.loads(l); m.setdefault(str(r.get('query_idx',r.get('idx'))),parse_prediction(r.get('response')))
    return np.array([m.get(str(r['idx'])) if m.get(str(r['idx'])) is not None else -1 for r in raw],dtype=float)
def ci(p):
    v,mask=R.pair_vectors(y,p); mask=mask&keep; labels=sorted(set(gr[mask])); l=np.array([v[mask&(gr==g)].sum(0) for g in labels])
    w=np.random.default_rng(20260920).multinomial(len(labels),np.ones(len(labels))/len(labels),size=3000)
    f,pc=R.metrics(w@l); F,PC=R.metrics(l.sum(0)[None]); known=(p>=0)&keepf
    return dict(F1=float(F[0]),F1lo=float(np.quantile(f,.025)),F1hi=float(np.quantile(f,.975)),PC=float(PC[0]),PClo=float(np.quantile(pc,.025)),PChi=float(np.quantile(pc,.975)),vuln=float((p[known]==1).mean()),pairs=int(mask.sum()))
A='outdir/prompt_ablation_glm51_test870/glm-5.1_std_cls_%s_fewshotegFalse_cls0.jsonl'
C='outdir/prompt_fewshot_chunks_glm51_test870/glm-5.1_std_cls_%s_fewshotegTrue_cls0.jsonl'
B='outdir/prompt_baseline_compatible_glm51_test870/glm-5.1_std_cls_%s_baseline_compatible_fewshotegTrue_cls0.jsonl'
G='outdir/prompt_gpt55_representative_ablation_test870/gpt-5.5_std_cls_%s_cls0.jsonl'
rows=[('RAG','Target focus',A%'target_focus'),('RAG','No-label',A%'no_label'),('RAG','Grouped no-label',A%'grouped_no_label'),('RAG','Inline func. label',A%'inline_function_label'),('RAG','Inline chunk label',A%'inline_chunk_label'),('RAG','Grouped chunk label',A%'grouped_chunk_label'),
('Combined','Target focus',C%'target_focus'),('Combined','Grouped no-label',C%'grouped_no_label'),('Combined','Grouped chunk label',C%'grouped_chunk_label'),('Combined','Inline func. label',C%'inline_function_label'),('Combined','Inline chunk label',C%'inline_chunk_label'),
('BC','Target focus',B%'target_focus'),('BC','Grouped no-label',B%'grouped_no_label'),('BC','Inline func. label',B%'inline_function_label'),
('Positive-only BC','Inline func. label','outdir/prompt_positive_only_baseline_compatible_glm51_test870/glm-5.1_std_cls_inline_function_label_baseline_compatible_fewshotegTrue_cls0.jsonl'),
('Positive-weighted','Inline func. label','outdir/prompt_positive_weighted_rag_glm51_test870/glm-5.1_std_cls_inline_function_label_rag_fewshotegTrue_cls0.jsonl'),
('GPT','Fixed few-shot','data/gpt-5.5_std_cls_logprobsFalse_fewshotegTrue1_baseline_870.jsonl'),('GPT','Positive-only RAG','data/gpt-5.5_std_cls_fewshotegFalse1_RAG_870.jsonl'),
('GPT','Target focus',G%'target_focus_rag_fewshotegFalse'),('GPT','Grouped no-label',G%'grouped_no_label_rag_fewshotegFalse'),('GPT','BC grouped no-label',G%'grouped_no_label_baseline_compatible_fewshotegTrue')]
out=[]
for fam,ev,p in rows:
    pr=load(p); r=ci(pr); r.update(fam=fam,ev=ev,unknown=int((pr<0).sum())); out.append(r)
    print(f"{fam:18s} {ev:20s} F1 {r['F1']:.3f} [{r['F1lo']:.3f},{r['F1hi']:.3f}] vuln {r['vuln']:.3f} PC {r['PC']:.3f} [{r['PClo']:.3f},{r['PChi']:.3f}] pairs {r['pairs']} unk {r['unknown']}")
json.dump(out,open('analysis/fse_revision_20260923/ablation_cluster_ci.json','w'),indent=1)

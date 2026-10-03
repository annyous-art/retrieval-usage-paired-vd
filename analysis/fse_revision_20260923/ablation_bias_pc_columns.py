import json, csv, collections, sys
sys.path.insert(0,'.')
from evaluate_prompt_outputs import parse_prediction
raw=[json.loads(l) for l in open('data/primevul_test_paired_labeled.jsonl')]
def load(path):
    m={}
    if path.endswith('.csv'):
        for r in csv.DictReader(open(path)):
            try: m.setdefault(str(r['idx']),int(float(r['prediction'])))
            except: m.setdefault(str(r['idx']),None)
    else:
        for l in open(path):
            r=json.loads(l); m.setdefault(str(r.get('query_idx',r.get('idx'))),parse_prediction(r.get('response')))
    return m
def stats(m):
    tp=fp=fn=tn=unk=0; pc=pv=pb=pr=known=0
    for r in raw:
        p=m.get(str(r['idx'])); y=int(r['target'])
        if p is None: unk+=1; continue
        tp+=y==1 and p==1; fp+=y==0 and p==1; fn+=y==1 and p==0; tn+=y==0 and p==0
    for i in range(0,len(raw),2):
        a,b=raw[i],raw[i+1]
        v,f=(a,b) if int(a['target'])==1 else (b,a)
        pvn,pfn=m.get(str(v['idx'])),m.get(str(f['idx']))
        if pvn is None or pfn is None: continue
        known+=1; pc+=pvn==1 and pfn==0; pv+=pvn==1 and pfn==1; pb+=pvn==0 and pfn==0; pr+=pvn==0 and pfn==1
    n=tp+fp+fn+tn
    return dict(F1=round(2*tp/(2*tp+fp+fn),3) if tp else 0.0, vuln=round((tp+fp)/n,3), unk=unk, pairs=known, PC=round(pc/435,3), PV=round(pv/435,3), PB=round(pb/435,3))
A='outdir/prompt_ablation_glm51_test870/glm-5.1_std_cls_%s_fewshotegFalse_cls0.jsonl'
C='outdir/prompt_fewshot_chunks_glm51_test870/glm-5.1_std_cls_%s_fewshotegTrue_cls0.jsonl'
B='outdir/prompt_baseline_compatible_glm51_test870/glm-5.1_std_cls_%s_baseline_compatible_fewshotegTrue_cls0.jsonl'
rows=[('RAG target focus',A%'target_focus'),('RAG no-label',A%'no_label'),('RAG grouped no-label',A%'grouped_no_label'),('RAG inline func',A%'inline_function_label'),('RAG inline chunk',A%'inline_chunk_label'),('RAG grouped chunk',A%'grouped_chunk_label'),
('Comb target focus',C%'target_focus'),('Comb grouped no-label',C%'grouped_no_label'),('Comb grouped chunk',C%'grouped_chunk_label'),('Comb inline func',C%'inline_function_label'),('Comb inline chunk',C%'inline_chunk_label'),
('BC target focus',B%'target_focus'),('BC grouped no-label',B%'grouped_no_label'),('BC inline func',B%'inline_function_label'),
('PosOnly BC inline func','outdir/prompt_positive_only_baseline_compatible_glm51_test870/glm-5.1_std_cls_inline_function_label_baseline_compatible_fewshotegTrue_cls0.jsonl'),
('PosWeighted inline func','outdir/prompt_positive_weighted_rag_glm51_test870/glm-5.1_std_cls_inline_function_label_rag_fewshotegTrue_cls0.jsonl')]
E='outdir/empirical_study/'
rows+=[('--bias--',None),('No-code prior',E+'bias_controlled_glm51_test870/glm-5.1_no_code_prior_predictions.csv'),('No-code+author',E+'bias_controlled_glm51_test870_extra/glm-5.1_no_code_author_fewshot_predictions.csv'),('No-code+flipped',E+'bias_controlled_glm51_test870_extra/glm-5.1_no_code_flipped_fewshot_predictions.csv'),('Skeleton',E+'bias_controlled_glm51_test870/glm-5.1_skeleton_only_predictions.csv'),('Meta no CWE',E+'bias_controlled_glm51_test870_extra/glm-5.1_metadata_no_cwe_predictions.csv'),('Meta with CWE',E+'bias_controlled_glm51_test870/glm-5.1_metadata_only_predictions.csv'),('CWE only',E+'bias_controlled_glm51_test870_extra/glm-5.1_cwe_only_predictions.csv'),('Code+meta no CWE',E+'bias_controlled_glm51_test870_code_metadata/glm-5.1_code_metadata_no_cwe_predictions.csv'),('Code+meta CWE',E+'bias_controlled_glm51_test870_code_metadata/glm-5.1_code_metadata_predictions.csv'),('Code+author',E+'bias_controlled_glm51_test870/glm-5.1_code_author_fewshot_predictions.csv'),('Code+flipped',E+'bias_controlled_glm51_test870/glm-5.1_code_author_flipped_fewshot_predictions.csv')]
for name,p in rows:
    if p is None: print(name); continue
    print(f'{name:26s}', stats(load(p)))

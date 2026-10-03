import json, collections, glob, re
raw=[json.loads(l) for l in open('data/primevul_test_paired_labeled.jsonl')]
cnt=collections.Counter(str(r['idx']) for r in raw)
def block(msg):
    f=msg.find('Focus Regions:')
    if f==-1: return None
    m=re.search(r'\n\nRetrieved ', msg[f:])
    return msg[f+m.start():] if m else ''
def roles(msg):
    return set(re.findall(r'\n\nRetrieved ([A-Za-z/ ]+?) Evidence', msg))
out={}
for p in sorted(glob.glob('outdir/prompt_*glm51_test870/*.jsonl')):
    by={}; pr={}
    for l in open(p):
        r=json.loads(l); k=str(r.get('query_idx',r.get('idx'))); by[k]=block(r['messages'][-1]['content'])
    same=n=0; ok=True
    for i in range(0,len(raw),2):
        a,b=str(raw[i]['idx']),str(raw[i+1]['idx'])
        if cnt[a]>1 or cnt[b]>1: continue
        if by.get(a) is None or by.get(b) is None or (by[a]=='' and by[b]==''): ok=False; continue
        n+=1; same+=by[a]==by[b]
    print(f'{same}/{n}' + (f' = {same/n:.1%}' if n else ''), p.split('/')[-2][7:], p.split('/')[-1][16:75])
r=json.loads(open('outdir/prompt_baseline_compatible_glm51_test870/glm-5.1_std_cls_grouped_no_label_baseline_compatible_fewshotegTrue_cls0.jsonl').readline())
print(roles(r['messages'][-1]['content']))

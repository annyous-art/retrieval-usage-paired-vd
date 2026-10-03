import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import extra_arm as x

class PreparationTests(unittest.TestCase):
 def test_fixed_and_shuffle_preserve_targets(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);src=root/'source';src.mkdir();targets=[];cands=[];ej=[];kr=[];jobs=[];pred=[];chunks=[]
   for i in range(4):
    cid=f'c{i}';func=f'int historical_{i}() {{ return {i}; }}'
    members=[{'func':func+str(t),'idx':i*2+t,'project':'p','commit_id':str(i)} for t in [1,0]]
    cands.append({'pair_id':cid,'members':members});ej.append({'job_id':cid,'candidate_id':cid});kr.append({'job_id':cid,'response':json.dumps(dict.fromkeys(x.exp.FIELDS,'knowledge '+str(i)))})
    for m in members:chunks.append(dict(m,function_id=m['idx'],score_combined=1,chunk_seq=0,chunk_id=m['idx'],start_line=1,end_line=1,code_raw=m['func']))
    target={'row_id':i,'pair_id':f'p{i//2}','target':i%2,'cluster':f'p{i//2}','func':f'int target_{i}() {{ return 0; }}','candidate_ids':[cid]};targets.append(target)
    for arm in x.exp.ARMS:
     msg=[{'role':'system','content':'system'},{'role':'user','content':'Historical evidence (may be irrelevant; check the target independently):\n(none)\n\n'+target['func']}]
     j=dict(target,arm=arm,job_id=f'{i}-{arm}',messages=msg);jobs.append(j);pred.append(dict(j,response='YES',config={'model':'gpt-5.5','max_tokens':4096,'temperature':0,'prompt_strategy':'std'}))
   for n,rs in [('targets',targets),('candidates',cands),('extraction_jobs',ej),('knowledge_responses',kr),('detection_jobs',jobs),('predictions',pred)]:x.exp.write(src/(n+'.jsonl'),rs)
   meta=root/'chunks.jsonl';x.exp.write(meta,chunks)
   for arm in ['slice_fixed','shuffled_knowledge']:
    out=root/arm;out.mkdir();a=NS(source=src,predictions=None,out=out,arm=arm,chunk_metadata=meta,stage='prepare')
    with patch('context_budget.audit_requests'):x.execute(a)
    made=x.exp.read(out/'jobs.jsonl');self.assertEqual(len(made),4)
    for j,t in zip(made,targets):
     self.assertTrue(j['messages'][1]['content'].endswith(t['func']))
     if arm=='slice_fixed':self.assertEqual(j['candidate_ids'],t['candidate_ids'])
     else:self.assertNotEqual(j['candidate_ids'],t['candidate_ids'])
    if arm=='shuffled_knowledge':self.assertEqual(sorted(j['candidate_ids'] for j in made),sorted(t['candidate_ids'] for t in targets))
    with patch('context_budget.audit_requests'):x.execute(a)
if __name__=='__main__':unittest.main()

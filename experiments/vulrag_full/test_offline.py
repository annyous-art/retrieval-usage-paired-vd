"""No paid/network calls. Run with unittest; rank_bm25/spaCy optional integration below."""
import ast, hashlib, json, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import run as app
from retrieval import Retriever

class Scripted:
    def __init__(self,texts):self.texts=iter(texts);self.ids=[]
    def call(self,id,messages,validator=app.nonempty):
        self.ids.append(id);text=next(self.texts);validator(text)
        return {'content':text,'call_id':id}

class TestProtocol(unittest.TestCase):
    def target(self):return {'row_id':1,'pair_id':'p0','idx':9,'target':1,'cluster':'c','func':'int f(){return 0;}'}
    def retrieved(self):return {'knowledge_ids':['a','b','c'],'evidence':[{'x':x} for x in 'abc']}
    def test_positive_requires_cause_yes_fix_no(self):
        c=Scripted(['<result>NO</result>','<result>NO</result>','<result>YES</result>','<result>NO</result>'])
        r=app.detect_one(self.target(),self.retrieved(),c);self.assertEqual(r['prediction'],1);self.assertEqual(len(c.ids),4)
    def test_fix_yes_early_return(self):
        c=Scripted(['<result>NO</result>','<result>YES</result>']);r=app.detect_one(self.target(),self.retrieved(),c);self.assertEqual(r['raw_final_result'],0);self.assertEqual(len(c.ids),2)
    def test_exhaustion_is_official_negative(self):
        c=Scripted(['<result>NO</result>']*6);r=app.detect_one(self.target(),self.retrieved(),c);self.assertEqual(r['raw_final_result'],-1);self.assertEqual(r['prediction'],0)
    def test_failed_response_not_negative(self):
        with self.assertRaises(ValueError):app.detect_one(self.target(),self.retrieved(),Scripted(['sorry']))
        with self.assertRaises(ValueError):app.label('<result>YES or NO</result>')
    def test_empty_retrieval_rejected(self):
        with self.assertRaises(ValueError):app.detect_one(self.target(),{'evidence':[]},Scripted([]))
    def test_parser_and_training_diff(self):
        self.assertEqual(app.label('<think>hidden</think>\n<result> no </result>'),0)
        with self.assertRaises(ValueError):app.knowledge_json('{"solution":"x"}')
        self.assertEqual(app.modified('a\nb','a\nc'),{'added':['c'],'deleted':['b']})
    def test_prompts_exact_upstream(self):
        selected={k for k in vars(app) if k.startswith('generate_')}
        import prompts
        for filename in ('extract_knowledge.py','vulnerability_detect.py'):
            text=(app.HERE/'vendor'/filename).read_text()
            for node in ast.parse(text).body:
                if isinstance(node,ast.FunctionDef) and node.name in selected:
                    target=next(n for n in ast.parse(Path(prompts.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==node.name)
                    self.assertEqual(ast.dump(node),ast.dump(target))
        doc=app.load(app.HERE/'vendor/provenance.json')
        for name,h in doc['files'].items():self.assertEqual(app.sha(app.HERE/'vendor'/name),h)
    def test_success_cache_and_identity_guard(self):
        class Fake:
            count=0
            def get_openai_chat(self,prompt,model,*args):
                Fake.count+=1;self.last_response_metadata={'returned_model':model,'generation_truncated':False};return 'hello',{'total_tokens':1},prompt['messages'],None
        with tempfile.TemporaryDirectory() as d,patch.object(app,'estimate',return_value={'status':'within_estimated_budget'}):
            a=SimpleNamespace(out=Path(d),model='glm-5.1',max_output=100,temperature=.01,max_calls=10,read_timeout=10,max_attempts=1,provider='uniapi')
            c=app.Calls(a,{'protocol':{'model':a.model}},Fake)
            for _ in range(2):c.call('same',app.message('hello'))
            self.assertEqual(Fake.count,1)
            c.call('different',app.message('hello'));self.assertEqual(Fake.count,2)
    def test_truncation_and_wrong_model_are_not_cached(self):
        for meta in ({'returned_model':'other'},{'returned_model':'glm-5.1','generation_truncated':True}):
            class Fake:
                def get_openai_chat(self,prompt,*args):self.last_response_metadata=meta;return '<result>YES</result>',{},prompt['messages'],None
            with tempfile.TemporaryDirectory() as d,patch.object(app,'estimate',return_value={'status':'within_estimated_budget'}):
                a=SimpleNamespace(out=Path(d),model='glm-5.1',max_output=100,temperature=.01,max_calls=10,read_timeout=10,max_attempts=1,provider='uniapi')
                with self.assertRaises(ValueError):app.Calls(a,{'protocol':{}},Fake).call('bad',app.message('x'),app.label)
                self.assertFalse((a.out/'calls').exists());self.assertEqual(len(list((a.out/'failures').glob('*.json'))),1)
    def test_unknown_score_preserves_denominator(self):
        with tempfile.TemporaryDirectory() as d:
            a=SimpleNamespace(out=Path(d));t=self.target();f={**t,'row_id':2,'target':0};app.atomic(a.out/'targets.json',[t,f]);app.atomic(a.out/'prediction_items/1.json',{**t,'prediction':1,'raw_final_result':1});app.score(a)
            m=app.load(a.out/'metrics.json');self.assertFalse(m['complete']);self.assertEqual(m['pair_outcomes'],{'unknown':1});self.assertEqual(m['completed'],1)
    def test_rank_sum_and_cve_dedup(self):
        class Toy:
            def __init__(self,corpus):self.corpus=corpus
            def search(self,text,n=-1):
                order=sorted(range(len(self.corpus)),key=lambda i:len(set(text.split())&set(self.corpus[i].split())),reverse=True)
                return order if n==-1 else order[:n]
        kb=[dict(id=str(i),CVE_id=c,purpose=p,function=p,code_before_change=p) for i,(c,p) in enumerate([('C2','free memory'),('C1','guard bounds'),('C1','allocation memory'),('C3','lock thread')])]
        index=Retriever(kb,Toy);selected,trace=index.search('guard bounds','guard bounds','guard bounds',20,3)
        self.assertEqual(selected[0]['id'],'1');self.assertEqual(len(set(x['CVE_id'] for x in selected)),3)
        self.assertEqual([x['id'] for x in selected],[x['id'] for x in index.search('guard bounds','guard bounds','guard bounds')[0]])

if __name__=='__main__':unittest.main()

class TestIntegration(unittest.TestCase):
    def test_real_bm25_matches_official_rank_sum(self):
        try:import rank_bm25,spacy
        except ImportError:self.skipTest('Install retrieval dependencies')
        import retrieval
        text=(app.HERE/'vendor/vulnerability_detect.py').read_text()
        node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='retrieve_knowledge_by_cve')
        class OfficialAdapter:
            def set_corpus(self,corpus):self.index=retrieval.BM25(corpus)
            def search(self,q,top_n=-1):return self.index.search(q,top_n)
        def k(i,c,t):return dict(id=i,CVE_id=c,purpose=t,function=t,code_before_change=t,vulnerability_cause_description='cause',trigger_condition='trigger',specific_code_behavior_causing_vulnerability='behavior',solution='fix')
        kb=[k('0','C0','guard bounds pointer'),k('1','C1','free allocation buffer'),k('2','C2','thread lock race'),k('3','C1','buffer allocation memory'),k('4','C3','parser protocol')]
        idx=retrieval.Retriever(kb);code='guard bounds pointer'
        ns={'GLOBAL_PURPOSE_RETRIEVER':idx.indices[0],'GLOBAL_FUNCTION_RETRIEVER':idx.indices[1],'GLOBAL_CODE_RETRIEVER':idx.indices[2],'GLOBAL_CVE_KNOWLEDGE_DICT':idx.groups,'bm25_retriever':SimpleNamespace(BM25Retriever=OfficialAdapter),'retrieve_weight':{'purpose':1,'function':1,'code':1},'json':json}
        # Support upstream search(top_n=...) without changing our implementation.
        class Wrapper:
            def __init__(self,i):self.i=i
            def search(self,q,top_n=-1):return self.i.search(q,top_n)
        for name,i in zip(('GLOBAL_PURPOSE_RETRIEVER','GLOBAL_FUNCTION_RETRIEVER','GLOBAL_CODE_RETRIEVER'),idx.indices):ns[name]=Wrapper(i)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'official','exec'),ns)
        from unittest.mock import mock_open
        with patch('builtins.open',mock_open(read_data=json.dumps(kb))):
            official=ns['retrieve_knowledge_by_cve'](SimpleNamespace(retrieval_top_k=20,max_knowledge=3,knowledge_file_name='fixture.json'),code,code,code)
        ours,_=idx.search(code,code,code)
        self.assertEqual(official,[app.evidence(x) for x in ours])
    def test_full_pipeline_resume_no_extra_calls(self):
        try:import rank_bm25,spacy
        except ImportError:self.skipTest('Install retrieval dependencies')
        class Fake:
            count=0
            def __init__(self,*args):pass
            def get_openai_chat(self,prompt,model,*args):
                Fake.count+=1;last=prompt['messages'][-1]['content']
                if 'organize vulnerability knowledge' in last:txt=json.dumps({'vulnerability_behavior':{'vulnerability_cause_description':'missing guard','trigger_condition':'index exceeds bounds','specific_code_behavior_causing_vulnerability':'unguarded array access'},'solution':'guard bounds'})
                elif 'What is the purpose' in last:txt='Function purpose: process buffer'
                elif 'summarize the functions' in last:txt='The functions of the code snippet are: 1. Access buffer'
                elif 'Why is the above modification necessary' in last:txt='Guard prevents invalid access.'
                elif 'contains similar solution behaviors' in last:txt='<result>NO</result>'
                else:txt='<result>YES</result>'
                self.last_response_metadata={'returned_model':model,'generation_truncated':False}
                return txt,{'total_tokens':12},prompt['messages'],None
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);source=d/'source';source.mkdir();train=[]
            for i in range(4):
                code=f'int train{i}() {{ return {i}; }}';train.append(dict(idx=i,project='train',commit_id='c'+str(i//2),cve='CVE-'+str(i//2),cve_desc='bounds',target=1-i%2,func=code,labels=[1],file_name='a.c'))
            test=[dict(idx=i+10,project='test',commit_id='testcommit',cve='CVE-test',target=1-i%2,func=f'int test{i}() {{ return {i}; }}',labels=[1],file_name='t.c') for i in range(2)]
            for name,rows in [('train',train),('test',test)]: (d/(name+'.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in rows))
            pool=app.pairs(train)[0];targets=[dict(row_id=i+1,idx=t['idx'],target=t['target'],func=t['func'],pair_id='p0',cluster='test:testcommit') for i,t in enumerate(test)]
            for name,rows in [('candidate_pool',pool),('targets',targets)]: (source/(name+'.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in rows))
            app.atomic(source/'manifest.json',{'train_sha256':app.digest(train),'test_sha256':app.digest(test)})
            a=SimpleNamespace(source=source,train=d/'train.jsonl',test=d/'test.jsonl',out=d/'output',model='gpt-5.5',provider='newapi',temperature=.01,max_output=16384,read_timeout=900,max_attempts=1,max_calls=100,workers=2)
            app.prepare(a);manifest=app.load(a.out/'manifest.json')
            with patch.object(app,'GatewayClient',Fake),patch.object(app,'get_key',return_value='test-placeholder'):
                app.build_kb(a,manifest,probe=True);app.build_kb(a,manifest);app.queries(a,manifest);app.retrieve(a);app.detect(a,manifest)
                n=Fake.count
                app.build_kb(a,manifest);app.queries(a,manifest);app.retrieve(a);app.detect(a,manifest)
                self.assertEqual(Fake.count,n)
            self.assertEqual(n,16);self.assertTrue(app.load(a.out/'metrics.json')['complete'])

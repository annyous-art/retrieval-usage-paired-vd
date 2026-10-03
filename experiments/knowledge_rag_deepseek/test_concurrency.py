import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import requests
import experiment as exp
from gateway_client import GatewayClient, GatewayError
from concurrent_runner import import_extractions


class TransportTests(unittest.TestCase):
    def test_option_numbers_and_conflicting_labels(self):
        self.assertEqual(exp.prediction('(1)'),1)
        self.assertEqual(exp.prediction('(2)'),0)
        self.assertEqual(exp.prediction('(1) YES: A security vulnerability detected.'),1)
        self.assertIsNone(exp.prediction('(1) NO'))
        self.assertIsNone(exp.prediction('YES or NO'))
        self.assertIsNone(exp.prediction('(1'))
        self.assertTrue(exp.complete_final_answer('YES: A security vulnerability detected.'))
        self.assertFalse(exp.complete_final_answer('YES because the function might'))
        self.assertFalse(exp.complete_final_answer('(1) NO: No security vulnerability.'))
        self.assertEqual(exp.prediction('There is NO length check.\n\nYES'),1)
        self.assertIsNone(exp.prediction('YES\nNO'))

    def test_literal_c_escape_repair_preserves_text(self):
        raw=r'{"functional_semantics":"writes \\0", "root_cause":"a", "fix_condition":"b"}'
        obj,repairs=exp.parse_knowledge(raw,with_repairs=True)
        self.assertEqual(obj['functional_semantics'],r'writes \0')
        self.assertEqual(repairs,[])
        broken=r'{"functional_semantics":"writes \0", "root_cause":"a", "fix_condition":"b"}'
        obj,repairs=exp.parse_knowledge(broken,with_repairs=True)
        self.assertEqual(obj['functional_semantics'],r'writes \0')
        self.assertEqual(repairs,['escaped_invalid_json_backslashes'])

    def test_timeout_and_429_retry_then_success(self):
        api=GatewayClient('https://example.test/v1','fixture',lambda ms,m,n:ms)
        good=Mock(status_code=200,json=lambda:{'choices':[{'finish_reason':'stop','message':{'content':'YES'}}]})
        limited=Mock(status_code=429,text='rate limited',headers={'Retry-After':'0'})
        with patch('gateway_client.requests.post',side_effect=[requests.ReadTimeout(),limited,good]) as post, patch('gateway_client.time.sleep'):
            self.assertEqual(api.get_openai_chat({'messages':[]},'glm-5.1','std',0,64,1)[0],'YES')
            self.assertEqual(post.call_count,3)
            self.assertEqual(post.call_args.kwargs['timeout'],(15,600))
            self.assertEqual(len(api.last_response_metadata['attempts']),3)

    def test_permanent_error_never_retried(self):
        api=GatewayClient('https://example.test/v1','fixture',lambda ms,m,n:ms)
        with patch('gateway_client.requests.post',return_value=Mock(status_code=401,text='bad fixture')) as post:
            with self.assertRaises(GatewayError) as err:
                api.get_openai_chat({'messages':[]},'glm-5.1','std',0,64,1)
            self.assertTrue(err.exception.fatal)
            self.assertEqual(post.call_count,1)
            self.assertNotIn('fixture',str(err.exception))

    def test_gateway_524_is_retried(self):
        api=GatewayClient('https://example.test/v1','fixture',lambda ms,m,n:ms)
        timeout=Mock(status_code=524,text='upstream timeout',headers={})
        good=Mock(status_code=200,json=lambda:{'choices':[{'finish_reason':'stop','message':{'content':'NO'}}]})
        with patch('gateway_client.requests.post',side_effect=[timeout,good]) as post,patch('gateway_client.time.sleep'):
            self.assertEqual(api.get_openai_chat({'messages':[]},'glm-5.1','std',0,64,1)[0],'NO')
            self.assertEqual(post.call_count,2)

    def test_concurrent_failure_resume_and_metadata_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);source=d/'api.py';source.write_text('# fixture')
            jobs=[dict(job_id=str(i),messages=[{'role':'user','content':str(i)}],row_id=i+1,idx=i,target=1,cluster='p:c',arm='knowledge') for i in range(8)]
            exp.write(d/'jobs.jsonl',jobs)
            lock=threading.Lock();state={'active':0,'peak':0,'fail':True,'calls':[]}
            class Client:
                __file__=str(source)
                client=NS(base_url='https://example.test/v1')
                def truncate_tokens_from_messages(self,ms,m,n): return ms
                def get_openai_chat(self,prompt,*args):
                    jid=prompt['messages'][0]['content']
                    with lock:
                        state['active']+=1;state['peak']=max(state['peak'],state['active']);state['calls'].append(jid)
                    time.sleep(.01)
                    with lock: state['active']-=1
                    if jid=='3' and state['fail']: raise GatewayError('timeout',[{'error_type':'ReadTimeout'}])
                    self.last_response_metadata={'fixture_job':jid}
                    return 'YES',{},prompt['messages'],None
            a=NS(jobs=d/'jobs.jsonl',output=d/'results.jsonl',model='glm-5.1',temperature=0,max_output=64,prompt_strategy='std',seed=1,max_calls=8,workers=3)
            with patch('experiment.load_client',side_effect=Client):
                with self.assertRaises(ValueError): exp.run(a)
                rows=exp.read(a.output)
                self.assertEqual(len(rows),7)
                self.assertTrue(1<state['peak']<=3)
                self.assertTrue(all(r['provider_metadata']['fixture_job']==r['job_id'] for r in rows))
                state['fail']=False;state['calls']=[]
                exp.run(a)
                self.assertEqual(state['calls'],['3'])
                self.assertEqual(len(exp.read(a.output)),8)

    def test_preflight_large_bytes_preserves_all_requests_without_api(self):
        from context_budget import policy
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            exp.write(d/'targets.jsonl',[{'row_id':651,'func':'target','candidate_ids':['p']}])
            exp.write(d/'candidates.jsonl',[{'pair_id':'p','members':[{'func':'int foo;\n'*9000},{'func':'int bar;\n'*9000}]}])
            exp.write(d/'extraction_jobs.jsonl',[])
            a=NS(out=d,model='glm-5.1',max_output=4096,extract_max_output=4096,prompt_strategy='std',temperature=0)
            with patch('experiment.load_client') as load:
                exp.preflight(a)
                load.assert_not_called()
            report=json.loads((d/'preflight.json').read_text())
            self.assertEqual(report['request_count'],2)
            self.assertGreater(report['requests'][1]['evidence_bytes'],128000)
            self.assertEqual(report['needs_capacity_review'],0)
            with patch('context_budget.policy',return_value={**policy(a.model),'gateway_context_tokens':100}):
                with self.assertRaisesRegex(ValueError,'2 requests'): exp.preflight(a)
            self.assertEqual(json.loads((d/'preflight.json').read_text())['needs_capacity_review'],2)

    def test_verified_import_retains_provenance_and_rejects_different_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);old=d/'old';new=d/'new';old.mkdir();new.mkdir()
            src=d/'client.py';src.write_text('# fixture')
            config={'data_hashes':{'train':'t','test':'e'},'model':'glm-5.1','temperature':0,'prompt_strategy':'std','seed':1,'extract_max_output':2048,'gateway_base_url':'https://example.test/v1','api_route':'chat/completions','effective_request_parameters':{'extract':{'model':'glm-5.1'}}}
            for path in (old,new): (path/'server_config.json').write_text(json.dumps(config))
            job={'job_id':'x','candidate_id':'p','messages':[{'role':'user','content':'historical code'}]}
            exp.write(new/'extraction_jobs.jsonl',[job])
            original=dict(model='glm-5.1',base_url='https://example.test/v1',temperature=0,max_tokens=2048,prompt_strategy='std',seed=1,api_client_sha256='old-hash')
            exp.write(old/'knowledge_responses.jsonl',[dict(job,config=original,sent_messages=job['messages'],response=json.dumps(dict(functional_semantics='f',root_cause='r',fix_condition='c')))])
            a=NS(model='glm-5.1',temperature=0,max_output=2048,prompt_strategy='std',seed=1)
            with patch('experiment.load_client',return_value=NS(__file__=str(src),client=NS(base_url='https://example.test/v1'))):
                import_extractions(old,new,a);import_extractions(old,new,a)
                rows=exp.read(new/'knowledge_responses.jsonl')
                self.assertEqual(len(rows),1)
                self.assertEqual(rows[0]['import_provenance']['original_config'],original)
                config['model']='gpt-5.5';(new/'server_config.json').write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError,'model'): import_extractions(old,new,a)

if __name__=='__main__': unittest.main()

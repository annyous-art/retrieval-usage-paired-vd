import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
from context_budget import estimate, check_context, audit_requests, policy, text_counts
from gateway_client import GatewayClient

class ContextTests(unittest.TestCase):
    def test_bytes_are_not_tokens_and_no_truncation(self):
        messages=[{'role':'system','content':'check'}, {'role':'user','content':'int foo;\n'*25000}]
        before=copy.deepcopy(messages)
        r=estimate(messages,'glm-5.1',4096)
        self.assertGreater(r['request_json_bytes'],128000)
        self.assertLess(r['estimated_total_tokens'],200000)
        self.assertIs(check_context(messages,'glm-5.1',4096),messages)
        self.assertEqual(messages,before)

    def test_complete_request_and_output_floor(self):
        m=[{'role':'system','content':'entire system'}, {'role':'user','content':'entire code <|endoftext|>'}]
        a=estimate(m,'gpt-5.5',1)
        self.assertEqual(a['effective_output_tokens'],4096)
        self.assertGreater(estimate(m+[{'role':'user','content':'code '*1000}],'gpt-5.5',1)['raw_token_estimate'],a['raw_token_estimate'])
        self.assertGreater(text_counts('Ünïcödé côde')['cl100k_base'],0)

    def test_all_failures_saved_before_raise_no_network(self):
        jobs=[dict(row_id=i,messages=[{'role':'user','content':'code'}]) for i in range(3)]
        small={**policy('glm-5.1'),'gateway_context_tokens':100}
        with tempfile.TemporaryDirectory() as d, patch('context_budget.policy',return_value=small), patch('requests.post') as post:
            path=Path(d)/'audit.json'
            with self.assertRaisesRegex(ValueError,'3 requests'):
                audit_requests(jobs,'glm-5.1',4096,path)
            self.assertEqual(json.loads(path.read_text())['needs_capacity_review'],3)
            api=GatewayClient('https://fixture.invalid','fixture')
            with self.assertRaises(ValueError):
                api.get_openai_chat(jobs[0],'glm-5.1','std',0,4096,1)
            post.assert_not_called()

    def test_separate_input_output_limits(self):
        p={**policy('gemini-3.1-pro-preview'),'input_tokens':100000,'output_tokens':10}
        with patch('context_budget.policy',return_value=p):
            r=estimate([{'role':'user','content':'hi'}],'gemini-3.1-pro-preview',11)
        self.assertEqual(r['status'],'needs_capacity_review')
        self.assertEqual([c['limit_kind'] for c in r['checks'] if c['headroom']<0],['output_tokens'])

    def test_assembly_audits_all_arms_and_writes_no_partial_jobs(self):
        from types import SimpleNamespace as NS
        import experiment as exp
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp)
            exp.write(d/'extraction_jobs.jsonl',[{'job_id':'e','candidate_id':'p'}])
            exp.write(d/'candidates.jsonl',[{'pair_id':'p','members':[{'func':'unsafe();'},{'func':'fixed();'}]}])
            exp.write(d/'targets.jsonl',[dict(row_id=1,idx=1,pair_id='t',target=1,cluster='c',func='target();',candidate_ids=['p'])])
            exp.write(d/'knowledge.jsonl',[{'job_id':'e','response':json.dumps(dict(functional_semantics='s',root_cause='r',fix_condition='f'))}])
            with patch('context_budget.policy',return_value={**policy('glm-5.1'),'gateway_context_tokens':100}):
                with self.assertRaisesRegex(ValueError,'4 requests'):
                    exp.assemble(NS(directory=d,knowledge=d/'knowledge.jsonl',model='glm-5.1',max_output=4096))
            self.assertFalse((d/'detection_jobs.jsonl').exists())
            report=json.loads((d/'assembly_context_audit.json').read_text())
            self.assertEqual(set(report['groups']),set(exp.ARMS))
            self.assertEqual(report['request_count'],4)

if __name__=='__main__': unittest.main()

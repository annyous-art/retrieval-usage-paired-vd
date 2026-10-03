import copy
import unittest
from retry_extraction_budget import assemble_merged, validate_records

class RetryBudgetTests(unittest.TestCase):
    def setUp(self):
        self.cfg=dict(model='glm-5.1',gateway_base_url='https://fixture.invalid',temperature=0,prompt_strategy='std',seed=1,extract_max_output=4096)
        self.jobs=[dict(job_id=str(i),messages=[dict(role='user',content='fixture '+str(i))]) for i in range(3)]
    def record(self,i,budget):
        j=self.jobs[i]
        return dict(j,sent_messages=copy.deepcopy(j['messages']),response='{"functional_semantics":"fixture","root_cause":"fixture","fix_condition":"fixture"}',
                    config=dict(model=self.cfg['model'],base_url=self.cfg['gateway_base_url'],temperature=0,prompt_strategy='std',seed=1,max_tokens=budget))
    def test_preserves_budgets_and_original_records(self):
        original=[self.record(0,4096),self.record(1,4096)];retry=[self.record(2,16384)]
        before=copy.deepcopy(original)
        merged=assemble_merged(original,retry,self.jobs,self.cfg,16384)
        self.assertEqual([r['config']['max_tokens'] for r in merged],[4096,4096,16384])
        self.assertEqual(original,before)
    def test_incomplete_blocks_assembly(self):
        with self.assertRaisesRegex(ValueError,'incomplete'):
            assemble_merged([self.record(0,4096)],[],self.jobs,self.cfg,16384)
    def test_wrong_budget_and_changed_prompt_rejected(self):
        with self.assertRaisesRegex(ValueError,'configuration'):
            validate_records([self.record(2,4096)],self.jobs,self.cfg,16384)
        r=self.record(2,16384);r['sent_messages']=[]
        with self.assertRaisesRegex(ValueError,'prompt'):
            validate_records([r],self.jobs,self.cfg,16384)
    def test_overlapping_retry_rejected(self):
        with self.assertRaisesRegex(ValueError,'overlaps'):
            assemble_merged([self.record(0,4096)],[self.record(0,16384)],self.jobs,self.cfg,16384)
    def test_truncated_json_cannot_be_reused(self):
        r=self.record(0,4096);r['provider_metadata']={'generation_truncated':True}
        with self.assertRaisesRegex(ValueError,'Truncated'):
            validate_records([r],self.jobs,self.cfg,4096)

if __name__=='__main__':unittest.main()

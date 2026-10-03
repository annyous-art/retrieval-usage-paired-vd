"""Offline integrity checks; fabricated fixtures are never paper evidence."""
import unittest
from unittest.mock import Mock, patch
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
from experiment import pairs, parse_knowledge, outcome, digest, read, write, assemble, score, run, prediction
from legacy_api import gateway_url, load_client

class IntegrityTests(unittest.TestCase):
    def test_gateway_validation_and_explicit_key(self):
        with patch.dict('os.environ', {'SLICERAG_BASE_URL':'https://gateway.example/v1/'}, clear=True):
            self.assertEqual(gateway_url(), 'https://gateway.example/v1')
            with self.assertRaises(ValueError):
                load_client()
        for url in ['https://docs.gateway.example/v1', 'https://example.com/v1/chat/completions', 'https://key@example.com/v1']:
            with patch.dict('os.environ', {'SLICERAG_BASE_URL':url}, clear=True), self.assertRaises(ValueError):
                gateway_url()

    def test_gateway_replaces_legacy_client(self):
        fake = SimpleNamespace(truncate_tokens_from_messages=lambda ms, model, n:ms)
        with patch.dict('os.environ', {'SLICERAG_BASE_URL':'https://gateway.example/v1', 'SLICERAG_API_KEY':'fixture'}, clear=True), patch('legacy_api.local_config', return_value=None), patch('legacy_api.load_file', return_value=fake):
            api = load_client()
            self.assertEqual(api.client.base_url, 'https://gateway.example/v1')
            with self.assertRaises(ValueError):
                api.get_openai_chat({}, 'other-model', 'std', 0, 32, 1)

    def row(self, target, **kw):
        return dict(project='p', commit_id='c', cve='CVE-test', file_name='f.c', target=target, **kw)

    def test_pair_mismatch_and_missing_filename(self):
        a,b = self.row(1), self.row(0)
        b['file_name'] = 'None'
        self.assertEqual(len(pairs([a,b])[0]), 1)
        b['commit_id'] = 'another'
        self.assertEqual(len(pairs([a,b])[1]), 1)

    def test_unknown_is_not_benign(self):
        self.assertEqual(outcome(1, None), 'unknown')
        self.assertEqual(outcome(1, 0), 'P-C')
        self.assertEqual(outcome(1, 1), 'P-V')

    def test_malformed_extraction_rejected(self):
        with self.assertRaises(ValueError):
            parse_knowledge('{"root_cause":"x"}')
        with self.assertRaises(ValueError):
            parse_knowledge('{"root_cause":"x","functional_semantics":"x","fix_condition":""}')

    def test_offline_assembly_scoring_and_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write(d/'extraction_jobs.jsonl', [{'job_id':'extract', 'candidate_id':'p0'}])
            write(d/'candidates.jsonl', [{'pair_id':'p0', 'members':[{'func':'unsafe();'}, {'func':'guard(); unsafe();'}]}])
            # Same code in two labelled rows must still have distinct job IDs.
            write(d/'targets.jsonl', [dict(row_id=i+1, idx=9, pair_id='test', target=y,
                func='target();', cluster='proj:commit', candidate_ids=['p0']) for i,y in enumerate([1,0])])
            write(d/'knowledge.jsonl', [{'job_id':'extract', 'response':json.dumps(dict(
                functional_semantics='fixture function', root_cause='fixture cause', fix_condition='fixture predicate'))}])
            assemble(SimpleNamespace(directory=str(d), knowledge=d/'knowledge.jsonl', model='glm-5.1', max_output=4096))
            jobs = read(d/'detection_jobs.jsonl')
            self.assertEqual(len({j['job_id'] for j in jobs}), 8)
            for j in jobs:
                user = j['messages'][1]['content']
                self.assertNotIn('pair_id', user)
                if j['arm'] == 'knowledge_no_fix':
                    self.assertNotIn('fixture predicate', user)
            write(d/'responses.jsonl', [dict(j, response='YES' if j['target'] else 'NO', config={}) for j in jobs])
            args = SimpleNamespace(jobs=d/'detection_jobs.jsonl', responses=d/'responses.jsonl',
                output=d/'metrics.json', bootstrap=50, seed=1)
            score(args)
            report = json.loads((d/'metrics.json').read_text())
            self.assertTrue(report['complete_and_parseable'])
            self.assertEqual(report['paired_comparisons']['knowledge minus code_pair']['delta_P-C'], 0)
            (d/'responses.jsonl').write_text('')
            score(args)
            report = json.loads((d/'metrics.json').read_text())
            self.assertFalse(report['complete_and_parseable'])
            self.assertEqual(report['paired_comparisons'], {})

    def test_legacy_api_call_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            source = d/'client.py'
            source.write_text('# fake client fixture')
            messages = [{'role':'user','content':'classify fixture'}]
            job = dict(job_id='job', messages=messages, row_id=1, idx=7, target=1, cluster='p:c', arm='knowledge')
            write(d/'jobs.jsonl', [job])
            api = SimpleNamespace(__file__=str(source), client=SimpleNamespace(base_url='https://fixture.invalid'),
                truncate_tokens_from_messages=lambda ms,model,n: ms,
                get_openai_chat=Mock(return_value=('YES: A security vulnerability detected.', {'total_tokens':3}, messages, None)))
            args = SimpleNamespace(jobs=d/'jobs.jsonl', output=d/'responses.jsonl', model='glm-5.1',
                temperature=0, prompt_strategy='std', seed=42, max_output=1024, max_calls=1)
            with patch('experiment.load_client', return_value=api):
                run(args)
                run(args)
            api.get_openai_chat.assert_called_once_with({'messages':messages}, 'glm-5.1','std',0,1024,42)
            records = read(d/'responses.jsonl')
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]['query_idx'], 7)
            self.assertEqual(prediction(records[0]['response']), 1)
            self.assertIsNone(prediction('YES or NO'))
            args.output = d/'failed.jsonl'
            api.get_openai_chat.return_value = ('[ERROR]', {}, messages, None)
            with patch('experiment.load_client', return_value=api), self.assertRaises(ValueError):
                run(args)
            self.assertEqual(read(args.output), [])
            self.assertEqual(len(read(str(args.output)+'.errors.jsonl')), 1)

    def test_job_hash_changes_with_prompt(self):
        self.assertNotEqual(digest(['YES']), digest(['NO']))

if __name__ == '__main__':
    unittest.main()

import unittest
from unittest.mock import patch, Mock
from gateway_client import GatewayClient, request_body, parse_response

class GatewayTests(unittest.TestCase):
    def test_routes_payloads_auth_and_usage(self):
        messages=[{'role':'system','content':'system'}, {'role':'user','content':'hello'}]
        fixtures=[
            ('glm-5.1','chat/completions', {'choices':[{'finish_reason':'stop','message':{'content':'OK'}}], 'usage':{'total_tokens':12}}),
            ('gpt-5.5','responses', {'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':'OK'}]}],'usage':{'input_tokens':10,'output_tokens':2}}),
            ('claude-opus-4-7','messages', {'stop_reason':'end_turn','content':[{'type':'text','text':'OK'}],'usage':{'input_tokens':10,'output_tokens':2}})]
        api=GatewayClient('https://example.test/v1','fixture-secret',lambda ms,model,n:ms)
        for model, route, data in fixtures:
            with self.subTest(model=model), patch('gateway_client.requests.post', return_value=Mock(status_code=200,json=lambda:data)) as post:
                result=api.get_openai_chat({'messages':messages},model,'std',0,128,42)
                self.assertEqual(result[0],'OK')
                self.assertEqual(result[1]['total_tokens'],12)
                self.assertEqual(result[2],messages)
                self.assertEqual(post.call_args.args[0],'https://example.test/v1/'+route)
                kwargs=post.call_args.kwargs
                self.assertFalse(kwargs['allow_redirects'])
                if route=='messages':
                    self.assertEqual(kwargs['json']['system'],'system')
                    self.assertEqual(kwargs['json']['messages'],messages[1:])
                    self.assertEqual(kwargs['headers']['x-api-key'],'fixture-secret')
                if route=='responses':
                    self.assertEqual(kwargs['json']['max_output_tokens'],4096)
                    self.assertNotIn('temperature',kwargs['json'])

    def test_truncated_responses_rejected(self):
        with self.assertRaises(ValueError):
            parse_response('responses',{'status':'incomplete'})
        with self.assertRaises(ValueError):
            parse_response('messages',{'stop_reason':'max_tokens'})
        with self.assertRaises(ValueError):
            parse_response('chat/completions',{'choices':[{'finish_reason':'length'}]})

    def test_error_redaction(self):
        api=GatewayClient('https://example.test/v1','fixture-secret',lambda ms,model,n:ms)
        with patch('gateway_client.requests.post',return_value=Mock(status_code=401,text='bad fixture-secret')):
            with self.assertRaises(RuntimeError) as error:
                api.get_openai_chat({'messages':[]},'glm-5.1','std',0,32,42)
        self.assertNotIn('fixture-secret',str(error.exception))

    def test_gemini_and_reasoning(self):
        self.assertEqual(request_body([], 'gemini-3.1-pro-preview','std',0,128)['reasoning_effort'],'low')
        self.assertEqual(request_body([], 'gpt-5.5','cot',0,128)['reasoning']['effort'],'xhigh')

if __name__=='__main__':
    unittest.main()

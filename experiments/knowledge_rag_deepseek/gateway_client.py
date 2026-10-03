"""Explicit, non-fallback gateway routes. No credentials in request metadata.

DeepSeek copy of experiments/knowledge_rag/gateway_client.py. Differences: the
deepseek-v4-pro-0813 route (UniAPI chat/completions, thinking disabled for std, as for
glm-5.1) and an exact check of the model name the gateway returns for it."""
from types import SimpleNamespace
import requests
import time
import random


from context_budget import check_context


class GatewayError(RuntimeError):
    def __init__(self, message, attempts, fatal=False):
        super().__init__(message)
        self.attempts = attempts
        self.fatal = fatal

ROUTES = {
    'glm-5.1': 'chat/completions',
    'gpt-5.5': 'responses',
    'claude-opus-4-7': 'messages',
    'claude-opus-4-6': 'messages',
    'gemini-3.1-pro-preview': 'chat/completions',
    'deepseek-v4-pro-0813': 'chat/completions',
}
# The gateway reports the 2026-08-13 GA build under this name; anything else is rejected.
EXPECTED_RETURNED = {'deepseek-v4-pro-0813': 'deepseek-v4-pro-ga-260813'}


def request_body(messages, model, strategy, temperature, limit):
    route = ROUTES[model]
    if strategy not in ('std', 'cot'):
        raise ValueError('Expected std or cot')
    if route == 'responses':
        return dict(model=model, input=messages, max_output_tokens=max(4096, limit),
                    reasoning={'effort':'xhigh' if strategy == 'cot' else 'none',
                               'summary':'detailed' if strategy == 'cot' else 'concise'},
                    text={'verbosity':'low'})
    if route == 'messages':
        body = dict(model=model, messages=[m for m in messages if m['role'] != 'system'],
                    max_tokens=limit, thinking={'type':'disabled'})
        system = '\n\n'.join(m['content'] for m in messages if m['role'] == 'system')
        if system:
            body['system'] = system
        if strategy == 'cot':
            body['thinking'] = {'type':'adaptive', 'display':'summarized'}
            body['output_config'] = {'effort':'max'}
        return body
    body = dict(model=model, messages=messages, max_tokens=limit, temperature=temperature)
    if model in ('glm-5.1', 'deepseek-v4-pro-0813'):
        body['thinking'] = {'type':'enabled' if strategy == 'cot' else 'disabled'}
    elif model == 'gemini-3.1-pro-preview':
        body['reasoning_effort'] = 'high' if strategy == 'cot' else 'low'
    return body


def parse_response(route, data, allow_truncated=False):
    reasoning = None
    if route == 'responses':
        if data.get('status') != 'completed' and not (allow_truncated and data.get('status')=='incomplete' and data.get('incomplete_details',{}).get('reason')=='max_output_tokens'):
            raise ValueError(f'Responses status={data.get("status")} incomplete_details={data.get("incomplete_details")}; no prediction cached')
        content = '\n'.join(c['text'] for item in data.get('output', []) if item.get('type') == 'message'
                            for c in item.get('content', []) if c.get('type') == 'output_text')
        reasoning = '\n'.join(c.get('text','') for item in data.get('output', []) if item.get('type') == 'reasoning'
                              for c in item.get('summary', []) if c.get('type') == 'summary_text') or None
    elif route == 'messages':
        if data.get('stop_reason') not in ('end_turn', 'stop_sequence') and not (allow_truncated and data.get('stop_reason')=='max_tokens'):
            raise ValueError(f'Claude stop_reason={data.get("stop_reason")}; no prediction cached')
        content = '\n'.join(c['text'] for c in data.get('content', []) if c.get('type') == 'text')
        reasoning = '\n'.join(c.get('thinking','') for c in data.get('content', []) if c.get('type') == 'thinking') or None
    else:
        choice = data['choices'][0]
        if choice.get('finish_reason') != 'stop' and not (allow_truncated and choice.get('finish_reason')=='length'):
            raise ValueError(f'Chat finish_reason={choice.get("finish_reason")}; no prediction cached')
        content = choice['message'].get('content')
        reasoning = choice['message'].get('reasoning_content')
    if not isinstance(content, str) or not content.strip():
        raise ValueError('Gateway returned no final text')
    usage = dict(data.get('usage') or {})
    if 'total_tokens' not in usage and 'input_tokens' in usage and 'output_tokens' in usage:
        usage['total_tokens'] = usage['input_tokens'] + usage['output_tokens']
        # Anthropic separates cached input from the uncached input count.
        if route == 'messages':
            usage['total_tokens'] += usage.get('cache_read_input_tokens', 0) + usage.get('cache_creation_input_tokens', 0)
    return content.strip(), usage, reasoning


class GatewayClient:
    def __init__(self, base_url, key, truncate=check_context):
        self.client = SimpleNamespace(base_url=base_url)
        self.key = key
        self.__file__ = __file__
        self.truncate_tokens_from_messages = truncate
        self.read_timeout = 600
        self.max_attempts = 3
        self.retry_base = 3

    def request_metadata(self, messages, model, strategy, temperature, limit):
        body = request_body(messages, model, strategy, temperature, limit)
        return {'endpoint':self.client.base_url+'/'+ROUTES[model],
                'effective_parameters':{k:v for k,v in body.items() if k not in ('input','messages','system')}}

    def get_openai_chat(self, prompt, model, prompt_strategy, temperature, max_gen_length, seed):
        if model not in ROUTES:
            raise ValueError('Unsupported gateway model: '+model)
        messages = prompt['messages']
        effective_limit = max(4096,max_gen_length) if ROUTES[model] == 'responses' else max_gen_length
        if self.truncate_tokens_from_messages(messages, model, effective_limit) != messages:
            raise ValueError('Prompt would be truncated; not sent')
        body = request_body(messages, model, prompt_strategy, temperature, max_gen_length)
        from context_budget import estimate
        context_estimate = estimate(messages,model,max_gen_length,prompt_strategy,temperature)
        if context_estimate['status'] != 'within_estimated_budget':
            raise ValueError('Actual request needs capacity review; no truncation or API call: '+str(context_estimate))
        headers = {'Content-Type':'application/json'}
        if ROUTES[model] == 'messages':
            headers.update({'x-api-key':self.key, 'anthropic-version':'2023-06-01'})
        else:
            headers['Authorization'] = 'Bearer '+self.key
        attempts = []
        response = None
        for number in range(1, self.max_attempts + 1):
            started = time.monotonic()
            retry_after = 0
            try:
                response = requests.post(self.client.base_url+'/'+ROUTES[model], headers=headers,
                                         json=body, timeout=(15,self.read_timeout), allow_redirects=False)
                attempt = {'attempt':number, 'elapsed_seconds':round(time.monotonic()-started,3),
                           'http_status':response.status_code}
                attempts.append(attempt)
                if response.status_code == 200:
                    break
                message = f'Gateway HTTP {response.status_code}: '+response.text[:1000].replace(self.key,'[REDACTED]')
                retryable = response.status_code in (408,429,500,502,503,504,520,521,522,523,524,529) and 'model_not_found' not in response.text
                if not retryable:
                    raise GatewayError(message, attempts, fatal=response.status_code in (401,403) or 'model_not_found' in response.text)
                try:
                    retry_after = min(60, max(0, float(response.headers.get('Retry-After',0))))
                except (ValueError,TypeError):
                    pass
            except (requests.Timeout, requests.ConnectionError) as exc:
                attempts.append({'attempt':number,'elapsed_seconds':round(time.monotonic()-started,3),'error_type':type(exc).__name__})
                message = 'Gateway connection failed ('+type(exc).__name__+')'
            except requests.RequestException as exc:
                raise GatewayError('Gateway request failed ('+type(exc).__name__+')',attempts) from None
            if number == self.max_attempts:
                raise GatewayError(message+'; retry budget exhausted',attempts)
            delay = max(retry_after, self.retry_base * 2**(number-1) + random.uniform(0,1))
            print(f'Retry model={model} attempt={number+1}/{self.max_attempts} after={delay:.1f}s reason={message[:160]}',flush=True)
            time.sleep(delay)
        data = response.json()
        if model in EXPECTED_RETURNED and data.get('model') != EXPECTED_RETURNED[model]:
            raise GatewayError('Returned model '+str(data.get('model'))+' differs from '+EXPECTED_RETURNED[model]+'; not cached',attempts,fatal=True)
        try:
            content, usage, reasoning = parse_response(ROUTES[model], data,allow_truncated=True)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            error=GatewayError(str(exc),attempts)
            error.response_data=data
            raise error from None
        self.last_response_metadata = {'returned_model':data.get('model'), 'request_id':data.get('id'),
                                       'generation_truncated':data.get('status')=='incomplete' or data.get('stop_reason')=='max_tokens' or any(c.get('finish_reason')=='length' for c in data.get('choices',[])),
                                       'stop_reason':data.get('stop_reason') or data.get('status') or next((c.get('finish_reason') for c in data.get('choices',[])),None),
                                       'attempts':attempts,
                                       'context_estimate':context_estimate,
                                       'returned_instructions':data.get('instructions'),
                                       **self.request_metadata(messages,model,prompt_strategy,temperature,max_gen_length)}
        return content, usage, messages, reasoning

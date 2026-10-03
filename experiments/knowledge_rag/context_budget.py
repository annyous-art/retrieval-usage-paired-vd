"""Offline estimates, NOT provider-exact counts or proof of endpoint capacity."""
import hashlib
import json
import math
import os
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
POLICY_PATH = HERE / 'context_budgets.json'
VOCABS = {
    'o200k_base': ('fb374d419588a4632f3f557e76b4b70aebbca790', '446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d'),
    'cl100k_base': ('9b5ad71b2ce5302211f9c61530b329a4922fc6a4', '223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7'),
}

@lru_cache(maxsize=1)
def encoders():
    import tiktoken
    if tiktoken.__version__ != '0.12.0':
        raise ValueError('Install pinned requirements-context.txt for reproducible token estimates')
    # Validate bundled files BEFORE tiktoken can attempt a network download.
    for filename, expected in VOCABS.values():
        path = HERE / 'tokenizer_cache' / filename
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Missing/corrupt offline tokenizer vocabulary: '+str(path))
    os.environ['TIKTOKEN_CACHE_DIR'] = str(HERE / 'tokenizer_cache')
    return {name:tiktoken.get_encoding(name) for name in VOCABS}

@lru_cache(maxsize=8192)
def text_counts(text):
    # Source code can contain strings that resemble special tokens: treat as text.
    return {name:len(enc.encode(text, disallowed_special=())) for name,enc in encoders().items()}

@lru_cache(maxsize=1)
def policy_document():
    return json.loads(POLICY_PATH.read_text())


def policy(model):
    doc = policy_document()
    if model not in doc['models']:
        raise ValueError('No explicit context budget for '+model)
    p = {**doc['estimator'], **doc['models'][model]}
    for key in ('context_tokens','input_tokens','output_tokens','gateway_context_tokens','gateway_input_tokens','gateway_output_tokens'):
        if p.get(key) is not None and (type(p[key]) is not int or p[key] <= 0):
            raise ValueError('Invalid context policy field: '+key)
    if p['multiplier'] < 1 or p['reserve_tokens'] < 0:
        raise ValueError('Invalid estimation margin')
    return p


def estimate(messages, model, output_limit, strategy='std', temperature=0):
    from gateway_client import request_body
    p = policy(model)
    body = request_body(messages, model, strategy, temperature, output_limit)
    output = body.get('max_output_tokens', body.get('max_tokens'))
    # Whole serialized request includes role/system text, instructions, ALL code,
    # evidence, formatting, and request parameters. JSON tokens are a proxy for
    # provider chat framing, not a claim that raw JSON is fed to the model.
    wire = json.dumps(body, ensure_ascii=True, allow_nan=False)
    counts = text_counts(wire)
    raw = max(counts.values())
    estimated_input = math.ceil(raw * p['multiplier']) + p['reserve_tokens']
    checks = []
    for key, used in [('context_tokens',estimated_input+output),('input_tokens',estimated_input),('output_tokens',output),
                      ('gateway_context_tokens',estimated_input+output),('gateway_input_tokens',estimated_input),('gateway_output_tokens',output)]:
        if p.get(key) is not None:
            checks.append({'limit_kind':key,'limit':p[key],'estimated_use':used,'headroom':p[key]-used})
    return {'model':model,'request_sha256':hashlib.sha256(wire.encode()).hexdigest(),
            'request_json_bytes':len(wire.encode()),'bpe_counts':counts,
            'raw_token_estimate':raw,'estimated_input_with_margin':estimated_input,
            'effective_output_tokens':output,'estimated_total_tokens':estimated_input+output,
            'checks':checks,'status':'needs_capacity_review' if any(c['headroom']<0 for c in checks) else 'within_estimated_budget',
            'endpoint_capacity_verified':False}


def check_context(messages, model, output_limit):
    result = estimate(messages,model,output_limit)
    if result['status'] != 'within_estimated_budget':
        raise ValueError('Estimated context budget exceeded; verify tokenizer/endpoint capacity before changing evidence. No truncation: '+json.dumps(result))
    return messages


def audit_requests(requests, model, output_limit, report_path, strategy='std', temperature=0, enforce=True, extra=None):
    rows=[]
    for job in requests:
        rows.append({**{k:v for k,v in job.items() if k!='messages'},
                     **estimate(job['messages'],model,job.get('output_limit',output_limit),strategy,temperature)})
    groups={}
    for stage in sorted({r.get('arm',r.get('stage','request')) for r in rows}):
        group=[r for r in rows if r.get('arm',r.get('stage','request'))==stage]
        values=sorted(r['estimated_total_tokens'] for r in group)
        groups[stage]={'count':len(group),'max_estimated_total_tokens':values[-1],
                       'p95_estimated_total_tokens':values[math.ceil(.95*len(values))-1],
                       'needs_capacity_review':sum(r['status']!='within_estimated_budget' for r in group)}
    failures=[r for r in rows if r['status']!='within_estimated_budget']
    report={'model':model,'policy':policy(model),'policy_sha256':hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest(),
            'method':'max(cl100k_base, o200k_base) on complete serialized request; multiplicative margin + reserve + effective output',
            'provider_exact_token_count':False,'endpoint_capacity_verified':False,
            'request_count':len(rows),'needs_capacity_review':len(failures),'groups':groups,'requests':rows,**(extra or {})}
    Path(report_path).write_text(json.dumps(report,indent=2))
    print(f'Offline token audit: requests={len(rows)} needs_capacity_review={len(failures)} report={report_path}',flush=True)
    if failures and enforce:
        raise ValueError(f'{len(failures)} requests need capacity verification. Full report: {report_path}. No samples skipped or truncated; this estimate does NOT prove endpoint overflow.')
    return report

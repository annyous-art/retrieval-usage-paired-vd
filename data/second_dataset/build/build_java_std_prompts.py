"""std-setting prompts for the MegaVul Java split: the PrimeVul two-shot prompt (system message and the two
PrimeVul demonstrations copied from a saved GLM-5.1 prompt, unchanged) with the Java target, and mode A with
the top-2 retrieved Java demonstrations (icl/run_prompting_k_icl.py construct_prompts, as for PrimeVul)."""
import json, sys
sys.path.insert(0, 'icl')
from utils import PROMPT_INST
from run_prompting_k_icl import construct_prompts
SIM, OUT = sys.argv[1], sys.argv[2]
tmpl = json.loads(open('outdir/glm-5.1_std_cls_logprobsFalse_fewshotegTrue_baseline_870.jsonl').readline())['messages']
first = json.loads(open('outdir/glm-5.1_std_cls_logprobsFalse_fewshotegTrue_baseline_870.jsonl').readline())
assert tmpl[-1]['content'] == PROMPT_INST.format(func=first['func'])
rows = [json.loads(l) for l in open(SIM)]
with open(f'{OUT}/prompts_two_shot.jsonl', 'w') as f:
    for r in rows:
        f.write(json.dumps({'idx': r['idx'], 'func': r['func'], 'target': r['target'],
                            'messages': tmpl[:-1] + [{'role': 'user', 'content': PROMPT_INST.format(func=r['func'])}]}) + '\n')
with open(f'{OUT}/prompts_A_top2.jsonl', 'w') as f:
    for p in construct_prompts(SIM, 'std_cls', 2, True):
        f.write(json.dumps(p) + '\n')
print('wrote', len(rows))

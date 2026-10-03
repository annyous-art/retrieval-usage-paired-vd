"""Java two-shot prompt with Java demonstrations (reviewer: the PrimeVul two-shot prompt shows C demonstrations
to Java targets). Same system message, message layout, and order as the PrimeVul two-shot prompt (a NO example,
then a YES example); the two demonstrations are drawn once from the MegaVul Java retrieval corpus with
random.Random(20260915): one fixed and one vulnerable function, from different commits, each 400-1,200
characters (the PrimeVul demonstrations have 656 and 993). The same two demonstrations are used for every target.
Writes outdir/java_std/prompts_two_shot_java.jsonl."""
import json, random, sys
sys.path.insert(0, 'icl')
from utils import PROMPT_INST
tmpl = json.loads(open('outdir/glm-5.1_std_cls_logprobsFalse_fewshotegTrue_baseline_870.jsonl').readline())['messages']
train = [json.loads(l) for l in open('data/second_dataset/megavul_java_train_paired_labeled.jsonl')]
ok = lambda r: 400 <= len(r['func']) <= 1200
rng = random.Random(20260915)
no = rng.choice(sorted([r for r in train if int(r['target']) == 0 and ok(r)], key=lambda r: str(r['idx'])))
yes = rng.choice(sorted([r for r in train if int(r['target']) == 1 and ok(r) and r['commit_id'] != no['commit_id']], key=lambda r: str(r['idx'])))
demo = [tmpl[0],
        {'role': 'user', 'content': PROMPT_INST.format(func=no['func'])}, {'role': 'assistant', 'content': 'NO'},
        {'role': 'user', 'content': PROMPT_INST.format(func=yes['func'])}, {'role': 'assistant', 'content': 'YES'}]
rows = [json.loads(l) for l in open('outdir/java_std/prompts_two_shot.jsonl')]
with open('outdir/java_std/prompts_two_shot_java.jsonl', 'w') as f:
    for r in rows:
        f.write(json.dumps({'idx': r['idx'], 'func': r['func'], 'target': r['target'],
                            'demo_idx': [no['idx'], yes['idx']],
                            'messages': demo + [r['messages'][-1]]}) + '\n')
print('demos: NO', no['idx'], no['project'], len(no['func']), 'chars; YES', yes['idx'], yes['project'], len(yes['func']), 'chars;', len(rows), 'prompts')

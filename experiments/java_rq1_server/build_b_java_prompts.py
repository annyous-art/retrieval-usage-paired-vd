"""Mode B (two-shot with retrieved chunks) on the MegaVul Java pairs.

The final user message is built by run_prompting_sliced_rag.format_baseline_compatible_combined_prompt
with the settings of the PrimeVul B configuration (one focus region, three vulnerable and three fixed
chunks, evidence_mode='grouped_no_label'); the system message and the two demonstrations are those of
the Java two-shot prompt (outdir/java_std/prompts_two_shot_java.jsonl), the reference for modes A and B
on Java.  --check rebuilds the PrimeVul B prompts from PrimeVul query results and compares them with
the saved ones.  Offline; no model calls.  Run from the repository root.
"""
import argparse, ast, json, sys
sys.path.insert(0, '.')
import run_prompting_sliced_rag as rp

NUM = ('query_rank', 'index_rank')
FLOAT = ('score', 'score_combined', 'score_cosine', 'index_score', 'retrieval_score')


def norm(x):
    """Server JSON keeps some numbers as strings; the PrimeVul prompts were built from numbers."""
    if isinstance(x, list):
        return [norm(v) for v in x]
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if isinstance(v, str) and k in NUM and v.lstrip('-').isdigit():
                v = int(v)
            elif isinstance(v, str) and (k in FLOAT or k.endswith('_score')):
                try:
                    v = float(v)
                except ValueError:
                    pass
            out[k] = norm(v)
        return out
    return x


def user_message(r):
    qc = r['query_chunks']
    qc = ast.literal_eval(qc) if isinstance(qc, str) else qc
    qc = sorted(norm(qc), key=lambda q: q.get('query_rank', 0))
    return rp.format_baseline_compatible_combined_prompt(query_func=r['query_func'], query_chunks=qc, encoding=None,
                                                         max_query_chunks=1, max_index_matches=3, evidence_mode='grouped_no_label')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', nargs=2, metavar=('QUERY_RESULTS', 'SAVED_B_PROMPTS'))
    ap.add_argument('--query', default='out/java_query/query_results.jsonl', help='B retrieval results of the Java test functions')
    ap.add_argument('--out', default='outdir/java_std/prompts_B_chunks.jsonl')
    a = ap.parse_args()
    if a.check:
        saved = {str(r['query_idx']): r['messages'][-1]['content'] for r in map(json.loads, open(a.check[1]))}
        same = n = 0
        for r in map(json.loads, open(a.check[0])):
            k = str(r['query_idx'])
            if k in saved:
                n += 1; same += user_message(r) == saved[k]
        print(f'PrimeVul check: {same}/{n} rebuilt prompts identical to the saved B prompts')
        return
    two = {str(r['idx']): r for r in map(json.loads, open('outdir/java_std/prompts_two_shot_java.jsonl'))}
    out = open(a.out, 'w')
    n = 0
    for r in map(json.loads, open(a.query)):
        t = two[str(r['query_idx'])]
        msgs = t['messages'][:-1] + [{'role': 'user', 'content': user_message(r)}]
        out.write(json.dumps({'idx': t['idx'], 'func': t['func'], 'target': t['target'], 'demo_idx': t['demo_idx'], 'messages': msgs}) + '\n')
        n += 1
    print(f'wrote {n} Java B prompts to {a.out}')


if __name__ == '__main__':
    main()

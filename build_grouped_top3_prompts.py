"""Build the 'grouped top-3' chunk-retrieval prompts from saved inline chunk-label prompts (no API calls).

Inline chunk-label prompts show the top-3 retrieved chunks of the focus region in rank order,
each with its role, diff label, and function label. The existing grouped prompts show up to three
chunks per role (six to nine in total) and a retrieval score per item, so they differ from inline
in the number of items and fields as well as in placement. The grouped top-3 variant takes exactly
the inline top-3 chunks and the same per-item fields, and only regroups them under the role
headers used by the grouped prompts; everything else in the prompt is unchanged.

The script first rebuilds every saved inline prompt with the original formatter and requires a
byte-identical match, then writes the new prompts. Run from the repository root:
python3 build_grouped_top3_prompts.py --source SRC --out OUT
"""
import argparse
import json
import sys
import types

# The formatter module imports API clients at load time; stub them (no calls are made here).
for name in ('tiktoken', 'tqdm', 'model_api_clients'):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules['tqdm'].tqdm = lambda x, **k: x
sys.modules['model_api_clients'].get_openai_chat = None
sys.modules['model_api_clients'].normalize_usage = None
import run_prompting_sliced_rag as R  # noqa: E402

MAX_Q, MAX_M = 1, 3


def item_lines(k, match):
    return [f'Index {k}:', 'Index Chunk Code:', match.get('index_code_clean', '') or '',
            f'Index Evidence Role: {R.evidence_role_text(R.get_match_evidence_role(match))}',
            f'Index Diff Label: {R.chunk_label_text(R.get_match_chunk_label(match))}',
            f'Index Function Label: {R.binary_label_text(R.get_match_function_label(match))}', '']


def grouped_top3(query_func, query_chunks):
    inline = R.format_query_result_prompt(query_func, query_chunks, None, MAX_Q, MAX_M, 'inline_chunk_label')
    head = inline[:inline.index('Retrieved Evidence:\n')]
    top = [m for c in R.selected_query_chunks(query_chunks, MAX_Q) for m in R.selected_index_matches(c, MAX_M)]
    groups = [('Retrieved Vulnerable Changed Evidence:', [m for m in top if R.get_match_evidence_polarity(m) == 1]),
              ('Retrieved Fixed/Safe Evidence:', [m for m in top if R.get_match_evidence_polarity(m) == 0])]
    unknown = [m for m in top if R.get_match_evidence_polarity(m) is None]
    if unknown:
        groups.append(('Retrieved Unknown-Label Evidence:', unknown))
    lines = []
    for title, ms in groups:
        lines += [title, '']
        if not ms:
            lines += ['None', '']
        for k, m in enumerate(ms, 1):
            lines += item_lines(k, m)
    return inline, head + '\n'.join(lines), top


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', required=True, help='saved inline chunk-label prompt file (evidence only)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.source)]
    out, mismatched, items = [], 0, []
    for r in rows:
        inline, grouped, top = grouped_top3(r['query_func'], r['query_chunks'])
        if inline != r['messages'][-1]['content']:
            mismatched += 1
            continue
        g_codes = sorted(l for l in grouped.split('\n'))
        i_codes = sorted(l for l in inline.split('\n'))
        # Same lines as the inline prompt except the section headers and item numbers.
        strip = lambda ls: sorted(x for x in ls if x.strip() and not x.startswith(('Index 1:', 'Index 2:', 'Index 3:', 'Retrieved Vulnerable', 'Retrieved Fixed', 'Retrieved Unknown', 'Retrieved Evidence:')) and x != 'None')
        assert strip(g_codes) == strip(i_codes), r['query_idx']
        # Keep identifiers and the prompt only; retrieval metadata stays in the source file.
        new = {k: r[k] for k in ('query_idx', 'query_func_id', 'query_target', 'sample_key') if k in r}
        new['messages'] = r['messages'][:-1] + [{'role': 'user', 'content': grouped}]
        new['evidence_mode'] = 'grouped_top3_chunk_label'
        out.append(new)
        items.append(len(top))
    if mismatched:
        raise SystemExit(f'{mismatched} saved inline prompts could not be rebuilt exactly; nothing written')
    with open(a.out, 'w') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'{len(out)} prompts written; items per prompt: {sorted(set(items))}; all saved inline prompts rebuilt exactly')


if __name__ == '__main__':
    main()

"""Export, for every test function, the mode B retrieval key and focus region as a small table
without code (offline; no model calls).

For each function: the SHA-256 of the B evidence key (the focus region and the first three
vulnerable and three fixed index chunks by rank, as rendered by the main B prompt;
rq1_same_items.b_key), and the line numbers (1-based) of the focus region, i.e. the top-ranked
query chunk. Two functions receive the same B items exactly when their key hashes are equal.

  PrimeVul: the query_chunks stored with the B prompts of the paper (GPT-5.5 run; the GLM-5.1
            prompts hold the same chunks).
  MegaVul:  the query results of build_query_chunks_sim_rank.py on the Java chunk index.

Run from the repository root:
  python3 tools/export_b_keys.py PRIMEVUL_B_PROMPTS.jsonl data/b_keys/primevul_b_keys.jsonl.gz
  python3 tools/export_b_keys.py JAVA_QUERY_RESULTS.jsonl data/b_keys/megavul_java_b_keys.jsonl.gz
"""
import ast
import gzip
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, 'experiments/java_rq1_server')
from rq1_same_items import b_key  # noqa: E402

lit = lambda x: ast.literal_eval(x) if isinstance(x, str) else (x or [])


def focus_lines(query_chunks):
    qs = sorted((q for q in lit(query_chunks) if isinstance(q, dict)), key=lambda q: int(q.get('query_rank', 0)))
    if not qs:
        return None
    t = qs[0]
    ch = lit(t['query_chunk']) if 'query_chunk' in t and isinstance(lit(t['query_chunk']), dict) else t
    return sorted(int(x) for x in lit(ch.get('chunk_line_ids') or ch.get('query_chunk_line_ids') or ch.get('raw_line_ids')))


src, dst = sys.argv[1], Path(sys.argv[2])
dst.parent.mkdir(parents=True, exist_ok=True)
n = 0
with gzip.open(dst, 'wt', encoding='utf-8') as out:
    for line in open(src, encoding='utf-8'):
        r = json.loads(line)
        qc = lit(r.get('query_chunks') or r.get('chunks') or [])
        key = b_key(qc)
        out.write(json.dumps({'idx': str(r.get('query_idx', r.get('query_func_id', r.get('idx')))),
                              'b_key_sha256': None if key is None else hashlib.sha256(json.dumps(key).encode()).hexdigest(),
                              'focus_line_ids': focus_lines(qc)}) + '\n')
        n += 1
print('wrote', dst, n, 'functions')

"""Offline analyses (no API calls).

(1) Identical-evidence rate across all retrieval families in the paper: how often
    the vulnerable and the fixed member of a pair receive the same evidence.
(3) Mechanism for chunk RAG: are the two members' focus regions identical, does the
    focus region cover the patch, and how does that relate to identical evidence and P-C?

Pairs touching the two duplicated test indices are excluded (431 resolvable pairs).
Run from the repository root: python3 analysis/fse_revision_20260923/evidence_mechanism.py
"""
import collections
import glob
import json
import re
import sys

from scipy.stats import fisher_exact

sys.path.insert(0, '.')
from evaluate_prompt_outputs import parse_prediction  # noqa: E402

RAW = [json.loads(l) for l in open('data/primevul_test_paired_labeled.jsonl')]
DUP = collections.Counter(str(r['idx']) for r in RAW)
PAIRS = []  # (vulnerable idx, fixed idx)
for i in range(0, len(RAW), 2):
    a, b = RAW[i], RAW[i + 1]
    if DUP[str(a['idx'])] > 1 or DUP[str(b['idx'])] > 1:
        continue
    v, f = (a, b) if int(a['target']) == 1 else (b, a)
    PAIRS.append((str(v['idx']), str(f['idx'])))
OUT = {'pairs': len(PAIRS)}


def rate(same):
    n = len(same)
    return {'identical': sum(same), 'pairs': n, 'rate': round(sum(same) / n, 4) if n else None}


def read_prompts(path):
    return {str(r.get('query_idx', r.get('idx'))): r for r in (json.loads(l) for l in open(path))}


def last_user(r):
    return r['messages'][-1]['content']


# ---------- (1a) chunk RAG: evidence block after the focus regions ----------
def chunk_block(msg):
    f = msg.find('Focus Regions:')
    if f == -1:
        return None
    m = re.search(r'\n\nRetrieved ', msg[f:])
    return msg[f + m.start():] if m else ''


chunk = {}
for p in sorted(glob.glob('outdir/prompt_*glm51_test870/*.jsonl')):
    if 'target_focus' in p or 'patch_contrast' in p:
        continue
    pr = read_prompts(p)
    same = [chunk_block(last_user(pr[v])) == chunk_block(last_user(pr[f])) for v, f in PAIRS]
    chunk[p.split('/')[-2] + '/' + p.split('/')[-1]] = rate(same)
rates = [x['rate'] for x in chunk.values()]
OUT['chunk_rag'] = {'variants': len(chunk), 'min': min(rates), 'max': max(rates), 'per_variant': chunk}

# ---------- (1b) function-level retrieved few-shot: top-2 demonstrations ----------
sim = {}
for l in open('icl/primevul_test_ccpp_with_similar_top5.jsonl'):
    r = json.loads(l)
    sim[str(r['idx'])] = [str(s.get('idx')) for s in (r.get('similar') or []) if isinstance(s, dict)]
OUT['function_level_top2_same_order'] = rate([sim[v][:2] == sim[f][:2] for v, f in PAIRS])
OUT['function_level_top2_same_set'] = rate([set(sim[v][:2]) == set(sim[f][:2]) for v, f in PAIRS])

# ---------- (1c) patch-contrast: retrieved patch-pair evidence ----------
SCORE = re.compile(r'^(Target-to-vulnerable score|Target-to-fixed score|Contrast margin|Closer historical side):.*$', re.M)


def patch_block(msg):
    j = msg.find('Historical Patch-Aware Contrastive Evidence:')
    return msg[j:] if j != -1 else None


pc = read_prompts('outdir/prompt_patch_contrast_vuln_margin_glm51_test870/glm-5.1_std_cls_patch_contrast_fewshotegTrue.jsonl')
blocks = {k: patch_block(last_user(r)) for k, r in pc.items()}
OUT['patch_contrast_byte_identical'] = rate([blocks[v] == blocks[f] for v, f in PAIRS])
OUT['patch_contrast_same_pairs_ignoring_scores'] = rate(
    [SCORE.sub('', blocks[v] or '') == SCORE.sub('', blocks[f] or '') for v, f in PAIRS])

# ---------- (1d) fixed-candidate knowledge experiment: IDF-selected historical pair ----------
cand = {}
for j in (json.loads(l) for l in open('outdir/knowledge_rag_gpt-5.5/detection_jobs.jsonl')):
    if j['arm'] == 'code_pair':
        cand[str(j['idx'])] = tuple(j['candidate_ids'])
OUT['knowledge_candidate_identical'] = rate([cand[v] == cand[f] for v, f in PAIRS])

# ---------- (3) mechanism for chunk RAG ----------
def focus(r):
    """The focus region shown in the prompt. Records keep two candidates, but every
    prompt in these runs renders only the first one."""
    q = r['query_chunks'][0]
    q = q.get('query_chunk', q)
    return q['code_clean'], int(q.get('chunk_label', 0)) == 1


def outcome(pv, pf):
    return {(1, 0): 'P-C', (1, 1): 'P-V', (0, 0): 'P-B', (0, 1): 'P-R'}.get((pv, pf), 'unknown')


MECH = {}
for tag, path in [
    ('positive_weighted', 'outdir/prompt_positive_weighted_rag_glm51_test870/glm-5.1_std_cls_inline_function_label_rag_fewshotegTrue_cls0.jsonl'),
    ('balanced_bc_grouped_no_label', 'outdir/prompt_baseline_compatible_glm51_test870/glm-5.1_std_cls_grouped_no_label_baseline_compatible_fewshotegTrue_cls0.jsonl'),
    ('patch_contrast', 'outdir/prompt_patch_contrast_vuln_margin_glm51_test870/glm-5.1_std_cls_patch_contrast_fewshotegTrue.jsonl'),
]:
    block = patch_block if tag == 'patch_contrast' else chunk_block
    pr = read_prompts(path)
    rows = []
    for v, f in PAIRS:
        (tv, cv), (tf, cf) = focus(pr[v]), focus(pr[f])
        rows.append({
            'focus_same': tv == tf,
            'covers_patch': cv or cf,
            'evidence_same': block(last_user(pr[v])) == block(last_user(pr[f])),
            'outcome': outcome(parse_prediction(pr[v].get('response')), parse_prediction(pr[f].get('response'))),
        })

    def cell(sel):
        sub = [x for x in rows if sel(x)]
        n = len(sub)
        pcn = sum(x['outcome'] == 'P-C' for x in sub)
        return {'pairs': n, 'evidence_same': sum(x['evidence_same'] for x in sub),
                'P-C': pcn, 'P-C_rate': round(pcn / n, 4) if n else None}

    MECH[tag] = {
        'focus_same': cell(lambda x: x['focus_same']),
        'focus_differs': cell(lambda x: not x['focus_same']),
        'focus_covers_patch (either member)': cell(lambda x: x['covers_patch']),
        'focus_misses_patch (both members)': cell(lambda x: not x['covers_patch']),
        'misses_patch_and_focus_same': cell(lambda x: not x['covers_patch'] and x['focus_same']),
        'evidence_same': cell(lambda x: x['evidence_same']),
        'evidence_differs': cell(lambda x: not x['evidence_same']),
        'all': cell(lambda x: True),
    }
    # P-C when evidence is identical vs. when it differs (two-sided Fisher exact test)
    a = [sum(x['outcome'] == 'P-C' for x in rows if x['evidence_same'] == s) for s in (True, False)]
    n = [sum(x['evidence_same'] == s for x in rows) for s in (True, False)]
    MECH[tag]['fisher_pc_same_vs_differs_p'] = round(fisher_exact([[a[0], n[0] - a[0]], [a[1], n[1] - a[1]]])[1], 4)
    MECH[tag]['outcomes_evidence_same'] = dict(collections.Counter(x['outcome'] for x in rows if x['evidence_same']))
    MECH[tag]['outcomes_evidence_differs'] = dict(collections.Counter(x['outcome'] for x in rows if not x['evidence_same']))
OUT['mechanism'] = MECH

json.dump(OUT, open('analysis/fse_revision_20260923/evidence_mechanism.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in OUT.items() if k != 'chunk_rag'}, indent=1))
print('chunk_rag range:', OUT['chunk_rag']['min'], '-', OUT['chunk_rag']['max'], 'over', OUT['chunk_rag']['variants'], 'variants')

"""Frozen paired-input helpers from the existing knowledge experiment."""
import json, hashlib

def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def read(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def known(x):
    return x not in (None, '', 'None', 'null', 'N/A')

def pairs(rows):
    good, excluded = [], []
    for i in range(0, len(rows), 2):
        block = rows[i:i+2]
        reason = None
        if len(block) != 2 or {int(r['target']) for r in block} != {0, 1}:
            reason = 'not_one_positive_one_negative'
        elif any(not known(block[0].get(k)) or block[0].get(k) != block[1].get(k)
                 for k in ('project', 'commit_id', 'cve')):
            reason = 'project_commit_cve_mismatch_or_missing'
        elif all(known(r.get('file_name')) for r in block) and block[0]['file_name'] != block[1]['file_name']:
            reason = 'known_file_mismatch'
        if reason:
            excluded.append({'row_start': i+1, 'reason': reason})
            continue
        ordered = sorted(enumerate(block, i+1), key=lambda p: -int(p[1]['target']))
        good.append({'pair_id': f'p{i//2}', 'members': [dict(r, row_id=n) for n, r in ordered]})
    return good, excluded

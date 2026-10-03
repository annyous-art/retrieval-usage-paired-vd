"""Build a PrimeVul-style paired split from MegaVul (Java, release 2024-04, megavul.json).

Each vulnerable function (is_vul) whose code changes in the fix (ignoring whitespace) gives one
pair: func_before (target 1) followed by func (target 0), in adjacent rows, as in PrimeVul's paired
files. Chronological split by fix-commit date, as in PrimeVul: commits before 2022-07-01 (UTC) form
the training pool, later ones the test split. Test pairs are dropped if either member equals
(ignoring whitespace) a training function or another test function, and training pairs sharing a
CVE with the test split are dropped. Line labels (1 = line deleted from the
vulnerable version / added to the fixed version, from MegaVul's diff_line_info) are kept only for
format compatibility; no prompt uses them.
"""
import ast, datetime, hashlib, json, re, collections, sys
SRC, OUT, CUT = sys.argv[1], sys.argv[2], '2022-07'
norm = lambda s: re.sub(r'\s+', '', s or '')
d = json.load(open(SRC))
vul = [r for r in d if r['is_vul'] in (True, 'True') and norm(r['func_before']) != norm(r['func'])]

def lab(func, lines):
    s = {l.strip() for l in lines if l.strip()}
    return [int(l.strip() in s and bool(l.strip())) for l in func.splitlines()]

def rows(r, n):
    info = r['diff_line_info'] if isinstance(r['diff_line_info'], dict) else ast.literal_eval(r['diff_line_info'])
    cwe = r['cwe_ids'] if isinstance(r['cwe_ids'], list) else ast.literal_eval(r['cwe_ids'])
    base = dict(project=r['repo_name'], commit_id=r['commit_hash'], commit_url=r['git_url'], commit_message=r['commit_msg'],
                file_name=r['file_path'], cwe=str(cwe), cve=r['cve_id'], publish_date=r['publish_date'], commit_date=r['commit_date'], func_name=r['func_name'])
    out = []
    for k, (func, t, lines) in enumerate([(r['func_before'], 1, info['deleted_lines']), (r['func'], 0, info['added_lines'])]):
        out.append(dict(base, idx=str(2 * n + k), target=str(t), func=func,
                        func_hash=str(int(hashlib.md5(func.encode()).hexdigest(), 16)), labels=lab(func, lines)))
    return out

month = lambda r: datetime.datetime.fromtimestamp(int(r['commit_date']), datetime.timezone.utc).strftime('%Y-%m')
train = [r for r in vul if month(r) < CUT]
test = [r for r in vul if month(r) >= CUT]
th = {norm(r['func_before']) for r in train} | {norm(r['func']) for r in train}
test = [r for r in test if norm(r['func_before']) not in th and norm(r['func']) not in th]
c = collections.Counter([norm(r['func_before']) for r in test] + [norm(r['func']) for r in test])
test = [r for r in test if c[norm(r['func_before'])] == 1 and c[norm(r['func'])] == 1]
train = [r for r in train if r['cve_id'] not in {t['cve_id'] for t in test}]
key = lambda r: (int(r['commit_date']), r['repo_name'], r['commit_hash'], r['file_path'], r['func_name'])
stats = {}
for name, part, off in (('train', sorted(train, key=key), 0), ('test', sorted(test, key=key), 100000)):
    with open(f'{OUT}/megavul_java_{name}_paired_labeled.jsonl', 'w') as f:
        for n, r in enumerate(part):
            for row in rows(r, off // 2 + n):
                f.write(json.dumps(row) + '\n')
    stats[name] = {'pairs': len(part), 'commits': len({r['commit_hash'] for r in part}), 'cves': len({r['cve_id'] for r in part}),
                   'projects': len({r['repo_name'] for r in part})}
print(json.dumps(stats))
json.dump(stats, open(f'{OUT}/megavul_java_split_stats.json', 'w'), indent=1)

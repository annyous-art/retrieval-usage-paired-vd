"""Line labels for the MegaVul Java pairs, computed as for PrimeVul.

PrimeVul's labels come from mark_pair_diffs.compute_labels applied to the line numbers that `diff -u`
reports between the vulnerable and the fixed function (test_diff_line_get.py): deleted lines in the
vulnerable version, added lines in the fixed version, protected lines for addition-only fixes, and
blank/brace/comment lines suppressed. The Java pairs were labeled instead by text matching against
MegaVul's diff_line_info (build_megavul_java_pairs.lab), which marks every line with the same text
(e.g. each `}`).  This script recomputes the Java labels with the PrimeVul procedure.

  --check   apply the procedure to the PrimeVul test pairs and compare with their stored labels
  default   write data/second_dataset/megavul_java_{train,test}_paired_labeled_diff.jsonl
Offline; no model calls.  Run from the repository root.
"""
import argparse, json, os, re, subprocess, sys, tempfile
sys.path.insert(0, '.')
import mark_pair_diffs as mpd


def diff_u(old, new):
    """Line numbers (1-based) deleted from `old` and added in `new` according to `diff -u`."""
    with tempfile.TemporaryDirectory() as d:
        fa, fb = os.path.join(d, 'old'), os.path.join(d, 'new')
        open(fa, 'w').write(old if old.endswith('\n') else old + '\n')
        open(fb, 'w').write(new if new.endswith('\n') else new + '\n')
        out = subprocess.run(['diff', '-u', fa, fb], capture_output=True, text=True).stdout.splitlines()
    o = n = 1
    deleted, added = [], []
    for line in out:
        if line.startswith(('--- ', '+++ ')):
            continue
        m = re.match(r'@@ -(\d+),?(\d*) \+(\d+),?(\d*)', line)
        if m:
            o, n = int(m.group(1)), int(m.group(3))
            continue
        if line.startswith('-'):
            deleted.append(o); o += 1
        elif line.startswith('+'):
            added.append(n); n += 1
        elif line.startswith(' '):
            o += 1; n += 1
    return deleted, added


def relabel(rows):
    out = []
    for i in range(0, len(rows) - 1, 2):
        a, b = dict(rows[i]), dict(rows[i + 1])
        v, f = (a, b) if int(a['target']) == 1 else (b, a)
        deleted, added = diff_u(v['func'], f['func'])
        lv, lf, *_ = mpd.compute_labels(v['func'].splitlines(), f['func'].splitlines(), changed_lines_a=deleted, changed_lines_b=added)
        v['labels'], f['labels'] = lv, lf
        out += [a, b]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    if a.check:
        rows = [json.loads(l) for l in open('data/primevul_test_paired_labeled.jsonl')]
        new = relabel(rows)
        same = sum(r['labels'] == n['labels'] for r, n in zip(rows, new))
        lines = sum(len(r['labels']) for r in rows)
        agree = sum(x == y for r, n in zip(rows, new) for x, y in zip(r['labels'], n['labels']))
        print(f'PrimeVul test: identical labels for {same}/{len(rows)} functions; {agree}/{lines} lines agree ({agree/lines:.2%})')
        return
    for split in ('train', 'test'):
        src = f'data/second_dataset/megavul_java_{split}_paired_labeled.jsonl'
        rows = [json.loads(l) for l in open(src)]
        new = relabel(rows)
        dst = src.replace('.jsonl', '_diff.jsonl')
        with open(dst, 'w') as f:
            for r in new:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        old1 = sum(sum(int(x) for x in r['labels']) for r in rows); new1 = sum(sum(r['labels']) for r in new)
        print(f'{split}: {len(new)} functions; labeled lines {old1} (text matching) -> {new1} (diff -u); wrote {dst}')


if __name__ == '__main__':
    main()

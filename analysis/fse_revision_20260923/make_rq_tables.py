"""Write the LaTeX tables of the paper's results section from the JSON results of the analysis scripts.

Outputs (default directory tables/; pass another directory as the first argument):
  tab_intermediate.tex  Table 2, retrieval results for the two members of each pair (RQ1)
                        rq2_missing_rows.json, java_rq1_retrieval.json, rq2_similarity_scores.json,
                        e_rq2_retrieval.json, rq2_match_by_member.json
  tab_rq1_main.tex      Table 3, pair outcomes of the configurations for GPT-5.5 and GLM-5.1
                        rq1_rq3_modes.json, gpt_std_prompts.json, glm_std_prompts.json, vulrag_comparison.json
  tab_rq1_cross.tex     Table 4, one configuration per mode for Claude Opus 4.7 and DeepSeek-V4-Pro (rq1_rq3_modes.json)
  tab_bias.tex          Table 5, context controls (bias_controls.json)
Values are rounded half away from zero (3.05 -> 3.1).

Run from the repository root: python3 analysis/fse_revision_20260923/make_rq_tables.py [OUT_DIR]
"""
import json
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

A = Path('analysis/fse_revision_20260923')
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else 'tables')
OUT.mkdir(parents=True, exist_ok=True)
load = lambda name: json.load(open(A / name))
R = load('rq1_rq3_modes.json')['models']
STD = {'gpt-5.5': load('gpt_std_prompts.json')['rows'], 'glm-5.1': load('glm_std_prompts.json')['rows']}
VC = load('vulrag_comparison.json')
VR = {'gpt-5.5': VC['methods']['Vul-RAG (E, end to end)'], 'glm-5.1': VC['glm-5.1']['retrieved knowledge']}
BC = load('bias_controls.json')


def rnd(x, places='0.1', scale=100):
    """Round x * scale half away from zero."""
    return str((Decimal(str(x)) * scale).quantize(Decimal(places), rounding=ROUND_HALF_UP))


pct = lambda x: rnd(x).replace('-', '$-$')
f1 = lambda x: rnd(x, '0.001', 1)


def write(name, lines):
    (OUT / name).write_text('\n'.join(lines) + '\n')


def intermediate_table():
    MR = load('rq2_missing_rows.json')['instances']
    JR = load('java_rq1_retrieval.json')
    SS = load('rq2_similarity_scores.json')['runs']
    ER = load('e_rq2_retrieval.json')['variants']
    MB = load('rq2_match_by_member.json')['rows']
    java = lambda k: JR['same_items'][k]['same_pct']  # same items on the 620 MegaVul Java pairs
    # the queries built from the function's code first, then the LLM-written descriptions (None separates the two)
    rows = [  # mode, configuration, query, same, same on MegaVul, close, match row
        ('A', 'Two demonstrations', 'function', MR['A function-level top-2']['same_items_set'][0], java('A two demonstrations'),
         MR['A function-level top-2']['score']['within_below_between_median'][0], 'A function-level top-2'),
        ('B', r'Code chunks$^\dagger$', 'focus region', JR['focus']['PrimeVul']['same_B_items_all_pairs_pct'] / 100, java('B retrieved chunks'),
         SS['B balanced BC']['top1']['within_below_between_median'], 'B balanced grouped BC'),
        ('C, D', 'Candidate', 'identifiers', MR['Fixed candidate']['same_items_set'][0], java('C, D candidate'),
         MR['Fixed candidate']['score']['within_below_between_median'][0], 'C/D fixed candidate'),
        ('E', 'CWE entries', 'function', ER['cwe_code']['same_top3_set'][0], java('E CWE entries, code query'),
         ER['cwe_code']['score_top1']['within_below_between_median'][0], 'E CWE entries, code'),
        None,
        ('D', 'Vul-RAG', 'description, code', MR['Vul-RAG top-3 knowledge']['same_items_set'][0], None,  # not run on MegaVul
         MR['Vul-RAG top-3 knowledge']['score']['within_below_between_median'][0], 'D Vul-RAG top-3 knowledge'),
        ('E', 'CWE entries', 'description', ER['cwe_desc']['same_top3_set'][0], java('E CWE entries, description query'),
         ER['cwe_desc']['score_top1']['within_below_between_median'][0], 'E CWE entries, description'),
        ('E', 'CVE descriptions', 'description', ER['cve_desc']['same_top3_set'][0], java('E CVE descriptions'),
         ER['cve_desc']['score_top1']['within_below_between_median'][0], 'E CVE descriptions'),
    ]
    lines = [r'\begin{table*}[t]', r'\centering',
             r"\caption{What the two member functions of a pair retrieve, in percent of pairs, with the measures of Section~\ref{sec:rq1-cmp}. Retrieval does not depend on the detection model. \emph{Same items} and \emph{close scores} compare the two member functions with each other. \emph{Match} compares the retrieved items with the pair's CWE. Same items is given for the 431 PrimeVul pairs and for the 620 Java pairs of MegaVul, where Vul-RAG is not run. The other columns are for PrimeVul. $^\dagger$For B, a match means that the focus region contains a line changed by the fix.}",
             r'\label{tab:intermediate}', r'\footnotesize', r'\setlength{\tabcolsep}{3.5pt}', r'\begin{tabular}{lllrrrrrrr}', r'\toprule',
             r" & & & \multicolumn{2}{c}{Same items} & Close & \multicolumn{4}{c}{Match with the pair's CWE} \\", r'\cmidrule(lr){4-5}\cmidrule(lr){7-10}',
             r'Mode & Configuration & Query & PrimeVul & MegaVul & scores & Either & Vulnerable & Fixed & Baseline \\', r'\midrule']
    for r in rows:
        if r is None:
            lines.append(r'\midrule'); continue
        mode, inst, query, same, jv, close, key = r
        m = MB[key]
        # match for either member of a pair; B checks the focus regions against the changed lines, so its baseline is
        # the expected rate for two randomly chosen regions
        either, base = ((m['either_member'], m['either_member_random_expected']) if 'either_member' in m else (m['own_pair'], m['other_pair']))
        lines.append(f"{mode} & {inst} & {query} & {pct(same)} & {'--' if jv is None else f'{jv:.1f}'} & {pct(close)} & {either:.1f} & {m['vulnerable']:.1f} & {m['fixed']:.1f} & {base:.1f} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    write('tab_intermediate.tex', lines)


def net(m, pairs=431):
    """Net P-C (P-C minus P-R) as a fraction, from the pair counts, so that rounding the stored rates does not add up."""
    return (round(m['P-C'] * pairs) - round(m['P-R'] * pairs)) / pairs


def outcome(model, key):
    """(F1, P-C, net P-C, P-V, P-B, YES) of a configuration on the 431 resolvable pairs."""
    if key == 'zero_shot':
        m = STD[model]['Zero-shot']
    elif key == 'vulrag':
        m = VR[model]
    else:
        m = R[model]['methods'][key]
    return m.get('F1'), m['P-C'], net(m), m['P-V'], m['P-B'], m['YES']


def main_table():
    # a string is the heading of a group of rows; the first row of a group is its prompt without retrieved content
    rows = [('', 'Zero-shot', 'zero_shot'),
            'Two-shot prompt',
            ('', 'Preset demonstrations, no retrieval', 'base'), ('A', 'Two demonstrations', 'A'), ('B', 'Code chunks', 'B_bc'),
            'Evidence-slot prompt',
            ('', 'Empty slot, no retrieval', 'no_retrieval'), ('C', 'Code pair', 'code_pair'), ('D', 'Knowledge', 'knowledge'),
            ('E', 'CWE entries, code query', 'cwe_code'), ('', 'CWE entries, description query', 'cwe_desc'),
            ('', 'CVE descriptions, description query', 'cve_desc'),
            'Own decision procedure',
            ('D', r'Vul-RAG$^\dagger$', 'vulrag')]
    lines = [r'\begin{table*}[t]', r'\centering',
             r"\caption{Pair outcomes on the 431 resolvable pairs for the two main models, in percent of pairs. Net is net P-C, YES is the YES rate, and F1 is on a 0--1 scale. $^\dagger$GLM-5.1 receives GPT-5.5's Vul-RAG knowledge base and queries.}",
             r'\label{tab:rq1-main}', r'\footnotesize', r'\setlength{\tabcolsep}{2.3pt}', r'\begin{tabular}{llrrrrrrrrrrrr}', r'\toprule',
             r' & & \multicolumn{6}{c}{GPT-5.5} & \multicolumn{6}{c}{GLM-5.1} \\', r'\cmidrule(lr){3-8}\cmidrule(lr){9-14}',
             r'Mode & Configuration & F1 & P-C & Net & P-V & P-B & YES & F1 & P-C & Net & P-V & P-B & YES \\', r'\midrule']
    for r in rows:
        if isinstance(r, str):
            lines += [r'\midrule', rf'\multicolumn{{14}}{{l}}{{\emph{{{r}}}}} \\']; continue
        mode, label, key = r
        cells = []
        for model in ('gpt-5.5', 'glm-5.1'):
            F1, *rest = outcome(model, key)
            cells += [f1(F1)] + [pct(x) for x in rest]
        lines.append(f'{mode} & {label} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    write('tab_rq1_main.tex', lines)


def cross_table():
    rows = ['Two-shot prompt',
            ('', 'Preset demonstrations, no retrieval', 'base'), ('A', 'Two demonstrations', 'A'), ('B', 'Code chunks', 'B_bc'),
            'Evidence-slot prompt',
            ('', 'Empty slot, no retrieval', 'no_retrieval'), ('C', 'Code pair', 'code_pair'), ('D', 'Knowledge', 'knowledge'),
            ('E', 'CWE entries, description query', 'cwe_desc')]
    lines = [r'\begin{table}[t]', r'\centering',
             r'\caption{Cross-model check: one configuration per mode with Claude Opus 4.7 and DeepSeek-V4-Pro on the 431 pairs. Values in percent, with Net and YES as in Table~\ref{tab:rq1-main}.}',
             r'\label{tab:rq1-cross}', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{llrrrrrrrrrr}', r'\toprule',
             r' & & \multicolumn{5}{c}{Claude Opus 4.7} & \multicolumn{5}{c}{DeepSeek-V4-Pro} \\', r'\cmidrule(lr){3-7}\cmidrule(lr){8-12}',
             r'Mode & Configuration & P-C & Net & P-V & P-B & YES & P-C & Net & P-V & P-B & YES \\']
    for r in rows:
        if isinstance(r, str):
            lines += [r'\midrule', rf'\multicolumn{{12}}{{l}}{{\emph{{{r}}}}} \\']; continue
        mode, label, key = r
        cells = [pct(x) for model in ('claude-opus-4-7', 'deepseek-v4-pro-0813') for x in outcome(model, key)[1:]]
        lines.append(f'{mode} & {label} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    write('tab_rq1_cross.tex', lines)


def bias_table():
    MODELS = [('GLM-5.1', 'GLM-5.1'), ('GPT-5.5', 'GPT-5.5'), ('Claude Opus 4.7', 'Claude'), ('DeepSeek-V4-Pro', 'DeepSeek')]
    GROUPS = [('Without code', [('No-code prior', 'No context'), ('No-code + PrimeVul two-shot', 'PrimeVul two-shot'),
                                ('No-code + flipped two-shot', 'Flipped two-shot'), ('Skeleton only', 'Function skeleton'),
                                ('Metadata without CWE', 'Metadata without CWE'), ('Metadata with CWE', 'Metadata with CWE'), ('CWE only', 'CWE only')]),
              ('With code', [('Code + metadata without CWE', 'Metadata without CWE'), ('Code + metadata with CWE', 'Metadata with CWE'),
                             ('Code + PrimeVul two-shot', 'PrimeVul two-shot'), ('Code + flipped two-shot', 'Flipped two-shot')])]
    left = [(g, lab, BC['rows'][v][m]['pairs']) for g, rows in GROUPS for v, lab in rows for m, _ in MODELS
            if m in BC['rows'][v] and BC['rows'][v][m]['pairs'] < BC['pairs']]
    words = {1: 'one', 2: 'two', 3: 'three', 4: 'four'}
    note = (f" Claude Opus 4.7 left some prompts unanswered, so {words[len(left)]} of its rows use "
            f"{min(n for *_, n in left)} to {max(n for *_, n in left)} pairs.") if left else ''
    lines = [r'\begin{table}[t]', r'\centering',
             r'\caption{Context controls on the 431 resolvable pairs: the YES rate and P-C when the prompt carries context without the code, and the same context with the code. Without the code, both member functions of a pair receive the same prompt. Function skeleton: the signature with the body removed, and the line and character counts.' + note
             + r' The controls are separate runs, so the two-shot row with code is a second run of the preset demonstrations of Tables~\ref{tab:rq1-main} and~\ref{tab:rq1-cross}.}',
             r'\label{tab:bias-controls}', r'\footnotesize', r'\setlength{\tabcolsep}{3.5pt}', r'\begin{tabular}{lrrrrrrrr}', r'\toprule',
             r' & ' + ' & '.join(rf'\multicolumn{{2}}{{c}}{{{short}}}' for _, short in MODELS) + r' \\',
             r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}',
             r'Context & ' + ' & '.join(['YES & P-C'] * 4) + r' \\']
    for g, rows in GROUPS:
        lines += [r'\midrule', rf'\multicolumn{{9}}{{l}}{{\emph{{{g}}}}} \\']
        for v, lab in rows:
            cells = [pct(BC['rows'][v][m][k]) for m, _ in MODELS for k in ('YES', 'P-C')]
            lines.append(rf'\quad {lab} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    write('tab_bias.tex', lines)


intermediate_table()
main_table()
cross_table()
bias_table()
print('wrote', *(str(p) for p in sorted(OUT.glob('tab_*.tex'))))

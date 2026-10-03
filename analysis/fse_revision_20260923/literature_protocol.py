"""Counts for the literature protocol that grounds the usage-mode taxonomy (no API calls).

Inputs (in literature/):
  s2_bulk_search_20260925.json  Semantic Scholar bulk search, fixed query, 2023-2026
  s2_forward_citations_20260924.json        forward citations of Vul-RAG, PrimeVul, LLM4Vuln
  included_studies.csv                     coded list of included studies (multi-label A-E, eval)

Codes: A similar-code demonstrations, B code-chunk retrieval, C vulnerable/fixed pairs as
evidence, D knowledge-level retrieval, E CWE/CVE entry retrieval; eval = the study evaluates retrieval rather than (only) proposing a detector.

Run from the repository root:
python3 analysis/fse_revision_20260923/literature_protocol.py
"""
import csv
import json
import re

F = 'literature/'
bulk = json.load(open(F + 's2_bulk_search_20260925.json'))
db = bulk['papers']
cites = json.load(open(F + 's2_forward_citations_20260924.json'))
rows = list(csv.DictReader(open(F + 'included_studies.csv', encoding='utf-8')))


def txt(p):
    return ((p.get('title') or '') + ' ' + (p.get('abstract') or '')).lower()


topic = re.compile(r'vulnerab\w* detect|detect\w* vulnerab|vulnerability identification|vulnerable code detection|vulnerability discovery|vulnerab\w* classif', re.I)
excl_title = re.compile(r'smart contract|solidity|blockchain|malware|intrusion|network traffic|phishing|jailbreak|prompt injection|web application firewall|iot device|android app|fuzz|repair|patch generation|fix generation|code generation|secure code generation|penetration|exploit generation|binary|firmware|log anomaly|spam|fake news|autonomous driving|image', re.I)
retr = re.compile(r'retriev|\brag\b|in-context|few-shot|demonstration|exemplar|knowledge base|knowledge graph|similar (code|function|example)|k-?nn|nearest', re.I)
s1 = [p for p in db.values() if topic.search(txt(p))]
s2 = [p for p in s1 if not excl_title.search(p.get('title') or '')]
s3 = [p for p in s2 if retr.search(txt(p))]

cite_topic = re.compile(r'vulnerab|cwe|security bug|secure code review|patch', re.I)
cite_retr = re.compile(r'retriev|\bRAG\b|-RAG|RAG-|in-context|few-shot|knowledge|demonstrat|exemplar|example|context|similar|history|historical|specification|memory', re.I)
c_screen = [v for v in cites.values() if cite_topic.search(v['title']) and cite_retr.search(v['title'])]


def found_in(title_match, pool):
    t = title_match.lower()
    return any(t in (p.get('title') or '').lower() for p in pool)


for r in rows:
    r['in_db'] = found_in(r['title_match'], s3)
    r['in_cites'] = found_in(r['title_match'], c_screen)
codes = ['A', 'B', 'C', 'D', 'E']
OUT = {
    'database_search': {'engine': 'Semantic Scholar bulk search', 'query': bulk['query'], 'years': bulk['year'],
                        'retrieved_at': bulk['retrieved_at'], 'identified': len(db),
                        'topic_vulnerability_detection': len(s1), 'after_title_exclusion': len(s2),
                        'mention_retrieval_icl_or_knowledge': len(s3)},
    'citation_tracking': {'seeds': ['Vul-RAG', 'PrimeVul', 'LLM4Vuln'], 'citing_papers': len(cites),
                          'title_screened_candidates': len(c_screen)},
    'included': len(rows),
    'included_found_by_database_search': sum(r['in_db'] for r in rows),
    'included_found_by_citation_tracking': sum(r['in_cites'] for r in rows),
    'included_found_by_either': sum(r['in_db'] or r['in_cites'] for r in rows),
    'included_only_by_venue_lists_or_manual_search': [r['key'] for r in rows if not (r['in_db'] or r['in_cites'])],
    'per_code_multilabel': {c: sum(int(r[c]) for r in rows) for c in codes},
    'compared_modes_A_to_D_any': sum(any(int(r[c]) for c in 'ABCD') for r in rows),
    'only_E': sum(int(r['E']) == 1 and not any(int(r[c]) for c in 'ABCD') for r in rows),
    'unclassified': [r['key'] for r in rows if not any(int(r[c]) for c in codes)],
    'evaluation_studies': sum(int(r['eval']) for r in rows),
    'multi_label_studies': [r['key'] for r in rows if sum(int(r[c]) for c in codes) > 1],
    'by_year': {y: sum(r['year'] == y for r in rows) for y in sorted({r['year'] for r in rows})},
}
json.dump(OUT, open('analysis/fse_revision_20260923/literature_protocol.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(OUT, ensure_ascii=False, indent=1))

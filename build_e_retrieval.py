"""Build the retrieval results of the CWE/CVE-entry usage mode (E) for the 870 test functions.

No LLM calls. Three variants, each storing the top-10 entries per test function
(the prompts use the top-3):
  X2 cwe_code : query = target function code, corpus = MITRE CWE weakness entries,
                encoder = CodeT5+ 110M embedding (trained for text-code retrieval), following
                the code-to-CWE matching of MIT WAITI D1. GraphCodeBERT, used for our chunk
                retrieval, is not trained to match code against text: zero-shot it maps 384 of
                the 870 functions to the same top-1 entry, so it is not used here.
  X3 cwe_desc : query = the LLM-generated purpose + behaviour description produced in the
                Vul-RAG run (queries.json, reused, not regenerated), corpus = the same CWE
                entries, encoder = SBERT all-mpnet-base-v2, following MIT WAITI D2.
  X4 cve_desc : query = as X3, corpus = CVE descriptions of the PrimeVul training split,
                excluding every CVE that also occurs in the test split. The prompt shows only
                the CVE description (no CVE identifier, no CWE label).
The CWE corpus keeps weakness entries whose Applicable_Platforms list C or C++, a
language-independent class, or no platform at all; entries specific to other languages or
technologies (for example Java autoboxing) are dropped, as are the members of the Hardware
Design view (CWE-1194). Deprecated entries are dropped.
Test labels (target, CWE, CVE) are never used to build queries or corpora.

Run from the repository root:
python3 build_e_retrieval.py --cwe-xml data/e_retrieval/cwec_v4.20.xml --out data/e_retrieval
"""
import argparse
import ast
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

TOPN = 10
EXT_WORDS = 150


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def clean(text):
    return re.sub(r'\s+', ' ', text or '').strip()


def first_sentence(text):
    m = re.match(r'(.+?[.!?])(\s|$)', text)
    return m.group(1) if m else text


def applies_to_c(w, ns):
    langs = [(x.attrib.get('Name'), x.attrib.get('Class')) for x in w.iterfind('c:Applicable_Platforms/c:Language', ns)]
    techs = list(w.iterfind('c:Applicable_Platforms/c:Technology', ns))
    if not langs and not techs:
        return True
    return any(n in ('C', 'C++') for n, _ in langs) or any(c in ('Not Language-Specific', 'Compiled') for _, c in langs)


def load_cwe(path):
    root = ET.parse(path).getroot()
    ns = {'c': root.tag.split('}')[0].strip('{')}
    version = root.attrib.get('Version')
    hardware = {m.attrib['CWE_ID'] for m in root.iterfind('.//c:Categories/c:Category/c:Relationships/c:Has_Member', ns)
                if m.attrib.get('View_ID') == '1194'}
    rows = []
    for w in root.iterfind('.//c:Weaknesses/c:Weakness', ns):
        if w.attrib.get('Status') == 'Deprecated' or w.attrib['ID'] in hardware or not applies_to_c(w, ns):
            continue
        desc = clean(''.join(w.find('c:Description', ns).itertext()))
        ext_el = w.find('c:Extended_Description', ns)
        ext = clean(''.join(ext_el.itertext())) if ext_el is not None else ''
        words = ext.split()
        ext_t = ' '.join(words[:EXT_WORDS]) + (' ...' if len(words) > EXT_WORDS else '')
        cid = 'CWE-' + w.attrib['ID']
        name = w.attrib['Name']
        full = f'{cid}: {name}\nDescription: {desc}' + (f'\nExtended description: {ext_t}' if ext_t else '')
        rows.append({'id': cid, 'name': name, 'description': desc, 'extended_description': ext_t,
                     'embed_text': f'{cid}: {name}. {desc} {ext_t}'.strip(),
                     'prompt_full': full,
                     'prompt_short': f'{cid}: {name}\nDescription: {first_sentence(desc)}'})
    return version, rows


def parse_cwes(v):
    try:
        x = ast.literal_eval(v) if isinstance(v, str) else v
    except (ValueError, SyntaxError):
        x = [v]
    return [str(c) for c in (x if isinstance(x, list) else [x]) if c]


def load_cve(train_path, test_path, cwe_names):
    test_cves = {json.loads(l).get('cve') for l in open(test_path)}
    by, removed = {}, set()
    for l in open(train_path):
        r = json.loads(l)
        c = r.get('cve')
        if c in test_cves:
            removed.add(c)
            continue
        if not c or not r.get('cve_desc'):
            continue
        e = by.setdefault(c, {'id': c, 'description': clean(r['cve_desc']), 'cwes': []})
        for w in parse_cwes(r.get('cwe')):
            if w not in e['cwes']:
                e['cwes'].append(w)
    rows = []
    for c in sorted(by):
        e = by[c]
        # Only the description is shown: no CVE identifier (the model could recall a specific
        # advisory) and no CWE label (PrimeVul and current NVD mappings disagree for some CVEs).
        e['prompt_full'] = f'Vulnerability description: {e["description"]}'
        e['embed_text'] = e['description']
        rows.append(e)
    return rows, len(removed - {None})


def encode_codet5p(texts, model_id, batch=16, max_length=512):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            t = tok(texts[i:i + batch], return_tensors='pt', padding=True, truncation=True, max_length=max_length)
            out.append(model(t['input_ids'], attention_mask=t['attention_mask']).numpy())
    v = np.concatenate(out)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def encode_sbert(texts, model_id):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_id).encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)


def topn(q, d, ids):
    s = q @ d.T
    order = np.argsort(-s, axis=1, kind='stable')[:, :TOPN]
    return [{'ids': [ids[j] for j in o], 'scores': [round(float(s[i, j]), 6) for j in o]} for i, o in enumerate(order)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cwe-xml', required=True)
    ap.add_argument('--targets', default='outdir/knowledge_rag_gpt-5.5/targets.jsonl')
    ap.add_argument('--queries', default='outdir/vulrag_gpt-5.5_first_run/queries.json')
    ap.add_argument('--train', default='data/primevul_train_paired_labeled.jsonl')
    ap.add_argument('--test', default='data/primevul_test_paired_labeled.jsonl')
    ap.add_argument('--code-model', default='Salesforce/codet5p-110m-embedding')
    ap.add_argument('--text-model', default='sentence-transformers/all-mpnet-base-v2')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    version, cwe = load_cwe(a.cwe_xml)
    cve, n_test_cves = load_cve(a.train, a.test, {c['id']: c['name'] for c in cwe})
    targets = [json.loads(l) for l in open(a.targets)]
    desc = {q['row_id']: clean(q['purpose']).strip('"') + '\n' + q['function'] for q in json.load(open(a.queries))}
    rows = sorted(t['row_id'] for t in targets)
    assert set(rows) == set(desc), 'queries.json must cover every target row'
    code = {t['row_id']: t['func'] for t in targets}

    for name, data in (('cwe_entries.jsonl', cwe), ('cve_entries.jsonl', cve)):
        with open(out / name, 'w') as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

    cwe_ids = [c['id'] for c in cwe]
    cve_ids = [c['id'] for c in cve]
    d_cwe_code = encode_codet5p([c['embed_text'] for c in cwe], a.code_model)
    q_code = encode_codet5p([code[r] for r in rows], a.code_model)
    d_cwe_text = encode_sbert([c['embed_text'] for c in cwe], a.text_model)
    d_cve_text = encode_sbert([c['embed_text'] for c in cve], a.text_model)
    q_desc = encode_sbert([desc[r] for r in rows], a.text_model)
    variants = {'cwe_code': topn(q_code, d_cwe_code, cwe_ids),
                'cwe_desc': topn(q_desc, d_cwe_text, cwe_ids),
                'cve_desc': topn(q_desc, d_cve_text, cve_ids)}
    for v, res in variants.items():
        with open(out / f'retrieval_{v}.jsonl', 'w') as f:
            for r, x in zip(rows, res):
                f.write(json.dumps({'row_id': r, **x}) + '\n')

    manifest = {
        'cwe_list_version': version, 'cwe_weakness_entries': len(cwe),
        'cve_entries': len(cve), 'test_cves_excluded_from_corpus': n_test_cves,
        'target_rows': len(rows), 'top_n_stored': TOPN, 'top_k_in_prompt': 3,
        'extended_description_words': EXT_WORDS,
        'code_model': a.code_model, 'text_model': a.text_model,
        'code_encoding': 'CodeT5+ 110M embedding output, max 512 tokens, L2-normalised, inner product',
        'cwe_filter': 'not deprecated; not in the Hardware Design view (CWE-1194); Applicable_Platforms lists C/C++, Not Language-Specific, Compiled, or no platform',
        'inputs_sha256': {k: sha(p) for k, p in (('cwe_xml', a.cwe_xml), ('targets', a.targets),
                                                   ('queries', a.queries), ('train', a.train), ('test', a.test))},
        'outputs_sha256': {p.name: sha(p) for p in sorted(out.glob('*.jsonl'))},
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k not in ('inputs_sha256', 'outputs_sha256')}, indent=1))


if __name__ == '__main__':
    main()

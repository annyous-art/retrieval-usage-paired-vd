"""Mode E with code queries on the MegaVul Java test functions (RQ1 only, no LLM calls).

Same procedure as variant X2 of build_e_retrieval.py: the query is the target function, encoded
with CodeT5+ 110M embedding (max 512 tokens, L2-normalised), matched by inner product against the
MITRE CWE weakness entries, top-10 stored (prompts use the top-3). The only change is the platform
filter: entries are kept when Applicable_Platforms lists Java, a language-independent class
(Not Language-Specific), or no platform at all, instead of C/C++. Deprecated entries and members of
the Hardware Design view (CWE-1194) are dropped, as before.

Run from the package root:
  python3 experiments/java_rq1_server/e_cwe_code_java.py --cwe-xml data/e_retrieval/cwec_v4.20.xml \
      --test data/second_dataset/megavul_java_test_paired_labeled.jsonl --out out/java_e
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import build_e_retrieval as be  # noqa: E402


def encode_codet5p(texts, model_id, batch=16, max_length=512):
    """CodeT5+ encoder as in build_e_retrieval, tolerant of newer transformers releases: the custom
    CodeT5pEmbeddingConfig lacks attributes such as is_decoder that T5Stack now reads, so missing
    ones are filled with the defaults of a plain T5Config before the model is built."""
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer, T5Config
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    base = T5Config()
    for k in ('is_decoder', 'use_cache', 'is_encoder_decoder', 'add_cross_attention', 'tie_word_embeddings',
              'output_attentions', 'output_hidden_states', 'use_return_dict', 'dense_act_fn', 'is_gated_act'):
        try:
            getattr(cfg, k)
        except AttributeError:
            setattr(cfg, k, getattr(base, k))
    cfg.is_decoder = False
    model = AutoModel.from_pretrained(model_id, config=cfg, trust_remote_code=True).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            t = tok(texts[i:i + batch], return_tensors='pt', padding=True, truncation=True, max_length=max_length)
            out.append(model(t['input_ids'], attention_mask=t['attention_mask']).numpy())
    v = np.concatenate(out)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def applies_to_java(w, ns):
    langs = [(x.attrib.get('Name'), x.attrib.get('Class')) for x in w.iterfind('c:Applicable_Platforms/c:Language', ns)]
    techs = list(w.iterfind('c:Applicable_Platforms/c:Technology', ns))
    if not langs and not techs:
        return True
    return any(n == 'Java' for n, _ in langs) or any(c == 'Not Language-Specific' for _, c in langs)


def load_cwe_java(path):
    root = ET.parse(path).getroot()
    ns = {'c': root.tag.split('}')[0].strip('{')}
    hardware = {m.attrib['CWE_ID'] for m in root.iterfind('.//c:Categories/c:Category/c:Relationships/c:Has_Member', ns)
                if m.attrib.get('View_ID') == '1194'}
    rows = []
    for w in root.iterfind('.//c:Weaknesses/c:Weakness', ns):
        if w.attrib.get('Status') == 'Deprecated' or w.attrib['ID'] in hardware or not applies_to_java(w, ns):
            continue
        desc = be.clean(''.join(w.find('c:Description', ns).itertext()))
        ext_el = w.find('c:Extended_Description', ns)
        ext = be.clean(''.join(ext_el.itertext())) if ext_el is not None else ''
        words = ext.split()
        ext_t = ' '.join(words[:be.EXT_WORDS]) + (' ...' if len(words) > be.EXT_WORDS else '')
        cid = 'CWE-' + w.attrib['ID']
        rows.append({'id': cid, 'name': w.attrib['Name'], 'embed_text': f"{cid}: {w.attrib['Name']}. {desc} {ext_t}".strip()})
    return root.attrib.get('Version'), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cwe-xml', required=True)
    ap.add_argument('--test', required=True)
    ap.add_argument('--code-model', default='Salesforce/codet5p-110m-embedding')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    version, cwe = load_cwe_java(a.cwe_xml)
    tests = [json.loads(l) for l in open(a.test)]
    d = encode_codet5p([e['embed_text'] for e in cwe], a.code_model)
    q = encode_codet5p([t['func'] for t in tests], a.code_model)
    ids = [e['id'] for e in cwe]
    with open(out / 'e_cwe_code.jsonl', 'w') as f:
        for t, r in zip(tests, be.topn(q, d, ids)):
            f.write(json.dumps({'idx': t['idx'], 'ids': r['ids'], 'scores': r['scores']}) + '\n')
    (out / 'manifest.json').write_text(json.dumps({'cwe_version': version, 'cwe_entries': len(cwe), 'targets': len(tests),
        'code_model': a.code_model, 'filter': 'not deprecated; not CWE-1194; Applicable_Platforms lists Java, Not Language-Specific, or no platform'}, indent=1))
    print(f'CWE entries kept: {len(cwe)}; targets: {len(tests)}; wrote {out / "e_cwe_code.jsonl"}')


if __name__ == '__main__':
    main()

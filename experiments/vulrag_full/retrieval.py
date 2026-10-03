"""Official repository rank-sum/CVE retrieval, with deterministic CVE tie breaks.
Not paper RRF: see README.md and vendor/vulnerability_detect.py.
"""
from collections import defaultdict
from functools import lru_cache

@lru_cache(maxsize=1)
def tokenizer():
    import spacy
    return spacy.load('en_core_web_sm', disable=['tagger','parser','ner','lemmatizer','attribute_ruler','tok2vec'])

class BM25:
    def __init__(self, corpus):
        from rank_bm25 import BM25Okapi
        self.nlp = tokenizer()
        # Only the tokenizer is used by upstream's retrieval. Longer code is never cut.
        self.nlp.max_length = max(self.nlp.max_length, max(map(len, corpus), default=0)+1)
        self.bm = BM25Okapi([self.tokens(t) for t in corpus])
    def tokens(self, text):
        self.nlp.max_length = max(self.nlp.max_length, len(text)+1)
        return [t.text.lower() for t in self.nlp.make_doc(text) if not t.is_punct]
    def search(self, text, n=-1):
        scores = self.bm.get_scores(self.tokens(text))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return order if n == -1 else order[:n]

class Retriever:
    def __init__(self, knowledge, factory=BM25):
        if not knowledge: raise ValueError('Empty knowledge base')
        self.k = knowledge; self.factory = factory
        self.groups = defaultdict(list)
        self.local_indices = {}
        for k in knowledge: self.groups[k['CVE_id']].append(k)
        self.indices = [factory([k[f] for k in knowledge]) for f in ('purpose','function','code_before_change')]
    def search(self, code, purpose, function, top_k=20, max_knowledge=3):
        qs = (purpose,function,code)
        ranked = []
        for idx,q in zip(self.indices, qs):
            ranked.append(list(dict.fromkeys(self.k[i]['CVE_id'] for i in idx.search(q,top_k))))
        def rank_sum(x, lists): return sum(r.index(x) if x in r else len(r) for r in lists)
        # Official code uses set iteration for ties; lexical tie breaks make this reproducible.
        cves=sorted(set(sum(ranked,[])),key=lambda c:(rank_sum(c,ranked),c))[:max_knowledge]
        selected=[]
        for c in cves:
            items=self.groups[c]
            if len(items)==1: best=items[0]
            else:
                if c not in self.local_indices:
                    self.local_indices[c]=[self.factory([k[f] for k in items]) for f in ('purpose','function','code_before_change')]
                rs=[idx.search(q) for idx,q in zip(self.local_indices[c],qs)]
                best=items[min(range(len(items)),key=lambda i:rank_sum(i,rs))]
            selected.append(best)
        return selected, {'channel_cve_rankings':ranked,'selected_cves':cves,'fusion':'zero-based rank sum; missing rank=list length; lexical CVE tie break'}

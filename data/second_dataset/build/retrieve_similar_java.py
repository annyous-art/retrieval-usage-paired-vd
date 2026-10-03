"""Mode A retrieval for the MegaVul Java split, as for PrimeVul (icl/primevul_test_ccpp_with_similar_top5.jsonl):
SFR-Embedding-Code-400M_R (sentence-transformers, CLS pooling, normalized), cosine similarity between each
test function and every function of the paired training split (vulnerable and fixed); top five kept."""
import json, sys, numpy as np
from sentence_transformers import SentenceTransformer
TRAIN, TEST, OUT = sys.argv[1:4]
train = [json.loads(l) for l in open(TRAIN)]; test = [json.loads(l) for l in open(TEST)]
m = SentenceTransformer('Salesforce/SFR-Embedding-Code-400M_R', trust_remote_code=True, device='mps')
enc = lambda rows: m.encode([r['func'] for r in rows], normalize_embeddings=True, batch_size=8, show_progress_bar=True)
T, Q = enc(train), enc(test)
np.save(OUT + '.train_emb.npy', T); np.save(OUT + '.test_emb.npy', Q)
S = Q @ T.T
with open(OUT, 'w') as f:
    for i, r in enumerate(test):
        top = np.argsort(-S[i], kind='stable')[:5]
        sim = [{'idx': train[j]['idx'], 'func': train[j]['func'], 'target': train[j]['target'],
                'Assistant': 'YES' if str(train[j]['target']) == '1' else 'NO', 'score': float(S[i, j])} for j in top]
        f.write(json.dumps(dict(r, lang='java', similar=sim)) + '\n')
print('wrote', len(test))

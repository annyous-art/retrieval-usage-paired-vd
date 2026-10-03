#!/usr/bin/env bash
# RQ1 on the MegaVul Java pairs for modes B and E (code query): retrieval only, no LLM calls.
# Run from the unpacked package root:  DEVICE=cuda bash experiments/java_rq1_server/run_java_rq1.sh
# Optional check that the same commands reproduce PrimeVul's B rate (47.8%):  VALIDATE=1 DEVICE=cuda bash ...
set -euo pipefail
cd "$(dirname "$0")/../.."
DEVICE=${DEVICE:-cuda}
# Offline servers: point to local model folders, e.g.
#   SLICERAG_EMBED_MODEL=/data/graphcodebert_model CODET5P_MODEL=/data/codet5p-110m-embedding
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
CODET5P_MODEL=${CODET5P_MODEL:-Salesforce/codet5p-110m-embedding}
JTRAIN=data/second_dataset/megavul_java_train_paired_labeled.jsonl
JTEST=data/second_dataset/megavul_java_test_paired_labeled.jsonl
mkdir -p out

if [ "${VALIDATE:-0}" = "1" ]; then
  echo "== Validation: rebuild PrimeVul B and compare with the paper (expected same rate 47.8%)"
  python3 build_chunk_index.py --input data/primevul_train_paired_labeled.jsonl --output-dir out/primevul_chunk_index \
    --model-type graphcodebert --use-ip --device "$DEVICE"
  python3 build_query_chunks_sim_rank.py --input data/primevul_test_paired_labeled.jsonl \
    --index out/primevul_chunk_index/chunk_index.faiss --index-metadata out/primevul_chunk_index/chunk_metadata.jsonl \
    --output-dir out/primevul_query --device "$DEVICE"
  python3 experiments/java_rq1_server/rq1_same_items.py --pairs data/primevul_test_paired_labeled.jsonl \
    --b-results out/primevul_query/query_results.jsonl --out out/primevul_rq1_check.json
fi

echo "== B: chunk index of the Java retrieval corpus"
python3 build_chunk_index.py --input "$JTRAIN" --output-dir out/java_chunk_index --model-type graphcodebert --use-ip --device "$DEVICE"
echo "== B: query chunks of the Java test functions"
python3 build_query_chunks_sim_rank.py --input "$JTEST" --index out/java_chunk_index/chunk_index.faiss \
  --index-metadata out/java_chunk_index/chunk_metadata.jsonl --output-dir out/java_query --device "$DEVICE"
echo "== E: CWE entries retrieved with the code query"
python3 experiments/java_rq1_server/e_cwe_code_java.py --cwe-xml data/e_retrieval/cwec_v4.20.xml --test "$JTEST" --out out/java_e --code-model "$CODET5P_MODEL"
echo "== RQ1: same retrieved items for the two versions"
python3 experiments/java_rq1_server/rq1_same_items.py --pairs "$JTEST" --b-results out/java_query/query_results.jsonl \
  --e-results out/java_e/e_cwe_code.jsonl --out out/java_rq1_summary.json
echo "Done. Send back: out/java_rq1_summary.json, out/java_e/, out/java_query/query_results.jsonl (and out/primevul_rq1_check.json if validated)"

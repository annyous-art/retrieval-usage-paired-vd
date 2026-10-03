#!/usr/bin/env bash
# Mode B on the MegaVul Java pairs with PrimeVul-style line labels (diff -u, mark_pair_diffs.compute_labels).
# Retrieval only, no LLM calls.  Run from the unpacked package root:
#   SLICERAG_EMBED_MODEL=/data/graphcodebert_model DEVICE=cuda bash experiments/java_b_relabel/run_java_b_relabel.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
DEVICE=${DEVICE:-cuda}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
JTRAIN=data/second_dataset/megavul_java_train_paired_labeled_diff.jsonl
JTEST=data/second_dataset/megavul_java_test_paired_labeled_diff.jsonl
mkdir -p out
echo "== B: chunk index of the Java retrieval corpus (diff labels)"
python3 build_chunk_index.py --input "$JTRAIN" --output-dir out/java_chunk_index_diff --model-type graphcodebert --use-ip --device "$DEVICE"
echo "== B: query chunks of the Java test functions (diff labels)"
python3 build_query_chunks_sim_rank.py --input "$JTEST" --index out/java_chunk_index_diff/chunk_index.faiss \
  --index-metadata out/java_chunk_index_diff/chunk_metadata.jsonl --output-dir out/java_query_diff --device "$DEVICE"
echo "== RQ1: same retrieved chunks for the two versions"
python3 experiments/java_rq1_server/rq1_same_items.py --pairs "$JTEST" --b-results out/java_query_diff/query_results.jsonl \
  --out out/java_rq1_summary_diff.json
echo "Done. Send back: out/java_query_diff/query_results.jsonl and out/java_rq1_summary_diff.json"

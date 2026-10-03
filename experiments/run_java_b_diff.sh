#!/usr/bin/env bash
# Mode B on the Java pairs rebuilt with PrimeVul-style line labels (2026-10-03): prompts from
# out/java_query_diff_new (server run of experiments/java_b_relabel), sent through the GLM endpoint as the Java
# two-shot reference. Resumable; up to 8 passes.  Usage: bash experiments/run_java_b_diff.sh MODEL [WORKERS]
set -uo pipefail
cd "$(dirname "$0")/.."
export SLICERAG_OPENAI_BASE_URL="${SLICERAG_UNIAPI_BASE_URL:?set the OpenAI-compatible base URL used for GLM-5.1}"
export SLICERAG_API_KEY="${SLICERAG_UNIAPI_KEY:?set the API key of that endpoint}"
M=$1; W=${2:-8}; f=outdir/java_std/${M}_B_chunks_diff.jsonl
for pass in 1 2 3 4 5 6 7 8; do
  n=$([ -f "$f" ] && wc -l < "$f" | tr -d ' ' || echo 0)
  [ "$n" -ge 1240 ] && break
  echo "[B-diff $M] pass $pass, $n/1240 done"
  python3 replay_saved_prompts.py --model "$M" --source outdir/java_std/prompts_B_chunks_diff.jsonl --out "$f" --workers "$W"
  sleep 20
done
echo "[B-diff $M] final $([ -f "$f" ] && wc -l < "$f" | tr -d ' ' || echo 0)/1240"

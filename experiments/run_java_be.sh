#!/usr/bin/env bash
# Modes B and E (code query) on the MegaVul Java pairs (2026-10-02). Each run is resumable and repeated
# until all 1,240 rows are answered (at most 8 passes). Gateways follow the reference prompts:
#   B (vs. Java two-shot, sent through the GLM endpoint): the GLM endpoint for both models
#   E (vs. Java no-retrieval of java_fc_*): GPT-5.5 through the GPT endpoint, GLM-5.1 through the GLM endpoint (e_arm.py checks this)
# Usage: bash experiments/run_java_be.sh {B|E} MODEL [ARM]   (ARM for E: cwe_code (default) | cwe_desc | cve_desc)
set -uo pipefail
cd "$(dirname "$0")/.."
mode=$1; M=$2; ARM=${3:-cwe_code}
for pass in 1 2 3 4 5 6 7 8; do
  if [[ $mode == B ]]; then
    f=outdir/java_std/${M}_B_chunks.jsonl
  else
    f=outdir/java_e_arm_${M}_${ARM}/predictions.jsonl
  fi
  n=$([ -f "$f" ] && wc -l < "$f" | tr -d ' ' || echo 0)
  [ "$n" -ge 1240 ] && break
  echo "[$mode $M $ARM] pass $pass, $n/1240 done"
  if [[ $mode == B ]]; then
    ( export SLICERAG_OPENAI_BASE_URL="${SLICERAG_UNIAPI_BASE_URL:?set the OpenAI-compatible base URL used for GLM-5.1}"
      export SLICERAG_API_KEY="${SLICERAG_UNIAPI_KEY:?set the API key of that endpoint}"
      python3 replay_saved_prompts.py --model "$M" --source outdir/java_std/prompts_B_chunks.jsonl --out "$f" --workers 4 )
  else
    ( unset SLICERAG_API_KEY SLICERAG_BASE_URL OPENAI_API_KEY
      w=4; [[ $M == gpt-5.5 ]] && w=8
      python3 -u experiments/knowledge_rag/next_stage/e_arm.py --source outdir/java_fc_$M --predictions outdir/java_fc_$M/predictions.jsonl \
        --retrieval-dir data/e_retrieval_java --out outdir/java_e_arm_${M}_${ARM} --arm $ARM --stage run --workers $w --max-calls 1240 )
  fi
  sleep 30
done
echo "[$mode $M $ARM] final $([ -f "$f" ] && wc -l < "$f" | tr -d ' ' || echo 0)/1240"

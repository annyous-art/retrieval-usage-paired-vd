#!/usr/bin/env bash
# RQ3 presentation arms on GPT-5.5 through the GPT endpoint (2026-10-02), 2 workers per arm; the arms run as
# separate processes in parallel. Modes C, D, E reversed (next_stage/order_arm.py; the source runs
# already used the GPT endpoint) and mode A with the demonstrations swapped, plus the original A prompts
# replayed through the GPT endpoint (the original GPT A run used the GLM endpoint). The A prompts are the GLM A prompts,
# which the GPT A run also used. Resumable; up to 3 passes per arm.
# Usage: bash experiments/run_order_gpt.sh c_fixed_first|d_fields_rev|e_cwe_desc_rev|a_demos_swapped|a_original [max_calls]
set -uo pipefail
cd "$(dirname "$0")/.."
O=outdir/order_gpt-5.5
ARM=${1:?arm}
MAX=${2:-100000}
GPT_BASE="${SLICERAG_NEWAPI_BASE_URL:?set the OpenAI-compatible base URL used for GPT-5.5}"
A2=outdir/icl_fixed_baseline_compatible_test870/glm-5.1_std_cls_fewshotegTrue_top2_baseline_compatible.jsonl

slot() {  # C, D, E through the knowledge_rag client (key from experiments/knowledge_rag/gateway_config_local.py, not included)
  cd experiments/knowledge_rag
  unset SLICERAG_API_KEY; export SLICERAG_BASE_URL=$GPT_BASE
  for pass in 1 2 3; do
    python3 next_stage/order_arm.py --arm "$ARM" --source "$1" --out ../../$O --stage run --workers 2 --max-calls $MAX
  done
}

replay() {  # A through model_api_clients.py, the client of the original A runs
  export SLICERAG_OPENAI_BASE_URL=$GPT_BASE
  export SLICERAG_API_KEY="${SLICERAG_NEWAPI_KEY:?set the API key of that endpoint}"
  mkdir -p "$(dirname "$2")"
  for pass in 1 2 3; do
    python3 replay_saved_prompts.py --model gpt-5.5 --source "$1" --out "$2" --workers 2 --max-calls $MAX
  done
}

case "$ARM" in
  c_fixed_first|d_fields_rev) slot ../../outdir/knowledge_rag_gpt-5.5_full_the GPT endpoint_v7 ;;
  e_cwe_desc_rev) slot ../../outdir/e_arm_gpt-5.5_cwe_desc ;;
  a_demos_swapped) replay outdir/order_glm-5.1/a_demos_swapped/prompts_a_demos_swapped.jsonl \
                          $O/a_demos_swapped/gpt-5.5_std_cls_fewshotegTrue_top2_demos_swapped.jsonl ;;
  a_original) replay $A2 $O/a_original_the GPT endpoint/gpt-5.5_std_cls_fewshotegTrue_top2_original_the GPT endpoint.jsonl ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
echo "run_order_gpt $ARM finished"

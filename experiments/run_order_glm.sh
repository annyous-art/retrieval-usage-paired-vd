#!/usr/bin/env bash
# RQ3 presentation arms on GLM-5.1 through the GLM endpoint (2026-10-02): the same retrieved items in reversed
# order for modes A, C, D, E (next_stage/order_arm.py), plus the original A prompts replayed through
# the GLM endpoint so that both A arms share a gateway. Resumable; each arm up to 3 passes.
# Usage: bash experiments/run_order_glm.sh cde | a
set -uo pipefail
cd "$(dirname "$0")/.."
O=outdir/order_glm-5.1
case "${1:-}" in
  cde)
    cd experiments/knowledge_rag
    for spec in "c_fixed_first ../../outdir/knowledge_rag_glm-5.1" \
                "d_fields_rev ../../outdir/knowledge_rag_glm-5.1" \
                "e_cwe_desc_rev ../../outdir/e_arm_glm-5.1_cwe_desc"; do
      set -- $spec
      for pass in 1 2 3; do
        python3 next_stage/order_arm.py --arm "$1" --source "$2" --out ../../$O --stage run --workers 4
      done
    done ;;
  a)
    export SLICERAG_OPENAI_BASE_URL="${SLICERAG_UNIAPI_BASE_URL:?set the OpenAI-compatible base URL used for GLM-5.1}"
    export SLICERAG_API_KEY="${SLICERAG_UNIAPI_KEY:?set the API key of that endpoint}"
    A2=outdir/icl_fixed_baseline_compatible_test870/glm-5.1_std_cls_fewshotegTrue_top2_baseline_compatible.jsonl
    for pass in 1 2 3; do
      python3 replay_saved_prompts.py --model glm-5.1 --source $O/a_demos_swapped/prompts_a_demos_swapped.jsonl \
        --out $O/a_demos_swapped/glm-5.1_std_cls_fewshotegTrue_top2_demos_swapped.jsonl --workers 4
    done
    mkdir -p $O/a_original
    for pass in 1 2 3; do
      python3 replay_saved_prompts.py --model glm-5.1 --source $A2 \
        --out $O/a_original/glm-5.1_std_cls_fewshotegTrue_top2_original.jsonl --workers 4
    done ;;
  *) echo "usage: $0 cde|a"; exit 2 ;;
esac
echo "run_order_glm $1 finished"

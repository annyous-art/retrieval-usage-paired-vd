#!/usr/bin/env bash
# Mode A with k = 1 and k = 4 retrieved demonstrations on GLM-5.1 (2026-09-28). The prompts are
# built by icl/run_prompting_k_icl.py's construct_prompts from the same top-5 neighbour file as the
# k = 2 run (its k = 2 rebuild equals the saved k = 2 prompts for all 870 functions), then sent with
# replay_saved_prompts.py (std_cls, temperature 0) through the GLM endpoint. Resumable; each file up to 3 passes.
# Usage: bash experiments/run_a_k_sensitivity.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export SLICERAG_OPENAI_BASE_URL="${SLICERAG_UNIAPI_BASE_URL:?set the OpenAI-compatible base URL used for GLM-5.1}"
export SLICERAG_API_KEY="${SLICERAG_UNIAPI_KEY:?set the API key of that endpoint}"
O=outdir/a_k_sensitivity
for k in 1 4; do
  for pass in 1 2 3; do
    python3 replay_saved_prompts.py --model glm-5.1 --source $O/prompts_top$k.jsonl \
      --out $O/glm-5.1_std_cls_fewshotegTrue_top${k}_baseline_compatible.jsonl --workers 4
  done
done
echo "a_k_sensitivity finished"

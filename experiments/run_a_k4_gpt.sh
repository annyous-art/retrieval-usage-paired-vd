#!/usr/bin/env bash
# Mode A with four retrieved demonstrations on GPT-5.5 (2026-10-02), through the GLM endpoint like the GPT-5.5
# zero-shot and two-demonstration runs it is compared with. Same prompts as the GLM-5.1 k = 4 run
# (outdir/a_k_sensitivity/prompts_top4.jsonl), sent with replay_saved_prompts.py (std_cls,
# temperature 0). Resumable; up to 3 passes.
set -uo pipefail
cd "$(dirname "$0")/.."
export SLICERAG_OPENAI_BASE_URL="${SLICERAG_UNIAPI_BASE_URL:?set the OpenAI-compatible base URL used for GLM-5.1}"
export SLICERAG_API_KEY="${SLICERAG_UNIAPI_KEY:?set the API key of that endpoint}"
O=outdir/a_k_sensitivity
for pass in 1 2 3; do
  python3 replay_saved_prompts.py --model gpt-5.5 --source $O/prompts_top4.jsonl \
    --out $O/gpt-5.5_std_cls_fewshotegTrue_top4_baseline_compatible.jsonl --workers ${WORKERS:-6}
done
echo "run_a_k4_gpt finished"

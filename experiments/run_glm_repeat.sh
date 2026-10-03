#!/usr/bin/env bash
# Second run of the one significant comparison (GLM-5.1, mode A with two demonstrations vs. the PrimeVul
# two-shot prompt), 2026-09-28: the saved prompts of both runs are sent again unchanged with
# replay_saved_prompts.py (std_cls, temperature 0) through the GLM endpoint. Resumable; up to 3 passes per file.
# Usage: bash experiments/run_glm_repeat.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export SLICERAG_OPENAI_BASE_URL="${SLICERAG_UNIAPI_BASE_URL:?set the OpenAI-compatible base URL used for GLM-5.1}"
export SLICERAG_API_KEY="${SLICERAG_UNIAPI_KEY:?set the API key of that endpoint}"
O=outdir/glm_repeat_run2
for pair in "outdir/glm-5.1_std_cls_logprobsFalse_fewshotegTrue_baseline_870.jsonl two_shot" \
            "outdir/icl_fixed_baseline_compatible_test870/glm-5.1_std_cls_fewshotegTrue_top2_baseline_compatible.jsonl A_top2"; do
  set -- $pair
  for pass in 1 2 3; do
    python3 replay_saved_prompts.py --model glm-5.1 --source $1 --out $O/glm-5.1_$2.jsonl --workers 4
  done
done
echo "glm repeat finished"

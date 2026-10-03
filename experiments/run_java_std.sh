#!/usr/bin/env bash
# std setting on the MegaVul Java split (2026-09-28): PrimeVul two-shot and mode A (two retrieved Java
# demonstrations), prompts from data/second_dataset/build/build_java_std_prompts.py, sent with
# replay_saved_prompts.py (std_cls, temperature 0) through the GLM endpoint. Resumable; up to 3 passes per file.
# Usage: bash experiments/run_java_std.sh glm-5.1
set -uo pipefail
cd "$(dirname "$0")/.."
export SLICERAG_OPENAI_BASE_URL="${SLICERAG_UNIAPI_BASE_URL:?set the OpenAI-compatible base URL used for GLM-5.1}"
export SLICERAG_API_KEY="${SLICERAG_UNIAPI_KEY:?set the API key of that endpoint}"
M=$1; O=outdir/java_std
for name in two_shot A_top2; do
  for pass in 1 2 3; do
    if [[ $M == deepseek* ]]; then
      python3 experiments/knowledge_rag_deepseek/replay_std_cls.py --source $O/prompts_$name.jsonl --out $O/${M}_$name.jsonl --workers 4
    else
      python3 replay_saved_prompts.py --model $M --source $O/prompts_$name.jsonl --out $O/${M}_$name.jsonl --workers 4
    fi
  done
done
echo "java std $M finished"

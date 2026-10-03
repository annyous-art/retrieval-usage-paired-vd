#!/usr/bin/env bash
# Usage: run_e.sh MODEL ARM [prepare|run|score] [WORKERS]
#   MODEL: gpt-5.5 | glm-5.1 | claude-opus-4-7 | deepseek-v4-pro-0813
#   ARM:   cwe_code (X2) | cwe_desc (X3) | cve_desc (X4) | cwe_desc_short (X6)
# Set SLICERAG_MAX_CALLS=4 for a first check; rerun without it to continue.
set -euo pipefail
# Each model must use the gateway of its original v7 run (e_arm.py checks this). Drop any
# key or URL exported for replay_saved_prompts.py; GLM/DeepSeek set their own via configure().
unset SLICERAG_API_KEY SLICERAG_BASE_URL OPENAI_API_KEY
cd "$(dirname "$0")/../../.."
model="${1:?MODEL required}"
arm="${2:?ARM required}"
stage="${3:-prepare}"
workers="${4:-2}"
# newapi (GPT-5.5, Claude) allows 30 requests per minute across all jobs. GPT-5.5 calls there take
# about 30 s each, so it gets more workers; together they stay below the limit.
case "$model" in
  gpt-5.5) workers="${SLICERAG_GPT_WORKERS:-8}" ;;
  claude-opus-4-7) workers="${SLICERAG_CLAUDE_WORKERS:-2}" ;;
esac
script=experiments/knowledge_rag/next_stage/e_arm.py
case "$model" in
  glm-5.1) source="outdir/knowledge_rag_${model}_full_uniapi_v7" ;;
  gpt-5.5|claude-opus-4-7) source="outdir/knowledge_rag_${model}_full_newapi_v7" ;;
  deepseek-v4-pro-0813)
    source="outdir/knowledge_rag_${model}_full_uniapi_v7"
    script=experiments/knowledge_rag_deepseek/next_stage/e_arm.py ;;
  *) echo 'Unsupported model'; exit 2 ;;
esac
pred="$source/predictions.jsonl"
if [[ "$model" == claude-opus-4-7 ]]; then
  recovered=outdir/next_stage_claude_offline_recovered
  if [[ ! -f "$recovered/metrics_recovered.json" ]]; then
    python3 experiments/knowledge_rag/recover_detection_offline.py --directory "$source" --out "$recovered"
  fi
  pred="$recovered/predictions_recovered.jsonl"
fi
extra=()
if [[ -n "${SLICERAG_MAX_CALLS:-}" ]]; then extra+=(--max-calls "$SLICERAG_MAX_CALLS"); fi
python3 -u "$script" --source "$source" --predictions "$pred" --retrieval-dir data/e_retrieval \
  --out "outdir/e_arm_${model}_${arm}" --arm "$arm" --stage "$stage" --workers "$workers" ${extra[@]+"${extra[@]}"}

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
model="${1:?MODEL required}"
task="${2:?offline|slice_fixed|shuffled_knowledge required}"
stage="${3:-prepare}"
workers="${4:-2}"
case "$model" in
  glm-5.1) source="outdir/knowledge_rag_${model}_full_uniapi_v7" ;;
  gpt-5.5|claude-opus-4-7) source="outdir/knowledge_rag_${model}_full_newapi_v7" ;;
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
if [[ "$task" == offline ]]; then
  python3 experiments/knowledge_rag/next_stage/offline_review.py --source "$source" --predictions "$pred" --out "outdir/next_stage_${model}_offline" --sample 50
else
  extra=()
  if [[ "$task" == slice_fixed ]]; then
    extra=(--chunk-metadata "${SLICERAG_CHUNK_METADATA:-index_balanced_chunk6_stride2/chunk_metadata.jsonl}")
  fi
  if [[ -n "${SLICERAG_MAX_CALLS:-}" ]]; then extra+=(--max-calls "$SLICERAG_MAX_CALLS"); fi
  python3 -u experiments/knowledge_rag/next_stage/extra_arm.py --source "$source" --predictions "$pred" --out "outdir/next_stage_${model}_${task}" --arm "$task" --stage "$stage" --workers "$workers" "${extra[@]}"
fi

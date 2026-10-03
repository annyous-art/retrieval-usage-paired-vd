#!/usr/bin/env bash
# Run E arms for one model, repeating each arm until all 870 rows are answered (at most 8 passes).
# A pass re-calls only rows without a saved answer, so repeats cost nothing for finished rows.
# Usage: bash experiments/run_e_until_complete.sh MODEL [ARM ...]
set -uo pipefail
cd "$(dirname "$0")/.."
model="$1"; shift
arms=("$@"); [ ${#arms[@]} -eq 0 ] && arms=(cwe_code cwe_desc cve_desc cwe_desc_short)
for a in "${arms[@]}"; do
  f="outdir/e_arm_${model}_${a}/predictions.jsonl"
  for pass in 1 2 3 4 5 6 7 8; do
    n=$([ -f "$f" ] && wc -l < "$f" | tr -d ' ' || echo 0)
    [ "$n" -ge 870 ] && break
    echo "[$model $a] pass $pass, $n/870 done"
    bash experiments/knowledge_rag/next_stage/run_e.sh "$model" "$a" run 4
    sleep 30   # let the per-minute window reset before retrying rejected rows
  done
  echo "[$model $a] final $([ -f "$f" ] && wc -l < "$f" | tr -d ' ' || echo 0)/870"
done
echo "chain ${model}_e finished"
